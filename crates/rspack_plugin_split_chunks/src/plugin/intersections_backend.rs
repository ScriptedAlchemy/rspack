//! Experimental round filtering. CPU publication, exact deduplication, and support stay unchanged.

use std::{
  fs::OpenOptions,
  io::Write,
  path::PathBuf,
  sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
  },
  time::Instant,
};

use hashbrown::HashTable;
use rspack_cuda::{FilterStatistics, RoundFilter};
use rspack_error::{Error, Result};
use serde::Serialize;

use super::{
  bitmap::ChunkBitmap,
  intersections::{
    CandidateGenerator, KnownBitmaps, bitmap_hash, discover_intersections_cpu, insert_candidate,
  },
};

// This is an experimental scheduling heuristic, not a measured crossover point.
// Forced CUDA bypasses it so benchmarks can establish the actual crossover.
const AUTO_MIN_WORD_OPS: u64 = 1_048_576;
static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);
static TELEMETRY_WRITE: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
  Cpu,
  Cuda,
  Auto,
}

impl Backend {
  fn name(self) -> &'static str {
    match self {
      Self::Cpu => "cpu",
      Self::Cuda => "cuda",
      Self::Auto => "auto",
    }
  }
}

pub(super) struct DiscoveryInput<'a> {
  pub(super) bitmaps: &'a [ChunkBitmap],
  pub(super) lengths: &'a [usize],
  pub(super) first: usize,
  pub(super) known: &'a KnownBitmaps<'a>,
  pub(super) excluded: Option<&'a ChunkBitmap>,
  pub(super) min_chunks: usize,
  pub(super) chunk_count: usize,
}

impl DiscoveryInput<'_> {
  fn cpu(&self) -> Vec<CandidateGenerator> {
    discover_intersections_cpu(
      self.bitmaps,
      self.lengths,
      self.first,
      self.known,
      self.excluded,
      self.min_chunks,
      self.chunk_count,
    )
  }

  fn pair_count(&self) -> u64 {
    let triangular = |n: usize| (n as u128) * (n.saturating_sub(1) as u128) / 2;
    u64::try_from(triangular(self.bitmaps.len()) - triangular(self.first)).unwrap_or(u64::MAX)
  }
}

pub(super) struct IntersectionExecution {
  backend: Backend,
  telemetry: Option<PathBuf>,
  verify: bool,
  call_id: u64,
}

impl IntersectionExecution {
  pub(super) fn from_env() -> Result<Self> {
    let backend = match read_env("RSPACK_SPLIT_CHUNKS_BACKEND")?.as_deref() {
      None | Some("cpu") => Backend::Cpu,
      Some("cuda") => Backend::Cuda,
      Some("auto") => Backend::Auto,
      Some(value) => {
        return Err(Error::error(format!(
          "RSPACK_SPLIT_CHUNKS_BACKEND must be cpu, cuda, or auto; received {value:?}"
        )));
      }
    };
    let verify = match read_env("RSPACK_SPLIT_CHUNKS_VERIFY")?.as_deref() {
      None | Some("0") => false,
      Some("1") => true,
      Some(value) => {
        return Err(Error::error(format!(
          "RSPACK_SPLIT_CHUNKS_VERIFY must be 0 or 1; received {value:?}"
        )));
      }
    };
    let telemetry = std::env::var_os("RSPACK_SPLIT_CHUNKS_TELEMETRY").map(PathBuf::from);
    if telemetry.as_ref().is_some_and(|path| !path.is_absolute()) {
      return Err(Error::error(
        "RSPACK_SPLIT_CHUNKS_TELEMETRY must be an absolute JSONL file path".into(),
      ));
    }
    Ok(Self {
      backend,
      call_id: if telemetry.is_some() {
        NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed)
      } else {
        0
      },
      telemetry,
      verify,
    })
  }

  pub(super) fn discover(
    &self,
    input: DiscoveryInput<'_>,
    round: u32,
    original_sets: usize,
    dedup_depth: u32,
  ) -> Result<Vec<CandidateGenerator>> {
    // Ordinary builds retain the original CPU implementation and perform no CUDA calls,
    // packing, timing, or telemetry serialization.
    if self.backend == Backend::Cpu && self.telemetry.is_none() {
      return Ok(input.cpu());
    }
    let started = Instant::now();
    let word_count = input.chunk_count.div_ceil(64);
    let pair_count = input.pair_count();
    let estimated_word_ops = pair_count.saturating_mul(word_count as u64);
    let mut metrics = CudaMeasurements::default();
    let mut fallback_reason = None;
    let mut executed = self.backend;

    if self.backend == Backend::Auto {
      if estimated_word_ops < AUTO_MIN_WORD_OPS {
        fallback_reason = Some("below-auto-word-ops-threshold".to_owned());
        executed = Backend::Cpu;
      } else if !rspack_cuda::is_compiled() {
        fallback_reason = Some("cuda-feature-not-enabled".to_owned());
        executed = Backend::Cpu;
      } else {
        executed = Backend::Cuda;
      }
    }

    let candidates = if executed == Backend::Cuda {
      match discover_cuda(&input, word_count, &mut metrics) {
        Ok(candidates) => candidates,
        Err(error) if self.backend == Backend::Auto => {
          fallback_reason = Some(format!("cuda-error: {error}"));
          executed = Backend::Cpu;
          let cpu_started = Instant::now();
          let candidates = input.cpu();
          metrics.backend_ms += elapsed_ms(cpu_started);
          candidates
        }
        Err(error) => return Err(error),
      }
    } else {
      let cpu_started = Instant::now();
      let candidates = input.cpu();
      metrics.backend_ms = elapsed_ms(cpu_started);
      candidates
    };
    let wall_ms = elapsed_ms(started);

    // Shadow execution is a correctness gate, excluded from the reported accelerator wall time.
    let mut verification_ms = 0.0;
    let verified = self.verify && executed == Backend::Cuda;
    if verified {
      let verification_started = Instant::now();
      let expected = input.cpu();
      verification_ms = elapsed_ms(verification_started);
      if candidates.len() != expected.len()
        || candidates
          .iter()
          .zip(&expected)
          .any(|(actual, expected)| actual.order != expected.order || actual.hash != expected.hash)
      {
        return Err(Error::error(format!(
          "SplitChunks CUDA verification failed in round {round}: CPU and CUDA candidate sets or discovery order differ (CPU {}, CUDA {})",
          expected.len(),
          candidates.len()
        )));
      }
    }

    if let Some(path) = &self.telemetry {
      let stats = metrics.statistics.as_ref();
      let record = RoundTelemetry {
        schema_version: 1,
        event: "split_chunks_intersection_round",
        pid: std::process::id(),
        call_id: self.call_id,
        round,
        requested_backend: self.backend.name(),
        backend: executed.name(),
        fallback_reason: fallback_reason.as_deref(),
        input_sets: input.bitmaps.len(),
        original_sets,
        chunk_count: input.chunk_count,
        word_count,
        first_set: input.first,
        min_chunks: input.min_chunks,
        dedup_depth,
        excluded: input.excluded.is_some(),
        pair_count,
        estimated_word_ops,
        auto_min_word_ops: AUTO_MIN_WORD_OPS,
        candidates: candidates.len(),
        kernel_launches: stats.map_or(0, |s| s.launches),
        examined_pairs: stats.map_or(0, |s| s.examined_pairs),
        accepted_pairs: stats.map_or(0, |s| s.accepted_pairs),
        packing_ms: metrics.packing_ms,
        backend_ms: metrics.backend_ms,
        cpu_dedup_ms: metrics.cpu_dedup_ms,
        wall_ms,
        allocation_ms: stats.map_or(0.0, |s| s.allocation_ms),
        kernel_ms: stats.map_or(0.0, |s| s.kernel_ms),
        upload_ms: stats.map_or(0.0, |s| s.upload_ms),
        download_ms: stats.map_or(0.0, |s| s.download_ms),
        upload_bytes: stats.map_or(0, |s| s.upload_bytes),
        download_bytes: stats.map_or(0, |s| s.download_bytes),
        device_bytes: stats.map_or(0, |s| s.device_bytes),
        verified,
        verification_ms,
      };
      let mut line = serde_json::to_vec(&record).map_err(|error| {
        Error::error(format!("Cannot serialize SplitChunks telemetry: {error}"))
      })?;
      line.push(b'\n');
      // MultiCompiler builds can publish simultaneously. One lock covers open and the whole line.
      let _guard = TELEMETRY_WRITE
        .lock()
        .map_err(|_| Error::error("SplitChunks telemetry writer is poisoned".into()))?;
      OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(&line))
        .map_err(|error| {
          Error::error(format!(
            "Cannot append SplitChunks telemetry to {}: {error}",
            path.display()
          ))
        })?;
    }
    Ok(candidates)
  }
}

#[derive(Default)]
struct CudaMeasurements {
  statistics: Option<FilterStatistics>,
  packing_ms: f64,
  backend_ms: f64,
  cpu_dedup_ms: f64,
}

fn discover_cuda(
  input: &DiscoveryInput<'_>,
  word_count: usize,
  metrics: &mut CudaMeasurements,
) -> Result<Vec<CandidateGenerator>> {
  let packing_started = Instant::now();
  let packed = pack_bitmaps(input, word_count);
  metrics.packing_ms = elapsed_ms(packing_started);
  let (words, lengths) = packed?;
  let backend_started = Instant::now();
  let filter = RoundFilter::new(
    &words,
    &lengths,
    word_count,
    input.first,
    input.min_chunks,
    input.excluded.map(ChunkBitmap::words),
  );
  metrics.backend_ms = elapsed_ms(backend_started);
  let mut filter = filter.map_err(cuda_error)?;
  let mut candidates = HashTable::new();
  let mut scratch = ChunkBitmap::new(input.chunk_count);
  loop {
    let backend_started = Instant::now();
    let batch = filter.next_pairs();
    metrics.backend_ms += elapsed_ms(backend_started);
    let pairs = match batch {
      Ok(Some(pairs)) => pairs,
      Ok(None) => break,
      Err(error) => {
        metrics.statistics = Some(filter.statistics());
        return Err(cuda_error(error));
      }
    };
    let dedup_started = Instant::now();
    for pair in pairs {
      let i = pair.left as usize;
      let j = pair.right as usize;
      if i < input.first || i >= input.bitmaps.len() || j >= i {
        return Err(Error::error(
          "SplitChunks CUDA returned an invalid candidate pair".into(),
        ));
      }
      scratch.assign_intersection(&input.bitmaps[i], &input.bitmaps[j]);
      insert_candidate(
        &mut candidates,
        &scratch,
        CandidateGenerator {
          hash: bitmap_hash(&scratch),
          order: (i, j),
        },
        input.bitmaps,
        Some(input.known),
      );
    }
    metrics.cpu_dedup_ms += elapsed_ms(dedup_started);
    metrics.statistics = Some(filter.statistics());
  }
  metrics.statistics = Some(filter.statistics());
  let dedup_started = Instant::now();
  let mut candidates = candidates.into_iter().collect::<Vec<_>>();
  candidates.sort_unstable_by_key(|entry| entry.order);
  metrics.cpu_dedup_ms += elapsed_ms(dedup_started);
  Ok(candidates)
}

fn pack_bitmaps(input: &DiscoveryInput<'_>, word_count: usize) -> Result<(Vec<u64>, Vec<u32>)> {
  let capacity = input.bitmaps.len().checked_mul(word_count).ok_or_else(|| {
    Error::error("SplitChunks CUDA bitmap dimensions exceed the host address space".into())
  })?;
  let mut words = Vec::new();
  words.try_reserve_exact(capacity).map_err(|error| {
    Error::error(format!(
      "Cannot allocate SplitChunks CUDA bitmap input: {error}"
    ))
  })?;
  for bitmap in input.bitmaps {
    words.extend_from_slice(bitmap.words());
  }
  let lengths = input
    .lengths
    .iter()
    .map(|&length| {
      u32::try_from(length)
        .map_err(|_| Error::error("SplitChunks CUDA set cardinality exceeds u32".into()))
    })
    .collect::<Result<Vec<_>>>()?;
  Ok((words, lengths))
}

fn read_env(name: &str) -> Result<Option<String>> {
  match std::env::var(name) {
    Ok(value) => Ok(Some(value)),
    Err(std::env::VarError::NotPresent) => Ok(None),
    Err(error) => Err(Error::error(format!("Cannot read {name}: {error}"))),
  }
}

fn cuda_error(error: rspack_cuda::Error) -> Error {
  Error::error(format!("SplitChunks CUDA backend failed: {error}"))
}

fn elapsed_ms(started: Instant) -> f64 {
  started.elapsed().as_secs_f64() * 1_000.0
}

#[derive(Serialize)]
struct RoundTelemetry<'a> {
  schema_version: u32,
  event: &'static str,
  pid: u32,
  call_id: u64,
  round: u32,
  requested_backend: &'static str,
  backend: &'static str,
  fallback_reason: Option<&'a str>,
  input_sets: usize,
  original_sets: usize,
  chunk_count: usize,
  word_count: usize,
  first_set: usize,
  min_chunks: usize,
  dedup_depth: u32,
  excluded: bool,
  pair_count: u64,
  estimated_word_ops: u64,
  auto_min_word_ops: u64,
  candidates: usize,
  kernel_launches: u64,
  examined_pairs: u64,
  accepted_pairs: u64,
  packing_ms: f64,
  backend_ms: f64,
  cpu_dedup_ms: f64,
  wall_ms: f64,
  allocation_ms: f64,
  kernel_ms: f64,
  upload_ms: f64,
  download_ms: f64,
  upload_bytes: u64,
  download_bytes: u64,
  device_bytes: u64,
  verified: bool,
  verification_ms: f64,
}

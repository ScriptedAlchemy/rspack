//! Optional, bounded CUDA pair filtering for SplitChunks experiments.
//!
//! This crate has no CUDA build or runtime dependency unless its `cuda` feature
//! is enabled on a supported target. Backend selection and fallback belong to
//! the caller. Exact bitmap deduplication and publication also remain there.

use std::fmt;

#[cfg(rspack_cuda_native)]
mod native;
#[cfg(rspack_cuda_native)]
pub use native::{RoundFilter, is_available};

#[cfg(not(rspack_cuda_native))]
mod unavailable;
#[cfg(not(rspack_cuda_native))]
pub use unavailable::{RoundFilter, is_available};

/// Maximum number of pairs covered by one batch, before rejection and compaction.
/// Output buffer memory is independent of the complete round's pair count.
pub const MAX_BATCH_PAIRS: usize = 256 * 256;

/// An input pair whose intersection passes the size and exclusion predicates.
///
/// `left` is the original `i` index and `right` is `j`, with `right < left`.
/// Results are not sorted or deduplicated. Copying this record copies two indices.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidatePair {
  /// Index of the first input bitmap.
  pub left: u32,
  /// Index of the second input bitmap.
  pub right: u32,
}

/// Counters for one frozen round. Copying this snapshot copies scalar counters.
///
/// Copy and kernel durations use CUDA events on the round's private stream.
/// They exclude packing, CPU deduplication, and host dispatch overhead; callers
/// should additionally measure elapsed time around the complete round.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FilterStatistics {
  /// Number of kernel launches.
  pub launches: u64,
  /// Number of `j < i` pairs examined, before length filtering.
  pub examined_pairs: u64,
  /// Number of pairs returned, before exact deduplication.
  pub accepted_pairs: u64,
  /// Input bytes transferred to the device.
  pub upload_bytes: u64,
  /// Counts and pair records transferred to the host.
  pub download_bytes: u64,
  /// Bytes allocated by this filter on the device, excluding runtime overhead.
  pub device_bytes: u64,
  /// Device event time for input transfers.
  pub upload_ms: f64,
  /// Device event time for the pair-filter kernels.
  pub kernel_ms: f64,
  /// Device event time for count and result transfers.
  pub download_ms: f64,
  /// Host elapsed time for runtime initialization and resource allocation.
  pub allocation_ms: f64,
}

/// An unavailable backend, invalid input, allocation failure, or CUDA error.
#[derive(Debug)]
pub struct Error {
  message: String,
}

impl Error {
  pub(crate) fn new(message: impl Into<String>) -> Self {
    Self {
      message: message.into(),
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter.write_str(&self.message)
  }
}

impl std::error::Error for Error {}

/// Whether this binary includes the native CUDA backend.
pub const fn is_compiled() -> bool {
  cfg!(rspack_cuda_native)
}

pub(crate) fn validate_input(
  words: &[u64],
  lengths: &[u32],
  word_count: usize,
  first: usize,
  min_chunks: usize,
  excluded: Option<&[u64]>,
) -> Result<(), Error> {
  let expected_words = lengths
    .len()
    .checked_mul(word_count)
    .ok_or_else(|| Error::new("CUDA bitmap matrix dimensions overflow usize"))?;
  if words.len() != expected_words {
    return Err(Error::new(
      "CUDA bitmap matrix length does not match its dimensions",
    ));
  }
  if excluded.is_some_and(|excluded| excluded.len() != word_count) {
    return Err(Error::new("CUDA excluded bitmap has a different width"));
  }
  if first > lengths.len() || min_chunks == 0 {
    return Err(Error::new(
      "CUDA requires first <= row count and min_chunks > 0",
    ));
  }
  let width = word_count
    .checked_mul(64)
    .ok_or_else(|| Error::new("CUDA bitmap bit width overflows usize"))?;
  if lengths
    .iter()
    .any(|&length| u64::from(length) > width as u64)
  {
    return Err(Error::new("CUDA bitmap length exceeds its word capacity"));
  }
  if u32::try_from(lengths.len()).is_err()
    || u32::try_from(word_count).is_err()
    || u32::try_from(first).is_err()
    || u32::try_from(min_chunks).is_err()
  {
    return Err(Error::new("CUDA bitmap dimensions exceed u32"));
  }
  Ok(())
}

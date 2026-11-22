use std::{
  ffi::{CStr, c_char, c_int, c_void},
  ptr::{self, NonNull},
};

use crate::{CandidatePair, Error, FilterStatistics, MAX_BATCH_PAIRS, validate_input};

#[repr(C)]
struct NativeError {
  code: c_int,
  message: [c_char; 256],
}

impl NativeError {
  fn empty() -> Self {
    Self {
      code: 0,
      message: [0; 256],
    }
  }

  fn into_error(self) -> Error {
    // SAFETY: The buffer starts zeroed and the bridge writes with snprintf,
    // which always preserves a terminating NUL within its fixed capacity.
    let message = unsafe { CStr::from_ptr(self.message.as_ptr()) }.to_string_lossy();
    Error::new(format!("{message} (CUDA status {})", self.code))
  }
}

unsafe extern "C" {
  fn rspack_cuda_probe(error: *mut NativeError) -> c_int;
  fn rspack_cuda_round_new(
    words: *const u64,
    lengths: *const u32,
    row_count: u32,
    word_count: u32,
    first: u32,
    min_chunks: u32,
    excluded: *const u64,
    output: *mut *mut c_void,
    error: *mut NativeError,
  ) -> c_int;
  fn rspack_cuda_round_next(
    filter: *mut c_void,
    pairs: *mut CandidatePair,
    capacity: usize,
    count: *mut u32,
    done: *mut c_int,
    error: *mut NativeError,
  ) -> c_int;
  fn rspack_cuda_round_statistics(filter: *const c_void, statistics: *mut FilterStatistics);
  fn rspack_cuda_round_free(filter: *mut c_void);
}

/// Checks that the CUDA runtime can access a device without allocating a round.
pub fn is_available() -> Result<(), Error> {
  let mut error = NativeError::empty();
  // SAFETY: The bridge writes only to the supplied error buffer and never throws.
  if unsafe { rspack_cuda_probe(&mut error) } != 0 {
    return Err(error.into_error());
  }
  Ok(())
}

/// A frozen, uploaded matrix and a bounded output buffer for one discovery round.
///
/// Input words are row-major: `words[row * word_count + word]`. The constructor
/// uploads them before returning, so no input references are retained. Each
/// instance owns a CUDA stream and is deliberately neither Send nor Sync.
pub struct RoundFilter {
  handle: NonNull<c_void>,
  pairs: Vec<CandidatePair>,
  failed: bool,
}

impl RoundFilter {
  /// Uploads one round and prepares bounded pair filtering.
  ///
  /// Enumerates `i >= first, j < i`. Accepted pairs have a proper intersection
  /// with at least `min_chunks` bits. If `excluded` is present, any intersection
  /// containing an excluded bit is rejected. The caller decides which rounds
  /// permit that pruning. Lengths must describe the corresponding input sets.
  ///
  /// Returns an error for invalid dimensions, allocation failure, an unavailable
  /// device, or any CUDA failure. No partially initialized filter is returned.
  pub fn new(
    words: &[u64],
    lengths: &[u32],
    word_count: usize,
    first: usize,
    min_chunks: usize,
    excluded: Option<&[u64]>,
  ) -> Result<Self, Error> {
    validate_input(words, lengths, word_count, first, min_chunks, excluded)?;
    // The shared validator checks every narrowing conversion before FFI.
    let row_count = lengths.len() as u32;
    let word_count = word_count as u32;
    let first = first as u32;
    let min_chunks = min_chunks as u32;

    let mut pairs = Vec::new();
    pairs
      .try_reserve_exact(MAX_BATCH_PAIRS)
      .map_err(|error| Error::new(format!("CUDA result buffer allocation failed: {error}")))?;
    pairs.resize(MAX_BATCH_PAIRS, CandidatePair { left: 0, right: 0 });
    let mut handle = ptr::null_mut();
    let mut error = NativeError::empty();
    // SAFETY: Validated dimensions cover every input dereference. The bridge
    // finishes input copies before returning and owns the returned allocation.
    let status = unsafe {
      rspack_cuda_round_new(
        words.as_ptr(),
        lengths.as_ptr(),
        row_count,
        word_count,
        first,
        min_chunks,
        excluded.map_or(ptr::null(), <[u64]>::as_ptr),
        &mut handle,
        &mut error,
      )
    };
    if status != 0 {
      return Err(error.into_error());
    }
    let handle = NonNull::new(handle)
      .ok_or_else(|| Error::new("CUDA returned a null filter after successful initialization"))?;
    Ok(Self {
      handle,
      pairs,
      failed: false,
    })
  }

  /// Returns the next bounded batch, which may be empty, or `None` at completion.
  ///
  /// The returned slice remains valid until the next mutable access to this
  /// filter. After a CUDA failure, discard the round's partial results; this
  /// filter rejects further work instead of silently skipping the failed tile.
  pub fn next_pairs(&mut self) -> Result<Option<&[CandidatePair]>, Error> {
    if self.failed {
      return Err(Error::new(
        "CUDA round cannot continue after a failed batch",
      ));
    }
    let mut count = 0;
    let mut done = 0;
    let mut error = NativeError::empty();
    // SAFETY: The live handle belongs to this object. The bridge bounds writes
    // by the provided capacity and synchronizes its stream before returning.
    let status = unsafe {
      rspack_cuda_round_next(
        self.handle.as_ptr(),
        self.pairs.as_mut_ptr(),
        self.pairs.len(),
        &mut count,
        &mut done,
        &mut error,
      )
    };
    if status != 0 {
      self.failed = true;
      return Err(error.into_error());
    }
    if count as usize > self.pairs.len() {
      self.failed = true;
      return Err(Error::new(
        "CUDA returned more candidate pairs than the buffer capacity",
      ));
    }
    if done != 0 {
      Ok(None)
    } else {
      Ok(Some(&self.pairs[..count as usize]))
    }
  }

  /// Copies counters for all work completed so far in this round.
  pub fn statistics(&self) -> FilterStatistics {
    let mut statistics = FilterStatistics::default();
    // SAFETY: Both pointers refer to live, correctly aligned objects with the
    // same repr(C) layout. No CUDA work is started by this accessor.
    unsafe { rspack_cuda_round_statistics(self.handle.as_ptr(), &mut statistics) };
    statistics
  }
}

impl Drop for RoundFilter {
  fn drop(&mut self) {
    // SAFETY: This is the unique owner of the handle; freeing it also releases
    // any resources left by an unsuccessful operation. The bridge never throws.
    unsafe { rspack_cuda_round_free(self.handle.as_ptr()) };
  }
}

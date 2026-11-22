use crate::{CandidatePair, Error, FilterStatistics, validate_input};

const UNAVAILABLE: &str = "CUDA SplitChunks support is not compiled into this binding; rebuild with the cuda feature on Linux x86_64 or aarch64";

/// Reports that the optional CUDA backend was not compiled for this target.
pub fn is_available() -> Result<(), Error> {
  Err(Error::new(UNAVAILABLE))
}

/// An unavailable backend with the same interface as a CUDA-enabled build.
pub struct RoundFilter;

impl RoundFilter {
  /// Returns an error because this binary does not contain the CUDA backend.
  pub fn new(
    words: &[u64],
    lengths: &[u32],
    word_count: usize,
    first: usize,
    min_chunks: usize,
    excluded: Option<&[u64]>,
  ) -> Result<Self, Error> {
    validate_input(words, lengths, word_count, first, min_chunks, excluded)?;
    Err(Error::new(UNAVAILABLE))
  }

  /// Returns an error because this binary does not contain the CUDA backend.
  pub fn next_pairs(&mut self) -> Result<Option<&[CandidatePair]>, Error> {
    Err(Error::new(UNAVAILABLE))
  }

  /// Returns empty counters for this unavailable backend.
  pub fn statistics(&self) -> FilterStatistics {
    FilterStatistics::default()
  }
}

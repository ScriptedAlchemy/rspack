//! These tests require an actual CUDA device and deliberately fail if unavailable.
//! Run explicitly with `cargo test -p rspack_cuda_test --features cuda`.

use rspack_cuda::{CandidatePair, MAX_BATCH_PAIRS, RoundFilter};

fn reference(
  words: &[u64],
  lengths: &[u32],
  width: usize,
  first: usize,
  minimum: usize,
  excluded: Option<&[u64]>,
) -> Vec<CandidatePair> {
  let mut result = Vec::new();
  for i in first..lengths.len() {
    for j in 0..i {
      let mut count = 0;
      let mut forbidden = false;
      for word in 0..width {
        let common = words[i * width + word] & words[j * width + word];
        count += common.count_ones() as usize;
        forbidden |= excluded.is_some_and(|mask| common & mask[word] != 0);
      }
      if !forbidden
        && count >= minimum
        && count < lengths[i] as usize
        && count < lengths[j] as usize
      {
        result.push(CandidatePair {
          left: i.try_into().expect("small test matrix"),
          right: j.try_into().expect("small test matrix"),
        });
      }
    }
  }
  result
}

fn check(words: &[u64], width: usize, first: usize, minimum: usize, excluded: Option<&[u64]>) {
  let lengths: Vec<u32> = words
    .chunks_exact(width)
    .map(|row| row.iter().map(|word| word.count_ones()).sum())
    .collect();
  let expected = reference(words, &lengths, width, first, minimum, excluded);
  let mut filter = RoundFilter::new(words, &lengths, width, first, minimum, excluded)
    .expect("CUDA must be available when the device suite is explicitly selected");
  let mut actual = Vec::new();
  while let Some(batch) = filter.next_pairs().expect("CUDA filtering failed") {
    assert!(batch.len() <= MAX_BATCH_PAIRS);
    actual.extend_from_slice(batch);
  }
  actual.sort_unstable_by_key(|pair| (pair.left, pair.right));
  assert_eq!(
    actual, expected,
    "width={width}, first={first}, minimum={minimum}"
  );
  let statistics = filter.statistics();
  assert_eq!(statistics.accepted_pairs, actual.len() as u64);
  let triangular = |n: usize| n * n.saturating_sub(1) / 2;
  assert_eq!(
    statistics.examined_pairs as usize,
    triangular(lengths.len()) - triangular(first)
  );
  assert!(statistics.upload_ms.is_finite() && statistics.upload_ms >= 0.0);
  assert!(statistics.kernel_ms.is_finite() && statistics.kernel_ms >= 0.0);
  assert!(statistics.download_ms.is_finite() && statistics.download_ms >= 0.0);
  if lengths.len() > 2 && first < lengths.len() {
    assert!(statistics.launches > 0);
  }
}

#[test]
fn test_empty_singleton_equal_subset_and_excluded_intersections() {
  let rows = [0, 1, 3, 6, 7, 3, 0b10101, 0b11100, 1 << 63, (1 << 63) | 3];
  for minimum in [1, 2, 3, 65] {
    for first in [0, 1, 5, rows.len()] {
      check(&rows, 1, first, minimum, None);
      check(&rows, 1, first, minimum, Some(&[2]));
      check(&rows, 1, first, minimum, Some(&[u64::MAX]));
    }
  }
  check(&[], 1, 0, 1, None);
  check(&[0], 1, 0, 1, None);
}

#[test]
fn test_seeded_multiword_and_partial_tile_inputs_match_scalar_reference() {
  let mut state = 0x4f6c_796d_7069_6164_u64;
  for bits in [1_usize, 63, 64, 65, 129, 1025] {
    let width = bits.div_ceil(64);
    let rows = 273;
    let mut words = Vec::with_capacity(rows * width);
    for _ in 0..rows {
      for word in 0..width {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = if word == width - 1 && bits % 64 != 0 {
          state & ((1_u64 << (bits % 64)) - 1)
        } else {
          state
        };
        words.push(value);
      }
    }
    let mut excluded = vec![0; width];
    excluded[width - 1] = 1;
    for first in [0, 17, 257, rows] {
      for minimum in [1, 2, 32] {
        check(&words, width, first, minimum, None);
        check(&words, width, first, minimum, Some(&excluded));
      }
    }
  }
}

#[test]
fn test_dense_survivors_cross_multiple_bounded_batches() {
  // Every row shares one bit, plus one of 63 independent bits. Most pairs
  // survive; duplicate rows must still be rejected as non-proper intersections.
  let words: Vec<_> = (0..1027).map(|row| 1 | (1_u64 << (1 + row % 63))).collect();
  check(&words, 1, 0, 1, None);
  check(&words, 1, 511, 1, None);
}

use rspack_cuda::RoundFilter;

fn assert_invalid(
  words: &[u64],
  lengths: &[u32],
  width: usize,
  first: usize,
  minimum: usize,
  excluded: Option<&[u64]>,
  expected: &str,
) {
  let error = match RoundFilter::new(words, lengths, width, first, minimum, excluded) {
    Ok(_) => panic!("accepted invalid bitmap dimensions"),
    Err(error) => error.to_string(),
  };
  assert!(
    error.contains(expected),
    "expected {expected:?}, received {error:?}"
  );
}

#[test]
fn test_rejects_invalid_dimensions_before_accessing_cuda() {
  assert_invalid(&[3], &[2, 2], 1, 0, 1, None, "dimensions");
  assert_invalid(&[3], &[2], 1, 0, 1, Some(&[]), "width");
  assert_invalid(&[3], &[2], 1, 2, 1, None, "first");
  assert_invalid(&[3], &[2], 1, 0, 0, None, "min_chunks");
  assert_invalid(&[3], &[65], 1, 0, 1, None, "capacity");
  assert_invalid(&[], &[1, 1], usize::MAX, 0, 1, None, "overflow");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_default_build_returns_unavailable_without_a_cuda_dependency() {
  assert!(!rspack_cuda::is_compiled());
  assert!(rspack_cuda::is_available().is_err());
  let error = match RoundFilter::new(&[3, 6], &[2, 2], 1, 0, 1, None) {
    Ok(_) => panic!("CUDA unexpectedly available in a default build"),
    Err(error) => error.to_string(),
  };
  assert!(error.contains("not compiled"), "{error}");
}

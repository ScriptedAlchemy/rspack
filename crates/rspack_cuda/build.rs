fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=src/intersections.cu");
  println!("cargo:rerun-if-env-changed=RSPACK_CUDA_ARCH");
  println!("cargo:rustc-check-cfg=cfg(rspack_cuda_native)");

  #[cfg(feature = "cuda")]
  build_cuda();
}

#[cfg(feature = "cuda")]
fn build_cuda() {
  let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("Cargo provides the target OS");
  let target_arch =
    std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo provides the target architecture");
  if target_os != "linux" || !matches!(target_arch.as_str(), "x86_64" | "aarch64") {
    println!(
      "cargo:warning=the CUDA experiment is available only on Linux x86_64 and aarch64; this target uses the unavailable backend"
    );
    return;
  }

  // Include PTX so the default build can JIT on newer GPUs without requiring a
  // GPU on the build host. Set 121 for a native DGX Spark build with a toolkit
  // supporting that target; NVCC reports unsupported architectures explicitly.
  let arch = std::env::var("RSPACK_CUDA_ARCH").unwrap_or_else(|_| "75".to_owned());
  assert!(
    (2..=3).contains(&arch.len()) && arch.bytes().all(|byte| byte.is_ascii_digit()),
    "RSPACK_CUDA_ARCH must be a numeric CUDA architecture, such as 75, 120 or 121"
  );

  cc::Build::new()
    .cuda(true)
    .cudart("static")
    .std("c++17")
    .pic(true)
    .opt_level(3)
    .flag(format!(
      "-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]"
    ))
    .file("src/intersections.cu")
    .compile("rspack_cuda_intersections");

  // The static CUDA runtime loads the driver at runtime. CPU-only machines can
  // still load a CUDA-enabled binding and let the caller select its fallback.
  println!("cargo:rustc-link-lib=dl");
  println!("cargo:rustc-link-lib=rt");
  println!("cargo:rustc-link-lib=pthread");
  println!("cargo:rustc-cfg=rspack_cuda_native");
}

use std::env;
use std::path::PathBuf;

const PORTABLE_CUDA_COMPUTE_CAP: &str = "80";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels/cuda/qwen35.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda/qwen38.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda/fp8.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda/sampling.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda/physical_state.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    // Release CUDA artifacts target the same Ampere floor used by the Docker
    // and CI builds instead of inheriting whichever GPU happens to be visible
    // on the build host. PTX compiled for compute_80 remains forward compatible
    // with newer NVIDIA architectures. Operators may still request another
    // deployment target explicitly through CUDA_COMPUTE_CAP.
    let compute_cap =
        env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| PORTABLE_CUDA_COMPUTE_CAP.to_string());
    cudaforge::GpuArch::parse(&compute_cap)?;
    let bindings = cudaforge::KernelBuilder::new()
        .compute_cap_arch(&compute_cap)
        .source_files(vec![
            "src/kernels/cuda/qwen35.cu",
            "src/kernels/cuda/qwen38.cu",
            "src/kernels/cuda/fp8.cu",
            "src/kernels/cuda/sampling.cu",
            "src/kernels/cuda/physical_state.cu",
        ])
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    bindings.write(out_dir.join("qwen35_ptx.rs"))?;
    Ok(())
}

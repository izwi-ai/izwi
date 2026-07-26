use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels/cuda/qwen35.cu");
    println!("cargo:rerun-if-changed=src/kernels/cuda/physical_state.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let bindings = cudaforge::KernelBuilder::new()
        .source_files(vec![
            "src/kernels/cuda/qwen35.cu",
            "src/kernels/cuda/physical_state.cu",
        ])
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    bindings.write(out_dir.join("qwen35_ptx.rs"))?;
    Ok(())
}

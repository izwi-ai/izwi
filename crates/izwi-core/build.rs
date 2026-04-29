use std::env;

const ARCH_ENV: &str = "IZWI_CUDA_KERNEL_ARCH";
const CANDLE_ARCH_ENV: &str = "CUDA_COMPUTE_CAP";

fn main() {
    println!("cargo:rerun-if-env-changed={ARCH_ENV}");
    println!("cargo:rerun-if-env-changed={CANDLE_ARCH_ENV}");
    println!("cargo:rustc-check-cfg=cfg(izwi_cuda_kernel_arch_configured)");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let raw_arch = env::var(ARCH_ENV)
        .ok()
        .or_else(|| env::var(CANDLE_ARCH_ENV).ok());

    let Some(raw_arch) = raw_arch else {
        println!("cargo:warning=CUDA kernels will use runtime/default architecture selection");
        return;
    };

    match normalize_arch(&raw_arch) {
        Some(arch) => {
            println!("cargo:rustc-env={ARCH_ENV}={arch}");
            println!("cargo:rustc-cfg=izwi_cuda_kernel_arch_configured");
        }
        None => {
            println!(
                "cargo:warning=Ignoring invalid CUDA kernel architecture override {raw_arch:?}; expected native, sm_80, compute_80, 80, or 8.0 style values"
            );
        }
    }
}

fn normalize_arch(raw: &str) -> Option<String> {
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "native" {
        return Some(trimmed);
    }

    let normalized = trimmed
        .strip_prefix("sm_")
        .or_else(|| trimmed.strip_prefix("compute_"))
        .unwrap_or(trimmed.as_str())
        .replace('.', "");

    if normalized.len() < 2 || normalized.len() > 3 {
        return None;
    }
    if !normalized.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }

    Some(format!("sm_{normalized}"))
}

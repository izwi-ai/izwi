//! GPU kernel implementations for model inference optimizations.
//!
//! This module provides fused kernels that combine multiple tensor operations
//! into single GPU dispatches, reducing memory bandwidth and kernel launch overhead.
//!
//! Currently supports:
//! - Metal (Apple Silicon) kernels for Qwen3.5 DeltaNet layers
//! - CUDA kernel dispatch scaffold with Candle fallbacks
//! - Fallback to Candle operations for unsupported backends and kernels

pub mod buffer_pool;
pub mod cuda;
pub mod metal;

use crate::error::Error;
use candle_core::Device;

/// Whether any fused kernel backend is available in this build.
pub fn fused_kernels_available() -> bool {
    metal_fused_kernels_available() || cuda::fused_kernels_available()
}

/// Whether fused kernels are available for a specific device backend.
pub fn fused_kernels_available_for_device(device: &Device) -> bool {
    if device.is_metal() {
        return metal_fused_kernels_available();
    }

    if device.is_cuda() {
        return cuda::fused_kernels_available();
    }

    false
}

fn metal_fused_kernels_available() -> bool {
    cfg!(target_os = "macos")
}

/// Whether to use fused kernels (can be disabled via environment).
pub fn use_fused_kernels() -> bool {
    std::env::var("IZWI_FUSED_KERNELS")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
        && fused_kernels_available()
}

/// Result type for fused kernel operations.
pub type FusedResult<T> = std::result::Result<T, FusedKernelError>;

#[derive(Debug, Clone)]
pub enum FusedKernelError {
    BackendNotSupported(String),
    KernelNotAvailable(String),
    ExecutionError(String),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    DtypeNotSupported(String),
}

impl std::fmt::Display for FusedKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotSupported(s) => write!(f, "Backend not supported: {}", s),
            Self::KernelNotAvailable(s) => write!(f, "Kernel not available: {}", s),
            Self::ExecutionError(s) => write!(f, "Kernel execution error: {}", s),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::DtypeNotSupported(s) => write!(f, "Dtype not supported: {}", s),
        }
    }
}

impl std::error::Error for FusedKernelError {}

impl From<FusedKernelError> for Error {
    fn from(e: FusedKernelError) -> Self {
        Error::InferenceError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_never_reports_fused_kernel_availability() {
        assert!(!fused_kernels_available_for_device(&Device::Cpu));
    }

    #[test]
    fn global_fused_kernel_availability_tracks_known_backends() {
        assert_eq!(
            fused_kernels_available(),
            cfg!(target_os = "macos") || cuda::fused_kernels_available()
        );
    }
}

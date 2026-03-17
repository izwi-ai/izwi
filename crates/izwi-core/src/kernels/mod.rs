//! GPU kernel implementations for model inference optimizations.
//!
//! This module provides fused kernels that combine multiple tensor operations
//! into single GPU dispatches, reducing memory bandwidth and kernel launch overhead.
//!
//! Currently supports:
//! - Metal (Apple Silicon) kernels for Qwen3.5 DeltaNet layers
//! - Fallback to Candle operations for other backends

pub mod metal;

use crate::error::Error;

/// Whether fused kernels are available for the current device.
pub fn fused_kernels_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        true // Metal kernels available on macOS
    }
    #[cfg(not(target_os = "macos"))]
    {
        false // TODO: CUDA support
    }
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

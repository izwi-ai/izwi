//! CUDA kernel scaffold.
//!
//! This module defines the CUDA kernel dispatch surface without enabling any
//! custom CUDA kernels yet. Callers can query the status and fall back to
//! Candle operations until concrete kernels are implemented behind `cuda`.

use candle_core::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelStatus {
    pub compiled: bool,
    pub available: bool,
    pub reason: &'static str,
}

pub fn cuda_kernels_compiled() -> bool {
    cfg!(feature = "cuda")
}

pub fn fused_kernels_available() -> bool {
    false
}

pub fn status() -> CudaKernelStatus {
    if !cuda_kernels_compiled() {
        return CudaKernelStatus {
            compiled: false,
            available: false,
            reason: "binary was not built with CUDA support",
        };
    }

    CudaKernelStatus {
        compiled: true,
        available: false,
        reason: "custom CUDA kernels are scaffolded but not implemented",
    }
}

pub fn try_fused_silu_mul(_gate: &Tensor, _up: &Tensor) -> Option<Tensor> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_kernel_status_is_explicit() {
        let status = status();
        assert_eq!(status.compiled, cfg!(feature = "cuda"));
        assert!(!status.available);
        assert!(!status.reason.trim().is_empty());
    }
}

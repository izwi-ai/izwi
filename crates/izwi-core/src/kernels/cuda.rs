//! CUDA kernel dispatch scaffolding.
//!
//! These functions are no-op contracts for future CUDA kernels. They record
//! CUDA-specific attempts/fallbacks and return `Ok(None)` until an implementation
//! lands in a later phase.

use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::models::shared::telemetry::{
    record_cuda_kernel_attempt, record_cuda_kernel_fallback, CudaKernelFallbackReason,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelKind {
    UnmaskedAttention,
    MaskedAttention,
    PagedDecodeAttention,
    Rope,
    MRope,
    RmsNorm,
    GatedRmsNorm,
    L2Norm,
    SiluMul,
    SwiGlu,
    KvQuantize,
    KvDequantize,
    QuantizedMatmul,
    ShortConv,
    DeltaNet,
    ArgmaxSampling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelDecision {
    Try,
    Skip(CudaKernelFallbackReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelBuildMode {
    NotCompiled,
    RuntimeDefault,
    ConfiguredArch(&'static str),
}

pub fn cuda_kernels_compiled() -> bool {
    cfg!(feature = "cuda")
}

pub fn configured_cuda_kernel_arch() -> Option<&'static str> {
    option_env!("IZWI_CUDA_KERNEL_ARCH")
}

pub fn cuda_kernel_build_mode() -> CudaKernelBuildMode {
    if !cuda_kernels_compiled() {
        return CudaKernelBuildMode::NotCompiled;
    }
    configured_cuda_kernel_arch()
        .map(CudaKernelBuildMode::ConfiguredArch)
        .unwrap_or(CudaKernelBuildMode::RuntimeDefault)
}

pub fn cuda_kernels_available(device: &Device) -> bool {
    matches!(cuda_kernel_preflight(device), CudaKernelDecision::Try)
}

pub fn cuda_kernel_preflight(device: &Device) -> CudaKernelDecision {
    if !device.is_cuda() {
        return CudaKernelDecision::Skip(CudaKernelFallbackReason::NotCudaDevice);
    }
    if !cuda_kernels_compiled() {
        return CudaKernelDecision::Skip(CudaKernelFallbackReason::NotCompiled);
    }
    CudaKernelDecision::Try
}

pub fn try_cuda_attention(
    q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _mask: Option<&Tensor>,
    _head_dim: usize,
    _causal: bool,
    _kind: CudaKernelKind,
) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(q.device())
}

pub fn try_cuda_paged_decode_attention(
    q: &Tensor,
    _head_dim: usize,
    _num_heads: usize,
    _num_kv_heads: usize,
) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(q.device())
}

pub fn try_cuda_rotary(
    x: &Tensor,
    _cos: &Tensor,
    _sin: &Tensor,
    _kind: CudaKernelKind,
) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(x.device())
}

pub fn try_cuda_rotary_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    kind: CudaKernelKind,
) -> Result<Option<(Tensor, Tensor)>> {
    if !q.device().is_cuda() {
        return Ok(None);
    }
    let q_out = try_cuda_rotary(q, cos, sin, kind)?;
    let k_out = try_cuda_rotary(k, cos, sin, kind)?;
    Ok(match (q_out, k_out) {
        (Some(q_out), Some(k_out)) => Some((q_out, k_out)),
        _ => None,
    })
}

pub fn try_cuda_norm(
    x: &Tensor,
    _weight: &Tensor,
    _eps: f64,
    _kind: CudaKernelKind,
) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(x.device())
}

pub fn try_cuda_binary_activation(
    lhs: &Tensor,
    _rhs: &Tensor,
    _kind: CudaKernelKind,
) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(lhs.device())
}

pub fn try_cuda_kv_quantize(page: &Tensor, _kind: CudaKernelKind) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(page.device())
}

pub fn try_cuda_quantized_matmul(input: &Tensor, _kind: CudaKernelKind) -> Result<Option<Tensor>> {
    record_cuda_scaffold_attempt(input.device())
}

fn record_cuda_scaffold_attempt(device: &Device) -> Result<Option<Tensor>> {
    record_cuda_kernel_attempt();
    match cuda_kernel_preflight(device) {
        CudaKernelDecision::Try => {
            record_cuda_kernel_fallback(CudaKernelFallbackReason::NotImplemented);
        }
        CudaKernelDecision::Skip(reason) => {
            record_cuda_kernel_fallback(reason);
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{
        configured_cuda_kernel_arch, cuda_kernel_build_mode, cuda_kernel_preflight,
        cuda_kernels_available, cuda_kernels_compiled, try_cuda_attention, CudaKernelBuildMode,
        CudaKernelDecision, CudaKernelKind,
    };
    use crate::models::shared::telemetry;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn cuda_kernel_preflight_skips_cpu_device() {
        assert_eq!(
            cuda_kernel_preflight(&Device::Cpu),
            CudaKernelDecision::Skip(telemetry::CudaKernelFallbackReason::NotCudaDevice)
        );
        assert!(!cuda_kernels_available(&Device::Cpu));
    }

    #[test]
    fn cuda_kernel_compiled_reflects_feature_flag() {
        assert_eq!(cuda_kernels_compiled(), cfg!(feature = "cuda"));
    }

    #[test]
    fn cuda_kernel_build_mode_reflects_compile_state() {
        match cuda_kernel_build_mode() {
            CudaKernelBuildMode::NotCompiled => assert!(!cuda_kernels_compiled()),
            CudaKernelBuildMode::RuntimeDefault => {
                assert!(cuda_kernels_compiled());
                assert!(configured_cuda_kernel_arch().is_none());
            }
            CudaKernelBuildMode::ConfiguredArch(arch) => {
                assert!(cuda_kernels_compiled());
                assert_eq!(configured_cuda_kernel_arch(), Some(arch));
            }
        }
    }

    #[test]
    fn cuda_attention_scaffold_records_cpu_fallback() {
        telemetry::reset_for_tests();
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 1, 1, 8), DType::F32, &device).expect("q tensor");
        let k = Tensor::zeros((1, 1, 1, 8), DType::F32, &device).expect("k tensor");
        let v = Tensor::zeros((1, 1, 1, 8), DType::F32, &device).expect("v tensor");

        let out = try_cuda_attention(&q, &k, &v, None, 8, true, CudaKernelKind::UnmaskedAttention)
            .expect("scaffold call");

        assert!(out.is_none());
        let snapshot = telemetry::snapshot();
        assert_eq!(snapshot.cuda_kernel_attempts_total, 1);
        assert_eq!(snapshot.cuda_kernel_fallback_total, 1);
        assert_eq!(snapshot.cuda_kernel_fallback_not_cuda_device_total, 1);
    }
}

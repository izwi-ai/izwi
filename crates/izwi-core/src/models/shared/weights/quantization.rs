//! Backend-aware quantized weight compatibility policy.
//!
//! This module does not load weights by itself. It centralizes the policy used
//! to decide whether CUDA may use a native/delegated quantized path, an explicit
//! dense fallback, or must reject an unsafe implicit host fallback.

use candle_core::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedBackend {
    Cpu,
    Metal,
    Cuda,
}

impl QuantizedBackend {
    pub fn from_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Cpu
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedWeightFormat {
    DenseSafetensors,
    GgufDense,
    GgufQ4_0,
    GgufQ4K,
    GgufQ5K,
    GgufQ8_0,
    GgufOther,
    MlxAffine,
    KvCacheInt8,
    KvCacheQ4_0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedExecutionMode {
    ExistingBackendPath,
    CandleCudaGeneric,
    DenseDequantFallback,
    ExplicitHostFallback,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedCompatibilityReason {
    ExistingCpuOrMetalPolicy,
    DenseWeights,
    DelegatedToCandleCuda,
    CudaDenseFallbackRequired,
    CudaKernelMissing,
    ExplicitCudaHostFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedCompatibility {
    pub mode: QuantizedExecutionMode,
    pub reason: QuantizedCompatibilityReason,
}

impl QuantizedCompatibility {
    pub const fn new(mode: QuantizedExecutionMode, reason: QuantizedCompatibilityReason) -> Self {
        Self { mode, reason }
    }

    pub const fn is_supported(self) -> bool {
        !matches!(self.mode, QuantizedExecutionMode::Unsupported)
    }
}

pub fn quantized_compatibility_for_device(
    device: &Device,
    format: QuantizedWeightFormat,
) -> QuantizedCompatibility {
    quantized_compatibility(
        QuantizedBackend::from_device(device),
        format,
        cuda_quantized_host_fallback_enabled(),
    )
}

pub fn quantized_compatibility(
    backend: QuantizedBackend,
    format: QuantizedWeightFormat,
    explicit_cuda_host_fallback: bool,
) -> QuantizedCompatibility {
    if backend != QuantizedBackend::Cuda {
        return QuantizedCompatibility::new(
            QuantizedExecutionMode::ExistingBackendPath,
            QuantizedCompatibilityReason::ExistingCpuOrMetalPolicy,
        );
    }

    match format {
        QuantizedWeightFormat::DenseSafetensors | QuantizedWeightFormat::GgufDense => {
            QuantizedCompatibility::new(
                QuantizedExecutionMode::ExistingBackendPath,
                QuantizedCompatibilityReason::DenseWeights,
            )
        }
        QuantizedWeightFormat::GgufQ4_0
        | QuantizedWeightFormat::GgufQ4K
        | QuantizedWeightFormat::GgufQ5K
        | QuantizedWeightFormat::GgufQ8_0 => QuantizedCompatibility::new(
            QuantizedExecutionMode::CandleCudaGeneric,
            QuantizedCompatibilityReason::DelegatedToCandleCuda,
        ),
        QuantizedWeightFormat::MlxAffine => QuantizedCompatibility::new(
            QuantizedExecutionMode::DenseDequantFallback,
            QuantizedCompatibilityReason::CudaDenseFallbackRequired,
        ),
        QuantizedWeightFormat::KvCacheInt8 => QuantizedCompatibility::new(
            QuantizedExecutionMode::ExistingBackendPath,
            QuantizedCompatibilityReason::DenseWeights,
        ),
        QuantizedWeightFormat::KvCacheQ4_0 if explicit_cuda_host_fallback => {
            QuantizedCompatibility::new(
                QuantizedExecutionMode::ExplicitHostFallback,
                QuantizedCompatibilityReason::ExplicitCudaHostFallback,
            )
        }
        QuantizedWeightFormat::KvCacheQ4_0 | QuantizedWeightFormat::GgufOther => {
            QuantizedCompatibility::new(
                QuantizedExecutionMode::Unsupported,
                QuantizedCompatibilityReason::CudaKernelMissing,
            )
        }
    }
}

fn cuda_quantized_host_fallback_enabled() -> bool {
    env_bool("IZWI_CUDA_QUANTIZED_HOST_FALLBACK") || env_bool("IZWI_CUDA_KV_Q4_0_HOST_FALLBACK")
}

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{
        quantized_compatibility, QuantizedBackend, QuantizedCompatibilityReason,
        QuantizedExecutionMode, QuantizedWeightFormat,
    };

    #[test]
    fn non_cuda_backends_keep_existing_policy_for_all_formats() {
        for backend in [QuantizedBackend::Cpu, QuantizedBackend::Metal] {
            for format in [
                QuantizedWeightFormat::DenseSafetensors,
                QuantizedWeightFormat::GgufQ4_0,
                QuantizedWeightFormat::GgufQ4K,
                QuantizedWeightFormat::GgufQ5K,
                QuantizedWeightFormat::GgufQ8_0,
                QuantizedWeightFormat::MlxAffine,
                QuantizedWeightFormat::KvCacheInt8,
                QuantizedWeightFormat::KvCacheQ4_0,
            ] {
                let policy = quantized_compatibility(backend, format, false);
                assert_eq!(policy.mode, QuantizedExecutionMode::ExistingBackendPath);
                assert_eq!(
                    policy.reason,
                    QuantizedCompatibilityReason::ExistingCpuOrMetalPolicy
                );
                assert!(policy.is_supported());
            }
        }
    }

    #[test]
    fn cuda_delegates_known_gguf_quantized_formats_to_candle() {
        for format in [
            QuantizedWeightFormat::GgufQ4_0,
            QuantizedWeightFormat::GgufQ4K,
            QuantizedWeightFormat::GgufQ5K,
            QuantizedWeightFormat::GgufQ8_0,
        ] {
            let policy = quantized_compatibility(QuantizedBackend::Cuda, format, false);
            assert_eq!(policy.mode, QuantizedExecutionMode::CandleCudaGeneric);
            assert_eq!(
                policy.reason,
                QuantizedCompatibilityReason::DelegatedToCandleCuda
            );
            assert!(policy.is_supported());
        }
    }

    #[test]
    fn cuda_marks_mlx_affine_as_dense_dequant_fallback() {
        let policy = quantized_compatibility(
            QuantizedBackend::Cuda,
            QuantizedWeightFormat::MlxAffine,
            false,
        );
        assert_eq!(policy.mode, QuantizedExecutionMode::DenseDequantFallback);
        assert_eq!(
            policy.reason,
            QuantizedCompatibilityReason::CudaDenseFallbackRequired
        );
        assert!(policy.is_supported());
    }

    #[test]
    fn cuda_blocks_unknown_and_implicit_q4_kv_fallbacks() {
        for format in [
            QuantizedWeightFormat::GgufOther,
            QuantizedWeightFormat::KvCacheQ4_0,
        ] {
            let policy = quantized_compatibility(QuantizedBackend::Cuda, format, false);
            assert_eq!(policy.mode, QuantizedExecutionMode::Unsupported);
            assert_eq!(
                policy.reason,
                QuantizedCompatibilityReason::CudaKernelMissing
            );
            assert!(!policy.is_supported());
        }
    }

    #[test]
    fn cuda_allows_explicit_host_fallback_for_q4_kv_only() {
        let policy = quantized_compatibility(
            QuantizedBackend::Cuda,
            QuantizedWeightFormat::KvCacheQ4_0,
            true,
        );
        assert_eq!(policy.mode, QuantizedExecutionMode::ExplicitHostFallback);
        assert_eq!(
            policy.reason,
            QuantizedCompatibilityReason::ExplicitCudaHostFallback
        );
        assert!(policy.is_supported());
    }
}

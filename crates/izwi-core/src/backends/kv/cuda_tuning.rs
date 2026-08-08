use crate::{Error, Result};
#[cfg(feature = "cuda")]
use candle_core::DeviceLocation;
use candle_core::{DType, Device};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaKvStorageFormat {
    Dense,
    Fp8E4M3,
}

impl CudaKvStorageFormat {
    pub(crate) const fn dtype(self, logical_dtype: DType) -> DType {
        match self {
            Self::Dense => logical_dtype,
            Self::Fp8E4M3 => DType::F8E4M3,
        }
    }
}

pub(crate) fn resolve_cuda_kv_storage_format(
    identity: &CudaDeviceIdentity,
    logical_dtype: DType,
    performance_certified: bool,
) -> Result<CudaKvStorageFormat> {
    if !performance_certified {
        return Ok(CudaKvStorageFormat::Dense);
    }
    if !matches!(logical_dtype, DType::F16 | DType::BF16) {
        return Err(Error::InvalidInput(
            "certified CUDA FP8 KV requires F16 or BF16 model KV dtype".into(),
        ));
    }
    if !identity
        .compute_capability
        .is_some_and(|(major, _)| major >= 9)
    {
        return Err(Error::InvalidInput(
            "certified CUDA FP8 KV requires an observed compute capability of 9.0 or newer".into(),
        ));
    }
    Ok(CudaKvStorageFormat::Fp8E4M3)
}

/// Only reviewed NVIDIA evidence may add an exact cell here. The empty table
/// makes FP8 unreachable by default, and there is deliberately no environment
/// override that can relabel an unverified route as certified.
pub(crate) fn cuda_fp8_kv_cell_certified(
    _identity: &CudaDeviceIdentity,
    _logical_dtype: DType,
) -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CudaDeviceIdentity {
    pub(crate) device_name: Option<String>,
    pub(crate) compute_capability: Option<(u32, u32)>,
}

impl CudaDeviceIdentity {
    pub(crate) const fn unobserved() -> Self {
        Self {
            device_name: None,
            compute_capability: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaPagedShapeKey {
    pub(crate) dtype: DType,
    pub(crate) page_tokens: u32,
    pub(crate) key_head_dim: usize,
    pub(crate) value_head_dim: usize,
    pub(crate) batch: usize,
    pub(crate) query_heads: usize,
    pub(crate) max_context_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaPagedTuningPolicy {
    pub(crate) flash_attention_allowed: bool,
    pub(crate) decode_partition_tuning: Option<(usize, usize)>,
    pub(crate) decode_graph_allowed: bool,
}

/// Resolve only conservative, architecture-safe defaults. Performance
/// certification can replace these values for an exact GPU/shape cell later;
/// an unobserved or pre-Ampere device always retains the eager native path.
pub(crate) fn resolve_cuda_paged_tuning(
    identity: &CudaDeviceIdentity,
    shape: CudaPagedShapeKey,
) -> CudaPagedTuningPolicy {
    let ampere_or_newer = identity
        .compute_capability
        .is_some_and(|(major, _)| major >= 8);
    let supported_page = matches!(shape.page_tokens, 16 | 32 | 64);
    let supported_dtype = matches!(shape.dtype, DType::F16 | DType::BF16);
    let matched_dims = shape.key_head_dim == shape.value_head_dim
        && shape.key_head_dim != 0
        && shape.key_head_dim <= 512
        && shape.key_head_dim % 8 == 0;
    let nonempty_shape = shape.batch > 0 && shape.query_heads > 0 && shape.max_context_tokens > 0;

    CudaPagedTuningPolicy {
        flash_attention_allowed: ampere_or_newer
            && supported_page
            && supported_dtype
            && matched_dims
            && nonempty_shape,
        decode_partition_tuning: (ampere_or_newer && supported_page && nonempty_shape)
            .then_some((2_048, 1_024)),
        decode_graph_allowed: ampere_or_newer
            && supported_page
            && supported_dtype
            && matched_dims
            && nonempty_shape,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn observe_cuda_identity(device: &Device) -> CudaDeviceIdentity {
    use candle_core::cuda_backend::cudarc::driver::CudaContext;

    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return CudaDeviceIdentity::unobserved();
    };
    let Ok(context) = CudaContext::new(gpu_id) else {
        return CudaDeviceIdentity::unobserved();
    };
    let compute_capability = context
        .compute_capability()
        .ok()
        .map(|(major, minor)| (major.max(0) as u32, minor.max(0) as u32));
    let device_name = context
        .name()
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    CudaDeviceIdentity {
        device_name,
        compute_capability,
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn observe_cuda_identity(_device: &Device) -> CudaDeviceIdentity {
    CudaDeviceIdentity::unobserved()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> CudaPagedShapeKey {
        CudaPagedShapeKey {
            dtype: DType::F16,
            page_tokens: 64,
            key_head_dim: 128,
            value_head_dim: 128,
            batch: 4,
            query_heads: 32,
            max_context_tokens: 8_192,
        }
    }

    #[test]
    fn unobserved_and_pre_ampere_devices_keep_eager_native_fallbacks() {
        for compute_capability in [None, Some((7, 5))] {
            let policy = resolve_cuda_paged_tuning(
                &CudaDeviceIdentity {
                    device_name: Some("test".into()),
                    compute_capability,
                },
                shape(),
            );
            assert!(!policy.flash_attention_allowed);
            assert_eq!(policy.decode_partition_tuning, None);
            assert!(!policy.decode_graph_allowed);
        }
    }

    #[test]
    fn ampere_policy_is_keyed_by_page_dtype_and_head_geometry() {
        let identity = CudaDeviceIdentity {
            device_name: Some("A100".into()),
            compute_capability: Some((8, 0)),
        };
        let policy = resolve_cuda_paged_tuning(&identity, shape());
        assert!(policy.flash_attention_allowed);
        assert_eq!(policy.decode_partition_tuning, Some((2_048, 1_024)));
        assert!(policy.decode_graph_allowed);

        let mut unsupported = shape();
        unsupported.page_tokens = 8;
        let policy = resolve_cuda_paged_tuning(&identity, unsupported);
        assert!(!policy.flash_attention_allowed);
        assert_eq!(policy.decode_partition_tuning, None);
        assert!(!policy.decode_graph_allowed);

        unsupported = shape();
        unsupported.dtype = DType::F32;
        let policy = resolve_cuda_paged_tuning(&identity, unsupported);
        assert!(!policy.flash_attention_allowed);
        assert!(!policy.decode_graph_allowed);
    }

    #[test]
    fn fp8_storage_is_dense_until_exact_hopper_cell_is_certified() {
        let hopper = CudaDeviceIdentity {
            device_name: Some("H100".into()),
            compute_capability: Some((9, 0)),
        };
        assert_eq!(
            resolve_cuda_kv_storage_format(&hopper, DType::F16, false).unwrap(),
            CudaKvStorageFormat::Dense
        );
        assert_eq!(
            resolve_cuda_kv_storage_format(&hopper, DType::BF16, true).unwrap(),
            CudaKvStorageFormat::Fp8E4M3
        );
        assert!(!cuda_fp8_kv_cell_certified(&hopper, DType::BF16));
    }

    #[test]
    fn fp8_certificate_fails_closed_without_supported_hardware_or_dtype() {
        assert!(resolve_cuda_kv_storage_format(
            &CudaDeviceIdentity::unobserved(),
            DType::F16,
            true
        )
        .is_err());
        assert!(resolve_cuda_kv_storage_format(
            &CudaDeviceIdentity {
                device_name: Some("H100".into()),
                compute_capability: Some((9, 0)),
            },
            DType::F32,
            true
        )
        .is_err());
    }
}

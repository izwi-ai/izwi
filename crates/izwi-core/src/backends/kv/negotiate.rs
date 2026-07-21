use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};
use crate::kv::{
    KvArenaId, KvCacheContract, KvDomainSpec, KvGroupId, KvLayerBinding, KvPhysicalLayout,
    KvStorageDType, KvStorageFormat, PagedAttentionKernel, ResolvedKvGroup, ResolvedKvGroupKind,
    ResolvedKvPlan,
};

/// Capacity and layout hints supplied to backend KV negotiation.
///
/// All model geometry comes from `KvCacheContract`; callers cannot override
/// layer counts, head counts, or head dimensions through this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBackendPlanRequest {
    pub model_instance: ModelInstanceId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub capacity_pages: u32,
    pub page_tokens_hint: Option<u32>,
    pub storage_dtype_hint: Option<KvStorageDType>,
    pub first_arena_generation: u32,
}

/// Resolve a validated loaded-model contract for an implemented physical KV
/// backend. Accelerator negotiation remains fail-closed until its arena and
/// direct attention kernels implement the same runtime ABI.
pub fn negotiate_kv_plan(
    contract: &KvCacheContract,
    request: &KvBackendPlanRequest,
) -> Result<ResolvedKvPlan> {
    contract.validate()?;
    if request.capacity_pages == 0 || request.first_arena_generation == 0 {
        return Err(Error::InvalidInput(
            "KV capacity and first arena generation must be non-zero".into(),
        ));
    }

    match request.backend {
        BackendKind::Cpu => negotiate_dense_paged_plan(
            contract,
            request,
            select_cpu_dtype,
            |spec, hint| {
                Ok(hint
                    .filter(|value| spec.page_tokens.accepts(*value))
                    .unwrap_or(spec.page_tokens.preferred))
            },
            PagedAttentionKernel::PortableReference,
        ),
        BackendKind::Cuda => {
            #[cfg(feature = "flash-attn")]
            {
                negotiate_dense_paged_plan(
                    contract,
                    request,
                    select_cuda_flash_dtype,
                    select_cuda_flash_page_tokens,
                    PagedAttentionKernel::CudaFlashAttention,
                )
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                Err(Error::InvalidInput(
                    "managed CUDA KV requires the flash-attn feature for direct paged attention"
                        .into(),
                ))
            }
        }
        BackendKind::Metal => Err(Error::InvalidInput(
            "managed Metal KV is unavailable because Candle 0.11 has no direct paged-attention kernel"
                .into(),
        )),
    }
}

fn negotiate_dense_paged_plan(
    contract: &KvCacheContract,
    request: &KvBackendPlanRequest,
    select_dtype: fn(&[KvStorageDType], Option<KvStorageDType>) -> Result<KvStorageDType>,
    select_page_tokens: fn(&crate::kv::PagedAttentionDomainSpec, Option<u32>) -> Result<u32>,
    kernel: PagedAttentionKernel,
) -> Result<ResolvedKvPlan> {
    let mut groups = Vec::with_capacity(contract.domains.len());
    for (ordinal, domain) in contract.domains.iter().enumerate() {
        let ordinal = u32::try_from(ordinal)
            .map_err(|_| Error::InvalidInput("KV group count exceeds u32".into()))?;
        let arena = KvArenaId {
            model_instance: request.model_instance,
            backend: request.backend,
            device_ordinal: request.device_ordinal,
            generation: request
                .first_arena_generation
                .checked_add(ordinal)
                .ok_or_else(|| Error::InvalidInput("KV arena generation overflow".into()))?,
        };
        let id = KvGroupId::new(ordinal);

        groups.push(match domain {
            KvDomainSpec::PagedAttention(spec) => {
                let page_tokens = select_page_tokens(spec, request.page_tokens_hint)?;
                let dtype = select_dtype(&spec.storage.dtypes, request.storage_dtype_hint)?;
                let dtype_bytes = dtype.dense_bytes().ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "{:?} cannot use dense {dtype:?} KV storage",
                        request.backend
                    ))
                })?;
                let mut bytes_per_page = 0_u64;
                let mut layers = Vec::with_capacity(spec.layers.len());
                for (physical_layer, layer) in spec.layers.iter().enumerate() {
                    if kernel == PagedAttentionKernel::CudaFlashAttention
                        && (layer.key_head_dim != layer.value_head_dim
                            || layer.key_head_dim > 512
                            || layer.key_head_dim % 8 != 0)
                    {
                        return Err(Error::InvalidInput(format!(
                            "CUDA paged flash attention requires equal K/V head dimensions that are multiples of 8 and at most 512; layer {} has K={} V={}",
                            layer.model_layer, layer.key_head_dim, layer.value_head_dim
                        )));
                    }
                    let one_side = u64::from(page_tokens)
                        .checked_mul(u64::from(layer.num_kv_heads))
                        .ok_or_else(geometry_overflow)?;
                    let elements = one_side
                        .checked_mul(u64::from(layer.key_head_dim))
                        .and_then(|keys| {
                            one_side
                                .checked_mul(u64::from(layer.value_head_dim))
                                .and_then(|values| keys.checked_add(values))
                        })
                        .ok_or_else(geometry_overflow)?;
                    bytes_per_page = bytes_per_page
                        .checked_add(
                            elements
                                .checked_mul(dtype_bytes)
                                .ok_or_else(geometry_overflow)?,
                        )
                        .ok_or_else(geometry_overflow)?;
                    layers.push(KvLayerBinding {
                        model_layer: layer.model_layer,
                        physical_layer: u32::try_from(physical_layer).map_err(|_| {
                            Error::InvalidInput("KV layer count exceeds u32".into())
                        })?,
                    });
                }
                ResolvedKvGroup {
                    id,
                    arena,
                    domain: spec.id,
                    page_tokens,
                    capacity_pages: request.capacity_pages,
                    bytes_per_page,
                    layout: KvPhysicalLayout::PageTokenHeadDim,
                    storage: KvStorageFormat::Dense { dtype },
                    kernel,
                    kind: ResolvedKvGroupKind::PagedAttention { layers },
                }
            }
            KvDomainSpec::ModelState(spec) => {
                if request.backend != BackendKind::Cpu {
                    return Err(Error::InvalidInput(format!(
                        "managed {:?} KV does not implement model-state domains",
                        request.backend
                    )));
                }
                let dtype = select_dtype(&spec.storage.dtypes, request.storage_dtype_hint)?;
                let dtype_bytes = dtype.dense_bytes().ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "{:?} cannot use dense {dtype:?} state storage",
                        request.backend
                    ))
                })?;
                let mut bytes_per_page = 0_u64;
                let mut layers = Vec::with_capacity(spec.layers.len());
                for (physical_layer, layer) in spec.layers.iter().enumerate() {
                    bytes_per_page = bytes_per_page
                        .checked_add(
                            layer
                                .elements_per_sequence
                                .checked_mul(dtype_bytes)
                                .ok_or_else(geometry_overflow)?,
                        )
                        .ok_or_else(geometry_overflow)?;
                    layers.push(KvLayerBinding {
                        model_layer: layer.model_layer,
                        physical_layer: u32::try_from(physical_layer).map_err(|_| {
                            Error::InvalidInput("KV state layer count exceeds u32".into())
                        })?,
                    });
                }
                ResolvedKvGroup {
                    id,
                    arena,
                    domain: spec.id,
                    page_tokens: 1,
                    capacity_pages: request.capacity_pages,
                    bytes_per_page,
                    layout: KvPhysicalLayout::PageTokenHeadDim,
                    storage: KvStorageFormat::Dense { dtype },
                    kernel: PagedAttentionKernel::PortableReference,
                    kind: ResolvedKvGroupKind::ModelState { layers },
                }
            }
        });
    }

    ResolvedKvPlan::build(
        request.model_instance,
        request.backend,
        request.device_ordinal,
        contract,
        groups,
    )
}

#[cfg(feature = "flash-attn")]
fn select_cuda_flash_dtype(
    accepted: &[KvStorageDType],
    hint: Option<KvStorageDType>,
) -> Result<KvStorageDType> {
    const SUPPORTED: [KvStorageDType; 2] = [KvStorageDType::F16, KvStorageDType::Bf16];
    if let Some(dtype) = hint.filter(|dtype| accepted.contains(dtype) && SUPPORTED.contains(dtype))
    {
        return Ok(dtype);
    }
    accepted
        .iter()
        .copied()
        .find(|dtype| SUPPORTED.contains(dtype))
        .ok_or_else(|| {
            Error::InvalidInput(
                "CUDA paged flash attention found no compatible F16/BF16 KV dtype".into(),
            )
        })
}

#[cfg(feature = "flash-attn")]
fn select_cuda_flash_page_tokens(
    spec: &crate::kv::PagedAttentionDomainSpec,
    hint: Option<u32>,
) -> Result<u32> {
    let accepts = |value: u32| spec.page_tokens.accepts(value) && value % 32 == 0;
    if let Some(value) = hint.filter(|value| accepts(*value)) {
        return Ok(value);
    }
    if accepts(spec.page_tokens.preferred) {
        return Ok(spec.page_tokens.preferred);
    }

    let step = lcm(spec.page_tokens.multiple_of, 32).ok_or_else(geometry_overflow)?;
    let first = spec
        .page_tokens
        .min
        .div_ceil(step)
        .checked_mul(step)
        .ok_or_else(geometry_overflow)?;
    if first <= spec.page_tokens.max {
        Ok(first)
    } else {
        Err(Error::InvalidInput(
            "CUDA paged flash attention requires a page size divisible by 32".into(),
        ))
    }
}

#[cfg(feature = "flash-attn")]
fn lcm(left: u32, right: u32) -> Option<u32> {
    fn gcd(mut left: u32, mut right: u32) -> u32 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

fn select_cpu_dtype(
    accepted: &[KvStorageDType],
    hint: Option<KvStorageDType>,
) -> Result<KvStorageDType> {
    const SUPPORTED: [KvStorageDType; 3] = [
        KvStorageDType::F32,
        KvStorageDType::F16,
        KvStorageDType::Bf16,
    ];
    if let Some(dtype) = hint.filter(|dtype| accepted.contains(dtype) && SUPPORTED.contains(dtype))
    {
        return Ok(dtype);
    }
    accepted
        .iter()
        .copied()
        .find(|dtype| SUPPORTED.contains(dtype))
        .ok_or_else(|| Error::InvalidInput("CPU found no compatible dense KV dtype".into()))
}

fn geometry_overflow() -> Error {
    Error::InvalidInput("resolved KV geometry overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::test_contract;

    #[test]
    fn cpu_negotiation_uses_loaded_geometry_and_hints() {
        let contract = test_contract();
        let plan = negotiate_kv_plan(
            &contract,
            &KvBackendPlanRequest {
                model_instance: ModelInstanceId::new(9),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                capacity_pages: 100,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(KvStorageDType::Bf16),
                first_arena_generation: 3,
            },
        )
        .unwrap();
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].page_tokens, 16);
        assert_eq!(plan.groups[0].capacity_pages, 100);
        assert_eq!(
            plan.groups[0].storage,
            KvStorageFormat::Dense {
                dtype: KvStorageDType::Bf16
            }
        );
        assert_eq!(plan.groups[0].bytes_per_page, 16 * 4 * 128 * 2);
    }

    #[test]
    fn metal_negotiation_fails_closed_without_direct_attention() {
        let error = negotiate_kv_plan(
            &test_contract(),
            &KvBackendPlanRequest {
                model_instance: ModelInstanceId::new(9),
                backend: BackendKind::Metal,
                device_ordinal: Some(0),
                capacity_pages: 10,
                page_tokens_hint: None,
                storage_dtype_hint: None,
                first_arena_generation: 1,
            },
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("no direct paged-attention kernel"));
    }

    #[cfg(not(feature = "flash-attn"))]
    #[test]
    fn cuda_negotiation_fails_closed_without_paged_flash_attention() {
        let error = negotiate_kv_plan(
            &test_contract(),
            &KvBackendPlanRequest {
                model_instance: ModelInstanceId::new(9),
                backend: BackendKind::Cuda,
                device_ordinal: Some(0),
                capacity_pages: 10,
                page_tokens_hint: None,
                storage_dtype_hint: None,
                first_arena_generation: 1,
            },
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires the flash-attn feature"));
    }

    #[cfg(feature = "flash-attn")]
    #[test]
    fn cuda_flash_negotiation_selects_only_supported_geometry() {
        let plan = negotiate_kv_plan(
            &test_contract(),
            &KvBackendPlanRequest {
                model_instance: ModelInstanceId::new(9),
                backend: BackendKind::Cuda,
                device_ordinal: Some(0),
                capacity_pages: 10,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(KvStorageDType::F16),
                first_arena_generation: 1,
            },
        )
        .unwrap();
        assert_eq!(plan.groups[0].page_tokens, 32);
        assert_eq!(
            plan.groups[0].kernel,
            PagedAttentionKernel::CudaFlashAttention
        );
        assert_eq!(
            plan.groups[0].storage,
            KvStorageFormat::Dense {
                dtype: KvStorageDType::F16
            }
        );
    }
}

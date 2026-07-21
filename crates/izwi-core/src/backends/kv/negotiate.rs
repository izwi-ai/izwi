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
    if request.backend != BackendKind::Cpu {
        return Err(Error::InvalidInput(format!(
            "managed KV negotiation is not implemented for {:?}",
            request.backend
        )));
    }
    if request.capacity_pages == 0 || request.first_arena_generation == 0 {
        return Err(Error::InvalidInput(
            "KV capacity and first arena generation must be non-zero".into(),
        ));
    }

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
                let page_tokens = request
                    .page_tokens_hint
                    .filter(|value| spec.page_tokens.accepts(*value))
                    .unwrap_or(spec.page_tokens.preferred);
                let dtype = select_cpu_dtype(&spec.storage.dtypes, request.storage_dtype_hint)?;
                let dtype_bytes = dtype.dense_bytes().ok_or_else(|| {
                    Error::InvalidInput(format!("CPU cannot use dense {dtype:?} KV storage"))
                })?;
                let mut bytes_per_page = 0_u64;
                let mut layers = Vec::with_capacity(spec.layers.len());
                for (physical_layer, layer) in spec.layers.iter().enumerate() {
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
                    kernel: PagedAttentionKernel::PortableReference,
                    kind: ResolvedKvGroupKind::PagedAttention { layers },
                }
            }
            KvDomainSpec::ModelState(spec) => {
                let dtype = select_cpu_dtype(&spec.storage.dtypes, request.storage_dtype_hint)?;
                let dtype_bytes = dtype.dense_bytes().ok_or_else(|| {
                    Error::InvalidInput(format!("CPU cannot use dense {dtype:?} state storage"))
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
    fn accelerator_negotiation_fails_closed_until_runtime_exists() {
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
        assert!(error.to_string().contains("not implemented"));
    }
}

//! Backend policy and attestation for inference-state ABI v2.
//!
//! This module deliberately contains no tensor allocation. It resolves the
//! physical choices a backend is prepared to implement and attests only
//! operation sets that are complete today. KV arenas expose in-place writes,
//! ragged causal prefill/extend, and paged decode without materializing K/V or
//! expanding grouped-query heads. Resource negotiation remains fail-closed
//! until the selected allocator can attest exact envelopes.

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionPattern, CapacityStrategy, GroupResourceQuery, InferenceStateContract,
    NonPagedStateOperationQuery, NonPagedStateOperationRegistry, OperationAbi,
    PagedAttentionDomainSpec, PagedAttentionOperationQuery, PlacementPolicy, RegisteredOperationId,
    ResolvedCapacityDomain, ResolvedGroupResourceEnvelope, ResolvedPagedAttentionGroup,
    ResolvedPlacement, ResolvedStatePlan, ResolvedWorkspaceResourceEnvelope, StateDType,
    StateDomainSpec, StateLayerBinding, StateOperationRegistry, StateOperationSet,
    StatePhysicalLayout, StateResourceRegistry, StateStorageFormat, WorkspacePlacement,
    WorkspaceResourceQuery,
};

const PAGED_OPERATION_ABI: OperationAbi = OperationAbi::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateBackendPlanRequest {
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) page_tokens_hint: Option<u32>,
    pub(crate) storage_dtype_hint: Option<StateDType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedPagedPolicy {
    pub(crate) page_tokens: u32,
    pub(crate) layout: StatePhysicalLayout,
    pub(crate) storage: StateStorageFormat,
    pub(crate) placement: ResolvedPlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PagedOperationAvailability {
    pub(crate) write: bool,
    pub(crate) prefill: bool,
    pub(crate) decode: bool,
}

impl PagedOperationAvailability {
    pub(crate) const fn complete(self) -> bool {
        self.write && self.prefill && self.decode
    }

    fn missing(self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.write {
            missing.push("paged_slot_write");
        }
        if !self.prefill {
            missing.push("paged_prefill");
        }
        if !self.decode {
            missing.push("paged_decode");
        }
        missing
    }
}

/// One model-neutral registry for the exact selected backend/device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateBackendRegistry {
    backend: BackendKind,
    device_ordinal: Option<u32>,
}

impl StateBackendRegistry {
    pub(crate) fn new(backend: BackendKind, device_ordinal: Option<u32>) -> Result<Self> {
        match (backend, device_ordinal) {
            (BackendKind::Cpu, None) | (BackendKind::Metal | BackendKind::Cuda, Some(_)) => {
                Ok(Self {
                    backend,
                    device_ordinal,
                })
            }
            (BackendKind::Cpu, Some(_)) => Err(invalid(
                "CPU inference-state negotiation cannot use a device ordinal",
            )),
            (BackendKind::Metal | BackendKind::Cuda, None) => Err(invalid(
                "accelerator inference-state negotiation requires a device ordinal",
            )),
        }
    }

    pub(crate) const fn backend(self) -> BackendKind {
        self.backend
    }

    pub(crate) const fn device_ordinal(self) -> Option<u32> {
        self.device_ordinal
    }

    pub(crate) const fn paged_operation_availability(self) -> PagedOperationAvailability {
        match self.backend {
            BackendKind::Cpu => PagedOperationAvailability {
                write: true,
                prefill: true,
                decode: true,
            },
            BackendKind::Metal => PagedOperationAvailability {
                write: cfg!(feature = "metal"),
                prefill: cfg!(feature = "metal"),
                decode: cfg!(feature = "metal"),
            },
            BackendKind::Cuda => PagedOperationAvailability {
                write: cfg!(feature = "cuda"),
                prefill: cfg!(feature = "flash-attn"),
                decode: cfg!(feature = "flash-attn"),
            },
        }
    }

    const fn backend_compiled(self) -> bool {
        match self.backend {
            BackendKind::Cpu => true,
            BackendKind::Metal => cfg!(feature = "metal"),
            BackendKind::Cuda => cfg!(feature = "flash-attn"),
        }
    }

    fn validate_identity(self, backend: BackendKind, device_ordinal: Option<u32>) -> bool {
        self.backend == backend && self.device_ordinal == device_ordinal
    }

    fn require_compiled(self) -> Result<()> {
        if self.backend_compiled() {
            Ok(())
        } else {
            Err(invalid(format!(
                "inference-state backend {:?} is not compiled with its direct paged-attention runtime",
                self.backend
            )))
        }
    }

    fn supports_paged_policy(self, query: &PagedAttentionOperationQuery<'_>) -> bool {
        if !self.validate_identity(query.backend, query.device_ordinal)
            || !self.backend_compiled()
            || query.layout != StatePhysicalLayout::PageTokenHeadDim
            || query.operations != &paged_operation_set()
            || query.layers.len() != query.semantic.layers.len()
        {
            return false;
        }
        let expected = resolve_paged_policy(
            query.semantic,
            &StateBackendPlanRequest {
                backend: query.backend,
                device_ordinal: query.device_ordinal,
                page_tokens_hint: Some(query.page_tokens),
                storage_dtype_hint: Some(query.storage.dtype()),
            },
        );
        expected.is_ok_and(|expected| {
            expected.page_tokens == query.page_tokens
                && expected.layout == query.layout
                && expected.storage == query.storage
                && expected.placement == query.placement
        })
    }
}

impl NonPagedStateOperationRegistry for StateBackendRegistry {
    fn supports_non_paged(&self, query: &NonPagedStateOperationQuery<'_>) -> bool {
        // No backend-owned static/tensor/append/ring arena exists yet. Merely
        // being able to allocate a Candle tensor is not operation attestation.
        self.validate_identity(query.backend, query.device_ordinal) && false
    }
}

impl StateOperationRegistry for StateBackendRegistry {
    fn supports_paged_attention(&self, query: &PagedAttentionOperationQuery<'_>) -> bool {
        self.supports_paged_policy(query) && self.paged_operation_availability().complete()
    }
}

impl StateResourceRegistry for StateBackendRegistry {
    fn resolve_group_resources(
        &self,
        query: &GroupResourceQuery<'_>,
    ) -> Result<ResolvedGroupResourceEnvelope> {
        if !self.validate_identity(query.backend, query.device_ordinal) {
            return Err(invalid(
                "state resource query targets a different backend or device",
            ));
        }
        self.require_compiled()?;
        if !matches!(query.strategy, CapacityStrategy::Fixed { .. }) {
            return Err(invalid(
                "state backend currently supports only fully-backed fixed capacity",
            ));
        }
        let ResolvedCapacityDomain::Paged(plan) = query.resolved else {
            return Err(invalid(
                "non-paged inference-state backing is not implemented",
            ));
        };
        if plan.layout != StatePhysicalLayout::PageTokenHeadDim
            || !placement_is_allocatable(self.backend, plan.placement)
            || !dtype_is_supported(self.backend, plan.storage.dtype())
            || (self.backend == BackendKind::Cuda && plan.page_tokens % 32 != 0)
        {
            return Err(invalid(
                "resolved state group is not allocatable by the selected backend",
            ));
        }

        Err(invalid(
            "v2 physical state allocator cannot yet issue an exact resource envelope",
        ))
    }

    fn resolve_workspace_resources(
        &self,
        query: &WorkspaceResourceQuery<'_>,
    ) -> Result<ResolvedWorkspaceResourceEnvelope> {
        if !self.validate_identity(query.backend, query.device_ordinal) {
            return Err(invalid(
                "workspace resource query targets a different backend or device",
            ));
        }
        self.require_compiled()?;
        query.workspace.validate()?;
        if query.workspace.placement == WorkspacePlacement::Pinned {
            return Err(invalid(
                "pinned invocation workspace has no v2 backend allocator",
            ));
        }
        Err(invalid(
            "v2 workspace allocator cannot yet issue an exact resource envelope",
        ))
    }
}

/// Resolve a complete semantic retained-state contract for one backend.
///
/// Complete paged contracts resolve only when the selected build contains all
/// three physical operations. Unsupported domains and unavailable accelerator
/// kernels continue to fail closed.
pub(crate) fn negotiate_state_plan(
    contract: &InferenceStateContract,
    request: &StateBackendPlanRequest,
) -> Result<ResolvedStatePlan> {
    contract.validate()?;
    let registry = StateBackendRegistry::new(request.backend, request.device_ordinal)?;
    registry.require_compiled()?;

    let mut paged_attention = Vec::new();
    for domain in &contract.domains {
        match domain {
            StateDomainSpec::PagedAttention(spec) => {
                let group = contract
                    .groups
                    .iter()
                    .find(|group| group.domains.contains(&spec.header.id))
                    .ok_or_else(|| invalid("paged state domain has no consistency group"))?;
                let policy = resolve_paged_policy(spec, request)?;
                let layers = spec
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(physical_layer, layer)| {
                        Ok(StateLayerBinding {
                            model_layer: layer.model_layer,
                            physical_layer: u32::try_from(physical_layer)
                                .map_err(|_| invalid("paged state layer count exceeds u32"))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let bytes_per_page = paged_bytes_per_page(spec, policy)?;
                let resolved = ResolvedPagedAttentionGroup {
                    group: group.id,
                    domain: spec.header.id,
                    page_tokens: policy.page_tokens,
                    bytes_per_page,
                    layout: policy.layout,
                    storage: policy.storage,
                    placement: policy.placement,
                    layers,
                    operations: paged_operation_set(),
                };
                let query = PagedAttentionOperationQuery {
                    backend: request.backend,
                    device_ordinal: request.device_ordinal,
                    page_tokens: resolved.page_tokens,
                    layout: resolved.layout,
                    storage: resolved.storage,
                    placement: resolved.placement,
                    semantic: spec,
                    layers: &resolved.layers,
                    operations: &resolved.operations,
                };
                if !registry.supports_paged_attention(&query) {
                    return Err(invalid(format!(
                        "backend {:?} cannot attest a complete paged state operation set; missing {}",
                        request.backend,
                        registry.paged_operation_availability().missing().join(", ")
                    )));
                }
                paged_attention.push(resolved);
            }
            _ => {
                return Err(invalid(format!(
                    "backend {:?} has no physical implementation for non-paged state domain {}",
                    request.backend,
                    domain.id().get()
                )));
            }
        }
    }
    paged_attention.sort_unstable_by_key(|group| (group.group, group.domain));
    ResolvedStatePlan::build(
        request.backend,
        request.device_ordinal,
        contract,
        paged_attention,
        Vec::new(),
        &registry,
    )
}

pub(crate) fn resolve_paged_policy(
    semantic: &PagedAttentionDomainSpec,
    request: &StateBackendPlanRequest,
) -> Result<ResolvedPagedPolicy> {
    semantic.validate_for_backend_policy()?;
    StateBackendRegistry::new(request.backend, request.device_ordinal)?;
    let page_tokens = select_page_tokens(semantic, request)?;
    let dtype = select_dtype(semantic, request.storage_dtype_hint, request.backend)?;
    validate_layer_matrix(semantic, request.backend, page_tokens, dtype)?;
    let placement = match (semantic.header.placement, request.backend) {
        (PlacementPolicy::Host, BackendKind::Cpu) => ResolvedPlacement::Host,
        (PlacementPolicy::BackendLocal, _) | (PlacementPolicy::BackendLocalWithHostOffload, _) => {
            ResolvedPlacement::BackendLocal
        }
        (PlacementPolicy::Host, BackendKind::Metal | BackendKind::Cuda) => {
            return Err(invalid(
                "accelerator host-resident paged state has no direct attention operation",
            ));
        }
    };
    Ok(ResolvedPagedPolicy {
        page_tokens,
        layout: StatePhysicalLayout::PageTokenHeadDim,
        storage: StateStorageFormat::Dense { dtype },
        placement,
    })
}

fn paged_operation_set() -> StateOperationSet {
    StateOperationSet {
        write: RegisteredOperationId::new("paged_slot_write", PAGED_OPERATION_ABI),
        prefill: RegisteredOperationId::new("paged_prefill", PAGED_OPERATION_ABI),
        decode: RegisteredOperationId::new("paged_decode", PAGED_OPERATION_ABI),
    }
}

fn select_page_tokens(
    semantic: &PagedAttentionDomainSpec,
    request: &StateBackendPlanRequest,
) -> Result<u32> {
    let backend_accepts = |value: u32| {
        semantic.page_size.accepts(value)
            && (request.backend != BackendKind::Cuda || value % 32 == 0)
    };
    if let Some(value) = request
        .page_tokens_hint
        .filter(|value| backend_accepts(*value))
    {
        return Ok(value);
    }
    if backend_accepts(semantic.page_size.preferred_tokens) {
        return Ok(semantic.page_size.preferred_tokens);
    }
    let remainder = semantic.page_size.min_tokens % semantic.page_size.multiple_of;
    let mut value = if remainder == 0 {
        semantic.page_size.min_tokens
    } else {
        semantic
            .page_size
            .min_tokens
            .checked_add(semantic.page_size.multiple_of - remainder)
            .ok_or_else(|| invalid("page-size selection overflow"))?
    };
    while value <= semantic.page_size.max_tokens {
        if backend_accepts(value) {
            return Ok(value);
        }
        value = value
            .checked_add(semantic.page_size.multiple_of)
            .ok_or_else(|| invalid("page-size selection overflow"))?;
    }
    Err(invalid(format!(
        "backend {:?} found no supported page size",
        request.backend
    )))
}

fn select_dtype(
    semantic: &PagedAttentionDomainSpec,
    hint: Option<StateDType>,
    backend: BackendKind,
) -> Result<StateDType> {
    if let Some(dtype) = hint.filter(|dtype| {
        semantic.accepted_dtypes.contains(dtype) && dtype_is_supported(backend, *dtype)
    }) {
        return Ok(dtype);
    }
    semantic
        .accepted_dtypes
        .iter()
        .copied()
        .find(|dtype| dtype_is_supported(backend, *dtype))
        .ok_or_else(|| {
            invalid(format!(
                "backend {backend:?} found no supported state dtype"
            ))
        })
}

const fn dtype_is_supported(backend: BackendKind, dtype: StateDType) -> bool {
    match backend {
        BackendKind::Cpu => matches!(dtype, StateDType::F32 | StateDType::F16 | StateDType::Bf16),
        BackendKind::Metal => matches!(dtype, StateDType::F32 | StateDType::F16),
        BackendKind::Cuda => matches!(dtype, StateDType::F16 | StateDType::Bf16),
    }
}

fn validate_layer_matrix(
    semantic: &PagedAttentionDomainSpec,
    backend: BackendKind,
    _page_tokens: u32,
    _dtype: StateDType,
) -> Result<()> {
    for layer in &semantic.layers {
        match backend {
            BackendKind::Cpu => {}
            BackendKind::Metal if layer.key_head_dim > 512 || layer.value_head_dim > 512 => {
                return Err(invalid(format!(
                    "Metal paged attention head dimensions exceed 512 at layer {}",
                    layer.model_layer
                )));
            }
            BackendKind::Cuda
                if layer.key_head_dim != layer.value_head_dim
                    || layer.key_head_dim > 512
                    || layer.key_head_dim % 8 != 0 =>
            {
                return Err(invalid(format!(
                    "CUDA paged attention requires equal K/V dimensions, multiples of 8, at most 512 at layer {}",
                    layer.model_layer
                )));
            }
            BackendKind::Cuda
                if matches!(layer.pattern, AttentionPattern::SlidingWindow { .. }) =>
            {
                return Err(invalid(
                    "CUDA paged attention cannot consume the non-zero first-page offsets required by sliding windows",
                ));
            }
            BackendKind::Metal | BackendKind::Cuda => {}
        }
    }
    Ok(())
}

fn paged_bytes_per_page(
    semantic: &PagedAttentionDomainSpec,
    policy: ResolvedPagedPolicy,
) -> Result<u64> {
    let elements = semantic.layers.iter().try_fold(0_u64, |total, layer| {
        let per_token = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| invalid("paged state geometry overflow"))?;
        total
            .checked_add(
                u64::from(policy.page_tokens)
                    .checked_mul(per_token)
                    .ok_or_else(|| invalid("paged state geometry overflow"))?,
            )
            .ok_or_else(|| invalid("paged state geometry overflow"))
    })?;
    policy.storage.bytes_for_elements(elements)
}

const fn placement_is_allocatable(backend: BackendKind, placement: ResolvedPlacement) -> bool {
    matches!(
        (backend, placement),
        (
            BackendKind::Cpu,
            ResolvedPlacement::Host | ResolvedPlacement::BackendLocal
        ) | (
            BackendKind::Metal | BackendKind::Cuda,
            ResolvedPlacement::BackendLocal
        )
    )
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

/// Keep backend-policy validation on the semantic type without broadening its
/// public API. Full contract validation still runs before production resolve.
trait PagedPolicyValidation {
    fn validate_for_backend_policy(&self) -> Result<()>;
}

impl PagedPolicyValidation for PagedAttentionDomainSpec {
    fn validate_for_backend_policy(&self) -> Result<()> {
        if self.layers.is_empty() || self.accepted_dtypes.is_empty() {
            return Err(invalid("paged state policy requires layers and dtypes"));
        }
        if !self.page_size.accepts(self.page_size.preferred_tokens) {
            return Err(invalid("paged state policy has invalid page constraints"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::{
        test_contract, GroupResourceQuery, ResolvedCapacityDomain, StateResourceRegistry,
        WorkspaceContract, WorkspaceDimensionBound, WorkspacePlacement, WorkspaceResourceQuery,
    };

    fn request(backend: BackendKind) -> StateBackendPlanRequest {
        StateBackendPlanRequest {
            backend,
            device_ordinal: (backend != BackendKind::Cpu).then_some(0),
            page_tokens_hint: None,
            storage_dtype_hint: None,
        }
    }

    fn paged_spec(contract: &InferenceStateContract) -> &PagedAttentionDomainSpec {
        let StateDomainSpec::PagedAttention(spec) = &contract.domains[0] else {
            panic!("test contract must be paged")
        };
        spec
    }

    fn resolved_group(
        contract: &InferenceStateContract,
        backend: BackendKind,
    ) -> ResolvedPagedAttentionGroup {
        let spec = paged_spec(contract);
        let policy = resolve_paged_policy(spec, &request(backend)).unwrap();
        ResolvedPagedAttentionGroup {
            group: contract.groups[0].id,
            domain: spec.header.id,
            page_tokens: policy.page_tokens,
            bytes_per_page: paged_bytes_per_page(spec, policy).unwrap(),
            layout: policy.layout,
            storage: policy.storage,
            placement: policy.placement,
            layers: spec
                .layers
                .iter()
                .enumerate()
                .map(|(physical, layer)| StateLayerBinding {
                    model_layer: layer.model_layer,
                    physical_layer: physical as u32,
                })
                .collect(),
            operations: paged_operation_set(),
        }
    }

    #[test]
    fn cpu_policy_attests_the_complete_direct_paged_operation_set() {
        let contract = test_contract();
        let spec = paged_spec(&contract);
        let policy = resolve_paged_policy(spec, &request(BackendKind::Cpu)).unwrap();
        assert_eq!(policy.page_tokens, 16);
        assert_eq!(policy.storage.dtype(), StateDType::F16);

        let registry = StateBackendRegistry::new(BackendKind::Cpu, None).unwrap();
        assert_eq!(
            registry.paged_operation_availability(),
            PagedOperationAvailability {
                write: true,
                prefill: true,
                decode: true,
            }
        );
        let plan = negotiate_state_plan(&contract, &request(BackendKind::Cpu)).unwrap();
        assert_eq!(plan.backend, BackendKind::Cpu);
        assert_eq!(plan.paged_attention.len(), 1);
    }

    #[test]
    fn backend_policy_matrix_is_shape_dtype_page_and_placement_exact() {
        let contract = test_contract();
        let spec = paged_spec(&contract);

        let metal = resolve_paged_policy(spec, &request(BackendKind::Metal)).unwrap();
        assert_eq!(metal.storage.dtype(), StateDType::F16);
        assert_eq!(metal.page_tokens, 16);

        let cuda = resolve_paged_policy(spec, &request(BackendKind::Cuda)).unwrap();
        assert_eq!(cuda.storage.dtype(), StateDType::F16);
        assert_eq!(cuda.page_tokens, 32);

        let mut incompatible = spec.clone();
        incompatible.layers[0].value_head_dim = 72;
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Cuda)).is_err());
        incompatible = spec.clone();
        incompatible.layers[0].pattern = AttentionPattern::SlidingWindow { window_tokens: 128 };
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Cuda)).is_err());
        incompatible = spec.clone();
        incompatible.header.placement = PlacementPolicy::Host;
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Metal)).is_err());
        incompatible = spec.clone();
        incompatible.accepted_dtypes = vec![StateDType::Bf16];
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Metal)).is_err());
    }

    #[test]
    fn accelerator_operation_registry_tracks_compiled_direct_paged_support() {
        let metal = StateBackendRegistry::new(BackendKind::Metal, Some(0)).unwrap();
        let metal_operations = metal.paged_operation_availability();
        assert_eq!(metal_operations.write, cfg!(feature = "metal"));
        assert_eq!(metal_operations.decode, cfg!(feature = "metal"));
        assert_eq!(metal_operations.prefill, cfg!(feature = "metal"));
        assert_eq!(metal_operations.complete(), cfg!(feature = "metal"));

        let cuda = StateBackendRegistry::new(BackendKind::Cuda, Some(0)).unwrap();
        let cuda_operations = cuda.paged_operation_availability();
        assert_eq!(cuda_operations.write, cfg!(feature = "cuda"));
        assert_eq!(cuda_operations.decode, cfg!(feature = "flash-attn"));
        assert_eq!(cuda_operations.prefill, cfg!(feature = "flash-attn"));
        assert_eq!(cuda_operations.complete(), cfg!(feature = "flash-attn"));
    }

    #[test]
    fn resource_registry_rejects_guesses_without_an_exact_allocator_envelope() {
        let contract = test_contract();
        let group = resolved_group(&contract, BackendKind::Cpu);
        let registry = StateBackendRegistry::new(BackendKind::Cpu, None).unwrap();
        let fixed = registry
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                resolved: ResolvedCapacityDomain::Paged(&group),
                strategy: CapacityStrategy::Fixed { blocks: 8 },
            })
            .unwrap_err();
        assert!(fixed
            .to_string()
            .contains("cannot yet issue an exact resource envelope"));

        assert!(registry
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                resolved: ResolvedCapacityDomain::Paged(&group),
                strategy: CapacityStrategy::Reserved {
                    initial_blocks: 4,
                    max_blocks: 8,
                },
            })
            .is_err());
    }

    #[test]
    fn workspace_resources_reject_missing_allocator_pinned_and_wrong_identity() {
        let registry = StateBackendRegistry::new(BackendKind::Cpu, None).unwrap();
        let workspace = WorkspaceContract {
            fixed_bytes: 64,
            dimensions: vec![WorkspaceDimensionBound {
                axis: crate::kv::v2::WorkspaceAxis::InputTokens,
                max_units: 4,
            }],
            terms: vec![],
            placement: WorkspacePlacement::Host,
            concurrency_slots: 2,
        };
        let envelope = registry
            .resolve_workspace_resources(&WorkspaceResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                workspace: &workspace,
            })
            .unwrap_err();
        assert!(envelope
            .to_string()
            .contains("cannot yet issue an exact resource envelope"));
        assert!(registry
            .resolve_workspace_resources(&WorkspaceResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: Some(0),
                workspace: &workspace,
            })
            .is_err());

        let mut pinned = workspace;
        pinned.placement = WorkspacePlacement::Pinned;
        assert!(registry
            .resolve_workspace_resources(&WorkspaceResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                workspace: &pinned,
            })
            .is_err());
    }
}

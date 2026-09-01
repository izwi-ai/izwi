//! Backend policy and attestation for inference-state ABI v2.
//!
//! This module resolves the physical choices a backend is prepared to
//! implement and attests only operation sets that are complete today. KV
//! arenas expose in-place writes,
//! ragged causal prefill/extend, and paged decode without materializing K/V or
//! expanding grouped-query heads. Fixed logical backing and unpinned workspace
//! envelopes are exact; unsupported growth, pinned memory, domains, and builds
//! remain fail-closed.

mod invocation;
mod static_attention;
mod tensor;

#[allow(unused_imports)]
pub(crate) use invocation::{
    InvocationRingDepthwiseConvTransaction, InvocationTensorArena,
    InvocationTensorBulkComponentValue, InvocationTensorChronologicalSegment,
    InvocationTensorComponentSlice, InvocationTensorComponentValue, InvocationTensorDomainKind,
    InvocationTensorSnapshot, InvocationTensorStepValues, InvocationTensorUpdateV2,
};
#[allow(unused_imports)]
pub(crate) use static_attention::{
    InvocationStaticAttentionArena, StaticAttentionLayerValue, StaticAttentionMetadata,
    StaticAttentionRaggedRow,
};
#[allow(unused_imports)]
pub(crate) use tensor::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, StateDomainSnapshot,
    TensorStateArena, TensorStateBatchCompletion, TensorStateCapacity, TensorStateOccupancy,
    TensorStateSelection,
};

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    align_bytes, AppendStateOperationSet, AttentionPattern, CapacityStrategy, GroupResourceQuery,
    InferenceStateContract, NonPagedStateOperationQuery, NonPagedStateOperationRegistry,
    OperationAbi, PagedAttentionDomainSpec, PagedAttentionOperationQuery,
    PagedOperationImplementation, PagedOperationImplementationSet, PlacementPolicy,
    RegisteredOperationId, ResolvedAppendStatePlan, ResolvedCapacityDomain,
    ResolvedGroupResourceEnvelope, ResolvedNonPagedDomainPlan, ResolvedPagedAttentionGroup,
    ResolvedPlacement, ResolvedRingStatePlan, ResolvedStatePlan, ResolvedStaticAttentionPlan,
    ResolvedStaticTensorPlan, ResolvedTensorComponent, ResolvedTensorStatePlan,
    ResolvedWorkspaceResourceEnvelope, RingStateOperationSet, StateDType, StateDomainSpec,
    StateLayerBinding, StateOperationRegistry, StateOperationSet, StatePhysicalLayout,
    StateResourceRegistry, StateStorageFormat, StaticAttentionOperationSet,
    StaticTensorOperationSet, TensorComponentSpec, TensorPhysicalLayout, TensorStateOperationSet,
    WorkspacePlacement, WorkspaceResourceQuery,
};
use crate::runtime::rollout::KvProviderRollout;

const PAGED_OPERATION_ABI: OperationAbi = OperationAbi::new(1);
const NON_PAGED_OPERATION_ABI: OperationAbi = OperationAbi::new(1);

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

/// Backend support for one stable paged-attention operation ABI.
///
/// Availability and implementation class are deliberately one capability:
/// negotiation cannot accidentally claim an optimized path merely because a
/// function with the right registry name exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PagedOperationCapability {
    Unavailable,
    Portable,
    Optimized,
}

impl PagedOperationCapability {
    const fn implementation(self) -> Option<PagedOperationImplementation> {
        match self {
            Self::Unavailable => None,
            Self::Portable => Some(PagedOperationImplementation::Portable),
            Self::Optimized => Some(PagedOperationImplementation::Optimized),
        }
    }
}

/// Backend capability matrix for the complete paged-attention operation set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PagedOperationCapabilities {
    pub(crate) write: PagedOperationCapability,
    pub(crate) prefill: PagedOperationCapability,
    pub(crate) decode: PagedOperationCapability,
}

impl PagedOperationCapabilities {
    const fn unavailable() -> Self {
        Self {
            write: PagedOperationCapability::Unavailable,
            prefill: PagedOperationCapability::Unavailable,
            decode: PagedOperationCapability::Unavailable,
        }
    }

    const fn portable() -> Self {
        Self {
            write: PagedOperationCapability::Portable,
            prefill: PagedOperationCapability::Portable,
            decode: PagedOperationCapability::Portable,
        }
    }

    fn implementation_plan(self) -> Option<PagedOperationImplementationSet> {
        Some(PagedOperationImplementationSet {
            write: self.write.implementation()?,
            prefill: self.prefill.implementation()?,
            decode: self.decode.implementation()?,
        })
    }
}

/// One model-neutral registry for the exact selected backend/device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateBackendRegistry {
    backend: BackendKind,
    device_ordinal: Option<u32>,
    rollout: KvProviderRollout,
}

impl StateBackendRegistry {
    pub(crate) fn new(backend: BackendKind, device_ordinal: Option<u32>) -> Result<Self> {
        let rollout = KvProviderRollout::from_process_env()?;
        match (backend, device_ordinal) {
            (BackendKind::Cpu, None) | (BackendKind::Metal | BackendKind::Cuda, Some(_)) => {
                Ok(Self {
                    backend,
                    device_ordinal,
                    rollout,
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
                prefill: cfg!(feature = "cuda"),
                decode: cfg!(feature = "cuda"),
            },
        }
    }

    pub(crate) fn paged_operation_capabilities(
        self,
        semantic: &PagedAttentionDomainSpec,
        policy: ResolvedPagedPolicy,
    ) -> PagedOperationCapabilities {
        resolve_paged_operation_capabilities(
            self.backend,
            self.paged_backend_compiled(),
            cfg!(feature = "flash-attn"),
            self.rollout.optimized_provider_enabled(),
            semantic,
            policy,
        )
    }

    const fn paged_backend_compiled(self) -> bool {
        match self.backend {
            BackendKind::Cpu => true,
            BackendKind::Metal => cfg!(feature = "metal"),
            BackendKind::Cuda => cfg!(feature = "cuda"),
        }
    }

    fn validate_identity(self, backend: BackendKind, device_ordinal: Option<u32>) -> bool {
        self.backend == backend && self.device_ordinal == device_ordinal
    }

    fn require_compiled(self) -> Result<()> {
        if self.paged_backend_compiled() {
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
            || !self.paged_backend_compiled()
            || query.layout != StatePhysicalLayout::PageTokenHeadDim
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
            let Some(implementations) = self
                .paged_operation_capabilities(query.semantic, expected)
                .implementation_plan()
            else {
                return false;
            };
            expected.page_tokens == query.page_tokens
                && expected.layout == query.layout
                && expected.storage == query.storage
                && expected.placement == query.placement
                && query.operations == &paged_operation_set(implementations)
        })
    }
}

impl NonPagedStateOperationRegistry for StateBackendRegistry {
    fn supports_non_paged(&self, query: &NonPagedStateOperationQuery<'_>) -> bool {
        self.validate_identity(query.backend, query.device_ordinal)
            && non_paged_backend_compiled(self.backend)
            && non_paged_plan_is_supported(query.resolved, query.semantic, self.backend)
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
        if matches!(query.resolved, ResolvedCapacityDomain::Paged(_)) {
            self.require_compiled()?;
        } else if !non_paged_backend_compiled(self.backend) {
            return Err(invalid(format!(
                "inference-state backend {:?} is not compiled for tensor state",
                self.backend
            )));
        }
        let strategy_supported = matches!(
            (query.resolved, query.strategy),
            (
                ResolvedCapacityDomain::Paged(_),
                CapacityStrategy::Fixed { .. }
            ) | (
                ResolvedCapacityDomain::Paged(_),
                CapacityStrategy::AdmissionGrowable { .. }
            ) | (
                ResolvedCapacityDomain::NonPaged(_),
                CapacityStrategy::BoundedLazy { .. }
            )
        );
        if !strategy_supported {
            return Err(invalid(
                "state backend does not support the requested capacity materialization strategy",
            ));
        }
        match query.resolved {
            ResolvedCapacityDomain::Paged(plan)
                if plan.layout == StatePhysicalLayout::PageTokenHeadDim
                    && placement_is_allocatable(self.backend, plan.placement)
                    && dtype_is_supported(self.backend, plan.storage.dtype()) => {}
            ResolvedCapacityDomain::NonPaged(plan)
                if placement_is_allocatable(self.backend, plan.placement())
                    && non_paged_resolved_dtypes_supported(plan, self.backend) => {}
            _ => {
                return Err(invalid(
                    "resolved state group is not allocatable by the selected backend",
                ));
            }
        }

        // Fixed arenas allocate one immutable logical backing. CUDA admission-
        // grown arenas replace that backing only at a maintenance barrier and
        // publish a cumulative receipt for each whole growth-quantum range.
        // Driver residency remains observation-only.
        Ok(ResolvedGroupResourceEnvelope {
            allocator_alignment_bytes: 1,
            allocator_overhead_per_allocation: 0,
            max_backing_allocations: query.strategy.minimum_backing_allocations(),
            reservation_metadata_bytes: 0,
            metadata_bytes_per_block: 0,
            pinned_bytes_per_block: 0,
        })
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
        Ok(ResolvedWorkspaceResourceEnvelope {
            allocator_alignment_bytes: 1,
            allocator_overhead_per_slot: 0,
            max_concurrency_slots: query.workspace.concurrency_slots,
        })
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
    if contract
        .domains
        .iter()
        .any(|domain| matches!(domain, StateDomainSpec::PagedAttention(_)))
    {
        registry.require_compiled()?;
    } else if !non_paged_backend_compiled(request.backend) {
        return Err(invalid(format!(
            "inference-state backend {:?} is not compiled for tensor state",
            request.backend
        )));
    }

    let mut paged_attention = Vec::new();
    let mut non_paged = Vec::new();
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
                let implementations = registry
                    .paged_operation_capabilities(spec, policy)
                    .implementation_plan()
                    .ok_or_else(|| {
                        invalid(format!(
                            "backend {:?} cannot attest a complete paged state operation set; missing {}",
                            request.backend,
                            registry.paged_operation_availability().missing().join(", ")
                        ))
                    })?;
                let resolved = ResolvedPagedAttentionGroup {
                    group: group.id,
                    domain: spec.header.id,
                    page_tokens: policy.page_tokens,
                    bytes_per_page,
                    layout: policy.layout,
                    storage: policy.storage,
                    placement: policy.placement,
                    layers,
                    operations: paged_operation_set(implementations),
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
            _ => non_paged.push(resolve_non_paged_domain(
                contract, domain, request, &registry,
            )?),
        }
    }
    paged_attention.sort_unstable_by_key(|group| (group.group, group.domain));
    non_paged.sort_unstable_by_key(|plan| (plan.group(), plan.domain()));
    ResolvedStatePlan::build(
        request.backend,
        request.device_ordinal,
        contract,
        paged_attention,
        non_paged,
        &registry,
    )
}

fn resolve_non_paged_domain(
    contract: &InferenceStateContract,
    domain: &StateDomainSpec,
    request: &StateBackendPlanRequest,
    registry: &StateBackendRegistry,
) -> Result<ResolvedNonPagedDomainPlan> {
    let group = contract
        .groups
        .iter()
        .find(|group| group.domains.contains(&domain.id()))
        .ok_or_else(|| invalid("non-paged state domain has no consistency group"))?;
    let placement = resolve_non_paged_placement(domain.header().placement, request.backend)?;
    let resolved = match domain {
        StateDomainSpec::StaticAttention(spec) => {
            let storage = dense_storage(&spec.accepted_dtypes, request.backend)?;
            let layers = spec
                .layers
                .iter()
                .enumerate()
                .map(|(physical_layer, layer)| {
                    Ok(StateLayerBinding {
                        model_layer: layer.model_layer,
                        physical_layer: u32::try_from(physical_layer)
                            .map_err(|_| invalid("static-attention layer count exceeds u32"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let elements = spec.layers.iter().try_fold(0_u64, |total, layer| {
                let per_token = u64::from(layer.kv_heads)
                    .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
                    .ok_or_else(|| invalid("static-attention geometry overflow"))?;
                total
                    .checked_add(
                        per_token
                            .checked_mul(spec.max_memory_tokens)
                            .ok_or_else(|| invalid("static-attention geometry overflow"))?,
                    )
                    .ok_or_else(|| invalid("static-attention geometry overflow"))
            })?;
            ResolvedNonPagedDomainPlan::StaticAttention(ResolvedStaticAttentionPlan {
                group: group.id,
                domain: spec.header.id,
                placement,
                layers,
                storage,
                layout: TensorPhysicalLayout::ContiguousRowMajor,
                alignment_bytes: 1,
                maximum_bytes: storage.bytes_for_elements(elements)?,
                operations: static_attention_operations(),
            })
        }
        StateDomainSpec::StaticTensor(spec) => {
            let components = resolve_components(&spec.components, request.backend)?;
            ResolvedNonPagedDomainPlan::StaticTensor(ResolvedStaticTensorPlan {
                group: group.id,
                domain: spec.header.id,
                placement,
                maximum_bytes: component_bytes(&components)?,
                components,
                operations: static_tensor_operations(),
            })
        }
        StateDomainSpec::Tensor(spec) => {
            let components = resolve_components(&spec.components, request.backend)?;
            ResolvedNonPagedDomainPlan::Tensor(ResolvedTensorStatePlan {
                group: group.id,
                domain: spec.header.id,
                placement,
                maximum_bytes: component_bytes(&components)?,
                components,
                operations: tensor_operations(),
            })
        }
        StateDomainSpec::Append(spec) => {
            let components_per_step =
                resolve_components(&spec.components_per_step, request.backend)?;
            let maximum_bytes = component_bytes(&components_per_step)?
                .checked_mul(spec.max_steps)
                .ok_or_else(|| invalid("append-state byte bound overflow"))?;
            ResolvedNonPagedDomainPlan::Append(ResolvedAppendStatePlan {
                group: group.id,
                domain: spec.header.id,
                placement,
                components_per_step,
                maximum_bytes,
                operations: append_operations(),
            })
        }
        StateDomainSpec::Ring(spec) => {
            let components_per_step =
                resolve_components(&spec.components_per_step, request.backend)?;
            let maximum_bytes = component_bytes(&components_per_step)?
                .checked_mul(spec.capacity_steps)
                .ok_or_else(|| invalid("ring-state byte bound overflow"))?;
            ResolvedNonPagedDomainPlan::Ring(ResolvedRingStatePlan {
                group: group.id,
                domain: spec.header.id,
                placement,
                components_per_step,
                maximum_bytes,
                operations: ring_operations(),
            })
        }
        StateDomainSpec::PagedAttention(_) => {
            return Err(invalid("paged state was routed to the non-paged resolver"));
        }
    };
    resolved.validate_against(domain, request.backend, request.device_ordinal, registry)?;
    Ok(resolved)
}

fn resolve_components(
    components: &[TensorComponentSpec],
    backend: BackendKind,
) -> Result<Vec<ResolvedTensorComponent>> {
    components
        .iter()
        .map(|component| {
            let storage = dense_storage(&component.accepted_dtypes, backend)?;
            Ok(ResolvedTensorComponent {
                component: component.id,
                layout: TensorPhysicalLayout::ContiguousRowMajor,
                storage,
                alignment_bytes: 1,
                maximum_bytes: align_bytes(
                    storage.bytes_for_elements(component.shape.maximum_elements()?)?,
                    1,
                )?,
            })
        })
        .collect()
}

fn component_bytes(components: &[ResolvedTensorComponent]) -> Result<u64> {
    components.iter().try_fold(0_u64, |total, component| {
        total
            .checked_add(component.maximum_bytes)
            .ok_or_else(|| invalid("non-paged state byte bound overflow"))
    })
}

fn dense_storage(accepted: &[StateDType], backend: BackendKind) -> Result<StateStorageFormat> {
    accepted
        .iter()
        .copied()
        .find(|dtype| tensor_dtype_is_supported(backend, *dtype))
        .map(|dtype| StateStorageFormat::Dense { dtype })
        .ok_or_else(|| {
            invalid(format!(
                "backend {backend:?} found no supported tensor-state dtype"
            ))
        })
}

fn resolve_non_paged_placement(
    policy: PlacementPolicy,
    backend: BackendKind,
) -> Result<ResolvedPlacement> {
    match (policy, backend) {
        (PlacementPolicy::Host, BackendKind::Cpu) => Ok(ResolvedPlacement::Host),
        (PlacementPolicy::BackendLocal, _) | (PlacementPolicy::BackendLocalWithHostOffload, _) => {
            Ok(ResolvedPlacement::BackendLocal)
        }
        (PlacementPolicy::Host, BackendKind::Metal | BackendKind::Cuda) => Err(invalid(
            "accelerator tensor state must be backend-local for direct operations",
        )),
    }
}

const fn non_paged_backend_compiled(backend: BackendKind) -> bool {
    match backend {
        BackendKind::Cpu => true,
        BackendKind::Metal => cfg!(feature = "metal"),
        BackendKind::Cuda => cfg!(feature = "cuda"),
    }
}

const fn tensor_dtype_is_supported(backend: BackendKind, dtype: StateDType) -> bool {
    match backend {
        BackendKind::Cpu => matches!(
            dtype,
            StateDType::F32 | StateDType::F16 | StateDType::Bf16 | StateDType::I64
        ),
        BackendKind::Metal => matches!(dtype, StateDType::F32 | StateDType::F16 | StateDType::I64),
        BackendKind::Cuda => matches!(
            dtype,
            StateDType::F32 | StateDType::F16 | StateDType::Bf16 | StateDType::I64
        ),
    }
}

fn non_paged_resolved_dtypes_supported(
    plan: &ResolvedNonPagedDomainPlan,
    backend: BackendKind,
) -> bool {
    let components = match plan {
        ResolvedNonPagedDomainPlan::StaticTensor(plan) => plan.components.as_slice(),
        ResolvedNonPagedDomainPlan::Tensor(plan) => plan.components.as_slice(),
        ResolvedNonPagedDomainPlan::Append(plan) => plan.components_per_step.as_slice(),
        ResolvedNonPagedDomainPlan::Ring(plan) => plan.components_per_step.as_slice(),
        ResolvedNonPagedDomainPlan::StaticAttention(plan) => {
            return tensor_dtype_is_supported(backend, plan.storage.dtype());
        }
    };
    components
        .iter()
        .all(|component| tensor_dtype_is_supported(backend, component.storage.dtype()))
}

fn non_paged_plan_is_supported(
    resolved: &ResolvedNonPagedDomainPlan,
    semantic: &StateDomainSpec,
    backend: BackendKind,
) -> bool {
    if !placement_is_allocatable(backend, resolved.placement())
        || !non_paged_resolved_dtypes_supported(resolved, backend)
    {
        return false;
    }
    match (resolved, semantic) {
        (
            ResolvedNonPagedDomainPlan::StaticAttention(plan),
            StateDomainSpec::StaticAttention(spec),
        ) => static_attention::static_plan_is_supported(plan, spec, backend),
        (ResolvedNonPagedDomainPlan::StaticTensor(plan), StateDomainSpec::StaticTensor(_)) => {
            plan.operations == static_tensor_operations()
        }
        (ResolvedNonPagedDomainPlan::Tensor(plan), StateDomainSpec::Tensor(_)) => {
            plan.operations == tensor_operations()
        }
        (ResolvedNonPagedDomainPlan::Append(plan), StateDomainSpec::Append(_)) => {
            plan.operations == append_operations()
        }
        (ResolvedNonPagedDomainPlan::Ring(plan), StateDomainSpec::Ring(_)) => {
            plan.operations == ring_operations()
        }
        _ => false,
    }
}

fn operation(name: &'static str) -> RegisteredOperationId {
    RegisteredOperationId::new(name, NON_PAGED_OPERATION_ABI)
}

fn static_attention_operations() -> StaticAttentionOperationSet {
    StaticAttentionOperationSet {
        install: operation("static_attention_install"),
        attend: operation("static_attention_attend"),
    }
}

fn static_tensor_operations() -> StaticTensorOperationSet {
    StaticTensorOperationSet {
        install: operation("static_tensor_install"),
        read: operation("static_tensor_read"),
    }
}

fn tensor_operations() -> TensorStateOperationSet {
    TensorStateOperationSet {
        initialize: operation("tensor_state_initialize"),
        read: operation("tensor_state_read"),
        stage_replace: operation("tensor_state_stage_replace"),
        reset: operation("tensor_state_reset"),
    }
}

fn append_operations() -> AppendStateOperationSet {
    AppendStateOperationSet {
        initialize: operation("append_state_initialize"),
        read: operation("append_state_read"),
        append: operation("append_state_append"),
        reset: operation("append_state_reset"),
    }
}

fn ring_operations() -> RingStateOperationSet {
    RingStateOperationSet {
        initialize: operation("ring_state_initialize"),
        read: operation("ring_state_read"),
        advance: operation("ring_state_advance"),
        reset: operation("ring_state_reset"),
    }
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

fn resolve_paged_operation_capabilities(
    backend: BackendKind,
    backend_compiled: bool,
    cuda_flash_attention_compiled: bool,
    optimized_provider_enabled: bool,
    semantic: &PagedAttentionDomainSpec,
    policy: ResolvedPagedPolicy,
) -> PagedOperationCapabilities {
    if !backend_compiled {
        return PagedOperationCapabilities::unavailable();
    }
    match backend {
        BackendKind::Cpu => PagedOperationCapabilities::portable(),
        BackendKind::Metal => PagedOperationCapabilities {
            write: PagedOperationCapability::Portable,
            // The native Metal kernels are the certified direct provider.
            // Keep them Portable until a distinct optimized implementation
            // has its own numerical and performance certification cell.
            prefill: PagedOperationCapability::Portable,
            decode: PagedOperationCapability::Portable,
        },
        BackendKind::Cuda => {
            let flash_compatible = optimized_provider_enabled
                && cuda_flash_attention_compiled
                && policy.page_tokens != 0
                && policy.page_tokens.is_multiple_of(32)
                && matches!(policy.storage.dtype(), StateDType::F16 | StateDType::Bf16)
                && semantic.layers.iter().all(|layer| {
                    layer.key_head_dim == layer.value_head_dim
                        && layer.key_head_dim != 0
                        && layer.key_head_dim <= 512
                        && layer.key_head_dim % 8 == 0
                        && matches!(layer.pattern, AttentionPattern::Full)
                });
            let attention = if flash_compatible {
                PagedOperationCapability::Optimized
            } else {
                PagedOperationCapability::Portable
            };
            PagedOperationCapabilities {
                write: PagedOperationCapability::Portable,
                prefill: attention,
                decode: attention,
            }
        }
    }
}

fn paged_operation_set(implementations: PagedOperationImplementationSet) -> StateOperationSet {
    StateOperationSet {
        write: RegisteredOperationId::new("paged_slot_write", PAGED_OPERATION_ABI),
        prefill: RegisteredOperationId::new("paged_prefill", PAGED_OPERATION_ABI),
        decode: RegisteredOperationId::new("paged_decode", PAGED_OPERATION_ABI),
        implementations,
    }
}

fn select_page_tokens(
    semantic: &PagedAttentionDomainSpec,
    request: &StateBackendPlanRequest,
) -> Result<u32> {
    let backend_accepts = |value: u32| semantic.page_size.accepts(value);
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
        BackendKind::Cuda => {
            matches!(dtype, StateDType::F32 | StateDType::F16 | StateDType::Bf16)
        }
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
                if layer.key_head_dim != layer.value_head_dim || layer.key_head_dim > 512 =>
            {
                return Err(invalid(format!(
                    "CUDA paged attention requires equal K/V dimensions at most 512 at layer {}",
                    layer.model_layer
                )));
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
    use crate::engine::ModelInstanceId;
    use crate::kv::v2::{
        tensor_test_contract, tensor_test_plan, test_contract, AttentionPattern,
        GroupCapacityRequest, GroupResourceQuery, ResolvedCapacityDomain, StateResourceRegistry,
        StateResourceVector, StateRuntimeAllocationPlan, TensorTestOperationRegistry,
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
        let registry = StateBackendRegistry::new(backend, request(backend).device_ordinal).unwrap();
        let implementations = registry
            .paged_operation_capabilities(spec, policy)
            .implementation_plan()
            .expect("resolved test backend must have complete paged operations");
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
            operations: paged_operation_set(implementations),
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
        assert_eq!(
            registry.paged_operation_capabilities(spec, policy),
            PagedOperationCapabilities {
                write: PagedOperationCapability::Portable,
                prefill: PagedOperationCapability::Portable,
                decode: PagedOperationCapability::Portable,
            }
        );
        let plan = negotiate_state_plan(&contract, &request(BackendKind::Cpu)).unwrap();
        assert_eq!(plan.backend, BackendKind::Cpu);
        assert_eq!(plan.paged_attention.len(), 1);
        assert_eq!(
            plan.paged_attention[0].operations.implementations,
            PagedOperationImplementationSet {
                write: PagedOperationImplementation::Portable,
                prefill: PagedOperationImplementation::Portable,
                decode: PagedOperationImplementation::Portable,
            }
        );

        let mut unattested = plan.paged_attention[0].clone();
        unattested.operations.implementations.decode = PagedOperationImplementation::Optimized;
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![unattested],
            vec![],
            &registry,
        )
        .is_err());
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
        assert_eq!(cuda.page_tokens, 16);

        let mut incompatible = spec.clone();
        incompatible.layers[0].value_head_dim = 72;
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Cuda)).is_err());
        incompatible = spec.clone();
        incompatible.layers[0].pattern = AttentionPattern::SlidingWindow { window_tokens: 128 };
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Cuda)).is_ok());
        incompatible = spec.clone();
        incompatible.header.placement = PlacementPolicy::Host;
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Metal)).is_err());
        incompatible = spec.clone();
        incompatible.accepted_dtypes = vec![StateDType::Bf16];
        assert!(resolve_paged_policy(&incompatible, &request(BackendKind::Metal)).is_err());
    }

    #[test]
    fn accelerator_operation_registry_tracks_compiled_direct_paged_support() {
        let contract = test_contract();
        let spec = paged_spec(&contract);

        let metal = StateBackendRegistry::new(BackendKind::Metal, Some(0)).unwrap();
        let metal_policy = resolve_paged_policy(spec, &request(BackendKind::Metal)).unwrap();
        let metal_operations = metal.paged_operation_availability();
        assert_eq!(metal_operations.write, cfg!(feature = "metal"));
        assert_eq!(metal_operations.decode, cfg!(feature = "metal"));
        assert_eq!(metal_operations.prefill, cfg!(feature = "metal"));
        assert_eq!(metal_operations.complete(), cfg!(feature = "metal"));
        let metal_capabilities = metal.paged_operation_capabilities(spec, metal_policy);
        assert_eq!(
            metal_capabilities.write,
            if cfg!(feature = "metal") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );
        assert_eq!(
            metal_capabilities.prefill,
            if cfg!(feature = "metal") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );
        assert_eq!(
            metal_capabilities.decode,
            if cfg!(feature = "metal") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );

        let cuda = StateBackendRegistry::new(BackendKind::Cuda, Some(0)).unwrap();
        let cuda_policy = resolve_paged_policy(spec, &request(BackendKind::Cuda)).unwrap();
        let cuda_operations = cuda.paged_operation_availability();
        assert_eq!(cuda_operations.write, cfg!(feature = "cuda"));
        assert_eq!(cuda_operations.decode, cfg!(feature = "cuda"));
        assert_eq!(cuda_operations.prefill, cfg!(feature = "cuda"));
        assert_eq!(cuda_operations.complete(), cfg!(feature = "cuda"));
        let cuda_capabilities = cuda.paged_operation_capabilities(spec, cuda_policy);
        assert_eq!(
            cuda_capabilities.write,
            if cfg!(feature = "cuda") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );
        assert_eq!(
            cuda_capabilities.prefill,
            if cfg!(feature = "cuda") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );
        assert_eq!(
            cuda_capabilities.decode,
            if cfg!(feature = "cuda") {
                PagedOperationCapability::Portable
            } else {
                PagedOperationCapability::Unavailable
            }
        );
    }

    #[test]
    fn paged_attention_optimization_classification_is_shape_and_semantics_exact() {
        let contract = test_contract();
        let spec = paged_spec(&contract);
        let mut cuda_request = request(BackendKind::Cuda);
        cuda_request.page_tokens_hint = Some(32);
        cuda_request.storage_dtype_hint = Some(StateDType::F16);
        let page_32_half = resolve_paged_policy(spec, &cuda_request).unwrap();
        assert!(spec
            .layers
            .iter()
            .all(|layer| layer.key_head_dim == layer.value_head_dim));

        let optimized = resolve_paged_operation_capabilities(
            BackendKind::Cuda,
            true,
            true,
            true,
            spec,
            page_32_half,
        );
        assert_eq!(optimized.write, PagedOperationCapability::Portable);
        assert_eq!(optimized.prefill, PagedOperationCapability::Optimized);
        assert_eq!(optimized.decode, PagedOperationCapability::Optimized);

        for (key_head_dim, value_head_dim) in [(64, 32), (0, 0), (520, 520), (66, 66)] {
            let mut incompatible_heads = spec.clone();
            incompatible_heads.layers[0].key_head_dim = key_head_dim;
            incompatible_heads.layers[0].value_head_dim = value_head_dim;
            let capabilities = resolve_paged_operation_capabilities(
                BackendKind::Cuda,
                true,
                true,
                true,
                &incompatible_heads,
                page_32_half,
            );
            assert_eq!(capabilities.prefill, PagedOperationCapability::Portable);
            assert_eq!(capabilities.decode, PagedOperationCapability::Portable);
        }

        let page_16 = ResolvedPagedPolicy {
            page_tokens: 16,
            ..page_32_half
        };
        let page_16_capabilities = resolve_paged_operation_capabilities(
            BackendKind::Cuda,
            true,
            true,
            true,
            spec,
            page_16,
        );
        assert_eq!(
            page_16_capabilities.prefill,
            PagedOperationCapability::Portable
        );
        assert_eq!(
            page_16_capabilities.decode,
            PagedOperationCapability::Portable
        );

        let f32 = ResolvedPagedPolicy {
            storage: StateStorageFormat::Dense {
                dtype: StateDType::F32,
            },
            ..page_32_half
        };
        let f32_capabilities =
            resolve_paged_operation_capabilities(BackendKind::Cuda, true, true, true, spec, f32);
        assert_eq!(f32_capabilities.prefill, PagedOperationCapability::Portable);
        assert_eq!(f32_capabilities.decode, PagedOperationCapability::Portable);

        let mut offset_sensitive = spec.clone();
        offset_sensitive.layers[0].pattern = AttentionPattern::SlidingWindow { window_tokens: 128 };
        let offset_capabilities = resolve_paged_operation_capabilities(
            BackendKind::Cuda,
            true,
            true,
            true,
            &offset_sensitive,
            page_32_half,
        );
        assert_eq!(
            offset_capabilities.prefill,
            PagedOperationCapability::Portable
        );
        assert_eq!(
            offset_capabilities.decode,
            PagedOperationCapability::Portable
        );

        let metal_capabilities = resolve_paged_operation_capabilities(
            BackendKind::Metal,
            true,
            true,
            true,
            spec,
            page_32_half,
        );
        assert_eq!(
            metal_capabilities,
            PagedOperationCapabilities {
                write: PagedOperationCapability::Portable,
                prefill: PagedOperationCapability::Portable,
                decode: PagedOperationCapability::Portable,
            }
        );

        let killed = resolve_paged_operation_capabilities(
            BackendKind::Cuda,
            true,
            true,
            false,
            spec,
            page_32_half,
        );
        assert_eq!(
            killed,
            PagedOperationCapabilities {
                write: PagedOperationCapability::Portable,
                prefill: PagedOperationCapability::Portable,
                decode: PagedOperationCapability::Portable,
            }
        );
    }

    #[test]
    fn resource_registry_attests_fixed_logical_backing_without_guessing_residency() {
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
            .unwrap();
        assert_eq!(fixed.allocator_alignment_bytes, 1);
        assert_eq!(fixed.allocator_overhead_per_allocation, 0);
        assert_eq!(fixed.max_backing_allocations, 1);

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
    fn resource_registry_separates_paged_backing_from_lazy_tensor_authorization() {
        let contract = tensor_test_contract();
        let state_plan = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![],
            vec![tensor_test_plan(16)],
            &TensorTestOperationRegistry,
        )
        .unwrap();
        let registry = StateBackendRegistry::new(BackendKind::Cpu, None).unwrap();
        let lazy = registry
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                resolved: ResolvedCapacityDomain::NonPaged(&state_plan.non_paged[0]),
                strategy: CapacityStrategy::BoundedLazy { max_blocks: 4 },
            })
            .unwrap();
        assert_eq!(lazy.max_backing_allocations, 0);
        assert!(registry
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                resolved: ResolvedCapacityDomain::NonPaged(&state_plan.non_paged[0]),
                strategy: CapacityStrategy::Fixed { blocks: 4 },
            })
            .is_err());

        let paged_contract = test_contract();
        let paged = resolved_group(&paged_contract, BackendKind::Cpu);
        assert!(registry
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                resolved: ResolvedCapacityDomain::Paged(&paged),
                strategy: CapacityStrategy::BoundedLazy { max_blocks: 4 },
            })
            .is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_resource_registry_attests_page_16_native_fallback_backing() {
        let contract = test_contract();
        let group = resolved_group(&contract, BackendKind::Cuda);
        assert_eq!(group.page_tokens, 16);
        StateBackendRegistry::new(BackendKind::Cuda, Some(0))
            .unwrap()
            .resolve_group_resources(&GroupResourceQuery {
                backend: BackendKind::Cuda,
                device_ordinal: Some(0),
                resolved: ResolvedCapacityDomain::Paged(&group),
                strategy: CapacityStrategy::Fixed { blocks: 8 },
            })
            .unwrap();
    }

    #[test]
    fn workspace_resources_attest_exact_logical_slots_and_reject_pinned_or_wrong_identity() {
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
            .unwrap();
        assert_eq!(envelope.allocator_alignment_bytes, 1);
        assert_eq!(envelope.allocator_overhead_per_slot, 0);
        assert_eq!(envelope.max_concurrency_slots, 2);
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

    #[test]
    fn cpu_fixed_capacity_builds_a_hard_admission_plan() {
        let contract = test_contract();
        let state_plan = negotiate_state_plan(&contract, &request(BackendKind::Cpu)).unwrap();
        let group = &state_plan.paged_attention[0];
        let blocks = 8_u32;
        let hard_limit = StateResourceVector {
            host_bytes: u64::from(blocks) * group.bytes_per_page,
            ..StateResourceVector::default()
        };
        let allocation = StateRuntimeAllocationPlan::build(
            &state_plan,
            ModelInstanceId::new(9),
            vec![GroupCapacityRequest {
                group: group.group,
                domain: group.domain,
                strategy: CapacityStrategy::Fixed { blocks },
            }],
            WorkspaceContract {
                fixed_bytes: 0,
                dimensions: vec![],
                terms: vec![],
                placement: WorkspacePlacement::Host,
                concurrency_slots: 1,
            },
            hard_limit,
            &StateBackendRegistry::new(BackendKind::Cpu, None).unwrap(),
        )
        .unwrap();
        allocation.validate_against(&state_plan).unwrap();
        assert_eq!(allocation.hard_limit, hard_limit);
    }
}

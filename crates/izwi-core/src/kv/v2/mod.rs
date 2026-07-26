//! Additive inference-state ABI v2.
//!
//! v2 separates three identities which v1 mixed together:
//! semantic state requirements, an immutable backend-resolved execution plan,
//! and generation-tagged runtime allocations. The module starts with the
//! paged-attention vocabulary required by the first vertical slice; other
//! state domains are added without weakening these boundaries.

mod batch;
mod capacity;
mod contract;
mod descriptor;
mod intent;
mod legacy;
mod resolved;
mod resolved_domains;
mod runtime;
mod scratch;

#[allow(unused_imports)]
pub(crate) use batch::{
    PhysicalArenaId, PhysicalArenaLease, PhysicalBlockRef, PhysicalSlotRef,
    PreparedPagedAttentionBatch, PreparedPagedAttentionRow, PreparedPagedWrite,
};
#[allow(unused_imports)]
pub(crate) use capacity::{
    AllocationReceipt, CapacityStrategy, GroupAllocationTotals, GroupCapacityPlan,
    GroupCapacityRequest, GroupResourceQuery, ResidencyMeasurement, ResolvedCapacityDomain,
    ResolvedGroupResourceEnvelope, ResolvedWorkspaceResourceEnvelope, StateAllocationLedger,
    StateAllocationPlanId, StateResourceRegistry, StateResourceVector, StateRuntimeAllocationPlan,
    WorkspaceAxis, WorkspaceContract, WorkspaceDimensionBound, WorkspacePlacement,
    WorkspaceResourceQuery, WorkspaceTerm,
};
#[allow(unused_imports)]
pub(crate) use contract::{
    AppendStateDomainSpec, AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy,
    InferenceStateAbi, InferenceStateContract, KeyEncoding, PageSizeConstraint,
    PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy, PositionSemantics,
    PrefixPolicy, RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, StaticAttentionDomainSpec, StaticAttentionLayerSpec,
    StaticTensorDomainSpec, TensorComponentSpec, TensorRole, TensorStateDomainSpec,
    CURRENT_INFERENCE_STATE_ABI,
};
#[allow(unused_imports)]
pub(crate) use descriptor::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InvocationLeaseScope,
    InvocationStageWorkspace, InvocationStateCapacity, InvocationWorkspaceDomain,
    InvocationWorkspaceProfile, InvocationWorkspaceSet, RetainedStateCapability, WorkspaceFormula,
};
#[allow(unused_imports)]
pub(crate) use intent::{
    AdapterStateIntent, ComponentShapeInstantiation, DomainStepIntent, IntentResourceUsage,
    PrefixIntent, ShapeDimensionValue, StateUpdateKind, WorkspaceDimensionValue,
    WorkspaceShapeInstantiation,
};
pub(crate) use legacy::upgrade_kv_contract_v1;
#[allow(unused_imports)]
pub(crate) use resolved::{
    OperationAbi, PagedAttentionOperationQuery, RegisteredOperationId, ResolvedPagedAttentionGroup,
    ResolvedPlacement, ResolvedStatePlan, StateLayerBinding, StateOperationRegistry,
    StateOperationSet, StatePhysicalLayout, StatePlanFingerprint, StatePlanId, StateStorageFormat,
};
#[allow(unused_imports)]
pub(crate) use resolved_domains::{
    align_bytes, AppendStateOperationSet, NonPagedStateOperationQuery,
    NonPagedStateOperationRegistry, ResolvedAppendStatePlan, ResolvedNonPagedDomainPlan,
    ResolvedRingStatePlan, ResolvedStaticAttentionPlan, ResolvedStaticTensorPlan,
    ResolvedTensorComponent, ResolvedTensorStatePlan, RingStateOperationSet,
    StaticAttentionOperationSet, StaticTensorOperationSet, TensorPhysicalLayout,
    TensorStateOperationSet,
};
#[allow(unused_imports)]
pub(crate) use runtime::{
    invocation_paged_workspace_backing_v2, CapabilityRuntimeIdentityV2, CapabilityStateRuntimeV2,
    InvocationCapabilityRuntimeV2, InvocationPagedDomainCompletionV2, InvocationPagedLeaseSetV2,
    InvocationPagedWorkspaceBindingV2, InvocationPagedWorkspaceKeyV2,
    InvocationPagedWorkspaceRuntimeV2, InvocationStateBackingKindV2,
    InvocationWorkspaceBackingIdentityV2, InvocationWorkspaceBackingV2,
    InvocationWorkspaceBindingV2, InvocationWorkspaceDomainCompletionV2, InvocationWorkspaceKeyV2,
    InvocationWorkspaceLeaseSetV2, InvocationWorkspaceLeaseV2,
    InvocationWorkspacePhysicalCompletionV2, InvocationWorkspacePhysicalLeaseV2,
    InvocationWorkspaceRuntimeV2, ManagedCapabilityRuntimeV2, RetainedStateRuntimeV2,
    RetainedStateUseV2, StatelessCapabilityRuntimeV2,
};
#[allow(unused_imports)]
pub(crate) use scratch::{
    ResolvedScratchPlacement, ResolvedScratchWorkspace, ScratchMemoryDomain,
    ScratchWorkspaceAllocationId, ScratchWorkspaceLease, ScratchWorkspaceOwner,
    ScratchWorkspacePlanId, ScratchWorkspacePool, ScratchWorkspaceSlotRef,
};

#[cfg(test)]
pub(crate) use contract::test_contract;

//! Additive inference-state ABI v2.
//!
//! v2 separates three identities which v1 mixed together:
//! semantic state requirements, an immutable backend-resolved execution plan,
//! and generation-tagged runtime allocations. The module starts with the
//! paged-attention vocabulary required by the first vertical slice; other
//! state domains are added without weakening these boundaries.

mod capacity;
mod contract;
mod descriptor;
mod intent;
mod resolved;
mod resolved_domains;
mod runtime;

#[cfg(test)]
pub(crate) use resolved_domains::tests::{
    contract as tensor_test_contract, tensor_plan as tensor_test_plan,
    TestRegistry as TensorTestOperationRegistry,
};

pub(crate) use capacity::{
    AllocationReceipt, CapacityStrategy, GroupCapacityRequest, GroupResourceQuery,
    ResidencyMeasurement, ResolvedCapacityDomain, ResolvedGroupResourceEnvelope,
    ResolvedWorkspaceResourceEnvelope, StateAllocationLedger, StateAllocationPlanId,
    StateResourceRegistry, StateResourceVector, StateRuntimeAllocationPlan, WorkspaceContract,
    WorkspacePlacement, WorkspaceResourceQuery,
};
pub(crate) use contract::{
    AppendStateDomainSpec, AttentionLogitSoftcap, AttentionMask, AttentionPattern, BoundedShape,
    CheckpointPolicy, InferenceStateContract, KeyEncoding, PageSizeConstraint,
    PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy, PositionSemantics,
    PrefixPolicy, RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeExtent, StateComponentId,
    StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupSpec, StateScope,
    StaticAttentionDomainSpec, StaticAttentionLayerSpec, StaticTensorDomainSpec,
    TensorComponentSpec, TensorRole, TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
};
pub use contract::{StateClock, StateGroupId};
pub(crate) use descriptor::{
    minimum_physical_bytes_for_capacity, stage_graph_fingerprint, CapabilityStateDescriptorV2,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet,
    RetainedStateCapability, StateCapacityAxis, StateCapacityBinding, WorkspaceFormula,
};
pub(crate) use intent::{
    ComponentShapeInstantiation, DomainStepIntent, ShapeDimensionValue, StateUpdateKind,
};
pub(crate) use resolved::{
    OperationAbi, PagedAttentionOperationQuery, PagedOperationImplementation,
    PagedOperationImplementationSet, RegisteredOperationId, ResolvedPagedAttentionGroup,
    ResolvedPlacement, ResolvedStatePlan, StateLayerBinding, StateOperationRegistry,
    StateOperationSet, StatePhysicalLayout, StatePlanId, StateStorageFormat,
};
pub(crate) use resolved_domains::{
    align_bytes, AppendStateOperationSet, NonPagedStateOperationQuery,
    NonPagedStateOperationRegistry, ResolvedAppendStatePlan, ResolvedNonPagedDomainPlan,
    ResolvedRingStatePlan, ResolvedStaticAttentionPlan, ResolvedStaticTensorPlan,
    ResolvedTensorComponent, ResolvedTensorStatePlan, RingStateOperationSet,
    StaticAttentionOperationSet, StaticTensorOperationSet, TensorPhysicalLayout,
    TensorStateOperationSet,
};
pub(crate) use runtime::{
    invocation_paged_workspace_backing_v2, CapabilityStateRuntimeV2, InvocationCapabilityRuntimeV2,
    InvocationPagedLeaseSetV2, InvocationStateBackingKindV2, InvocationWorkspaceBackingIdentityV2,
    InvocationWorkspaceBackingV2, InvocationWorkspaceBindingV2, InvocationWorkspaceKeyV2,
    InvocationWorkspaceLeaseSetV2, InvocationWorkspacePhysicalCompletionV2,
    InvocationWorkspacePhysicalLeaseV2, InvocationWorkspaceRuntimeV2, ManagedCapabilityRuntimeV2,
    RetainedStateRuntimeV2, RetainedStateUseV2, StatelessCapabilityRuntimeV2,
};

#[cfg(test)]
pub(crate) use capacity::{WorkspaceAxis, WorkspaceDimensionBound};
#[cfg(test)]
pub(crate) use contract::test_contract;
#[cfg(test)]
pub(crate) use runtime::fence_probe;

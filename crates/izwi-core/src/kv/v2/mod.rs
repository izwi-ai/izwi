//! Additive inference-state ABI v2.
//!
//! v2 separates three identities which v1 mixed together:
//! semantic state requirements, an immutable backend-resolved execution plan,
//! and generation-tagged runtime allocations. The module starts with the
//! paged-attention vocabulary required by the first vertical slice; other
//! state domains are added without weakening these boundaries.

mod batch;
mod contract;
mod resolved;

#[allow(unused_imports)]
pub(crate) use batch::{
    PhysicalArenaId, PhysicalArenaLease, PhysicalBlockRef, PhysicalSlotRef,
    PreparedPagedAttentionBatch, PreparedPagedAttentionRow, PreparedPagedWrite,
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
pub(crate) use resolved::{
    OperationAbi, PagedAttentionOperationQuery, RegisteredOperationId, ResolvedPagedAttentionGroup,
    ResolvedStatePlan, StateOperationRegistry, StateOperationSet, StatePhysicalLayout,
    StatePlanFingerprint, StatePlanId, StateStorageFormat,
};

#[cfg(test)]
pub(crate) use contract::test_contract;

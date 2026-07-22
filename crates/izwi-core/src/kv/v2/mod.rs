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
    AttentionMask, AttentionPattern, InferenceStateAbi, InferenceStateContract, KeyEncoding,
    PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PositionSemantics,
    PrefixPolicy, StateClock, StateDType, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, CURRENT_INFERENCE_STATE_ABI,
};
#[allow(unused_imports)]
pub(crate) use resolved::{
    OperationAbi, PagedAttentionOperationQuery, RegisteredOperationId, ResolvedPagedAttentionGroup,
    ResolvedStatePlan, StateOperationRegistry, StateOperationSet, StatePhysicalLayout,
    StatePlanFingerprint, StatePlanId, StateStorageFormat,
};

#[cfg(test)]
pub(crate) use contract::test_contract;

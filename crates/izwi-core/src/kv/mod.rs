//! Model-declared KV cache requirements and backend-resolved physical plans.
//!
//! This module is deliberately a control-plane contract. It contains no
//! tensors, device pointers, scheduler tables, or cache ownership. Loaded
//! models describe semantic state here; backends resolve that description
//! into a physical layout before an arena is allocated.

mod batch;
mod contract;
mod residency;
mod resolved;

/// Additive inference-state ABI v2. This remains crate-private until loaded
/// adapters and every backend can publish and execute it without fallback.
pub(crate) mod v2;

pub use batch::{CacheBlockRef, KvDecodeBatchMetadata, KvSequenceBlockTable, KvSlotRef};
pub use contract::{
    AttentionSemantics, CacheCapability, CacheDomainId, CacheTokenAxis, KeyEncoding,
    KvCacheContract, KvCacheContractProvider, KvContractAbi, KvDomainSpec, KvPrefixSemantics,
    KvStorageDType, KvStorageRequest, ModelStateDomainSpec, ModelStateKind, ModelStateLayerSpec,
    PageTokenConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PositionSemantics,
    CURRENT_KV_CONTRACT_ABI,
};
pub use residency::{KvResidencyError, KvResidencyState, KvStorageTier, KvTransferId};
pub use resolved::{
    KvArenaId, KvGroupId, KvLayerBinding, KvPhysicalLayout, KvPlanFingerprint, KvPlanId,
    KvStorageFormat, PagedAttentionKernel, ResolvedKvGroup, ResolvedKvGroupKind, ResolvedKvPlan,
};

/// Explicit namespace for the compatibility ABI while the v2 migration is in
/// progress. Existing top-level re-exports are intentionally retained so this
/// commit does not change current callers.
pub mod v1 {
    pub use super::{
        AttentionSemantics, CacheBlockRef, CacheCapability, CacheDomainId, CacheTokenAxis,
        KeyEncoding, KvArenaId, KvCacheContract, KvCacheContractProvider, KvContractAbi,
        KvDecodeBatchMetadata, KvDomainSpec, KvGroupId, KvLayerBinding, KvPhysicalLayout,
        KvPlanFingerprint, KvPlanId, KvPrefixSemantics, KvResidencyError, KvResidencyState,
        KvSequenceBlockTable, KvSlotRef, KvStorageDType, KvStorageFormat, KvStorageRequest,
        KvStorageTier, KvTransferId, ModelStateDomainSpec, ModelStateKind, ModelStateLayerSpec,
        PageTokenConstraint, PagedAttentionDomainSpec, PagedAttentionKernel,
        PagedAttentionLayerSpec, PositionSemantics, ResolvedKvGroup, ResolvedKvGroupKind,
        ResolvedKvPlan, CURRENT_KV_CONTRACT_ABI,
    };
}

#[cfg(test)]
pub(crate) use contract::test_contract;

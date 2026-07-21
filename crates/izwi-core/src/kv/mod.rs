//! Model-declared KV cache requirements and backend-resolved physical plans.
//!
//! This module is deliberately a control-plane contract. It contains no
//! tensors, device pointers, scheduler tables, or cache ownership. Loaded
//! models describe semantic state here; backends resolve that description
//! into a physical layout before an arena is allocated.

mod batch;
mod contract;
mod resolved;
mod residency;

pub use batch::{CacheBlockRef, KvDecodeBatchMetadata, KvSequenceBlockTable, KvSlotRef};
pub use contract::{
    AttentionSemantics, CacheCapability, CacheDomainId, CacheTokenAxis, KeyEncoding,
    KvCacheContract, KvCacheContractProvider, KvContractAbi, KvDomainSpec, KvPrefixSemantics,
    KvStorageDType, KvStorageRequest, LoadedKvCacheCapability, ModelStateDomainSpec, ModelStateKind,
    ModelStateLayerSpec, PageTokenConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PositionSemantics, CURRENT_KV_CONTRACT_ABI,
};
pub use resolved::{
    KvArenaId, KvGroupId, KvLayerBinding, KvPhysicalLayout, KvPlanFingerprint, KvPlanId,
    KvStorageFormat, PagedAttentionKernel, ResolvedKvGroup, ResolvedKvGroupKind, ResolvedKvPlan,
};
pub use residency::{KvResidencyError, KvResidencyState, KvStorageTier, KvTransferId};

#[cfg(test)]
pub(crate) use contract::test_contract;

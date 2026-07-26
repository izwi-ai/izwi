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

/// The one semantic inference-state ABI consumed by model adapters, backend
/// negotiation, and physical runtime allocation.
pub(crate) mod v2;

pub use batch::{CacheBlockRef, KvDecodeBatchMetadata, KvSequenceBlockTable, KvSlotRef};
pub(crate) use contract::{InferenceStateCapability, InferenceStateContractProvider};
pub use residency::{KvResidencyError, KvResidencyState, KvStorageTier, KvTransferId};
pub use resolved::{
    KvArenaId, KvGroupId, KvLayerBinding, KvPhysicalLayout, KvPlanFingerprint, KvPlanId,
    KvStorageFormat, ResolvedKvGroup, ResolvedKvPlan,
};
pub(crate) use v2::{StateDType as KvStorageDType, StateDomainId as CacheDomainId};

#[cfg(test)]
pub(crate) use v2::test_contract;

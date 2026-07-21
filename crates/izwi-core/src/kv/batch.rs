use serde::{Deserialize, Serialize};

use super::{KvArenaId, KvGroupId};

/// Generation-safe control-plane reference to one physical cache page.
///
/// Raw page indices are lowered only after an arena validates both its own
/// generation and the slot generation. This prevents a stale completion,
/// free, or unpin from targeting a subsequently reused page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheBlockRef {
    pub arena: KvArenaId,
    pub group: KvGroupId,
    pub index: u32,
    pub slot_generation: u32,
}

/// One token position within a generation-safe physical cache page.
///
/// `offset` is checked against the resolved group's page size when a prepared
/// batch is validated; keeping that policy out of this identity type allows
/// page size to remain backend-negotiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvSlotRef {
    pub block: CacheBlockRef,
    pub offset: u32,
}

/// Logical page order and valid token count for one ragged decode row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSequenceBlockTable {
    pub blocks: Vec<CacheBlockRef>,
    pub context_len: u32,
}

/// Immutable block-table metadata for a ragged physical decode batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDecodeBatchMetadata {
    pub sequences: Vec<KvSequenceBlockTable>,
}

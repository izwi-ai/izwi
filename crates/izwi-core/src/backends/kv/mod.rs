//! Backend-owned physical KV-cache arenas.
//!
//! This module is intentionally independent from the scheduler's allocation,
//! reference-counting, and prefix-cache metadata. The control plane validates
//! generational block references; an arena validates its identity and physical
//! bounds before lowering those references to backend slot indices.

mod cpu;

use std::any::Any;
use std::sync::Arc;

use candle_core::{DType, Tensor};

use crate::backends::BackendKind;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding, KvSlotRef};
use crate::Result;

pub use cpu::{CpuKvArena, CpuKvBackendRuntime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerConfig {
    pub binding: KvLayerBinding,
    pub num_kv_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
}

/// Fully resolved physical shape for one backend arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvArenaConfig {
    pub id: KvArenaId,
    pub group: KvGroupId,
    pub page_tokens: u32,
    pub capacity_pages: u32,
    pub dtype: DType,
    pub layers: Vec<KvLayerConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPageCopy {
    pub source: CacheBlockRef,
    pub destination: CacheBlockRef,
}

/// Backend-specific, immutable lowering of host slot references.
///
/// Accelerator implementations can keep this mapping resident on device and
/// reuse it across all layer writes in a prepared physical batch.
pub trait KvSlotMap: Any + Send + Sync {
    fn arena_id(&self) -> KvArenaId;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_any(&self) -> &dyn Any;
}

pub struct KvWriteArgs<'a> {
    pub keys: &'a Tensor,
    pub values: &'a Tensor,
    pub slots: &'a dyn KvSlotMap,
}

/// Completion token for an ordered backend mutation.
pub trait KvDeviceFence: Send + Sync {
    fn is_complete(&self) -> bool;
    fn wait(&self) -> Result<()>;
}

pub type DeviceFence = Arc<dyn KvDeviceFence>;

/// Physical arena mutation ABI shared by CPU and accelerator backends.
pub trait KvArena: Send + Sync {
    fn id(&self) -> KvArenaId;
    fn backend_kind(&self) -> BackendKind;
    fn config(&self) -> &KvArenaConfig;

    /// Validate arena identity and bounds, then lower to backend slot indices.
    /// Slot generations must already have been validated by the control plane.
    fn lower_slots(&self, slots: &[KvSlotRef]) -> Result<Arc<dyn KvSlotMap>>;

    fn zero_pages(&self, pages: &[CacheBlockRef]) -> Result<DeviceFence>;
    fn copy_pages(&self, copies: &[KvPageCopy]) -> Result<DeviceFence>;
    fn write_slots(&self, layer: KvLayerBinding, args: KvWriteArgs<'_>) -> Result<DeviceFence>;
}

/// Allocates backend-owned arenas from resolved physical configurations.
pub trait KvBackendRuntime: Send + Sync {
    fn backend_kind(&self) -> BackendKind;
    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>>;
}

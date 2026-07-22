//! Backend-owned physical KV-cache arenas.
//!
//! This module is intentionally independent from the scheduler's allocation,
//! reference-counting, and prefix-cache metadata. The control plane validates
//! generational block references; an arena validates its identity and physical
//! bounds before lowering those references to backend slot indices.

#[cfg(any(feature = "cuda", feature = "metal"))]
mod accelerator;
mod cpu;
mod negotiate;

use std::any::Any;
use std::sync::Arc;

use candle_core::{DType, DeviceLocation, Tensor};

use crate::backends::BackendKind;
use crate::kv::{
    CacheBlockRef, KvArenaId, KvCacheContract, KvDecodeBatchMetadata, KvGroupId, KvLayerBinding,
    KvSlotRef, ResolvedKvPlan,
};
use crate::Result;

#[cfg(feature = "flash-attn")]
pub use accelerator::CudaKvBackendRuntime;
#[cfg(feature = "metal")]
pub use accelerator::MetalKvBackendRuntime;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use accelerator::{
    candle_accelerator_kv_support, CandleAcceleratorKvArena, CandleAcceleratorKvSupport,
};
pub use cpu::{CpuKvArena, CpuKvBackendRuntime};
pub use negotiate::{negotiate_kv_plan, KvBackendPlanRequest};

/// Whether this binary contains a complete managed-KV runtime for a backend.
/// Capability publication and live worker binding share this gate so a loaded
/// adapter cannot advertise managed paging without a direct attention kernel.
pub const fn managed_kv_backend_compiled(backend: BackendKind) -> bool {
    match backend {
        BackendKind::Cpu => true,
        BackendKind::Metal => cfg!(feature = "metal"),
        BackendKind::Cuda => cfg!(feature = "flash-attn"),
    }
}

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

/// Monotonic physical-operation counters exposed without leaking arena
/// tensors through the control-plane boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvArenaOperationStats {
    pub slot_write_dispatches: u64,
    pub paged_decode_dispatches: u64,
    pub page_zero_dispatches: u64,
    pub page_copy_dispatches: u64,
    /// Explicit device synchronization that blocks the calling host thread.
    pub host_synchronizations: u64,
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

/// One-token-per-row paged decode over authoritative arena storage.
pub struct KvPagedDecodeArgs<'a> {
    /// `[batch, query_heads, key_head_dim]`.
    pub queries: &'a Tensor,
    pub batch: &'a KvDecodeBatchMetadata,
    pub softmax_scale: f32,
}

/// One ragged row in a multi-query paged prefill/extend operation.
///
/// `context_len` is the visible context after the final query token has been
/// written. Earlier query tokens observe the causal prefix ending at their own
/// position, so no dense causal mask or repeated KV heads are materialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPagedPrefillRow {
    pub blocks: Vec<CacheBlockRef>,
    pub first_page_offset: u32,
    pub query_start: u32,
    pub query_len: u32,
    pub context_len: u32,
}

/// Ragged multi-query attention over authoritative paged arena storage.
pub struct KvPagedPrefillArgs<'a> {
    /// `[total_queries, query_heads, key_head_dim]`, flattened row-major.
    pub queries: &'a Tensor,
    pub rows: &'a [KvPagedPrefillRow],
    pub softmax_scale: f32,
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
    fn device_location(&self) -> DeviceLocation;
    fn config(&self) -> &KvArenaConfig;

    /// Validate arena identity and bounds, then lower to backend slot indices.
    /// Slot generations must already have been validated by the control plane.
    fn lower_slots(&self, slots: &[KvSlotRef]) -> Result<Arc<dyn KvSlotMap>>;

    fn zero_pages(&self, pages: &[CacheBlockRef]) -> Result<DeviceFence>;
    fn copy_pages(&self, copies: &[KvPageCopy]) -> Result<DeviceFence>;
    fn write_slots(&self, layer: KvLayerBinding, args: KvWriteArgs<'_>) -> Result<DeviceFence>;
    /// Direct paged prefill/extend. Backends may fuse this operation; the
    /// portable default remains page-native by issuing the already-attested
    /// direct decode operation for each causal query position.
    fn paged_prefill(&self, layer: KvLayerBinding, args: KvPagedPrefillArgs<'_>) -> Result<Tensor> {
        let query_dims = args.queries.dims();
        if query_dims.len() != 3 {
            return Err(crate::Error::InferenceError(format!(
                "paged prefill queries must have rank 3, got {query_dims:?}"
            )));
        }
        if args.rows.is_empty() || !args.softmax_scale.is_finite() || args.softmax_scale <= 0.0 {
            return Err(crate::Error::InferenceError(
                "paged prefill requires rows and a finite positive scale".into(),
            ));
        }

        let mut next_query = 0_u32;
        let mut outputs = Vec::with_capacity(query_dims[0]);
        for row in args.rows {
            if row.query_start != next_query
                || row.query_len == 0
                || row.query_len > row.context_len
                || row.first_page_offset >= self.config().page_tokens
            {
                return Err(crate::Error::InferenceError(
                    "paged prefill rows are not canonical valid causal ranges".into(),
                ));
            }
            let prefix_len = row.context_len - row.query_len;
            for local_query in 0..row.query_len {
                let visible = prefix_len
                    .checked_add(local_query)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        crate::Error::InferenceError("paged prefill context overflow".into())
                    })?;
                let physical_tokens =
                    visible.checked_add(row.first_page_offset).ok_or_else(|| {
                        crate::Error::InferenceError("paged prefill physical range overflow".into())
                    })?;
                let required_pages = physical_tokens.div_ceil(self.config().page_tokens) as usize;
                if required_pages == 0 || required_pages > row.blocks.len() {
                    return Err(crate::Error::InferenceError(
                        "paged prefill block table does not cover its causal context".into(),
                    ));
                }
                let query_index = row.query_start.checked_add(local_query).ok_or_else(|| {
                    crate::Error::InferenceError("paged prefill query index overflow".into())
                })?;
                let query = args.queries.narrow(0, query_index as usize, 1)?;
                let batch = KvDecodeBatchMetadata {
                    sequences: vec![crate::kv::KvSequenceBlockTable {
                        blocks: row.blocks[..required_pages].to_vec(),
                        first_page_offset: row.first_page_offset,
                        context_len: visible,
                    }],
                };
                outputs.push(self.paged_decode(
                    layer,
                    KvPagedDecodeArgs {
                        queries: &query,
                        batch: &batch,
                        softmax_scale: args.softmax_scale,
                    },
                )?);
            }
            next_query = next_query.checked_add(row.query_len).ok_or_else(|| {
                crate::Error::InferenceError("paged prefill query range overflow".into())
            })?;
        }
        if next_query as usize != query_dims[0] {
            return Err(crate::Error::InferenceError(
                "paged prefill rows do not cover every query exactly once".into(),
            ));
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, 0).map_err(crate::Error::from)
    }
    fn paged_decode(&self, layer: KvLayerBinding, args: KvPagedDecodeArgs<'_>) -> Result<Tensor>;

    fn operation_stats(&self) -> KvArenaOperationStats {
        KvArenaOperationStats::default()
    }

    /// Wait until every operation that can still reference this arena's
    /// storage has completed. Model unload calls this before dropping the
    /// arena generation and its physical resource lease.
    fn drain(&self) -> Result<()>;
}

/// Allocates backend-owned arenas from resolved physical configurations.
pub trait KvBackendRuntime: Send + Sync {
    fn backend_kind(&self) -> BackendKind;
    fn negotiate(
        &self,
        contract: &KvCacheContract,
        request: &KvBackendPlanRequest,
    ) -> Result<ResolvedKvPlan> {
        if request.backend != self.backend_kind() {
            return Err(crate::Error::InvalidInput(format!(
                "KV negotiation request targets {:?}, but runtime is {:?}",
                request.backend,
                self.backend_kind()
            )));
        }
        negotiate_kv_plan(contract, request)
    }
    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>>;
}

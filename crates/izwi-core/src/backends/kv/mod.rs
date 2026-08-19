//! Backend-owned physical KV-cache arenas.
//!
//! This module is intentionally independent from the scheduler's allocation,
//! reference-counting, and prefix-cache metadata. The control plane validates
//! generational block references; an arena validates its identity and physical
//! bounds before lowering those references to backend slot indices.

#[cfg(any(feature = "cuda", feature = "metal"))]
mod accelerator;
mod cpu;
mod cuda_tuning;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_core::{DType, DeviceLocation, Tensor};

use crate::backends::BackendKind;
use crate::kv::{
    CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvGroupId, KvLayerBinding, KvSlotRef,
};
use crate::{Error, Result};

#[cfg(feature = "cuda")]
pub use accelerator::CudaKvBackendRuntime;
#[cfg(feature = "metal")]
pub use accelerator::MetalKvBackendRuntime;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use accelerator::{
    candle_accelerator_kv_support, CandleAcceleratorKvArena, CandleAcceleratorKvSupport,
    CandleAttentionPlanCacheStats,
};
pub use cpu::{CpuKvArena, CpuKvBackendRuntime};

/// Whether this binary contains a complete managed-KV runtime for a backend.
/// Capability publication and live worker binding share this gate so a loaded
/// adapter cannot advertise managed paging without a direct attention kernel.
pub const fn managed_kv_backend_compiled(backend: BackendKind) -> bool {
    match backend {
        BackendKind::Cpu => true,
        BackendKind::Metal => cfg!(feature = "metal"),
        BackendKind::Cuda => cfg!(feature = "cuda"),
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
    /// CUDA-only physical growth geometry. `None` keeps the complete logical
    /// capacity resident from construction, as required by CPU, Metal, and
    /// invocation-owned fixed arenas.
    pub growth: Option<KvArenaGrowthConfig>,
    pub dtype: DType,
    pub layers: Vec<KvLayerConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvArenaGrowthConfig {
    pub initial_pages: u32,
    pub growth_quantum_pages: u32,
}

const CUDA_PAGED_GROWTH_QUANTUM_PAGES: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaPagedGrowthGeometry {
    pub initial_pages: u32,
    pub growth_quantum_pages: u32,
}

/// Resolve one deterministic CUDA growth geometry from a logical maximum.
///
/// The initial allocation is congruent with the maximum modulo the growth
/// quantum, so every later admission can add complete quanta without a special
/// terminal allocation. CPU and Metal never use this policy.
pub(crate) const fn cuda_paged_growth_geometry(maximum_pages: u32) -> CudaPagedGrowthGeometry {
    let quantum = CUDA_PAGED_GROWTH_QUANTUM_PAGES;
    let remainder = maximum_pages % quantum;
    let initial_pages = if maximum_pages <= quantum {
        maximum_pages
    } else if remainder == 0 {
        quantum
    } else {
        remainder
    };
    CudaPagedGrowthGeometry {
        initial_pages,
        growth_quantum_pages: quantum,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvArenaGrowthPlan {
    pub arena: KvArenaId,
    pub previous_pages: u32,
    pub target_pages: u32,
}

impl KvArenaGrowthPlan {
    pub fn added_pages(self) -> u32 {
        self.target_pages.saturating_sub(self.previous_pages)
    }
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
    pub paged_prefill_dispatches: u64,
    pub paged_decode_dispatches: u64,
    pub page_zero_dispatches: u64,
    pub page_copy_dispatches: u64,
    pub attention_plan_cache_hits: u64,
    pub attention_plan_cache_misses: u64,
    pub attention_plan_cache_evictions: u64,
    pub attention_plan_device_uploads: u64,
    pub attention_plan_resident_bytes: u64,
    /// Number of long-lived K/V backing allocations owned by the arena.
    pub backing_allocations: Option<u64>,
    /// Provider workspace bytes currently retained by the arena. `None`
    /// means the provider cannot meter this value yet.
    pub workspace_bytes: Option<u64>,
    /// Configured hard ceiling for provider workspace retention.
    pub workspace_budget_bytes: Option<u64>,
    /// Largest provider workspace reservation observed since arena creation.
    pub workspace_high_water_bytes: Option<u64>,
    /// Number of provider workspace allocations made by this arena.
    pub workspace_allocations: Option<u64>,
    pub cpu_reference_attention_dispatches: u64,
    pub portable_attention_dispatches: u64,
    pub cuda_native_attention_dispatches: u64,
    pub cuda_flash_attention_dispatches: u64,
    pub metal_native_attention_dispatches: u64,
    pub cuda_graph_warmups: u64,
    pub cuda_graph_captures: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_fallbacks: u64,
    pub cuda_graph_backoff_hits: u64,
    pub cuda_graph_evictions: u64,
    /// Provider that completed the most recent attention operation.
    pub last_attention_provider: Option<KvAttentionProvider>,
    /// Explicit device synchronization that blocks the calling host thread.
    pub host_synchronizations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAttentionProvider {
    CpuReference,
    Portable,
    CudaNative,
    CudaFlashAttention,
    MetalNative,
}

impl KvAttentionProvider {
    pub const fn name(self) -> &'static str {
        match self {
            Self::CpuReference => "cpu_reference",
            Self::Portable => "portable",
            Self::CudaNative => "cuda_native",
            Self::CudaFlashAttention => "cuda_flash_attention",
            Self::MetalNative => "metal_native",
        }
    }

    const fn code(self) -> u64 {
        self as u64 + 1
    }

    fn from_code(code: u64) -> Option<Self> {
        match code {
            1 => Some(Self::CpuReference),
            2 => Some(Self::Portable),
            3 => Some(Self::CudaNative),
            4 => Some(Self::CudaFlashAttention),
            5 => Some(Self::MetalNative),
            _ => None,
        }
    }
}

/// Backend-specific, immutable lowering of host slot references.
///
/// Accelerator implementations can keep this mapping resident on device and
/// reuse it across all layer writes in a prepared physical batch.
pub trait KvSlotMap: Any + Send + Sync {
    fn arena_id(&self) -> KvArenaId;
    fn len(&self) -> usize;
    /// Exact generation-safe logical slots represented by this lowering.
    /// Backends retain this identity beside any device-native index buffer so
    /// completion proofs can be reconciled without trusting raw indices.
    fn logical_slots(&self) -> Arc<[KvSlotRef]>;
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
pub struct PagedKvDecodeArgs<'a> {
    /// `[batch, query_heads, key_head_dim]`.
    pub queries: &'a Tensor,
    pub batch: &'a KvDecodeBatchMetadata,
    pub softmax_scale: f32,
    /// Optional Gemma-style logit softcap applied after scaling and before
    /// online softmax: `cap * tanh(score / cap)`.
    pub softcap: Option<f32>,
}

/// One ragged row in a multi-query paged prefill/extend operation.
///
/// `context_len` is the visible context after the final query token has been
/// written. Earlier query tokens observe the causal prefix ending at their own
/// position, so no dense causal mask or repeated KV heads are materialized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvPrefillRow {
    pub blocks: Vec<CacheBlockRef>,
    pub first_page_offset: u32,
    pub query_start: u32,
    pub query_len: u32,
    pub context_len: u32,
}

/// Ragged multi-query attention over authoritative paged arena storage.
pub struct PagedKvPrefillArgs<'a> {
    /// `[total_queries, query_heads, key_head_dim]`, flattened row-major.
    pub queries: &'a Tensor,
    pub rows: &'a [PagedKvPrefillRow],
    pub softmax_scale: f32,
    /// Optional Gemma-style logit softcap applied after scaling and before
    /// online softmax.
    pub softcap: Option<f32>,
    /// Optional causal window size, including the current query token.
    pub window_tokens: Option<u32>,
}

/// Completion token for an ordered backend mutation.
pub trait KvDeviceFence: Send + Sync {
    fn is_complete(&self) -> bool;
    fn wait(&self) -> Result<()>;
}

pub type DeviceFence = Arc<dyn KvDeviceFence>;

/// Backend-authenticated completion for one physical K/V write dispatch.
///
/// The constructor is private to the backend module tree, so model and
/// executor code can wait on or collect a real dispatch result but cannot
/// fabricate one from reservation metadata.
pub struct KvWriteCompletion {
    arena: KvArenaId,
    layer: KvLayerBinding,
    slots: Arc<[KvSlotRef]>,
    fence: DeviceFence,
}

impl KvWriteCompletion {
    fn new(
        arena: KvArenaId,
        layer: KvLayerBinding,
        slots: Arc<[KvSlotRef]>,
        fence: DeviceFence,
    ) -> Self {
        Self {
            arena,
            layer,
            slots,
            fence,
        }
    }

    pub(crate) fn arena(&self) -> KvArenaId {
        self.arena
    }

    pub(crate) fn layer(&self) -> KvLayerBinding {
        self.layer
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn slot_refs(&self) -> &[KvSlotRef] {
        &self.slots
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.fence.is_complete()
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.fence.wait()
    }
}

/// Submit an arena operation ordered after one physical write without forcing
/// immediate host synchronization. A failed submission drains the write fence
/// before returning so aborted pages cannot be reused while mutation remains
/// in flight. Successful callers retain the completion for batch sealing.
pub(crate) fn submit_ordered_after_write<T>(
    completion: KvWriteCompletion,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, KvWriteCompletion)> {
    match operation() {
        Ok(output) => Ok((output, completion)),
        Err(operation_error) => match completion.wait() {
            Ok(()) => Err(operation_error),
            Err(drain_error) => Err(Error::InferenceError(format!(
                "ordered KV operation failed: {operation_error}; write drain also failed: {drain_error}"
            ))),
        },
    }
}

/// Sealed proof that every expected layer write in one physical batch has
/// completed on the exact arena that issued the write tokens.
///
/// Construction is private to [`KvWriteCompletionCollector`]. Consumers can
/// inspect the completed shape, but cannot fabricate a successful batch from
/// scheduler metadata or individual fences.
#[derive(Debug)]
pub(crate) struct KvWriteBatchCompletion {
    arena: KvArenaId,
    layers: Vec<KvLayerBinding>,
    slots: Arc<[KvSlotRef]>,
    page_tokens: u32,
}

impl KvWriteBatchCompletion {
    pub(crate) fn arena(&self) -> KvArenaId {
        self.arena
    }

    pub(crate) fn layers(&self) -> &[KvLayerBinding] {
        &self.layers
    }

    pub(crate) fn slots_per_layer(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn slots(&self) -> &[KvSlotRef] {
        &self.slots
    }

    pub(crate) fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    pub(crate) fn total_slots(&self) -> usize {
        self.slots
            .len()
            .checked_mul(self.layers.len())
            .expect("validated KV write batch slot count overflowed")
    }

    /// Narrow one already sealed batch proof to an exact leading slot range.
    ///
    /// Sealing has already waited every layer fence for the original slot set,
    /// so this cannot expose unfinished writes. The returned proof deliberately
    /// omits the discarded suffix, allowing exact-slot consumers to authenticate
    /// only the logical prefix that was accepted by the model.
    pub(crate) fn into_slot_prefix(mut self, slots_per_layer: usize) -> Result<Self> {
        if slots_per_layer == 0 || slots_per_layer > self.slots.len() {
            return Err(Error::InvalidInput(format!(
                "KV write completion prefix {slots_per_layer} is outside 1..={}",
                self.slots.len()
            )));
        }
        if slots_per_layer < self.slots.len() {
            self.slots = self.slots[..slots_per_layer].to_vec().into();
        }
        Ok(self)
    }
}

/// Collects authenticated backend write tokens for one exact physical batch.
///
/// Expected bindings are fixed at construction and must be unique. Each layer
/// must contribute exactly one token for the same arena and slot count. Sealing
/// first proves that the layer set is complete, then waits every fence; it
/// returns a [`KvWriteBatchCompletion`] only when all waits succeed and every
/// fence reports completion.
pub(crate) struct KvWriteCompletionCollector {
    arena: KvArenaId,
    expected_layers: Vec<KvLayerBinding>,
    expected_slots: Arc<[KvSlotRef]>,
    page_tokens: u32,
    completions: HashMap<KvLayerBinding, KvWriteCompletion>,
}

impl KvWriteCompletionCollector {
    pub(crate) fn new(config: &KvArenaConfig, expected_slots: Arc<[KvSlotRef]>) -> Result<Self> {
        if expected_slots.is_empty() {
            return Err(Error::InvalidInput(
                "KV write completion batch must contain at least one slot".into(),
            ));
        }

        let expected_layers = config
            .layers
            .iter()
            .map(|layer| layer.binding)
            .collect::<Vec<_>>();
        if expected_layers.is_empty() {
            return Err(Error::InvalidInput(
                "KV write completion batch must contain at least one layer".into(),
            ));
        }
        let mut unique = HashSet::with_capacity(expected_layers.len());
        for layer in &expected_layers {
            if !unique.insert(*layer) {
                return Err(Error::InvalidInput(format!(
                    "KV write completion batch repeats layer binding {}:{}",
                    layer.model_layer, layer.physical_layer
                )));
            }
        }
        let mut unique_slots = HashSet::with_capacity(expected_slots.len());
        for slot in expected_slots.iter() {
            if slot.block.arena != config.id
                || slot.block.group != config.group
                || slot.block.index >= config.capacity_pages
                || slot.offset >= config.page_tokens
                || !unique_slots.insert(*slot)
            {
                return Err(Error::InvalidInput(
                    "KV write completion batch contains a foreign, duplicate, or out-of-range slot"
                        .into(),
                ));
            }
        }
        expected_slots
            .len()
            .checked_mul(expected_layers.len())
            .ok_or_else(|| {
                Error::InvalidInput("KV write completion batch slot count overflow".into())
            })?;

        Ok(Self {
            arena: config.id,
            completions: HashMap::with_capacity(expected_layers.len()),
            expected_layers,
            expected_slots,
            page_tokens: config.page_tokens,
        })
    }

    pub(crate) fn collect(&mut self, completion: KvWriteCompletion) -> Result<()> {
        if completion.arena != self.arena {
            let message = format!(
                "KV write completion belongs to arena {:?}, expected {:?}",
                completion.arena, self.arena
            );
            return Self::reject_completion(completion, message);
        }
        if !self.expected_layers.contains(&completion.layer) {
            let message = format!(
                "KV write completion has unexpected layer binding {}:{}",
                completion.layer.model_layer, completion.layer.physical_layer
            );
            return Self::reject_completion(completion, message);
        }
        if completion.slots.as_ref() != self.expected_slots.as_ref() {
            let message = format!(
                "KV write completion for layer {}:{} covers a different physical slot set",
                completion.layer.model_layer, completion.layer.physical_layer
            );
            return Self::reject_completion(completion, message);
        }
        if self.completions.contains_key(&completion.layer) {
            let message = format!(
                "KV write completion batch received layer {}:{} more than once",
                completion.layer.model_layer, completion.layer.physical_layer
            );
            return Self::reject_completion(completion, message);
        }
        self.completions.insert(completion.layer, completion);
        Ok(())
    }

    pub(crate) fn seal(mut self) -> Result<KvWriteBatchCompletion> {
        let missing = self
            .expected_layers
            .iter()
            .filter(|layer| !self.completions.contains_key(layer))
            .map(|layer| format!("{}:{}", layer.model_layer, layer.physical_layer))
            .collect::<Vec<_>>();
        let mut first_error = (!missing.is_empty()).then(|| {
            Error::InvalidInput(format!(
                "KV write completion batch is missing layer bindings {}",
                missing.join(", ")
            ))
        });

        // Drain every collected fence even when the layer set is incomplete or
        // a prior wait fails. Returning an error must not orphan an in-flight
        // mutation merely because no batch proof can be issued.
        for layer in &self.expected_layers {
            let Some(completion) = self.completions.remove(layer) else {
                continue;
            };
            if let Err(error) = completion.wait() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            } else if !completion.is_complete() && first_error.is_none() {
                first_error = Some(Error::InferenceError(format!(
                    "KV write fence for layer {}:{} returned before completion",
                    layer.model_layer, layer.physical_layer
                )));
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }

        Ok(KvWriteBatchCompletion {
            arena: self.arena,
            layers: self.expected_layers,
            slots: self.expected_slots,
            page_tokens: self.page_tokens,
        })
    }

    fn reject_completion(completion: KvWriteCompletion, message: String) -> Result<()> {
        if let Err(error) = completion.wait() {
            return Err(Error::InferenceError(format!(
                "{message}; rejected completion also failed to drain: {error}"
            )));
        }
        if !completion.is_complete() {
            return Err(Error::InferenceError(format!(
                "{message}; rejected completion fence returned before completion"
            )));
        }
        Err(Error::InvalidInput(message))
    }
}

/// Physical arena mutation ABI shared by CPU and accelerator backends.
pub trait KvArena: Send + Sync {
    fn id(&self) -> KvArenaId;
    fn backend_kind(&self) -> BackendKind;
    fn device_location(&self) -> DeviceLocation;
    fn config(&self) -> &KvArenaConfig;

    /// Physically materialized page count. The logical capacity remains the
    /// immutable `config().capacity_pages` envelope.
    fn resident_capacity_pages(&self) -> u32 {
        self.config().capacity_pages
    }

    fn resident_bytes(&self) -> u64 {
        let per_page = self.config().layers.iter().fold(0_u64, |total, layer| {
            let heads = u64::from(layer.num_kv_heads);
            let width =
                u64::from(layer.key_head_dim).saturating_add(u64::from(layer.value_head_dim));
            total.saturating_add(
                u64::from(self.config().page_tokens)
                    .saturating_mul(heads)
                    .saturating_mul(width)
                    .saturating_mul(self.config().dtype.size_in_bytes() as u64),
            )
        });
        per_page.saturating_mul(u64::from(self.resident_capacity_pages()))
    }

    /// Plan physical growth without allocating. Callers must authorize the
    /// returned byte delta before invoking `grow_resident_pages`.
    fn plan_resident_growth(&self, required_pages: u32) -> Result<Option<KvArenaGrowthPlan>> {
        if required_pages > self.config().capacity_pages {
            return Err(Error::Backpressure(format!(
                "KV arena requires {required_pages} pages but its logical capacity is {}",
                self.config().capacity_pages
            )));
        }
        if required_pages <= self.resident_capacity_pages() {
            Ok(None)
        } else {
            Err(Error::Backpressure(
                "KV arena backing is fixed and cannot grow".to_string(),
            ))
        }
    }

    /// Materialize an exactly pre-authorized growth plan at an admission
    /// barrier. Implementations must reject stale plans.
    fn grow_resident_pages(&self, _plan: KvArenaGrowthPlan) -> Result<()> {
        Err(Error::Backpressure(
            "KV arena backing is fixed and cannot grow".to_string(),
        ))
    }

    /// Validate arena identity and bounds, then lower to backend slot indices.
    /// Slot generations must already have been validated by the control plane.
    fn lower_slots(&self, slots: &[KvSlotRef]) -> Result<Arc<dyn KvSlotMap>>;

    fn zero_pages(&self, pages: &[CacheBlockRef]) -> Result<DeviceFence>;
    fn copy_pages(&self, copies: &[KvPageCopy]) -> Result<DeviceFence>;
    /// Operations submitted through this arena after the write observe it in
    /// device submission order. The completion fence proves host visibility
    /// and safe reuse; callers may defer its wait until batch seal.
    fn write_slots(
        &self,
        layer: KvLayerBinding,
        args: KvWriteArgs<'_>,
    ) -> Result<KvWriteCompletion>;
    /// Direct paged prefill/extend. Backends may fuse this operation; the
    /// portable default remains page-native by issuing the already-attested
    /// direct decode operation for each causal query position.
    fn paged_prefill(&self, layer: KvLayerBinding, args: PagedKvPrefillArgs<'_>) -> Result<Tensor> {
        portable_paged_prefill(self, layer, args)
    }
    fn paged_decode(&self, layer: KvLayerBinding, args: PagedKvDecodeArgs<'_>) -> Result<Tensor>;

    fn operation_stats(&self) -> KvArenaOperationStats {
        KvArenaOperationStats::default()
    }

    /// Wait until every operation that can still reference this arena's
    /// storage has completed. Model unload calls this before dropping the
    /// arena generation and its physical resource lease.
    fn drain(&self) -> Result<()>;
}

/// Page-native correctness fallback for backends without a fused prefill
/// provider. Kept outside the trait default so an overriding backend can
/// explicitly select this path without recursively dispatching to itself.
pub(crate) fn portable_paged_prefill<A: KvArena + ?Sized>(
    arena: &A,
    layer: KvLayerBinding,
    args: PagedKvPrefillArgs<'_>,
) -> Result<Tensor> {
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
    validate_attention_softcap(args.softcap)?;
    if args.window_tokens == Some(0) {
        return Err(crate::Error::InferenceError(
            "paged prefill window cannot be zero".into(),
        ));
    }

    let mut next_query = 0_u32;
    let mut sequences = Vec::with_capacity(query_dims[0]);
    for row in args.rows {
        if row.query_start != next_query
            || row.query_len == 0
            || row.query_len > row.context_len
            || row.first_page_offset >= arena.config().page_tokens
        {
            return Err(crate::Error::InferenceError(
                "paged prefill rows are not canonical valid causal ranges".into(),
            ));
        }
        let prefix_len = row.context_len - row.query_len;
        for local_query in 0..row.query_len {
            let causal_visible = prefix_len
                .checked_add(local_query)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    crate::Error::InferenceError("paged prefill context overflow".into())
                })?;
            let visible = args
                .window_tokens
                .map_or(causal_visible, |window| causal_visible.min(window));
            let dropped = causal_visible - visible;
            let physical_start = row.first_page_offset.checked_add(dropped).ok_or_else(|| {
                crate::Error::InferenceError("paged prefill physical range overflow".into())
            })?;
            let page_tokens = arena.config().page_tokens;
            let first_block = (physical_start / page_tokens) as usize;
            let first_page_offset = physical_start % page_tokens;
            let required_pages = first_page_offset
                .checked_add(visible)
                .ok_or_else(|| {
                    crate::Error::InferenceError("paged prefill physical range overflow".into())
                })?
                .div_ceil(page_tokens) as usize;
            let end_block = first_block.checked_add(required_pages).ok_or_else(|| {
                crate::Error::InferenceError("paged prefill block range overflow".into())
            })?;
            if required_pages == 0 || end_block > row.blocks.len() {
                return Err(crate::Error::InferenceError(
                    "paged prefill block table does not cover its causal context".into(),
                ));
            }
            sequences.push(crate::kv::KvSequenceBlockTable {
                blocks: row.blocks[first_block..end_block].to_vec(),
                first_page_offset,
                context_len: visible,
            });
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
    arena.paged_decode(
        layer,
        PagedKvDecodeArgs {
            queries: args.queries,
            batch: &KvDecodeBatchMetadata { sequences },
            softmax_scale: args.softmax_scale,
            softcap: args.softcap,
        },
    )
}

pub(crate) fn validate_attention_softcap(softcap: Option<f32>) -> Result<()> {
    if softcap.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(crate::Error::InferenceError(
            "paged attention softcap must be finite and positive".into(),
        ));
    }
    Ok(())
}

/// Allocates backend-owned arenas from resolved physical configurations.
pub trait KvBackendRuntime: Send + Sync {
    fn backend_kind(&self) -> BackendKind;
    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::engine::ModelInstanceId;

    #[test]
    fn cuda_growth_geometry_reaches_the_exact_logical_maximum() {
        assert_eq!(
            cuda_paged_growth_geometry(4_096),
            CudaPagedGrowthGeometry {
                initial_pages: 64,
                growth_quantum_pages: 64,
            }
        );
        assert_eq!(
            cuda_paged_growth_geometry(2_000),
            CudaPagedGrowthGeometry {
                initial_pages: 16,
                growth_quantum_pages: 64,
            }
        );
        assert_eq!(
            cuda_paged_growth_geometry(40),
            CudaPagedGrowthGeometry {
                initial_pages: 40,
                growth_quantum_pages: 64,
            }
        );
    }

    struct RecordingDecodeArena {
        config: KvArenaConfig,
        calls: AtomicUsize,
        batches: Mutex<Vec<KvDecodeBatchMetadata>>,
        softcaps: Mutex<Vec<Option<f32>>>,
    }

    impl RecordingDecodeArena {
        fn new() -> Self {
            let arena = test_arena(1);
            Self {
                config: config(arena, &[layer(0, 0)]),
                calls: AtomicUsize::new(0),
                batches: Mutex::new(Vec::new()),
                softcaps: Mutex::new(Vec::new()),
            }
        }
    }

    impl KvArena for RecordingDecodeArena {
        fn id(&self) -> KvArenaId {
            self.config.id
        }

        fn backend_kind(&self) -> BackendKind {
            BackendKind::Cpu
        }

        fn device_location(&self) -> DeviceLocation {
            DeviceLocation::Cpu
        }

        fn config(&self) -> &KvArenaConfig {
            &self.config
        }

        fn lower_slots(&self, _slots: &[KvSlotRef]) -> Result<Arc<dyn KvSlotMap>> {
            Err(Error::InferenceError(
                "unused recording slot lowering".into(),
            ))
        }

        fn zero_pages(&self, _pages: &[CacheBlockRef]) -> Result<DeviceFence> {
            Err(Error::InferenceError("unused recording page zero".into()))
        }

        fn copy_pages(&self, _copies: &[KvPageCopy]) -> Result<DeviceFence> {
            Err(Error::InferenceError("unused recording page copy".into()))
        }

        fn write_slots(
            &self,
            _layer: KvLayerBinding,
            _args: KvWriteArgs<'_>,
        ) -> Result<KvWriteCompletion> {
            Err(Error::InferenceError("unused recording slot write".into()))
        }

        fn paged_decode(
            &self,
            _layer: KvLayerBinding,
            args: PagedKvDecodeArgs<'_>,
        ) -> Result<Tensor> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.batches.lock().unwrap().push(args.batch.clone());
            self.softcaps.lock().unwrap().push(args.softcap);
            Ok(args.queries.clone())
        }

        fn drain(&self) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TestFence {
        waits: Arc<AtomicUsize>,
        complete: AtomicBool,
        fail: bool,
        complete_after_wait: bool,
    }

    impl TestFence {
        fn new(waits: Arc<AtomicUsize>) -> Self {
            Self {
                waits,
                complete: AtomicBool::new(false),
                fail: false,
                complete_after_wait: true,
            }
        }

        fn failing(waits: Arc<AtomicUsize>) -> Self {
            Self {
                fail: true,
                ..Self::new(waits)
            }
        }

        fn incomplete(waits: Arc<AtomicUsize>) -> Self {
            Self {
                complete_after_wait: false,
                ..Self::new(waits)
            }
        }
    }

    impl KvDeviceFence for TestFence {
        fn is_complete(&self) -> bool {
            self.complete.load(Ordering::Acquire)
        }

        fn wait(&self) -> Result<()> {
            self.waits.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                return Err(Error::InferenceError("injected KV fence failure".into()));
            }
            if self.complete_after_wait {
                self.complete.store(true, Ordering::Release);
            }
            Ok(())
        }
    }

    fn test_arena(generation: u32) -> KvArenaId {
        KvArenaId {
            model_instance: ModelInstanceId::new(7),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
        }
    }

    const fn layer(model_layer: u32, physical_layer: u32) -> KvLayerBinding {
        KvLayerBinding {
            model_layer,
            physical_layer,
        }
    }

    fn config(arena: KvArenaId, layers: &[KvLayerBinding]) -> KvArenaConfig {
        KvArenaConfig {
            id: arena,
            group: KvGroupId::new(1),
            page_tokens: 4,
            capacity_pages: 4,
            growth: None,
            dtype: DType::F32,
            layers: layers
                .iter()
                .map(|binding| KvLayerConfig {
                    binding: *binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                })
                .collect(),
        }
    }

    fn slots(arena: KvArenaId, count: usize) -> Arc<[KvSlotRef]> {
        (0..count)
            .map(|position| KvSlotRef {
                block: CacheBlockRef {
                    arena,
                    group: KvGroupId::new(1),
                    index: (position / 4) as u32,
                    slot_generation: 1,
                },
                offset: (position % 4) as u32,
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn completion(
        arena: KvArenaId,
        layer: KvLayerBinding,
        slots: Arc<[KvSlotRef]>,
        fence: impl KvDeviceFence + 'static,
    ) -> KvWriteCompletion {
        KvWriteCompletion::new(arena, layer, slots, Arc::new(fence))
    }

    fn block(arena: KvArenaId, index: u32) -> CacheBlockRef {
        CacheBlockRef {
            arena,
            group: KvGroupId::new(1),
            index,
            slot_generation: 1,
        }
    }

    #[test]
    fn ordered_write_submission_defers_success_wait_but_drains_failure() {
        let arena = test_arena(1);
        let binding = layer(0, 0);
        let logical_slots = slots(arena, 1);
        let waits = Arc::new(AtomicUsize::new(0));
        let successful = completion(
            arena,
            binding,
            logical_slots.clone(),
            TestFence::new(waits.clone()),
        );
        let (value, successful) =
            submit_ordered_after_write(successful, || Ok::<_, Error>(7)).unwrap();
        assert_eq!(value, 7);
        assert_eq!(waits.load(Ordering::Relaxed), 0);
        successful.wait().unwrap();
        assert_eq!(waits.load(Ordering::Relaxed), 1);

        let failed = completion(arena, binding, logical_slots, TestFence::new(waits.clone()));
        assert!(submit_ordered_after_write::<()>(failed, || {
            Err(Error::InferenceError(
                "injected ordered operation failure".into(),
            ))
        })
        .is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn paged_prefill_batches_every_causal_query_into_one_decode_dispatch() {
        let arena = RecordingDecodeArena::new();
        let id = arena.id();
        let queries = Tensor::from_vec(
            (0..10).map(|value| value as f32).collect(),
            (5, 1, 2),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = arena
            .paged_prefill(
                layer(0, 0),
                PagedKvPrefillArgs {
                    queries: &queries,
                    rows: &[
                        PagedKvPrefillRow {
                            blocks: vec![block(id, 0), block(id, 1)],
                            first_page_offset: 1,
                            query_start: 0,
                            query_len: 3,
                            context_len: 5,
                        },
                        PagedKvPrefillRow {
                            blocks: vec![block(id, 2)],
                            first_page_offset: 0,
                            query_start: 3,
                            query_len: 2,
                            context_len: 2,
                        },
                    ],
                    softmax_scale: 0.5,
                    softcap: Some(1.25),
                    window_tokens: Some(3),
                },
            )
            .unwrap();

        assert_eq!(arena.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            output.to_vec3::<f32>().unwrap(),
            queries.to_vec3::<f32>().unwrap()
        );
        let batches = arena.batches.lock().unwrap();
        let sequences = &batches[0].sequences;
        assert_eq!(sequences.len(), 5);
        assert_eq!(
            sequences
                .iter()
                .map(|sequence| sequence.context_len)
                .collect::<Vec<_>>(),
            vec![3, 3, 3, 1, 2]
        );
        assert_eq!(
            sequences
                .iter()
                .map(|sequence| sequence.blocks.len())
                .collect::<Vec<_>>(),
            vec![1, 2, 2, 1, 1]
        );
        assert!(sequences[..3]
            .iter()
            .map(|sequence| sequence.first_page_offset)
            .eq([1, 2, 3]));
        assert_eq!(*arena.softcaps.lock().unwrap(), vec![Some(1.25)]);
    }

    #[test]
    fn invalid_paged_prefill_never_dispatches_decode() {
        let arena = RecordingDecodeArena::new();
        let id = arena.id();
        let queries = Tensor::zeros((2, 1, 2), DType::F32, &candle_core::Device::Cpu).unwrap();
        let invalid_start = [PagedKvPrefillRow {
            blocks: vec![block(id, 0)],
            first_page_offset: 0,
            query_start: 1,
            query_len: 2,
            context_len: 2,
        }];
        assert!(arena
            .paged_prefill(
                layer(0, 0),
                PagedKvPrefillArgs {
                    queries: &queries,
                    rows: &invalid_start,
                    softmax_scale: 0.5,
                    softcap: None,
                    window_tokens: None,
                },
            )
            .is_err());
        let incomplete = [PagedKvPrefillRow {
            blocks: vec![block(id, 0)],
            first_page_offset: 1,
            query_start: 0,
            query_len: 2,
            context_len: 5,
        }];
        assert!(arena
            .paged_prefill(
                layer(0, 0),
                PagedKvPrefillArgs {
                    queries: &queries,
                    rows: &incomplete,
                    softmax_scale: 0.5,
                    softcap: None,
                    window_tokens: None,
                },
            )
            .is_err());
        assert_eq!(arena.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn collector_seals_exact_complete_batch_in_expected_layer_order() {
        let arena = test_arena(1);
        let layers = [layer(4, 0), layer(9, 1)];
        let config = config(arena, &layers);
        let slots = slots(arena, 3);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");

        collector
            .collect(completion(
                arena,
                layers[1],
                slots.clone(),
                TestFence::new(waits.clone()),
            ))
            .expect("second layer completion");
        collector
            .collect(completion(
                arena,
                layers[0],
                slots.clone(),
                TestFence::new(waits.clone()),
            ))
            .expect("first layer completion");

        let sealed = collector.seal().expect("sealed completion");
        assert_eq!(sealed.arena(), arena);
        assert_eq!(sealed.layers(), &layers);
        assert_eq!(sealed.slots_per_layer(), 3);
        assert_eq!(sealed.slots(), slots.as_ref());
        assert_eq!(sealed.page_tokens(), 4);
        assert_eq!(sealed.total_slots(), 6);
        assert_eq!(waits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn sealed_completion_prefix_fences_full_batch_and_exposes_exact_slots() {
        let arena = test_arena(1);
        let layers = [layer(4, 0), layer(9, 1)];
        let config = config(arena, &layers);
        let slots = slots(arena, 3);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");
        for binding in layers {
            collector
                .collect(completion(
                    arena,
                    binding,
                    slots.clone(),
                    TestFence::new(waits.clone()),
                ))
                .expect("layer completion");
        }

        let prefix = collector
            .seal()
            .expect("sealed completion")
            .into_slot_prefix(2)
            .expect("authenticated prefix");

        assert_eq!(waits.load(Ordering::Relaxed), 2);
        assert_eq!(prefix.layers(), &layers);
        assert_eq!(prefix.slots(), &slots[..2]);
        assert_eq!(prefix.slots_per_layer(), 2);
        assert_eq!(prefix.total_slots(), 4);
    }

    #[test]
    fn collector_rejects_empty_duplicate_or_foreign_expectations() {
        let arena = test_arena(1);
        let binding = layer(0, 0);
        assert!(KvWriteCompletionCollector::new(&config(arena, &[]), slots(arena, 1)).is_err());
        assert!(
            KvWriteCompletionCollector::new(&config(arena, &[binding]), Vec::new().into()).is_err()
        );
        assert!(KvWriteCompletionCollector::new(
            &config(arena, &[binding, binding]),
            slots(arena, 1),
        )
        .is_err());
        assert!(KvWriteCompletionCollector::new(
            &config(arena, &[binding]),
            slots(test_arena(2), 1),
        )
        .is_err());
    }

    #[test]
    fn collector_rejects_foreign_unexpected_and_wrong_sized_tokens() {
        let arena = test_arena(1);
        let binding = layer(0, 0);
        let config = config(arena, &[binding]);
        let expected_slots = slots(arena, 2);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, expected_slots.clone()).expect("collector");

        assert!(collector
            .collect(completion(
                test_arena(2),
                binding,
                expected_slots.clone(),
                TestFence::new(waits.clone()),
            ))
            .is_err());
        assert!(collector
            .collect(completion(
                arena,
                layer(1, 1),
                expected_slots.clone(),
                TestFence::new(waits.clone()),
            ))
            .is_err());
        assert!(collector
            .collect(completion(
                arena,
                binding,
                slots(arena, 3),
                TestFence::new(waits.clone()),
            ))
            .is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn collector_rejects_duplicate_layer_tokens() {
        let arena = test_arena(1);
        let binding = layer(0, 0);
        let config = config(arena, &[binding]);
        let slots = slots(arena, 1);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");
        collector
            .collect(completion(
                arena,
                binding,
                slots.clone(),
                TestFence::new(waits.clone()),
            ))
            .expect("first completion");
        assert!(collector
            .collect(completion(
                arena,
                binding,
                slots,
                TestFence::new(waits.clone()),
            ))
            .is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn incomplete_batch_drains_collected_fences_without_sealing() {
        let arena = test_arena(1);
        let layers = [layer(0, 0), layer(1, 1)];
        let config = config(arena, &layers);
        let slots = slots(arena, 1);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");
        collector
            .collect(completion(
                arena,
                layers[0],
                slots,
                TestFence::new(waits.clone()),
            ))
            .expect("first completion");

        assert!(collector.seal().is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn seal_drains_all_fences_after_failure_without_authenticating_batch() {
        let arena = test_arena(1);
        let layers = [layer(0, 0), layer(1, 1)];
        let config = config(arena, &layers);
        let slots = slots(arena, 1);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");
        collector
            .collect(completion(
                arena,
                layers[0],
                slots.clone(),
                TestFence::failing(waits.clone()),
            ))
            .expect("failed fence is still an authenticated dispatch");
        collector
            .collect(completion(
                arena,
                layers[1],
                slots,
                TestFence::new(waits.clone()),
            ))
            .expect("second completion");

        assert!(collector.seal().is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn seal_rejects_fence_that_returns_before_completion() {
        let arena = test_arena(1);
        let binding = layer(0, 0);
        let config = config(arena, &[binding]);
        let slots = slots(arena, 1);
        let waits = Arc::new(AtomicUsize::new(0));
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.clone()).expect("collector");
        collector
            .collect(completion(
                arena,
                binding,
                slots,
                TestFence::incomplete(waits.clone()),
            ))
            .expect("completion token");

        assert!(collector.seal().is_err());
        assert_eq!(waits.load(Ordering::Relaxed), 1);
    }
}

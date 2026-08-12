use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use candle_core::{
    backend::BackendStorage, CpuStorage, DType, Device, InplaceOp1, InplaceOp3, Layout, Storage,
    Tensor,
};
use rayon::prelude::*;

use crate::backends::BackendKind;
use crate::error::Error;
use crate::kv::{CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvLayerBinding, KvSlotRef};
use crate::Result;

use super::{
    validate_attention_softcap, DeviceFence, KvArena, KvArenaConfig, KvArenaOperationStats,
    KvAttentionProvider, KvBackendRuntime, KvDeviceFence, KvPageCopy, KvSlotMap, KvWriteArgs,
    KvWriteCompletion, PagedKvDecodeArgs, PagedKvPrefillArgs,
};

#[derive(Debug)]
struct ReadyFence;

impl KvDeviceFence for ReadyFence {
    fn is_complete(&self) -> bool {
        true
    }

    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

fn ready_fence() -> DeviceFence {
    Arc::new(ReadyFence)
}

#[derive(Debug)]
struct CpuKvSlotMap {
    arena: KvArenaId,
    flat_slots: Tensor,
    logical_slots: Arc<[KvSlotRef]>,
}

impl KvSlotMap for CpuKvSlotMap {
    fn arena_id(&self) -> KvArenaId {
        self.arena
    }

    fn len(&self) -> usize {
        self.logical_slots.len()
    }

    fn logical_slots(&self) -> Arc<[KvSlotRef]> {
        self.logical_slots.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct CpuLayerStorage {
    keys: Tensor,
    values: Tensor,
    num_kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

/// CPU reference implementation of a physical paged KV arena.
///
/// Each layer owns K and V tensors with shape
/// `[capacity_pages, page_tokens, kv_heads, head_dim]`. All mutations preserve
/// those allocations; no append, concatenation, or page materialization occurs.
#[derive(Debug)]
pub struct CpuKvArena {
    config: KvArenaConfig,
    layers: HashMap<KvLayerBinding, CpuLayerStorage>,
    mutation_lock: RwLock<()>,
    slot_write_dispatches: AtomicU64,
    paged_prefill_dispatches: AtomicU64,
    paged_decode_dispatches: AtomicU64,
    page_zero_dispatches: AtomicU64,
    page_copy_dispatches: AtomicU64,
    last_attention_provider: AtomicU64,
    cpu_reference_attention_dispatches: AtomicU64,
}

impl CpuKvArena {
    pub fn new(config: KvArenaConfig) -> Result<Self> {
        validate_config(&config)?;

        let mut layers = HashMap::with_capacity(config.layers.len());
        for layer in &config.layers {
            let common = (
                config.capacity_pages as usize,
                config.page_tokens as usize,
                layer.num_kv_heads as usize,
            );
            let keys = Tensor::zeros(
                (common.0, common.1, common.2, layer.key_head_dim as usize),
                config.dtype,
                &Device::Cpu,
            )?;
            let values = Tensor::zeros(
                (common.0, common.1, common.2, layer.value_head_dim as usize),
                config.dtype,
                &Device::Cpu,
            )?;
            layers.insert(
                layer.binding,
                CpuLayerStorage {
                    keys,
                    values,
                    num_kv_heads: common.2,
                    key_head_dim: layer.key_head_dim as usize,
                    value_head_dim: layer.value_head_dim as usize,
                },
            );
        }

        Ok(Self {
            config,
            layers,
            mutation_lock: RwLock::new(()),
            slot_write_dispatches: AtomicU64::new(0),
            paged_prefill_dispatches: AtomicU64::new(0),
            paged_decode_dispatches: AtomicU64::new(0),
            page_zero_dispatches: AtomicU64::new(0),
            page_copy_dispatches: AtomicU64::new(0),
            last_attention_provider: AtomicU64::new(0),
            cpu_reference_attention_dispatches: AtomicU64::new(0),
        })
    }

    /// Read-only handles to authoritative layer storage for CPU attention.
    /// Cloning a Candle tensor shares its allocation.
    pub fn layer_tensors(&self, layer: KvLayerBinding) -> Result<(Tensor, Tensor)> {
        let layer = self.layer(layer)?;
        Ok((layer.keys.clone(), layer.values.clone()))
    }

    fn layer(&self, binding: KvLayerBinding) -> Result<&CpuLayerStorage> {
        self.layers.get(&binding).ok_or_else(|| {
            Error::InferenceError(format!(
                "KV layer binding {} is not present in arena {:?}",
                binding.physical_layer, self.config.id
            ))
        })
    }

    fn validate_block(&self, block: CacheBlockRef) -> Result<usize> {
        if block.arena != self.config.id {
            return Err(Error::InferenceError(format!(
                "KV block belongs to arena {:?}, expected {:?}",
                block.arena, self.config.id
            )));
        }
        if block.group != self.config.group {
            return Err(Error::InferenceError(format!(
                "KV block belongs to group {}, expected {}",
                block.group.get(),
                self.config.group.get()
            )));
        }
        let page = block.index as usize;
        if page >= self.config.capacity_pages as usize {
            return Err(Error::InferenceError(format!(
                "KV page {} is outside arena capacity {}",
                page, self.config.capacity_pages
            )));
        }
        Ok(page)
    }

    fn cpu_slots<'a>(&self, slots: &'a dyn KvSlotMap) -> Result<&'a CpuKvSlotMap> {
        if slots.arena_id() != self.config.id {
            return Err(Error::InferenceError(format!(
                "KV slot map belongs to arena {:?}, expected {:?}",
                slots.arena_id(),
                self.config.id
            )));
        }
        slots
            .as_any()
            .downcast_ref::<CpuKvSlotMap>()
            .ok_or_else(|| Error::InferenceError("KV slot map backend mismatch".into()))
    }
}

impl KvArena for CpuKvArena {
    fn id(&self) -> KvArenaId {
        self.config.id
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn device_location(&self) -> candle_core::DeviceLocation {
        candle_core::DeviceLocation::Cpu
    }

    fn config(&self) -> &KvArenaConfig {
        &self.config
    }

    fn lower_slots(&self, slots: &[KvSlotRef]) -> Result<Arc<dyn KvSlotMap>> {
        let mut flat_slots = Vec::with_capacity(slots.len());
        let mut unique = HashSet::with_capacity(slots.len());
        for slot in slots {
            let page = self.validate_block(slot.block)?;
            if slot.offset >= self.config.page_tokens {
                return Err(Error::InferenceError(format!(
                    "KV page offset {} is outside page size {}",
                    slot.offset, self.config.page_tokens
                )));
            }
            let flat = page
                .checked_mul(self.config.page_tokens as usize)
                .and_then(|base| base.checked_add(slot.offset as usize))
                .ok_or_else(|| Error::InferenceError("KV slot index overflow".into()))?;
            if !unique.insert(flat) {
                return Err(Error::InferenceError(format!(
                    "KV slot map contains duplicate physical slot {flat}"
                )));
            }
            flat_slots.push(u32::try_from(flat).map_err(|_| {
                Error::InferenceError(format!("KV slot index {flat} exceeds u32 range"))
            })?);
        }

        let len = flat_slots.len();
        let flat_slots = Tensor::from_vec(flat_slots, len, &Device::Cpu)?;
        Ok(Arc::new(CpuKvSlotMap {
            arena: self.config.id,
            flat_slots,
            logical_slots: Arc::from(slots),
        }))
    }

    fn zero_pages(&self, pages: &[CacheBlockRef]) -> Result<DeviceFence> {
        if pages.is_empty() {
            return Ok(ready_fence());
        }
        let page_indices = pages
            .iter()
            .copied()
            .map(|page| self.validate_block(page))
            .collect::<Result<Vec<_>>>()?;
        reject_duplicate_pages(&page_indices, "zero")?;

        let _guard = self
            .mutation_lock
            .write()
            .map_err(|_| Error::InferenceError("CPU KV arena mutation lock was poisoned".into()))?;
        let op = PageZeroOp {
            pages: page_indices,
            page_tokens: self.config.page_tokens as usize,
        };
        for layer in self.layers.values() {
            layer.keys.inplace_op1(&op)?;
            layer.values.inplace_op1(&op)?;
        }
        self.page_zero_dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(ready_fence())
    }

    fn copy_pages(&self, copies: &[KvPageCopy]) -> Result<DeviceFence> {
        if copies.is_empty() {
            return Ok(ready_fence());
        }
        let mut page_copies = Vec::with_capacity(copies.len());
        let mut destinations = HashSet::with_capacity(copies.len());
        for copy in copies {
            let source = self.validate_block(copy.source)?;
            let destination = self.validate_block(copy.destination)?;
            if !destinations.insert(destination) {
                return Err(Error::InferenceError(format!(
                    "KV page copy has duplicate destination page {destination}"
                )));
            }
            page_copies.push((source, destination));
        }

        // A single op snapshots all source pages before writing destinations, so
        // chains and cycles have parallel-copy rather than sequential-copy semantics.
        let _guard = self
            .mutation_lock
            .write()
            .map_err(|_| Error::InferenceError("CPU KV arena mutation lock was poisoned".into()))?;
        let op = PageCopyOp {
            copies: page_copies,
            page_tokens: self.config.page_tokens as usize,
        };
        for layer in self.layers.values() {
            layer.keys.inplace_op1(&op)?;
            layer.values.inplace_op1(&op)?;
        }
        self.page_copy_dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(ready_fence())
    }

    fn write_slots(
        &self,
        binding: KvLayerBinding,
        args: KvWriteArgs<'_>,
    ) -> Result<KvWriteCompletion> {
        let slots = self.cpu_slots(args.slots)?;
        let layer = self.layer(binding)?;
        validate_write_tensor(
            args.keys,
            slots.len(),
            layer.num_kv_heads,
            layer.key_head_dim,
            self.config.dtype,
            "key",
        )?;
        validate_write_tensor(
            args.values,
            slots.len(),
            layer.num_kv_heads,
            layer.value_head_dim,
            self.config.dtype,
            "value",
        )?;

        let _guard = self
            .mutation_lock
            .write()
            .map_err(|_| Error::InferenceError("CPU KV arena mutation lock was poisoned".into()))?;
        layer
            .keys
            .inplace_op3(args.keys, &slots.flat_slots, &SlotScatterOp)?;
        layer
            .values
            .inplace_op3(args.values, &slots.flat_slots, &SlotScatterOp)?;
        self.slot_write_dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(KvWriteCompletion::new(
            self.config.id,
            binding,
            args.slots.logical_slots(),
            ready_fence(),
        ))
    }

    fn paged_decode(&self, binding: KvLayerBinding, args: PagedKvDecodeArgs<'_>) -> Result<Tensor> {
        let layer = self.layer(binding)?;
        let query_dims = args.queries.dims();
        if query_dims.len() != 3 {
            return Err(Error::InferenceError(format!(
                "CPU paged decode queries must have rank 3, got {query_dims:?}"
            )));
        }
        let batch_size = query_dims[0];
        let query_heads = query_dims[1];
        if query_dims[2] != layer.key_head_dim {
            return Err(Error::InferenceError(format!(
                "CPU paged decode query head dimension {} does not match key head dimension {}",
                query_dims[2], layer.key_head_dim
            )));
        }
        if query_heads == 0 || !query_heads.is_multiple_of(layer.num_kv_heads) {
            return Err(Error::InferenceError(format!(
                "CPU paged decode query heads {query_heads} are not divisible by KV heads {}",
                layer.num_kv_heads
            )));
        }
        if args.batch.sequences.len() != batch_size {
            return Err(Error::InferenceError(format!(
                "CPU paged decode metadata has {} rows, expected {batch_size}",
                args.batch.sequences.len()
            )));
        }
        if !args.softmax_scale.is_finite() || args.softmax_scale <= 0.0 {
            return Err(Error::InferenceError(
                "CPU paged decode softmax scale must be finite and positive".into(),
            ));
        }
        validate_attention_softcap(args.softcap)?;
        validate_decode_query(args.queries, self.config.dtype)?;
        let tables = self.lower_decode_tables(args.batch)?;

        // Serialize against paired K/V mutations. The storage read guards then
        // provide stable direct slices for the complete online-softmax pass.
        let _guard = self
            .mutation_lock
            .read()
            .map_err(|_| Error::InferenceError("CPU KV arena mutation lock was poisoned".into()))?;
        let (key_storage, key_layout) = layer.keys.storage_and_layout();
        let (value_storage, value_layout) = layer.values.storage_and_layout();
        let (query_storage, query_layout) = args.queries.storage_and_layout();
        let key_start = contiguous_range(key_layout, "kv-paged-decode-cpu")?.0;
        let value_start = contiguous_range(value_layout, "kv-paged-decode-cpu")?.0;
        let query_start = contiguous_range(query_layout, "kv-paged-decode-cpu")?.0;

        macro_rules! decode_typed {
            ($keys:expr, $values:expr, $queries:expr) => {
                online_paged_decode(
                    $keys,
                    $values,
                    $queries,
                    key_start,
                    value_start,
                    query_start,
                    &tables,
                    self.config.page_tokens as usize,
                    layer.num_kv_heads,
                    layer.key_head_dim,
                    layer.value_head_dim,
                    query_heads,
                    args.softmax_scale,
                    args.softcap,
                )
            };
        }

        let output = match (&*key_storage, &*value_storage, &*query_storage) {
            (
                Storage::Cpu(CpuStorage::F32(keys)),
                Storage::Cpu(CpuStorage::F32(values)),
                Storage::Cpu(CpuStorage::F32(queries)),
            ) => {
                decode_typed!(keys, values, queries)
            }
            (
                Storage::Cpu(CpuStorage::F16(keys)),
                Storage::Cpu(CpuStorage::F16(values)),
                Storage::Cpu(CpuStorage::F16(queries)),
            ) => {
                decode_typed!(keys, values, queries)
            }
            (
                Storage::Cpu(CpuStorage::BF16(keys)),
                Storage::Cpu(CpuStorage::BF16(values)),
                Storage::Cpu(CpuStorage::BF16(queries)),
            ) => {
                decode_typed!(keys, values, queries)
            }
            _ => {
                return Err(Error::InferenceError(
                    "CPU paged decode storage dtype mismatch".into(),
                ));
            }
        }?;

        let output = Tensor::from_vec(
            output,
            (batch_size, query_heads, layer.value_head_dim),
            &Device::Cpu,
        )?
        .to_dtype(self.config.dtype)?;
        self.paged_decode_dispatches.fetch_add(1, Ordering::Relaxed);
        self.cpu_reference_attention_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.last_attention_provider
            .store(KvAttentionProvider::CpuReference.code(), Ordering::Relaxed);
        Ok(output)
    }

    fn paged_prefill(
        &self,
        binding: KvLayerBinding,
        args: PagedKvPrefillArgs<'_>,
    ) -> Result<Tensor> {
        let layer = self.layer(binding)?;
        let query_dims = args.queries.dims();
        if query_dims.len() != 3 || query_dims[2] != layer.key_head_dim {
            return Err(Error::InferenceError(format!(
                "CPU paged prefill query shape {query_dims:?} does not match key head dimension {}",
                layer.key_head_dim
            )));
        }
        let query_heads = query_dims[1];
        if query_heads == 0 || !query_heads.is_multiple_of(layer.num_kv_heads) {
            return Err(Error::InferenceError(
                "CPU paged prefill query heads are incompatible with KV heads".into(),
            ));
        }
        if !args.softmax_scale.is_finite() || args.softmax_scale <= 0.0 {
            return Err(Error::InferenceError(
                "CPU paged prefill softmax scale must be finite and positive".into(),
            ));
        }
        validate_attention_softcap(args.softcap)?;
        if args.window_tokens == Some(0) {
            return Err(Error::InferenceError(
                "CPU paged prefill window cannot be zero".into(),
            ));
        }
        validate_decode_query(args.queries, self.config.dtype)?;

        let mut tables = LoweredAttentionTables {
            pages: Vec::new(),
            rows: Vec::with_capacity(query_dims[0]),
        };
        let mut next_query = 0_u32;
        for (row_index, row) in args.rows.iter().enumerate() {
            if row.query_start != next_query
                || row.query_len == 0
                || row.query_len > row.context_len
                || row.first_page_offset >= self.config.page_tokens
            {
                return Err(Error::InferenceError(format!(
                    "CPU paged prefill row {row_index} is not a canonical causal range"
                )));
            }
            let pages = row
                .blocks
                .iter()
                .copied()
                .map(|block| self.validate_block(block))
                .collect::<Result<Vec<_>>>()?;
            let row_pages_start = tables.pages.len();
            tables.pages.extend_from_slice(&pages);
            let prefix_len = row.context_len - row.query_len;
            for local_query in 0..row.query_len {
                let causal_context_len = prefix_len
                    .checked_add(local_query)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| Error::InferenceError("prefill context overflow".into()))?;
                let context_len = args
                    .window_tokens
                    .map_or(causal_context_len, |window| causal_context_len.min(window));
                let dropped = causal_context_len - context_len;
                let physical_start =
                    row.first_page_offset.checked_add(dropped).ok_or_else(|| {
                        Error::InferenceError("prefill physical range overflow".into())
                    })?;
                let first_page = (physical_start / self.config.page_tokens) as usize;
                let first_page_offset = (physical_start % self.config.page_tokens) as usize;
                let required_pages = first_page_offset
                    .checked_add(context_len as usize)
                    .ok_or_else(|| Error::InferenceError("prefill physical range overflow".into()))?
                    .div_ceil(self.config.page_tokens as usize);
                let end_page = first_page
                    .checked_add(required_pages)
                    .ok_or_else(|| Error::InferenceError("prefill block range overflow".into()))?;
                if required_pages == 0 || end_page > pages.len() {
                    return Err(Error::InferenceError(format!(
                        "CPU paged prefill row {row_index} has an incomplete block table"
                    )));
                }
                tables.rows.push(LoweredDecodeRow {
                    page_start: row_pages_start + first_page,
                    page_len: required_pages,
                    first_page_offset,
                    context_len: context_len as usize,
                });
            }
            next_query = next_query
                .checked_add(row.query_len)
                .ok_or_else(|| Error::InferenceError("prefill query range overflow".into()))?;
        }
        if next_query as usize != query_dims[0] || tables.rows.is_empty() {
            return Err(Error::InferenceError(
                "CPU paged prefill rows do not cover every query exactly once".into(),
            ));
        }

        let _guard = self
            .mutation_lock
            .read()
            .map_err(|_| Error::InferenceError("CPU KV arena mutation lock was poisoned".into()))?;
        let (key_storage, key_layout) = layer.keys.storage_and_layout();
        let (value_storage, value_layout) = layer.values.storage_and_layout();
        let (query_storage, query_layout) = args.queries.storage_and_layout();
        let key_start = contiguous_range(key_layout, "kv-paged-prefill-cpu")?.0;
        let value_start = contiguous_range(value_layout, "kv-paged-prefill-cpu")?.0;
        let query_start = contiguous_range(query_layout, "kv-paged-prefill-cpu")?.0;

        macro_rules! prefill_typed {
            ($keys:expr, $values:expr, $queries:expr) => {
                online_paged_decode(
                    $keys,
                    $values,
                    $queries,
                    key_start,
                    value_start,
                    query_start,
                    &tables,
                    self.config.page_tokens as usize,
                    layer.num_kv_heads,
                    layer.key_head_dim,
                    layer.value_head_dim,
                    query_heads,
                    args.softmax_scale,
                    args.softcap,
                )
            };
        }
        let output = match (&*key_storage, &*value_storage, &*query_storage) {
            (
                Storage::Cpu(CpuStorage::F32(k)),
                Storage::Cpu(CpuStorage::F32(v)),
                Storage::Cpu(CpuStorage::F32(q)),
            ) => prefill_typed!(k, v, q),
            (
                Storage::Cpu(CpuStorage::F16(k)),
                Storage::Cpu(CpuStorage::F16(v)),
                Storage::Cpu(CpuStorage::F16(q)),
            ) => prefill_typed!(k, v, q),
            (
                Storage::Cpu(CpuStorage::BF16(k)),
                Storage::Cpu(CpuStorage::BF16(v)),
                Storage::Cpu(CpuStorage::BF16(q)),
            ) => prefill_typed!(k, v, q),
            _ => {
                return Err(Error::InferenceError(
                    "CPU paged prefill storage dtype mismatch".into(),
                ))
            }
        }?;
        let output = Tensor::from_vec(
            output,
            (query_dims[0], query_heads, layer.value_head_dim),
            &Device::Cpu,
        )?
        .to_dtype(self.config.dtype)
        .map_err(Error::from)?;
        self.paged_prefill_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.cpu_reference_attention_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.last_attention_provider
            .store(KvAttentionProvider::CpuReference.code(), Ordering::Relaxed);
        Ok(output)
    }

    fn operation_stats(&self) -> KvArenaOperationStats {
        KvArenaOperationStats {
            slot_write_dispatches: self.slot_write_dispatches.load(Ordering::Relaxed),
            paged_prefill_dispatches: self.paged_prefill_dispatches.load(Ordering::Relaxed),
            paged_decode_dispatches: self.paged_decode_dispatches.load(Ordering::Relaxed),
            page_zero_dispatches: self.page_zero_dispatches.load(Ordering::Relaxed),
            page_copy_dispatches: self.page_copy_dispatches.load(Ordering::Relaxed),
            attention_plan_cache_hits: 0,
            attention_plan_cache_misses: 0,
            attention_plan_cache_evictions: 0,
            attention_plan_device_uploads: 0,
            attention_plan_resident_bytes: 0,
            backing_allocations: Some((self.layers.len() * 2) as u64),
            workspace_bytes: Some(0),
            workspace_allocations: Some(0),
            cpu_reference_attention_dispatches: self
                .cpu_reference_attention_dispatches
                .load(Ordering::Relaxed),
            portable_attention_dispatches: 0,
            cuda_native_attention_dispatches: 0,
            cuda_flash_attention_dispatches: 0,
            metal_native_attention_dispatches: 0,
            cuda_graph_warmups: 0,
            cuda_graph_captures: 0,
            cuda_graph_replays: 0,
            cuda_graph_fallbacks: 0,
            cuda_graph_backoff_hits: 0,
            cuda_graph_evictions: 0,
            last_attention_provider: KvAttentionProvider::from_code(
                self.last_attention_provider.load(Ordering::Relaxed),
            ),
            host_synchronizations: 0,
        }
    }

    fn drain(&self) -> Result<()> {
        Ok(())
    }
}

impl CpuKvArena {
    fn lower_decode_tables(&self, batch: &KvDecodeBatchMetadata) -> Result<LoweredAttentionTables> {
        let total_pages = batch
            .sequences
            .iter()
            .try_fold(0_usize, |total, sequence| {
                total.checked_add(sequence.blocks.len()).ok_or_else(|| {
                    Error::InferenceError("CPU paged decode table size overflow".into())
                })
            })?;
        let mut lowered = LoweredAttentionTables {
            pages: Vec::with_capacity(total_pages),
            rows: Vec::with_capacity(batch.sequences.len()),
        };
        for (row, sequence) in batch.sequences.iter().enumerate() {
            if sequence.context_len == 0 {
                return Err(Error::InferenceError(format!(
                    "CPU paged decode row {row} has an empty context"
                )));
            }
            if sequence.first_page_offset >= self.config.page_tokens {
                return Err(Error::InferenceError(format!(
                    "CPU paged decode row {row} first-page offset {} exceeds page size {}",
                    sequence.first_page_offset, self.config.page_tokens
                )));
            }
            let physical_tokens = sequence
                .context_len
                .checked_add(sequence.first_page_offset)
                .ok_or_else(|| {
                    Error::InferenceError(
                        "CPU paged decode physical token range exceeds u32".into(),
                    )
                })?;
            let required_pages =
                (physical_tokens as usize).div_ceil(self.config.page_tokens as usize);
            if sequence.blocks.len() != required_pages {
                return Err(Error::InferenceError(format!(
                    "CPU paged decode row {row} has {} pages, expected {required_pages}",
                    sequence.blocks.len()
                )));
            }
            let page_start = lowered.pages.len();
            for block in sequence.blocks.iter().copied() {
                lowered.pages.push(self.validate_block(block)?);
            }
            lowered.rows.push(LoweredDecodeRow {
                page_start,
                page_len: required_pages,
                first_page_offset: sequence.first_page_offset as usize,
                context_len: sequence.context_len as usize,
            });
        }
        Ok(lowered)
    }
}

struct LoweredAttentionTables {
    pages: Vec<usize>,
    rows: Vec<LoweredDecodeRow>,
}

struct LoweredDecodeRow {
    page_start: usize,
    page_len: usize,
    first_page_offset: usize,
    context_len: usize,
}

fn validate_decode_query(query: &Tensor, dtype: DType) -> Result<()> {
    if query.device().location() != Device::Cpu.location() {
        return Err(Error::InferenceError(
            "CPU paged decode queries must be on CPU".into(),
        ));
    }
    if query.dtype() != dtype {
        return Err(Error::InferenceError(format!(
            "CPU paged decode query dtype {:?} does not match arena dtype {dtype:?}",
            query.dtype()
        )));
    }
    if !query.layout().is_contiguous() {
        return Err(Error::InferenceError(
            "CPU paged decode queries must be contiguous".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn online_paged_decode<T>(
    keys: &[T],
    values: &[T],
    queries: &[T],
    key_start: usize,
    value_start: usize,
    query_start: usize,
    tables: &LoweredAttentionTables,
    page_tokens: usize,
    kv_heads: usize,
    key_dim: usize,
    value_dim: usize,
    query_heads: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
) -> Result<Vec<f32>>
where
    T: Copy + Sync,
    f32: From<T>,
{
    let batch_size = tables.rows.len();
    let mut output = vec![0.0f32; batch_size * query_heads * value_dim];
    let queries_per_kv_head = query_heads / kv_heads;
    let key_page_stride = page_tokens * kv_heads * key_dim;
    let value_page_stride = page_tokens * kv_heads * value_dim;

    for table in &tables.rows {
        let covered_pages = table
            .first_page_offset
            .checked_add(table.context_len)
            .ok_or_else(|| Error::InferenceError("CPU paged attention range overflow".into()))?
            .div_ceil(page_tokens);
        if covered_pages > table.page_len
            || table.page_start.saturating_add(table.page_len) > tables.pages.len()
        {
            return Err(Error::InferenceError(
                "CPU paged attention lowered table is incomplete".into(),
            ));
        }
    }

    let compute_head = |row_head: usize, output: &mut [f32], accumulator: &mut Vec<f32>| {
        let row = row_head / query_heads;
        let query_head = row_head % query_heads;
        let table = &tables.rows[row];
        let kv_head = query_head / queries_per_kv_head;
        let query_offset = query_start + row_head * key_dim;
        accumulator.resize(value_dim, 0.0);
        accumulator.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;

        for token in 0..table.context_len {
            let physical_token = table.first_page_offset + token;
            let logical_page = physical_token / page_tokens;
            let page = tables.pages[table.page_start + logical_page];
            let page_offset = physical_token % page_tokens;
            let key_offset =
                key_start + page * key_page_stride + (page_offset * kv_heads + kv_head) * key_dim;
            let value_offset = value_start
                + page * value_page_stride
                + (page_offset * kv_heads + kv_head) * value_dim;
            let mut score = 0.0f32;
            for dim in 0..key_dim {
                score += f32::from(queries[query_offset + dim]) * f32::from(keys[key_offset + dim]);
            }
            score *= softmax_scale;
            if let Some(softcap) = softcap {
                score = softcap * (score / softcap).tanh();
            }

            let next_max = running_max.max(score);
            let previous_weight = (running_max - next_max).exp();
            let token_weight = (score - next_max).exp();
            running_sum = running_sum * previous_weight + token_weight;
            for dim in 0..value_dim {
                accumulator[dim] = accumulator[dim] * previous_weight
                    + f32::from(values[value_offset + dim]) * token_weight;
            }
            running_max = next_max;
        }
        for dim in 0..value_dim {
            output[dim] = accumulator[dim] / running_sum;
        }
    };

    let row_heads = batch_size * query_heads;
    if row_heads >= 8 {
        output.par_chunks_mut(value_dim).enumerate().for_each_init(
            || Vec::with_capacity(value_dim),
            |acc, (row_head, output)| compute_head(row_head, output, acc),
        );
    } else {
        let mut accumulator = Vec::with_capacity(value_dim);
        for (row_head, output) in output.chunks_mut(value_dim).enumerate() {
            compute_head(row_head, output, &mut accumulator);
        }
    }
    Ok(output)
}

#[derive(Debug, Default)]
pub struct CpuKvBackendRuntime;

impl KvBackendRuntime for CpuKvBackendRuntime {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>> {
        Ok(Arc::new(CpuKvArena::new(config)?))
    }
}

fn validate_config(config: &KvArenaConfig) -> Result<()> {
    if config.page_tokens == 0 || config.capacity_pages == 0 {
        return Err(Error::InferenceError(
            "CPU KV arena page size and capacity must be non-zero".into(),
        ));
    }
    if !matches!(config.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::InferenceError(format!(
            "CPU KV arena does not support {:?} storage",
            config.dtype
        )));
    }
    if config.layers.is_empty() {
        return Err(Error::InferenceError(
            "CPU KV arena must contain at least one layer".into(),
        ));
    }
    let total_slots = (config.page_tokens as u64)
        .checked_mul(config.capacity_pages as u64)
        .ok_or_else(|| Error::InferenceError("CPU KV arena slot count overflow".into()))?;
    if total_slots > u32::MAX as u64 {
        return Err(Error::InferenceError(format!(
            "CPU KV arena has {total_slots} slots, exceeding the u32 slot ABI"
        )));
    }
    let mut bindings = HashSet::with_capacity(config.layers.len());
    for layer in &config.layers {
        if !bindings.insert(layer.binding) {
            return Err(Error::InferenceError(format!(
                "CPU KV arena contains duplicate layer binding {}",
                layer.binding.physical_layer
            )));
        }
        if layer.num_kv_heads == 0 || layer.key_head_dim == 0 || layer.value_head_dim == 0 {
            return Err(Error::InferenceError(format!(
                "CPU KV layer {} has zero-sized geometry",
                layer.binding.physical_layer
            )));
        }
    }
    Ok(())
}

fn validate_write_tensor(
    tensor: &Tensor,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    dtype: DType,
    label: &str,
) -> Result<()> {
    if tensor.device().location() != Device::Cpu.location() {
        return Err(Error::InferenceError(format!(
            "CPU KV {label} source must be on CPU"
        )));
    }
    if tensor.dtype() != dtype {
        return Err(Error::InferenceError(format!(
            "CPU KV {label} source dtype {:?} does not match arena dtype {:?}",
            tensor.dtype(),
            dtype
        )));
    }
    let expected = [tokens, heads, head_dim];
    if tensor.dims() != expected {
        return Err(Error::InferenceError(format!(
            "CPU KV {label} source shape {:?} does not match {:?}",
            tensor.dims(),
            expected
        )));
    }
    if !tensor.layout().is_contiguous() {
        return Err(Error::InferenceError(format!(
            "CPU KV {label} source must be contiguous"
        )));
    }
    Ok(())
}

fn reject_duplicate_pages(pages: &[usize], operation: &str) -> Result<()> {
    let mut unique = HashSet::with_capacity(pages.len());
    for &page in pages {
        if !unique.insert(page) {
            return Err(Error::InferenceError(format!(
                "KV page {operation} contains duplicate page {page}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SlotScatterOp;

impl InplaceOp3 for SlotScatterOp {
    fn name(&self) -> &'static str {
        "kv-slot-scatter-cpu"
    }

    fn cpu_fwd(
        &self,
        destination: &mut CpuStorage,
        destination_layout: &Layout,
        source: &CpuStorage,
        source_layout: &Layout,
        slots: &CpuStorage,
        slots_layout: &Layout,
    ) -> candle_core::Result<()> {
        let (destination_start, destination_end) =
            contiguous_range(destination_layout, self.name())?;
        let (source_start, source_end) = contiguous_range(source_layout, self.name())?;
        let (slot_start, slot_end) = contiguous_range(slots_layout, self.name())?;
        let slots = match slots {
            CpuStorage::U32(values) => &values[slot_start..slot_end],
            _ => candle_core::bail!("{} expects a u32 slot map", self.name()),
        };
        if slots.is_empty() {
            return Ok(());
        }
        let source_len = source_end - source_start;
        if source_len % slots.len() != 0 {
            candle_core::bail!(
                "{} source element count {} is not divisible by slot count {}",
                self.name(),
                source_len,
                slots.len()
            )
        }
        let row_len = source_len / slots.len();
        let destination_len = destination_end - destination_start;
        scatter_storage(
            destination,
            destination_start,
            destination_len,
            source,
            source_start,
            slots,
            row_len,
            self.name(),
        )
    }
}

#[derive(Debug)]
struct PageZeroOp {
    pages: Vec<usize>,
    page_tokens: usize,
}

impl InplaceOp1 for PageZeroOp {
    fn name(&self) -> &'static str {
        "kv-page-zero-cpu"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        let (start, end) = contiguous_range(layout, self.name())?;
        let capacity_pages = layout.dims()[0];
        let page_len = (end - start) / capacity_pages;
        debug_assert_eq!(layout.dims()[1], self.page_tokens);
        zero_storage(storage, start, page_len, &self.pages, self.name())
    }
}

#[derive(Debug)]
struct PageCopyOp {
    copies: Vec<(usize, usize)>,
    page_tokens: usize,
}

impl InplaceOp1 for PageCopyOp {
    fn name(&self) -> &'static str {
        "kv-page-copy-cpu"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        let (start, end) = contiguous_range(layout, self.name())?;
        let capacity_pages = layout.dims()[0];
        let page_len = (end - start) / capacity_pages;
        debug_assert_eq!(layout.dims()[1], self.page_tokens);
        copy_storage_pages(storage, start, page_len, &self.copies, self.name())
    }
}

fn contiguous_range(layout: &Layout, op: &str) -> candle_core::Result<(usize, usize)> {
    layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg(format!("{op} requires contiguous storage")))
}

macro_rules! with_float_storage {
    ($storage:expr, $values:ident, $body:expr, $op:expr) => {
        match $storage {
            CpuStorage::F32($values) => $body,
            CpuStorage::F16($values) => $body,
            CpuStorage::BF16($values) => $body,
            other => candle_core::bail!("{} does not support {:?}", $op, other.dtype()),
        }
    };
}

fn scatter_storage(
    destination: &mut CpuStorage,
    destination_start: usize,
    destination_len: usize,
    source: &CpuStorage,
    source_start: usize,
    slots: &[u32],
    row_len: usize,
    op: &str,
) -> candle_core::Result<()> {
    macro_rules! scatter_typed {
        ($destination:expr, $source:expr) => {{
            for (source_row, &slot) in slots.iter().enumerate() {
                let destination_offset = (slot as usize)
                    .checked_mul(row_len)
                    .ok_or_else(|| candle_core::Error::Msg(format!("{op} slot offset overflow")))?;
                if destination_offset + row_len > destination_len {
                    candle_core::bail!("{op} slot {slot} exceeds destination capacity")
                }
                let src = source_start + source_row * row_len;
                let dst = destination_start + destination_offset;
                $destination[dst..dst + row_len].copy_from_slice(&$source[src..src + row_len]);
            }
            Ok(())
        }};
    }

    match (destination, source) {
        (CpuStorage::F32(destination), CpuStorage::F32(source)) => {
            scatter_typed!(destination, source)
        }
        (CpuStorage::F16(destination), CpuStorage::F16(source)) => {
            scatter_typed!(destination, source)
        }
        (CpuStorage::BF16(destination), CpuStorage::BF16(source)) => {
            scatter_typed!(destination, source)
        }
        (destination, source) => candle_core::bail!(
            "{op} storage mismatch: destination {:?}, source {:?}",
            destination.dtype(),
            source.dtype()
        ),
    }
}

fn zero_storage(
    storage: &mut CpuStorage,
    start: usize,
    page_len: usize,
    pages: &[usize],
    op: &str,
) -> candle_core::Result<()> {
    with_float_storage!(
        storage,
        values,
        {
            for &page in pages {
                let offset = start + page * page_len;
                values[offset..offset + page_len].fill(Default::default());
            }
            Ok(())
        },
        op
    )
}

fn copy_storage_pages(
    storage: &mut CpuStorage,
    start: usize,
    page_len: usize,
    copies: &[(usize, usize)],
    op: &str,
) -> candle_core::Result<()> {
    macro_rules! copy_typed {
        ($values:expr) => {{
            let snapshots = copies
                .iter()
                .map(|&(source, _)| {
                    let source = start + source * page_len;
                    $values[source..source + page_len].to_vec()
                })
                .collect::<Vec<_>>();
            for ((_, destination), source) in copies.iter().zip(snapshots) {
                let destination = start + destination * page_len;
                $values[destination..destination + page_len].copy_from_slice(&source);
            }
            Ok(())
        }};
    }

    with_float_storage!(storage, values, copy_typed!(values), op)
}

#[cfg(test)]
mod tests {
    use candle_core::IndexOp;

    use super::*;
    use crate::backends::kv::{KvLayerConfig, PagedKvPrefillRow};
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvDecodeBatchMetadata, KvGroupId, KvSequenceBlockTable, KvSlotRef};

    const ARENA: KvArenaId = KvArenaId {
        model_instance: ModelInstanceId::new(41),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 3,
    };
    const GROUP: KvGroupId = KvGroupId::new(5);
    const LAYER: KvLayerBinding = KvLayerBinding {
        model_layer: 7,
        physical_layer: 0,
    };

    fn config(dtype: DType) -> KvArenaConfig {
        config_with_heads(dtype, 2)
    }

    fn config_with_heads(dtype: DType, num_kv_heads: u32) -> KvArenaConfig {
        KvArenaConfig {
            id: ARENA,
            group: GROUP,
            page_tokens: 2,
            capacity_pages: 3,
            growth: None,
            dtype,
            layers: vec![KvLayerConfig {
                binding: LAYER,
                num_kv_heads,
                key_head_dim: 2,
                value_head_dim: 1,
            }],
        }
    }

    fn block(index: u32) -> CacheBlockRef {
        CacheBlockRef {
            arena: ARENA,
            group: GROUP,
            index,
            slot_generation: 11,
        }
    }

    fn cpu_storage_ptr(tensor: &Tensor) -> usize {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            Storage::Cpu(CpuStorage::F32(values)) => values.as_ptr() as usize,
            Storage::Cpu(CpuStorage::F16(values)) => values.as_ptr() as usize,
            Storage::Cpu(CpuStorage::BF16(values)) => values.as_ptr() as usize,
            other => panic!("unexpected test storage: {:?}", other.dtype()),
        }
    }

    fn dense_reference(
        queries: &[f32],
        query_heads: usize,
        keys: &[f32],
        values: &[f32],
        kv_heads: usize,
        key_dim: usize,
        value_dim: usize,
        logical_tokens: &[Vec<usize>],
        scale: f32,
        softcap: Option<f32>,
    ) -> Vec<f32> {
        let mut output = Vec::new();
        let queries_per_kv_head = query_heads / kv_heads;
        for (row, tokens) in logical_tokens.iter().enumerate() {
            for query_head in 0..query_heads {
                let kv_head = query_head / queries_per_kv_head;
                let query = &queries[(row * query_heads + query_head) * key_dim..][..key_dim];
                let scores = tokens
                    .iter()
                    .map(|&token| {
                        let key = &keys[(token * kv_heads + kv_head) * key_dim..][..key_dim];
                        let score = query
                            .iter()
                            .zip(key)
                            .map(|(query, key)| query * key)
                            .sum::<f32>()
                            * scale;
                        match softcap {
                            Some(softcap) => softcap * (score / softcap).tanh(),
                            None => score,
                        }
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f32>();
                for dim in 0..value_dim {
                    output.push(
                        tokens
                            .iter()
                            .zip(&weights)
                            .map(|(&token, weight)| {
                                values[(token * kv_heads + kv_head) * value_dim + dim] * weight
                            })
                            .sum::<f32>()
                            / sum,
                    );
                }
            }
        }
        output
    }

    #[test]
    fn scatter_writes_flat_slots_without_reallocating_storage() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let (keys, values) = arena.layer_tensors(LAYER)?;
        let key_ptr = cpu_storage_ptr(&keys);
        let value_ptr = cpu_storage_ptr(&values);
        let slots = arena.lower_slots(&[
            KvSlotRef {
                block: block(0),
                offset: 1,
            },
            KvSlotRef {
                block: block(2),
                offset: 0,
            },
        ])?;
        let source_keys = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (2, 2, 2),
            &Device::Cpu,
        )?;
        let source_values = Tensor::from_vec(vec![9f32, 10., 11., 12.], (2, 2, 1), &Device::Cpu)?;

        let fence = arena.write_slots(
            LAYER,
            KvWriteArgs {
                keys: &source_keys,
                values: &source_values,
                slots: slots.as_ref(),
            },
        )?;
        assert!(fence.is_complete());
        fence.wait()?;

        assert_eq!(cpu_storage_ptr(&keys), key_ptr);
        assert_eq!(cpu_storage_ptr(&values), value_ptr);
        assert_eq!(
            keys.i((0, 1))?.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.]]
        );
        assert_eq!(
            keys.i((2, 0))?.to_vec2::<f32>()?,
            vec![vec![5., 6.], vec![7., 8.]]
        );
        assert_eq!(
            values.i((0, 1))?.to_vec2::<f32>()?,
            vec![vec![9.], vec![10.]]
        );
        assert_eq!(
            values.i((2, 0))?.to_vec2::<f32>()?,
            vec![vec![11.], vec![12.]]
        );
        Ok(())
    }

    #[test]
    fn copy_and_zero_preserve_backing_allocations() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let (keys, values) = arena.layer_tensors(LAYER)?;
        let key_ptr = cpu_storage_ptr(&keys);
        let value_ptr = cpu_storage_ptr(&values);
        let slots = arena.lower_slots(&[
            KvSlotRef {
                block: block(0),
                offset: 0,
            },
            KvSlotRef {
                block: block(0),
                offset: 1,
            },
        ])?;
        let source_keys = Tensor::from_vec(
            (1..=8).map(|v| v as f32).collect::<Vec<_>>(),
            (2, 2, 2),
            &Device::Cpu,
        )?;
        let source_values = Tensor::from_vec(vec![9f32, 10., 11., 12.], (2, 2, 1), &Device::Cpu)?;
        arena.write_slots(
            LAYER,
            KvWriteArgs {
                keys: &source_keys,
                values: &source_values,
                slots: slots.as_ref(),
            },
        )?;

        arena.copy_pages(&[KvPageCopy {
            source: block(0),
            destination: block(1),
        }])?;
        arena.zero_pages(&[block(0)])?;

        assert_eq!(cpu_storage_ptr(&keys), key_ptr);
        assert_eq!(cpu_storage_ptr(&values), value_ptr);
        assert!(keys
            .i(0)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&value| value == 0.));
        assert!(values
            .i(0)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&value| value == 0.));
        assert_eq!(
            keys.i((1, 0))?.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.]]
        );
        assert_eq!(
            keys.i((1, 1))?.to_vec2::<f32>()?,
            vec![vec![5., 6.], vec![7., 8.]]
        );
        assert_eq!(
            values.i((1, 0))?.to_vec2::<f32>()?,
            vec![vec![9.], vec![10.]]
        );
        assert_eq!(
            values.i((1, 1))?.to_vec2::<f32>()?,
            vec![vec![11.], vec![12.]]
        );
        Ok(())
    }

    #[test]
    fn parallel_page_copy_uses_pre_write_sources() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let slots = arena.lower_slots(&[
            KvSlotRef {
                block: block(0),
                offset: 0,
            },
            KvSlotRef {
                block: block(1),
                offset: 0,
            },
        ])?;
        let source_keys = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 9., 8., 7., 6.],
            (2, 2, 2),
            &Device::Cpu,
        )?;
        let source_values = Tensor::from_vec(vec![5f32, 6., 4., 3.], (2, 2, 1), &Device::Cpu)?;
        arena.write_slots(
            LAYER,
            KvWriteArgs {
                keys: &source_keys,
                values: &source_values,
                slots: slots.as_ref(),
            },
        )?;

        arena.copy_pages(&[
            KvPageCopy {
                source: block(0),
                destination: block(1),
            },
            KvPageCopy {
                source: block(1),
                destination: block(0),
            },
        ])?;

        let (keys, _) = arena.layer_tensors(LAYER)?;
        assert_eq!(
            keys.i((0, 0))?.to_vec2::<f32>()?,
            vec![vec![9., 8.], vec![7., 6.]]
        );
        assert_eq!(
            keys.i((1, 0))?.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![3., 4.]]
        );
        Ok(())
    }

    #[test]
    fn lowering_rejects_wrong_arena_bounds_and_duplicate_slots() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let wrong_arena = KvSlotRef {
            block: CacheBlockRef {
                arena: KvArenaId {
                    model_instance: ARENA.model_instance,
                    backend: ARENA.backend,
                    device_ordinal: ARENA.device_ordinal,
                    generation: ARENA.generation + 1,
                },
                group: GROUP,
                index: 0,
                slot_generation: 1,
            },
            offset: 0,
        };
        assert!(arena.lower_slots(&[wrong_arena]).is_err());
        assert!(arena
            .lower_slots(&[KvSlotRef {
                block: block(3),
                offset: 0,
            }])
            .is_err());
        assert!(arena
            .lower_slots(&[KvSlotRef {
                block: block(0),
                offset: 2,
            }])
            .is_err());
        let duplicate = KvSlotRef {
            block: block(0),
            offset: 1,
        };
        assert!(arena.lower_slots(&[duplicate, duplicate]).is_err());
        Ok(())
    }

    #[test]
    fn f16_and_bf16_mutation_paths_are_supported() -> Result<()> {
        for dtype in [DType::F16, DType::BF16] {
            let arena = CpuKvArena::new(config(dtype))?;
            let slots = arena.lower_slots(&[KvSlotRef {
                block: block(0),
                offset: 0,
            }])?;
            let keys = Tensor::ones((1, 2, 2), dtype, &Device::Cpu)?;
            let values = Tensor::ones((1, 2, 1), dtype, &Device::Cpu)?;
            arena.write_slots(
                LAYER,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )?;
            arena.copy_pages(&[KvPageCopy {
                source: block(0),
                destination: block(1),
            }])?;
            arena.zero_pages(&[block(0)])?;
        }
        Ok(())
    }

    #[test]
    fn paged_decode_matches_dense_gqa_with_shuffled_ragged_tables() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let physical_slots = (0..3)
            .flat_map(|page| {
                (0..2).map(move |offset| KvSlotRef {
                    block: block(page),
                    offset,
                })
            })
            .collect::<Vec<_>>();
        let slot_map = arena.lower_slots(&physical_slots)?;
        let keys = vec![
            1., 0., 0., 1., // page 0, token 0
            0., 1., 1., 0., // page 0, token 1
            1., 1., 1., -1., // page 1, token 0
            -1., 1., 2., 1., // page 1, token 1
            2., 0., 0., 2., // page 2, token 0
            0., 2., 2., 0., // page 2, token 1
        ];
        let values = vec![
            1., 10., // page 0, token 0
            2., 20., // page 0, token 1
            3., 30., // page 1, token 0
            4., 40., // page 1, token 1
            5., 50., // page 2, token 0
            6., 60., // page 2, token 1
        ];
        let key_tensor = Tensor::from_vec(keys.clone(), (6, 2, 2), &Device::Cpu)?;
        let value_tensor = Tensor::from_vec(values.clone(), (6, 2, 1), &Device::Cpu)?;
        arena.write_slots(
            LAYER,
            KvWriteArgs {
                keys: &key_tensor,
                values: &value_tensor,
                slots: slot_map.as_ref(),
            },
        )?;

        let query_values = vec![
            1., 0., 0., 1., 1., 1., 1., -1., // row 0, four query heads
            0.5, 1., 1., 0.5, -1., 1., 1., 1., // row 1
        ];
        let queries = Tensor::from_vec(query_values.clone(), (2, 4, 2), &Device::Cpu)?;
        let batch = KvDecodeBatchMetadata {
            sequences: vec![
                KvSequenceBlockTable {
                    blocks: vec![block(2), block(0)],
                    first_page_offset: 1,
                    context_len: 3,
                },
                KvSequenceBlockTable {
                    blocks: vec![block(1)],
                    first_page_offset: 0,
                    context_len: 1,
                },
            ],
        };
        let scale = 1.0 / 2.0f32.sqrt();
        let actual = arena
            .paged_decode(
                LAYER,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: &batch,
                    softmax_scale: scale,
                    softcap: None,
                },
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Physical tokens 5, 0, 1 correspond to shuffled table [page 2, page 0]
        // after hiding its first slot; row 1 contains only physical token 2.
        let expected = dense_reference(
            &query_values,
            4,
            &keys,
            &values,
            2,
            2,
            1,
            &[vec![5, 0, 1], vec![2]],
            scale,
            None,
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn paged_decode_supports_mha_and_mqa_head_mapping() -> Result<()> {
        for (kv_heads, query_heads) in [(2, 2), (1, 4)] {
            let arena = CpuKvArena::new(config_with_heads(DType::F32, kv_heads))?;
            let slots = arena.lower_slots(&[
                KvSlotRef {
                    block: block(0),
                    offset: 0,
                },
                KvSlotRef {
                    block: block(0),
                    offset: 1,
                },
            ])?;
            let keys = (0..2 * kv_heads as usize * 2)
                .map(|index| (index + 1) as f32 / 4.0)
                .collect::<Vec<_>>();
            let values = (0..2 * kv_heads as usize)
                .map(|index| (index + 1) as f32)
                .collect::<Vec<_>>();
            let keys_tensor =
                Tensor::from_vec(keys.clone(), (2, kv_heads as usize, 2), &Device::Cpu)?;
            let values_tensor =
                Tensor::from_vec(values.clone(), (2, kv_heads as usize, 1), &Device::Cpu)?;
            arena.write_slots(
                LAYER,
                KvWriteArgs {
                    keys: &keys_tensor,
                    values: &values_tensor,
                    slots: slots.as_ref(),
                },
            )?;
            let query_values = (0..query_heads * 2)
                .map(|index| (index + 1) as f32 / 3.0)
                .collect::<Vec<_>>();
            let queries =
                Tensor::from_vec(query_values.clone(), (1, query_heads, 2), &Device::Cpu)?;
            let batch = KvDecodeBatchMetadata {
                sequences: vec![KvSequenceBlockTable {
                    blocks: vec![block(0)],
                    first_page_offset: 0,
                    context_len: 2,
                }],
            };
            let actual = arena
                .paged_decode(
                    LAYER,
                    PagedKvDecodeArgs {
                        queries: &queries,
                        batch: &batch,
                        softmax_scale: 0.5,
                        softcap: None,
                    },
                )?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = dense_reference(
                &query_values,
                query_heads,
                &keys,
                &values,
                kv_heads as usize,
                2,
                1,
                &[vec![0, 1]],
                0.5,
                None,
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn paged_prefill_applies_per_query_window_and_softcap() -> Result<()> {
        let arena = CpuKvArena::new(config(DType::F32))?;
        let physical_slots = (0..3)
            .flat_map(|page| {
                (0..2).map(move |offset| KvSlotRef {
                    block: block(page),
                    offset,
                })
            })
            .collect::<Vec<_>>();
        let slot_map = arena.lower_slots(&physical_slots)?;
        let keys = vec![
            1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 1., -1., -1., 1., 2., 1., 2., 0., 0., 2., 0.,
            2., 2., 0.,
        ];
        let values = vec![1., 10., 2., 20., 3., 30., 4., 40., 5., 50., 6., 60.];
        let key_tensor = Tensor::from_vec(keys.clone(), (6, 2, 2), &Device::Cpu)?;
        let value_tensor = Tensor::from_vec(values.clone(), (6, 2, 1), &Device::Cpu)?;
        arena.write_slots(
            LAYER,
            KvWriteArgs {
                keys: &key_tensor,
                values: &value_tensor,
                slots: slot_map.as_ref(),
            },
        )?;

        let query_values = vec![
            1., 0., 0., 1., 1., 1., 1., -1., // row 0 query 0
            0.5, 1., 1., 0.5, -1., 1., 1., 1., // row 0 query 1
            1., 0.5, 0.5, 1., 1., -0.5, -0.5, 1., // row 1 query 0
        ];
        let queries = Tensor::from_vec(query_values.clone(), (3, 4, 2), &Device::Cpu)?;
        let rows = vec![
            PagedKvPrefillRow {
                blocks: vec![block(2), block(0)],
                first_page_offset: 1,
                query_start: 0,
                query_len: 2,
                context_len: 3,
            },
            PagedKvPrefillRow {
                blocks: vec![block(1)],
                first_page_offset: 0,
                query_start: 2,
                query_len: 1,
                context_len: 1,
            },
        ];
        let scale = 1.0 / 2.0f32.sqrt();
        let softcap = Some(0.75);
        let actual = arena
            .paged_prefill(
                LAYER,
                PagedKvPrefillArgs {
                    queries: &queries,
                    rows: &rows,
                    softmax_scale: scale,
                    softcap,
                    window_tokens: Some(2),
                },
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = dense_reference(
            &query_values,
            4,
            &keys,
            &values,
            2,
            2,
            1,
            &[vec![5, 0], vec![0, 1], vec![2]],
            scale,
            softcap,
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }
}

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::backends::BackendKind;
use crate::error::Error;
use crate::kv::{CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvLayerBinding, KvSlotRef};
use crate::Result;

#[cfg(feature = "cuda")]
use super::KvBackendRuntime;
use super::{
    DeviceFence, KvArena, KvArenaConfig, KvArenaOperationStats, KvDeviceFence, KvPageCopy,
    KvSlotMap, KvWriteArgs, KvWriteCompletion, PagedKvDecodeArgs,
};

/// Operations Candle 0.11 can execute without moving KV data through host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleAcceleratorKvSupport {
    pub in_place_zero: bool,
    pub device_page_copy: bool,
    pub in_place_slot_write: bool,
    pub direct_paged_attention: bool,
}

impl CandleAcceleratorKvSupport {
    pub const fn is_complete(self) -> bool {
        self.in_place_zero
            && self.device_page_copy
            && self.in_place_slot_write
            && self.direct_paged_attention
    }
}

/// Report managed-KV support compiled into this binary.
///
/// CUDA and Metal use izwi-native block-table kernels. CUDA builds may also
/// select Candle FlashAttention for compatible half-precision pages, but the
/// physical runtime does not depend on that optional optimization.
pub const fn candle_accelerator_kv_support(backend: BackendKind) -> CandleAcceleratorKvSupport {
    match backend {
        BackendKind::Cpu => CandleAcceleratorKvSupport {
            in_place_zero: false,
            device_page_copy: false,
            in_place_slot_write: false,
            direct_paged_attention: false,
        },
        BackendKind::Metal => CandleAcceleratorKvSupport {
            in_place_zero: cfg!(feature = "metal"),
            device_page_copy: cfg!(feature = "metal"),
            in_place_slot_write: cfg!(feature = "metal"),
            direct_paged_attention: cfg!(feature = "metal"),
        },
        BackendKind::Cuda => CandleAcceleratorKvSupport {
            in_place_zero: cfg!(feature = "cuda"),
            device_page_copy: cfg!(feature = "cuda"),
            in_place_slot_write: cfg!(feature = "cuda"),
            direct_paged_attention: cfg!(feature = "cuda"),
        },
    }
}

#[derive(Debug)]
struct AcceleratorFence {
    timeline: Arc<AcceleratorFenceTimeline>,
    target_epoch: u64,
}

impl KvDeviceFence for AcceleratorFence {
    fn is_complete(&self) -> bool {
        self.timeline.completed.load(Ordering::Acquire) >= self.target_epoch
    }

    fn wait(&self) -> Result<()> {
        if !self.is_complete() {
            self.timeline.synchronize_issued()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct AcceleratorFenceTimeline {
    device: Device,
    issued: AtomicU64,
    completed: AtomicU64,
    synchronization_lock: Mutex<()>,
    host_synchronizations: Arc<AtomicU64>,
}

impl AcceleratorFenceTimeline {
    fn issue(self: &Arc<Self>) -> Result<DeviceFence> {
        let _guard = self.synchronization_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV fence timeline was poisoned".into())
        })?;
        let current = self.issued.load(Ordering::Acquire);
        let target_epoch = current
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("accelerator KV fence epoch overflow".into()))?;
        self.issued.store(target_epoch, Ordering::Release);
        Ok(Arc::new(AcceleratorFence {
            timeline: self.clone(),
            target_epoch,
        }))
    }

    fn synchronize_issued(&self) -> Result<()> {
        let _guard = self.synchronization_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV fence timeline was poisoned".into())
        })?;
        let issued = self.issued.load(Ordering::Acquire);
        if self.completed.load(Ordering::Acquire) >= issued {
            return Ok(());
        }
        self.device.synchronize()?;
        self.completed.store(issued, Ordering::Release);
        self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn drain(&self) -> Result<()> {
        let _guard = self.synchronization_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV fence timeline was poisoned".into())
        })?;
        self.device.synchronize()?;
        self.completed
            .store(self.issued.load(Ordering::Acquire), Ordering::Release);
        self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[derive(Debug)]
struct CompletedAcceleratorFence;

impl KvDeviceFence for CompletedAcceleratorFence {
    fn is_complete(&self) -> bool {
        true
    }

    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

fn completed_device_fence(device: &Device) -> Result<DeviceFence> {
    // Candle does not expose the current Metal command buffer as a clonable
    // completion token. Complete this mutation before publishing its fence so
    // coordinator commit/reuse never races queued private-buffer work.
    device.synchronize()?;
    Ok(Arc::new(CompletedAcceleratorFence))
}

#[derive(Debug)]
struct AcceleratorSlotMap {
    arena: KvArenaId,
    flat_slots: Vec<usize>,
    logical_slots: Arc<[KvSlotRef]>,
}

impl KvSlotMap for AcceleratorSlotMap {
    fn arena_id(&self) -> KvArenaId {
        self.arena
    }

    fn len(&self) -> usize {
        self.flat_slots.len()
    }

    fn logical_slots(&self) -> Arc<[KvSlotRef]> {
        self.logical_slots.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct AcceleratorLayerStorage {
    keys: Tensor,
    values: Tensor,
    num_kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

/// Device-resident KV storage backed by Candle's accelerator tensors.
///
/// `new_mutation_only` is deliberately explicit: it exposes the independently
/// useful write/copy/zero slice without claiming that a backend has direct
/// paged attention. Production allocation uses the feature-gated CUDA or Metal
/// runtime only when that backend's complete direct-attention path is compiled.
#[derive(Debug)]
pub struct CandleAcceleratorKvArena {
    config: KvArenaConfig,
    backend: BackendKind,
    device: Device,
    layers: HashMap<KvLayerBinding, AcceleratorLayerStorage>,
    mutation_lock: Mutex<()>,
    slot_write_dispatches: AtomicU64,
    paged_decode_dispatches: AtomicU64,
    page_zero_dispatches: AtomicU64,
    page_copy_dispatches: AtomicU64,
    host_synchronizations: Arc<AtomicU64>,
    fence_timeline: Option<Arc<AcceleratorFenceTimeline>>,
}

impl CandleAcceleratorKvArena {
    pub fn new_mutation_only(config: KvArenaConfig, device: Device) -> Result<Self> {
        let backend = backend_for_device(&device)?;
        validate_config(&config, backend, &device)?;
        let support = candle_accelerator_kv_support(backend);
        if !support.in_place_zero || !support.device_page_copy || !support.in_place_slot_write {
            return Err(Error::InferenceError(format!(
                "managed KV mutation support is not compiled for {backend:?}"
            )));
        }

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
                &device,
            )?;
            let values = Tensor::zeros(
                (common.0, common.1, common.2, layer.value_head_dim as usize),
                config.dtype,
                &device,
            )?;
            layers.insert(
                layer.binding,
                AcceleratorLayerStorage {
                    keys,
                    values,
                    num_kv_heads: common.2,
                    key_head_dim: layer.key_head_dim as usize,
                    value_head_dim: layer.value_head_dim as usize,
                },
            );
        }

        let host_synchronizations = Arc::new(AtomicU64::new(0));
        let fence_timeline = (backend == BackendKind::Cuda).then(|| {
            Arc::new(AcceleratorFenceTimeline {
                device: device.clone(),
                issued: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                synchronization_lock: Mutex::new(()),
                host_synchronizations: host_synchronizations.clone(),
            })
        });
        Ok(Self {
            config,
            backend,
            device,
            layers,
            mutation_lock: Mutex::new(()),
            slot_write_dispatches: AtomicU64::new(0),
            paged_decode_dispatches: AtomicU64::new(0),
            page_zero_dispatches: AtomicU64::new(0),
            page_copy_dispatches: AtomicU64::new(0),
            host_synchronizations,
            fence_timeline,
        })
    }

    pub fn layer_tensors(&self, binding: KvLayerBinding) -> Result<(Tensor, Tensor)> {
        let layer = self.layer(binding)?;
        Ok((layer.keys.clone(), layer.values.clone()))
    }

    fn layer(&self, binding: KvLayerBinding) -> Result<&AcceleratorLayerStorage> {
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
                "KV block index {page} is outside arena capacity {}",
                self.config.capacity_pages
            )));
        }
        if block.slot_generation == 0 {
            return Err(Error::InferenceError(
                "KV block has a zero slot generation".into(),
            ));
        }
        Ok(page)
    }

    fn mutation_fence(&self) -> Result<DeviceFence> {
        if self.backend == BackendKind::Metal {
            let fence = completed_device_fence(&self.device)?;
            self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
            Ok(fence)
        } else {
            self.fence_timeline
                .as_ref()
                .ok_or_else(|| Error::InferenceError("CUDA KV fence timeline is missing".into()))?
                .issue()
        }
    }

    fn accelerator_slots<'a>(&self, slots: &'a dyn KvSlotMap) -> Result<&'a AcceleratorSlotMap> {
        let slots = slots
            .as_any()
            .downcast_ref::<AcceleratorSlotMap>()
            .ok_or_else(|| {
                Error::InferenceError("KV slot map belongs to another backend".into())
            })?;
        if slots.arena != self.config.id {
            return Err(Error::InferenceError(format!(
                "KV slot map belongs to arena {:?}, expected {:?}",
                slots.arena, self.config.id
            )));
        }
        Ok(slots)
    }

    fn lower_decode_tables(
        &self,
        batch: &KvDecodeBatchMetadata,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, usize, usize)> {
        let batch_size = batch.sequences.len();
        let max_blocks = batch
            .sequences
            .iter()
            .map(|sequence| sequence.blocks.len())
            .max()
            .unwrap_or(0);
        if batch_size == 0 || max_blocks == 0 {
            return Err(Error::InferenceError(
                "accelerator paged decode requires a non-empty batch and block table".into(),
            ));
        }

        let mut table = vec![0_u32; batch_size * max_blocks];
        let mut cumulative = Vec::with_capacity(batch_size + 1);
        let mut first_page_offsets = Vec::with_capacity(batch_size);
        cumulative.push(0_u32);
        let mut total = 0_u32;
        let mut max_context = 0_usize;
        for (row, sequence) in batch.sequences.iter().enumerate() {
            if sequence.context_len == 0 {
                return Err(Error::InferenceError(format!(
                    "accelerator paged decode row {row} has an empty context"
                )));
            }
            if sequence.first_page_offset >= self.config.page_tokens {
                return Err(Error::InferenceError(format!(
                    "accelerator paged decode row {row} first-page offset {} exceeds page size {}",
                    sequence.first_page_offset, self.config.page_tokens
                )));
            }
            let physical_tokens = sequence
                .context_len
                .checked_add(sequence.first_page_offset)
                .ok_or_else(|| {
                    Error::InferenceError(
                        "accelerator paged decode physical token range exceeds u32".into(),
                    )
                })?;
            let required_pages =
                (physical_tokens as usize).div_ceil(self.config.page_tokens as usize);
            if sequence.blocks.len() != required_pages {
                return Err(Error::InferenceError(format!(
                    "accelerator paged decode row {row} has {} pages, expected {required_pages}",
                    sequence.blocks.len()
                )));
            }
            for (logical, block) in sequence.blocks.iter().copied().enumerate() {
                let physical = self.validate_block(block)?;
                table[row * max_blocks + logical] = u32::try_from(physical)
                    .map_err(|_| Error::InferenceError("KV page index exceeds u32".into()))?;
            }
            total = total.checked_add(sequence.context_len).ok_or_else(|| {
                Error::InferenceError("cumulative accelerator context length exceeds u32".into())
            })?;
            cumulative.push(total);
            first_page_offsets.push(sequence.first_page_offset);
            max_context = max_context.max(sequence.context_len as usize);
        }
        Ok((
            table,
            cumulative,
            first_page_offsets,
            max_blocks,
            max_context,
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_paged_decode(
        &self,
        layer: &AcceleratorLayerStorage,
        args: PagedKvDecodeArgs<'_>,
    ) -> Result<Tensor> {
        let batch_size = args.batch.sequences.len();
        let (table, seqlens_k, first_page_offsets, max_blocks, _max_context) =
            self.lower_decode_tables(args.batch)?;
        #[cfg(feature = "flash-attn")]
        if matches!(self.config.dtype, DType::F16 | DType::BF16)
            && self.config.page_tokens % 32 == 0
            && first_page_offsets.iter().all(|offset| *offset == 0)
        {
            let mut seqlens_q = Vec::with_capacity(batch_size + 1);
            for value in 0..=batch_size {
                seqlens_q.push(u32::try_from(value).map_err(|_| {
                    Error::InferenceError("CUDA paged decode batch exceeds u32".into())
                })?);
            }
            let seqlens_q = Tensor::from_vec(seqlens_q, batch_size + 1, &self.device)?;
            let seqlens_k = Tensor::from_vec(seqlens_k, batch_size + 1, &self.device)?;
            let block_table = Tensor::from_vec(table, (batch_size, max_blocks), &self.device)?;
            return Ok(candle_flash_attn::flash_attn_varlen_paged_windowed(
                args.queries,
                &layer.keys,
                &layer.values,
                &seqlens_q,
                &seqlens_k,
                &block_table,
                None,
                1,
                _max_context,
                args.softmax_scale,
                None,
                None,
                self.config.page_tokens as usize,
                None,
            )?);
        }

        let context_lens = seqlens_k
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        let mut metadata =
            Vec::with_capacity(context_lens.len() + first_page_offsets.len() + table.len());
        metadata.extend(context_lens);
        metadata.extend(first_page_offsets);
        metadata.extend(table);
        Ok(crate::kernels::cuda::paged_decode_attention(
            args.queries,
            &layer.keys,
            &layer.values,
            metadata,
            batch_size,
            args.queries.dims()[1],
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
        )?)
    }

    #[cfg(feature = "metal")]
    fn metal_paged_decode(
        &self,
        layer: &AcceleratorLayerStorage,
        args: PagedKvDecodeArgs<'_>,
    ) -> Result<Tensor> {
        let batch_size = args.batch.sequences.len();
        let num_heads = args.queries.dims()[1];
        let (table, cumulative, first_page_offsets, max_blocks, _) =
            self.lower_decode_tables(args.batch)?;
        let context_lens = cumulative
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        let mut metadata =
            Vec::with_capacity(context_lens.len() + first_page_offsets.len() + table.len());
        metadata.extend(context_lens);
        metadata.extend(first_page_offsets);
        metadata.extend(table);
        Ok(crate::kernels::metal::paged_decode_attention(
            args.queries,
            &layer.keys,
            &layer.values,
            metadata,
            batch_size,
            num_heads,
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
        )?)
    }
}

impl KvArena for CandleAcceleratorKvArena {
    fn id(&self) -> KvArenaId {
        self.config.id
    }

    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn device_location(&self) -> DeviceLocation {
        self.device.location()
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
            flat_slots.push(flat);
        }
        Ok(Arc::new(AcceleratorSlotMap {
            arena: self.config.id,
            flat_slots,
            logical_slots: Arc::from(slots),
        }))
    }

    fn zero_pages(&self, pages: &[CacheBlockRef]) -> Result<DeviceFence> {
        let page_indices = pages
            .iter()
            .copied()
            .map(|page| self.validate_block(page))
            .collect::<Result<Vec<_>>>()?;
        reject_duplicate_pages(&page_indices, "zero")?;
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        for layer in self.layers.values() {
            for &page in &page_indices {
                layer.keys.narrow(0, page, 1)?.zero_set()?;
                layer.values.narrow(0, page, 1)?.zero_set()?;
            }
        }
        self.page_zero_dispatches.fetch_add(1, Ordering::Relaxed);
        self.mutation_fence()
    }

    fn copy_pages(&self, copies: &[KvPageCopy]) -> Result<DeviceFence> {
        let mut lowered = Vec::with_capacity(copies.len());
        let mut destinations = HashSet::with_capacity(copies.len());
        for copy in copies {
            let source = self.validate_block(copy.source)?;
            let destination = self.validate_block(copy.destination)?;
            if !destinations.insert(destination) {
                return Err(Error::InferenceError(format!(
                    "KV page copy has duplicate destination page {destination}"
                )));
            }
            lowered.push((source, destination));
        }

        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        for layer in self.layers.values() {
            // Snapshot every source on device before any destination is changed;
            // this preserves parallel-copy semantics for chains and cycles.
            let key_sources = lowered
                .iter()
                .map(|(source, _)| layer.keys.narrow(0, *source, 1)?.copy())
                .collect::<candle_core::Result<Vec<_>>>()?;
            let value_sources = lowered
                .iter()
                .map(|(source, _)| layer.values.narrow(0, *source, 1)?.copy())
                .collect::<candle_core::Result<Vec<_>>>()?;
            for (((_, destination), keys), values) in lowered
                .iter()
                .zip(key_sources.iter())
                .zip(value_sources.iter())
            {
                layer.keys.slice_set(keys, 0, *destination)?;
                layer.values.slice_set(values, 0, *destination)?;
            }
        }
        self.page_copy_dispatches.fetch_add(1, Ordering::Relaxed);
        self.mutation_fence()
    }

    fn write_slots(
        &self,
        binding: KvLayerBinding,
        args: KvWriteArgs<'_>,
    ) -> Result<KvWriteCompletion> {
        let slots = self.accelerator_slots(args.slots)?;
        let layer = self.layer(binding)?;
        validate_write_tensor(
            args.keys,
            slots.len(),
            layer.num_kv_heads,
            layer.key_head_dim,
            self.config.dtype,
            &self.device,
            "key",
        )?;
        validate_write_tensor(
            args.values,
            slots.len(),
            layer.num_kv_heads,
            layer.value_head_dim,
            self.config.dtype,
            &self.device,
            "value",
        )?;

        let flat_capacity = (self.config.capacity_pages as usize)
            .checked_mul(self.config.page_tokens as usize)
            .ok_or_else(|| Error::InferenceError("KV slot count overflow".into()))?;
        let flat_keys =
            layer
                .keys
                .reshape((flat_capacity, layer.num_kv_heads, layer.key_head_dim))?;
        let flat_values =
            layer
                .values
                .reshape((flat_capacity, layer.num_kv_heads, layer.value_head_dim))?;
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        for (token, &slot) in slots.flat_slots.iter().enumerate() {
            flat_keys.slice_set(&args.keys.narrow(0, token, 1)?, 0, slot)?;
            flat_values.slice_set(&args.values.narrow(0, token, 1)?, 0, slot)?;
        }
        self.slot_write_dispatches.fetch_add(1, Ordering::Relaxed);
        let fence = self.mutation_fence()?;
        Ok(KvWriteCompletion::new(
            self.config.id,
            binding,
            args.slots.logical_slots(),
            fence,
        ))
    }

    fn paged_decode(&self, binding: KvLayerBinding, args: PagedKvDecodeArgs<'_>) -> Result<Tensor> {
        if !candle_accelerator_kv_support(self.backend).direct_paged_attention {
            return Err(Error::InferenceError(format!(
                "direct paged attention is not compiled for {:?}",
                self.backend
            )));
        }
        let layer = self.layer(binding)?;
        validate_decode_query(layer, &args, self.config.dtype, &self.device, self.backend)?;
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;

        #[cfg(feature = "cuda")]
        if self.backend == BackendKind::Cuda {
            let output = self.cuda_paged_decode(layer, args)?;
            self.paged_decode_dispatches.fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        #[cfg(feature = "metal")]
        if self.backend == BackendKind::Metal {
            let output = self.metal_paged_decode(layer, args)?;
            self.paged_decode_dispatches.fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        Err(Error::InferenceError(format!(
            "direct paged attention is unavailable for {:?}",
            self.backend
        )))
    }

    fn operation_stats(&self) -> KvArenaOperationStats {
        KvArenaOperationStats {
            slot_write_dispatches: self.slot_write_dispatches.load(Ordering::Relaxed),
            paged_decode_dispatches: self.paged_decode_dispatches.load(Ordering::Relaxed),
            page_zero_dispatches: self.page_zero_dispatches.load(Ordering::Relaxed),
            page_copy_dispatches: self.page_copy_dispatches.load(Ordering::Relaxed),
            host_synchronizations: self.host_synchronizations.load(Ordering::Relaxed),
        }
    }

    fn drain(&self) -> Result<()> {
        if let Some(timeline) = self.fence_timeline.as_ref() {
            timeline.drain()
        } else {
            self.device.synchronize()?;
            self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
}

/// Complete managed CUDA runtime backed by native block-table attention.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaKvBackendRuntime {
    device: Device,
}

/// Complete managed Metal runtime backed by native block-table MSL attention.
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalKvBackendRuntime {
    device: Device,
}

#[cfg(feature = "metal")]
impl MetalKvBackendRuntime {
    pub fn new(device: Device) -> Result<Self> {
        if !device.is_metal() {
            return Err(Error::InvalidInput(
                "Metal KV runtime requires a Metal device".into(),
            ));
        }
        Ok(Self { device })
    }
}

#[cfg(feature = "metal")]
impl super::KvBackendRuntime for MetalKvBackendRuntime {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>> {
        Ok(Arc::new(CandleAcceleratorKvArena::new_mutation_only(
            config,
            self.device.clone(),
        )?))
    }
}

#[cfg(feature = "cuda")]
impl CudaKvBackendRuntime {
    pub fn new(device: Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(Error::InvalidInput(
                "CUDA KV runtime requires a CUDA device".into(),
            ));
        }
        Ok(Self { device })
    }
}

#[cfg(feature = "cuda")]
impl KvBackendRuntime for CudaKvBackendRuntime {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn allocate_arena(&self, config: KvArenaConfig) -> Result<Arc<dyn KvArena>> {
        Ok(Arc::new(CandleAcceleratorKvArena::new_mutation_only(
            config,
            self.device.clone(),
        )?))
    }
}

fn backend_for_device(device: &Device) -> Result<BackendKind> {
    if device.is_cuda() {
        Ok(BackendKind::Cuda)
    } else if device.is_metal() {
        Ok(BackendKind::Metal)
    } else {
        Err(Error::InvalidInput(
            "accelerator KV arena requires a CUDA or Metal device".into(),
        ))
    }
}

fn validate_config(config: &KvArenaConfig, backend: BackendKind, device: &Device) -> Result<()> {
    if config.id.backend != backend {
        return Err(Error::InferenceError(format!(
            "KV arena id targets {:?}, but storage device is {backend:?}",
            config.id.backend
        )));
    }
    match device.location() {
        DeviceLocation::Cuda { gpu_id }
            if config.id.device_ordinal != u32::try_from(gpu_id).ok() =>
        {
            return Err(Error::InferenceError(
                "CUDA KV arena has an invalid device ordinal".into(),
            ));
        }
        // Candle exposes Metal's registry id as its DeviceLocation rather than
        // the selector ordinal accepted by Device::new_metal. Require the
        // resolved ordinal to be explicit, but do not compare unlike ids.
        DeviceLocation::Metal { .. } if config.id.device_ordinal.is_none() => {
            return Err(Error::InferenceError(
                "Metal KV arena requires an explicit device ordinal".into(),
            ));
        }
        DeviceLocation::Cpu => unreachable!(),
        DeviceLocation::Cuda { .. } | DeviceLocation::Metal { .. } => {}
    }
    if config.page_tokens == 0 || config.capacity_pages == 0 {
        return Err(Error::InferenceError(
            "accelerator KV arena page size and capacity must be non-zero".into(),
        ));
    }
    if !matches!(config.dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(Error::InferenceError(format!(
            "accelerator KV arena does not support {:?} storage",
            config.dtype
        )));
    }
    let direct_cuda = backend == BackendKind::Cuda
        && candle_accelerator_kv_support(backend).direct_paged_attention;
    let direct_metal = backend == BackendKind::Metal
        && candle_accelerator_kv_support(backend).direct_paged_attention;
    if config.layers.is_empty() {
        return Err(Error::InferenceError(
            "accelerator KV arena must contain at least one layer".into(),
        ));
    }
    let total_slots = u64::from(config.page_tokens)
        .checked_mul(u64::from(config.capacity_pages))
        .ok_or_else(|| Error::InferenceError("accelerator KV slot count overflow".into()))?;
    if total_slots > u64::from(u32::MAX) {
        return Err(Error::InferenceError(format!(
            "accelerator KV arena has {total_slots} slots, exceeding the u32 slot ABI"
        )));
    }
    let mut bindings = HashSet::with_capacity(config.layers.len());
    for layer in &config.layers {
        if !bindings.insert(layer.binding) {
            return Err(Error::InferenceError(format!(
                "accelerator KV arena contains duplicate layer binding {}",
                layer.binding.physical_layer
            )));
        }
        if layer.num_kv_heads == 0 || layer.key_head_dim == 0 || layer.value_head_dim == 0 {
            return Err(Error::InferenceError(format!(
                "accelerator KV layer {} has zero-sized geometry",
                layer.binding.physical_layer
            )));
        }
        if direct_cuda && (layer.key_head_dim != layer.value_head_dim || layer.key_head_dim > 512) {
            return Err(Error::InferenceError(format!(
                "CUDA paged attention requires equal K/V dimensions at most 512; layer {} has K={} V={}",
                layer.binding.physical_layer, layer.key_head_dim, layer.value_head_dim
            )));
        }
        if direct_metal
            && (!matches!(config.dtype, DType::F16 | DType::F32)
                || layer.key_head_dim > 512
                || layer.value_head_dim > 512)
        {
            return Err(Error::InferenceError(format!(
                "Metal paged attention requires F16/F32 storage and head dimensions at most 512; layer {} has dtype {:?}, K={} V={}",
                layer.binding.physical_layer,
                config.dtype,
                layer.key_head_dim,
                layer.value_head_dim
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
    device: &Device,
    label: &str,
) -> Result<()> {
    if tensor.device().location() != device.location() {
        return Err(Error::InferenceError(format!(
            "accelerator KV {label} source is on the wrong device"
        )));
    }
    if tensor.dtype() != dtype {
        return Err(Error::InferenceError(format!(
            "accelerator KV {label} source dtype {:?} does not match arena dtype {dtype:?}",
            tensor.dtype()
        )));
    }
    let expected = [tokens, heads, head_dim];
    if tensor.dims() != expected {
        return Err(Error::InferenceError(format!(
            "accelerator KV {label} source shape {:?} does not match {expected:?}",
            tensor.dims()
        )));
    }
    if !tensor.layout().is_contiguous() {
        return Err(Error::InferenceError(format!(
            "accelerator KV {label} source must be contiguous"
        )));
    }
    Ok(())
}

fn validate_decode_query(
    layer: &AcceleratorLayerStorage,
    args: &PagedKvDecodeArgs<'_>,
    dtype: DType,
    device: &Device,
    backend: BackendKind,
) -> Result<()> {
    if args.queries.device().location() != device.location() {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged decode queries are on the wrong device"
        )));
    }
    if args.queries.dtype() != dtype {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged decode query dtype {:?} does not match arena dtype {dtype:?}",
            args.queries.dtype()
        )));
    }
    if !args.queries.layout().is_contiguous() {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged decode queries must be contiguous"
        )));
    }
    let dims = args.queries.dims();
    if dims.len() != 3 || dims[0] != args.batch.sequences.len() || dims[2] != layer.key_head_dim {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged decode query shape {dims:?} does not match batch {} and head dim {}",
            args.batch.sequences.len(),
            layer.key_head_dim
        )));
    }
    if dims[1] == 0 || dims[1] % layer.num_kv_heads != 0 {
        return Err(Error::InferenceError(format!(
            "{backend:?} query heads {} are not divisible by KV heads {}",
            dims[1], layer.num_kv_heads
        )));
    }
    if backend == BackendKind::Cuda && layer.key_head_dim != layer.value_head_dim {
        return Err(Error::InferenceError(format!(
            "{backend:?} direct paged attention requires equal K/V head dimensions"
        )));
    }
    if !args.softmax_scale.is_finite() || args.softmax_scale <= 0.0 {
        return Err(Error::InferenceError(
            "paged decode softmax scale must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn reject_duplicate_pages(pages: &[usize], operation: &str) -> Result<()> {
    let mut unique = HashSet::with_capacity(pages.len());
    for &page in pages {
        if !unique.insert(page) {
            return Err(Error::InferenceError(format!(
                "KV page {operation} repeats page {page}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelInstanceId;
    use crate::kv::KvGroupId;

    #[test]
    fn support_matrix_matches_compiled_direct_attention() {
        let metal = candle_accelerator_kv_support(BackendKind::Metal);
        assert_eq!(metal.direct_paged_attention, cfg!(feature = "metal"));
        assert_eq!(metal.is_complete(), cfg!(feature = "metal"));

        let cuda = candle_accelerator_kv_support(BackendKind::Cuda);
        assert_eq!(cuda.direct_paged_attention, cfg!(feature = "cuda"));
        assert_eq!(cuda.is_complete(), cfg!(feature = "cuda"));
    }

    #[test]
    fn fence_timeline_coalesces_all_issued_mutations_into_one_wait() -> Result<()> {
        let host_synchronizations = Arc::new(AtomicU64::new(0));
        let timeline = Arc::new(AcceleratorFenceTimeline {
            device: Device::Cpu,
            issued: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            synchronization_lock: Mutex::new(()),
            host_synchronizations: host_synchronizations.clone(),
        });
        let first = timeline.issue()?;
        let second = timeline.issue()?;
        let third = timeline.issue()?;
        assert!(!first.is_complete());
        assert!(!third.is_complete());

        first.wait()?;
        assert!(first.is_complete());
        assert!(second.is_complete());
        assert!(third.is_complete());
        assert_eq!(host_synchronizations.load(Ordering::Relaxed), 1);

        second.wait()?;
        third.wait()?;
        assert_eq!(host_synchronizations.load(Ordering::Relaxed), 1);
        timeline.drain()?;
        assert_eq!(host_synchronizations.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_paged_decode_matches_cpu_for_ragged_shuffled_mha_gqa_mqa() -> Result<()> {
        // Candle 0.11 panics inside Device::new_metal when Metal reports an
        // empty device list, so feature-only CI must guard both failure modes.
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return Ok(());
        };
        for dtype in [DType::F32, DType::F16] {
            for (num_kv_heads, num_query_heads) in [(1_usize, 2_usize), (2, 2), (2, 4)] {
                let binding = KvLayerBinding {
                    model_layer: 0,
                    physical_layer: 0,
                };
                let metal_arena_id = KvArenaId {
                    model_instance: ModelInstanceId::new(1),
                    backend: BackendKind::Metal,
                    device_ordinal: Some(0),
                    generation: 1,
                };
                let cpu_arena_id = KvArenaId {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    ..metal_arena_id
                };
                let group = KvGroupId::new(0);
                let layer_config = super::super::KvLayerConfig {
                    binding,
                    num_kv_heads: num_kv_heads as u32,
                    key_head_dim: 2,
                    value_head_dim: 3,
                };
                let metal_config = KvArenaConfig {
                    id: metal_arena_id,
                    group,
                    page_tokens: 2,
                    capacity_pages: 4,
                    dtype,
                    layers: vec![layer_config],
                };
                let cpu_config = KvArenaConfig {
                    id: cpu_arena_id,
                    ..metal_config.clone()
                };
                let metal_arena =
                    CandleAcceleratorKvArena::new_mutation_only(metal_config, device.clone())?;
                let cpu_arena = super::super::CpuKvArena::new(cpu_config)?;
                let metal_block = |index| CacheBlockRef {
                    arena: metal_arena_id,
                    group,
                    index,
                    slot_generation: 1,
                };
                let cpu_block = |index| CacheBlockRef {
                    arena: cpu_arena_id,
                    group,
                    index,
                    slot_generation: 1,
                };
                let metal_slot_refs = (0..4)
                    .flat_map(|page| {
                        (0..2).map(move |offset| KvSlotRef {
                            block: metal_block(page),
                            offset,
                        })
                    })
                    .collect::<Vec<_>>();
                let cpu_slot_refs = (0..4)
                    .flat_map(|page| {
                        (0..2).map(move |offset| KvSlotRef {
                            block: cpu_block(page),
                            offset,
                        })
                    })
                    .collect::<Vec<_>>();
                let metal_slots = metal_arena.lower_slots(&metal_slot_refs)?;
                let cpu_slots = cpu_arena.lower_slots(&cpu_slot_refs)?;
                let key_data = (0..(8 * num_kv_heads * 2))
                    .map(|index| (index as f32 - 7.0) / 5.0)
                    .collect::<Vec<_>>();
                let value_data = (0..(8 * num_kv_heads * 3))
                    .map(|index| (index as f32 + 1.0) / 7.0)
                    .collect::<Vec<_>>();
                let metal_keys = Tensor::from_vec(key_data.clone(), (8, num_kv_heads, 2), &device)?
                    .to_dtype(dtype)?;
                let metal_values =
                    Tensor::from_vec(value_data.clone(), (8, num_kv_heads, 3), &device)?
                        .to_dtype(dtype)?;
                let cpu_keys = Tensor::from_vec(key_data, (8, num_kv_heads, 2), &Device::Cpu)?
                    .to_dtype(dtype)?;
                let cpu_values = Tensor::from_vec(value_data, (8, num_kv_heads, 3), &Device::Cpu)?
                    .to_dtype(dtype)?;
                let fence = metal_arena.write_slots(
                    binding,
                    KvWriteArgs {
                        keys: &metal_keys,
                        values: &metal_values,
                        slots: metal_slots.as_ref(),
                    },
                )?;
                assert!(fence.is_complete());
                cpu_arena.write_slots(
                    binding,
                    KvWriteArgs {
                        keys: &cpu_keys,
                        values: &cpu_values,
                        slots: cpu_slots.as_ref(),
                    },
                )?;

                let metal_batch = KvDecodeBatchMetadata {
                    sequences: vec![
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![metal_block(2), metal_block(0)],
                            first_page_offset: 1,
                            context_len: 3,
                        },
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![metal_block(3)],
                            first_page_offset: 0,
                            context_len: 2,
                        },
                    ],
                };
                let cpu_batch = KvDecodeBatchMetadata {
                    sequences: vec![
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cpu_block(2), cpu_block(0)],
                            first_page_offset: 1,
                            context_len: 3,
                        },
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cpu_block(3)],
                            first_page_offset: 0,
                            context_len: 2,
                        },
                    ],
                };
                let query_data = (0..(2 * num_query_heads * 2))
                    .map(|index| (index as f32 - 3.0) / 4.0)
                    .collect::<Vec<_>>();
                let metal_query =
                    Tensor::from_vec(query_data.clone(), (2, num_query_heads, 2), &device)?
                        .to_dtype(dtype)?;
                let cpu_query =
                    Tensor::from_vec(query_data, (2, num_query_heads, 2), &Device::Cpu)?
                        .to_dtype(dtype)?;
                let metal_output = metal_arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &metal_query,
                        batch: &metal_batch,
                        softmax_scale: 0.5,
                    },
                )?;
                let cpu_output = cpu_arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &cpu_query,
                        batch: &cpu_batch,
                        softmax_scale: 0.5,
                    },
                )?;
                let metal_values = metal_output
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let cpu_values = cpu_output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(metal_values.len(), cpu_values.len());
                let tolerance = if dtype == DType::F16 { 3e-3 } else { 1e-5 };
                for (actual, expected) in metal_values.iter().zip(cpu_values.iter()) {
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "{dtype:?} {num_query_heads}Q/{num_kv_heads}KV: {actual} != {expected}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_paged_decode_matches_cpu_for_offsets_and_gqa() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (num_kv_heads, num_query_heads) in [(1_usize, 2_usize), (2, 2), (2, 4)] {
                let binding = KvLayerBinding {
                    model_layer: 0,
                    physical_layer: 0,
                };
                let cuda_arena_id = KvArenaId {
                    model_instance: ModelInstanceId::new(1),
                    backend: BackendKind::Cuda,
                    device_ordinal: Some(0),
                    generation: 1,
                };
                let cpu_arena_id = KvArenaId {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    ..cuda_arena_id
                };
                let group = KvGroupId::new(0);
                let layer_config = super::super::KvLayerConfig {
                    binding,
                    num_kv_heads: num_kv_heads as u32,
                    key_head_dim: 4,
                    value_head_dim: 4,
                };
                let cuda_config = KvArenaConfig {
                    id: cuda_arena_id,
                    group,
                    page_tokens: 2,
                    capacity_pages: 4,
                    dtype,
                    layers: vec![layer_config],
                };
                let cpu_config = KvArenaConfig {
                    id: cpu_arena_id,
                    ..cuda_config.clone()
                };
                let cuda_arena =
                    CandleAcceleratorKvArena::new_mutation_only(cuda_config, device.clone())?;
                let cpu_arena = super::super::CpuKvArena::new(cpu_config)?;
                let cuda_block = |index| CacheBlockRef {
                    arena: cuda_arena_id,
                    group,
                    index,
                    slot_generation: 1,
                };
                let cpu_block = |index| CacheBlockRef {
                    arena: cpu_arena_id,
                    group,
                    index,
                    slot_generation: 1,
                };
                let cuda_slots = (0..4)
                    .flat_map(|page| {
                        (0..2).map(move |offset| KvSlotRef {
                            block: cuda_block(page),
                            offset,
                        })
                    })
                    .collect::<Vec<_>>();
                let cpu_slots = (0..4)
                    .flat_map(|page| {
                        (0..2).map(move |offset| KvSlotRef {
                            block: cpu_block(page),
                            offset,
                        })
                    })
                    .collect::<Vec<_>>();
                let cuda_slots = cuda_arena.lower_slots(&cuda_slots)?;
                let cpu_slots = cpu_arena.lower_slots(&cpu_slots)?;
                let key_data = (0..(8 * num_kv_heads * 4))
                    .map(|index| (index as f32 - 9.0) / 11.0)
                    .collect::<Vec<_>>();
                let value_data = (0..(8 * num_kv_heads * 4))
                    .map(|index| (index as f32 + 2.0) / 13.0)
                    .collect::<Vec<_>>();
                let cuda_keys = Tensor::from_vec(key_data.clone(), (8, num_kv_heads, 4), &device)?
                    .to_dtype(dtype)?;
                let cuda_values =
                    Tensor::from_vec(value_data.clone(), (8, num_kv_heads, 4), &device)?
                        .to_dtype(dtype)?;
                let cpu_keys = Tensor::from_vec(key_data, (8, num_kv_heads, 4), &Device::Cpu)?
                    .to_dtype(dtype)?;
                let cpu_values = Tensor::from_vec(value_data, (8, num_kv_heads, 4), &Device::Cpu)?
                    .to_dtype(dtype)?;
                cuda_arena
                    .write_slots(
                        binding,
                        KvWriteArgs {
                            keys: &cuda_keys,
                            values: &cuda_values,
                            slots: cuda_slots.as_ref(),
                        },
                    )?
                    .wait()?;
                cpu_arena
                    .write_slots(
                        binding,
                        KvWriteArgs {
                            keys: &cpu_keys,
                            values: &cpu_values,
                            slots: cpu_slots.as_ref(),
                        },
                    )?
                    .wait()?;

                let cuda_batch = KvDecodeBatchMetadata {
                    sequences: vec![
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cuda_block(2), cuda_block(0)],
                            first_page_offset: 1,
                            context_len: 3,
                        },
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cuda_block(3)],
                            first_page_offset: 0,
                            context_len: 2,
                        },
                    ],
                };
                let cpu_batch = KvDecodeBatchMetadata {
                    sequences: vec![
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cpu_block(2), cpu_block(0)],
                            first_page_offset: 1,
                            context_len: 3,
                        },
                        crate::kv::KvSequenceBlockTable {
                            blocks: vec![cpu_block(3)],
                            first_page_offset: 0,
                            context_len: 2,
                        },
                    ],
                };
                let query_data = (0..(2 * num_query_heads * 4))
                    .map(|index| (index as f32 - 5.0) / 7.0)
                    .collect::<Vec<_>>();
                let cuda_query =
                    Tensor::from_vec(query_data.clone(), (2, num_query_heads, 4), &device)?
                        .to_dtype(dtype)?;
                let cpu_query =
                    Tensor::from_vec(query_data, (2, num_query_heads, 4), &Device::Cpu)?
                        .to_dtype(dtype)?;
                let cuda_output = cuda_arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &cuda_query,
                        batch: &cuda_batch,
                        softmax_scale: 0.5,
                    },
                )?;
                let cpu_output = cpu_arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &cpu_query,
                        batch: &cpu_batch,
                        softmax_scale: 0.5,
                    },
                )?;
                let actual = cuda_output
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = cpu_output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let tolerance = match dtype {
                    DType::F32 => 1e-5,
                    DType::F16 => 3e-3,
                    DType::BF16 => 2e-2,
                    _ => unreachable!(),
                };
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "{dtype:?} {num_query_heads}Q/{num_kv_heads}KV: {actual} != {expected}"
                    );
                }
            }
        }
        Ok(())
    }
}

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::backends::BackendKind;
use crate::error::Error;
use crate::kv::{CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvLayerBinding, KvSlotRef};
use crate::Result;

#[cfg(feature = "flash-attn")]
use super::KvBackendRuntime;
use super::{
    DeviceFence, KvArena, KvArenaConfig, KvDeviceFence, KvPageCopy, KvPagedDecodeArgs, KvSlotMap,
    KvWriteArgs,
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
/// Metal intentionally reports an incomplete implementation: Candle provides
/// device-resident mutation primitives, but no kernel that consumes the page
/// table directly. CUDA becomes complete only with `flash-attn`, whose Candle
/// 0.11 binding includes variable-length paged attention.
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
            direct_paged_attention: false,
        },
        BackendKind::Cuda => CandleAcceleratorKvSupport {
            in_place_zero: cfg!(feature = "cuda"),
            device_page_copy: cfg!(feature = "cuda"),
            in_place_slot_write: cfg!(feature = "cuda"),
            direct_paged_attention: cfg!(feature = "flash-attn"),
        },
    }
}

#[derive(Debug)]
struct AcceleratorFence {
    device: Device,
    complete: AtomicBool,
}

impl KvDeviceFence for AcceleratorFence {
    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    fn wait(&self) -> Result<()> {
        if !self.is_complete() {
            self.device.synchronize()?;
            self.complete.store(true, Ordering::Release);
        }
        Ok(())
    }
}

fn device_fence(device: &Device) -> DeviceFence {
    Arc::new(AcceleratorFence {
        device: device.clone(),
        complete: AtomicBool::new(false),
    })
}

#[derive(Debug)]
struct AcceleratorSlotMap {
    arena: KvArenaId,
    flat_slots: Vec<usize>,
}

impl KvSlotMap for AcceleratorSlotMap {
    fn arena_id(&self) -> KvArenaId {
        self.arena
    }

    fn len(&self) -> usize {
        self.flat_slots.len()
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
/// paged attention. Production allocation uses `CudaKvBackendRuntime`, which
/// is available only when the complete CUDA path is compiled.
#[derive(Debug)]
pub struct CandleAcceleratorKvArena {
    config: KvArenaConfig,
    backend: BackendKind,
    device: Device,
    layers: HashMap<KvLayerBinding, AcceleratorLayerStorage>,
    mutation_lock: Mutex<()>,
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

        Ok(Self {
            config,
            backend,
            device,
            layers,
            mutation_lock: Mutex::new(()),
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
    ) -> Result<(Vec<u32>, Vec<u32>, usize, usize)> {
        let batch_size = batch.sequences.len();
        let max_blocks = batch
            .sequences
            .iter()
            .map(|sequence| sequence.blocks.len())
            .max()
            .unwrap_or(0);
        if batch_size == 0 || max_blocks == 0 {
            return Err(Error::InferenceError(
                "CUDA paged decode requires a non-empty batch and block table".into(),
            ));
        }

        let mut table = vec![0_u32; batch_size * max_blocks];
        let mut cumulative = Vec::with_capacity(batch_size + 1);
        cumulative.push(0_u32);
        let mut total = 0_u32;
        let mut max_context = 0_usize;
        for (row, sequence) in batch.sequences.iter().enumerate() {
            if sequence.context_len == 0 {
                return Err(Error::InferenceError(format!(
                    "CUDA paged decode row {row} has an empty context"
                )));
            }
            let required_pages =
                (sequence.context_len as usize).div_ceil(self.config.page_tokens as usize);
            if sequence.blocks.len() != required_pages {
                return Err(Error::InferenceError(format!(
                    "CUDA paged decode row {row} has {} pages, expected {required_pages}",
                    sequence.blocks.len()
                )));
            }
            for (logical, block) in sequence.blocks.iter().copied().enumerate() {
                let physical = self.validate_block(block)?;
                table[row * max_blocks + logical] = u32::try_from(physical)
                    .map_err(|_| Error::InferenceError("KV page index exceeds u32".into()))?;
            }
            total = total.checked_add(sequence.context_len).ok_or_else(|| {
                Error::InferenceError("CUDA cumulative context length exceeds u32".into())
            })?;
            cumulative.push(total);
            max_context = max_context.max(sequence.context_len as usize);
        }
        Ok((table, cumulative, max_blocks, max_context))
    }

    #[cfg(feature = "flash-attn")]
    fn cuda_paged_decode(
        &self,
        layer: &AcceleratorLayerStorage,
        args: KvPagedDecodeArgs<'_>,
    ) -> Result<Tensor> {
        let batch_size = args.batch.sequences.len();
        let (table, seqlens_k, max_blocks, max_context) = self.lower_decode_tables(args.batch)?;
        let mut seqlens_q = Vec::with_capacity(batch_size + 1);
        for value in 0..=batch_size {
            seqlens_q.push(u32::try_from(value).map_err(|_| {
                Error::InferenceError("CUDA paged decode batch exceeds u32".into())
            })?);
        }
        let seqlens_q = Tensor::from_vec(seqlens_q, batch_size + 1, &self.device)?;
        let seqlens_k = Tensor::from_vec(seqlens_k, batch_size + 1, &self.device)?;
        let block_table = Tensor::from_vec(table, (batch_size, max_blocks), &self.device)?;

        Ok(candle_flash_attn::flash_attn_varlen_paged_windowed(
            args.queries,
            &layer.keys,
            &layer.values,
            &seqlens_q,
            &seqlens_k,
            &block_table,
            None,
            1,
            max_context,
            args.softmax_scale,
            None,
            None,
            self.config.page_tokens as usize,
            None,
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
        Ok(device_fence(&self.device))
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
        Ok(device_fence(&self.device))
    }

    fn write_slots(&self, binding: KvLayerBinding, args: KvWriteArgs<'_>) -> Result<DeviceFence> {
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
        Ok(device_fence(&self.device))
    }

    fn paged_decode(&self, binding: KvLayerBinding, args: KvPagedDecodeArgs<'_>) -> Result<Tensor> {
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

        #[cfg(feature = "flash-attn")]
        if self.backend == BackendKind::Cuda {
            return self.cuda_paged_decode(layer, args);
        }

        Err(Error::InferenceError(format!(
            "direct paged attention is unavailable for {:?}",
            self.backend
        )))
    }
}

/// Complete managed CUDA runtime, available only with Candle's paged
/// flash-attention binding.
#[cfg(feature = "flash-attn")]
#[derive(Debug, Clone)]
pub struct CudaKvBackendRuntime {
    device: Device,
}

#[cfg(feature = "flash-attn")]
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

#[cfg(feature = "flash-attn")]
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
    if direct_cuda {
        if !matches!(config.dtype, DType::F16 | DType::BF16) {
            return Err(Error::InferenceError(
                "CUDA paged flash attention requires F16 or BF16 storage".into(),
            ));
        }
        if config.page_tokens % 32 != 0 {
            return Err(Error::InferenceError(
                "CUDA paged flash attention requires page size divisible by 32".into(),
            ));
        }
    }
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
        if direct_cuda
            && (layer.key_head_dim != layer.value_head_dim
                || layer.key_head_dim > 512
                || layer.key_head_dim % 8 != 0)
        {
            return Err(Error::InferenceError(format!(
                "CUDA paged flash attention cannot execute layer {} with K={} V={}",
                layer.binding.physical_layer, layer.key_head_dim, layer.value_head_dim
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
    args: &KvPagedDecodeArgs<'_>,
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
    if layer.key_head_dim != layer.value_head_dim {
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
    fn support_matrix_never_claims_metal_paged_attention() {
        let metal = candle_accelerator_kv_support(BackendKind::Metal);
        assert!(!metal.direct_paged_attention);
        assert!(!metal.is_complete());

        let cuda = candle_accelerator_kv_support(BackendKind::Cuda);
        assert_eq!(cuda.direct_paged_attention, cfg!(feature = "flash-attn"));
        assert_eq!(cuda.is_complete(), cfg!(feature = "flash-attn"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_mutations_are_device_resident_and_decode_fails_closed() -> Result<()> {
        // Candle 0.11 panics inside Device::new_metal when Metal reports an
        // empty device list, so feature-only CI must guard both failure modes.
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return Ok(());
        };
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(1),
            backend: BackendKind::Metal,
            device_ordinal: Some(0),
            generation: 1,
        };
        let group = KvGroupId::new(0);
        let config = KvArenaConfig {
            id: arena_id,
            group,
            page_tokens: 2,
            capacity_pages: 3,
            dtype: DType::F32,
            layers: vec![super::super::KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        };
        let arena = CandleAcceleratorKvArena::new_mutation_only(config, device.clone())?;
        let block = |index| CacheBlockRef {
            arena: arena_id,
            group,
            index,
            slot_generation: 1,
        };
        let slots = arena.lower_slots(&[
            KvSlotRef {
                block: block(0),
                offset: 0,
            },
            KvSlotRef {
                block: block(1),
                offset: 1,
            },
        ])?;
        let keys = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (2, 1, 2), &device)?;
        let values = Tensor::from_vec(vec![5_f32, 6., 7., 8.], (2, 1, 2), &device)?;
        arena
            .write_slots(
                binding,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )?
            .wait()?;
        arena
            .copy_pages(&[KvPageCopy {
                source: block(1),
                destination: block(2),
            }])?
            .wait()?;
        arena.zero_pages(&[block(0)])?.wait()?;

        let (stored_keys, _) = arena.layer_tensors(binding)?;
        let stored = stored_keys
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(&stored[0..4], &[0., 0., 0., 0.]);
        assert_eq!(&stored[4..8], &[0., 0., 3., 4.]);
        assert_eq!(&stored[8..12], &[0., 0., 3., 4.]);

        let query = Tensor::zeros((1, 1, 2), DType::F32, &device)?;
        let batch = KvDecodeBatchMetadata {
            sequences: vec![crate::kv::KvSequenceBlockTable {
                blocks: vec![block(2)],
                context_len: 1,
            }],
        };
        let error = arena
            .paged_decode(
                binding,
                KvPagedDecodeArgs {
                    queries: &query,
                    batch: &batch,
                    softmax_scale: 1.0,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("not compiled"));
        Ok(())
    }
}

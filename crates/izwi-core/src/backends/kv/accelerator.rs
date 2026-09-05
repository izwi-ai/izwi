use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use candle_core::{DType, Device, DeviceLocation, Tensor};

use crate::backends::BackendKind;
use crate::error::Error;
use crate::kv::{CacheBlockRef, KvArenaId, KvDecodeBatchMetadata, KvLayerBinding, KvSlotRef};
use crate::runtime::rollout::KvProviderRollout;
use crate::Result;

use super::cuda_tuning::{
    cuda_fp8_kv_evidence, observe_cuda_identity, resolve_cuda_kv_storage_format,
    resolve_cuda_paged_tuning, CudaDeviceIdentity, CudaKvStorageFormat, CudaPagedShapeKey,
};
#[cfg(feature = "cuda")]
use super::KvBackendRuntime;
use super::{
    DeviceFence, KvArena, KvArenaConfig, KvArenaGrowthPlan, KvArenaOperationStats,
    KvAttentionProvider, KvDeviceFence, KvPageCopy, KvSlotMap, KvWriteArgs, KvWriteCompletion,
    PagedKvDecodeArgs, PagedKvPrefillArgs, PagedKvPrefillRow,
};

/// Operations Candle 0.11 can execute without moving KV data through host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleAcceleratorKvSupport {
    pub in_place_zero: bool,
    pub device_page_copy: bool,
    pub in_place_slot_write: bool,
    pub direct_paged_attention: bool,
}

/// Resource telemetry for device-resident paged-attention metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandleAttentionPlanCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub device_uploads: u64,
    pub resident_bytes: u64,
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

/// Return whether Candle FlashAttention can consume this physical paged layout.
///
/// Keep every runtime call site on this predicate. In particular, Candle's
/// paged FA2 binding rejects head dimensions that are not multiples of eight;
/// those otherwise-valid arenas must use izwi's native paged kernel instead.
#[allow(dead_code)]
fn cuda_flash_paged_attention_eligible(
    dtype: DType,
    page_tokens: u32,
    key_head_dim: usize,
    value_head_dim: usize,
    all_first_page_offsets_zero: bool,
) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
        && page_tokens != 0
        && page_tokens.is_multiple_of(32)
        && all_first_page_offsets_zero
        && key_head_dim == value_head_dim
        && key_head_dim != 0
        && key_head_dim <= 512
        && key_head_dim.is_multiple_of(8)
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaEventFence {
    event: cudarc::driver::CudaEvent,
    host_synchronizations: Arc<AtomicU64>,
}

#[cfg(feature = "cuda")]
impl KvDeviceFence for CudaEventFence {
    fn is_complete(&self) -> bool {
        self.event.is_complete()
    }

    fn wait(&self) -> Result<()> {
        if !self.is_complete() {
            self.event.synchronize().map_err(|error| {
                Error::InferenceError(format!("CUDA KV completion event failed: {error}"))
            })?;
            self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
        }
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

#[derive(Clone)]
enum AcceleratorPageCleanliness {
    Clean,
    Dirty,
    ZeroPending(DeviceFence),
}

impl std::fmt::Debug for AcceleratorPageCleanliness {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Clean => "Clean",
            Self::Dirty => "Dirty",
            Self::ZeroPending(_) => "ZeroPending",
        })
    }
}

fn page_needs_zero(state: &mut AcceleratorPageCleanliness) -> bool {
    match state.clone() {
        AcceleratorPageCleanliness::Clean => false,
        AcceleratorPageCleanliness::Dirty => true,
        AcceleratorPageCleanliness::ZeroPending(fence) if fence.is_complete() => {
            *state = AcceleratorPageCleanliness::Clean;
            false
        }
        AcceleratorPageCleanliness::ZeroPending(_) => true,
    }
}

fn completed_device_fence(device: &Device) -> Result<DeviceFence> {
    // Candle does not expose the current Metal command buffer as a clonable
    // completion token. Complete this mutation before publishing its fence so
    // coordinator commit/reuse never races queued private-buffer work.
    device.synchronize()?;
    Ok(Arc::new(CompletedAcceleratorFence))
}

const ACCELERATOR_WORKSPACE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

fn zero_workspace_pages_per_chunk(
    page_count: usize,
    trailing_shape: &[usize],
    dtype: DType,
    per_shape_budget: usize,
) -> Result<usize> {
    let bytes_per_page =
        trailing_shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
                bytes.checked_mul(*dimension).ok_or_else(|| {
                    Error::Overloaded("accelerator zero workspace byte count overflow".into())
                })
            })?;
    let pages_per_chunk = page_count.min(per_shape_budget.checked_div(bytes_per_page).unwrap_or(0));
    if pages_per_chunk == 0 {
        return Err(Error::Overloaded(format!(
            "one accelerator KV page requires {bytes_per_page} zero-workspace bytes, exceeding per-shape budget {per_shape_budget}"
        )));
    }
    Ok(pages_per_chunk)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AcceleratorWorkspaceKey {
    shape: Vec<usize>,
    dtype: DType,
}

#[derive(Debug)]
struct AcceleratorWorkspaceLease {
    key: AcceleratorWorkspaceKey,
    tensor: Tensor,
    bytes: usize,
}

struct RetiredAcceleratorWorkspace {
    lease: AcceleratorWorkspaceLease,
    retirement: DeviceFence,
}

/// A bounded scratch pool whose entries are unavailable until their exact
/// submission token proves completion. The byte count includes checked-out and
/// retired storage, so allocation failure is deterministic rather than a
/// driver-OOM side effect.
struct AcceleratorWorkspacePool {
    budget_bytes: usize,
    reserved_bytes: usize,
    high_water_bytes: usize,
    allocations: u64,
    retired: Vec<RetiredAcceleratorWorkspace>,
}

impl std::fmt::Debug for AcceleratorWorkspacePool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorWorkspacePool")
            .field("budget_bytes", &self.budget_bytes)
            .field("reserved_bytes", &self.reserved_bytes)
            .field("high_water_bytes", &self.high_water_bytes)
            .field("allocations", &self.allocations)
            .field("retired_entries", &self.retired.len())
            .finish()
    }
}

impl AcceleratorWorkspacePool {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            reserved_bytes: 0,
            high_water_bytes: 0,
            allocations: 0,
            retired: Vec::new(),
        }
    }

    fn acquire(
        &mut self,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<AcceleratorWorkspaceLease> {
        let key = AcceleratorWorkspaceKey {
            shape: shape.to_vec(),
            dtype,
        };
        if let Some(index) = self
            .retired
            .iter()
            .position(|entry| entry.lease.key == key && entry.retirement.is_complete())
        {
            return Ok(self.retired.swap_remove(index).lease);
        }

        let elements = shape.iter().try_fold(1usize, |total, dimension| {
            total.checked_mul(*dimension).ok_or_else(|| {
                Error::Overloaded("accelerator workspace element count overflow".into())
            })
        })?;
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Overloaded("accelerator workspace byte count overflow".into()))?;
        while self
            .reserved_bytes
            .checked_add(bytes)
            .is_some_and(|total| total > self.budget_bytes)
        {
            let Some(index) = self
                .retired
                .iter()
                .position(|entry| entry.retirement.is_complete())
            else {
                break;
            };
            let retired = self.retired.swap_remove(index);
            self.reserved_bytes = self.reserved_bytes.saturating_sub(retired.lease.bytes);
        }
        let requested_total = self.reserved_bytes.checked_add(bytes).ok_or_else(|| {
            Error::Overloaded("accelerator workspace reservation overflow".into())
        })?;
        if requested_total > self.budget_bytes {
            return Err(Error::Overloaded(format!(
                "accelerator workspace budget exceeded: requested_bytes={bytes}, reserved_bytes={}, budget_bytes={}",
                self.reserved_bytes, self.budget_bytes
            )));
        }
        let tensor = Tensor::zeros(shape, dtype, device)?;
        self.reserved_bytes = requested_total;
        self.high_water_bytes = self.high_water_bytes.max(requested_total);
        self.allocations = self.allocations.saturating_add(1);
        Ok(AcceleratorWorkspaceLease { key, tensor, bytes })
    }

    fn retire(&mut self, lease: AcceleratorWorkspaceLease, retirement: DeviceFence) {
        self.retired
            .push(RetiredAcceleratorWorkspace { lease, retirement });
    }

    fn discard(&mut self, lease: AcceleratorWorkspaceLease) {
        self.reserved_bytes = self.reserved_bytes.saturating_sub(lease.bytes);
    }
}

#[derive(Debug)]
struct AcceleratorSlotMap {
    arena: KvArenaId,
    flat_slots: Vec<usize>,
    device_indices: Tensor,
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

fn resident_flat_slot_capacity(
    layer: &AcceleratorLayerStorage,
    resident_pages: usize,
    page_tokens: usize,
) -> Result<usize> {
    let key_pages = layer.keys.dim(0)?;
    let value_pages = layer.values.dim(0)?;
    if key_pages != resident_pages || value_pages != resident_pages {
        return Err(Error::InferenceError(format!(
            "accelerator KV backing/residency mismatch: resident_pages={resident_pages}, key_pages={key_pages}, value_pages={value_pages}"
        )));
    }
    resident_pages
        .checked_mul(page_tokens)
        .ok_or_else(|| Error::InferenceError("resident KV slot count overflow".into()))
}

#[derive(Debug)]
struct LoweredPrefillMetadata {
    cache_key: PrefillMetadataCacheKey,
    compact_rows: Vec<u32>,
    cumulative_queries: Vec<u32>,
    cumulative_contexts: Vec<u32>,
    block_table: Vec<u32>,
    sequence_count: usize,
    total_queries: usize,
    max_blocks: usize,
    max_query_len: usize,
    max_context_len: usize,
    all_first_page_offsets_zero: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrefillMetadataCacheKey {
    rows: Vec<PagedKvPrefillRow>,
    total_queries: usize,
}

#[derive(Debug)]
struct CachedPrefillDeviceMetadata {
    key: PrefillMetadataCacheKey,
    compact_rows: Tensor,
    cumulative_queries: Tensor,
    cumulative_contexts: Tensor,
    block_table: Tensor,
}

#[derive(Debug)]
struct PrefillDeviceMetadata {
    compact_rows: Tensor,
    cumulative_queries: Tensor,
    cumulative_contexts: Tensor,
    block_table: Tensor,
}

#[derive(Debug)]
struct CachedDecodeDeviceMetadata {
    key: KvDecodeBatchMetadata,
    host_cumulative_contexts: Vec<u32>,
    host_block_table: Vec<u32>,
    host_native_metadata: Vec<u32>,
    cumulative_queries: Tensor,
    cumulative_contexts: Tensor,
    block_table: Tensor,
    native_metadata: Tensor,
}

#[derive(Debug)]
struct DecodeDeviceMetadata {
    host_native_metadata: Vec<u32>,
    cumulative_queries: Tensor,
    cumulative_contexts: Tensor,
    block_table: Tensor,
    native_metadata: Tensor,
    max_blocks: usize,
    max_context: usize,
    all_first_page_offsets_zero: bool,
}

const ATTENTION_PLAN_CACHE_BYTES_PER_KIND: usize = 4 * 1024 * 1024;
const ATTENTION_PLAN_CACHE_ENTRIES_PER_KIND: usize = 32;

trait ResidentAttentionPlan {
    fn resident_bytes(&self) -> usize;
}

impl ResidentAttentionPlan for CachedPrefillDeviceMetadata {
    fn resident_bytes(&self) -> usize {
        self.cumulative_queries
            .elem_count()
            .saturating_add(self.compact_rows.elem_count())
            .saturating_add(self.cumulative_contexts.elem_count())
            .saturating_add(self.block_table.elem_count())
            .saturating_mul(std::mem::size_of::<u32>())
    }
}

impl CachedDecodeDeviceMetadata {
    fn device_metadata(&self) -> DecodeDeviceMetadata {
        DecodeDeviceMetadata {
            host_native_metadata: self.host_native_metadata.clone(),
            cumulative_queries: self.cumulative_queries.clone(),
            cumulative_contexts: self.cumulative_contexts.clone(),
            block_table: self.block_table.clone(),
            native_metadata: self.native_metadata.clone(),
            max_blocks: self.block_table.dims()[1],
            max_context: self
                .key
                .sequences
                .iter()
                .map(|sequence| sequence.context_len as usize)
                .max()
                .unwrap_or(0),
            all_first_page_offsets_zero: self
                .key
                .sequences
                .iter()
                .all(|sequence| sequence.first_page_offset == 0),
        }
    }

    fn shape_compatible(&self, batch_size: usize, max_blocks: usize) -> bool {
        self.key.sequences.len() == batch_size
            && self.block_table.dims() == [batch_size, max_blocks]
    }
}

impl ResidentAttentionPlan for CachedDecodeDeviceMetadata {
    fn resident_bytes(&self) -> usize {
        self.cumulative_queries
            .elem_count()
            .saturating_add(self.cumulative_contexts.elem_count())
            .saturating_add(self.block_table.elem_count())
            .saturating_add(self.native_metadata.elem_count())
            .saturating_mul(std::mem::size_of::<u32>())
    }
}

#[derive(Debug)]
struct ResidentAttentionPlanCache<T> {
    entries: VecDeque<T>,
    resident_bytes: usize,
}

impl<T> Default for ResidentAttentionPlanCache<T> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            resident_bytes: 0,
        }
    }
}

impl<T: ResidentAttentionPlan> ResidentAttentionPlanCache<T> {
    fn promote(&mut self, index: usize) -> Option<&T> {
        let entry = self.entries.remove(index)?;
        self.entries.push_back(entry);
        self.entries.back()
    }

    fn insert(&mut self, entry: T) -> usize {
        let entry_bytes = entry.resident_bytes();
        if entry_bytes > ATTENTION_PLAN_CACHE_BYTES_PER_KIND {
            return 0;
        }
        let mut evictions = 0;
        while self.entries.len() >= ATTENTION_PLAN_CACHE_ENTRIES_PER_KIND
            || self.resident_bytes.saturating_add(entry_bytes) > ATTENTION_PLAN_CACHE_BYTES_PER_KIND
        {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.resident_bytes = self.resident_bytes.saturating_sub(evicted.resident_bytes());
            evictions += 1;
        }
        self.resident_bytes = self.resident_bytes.saturating_add(entry_bytes);
        self.entries.push_back(entry);
        evictions
    }
}

fn update_resident_byte_gauge(gauge: &AtomicU64, before: usize, after: usize) {
    if after >= before {
        gauge.fetch_add((after - before) as u64, Ordering::Relaxed);
    } else {
        gauge.fetch_sub((before - after) as u64, Ordering::Relaxed);
    }
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
    optimized_provider_enabled: bool,
    cuda_identity: CudaDeviceIdentity,
    cuda_kv_storage: CudaKvStorageFormat,
    storage_dtype: DType,
    layers: RwLock<HashMap<KvLayerBinding, AcceleratorLayerStorage>>,
    resident_capacity_pages: AtomicU32,
    backing_generation: AtomicU64,
    clean_pages: Mutex<Vec<AcceleratorPageCleanliness>>,
    mutation_lock: Mutex<()>,
    prefill_metadata_cache: Mutex<ResidentAttentionPlanCache<CachedPrefillDeviceMetadata>>,
    decode_metadata_cache: Mutex<ResidentAttentionPlanCache<CachedDecodeDeviceMetadata>>,
    // Candle 0.11 does not expose a Metal command-buffer token or custom-op
    // output-buffer ABI. Pool only arena-owned mutation scratch here; native
    // attention partials/metadata must remain unpooled until that boundary can
    // return an exact retirement token.
    workspace_pool: Mutex<AcceleratorWorkspacePool>,
    attention_plan_cache_hits: AtomicU64,
    attention_plan_cache_misses: AtomicU64,
    attention_plan_cache_evictions: AtomicU64,
    attention_plan_device_uploads: AtomicU64,
    attention_plan_resident_bytes: AtomicU64,
    slot_write_dispatches: AtomicU64,
    paged_prefill_dispatches: AtomicU64,
    paged_decode_dispatches: AtomicU64,
    page_zero_dispatches: AtomicU64,
    page_copy_dispatches: AtomicU64,
    last_attention_provider: AtomicU64,
    portable_attention_dispatches: AtomicU64,
    cuda_native_attention_dispatches: AtomicU64,
    cuda_flash_attention_dispatches: AtomicU64,
    metal_native_attention_dispatches: AtomicU64,
    cuda_graph_warmups: AtomicU64,
    cuda_graph_captures: AtomicU64,
    cuda_graph_replays: AtomicU64,
    cuda_graph_fallbacks: AtomicU64,
    cuda_graph_backoff_hits: AtomicU64,
    cuda_graph_evictions: AtomicU64,
    host_synchronizations: Arc<AtomicU64>,
}

impl CandleAcceleratorKvArena {
    pub fn new_mutation_only(config: KvArenaConfig, device: Device) -> Result<Self> {
        let backend = backend_for_device(&device)?;
        let optimized_provider_enabled =
            KvProviderRollout::from_process_env()?.optimized_provider_enabled();
        let cuda_identity = if backend == BackendKind::Cuda {
            observe_cuda_identity(&device)
        } else {
            CudaDeviceIdentity::unobserved()
        };
        let cuda_kv_storage = if backend == BackendKind::Cuda {
            resolve_cuda_kv_storage_format(
                &cuda_identity,
                config.dtype,
                cuda_fp8_kv_evidence(&cuda_identity, config.dtype),
            )?
        } else {
            CudaKvStorageFormat::Dense
        };
        let storage_dtype = cuda_kv_storage.dtype(config.dtype);
        validate_config(&config, backend, &device)?;
        let support = candle_accelerator_kv_support(backend);
        if !support.in_place_zero || !support.device_page_copy || !support.in_place_slot_write {
            return Err(Error::InferenceError(format!(
                "managed KV mutation support is not compiled for {backend:?}"
            )));
        }

        let resident_capacity_pages = config
            .growth
            .map(|growth| growth.initial_pages)
            .unwrap_or(config.capacity_pages);
        let mut layers = HashMap::with_capacity(config.layers.len());
        for layer in &config.layers {
            let common = (
                resident_capacity_pages as usize,
                config.page_tokens as usize,
                layer.num_kv_heads as usize,
            );
            let keys = Tensor::zeros(
                (common.0, common.1, common.2, layer.key_head_dim as usize),
                storage_dtype,
                &device,
            )?;
            let values = Tensor::zeros(
                (common.0, common.1, common.2, layer.value_head_dim as usize),
                storage_dtype,
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
        Ok(Self {
            config,
            backend,
            device,
            optimized_provider_enabled,
            cuda_identity,
            cuda_kv_storage,
            storage_dtype,
            layers: RwLock::new(layers),
            resident_capacity_pages: AtomicU32::new(resident_capacity_pages),
            backing_generation: AtomicU64::new(1),
            clean_pages: Mutex::new(vec![
                AcceleratorPageCleanliness::Clean;
                resident_capacity_pages as usize
            ]),
            mutation_lock: Mutex::new(()),
            prefill_metadata_cache: Mutex::new(ResidentAttentionPlanCache::default()),
            decode_metadata_cache: Mutex::new(ResidentAttentionPlanCache::default()),
            workspace_pool: Mutex::new(AcceleratorWorkspacePool::new(
                ACCELERATOR_WORKSPACE_BUDGET_BYTES,
            )),
            attention_plan_cache_hits: AtomicU64::new(0),
            attention_plan_cache_misses: AtomicU64::new(0),
            attention_plan_cache_evictions: AtomicU64::new(0),
            attention_plan_device_uploads: AtomicU64::new(0),
            attention_plan_resident_bytes: AtomicU64::new(0),
            slot_write_dispatches: AtomicU64::new(0),
            paged_prefill_dispatches: AtomicU64::new(0),
            paged_decode_dispatches: AtomicU64::new(0),
            page_zero_dispatches: AtomicU64::new(0),
            page_copy_dispatches: AtomicU64::new(0),
            last_attention_provider: AtomicU64::new(0),
            portable_attention_dispatches: AtomicU64::new(0),
            cuda_native_attention_dispatches: AtomicU64::new(0),
            cuda_flash_attention_dispatches: AtomicU64::new(0),
            metal_native_attention_dispatches: AtomicU64::new(0),
            cuda_graph_warmups: AtomicU64::new(0),
            cuda_graph_captures: AtomicU64::new(0),
            cuda_graph_replays: AtomicU64::new(0),
            cuda_graph_fallbacks: AtomicU64::new(0),
            cuda_graph_backoff_hits: AtomicU64::new(0),
            cuda_graph_evictions: AtomicU64::new(0),
            host_synchronizations,
        })
    }

    pub fn layer_tensors(&self, binding: KvLayerBinding) -> Result<(Tensor, Tensor)> {
        let layers = self
            .layers
            .read()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        let layer = self.layer_from(&layers, binding)?;
        Ok((layer.keys.clone(), layer.values.clone()))
    }

    pub fn attention_plan_cache_stats(&self) -> CandleAttentionPlanCacheStats {
        CandleAttentionPlanCacheStats {
            hits: self.attention_plan_cache_hits.load(Ordering::Relaxed),
            misses: self.attention_plan_cache_misses.load(Ordering::Relaxed),
            evictions: self.attention_plan_cache_evictions.load(Ordering::Relaxed),
            device_uploads: self.attention_plan_device_uploads.load(Ordering::Relaxed),
            resident_bytes: self.attention_plan_resident_bytes.load(Ordering::Relaxed),
        }
    }

    fn record_attention_provider(&self, provider: KvAttentionProvider) {
        self.last_attention_provider
            .store(provider.code(), Ordering::Relaxed);
        let counter = match provider {
            KvAttentionProvider::Portable => &self.portable_attention_dispatches,
            KvAttentionProvider::CudaNative => &self.cuda_native_attention_dispatches,
            KvAttentionProvider::CudaFlashAttention => &self.cuda_flash_attention_dispatches,
            KvAttentionProvider::MetalNative => &self.metal_native_attention_dispatches,
            KvAttentionProvider::CpuReference => return,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn layer_from<'a>(
        &self,
        layers: &'a HashMap<KvLayerBinding, AcceleratorLayerStorage>,
        binding: KvLayerBinding,
    ) -> Result<&'a AcceleratorLayerStorage> {
        layers.get(&binding).ok_or_else(|| {
            Error::InferenceError(format!(
                "KV layer binding {} is not present in arena {:?}",
                binding.physical_layer, self.config.id
            ))
        })
    }

    fn cuda_paged_tuning(
        &self,
        key_head_dim: usize,
        value_head_dim: usize,
        batch: usize,
        query_heads: usize,
        max_context_tokens: usize,
    ) -> super::cuda_tuning::CudaPagedTuningPolicy {
        resolve_cuda_paged_tuning(
            &self.cuda_identity,
            CudaPagedShapeKey {
                dtype: self.config.dtype,
                page_tokens: self.config.page_tokens,
                key_head_dim,
                value_head_dim,
                batch,
                query_heads,
                max_context_tokens,
            },
        )
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
        let resident_pages = self.resident_capacity_pages() as usize;
        if page >= resident_pages {
            return Err(Error::InferenceError(format!(
                "KV block index {page} is outside resident arena capacity {resident_pages}"
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
            #[cfg(feature = "cuda")]
            {
                let stream = self.device.as_cuda_device()?.cuda_stream();
                let event = stream.record_event(None).map_err(|error| {
                    Error::InferenceError(format!(
                        "failed to record CUDA KV completion event: {error}"
                    ))
                })?;
                Ok(Arc::new(CudaEventFence {
                    event,
                    host_synchronizations: self.host_synchronizations.clone(),
                }))
            }
            #[cfg(not(feature = "cuda"))]
            Err(Error::InferenceError(
                "CUDA KV completion events are not compiled".into(),
            ))
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

    fn lower_prefill_metadata(
        &self,
        args: &PagedKvPrefillArgs<'_>,
    ) -> Result<LoweredPrefillMetadata> {
        let sequence_count = args.rows.len();
        let total_queries = args.queries.dims()[0];
        let max_blocks = args
            .rows
            .iter()
            .map(|row| row.blocks.len())
            .max()
            .unwrap_or(0);
        if sequence_count == 0 || total_queries == 0 || max_blocks == 0 {
            return Err(Error::InferenceError(
                "accelerator paged prefill requires non-empty rows, queries, and block tables"
                    .into(),
            ));
        }

        let compact_len = sequence_count
            .checked_mul(4_usize.checked_add(max_blocks).ok_or_else(|| {
                Error::InferenceError("accelerator prefill metadata width overflow".into())
            })?)
            .ok_or_else(|| {
                Error::InferenceError("accelerator prefill metadata length overflow".into())
            })?;
        let mut compact_rows = vec![0_u32; compact_len];
        let mut block_table = vec![0_u32; sequence_count * max_blocks];
        let mut cumulative_queries = Vec::with_capacity(sequence_count + 1);
        let mut cumulative_contexts = Vec::with_capacity(sequence_count + 1);
        cumulative_queries.push(0);
        cumulative_contexts.push(0);
        let mut next_query = 0_u32;
        let mut cumulative_context = 0_u32;
        let mut max_query_len = 0_usize;
        let mut max_context_len = 0_usize;
        let mut all_first_page_offsets_zero = true;

        for (row_index, row) in args.rows.iter().enumerate() {
            if row.query_start != next_query
                || row.query_len == 0
                || row.query_len > row.context_len
                || row.first_page_offset >= self.config.page_tokens
            {
                return Err(Error::InferenceError(format!(
                    "accelerator paged prefill row {row_index} is not a canonical causal range"
                )));
            }
            let physical_tokens = row
                .context_len
                .checked_add(row.first_page_offset)
                .ok_or_else(|| {
                    Error::InferenceError("accelerator prefill physical range overflow".into())
                })?;
            let required_pages =
                (physical_tokens as usize).div_ceil(self.config.page_tokens as usize);
            if required_pages == 0 || required_pages > row.blocks.len() {
                return Err(Error::InferenceError(format!(
                    "accelerator paged prefill row {row_index} has an incomplete block table"
                )));
            }

            compact_rows[row_index] = row.query_start;
            compact_rows[sequence_count + row_index] = row.query_len;
            compact_rows[2 * sequence_count + row_index] = row.context_len;
            compact_rows[3 * sequence_count + row_index] = row.first_page_offset;
            all_first_page_offsets_zero &= row.first_page_offset == 0;
            max_query_len = max_query_len.max(row.query_len as usize);
            max_context_len = max_context_len.max(row.context_len as usize);

            let table_start = row_index * max_blocks;
            let compact_table_start = 4 * sequence_count + table_start;
            for (logical_page, block) in row.blocks.iter().copied().enumerate() {
                let physical_page = u32::try_from(self.validate_block(block)?).map_err(|_| {
                    Error::InferenceError("accelerator prefill page index exceeds u32".into())
                })?;
                block_table[table_start + logical_page] = physical_page;
                compact_rows[compact_table_start + logical_page] = physical_page;
            }

            next_query = next_query.checked_add(row.query_len).ok_or_else(|| {
                Error::InferenceError("accelerator prefill query range overflow".into())
            })?;
            cumulative_context =
                cumulative_context
                    .checked_add(row.context_len)
                    .ok_or_else(|| {
                        Error::InferenceError("accelerator prefill context range overflow".into())
                    })?;
            cumulative_queries.push(next_query);
            cumulative_contexts.push(cumulative_context);
        }
        if next_query as usize != total_queries {
            return Err(Error::InferenceError(
                "accelerator paged prefill rows do not cover every query exactly once".into(),
            ));
        }

        Ok(LoweredPrefillMetadata {
            cache_key: PrefillMetadataCacheKey {
                rows: args.rows.to_vec(),
                total_queries,
            },
            compact_rows,
            cumulative_queries,
            cumulative_contexts,
            block_table,
            sequence_count,
            total_queries,
            max_blocks,
            max_query_len,
            max_context_len,
            all_first_page_offsets_zero,
        })
    }

    fn cached_prefill_device_metadata(
        &self,
        lowered: &LoweredPrefillMetadata,
    ) -> Result<PrefillDeviceMetadata> {
        let mut cache = self.prefill_metadata_cache.lock().map_err(|_| {
            Error::InferenceError("accelerator prefill metadata cache was poisoned".into())
        })?;
        if let Some(index) = cache
            .entries
            .iter()
            .position(|cached| cached.key == lowered.cache_key)
        {
            self.attention_plan_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            let cached = cache.promote(index).ok_or_else(|| {
                Error::InferenceError("accelerator prefill plan cache lost an entry".into())
            })?;
            return Ok(PrefillDeviceMetadata {
                compact_rows: cached.compact_rows.clone(),
                cumulative_queries: cached.cumulative_queries.clone(),
                cumulative_contexts: cached.cumulative_contexts.clone(),
                block_table: cached.block_table.clone(),
            });
        }

        let compact_rows = Tensor::from_vec(
            lowered.compact_rows.clone(),
            lowered.compact_rows.len(),
            &self.device,
        )?;
        let cumulative_queries = Tensor::from_vec(
            lowered.cumulative_queries.clone(),
            lowered.cumulative_queries.len(),
            &self.device,
        )?;
        let cumulative_contexts = Tensor::from_vec(
            lowered.cumulative_contexts.clone(),
            lowered.cumulative_contexts.len(),
            &self.device,
        )?;
        let block_table = Tensor::from_vec(
            lowered.block_table.clone(),
            (lowered.sequence_count, lowered.max_blocks),
            &self.device,
        )?;
        let before_bytes = cache.resident_bytes;
        let evictions = cache.insert(CachedPrefillDeviceMetadata {
            key: lowered.cache_key.clone(),
            compact_rows: compact_rows.clone(),
            cumulative_queries: cumulative_queries.clone(),
            cumulative_contexts: cumulative_contexts.clone(),
            block_table: block_table.clone(),
        });
        update_resident_byte_gauge(
            &self.attention_plan_resident_bytes,
            before_bytes,
            cache.resident_bytes,
        );
        self.attention_plan_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        self.attention_plan_cache_evictions
            .fetch_add(evictions as u64, Ordering::Relaxed);
        self.attention_plan_device_uploads
            .fetch_add(1, Ordering::Relaxed);
        Ok(PrefillDeviceMetadata {
            compact_rows,
            cumulative_queries,
            cumulative_contexts,
            block_table,
        })
    }

    fn cached_decode_device_metadata(
        &self,
        batch: &KvDecodeBatchMetadata,
        cumulative_contexts: &[u32],
        block_table: &[u32],
        native_metadata: &[u32],
        max_blocks: usize,
    ) -> Result<DecodeDeviceMetadata> {
        let mut cache = self.decode_metadata_cache.lock().map_err(|_| {
            Error::InferenceError("accelerator decode metadata cache was poisoned".into())
        })?;
        if let Some(index) = cache.entries.iter().position(|cached| cached.key == *batch) {
            self.attention_plan_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            let cached = cache.promote(index).ok_or_else(|| {
                Error::InferenceError("accelerator decode plan cache lost an entry".into())
            })?;
            return Ok(cached.device_metadata());
        }

        let batch_size = batch.sequences.len();
        if let Some(index) = cache
            .entries
            .iter()
            .position(|cached| cached.shape_compatible(batch_size, max_blocks))
        {
            let cached = cache.entries.get_mut(index).ok_or_else(|| {
                Error::InferenceError("accelerator decode plan cache lost an entry".into())
            })?;
            let mut updated = false;
            updated |= update_u32_tensor(
                &cached.cumulative_contexts,
                &cached.host_cumulative_contexts,
                cumulative_contexts,
                &self.device,
            )?;
            updated |= update_u32_tensor(
                &cached.block_table,
                &cached.host_block_table,
                block_table,
                &self.device,
            )?;
            updated |= update_u32_tensor(
                &cached.native_metadata,
                &cached.host_native_metadata,
                native_metadata,
                &self.device,
            )?;
            cached.key = batch.clone();
            cached.host_cumulative_contexts.clear();
            cached
                .host_cumulative_contexts
                .extend_from_slice(cumulative_contexts);
            cached.host_block_table.clear();
            cached.host_block_table.extend_from_slice(block_table);
            cached.host_native_metadata.clear();
            cached
                .host_native_metadata
                .extend_from_slice(native_metadata);
            let device_metadata = cached.device_metadata();
            let entry = cache.entries.remove(index).ok_or_else(|| {
                Error::InferenceError("accelerator decode plan cache lost an entry".into())
            })?;
            cache.entries.push_back(entry);
            self.attention_plan_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            if updated {
                self.attention_plan_device_uploads
                    .fetch_add(1, Ordering::Relaxed);
            }
            return Ok(device_metadata);
        }

        let cumulative_queries = (0..=batch_size)
            .map(|value| {
                u32::try_from(value).map_err(|_| {
                    Error::InferenceError("CUDA paged decode batch exceeds u32".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cumulative_queries =
            Tensor::from_vec(cumulative_queries, batch_size + 1, &self.device)?;
        let cumulative_contexts_device = Tensor::from_vec(
            cumulative_contexts.to_vec(),
            cumulative_contexts.len(),
            &self.device,
        )?;
        let block_table_device =
            Tensor::from_vec(block_table.to_vec(), (batch_size, max_blocks), &self.device)?;
        let native_metadata_device = Tensor::from_vec(
            native_metadata.to_vec(),
            native_metadata.len(),
            &self.device,
        )?;
        let before_bytes = cache.resident_bytes;
        let evictions = cache.insert(CachedDecodeDeviceMetadata {
            key: batch.clone(),
            host_cumulative_contexts: cumulative_contexts.to_vec(),
            host_block_table: block_table.to_vec(),
            host_native_metadata: native_metadata.to_vec(),
            cumulative_queries: cumulative_queries.clone(),
            cumulative_contexts: cumulative_contexts_device.clone(),
            block_table: block_table_device.clone(),
            native_metadata: native_metadata_device.clone(),
        });
        update_resident_byte_gauge(
            &self.attention_plan_resident_bytes,
            before_bytes,
            cache.resident_bytes,
        );
        self.attention_plan_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        self.attention_plan_cache_evictions
            .fetch_add(evictions as u64, Ordering::Relaxed);
        self.attention_plan_device_uploads
            .fetch_add(1, Ordering::Relaxed);
        Ok(DecodeDeviceMetadata {
            host_native_metadata: native_metadata.to_vec(),
            cumulative_queries,
            cumulative_contexts: cumulative_contexts_device,
            block_table: block_table_device,
            native_metadata: native_metadata_device,
            max_blocks,
            max_context: batch
                .sequences
                .iter()
                .map(|sequence| sequence.context_len as usize)
                .max()
                .unwrap_or(0),
            all_first_page_offsets_zero: batch
                .sequences
                .iter()
                .all(|sequence| sequence.first_page_offset == 0),
        })
    }

    fn cached_decode_plan(&self, batch: &KvDecodeBatchMetadata) -> Result<DecodeDeviceMetadata> {
        {
            let mut cache = self.decode_metadata_cache.lock().map_err(|_| {
                Error::InferenceError("accelerator decode metadata cache was poisoned".into())
            })?;
            if let Some(index) = cache.entries.iter().position(|cached| cached.key == *batch) {
                self.attention_plan_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                let cached = cache.promote(index).ok_or_else(|| {
                    Error::InferenceError("accelerator decode plan cache lost an entry".into())
                })?;
                return Ok(cached.device_metadata());
            }
        }

        let (table, cumulative_contexts, first_page_offsets, max_blocks, _) =
            self.lower_decode_tables(batch)?;
        let native_metadata =
            packed_decode_metadata(&cumulative_contexts, &first_page_offsets, &table);
        self.cached_decode_device_metadata(
            batch,
            &cumulative_contexts,
            &table,
            &native_metadata,
            max_blocks,
        )
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    fn cuda_flash_paged_prefill(
        &self,
        layer: &AcceleratorLayerStorage,
        args: &PagedKvPrefillArgs<'_>,
        lowered: &LoweredPrefillMetadata,
    ) -> Result<Tensor> {
        let metadata = self.cached_prefill_device_metadata(lowered)?;
        Ok(candle_flash_attn::flash_attn_varlen_paged_windowed(
            args.queries,
            &layer.keys,
            &layer.values,
            &metadata.cumulative_queries,
            &metadata.cumulative_contexts,
            &metadata.block_table,
            None,
            lowered.max_query_len,
            lowered.max_context_len,
            args.softmax_scale,
            args.window_tokens
                .map(|window| window.saturating_sub(1) as usize),
            Some(0),
            self.config.page_tokens as usize,
            args.softcap,
        )?)
    }

    #[cfg(feature = "cuda")]
    fn cuda_native_paged_prefill(
        &self,
        layer: &AcceleratorLayerStorage,
        args: &PagedKvPrefillArgs<'_>,
        lowered: &LoweredPrefillMetadata,
    ) -> Result<Tensor> {
        let metadata = self.cached_prefill_device_metadata(lowered)?;
        Ok(crate::kernels::cuda::paged_prefill_attention(
            args.queries,
            &layer.keys,
            &layer.values,
            &metadata.compact_rows,
            lowered.sequence_count,
            lowered.total_queries,
            args.queries.dims()[1],
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            lowered.max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
            args.softcap,
            args.window_tokens,
        )?)
    }

    #[cfg(feature = "metal")]
    fn metal_paged_prefill(
        &self,
        layer: &AcceleratorLayerStorage,
        args: &PagedKvPrefillArgs<'_>,
        lowered: &LoweredPrefillMetadata,
    ) -> Result<Tensor> {
        Ok(crate::kernels::metal::paged_prefill_attention(
            args.queries,
            &layer.keys,
            &layer.values,
            lowered.compact_rows.clone(),
            lowered.sequence_count,
            lowered.total_queries,
            args.queries.dims()[1],
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            lowered.max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
            args.softcap,
            args.window_tokens,
        )?)
    }

    #[cfg(feature = "cuda")]
    fn cuda_paged_decode(
        &self,
        layer: &AcceleratorLayerStorage,
        args: PagedKvDecodeArgs<'_>,
    ) -> Result<Tensor> {
        let batch_size = args.batch.sequences.len();
        let device_metadata = self.cached_decode_plan(args.batch)?;
        let tuning = self.cuda_paged_tuning(
            layer.key_head_dim,
            layer.value_head_dim,
            batch_size,
            args.queries.dims()[1],
            device_metadata.max_context,
        );
        #[cfg(feature = "flash-attn")]
        if self.optimized_provider_enabled
            && tuning.flash_attention_allowed
            && cuda_flash_paged_attention_eligible(
                self.storage_dtype,
                self.config.page_tokens,
                layer.key_head_dim,
                layer.value_head_dim,
                device_metadata.all_first_page_offsets_zero,
            )
        {
            let output = candle_flash_attn::flash_attn_varlen_paged_windowed(
                args.queries,
                &layer.keys,
                &layer.values,
                &device_metadata.cumulative_queries,
                &device_metadata.cumulative_contexts,
                &device_metadata.block_table,
                None,
                1,
                device_metadata.max_context,
                args.softmax_scale,
                None,
                None,
                self.config.page_tokens as usize,
                args.softcap,
            )?;
            self.record_attention_provider(KvAttentionProvider::CudaFlashAttention);
            return Ok(output);
        }

        let (output, graph_outcome) = crate::kernels::cuda::paged_decode_attention_with_graph(
            args.queries,
            &layer.keys,
            &layer.values,
            &device_metadata.native_metadata,
            batch_size,
            args.queries.dims()[1],
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            device_metadata.max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
            args.softcap,
            device_metadata.max_context,
            tuning.decode_partition_tuning,
            tuning.decode_graph_allowed && self.cuda_kv_storage == CudaKvStorageFormat::Dense,
            self.backing_generation.load(Ordering::Acquire),
        )?;
        use crate::kernels::cuda::CudaPagedDecodeGraphOutcome;
        match graph_outcome {
            CudaPagedDecodeGraphOutcome::Disabled => {}
            CudaPagedDecodeGraphOutcome::Warmed => {
                self.cuda_graph_warmups.fetch_add(1, Ordering::Relaxed);
            }
            CudaPagedDecodeGraphOutcome::WarmedAfterEviction => {
                self.cuda_graph_warmups.fetch_add(1, Ordering::Relaxed);
                self.cuda_graph_evictions.fetch_add(1, Ordering::Relaxed);
            }
            CudaPagedDecodeGraphOutcome::Captured => {
                self.cuda_graph_captures.fetch_add(1, Ordering::Relaxed);
            }
            CudaPagedDecodeGraphOutcome::Replayed => {
                self.cuda_graph_replays.fetch_add(1, Ordering::Relaxed);
            }
            CudaPagedDecodeGraphOutcome::EagerFallback => {
                self.cuda_graph_fallbacks.fetch_add(1, Ordering::Relaxed);
            }
            CudaPagedDecodeGraphOutcome::Backoff => {
                self.cuda_graph_backoff_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.record_attention_provider(KvAttentionProvider::CudaNative);
        Ok(output)
    }

    #[cfg(feature = "metal")]
    fn metal_paged_decode(
        &self,
        layer: &AcceleratorLayerStorage,
        args: PagedKvDecodeArgs<'_>,
    ) -> Result<Tensor> {
        let batch_size = args.batch.sequences.len();
        let num_heads = args.queries.dims()[1];
        let device_metadata = self.cached_decode_plan(args.batch)?;
        Ok(crate::kernels::metal::paged_decode_attention(
            args.queries,
            &layer.keys,
            &layer.values,
            device_metadata.host_native_metadata,
            &device_metadata.native_metadata,
            batch_size,
            num_heads,
            layer.num_kv_heads,
            self.config.page_tokens as usize,
            device_metadata.max_blocks,
            layer.key_head_dim,
            layer.value_head_dim,
            args.softmax_scale,
            args.softcap,
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

    fn resident_capacity_pages(&self) -> u32 {
        self.resident_capacity_pages.load(Ordering::Acquire)
    }

    fn plan_resident_growth(&self, required_pages: u32) -> Result<Option<KvArenaGrowthPlan>> {
        if required_pages > self.config.capacity_pages {
            return Err(Error::Backpressure(format!(
                "KV arena requires {required_pages} pages but its logical capacity is {}",
                self.config.capacity_pages
            )));
        }
        let previous_pages = self.resident_capacity_pages();
        if required_pages <= previous_pages {
            return Ok(None);
        }
        if self.backend != BackendKind::Cuda {
            return Err(Error::Backpressure(
                "only CUDA paged arenas support admission growth".into(),
            ));
        }
        let geometry = self.config.growth.ok_or_else(|| {
            Error::Backpressure("CUDA KV arena was not configured for admission growth".into())
        })?;
        let missing = required_pages - previous_pages;
        let rounded_missing = missing
            .div_ceil(geometry.growth_quantum_pages)
            .saturating_mul(geometry.growth_quantum_pages);
        let rounded_target = previous_pages.saturating_add(rounded_missing);
        // Amortize maintenance-barrier copies while retaining the sealed
        // quantum geometry required by the allocation ledger.
        let amortized_addition = previous_pages
            .max(geometry.growth_quantum_pages)
            .div_ceil(geometry.growth_quantum_pages)
            .saturating_mul(geometry.growth_quantum_pages);
        let doubled = previous_pages
            .saturating_add(amortized_addition)
            .min(self.config.capacity_pages);
        let target_pages = rounded_target.max(doubled).min(self.config.capacity_pages);
        Ok(Some(KvArenaGrowthPlan {
            arena: self.config.id,
            previous_pages,
            target_pages,
        }))
    }

    fn grow_resident_pages(&self, plan: KvArenaGrowthPlan) -> Result<()> {
        if self.backend != BackendKind::Cuda || plan.arena != self.config.id {
            return Err(Error::InvalidInput(
                "KV growth plan targets a different or fixed arena".into(),
            ));
        }
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        let current = self.resident_capacity_pages();
        if current != plan.previous_pages
            || plan.target_pages <= current
            || plan.target_pages > self.config.capacity_pages
        {
            return Err(Error::Backpressure(
                "KV growth plan is stale or exceeds the sealed arena".into(),
            ));
        }
        let target = plan.target_pages as usize;
        let mut layers = self
            .layers
            .write()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        let mut replacements = HashMap::with_capacity(layers.len());
        let current_rows = (0..current as usize).collect::<Vec<_>>();
        let current_indices = accelerator_indices(&current_rows, &self.device)?;
        for (binding, layer) in layers.iter() {
            let common = (target, self.config.page_tokens as usize, layer.num_kv_heads);
            let keys = Tensor::zeros(
                (common.0, common.1, common.2, layer.key_head_dim),
                self.storage_dtype,
                &self.device,
            )?;
            let values = Tensor::zeros(
                (common.0, common.1, common.2, layer.value_head_dim),
                self.storage_dtype,
                &self.device,
            )?;
            scatter_rows(&keys, &current_indices, &layer.keys)?;
            scatter_rows(&values, &current_indices, &layer.values)?;
            replacements.insert(*binding, (keys, values));
        }
        // The old backing must remain alive until every device copy into the
        // replacement has completed. Growth is an admission barrier, so this
        // synchronization is never part of steady decode.
        self.device.synchronize()?;
        self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
        for (binding, (keys, values)) in replacements {
            let layer = layers.get_mut(&binding).ok_or_else(|| {
                Error::InferenceError("KV layer disappeared during growth".into())
            })?;
            layer.keys = keys;
            layer.values = values;
        }
        self.resident_capacity_pages
            .store(plan.target_pages, Ordering::Release);
        self.clean_pages
            .lock()
            .map_err(|_| Error::InferenceError("accelerator KV cleanliness was poisoned".into()))?
            .resize(
                plan.target_pages as usize,
                AcceleratorPageCleanliness::Clean,
            );
        self.backing_generation.fetch_add(1, Ordering::Relaxed);
        Ok(())
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
        let device_indices = accelerator_indices(&flat_slots, &self.device)?;
        Ok(Arc::new(AcceleratorSlotMap {
            arena: self.config.id,
            flat_slots,
            device_indices,
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
        let page_indices = {
            let mut clean = self.clean_pages.lock().map_err(|_| {
                Error::InferenceError("accelerator KV cleanliness was poisoned".into())
            })?;
            page_indices
                .into_iter()
                .filter(|index| page_needs_zero(&mut clean[*index]))
                .collect::<Vec<_>>()
        };
        if page_indices.is_empty() {
            return Ok(Arc::new(CompletedAcceleratorFence));
        }
        let layers = self
            .layers
            .read()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        let mut trailing_shapes = layers
            .values()
            .flat_map(|layer| {
                [
                    vec![
                        self.config.page_tokens as usize,
                        layer.num_kv_heads,
                        layer.key_head_dim,
                    ],
                    vec![
                        self.config.page_tokens as usize,
                        layer.num_kv_heads,
                        layer.value_head_dim,
                    ],
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        trailing_shapes.sort_unstable();
        let per_shape_budget = ACCELERATOR_WORKSPACE_BUDGET_BYTES
            .checked_div(trailing_shapes.len())
            .ok_or_else(|| Error::InferenceError("accelerator arena has no KV layers".into()))?;
        let mut workspaces = Vec::with_capacity(trailing_shapes.len());
        let acquisition_result = (|| -> Result<()> {
            let mut pool = self.workspace_pool.lock().map_err(|_| {
                Error::InferenceError("accelerator workspace pool was poisoned".into())
            })?;
            for trailing_shape in trailing_shapes {
                let pages_per_chunk = zero_workspace_pages_per_chunk(
                    page_indices.len(),
                    &trailing_shape,
                    self.storage_dtype,
                    per_shape_budget,
                )?;
                let mut shape = Vec::with_capacity(trailing_shape.len() + 1);
                shape.push(pages_per_chunk);
                shape.extend_from_slice(&trailing_shape);
                match pool.acquire(&shape, self.storage_dtype, &self.device) {
                    Ok(workspace) => workspaces.push((trailing_shape, pages_per_chunk, workspace)),
                    Err(error) => {
                        for (_, _, workspace) in workspaces.drain(..) {
                            pool.discard(workspace);
                        }
                        return Err(error);
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = acquisition_result {
            if let Ok(mut pool) = self.workspace_pool.lock() {
                for (_, _, workspace) in workspaces.drain(..) {
                    pool.discard(workspace);
                }
            }
            return Err(error);
        }

        let dispatch_result = (|| -> Result<()> {
            for (trailing_shape, pages_per_chunk, workspace) in &workspaces {
                let destinations = layers
                    .values()
                    .flat_map(|layer| [&layer.keys, &layer.values])
                    .filter(|tensor| tensor.dims()[1..] == trailing_shape[..])
                    .collect::<Vec<_>>();
                for chunk in page_indices.chunks(*pages_per_chunk) {
                    let indices = accelerator_indices(chunk, &self.device)?;
                    let source = if chunk.len() == *pages_per_chunk {
                        workspace.tensor.clone()
                    } else {
                        workspace.tensor.narrow(0, 0, chunk.len())?
                    };
                    for destination in &destinations {
                        scatter_rows(destination, &indices, &source)?;
                    }
                }
            }
            Ok(())
        })();
        self.page_zero_dispatches.fetch_add(1, Ordering::Relaxed);
        let retirement = match self.mutation_fence() {
            Ok(retirement) => retirement,
            Err(error) => {
                if let Ok(mut pool) = self.workspace_pool.lock() {
                    for (_, _, workspace) in workspaces {
                        pool.discard(workspace);
                    }
                }
                return Err(error);
            }
        };
        let mut pool = self
            .workspace_pool
            .lock()
            .map_err(|_| Error::InferenceError("accelerator workspace pool was poisoned".into()))?;
        for (_, _, workspace) in workspaces {
            pool.retire(workspace, retirement.clone());
        }
        dispatch_result?;
        let mut clean = self
            .clean_pages
            .lock()
            .map_err(|_| Error::InferenceError("accelerator KV cleanliness was poisoned".into()))?;
        for page in page_indices {
            clean[page] = AcceleratorPageCleanliness::ZeroPending(retirement.clone());
        }
        Ok(retirement)
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
        let layers = self
            .layers
            .read()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        if !lowered.is_empty() {
            let source_indices = accelerator_indices(
                &lowered
                    .iter()
                    .map(|(source, _)| *source)
                    .collect::<Vec<_>>(),
                &self.device,
            )?;
            let destination_indices = accelerator_indices(
                &lowered
                    .iter()
                    .map(|(_, destination)| *destination)
                    .collect::<Vec<_>>(),
                &self.device,
            )?;
            for layer in layers.values() {
                // index_select snapshots every source before either destination
                // tensor is mutated, preserving parallel-copy chains and cycles.
                copy_rows_parallel(&layer.keys, &source_indices, &destination_indices)?;
                copy_rows_parallel(&layer.values, &source_indices, &destination_indices)?;
            }
            let mut clean = self.clean_pages.lock().map_err(|_| {
                Error::InferenceError("accelerator KV cleanliness was poisoned".into())
            })?;
            for (_, destination) in &lowered {
                clean[*destination] = AcceleratorPageCleanliness::Dirty;
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
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        let layers = self
            .layers
            .read()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        let layer = self.layer_from(&layers, binding)?;
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

        // CUDA growth seals a larger logical envelope than the allocation that
        // is currently resident. Flatten the actual backing, not that logical
        // maximum, or every write before the final growth step has a different
        // element count from the requested reshape.
        let flat_capacity = resident_flat_slot_capacity(
            layer,
            self.resident_capacity_pages() as usize,
            self.config.page_tokens as usize,
        )?;
        let flat_keys =
            layer
                .keys
                .reshape((flat_capacity, layer.num_kv_heads, layer.key_head_dim))?;
        let flat_values =
            layer
                .values
                .reshape((flat_capacity, layer.num_kv_heads, layer.value_head_dim))?;
        if !slots.flat_slots.is_empty() {
            let stored_keys = if self.storage_dtype == self.config.dtype {
                args.keys.clone()
            } else {
                args.keys.to_dtype(self.storage_dtype)?
            };
            let stored_values = if self.storage_dtype == self.config.dtype {
                args.values.clone()
            } else {
                args.values.to_dtype(self.storage_dtype)?
            };
            scatter_rows(&flat_keys, &slots.device_indices, &stored_keys)?;
            scatter_rows(&flat_values, &slots.device_indices, &stored_values)?;
            let mut clean = self.clean_pages.lock().map_err(|_| {
                Error::InferenceError("accelerator KV cleanliness was poisoned".into())
            })?;
            for slot in &slots.flat_slots {
                clean[*slot / self.config.page_tokens as usize] = AcceleratorPageCleanliness::Dirty;
            }
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

    fn paged_prefill(
        &self,
        binding: KvLayerBinding,
        args: PagedKvPrefillArgs<'_>,
    ) -> Result<Tensor> {
        let (_key_head_dim, _value_head_dim) = {
            let layers = self.layers.read().map_err(|_| {
                Error::InferenceError("accelerator KV layer map was poisoned".into())
            })?;
            let layer = self.layer_from(&layers, binding)?;
            validate_prefill_query(layer, &args, self.config.dtype, &self.device, self.backend)?;
            (layer.key_head_dim, layer.value_head_dim)
        };
        let lowered = self.lower_prefill_metadata(&args)?;

        #[cfg(feature = "metal")]
        if self.backend == BackendKind::Metal
            && matches!(self.config.dtype, DType::F32 | DType::F16)
        {
            let _guard = self.mutation_lock.lock().map_err(|_| {
                Error::InferenceError("accelerator KV mutation lock was poisoned".into())
            })?;
            let layers = self.layers.read().map_err(|_| {
                Error::InferenceError("accelerator KV layer map was poisoned".into())
            })?;
            let layer = self.layer_from(&layers, binding)?;
            let output = self.metal_paged_prefill(layer, &args, &lowered)?;
            self.record_attention_provider(KvAttentionProvider::MetalNative);
            self.paged_prefill_dispatches
                .fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        #[cfg(all(feature = "cuda", feature = "flash-attn"))]
        if self.backend == BackendKind::Cuda
            && self.optimized_provider_enabled
            && self
                .cuda_paged_tuning(
                    _key_head_dim,
                    _value_head_dim,
                    lowered.sequence_count,
                    args.queries.dims()[1],
                    lowered.max_context_len,
                )
                .flash_attention_allowed
            && cuda_flash_paged_attention_eligible(
                self.storage_dtype,
                self.config.page_tokens,
                _key_head_dim,
                _value_head_dim,
                lowered.all_first_page_offsets_zero,
            )
        {
            let _guard = self.mutation_lock.lock().map_err(|_| {
                Error::InferenceError("accelerator KV mutation lock was poisoned".into())
            })?;
            let layers = self.layers.read().map_err(|_| {
                Error::InferenceError("accelerator KV layer map was poisoned".into())
            })?;
            let layer = self.layer_from(&layers, binding)?;
            let output = self.cuda_flash_paged_prefill(layer, &args, &lowered)?;
            self.record_attention_provider(KvAttentionProvider::CudaFlashAttention);
            self.paged_prefill_dispatches
                .fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        #[cfg(feature = "cuda")]
        if self.backend == BackendKind::Cuda {
            let _guard = self.mutation_lock.lock().map_err(|_| {
                Error::InferenceError("accelerator KV mutation lock was poisoned".into())
            })?;
            let layers = self.layers.read().map_err(|_| {
                Error::InferenceError("accelerator KV layer map was poisoned".into())
            })?;
            let layer = self.layer_from(&layers, binding)?;
            let output = self.cuda_native_paged_prefill(layer, &args, &lowered)?;
            self.record_attention_provider(KvAttentionProvider::CudaNative);
            self.paged_prefill_dispatches
                .fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        let output = super::portable_paged_prefill(self, binding, args)?;
        self.record_attention_provider(KvAttentionProvider::Portable);
        self.paged_prefill_dispatches
            .fetch_add(1, Ordering::Relaxed);
        Ok(output)
    }

    fn paged_decode(&self, binding: KvLayerBinding, args: PagedKvDecodeArgs<'_>) -> Result<Tensor> {
        if !candle_accelerator_kv_support(self.backend).direct_paged_attention {
            return Err(Error::InferenceError(format!(
                "direct paged attention is not compiled for {:?}",
                self.backend
            )));
        }
        let _guard = self.mutation_lock.lock().map_err(|_| {
            Error::InferenceError("accelerator KV mutation lock was poisoned".into())
        })?;
        let layers = self
            .layers
            .read()
            .map_err(|_| Error::InferenceError("accelerator KV layer map was poisoned".into()))?;
        let layer = self.layer_from(&layers, binding)?;
        validate_decode_query(layer, &args, self.config.dtype, &self.device, self.backend)?;

        #[cfg(feature = "cuda")]
        if self.backend == BackendKind::Cuda {
            let output = self.cuda_paged_decode(layer, args)?;
            self.paged_decode_dispatches.fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        #[cfg(feature = "metal")]
        if self.backend == BackendKind::Metal {
            let output = self.metal_paged_decode(layer, args)?;
            self.record_attention_provider(KvAttentionProvider::MetalNative);
            self.paged_decode_dispatches.fetch_add(1, Ordering::Relaxed);
            return Ok(output);
        }

        Err(Error::InferenceError(format!(
            "direct paged attention is unavailable for {:?}",
            self.backend
        )))
    }

    fn operation_stats(&self) -> KvArenaOperationStats {
        let (
            workspace_bytes,
            workspace_budget_bytes,
            workspace_high_water_bytes,
            workspace_allocations,
        ) = self
            .workspace_pool
            .lock()
            .ok()
            .map(|pool| {
                (
                    u64::try_from(pool.reserved_bytes).ok(),
                    u64::try_from(pool.budget_bytes).ok(),
                    u64::try_from(pool.high_water_bytes).ok(),
                    Some(pool.allocations),
                )
            })
            .unwrap_or((None, None, None, None));
        KvArenaOperationStats {
            slot_write_dispatches: self.slot_write_dispatches.load(Ordering::Relaxed),
            paged_prefill_dispatches: self.paged_prefill_dispatches.load(Ordering::Relaxed),
            paged_decode_dispatches: self.paged_decode_dispatches.load(Ordering::Relaxed),
            page_zero_dispatches: self.page_zero_dispatches.load(Ordering::Relaxed),
            page_copy_dispatches: self.page_copy_dispatches.load(Ordering::Relaxed),
            attention_plan_cache_hits: self.attention_plan_cache_hits.load(Ordering::Relaxed),
            attention_plan_cache_misses: self.attention_plan_cache_misses.load(Ordering::Relaxed),
            attention_plan_cache_evictions: self
                .attention_plan_cache_evictions
                .load(Ordering::Relaxed),
            attention_plan_device_uploads: self
                .attention_plan_device_uploads
                .load(Ordering::Relaxed),
            attention_plan_resident_bytes: self
                .attention_plan_resident_bytes
                .load(Ordering::Relaxed),
            backing_allocations: self.layers.read().ok().map(|layers| {
                (layers.len() as u64)
                    .saturating_mul(2)
                    .saturating_mul(self.backing_generation.load(Ordering::Relaxed))
            }),
            workspace_bytes,
            workspace_budget_bytes,
            workspace_high_water_bytes,
            workspace_allocations,
            cpu_reference_attention_dispatches: 0,
            portable_attention_dispatches: self
                .portable_attention_dispatches
                .load(Ordering::Relaxed),
            cuda_native_attention_dispatches: self
                .cuda_native_attention_dispatches
                .load(Ordering::Relaxed),
            cuda_flash_attention_dispatches: self
                .cuda_flash_attention_dispatches
                .load(Ordering::Relaxed),
            metal_native_attention_dispatches: self
                .metal_native_attention_dispatches
                .load(Ordering::Relaxed),
            cuda_graph_warmups: self.cuda_graph_warmups.load(Ordering::Relaxed),
            cuda_graph_captures: self.cuda_graph_captures.load(Ordering::Relaxed),
            cuda_graph_replays: self.cuda_graph_replays.load(Ordering::Relaxed),
            cuda_graph_fallbacks: self.cuda_graph_fallbacks.load(Ordering::Relaxed),
            cuda_graph_backoff_hits: self.cuda_graph_backoff_hits.load(Ordering::Relaxed),
            cuda_graph_evictions: self.cuda_graph_evictions.load(Ordering::Relaxed),
            last_attention_provider: KvAttentionProvider::from_code(
                self.last_attention_provider.load(Ordering::Relaxed),
            ),
            host_synchronizations: self.host_synchronizations.load(Ordering::Relaxed),
        }
    }

    fn drain(&self) -> Result<()> {
        if self.backend == BackendKind::Metal {
            self.device.synchronize()?;
            self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            #[cfg(feature = "cuda")]
            {
                self.device
                    .as_cuda_device()?
                    .cuda_stream()
                    .synchronize()
                    .map_err(|error| {
                        Error::InferenceError(format!("CUDA KV stream drain failed: {error}"))
                    })?;
                self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            #[cfg(not(feature = "cuda"))]
            Err(Error::InferenceError(
                "CUDA KV stream drain is not compiled".into(),
            ))
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
        // the selector ordinal accepted by Candle's Metal constructor. Require the
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
    if dims[1] == 0 || !dims[1].is_multiple_of(layer.num_kv_heads) {
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
    super::validate_attention_softcap(args.softcap)
}

fn validate_prefill_query(
    layer: &AcceleratorLayerStorage,
    args: &PagedKvPrefillArgs<'_>,
    dtype: DType,
    device: &Device,
    backend: BackendKind,
) -> Result<()> {
    if args.queries.device().location() != device.location() {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged prefill queries are on the wrong device"
        )));
    }
    if args.queries.dtype() != dtype {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged prefill query dtype {:?} does not match arena dtype {dtype:?}",
            args.queries.dtype()
        )));
    }
    if !args.queries.layout().is_contiguous() {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged prefill queries must be contiguous"
        )));
    }
    let dims = args.queries.dims();
    if dims.len() != 3 || dims[0] == 0 || dims[2] != layer.key_head_dim {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged prefill query shape {dims:?} does not match key head dimension {}",
            layer.key_head_dim
        )));
    }
    if dims[1] == 0 || !dims[1].is_multiple_of(layer.num_kv_heads) {
        return Err(Error::InferenceError(format!(
            "{backend:?} paged prefill query heads {} are not divisible by KV heads {}",
            dims[1], layer.num_kv_heads
        )));
    }
    if !args.softmax_scale.is_finite() || args.softmax_scale <= 0.0 {
        return Err(Error::InferenceError(
            "paged prefill softmax scale must be finite and positive".into(),
        ));
    }
    if args.window_tokens == Some(0) {
        return Err(Error::InferenceError(
            "paged prefill attention window must be non-zero".into(),
        ));
    }
    super::validate_attention_softcap(args.softcap)
}

fn accelerator_indices(indices: &[usize], device: &Device) -> Result<Tensor> {
    let indices = indices
        .iter()
        .copied()
        .map(|index| {
            u32::try_from(index)
                .map_err(|_| Error::InferenceError("accelerator KV index exceeds u32".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let len = indices.len();
    Ok(Tensor::from_vec(indices, len, device)?)
}

fn scatter_rows(destination: &Tensor, indices: &Tensor, source: &Tensor) -> Result<()> {
    let mut index_shape = vec![1; source.rank()];
    index_shape[0] = indices.elem_count();
    let indices = indices
        .reshape(index_shape)?
        .broadcast_as(source.shape())?
        .contiguous()?;
    destination.scatter_set(&indices, source, 0)?;
    Ok(())
}

fn update_u32_tensor(
    destination: &Tensor,
    previous: &[u32],
    next: &[u32],
    device: &Device,
) -> Result<bool> {
    if previous.len() != next.len() || destination.elem_count() != next.len() {
        return Err(Error::InferenceError(
            "accelerator attention metadata update changed its sealed shape".into(),
        ));
    }
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (index, (&previous, &next)) in previous.iter().zip(next).enumerate() {
        if previous != next {
            indices.push(index);
            values.push(next);
        }
    }
    if indices.is_empty() {
        return Ok(false);
    }
    let indices = accelerator_indices(&indices, device)?;
    let value_count = values.len();
    let values = Tensor::from_vec(values, value_count, device)?;
    scatter_rows(&destination.flatten_all()?, &indices, &values)?;
    Ok(true)
}

fn packed_decode_metadata(
    cumulative_contexts: &[u32],
    first_page_offsets: &[u32],
    block_table: &[u32],
) -> Vec<u32> {
    let context_lens = cumulative_contexts
        .windows(2)
        .map(|window| window[1] - window[0]);
    let mut metadata = Vec::with_capacity(
        cumulative_contexts
            .len()
            .saturating_sub(1)
            .saturating_add(first_page_offsets.len())
            .saturating_add(block_table.len()),
    );
    metadata.extend(context_lens);
    metadata.extend(first_page_offsets.iter().copied());
    metadata.extend(block_table.iter().copied());
    metadata
}

fn copy_rows_parallel(
    tensor: &Tensor,
    source_indices: &Tensor,
    destination_indices: &Tensor,
) -> Result<()> {
    let source_rows = tensor.index_select(source_indices, 0)?;
    scatter_rows(tensor, destination_indices, &source_rows)
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

    #[derive(Debug)]
    struct MockResidentPlan(usize);

    impl ResidentAttentionPlan for MockResidentPlan {
        fn resident_bytes(&self) -> usize {
            self.0
        }
    }

    #[derive(Debug)]
    struct TestCompletionFence(Arc<std::sync::atomic::AtomicBool>);

    impl KvDeviceFence for TestCompletionFence {
        fn is_complete(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }

        fn wait(&self) -> Result<()> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[test]
    fn resident_flat_capacity_uses_materialized_pages_not_logical_envelope() -> Result<()> {
        let layer = AcceleratorLayerStorage {
            keys: Tensor::zeros((2, 4, 1, 3), DType::F32, &Device::Cpu)?,
            values: Tensor::zeros((2, 4, 1, 5), DType::F32, &Device::Cpu)?,
            num_kv_heads: 1,
            key_head_dim: 3,
            value_head_dim: 5,
        };

        assert_eq!(resident_flat_slot_capacity(&layer, 2, 4)?, 8);
        assert_eq!(
            layer.keys.reshape((8, 1, 3))?.dims(),
            &[8, 1, 3],
            "a partially resident backing must flatten independently of its logical maximum"
        );
        Ok(())
    }

    #[test]
    fn resident_flat_capacity_rejects_published_backing_divergence() {
        let layer = AcceleratorLayerStorage {
            keys: Tensor::zeros((2, 4, 1, 3), DType::F32, &Device::Cpu).unwrap(),
            values: Tensor::zeros((3, 4, 1, 5), DType::F32, &Device::Cpu).unwrap(),
            num_kv_heads: 1,
            key_head_dim: 3,
            value_head_dim: 5,
        };

        let error = resident_flat_slot_capacity(&layer, 2, 4).unwrap_err();
        assert!(format!("{error}").contains("backing/residency mismatch"));
    }

    fn test_completion(complete: bool) -> (DeviceFence, Arc<std::sync::atomic::AtomicBool>) {
        let state = Arc::new(std::sync::atomic::AtomicBool::new(complete));
        (Arc::new(TestCompletionFence(state.clone())), state)
    }

    #[test]
    fn page_cleanliness_never_elides_an_in_flight_zero() {
        let mut clean = AcceleratorPageCleanliness::Clean;
        assert!(!page_needs_zero(&mut clean));

        let mut dirty = AcceleratorPageCleanliness::Dirty;
        assert!(page_needs_zero(&mut dirty));

        let (pending, completion) = test_completion(false);
        let mut zero_pending = AcceleratorPageCleanliness::ZeroPending(pending);
        assert!(page_needs_zero(&mut zero_pending));
        completion.store(true, Ordering::Release);
        assert!(!page_needs_zero(&mut zero_pending));
        assert!(matches!(zero_pending, AcceleratorPageCleanliness::Clean));
    }

    #[test]
    fn accelerator_workspace_pool_reuses_one_allocation_for_one_hundred_operations() -> Result<()> {
        let mut pool = AcceleratorWorkspacePool::new(64);
        let (completed, _) = test_completion(true);
        for _ in 0..100 {
            let workspace = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
            pool.retire(workspace, completed.clone());
        }
        assert_eq!(pool.allocations, 1);
        assert_eq!(pool.reserved_bytes, 64);
        assert_eq!(pool.high_water_bytes, 64);
        assert_eq!(pool.retired.len(), 1);
        Ok(())
    }

    #[test]
    fn accelerator_workspace_pool_never_reuses_before_retirement() -> Result<()> {
        let mut pool = AcceleratorWorkspacePool::new(128);
        let (pending, pending_state) = test_completion(false);
        let first = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        let first_id = first.tensor.id();
        pool.retire(first, pending);

        let (also_pending, _) = test_completion(false);
        let second = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        assert_ne!(second.tensor.id(), first_id);
        pool.retire(second, also_pending);
        assert_eq!(pool.allocations, 2);

        pending_state.store(true, Ordering::Release);
        let reused = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        assert_eq!(reused.tensor.id(), first_id);
        assert_eq!(pool.allocations, 2);
        Ok(())
    }

    #[test]
    fn accelerator_workspace_pool_evicts_completed_mismatched_shapes() -> Result<()> {
        let mut pool = AcceleratorWorkspacePool::new(64);
        let (completed, _) = test_completion(true);
        let workspace = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        pool.retire(workspace, completed);

        let replacement = pool.acquire(&[32], DType::F16, &Device::Cpu)?;
        assert_eq!(replacement.bytes, 64);
        assert_eq!(pool.reserved_bytes, 64);
        assert!(pool.retired.is_empty());
        assert_eq!(pool.allocations, 2);
        Ok(())
    }

    #[test]
    fn accelerator_zero_workspace_chunks_distinct_kv_shapes_within_one_budget() -> Result<()> {
        let per_shape_budget = ACCELERATOR_WORKSPACE_BUDGET_BYTES / 2;
        let key_shape = [64, 8, 256];
        let value_shape = [64, 8, 128];
        let key_pages =
            zero_workspace_pages_per_chunk(1_024, &key_shape, DType::BF16, per_shape_budget)?;
        let value_pages =
            zero_workspace_pages_per_chunk(1_024, &value_shape, DType::BF16, per_shape_budget)?;
        let key_bytes =
            key_pages * key_shape.iter().product::<usize>() * DType::BF16.size_in_bytes();
        let value_bytes =
            value_pages * value_shape.iter().product::<usize>() * DType::BF16.size_in_bytes();

        assert!(key_pages < 1_024);
        assert!(value_pages < 1_024);
        assert!(key_bytes <= per_shape_budget);
        assert!(value_bytes <= per_shape_budget);
        assert!(key_bytes + value_bytes <= ACCELERATOR_WORKSPACE_BUDGET_BYTES);
        Ok(())
    }

    #[test]
    fn accelerator_workspace_pool_returns_structured_budget_error() -> Result<()> {
        let mut pool = AcceleratorWorkspacePool::new(64);
        let (pending, _) = test_completion(false);
        let workspace = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        pool.retire(workspace, pending);

        let error = pool
            .acquire(&[16], DType::F32, &Device::Cpu)
            .expect_err("an in-flight workspace must still consume the budget");
        let message = error.to_string();
        assert!(message.contains("requested_bytes=64"));
        assert!(message.contains("reserved_bytes=64"));
        assert!(message.contains("budget_bytes=64"));
        Ok(())
    }

    #[test]
    fn accelerator_workspace_high_water_survives_scratch_release() -> Result<()> {
        let mut pool = AcceleratorWorkspacePool::new(128);
        let workspace = pool.acquire(&[16], DType::F32, &Device::Cpu)?;
        pool.discard(workspace);

        assert_eq!(pool.reserved_bytes, 0);
        assert_eq!(pool.high_water_bytes, 64);
        Ok(())
    }

    #[test]
    fn resident_attention_plan_cache_is_byte_bounded_and_lru() {
        let mut cache = ResidentAttentionPlanCache::default();
        let half_budget = ATTENTION_PLAN_CACHE_BYTES_PER_KIND / 2;
        assert_eq!(cache.insert(MockResidentPlan(half_budget)), 0);
        assert_eq!(cache.insert(MockResidentPlan(half_budget)), 0);
        assert_eq!(cache.resident_bytes, ATTENTION_PLAN_CACHE_BYTES_PER_KIND);

        // Promote the oldest entry, making the other one the deterministic victim.
        assert_eq!(cache.promote(0).map(|entry| entry.0), Some(half_budget));
        assert_eq!(cache.insert(MockResidentPlan(1)), 1);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.resident_bytes, half_budget + 1);

        // Plans larger than the configured budget are used by the caller but not retained.
        assert_eq!(
            cache.insert(MockResidentPlan(ATTENTION_PLAN_CACHE_BYTES_PER_KIND + 1)),
            0
        );
        assert_eq!(cache.resident_bytes, half_budget + 1);
    }

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
    fn cuda_flash_paged_attention_eligibility_matches_fa2_shape_contract() {
        assert!(cuda_flash_paged_attention_eligible(
            DType::F16,
            32,
            64,
            64,
            true,
        ));
        assert!(cuda_flash_paged_attention_eligible(
            DType::BF16,
            64,
            512,
            512,
            true,
        ));

        for (dtype, page_tokens, key_dim, value_dim, offsets_zero) in [
            (DType::F32, 32, 64, 64, true),
            (DType::F16, 16, 64, 64, true),
            (DType::F16, 32, 4, 4, true),
            (DType::F16, 32, 513, 513, true),
            (DType::F16, 32, 64, 32, true),
            (DType::F16, 32, 64, 64, false),
        ] {
            assert!(!cuda_flash_paged_attention_eligible(
                dtype,
                page_tokens,
                key_dim,
                value_dim,
                offsets_zero,
            ));
        }
    }

    #[test]
    fn batched_row_mutations_preserve_order_dtypes_and_parallel_copy_semantics() -> Result<()> {
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "metal")]
        if let Some(device) = crate::backends::metal_device_if_available(0) {
            devices.push(device);
        }
        #[cfg(feature = "cuda")]
        if let Ok(device) = Device::new_cuda(0) {
            devices.push(device);
        }

        for device in devices {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let destination = Tensor::from_vec(
                    vec![0_f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                    (4, 2),
                    &device,
                )?
                .to_dtype(dtype)?;
                let source = Tensor::from_vec(vec![100_f32, 101.0, 200.0, 201.0], (2, 2), &device)?
                    .to_dtype(dtype)?;
                let rows = accelerator_indices(&[3, 1], &device)?;
                scatter_rows(&destination, &rows, &source)?;
                assert_eq!(
                    destination
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .to_vec2::<f32>()?,
                    vec![
                        vec![0.0, 1.0],
                        vec![200.0, 201.0],
                        vec![20.0, 21.0],
                        vec![100.0, 101.0],
                    ],
                    "batched scatter failed for {:?} {dtype:?}",
                    device.location()
                );

                let chain = Tensor::from_vec(vec![0_f32, 1.0, 2.0, 3.0], (4, 1), &device)?
                    .to_dtype(dtype)?;
                let chain_sources = accelerator_indices(&[0, 1], &device)?;
                let chain_destinations = accelerator_indices(&[1, 2], &device)?;
                copy_rows_parallel(&chain, &chain_sources, &chain_destinations)?;
                assert_eq!(
                    chain
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    vec![0.0, 0.0, 1.0, 3.0],
                    "parallel copy chain failed for {:?} {dtype:?}",
                    device.location()
                );

                let cycle =
                    Tensor::from_vec(vec![0_f32, 1.0, 2.0], (3, 1), &device)?.to_dtype(dtype)?;
                let cycle_sources = accelerator_indices(&[0, 1, 2], &device)?;
                let cycle_destinations = accelerator_indices(&[1, 2, 0], &device)?;
                copy_rows_parallel(&cycle, &cycle_sources, &cycle_destinations)?;
                assert_eq!(
                    cycle
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    vec![2.0, 0.0, 1.0],
                    "parallel copy cycle failed for {:?} {dtype:?}",
                    device.location()
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn prefill_lowering_is_compact_and_cache_key_tracks_generations() -> Result<()> {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
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
            capacity_pages: 4,
            growth: None,
            dtype: DType::F32,
            layers: vec![super::super::KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 3,
            }],
        };
        let arena = CandleAcceleratorKvArena::new_mutation_only(config, device.clone())?;
        let block = |index, slot_generation| CacheBlockRef {
            arena: arena_id,
            group,
            index,
            slot_generation,
        };
        let rows = vec![
            PagedKvPrefillRow {
                blocks: vec![block(2, 1), block(0, 1)],
                first_page_offset: 1,
                query_start: 0,
                query_len: 2,
                context_len: 3,
            },
            PagedKvPrefillRow {
                blocks: vec![block(3, 1)],
                first_page_offset: 0,
                query_start: 2,
                query_len: 1,
                context_len: 2,
            },
        ];
        let queries = Tensor::zeros((3, 2, 2), DType::F32, &device)?;
        let args = PagedKvPrefillArgs {
            queries: &queries,
            rows: &rows,
            softmax_scale: 0.5,
            softcap: None,
            window_tokens: None,
        };
        let lowered = arena.lower_prefill_metadata(&args)?;
        assert_eq!(lowered.cumulative_queries, vec![0, 2, 3]);
        assert_eq!(lowered.cumulative_contexts, vec![0, 3, 5]);
        assert_eq!(lowered.block_table, vec![2, 0, 3, 0]);
        assert_eq!(
            lowered.compact_rows,
            vec![0, 2, 2, 1, 3, 2, 1, 0, 2, 0, 3, 0]
        );
        assert_eq!(lowered.max_query_len, 2);
        assert_eq!(lowered.max_context_len, 3);
        assert!(!lowered.all_first_page_offsets_zero);

        let mut next_generation_rows = rows.clone();
        next_generation_rows[0].blocks[0].slot_generation = 2;
        let next_generation = arena.lower_prefill_metadata(&PagedKvPrefillArgs {
            rows: &next_generation_rows,
            ..args
        })?;
        assert_ne!(lowered.cache_key, next_generation.cache_key);
        arena.cached_prefill_device_metadata(&lowered)?;
        arena.cached_prefill_device_metadata(&lowered)?;
        arena.cached_prefill_device_metadata(&next_generation)?;
        arena.cached_prefill_device_metadata(&lowered)?;

        let decode = KvDecodeBatchMetadata {
            sequences: vec![
                crate::kv::KvSequenceBlockTable {
                    blocks: rows[0].blocks.clone(),
                    first_page_offset: 1,
                    context_len: 3,
                },
                crate::kv::KvSequenceBlockTable {
                    blocks: rows[1].blocks.clone(),
                    first_page_offset: 0,
                    context_len: 2,
                },
            ],
        };
        let (table, cumulative, offsets, max_blocks, _) = arena.lower_decode_tables(&decode)?;
        let native = packed_decode_metadata(&cumulative, &offsets, &table);
        arena.cached_decode_device_metadata(&decode, &cumulative, &table, &native, max_blocks)?;
        arena.cached_decode_plan(&decode)?;
        let mut next_decode_generation = decode.clone();
        next_decode_generation.sequences[0].blocks[0].slot_generation = 2;
        arena.cached_decode_plan(&next_decode_generation)?;
        let mut next_token = next_decode_generation.clone();
        next_token.sequences[0].context_len += 1;
        next_token.sequences[0].first_page_offset = 0;
        let (table, cumulative, offsets, _, _) = arena.lower_decode_tables(&next_token)?;
        let native = packed_decode_metadata(&cumulative, &offsets, &table);
        let updated = arena.cached_decode_plan(&next_token)?;
        assert_eq!(updated.cumulative_contexts.to_vec1::<u32>()?, cumulative);
        assert_eq!(updated.native_metadata.to_vec1::<u32>()?, native);
        let stats = arena.operation_stats();
        assert_eq!(stats.attention_plan_cache_hits, 5);
        assert_eq!(stats.attention_plan_cache_misses, 3);
        let cache_stats = arena.attention_plan_cache_stats();
        assert_eq!(cache_stats.hits, 5);
        assert_eq!(cache_stats.misses, 3);
        assert_eq!(cache_stats.evictions, 0);
        assert_eq!(cache_stats.device_uploads, 4);
        assert!(cache_stats.resident_bytes > 0);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn clean_page_tracking_elides_only_proven_redundant_zeroes() -> Result<()> {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            return Ok(());
        };
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(91),
            backend: BackendKind::Metal,
            device_ordinal: Some(0),
            generation: 1,
        };
        let group = KvGroupId::new(0);
        let arena = CandleAcceleratorKvArena::new_mutation_only(
            KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 2,
                capacity_pages: 2,
                growth: None,
                dtype: DType::F32,
                layers: vec![super::super::KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 1,
                    value_head_dim: 1,
                }],
            },
            device.clone(),
        )?;
        let block = CacheBlockRef {
            arena: arena_id,
            group,
            index: 0,
            slot_generation: 1,
        };

        assert!(arena.zero_pages(&[block])?.is_complete());
        assert_eq!(arena.operation_stats().page_zero_dispatches, 0);

        let slots = arena.lower_slots(&[KvSlotRef { block, offset: 0 }])?;
        let keys = Tensor::ones((1, 1, 1), DType::F32, &device)?;
        let values = Tensor::ones((1, 1, 1), DType::F32, &device)?;
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
        arena.zero_pages(&[block])?.wait()?;
        assert_eq!(arena.operation_stats().page_zero_dispatches, 1);
        let (keys, values) = arena.layer_tensors(binding)?;
        assert_eq!(keys.flatten_all()?.to_vec1::<f32>()?, vec![0.0; 4]);
        assert_eq!(values.flatten_all()?.to_vec1::<f32>()?, vec![0.0; 4]);

        assert!(arena.zero_pages(&[block])?.is_complete());
        assert_eq!(arena.operation_stats().page_zero_dispatches, 1);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_paged_decode_and_prefill_match_cpu_for_ragged_shuffled_mha_gqa_mqa() -> Result<()> {
        // Feature-only CI must tolerate an unavailable or unsupported Metal
        // runtime without entering Candle's device constructor.
        let Some(device) = crate::backends::metal_device_if_available(0) else {
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
                    growth: None,
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
                        softcap: None,
                    },
                )?;
                let cpu_output = cpu_arena.paged_decode(
                    binding,
                    PagedKvDecodeArgs {
                        queries: &cpu_query,
                        batch: &cpu_batch,
                        softmax_scale: 0.5,
                        softcap: None,
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

                let metal_rows = vec![
                    PagedKvPrefillRow {
                        blocks: vec![metal_block(2), metal_block(0)],
                        first_page_offset: 1,
                        query_start: 0,
                        query_len: 2,
                        context_len: 3,
                    },
                    PagedKvPrefillRow {
                        blocks: vec![metal_block(3)],
                        first_page_offset: 0,
                        query_start: 2,
                        query_len: 1,
                        context_len: 2,
                    },
                ];
                let cpu_rows = vec![
                    PagedKvPrefillRow {
                        blocks: vec![cpu_block(2), cpu_block(0)],
                        first_page_offset: 1,
                        query_start: 0,
                        query_len: 2,
                        context_len: 3,
                    },
                    PagedKvPrefillRow {
                        blocks: vec![cpu_block(3)],
                        first_page_offset: 0,
                        query_start: 2,
                        query_len: 1,
                        context_len: 2,
                    },
                ];
                let prefill_query_data = (0..(3 * num_query_heads * 2))
                    .map(|index| (index as f32 - 4.0) / 6.0)
                    .collect::<Vec<_>>();
                let metal_prefill_queries =
                    Tensor::from_vec(prefill_query_data.clone(), (3, num_query_heads, 2), &device)?
                        .to_dtype(dtype)?;
                let cpu_prefill_queries =
                    Tensor::from_vec(prefill_query_data, (3, num_query_heads, 2), &Device::Cpu)?
                        .to_dtype(dtype)?;
                let metal_prefill = metal_arena.paged_prefill(
                    binding,
                    PagedKvPrefillArgs {
                        queries: &metal_prefill_queries,
                        rows: &metal_rows,
                        softmax_scale: 0.5,
                        softcap: None,
                        window_tokens: None,
                    },
                )?;
                let cpu_prefill = cpu_arena.paged_prefill(
                    binding,
                    PagedKvPrefillArgs {
                        queries: &cpu_prefill_queries,
                        rows: &cpu_rows,
                        softmax_scale: 0.5,
                        softcap: None,
                        window_tokens: None,
                    },
                )?;
                let metal_prefill = metal_prefill
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let cpu_prefill = cpu_prefill
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(metal_prefill.len(), cpu_prefill.len());
                for (actual, expected) in metal_prefill.iter().zip(cpu_prefill.iter()) {
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "prefill {dtype:?} {num_query_heads}Q/{num_kv_heads}KV: {actual} != {expected}"
                    );
                }

                if dtype == DType::F32 && num_kv_heads == 1 && num_query_heads == 2 {
                    let metal_windowed = metal_arena.paged_prefill(
                        binding,
                        PagedKvPrefillArgs {
                            queries: &metal_prefill_queries,
                            rows: &metal_rows,
                            softmax_scale: 0.5,
                            softcap: None,
                            window_tokens: Some(2),
                        },
                    )?;
                    let cpu_windowed = cpu_arena.paged_prefill(
                        binding,
                        PagedKvPrefillArgs {
                            queries: &cpu_prefill_queries,
                            rows: &cpu_rows,
                            softmax_scale: 0.5,
                            softcap: None,
                            window_tokens: Some(2),
                        },
                    )?;
                    let metal_windowed = metal_windowed
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let cpu_windowed = cpu_windowed
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for (actual, expected) in metal_windowed.iter().zip(cpu_windowed.iter()) {
                        assert!(
                            (actual - expected).abs() < 1e-5,
                            "windowed portable prefill fallback: {actual} != {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "required CUDA evidence: run explicitly on a CUDA device; absence must fail"]
    fn cuda_paged_decode_matches_cpu_for_offsets_and_gqa() -> Result<()> {
        let device = Device::new_cuda(0)?;
        // Cover every supported page size and accelerator dtype with realistic
        // MQA/GQA geometry. Non-zero first-page offsets deliberately make the
        // FlashAttention path ineligible so this test certifies CUDA-native
        // grouped-head addressing rather than silently accepting a provider
        // fallback.
        let cases = [
            (16_u32, DType::F32, 1_usize, 8_usize, 64_usize),
            (32, DType::F16, 8, 32, 128),
            (64, DType::BF16, 4, 16, 64),
        ];
        for (case_index, (page_tokens, dtype, num_kv_heads, num_query_heads, head_dim)) in
            cases.into_iter().enumerate()
        {
            let binding = KvLayerBinding {
                model_layer: 0,
                physical_layer: 0,
            };
            let cuda_arena_id = KvArenaId {
                model_instance: ModelInstanceId::new(case_index as u64 + 1),
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
                key_head_dim: head_dim as u32,
                value_head_dim: head_dim as u32,
            };
            let cuda_config = KvArenaConfig {
                id: cuda_arena_id,
                group,
                page_tokens,
                capacity_pages: 4,
                growth: None,
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
                    (0..page_tokens).map(move |offset| KvSlotRef {
                        block: cuda_block(page),
                        offset,
                    })
                })
                .collect::<Vec<_>>();
            let cpu_slots = (0..4)
                .flat_map(|page| {
                    (0..page_tokens).map(move |offset| KvSlotRef {
                        block: cpu_block(page),
                        offset,
                    })
                })
                .collect::<Vec<_>>();
            let cuda_slots = cuda_arena.lower_slots(&cuda_slots)?;
            let cpu_slots = cpu_arena.lower_slots(&cpu_slots)?;
            let token_capacity = 4 * page_tokens as usize;
            let key_data = (0..(token_capacity * num_kv_heads * head_dim))
                .map(|index| ((index % 251) as f32 - 125.0) / 64.0)
                .collect::<Vec<_>>();
            let value_data = (0..(token_capacity * num_kv_heads * head_dim))
                .map(|index| ((index % 239) as f32 - 119.0) / 71.0)
                .collect::<Vec<_>>();
            let cuda_keys = Tensor::from_vec(
                key_data.clone(),
                (token_capacity, num_kv_heads, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let cuda_values = Tensor::from_vec(
                value_data.clone(),
                (token_capacity, num_kv_heads, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let cpu_keys = Tensor::from_vec(
                key_data,
                (token_capacity, num_kv_heads, head_dim),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let cpu_values = Tensor::from_vec(
                value_data,
                (token_capacity, num_kv_heads, head_dim),
                &Device::Cpu,
            )?
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
                        context_len: page_tokens + 5,
                    },
                    crate::kv::KvSequenceBlockTable {
                        blocks: vec![cuda_block(3)],
                        first_page_offset: 0,
                        context_len: page_tokens - 1,
                    },
                ],
            };
            let cpu_batch = KvDecodeBatchMetadata {
                sequences: vec![
                    crate::kv::KvSequenceBlockTable {
                        blocks: vec![cpu_block(2), cpu_block(0)],
                        first_page_offset: 1,
                        context_len: page_tokens + 5,
                    },
                    crate::kv::KvSequenceBlockTable {
                        blocks: vec![cpu_block(3)],
                        first_page_offset: 0,
                        context_len: page_tokens - 1,
                    },
                ],
            };
            let query_data = (0..(2 * num_query_heads * head_dim))
                .map(|index| ((index % 127) as f32 - 63.0) / 53.0)
                .collect::<Vec<_>>();
            let cuda_query =
                Tensor::from_vec(query_data.clone(), (2, num_query_heads, head_dim), &device)?
                    .to_dtype(dtype)?;
            let cpu_query =
                Tensor::from_vec(query_data, (2, num_query_heads, head_dim), &Device::Cpu)?
                    .to_dtype(dtype)?;
            let softmax_scale = 1.0 / (head_dim as f32).sqrt();
            let cuda_output = cuda_arena.paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &cuda_query,
                    batch: &cuda_batch,
                    softmax_scale,
                    softcap: None,
                },
            )?;
            assert_eq!(
                cuda_arena
                    .operation_stats()
                    .last_attention_provider
                    .map(|provider| provider.name()),
                Some("cuda_native"),
                "required CUDA-native evidence used a different provider"
            );
            let cpu_output = cpu_arena.paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &cpu_query,
                    batch: &cpu_batch,
                    softmax_scale,
                    softcap: None,
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
                        "page={page_tokens} dtype={dtype:?} dim={head_dim} {num_query_heads}Q/{num_kv_heads}KV: {actual} != {expected}"
                    );
            }
        }
        Ok(())
    }
}

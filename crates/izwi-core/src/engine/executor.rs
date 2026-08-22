//! Model executor - handles forward pass execution.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[path = "executor/audio.rs"]
mod audio;
#[path = "executor/dispatch.rs"]
mod dispatch;
#[path = "executor/handler_asr.rs"]
mod handler_asr;
#[path = "executor/handler_audio_chat.rs"]
mod handler_audio_chat;
#[path = "executor/handler_chat.rs"]
mod handler_chat;
#[path = "executor/handler_tts.rs"]
mod handler_tts;
#[path = "executor/state.rs"]
mod state;
#[path = "executor/streaming.rs"]
mod streaming;

pub(crate) use streaming::{
    deliver_committed_streams, CommittedStreamDelivery, IncrementalStreamDeliveryWorkers,
    StreamDeliveryFailure, StreamDeliveryFailureKind,
};

pub(super) fn decode_request_audio_with_rate(
    request: &EngineCoreRequest,
) -> Result<(Vec<f32>, u32)> {
    audio::decode_request_audio_with_rate(request)
}

use super::config::EngineCoreConfig;
use super::execution::{
    BatchDispatch, CacheMode, CancellationGranularity, ConcurrencyClass, DispatchState,
    ExecutionCapabilities, ExecutionDisposition, ExecutionFailure, ExecutionMode, ExecutionProfile,
    FailureKind, FailureOrigin, FailureScope, FinishReason, HealthImpact, NativeBatchMode,
    OutcomeProvenance, PhysicalBatch, PlanId, PrefillMode, RetryDisposition, SessionKey,
    StageProgressKind, WorkUnit, YieldReason,
};
use super::output::StreamingOutput;
use super::request::EngineCoreRequest;
use super::resources::{BatchWorkspaceLease, ResourceAuthority, ResourceVector};
use super::scheduler::ScheduledRequest;
use super::types::AudioOutput;
use crate::backends::{
    can_parallelize_requests, BackendContext, BackendKind, BackendPreference, BackendRouter,
    BackendSelectionSource,
};
use crate::error::{Error, Result};
use crate::kv::{CacheDomainId, KvArenaId, KvGroupId, KvStorageDType, KvStorageFormat};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::registry::{AsrModelLease, NativeAsrModel, NativeChatModel, QwenTtsModelLease};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::ModelRegistry;
use state::{ActiveAsrDecode, ActiveChatDecode, ActiveQwenTtsDecode};

const QWEN38_TARGET_ATTENTION_DOMAIN: CacheDomainId = CacheDomainId::new(1);
const QWEN38_MTP_ATTENTION_DOMAIN: CacheDomainId = CacheDomainId::new(4);

struct Qwen38ManagedCaches {
    target: PhysicalPagedKvCache,
    mtp: Option<PhysicalPagedKvCache>,
}

fn exact_managed_group_for_domain(
    groups: &[(CacheDomainId, KvGroupId, KvArenaId)],
    reservation: &super::ManagedCacheReservation,
    domain_id: CacheDomainId,
    required: bool,
) -> Result<Option<KvGroupId>> {
    let planned = groups
        .iter()
        .filter(|(domain, _, _)| *domain == domain_id)
        .collect::<Vec<_>>();
    let reserved = reservation
        .domains
        .iter()
        .filter(|domain| domain.domain == domain_id)
        .collect::<Vec<_>>();

    if planned.is_empty() && reserved.is_empty() {
        if required {
            return Err(Error::InferenceError(format!(
                "managed Qwen3.8 reservation omitted required domain {}",
                domain_id.get()
            )));
        }
        return Ok(None);
    }
    if planned.len() != 1 || reserved.len() != 1 {
        return Err(Error::InferenceError(format!(
            "managed Qwen3.8 domain {} must resolve exactly once in both the plan and reservation",
            domain_id.get()
        )));
    }
    let (_, group_id, arena) = *planned[0];
    if reserved[0].arena != arena {
        return Err(Error::InferenceError(format!(
            "managed Qwen3.8 domain {} crossed its planned arena",
            domain_id.get()
        )));
    }
    Ok(Some(group_id))
}

fn qwen38_managed_group_ids(
    groups: &[(CacheDomainId, KvGroupId, KvArenaId)],
    reservation: &super::ManagedCacheReservation,
) -> Result<(KvGroupId, Option<KvGroupId>)> {
    let target =
        exact_managed_group_for_domain(groups, reservation, QWEN38_TARGET_ATTENTION_DOMAIN, true)?
            .ok_or_else(|| {
                Error::InferenceError("managed Qwen3.8 target domain did not resolve".into())
            })?;
    let mtp =
        exact_managed_group_for_domain(groups, reservation, QWEN38_MTP_ATTENTION_DOMAIN, false)?;
    Ok((target, mtp))
}

fn qwen38_managed_caches_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<Qwen38ManagedCaches> {
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed Qwen3.8 row has no model runtime".to_string())
    })?;
    let groups = runtime
        .plan()
        .groups
        .iter()
        .map(|group| (group.domain, group.id, group.arena))
        .collect::<Vec<_>>();
    let (target_group, mtp_group) = qwen38_managed_group_ids(&groups, reservation)?;
    let target = physical_paged_cache_for_row(
        request,
        scheduled,
        reservation,
        QWEN38_TARGET_ATTENTION_DOMAIN,
        target_group,
    )?;
    let mtp = mtp_group
        .map(|group| {
            physical_paged_cache_for_row(
                request,
                scheduled,
                reservation,
                QWEN38_MTP_ATTENTION_DOMAIN,
                group,
            )
        })
        .transpose()?;
    Ok(Qwen38ManagedCaches { target, mtp })
}

fn qwen3_managed_cache_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<PhysicalPagedKvCache> {
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed Qwen3 row has no model runtime".to_string())
    })?;
    let mut groups = runtime.plan().groups.iter().filter(|group| {
        reservation
            .domains
            .iter()
            .any(|domain| domain.domain == group.domain && domain.arena == group.arena)
    });
    let group = groups.next().ok_or_else(|| {
        Error::InvalidInput("native Qwen3 reservation has no resolved paged-attention group".into())
    })?;
    if groups.next().is_some() {
        return Err(Error::InvalidInput(
            "native Qwen3 reservation resolves more than one paged-attention group".into(),
        ));
    }
    physical_paged_cache_for_row(request, scheduled, reservation, group.domain, group.id)
}

/// Per-row scheduler-owned KV views for one continuous chat quantum. Dense
/// families carry a single paged view; hybrid families also own an optional
/// speculative arena that must swap with the same transaction.
pub(super) enum ContinuousRowManagedCache {
    Dense(PhysicalPagedKvCache),
    Hybrid {
        target: PhysicalPagedKvCache,
        mtp: Option<PhysicalPagedKvCache>,
    },
}

fn continuous_row_managed_caches_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<ContinuousRowManagedCache> {
    if matches!(
        request.model_variant,
        Some(variant) if variant.family() == crate::catalog::ModelFamily::Qwen38Chat
    ) {
        let Qwen38ManagedCaches { target, mtp } =
            qwen38_managed_caches_for_row(request, scheduled, reservation)?;
        Ok(ContinuousRowManagedCache::Hybrid { target, mtp })
    } else {
        Ok(ContinuousRowManagedCache::Dense(
            qwen3_managed_cache_for_row(request, scheduled, reservation)?,
        ))
    }
}

/// Resolve one exact scheduler-owned paged-attention view without assuming
/// that the row reservation contains only one state domain or physical group.
fn physical_paged_cache_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
    domain_id: CacheDomainId,
    group_id: KvGroupId,
) -> Result<PhysicalPagedKvCache> {
    if reservation.txn_id != scheduled.plan_id || reservation.session != scheduled.session_key() {
        return Err(Error::InferenceError(
            "managed-cache reservation crossed its scheduled row fence".to_string(),
        ));
    }
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed-cache row has no physical model runtime".to_string())
    })?;
    let plan = runtime.plan();
    if request.model_instance_id() != Some(plan.model_instance) {
        return Err(Error::InferenceError(
            "managed-cache runtime does not match the row's loaded model instance".into(),
        ));
    }

    let mut groups = plan
        .groups
        .iter()
        .filter(|group| group.domain == domain_id && group.id == group_id);
    let group = groups.next().ok_or_else(|| {
        Error::InferenceError("managed-cache row references an unresolved domain/group pair".into())
    })?;
    if groups.next().is_some() {
        return Err(Error::InferenceError(
            "managed-cache plan repeats a domain/group pair".into(),
        ));
    }
    let layers = &group.layers;
    if layers.is_empty() {
        return Err(Error::InferenceError(
            "managed-cache paged-attention group has no layer bindings".into(),
        ));
    }
    if group.arena.model_instance != plan.model_instance
        || group.arena.backend != plan.backend
        || group.arena.device_ordinal != plan.device_ordinal
    {
        return Err(Error::InferenceError(
            "managed-cache group crossed its resolved runtime identity".into(),
        ));
    }

    let mut domains = reservation
        .domains
        .iter()
        .filter(|domain| domain.domain == domain_id && domain.arena == group.arena);
    let domain = domains.next().ok_or_else(|| {
        Error::InferenceError(
            "managed-cache reservation omitted the selected domain/group arena".into(),
        )
    })?;
    if domains.next().is_some() {
        return Err(Error::InvalidInput(
            "managed-cache reservation repeats the selected domain/group arena".into(),
        ));
    }
    if domain.execution_start_tokens < domain.expected_committed_tokens
        || domain.target_committed_tokens < domain.execution_start_tokens
        || domain.target_window_start > domain.execution_start_tokens
    {
        return Err(Error::InvalidInput(
            "managed-cache domain has an invalid execution/window range".into(),
        ));
    }
    if group.page_tokens == 0
        || domain.first_page_offset >= group.page_tokens
        || domain.first_page_offset != domain.target_window_start % group.page_tokens
    {
        return Err(Error::InvalidInput(
            "managed-cache first-page offset does not match its logical window".to_string(),
        ));
    }

    let mut tables = domain
        .provisional_groups
        .iter()
        .filter(|table| table.group == group_id);
    let table = tables.next().ok_or_else(|| {
        Error::InferenceError("managed-cache reservation omitted its selected block table".into())
    })?;
    if tables.next().is_some() {
        return Err(Error::InvalidInput(
            "managed-cache reservation repeats its selected block table".into(),
        ));
    }

    let arena = runtime.arena(group.arena).ok_or_else(|| {
        Error::InferenceError("managed-cache physical arena is no longer live".to_string())
    })?;
    let config = arena.config();
    if arena.id() != group.arena
        || arena.backend_kind() != plan.backend
        || config.id != group.arena
        || config.group != group_id
        || config.page_tokens != group.page_tokens
        || config.capacity_pages != group.capacity_pages
    {
        return Err(Error::InferenceError(
            "managed-cache arena geometry does not match its resolved group".into(),
        ));
    }
    let storage_matches = matches!(
        (group.storage, config.dtype),
        (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::F32
            },
            candle_core::DType::F32
        ) | (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::F16
            },
            candle_core::DType::F16
        ) | (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::Bf16
            },
            candle_core::DType::BF16
        )
    );
    if !storage_matches
        || config.layers.len() != layers.len()
        || config
            .layers
            .iter()
            .zip(layers)
            .any(|(configured, resolved)| {
                configured.binding != *resolved
                    || configured.num_kv_heads == 0
                    || configured.key_head_dim == 0
                    || configured.value_head_dim == 0
            })
    {
        return Err(Error::InferenceError(
            "managed-cache arena layer or storage geometry is stale".into(),
        ));
    }
    let element_bytes = match config.dtype {
        candle_core::DType::F32 => 4_u64,
        candle_core::DType::F16 | candle_core::DType::BF16 => 2_u64,
        _ => {
            return Err(Error::InferenceError(
                "managed-cache arena uses unsupported paged storage".into(),
            ));
        }
    };
    let bytes_per_page = config.layers.iter().try_fold(0_u64, |total, layer| {
        let per_token = u64::from(layer.num_kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::InferenceError("managed-cache layer geometry overflow".into()))?;
        let bytes = u64::from(config.page_tokens)
            .checked_mul(per_token)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| Error::InferenceError("managed-cache page geometry overflow".into()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::InferenceError("managed-cache page geometry overflow".into()))
    })?;
    if bytes_per_page != group.bytes_per_page {
        return Err(Error::InferenceError(
            "managed-cache arena byte geometry does not match its resolved group".into(),
        ));
    }

    let visible_target = domain
        .target_committed_tokens
        .checked_sub(domain.target_window_start)
        .ok_or_else(|| Error::InvalidInput("managed-cache window exceeds its target".into()))?;
    let physical_target = visible_target
        .checked_add(domain.first_page_offset)
        .ok_or_else(|| Error::InvalidInput("managed-cache window geometry overflow".into()))?;
    let required_pages = usize::try_from(physical_target.div_ceil(group.page_tokens))
        .map_err(|_| Error::InvalidInput("managed-cache page count exceeds usize".into()))?;
    if required_pages == 0 || table.blocks.len() != required_pages {
        return Err(Error::InvalidInput(format!(
            "managed-cache block table has {} pages, expected {required_pages}",
            table.blocks.len()
        )));
    }
    let mut unique_blocks = HashSet::with_capacity(table.blocks.len());
    if table.blocks.iter().any(|block| {
        block.arena != group.arena
            || block.group != group_id
            || block.index >= group.capacity_pages
            || block.slot_generation == 0
            || !unique_blocks.insert(*block)
    }) {
        return Err(Error::InvalidInput(
            "managed-cache block table contains a foreign, stale, duplicate, or out-of-range page"
                .into(),
        ));
    }

    PhysicalPagedKvCache::new_windowed(
        arena.clone(),
        layers.clone(),
        table.blocks.clone(),
        usize::try_from(domain.target_window_start)
            .map_err(|_| Error::InvalidInput("managed-cache window exceeds usize".into()))?,
        usize::try_from(domain.execution_start_tokens)
            .map_err(|_| Error::InvalidInput("managed-cache context exceeds usize".into()))?,
    )
}

fn invocation_paged_stage_and_domains<'a>(
    request: &'a EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<(
    &'a super::StageDescriptor,
    Vec<crate::kv::v2::StateDomainId>,
)> {
    let binding = request.execution_adapter_binding().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no loaded adapter binding".to_string())
    })?;
    let stage = binding.stage_for_work(&scheduled.work)?;
    let graph = crate::kv::v2::stage_graph_fingerprint(&binding.stages)?;
    let descriptor = request.v2_state_descriptor().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no state descriptor".to_string())
    })?;
    let crate::kv::v2::InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
        return Err(Error::InferenceError(
            "physical invocation row has no bounded workspace profile".to_string(),
        ));
    };
    let profile = profiles
        .iter()
        .find(|profile| profile.stage_graph_fingerprint == graph)
        .ok_or_else(|| {
            Error::InferenceError(
                "physical invocation row has no workspace for its adapter graph".to_string(),
            )
        })?;
    let workspace = profile
        .stages
        .iter()
        .find(|workspace| workspace.stage == stage.id)
        .ok_or_else(|| {
            Error::InferenceError(
                "physical invocation row has no workspace for its scheduled stage".to_string(),
            )
        })?;
    let paged = workspace
        .domains
        .iter()
        .filter_map(|domain| match domain {
            crate::kv::v2::InvocationWorkspaceDomain::State {
                state: crate::kv::v2::StateDomainSpec::PagedAttention(state),
                capacity,
                ..
            } if capacity.paged_max_tokens().is_some() => Some(state.header.id),
            _ => None,
        })
        .collect::<Vec<_>>();
    if paged.is_empty() {
        return Err(Error::InferenceError(
            "physical invocation stage has no paged workspace domain".to_string(),
        ));
    }
    Ok((stage, paged))
}

/// Lease the one paged invocation domain authored for this exact scheduled
/// stage. Models receive only the physical cache view and cannot select a pool
/// by convention or by model-family-specific IDs.
fn invocation_paged_lease_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<super::InvocationPagedKvLease> {
    let (stage, domains) = invocation_paged_stage_and_domains(request, scheduled)?;
    if domains.len() != 1 {
        return Err(Error::InferenceError(
            "physical invocation stage has multiple paged workspace domains".to_string(),
        ));
    }
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_invocation_paged(stage.id, domains[0])
}

fn validate_atomic_scalar_invocation_stage(
    stage: &super::StageDescriptor,
    work: &WorkUnit,
) -> Result<()> {
    if !matches!(work, WorkUnit::AtomicJob { .. }) {
        return Err(Error::InvalidInput(
            "scalar invocation workspace requires an atomic scheduled row".to_string(),
        ));
    }
    if stage.progress != StageProgressKind::Atomic || stage.batch_mode != NativeBatchMode::None {
        return Err(Error::InvalidInput(
            "atomic invocation workspace requires a scalar atomic execution stage".to_string(),
        ));
    }
    Ok(())
}

/// Acquire one atomic scalar row's complete authored typed workspace in
/// canonical domain order. This is the model-neutral path for mixed paged,
/// recurrent, append, ring, and static state.
pub(super) fn invocation_workspace_leases_for_atomic_scalar_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<crate::kv::v2::InvocationWorkspaceLeaseSetV2> {
    let binding = request.execution_adapter_binding().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no loaded adapter binding".to_string())
    })?;
    let stage = binding.stage_for_work(&scheduled.work)?;
    validate_atomic_scalar_invocation_stage(stage, &scheduled.work)?;
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_complete_invocation_workspace_set(stage.id)
}

/// Acquire one atomic scalar row's complete authored paged-domain set in
/// canonical identity order. Callers cannot omit a required domain. The
/// returned set releases every already-acquired lease if a later domain fails,
/// and explicit completion returns only authenticated writes.
pub(super) fn invocation_paged_leases_for_atomic_scalar_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<crate::kv::v2::InvocationPagedLeaseSetV2> {
    let (stage, _) = invocation_paged_stage_and_domains(request, scheduled)?;
    validate_atomic_scalar_invocation_stage(stage, &scheduled.work)?;
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_complete_invocation_paged_set(stage.id)
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Configuration for the model executor.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Backend to use (cpu, metal, cuda)
    pub backend: BackendKind,
    /// Resolved backend/device context for this worker.
    pub backend_context: BackendContext,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// KV cache storage dtype hint (e.g. float16, int8).
    pub kv_cache_dtype: String,
    /// Number of threads
    pub num_threads: usize,
    /// Maximum number of requests to execute in parallel.
    pub request_parallelism: usize,
    /// Decode-time KV cache page size.
    pub kv_page_size: usize,
    /// Optional shared model registry for loaded runtime models.
    pub model_registry: Option<Arc<ModelRegistry>>,
    /// Shared physical resource authority used for bounded executor workspaces.
    pub resource_authority: Option<Arc<ResourceAuthority>>,
    /// Maximum width of a model-native tensor batch on this backend.
    pub max_tensor_batch_size: usize,
    /// Exact model variants enabled for static tensor execution on this worker.
    pub static_tensor_batch_variants: Arc<HashSet<ModelVariant>>,
    /// Opt-in scheduler-level chunked prefill for resumable-prefill models.
    pub enable_chunked_prefill: bool,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("models_dir", &self.models_dir)
            .field("backend", &self.backend)
            .field("backend_context", &self.backend_context)
            .field("dtype", &self.dtype)
            .field("kv_cache_dtype", &self.kv_cache_dtype)
            .field("num_threads", &self.num_threads)
            .field("request_parallelism", &self.request_parallelism)
            .field("kv_page_size", &self.kv_page_size)
            .field(
                "model_registry",
                &self.model_registry.as_ref().map(|_| "<shared>"),
            )
            .field(
                "resource_authority",
                &self.resource_authority.as_ref().map(|_| "<shared>"),
            )
            .field("max_tensor_batch_size", &self.max_tensor_batch_size)
            .field(
                "static_tensor_batch_variants",
                &self.static_tensor_batch_variants.len(),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let backend_context = BackendRouter::resolve_context(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        );
        let backend_kind = backend_context.backend_kind;
        let num_threads = 4;
        Self {
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind),
            kv_page_size: 64,
            model_registry: None,
            resource_authority: None,
            max_tensor_batch_size: 1,
            static_tensor_batch_variants: Arc::new(HashSet::new()),
            enable_chunked_prefill: false,
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        let backend_context =
            BackendRouter::resolve_context_for_kind(config.backend, BackendSelectionSource::Config);
        let backend_kind = backend_context.backend_kind;
        let num_threads = config.num_threads.max(1);
        let max_tensor_batch_size = config
            .max_tensor_batch_size
            .resolve(backend_kind)
            .min(Self::tensor_batch_cap(backend_kind))
            .max(1);
        let request_parallelism = Self::resolve_batch_request_parallelism(
            backend_kind,
            Self::request_parallelism_override(),
        );
        Self {
            models_dir: config.models_dir.clone(),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            num_threads,
            request_parallelism,
            kv_page_size: config.block_size.max(1),
            model_registry: None,
            resource_authority: None,
            max_tensor_batch_size,
            static_tensor_batch_variants: Arc::new(HashSet::new()),
            enable_chunked_prefill: config.enable_chunked_prefill,
        }
    }
}

impl WorkerConfig {
    fn tensor_batch_cap(backend: BackendKind) -> usize {
        match backend {
            BackendKind::Cpu | BackendKind::Metal => 2,
            // Runtime CUDA defaults remain VRAM-tiered and resource-admitted;
            // this is only the hard kernel/metadata width ceiling.
            BackendKind::Cuda => 32,
        }
    }

    fn request_parallelism_override() -> Option<usize> {
        std::env::var("IZWI_REQUEST_PARALLELISM")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn resolve_request_parallelism(backend: BackendKind, override_value: Option<usize>) -> usize {
        // Candle's Metal path is intentionally serialized in dispatch. Do not
        // let an environment override inflate coordinator capacity beyond what
        // the executor can actually run concurrently.
        if backend == BackendKind::Metal {
            return 1;
        }
        let default_parallelism = match backend {
            // CPU workloads already use `num_threads` for BLAS/Rayon/intra-op work, so
            // keep inter-request fan-out conservative unless explicitly overridden.
            BackendKind::Cpu => 1,
            BackendKind::Metal => unreachable!("Metal is clamped above"),
            BackendKind::Cuda => 1,
        };

        override_value.unwrap_or(default_parallelism).max(1)
    }

    fn request_parallelism_for(backend: BackendKind) -> usize {
        Self::resolve_request_parallelism(backend, Self::request_parallelism_override())
    }

    fn resolve_batch_request_parallelism(
        backend: BackendKind,
        override_value: Option<usize>,
    ) -> usize {
        Self::resolve_request_parallelism(backend, override_value)
    }
}

/// Output from the executor after a forward pass.
pub const REQUEST_DEADLINE_EXCEEDED: &str = "request deadline exceeded";

#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Optional input transcription for speech-to-speech requests.
    pub input_transcription: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Optional per-request phase timing override from model-specific execution paths.
    pub phase_timing_override: Option<ExecutorPhaseTiming>,
    /// Optional ASR diagnostics payload surfaced by model-specific paths.
    pub asr_diagnostics: Option<serde_json::Value>,
    /// Error if any
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorPhaseTiming {
    /// Audio/media decode duration in milliseconds.
    pub media_decode_ms: Option<f64>,
    /// Input normalization duration in milliseconds.
    pub normalization_ms: Option<f64>,
    /// Prefill phase duration in milliseconds.
    pub prefill_ms: Option<f64>,
    /// Decode phase duration in milliseconds.
    pub decode_ms: Option<f64>,
    /// Sampling duration in milliseconds.
    pub sampling_ms: Option<f64>,
    /// Codec encode/decode duration in milliseconds.
    pub codec_ms: Option<f64>,
    /// Postprocess duration in milliseconds.
    pub postprocess_ms: Option<f64>,
    /// Time to first user-visible output in milliseconds since model execution start.
    pub first_output_ms_since_start: Option<f64>,
    /// Number of prefill steps attributed to this request.
    pub prefill_steps: Option<u32>,
    /// Number of decode steps attributed to this request.
    pub decode_steps: Option<u32>,
}

impl ExecutorPhaseTiming {
    pub fn with_media_decode_ms(media_decode_ms: f64) -> Self {
        Self {
            media_decode_ms: Some(media_decode_ms.max(0.0)),
            ..Self::default()
        }
    }
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: Some(error.into()),
        }
    }

    pub fn cancelled(request_id: String) -> Self {
        Self::terminal(request_id)
    }

    /// Construct a terminal payload whose precise outcome is carried by the
    /// authoritative execution disposition.
    pub fn terminal(request_id: String) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        }
    }
}

/// Backend-neutral result produced by one model-owned session safe point.
/// Native handlers must choose sequence, yield, or atomic semantics explicitly.
#[derive(Debug, Clone)]
pub struct ModelSessionResult {
    pub output: ExecutorOutput,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub provenance: OutcomeProvenance,
    pub staged_stream_outputs: Vec<StreamingOutput>,
    /// Backend-sealed physical write batches produced by this safe point.
    /// The executor reconciles these against the exact row reservation before
    /// it can construct a managed-cache receipt.
    pub(crate) managed_cache_completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
}

impl ModelSessionResult {
    fn executor_failure(message: String) -> ExecutionDisposition {
        ExecutionDisposition::Failed(ExecutionFailure {
            kind: FailureKind::Executor,
            scope: FailureScope::Row,
            retry: RetryDisposition::Never,
            health: HealthImpact::None,
            message,
        })
    }

    pub fn sequence(output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
        }
    }

    pub fn yielded(output: ExecutorOutput, reason: YieldReason) -> Self {
        Self {
            output,
            disposition: ExecutionDisposition::Yielded(reason),
            safe_point: true,
            provenance: OutcomeProvenance::produced_output(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
        }
    }

    pub fn cancelled(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::started(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
        }
    }

    pub fn cancelled_before_dispatch(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::not_started(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
        }
    }

    pub fn atomic(mut output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            let message = "atomic model session returned before reaching a terminal state";
            output.error = Some(message.to_string());
            output.finished = true;
            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message))
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
        }
    }

    fn with_staged_stream_outputs(mut self, outputs: Vec<StreamingOutput>) -> Self {
        self.staged_stream_outputs = outputs;
        self
    }

    pub(crate) fn with_managed_cache_completions(
        mut self,
        completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    ) -> Self {
        self.managed_cache_completions = completions;
        self
    }
}

/// Executor payload fenced to the exact scheduler transaction that produced it.
#[derive(Debug, Clone)]
pub struct ExecutorStepResult {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
    /// Executor-owned resources retained after this safe point. Persistent
    /// inference state is reported by its lifecycle-owned physical manager.
    pub observed_resources: ResourceVector,
    pub output: ExecutorOutput,
    pub staged_stream_outputs: Vec<StreamingOutput>,
    /// Optional physical KV write acknowledgement for this exact row.
    pub managed_cache: Option<super::ManagedCacheReceipt>,
    pub(crate) managed_cache_completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
}

impl ExecutorStepResult {
    pub fn new(scheduled: &ScheduledRequest, output: ExecutorOutput) -> Self {
        let session_result = if output.finished || output.error.is_some() {
            ModelSessionResult::atomic(output)
        } else {
            // Compatibility for third-party/test executors. Native production
            // handlers use `from_session` with an explicit session result.
            ModelSessionResult::sequence(output)
        };
        Self::from_session(scheduled, session_result)
    }

    pub fn from_session(scheduled: &ScheduledRequest, session_result: ModelSessionResult) -> Self {
        Self {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            disposition: session_result.disposition,
            safe_point: session_result.safe_point,
            dispatch: BatchDispatch::serial(),
            provenance: session_result.provenance,
            observed_resources: ResourceVector::zero(),
            output: session_result.output,
            staged_stream_outputs: session_result.staged_stream_outputs,
            managed_cache: None,
            managed_cache_completions: session_result.managed_cache_completions,
        }
    }

    pub fn with_dispatch(mut self, dispatch: BatchDispatch) -> Self {
        self.dispatch = dispatch;
        self
    }

    pub fn with_provenance(mut self, provenance: OutcomeProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_observed_resources(mut self, resources: ResourceVector) -> Self {
        self.observed_resources = resources;
        self
    }

    pub fn with_managed_cache_receipt(mut self, receipt: super::ManagedCacheReceipt) -> Self {
        self.managed_cache = Some(receipt);
        self
    }
}

/// Model executor trait - abstracts the model inference backend.
pub struct PhysicalBatchExecution<'a> {
    pub batch: &'a PhysicalBatch,
    pub requests: &'a [&'a EngineCoreRequest],
    pub scheduled: &'a [ScheduledRequest],
}

#[derive(Debug)]
pub struct PhysicalDispatchError {
    pub error: Error,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
}

impl PhysicalDispatchError {
    pub(crate) fn not_started(error: Error, width: usize, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch: BatchDispatch::not_dispatched(width),
            provenance: OutcomeProvenance::failure(origin, DispatchState::NotStarted),
        }
    }

    pub(crate) fn started(error: Error, dispatch: BatchDispatch, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch,
            provenance: OutcomeProvenance::failure(origin, DispatchState::Started),
        }
    }
}

pub type PhysicalDispatchResult =
    std::result::Result<Vec<ExecutorStepResult>, PhysicalDispatchError>;

impl PhysicalBatchExecution<'_> {
    pub fn expected_dispatch(&self) -> BatchDispatch {
        self.batch.expected_dispatch()
    }

    pub fn validate(&self) -> Result<()> {
        self.batch.validate()?;
        if self.batch.rows.len() != self.scheduled.len()
            || self.scheduled.len() != self.requests.len()
        {
            return Err(Error::InferenceError(
                "physical executor inputs do not match the batch width".to_string(),
            ));
        }

        let expected = self
            .batch
            .rows
            .iter()
            .map(|row| ((row.plan_id, row.session.clone()), &row.work))
            .collect::<HashMap<_, _>>();
        let mut scheduled_ids = HashSet::with_capacity(self.scheduled.len());
        for scheduled in self.scheduled {
            let key = (scheduled.plan_id, scheduled.session_key());
            let work = expected.get(&key).ok_or_else(|| {
                Error::InferenceError(
                    "scheduled work is not present in the physical batch envelope".to_string(),
                )
            })?;
            if **work != scheduled.work {
                return Err(Error::InferenceError(
                    "scheduled work differs from the physical batch quantum".to_string(),
                ));
            }
            if !scheduled_ids.insert(scheduled.request_id.as_str()) {
                return Err(Error::InferenceError(
                    "physical executor inputs contain a duplicate request".to_string(),
                ));
            }
        }

        let request_ids = self
            .requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<HashSet<_>>();
        if request_ids.len() != self.requests.len() || request_ids != scheduled_ids {
            return Err(Error::InferenceError(
                "physical executor request snapshots do not match scheduled rows".to_string(),
            ));
        }

        let is_prefill = self.scheduled[0].is_prefill;
        if self
            .scheduled
            .iter()
            .any(|scheduled| scheduled.is_prefill != is_prefill)
        {
            return Err(Error::InferenceError(
                "one physical batch cannot mix prefill and decode dispatch".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_prefill(&self) -> bool {
        self.scheduled
            .first()
            .is_some_and(|scheduled| scheduled.is_prefill)
    }
}

pub trait ModelExecutor: Send + Sync {
    /// Effective loaded-model/request/backend execution profile. Executors
    /// that cannot prove their behavior return `None` and therefore remain on
    /// the conservative compatibility path.
    fn execution_profile(&self, _request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        None
    }

    /// Effective capabilities. The default is deliberately conservative so an
    /// executor must opt in before the scheduler relies on incremental or batch behavior.
    fn execution_capabilities(&self, request: &EngineCoreRequest) -> ExecutionCapabilities {
        self.execution_profile(request)
            .map(|profile| profile.capabilities())
            .unwrap_or_default()
    }

    /// Execute one already-validated physical batch transaction. Native
    /// tensor adapters override this boundary; compatibility executors retain
    /// their existing phase methods at width one.
    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        let dispatch = execution.expected_dispatch();
        let result = if execution.is_prefill() {
            self.execute_prefill(execution.requests, execution.scheduled)
        } else {
            self.execute_decode(execution.requests, execution.scheduled)
        };
        result
            .map(|mut outputs| {
                let actual_dispatch = if !outputs.is_empty()
                    && outputs
                        .iter()
                        .all(|output| output.provenance.dispatch_state == DispatchState::NotStarted)
                {
                    BatchDispatch::not_dispatched(width)
                } else {
                    dispatch
                };
                for output in &mut outputs {
                    output.dispatch = actual_dispatch;
                }
                outputs
            })
            .map_err(|error| PhysicalDispatchError::started(error, dispatch, FailureOrigin::Model))
    }

    /// Execute prefill pass for newly admitted or in-progress prefill requests.
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Execute decode pass for running requests.
    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;

    /// Cleanup transient per-request state held by the executor backend.
    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }

    /// Cleanup state for one exact request incarnation. Legacy executors may
    /// conservatively clear all state for the public request ID.
    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        self.cleanup_request(&session.request_id)
    }

    /// Purge model-owned reusable cache state before one model is unloaded.
    fn purge_model_cache(&self, _variant: ModelVariant) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }
}

/// Proof returned after an executor cache cleanup request. Preemption may only
/// recompute when the executor confirms that the exact session no longer owns
/// tensor cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReleaseReport {
    pub confirmed: bool,
    pub released_sessions: usize,
}

impl CacheReleaseReport {
    pub const fn confirmed(released_sessions: usize) -> Self {
        Self {
            confirmed: true,
            released_sessions,
        }
    }

    pub const fn unconfirmed() -> Self {
        Self {
            confirmed: false,
            released_sessions: 0,
        }
    }
}

pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: Mutex<HashMap<SessionKey, ActiveChatDecode>>,
    asr_decode_states: Mutex<HashMap<SessionKey, ActiveAsrDecode>>,
    qwen_tts_decode_states: Mutex<HashMap<SessionKey, ActiveQwenTtsDecode>>,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            initialized: false,
            loaded_tts_model: None,
            chat_decode_states: Mutex::new(HashMap::new()),
            asr_decode_states: Mutex::new(HashMap::new()),
            qwen_tts_decode_states: Mutex::new(HashMap::new()),
        }
    }

    fn qwen_model_for_request(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<(Arc<Qwen3TtsModel>, Option<QwenTtsModelLease>)> {
        if let Some(lease) = request.prepared_qwen_tts_model_lease_for_executor()? {
            return Ok((lease.model_arc(), Some(lease)));
        }
        if let Some(registry) = &self.config.model_registry {
            let variant = request.model_variant.ok_or_else(|| {
                Error::InferenceError("Qwen TTS request is missing model variant".to_string())
            })?;
            let lease = registry.try_get_qwen_tts_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("Qwen TTS model {variant} is not loaded"))
            })?;
            return Ok((lease.model_arc(), Some(lease)));
        }
        self.loaded_tts_model
            .clone()
            .map(|model| (model, None))
            .ok_or_else(|| Error::InferenceError("Executor model not initialized".to_string()))
    }

    fn asr_model_for_request(
        &self,
        request: &EngineCoreRequest,
        variant: ModelVariant,
    ) -> Result<(Arc<NativeAsrModel>, AsrModelLease)> {
        if let Some(lease) = request.prepared_asr_model_lease_for_executor()? {
            return Ok((lease.model_arc(), lease));
        }
        self.with_registry(|registry| {
            let lease = registry.try_get_asr_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
            Ok((lease.model_arc(), lease))
        })
    }

    fn with_registry<T>(&self, f: impl FnOnce(&ModelRegistry) -> Result<T>) -> Result<T> {
        let registry =
            self.config.model_registry.as_ref().ok_or_else(|| {
                Error::InferenceError("Model registry is not configured".to_string())
            })?;
        f(registry)
    }

    fn run_blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        let run_catching_panic = || {
            let unwind_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            match unwind_result {
                Ok(result) => result,
                Err(payload) => {
                    let message = panic_payload_to_string(payload.as_ref());
                    error!("Model execution panicked: {message}");
                    Err(Error::InferenceError(format!(
                        "Model execution panicked: {message}"
                    )))
                }
            }
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                // Long-running CPU inference should not monopolize Tokio workers; this allows
                // async tasks (including SSE stream forwarding) to continue making progress.
                tokio::task::block_in_place(run_catching_panic)
            }
            _ => run_catching_panic(),
        }
    }
}

fn is_isolated_continuous_model_quantum(scheduled: &[ScheduledRequest]) -> bool {
    scheduled.len() == 1 && !scheduled[0].is_prefill && scheduled[0].num_tokens > 1
}

fn resolved_resumable_prefill_mode(
    chunking_enabled: bool,
    exact_model_proof: Option<bool>,
) -> PrefillMode {
    if chunking_enabled && exact_model_proof == Some(true) {
        PrefillMode::Incremental
    } else {
        PrefillMode::Full
    }
}

impl ModelExecutor for NativeExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let variant = request.model_variant?;
        let mut profile = ExecutionProfile::fail_closed(
            self.config.backend,
            Some(variant),
            ExecutionMode::Atomic,
        );
        profile.compute_dtype = self.config.dtype.clone();
        profile.kv_dtype = self.config.kv_cache_dtype.clone();
        profile.cache_namespace = Some(format!(
            "{}:{}:{}:{}",
            variant,
            self.config.backend.as_str(),
            self.config.dtype,
            self.config.kv_cache_dtype
        ));

        let loaded_incremental = match request.task_type {
            super::types::TaskType::Chat => {
                request
                    .prepared_chat_model_for_executor()
                    .ok()
                    .map(|model| match model.as_ref() {
                        NativeChatModel::Qwen3(model) => model.supports_incremental_decode(),
                        NativeChatModel::Qwen35(model) => model.supports_incremental_decode(),
                        NativeChatModel::Qwen38(model) => model.supports_incremental_decode(),
                        NativeChatModel::Gemma3(model) => model.supports_incremental_decode(),
                        NativeChatModel::Lfm2(_) => false,
                    })
            }
            super::types::TaskType::ASR => request
                .prepared_asr_model_for_executor()
                .ok()
                .flatten()
                .or_else(|| {
                    self.config
                        .model_registry
                        .as_ref()
                        .and_then(|registry| registry.try_get_asr(variant))
                })
                .map(|model| model.supports_incremental_decode()),
            super::types::TaskType::TTS => {
                let loaded = request
                    .prepared_qwen_tts_model_for_executor()
                    .ok()
                    .flatten()
                    .or_else(|| {
                        self.config
                            .model_registry
                            .as_ref()
                            .and_then(|registry| registry.try_get_qwen_tts(variant))
                    })
                    .is_some()
                    || (self.config.model_registry.is_none() && self.loaded_tts_model.is_some());
                loaded.then_some(variant.family() == crate::catalog::ModelFamily::Qwen3Tts)
            }
            super::types::TaskType::SpeechToSpeech => self
                .config
                .model_registry
                .as_ref()
                .and_then(|registry| registry.try_get_audio_chat(variant))
                .map(|_| false),
        };
        let continuous_chat_batch = matches!(request.task_type, super::types::TaskType::Chat)
            && request
                .prepared_chat_model_for_executor()
                .ok()
                .is_some_and(|model| model.supports_continuous_decode_batch())
            && request.execution_adapter_binding().is_some_and(|binding| {
                binding
                    .stages
                    .iter()
                    .any(|stage| stage.batch_mode == NativeBatchMode::Continuous)
            });
        profile.resolved_from_loaded_model = loaded_incremental.is_some();
        let implementation_incremental =
            loaded_incremental.unwrap_or_else(|| match request.task_type {
                super::types::TaskType::Chat => {
                    matches!(
                        variant.family(),
                        crate::catalog::ModelFamily::Qwen35Chat
                            | crate::catalog::ModelFamily::Qwen38Chat
                    ) || matches!(
                        variant,
                        ModelVariant::Qwen306B
                            | ModelVariant::Qwen306B4Bit
                            | ModelVariant::Qwen317B
                            | ModelVariant::Qwen317B4Bit
                    )
                }
                super::types::TaskType::ASR => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Asr
                }
                super::types::TaskType::TTS => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Tts
                }
                super::types::TaskType::SpeechToSpeech => false,
            });
        let resumable_prefill_proof =
            if matches!(request.task_type, super::types::TaskType::Chat) {
                request
                    .prepared_chat_model_for_executor()
                    .ok()
                    .map(|model| model.supports_resumable_prefill())
            } else {
                None
            };

        if implementation_incremental
            && (!matches!(request.task_type, super::types::TaskType::ASR) || request.streaming)
        {
            profile.mode = ExecutionMode::Sequence;
            // Scheduler-level spans require a stronger capability than
            // incremental decode: the exact loaded family must publish a
            // resumable prefill safe point. Unsupported families remain full.
            profile.prefill = resolved_resumable_prefill_mode(
                self.config.enable_chunked_prefill,
                resumable_prefill_proof,
            );
            profile.incremental_decode = true;
            profile.recompute_safe = profile.resolved_from_loaded_model;
            profile.cache_release_safe = profile.resolved_from_loaded_model;
        }
        if matches!(request.task_type, super::types::TaskType::ASR) {
            // Long audio can switch to a full chunk-plan operation after media
            // decode, so cancellation is conservatively operation-boundary.
            profile.cancellation = CancellationGranularity::OperationBoundary;
        }

        if continuous_chat_batch {
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::Continuous;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = self.config.max_tensor_batch_size.max(1);
        } else {
            let request_parallel_width = if can_parallelize_requests(self.config.backend) {
                self.config.request_parallelism.max(1)
            } else {
                1
            };
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.concurrency = if request_parallel_width > 1 {
                ConcurrencyClass::Batchable
            } else {
                ConcurrencyClass::Exclusive
            };
            profile.max_batch_size = request_parallel_width;
        }
        if request.managed_cache_runtime().is_some() {
            profile.cache_mode = CacheMode::ExternalPaged;
        }
        if matches!(request.task_type, super::types::TaskType::Chat) {
            profile.preferred_decode_tokens = request
                .prepared_chat_model_for_executor()
                .ok()
                .and_then(|model| match model.as_ref() {
                    NativeChatModel::Qwen38(model) => Some(model.preferred_decode_tokens()),
                    _ => None,
                })
                .unwrap_or(1);
        }
        Some(profile)
    }

    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        let expected_dispatch = execution.expected_dispatch();
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        if !self.initialized {
            return Err(PhysicalDispatchError::not_started(
                Error::InferenceError("Executor not initialized".into()),
                width,
                FailureOrigin::ExecutorValidation,
            ));
        }
        if execution.batch.mode == NativeBatchMode::Static {
            if !execution.is_prefill()
                || execution.batch.lane.capability_id != "tts"
                || execution
                    .requests
                    .iter()
                    .any(|request| request.task_type != super::types::TaskType::TTS)
            {
                return Err(PhysicalDispatchError::not_started(
                    Error::InferenceError(
                        "static tensor batch was routed to an incompatible native stage"
                            .to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if execution.scheduled.len() > self.config.max_tensor_batch_size.max(1) {
                return Err(PhysicalDispatchError::not_started(
                    Error::Overloaded(
                        "static tensor batch exceeds the backend width cap".to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            return Err(PhysicalDispatchError::not_started(
                Error::InferenceError(
                    "no loaded physical static-batch implementation is registered".to_string(),
                ),
                width,
                FailureOrigin::ExecutorValidation,
            ));
        }
        if execution.batch.mode == NativeBatchMode::Continuous {
            if execution.is_prefill()
                || execution.batch.lane.capability_id != "chat"
                || execution
                    .requests
                    .iter()
                    .any(|request| request.task_type != super::types::TaskType::Chat)
            {
                return Err(PhysicalDispatchError::not_started(
                    Error::InferenceError(
                        "continuous tensor batch was routed to an incompatible native stage"
                            .to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if execution.scheduled.len() > self.config.max_tensor_batch_size.max(1) {
                return Err(PhysicalDispatchError::not_started(
                    Error::Overloaded(
                        "continuous tensor batch exceeds the backend width cap".to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if is_isolated_continuous_model_quantum(execution.scheduled) {
                // An isolated model-preferred quantum is still planned through
                // the continuous stage so it can yield back to shared
                // membership afterwards, but its model work is scalar/MTP and
                // must use the existing transactional scalar handler. This
                // keeps tensor-batch telemetry truthful and preserves the
                // shared handler's one-token-per-row invariant.
                let result = self.execute_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                );
                if result.as_ref().is_ok_and(|outputs| {
                    outputs.iter().all(|output| output.output.error.is_none())
                }) {
                    crate::engine::metrics::record_engine_chat_model_dispatch(false, 1);
                }
                return result.map_err(|error| {
                    PhysicalDispatchError::started(
                        error,
                        BatchDispatch::serial(),
                        FailureOrigin::Model,
                    )
                });
            }
            return self
                .execute_continuous_chat_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                )
                .map_err(|error| {
                    PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
                });
        }
        let result = self.execute_requests_with_rows(
            execution.requests,
            execution.scheduled,
            Some(&execution.batch.rows),
        );
        result.map_err(|error| {
            PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
        })
    }

    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        if self.config.model_registry.is_none() {
            let device = self.config.backend_context.device.clone();
            let model = Qwen3TtsModel::load(
                &self.config.models_dir,
                device,
                self.config.kv_page_size.max(1),
                &self.config.kv_cache_dtype,
            )?;
            self.loaded_tts_model = Some(Arc::new(model));
            debug!(
                "Native executor loaded TTS model from {:?}",
                self.config.models_dir
            );
        } else {
            debug!("Native executor will use shared model registry");
        }
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        let mut chat = self
            .chat_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("chat decode state mutex poisoned".to_string()))?;
        let mut asr = self
            .asr_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("ASR decode state mutex poisoned".to_string()))?;
        let mut tts = self.qwen_tts_decode_states.lock().map_err(|_| {
            Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
        })?;
        chat.clear();
        asr.clear();
        tts.clear();
        drop((chat, asr, tts));
        self.initialized = false;
        self.loaded_tts_model = None;
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        let (Ok(mut chat), Ok(mut asr), Ok(mut tts)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.qwen_tts_decode_states.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };

        let mut released = 0usize;
        released = released.saturating_add(retain_other_sessions_locked(&mut chat, request_id));
        released = released.saturating_add(retain_other_sessions_locked(&mut asr, request_id));
        released = released.saturating_add(retain_other_sessions_locked(&mut tts, request_id));
        CacheReleaseReport::confirmed(released)
    }

    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        let (Ok(mut chat), Ok(mut asr), Ok(mut tts)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.qwen_tts_decode_states.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };

        let released = usize::from(chat.remove(session).is_some())
            .saturating_add(usize::from(asr.remove(session).is_some()))
            .saturating_add(usize::from(tts.remove(session).is_some()));
        CacheReleaseReport::confirmed(released)
    }

    fn purge_model_cache(&self, _variant: ModelVariant) -> CacheReleaseReport {
        CacheReleaseReport::confirmed(0)
    }
}

/// Unified executor that wraps a model executor implementation.
#[derive(Clone)]
struct BatchWorkspaceContext {
    backend: BackendKind,
    authority: Arc<ResourceAuthority>,
}

#[derive(Clone)]
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
    batch_workspace: Option<BatchWorkspaceContext>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        let batch_workspace =
            config
                .resource_authority
                .as_ref()
                .map(|authority| BatchWorkspaceContext {
                    backend: config.backend,
                    authority: authority.clone(),
                });
        Self {
            inner: Arc::new(RwLock::new(Box::new(NativeExecutor::new(config)))),
            batch_workspace,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(executor: Box<dyn ModelExecutor>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
            batch_workspace: None,
        }
    }

    pub(super) fn reserve_batch_workspace(
        &self,
        batch: &PhysicalBatch,
    ) -> Result<Option<BatchWorkspaceLease>> {
        if batch.workspace.workspace_bytes()? == 0 {
            return Ok(None);
        }
        let context = self.batch_workspace.as_ref().ok_or_else(|| {
            Error::Overloaded(
                "physical batch requires workspace but no resource authority is installed"
                    .to_string(),
            )
        })?;
        if batch.lane.backend != context.backend {
            return Err(Error::InvalidInput(
                "physical batch workspace backend does not match its executor".to_string(),
            ));
        }
        context
            .authority
            .reserve_batch_workspace(batch.lane.execution_group, batch.batch_id, batch.workspace)
            .map(Some)
    }

    /// Execute one exact physical batch envelope.
    pub async fn execute_physical_batch(
        &self,
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> PhysicalDispatchResult {
        let executor = self.inner.read().await;
        executor.execute_physical_batch(PhysicalBatchExecution {
            batch,
            requests,
            scheduled,
        })
    }

    pub async fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let executor = self.inner.read().await;
        executor.execution_profile(request)
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }

    /// Cleanup transient backend state for a completed/aborted request.
    pub async fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_request(request_id)
    }

    pub async fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_session(session)
    }

    pub async fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.purge_model_cache(variant)
    }
}

fn retain_other_sessions_locked<T>(states: &mut HashMap<SessionKey, T>, request_id: &str) -> usize {
    let before = states.len();
    states.retain(|session, _| session.request_id != request_id);
    before.saturating_sub(states.len())
}

/// Decode base64-encoded audio to samples.
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    let (samples, _) = decode_audio_base64_with_rate(audio_b64)?;
    Ok(samples)
}

fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    audio::decode_audio_base64_with_rate(audio_b64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::request::StreamStagingBuffer;
    use crate::engine::{
        CapacitySource, ManagedCacheDomainReservation, ManagedCacheReservation,
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, ResourceAmount,
    };
    use crate::model::ModelVariant;
    use base64::Engine;

    #[derive(Debug)]
    struct FixedCapacityProvider {
        capacity: ResourceVector,
    }

    impl PhysicalCapacityProvider for FixedCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: self.capacity,
                source: CapacitySource::Test,
            }
        }
    }

    fn qwen38_test_arena(generation: u32) -> KvArenaId {
        KvArenaId {
            model_instance: super::super::ModelInstanceId::new(38),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
        }
    }

    #[test]
    fn only_multi_token_solo_decode_uses_the_scalar_continuous_route() {
        let scheduled = |num_tokens: usize, is_prefill: bool| ScheduledRequest {
            plan_id: 1,
            request_id: "route".to_string(),
            sequence_id: 1,
            num_tokens,
            is_prefill,
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: if is_prefill {
                    crate::engine::SequencePhase::Prefill
                } else {
                    crate::engine::SequencePhase::Decode
                },
                input: crate::engine::InputRange {
                    start: 0,
                    end: num_tokens,
                },
                max_output_steps: num_tokens,
            },
        };

        assert!(is_isolated_continuous_model_quantum(&[scheduled(4, false)]));
        assert!(!is_isolated_continuous_model_quantum(&[scheduled(
            1, false
        )]));
        assert!(!is_isolated_continuous_model_quantum(&[scheduled(4, true)]));
        assert!(!is_isolated_continuous_model_quantum(&[
            scheduled(1, false),
            scheduled(1, false),
        ]));
    }

    fn qwen38_test_reservation(domains: &[(CacheDomainId, KvArenaId)]) -> ManagedCacheReservation {
        ManagedCacheReservation {
            txn_id: 1,
            session: SessionKey::new("qwen38-domain-selection".into(), 1),
            domains: domains
                .iter()
                .map(|(domain, arena)| ManagedCacheDomainReservation {
                    arena: *arena,
                    domain: *domain,
                    expected_version: 0,
                    expected_committed_tokens: 0,
                    execution_start_tokens: 0,
                    target_committed_tokens: 1,
                    target_window_start: 0,
                    first_page_offset: 0,
                    provisional_groups: Vec::new(),
                    writable_blocks: Vec::new(),
                })
                .collect(),
            tensor_state: None,
        }
    }

    #[test]
    fn qwen38_managed_group_selection_resolves_exact_target_and_optional_mtp() {
        let target_arena = qwen38_test_arena(1);
        let mtp_arena = qwen38_test_arena(2);
        let target_group = KvGroupId::new(1);
        let mtp_group = KvGroupId::new(1);
        let dual_groups = [
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_group, mtp_arena),
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_group, target_arena),
        ];
        let dual_reservation = qwen38_test_reservation(&[
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_arena),
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_arena),
        ]);
        assert_eq!(
            qwen38_managed_group_ids(&dual_groups, &dual_reservation).unwrap(),
            (target_group, Some(mtp_group))
        );

        let target_groups = [(QWEN38_TARGET_ATTENTION_DOMAIN, target_group, target_arena)];
        let target_reservation =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert_eq!(
            qwen38_managed_group_ids(&target_groups, &target_reservation).unwrap(),
            (target_group, None)
        );
    }

    #[test]
    fn qwen38_managed_group_selection_rejects_half_resolved_mtp_domain() {
        let target_arena = qwen38_test_arena(1);
        let mtp_arena = qwen38_test_arena(2);
        let groups = [
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(1),
                target_arena,
            ),
            (QWEN38_MTP_ATTENTION_DOMAIN, KvGroupId::new(1), mtp_arena),
        ];
        let target_only =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert!(qwen38_managed_group_ids(&groups, &target_only).is_err());

        let target_group_only = &groups[..1];
        let reservation_with_mtp = qwen38_test_reservation(&[
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_arena),
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_arena),
        ]);
        assert!(qwen38_managed_group_ids(target_group_only, &reservation_with_mtp).is_err());
    }

    #[test]
    fn qwen38_managed_group_selection_rejects_missing_duplicate_or_foreign_target() {
        let target_arena = qwen38_test_arena(1);
        let foreign_arena = qwen38_test_arena(2);
        let empty_reservation = qwen38_test_reservation(&[]);
        assert!(qwen38_managed_group_ids(&[], &empty_reservation).is_err());

        let target_reservation =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert!(qwen38_managed_group_ids(&[], &target_reservation).is_err());

        let duplicate_groups = [
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(1),
                target_arena,
            ),
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(2),
                target_arena,
            ),
        ];
        assert!(qwen38_managed_group_ids(&duplicate_groups, &target_reservation).is_err());

        let foreign_groups = [(
            QWEN38_TARGET_ATTENTION_DOMAIN,
            KvGroupId::new(1),
            foreign_arena,
        )];
        assert!(qwen38_managed_group_ids(&foreign_groups, &target_reservation).is_err());
    }

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.backend, config.backend_context.backend_kind);
    }

    #[test]
    fn atomic_invocation_leases_reject_non_atomic_or_tensor_stages() {
        let profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
        let scalar = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "atomic.scalar",
            &profile,
            NativeBatchMode::None,
        );
        let atomic = WorkUnit::AtomicJob {
            kind: "test".to_string(),
        };
        validate_atomic_scalar_invocation_stage(&scalar, &atomic).unwrap();

        let sequence = WorkUnit::SequenceStep {
            phase: super::super::SequencePhase::Decode,
            input: super::super::InputRange { start: 0, end: 1 },
            max_output_steps: 1,
        };
        assert!(validate_atomic_scalar_invocation_stage(&scalar, &sequence).is_err());

        let tensor = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "atomic.tensor",
            &profile,
            NativeBatchMode::Static,
        );
        assert!(validate_atomic_scalar_invocation_stage(&tensor, &atomic).is_err());
    }

    #[test]
    fn test_worker_config_from_engine_config_uses_backend_context() {
        let engine = EngineCoreConfig {
            backend: BackendKind::Cpu,
            ..Default::default()
        };

        let config = WorkerConfig::from(&engine);
        assert_eq!(config.backend, config.backend_context.backend_kind);
        assert_eq!(config.request_parallelism, 1);
        assert_eq!(
            config.backend_context.source,
            BackendSelectionSource::Config
        );
    }

    #[test]
    fn test_request_parallelism_defaults_are_backend_aware() {
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cuda, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, Some(3)),
            3
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, Some(3)),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_batch_request_parallelism(BackendKind::Cuda, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_batch_request_parallelism(BackendKind::Cuda, Some(4)),
            4
        );
        assert_eq!(
            WorkerConfig::resolve_batch_request_parallelism(BackendKind::Metal, None),
            1
        );
    }

    #[test]
    fn automatic_tensor_width_follows_resolved_backend_and_not_scheduler_rows() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let engine = EngineCoreConfig {
                backend,
                max_batch_size: 19,
                ..EngineCoreConfig::default()
            };
            let worker = WorkerConfig::from(&engine);
            assert_eq!(
                worker.max_tensor_batch_size,
                engine.max_tensor_batch_size.resolve(worker.backend)
            );
            assert_eq!(worker.request_parallelism, 1);
            assert_eq!(engine.max_batch_size, 19);
        }
    }

    #[test]
    fn tensor_batch_caps_are_backend_conservative() {
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cpu), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Metal), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cuda), 32);
    }

    #[test]
    fn physical_batch_workspace_uses_the_backend_resource_domain_and_releases() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut capacity = ResourceVector::zero();
            match backend {
                BackendKind::Cpu => capacity.host_bytes = ResourceAmount::Known(64),
                BackendKind::Metal => capacity.unified_bytes = ResourceAmount::Known(64),
                BackendKind::Cuda => {
                    capacity.host_bytes = ResourceAmount::Known(64);
                    capacity.device_bytes = ResourceAmount::Known(64);
                }
            }
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let mut executor = UnifiedExecutor::new_for_test(Box::new(NativeExecutor::new(
                WorkerConfig::default(),
            )));
            executor.batch_workspace = Some(BatchWorkspaceContext {
                backend,
                authority: authority.clone(),
            });
            let lane = super::super::BatchLaneKey {
                execution_group: super::super::ExecutionGroupId::new(7),
                model_instance: super::super::ModelInstanceId::new(8),
                adapter_instance: super::super::AdapterInstanceId::new(9),
                adapter_abi: super::super::AdapterAbiRevision::new(1),
                capability_id: "test".to_string(),
                stage_id: super::super::StageId::new(1),
                backend,
                device_ordinal: None,
                compute_dtype: "f32".to_string(),
                state_dtype: "f32".to_string(),
                tensor_layout: "exact".to_string(),
                quantization: "none".to_string(),
                state_schema: "none".to_string(),
                kernel_mode: "test".to_string(),
                semantic_mode: "test".to_string(),
                shape_bucket: "exact.1".to_string(),
            };
            let expected_workspace = match backend {
                BackendKind::Cpu => ResourceVector {
                    host_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Metal => ResourceVector {
                    unified_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Cuda => ResourceVector {
                    host_bytes: ResourceAmount::Known(3),
                    device_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
            };
            let batch = PhysicalBatch {
                batch_id: super::super::BatchId::new(10),
                lane: lane.clone(),
                mode: NativeBatchMode::None,
                budget: super::super::BatchBudget::width_one(),
                rows: vec![super::super::ReadyQuantum {
                    plan_id: 1,
                    session: SessionKey::new("workspace".to_string(), 1),
                    lane,
                    work: super::super::WorkUnit::AtomicJob {
                        kind: "test".to_string(),
                    },
                    cost: super::super::WorkCost::new(1, 1, 8),
                    managed_cache: None,
                }],
                materialized_tensor_elements: 1,
                workspace: expected_workspace,
            };

            let workspace = executor
                .reserve_batch_workspace(&batch)
                .unwrap()
                .expect("workspace lease");
            assert_eq!(workspace.resources(), expected_workspace);
            assert_eq!(authority.snapshot().reservations, 1);
            drop(workspace);
            assert_eq!(authority.snapshot().reservations, 0);
        }
    }

    #[test]
    fn test_run_blocking_converts_panic_to_error() {
        let result = NativeExecutor::run_blocking(|| -> Result<()> {
            panic!("executor panic sentinel");
        });

        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error from panic");
        };
        assert!(message.contains("executor panic sentinel"));
    }

    #[test]
    fn test_run_blocking_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result =
            runtime.block_on(async { NativeExecutor::run_blocking(|| Ok::<_, Error>(())) });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_stages_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result = runtime.block_on(async {
            let tx = StreamStagingBuffer::default();
            let mut sequence = 0usize;
            NativeExecutor::stream_audio(
                &tx,
                "req-1",
                &mut sequence,
                vec![0.1, -0.1],
                24_000,
                false,
            )?;
            let chunk = tx
                .take()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::InferenceError("missing staged chunk".to_string()))?;
            if chunk.request_id != "req-1" || chunk.sequence != 0 || chunk.samples.len() != 2 {
                return Err(Error::InferenceError(
                    "unexpected streamed chunk payload".to_string(),
                ));
            }
            Ok::<(), Error>(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_tts_params_uses_model_native_auto_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        request.params.max_tokens = 0;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn test_to_tts_params_clamps_to_model_native_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        request.params.max_tokens = 50_000;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn resumable_prefill_classification_is_opt_in_and_fail_closed() {
        let mut request = EngineCoreRequest::chat(vec![crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::User,
            content: "chunk me".to_string(),
        }]);
        request.model_variant = Some(ModelVariant::Qwen3827BFp8);
        request.streaming = true;

        let default_executor = NativeExecutor::new(WorkerConfig::default());
        assert_eq!(
            default_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );

        let chunking_executor = NativeExecutor::new(WorkerConfig {
            enable_chunked_prefill: true,
            ..Default::default()
        });
        assert_eq!(
            chunking_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );

        // Sibling hybrid without a resumable prefill path stays full.
        request.model_variant = Some(ModelVariant::Qwen3508BGguf);
        assert_eq!(
            chunking_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );
    }

    #[test]
    fn resumable_prefill_mode_requires_exact_positive_model_proof() {
        assert_eq!(
            resolved_resumable_prefill_mode(true, Some(true)),
            PrefillMode::Incremental
        );
        assert_eq!(
            resolved_resumable_prefill_mode(false, Some(true)),
            PrefillMode::Full
        );
        assert_eq!(
            resolved_resumable_prefill_mode(true, Some(false)),
            PrefillMode::Full
        );
        assert_eq!(
            resolved_resumable_prefill_mode(true, None),
            PrefillMode::Full
        );
    }

    #[test]
    fn unloaded_models_cannot_claim_native_batch_capability() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let config = WorkerConfig {
                backend,
                request_parallelism: 4,
                ..Default::default()
            };
            let executor = NativeExecutor::new(config);
            let mut request = EngineCoreRequest::tts("batch me");
            request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);

            let profile = executor.execution_profile(&request).unwrap();
            assert_eq!(profile.backend, backend);
            assert_eq!(profile.mode, ExecutionMode::Sequence);
            assert_eq!(profile.prefill, PrefillMode::Full);
            assert!(!profile.capabilities().native_batch);
            assert_eq!(profile.decode_batch, NativeBatchMode::None);
            let expected_parallelism = if backend == BackendKind::Metal { 1 } else { 4 };
            assert_eq!(profile.max_batch_size, expected_parallelism);
            assert_eq!(
                profile.concurrency,
                if expected_parallelism > 1 {
                    ConcurrencyClass::Batchable
                } else {
                    ConcurrencyClass::Exclusive
                }
            );
            request.streaming = true;
            assert!(!executor.execution_capabilities(&request).native_batch);
            request.streaming = false;
            request.reference_audio = Some("reference".to_string());
            assert!(!executor.execution_capabilities(&request).native_batch);
        }
    }

    #[test]
    fn model_session_results_declare_safe_points_and_terminal_semantics() {
        let sequence = ModelSessionResult::sequence(ExecutorOutput {
            request_id: "sequence".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert_eq!(
            sequence.disposition,
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        );
        assert!(sequence.safe_point);

        let atomic = ModelSessionResult::atomic(ExecutorOutput {
            request_id: "atomic".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert!(matches!(
            atomic.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::InvalidOutput,
                ..
            })
        ));
        assert!(atomic.output.finished);

        let cancelled =
            ModelSessionResult::cancelled(ExecutorOutput::cancelled("cancelled".to_string()));
        assert_eq!(
            cancelled.disposition,
            ExecutionDisposition::Finished(FinishReason::Cancelled)
        );
        assert!(cancelled.output.error.is_none());
    }

    #[test]
    fn decode_audio_base64_with_rate_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            // 2 stereo frames: [L,R]=[0.25,0.75] then [0.5,-0.5]
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let (samples, sample_rate) =
            decode_audio_base64_with_rate(&b64).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        // After downmixing, expected mono values are averages: 0.5 and 0.0.
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample was {}", samples[1]);
    }

    #[test]
    fn decode_request_audio_with_rate_accepts_raw_audio_bytes() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.25f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let request = EngineCoreRequest::asr_bytes(wav_bytes);
        let (samples, sample_rate) =
            audio::decode_request_audio_with_rate(&request).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.25).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(
            (samples[1] + 0.25).abs() < 0.02,
            "second sample was {}",
            samples[1]
        );
    }
}

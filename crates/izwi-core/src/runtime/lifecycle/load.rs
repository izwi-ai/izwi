use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use tracing::{info, warn};

use crate::backends::kv::managed_kv_backend_compiled;
use crate::backends::BackendKind;
use crate::config::ContextLengthPreference;
use crate::engine::resources::{capacity_shortfall_reason, state_shortfall_reason};
use crate::engine::{
    AdapterInstanceId, CacheMode, ModelInstanceId, ReservationClass, ReservationOwner,
    ResourceAmount, ResourceLease, ResourceVector, StageDescriptor, StateFitReserve,
};
use crate::error::{Error, Result};
use crate::kernels::numeric_check::PrecisionReloadGate;
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InferenceStateContract,
    InvocationStateCapacity, InvocationWorkspaceBindingV2, InvocationWorkspaceDomain,
    InvocationWorkspaceKeyV2, InvocationWorkspaceRuntimeV2, InvocationWorkspaceSet,
    RetainedStateCapability, RetainedStateRuntimeV2, RetainedStateUseV2, StateCapacityAxis,
    StateDomainId, StateDomainSpec, StateScope,
};
use crate::kv::InferenceStateContractProvider;
use crate::model::ModelVariant;
use crate::models::architectures::fish_s2::FishS2PhysicalStateSpec;
use crate::models::architectures::qwen3::asr::Qwen3AsrContextBound;
use crate::models::registry::NativeAsrModel;
use crate::models::shared::weights::gguf::TensorStorageInventory;
use crate::runtime::adapters::{
    CapabilityKind, LoadedExecutionContract, LoadedModelBundleDraft, LoadedStatePublication,
};
use crate::runtime::lifecycle::controller::{
    ModelLifecycleController, SharedLoadFailure, SharedLoadOutcome,
};
use crate::runtime::service::RuntimeService;

mod admission;
#[path = "asr_memory.rs"]
mod asr_memory;
mod checkpoint;
mod invocation;
#[path = "qwen36moe_memory.rs"]
mod qwen36moe_memory;
#[path = "qwen38_memory.rs"]
mod qwen38_memory;
mod state;
#[path = "tts_memory.rs"]
mod tts_memory;
#[cfg(test)]
use admission::{
    admit_asr_request_floor, materialization_tolerance_bytes, request_envelope_resources,
    request_envelope_shortfall, CudaStateFloor,
};
use admission::{
    admit_cuda_state_floor, load_authorization_with_request_floor, portable_preload_bytes,
    reconcile_observed_growth,
};
use checkpoint::{
    census_checkpoint_file, checkpoint_resource_plan, checkpoint_tensor_inventory,
    model_memory_estimate, model_resource_plan, portable_model_memory_estimate,
    qwen38_synthetic_fixture_estimate, retry_load_peak_bytes, CensusFile, CensusTensor,
    StoredDType, PER_TENSOR_INSTANTIATION_SLACK_BYTES, QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES,
    QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES, QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES,
};
#[cfg(test)]
use checkpoint::{
    checkpoint_census, estimate_from_tensor_inventory, fish_s2_resource_plan,
    qwen38_representation_memory_estimate, CheckpointCensus,
};
use invocation::{invocation_state_bytes, LaterStateGroups};
#[cfg(test)]
use invocation::{
    plan_invocation_allocations, validate_physical_publication_backing,
    validate_scratch_only_invocation_publication,
};

pub(super) fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn is_metal_command_buffer_oom(error: &Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("metal error")
        && (message.contains("insufficient memory")
            || message.contains("commandbuffercallbackerroroutofmemory")
            || message.contains("kiogpucommandbuffercallbackerroroutofmemory"))
}

fn select_lru_eviction_candidate(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    active_variants: &HashSet<ModelVariant>,
    pinned_variants: &HashSet<ModelVariant>,
    last_used: &HashMap<ModelVariant, u64>,
) -> Option<ModelVariant> {
    resident_variants
        .iter()
        .copied()
        .filter(|variant| {
            *variant != requested_variant
                && !active_variants.contains(variant)
                && !pinned_variants.contains(variant)
        })
        .min_by(|left, right| {
            last_used
                .get(left)
                .copied()
                .unwrap_or(0)
                .cmp(&last_used.get(right).copied().unwrap_or(0))
                .then_with(|| left.to_string().cmp(&right.to_string()))
        })
}

fn residency_budget_has_capacity(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    max_loaded_models: usize,
) -> bool {
    resident_variants.contains(&requested_variant) || resident_variants.len() < max_loaded_models
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadedAsrStatePublicationRoute {
    Qwen3,
    VibeVoice,
    Whisper,
    Parakeet,
    NemotronOffline,
    GraniteSpeech5,
    LegacyCache,
}

fn loaded_asr_state_publication_route(variant: ModelVariant) -> LoadedAsrStatePublicationRoute {
    match variant.family() {
        crate::catalog::ModelFamily::Qwen3Asr => LoadedAsrStatePublicationRoute::Qwen3,
        crate::catalog::ModelFamily::VibeVoiceAsr => LoadedAsrStatePublicationRoute::VibeVoice,
        crate::catalog::ModelFamily::WhisperAsr => LoadedAsrStatePublicationRoute::Whisper,
        crate::catalog::ModelFamily::ParakeetAsr => LoadedAsrStatePublicationRoute::Parakeet,
        crate::catalog::ModelFamily::NemotronAsr => LoadedAsrStatePublicationRoute::NemotronOffline,
        crate::catalog::ModelFamily::GraniteSpeech5Asr => {
            LoadedAsrStatePublicationRoute::GraniteSpeech5
        }
        _ => LoadedAsrStatePublicationRoute::LegacyCache,
    }
}

/// Parakeet's retained predictor is owned by the exact executor session, so
/// no graph activates managed retained state: sequence graphs read it through
/// their session and the atomic long-form graph leases invocation tensors.
fn parakeet_retained_uses(
    contracts: &[LoadedExecutionContract],
) -> Result<HashMap<[u8; 32], RetainedStateUseV2>> {
    contracts
        .iter()
        .map(|contract| {
            let graph = stage_graph_fingerprint(&contract.stages)?;
            match contract.execution_profile.mode {
                crate::engine::ExecutionMode::Sequence | crate::engine::ExecutionMode::Atomic => {
                    Ok((graph, RetainedStateUseV2::Inactive))
                }
                _ => Err(Error::ModelLoadError(
                    "Parakeet graph has an incompatible execution mode".into(),
                )),
            }
        })
        .collect()
}

fn uses_asr_model_registry(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        crate::catalog::ModelFamily::ParakeetAsr
            | crate::catalog::ModelFamily::WhisperAsr
            | crate::catalog::ModelFamily::Qwen3Asr
            | crate::catalog::ModelFamily::VibeVoiceAsr
            | crate::catalog::ModelFamily::NemotronAsr
            | crate::catalog::ModelFamily::GraniteSpeechAsr
            | crate::catalog::ModelFamily::GraniteSpeech5Asr
            | crate::catalog::ModelFamily::Qwen3ForcedAligner
    )
}

fn kokoro_effective_context_tokens(
    model_context_tokens: usize,
    portable_context_tokens: usize,
) -> Result<u64> {
    let effective = model_context_tokens.min(portable_context_tokens);
    if effective == 0 {
        return Err(Error::ModelLoadError(
            "Kokoro effective context must be greater than zero".into(),
        ));
    }
    u64::try_from(effective)
        .map_err(|_| Error::ModelLoadError("Kokoro effective context exceeds u64".into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortableInvocationContextIntent {
    Automatic { ceiling: u64 },
    Explicit { selected: u64 },
}

fn portable_invocation_context_intent(
    preference: ContextLengthPreference,
    published_context: Option<usize>,
    maximum: u64,
) -> PortableInvocationContextIntent {
    match preference.explicit_tokens() {
        Some(tokens) => PortableInvocationContextIntent::Explicit {
            selected: maximum.min(tokens as u64),
        },
        None => PortableInvocationContextIntent::Automatic {
            ceiling: published_context
                .and_then(|tokens| u64::try_from(tokens).ok())
                .map_or(maximum, |tokens| maximum.min(tokens)),
        },
    }
}

/// `(floor, growth)` budgets of a portable state fit. Required state (the
/// model-authored minimum, an explicit context) is admitted against
/// planning headroom; live headroom (`live`, already clamped to planning)
/// caps only the automatic growth above it. Right after a large load the
/// live sample undercounts memory the OS reclaims on demand (clean
/// file-backed pages of the checkpoint just read, compressible pages):
/// Granite's 10 MiB minimum was refused against 0.8 GB live beside 1 GiB of
/// reserve while planning headroom held gigabytes.
fn state_fit_budgets(
    planning_bytes: u64,
    live: ResourceAmount,
    safety_reserve_bytes: u64,
) -> (u64, u64) {
    let floor = planning_bytes.saturating_sub(safety_reserve_bytes);
    let growth = match live {
        ResourceAmount::Known(bytes) => bytes.saturating_sub(safety_reserve_bytes),
        ResourceAmount::Unknown => floor,
    };
    (floor, growth.min(floor))
}

/// The one-chunk floor of an ASR variant's decoder context (`0` for other
/// variants, and for ASR families without an LLM decoder context).
fn asr_minimum_tokens(variant: ModelVariant) -> u64 {
    super::asr_envelope::minimum_sequence_tokens(variant).unwrap_or(0)
}

/// The retained-state floor of a TTS variant, in tokens per sequence: one
/// prompt plus the family's largest automatic output budget (per CFG domain
/// for VibeVoice). `0` (the page minimum) for every other variant.
fn tts_state_minimum_tokens(variant: ModelVariant) -> u64 {
    let tokens = match variant.family() {
        crate::catalog::ModelFamily::Qwen3Tts => {
            crate::models::architectures::qwen3::tts::qwen3_tts_state_floor_tokens()
        }
        crate::catalog::ModelFamily::VibeVoiceTts => {
            crate::models::architectures::vibevoice::tts::vibevoice_tts_state_floor_tokens()
        }
        crate::catalog::ModelFamily::VoxtralTts => {
            crate::models::architectures::voxtral::tts::VOXTRAL_TTS_STATE_CONTEXT_TOKENS
        }
        _ => 0,
    };
    tokens as u64
}

fn automatic_state_group_budget(available_bytes: u64, remaining_groups: u64) -> u64 {
    available_bytes / remaining_groups.max(1)
}

fn portable_context_reserve_bytes(variant: ModelVariant, configured_reserve_bytes: u64) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;

    // Synthetic fixture loads must not inherit the pinned 27B catalog
    // inference size: their actual residency is tiny, so subtracting it from
    // the 80 GiB catalog value would reserve an absurd request-scoped budget
    // and collapse the portable context fit.
    if variant == ModelVariant::Qwen3827BFp8
        && crate::models::architectures::qwen38::native::synthetic_geometry_enabled()
    {
        return configured_reserve_bytes;
    }
    if variant == ModelVariant::Qwen36Moe35BA3BFp8
        && crate::models::architectures::qwen36moe::native::synthetic_geometry_enabled()
    {
        return configured_reserve_bytes;
    }
    let total_inference_bytes = (variant.memory_required_gb() as f64 * GIB as f64).ceil() as u64;
    let resident_bytes = model_memory_estimate(variant).resident_bytes;
    configured_reserve_bytes.saturating_add(total_inference_bytes.saturating_sub(resident_bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManagedChatCapacityPolicy {
    /// `None` delegates staged transaction width to the engine-wide
    /// `max_staged_transactions` setting.
    staged_transaction_rows: Option<u32>,
    /// How the CUDA KV pool is sized: Qwen3.8 and Qwen3.5/3.6-MoE fit every
    /// retained row's context against resident device headroom; LFM2.5 chat
    /// fits one context, so a small model does not take the GPU's free memory.
    /// Independent of staged transaction width.
    cuda_resident_pool: crate::engine::CudaResidentPool,
}

fn qwen36moe_state_geometry(
    text: &crate::models::architectures::qwen36moe::native::Qwen36MoeTextConfig,
) -> HybridStateGeometry {
    HybridStateGeometry {
        block_count: text.block_count,
        full_attention_interval: text.full_attention_interval,
        ssm_state_size: text.ssm_state_size,
        ssm_group_count: text.ssm_group_count,
        ssm_time_step_rank: text.ssm_time_step_rank,
        ssm_inner_size: text.ssm_time_step_rank * text.ssm_value_head_dim,
        ssm_conv_kernel: text.ssm_conv_kernel,
    }
}

fn managed_chat_capacity_policy(
    variant: ModelVariant,
    backend: BackendKind,
) -> ManagedChatCapacityPolicy {
    use crate::catalog::ModelFamily;
    use crate::engine::CudaResidentPool;
    let cuda_resident_pool = match variant.family() {
        _ if backend != BackendKind::Cuda => CudaResidentPool::Disabled,
        ModelFamily::Qwen38Chat | ModelFamily::Qwen35MoeChat => CudaResidentPool::RetainedRows,
        // Without a fit LFM2.5 grew a 128K-token arena on demand, with a
        // replacement-copy peak and no CUDA context reserve (A1).
        ModelFamily::Lfm2Chat => CudaResidentPool::OneContext,
        _ => CudaResidentPool::Disabled,
    };
    ManagedChatCapacityPolicy {
        staged_transaction_rows: None,
        cuda_resident_pool,
    }
}

/// Gated DeltaNet geometry that sizes one sequence's recurrent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HybridStateGeometry {
    block_count: usize,
    full_attention_interval: usize,
    ssm_state_size: usize,
    ssm_group_count: usize,
    ssm_time_step_rank: usize,
    ssm_inner_size: usize,
    ssm_conv_kernel: usize,
}

/// F32 recurrent and convolution state of one sequence, summed over the
/// linear-attention layers: the bytes the hybrid chat contracts (Qwen3.8,
/// Qwen3.5/3.6-MoE) authorize per retained or staged row.
fn hybrid_recurrent_state_bytes_per_sequence(geometry: HybridStateGeometry) -> Result<u64> {
    let HybridStateGeometry {
        block_count,
        full_attention_interval,
        ssm_state_size,
        ssm_group_count,
        ssm_time_step_rank,
        ssm_inner_size,
        ssm_conv_kernel,
    } = geometry;
    if full_attention_interval == 0
        || ssm_conv_kernel == 0
        || ssm_time_step_rank == 0
        || !ssm_inner_size.is_multiple_of(ssm_time_step_rank)
    {
        return Err(Error::ModelLoadError(
            "hybrid recurrent state geometry is invalid".into(),
        ));
    }
    let linear_layers = block_count - block_count / full_attention_interval;
    let recurrent = ssm_time_step_rank
        .checked_mul(ssm_state_size)
        .and_then(|elements| elements.checked_mul(ssm_inner_size / ssm_time_step_rank));
    let convolution = ssm_state_size
        .checked_mul(ssm_group_count)
        .and_then(|keys| keys.checked_mul(2))
        .and_then(|width| width.checked_add(ssm_inner_size))
        .and_then(|width| width.checked_mul(ssm_conv_kernel - 1));
    recurrent
        .and_then(|recurrent| recurrent.checked_add(convolution?))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(linear_layers))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::ModelLoadError("hybrid recurrent state size overflow".into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelResourcePlan {
    /// Maximum simultaneous memory authorized before physical instantiation.
    load_authorization: ResourceVector,
    /// Long-lived memory retained after publication completes.
    resident_authorization: ResourceVector,
    /// Resident capacity reserved for lazy model-owned allocations. It remains
    /// a pending claim while physical state is fitted before first inference.
    deferred_resident_authorization: ResourceVector,
    /// Metal: host memory the load stages beside its unified authorization,
    /// which then books only what the GPU allocates. It is not part of the
    /// GPU working set; it gates the load only under
    /// `IZWI_METAL_LOAD_ADMISSION=strict`, and is advisory otherwise, like
    /// any host memory (Fish's staged transformer holds 1.6 GB of host copies
    /// beside a 10.7 GB GPU peak).
    host_staging_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMemoryEstimate {
    /// Maximum model-owned memory while tensors are being instantiated.
    load_peak_bytes: u64,
    /// Long-lived model-owned memory after publication.
    resident_bytes: u64,
}

/// The load-domain bytes of a load authorization: host memory on CPU, unified
/// memory on Metal, device memory on CUDA.
fn load_domain_bytes(authorization: ResourceVector, backend: BackendKind) -> ResourceAmount {
    match backend {
        BackendKind::Cpu => authorization.host_bytes,
        BackendKind::Metal => authorization.unified_bytes,
        BackendKind::Cuda => authorization.device_bytes,
    }
}

/// A loading thread's request to admit a precision fallback's reload
/// ([`PrecisionReloadGate`]): the lifecycle answers once it has grown the
/// load's authorization, or refused.
struct PrecisionReloadRequest {
    component: String,
    dtype: candle_core::DType,
    reply: tokio::sync::oneshot::Sender<Result<()>>,
}

/// A [`PrecisionReloadGate`] for a loading thread, and the requests it sends
/// to the load task, which owns admission. The gate blocks its thread until
/// the reply, so the reload allocates only once it is authorized. Loads run
/// on the device thread, which keeps serving control jobs while it waits:
/// the admission may evict a model, and that model's drop runs there.
/// The gate a loader asks before it reloads a component whose
/// lowered-precision probe failed. It first synchronizes `device`: Candle frees
/// the dropped component's buffers only then, so the admission that follows
/// sees them free (and the reload does not stack on them).
fn precision_reload_gate(
    device: candle_core::Device,
) -> (
    PrecisionReloadGate,
    tokio::sync::mpsc::UnboundedReceiver<PrecisionReloadRequest>,
) {
    let (requests, receiver) = tokio::sync::mpsc::unbounded_channel();
    let gate = Box::new(move |component: &str, dtype| {
        device.synchronize().map_err(Error::from)?;
        let (reply, outcome) = tokio::sync::oneshot::channel();
        let ended = || {
            Error::ModelLoadError(format!(
                "the load ended before admitting the {component} reload at {dtype:?}"
            ))
        };
        requests
            .send(PrecisionReloadRequest {
                component: component.to_string(),
                dtype,
                reply,
            })
            .map_err(|_| ended())?;
        crate::backends::recv_serving_device(outcome).ok_or_else(ended)?
    });
    (gate, receiver)
}

/// `(retained, staged)` rows of ASR retained state (ADM-6). Automatic engine
/// rows are sized for chat (on CUDA one per 256 MiB of device memory, about
/// 320 on an H100), and every ASR retained row owns a prepared-input tensor
/// and a sequence's worth of KV. ASR keeps the portable default of 8
/// concurrent sequences, never fewer than one decode batch; operator-set rows
/// are kept as configured.
fn asr_state_rows(
    automatic_rows: bool,
    engine_retained_rows: usize,
    engine_staged_rows: usize,
) -> (usize, usize) {
    const ASR_AUTOMATIC_RETAINED_ROWS: usize = 8;
    let retained = if automatic_rows {
        engine_retained_rows.min(ASR_AUTOMATIC_RETAINED_ROWS.max(engine_staged_rows))
    } else {
        engine_retained_rows
    }
    .max(1);
    (retained, engine_staged_rows.min(retained).max(1))
}

/// Whether a load of `variant` on `backend` is reconciled against the device
/// observer: the families whose admission is a residency plan (ASR, the
/// forced aligner, the Sortformer diarizers, TTS and LFM2 chat), so an
/// observed excess means the plan under-prices the loader, as long as the
/// weights land where the observer looks (Sortformer on Metal loads into host
/// memory).
fn reconciles_device_usage(variant: ModelVariant, backend: BackendKind) -> bool {
    (asr_memory::has_plan(variant)
        || tts_memory::has_plan(variant)
        || matches!(
            variant.family(),
            crate::catalog::ModelFamily::Lfm2Chat
                | crate::catalog::ModelFamily::Qwen35Chat
                | crate::catalog::ModelFamily::FishS2Tts
        ))
        && asr_memory::weights_on_backend_device(variant, backend)
}

/// One observed-usage checkpoint of a load (ADM-8): the device observer's
/// reading and the ledger's reserved bytes, both in the load domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeviceUsageSample {
    observed: u64,
    reserved: u64,
}

impl ModelLifecycleController {
    pub(super) async fn touch_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.insert(variant, now_unix_millis());
    }

    /// Mark an explicitly loaded resident as pinned: it survives budget and
    /// memory-pressure eviction (and the idle-TTL reaper) until an explicit
    /// unload. Job auto-loads never call this.
    pub(super) async fn pin_model(&self, variant: ModelVariant) {
        let mut pinned = self.pinned_variants.lock().await;
        if pinned.insert(variant) {
            info!(model = %variant, "Pinning explicitly loaded model against eviction");
        }
    }

    pub(super) async fn unpin_model(&self, variant: ModelVariant) {
        self.pinned_variants.lock().await.remove(&variant);
    }

    /// Keep a resident loaded on request while it is idle (the idle reaper
    /// skips it) without protecting it from eviction.
    pub(super) async fn keep_model(&self, variant: ModelVariant) {
        self.kept_variants.lock().await.insert(variant);
    }

    pub(super) async fn forget_kept_model(&self, variant: ModelVariant) {
        self.kept_variants.lock().await.remove(&variant);
    }

    pub(super) async fn kept_model_variants(&self) -> HashSet<ModelVariant> {
        self.kept_variants.lock().await.clone()
    }

    pub(super) async fn pinned_model_variants(&self) -> HashSet<ModelVariant> {
        self.pinned_variants.lock().await.clone()
    }

    pub(super) async fn forget_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.remove(&variant);
    }

    async fn known_resident_variants(&self) -> Vec<ModelVariant> {
        let mut variants = self.authoritative_resident_variants();
        variants.extend(self.model_manager.resident_variants().await);
        variants.sort_by_key(|variant| variant.to_string());
        variants.dedup();
        variants
    }

    /// Releases the Metal buffers nothing references any more: izwi's
    /// padding pool, and Candle's buffer pool, which keeps every released
    /// buffer until the device next synchronizes (a failed load leaves all of
    /// its uploads there). Candle reports a failed command buffer once and
    /// stops tracking it, and releases its pool only after a clean wait, so a
    /// failed synchronization is retried once.
    async fn flush_metal_pools(&self) {
        self.clear_metal_pools_on_device();
        for attempt in 1..=2 {
            match self.core_engine.synchronize_worker_device().await {
                Ok(()) => return,
                Err(error) => warn!(
                    attempt,
                    error = %error,
                    "Synchronizing the device to release Candle's buffer pool failed"
                ),
            }
        }
    }

    /// Whether a load failure must trigger the Metal command-buffer OOM
    /// ladder (pooled-scratch flush and one retry before poisoning). Real
    /// Metal backends classify by backend kind; CPU test harnesses opt in
    /// through the injection flag so the ladder stays regression-tested
    /// without a Metal device.
    fn load_failure_is_metal_command_buffer_oom(&self, error: &Error) -> bool {
        if !is_metal_command_buffer_oom(error) {
            return false;
        }
        if self.backend_router.context().backend_kind == BackendKind::Metal {
            return true;
        }
        #[cfg(test)]
        {
            if self.load_test_metal_oom_ladder.load(std::sync::atomic::Ordering::Acquire) {
                return true;
            }
        }
        false
    }

    async fn run_load_transaction_locked(
        &self,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        generation: u64,
    ) -> Result<()> {
        let outcome = self
            .run_load_attempt_locked(variant, max_loaded_models, generation)
            .await;
        let Err(error) = outcome else {
            return Ok(());
        };
        if !self.load_failure_is_metal_command_buffer_oom(&error) {
            return Err(error);
        }
        if self.coordinator.resource_authority().poison_reason().is_some() {
            // The failed attempt could not roll back cleanly; retrying over
            // dirty residency state would compound the damage. Fail closed.
            return Err(error);
        }
        self.flush_metal_pools().await;

        // A retry only makes sense if the flush actually freed enough pooled
        // scratch to fit the model's load peak. Re-probe live headroom after
        // the flush; when it is still clearly below the model's static load
        // peak, a second full load is guaranteed to hit the same command-buffer
        // OOM and would burn another GGUF read + F16 materialization + fence
        // (tens of seconds) before poisoning. Skip straight to the poison.
        let context = self.backend_router.context();
        let backend = context.backend_kind;
        let model_path = self.model_manager.downloaded_model_path(variant);
        let required_bytes = retry_load_peak_bytes(
            variant,
            backend,
            &self.config.performance,
            model_path
                .as_deref()
                .map(|path| (path, &context.device, &self.config.granite_speech)),
        );
        self.coordinator
            .resource_authority()
            .refresh_physical_capacity_after_release();
        let retry_futile = match self
            .coordinator
            .resource_authority()
            .live_preload_headroom_bytes(backend)
        {
            Ok(Some(live_headroom)) => live_headroom < required_bytes,
            // No trustworthy live sample: preserve the original single-retry
            // behavior rather than treating the unknown as unlimited.
            Ok(None) | Err(_) => false,
        };
        if retry_futile {
            self.coordinator.resource_authority().poison(format!(
                "Metal command-buffer OOM while loading {variant}: live device headroom after a pooled-scratch flush is still below the model's load peak (load_peak_bytes={required_bytes}); not retrying a doomed load: {error}"
            ));
            self.drain_unpinned_residents_for_recovery().await;
            return Err(error);
        }

        info!(
            model = %variant,
            %error,
            "Metal command-buffer OOM while loading; flushed pooled Metal scratch and retrying the load once"
        );
        let retry = self
            .run_load_attempt_locked(variant, max_loaded_models, generation)
            .await;
        if let Err(retry_error) = &retry {
            if self.load_failure_is_metal_command_buffer_oom(retry_error) {
                self.coordinator.resource_authority().poison(format!(
                    "Metal command-buffer OOM while loading {variant} persisted after a pooled-scratch flush and one retry: {retry_error}"
                ));
                // The authority now rejects every new reservation. Evicting
                // unpinned idle residents lets the device drain so the
                // poison clears without operator action; pinned residents
                // keep their protection and hold the poison until unloaded.
                self.drain_unpinned_residents_for_recovery().await;
            }
        }
        retry
    }

    async fn run_load_attempt_locked(
        &self,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        generation: u64,
    ) -> Result<()> {
        if self.resident_phase(variant)
            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
        {
            return Ok(());
        }
        #[cfg(test)]
        self.maybe_panic_during_load();
        #[cfg(test)]
        self.maybe_fail_load_with_metal_oom()?;

        let load_started = Instant::now();
        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;
        let artifacts_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let admission_started = Instant::now();

        // Price the plan, and refuse one that can never fit, before evicting
        // anything for it.
        let mut resource_plan = self.model_resource_plan(variant, &acquired.model_path)?;
        if let Some(floor) = self.cuda_state_floor(variant, &acquired.model_path)? {
            let device_capacity = self
                .coordinator
                .resource_authority()
                .snapshot()
                .physical
                .capacity
                .device_bytes;
            // The same reserve the context fit keeps after the weights load.
            let safety_reserve_bytes = self
                .core_engine
                .config()
                .cuda_context_reserve_bytes(device_capacity);
            resource_plan.load_authorization = admit_cuda_state_floor(
                variant,
                &resource_plan,
                floor,
                safety_reserve_bytes,
                device_capacity,
            )?;
        }
        resource_plan.load_authorization = load_authorization_with_request_floor(
            variant,
            &resource_plan,
            self.backend_router.context().backend_kind,
            self.coordinator.physical_execution_admission().capacity(),
        )?;
        self.ensure_model_budget_before_load(variant, max_loaded_models)
            .await?;
        let portable_load_bytes = match self.backend_router.context().backend_kind {
            BackendKind::Cpu => resource_plan.load_authorization.host_bytes,
            BackendKind::Metal => resource_plan.load_authorization.unified_bytes,
            BackendKind::Cuda => ResourceAmount::Known(0),
        };
        if let ResourceAmount::Known(required_bytes) = portable_load_bytes {
            let backend = self.backend_router.context().backend_kind;
            let envelope_bytes = match super::asr_envelope::deferred_request_envelope(
                variant,
                backend,
                self.coordinator.physical_execution_admission().capacity(),
            )? {
                Some(envelope) => envelope.total_bytes()?,
                None => 0,
            };
            let state_floor_bytes = self
                .portable_state_floor_bytes(variant, &acquired.model_path, backend)
                .await?;
            let preload_bytes = portable_preload_bytes(
                &resource_plan,
                backend,
                required_bytes,
                envelope_bytes,
                state_floor_bytes,
            )?;
            self.ensure_portable_memory_before_load(
                variant,
                preload_bytes,
                resource_plan.host_staging_bytes,
                state_floor_bytes,
            )
            .await?;
        }
        let resource_lease = self
            .reserve_model_resources(variant, resource_plan.load_authorization)
            .await?;
        let model_instance_id = self.install_loading_slot(variant, resource_lease)?;
        if model_instance_id != crate::engine::ModelInstanceId::new(generation) {
            let error = Error::ModelLoadError(format!(
                "model {variant} loading slot does not match generation {generation}"
            ));
            if let Err(rollback_error) = self.rollback_model_locked(variant).await {
                self.mark_slot_cleanup_required(variant);
                tracing::error!(
                    model = %variant,
                    error = %rollback_error,
                    "Mismatched model generation rollback failed"
                );
            }
            return Err(error);
        }

        let publication = async {
            let admission_ms = admission_started.elapsed().as_secs_f64() * 1000.0;
            // Adapter factories are one-shot. Freeze their exact identities
            // and selectable stage graphs before model-derived state planning
            // or any physical state allocation can occur.
            let bundle_draft =
                self.draft_loaded_model_bundle(variant, model_instance_id)?;
            // This is the first operation allowed to allocate model tensors;
            // the peak host/device authorization and authoritative Loading slot
            // are both installed above.
            let weights_started = Instant::now();
            let usage_before_weights = self.sample_device_usage(variant);
            // Admission priced the selected precision. A component whose
            // load-time probe fails reloads at F32 only once this task has
            // grown the authorization for it.
            let model_path = acquired.model_path.clone();
            let (reload_gate, mut reload_requests) =
                precision_reload_gate(self.backend_router.context().device.device.clone());
            let instantiate = self.instantiate_model(acquired, reload_gate);
            tokio::pin!(instantiate);
            let instantiated = loop {
                tokio::select! {
                    instantiated = &mut instantiate => break instantiated?,
                    Some(request) = reload_requests.recv() => {
                        let admitted = self
                            .admit_precision_reload(variant, &model_path, &resource_plan, &request)
                            .await;
                        if let Ok(Some(reload)) = admitted {
                            resource_plan = reload;
                        }
                        let _ = request.reply.send(admitted.map(|_| ()));
                    }
                }
            };
            let weights_ms = weights_started.elapsed().as_secs_f64() * 1000.0;
            let preparation_started = Instant::now();
            if variant.family() == crate::catalog::ModelFamily::Qwen3Asr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Qwen3 ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::Qwen3(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Qwen3 ASR geometry"
                    )));
                };
                // The lifecycle slot is still Loading, so no request can bind
                // this model while its exact backend/dtype/width geometry is
                // being frozen into the already-selected adapter identity.
                bundle_draft.seal_qwen3_asr_audio_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::WhisperAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Whisper ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::WhisperTurbo(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Whisper ASR geometry"
                    )));
                };
                bundle_draft.seal_whisper_audio_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::VibeVoiceAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated VibeVoice ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::VibeVoice(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose VibeVoice ASR geometry"
                    )));
                };
                bundle_draft.seal_vibevoice_asr_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Granite Speech ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::GraniteSpeech(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Granite Speech ASR geometry"
                    )));
                };
                bundle_draft.seal_granite_speech_asr_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                let model = self
                    .model_registry
                    .get_loading_lfm25_audio_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated LFM2.5 Audio model {variant} is missing before ASR adapter sealing"
                        ))
                    })?;
                bundle_draft.seal_lfm25_audio_asr_preparation(&model)?;
                bundle_draft.seal_lfm25_audio_tts_preparation(&model)?;
            } else if variant.family() == crate::catalog::ModelFamily::Voxtral
                && self
                    .adapter_registry
                    .require(CapabilityKind::RealtimeAsr, variant)
                    .is_ok()
            {
                let model = self
                    .model_registry
                    .get_loading_voxtral(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Voxtral realtime model {variant} is missing before adapter sealing"
                        ))
                    })?;
                bundle_draft.seal_voxtral_realtime_preparation(model.as_ref())?;
            }
            // Metal records many tensor uploads asynchronously. Flush them
            // before publishing the model so an allocation failure is owned by
            // this load transaction rather than a later state/request fence.
            let fence_started = Instant::now();
            self.core_engine.synchronize_worker_device().await?;
            let upload_fence_ms = fence_started.elapsed().as_secs_f64() * 1000.0;
            self.publish_loaded_model(instantiated).await?;
            let backend = self.backend_router.context().backend_kind;
            // Truth reconciliation (ADM-8): the fence has executed every
            // upload and dropped the staging Candle held, so the observed
            // growth is what the weights occupy. A load that outgrew its
            // authorized peak fails here, before any state is fitted beside
            // a model the ledger under-counts.
            let observed_weights = match usage_before_weights {
                Some(before) => match self.sample_device_usage(variant) {
                    Some(after) => {
                        let own_bytes = match load_domain_bytes(
                            resource_plan.load_authorization,
                            backend,
                        ) {
                            ResourceAmount::Known(bytes) => bytes,
                            ResourceAmount::Unknown => u64::MAX,
                        };
                        let observed =
                            reconcile_observed_growth(variant, "weight load", before, after, own_bytes)?;
                        info!(
                            model = %variant,
                            observed_bytes = observed,
                            authorized_bytes = own_bytes,
                            "Observed device growth of the weight load"
                        );
                        self.model_registry.publish_weight_residency(
                            variant,
                            crate::models::registry::WeightResidency {
                                priced_bytes: own_bytes,
                                observed_bytes: observed,
                            },
                        );
                        Some((backend, observed))
                    }
                    None => None,
                },
                None => None,
            };
            // A load without an observation leaves its weight under-pricing
            // untracked; its live state fits keep only the configured reserve
            // so those bytes, already charged by the live sample, are not
            // reserved a second time (R-1).
            let state_fit_reserve = StateFitReserve::for_observed_load(observed_weights.is_some());
            // Model tensors are now visible in the backend provider's used
            // memory. Reconcile the lease before reserving any retained or
            // invocation state so live headroom does not charge the model once
            // through the provider and again as unmaterialized ledger work.
            self.finalize_slot_materialization_with_pending(
                variant,
                resource_plan.resident_authorization,
                resource_plan.deferred_resident_authorization,
                observed_weights,
            )?;
            // The request envelope stays out of reach of every state fit below
            // (ADM-4): the fits spend headroom less pending claims, so held
            // here, one full chunk per execution slot is still free when the
            // first request runs beside the fitted state.
            if reconciles_device_usage(variant, backend) {
                // The cached live sample can predate the fence that released
                // the load's staging; the state fits read a fresh one.
                self.coordinator
                    .resource_authority()
                    .refresh_physical_capacity_after_release();
            }
            let request_envelope_hold = self.hold_request_envelope(variant, backend).await?;
            let usage_before_state = self.sample_device_usage(variant);
            // Resolve physical state from the exact loaded chat implementation
            // without wrapping or replacing the already-selected adapter.
            let state_started = Instant::now();
            let mut state_publications = HashMap::new();
            self.allocate_chat_state(
                variant,
                model_instance_id,
                backend,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_granite_speech_state(
                variant,
                model_instance_id,
                backend,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_asr_state(
                variant,
                model_instance_id,
                backend,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_realtime_asr_state(
                variant,
                model_instance_id,
                backend,
                state_fit_reserve,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_diarization_state(
                variant,
                model_instance_id,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_lfm25_audio_state(
                variant,
                model_instance_id,
                backend,
                state_fit_reserve,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_qwen_tts_state(
                variant,
                model_instance_id,
                backend,
                state_fit_reserve,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_vibevoice_tts_state(
                variant,
                model_instance_id,
                backend,
                state_fit_reserve,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_fish_s2_tts_state(
                variant,
                model_instance_id,
                backend,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_voxtral_tts_state(
                variant,
                model_instance_id,
                backend,
                state_fit_reserve,
                &bundle_draft,
                &mut state_publications,
            )
            .await?;
            self.allocate_kokoro_tts_state(
                variant,
            )
            .await?;
            if let Some(before) = usage_before_state {
                // State arenas are zero-filled by queued blits; synchronize so
                // the observation covers what they really occupy.
                self.core_engine.synchronize_worker_device().await?;
                if let Some(after) = self.sample_device_usage(variant) {
                    let observed =
                        reconcile_observed_growth(variant, "state allocation", before, after, 0)?;
                    info!(
                        model = %variant,
                        observed_bytes = observed,
                        authorized_bytes = after.reserved.saturating_sub(before.reserved),
                        "Observed device growth of the state allocation"
                    );
                }
            }
            drop(request_envelope_hold);
            let state_allocation_ms = state_started.elapsed().as_secs_f64() * 1000.0;
            let binding_started = Instant::now();
            self.bind_loaded_model_bundle_draft(
                bundle_draft,
                variant,
                model_instance_id,
                state_publications,
            )?;
            // Install the legacy manager projection before the authoritative
            // commit. Inference pins consult the slot, so no caller can observe
            // Ready while this await is still in progress.
            self.model_manager.mark_loaded(variant).await;
            self.mark_slot_ready_for_instance(variant, model_instance_id)?;
            if uses_asr_model_registry(variant) {
                // This is the registry's external publication barrier. Every
                // adapter seal, backend fence, state plan, and bundle binding
                // has committed and the authoritative slot is already Ready.
                // The safe transient is Ready-but-hidden; an ASR handle must
                // never be visible while its slot is still Loading.
                self.model_registry.publish_asr_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::Voxtral {
                self.model_registry.publish_voxtral_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                self.model_registry
                    .publish_lfm25_audio_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::VibeVoiceTts {
                self.model_registry
                    .publish_vibevoice_tts_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::FishS2Tts {
                self.model_registry
                    .publish_fish_s2_tts_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::KokoroTts {
                self.model_registry.publish_kokoro_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::VoxtralTts {
                self.model_registry
                    .publish_voxtral_tts_ready(variant)
                    .await?;
            }
            self.touch_model_usage(variant).await;
            info!(
                model = %variant,
                generation,
                artifacts_ms,
                admission_ms,
                weights_ms,
                upload_fence_ms,
                state_allocation_ms,
                binding_publication_ms = binding_started.elapsed().as_secs_f64() * 1000.0,
                preparation_ms = preparation_started.elapsed().as_secs_f64() * 1000.0,
                ready_ms = load_started.elapsed().as_secs_f64() * 1000.0,
                "Model load reached physical Ready"
            );
            Ok(())
        }
        .await;

        if let Err(error) = publication {
            let metal_oom = self.load_failure_is_metal_command_buffer_oom(&error);
            if let Err(rollback_error) = self.rollback_model_locked(variant).await {
                self.mark_slot_cleanup_required(variant);
                tracing::error!(
                    model = %variant,
                    error = %rollback_error,
                    "Model load rollback failed"
                );
                if metal_oom {
                    // Dirty residency state after a device-fatal OOM: new
                    // physical work must fail closed until recreation, so a
                    // later attempt cannot be blamed for earlier queued work.
                    self.coordinator.resource_authority().poison(format!(
                        "Metal command-buffer OOM while loading {variant} and rollback failed: {error}"
                    ));
                }
            }
            return Err(error);
        }

        Ok(())
    }

    pub(crate) fn spawn_load_transaction(
        self: &Arc<Self>,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        leader: crate::runtime::lifecycle::controller::LoadLeader,
    ) -> tokio::task::JoinHandle<()> {
        let controller = self.clone();
        tokio::spawn(async move {
            // Publication of both the Ready slot and the shared terminal
            // outcome is one mutation-gated transaction. Explicit unload can
            // neither erase a successful load before waiters are notified nor
            // observe a half-published failure rollback.
            let _mutation_guard = controller.mutation_gate.lock().await;
            if !controller.is_current_load_generation_locked(variant, leader.generation) {
                return;
            }
            let _coordinator_load = match controller
                .coordinator
                .begin_model_load(format!("model-load:{variant}"))
            {
                Ok(load) => load,
                Err(error) => {
                    controller.finish_load_locked(
                        variant,
                        leader.generation,
                        &leader.completion,
                        SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                    );
                    return;
                }
            };
            let outcome = match AssertUnwindSafe(controller.run_load_transaction_locked(
                variant,
                max_loaded_models,
                leader.generation,
            ))
            .catch_unwind()
            .await
            {
                Ok(Ok(())) => SharedLoadOutcome::Ready,
                Ok(Err(error)) => SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                Err(payload) => {
                    let message = if let Some(message) = payload.downcast_ref::<&str>() {
                        (*message).to_string()
                    } else if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "unknown model load panic".to_string()
                    };
                    if let Err(error) = controller.rollback_model_after_panic_locked(variant).await
                    {
                        tracing::error!(model = %variant, %error, "Panicked model load rollback failed");
                    }
                    SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad(format!(
                        "model load task panicked: {message}"
                    )))
                }
            };
            controller.finish_load_locked(variant, leader.generation, &leader.completion, outcome);
        })
    }
}

impl RuntimeService {
    pub(crate) async fn load_model_for_inference(
        &self,
        variant: ModelVariant,
    ) -> Result<crate::model::ModelResidencyLease> {
        loop {
            if let Some(lease) = self.model_lifecycle.try_acquire_ready_lease(variant) {
                self.model_lifecycle.touch_model_usage(variant).await;
                return Ok(lease);
            }

            let (waiter, leader) = self.model_lifecycle.join_or_start_load(variant);
            if let Some(leader) = leader {
                let _load_task = self.model_lifecycle.spawn_load_transaction(
                    variant,
                    self.max_loaded_models,
                    leader,
                );
            }
            waiter.wait().await?;
        }
    }

    /// Load a model without retaining an inference pin. The explicit load
    /// entry (admin API, preload list) marks the resident as pinned so it
    /// survives budget and memory-pressure eviction until explicitly
    /// unloaded; job auto-loads use `load_model_for_inference` directly and
    /// stay transient.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        drop(self.load_model_for_inference(variant).await?);
        self.model_lifecycle.pin_model(variant).await;
        Ok(())
    }

    /// Load a model on request without pinning it: it stays loaded while
    /// idle, but a load that needs its room evicts it like any idle model.
    /// The app loads this way, so models loaded on different pages make
    /// room for each other.
    pub async fn load_model_unpinned(&self, variant: ModelVariant) -> Result<()> {
        drop(self.load_model_for_inference(variant).await?);
        self.model_lifecycle.keep_model(variant).await;
        Ok(())
    }

    /// Pin a resident model against eviction until it is unloaded ("Keep
    /// loaded"). False when the model is not loaded.
    pub async fn pin_resident_model(&self, variant: ModelVariant) -> bool {
        if !self.model_manager.is_ready(variant).await {
            return false;
        }
        self.model_lifecycle.pin_model(variant).await;
        true
    }

    /// Lift a pin: the model stays loaded while idle, and is evicted when a
    /// load needs its room. False when the model is not loaded.
    pub async fn unpin_resident_model(&self, variant: ModelVariant) -> bool {
        if !self.model_manager.is_ready(variant).await {
            return false;
        }
        self.model_lifecycle.unpin_model(variant).await;
        self.model_lifecycle.keep_model(variant).await;
        true
    }

    /// Memory a model needs to load, from the catalog: what the app shows
    /// before the model is on disk. A load prices the checkpoint itself.
    pub fn estimated_model_load_bytes(&self, variant: ModelVariant) -> u64 {
        checkpoint::model_memory_estimate(variant).load_peak_bytes
    }

    /// The memory loads are planned against on this backend: the operator's
    /// budget, else the device's capacity. `None` when it is unknown.
    pub fn model_capacity_bytes(&self) -> Option<u64> {
        self.coordinator
            .resource_authority()
            .planning_ceiling_bytes(self.backend_context().backend_kind)
    }

    /// Variants explicitly loaded (pinned) in this server session.
    pub async fn pinned_model_variants(&self) -> std::collections::HashSet<ModelVariant> {
        self.model_lifecycle.pinned_model_variants().await
    }

    /// Why the latest load of `variant` may be slow, if admission noted
    /// anything (a Metal load larger than the host memory macOS reports
    /// available). Taking it clears it.
    pub fn take_load_warning(&self, variant: ModelVariant) -> Option<String> {
        self.model_lifecycle.take_load_warning(variant)
    }

    #[cfg(test)]
    async fn ensure_model_budget_before_load(&self, requested_variant: ModelVariant) -> Result<()> {
        let _mutation_guard = self.model_lifecycle.mutation_gate.lock().await;
        self.model_lifecycle
            .ensure_model_budget_before_load(requested_variant, self.max_loaded_models)
            .await
    }
}

/// Residency leases a long-lived session (a realtime voice connection) holds
/// on the models it uses. Each model loads like an inference auto-load and
/// is never pinned: it stays out of eviction only while this set, or a clone
/// of it, is alive, and becomes evictable again when the session ends.
#[derive(Debug, Clone, Default)]
pub struct SessionModelLeases {
    leases: Arc<std::sync::Mutex<HashMap<ModelVariant, crate::model::ModelResidencyLease>>>,
}

impl SessionModelLeases {
    /// Loads `variant` if needed and keeps a lease on it for the session.
    pub async fn hold(&self, runtime: &RuntimeService, variant: ModelVariant) -> Result<()> {
        let lease = runtime.load_model_for_inference(variant).await?;
        self.leases
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(variant, lease);
        Ok(())
    }
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod footprint_tests;

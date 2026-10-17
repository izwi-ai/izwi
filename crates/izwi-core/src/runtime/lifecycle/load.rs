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
use crate::engine::resources::capacity_shortfall_reason;
use crate::engine::{
    AdapterInstanceId, CacheMode, ModelInstanceId, ReservationClass, ReservationOwner,
    ResourceAmount, ResourceLease, ResourceVector, StageDescriptor, StateFitReserve,
};
use crate::error::{Error, Result};
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

#[path = "asr_memory.rs"]
mod asr_memory;
#[path = "qwen36moe_memory.rs"]
mod qwen36moe_memory;
#[path = "qwen38_memory.rs"]
mod qwen38_memory;
#[path = "tts_memory.rs"]
mod tts_memory;
mod state;

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
    LegacyCache,
}

fn loaded_asr_state_publication_route(variant: ModelVariant) -> LoadedAsrStatePublicationRoute {
    match variant.family() {
        crate::catalog::ModelFamily::Qwen3Asr => LoadedAsrStatePublicationRoute::Qwen3,
        crate::catalog::ModelFamily::VibeVoiceAsr => LoadedAsrStatePublicationRoute::VibeVoice,
        crate::catalog::ModelFamily::WhisperAsr => LoadedAsrStatePublicationRoute::Whisper,
        crate::catalog::ModelFamily::ParakeetAsr => LoadedAsrStatePublicationRoute::Parakeet,
        crate::catalog::ModelFamily::NemotronAsr => LoadedAsrStatePublicationRoute::NemotronOffline,
        _ => LoadedAsrStatePublicationRoute::LegacyCache,
    }
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

/// Recurrent state the CUDA resident fit must hold right after the weights:
/// its fewest rows (the staged width when retained rows are automatic, the
/// operator's rows otherwise) plus the committed prefix snapshots reserved
/// beside them, each priced at the model's per-row bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaStateFloor {
    retained_rows: usize,
    staged_rows: usize,
    /// Prefix snapshot budget when reuse engages for the model (an upper
    /// bound: a contract that keeps its state private reserves none).
    snapshot_rows: usize,
    per_sequence_bytes: u64,
    automatic_rows: bool,
}

impl CudaStateFloor {
    fn bytes(self) -> Result<u64> {
        self.retained_rows
            .checked_add(self.staged_rows)
            .and_then(|rows| rows.checked_add(self.snapshot_rows))
            .and_then(|rows| u64::try_from(rows).ok())
            .and_then(|rows| rows.checked_mul(self.per_sequence_bytes))
            .ok_or_else(|| Error::ModelLoadError("CUDA recurrent state floor overflow".into()))
    }
}

/// Price the recurrent state floor before any weight streams. A plan whose
/// weights, floor state and safety reserve exceed the device can never pass
/// the resident fit, so it fails here with the fit's guidance. Otherwise the
/// load authorization also covers the floor, so a resident model in the way
/// is evicted, or the load refused, before the weights are read.
fn admit_cuda_state_floor(
    variant: ModelVariant,
    plan: &ModelResourcePlan,
    floor: CudaStateFloor,
    safety_reserve_bytes: u64,
    device_capacity: ResourceAmount,
) -> Result<ResourceVector> {
    let (ResourceAmount::Known(weights_bytes), ResourceAmount::Known(load_peak_bytes)) = (
        plan.resident_authorization.device_bytes,
        plan.load_authorization.device_bytes,
    ) else {
        return Err(Error::ModelLoadError(format!(
            "{variant} CUDA weight plan must be known before state admission"
        )));
    };
    let minimum_state_bytes = floor.bytes()?;
    let required_bytes = weights_bytes
        .checked_add(minimum_state_bytes)
        .and_then(|bytes| bytes.checked_add(safety_reserve_bytes))
        .ok_or_else(|| Error::ModelLoadError("CUDA state floor overflow".into()))?;
    if let ResourceAmount::Known(device_bytes) = device_capacity {
        if required_bytes > device_bytes {
            let guidance = if floor.automatic_rows {
                crate::engine::CUDA_AUTOMATIC_ROWS_GUIDANCE
            } else {
                crate::engine::CUDA_EXPLICIT_ROWS_GUIDANCE
            };
            return Err(Error::ModelLoadError(format!(
                "CUDA managed context cannot fit the model-authored minimum before loading {variant}: retained_sequences={}, staged_transactions={}, prefix_snapshots={}, per_sequence_state_bytes={}, minimum_state_bytes={minimum_state_bytes}, weights_bytes={weights_bytes}, safety_reserve={safety_reserve_bytes}, device_capacity={device_bytes}; {guidance}",
                floor.retained_rows, floor.staged_rows, floor.snapshot_rows, floor.per_sequence_bytes,
            )));
        }
    }
    Ok(ResourceVector {
        device_bytes: ResourceAmount::Known(load_peak_bytes.max(required_bytes)),
        ..plan.load_authorization
    })
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMemoryEstimate {
    /// Maximum model-owned memory while tensors are being instantiated.
    load_peak_bytes: u64,
    /// Long-lived model-owned memory after publication.
    resident_bytes: u64,
}

const QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = 1024 * 1024 * 1024;
const QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

/// Static estimate of the pinned Qwen3.8-27B checkpoint on `backend`, priced
/// by the Qwen3.8 residency policy over the checkpoint's tensor list (the
/// same pricing load admission applies to a real checkpoint). The CUDA arm
/// is not the CUDA admission path: without the device it cannot run the
/// native block-FP8 gates, so it prices Q8_0 (an upper bound).
fn qwen38_representation_memory_estimate(
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
) -> ModelMemoryEstimate {
    qwen38_memory::static_estimate(backend, performance)
        .expect("the pinned Qwen3.8 inventory prices without overflow")
}

fn model_memory_estimate(variant: ModelVariant) -> ModelMemoryEstimate {
    const GIB: u64 = 1024 * 1024 * 1024;

    let inference_bytes = (variant.memory_required_gb() as f64 * GIB as f64).ceil() as u64;
    let base = match variant {
        // The 5 GiB catalog value describes total inference memory, including
        // request-scoped activations and audio workspace that the coordinator
        // reserves separately. The GGUF loader retains about 2.25 GiB of
        // quantized/dequantized tensors; 3 GiB covers model-owned load overlap,
        // tokenizer metadata, allocator alignment, and steady residency.
        ModelVariant::Lfm25Audio15BGguf => ModelMemoryEstimate {
            load_peak_bytes: 3 * GIB,
            resident_bytes: 3 * GIB,
        },
        _ => ModelMemoryEstimate {
            load_peak_bytes: inference_bytes,
            resident_bytes: inference_bytes,
        },
    };
    let memo_bytes = match variant {
        ModelVariant::Nemotron35AsrStreaming06B => {
            crate::models::architectures::nemotron::asr::NEMOTRON_MODEL_MEMO_MAX_BYTES
        }
        ModelVariant::Kokoro82M => {
            crate::models::architectures::kokoro::KOKORO_MODEL_MEMO_MAX_BYTES
        }
        _ => 0,
    };
    ModelMemoryEstimate {
        load_peak_bytes: base
            .load_peak_bytes
            .checked_add(memo_bytes)
            .expect("catalog model load estimate overflowed"),
        resident_bytes: base
            .resident_bytes
            .checked_add(memo_bytes)
            .expect("catalog model residency estimate overflowed"),
    }
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

/// Fold the request floor into an ASR load's authorization before anything is
/// evicted for it (ADM-10): the weights must leave room for one request
/// envelope per execution slot, or no request could run beside the loaded
/// model. The minimal state itself (one page per row) is small beside it.
fn admit_asr_request_floor(
    plan: &ModelResourcePlan,
    backend: BackendKind,
    floor_bytes: u64,
) -> Result<ResourceVector> {
    let (ResourceAmount::Known(resident), ResourceAmount::Known(peak)) = (
        load_domain_bytes(plan.resident_authorization, backend),
        load_domain_bytes(plan.load_authorization, backend),
    ) else {
        return Ok(plan.load_authorization);
    };
    let required = resident
        .checked_add(floor_bytes)
        .ok_or_else(|| Error::ModelLoadError("ASR request floor overflow".into()))?
        .max(peak);
    let mut authorization = plan.load_authorization;
    match backend {
        BackendKind::Cpu => authorization.host_bytes = ResourceAmount::Known(required),
        BackendKind::Metal => authorization.unified_bytes = ResourceAmount::Known(required),
        BackendKind::Cuda => authorization.device_bytes = ResourceAmount::Known(required),
    }
    Ok(authorization)
}

/// Whether a load of `variant` is reconciled against the device observer:
/// the families whose admission is a residency plan (ASR and TTS), so an
/// observed excess means the plan under-prices the loader.
fn reconciles_device_usage(variant: ModelVariant) -> bool {
    asr_memory::is_asr_variant(variant)
        || tts_memory::has_plan(variant)
        || variant.family() == crate::catalog::ModelFamily::Lfm2Chat
}

/// A request envelope in `backend`'s ledger domains: host and accelerator
/// share one memory on CPU and Metal; CUDA splits them.
fn request_envelope_resources(
    backend: BackendKind,
    envelope: super::asr_envelope::AsrRequestEnvelope,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(envelope.total_bytes()?);
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(envelope.total_bytes()?);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(envelope.host_bytes);
            resources.device_bytes = ResourceAmount::Known(envelope.accelerator_bytes);
        }
    }
    Ok(resources)
}

/// The short refusal of a load whose per-slot request envelope does not
/// fit beside its weights: what the envelope needs in the load's domain and
/// what planning has left (a 503; the terms are logged by the caller).
fn request_envelope_shortfall(
    variant: ModelVariant,
    backend: BackendKind,
    envelope: super::asr_envelope::AsrRequestEnvelope,
    available_bytes: u64,
) -> Result<Error> {
    let need = match backend {
        BackendKind::Cuda => envelope.accelerator_bytes,
        BackendKind::Cpu | BackendKind::Metal => envelope.total_bytes()?,
    };
    Ok(Error::Overloaded(capacity_shortfall_reason(
        format_args!("not enough memory to run {variant} requests"),
        need,
        available_bytes,
    )))
}

/// One observed-usage checkpoint of a load (ADM-8): the device observer's
/// reading and the ledger's reserved bytes, both in the load domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeviceUsageSample {
    observed: u64,
    reserved: u64,
}

/// Fixed part of the observed-materialization tolerance. Candle's Metal pool
/// keeps a buffer per size class it has served even after a synchronization
/// drops released ones, and the plan leaves out derived tensors under about
/// 1 MiB per family; a CUDA context lazily allocates module and stream
/// memory on its first kernels. A few hundred MiB covers these.
const MATERIALIZATION_TOLERANCE_FLOOR_BYTES: u64 = 256 * 1024 * 1024;

/// How far observed growth may exceed what the ledger authorized before the
/// load fails: an eighth of the authorization (the plans' own conservative
/// approximations, such as the Metal pool serving a smaller tensor from a
/// larger freed buffer, are well inside that), and never less than
/// [`MATERIALIZATION_TOLERANCE_FLOOR_BYTES`]. An excess beyond it means the
/// plan prices a different representation than the loader allocates, which
/// is the defect class the reconciliation exists to catch.
fn materialization_tolerance_bytes(authorized_bytes: u64) -> u64 {
    (authorized_bytes / 8).max(MATERIALIZATION_TOLERANCE_FLOOR_BYTES)
}

/// The device growth observed between two checkpoints of a load, failing the
/// load when it exceeds what the ledger authorized over the same interval
/// plus [`materialization_tolerance_bytes`]. `own_authorized_bytes` is what
/// this load held before `before` was sampled (its weight peak); every lease
/// reserved in between, the load's own state leases and any concurrent
/// request workspace alike, is authorized through the ledger growth, so a
/// concurrent allocation counts on both sides.
fn reconcile_observed_growth(
    variant: ModelVariant,
    stage: &str,
    before: DeviceUsageSample,
    after: DeviceUsageSample,
    own_authorized_bytes: u64,
) -> Result<u64> {
    let observed_growth = after.observed.saturating_sub(before.observed);
    let authorized_bytes =
        own_authorized_bytes.saturating_add(after.reserved.saturating_sub(before.reserved));
    let tolerance_bytes = materialization_tolerance_bytes(authorized_bytes);
    if observed_growth > authorized_bytes.saturating_add(tolerance_bytes) {
        return Err(Error::ModelLoadError(format!(
            "{variant} {stage} allocated more device memory than its admission authorized: observed_bytes={observed_growth}, authorized_bytes={authorized_bytes}, tolerance_bytes={tolerance_bytes}; the residency plan under-prices this load"
        )));
    }
    Ok(observed_growth)
}

/// The residency plan first admission prices for a checkpoint-planned
/// family (Fish, ASR, TTS, LFM2.5 chat), or `None` for one without a plan. The Metal OOM
/// retry judges futility by the same plan.
fn checkpoint_resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &crate::backends::DeviceProfile,
    granite: &crate::config::GraniteSpeechOptions,
) -> Result<Option<ModelResourcePlan>> {
    if variant == ModelVariant::FishAudioS2Pro {
        let memory = crate::models::architectures::fish_s2::weights::fish_s2_model_memory(
            model_path, device,
        )?;
        return fish_s2_resource_plan(BackendKind::from(device.kind), memory).map(Some);
    }
    // ASR, TTS and LFM2.5 chat price what their loader allocates on every
    // backend; the catalog figure stays a display value for them.
    if let Some(plan) = asr_memory::resource_plan(variant, model_path, device, granite)? {
        return Ok(Some(plan));
    }
    if variant.family() == crate::catalog::ModelFamily::Lfm2Chat {
        return asr_memory::lfm2_chat_resource_plan(
            variant,
            model_path,
            BackendKind::from(device.kind),
        );
    }
    tts_memory::resource_plan(variant, model_path, device)
}

/// Static load peak that decides whether a Metal OOM retry can fit. The
/// catalog's `memory_required_gb` is one coarse figure for every backend (80
/// GiB for Qwen3.8, 140 GiB for Qwen3.6, sized for the CPU F32 expansion);
/// these checkpoints use the per-backend representation estimate that load
/// admission prices instead. A planned checkpoint (`checkpoint`: its path and
/// the load device; Fish, ASR and TTS) is judged by the plan first admission
/// priced ([`checkpoint_resource_plan`], with the engine's Granite overrides).
fn retry_load_peak_bytes(
    variant: ModelVariant,
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
    checkpoint: Option<(
        &Path,
        &crate::backends::DeviceProfile,
        &crate::config::GraniteSpeechOptions,
    )>,
) -> u64 {
    let catalog = || model_memory_estimate(variant).load_peak_bytes;
    if let Some((model_path, device, granite)) = checkpoint {
        if let Ok(Some(plan)) = checkpoint_resource_plan(variant, model_path, device, granite) {
            if let ResourceAmount::Known(bytes) =
                load_domain_bytes(plan.load_authorization, backend)
            {
                return bytes;
            }
        }
    }
    match variant {
        ModelVariant::Qwen3827BFp8
            if !crate::models::architectures::qwen38::native::synthetic_geometry_enabled() =>
        {
            qwen38_representation_memory_estimate(backend, performance).load_peak_bytes
        }
        ModelVariant::Qwen36Moe35BA3BFp8
            if !crate::models::architectures::qwen36moe::native::synthetic_geometry_enabled() =>
        {
            let peak = qwen36moe_memory::resource_plan(backend, performance)
                .map(|plan| load_domain_bytes(plan.load_authorization, backend));
            match peak {
                Ok(ResourceAmount::Known(bytes)) => bytes,
                _ => catalog(),
            }
        }
        _ => catalog(),
    }
}

fn collect_checkpoint_files(path: &Path, depth: usize, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    if depth == 0 || !path.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if entry.file_type()?.is_dir() {
            collect_checkpoint_files(&child, depth - 1, files)?;
        } else {
            files.push(child);
        }
    }
    Ok(())
}

/// How a checkpoint stores one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoredDType {
    /// A safetensors or PyTorch tensor, which Candle loads as a dense tensor.
    Dense(candle_core::DType),
    /// A GGUF tensor. Candle loads every GGUF tensor, F32 ones included, as a
    /// `QTensor`.
    Ggml(candle_core::quantized::GgmlDType),
    /// A safetensors dtype Candle cannot load: only its stored bytes are known.
    Unsupported,
}

/// One tensor as its checkpoint header records it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CensusTensor {
    name: String,
    dtype: StoredDType,
    shape: Vec<usize>,
    /// Payload bytes in the file.
    stored_bytes: u64,
}

impl CensusTensor {
    fn elements(&self) -> u64 {
        u64::try_from(
            self.shape
                .iter()
                .try_fold(1_usize, |acc, &dim| acc.checked_mul(dim))
                .unwrap_or(usize::MAX),
        )
        .unwrap_or(u64::MAX)
    }
}

/// The header census of one checkpoint file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CensusFile {
    path: PathBuf,
    tensors: Vec<CensusTensor>,
}

/// Header census of the checkpoint files under a model path: the tensor list
/// of every readable checkpoint, without materializing any payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CheckpointCensus {
    files: Vec<CensusFile>,
    /// Size of the largest container whose tensors cannot be listed (a `.nemo`
    /// archive, or a PTH file Candle cannot parse).
    container_fallback_bytes: u64,
}

impl CheckpointCensus {
    fn tensors(&self) -> impl Iterator<Item = &CensusTensor> {
        self.files.iter().flat_map(|file| file.tensors.iter())
    }

    fn inventory(&self) -> Option<TensorStorageInventory> {
        if self.files.is_empty() {
            // The compressed artifact size is a conservative lower bound until
            // the extracted checkpoint header exists.
            let bytes = self.container_fallback_bytes;
            return (bytes > 0).then_some(TensorStorageInventory {
                total_bytes: bytes,
                largest_tensor_bytes: bytes,
                largest_tensor_elements: 0,
                tensor_count: 1,
            });
        }
        Some(
            self.tensors()
                .fold(TensorStorageInventory::default(), |inv, tensor| {
                    inv.push(tensor.stored_bytes, tensor.elements())
                }),
        )
    }
}

/// The header census of one checkpoint file, or `None` for a file that is not
/// a checkpoint or a PTH file Candle cannot parse.
fn census_checkpoint_file(file: &Path) -> Result<Option<CensusFile>> {
    let tensors = match file.extension().and_then(|value| value.to_str()) {
        Some("gguf") => crate::models::shared::weights::gguf::GgufLoader::from_path(file)?
            .tensor_census()?
            .into_iter()
            .map(|tensor| CensusTensor {
                name: tensor.name,
                dtype: StoredDType::Ggml(tensor.dtype),
                shape: tensor.shape,
                stored_bytes: tensor.bytes,
            })
            .collect(),
        Some("safetensors") => {
            // SAFETY: Candle owns the read-only mapping for the lifetime of
            // the parsed tensor views read below.
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(file) }?;
            tensors
                .tensors()
                .into_iter()
                .map(|(name, view)| {
                    let stored_bytes = u64::try_from(view.data().len()).map_err(|_| {
                        Error::ModelLoadError("safetensors tensor size exceeds u64".into())
                    })?;
                    Ok(CensusTensor {
                        name,
                        dtype: candle_core::DType::try_from(view.dtype())
                            .map_or(StoredDType::Unsupported, StoredDType::Dense),
                        shape: view.shape().to_vec(),
                        stored_bytes,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
        Some("pth" | "ckpt") => {
            let Ok(infos) = candle_core::pickle::read_pth_tensor_info(file, false, None) else {
                return Ok(None);
            };
            infos
                .into_iter()
                .map(|info| {
                    let elements =
                        u64::try_from(info.layout.shape().elem_count()).unwrap_or(u64::MAX);
                    CensusTensor {
                        name: info.name,
                        dtype: StoredDType::Dense(info.dtype),
                        shape: info.layout.shape().dims().to_vec(),
                        stored_bytes: elements.saturating_mul(info.dtype.size_in_bytes() as u64),
                    }
                })
                .collect()
        }
        _ => return Ok(None),
    };
    Ok(Some(CensusFile {
        path: file.to_path_buf(),
        tensors,
    }))
}

fn checkpoint_census(path: &Path) -> Result<CheckpointCensus> {
    let mut files = Vec::new();
    collect_checkpoint_files(path, 3, &mut files)?;
    let mut census = CheckpointCensus::default();
    for file in files {
        let extension = file.extension().and_then(|value| value.to_str());
        let container = match extension {
            // NeMo is a tar container (gzip-compressed for some checkpoints).
            Some("nemo") => true,
            _ => match census_checkpoint_file(&file)? {
                Some(parsed) => {
                    census.files.push(parsed);
                    false
                }
                None => matches!(extension, Some("pth" | "ckpt")),
            },
        };
        if container {
            census.container_fallback_bytes =
                census.container_fallback_bytes.max(file.metadata()?.len());
        }
    }
    Ok(census)
}

fn checkpoint_tensor_inventory(path: &Path) -> Result<Option<TensorStorageInventory>> {
    Ok(checkpoint_census(path)?.inventory())
}

/// Per-tensor instantiation slack: device buffer page rounding, storage
/// headers, and allocator slack for every materialized weight tensor. Only
/// becomes material at MoE scale (thousands of same-sized expert tensors);
/// negligible for the few hundred tensors of a dense checkpoint.
const PER_TENSOR_INSTANTIATION_SLACK_BYTES: u64 = 32 * 1024;
/// Worst-case load-time destination expansion of one tensor: a quantized
/// source dequantizes to F32 (4 bytes/element) and the source bytes stay
/// alive while the destination is built.
const SINGLE_TENSOR_F32_DESTINATION_BYTES_PER_ELEMENT: u64 = 4;

fn estimate_from_tensor_inventory(
    catalog: ModelMemoryEstimate,
    inventory: Option<TensorStorageInventory>,
) -> Result<ModelMemoryEstimate> {
    let Some(inventory) = inventory else {
        return Ok(catalog);
    };
    if inventory.total_bytes == 0 {
        return Ok(catalog);
    }
    // Size-class bound for one tensor's working copy (the historic term).
    let single_tensor_working_copy = inventory
        .largest_tensor_bytes
        .checked_next_power_of_two()
        .unwrap_or(inventory.largest_tensor_bytes);
    // Conversion bound for a dequantizing loader: source bytes plus the F32
    // destination of the largest tensor. Dominates the size-class bound only
    // when the source is sub-F32 (quantized or half-precision checkpoints).
    let single_tensor_conversion_bound = inventory.largest_tensor_bytes.saturating_add(
        inventory
            .largest_tensor_elements
            .saturating_mul(SINGLE_TENSOR_F32_DESTINATION_BYTES_PER_ELEMENT),
    );
    // Allocation-count bound: instantiation overhead of the whole checkpoint.
    // A MoE-shaped checkpoint (many same-sized expert tensors) collapses the
    // largest-tensor terms; this term keeps its load scratch reserved.
    let whole_checkpoint_instantiation = inventory
        .tensor_count
        .saturating_mul(PER_TENSOR_INSTANTIATION_SLACK_BYTES);
    let load_scratch_bytes = single_tensor_working_copy
        .max(single_tensor_conversion_bound)
        .max(whole_checkpoint_instantiation);
    let load_peak_bytes = inventory
        .total_bytes
        .checked_add(load_scratch_bytes)
        .ok_or_else(|| Error::ModelLoadError("portable model load estimate overflow".into()))?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes: inventory.total_bytes,
    })
}

fn portable_model_memory_estimate(
    backend: BackendKind,
    variant: ModelVariant,
    model_path: &Path,
) -> Result<ModelMemoryEstimate> {
    if variant == ModelVariant::Qwen3827BFp8 {
        return Ok(qwen38_representation_memory_estimate(
            backend,
            &crate::performance::PerformanceConfig::default(),
        ));
    }
    let catalog = model_memory_estimate(variant);
    estimate_from_tensor_inventory(catalog, checkpoint_tensor_inventory(model_path)?)
}

fn model_resource_plan(backend: BackendKind, estimate: ModelMemoryEstimate) -> ModelResourcePlan {
    let mut resident_authorization = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resident_authorization.host_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
        BackendKind::Metal => {
            resident_authorization.unified_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
        BackendKind::Cuda => {
            resident_authorization.device_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
    }

    let mut load_authorization = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            load_authorization.host_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
        BackendKind::Metal => {
            load_authorization.unified_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
        BackendKind::Cuda => {
            load_authorization.device_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
    }
    if backend == BackendKind::Cuda {
        // CUDA loaders materialize host-side artifact/tensor state before or
        // while copying the resident weights to the device. Authorize both
        // peaks up front; the host component is shed after publication.
        load_authorization.host_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
    }

    ModelResourcePlan {
        load_authorization,
        resident_authorization,
        deferred_resident_authorization: ResourceVector::zero(),
    }
}

/// Fixture-mode qwen38 estimate (synthetic geometry opt-in): derived from the
/// actual checkpoint inventory instead of the pinned 27B constants, while
/// reserving the same portable conversion scratch the pinned estimate carries.
fn qwen38_synthetic_fixture_estimate(model_path: &Path) -> Result<ModelMemoryEstimate> {
    let overflow = || Error::ModelLoadError("Qwen3.8 fixture memory estimate overflow".into());
    let Some(inventory) = checkpoint_tensor_inventory(model_path)? else {
        return Err(Error::ModelLoadError(
            "Synthetic Qwen3.8 fixture has no readable tensor inventory".into(),
        ));
    };
    // The portable load path materializes expanded F32 projections from the
    // FP8/BF16 source bytes; a 4x envelope covers the resident expansion and
    // the portable conversion scratch bounds the load peak like the pinned
    // estimate does.
    let resident_bytes = inventory.total_bytes.checked_mul(4).ok_or_else(overflow)?;
    let load_peak_bytes = resident_bytes
        .checked_add(QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES)
        .and_then(|bytes| bytes.checked_add(inventory.largest_tensor_bytes.next_power_of_two()))
        .ok_or_else(overflow)?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes,
    })
}

fn fish_s2_resource_plan(
    backend: BackendKind,
    memory: crate::models::architectures::fish_s2::weights::FishS2ModelMemory,
) -> Result<ModelResourcePlan> {
    let mut plan = model_resource_plan(
        backend,
        ModelMemoryEstimate {
            load_peak_bytes: memory.load_peak_bytes,
            resident_bytes: memory.resident_bytes,
        },
    );
    if backend == BackendKind::Cuda {
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(memory.cuda_host_load_peak_bytes);
    }
    // Reference VQ codes are immutable CPU data owned by this model instance,
    // even when weights reside on CUDA or Metal. Reserve their complete bounded
    // allocation before load and keep it charged until model unload.
    let reference_cache = ResourceVector {
        host_bytes: ResourceAmount::Known(
            crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES,
        ),
        ..ResourceVector::zero()
    };
    plan.load_authorization = plan.load_authorization.checked_add(reference_cache)?;
    plan.resident_authorization = plan.resident_authorization.checked_add(reference_cache)?;
    // Cache entries are populated lazily. Keep their promise unmaterialized so
    // state fitting cannot consume this headroom before the first reference is
    // encoded. As with lazy graph caches, this remains conservative once filled.
    plan.deferred_resident_authorization = reference_cache;
    Ok(plan)
}

#[derive(Debug, Clone)]
struct InvocationAllocationV2 {
    adapter_instance: AdapterInstanceId,
    key: InvocationWorkspaceKeyV2,
    domain: InvocationWorkspaceDomain,
    slot_count: u32,
}

fn invalid_invocation_publication(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}

fn validate_physical_publication_backing(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
    retained: Option<&RetainedStateRuntimeV2>,
    retained_uses: &HashMap<[u8; 32], RetainedStateUseV2>,
) -> Result<()> {
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    match (&descriptor.retained, retained) {
        (RetainedStateCapability::Stateless, None) if retained_uses.is_empty() => Ok(()),
        (RetainedStateCapability::Managed { contract }, Some(retained))
            if contract.fingerprint()? == retained.state_plan_v2().contract_fingerprint
                && retained_uses.keys().copied().collect::<HashSet<_>>() == execution_graphs =>
        {
            Ok(())
        }
        (RetainedStateCapability::Stateless, None) => Err(invalid_invocation_publication(
            "invocation-only publication cannot declare retained graph mappings",
        )),
        (RetainedStateCapability::Stateless, Some(_)) => Err(invalid_invocation_publication(
            "invocation-only publication unexpectedly owns retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, None) => Err(invalid_invocation_publication(
            "retained invocation publication is missing physical retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, Some(_)) => Err(invalid_invocation_publication(
            "retained invocation publication has mismatched backing or graph mappings",
        )),
    }
}

/// Resolve one capability-authored invocation descriptor into exact physical
/// allocations. This is deliberately model-neutral: graph, stage, domain, and
/// concurrency identities all come from the sealed execution/state contracts.
fn plan_invocation_allocations(
    descriptor: &CapabilityStateDescriptorV2,
    invocation_contract: &InferenceStateContract,
    executions: &[LoadedExecutionContract],
) -> Result<Vec<InvocationAllocationV2>> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no execution stage graphs",
        ));
    }
    invocation_contract.validate()?;

    let mut contract_domains = HashMap::new();
    for domain in &invocation_contract.domains {
        if domain.scope() != StateScope::Invocation {
            return Err(invalid_invocation_publication(
                "physical invocation contract contains retained state",
            ));
        }
        if contract_domains.insert(domain.id(), domain).is_some() {
            return Err(invalid_invocation_publication(
                "physical invocation contract repeats a state domain",
            ));
        }
    }

    let mut executions_by_graph = HashMap::new();
    for execution in executions {
        let graph = stage_graph_fingerprint(&execution.stages)?;
        match executions_by_graph.entry(graph) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(execution);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let current = entry.get();
                if current.adapter_instance_id != execution.adapter_instance_id
                    || current.stages != execution.stages
                {
                    return Err(invalid_invocation_publication(
                        "one invocation stage graph maps to multiple loaded adapters",
                    ));
                }
            }
        }
    }

    let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no bounded workspace profiles",
        ));
    };
    let execution_graphs = executions_by_graph.keys().copied().collect::<HashSet<_>>();
    let profile_graphs = profiles
        .iter()
        .map(|profile| profile.stage_graph_fingerprint)
        .collect::<HashSet<_>>();
    if profile_graphs.len() != profiles.len() || profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "physical invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }

    let mut mapped_domains: HashMap<StateDomainId, &StateDomainSpec> = HashMap::new();
    let mut physical_keys = HashSet::new();
    let mut allocations = Vec::new();
    for profile in profiles {
        let execution = executions_by_graph
            .get(&profile.stage_graph_fingerprint)
            .copied()
            .ok_or_else(|| {
                invalid_invocation_publication(
                    "physical invocation profile lost its loaded stage graph",
                )
            })?;
        descriptor.validate_against_stages(&execution.stages)?;
        for workspace in &profile.stages {
            let stage = execution
                .stages
                .iter()
                .find(|candidate| candidate.id == workspace.stage)
                .ok_or_else(|| {
                    invalid_invocation_publication(
                        "physical invocation profile lost its execution stage",
                    )
                })?;
            let slot_count = workspace.slot_count(stage.max_batch_size)?;
            for domain in &workspace.domains {
                let InvocationWorkspaceDomain::State {
                    state, capacity, ..
                } = domain
                else {
                    // Scratch is accounted by the stage workspace formula and
                    // has no persistent typed backing to allocate here.
                    continue;
                };
                let capacity_matches = matches!(state, StateDomainSpec::PagedAttention(_))
                    && capacity.paged_max_tokens().is_some()
                    || matches!(
                        (state, capacity),
                        (
                            StateDomainSpec::StaticAttention(_)
                                | StateDomainSpec::Tensor(_)
                                | StateDomainSpec::Append(_)
                                | StateDomainSpec::Ring(_)
                                | StateDomainSpec::StaticTensor(_),
                            InvocationStateCapacity::SemanticBounded
                        )
                    );
                if !capacity_matches {
                    return Err(invalid_invocation_publication(
                        "physical invocation descriptor uses a capacity incompatible with its state domain",
                    ));
                }
                match mapped_domains.entry(state.id()) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(state);
                    }
                    std::collections::hash_map::Entry::Occupied(entry) if *entry.get() == state => {
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {
                        return Err(invalid_invocation_publication(
                            "one invocation domain has inconsistent physical definitions",
                        ));
                    }
                }
                let key = InvocationWorkspaceKeyV2 {
                    stage_graph: profile.stage_graph_fingerprint,
                    stage: workspace.stage,
                    domain: state.id(),
                };
                if !physical_keys.insert(key) {
                    return Err(invalid_invocation_publication(
                        "physical invocation publication repeats a graph/stage/domain mapping",
                    ));
                }
                allocations.push(InvocationAllocationV2 {
                    adapter_instance: execution.adapter_instance_id,
                    key,
                    domain: domain.clone(),
                    slot_count,
                });
            }
        }
    }

    if mapped_domains.len() != contract_domains.len()
        || mapped_domains.iter().any(|(id, state)| {
            contract_domains
                .get(id)
                .is_none_or(|contract_state| *contract_state != *state)
        })
    {
        return Err(invalid_invocation_publication(
            "physical invocation descriptor and contract have missing or extra domain mappings",
        ));
    }
    if allocations.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no typed state domains",
        ));
    }
    Ok(allocations)
}

fn validate_scratch_only_invocation_publication(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
) -> Result<()> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "scratch-only invocation publication has no execution stage graphs",
        ));
    }
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    let profile_graphs = match &descriptor.invocation {
        InvocationWorkspaceSet::None {
            stage_graph_fingerprints,
        } => stage_graph_fingerprints
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        InvocationWorkspaceSet::Bounded { profiles } => {
            if profiles.iter().any(|profile| {
                profile.stages.iter().any(|stage| {
                    stage
                        .domains
                        .iter()
                        .any(|domain| matches!(domain, InvocationWorkspaceDomain::State { .. }))
                })
            }) {
                return Err(invalid_invocation_publication(
                    "scratch-only invocation publication contains typed state",
                ));
            }
            profiles
                .iter()
                .map(|profile| profile.stage_graph_fingerprint)
                .collect::<HashSet<_>>()
        }
    };
    if profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "scratch-only invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }
    for execution in executions {
        descriptor.validate_against_stages(&execution.stages)?;
    }
    Ok(())
}

impl ModelLifecycleController {
    /// Seal each advertised Fish route against its own adapter identity. Slow
    /// retained state is model-scoped and reused; invocation backing is per adapter.
    async fn load_fish_s2_state_publications(
        &self,
        model_instance_id: ModelInstanceId,
        bundle_draft: &LoadedModelBundleDraft,
        max_sequence_tokens: usize,
        capacity: &crate::runtime::fish_capacity::ResolvedFishServingCapacity,
        physical_state_spec: impl Fn(&[&[StageDescriptor]]) -> Result<FishS2PhysicalStateSpec>,
    ) -> Result<HashMap<CapabilityKind, LoadedStatePublication>> {
        let variant = ModelVariant::FishAudioS2Pro;
        let mut publications = HashMap::new();
        let mut retained_runtime: Option<Arc<crate::engine::ManagedKvModelRuntime>> = None;
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let contracts = bundle_draft.execution_contracts(capability)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = physical_state_spec(&stage_graphs)?;
            let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                Error::ModelLoadError(
                    "Fish S2 TTS normal graph did not publish retained state".into(),
                )
            })?;
            // Fit and allocate slow state once. Refitting after its allocation
            // would incorrectly price another copy against remaining headroom.
            let retained = match &retained_runtime {
                Some(retained) => {
                    if retained.state_plan_v2().contract_fingerprint
                        != retained_contract.fingerprint()?
                    {
                        return Err(Error::ModelLoadError(
                            "Fish capabilities disagree on retained state".into(),
                        ));
                    }
                    retained.clone()
                }
                None => {
                    let retained = self
                        .core_engine
                        .load_managed_model_state_with_row_limits(
                            model_instance_id,
                            retained_contract,
                            Some(max_sequence_tokens),
                            capacity.active_rows,
                            capacity.staged_rows,
                        )
                        .await?;
                    retained_runtime = Some(retained.clone());
                    retained
                }
            };
            self.model_registry
                .publish_effective_context(variant, retained.logical_token_reach())?;
            self.model_registry.publish_kv_storage(variant, &retained);
            crate::runtime::rollout::validate_managed_state_plan_eligibility(
                variant,
                capability,
                retained.state_plan_v2(),
            )?;
            let retained_uses = contracts
                .iter()
                .map(|contract| {
                    let graph = stage_graph_fingerprint(&contract.stages)?;
                    let retained_use = match contract.execution_profile.cache_mode {
                        CacheMode::ExternalPaged => RetainedStateUseV2::ExternalPaged,
                        CacheMode::None => RetainedStateUseV2::Inactive,
                    };
                    Ok((graph, retained_use))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let publication = self
                .load_invocation_workspace_publication(
                    model_instance_id,
                    &contracts,
                    physical_spec.descriptor,
                    &physical_spec.invocation,
                    Some(retained.into()),
                    retained_uses,
                )
                .await?;
            publications.insert(capability, publication);
        }
        Ok(publications)
    }

    fn load_scratch_only_workspace_publication(
        &self,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        validate_scratch_only_invocation_publication(&descriptor, executions)?;
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
        })
    }

    fn fit_invocation_decoder_context(
        &self,
        variant: ModelVariant,
        descriptor: &mut CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        executions: &[LoadedExecutionContract],
        remaining_automatic_state_groups: u64,
    ) -> Result<()> {
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        let axis_bounds = descriptor.capacity_axis_bounds(StateCapacityAxis::DecoderContext);
        // ASR state leaves live room for the requests it serves (ADM-7), and
        // keeps the one reserve the retained fit keeps too (ADM-11): the
        // configured floor or the measured untracked growth, not a catalog
        // difference.
        let authority = self.coordinator.resource_authority();
        let asr = asr_memory::is_asr_variant(variant);
        let headroom = authority.planning_headroom_bytes(backend)?;
        let (live_headroom, safety_reserve_bytes) = if asr {
            (
                authority.state_fit_headroom_bytes(backend)?,
                authority.state_safety_reserve_bytes(
                    backend,
                    self.config.portable_context_reserve_bytes,
                ),
            )
        } else {
            (
                headroom,
                portable_context_reserve_bytes(variant, self.config.portable_context_reserve_bytes),
            )
        };
        // A capacity shortfall of ASR state is a 503, not a load defect.
        let does_not_fit = |message: String| {
            if asr {
                Error::Overloaded(message)
            } else {
                Error::ModelLoadError(message)
            }
        };
        let ResourceAmount::Known(headroom_bytes) = headroom else {
            let Some((_, maximum)) = axis_bounds else {
                return Ok(());
            };
            let fallback = self
                .config
                .max_sequence_length
                .explicit_tokens()
                .map_or(maximum.min(4_096), |tokens| maximum.min(tokens as u64));
            descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, fallback)?;
            self.model_registry
                .publish_effective_context(variant, fallback)?;
            return Ok(());
        };
        let (budget, growth_budget) =
            state_fit_budgets(headroom_bytes, live_headroom, safety_reserve_bytes);
        let automatic_budget =
            automatic_state_group_budget(growth_budget, remaining_automatic_state_groups);
        let required = |tokens: Option<u64>| -> Result<u64> {
            let mut candidate = descriptor.clone();
            if let Some(tokens) = tokens {
                candidate.resolve_capacity_axis(StateCapacityAxis::DecoderContext, tokens)?;
            }
            plan_invocation_allocations(&candidate, invocation_contract, executions)?
                .into_iter()
                .try_fold(0_u64, |total, allocation| {
                    total
                        .checked_add(
                            allocation
                                .domain
                                .maximum_bytes()?
                                .checked_mul(u64::from(allocation.slot_count))
                                .ok_or_else(|| {
                                    Error::ModelLoadError(
                                        "invocation context byte plan overflow".into(),
                                    )
                                })?,
                        )
                        .ok_or_else(|| {
                            Error::ModelLoadError("invocation context byte plan overflow".into())
                        })
                })
        };
        let Some((minimum, maximum)) = axis_bounds else {
            let bytes = required(None)?;
            if bytes > budget {
                tracing::warn!(
                    "portable fixed invocation state does not fit: state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(capacity_shortfall_reason(
                    format_args!("not enough memory for {variant}'s working state"),
                    bytes,
                    budget,
                )));
            }
            return Ok(());
        };
        // The invocation context ceiling is the model-authored maximum; the
        // memory-budgeted binary search below decides how much of it fits.
        let intent = portable_invocation_context_intent(
            self.config.max_sequence_length,
            self.model_registry.effective_context(variant),
            maximum,
        );
        let selected = if let PortableInvocationContextIntent::Explicit { selected } = intent {
            let bytes = required(Some(selected))?;
            if bytes > budget {
                tracing::warn!(
                    "explicit portable context does not fit: context={selected}, state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(capacity_shortfall_reason(
                    format_args!("not enough memory for {variant} at a {selected}-token context"),
                    bytes,
                    budget,
                )));
            }
            selected
        } else {
            let PortableInvocationContextIntent::Automatic { ceiling } = intent else {
                unreachable!("portable invocation context intent was matched above")
            };
            // An ASR invocation holds at least one chunk per sequence, unless
            // the model's own context is smaller (P2.5).
            let minimum = if asr {
                minimum.max(asr_minimum_tokens(variant).min(maximum))
            } else {
                minimum
            };
            if ceiling < minimum {
                return Err(does_not_fit(format!(
                    "published automatic context is below the invocation minimum: context={ceiling}, minimum={minimum}"
                )));
            }
            let minimum_bytes = required(Some(minimum))?;
            if minimum_bytes > budget {
                tracing::warn!(
                    "portable invocation context minimum does not fit: context={minimum}, state_bytes={minimum_bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                );
                return Err(does_not_fit(capacity_shortfall_reason(
                    format_args!(
                        "not enough memory for {variant}'s minimum {minimum}-token context"
                    ),
                    minimum_bytes,
                    budget,
                )));
            }
            let (mut low, mut high) = (minimum, ceiling);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required(Some(middle))? <= automatic_budget {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        };
        descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, selected)?;
        let published = self
            .model_registry
            .effective_context(variant)
            .map_or(selected, |current| selected.min(current as u64));
        self.model_registry
            .publish_effective_context(variant, published)?;
        Ok(())
    }

    async fn load_invocation_workspace_publication(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        self.load_invocation_workspace_publication_with_remaining_groups(
            model_instance_id,
            executions,
            descriptor,
            invocation_contract,
            retained,
            retained_uses,
            1,
        )
        .await
    }

    async fn load_invocation_workspace_publication_with_remaining_groups(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        mut descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
        remaining_automatic_state_groups: u64,
    ) -> Result<LoadedStatePublication> {
        let variant = self
            .resident_variant_for_instance(model_instance_id)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "invocation state lost its authoritative model generation".into(),
                )
            })?;
        self.fit_invocation_decoder_context(
            variant,
            &mut descriptor,
            invocation_contract,
            executions,
            remaining_automatic_state_groups,
        )?;
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        let allocations =
            plan_invocation_allocations(&descriptor, invocation_contract, executions)?;
        let mut bindings = Vec::with_capacity(allocations.len());
        for allocation in allocations {
            let backing = self
                .core_engine
                .resolve_and_load_invocation_workspace(
                    model_instance_id,
                    allocation.adapter_instance,
                    allocation.key.stage_graph,
                    allocation.key.stage,
                    invocation_contract,
                    &allocation.domain,
                    allocation.slot_count,
                )
                .await?;
            bindings.push(InvocationWorkspaceBindingV2 {
                key: allocation.key,
                backing,
            });
        }
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::new(bindings)?,
        })
    }

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

    pub(super) async fn ensure_model_budget_before_load(
        &self,
        requested_variant: ModelVariant,
        max_loaded_models: Option<usize>,
    ) -> Result<()> {
        let Some(max_loaded_models) = max_loaded_models else {
            return Ok(());
        };

        loop {
            let resident_variants = self.known_resident_variants().await;
            if residency_budget_has_capacity(
                &resident_variants,
                requested_variant,
                max_loaded_models,
            ) {
                return Ok(());
            }

            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let mut ready_variants = Vec::with_capacity(resident_variants.len());
            for variant in &resident_variants {
                if self.resident_phase(*variant)
                    == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                    || self.model_manager.is_ready(*variant).await
                {
                    ready_variants.push(*variant);
                }
            }
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &ready_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                return Err(Error::Overloaded(format!(
                    "Cannot load {requested_variant}: the {max_loaded_models}-model residency budget is full and no resident model is idle and available for eviction; unload a model to make room"
                )));
            };

            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                max_loaded_models,
                "Evicting idle model before loading its replacement"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    fn model_resource_plan(
        &self,
        variant: ModelVariant,
        model_path: &Path,
    ) -> Result<ModelResourcePlan> {
        let backend = self.backend_router.context().backend_kind;
        if variant == ModelVariant::Qwen3827BFp8 {
            if backend != BackendKind::Cuda
                && crate::models::architectures::qwen38::native::synthetic_geometry_enabled()
            {
                // Fixture load (benchmark/CI): price the actual checkpoint
                // instead of the pinned 27B constants.
                let estimate = qwen38_synthetic_fixture_estimate(model_path)?;
                return Ok(model_resource_plan(backend, estimate));
            }
            // Every backend prices the checkpoint under the residency policy
            // the loader resolves (CUDA runs its cached device gates here).
            return qwen38_memory::resource_plan(
                model_path,
                &self.backend_router.context().device.device,
                &self.config.performance,
            );
        }
        if variant == ModelVariant::Qwen36Moe35BA3BFp8 {
            if crate::models::architectures::qwen36moe::native::synthetic_geometry_enabled() {
                // Fixture load (benchmark/CI): price the actual checkpoint
                // instead of the pinned 35B constants.
                let estimate = qwen36moe_memory::synthetic_fixture_estimate(model_path)?;
                return Ok(model_resource_plan(backend, estimate));
            }
            return qwen36moe_memory::resource_plan(backend, &self.config.performance);
        }
        if let Some(plan) = checkpoint_resource_plan(
            variant,
            model_path,
            &self.backend_router.context().device,
            &self.config.granite_speech,
        )? {
            return Ok(plan);
        }
        let estimate = if backend == BackendKind::Cuda {
            model_memory_estimate(variant)
        } else {
            portable_model_memory_estimate(backend, variant, model_path)?
        };
        Ok(model_resource_plan(backend, estimate))
    }

    /// The recurrent state floor of a model whose CUDA context is fitted
    /// against resident headroom; `None` for every other load.
    fn cuda_state_floor(
        &self,
        variant: ModelVariant,
        model_path: &Path,
    ) -> Result<Option<CudaStateFloor>> {
        let policy =
            managed_chat_capacity_policy(variant, self.backend_router.context().backend_kind);
        if policy.cuda_resident_pool == crate::engine::CudaResidentPool::Disabled {
            return Ok(None);
        }
        let per_sequence_bytes = match variant {
            ModelVariant::Qwen3827BFp8 => qwen38_memory::per_sequence_state_bytes(model_path)?,
            ModelVariant::Qwen36Moe35BA3BFp8 => {
                use crate::models::architectures::qwen36moe::native as qwen36moe;
                if qwen36moe::synthetic_geometry_enabled() {
                    // Fixture loads do not have the pinned 35B geometry.
                    return Ok(None);
                }
                hybrid_recurrent_state_bytes_per_sequence(qwen36moe_state_geometry(
                    &qwen36moe::pinned_native_config().text,
                ))?
            }
            _ => return Ok(None),
        };
        let engine = self.core_engine.config();
        let staged_rows = match policy.staged_transaction_rows {
            Some(rows) => usize::try_from(rows).unwrap_or(usize::MAX),
            None => engine.max_staged_transactions,
        };
        let automatic_rows = engine.automatic_row_limits.retained_sequences;
        let retained_rows = if automatic_rows {
            staged_rows.min(engine.max_retained_sequences)
        } else {
            engine.max_retained_sequences
        };
        // The fit reserves the prefix snapshot budget beside the rows when
        // the model shares hybrid prefixes (see `prefix_snapshot_rows_for`).
        let snapshot_rows = if engine.enable_prefix_caching
            && engine.prefix_cache_pages_or_default() > 0
            && crate::catalog::prefix_reuse_engages(
                variant,
                BackendKind::Cuda,
                self.model_registry.prefix_reuse_mode(),
            ) {
            usize::try_from(engine.max_prefix_snapshots).unwrap_or(usize::MAX)
        } else {
            0
        };
        Ok(Some(CudaStateFloor {
            retained_rows,
            staged_rows: staged_rows.min(retained_rows),
            snapshot_rows,
            per_sequence_bytes,
            automatic_rows,
        }))
    }

    async fn ensure_portable_memory_before_load(
        &self,
        requested_variant: ModelVariant,
        required_bytes: u64,
    ) -> Result<()> {
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        let mut pooled_scratch_flushed = false;
        loop {
            let ResourceAmount::Known(planning_headroom) = self
                .coordinator
                .resource_authority()
                .planning_headroom_bytes(backend)?
            else {
                return Ok(());
            };
            // Metal loads allocate the full working set at the load fence, so
            // the pre-load check must also respect probed live availability —
            // static ledger estimates alone under-count GGUF materialization
            // and pooled scratch. An unavailable live probe never blocks a
            // load; planning headroom remains the contract then.
            let live_headroom = self
                .coordinator
                .resource_authority()
                .live_preload_headroom_bytes(backend)?;
            let headroom = live_headroom
                .map(|live| planning_headroom.min(live))
                .unwrap_or(planning_headroom);
            if required_bytes <= headroom {
                return Ok(());
            }
            let resident_variants = self.known_resident_variants().await;
            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &resident_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                if backend == BackendKind::Metal && !pooled_scratch_flushed {
                    // The pool retains released activation scratch until the
                    // last model unloads, which can hide gigabytes of
                    // reclaimable headroom from the live sample. Flushing it
                    // is safe (only unchecked-out buffers are dropped) and
                    // beats failing a load that would fit. One flush per
                    // admission decision; persistent shortage still errors.
                    pooled_scratch_flushed = true;
                    info!(
                        requested_variant = %requested_variant,
                        planning_headroom,
                        live_headroom,
                        "Reclaiming pooled Metal scratch before rejecting the load"
                    );
                    self.flush_metal_pools().await;
                    self.coordinator
                        .resource_authority()
                        .refresh_physical_capacity_after_release();
                    continue;
                }
                tracing::warn!(
                    "Cannot fit {requested_variant} model tensors before state allocation: load_peak_bytes={required_bytes}, planning_headroom={planning_headroom}, live_headroom={live_headroom:?}, backend={backend:?}"
                );
                return Err(Error::Overloaded(format!(
                    "{}; no idle resident model is available for eviction; unload a model to make room",
                    capacity_shortfall_reason(
                        format_args!("Cannot load {requested_variant}"),
                        required_bytes,
                        headroom,
                    )
                )));
            };
            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                required_bytes,
                planning_headroom = headroom,
                "Evicting idle model before portable tensor allocation"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    /// One reconciliation checkpoint for a load priced by a residency plan
    /// (ASR and TTS), or `None` for other families and backends without a
    /// device observer (CPU).
    fn sample_device_usage(&self, variant: ModelVariant) -> Option<DeviceUsageSample> {
        if !reconciles_device_usage(variant) {
            return None;
        }
        let authority = self.coordinator.resource_authority();
        let observed = authority.observed_device_usage_bytes()?;
        Some(DeviceUsageSample {
            observed,
            reserved: authority
                .reserved_bytes_in_domain(self.backend_router.context().backend_kind),
        })
    }

    /// Staged rows for a chat model on CPU or Metal when the engine's are
    /// automatic: one step's launch at the model's own decode width, which
    /// may exceed the engine rows' portable width (`catalog::batch_width`).
    /// `None` keeps the engine rows (CUDA, or an operator's explicit limit).
    fn portable_chat_staged_rows(
        &self,
        bundle_draft: &LoadedModelBundleDraft,
        backend: BackendKind,
    ) -> Result<Option<u32>> {
        let engine = self.core_engine.config();
        if backend == BackendKind::Cuda || !engine.automatic_row_limits.staged_transactions {
            return Ok(None);
        }
        let width = bundle_draft
            .execution_contracts(CapabilityKind::Chat)?
            .iter()
            .flat_map(|contract| contract.stages.iter())
            .map(|stage| stage.max_batch_size)
            .max()
            .unwrap_or(1);
        let rows = crate::runtime::service::automatic_staged_transaction_rows(
            width,
            engine
                .resolved_physical_execution_capacity()
                .candidate_dispatch_limit
                .get(),
            engine.max_retained_sequences,
        );
        u32::try_from(rows)
            .map(Some)
            .map_err(|_| Error::InvalidInput("managed state transaction limit exceeds u32".into()))
    }

    fn asr_state_rows(&self) -> (usize, usize) {
        let engine = self.core_engine.config();
        asr_state_rows(
            engine.automatic_row_limits.retained_sequences,
            engine.max_retained_sequences,
            engine.max_staged_transactions,
        )
    }

    /// Reserve one request envelope per execution slot for the state fits
    /// to leave free (ASR: one full chunk; TTS: one request at the largest
    /// automatic budget), or `None` for a variant without one.
    async fn hold_request_envelope(
        &self,
        variant: ModelVariant,
        backend: BackendKind,
    ) -> Result<Option<ResourceLease>> {
        let slots = self.coordinator.physical_execution_admission().capacity();
        let (envelope, label) =
            match super::asr_envelope::deferred_request_envelope(variant, backend, slots)? {
                Some(envelope) => (envelope, "asr"),
                None => match self.tts_request_envelope(variant).await? {
                    Some(envelope) => (envelope.scaled(slots)?, "tts"),
                    None => return Ok(None),
                },
            };
        let resources = request_envelope_resources(backend, envelope)?;
        let authority = self.coordinator.resource_authority();
        let lease = match authority.reserve(
            ReservationOwner::new(
                ReservationClass::Model,
                format!("{variant}:{label}-request-envelope"),
            ),
            resources,
        ) {
            Ok(lease) => lease,
            Err(Error::Overloaded(detail)) => {
                let available = match authority.planning_headroom_bytes(backend)? {
                    ResourceAmount::Known(bytes) => bytes,
                    ResourceAmount::Unknown => 0,
                };
                tracing::warn!(
                    model = %variant,
                    request_slots = slots,
                    host_bytes = envelope.host_bytes,
                    accelerator_bytes = envelope.accelerator_bytes,
                    planning_headroom = available,
                    %detail,
                    "The {label} request envelope does not fit beside the loaded weights"
                );
                return Err(request_envelope_shortfall(
                    variant, backend, envelope, available,
                )?);
            }
            Err(error) => return Err(error),
        };
        info!(
            model = %variant,
            request_slots = slots,
            host_bytes = envelope.host_bytes,
            accelerator_bytes = envelope.accelerator_bytes,
            "Holding the {label} request envelope out of the state fits"
        );
        Ok(Some(lease))
    }

    /// One TTS request's envelope, priced by the loaded model (P2.3), for
    /// the families whose load fits state beside it: Qwen3-TTS.
    async fn tts_request_envelope(
        &self,
        variant: ModelVariant,
    ) -> Result<Option<super::asr_envelope::AsrRequestEnvelope>> {
        if variant.family() != crate::catalog::ModelFamily::Qwen3Tts {
            return Ok(None);
        }
        let Some(model) = self.model_registry.get_qwen_tts(variant).await else {
            return Ok(None);
        };
        let (host_bytes, accelerator_bytes) = model.request_envelope_bytes()?;
        Ok(Some(super::asr_envelope::AsrRequestEnvelope {
            host_bytes,
            accelerator_bytes,
        }))
    }

    async fn reserve_model_resources(
        &self,
        requested_variant: ModelVariant,
        load_authorization: ResourceVector,
    ) -> Result<ResourceLease> {
        loop {
            match self.coordinator.resource_authority().reserve(
                ReservationOwner::new(ReservationClass::Model, requested_variant.to_string()),
                load_authorization,
            ) {
                Ok(lease) => return Ok(lease),
                Err(Error::Overloaded(resource_error)) => {
                    let resident_variants = self.known_resident_variants().await;
                    let mut active_variants = self.core_engine.active_model_variants().await;
                    active_variants.extend(resident_variants.iter().copied().filter(|variant| {
                        self.model_manager.active_residency_leases(*variant) > 0
                    }));
                    let mut ready_variants = Vec::new();
                    for variant in &resident_variants {
                        if self.resident_phase(*variant)
                            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                            || self.model_manager.is_ready(*variant).await
                        {
                            ready_variants.push(*variant);
                        }
                    }
                    let last_used = self.model_last_used.lock().await.clone();
                    let pinned_variants = self.pinned_variants.lock().await.clone();
                    let Some(victim) = select_lru_eviction_candidate(
                        &ready_variants,
                        requested_variant,
                        &active_variants,
                        &pinned_variants,
                        &last_used,
                    ) else {
                        return Err(Error::Overloaded(format!(
                            "Cannot reserve memory for {requested_variant}: {resource_error}; no idle resident model is available for eviction; unload a model to make room"
                        )));
                    };
                    info!(
                        requested_variant = %requested_variant,
                        victim = %victim,
                        "Evicting idle model to satisfy the physical memory budget"
                    );
                    self.unload_model_locked(victim).await?;
                }
                Err(err) => return Err(err),
            }
        }
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
        if asr_memory::is_asr_variant(variant) {
            let backend = self.backend_router.context().backend_kind;
            let slots = self.coordinator.physical_execution_admission().capacity();
            if let Some(envelope) =
                super::asr_envelope::deferred_request_envelope(variant, backend, slots)?
            {
                let floor_bytes = match backend {
                    BackendKind::Cuda => envelope.accelerator_bytes,
                    BackendKind::Cpu | BackendKind::Metal => envelope.total_bytes()?,
                };
                resource_plan.load_authorization =
                    admit_asr_request_floor(&resource_plan, backend, floor_bytes)?;
            }
        }
        self.ensure_model_budget_before_load(variant, max_loaded_models)
            .await?;
        let portable_load_bytes = match self.backend_router.context().backend_kind {
            BackendKind::Cpu => resource_plan.load_authorization.host_bytes,
            BackendKind::Metal => resource_plan.load_authorization.unified_bytes,
            BackendKind::Cuda => ResourceAmount::Known(0),
        };
        if let ResourceAmount::Known(required_bytes) = portable_load_bytes {
            self.ensure_portable_memory_before_load(variant, required_bytes)
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
            let instantiated = self.instantiate_model(acquired).await?;
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
            if reconciles_device_usage(variant) {
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

    /// Variants explicitly loaded (pinned) in this server session.
    pub async fn pinned_model_variants(&self) -> std::collections::HashSet<ModelVariant> {
        self.model_lifecycle.pinned_model_variants().await
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

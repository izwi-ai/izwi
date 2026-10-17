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
) -> Result<Option<ModelResourcePlan>> {
    if variant == ModelVariant::FishAudioS2Pro {
        let memory = crate::models::architectures::fish_s2::weights::fish_s2_model_memory(
            model_path, device,
        )?;
        return fish_s2_resource_plan(BackendKind::from(device.kind), memory).map(Some);
    }
    // ASR, TTS and LFM2.5 chat price what their loader allocates on every
    // backend; the catalog figure stays a display value for them.
    if let Some(plan) = asr_memory::resource_plan(variant, model_path, device)? {
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
/// priced ([`checkpoint_resource_plan`]).
fn retry_load_peak_bytes(
    variant: ModelVariant,
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
    checkpoint: Option<(&Path, &crate::backends::DeviceProfile)>,
) -> u64 {
    let catalog = || model_memory_estimate(variant).load_peak_bytes;
    if let Some((model_path, device)) = checkpoint {
        if let Ok(Some(plan)) = checkpoint_resource_plan(variant, model_path, device) {
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
        if let Some(plan) =
            checkpoint_resource_plan(variant, model_path, &self.backend_router.context().device)?
        {
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
            model_path.as_deref().map(|path| (path, &context.device)),
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
            if let Some(loaded) = self.model_registry.get_chat(variant).await {
                // Freeze exact MTP/collation geometry before deriving any
                // selectable stage graphs or planning physical state.
                bundle_draft.seal_chat_workspace(
                    loaded.continuous_decode_batch_workspace_per_row_bytes()?,
                )?;
                bundle_draft.seal_chat_decode_quantum(
                    loaded.speculative_decode_profile().0.max(1) as u64,
                )?;
                if let Some((per_token_bytes, max_span_tokens)) = loaded.prefill_workspace()? {
                    bundle_draft.seal_chat_prefill_workspace(per_token_bytes, max_span_tokens)?;
                }
                if loaded.supports_mixed_step()
                    && crate::catalog::mixed_step_enabled(variant.family(), backend)
                {
                    let token_budget = match self.config.max_tokens_per_step {
                        0 => crate::engine::backend_step_sizes(backend).max_tokens_per_step,
                        tokens => tokens,
                    };
                    bundle_draft.seal_chat_mixed_step(token_budget as u64)?;
                }
                let decode_workspace_reserve_bytes = bundle_draft
                    .execution_contracts(CapabilityKind::Chat)?
                    .iter()
                    .flat_map(|contract| contract.stages.iter())
                    .map(|stage| stage.max_workspace_bytes)
                    .max()
                    .unwrap_or(0);
                let loaded_cache = loaded.inference_state_contract()?;
                loaded_cache.validate()?;
                let publication = match &loaded_cache {
                    crate::kv::InferenceStateCapability::Managed(contract) => {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} publishes managed KV, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let mut capacity_policy = managed_chat_capacity_policy(variant, backend);
                        if capacity_policy.staged_transaction_rows.is_none() {
                            capacity_policy.staged_transaction_rows =
                                self.portable_chat_staged_rows(&bundle_draft, backend)?;
                        }
                        let physical = self
                            .core_engine
                            .load_managed_model_cache_with_capacity_policy(
                                model_instance_id,
                                &loaded_cache,
                                Some(loaded.max_context_tokens()?),
                                capacity_policy.staged_transaction_rows,
                                capacity_policy.cuda_resident_pool,
                                decode_workspace_reserve_bytes,
                            )
                            .await?;
                        let physical = physical.ok_or_else(|| {
                            Error::ModelLoadError(
                                "managed state allocation returned no physical runtime".to_string(),
                            )
                        })?;
                        self.model_registry.publish_effective_context(
                            variant,
                            physical.maximum_sequence_tokens(),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &physical);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Chat,
                            physical.state_plan_v2(),
                        )?;
                        Some(LoadedStatePublication::ManagedV2 {
                            contract: contract.clone(),
                            physical,
                        })
                    }
                    crate::kv::InferenceStateCapability::Stateless => None,
                };
                if let Some(publication) = publication {
                    state_publications.insert(CapabilityKind::Chat, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical Granite Speech invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Granite Speech model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [
                    CapabilityKind::Asr,
                    CapabilityKind::SpeakerAttributedAsr,
                ] {
                    if self.adapter_registry.require(capability, variant).is_err() {
                        continue;
                    }
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec =
                        model.granite_speech_physical_state_spec(&stage_graphs)?;
                    // Without resumable prefill the model authors no retained
                    // route, and its state is not allocated (GRN-8).
                    let retained_route = capability == CapabilityKind::Asr
                        && physical_spec.retained.is_some();
                    let (retained, retained_uses) = if retained_route {
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "Granite Speech ASR graph did not author retained decoder state"
                                    .into(),
                            )
                        })?;
                        let retained_max_tokens = physical_spec
                            .retained_max_tokens
                            .ok_or_else(|| {
                                Error::ModelLoadError(
                                    "Granite Speech retained decoder has no context bound".into(),
                                )
                            })?;
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                retained_contract,
                                Some(retained_max_tokens),
                                2,
                                self.asr_state_rows(),
                                asr_minimum_tokens(variant),
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &retained);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let use_kind = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "Granite Speech ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
                                };
                                Ok((graph, use_kind))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        (Some(retained.into()), uses)
                    } else {
                        if physical_spec.retained.is_some()
                            || physical_spec.retained_max_tokens.is_some()
                        {
                            return Err(Error::ModelLoadError(
                                "speaker-attributed Granite Speech graph unexpectedly requested retained state"
                                    .into(),
                            ));
                        }
                        (None, HashMap::new())
                    };
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            retained,
                            retained_uses,
                            if capability == CapabilityKind::Asr { 2 } else { 1 },
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if self
                .adapter_registry
                .require(CapabilityKind::Asr, variant)
                .is_ok()
                && variant.family() != crate::catalog::ModelFamily::GraniteSpeechAsr
            {
                if variant.family() == crate::catalog::ModelFamily::Voxtral {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let loaded = self
                        .model_registry
                        .get_loading_voxtral(variant)
                        .await
                        .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Voxtral model {variant} is missing from the registry"
                        ))
                        })?;
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = loaded.physical_state_spec(&stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if let Some(loaded) =
                    self.model_registry.get_loading_asr(variant).await
                {
                    let loaded_cache = loaded.inference_state_contract()?;
                    loaded_cache.validate()?;
                    let publication_route = loaded_asr_state_publication_route(variant);
                    if publication_route == LoadedAsrStatePublicationRoute::Qwen3 {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical ASR state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        // A configured context wins; otherwise Qwen3-ASR bounds
                        // its state by one planned chunk per sequence.
                        let context_bound =
                            if self.config.max_sequence_length.explicit_tokens().is_some() {
                                Qwen3AsrContextBound::Configured
                            } else {
                                Qwen3AsrContextBound::Automatic {
                                    retained_rows: self.asr_state_rows().0,
                                }
                            };
                        let physical_spec =
                            loaded.qwen3_physical_state_spec(&stage_graphs, context_bound)?;
                        let retained_max_tokens = physical_spec.retained_max_tokens;
                        let physical = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                                2,
                                self.asr_state_rows(),
                                asr_minimum_tokens(variant),
                            )
                            .await?;
                        // The retained pool serves several sequences, but one
                        // sequence or invocation never exceeds its own bound.
                        self.model_registry.publish_effective_context(
                            variant,
                            physical
                                .logical_token_reach()
                                .min(physical_spec.sequence_tokens as u64),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &physical);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            physical.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.cache_mode {
                                    CacheMode::ExternalPaged
                                        if contract.execution_profile.cache_namespace.is_some()
                                            && contract.execution_profile.kv_dtype != "none" =>
                                    {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    CacheMode::None
                                        if contract.execution_profile.cache_namespace.is_none()
                                            && contract.execution_profile.kv_dtype == "none" =>
                                    {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "Qwen3 ASR graph has an incompatible retained-state profile"
                                                .to_string(),
                                        ));
                                    }
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
                                Some(physical.into()),
                                retained_uses,
                            )
                            .await?;
                        state_publications.insert(
                            CapabilityKind::Asr,
                            publication,
                        );
                    } else if publication_route == LoadedAsrStatePublicationRoute::VibeVoice {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.vibevoice_physical_state_spec(&stage_graphs)?;
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "VibeVoice ASR normal graph did not author retained decoder state"
                                    .into(),
                            )
                        })?;
                        let native_tokens = physical_spec.retained_max_tokens.ok_or_else(|| {
                            Error::ModelLoadError(
                                "VibeVoice ASR retained decoder state has no context bound".into(),
                            )
                        })?;
                        // The retained route takes one tokenizer chunk, so
                        // without a configured context each retained row is
                        // bounded by one chunk's prompt and duration budget
                        // (P2.5), not the decoder's native context.
                        let retained_max_tokens =
                            if self.config.max_sequence_length.explicit_tokens().is_some() {
                                native_tokens
                            } else {
                                crate::runtime::asr_automatic_context(
                                    native_tokens,
                                    loaded.vibevoice_retained_sequence_ceiling()?,
                                    self.asr_state_rows().0,
                                )
                                .1
                            };
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                retained_contract,
                                Some(retained_max_tokens),
                                2,
                                self.asr_state_rows(),
                                asr_minimum_tokens(variant),
                            )
                            .await?;
                        // The legacy invocation route keeps the duration-budget
                        // ceiling (32,768 decode tokens); its own fit bounds it
                        // by memory.
                        let invocation_ceiling = native_tokens.min(
                            crate::models::architectures::vibevoice::asr::VIBEVOICE_ASR_AUTO_MAX_TOKENS,
                        ) as u64;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach().max(invocation_ceiling),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &retained);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "VibeVoice ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
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
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route == LoadedAsrStatePublicationRoute::Whisper {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical Whisper ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.whisper_physical_state_spec(&stage_graphs)?;
                        let retained = self
                            .core_engine
                            .load_composite_retained_state(
                                model_instance_id,
                                &physical_spec.retained,
                                physical_spec.retained_static_domain,
                                Some(physical_spec.retained_max_tokens),
                                asr_minimum_tokens(variant),
                            )
                            .await?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = if contract.execution_profile.mode
                                    == crate::engine::ExecutionMode::Sequence
                                {
                                    RetainedStateUseV2::ExternalPagedStatic
                                } else if contract.execution_profile.mode
                                    == crate::engine::ExecutionMode::Atomic
                                {
                                    RetainedStateUseV2::Inactive
                                } else {
                                    return Err(Error::ModelLoadError(
                                        "Whisper graph has an incompatible retained-state profile"
                                            .into(),
                                    ));
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
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route == LoadedAsrStatePublicationRoute::Parakeet {
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.parakeet_physical_state_spec(&stage_graphs)?;
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "Parakeet sequence graph did not author retained recurrent state"
                                    .into(),
                            )
                        })?;
                        let retained = self
                            .core_engine
                            .load_retained_tensor_state(
                                model_instance_id,
                                retained_contract,
                                self.realtime_asr_sequence_capacity,
                            )
                            .await?;
                        let effective_context = self.config.portable_context_ceiling();
                        self.model_registry.publish_effective_context(
                            variant,
                            u64::try_from(effective_context).map_err(|_| {
                                Error::ModelLoadError(
                                    "Parakeet effective context exceeds u64".into(),
                                )
                            })?,
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                if contract.execution_profile.mode
                                    != crate::engine::ExecutionMode::Sequence
                                {
                                    return Err(Error::ModelLoadError(
                                        "Parakeet retained graph has an incompatible execution mode"
                                            .into(),
                                    ));
                                }
                                Ok((graph, RetainedStateUseV2::Inactive))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        if physical_spec.invocation.is_some() {
                            return Err(Error::ModelLoadError(
                                "Parakeet sequence graph unexpectedly published invocation predictor state"
                                    .into(),
                            ));
                        }
                        let publication = self.load_scratch_only_workspace_publication(
                            &contracts,
                            physical_spec.descriptor,
                            Some(retained.into()),
                            retained_uses,
                        )?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route
                        == LoadedAsrStatePublicationRoute::NemotronOffline
                    {
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec =
                            loaded.nemotron_offline_physical_state_spec(&stage_graphs)?;
                        let publication = self
                            .load_invocation_workspace_publication(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                &physical_spec.invocation,
                                None,
                                HashMap::new(),
                            )
                            .await?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if let crate::kv::InferenceStateCapability::Managed(contract) =
                        &loaded_cache
                    {
                        return Err(Error::ModelLoadError(format!(
                            "loaded non-Qwen ASR model {variant} still publishes legacy managed state with {} domains",
                            contract.domains.len()
                        )));
                    }
                }
            }
            if self
                .adapter_registry
                .require(CapabilityKind::RealtimeAsr, variant)
                .is_ok()
            {
                let contracts =
                    bundle_draft.execution_contracts(CapabilityKind::RealtimeAsr)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                if variant.family() == crate::catalog::ModelFamily::Voxtral {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical realtime ASR state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let model = self
                        .model_registry
                        .get_loading_voxtral(variant)
                        .await
                        .ok_or_else(|| {
                            Error::ModelLoadError(format!(
                                "loaded Voxtral realtime ASR model {variant} is missing from the registry"
                            ))
                        })?;
                    let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                    // One planned ASR chunk per sequence, as every other
                    // ASR retained-state fit guarantees.
                    let retained = self
                        .core_engine
                        .load_managed_model_state(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(physical_spec.retained_max_tokens),
                            asr_minimum_tokens(variant),
                            state_fit_reserve,
                        )
                        .await?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            Ok((
                                stage_graph_fingerprint(&contract.stages)?,
                                RetainedStateUseV2::ExternalPaged,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    state_publications.insert(
                        CapabilityKind::RealtimeAsr,
                        LoadedStatePublication::PhysicalV2 {
                            descriptor: physical_spec.descriptor,
                            retained: Some(retained.into()),
                            retained_uses,
                            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                        },
                    );
                } else {
                    let model = self
                        .model_registry
                        .get_loading_asr(variant)
                        .await
                        .ok_or_else(|| {
                            Error::ModelLoadError(format!(
                                "loaded realtime ASR model {variant} is missing from the registry"
                            ))
                        })?;
                    let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                    let retained = self
                        .core_engine
                        .load_retained_tensor_state(
                            model_instance_id,
                            &physical_spec.retained,
                            self.realtime_asr_sequence_capacity,
                        )
                        .await?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            Ok((
                                stage_graph_fingerprint(&contract.stages)?,
                                RetainedStateUseV2::ExternalTensor,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    state_publications.insert(
                        CapabilityKind::RealtimeAsr,
                        LoadedStatePublication::PhysicalV2 {
                            descriptor: physical_spec.descriptor,
                            retained: Some(retained.into()),
                            retained_uses,
                            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                        },
                    );
                }
            }
            if variant.family() == crate::catalog::ModelFamily::SortformerDiarization {
                let model = self
                    .model_registry
                    .get_diarization(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Sortformer model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts =
                    bundle_draft.execution_contracts(CapabilityKind::Diarization)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                let publication = self
                    .load_invocation_workspace_publication(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        None,
                        HashMap::new(),
                    )
                    .await?;
                state_publications.insert(CapabilityKind::Diarization, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical LFM2.5 Audio invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_lfm25_audio_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded LFM2.5 Audio model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [
                    CapabilityKind::Asr,
                    CapabilityKind::Tts,
                    CapabilityKind::AudioChat,
                    CapabilityKind::SpeechToSpeech,
                ] {
                    let remaining_automatic_state_groups = match capability {
                        CapabilityKind::Asr => 5,
                        CapabilityKind::Tts => 3,
                        CapabilityKind::AudioChat => 2,
                        CapabilityKind::SpeechToSpeech => 1,
                        _ => unreachable!("LFM2.5 Audio capability list is closed above"),
                    };
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    if capability == CapabilityKind::Asr {
                        let physical_spec = model.retained_asr_state_spec(&stage_graphs)?;
                        let retained_max_tokens = physical_spec.retained_max_tokens;
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                                2,
                                self.asr_state_rows(),
                                asr_minimum_tokens(variant),
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &retained);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "LFM2.5 Audio ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
                                };
                                Ok((graph, retained_use))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let main_invocation = physical_spec.main_invocation.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "LFM2.5 Audio ASR retained topology is missing long-form main invocation state"
                                    .into(),
                            )
                        })?;
                        let publication = self
                            .load_invocation_workspace_publication_with_remaining_groups(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                main_invocation,
                                Some(retained.into()),
                                retained_uses,
                                remaining_automatic_state_groups,
                            )
                            .await?;
                        state_publications.insert(capability, publication);
                        continue;
                    }
                    if capability == CapabilityKind::Tts {
                        let physical_spec = model.retained_tts_state_spec(&stage_graphs)?;
                        let retained_max_tokens = self
                            .model_registry
                            .effective_context(variant)
                            .map_or(physical_spec.retained_max_tokens, |tokens| {
                                physical_spec.retained_max_tokens.min(tokens)
                            });
                        // The ASR fit above already sized this model's
                        // retained runtime (bound to the same contract), so
                        // this fit asks for no floor of its own.
                        let retained = self
                            .core_engine
                            .load_managed_model_state(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                                0,
                                state_fit_reserve,
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        self.model_registry.publish_kv_storage(variant, &retained);
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Tts,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                Ok((
                                    stage_graph_fingerprint(&contract.stages)?,
                                    RetainedStateUseV2::ExternalPaged,
                                ))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let depthformer = physical_spec.depthformer_invocation.as_ref().ok_or_else(
                            || {
                                Error::ModelLoadError(
                                    "LFM2.5 Audio TTS topology is missing Depthformer invocation state"
                                        .into(),
                                )
                            },
                        )?;
                        let publication = self
                            .load_invocation_workspace_publication_with_remaining_groups(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                depthformer,
                                Some(retained.into()),
                                retained_uses,
                                remaining_automatic_state_groups,
                            )
                            .await?;
                        state_publications.insert(capability, publication);
                        continue;
                    }
                    let mode = if capability == CapabilityKind::Asr {
                        crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainOnly
                    } else {
                        crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainAndDepthformer
                    };
                    let physical_spec = model.physical_state_spec(mode, &stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                            remaining_automatic_state_groups,
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::Qwen3Tts {
                let model = self
                    .model_registry
                    .get_qwen_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Qwen3 TTS model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
                    if self.adapter_registry.require(capability, variant).is_err() {
                        continue;
                    }
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = model.physical_state_spec(&stage_graphs)?;
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical TTS state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let retained_max_tokens = self
                        .model_registry
                        .effective_context(variant)
                        .map_or(physical_spec.retained_max_tokens, |tokens| {
                            physical_spec.retained_max_tokens.min(tokens)
                        });
                    let retained = self
                        .core_engine
                        .load_managed_model_state(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(retained_max_tokens),
                            tts_state_minimum_tokens(variant),
                            state_fit_reserve,
                        )
                        .await?;
                    // Each request reserves the encoder memory it creates
                    // (prefill workspace), so the rows are not also pending.
                    self.core_engine
                        .price_tensor_rows_with_requests(model_instance_id)
                        .await?;
                    self.model_registry.publish_effective_context(
                        variant,
                        retained.logical_token_reach(),
                    )?;
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
                                CacheMode::ExternalPaged
                                    if contract.execution_profile.cache_namespace.is_some()
                                        && contract.execution_profile.kv_dtype != "none" =>
                                {
                                    RetainedStateUseV2::ExternalPaged
                                }
                                CacheMode::None
                                    if contract.execution_profile.cache_namespace.is_none()
                                        && contract.execution_profile.kv_dtype == "none" =>
                                {
                                    RetainedStateUseV2::Inactive
                                }
                                _ => {
                                    return Err(Error::ModelLoadError(
                                        "Qwen3 TTS retained-state graph has an incompatible cache profile"
                                            .to_string(),
                                    ));
                                }
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.predictor_contract,
                            Some(retained.into()),
                            retained_uses,
                            if capability == CapabilityKind::Tts
                                && self
                                    .adapter_registry
                                    .require(CapabilityKind::StreamingTts, variant)
                                    .is_ok()
                            {
                                3
                            } else {
                                1
                            },
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::VibeVoiceTts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_vibevoice_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded VibeVoice TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                    Error::ModelLoadError(
                        "VibeVoice TTS normal graph did not publish retained state".into(),
                    )
                })?;
                let retained_max_tokens = physical_spec.retained_max_tokens;
                let retained = self
                    .core_engine
                    .load_managed_model_state(
                        model_instance_id,
                        retained_contract,
                        retained_max_tokens,
                        tts_state_minimum_tokens(variant),
                        state_fit_reserve,
                    )
                    .await?;
                self.model_registry.publish_effective_context(
                    variant,
                    retained.logical_token_reach(),
                )?;
                self.model_registry.publish_kv_storage(variant, &retained);
                crate::runtime::rollout::validate_managed_state_plan_eligibility(
                    variant,
                    CapabilityKind::Tts,
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
                state_publications.insert(CapabilityKind::Tts, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::FishS2Tts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_fish_s2_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Fish S2 TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let headroom = match self.coordinator.resource_authority().planning_headroom_bytes(backend)? {
                    ResourceAmount::Known(bytes) => Some(bytes),
                    _ => None,
                };
                let native_batching = match crate::env::value("IZWI_FISH_NATIVE_BATCHING") {
                    Some(value) => izwi_env::parse_bool("IZWI_FISH_NATIVE_BATCHING", &value)
                        .map_err(|error| Error::ConfigError(error.to_string()))?,
                    None => false,
                };
                // Automatic staged rows are sized for tensor-width launches. Fish
                // fits its own autoregressive width (no tensor arena backs its
                // transactions), so it reads an automatic limit as automatic.
                let fish_limits = if self.core_engine.config().automatic_row_limits.staged_transactions {
                    std::borrow::Cow::Owned(crate::config::EngineConfig { max_staged_transactions: 0, ..self.config.clone() })
                } else {
                    std::borrow::Cow::Borrowed(&self.config)
                };
                let capacity = crate::runtime::fish_capacity::ResolvedFishServingCapacity::resolve(
                    model.config(), &fish_limits, headroom, native_batching,
                    model.state_element_bytes(),
                )?;
                tracing::info!(?capacity, ?backend, "Sealing Fish serving capacity; device throughput qualification is separate");
                bundle_draft.seal_fish_capacity(capacity.clone())?;
                state_publications.extend(self.load_fish_s2_state_publications(
                    model_instance_id, &bundle_draft,
                    self.config.max_sequence_length.explicit_tokens().unwrap_or(model.config().max_seq_len).min(model.config().max_seq_len),
                    &capacity,
                    |graphs| model.physical_state_spec(graphs),
                ).await?);
            }
            if variant.family() == crate::catalog::ModelFamily::VoxtralTts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_voxtral_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Voxtral TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                // The engine route runs the LM on retained paged K/V (one
                // context per request, shared by prefill and decode).
                let retained = self
                    .core_engine
                    .load_managed_model_state(
                        model_instance_id,
                        &physical_spec.retained,
                        Some(physical_spec.retained_max_tokens),
                        tts_state_minimum_tokens(variant),
                        state_fit_reserve,
                    )
                    .await?;
                self.model_registry.publish_effective_context(
                    variant,
                    retained.logical_token_reach(),
                )?;
                self.model_registry.publish_kv_storage(variant, &retained);
                crate::runtime::rollout::validate_managed_state_plan_eligibility(
                    variant,
                    CapabilityKind::Tts,
                    retained.state_plan_v2(),
                )?;
                let retained_uses = contracts
                    .iter()
                    .map(|contract| {
                        Ok((
                            stage_graph_fingerprint(&contract.stages)?,
                            RetainedStateUseV2::ExternalPaged,
                        ))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                // Stages keep only scratch; no stage owns K/V of its own.
                let publication = self.load_scratch_only_workspace_publication(
                    &contracts,
                    physical_spec.descriptor,
                    Some(retained.into()),
                    retained_uses,
                )?;
                state_publications.insert(CapabilityKind::Tts, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::KokoroTts {
                let model = self
                    .model_registry
                    .get_loading_kokoro(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Kokoro TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let effective_context = kokoro_effective_context_tokens(
                    model.config().context_length(),
                    self.config.portable_context_ceiling(),
                )?;
                self.model_registry
                    .publish_effective_context(variant, effective_context)?;
            }
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
mod tests {
    use super::{
        admit_asr_request_floor, asr_state_rows, automatic_state_group_budget,
        estimate_from_tensor_inventory, fish_s2_resource_plan, is_metal_command_buffer_oom,
        kokoro_effective_context_tokens, loaded_asr_state_publication_route,
        managed_chat_capacity_policy, materialization_tolerance_bytes, model_memory_estimate,
        model_resource_plan, now_unix_millis, plan_invocation_allocations,
        portable_context_reserve_bytes, portable_invocation_context_intent,
        qwen38_representation_memory_estimate, reconcile_observed_growth,
        residency_budget_has_capacity, select_lru_eviction_candidate, state_fit_budgets,
        validate_scratch_only_invocation_publication, DeviceUsageSample,
        LoadedAsrStatePublicationRoute, ModelMemoryEstimate, PortableInvocationContextIntent,
        QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES, QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES,
        QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES,
    };
    use crate::backends::kv::managed_kv_backend_compiled;
    use crate::backends::{BackendKind, BackendPreference};
    use crate::config::{ContextLengthPreference, EngineConfig};
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, CapacitySource, ConcurrencyClass, ExecutionGroupId,
        ExecutionMode, ExecutionProfile, ModelInstanceId, NativeBatchMode,
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass, ReservationOwner,
        ResourceAmount, ResourceAuthority, ResourceLease, ResourceVector, StageDescriptor, StageId,
        StageWorkSelector,
    };
    use crate::error::Error;
    use crate::kv::v2::{
        stage_graph_fingerprint, test_contract, BoundedShape, CapabilityStateDescriptorV2,
        CheckpointPolicy, InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
        InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet,
        PrefixPolicy, RetainedStateCapability, ShapeAxis, ShapeDimension, ShapeExtent,
        StateComponentId, StateDType, StateDomainId, StateDomainSpec, StateScope,
        StaticTensorDomainSpec, TensorComponentSpec, TensorRole, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::model::ModelVariant;
    use crate::models::architectures::fish_s2::weights::FishS2ModelMemory;
    use crate::models::shared::weights::gguf::TensorStorageInventory;
    use crate::runtime::adapters::{
        CapabilityKind, LoadedExecutionContract, RuntimeAdapterRegistry,
    };
    use crate::runtime::lifecycle::controller::{
        ResidentPhase, SharedLoadFailure, SharedLoadOutcome,
    };
    use crate::runtime::service::RuntimeService;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::{Duration, Instant};
    use tokio::sync::{oneshot, Barrier};
    use uuid::Uuid;

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn fish_s2_load_publishes_and_seals_both_tts_capabilities() {
        use crate::models::architectures::fish_s2::{fish_s2_physical_state_spec, FishS2TtsModel};
        use crate::runtime::adapters::LoadedModelBundleDraft;
        use crate::runtime::adapters::StreamingRequirements;
        // Serialize against OOM-ladder poison windows and load-transaction
        // reservations on the process-global resource authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let directory =
            std::env::temp_dir().join(format!("izwi-fish-state-publication-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: directory.clone(),
            backend: BackendPreference::Cpu,
            max_sequence_length: ContextLengthPreference::explicit(32).unwrap(),
            max_retained_sequences: 2,
            max_staged_transactions: 2,
            ..Default::default()
        })
        .unwrap();
        let variant = ModelVariant::FishAudioS2Pro;
        let authority = vector_authority(all_memory_capacity(1024));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-load-publication"),
                ResourceVector::zero(),
            )
            .unwrap();
        let instance = runtime
            .model_lifecycle
            .install_loading_slot(variant, lease)
            .unwrap();
        let registry = RuntimeAdapterRegistry::built_in();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(991),
            instance,
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        // Use the model-authored physical contract with small neural geometry;
        // no weights or inference mocks are needed to exercise publication.
        let mut config = FishS2TtsModel::for_test().config().clone();
        config.text_config.num_hidden_layers = 1;
        config.text_config.hidden_size = 128;
        config.text_config.num_attention_heads = 1;
        config.text_config.num_key_value_heads = 1;
        config.text_config.head_dim = Some(128);
        config.audio_decoder_config.num_hidden_layers = 1;
        config.audio_decoder_config.hidden_size = 128;
        config.audio_decoder_config.num_attention_heads = 1;
        config.audio_decoder_config.num_key_value_heads = 1;
        config.audio_decoder_config.head_dim = Some(128);
        let mut capacity =
            crate::runtime::fish_capacity::ResolvedFishServingCapacity::scalar().unwrap();
        capacity.ar_rows = 2;
        capacity.codec_rows = 2;
        capacity.active_rows = 2;
        capacity.staged_rows = 2;
        capacity.ar_workspace_per_row = 8192;
        capacity.prefill_workspace_per_token = 8192;
        capacity.codec_workspace_per_row = 8192;
        capacity.native_batching = true;
        draft.seal_fish_capacity(capacity.clone()).unwrap();
        let publications = runtime
            .model_lifecycle
            .load_fish_s2_state_publications(instance, &draft, 32, &capacity, |graphs| {
                fish_s2_physical_state_spec(&config, candle_core::DType::F32, graphs)
            })
            .await
            .unwrap();
        assert_eq!(publications.len(), 2);
        let bundle = draft
            .seal(publications)
            .expect("all advertised capabilities must seal");
        let mut bindings = Vec::new();
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let binding = bundle
                .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
                .unwrap();
            assert_eq!(binding.execution.model_instance_id, instance);
            assert_eq!(
                binding
                    .execution
                    .stages
                    .iter()
                    .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
                    .unwrap()
                    .max_batch_size,
                2
            );
            assert!(binding.state.managed_kv_runtime().is_some());
            assert!(binding
                .execution
                .stages
                .iter()
                .any(|stage| stage.selector == StageWorkSelector::SequenceAudioDecode));
            bindings.push(binding);
        }
        assert_ne!(
            bindings[0].execution.adapter_instance_id,
            bindings[1].execution.adapter_instance_id
        );
        assert!(Arc::ptr_eq(
            &bindings[0].state.managed_kv_runtime().unwrap(),
            &bindings[1].state.managed_kv_runtime().unwrap()
        ));
        drop(bindings);
        drop(bundle);
        drop(runtime);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn invocation_execution(max_batch_size: usize) -> LoadedExecutionContract {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let metadata = *RuntimeAdapterRegistry::built_in()
            .require(CapabilityKind::Tts, variant)
            .unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.max_batch_size = max_batch_size;
        profile.concurrency = if max_batch_size > 1 {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "test.atomic.scalar",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.max_workspace_bytes = 0;
        stage.validate().unwrap();
        LoadedExecutionContract {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(4),
            metadata,
            execution_profile: profile,
            stages: Arc::from([stage]),
        }
    }

    #[test]
    fn portable_tensor_inventory_replaces_catalog_guess_and_accounts_load_overlap() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 12 * 1024 * 1024 * 1024,
            resident_bytes: 12 * 1024 * 1024 * 1024,
        };
        // F32-shaped largest tensor: 160 MB storage over 40M elements.
        let estimate = estimate_from_tensor_inventory(
            catalog,
            Some(TensorStorageInventory {
                total_bytes: 2_400_000_000,
                largest_tensor_bytes: 160_000_000,
                largest_tensor_elements: 40_000_000,
                tensor_count: 400,
            }),
        )
        .unwrap();
        assert_eq!(estimate.resident_bytes, 2_400_000_000);
        // Scratch = max(next_pow2(160M) = 268,435,456; 160M + 40M*4 = 320M;
        // 400 * 32 KiB = 13,107,200) = 320,000,000: the F32 conversion bound
        // of the largest tensor exceeds the historic size-class proxy.
        assert_eq!(estimate.load_peak_bytes, 2_720_000_000);
        assert_eq!(
            estimate_from_tensor_inventory(catalog, None).unwrap(),
            catalog
        );
        assert_eq!(
            estimate_from_tensor_inventory(catalog, Some(TensorStorageInventory::default()),)
                .unwrap(),
            catalog
        );
    }

    /// The header census lists every tensor with its stored dtype, shape and
    /// payload bytes, and the inventory folds the same records.
    #[test]
    fn checkpoint_census_records_every_tensor_header() {
        use super::{checkpoint_census, checkpoint_tensor_inventory, StoredDType};
        let dir = std::env::temp_dir().join(format!("izwi-census-{}", Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("shards")).unwrap();
        let device = candle_core::Device::Cpu;
        let zeros = |shape: &[usize], dtype| candle_core::Tensor::zeros(shape, dtype, &device).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([
                ("embed".to_string(), zeros(&[300, 16], candle_core::DType::BF16)),
                ("norm".to_string(), zeros(&[16], candle_core::DType::F32)),
            ]),
            dir.join("shards/model.safetensors"),
        )
        .unwrap();
        std::fs::write(dir.join("model.nemo"), vec![0u8; 4096]).unwrap();

        let census = checkpoint_census(&dir).unwrap();
        assert_eq!(census.files.len(), 1);
        assert_eq!(census.container_fallback_bytes, 4096);
        let mut tensors = census.files[0]
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.dtype, t.shape.clone(), t.stored_bytes))
            .collect::<Vec<_>>();
        tensors.sort_by_key(|tensor| tensor.0);
        assert_eq!(
            tensors,
            [
                ("embed", StoredDType::Dense(candle_core::DType::BF16), vec![300, 16], 9600),
                ("norm", StoredDType::Dense(candle_core::DType::F32), vec![16], 64),
            ]
        );
        assert_eq!(
            checkpoint_tensor_inventory(&dir).unwrap(),
            Some(TensorStorageInventory {
                total_bytes: 9664,
                largest_tensor_bytes: 9600,
                largest_tensor_elements: 4800,
                tensor_count: 2,
            })
        );
        // Without a readable checkpoint the container size stands in.
        std::fs::remove_dir_all(dir.join("shards")).unwrap();
        assert_eq!(
            checkpoint_tensor_inventory(&dir).unwrap(),
            Some(TensorStorageInventory {
                total_bytes: 4096,
                largest_tensor_bytes: 4096,
                largest_tensor_elements: 0,
                tensor_count: 1,
            })
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn moe_shaped_inventory_reserves_whole_checkpoint_instantiation_scratch() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 40 * 1024 * 1024 * 1024,
            resident_bytes: 40 * 1024 * 1024 * 1024,
        };
        // MoE shape: tens of thousands of same-sized expert tensors collapse
        // the largest-tensor terms; only the allocation-count term keeps the
        // instantiation scratch reserved.
        let inventory = TensorStorageInventory {
            total_bytes: 30_000_000_000,
            largest_tensor_bytes: 20_971_520,
            largest_tensor_elements: 5_242_880,
            tensor_count: 18_432,
        };
        let estimate = estimate_from_tensor_inventory(catalog, Some(inventory)).unwrap();
        assert_eq!(estimate.resident_bytes, 30_000_000_000);
        assert_eq!(estimate.load_peak_bytes, 30_603_979_776);
        // The historic formula would have reserved only the size-class proxy
        // of one expert tensor (30,033,554,432) — under-reserving by the
        // whole-checkpoint instantiation term the fix exists for.
        let historic_load_peak = inventory.total_bytes
            + inventory
                .largest_tensor_bytes
                .checked_next_power_of_two()
                .unwrap();
        assert_eq!(historic_load_peak, 30_033_554_432);
        assert!(estimate.load_peak_bytes > historic_load_peak);
    }

    #[test]
    fn quantized_largest_tensor_reserves_f32_conversion_scratch() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 1024 * 1024 * 1024,
            resident_bytes: 1024 * 1024 * 1024,
        };
        // Q4_0-shaped largest tensor: 1M elements over 562.5 KB of storage.
        // A dequantizing loader holds the source while building the F32
        // destination (4 MB), which the size-class proxy under-covers.
        let estimate = estimate_from_tensor_inventory(
            catalog,
            Some(TensorStorageInventory {
                total_bytes: 5_000_000,
                largest_tensor_bytes: 562_500,
                largest_tensor_elements: 1_000_000,
                tensor_count: 10,
            }),
        )
        .unwrap();
        assert_eq!(estimate.load_peak_bytes, 9_562_500);
    }

    /// Element census of the pinned Qwen3.8-27B checkpoint by resident class,
    /// computed independently of the residency policy's tensor list.
    mod qwen38_27b {
        /// Block-FP8 projections: 48 DeltaNet layers x 382,730,240, 16
        /// attention layers x 372,244,480, and the MTP layer's 372,244,480.
        pub const TRUNK_FP8: u64 = 48 * 382_730_240 + 16 * 372_244_480;
        pub const MTP_FP8: u64 = 372_244_480;
        /// Embedding and LM head (248,320 x 5,120 each).
        pub const EMBEDDING: u64 = 1_271_398_400;
        pub const LM_HEAD: u64 = 1_271_398_400;
        /// DeltaNet a/b projections, kept in the activation dtype.
        pub const TRUNK_DENSE: u64 = 48 * 2 * 48 * 5_120;
        /// Norm gains (F32 since C2) and DeltaNet math/conv (F32).
        pub const TRUNK_F32: u64 =
            5_120 + 64 * 2 * 5_120 + 16 * 2 * 256 + 48 * (128 + 48 + 48 + 10_240 * 4);
        pub const MTP_DENSE: u64 = 5_120 * 10_240;
        pub const MTP_F32: u64 = 5 * 5_120 + 2 * 256;
        /// F32 block scales of raw block FP8 (one per 128x128 block).
        pub const TRUNK_SCALES: u64 = 48 * 23_360 + 16 * 22_720;
        pub const MTP_SCALES: u64 = 22_720;
        /// The source checkpoint's BF16 elements also include a 460,730,096
        /// element vision tower the text runtime never loads.
        pub const VISION: u64 = 460_730_096;
    }

    fn qwen38_resource_plan(backend: BackendKind) -> super::ModelResourcePlan {
        super::qwen38_memory::static_resource_plan(
            backend,
            &crate::performance::PerformanceConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn qwen38_resource_plan_prices_backend_resident_representations() {
        use qwen38_27b::*;
        let default = crate::performance::PerformanceConfig::default();
        let estimate = |backend| qwen38_representation_memory_estimate(backend, &default);
        let (cpu, metal, cuda) = (
            estimate(BackendKind::Cpu),
            estimate(BackendKind::Metal),
            estimate(BackendKind::Cuda),
        );
        let fp8 = TRUNK_FP8 + MTP_FP8;
        assert_eq!(fp8, 24_699_207_680);
        let dense = EMBEDDING + LM_HEAD + TRUNK_DENSE + MTP_DENSE;
        let f32_bytes = (TRUNK_F32 + MTP_F32) * 4;
        // The old static figure priced every BF16 source element (vision
        // included) at two bytes, and the F32 gains at two as well.
        assert_eq!(
            dense + TRUNK_F32 + MTP_F32 + VISION,
            3_082_220_272,
            "census accounts for every BF16 checkpoint element"
        );

        // The CPU keeps projections and the LM head as Q8_0 (34 bytes per 32
        // elements) and the rest F32; `expanded` expands everything to F32.
        assert_eq!(
            cpu.resident_bytes,
            (fp8 + LM_HEAD) / 32 * 34 + (dense - LM_HEAD) * 4 + f32_bytes
        );
        assert_eq!(cpu.resident_bytes, 32_994_136_064);
        let mut cpu_expanded = crate::performance::PerformanceConfig::default();
        cpu_expanded.cpu.projection_backend = crate::performance::CpuProjectionBackend::Expanded;
        assert_eq!(
            qwen38_representation_memory_estimate(BackendKind::Cpu, &cpu_expanded).resident_bytes,
            109_282_791_424
        );
        assert_eq!((fp8 + dense) * 4 + f32_bytes, 109_282_791_424);
        // Metal keeps raw block FP8 (bytes + F32 block scales) with F16 dense
        // tensors and the 1 GiB F16 prefill scratch; a build without the
        // Metal kernels expands to F16.
        let scratch =
            crate::models::architectures::qwen38::residency::METAL_FP8_PREFILL_DEQUANT_BYTES;
        let metal_native = fp8 + (TRUNK_SCALES + MTP_SCALES) * 4 + dense * 2 + f32_bytes + scratch;
        let metal_expanded = (fp8 + dense) * 2 + f32_bytes;
        if cfg!(feature = "metal") {
            assert_eq!(metal.resident_bytes, metal_native);
            assert_eq!(metal.resident_bytes, 31_027_303_168);
            // Fits a 48 GB Mac's 32.4 GiB budget (with a little context);
            // a 32 GB Mac's 21.6 GiB budget refuses it.
            const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
            assert!((metal.load_peak_bytes as f64) < 32.4 * GIB);
            assert!((metal.load_peak_bytes as f64) > 21.6 * GIB);
            let plan = qwen38_resource_plan(BackendKind::Metal);
            assert_eq!(
                plan.deferred_resident_authorization.unified_bytes,
                ResourceAmount::Known(scratch)
            );
        } else {
            assert_eq!(metal.resident_bytes, metal_expanded);
        }
        assert_eq!(metal_expanded, 54_646_738_944);
        // CUDA's static arm prices Q8_0 (34 bytes per 32 elements), plus the
        // F32 rows the DeltaNet alpha/beta router GEMV keeps beside the
        // dense pair.
        assert_eq!(
            cuda.resident_bytes,
            fp8 / 32 * 34 + dense * 2 + f32_bytes + TRUNK_DENSE * 4
        );
        assert_eq!(cuda.resident_bytes, 31_585_603_584);
        assert_eq!(
            cuda.load_peak_bytes,
            cuda.resident_bytes + QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES
        );
        assert_eq!(
            cpu.load_peak_bytes - cpu.resident_bytes,
            QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES
        );
        assert_eq!(
            metal.load_peak_bytes - metal.resident_bytes,
            QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES
        );
        let cuda_plan = qwen38_resource_plan(BackendKind::Cuda);
        assert_eq!(
            cuda_plan.load_authorization.host_bytes,
            ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES)
        );
        assert_eq!(
            cuda_plan.load_authorization.device_bytes,
            ResourceAmount::Known(cuda.load_peak_bytes)
        );
        assert_eq!(
            cuda_plan.resident_authorization.device_bytes,
            ResourceAmount::Known(cuda.resident_bytes)
        );

        // Admission follows the loader's MTP rule: without the head its
        // projections, fc and norms are not charged.
        let mut no_mtp = crate::performance::PerformanceConfig::default();
        no_mtp.cuda.mtp = crate::performance::OptimizationMode::Off;
        let mtp_metal = if cfg!(feature = "metal") {
            MTP_FP8 + MTP_SCALES * 4 + MTP_DENSE * 2 + MTP_F32 * 4
        } else {
            (MTP_FP8 + MTP_DENSE) * 2 + MTP_F32 * 4
        };
        assert_eq!(
            qwen38_representation_memory_estimate(BackendKind::Metal, &no_mtp).resident_bytes,
            metal.resident_bytes - mtp_metal
        );
        // `metal.projection_backend = "expanded"` prices the F16 expansion.
        let mut expanded = crate::performance::PerformanceConfig::default();
        expanded.metal.projection_backend = crate::performance::MetalProjectionBackend::Expanded;
        assert_eq!(
            qwen38_representation_memory_estimate(BackendKind::Metal, &expanded).resident_bytes,
            metal_expanded
        );
        // Off CUDA the CUDA master switch does not govern MTP; on CUDA it does.
        let mut cuda_off = crate::performance::PerformanceConfig::default();
        cuda_off.cuda.mode = crate::performance::OptimizationMode::Off;
        assert_eq!(
            qwen38_representation_memory_estimate(BackendKind::Metal, &cuda_off),
            metal
        );
        // The master switch also turns the alpha/beta GEMV (and its rows) off.
        assert_eq!(
            qwen38_representation_memory_estimate(BackendKind::Cuda, &cuda_off).resident_bytes,
            cuda.resident_bytes - MTP_FP8 / 32 * 34 - MTP_DENSE * 2 - MTP_F32 * 4 - TRUNK_DENSE * 4
        );
    }

    #[test]
    fn qwen38_uses_engine_staged_width_while_cuda_retains_context_fitting() {
        use crate::engine::CudaResidentPool;
        let cuda_qwen38 =
            managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Cuda);
        assert_eq!(cuda_qwen38.staged_transaction_rows, None);
        assert_eq!(cuda_qwen38.cuda_resident_pool, CudaResidentPool::RetainedRows);

        let metal_qwen38 =
            managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Metal);
        assert_eq!(metal_qwen38.staged_transaction_rows, None);
        assert_eq!(metal_qwen38.cuda_resident_pool, CudaResidentPool::Disabled);

        let cuda_qwen35 =
            managed_chat_capacity_policy(ModelVariant::Qwen359BGguf, BackendKind::Cuda);
        assert_eq!(cuda_qwen35.staged_transaction_rows, None);
        assert_eq!(cuda_qwen35.cuda_resident_pool, CudaResidentPool::Disabled);

        // The Qwen3.5/3.6 MoE family fits its invocation-state arena against
        // CUDA resident headroom like Qwen3.8: its authored-context arena
        // otherwise cannot coexist with the resident representation on an
        // 80 GB device.
        let cuda_qwen36_moe =
            managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Cuda);
        assert_eq!(cuda_qwen36_moe.staged_transaction_rows, None);
        assert_eq!(cuda_qwen36_moe.cuda_resident_pool, CudaResidentPool::RetainedRows);

        let metal_qwen36_moe =
            managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Metal);
        assert_eq!(metal_qwen36_moe.cuda_resident_pool, CudaResidentPool::Disabled);

        // LFM2.5 chat fits one materialized context on CUDA and keeps the
        // portable fit elsewhere; families without a resident fit stay off.
        for variant in [
            ModelVariant::Lfm2512BInstructGguf,
            ModelVariant::Lfm2512BThinkingGguf,
        ] {
            assert_eq!(
                managed_chat_capacity_policy(variant, BackendKind::Cuda).cuda_resident_pool,
                CudaResidentPool::OneContext
            );
            for backend in [BackendKind::Cpu, BackendKind::Metal] {
                assert_eq!(
                    managed_chat_capacity_policy(variant, backend).cuda_resident_pool,
                    CudaResidentPool::Disabled
                );
            }
        }
    }

    #[test]
    fn qwen36_config_state_bytes_match_its_contract() {
        use super::{hybrid_recurrent_state_bytes_per_sequence, qwen36moe_state_geometry};
        use crate::models::architectures::qwen36moe::{
            cache::qwen35_composite_cache_contract_with_mtp, native::pinned_native_config,
            native_model::qwen35_text_config_from_native,
        };
        let native = pinned_native_config();
        let from_config =
            hybrid_recurrent_state_bytes_per_sequence(qwen36moe_state_geometry(&native.text))
                .unwrap();
        // The bytes the managed planner authorizes per row for the real
        // contract builder, so the pre-load formula cannot drift from it.
        let contract = qwen35_composite_cache_contract_with_mtp(
            &qwen35_text_config_from_native(&native.text),
            candle_core::DType::BF16,
            64,
            false,
            false,
        )
        .unwrap();
        let plan = crate::backends::state::negotiate_state_plan(
            &contract,
            &crate::backends::state::StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(64),
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let (_, tensor) = crate::engine::plan_managed_state_capacity(
            &plan,
            ModelInstanceId::new(36),
            crate::engine::ManagedStateCapacityRequest {
                total_paged_pages: 64,
                logical_token_reach: None,
                retained_sequence_rows: 1,
                staged_transaction_rows: 1,
            },
        )
        .unwrap();
        assert_eq!(from_config, tensor.unwrap().per_sequence_bytes());
        assert_eq!(from_config, 65_863_680);
    }

    #[test]
    fn cuda_state_floor_refuses_an_unfittable_plan_before_weights() {
        use super::{admit_cuda_state_floor, CudaStateFloor};
        const GIB: u64 = 1024 * 1024 * 1024;
        const H100_BYTES: u64 = 85_017_624_576;
        const QWEN38_ROW_BYTES: u64 = 156_893_184;
        let plan = qwen38_resource_plan(BackendKind::Cuda);
        let ResourceAmount::Known(weights) = plan.resident_authorization.device_bytes else {
            panic!("Qwen3.8 CUDA residency must be known");
        };
        let floor = |rows, automatic_rows| CudaStateFloor {
            retained_rows: rows,
            staged_rows: rows.min(8),
            snapshot_rows: 0,
            per_sequence_bytes: QWEN38_ROW_BYTES,
            automatic_rows,
        };
        let admit = |floor, capacity| {
            admit_cuda_state_floor(
                ModelVariant::Qwen3827BFp8,
                &plan,
                floor,
                GIB,
                ResourceAmount::Known(capacity),
            )
        };

        // The decode-width floor fits an H100 and joins the load authorization.
        let authorization = admit(floor(8, true), H100_BYTES).unwrap();
        assert_eq!(
            authorization.device_bytes,
            ResourceAmount::Known(weights + 16 * QWEN38_ROW_BYTES + GIB)
        );
        assert_eq!(authorization.host_bytes, plan.load_authorization.host_bytes);

        // The pre-fix automatic rows, now explicit, can never fit.
        let legacy = CudaStateFloor {
            retained_rows: 316,
            staged_rows: 316,
            snapshot_rows: 0,
            per_sequence_bytes: QWEN38_ROW_BYTES,
            automatic_rows: false,
        };
        let message = admit(legacy, H100_BYTES).unwrap_err().to_string();
        assert!(message.contains("before loading Qwen3.8"), "{message}");
        assert!(
            message.contains("minimum_state_bytes=99156492288"),
            "{message}"
        );
        assert!(message.contains("IZWI_MAX_RETAINED_SEQUENCES"), "{message}");

        // Automatic rows on a GPU smaller than the weights name the staged knob.
        let message = admit(floor(8, true), 24 * GIB).unwrap_err().to_string();
        assert!(
            message.contains("IZWI_MAX_STAGED_TRANSACTIONS"),
            "{message}"
        );
        assert!(
            message.contains("per_sequence_state_bytes=156893184"),
            "{message}"
        );

        // Unknown capacity defers to live admission, still pricing the floor.
        let authorization = admit_cuda_state_floor(
            ModelVariant::Qwen3827BFp8,
            &plan,
            floor(8, true),
            GIB,
            ResourceAmount::Unknown,
        )
        .unwrap();
        assert_eq!(
            authorization.device_bytes,
            ResourceAmount::Known(weights + 16 * QWEN38_ROW_BYTES + GIB)
        );

        // With prefix reuse engaged, the floor also holds the snapshot budget.
        let shared = CudaStateFloor {
            snapshot_rows: 8,
            ..floor(8, true)
        };
        let authorization = admit(shared, H100_BYTES).unwrap();
        assert_eq!(
            authorization.device_bytes,
            ResourceAmount::Known(weights + 24 * QWEN38_ROW_BYTES + GIB)
        );
    }

    #[test]
    fn qwen38_cuda_q8_0_plan_fits_reported_l40s_available_bytes() {
        const REPORTED_L40S_AVAILABLE_BYTES: u64 = 47_196_667_904;

        let plan = qwen38_resource_plan(BackendKind::Cuda);
        let ResourceAmount::Known(device_load_peak_bytes) = plan.load_authorization.device_bytes
        else {
            panic!("Qwen3.8 CUDA device load peak must be known")
        };
        // Every FP8 and BF16 source element at two bytes, plus the portable
        // conversion scratch: the old fully expanded CUDA plan.
        let old_expanded_load_peak_bytes =
            (24_699_207_680 + 3_082_220_272) * 2 + QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES;
        assert_eq!(
            plan.load_authorization.host_bytes,
            ResourceAmount::Known(8_589_934_592)
        );
        assert_eq!(device_load_peak_bytes, 31_854_039_040);
        assert_eq!(old_expanded_load_peak_bytes, 56_636_597_728);
        assert!(device_load_peak_bytes < REPORTED_L40S_AVAILABLE_BYTES);
        assert!(old_expanded_load_peak_bytes > REPORTED_L40S_AVAILABLE_BYTES);

        let l40s_capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES),
            device_bytes: ResourceAmount::Known(REPORTED_L40S_AVAILABLE_BYTES),
            ..ResourceVector::zero()
        };
        let authority = vector_authority(l40s_capacity);
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-q8_0"),
                plan.load_authorization,
            )
            .expect("Q8_0 CUDA representation should fit reported L40S headroom");
        drop(lease);

        let old_expanded = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES),
            device_bytes: ResourceAmount::Known(old_expanded_load_peak_bytes),
            ..ResourceVector::zero()
        };
        let error = vector_authority(l40s_capacity)
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-expanded"),
                old_expanded,
            )
            .expect_err("the old expanded CUDA peak must not fit reported L40S headroom");
        assert!(matches!(error, Error::Overloaded(_)));

        let insufficient_host = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES - 1),
            device_bytes: ResourceAmount::Known(REPORTED_L40S_AVAILABLE_BYTES),
            ..ResourceVector::zero()
        };
        let error = vector_authority(insufficient_host)
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-host-scratch"),
                plan.load_authorization,
            )
            .expect_err("CUDA load must reserve the complete 8 GiB host conversion window");
        assert!(matches!(error, Error::Overloaded(_)));
    }

    #[test]
    fn metal_oom_retry_judges_hybrid_checkpoints_by_their_backend_load_peak() {
        use super::retry_load_peak_bytes;
        const GIB: u64 = 1024 * 1024 * 1024;
        let _env_guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let performance = crate::performance::PerformanceConfig::default();
        let peak = |variant, backend| retry_load_peak_bytes(variant, backend, &performance, None);

        // Qwen3.8 on Metal (raw block FP8, about 29.9 GiB at load; 51.9 GiB
        // expanded to F16) sits far below the 80 GiB catalog figure that used
        // to veto every retry on a 96 GB Mac.
        let qwen38_metal = peak(ModelVariant::Qwen3827BFp8, BackendKind::Metal);
        assert_eq!(
            qwen38_metal,
            qwen38_representation_memory_estimate(BackendKind::Metal, &performance).load_peak_bytes
        );
        assert!(qwen38_metal < 54 * GIB);
        assert!(peak(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Metal) < 140 * GIB);
        // Variants without a representation plan keep the catalog estimate.
        assert_eq!(
            peak(ModelVariant::Kokoro82M, BackendKind::Metal),
            model_memory_estimate(ModelVariant::Kokoro82M).load_peak_bytes
        );
    }

    #[test]
    fn metal_command_buffer_oom_is_backend_fatal_but_other_errors_are_not() {
        assert!(is_metal_command_buffer_oom(&Error::InferenceError(
            "Metal error Command buffer had following error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".into(),
        )));
        assert!(!is_metal_command_buffer_oom(&Error::ModelLoadError(
            "portable context minimum does not fit".into(),
        )));
    }

    fn invocation_contract(domain_count: u32) -> crate::kv::v2::InferenceStateContract {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(first) = &mut contract.domains[0] else {
            unreachable!()
        };
        first.header.scope = StateScope::Invocation;
        first.header.prefix = PrefixPolicy::Disabled;
        first.header.checkpoint = CheckpointPolicy::None;
        first.accepted_dtypes = vec![StateDType::F32];
        contract.groups[0].prefix_shareable = false;
        for id in 2..=domain_count {
            let mut state = contract.domains[0].clone();
            let StateDomainSpec::PagedAttention(domain) = &mut state else {
                unreachable!()
            };
            domain.header.id = StateDomainId::new(id);
            contract.domains.push(state);
        }
        contract.groups[0].domains = (1..=domain_count).map(StateDomainId::new).collect();
        contract.validate().unwrap();
        contract
    }

    fn invocation_descriptor(
        execution: &LoadedExecutionContract,
        contract: &crate::kv::v2::InferenceStateContract,
        lease_scope: InvocationLeaseScope,
    ) -> CapabilityStateDescriptorV2 {
        let domains = contract
            .domains
            .iter()
            .map(|state| InvocationWorkspaceDomain::State {
                state: state.clone(),
                capacity: if matches!(state, StateDomainSpec::PagedAttention(_)) {
                    InvocationStateCapacity::PagedTokens { max_tokens: 16 }
                } else {
                    InvocationStateCapacity::SemanticBounded
                },
                placement: state.header().placement,
                formula: WorkspaceFormula {
                    fixed_bytes: 1024 * 1024,
                    dimensions: vec![],
                    terms: vec![],
                },
            })
            .collect();
        CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: stage_graph_fingerprint(&execution.stages).unwrap(),
                    stages: vec![InvocationStageWorkspace {
                        stage: execution.stages[0].id,
                        lease_scope,
                        groups: contract.groups.clone(),
                        domains,
                    }],
                }],
            },
        }
    }

    fn mixed_invocation_contract() -> crate::kv::v2::InferenceStateContract {
        let mut contract = invocation_contract(1);
        let mut header = contract.domains[0].header().clone();
        header.id = StateDomainId::new(2);
        contract
            .domains
            .push(StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                header,
                components: vec![TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::EncoderMemory,
                    shape: BoundedShape {
                        dimensions: vec![
                            ShapeDimension {
                                axis: ShapeAxis::Sequence,
                                extent: ShapeExtent::RuntimeBounded { min: 1, max: 16 },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Hidden,
                                extent: ShapeExtent::Fixed { value: 8 },
                            },
                        ],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                }],
            }));
        contract.groups[0].domains.push(StateDomainId::new(2));
        contract.validate().unwrap();
        contract
    }

    #[test]
    fn managed_capability_cache_truth_tracks_compiled_direct_kernels() {
        assert!(managed_kv_backend_compiled(BackendKind::Cpu));
        assert_eq!(
            managed_kv_backend_compiled(BackendKind::Metal),
            cfg!(feature = "metal")
        );
        assert_eq!(
            managed_kv_backend_compiled(BackendKind::Cuda),
            cfg!(feature = "cuda")
        );
    }

    #[test]
    fn whisper_asr_selects_physical_invocation_publication() {
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::WhisperLargeV3Turbo),
            LoadedAsrStatePublicationRoute::Whisper
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::VibeVoiceAsr),
            LoadedAsrStatePublicationRoute::VibeVoice
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::ParakeetTdt06BV3),
            LoadedAsrStatePublicationRoute::Parakeet
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::Nemotron35AsrStreaming06B),
            LoadedAsrStatePublicationRoute::NemotronOffline
        );
    }

    #[test]
    fn generic_invocation_planner_allocates_every_domain_at_exact_row_concurrency() {
        let execution = invocation_execution(3);
        let contract = invocation_contract(2);
        let descriptor = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);

        let allocations =
            plan_invocation_allocations(&descriptor, &contract, &[execution]).unwrap();
        assert_eq!(allocations.len(), 2);
        assert!(allocations
            .iter()
            .all(|allocation| allocation.slot_count == 3));
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.key.domain)
                .collect::<Vec<_>>(),
            vec![StateDomainId::new(1), StateDomainId::new(2)]
        );
    }

    #[test]
    fn generic_invocation_planner_accepts_mixed_paged_and_semantic_domains() {
        let execution = invocation_execution(2);
        let contract = mixed_invocation_contract();
        let descriptor = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);

        let allocations =
            plan_invocation_allocations(&descriptor, &contract, &[execution]).unwrap();
        assert_eq!(allocations.len(), 2);
        assert!(allocations
            .iter()
            .all(|allocation| allocation.slot_count == 2));
        assert!(matches!(
            allocations[0].domain,
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::PagedAttention(_),
                capacity: InvocationStateCapacity::PagedTokens { .. },
                ..
            }
        ));
        assert!(matches!(
            allocations[1].domain,
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::StaticTensor(_),
                capacity: InvocationStateCapacity::SemanticBounded,
                ..
            }
        ));
    }

    #[test]
    fn generic_invocation_planner_rejects_missing_extra_and_foreign_graph_mappings() {
        let execution = invocation_execution(1);
        let contract = invocation_contract(2);
        let descriptor =
            invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerStageBatch);

        let mut missing_contract = contract.clone();
        missing_contract.domains.pop();
        missing_contract.groups[0].domains.pop();
        missing_contract.validate().unwrap();
        assert!(plan_invocation_allocations(
            &descriptor,
            &missing_contract,
            std::slice::from_ref(&execution),
        )
        .is_err());

        let mut missing_descriptor = descriptor.clone();
        let InvocationWorkspaceSet::Bounded { profiles } = &mut missing_descriptor.invocation
        else {
            unreachable!()
        };
        profiles[0].stages[0].domains.pop();
        profiles[0].stages[0].groups[0].domains.pop();
        assert!(plan_invocation_allocations(
            &missing_descriptor,
            &contract,
            std::slice::from_ref(&execution),
        )
        .is_err());

        let foreign_execution = invocation_execution(2);
        assert!(plan_invocation_allocations(
            &descriptor,
            &contract,
            &[execution, foreign_execution]
        )
        .is_err());
    }

    #[test]
    fn scratch_only_publication_accepts_stage_scratch_and_rejects_typed_state() {
        let mut execution = invocation_execution(1);
        let mut stage = execution.stages[0].clone();
        stage.max_workspace_bytes = 4096;
        stage.validate().unwrap();
        execution.stages = Arc::from([stage]);
        let descriptor =
            CapabilityStateDescriptorV2::stateless_for_stage_graphs(&[execution.stages.as_ref()])
                .unwrap();
        validate_scratch_only_invocation_publication(&descriptor, std::slice::from_ref(&execution))
            .unwrap();

        let contract = invocation_contract(1);
        let typed = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);
        let error =
            validate_scratch_only_invocation_publication(&typed, std::slice::from_ref(&execution))
                .unwrap_err();
        assert!(error.to_string().contains("contains typed state"));
    }

    fn one_byte_host_reservation() -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            ..ResourceVector::zero()
        }
    }

    #[derive(Debug)]
    struct TestCapacityProvider;

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let capacity = ResourceVector {
                host_bytes: ResourceAmount::Known(1024),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            }
        }
    }

    #[derive(Debug)]
    struct VectorCapacityProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for VectorCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    fn all_memory_capacity(bytes: u64) -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(bytes),
            device_bytes: ResourceAmount::Known(bytes),
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        }
    }

    fn vector_authority(capacity: ResourceVector) -> Arc<ResourceAuthority> {
        Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            },
        })))
    }

    fn isolated_resource_lease(key: &str) -> (Arc<ResourceAuthority>, ResourceLease) {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, key),
                one_byte_host_reservation(),
            )
            .expect("test resource reservation");
        (authority, lease)
    }

    #[test]
    fn select_lru_eviction_candidate_skips_requested_and_active_models() {
        let resident_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen359BGguf,
            ModelVariant::Kokoro82M,
        ];
        let requested_variant = ModelVariant::Kokoro82M;
        let active_variants = HashSet::from([ModelVariant::Qwen359BGguf]);
        let pinned_variants = HashSet::new();
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen359BGguf, 5_u64),
            (ModelVariant::Kokoro82M, 20_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &resident_variants,
            requested_variant,
            &active_variants,
            &pinned_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));
    }

    #[test]
    fn select_lru_eviction_candidate_skips_pinned_models() {
        let resident_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen359BGguf,
        ];
        let requested_variant = ModelVariant::Kokoro82M;
        let active_variants = HashSet::new();
        let pinned_variants = HashSet::from([ModelVariant::Qwen359BGguf]);
        // The pinned model is older than the unpinned one but must not be
        // selected.
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen359BGguf, 5_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &resident_variants,
            requested_variant,
            &active_variants,
            &pinned_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));

        // When every resident is pinned there is no candidate.
        let all_pinned = HashSet::from([
            ModelVariant::Qwen359BGguf,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        ]);
        assert_eq!(
            select_lru_eviction_candidate(
                &resident_variants,
                requested_variant,
                &active_variants,
                &all_pinned,
                &last_used,
            ),
            None
        );
    }

    #[test]
    fn residency_budget_requires_space_before_loading_a_replacement() {
        let resident_variants = vec![ModelVariant::Kokoro82M];

        assert!(!residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen359BGguf,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Kokoro82M,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen359BGguf,
            2,
        ));
    }

    #[test]
    fn model_load_resource_plan_authorizes_backend_specific_peaks() {
        let estimate = ModelMemoryEstimate {
            load_peak_bytes: 96,
            resident_bytes: 64,
        };

        let cpu = model_resource_plan(BackendKind::Cpu, estimate);
        assert_eq!(
            cpu.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            cpu.resident_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );

        let metal = model_resource_plan(BackendKind::Metal, estimate);
        assert_eq!(
            metal.load_authorization,
            ResourceVector {
                unified_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            metal.resident_authorization,
            ResourceVector {
                unified_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );

        let cuda = model_resource_plan(BackendKind::Cuda, estimate);
        assert_eq!(
            cuda.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(96),
                device_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            cuda.resident_authorization,
            ResourceVector {
                device_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );
    }

    #[test]
    fn fish_s2_cuda_weight_lease_leaves_room_for_separately_charged_state() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Pinned checkpoint header arithmetic is verified in the Fish weight
        // tests. This regression checks the resource-authority wiring, without
        // claiming that every request or native-context cache fits this device.
        let memory = FishS2ModelMemory {
            resident_bytes: 10_712_372_164,
            load_peak_bytes: 12_307_518_404,
            cuda_host_load_peak_bytes: 9_944_351_744,
        };
        let plan = fish_s2_resource_plan(BackendKind::Cuda, memory).unwrap();
        assert_eq!(
            plan.resident_authorization,
            ResourceVector {
                device_bytes: ResourceAmount::Known(memory.resident_bytes),
                host_bytes: ResourceAmount::Known(
                    crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
                ),
                ..ResourceVector::zero()
            }
        );
        let authority = vector_authority(ResourceVector {
            host_bytes: ResourceAmount::Known(32 * GIB),
            device_bytes: ResourceAmount::Known(24 * GIB),
            ..ResourceVector::zero()
        });
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "fish-s2-state"),
                ResourceVector {
                    device_bytes: ResourceAmount::Known(4 * GIB),
                    ..ResourceVector::zero()
                },
            )
            .unwrap();
        let weights = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-s2-weights"),
                plan.load_authorization,
            )
            .expect("the weight peak and an independent state lease should fit");
        drop(weights);
        let old_catalog_plan = model_resource_plan(
            BackendKind::Cuda,
            ModelMemoryEstimate {
                load_peak_bytes: 24 * GIB,
                resident_bytes: 24 * GIB,
            },
        );
        let error = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-s2-flat-catalog"),
                old_catalog_plan.load_authorization,
            )
            .expect_err("the old flat 24 GiB lease left no room for state");
        assert!(matches!(error, Error::Overloaded(_)));
        drop(state);
    }

    #[test]
    fn fish_s2_lazy_reference_cache_keeps_headroom_reserved_after_weight_load() {
        let cache = crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES;
        let host = |n| ResourceVector {
            host_bytes: ResourceAmount::Known(n),
            ..ResourceVector::zero()
        };
        let plan = fish_s2_resource_plan(
            BackendKind::Cpu,
            FishS2ModelMemory {
                resident_bytes: 100,
                load_peak_bytes: 200,
                cuda_host_load_peak_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(plan.deferred_resident_authorization, host(cache));
        let authority = Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: host(cache + 1000),
                available: host(cache + 200),
                source: CapacitySource::Test,
            },
        })));
        let mut lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-weights-and-reference-cache"),
                plan.load_authorization,
            )
            .unwrap();
        lease
            .reconcile_materialized(
                plan.resident_authorization
                    .checked_sub(plan.deferred_resident_authorization)
                    .unwrap(),
            )
            .unwrap();
        lease.resize(plan.resident_authorization).unwrap();
        assert!(authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "oversized-state"),
                host(201)
            )
            .is_err());
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "fitted-state"),
                host(200),
            )
            .unwrap();
        drop(state);
        drop(lease);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[test]
    fn fish_s2_portable_weight_plan_charges_the_expanded_representation() {
        let cpu_memory = FishS2ModelMemory {
            resident_bytes: 19_836_076_996,
            load_peak_bytes: 22_228_796_356,
            cuda_host_load_peak_bytes: 9_944_351_744,
        };
        let cpu = fish_s2_resource_plan(BackendKind::Cpu, cpu_memory).unwrap();
        assert_eq!(
            cpu.resident_authorization.host_bytes,
            ResourceAmount::Known(
                cpu_memory.resident_bytes
                    + crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
        assert_eq!(
            cpu.load_authorization.host_bytes,
            ResourceAmount::Known(
                cpu_memory.load_peak_bytes
                    + crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
        assert_eq!(
            cpu.load_authorization.device_bytes,
            ResourceAmount::Known(0)
        );
        let half_memory = FishS2ModelMemory {
            resident_bytes: 10_712_372_164,
            load_peak_bytes: 12_307_518_404,
            ..cpu_memory
        };
        let metal = fish_s2_resource_plan(BackendKind::Metal, half_memory).unwrap();
        assert_eq!(
            metal.load_authorization.unified_bytes,
            ResourceAmount::Known(half_memory.load_peak_bytes)
        );
        assert_eq!(
            metal.resident_authorization.unified_bytes,
            ResourceAmount::Known(half_memory.resident_bytes)
        );
        assert_eq!(
            metal.load_authorization.host_bytes,
            ResourceAmount::Known(
                crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
    }

    #[test]
    fn lfm25_audio_model_memory_excludes_request_scoped_inference_workspace() {
        const GIB: u64 = 1024 * 1024 * 1024;

        assert_eq!(ModelVariant::Lfm25Audio15BGguf.memory_required_gb(), 5.0);
        assert_eq!(
            model_memory_estimate(ModelVariant::Lfm25Audio15BGguf),
            ModelMemoryEstimate {
                load_peak_bytes: 3 * GIB,
                resident_bytes: 3 * GIB,
            }
        );
    }

    #[test]
    fn bounded_model_memos_are_part_of_resident_authorization() {
        const GIB: u64 = 1024 * 1024 * 1024;

        let nemotron = model_memory_estimate(ModelVariant::Nemotron35AsrStreaming06B);
        assert_eq!(
            nemotron.resident_bytes,
            6 * GIB + crate::models::architectures::nemotron::asr::NEMOTRON_MODEL_MEMO_MAX_BYTES
        );
        let kokoro = model_memory_estimate(ModelVariant::Kokoro82M);
        assert_eq!(
            kokoro.resident_bytes,
            2 * GIB + crate::models::architectures::kokoro::KOKORO_MODEL_MEMO_MAX_BYTES
        );
        assert_eq!(nemotron.load_peak_bytes, nemotron.resident_bytes);
        assert_eq!(kokoro.load_peak_bytes, kokoro.resident_bytes);
    }

    #[test]
    fn kokoro_effective_context_respects_model_and_runtime_limits() {
        assert_eq!(kokoro_effective_context_tokens(512, 1_024).unwrap(), 512);
        assert_eq!(kokoro_effective_context_tokens(512, 256).unwrap(), 256);
        assert!(kokoro_effective_context_tokens(0, 1_024).is_err());
        assert!(kokoro_effective_context_tokens(512, 0).is_err());
    }

    #[test]
    fn automatic_invocation_context_treats_published_reach_as_a_ceiling() {
        assert_eq!(
            portable_invocation_context_intent(
                ContextLengthPreference::Auto,
                Some(128_000),
                131_072
            ),
            PortableInvocationContextIntent::Automatic { ceiling: 128_000 }
        );
        assert_eq!(
            portable_invocation_context_intent(
                ContextLengthPreference::Auto,
                Some(196_608),
                65_536
            ),
            PortableInvocationContextIntent::Automatic { ceiling: 65_536 }
        );
    }

    #[test]
    fn automatic_state_groups_share_remaining_headroom() {
        assert_eq!(automatic_state_group_budget(5_000, 5), 1_000);
        assert_eq!(automatic_state_group_budget(5_000, 2), 2_500);
        assert_eq!(automatic_state_group_budget(5_000, 1), 5_000);
        assert_eq!(automatic_state_group_budget(5_000, 0), 5_000);
    }

    #[test]
    fn lfm25_audio_context_fit_preserves_request_workspace_and_safety_headroom() {
        const GIB: u64 = 1024 * 1024 * 1024;

        assert_eq!(
            portable_context_reserve_bytes(ModelVariant::Lfm25Audio15BGguf, GIB),
            3 * GIB
        );
        assert_eq!(
            portable_context_reserve_bytes(ModelVariant::Kokoro82M, GIB),
            GIB
        );
    }

    /// Truth reconciliation (ADM-8): growth beyond the authorization plus the
    /// tolerance fails the load; growth inside it, or matched by leases
    /// reserved over the same interval (the load's own state, a concurrent
    /// request), passes and reports the observed bytes.
    #[test]
    fn observed_growth_beyond_the_authorization_fails_the_load() {
        const MIB: u64 = 1024 * 1024;
        let variant = ModelVariant::WhisperLargeV3Turbo;
        let sample = |observed, reserved| DeviceUsageSample { observed, reserved };
        let authorized = 4 * 1024 * MIB;
        let tolerance = materialization_tolerance_bytes(authorized);
        assert_eq!(tolerance, 512 * MIB);
        assert_eq!(materialization_tolerance_bytes(MIB), 256 * MIB);

        let before = sample(100 * MIB, authorized);
        let within = sample(100 * MIB + authorized + tolerance, authorized);
        assert_eq!(
            reconcile_observed_growth(variant, "weight load", before, within, authorized).unwrap(),
            authorized + tolerance
        );
        let over = sample(within.observed + 1, authorized);
        let error = reconcile_observed_growth(variant, "weight load", before, over, authorized)
            .unwrap_err()
            .to_string();
        assert!(error.contains("under-prices"), "{error}");

        // State leases reserved between the samples authorize their growth.
        let state_before = sample(0, 1_000 * MIB);
        let state_after = sample(1_200 * MIB, 2_000 * MIB);
        assert!(reconcile_observed_growth(
            variant,
            "state allocation",
            state_before,
            state_after,
            0
        )
        .is_ok());
        let state_unauthorized = sample(1_200 * MIB, 1_000 * MIB);
        assert!(reconcile_observed_growth(
            variant,
            "state allocation",
            state_before,
            state_unauthorized,
            0
        )
        .is_err());
        // A release elsewhere (observed shrinking) never fails a load.
        assert_eq!(
            reconcile_observed_growth(variant, "weight load", sample(MIB, 0), sample(0, 0), 0)
                .unwrap(),
            0
        );
    }

    /// The Granite shape from a live Metal run: after the weights land, live
    /// headroom (0.8 GB) is below the 1 GiB reserve while planning headroom
    /// holds gigabytes. The minimum state must still be admitted (against
    /// planning); live memory caps only growth above it.
    #[test]
    fn asr_state_floor_is_admitted_against_planning_when_live_undercounts() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let reserve = GIB;
        let (floor, growth) =
            state_fit_budgets(5 * GIB, ResourceAmount::Known(796_545_836), reserve);
        assert_eq!(floor, 4 * GIB);
        assert_eq!(growth, 0);
        assert!(
            10_485_760 <= floor,
            "a one-page minimum fits the floor budget"
        );
        let (floor, growth) = state_fit_budgets(5 * GIB, ResourceAmount::Known(3 * GIB), reserve);
        assert_eq!((floor, growth), (4 * GIB, 2 * GIB));
        // Live never raises growth above the planning floor.
        let (floor, growth) = state_fit_budgets(2 * GIB, ResourceAmount::Known(9 * GIB), reserve);
        assert_eq!((floor, growth), (GIB, GIB));
        assert_eq!(
            state_fit_budgets(2 * GIB, ResourceAmount::Unknown, reserve),
            (GIB, GIB)
        );
    }

    /// ASR rows: chat's automatic rows (320 on an H100) collapse to 8, never
    /// below one decode batch; operator rows stay as configured.
    /// On CPU and Metal the engine rows stay at the portable width (TTS and
    /// ASR fit their state on them), and a chat model stages one step's
    /// launch at its own certified decode width (EXE-F7).
    #[test]
    fn portable_chat_models_stage_rows_at_their_decode_width() {
        use crate::runtime::adapters::LoadedModelBundleDraft;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
        let staged = |runtime: &RuntimeService, variant: ModelVariant| {
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(992),
                ModelInstanceId::new(1),
                variant,
                BackendKind::Cpu,
            )
            .unwrap();
            draft.seal_chat_workspace(8_192).unwrap();
            runtime
                .model_lifecycle
                .portable_chat_staged_rows(&draft, BackendKind::Cpu)
                .unwrap()
        };
        let automatic = RuntimeService::new(EngineConfig {
            backend: BackendPreference::Cpu,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(automatic.core_engine.config().max_staged_transactions, 2);
        assert_eq!(
            staged(&automatic, ModelVariant::Lfm2512BInstructGguf),
            Some(8)
        );
        assert_eq!(staged(&automatic, ModelVariant::Qwen306B), Some(2));

        let explicit = RuntimeService::new(EngineConfig {
            backend: BackendPreference::Cpu,
            max_staged_transactions: 3,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(staged(&explicit, ModelVariant::Lfm2512BInstructGguf), None);
    }

    #[test]
    fn asr_state_rows_come_from_the_asr_policy_not_chat_rows() {
        assert_eq!(asr_state_rows(true, 320, 4), (8, 4));
        assert_eq!(asr_state_rows(true, 320, 16), (16, 16));
        assert_eq!(asr_state_rows(true, 4, 4), (4, 4));
        assert_eq!(asr_state_rows(false, 320, 32), (320, 32));
        assert_eq!(asr_state_rows(true, 0, 0), (1, 1));
    }

    /// The request floor joins the load authorization before eviction: the
    /// authorization is the larger of the weight peak and resident weights
    /// plus the floor.
    #[test]
    fn asr_request_floor_joins_the_load_authorization() {
        let estimate = ModelMemoryEstimate {
            load_peak_bytes: 1_000,
            resident_bytes: 800,
        };
        let plan = model_resource_plan(BackendKind::Metal, estimate);
        let unified = |authorization: ResourceVector| authorization.unified_bytes;
        assert_eq!(
            unified(admit_asr_request_floor(&plan, BackendKind::Metal, 100).unwrap()),
            ResourceAmount::Known(1_000)
        );
        assert_eq!(
            unified(admit_asr_request_floor(&plan, BackendKind::Metal, 500).unwrap()),
            ResourceAmount::Known(1_300)
        );
        let cuda = model_resource_plan(BackendKind::Cuda, estimate);
        let authorization = admit_asr_request_floor(&cuda, BackendKind::Cuda, 500).unwrap();
        assert_eq!(authorization.device_bytes, ResourceAmount::Known(1_300));
        assert_eq!(authorization.host_bytes, cuda.load_authorization.host_bytes);
    }

    /// On a 16 GB M1 Pro-like Metal budget (0.9 × the 12.7 GB working set),
    /// a state fit that spends everything it may see after the model's
    /// weights and the held request envelope still leaves one full-chunk
    /// envelope free once the hold is released (ADM-1, ADM-4).
    #[test]
    fn state_fits_leave_one_request_envelope_on_a_16_gb_metal_budget() {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        const BUDGET: u64 = 11_441_804_076;
        let unified = |bytes| ResourceVector {
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        };
        // Metal-allocated resident weights from the residency plans.
        for (variant, resident_gib) in [
            (ModelVariant::Qwen3Asr06BGguf, 1.54),
            (ModelVariant::Qwen3Asr17BGguf, 3.30),
            (ModelVariant::Lfm25Audio15BGguf, 2.08),
            (ModelVariant::GraniteSpeech412BPlus, 5.87),
        ] {
            let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
                Arc::new(VectorCapacityProvider {
                    snapshot: PhysicalCapacitySnapshot {
                        capacity: unified(BUDGET),
                        available: unified(BUDGET),
                        source: CapacitySource::Test,
                    },
                }),
            ));
            let resident = (resident_gib * GIB) as u64;
            let _model = authority
                .reserve_with_initial_materialized(
                    ReservationOwner::new(ReservationClass::Model, "weights"),
                    unified(resident),
                    unified(resident),
                )
                .unwrap();
            let envelope = super::super::asr_envelope::deferred_request_envelope(
                variant,
                BackendKind::Metal,
                1,
            )
            .unwrap()
            .unwrap()
            .total_bytes()
            .unwrap();
            let hold = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Model, "envelope"),
                    unified(envelope),
                )
                .unwrap();
            let ResourceAmount::Known(fit) = authority
                .state_fit_headroom_bytes(BackendKind::Metal)
                .unwrap()
            else {
                panic!("known headroom");
            };
            let _state = authority
                .reserve(
                    ReservationOwner::new(ReservationClass::Cache, "greedy-state"),
                    unified(fit),
                )
                .unwrap();
            drop(hold);
            let ResourceAmount::Known(left) = authority
                .planning_headroom_bytes(BackendKind::Metal)
                .unwrap()
            else {
                panic!("known headroom");
            };
            assert!(left >= envelope, "{variant}: {left} < {envelope}");
            assert!(fit > 0, "{variant}");
        }
    }

    /// The weight lease records the smaller of the estimate and the observed
    /// growth: priced bytes the loader never allocated stay pending.
    #[tokio::test]
    async fn observed_weight_growth_is_recorded_as_materialized() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-observed-weights-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..Default::default()
        })
        .unwrap();
        let authority = vector_authority(all_memory_capacity(1024));
        let bytes = |n| ResourceVector {
            unified_bytes: ResourceAmount::Known(n),
            ..ResourceVector::zero()
        };
        let variant = ModelVariant::WhisperLargeV3Turbo;
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "observed-weights"),
                bytes(256),
            )
            .unwrap();
        runtime
            .model_lifecycle
            .install_loading_slot(variant, lease)
            .unwrap();
        runtime
            .model_lifecycle
            .finalize_slot_materialization_with_pending(
                variant,
                bytes(200),
                ResourceVector::zero(),
                Some((BackendKind::Metal, 150)),
            )
            .unwrap();
        let snapshot = authority.snapshot();
        assert_eq!(snapshot.reserved, bytes(200));
        assert_eq!(snapshot.materialized, bytes(150));
        assert_eq!(snapshot.pending, bytes(50));
        assert!(runtime.model_lifecycle.remove_resident_slot(variant));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[test]
    fn explicit_invocation_context_remains_mandatory_after_state_publication() {
        let explicit = ContextLengthPreference::explicit(32_768).unwrap();
        assert_eq!(
            portable_invocation_context_intent(explicit, Some(8_192), 65_536),
            PortableInvocationContextIntent::Explicit { selected: 32_768 }
        );
    }

    #[test]
    fn lfm25_audio_cold_load_fits_with_separately_reserved_request_workspace() {
        const GIB: u64 = 1024 * 1024 * 1024;

        let authority = vector_authority(ResourceVector {
            unified_bytes: ResourceAmount::Known(5 * GIB),
            ..ResourceVector::zero()
        });
        let _request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "lfm-request"),
                ResourceVector {
                    unified_bytes: ResourceAmount::Known(2 * GIB),
                    ..ResourceVector::zero()
                },
            )
            .expect("request workspace should fit");
        let plan = model_resource_plan(
            BackendKind::Metal,
            model_memory_estimate(ModelVariant::Lfm25Audio15BGguf),
        );

        let _model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "lfm-model"),
                plan.load_authorization,
            )
            .expect("model-owned memory should not include request workspace twice");
    }

    #[test]
    fn cuda_load_is_rejected_when_only_device_peak_has_capacity() {
        let plan = model_resource_plan(
            BackendKind::Cuda,
            ModelMemoryEstimate {
                load_peak_bytes: 64,
                resident_bytes: 64,
            },
        );
        let capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(63),
            device_bytes: ResourceAmount::Known(64),
            ..ResourceVector::zero()
        };
        let authority = vector_authority(capacity);

        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "cuda-host-peak"),
                plan.load_authorization,
            ),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[tokio::test]
    async fn published_models_retain_only_backend_residency_authorization() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-resource-finalize-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let plan = model_resource_plan(
                backend,
                ModelMemoryEstimate {
                    load_peak_bytes: 64,
                    resident_bytes: 64,
                },
            );
            let authority = vector_authority(all_memory_capacity(1024));
            let resource_lease = authority
                .reserve(
                    ReservationOwner::new(
                        ReservationClass::Model,
                        format!("{backend:?}-publication"),
                    ),
                    plan.load_authorization,
                )
                .expect("peak load authorization");
            runtime
                .model_lifecycle
                .install_loading_slot(variant, resource_lease)
                .expect("loading slot");

            runtime
                .model_lifecycle
                .finalize_slot_materialization(variant, plan.resident_authorization)
                .expect("publication resource finalization");
            assert_eq!(
                authority.snapshot().reserved,
                plan.resident_authorization,
                "{backend:?} retained the wrong resident authorization"
            );

            assert!(runtime.model_lifecycle.remove_resident_slot(variant));
            assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        }

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn deferred_model_graph_capacity_remains_pending_during_state_fitting() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-deferred-graphs-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..Default::default()
        })
        .unwrap();
        let authority = Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: all_memory_capacity(1024),
                available: all_memory_capacity(100),
                source: CapacitySource::Test,
            },
        })));
        let bytes = |n| ResourceVector {
            device_bytes: ResourceAmount::Known(n),
            ..ResourceVector::zero()
        };
        let variant = ModelVariant::Qwen3827BFp8;
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "weights-and-graphs"),
                bytes(64),
            )
            .unwrap();
        runtime
            .model_lifecycle
            .install_loading_slot(variant, lease)
            .unwrap();
        runtime
            .model_lifecycle
            .finalize_slot_materialization_with_pending(variant, bytes(64), bytes(16), None)
            .unwrap();
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "oversized-state"),
                bytes(85)
            ),
            Err(Error::Overloaded(_))
        ));
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fitted-state"),
                bytes(84),
            )
            .unwrap();
        assert!(runtime.model_lifecycle.remove_resident_slot(variant));
        drop(state);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_unload_supersedes_registered_load_before_spawn() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;

        // Register the generation but deliberately delay spawning its detached
        // transaction. This is the exact window where unload used to return
        // before the stale task acquired the gate and published the model.
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("registered load leader");
        let stale_generation = leader.generation;

        runtime
            .unload_model(variant)
            .await
            .expect("explicit unload supersedes pending load");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        }
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("superseded waiter timed out")
            .expect_err("superseded load must fail");
        assert!(matches!(
            error,
            Error::Cancelled(message)
                if message.contains("superseded by explicit unload")
        ));

        let stale_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader,
        );
        tokio::time::timeout(Duration::from_secs(1), stale_task)
            .await
            .expect("stale detached load timed out")
            .expect("stale detached load task");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        // Removal of the stale registration must allow a later request to own
        // a new generation rather than coalescing with cancelled work.
        let (retry_waiter, retry_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let retry_leader = retry_leader.expect("new load generation after unload");
        assert_ne!(retry_leader.generation, stale_generation);
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                retry_leader.generation,
                &retry_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::Cancelled("test cleanup".to_string())),
            );
        }
        assert!(matches!(
            retry_waiter.wait().await,
            Err(Error::Cancelled(_))
        ));

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unload_all_supersedes_every_registered_load_before_spawn() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority (the GGUF variant alone claims ~2.25 GiB).
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-all-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variants = [ModelVariant::Kokoro82M, ModelVariant::Qwen359BGguf];
        let mut registrations = Vec::new();
        for variant in variants {
            let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
            registrations.push((variant, waiter, leader.expect("registered load leader")));
        }

        assert_eq!(
            runtime
                .unload_all_models()
                .await
                .expect("unload-all supersedes pending loads"),
            0
        );

        for (variant, waiter, leader) in registrations {
            let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
                .await
                .expect("superseded unload-all waiter timed out")
                .expect_err("superseded load must fail");
            assert!(matches!(
                error,
                Error::Cancelled(message)
                    if message.contains("superseded by explicit unload-all")
            ));
            let stale_task = runtime.model_lifecycle.spawn_load_transaction(
                variant,
                runtime.max_loaded_models,
                leader,
            );
            tokio::time::timeout(Duration::from_secs(1), stale_task)
                .await
                .expect("stale unload-all load timed out")
                .expect("stale unload-all task");
            assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
            assert!(!runtime.model_manager.is_ready(variant).await);
        }
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_outcome_is_published_before_explicit_unload_enters_the_gate() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-publication-race-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("publication-race");
        let installed_instance = runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, resources)
            .expect("materialized slot");
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("load leader");

        let publication_reached = Arc::new(Barrier::new(2));
        let publication_release = Arc::new(Barrier::new(2));
        let events = Arc::new(StdMutex::new(Vec::new()));
        let controller = runtime.model_lifecycle.clone();
        let reached = publication_reached.clone();
        let release = publication_release.clone();
        let publication_events = events.clone();
        let publisher = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            reached.wait().await;
            release.wait().await;
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
            publication_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("outcome");
        });

        publication_reached.wait().await;
        let lease = runtime
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .expect("ready instance lease");
        assert_eq!(lease.model_instance_id(), Some(installed_instance));
        drop(lease);
        let bundle = runtime
            .model_lifecycle
            .try_get_ready_bundle(variant)
            .expect("ready execution bundle");
        assert_eq!(bundle.model_instance_id(), installed_instance);
        assert_eq!(bundle.model_variant(), variant);
        assert_eq!(
            bundle.execution_group_id(),
            runtime.coordinator.execution_group_id()
        );
        assert!(bundle.adapter_count() > 0);
        let unload_controller = runtime.model_lifecycle.clone();
        let unload_events = events.clone();
        let mut unload = tokio::spawn(async move {
            unload_controller
                .unload_model_detached(variant)
                .await
                .expect("explicit unload");
            unload_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("unload");
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut unload)
                .await
                .is_err(),
            "unload must not cross the gate before the load outcome is published"
        );

        publication_release.wait().await;
        waiter.wait().await.expect("shared ready outcome");
        publisher.await.expect("publisher task");
        unload.await.expect("unload task");
        assert_eq!(
            *events.lock().unwrap_or_else(|poison| poison.into_inner()),
            vec!["outcome", "unload"]
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    // The OOM-ladder cases and every other test that reserves against the
    // process-global authority serialize on the shared
    // crate::engine::resources::test_support::authority_test_lock().

    struct ClearAuthorityPoisonOnDrop(Arc<ResourceAuthority>);

    impl Drop for ClearAuthorityPoisonOnDrop {
        fn drop(&mut self) {
            self.0.clear_poison();
        }
    }

    // The ladder lock deliberately spans every await in these tests: it
    // serializes the process-global resource authority against the other
    // Metal OOM ladder cases, not shared state inside this runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metal_oom_load_is_flushed_and_retried_once_before_poisoning() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-oom-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        runtime.model_lifecycle.set_load_test_metal_ooms(1);

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        // The first attempt fails with the injected command-buffer OOM; the
        // wrapper must flush pooled scratch and retry once. The retry is not
        // injected, so it fails with an ordinary missing-model error instead
        // of the OOM — proving the attempt actually re-ran.
        let error = tokio::time::timeout(Duration::from_secs(2), waiter.wait())
            .await
            .expect("retry outcome timed out")
            .expect_err("missing model artifacts must fail");
        assert!(
            !format!("{error}").contains("kIOGPUCommandBufferCallbackErrorOutOfMemory"),
            "the retried attempt must not surface the injected OOM: {error}"
        );
        assert!(
            runtime
                .coordinator
                .resource_authority()
                .poison_reason()
                .is_none(),
            "a single OOM must not poison the authority"
        );
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn persisted_metal_oom_after_the_retry_poisons_the_authority() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-oom2-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        runtime.model_lifecycle.set_load_test_metal_ooms(2);
        // A resident blocks the recovery drain, so the poison is observable
        // after the failed load. Unpinned residents would be evicted and the
        // authority would recover before the outcome lands.
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        let error = tokio::time::timeout(Duration::from_secs(2), waiter.wait())
            .await
            .expect("poison outcome timed out")
            .expect_err("persisted OOM must fail");
        assert!(
            format!("{error}").contains("kIOGPUCommandBufferCallbackErrorOutOfMemory"),
            "the injected OOM must surface after the retry: {error}"
        );
        let reason = runtime
            .coordinator
            .resource_authority()
            .poison_reason()
            .expect("persisted OOM must poison the authority");
        assert!(
            reason.contains("persisted after a pooled-scratch flush and one retry"),
            "unexpected poison reason: {reason}"
        );
        drop(ClearAuthorityPoisonOnDrop(
            runtime.coordinator.resource_authority(),
        ));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn poisoned_authority_recovers_when_the_device_fully_drains() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-poison-recover-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        let authority = runtime.coordinator.resource_authority();
        authority.poison("simulated backend-fatal error");
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::GraniteSpeech412BPlus,
                now_unix_millis().saturating_sub(10_000),
            );

        // The unpinned resident is the only thing keeping the device from
        // draining; reaping it via the standard unload path must clear the
        // poison.
        let reaped = runtime.reap_idle_models().await;
        assert_eq!(reaped, vec![ModelVariant::GraniteSpeech412BPlus]);
        assert!(
            authority.poison_reason().is_none(),
            "a fully drained device must recover from poison"
        );

        // A pinned resident blocks the drain, so the poison persists.
        authority.poison("simulated backend-fatal error");
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::WhisperLargeV3Turbo,
                now_unix_millis().saturating_sub(10_000),
            );
        assert!(runtime.reap_idle_models().await.is_empty());
        assert!(
            authority.poison_reason().is_some(),
            "a pinned resident must keep the poison until it is unloaded"
        );
        drop(ClearAuthorityPoisonOnDrop(authority));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn detached_load_panic_rolls_back_before_publishing_failure() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-panic-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (authority, resource_lease) = isolated_resource_lease("load-panic");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime.model_manager.mark_loaded(variant).await;
        runtime.model_lifecycle.set_load_test_panics(1);

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("panic outcome timed out")
            .expect_err("injected load panic must fail");
        assert!(
            matches!(error, Error::ModelLoadError(message) if message.contains("injected model load panic"))
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);

        let (cleanup_waiter, cleanup_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let cleanup_leader = cleanup_leader.expect("failed generation must be removable");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                cleanup_leader.generation,
                &cleanup_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad("test cleanup".to_string())),
            );
        }
        assert!(cleanup_waiter.wait().await.is_err());

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn idle_reaper_unloads_idle_transient_models_but_preserves_pins() {
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-reap-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;

        let stale = now_unix_millis().saturating_sub(10_000);
        {
            let mut last_used = runtime.model_lifecycle.model_last_used.lock().await;
            last_used.insert(ModelVariant::GraniteSpeech412BPlus, stale);
            last_used.insert(ModelVariant::WhisperLargeV3Turbo, stale);
        }

        let reaped = runtime.reap_idle_models().await;
        assert_eq!(reaped, vec![ModelVariant::GraniteSpeech412BPlus]);

        let residents = runtime.model_manager.resident_variants().await;
        assert!(!residents.contains(&ModelVariant::GraniteSpeech412BPlus));
        assert!(residents.contains(&ModelVariant::WhisperLargeV3Turbo));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn idle_reaper_is_disabled_without_keep_alive() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-reap-off-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 0,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::GraniteSpeech412BPlus,
                now_unix_millis().saturating_sub(10_000),
            );

        assert!(runtime.reap_idle_models().await.is_empty());
        assert!(runtime
            .model_manager
            .resident_variants()
            .await
            .contains(&ModelVariant::GraniteSpeech412BPlus));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn residency_budget_preserves_pinned_residents_and_reports_make_room() {        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-residency-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            max_loaded_models: Some(1),
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::GraniteSpeech412BPlus)
            .await;

        let error = runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .expect_err("a pinned resident must not be silently evicted");
        assert!(
            error.to_string().contains("unload a model to make room"),
            "unexpected error: {error}"
        );
        assert_eq!(error.kind(), crate::error::ErrorKind::CapacityExhausted);

        assert!(runtime
            .model_manager
            .resident_variants()
            .await
            .contains(&ModelVariant::GraniteSpeech412BPlus));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn residency_budget_evicts_granite_before_loading_another_asr_model() {
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-residency-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            max_loaded_models: Some(1),
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;

        runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .unwrap();

        assert!(runtime.model_manager.resident_variants().await.is_empty());
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cold_model_load_is_rejected_before_artifact_work_during_drain() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-drain-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime.coordinator.begin_drain();

        assert!(matches!(
            runtime
                .load_model_for_inference(ModelVariant::Kokoro82M)
                .await,
            Err(Error::Overloaded(_))
        ));
        assert!(runtime.model_manager.resident_variants().await.is_empty());

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    /// SPK-F7: a realtime voice session holds its models through
    /// `SessionModelLeases`. The models are never pinned, a held model is not
    /// evicted to make room for another, and it becomes evictable once the
    /// session drops its leases.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn session_model_leases_protect_without_pinning() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-session-lease-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            max_loaded_models: Some(1),
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (_authority, resource_lease) = isolated_resource_lease("session-lease");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, one_byte_host_reservation())
            .expect("materialized slot");
        runtime.model_manager.mark_loaded(variant).await;
        runtime
            .model_lifecycle
            .mark_slot_ready(variant)
            .expect("ready slot");

        let leases = super::SessionModelLeases::default();
        leases.hold(&runtime, variant).await.expect("first turn");
        leases
            .clone()
            .hold(&runtime, variant)
            .await
            .expect("next turn");
        assert!(!runtime.pinned_model_variants().await.contains(&variant));
        assert_eq!(runtime.model_manager.active_residency_leases(variant), 1);

        let error = runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .expect_err("a model held by a session is not evicted");
        assert!(error.to_string().contains("no resident model is idle"));
        assert_eq!(
            runtime.model_manager.resident_variants().await,
            vec![variant]
        );

        drop(leases);
        assert_eq!(runtime.model_manager.active_residency_leases(variant), 0);
        runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .expect("an unpinned model is evictable after the session");
        assert!(runtime.model_manager.resident_variants().await.is_empty());

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cancelled_waiter_keeps_shared_load_accounted_and_visible_to_drain() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-cancelled-load-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("cancelled-load-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        let (first_waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("first waiter leads the load");
        let (second_waiter, second_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        assert!(second_leader.is_none(), "second waiter must coalesce");

        let first_task = tokio::spawn(first_waiter.wait());
        let second_task = tokio::spawn(second_waiter.wait());
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let controller = runtime.model_lifecycle.clone();
        let calls = loader_calls.clone();
        let transaction = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            let _coordinator_load = controller
                .coordinator
                .begin_model_load("cancelled-shared-load")
                .expect("model load admission");
            calls.fetch_add(1, Ordering::AcqRel);
            let _ = started_tx.send(());
            let _ = release_rx.await;
            controller
                .finalize_slot_materialization(variant, resources)
                .expect("materialized lease");
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        });

        started_rx.await.expect("fake load started");
        first_task.abort();
        assert!(first_task
            .await
            .expect_err("first waiter should be cancelled")
            .is_cancelled());
        assert_eq!(loader_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 1);
        assert_eq!(authority.snapshot().reservations, 1);

        runtime.begin_drain();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(25),
                runtime
                    .coordinator
                    .wait_for_idle(Instant::now() + Duration::from_secs(1)),
            )
            .await
            .is_err(),
            "drain must still observe the detached load"
        );

        release_tx.send(()).expect("release fake load");
        second_task
            .await
            .expect("second waiter join")
            .expect("coalesced waiter succeeds");
        transaction.await.expect("fake load transaction");
        runtime
            .coordinator
            .wait_for_idle(Instant::now() + Duration::from_secs(1))
            .await
            .expect("drain after load completion");
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Ready)
        );
        assert!(runtime.model_manager.is_ready(variant).await);

        assert_eq!(
            runtime.unload_all_models().await.expect("shutdown unload"),
            1
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn failed_publication_rolls_back_slot_before_releasing_lease() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-rollback-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (authority, resource_lease) = isolated_resource_lease("rollback-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(authority.snapshot().reservations, 1);
        let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
        runtime
            .model_lifecycle
            .rollback_model_locked(variant)
            .await
            .expect("rollback");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    /// A unified-memory machine whose provider samples live memory and
    /// observes device usage (Metal's `currentAllocatedSize`).
    #[derive(Debug)]
    struct ObservedUnifiedProvider {
        capacity: u64,
        available: u64,
        observed: u64,
    }

    impl PhysicalCapacityProvider for ObservedUnifiedProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let unified = |bytes| ResourceVector {
                unified_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity: unified(self.capacity),
                available: unified(self.available),
                source: CapacitySource::Test,
            }
        }

        fn observed_device_usage_bytes(&self) -> Option<u64> {
            Some(self.observed)
        }
    }

    /// The final Metal run's Qwen3-TTS-0.6B-Base load (about 11.4 GB of
    /// planning capacity, 10 GB live, 2.67 GB of weights) reserves its one
    /// request slot's envelope: CustomVoice's live envelope plus a 30 s
    /// reference encode on the shipped encoder. The reference term was
    /// 47 GB and refused the load with the generic capacity error; a
    /// refusal now names what the envelope needs and what is available.
    #[test]
    fn qwen_base_request_envelope_fits_beside_its_weights() {
        use crate::models::architectures::qwen3::tts::speech_tokenizer::{
            reference_encode_workspace_upper_bound_bytes, EncoderConfig,
            QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON,
        };
        let encoder: EncoderConfig =
            serde_json::from_str(QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON).unwrap();
        let reference =
            reference_encode_workspace_upper_bound_bytes(&encoder, 24_000, 1_920, 30, 4).unwrap();
        // CustomVoice's envelope from the same run (no reference encode).
        let envelope = super::super::asr_envelope::AsrRequestEnvelope {
            host_bytes: 46_377_456,
            accelerator_bytes: 471_076_352 + reference,
        };
        /// Metal as the server builds it: a configured planning budget
        /// beside a live sample.
        #[derive(Debug)]
        struct BudgetedUnifiedProvider;
        impl PhysicalCapacityProvider for BudgetedUnifiedProvider {
            fn snapshot(&self) -> PhysicalCapacitySnapshot {
                let unified = |bytes| ResourceVector {
                    unified_bytes: ResourceAmount::Known(bytes),
                    ..ResourceVector::zero()
                };
                PhysicalCapacitySnapshot {
                    capacity: unified(16_000_000_000),
                    available: unified(10_000_000_000),
                    source: CapacitySource::MetalWorkingSet,
                }
            }
            fn configured_budget(&self) -> Option<ResourceVector> {
                Some(ResourceVector {
                    host_bytes: ResourceAmount::Known(11_400_000_000),
                    unified_bytes: ResourceAmount::Known(11_400_000_000),
                    ..ResourceVector::zero()
                })
            }
            fn observed_device_usage_bytes(&self) -> Option<u64> {
                Some(2_668_085_248)
            }
        }
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(BudgetedUnifiedProvider),
        ));
        let weights = ResourceVector {
            unified_bytes: ResourceAmount::Known(2_667_798_788),
            ..ResourceVector::zero()
        };
        let _weights = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Model, "weights"),
                weights,
                weights,
            )
            .unwrap();
        authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "envelope"),
                super::request_envelope_resources(BackendKind::Metal, envelope).unwrap(),
            )
            .unwrap_or_else(|error| panic!("reference {reference} bytes: {error}"));

        let refusal = super::request_envelope_shortfall(
            ModelVariant::Qwen3Tts12Hz06BBase,
            BackendKind::Metal,
            envelope,
            1_000_000_000,
        )
        .unwrap();
        assert!(matches!(refusal, crate::error::Error::Overloaded(_)));
        let message = refusal.to_string();
        assert!(message.contains("needs "), "{message}");
        assert!(message.contains("1.00 GB available"), "{message}");
    }

    const GIB: u64 = 1024 * 1024 * 1024;
    /// What load admission charged Qwen3-TTS 0.6B's weights at 0adabf16.
    const TTS_CHARGED_WEIGHT_BYTES: u64 = 2_493_807_108;

    /// Load TTS retained state through the engine's real load path the way
    /// `load_model` does for an unobserved (non-ASR) load, beside weights the
    /// ledger charged at [`TTS_CHARGED_WEIGHT_BYTES`] while the device
    /// observer reports `observed` bytes in use.
    fn load_tts_retained_state(
        variant: ModelVariant,
        contract: &crate::kv::v2::InferenceStateContract,
        maximum_tokens: usize,
        capacity: u64,
        available: u64,
        observed: u64,
    ) -> crate::error::Result<u64> {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(ObservedUnifiedProvider {
                capacity,
                available,
                observed,
            }),
        ));
        let weights = ResourceVector {
            unified_bytes: ResourceAmount::Known(TTS_CHARGED_WEIGHT_BYTES),
            ..ResourceVector::zero()
        };
        let _weights = authority.reserve_with_initial_materialized(
            ReservationOwner::new(ReservationClass::Model, variant.to_string()),
            weights,
            weights,
        )?;
        assert!(
            authority.untracked_device_bytes(BackendKind::Cpu).unwrap() > 0,
            "observed usage must exceed what the ledger materialized"
        );
        let mut config = crate::engine::EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_retained_sequences: 1,
            max_staged_transactions: 1,
            ..Default::default()
        };
        config.portable_context_auto = true;
        let mut core =
            crate::engine::EngineCore::new_with_managed_authority_for_test(config, authority)?;
        let runtime = core.load_managed_model_state(
            ModelInstanceId::new(4_101),
            contract,
            Some(maximum_tokens),
            super::tts_state_minimum_tokens(variant),
            crate::engine::StateFitReserve::for_observed_load(false),
        )?;
        Ok(runtime.logical_token_reach())
    }

    /// A2: portable chat arenas are fully resident from load, so fitting the
    /// model's whole advertised context pinned gigabytes for an idle model.
    /// The automatic fit is capped; a zero cap restores the full context.
    #[test]
    fn portable_auto_chat_context_is_capped() -> crate::error::Result<()> {
        for (cap, expected) in [(4_096_usize, 4_096_u64), (0, 16_384)] {
            let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
                Arc::new(ObservedUnifiedProvider {
                    capacity: 64 * GIB,
                    available: 64 * GIB,
                    observed: 0,
                }),
            ));
            let mut config = crate::engine::EngineCoreConfig {
                backend: BackendKind::Cpu,
                max_retained_sequences: 1,
                max_staged_transactions: 1,
                portable_auto_context_cap_tokens: cap,
                ..Default::default()
            };
            config.portable_context_auto = true;
            let mut core =
                crate::engine::EngineCore::new_with_managed_authority_for_test(config, authority)?;
            let runtime = core
                .load_managed_model_cache(
                    ModelInstanceId::new(4_102),
                    &crate::kv::InferenceStateCapability::Managed(crate::kv::test_contract()),
                    Some(16_384),
                )?
                .expect("managed runtime");
            assert_eq!(runtime.logical_token_reach(), expected, "cap {cap}");
        }
        Ok(())
    }

    /// R-1: Qwen3-TTS retained state collapsed to one 64-token page on
    /// Metal, cutting every CustomVoice request at 54 frames and refusing
    /// every clone. The fit must reach every request's prompt plus its
    /// automatic budget whatever live memory reads, spend live growth above
    /// it without reserving the unobserved weight under-pricing a second
    /// time, and refuse with a 503 naming the shortfall when planning
    /// headroom cannot hold the floor.
    #[test]
    fn qwen3_tts_effective_context_covers_prompt_and_frame_budget_under_low_live_headroom() {
        use crate::models::architectures::qwen3::tts::{
            qwen3_tts_auto_max_frames_for_text, qwen3_tts_auto_max_output_frames,
            qwen3_tts_inference_state_contract, qwen3_tts_retained_talker_contract,
            tests::cache_test_config,
        };
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let config = cache_test_config();
        let full = qwen3_tts_inference_state_contract(
            &config,
            candle_core::DType::F32,
            candle_core::DType::F32,
            crate::models::shared::attention::paged::default_kv_page_size(),
        )
        .unwrap();
        let contract =
            qwen3_tts_retained_talker_contract(&full, &config, candle_core::DType::F32).unwrap();
        // A clone prompt with the longest reference: role and codec-prefix
        // tokens plus 320 reference codes.
        let max_clone_prefill = 9 + 320 + 1;
        let paragraph = "Good morning. Today we are testing a speech system that reads \
            longer passages aloud, and we want to hear whether every sentence comes out \
            clearly without the voice drifting, rushing, or stopping before the final word.";
        assert!(qwen3_tts_auto_max_frames_for_text(paragraph) < qwen3_tts_auto_max_output_frames());
        let required = (max_clone_prefill + qwen3_tts_auto_max_output_frames()) as u64;
        let maximum = 3_072;
        let budget = 11 * GIB + 440 * 1024 * 1024;
        let observed = 6_000_000_000;

        // Live memory is far below planning and the observed usage leaves
        // nothing above the measured untracked bytes: the floor holds.
        let reach = load_tts_retained_state(variant, &contract, maximum, budget, GIB / 2, observed)
            .unwrap();
        assert!(reach >= required, "effective context {reach} < {required}");
        assert!(reach < maximum as u64, "live memory must still cap growth");

        // With 3 GiB live, the configured reserve leaves growth for the full
        // context; the unobserved weight under-pricing is not reserved again.
        let reach = load_tts_retained_state(variant, &contract, maximum, budget, 3 * GIB, observed)
            .unwrap();
        assert_eq!(reach, maximum as u64);

        // A floor planning headroom cannot hold refuses the load (503).
        let tight = TTS_CHARGED_WEIGHT_BYTES + GIB + 64 * 1024 * 1024;
        match load_tts_retained_state(variant, &contract, maximum, tight, tight, observed) {
            Err(Error::Overloaded(message)) => {
                assert!(
                    message.starts_with("not enough memory for this model's minimum")
                        && message.contains(" GB, ")
                        && !message.contains("planning_headroom="),
                    "{message}"
                )
            }
            other => panic!("expected a capacity refusal, got {other:?}"),
        }
    }

    /// Phase 0 state-ceiling contract: every TTS family whose load fits
    /// retained state guarantees one full request (its longest prompt plus
    /// its largest automatic output budget), and families sized elsewhere
    /// ask for no floor.
    #[test]
    fn tts_state_floor_covers_one_full_request_per_family() {
        use crate::catalog::ModelFamily;
        use crate::models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames;
        use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
        let flood = "word ".repeat(20_000);
        let floor = |variant| super::tts_state_minimum_tokens(variant) as usize;
        for variant in ModelVariant::all().iter().copied() {
            if !variant.is_enabled() || !variant.is_tts() {
                continue;
            }
            let required = match variant.family() {
                // 320 reference codes plus role and codec-prefix tokens.
                ModelFamily::Qwen3Tts => 330 + qwen3_tts_auto_max_output_frames(),
                // A 12 s voice reference (90 latents) plus its markers.
                ModelFamily::VibeVoiceTts => 96 + vibevoice_tts_auto_max_frames_for_text(&flood),
                // A preset voice prompt plus CUDA's frame maximum.
                ModelFamily::VoxtralTts => 256 + ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES,
                _ => {
                    assert_eq!(floor(variant), 0, "{variant}");
                    continue;
                }
            };
            assert!(
                floor(variant) >= required,
                "{variant}: floor {} < {required}",
                floor(variant)
            );
        }
    }

    /// R-1 for VibeVoice: every request returned 400 against a 64-token
    /// context. Both CFG domains must reach a prompt plus the largest
    /// automatic budget.
    #[test]
    fn vibevoice_tts_effective_context_covers_prompt_and_frame_budget_under_low_live_headroom() {
        use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
        let variant = ModelVariant::VibeVoice15BTts;
        let contract =
            crate::models::architectures::vibevoice::vibevoice_tts_retained_test_contract(
                candle_core::DType::F32,
            );
        // System prompt, a 12 s voice reference (90 latents plus markers)
        // and the target text that fills the largest automatic budget.
        let longest = "word ".repeat(1_000);
        let required = (30 + 96 + 200 + vibevoice_tts_auto_max_frames_for_text(&longest)) as u64;
        let maximum = 1_536;
        let budget = 11 * GIB + 440 * 1024 * 1024;
        let reach =
            load_tts_retained_state(variant, &contract, maximum, budget, GIB / 2, 6_000_000_000)
                .unwrap();
        assert!(reach >= required, "effective context {reach} < {required}");
        assert!(reach < maximum as u64, "live memory must still cap growth");
    }
}

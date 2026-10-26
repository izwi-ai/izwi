//! Runtime service orchestrator.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::FutureExt;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex, Notify, RwLock};
use tokio::task::yield_now;
use tracing::{debug, error, info_span, warn};

use crate::artifacts::{DownloadProgress, ModelLifecycleSnapshot, ModelManager};
use crate::audio::{AudioCodec, AudioEncoder};
use crate::backends::{
    BackendKind, BackendPreference, BackendRouter, BackendSelectionSource, RuntimeDeviceAssignment,
};
use crate::catalog::{ModelFamily, ModelInfo, ModelVariant};
use crate::config::{
    BatchSizePreference, ConcurrencyProfile, EngineConfig, PrefixCachePolicy, ResolvedConcurrency,
    ResolvedKvCachePolicy,
};
use crate::engine::metrics::{
    ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_WIDTH_CALLS_TOTAL,
    ENGINE_KV_MODEL_ADMISSION_AVAILABLE_PAGES, ENGINE_KV_MODEL_ADMISSION_CLAIMED_PAGES,
    ENGINE_KV_MODEL_ALLOCATED_PAGES, ENGINE_KV_MODEL_CAPACITY_PAGES,
    ENGINE_KV_MODEL_PREFIX_RETAINED_PAGES, ENGINE_KV_PREFIX_HIT_TOKENS_TOTAL,
    ENGINE_KV_PREFIX_QUERY_TOKENS_TOTAL, ENGINE_SCHEDULER_CAPACITY_REPLAY_TOKENS_TOTAL,
    ENGINE_SCHEDULER_CAPACITY_SUSPENSIONS_TOTAL, ENGINE_SCHEDULER_KV_BLOCKED_REQUESTS,
    ENGINE_SCHEDULER_PREEMPTIONS_TOTAL,
};
use crate::engine::{
    engine_batch_metrics_snapshot, engine_stream_metrics_snapshot, AdapterBindingKey,
    Engine as CoreEngine, EngineAudioInput, EngineCoreConfig, EngineCoreRequest, EngineOutput,
    EngineStreamPolicy, EngineTask, GenerationParams, OutputFinishReason, ResourceAmount,
    ResourceVector, SessionKey, StreamingOutput, TaskType, WorkUnit, WorkerConfig, WorkloadClass,
    ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL,
    ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL,
    ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL,
    ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL, ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL,
    ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL, ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL,
    ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL, ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL,
    ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH, ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL,
    ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL,
    ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL,
    ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO,
    ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH, ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO,
    ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL,
    ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL, ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
    ENGINE_KV_CACHE_EVICTIONS_TOTAL, ENGINE_KV_CACHE_FREE_BLOCKS,
    ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS, ENGINE_KV_CACHE_HITS_TOTAL,
    ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES, ENGINE_KV_CACHE_MEMORY_USED_BYTES,
    ENGINE_KV_CACHE_MISSES_TOTAL, ENGINE_KV_CACHE_UTILIZATION_RATIO, ENGINE_KV_DEMOTIONS_TOTAL,
    ENGINE_KV_HOST_PAGES, ENGINE_KV_PROMOTIONS_TOTAL, ENGINE_KV_PROMOTION_LATENCY_AVG_SECONDS,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_QUANTA_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_TOKENS_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_MULTISPAN_PREFILL_REQUESTS_TOTAL, ENGINE_SCHEDULER_QUEUE_DEPTH,
    ENGINE_SCHEDULER_RUNNING_REQUESTS, ENGINE_STREAM_BACKPRESSURE_TOTAL,
    ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL, ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL,
    ENGINE_STREAM_DELIVERY_FAILURES_TOTAL, REQUEST_DEADLINE_EXCEEDED,
};
use crate::error::{Error, Result};
use crate::model::ModelResidencyLease;
use crate::models::architectures::chatterbox::{ChatterboxGenerationParams, T3SamplingParams};
use crate::models::architectures::fish_s2::{FishS2GenerationParams, FishS2Reference};
use crate::models::architectures::granite_speech::asr::{
    GraniteSpeechPreparationBatchRow, GraniteSpeechPreparedGeometry,
    GraniteSpeechPreparedPromptArtifact,
};
use crate::models::architectures::kokoro::kokoro_output_budget;
use crate::models::architectures::qwen3::asr::{
    Qwen3AsrAudioBatchRow, Qwen3AsrAudioPreparationGeometry, Qwen3AsrPreparedAudio,
};
use crate::models::architectures::vibevoice::asr::{
    VibeVoiceAsrPreparationDecision, VibeVoiceAsrPreparedArtifact, VibeVoiceAsrPreparedGeometry,
};
use crate::models::architectures::vibevoice::tts::{
    vibevoice_tts_auto_max_frames_for_text, VibeVoiceSpeakerReference,
    VibeVoiceTtsGenerationParams, VIBEVOICE_TTS_DEFAULT_CFG_SCALE,
};
use crate::models::architectures::voxtral::tts::VoxtralTtsGenerationParams;
use crate::models::architectures::whisper::asr::{
    WhisperAudioBatchRow, WhisperPreparedWindow, WhisperWindowPreparationGeometry,
};
use crate::models::registry::AsrModelLease;
use crate::models::shared::chat::{ChatMessage, ChatRequestConfig};
use crate::runtime::adapters::{
    capability_row, contract_supports_atomic_blocking, CapabilityKind, ExecutionTargetKind,
    LoadedCapabilityBinding, LoadedExecutionContract, LoadedModelBundle, RuntimeAdapterRegistry,
    StreamingRequirements,
};
use crate::runtime::asr::RealtimeAsrSessionPolicy;
use crate::runtime::asr_no_speech::{asr_no_speech_gate_applies, AsrNoSpeechGate};
use crate::runtime::audio_io::decode_reference_audio_base64;
use crate::runtime::broker::{InferenceBroker, InferenceBrokerObservation};
use crate::runtime::coordinator::{
    CoordinatorLane, CoordinatorSnapshot, InferenceCoordinator, JobLease, JobResourceObservation,
    JobSpec, PreparationArtifact, PreparationCancellation, PreparationRowOutcome,
};
use crate::runtime::health::RuntimeHealth;
use crate::runtime::lifecycle::controller::ModelLifecycleController;
use crate::runtime::pipeline::{PipelineExecutor, PipelineGraph};
use crate::runtime::routing::RouteSource;
use crate::runtime::telemetry::{
    push_engine_labeled_metric, push_engine_labeled_metric_f64, push_engine_metric,
    push_engine_metric_f64, push_engine_multi_labeled_metric,
    push_engine_physical_execution_metrics, EngineRuntimeTelemetrySnapshot,
    RuntimeObservationContext, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters, RuntimeStageTiming, RuntimeTelemetryCollector,
    RuntimeTelemetrySnapshot, ENGINE_REQUEST_STAGE, ENGINE_STREAMING_REQUEST_STAGE,
};
use crate::runtime::types::{ChatGeneration, RuntimeRequestContext};
use crate::runtime_models::{LoadedModelDiagnostics, ModelRegistry};

#[cfg(test)]
use crate::backends::DeviceProfile;

mod asr_encoders;
mod asr_preparation;
mod chat_invocation;
mod driver;
mod execution;
mod lifecycle;
mod observability;
mod tts_preparation;
#[cfg(test)]
use asr_encoders::granite_speech_deadline_budget_bucket;
use asr_encoders::{
    GraniteSpeechEncoderBatcher, GraniteSpeechEncoderPending, QwenAsrEncoderBatcher,
    QwenAsrEncoderPending, VibeVoiceEncoderBatcher, VibeVoiceEncoderPending, WhisperEncoderBatcher,
    WhisperEncoderPending,
};
use asr_preparation::asr_encoder_retained_resources;
#[cfg(test)]
use asr_preparation::install_parakeet_decode_budget;
#[cfg(test)]
use chat_invocation::RuntimeChatInvocationDriver;
pub use chat_invocation::{
    stream_slow_consumer_timeout, RuntimeChatInvocation, RuntimeChatInvocationEvent,
    RuntimeChatInvocationRequest, RuntimeChatTeardown, RuntimeChatTeardownDisposition,
    DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT,
};
#[cfg(test)]
use driver::{
    cleanup_pending_request, remove_waiter_registration, route_terminal_output,
    DeferredRequestOwnership,
};
use driver::{
    PendingRequestGuard, RuntimeCompletionWaiters, StreamOutputOrder, WaiterRegistrationGuard,
};
use observability::{engine_error_observation, engine_output_observation};
#[cfg(test)]
use observability::{
    managed_kv_device_pages, managed_kv_used_bytes, push_managed_kv_arena_metrics,
};
#[cfg(test)]
use tts_preparation::{
    fish_s2_request_params, kokoro_synthesis_resources, lfm25_audio_tts_preparation_workspace,
    retained_lfm25_audio_tts_artifact_host_bytes, vibevoice_tts_request_params,
    voxtral_tts_request_params,
};

fn effective_physical_execution_parallelism(
    mode: crate::config::PhysicalExecutionMode,
    request_parallelism: usize,
    configured_launch_limit: usize,
) -> usize {
    if mode == crate::config::PhysicalExecutionMode::Concurrent {
        request_parallelism.max(configured_launch_limit).max(1)
    } else {
        // Serial is the emergency rollback path and Shadow must observe
        // decisions without allowing direct-capability calls to overlap.
        1
    }
}

/// Automatic staged-transaction rows. A sequence stages at most one managed
/// transaction, from batch formation until its step commits, and one engine
/// step launches at most `candidate_dispatch_limit` physical batches of at
/// most `tensor_width` rows (formation defers rows it cannot launch before
/// staging them). The memory-derived row ceiling bounds retained sessions
/// only; staged rows past the launch width are reserved state never used.
pub(crate) fn automatic_staged_transaction_rows(
    tensor_width: usize,
    candidate_dispatch_limit: usize,
    retained_rows: usize,
) -> usize {
    tensor_width
        .max(1)
        .saturating_mul(candidate_dispatch_limit.max(1))
        .min(retained_rows.max(1))
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

fn runtime_completion(output: EngineOutput) -> Result<EngineOutput> {
    if output.finish_reason == Some(OutputFinishReason::Aborted) {
        return Err(Error::Cancelled(output.request_id));
    }
    if let Some(err) = output.error.clone() {
        return Err(if err == REQUEST_DEADLINE_EXCEEDED {
            Error::Timeout(output.request_id)
        } else {
            Error::from_display(err)
        });
    }
    Ok(output)
}

pub(crate) fn transient_resources(backend: BackendKind, input_bytes: usize) -> ResourceVector {
    const BASE_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
    let input_bytes = input_bytes as u64;
    let host_preparation_bytes = BASE_WORKSPACE_BYTES.saturating_add(input_bytes.saturating_mul(8));
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(host_preparation_bytes),
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(host_preparation_bytes)
        }
        BackendKind::Cuda => {
            // Decode, resample, tokenization, media parsing, and request
            // construction all occur in host memory before CUDA upload. The
            // device scratch an execution needs is charged per row of each
            // dispatch (`engine::CUDA_DISPATCH_SCRATCH_BYTES_PER_ROW`), not
            // for the request's life: a queued request held 64 MiB of live
            // headroom it never allocated, so a burst past headroom / 64 MiB
            // failed admission with nothing on the device.
            resources.host_bytes = ResourceAmount::Known(host_preparation_bytes);
        }
    }
    resources
}

/// The working set of one ASR request for `variant` (PIPE-15), replacing the
/// generic "64 MiB + 8 × input" price: the request's retained input plus the
/// model's request envelope over `audio_secs` (one full chunk while the
/// length is still unknown, which bounds any length because longer audio is
/// chunked). The envelope's accelerator part is charged here only for the
/// families whose encoder workspace no preparation stage prices (Nemotron,
/// Voxtral); the others reserve it per physical batch when they dispatch.
/// `None` for a variant without an envelope, which keeps the generic price.
fn asr_transient_resources(
    backend: BackendKind,
    variant: ModelVariant,
    input_bytes: usize,
    audio_secs: Option<f32>,
) -> Result<Option<ResourceVector>> {
    use super::lifecycle::asr_envelope;
    let Some(secs) = audio_secs.or_else(|| asr_envelope::max_chunk_secs(variant)) else {
        return Ok(None);
    };
    let Some(envelope) = asr_envelope::request_envelope(variant, secs, backend)? else {
        return Ok(None);
    };
    let unpriced_encoder = matches!(
        variant.family(),
        ModelFamily::NemotronAsr | ModelFamily::Voxtral
    );
    let accelerator_bytes = if unpriced_encoder {
        envelope.accelerator_bytes
    } else {
        0
    };
    let host_bytes = (input_bytes as u64)
        .checked_add(envelope.host_bytes)
        .ok_or_else(|| Error::Overloaded("ASR request envelope overflow".into()))?;
    let overflow = || Error::Overloaded("ASR request envelope overflow".into());
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(
                host_bytes
                    .checked_add(accelerator_bytes)
                    .ok_or_else(overflow)?,
            )
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(
                host_bytes
                    .checked_add(accelerator_bytes)
                    .ok_or_else(overflow)?,
            )
        }
        BackendKind::Cuda => {
            // Dispatch scratch is charged per row of each dispatch; only an
            // encoder no preparation stage prices stays on the request.
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(accelerator_bytes);
        }
    }
    Ok(Some(resources))
}

const AUDIO_DECODE_WORKSPACE_BYTES: u64 = 256 * 1024 * 1024;
const PREPARATION_COPY_QUANTUM_BYTES: usize = 1024 * 1024;

pub(crate) fn audio_decode_resources(backend: BackendKind) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        }
        BackendKind::Cuda => {
            // Audio parsing, decoder packets, and mono output are host-side
            // even when inference itself runs on CUDA.
            resources.host_bytes = ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        }
    }
    resources
}

fn task_decodes_audio(task_type: TaskType) -> bool {
    matches!(task_type, TaskType::ASR | TaskType::SpeechToSpeech)
}

fn add_retained_allocation(
    total: &mut usize,
    capacity: usize,
    element_size: usize,
    label: &str,
) -> Result<()> {
    let bytes = capacity
        .checked_mul(element_size)
        .ok_or_else(|| Error::Overloaded(format!("{label} capacity overflow")))?;
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| Error::Overloaded(format!("{label} size overflow")))?;
    Ok(())
}

fn add_owned_string_capacity(total: &mut usize, value: &String, label: &str) -> Result<()> {
    add_retained_allocation(total, value.capacity(), std::mem::size_of::<u8>(), label)
}

fn add_optional_string_capacity(
    total: &mut usize,
    value: Option<&String>,
    label: &str,
) -> Result<()> {
    if let Some(value) = value {
        add_owned_string_capacity(total, value, label)?;
    }
    Ok(())
}

fn add_json_allocations(total: &mut usize, value: &serde_json::Value, label: &str) -> Result<()> {
    match value {
        serde_json::Value::String(value) => add_owned_string_capacity(total, value, label),
        serde_json::Value::Array(values) => {
            add_retained_allocation(
                total,
                values.capacity(),
                std::mem::size_of::<serde_json::Value>(),
                label,
            )?;
            for value in values {
                add_json_allocations(total, value, label)?;
            }
            Ok(())
        }
        serde_json::Value::Object(values) => {
            // serde_json does not expose the backing map capacity. Count only
            // allocations whose capacities are observable; any map-node
            // overhead remains pending instead of being falsely materialized.
            for (key, value) in values {
                add_owned_string_capacity(total, key, label)?;
                add_json_allocations(total, value, label)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn add_chat_messages_allocations(
    total: &mut usize,
    messages: &[ChatMessage],
    capacity: usize,
) -> Result<()> {
    add_retained_allocation(
        total,
        capacity,
        std::mem::size_of::<ChatMessage>(),
        "chat messages",
    )?;
    for message in messages {
        add_owned_string_capacity(total, &message.content, "chat message content")?;
    }
    Ok(())
}

fn add_chat_config_allocations(total: &mut usize, config: &ChatRequestConfig) -> Result<()> {
    add_retained_allocation(
        total,
        config.media_inputs.capacity(),
        std::mem::size_of::<crate::models::shared::chat::ChatMediaInput>(),
        "chat media inputs",
    )?;
    for media in &config.media_inputs {
        add_owned_string_capacity(total, &media.source, "chat media source")?;
    }
    add_retained_allocation(
        total,
        config.tools.capacity(),
        std::mem::size_of::<serde_json::Value>(),
        "chat tools",
    )?;
    for tool in &config.tools {
        add_json_allocations(total, tool, "chat tool")?;
    }
    Ok(())
}

fn add_generation_param_allocations(total: &mut usize, params: &GenerationParams) -> Result<()> {
    add_optional_string_capacity(total, params.speaker.as_ref(), "generation speaker")?;
    add_optional_string_capacity(total, params.voice.as_ref(), "generation voice")?;
    add_retained_allocation(
        total,
        params.stop_sequences.capacity(),
        std::mem::size_of::<String>(),
        "generation stop sequences",
    )?;
    for sequence in &params.stop_sequences {
        add_owned_string_capacity(total, sequence, "generation stop sequence")?;
    }
    add_retained_allocation(
        total,
        params.stop_token_ids.capacity(),
        std::mem::size_of::<u32>(),
        "generation stop token IDs",
    )
}

pub(crate) fn retained_chat_preparation_input_bytes(
    messages: &[ChatMessage],
    messages_capacity: usize,
    config: &ChatRequestConfig,
    params: &GenerationParams,
    correlation_id: Option<&String>,
) -> Result<usize> {
    let mut total = 0usize;
    add_chat_messages_allocations(&mut total, messages, messages_capacity)?;
    add_chat_config_allocations(&mut total, config)?;
    add_generation_param_allocations(&mut total, params)?;
    add_optional_string_capacity(&mut total, correlation_id, "chat correlation ID")?;
    Ok(total)
}

pub(crate) fn retained_speech_to_speech_preparation_input_bytes(
    audio_input_bytes: usize,
    messages: &[ChatMessage],
    messages_capacity: usize,
    params: &GenerationParams,
    system_prompt: Option<&str>,
    correlation_id: Option<&str>,
) -> Result<usize> {
    let mut total = 0usize;
    add_retained_allocation(
        &mut total,
        audio_input_bytes,
        std::mem::size_of::<u8>(),
        "speech-to-speech borrowed audio",
    )?;
    add_chat_messages_allocations(&mut total, messages, messages_capacity)?;
    add_generation_param_allocations(&mut total, params)?;
    add_retained_allocation(
        &mut total,
        system_prompt.map(str::len).unwrap_or_default(),
        std::mem::size_of::<u8>(),
        "speech-to-speech borrowed system prompt",
    )?;
    add_retained_allocation(
        &mut total,
        correlation_id.map(str::len).unwrap_or_default(),
        std::mem::size_of::<u8>(),
        "speech-to-speech borrowed correlation ID",
    )?;
    Ok(total)
}

fn add_audio_input_allocation(
    total: &mut usize,
    audio: &EngineAudioInput,
    label: &str,
) -> Result<()> {
    match audio {
        EngineAudioInput::Base64(value) => add_owned_string_capacity(total, value, label),
        EngineAudioInput::Bytes(value) => {
            add_retained_allocation(total, value.capacity(), std::mem::size_of::<u8>(), label)
        }
    }
}

pub(super) fn retained_engine_request_input_bytes(request: &EngineCoreRequest) -> Result<usize> {
    let mut total = 0usize;
    add_owned_string_capacity(&mut total, &request.id, "request ID")?;
    add_optional_string_capacity(&mut total, request.text.as_ref(), "request text")?;
    if let Some(messages) = request.chat_messages.as_ref() {
        add_chat_messages_allocations(&mut total, messages, messages.capacity())?;
    }
    add_chat_config_allocations(&mut total, &request.chat_config)?;
    add_optional_string_capacity(&mut total, request.language.as_ref(), "request language")?;
    add_optional_string_capacity(
        &mut total,
        request.correlation_id.as_ref(),
        "request correlation ID",
    )?;
    add_optional_string_capacity(
        &mut total,
        request.audio_input.as_ref(),
        "request base64 audio",
    )?;
    if let Some(audio) = request.audio_bytes.as_ref() {
        add_retained_allocation(
            &mut total,
            audio.capacity(),
            std::mem::size_of::<u8>(),
            "request audio bytes",
        )?;
    }
    for (value, label) in [
        (request.asr_prompt.as_ref(), "request ASR prompt"),
        (request.reference_audio.as_ref(), "request reference audio"),
        (request.reference_text.as_ref(), "request reference text"),
        (
            request.voice_description.as_ref(),
            "request voice description",
        ),
        (request.system_prompt.as_ref(), "request system prompt"),
    ] {
        add_optional_string_capacity(&mut total, value, label)?;
    }
    add_optional_string_capacity(
        &mut total,
        request.asr_task.translate_to.as_ref(),
        "request ASR translation target",
    )?;
    add_retained_allocation(
        &mut total,
        request.asr_task.keywords.capacity(),
        std::mem::size_of::<String>(),
        "request ASR keywords",
    )?;
    for keyword in &request.asr_task.keywords {
        add_owned_string_capacity(&mut total, keyword, "request ASR keyword")?;
    }
    add_generation_param_allocations(&mut total, &request.params)?;
    add_retained_allocation(
        &mut total,
        request.prompt_tokens.capacity(),
        std::mem::size_of::<u32>(),
        "request prompt tokens",
    )?;

    // The typed task owns separately cloned input buffers. Count them as
    // distinct materialized allocations rather than assuming the broad
    // compatibility fields share storage.
    match &request.task {
        EngineTask::Tts(input) => {
            add_owned_string_capacity(&mut total, &input.text, "typed TTS text")?;
            add_optional_string_capacity(
                &mut total,
                input.reference_audio.as_ref(),
                "typed TTS reference audio",
            )?;
            add_optional_string_capacity(
                &mut total,
                input.reference_text.as_ref(),
                "typed TTS reference text",
            )?;
            add_optional_string_capacity(
                &mut total,
                input.voice_description.as_ref(),
                "typed TTS voice description",
            )?;
        }
        EngineTask::Asr(input) => {
            add_audio_input_allocation(&mut total, &input.audio, "typed ASR audio")?;
            add_optional_string_capacity(
                &mut total,
                input.language.as_ref(),
                "typed ASR language",
            )?;
            add_optional_string_capacity(&mut total, input.prompt.as_ref(), "typed ASR prompt")?;
        }
        EngineTask::Chat(input) => {
            add_chat_messages_allocations(&mut total, &input.messages, input.messages.capacity())?;
            add_chat_config_allocations(&mut total, &input.chat_config)?;
            add_retained_allocation(
                &mut total,
                input.prompt_tokens.capacity(),
                std::mem::size_of::<u32>(),
                "typed chat prompt tokens",
            )?;
        }
        EngineTask::SpeechToSpeech(input) => {
            add_audio_input_allocation(&mut total, &input.audio, "typed speech-to-speech audio")?;
            add_chat_messages_allocations(&mut total, &input.messages, input.messages.capacity())?;
            add_optional_string_capacity(
                &mut total,
                input.system_prompt.as_ref(),
                "typed speech-to-speech system prompt",
            )?;
        }
    }

    total = total
        .checked_add(request.prepared_asr_audio_retained_bytes()?)
        .ok_or_else(|| Error::Overloaded("prepared ASR input storage overflowed".to_string()))?;
    // Device-resident prepared multimodal tensors remain excluded here. The
    // decoded Qwen3 ASR artifact above is host-owned and therefore counted
    // exactly by its immutable f32 slice length.
    Ok(total)
}

fn host_input_observation(input_bytes: usize) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::host(
        u64::try_from(input_bytes).map_err(|_| {
            Error::InvalidInput("runtime retained input size exceeds u64".to_string())
        })?,
    ))
}

fn retained_artifact_resources(
    backend: BackendKind,
    host_bytes: u64,
    accelerator_bytes: u64,
) -> Result<ResourceVector> {
    let total = host_bytes
        .checked_add(accelerator_bytes)
        .ok_or_else(|| Error::Overloaded("ASR encoder retained resource overflow".to_string()))?;
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(total),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(total),
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(accelerator_bytes);
        }
    }
    Ok(resources)
}

/// Fish's request lease owns host preparation buffers and the eventual artifact.
/// Codec tensor workspace is leased separately by the physical batch runner.
pub(super) fn fish_s2_preparation_resources(
    backend: BackendKind,
    request_bytes: u64,
    preparation_host_bytes: u64,
) -> Result<ResourceVector> {
    let host_bytes = request_bytes
        .checked_add(preparation_host_bytes)
        .ok_or_else(|| Error::Overloaded("Fish S2 preparation host reservation overflow".into()))?;
    retained_artifact_resources(backend, host_bytes, 0)
}

pub(super) fn fish_s2_artifact_resources(
    backend: BackendKind,
    artifact_bytes: u64,
) -> Result<ResourceVector> {
    retained_artifact_resources(backend, artifact_bytes, 0)
}

pub(super) fn add_fish_s2_artifact_to_admission(
    backend: BackendKind,
    spec: &mut JobSpec,
    observation: &mut JobResourceObservation,
    artifact_bytes: u64,
) -> Result<()> {
    let resources = spec
        .resources
        .checked_add(fish_s2_artifact_resources(backend, artifact_bytes)?)?;
    let host_bytes = observation
        .host_bytes
        .checked_add(artifact_bytes)
        .ok_or_else(|| Error::Overloaded("Fish S2 execution host observation overflow".into()))?;
    spec.resources = resources;
    observation.host_bytes = host_bytes;
    Ok(())
}

fn ensure_preparation_copy_deadline(job: &JobLease) -> Result<()> {
    if job
        .spec
        .deadline
        .is_some_and(|deadline| deadline <= Instant::now())
    {
        return Err(Error::Timeout(job.spec.request_id.clone()));
    }
    Ok(())
}

pub(crate) async fn copy_preparation_bytes(
    job: &JobLease,
    input: &[u8],
    label: &str,
) -> Result<Vec<u8>> {
    ensure_preparation_copy_deadline(job)?;
    let mut output = Vec::new();
    output.try_reserve_exact(input.len()).map_err(|_| {
        Error::Overloaded(format!(
            "unable to reserve {} bytes for {label}",
            input.len()
        ))
    })?;
    ensure_preparation_copy_deadline(job)?;
    for chunk in input.chunks(PREPARATION_COPY_QUANTUM_BYTES) {
        output.extend_from_slice(chunk);
        ensure_preparation_copy_deadline(job)?;
        tokio::task::yield_now().await;
        ensure_preparation_copy_deadline(job)?;
    }
    Ok(output)
}

pub(crate) async fn copy_preparation_string(
    job: &JobLease,
    input: &str,
    label: &str,
) -> Result<String> {
    ensure_preparation_copy_deadline(job)?;
    let mut output = String::new();
    output.try_reserve_exact(input.len()).map_err(|_| {
        Error::Overloaded(format!(
            "unable to reserve {} bytes for {label}",
            input.len()
        ))
    })?;
    ensure_preparation_copy_deadline(job)?;

    let mut start = 0usize;
    while start < input.len() {
        let mut end = start
            .saturating_add(PREPARATION_COPY_QUANTUM_BYTES)
            .min(input.len());
        while !input.is_char_boundary(end) {
            end -= 1;
        }
        output.push_str(&input[start..end]);
        start = end;
        ensure_preparation_copy_deadline(job)?;
        tokio::task::yield_now().await;
        ensure_preparation_copy_deadline(job)?;
    }
    Ok(output)
}

pub(crate) async fn copy_optional_preparation_string(
    job: &JobLease,
    input: Option<&str>,
    label: &str,
) -> Result<Option<String>> {
    match input {
        Some(input) => copy_preparation_string(job, input, label).await.map(Some),
        None => Ok(None),
    }
}

pub(crate) fn media_preparation_resources(
    backend: BackendKind,
    estimate: crate::models::architectures::qwen35::Qwen35MediaResourceEstimate,
) -> Result<ResourceVector> {
    let total_unified = estimate
        .host_bytes
        .checked_add(estimate.backend_tensor_bytes)
        .ok_or_else(|| Error::Overloaded("Qwen3.5 media resource overflow".to_string()))?;
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(total_unified),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(total_unified),
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(estimate.host_bytes);
            resources.device_bytes = ResourceAmount::Known(estimate.backend_tensor_bytes);
        }
    }
    Ok(resources)
}

/// Whether the capability table lets `capability` on `variant` run in the
/// coordinator's resumable lane.
fn resumable_capability(capability: CapabilityKind, variant: ModelVariant) -> bool {
    capability_row(capability, variant).is_some_and(|row| row.resumable)
}

fn coordinator_lane_for_metadata(
    task_type: TaskType,
    model_variant: Option<ModelVariant>,
    _streaming: bool,
    workload_class: WorkloadClass,
) -> CoordinatorLane {
    let sequence = model_variant.is_some_and(|variant| {
        resumable_capability(CapabilityKind::for_engine_task(task_type), variant)
    });
    if workload_class == WorkloadClass::Realtime {
        CoordinatorLane::Realtime
    } else if sequence {
        CoordinatorLane::Resumable
    } else {
        CoordinatorLane::Atomic
    }
}

fn coordinator_lane_for_request(request: &EngineCoreRequest) -> CoordinatorLane {
    if request.task_type == TaskType::ASR
        && request
            .model_variant
            .is_some_and(|variant| resumable_capability(CapabilityKind::Asr, variant))
        && request.prepared_asr_execution_shape().is_none()
    {
        // This admission owns only decoded-media preparation. It must not be
        // eligible for retained decoder execution before the exact route and
        // immutable audio-tower artifact are known.
        return CoordinatorLane::Atomic;
    }
    if request.workload_class != WorkloadClass::Realtime && request.uses_asr_long_form_atomic() {
        return CoordinatorLane::Atomic;
    }
    coordinator_lane_for_metadata(
        request.task_type,
        request.model_variant,
        request.streaming,
        request.workload_class,
    )
}

/// Keep decoded inputs covered when row sealing or dispatch fails. Field order
/// frees physical inputs before the final lease clone, including during unwind.
pub(super) struct PreparationOwnedInputs<T> {
    inputs: T,
    lease: JobLease,
}

impl<T> PreparationOwnedInputs<T> {
    pub(super) fn new(inputs: T, lease: JobLease) -> Self {
        Self { inputs, lease }
    }

    pub(super) fn run<R>(self, operation: impl FnOnce(T) -> Result<R>) -> Result<R> {
        let Self { inputs, lease } = self;
        let result = operation(inputs);
        drop(lease);
        result
    }
}

pub(super) fn release_failed_preparation<T>(
    failure: crate::runtime::coordinator::PreparationAdmissionFailure,
    inputs: T,
) -> Error {
    drop(inputs);
    drop(failure.bridge);
    failure.error
}

struct PreparationCancellationGuard {
    cancellation: PreparationCancellation,
    armed: bool,
}

impl Drop for PreparationCancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Main inference engine runtime.
pub struct RuntimeService {
    pub(crate) config: EngineConfig,
    cache_policy: ResolvedKvCachePolicy,
    pub(crate) backend_router: BackendRouter,
    pub(crate) inference_broker: InferenceBroker,
    pub(crate) adapter_registry: Arc<RuntimeAdapterRegistry>,
    pub(crate) model_manager: Arc<ModelManager>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) codec: Arc<RwLock<AudioCodec>>,
    pub(crate) core_engine: Arc<CoreEngine>,
    pub(crate) coordinator: Arc<InferenceCoordinator>,
    qwen_asr_encoder_batcher: Arc<QwenAsrEncoderBatcher>,
    whisper_encoder_batcher: Arc<WhisperEncoderBatcher>,
    vibevoice_encoder_batcher: Arc<VibeVoiceEncoderBatcher>,
    granite_speech_encoder_batcher: Arc<GraniteSpeechEncoderBatcher>,
    pub(super) asr_realtime_sessions: RealtimeAsrSessionPolicy,
    telemetry: Arc<RuntimeTelemetryCollector>,
    completion_waiters: Arc<RuntimeCompletionWaiters>,
    next_completion_waiter_registration: AtomicU64,
    step_driver_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    step_driver_wakeup: Arc<Notify>,
    step_driver_started: AtomicBool,
    /// Uploads the diarization pipeline decoded itself (PIPE-14): callers
    /// that pass decoded samples add none.
    pub(crate) diarization_audio_decodes: Arc<AtomicU64>,
    pub(crate) loaded_tts_variant: Arc<RwLock<Option<ModelVariant>>>,
    pub(crate) max_loaded_models: Option<usize>,
    pub(crate) model_lifecycle: Arc<ModelLifecycleController>,
    concurrency: ResolvedConcurrency,
}

pub(crate) struct AdmittedEngineRequest {
    request: EngineCoreRequest,
    job: JobLease,
    residency_lease: ModelResidencyLease,
}

fn bind_request_to_residency(
    request: &mut EngineCoreRequest,
    residency_lease: Option<&ModelResidencyLease>,
    loaded_bundle: Option<&LoadedModelBundle>,
    model_streaming_required: bool,
) -> Result<()> {
    let Some(lease) = residency_lease else {
        return Ok(());
    };
    if request.model_variant != Some(lease.variant()) {
        return Err(Error::InvalidInput(
            "engine request model does not match its residency lease".to_string(),
        ));
    }
    let Some(model_instance_id) = lease.model_instance_id() else {
        return Ok(());
    };
    let bundle = loaded_bundle.ok_or_else(|| {
        Error::InferenceError(
            "authoritative model residency is missing its loaded execution bundle".to_string(),
        )
    })?;
    if bundle.model_variant() != lease.variant() || bundle.model_instance_id() != model_instance_id
    {
        return Err(Error::InferenceError(
            "loaded execution bundle does not match authoritative model residency".to_string(),
        ));
    }
    if request.model_variant.is_some_and(|variant| {
        matches!(
            variant.family(),
            ModelFamily::Qwen3Asr
                | ModelFamily::WhisperAsr
                | ModelFamily::VibeVoiceAsr
                | ModelFamily::GraniteSpeechAsr
                | ModelFamily::GraniteSpeech5Asr
                | ModelFamily::Lfm25Audio
        ) && request.prepared_asr_execution_shape().is_some()
    }) && request.prepared_asr_audio_for_executor()?.is_none()
    {
        return Err(Error::InvalidInput(
            "ASR execution shape has no matching decoded-audio artifact".to_string(),
        ));
    }
    if request.model_variant.is_some_and(|variant| {
        variant.family() == ModelFamily::WhisperAsr
            && request.prepared_asr_execution_shape().is_some()
            && !request.uses_asr_long_form_atomic()
    }) && request.prepared_whisper_window_for_executor()?.is_none()
    {
        return Err(Error::InvalidInput(
            "Whisper normal execution shape has no matching prepared window".into(),
        ));
    }
    if request.model_variant.is_some_and(|variant| {
        variant.family() == ModelFamily::GraniteSpeechAsr
            && request.prepared_asr_execution_shape().is_some()
            && !request.uses_asr_long_form_atomic()
    }) && request
        .prepared_granite_speech_artifact_for_executor()?
        .is_none()
    {
        return Err(Error::InvalidInput(
            "Granite Speech normal execution shape has no matching prepared artifact".into(),
        ));
    }
    if request.model_variant.is_some_and(|variant| {
        variant.family() == ModelFamily::VibeVoiceAsr
            && request.prepared_asr_execution_shape().is_some()
            && !request.uses_asr_long_form_atomic()
    }) && request
        .prepared_vibevoice_artifact_for_executor()?
        .is_none()
    {
        return Err(Error::InvalidInput(
            "VibeVoice normal execution shape has no matching prepared artifact".into(),
        ));
    }
    if request.model_variant.is_some_and(|variant| {
        variant.family() == ModelFamily::Lfm25Audio
            && request.prepared_asr_execution_shape().is_some()
            && !request.uses_asr_long_form_atomic()
    }) && request
        .prepared_lfm25_audio_asr_artifact_for_executor()?
        .is_none()
    {
        return Err(Error::InvalidInput(
            "LFM2.5 Audio normal execution shape has no matching prepared artifact".into(),
        ));
    }
    let streaming = if request.streaming && !model_streaming_required {
        StreamingRequirements::transport_only()
    } else {
        StreamingRequirements::native(model_streaming_required)
    }
    .with_asr_long_form(request.uses_asr_long_form_atomic());
    let LoadedCapabilityBinding { execution, state } = bundle.capability_binding_for_streaming(
        CapabilityKind::for_engine_task(request.task_type),
        streaming,
    )?;
    request.bind_execution_adapter(execution)?;
    request.bind_v2_state_runtime(
        state.clone(),
        state.state_fingerprint,
        bundle.backend_kind(),
    )?;
    Ok(())
}

/// Requirements for loading a capability for scheduled job work.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CapabilityLoadOptions {
    /// Request the adapter's long-form atomic graph. Required to schedule
    /// blocking `AtomicJob` work (pipeline stages) against the loaded binding;
    /// sequence-graph loads keep the default.
    pub(crate) asr_long_form: bool,
}

fn loaded_contract_for_residency(
    lease: &ModelResidencyLease,
    bundle: Option<&LoadedModelBundle>,
    capability: CapabilityKind,
    streaming_required: bool,
    asr_long_form: bool,
    execution_group_id: crate::engine::ExecutionGroupId,
    backend_kind: BackendKind,
    expected_target: Option<ExecutionTargetKind>,
) -> Result<LoadedExecutionContract> {
    let model_instance_id = lease.model_instance_id().ok_or_else(|| {
        Error::InferenceError(
            "model residency lease has no authoritative load generation".to_string(),
        )
    })?;
    let bundle = bundle.ok_or_else(|| {
        Error::InferenceError(
            "authoritative model residency is missing its loaded execution bundle".to_string(),
        )
    })?;
    if bundle.model_variant() != lease.variant()
        || bundle.model_instance_id() != model_instance_id
        || bundle.execution_group_id() != execution_group_id
        || bundle.backend_kind() != backend_kind
    {
        return Err(Error::InferenceError(
            "loaded execution bundle does not match authoritative runtime residency".to_string(),
        ));
    }
    let streaming = StreamingRequirements::native(streaming_required)
        .with_asr_long_form(asr_long_form);
    let contract = bundle.contract_for_streaming(capability, streaming)?;
    if contract.execution_group_id != execution_group_id
        || contract.model_instance_id != model_instance_id
        || contract.metadata.model_variant != lease.variant()
        || contract.execution_profile.backend != backend_kind
        || !contract.execution_profile.resolved_from_loaded_model
    {
        return Err(Error::InferenceError(
            "loaded capability contract does not match its runtime execution identity".to_string(),
        ));
    }
    if expected_target.is_some_and(|target| contract.metadata.execution_target != target) {
        return Err(Error::InvalidInput(format!(
            "loaded capability {:?} for {} targets {:?}, not {:?}",
            capability,
            lease.variant(),
            contract.metadata.execution_target,
            expected_target.expect("checked as some")
        )));
    }
    if asr_long_form && !contract_supports_atomic_blocking(&contract) {
        return Err(Error::InvalidInput(format!(
            "loaded capability {:?} for {} does not support atomic blocking execution",
            capability,
            lease.variant(),
        )));
    }
    let state_binding = bundle.capability_binding_for_streaming(
        capability,
        StreamingRequirements::native(streaming_required)
            .with_asr_long_form(asr_long_form),
    )?;
    state_binding
        .state
        .validate_against(backend_kind, &state_binding.execution)?;
    Ok(contract)
}

fn loaded_binding_for_residency(
    lease: &ModelResidencyLease,
    bundle: Option<&LoadedModelBundle>,
    capability: CapabilityKind,
    streaming_required: bool,
    asr_long_form: bool,
    execution_group_id: crate::engine::ExecutionGroupId,
    backend_kind: BackendKind,
    expected_target: Option<ExecutionTargetKind>,
) -> Result<(LoadedExecutionContract, LoadedCapabilityBinding)> {
    let contract = loaded_contract_for_residency(
        lease,
        bundle,
        capability,
        streaming_required,
        asr_long_form,
        execution_group_id,
        backend_kind,
        expected_target,
    )?;
    let bundle = bundle.expect("validated by loaded_contract_for_residency");
    let binding = bundle.capability_binding_for_streaming(
        capability,
        StreamingRequirements::native(streaming_required)
            .with_asr_long_form(asr_long_form),
    )?;
    if binding.execution != contract.adapter_binding()? {
        return Err(Error::InferenceError(
            "loaded capability state binding does not match its execution contract".into(),
        ));
    }
    binding
        .state
        .validate_against(backend_kind, &binding.execution)?;
    Ok((contract, binding))
}

impl RuntimeService {
}

fn positive_usize_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn requested_backend_unavailable_message(
    backend_context: &crate::backends::BackendContext,
) -> String {
    let requested = backend_context.preference.as_str();
    let selected = backend_context.backend_kind.as_str();

    if backend_context.preference == BackendPreference::Cuda {
        let detail = if backend_context.capabilities.cuda_compiled {
            "CUDA support is compiled in, but no usable CUDA device was selected"
        } else {
            "this runtime is not compiled with CUDA support"
        };

        return format!(
            "CUDA backend was requested, but the selected backend is `{selected}`. {detail}. Use `izwi status --detailed` or `/v1/health` to inspect CUDA runtime diagnostics."
        );
    }

    format!(
        "Requested backend `{requested}` is not available on this runtime (selected `{selected}`)"
    )
}

/// Page capacity the startup cache-policy check assumes. CPU and Metal keep
/// the portable baseline (`EngineCoreConfig::default().max_blocks`). CUDA
/// sizes its logical page cap for one maximum-length request plus the prefix
/// budget (`EngineCoreConfig::apply_backend_context_capacity`) and fits the
/// arena itself from device memory, so the baseline must not clamp a CUDA
/// prefix budget: chat history reuse needs a budget as long as the
/// conversations it serves, and retained prefix pages live inside the fitted
/// arena, evicted when a request needs them.
fn startup_policy_capacity_pages(backend: BackendKind, config: &EngineConfig) -> usize {
    let portable = EngineCoreConfig::default().max_blocks;
    if backend != BackendKind::Cuda {
        return portable;
    }
    let request_pages = config
        .portable_context_ceiling()
        .max(1)
        .div_ceil(config.kv_page_size.max(1));
    portable.max(request_pages.saturating_add(config.prefix_cache_pages_or_default()))
}

/// The prefix page budget the engine core receives: zero when reuse is
/// disabled, an operator's budget as the startup check resolved it (possibly
/// clamped), and unset when the operator left it unset, so that each model
/// sizes its prefix index from its own KV arena at load (`prefix_index_pages`
/// in the managed cache) from the same default floor.
fn engine_core_prefix_cache_pages(
    configured: Option<usize>,
    effective: &PrefixCachePolicy,
) -> Option<usize> {
    match effective {
        PrefixCachePolicy::Disabled => Some(0),
        PrefixCachePolicy::Namespaced { max_pages, .. } => configured.map(|_| *max_pages),
    }
}

#[cfg(test)]
mod startup_policy_capacity_tests {
    use super::{engine_core_prefix_cache_pages, startup_policy_capacity_pages};
    use crate::backends::BackendKind;
    use crate::config::{EngineConfig, PrefixCachePolicy};

    /// A 4,096-page prefix budget (one 262,144-token conversation) survives
    /// the CUDA startup check unclamped; CPU and Metal still clamp it to the
    /// portable baseline minus one request.
    #[test]
    fn cuda_startup_check_keeps_the_configured_prefix_budget() {
        let config = EngineConfig {
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("tenant".into()),
            max_prefix_cache_pages: Some(4_096),
            kv_page_size: 64,
            ..EngineConfig::default()
        };
        let pages = |backend| match config
            .resolved_kv_cache_policy(startup_policy_capacity_pages(backend, &config))
            .unwrap()
            .effective
            .prefix
        {
            PrefixCachePolicy::Namespaced { max_pages, .. } => max_pages,
            PrefixCachePolicy::Disabled => 0,
        };
        assert_eq!(pages(BackendKind::Cuda), 4_096);
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(pages(backend), 1_024 - 4_096 / 64, "{backend:?}");
        }
        // The shipped default budget is unchanged everywhere.
        let default = EngineConfig {
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("tenant".into()),
            ..EngineConfig::default()
        };
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(
                startup_policy_capacity_pages(backend, &default),
                1_024,
                "{backend:?}"
            );
        }
    }

    /// The engine core receives an unset prefix budget unset (each model
    /// sizes its own at load), an operator's budget as the startup check
    /// resolved it, and zero when reuse is disabled. An explicit zero still
    /// fails enabled reuse at startup, as before.
    #[test]
    fn engine_core_receives_an_unset_prefix_budget_unset() {
        let config = |pages| EngineConfig {
            enable_prefix_caching: true,
            managed_prefix_cache_salt: Some("tenant".into()),
            max_prefix_cache_pages: pages,
            kv_page_size: 64,
            ..EngineConfig::default()
        };
        let core_pages = |config: &EngineConfig, backend| {
            let policy = config
                .resolved_kv_cache_policy(startup_policy_capacity_pages(backend, config))
                .unwrap();
            engine_core_prefix_cache_pages(config.max_prefix_cache_pages, &policy.effective.prefix)
        };
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(core_pages(&config(None), backend), None, "{backend:?}");
            let zero = config(Some(0));
            assert!(zero
                .resolved_kv_cache_policy(startup_policy_capacity_pages(backend, &zero))
                .is_err());
        }
        assert_eq!(
            core_pages(&config(Some(4_096)), BackendKind::Cuda),
            Some(4_096)
        );
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(core_pages(&config(Some(4_096)), backend), Some(960));
        }
        for configured in [None, Some(0), Some(128)] {
            assert_eq!(
                engine_core_prefix_cache_pages(configured, &PrefixCachePolicy::Disabled),
                Some(0)
            );
        }
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;

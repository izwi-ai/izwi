//! Runtime service orchestrator.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::FutureExt;
use tokio::sync::{broadcast, oneshot, Mutex, Notify, RwLock};
use tokio::task::yield_now;
use tracing::{debug, error, info_span, warn};

use crate::artifacts::{DownloadProgress, ModelLifecycleSnapshot, ModelManager};
use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{
    BackendKind, BackendPreference, BackendRouter, BackendSelectionSource, DeviceProfile,
};
use crate::catalog::{ModelFamily, ModelInfo, ModelVariant};
use crate::config::{EngineConfig, PrefixCachePolicy, ResolvedKvCachePolicy};
use crate::engine::{
    engine_batch_metrics_snapshot, engine_stream_metrics_snapshot, Engine as CoreEngine,
    EngineAudioInput, EngineCoreConfig, EngineCoreRequest, EngineOutput, EngineStreamPolicy,
    EngineTask, GenerationParams, OutputFinishReason, ResourceAmount, ResourceVector, SessionKey,
    StreamingOutput, TaskType, WorkerConfig, WorkloadClass,
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
    ENGINE_KV_CACHE_MISSES_TOTAL, ENGINE_KV_CACHE_UTILIZATION_RATIO,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_QUANTA_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_INCREMENTAL_PREFILL_TOKENS_COMMITTED_TOTAL,
    ENGINE_SCHEDULER_MULTISPAN_PREFILL_REQUESTS_TOTAL, ENGINE_SCHEDULER_QUEUE_DEPTH,
    ENGINE_SCHEDULER_RUNNING_REQUESTS, ENGINE_STREAM_BACKPRESSURE_TOTAL,
    ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL, ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL,
    ENGINE_STREAM_DELIVERY_FAILURES_TOTAL, REQUEST_DEADLINE_EXCEEDED,
};
use crate::error::{Error, Result};
use crate::model::ModelResidencyLease;
use crate::models::shared::chat::{ChatMessage, ChatRequestConfig};
use crate::runtime::adapters::{
    CapabilityKind, ExecutionTargetKind, LoadedCapabilityBinding, LoadedExecutionContract,
    LoadedModelBundle, RuntimeAdapterRegistry, StreamingRequirements,
};
use crate::runtime::asr::RealtimeAsrSessionPolicy;
use crate::runtime::broker::{
    InferenceBroker, InferenceBrokerObservation, InferenceBrokerSnapshot,
};
use crate::runtime::coordinator::{
    CoordinatorLane, CoordinatorSnapshot, InferenceCoordinator, JobLease, JobResourceObservation,
    JobSpec,
};
use crate::runtime::lifecycle::controller::ModelLifecycleController;
use crate::runtime::pipeline::{PipelineExecutor, PipelineGraph};
use crate::runtime::routing::RouteSource;
use crate::runtime::telemetry::{
    push_engine_labeled_metric, push_engine_labeled_metric_f64, push_engine_metric,
    push_engine_metric_f64, push_engine_physical_execution_metrics, EngineRuntimeTelemetrySnapshot,
    RuntimeObservationContext, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters, RuntimeStageTiming, RuntimeTelemetryCollector,
    RuntimeTelemetrySnapshot,
};
use crate::runtime::types::RuntimeRequestContext;
use crate::runtime_models::{LoadedModelDiagnostics, ModelRegistry};
use crate::tokenizer::Tokenizer;

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
            Error::InferenceError(err)
        });
    }
    Ok(output)
}

#[derive(Debug)]
struct StreamOutputOrder {
    last_sequence: Option<usize>,
    final_seen: bool,
    allow_gaps: bool,
}

impl StreamOutputOrder {
    fn new(policy: EngineStreamPolicy) -> Self {
        Self {
            last_sequence: None,
            final_seen: false,
            allow_gaps: policy == EngineStreamPolicy::DropNewest,
        }
    }

    fn observe(&mut self, request_id: &str, chunk: &StreamingOutput) -> Result<()> {
        if chunk.request_id != request_id {
            return Err(Error::InferenceError(format!(
                "stream output for {request_id} carried request ID {}",
                chunk.request_id
            )));
        }
        if self.final_seen {
            return Err(Error::InferenceError(format!(
                "stream output for {request_id} arrived after its final marker"
            )));
        }
        if self
            .last_sequence
            .is_some_and(|last| chunk.sequence <= last)
        {
            return Err(Error::InferenceError(format!(
                "stream output sequence {} for {request_id} was not greater than its predecessor",
                chunk.sequence
            )));
        }
        let expected = self.last_sequence.map_or(0, |last| last.saturating_add(1));
        if !self.allow_gaps && chunk.sequence != expected {
            return Err(Error::InferenceError(format!(
                "stream output sequence {} for {request_id} did not match expected {expected}",
                chunk.sequence
            )));
        }
        self.last_sequence = Some(chunk.sequence);
        self.final_seen = chunk.is_final;
        Ok(())
    }

    fn require_final(&self, request_id: &str) -> Result<()> {
        if self.final_seen {
            Ok(())
        } else {
            Err(Error::InferenceError(format!(
                "stream for {request_id} closed without a final marker"
            )))
        }
    }
}

struct RuntimeCompletionWaiter {
    registration_id: u64,
    session_epoch: Option<u64>,
    sender: oneshot::Sender<Result<EngineOutput>>,
}

type RuntimeCompletionWaiters = Mutex<HashMap<String, RuntimeCompletionWaiter>>;

async fn remove_waiter_registration(
    waiters: &RuntimeCompletionWaiters,
    request_id: &str,
    registration_id: u64,
) -> bool {
    let mut waiters = waiters.lock().await;
    if waiters
        .get(request_id)
        .is_some_and(|waiter| waiter.registration_id == registration_id)
    {
        waiters.remove(request_id);
        true
    } else {
        false
    }
}

async fn bind_waiter_registration(
    waiters: &RuntimeCompletionWaiters,
    request_id: &str,
    registration_id: u64,
    session_epoch: u64,
) -> bool {
    let mut waiters = waiters.lock().await;
    let Some(waiter) = waiters.get_mut(request_id) else {
        return false;
    };
    if waiter.registration_id != registration_id || waiter.session_epoch.is_some() {
        return false;
    }
    waiter.session_epoch = Some(session_epoch);
    true
}

async fn route_terminal_output(
    engine: &CoreEngine,
    waiters: &RuntimeCompletionWaiters,
    telemetry: &RuntimeTelemetryCollector,
    output: EngineOutput,
) {
    debug_assert!(output.is_finished);

    // Removing the waiter is the routing hand-off. A missing waiter or a
    // dropped receiver still means there is no live runtime consumer left, so
    // either case must release the exact-session delivery fence.
    let waiter = loop {
        let mut waiters = waiters.lock().await;
        match waiters.get(&output.request_id) {
            Some(waiter) if waiter.session_epoch == Some(output.sequence_id) => {
                break waiters.remove(&output.request_id);
            }
            // A registration is installed before its engine session exists.
            // Let admission bind it before deciding whether this terminal
            // belongs to that waiter; otherwise an old output can steal a
            // newly reused public request ID.
            Some(waiter) if waiter.session_epoch.is_none() => {
                drop(waiters);
                tokio::task::yield_now().await;
            }
            // A waiter for a later exact session must remain registered.
            Some(_) | None => break None,
        }
    };
    if let Some(waiter) = waiter {
        let _ = waiter.sender.send(runtime_completion(output.clone()));
    }

    // Do not make the public request ID reusable until the runtime has
    // attempted terminal delivery to the waiter selected above.
    if !engine.acknowledge_dispatched_terminal(&output).await {
        warn!(
            request_id = %output.request_id,
            session_epoch = output.sequence_id,
            "Runtime terminal output had no matching delivery fence"
        );
    }
    telemetry.record_request_finished(&output).await;
}

fn managed_kv_used_bytes(snapshot: &crate::engine::ManagedKvRuntimeSnapshot) -> u64 {
    snapshot
        .models
        .iter()
        .flat_map(|model| &model.arenas)
        .map(|arena| {
            arena
                .coordinator
                .allocated_pages
                .saturating_mul(arena.bytes_per_page)
        })
        .fold(0_u64, u64::saturating_add)
}

fn managed_kv_device_pages(snapshot: &crate::engine::ManagedKvRuntimeSnapshot) -> u64 {
    snapshot
        .models
        .iter()
        .filter(|model| model.backend != BackendKind::Cpu)
        .flat_map(|model| &model.arenas)
        .map(|arena| arena.coordinator.allocated_pages)
        .fold(0_u64, u64::saturating_add)
}

fn transient_resources(backend: BackendKind, input_bytes: usize) -> ResourceVector {
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
            // construction all occur in host memory before CUDA upload. Keep
            // the input expansion in the host domain and reserve only the
            // backend-neutral execution workspace in VRAM.
            resources.host_bytes = ResourceAmount::Known(host_preparation_bytes);
            resources.device_bytes = ResourceAmount::Known(BASE_WORKSPACE_BYTES);
        }
    }
    resources
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

fn coordinator_lane_for_metadata(
    task_type: TaskType,
    model_variant: Option<ModelVariant>,
    _streaming: bool,
    workload_class: WorkloadClass,
) -> CoordinatorLane {
    let sequence = model_variant.is_some_and(|variant| match task_type {
        TaskType::Chat => {
            matches!(
                variant.family(),
                ModelFamily::Qwen35Chat | ModelFamily::Qwen38Chat
            ) || matches!(
                variant,
                ModelVariant::Qwen306B
                    | ModelVariant::Qwen306B4Bit
                    | ModelVariant::Qwen317B
                    | ModelVariant::Qwen317B4Bit
            )
        }
        TaskType::ASR => variant.family() == ModelFamily::Qwen3Asr,
        TaskType::TTS => variant.family() == ModelFamily::Qwen3Tts,
        TaskType::SpeechToSpeech => false,
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

/// Main inference engine runtime.
pub struct RuntimeService {
    pub(crate) config: EngineConfig,
    cache_policy: ResolvedKvCachePolicy,
    pub(crate) backend_router: BackendRouter,
    pub(crate) inference_broker: InferenceBroker,
    pub(crate) adapter_registry: Arc<RuntimeAdapterRegistry>,
    pub(crate) model_manager: Arc<ModelManager>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    pub(crate) codec: Arc<RwLock<AudioCodec>>,
    #[allow(dead_code)]
    pub(crate) streaming_config: StreamingConfig,
    pub(crate) core_engine: Arc<CoreEngine>,
    pub(crate) coordinator: Arc<InferenceCoordinator>,
    pub(super) asr_realtime_sessions: RealtimeAsrSessionPolicy,
    telemetry: Arc<RuntimeTelemetryCollector>,
    completion_waiters: Arc<RuntimeCompletionWaiters>,
    next_completion_waiter_registration: AtomicU64,
    step_driver_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    step_driver_wakeup: Arc<Notify>,
    step_driver_started: AtomicBool,
    pub(crate) loaded_tts_variant: Arc<RwLock<Option<ModelVariant>>>,
    pub(crate) max_loaded_models: Option<usize>,
    pub(crate) model_lifecycle: Arc<ModelLifecycleController>,
    pub(crate) device: DeviceProfile,
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
        variant.family() == ModelFamily::Qwen3Asr
            && request.prepared_asr_execution_shape().is_some()
    }) && request.prepared_asr_audio_for_executor()?.is_none()
    {
        return Err(Error::InvalidInput(
            "Qwen3 ASR execution shape has no matching decoded-audio artifact".to_string(),
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

fn loaded_contract_for_residency(
    lease: &ModelResidencyLease,
    bundle: Option<&LoadedModelBundle>,
    capability: CapabilityKind,
    streaming_required: bool,
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
    let contract = bundle.contract(capability, streaming_required)?;
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
    let state_binding = bundle.capability_binding_for_streaming(
        capability,
        StreamingRequirements::native(streaming_required),
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
    execution_group_id: crate::engine::ExecutionGroupId,
    backend_kind: BackendKind,
    expected_target: Option<ExecutionTargetKind>,
) -> Result<(LoadedExecutionContract, LoadedCapabilityBinding)> {
    let contract = loaded_contract_for_residency(
        lease,
        bundle,
        capability,
        streaming_required,
        execution_group_id,
        backend_kind,
        expected_target,
    )?;
    let bundle = bundle.expect("validated by loaded_contract_for_residency");
    let binding = bundle.capability_binding_for_streaming(
        capability,
        StreamingRequirements::native(streaming_required),
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

struct WaiterRegistrationGuard {
    request_id: String,
    registration_id: u64,
    completion_waiters: Arc<RuntimeCompletionWaiters>,
    active: bool,
}

impl WaiterRegistrationGuard {
    fn new(
        request_id: String,
        registration_id: u64,
        completion_waiters: Arc<RuntimeCompletionWaiters>,
    ) -> Self {
        Self {
            request_id,
            registration_id,
            completion_waiters,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for WaiterRegistrationGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let request_id = self.request_id.clone();
        let registration_id = self.registration_id;
        let waiters = self.completion_waiters.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                remove_waiter_registration(waiters.as_ref(), &request_id, registration_id).await;
            });
        }
    }
}

struct PendingRequestGuard {
    session: SessionKey,
    core_engine: Arc<CoreEngine>,
    completion_waiters: Arc<RuntimeCompletionWaiters>,
    waiter_registration_id: u64,
    telemetry: Arc<RuntimeTelemetryCollector>,
    job: Option<JobLease>,
    residency_lease: Option<ModelResidencyLease>,
    active: bool,
}

/// Owns cancellation-sensitive admission state after its request future is
/// gone. Dropping the detached cleanup task (runtime shutdown, panic, or task
/// abortion) deliberately leaks these leases instead of allowing a model or
/// resource reservation to disappear underneath native work that was never
/// proven stopped.
struct DeferredRequestOwnership {
    job: Option<JobLease>,
    residency_lease: Option<ModelResidencyLease>,
    release_confirmed: bool,
}

impl DeferredRequestOwnership {
    fn new(job: Option<JobLease>, residency_lease: Option<ModelResidencyLease>) -> Self {
        Self {
            job,
            residency_lease,
            release_confirmed: false,
        }
    }

    fn release(mut self) {
        self.release_confirmed = true;
    }
}

impl Drop for DeferredRequestOwnership {
    fn drop(&mut self) {
        if self.release_confirmed {
            return;
        }
        if let Some(residency_lease) = self.residency_lease.take() {
            std::mem::forget(residency_lease);
        }
        if let Some(job) = self.job.take() {
            std::mem::forget(job);
        }
    }
}

async fn cleanup_pending_request(
    session: SessionKey,
    engine: Arc<CoreEngine>,
    waiters: Arc<RuntimeCompletionWaiters>,
    waiter_registration_id: u64,
    telemetry: Arc<RuntimeTelemetryCollector>,
    ownership: DeferredRequestOwnership,
) {
    remove_waiter_registration(
        waiters.as_ref(),
        &session.request_id,
        waiter_registration_id,
    )
    .await;

    let aborted = match engine.abort_request_session(&session).await {
        Ok(aborted) => aborted,
        Err(err) => {
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                error = %err,
                "Exact-session cancellation cleanup failed; retaining request ownership"
            );
            return;
        }
    };
    // The exact abort waits for any in-flight engine step and its first
    // physical cleanup attempt. Only that successful hand-off permits the
    // model and coordinator admission to be released.
    ownership.release();
    if aborted {
        telemetry
            .record_request_cancelled(&session.request_id)
            .await;
    }
}

impl PendingRequestGuard {
    fn new(
        session: SessionKey,
        core_engine: Arc<CoreEngine>,
        completion_waiters: Arc<RuntimeCompletionWaiters>,
        waiter_registration_id: u64,
        telemetry: Arc<RuntimeTelemetryCollector>,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
    ) -> Self {
        Self {
            session,
            core_engine,
            completion_waiters,
            waiter_registration_id,
            telemetry,
            job: Some(job),
            residency_lease,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
        self.job.take();
        self.residency_lease.take();
    }

    /// Transfer cancellation cleanup to a detached exact-session task without
    /// waiting for the engine core lock. Streaming callbacks execute outside
    /// the engine, so a failed or timed-out transport must be able to return
    /// promptly even while a native step still owns that lock.
    fn defer_cleanup(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        let session = self.session.clone();
        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let waiter_registration_id = self.waiter_registration_id;
        let telemetry = self.telemetry.clone();
        let job = self.job.take();
        let residency_lease = self.residency_lease.take();
        let ownership = DeferredRequestOwnership::new(job, residency_lease);

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(cleanup_pending_request(
                session,
                engine,
                waiters,
                waiter_registration_id,
                telemetry,
                ownership,
            ));
        } else {
            drop(ownership);
        }
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        self.defer_cleanup();
    }
}

impl RuntimeService {
    pub fn backend_context(&self) -> crate::backends::BackendContext {
        self.backend_router.context().clone()
    }

    fn ensure_requested_backend_available(
        backend_context: &crate::backends::BackendContext,
    ) -> Result<()> {
        if backend_context.matches_preference() {
            return Ok(());
        }

        Err(Error::InferenceError(
            requested_backend_unavailable_message(backend_context),
        ))
    }

    /// Create a new inference engine.
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Reject unsupported or unsafe cache policy before any model registry,
        // device arena, or readiness state can be created.
        let cache_policy =
            config.resolved_kv_cache_policy(EngineCoreConfig::default().max_blocks)?;
        configure_runtime_threading(config.num_threads.max(1));
        let model_manager = Arc::new(ModelManager::new(config.clone())?);

        let backend_context =
            BackendRouter::resolve_context(config.backend, BackendSelectionSource::Config);
        let device = backend_context.device.clone();
        Self::ensure_requested_backend_available(&backend_context)?;
        let selected_backend_kind = backend_context.backend_kind;

        let model_registry = Arc::new(ModelRegistry::new(
            config.models_dir.clone(),
            device.clone(),
        ));

        let mut core_config = EngineCoreConfig::for_qwen3_tts();
        core_config.portable_context_auto = config.max_sequence_length.explicit_tokens().is_none();
        core_config.portable_context_reserve_bytes = config.portable_context_reserve_bytes;
        core_config.models_dir = config.models_dir.clone();
        core_config.max_batch_size = config.max_scheduler_batch_size.max(1);
        core_config.max_tensor_batch_size = config.max_batch_size;
        core_config.physical_execution_mode = config.physical_execution_mode;
        core_config.max_physical_in_flight = config.max_physical_in_flight;
        core_config.max_retained_sequences = config.max_retained_sequences.max(1);
        core_config.max_staged_transactions = config.max_staged_transactions.max(1);
        core_config.max_queued_requests = config.max_queued_requests.max(1);
        core_config.backend = selected_backend_kind;
        core_config.num_threads = config.num_threads.max(1);
        core_config.block_size = config.kv_page_size.max(1);
        core_config.apply_backend_context_capacity(config.portable_context_ceiling());
        core_config.kv_cache_dtype = cache_policy.effective.dtype.to_string();
        core_config.enable_prefix_caching = config.enable_prefix_caching;
        core_config.managed_prefix_cache_salt = config.managed_prefix_cache_salt.clone();
        core_config.max_prefix_cache_pages = match &cache_policy.effective.prefix {
            PrefixCachePolicy::Disabled => 0,
            PrefixCachePolicy::Namespaced { max_pages, .. } => *max_pages,
        };
        core_config.enable_chunked_prefill = config.enable_chunked_prefill;
        core_config.chunked_prefill_threshold = config.chunked_prefill_threshold.max(1);

        let mut worker_config = WorkerConfig::from(&core_config);
        worker_config.models_dir = config.models_dir.clone();
        worker_config.kv_cache_dtype = cache_policy.effective.dtype.to_string();
        worker_config.kv_page_size = config.kv_page_size.max(1);
        worker_config.model_registry = Some(model_registry.clone());
        worker_config.backend = selected_backend_kind;
        worker_config.backend_context = backend_context.clone();
        let adapter_registry = Arc::new(RuntimeAdapterRegistry::built_in_with_execution_limits(
            worker_config.max_tensor_batch_size,
            worker_config.request_parallelism,
        )?);
        worker_config.static_tensor_batch_variants =
            Arc::new(adapter_registry.static_tensor_batch_variants(selected_backend_kind));
        let execution_parallelism = effective_physical_execution_parallelism(
            core_config.physical_execution_mode,
            worker_config.request_parallelism,
            core_config
                .resolved_physical_execution_capacity()
                .physical_launch_limit
                .get(),
        );
        let coordinator = Arc::new(InferenceCoordinator::new_with_device(
            selected_backend_kind,
            device.clone(),
            execution_parallelism,
            core_config.max_queued_requests,
        )?);
        let asr_realtime_sessions = RealtimeAsrSessionPolicy::from_env()?;
        let realtime_asr_sequence_capacity = asr_realtime_sessions.retained_sequence_capacity()?;
        worker_config.resource_authority = Some(coordinator.resource_authority());
        worker_config.physical_execution_admission =
            Some(coordinator.physical_execution_admission());
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);
        let backend_router = BackendRouter::from_context(backend_context);
        let tokenizer = Arc::new(RwLock::new(None));
        let codec = Arc::new(RwLock::new(AudioCodec::new()));
        let loaded_tts_variant = Arc::new(RwLock::new(None));
        let model_lifecycle = Arc::new(ModelLifecycleController::new(
            config.clone(),
            backend_router.clone(),
            adapter_registry.clone(),
            model_manager.clone(),
            model_registry.clone(),
            core_engine.clone(),
            coordinator.clone(),
            tokenizer.clone(),
            codec.clone(),
            loaded_tts_variant.clone(),
            realtime_asr_sequence_capacity,
        ));

        Ok(Self {
            config,
            cache_policy,
            backend_router,
            inference_broker: InferenceBroker::from_env(),
            adapter_registry,
            model_manager,
            model_registry,
            tokenizer,
            codec,
            streaming_config: StreamingConfig::default(),
            core_engine,
            coordinator,
            asr_realtime_sessions,
            telemetry: Arc::new(RuntimeTelemetryCollector::new(2048)),
            completion_waiters: Arc::new(Mutex::new(HashMap::new())),
            next_completion_waiter_registration: AtomicU64::new(1),
            step_driver_task: Mutex::new(None),
            step_driver_wakeup: Arc::new(Notify::new()),
            step_driver_started: AtomicBool::new(false),
            loaded_tts_variant,
            max_loaded_models: positive_usize_env("IZWI_MAX_LOADED_MODELS"),
            model_lifecycle,
            device,
        })
    }

    /// Get reference to model manager.
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// List available models.
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models().await
    }

    /// Get explicit artifact and residency state for a specific model.
    pub async fn model_lifecycle_snapshot(
        &self,
        variant: ModelVariant,
    ) -> Option<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshot(variant).await
    }

    /// Get explicit artifact and residency states for all known models.
    pub async fn model_lifecycle_snapshots(&self) -> Vec<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshots().await
    }

    /// Snapshot of inference broker rollout state.
    pub(crate) fn inference_broker_snapshot(&self) -> InferenceBrokerSnapshot {
        self.inference_broker.snapshot()
    }

    pub fn record_stage_observation(&self, observation: RuntimeStageObservation) {
        self.telemetry.record_stage_observation(observation);
    }

    fn observe_broker_request(&self, request: &EngineCoreRequest) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, request.streaming)
    }

    fn observe_broker_request_with_streaming_required(
        &self,
        request: &EngineCoreRequest,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self
            .inference_broker
            .observe_engine_request_with_streaming_required(
                request,
                streaming_required,
                &self.adapter_registry,
                &self.backend_router,
            )
        else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    fn observe_broker_request_with_transport_streaming(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, false)
    }

    pub(crate) fn observe_broker_capability_request(
        &self,
        capability: CapabilityKind,
        model_variant: Option<ModelVariant>,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self.inference_broker.observe_capability_request(
            RouteSource::InternalRuntime,
            capability,
            model_variant,
            streaming_required,
            &self.adapter_registry,
            &self.backend_router,
        ) else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    fn record_broker_observation(&self, observation: InferenceBrokerObservation) -> Result<()> {
        if observation.shadow_enabled {
            self.telemetry.record_broker_shadow_request();
        }
        if observation.execution_enabled {
            self.telemetry.record_broker_execution_request();
        }
        if observation.routing_decision.is_some() {
            self.telemetry.record_broker_route_decision();
        }

        if let Some(message) = observation.validation_error {
            self.telemetry.record_broker_validation_failure();
            self.telemetry.record_stage_observation(
                RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: observation
                            .model_variant
                            .map(|variant| variant.dir_name().to_string()),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Failed,
                )
                .with_error_kind("routing_validation_failed"),
            );
            if observation.execution_enabled {
                return Err(Error::InvalidInput(message));
            }
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                model_variant = ?observation.model_variant,
                "Inference broker shadow validation failed: {message}"
            );
        } else if let Some(decision) = observation.routing_decision {
            self.telemetry
                .record_stage_observation(RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: Some(decision.selected_model_variant.dir_name().to_string()),
                        backend_kind: Some(decision.backend_kind.as_str().to_string()),
                        execution_target: Some(format!(
                            "{:?}",
                            decision.execution_plan.execution_target
                        )),
                        streaming_mode: Some(format!(
                            "{:?}",
                            decision.execution_plan.streaming_mode
                        )),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Observed,
                ));
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                requested_model_variant = ?observation.model_variant,
                selected_model_variant = ?decision.selected_model_variant,
                execution_target = ?decision.execution_plan.execution_target,
                backend_kind = ?decision.backend_kind,
                "Inference broker route decision recorded"
            );
        }

        Ok(())
    }

    /// Download a model.
    pub async fn download_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_manager.download_model(variant).await?;
        Ok(())
    }

    /// Spawn a non-blocking background download.
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.model_manager.spawn_download(variant).await
    }

    /// Check if a download is active.
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.model_manager.is_download_active(variant).await
    }

    /// Get runtime configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Immutable requested/effective KV cache policy selected at startup.
    pub fn resolved_kv_cache_policy(&self) -> &ResolvedKvCachePolicy {
        &self.cache_policy
    }

    /// Get codec sample rate.
    pub async fn sample_rate(&self) -> u32 {
        self.codec.read().await.sample_rate()
    }

    /// Create audio encoder.
    pub async fn audio_encoder(&self) -> AudioEncoder {
        let codec = self.codec.read().await;
        AudioEncoder::new(codec.sample_rate(), 1)
    }

    /// Get available speakers for loaded TTS model.
    pub async fn available_speakers(&self) -> Result<Vec<String>> {
        let variant = (*self.loaded_tts_variant.read().await)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
        let _lease = self
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self
                    .model_registry
                    .get_qwen_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers().into_iter().cloned().collect())
            }
            crate::catalog::ModelFamily::KokoroTts => {
                let model = self
                    .model_registry
                    .get_kokoro(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                model.available_speakers()
            }
            crate::catalog::ModelFamily::VoxtralTts => {
                let model = self
                    .model_registry
                    .get_voxtral_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self
                    .model_registry
                    .get_vibevoice_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
            }
            crate::catalog::ModelFamily::Lfm25Audio => Ok(
                crate::models::architectures::lfm25_audio::LFM25_AUDIO_BUILT_IN_SPEAKERS
                    .iter()
                    .map(|speaker| (*speaker).to_string())
                    .collect(),
            ),
            _ => Err(Error::InferenceError(format!(
                "Model {variant} does not expose TTS speakers"
            ))),
        }
    }

    /// Machine-readable diagnostics for the currently loaded direct TTS model.
    pub async fn loaded_tts_model_diagnostics(&self) -> Option<serde_json::Value> {
        let variant = (*self.loaded_tts_variant.read().await)?;
        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self.model_registry.get_qwen_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self.model_registry.get_vibevoice_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::FishS2Tts => {
                let model = self.model_registry.get_fish_s2_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            _ => None,
        }
    }

    /// Registry-backed diagnostics for native model handles loaded in memory.
    pub async fn loaded_model_diagnostics(&self) -> Vec<LoadedModelDiagnostics> {
        self.model_registry.loaded_model_diagnostics().await
    }

    async fn ensure_step_driver_started(&self) {
        let mut guard = self.step_driver_task.lock().await;
        let restart_needed = match guard.as_ref() {
            Some(handle) if !handle.is_finished() => false,
            Some(_) => true,
            None => true,
        };

        if !restart_needed {
            self.step_driver_started.store(true, Ordering::Release);
            return;
        }

        if guard.is_some() {
            self.telemetry.record_worker_restart();
        }

        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let telemetry = self.telemetry.clone();
        let wakeup = self.step_driver_wakeup.clone();
        let task = tokio::spawn(async move {
            let mut idle_backoff_ms = 1u64;
            loop {
                if !engine.has_pending_work().await {
                    let sleep_for = tokio::time::Duration::from_millis(idle_backoff_ms);
                    tokio::select! {
                        _ = tokio::time::sleep(sleep_for) => {}
                        _ = wakeup.notified() => {}
                    }
                    idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                    continue;
                }
                let step_result = std::panic::AssertUnwindSafe(engine.step_for_dispatch())
                    .catch_unwind()
                    .await;
                match step_result {
                    Ok(Ok(outputs)) => {
                        if outputs.is_empty() {
                            let sleep_for = tokio::time::Duration::from_millis(idle_backoff_ms);
                            tokio::select! {
                                _ = tokio::time::sleep(sleep_for) => {}
                                _ = wakeup.notified() => {}
                            }
                            idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                            continue;
                        }
                        idle_backoff_ms = 1;

                        for output in outputs {
                            if !output.is_finished {
                                continue;
                            }
                            route_terminal_output(
                                engine.as_ref(),
                                waiters.as_ref(),
                                telemetry.as_ref(),
                                output,
                            )
                            .await;
                        }
                    }
                    Ok(Err(err)) => {
                        error!(
                            error = %err,
                            "Engine step failed before commit; scheduled quanta were rolled back"
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                    }
                    Err(payload) => {
                        let panic_message = panic_payload_to_string(payload.as_ref());
                        telemetry.record_worker_panic();
                        let mut w = waiters.lock().await;
                        let pending: Vec<_> = w.drain().collect();
                        drop(w);
                        let request_ids: Vec<_> =
                            pending.iter().map(|(id, _)| id.as_str()).collect();
                        telemetry.record_forced_failures(request_ids).await;
                        let _ = engine.abort_all_requests().await;
                        for (_, waiter) in pending {
                            let _ = waiter.sender.send(Err(Error::InferenceError(format!(
                                "Engine worker panicked: {}",
                                panic_message
                            ))));
                        }
                        error!(
                            "Engine step worker panicked ({}); continuing with isolated loop",
                            panic_message
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                    }
                }
            }
        });

        *guard = Some(task);
        self.step_driver_started.store(true, Ordering::Release);
    }

    async fn register_waiter(
        &self,
        request_id: &str,
    ) -> Result<(u64, oneshot::Receiver<Result<EngineOutput>>)> {
        use std::collections::hash_map::Entry;

        let registration_id = self
            .next_completion_waiter_registration
            .fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        let mut waiters = self.completion_waiters.lock().await;
        match waiters.entry(request_id.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(RuntimeCompletionWaiter {
                    registration_id,
                    session_epoch: None,
                    sender: tx,
                });
                Ok((registration_id, rx))
            }
            Entry::Occupied(_) => Err(Error::InvalidInput(format!(
                "request {request_id} already has a completion waiter"
            ))),
        }
    }

    async fn remove_waiter(&self, request_id: &str, registration_id: u64) {
        remove_waiter_registration(
            self.completion_waiters.as_ref(),
            request_id,
            registration_id,
        )
        .await;
    }

    async fn bind_waiter(
        &self,
        request_id: &str,
        registration_id: u64,
        session_epoch: u64,
    ) -> Result<()> {
        if bind_waiter_registration(
            self.completion_waiters.as_ref(),
            request_id,
            registration_id,
            session_epoch,
        )
        .await
        {
            Ok(())
        } else {
            Err(Error::InferenceError(format!(
                "request {request_id} lost its completion waiter before session binding"
            )))
        }
    }

    async fn await_completion(
        &self,
        request_id: &str,
        rx: oneshot::Receiver<Result<EngineOutput>>,
        deadline: Option<std::time::Instant>,
    ) -> Result<EngineOutput> {
        let completion = async {
            rx.await.map_err(|_| {
                Error::InferenceError(format!(
                    "Request {} completion channel closed unexpectedly",
                    request_id
                ))
            })?
        };
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), completion)
                .await
                .map_err(|_| Error::Timeout(request_id.to_string()))?,
            None => completion.await,
        }
    }

    fn engine_observation_context(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
    ) -> RuntimeObservationContext {
        RuntimeObservationContext {
            route_source: Some(format!("{:?}", RouteSource::InternalEngine)),
            capability: Some(capability_name_for_task(request.task_type).to_string()),
            model_variant: request
                .model_variant
                .map(|variant| variant.dir_name().to_string()),
            backend_kind: Some(
                self.backend_router
                    .default_backend()
                    .kind()
                    .as_str()
                    .to_string(),
            ),
            pipeline_stage: Some(if streaming {
                "engine.streaming_request".to_string()
            } else {
                "engine.request".to_string()
            }),
            workload_class: Some(request.workload_class.as_str().to_string()),
            request_id: Some(request.id.clone()),
            correlation_id: request.correlation_id.clone(),
            ..RuntimeObservationContext::default()
        }
    }

    fn record_engine_output_observation(
        &self,
        request: &EngineCoreRequest,
        output: &EngineOutput,
        streaming: bool,
    ) {
        let mut timing = RuntimeStageTiming {
            admission_ms: request.admission_ms,
            total_ms: Some(output.generation_time.as_secs_f64() * 1000.0),
            ..RuntimeStageTiming::default()
        };
        if let Some(latency) = output.latency_breakdown.as_ref() {
            timing.queue_wait_ms = Some(latency.queue_wait_ms);
            timing.media_decode_ms = latency.media_decode_ms;
            timing.normalization_ms = latency.normalization_ms;
            timing.prefill_ms = Some(latency.prefill_ms);
            timing.decode_ms = Some(latency.decode_ms);
            timing.ttft_ms = latency.ttft_ms;
            timing.sampling_ms = latency.sampling_ms;
            timing.codec_ms = latency.codec_ms;
            timing.postprocess_ms = latency.postprocess_ms;
            timing.total_ms = Some(latency.total_ms);
        }

        let outcome = if output.error.is_some() {
            RuntimeStageOutcome::Failed
        } else {
            RuntimeStageOutcome::Completed
        };
        let mut observation = RuntimeStageObservation::new(
            self.engine_observation_context(request, streaming),
            outcome,
        );
        observation.timing = timing;
        observation.outputs = RuntimeStageOutputCounters {
            prompt_tokens: Some(output.token_stats.prompt_tokens as u64),
            generated_tokens: Some(output.token_stats.generated_tokens as u64),
            audio_samples: Some(output.audio.samples.len() as u64),
            transcript_chars: output.text.as_ref().map(|text| text.chars().count() as u64),
            stop_reason: output.finish_reason.map(|reason| format!("{reason:?}")),
            ..RuntimeStageOutputCounters::default()
        };
        if let Some(error) = output.error.as_ref() {
            observation.error_kind = Some(error.clone());
        }
        self.telemetry.record_stage_observation(observation);
    }

    fn record_engine_error_observation(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
        error_kind: impl Into<String>,
    ) {
        let mut observation = RuntimeStageObservation::new(
            self.engine_observation_context(request, streaming),
            RuntimeStageOutcome::Failed,
        )
        .with_error_kind(error_kind);
        observation.timing.admission_ms = request.admission_ms;
        self.telemetry.record_stage_observation(observation);
    }

    pub(crate) fn coordinator_job_for_input(
        &self,
        request_id: impl Into<String>,
        lane: CoordinatorLane,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
    ) -> JobSpec {
        let resources =
            transient_resources(self.backend_router.context().backend_kind, input_bytes);
        JobSpec {
            request_id: request_id.into(),
            lane,
            priority: runtime_context.priority,
            workload_class: runtime_context.workload_class,
            deadline: runtime_context.deadline,
            resources,
        }
    }

    pub(crate) fn coordinator_job_for_audio_input(
        &self,
        request_id: impl Into<String>,
        lane: CoordinatorLane,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
    ) -> Result<JobSpec> {
        let mut spec =
            self.coordinator_job_for_input(request_id, lane, runtime_context, input_bytes);
        spec.resources = spec.resources.checked_add(audio_decode_resources(
            self.backend_router.context().backend_kind,
        ))?;
        Ok(spec)
    }

    pub(crate) async fn prepare_engine_request_blocking<F>(
        &self,
        variant: ModelVariant,
        task_type: TaskType,
        streaming: bool,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
        additional_resources: ResourceVector,
        build: F,
    ) -> Result<AdmittedEngineRequest>
    where
        F: FnOnce(Arc<ModelRegistry>) -> Result<EngineCoreRequest> + Send + 'static,
    {
        self.prepare_engine_request_blocking_with_input(
            variant,
            task_type,
            streaming,
            runtime_context,
            input_bytes,
            additional_resources,
            |_job| async { Ok(()) },
            move |registry, ()| build(registry),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn prepare_engine_request_blocking_with_input<P, PFut, T, F>(
        &self,
        variant: ModelVariant,
        task_type: TaskType,
        streaming: bool,
        runtime_context: RuntimeRequestContext,
        input_bytes: usize,
        additional_resources: ResourceVector,
        prepare_input: P,
        build: F,
    ) -> Result<AdmittedEngineRequest>
    where
        P: FnOnce(JobLease) -> PFut,
        PFut: Future<Output = Result<T>>,
        T: Send + 'static,
        F: FnOnce(Arc<ModelRegistry>, T) -> Result<EngineCoreRequest> + Send + 'static,
    {
        let mut effective_context = runtime_context;
        if streaming && effective_context.workload_class == WorkloadClass::Online {
            effective_context.workload_class = WorkloadClass::Streaming;
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let mut spec = self.coordinator_job_for_input(
            request_id,
            coordinator_lane_for_metadata(
                task_type,
                Some(variant),
                streaming,
                effective_context.workload_class,
            ),
            effective_context,
            input_bytes,
        );
        spec.resources = spec.resources.checked_add(additional_resources)?;
        if task_decodes_audio(task_type) {
            spec.resources = spec.resources.checked_add(audio_decode_resources(
                self.backend_router.context().backend_kind,
            ))?;
        }
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;

        let prepared_input = match job.spec.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), prepare_input(job.clone()))
                .await
                .map_err(|_| Error::Timeout(job.spec.request_id.clone()))??,
            None => prepare_input(job.clone()).await?,
        };
        let residency_lease = match job.spec.deadline {
            Some(deadline) => {
                tokio::time::timeout_at(deadline.into(), self.load_model_for_inference(variant))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))??
            }
            None => self.load_model_for_inference(variant).await?,
        };
        let registry = self.model_registry.clone();
        let (residency_lease, mut request) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let request = build(registry, prepared_input)?;
                Ok((residency_lease, request))
            })
            .await?;
        if request.task_type != task_type || request.model_variant != Some(variant) {
            return Err(Error::InvalidInput(
                "prepared engine request does not match its admitted task/model".to_string(),
            ));
        }
        request.id = job.spec.request_id.clone();
        request.streaming = streaming;
        request.workload_class = effective_context.workload_class;
        request.priority = effective_context.priority;
        request.admission_ms = effective_context.admission_ms;
        request.deadline = effective_context.deadline;
        Ok(AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        })
    }

    fn coordinator_job_for_request(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<(JobSpec, JobResourceObservation)> {
        let input_bytes = retained_engine_request_input_bytes(request)?;
        let mut spec = self.coordinator_job_for_input(
            request.id.clone(),
            coordinator_lane_for_request(request),
            RuntimeRequestContext {
                workload_class: request.workload_class,
                admission_ms: request.admission_ms,
                priority: request.priority,
                deadline: request.deadline,
            },
            input_bytes,
        );
        let audio_decode_required = task_decodes_audio(request.task_type)
            && !(request.task_type == TaskType::ASR
                && request
                    .model_variant
                    .is_some_and(|variant| variant.family() == ModelFamily::Qwen3Asr)
                && request.prepared_asr_audio_for_executor()?.is_some());
        if audio_decode_required
            || (request.task_type == TaskType::TTS && request.has_tts_reference_for_execution())
        {
            spec.resources = spec.resources.checked_add(audio_decode_resources(
                self.backend_router.context().backend_kind,
            ))?;
        }
        if request.task_type == TaskType::Chat && !request.chat_config.media_inputs.is_empty() {
            if !request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::Qwen35Chat)
            {
                return Err(Error::InvalidInput(
                    "Only Qwen3.5 chat requests may include media inputs".to_string(),
                ));
            }
            let estimate = crate::models::architectures::qwen35::media_resource_estimate(
                &request.chat_config.media_inputs,
            )?;
            spec.resources = spec.resources.checked_add(media_preparation_resources(
                self.backend_router.context().backend_kind,
                estimate,
            )?)?;
        }
        Ok((spec, host_input_observation(input_bytes)?))
    }

    /// Load or pin a model under the admitted request's absolute deadline.
    /// The detached lifecycle transaction remains cancellation-safe, while the
    /// caller never waits beyond its end-to-end budget.
    pub(crate) async fn load_model_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
    ) -> Result<ModelResidencyLease> {
        match job.spec.deadline {
            Some(deadline) => {
                tokio::time::timeout_at(deadline.into(), self.load_model_for_inference(variant))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
            }
            None => self.load_model_for_inference(variant).await,
        }
    }

    /// Load and pin the exact model generation selected for a non-engine
    /// capability stage, then resolve its immutable loaded-adapter contract.
    /// Returning both values makes it impossible for direct runners to execute
    /// against catalog metadata or a different load generation.
    pub(crate) async fn load_capability_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
        capability: CapabilityKind,
        streaming_required: bool,
        expected_target: ExecutionTargetKind,
    ) -> Result<(ModelResidencyLease, LoadedExecutionContract)> {
        let lease = self.load_model_for_job(job, variant).await?;
        let bundle = self.model_lifecycle.try_get_ready_bundle(variant);
        let contract = loaded_contract_for_residency(
            &lease,
            bundle.as_deref(),
            capability,
            streaming_required,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(expected_target),
        )?;
        Ok((lease, contract))
    }

    pub(crate) async fn load_capability_with_state_for_job(
        &self,
        job: &JobLease,
        variant: ModelVariant,
        capability: CapabilityKind,
        streaming_required: bool,
        expected_target: ExecutionTargetKind,
    ) -> Result<(
        ModelResidencyLease,
        LoadedExecutionContract,
        LoadedCapabilityBinding,
    )> {
        let lease = self.load_model_for_job(job, variant).await?;
        let bundle = self.model_lifecycle.try_get_ready_bundle(variant);
        let (contract, binding) = loaded_binding_for_residency(
            &lease,
            bundle.as_deref(),
            capability,
            streaming_required,
            self.coordinator.execution_group_id(),
            self.backend_router.context().backend_kind,
            Some(expected_target),
        )?;
        Ok((lease, contract, binding))
    }

    /// Bound the pre-session Engine admission transaction by the request's
    /// absolute deadline. Cancelling this future before the core write lock is
    /// acquired cannot create a scheduler session; once that lock is acquired,
    /// Engine admission contains no further await point and returns its exact
    /// session atomically.
    async fn await_engine_admission_for_job<T, F>(&self, job: &JobLease, admission: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match job.spec.deadline {
            Some(deadline) => {
                if deadline <= Instant::now() {
                    return Err(Error::Timeout(job.spec.request_id.clone()));
                }
                tokio::time::timeout_at(deadline.into(), admission)
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
            }
            None => admission.await,
        }
    }

    pub(crate) async fn run_request(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        self.observe_broker_request(&request)?;
        let (spec, observation) = self.coordinator_job_for_request(&request)?;
        let job = self.coordinator.admit_observed(spec, observation).await?;
        let _residency_lease = match request.model_variant {
            Some(variant) => Some(self.load_model_for_job(&job, variant).await?),
            None => None,
        };
        self.run_request_after_admission(request, job, _residency_lease)
            .await
    }

    pub(crate) async fn run_admitted_request(
        &self,
        admitted: AdmittedEngineRequest,
    ) -> Result<EngineOutput> {
        let AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        } = admitted;
        self.observe_broker_request(&request)?;
        self.run_request_after_admission(request, job, Some(residency_lease))
            .await
    }

    async fn prepare_qwen3_asr_shape_for_binding(
        &self,
        request: EngineCoreRequest,
        job: &JobLease,
    ) -> Result<EngineCoreRequest> {
        if request.task_type != TaskType::ASR
            || request
                .model_variant
                .is_none_or(|variant| variant.family() != ModelFamily::Qwen3Asr)
            || request.prepared_asr_execution_shape().is_some()
        {
            return Ok(request);
        }
        let variant = request.model_variant.expect("validated Qwen3 ASR variant");
        let model =
            self.model_registry.get_asr(variant).await.ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
        let context_limit = self
            .model_registry
            .effective_context(variant)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Qwen3 ASR model {variant} has no load-sealed effective context"
                ))
            })?;
        let prepared = self
            .coordinator
            .run_host_blocking_stage(job, move || {
                let mut request = request;
                let (samples, sample_rate) =
                    crate::engine::decode_request_audio_with_rate(&request)?;
                let long_form = crate::engine::qwen3_asr_requires_long_form(
                    &samples,
                    sample_rate,
                    model.max_audio_seconds_hint(),
                );
                let input_tokens = (!long_form)
                    .then(|| {
                        model.incremental_prompt_token_count(
                            &samples,
                            sample_rate,
                            request.asr_language_for_execution(),
                            request.asr_prompt_for_execution(),
                        )
                    })
                    .transpose()?;
                request.install_prepared_asr_audio(variant, samples, sample_rate)?;
                if long_form {
                    request.install_prepared_asr_long_form_atomic()?;
                } else {
                    request.install_prepared_sequence_input_tokens(
                        input_tokens.expect("normal Qwen3 ASR shape"),
                        context_limit,
                    )?;
                }
                Ok(request)
            })
            .await?;
        job.record_materialized_usage(host_input_observation(
            retained_engine_request_input_bytes(&prepared)?,
        )?)?;
        Ok(prepared)
    }

    async fn run_request_after_admission(
        &self,
        request: EngineCoreRequest,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
    ) -> Result<EngineOutput> {
        let mut request = self
            .prepare_qwen3_asr_shape_for_binding(request, &job)
            .await?;
        let loaded_bundle = residency_lease
            .as_ref()
            .and_then(|lease| self.model_lifecycle.try_get_ready_bundle(lease.variant()));
        bind_request_to_residency(
            &mut request,
            residency_lease.as_ref(),
            loaded_bundle.as_deref(),
            false,
        )?;
        if job.spec.request_id != request.id || job.spec.deadline != request.deadline {
            return Err(Error::InvalidInput(
                "engine request does not match its coordinator admission".to_string(),
            ));
        }
        let observation_request = request.clone();
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = false
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let (waiter_registration_id, completion_rx) = self.register_waiter(&request_id).await?;
        let mut waiter_guard = WaiterRegistrationGuard::new(
            request_id.clone(),
            waiter_registration_id,
            self.completion_waiters.clone(),
        );

        let session = match self
            .await_engine_admission_for_job(
                &job,
                self.core_engine.add_request_with_session(request),
            )
            .await
        {
            Ok(session) => session,
            Err(err) => {
                self.remove_waiter(&request_id, waiter_registration_id)
                    .await;
                waiter_guard.disarm();
                self.record_engine_error_observation(&observation_request, false, err.to_string());
                return Err(err);
            }
        };
        // Establish exact-session cancellation ownership before the next await.
        // A caller may drop this future while waiter binding is contended; the
        // admitted request and its coordinator lease must still be reclaimed.
        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            waiter_registration_id,
            self.telemetry.clone(),
            job,
            residency_lease,
        );
        // If Engine admission became ready in the same poll as the deadline,
        // the exact session now exists. Establish cleanup ownership first, then
        // fail the caller without leaving that session or its waiter behind.
        if observation_request
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            let err = Error::Timeout(request_id.clone());
            guard.defer_cleanup();
            self.record_engine_error_observation(&observation_request, false, err.to_string());
            return Err(err);
        }
        self.bind_waiter(&request_id, waiter_registration_id, guard.session.epoch)
            .await?;
        waiter_guard.disarm();
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();
        let completion = self
            .await_completion(&request_id, completion_rx, observation_request.deadline)
            .await;
        match completion.as_ref() {
            Ok(output) => {
                self.record_engine_output_observation(&observation_request, output, false)
            }
            Err(err) => {
                self.record_engine_error_observation(&observation_request, false, err.to_string())
            }
        }
        let output = completion?;
        guard.disarm();
        Ok(output)
    }

    fn defer_streaming_failure(
        &self,
        guard: &mut PendingRequestGuard,
        observation_request: &EngineCoreRequest,
        err: Error,
    ) -> Error {
        // The exact abort may need the same core write lock held by a running
        // native step. Hand cleanup off before returning to the transport;
        // DeferredRequestOwnership retains admission and residency until that
        // abort proves the exact session is no longer executing.
        guard.defer_cleanup();
        self.record_engine_error_observation(observation_request, true, err.to_string());
        err
    }

    async fn deliver_streaming_chunk_before_deadline<F, Fut>(
        &self,
        on_chunk: &mut F,
        chunk: StreamingOutput,
        deadline: Option<Instant>,
        stream_request_id: &str,
        observation_request: &EngineCoreRequest,
        guard: &mut PendingRequestGuard,
    ) -> Result<()>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        // Calling `on_chunk` constructs the future and may itself perform
        // synchronous transport work. Reject an expired request before that
        // callback can observe or emit another chunk.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.to_string());
            return Err(self.defer_streaming_failure(guard, observation_request, err));
        }
        let delivery = match deadline {
            Some(deadline) => {
                match tokio::time::timeout_at(deadline.into(), on_chunk(chunk)).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout(stream_request_id.to_string())),
                }
            }
            None => on_chunk(chunk).await,
        };

        delivery.map_err(|err| self.defer_streaming_failure(guard, observation_request, err))?;
        // `timeout_at` and the callback can become ready in the same poll. Do
        // not let select/poll ordering turn an already-expired absolute
        // request deadline into a successful transport delivery.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.to_string());
            return Err(self.defer_streaming_failure(guard, observation_request, err));
        }
        Ok(())
    }

    pub(crate) async fn run_streaming_request<F, Fut>(
        &self,
        request: EngineCoreRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_streaming_request_with_broker_streaming(request, on_chunk, true)
            .await
    }

    pub(crate) async fn run_transport_streaming_request<F, Fut>(
        &self,
        request: EngineCoreRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_streaming_request_with_broker_streaming(request, on_chunk, false)
            .await
    }

    pub(crate) async fn run_admitted_streaming_request<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_admitted_streaming_request_with_broker_streaming(admitted, on_chunk, true)
            .await
    }

    pub(crate) async fn run_admitted_transport_streaming_request<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_admitted_streaming_request_with_broker_streaming(admitted, on_chunk, false)
            .await
    }

    async fn run_admitted_streaming_request_with_broker_streaming<F, Fut>(
        &self,
        admitted: AdmittedEngineRequest,
        on_chunk: F,
        broker_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let AdmittedEngineRequest {
            request,
            job,
            residency_lease,
        } = admitted;
        if broker_streaming_required {
            self.observe_broker_request(&request)?;
        } else {
            self.observe_broker_request_with_transport_streaming(&request)?;
        }
        self.run_streaming_request_after_admission(
            request,
            on_chunk,
            job,
            Some(residency_lease),
            broker_streaming_required,
        )
        .await
    }

    async fn run_streaming_request_with_broker_streaming<F, Fut>(
        &self,
        mut request: EngineCoreRequest,
        on_chunk: F,
        broker_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        request.streaming = true;
        if request.workload_class == WorkloadClass::Online {
            request.workload_class = WorkloadClass::Streaming;
        }
        if broker_streaming_required {
            self.observe_broker_request(&request)?;
        } else {
            self.observe_broker_request_with_transport_streaming(&request)?;
        }
        let (spec, observation) = self.coordinator_job_for_request(&request)?;
        let job = self.coordinator.admit_observed(spec, observation).await?;
        let _residency_lease = match request.model_variant {
            Some(variant) => Some(self.load_model_for_job(&job, variant).await?),
            None => None,
        };
        self.run_streaming_request_after_admission(
            request,
            on_chunk,
            job,
            _residency_lease,
            broker_streaming_required,
        )
        .await
    }

    async fn run_streaming_request_after_admission<F, Fut>(
        &self,
        request: EngineCoreRequest,
        mut on_chunk: F,
        job: JobLease,
        residency_lease: Option<ModelResidencyLease>,
        model_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let mut request = self
            .prepare_qwen3_asr_shape_for_binding(request, &job)
            .await?;
        let loaded_bundle = residency_lease
            .as_ref()
            .and_then(|lease| self.model_lifecycle.try_get_ready_bundle(lease.variant()));
        bind_request_to_residency(
            &mut request,
            residency_lease.as_ref(),
            loaded_bundle.as_deref(),
            model_streaming_required,
        )?;
        if job.spec.request_id != request.id || job.spec.deadline != request.deadline {
            return Err(Error::InvalidInput(
                "streaming engine request does not match its coordinator admission".to_string(),
            ));
        }
        let observation_request = request.clone();
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = true
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let (waiter_registration_id, mut completion_rx) = self.register_waiter(&request_id).await?;
        let mut waiter_guard = WaiterRegistrationGuard::new(
            request_id.clone(),
            waiter_registration_id,
            self.completion_waiters.clone(),
        );
        let (session, mut stream_rx) = match self
            .await_engine_admission_for_job(
                &job,
                self.core_engine.generate_streaming_with_session(request),
            )
            .await
        {
            Ok(v) => v,
            Err(err) => {
                self.remove_waiter(&request_id, waiter_registration_id)
                    .await;
                waiter_guard.disarm();
                self.record_engine_error_observation(&observation_request, true, err.to_string());
                return Err(err);
            }
        };
        let stream_request_id = session.request_id.clone();
        debug_assert_eq!(stream_request_id, request_id);
        // Establish exact-session cancellation ownership before the next await.
        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            waiter_registration_id,
            self.telemetry.clone(),
            job,
            residency_lease,
        );
        // As above, a session returned on the deadline boundary must be
        // cancelled through exact-session ownership rather than treated as a
        // pre-admission timeout.
        if observation_request
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            let err = Error::Timeout(stream_request_id.clone());
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        self.bind_waiter(
            &stream_request_id,
            waiter_registration_id,
            guard.session.epoch,
        )
        .await?;
        waiter_guard.disarm();
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();
        let mut completion_result: Option<EngineOutput> = None;
        let deadline = observation_request.deadline;
        let deadline_wait = async move {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline_wait);
        let mut stream_order = StreamOutputOrder::new(observation_request.stream_policy);

        loop {
            tokio::select! {
                maybe_chunk = stream_rx.recv() => {
                    let Some(chunk) = maybe_chunk else {
                        break;
                    };

                    if let Err(err) = stream_order.observe(&stream_request_id, &chunk) {
                        return Err(self.defer_streaming_failure(
                            &mut guard,
                            &observation_request,
                            err,
                        ));
                    }

                    self
                        .deliver_streaming_chunk_before_deadline(
                            &mut on_chunk,
                            chunk,
                            deadline,
                            &stream_request_id,
                            &observation_request,
                            &mut guard,
                        )
                        .await?
                }
                completion = &mut completion_rx, if completion_result.is_none() => {
                    let completion = match completion {
                        Ok(completion) => completion,
                        Err(_) => {
                            let err = Error::InferenceError(format!(
                                "Request {stream_request_id} completion channel closed unexpectedly"
                            ));
                            return Err(self.defer_streaming_failure(
                                &mut guard,
                                &observation_request,
                                err,
                            ));
                        }
                    };

                    match completion {
                        Ok(output) => {
                            completion_result = Some(output);
                        }
                        Err(err) => {
                            // If engine worker panics, fail fast so streaming callers
                            // don't hang waiting for a chunk channel that may never close.
                            return Err(self.defer_streaming_failure(
                                &mut guard,
                                &observation_request,
                                err,
                            ));
                        }
                    }
                }
                _ = &mut deadline_wait => {
                    let err = Error::Timeout(stream_request_id.clone());
                    return Err(self.defer_streaming_failure(
                        &mut guard,
                        &observation_request,
                        err,
                    ));
                }
            }
        }

        let output = if let Some(output) = completion_result {
            output
        } else {
            match self
                .await_completion(
                    &stream_request_id,
                    completion_rx,
                    observation_request.deadline,
                )
                .await
            {
                Ok(output) => output,
                Err(err) => {
                    return Err(self.defer_streaming_failure(
                        &mut guard,
                        &observation_request,
                        err,
                    ));
                }
            }
        };
        // Completion and stream closure can be ready in the same select poll
        // as the deadline. Preserve an absolute end-to-end deadline regardless
        // of which ready branch Tokio chooses first.
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let err = Error::Timeout(stream_request_id.clone());
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        if let Err(err) = stream_order.require_final(&stream_request_id) {
            return Err(self.defer_streaming_failure(&mut guard, &observation_request, err));
        }
        self.record_engine_output_observation(&observation_request, &output, true);
        guard.disarm();
        // Allow pending tasks to progress before returning to upper layers.
        yield_now().await;
        Ok(output)
    }

    /// Snapshot of runtime/engine telemetry (queue/prefill/decode/worker health).
    pub async fn telemetry_snapshot(&self) -> RuntimeTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot().await;
        snapshot.engine = self.engine_telemetry_snapshot().await;
        snapshot.coordinator = self.coordinator.snapshot();
        snapshot.models = self.loaded_model_diagnostics().await;
        snapshot
    }

    /// Prometheus exposition format telemetry payload.
    pub async fn telemetry_prometheus(&self) -> String {
        let mut payload = self.telemetry.prometheus().await;
        self.push_engine_prometheus_metrics(&mut payload).await;
        self.push_coordinator_prometheus_metrics(&mut payload);
        payload
    }

    pub fn coordinator_snapshot(&self) -> CoordinatorSnapshot {
        self.coordinator.snapshot()
    }

    pub fn is_draining(&self) -> bool {
        self.coordinator.is_draining()
    }

    pub fn begin_drain(&self) {
        self.coordinator.begin_drain();
        self.step_driver_wakeup.notify_waiters();
    }

    pub async fn wait_for_drain(&self, timeout: Duration) -> Result<()> {
        self.begin_drain();
        self.coordinator
            .wait_for_idle(Instant::now() + timeout)
            .await
    }

    async fn engine_telemetry_snapshot(&self) -> EngineRuntimeTelemetrySnapshot {
        let queue_depth = self.core_engine.pending_requests().await as u64;
        let running_requests = self.core_engine.running_requests().await as u64;
        let kv_cache = self.core_engine.kv_cache_snapshot().await;
        let stream = engine_stream_metrics_snapshot();

        let batch = engine_batch_metrics_snapshot();
        EngineRuntimeTelemetrySnapshot {
            scheduler_queue_depth: queue_depth,
            scheduler_running_requests: running_requests,
            incremental_prefill_quanta_committed_total: batch
                .incremental_prefill_quanta_committed_total,
            incremental_prefill_tokens_committed_total: batch
                .incremental_prefill_tokens_committed_total,
            multispan_prefill_requests_total: batch.multispan_prefill_requests_total,
            stream_backpressure_total: stream.backpressure_total,
            stream_checkpoints_committed_total: stream.checkpoints_committed_total,
            stream_checkpoint_rejections_total: stream.checkpoint_rejections_total,
            stream_delivery_failures_total: stream.delivery_failures_total,
            tensor_batches_total: batch.tensor_batches_total,
            tensor_static_batches_total: batch.tensor_static_batches_total,
            tensor_continuous_batches_total: batch.tensor_continuous_batches_total,
            tensor_continuous_multirow_batches_total: batch
                .tensor_continuous_multirow_batches_total,
            request_parallel_batches_total: batch.request_parallel_batches_total,
            physical_batch_rejections_total: batch.physical_batch_rejections_total,
            tensor_batch_max_width: batch.tensor_batch_max_width,
            tensor_batch_rows_total: batch.tensor_batch_rows_total,
            tensor_batch_capacity_rows_total: batch.tensor_batch_capacity_rows_total,
            tensor_batch_useful_elements_total: batch.tensor_batch_useful_elements_total,
            tensor_batch_materialized_elements_total: batch
                .tensor_batch_materialized_elements_total,
            batch_workspace_bytes_total: batch.batch_workspace_bytes_total,
            dispatch_states: batch.dispatch_states,
            failure_origins: batch.failure_origins,
            deadline_phases: batch.deadline_phases,
            workspace_domains: batch.workspace_domains,
            tensor_batch_fill_ratio: batch.tensor_batch_fill_ratio,
            tensor_batch_padding_ratio: batch.tensor_batch_padding_ratio,
            model_tensor_batches_total: batch.model_tensor_batches_total,
            model_tensor_batch_rows_total: batch.model_tensor_batch_rows_total,
            model_tensor_batch_max_width: batch.model_tensor_batch_max_width,
            model_scalar_row_dispatches_total: batch.model_scalar_row_dispatches_total,
            model_decode_calls_total: batch.model_decode_calls_total,
            model_tensor_multirow_calls_total: batch.model_tensor_multirow_calls_total,
            continuous_envelope_scalar_fallbacks_total: batch
                .continuous_envelope_scalar_fallbacks_total,
            physical_execution: batch.physical_execution,
            kv_cache,
        }
    }

    async fn push_engine_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.engine_telemetry_snapshot().await;
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_QUEUE_DEPTH,
            snapshot.scheduler_queue_depth,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_RUNNING_REQUESTS,
            snapshot.scheduler_running_requests,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_INCREMENTAL_PREFILL_QUANTA_COMMITTED_TOTAL,
            snapshot.incremental_prefill_quanta_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_INCREMENTAL_PREFILL_TOKENS_COMMITTED_TOTAL,
            snapshot.incremental_prefill_tokens_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_MULTISPAN_PREFILL_REQUESTS_TOTAL,
            snapshot.multispan_prefill_requests_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_HITS_TOTAL,
            snapshot.kv_cache.counters.prefix_hits,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_MISSES_TOTAL,
            snapshot.kv_cache.counters.prefix_misses,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_EVICTIONS_TOTAL,
            snapshot.kv_cache.counters.prefix_evictions,
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
            "accounting",
            &[(
                "physical_pages",
                snapshot.kv_cache.totals.coordinator.allocated_pages,
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_FREE_BLOCKS,
            "accounting",
            &[(
                "physical_pages",
                snapshot.kv_cache.totals.coordinator.free_pages,
            )],
        );
        push_engine_labeled_metric_f64(
            payload,
            ENGINE_KV_CACHE_UTILIZATION_RATIO,
            "accounting",
            &[(
                "physical_pages",
                if snapshot.kv_cache.totals.coordinator.capacity_pages == 0 {
                    0.0
                } else {
                    snapshot.kv_cache.totals.coordinator.allocated_pages as f64
                        / snapshot.kv_cache.totals.coordinator.capacity_pages as f64
                },
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_USED_BYTES,
            "accounting",
            &[(
                "physical_managed_pages",
                managed_kv_used_bytes(&snapshot.kv_cache),
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES,
            "accounting",
            &[(
                snapshot.kv_cache.memory_accounting,
                snapshot.kv_cache.totals.physical_bytes,
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS,
            "accounting",
            &[(
                "physical_device_pages",
                managed_kv_device_pages(&snapshot.kv_cache),
            )],
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_BACKPRESSURE_TOTAL,
            snapshot.stream_backpressure_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL,
            snapshot.stream_checkpoints_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL,
            snapshot.stream_checkpoint_rejections_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_DELIVERY_FAILURES_TOTAL,
            snapshot.stream_delivery_failures_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL,
            snapshot.tensor_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL,
            snapshot.request_parallel_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH,
            snapshot.tensor_batch_max_width,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL,
            snapshot.tensor_static_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL,
            snapshot.tensor_continuous_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL,
            snapshot.tensor_continuous_multirow_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL,
            snapshot.model_tensor_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL,
            snapshot.model_tensor_batch_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH,
            snapshot.model_tensor_batch_max_width,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL,
            snapshot.model_scalar_row_dispatches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL,
            snapshot.model_decode_calls_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL,
            snapshot.model_tensor_multirow_calls_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL,
            snapshot.continuous_envelope_scalar_fallbacks_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL,
            snapshot.physical_batch_rejections_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL,
            snapshot.tensor_batch_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL,
            snapshot.tensor_batch_capacity_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL,
            snapshot.tensor_batch_useful_elements_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
            snapshot.tensor_batch_materialized_elements_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL,
            snapshot.batch_workspace_bytes_total,
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL,
            "state",
            &snapshot.dispatch_states.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL,
            "origin",
            &snapshot.failure_origins.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL,
            "phase",
            &snapshot.deadline_phases.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL,
            "domain",
            &snapshot.workspace_domains.labeled_values(),
        );
        push_engine_metric_f64(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO,
            snapshot.tensor_batch_fill_ratio,
        );
        push_engine_metric_f64(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO,
            snapshot.tensor_batch_padding_ratio,
        );
        push_engine_physical_execution_metrics(payload, &snapshot.physical_execution);
    }

    fn push_coordinator_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.coordinator.snapshot();
        payload.push_str(&format!(
            "# TYPE izwi_inference_coordinator_capacity gauge\n\
izwi_inference_coordinator_capacity {}\n\
# TYPE izwi_inference_coordinator_active_jobs gauge\n\
izwi_inference_coordinator_active_jobs {}\n\
# TYPE izwi_inference_coordinator_active_executions gauge\n\
izwi_inference_coordinator_active_executions {}\n\
# TYPE izwi_inference_coordinator_reserved_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_host_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_host_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_device_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_device_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_unified_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_unified_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_admitted_total counter\n\
izwi_inference_coordinator_admitted_total {}\n\
# TYPE izwi_inference_coordinator_rejected_total counter\n\
izwi_inference_coordinator_rejected_total {}\n\
# TYPE izwi_inference_coordinator_expired_total counter\n\
izwi_inference_coordinator_expired_total {}\n\
# TYPE izwi_inference_coordinator_draining gauge\n\
izwi_inference_coordinator_draining {}\n\
# TYPE izwi_inference_coordinator_poisoned gauge\n\
izwi_inference_coordinator_poisoned {}\n",
            snapshot.capacity,
            snapshot.active_jobs,
            snapshot.active_executions,
            snapshot.reserved_memory_bytes,
            snapshot.reserved_host_memory_bytes,
            snapshot.reserved_device_memory_bytes,
            snapshot.reserved_unified_memory_bytes,
            snapshot.admitted_total,
            snapshot.rejected_total,
            snapshot.expired_total,
            u8::from(snapshot.draining),
            u8::from(snapshot.poisoned),
        ));
    }

    pub fn record_voice_session_started(&self) {
        self.telemetry.record_voice_session_started();
        self.record_voice_stage_observation("voice.session_started");
    }

    pub fn record_voice_session_closed(&self) {
        self.telemetry.record_voice_session_closed();
        self.record_voice_stage_observation("voice.session_closed");
    }

    pub fn record_voice_interruption(&self) {
        self.telemetry.record_voice_interruption();
        self.record_voice_stage_observation("voice.interruption");
    }

    pub fn record_voice_barge_in(&self) {
        self.telemetry.record_voice_barge_in();
        self.record_voice_stage_observation("voice.barge_in");
    }

    pub fn record_voice_stream_backpressure(&self) {
        self.telemetry.record_voice_stream_backpressure();
        self.record_voice_stage_observation("voice.stream_backpressure");
    }

    pub fn record_transcription_stream_backpressure(&self) {
        self.telemetry.record_transcription_stream_backpressure();
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some("realtime_transcription".to_string()),
                    capability: Some("asr".to_string()),
                    pipeline_kind: Some("realtime_transcription".to_string()),
                    pipeline_stage: Some("transcription.stream_backpressure".to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
    }

    pub fn record_modular_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::modular_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_unified_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::unified_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub(crate) fn record_diarization_transcript_pipeline(&self, enable_llm_refinement: bool) {
        let graph = PipelineGraph::diarization_transcript(enable_llm_refinement);
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_batch_asr_pipeline_job(&self) {
        let graph = PipelineGraph::batch_asr_transcription();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_batch_tts_pipeline_job(&self) {
        let graph = PipelineGraph::batch_tts_speech();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    fn record_voice_stage_observation(&self, pipeline_stage: &'static str) {
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some(format!("{:?}", RouteSource::RealtimeVoice)),
                    capability: Some(format!("{:?}", CapabilityKind::SpeechToSpeech)),
                    pipeline_kind: Some("realtime_voice".to_string()),
                    pipeline_stage: Some(pipeline_stage.to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
    }
}

fn capability_name_for_task(task_type: TaskType) -> &'static str {
    match task_type {
        TaskType::TTS => "tts",
        TaskType::ASR => "asr",
        TaskType::Chat => "chat",
        TaskType::SpeechToSpeech => "speech_to_speech",
    }
}

fn configure_runtime_threading(num_threads: usize) {
    let value = num_threads.max(1).to_string();
    for key in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        if std::env::var(key).is_err() {
            std::env::set_var(key, &value);
        }
    }
    debug!("Configured runtime threading hints to {} threads", value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendCapabilities, BackendContext, BackendSelectionSource};
    use crate::engine::{
        ExecutionDisposition, ExecutorOutput, FinishReason as ExecutionFinishReason,
        OutputProcessor,
    };
    use crate::runtime::broker::{InferenceBroker, InferenceBrokerMode};

    #[test]
    fn direct_execution_capacity_obeys_rollout_mode() {
        assert_eq!(
            effective_physical_execution_parallelism(
                crate::config::PhysicalExecutionMode::Serial,
                4,
                8,
            ),
            1
        );
        assert_eq!(
            effective_physical_execution_parallelism(
                crate::config::PhysicalExecutionMode::Shadow,
                4,
                8,
            ),
            1
        );
        assert_eq!(
            effective_physical_execution_parallelism(
                crate::config::PhysicalExecutionMode::Concurrent,
                4,
                8,
            ),
            8
        );
    }

    fn terminal_output(reason: ExecutionFinishReason) -> EngineOutput {
        OutputProcessor::new(24_000).process_execution(
            ExecutorOutput::terminal("terminal-request".to_string()),
            &ExecutionDisposition::Finished(reason),
            7,
            std::time::Duration::ZERO,
        )
    }

    #[test]
    fn stream_output_order_rejects_wrong_identity_reordering_and_truncation() {
        let request_id = "ordered-stream";
        let mut order = StreamOutputOrder::new(EngineStreamPolicy::FailOnFull);
        let first = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
        order.observe(request_id, &first).unwrap();

        let duplicate = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
        assert!(order
            .observe(request_id, &duplicate)
            .unwrap_err()
            .to_string()
            .contains("not greater"));
        let wrong_request = StreamingOutput::new("stale".to_string(), 1, vec![0.0], 24_000);
        assert!(order
            .observe(request_id, &wrong_request)
            .unwrap_err()
            .to_string()
            .contains("carried request ID"));
        assert!(order
            .require_final(request_id)
            .unwrap_err()
            .to_string()
            .contains("without a final marker"));

        let gap = StreamingOutput::new(request_id.to_string(), 4, Vec::new(), 0);
        assert!(order
            .observe(request_id, &gap)
            .unwrap_err()
            .to_string()
            .contains("did not match expected 1"));

        // Gaps remain valid only for an explicitly lossy DropNewest transport,
        // while every observed sequence must still advance monotonically.
        let mut order = StreamOutputOrder::new(EngineStreamPolicy::DropNewest);
        order.observe(request_id, &first).unwrap();
        let mut final_output = StreamingOutput::new(request_id.to_string(), 4, Vec::new(), 0);
        final_output.is_final = true;
        order.observe(request_id, &final_output).unwrap();
        order.require_final(request_id).unwrap();

        let after_final = StreamingOutput::new(request_id.to_string(), 5, vec![0.0], 24_000);
        assert!(order
            .observe(request_id, &after_final)
            .unwrap_err()
            .to_string()
            .contains("after its final marker"));
    }

    #[test]
    fn residency_binding_carries_exact_instance_and_rejects_wrong_variant() {
        let residency = crate::model::ModelResidency::default();
        let instance = crate::engine::ModelInstanceId::new(17);
        let lease = residency.acquire_instance_lease(ModelVariant::Kokoro82M, instance);
        let adapters = RuntimeAdapterRegistry::built_in();
        let bundle = LoadedModelBundle::bind(
            &adapters,
            crate::engine::ExecutionGroupId::new(3),
            instance,
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .expect("loaded bundle");
        let mut request =
            EngineCoreRequest::tts("bind me").with_model_variant(ModelVariant::Kokoro82M);

        bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
            .expect("matching residency");
        assert_eq!(request.model_instance_id(), Some(instance));
        assert!(request.execution_adapter_binding().is_some());
        assert!(request.v2_state_runtime().is_some());

        let mut wrong = EngineCoreRequest::tts("wrong").with_model_variant(ModelVariant::Qwen306B);
        assert!(bind_request_to_residency(&mut wrong, Some(&lease), Some(&bundle), false).is_err());
        assert_eq!(wrong.model_instance_id(), None);

        let mut missing_bundle =
            EngineCoreRequest::tts("missing").with_model_variant(ModelVariant::Kokoro82M);
        assert!(bind_request_to_residency(&mut missing_bundle, Some(&lease), None, false).is_err());
        assert_eq!(missing_bundle.model_instance_id(), None);
    }

    #[test]
    fn residency_binding_uses_one_descriptor_for_execution_and_state_truth() {
        let residency = crate::model::ModelResidency::default();
        let instance = crate::engine::ModelInstanceId::new(19);
        let variant = ModelVariant::Kokoro82M;
        let lease = residency.acquire_instance_lease(variant, instance);
        let bundle = LoadedModelBundle::bind(
            &RuntimeAdapterRegistry::built_in(),
            crate::engine::ExecutionGroupId::new(3),
            instance,
            variant,
            BackendKind::Cpu,
        )
        .expect("loaded bundle");
        let mut request = EngineCoreRequest::tts("state").with_model_variant(variant);

        bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
            .expect("matching loaded capability descriptor");

        let execution = request
            .execution_adapter_binding()
            .expect("execution binding");
        let state = request.v2_state_runtime().expect("state binding");
        assert_eq!(execution.model_instance_id, instance);
        state.validate_against(BackendKind::Cpu, execution).unwrap();
    }

    #[test]
    fn residency_binding_preserves_v2_state_without_opaque_fallback() {
        let residency = crate::model::ModelResidency::default();
        let instance = crate::engine::ModelInstanceId::new(20);
        let variant = ModelVariant::Kokoro82M;
        let lease = residency.acquire_instance_lease(variant, instance);
        let registry = RuntimeAdapterRegistry::built_in();
        let compatibility = LoadedModelBundle::bind(
            &registry,
            crate::engine::ExecutionGroupId::new(3),
            instance,
            variant,
            BackendKind::Cpu,
        )
        .expect("compatibility bundle");
        let offline = compatibility
            .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
            .expect("offline TTS contract")
            .stages;
        let transport = compatibility
            .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::transport_only())
            .expect("transport TTS contract")
            .stages;
        let native = compatibility
            .contract_for_streaming(
                CapabilityKind::Tts,
                StreamingRequirements {
                    transport_output: false,
                    model_native: true,
                    asr_long_form: false,
                },
            )
            .expect("native TTS contract")
            .stages;
        let native_transport = compatibility
            .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::native(true))
            .expect("native transport TTS contract")
            .stages;
        let descriptor =
            crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stage_graphs_test(&[
                &offline,
                &transport,
                &native,
                &native_transport,
            ]);
        let bundle = LoadedModelBundle::bind_with_state_publications(
            &registry,
            crate::engine::ExecutionGroupId::new(3),
            instance,
            variant,
            BackendKind::Cpu,
            HashMap::from([(
                CapabilityKind::Tts,
                crate::runtime::adapters::LoadedStatePublication::V2(descriptor.clone()),
            )]),
        )
        .expect("v2 loaded bundle");
        let mut request = EngineCoreRequest::tts("stateless").with_model_variant(variant);
        bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
            .expect("matching v2 loaded capability descriptor");

        assert_eq!(request.v2_state_descriptor(), Some(&descriptor));
        assert_eq!(
            request.v2_state_fingerprint(),
            Some(descriptor.fingerprint(&offline).unwrap())
        );
        assert!(request.v2_state_runtime().is_some());
        let direct = loaded_contract_for_residency(
            &lease,
            Some(&bundle),
            CapabilityKind::Tts,
            false,
            crate::engine::ExecutionGroupId::new(3),
            BackendKind::Cpu,
            None,
        )
        .expect("direct runners may use a load-sealed stateless v2 runtime");
        assert_eq!(direct.model_instance_id, instance);
    }

    #[test]
    fn direct_loaded_contract_requires_exact_generation_group_backend_and_target() {
        let residency = crate::model::ModelResidency::default();
        let instance = crate::engine::ModelInstanceId::new(21);
        let group = crate::engine::ExecutionGroupId::new(8);
        let lease = residency.acquire_instance_lease(ModelVariant::Kokoro82M, instance);
        let adapters = RuntimeAdapterRegistry::built_in();
        let bundle = LoadedModelBundle::bind(
            &adapters,
            group,
            instance,
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .unwrap();

        let contract = loaded_contract_for_residency(
            &lease,
            Some(&bundle),
            CapabilityKind::Tts,
            false,
            group,
            BackendKind::Cpu,
            Some(ExecutionTargetKind::DirectModel),
        )
        .unwrap();
        assert_eq!(contract.model_instance_id, instance);
        assert_eq!(contract.execution_group_id, group);

        assert!(loaded_contract_for_residency(
            &lease,
            Some(&bundle),
            CapabilityKind::Tts,
            false,
            group,
            BackendKind::Cpu,
            Some(ExecutionTargetKind::TokenEngine),
        )
        .is_err());
        assert!(loaded_contract_for_residency(
            &lease,
            Some(&bundle),
            CapabilityKind::Tts,
            false,
            crate::engine::ExecutionGroupId::new(group.get() + 1),
            BackendKind::Cpu,
            Some(ExecutionTargetKind::DirectModel),
        )
        .is_err());
    }

    async fn pending_streaming_guard_fixture(
        runtime: &RuntimeService,
        request_id: &str,
        deadline: Option<Instant>,
    ) -> (
        EngineCoreRequest,
        PendingRequestGuard,
        oneshot::Receiver<Result<EngineOutput>>,
        ModelVariant,
    ) {
        let residency_variant = ModelVariant::Kokoro82M;
        let mut request = EngineCoreRequest::tts("streaming callback fixture")
            .with_model_variant(residency_variant)
            .with_deadline(deadline);
        request.id = request_id.to_string();
        request.prompt_tokens = vec![1];
        request.streaming = true;
        let observation_request = request.clone();
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let (registration_id, receiver) = runtime
            .register_waiter(request_id)
            .await
            .expect("waiter registration");
        let session = runtime
            .core_engine
            .add_request_with_session(request)
            .await
            .expect("engine admission");
        runtime
            .bind_waiter(request_id, registration_id, session.epoch)
            .await
            .expect("waiter binding");
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);
        let guard = PendingRequestGuard::new(
            session,
            runtime.core_engine.clone(),
            runtime.completion_waiters.clone(),
            registration_id,
            runtime.telemetry.clone(),
            job,
            Some(residency_lease),
        );

        (observation_request, guard, receiver, residency_variant)
    }

    #[test]
    fn runtime_completion_preserves_typed_terminal_failures() {
        assert!(matches!(
            runtime_completion(terminal_output(ExecutionFinishReason::Cancelled)),
            Err(Error::Cancelled(request_id)) if request_id == "terminal-request"
        ));
        assert!(matches!(
            runtime_completion(terminal_output(ExecutionFinishReason::TimedOut)),
            Err(Error::Timeout(request_id)) if request_id == "terminal-request"
        ));
    }

    #[tokio::test]
    async fn duplicate_waiter_registration_preserves_original_owner() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let (original_registration, original) = runtime
            .register_waiter("same-request")
            .await
            .expect("first waiter");
        let duplicate = runtime.register_waiter("same-request").await;

        assert!(matches!(duplicate, Err(Error::InvalidInput(_))));
        assert_eq!(runtime.completion_waiters.lock().await.len(), 1);
        assert!(runtime
            .completion_waiters
            .lock()
            .await
            .contains_key("same-request"));
        drop(original);
        runtime
            .remove_waiter("same-request", original_registration)
            .await;
    }

    #[tokio::test]
    async fn delayed_waiter_cleanup_cannot_remove_a_reused_request_registration() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let (old_registration, old_receiver) = runtime
            .register_waiter("reused-request")
            .await
            .expect("old waiter");
        runtime
            .remove_waiter("reused-request", old_registration)
            .await;
        drop(old_receiver);

        let (new_registration, _new_receiver) = runtime
            .register_waiter("reused-request")
            .await
            .expect("new waiter");
        assert_ne!(old_registration, new_registration);

        assert!(
            !remove_waiter_registration(
                runtime.completion_waiters.as_ref(),
                "reused-request",
                old_registration,
            )
            .await
        );
        assert_eq!(
            runtime
                .completion_waiters
                .lock()
                .await
                .get("reused-request")
                .map(|waiter| waiter.registration_id),
            Some(new_registration)
        );
    }

    #[tokio::test]
    async fn dropped_waiter_guard_cleans_an_unbound_registration() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let (registration_id, receiver) = runtime
            .register_waiter("cancelled-before-admission")
            .await
            .expect("waiter registration");
        let guard = WaiterRegistrationGuard::new(
            "cancelled-before-admission".to_string(),
            registration_id,
            runtime.completion_waiters.clone(),
        );

        drop(guard);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if !runtime
                    .completion_waiters
                    .lock()
                    .await
                    .contains_key("cancelled-before-admission")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped guard did not remove its exact registration");
        assert!(receiver.await.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_pending_guard_reclaims_admitted_session_and_job() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        runtime.ensure_step_driver_started().await;
        let mut request = EngineCoreRequest::tts("cancel between admission and waiter binding")
            .with_model_variant(ModelVariant::Kokoro82M);
        request.id = "cancel-during-waiter-binding".to_string();
        request.prompt_tokens = vec![1];
        let request_id = request.id.clone();
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let (registration_id, receiver) = runtime
            .register_waiter(&request_id)
            .await
            .expect("waiter registration");
        let abandoned_session = runtime
            .core_engine
            .add_request_with_session(request)
            .await
            .expect("engine admission");
        let residency_variant = ModelVariant::Kokoro82M;
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1
        );
        let waiter_lock = runtime.completion_waiters.lock().await;
        let guard = PendingRequestGuard::new(
            abandoned_session.clone(),
            runtime.core_engine.clone(),
            runtime.completion_waiters.clone(),
            registration_id,
            runtime.telemetry.clone(),
            job,
            Some(residency_lease),
        );

        drop(guard);
        tokio::task::yield_now().await;
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1,
            "cancellation cleanup must retain the model residency lease"
        );
        drop(waiter_lock);
        assert!(receiver.await.is_err());

        let mut replacement = EngineCoreRequest::tts("reuse cancelled binding request id")
            .with_model_variant(ModelVariant::Kokoro82M);
        replacement.id = request_id.clone();
        replacement.prompt_tokens = vec![2];
        let replacement_session = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Ok(session) = runtime
                    .core_engine
                    .add_request_with_session(replacement.clone())
                    .await
                {
                    break session;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped pending guard kept the exact session fenced");
        assert_ne!(replacement_session.epoch, abandoned_session.epoch);
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            0
        );
        runtime.core_engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_pending_guard_retains_ownership_while_exact_abort_waits_for_core_step_lock() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("cancel during an in-flight engine step");
        request.id = "cancel-during-engine-step".to_string();
        request.prompt_tokens = vec![1];
        request.model_variant = Some(ModelVariant::Kokoro82M);
        let request_id = request.id.clone();
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let (registration_id, receiver) = runtime
            .register_waiter(&request_id)
            .await
            .expect("waiter registration");
        let session = runtime
            .core_engine
            .add_request_with_session(request)
            .await
            .expect("engine admission");
        let residency_variant = ModelVariant::Kokoro82M;
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);
        let guard = PendingRequestGuard::new(
            session,
            runtime.core_engine.clone(),
            runtime.completion_waiters.clone(),
            registration_id,
            runtime.telemetry.clone(),
            job,
            Some(residency_lease),
        );

        let (step_entered_tx, step_entered_rx) = oneshot::channel();
        let (release_step_tx, release_step_rx) = oneshot::channel();
        let engine = runtime.core_engine.clone();
        let step_lock = tokio::spawn(async move {
            engine
                .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
                .await;
        });
        step_entered_rx.await.expect("step lock was not acquired");

        drop(guard);
        assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
            .await
            .expect("cleanup did not remove its waiter before exact abort")
            .is_err());
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 1);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1,
            "exact abort must retain residency while an engine step owns the core lock"
        );

        release_step_tx.send(()).expect("release step lock");
        step_lock.await.expect("step-lock task");
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if runtime.coordinator_snapshot().active_jobs == 0
                    && runtime
                        .model_manager
                        .active_residency_leases(residency_variant)
                        == 0
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("exact abort did not release request ownership after the core lock became safe");
        runtime.core_engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nonstreaming_engine_admission_returns_at_deadline_while_core_lock_is_held() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let (step_entered_tx, step_entered_rx) = oneshot::channel();
        let (release_step_tx, release_step_rx) = oneshot::channel();
        let engine = runtime.core_engine.clone();
        let step_lock = tokio::spawn(async move {
            engine
                .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
                .await;
        });
        step_entered_rx.await.expect("step lock was not acquired");

        let request_id = "nonstreaming-admission-deadline".to_string();
        let deadline = Instant::now() + Duration::from_millis(25);
        let residency_variant = ModelVariant::Kokoro82M;
        let mut request = EngineCoreRequest::tts("bounded Engine admission")
            .with_model_variant(residency_variant)
            .with_deadline(Some(deadline));
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);

        let err = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.run_request_after_admission(request, job, Some(residency_lease)),
        )
        .await
        .expect("Engine admission waited for the core lock past its deadline")
        .expect_err("expired Engine admission unexpectedly succeeded");
        assert!(matches!(err, Error::Timeout(id) if id == request_id));
        assert!(
            !step_lock.is_finished(),
            "the core lock was released too early"
        );
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            0
        );
        assert!(!runtime
            .completion_waiters
            .lock()
            .await
            .contains_key(&request_id));

        release_step_tx.send(()).expect("release step lock");
        step_lock.await.expect("step-lock task");
        assert_eq!(
            runtime.core_engine.request_session_key(&request_id).await,
            None
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_engine_admission_returns_at_deadline_while_core_lock_is_held() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let (step_entered_tx, step_entered_rx) = oneshot::channel();
        let (release_step_tx, release_step_rx) = oneshot::channel();
        let engine = runtime.core_engine.clone();
        let step_lock = tokio::spawn(async move {
            engine
                .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
                .await;
        });
        step_entered_rx.await.expect("step lock was not acquired");

        let request_id = "streaming-admission-deadline".to_string();
        let deadline = Instant::now() + Duration::from_millis(25);
        let residency_variant = ModelVariant::Kokoro82M;
        let mut request = EngineCoreRequest::tts("bounded streaming Engine admission")
            .with_model_variant(residency_variant)
            .with_deadline(Some(deadline));
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.streaming = true;
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);

        let err = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.run_streaming_request_after_admission(
                request,
                |_| std::future::ready(Ok(())),
                job,
                Some(residency_lease),
                true,
            ),
        )
        .await
        .expect("streaming Engine admission waited for the core lock past its deadline")
        .expect_err("expired streaming Engine admission unexpectedly succeeded");
        assert!(matches!(err, Error::Timeout(id) if id == request_id));
        assert!(
            !step_lock.is_finished(),
            "the core lock was released too early"
        );
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            0
        );
        assert!(!runtime
            .completion_waiters
            .lock()
            .await
            .contains_key(&request_id));

        release_step_tx.send(()).expect("release step lock");
        step_lock.await.expect("step-lock task");
        assert_eq!(
            runtime.core_engine.request_session_key(&request_id).await,
            None
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn expired_streaming_deadline_does_not_invoke_callback() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let request_id = "expired-streaming-callback";
        let deadline = Instant::now() + Duration::from_millis(50);
        let (observation_request, mut guard, receiver, residency_variant) =
            pending_streaming_guard_fixture(&runtime, request_id, Some(deadline)).await;
        tokio::time::sleep_until(deadline.into()).await;

        let callback_invoked = Arc::new(AtomicBool::new(false));
        let callback_observer = callback_invoked.clone();
        let mut callback = move |_| {
            callback_observer.store(true, Ordering::Release);
            std::future::ready(Ok(()))
        };
        let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);

        let err = runtime
            .deliver_streaming_chunk_before_deadline(
                &mut callback,
                chunk,
                Some(deadline),
                request_id,
                &observation_request,
                &mut guard,
            )
            .await
            .expect_err("expired callback unexpectedly ran");
        assert!(matches!(err, Error::Timeout(id) if id == request_id));
        assert!(
            !callback_invoked.load(Ordering::Acquire),
            "an expired request invoked synchronous callback code"
        );
        assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
            .await
            .expect("deadline cleanup did not remove its exact waiter")
            .is_err());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if runtime.coordinator_snapshot().active_jobs == 0
                    && runtime
                        .model_manager
                        .active_residency_leases(residency_variant)
                        == 0
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("deadline cleanup did not release request ownership");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hung_streaming_callback_is_bounded_by_absolute_request_deadline() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let request_id = "hung-streaming-callback-deadline";
        let deadline = Instant::now() + Duration::from_millis(25);
        let (observation_request, mut guard, receiver, residency_variant) =
            pending_streaming_guard_fixture(&runtime, request_id, Some(deadline)).await;
        let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
        let mut callback = |_| std::future::pending::<Result<()>>();

        let err = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.deliver_streaming_chunk_before_deadline(
                &mut callback,
                chunk,
                Some(deadline),
                request_id,
                &observation_request,
                &mut guard,
            ),
        )
        .await
        .expect("hung callback outlived the absolute request deadline")
        .expect_err("hung callback unexpectedly succeeded");
        assert!(matches!(err, Error::Timeout(id) if id == request_id));
        assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
            .await
            .expect("deadline cleanup did not remove its exact waiter")
            .is_err());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if runtime.coordinator_snapshot().active_jobs == 0
                    && runtime
                        .model_manager
                        .active_residency_leases(residency_variant)
                        == 0
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("deadline cleanup did not release request ownership");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_callback_failure_returns_while_exact_abort_waits_for_core_step_lock() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let request_id = "streaming-callback-failure-during-step";
        let (observation_request, mut guard, receiver, residency_variant) =
            pending_streaming_guard_fixture(&runtime, request_id, None).await;
        let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
        let (step_entered_tx, step_entered_rx) = oneshot::channel();
        let (release_step_tx, release_step_rx) = oneshot::channel();
        let engine = runtime.core_engine.clone();
        let step_lock = tokio::spawn(async move {
            engine
                .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
                .await;
        });
        step_entered_rx.await.expect("step lock was not acquired");
        let mut callback = |_| {
            std::future::ready(Err(Error::InferenceError(
                "streaming callback failed".to_string(),
            )))
        };

        let err = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.deliver_streaming_chunk_before_deadline(
                &mut callback,
                chunk,
                None,
                request_id,
                &observation_request,
                &mut guard,
            ),
        )
        .await
        .expect("callback failure waited for the in-flight core step")
        .expect_err("failing callback unexpectedly succeeded");
        assert!(err.to_string().contains("streaming callback failed"));
        assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
            .await
            .expect("detached cleanup did not remove its exact waiter")
            .is_err());
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 1);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1,
            "detached exact abort released residency while the core lock was held"
        );

        release_step_tx.send(()).expect("release step lock");
        step_lock.await.expect("step-lock task");
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if runtime.coordinator_snapshot().active_jobs == 0
                    && runtime
                        .model_manager
                        .active_residency_leases(residency_variant)
                        == 0
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("exact abort did not release ownership after the core lock became safe");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aborted_detached_cleanup_retains_request_ownership_fail_closed() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("abort detached cancellation cleanup");
        request.id = "abort-detached-cleanup".to_string();
        request.prompt_tokens = vec![1];
        request.model_variant = Some(ModelVariant::Kokoro82M);
        let request_id = request.id.clone();
        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("job shape");
        let isolated_coordinator = Arc::new(InferenceCoordinator::new(
            runtime.backend_context().backend_kind,
            1,
            1,
        ));
        let job = isolated_coordinator
            .admit_observed(spec, observation)
            .await
            .expect("isolated job admission");
        let (registration_id, receiver) = runtime
            .register_waiter(&request_id)
            .await
            .expect("waiter registration");
        let session = runtime
            .core_engine
            .add_request_with_session(request)
            .await
            .expect("engine admission");
        let residency_variant = ModelVariant::Kokoro82M;
        let residency_lease = runtime
            .model_manager
            .acquire_residency_lease(residency_variant);
        let ownership = DeferredRequestOwnership::new(Some(job), Some(residency_lease));
        let waiter_lock = runtime.completion_waiters.lock().await;
        let cleanup = tokio::spawn(cleanup_pending_request(
            session.clone(),
            runtime.core_engine.clone(),
            runtime.completion_waiters.clone(),
            registration_id,
            runtime.telemetry.clone(),
            ownership,
        ));

        tokio::task::yield_now().await;
        cleanup.abort();
        assert!(cleanup
            .await
            .expect_err("cleanup task unexpectedly completed")
            .is_cancelled());
        assert_eq!(isolated_coordinator.snapshot().active_jobs, 1);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1,
            "aborting detached cleanup must retain model residency fail-closed"
        );

        drop(waiter_lock);
        assert!(
            remove_waiter_registration(
                runtime.completion_waiters.as_ref(),
                &request_id,
                registration_id,
            )
            .await
        );
        assert!(receiver.await.is_err());
        assert!(runtime
            .core_engine
            .abort_request_session(&session)
            .await
            .expect("manual exact abort"));
        assert_eq!(isolated_coordinator.snapshot().active_jobs, 1);
        assert_eq!(
            runtime
                .model_manager
                .active_residency_leases(residency_variant),
            1,
            "cancelled cleanup ownership is intentionally unrecoverable"
        );
    }

    #[test]
    fn retained_chat_observation_uses_allocated_capacities_and_media_sources() {
        use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind, ChatRole};

        let mut content = String::with_capacity(64);
        content.push('x');
        let mut messages = Vec::with_capacity(8);
        messages.push(ChatMessage {
            role: ChatRole::User,
            content,
        });

        let mut source = String::with_capacity(128);
        source.push_str("image.png");
        let mut media_inputs = Vec::with_capacity(3);
        media_inputs.push(ChatMediaInput {
            kind: ChatMediaKind::Image,
            source,
        });
        let config = ChatRequestConfig {
            media_inputs,
            ..ChatRequestConfig::default()
        };
        let correlation_id = {
            let mut value = String::with_capacity(32);
            value.push_str("correlation");
            value
        };

        let retained = retained_chat_preparation_input_bytes(
            &messages,
            messages.capacity(),
            &config,
            &GenerationParams::default(),
            Some(&correlation_id),
        )
        .unwrap();
        let minimum = messages.capacity() * std::mem::size_of::<ChatMessage>()
            + 64
            + config.media_inputs.capacity() * std::mem::size_of::<ChatMediaInput>()
            + 128
            + 32;

        assert_eq!(retained, minimum);
        assert!(retained > messages[0].content.len() + config.media_inputs[0].source.len());
    }

    #[test]
    fn speech_to_speech_preparation_accounts_allocated_capacities() {
        use crate::models::shared::chat::ChatRole;

        let mut content = String::with_capacity(64);
        content.push_str("hello");
        let mut messages = Vec::with_capacity(8);
        messages.push(ChatMessage {
            role: ChatRole::User,
            content,
        });
        let mut speaker = String::with_capacity(32);
        speaker.push_str("voice");
        let params = GenerationParams {
            speaker: Some(speaker),
            ..GenerationParams::default()
        };

        let retained = retained_speech_to_speech_preparation_input_bytes(
            4,
            &messages,
            messages.capacity(),
            &params,
            Some("system"),
            Some("correlation"),
        )
        .expect("retained input");
        let minimum = 4
            + messages.capacity() * std::mem::size_of::<ChatMessage>()
            + 64
            + 32
            + "system".len()
            + "correlation".len();

        assert_eq!(retained, minimum);
    }

    #[tokio::test]
    async fn runtime_routes_terminal_before_releasing_exact_session_fence() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("expired terminal routing")
            .with_model_variant(ModelVariant::Kokoro82M);
        request.id = "runtime-terminal-routing".to_string();
        request.prompt_tokens = vec![1];

        let (registration, completion) = runtime
            .register_waiter(&request.id)
            .await
            .expect("waiter registration");
        runtime
            .core_engine
            .add_request(request.clone())
            .await
            .expect("request admission");
        let session = runtime
            .core_engine
            .request_session_key(&request.id)
            .await
            .expect("request session");
        runtime
            .bind_waiter(&request.id, registration, session.epoch)
            .await
            .expect("waiter session binding");
        assert!(
            runtime
                .core_engine
                .set_request_hard_deadline_for_test(
                    &request.id,
                    Instant::now() - Duration::from_millis(1),
                )
                .await
        );

        let outputs = runtime
            .core_engine
            .step_for_dispatch()
            .await
            .expect("terminal step");
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finished);
        assert!(
            runtime
                .core_engine
                .add_request(request.clone())
                .await
                .is_err(),
            "the request ID must remain fenced while its terminal output awaits runtime routing"
        );

        route_terminal_output(
            runtime.core_engine.as_ref(),
            runtime.completion_waiters.as_ref(),
            runtime.telemetry.as_ref(),
            outputs.into_iter().next().unwrap(),
        )
        .await;
        assert!(matches!(
            completion.await.expect("completion channel"),
            Err(Error::Timeout(request_id)) if request_id == request.id
        ));

        runtime
            .core_engine
            .add_request(request)
            .await
            .expect("the request ID must be reusable after routing acknowledgement");
        runtime.core_engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fast_terminal_can_route_before_runtime_binds_atomic_admission_session() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("fast terminal waiter binding")
            .with_model_variant(ModelVariant::Kokoro82M);
        request.id = "runtime-fast-terminal-binding".to_string();
        request.prompt_tokens = vec![1];

        let (registration, completion) = runtime
            .register_waiter(&request.id)
            .await
            .expect("waiter registration");
        let session = runtime
            .core_engine
            .add_request_with_session(request.clone())
            .await
            .expect("atomic request admission");
        assert!(
            runtime
                .core_engine
                .set_request_hard_deadline_for_test(
                    &request.id,
                    Instant::now() - Duration::from_millis(1),
                )
                .await
        );
        let output = runtime
            .core_engine
            .step_for_dispatch()
            .await
            .expect("fast terminal step")
            .into_iter()
            .next()
            .expect("terminal output");
        assert_eq!(output.sequence_id, session.epoch);

        let engine = runtime.core_engine.clone();
        let waiters = runtime.completion_waiters.clone();
        let telemetry = runtime.telemetry.clone();
        let routing = tokio::spawn(async move {
            route_terminal_output(
                engine.as_ref(),
                waiters.as_ref(),
                telemetry.as_ref(),
                output,
            )
            .await;
        });
        // Let routing observe the intentionally unbound registration first.
        tokio::task::yield_now().await;
        runtime
            .bind_waiter(&request.id, registration, session.epoch)
            .await
            .expect("waiter session binding");
        tokio::time::timeout(Duration::from_secs(1), routing)
            .await
            .expect("terminal routing waited forever for binding")
            .expect("routing task panicked");
        assert!(matches!(
            completion.await.expect("completion channel"),
            Err(Error::Timeout(request_id)) if request_id == request.id
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_pending_guard_cleanup_cannot_abort_reused_session() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut old_request =
            EngineCoreRequest::tts("old exact session").with_model_variant(ModelVariant::Kokoro82M);
        old_request.id = "runtime-stale-pending-guard".to_string();
        old_request.prompt_tokens = vec![1];
        let request_id = old_request.id.clone();
        let (spec, observation) = runtime
            .coordinator_job_for_request(&old_request)
            .expect("job shape");
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .expect("job admission");
        let (registration, completion) = runtime
            .register_waiter(&request_id)
            .await
            .expect("waiter registration");
        let old_session = runtime
            .core_engine
            .add_request_with_session(old_request)
            .await
            .expect("old request admission");
        runtime
            .bind_waiter(&request_id, registration, old_session.epoch)
            .await
            .expect("old waiter binding");
        let guard = PendingRequestGuard::new(
            old_session.clone(),
            runtime.core_engine.clone(),
            runtime.completion_waiters.clone(),
            registration,
            runtime.telemetry.clone(),
            job,
            None,
        );

        assert!(runtime
            .core_engine
            .abort_request_session(&old_session)
            .await
            .expect("old exact abort"));
        let old_terminal = runtime
            .core_engine
            .step_for_dispatch()
            .await
            .expect("old cancellation step")
            .into_iter()
            .next()
            .expect("old cancellation output");
        route_terminal_output(
            runtime.core_engine.as_ref(),
            runtime.completion_waiters.as_ref(),
            runtime.telemetry.as_ref(),
            old_terminal,
        )
        .await;
        assert!(matches!(
            completion.await.expect("completion channel"),
            Err(Error::Cancelled(id)) if id == request_id
        ));

        let mut replacement = EngineCoreRequest::tts("replacement exact session")
            .with_model_variant(ModelVariant::Kokoro82M);
        replacement.id = request_id.clone();
        replacement.prompt_tokens = vec![2];
        let replacement_session = runtime
            .core_engine
            .add_request_with_session(replacement)
            .await
            .expect("replacement admission");
        assert_ne!(replacement_session.epoch, old_session.epoch);

        // The stale fallback runs after public-ID reuse. It must target only
        // the old epoch and leave the replacement request untouched.
        drop(guard);
        tokio::time::timeout(Duration::from_secs(1), async {
            while runtime.coordinator_snapshot().active_jobs != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stale pending cleanup did not finish");
        assert_eq!(
            runtime.core_engine.request_session_key(&request_id).await,
            Some(replacement_session)
        );
        runtime.core_engine.abort_all_requests().await;
    }

    #[tokio::test]
    async fn stale_terminal_output_cannot_steal_a_later_session_waiter() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("stale terminal routing")
            .with_model_variant(ModelVariant::Kokoro82M);
        request.id = "runtime-stale-terminal-routing".to_string();
        request.prompt_tokens = vec![1];
        runtime
            .core_engine
            .add_request(request.clone())
            .await
            .expect("old request admission");
        assert!(
            runtime
                .core_engine
                .set_request_hard_deadline_for_test(
                    &request.id,
                    Instant::now() - Duration::from_millis(1),
                )
                .await
        );
        let old_output = runtime
            .core_engine
            .step_for_dispatch()
            .await
            .expect("old terminal step")
            .into_iter()
            .next()
            .expect("old terminal output");

        let later_epoch = old_output
            .sequence_id
            .checked_add(1)
            .expect("session epoch");
        let (later_registration, mut later_completion) = runtime
            .register_waiter(&request.id)
            .await
            .expect("later waiter registration");
        runtime
            .bind_waiter(&request.id, later_registration, later_epoch)
            .await
            .expect("later waiter binding");

        route_terminal_output(
            runtime.core_engine.as_ref(),
            runtime.completion_waiters.as_ref(),
            runtime.telemetry.as_ref(),
            old_output,
        )
        .await;

        assert!(matches!(
            later_completion.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        let waiters = runtime.completion_waiters.lock().await;
        let later = waiters.get(&request.id).expect("later waiter retained");
        assert_eq!(later.registration_id, later_registration);
        assert_eq!(later.session_epoch, Some(later_epoch));
        drop(waiters);
        runtime.remove_waiter(&request.id, later_registration).await;
    }

    #[test]
    fn explicit_cuda_mismatch_gets_cuda_specific_error() {
        let context = BackendContext::new(
            BackendPreference::Cuda,
            BackendSelectionSource::Config,
            BackendCapabilities {
                cpu_compiled: true,
                metal_compiled: false,
                cuda_compiled: true,
            },
            DeviceProfile::cpu(),
            "Requested cuda backend fell back to cpu",
        );

        let err = RuntimeService::ensure_requested_backend_available(&context).unwrap_err();
        let message = err.to_string();

        assert!(message.contains("CUDA backend was requested"));
        assert!(message.contains("selected backend is `cpu`"));
        assert!(message.contains("no usable CUDA device"));
    }

    #[tokio::test]
    async fn runtime_prometheus_includes_engine_metric_values() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        let payload = runtime.telemetry_prometheus().await;

        assert!(payload.contains("izwi_engine_scheduler_queue_depth"));
        assert!(payload.contains("izwi_engine_scheduler_running_requests"));
        assert!(payload
            .contains("izwi_engine_kv_cache_allocated_blocks{accounting=\"physical_pages\"}"));
        assert!(payload
            .contains("izwi_engine_kv_cache_utilization_ratio{accounting=\"physical_pages\"}"));
        assert!(payload.contains(
            "izwi_engine_kv_cache_memory_capacity_bytes{accounting=\"resident_paged_plus_authorized_tensor\"}"
        ));
        assert!(payload.contains("allocated physical KV-cache pages"));
        assert!(payload
            .contains("Resident managed KV pages plus authorized retained tensor-state bytes"));
        assert!(!payload.contains("izwi_engine_kv_cache_soft_max_blocks"));
        assert!(!payload.contains("izwi_engine_kv_cache_copy_on_write_splits_total"));
        assert!(payload.contains("izwi_engine_stream_backpressure_total"));
        assert!(payload.contains("izwi_engine_stream_checkpoints_committed_total"));
        assert!(payload.contains("izwi_engine_stream_checkpoint_rejections_total"));
        assert!(payload.contains("izwi_engine_stream_delivery_failures_total"));
        assert!(payload.contains("izwi_engine_executor_tensor_batches_total"));
        assert!(payload.contains("izwi_engine_executor_request_parallel_batches_total"));
        assert!(payload.contains("izwi_engine_executor_tensor_batch_max_width"));
        assert!(payload.contains("izwi_engine_executor_tensor_static_batches_total"));
        assert!(payload.contains("izwi_engine_executor_tensor_continuous_batches_total"));
        assert!(payload.contains("izwi_engine_executor_tensor_continuous_multirow_batches_total"));
        assert!(payload.contains("izwi_engine_executor_model_decode_calls_total"));
        assert!(payload.contains("izwi_engine_executor_model_tensor_batches_total"));
        assert!(payload.contains("izwi_engine_executor_model_tensor_multirow_calls_total"));
        assert!(payload.contains("izwi_engine_executor_model_tensor_batch_rows_total"));
        assert!(payload.contains("izwi_engine_executor_model_tensor_batch_max_width"));
        assert!(payload.contains("izwi_engine_executor_model_scalar_row_dispatches_total"));
        assert!(payload.contains("izwi_engine_executor_continuous_envelope_scalar_fallbacks_total"));
        assert!(payload.contains("izwi_engine_executor_physical_batch_rejections_total"));
        assert!(payload
            .contains("izwi_engine_executor_dispatch_state_rows_total{state=\"not_started\"}"));
        assert!(
            payload.contains("izwi_engine_executor_failure_origin_rows_total{origin=\"model\"}")
        );
        assert!(payload
            .contains("izwi_engine_executor_deadline_phase_rows_total{phase=\"dispatch_wait\"}"));
        assert!(payload.contains(
            "izwi_engine_executor_batch_workspace_domain_bytes_total{domain=\"device\"}"
        ));
        assert!(payload.contains("izwi_engine_executor_tensor_batch_fill_ratio"));
        assert!(payload.contains("izwi_engine_executor_tensor_batch_padding_ratio"));
        assert!(payload.contains("izwi_engine_executor_physical_execution_mode{mode=\"serial\"}"));
        assert!(payload.contains("izwi_engine_executor_physical_execution_cap"));
        assert!(payload.contains("izwi_engine_executor_physical_dispatches_in_flight"));
        assert!(payload.contains("izwi_engine_executor_physical_dispatch_seconds_total"));
        assert!(payload.contains(
            "izwi_engine_executor_physical_fallbacks_total{reason=\"uncertified_profile\"}"
        ));
        assert!(payload
            .contains("izwi_engine_executor_physical_defers_total{reason=\"workspace_capacity\"}"));
        assert!(payload.contains(
            "izwi_engine_executor_physical_workspace_high_water_bytes{domain=\"device\"}"
        ));
        assert!(payload.contains("izwi_engine_executor_physical_batch_fill_ratio"));
        assert!(payload.contains("# TYPE izwi_inference_coordinator_active_jobs gauge"));
        assert!(payload.contains("# TYPE izwi_inference_coordinator_admitted_total counter"));
        assert!(payload.contains("izwi_inference_coordinator_reserved_memory_bytes"));
        assert!(payload.contains("izwi_inference_coordinator_reserved_host_memory_bytes"));
        assert!(payload.contains("izwi_inference_coordinator_reserved_device_memory_bytes"));
        assert!(payload.contains("izwi_inference_coordinator_reserved_unified_memory_bytes"));
        assert!(payload.contains("izwi_inference_coordinator_draining 0"));
        assert!(payload.contains("izwi_inference_coordinator_poisoned 0"));

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.coordinator, runtime.coordinator_snapshot());
        assert!(snapshot.engine.physical_execution.effective_cap >= 1);
        let serialized = serde_json::to_value(&snapshot).expect("serialize runtime telemetry");
        let effective_mode = serialized["engine"]["physical_execution"]["effective_mode"]
            .as_str()
            .expect("bounded effective physical execution mode");
        assert!(["serial", "shadow", "concurrent"].contains(&effective_mode));
        assert_eq!(
            snapshot.engine.kv_cache.memory_accounting,
            "resident_paged_plus_authorized_tensor"
        );
        assert_eq!(snapshot.engine.kv_cache.totals.models, 0);
        assert_eq!(
            snapshot.engine.kv_cache.totals.coordinator.capacity_pages,
            0
        );
    }

    #[tokio::test]
    async fn runtime_drain_is_observable_and_completes_when_idle() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        runtime
            .wait_for_drain(Duration::from_millis(50))
            .await
            .unwrap();

        assert!(runtime.is_draining());
        assert!(runtime.telemetry_snapshot().await.coordinator.draining);
        assert!(runtime
            .telemetry_prometheus()
            .await
            .contains("izwi_inference_coordinator_draining 1"));
    }

    #[test]
    fn managed_kv_prometheus_projections_use_physical_arenas() {
        use crate::engine::{
            ManagedKvArenaRuntimeSnapshot, ManagedKvCoordinatorSnapshot,
            ManagedKvModelRuntimeSnapshot, ManagedKvOperationSnapshot, ModelInstanceId,
        };

        let arena = |allocated_pages| ManagedKvArenaRuntimeSnapshot {
            generation: 0,
            group_id: 1,
            domain_id: 1,
            device_ordinal: Some(0),
            page_tokens: 16,
            token_capacity: 160,
            bytes_per_page: 128,
            physical_bytes: 1_280,
            coordinator: ManagedKvCoordinatorSnapshot {
                capacity_pages: 10,
                allocated_pages,
                free_pages: 10 - allocated_pages,
                ..ManagedKvCoordinatorSnapshot::default()
            },
            operations: ManagedKvOperationSnapshot::default(),
        };
        let model = |model_instance, backend, allocated_pages| ManagedKvModelRuntimeSnapshot {
            model_instance: ModelInstanceId::new(model_instance),
            plan_fingerprint: format!("plan-{model_instance}"),
            state_plan_v2_fingerprint: format!("state-plan-v2-{model_instance}"),
            backend,
            device_ordinal: Some(0),
            resident_paged_bytes: 1_280,
            authorized_tensor_bytes: 0,
            physical_bytes: 1_280,
            registered_sessions: 1,
            single_sequence_token_capacity: 160,
            full_context_sequence_capacity: 1,
            arenas: vec![arena(allocated_pages)],
        };
        let snapshot = crate::engine::ManagedKvRuntimeSnapshot {
            models: vec![
                model(1, BackendKind::Cpu, 3),
                model(2, BackendKind::Metal, 4),
                model(3, BackendKind::Cuda, 5),
            ],
            ..Default::default()
        };

        assert_eq!(managed_kv_used_bytes(&snapshot), 12 * 128);
        assert_eq!(managed_kv_device_pages(&snapshot), 9);
    }

    #[tokio::test]
    async fn streaming_requests_are_validated_as_streaming_by_broker() {
        let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::On);
        let request =
            EngineCoreRequest::asr("audio").with_model_variant(ModelVariant::WhisperLargeV3Turbo);

        let err = runtime
            .run_streaming_request(request, |_| std::future::ready(Ok(())))
            .await
            .expect_err("batch-only ASR should be rejected before streaming execution");

        assert!(err.to_string().contains("not streaming execution"));

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.observability.stage_observations_total, 1);
        assert_eq!(snapshot.observability.stage_failures_total, 1);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .pipeline_stage
                .as_deref(),
            Some("runtime.routing")
        );
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .error_kind
                .as_deref(),
            Some("routing_validation_failed")
        );
    }

    #[tokio::test]
    async fn transport_streaming_requests_can_validate_as_offline_broker_execution() {
        let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
        let mut request =
            EngineCoreRequest::asr("audio").with_model_variant(ModelVariant::ParakeetTdt06BV3);
        request.streaming = true;

        runtime
            .observe_broker_request_with_streaming_required(&request, false)
            .expect("transport streaming should validate as offline ASR execution");

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.broker.shadow_requests, 1);
        assert_eq!(snapshot.broker.route_decisions, 1);
        assert_eq!(snapshot.broker.validation_failures, 0);
        assert_eq!(snapshot.observability.stage_observations_total, 1);
        assert_eq!(snapshot.observability.stage_failures_total, 0);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .execution_target
                .as_deref(),
            Some("TokenEngine")
        );
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .streaming_mode
                .as_deref(),
            Some("None")
        );
    }

    #[tokio::test]
    async fn voice_runtime_events_record_stage_observations() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        runtime.record_voice_session_started();

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.voice.sessions_started, 1);
        assert_eq!(snapshot.observability.stage_observations_total, 1);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .route_source
                .as_deref(),
            Some("RealtimeVoice")
        );
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .pipeline_stage
                .as_deref(),
            Some("voice.session_started")
        );
    }

    #[tokio::test]
    async fn direct_capability_observation_records_broker_telemetry() {
        let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);

        runtime
            .observe_broker_capability_request(
                CapabilityKind::Tts,
                Some(ModelVariant::Kokoro82M),
                true,
            )
            .expect("direct capability observation should validate");

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.broker.shadow_requests, 1);
        assert_eq!(snapshot.broker.execution_requests, 0);
        assert_eq!(snapshot.broker.route_decisions, 1);
        assert_eq!(snapshot.broker.validation_failures, 0);
        assert_eq!(snapshot.observability.stage_observations_total, 1);
        assert_eq!(snapshot.observability.stage_failures_total, 0);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .pipeline_stage
                .as_deref(),
            Some("runtime.routing")
        );
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .model_variant
                .as_deref(),
            Some(ModelVariant::Kokoro82M.dir_name())
        );
    }

    #[tokio::test]
    async fn batch_pipeline_observation_records_pipeline_telemetry() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        runtime.record_batch_asr_pipeline_job();
        runtime.record_batch_tts_pipeline_job();

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.pipelines.batch_asr_transcriptions, 1);
        assert_eq!(snapshot.pipelines.batch_tts_speech, 1);
        assert_eq!(snapshot.pipelines.stages_recorded, 8);
    }

    #[test]
    fn transient_estimates_are_fully_known_for_every_backend() {
        const BASE_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
        const INPUT_BYTES: usize = 1024;
        let host_preparation_bytes = BASE_WORKSPACE_BYTES + (INPUT_BYTES as u64 * 8);
        let cpu = transient_resources(BackendKind::Cpu, INPUT_BYTES);
        let metal = transient_resources(BackendKind::Metal, INPUT_BYTES);
        let cuda = transient_resources(BackendKind::Cuda, INPUT_BYTES);

        assert!(cpu.is_fully_known());
        assert!(metal.is_fully_known());
        assert!(cuda.is_fully_known());
        assert_eq!(
            cpu.host_bytes,
            ResourceAmount::Known(host_preparation_bytes)
        );
        assert_eq!(
            metal.unified_bytes,
            ResourceAmount::Known(host_preparation_bytes)
        );
        assert_eq!(
            cuda.host_bytes,
            ResourceAmount::Known(host_preparation_bytes)
        );
        assert_eq!(
            cuda.device_bytes,
            ResourceAmount::Known(BASE_WORKSPACE_BYTES)
        );
    }

    #[test]
    fn media_preparation_estimates_map_to_physical_backend_domains() {
        let estimate = crate::models::architectures::qwen35::Qwen35MediaResourceEstimate {
            host_bytes: 300,
            backend_tensor_bytes: 700,
        };
        let cpu = media_preparation_resources(BackendKind::Cpu, estimate).unwrap();
        let metal = media_preparation_resources(BackendKind::Metal, estimate).unwrap();
        let cuda = media_preparation_resources(BackendKind::Cuda, estimate).unwrap();

        assert_eq!(cpu.host_bytes, ResourceAmount::Known(1_000));
        assert_eq!(metal.unified_bytes, ResourceAmount::Known(1_000));
        assert_eq!(cuda.host_bytes, ResourceAmount::Known(300));
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(700));
        assert!(cpu.is_fully_known());
        assert!(metal.is_fully_known());
        assert!(cuda.is_fully_known());
    }

    #[test]
    fn audio_decode_workspace_maps_to_host_or_unified_memory() {
        let cpu = audio_decode_resources(BackendKind::Cpu);
        let metal = audio_decode_resources(BackendKind::Metal);
        let cuda = audio_decode_resources(BackendKind::Cuda);

        assert_eq!(
            cpu.host_bytes,
            ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        );
        assert_eq!(
            metal.unified_bytes,
            ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        );
        assert_eq!(
            cuda.host_bytes,
            ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
        );
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(0));
        assert!(task_decodes_audio(TaskType::ASR));
        assert!(task_decodes_audio(TaskType::SpeechToSpeech));
        assert!(!task_decodes_audio(TaskType::TTS));
        assert!(!task_decodes_audio(TaskType::Chat));
    }

    #[test]
    fn direct_audio_jobs_include_decoder_workspace_before_admission() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let backend = runtime.backend_context().backend_kind;
        let input_bytes = 4096;
        let context = RuntimeRequestContext::default();
        let expected = transient_resources(backend, input_bytes)
            .checked_add(audio_decode_resources(backend))
            .expect("resource estimate");
        let spec = runtime
            .coordinator_job_for_audio_input(
                "direct-asr-audio",
                CoordinatorLane::Atomic,
                context,
                input_bytes,
            )
            .expect("direct audio job");

        assert_eq!(spec.resources, expected);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preparation_copy_yields_between_bounded_quanta() {
        use std::sync::atomic::AtomicBool;

        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let input = vec![7_u8; PREPARATION_COPY_QUANTUM_BYTES * 3 + 17];
        let spec = runtime
            .coordinator_job_for_audio_input(
                "yielding-audio-copy",
                CoordinatorLane::Atomic,
                RuntimeRequestContext::default(),
                input.len(),
            )
            .expect("audio job");
        let job = runtime.coordinator.admit(spec).await.expect("admission");
        let peer_ran = Arc::new(AtomicBool::new(false));
        let task_peer_ran = peer_ran.clone();
        let peer = tokio::spawn(async move {
            task_peer_ran.store(true, Ordering::Release);
        });

        let copied = copy_preparation_bytes(&job, &input, "test audio")
            .await
            .expect("copy");
        assert!(
            peer_ran.load(Ordering::Acquire),
            "a multi-quantum copy must yield to another Tokio task"
        );
        let mut utf8_input = "x".repeat(PREPARATION_COPY_QUANTUM_BYTES - 1);
        utf8_input.push('é');
        let utf8_copy = copy_preparation_string(&job, &utf8_input, "test base64")
            .await
            .expect("UTF-8 copy");

        assert_eq!(copied, input);
        assert_eq!(utf8_copy, utf8_input);
        peer.await.expect("peer task");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preparation_copy_stops_at_absolute_deadline() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let deadline = Instant::now() + Duration::from_millis(50);
        let context = RuntimeRequestContext::default().with_deadline(deadline);
        let spec = runtime
            .coordinator_job_for_audio_input(
                "expired-audio-copy",
                CoordinatorLane::Atomic,
                context,
                8,
            )
            .expect("audio job");
        let job = runtime.coordinator.admit(spec).await.expect("admission");
        tokio::time::sleep(Duration::from_millis(60)).await;

        let error = copy_preparation_bytes(&job, &[0_u8; 8], "expired audio")
            .await
            .expect_err("copy must honor its absolute deadline");

        assert!(matches!(error, Error::Timeout(request_id) if request_id == "expired-audio-copy"));
    }

    #[test]
    fn core_audio_requests_include_decoder_workspace_in_job_authorization() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let backend = runtime.backend_context().backend_kind;

        let asr = EngineCoreRequest::asr("AAAA");
        let speech_to_speech = EngineCoreRequest::speech_to_speech("AAAA");
        let plain_tts = EngineCoreRequest::tts("hello");
        let mut reference_tts = EngineCoreRequest::tts("hello");
        reference_tts.reference_audio = Some("AAAA".to_string());
        reference_tts.reference_text = Some("reference transcript".to_string());
        let mut canonical_reference_tts = reference_tts.clone();
        canonical_reference_tts
            .canonicalize_direct_payloads(runtime.config.portable_context_ceiling())
            .expect("canonical TTS reference");
        assert!(canonical_reference_tts.reference_audio.is_none());
        assert!(canonical_reference_tts.has_tts_reference_for_execution());

        for request in [
            &asr,
            &speech_to_speech,
            &reference_tts,
            &canonical_reference_tts,
        ] {
            let input_bytes =
                retained_engine_request_input_bytes(request).expect("retained request input");
            let expected = transient_resources(backend, input_bytes)
                .checked_add(audio_decode_resources(backend))
                .expect("resource estimate");
            let (spec, _) = runtime
                .coordinator_job_for_request(request)
                .expect("coordinator job");

            assert_eq!(spec.resources, expected);
        }

        let plain_tts_input =
            retained_engine_request_input_bytes(&plain_tts).expect("retained request input");
        let expected_plain_tts = transient_resources(backend, plain_tts_input);
        let (plain_tts_spec, _) = runtime
            .coordinator_job_for_request(&plain_tts)
            .expect("coordinator job");
        assert_eq!(plain_tts_spec.resources, expected_plain_tts);
    }

    #[test]
    fn direct_qwen_asr_job_accounts_the_prepared_audio_artifact_exactly() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let backend = runtime.backend_context().backend_kind;
        let variant = ModelVariant::Qwen3Asr06BGguf;
        let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
        let before = retained_engine_request_input_bytes(&request).unwrap();
        request
            .install_prepared_asr_audio(variant, vec![0.0; 257], 16_000)
            .unwrap();
        request
            .install_prepared_sequence_input_tokens(32, 4096)
            .unwrap();
        let after = retained_engine_request_input_bytes(&request).unwrap();
        assert_eq!(after - before, 257 * std::mem::size_of::<f32>());

        let (spec, observation) = runtime
            .coordinator_job_for_request(&request)
            .expect("prepared Qwen ASR coordinator job");
        assert_eq!(observation.host_bytes, after as u64);
        assert_eq!(spec.resources, transient_resources(backend, after));
    }

    #[test]
    fn engine_requests_use_truthful_controller_lanes() {
        let mut qwen_tts = EngineCoreRequest::tts("hello");
        qwen_tts.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
        assert_eq!(
            coordinator_lane_for_request(&qwen_tts),
            CoordinatorLane::Resumable
        );

        let mut offline_asr = EngineCoreRequest::asr("audio");
        offline_asr.model_variant = Some(ModelVariant::Qwen3Asr06BGguf);
        assert_eq!(
            coordinator_lane_for_request(&offline_asr),
            CoordinatorLane::Resumable
        );
        offline_asr.streaming = true;
        assert_eq!(
            coordinator_lane_for_request(&offline_asr),
            CoordinatorLane::Resumable
        );
        offline_asr
            .install_prepared_asr_audio(ModelVariant::Qwen3Asr06BGguf, vec![0.0; 16_000], 16_000)
            .unwrap();
        offline_asr.install_prepared_asr_long_form_atomic().unwrap();
        assert_eq!(
            coordinator_lane_for_request(&offline_asr),
            CoordinatorLane::Atomic
        );
        offline_asr.workload_class = WorkloadClass::Realtime;
        assert_eq!(
            coordinator_lane_for_request(&offline_asr),
            CoordinatorLane::Realtime
        );
    }
}

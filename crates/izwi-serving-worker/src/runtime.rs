use super::{
    AdmissionFailure, AdmittedExecution, AdmittedInvocation, ExecutionEvent, ExecutionFailure,
    ExecutionTeardown, InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner,
    RealtimeTtsOptions, RealtimeTtsStageStream,
};
use async_trait::async_trait;
use izwi_core::{
    engine::{OutputFinishReason, WorkloadClass},
    resolve_chat_sampling, AudioChunk, ChatGenerationControls, ChatMessage as CoreChatMessage,
    ChatRequestConfig, ChatRole as CoreChatRole, ChatSamplingRequest, ChatTruncation,
    Error as CoreError, ErrorKind, GenerationConfig, GenerationParams, GenerationRequest,
    ModelVariant, RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream, RuntimeChatInvocation,
    RuntimeChatInvocationEvent, RuntimeChatInvocationRequest, RuntimeChatTeardownDisposition,
    RuntimeHealth, RuntimeRequestContext, RuntimeService, RuntimeTelemetrySnapshot,
    CHAT_MAX_TOKENS_UNSET,
};
use izwi_serving_protocol::{
    ChatParameters, ChatResponseFormat, ChatRole, ChatTruncation as WireChatTruncation,
    FinishReason, InvocationErrorCode, InvocationInput, InvocationRequest, RealtimeAudioCodec,
    RealtimeAudioSpec, RejectionCode, ServiceClass, TaskKind,
};
use std::{collections::VecDeque, sync::Arc, time::Duration};

const MAX_WARMUP_EVENTS: usize = 4;
const MAX_WARMUP_TEXT_BYTES: usize = 64 * 1024;

/// Execute one bounded real chat request before a worker advertises readiness.
///
/// Deadline expiry requests cancellation and still awaits the core teardown
/// proof; the caller must not bind a listener when this function fails.
pub async fn warm_up_chat_runtime(
    runtime: &RuntimeService,
    variant: ModelVariant,
    timeout: Duration,
) -> Result<(), CoreError> {
    if timeout.is_zero() {
        return Err(CoreError::ConfigError(
            "worker warm-up timeout must be non-zero".into(),
        ));
    }
    let runtime_context = RuntimeRequestContext::new(WorkloadClass::Interactive)
        .with_deadline(std::time::Instant::now() + timeout);
    let mut invocation = runtime
        .start_chat_invocation(RuntimeChatInvocationRequest {
            variant,
            messages: vec![CoreChatMessage {
                role: CoreChatRole::User,
                content: "Hello".into(),
            }],
            params: GenerationParams {
                temperature: 0.0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                max_tokens: 1,
                ..GenerationParams::default()
            },
            chat_config: Default::default(),
            correlation_id: Some("worker-startup-warmup".into()),
            runtime_context,
            streaming: false,
            max_tokens_ceiling: None,
        })
        .await?;
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    let mut event_count = 0usize;
    let mut text_bytes = 0usize;
    let mut completed = false;
    let mut failure = None;
    loop {
        tokio::select! {
            () = &mut deadline => {
                failure = Some(CoreError::Timeout("worker startup warm-up".into()));
                break;
            }
            event = invocation.next_event() => {
                match event {
                    Ok(Some(RuntimeChatInvocationEvent::TextDelta { text, .. })) => {
                        event_count = event_count.saturating_add(1);
                        text_bytes = text_bytes.saturating_add(text.len());
                    }
                    Ok(Some(RuntimeChatInvocationEvent::Completed(generation))) => {
                        event_count = event_count.saturating_add(1);
                        text_bytes = text_bytes.saturating_add(generation.text.len());
                        if event_count > MAX_WARMUP_EVENTS || text_bytes > MAX_WARMUP_TEXT_BYTES {
                            failure = Some(CoreError::InferenceError(
                                "worker startup warm-up exceeded its output bounds".into(),
                            ));
                        } else {
                            completed = true;
                        }
                        break;
                    }
                    Ok(None) => {
                        failure = Some(CoreError::InferenceError(
                            "worker startup warm-up ended without completion".into(),
                        ));
                        break;
                    }
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
                if event_count > MAX_WARMUP_EVENTS || text_bytes > MAX_WARMUP_TEXT_BYTES {
                    failure = Some(CoreError::InferenceError(
                        "worker startup warm-up exceeded its output bounds".into(),
                    ));
                    break;
                }
            }
        }
    }
    if failure.is_some() {
        invocation.request_cancel();
    }
    let teardown = invocation.wait_for_teardown().await?;
    if let Some(error) = failure {
        return Err(error);
    }
    if !completed || teardown.disposition != RuntimeChatTeardownDisposition::Completed {
        return Err(CoreError::InferenceError(format!(
            "worker startup warm-up teardown was {:?}",
            teardown.disposition
        )));
    }
    Ok(())
}

/// Phase-2 adapter for one already selected chat model in an existing Izwi runtime.
///
/// Device selection and model loading happen while constructing `RuntimeService`;
/// individual requests cannot change either value.
pub struct RuntimeChatExecutor {
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
    streaming: bool,
}

impl RuntimeChatExecutor {
    pub fn new(runtime: Arc<RuntimeService>, variant: ModelVariant, streaming: bool) -> Self {
        Self {
            runtime,
            variant,
            streaming,
        }
    }
}

#[async_trait]
impl InvocationExecutor for RuntimeChatExecutor {
    async fn runtime_telemetry(&self) -> Option<RuntimeTelemetrySnapshot> {
        Some(self.runtime.telemetry_snapshot().await)
    }

    fn runtime_health(&self) -> RuntimeHealth {
        self.runtime.health()
    }

    async fn admit(
        &self,
        request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        let InvocationInput::Chat { input, parameters } = &request.input;
        let messages = input
            .messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    ChatRole::System => CoreChatRole::System,
                    ChatRole::User => CoreChatRole::User,
                    ChatRole::Assistant => CoreChatRole::Assistant,
                    // Every family's prompt builder renders a tool turn.
                    ChatRole::Tool => CoreChatRole::Tool,
                };
                Ok(CoreChatMessage {
                    role,
                    content: message.content.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ceiling = request.output_limits.max_tokens as usize;
        let (params, chat_config, max_tokens_ceiling) =
            chat_request(self.variant, parameters, ceiling);
        let workload_class = match request.service_class {
            ServiceClass::Realtime => WorkloadClass::Realtime,
            ServiceClass::Interactive if self.streaming => WorkloadClass::Streaming,
            ServiceClass::Interactive => WorkloadClass::Interactive,
            ServiceClass::Batch => WorkloadClass::Batch,
        };
        // The gateway-attested tenant scopes fair scheduling and the managed
        // prefix-cache namespace: tenants on one worker never share KV pages.
        let runtime_context = RuntimeRequestContext::new(workload_class)
            .with_deadline(
                std::time::Instant::now() + Duration::from_millis(request.remaining_time_ms),
            )
            .with_tenant_id(request.caller.tenant_id.as_str());
        let invocation = self
            .runtime
            .start_chat_invocation(RuntimeChatInvocationRequest {
                variant: self.variant,
                messages,
                params,
                chat_config,
                correlation_id: Some(request.request_id.as_str().to_owned()),
                runtime_context,
                streaming: self.streaming,
                max_tokens_ceiling,
            })
            .await
            .map_err(map_admission_error)?;
        Ok(AdmittedInvocation::new(Box::new(
            RuntimeAdmittedExecution {
                invocation: Some(invocation),
                streaming: self.streaming,
                pending: VecDeque::with_capacity(1),
            },
        )))
    }
}

/// The engine request for one invocation's chat parameters, built the way
/// the local server builds it: an absent sampling value takes the model
/// family's profile, else the engine default (`resolve_chat_sampling`).
/// `ceiling` is the invocation's output limit.
fn chat_request(
    variant: ModelVariant,
    parameters: &ChatParameters,
    ceiling: usize,
) -> (GenerationParams, ChatRequestConfig, Option<usize>) {
    let sampling = resolve_chat_sampling(
        variant,
        None,
        ChatSamplingRequest {
            temperature: parameters.temperature,
            top_p: parameters.top_p,
            top_k: parameters.top_k.map(|top_k| top_k as usize),
            repetition_penalty: parameters.repetition_penalty,
            presence_penalty: parameters.presence_penalty,
        },
    );
    // D8 as on the local server: the caller's own budget, or the rest of the
    // context when it set none, capped at the gateway's ceiling. A gateway
    // before minor 4 sends no policy; its ceiling is the budget and the
    // request truncates to fit, as it always did.
    let (max_tokens, truncation, max_tokens_ceiling) = match parameters.context {
        Some(context) => (
            context
                .max_tokens
                .map_or(CHAT_MAX_TOKENS_UNSET, |max_tokens| max_tokens as usize),
            match context.truncation {
                WireChatTruncation::Disabled => ChatTruncation::Disabled,
                WireChatTruncation::Auto => ChatTruncation::Auto,
            },
            Some(ceiling),
        ),
        None => (ceiling, ChatTruncation::Auto, None),
    };
    let params = GenerationParams::for_chat(
        sampling,
        max_tokens,
        ChatGenerationControls {
            seed: parameters.seed,
            stop: parameters.stop.clone(),
            logprobs: parameters.logprobs.map(usize::from),
            constrain_json_object: parameters.response_format
                == Some(ChatResponseFormat::JsonObject),
            keep_special_tokens: parameters.skip_special_tokens == Some(false),
        },
    );
    let chat_config = ChatRequestConfig {
        truncation,
        tools: parameters.tools.clone(),
        ..ChatRequestConfig::default()
    };
    (params, chat_config, max_tokens_ceiling)
}

struct RuntimeAdmittedExecution {
    invocation: Option<RuntimeChatInvocation>,
    streaming: bool,
    // At most one terminal event follows a non-streaming text result.
    pending: VecDeque<ExecutionEvent>,
}

#[async_trait]
impl AdmittedExecution for RuntimeAdmittedExecution {
    async fn next_event(&mut self) -> Option<ExecutionEvent> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        let invocation = self
            .invocation
            .as_mut()
            .expect("runtime invocation exists until teardown");
        match invocation.next_event().await {
            Ok(Some(RuntimeChatInvocationEvent::TextDelta { text, logprobs })) => {
                Some(ExecutionEvent::TextDelta { text, logprobs })
            }
            Ok(Some(RuntimeChatInvocationEvent::Completed(generation))) => {
                let completed = ExecutionEvent::Completed {
                    text: None,
                    finish_reason: map_finish_reason(generation.finish_reason),
                    input_tokens: generation.prompt_tokens as u64,
                    output_tokens: generation.tokens_generated as u64,
                    cached_input_tokens: generation.cached_prompt_tokens,
                };
                if self.streaming || generation.text.is_empty() {
                    Some(completed)
                } else {
                    self.pending.push_back(completed);
                    Some(ExecutionEvent::TextDelta {
                        text: generation.text,
                        logprobs: generation.logprobs,
                    })
                }
            }
            Ok(None) => None,
            Err(error) => Some(ExecutionEvent::Failed(map_execution_error(error))),
        }
    }

    fn request_cancel(&self) {
        if let Some(invocation) = &self.invocation {
            invocation.request_cancel();
        }
    }

    async fn wait_for_teardown(mut self: Box<Self>) -> ExecutionTeardown {
        let Some(invocation) = self.invocation.take() else {
            return ExecutionTeardown::Failed;
        };
        match invocation.wait_for_teardown().await {
            Ok(teardown) => match teardown.disposition {
                RuntimeChatTeardownDisposition::Completed => ExecutionTeardown::Completed,
                RuntimeChatTeardownDisposition::Cancelled => ExecutionTeardown::Cancelled,
                RuntimeChatTeardownDisposition::Failed => ExecutionTeardown::Failed,
            },
            Err(_) => ExecutionTeardown::Unconfirmed,
        }
    }
}

fn map_finish_reason(reason: Option<OutputFinishReason>) -> FinishReason {
    match reason {
        Some(OutputFinishReason::MaxTokens) => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

/// The rejection a worker sends for a core error kind before it accepts an
/// invocation. The gateway maps the code through the same error classes as
/// the local role maps the kind, except where the worker's view differs: a
/// model the gateway routed here that this worker cannot serve right now
/// (not found, busy, an internal fault, a missing dependency) is "not
/// ready", so the gateway tries another worker; a queue timeout is a queue
/// wait; and a cancellation the client did not ask for means the worker is
/// draining.
pub fn admission_rejection_code(kind: ErrorKind) -> RejectionCode {
    match kind {
        ErrorKind::InvalidInput => RejectionCode::InvalidRequest,
        ErrorKind::ContextLengthExceeded => RejectionCode::ContextLengthExceeded,
        ErrorKind::CapacityExhausted => RejectionCode::CapacityExhausted,
        ErrorKind::Timeout => RejectionCode::QueueWaitExceeded,
        ErrorKind::Cancelled => RejectionCode::WorkerDraining,
        ErrorKind::ModelNotFound
        | ErrorKind::ModelBusy
        | ErrorKind::Unavailable
        | ErrorKind::Internal => RejectionCode::ModelNotReady,
    }
}

/// The terminal error code a worker sends for a core error kind after it
/// accepted an invocation.
pub fn execution_error_code(kind: ErrorKind) -> InvocationErrorCode {
    match kind {
        ErrorKind::InvalidInput | ErrorKind::ContextLengthExceeded => {
            InvocationErrorCode::InvalidInput
        }
        ErrorKind::Timeout => InvocationErrorCode::DeadlineExceeded,
        ErrorKind::CapacityExhausted | ErrorKind::Unavailable => {
            InvocationErrorCode::WorkerUnavailable
        }
        ErrorKind::ModelNotFound
        | ErrorKind::ModelBusy
        | ErrorKind::Cancelled
        | ErrorKind::Internal => InvocationErrorCode::ExecutionFailed,
    }
}

/// The message a gateway shows for `error`: an overflow is OpenAI's bare
/// explanation, as on the local role; anything else is the error's text.
fn client_message(error: CoreError) -> String {
    match error {
        CoreError::ContextLengthExceeded(message) => message,
        error => error.to_string(),
    }
}

fn map_admission_error(error: CoreError) -> AdmissionFailure {
    AdmissionFailure::new(
        admission_rejection_code(error.kind()),
        client_message(error),
    )
}

pub(crate) fn map_execution_error(error: CoreError) -> ExecutionFailure {
    ExecutionFailure {
        code: execution_error_code(error.kind()),
        message: client_message(error),
    }
}

const ASR_WARMUP_SAMPLE_RATE: u32 = 16_000;
const ASR_WARMUP_FRAME_SAMPLES: usize = 1_600;
const ASR_WARMUP_FRAMES: usize = 5;

/// Execute one bounded realtime-ASR stream before a worker advertises
/// readiness: start the stream, push a few frames of 16 kHz silence, and
/// finish. This proves the deployed variant resolves back through the
/// runtime's own ASR model resolution and that the realtime stream decode
/// path actually executes; the caller must not bind a listener on failure.
pub async fn warm_up_asr_runtime(
    runtime: &RuntimeService,
    variant: ModelVariant,
    timeout: Duration,
) -> Result<(), CoreError> {
    if timeout.is_zero() {
        return Err(CoreError::ConfigError(
            "worker warm-up timeout must be non-zero".into(),
        ));
    }
    let model_name = variant.dir_name();
    let resolved = izwi_core::resolve_asr_model_variant(Some(model_name));
    if resolved != variant {
        return Err(CoreError::ConfigError(format!(
            "worker ASR model {model_name} does not resolve back to the deployed variant"
        )));
    }
    let started = tokio::time::timeout(
        timeout,
        runtime.try_start_asr_realtime_stream(Some(model_name), None, None),
    )
    .await
    .map_err(|_| CoreError::Timeout("worker startup warm-up".into()))??;
    let Some(mut stream) = started else {
        return Err(CoreError::ConfigError(format!(
            "deployed ASR model {model_name} does not support realtime stream decode"
        )));
    };
    let silence = vec![0.0_f32; ASR_WARMUP_FRAME_SAMPLES];
    for _ in 0..ASR_WARMUP_FRAMES {
        let pushed = tokio::time::timeout(
            timeout,
            runtime.push_asr_realtime_samples(&mut stream, &silence, ASR_WARMUP_SAMPLE_RATE),
        )
        .await
        .map_err(|_| CoreError::Timeout("worker startup warm-up".into()))??;
        drop(pushed);
    }
    tokio::time::timeout(timeout, runtime.finish_asr_realtime_stream(&mut stream))
        .await
        .map_err(|_| CoreError::Timeout("worker startup warm-up".into()))??;
    Ok(())
}

/// Serve realtime ASR sessions for one already loaded ASR model variant.
///
/// HTTP invocations are rejected for this executor: speech-to-text workers
/// serve realtime WebSocket sessions only, and the deployment's capability
/// gates keep chat traffic away before this executor is ever consulted.
pub struct RuntimeRealtimeAsrExecutor {
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
}

impl RuntimeRealtimeAsrExecutor {
    pub fn new(runtime: Arc<RuntimeService>, variant: ModelVariant) -> Self {
        Self { runtime, variant }
    }
}

#[async_trait]
impl InvocationExecutor for RuntimeRealtimeAsrExecutor {
    async fn runtime_telemetry(&self) -> Option<RuntimeTelemetrySnapshot> {
        Some(self.runtime.telemetry_snapshot().await)
    }

    fn runtime_health(&self) -> RuntimeHealth {
        self.runtime.health()
    }

    async fn admit(
        &self,
        _request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        Err(AdmissionFailure::new(
            RejectionCode::IncompatibleTask,
            "speech_to_text workers serve realtime WebSocket sessions only; HTTP invocations are not accepted",
        ))
    }

    fn realtime_runner(&self) -> Option<Arc<dyn RealtimeStageRunner>> {
        Some(Arc::new(RuntimeRealtimeAsrRunner {
            runtime: Arc::clone(&self.runtime),
            variant: self.variant,
        }))
    }
}

/// Realtime ASR stage execution against the worker's runtime.
pub struct RuntimeRealtimeAsrRunner {
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
}

#[async_trait]
impl RealtimeStageRunner for RuntimeRealtimeAsrRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::SpeechToText
    }

    async fn start_asr_stream(
        &self,
        language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, CoreError> {
        let model_name = self.variant.dir_name();
        let resolved = izwi_core::resolve_asr_model_variant(Some(model_name));
        if resolved != self.variant {
            return Err(CoreError::ConfigError(format!(
                "worker ASR model {model_name} does not resolve back to the deployed variant"
            )));
        }
        let started = self
            .runtime
            .try_start_asr_realtime_stream(Some(model_name), language, None)
            .await?;
        let stream = started.ok_or_else(|| {
            CoreError::ConfigError(format!(
                "deployed ASR model {model_name} does not support realtime stream decode"
            ))
        })?;
        Ok(Box::new(RuntimeAsrStageStream {
            runtime: Arc::clone(&self.runtime),
            stream,
        }))
    }
}

/// Adapts the runtime's ASR realtime stream onto the worker's opaque stage
/// stream handle. Dropping the adapter drops the runtime stream, which
/// releases its job, session, and residency leases.
struct RuntimeAsrStageStream {
    runtime: Arc<RuntimeService>,
    stream: RuntimeAsrRealtimeStream,
}

#[async_trait]
impl RealtimeAsrStageStream for RuntimeAsrStageStream {
    async fn push_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>, CoreError> {
        self.runtime
            .push_asr_realtime_samples(&mut self.stream, samples, sample_rate)
            .await
    }

    async fn finish(&mut self) -> Result<Vec<RuntimeAsrRealtimeEvent>, CoreError> {
        self.runtime
            .finish_asr_realtime_stream(&mut self.stream)
            .await
    }
}

const TTS_WARMUP_TEXT: &str = "Warm up.";
const TTS_WARMUP_MAX_CHUNKS: usize = 8;

/// Execute one bounded streaming synthesis before a worker advertises
/// readiness: synthesize a short sentence through the same
/// [`RuntimeService::generate_streaming`] surface the realtime stage uses and
/// require at least the terminal chunk. This proves the deployed variant
/// resolves and that streaming synthesis actually executes; the caller must
/// not bind a listener on failure.
pub async fn warm_up_tts_runtime(
    runtime: &Arc<RuntimeService>,
    variant: ModelVariant,
    timeout: Duration,
) -> Result<(), CoreError> {
    if timeout.is_zero() {
        return Err(CoreError::ConfigError(
            "worker warm-up timeout must be non-zero".into(),
        ));
    }
    let runtime_context = RuntimeRequestContext::new(WorkloadClass::Realtime)
        .with_deadline(std::time::Instant::now() + timeout);
    let request = GenerationRequest::new(TTS_WARMUP_TEXT)
        .with_config(GenerationConfig {
            streaming: true,
            options: GenerationParams {
                max_tokens: 0,
                ..GenerationParams::default()
            },
        })
        .with_model_variant(variant)
        .with_runtime_context(runtime_context);
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<AudioChunk>(4);
    let generation_runtime = Arc::clone(runtime);
    let generation = tokio::spawn(async move {
        let _ = generation_runtime
            .generate_streaming(request, chunk_tx)
            .await;
    });
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    let mut chunks = 0usize;
    let mut saw_final = false;
    let mut saw_samples = false;
    let outcome = loop {
        tokio::select! {
            () = &mut deadline => break Err(CoreError::Timeout("worker startup warm-up".into())),
            chunk = chunk_rx.recv() => match chunk {
                Some(chunk) => {
                    chunks = chunks.saturating_add(1);
                    saw_final = saw_final || chunk.is_final;
                    saw_samples = saw_samples || !chunk.samples.is_empty();
                    if chunks > TTS_WARMUP_MAX_CHUNKS {
                        break Err(CoreError::InferenceError(
                            "worker startup warm-up exceeded its output bounds".into(),
                        ));
                    }
                }
                None => {
                    // A terminal marker alone proves nothing: the engine
                    // streaming path always emits one, so completion requires
                    // actual PCM as well.
                    break if saw_final && saw_samples {
                        Ok(chunks)
                    } else if !saw_final {
                        Err(CoreError::InferenceError(
                            "worker startup warm-up ended without completion".into(),
                        ))
                    } else {
                        Err(CoreError::InferenceError(
                            "worker startup warm-up completed without any synthesized audio".into(),
                        ))
                    }
                }
            },
        }
    };
    drop(chunk_rx);
    let _ = generation.await;
    outcome.map(|_| ())
}

/// Serve realtime TTS sessions for one already loaded TTS model variant.
///
/// HTTP invocations are rejected for this executor: text-to-speech workers
/// serve realtime WebSocket sessions only, and the deployment's capability
/// gates keep chat traffic away before this executor is ever consulted.
pub struct RuntimeRealtimeTtsExecutor {
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
    output_audio: RealtimeAudioSpec,
}

impl RuntimeRealtimeTtsExecutor {
    /// The output spec is announced at session admission, before any audio,
    /// so it carries the model's own output rate (Fish 44.1 kHz), not the
    /// codec default (TPIPE-22); a chunk at another rate is refused.
    pub async fn new(runtime: Arc<RuntimeService>, variant: ModelVariant) -> Self {
        let sample_rate = match variant.tts_output_sample_rate_hint() {
            Some(rate) => rate,
            None => runtime.sample_rate().await,
        };
        Self {
            runtime,
            variant,
            output_audio: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate,
                channels: 1,
            },
        }
    }
}

#[async_trait]
impl InvocationExecutor for RuntimeRealtimeTtsExecutor {
    async fn runtime_telemetry(&self) -> Option<RuntimeTelemetrySnapshot> {
        Some(self.runtime.telemetry_snapshot().await)
    }

    fn runtime_health(&self) -> RuntimeHealth {
        self.runtime.health()
    }

    async fn admit(
        &self,
        _request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        Err(AdmissionFailure::new(
            RejectionCode::IncompatibleTask,
            "text_to_speech workers serve realtime WebSocket sessions only; HTTP invocations are not accepted",
        ))
    }

    fn realtime_runner(&self) -> Option<Arc<dyn RealtimeStageRunner>> {
        Some(Arc::new(RuntimeRealtimeTtsRunner {
            runtime: Arc::clone(&self.runtime),
            variant: self.variant,
            output_audio: self.output_audio,
        }))
    }
}

/// Realtime TTS stage execution against the worker's runtime.
pub struct RuntimeRealtimeTtsRunner {
    runtime: Arc<RuntimeService>,
    variant: ModelVariant,
    output_audio: RealtimeAudioSpec,
}

#[async_trait]
impl RealtimeStageRunner for RuntimeRealtimeTtsRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::TextToSpeech
    }

    async fn start_asr_stream(
        &self,
        _language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, CoreError> {
        Err(CoreError::ConfigError(
            "text_to_speech workers do not serve realtime ASR sessions".into(),
        ))
    }

    fn output_audio_spec(&self) -> Option<RealtimeAudioSpec> {
        Some(self.output_audio)
    }

    async fn start_tts_stream(
        &self,
        text: String,
        options: &RealtimeTtsOptions,
        deadline: std::time::Instant,
    ) -> Result<Box<dyn RealtimeTtsStageStream>, CoreError> {
        let runtime_context =
            RuntimeRequestContext::new(WorkloadClass::Realtime).with_deadline(deadline);
        let mut request = GenerationRequest::new(text)
            .with_config(GenerationConfig {
                streaming: true,
                // Zero mirrors the single-node realtime voice route: audio
                // token generation is bounded by the model, not a text ceiling.
                options: GenerationParams {
                    max_tokens: 0,
                    speaker: options.speaker.clone(),
                    voice: options.speaker.clone(),
                    ..GenerationParams::default()
                },
            })
            .with_model_variant(self.variant)
            .with_runtime_context(runtime_context);
        request.language = options.language.clone();
        // A dropped receiver is the natural cancellation signal: the
        // generation task observes the closed channel and unwinds. The abort
        // handle only backstops a task wedged outside a send.
        let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<AudioChunk>(32);
        let generation_runtime = Arc::clone(&self.runtime);
        let generation = tokio::spawn(async move {
            generation_runtime
                .generate_streaming(request, chunk_tx)
                .await
        });
        Ok(Box::new(RuntimeTtsStageStream {
            chunks: chunk_rx,
            generation: Some(generation),
            sample_rate: self.output_audio.sample_rate,
        }))
    }
}

struct RuntimeTtsStageStream {
    chunks: tokio::sync::mpsc::Receiver<AudioChunk>,
    generation: Option<tokio::task::JoinHandle<Result<(), CoreError>>>,
    /// The rate the session announced.
    sample_rate: u32,
}

impl Drop for RuntimeTtsStageStream {
    fn drop(&mut self) {
        if let Some(generation) = self.generation.take() {
            generation.abort();
        }
    }
}

#[async_trait]
impl RealtimeTtsStageStream for RuntimeTtsStageStream {
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, CoreError> {
        match self.chunks.recv().await {
            Some(chunk) if chunk.sample_rate != 0 && chunk.sample_rate != self.sample_rate => {
                Err(CoreError::InferenceError(format!(
                    "TTS audio at {} Hz does not match the session's announced {} Hz",
                    chunk.sample_rate, self.sample_rate
                )))
            }
            Some(chunk) => Ok(Some(chunk)),
            None => match self.generation.take() {
                None => Ok(None),
                Some(generation) => match generation.await {
                    Ok(Ok(())) => Ok(None),
                    Ok(Err(error)) => Err(error),
                    Err(join_error) => Err(CoreError::InferenceError(format!(
                        "synthesis task ended abnormally: {join_error}"
                    ))),
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::ChatContextPolicy;

    const VARIANT: ModelVariant = ModelVariant::Lfm2512BInstructGguf;

    /// DP-F2: every minor-4 control reaches the engine request, through the
    /// same core resolver the local server uses.
    #[test]
    fn every_chat_control_reaches_the_engine_request() {
        let parameters = ChatParameters {
            temperature: Some(0.5),
            top_p: Some(0.75),
            seed: Some(7),
            stop: vec!["END".into()],
            top_k: Some(9),
            repetition_penalty: Some(1.25),
            presence_penalty: Some(-0.5),
            logprobs: Some(3),
            response_format: Some(ChatResponseFormat::JsonObject),
            context: Some(ChatContextPolicy {
                max_tokens: Some(16),
                truncation: WireChatTruncation::Disabled,
            }),
            tools: vec![serde_json::json!({"type": "function", "function": {"name": "now"}})],
            skip_special_tokens: Some(false),
        };
        let (params, chat_config, ceiling) = chat_request(VARIANT, &parameters, 64);
        assert_eq!(
            (params.temperature, params.top_p, params.top_k),
            (0.5, 0.75, 9)
        );
        assert_eq!(
            (params.repetition_penalty, params.presence_penalty),
            (1.25, -0.5)
        );
        assert_eq!(params.seed, Some(7));
        assert_eq!(params.stop_sequences, vec!["END".to_string()]);
        assert!(params.logprobs);
        assert_eq!(params.top_logprobs, 3);
        assert!(params.constrain_json_object);
        assert!(!params.skip_special_tokens);
        assert_eq!(params.max_tokens, 16);
        assert_eq!(chat_config.truncation, ChatTruncation::Disabled);
        assert_eq!(chat_config.tools, parameters.tools);
        assert_eq!(ceiling, Some(64));

        let expected = GenerationParams::for_chat(
            resolve_chat_sampling(
                VARIANT,
                None,
                ChatSamplingRequest {
                    temperature: Some(0.5),
                    top_p: Some(0.75),
                    top_k: Some(9),
                    repetition_penalty: Some(1.25),
                    presence_penalty: Some(-0.5),
                },
            ),
            16,
            ChatGenerationControls {
                seed: Some(7),
                stop: vec!["END".into()],
                logprobs: Some(3),
                constrain_json_object: true,
                keep_special_tokens: true,
            },
        );
        assert_eq!(format!("{params:?}"), format!("{expected:?}"));
    }

    /// An absent sampling value takes the model family's profile, as on the
    /// local server, not greedy decoding; an unset budget under the D8
    /// policy takes the rest of the context, capped at the ceiling.
    #[test]
    fn absent_controls_take_the_local_defaults() {
        let parameters = ChatParameters {
            context: Some(ChatContextPolicy {
                max_tokens: None,
                truncation: WireChatTruncation::Auto,
            }),
            ..ChatParameters::default()
        };
        let (params, chat_config, ceiling) = chat_request(VARIANT, &parameters, 64);
        let profile = resolve_chat_sampling(VARIANT, None, ChatSamplingRequest::default());
        assert_eq!(
            (
                params.temperature,
                params.top_p,
                params.top_k,
                params.repetition_penalty,
                params.presence_penalty
            ),
            (
                profile.temperature,
                profile.top_p,
                profile.top_k,
                profile.repetition_penalty,
                profile.presence_penalty
            )
        );
        assert_eq!((params.temperature, params.top_k), (0.1, 50));
        assert_eq!(params.seed, None);
        assert!(!params.logprobs && !params.constrain_json_object);
        assert!(params.skip_special_tokens);
        assert_eq!(params.max_tokens, CHAT_MAX_TOKENS_UNSET);
        assert_eq!(chat_config.truncation, ChatTruncation::Auto);
        assert_eq!(ceiling, Some(64));
    }

    /// A gateway before minor 4 sends no context policy: its ceiling is the
    /// budget and the request truncates to fit, as before.
    #[test]
    fn a_minor3_gateway_keeps_its_ceiling_budget_and_truncation() {
        let (params, chat_config, ceiling) = chat_request(VARIANT, &ChatParameters::default(), 64);
        assert_eq!(params.max_tokens, 64);
        assert_eq!(chat_config.truncation, ChatTruncation::Auto);
        assert_eq!(ceiling, None);
    }

    #[test]
    fn context_overflow_is_its_own_rejection_code() {
        let failure = map_admission_error(CoreError::ContextLengthExceeded("too long".into()));
        assert_eq!(failure.code, RejectionCode::ContextLengthExceeded);
        // The bare explanation, as the local role reports it.
        assert_eq!(failure.message, "too long");
    }
}

use super::{
    AdmissionFailure, AdmittedExecution, AdmittedInvocation, ExecutionEvent, ExecutionFailure,
    ExecutionTeardown, InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner,
    RealtimeTtsStageStream,
};
use async_trait::async_trait;
use izwi_core::{
    engine::{OutputFinishReason, WorkloadClass},
    AudioChunk, ChatMessage as CoreChatMessage, ChatRole as CoreChatRole, Error as CoreError,
    GenerationConfig, GenerationParams, GenerationRequest, ModelVariant, RuntimeAsrRealtimeEvent,
    RuntimeAsrRealtimeStream, RuntimeChatInvocation, RuntimeChatInvocationEvent,
    RuntimeChatInvocationRequest, RuntimeChatTeardownDisposition, RuntimeRequestContext,
    RuntimeService, RuntimeTelemetrySnapshot,
};
use izwi_serving_protocol::{
    ChatRole, FinishReason, InvocationErrorCode, InvocationInput, InvocationRequest,
    RealtimeAudioCodec, RealtimeAudioSpec, RejectionCode, ServiceClass, TaskKind,
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
                    Ok(Some(RuntimeChatInvocationEvent::TextDelta(text))) => {
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

    async fn admit(
        &self,
        request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        let InvocationInput::Chat { input, parameters } = &request.input;
        if parameters.seed.is_some_and(|seed| seed != 0) {
            return Err(AdmissionFailure::new(
                RejectionCode::InvalidRequest,
                "the selected runtime does not expose per-request sampling seeds",
            ));
        }
        let messages = input
            .messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    ChatRole::System => CoreChatRole::System,
                    ChatRole::User => CoreChatRole::User,
                    ChatRole::Assistant => CoreChatRole::Assistant,
                    ChatRole::Tool => {
                        return Err(AdmissionFailure::new(
                            RejectionCode::InvalidRequest,
                            "tool-role messages are not supported by the selected CPU model",
                        ))
                    }
                };
                Ok(CoreChatMessage {
                    role,
                    content: message.content.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut params = GenerationParams {
            max_tokens: request.output_limits.max_tokens as usize,
            temperature: parameters.temperature.unwrap_or(0.0),
            top_p: parameters.top_p.unwrap_or(1.0),
            stop_sequences: parameters.stop.clone(),
            ..GenerationParams::default()
        };
        // Chat defaults are deterministic and do not need audio-generation penalties.
        params.repetition_penalty = 1.0;
        let workload_class = match request.service_class {
            ServiceClass::Realtime => WorkloadClass::Realtime,
            ServiceClass::Interactive if self.streaming => WorkloadClass::Streaming,
            ServiceClass::Interactive => WorkloadClass::Interactive,
            ServiceClass::Batch => WorkloadClass::Batch,
        };
        let runtime_context = RuntimeRequestContext::new(workload_class).with_deadline(
            std::time::Instant::now() + Duration::from_millis(request.remaining_time_ms),
        );
        let invocation = self
            .runtime
            .start_chat_invocation(RuntimeChatInvocationRequest {
                variant: self.variant,
                messages,
                params,
                chat_config: Default::default(),
                correlation_id: Some(request.request_id.as_str().to_owned()),
                runtime_context,
                streaming: self.streaming,
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
            Ok(Some(RuntimeChatInvocationEvent::TextDelta(text))) => {
                Some(ExecutionEvent::TextDelta(text))
            }
            Ok(Some(RuntimeChatInvocationEvent::Completed(generation))) => {
                let completed = ExecutionEvent::Completed {
                    text: None,
                    finish_reason: map_finish_reason(generation.finish_reason),
                    input_tokens: generation.prompt_tokens as u64,
                    output_tokens: generation.tokens_generated as u64,
                };
                if self.streaming || generation.text.is_empty() {
                    Some(completed)
                } else {
                    self.pending.push_back(completed);
                    Some(ExecutionEvent::TextDelta(generation.text))
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

fn map_admission_error(error: CoreError) -> AdmissionFailure {
    let code = match error {
        CoreError::Overloaded(_) => RejectionCode::CapacityExhausted,
        CoreError::Timeout(_) => RejectionCode::QueueWaitExceeded,
        CoreError::ModelNotFound(_) | CoreError::ModelLoadError(_) => RejectionCode::ModelNotReady,
        CoreError::Cancelled(_) => RejectionCode::WorkerDraining,
        CoreError::InvalidInput(_) | CoreError::ConfigError(_) => RejectionCode::InvalidRequest,
        _ => RejectionCode::ModelNotReady,
    };
    AdmissionFailure::new(code, error.to_string())
}

pub(crate) fn map_execution_error(error: CoreError) -> ExecutionFailure {
    let code = match error {
        CoreError::Timeout(_) => InvocationErrorCode::DeadlineExceeded,
        CoreError::InvalidInput(_) => InvocationErrorCode::InvalidInput,
        CoreError::Overloaded(_) | CoreError::Backpressure(_) => {
            InvocationErrorCode::WorkerUnavailable
        }
        _ => InvocationErrorCode::ExecutionFailed,
    };
    ExecutionFailure {
        code,
        message: error.to_string(),
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
    let outcome = loop {
        tokio::select! {
            () = &mut deadline => break Err(CoreError::Timeout("worker startup warm-up".into())),
            chunk = chunk_rx.recv() => match chunk {
                Some(chunk) => {
                    chunks = chunks.saturating_add(1);
                    saw_final = saw_final || chunk.is_final;
                    if chunks > TTS_WARMUP_MAX_CHUNKS {
                        break Err(CoreError::InferenceError(
                            "worker startup warm-up exceeded its output bounds".into(),
                        ));
                    }
                }
                None => {
                    break if saw_final {
                        Ok(chunks)
                    } else {
                        Err(CoreError::InferenceError(
                            "worker startup warm-up ended without completion".into(),
                        ))
                    }
                }
            },
        }
    };
    drop(chunk_rx);
    let _ = generation.await;
    let chunks = outcome?;
    if chunks == 0 {
        return Err(CoreError::InferenceError(
            "worker startup warm-up ended without any synthesized audio".into(),
        ));
    }
    Ok(())
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
    /// The output spec is resolved from the loaded runtime's codec rate, the
    /// same surface the single-node realtime voice route announces.
    pub async fn new(runtime: Arc<RuntimeService>, variant: ModelVariant) -> Self {
        let sample_rate = runtime.sample_rate().await;
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
        deadline: std::time::Instant,
    ) -> Result<Box<dyn RealtimeTtsStageStream>, CoreError> {
        let runtime_context =
            RuntimeRequestContext::new(WorkloadClass::Realtime).with_deadline(deadline);
        let request = GenerationRequest::new(text)
            .with_config(GenerationConfig {
                streaming: true,
                // Zero mirrors the single-node realtime voice route: audio
                // token generation is bounded by the model, not a text ceiling.
                options: GenerationParams {
                    max_tokens: 0,
                    ..GenerationParams::default()
                },
            })
            .with_model_variant(self.variant)
            .with_runtime_context(runtime_context);
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
        }))
    }
}

struct RuntimeTtsStageStream {
    chunks: tokio::sync::mpsc::Receiver<AudioChunk>,
    generation: Option<tokio::task::JoinHandle<Result<(), CoreError>>>,
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

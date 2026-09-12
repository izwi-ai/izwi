use super::{
    AdmissionFailure, AdmittedExecution, AdmittedInvocation, ExecutionEvent, ExecutionFailure,
    ExecutionTeardown, InvocationExecutor,
};
use async_trait::async_trait;
use izwi_core::{
    engine::{OutputFinishReason, WorkloadClass},
    ChatMessage as CoreChatMessage, ChatRole as CoreChatRole, Error as CoreError, GenerationParams,
    ModelVariant, RuntimeChatInvocation, RuntimeChatInvocationEvent, RuntimeChatInvocationRequest,
    RuntimeChatTeardownDisposition, RuntimeRequestContext, RuntimeService,
};
use izwi_serving_protocol::{
    ChatRole, FinishReason, InvocationErrorCode, InvocationInput, InvocationRequest, RejectionCode,
    ServiceClass,
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

fn map_execution_error(error: CoreError) -> ExecutionFailure {
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

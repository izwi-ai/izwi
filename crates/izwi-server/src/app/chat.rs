use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use izwi_serving_client::{InvocationStream, WorkerClient, WorkerClientError};
use izwi_serving_protocol::{
    AttemptId, CallerId, ChatInput, ChatMessage as WorkerChatMessage,
    ChatParameters as WorkerChatParameters, ChatRole as WorkerChatRole, DeploymentId,
    FinishReason as WorkerFinishReason, GatewayAttestedCallerContext, IncarnationId,
    InvocationErrorCode, InvocationEventKind, InvocationInput, InvocationRequest, ModelGeneration,
    ModelReadiness, OutputFormat, OutputLimits, PermittedAction, PolicyRevision, RejectionCode,
    RequestDigest, RequestId, ServiceClass, TaskKind, TenantId, Usage, WorkerProcessState,
    PROTOCOL_V1,
};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::Notify;
use tokio::sync::{mpsc, oneshot};

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::ids::new_uuid;
use crate::state::AppState;
use izwi_core::{
    parse_chat_model_variant, ChatGeneration, ChatMediaInput, ChatMessage, ChatReasoningEffort,
    ChatRequestConfig, ChatTemplateKwargs, GenerationParams, ModelVariant, WorkloadClass,
};

#[derive(Debug, Clone)]
pub struct ChatExecutionRequest {
    pub variant: ModelVariant,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub chat_config: ChatRequestConfig,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        max_new_tokens(self.variant, self.max_completion_tokens, self.max_tokens)
    }

    fn resolved_generation_params(&self) -> GenerationParams {
        let max_new_tokens = self.resolved_max_new_tokens();
        let mut params = if self.variant == ModelVariant::Qwen3827BFp8 {
            let thinking = self.chat_config.enable_thinking.unwrap_or(true);
            GenerationParams {
                temperature: if thinking { 1.0 } else { 0.7 },
                top_p: if thinking { 0.95 } else { 0.80 },
                top_k: 20,
                repetition_penalty: 1.0,
                presence_penalty: if thinking { 0.0 } else { 1.5 },
                max_tokens: max_new_tokens,
                ..Default::default()
            }
        } else {
            GenerationParams {
                max_tokens: max_new_tokens,
                ..Default::default()
            }
        };

        if let Some(temperature) = self.temperature {
            params.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            params.top_p = top_p;
        }
        if let Some(top_k) = self.top_k {
            params.top_k = top_k;
        }
        if let Some(repetition_penalty) = self.repetition_penalty {
            params.repetition_penalty = repetition_penalty;
        }
        if let Some(presence_penalty) = self.presence_penalty {
            params.presence_penalty = presence_penalty;
        }
        params
    }

    fn resolved_chat_config(&self) -> ChatRequestConfig {
        self.chat_config.clone()
    }
}

/// Pinned Phase 1 route target for plain non-streaming chat. Discovery and
/// multi-worker selection are deliberately deferred; the worker still makes
/// authoritative admission and generation checks on every invocation.
#[derive(Debug, Clone)]
pub struct RemoteChatExecutionConfig {
    pub public_model_variant: ModelVariant,
    pub expected_worker_incarnation: IncarnationId,
    pub deployment_id: DeploymentId,
    pub expected_model_generation: ModelGeneration,
    pub policy_revision: PolicyRevision,
    pub max_queue_wait: Duration,
    pub max_output_tokens: u32,
    pub max_output_bytes: u64,
}

impl RemoteChatExecutionConfig {
    fn validate(&self) -> Result<(), ApiError> {
        if self.max_output_tokens == 0 || self.max_output_bytes == 0 {
            return Err(ApiError::internal(
                "Remote chat output limits must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RemoteChatExecution {
    client: WorkerClient,
    config: RemoteChatExecutionConfig,
}

impl RemoteChatExecution {
    pub fn new(client: WorkerClient, config: RemoteChatExecutionConfig) -> Result<Self, ApiError> {
        config.validate()?;
        Ok(Self { client, config })
    }

    #[cfg(test)]
    pub fn config(&self) -> &RemoteChatExecutionConfig {
        &self.config
    }

    pub(crate) const fn max_output_tokens(&self) -> u32 {
        self.config.max_output_tokens
    }

    /// Verify that the pinned worker incarnation still advertises the exact
    /// chat deployment and model generation configured by the gateway.
    pub async fn readiness_check(&self) -> Result<(), String> {
        let status = self
            .client
            .status()
            .await
            .map_err(|error| format!("worker status request failed: {error}"))?;
        if status.incarnation_id != self.config.expected_worker_incarnation {
            return Err("worker incarnation does not match the pinned gateway target".into());
        }
        if status.process_state != WorkerProcessState::Running {
            return Err(format!(
                "worker process is {:?}, expected running",
                status.process_state
            ));
        }
        let deployment = status
            .deployments
            .iter()
            .find(|deployment| deployment.deployment_id == self.config.deployment_id)
            .ok_or_else(|| "pinned chat deployment is not advertised by the worker".to_string())?;
        if deployment.model_generation != self.config.expected_model_generation {
            return Err("worker model generation does not match the pinned gateway target".into());
        }
        if deployment.public_model.as_str() != self.config.public_model_variant.dir_name() {
            return Err("worker public model does not match the gateway route model".into());
        }
        if deployment.task != TaskKind::Chat || deployment.readiness != ModelReadiness::Ready {
            return Err("pinned worker deployment is not ready for chat".into());
        }
        Ok(())
    }
}

fn resolve_compatible_field<T>(
    name: &str,
    direct: Option<T>,
    template: Option<T>,
) -> Result<Option<T>, ApiError>
where
    T: Copy + PartialEq,
{
    match (direct, template) {
        (Some(direct), Some(template)) if direct != template => Err(ApiError::bad_request(
            format!("Conflicting `{name}` values in request and `chat_template_kwargs`"),
        )),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub fn resolve_chat_request_config(
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ChatReasoningEffort>,
    preserve_thinking: Option<bool>,
    chat_template_kwargs: Option<&ChatTemplateKwargs>,
    tools: Vec<serde_json::Value>,
    media_inputs: Vec<ChatMediaInput>,
) -> Result<ChatRequestConfig, ApiError> {
    let template = chat_template_kwargs.cloned().unwrap_or_default();
    Ok(ChatRequestConfig {
        enable_thinking: resolve_compatible_field(
            "enable_thinking",
            enable_thinking,
            template.enable_thinking,
        )?,
        reasoning_effort: resolve_compatible_field(
            "reasoning_effort",
            reasoning_effort,
            template.reasoning_effort,
        )?,
        preserve_thinking: resolve_compatible_field(
            "preserve_thinking",
            preserve_thinking,
            template.preserve_thinking,
        )?,
        tools,
        media_inputs,
    })
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started,
    Delta(String),
    Completed(Box<ChatGeneration>),
    Failed(String),
    ShuttingDown,
}

/// Completion acknowledgement for a stream whose producer owns an external
/// lifecycle guard. Dropping this value also acknowledges cancellation, while
/// an explicit acknowledgement records that terminal handling (including any
/// persistence) completed successfully in the response consumer.
pub struct ChatStreamCompletion {
    sender: Option<oneshot::Sender<()>>,
}

impl ChatStreamCompletion {
    pub fn acknowledge(mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(());
        }
    }
}

const CHAT_STREAM_CAPACITY: usize = 64;
const CHAT_STREAM_BACKPRESSURE_ERROR: &str =
    "Chat stream consumer is too slow; generation was cancelled";
#[cfg(not(test))]
const CHAT_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const CHAT_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Default)]
struct ChatStreamBackpressure {
    full: AtomicBool,
    notify: Notify,
}

impl ChatStreamBackpressure {
    fn trip(&self) {
        self.full.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    fn is_tripped(&self) -> bool {
        self.full.load(Ordering::Acquire)
    }

    async fn notified(&self) {
        if self.is_tripped() {
            return;
        }
        self.notify.notified().await;
    }
}

fn try_send_chat_delta(
    event_tx: &mpsc::Sender<ChatStreamEvent>,
    backpressure: &ChatStreamBackpressure,
    delta: String,
) {
    match event_tx.try_send(ChatStreamEvent::Delta(delta)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => backpressure.trip(),
        // The receiver has gone away, so there is nobody to notify with a
        // terminal event. `event_tx.closed()` cancels the generation task.
        Err(TrySendError::Closed(_)) => {}
    }
}

async fn resolve_chat_terminal<F>(
    event_tx: &mpsc::Sender<ChatStreamEvent>,
    backpressure: Arc<ChatStreamBackpressure>,
    generation: F,
) -> Option<ChatStreamEvent>
where
    F: Future<Output = izwi_core::Result<ChatGeneration>>,
{
    tokio::pin!(generation);
    let result = tokio::select! {
        result = &mut generation => Some(result),
        _ = event_tx.closed() => return None,
        _ = backpressure.notified() => None,
    };

    // A generation can produce the overflowing delta and complete in the same
    // poll. Check the latch even when the generation branch wins the select so
    // a dropped delta can never be reported as a successful completion.
    if result.is_none() || backpressure.is_tripped() {
        return Some(ChatStreamEvent::Failed(
            CHAT_STREAM_BACKPRESSURE_ERROR.to_string(),
        ));
    }

    match result.expect("completed generation result must be present") {
        Ok(generation) => Some(ChatStreamEvent::Completed(Box::new(generation))),
        Err(err) => Some(ChatStreamEvent::Failed(err.to_string())),
    }
}

async fn send_chat_terminal(event_tx: mpsc::Sender<ChatStreamEvent>, event: ChatStreamEvent) {
    // A connected receiver may stop polling forever. Terminal delivery remains
    // best-effort for that transport, but it must never retain inference or
    // workload capacity indefinitely.
    let _ = tokio::time::timeout(CHAT_TERMINAL_SEND_TIMEOUT, event_tx.send(event)).await;
}

pub fn max_new_tokens(
    _variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    // Preserve an omitted OpenAI output limit until the exact prompt has been
    // tokenized. The engine first bounds this sentinel by its backend context
    // capacity, then `enforce_chat_context_window` reduces it to the loaded
    // model's exact remaining context. This matches production serving engines
    // such as vLLM and avoids imposing an unrelated API-layer token ceiling.
    requested.unwrap_or(usize::MAX).max(1)
}

pub fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

pub async fn generate_chat(
    state: &AppState,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;
    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;

    state
        .runtime
        .chat_generate_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id.as_deref(),
            permit.runtime_context(),
        )
        .await
        .map_err(ApiError::from)
}

/// Execute the explicitly configured Phase 1 chat slice over the private HTTP
/// worker contract. This function never falls back to local execution and never
/// retries a request whose acceptance may be uncertain.
pub async fn generate_remote_chat(
    state: &AppState,
    context: &RequestContext,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let remote = state
        .remote_chat_execution
        .as_ref()
        .ok_or_else(|| ApiError::service_unavailable("Remote chat execution is not configured"))?;
    generate_remote_chat_with_execution(remote, state.request_timeout_secs, context, request).await
}

pub async fn generate_remote_chat_with_execution(
    remote: &RemoteChatExecution,
    request_timeout_secs: u64,
    context: &RequestContext,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    generate_remote_chat_with_execution_and_acceptance(
        remote,
        request_timeout_secs,
        context,
        request,
        (),
        |_| Ok(()),
    )
    .await
}

pub(crate) async fn generate_remote_chat_with_execution_and_acceptance<K, F>(
    remote: &RemoteChatExecution,
    request_timeout_secs: u64,
    context: &RequestContext,
    request: ChatExecutionRequest,
    mut keepalive: K,
    on_accepted: F,
) -> Result<ChatGeneration, ApiError>
where
    F: FnOnce(&mut K) -> Result<(), ApiError>,
{
    let invocation =
        prepare_remote_chat_invocation(remote, request_timeout_secs, context, request)?;
    let started = Instant::now();
    let mut stream = remote
        .client
        .invoke(invocation)
        .await
        .map_err(map_worker_client_error)?;
    // WorkerClient::invoke returns only after validating the first accepted
    // event. This is the earliest safe point for dynamic dispatch accounting
    // to transition out of its pre-admission state.
    on_accepted(&mut keepalive)?;
    collect_started_remote_chat(remote, &mut stream, started, |_| {}).await
}

pub(crate) async fn collect_started_remote_chat<F>(
    remote: &RemoteChatExecution,
    stream: &mut InvocationStream,
    started: Instant,
    mut on_worker_error: F,
) -> Result<ChatGeneration, ApiError>
where
    F: FnMut(&WorkerClientError),
{
    let mut text = String::new();
    let mut latest_usage = None;

    loop {
        let event = match stream.next_event().await {
            Ok(Some(event)) => event,
            Ok(None) => break,
            Err(error) => {
                on_worker_error(&error);
                return Err(map_worker_client_error(error));
            }
        };
        match event.event {
            InvocationEventKind::Accepted { .. } => {}
            InvocationEventKind::TextDelta { text: delta } => {
                append_remote_text(&mut text, &delta, remote.config.max_output_bytes)?;
            }
            InvocationEventKind::Usage { usage } => latest_usage = Some(usage),
            InvocationEventKind::Completed {
                finish_reason,
                usage,
            } => {
                return worker_chat_generation(
                    text,
                    finish_reason,
                    usage.or(latest_usage),
                    started,
                );
            }
            InvocationEventKind::Error { code, message } => {
                return Err(map_worker_terminal_error(code, message));
            }
            InvocationEventKind::Cancelled { reason } => {
                return Err(cancelled_error(
                    reason.unwrap_or_else(|| "Worker cancelled chat invocation".to_string()),
                ));
            }
        }
    }

    Err(bad_gateway_error(
        "Worker chat stream ended without a terminal event",
    ))
}

pub(crate) fn prepare_remote_chat_invocation(
    remote: &RemoteChatExecution,
    request_timeout_secs: u64,
    context: &RequestContext,
    request: ChatExecutionRequest,
) -> Result<InvocationRequest, ApiError> {
    if request.variant != remote.config.public_model_variant {
        return Err(ApiError::bad_request(format!(
            "Requested model is incompatible with remote deployment {}",
            remote.config.deployment_id
        )));
    }
    validate_remote_chat_scope(&request)?;

    let remaining = context
        .remaining_budget(Duration::from_secs(request_timeout_secs.max(1)))
        .filter(|budget| !budget.is_zero())
        .ok_or_else(|| request_timeout_error("Chat request deadline expired before dispatch"))?;
    let remaining_time_ms = u64::try_from(remaining.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    build_remote_chat_invocation(context, request, remote, remaining_time_ms)
}

pub(crate) fn retarget_remote_chat_invocation(
    invocation: &InvocationRequest,
    remote: &RemoteChatExecution,
    remaining: Duration,
) -> Result<InvocationRequest, ApiError> {
    if remaining.is_zero() {
        return Err(request_timeout_error(
            "Chat request deadline expired before alternate dispatch",
        ));
    }
    let remaining_time_ms = u64::try_from(remaining.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let mut alternate = invocation.clone();
    alternate.attempt_id = AttemptId::new(new_uuid()).map_err(identifier_error)?;
    alternate.expected_worker_incarnation = remote.config.expected_worker_incarnation.clone();
    alternate.deployment_id = remote.config.deployment_id.clone();
    alternate.expected_model_generation = remote.config.expected_model_generation;
    alternate.remaining_time_ms = remaining_time_ms;
    alternate.max_queue_wait_ms = u64::try_from(remote.config.max_queue_wait.as_millis())
        .unwrap_or(u64::MAX)
        .min(remaining_time_ms);
    alternate.output_limits.max_tokens = alternate
        .output_limits
        .max_tokens
        .min(remote.config.max_output_tokens)
        .max(1);
    alternate.output_limits.max_bytes = remote.config.max_output_bytes;
    alternate.request_digest = request_digest(
        &alternate.caller.tenant_id,
        &alternate.deployment_id,
        alternate.expected_model_generation,
        &alternate.input,
        alternate.output_limits.max_tokens,
        alternate.output_limits.max_bytes,
    )?;
    alternate
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    Ok(alternate)
}

pub(crate) async fn start_remote_chat_invocation(
    remote: &RemoteChatExecution,
    invocation: InvocationRequest,
) -> Result<InvocationStream, WorkerClientError> {
    remote.client.invoke(invocation).await
}

fn append_remote_text(text: &mut String, delta: &str, max_bytes: u64) -> Result<(), ApiError> {
    let next_len = text.len().checked_add(delta.len()).ok_or_else(|| {
        bad_gateway_error("Worker chat output exceeded the configured byte limit")
    })?;
    if u64::try_from(next_len).unwrap_or(u64::MAX) > max_bytes {
        return Err(bad_gateway_error(
            "Worker chat output exceeded the configured byte limit",
        ));
    }
    text.push_str(delta);
    Ok(())
}

fn worker_chat_generation(
    text: String,
    finish_reason: WorkerFinishReason,
    usage: Option<Usage>,
    started: Instant,
) -> Result<ChatGeneration, ApiError> {
    let usage = usage.unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
    });
    Ok(ChatGeneration {
        latency_breakdown: None,
        finish_reason: Some(match finish_reason {
            WorkerFinishReason::Stop => izwi_core::engine::OutputFinishReason::StopToken,
            WorkerFinishReason::Length => izwi_core::engine::OutputFinishReason::MaxTokens,
        }),
        text,
        prompt_tokens: usize::try_from(usage.input_tokens)
            .map_err(|_| bad_gateway_error("Worker reported an invalid input token count"))?,
        tokens_generated: usize::try_from(usage.output_tokens)
            .map_err(|_| bad_gateway_error("Worker reported an invalid output token count"))?,
        generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Start an accepted private worker invocation and forward its bounded events
/// through the existing public chat stream channel. Dropping the receiver
/// closes the channel, drops the private stream, and schedules exact-attempt
/// cancellation; neither layer retries the invocation.
pub async fn spawn_remote_chat_stream_with_execution(
    remote: &RemoteChatExecution,
    request_timeout_secs: u64,
    context: &RequestContext,
    request: ChatExecutionRequest,
) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError> {
    let invocation =
        prepare_remote_chat_invocation(remote, request_timeout_secs, context, request)?;
    let worker_stream = remote
        .client
        .invoke(invocation)
        .await
        .map_err(map_worker_client_error)?;
    Ok(spawn_started_remote_chat_stream_with_execution(
        remote,
        worker_stream,
        |_| {},
    ))
}

pub(crate) fn spawn_started_remote_chat_stream_with_execution<F>(
    remote: &RemoteChatExecution,
    mut worker_stream: InvocationStream,
    on_worker_error: F,
) -> mpsc::Receiver<ChatStreamEvent>
where
    F: Fn(&WorkerClientError) + Send + Sync + 'static,
{
    let max_output_bytes = remote.config.max_output_bytes;
    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    tokio::spawn(async move {
        let started = Instant::now();
        let backpressure = Arc::new(ChatStreamBackpressure::default());
        let mut text = String::new();
        let mut latest_usage = None;
        let terminal = loop {
            let event = tokio::select! {
                event = worker_stream.next_event() => event,
                () = event_tx.closed() => return,
            };
            let event = match event {
                Ok(Some(event)) => event,
                Ok(None) => {
                    break ChatStreamEvent::Failed(
                        "Worker chat stream ended without a terminal event".into(),
                    )
                }
                Err(error) => {
                    on_worker_error(&error);
                    break ChatStreamEvent::Failed(map_worker_client_error(error).message);
                }
            };
            match event.event {
                InvocationEventKind::Accepted { .. } => {
                    if event_tx.try_send(ChatStreamEvent::Started).is_err() {
                        return;
                    }
                }
                InvocationEventKind::TextDelta { text: delta } => {
                    if let Err(error) = append_remote_text(&mut text, &delta, max_output_bytes) {
                        break ChatStreamEvent::Failed(error.message);
                    }
                    try_send_chat_delta(&event_tx, &backpressure, delta);
                    if event_tx.is_closed() {
                        return;
                    }
                    if backpressure.is_tripped() {
                        break ChatStreamEvent::Failed(CHAT_STREAM_BACKPRESSURE_ERROR.into());
                    }
                }
                InvocationEventKind::Usage { usage } => latest_usage = Some(usage),
                InvocationEventKind::Completed {
                    finish_reason,
                    usage,
                } => match worker_chat_generation(
                    text,
                    finish_reason,
                    usage.or(latest_usage),
                    started,
                ) {
                    Ok(generation) => break ChatStreamEvent::Completed(Box::new(generation)),
                    Err(error) => break ChatStreamEvent::Failed(error.message),
                },
                InvocationEventKind::Error { code, message } => {
                    break ChatStreamEvent::Failed(map_worker_terminal_error(code, message).message)
                }
                InvocationEventKind::Cancelled { reason } => {
                    break ChatStreamEvent::Failed(
                        reason.unwrap_or_else(|| "Worker cancelled chat invocation".into()),
                    )
                }
            }
        };
        // Release or cancel the private stream before terminal delivery can
        // wait on a slow public consumer. Worker capacity remains governed by
        // confirmed teardown, not by this gateway channel.
        drop(worker_stream);
        send_chat_terminal(event_tx, terminal).await;
    });
    event_rx
}

fn validate_remote_chat_scope(request: &ChatExecutionRequest) -> Result<(), ApiError> {
    if !request.chat_config.media_inputs.is_empty() {
        return Err(ApiError::bad_request(
            "Remote chat execution currently supports text-only requests",
        ));
    }
    if !request.chat_config.tools.is_empty()
        || request.chat_config.enable_thinking.is_some()
        || request.chat_config.reasoning_effort.is_some()
        || request.chat_config.preserve_thinking.is_some()
        || request.top_k.is_some()
        || request.repetition_penalty.is_some()
        || request.presence_penalty.is_some()
    {
        return Err(ApiError::bad_request(
            "Remote chat execution does not yet support tools or advanced generation controls",
        ));
    }
    Ok(())
}

fn build_remote_chat_invocation(
    context: &RequestContext,
    request: ChatExecutionRequest,
    remote: &RemoteChatExecution,
    remaining_time_ms: u64,
) -> Result<InvocationRequest, ApiError> {
    let requested_tokens = request.resolved_max_new_tokens();
    let max_tokens = u32::try_from(requested_tokens)
        .unwrap_or(u32::MAX)
        .min(remote.config.max_output_tokens)
        .max(1);
    let input = InvocationInput::Chat {
        input: ChatInput {
            messages: request
                .messages
                .into_iter()
                .map(|message| WorkerChatMessage {
                    role: match message.role {
                        izwi_core::ChatRole::System => WorkerChatRole::System,
                        izwi_core::ChatRole::User => WorkerChatRole::User,
                        izwi_core::ChatRole::Assistant => WorkerChatRole::Assistant,
                    },
                    content: message.content,
                })
                .collect(),
        },
        parameters: WorkerChatParameters {
            temperature: request.temperature,
            top_p: request.top_p,
            seed: None,
            stop: Vec::new(),
        },
    };
    let request_id = RequestId::new(new_uuid()).map_err(identifier_error)?;
    let attempt_id = AttemptId::new(new_uuid()).map_err(identifier_error)?;
    let tenant_namespace = context
        .principal
        .tenant_id
        .as_deref()
        .map(|tenant| format!("tenant:{tenant}"))
        .unwrap_or_else(|| format!("principal:{}", context.principal.id));
    let caller = GatewayAttestedCallerContext {
        tenant_id: TenantId::new(hashed_identity("tenant", &tenant_namespace))
            .map_err(identifier_error)?,
        caller_id: CallerId::new(hashed_identity("caller", &context.principal.id))
            .map_err(identifier_error)?,
        policy_revision: remote.config.policy_revision.clone(),
        permitted_actions: [
            PermittedAction::Invoke,
            PermittedAction::CancelOwnInvocation,
        ]
        .into_iter()
        .collect(),
        allowed_data_regions: Vec::new(),
    };
    let request_digest = request_digest(
        &caller.tenant_id,
        &remote.config.deployment_id,
        remote.config.expected_model_generation,
        &input,
        max_tokens,
        remote.config.max_output_bytes,
    )?;

    let invocation = InvocationRequest {
        schema_version: PROTOCOL_V1,
        request_id,
        attempt_id,
        expected_worker_incarnation: remote.config.expected_worker_incarnation.clone(),
        deployment_id: remote.config.deployment_id.clone(),
        expected_model_generation: remote.config.expected_model_generation,
        caller,
        task: TaskKind::Chat,
        service_class: ServiceClass::Interactive,
        remaining_time_ms,
        max_queue_wait_ms: u64::try_from(remote.config.max_queue_wait.as_millis())
            .unwrap_or(u64::MAX)
            .min(remaining_time_ms),
        output_limits: OutputLimits {
            max_tokens,
            max_bytes: remote.config.max_output_bytes,
        },
        requested_output_format: OutputFormat::Text,
        session_id: None,
        request_digest,
        input,
    };
    invocation
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    Ok(invocation)
}

fn request_digest(
    tenant_id: &TenantId,
    deployment_id: &DeploymentId,
    generation: ModelGeneration,
    input: &InvocationInput,
    max_tokens: u32,
    max_bytes: u64,
) -> Result<RequestDigest, ApiError> {
    let canonical = serde_json::to_vec(&(
        tenant_id,
        deployment_id,
        generation,
        input,
        max_tokens,
        max_bytes,
    ))
    .map_err(|error| ApiError::internal(error.to_string()))?;
    RequestDigest::new(hashed_identity("sha256", &canonical)).map_err(identifier_error)
}

fn hashed_identity(prefix: &str, value: impl AsRef<[u8]>) -> String {
    let digest = Sha256::digest(value.as_ref());
    let mut encoded = String::with_capacity(prefix.len() + 1 + digest.len() * 2);
    encoded.push_str(prefix);
    encoded.push(':');
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

fn identifier_error(error: izwi_serving_protocol::IdentifierError) -> ApiError {
    ApiError::internal(format!(
        "Failed to construct private serving identity: {error}"
    ))
}

pub(crate) fn map_worker_client_error(error: WorkerClientError) -> ApiError {
    match error {
        WorkerClientError::Rejected { rejection } => {
            let message = rejection.message;
            match rejection.code {
                RejectionCode::Unauthorized | RejectionCode::PolicyDenied => {
                    ApiError::forbidden(message)
                }
                RejectionCode::UnknownDeployment | RejectionCode::IncompatibleTask => {
                    ApiError::bad_request(message)
                }
                RejectionCode::WrongWorkerIncarnation
                | RejectionCode::WrongModelGeneration
                | RejectionCode::ModelNotReady
                | RejectionCode::WorkerDraining
                | RejectionCode::CapacityExhausted
                | RejectionCode::QueueWaitExceeded => ApiError::service_unavailable(message),
                RejectionCode::InvalidRequest => ApiError::bad_request(message),
                RejectionCode::DuplicateAttemptConflict => ApiError {
                    status: StatusCode::CONFLICT,
                    message,
                },
                RejectionCode::Unauthenticated | RejectionCode::UnsupportedProtocolVersion => {
                    bad_gateway_error(message)
                }
            }
        }
        WorkerClientError::Deadline(_) => request_timeout_error(error.to_string()),
        WorkerClientError::InvalidInvocation(validation) => {
            ApiError::bad_request(validation.to_string())
        }
        WorkerClientError::InvalidConfiguration(_)
        | WorkerClientError::InvalidEndpoint(_)
        | WorkerClientError::Build(_)
        | WorkerClientError::Encode(_)
        | WorkerClientError::RequestTooLarge { .. }
        | WorkerClientError::ConnectionNotEstablished(_)
        | WorkerClientError::Transport(_)
        | WorkerClientError::HttpStatus { .. }
        | WorkerClientError::ResponseTooLarge { .. }
        | WorkerClientError::InvalidJson(_)
        | WorkerClientError::Ndjson(_)
        | WorkerClientError::Protocol(_)
        | WorkerClientError::InterruptedUnknown => bad_gateway_error(error.to_string()),
    }
}

fn map_worker_terminal_error(code: InvocationErrorCode, message: String) -> ApiError {
    match code {
        InvocationErrorCode::InvalidInput => ApiError::bad_request(message),
        InvocationErrorCode::DeadlineExceeded => request_timeout_error(message),
        InvocationErrorCode::WorkerUnavailable => ApiError::service_unavailable(message),
        InvocationErrorCode::ExecutionFailed
        | InvocationErrorCode::OutputLimitExceeded
        | InvocationErrorCode::Internal => ApiError::internal(message),
    }
}

fn request_timeout_error(message: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::REQUEST_TIMEOUT,
        message: message.into(),
    }
}

fn cancelled_error(message: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::from_u16(499).unwrap_or(StatusCode::REQUEST_TIMEOUT),
        message: message.into(),
    }
}

fn bad_gateway_error(message: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::BAD_GATEWAY,
        message: message.into(),
    }
}

pub fn spawn_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> mpsc::Receiver<ChatStreamEvent> {
    spawn_chat_stream_inner(state, request, (), None)
}

/// Spawn a chat stream while retaining `keepalive` until inference has fully
/// unwound and the response consumer has handled the terminal event. This is
/// used by persisted thread chats to keep their per-thread turn lock through
/// cancellation and atomic terminal persistence.
pub fn spawn_chat_stream_with_keepalive<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
) -> (mpsc::Receiver<ChatStreamEvent>, ChatStreamCompletion)
where
    K: Send + 'static,
{
    let (completion_tx, completion_rx) = oneshot::channel();
    let events = spawn_chat_stream_inner(state, request, keepalive, Some(completion_rx));
    (
        events,
        ChatStreamCompletion {
            sender: Some(completion_tx),
        },
    )
}

fn spawn_chat_stream_inner<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
    completion_rx: Option<oneshot::Receiver<()>>,
) -> mpsc::Receiver<ChatStreamEvent>
where
    K: Send + 'static,
{
    let runtime = state.runtime.clone();
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;

    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let backpressure = Arc::new(ChatStreamBackpressure::default());
    tokio::spawn(async move {
        async move {
            let permit = match state
                .acquire_owned_workload_permit(WorkloadClass::Streaming)
                .await
            {
                Ok(permit) => permit,
                Err(_) => {
                    send_chat_terminal(event_tx, ChatStreamEvent::ShuttingDown).await;
                    return;
                }
            };

            if event_tx.send(ChatStreamEvent::Started).await.is_err() {
                return;
            }

            let generation = runtime.chat_generate_streaming_with_runtime_context(
                variant,
                messages,
                params,
                chat_config,
                correlation_id.as_deref(),
                permit.runtime_context(),
                {
                    let event_tx = event_tx.clone();
                    let backpressure = backpressure.clone();
                    move |delta| {
                        try_send_chat_delta(&event_tx, &backpressure, delta);
                    }
                },
            );
            let terminal = resolve_chat_terminal(&event_tx, backpressure, generation).await;
            drop(permit);
            if let Some(terminal) = terminal {
                send_chat_terminal(event_tx, terminal).await;
            }
        }
        .await;

        if let Some(completion_rx) = completion_rx {
            let _ = completion_rx.await;
        }
        drop(keepalive);
    });

    event_rx
}

#[cfg(test)]
fn spawn_chat_stream_with_task<G, Fut>(
    semaphore: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    generation_task: G,
) -> mpsc::Receiver<ChatStreamEvent>
where
    G: FnOnce(mpsc::Sender<ChatStreamEvent>, Arc<ChatStreamBackpressure>) -> Fut + Send + 'static,
    Fut: Future<Output = izwi_core::Result<ChatGeneration>> + Send + 'static,
{
    let (event_tx, event_rx) = mpsc::channel(capacity);
    let backpressure = Arc::new(ChatStreamBackpressure::default());

    tokio::spawn(async move {
        let permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                send_chat_terminal(event_tx, ChatStreamEvent::ShuttingDown).await;
                return;
            }
        };

        if event_tx.send(ChatStreamEvent::Started).await.is_err() {
            return;
        }

        let generation = generation_task(event_tx.clone(), backpressure.clone());
        let terminal = resolve_chat_terminal(&event_tx, backpressure, generation).await;
        drop(permit);
        if let Some(terminal) = terminal {
            send_chat_terminal(event_tx, terminal).await;
        }
    });

    event_rx
}

#[cfg(test)]
fn spawn_chat_stream_with_task_and_keepalive<G, Fut, K>(
    semaphore: Arc<tokio::sync::Semaphore>,
    keepalive: K,
    generation_task: G,
) -> (mpsc::Receiver<ChatStreamEvent>, ChatStreamCompletion)
where
    G: FnOnce(mpsc::Sender<ChatStreamEvent>, Arc<ChatStreamBackpressure>) -> Fut + Send + 'static,
    Fut: Future<Output = izwi_core::Result<ChatGeneration>> + Send + 'static,
    K: Send + 'static,
{
    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let (completion_tx, completion_rx) = oneshot::channel();
    let backpressure = Arc::new(ChatStreamBackpressure::default());

    tokio::spawn(async move {
        async move {
            let permit = semaphore.acquire_owned().await.expect("test semaphore");
            if event_tx.send(ChatStreamEvent::Started).await.is_err() {
                return;
            }
            let generation = generation_task(event_tx.clone(), backpressure.clone());
            let terminal = resolve_chat_terminal(&event_tx, backpressure, generation).await;
            drop(permit);
            if let Some(terminal) = terminal {
                send_chat_terminal(event_tx, terminal).await;
            }
        }
        .await;

        let _ = completion_rx.await;
        drop(keepalive);
    });

    (
        event_rx,
        ChatStreamCompletion {
            sender: Some(completion_tx),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::ChatRole;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::sync::Semaphore;

    #[test]
    fn explicit_overrides_win_over_default_generation_params() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen34BGguf,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: Some(0.42),
            top_p: Some(0.73),
            top_k: None,
            repetition_penalty: None,
            presence_penalty: Some(0.25),
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
    }

    fn qwen38_request(enable_thinking: Option<bool>) -> ChatExecutionRequest {
        ChatExecutionRequest {
            variant: ModelVariant::Qwen3827BFp8,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            presence_penalty: None,
            chat_config: ChatRequestConfig {
                enable_thinking,
                ..Default::default()
            },
            correlation_id: None,
        }
    }

    #[test]
    fn qwen38_uses_thinking_sampling_profile_when_values_are_omitted() {
        let params = qwen38_request(None).resolved_generation_params();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 0.0);
    }

    #[test]
    fn qwen38_uses_non_thinking_sampling_profile_when_disabled() {
        let params = qwen38_request(Some(false)).resolved_generation_params();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.80);
        assert_eq!(params.top_k, 20);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 1.5);
    }

    #[test]
    fn qwen38_explicit_sampling_values_win_over_profile() {
        let mut request = qwen38_request(Some(true));
        request.temperature = Some(0.2);
        request.top_p = Some(0.3);
        request.top_k = Some(7);
        request.repetition_penalty = Some(1.2);
        request.presence_penalty = Some(-0.4);

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.2);
        assert_eq!(params.top_p, 0.3);
        assert_eq!(params.top_k, 7);
        assert_eq!(params.repetition_penalty, 1.2);
        assert_eq!(params.presence_penalty, -0.4);
    }

    #[test]
    fn chat_template_kwargs_conflicts_are_rejected() {
        let err = resolve_chat_request_config(
            Some(true),
            None,
            None,
            Some(&ChatTemplateKwargs {
                enable_thinking: Some(false),
                ..Default::default()
            }),
            Vec::new(),
            Vec::new(),
        )
        .expect_err("conflicting enable_thinking must fail");

        assert!(err.message.contains("Conflicting `enable_thinking`"));
    }

    #[test]
    fn compatible_chat_template_kwargs_resolve_reasoning_controls() {
        let config = resolve_chat_request_config(
            None,
            Some(ChatReasoningEffort::Low),
            Some(false),
            Some(&ChatTemplateKwargs {
                enable_thinking: Some(true),
                reasoning_effort: Some(ChatReasoningEffort::Low),
                preserve_thinking: Some(false),
            }),
            Vec::new(),
            Vec::new(),
        )
        .expect("matching fields should resolve");

        assert_eq!(config.enable_thinking, Some(true));
        assert_eq!(config.reasoning_effort, Some(ChatReasoningEffort::Low));
        assert_eq!(config.preserve_thinking, Some(false));
    }

    #[test]
    fn chat_models_leave_omitted_output_limits_for_exact_context_resolution() {
        for variant in [
            ModelVariant::Gemma34BIt,
            ModelVariant::Lfm2512BInstructGguf,
            ModelVariant::Qwen306BGguf,
            ModelVariant::Qwen317BGguf,
            ModelVariant::Qwen34BGguf,
            ModelVariant::Qwen38BGguf,
            ModelVariant::Qwen314BGguf,
            ModelVariant::Qwen352BGguf,
        ] {
            assert_eq!(max_new_tokens(variant, None, None), usize::MAX);
        }
    }

    #[test]
    fn explicit_chat_output_limits_are_not_capped_at_4096() {
        assert_eq!(
            max_new_tokens(ModelVariant::Qwen3827BFp8, None, Some(32_768)),
            32_768
        );
        assert_eq!(
            max_new_tokens(ModelVariant::Qwen3827BFp8, Some(65_536), Some(32_768)),
            65_536,
            "max_completion_tokens must retain precedence over legacy max_tokens"
        );
        assert_eq!(max_new_tokens(ModelVariant::Qwen3827BFp8, Some(0), None), 1);
    }

    #[tokio::test]
    async fn streaming_chat_allows_long_running_generations_to_complete() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx =
            spawn_chat_stream_with_task(semaphore, 4, |event_tx, backpressure| async move {
                try_send_chat_delta(&event_tx, &backpressure, "Hello".to_string());
                tokio::time::sleep(Duration::from_millis(25)).await;
                try_send_chat_delta(&event_tx, &backpressure, " world".to_string());
                Ok(ChatGeneration {
                    text: "Hello world".to_string(),
                    prompt_tokens: 12,
                    tokens_generated: 2,
                    generation_time_ms: 25.0,
                    latency_breakdown: None,
                    finish_reason: None,
                })
            });

        match event_rx.recv().await {
            Some(ChatStreamEvent::Started) => {}
            other => panic!("expected stream start event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, "Hello"),
            other => panic!("expected first delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, " world"),
            other => panic!("expected second delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Completed(generation)) => {
                assert_eq!(generation.text, "Hello world");
                assert_eq!(generation.tokens_generated, 2);
            }
            other => panic!("expected completed event, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn saturated_chat_stream_emits_explicit_terminal_failure() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx =
            spawn_chat_stream_with_task(semaphore, 2, |event_tx, backpressure| async move {
                try_send_chat_delta(&event_tx, &backpressure, "first".to_string());
                try_send_chat_delta(&event_tx, &backpressure, "overflow".to_string());
                // Completion in the same poll as the overflow must not win the
                // race and turn a truncated stream into apparent success.
                Ok(ChatGeneration {
                    text: "firstoverflow".to_string(),
                    prompt_tokens: 1,
                    tokens_generated: 2,
                    generation_time_ms: 1.0,
                    latency_breakdown: None,
                    finish_reason: None,
                })
            });

        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Delta(delta)) if delta == "first"
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Failed(error)) if error == CHAT_STREAM_BACKPRESSURE_ERROR
        ));
        assert!(event_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn non_draining_chat_consumer_cannot_retain_workload_capacity() {
        let semaphore = Arc::new(Semaphore::new(1));
        let generation_started = Arc::new(Notify::new());
        let release_generation = Arc::new(Notify::new());
        let started = generation_started.clone();
        let release = release_generation.clone();
        let mut event_rx = spawn_chat_stream_with_task(
            semaphore.clone(),
            1,
            move |_event_tx, _backpressure| async move {
                started.notify_one();
                release.notified().await;
                Ok(ChatGeneration {
                    text: "done".to_string(),
                    prompt_tokens: 1,
                    tokens_generated: 1,
                    generation_time_ms: 1.0,
                    latency_breakdown: None,
                    finish_reason: None,
                })
            },
        );

        generation_started.notified().await;
        assert_eq!(semaphore.available_permits(), 0);
        assert_eq!(
            event_rx.len(),
            1,
            "the unread Started event fills the queue"
        );
        release_generation.notify_one();
        tokio::time::timeout(Duration::from_millis(50), async {
            while semaphore.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal transport wait must not retain the workload permit");

        tokio::time::sleep(CHAT_TERMINAL_SEND_TIMEOUT + Duration::from_millis(25)).await;
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(event_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn disconnect_unwinds_generation_before_releasing_stream_keepalive() {
        struct GenerationDrop(Arc<AtomicBool>);
        impl Drop for GenerationDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        struct KeepaliveDrop {
            generation_dropped: Arc<AtomicBool>,
            released: Arc<AtomicBool>,
            released_early: Arc<AtomicBool>,
        }
        impl Drop for KeepaliveDrop {
            fn drop(&mut self) {
                if !self.generation_dropped.load(Ordering::Acquire) {
                    self.released_early.store(true, Ordering::Release);
                }
                self.released.store(true, Ordering::Release);
            }
        }

        let generation_started = Arc::new(Notify::new());
        let generation_dropped = Arc::new(AtomicBool::new(false));
        let keepalive_released = Arc::new(AtomicBool::new(false));
        let keepalive_released_early = Arc::new(AtomicBool::new(false));
        let started = generation_started.clone();
        let generation_drop_for_task = generation_dropped.clone();
        let keepalive = KeepaliveDrop {
            generation_dropped: generation_dropped.clone(),
            released: keepalive_released.clone(),
            released_early: keepalive_released_early.clone(),
        };
        let (mut event_rx, completion) = spawn_chat_stream_with_task_and_keepalive(
            Arc::new(Semaphore::new(1)),
            keepalive,
            move |_event_tx, _backpressure| async move {
                let _drop = GenerationDrop(generation_drop_for_task);
                started.notify_one();
                std::future::pending::<()>().await;
                unreachable!("pending generation should be cancelled")
            },
        );

        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        generation_started.notified().await;
        drop(event_rx);
        drop(completion);

        tokio::time::timeout(Duration::from_secs(1), async {
            while !keepalive_released.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("keepalive should release after cancellation");
        assert!(generation_dropped.load(Ordering::Acquire));
        assert!(!keepalive_released_early.load(Ordering::Acquire));
    }
}

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use izwi_serving_client::{InvocationStream, TraceContext, WorkerClient, WorkerClientError};
use izwi_serving_protocol::{
    AttemptId, AttemptIdentity, CallerId, ChatContextPolicy, ChatInput,
    ChatMessage as WorkerChatMessage, ChatParameters as WorkerChatParameters, ChatResponseFormat,
    ChatRole as WorkerChatRole, ChatTruncation as WorkerChatTruncation, DeploymentId,
    FinishReason as WorkerFinishReason, GatewayAttestedCallerContext, IncarnationId,
    InvocationErrorCode, InvocationEventKind, InvocationInput, InvocationRequest, ModelGeneration,
    OutputFormat, OutputLimits, PermittedAction, PolicyRevision, RequestDigest, RequestId,
    SchemaVersion, ServiceClass, TaskKind, TenantId, Usage, WorkerFeature,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use tokio::sync::Notify;
use tokio::sync::{mpsc, oneshot};

use crate::api::request_context::RequestContext;
use crate::error::{invocation_error_class, rejection_class, ApiError, ErrorClass};
use crate::gateway_tenant_concurrency::{BoundTenantWorkLease, UnboundTenantWorkLease};
use crate::ids::new_uuid;
use crate::state::AppState;
use izwi_core::{
    parse_chat_model_variant, resolve_chat_max_tokens, resolve_chat_sampling, ChatGeneration,
    ChatGenerationControls, ChatMediaInput, ChatMessage, ChatReasoningEffort, ChatRequestConfig,
    ChatSamplingRequest, ChatTemplateKwargs, ChatTruncation, GenerationParams, ModelVariant,
    ResolvedChatSampling, WorkloadClass, CHAT_MAX_TOKENS_UNSET,
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
    /// OpenAI `seed`: the same seed and parameters sample the same tokens.
    pub seed: Option<u64>,
    /// OpenAI `stop`: generation ends before the first of these strings,
    /// which is never returned.
    pub stop: Vec<String>,
    /// DS9.3: OpenAI `logprobs` flag and `top_logprobs` count (0-20).
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    /// DS9.2: the request carried `response_format: json_object`.
    pub response_format_json_object: bool,
    pub chat_config: ChatRequestConfig,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        resolve_chat_max_tokens(self.max_completion_tokens, self.max_tokens)
    }

    /// The sampling this request runs with: explicit values, else the model
    /// family's published profile, else the engine default.
    fn resolved_sampling(&self) -> ResolvedChatSampling {
        resolve_chat_sampling(
            self.variant,
            self.chat_config.enable_thinking,
            ChatSamplingRequest {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                repetition_penalty: self.repetition_penalty,
                presence_penalty: self.presence_penalty,
            },
        )
    }

    fn resolved_generation_params(&self) -> GenerationParams {
        self.resolved_sampling().into_generation_params(
            self.resolved_max_new_tokens(),
            ChatGenerationControls {
                seed: self.seed,
                stop: self.stop.clone(),
                logprobs: self.requested_logprobs(),
                constrain_json_object: self.response_format_json_object,
            },
        )
    }

    /// `Some(top_logprobs)` when the request asked for logprobs.
    fn requested_logprobs(&self) -> Option<usize> {
        self.logprobs
            .unwrap_or(false)
            .then(|| self.top_logprobs.map_or(0, usize::from))
    }

    fn resolved_chat_config(&self) -> ChatRequestConfig {
        self.chat_config.clone()
    }

    /// The worker parameters for this request on a worker advertising
    /// `features`. Temperature and top-p are sent resolved (a worker before
    /// minor 4 decodes greedily without them; a later one resolves the same
    /// profile); every other control is sent only when the request set it,
    /// so a later worker resolves the rest of the profile itself. The D8
    /// context policy is sent to every worker that takes it.
    fn worker_chat_parameters(&self, features: &BTreeSet<WorkerFeature>) -> WorkerChatParameters {
        let sampling = self.resolved_sampling();
        let saturating = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
        WorkerChatParameters {
            temperature: Some(sampling.temperature),
            top_p: Some(sampling.top_p),
            seed: self.seed,
            stop: self.stop.clone(),
            top_k: self.top_k.map(saturating),
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            logprobs: self
                .logprobs
                .unwrap_or(false)
                .then(|| self.top_logprobs.unwrap_or(0)),
            response_format: self
                .response_format_json_object
                .then_some(ChatResponseFormat::JsonObject),
            tools: self.chat_config.tools.clone(),
            context: features
                .contains(&WorkerFeature::ChatContextPolicy)
                .then(|| ChatContextPolicy {
                    max_tokens: match self.resolved_max_new_tokens() {
                        CHAT_MAX_TOKENS_UNSET => None,
                        max_tokens => Some(saturating(max_tokens)),
                    },
                    truncation: match self.chat_config.truncation {
                        ChatTruncation::Disabled => WorkerChatTruncation::Disabled,
                        ChatTruncation::Auto => WorkerChatTruncation::Auto,
                    },
                }),
        }
    }

    /// The worker features this request needs: one per control it set.
    pub(crate) fn required_worker_features(&self) -> BTreeSet<WorkerFeature> {
        self.worker_chat_parameters(&BTreeSet::new())
            .required_features()
    }

    /// The OpenAI name of the first control this request set that needs a
    /// worker feature.
    pub(crate) fn first_feature_gated_parameter(&self) -> Option<&'static str> {
        self.worker_chat_parameters(&BTreeSet::new())
            .first_unsupported(&BTreeSet::new())
    }
}

/// The protocol surface of one worker incarnation, from its descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemoteWorkerContract {
    /// The version negotiated with the worker; invocations carry it.
    pub(crate) protocol_version: SchemaVersion,
    pub(crate) features: BTreeSet<WorkerFeature>,
}

/// One selected worker incarnation as a chat invocation target. Registry
/// dispatch builds it per attempt from the worker it selected; the worker
/// still makes authoritative admission and generation checks on every
/// invocation.
#[derive(Debug, Clone)]
pub(crate) struct RemoteChatExecutionConfig {
    pub public_model_variant: ModelVariant,
    pub expected_worker_incarnation: IncarnationId,
    pub deployment_id: DeploymentId,
    pub expected_model_generation: ModelGeneration,
    pub policy_revision: PolicyRevision,
    pub max_queue_wait: Duration,
    pub max_output_tokens: u32,
    pub max_output_bytes: u64,
    pub slow_consumer_timeout: Duration,
}

impl RemoteChatExecutionConfig {
    fn validate(&self) -> Result<(), ApiError> {
        if self.max_output_tokens == 0
            || self.max_output_bytes == 0
            || self.slow_consumer_timeout.is_zero()
        {
            return Err(ApiError::internal(
                "Remote chat output limits and slow-consumer timeout must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RemoteChatExecution {
    client: WorkerClient,
    config: RemoteChatExecutionConfig,
    /// The selected worker's protocol surface, from the registry.
    contract: RemoteWorkerContract,
}

impl RemoteChatExecution {
    pub(crate) fn new(
        client: WorkerClient,
        config: RemoteChatExecutionConfig,
        contract: RemoteWorkerContract,
    ) -> Result<Self, ApiError> {
        config.validate()?;
        Ok(Self {
            client,
            config,
            contract,
        })
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
    truncation: Option<ChatTruncation>,
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
        truncation: truncation.unwrap_or_default(),
    })
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started,
    /// DS9.3: a text delta with its per-token logprob entries (empty unless
    /// the request asked for logprobs).
    Delta {
        text: String,
        logprobs: Vec<izwi_core::engine::TokenLogprob>,
    },
    Completed(Box<ChatGeneration>),
    /// The stream failed after it started; the error is sent to the client
    /// as an in-band error event.
    Failed(ApiError),
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
/// The slow-consumer timeout of the test stream spawners.
#[cfg(test)]
const CHAT_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_millis(100);

/// Upper bound on text merged for a lagging consumer. A consumer that owes
/// this much is not catching up, so the stream is cancelled.
const MAX_COALESCED_DELTA_BYTES: usize = 4 * 1024 * 1024;

/// Text deltas on their way to a chat stream consumer. While the consumer
/// lags (the bounded channel is full), deltas and their logprobs merge, in
/// order, into one pending delta that is sent as a single event once the
/// consumer frees a slot. A lagging consumer therefore costs bytes, not
/// queued events; only one that accepts nothing for the slow-consumer
/// timeout, or owes more than `MAX_COALESCED_DELTA_BYTES`, stalls the stream.
struct ChatDeltaSink {
    event_tx: mpsc::Sender<ChatStreamEvent>,
    slow_consumer_timeout: Duration,
    outbox: std::sync::Mutex<ChatDeltaOutbox>,
    pending_ready: Notify,
}

#[derive(Default)]
struct ChatDeltaOutbox {
    pending: Option<PendingChatDelta>,
    stalled: bool,
}

struct PendingChatDelta {
    text: String,
    logprobs: Vec<izwi_core::engine::TokenLogprob>,
    since: tokio::time::Instant,
}

impl ChatDeltaSink {
    fn new(event_tx: mpsc::Sender<ChatStreamEvent>, slow_consumer_timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            event_tx,
            slow_consumer_timeout,
            outbox: std::sync::Mutex::new(ChatDeltaOutbox::default()),
            pending_ready: Notify::new(),
        })
    }

    fn outbox(&self) -> std::sync::MutexGuard<'_, ChatDeltaOutbox> {
        self.outbox
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_stalled(&self) -> bool {
        self.outbox().stalled
    }

    /// Sends a delta now, or merges it into the pending delta while the
    /// consumer lags. Never blocks, so a synchronous runtime callback can
    /// call it.
    fn offer(&self, text: String, logprobs: Vec<izwi_core::engine::TokenLogprob>) {
        let mut outbox = self.outbox();
        if outbox.stalled || self.event_tx.is_closed() {
            return;
        }
        if outbox.pending.is_none() && self.event_tx.capacity() > 0 {
            // Deltas are the only events sent while a generation runs, and
            // they are sent under this lock, so the free slot is still free.
            let _ = self
                .event_tx
                .try_send(ChatStreamEvent::Delta { text, logprobs });
            return;
        }
        let pending = outbox.pending.get_or_insert_with(|| PendingChatDelta {
            text: String::new(),
            logprobs: Vec::new(),
            since: tokio::time::Instant::now(),
        });
        pending.text.push_str(&text);
        pending.logprobs.extend(logprobs);
        if pending.text.len() > MAX_COALESCED_DELTA_BYTES {
            outbox.stalled = true;
        }
        drop(outbox);
        self.pending_ready.notify_one();
    }

    fn send_pending(&self, permit: mpsc::Permit<'_, ChatStreamEvent>) {
        let mut outbox = self.outbox();
        if let Some(pending) = outbox.pending.take() {
            permit.send(ChatStreamEvent::Delta {
                text: pending.text,
                logprobs: pending.logprobs,
            });
        }
    }

    /// Delivers the pending delta whenever the consumer frees a slot, and
    /// completes only once the consumer has stalled.
    async fn stalled(&self) {
        loop {
            let since = {
                let outbox = self.outbox();
                if outbox.stalled {
                    return;
                }
                outbox.pending.as_ref().map(|pending| pending.since)
            };
            let Some(since) = since else {
                self.pending_ready.notified().await;
                continue;
            };
            tokio::select! {
                permit = self.event_tx.reserve() => match permit {
                    Ok(permit) => self.send_pending(permit),
                    // The consumer is gone; callers watch the closed channel.
                    Err(_) => std::future::pending().await,
                },
                () = tokio::time::sleep_until(since + self.slow_consumer_timeout) => {
                    self.outbox().stalled = true;
                    return;
                }
            }
        }
    }

    /// Delivers any pending delta and then `terminal`, each within the
    /// slow-consumer timeout. A terminal event never overtakes text: if the
    /// pending delta cannot be delivered the stream ends as failed.
    async fn finish(&self, terminal: ChatStreamEvent) {
        let (stalled, pending_since) = {
            let outbox = self.outbox();
            (
                outbox.stalled,
                outbox.pending.as_ref().map(|pending| pending.since),
            )
        };
        if stalled {
            // The consumer already spent its budget.
            let _ = self.event_tx.try_send(terminal);
            return;
        }
        if let Some(since) = pending_since {
            let deadline = since + self.slow_consumer_timeout;
            match tokio::time::timeout_at(deadline, self.event_tx.reserve()).await {
                Ok(Ok(permit)) => self.send_pending(permit),
                Ok(Err(_)) => return,
                Err(_) => {
                    let _ = self
                        .event_tx
                        .try_send(ChatStreamEvent::Failed(backpressure_error()));
                    return;
                }
            }
        }
        send_chat_terminal_with_timeout(
            self.event_tx.clone(),
            terminal,
            self.slow_consumer_timeout,
        )
        .await;
    }
}

/// The stream's terminal event once `generation` ends, the consumer goes
/// away (`None`), the consumer stalls, or a `true` on `shutdown` (the local
/// role's stream notice) ends the stream with a "shutting down" failure.
/// Dropping `generation` cancels the request.
async fn resolve_chat_terminal<F>(
    sink: &ChatDeltaSink,
    generation: F,
    mut shutdown: Option<tokio::sync::watch::Receiver<bool>>,
) -> Option<ChatStreamEvent>
where
    F: Future<Output = izwi_core::Result<ChatGeneration>>,
{
    tokio::pin!(generation);
    let result = tokio::select! {
        result = &mut generation => Some(result),
        _ = sink.event_tx.closed() => return None,
        _ = sink.stalled() => None,
        () = shutdown_requested(shutdown.as_mut()) => {
            return Some(ChatStreamEvent::Failed(ApiError::service_unavailable(
                "The server is shutting down; retry the request",
            )));
        }
    };

    // A generation can overflow the merged delta and complete in the same
    // poll. Check the latch even when the generation branch wins the select
    // so lost text can never be reported as a successful completion.
    if result.is_none() || sink.is_stalled() {
        return Some(ChatStreamEvent::Failed(backpressure_error()));
    }

    match result.expect("completed generation result must be present") {
        Ok(generation) => Some(ChatStreamEvent::Completed(Box::new(generation))),
        Err(err) => Some(ChatStreamEvent::Failed(chat_error(err))),
    }
}

fn backpressure_error() -> ApiError {
    ApiError::service_unavailable(CHAT_STREAM_BACKPRESSURE_ERROR)
}

async fn send_chat_terminal_with_timeout(
    event_tx: mpsc::Sender<ChatStreamEvent>,
    event: ChatStreamEvent,
    timeout: Duration,
) {
    // A connected receiver may stop polling forever. Terminal delivery remains
    // best-effort for that transport, but it must never retain inference or
    // workload capacity indefinitely.
    let _ = tokio::time::timeout(timeout, event_tx.send(event)).await;
}

/// EDGE-F19: the OpenAI and vLLM ranges of the chat sampling controls,
/// enforced at the edge so every role and model family rejects the same
/// values with the same `400` instead of clamping them or failing inside a
/// family's sampler.
pub(crate) fn validate_chat_request_ranges(request: &ChatExecutionRequest) -> Result<(), ApiError> {
    let within = |value: Option<f32>, min: f32, max: f32| {
        value.is_none_or(|value| value.is_finite() && (min..=max).contains(&value))
    };
    if !within(request.temperature, 0.0, 2.0) {
        return Err(ApiError::invalid_param(
            "temperature",
            "`temperature` must be between 0 and 2",
        ));
    }
    if request
        .top_p
        .is_some_and(|top_p| !(top_p > 0.0 && top_p <= 1.0))
    {
        return Err(ApiError::invalid_param(
            "top_p",
            "`top_p` must be greater than 0 and at most 1",
        ));
    }
    // The samplers penalize repetition only; a value below 1 cannot be
    // honoured, so it is refused rather than silently raised to 1.
    if !within(request.repetition_penalty, 1.0, 2.0) {
        return Err(ApiError::invalid_param(
            "repetition_penalty",
            "`repetition_penalty` must be between 1 and 2",
        ));
    }
    if !within(request.presence_penalty, -2.0, 2.0) {
        return Err(ApiError::invalid_param(
            "presence_penalty",
            "`presence_penalty` must be between -2 and 2",
        ));
    }
    for (param, value) in [
        ("max_tokens", request.max_tokens),
        ("max_completion_tokens", request.max_completion_tokens),
    ] {
        if value == Some(0) {
            return Err(ApiError::invalid_param(
                param,
                format!("`{param}` must be at least 1"),
            ));
        }
    }
    Ok(())
}

/// OpenAI-compatible `top_k`: `-1` (vLLM) and `0` disable top-k sampling;
/// any other negative value is refused.
pub(crate) fn chat_top_k(top_k: Option<i64>) -> Result<Option<usize>, ApiError> {
    match top_k {
        None => Ok(None),
        Some(-1) => Ok(Some(0)),
        Some(top_k) => usize::try_from(top_k).map(Some).map_err(|_| {
            ApiError::invalid_param(
                "top_k",
                "`top_k` must be -1 or 0 (disabled) or a positive integer",
            )
        }),
    }
}

pub fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

/// The execution target behind the public chat completions route: the
/// in-process runtime ([`LocalChat`]) or the gateway's worker registry
/// (`GatewayChat`). A backend value serves exactly one request, so it can own
/// that request's admission and tenant leases until they are released.
pub(crate) trait ChatBackend: Send {
    /// Held by a streaming response until its body ends.
    type StreamGuard: ChatStreamObserver;

    fn generate(
        self,
        request: ChatExecutionRequest,
    ) -> impl Future<Output = Result<ChatGeneration, ApiError>> + Send;

    /// Returns once the stream has started; its events then arrive on the
    /// receiver.
    fn stream(
        self,
        request: ChatExecutionRequest,
    ) -> impl Future<Output = Result<(mpsc::Receiver<ChatStreamEvent>, Self::StreamGuard), ApiError>>
           + Send;
}

/// Told when a streaming chat response emits text and how it ended.
pub(crate) trait ChatStreamObserver: Send + 'static {
    /// A delta carrying answer text was sent to the client.
    fn record_delta(&mut self) {}
    fn record_completion(&mut self, _generation: &ChatGeneration) {}
    fn record_failure(&mut self) {}
}

impl ChatStreamObserver for () {}

/// Chat on this process's own runtime.
pub(crate) struct LocalChat(pub(crate) AppState);

impl ChatBackend for LocalChat {
    type StreamGuard = ();

    async fn generate(self, request: ChatExecutionRequest) -> Result<ChatGeneration, ApiError> {
        generate_chat(&self.0, request).await
    }

    async fn stream(
        self,
        request: ChatExecutionRequest,
    ) -> Result<(mpsc::Receiver<ChatStreamEvent>, ()), ApiError> {
        Ok((start_chat_stream(self.0, request).await?, ()))
    }
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
        .map_err(chat_error)
}

/// A local chat failure as an API error. An overflow is about the request's
/// `messages`, as the gateway reports it.
fn chat_error(err: izwi_core::Error) -> ApiError {
    let overflow = err.kind() == izwi_core::ErrorKind::ContextLengthExceeded;
    let error = ApiError::from(err);
    if overflow {
        error.with_param("messages")
    } else {
        error
    }
}

/// Collect an accepted invocation's whole answer. The tenant lease is
/// released as soon as a validated terminal event proves the worker stopped;
/// any other exit drops it, which reconciles the exact attempt.
pub(crate) async fn collect_started_remote_chat<F>(
    remote: &RemoteChatExecution,
    stream: &mut InvocationStream,
    started: Instant,
    mut on_worker_error: F,
    tenant_work: BoundTenantWorkLease,
) -> Result<ChatGeneration, ApiError>
where
    F: FnMut(&WorkerClientError),
{
    let mut text = String::new();
    let mut latest_usage = None;
    let mut logprob_entries = RemoteLogprobCollector::default();

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
            // An event type from a newer protocol minor is not terminal.
            InvocationEventKind::Accepted { .. } | InvocationEventKind::Unknown => {}
            InvocationEventKind::TextDelta {
                text: delta,
                logprobs,
            } => {
                append_remote_text(&mut text, &delta, remote.config.max_output_bytes)?;
                logprob_entries.extend(logprobs);
            }
            InvocationEventKind::Usage { usage } => latest_usage = Some(usage),
            InvocationEventKind::Completed {
                finish_reason,
                usage,
            } => {
                tenant_work.confirm_stopped();
                return worker_chat_generation(
                    text,
                    finish_reason,
                    usage.or(latest_usage),
                    started,
                    logprob_entries.finish(),
                );
            }
            InvocationEventKind::Error { code, message } => {
                tenant_work.confirm_stopped();
                return Err(map_worker_terminal_error(code, message));
            }
            InvocationEventKind::Cancelled { reason } => {
                tenant_work.confirm_stopped();
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

pub(crate) struct StartedTenantInvocation {
    pub(crate) stream: InvocationStream,
    pub(crate) tenant_work: BoundTenantWorkLease,
}

pub(crate) struct TenantInvocationStartFailure {
    pub(crate) error: WorkerClientError,
    /// Present only when the failure proves the exact attempt was never
    /// admitted. The same public-request lease may then be rebound to the one
    /// permitted alternate attempt.
    pub(crate) retry_tenant_work: Option<UnboundTenantWorkLease>,
}

pub(crate) async fn start_remote_chat_invocation_with_tenant(
    remote: &RemoteChatExecution,
    invocation: InvocationRequest,
    tenant_work: UnboundTenantWorkLease,
    trace: &TraceContext,
) -> Result<StartedTenantInvocation, TenantInvocationStartFailure> {
    let identity = AttemptIdentity::from(&invocation);
    let bound = tenant_work.bind(remote.client.clone(), identity);
    match remote.client.invoke_traced(invocation, Some(trace)).await {
        Ok(stream) => Ok(StartedTenantInvocation {
            stream,
            tenant_work: bound,
        }),
        Err(error) if error.proves_attempt_unaccepted() => Err(TenantInvocationStartFailure {
            error,
            retry_tenant_work: Some(bound.prove_unaccepted()),
        }),
        Err(error) => {
            // Dropping a bound lease retains ownership and reconciles the exact
            // attempt; an uncertain failure is never made eligible for retry.
            drop(bound);
            Err(TenantInvocationStartFailure {
                error,
                retry_tenant_work: None,
            })
        }
    }
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

/// A worker cannot honour a control it does not advertise; refuse it rather
/// than let the worker drop it.
fn ensure_worker_supports(
    parameters: &WorkerChatParameters,
    contract: &RemoteWorkerContract,
) -> Result<(), ApiError> {
    match parameters.first_unsupported(&contract.features) {
        Some(parameter) => Err(ApiError::unsupported_param(
            parameter,
            format!("The selected worker does not support `{parameter}`"),
        )),
        None => Ok(()),
    }
}

/// The same invocation for an alternate worker. Registry selection gives it
/// every feature the request's own controls need; the D8 context policy is
/// dropped for an alternate that does not take it, which then keeps its
/// ceiling-as-budget behaviour.
pub(crate) fn retarget_remote_chat_invocation(
    invocation: &InvocationRequest,
    remote: &RemoteChatExecution,
    remaining: Duration,
) -> Result<InvocationRequest, ApiError> {
    let contract = &remote.contract;
    if remaining.is_zero() {
        return Err(request_timeout_error(
            "Chat request deadline expired before alternate dispatch",
        ));
    }
    let remaining_time_ms = u64::try_from(remaining.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let mut alternate = invocation.clone();
    alternate.schema_version = contract.protocol_version;
    let InvocationInput::Chat { parameters, .. } = &mut alternate.input;
    if !contract
        .features
        .contains(&WorkerFeature::ChatContextPolicy)
    {
        parameters.context = None;
    }
    ensure_worker_supports(parameters, contract)?;
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

/** DS9.3: bound on accumulated worker logprob entries per response. */
const REMOTE_CHAT_LOGPROB_ENTRY_LIMIT: usize = 65_536;

/// DS9.3: map protocol logprob entries onto the core payload shape.
fn remote_logprobs(
    entries: Option<Vec<izwi_serving_protocol::TokenLogprob>>,
) -> Vec<izwi_core::engine::TokenLogprob> {
    entries
        .unwrap_or_default()
        .into_iter()
        .map(|entry| izwi_core::engine::TokenLogprob {
            token: entry.token,
            logprob: entry.logprob,
            bytes: entry.bytes,
            top_logprobs: entry
                .top_logprobs
                .into_iter()
                .map(|top| izwi_core::engine::TopTokenLogprob {
                    token: top.token,
                    logprob: top.logprob,
                    bytes: top.bytes,
                })
                .collect(),
        })
        .collect()
}

/// DS9.3: bounded accumulator for entries collected across worker deltas.
#[derive(Default)]
struct RemoteLogprobCollector {
    entries: Vec<izwi_core::engine::TokenLogprob>,
}

impl RemoteLogprobCollector {
    /// Records one delta's entries and returns the ones it kept: a streamed
    /// delta carries only its own logprobs, and the cumulative list is kept
    /// for the terminal generation.
    fn extend(
        &mut self,
        entries: Option<Vec<izwi_serving_protocol::TokenLogprob>>,
    ) -> Vec<izwi_core::engine::TokenLogprob> {
        let mut mapped = remote_logprobs(entries);
        mapped.truncate(REMOTE_CHAT_LOGPROB_ENTRY_LIMIT.saturating_sub(self.entries.len()));
        self.entries.extend(mapped.iter().cloned());
        mapped
    }

    fn finish(self) -> Vec<izwi_core::engine::TokenLogprob> {
        self.entries
    }
}

fn worker_chat_generation(
    text: String,
    finish_reason: WorkerFinishReason,
    usage: Option<Usage>,
    started: Instant,
    logprobs: Vec<izwi_core::engine::TokenLogprob>,
) -> Result<ChatGeneration, ApiError> {
    let usage = usage.unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        cached_tokens: None,
    });
    Ok(ChatGeneration {
        latency_breakdown: None,
        finish_reason: Some(match finish_reason {
            WorkerFinishReason::Stop | WorkerFinishReason::Unknown => {
                izwi_core::engine::OutputFinishReason::StopToken
            }
            WorkerFinishReason::Length => izwi_core::engine::OutputFinishReason::MaxTokens,
        }),
        text,
        prompt_tokens: usize::try_from(usage.input_tokens)
            .map_err(|_| bad_gateway_error("Worker reported an invalid input token count"))?,
        tokens_generated: usize::try_from(usage.output_tokens)
            .map_err(|_| bad_gateway_error("Worker reported an invalid output token count"))?,
        generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        cached_prompt_tokens: usage.cached_tokens,
        logprobs,
    })
}

/// Forward an accepted private worker invocation's bounded events through
/// the public chat stream channel. Dropping the receiver closes the channel,
/// drops the private stream, and schedules exact-attempt cancellation;
/// neither layer retries the invocation.
/// `reservation` (the registry dispatch) lives in the relay task until the
/// terminal event is delivered or the consumer goes away.
/// A `true` on `shutdown` ends the stream with an explicit "shutting down"
/// failure, so a stream still running when the gateway stops sends the
/// client a terminal event instead of being cut.
pub(crate) fn spawn_started_remote_chat_stream<F, R>(
    remote: &RemoteChatExecution,
    mut worker_stream: InvocationStream,
    on_worker_error: F,
    tenant_work: BoundTenantWorkLease,
    reservation: R,
    mut shutdown: Option<tokio::sync::watch::Receiver<bool>>,
) -> mpsc::Receiver<ChatStreamEvent>
where
    F: Fn(&WorkerClientError) + Send + Sync + 'static,
    R: Send + 'static,
{
    let max_output_bytes = remote.config.max_output_bytes;
    let slow_consumer_timeout = remote.config.slow_consumer_timeout;
    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    tokio::spawn(async move {
        let _reservation = reservation;
        let started = Instant::now();
        let mut logprob_entries = RemoteLogprobCollector::default();
        let sink = ChatDeltaSink::new(event_tx.clone(), slow_consumer_timeout);
        let mut text = String::new();
        let mut latest_usage = None;
        let mut tenant_work = Some(tenant_work);
        let terminal = loop {
            let event = tokio::select! {
                event = worker_stream.next_event() => event,
                () = event_tx.closed() => return,
                () = sink.stalled() => {
                    break ChatStreamEvent::Failed(backpressure_error());
                }
                () = shutdown_requested(shutdown.as_mut()) => {
                    break ChatStreamEvent::Failed(ApiError::service_unavailable(
                        "The gateway is shutting down; retry the request",
                    ));
                }
            };
            let event = match event {
                Ok(Some(event)) => event,
                Ok(None) => {
                    break ChatStreamEvent::Failed(bad_gateway_error(
                        "Worker chat stream ended without a terminal event",
                    ))
                }
                Err(error) => {
                    on_worker_error(&error);
                    break ChatStreamEvent::Failed(map_worker_client_error(error));
                }
            };
            match event.event {
                InvocationEventKind::Accepted { .. } => {
                    if event_tx.try_send(ChatStreamEvent::Started).is_err() {
                        return;
                    }
                }
                InvocationEventKind::TextDelta {
                    text: delta,
                    logprobs,
                } => {
                    if let Err(error) = append_remote_text(&mut text, &delta, max_output_bytes) {
                        break ChatStreamEvent::Failed(error);
                    }
                    let entries = logprob_entries.extend(logprobs);
                    sink.offer(delta, entries);
                    if event_tx.is_closed() {
                        return;
                    }
                    if sink.is_stalled() {
                        break ChatStreamEvent::Failed(backpressure_error());
                    }
                }
                InvocationEventKind::Usage { usage } => latest_usage = Some(usage),
                // An event type from a newer protocol minor is not terminal.
                InvocationEventKind::Unknown => {}
                InvocationEventKind::Completed {
                    finish_reason,
                    usage,
                } => {
                    if let Some(lease) = tenant_work.take() {
                        lease.confirm_stopped();
                    }
                    match worker_chat_generation(
                        text,
                        finish_reason,
                        usage.or(latest_usage),
                        started,
                        logprob_entries.finish(),
                    ) {
                        Ok(generation) => break ChatStreamEvent::Completed(Box::new(generation)),
                        Err(error) => break ChatStreamEvent::Failed(error),
                    }
                }
                InvocationEventKind::Error { code, message } => {
                    if let Some(lease) = tenant_work.take() {
                        lease.confirm_stopped();
                    }
                    break ChatStreamEvent::Failed(map_worker_terminal_error(code, message));
                }
                InvocationEventKind::Cancelled { reason } => {
                    if let Some(lease) = tenant_work.take() {
                        lease.confirm_stopped();
                    }
                    break ChatStreamEvent::Failed(cancelled_error(
                        reason.unwrap_or_else(|| "Worker cancelled chat invocation".into()),
                    ));
                }
            }
        };
        // Release or cancel the private stream before terminal delivery can
        // wait on a slow public consumer. Worker capacity remains governed by
        // confirmed teardown, not by this gateway channel.
        drop(worker_stream);
        // On a transport/protocol interruption this starts exact-attempt
        // reconciliation. On a validated terminal event the lease was already
        // consumed above, so this is a no-op.
        drop(tenant_work);
        sink.finish(terminal).await;
    });
    event_rx
}

/// Resolves once `shutdown` holds `true`; never without a signal.
async fn shutdown_requested(shutdown: Option<&mut tokio::sync::watch::Receiver<bool>>) {
    match shutdown {
        Some(shutdown) => {
            if shutdown.wait_for(|stopping| *stopping).await.is_err() {
                std::future::pending::<()>().await;
            }
        }
        None => std::future::pending().await,
    }
}

fn validate_remote_chat_scope(request: &ChatExecutionRequest) -> Result<(), ApiError> {
    if !request.chat_config.media_inputs.is_empty() {
        return Err(ApiError::bad_request(
            "Remote chat execution currently supports text-only requests",
        ));
    }
    // The worker protocol carries the sampling and output controls and the
    // tools (each gated by a worker feature) but not reasoning controls or
    // media; those are refused, never silently dropped.
    let unsupported = [
        (
            "enable_thinking",
            request.chat_config.enable_thinking.is_some(),
        ),
        (
            "reasoning_effort",
            request.chat_config.reasoning_effort.is_some(),
        ),
        (
            "preserve_thinking",
            request.chat_config.preserve_thinking.is_some(),
        ),
    ];
    if let Some((parameter, _)) = unsupported.iter().find(|(_, requested)| *requested) {
        return Err(ApiError::bad_request(format!(
            "Remote chat execution does not support `{parameter}`"
        )));
    }
    Ok(())
}

fn build_remote_chat_invocation(
    context: &RequestContext,
    request: ChatExecutionRequest,
    remote: &RemoteChatExecution,
    remaining_time_ms: u64,
) -> Result<InvocationRequest, ApiError> {
    let contract = &remote.contract;
    let requested_tokens = request.resolved_max_new_tokens();
    let max_tokens = u32::try_from(requested_tokens)
        .unwrap_or(u32::MAX)
        .min(remote.config.max_output_tokens)
        .max(1);
    let parameters = request.worker_chat_parameters(&contract.features);
    ensure_worker_supports(&parameters, contract)?;
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
                        izwi_core::ChatRole::Tool => WorkerChatRole::Tool,
                    },
                    content: message.content,
                })
                .collect(),
        },
        parameters,
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
        schema_version: contract.protocol_version,
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
            let class = rejection_class(rejection.code);
            let error = ApiError::of(class, rejection.message);
            if class == ErrorClass::ContextLengthExceeded {
                error.with_param("messages")
            } else {
                error
            }
        }
        WorkerClientError::AdmissionUnknown { .. } => {
            ApiError::service_unavailable(error.to_string())
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
    ApiError::of(invocation_error_class(code), message)
}

fn request_timeout_error(message: impl Into<String>) -> ApiError {
    ApiError::of(ErrorClass::Timeout, message)
}

fn cancelled_error(message: impl Into<String>) -> ApiError {
    ApiError::of(ErrorClass::Cancelled, message)
}

fn bad_gateway_error(message: impl Into<String>) -> ApiError {
    ApiError::of(ErrorClass::BadGateway, message)
}

/// Start a chat stream on this process's runtime. The request is admitted
/// first, so a request that cannot run (its prompt does not fit the context,
/// the server is shutting down) fails here, before a streamed response
/// commits to `200`.
pub async fn start_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError> {
    start_chat_stream_inner(state, request, (), None).await
}

/// [`start_chat_stream`] retaining `keepalive` until inference has fully
/// unwound and the response consumer has handled the terminal event. This is
/// used by persisted thread chats to keep their per-thread turn lock through
/// cancellation and atomic terminal persistence.
pub async fn start_chat_stream_with_keepalive<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
) -> Result<(mpsc::Receiver<ChatStreamEvent>, ChatStreamCompletion), ApiError>
where
    K: Send + 'static,
{
    let (completion_tx, completion_rx) = oneshot::channel();
    let events = start_chat_stream_inner(state, request, keepalive, Some(completion_rx)).await?;
    Ok((
        events,
        ChatStreamCompletion {
            sender: Some(completion_tx),
        },
    ))
}

async fn start_chat_stream_inner<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
    completion_rx: Option<oneshot::Receiver<()>>,
) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError>
where
    K: Send + 'static,
{
    let runtime = state.runtime.clone();
    let shutdown = state.stream_shutdown.subscribe();
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let permit = state
        .acquire_owned_workload_permit(WorkloadClass::Streaming)
        .await
        .map_err(|_| ApiError::service_unavailable("Server is shutting down"))?;
    let admitted = runtime
        .admit_streaming_chat(
            request.variant,
            request.messages,
            params,
            chat_config,
            request.correlation_id.as_deref(),
            permit.runtime_context(),
        )
        .await
        .map_err(chat_error)?;

    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let sink = ChatDeltaSink::new(event_tx.clone(), izwi_core::stream_slow_consumer_timeout());
    tokio::spawn(async move {
        async move {
            if event_tx.send(ChatStreamEvent::Started).await.is_err() {
                return;
            }

            let generation = runtime.run_admitted_streaming_chat(admitted, {
                let sink = sink.clone();
                move |delta: String, logprobs: Vec<izwi_core::engine::TokenLogprob>| {
                    sink.offer(delta, logprobs);
                }
            });
            let terminal = resolve_chat_terminal(&sink, generation, Some(shutdown)).await;
            drop(permit);
            if let Some(terminal) = terminal {
                sink.finish(terminal).await;
            }
        }
        .await;

        if let Some(completion_rx) = completion_rx {
            let _ = completion_rx.await;
        }
        drop(keepalive);
    });

    Ok(event_rx)
}

#[cfg(test)]
fn spawn_chat_stream_with_task<G, Fut>(
    semaphore: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    generation_task: G,
) -> mpsc::Receiver<ChatStreamEvent>
where
    G: FnOnce(Arc<ChatDeltaSink>) -> Fut + Send + 'static,
    Fut: Future<Output = izwi_core::Result<ChatGeneration>> + Send + 'static,
{
    let (event_tx, event_rx) = mpsc::channel(capacity);
    let sink = ChatDeltaSink::new(event_tx.clone(), CHAT_TERMINAL_SEND_TIMEOUT);

    tokio::spawn(async move {
        let permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                send_chat_terminal_with_timeout(
                    event_tx,
                    ChatStreamEvent::Failed(ApiError::service_unavailable(
                        "Server is shutting down",
                    )),
                    CHAT_TERMINAL_SEND_TIMEOUT,
                )
                .await;
                return;
            }
        };

        if event_tx.send(ChatStreamEvent::Started).await.is_err() {
            return;
        }

        let generation = generation_task(sink.clone());
        let terminal = resolve_chat_terminal(&sink, generation, None).await;
        drop(permit);
        if let Some(terminal) = terminal {
            sink.finish(terminal).await;
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
    G: FnOnce(Arc<ChatDeltaSink>) -> Fut + Send + 'static,
    Fut: Future<Output = izwi_core::Result<ChatGeneration>> + Send + 'static,
    K: Send + 'static,
{
    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let (completion_tx, completion_rx) = oneshot::channel();
    let sink = ChatDeltaSink::new(event_tx.clone(), CHAT_TERMINAL_SEND_TIMEOUT);

    tokio::spawn(async move {
        async move {
            let permit = semaphore.acquire_owned().await.expect("test semaphore");
            if event_tx.send(ChatStreamEvent::Started).await.is_err() {
                return;
            }
            let generation = generation_task(sink.clone());
            let terminal = resolve_chat_terminal(&sink, generation, None).await;
            drop(permit);
            if let Some(terminal) = terminal {
                sink.finish(terminal).await;
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
    use axum::http::StatusCode;
    use izwi_core::ChatRole;
    use izwi_serving_protocol::RejectionCode;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::sync::Semaphore;

    /// The local role's stream notice ends a running stream with a
    /// terminal "shutting down" failure instead of leaving it to be cut.
    #[tokio::test]
    async fn shutdown_notice_ends_a_local_stream_with_a_terminal_error() {
        let (event_tx, _event_rx) = mpsc::channel(4);
        let sink = ChatDeltaSink::new(event_tx, CHAT_TERMINAL_SEND_TIMEOUT);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let terminal = tokio::spawn(async move {
            resolve_chat_terminal(
                &sink,
                std::future::pending::<izwi_core::Result<ChatGeneration>>(),
                Some(shutdown_rx),
            )
            .await
        });
        tokio::task::yield_now().await;
        shutdown_tx.send_replace(true);
        match tokio::time::timeout(Duration::from_secs(5), terminal)
            .await
            .expect("the notice ends the stream")
            .unwrap()
        {
            Some(ChatStreamEvent::Failed(error)) => {
                assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
                assert!(error.message.contains("shutting down"), "{}", error.message);
            }
            other => panic!("expected a shutting-down failure, got {other:?}"),
        }
    }

    #[test]
    fn worker_usage_maps_cached_tokens_into_the_generation() {
        let measured = worker_chat_generation(
            "text".to_string(),
            WorkerFinishReason::Stop,
            Some(Usage {
                input_tokens: 12,
                output_tokens: 3,
                cached_tokens: Some(7),
            }),
            std::time::Instant::now(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(measured.prompt_tokens, 12);
        assert_eq!(measured.tokens_generated, 3);
        assert_eq!(measured.cached_prompt_tokens, Some(7));

        let unmeasured = worker_chat_generation(
            "text".to_string(),
            WorkerFinishReason::Stop,
            None,
            std::time::Instant::now(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(unmeasured.prompt_tokens, 0);
        assert_eq!(
            unmeasured.cached_prompt_tokens, None,
            "unmeasured cache stays absent so the public surface renders zero"
        );
    }

    /// D8 over the wire: a worker's context overflow is the same OpenAI
    /// `context_length_exceeded` 400 the local server returns, and codes or
    /// finish reasons from a newer worker never read as success.
    #[test]
    fn worker_context_overflow_and_newer_codes_map_like_the_local_server() {
        let rejected = |code| {
            map_worker_client_error(WorkerClientError::Rejected {
                rejection: izwi_serving_protocol::InvocationRejection::new(
                    RequestId::new("request-1").unwrap(),
                    AttemptId::new("attempt-1").unwrap(),
                    code,
                    "This model's maximum context length is 32 tokens.",
                ),
            })
        };
        let overflow = rejected(RejectionCode::ContextLengthExceeded);
        let local = chat_error(izwi_core::Error::ContextLengthExceeded(
            "This model's maximum context length is 32 tokens.".into(),
        ));
        assert_eq!(overflow.status, local.status);
        assert_eq!(overflow.message, local.message);
        assert_eq!(overflow.detail, local.detail);

        assert_eq!(
            rejected(RejectionCode::Unknown).status,
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            map_worker_terminal_error(InvocationErrorCode::Unknown, "new".into()).status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let generation = worker_chat_generation(
            String::new(),
            WorkerFinishReason::Unknown,
            None,
            std::time::Instant::now(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            generation.finish_reason,
            Some(izwi_core::engine::OutputFinishReason::StopToken)
        );
    }

    #[test]
    fn streamed_remote_logprobs_are_per_delta_and_cumulative_at_the_end() {
        let entry = |token: &str| izwi_serving_protocol::TokenLogprob {
            token: token.into(),
            logprob: -0.5,
            bytes: token.as_bytes().to_vec(),
            top_logprobs: Vec::new(),
        };
        let tokens = |entries: &[izwi_core::engine::TokenLogprob]| {
            entries
                .iter()
                .map(|entry| entry.token.clone())
                .collect::<Vec<_>>()
        };
        let mut collector = RemoteLogprobCollector::default();
        assert_eq!(
            tokens(&collector.extend(Some(vec![entry("a"), entry("b")]))),
            ["a", "b"]
        );
        assert_eq!(tokens(&collector.extend(Some(vec![entry("c")]))), ["c"]);
        assert!(collector.extend(None).is_empty());
        assert_eq!(tokens(&collector.finish()), ["a", "b", "c"]);

        let mut full = RemoteLogprobCollector::default();
        full.extend(Some(vec![entry("x"); REMOTE_CHAT_LOGPROB_ENTRY_LIMIT - 1]));
        assert_eq!(
            tokens(&full.extend(Some(vec![entry("y"), entry("z")]))),
            ["y"]
        );
        assert!(full.extend(Some(vec![entry("w")])).is_empty());
        assert_eq!(full.finish().len(), REMOTE_CHAT_LOGPROB_ENTRY_LIMIT);
    }

    #[test]
    fn explicit_overrides_win_over_default_generation_params() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Gemma34BIt,
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
            seed: Some(7),
            stop: vec!["END".into()],
            logprobs: None,
            top_logprobs: None,
            response_format_json_object: false,
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
        assert_eq!(params.seed, Some(7));
        assert_eq!(params.stop_sequences, ["END"]);
    }

    /// Sampling and output controls and tools travel to the worker (each
    /// gated by a feature); reasoning controls still cannot, and are refused.
    #[test]
    fn remote_chat_scope_refuses_only_what_the_protocol_cannot_carry() {
        let mut request = qwen_request(ModelVariant::Gemma34BIt, None);
        request.seed = Some(7);
        request.top_k = Some(9);
        request.logprobs = Some(true);
        request.response_format_json_object = true;
        request.chat_config.tools = vec![serde_json::json!({"type": "function"})];
        validate_remote_chat_scope(&request).expect("sampling controls and tools are in scope");
        assert_eq!(
            request.worker_chat_parameters(&BTreeSet::new()).tools,
            request.chat_config.tools
        );
        assert!(request
            .required_worker_features()
            .contains(&WorkerFeature::ChatTools));
        let thinking = qwen_request(ModelVariant::Gemma34BIt, Some(true));
        let error = validate_remote_chat_scope(&thinking).expect_err("thinking is refused");
        assert!(
            error.message.contains("`enable_thinking`"),
            "{}",
            error.message
        );
    }

    /// The D8 policy carries the caller's own budget (none when unset) and
    /// truncation, and goes only to a worker that takes it.
    #[test]
    fn the_context_policy_carries_the_callers_budget() {
        let features = BTreeSet::from([WorkerFeature::ChatContextPolicy]);
        let mut request = qwen_request(ModelVariant::Gemma34BIt, None);
        request.max_tokens = None;
        request.chat_config.truncation = ChatTruncation::Auto;
        assert_eq!(
            request.worker_chat_parameters(&features).context,
            Some(ChatContextPolicy {
                max_tokens: None,
                truncation: WorkerChatTruncation::Auto,
            })
        );
        request.max_completion_tokens = Some(5_000_000_000);
        assert_eq!(
            request
                .worker_chat_parameters(&features)
                .context
                .and_then(|context| context.max_tokens),
            Some(u32::MAX)
        );
        assert_eq!(
            request.worker_chat_parameters(&BTreeSet::new()).context,
            None
        );
        assert!(request.required_worker_features().is_empty());
    }

    fn qwen_request(variant: ModelVariant, enable_thinking: Option<bool>) -> ChatExecutionRequest {
        ChatExecutionRequest {
            variant,
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
            seed: None,
            stop: Vec::new(),
            logprobs: None,
            top_logprobs: None,
            response_format_json_object: false,
            chat_config: ChatRequestConfig {
                enable_thinking,
                ..Default::default()
            },
            correlation_id: None,
        }
    }

    /// Pins the sampling every chat model resolves to when the request sets
    /// none, per thinking mode: `(temperature, top_p, top_k,
    /// repetition_penalty, presence_penalty)`. Models not listed use the
    /// engine default.
    #[test]
    fn every_chat_model_resolves_its_pinned_default_sampling() {
        type Sampling = (f32, f32, usize, f32, f32);
        let profiled: [(ModelVariant, [Sampling; 3]); 8] = [
            (
                ModelVariant::Qwen3827BFp8,
                [
                    (1.0, 0.95, 20, 1.0, 0.0),
                    (1.0, 0.95, 20, 1.0, 0.0),
                    (0.7, 0.80, 20, 1.0, 1.5),
                ],
            ),
            (
                ModelVariant::Qwen36Moe35BA3BFp8,
                [
                    (1.0, 0.95, 20, 1.0, 0.0),
                    (1.0, 0.95, 20, 1.0, 0.0),
                    (0.7, 0.80, 20, 1.0, 1.5),
                ],
            ),
            (
                ModelVariant::Qwen354BGguf,
                [
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (0.7, 0.80, 20, 1.0, 1.5),
                ],
            ),
            (
                ModelVariant::Qwen359BGguf,
                [
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (0.7, 0.80, 20, 1.0, 1.5),
                ],
            ),
            (
                ModelVariant::Qwen3508BGguf,
                [
                    (1.0, 1.0, 20, 1.0, 2.0),
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (1.0, 1.0, 20, 1.0, 2.0),
                ],
            ),
            (
                ModelVariant::Qwen352BGguf,
                [
                    (1.0, 1.0, 20, 1.0, 2.0),
                    (1.0, 0.95, 20, 1.0, 1.5),
                    (1.0, 1.0, 20, 1.0, 2.0),
                ],
            ),
            (
                ModelVariant::Lfm2512BInstructGguf,
                [(0.1, 1.0, 50, 1.05, 0.0); 3],
            ),
            (
                ModelVariant::Lfm2512BThinkingGguf,
                [(0.05, 1.0, 50, 1.05, 0.0); 3],
            ),
        ];
        let engine_default = (0.7, 0.9, 0, 1.1, 0.0);
        let mut chat_models = 0;
        for &variant in ModelVariant::all().iter().filter(|v| v.is_chat()) {
            chat_models += 1;
            let expected = profiled
                .iter()
                .find(|(profiled, _)| *profiled == variant)
                .map_or([engine_default; 3], |(_, modes)| *modes);
            for (mode, enable_thinking) in [None, Some(true), Some(false)].into_iter().enumerate() {
                let params = qwen_request(variant, enable_thinking).resolved_generation_params();
                let resolved = (
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    params.repetition_penalty,
                    params.presence_penalty,
                );
                assert_eq!(
                    resolved, expected[mode],
                    "{variant:?} enable_thinking={enable_thinking:?}"
                );
                assert_eq!(params.max_tokens, 32, "{variant:?}");
            }
        }
        assert!(
            profiled.iter().all(|(variant, _)| variant.is_chat()),
            "every profiled model is a chat model"
        );
        assert!(
            chat_models > profiled.len(),
            "unprofiled chat models are covered"
        );
    }

    /// EDGE-F19: boundary values are accepted and values just outside them
    /// are refused with the parameter named.
    #[test]
    fn sampling_ranges_accept_their_bounds_and_name_what_they_refuse() {
        let request = |apply: fn(&mut ChatExecutionRequest)| {
            let mut request = qwen_request(ModelVariant::Gemma34BIt, None);
            apply(&mut request);
            validate_chat_request_ranges(&request).map_err(|error| error.detail)
        };
        let refused = |param| {
            Err(Some(crate::error::ApiErrorDetail {
                param: Some(param),
                code: Some("invalid_value"),
            }))
        };
        for apply in [
            (|r: &mut ChatExecutionRequest| r.temperature = Some(0.0)) as fn(&mut _),
            |r| r.temperature = Some(2.0),
            |r| r.top_p = Some(1.0),
            |r| r.top_p = Some(f32::MIN_POSITIVE),
            |r| r.repetition_penalty = Some(1.0),
            |r| r.repetition_penalty = Some(2.0),
            |r| r.presence_penalty = Some(-2.0),
            |r| r.presence_penalty = Some(2.0),
            |r| r.max_tokens = Some(1),
            |r| r.max_completion_tokens = Some(1),
        ] {
            assert_eq!(request(apply), Ok(()));
        }
        assert_eq!(
            request(|r| r.temperature = Some(2.01)),
            refused("temperature")
        );
        assert_eq!(
            request(|r| r.temperature = Some(f32::NAN)),
            refused("temperature")
        );
        assert_eq!(request(|r| r.top_p = Some(0.0)), refused("top_p"));
        assert_eq!(request(|r| r.top_p = Some(f32::NAN)), refused("top_p"));
        assert_eq!(
            request(|r| r.repetition_penalty = Some(0.99)),
            refused("repetition_penalty")
        );
        assert_eq!(
            request(|r| r.presence_penalty = Some(-2.01)),
            refused("presence_penalty")
        );
        assert_eq!(request(|r| r.max_tokens = Some(0)), refused("max_tokens"));
        assert_eq!(
            request(|r| r.max_completion_tokens = Some(0)),
            refused("max_completion_tokens")
        );
    }

    #[test]
    fn top_k_accepts_the_vllm_disabled_value() {
        assert_eq!(chat_top_k(None).unwrap(), None);
        assert_eq!(chat_top_k(Some(-1)).unwrap(), Some(0));
        assert_eq!(chat_top_k(Some(0)).unwrap(), Some(0));
        assert_eq!(chat_top_k(Some(40)).unwrap(), Some(40));
        let error = chat_top_k(Some(-2)).unwrap_err();
        assert_eq!(error.detail.and_then(|detail| detail.param), Some("top_k"));
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
            None,
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
            Some(ChatTruncation::Auto),
        )
        .expect("matching fields should resolve");

        assert_eq!(config.enable_thinking, Some(true));
        assert_eq!(config.reasoning_effort, Some(ChatReasoningEffort::Low));
        assert_eq!(config.preserve_thinking, Some(false));
        assert_eq!(config.truncation, ChatTruncation::Auto);
    }

    #[tokio::test]
    async fn streaming_chat_allows_long_running_generations_to_complete() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx = spawn_chat_stream_with_task(semaphore, 4, |sink| async move {
            sink.offer("Hello".to_string(), Vec::new());
            tokio::time::sleep(Duration::from_millis(25)).await;
            sink.offer(" world".to_string(), Vec::new());
            Ok(ChatGeneration {
                text: "Hello world".to_string(),
                prompt_tokens: 12,
                tokens_generated: 2,
                generation_time_ms: 25.0,
                latency_breakdown: None,
                finish_reason: None,
                cached_prompt_tokens: None,
                logprobs: Vec::new(),
            })
        });

        match event_rx.recv().await {
            Some(ChatStreamEvent::Started) => {}
            other => panic!("expected stream start event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta { text, .. }) => assert_eq!(text, "Hello"),
            other => panic!("expected first delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta { text, .. }) => assert_eq!(text, " world"),
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

    /// Drains a chat stream: its concatenated delta text, the number of
    /// delta events, and the event that ended it (None if it just closed).
    async fn drain_chat_stream(
        event_rx: &mut mpsc::Receiver<ChatStreamEvent>,
    ) -> (String, usize, Option<ChatStreamEvent>) {
        let (mut text, mut deltas) = (String::new(), 0);
        while let Some(event) = event_rx.recv().await {
            match event {
                ChatStreamEvent::Started => {}
                ChatStreamEvent::Delta { text: delta, .. } => {
                    text.push_str(&delta);
                    deltas += 1;
                }
                terminal => return (text, deltas, Some(terminal)),
            }
        }
        (text, deltas, None)
    }

    #[tokio::test]
    async fn paused_chat_reader_receives_coalesced_text_without_cancellation() {
        let semaphore = Arc::new(Semaphore::new(1));
        let expected = (0..100)
            .map(|index| format!("t{index} "))
            .collect::<String>();
        let mut event_rx = spawn_chat_stream_with_task(semaphore, 2, |sink| async move {
            for index in 0..100 {
                sink.offer(format!("t{index} "), Vec::new());
            }
            // Completion in the same poll as the burst must still deliver
            // the merged text ahead of the terminal event.
            Ok(ChatGeneration {
                text: (0..100).map(|index| format!("t{index} ")).collect(),
                prompt_tokens: 1,
                tokens_generated: 100,
                generation_time_ms: 1.0,
                latency_breakdown: None,
                finish_reason: None,
                cached_prompt_tokens: None,
                logprobs: Vec::new(),
            })
        });

        // Pause for less than the slow-consumer timeout, then read.
        tokio::time::sleep(CHAT_TERMINAL_SEND_TIMEOUT / 3).await;
        let (text, deltas, terminal) = drain_chat_stream(&mut event_rx).await;
        assert_eq!(text, expected);
        assert!(
            deltas < 100,
            "a lagging reader gets merged deltas: {deltas}"
        );
        assert!(
            matches!(&terminal, Some(ChatStreamEvent::Completed(generation)) if generation.text == expected),
            "{terminal:?}"
        );
    }

    #[tokio::test]
    async fn chat_reader_stalled_past_the_timeout_cancels_generation() {
        let semaphore = Arc::new(Semaphore::new(1));
        let generation_dropped = Arc::new(AtomicBool::new(false));
        let dropped = generation_dropped.clone();
        struct GenerationDrop(Arc<AtomicBool>);
        impl Drop for GenerationDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let mut event_rx = spawn_chat_stream_with_task(semaphore.clone(), 2, |sink| async move {
            let _drop = GenerationDrop(dropped);
            for index in 0..10 {
                sink.offer(format!("t{index} "), Vec::new());
            }
            std::future::pending::<()>().await;
            unreachable!("a stalled reader cancels the generation")
        });

        tokio::time::sleep(CHAT_TERMINAL_SEND_TIMEOUT * 3).await;
        assert!(generation_dropped.load(Ordering::Acquire));
        assert_eq!(semaphore.available_permits(), 1);
        // The stalled reader gets no completion: at most the explicit
        // failure, otherwise a closed channel the SSE encoder reports.
        match drain_chat_stream(&mut event_rx).await {
            (_, _, None) => {}
            (_, _, Some(ChatStreamEvent::Failed(error))) => {
                assert_eq!(error.message, CHAT_STREAM_BACKPRESSURE_ERROR)
            }
            (_, _, other) => panic!("unexpected terminal {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_draining_chat_consumer_cannot_retain_workload_capacity() {
        let semaphore = Arc::new(Semaphore::new(1));
        let generation_started = Arc::new(Notify::new());
        let release_generation = Arc::new(Notify::new());
        let started = generation_started.clone();
        let release = release_generation.clone();
        let mut event_rx =
            spawn_chat_stream_with_task(semaphore.clone(), 1, move |_sink| async move {
                started.notify_one();
                release.notified().await;
                Ok(ChatGeneration {
                    text: "done".to_string(),
                    prompt_tokens: 1,
                    tokens_generated: 1,
                    generation_time_ms: 1.0,
                    latency_breakdown: None,
                    finish_reason: None,
                    cached_prompt_tokens: None,
                    logprobs: Vec::new(),
                })
            });

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
            move |_sink| async move {
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

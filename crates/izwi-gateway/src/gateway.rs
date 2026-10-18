//! Hardware-independent public gateway state and routes.
//!
//! This module deliberately has no `RuntimeService`, persistence, model, or
//! accelerator initialization path. Its only execution dependency is the
//! worker registry behind `RemoteChatDispatcher`.

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_hooks::{
    EnterpriseAction, EnterpriseHooks, HookMetadata, QuotaRequest, ResourceDescriptor,
};
use izwi_model_catalog::ChatGeneration;
use izwi_prometheus::{buckets, CounterFamily, HistogramFamily};
use serde::Serialize;
use std::fmt::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{field, info, info_span, warn, Span};

use crate::api::openai::compat::OpenAiCompatibilityProfile;
use crate::api::request_context::RequestContext;
use crate::api::request_context::{
    attach_gateway_request_context, attach_gateway_request_id, RequestTrace,
};
use crate::app::chat::{ChatBackend, ChatExecutionRequest, ChatStreamEvent, ChatStreamObserver};
use crate::app::remote_chat_dispatch::RemoteChatDispatcher;
use crate::config::GatewayServeConfig;
use crate::error::ApiError;
use crate::gateway_rate_quota::{
    GatewayRateDecision, GatewayRateQuota, GatewayRateQuotaConfig, RateLimitSnapshot,
    TokenDecision, TokenReservation,
};
use crate::gateway_security::GatewayPerimeterConfig;
use crate::gateway_tenant_concurrency::{
    GatewayTenantConcurrency, GatewayTenantConcurrencyConfig, TenantWorkAdmissionError,
    UnboundTenantWorkLease,
};
use crate::lifecycle::ServerLifecycle;
use crate::logging::{service_name, SERVICE_VERSION};

const CHAT_COMPLETIONS_RESOURCE: &str = "/v1/chat/completions";
const MAX_GATEWAY_QUOTA_HOOK_TIME: Duration = Duration::from_secs(2);
/// Default deadline for reading a chat request body.
pub const DEFAULT_GATEWAY_BODY_READ_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct GatewayState {
    /// One dispatcher per public chat model, the primary model first.
    pub chat_dispatchers: Vec<RemoteChatDispatcher>,
    pub enterprise_hooks: EnterpriseHooks,
    pub lifecycle: ServerLifecycle,
    pub perimeter: GatewayPerimeterConfig,
    pub principal_keys: crate::gateway_principal_keys::GatewayPrincipalDirectory,
    pub request_timeout_secs: u64,
    request_admission: Arc<Semaphore>,
    rate_quota: GatewayRateQuota,
    tenant_concurrency: GatewayTenantConcurrency,
    max_output_tokens: u32,
    /// Deadline for reading a request body, so a slow upload cannot hold a
    /// global admission permit for long.
    body_read_timeout: Duration,
    metrics: GatewayMetrics,
    /// Realtime relay surface (DS3.6). `None` keeps the gateway
    /// byte-identical to the pre-realtime shape.
    pub realtime_relay: Option<std::sync::Arc<crate::app::realtime_relay::GatewayRealtimeRelay>>,
    /// The adopted approvals view, for the admin status route.
    approvals: Option<crate::gateway_shared_approvals::GatewayApprovalsReporter>,
    /// Set during the pre-stop window: readiness reports draining while
    /// admission still serves, so load balancers move traffic away first.
    prestop: Arc<std::sync::atomic::AtomicBool>,
    /// Ends every live chat stream with a terminal event.
    stream_shutdown: Arc<tokio::sync::watch::Sender<bool>>,
    /// The OpenAI compatibility profile chat requests are checked and
    /// answered under (`IZWI_OPENAI_COMPAT_PROFILE`).
    pub compat_profile: OpenAiCompatibilityProfile,
}

impl GatewayState {
    pub fn new(
        dispatcher: RemoteChatDispatcher,
        enterprise_hooks: EnterpriseHooks,
        perimeter: GatewayPerimeterConfig,
        request_timeout_secs: u64,
        max_in_flight: usize,
    ) -> Self {
        debug_assert!(max_in_flight > 0);
        let max_output_tokens = dispatcher.max_output_tokens();
        let tenant_concurrency_config =
            GatewayTenantConcurrencyConfig::new(8_usize.min(max_in_flight), max_in_flight)
                .expect("validated gateway admission limit supports tenant ownership");
        let stream_shutdown = Arc::new(tokio::sync::watch::Sender::new(false));
        Self {
            chat_dispatchers: vec![dispatcher.with_shutdown(stream_shutdown.subscribe())],
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            perimeter,
            principal_keys: crate::gateway_principal_keys::GatewayPrincipalDirectory::empty(),
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
            rate_quota: GatewayRateQuota::new(GatewayRateQuotaConfig::default()),
            tenant_concurrency: GatewayTenantConcurrency::new(tenant_concurrency_config),
            max_output_tokens,
            body_read_timeout: DEFAULT_GATEWAY_BODY_READ_TIMEOUT,
            metrics: GatewayMetrics::default(),
            realtime_relay: None,
            approvals: None,
            prestop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stream_shutdown,
            compat_profile: OpenAiCompatibilityProfile::Strict,
        }
    }

    /// Check and answer chat requests under `profile` (strict by default).
    pub fn with_compat_profile(mut self, profile: OpenAiCompatibilityProfile) -> Self {
        self.compat_profile = profile;
        self
    }

    pub fn with_body_read_timeout(mut self, timeout: Duration) -> Self {
        self.body_read_timeout = timeout;
        self
    }

    /// Reads and decodes a chat request body within the body-read deadline
    /// and the configured size limit.
    pub async fn read_chat_body(
        &self,
        headers: &HeaderMap,
        body: axum::body::Body,
    ) -> Result<crate::api::openai::chat::dto::ChatCompletionRequest, ApiError> {
        let not_json = || ApiError::bad_request("Gateway chat request body is not valid JSON");
        let is_json = headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .is_some_and(|mime| {
                let mime = mime.trim();
                mime.eq_ignore_ascii_case("application/json")
                    || mime.to_ascii_lowercase().ends_with("+json")
            });
        if !is_json {
            return Err(not_json());
        }
        let limit = self.perimeter.max_chat_body_bytes();
        let bytes = tokio::time::timeout(self.body_read_timeout, axum::body::to_bytes(body, limit))
            .await
            .map_err(|_| {
                self.metrics
                    .inner
                    .body_timeout_rejections
                    .fetch_add(1, Ordering::Relaxed);
                ApiError::of(
                    crate::error::ErrorClass::Timeout,
                    "Gateway chat request body was not received in time",
                )
            })?
            .map_err(|error| {
                let too_large = std::error::Error::source(&error)
                    .is_some_and(|source| source.is::<http_body_util::LengthLimitError>());
                if too_large {
                    ApiError::payload_too_large(
                        "Gateway chat request body exceeds the configured limit",
                    )
                } else {
                    not_json()
                }
            })?;
        serde_json::from_slice(&bytes).map_err(|_| not_json())
    }

    /// Serve one more public chat model. Requests route by their `model`.
    pub fn with_chat_model(mut self, dispatcher: RemoteChatDispatcher) -> Self {
        self.chat_dispatchers
            .push(dispatcher.with_shutdown(self.stream_shutdown.subscribe()));
        self
    }

    /// The dispatcher serving public model `model`, or the OpenAI 404 for a
    /// model this gateway does not serve.
    pub fn chat_dispatcher_for(&self, model: &str) -> Result<&RemoteChatDispatcher, ApiError> {
        let variant = crate::app::chat::parse_chat_model(model).ok();
        self.chat_dispatchers
            .iter()
            .find(|dispatcher| Some(dispatcher.public_model()) == variant)
            .ok_or_else(|| ApiError::model_not_found(format!("The model `{model}` does not exist")))
    }

    /// Install the adopted-approvals reporter behind
    /// `GET /internal/admin/approvals`.
    pub fn with_approvals_reporter(
        mut self,
        reporter: crate::gateway_shared_approvals::GatewayApprovalsReporter,
    ) -> Self {
        self.approvals = Some(reporter);
        self
    }

    /// Override process-local request-rate policy and the exact private worker
    /// output ceiling. This remains rate-only: response-owned global admission
    /// and teardown-aware tenant concurrent-work ownership are separate gates.
    pub fn with_rate_quota_config(mut self, config: GatewayRateQuotaConfig) -> Self {
        self.rate_quota = GatewayRateQuota::new(config);
        self
    }

    pub fn with_tenant_concurrency_config(
        mut self,
        config: GatewayTenantConcurrencyConfig,
    ) -> Self {
        self.tenant_concurrency = GatewayTenantConcurrency::new(config);
        self
    }

    /// Install the scoped per-principal key directory (DS0.5). The default is
    /// an empty directory: authentication then behaves exactly as before the
    /// scoped-key feature existed.
    /// Install the realtime relay surface (DS3.6). Absent keeps the public
    /// surface unchanged.
    pub fn with_realtime_relay(
        mut self,
        relay: crate::app::realtime_relay::GatewayRealtimeRelay,
    ) -> Self {
        self.realtime_relay = Some(std::sync::Arc::new(relay));
        self
    }

    pub fn with_principal_keys(
        mut self,
        principal_keys: crate::gateway_principal_keys::GatewayPrincipalDirectory,
    ) -> Self {
        self.principal_keys = principal_keys;
        self
    }

    /// The tenant's request rate, then its token rate, then the enterprise
    /// quota hook: local limits are checked first so a request they refuse
    /// is never charged by the hook. Returns the meter that settles the
    /// request's tokens and reports its usage when it completes.
    pub async fn enforce_chat_rate_quota(
        &self,
        context: &crate::api::request_context::RequestContext,
        request: &ChatExecutionRequest,
    ) -> Result<ChatUsageMeter, ApiError> {
        let requested_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(usize::MAX)
            .max(1);
        let estimated_units = u32::try_from(requested_tokens)
            .unwrap_or(u32::MAX)
            .min(self.max_output_tokens)
            .max(1);
        let tenant_key = context.tenant_key().ok_or_else(|| {
            ApiError::service_unavailable("Gateway tenant identity is unavailable")
        })?;
        match self.rate_quota.check(tenant_key) {
            Ok(GatewayRateDecision::Allowed) => {}
            Ok(GatewayRateDecision::Limited) => {
                self.metrics
                    .inner
                    .quota_rejections
                    .fetch_add(1, Ordering::Relaxed);
                return Err(ApiError::too_many_requests(
                    "Gateway tenant rate limit exceeded",
                ));
            }
            Err(_) => {
                return Err(ApiError::service_unavailable(
                    "Gateway tenant rate limiter unavailable",
                ))
            }
        }
        // Prompt tokens are estimated at four bytes a token; usage settles
        // the reservation to the worker's exact count.
        let prompt_bytes = request
            .messages
            .iter()
            .map(|message| message.content.len() as u64)
            .sum::<u64>();
        let token_estimate = u64::from(estimated_units).saturating_add(prompt_bytes.div_ceil(4));
        let tokens = match self.rate_quota.reserve_tokens(tenant_key, token_estimate) {
            Ok((TokenDecision::Allowed, reservation)) => reservation,
            Ok((TokenDecision::Limited { .. }, _)) => {
                self.metrics
                    .inner
                    .quota_rejections
                    .fetch_add(1, Ordering::Relaxed);
                return Err(ApiError::too_many_requests(
                    "Gateway tenant token rate limit exceeded",
                ));
            }
            Err(_) => {
                return Err(ApiError::service_unavailable(
                    "Gateway tenant rate limiter unavailable",
                ))
            }
        };

        let hook_budget = context
            .remaining_budget(Duration::from_secs(self.request_timeout_secs))
            .unwrap_or(Duration::ZERO)
            .min(MAX_GATEWAY_QUOTA_HOOK_TIME);
        if hook_budget.is_zero() {
            return Err(ApiError::service_unavailable(
                "Gateway quota service unavailable",
            ));
        }
        let quota_request = QuotaRequest {
            principal: Some(context.principal.clone()),
            action: EnterpriseAction::Inference,
            resource: ResourceDescriptor::http_route(CHAT_COMPLETIONS_RESOURCE),
            estimated_units: Some(u64::from(estimated_units)),
            metadata: HookMetadata::new(),
        };
        let evaluation = self.enterprise_hooks.quotas.evaluate(&quota_request);
        let decision = tokio::time::timeout(hook_budget, evaluation)
            .await
            .map_err(|_| {
                warn!("Enterprise gateway quota hook timed out");
                ApiError::service_unavailable("Gateway quota service unavailable")
            })?
            .map_err(|_| {
                warn!("Enterprise gateway quota hook failed");
                ApiError::service_unavailable("Gateway quota service unavailable")
            })?;
        if !decision.allowed {
            self.metrics
                .inner
                .quota_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::too_many_requests(
                "Gateway tenant rate limit exceeded",
            ));
        }
        Ok(ChatUsageMeter {
            quotas: Arc::clone(&self.enterprise_hooks.quotas),
            principal: context.principal.clone(),
            correlation_id: context.correlation_id.clone(),
            model: request.variant.dir_name(),
            estimated_units: u64::from(estimated_units),
            tokens,
        })
    }

    pub fn begin_tenant_work(
        &self,
        context: &crate::api::request_context::RequestContext,
    ) -> Result<UnboundTenantWorkLease, ApiError> {
        let tenant_key = context.tenant_key().ok_or_else(|| {
            ApiError::service_unavailable("Gateway tenant identity is unavailable")
        })?;
        match self.tenant_concurrency.try_reserve(tenant_key) {
            Ok(lease) => Ok(lease),
            Err(TenantWorkAdmissionError::TenantLimit) => {
                self.metrics
                    .inner
                    .tenant_concurrency_rejections
                    .fetch_add(1, Ordering::Relaxed);
                Err(ApiError::too_many_requests(
                    "Gateway tenant concurrent-work limit exceeded",
                ))
            }
            Err(
                TenantWorkAdmissionError::OwnershipCapacity
                | TenantWorkAdmissionError::StateUnavailable,
            ) => Err(ApiError::service_unavailable(
                "Gateway concurrent-work ownership capacity is unavailable",
            )),
        }
    }

    pub fn mark_ready(&self) {
        self.lifecycle.mark_ready();
    }

    /// Starts the pre-stop window: readiness turns unready, requests are
    /// still admitted.
    pub fn begin_prestop(&self) {
        self.prestop.store(true, Ordering::Release);
    }

    /// Ends every live chat stream with a terminal "shutting down" event.
    pub fn end_streams(&self) {
        self.stream_shutdown.send_replace(true);
    }

    pub fn begin_drain(&self) {
        // Publishing the lifecycle transition before closing admission avoids
        // a gap where a newly acquired permit could pass the post-acquire
        // lifecycle check. Existing response-owned permits remain valid.
        self.lifecycle.mark_draining();
        self.request_admission.close();
    }
}

/// Settles one chat request's usage: the token reservation against what the
/// worker reported, and the enterprise quota hook's usage commit.
pub struct ChatUsageMeter {
    quotas: Arc<dyn izwi_hooks::QuotaLimiter>,
    principal: izwi_hooks::Principal,
    correlation_id: String,
    model: &'static str,
    estimated_units: u64,
    tokens: Option<TokenReservation>,
}

impl ChatUsageMeter {
    /// Output is flowing: from now on, an unsettled reservation stands.
    fn started(&mut self) {
        if let Some(tokens) = self.tokens.as_mut() {
            tokens.mark_started();
        }
    }

    fn complete(self, generation: &ChatGeneration) {
        let input_units = generation.prompt_tokens as u64;
        let output_units = generation.tokens_generated as u64;
        if let Some(tokens) = self.tokens {
            tokens.settle(input_units.saturating_add(output_units));
        }
        let usage = izwi_hooks::QuotaUsage {
            principal: Some(self.principal),
            action: EnterpriseAction::Inference,
            resource: ResourceDescriptor::http_route(CHAT_COMPLETIONS_RESOURCE),
            estimated_units: Some(self.estimated_units),
            input_units,
            output_units,
            metadata: HookMetadata::from([
                ("correlation_id".to_string(), self.correlation_id),
                ("model".to_string(), self.model.to_string()),
            ]),
        };
        let quotas = self.quotas;
        tokio::spawn(async move {
            match tokio::time::timeout(MAX_GATEWAY_QUOTA_HOOK_TIME, quotas.commit(&usage)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => warn!("Enterprise gateway quota usage commit failed"),
                Err(_) => warn!("Enterprise gateway quota usage commit timed out"),
            }
        });
    }
}

/// The tenant concurrent-work lease taken before global admission, handed
/// to the chat handler through the request extensions.
#[derive(Clone)]
pub struct PreAdmittedTenantWork(Arc<std::sync::Mutex<Option<UnboundTenantWorkLease>>>);

impl PreAdmittedTenantWork {
    pub fn take(&self) -> Result<UnboundTenantWorkLease, ApiError> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| ApiError::service_unavailable("Gateway tenant admission is unavailable"))
    }
}

#[derive(Clone)]
pub struct GatewayAdmissionGuard {
    _permit: Arc<OwnedSemaphorePermit>,
    _active_request: Arc<GatewayActiveRequestGuard>,
}

/// Chat through the worker registry for one admitted public request. The
/// backend owns the request's global admission and tenant lease and counts
/// its dispatch in the gateway metrics.
pub struct GatewayChat<'a> {
    pub state: &'a GatewayState,
    pub dispatcher: &'a RemoteChatDispatcher,
    pub context: &'a RequestContext,
    pub admission: GatewayAdmissionGuard,
    pub tenant_work: UnboundTenantWorkLease,
    pub usage: ChatUsageMeter,
}

/// Holds a streaming response's admission and stream metrics until its body
/// ends.
pub struct GatewayChatStream {
    _admission: GatewayAdmissionGuard,
    metrics: GatewayStreamMetricsGuard,
    usage: Option<ChatUsageMeter>,
}

impl ChatStreamObserver for GatewayChatStream {
    fn record_delta(&mut self) {
        self.metrics.record_delta();
    }

    fn record_completion(&mut self, generation: &ChatGeneration) {
        self.metrics.record_completion(generation);
        if let Some(usage) = self.usage.take() {
            usage.complete(generation);
        }
    }

    fn record_failure(&mut self) {
        self.metrics.record_failure();
    }
}

impl ChatBackend for GatewayChat<'_> {
    type StreamGuard = GatewayChatStream;

    async fn generate(self, request: ChatExecutionRequest) -> Result<ChatGeneration, ApiError> {
        let started = Instant::now();
        let model = request.variant.dir_name();
        let generation = self
            .dispatcher
            .generate(
                self.state.request_timeout_secs,
                self.context,
                request,
                self.tenant_work,
            )
            .await;
        let metrics = &self.state.metrics;
        metrics.record_dispatch(started.elapsed(), generation.as_ref().err());
        let chat = &metrics.inner.chat;
        let elapsed = self.context.received_at().elapsed();
        match &generation {
            Ok(generation) => {
                chat.duration
                    .observe_duration(&[model, "false", "ok"], elapsed);
                chat.record_tokens(model, generation);
                self.usage.complete(generation);
            }
            Err(_) => chat
                .duration
                .observe_duration(&[model, "false", "error"], elapsed),
        }
        generation
    }

    async fn stream(
        self,
        request: ChatExecutionRequest,
    ) -> Result<(mpsc::Receiver<ChatStreamEvent>, GatewayChatStream), ApiError> {
        let started = Instant::now();
        let model = request.variant.dir_name();
        let events = self
            .dispatcher
            .stream(
                self.state.request_timeout_secs,
                self.context,
                request,
                self.tenant_work,
            )
            .await;
        self.state
            .metrics
            .record_dispatch(started.elapsed(), events.as_ref().err());
        let events = match events {
            Ok(events) => events,
            Err(error) => {
                self.state.metrics.inner.chat.duration.observe_duration(
                    &[model, "true", "error"],
                    self.context.received_at().elapsed(),
                );
                return Err(error);
            }
        };
        let mut usage = self.usage;
        usage.started();
        Ok((
            events,
            GatewayChatStream {
                _admission: self.admission,
                metrics: self
                    .state
                    .metrics
                    .begin_stream(model, self.context.received_at()),
                usage: Some(usage),
            },
        ))
    }
}

/// Every family has a bounded label set (labelled families keep at most
/// [`GATEWAY_MAX_SERIES`] series), so the exposition stays under this size.
const MAX_PROMETHEUS_RESPONSE_BYTES: usize = 256 * 1024;
const GATEWAY_MAX_SERIES: usize = 16;

#[derive(Clone, Default)]
struct GatewayMetrics {
    inner: Arc<GatewayMetricCounters>,
}

impl fmt::Debug for GatewayMetrics {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayMetrics")
            .field("cardinality", &"fixed")
            .finish()
    }
}

#[derive(Default)]
struct GatewayMetricCounters {
    active_requests: AtomicU64,
    active_streams: AtomicU64,
    auth_rejections: AtomicU64,
    quota_rejections: AtomicU64,
    tenant_concurrency_rejections: AtomicU64,
    body_limit_rejections: AtomicU64,
    body_timeout_rejections: AtomicU64,
    routing_dispatch_failures: AtomicU64,
    dispatch_calls: AtomicU64,
    dispatch_latency_micros: AtomicU64,
    http_2xx: AtomicU64,
    http_3xx: AtomicU64,
    http_4xx: AtomicU64,
    http_5xx: AtomicU64,
    http_other: AtomicU64,
    chat: GatewayChatMetrics,
}

/// Labelled latency and token metrics. Model labels are the parsed
/// `ModelVariant` names, a closed set, never request text; routes are the
/// gateway's own paths; statuses are HTTP codes. No request or tenant ids.
struct GatewayChatMetrics {
    duration: HistogramFamily,
    ttft: HistogramFamily,
    inter_token: HistogramFamily,
    prompt_tokens: CounterFamily,
    completion_tokens: CounterFamily,
    dispatch: HistogramFamily,
    http: HistogramFamily,
}

impl Default for GatewayChatMetrics {
    fn default() -> Self {
        Self {
            duration: HistogramFamily::new(
                "izwi_gateway_chat_request_duration_seconds",
                "Chat request latency from gateway ingress to the last byte of the answer; _count is requests by model, stream and outcome.",
                &["model", "stream", "outcome"],
                buckets::REQUEST_SECONDS,
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            ttft: HistogramFamily::new(
                "izwi_gateway_chat_ttft_seconds",
                "Streaming chat time from gateway ingress to the first answer text sent.",
                &["model"],
                buckets::FIRST_OUTPUT_SECONDS,
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            inter_token: HistogramFamily::new(
                "izwi_gateway_chat_inter_token_latency_seconds",
                "Gap between successive answer-text deltas of one stream; a worker may coalesce several tokens into one delta.",
                &["model"],
                buckets::INTER_TOKEN_SECONDS,
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            prompt_tokens: CounterFamily::new(
                "izwi_gateway_chat_prompt_tokens_total",
                "Prompt tokens of completed chat requests.",
                &["model"],
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            completion_tokens: CounterFamily::new(
                "izwi_gateway_chat_completion_tokens_total",
                "Completion tokens of completed chat requests.",
                &["model"],
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            dispatch: HistogramFamily::new(
                "izwi_gateway_worker_dispatch_seconds",
                "Worker call latency (streams stop at acceptance) by outcome: accepted, or the HTTP status the gateway answered.",
                &["outcome"],
                buckets::REQUEST_SECONDS,
            )
            .with_max_series(GATEWAY_MAX_SERIES),
            http: HistogramFamily::new(
                "izwi_gateway_http_request_duration_seconds",
                "Time to response headers by gateway route and status.",
                &["route", "status"],
                buckets::REQUEST_SECONDS,
            )
            .with_max_series(GATEWAY_MAX_SERIES),
        }
    }
}

impl GatewayChatMetrics {
    fn record_tokens(&self, model: &str, generation: &ChatGeneration) {
        self.prompt_tokens
            .add(&[model], generation.prompt_tokens as u64);
        self.completion_tokens
            .add(&[model], generation.tokens_generated as u64);
    }

    fn render(&self, output: &mut String) {
        self.duration.render(output);
        self.ttft.render(output);
        self.inter_token.render(output);
        self.prompt_tokens.render(output);
        self.completion_tokens.render(output);
        self.dispatch.render(output);
        self.http.render(output);
    }
}

/// The gateway route label: its own paths, never the caller's.
fn gateway_route_label(path: &str) -> &'static str {
    match path {
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/realtime/ws" => "/v1/realtime/ws",
        "/v1/speech-to-text/realtime/ws" => "/v1/speech-to-text/realtime/ws",
        "/livez" => "/livez",
        "/readyz" => "/readyz",
        "/internal/metrics" | "/internal/metrics/prometheus" => "/internal/metrics",
        "/internal/admin/drain" => "/internal/admin/drain",
        "/internal/admin/approvals" => "/internal/admin/approvals",
        _ => "other",
    }
}

struct GatewayActiveRequestGuard {
    metrics: GatewayMetrics,
}

impl Drop for GatewayActiveRequestGuard {
    fn drop(&mut self) {
        self.metrics
            .inner
            .active_requests
            .fetch_sub(1, Ordering::Relaxed);
    }
}

struct GatewayStreamMetricsGuard {
    metrics: GatewayMetrics,
    terminal_observed: bool,
    model: &'static str,
    received_at: Instant,
    last_delta_at: Option<Instant>,
}

impl GatewayStreamMetricsGuard {
    fn record_delta(&mut self) {
        let now = Instant::now();
        let chat = &self.metrics.inner.chat;
        match self.last_delta_at {
            None => chat
                .ttft
                .observe_duration(&[self.model], now - self.received_at),
            Some(previous) => chat
                .inter_token
                .observe_duration(&[self.model], now - previous),
        }
        self.last_delta_at = Some(now);
    }

    fn record_completion(&mut self, generation: &ChatGeneration) {
        if !self.terminal_observed {
            self.terminal_observed = true;
            self.record_duration("ok");
            self.metrics
                .inner
                .chat
                .record_tokens(self.model, generation);
        }
    }

    fn record_failure(&mut self) {
        if !self.terminal_observed {
            self.metrics
                .inner
                .routing_dispatch_failures
                .fetch_add(1, Ordering::Relaxed);
            self.terminal_observed = true;
            self.record_duration("error");
        }
    }

    fn record_duration(&self, outcome: &str) {
        self.metrics
            .inner
            .chat
            .duration
            .observe_duration(&[self.model, "true", outcome], self.received_at.elapsed());
    }
}

impl Drop for GatewayStreamMetricsGuard {
    fn drop(&mut self) {
        // A body dropped before its terminal event failed.
        self.record_failure();
        self.metrics
            .inner
            .active_streams
            .fetch_sub(1, Ordering::Relaxed);
    }
}

impl GatewayMetrics {
    fn begin_request(&self) -> GatewayActiveRequestGuard {
        self.inner.active_requests.fetch_add(1, Ordering::Relaxed);
        GatewayActiveRequestGuard {
            metrics: self.clone(),
        }
    }

    fn begin_stream(&self, model: &'static str, received_at: Instant) -> GatewayStreamMetricsGuard {
        self.inner.active_streams.fetch_add(1, Ordering::Relaxed);
        GatewayStreamMetricsGuard {
            metrics: self.clone(),
            terminal_observed: false,
            model,
            received_at,
            last_delta_at: None,
        }
    }

    fn record_dispatch(&self, elapsed: Duration, error: Option<&ApiError>) {
        self.inner.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.dispatch_latency_micros.fetch_add(
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if error.is_some() {
            self.inner
                .routing_dispatch_failures
                .fetch_add(1, Ordering::Relaxed);
        }
        let outcome = error.map_or("accepted", |error| error.status.as_str());
        self.inner
            .chat
            .dispatch
            .observe_duration(&[outcome], elapsed);
    }

    fn record_http_response(&self, path: &str, status: StatusCode, elapsed: Duration) {
        self.record_http_outcome(status);
        self.inner
            .chat
            .http
            .observe_duration(&[gateway_route_label(path), status.as_str()], elapsed);
    }

    fn record_http_outcome(&self, status: StatusCode) {
        let counter = match status.as_u16() / 100 {
            2 => &self.inner.http_2xx,
            3 => &self.inner.http_3xx,
            4 => &self.inner.http_4xx,
            5 => &self.inner.http_5xx,
            _ => &self.inner.http_other,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn render_prometheus(&self, active_tenant_work: u64) -> String {
        let snapshot = self.snapshot();
        let mut output = String::with_capacity(8 * 1024);
        macro_rules! metric {
            ($name:literal, $kind:literal, $help:literal, $value:expr) => {{
                let _ = writeln!(output, concat!("# HELP ", $name, " ", $help));
                let _ = writeln!(output, concat!("# TYPE ", $name, " ", $kind));
                let _ = writeln!(output, concat!($name, " {}"), $value);
            }};
        }
        metric!(
            "izwi_gateway_active_requests",
            "gauge",
            "Currently admitted public requests.",
            snapshot.active_requests
        );
        metric!(
            "izwi_gateway_active_streams",
            "gauge",
            "Currently open public streams.",
            snapshot.active_streams
        );
        metric!(
            "izwi_gateway_auth_rejections_total",
            "counter",
            "Public or metrics authentication rejections.",
            snapshot.auth_rejections
        );
        metric!(
            "izwi_gateway_quota_rejections_total",
            "counter",
            "Public tenant quota rejections.",
            snapshot.quota_rejections
        );
        metric!(
            "izwi_gateway_tenant_concurrency_rejections_total",
            "counter",
            "Public tenant concurrent-work admission rejections.",
            snapshot.tenant_concurrency_rejections
        );
        metric!(
            "izwi_gateway_active_tenant_work",
            "gauge",
            "Tenant work retained until confirmed worker teardown.",
            active_tenant_work
        );
        metric!(
            "izwi_gateway_body_limit_rejections_total",
            "counter",
            "Public body-size rejections.",
            snapshot.body_limit_rejections
        );
        metric!(
            "izwi_gateway_body_timeout_rejections_total",
            "counter",
            "Public requests whose body was not received within the body-read deadline.",
            snapshot.body_timeout_rejections
        );
        metric!(
            "izwi_gateway_routing_dispatch_failures_total",
            "counter",
            "Worker routing, dispatch, or accepted-stream failures.",
            snapshot.routing_dispatch_failures
        );
        metric!(
            "izwi_gateway_dispatch_calls_total",
            "counter",
            "Worker dispatch calls observed.",
            snapshot.dispatch_calls
        );
        metric!("izwi_gateway_dispatch_latency_microseconds_total", "counter", "Accumulated worker-call latency; streams stop at acceptance and non-streaming calls at completion.", snapshot.dispatch_latency_micros);
        metric!(
            "izwi_gateway_http_responses_2xx_total",
            "counter",
            "HTTP responses in the 2xx class.",
            snapshot.http_2xx
        );
        metric!(
            "izwi_gateway_http_responses_3xx_total",
            "counter",
            "HTTP responses in the 3xx class.",
            snapshot.http_3xx
        );
        metric!(
            "izwi_gateway_http_responses_4xx_total",
            "counter",
            "HTTP responses in the 4xx class.",
            snapshot.http_4xx
        );
        metric!(
            "izwi_gateway_http_responses_5xx_total",
            "counter",
            "HTTP responses in the 5xx class.",
            snapshot.http_5xx
        );
        metric!(
            "izwi_gateway_http_responses_other_total",
            "counter",
            "HTTP responses outside standard classes.",
            snapshot.http_other
        );
        self.inner.chat.render(&mut output);
        debug_assert!(output.len() <= MAX_PROMETHEUS_RESPONSE_BYTES);
        output
    }

    fn snapshot(&self) -> GatewayMetricsSnapshot {
        GatewayMetricsSnapshot {
            active_requests: self.inner.active_requests.load(Ordering::Relaxed),
            active_streams: self.inner.active_streams.load(Ordering::Relaxed),
            auth_rejections: self.inner.auth_rejections.load(Ordering::Relaxed),
            quota_rejections: self.inner.quota_rejections.load(Ordering::Relaxed),
            tenant_concurrency_rejections: self
                .inner
                .tenant_concurrency_rejections
                .load(Ordering::Relaxed),
            body_limit_rejections: self.inner.body_limit_rejections.load(Ordering::Relaxed),
            body_timeout_rejections: self.inner.body_timeout_rejections.load(Ordering::Relaxed),
            routing_dispatch_failures: self.inner.routing_dispatch_failures.load(Ordering::Relaxed),
            dispatch_calls: self.inner.dispatch_calls.load(Ordering::Relaxed),
            dispatch_latency_micros: self.inner.dispatch_latency_micros.load(Ordering::Relaxed),
            http_2xx: self.inner.http_2xx.load(Ordering::Relaxed),
            http_3xx: self.inner.http_3xx.load(Ordering::Relaxed),
            http_4xx: self.inner.http_4xx.load(Ordering::Relaxed),
            http_5xx: self.inner.http_5xx.load(Ordering::Relaxed),
            http_other: self.inner.http_other.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy)]
struct GatewayMetricsSnapshot {
    active_requests: u64,
    active_streams: u64,
    auth_rejections: u64,
    quota_rejections: u64,
    tenant_concurrency_rejections: u64,
    body_limit_rejections: u64,
    body_timeout_rejections: u64,
    routing_dispatch_failures: u64,
    dispatch_calls: u64,
    dispatch_latency_micros: u64,
    http_2xx: u64,
    http_3xx: u64,
    http_4xx: u64,
    http_5xx: u64,
    http_other: u64,
}

#[derive(Debug, Serialize)]
struct GatewayLiveResponse {
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
}

#[derive(Debug, Serialize)]
struct GatewayReadyResponse {
    status: &'static str,
    version: &'static str,
    ready: bool,
    phase: String,
    draining: bool,
    uptime_secs: u64,
    checks: Vec<GatewayProbeCheck>,
}

#[derive(Debug, Serialize)]
struct GatewayProbeCheck {
    name: &'static str,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

async fn live_check(State(state): State<GatewayState>) -> Json<GatewayLiveResponse> {
    let lifecycle = state.lifecycle.snapshot();
    Json(GatewayLiveResponse {
        status: "alive",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: now_saturating_sub(lifecycle.started_at),
    })
}

async fn ready_check(State(state): State<GatewayState>) -> Response {
    let lifecycle = state.lifecycle.snapshot();
    let draining = lifecycle.draining || state.prestop.load(Ordering::Acquire);
    // Ready while any served model has a ready worker; the message names the
    // models that have none.
    let unready: Vec<String> = state
        .chat_dispatchers
        .iter()
        .filter_map(|dispatcher| dispatcher.readiness_check().err())
        .collect();
    let worker_result = if unready.len() < state.chat_dispatchers.len() {
        Ok(())
    } else {
        Err(unready.join("; "))
    };
    let checks = vec![
        GatewayProbeCheck {
            name: "lifecycle_ready",
            ok: lifecycle.ready,
            message: (!lifecycle.ready).then(|| format!("phase is {}", lifecycle.phase)),
        },
        GatewayProbeCheck {
            name: "not_draining",
            ok: !draining,
            message: draining.then(|| "gateway is draining for shutdown".to_string()),
        },
        GatewayProbeCheck {
            name: "compatible_worker_ready",
            ok: worker_result.is_ok(),
            message: worker_result.err(),
        },
    ];
    let ready = checks.iter().all(|check| check.ok);
    let response = GatewayReadyResponse {
        status: if ready { "ready" } else { "unready" },
        version: env!("CARGO_PKG_VERSION"),
        ready,
        phase: lifecycle.phase,
        draining,
        uptime_secs: now_saturating_sub(lifecycle.started_at),
        checks,
    };
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(response),
    )
        .into_response()
}

async fn api_not_found() -> Response {
    ApiError::not_found("unknown gateway route").into_response()
}

/// Admin routes exist only with an admin key or an admin-role principal,
/// and answer only to it: the rejection is 404 without one configured and
/// 401 for any other caller.
fn admin_rejection(state: &GatewayState, headers: &HeaderMap) -> Option<Response> {
    let admin_available = state.perimeter.admin_enabled() || state.principal_keys.has_role("admin");
    if !admin_available {
        return Some(StatusCode::NOT_FOUND.into_response());
    }
    let admin_authorized = (state.perimeter.admin_enabled()
        && state.perimeter.authenticate_admin(headers))
        || state.principal_keys.authorize(headers, "admin");
    if !admin_authorized {
        let mut response = StatusCode::UNAUTHORIZED.into_response();
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            axum::http::HeaderValue::from_static("Bearer realm=\"izwi-gateway-admin\""),
        );
        return Some(response);
    }
    None
}

/// The approvals view this gateway adopted and the generation each pool
/// routes to, so a rollout can confirm adoption before it moves on.
async fn gateway_approvals_status(
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response {
    if let Some(rejection) = admin_rejection(&state, &headers) {
        return rejection;
    }
    match state.approvals.as_ref() {
        Some(reporter) => Json(reporter.status()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn gateway_drain(State(state): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Some(rejection) = admin_rejection(&state, &headers) {
        return rejection;
    }
    state.begin_drain();
    let lifecycle = state.lifecycle.snapshot();
    (
        StatusCode::ACCEPTED,
        Json(GatewayDrainResponse {
            status: "draining",
            draining: true,
            ready: !lifecycle.ready,
            uptime_secs: now_saturating_sub(lifecycle.started_at),
        }),
    )
        .into_response()
}

#[derive(Debug, Serialize)]
struct GatewayDrainResponse {
    status: &'static str,
    draining: bool,
    ready: bool,
    uptime_secs: u64,
}

async fn gateway_metrics(State(state): State<GatewayState>, headers: HeaderMap) -> Response {
    let metrics_available =
        state.perimeter.metrics_enabled() || state.principal_keys.has_role("metrics");
    if !metrics_available {
        return StatusCode::NOT_FOUND.into_response();
    }
    let metrics_authorized = (state.perimeter.metrics_enabled()
        && state.perimeter.authenticate_metrics(&headers))
        || state.principal_keys.authorize(&headers, "metrics");
    if !metrics_authorized {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        let mut response = StatusCode::UNAUTHORIZED.into_response();
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            axum::http::HeaderValue::from_static("Bearer realm=\"izwi-gateway-metrics\""),
        );
        return response;
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state
            .metrics
            .render_prometheus(state.tenant_concurrency.active_owned_work()),
    )
        .into_response()
}

async fn observe_gateway_http_outcome(
    State(state): State<GatewayState>,
    request: Request,
    next: Next,
) -> Response {
    let started = Instant::now();
    let path = request.uri().path().to_owned();
    let response = next.run(request).await;
    state
        .metrics
        .record_http_response(&path, response.status(), started.elapsed());
    response
}

async fn observe_gateway_v1_rejections(
    State(state): State<GatewayState>,
    request: Request,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    match response.status() {
        StatusCode::UNAUTHORIZED => {
            state
                .metrics
                .inner
                .auth_rejections
                .fetch_add(1, Ordering::Relaxed);
        }
        StatusCode::PAYLOAD_TOO_LARGE => {
            state
                .metrics
                .inner
                .body_limit_rejections
                .fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    response
}

/// Per-tenant admission ahead of the global permit: a tenant over its
/// request rate or concurrent-work limit is refused before it can take one
/// of the gateway's global permits, so one tenant's slow or abundant
/// requests cannot exhaust admission for every other tenant. The rate token
/// itself is spent only after the body validates.
async fn tenant_pre_admission(
    State(state): State<GatewayState>,
    mut request: Request,
    next: Next,
) -> Response {
    let lifecycle = state.lifecycle.snapshot();
    if !lifecycle.ready || lifecycle.draining {
        return ApiError::service_unavailable("Gateway is not accepting new requests")
            .into_response();
    }
    let Some(context) = request.extensions().get::<RequestContext>().cloned() else {
        return ApiError::service_unavailable("Gateway tenant identity is unavailable")
            .into_response();
    };
    let Some(tenant_key) = context.tenant_key() else {
        return ApiError::service_unavailable("Gateway tenant identity is unavailable")
            .into_response();
    };
    let response = match state.rate_quota.peek(tenant_key) {
        Ok(GatewayRateDecision::Allowed) => match state.begin_tenant_work(&context) {
            Ok(lease) => {
                request
                    .extensions_mut()
                    .insert(PreAdmittedTenantWork(Arc::new(std::sync::Mutex::new(
                        Some(lease),
                    ))));
                next.run(request).await
            }
            Err(error) => error.into_response(),
        },
        Ok(GatewayRateDecision::Limited) => {
            state
                .metrics
                .inner
                .quota_rejections
                .fetch_add(1, Ordering::Relaxed);
            ApiError::too_many_requests("Gateway tenant rate limit exceeded").into_response()
        }
        Err(_) => {
            ApiError::service_unavailable("Gateway tenant rate limiter unavailable").into_response()
        }
    };
    with_rate_limit_headers(state.rate_quota.snapshot(tenant_key), response)
}

/// OpenAI's `x-ratelimit-*` headers for the tenant's request (and, when
/// configured, token) limits on every chat response, and `Retry-After` on
/// 429 and 503 answers that do not carry one.
fn with_rate_limit_headers(
    snapshot: Option<RateLimitSnapshot>,
    mut response: Response,
) -> Response {
    let whole_seconds =
        |duration: Duration| duration.as_secs() + u64::from(duration.subsec_nanos() > 0);
    let status = response.status();
    let headers = response.headers_mut();
    let mut retry_after = Duration::from_secs(1);
    if let Some(snapshot) = snapshot {
        let mut windows = vec![("requests", snapshot.requests)];
        windows.extend(snapshot.tokens.map(|tokens| ("tokens", tokens)));
        for (name, window) in &windows {
            let values = [
                ("limit", window.limit.to_string()),
                ("remaining", window.remaining.to_string()),
                ("reset", format!("{}s", whole_seconds(window.reset))),
            ];
            for (field, value) in values {
                if let (Ok(header), Ok(value)) = (
                    header::HeaderName::try_from(format!("x-ratelimit-{field}-{name}")),
                    axum::http::HeaderValue::try_from(value),
                ) {
                    headers.insert(header, value);
                }
            }
        }
        if let Some((_, exhausted)) = windows.iter().find(|(_, window)| window.remaining == 0) {
            retry_after = retry_after.max(exhausted.retry_after);
        }
    }
    if matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
    ) && !headers.contains_key(header::RETRY_AFTER)
    {
        headers.insert(
            header::RETRY_AFTER,
            axum::http::HeaderValue::from(whole_seconds(retry_after)),
        );
    }
    response
}

async fn bounded_gateway_admission(
    State(state): State<GatewayState>,
    mut request: Request,
    next: Next,
) -> Response {
    let lifecycle = state.lifecycle.snapshot();
    if !lifecycle.ready || lifecycle.draining {
        return ApiError::service_unavailable("Gateway is not accepting new requests")
            .into_response();
    }
    let Ok(permit) = state.request_admission.clone().try_acquire_owned() else {
        return ApiError::service_unavailable("Gateway request capacity is currently unavailable")
            .into_response();
    };
    // Recheck after reserving capacity so a concurrent drain cannot leave the
    // permit attached to a request that was still waiting to enter admission.
    let lifecycle = state.lifecycle.snapshot();
    if !lifecycle.ready || lifecycle.draining {
        return ApiError::service_unavailable("Gateway is not accepting new requests")
            .into_response();
    }
    request.extensions_mut().insert(GatewayAdmissionGuard {
        _permit: Arc::new(permit),
        _active_request: Arc::new(state.metrics.begin_request()),
    });
    next.run(request).await
}

pub fn create_gateway_router(state: GatewayState, serve_config: &GatewayServeConfig) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request| {
            // Resolved by the request-id middleware outside this layer.
            let trace = RequestTrace::of(request).unwrap_or_else(RequestTrace::detached);
            let request_id = crate::api::request_context::gateway_trace_request_id(request);
            info_span!(
                "http_request",
                service = service_name(),
                version = SERVICE_VERSION,
                method = %request.method(),
                path = %request.uri().path(),
                correlation_id = %request_id,
                trace_id = %trace.trace_id(),
                span_id = %trace.span_id(),
                parent_span_id = %trace.parent_span_id(),
                status = field::Empty,
                latency_ms = field::Empty,
                error = field::Empty
            )
        })
        .on_response(|response: &Response, latency: Duration, span: &Span| {
            let status = response.status().as_u16();
            let latency_ms = latency.as_millis() as u64;
            span.record("status", status);
            span.record("latency_ms", latency_ms);
            info!(
                parent: span,
                service = service_name(),
                version = SERVICE_VERSION,
                status,
                latency_ms,
                "HTTP request completed"
            );
        })
        .on_failure(
            |failure: ServerErrorsFailureClass, latency: Duration, span: &Span| {
                let latency_ms = latency.as_millis() as u64;
                span.record("latency_ms", latency_ms);
                span.record("error", field::display(&failure));
                warn!(
                    parent: span,
                    service = service_name(),
                    version = SERVICE_VERSION,
                    error = %failure,
                    latency_ms,
                    "HTTP request failed"
                );
            },
        );

    // Chat takes its tenant lease before global admission; realtime sessions
    // take theirs per stage once upgraded.
    let mut inference_routes = Router::new()
        .route(
            "/chat/completions",
            post(crate::api::openai::chat::completions::gateway_completions),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            bounded_gateway_admission,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            tenant_pre_admission,
        ));
    if state.realtime_relay.is_some() {
        inference_routes = inference_routes.merge(
            Router::new()
                .route(
                    "/realtime/ws",
                    get(crate::app::realtime_relay::relay_socket),
                )
                // Alias of the single-node transcription socket path so a
                // public realtime client can repoint at the gateway without
                // changing its URL; dispatch still keys off the subprotocol
                // offer.
                .route(
                    "/speech-to-text/realtime/ws",
                    get(crate::app::realtime_relay::relay_socket),
                )
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    bounded_gateway_admission,
                )),
        );
    }
    let inference_routes =
        inference_routes.layer(DefaultBodyLimit::max(state.perimeter.max_chat_body_bytes()));
    // The model listing is authenticated like inference but takes no
    // inference capacity.
    let v1_routes = Router::new()
        .route(
            "/models",
            get(crate::api::openai::models::handlers::gateway_list_models),
        )
        .route(
            "/models/{model}",
            get(crate::api::openai::models::handlers::gateway_get_model),
        )
        .merge(inference_routes)
        .fallback(api_not_found)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            attach_gateway_request_context,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            observe_gateway_v1_rejections,
        ));
    let app = Router::new()
        .route("/livez", get(live_check))
        .route("/readyz", get(ready_check))
        .route("/internal/admin/drain", post(gateway_drain))
        .route(
            izwi_serving_protocol::GATEWAY_APPROVALS_STATUS_PATH,
            get(gateway_approvals_status),
        )
        .route("/internal/metrics", get(gateway_metrics))
        .route("/internal/metrics/prometheus", get(gateway_metrics))
        .route(
            "/openapi.json",
            get(crate::api::openapi::gateway_openapi_json),
        )
        .merge(crate::api::docs::router())
        .nest("/v1", v1_routes)
        .fallback(api_not_found)
        .with_state(state.clone());

    apply_gateway_cors(app, serve_config)
        .layer(trace_layer)
        .layer(middleware::from_fn(attach_gateway_request_id))
        .layer(middleware::from_fn_with_state(
            state,
            observe_gateway_http_outcome,
        ))
}

fn apply_gateway_cors(app: Router, serve_config: &GatewayServeConfig) -> Router {
    if !serve_config.cors_enabled {
        return app;
    }
    let layer = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if serve_config.cors_origins.is_empty()
        || serve_config
            .cors_origins
            .iter()
            .any(|origin| origin.trim() == "*")
    {
        return app.layer(layer.allow_origin(Any));
    }
    let origins = serve_config
        .cors_origins
        .iter()
        .filter_map(|origin| axum::http::HeaderValue::from_str(origin).ok())
        .collect::<Vec<_>>();
    app.layer(layer.allow_origin(origins))
}

fn now_saturating_sub(started_at: u64) -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_sub(started_at)
}

#[cfg(test)]
#[path = "gateway_tests.rs"]
mod tests;

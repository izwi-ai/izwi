//! Hardware-independent public gateway state and routes.
//!
//! This module deliberately has no `RuntimeService`, persistence, model, or
//! accelerator initialization path. Its only execution dependency is the
//! bounded private worker client wrapped by `RemoteChatExecution`.

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_core::ServeRuntimeConfig;
use izwi_hooks::{
    EnterpriseAction, EnterpriseHooks, HookMetadata, QuotaRequest, ResourceDescriptor,
};
use serde::Serialize;
use std::fmt::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{field, info, info_span, warn, Span};

use crate::api::request_context::{attach_gateway_request_context, attach_gateway_request_id};
use crate::app::chat::{ChatExecutionRequest, RemoteChatExecution};
use crate::app::remote_chat_dispatch::RemoteChatDispatcher;
use crate::error::ApiError;
use crate::gateway_rate_quota::{GatewayRateDecision, GatewayRateQuota, GatewayRateQuotaConfig};
use crate::gateway_security::GatewayPerimeterConfig;
use crate::gateway_tenant_concurrency::{
    GatewayTenantConcurrency, GatewayTenantConcurrencyConfig, TenantWorkAdmissionError,
    UnboundTenantWorkLease,
};
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::state::ServerLifecycle;

const CHAT_COMPLETIONS_RESOURCE: &str = "/v1/chat/completions";
const MAX_GATEWAY_QUOTA_HOOK_TIME: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct GatewayState {
    pub(crate) chat_execution: GatewayChatExecution,
    pub(crate) enterprise_hooks: EnterpriseHooks,
    pub(crate) lifecycle: ServerLifecycle,
    pub(crate) perimeter: GatewayPerimeterConfig,
    pub request_timeout_secs: u64,
    request_admission: Arc<Semaphore>,
    rate_quota: GatewayRateQuota,
    tenant_concurrency: GatewayTenantConcurrency,
    max_output_tokens: u32,
    metrics: GatewayMetrics,
}

impl GatewayState {
    pub fn new(
        remote_chat_execution: RemoteChatExecution,
        enterprise_hooks: EnterpriseHooks,
        perimeter: GatewayPerimeterConfig,
        request_timeout_secs: u64,
        max_in_flight: usize,
    ) -> Self {
        debug_assert!(max_in_flight > 0);
        let max_output_tokens = remote_chat_execution.max_output_tokens();
        let tenant_concurrency_config =
            GatewayTenantConcurrencyConfig::new(8_usize.min(max_in_flight), max_in_flight)
                .expect("validated gateway admission limit supports tenant ownership");
        Self {
            chat_execution: GatewayChatExecution::Pinned(remote_chat_execution),
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            perimeter,
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
            rate_quota: GatewayRateQuota::new(GatewayRateQuotaConfig::default()),
            tenant_concurrency: GatewayTenantConcurrency::new(tenant_concurrency_config),
            max_output_tokens,
            metrics: GatewayMetrics::default(),
        }
    }

    pub fn with_dispatcher(
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
        Self {
            chat_execution: GatewayChatExecution::Registry(dispatcher),
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            perimeter,
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
            rate_quota: GatewayRateQuota::new(GatewayRateQuotaConfig::default()),
            tenant_concurrency: GatewayTenantConcurrency::new(tenant_concurrency_config),
            max_output_tokens,
            metrics: GatewayMetrics::default(),
        }
    }

    /// Override process-local request-rate policy and the exact private worker
    /// output ceiling. This remains rate-only: response-owned global admission
    /// and teardown-aware tenant concurrent-work ownership are separate gates.
    pub fn with_rate_quota_config(mut self, config: GatewayRateQuotaConfig) -> Self {
        self.rate_quota = GatewayRateQuota::new(config);
        self
    }

    pub(crate) fn with_tenant_concurrency_config(
        mut self,
        config: GatewayTenantConcurrencyConfig,
    ) -> Self {
        self.tenant_concurrency = GatewayTenantConcurrency::new(config);
        self
    }

    pub(crate) async fn enforce_chat_rate_quota(
        &self,
        context: &crate::api::request_context::RequestContext,
        request: &ChatExecutionRequest,
    ) -> Result<(), ApiError> {
        let requested_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(usize::MAX)
            .max(1);
        let estimated_units = u32::try_from(requested_tokens)
            .unwrap_or(u32::MAX)
            .min(self.max_output_tokens)
            .max(1);
        let resource = ResourceDescriptor::http_route(CHAT_COMPLETIONS_RESOURCE);
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
            resource,
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

        let tenant_key = context.tenant_key().ok_or_else(|| {
            ApiError::service_unavailable("Gateway tenant identity is unavailable")
        })?;
        match self.rate_quota.check(tenant_key) {
            Ok(GatewayRateDecision::Allowed) => Ok(()),
            Ok(GatewayRateDecision::Limited) => {
                self.metrics
                    .inner
                    .quota_rejections
                    .fetch_add(1, Ordering::Relaxed);
                Err(ApiError::too_many_requests(
                    "Gateway tenant rate limit exceeded",
                ))
            }
            Err(_) => Err(ApiError::service_unavailable(
                "Gateway tenant rate limiter unavailable",
            )),
        }
    }

    pub(crate) fn begin_tenant_work(
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

    pub(crate) fn record_dispatch_success(&self, elapsed: Duration) {
        self.metrics.record_dispatch(elapsed, false);
    }

    pub(crate) fn record_dispatch_failure(&self, elapsed: Duration) {
        self.metrics.record_dispatch(elapsed, true);
    }

    pub(crate) fn begin_stream_observation(&self) -> GatewayStreamMetricsGuard {
        self.metrics.begin_stream()
    }

    pub fn mark_ready(&self) {
        self.lifecycle.mark_ready();
    }

    pub fn begin_drain(&self) {
        // Publishing the lifecycle transition before closing admission avoids
        // a gap where a newly acquired permit could pass the post-acquire
        // lifecycle check. Existing response-owned permits remain valid.
        self.lifecycle.mark_draining();
        self.request_admission.close();
    }
}

/// Execution target for the migrated public chat route. `Pinned` preserves the
/// original one-worker deployment shape while `Registry` selects among fresh,
/// compatible worker incarnations without changing the public API.
#[derive(Clone)]
pub(crate) enum GatewayChatExecution {
    Pinned(RemoteChatExecution),
    Registry(RemoteChatDispatcher),
}

impl GatewayChatExecution {
    pub(crate) async fn readiness_check(&self) -> Result<(), String> {
        match self {
            Self::Pinned(remote) => remote.readiness_check().await,
            Self::Registry(dispatcher) => dispatcher.readiness_check(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct GatewayAdmissionGuard {
    _permit: Arc<OwnedSemaphorePermit>,
    _active_request: Arc<GatewayActiveRequestGuard>,
}

const MAX_PROMETHEUS_RESPONSE_BYTES: usize = 4096;

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
    routing_dispatch_failures: AtomicU64,
    dispatch_calls: AtomicU64,
    dispatch_latency_micros: AtomicU64,
    http_2xx: AtomicU64,
    http_3xx: AtomicU64,
    http_4xx: AtomicU64,
    http_5xx: AtomicU64,
    http_other: AtomicU64,
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

pub(crate) struct GatewayStreamMetricsGuard {
    metrics: GatewayMetrics,
    terminal_observed: bool,
}

impl GatewayStreamMetricsGuard {
    pub(crate) fn record_completion(&mut self) {
        self.terminal_observed = true;
    }

    pub(crate) fn record_failure(&mut self) {
        if !self.terminal_observed {
            self.metrics
                .inner
                .routing_dispatch_failures
                .fetch_add(1, Ordering::Relaxed);
            self.terminal_observed = true;
        }
    }
}

impl Drop for GatewayStreamMetricsGuard {
    fn drop(&mut self) {
        if !self.terminal_observed {
            self.metrics
                .inner
                .routing_dispatch_failures
                .fetch_add(1, Ordering::Relaxed);
        }
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

    fn begin_stream(&self) -> GatewayStreamMetricsGuard {
        self.inner.active_streams.fetch_add(1, Ordering::Relaxed);
        GatewayStreamMetricsGuard {
            metrics: self.clone(),
            terminal_observed: false,
        }
    }

    fn record_dispatch(&self, elapsed: Duration, failed: bool) {
        self.inner.dispatch_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.dispatch_latency_micros.fetch_add(
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if failed {
            self.inner
                .routing_dispatch_failures
                .fetch_add(1, Ordering::Relaxed);
        }
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
        let mut output = String::with_capacity(MAX_PROMETHEUS_RESPONSE_BYTES);
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
    let worker_result = state.chat_execution.readiness_check().await;
    let checks = vec![
        GatewayProbeCheck {
            name: "lifecycle_ready",
            ok: lifecycle.ready,
            message: (!lifecycle.ready).then(|| format!("phase is {}", lifecycle.phase)),
        },
        GatewayProbeCheck {
            name: "not_draining",
            ok: !lifecycle.draining,
            message: lifecycle
                .draining
                .then(|| "gateway is draining for shutdown".to_string()),
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
        draining: lifecycle.draining,
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

async fn api_not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

async fn gateway_drain(State(state): State<GatewayState>, headers: HeaderMap) -> Response {
    if !state.perimeter.admin_enabled() {
        return StatusCode::NOT_FOUND.into_response();
    }
    if !state.perimeter.authenticate_admin(&headers) {
        let mut response = StatusCode::UNAUTHORIZED.into_response();
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            axum::http::HeaderValue::from_static("Bearer realm=\"izwi-gateway-admin\""),
        );
        return response;
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
    if !state.perimeter.metrics_enabled() {
        return StatusCode::NOT_FOUND.into_response();
    }
    if !state.perimeter.authenticate_metrics(&headers) {
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
    let response = next.run(request).await;
    state.metrics.record_http_outcome(response.status());
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

pub fn create_gateway_router(state: GatewayState, serve_config: &ServeRuntimeConfig) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request| {
            let request_id = crate::api::request_context::gateway_trace_request_id(request);
            info_span!(
                "http_request",
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                method = %request.method(),
                path = %request.uri().path(),
                correlation_id = %request_id,
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
                service = SERVICE_NAME,
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
                    service = SERVICE_NAME,
                    version = SERVICE_VERSION,
                    error = %failure,
                    latency_ms,
                    "HTTP request failed"
                );
            },
        );

    let v1_routes = Router::new()
        .route(
            "/chat/completions",
            post(crate::api::openai::chat::completions::gateway_completions),
        )
        .fallback(api_not_found)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            bounded_gateway_admission,
        ))
        .layer(DefaultBodyLimit::max(state.perimeter.max_chat_body_bytes()))
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

fn apply_gateway_cors(app: Router, serve_config: &ServeRuntimeConfig) -> Router {
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
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use futures::StreamExt;
    use izwi_core::ModelVariant;
    use izwi_hooks::{
        AuthorizationDecision, AuthorizationRequest, EnterpriseAction, HookError, HookResult,
        PolicyEngine, QuotaDecision, QuotaLimiter, QuotaRequest,
    };
    use izwi_serving_client::{
        mock::{MockFault, MockWorker, MockWorkerConfig},
        WorkerClient, WorkerClientConfig,
    };
    use izwi_serving_protocol::{ModelAlias, ModelGeneration, PolicyRevision};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use tower::Service;

    use crate::api::request_context::RequestContext;
    use crate::app::remote_chat_dispatch::RemoteChatDispatchConfig;
    use crate::gateway_security::MAX_GATEWAY_REQUEST_ID_BYTES;
    use crate::worker_registry::{
        ApprovedDeployment, ApprovedWorker, BackendPolicy, WorkerRegistry, WorkerRegistryConfig,
    };

    const TEST_API_KEY: &str = "test-public-api-key-123456";
    const TEST_METRICS_API_KEY: &str = "test-metrics-api-key-654321";

    fn test_perimeter() -> GatewayPerimeterConfig {
        GatewayPerimeterConfig::new_for_test(TEST_API_KEY, 1024 * 1024)
            .expect("test perimeter should be valid")
    }

    async fn send(mut app: Router, mut request: Request<Body>) -> Response {
        if request.uri().path().starts_with("/v1/")
            && !request
                .headers()
                .contains_key(axum::http::header::AUTHORIZATION)
        {
            request.headers_mut().insert(
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderValue::from_static("Bearer test-public-api-key-123456"),
            );
        }
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("gateway router request should succeed")
    }

    async fn send_raw(mut app: Router, request: Request<Body>) -> Response {
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("gateway router request should succeed")
    }

    fn get(path: &str) -> Request<Body> {
        Request::builder()
            .uri(path)
            .body(Body::empty())
            .expect("request should build")
    }

    fn post(path: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .body(Body::empty())
            .expect("request should build")
    }

    fn post_with_bearer(path: &str, bearer: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::empty())
            .expect("request should build")
    }

    fn get_with_bearer(path: &str, bearer: &str) -> Request<Body> {
        Request::builder()
            .uri(path)
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::empty())
            .expect("request should build")
    }

    fn valid_chat_request(max_tokens: usize) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": ModelVariant::Qwen34BGguf.dir_name(),
                    "messages": [{"role": "user", "content": "quota test"}],
                    "max_tokens": max_tokens
                })
                .to_string(),
            ))
            .expect("chat request should build")
    }

    #[test]
    fn fixed_gateway_metrics_track_lifetimes_latency_and_status_classes() {
        let metrics = GatewayMetrics::default();
        let request = metrics.begin_request();
        let mut stream = metrics.begin_stream();
        metrics.record_dispatch(Duration::from_micros(7), true);
        for status in [
            StatusCode::CONTINUE,
            StatusCode::OK,
            StatusCode::TEMPORARY_REDIRECT,
            StatusCode::BAD_REQUEST,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            metrics.record_http_outcome(status);
        }
        let active = metrics.snapshot();
        assert_eq!(active.active_requests, 1);
        assert_eq!(active.active_streams, 1);
        assert_eq!(active.dispatch_calls, 1);
        assert_eq!(active.dispatch_latency_micros, 7);
        assert_eq!(active.routing_dispatch_failures, 1);
        assert_eq!(active.http_2xx, 1);
        assert_eq!(active.http_3xx, 1);
        assert_eq!(active.http_4xx, 1);
        assert_eq!(active.http_5xx, 1);
        assert_eq!(active.http_other, 1);
        stream.record_completion();
        drop(stream);
        drop(request);
        assert_eq!(metrics.snapshot().active_requests, 0);
        assert_eq!(metrics.snapshot().active_streams, 0);
        let abandoned_stream = metrics.begin_stream();
        drop(abandoned_stream);
        assert_eq!(metrics.snapshot().routing_dispatch_failures, 2);
        assert!(metrics.render_prometheus(0).len() <= MAX_PROMETHEUS_RESPONSE_BYTES);
    }

    async fn registry_gateway_state(model: ModelVariant) -> (GatewayState, MockWorker) {
        registry_gateway_state_with_config(
            model,
            MockWorkerConfig {
                public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
                ..MockWorkerConfig::default()
            },
        )
        .await
    }

    async fn registry_gateway_state_with_config(
        model: ModelVariant,
        worker_config: MockWorkerConfig,
    ) -> (GatewayState, MockWorker) {
        let credentials = worker_config.credentials.clone();
        let worker = MockWorker::spawn(worker_config)
            .await
            .expect("mock worker should bind");
        let client = WorkerClient::new(
            &worker.endpoint(),
            credentials,
            WorkerClientConfig::default(),
        )
        .expect("worker client should initialize");
        let descriptor = client
            .descriptor()
            .await
            .expect("descriptor should be available");
        let status = client.status().await.expect("status should be available");
        let deployment = status
            .deployments
            .first()
            .expect("mock deployment should exist")
            .clone();
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default())
            .expect("registry should initialize");
        registry
            .approve(ApprovedWorker {
                descriptor,
                client,
                approved_deployments: BTreeMap::from([(
                    deployment.deployment_id.clone(),
                    ApprovedDeployment::from_loaded(&deployment),
                )]),
                validated_capacity: status.capacity.max_active_invocations
                    + status.capacity.max_queued_invocations,
            })
            .expect("worker should be approved");
        registry
            .observe_status(status)
            .expect("initial status should be valid");
        let dispatcher = RemoteChatDispatcher::new(
            registry,
            RemoteChatDispatchConfig {
                public_model_variant: model,
                deployment_id: deployment.deployment_id,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                backend_policy: BackendPolicy::ANY,
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("dispatcher should initialize");
        let state = GatewayState::with_dispatcher(
            dispatcher,
            EnterpriseHooks::noop(),
            test_perimeter(),
            2,
            4,
        );
        state.lifecycle.mark_ready();
        (state, worker)
    }

    #[tokio::test]
    async fn public_chat_nonstream_and_stream_use_registry_dispatcher() {
        let model = ModelVariant::Qwen34BGguf;
        let (state, _worker) = registry_gateway_state(model).await;
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());

        assert_eq!(
            send(app.clone(), get("/readyz")).await.status(),
            StatusCode::OK
        );
        let nonstream = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "registry nonstream"}],
                    "stream": false,
                    "max_tokens": 32
                })
                .to_string(),
            ))
            .expect("nonstream request should build");
        let response = send(app.clone(), nonstream).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("response should be bounded");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("response should be JSON")
                ["choices"][0]["message"]["content"],
            "deterministic mock response"
        );

        let streaming = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "registry stream"}],
                    "stream": true,
                    "stream_options": {"include_usage": true},
                    "max_tokens": 32
                })
                .to_string(),
            ))
            .expect("streaming request should build");
        let response = send(app, streaming).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("SSE response should be bounded");
        let body = String::from_utf8(body.to_vec()).expect("SSE should be UTF-8");
        assert!(body.contains("deterministic mock response"));
        assert!(body.contains("data: [DONE]"));
        let metrics = state.metrics.snapshot();
        assert_eq!(metrics.dispatch_calls, 2);
        assert_eq!(metrics.routing_dispatch_failures, 0);
        assert_eq!(metrics.active_requests, 0);
        assert_eq!(metrics.active_streams, 0);
    }

    #[tokio::test]
    async fn tenant_work_survives_stream_disconnect_until_exact_worker_teardown() {
        let model = ModelVariant::Qwen34BGguf;
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            fault: MockFault::Hang,
            cancellation_delay: Duration::from_millis(100),
            ..MockWorkerConfig::default()
        };
        let credentials = worker_config.credentials.clone();
        let incarnation = worker_config.incarnation_id.clone();
        let deployment = worker_config.deployment_id.clone();
        let generation = worker_config.model_generation;
        let worker = MockWorker::spawn(worker_config)
            .await
            .expect("mock worker should bind");
        let client = WorkerClient::new(
            &worker.endpoint(),
            credentials,
            WorkerClientConfig::default(),
        )
        .expect("worker client should initialize");
        let remote = RemoteChatExecution::new(
            client,
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: model,
                expected_worker_incarnation: incarnation,
                deployment_id: deployment,
                expected_model_generation: generation,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), test_perimeter(), 2, 4)
            .with_tenant_concurrency_config(
                GatewayTenantConcurrencyConfig::new(1, 2).expect("test tenant limit"),
            );
        state.lifecycle.mark_ready();
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());

        let stream_request = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": model.dir_name(),
                        "messages": [{"role": "user", "content": "tenant ownership"}],
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("stream request should build")
        };

        let response = send(app.clone(), stream_request()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 1);
        drop(response);

        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(
            state.tenant_concurrency.active_owned_work(),
            1,
            "public disconnect and cancellation acknowledgement are not teardown proof"
        );
        let rejected = send(app.clone(), valid_chat_request(8)).await;
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);

        for _ in 0..100 {
            if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(worker.active_invocations(), 0);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 0);

        let admitted_again = send(app, stream_request()).await;
        assert_eq!(admitted_again.status(), StatusCode::OK);
        drop(admitted_again);
        for _ in 0..100 {
            if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(worker.active_invocations(), 0);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 0);
    }

    #[tokio::test]
    async fn registry_stream_disconnect_releases_only_after_exact_worker_teardown() {
        let model = ModelVariant::Qwen34BGguf;
        let (state, worker) = registry_gateway_state_with_config(
            model,
            MockWorkerConfig {
                public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
                fault: MockFault::Hang,
                cancellation_delay: Duration::from_millis(100),
                ..MockWorkerConfig::default()
            },
        )
        .await;
        let state = state.with_tenant_concurrency_config(
            GatewayTenantConcurrencyConfig::new(1, 2).expect("test tenant limit"),
        );
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let request = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": model.dir_name(),
                        "messages": [{"role": "user", "content": "registry ownership"}],
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("stream request should build")
        };

        let response = send(app.clone(), request()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 1);
        drop(response);

        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 1);
        assert_eq!(
            send(app.clone(), valid_chat_request(8)).await.status(),
            StatusCode::TOO_MANY_REQUESTS
        );

        for _ in 0..100 {
            if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(worker.active_invocations(), 0);
        assert_eq!(state.tenant_concurrency.active_owned_work(), 0);

        let admitted_again = send(app, request()).await;
        assert_eq!(admitted_again.status(), StatusCode::OK);
        drop(admitted_again);
    }

    #[tokio::test]
    async fn public_chat_routes_concurrent_requests_to_distinct_http_replicas_without_retry() {
        let model = ModelVariant::Qwen34BGguf;
        let deployment_id = MockWorkerConfig::default().deployment_id;
        let replica = |suffix: &str, output_text: &str| MockWorkerConfig {
            worker_id: format!("mock-replica-{suffix}")
                .try_into()
                .expect("test worker id should be valid"),
            node_id: format!("mock-node-{suffix}")
                .try_into()
                .expect("test node id should be valid"),
            incarnation_id: format!("mock-incarnation-{suffix}")
                .try_into()
                .expect("test incarnation should be valid"),
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            output_text: output_text.to_string(),
            // Once the first delta is observed, this leaves a deterministic
            // window in which the first worker still owns its sole permit.
            output_cadence: Duration::from_secs(1),
            ..MockWorkerConfig::default()
        };
        let worker_a = MockWorker::spawn(replica("a", "replica-a-only"))
            .await
            .expect("first mock replica should bind");
        let worker_b = MockWorker::spawn(replica("b", "replica-b-only"))
            .await
            .expect("second mock replica should bind");
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default())
            .expect("registry should initialize");

        for worker in [&worker_a, &worker_b] {
            let client = WorkerClient::new(
                &worker.endpoint(),
                worker.config().credentials.clone(),
                WorkerClientConfig::default(),
            )
            .expect("worker client should initialize");
            let descriptor = client
                .descriptor()
                .await
                .expect("descriptor should be available");
            let status = client.status().await.expect("status should be available");
            let deployment = status
                .deployments
                .first()
                .expect("mock deployment should exist")
                .clone();
            registry
                .approve(ApprovedWorker {
                    descriptor,
                    client,
                    approved_deployments: BTreeMap::from([(
                        deployment.deployment_id.clone(),
                        ApprovedDeployment::from_loaded(&deployment),
                    )]),
                    validated_capacity: status.capacity.max_active_invocations
                        + status.capacity.max_queued_invocations,
                })
                .expect("worker should be approved");
            registry
                .observe_status(status)
                .expect("initial status should be valid");
        }

        let dispatcher = RemoteChatDispatcher::new(
            registry,
            RemoteChatDispatchConfig {
                public_model_variant: model,
                deployment_id,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                backend_policy: BackendPolicy::ANY,
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("dispatcher should initialize");
        let state = GatewayState::with_dispatcher(
            dispatcher,
            EnterpriseHooks::noop(),
            test_perimeter(),
            5,
            2,
        );
        state.lifecycle.mark_ready();
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());
        let stream_request = |prompt: &str| {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": model.dir_name(),
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("stream request should build")
        };

        let first = send(app.clone(), stream_request("first replica request")).await;
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(worker_a.active_invocations(), 1);
        assert_eq!(worker_b.active_invocations(), 0);

        let mut first_body = first.into_body().into_data_stream();
        let mut first_bytes = Vec::new();
        tokio::time::timeout(Duration::from_secs(3), async {
            while !String::from_utf8_lossy(&first_bytes).contains("replica-a-only") {
                let chunk = first_body
                    .next()
                    .await
                    .expect("first replica stream should remain open")
                    .expect("first replica stream chunk should be readable");
                assert!(first_bytes.len() + chunk.len() <= 64 * 1024);
                first_bytes.extend_from_slice(&chunk);
            }
        })
        .await
        .expect("first replica should emit its distinctive delta");

        let second = send(app, stream_request("second replica request")).await;
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            worker_a.active_invocations(),
            1,
            "the first replica must retain its sole authoritative admission"
        );
        assert_eq!(
            worker_b.active_invocations(),
            1,
            "the second request must execute exactly once on the other replica"
        );

        let second_bytes = axum::body::to_bytes(second.into_body(), 64 * 1024)
            .await
            .expect("second SSE response should be bounded");
        while let Some(chunk) = first_body.next().await {
            let chunk = chunk.expect("first replica stream chunk should be readable");
            assert!(first_bytes.len() + chunk.len() <= 64 * 1024);
            first_bytes.extend_from_slice(&chunk);
        }
        let first_text = String::from_utf8(first_bytes).expect("first SSE should be UTF-8");
        let second_text =
            String::from_utf8(second_bytes.to_vec()).expect("second SSE should be UTF-8");

        assert_eq!(first_text.matches("replica-a-only").count(), 1);
        assert!(!first_text.contains("replica-b-only"));
        assert!(first_text.contains("data: [DONE]"));
        assert_eq!(second_text.matches("replica-b-only").count(), 1);
        assert!(!second_text.contains("replica-a-only"));
        assert!(second_text.contains("data: [DONE]"));
        assert_eq!(worker_a.active_invocations(), 0);
        assert_eq!(worker_b.active_invocations(), 0);
    }

    #[tokio::test]
    async fn gateway_router_uses_only_the_remote_worker_and_hides_local_routes() {
        let model = ModelVariant::Qwen34BGguf;
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            ..MockWorkerConfig::default()
        };
        let credentials = worker_config.credentials.clone();
        let incarnation = worker_config.incarnation_id.clone();
        let deployment = worker_config.deployment_id.clone();
        let generation = worker_config.model_generation;
        let worker = MockWorker::spawn(worker_config)
            .await
            .expect("mock worker should bind");
        let client = WorkerClient::new(
            &worker.endpoint(),
            credentials,
            WorkerClientConfig::default(),
        )
        .expect("worker client should initialize");
        let remote = RemoteChatExecution::new(
            client,
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: model,
                expected_worker_incarnation: incarnation,
                deployment_id: deployment,
                expected_model_generation: generation,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
        state.lifecycle.mark_ready();
        let app = create_gateway_router(
            state,
            &ServeRuntimeConfig {
                ui_enabled: true,
                ..ServeRuntimeConfig::default()
            },
        );

        assert_eq!(
            send(app.clone(), get("/livez")).await.status(),
            StatusCode::OK
        );
        assert_eq!(
            send(app.clone(), get("/readyz")).await.status(),
            StatusCode::OK
        );
        for path in [
            "/v1/models",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
            "/v1/responses",
            "/v1/admin/models",
            "/v1/voice/sessions",
        ] {
            assert_eq!(
                send(app.clone(), get(path)).await.status(),
                StatusCode::NOT_FOUND,
                "gateway mode must not expose unmigrated route {path} or a UI fallback"
            );
        }

        let chat_request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": false,
                    "max_tokens": 32
                })
                .to_string(),
            ))
            .expect("chat request should build");
        let response = send(app.clone(), chat_request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("response should be bounded");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("response should be JSON")
                ["choices"][0]["message"]["content"],
            "deterministic mock response"
        );

        let streaming_request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": true,
                    "stream_options": {"include_usage": true},
                    "max_tokens": 32
                })
                .to_string(),
            ))
            .expect("streaming chat request should build");
        let response = send(app.clone(), streaming_request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("SSE response should be bounded");
        let body = String::from_utf8(body.to_vec()).expect("SSE should be UTF-8");
        assert!(body.contains("deterministic mock response"));
        assert!(body.contains("\"completion_tokens\":3"));
        assert!(body.contains("data: [DONE]"));

        let openapi = send(app, get("/openapi.json")).await;
        let body = axum::body::to_bytes(openapi.into_body(), 1024 * 1024)
            .await
            .expect("OpenAPI response should be bounded");
        let document: serde_json::Value =
            serde_json::from_slice(&body).expect("OpenAPI response should be JSON");
        let paths = document["paths"]
            .as_object()
            .expect("OpenAPI should contain paths");
        assert!(paths.contains_key("/v1/chat/completions"));
        assert!(!paths.contains_key("/v1/models"));
    }

    #[tokio::test]
    async fn gateway_readiness_rejects_a_stale_model_generation() {
        let model = ModelVariant::Qwen34BGguf;
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            ..MockWorkerConfig::default()
        };
        let credentials = worker_config.credentials.clone();
        let incarnation = worker_config.incarnation_id.clone();
        let deployment = worker_config.deployment_id.clone();
        let worker = MockWorker::spawn(worker_config)
            .await
            .expect("mock worker should bind");
        let client = WorkerClient::new(
            &worker.endpoint(),
            credentials,
            WorkerClientConfig::default(),
        )
        .expect("worker client should initialize");
        let remote = RemoteChatExecution::new(
            client,
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: model,
                expected_worker_incarnation: incarnation,
                deployment_id: deployment,
                expected_model_generation: ModelGeneration::new(2)
                    .expect("non-zero stale generation"),
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
        state.lifecycle.mark_ready();
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());

        assert_eq!(
            send(app, get("/readyz")).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn gateway_admission_rejects_instead_of_queueing() {
        let model = ModelVariant::Qwen34BGguf;
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            ..MockWorkerConfig::default()
        };
        let client = WorkerClient::new(
            "http://127.0.0.1:1",
            worker_config.credentials.clone(),
            WorkerClientConfig::default(),
        )
        .expect("worker client should initialize without connecting");
        let remote = RemoteChatExecution::new(
            client,
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: model,
                expected_worker_incarnation: worker_config.incarnation_id,
                deployment_id: worker_config.deployment_id,
                expected_model_generation: worker_config.model_generation,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 128,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), test_perimeter(), 2, 1);
        state.lifecycle.mark_ready();
        let held_permit = state
            .request_admission
            .clone()
            .try_acquire_owned()
            .expect("test should occupy the only gateway permit");
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}]
                })
                .to_string(),
            ))
            .expect("request should build");

        assert_eq!(
            send(app, request).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        drop(held_permit);
    }

    #[tokio::test]
    async fn gateway_admission_rejects_after_lifecycle_drain() {
        let state = unreachable_gateway_state(test_perimeter());
        state.begin_drain();
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": ModelVariant::Qwen34BGguf.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}]
                })
                .to_string(),
            ))
            .expect("request should build");

        let response = send(app, request).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("drain rejection should be bounded");
        let body: serde_json::Value =
            serde_json::from_slice(&body).expect("drain rejection should be JSON");
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(
            body["error"]["message"],
            "Gateway is not accepting new requests"
        );
    }

    #[tokio::test]
    async fn gateway_admission_rejects_before_lifecycle_ready() {
        let mut state = unreachable_gateway_state(test_perimeter());
        state.lifecycle = ServerLifecycle::new();
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());
        let response = send(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": ModelVariant::Qwen34BGguf.dir_name(),
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("request should build"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("startup rejection should be bounded");
        let body: serde_json::Value =
            serde_json::from_slice(&body).expect("startup rejection should be JSON");
        assert_eq!(
            body["error"]["message"],
            "Gateway is not accepting new requests"
        );
    }

    #[tokio::test]
    async fn gateway_admission_is_held_for_the_stream_body_lifetime() {
        let model = ModelVariant::Qwen34BGguf;
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            fault: MockFault::Hang,
            cancellation_delay: Duration::from_millis(100),
            ..MockWorkerConfig::default()
        };
        let credentials = worker_config.credentials.clone();
        let incarnation = worker_config.incarnation_id.clone();
        let deployment = worker_config.deployment_id.clone();
        let generation = worker_config.model_generation;
        let worker = MockWorker::spawn(worker_config)
            .await
            .expect("mock worker should bind");
        let remote = RemoteChatExecution::new(
            WorkerClient::new(
                &worker.endpoint(),
                credentials,
                WorkerClientConfig::default(),
            )
            .expect("worker client should initialize"),
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: model,
                expected_worker_incarnation: incarnation,
                deployment_id: deployment,
                expected_model_generation: generation,
                policy_revision: PolicyRevision::new("test-policy-v1")
                    .expect("static policy revision"),
                max_queue_wait: Duration::from_millis(100),
                max_output_tokens: 32,
                max_output_bytes: 4096,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), test_perimeter(), 2, 1);
        state.lifecycle.mark_ready();
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let stream_request = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": model.dir_name(),
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true,
                        "max_tokens": 8
                    })
                    .to_string(),
                ))
                .expect("stream request should build")
        };

        let first = send(app.clone(), stream_request()).await;
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(state.request_admission.available_permits(), 0);
        assert_eq!(state.metrics.snapshot().active_requests, 1);
        assert_eq!(state.metrics.snapshot().active_streams, 1);

        state.begin_drain();

        let rejected = send(app.clone(), stream_request()).await;
        assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(rejected.into_body(), 4096)
            .await
            .expect("gateway rejection should be bounded");
        let body: serde_json::Value =
            serde_json::from_slice(&body).expect("gateway rejection should be JSON");
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(
            body["error"]["message"],
            "Gateway is not accepting new requests"
        );
        assert_eq!(worker.active_invocations(), 1);
        assert_eq!(state.request_admission.available_permits(), 0);

        drop(first);
        tokio::time::timeout(Duration::from_secs(1), async {
            while worker.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker teardown should be confirmed");
        assert_eq!(state.request_admission.available_permits(), 1);
        assert_eq!(state.metrics.snapshot().active_requests, 0);
        assert_eq!(state.metrics.snapshot().active_streams, 0);

        let still_draining = send(app, stream_request()).await;
        assert_eq!(still_draining.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn gateway_v1_requires_api_key_while_probes_and_docs_remain_public() {
        let state = unreachable_gateway_state(test_perimeter());
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());

        for path in ["/livez", "/readyz", "/openapi.json", "/docs"] {
            assert_ne!(
                send_raw(app.clone(), get(path)).await.status(),
                StatusCode::UNAUTHORIZED,
                "{path} is an explicitly public operational surface"
            );
        }

        let response = send_raw(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::WWW_AUTHENTICATE)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer realm=\"izwi-gateway\"")
        );
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("authentication error must be bounded");
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "authentication_error");
        assert_eq!(body["error"]["code"], "401");
        assert_eq!(state.metrics.snapshot().auth_rejections, 1);
    }

    #[tokio::test]
    async fn gateway_metrics_are_absent_without_a_separate_credential() {
        let app = create_gateway_router(
            unreachable_gateway_state(test_perimeter()),
            &ServeRuntimeConfig::default(),
        );
        for path in ["/internal/metrics", "/internal/metrics/prometheus"] {
            assert_eq!(
                send_raw(app.clone(), get(path)).await.status(),
                StatusCode::NOT_FOUND
            );
        }
    }

    #[tokio::test]
    async fn gateway_metrics_require_their_own_key_and_have_fixed_redacted_cardinality() {
        let perimeter = test_perimeter()
            .with_metrics_api_key_for_test(TEST_METRICS_API_KEY)
            .unwrap();
        let state = unreachable_gateway_state(perimeter);
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());

        assert_eq!(
            send_raw(app.clone(), get("/internal/metrics"))
                .await
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            send_raw(
                app.clone(),
                get_with_bearer("/internal/metrics", TEST_API_KEY),
            )
            .await
            .status(),
            StatusCode::UNAUTHORIZED
        );
        let response = send_raw(
            app,
            get_with_bearer("/internal/metrics/prometheus", TEST_METRICS_API_KEY),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), MAX_PROMETHEUS_RESPONSE_BYTES)
            .await
            .expect("metrics response must remain bounded");
        let body = String::from_utf8(body.to_vec()).expect("metrics must be UTF-8");
        assert!(body.contains("izwi_gateway_active_requests 0"));
        assert!(body.contains("izwi_gateway_active_tenant_work 0"));
        assert!(body.contains("izwi_gateway_auth_rejections_total 2"));
        assert_eq!(body.matches("# TYPE ").count(), 15);
        assert!(
            !body.contains('{'),
            "metrics must not contain dynamic labels"
        );
        for private in [
            TEST_API_KEY,
            TEST_METRICS_API_KEY,
            "test-gateway-principal",
            "test-tenant",
            ModelVariant::Qwen34BGguf.dir_name(),
            "quota test",
        ] {
            assert!(!body.contains(private));
        }
        let snapshot = state.metrics.snapshot();
        assert_eq!(snapshot.auth_rejections, 2);
        assert_eq!(snapshot.http_2xx, 1);
        assert_eq!(snapshot.http_4xx, 2);
    }

    #[tokio::test]
    async fn admin_drain_endpoint_requires_its_own_key_and_404s_without_it() {
        let app = create_gateway_router(
            unreachable_gateway_state(test_perimeter()),
            &ServeRuntimeConfig::default(),
        );
        assert_eq!(
            send_raw(app.clone(), post("/internal/admin/drain"))
                .await
                .status(),
            StatusCode::NOT_FOUND,
            "drain endpoint must 404 when admin key is not configured"
        );
    }

    #[tokio::test]
    async fn admin_drain_endpoint_rejects_inference_key_and_accepts_admin_key() {
        const TEST_ADMIN_API_KEY: &str = "test-admin-api-key-999888";
        let perimeter = test_perimeter()
            .with_admin_api_key_for_test(TEST_ADMIN_API_KEY)
            .unwrap();
        let state = unreachable_gateway_state(perimeter);
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        assert_eq!(
            send_raw(
                app.clone(),
                post_with_bearer("/internal/admin/drain", TEST_API_KEY)
            )
            .await
            .status(),
            StatusCode::UNAUTHORIZED,
            "public inference key must not authorize admin drain"
        );
        let response = send_raw(
            app.clone(),
            post_with_bearer("/internal/admin/drain", TEST_ADMIN_API_KEY),
        )
        .await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let lifecycle = state.lifecycle.snapshot();
        assert!(
            lifecycle.draining,
            "drain endpoint must set the draining flag"
        );
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "draining");
        assert_eq!(body["draining"], true);
        assert_eq!(
            send_raw(app, post("/internal/admin/drain")).await.status(),
            StatusCode::UNAUTHORIZED,
            "drain without auth still 401s even after drain started"
        );
    }

    #[tokio::test]
    async fn gateway_dispatch_failure_and_latency_are_counted_without_dynamic_dimensions() {
        let state = unreachable_gateway_state(test_perimeter());
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let response = send(app, valid_chat_request(4)).await;
        assert_ne!(response.status(), StatusCode::OK);
        let snapshot = state.metrics.snapshot();
        assert_eq!(snapshot.dispatch_calls, 1);
        assert_eq!(snapshot.routing_dispatch_failures, 1);
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.active_streams, 0);
        assert_eq!(snapshot.http_5xx, 1);
    }

    #[tokio::test]
    async fn gateway_disabled_cors_does_not_inherit_desktop_origins() {
        let app = create_gateway_router(
            unreachable_gateway_state(test_perimeter()),
            &ServeRuntimeConfig {
                cors_enabled: false,
                ..ServeRuntimeConfig::default()
            },
        );
        let response = send_raw(
            app,
            Request::builder()
                .uri("/livez")
                .header(axum::http::header::ORIGIN, "tauri://localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(response
            .headers()
            .get(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none());
    }

    #[tokio::test]
    async fn gateway_chat_body_limit_rejects_before_json_decode_with_uniform_error() {
        let perimeter = GatewayPerimeterConfig::new_for_test(TEST_API_KEY, 1024).unwrap();
        let state = unreachable_gateway_state(perimeter);
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let response = send(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(vec![b' '; 1025]))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body-limit error must be bounded");
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "413");
        assert_eq!(
            body["error"]["message"],
            "Gateway chat request body exceeds the configured limit"
        );
        assert_eq!(state.metrics.snapshot().body_limit_rejections, 1);
        assert_eq!(state.metrics.snapshot().http_4xx, 1);
    }

    #[tokio::test]
    async fn gateway_request_id_is_bounded_and_validated_before_tracing() {
        let state = unreachable_gateway_state(test_perimeter());
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let response = send_raw(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_API_KEY}"))
                .header("x-request-id", "x".repeat(MAX_GATEWAY_REQUEST_ID_BYTES + 1))
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .expect("rejection should carry a server-authored request id");
        assert!(request_id.len() <= MAX_GATEWAY_REQUEST_ID_BYTES);
        uuid::Uuid::parse_str(request_id).expect("replacement should be a UUID");
        assert_eq!(state.metrics.snapshot().http_4xx, 1);
    }

    #[tokio::test]
    async fn gateway_headers_are_bounded_before_policy_evaluation() {
        let app = create_gateway_router(
            unreachable_gateway_state(test_perimeter()),
            &ServeRuntimeConfig::default(),
        );
        let response = send_raw(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_API_KEY}"))
                .header("x-padding", "x".repeat(8 * 1024 + 1))
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("header-limit error must be bounded");
        assert!(String::from_utf8_lossy(&body).contains("headers exceed"));
    }

    #[tokio::test]
    async fn gateway_enterprise_policy_is_additional_to_server_authored_inference_identity() {
        #[derive(Default)]
        struct CapturingPolicy(Mutex<Option<AuthorizationRequest>>);

        #[async_trait::async_trait]
        impl PolicyEngine for CapturingPolicy {
            async fn authorize(
                &self,
                request: &AuthorizationRequest,
            ) -> HookResult<AuthorizationDecision> {
                *self.0.lock().unwrap() = Some(request.clone());
                Ok(AuthorizationDecision::deny("internal policy detail"))
            }
        }

        let policy = Arc::new(CapturingPolicy::default());
        let mut hooks = EnterpriseHooks::noop();
        hooks.policy = policy.clone();
        let app = create_gateway_router(
            unreachable_gateway_state_with_hooks(test_perimeter(), hooks),
            &ServeRuntimeConfig::default(),
        );
        let response = send_raw(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {TEST_API_KEY}"))
                .header("x-principal-id", "forged-principal")
                .header("x-tenant-id", "forged-tenant")
                .header("x-scopes", "admin")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let captured = policy.0.lock().unwrap().clone().unwrap();
        assert_eq!(captured.action, EnterpriseAction::Inference);
        assert_eq!(captured.principal.id, "test-gateway-principal");
        assert_eq!(captured.principal.tenant_id.as_deref(), Some("test-tenant"));
        assert_eq!(captured.principal.roles, vec!["inference"]);
        let forwarded_headers = captured.request.unwrap().headers;
        assert!(!forwarded_headers.iter().any(|header| {
            matches!(
                header.name.as_str(),
                "authorization" | "x-principal-id" | "x-tenant-id" | "x-scopes"
            )
        }));
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("policy error must be bounded");
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "permission_denied_error");
        assert_eq!(
            body["error"]["message"],
            "Gateway inference permission denied"
        );
        assert!(!body.to_string().contains("internal policy detail"));
    }

    #[tokio::test]
    async fn gateway_quota_hook_receives_server_identity_and_effective_output_units() {
        #[derive(Default)]
        struct CapturingQuota(Mutex<Option<QuotaRequest>>);

        #[async_trait::async_trait]
        impl QuotaLimiter for CapturingQuota {
            async fn evaluate(&self, request: &QuotaRequest) -> HookResult<QuotaDecision> {
                *self.0.lock().unwrap() = Some(request.clone());
                Ok(QuotaDecision {
                    allowed: false,
                    reason: Some("private quota detail".into()),
                })
            }
        }

        let quota = Arc::new(CapturingQuota::default());
        let mut hooks = EnterpriseHooks::noop();
        hooks.quotas = quota.clone();
        let state = unreachable_gateway_state_with_hooks(test_perimeter(), hooks)
            .with_rate_quota_config(GatewayRateQuotaConfig::new(60, 4, 8).unwrap());
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());
        let mut request = valid_chat_request(100);
        request.headers_mut().insert(
            "x-principal-id",
            axum::http::HeaderValue::from_static("forged-principal"),
        );
        request.headers_mut().insert(
            "x-tenant-id",
            axum::http::HeaderValue::from_static("forged-tenant"),
        );
        let response = send(app, request).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

        let captured = quota.0.lock().unwrap().clone().unwrap();
        assert_eq!(captured.action, EnterpriseAction::Inference);
        let principal = captured.principal.unwrap();
        assert_eq!(principal.id, "test-gateway-principal");
        assert_eq!(principal.tenant_id.as_deref(), Some("test-tenant"));
        assert_eq!(
            captured.resource,
            ResourceDescriptor::http_route(CHAT_COMPLETIONS_RESOURCE)
        );
        assert_eq!(captured.estimated_units, Some(16));
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("quota denial must be bounded");
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(
            body["error"]["message"],
            "Gateway tenant rate limit exceeded"
        );
        assert!(!body.to_string().contains("private quota detail"));
        assert_eq!(state.metrics.snapshot().quota_rejections, 1);
    }

    #[tokio::test]
    async fn gateway_quota_hook_failure_fails_closed_with_stable_error() {
        struct FailingQuota;

        #[async_trait::async_trait]
        impl QuotaLimiter for FailingQuota {
            async fn evaluate(&self, _request: &QuotaRequest) -> HookResult<QuotaDecision> {
                Err(HookError::Failed("private dependency detail".into()))
            }
        }

        let mut hooks = EnterpriseHooks::noop();
        hooks.quotas = Arc::new(FailingQuota);
        let app = create_gateway_router(
            unreachable_gateway_state_with_hooks(test_perimeter(), hooks),
            &ServeRuntimeConfig::default(),
        );
        let response = send(app, valid_chat_request(4)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("quota failure must be bounded");
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body["error"]["message"],
            "Gateway quota service unavailable"
        );
        assert!(!body.to_string().contains("private dependency detail"));
    }

    #[tokio::test]
    async fn local_tenant_rate_limit_runs_only_after_validated_chat_input() {
        let state = unreachable_gateway_state(test_perimeter())
            .with_rate_quota_config(GatewayRateQuotaConfig::new(1, 1, 1).unwrap());
        let principal = state
            .perimeter
            .authenticate(&axum::http::HeaderMap::from_iter([(
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderValue::from_static("Bearer test-public-api-key-123456"),
            )]))
            .expect("test API key should authenticate");
        let tenant_key = RequestContext::new("quota-test".into(), principal)
            .tenant_key()
            .expect("gateway principal has a tenant key");
        let app = create_gateway_router(state.clone(), &ServeRuntimeConfig::default());

        let invalid = send(
            app.clone(),
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            state.rate_quota.check(tenant_key).unwrap(),
            GatewayRateDecision::Allowed,
            "invalid input must not spend a rate token"
        );

        let limited = send(app, valid_chat_request(4)).await;
        assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    fn unreachable_gateway_state(perimeter: GatewayPerimeterConfig) -> GatewayState {
        unreachable_gateway_state_with_hooks(perimeter, EnterpriseHooks::noop())
    }

    fn unreachable_gateway_state_with_hooks(
        perimeter: GatewayPerimeterConfig,
        hooks: EnterpriseHooks,
    ) -> GatewayState {
        let worker_config = MockWorkerConfig::default();
        let client = WorkerClient::new(
            "http://127.0.0.1:1",
            worker_config.credentials.clone(),
            WorkerClientConfig::default(),
        )
        .expect("test client should initialize");
        let remote = RemoteChatExecution::new(
            client,
            crate::app::chat::RemoteChatExecutionConfig {
                public_model_variant: ModelVariant::Qwen34BGguf,
                expected_worker_incarnation: worker_config.incarnation_id,
                deployment_id: worker_config.deployment_id,
                expected_model_generation: worker_config.model_generation,
                policy_revision: PolicyRevision::new("test-policy-v1").unwrap(),
                max_queue_wait: Duration::ZERO,
                max_output_tokens: 16,
                max_output_bytes: 1024,
                slow_consumer_timeout: Duration::from_millis(100),
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, hooks, perimeter, 2, 1);
        state.lifecycle.mark_ready();
        state
    }
}

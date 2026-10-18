#[path = "speech_admission.rs"]
mod speech_admission;

use axum::extract::Request;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::Body as _;
use izwi_hooks::{
    AuditCategory, AuditEvent, AuditOutcome, AuthorizationRequest, EnterpriseAction, HeaderPair,
    HookError, HookMetadata, ObservabilityEvent, ObservabilityEventKind, Principal,
    RequestEnvelope, ResourceDescriptor,
};
use izwi_serving_protocol::{trace::hex, TraceContext, TRACEPARENT_HEADER};
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tracing::warn;
use uuid::Uuid;

use crate::error::with_request_id;
use crate::gateway::GatewayState;
use crate::gateway_security::MAX_GATEWAY_REQUEST_ID_BYTES;

const REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_GATEWAY_HEADER_COUNT: usize = 128;
const MAX_GATEWAY_HEADER_BYTES: usize = 32 * 1024;
const MAX_GATEWAY_HEADER_VALUE_BYTES: usize = 8 * 1024;

#[derive(Clone, Debug)]
pub struct RequestContext {
    pub correlation_id: String,
    pub principal: Principal,
    /// Receiver-local ingress time used to preserve the original end-to-end
    /// request budget when work crosses another process boundary.
    received_at: Instant,
    /// How a streamed answer ended in-band; the request audit reads it when
    /// the body ends, since the HTTP status was sent before the stream ran.
    pub stream_outcome: StreamOutcome,
    /// This process's span of the request's W3C trace: the parent of every
    /// worker call made for it.
    pub trace: TraceContext,
}

impl RequestContext {
    pub fn new(correlation_id: String, principal: Principal) -> Self {
        Self {
            correlation_id,
            principal,
            received_at: Instant::now(),
            stream_outcome: StreamOutcome::default(),
            trace: TraceContext::new_root(),
        }
    }

    pub fn with_trace(mut self, trace: TraceContext) -> Self {
        self.trace = trace;
        self
    }

    /// Scheduling identity comes exclusively from the authenticated extension.
    /// Legacy durable jobs and the local anonymous principal share None.
    pub fn tenant_key(&self) -> Option<[u8; 32]> {
        if self.principal == Principal::local_anonymous() {
            return None;
        }
        use sha2::{Digest, Sha256};
        Some(Sha256::digest(principal_namespace(&self.principal).as_bytes()).into())
    }

    /// Stable durable-store namespace derived only from authenticated identity.
    /// Unlike `tenant_key`, the local anonymous principal remains explicitly
    /// namespaced so its public idempotency keys cannot collide with another
    /// principal representation.
    pub fn tenant_scope(&self) -> String {
        principal_namespace(&self.principal)
    }

    /// When this process received the request.
    pub fn received_at(&self) -> Instant {
        self.received_at
    }

    pub fn remaining_budget(&self, total: Duration) -> Option<Duration> {
        total.checked_sub(self.received_at.elapsed())
    }
}

/// This process's span for one incoming request, continuing the caller's
/// `traceparent` when it is valid and starting a trace otherwise.
#[derive(Clone, Copy, Debug)]
pub struct RequestTrace {
    pub context: TraceContext,
    parent_span_id: Option<[u8; 8]>,
}

impl RequestTrace {
    /// The request's trace, resolved once by the outermost middleware and
    /// kept in the request extensions for the layers inside it.
    pub fn resolve(req: &mut Request) -> Self {
        if let Some(trace) = req.extensions().get::<Self>() {
            return *trace;
        }
        let incoming = req
            .headers()
            .get(TRACEPARENT_HEADER)
            .and_then(|value| value.to_str().ok());
        let (context, parent_span_id) = TraceContext::continue_from(incoming);
        let trace = Self {
            context,
            parent_span_id,
        };
        req.extensions_mut().insert(trace);
        trace
    }

    /// A trace this process starts, for a request no middleware resolved.
    pub fn detached() -> Self {
        Self {
            context: TraceContext::new_root(),
            parent_span_id: None,
        }
    }

    /// The trace of a request a middleware already resolved, if any.
    pub fn of(req: &Request) -> Option<Self> {
        req.extensions().get::<Self>().copied()
    }

    pub fn trace_id(&self) -> String {
        self.context.trace_id()
    }

    pub fn span_id(&self) -> String {
        self.context.span_id()
    }

    /// The caller's span id, or `-` for a trace this process started.
    pub fn parent_span_id(&self) -> String {
        self.parent_span_id
            .map(|span| hex(&span))
            .unwrap_or_else(|| "-".to_string())
    }
}

/// The in-band end of a streamed answer (an SSE stream answers 200 before it
/// runs, and reports a failure inside the body).
#[derive(Clone, Debug, Default)]
pub struct StreamOutcome(Arc<AtomicU8>);

impl StreamOutcome {
    const SUCCEEDED: u8 = 1;
    const FAILED: u8 = 2;

    pub fn record_success(&self) {
        let _ = self
            .0
            .compare_exchange(0, Self::SUCCEEDED, Ordering::AcqRel, Ordering::Acquire);
    }

    pub fn record_failure(&self) {
        let _ = self
            .0
            .compare_exchange(0, Self::FAILED, Ordering::AcqRel, Ordering::Acquire);
    }

    fn failed(&self) -> bool {
        self.0.load(Ordering::Acquire) == Self::FAILED
    }
}

pub fn principal_namespace(principal: &Principal) -> String {
    match &principal.tenant_id {
        Some(tenant) => format!("tenant:{tenant}"),
        None => format!("principal:{}", principal.id),
    }
}

pub async fn attach_gateway_request_context(
    State(state): State<GatewayState>,
    mut req: Request,
    next: Next,
) -> Response {
    let correlation_id = match resolve_gateway_correlation_id(&mut req) {
        Ok(correlation_id) => correlation_id,
        Err(generated_id) => {
            return gateway_rejection_response(
                StatusCode::BAD_REQUEST,
                &generated_id,
                "Invalid x-request-id header",
            )
        }
    };
    if !gateway_headers_are_bounded(req.headers()) {
        return gateway_rejection_response(
            StatusCode::BAD_REQUEST,
            &correlation_id,
            "Gateway request headers exceed the configured limit",
        );
    }
    // Bootstrap root key first (unchanged backward-compatible path), then the
    // DS0.5 scoped per-principal directory. Both yield a fully constructed
    // Principal; quota and concurrency identity derive from it downstream.
    let principal = match state.perimeter.authenticate(req.headers()) {
        Some(principal) => principal,
        None => match state.principal_keys.authenticate(req.headers()) {
            Some(principal) => principal,
            None => {
                return gateway_rejection_response(
                    StatusCode::UNAUTHORIZED,
                    &correlation_id,
                    "Valid gateway API key required",
                )
            }
        },
    };
    if !principal.roles.iter().any(|role| role == "inference") {
        return gateway_rejection_response(
            StatusCode::FORBIDDEN,
            &correlation_id,
            "Gateway principal lacks the inference role",
        );
    }
    let request_envelope = build_gateway_request_envelope(&req, &correlation_id);
    let resource = ResourceDescriptor::http_route(req.uri().path());
    let decision = match state
        .enterprise_hooks
        .policy
        .authorize(&AuthorizationRequest {
            principal: principal.clone(),
            action: EnterpriseAction::Inference,
            resource: resource.clone(),
            request: Some(request_envelope.clone()),
            metadata: HookMetadata::new(),
        })
        .await
    {
        Ok(decision) => decision,
        Err(err) => {
            warn!(error = %err, "Enterprise gateway policy hook failed");
            return gateway_rejection_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &correlation_id,
                "Gateway authorization service unavailable",
            );
        }
    };
    if !decision.allowed {
        record_denied_request(
            &state.enterprise_hooks,
            &principal,
            &resource,
            &correlation_id,
            "gateway inference policy denied request",
        )
        .await;
        return gateway_rejection_response(
            StatusCode::FORBIDDEN,
            &correlation_id,
            "Gateway inference permission denied",
        );
    }

    let trace = RequestTrace::resolve(&mut req).context;
    let context = RequestContext::new(correlation_id.clone(), principal.clone()).with_trace(trace);
    let stream_outcome = context.stream_outcome.clone();
    req.extensions_mut().insert(context);
    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    let mut response = with_request_id(correlation_id.clone(), next.run(req)).await;
    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    audit_at_body_end(
        response,
        PendingRequestAudit {
            hooks: state.enterprise_hooks.clone(),
            principal,
            resource,
            correlation_id,
            request: request_envelope,
            stream_outcome,
        },
    )
}

pub async fn attach_gateway_request_id(mut req: Request, next: Next) -> Response {
    RequestTrace::resolve(&mut req);
    let correlation_id = match resolve_gateway_correlation_id(&mut req) {
        Ok(correlation_id) => correlation_id,
        Err(generated_id) => {
            return gateway_rejection_response(
                StatusCode::BAD_REQUEST,
                &generated_id,
                "Invalid x-request-id header",
            )
        }
    };
    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    let mut response = with_request_id(correlation_id.clone(), next.run(req)).await;
    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

fn resolve_gateway_correlation_id(req: &mut Request) -> Result<String, String> {
    let mut request_ids = req.headers().get_all(REQUEST_ID_HEADER).iter();
    let supplied = request_ids.next();
    if request_ids.next().is_some() {
        return Err(Uuid::new_v4().to_string());
    }
    let correlation_id = match supplied {
        Some(value) => value
            .to_str()
            .ok()
            .map(str::trim)
            .filter(|value| valid_gateway_request_id(value))
            .map(str::to_string)
            .ok_or_else(|| Uuid::new_v4().to_string())?,
        None => Uuid::new_v4().to_string(),
    };
    Ok(correlation_id)
}

pub fn gateway_trace_request_id(req: &Request) -> &str {
    req.headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| valid_gateway_request_id(value))
        .unwrap_or("-")
}

fn valid_gateway_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_GATEWAY_REQUEST_ID_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn gateway_headers_are_bounded(headers: &axum::http::HeaderMap) -> bool {
    if headers.len() > MAX_GATEWAY_HEADER_COUNT {
        return false;
    }
    let mut total = 0usize;
    for (name, value) in headers {
        if value.as_bytes().len() > MAX_GATEWAY_HEADER_VALUE_BYTES {
            return false;
        }
        let Some(next) = total
            .checked_add(name.as_str().len())
            .and_then(|total| total.checked_add(value.as_bytes().len()))
        else {
            return false;
        };
        if next > MAX_GATEWAY_HEADER_BYTES {
            return false;
        }
        total = next;
    }
    true
}

fn build_gateway_request_envelope(req: &Request, correlation_id: &str) -> RequestEnvelope {
    const UNTRUSTED_IDENTITY_HEADERS: &[&str] =
        &["x-principal-id", "x-tenant-id", "x-scopes", "x-roles"];
    RequestEnvelope {
        correlation_id: correlation_id.to_string(),
        method: req.method().to_string(),
        path: req.uri().path().to_string(),
        headers: req
            .headers()
            .iter()
            .filter(|(name, _)| {
                name.as_str() != header::AUTHORIZATION.as_str()
                    && !UNTRUSTED_IDENTITY_HEADERS.contains(&name.as_str())
            })
            .map(|(name, value)| HeaderPair {
                name: name.as_str().to_string(),
                value: header_to_string(value).unwrap_or_default(),
            })
            .collect(),
        remote_addr: None,
    }
}

fn gateway_rejection_response(
    status: StatusCode,
    correlation_id: &str,
    message: &'static str,
) -> Response {
    use crate::error::ApiError;

    let mut response = match status {
        StatusCode::UNAUTHORIZED => ApiError::unauthorized(message),
        StatusCode::FORBIDDEN => ApiError::forbidden(message),
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::payload_too_large(message),
        StatusCode::TOO_MANY_REQUESTS => ApiError::too_many_requests(message),
        StatusCode::SERVICE_UNAVAILABLE => ApiError::service_unavailable(message),
        _ => ApiError::bad_request(message),
    }
    .into_response();
    if status == StatusCode::UNAUTHORIZED {
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"izwi-gateway\""),
        );
    }
    if let Ok(value) = HeaderValue::from_str(correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

/// The local role's request context: enterprise authentication, policy
/// and audit around every request, and the speech admission quota
/// (`speech_capacity` requests at once, at most `speech_per_tenant` of them
/// for one tenant) on speech-generating routes.
pub async fn attach_enterprise_request_context_with_hooks(
    hooks: &izwi_hooks::EnterpriseHooks,
    speech_capacity: Option<usize>,
    speech_per_tenant: Option<usize>,
    mut req: Request,
    next: Next,
) -> Response {
    let trace = RequestTrace::resolve(&mut req).context;
    let correlation_id = resolve_correlation_id(&mut req);
    let request_envelope = build_request_envelope(&req, &correlation_id);

    let principal = match hooks.auth.authenticate(&request_envelope).await {
        Ok(principal) => principal,
        Err(err) => {
            warn!(error = %err, "Enterprise authentication hook rejected request");
            return rejection_response(auth_failure_status(&err), &correlation_id, err);
        }
    };

    let resource = ResourceDescriptor::http_route(req.uri().path());
    let decision = match hooks
        .policy
        .authorize(&AuthorizationRequest {
            principal: principal.clone(),
            action: EnterpriseAction::HttpRequest,
            resource: resource.clone(),
            request: Some(request_envelope.clone()),
            metadata: HookMetadata::new(),
        })
        .await
    {
        Ok(decision) => decision,
        Err(err) => {
            warn!(error = %err, "Enterprise policy hook failed");
            return rejection_response(policy_failure_status(&err), &correlation_id, err);
        }
    };

    if !decision.allowed {
        record_denied_request(
            hooks,
            &principal,
            &resource,
            &correlation_id,
            decision
                .reason
                .as_deref()
                .unwrap_or("policy denied request"),
        )
        .await;
        return response_with_request_id(
            StatusCode::FORBIDDEN,
            &correlation_id,
            decision
                .reason
                .unwrap_or_else(|| "request denied by enterprise policy".to_string()),
        );
    }

    let speech_permit = if speech_admission::is_speech_generation(req.method(), req.uri().path()) {
        match speech_admission::acquire(&principal, speech_capacity.unwrap_or(1), speech_per_tenant)
        {
            Ok(permit) => Some(permit),
            Err(status) => {
                return response_with_request_id(
                    status,
                    &correlation_id,
                    if status == StatusCode::TOO_MANY_REQUESTS {
                        "speech tenant quota exceeded"
                    } else {
                        "speech serving capacity unavailable"
                    }
                    .to_string(),
                )
            }
        }
    } else {
        None
    };

    let context = RequestContext::new(correlation_id.clone(), principal.clone()).with_trace(trace);
    let stream_outcome = context.stream_outcome.clone();
    req.extensions_mut().insert(context);

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    let mut response = with_request_id(correlation_id.clone(), next.run(req)).await;

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    if let Some(permit) = speech_permit {
        response = speech_admission::guard_response(response, permit);
    }

    audit_at_body_end(
        response,
        PendingRequestAudit {
            hooks: hooks.clone(),
            principal,
            resource,
            correlation_id,
            request: request_envelope,
            stream_outcome,
        },
    )
}

/// The request audit, recorded once the response body has ended so a
/// streamed answer is audited with how it actually ended. The hooks run on
/// their own task: they never delay the response.
struct PendingRequestAudit {
    hooks: izwi_hooks::EnterpriseHooks,
    principal: Principal,
    resource: ResourceDescriptor,
    correlation_id: String,
    request: RequestEnvelope,
    stream_outcome: StreamOutcome,
}

impl PendingRequestAudit {
    fn finish(self, status: StatusCode, body: BodyEnd) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            warn!("No runtime to record the request audit on");
            return;
        };
        let outcome = if body != BodyEnd::Completed || self.stream_outcome.failed() {
            AuditOutcome::Failure
        } else if status.is_success() || status.is_redirection() {
            AuditOutcome::Success
        } else {
            AuditOutcome::Failure
        };
        runtime.spawn(async move {
            record_request_outcome(
                &self.hooks,
                &self.principal,
                &self.resource,
                &self.correlation_id,
                &self.request,
                status,
                outcome,
                body,
            )
            .await;
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyEnd {
    Completed,
    Failed,
    /// Dropped before its end, e.g. the client disconnected mid-stream.
    Interrupted,
}

impl BodyEnd {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }
}

fn audit_at_body_end(response: Response, audit: PendingRequestAudit) -> Response {
    let status = response.status();
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        axum::body::Body::new(AuditedBody {
            inner: body,
            status,
            audit: Some(audit),
        }),
    )
}

/// Forwards a response body unchanged (size hint included) and records the
/// request audit when it ends.
struct AuditedBody {
    inner: axum::body::Body,
    status: StatusCode,
    audit: Option<PendingRequestAudit>,
}

impl AuditedBody {
    fn finish(&mut self, end: BodyEnd) {
        if let Some(audit) = self.audit.take() {
            audit.finish(self.status, end);
        }
    }
}

impl http_body::Body for AuditedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let polled = Pin::new(&mut self.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(None) => self.finish(BodyEnd::Completed),
            Poll::Ready(Some(Err(_))) => self.finish(BodyEnd::Failed),
            // A sized body may never be polled past its last frame.
            Poll::Ready(Some(Ok(_))) if self.inner.is_end_stream() => {
                self.finish(BodyEnd::Completed)
            }
            _ => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for AuditedBody {
    fn drop(&mut self) {
        let end = if self.inner.is_end_stream() {
            BodyEnd::Completed
        } else {
            BodyEnd::Interrupted
        };
        self.finish(end);
    }
}

/// Use the caller's x-request-id only when it passes the gateway's bounded
/// charset; anything else (oversized, control or log-breaking characters,
/// duplicates) is replaced with a fresh id instead of being echoed into logs,
/// headers and engine correlation.
fn resolve_correlation_id(req: &mut Request) -> String {
    let correlation_id = resolve_gateway_correlation_id(req).unwrap_or_else(|generated| generated);

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    correlation_id
}

fn build_request_envelope(req: &Request, correlation_id: &str) -> RequestEnvelope {
    RequestEnvelope {
        correlation_id: correlation_id.to_string(),
        method: req.method().to_string(),
        path: req.uri().path().to_string(),
        headers: req
            .headers()
            .iter()
            .map(|(name, value)| HeaderPair {
                name: name.as_str().to_string(),
                value: header_to_string(value).unwrap_or_default(),
            })
            .collect(),
        remote_addr: None,
    }
}

async fn record_denied_request(
    hooks: &izwi_hooks::EnterpriseHooks,
    principal: &Principal,
    resource: &ResourceDescriptor,
    correlation_id: &str,
    reason: &str,
) {
    let mut metadata = HookMetadata::new();
    metadata.insert("reason".to_string(), reason.to_string());

    let event = AuditEvent {
        category: AuditCategory::Security,
        action: EnterpriseAction::HttpRequest,
        outcome: AuditOutcome::Denied,
        principal: Some(principal.clone()),
        resource: Some(resource.clone()),
        correlation_id: Some(correlation_id.to_string()),
        metadata,
    };

    if let Err(err) = hooks.audit.record(event).await {
        warn!(error = %err, "Enterprise audit hook failed for denied request");
    }
}

#[allow(clippy::too_many_arguments)]
async fn record_request_outcome(
    hooks: &izwi_hooks::EnterpriseHooks,
    principal: &Principal,
    resource: &ResourceDescriptor,
    correlation_id: &str,
    request: &RequestEnvelope,
    status: StatusCode,
    outcome: AuditOutcome,
    body: BodyEnd,
) {
    let mut metadata = HookMetadata::new();
    metadata.insert("method".to_string(), request.method.clone());
    metadata.insert("path".to_string(), request.path.clone());
    metadata.insert("status".to_string(), status.as_u16().to_string());
    metadata.insert("body".to_string(), body.as_str().to_string());

    let audit_event = AuditEvent {
        category: AuditCategory::Request,
        action: EnterpriseAction::HttpRequest,
        outcome,
        principal: Some(principal.clone()),
        resource: Some(resource.clone()),
        correlation_id: Some(correlation_id.to_string()),
        metadata: metadata.clone(),
    };

    if let Err(err) = hooks.audit.record(audit_event).await {
        warn!(error = %err, "Enterprise audit hook failed for request");
    }

    let observability_event = ObservabilityEvent {
        kind: ObservabilityEventKind::Request,
        name: "http_request".to_string(),
        principal: Some(principal.clone()),
        correlation_id: Some(correlation_id.to_string()),
        attributes: metadata,
    };

    if let Err(err) = hooks.observability.record(observability_event).await {
        warn!(error = %err, "Enterprise observability hook failed for request");
    }
}

fn auth_failure_status(err: &HookError) -> StatusCode {
    match err {
        HookError::Denied(_) => StatusCode::UNAUTHORIZED,
        HookError::NotFound(_) => StatusCode::INTERNAL_SERVER_ERROR,
        HookError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn policy_failure_status(err: &HookError) -> StatusCode {
    match err {
        HookError::Denied(_) => StatusCode::FORBIDDEN,
        HookError::NotFound(_) => StatusCode::INTERNAL_SERVER_ERROR,
        HookError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn rejection_response(status: StatusCode, correlation_id: &str, err: HookError) -> Response {
    response_with_request_id(status, correlation_id, err.to_string())
}

fn response_with_request_id(
    status: StatusCode,
    correlation_id: &str,
    body: impl Into<String>,
) -> Response {
    let mut response = (status, body.into()).into_response();
    if let Ok(value) = HeaderValue::from_str(correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

fn header_to_string(value: &HeaderValue) -> Option<String> {
    value
        .to_str()
        .map(str::to_string)
        .ok()
        .or_else(|| Some(String::from_utf8_lossy(value.as_bytes()).to_string()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn scheduling_tenant_identity_is_trusted_namespaced_and_anonymous_stable() {
        let mut context = super::RequestContext::new(
            "untrusted-header".into(),
            izwi_hooks::Principal::local_anonymous(),
        );
        assert_eq!(context.tenant_key(), None);
        context.principal.id = "a".into();
        let individual = context.tenant_key();
        context.principal.tenant_id = Some("a".into());
        let tenant = context.tenant_key();
        assert_ne!(individual, tenant);
        context.principal.id = "another-member".into();
        context.correlation_id = "different-header".into();
        assert_eq!(context.tenant_key(), tenant);
        assert_eq!(context.tenant_scope(), "tenant:a");
    }

    use super::*;
    use axum::{
        body::Body,
        extract::Extension,
        http::{header::HeaderName, HeaderMap, Request},
        middleware,
        routing::get,
        Router,
    };
    use tower::Service;

    async fn echo_context_and_header(
        headers: HeaderMap,
        Extension(context): Extension<RequestContext>,
    ) -> String {
        let header = headers
            .get(REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        format!("{}|{}", context.correlation_id, header)
    }

    #[tokio::test]
    async fn generated_request_id_is_inserted_and_returned() {
        let response = send_request(
            Router::new()
                .route("/", get(echo_context_and_header))
                .layer(middleware::from_fn(|req, next| async move {
                    let hooks = izwi_hooks::EnterpriseHooks::noop();
                    attach_enterprise_request_context_with_hooks(&hooks, None, None, req, next)
                        .await
                })),
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("request id header should exist")
            .to_str()
            .expect("request id should be ascii")
            .to_string();
        Uuid::parse_str(&request_id).expect("generated request id should be a UUID");

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert_eq!(body, format!("{request_id}|{request_id}").as_bytes());
    }

    #[tokio::test]
    async fn local_request_ids_are_bounded_like_the_gateway() {
        let hooks = izwi_hooks::EnterpriseHooks::noop();
        let app = Router::new()
            .route("/", get(echo_context_and_header))
            .layer(middleware::from_fn(move |req, next| {
                let hooks = hooks.clone();
                async move {
                    attach_enterprise_request_context_with_hooks(&hooks, None, None, req, next)
                        .await
                }
            }));
        let echo = |supplied: String| {
            let app = app.clone();
            async move {
                let response = send_request(
                    app,
                    Request::builder()
                        .uri("/")
                        .header(HeaderName::from_static(REQUEST_ID_HEADER), supplied)
                        .body(Body::empty())
                        .expect("request should build"),
                )
                .await;
                let returned = response.headers()[REQUEST_ID_HEADER]
                    .to_str()
                    .unwrap()
                    .to_string();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("body should read");
                (returned, String::from_utf8(body.to_vec()).unwrap())
            }
        };

        let (returned, body) = echo("client-request-123".to_string()).await;
        assert_eq!(returned, "client-request-123");
        assert_eq!(body, "client-request-123|client-request-123");

        for invalid in ["has spaces in it".to_string(), "x".repeat(129)] {
            let (returned, body) = echo(invalid.clone()).await;
            assert_ne!(returned, invalid);
            Uuid::parse_str(&returned).expect("invalid ids are replaced with a UUID");
            assert_eq!(body, format!("{returned}|{returned}"));
        }
    }

    /// Request audits, delivered after `delay`.
    struct SlowAudit {
        delay: Duration,
        events: tokio::sync::mpsc::UnboundedSender<AuditEvent>,
    }

    #[async_trait::async_trait]
    impl izwi_hooks::AuditSink for SlowAudit {
        async fn record(&self, event: AuditEvent) -> izwi_hooks::HookResult<()> {
            tokio::time::sleep(self.delay).await;
            let _ = self.events.send(event);
            Ok(())
        }
    }

    fn audited_app(
        delay: Duration,
        route: axum::routing::MethodRouter,
    ) -> (Router, tokio::sync::mpsc::UnboundedReceiver<AuditEvent>) {
        let (events, received) = tokio::sync::mpsc::unbounded_channel();
        let mut hooks = izwi_hooks::EnterpriseHooks::noop();
        hooks.audit = std::sync::Arc::new(SlowAudit { delay, events });
        let app = Router::new()
            .route("/", route)
            .layer(middleware::from_fn(move |req, next| {
                let hooks = hooks.clone();
                async move {
                    attach_enterprise_request_context_with_hooks(&hooks, None, None, req, next)
                        .await
                }
            }));
        (app, received)
    }

    fn request_audit(event: AuditEvent) -> Option<AuditEvent> {
        (event.category == izwi_hooks::AuditCategory::Request).then_some(event)
    }

    #[tokio::test]
    async fn request_audit_runs_off_the_response_path() {
        let (app, mut audits) = audited_app(Duration::from_secs(30), get(|| async { "ok" }));
        let started = Instant::now();
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            send_request(
                app,
                Request::builder().uri("/").body(Body::empty()).unwrap(),
            ),
        )
        .await
        .expect("a slow audit hook never delays the response");
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body, "ok");
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(audits.try_recv().is_err(), "the hook is still running");
    }

    /// A stream that reports a failure in-band while answering 200.
    async fn failing_stream(Extension(context): Extension<RequestContext>) -> Response {
        let body = async_stream::stream! {
            yield Ok::<_, std::convert::Infallible>(axum::body::Bytes::from("data: part\n\n"));
            tokio::time::sleep(Duration::from_millis(50)).await;
            context.stream_outcome.record_failure();
            yield Ok(axum::body::Bytes::from("data: {\"error\":\"boom\"}\n\n"));
        };
        Response::new(Body::from_stream(body))
    }

    #[tokio::test]
    async fn streamed_answers_are_audited_when_the_body_ends_with_the_in_band_outcome() {
        let (app, mut audits) = audited_app(Duration::ZERO, get(failing_stream));
        let response = send_request(
            app,
            Request::builder().uri("/").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            audits.try_recv().is_err(),
            "nothing is audited while the stream runs"
        );
        let _ = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let event = loop {
            let event = tokio::time::timeout(Duration::from_secs(5), audits.recv())
                .await
                .expect("audited at body end")
                .expect("audit sink open");
            if let Some(event) = request_audit(event) {
                break event;
            }
        };
        assert_eq!(event.outcome, AuditOutcome::Failure);
        assert_eq!(event.metadata["status"], "200");
        assert_eq!(event.metadata["body"], "completed");
    }

    #[tokio::test]
    async fn a_body_dropped_mid_stream_is_audited_as_interrupted() {
        let (app, mut audits) = audited_app(Duration::ZERO, get(failing_stream));
        let response = send_request(
            app,
            Request::builder().uri("/").body(Body::empty()).unwrap(),
        )
        .await;
        let mut body = response.into_body().into_data_stream();
        let _first = futures::StreamExt::next(&mut body).await;
        drop(body);
        let event = loop {
            let event = tokio::time::timeout(Duration::from_secs(5), audits.recv())
                .await
                .expect("audited when the body is dropped")
                .expect("audit sink open");
            if let Some(event) = request_audit(event) {
                break event;
            }
        };
        assert_eq!(event.outcome, AuditOutcome::Failure);
        assert_eq!(event.metadata["body"], "interrupted");
    }

    #[tokio::test]
    async fn sized_responses_are_audited_as_completed_and_keep_their_length() {
        let (app, mut audits) = audited_app(Duration::ZERO, get(|| async { "sized" }));
        let response = send_request(
            app,
            Request::builder().uri("/").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(
            http_body::Body::size_hint(response.body()).exact(),
            Some(5),
            "the audit wrapper keeps the body's exact length"
        );
        let _ = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let event = loop {
            let event = tokio::time::timeout(Duration::from_secs(5), audits.recv())
                .await
                .expect("audited")
                .expect("audit sink open");
            if let Some(event) = request_audit(event) {
                break event;
            }
        };
        assert_eq!(event.outcome, AuditOutcome::Success);
        assert_eq!(event.metadata["body"], "completed");
    }

    async fn send_request(mut app: Router, request: Request<Body>) -> Response {
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("request should succeed")
    }
}

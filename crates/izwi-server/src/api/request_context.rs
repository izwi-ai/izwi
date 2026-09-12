#[path = "speech_admission.rs"]
mod speech_admission;

use axum::extract::Request;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use izwi_hooks::{
    AuditCategory, AuditEvent, AuditOutcome, AuthorizationRequest, EnterpriseAction, HeaderPair,
    HookError, HookMetadata, ObservabilityEvent, ObservabilityEventKind, Principal,
    RequestEnvelope, ResourceDescriptor,
};
use std::time::{Duration, Instant};
use tracing::warn;
use uuid::Uuid;

use crate::gateway::GatewayState;
use crate::state::AppState;

const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone, Debug)]
pub struct RequestContext {
    pub correlation_id: String,
    #[allow(dead_code)]
    pub principal: Principal,
    /// Receiver-local ingress time used to preserve the original end-to-end
    /// request budget when work crosses another process boundary.
    received_at: Instant,
}

impl RequestContext {
    pub(crate) fn new(correlation_id: String, principal: Principal) -> Self {
        Self {
            correlation_id,
            principal,
            received_at: Instant::now(),
        }
    }

    /// Scheduling identity comes exclusively from the authenticated extension.
    /// Legacy durable jobs and the local anonymous principal share None.
    pub(crate) fn tenant_key(&self) -> Option<[u8; 32]> {
        if self.principal == Principal::local_anonymous() {
            return None;
        }
        use sha2::{Digest, Sha256};
        Some(Sha256::digest(principal_namespace(&self.principal).as_bytes()).into())
    }

    pub(crate) fn remaining_budget(&self, total: Duration) -> Option<Duration> {
        total.checked_sub(self.received_at.elapsed())
    }
}

pub(super) fn principal_namespace(principal: &Principal) -> String {
    match &principal.tenant_id {
        Some(tenant) => format!("tenant:{tenant}"),
        None => format!("principal:{}", principal.id),
    }
}

#[allow(dead_code)]
pub async fn attach_request_context(mut req: Request, next: Next) -> Response {
    let correlation_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(header_to_string)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    req.extensions_mut().insert(RequestContext::new(
        correlation_id.clone(),
        Principal::local_anonymous(),
    ));

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    let mut response = next.run(req).await;
    if let Ok(value) = correlation_id.parse() {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

pub async fn attach_enterprise_request_context(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    attach_enterprise_request_context_with_hooks(
        &state.enterprise_hooks,
        Some(state.request_admission_snapshot().global.capacity),
        req,
        next,
    )
    .await
}

pub async fn attach_gateway_request_context(
    State(state): State<GatewayState>,
    req: Request,
    next: Next,
) -> Response {
    attach_enterprise_request_context_with_hooks(&state.enterprise_hooks, None, req, next).await
}

async fn attach_enterprise_request_context_with_hooks(
    hooks: &izwi_hooks::EnterpriseHooks,
    speech_capacity: Option<usize>,
    mut req: Request,
    next: Next,
) -> Response {
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
        match speech_admission::acquire(&principal, speech_capacity.unwrap_or(1)) {
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

    req.extensions_mut().insert(RequestContext::new(
        correlation_id.clone(),
        principal.clone(),
    ));

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    let mut response = next.run(req).await;
    let status = response.status();

    record_request_outcome(
        hooks,
        &principal,
        &resource,
        &correlation_id,
        &request_envelope,
        status,
    )
    .await;

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    if let Some(permit) = speech_permit {
        response = speech_admission::guard_response(response, permit);
    }

    response
}

fn resolve_correlation_id(req: &mut Request) -> String {
    let correlation_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(header_to_string)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

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

async fn record_request_outcome(
    hooks: &izwi_hooks::EnterpriseHooks,
    principal: &Principal,
    resource: &ResourceDescriptor,
    correlation_id: &str,
    request: &RequestEnvelope,
    status: StatusCode,
) {
    let mut metadata = HookMetadata::new();
    metadata.insert("method".to_string(), request.method.clone());
    metadata.insert("path".to_string(), request.path.clone());
    metadata.insert("status".to_string(), status.as_u16().to_string());

    let outcome = if status.is_success() || status.is_redirection() {
        AuditOutcome::Success
    } else {
        AuditOutcome::Failure
    };

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

    async fn echo_context(Extension(context): Extension<RequestContext>) -> String {
        context.correlation_id
    }

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
                .layer(middleware::from_fn(attach_request_context)),
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
    async fn supplied_request_id_is_preserved() {
        let supplied = "client-request-123";
        let response = send_request(
            Router::new()
                .route("/", get(echo_context))
                .layer(middleware::from_fn(attach_request_context)),
            Request::builder()
                .uri("/")
                .header(HeaderName::from_static(REQUEST_ID_HEADER), supplied)
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            response
                .headers()
                .get(REQUEST_ID_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some(supplied)
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert_eq!(body, supplied.as_bytes());
    }

    async fn send_request(mut app: Router, request: Request<Body>) -> Response {
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("request should succeed")
    }
}

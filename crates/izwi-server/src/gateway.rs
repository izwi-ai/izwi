//! Hardware-independent public gateway state and routes.
//!
//! This module deliberately has no `RuntimeService`, persistence, model, or
//! accelerator initialization path. Its only execution dependency is the
//! bounded private worker client wrapped by `RemoteChatExecution`.

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_core::ServeRuntimeConfig;
use izwi_hooks::EnterpriseHooks;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::trace::TraceLayer;
use tracing::{field, info, info_span, warn, Span};

use crate::api::request_context::attach_gateway_request_context;
use crate::app::chat::RemoteChatExecution;
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::state::ServerLifecycle;

#[derive(Clone)]
pub struct GatewayState {
    pub remote_chat_execution: RemoteChatExecution,
    pub enterprise_hooks: EnterpriseHooks,
    pub lifecycle: ServerLifecycle,
    pub request_timeout_secs: u64,
    request_admission: Arc<Semaphore>,
}

impl GatewayState {
    pub fn new(
        remote_chat_execution: RemoteChatExecution,
        enterprise_hooks: EnterpriseHooks,
        request_timeout_secs: u64,
        max_in_flight: usize,
    ) -> Self {
        debug_assert!(max_in_flight > 0);
        Self {
            remote_chat_execution,
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
        }
    }
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
    let worker_result = state.remote_chat_execution.readiness_check().await;
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

async fn bounded_gateway_admission(
    State(state): State<GatewayState>,
    request: Request,
    next: Next,
) -> Response {
    let Ok(_permit) = state.request_admission.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Gateway request capacity is currently unavailable",
        )
            .into_response();
    };
    next.run(request).await
}

pub fn create_gateway_router(state: GatewayState, serve_config: &ServeRuntimeConfig) -> Router {
    let middleware_state = state.clone();
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request| {
            let request_id = request
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("-");
            info_span!(
                "http_request",
                service = SERVICE_NAME,
                version = SERVICE_VERSION,
                method = %request.method(),
                path = %request.uri().path(),
                uri = %request.uri(),
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
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            bounded_gateway_admission,
        ))
        .fallback(api_not_found);
    let app = Router::new()
        .route("/livez", get(live_check))
        .route("/readyz", get(ready_check))
        .route(
            "/openapi.json",
            get(crate::api::openapi::gateway_openapi_json),
        )
        .merge(crate::api::docs::router())
        .nest("/v1", v1_routes)
        .fallback(api_not_found)
        .with_state(state);

    // Gateway mode never serves the desktop SPA because its fallback would
    // make unmigrated API paths appear successful.
    let mut gateway_contract = serve_config.clone();
    gateway_contract.ui_enabled = false;
    crate::api::apply_runtime_contract(app, &gateway_contract)
        .layer(trace_layer)
        .layer(middleware::from_fn_with_state(
            middleware_state,
            attach_gateway_request_context,
        ))
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
    use izwi_core::ModelVariant;
    use izwi_serving_client::{
        mock::{MockWorker, MockWorkerConfig},
        WorkerClient, WorkerClientConfig,
    };
    use izwi_serving_protocol::{ModelAlias, ModelGeneration, PolicyRevision};
    use serde_json::json;
    use tower::Service;

    async fn send(mut app: Router, request: Request<Body>) -> Response {
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
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), 2, 4);
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
        assert_eq!(
            send(app.clone(), get("/v1/models")).await.status(),
            StatusCode::NOT_FOUND,
            "gateway mode must not expose an unmigrated local route or UI fallback"
        );

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
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), 2, 4);
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
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, EnterpriseHooks::noop(), 2, 1);
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
}

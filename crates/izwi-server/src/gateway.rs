//! Hardware-independent public gateway state and routes.
//!
//! This module deliberately has no `RuntimeService`, persistence, model, or
//! accelerator initialization path. Its only execution dependency is the
//! bounded private worker client wrapped by `RemoteChatExecution`.

use axum::{
    extract::{DefaultBodyLimit, Request, State},
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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{field, info, info_span, warn, Span};

use crate::api::request_context::{attach_gateway_request_context, attach_gateway_request_id};
use crate::app::chat::RemoteChatExecution;
use crate::app::remote_chat_dispatch::RemoteChatDispatcher;
use crate::error::ApiError;
use crate::gateway_security::GatewayPerimeterConfig;
use crate::logging::{SERVICE_NAME, SERVICE_VERSION};
use crate::state::ServerLifecycle;

#[derive(Clone)]
pub struct GatewayState {
    pub(crate) chat_execution: GatewayChatExecution,
    pub(crate) enterprise_hooks: EnterpriseHooks,
    pub(crate) lifecycle: ServerLifecycle,
    pub(crate) perimeter: GatewayPerimeterConfig,
    pub request_timeout_secs: u64,
    request_admission: Arc<Semaphore>,
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
        Self {
            chat_execution: GatewayChatExecution::Pinned(remote_chat_execution),
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            perimeter,
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
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
        Self {
            chat_execution: GatewayChatExecution::Registry(dispatcher),
            enterprise_hooks,
            lifecycle: ServerLifecycle::new(),
            perimeter,
            request_timeout_secs: request_timeout_secs.max(1),
            request_admission: Arc::new(Semaphore::new(max_in_flight)),
        }
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
        ));
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

    apply_gateway_cors(app, serve_config)
        .layer(trace_layer)
        .layer(middleware::from_fn(attach_gateway_request_id))
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
        AuthorizationDecision, AuthorizationRequest, EnterpriseAction, HookResult, PolicyEngine,
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

    use crate::app::remote_chat_dispatch::RemoteChatDispatchConfig;
    use crate::gateway_security::MAX_GATEWAY_REQUEST_ID_BYTES;
    use crate::worker_registry::{
        ApprovedDeployment, ApprovedWorker, BackendPolicy, WorkerRegistry, WorkerRegistryConfig,
    };

    const TEST_API_KEY: &str = "test-public-api-key-123456";

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

    async fn registry_gateway_state(model: ModelVariant) -> (GatewayState, MockWorker) {
        let worker_config = MockWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            ..MockWorkerConfig::default()
        };
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
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());

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

        let still_draining = send(app, stream_request()).await;
        assert_eq!(still_draining.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn gateway_v1_requires_api_key_while_probes_and_docs_remain_public() {
        let state = unreachable_gateway_state(test_perimeter());
        let app = create_gateway_router(state, &ServeRuntimeConfig::default());

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
        let app = create_gateway_router(
            unreachable_gateway_state(perimeter),
            &ServeRuntimeConfig::default(),
        );
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
    }

    #[tokio::test]
    async fn gateway_request_id_is_bounded_and_validated_before_tracing() {
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
            },
        )
        .expect("remote execution should initialize");
        let state = GatewayState::new(remote, hooks, perimeter, 2, 1);
        state.lifecycle.mark_ready();
        state
    }
}

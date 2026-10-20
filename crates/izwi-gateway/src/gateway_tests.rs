use super::*;
use axum::{body::Body, http::Request};
use futures::StreamExt;
use izwi_hooks::{
    AuthorizationDecision, AuthorizationRequest, EnterpriseAction, HookError, HookResult,
    PolicyEngine, QuotaDecision, QuotaLimiter, QuotaRequest,
};
use izwi_model_catalog::ModelVariant;
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    DeploymentId, ModelAlias, ModelGeneration, PolicyRevision, WorkerStatus,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Mutex;
use tower::Service;

use crate::app::remote_chat_dispatch::RemoteChatDispatchConfig;
use crate::gateway_security::MAX_GATEWAY_REQUEST_ID_BYTES;
use crate::test_support::{single_worker_dispatcher, single_worker_dispatcher_with_client};
use crate::worker_registry::{
    ApprovedDeployment, ApprovedWorker, BackendPolicy, WorkerRegistry, WorkerRegistryConfig,
    WorkerRegistryError,
};
use izwi_serving_protocol::InvocationErrorCode;
use izwi_serving_worker::test_support::{TestWorker, TestWorkerConfig, WorkerScript};

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
                "model": ModelVariant::Qwen354BGguf.dir_name(),
                "messages": [{"role": "user", "content": "quota test"}],
                "max_tokens": max_tokens
            })
            .to_string(),
        ))
        .expect("chat request should build")
}

fn chat_request_with_bearer(bearer: &str, max_tokens: usize) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": ModelVariant::Qwen354BGguf.dir_name(),
                "messages": [{"role": "user", "content": "scoped principal test"}],
                "max_tokens": max_tokens
            })
            .to_string(),
        ))
        .expect("chat request should build")
}

fn test_generation(prompt_tokens: usize, tokens_generated: usize) -> ChatGeneration {
    ChatGeneration {
        latency_breakdown: None,
        finish_reason: None,
        text: String::new(),
        prompt_tokens,
        tokens_generated,
        generation_time_ms: 0.0,
        cached_prompt_tokens: None,
        logprobs: Vec::new(),
    }
}

#[test]
fn fixed_gateway_metrics_track_lifetimes_latency_and_status_classes() {
    let metrics = GatewayMetrics::default();
    let request = metrics.begin_request();
    let mut stream = metrics.begin_stream("model-a", Instant::now());
    metrics.record_dispatch(
        Duration::from_micros(7),
        Some(&ApiError::service_unavailable("down")),
    );
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
    stream.record_delta();
    stream.record_delta();
    stream.record_delta();
    stream.record_completion(&test_generation(11, 3));
    drop(stream);
    drop(request);
    assert_eq!(metrics.snapshot().active_requests, 0);
    assert_eq!(metrics.snapshot().active_streams, 0);
    let abandoned_stream = metrics.begin_stream("model-a", Instant::now());
    drop(abandoned_stream);
    assert_eq!(metrics.snapshot().routing_dispatch_failures, 2);

    let text = metrics.render_prometheus(0);
    let counts = izwi_prometheus::check_histograms(&text).expect("valid histogram exposition");
    let duration = |outcome: &str| {
        counts[&format!(
            "izwi_gateway_chat_request_duration_seconds{{model=\"model-a\",outcome=\"{outcome}\",stream=\"true\"}}"
        )]
    };
    assert_eq!(duration("ok"), 1);
    // A body dropped before its terminal event is an error.
    assert_eq!(duration("error"), 1);
    assert_eq!(
        counts["izwi_gateway_chat_ttft_seconds{model=\"model-a\"}"],
        1
    );
    // Three deltas: one first output and two gaps.
    assert_eq!(
        counts["izwi_gateway_chat_inter_token_latency_seconds{model=\"model-a\"}"],
        2
    );
    assert_eq!(
        counts["izwi_gateway_worker_dispatch_seconds{outcome=\"503\"}"],
        1
    );
    assert!(text.contains("izwi_gateway_chat_prompt_tokens_total{model=\"model-a\"} 11"));
    assert!(text.contains("izwi_gateway_chat_completion_tokens_total{model=\"model-a\"} 3"));
    assert!(text.len() <= MAX_PROMETHEUS_RESPONSE_BYTES);
}

#[test]
fn gateway_metrics_stay_bounded_under_label_churn() {
    let metrics = GatewayMetrics::default();
    let models = (0..100)
        .map(|index| &*Box::leak(format!("model-with-a-long-name-{index:03}").into_boxed_str()))
        .collect::<Vec<&'static str>>();
    for model in &models {
        let mut stream = metrics.begin_stream(model, Instant::now());
        stream.record_delta();
        stream.record_delta();
        stream.record_completion(&test_generation(1, 1));
        stream.record_failure();
        metrics.record_dispatch(Duration::from_millis(1), None);
    }
    for code in 100..600u16 {
        let status = StatusCode::from_u16(code).expect("valid status");
        metrics.record_http_response("/v1/unknown/path", status, Duration::from_millis(1));
        metrics.record_http_response("/livez", status, Duration::from_millis(1));
    }
    let text = metrics.render_prometheus(u64::MAX);
    assert!(
        text.len() <= MAX_PROMETHEUS_RESPONSE_BYTES,
        "{} bytes",
        text.len()
    );
    let counts = izwi_prometheus::check_histograms(&text).unwrap();
    // Unknown paths share one route label; extra label sets share `other`.
    assert!(!text.contains("/v1/unknown/path"));
    assert!(counts
        .keys()
        .any(|series| series.contains("model=\"other\"")));
    let ttft_series = counts
        .keys()
        .filter(|series| series.starts_with("izwi_gateway_chat_ttft_seconds{"))
        .count();
    assert_eq!(ttft_series, GATEWAY_MAX_SERIES + 1);
}

async fn registry_gateway_state(model: ModelVariant) -> (GatewayState, TestWorker) {
    registry_gateway_state_with_config(
        model,
        TestWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            ..TestWorkerConfig::default()
        },
    )
    .await
}

async fn registry_gateway_state_with_config(
    model: ModelVariant,
    worker_config: TestWorkerConfig,
) -> (GatewayState, TestWorker) {
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    let dispatcher = single_worker_dispatcher(model, &worker.endpoint(), credentials).await;
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
    state.lifecycle.mark_ready();
    (state, worker)
}

#[tokio::test]
async fn public_chat_nonstream_and_stream_use_registry_dispatcher() {
    // A model that answers without thinking by default, as the test worker does.
    let model = ModelVariant::Qwen352BGguf;
    let (state, _worker) = registry_gateway_state(model).await;
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

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
        "deterministic test response"
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
    assert!(body.contains("deterministic test response"));
    assert!(body.contains("data: [DONE]"));
    let metrics = state.metrics.snapshot();
    assert_eq!(metrics.dispatch_calls, 2);
    assert_eq!(metrics.routing_dispatch_failures, 0);
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.active_streams, 0);
}

/// EDGE-F5 through the gateway: the worker streams the model's tool-call
/// markup as text, split mid-tag, and the gateway sends it as
/// `delta.tool_calls` with no markup in `delta.content`.
#[tokio::test]
async fn gateway_streams_worker_tool_markup_as_tool_calls() {
    // A model that answers without thinking by default, as the test worker does.
    let model = ModelVariant::Qwen352BGguf;
    let deltas = [
        "Checking.\n<tool",
        "_call>\n<function=get_",
        "weather>\n<parameter=city>\nPar",
        "is\n</parameter>\n</func",
        "tion>\n</tool_call>",
    ];
    let (state, _worker) = registry_gateway_state_with_config(
        model,
        TestWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            script: WorkerScript::TextDeltas(deltas.map(str::to_owned).to_vec()),
            ..TestWorkerConfig::default()
        },
    )
    .await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": model.dir_name(),
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": true,
                "tools": [{"type": "function", "function": {"name": "get_weather"}}]
            })
            .to_string(),
        ))
        .expect("streaming request should build");
    let response = send(app, request).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("SSE response should be bounded");
    let body = String::from_utf8(body.to_vec()).expect("SSE should be UTF-8");
    let (mut content, mut name, mut arguments, mut finish) =
        (String::new(), None, String::new(), None);
    for data in body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
    {
        let chunk: serde_json::Value = serde_json::from_str(data).expect("chunk JSON");
        let choice = &chunk["choices"][0];
        content.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        if let Some(call) = choice["delta"]["tool_calls"].get(0) {
            if let Some(call_name) = call["function"]["name"].as_str() {
                name = Some(call_name.to_owned());
            }
            arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            finish = Some(reason.to_owned());
        }
    }
    assert_eq!(content, "Checking.", "{body}");
    assert_eq!(name.as_deref(), Some("get_weather"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).expect("arguments JSON"),
        json!({"city": "Paris"})
    );
    assert_eq!(finish.as_deref(), Some("tool_calls"));
}

/// A model failure after the stream started reaches the client as an
/// OpenAI error event with the worker's error class, not a generic one.
#[tokio::test]
async fn gateway_stream_errors_keep_the_worker_error_type() {
    let model = ModelVariant::Qwen354BGguf;
    let (state, _worker) = registry_gateway_state_with_config(
        model,
        TestWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            script: WorkerScript::TextDeltaThenFail(InvocationErrorCode::InvalidInput),
            ..TestWorkerConfig::default()
        },
    )
    .await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    let request = chat_request(json!({
        "model": model.dir_name(),
        "messages": [{"role": "user", "content": "fail"}],
        "stream": true
    }));
    let response = send(app, request).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("SSE response should be bounded");
    let body = String::from_utf8(body.to_vec()).expect("SSE should be UTF-8");
    let events: Vec<&str> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{body}");
    let error: serde_json::Value =
        serde_json::from_str(events[events.len() - 2]).expect("error event JSON");
    assert_eq!(
        error["error"]["message"], "scripted model failure",
        "{body}"
    );
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(body.contains("deterministic test response"));
}

/// EDGE-F10: the standard client bootstrap (`GET /v1/models`) works
/// against the gateway, behind the same key as inference.
#[tokio::test]
async fn gateway_lists_its_public_model_behind_the_api_key() {
    let model = ModelVariant::Qwen354BGguf;
    let (state, _worker) = registry_gateway_state(model).await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let response = send(app.clone(), get("/v1/models")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_body(response).await,
        json!({
            "object": "list",
            "data": [{
                "id": model.dir_name(), "object": "model", "created": 0, "owned_by": "izwi"
            }]
        })
    );
    let response = send(
        app.clone(),
        get(&format!("/v1/models/{}", model.dir_name())),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_body(response).await["id"], model.dir_name());

    for unknown in ["/v1/models/gpt-imaginary", "/v1/models/Qwen3-8B-GGUF"] {
        let response = send(app.clone(), get(unknown)).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{unknown}");
        assert_eq!(
            json_body(response).await["error"]["code"],
            "model_not_found"
        );
    }

    let unauthenticated = send_raw(app, get("/v1/models")).await;
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn gateway_chat_histograms_count_every_dispatched_request() {
    let model = ModelVariant::Qwen354BGguf;
    let (state, _worker) = registry_gateway_state(model).await;
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
    let body = |stream: bool| {
        json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "metrics"}],
            "stream": stream,
            "stream_options": stream.then(|| json!({"include_usage": true})),
            "max_tokens": 32
        })
    };
    let response = send(app.clone(), chat_request(body(false))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let usage = json_body(response).await["usage"].clone();
    for _ in 0..2 {
        let response = send(app.clone(), chat_request(body(true))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let text = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("bounded SSE body");
        assert!(String::from_utf8_lossy(&text).contains("[DONE]"));
    }

    let text = state.metrics.render_prometheus(0);
    let counts = izwi_prometheus::check_histograms(&text).expect("valid histogram exposition");
    let model = model.dir_name();
    let duration = |stream: &str| {
        counts[&format!(
            "izwi_gateway_chat_request_duration_seconds{{model=\"{model}\",outcome=\"ok\",stream=\"{stream}\"}}"
        )]
    };
    assert_eq!(duration("false"), 1);
    assert_eq!(duration("true"), 2);
    assert_eq!(
        counts[&format!("izwi_gateway_chat_ttft_seconds{{model=\"{model}\"}}")],
        2
    );
    assert_eq!(
        counts["izwi_gateway_worker_dispatch_seconds{outcome=\"accepted\"}"],
        3
    );
    assert_eq!(
        counts["izwi_gateway_http_request_duration_seconds{route=\"/v1/chat/completions\",status=\"200\"}"],
        3
    );
    let tokens = |name: &str| {
        izwi_prometheus::parse_exposition(&text)
            .unwrap()
            .into_iter()
            .find(|sample| sample.name == name)
            .map(|sample| sample.value as u64)
    };
    assert_eq!(
        tokens("izwi_gateway_chat_prompt_tokens_total"),
        Some(3 * usage["prompt_tokens"].as_u64().unwrap())
    );
    assert_eq!(
        tokens("izwi_gateway_chat_completion_tokens_total"),
        Some(3 * usage["completion_tokens"].as_u64().unwrap())
    );
}

async fn json_body(response: Response) -> serde_json::Value {
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("response should be bounded");
    serde_json::from_slice(&body).expect("response should be JSON")
}

fn chat_request(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("chat request should build")
}

async fn wait_for_active(worker: &TestWorker, expected: usize) {
    for _ in 0..100 {
        if worker.active_invocations() == expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(worker.active_invocations(), expected);
}

#[tokio::test]
async fn public_chat_answers_404_for_a_model_it_does_not_serve() {
    let (state, worker) = registry_gateway_state(ModelVariant::Qwen354BGguf).await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    for model in [ModelVariant::Qwen359BGguf.dir_name(), "no-such-model"] {
        let response = send(
            app.clone(),
            chat_request(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32
            })),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{model}");
        assert_eq!(
            json_body(response).await["error"]["code"],
            "model_not_found",
            "{model}"
        );
    }
    assert_eq!(worker.served_invocations(), 0);
}

#[tokio::test]
async fn gateway_routes_each_request_to_its_models_workers() {
    let models = [ModelVariant::Qwen354BGguf, ModelVariant::Qwen359BGguf];
    let mut workers = Vec::new();
    let mut state: Option<GatewayState> = None;
    for (index, model) in models.into_iter().enumerate() {
        let worker = TestWorker::spawn(TestWorkerConfig {
            worker_id: format!("test-worker-{index}").try_into().unwrap(),
            deployment_id: format!("test-chat-{index}").try_into().unwrap(),
            public_model: ModelAlias::new(model.dir_name()).unwrap(),
            ..TestWorkerConfig::default()
        })
        .await
        .unwrap();
        let dispatcher = single_worker_dispatcher(
            model,
            &worker.endpoint(),
            worker.config().credentials.clone(),
        )
        .await;
        state = Some(match state {
            None => GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4),
            Some(state) => state.with_chat_model(dispatcher),
        });
        workers.push(worker);
    }
    let state = state.unwrap();
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let listed = json_body(send(app.clone(), get("/v1/models")).await).await;
    let ids: Vec<&str> = listed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, models.map(|model| model.dir_name()));
    for model in models {
        let path = format!("/v1/models/{}", model.dir_name());
        assert_eq!(send(app.clone(), get(&path)).await.status(), StatusCode::OK);
    }

    for (index, model) in models.into_iter().enumerate() {
        let response = send(
            app.clone(),
            chat_request(json!({
                "model": model.dir_name(),
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8
            })),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK, "{model:?}");
        let served: Vec<u64> = workers.iter().map(TestWorker::served_invocations).collect();
        let mut expected = vec![0; models.len()];
        expected[..=index].fill(1);
        assert_eq!(served, expected, "{model:?}");
    }
}

/// EDGE-F19: out-of-range sampling values are OpenAI-shaped 400s that
/// name the parameter, and never reach a worker.
#[tokio::test]
async fn out_of_range_sampling_values_are_rejected_at_the_edge() {
    let (state, worker) = registry_gateway_state(ModelVariant::Qwen354BGguf).await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    for (field, value) in [
        ("temperature", json!(2.5)),
        ("temperature", json!(-0.1)),
        ("top_p", json!(0.0)),
        ("top_p", json!(1.5)),
        ("top_k", json!(-2)),
        ("presence_penalty", json!(2.5)),
        ("frequency_penalty", json!(-3.0)),
        ("repetition_penalty", json!(0.5)),
        ("max_tokens", json!(0)),
        ("max_completion_tokens", json!(0)),
        ("n", json!(2)),
    ] {
        let mut body = json!({
            "model": ModelVariant::Qwen354BGguf.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
        });
        body[field] = value.clone();
        let response = send(app.clone(), chat_request(body)).await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "{field}={value}"
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error", "{field}");
        assert_eq!(body["error"]["param"], field, "{field}={value}");
        assert_eq!(body["error"]["code"], "invalid_value", "{field}");
    }
    assert_eq!(worker.served_invocations(), 0);
}

#[tokio::test]
async fn public_chat_timeout_requests_cancel_without_early_capacity_release() {
    let model = ModelVariant::Qwen354BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        output_cadence: Duration::from_millis(1),
        cancellation_delay: Duration::from_millis(150),
        script: WorkerScript::Hang,
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    let client = WorkerClient::new(
        &worker.endpoint(),
        credentials,
        WorkerClientConfig {
            first_output_timeout: Duration::from_millis(30),
            progress_timeout: Duration::from_millis(30),
            ..WorkerClientConfig::default()
        },
    )
    .expect("worker client should initialize");
    let dispatcher = single_worker_dispatcher_with_client(model, client).await;
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let response = send(app, valid_chat_request(32)).await;
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(worker.active_invocations(), 1);
    wait_for_active(&worker, 0).await;
}

#[tokio::test]
async fn public_chat_reports_accepted_then_interrupted_as_unknown_failure() {
    let model = ModelVariant::Qwen354BGguf;
    let (state, _worker) = registry_gateway_state_with_config(
        model,
        TestWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            script: WorkerScript::LoseAfterAccept,
            ..TestWorkerConfig::default()
        },
    )
    .await;
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let response = send(app, valid_chat_request(32)).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    // A server fault reaches the client as a generic message naming the
    // request; the worker's account of it is only logged.
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "bad_gateway");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.starts_with(crate::error::SERVER_FAULT_MESSAGE) && message.contains("request ID"),
        "{message}"
    );
}

#[tokio::test]
async fn tenant_work_survives_stream_disconnect_until_exact_worker_teardown() {
    let model = ModelVariant::Qwen354BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        script: WorkerScript::Hang,
        cancellation_delay: Duration::from_millis(100),
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    let dispatcher = single_worker_dispatcher(model, &worker.endpoint(), credentials).await;
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4)
        .with_tenant_concurrency_config(
            GatewayTenantConcurrencyConfig::new(1, 2).expect("test tenant limit"),
        );
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

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
        if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0 {
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
        if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(worker.active_invocations(), 0);
    assert_eq!(state.tenant_concurrency.active_owned_work(), 0);
}

#[tokio::test]
async fn gateway_crash_cannot_duplicate_execution_and_recovery_is_explicit() {
    let model = ModelVariant::Qwen354BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        script: WorkerScript::Hang,
        cancellation_delay: Duration::from_millis(100),
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config.clone())
        .await
        .expect("test worker should bind");
    let endpoint = worker.endpoint();
    let bootstrapping = |endpoint: String| {
        let credentials = credentials.clone();
        async move {
            let dispatcher = single_worker_dispatcher(model, &endpoint, credentials).await;
            let state =
                GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4)
                    .with_tenant_concurrency_config(
                        GatewayTenantConcurrencyConfig::new(1, 2).expect("test tenant limit"),
                    );
            state.lifecycle.mark_ready();
            let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
            (app, state)
        }
    };
    let stream_request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "crash mid-stream"}],
                    "stream": true,
                    "max_tokens": 8
                })
                .to_string(),
            ))
            .expect("stream request should build")
    };

    let (app, state) = bootstrapping(endpoint.clone()).await;
    let response = send(app.clone(), stream_request()).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(worker.active_invocations(), 1);

    // Simulate a gateway crash: drop the response, router, and all
    // gateway state without any graceful teardown or cancellation.
    drop((response, app, state));
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        worker.active_invocations(),
        1,
        "execution ownership survives gateway death on the worker"
    );

    // A restarted gateway has no memory of the orphaned work. Its fresh
    // tenant view admits, but the worker's held capacity refuses the
    // second concurrent invocation instead of duplicating execution.
    let (restarted_app, _) = bootstrapping(endpoint).await;
    let rejected = send(restarted_app.clone(), stream_request()).await;
    assert_ne!(
        rejected.status(),
        StatusCode::OK,
        "a crashed gateway's orphan must surface as rejection, never duplicate execution"
    );
    assert_eq!(worker.active_invocations(), 1);

    // The orphan cannot be cancelled without its attempt identity, which
    // died with the gateway — exactly the production crash case. Recovery
    // proceeds by replacing the worker: the orphan dies with its process
    // (fate sharing), and a fresh worker under the same approved identity
    // resumes service for the restarted gateway.
    drop(worker);
    let worker_next = TestWorker::spawn(worker_config)
        .await
        .expect("replacement test worker should bind");
    let (resumed_app, _) = bootstrapping(worker_next.endpoint()).await;
    assert_eq!(
        send(resumed_app, stream_request()).await.status(),
        StatusCode::OK,
        "service resumes once the orphaned worker is replaced"
    );
}

#[tokio::test]
async fn registry_stream_disconnect_releases_only_after_exact_worker_teardown() {
    let model = ModelVariant::Qwen354BGguf;
    let (state, worker) = registry_gateway_state_with_config(
        model,
        TestWorkerConfig {
            public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
            script: WorkerScript::Hang,
            cancellation_delay: Duration::from_millis(100),
            ..TestWorkerConfig::default()
        },
    )
    .await;
    let state = state.with_tenant_concurrency_config(
        GatewayTenantConcurrencyConfig::new(1, 2).expect("test tenant limit"),
    );
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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
        if worker.active_invocations() == 0 && state.tenant_concurrency.active_owned_work() == 0 {
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

async fn approve_worker(registry: &WorkerRegistry, worker: &TestWorker) -> WorkerStatus {
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
        .expect("test deployment should exist")
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
        .observe_status(status.clone())
        .expect("initial status should be valid");
    status
}

/// A realtime client that stops reading its socket must not stall the
/// relay's client-to-worker direction: its cancel still reaches the
/// worker, which ends the session at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn realtime_relay_forwards_a_cancel_while_the_client_is_not_reading() {
    use crate::app::realtime_relay::{GatewayRealtimeRelay, RealtimeRelayConfig};
    use futures::SinkExt;
    use izwi_serving_protocol::{
        GatewayAttestedCallerContext, ModelGeneration, PermittedAction, RealtimeClientFrame,
        RealtimeSessionAdmit, RealtimeStageInput, ServiceClass, TaskKind, PROTOCOL_V1,
        REALTIME_SUBPROTOCOL,
    };
    use izwi_serving_worker::test_support::TestTtsStage;
    use tokio_tungstenite::tungstenite::{client::IntoClientRequest, http::HeaderValue, Message};

    let model = ModelVariant::Qwen354BGguf;
    let alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let (state, _chat_worker) = registry_gateway_state(model).await;
    // A long synthesis that outruns any socket buffer, and a worker that
    // queues frames rather than dropping a slow gateway, so only a cancel
    // can end the session early.
    let tts_worker = TestWorker::spawn(TestWorkerConfig {
        worker_id: "test-tts-worker".try_into().expect("test id valid"),
        deployment_id: "test-tts-v1".try_into().expect("test id valid"),
        public_model: alias.clone(),
        realtime_tts: Some(TestTtsStage {
            chunk_bytes: 4096,
            chunk_count: 1_000_000,
            push_cadence: Duration::from_millis(1),
            output_sample_rate: 24_000,
            disconnect_after_frames: None,
        }),
        realtime_max_outbound_queue: 1_000_000,
        ..TestWorkerConfig::default()
    })
    .await
    .expect("tts worker binds");
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    approve_worker(&registry, &tts_worker).await;
    let relay = GatewayRealtimeRelay::new(
        registry,
        RealtimeRelayConfig {
            deployment_id: None,
            tts_deployment_id: Some("test-tts-v1".try_into().expect("test id valid")),
            public_model: alias,
            policy_revision: PolicyRevision::new("policy-v1").unwrap(),
            backend_policy: BackendPolicy::ANY,
            max_sessions: 4,
            session_budget: Duration::from_secs(60),
        },
        tts_worker.config().credentials.clone(),
        Default::default(),
    )
    .unwrap();
    let app = create_gateway_router(
        state.with_realtime_relay(relay),
        &GatewayServeConfig::default(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let mut request = format!("ws://127.0.0.1:{port}/v1/realtime/ws")
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        "sec-websocket-protocol",
        HeaderValue::from_static(REALTIME_SUBPROTOCOL),
    );
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {TEST_API_KEY}")).unwrap(),
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    let control =
        |frame: &RealtimeClientFrame| Message::Text(serde_json::to_string(frame).unwrap().into());
    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: "relay-cancel-session".try_into().unwrap(),
        request_id: "relay-cancel-request".try_into().unwrap(),
        attempt_id: "relay-cancel-attempt".try_into().unwrap(),
        expected_worker_incarnation: "ignored".try_into().unwrap(),
        deployment_id: "test-tts-v1".try_into().unwrap(),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: "client-tenant".try_into().unwrap(),
            caller_id: "client-caller".try_into().unwrap(),
            policy_revision: "client-policy".try_into().unwrap(),
            permitted_actions: [PermittedAction::Invoke].into_iter().collect(),
            allowed_data_regions: vec![],
        },
        task: TaskKind::TextToSpeech,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 60_000,
        input: RealtimeStageInput::text_stream(),
    };
    ws.send(control(&RealtimeClientFrame::Admit {
        admit: Box::new(admit),
    }))
    .await
    .unwrap();
    ws.send(control(&RealtimeClientFrame::Input {
        text: "a long utterance".into(),
    }))
    .await
    .unwrap();
    ws.send(control(&RealtimeClientFrame::Finish))
        .await
        .unwrap();

    // The client reads nothing while about 12 MB of audio is synthesized,
    // far more than the sockets between it and the relay can hold.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(tts_worker.active_invocations(), 1, "synthesis is running");
    ws.send(control(&RealtimeClientFrame::Cancel))
        .await
        .unwrap();
    let stopped = tokio::time::timeout(Duration::from_secs(2), async {
        while tts_worker.active_invocations() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        stopped.is_ok(),
        "the cancel must reach the worker while the client is not reading"
    );
    drop(ws);
    server.abort();
}

#[tokio::test]
async fn gateway_worker_loss_fails_over_to_replacement_incarnation() {
    // A model that answers without thinking by default, as the test worker does.
    let model = ModelVariant::Qwen352BGguf;
    let deployment_id = TestWorkerConfig::default().deployment_id;
    let incarnation = |suffix: &str| TestWorkerConfig {
        worker_id: "test-worker-fleet".try_into().expect("test id valid"),
        node_id: "test-node-fleet".try_into().expect("test id valid"),
        incarnation_id: format!("test-incarnation-{suffix}")
            .try_into()
            .expect("test incarnation should be valid"),
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        output_text: format!("from-{suffix}"),
        ..TestWorkerConfig::default()
    };
    let worker_a = TestWorker::spawn(incarnation("one"))
        .await
        .expect("first test worker should bind");
    // Short status TTL keeps the loss simulation fast; production uses
    // seconds, which only changes how long a dead worker shadows routing.
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        status_ttl: Duration::from_millis(50),
        ..WorkerRegistryConfig::default()
    })
    .expect("registry should initialize");
    let stale_a = approve_worker(&registry, &worker_a).await;
    let dispatcher = RemoteChatDispatcher::new(
        registry.clone(),
        RemoteChatDispatchConfig {
            public_model_variant: model,
            deployment_id,
            policy_revision: PolicyRevision::new("test-policy-v1").expect("static policy revision"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(1),
            max_output_tokens: 128,
            max_output_bytes: 4096,
            slow_consumer_timeout: Duration::from_millis(100),
            prefix_affinity: None,
        },
    )
    .expect("dispatcher should initialize");
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 5, 2);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    let chat = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": model.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": false,
                    "max_tokens": 8
                })
                .to_string(),
            ))
            .expect("chat request should build")
    };
    let first = send(app.clone(), chat()).await;
    assert_eq!(first.status(), StatusCode::OK);
    let body = axum::body::to_bytes(first.into_body(), 64 * 1024)
        .await
        .expect("response should be bounded");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("response should be JSON")
            ["choices"][0]["message"]["content"],
        "from-one"
    );

    // Kill the worker process. After the status TTL expires, the dead
    // incarnation must stop receiving traffic instead of hanging it.
    drop(worker_a);
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        send(app.clone(), chat()).await.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "expired observations must fail fast, never route to a dead worker"
    );

    // A replacement incarnation for the same logical worker takes over.
    let worker_b = TestWorker::spawn(incarnation("two"))
        .await
        .expect("replacement test worker should bind");
    approve_worker(&registry, &worker_b).await;
    let second = send(app.clone(), chat()).await;
    assert_eq!(second.status(), StatusCode::OK);
    let body = axum::body::to_bytes(second.into_body(), 64 * 1024)
        .await
        .expect("response should be bounded");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("response should be JSON")
            ["choices"][0]["message"]["content"],
        "from-two",
        "traffic must resume on the replacement incarnation"
    );

    // A late status from the dead incarnation must not resurrect it.
    assert_eq!(
        registry.observe_status(stale_a),
        Err(WorkerRegistryError::UnknownOrStaleIncarnation)
    );
}

#[tokio::test]
async fn public_chat_routes_concurrent_requests_to_distinct_http_replicas_without_retry() {
    let model = ModelVariant::Qwen354BGguf;
    let deployment_id = TestWorkerConfig::default().deployment_id;
    let replica = |suffix: &str, output_text: &str| TestWorkerConfig {
        worker_id: format!("test-replica-{suffix}")
            .try_into()
            .expect("test worker id should be valid"),
        node_id: format!("test-node-{suffix}")
            .try_into()
            .expect("test node id should be valid"),
        incarnation_id: format!("test-incarnation-{suffix}")
            .try_into()
            .expect("test incarnation should be valid"),
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        output_text: output_text.to_string(),
        // Once the first delta is observed, this leaves a deterministic
        // window in which the first worker still owns its sole permit.
        output_cadence: Duration::from_secs(1),
        ..TestWorkerConfig::default()
    };
    let worker_a = TestWorker::spawn(replica("a", "replica-a-only"))
        .await
        .expect("first test replica should bind");
    let worker_b = TestWorker::spawn(replica("b", "replica-b-only"))
        .await
        .expect("second test replica should bind");
    let registry =
        WorkerRegistry::new(WorkerRegistryConfig::default()).expect("registry should initialize");

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
            .expect("test deployment should exist")
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
            policy_revision: PolicyRevision::new("test-policy-v1").expect("static policy revision"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(1),
            max_output_tokens: 128,
            max_output_bytes: 4096,
            slow_consumer_timeout: Duration::from_millis(100),
            prefix_affinity: None,
        },
    )
    .expect("dispatcher should initialize");
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 5, 2);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());
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
    let second_text = String::from_utf8(second_bytes.to_vec()).expect("second SSE should be UTF-8");

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
    // A model that answers without thinking by default, as the test worker does.
    let model = ModelVariant::Qwen352BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    let dispatcher = single_worker_dispatcher(model, &worker.endpoint(), credentials).await;
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    assert_eq!(
        send(app.clone(), get("/livez")).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(app.clone(), get("/readyz")).await.status(),
        StatusCode::OK
    );
    for path in [
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
        "deterministic test response"
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
    assert!(body.contains("deterministic test response"));
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
    assert!(paths.contains_key("/v1/models"));
    assert!(!paths.contains_key("/v1/responses"));
}

#[tokio::test]
async fn gateway_readiness_rejects_a_stale_model_generation() {
    let model = ModelVariant::Qwen354BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    // The registry approved generation 2; the worker still serves 1.
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
    let mut approved = ApprovedDeployment::from_loaded(&status.deployments[0]);
    approved.model_generation = ModelGeneration::new(2).expect("non-zero stale generation");
    let deployment_id = approved.deployment_id.clone();
    let registry =
        WorkerRegistry::new(WorkerRegistryConfig::default()).expect("registry should initialize");
    registry
        .approve(ApprovedWorker {
            descriptor,
            client,
            approved_deployments: BTreeMap::from([(deployment_id.clone(), approved)]),
            validated_capacity: 1,
        })
        .expect("worker should be approved");
    assert!(registry.observe_status(status).is_err());
    let dispatcher = RemoteChatDispatcher::new(
        registry,
        RemoteChatDispatchConfig {
            public_model_variant: model,
            deployment_id,
            policy_revision: PolicyRevision::new("test-policy-v1").expect("static policy revision"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(1),
            max_output_tokens: 128,
            max_output_bytes: 4096,
            slow_consumer_timeout: Duration::from_millis(100),
            prefix_affinity: None,
        },
    )
    .expect("dispatcher should initialize");
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 4);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    assert_eq!(
        send(app, get("/readyz")).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn gateway_admission_rejects_instead_of_queueing() {
    let model = ModelVariant::Qwen354BGguf;
    let state = unreachable_gateway_state(test_perimeter());
    let held_permit = state
        .request_admission
        .clone()
        .try_acquire_owned()
        .expect("test should occupy the only gateway permit");
    let app = create_gateway_router(state, &GatewayServeConfig::default());
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
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": ModelVariant::Qwen354BGguf.dir_name(),
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
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    let response = send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": ModelVariant::Qwen354BGguf.dir_name(),
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
    let model = ModelVariant::Qwen354BGguf;
    let worker_config = TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        script: WorkerScript::Hang,
        cancellation_delay: Duration::from_millis(100),
        ..TestWorkerConfig::default()
    };
    let credentials = worker_config.credentials.clone();
    let worker = TestWorker::spawn(worker_config)
        .await
        .expect("test worker should bind");
    let dispatcher = single_worker_dispatcher(model, &worker.endpoint(), credentials).await;
    let state = GatewayState::new(dispatcher, EnterpriseHooks::noop(), test_perimeter(), 2, 1);
    state.lifecycle.mark_ready();
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

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
    assert_eq!(body["error"]["code"], "invalid_api_key");
    assert_eq!(state.metrics.snapshot().auth_rejections, 1);
}

#[tokio::test]
async fn gateway_metrics_are_absent_without_a_separate_credential() {
    let app = create_gateway_router(
        unreachable_gateway_state(test_perimeter()),
        &GatewayServeConfig::default(),
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

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
    assert_eq!(body.matches("# TYPE ").count(), 23);
    // Labels are the gateway's own fixed vocabulary, never caller text.
    let samples = izwi_prometheus::parse_exposition(&body).expect("valid exposition");
    for sample in &samples {
        for label in sample.labels.keys() {
            assert!(
                ["route", "status", "le", "model", "stream", "outcome"].contains(&label.as_str()),
                "{label}"
            );
        }
    }
    assert!(body.contains(
        "izwi_gateway_http_request_duration_seconds_count{route=\"/internal/metrics\",status=\"401\"} 2"
    ));
    for private in [
        TEST_API_KEY,
        TEST_METRICS_API_KEY,
        "test-gateway-principal",
        "test-tenant",
        ModelVariant::Qwen354BGguf.dir_name(),
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
async fn gateway_conformance_error_envelopes_and_advertised_surface() {
    let state = unreachable_gateway_state(test_perimeter());
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    // The OpenAPI document must expose exactly the advertised gateway
    // surface: text-only chat plus public probes/docs. Any unmigrated
    // route in the document is a migration-ledger violation.
    let openapi = send_raw(app.clone(), get("/openapi.json")).await;
    assert_eq!(openapi.status(), StatusCode::OK);
    let document: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(openapi.into_body(), 256 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    let mut paths: Vec<&str> = document["paths"]
        .as_object()
        .expect("openapi document must have paths")
        .keys()
        .map(String::as_str)
        .collect();
    paths.sort_unstable();
    for advertised in [
        "/livez",
        "/openapi.json",
        "/readyz",
        "/v1/chat/completions",
        "/v1/models",
    ] {
        assert!(
            paths.contains(&advertised),
            "gateway OpenAPI must advertise {advertised}, got {paths:?}"
        );
    }
    for gated in [
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/responses",
        "/v1/admin/models",
        "/v1/voice/sessions",
        "/v1/media",
        "/v1/voices",
        "/v1/studio/projects",
        "/v1/jobs",
    ] {
        assert!(
            !paths.contains(&gated),
            "gateway OpenAPI must not advertise unmigrated route {gated}"
        );
    }

    // 401, 404, and 413 must share one bounded OpenAI-style envelope
    // and carry a server-authored request id.
    let unauthorized = send_raw(app.clone(), post("/v1/chat/completions")).await;
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    let unauthorized_body = error_envelope(unauthorized).await;
    assert_eq!(unauthorized_body["error"]["type"], "authentication_error");
    assert!(unauthorized_body["error"]["message"].is_string());

    let not_found = send_raw(
        app.clone(),
        get_with_bearer("/v1/audio/speech", TEST_API_KEY),
    )
    .await;
    assert_eq!(not_found.status(), StatusCode::NOT_FOUND);
    let not_found_body = error_envelope(not_found).await;
    assert_eq!(not_found_body["error"]["type"], "not_found_error");
    assert!(not_found_body["error"]["message"].is_string());

    let oversized = send_raw(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {TEST_API_KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(vec![b'x'; 2 * 1024 * 1024]))
            .unwrap(),
    )
    .await;
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        oversized
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .is_some(),
        "413 rejection must still carry a server-authored request id"
    );
}

#[tokio::test]
async fn enabled_gateway_chat_route_eliminates_raw_local_path_assumptions() {
    let state = unreachable_gateway_state(test_perimeter());
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    // Attempting to inject file path references or media content in chat request
    // must fail closed at the gateway perimeter before reaching any execution.
    let file_path_injection = json!({
        "model": "LFM2.5-1.2B-Instruct-GGUF",
        "messages": [{
            "role": "user",
            "content": "test message with no local paths"
        }],
        "enable_thinking": true,
        "max_tokens": 16
    });
    let response = send_raw(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {TEST_API_KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(file_path_injection.to_string()))
            .unwrap(),
    )
    .await;
    // This gateway serves another model: a 404 before any execution.
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = error_envelope(response).await;
    assert_eq!(body["error"]["code"], "model_not_found");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    // Assert no server directories or local path leaks
    assert!(!msg.contains("/var/lib/izwi"));
    assert!(!msg.contains("/app/"));
    assert!(!msg.contains("/etc/"));
}

async fn error_envelope(response: Response) -> serde_json::Value {
    response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("error response must carry a server-authored request id");
    let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("error response must be bounded");
    serde_json::from_slice(&body).expect("error response must be JSON")
}

#[tokio::test]
async fn admin_drain_endpoint_requires_its_own_key_and_404s_without_it() {
    let app = create_gateway_router(
        unreachable_gateway_state(test_perimeter()),
        &GatewayServeConfig::default(),
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
async fn admin_approvals_status_reports_the_adopted_view_to_the_admin_key_only() {
    const TEST_ADMIN_API_KEY: &str = "test-admin-api-key-999888";
    let mut table = crate::gateway_deployments::GatewayDeploymentTable::default();
    let approvals: Vec<crate::gateway_deployments::GatewayWorkerApproval> = [
        "v1|http://127.0.0.1:9470|node-a|worker-old|chat|chat-model|chat-prod|1",
        "v1|http://127.0.0.1:9471|node-a|worker-new|chat|chat-model|chat-prod|2",
    ]
    .iter()
    .map(|line| line.parse().unwrap())
    .collect();
    table.apply_view(&approvals).unwrap();
    let reporter = crate::gateway_shared_approvals::GatewayApprovalsReporter::new(
        std::sync::Arc::new(std::sync::Mutex::new(table)),
    );
    reporter.record(Some("sha256:window"), &approvals);

    let without_admin =
        unreachable_gateway_state(test_perimeter()).with_approvals_reporter(reporter.clone());
    let app = create_gateway_router(without_admin, &GatewayServeConfig::default());
    assert_eq!(
        send_raw(app, get("/internal/admin/approvals"))
            .await
            .status(),
        StatusCode::NOT_FOUND,
        "the route needs an admin key configured"
    );

    let perimeter = test_perimeter()
        .with_admin_api_key_for_test(TEST_ADMIN_API_KEY)
        .unwrap();
    let state = unreachable_gateway_state(perimeter).with_approvals_reporter(reporter);
    let app = create_gateway_router(state, &GatewayServeConfig::default());
    assert_eq!(
        send_raw(
            app.clone(),
            get_with_bearer("/internal/admin/approvals", TEST_API_KEY)
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED,
        "the inference key must not read admin state"
    );
    let response = send_raw(
        app,
        get_with_bearer("/internal/admin/approvals", TEST_ADMIN_API_KEY),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let status: izwi_serving_protocol::GatewayApprovalsStatus = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(status.file_digest.as_deref(), Some("sha256:window"));
    assert_eq!(status.adoptions, 1);
    assert_eq!(status.approvals.len(), 2);
    assert_eq!(status.pools.len(), 1);
    let pool = &status.pools[0];
    assert_eq!(pool.deployment_id.as_str(), "chat-prod");
    assert_eq!(pool.generations.len(), 2, "a rollout window holds both");
    assert_eq!(
        pool.eligible_generation.map(|generation| generation.get()),
        Some(1),
        "before the successor is observed ready, the predecessor routes"
    );
}

#[tokio::test]
async fn admin_drain_endpoint_rejects_inference_key_and_accepts_admin_key() {
    const TEST_ADMIN_API_KEY: &str = "test-admin-api-key-999888";
    let perimeter = test_perimeter()
        .with_admin_api_key_for_test(TEST_ADMIN_API_KEY)
        .unwrap();
    let state = unreachable_gateway_state(perimeter);
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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

const TEST_SCOPED_INFERENCE_KEY: &str = "scoped-inference-key-123456";
const TEST_SCOPED_OPS_KEY: &str = "scoped-ops-metrics-key-654321";
const TEST_SCOPED_OTHER_TENANT_KEY: &str = "scoped-other-tenant-key-246810";

async fn scoped_key_directory(
    entries: &str,
) -> crate::gateway_principal_keys::GatewayPrincipalDirectory {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = dir.path().join("principals.json");
    std::fs::write(&manifest, entries).expect("manifest write");
    let store = crate::db::sqlite::StoreDatabase::new(dir.path().join("store.sqlite3"));
    crate::gateway_principal_keys::GatewayPrincipalDirectory::bootstrap(
        &store,
        &manifest,
        &test_perimeter(),
    )
    .await
    .expect("scoped principal directory should bootstrap")
}

#[tokio::test]
async fn scoped_principal_key_authenticates_the_inference_route() {
    let _guard = crate::test_support::env_lock();
    std::env::set_var("IZWI_TEST_GW_SCOPED_A", TEST_SCOPED_INFERENCE_KEY);
    let directory = scoped_key_directory(
        r#"{"version":1,"principals":[{"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_GW_SCOPED_A"}]}"#,
    )
    .await;
    std::env::remove_var("IZWI_TEST_GW_SCOPED_A");

    let app = create_gateway_router(
        unreachable_gateway_state(test_perimeter()).with_principal_keys(directory),
        &GatewayServeConfig::default(),
    );
    let response = send_raw(app, chat_request_with_bearer(TEST_SCOPED_INFERENCE_KEY, 4)).await;
    assert_ne!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "a valid scoped principal key must pass gateway authentication"
    );
    assert_ne!(
        response.status(),
        StatusCode::FORBIDDEN,
        "an inference-role principal must reach the route"
    );
}

#[tokio::test]
async fn scoped_principal_without_inference_role_is_forbidden_on_chat() {
    let _guard = crate::test_support::env_lock();
    std::env::set_var("IZWI_TEST_GW_SCOPED_B", TEST_SCOPED_OPS_KEY);
    let directory = scoped_key_directory(
        r#"{"version":1,"principals":[{"principal_id":"ops","roles":["metrics","admin"],"key_ref":"env:IZWI_TEST_GW_SCOPED_B"}]}"#,
    )
    .await;
    std::env::remove_var("IZWI_TEST_GW_SCOPED_B");

    let app = create_gateway_router(
        unreachable_gateway_state(test_perimeter()).with_principal_keys(directory),
        &GatewayServeConfig::default(),
    );
    let response = send_raw(app, chat_request_with_bearer(TEST_SCOPED_OPS_KEY, 4)).await;
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "a metrics/admin-only principal must not run inference"
    );
}

#[tokio::test]
async fn scoped_role_keys_open_internal_endpoints_without_dedicated_credentials() {
    let _guard = crate::test_support::env_lock();
    std::env::set_var("IZWI_TEST_GW_SCOPED_A", TEST_SCOPED_INFERENCE_KEY);
    std::env::set_var("IZWI_TEST_GW_SCOPED_B", TEST_SCOPED_OPS_KEY);
    let directory = scoped_key_directory(
        r#"{"version":1,"principals":[
                {"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_GW_SCOPED_A"},
                {"principal_id":"ops","roles":["metrics","admin"],"key_ref":"env:IZWI_TEST_GW_SCOPED_B"}]}"#,
    )
    .await;
    std::env::remove_var("IZWI_TEST_GW_SCOPED_A");
    std::env::remove_var("IZWI_TEST_GW_SCOPED_B");

    let state = unreachable_gateway_state(test_perimeter()).with_principal_keys(directory);
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

    let metrics = send_raw(
        app.clone(),
        get_with_bearer("/internal/metrics", TEST_SCOPED_OPS_KEY),
    )
    .await;
    assert_eq!(
        metrics.status(),
        StatusCode::OK,
        "a metrics-role scoped principal must open the metrics endpoint"
    );
    let drain = send_raw(
        app.clone(),
        post_with_bearer("/internal/admin/drain", TEST_SCOPED_OPS_KEY),
    )
    .await;
    assert_eq!(
        drain.status(),
        StatusCode::ACCEPTED,
        "an admin-role scoped principal must open the drain endpoint"
    );

    let inference_on_metrics = send_raw(
        app.clone(),
        get_with_bearer("/internal/metrics", TEST_SCOPED_INFERENCE_KEY),
    )
    .await;
    assert_eq!(
        inference_on_metrics.status(),
        StatusCode::UNAUTHORIZED,
        "an inference-only principal must not open the metrics endpoint"
    );
    let garbage = send_raw(
        app,
        get_with_bearer("/internal/metrics", "wrong-principal-key-000000"),
    )
    .await;
    assert_eq!(garbage.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn internal_endpoints_stay_not_found_without_any_role_capability() {
    let app = create_gateway_router(
        unreachable_gateway_state(test_perimeter()),
        &GatewayServeConfig::default(),
    );
    assert_eq!(
        send_raw(app.clone(), get("/internal/metrics"))
            .await
            .status(),
        StatusCode::NOT_FOUND,
        "metrics stays 404 without a dedicated key or a scoped metrics principal"
    );
    assert_eq!(
        send_raw(app, post("/internal/admin/drain")).await.status(),
        StatusCode::NOT_FOUND,
        "drain stays 404 without a dedicated key or a scoped admin principal"
    );
}

#[tokio::test]
async fn scoped_principal_tenant_scope_partitions_the_rate_quota() {
    let _guard = crate::test_support::env_lock();
    std::env::set_var("IZWI_TEST_GW_SCOPED_A", TEST_SCOPED_INFERENCE_KEY);
    std::env::set_var("IZWI_TEST_GW_SCOPED_B", TEST_SCOPED_OTHER_TENANT_KEY);
    let directory = scoped_key_directory(
        r#"{"version":1,"principals":[
                {"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_GW_SCOPED_A"},
                {"principal_id":"svc-beta","roles":["inference"],"tenant_id":"tenant-beta","key_ref":"env:IZWI_TEST_GW_SCOPED_B"}]}"#,
    )
    .await;
    std::env::remove_var("IZWI_TEST_GW_SCOPED_A");
    std::env::remove_var("IZWI_TEST_GW_SCOPED_B");

    let state = unreachable_gateway_state(test_perimeter())
        .with_rate_quota_config(GatewayRateQuotaConfig::new(1, 1, 4).unwrap())
        .with_principal_keys(directory);
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

    let tenant_a_first = send_raw(
        app.clone(),
        chat_request_with_bearer(TEST_SCOPED_INFERENCE_KEY, 4),
    )
    .await;
    assert_ne!(
        tenant_a_first.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "tenant-alpha starts with a full quota bucket"
    );
    let tenant_a_second = send_raw(
        app.clone(),
        chat_request_with_bearer(TEST_SCOPED_INFERENCE_KEY, 4),
    )
    .await;
    assert_eq!(
        tenant_a_second.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "tenant-alpha must exhaust its own bucket"
    );
    let tenant_b_first = send_raw(
        app,
        chat_request_with_bearer(TEST_SCOPED_OTHER_TENANT_KEY, 4),
    )
    .await;
    assert_ne!(
        tenant_b_first.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "tenant-beta must hold an independent quota bucket"
    );
}

#[tokio::test]
async fn gateway_dispatch_failure_and_latency_are_counted_without_dynamic_dimensions() {
    let state = unreachable_gateway_state(test_perimeter());
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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
        &GatewayServeConfig {
            cors_enabled: false,
            ..GatewayServeConfig::default()
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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
    assert_eq!(body["error"]["code"], "payload_too_large");
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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
        &GatewayServeConfig::default(),
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
        &GatewayServeConfig::default(),
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());
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

fn header<'a>(response: &'a Response, name: &str) -> Option<&'a str> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

#[tokio::test]
async fn chat_responses_carry_rate_limit_headers_and_429s_a_retry_after() {
    let (state, _worker) = registry_gateway_state(ModelVariant::Qwen354BGguf).await;
    let state = state.with_rate_quota_config(GatewayRateQuotaConfig::new(2, 1, 8).unwrap());
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let served = send(app.clone(), valid_chat_request(4)).await;
    assert_eq!(served.status(), StatusCode::OK);
    assert_eq!(header(&served, "x-ratelimit-limit-requests"), Some("2"));
    assert_eq!(header(&served, "x-ratelimit-remaining-requests"), Some("0"));
    assert_eq!(header(&served, "x-ratelimit-reset-requests"), Some("30s"));
    assert_eq!(header(&served, "x-ratelimit-limit-tokens"), None);

    let limited = send(app, valid_chat_request(4)).await;
    assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);
    // One request every 30 s refills the spent one.
    let retry_after: u64 = header(&limited, "retry-after").unwrap().parse().unwrap();
    assert!((29..=30).contains(&retry_after), "{retry_after}");
}

#[tokio::test]
async fn token_quota_admits_by_estimate_and_settles_to_reported_usage() {
    let (state, _worker) = registry_gateway_state(ModelVariant::Qwen354BGguf).await;
    // 60 tokens a minute: one a second refills during the test at most.
    let state = state.with_rate_quota_config(
        GatewayRateQuotaConfig::new(600, 64, 8)
            .unwrap()
            .with_tokens_per_minute(Some(60))
            .unwrap(),
    );
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let served = send(app.clone(), valid_chat_request(16)).await;
    assert_eq!(served.status(), StatusCode::OK);
    let body = json_body(served).await;
    let used = body["usage"]["total_tokens"].as_u64().unwrap();
    assert!(used > 0 && used < 16, "{used}");

    // The next response reports the balance after settling to actual
    // usage, not to the 16 + prompt estimate.
    let next = send(app.clone(), valid_chat_request(16)).await;
    assert_eq!(next.status(), StatusCode::OK);
    assert_eq!(header(&next, "x-ratelimit-limit-tokens"), Some("60"));
    let remaining: u64 = header(&next, "x-ratelimit-remaining-tokens")
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (60 - 2 * used..=60 - 2 * used + 2).contains(&remaining),
        "{remaining} after two requests of {used}"
    );

    // A request whose estimate exceeds what is left is refused.
    let limited = send(app, valid_chat_request(60)).await;
    assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(header(&limited, "retry-after").is_some());
    assert_eq!(
        json_body(limited).await["error"]["message"],
        "Gateway tenant token rate limit exceeded"
    );
}

#[tokio::test]
async fn completed_chat_commits_its_usage_to_the_quota_hook() {
    #[derive(Default)]
    struct MeteringQuota(Mutex<Vec<izwi_hooks::QuotaUsage>>);

    #[async_trait::async_trait]
    impl QuotaLimiter for MeteringQuota {
        async fn evaluate(&self, _request: &QuotaRequest) -> HookResult<QuotaDecision> {
            Ok(QuotaDecision::allow())
        }

        async fn commit(&self, usage: &izwi_hooks::QuotaUsage) -> HookResult<()> {
            self.0.lock().unwrap().push(usage.clone());
            Ok(())
        }
    }

    let (mut state, _worker) = registry_gateway_state(ModelVariant::Qwen354BGguf).await;
    let quota = Arc::new(MeteringQuota::default());
    state.enterprise_hooks.quotas = quota.clone();
    let app = create_gateway_router(state, &GatewayServeConfig::default());

    let body = json_body(send(app.clone(), valid_chat_request(16)).await).await;
    let stream = send(
        app,
        chat_request(json!({
            "model": ModelVariant::Qwen354BGguf.dir_name(),
            "messages": [{"role": "user", "content": "quota test"}],
            "max_tokens": 16,
            "stream": true
        })),
    )
    .await;
    assert_eq!(stream.status(), StatusCode::OK);
    axum::body::to_bytes(stream.into_body(), 1024 * 1024)
        .await
        .unwrap();

    let commits = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let commits = quota.0.lock().unwrap().clone();
            if commits.len() == 2 {
                break commits;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("both completions commit their usage");
    for usage in commits {
        assert_eq!(usage.estimated_units, Some(16));
        assert_eq!(
            usage.input_units,
            body["usage"]["prompt_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            usage.output_units,
            body["usage"]["completion_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            usage.principal.unwrap().tenant_id.as_deref(),
            Some("test-tenant")
        );
        assert_eq!(
            usage.metadata.get("model").map(String::as_str),
            Some(ModelVariant::Qwen354BGguf.dir_name())
        );
    }
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
        &GatewayServeConfig::default(),
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
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

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

/// A gateway whose registry has no workers: a request that passes the
/// edge gates fails at worker selection.
fn unreachable_gateway_state_with_hooks(
    perimeter: GatewayPerimeterConfig,
    hooks: EnterpriseHooks,
) -> GatewayState {
    unreachable_gateway_state_sized(perimeter, hooks, 1)
}

fn unreachable_gateway_state_sized(
    perimeter: GatewayPerimeterConfig,
    hooks: EnterpriseHooks,
    max_in_flight: usize,
) -> GatewayState {
    let registry =
        WorkerRegistry::new(WorkerRegistryConfig::default()).expect("registry should initialize");
    let dispatcher = RemoteChatDispatcher::new(
        registry,
        RemoteChatDispatchConfig {
            public_model_variant: ModelVariant::Qwen354BGguf,
            deployment_id: DeploymentId::new("chat-v1").expect("static deployment"),
            policy_revision: PolicyRevision::new("test-policy-v1").unwrap(),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(1),
            max_output_tokens: 16,
            max_output_bytes: 1024,
            slow_consumer_timeout: Duration::from_millis(100),
            prefix_affinity: None,
        },
    )
    .expect("dispatcher should initialize");
    let state = GatewayState::new(dispatcher, hooks, perimeter, 2, max_in_flight);
    state.lifecycle.mark_ready();
    state
}

/// A chat request whose headers arrive but whose body never does.
fn stalled_body_chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {TEST_API_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from_stream(futures::stream::pending::<
            Result<bytes::Bytes, std::io::Error>,
        >()))
        .unwrap()
}

async fn wait_for_permits(state: &GatewayState, available: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while state.request_admission.available_permits() != available {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("global admission permits settle");
}

#[tokio::test]
async fn a_tenant_at_its_limit_is_refused_before_taking_a_global_permit() {
    let state = unreachable_gateway_state_sized(test_perimeter(), EnterpriseHooks::noop(), 2)
        .with_tenant_concurrency_config(GatewayTenantConcurrencyConfig::new(1, 2).unwrap());
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

    let first = tokio::spawn(send_raw(app.clone(), stalled_body_chat_request()));
    wait_for_permits(&state, 1).await;
    // The same tenant's second slow upload is refused at once and never
    // holds the gateway's last global permit.
    let second = tokio::time::timeout(
        Duration::from_secs(1),
        send_raw(app.clone(), stalled_body_chat_request()),
    )
    .await
    .expect("a tenant over its limit is refused without reading its body");
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(state.request_admission.available_permits(), 1);
    first.abort();
}

#[tokio::test]
async fn a_body_that_does_not_arrive_in_time_is_refused_and_frees_admission() {
    let state = unreachable_gateway_state_sized(test_perimeter(), EnterpriseHooks::noop(), 2)
        .with_body_read_timeout(Duration::from_millis(100));
    let app = create_gateway_router(state.clone(), &GatewayServeConfig::default());

    let response = tokio::time::timeout(
        Duration::from_secs(2),
        send_raw(app, stalled_body_chat_request()),
    )
    .await
    .expect("the body-read deadline ends the request");
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    wait_for_permits(&state, 2).await;
    assert_eq!(state.tenant_concurrency.active_owned_work(), 0);
    assert_eq!(state.metrics.snapshot().body_timeout_rejections, 1);
}

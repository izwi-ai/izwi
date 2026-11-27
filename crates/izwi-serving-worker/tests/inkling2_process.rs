//! Real CPU/process-boundary proof for the Inkling 2 chat path.
//!
//! A tiny synthetic Inkling 2 checkpoint (izwi-core's reference fixture in
//! the published layout: a looped sparse-expert decoder with packed NVFP4
//! and MXFP8 experts, byte-level tokenizer) exercises the real
//! RuntimeService loader, engine-owned state, scheduler, and decode path
//! through an actual worker process over HTTP, including the public gateway
//! surface. No artifact is downloaded: the checkpoint carries the synthetic
//! marker, which only a test build of izwi-core honours.

mod common;

use axum::{body::Body, http::Request};
use common::{id, single_worker_dispatcher, write_tiny_inkling2_fixture};
use izwi_core::ModelVariant;
use izwi_hooks::EnterpriseHooks;
use izwi_server::worker_registry::BackendPolicy;
use izwi_server::{
    create_gateway_router, GatewayPerimeterConfig, GatewayState, RemoteChatDispatchConfig,
};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use izwi_test_process::ProcessGuard;
use std::{collections::BTreeSet, net::TcpListener, process::Stdio, time::Duration};
use tower::ServiceExt;

const VARIANT: ModelVariant = ModelVariant::Inkling2Mini;

#[tokio::test]
async fn separate_cpu_worker_executes_tiny_inkling2_over_real_http() {
    let models = tempfile::tempdir().unwrap();
    let model_dir = write_tiny_inkling2_fixture(models.path());
    assert!(model_dir.join("config.json").exists());
    assert!(model_dir.join("model.safetensors").exists());

    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"));
    command
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models.path())
        .env("IZWI_WORKER_MODEL", VARIANT.dir_name())
        .env("IZWI_WORKER_DEPLOYMENT_ID", "inkling2-cpu-v1")
        .env("IZWI_WORKER_CREDENTIAL_ID", "inkling2-cpu-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "inkling2-cpu-secret")
        .env(
            "IZWI_WORKER_ARTIFACT_REVISION",
            VARIANT
                .artifact_revision()
                .expect("Inkling 2 Mini artifact revision is catalog-pinned"),
        )
        .env("IZWI_WORKER_CPU_THREADS", "1")
        .env("IZWI_WORKER_MAX_ACTIVE", "1")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    let _child = ProcessGuard::spawn(&mut command).unwrap();

    let credentials = ServiceCredentials {
        credential_id: id("inkling2-cpu-credential"),
        bearer_token: ServiceBearerToken::new("inkling2-cpu-secret").unwrap(),
    };
    let client = WorkerClient::new(
        &format!("http://{address}/"),
        credentials,
        WorkerClientConfig {
            connect_timeout: Duration::from_millis(100),
            request_timeout: Duration::from_millis(500),
            progress_timeout: Duration::from_secs(5),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let descriptor = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("worker loads and warms the tiny Inkling 2 model before its startup deadline");
    assert_eq!(descriptor.assignment.backend(), BackendKind::Cpu);

    let events = client
        .invoke_collect(InvocationRequest {
            schema_version: PROTOCOL_V1,
            request_id: id("inkling2-request-1"),
            attempt_id: id("inkling2-attempt-1"),
            expected_worker_incarnation: descriptor.incarnation_id.clone(),
            deployment_id: id("inkling2-cpu-v1"),
            expected_model_generation: ModelGeneration::new(1).unwrap(),
            caller: GatewayAttestedCallerContext {
                tenant_id: id("local-test"),
                caller_id: id("inkling2-process-test"),
                policy_revision: id("test-policy-v1"),
                permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                allowed_data_regions: vec!["local".into()],
            },
            task: TaskKind::Chat,
            service_class: ServiceClass::Interactive,
            remaining_time_ms: 10_000,
            max_queue_wait_ms: 2_000,
            output_limits: OutputLimits {
                max_tokens: 4,
                max_bytes: 1024,
            },
            requested_output_format: OutputFormat::Text,
            session_id: None,
            request_digest: id("sha256:real-cpu-inkling2-fixture"),
            input: InvocationInput::Chat {
                input: ChatInput {
                    messages: vec![ChatMessage {
                        role: ChatRole::User,
                        content: "hello".into(),
                    }],
                },
                parameters: ChatParameters {
                    temperature: Some(0.0),
                    top_p: Some(1.0),
                    seed: Some(0),
                    stop: Vec::new(),
                    ..ChatParameters::default()
                },
            },
        })
        .await
        .unwrap();
    assert!(matches!(
        events.first().map(|event| &event.event),
        Some(InvocationEventKind::Accepted { .. })
    ));
    // The fixture's weights are random, so a reply may hold no visible text
    // (text shows only inside a message the model opens itself). The proof is
    // the decode: the worker read the whole prompt and produced tokens.
    let Some(InvocationEventKind::Completed {
        usage: Some(usage), ..
    }) = events.last().map(|event| &event.event)
    else {
        panic!("the invocation completes with its usage: {events:?}");
    };
    // The prompt is the effort line and the user message.
    assert!(usage.input_tokens > "hello".len() as u64, "{usage:?}");
    assert!((1..=4).contains(&usage.output_tokens), "{usage:?}");

    // The family's engine-owned state reports the same managed-KV routing
    // signals as the other chat families once it has completed one invocation.
    let status = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match client.status().await {
                Ok(status) => break status,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("status after a completed invocation");
    let deployment = &status.deployments[0];
    let usage_pct = deployment
        .kv_cache_usage_pct
        .expect("engine-backed Inkling 2 worker reports managed-KV utilization");
    assert!((0.0..=100.0).contains(&usage_pct));
    assert_eq!(deployment.observation_cost_units, Some(1));

    let dispatcher = single_worker_dispatcher(
        client,
        RemoteChatDispatchConfig {
            public_model_variant: VARIANT,
            deployment_id: id("inkling2-cpu-v1"),
            policy_revision: id("inkling2-cpu-policy-v1"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::from_secs(2),
            max_output_tokens: 4,
            max_output_bytes: 1024,
            slow_consumer_timeout: Duration::from_secs(5),
            prefix_affinity: None,
        },
    )
    .await;
    let gateway = GatewayState::new(
        dispatcher,
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("inkling2-test-api-key", 1024 * 1024).unwrap(),
        10,
        2,
    );
    gateway.mark_ready();
    let app = create_gateway_router(
        gateway,
        &izwi_core::ServeRuntimeConfig {
            ui_enabled: false,
            ..izwi_core::ServeRuntimeConfig::default()
        },
    );
    let public_request = |stream: bool| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer inkling2-test-api-key")
            .body(Body::from(
                serde_json::json!({
                    "model": VARIANT.dir_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 4,
                    "stream": stream,
                    "stream_options": {"include_usage": true}
                })
                .to_string(),
            ))
            .unwrap()
    };

    let response = app.clone().oneshot(public_request(false)).await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(
        body["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|tokens| (1..=4).contains(&tokens)),
        "{body}"
    );

    let response = app.clone().oneshot(public_request(true)).await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        body.contains("chat.completion.chunk"),
        "streaming chunks expected"
    );
    assert!(body.contains("[DONE]"), "terminal stream marker expected");

    // JSON mode reaches the worker (protocol minor 4) instead of a 400; the
    // local/remote parity suite compares the constrained output itself.
    let constrained = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer inkling2-test-api-key")
        .body(Body::from(
            serde_json::json!({
                "model": VARIANT.dir_name(),
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0,
                "max_tokens": 4,
                "response_format": {"type": "json_object"}
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.oneshot(constrained).await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["object"], "chat.completion");
}

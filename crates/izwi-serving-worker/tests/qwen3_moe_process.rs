//! Real CPU/process-boundary proof for the sparse-expert (MoE) chat path.
//!
//! A tiny genuine Qwen3-MoE GGUF (fused llama.cpp expert tensors, `qwen3moe`
//! metadata, 2 routed experts) exercises the real RuntimeService loader,
//! sparse dispatch, scheduler, managed memory, and decode path through an
//! actual worker process over HTTP — including the public gateway surface.
//! DS10 groundwork: no MoE artifact is downloaded (ADR 0008).

mod common;

use axum::{body::Body, http::Request};
use common::{id, write_tiny_qwen3_moe_fixture};
use izwi_core::ModelVariant;
use izwi_hooks::EnterpriseHooks;
use izwi_server::{
    create_gateway_router, GatewayPerimeterConfig, GatewayState, RemoteChatExecution,
    RemoteChatExecutionConfig,
};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use std::{collections::BTreeSet, net::TcpListener, process::Stdio, time::Duration};
use tower::ServiceExt;

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

#[tokio::test]
async fn separate_cpu_worker_executes_tiny_qwen3_moe_over_real_http() {
    let models = tempfile::tempdir().unwrap();
    let model_dir = write_tiny_qwen3_moe_fixture(models.path());
    assert!(model_dir.join("Qwen3-30B-A3B-Q4_K_M.gguf").exists());

    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"))
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models.path())
        .env("IZWI_WORKER_MODEL", "Qwen3-30B-A3B-GGUF")
        .env("IZWI_WORKER_DEPLOYMENT_ID", "qwen3-moe-cpu-v1")
        .env("IZWI_WORKER_CREDENTIAL_ID", "moe-cpu-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "moe-cpu-secret")
        .env("IZWI_WORKER_ARTIFACT_REVISION", "tiny-qwen3-moe-fixture-v1")
        .env("IZWI_WORKER_CPU_THREADS", "1")
        .env("IZWI_WORKER_MAX_ACTIVE", "1")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let _child = ChildGuard(child);

    let credentials = ServiceCredentials {
        credential_id: id("moe-cpu-credential"),
        bearer_token: ServiceBearerToken::new("moe-cpu-secret").unwrap(),
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
    let descriptor = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("worker loads and warms the tiny MoE model before its startup deadline");
    assert_eq!(descriptor.assignment.backend(), BackendKind::Cpu);

    let events = client
        .invoke_collect(InvocationRequest {
            schema_version: PROTOCOL_V1,
            request_id: id("moe-request-1"),
            attempt_id: id("moe-attempt-1"),
            expected_worker_incarnation: descriptor.incarnation_id.clone(),
            deployment_id: id("qwen3-moe-cpu-v1"),
            expected_model_generation: ModelGeneration::new(1).unwrap(),
            caller: GatewayAttestedCallerContext {
                tenant_id: id("local-test"),
                caller_id: id("moe-process-test"),
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
            request_digest: id("sha256:real-cpu-moe-fixture"),
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
                },
            },
        })
        .await
        .unwrap();
    assert!(matches!(
        events.first().map(|event| &event.event),
        Some(InvocationEventKind::Accepted { .. })
    ));
    assert!(matches!(
        events.last().map(|event| &event.event),
        Some(InvocationEventKind::Completed { .. })
    ));
    assert!(events.iter().any(|event| {
        matches!(&event.event, InvocationEventKind::TextDelta { text, .. } if !text.is_empty())
    }));

    // The sparse-expert engine reports the same managed-KV routing signals as
    // the dense families once it has completed one invocation.
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
        .expect("engine-backed MoE worker reports managed-KV utilization");
    assert!((0.0..=100.0).contains(&usage_pct));
    assert_eq!(deployment.observation_cost_units, Some(1));

    let remote = RemoteChatExecution::new(
        client,
        RemoteChatExecutionConfig {
            public_model_variant: ModelVariant::Qwen3Moe30bA3bGguf,
            expected_worker_incarnation: descriptor.incarnation_id,
            deployment_id: id("qwen3-moe-cpu-v1"),
            expected_model_generation: ModelGeneration::new(1).unwrap(),
            policy_revision: id("moe-cpu-policy-v1"),
            max_queue_wait: Duration::from_secs(2),
            max_output_tokens: 4,
            max_output_bytes: 1024,
            slow_consumer_timeout: Duration::from_secs(5),
        },
    )
    .unwrap();
    let gateway = GatewayState::new(
        remote,
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("moe-cpu-test-api-key", 1024 * 1024).unwrap(),
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
            .header("authorization", "Bearer moe-cpu-test-api-key")
            .body(Body::from(
                serde_json::json!({
                    "model": ModelVariant::Qwen3Moe30bA3bGguf.dir_name(),
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
    assert!(body["choices"][0]["message"]["content"]
        .as_str()
        .is_some_and(|text| !text.is_empty()));

    let response = app.oneshot(public_request(true)).await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("chat.completion.chunk"), "streaming chunks expected");
    assert!(body.contains("[DONE]"), "terminal stream marker expected");
}

//! Process-level T01 (serving plan §16.1): the gateway binary starts, serves
//! its probes, and routes an inference request through an approved worker —
//! without constructing an engine. The negative control is structural and
//! observable: the process starts with no model artifacts or models directory
//! anywhere, which is only possible because gateway mode owns no
//! RuntimeService. An inference request succeeding through the mock worker
//! proves the request was executed remotely, not locally.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use izwi_core::ModelVariant;
use izwi_serving_client::mock::{MockWorker, MockWorkerConfig};
use izwi_serving_protocol::ModelAlias;

const GATEWAY_API_KEY: &str = "t01-gateway-api-key";

#[tokio::test]
async fn gateway_process_serves_probes_and_remote_inference_without_an_engine() {
    let model = ModelVariant::Qwen34BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let worker_config = MockWorkerConfig {
        public_model: model_alias,
        output_text: "gateway-process-t01".to_string(),
        // The CLI dispatcher requires capacity for its 4096-token output
        // budget; the mock's 1024-token default would fail eligibility.
        max_output_tokens: 8_192,
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(worker_config)
        .await
        .expect("mock worker should bind a real loopback TCP listener");

    // Reserve a port the same way an operator would: bind, learn it, release.
    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let endpoint = format!("http://127.0.0.1:{port}");

    // Standalone compatibility approval: URL|TASK|PUBLIC_MODEL|DEPLOYMENT|GEN.
    // The mock's static deployment identity is mock-chat-v1 @ generation 1.
    let approval = format!(
        "{}|chat|{}|mock-chat-v1|1",
        worker.endpoint(),
        model.dir_name()
    );

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &approval,
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    // Drain diagnostics so bounded startup/restart logs cannot fill the pipe.
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for _ in BufReader::new(stderr).lines() {}
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);

    // /livez answers while the gateway is running.
    let livez = loop {
        match client.get(format!("{base}/livez")).send().await {
            Ok(response) => break response,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => panic!("gateway never answered /livez: {error}"),
        }
    };
    assert_eq!(livez.status(), 200, "liveness must answer 200");

    // /readyz covers registry-backed readiness: the approved worker answered
    // its initial authenticated descriptor/status exchange.
    let readyz = loop {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => break response,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway never became ready: {:?}",
                other.map(|r| r.status())
            ),
        }
    };
    assert_eq!(readyz.status(), 200);

    // Inference: authenticated request must be executed by the mock worker and
    // relayed with the exact public JSON shape (no local engine is possible:
    // the process started without any model artifacts on disk).
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {GATEWAY_API_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("chat request reaches the gateway");
    assert_eq!(
        response.status(),
        200,
        "remote chat request must succeed: {}",
        response.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = response.json().await.expect("JSON completion body");
    assert_eq!(
        body["object"], "chat.completion",
        "public completion shape must be preserved through the worker boundary"
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("message content present");
    assert!(
        content.contains("gateway-process-t01"),
        "output must come from the approved worker, got {content:?}"
    );

    // Unauthenticated callers are rejected at the perimeter.
    let rejected = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("unauthenticated request reaches the gateway");
    assert_eq!(
        rejected.status(),
        401,
        "missing credentials must be rejected"
    );

    let _ = gateway.kill();
    let _ = gateway.wait();
}

const SCOPED_INFERENCE_KEY: &str = "ds05-scoped-inference-key";
const SCOPED_OPS_KEY: &str = "ds05-scoped-ops-metrics-key";

/// DS0.5 process leg: scoped per-principal keys are provisioned from a bounded
/// manifest into the durable store at boot, authenticate through the real
/// gateway perimeter with role and tenant semantics, and leave the shared
/// root key fully backward compatible.
#[tokio::test]
async fn gateway_process_authenticates_scoped_principal_keys_from_the_durable_store() {
    let model = ModelVariant::Qwen34BGguf;
    let model_alias = ModelAlias::new(model.dir_name()).expect("static model alias");
    let worker_config = MockWorkerConfig {
        public_model: model_alias,
        output_text: "gateway-process-ds05".to_string(),
        max_output_tokens: 8_192,
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(worker_config)
        .await
        .expect("mock worker should bind a real loopback TCP listener");

    let storage = tempfile::tempdir().expect("temp storage dir");
    let manifest_path = storage.path().join("principals.json");
    std::fs::write(
        &manifest_path,
        format!(
            r#"{{"version":1,"principals":[
                {{"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_PROCESS_SCOPED_A"}},
                {{"principal_id":"ops","roles":["metrics","admin"],"key_ref":"env:IZWI_TEST_PROCESS_SCOPED_B"}}]}}"#
        ),
    )
    .expect("manifest write");

    let port = {
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("probe listener binds");
        probe.local_addr().expect("probe addr").port()
    };
    let approval = format!(
        "{}|chat|{}|mock-chat-v1|1",
        worker.endpoint(),
        model.dir_name()
    );

    let mut gateway = Command::new(env!("CARGO_BIN_EXE_izwi-server"))
        .args([
            "--role",
            "gateway",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--public-model",
            model.dir_name(),
            "--gateway-worker-approval",
            &approval,
        ])
        .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
        .env(
            "IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST",
            manifest_path.display().to_string(),
        )
        .env(
            "IZWI_DB_PATH",
            storage.path().join("gateway.sqlite3").display().to_string(),
        )
        .env("IZWI_TEST_PROCESS_SCOPED_A", SCOPED_INFERENCE_KEY)
        .env("IZWI_TEST_PROCESS_SCOPED_B", SCOPED_OPS_KEY)
        .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
        .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
        .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
        .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gateway binary spawns");
    if let Some(stderr) = gateway.stderr.take() {
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for _ in BufReader::new(stderr).lines() {}
        });
    }

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let readyz = loop {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => break response,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway never became ready: {:?}",
                other.map(|r| r.status())
            ),
        }
    };
    assert_eq!(readyz.status(), 200);

    // A scoped inference principal executes remotely like the root key does.
    let scoped = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {SCOPED_INFERENCE_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("scoped chat request reaches the gateway");
    assert_eq!(
        scoped.status(),
        200,
        "scoped inference key must be admitted: {}",
        scoped.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = scoped.json().await.expect("JSON completion body");
    assert_eq!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("message content present"),
        "gateway-process-ds05",
        "scoped principal request must execute through the worker"
    );

    // The shared root key remains valid as the bootstrap root principal.
    let root = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {GATEWAY_API_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("root-key chat request reaches the gateway");
    assert_eq!(root.status(), 200, "root key must stay backward compatible");

    // A scoped principal without the inference role is forbidden, and unknown
    // credentials are rejected at the perimeter exactly as before.
    let forbidden = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {SCOPED_OPS_KEY}"))
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("ops-key chat request reaches the gateway");
    assert_eq!(
        forbidden.status(),
        403,
        "a metrics/admin-only principal must not run inference"
    );
    let rejected = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", "Bearer ds05-unknown-key-000000")
        .json(&serde_json::json!({
            "model": model.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("unknown-key chat request reaches the gateway");
    assert_eq!(rejected.status(), 401);

    // A scoped metrics principal opens the internal metrics side-channel even
    // though no dedicated metrics credential was configured.
    let metrics = client
        .get(format!("{base}/internal/metrics"))
        .header("Authorization", format!("Bearer {SCOPED_OPS_KEY}"))
        .send()
        .await
        .expect("ops-key metrics request reaches the gateway");
    assert_eq!(
        metrics.status(),
        200,
        "a metrics-role scoped principal must open the metrics endpoint"
    );

    let _ = gateway.kill();
    let _ = gateway.wait();
}

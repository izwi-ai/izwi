//! `izwi-server --role gateway` still serves the gateway: the role delegates
//! to the izwi-gateway library, whose own process suites run against the
//! `izwi-gateway` binary. This one routes an inference request through an
//! approved worker and checks the process still logs as izwi-server.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use izwi_core::ModelVariant;
use izwi_serving_protocol::ModelAlias;
use izwi_serving_worker::test_support::{TestWorker, TestWorkerConfig};
use izwi_test_process::ProcessGuard;

const GATEWAY_API_KEY: &str = "role-gateway-api-key";

#[tokio::test]
async fn server_gateway_role_routes_inference_through_an_approved_worker() {
    let model = ModelVariant::Qwen354BGguf;
    let worker = TestWorker::spawn(TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        output_text: "server-gateway-role".to_string(),
        max_output_tokens: 8_192,
        ..TestWorkerConfig::default()
    })
    .await
    .expect("test worker binds");
    let approval = format!(
        "{}|chat|{}|test-chat-v1|1",
        worker.endpoint(),
        model.dir_name()
    );
    let mut child = ProcessGuard::spawn(
        Command::new(env!("CARGO_BIN_EXE_izwi-server"))
            .args([
                "--role",
                "gateway",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--public-model",
                model.dir_name(),
                "--gateway-worker-approval",
                &approval,
            ])
            .env("IZWI_LOG_FORMAT", "json")
            .env("RUST_LOG", "izwi_gateway=info")
            .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
            .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "test-credential-1")
            .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "test-secret-token")
            .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "200")
            .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .expect("izwi-server spawns");
    let (lines_tx, lines) = mpsc::channel();
    let stdout = child.stdout.take().expect("piped stdout");
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = lines_tx.send(line);
        }
    });
    let next_line = |deadline: Instant| -> serde_json::Value {
        let line = lines
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("the gateway keeps logging");
        serde_json::from_str(&line).unwrap_or_else(|_| panic!("a JSON log line: {line}"))
    };

    // The gateway binds port 0 and logs the address it got.
    let deadline = Instant::now() + Duration::from_secs(30);
    let port = loop {
        let line = next_line(deadline);
        if let Some(address) = line["message"]
            .as_str()
            .and_then(|message| message.strip_prefix("Gateway listening on http://"))
        {
            break address
                .rsplit(':')
                .next()
                .and_then(|port| port.parse::<u16>().ok())
                .expect("listening address has a port");
        }
    };

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    loop {
        match client.get(format!("{base}/readyz")).send().await {
            Ok(response) if response.status().is_success() => break,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway never became ready: {:?}",
                other.map(|r| r.status())
            ),
        }
    }
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
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("JSON completion body");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("message content present");
    assert!(content.contains("server-gateway-role"), "{content:?}");

    // The delegated role logs, and exports spans, as izwi-server.
    let deadline = Instant::now() + Duration::from_secs(10);
    let completed = loop {
        let line = next_line(deadline);
        if line["message"] == "HTTP request completed" {
            break line;
        }
    };
    assert_eq!(completed["service"], "izwi-server", "{completed}");

    let _ = child.kill();
    let _ = child.wait();
}

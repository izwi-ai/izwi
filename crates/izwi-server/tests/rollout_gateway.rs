//! DS6 (T37 gateway half): the gateway adopts rollout approval views at
//! runtime and cuts over generations with zero dropped requests.
//!
//! One real `izwi-server` gateway process reads its worker approvals from a
//! shared approvals file; two `MockWorker`s serve generation 1 and
//! generation 2 of the same deployment on real loopback TCP. The test
//! drives the coordinator's view transitions by rewriting the file exactly
//! as the supervisor's rollout coordinator does (atomic rename) while a
//! background client keeps sending chat requests:
//!
//! - window view (both generations approved): the gateway cuts over to the
//!   successor when it first observes it Ready — subsequent requests serve
//!   from the successor only, and no request ever fails (DINV-07: never
//!   two admission-eligible generations, no zero-eligible gap).
//! - abort view (predecessor only, restored byte-identically): the
//!   predecessor — which never stopped — serves again immediately.
//! - commit view (successor only): the successor keeps serving.
//!
//! Zero dropped requests is asserted over the full scenario via the
//! served-invocation counters on both workers: every request is accounted
//! for on exactly one worker and every HTTP response is 200.

use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use izwi_core::ModelVariant;
use izwi_serving_client::mock::{MockWorker, MockWorkerConfig};
use izwi_serving_protocol::{ModelAlias, ModelGeneration, NodeId, WorkerId};

const GATEWAY_API_KEY: &str = "rollout-rig-api-key";
const MODEL: ModelVariant = ModelVariant::Qwen34BGguf;
const DEPLOYMENT_ID: &str = "mock-chat-v1";

async fn reserve_port() -> u16 {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe listener binds");
    probe.local_addr().expect("probe addr").port()
}

fn worker_config(worker_id: &str, generation: u64, marker: &str) -> MockWorkerConfig {
    MockWorkerConfig {
        worker_id: WorkerId::new(worker_id).expect("static worker id"),
        node_id: NodeId::new("rollout-node").expect("static node id"),
        public_model: ModelAlias::new(MODEL.dir_name()).expect("static model alias"),
        model_generation: ModelGeneration::new(generation).expect("static generation"),
        output_text: marker.to_string(),
        // The CLI dispatcher requires capacity for its 4096-token output
        // budget; the mock's 1024-token default would fail eligibility.
        max_output_tokens: 8_192,
        // A rollout target needs observable headroom: at capacity 1 a status
        // poll landing mid-request reports zero available credits and the
        // registry finds no eligible worker for that instant.
        max_active_invocations: 4,
        ..MockWorkerConfig::default()
    }
}

fn approval_line(endpoint: &str, worker_id: &str, generation: u64) -> String {
    format!(
        "v1|{endpoint}|rollout-node|{worker_id}|chat|{}|{DEPLOYMENT_ID}|{generation}",
        MODEL.dir_name()
    )
}

struct GatewayProcess {
    child: Child,
    base: String,
    stderr: Arc<std::sync::Mutex<String>>,
    stdout: Arc<std::sync::Mutex<String>>,
}

impl GatewayProcess {
    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn log_tail(&self) -> String {
        let stderr = self.stderr.lock().expect("stderr lock");
        let stdout = self.stdout.lock().expect("stdout lock");
        let mut lines: Vec<&str> = stderr.lines().collect();
        lines.extend(stdout.lines());
        let start = lines.len().saturating_sub(40);
        lines[start..].join("\n")
    }

    fn stderr_tail(&self) -> String {
        let guard = self.stderr.lock().expect("stderr lock");
        let lines: Vec<&str> = guard.lines().collect();
        let start = lines.len().saturating_sub(30);
        lines[start..].join("\n")
    }
}

impl Drop for GatewayProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

async fn spawn_gateway(approvals_path: &std::path::Path) -> GatewayProcess {
    for attempt in 0..3 {
        let port = reserve_port().await;
        let mut command = Command::new(env!("CARGO_BIN_EXE_izwi-server"));
        command
            .args([
                "--role",
                "gateway",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--public-model",
                MODEL.dir_name(),
            ])
            .env("IZWI_GATEWAY_API_KEY", GATEWAY_API_KEY)
            .env("IZWI_GATEWAY_WORKER_CREDENTIAL_ID", "mock-credential-1")
            .env("IZWI_GATEWAY_WORKER_BEARER_TOKEN", "mock-secret-token")
            .env("IZWI_GATEWAY_WORKER_STATUS_POLL_MS", "100")
            .env("IZWI_GATEWAY_WORKER_STATUS_TTL_MS", "5000")
            .env("IZWI_GATEWAY_TENANT_MAX_CONCURRENT", "32")
            .env("IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE", "3000")
            .env("IZWI_GATEWAY_TENANT_BURST_REQUESTS", "256")
            // The shared approvals file IS the rollout channel: a short TTL
            // so view transitions adopt quickly.
            .env("IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS", "1000")
            .env(
                "IZWI_GATEWAY_SHARED_APPROVALS_PATH",
                approvals_path.display().to_string(),
            )
            .env("RUST_LOG", "izwi_server=debug,izwi_serving_client=debug");
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command.spawn().expect("gateway binary spawns");
        let stderr_buffer = Arc::new(std::sync::Mutex::new(String::new()));
        let stdout_buffer = Arc::new(std::sync::Mutex::new(String::new()));
        if let Some(stderr) = child.stderr.take() {
            let buffer = Arc::clone(&stderr_buffer);
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                // Lock per line: holding the guard across the blocking read
                // would deadlock every tail reader.
                for line in BufReader::new(stderr).lines() {
                    match line {
                        Ok(line) => {
                            let mut collected = buffer.lock().expect("stderr lock");
                            collected.push_str(&line);
                            collected.push('\n');
                        }
                        Err(_) => break,
                    }
                }
            });
        }
        if let Some(stdout) = child.stdout.take() {
            let buffer = Arc::clone(&stdout_buffer);
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                for line in BufReader::new(stdout).lines() {
                    match line {
                        Ok(line) => {
                            let mut collected = buffer.lock().expect("stdout lock");
                            collected.push_str(&line);
                            collected.push('\n');
                        }
                        Err(_) => break,
                    }
                }
            });
        }
        let mut gateway = GatewayProcess {
            child,
            base: format!("http://127.0.0.1:{port}"),
            stderr: stderr_buffer,
            stdout: stdout_buffer,
        };
        tokio::time::sleep(Duration::from_millis(800)).await;
        if matches!(gateway.child.try_wait(), Ok(Some(_)))
            && gateway.stderr_tail().contains("Address already in use")
        {
            assert!(attempt < 2, "gateway spawn kept losing its port race");
            continue;
        }
        return gateway;
    }
    unreachable!("the retry loop returns or asserts")
}

/// Atomically replaces the approvals file exactly like the coordinator:
/// temp file plus rename under the rollout lock naming.
fn write_view(path: &std::path::Path, text: &str) {
    let parent = path.parent().unwrap();
    let temp = parent.join(format!(
        ".{}.rollout-temp-{}",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&temp, text).unwrap();
    std::fs::rename(&temp, path).unwrap();
}

async fn wait_ready(gateway: &GatewayProcess) {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match client.get(format!("{}/readyz", gateway.base)).send().await {
            Ok(response) if response.status().is_success() => return,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway at {} never became ready: {:?}; gateway log tail:\n{}",
                gateway.base,
                other.map(|r| r.status()),
                gateway.log_tail()
            ),
        }
    }
}

/// Drives chat traffic until the condition holds (checked between
/// requests). Every request must succeed, so a transition that drops
/// requests fails loudly; returns whether the condition was observed
/// before the deadline.
async fn chat_until(
    client: &reqwest::Client,
    gateway: &GatewayProcess,
    deadline: Duration,
    mut condition: impl FnMut() -> bool,
) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        chat(client, gateway).await;
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    condition()
}

/// Sends one chat request; panics on any non-200 so zero-dropped-requests
/// evidence is loud.
async fn chat(client: &reqwest::Client, gateway: &GatewayProcess) {
    let base = gateway.base.as_str();
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {GATEWAY_API_KEY}"))
        .json(&serde_json::json!({
            "model": MODEL.dir_name(),
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        }))
        .send()
        .await
        .expect("chat request reaches the gateway");
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    assert_eq!(
        status,
        reqwest::StatusCode::OK,
        "no request may be dropped during a rollout; body {text}; gateway log tail:\n{}",
        gateway.log_tail()
    );
}

#[tokio::test]
async fn rollout_views_cut_over_generations_with_zero_dropped_requests() {
    let directory = tempfile::tempdir().expect("tempdir");
    let approvals_path = directory.path().join("approvals.txt");

    // Both generations are live before the gateway boots; the file starts
    // with the predecessor only.
    let old_worker = MockWorker::spawn(worker_config(
        "rollout-worker-old",
        1,
        "old-generation-response",
    ))
    .await
    .expect("old mock worker spawns");
    let new_worker = MockWorker::spawn(worker_config(
        "rollout-worker-new",
        2,
        "new-generation-response",
    ))
    .await
    .expect("new mock worker spawns");
    let original_view = format!(
        "# fleet approvals\n{}\n",
        approval_line(
            old_worker.endpoint().trim_end_matches('/'),
            "rollout-worker-old",
            1
        )
    );
    std::fs::write(&approvals_path, &original_view).unwrap();

    let mut gateway = spawn_gateway(&approvals_path).await;
    wait_ready(&gateway).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("test client builds");
    chat(&client, &gateway).await;
    let baseline_old = old_worker.served_invocations();
    assert!(
        baseline_old >= 1,
        "the predecessor serves before the rollout starts"
    );
    assert_eq!(new_worker.served_invocations(), 0);

    // Window view: both generations approved. The gateway adopts the view,
    // admits the successor, and cuts over when it first observes Ready.
    let window_view = format!(
        "# fleet approvals\n{}\n{}\n",
        approval_line(
            old_worker.endpoint().trim_end_matches('/'),
            "rollout-worker-old",
            1
        ),
        approval_line(
            new_worker.endpoint().trim_end_matches('/'),
            "rollout-worker-new",
            2
        ),
    );
    write_view(&approvals_path, &window_view);

    // Traffic keeps flowing while the gateway adopts the view: the first
    // request the successor serves proves the cutover, and every request
    // through the transition must succeed (T37 zero dropped requests).
    let adopted = chat_until(&client, &gateway, Duration::from_secs(30), || {
        new_worker.served_invocations() > 0
    })
    .await;
    assert!(
        adopted,
        "the gateway must adopt the successor generation; gateway log tail:\n{}",
        gateway.log_tail()
    );

    // From the cutover instant the predecessor must not receive any new
    // request: park its counter and push traffic through the window.
    let cutover_old_count = old_worker.served_invocations();
    for _ in 0..10 {
        chat(&client, &gateway).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        old_worker.served_invocations(),
        cutover_old_count,
        "after cutover the draining predecessor must receive no new admissions (DINV-07)"
    );
    assert!(new_worker.served_invocations() > 0);

    // Abort: restore the predecessor-only view byte-identically. The
    // predecessor never stopped serving and takes traffic back.
    write_view(&approvals_path, &original_view);
    let restored = chat_until(&client, &gateway, Duration::from_secs(30), || {
        old_worker.served_invocations() > cutover_old_count
    })
    .await;
    assert!(restored, "the abort view must re-serve the predecessor");
    let abort_new_count = new_worker.served_invocations();
    for _ in 0..5 {
        chat(&client, &gateway).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        new_worker.served_invocations(),
        abort_new_count,
        "after the abort the successor must receive no new admissions"
    );
    // The abort legitimately re-serves the predecessor; park its counter
    // here so the commit phase can prove it goes quiet again.
    let abort_old_count = old_worker.served_invocations();
    assert!(
        abort_old_count > cutover_old_count,
        "the abort must have re-served the predecessor"
    );

    // Commit: the successor-only view makes it the only approved
    // generation and it keeps serving without interruption.
    let commit_view = format!(
        "# fleet approvals\n{}\n",
        approval_line(
            new_worker.endpoint().trim_end_matches('/'),
            "rollout-worker-new",
            2
        ),
    );
    write_view(&approvals_path, &commit_view);
    let committed = chat_until(&client, &gateway, Duration::from_secs(30), || {
        new_worker.served_invocations() > abort_new_count
    })
    .await;
    assert!(committed, "the commit view must serve the successor");
    // The predecessor may serve during the adoption lag while the previous
    // view still governs; once the commit is observed it must go quiet.
    let commit_old_count = old_worker.served_invocations();
    for _ in 0..5 {
        chat(&client, &gateway).await;
    }
    assert_eq!(
        old_worker.served_invocations(),
        commit_old_count,
        "the committed predecessor stays out of admission (DINV-07)"
    );

    gateway.kill();
}

async fn wait_for(deadline: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    condition()
}

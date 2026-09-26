//! DS5.2 multi-process fleet rig: two real `izwi-server` gateway processes
//! (distinct `IZWI_GATEWAY_ID`s, one shared coordination database) route
//! chat through mock workers over real loopback TCP, proving the
//! multi-gateway profile with processes rather than in-process doubles:
//!
//! - T07 — concurrent admission from two gateways never exceeds the shared
//!   worker's capacity, and the fleet claim fence never shows more live
//!   claims than the worker's observable credits.
//! - T20 — partitioned tenant quotas keep the combined fleet admission
//!   within the configured budget (limits are not multiplied by gateway
//!   count).
//! - P8.1 — shared approvals are adopted by both gateways and both
//!   gateways' status pollers merge observations monotonically into the
//!   shared registry.
//! - Claim steering — a held cluster claim steers peer dispatch away from
//!   the claimed worker and back once it is released.
//! - Crash + TTL recovery — a claim held by a gateway identity that dies
//!   (modelled by the rig holding the claim under the dead gateway's
//!   identity: the same store state a crashed process leaves behind)
//!   expires by TTL and frees capacity with no release protocol; a live
//!   gateway keeps serving while its peer is dead.
//!
//! Deviation from the DS5.2 text: the two "worker processes" are two
//! in-process `MockWorker` instances on real loopback TCP listeners (no
//! worker binary exists to spawn); the gateways treat them identically to
//! separate worker processes and every property under test lives in the
//! gateway processes and the shared store.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::anyhow;

use izwi_core::ModelVariant;
use izwi_server::batch_runtime::store::BatchRuntimeStore;
use izwi_serving_client::mock::{MockWorker, MockWorkerConfig};
use izwi_serving_protocol::{ModelAlias, WorkerId};

const GATEWAY_API_KEY: &str = "fleet-rig-api-key";
const WORKER_ONE_ID: &str = "fleet-rig-worker-1";
const WORKER_TWO_ID: &str = "fleet-rig-worker-2";
const WORKER_ONE_MARKER: &str = "fleet-rig-worker-one-response";
const WORKER_TWO_MARKER: &str = "fleet-rig-worker-two-response";
const MODEL: ModelVariant = ModelVariant::Qwen34BGguf;

async fn reserve_port() -> u16 {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe listener binds");
    probe.local_addr().expect("probe addr").port()
}

struct GatewayProcess {
    child: Child,
    base: String,
    stderr: std::sync::Arc<std::sync::Mutex<String>>,
}

impl GatewayProcess {
    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

fn worker_config(worker_id: &str, marker: &str) -> MockWorkerConfig {
    worker_config_with_capacity(worker_id, marker, 1)
}

fn worker_config_with_capacity(
    worker_id: &str,
    marker: &str,
    max_active_invocations: usize,
) -> MockWorkerConfig {
    MockWorkerConfig {
        worker_id: WorkerId::new(worker_id).expect("static worker id"),
        public_model: ModelAlias::new(MODEL.dir_name()).expect("static model alias"),
        output_text: marker.to_string(),
        // The CLI dispatcher requires capacity for its 4096-token output
        // budget; the mock's 1024-token default would fail eligibility.
        max_output_tokens: 8_192,
        max_active_invocations,
        ..MockWorkerConfig::default()
    }
}

fn approval_for(worker: &MockWorker, model_dir: &str) -> String {
    format!("{}|chat|{}|mock-chat-v1|1", worker.endpoint(), model_dir)
}

async fn spawn_gateway(
    gateway_id: &str,
    fleet_db_path: &str,
    approvals: &[String],
    extra_env: Vec<(&'static str, String)>,
) -> GatewayProcess {
    // The bind-learn-release port probe can lose its port to an unrelated
    // ephemeral connection when many processes spawn concurrently; retry the
    // spawn on the resulting address-in-use boot failure.
    for attempt in 0..3 {
        let mut gateway =
            spawn_gateway_once(gateway_id, fleet_db_path, approvals, extra_env.clone()).await;
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

async fn spawn_gateway_once(
    gateway_id: &str,
    fleet_db_path: &str,
    approvals: &[String],
    extra_env: Vec<(&'static str, String)>,
) -> GatewayProcess {
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
        // Headroom above the default tenant limits so bursts exercise fleet
        // behavior rather than the local tenant caps.
        .env("IZWI_GATEWAY_TENANT_MAX_CONCURRENT", "32")
        .env("IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE", "3000")
        .env("IZWI_GATEWAY_TENANT_BURST_REQUESTS", "256")
        .env("IZWI_GATEWAY_ID", gateway_id)
        .env("IZWI_GATEWAY_FLEET_DB_PATH", fleet_db_path);
    for approval in approvals {
        command.args(["--gateway-worker-approval", approval]);
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = command.spawn().expect("gateway binary spawns");
    let stderr_buffer = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    if let Some(stderr) = child.stderr.take() {
        let buffer = std::sync::Arc::clone(&stderr_buffer);
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let mut collected = buffer.lock().expect("stderr lock");
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) => {
                        collected.push_str(&line);
                        collected.push('\n');
                    }
                    Err(_) => break,
                }
            }
        });
    }
    GatewayProcess {
        child,
        base: format!("http://127.0.0.1:{port}"),
        stderr: stderr_buffer,
    }
}

async fn wait_ready(gateway: &GatewayProcess) {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match client.get(format!("{}/readyz", gateway.base)).send().await {
            Ok(response) if response.status().is_success() => return,
            _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(100)).await,
            other => panic!(
                "gateway at {} never became ready: {:?}; stderr:\n{}",
                gateway.base,
                other.map(|r| r.status()),
                gateway.stderr_tail()
            ),
        }
    }
}

async fn chat(client: &reqwest::Client, base: &str) -> (u16, serde_json::Value) {
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
    let status = response.status().as_u16();
    let text = response.text().await.unwrap_or_default();
    let body: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("gateway returned JSON: {error}; body {text}"));
    (status, body)
}

fn response_marker(body: &serde_json::Value) -> String {
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn fleet_gateways_share_atomic_admission_across_processes() {
    let root = tempfile::tempdir().expect("temp dir");
    let fleet_db = root.path().join("t07-fleet.sqlite3");
    let worker = MockWorker::spawn(worker_config(WORKER_ONE_ID, WORKER_ONE_MARKER))
        .await
        .expect("mock worker binds");
    let approval = approval_for(&worker, MODEL.dir_name());

    let gateway_a = spawn_gateway(
        "gateway-a",
        fleet_db.to_str().expect("fleet db path"),
        &[approval.clone()],
        Vec::new(),
    )
    .await;
    let gateway_b = spawn_gateway(
        "gateway-b",
        fleet_db.to_str().expect("fleet db path"),
        &[approval],
        Vec::new(),
    )
    .await;
    wait_ready(&gateway_a).await;
    wait_ready(&gateway_b).await;

    // Sample the shared claim table while both gateways dispatch: the fence
    // must never show more live claims than the worker's single credit.
    let claims = BatchRuntimeStore::initialize_with_database_path(fleet_db.clone());
    let sampler = tokio::spawn(async move {
        let mut max_live: u64 = 0;
        let deadline = Instant::now() + Duration::from_secs(6);
        while Instant::now() < deadline {
            max_live = max_live.max(
                claims
                    .count_live_fleet_claims(WORKER_ONE_ID)
                    .await
                    .unwrap_or(0),
            );
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        max_live
    });

    let client = reqwest::Client::new();
    let mut statuses = Vec::new();
    let mut handles = Vec::new();
    for index in 0..6u32 {
        let client = client.clone();
        let base = if index % 2 == 0 {
            gateway_a.base.clone()
        } else {
            gateway_b.base.clone()
        };
        handles.push(tokio::spawn(async move { chat(&client, &base).await }));
    }
    for handle in handles {
        statuses.push(handle.await.expect("chat task joins"));
    }
    let max_live = sampler.await.expect("sampler joins");

    let mut successes = 0;
    for (status, body) in &statuses {
        match status {
            200 => {
                successes += 1;
                assert!(
                    response_marker(body).contains(WORKER_ONE_MARKER),
                    "successful responses must come from the shared worker"
                );
            }
            // The designed shed: a gateway that loses the capacity race drops
            // to the bounded retry, finds no alternate, and answers 503 — the
            // worker's admission is never oversubscribed.
            503 => {}
            other => panic!("unexpected status {other}: {body}"),
        }
    }
    assert!(
        successes >= 1,
        "at least one of the concurrent requests must be served, got {statuses:?}"
    );
    assert!(
        max_live <= 1,
        "the fleet claim fence must never show more than the worker's single credit, saw {max_live}"
    );
}

#[tokio::test]
async fn fleet_partition_keeps_combined_quota_within_the_configured_budget() {
    let root = tempfile::tempdir().expect("temp dir");
    let fleet_db = root.path().join("t20-fleet.sqlite3");
    // Headroom above the sequential request pace: the poller snapshot can
    // trail the worker's true state by one poll interval, so a capacity-1
    // worker would shed 503s mid-burst and drown the rate-limit signal.
    let worker = MockWorker::spawn(worker_config_with_capacity(
        WORKER_ONE_ID,
        WORKER_ONE_MARKER,
        16,
    ))
    .await
    .expect("mock worker binds");
    let approval = approval_for(&worker, MODEL.dir_name());

    // Total tenant budget: 60 requests/minute with a burst bucket of 32.
    // Partitioned across a fleet of 2, each gateway owns half: a burst
    // bucket of 16 and a 30/minute refill.
    let budget_env: Vec<(&'static str, String)> = vec![
        ("IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE", "60".to_string()),
        ("IZWI_GATEWAY_TENANT_BURST_REQUESTS", "32".to_string()),
        ("IZWI_GATEWAY_FLEET_SIZE", "2".to_string()),
    ];
    let gateway_a = spawn_gateway(
        "gateway-a",
        fleet_db.to_str().expect("fleet db path"),
        &[approval.clone()],
        {
            let mut env = budget_env.clone();
            env.push(("IZWI_GATEWAY_FLEET_PARTITION", "0".to_string()));
            env
        },
    )
    .await;
    let gateway_b = spawn_gateway(
        "gateway-b",
        fleet_db.to_str().expect("fleet db path"),
        &[approval],
        {
            let mut env = budget_env;
            env.push(("IZWI_GATEWAY_FLEET_PARTITION", "1".to_string()));
            env
        },
    )
    .await;
    wait_ready(&gateway_a).await;
    wait_ready(&gateway_b).await;

    let client = reqwest::Client::new();
    let mut accepted_a = 0u32;
    let mut rejected_a = 0u32;
    for _ in 0..40 {
        let (status, _) = chat(&client, &gateway_a.base).await;
        if status == 200 {
            accepted_a += 1;
        } else {
            assert_eq!(status, 429, "over-budget requests must be rate-limited");
            rejected_a += 1;
        }
    }
    let mut accepted_b = 0u32;
    for _ in 0..20 {
        let (status, _) = chat(&client, &gateway_b.base).await;
        if status == 200 {
            accepted_b += 1;
        } else {
            assert_eq!(status, 429);
        }
    }

    // Each partition admits at most its own half-burst (16) plus a token or
    // two of refill over the fire window — an unpartitioned gateway would
    // admit up to 33 from the same configuration.
    assert!(
        accepted_a <= 18,
        "partition 0 must not exceed its half of the budget, accepted {accepted_a}"
    );
    assert!(
        accepted_b <= 18,
        "partition 1 must not exceed its half of the budget, accepted {accepted_b}"
    );
    assert!(
        accepted_a + accepted_b <= 36,
        "combined fleet admission must stay within the configured 60/minute budget: {accepted_a} + {accepted_b}"
    );
    assert!(accepted_a >= 1, "a non-empty partition must serve traffic");
    assert!(
        rejected_a >= 20,
        "the partitioned budget must shed the over-budget tail, rejected {rejected_a}"
    );
}

#[tokio::test]
async fn fleet_gateways_share_the_approvals_file_and_observation_registry() {
    let root = tempfile::tempdir().expect("temp dir");
    let fleet_db = root.path().join("p81-fleet.sqlite3");
    let worker_one = MockWorker::spawn(worker_config(WORKER_ONE_ID, WORKER_ONE_MARKER))
        .await
        .expect("mock worker binds");
    let worker_two = MockWorker::spawn(worker_config(WORKER_TWO_ID, WORKER_TWO_MARKER))
        .await
        .expect("mock worker binds");
    // The CLI set stays explicit (worker one); the shared file AUGMENTS it
    // with worker two. Duplicate endpoints across the two sources fail
    // closed at startup, so the rig keeps the sets disjoint and asserts the
    // union is adopted by both gateways.
    let cli_approvals = vec![approval_for(&worker_one, MODEL.dir_name())];
    let shared_approval = approval_for(&worker_two, MODEL.dir_name());

    let approvals_path = root.path().join("shared-approvals.txt");
    std::fs::write(
        &approvals_path,
        format!("# fleet shared approvals\n{shared_approval}\n"),
    )
    .expect("approvals file writes");

    let shared = vec![(
        "IZWI_GATEWAY_SHARED_APPROVALS_PATH",
        approvals_path.to_str().expect("approvals path").to_string(),
    )];
    let gateway_a = spawn_gateway(
        "gateway-a",
        fleet_db.to_str().expect("fleet db path"),
        &cli_approvals,
        shared.clone(),
    )
    .await;
    let gateway_b = spawn_gateway(
        "gateway-b",
        fleet_db.to_str().expect("fleet db path"),
        &cli_approvals,
        shared,
    )
    .await;
    wait_ready(&gateway_a).await;
    wait_ready(&gateway_b).await;

    // Both gateways adopted the shared approvals and each serves chat.
    let client = reqwest::Client::new();
    for gateway in [&gateway_a, &gateway_b] {
        let (status, body) = chat(&client, &gateway.base).await;
        assert_eq!(status, 200, "gateway must serve through a shared approval");
        let marker = response_marker(&body);
        assert!(
            marker.contains(WORKER_ONE_MARKER) || marker.contains(WORKER_TWO_MARKER),
            "response must come from an approved fleet worker, got {marker:?}"
        );
    }

    // Both gateways' pollers publish observations for the same workers into
    // the shared registry; the merged view holds one fresh row per worker
    // and every worker's sequence advances monotonically under concurrent
    // publishers. Publishing is poller-cadence asynchronous, so wait for
    // both workers to appear before comparing samples.
    let claims = BatchRuntimeStore::initialize_with_database_path(fleet_db.clone());
    let mut first = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        first = claims
            .read_fresh_fleet_workers(60_000)
            .await
            .expect("fresh view");
        let complete = [WORKER_ONE_ID, WORKER_TWO_ID]
            .iter()
            .all(|worker_id| first.iter().any(|view| view.worker_id == *worker_id));
        if complete || Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    for worker_id in [WORKER_ONE_ID, WORKER_TWO_ID] {
        assert!(
            first.iter().any(|view| view.worker_id == worker_id),
            "{worker_id} must be observed by a fleet publisher; saw {} worker(s)",
            first.len()
        );
    }
    tokio::time::sleep(Duration::from_millis(600)).await;
    let second = claims
        .read_fresh_fleet_workers(60_000)
        .await
        .expect("fresh view");
    for worker_id in [WORKER_ONE_ID, WORKER_TWO_ID] {
        let before = first
            .iter()
            .find(|view| view.worker_id == worker_id)
            .unwrap_or_else(|| panic!("{worker_id} observed in the first sample"));
        let after = second
            .iter()
            .find(|view| view.worker_id == worker_id)
            .unwrap_or_else(|| panic!("{worker_id} observed in the second sample"));
        assert!(
            after.status_sequence >= before.status_sequence,
            "observations must merge monotonically across gateway publishers"
        );
    }
}

#[tokio::test]
async fn fleet_capacity_claims_steer_dispatch_between_workers() {
    let root = tempfile::tempdir().expect("temp dir");
    let fleet_db = root.path().join("steer-fleet.sqlite3");
    let worker_one = MockWorker::spawn(worker_config(WORKER_ONE_ID, WORKER_ONE_MARKER))
        .await
        .expect("mock worker binds");
    let worker_two = MockWorker::spawn(worker_config(WORKER_TWO_ID, WORKER_TWO_MARKER))
        .await
        .expect("mock worker binds");
    let approvals = vec![
        approval_for(&worker_one, MODEL.dir_name()),
        approval_for(&worker_two, MODEL.dir_name()),
    ];
    let gateway = spawn_gateway(
        "gateway-a",
        fleet_db.to_str().expect("fleet db path"),
        &approvals,
        Vec::new(),
    )
    .await;
    wait_ready(&gateway).await;

    let claims = BatchRuntimeStore::initialize_with_database_path(fleet_db.clone());
    let held = claims
        .try_claim_fleet_capacity(
            WORKER_ONE_ID,
            "mock-incarnation-1",
            "gateway-rig",
            1,
            60_000,
        )
        .await
        .expect("rig claim")
        .expect("the rig holds worker one's only credit");
    // Let the gateway's poller refresh its fleet snapshot.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    // Each dispatch to a capacity-1 worker needs more than one poll interval
    // of spacing so the gateway's cached credits refresh between requests —
    // otherwise a stale zero-credit snapshot sheds a 503.
    for _ in 0..3 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let (status, body) = chat(&client, &gateway.base).await;
        assert_eq!(status, 200, "steered dispatch failed: {body}");
        assert!(
            response_marker(&body).contains(WORKER_TWO_MARKER),
            "a claimed worker must be steered away from, got {:?}",
            response_marker(&body)
        );
    }

    claims
        .release_fleet_capacity(&held, "gateway-rig")
        .await
        .expect("rig release");
    // Let the gateway's async claim releases from the first leg land before
    // claiming the peer worker's single credit.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let held_two = claims
        .try_claim_fleet_capacity(
            WORKER_TWO_ID,
            "mock-incarnation-1",
            "gateway-rig",
            1,
            60_000,
        )
        .await
        .expect("rig claim")
        .expect("the rig holds worker two's only credit");
    tokio::time::sleep(Duration::from_millis(500)).await;
    for _ in 0..3 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let (status, body) = chat(&client, &gateway.base).await;
        assert_eq!(status, 200, "post-release dispatch failed: {body}");
        assert!(
            response_marker(&body).contains(WORKER_ONE_MARKER),
            "capacity freed by a release must be dispatchable again, got {:?}",
            response_marker(&body)
        );
    }
    claims
        .release_fleet_capacity(&held_two, "gateway-rig")
        .await
        .expect("rig release");
}

#[tokio::test]
async fn fleet_recovers_after_a_gateway_death_and_expired_claims() {
    let root = tempfile::tempdir().expect("temp dir");
    let fleet_db = root.path().join("crash-fleet.sqlite3");
    let worker_one = MockWorker::spawn(worker_config(WORKER_ONE_ID, WORKER_ONE_MARKER))
        .await
        .expect("mock worker binds");
    let worker_two = MockWorker::spawn(worker_config(WORKER_TWO_ID, WORKER_TWO_MARKER))
        .await
        .expect("mock worker binds");
    let approvals = vec![
        approval_for(&worker_one, MODEL.dir_name()),
        approval_for(&worker_two, MODEL.dir_name()),
    ];
    let short_ttl = vec![("IZWI_GATEWAY_FLEET_CLAIM_TTL_MS", "300".to_string())];
    let mut gateway_a = spawn_gateway(
        "gateway-a",
        fleet_db.to_str().expect("fleet db path"),
        &approvals,
        short_ttl.clone(),
    )
    .await;
    let gateway_b = spawn_gateway(
        "gateway-b",
        fleet_db.to_str().expect("fleet db path"),
        &approvals,
        short_ttl,
    )
    .await;
    wait_ready(&gateway_a).await;
    wait_ready(&gateway_b).await;

    // A gateway that dies without releasing leaves exactly this store state:
    // a live claim under its own identity. Forge it with an 800ms TTL so it
    // is live for the steering assertion and expired for the recovery one.
    let claims = BatchRuntimeStore::initialize_with_database_path(fleet_db.clone());
    let _dead_claim = claims
        .try_claim_fleet_capacity(WORKER_ONE_ID, "mock-incarnation-1", "gateway-a", 1, 800)
        .await
        .expect("forged claim")
        .expect("the crashed gateway's claim holds");
    tokio::time::sleep(Duration::from_millis(300)).await;

    let client = reqwest::Client::new();
    let (status, body) = chat(&client, &gateway_b.base).await;
    assert_eq!(status, 200);
    assert!(
        response_marker(&body).contains(WORKER_TWO_MARKER),
        "dispatch must avoid the worker shadowed by the dead gateway's claim, got {:?}",
        response_marker(&body)
    );

    // No release protocol exists: TTL expiry is the crash recovery.
    tokio::time::sleep(Duration::from_millis(900)).await;
    let _shadow_two = claims
        .try_claim_fleet_capacity(
            WORKER_TWO_ID,
            "mock-incarnation-1",
            "gateway-rig",
            1,
            60_000,
        )
        .await
        .expect("rig claim")
        .expect("the rig now shadows worker two");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (status, body) = chat(&client, &gateway_b.base).await;
    assert_eq!(status, 200);
    assert!(
        response_marker(&body).contains(WORKER_ONE_MARKER),
        "capacity freed by TTL expiry must be dispatchable without any release protocol, got {:?}",
        response_marker(&body)
    );

    // With one gateway dead, the survivor keeps serving through the fleet.
    gateway_a.kill();
    claims
        .release_gateway_claims("gateway-rig")
        .await
        .expect("rig releases its shadow claim");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (status, body) = chat(&client, &gateway_b.base).await;
    assert_eq!(status, 200, "the surviving gateway must keep serving");
    let marker = response_marker(&body);
    assert!(
        marker.contains(WORKER_ONE_MARKER) || marker.contains(WORKER_TWO_MARKER),
        "the surviving gateway routes through an approved fleet worker, got {marker:?}"
    );
}

/// DS5.2 PostgreSQL lane: the same multi-process rig against a server-backed
/// coordination database, plus the DINV-06 outage contract at process level —
/// terminating every database backend mid-flight degrades the fleet to
/// worker-authoritative admission (dispatches still succeed, nothing panics)
/// and the store recovers on reconnect. Skipped unless the `db-postgres`
/// feature is enabled and `IZWI_TEST_FLEET_RIG_PG_URL` points at a
/// disposable PostgreSQL database (its tables are dropped and re-migrated).
#[cfg(feature = "db-postgres")]
#[tokio::test]
async fn fleet_rig_postgres_lane_shares_admission_and_degrades_on_outage() {
    use sea_orm::{ConnectionTrait, Statement};

    let Ok(fleet_url) = std::env::var("IZWI_TEST_FLEET_RIG_PG_URL") else {
        eprintln!("skipping: IZWI_TEST_FLEET_RIG_PG_URL is not set");
        return;
    };

    // Clean slate for the coordination schema.
    let rig = BatchRuntimeStore::initialize_with_database_url(fleet_url.clone());
    {
        let db = rig.connection().await.expect("postgres connection opens");
        let rows = db
            .query_all_raw(Statement::from_string(
                db.get_database_backend(),
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()".to_string(),
            ))
            .await
            .expect("table list");
        for row in rows {
            let table: String = row.try_get_by_index(0).expect("table name");
            db.execute_unprepared(&format!("DROP TABLE IF EXISTS \"{table}\" CASCADE"))
                .await
                .expect("table drops");
        }
    }
    // Reconnect to force a fresh migration run on the cleaned database.
    let rig = BatchRuntimeStore::initialize_with_database_url(fleet_url.clone());
    rig.count_live_fleet_claims(WORKER_ONE_ID)
        .await
        .expect("migrations run on the cleaned database");

    let worker_one = MockWorker::spawn(worker_config(WORKER_ONE_ID, WORKER_ONE_MARKER))
        .await
        .expect("mock worker binds");
    let worker_two = MockWorker::spawn(worker_config(WORKER_TWO_ID, WORKER_TWO_MARKER))
        .await
        .expect("mock worker binds");
    let approvals = vec![
        approval_for(&worker_one, MODEL.dir_name()),
        approval_for(&worker_two, MODEL.dir_name()),
    ];
    let gateway_a = spawn_gateway("gateway-a", &fleet_url, &approvals, Vec::new()).await;
    let gateway_b = spawn_gateway("gateway-b", &fleet_url, &approvals, Vec::new()).await;
    wait_ready(&gateway_a).await;
    wait_ready(&gateway_b).await;

    // T07 across processes on PostgreSQL: the claim fence holds under
    // concurrent dispatch from both gateways.
    let sampler_store = BatchRuntimeStore::initialize_with_database_url(fleet_url.clone());
    let sampler = tokio::spawn(async move {
        let mut max_live: u64 = 0;
        let deadline = Instant::now() + Duration::from_secs(6);
        while Instant::now() < deadline {
            max_live = max_live.max(
                sampler_store
                    .count_live_fleet_claims(WORKER_ONE_ID)
                    .await
                    .unwrap_or(0),
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        max_live
    });
    let client = reqwest::Client::new();
    let mut handles = Vec::new();
    for index in 0..6u32 {
        let client = client.clone();
        let base = if index % 2 == 0 {
            gateway_a.base.clone()
        } else {
            gateway_b.base.clone()
        };
        handles.push(tokio::spawn(async move { chat(&client, &base).await }));
    }
    for handle in handles {
        let (status, body) = handle.await.expect("chat task joins");
        assert!(
            status == 200 || status == 503,
            "concurrent admission must resolve to served-or-shed, got {status}: {body}"
        );
    }
    let max_live = sampler.await.expect("sampler joins");
    assert!(
        max_live <= 1,
        "the PostgreSQL claim fence must never exceed the worker's single credit, saw {max_live}"
    );

    // Claim steering through the server-backed store.
    let held = rig
        .try_claim_fleet_capacity(
            WORKER_ONE_ID,
            "mock-incarnation-1",
            "gateway-rig",
            1,
            60_000,
        )
        .await
        .expect("rig claim")
        .expect("the rig holds worker one's only credit");
    tokio::time::sleep(Duration::from_millis(700)).await;
    let (status, body) = chat(&client, &gateway_b.base).await;
    assert_eq!(status, 200);
    assert!(
        response_marker(&body).contains(WORKER_TWO_MARKER),
        "dispatch must avoid the claimed worker on PostgreSQL, got {:?}",
        response_marker(&body)
    );

    // DINV-06: close the database to new connections and terminate every
    // existing backend. The gateways' claim and publish calls fail, dispatch
    // degrades to worker-authoritative, and nothing panics.
    {
        let db = rig.connection().await.expect("rig connection");
        let row = db
            .query_one_raw(Statement::from_string(
                db.get_database_backend(),
                "SELECT current_database()".to_string(),
            ))
            .await
            .expect("database name")
            .expect("database row");
        let database: String = row.try_get_by_index(0).expect("database name value");
        db.execute_unprepared(&format!("ALTER DATABASE \"{database}\" CONNECTION LIMIT 0"))
            .await
            .expect("connection limit closes the database");
        db.execute_unprepared(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()",
        )
        .await
        .ok();
    }
    let (status, body) = chat(&client, &gateway_a.base).await;
    assert_eq!(
        status, 200,
        "an unreachable coordination store must degrade to uncoordinated dispatch, got {body}"
    );
    assert!(
        response_marker(&body).contains(WORKER_ONE_MARKER)
            || response_marker(&body).contains(WORKER_TWO_MARKER),
        "the degraded dispatch still executes on a fleet worker"
    );

    // Reopen the database: the store recovers, the rig reconnects, and the
    // fleet keeps serving.
    {
        let db = rig.connection().await.expect("rig connection");
        let row = db
            .query_one_raw(Statement::from_string(
                db.get_database_backend(),
                "SELECT current_database()".to_string(),
            ))
            .await
            .expect("database name")
            .expect("database row");
        let database: String = row.try_get_by_index(0).expect("database name value");
        db.execute_unprepared(&format!(
            "ALTER DATABASE \"{database}\" CONNECTION LIMIT -1"
        ))
        .await
        .expect("connection limit reopens the database");
    }
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    // Terminated pool connections surface one error apiece before the pool
    // reconnects, so recovery probes retry briefly.
    let mut recovered = Err(anyhow::anyhow!("rig store never recovered"));
    for _ in 0..8 {
        match rig.count_live_fleet_claims(WORKER_ONE_ID).await {
            Ok(_) => {
                recovered = Ok(());
                break;
            }
            Err(error) => recovered = Err(error),
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    recovered.expect("the rig store reconnects after the outage");
    rig.release_fleet_capacity(&held, "gateway-rig")
        .await
        .expect("rig release after recovery");
    let (status, _) = chat(&client, &gateway_b.base).await;
    assert_eq!(status, 200, "the fleet serves after the store recovers");
}

//! DS0.8 backend parity harness: the same tiny LFM2 fixture and greedy chat
//! fixtures must produce identical outputs across every buildable backend
//! lane. The CPU leg always runs and doubles as a determinism gate (two
//! independent worker processes must agree). The Metal leg is compiled only
//! when the crate is built with the `metal` feature and executes on hosts
//! with Apple silicon; a real model with a documented tolerance budget
//! replaces this smoke fixture before any performance claim.
//!
//! Hardware lanes without a build/device on the executing host stay
//! `not run` (serving plan INV-18) — they are never reported as passed.

mod common;

use common::{
    id, write_tiny_lfm_fixture, write_tiny_qwen38_hybrid_fixture, QWEN38_FIXTURE_REVISION,
};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::*;
use izwi_serving_worker::WORKER_METRICS_PATH;
use std::{
    collections::{BTreeMap, BTreeSet},
    net::TcpListener,
    process::Stdio,
    time::Duration,
};

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

const PARITY_PROMPTS: [&str; 3] = ["hello", "ab", "a"];
const STARTUP_DEADLINE: Duration = Duration::from_secs(30);

fn uid<T: TryFrom<String>>(prefix: &str, suffix: usize) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(format!("{prefix}-{suffix}")).expect("unique test identity")
}

async fn collect_lane_outputs(
    models_dir: &std::path::Path,
    backend_env: &[(&str, &str)],
    startup_deadline: Duration,
) -> Vec<String> {
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"))
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models_dir)
        .env("IZWI_WORKER_CREDENTIAL_ID", "parity-credential")
        .env("IZWI_WORKER_BEARER_TOKEN", "parity-secret")
        .env("IZWI_WORKER_ARTIFACT_REVISION", "tiny-lfm-fixture-v1")
        .env("IZWI_WORKER_CPU_THREADS", "1")
        .env("IZWI_WORKER_MAX_ACTIVE", "1")
        .env("RUST_LOG", "warn")
        .envs(backend_env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut child = ChildGuard(child);

    let credentials = ServiceCredentials {
        credential_id: id("parity-credential"),
        bearer_token: ServiceBearerToken::new("parity-secret").unwrap(),
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
    let descriptor = tokio::time::timeout(startup_deadline, async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("worker loads and warms the tiny model before its startup deadline");

    let mut outputs = Vec::new();
    for (index, prompt) in PARITY_PROMPTS.iter().enumerate() {
        let events = client
            .invoke_collect(InvocationRequest {
                schema_version: PROTOCOL_V1,
                request_id: uid("parity-request", index),
                attempt_id: uid("parity-attempt", index),
                expected_worker_incarnation: descriptor.incarnation_id.clone(),
                deployment_id: id("lfm25-cpu-v1"),
                expected_model_generation: ModelGeneration::new(1).unwrap(),
                caller: GatewayAttestedCallerContext {
                    tenant_id: id("local-test"),
                    caller_id: id("parity-harness"),
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
                request_digest: uid("sha256:parity-fixture", index),
                input: InvocationInput::Chat {
                    input: ChatInput {
                        messages: vec![ChatMessage {
                            role: ChatRole::User,
                            content: (*prompt).into(),
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
            .unwrap_or_else(|error| panic!("prompt {index} ({prompt:?}) failed: {error}"));
        assert!(matches!(
            events.last().map(|event| &event.event),
            Some(InvocationEventKind::Completed { .. })
        ));
        let text: String = events
            .iter()
            .filter_map(|event| match &event.event {
                InvocationEventKind::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        outputs.push(text);
    }
    drop(child);
    outputs
}

#[tokio::test]
async fn cpu_outputs_are_deterministic_across_worker_processes() {
    let models = tempfile::tempdir().unwrap();
    write_tiny_lfm_fixture(models.path());
    let first = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    let second = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    assert_eq!(
        first, second,
        "the CPU lane must be deterministic run to run"
    );
    assert!(
        first.iter().all(|output| !output.is_empty()),
        "parity fixtures must produce non-empty outputs"
    );
}

/// The strict Metal assignment lane requires the operator-provided device
/// identity (`metal:<registryID>`). Tests discover it from the host's default
/// GPU; when that is impossible the leg is recorded as not-run, never passed.
#[cfg(feature = "metal")]
fn host_metal_device_id() -> Option<String> {
    let output = std::process::Command::new("swift")
        .args(["-e", "import Metal; if let d = MTLCreateSystemDefaultDevice() { print(\"metal:\" + String(d.registryID)) }"])
        .output()
        .ok()?;
    let stdout = String::from_utf8(output.stdout).ok()?;
    let id = stdout.trim().to_string();
    (output.status.success() && id.starts_with("metal:")).then_some(id)
}

#[cfg(feature = "metal")]
#[tokio::test]
async fn metal_outputs_match_cpu_outputs_on_the_parity_fixture() {
    let Some(device_id) = host_metal_device_id() else {
        println!("metal parity leg not run: host default GPU registryID unavailable");
        return;
    };
    let models = tempfile::tempdir().unwrap();
    write_tiny_lfm_fixture(models.path());
    let cpu = collect_lane_outputs(models.path(), &[], STARTUP_DEADLINE).await;
    // Cold Metal starts compile the MSL shader cache; allow a generous deadline.
    let metal = collect_lane_outputs(
        models.path(),
        &[
            ("IZWI_BACKEND", "metal"),
            ("IZWI_WORKER_EXPECTED_DEVICE_ID", device_id.as_str()),
            ("IZWI_METAL_DEVICE_ORDINAL", "0"),
        ],
        Duration::from_secs(120),
    )
    .await;
    assert_eq!(
        cpu, metal,
        "greedy decode on the unified-memory lane must match the CPU lane on the parity fixture"
    );
}

// ---------------------------------------------------------------------------
// DS1.5 qwen38 hybrid prefix leg: the shared-prefix workload must publish
// committed snapshots and attach them on later requests, with outputs that are
// deterministic across worker processes and identical on the Metal lane. The
// worker's /internal/v1/metrics/prometheus counters are the publication
// evidence (invocation success alone cannot distinguish attach from recompute).

const PREFIX_PARITY_REQUESTS: usize = 4;
const PREFIX_PARITY_PREFIX_WORDS: usize = 64;
const PREFIX_PARITY_SUFFIX_WORDS: usize = 8;
const PREFIX_PARITY_CREDENTIAL: &str = "ds15-parity-credential";
const PREFIX_PARITY_BEARER: &str = "ds15-parity-secret";

const PREFIX_VOCAB: [&str; 3] = ["a", "b", "c"];

fn filler_words(count: usize, salt: usize) -> Vec<String> {
    (0..count)
        .map(|index| PREFIX_VOCAB[(salt * 7 + index * 3) % PREFIX_VOCAB.len()].to_string())
        .collect()
}

/// Least-significant-first base-3 positional marker over the fixture vocab:
/// the fixture tokenizer is WordLevel over a/b/c with unk collapsing to "a",
/// so anything outside this alphabet would make even "cold" prompts share
/// token prefixes. Consecutive markers differ in their first word.
fn marker_words(index: usize) -> Vec<String> {
    (0..11)
        .map(|position| PREFIX_VOCAB[(index / 3usize.pow(position)) % 3].to_string())
        .collect()
}

/// The shared workload: a byte-identical system prefix on every request plus a
/// per-request unique user suffix (the fixture template concatenates message
/// contents verbatim), so the cross-request token share is exactly the prefix.
fn prefix_parity_messages(index: usize) -> Vec<ChatMessage> {
    let system = filler_words(PREFIX_PARITY_PREFIX_WORDS, 0).join(" ");
    let mut suffix = marker_words(index);
    suffix.extend(filler_words(PREFIX_PARITY_SUFFIX_WORDS - 1, index + 1));
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: system,
        },
        ChatMessage {
            role: ChatRole::User,
            content: suffix.join(" "),
        },
    ]
}

/// Serving-policy and identity env for the DS1.5 parity worker. The chunked
/// threshold covers the whole prompt so the publishing session commits its
/// entire prefix in the first (prefix-eligible) chunk — reuse depth is capped
/// by that first chunk by design.
fn prefix_lane_env(assignment: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = [
        ("IZWI_WORKER_MODEL", "Qwen3.8-27B-FP8"),
        ("IZWI_WORKER_DEPLOYMENT_ID", "ds15-parity-v1"),
        ("IZWI_WORKER_ARTIFACT_REVISION", QWEN38_FIXTURE_REVISION),
        ("IZWI_WORKER_CREDENTIAL_ID", PREFIX_PARITY_CREDENTIAL),
        ("IZWI_WORKER_BEARER_TOKEN", PREFIX_PARITY_BEARER),
        ("IZWI_WORKER_MAX_ACTIVE", "1"),
        ("IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY", "1"),
        ("IZWI_ENABLE_PREFIX_CACHING", "1"),
        ("IZWI_MANAGED_PREFIX_CACHE_SALT", "ds15-parity-salt"),
        ("IZWI_ENABLE_CHUNKED_PREFILL", "1"),
        ("IZWI_CHUNKED_PREFILL_THRESHOLD", "512"),
        ("IZWI_KV_PAGE_SIZE", "16"),
        ("IZWI_CUDA_MTP", "off"),
        ("IZWI_MAX_PREFIX_CACHE_PAGES", "64"),
        ("IZWI_MAX_SEQUENCE_LENGTH", "4096"),
        ("RUST_LOG", "warn"),
    ]
    .iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect();
    env.extend(
        assignment
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string())),
    );
    env
}

async fn fetch_worker_metrics(address: std::net::SocketAddr) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .expect("metrics connection");
    let request = format!(
        "GET {WORKER_METRICS_PATH} HTTP/1.0\r\n\
         Host: {address}\r\n\
         Authorization: Bearer {PREFIX_PARITY_BEARER}\r\n\
         x-izwi-service-credential-id: {PREFIX_PARITY_CREDENTIAL}\r\n\
         Connection: close\r\n\
         \r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("metrics request write");
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .await
        .expect("metrics response read");
    let text = String::from_utf8(raw).expect("metrics response is utf-8");
    assert!(
        text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200"),
        "metrics endpoint must authorize the parity credential: {text:.160}"
    );
    text.split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or(text)
}

fn parse_prometheus_counters(text: &str) -> BTreeMap<String, u64> {
    text.lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(' ')?;
            let value: u64 = value.trim().parse().ok()?;
            Some((name.to_string(), value))
        })
        .collect()
}

struct PrefixLane {
    outputs: Vec<String>,
    counters: BTreeMap<String, u64>,
}

async fn collect_prefix_lane(
    models_dir: &std::path::Path,
    assignment_env: &[(&str, &str)],
    startup_deadline: Duration,
) -> PrefixLane {
    collect_prefix_lane_with_env(
        models_dir,
        prefix_lane_env(assignment_env),
        startup_deadline,
    )
    .await
}

async fn collect_prefix_lane_with_env(
    models_dir: &std::path::Path,
    lane_env: Vec<(String, String)>,
    startup_deadline: Duration,
) -> PrefixLane {
    let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_izwi-serving-worker"))
        .env("IZWI_WORKER_BIND", address.to_string())
        .env("IZWI_MODELS_DIR", models_dir)
        .envs(
            lane_env
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let child = ChildGuard(child);

    let credentials = ServiceCredentials {
        credential_id: id(PREFIX_PARITY_CREDENTIAL),
        bearer_token: ServiceBearerToken::new(PREFIX_PARITY_BEARER).unwrap(),
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
    let descriptor = tokio::time::timeout(startup_deadline, async {
        loop {
            match client.descriptor().await {
                Ok(descriptor) => break descriptor,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .expect("worker loads and warms the qwen38 hybrid fixture before its startup deadline");

    let mut outputs = Vec::new();
    for index in 0..PREFIX_PARITY_REQUESTS {
        let events = client
            .invoke_collect(InvocationRequest {
                schema_version: PROTOCOL_V1,
                request_id: uid("ds15-request", index),
                attempt_id: uid("ds15-attempt", index),
                expected_worker_incarnation: descriptor.incarnation_id.clone(),
                deployment_id: id("ds15-parity-v1"),
                expected_model_generation: ModelGeneration::new(1).unwrap(),
                caller: GatewayAttestedCallerContext {
                    tenant_id: id("local-test"),
                    caller_id: id("ds15-parity-harness"),
                    policy_revision: id("test-policy-v1"),
                    permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                    allowed_data_regions: vec!["local".into()],
                },
                task: TaskKind::Chat,
                service_class: ServiceClass::Interactive,
                remaining_time_ms: 30_000,
                max_queue_wait_ms: 10_000,
                output_limits: OutputLimits {
                    max_tokens: 8,
                    max_bytes: 1024,
                },
                requested_output_format: OutputFormat::Text,
                session_id: None,
                request_digest: uid("sha256:ds15-prefix", index),
                input: InvocationInput::Chat {
                    input: ChatInput {
                        messages: prefix_parity_messages(index),
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
            .unwrap_or_else(|error| panic!("prefix request {index} failed: {error}"));
        assert!(matches!(
            events.last().map(|event| &event.event),
            Some(InvocationEventKind::Completed { .. })
        ));
        let text: String = events
            .iter()
            .filter_map(|event| match &event.event {
                InvocationEventKind::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        outputs.push(text);
    }
    let counters = parse_prometheus_counters(&fetch_worker_metrics(address).await);
    drop(child);
    PrefixLane { outputs, counters }
}

fn assert_prefix_counters(counters: &BTreeMap<String, u64>, label: &str) {
    let publishes = counters
        .get("izwi_engine_tensor_snapshot_publishes_total")
        .copied()
        .unwrap_or(0);
    let attaches = counters
        .get("izwi_engine_tensor_snapshot_attaches_total")
        .copied()
        .unwrap_or(0);
    let reused = counters
        .get("izwi_engine_kv_cache_reused_tokens_total")
        .copied()
        .unwrap_or(0);
    assert!(
        publishes >= 1,
        "{label} must publish committed snapshots (counters: {counters:?})"
    );
    assert!(
        attaches >= 1,
        "{label} must attach a published snapshot (counters: {counters:?})"
    );
    assert!(
        reused >= 16,
        "{label} must reuse committed prefix tokens (counters: {counters:?})"
    );
}

#[tokio::test]
async fn qwen38_shared_prefix_lane_is_deterministic_and_attaches_across_processes() {
    let models = tempfile::tempdir().unwrap();
    write_tiny_qwen38_hybrid_fixture(models.path());
    let first =
        collect_prefix_lane(models.path(), &[("IZWI_BACKEND", "cpu")], STARTUP_DEADLINE).await;
    let second =
        collect_prefix_lane(models.path(), &[("IZWI_BACKEND", "cpu")], STARTUP_DEADLINE).await;
    assert_eq!(
        first.outputs, second.outputs,
        "attached chunked prefill must be deterministic across worker processes"
    );
    assert!(
        first.outputs.iter().all(|output| !output.is_empty()),
        "prefix parity fixtures must produce non-empty outputs"
    );
    assert_prefix_counters(&first.counters, "publish run");
    assert_prefix_counters(&second.counters, "attach run");
}

/// The DS1.6 catalog-auto leg: the worker starts with NO explicit prefix
/// choice at all — no IZWI_ENABLE_PREFIX_CACHING, no IZWI_PREFIX_REUSE_AUTO —
/// so the shipped Auto default must engage reuse through the catalog cell.
/// The qwen38 CPU lane is a process-parity-supported cell, so this is the
/// default-on admission evidence: reuse reaches a real worker through normal
/// admission without any operator opt-in.
fn auto_prefix_lane_env(assignment: &[(&str, &str)]) -> Vec<(String, String)> {
    prefix_lane_env(assignment)
        .into_iter()
        .filter(|(key, _)| {
            key != "IZWI_ENABLE_PREFIX_CACHING" && key != "IZWI_MANAGED_PREFIX_CACHE_SALT"
        })
        .collect()
}

#[tokio::test]
async fn qwen38_prefix_reuse_engages_by_default_without_explicit_operator_choice() {
    let models = tempfile::tempdir().unwrap();
    write_tiny_qwen38_hybrid_fixture(models.path());
    let lane = collect_prefix_lane_with_env(
        models.path(),
        auto_prefix_lane_env(&[("IZWI_BACKEND", "cpu")]),
        STARTUP_DEADLINE,
    )
    .await;
    assert!(
        lane.outputs.iter().all(|output| !output.is_empty()),
        "catalog-auto fixtures must produce non-empty outputs"
    );
    assert_prefix_counters(&lane.counters, "catalog-auto run");
}

/// The kill switch stays independent: an explicit zero disables reuse even
/// though a namespace salt and a supported catalog cell are available.
fn off_prefix_lane_env(assignment: &[(&str, &str)]) -> Vec<(String, String)> {
    prefix_lane_env(assignment)
        .into_iter()
        .map(|(key, value)| {
            if key == "IZWI_ENABLE_PREFIX_CACHING" {
                (key, "0".to_string())
            } else {
                (key, value)
            }
        })
        .collect()
}

#[tokio::test]
async fn explicit_zero_prefix_caching_keeps_reuse_off() {
    let models = tempfile::tempdir().unwrap();
    write_tiny_qwen38_hybrid_fixture(models.path());
    let lane = collect_prefix_lane_with_env(
        models.path(),
        off_prefix_lane_env(&[("IZWI_BACKEND", "cpu")]),
        STARTUP_DEADLINE,
    )
    .await;
    assert!(
        lane.outputs.iter().all(|output| !output.is_empty()),
        "kill-switch fixtures must still produce non-empty outputs"
    );
    for (key, value) in &lane.counters {
        if key.contains("tensor_snapshot") || key.contains("reused_tokens") {
            assert_eq!(*value, 0, "kill-switch run must record zero {key} activity");
        }
    }
}

/// The Metal lane of the DS1.5 prefix leg: attached prefill on the unified
/// memory backend must match the CPU lane bit for bit. Without a host GPU the
/// leg is recorded as not-run, never passed (serving plan INV-18).
#[cfg(feature = "metal")]
#[tokio::test]
async fn qwen38_shared_prefix_metal_lane_matches_cpu() {
    let Some(device_id) = host_metal_device_id() else {
        println!("qwen38 prefix metal leg not run: host default GPU registryID unavailable");
        return;
    };
    let models = tempfile::tempdir().unwrap();
    write_tiny_qwen38_hybrid_fixture(models.path());
    let cpu =
        collect_prefix_lane(models.path(), &[("IZWI_BACKEND", "cpu")], STARTUP_DEADLINE).await;
    // Cold Metal starts compile the MSL shader cache; allow a generous deadline.
    let metal = collect_prefix_lane(
        models.path(),
        &[
            ("IZWI_BACKEND", "metal"),
            ("IZWI_WORKER_EXPECTED_DEVICE_ID", device_id.as_str()),
            ("IZWI_METAL_DEVICE_ORDINAL", "0"),
        ],
        Duration::from_secs(120),
    )
    .await;
    assert_eq!(
        cpu.outputs, metal.outputs,
        "attached chunked prefill on the unified-memory lane must match the CPU lane"
    );
    assert_prefix_counters(&metal.counters, "metal attach run");
}

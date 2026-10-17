//! Per-worker supervision against real supervisor processes: one actor per
//! worker, the model-load bound, liveness probing and quarantine.
//!
//! The workers are python fakes that serve the exact readiness contract (the
//! descriptor and status JSON are rendered here through the protocol serde)
//! and can be told to never become ready, to die, or to stop answering.

// The fake workers are executable scripts launched through unix permissions.
#![cfg(unix)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use izwi_serving_protocol::{
    BackendKind, CancellationBehavior, Capability, CapacitySnapshot, DeploymentId,
    DeviceAssignment, IncarnationId, InputFormat, LoadedDeployment, ModelAlias, ModelGeneration,
    ModelReadiness, NodeId, OutputFormat, TaskKind, WorkerDescriptor, WorkerFeature, WorkerId,
    WorkerProcessState, WorkerStatus, PROTOCOL_V1,
};
use izwi_test_process::ProcessGuard;

const FAKE_WORKER_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MANIFEST = json.load(open("{manifest}"))
EVENTS_PATH = "{manifest}" + ".events"
WORKER_ID = os.environ["IZWI_WORKER_ID"]
ENTRY = MANIFEST[WORKER_ID]
DESCRIPTOR = json.load(open(ENTRY["descriptor"]))
STATUS = json.load(open(ENTRY["status"]))
for doc in (DESCRIPTOR, STATUS):
    doc["worker_id"] = WORKER_ID
    doc["node_id"] = os.environ["IZWI_WORKER_NODE_ID"]
    doc["incarnation_id"] = os.environ["IZWI_WORKER_INCARNATION_ID"]
SEQUENCE = 0
LOCK = threading.Lock()
STARTED = time.monotonic()
with open(ENTRY["launches"], "a") as launches:
    launches.write(os.environ["IZWI_WORKER_INCARNATION_ID"] + "\n")
FIRST_STATUS_SERVED = False

def record_event(kind):
    # One shared, append-only log on the system-wide monotonic clock, so a
    # test can order events across workers without trusting thread timing.
    with open(EVENTS_PATH, "a") as events:
        events.write(f"{kind} {WORKER_ID} {time.monotonic()}\n")

record_event("launch")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global SEQUENCE, FIRST_STATUS_SERVED
        hang_after = ENTRY.get("hang_status_after_secs")
        if hang_after is not None and time.monotonic() - STARTED > float(hang_after):
            # A wedged runtime: the listener accepts but never answers.
            time.sleep(3600)
        if self.path == "/internal/v1/worker":
            body = json.dumps(DESCRIPTOR).encode()
        elif self.path == "/internal/v1/status":
            with LOCK:
                SEQUENCE += 1
                STATUS["status_sequence"] = SEQUENCE
                if not FIRST_STATUS_SERVED:
                    FIRST_STATUS_SERVED = True
                    record_event("status")
            body = json.dumps(STATUS).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

def exit_when_stdin_closes():
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except Exception:
        pass
    os._exit(0)

if not ENTRY.get("serve", True):
    if ENTRY.get("die_after_secs") is not None:
        time.sleep(float(ENTRY["die_after_secs"]))
        os._exit(1)
    exit_when_stdin_closes()

server = ThreadingHTTPServer(
    ("127.0.0.1", int(os.environ["IZWI_WORKER_BIND"].rsplit(":", 1)[1])), Handler
)
threading.Thread(target=server.serve_forever, daemon=True).start()
exit_when_stdin_closes()
"#;

/// How one fake worker behaves.
#[derive(Clone, Default)]
struct Behavior {
    /// Never serve readiness (a load that never finishes).
    never_ready: bool,
    /// Stop answering any request this long after starting.
    hang_status_after_secs: Option<f64>,
    /// With `never_ready`, exit with a failure this long after starting.
    die_after_secs: Option<f64>,
}

struct Rig {
    _tempdir: tempfile::TempDir,
    root: PathBuf,
    config: PathBuf,
    worker_binary: PathBuf,
    stderr: PathBuf,
}

impl Rig {
    /// One CPU worker per entry, in config order.
    fn build(workers: &[(&'static str, Behavior)], node_extra: &str) -> Self {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().to_path_buf();
        for dir in ["run", "work", "models"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        let mut manifest = serde_json::Map::new();
        let mut config = format!(
            "schema_version = 2\nnode_id = \"node-a\"\nworking_directory = \"{}\"\nruntime_directory = \"{}\"\nhost_memory_budget_bytes = 8589934592\n{node_extra}\n[shutdown]\ndrain_grace_ms = 500\ncancellation_grace_ms = 200\ntermination_grace_ms = 500\n",
            root.join("work").display(),
            root.join("run").display(),
        );
        for (worker_id, behavior) in workers {
            let descriptor = WorkerDescriptor {
                schema_version: PROTOCOL_V1,
                supported_protocol_versions: vec![PROTOCOL_V1],
                worker_id: WorkerId::new(*worker_id).unwrap(),
                node_id: NodeId::new("node-a").unwrap(),
                incarnation_id: IncarnationId::new("placeholder").unwrap(),
                build_version: "fake".into(),
                assignment: DeviceAssignment::Cpu {
                    thread_budget: 1,
                    affinity: vec![0],
                    host_memory_limit_bytes: 1024,
                },
                features: BTreeSet::from([WorkerFeature::Streaming]),
            };
            let status = WorkerStatus {
                schema_version: PROTOCOL_V1,
                worker_id: WorkerId::new(*worker_id).unwrap(),
                node_id: NodeId::new("node-a").unwrap(),
                incarnation_id: IncarnationId::new("placeholder").unwrap(),
                status_sequence: 1,
                process_state: WorkerProcessState::Running,
                deployments: vec![LoadedDeployment {
                    deployment_id: DeploymentId::new("chat-prod").unwrap(),
                    public_model: ModelAlias::new("chat-model").unwrap(),
                    artifact_revision: izwi_serving_protocol::ArtifactRevision::new("rev-1")
                        .unwrap(),
                    model_generation: ModelGeneration::new(1).unwrap(),
                    task: TaskKind::Chat,
                    backend: BackendKind::Cpu,
                    precision: "f32".into(),
                    execution_representation: "gguf".into(),
                    tokenizer_revision: None,
                    readiness: ModelReadiness::Ready,
                    capability: Capability {
                        task: TaskKind::Chat,
                        streaming: true,
                        realtime: false,
                        cancellation: CancellationBehavior::Cooperative,
                        accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                        output_formats: BTreeSet::from([OutputFormat::Text]),
                        max_input_bytes: 1048576,
                        max_context_tokens: Some(32),
                        max_output_tokens: Some(32),
                    },
                    kv_cache_usage_pct: None,
                    prefix_hits_total: None,
                    prefix_queries_total: None,
                    prefix_evictions_total: None,
                    kv_host_pages: None,
                    kv_demotions_total: None,
                    kv_promotions_total: None,
                    kv_promotion_latency_avg_seconds: None,
                    tokens_out_per_s_ema: None,
                    observation_cost_units: None,
                }],
                capacity: CapacitySnapshot {
                    max_active_invocations: 1,
                    active_invocations: 0,
                    max_queued_invocations: 0,
                    queued_invocations: 0,
                    max_sessions: 0,
                    reserved_sessions: 0,
                    available_admission_credits: 1,
                    outstanding_cost_units: 0,
                },
            };
            let descriptor_path = root.join(format!("{worker_id}.descriptor.json"));
            let status_path = root.join(format!("{worker_id}.status.json"));
            std::fs::write(&descriptor_path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
            std::fs::write(&status_path, serde_json::to_vec(&status).unwrap()).unwrap();
            let mut entry = serde_json::json!({
                "descriptor": descriptor_path.display().to_string(),
                "status": status_path.display().to_string(),
                "launches": root.join(format!("{worker_id}.launches")).display().to_string(),
                "serve": !behavior.never_ready,
            });
            if let Some(after) = behavior.hang_status_after_secs {
                entry["hang_status_after_secs"] = serde_json::json!(after);
            }
            if let Some(after) = behavior.die_after_secs {
                entry["die_after_secs"] = serde_json::json!(after);
            }
            manifest.insert(worker_id.to_string(), entry);
            config.push_str(&format!(
                "\n[[workers]]\nworker_id = \"{worker_id}\"\nbind = \"127.0.0.1:{port}\"\nbinary = \"cpu\"\ncredential_id = \"cred-a\"\nbearer_token_env = \"WORKER_TOKEN\"\nmax_active_invocations = 1\nmax_request_bytes = 1048576\n\n[workers.assignment]\nbackend = \"cpu\"\nthread_budget = 1\naffinity = [0]\nhost_memory_limit_bytes = 1024\n\n[workers.deployment]\ndeployment_id = \"chat-prod\"\npublic_model = \"chat-model\"\nartifact_revision = \"rev-1\"\nmodel_generation = 1\ntask = \"chat\"\nbackend = \"cpu\"\nprecision = \"f32\"\nexecution_representation = \"gguf\"\nmodels_directory = \"{models}\"\n\n[workers.deployment.capability]\nstreaming = true\nrealtime = false\ncancellation = \"cooperative\"\naccepted_input_formats = [\"chat_messages\"]\noutput_formats = [\"text\"]\nmax_input_bytes = 1048576\nmax_context_tokens = 32\nmax_output_tokens = 32\n",
                port = free_port(),
                models = root.join("models").display(),
            ));
        }
        let manifest_path = root.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let worker_binary = root.join("fake-worker.py");
        std::fs::write(
            &worker_binary,
            FAKE_WORKER_SCRIPT.replace("{manifest}", &manifest_path.display().to_string()),
        )
        .unwrap();
        make_executable(&worker_binary);
        let config_path = root.join("node.toml");
        std::fs::write(&config_path, config).unwrap();
        Self {
            _tempdir: tempdir,
            stderr: root.join("supervisor.stderr"),
            root,
            config: config_path,
            worker_binary,
        }
    }

    fn supervisor_command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"));
        command
            .arg("--config")
            .arg(&self.config)
            .arg("--cpu-worker-binary")
            .arg(&self.worker_binary)
            .arg("--cpu-ids")
            .arg("0,1")
            .arg("--allocatable-host-memory-bytes")
            .arg("8589934592")
            .env("WORKER_TOKEN", "secret-token");
        command
    }

    /// Makes later launches of `worker_id` serve readiness.
    fn heal(&self, worker_id: &str) {
        let path = self.root.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest[worker_id]["serve"] = serde_json::json!(true);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    fn spawn_supervisor(&self) -> SupervisorGuard {
        let stderr = std::fs::File::create(&self.stderr).unwrap();
        let child = ProcessGuard::spawn(
            self.supervisor_command()
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::from(stderr)),
        )
        .expect("spawn supervisor");
        SupervisorGuard(child)
    }

    fn stderr_text(&self) -> String {
        std::fs::read_to_string(&self.stderr).unwrap_or_default()
    }

    /// The monotonic time of `worker_id`'s first `kind` event in the shared
    /// fake-worker event log.
    fn first_event(&self, kind: &str, worker_id: &str) -> Option<f64> {
        let events = std::fs::read_to_string(self.root.join("manifest.json.events")).ok()?;
        events.lines().find_map(|line| {
            let mut fields = line.split(' ');
            (fields.next() == Some(kind) && fields.next() == Some(worker_id))
                .then(|| fields.next()?.parse().ok())
                .flatten()
        })
    }

    fn launches(&self, worker_id: &str) -> usize {
        std::fs::read_to_string(self.root.join(format!("{worker_id}.launches")))
            .map(|text| text.lines().count())
            .unwrap_or(0)
    }
}

struct SupervisorGuard(ProcessGuard);

impl SupervisorGuard {
    fn signal(&self, signal: &str) {
        let status = Command::new("kill")
            .arg(format!("-{signal}"))
            .arg(self.0.id().to_string())
            .status()
            .unwrap();
        assert!(status.success());
    }
}

fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_until(deadline: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    condition()
}

/// A worker whose load never finishes holds one load slot; with two slots
/// the next worker still launches, and the supervisor still answers
/// diagnostics, instead of waiting out the stuck worker's startup timeout.
#[test]
fn a_stuck_load_does_not_hold_back_other_workers() {
    let rig = Rig::build(
        &[
            (
                "worker-a",
                Behavior {
                    never_ready: true,
                    ..Behavior::default()
                },
            ),
            ("worker-b", Behavior::default()),
        ],
        "max_parallel_model_loads = 2\n\n[readiness]\nstartup_timeout_ms = 60000\npoll_interval_ms = 50\n",
    );
    let supervisor = rig.spawn_supervisor();
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("worker-b ready as incarnation")),
        "worker-b must not wait for worker-a's 60 s load; stderr:\n{}",
        rig.stderr_text()
    );
    supervisor.signal("USR1");
    assert!(
        wait_until(Duration::from_secs(5), || rig
            .stderr_text()
            .contains("worker=worker-a state=starting")),
        "diagnostics must answer while a load is in flight; stderr:\n{}",
        rig.stderr_text()
    );
}

/// With one load slot, loads run one at a time. The actors race for the
/// slot, so either worker may load first; the second must launch only once
/// the first's load has ended.
#[test]
fn one_load_slot_serializes_loads() {
    let rig = Rig::build(
        &[
            (
                "worker-a",
                Behavior {
                    never_ready: true,
                    ..Behavior::default()
                },
            ),
            ("worker-b", Behavior::default()),
        ],
        "max_parallel_model_loads = 1\n\n[readiness]\nstartup_timeout_ms = 2000\npoll_interval_ms = 50\n\n[restart]\ninitial_backoff_ms = 60000\nmaximum_backoff_ms = 60000\n",
    );
    let _supervisor = rig.spawn_supervisor();
    // Converged outcome: both workers have launched (worker-b is ready and
    // worker-a has loaded before or after it), whichever took the slot first.
    assert!(
        wait_until(Duration::from_secs(30), || rig
            .stderr_text()
            .contains("worker-b ready as incarnation")
            && rig.launches("worker-a") >= 1),
        "both workers must launch through the one slot; stderr:\n{}",
        rig.stderr_text()
    );
    let event = |kind: &str, worker_id: &str| rig.first_event(kind, worker_id);
    let a_launch = event("launch", "worker-a").expect("worker-a launched");
    let b_launch = event("launch", "worker-b").expect("worker-b launched");
    if a_launch < b_launch {
        // worker-a never turns ready, so it holds the slot until its 2 s
        // startup timeout; worker-b cannot launch inside that window. The
        // launch events are written once each interpreter has started, so
        // allow up to a second of start-up skew between the two.
        assert!(
            b_launch - a_launch >= 1.0,
            "worker-b launched {:.3} s after worker-a, inside worker-a's load; stderr:\n{}",
            b_launch - a_launch,
            rig.stderr_text()
        );
        assert!(rig
            .stderr_text()
            .contains("worker worker-a failed readiness"));
    } else {
        // worker-b's load ends when it serves a ready status; worker-a can
        // only launch after that.
        let b_status = event("status", "worker-b").expect("worker-b served status");
        assert!(
            a_launch > b_status,
            "worker-a launched while worker-b held the only slot; stderr:\n{}",
            rig.stderr_text()
        );
    }
    assert_eq!(rig.launches("worker-b"), 1);
}

/// A worker that stops answering (a wedged device or a deadlock) keeps its
/// process and listener; the liveness probe restarts it.
#[test]
fn a_worker_that_stops_answering_is_restarted() {
    let rig = Rig::build(
        &[(
            "worker-a",
            Behavior {
                hang_status_after_secs: Some(1.5),
                ..Behavior::default()
            },
        )],
        "\n[readiness]\nstartup_timeout_ms = 5000\npoll_interval_ms = 50\n\n[liveness]\nperiod_ms = 300\ntimeout_ms = 200\nfailure_threshold = 2\n\n[restart]\ninitial_backoff_ms = 100\nmaximum_backoff_ms = 100\n",
    );
    let _supervisor = rig.spawn_supervisor();
    assert!(
        wait_until(Duration::from_secs(20), || rig.launches("worker-a") >= 2),
        "the hung worker must be restarted; stderr:\n{}",
        rig.stderr_text()
    );
    let stderr = rig.stderr_text();
    assert!(
        stderr.contains("worker worker-a failed 2 consecutive liveness probes"),
        "{stderr}"
    );
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .matches("worker-a ready as incarnation")
            .count()
            >= 2),
        "the replacement incarnation must come back ready; stderr:\n{}",
        rig.stderr_text()
    );
}

fn crash_looping() -> Behavior {
    Behavior {
        never_ready: true,
        die_after_secs: Some(0.0),
        ..Behavior::default()
    }
}

const QUARANTINE_FAST: &str = "\n[readiness]\nstartup_timeout_ms = 5000\npoll_interval_ms = 50\n\n[restart]\ninitial_backoff_ms = 50\nmaximum_backoff_ms = 50\nmax_restarts_per_window = 1\njitter_percent = 0\n";

/// Quarantine is not permanent: the worker gets one more launch every
/// quarantine_retry_ms.
#[test]
fn a_quarantined_worker_is_retried_on_a_slow_cadence() {
    let rig = Rig::build(
        &[("worker-a", crash_looping())],
        &format!("{QUARANTINE_FAST}quarantine_retry_ms = 1500\n"),
    );
    let _supervisor = rig.spawn_supervisor();
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("quarantined; retrying it in 1 s")),
        "stderr:\n{}",
        rig.stderr_text()
    );
    let quarantined_launches = rig.launches("worker-a");
    rig.heal("worker-a");
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("worker-a ready as incarnation")),
        "the retry must launch the worker again; stderr:\n{}",
        rig.stderr_text()
    );
    assert_eq!(rig.launches("worker-a"), quarantined_launches + 1);
}

/// With retries off, `--clear-quarantine` relaunches the worker at once.
#[test]
fn the_clear_quarantine_command_relaunches_a_quarantined_worker() {
    let rig = Rig::build(
        &[("worker-a", crash_looping())],
        &format!("{QUARANTINE_FAST}quarantine_retry_ms = 0\n"),
    );
    let _supervisor = rig.spawn_supervisor();
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("quarantined until an operator clears it")),
        "stderr:\n{}",
        rig.stderr_text()
    );
    rig.heal("worker-a");
    let unknown = rig
        .supervisor_command()
        .args(["--clear-quarantine", "worker-z"])
        .output()
        .unwrap();
    assert!(!unknown.status.success(), "an unknown worker is refused");
    let clear = rig
        .supervisor_command()
        .args(["--clear-quarantine", "worker-a"])
        .output()
        .unwrap();
    assert!(
        clear.status.success(),
        "{}",
        String::from_utf8_lossy(&clear.stderr)
    );
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("worker-a ready as incarnation")),
        "the cleared worker must launch again; stderr:\n{}",
        rig.stderr_text()
    );
}

/// SIGHUP clears every quarantine.
#[test]
fn sighup_clears_every_quarantine() {
    let rig = Rig::build(
        &[("worker-a", crash_looping())],
        &format!("{QUARANTINE_FAST}quarantine_retry_ms = 0\n"),
    );
    let supervisor = rig.spawn_supervisor();
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("quarantined until an operator clears it")),
        "stderr:\n{}",
        rig.stderr_text()
    );
    rig.heal("worker-a");
    supervisor.signal("HUP");
    assert!(
        wait_until(Duration::from_secs(10), || rig
            .stderr_text()
            .contains("worker-a ready as incarnation")),
        "SIGHUP must relaunch the quarantined worker; stderr:\n{}",
        rig.stderr_text()
    );
}

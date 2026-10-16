//! DS6 (T37 supervisor half): coordinated blue-green rollout process tests.
//!
//! Each test runs the real supervisor binary against a node config whose
//! workers are python fakes that serve the exact readiness contract. The
//! fakes' descriptor and status JSON are rendered in this test through the
//! protocol serde so the ReadinessTracker verifies the same documents a
//! real worker produces. The rollout is exercised end to end on disk: the
//! approvals file transitions, the persisted rollout state, replacement
//! liveness, and old-generation drain are all asserted on real files and
//! real processes.

// The fake workers are executable scripts launched through unix permissions.
#![cfg(unix)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use izwi_serving_protocol::{
    BackendKind, CancellationBehavior, Capability, CapacitySnapshot, DeploymentId,
    DeviceAssignment, IncarnationId, InputFormat, LoadedDeployment, ModelAlias, ModelGeneration,
    ModelReadiness, NodeId, OutputFormat, TaskKind, WorkerDescriptor, WorkerFeature,
    WorkerId, WorkerProcessState, WorkerStatus, PROTOCOL_V1,
};
use izwi_test_process::ProcessGuard;

const FAKE_WORKER_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MANIFEST = json.load(open("{manifest}"))
WORKER_ID = os.environ["IZWI_WORKER_ID"]
NODE_ID = os.environ["IZWI_WORKER_NODE_ID"]
INCARNATION = os.environ["IZWI_WORKER_INCARNATION_ID"]
ENTRY = MANIFEST[WORKER_ID]
DESCRIPTOR = json.load(open(ENTRY["descriptor"]))
STATUS = json.load(open(ENTRY["status"]))
for doc in (DESCRIPTOR, STATUS):
    doc["worker_id"] = WORKER_ID
    doc["node_id"] = NODE_ID
    doc["incarnation_id"] = INCARNATION
SEQUENCE = 0
LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global SEQUENCE
        if self.path == "/internal/v1/worker":
            body = json.dumps(DESCRIPTOR).encode()
        elif self.path == "/internal/v1/status":
            with LOCK:
                SEQUENCE += 1
                STATUS["status_sequence"] = SEQUENCE
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

def write_exit_marker():
    marker = ENTRY.get("exit_marker")
    if marker:
        open(marker, "w").write("exited")

def exit_when_stdin_closes(server):
    try:
        while sys.stdin.buffer.read(4096):
            pass
    except Exception:
        pass
    write_exit_marker()
    server.shutdown()
    os._exit(0)

if not ENTRY.get("serve", True):
    die_after = ENTRY.get("die_after_secs")
    if die_after is not None:
        time.sleep(float(die_after))
        os._exit(1)
    time.sleep(300)
    os._exit(0)

server = ThreadingHTTPServer(
    ("127.0.0.1", int(os.environ["IZWI_WORKER_BIND"].rsplit(":", 1)[1])), Handler
)
threading.Thread(target=server.serve_forever, daemon=True).start()
die_after = ENTRY.get("die_after_secs")
if die_after is not None:
    threading.Thread(
        target=lambda: (time.sleep(float(die_after)), os._exit(1)), daemon=True
    ).start()
exit_when_stdin_closes(server)
"#;

/// A gateway admin endpoint that reports whatever status file the test
/// writes, so the test decides when "the gateway" adopted a view.
const FAKE_GATEWAY_SCRIPT: &str = r#"#!/usr/bin/env python3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS, KEY, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/internal/admin/approvals":
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Authorization") != "Bearer " + KEY:
            self.send_response(401)
            self.end_headers()
            return
        body = open(STATUS, "rb").read()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"#;

const GATEWAY_ADMIN_KEY: &str = "rollout-rig-gateway-admin-key";

/// One fake worker the test renders configs and payloads for.
struct FakeWorker {
    worker_id: &'static str,
    token_env: &'static str,
    deployment_id: &'static str,
    public_model: &'static str,
    generation: u64,
}

const OLD_WORKER: FakeWorker = FakeWorker {
    worker_id: "worker-old",
    token_env: "TOKEN_OLD",
    deployment_id: "chat-prod",
    public_model: "chat-model",
    generation: 1,
};
const REPLACEMENT: FakeWorker = FakeWorker {
    worker_id: "worker-new",
    token_env: "TOKEN_NEW",
    deployment_id: "chat-prod",
    public_model: "chat-model",
    generation: 2,
};

struct Rig {
    root: PathBuf,
    current_config: PathBuf,
    target_config: PathBuf,
    plan: PathBuf,
    approvals: PathBuf,
    original_approvals: String,
    worker_binary: PathBuf,
    old_port: u16,
}

impl Rig {
    fn build(window_secs: u64, abort_grace_secs: u64) -> Self {
        Self::build_with_commit_grace(window_secs, abort_grace_secs, 1)
    }

    /// The rig's gateways are modelled with a 1s approvals refresh, the
    /// smallest the gateway accepts, so windows of 2s and up are valid.
    fn build_with_commit_grace(
        window_secs: u64,
        abort_grace_secs: u64,
        commit_grace_secs: u64,
    ) -> Self {
        Self::build_full(window_secs, abort_grace_secs, commit_grace_secs, false)
    }

    /// The replacement takes over the old worker's bind, as on a one-GPU
    /// node where both generations want the same device: the two cannot run
    /// side by side, so the rollout recreates.
    fn build_recreate() -> Self {
        Self::build_full(2, 1, 1, true)
    }

    fn build_full(
        window_secs: u64,
        abort_grace_secs: u64,
        commit_grace_secs: u64,
        share_bind: bool,
    ) -> Self {
        let root = tempfile::tempdir().unwrap().keep();
        let run_dir = root.join("run");
        let work_dir = root.join("work");
        let models_dir = root.join("models");
        for dir in [&run_dir, &work_dir, &models_dir] {
            std::fs::create_dir_all(dir).unwrap();
        }

        let old_port = free_port();
        let new_port = if share_bind { old_port } else { free_port() };
        let workers = [
            (OLD_WORKER, old_port, "TOKEN_OLD"),
            (REPLACEMENT, new_port, "TOKEN_NEW"),
        ];

        // Current config: only the old worker. Target config: only the
        // replacement, at the new generation.
        let mut manifest = serde_json::Map::new();
        for (worker, _port, _token) in workers {
            let assignment = DeviceAssignment::Cpu {
                thread_budget: 1,
                affinity: vec![0],
                host_memory_limit_bytes: 1024,
            };
            let descriptor = WorkerDescriptor {
                schema_version: PROTOCOL_V1,
                supported_protocol_versions: vec![PROTOCOL_V1],
                worker_id: WorkerId::new(worker.worker_id).unwrap(),
                node_id: NodeId::new("node-a").unwrap(),
                incarnation_id: IncarnationId::new("placeholder").unwrap(),
                build_version: "fake".into(),
                assignment,
                features: BTreeSet::from([WorkerFeature::Streaming]),
            };
            let status = WorkerStatus {
                schema_version: PROTOCOL_V1,
                worker_id: WorkerId::new(worker.worker_id).unwrap(),
                node_id: NodeId::new("node-a").unwrap(),
                incarnation_id: IncarnationId::new("placeholder").unwrap(),
                status_sequence: 1,
                process_state: WorkerProcessState::Running,
                deployments: vec![LoadedDeployment {
                    deployment_id: DeploymentId::new(worker.deployment_id).unwrap(),
                    public_model: ModelAlias::new(worker.public_model).unwrap(),
                    artifact_revision: izwi_serving_protocol::ArtifactRevision::new("rev-1")
                        .unwrap(),
                    model_generation: ModelGeneration::new(worker.generation).unwrap(),
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
                    max_active_invocations: 4,
                    active_invocations: 0,
                    max_queued_invocations: 0,
                    queued_invocations: 0,
                    max_sessions: 0,
                    reserved_sessions: 0,
                    available_admission_credits: 4,
                    outstanding_cost_units: 0,
                },
            };
            let descriptor_path = root.join(format!("{}.descriptor.json", worker.worker_id));
            let status_path = root.join(format!("{}.status.json", worker.worker_id));
            std::fs::write(&descriptor_path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
            std::fs::write(&status_path, serde_json::to_vec(&status).unwrap()).unwrap();
            manifest.insert(
                worker.worker_id.to_string(),
                serde_json::json!({
                    "descriptor": descriptor_path.display().to_string(),
                    "status": status_path.display().to_string(),
                    "exit_marker": root.join(format!("{}.exited", worker.worker_id)).display().to_string(),
                }),
            );
        }

        // The manifest is baked into the generated fake worker script.
        let manifest_path = root.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let script = FAKE_WORKER_SCRIPT.replace("{manifest}", &manifest_path.display().to_string());
        let worker_binary = root.join("fake-worker.py");
        std::fs::write(&worker_binary, script).unwrap();
        make_executable(&worker_binary);

        let current_config = root.join("current-node.toml");
        std::fs::write(
            &current_config,
            node_toml(&root, &[(&OLD_WORKER, old_port)]),
        )
        .unwrap();
        let target_config = root.join("target-node.toml");
        std::fs::write(
            &target_config,
            node_toml(&root, &[(&REPLACEMENT, new_port)]),
        )
        .unwrap();

        let approvals = root.join("approvals.txt");
        let original_approvals = format!(
            "# fleet approvals\nv1|http://127.0.0.1:{old_port}|node-a|worker-old|chat|chat-model|chat-prod|1\nv1|http://127.0.0.1:{free}|node-a|asr-worker|speech_to_text|asr-model|asr-prod|4\n",
            free = free_port(),
        );
        std::fs::write(&approvals, &original_approvals).unwrap();

        let plan = root.join("rollout.toml");
        std::fs::write(
            &plan,
            format!(
                "schema_version = 1\ntarget_node_config = \"{}\"\nshared_approvals_path = \"{}\"\ncanary_worker_id = \"{}\"\ngateway_approvals_refresh_secs = 1\nwindow_secs = {window_secs}\nabort_grace_secs = {abort_grace_secs}\ncommit_grace_secs = {commit_grace_secs}\n",
                target_config.display(),
                approvals.display(),
                REPLACEMENT.worker_id,
            ),
        )
        .unwrap();

        Self {
            root,
            current_config,
            target_config,
            plan,
            approvals,
            original_approvals,
            worker_binary,
            old_port,
        }
    }

    fn runtime_directory(&self) -> PathBuf {
        self.root.join("run")
    }

    fn state_file(&self) -> PathBuf {
        self.runtime_directory().join("rollout-state.json")
    }

    fn spawn_supervisor(&self, extra_args: &[&str]) -> ProcessGuard {
        let mut command = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"));
        command
            .arg("--config")
            .arg(&self.current_config)
            .arg("--cpu-worker-binary")
            .arg(&self.worker_binary)
            .arg("--cpu-ids")
            .arg("0,1")
            .arg("--allocatable-host-memory-bytes")
            .arg("8589934592")
            .env("TOKEN_OLD", "secret-old")
            .env("TOKEN_NEW", "secret-new")
            .env("ROLLOUT_RIG_GATEWAY_ADMIN_KEY", GATEWAY_ADMIN_KEY)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for arg in extra_args {
            command.arg(arg);
        }
        ProcessGuard::spawn(&mut command).expect("spawn supervisor")
    }

    fn spawn_supervisor_captured(&self, extra_args: &[&str]) -> ProcessGuard {
        let mut command = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"));
        command
            .arg("--config")
            .arg(&self.current_config)
            .arg("--cpu-worker-binary")
            .arg(&self.worker_binary)
            .arg("--cpu-ids")
            .arg("0,1")
            .arg("--allocatable-host-memory-bytes")
            .arg("8589934592")
            .env("TOKEN_OLD", "secret-old")
            .env("TOKEN_NEW", "secret-new")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for arg in extra_args {
            command.arg(arg);
        }
        ProcessGuard::spawn(&mut command).expect("spawn supervisor")
    }
}

/// A fake gateway admin endpoint plus the plan section that points the
/// rollout at it.
struct FakeGateway {
    _child: ProcessGuard,
    status: PathBuf,
}

impl FakeGateway {
    fn start(rig: &Rig, timeout_secs: u64) -> Self {
        let port = free_port();
        let script = rig.root.join("fake-gateway.py");
        std::fs::write(&script, FAKE_GATEWAY_SCRIPT).unwrap();
        make_executable(&script);
        let status = rig.root.join("gateway-status.json");
        let gateway = Self {
            _child: ProcessGuard::spawn(
                Command::new(&script)
                    .arg(&status)
                    .arg(GATEWAY_ADMIN_KEY)
                    .arg(port.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null()),
            )
            .unwrap(),
            status,
        };
        gateway.report(&rig.original_approvals, 1);
        let mut plan = read_file(&rig.plan);
        // With acknowledgement the timers are no longer needed.
        plan = plan
            .lines()
            .filter(|line| {
                !line.starts_with("window_secs") && !line.starts_with("commit_grace_secs")
            })
            .map(|line| format!("{line}\n"))
            .collect();
        plan.push_str(&format!(
            "\n[gateway_ack]\nadmin_urls = [\"http://127.0.0.1:{port}\"]\nadmin_key_env = \"ROLLOUT_RIG_GATEWAY_ADMIN_KEY\"\ntimeout_secs = {timeout_secs}\n"
        ));
        std::fs::write(&rig.plan, plan).unwrap();
        gateway
    }

    /// Reports the approvals of `view` adopted, with the chat pool routing
    /// to `eligible`.
    fn report(&self, view: &str, eligible: u64) {
        let status = izwi_serving_protocol::GatewayApprovalsStatus {
            file_digest: None,
            adoptions: 1,
            approvals: entries(view),
            pools: vec![izwi_serving_protocol::GatewayPoolStatus {
                task: TaskKind::Chat,
                public_model: ModelAlias::new("chat-model").unwrap(),
                deployment_id: DeploymentId::new("chat-prod").unwrap(),
                eligible_generation: Some(ModelGeneration::new(eligible).unwrap()),
                generations: vec![ModelGeneration::new(eligible).unwrap()],
            }],
        };
        let temporary = self.status.with_extension("tmp");
        std::fs::write(&temporary, serde_json::to_vec(&status).unwrap()).unwrap();
        std::fs::rename(&temporary, &self.status).unwrap();
    }
}

fn node_toml(root: &Path, workers: &[(&FakeWorker, u16)]) -> String {
    let mut text = format!(
        "schema_version = 2\nnode_id = \"node-a\"\nworking_directory = \"{}\"\nruntime_directory = \"{}\"\nhost_memory_budget_bytes = 8589934592\n\n[readiness]\nstartup_timeout_ms = 4000\npoll_interval_ms = 100\n\n[shutdown]\ndrain_grace_ms = 4000\ncancellation_grace_ms = 1000\ntermination_grace_ms = 1000\n",
        root.join("work").display(),
        root.join("run").display(),
    );
    for (worker, port) in workers {
        text.push_str(&format!(
            "\n[[workers]]\nworker_id = \"{id}\"\nbind = \"127.0.0.1:{port}\"\nbinary = \"cpu\"\ncredential_id = \"cred-a\"\nbearer_token_env = \"{token}\"\nmax_active_invocations = 4\nmax_request_bytes = 1048576\n\n[workers.assignment]\nbackend = \"cpu\"\nthread_budget = 1\naffinity = [0]\nhost_memory_limit_bytes = 1024\n\n[workers.deployment]\ndeployment_id = \"{deployment}\"\npublic_model = \"{model}\"\nartifact_revision = \"rev-1\"\nmodel_generation = {generation}\ntask = \"chat\"\nbackend = \"cpu\"\nprecision = \"f32\"\nexecution_representation = \"gguf\"\nmodels_directory = \"{models}\"\n\n[workers.deployment.capability]\nstreaming = true\nrealtime = false\ncancellation = \"cooperative\"\naccepted_input_formats = [\"chat_messages\"]\noutput_formats = [\"text\"]\nmax_input_bytes = 1048576\nmax_context_tokens = 32\nmax_output_tokens = 32\n",
            id = worker.worker_id,
            port = port,
            token = worker.token_env,
            deployment = worker.deployment_id,
            model = worker.public_model,
            generation = worker.generation,
            models = root.join("models").display(),
        ));
    }
    text
}

fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
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
        std::thread::sleep(Duration::from_millis(100));
    }
    condition()
}

fn read_file(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn canary_readiness_failure_aborts_and_restores_approvals() {
    let rig = Rig::build(2, 1);
    // The canary (the only replacement) never serves: readiness fails and
    // the supervisor must abort without touching the old generation.
    let mut manifest = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(rig.root.join("manifest.json")).unwrap(),
    )
    .unwrap();
    manifest[REPLACEMENT.worker_id]["serve"] = serde_json::json!(false);
    std::fs::write(
        rig.root.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    // The fake worker script reads the manifest at startup, so it must be
    // regenerated after the edit.
    let script = FAKE_WORKER_SCRIPT.replace(
        "{manifest}",
        &rig.root.join("manifest.json").display().to_string(),
    );
    std::fs::write(&rig.worker_binary, script).unwrap();
    make_executable(&rig.worker_binary);

    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let aborted = wait_until(Duration::from_secs(30), || {
        !rig.state_file().exists() && read_file(&rig.approvals) == rig.original_approvals
    });
    assert!(
        aborted,
        "the rollout must abort and restore the approvals byte-identically"
    );
    assert!(
        supervisor.try_wait().unwrap().is_none(),
        "the supervisor keeps supervising the old generation after the abort"
    );
    let _ = supervisor.kill();
}

#[test]
fn promote_path_commits_and_drains_the_old_generation() {
    let commit_grace = Duration::from_secs(3);
    let rig = Rig::build_with_commit_grace(2, 1, commit_grace.as_secs());
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);

    // Window view: both generations approved, and the persisted state
    // records the open window (the coordinator persists after the view).
    let window_open = wait_until(Duration::from_secs(30), || {
        let approvals = read_file(&rig.approvals);
        approvals.contains("chat-prod|1")
            && approvals.contains("chat-prod|2")
            && read_file(&rig.state_file()).contains("window_open")
    });
    assert!(window_open, "the window view must approve both generations");

    // Commit view: only the successor generation remains.
    let commit_view = wait_until(Duration::from_secs(60), || {
        let approvals = read_file(&rig.approvals);
        approvals.contains("chat-prod|2")
            && !approvals.contains("chat-prod|1")
            && approvals.contains("asr-prod|4")
    });
    assert!(
        commit_view,
        "the commit view must replace the old generation"
    );
    let commit_view_seen = Instant::now();
    // The gateways keep routing to the old generation until their next
    // approvals refresh, so it must keep serving through the commit grace.
    assert!(
        !rig.root.join("worker-old.exited").exists(),
        "the old generation must not drain the moment the commit view lands"
    );

    // The old generation drained only after it stopped serving; its fake
    // writes the exit marker when stdin (the ownership pipe) closes.
    let old_exited = wait_until(Duration::from_secs(30), || {
        rig.root.join("worker-old.exited").exists()
    });
    assert!(old_exited, "the old worker must drain to exit");
    assert!(
        commit_view_seen.elapsed() >= commit_grace / 2,
        "the old generation drained {:?} after the commit view was seen, inside the {commit_grace:?} grace",
        commit_view_seen.elapsed()
    );

    // The state records the commit and the replacement worker record.
    let committed = wait_until(Duration::from_secs(30), || {
        let state = read_file(&rig.state_file());
        state.contains("committed") && state.contains("worker-new")
    });
    assert!(committed, "the rollout state must record the commit");

    let _ = supervisor.kill();
}

#[test]
fn replacement_exit_during_window_aborts_and_restores() {
    let rig = Rig::build(6, 1);
    // The replacement serves readiness, then dies inside the window.
    let mut manifest = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(rig.root.join("manifest.json")).unwrap(),
    )
    .unwrap();
    manifest[REPLACEMENT.worker_id]["die_after_secs"] = serde_json::json!(2);
    std::fs::write(
        rig.root.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let script = FAKE_WORKER_SCRIPT.replace(
        "{manifest}",
        &rig.root.join("manifest.json").display().to_string(),
    );
    std::fs::write(&rig.worker_binary, script).unwrap();
    make_executable(&rig.worker_binary);

    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let restored = wait_until(Duration::from_secs(60), || {
        !rig.state_file().exists() && read_file(&rig.approvals) == rig.original_approvals
    });
    assert!(
        restored,
        "a replacement exit during the window must abort and restore the approvals"
    );
    assert!(
        !rig.root.join("worker-old.exited").exists(),
        "the old generation must keep serving through the abort"
    );
    assert!(
        supervisor.try_wait().unwrap().is_none(),
        "the supervisor keeps supervising the old generation after the abort"
    );
    let _ = supervisor.kill();
}

/// The approvals as a sorted list of entries, ignoring order and comments.
fn entries(text: &str) -> Vec<String> {
    let mut entries: Vec<String> = text
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim().to_string())
        .filter(|entry| !entry.is_empty())
        .collect();
    entries.sort();
    entries
}

fn serving(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// With gateway_ack the rollout commits only once the gateway reports the
/// window view adopted and cut over, and drains the old generation only
/// once it reports the commit view adopted, however long that takes.
#[test]
fn commit_waits_for_the_gateways_to_acknowledge_each_view() {
    let rig = Rig::build(2, 1);
    let gateway = FakeGateway::start(&rig, 60);
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);

    let window_open = wait_until(Duration::from_secs(30), || {
        read_file(&rig.approvals).contains("chat-prod|2")
    });
    assert!(window_open, "the window view must be written");
    let window = read_file(&rig.approvals);
    std::thread::sleep(Duration::from_secs(3));
    assert!(
        read_file(&rig.approvals).contains("chat-prod|1"),
        "no commit view before the gateway acknowledges the window"
    );

    gateway.report(&window, 2);
    let commit_view = wait_until(Duration::from_secs(30), || {
        !read_file(&rig.approvals).contains("chat-prod|1")
    });
    assert!(commit_view, "the acknowledged window must commit");
    let commit = read_file(&rig.approvals);
    std::thread::sleep(Duration::from_secs(3));
    assert!(
        !rig.root.join("worker-old.exited").exists(),
        "the old generation must not drain before the gateway acknowledges the commit view"
    );

    gateway.report(&commit, 2);
    let drained = wait_until(Duration::from_secs(30), || {
        rig.root.join("worker-old.exited").exists()
            && read_file(&rig.state_file()).contains("committed")
    });
    assert!(
        drained,
        "the acknowledged commit view must drain the old generation"
    );
    let _ = supervisor.kill();
}

/// A gateway that never acknowledges the window view aborts the rollout.
#[test]
fn an_unacknowledged_window_aborts_the_rollout() {
    let rig = Rig::build(2, 1);
    let _gateway = FakeGateway::start(&rig, 2);
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let window_open = wait_until(Duration::from_secs(30), || {
        read_file(&rig.approvals).contains("chat-prod|2")
    });
    assert!(window_open, "the window view must be written");
    let aborted = wait_until(Duration::from_secs(30), || {
        !rig.state_file().exists() && read_file(&rig.approvals) == rig.original_approvals
    });
    assert!(
        aborted,
        "an unacknowledged window must abort and restore the view"
    );
    assert!(
        !rig.root.join("worker-old.exited").exists(),
        "the old generation never stopped serving"
    );
    let _ = supervisor.kill();
}

#[test]
fn a_node_without_room_for_both_generations_recreates() {
    let rig = Rig::build_recreate();
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let committed = wait_until(Duration::from_secs(60), || {
        read_file(&rig.state_file()).contains("committed")
    });
    let state = read_file(&rig.state_file());
    assert!(committed, "the recreate rollout must commit: {state}");
    assert!(state.contains("\"recreate\""), "{state}");
    assert!(
        rig.root.join("worker-old.exited").exists(),
        "the old generation drains before the replacement takes its bind"
    );
    let approvals = read_file(&rig.approvals);
    assert!(approvals.contains("worker-new|chat|chat-model|chat-prod|2"));
    assert!(!approvals.contains("chat-prod|1"), "{approvals}");
    assert!(approvals.contains("asr-prod|4"), "{approvals}");
    assert!(
        serving(rig.old_port),
        "the replacement serves on the shared bind"
    );
    let _ = supervisor.kill();
}

#[test]
fn a_failed_recreate_rolls_back_to_the_old_generation() {
    let rig = Rig::build_recreate();
    let mut manifest = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(rig.root.join("manifest.json")).unwrap(),
    )
    .unwrap();
    manifest[REPLACEMENT.worker_id]["serve"] = serde_json::json!(false);
    std::fs::write(
        rig.root.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let script = FAKE_WORKER_SCRIPT.replace(
        "{manifest}",
        &rig.root.join("manifest.json").display().to_string(),
    );
    std::fs::write(&rig.worker_binary, script).unwrap();
    make_executable(&rig.worker_binary);

    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let drained = wait_until(Duration::from_secs(30), || {
        rig.root.join("worker-old.exited").exists()
    });
    assert!(drained, "the recreate drains the old generation first");
    let rolled_back = wait_until(Duration::from_secs(60), || {
        !rig.state_file().exists()
            && entries(&read_file(&rig.approvals)) == entries(&rig.original_approvals)
            && serving(rig.old_port)
    });
    assert!(
        rolled_back,
        "a replacement that never gets ready restores the approvals and relaunches the old generation: {}",
        read_file(&rig.approvals)
    );
    assert!(
        supervisor.try_wait().unwrap().is_none(),
        "the supervisor keeps supervising the old generation"
    );
    let _ = supervisor.kill();
}

#[test]
fn abort_keeps_a_peer_edit_made_during_the_window() {
    let rig = Rig::build(8, 1);
    // The replacement serves readiness, then dies inside the window.
    let mut manifest = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(rig.root.join("manifest.json")).unwrap(),
    )
    .unwrap();
    manifest[REPLACEMENT.worker_id]["die_after_secs"] = serde_json::json!(4);
    std::fs::write(
        rig.root.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let script = FAKE_WORKER_SCRIPT.replace(
        "{manifest}",
        &rig.root.join("manifest.json").display().to_string(),
    );
    std::fs::write(&rig.worker_binary, script).unwrap();
    make_executable(&rig.worker_binary);

    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let window_open = wait_until(Duration::from_secs(30), || {
        read_file(&rig.approvals).contains("chat-prod|2")
    });
    assert!(window_open, "the window view must land first");

    // Another writer (an operator, or a second supervisor's autoscaler)
    // approves its own worker while the window is open.
    let peer = "v1|http://127.0.0.1:9700|node-b|worker-p|chat|chat-model|chat-prod|1\n";
    let window = read_file(&rig.approvals);
    std::fs::write(&rig.approvals, format!("{window}{peer}")).unwrap();

    let expected = format!("{}{peer}", rig.original_approvals);
    let restored = wait_until(Duration::from_secs(60), || {
        !rig.state_file().exists() && read_file(&rig.approvals) == expected
    });
    assert!(
        restored,
        "the abort removes only the rollout's lines and keeps the peer's: {}",
        read_file(&rig.approvals)
    );
    let _ = supervisor.kill();
}

#[test]
fn resume_after_crash_commits_and_fresh_start_fails_closed() {
    let rig = Rig::build(4, 1);
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    // The coordinator writes the window view first and persists
    // `window_open` after it; a crash between the two resumes from
    // `launching_replacement`. Crash only once both have landed, so the test
    // exercises the window-open resume it names.
    let window_open = wait_until(Duration::from_secs(30), || {
        read_file(&rig.approvals).contains("chat-prod|2")
            && read_file(&rig.state_file()).contains("window_open")
    });
    assert!(window_open, "the window must open before the crash");

    // Crash the supervisor mid-window: state stays, workers self-drain. Kill
    // the supervisor alone, not its process group, so the workers outlive it.
    let _ = Child::kill(&mut supervisor);
    let _ = supervisor.wait();
    let state = read_file(&rig.state_file());
    assert!(
        state.contains("window_open"),
        "the crashed rollout must leave resumable state: {state}"
    );
    let workers_exited = wait_until(Duration::from_secs(30), || {
        rig.root.join("worker-old.exited").exists() && rig.root.join("worker-new.exited").exists()
    });
    assert!(
        workers_exited,
        "managed workers self-drain on supervisor death"
    );

    // A fresh start without the plan must fail closed on the non-terminal
    // state.
    let refuse = rig.spawn_supervisor_captured(&[]);
    let output = refuse.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "a fresh start must not silently relaunch the old generation mid-rollout"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("rollout"),
        "the refusal must name the rollout: {stderr}"
    );

    // Resuming with the same plan completes the rollout.
    let mut resumed = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let committed = wait_until(Duration::from_secs(90), || {
        read_file(&rig.state_file()).contains("committed")
            && !read_file(&rig.approvals).contains("chat-prod|1")
    });
    assert!(committed, "the resumed rollout must commit");
    let _ = resumed.kill();
}

#[test]
fn rollout_status_and_abort_commands_manage_persisted_state() {
    let rig = Rig::build(2, 1);
    // Stage a hand-written non-terminal state, exactly as the coordinator
    // would after writing its window view.
    let added = "v1|http://127.0.0.1:9999|node-a|worker-x|chat|chat-model|chat-prod|2";
    std::fs::write(
        rig.state_file(),
        format!(
            r#"{{"schema_version":2,"plan_digest":"sha256:placeholder","phase":"window_open","strategy":"blue_green","started_at_ms":1,"updated_at_ms":2,"window_deadline_ms":null,"replacement_workers":[],"shared_approvals_path":"{}","target_node_config":"{}","target_config_digest":"sha256:placeholder","approval_lines":["{added}"]}}"#,
            rig.approvals.display(),
            rig.target_config.display(),
        ),
    )
    .unwrap();
    // The rollout appended its line; abort must remove it.
    std::fs::write(
        &rig.approvals,
        format!("{}{added}\n", rig.original_approvals),
    )
    .unwrap();

    let status = rig.spawn_supervisor_captured(&["--rollout-status"]);
    let output = status.wait_with_output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("rollout_state=window_open"), "{stderr}");

    let abort = rig.spawn_supervisor_captured(&["--rollout-abort"]);
    let output = abort.wait_with_output().unwrap();
    assert!(output.status.success(), "abort must succeed");
    assert_eq!(
        read_file(&rig.approvals),
        rig.original_approvals,
        "abort restores the pre-rollout approvals byte-identically"
    );
    assert!(!rig.state_file().exists(), "abort clears the state");

    let repeat = rig.spawn_supervisor_captured(&["--rollout-abort"]);
    let output = repeat.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "aborting with no rollout state must fail loudly"
    );
}

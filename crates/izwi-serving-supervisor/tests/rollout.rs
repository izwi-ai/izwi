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

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use izwi_serving_protocol::{
    BackendKind, CancellationBehavior, Capability, CapacitySnapshot, DeploymentId,
    DeviceAssignment, IncarnationId, InputFormat, LoadedDeployment, ModelAlias, ModelGeneration,
    ModelReadiness, NodeId, OutputFormat, SchemaVersion, TaskKind, WorkerDescriptor, WorkerFeature,
    WorkerId, WorkerProcessState, WorkerStatus, PROTOCOL_V1,
};

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
}

impl Rig {
    fn build(window_secs: u64, abort_grace_secs: u64) -> Self {
        let root = tempfile::tempdir().unwrap().keep();
        let run_dir = root.join("run");
        let work_dir = root.join("work");
        let models_dir = root.join("models");
        for dir in [&run_dir, &work_dir, &models_dir] {
            std::fs::create_dir_all(dir).unwrap();
        }

        let old_port = free_port();
        let new_port = free_port();
        let workers = [
            (OLD_WORKER, old_port, "TOKEN_OLD"),
            (REPLACEMENT, new_port, "TOKEN_NEW"),
        ];

        // Current config: only the old worker. Target config: only the
        // replacement, at the new generation.
        let mut manifest = serde_json::Map::new();
        for (worker, port, token) in workers {
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
                "schema_version = 1\ntarget_node_config = \"{}\"\nshared_approvals_path = \"{}\"\ncanary_worker_id = \"{}\"\nwindow_secs = {window_secs}\nabort_grace_secs = {abort_grace_secs}\n",
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
        }
    }

    fn runtime_directory(&self) -> PathBuf {
        self.root.join("run")
    }

    fn state_file(&self) -> PathBuf {
        self.runtime_directory().join("rollout-state.json")
    }

    fn spawn_supervisor(&self, extra_args: &[&str]) -> Child {
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
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for arg in extra_args {
            command.arg(arg);
        }
        command.spawn().expect("spawn supervisor")
    }

    fn spawn_supervisor_captured(&self, extra_args: &[&str]) -> Child {
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
        command.spawn().expect("spawn supervisor")
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

fn terminate(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
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
    terminate(&mut supervisor);
}

#[test]
fn promote_path_commits_and_drains_the_old_generation() {
    let rig = Rig::build(2, 1);
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

    // Commit view: only the successor generation remains, with the state
    // recording the commit and the replacement worker record.
    let committed = wait_until(Duration::from_secs(60), || {
        let approvals = read_file(&rig.approvals);
        let state = read_file(&rig.state_file());
        approvals.contains("chat-prod|2")
            && !approvals.contains("chat-prod|1")
            && approvals.contains("asr-prod|4")
            && state.contains("committed")
            && state.contains("worker-new")
    });
    assert!(committed, "the commit view must replace the old generation");

    // The old generation drained only after it stopped serving; its fake
    // writes the exit marker when stdin (the ownership pipe) closes.
    let old_exited = wait_until(Duration::from_secs(30), || {
        rig.root.join("worker-old.exited").exists()
    });
    assert!(old_exited, "the old worker must drain to exit");

    terminate(&mut supervisor);
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
    terminate(&mut supervisor);
}

#[test]
fn resume_after_crash_commits_and_fresh_start_fails_closed() {
    let rig = Rig::build(4, 1);
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", rig.plan.to_str().unwrap()]);
    let window_open = wait_until(Duration::from_secs(30), || {
        read_file(&rig.approvals).contains("chat-prod|2")
    });
    assert!(window_open, "the window must open before the crash");

    // Crash the supervisor mid-window: state stays, workers self-drain.
    let _ = supervisor.kill();
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
    let mut refuse = rig.spawn_supervisor_captured(&[]);
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
    terminate(&mut resumed);
}

#[test]
fn rollout_status_and_abort_commands_manage_persisted_state() {
    let rig = Rig::build(2, 1);
    // Stage a hand-written non-terminal state with a matching approvals
    // backup, exactly as the coordinator would.
    let backup = rig.original_approvals.clone();
    std::fs::write(
        rig.runtime_directory().join("rollout-approvals-backup"),
        &backup,
    )
    .unwrap();
    let digest = {
        use sha2::{Digest, Sha256};
        format!("sha256:{:x}", Sha256::digest(backup.as_bytes()))
    };
    std::fs::write(
        rig.state_file(),
        format!(
            r#"{{"schema_version":1,"plan_digest":"sha256:placeholder","phase":"window_open","started_at_ms":1,"updated_at_ms":2,"window_deadline_ms":null,"replacement_workers":[],"shared_approvals_path":"{}","target_node_config":"{}","target_config_digest":"sha256:placeholder","approvals_backup_digest":"{digest}"}}"#,
            rig.approvals.display(),
            rig.target_config.display(),
        ),
    )
    .unwrap();
    // The rollout rewrote the approvals; abort must restore them.
    std::fs::write(
        &rig.approvals,
        "v1|http://127.0.0.1:9999|node-a|worker-x|chat|chat-model|chat-prod|2\n",
    )
    .unwrap();

    let mut status = rig.spawn_supervisor_captured(&["--rollout-status"]);
    let output = status.wait_with_output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("rollout_state=window_open"), "{stderr}");

    let mut abort = rig.spawn_supervisor_captured(&["--rollout-abort"]);
    let output = abort.wait_with_output().unwrap();
    assert!(output.status.success(), "abort must succeed");
    assert_eq!(
        read_file(&rig.approvals),
        rig.original_approvals,
        "abort restores the pre-rollout approvals byte-identically"
    );
    assert!(!rig.state_file().exists(), "abort clears the state");
    assert!(
        !rig.runtime_directory()
            .join("rollout-approvals-backup")
            .exists(),
        "abort clears the backup"
    );

    let mut repeat = rig.spawn_supervisor_captured(&["--rollout-abort"]);
    let output = repeat.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "aborting with no rollout state must fail loudly"
    );
}

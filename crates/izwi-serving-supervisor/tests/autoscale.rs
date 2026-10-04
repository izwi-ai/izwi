//! DS7 T38: signal-driven worker autoscaling against real supervisor
//! processes (T38 — bounds, budgets, stabilization respected).
//!
//! The rig launches the real supervisor binary with two fake CPU workers on
//! one deployment: a core worker (the startup min set) and a standby. The
//! fake workers serve descriptor/status over real loopback HTTP and re-read
//! their status document on every request, so the test drives the capacity
//! signals (sustained queue depth, idle windows) by rewriting files. The
//! shared approvals view is the observable routing contract: scale-up must
//! append the standby's line only after its readiness, and scale-down must
//! remove it, drain the worker, and never touch the core set.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use izwi_serving_protocol::{
    BackendKind, CancellationBehavior, Capability, CapacitySnapshot, DeploymentId,
    DeviceAssignment, IncarnationId, InputFormat, LoadedDeployment, ModelAlias, ModelGeneration,
    ModelReadiness, NodeId, OutputFormat, TaskKind, WorkerDescriptor, WorkerFeature, WorkerId,
    WorkerProcessState, WorkerStatus, PROTOCOL_V1,
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
DESCRIPTOR["worker_id"] = WORKER_ID
DESCRIPTOR["node_id"] = NODE_ID
DESCRIPTOR["incarnation_id"] = INCARNATION
SEQUENCE = 0
LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global SEQUENCE
        if self.path == "/internal/v1/worker":
            body = json.dumps(DESCRIPTOR).encode()
        elif self.path == "/internal/v1/status":
            # Re-read on every request so the test can drive capacity signals.
            STATUS = json.load(open(ENTRY["status"]))
            STATUS["worker_id"] = WORKER_ID
            STATUS["node_id"] = NODE_ID
            STATUS["incarnation_id"] = INCARNATION
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

server = ThreadingHTTPServer(
    ("127.0.0.1", int(os.environ["IZWI_WORKER_BIND"].rsplit(":", 1)[1])), Handler
)
threading.Thread(target=server.serve_forever, daemon=True).start()
exit_when_stdin_closes(server)
"#;

struct FakeWorker {
    worker_id: &'static str,
    token_env: &'static str,
}

const CORE: FakeWorker = FakeWorker {
    worker_id: "worker-a",
    token_env: "TOKEN_CORE",
};
const STANDBY: FakeWorker = FakeWorker {
    worker_id: "worker-b",
    token_env: "TOKEN_STANDBY",
};

struct Rig {
    /// Keeps the rig directory alive; removed when the rig drops.
    _tempdir: tempfile::TempDir,
    config: std::path::PathBuf,
    approvals: std::path::PathBuf,
    worker_binary: std::path::PathBuf,
    status_paths: BTreeMap<&'static str, std::path::PathBuf>,
    exit_markers: BTreeMap<&'static str, std::path::PathBuf>,
    stderr: std::path::PathBuf,
}

impl Rig {
    /// Builds the two-worker rig; `autoscaling` appends the DS7 block wired
    /// to the rig's shared approvals path.
    fn build(autoscaling: bool) -> Self {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path().to_path_buf();
        let run_dir = root.join("run");
        let work_dir = root.join("work");
        let models_dir = root.join("models");
        for dir in [&run_dir, &work_dir, &models_dir] {
            std::fs::create_dir_all(dir).unwrap();
        }

        let core_port = free_port();
        let standby_port = free_port();
        let workers = [
            (CORE, core_port, "worker-a"),
            (STANDBY, standby_port, "worker-b"),
        ];

        let approvals = root.join("shared-approvals");
        let foreign_line = format!(
            "v1|http://127.0.0.1:{foreign_port}|node-z|worker-z|chat|chat-model|chat-prod|1\n",
            foreign_port = free_port(),
        );

        let mut manifest = serde_json::Map::new();
        let mut status_paths = BTreeMap::new();
        let mut exit_markers = BTreeMap::new();
        for (worker, _port, file_stem) in workers {
            let descriptor = WorkerDescriptor {
                schema_version: PROTOCOL_V1,
                supported_protocol_versions: vec![PROTOCOL_V1],
                worker_id: WorkerId::new(worker.worker_id).unwrap(),
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
                worker_id: WorkerId::new(worker.worker_id).unwrap(),
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
                    max_active_invocations: 4,
                    active_invocations: 0,
                    max_queued_invocations: 8,
                    queued_invocations: 0,
                    max_sessions: 0,
                    reserved_sessions: 0,
                    available_admission_credits: 4,
                    outstanding_cost_units: 0,
                },
            };
            let descriptor_path = root.join(format!("{file_stem}.descriptor.json"));
            let status_path = root.join(format!("{file_stem}.status.json"));
            std::fs::write(&descriptor_path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
            std::fs::write(&status_path, serde_json::to_vec(&status).unwrap()).unwrap();
            let exit_marker = root.join(format!("{file_stem}.exited"));
            status_paths.insert(worker.worker_id, status_path.clone());
            exit_markers.insert(worker.worker_id, exit_marker.clone());
            manifest.insert(
                worker.worker_id.to_string(),
                serde_json::json!({
                    "descriptor": descriptor_path.display().to_string(),
                    "status": status_path.display().to_string(),
                    "exit_marker": exit_marker.display().to_string(),
                }),
            );
        }

        let manifest_path = root.join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let script = FAKE_WORKER_SCRIPT.replace("{manifest}", &manifest_path.display().to_string());
        let worker_binary = root.join("fake-worker.py");
        std::fs::write(&worker_binary, script).unwrap();
        make_executable(&worker_binary);

        let autoscaling_block = autoscaling.then(|| autoscaling_block(&approvals));
        let config = root.join("node.toml");
        std::fs::write(
            &config,
            node_toml(
                &root,
                &[(&CORE, core_port), (&STANDBY, standby_port)],
                autoscaling_block.as_deref(),
            ),
        )
        .unwrap();

        let initial_view = match autoscaling {
            // The supervisor reconciles the view to the min set at startup;
            // start from a foreign-only view and expect the core line added.
            true => format!("# fleet approvals\n{foreign_line}"),
            // The static supervisor never touches the view.
            false => format!(
                "# fleet approvals\n{foreign_line}v1|http://127.0.0.1:{core_port}|node-a|worker-a|chat|chat-model|chat-prod|1\nv1|http://127.0.0.1:{standby_port}|node-a|worker-b|chat|chat-model|chat-prod|1\n",
            ),
        };
        std::fs::write(&approvals, &initial_view).unwrap();

        let stderr = root.join("supervisor.stderr");
        Self {
            _tempdir: tempdir,
            config,
            approvals,
            worker_binary,
            status_paths,
            exit_markers,
            stderr,
        }
    }

    fn spawn_supervisor(&self, extra_args: &[&str]) -> Child {
        let stderr = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.stderr)
            .unwrap();
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
            .env("TOKEN_CORE", "secret-core")
            .env("TOKEN_STANDBY", "secret-standby")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::process::Stdio::from(stderr));
        for arg in extra_args {
            command.arg(arg);
        }
        command.spawn().expect("spawn supervisor")
    }

    /// Rewrites one worker's status document with new queue/active signals,
    /// keeping the readiness invariants (active + credits == max_active).
    fn set_capacity(&self, worker: &'static str, queued: u32, active: u32) {
        let path = &self.status_paths[&worker];
        let mut status: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        status["capacity"]["queued_invocations"] = queued.into();
        status["capacity"]["active_invocations"] = active.into();
        status["capacity"]["available_admission_credits"] = (4 - active).into();
        std::fs::write(path, serde_json::to_vec(&status).unwrap()).unwrap();
    }

    fn approvals_text(&self) -> String {
        std::fs::read_to_string(&self.approvals).unwrap()
    }

    fn approvals_contain(&self, needle: &str) -> bool {
        self.approvals_text().contains(needle)
    }

    fn stderr_text(&self) -> String {
        std::fs::read_to_string(&self.stderr).unwrap_or_default()
    }

    fn exited(&self, worker: &'static str) -> bool {
        self.exit_markers[&worker].exists()
    }
}

fn autoscaling_block(approvals: &Path) -> String {
    format!(
        "\n[autoscaling]\nshared_approvals_path = \"{}\"\nevaluation_interval_ms = 100\n\n[autoscaling.deployments.chat-prod]\nmin_workers = 1\nmax_workers = 2\nscale_up_queue_depth = 2\nscale_up_sustained_polls = 2\nscale_down_stabilization_window_ms = 1000\n",
        approvals.display()
    )
}

fn node_toml(
    root: &Path,
    workers: &[(&FakeWorker, u16)],
    autoscaling_block: Option<&str>,
) -> String {
    let mut text = format!(
        "schema_version = 2\nnode_id = \"node-a\"\nworking_directory = \"{}\"\nruntime_directory = \"{}\"\nhost_memory_budget_bytes = 8589934592\n\n[readiness]\nstartup_timeout_ms = 4000\npoll_interval_ms = 100\n\n[shutdown]\ndrain_grace_ms = 4000\ncancellation_grace_ms = 1000\ntermination_grace_ms = 1000\n",
        root.join("work").display(),
        root.join("run").display(),
    );
    for (worker, port) in workers {
        text.push_str(&format!(
            "\n[[workers]]\nworker_id = \"{id}\"\nbind = \"127.0.0.1:{port}\"\nbinary = \"cpu\"\ncredential_id = \"cred-a\"\nbearer_token_env = \"{token}\"\nmax_active_invocations = 4\nmax_request_bytes = 1048576\n\n[workers.assignment]\nbackend = \"cpu\"\nthread_budget = 1\naffinity = [0]\nhost_memory_limit_bytes = 1024\n\n[workers.deployment]\ndeployment_id = \"chat-prod\"\npublic_model = \"chat-model\"\nartifact_revision = \"rev-1\"\nmodel_generation = 1\ntask = \"chat\"\nbackend = \"cpu\"\nprecision = \"f32\"\nexecution_representation = \"gguf\"\nmodels_directory = \"{models}\"\n\n[workers.deployment.capability]\nstreaming = true\nrealtime = false\ncancellation = \"cooperative\"\naccepted_input_formats = [\"chat_messages\"]\noutput_formats = [\"text\"]\nmax_input_bytes = 1048576\nmax_context_tokens = 32\nmax_output_tokens = 32\n",
            id = worker.worker_id,
            port = port,
            token = worker.token_env,
            models = root.join("models").display(),
        ));
    }
    if let Some(block) = autoscaling_block {
        text.push_str(block);
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
        std::thread::sleep(Duration::from_millis(25));
    }
    condition()
}

/// Kills the supervisor on drop so a failed assertion never leaves orphaned
/// fake workers or a live supervisor behind on this shared host.
struct SupervisorGuard(Child);

impl Drop for SupervisorGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn sustained_queue_depth_scales_up_and_idle_window_scales_back_down() {
    let rig = Rig::build(true);
    let supervisor = SupervisorGuard(rig.spawn_supervisor(&[]));

    // Startup reconciliation approves exactly the min set: the core line is
    // added, the foreign line passes through, the standby is not approved.
    assert!(
        wait_until(Duration::from_secs(5), || rig
            .approvals_contain("|node-a|worker-a|")),
        "startup must approve the core worker; view:\n{}",
        rig.approvals_text()
    );
    assert!(
        rig.approvals_contain("|node-z|worker-z|"),
        "unrelated lines are preserved"
    );
    assert!(
        !rig.approvals_contain("|node-a|worker-b|"),
        "the standby must not be approved before a scale-up"
    );
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("worker-a ready as incarnation")),
        "the core worker must launch at startup; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(!rig.exited("worker-a"));

    // Sustained queue depth above the threshold scales 1→2: the standby
    // launches, reaches readiness, and only then gains its approvals line.
    rig.set_capacity("worker-a", 4, 0);
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("scaled up; launching standby worker-b")),
        "the supervisor must launch the standby on sustained queue depth; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("worker-b ready as incarnation")),
        "the standby must reach readiness; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("worker-b is ready and approved")),
        "the approvals line must be published after readiness; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        rig.approvals_contain("|node-a|worker-b|"),
        "the standby must be approved after readiness; view:\n{}",
        rig.approvals_text()
    );
    assert!(!rig.exited("worker-b"));

    // Hysteresis: idling immediately after the scale-up must not scale down
    // within the stabilization window (1000 ms here).
    rig.set_capacity("worker-a", 0, 0);
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        !rig.stderr_text().contains("marked draining"),
        "no scale-down may fire inside the stabilization window; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(!rig.exited("worker-b"));

    // After the idle window scales 2→1: the standby's line is removed, its
    // admission stops, and it drains to a clean exit. The core set is never
    // a scale-down candidate.
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("marked draining")),
        "the idle window must scale the deployment down; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        wait_until(Duration::from_secs(8), || rig.exited("worker-b")),
        "the drained standby must exit; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        wait_until(Duration::from_secs(8), || rig
            .stderr_text()
            .contains("slot returned to standby")),
        "the drain must complete cleanly; stderr:\n{}",
        rig.stderr_text()
    );
    assert!(
        !rig.approvals_contain("|node-a|worker-b|"),
        "the drained standby must be unapproved; view:\n{}",
        rig.approvals_text()
    );
    assert!(
        rig.approvals_contain("|node-a|worker-a|"),
        "the core worker must stay approved; view:\n{}",
        rig.approvals_text()
    );
    assert!(!rig.exited("worker-a"), "the core worker must keep running");
    drop(supervisor);
}

#[test]
fn disabled_autoscaling_launches_everything_and_never_touches_the_view() {
    let rig = Rig::build(false);
    let original_view = rig.approvals_text();
    let _supervisor = SupervisorGuard(rig.spawn_supervisor(&[]));

    for worker in ["worker-a", "worker-b"] {
        assert!(
            wait_until(Duration::from_secs(8), || rig
                .stderr_text()
                .contains(&format!("{worker} ready as incarnation"))),
            "the static supervisor must launch {worker}; stderr:\n{}",
            rig.stderr_text()
        );
    }
    assert!(!rig.exited("worker-a") && !rig.exited("worker-b"));
    assert_eq!(
        rig.approvals_text(),
        original_view,
        "without the autoscaling block the supervisor must not touch the shared view"
    );
}

#[test]
fn validate_only_reports_the_autoscaling_policy() {
    let rig = Rig::build(true);
    let mut supervisor = rig.spawn_supervisor(&["--validate-only"]);
    let mut stdout = String::new();
    if let Some(mut pipe) = supervisor.stdout.take() {
        std::io::Read::read_to_string(&mut pipe, &mut stdout).expect("read validate-only output");
    }
    let status = supervisor.wait().expect("validate-only exits");
    drop(supervisor);
    assert!(status.success(), "validate-only must pass: {stdout}");
    assert!(
        stdout.contains("autoscaling enabled deployments=1"),
        "the diagnostic must summarize the autoscaling block: {stdout}"
    );
    assert!(
        stdout.contains(
            "autoscale deployment=chat-prod min_workers=1 max_workers=2 scale_up_queue_depth=2 scale_up_sustained_polls=2 scale_down_stabilization_window_ms=1000"
        ),
        "the diagnostic must summarize the deployment policy: {stdout}"
    );
}

#[test]
fn rollout_plans_are_rejected_when_autoscaling_is_enabled() {
    let rig = Rig::build(true);
    // The conflict check precedes rollout preparation, so a plan file that
    // does not exist proves ordering: the refusal happens before any staging.
    let mut supervisor = rig.spawn_supervisor(&["--rollout-plan", "/nonexistent/rollout.toml"]);
    let status = supervisor.wait().expect("supervisor exits with a conflict");
    drop(supervisor);
    assert!(
        !status.success(),
        "a rollout plan must be refused under autoscaling"
    );
    assert!(
        rig.stderr_text().contains("RolloutAutoscalingConflict"),
        "the diagnostic must name the conflict; stderr:\n{}",
        rig.stderr_text()
    );
}

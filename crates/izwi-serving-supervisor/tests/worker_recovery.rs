//! Process-level supervisor recovery: a worker that crashes on every start
//! must be restarted with bounded backoff and then quarantined, never
//! restarted without bound.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
#[test]
fn repeatedly_crashing_worker_is_quarantined() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let workdir = root.path().join("work");
    let rundir = root.path().join("run");
    let models = root.path().join("models");
    std::fs::create_dir_all(&workdir).unwrap();
    std::fs::create_dir_all(&rundir).unwrap();
    std::fs::create_dir_all(&models).unwrap();

    let fake_worker = root.path().join("fake-worker");
    std::fs::write(&fake_worker, "#!/bin/sh\nexit 1\n").unwrap();
    std::fs::set_permissions(&fake_worker, std::fs::Permissions::from_mode(0o755)).unwrap();

    let config = root.path().join("node.toml");
    std::fs::write(
        &config,
        format!(
            r#"
schema_version = 2
node_id = "node-recovery"
working_directory = "{}"
runtime_directory = "{}"
host_memory_budget_bytes = 8589934592

[readiness]
startup_timeout_ms = 5000
poll_interval_ms = 25

[restart]
initial_backoff_ms = 10
maximum_backoff_ms = 50
restart_window_ms = 60000
stable_reset_ms = 3600000
max_restarts_per_window = 2
jitter_percent = 0

[shutdown]
drain_grace_ms = 100
cancellation_grace_ms = 100
termination_grace_ms = 100

[[workers]]
worker_id = "recovery-test-worker"
bind = "127.0.0.1:9499"
binary = "cpu"
credential_id = "recovery-credential"
bearer_token_env = "IZWI_SUPERVISOR_RECOVERY_TEST_TOKEN"
max_active_invocations = 1
max_request_bytes = 1048576
max_retained_attempts = 64
attempt_retention_secs = 300
streaming = true

[workers.assignment]
backend = "cpu"
thread_budget = 1
affinity = []
host_memory_limit_bytes = 1073741824

[workers.deployment]
deployment_id = "recovery-deploy-v1"
public_model = "LFM2.5-1.2B-Instruct-GGUF"
artifact_revision = "recovery-rev"
model_generation = 1
task = "chat"
backend = "cpu"
precision = "gguf-q4_k_m"
execution_representation = "native-lfm2"
models_directory = "{}"

[workers.deployment.capability]
streaming = true
realtime = false
cancellation = "cooperative"
accepted_input_formats = ["chat_messages"]
output_formats = ["text"]
max_input_bytes = 1048576
max_context_tokens = 32
max_output_tokens = 32
"#,
            workdir.display(),
            rundir.display(),
            models.display(),
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"))
        .arg("--config")
        .arg(&config)
        .arg("--cpu-worker-binary")
        .arg(&fake_worker)
        .arg("--cpu-ids")
        .arg("0")
        .arg("--allocatable-host-memory-bytes")
        .arg("8589934592")
        .env(
            "IZWI_SUPERVISOR_RECOVERY_TEST_TOKEN",
            "recovery-test-secret-1234567890",
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("supervisor should spawn");

    let stderr = child.stderr.take().expect("stderr should be piped");
    let (line_tx, line_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = line_tx.send(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(25);
    let mut quarantined = false;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match line_rx.recv_timeout(remaining.min(Duration::from_secs(1))) {
            Ok(line) if line.contains("quarantined") => {
                quarantined = true;
                break;
            }
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        quarantined,
        "a worker that crashes on every start must be quarantined instead of restarted without bound"
    );
}

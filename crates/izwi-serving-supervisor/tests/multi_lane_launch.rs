//! Process-level proof that the supervisor launches and supervises non-CPU
//! lanes: a Metal or CUDA node configuration is accepted, the configured
//! worker binary is launched with the lane's exact device environment, and a
//! graceful supervisor stop reaps the child. The fake worker cannot perform
//! real accelerator work (and does not try); real accelerator execution
//! evidence remains a hardware-gated lane recorded separately in the support
//! matrix.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
#[test]
fn supervisor_launches_metal_and_cuda_lane_workers_with_exact_assignments() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let rundir = root.path().join("run");
    let models = root.path().join("models");
    std::fs::create_dir_all(&rundir).unwrap();
    std::fs::create_dir_all(&models).unwrap();

    // One fake worker binary per lane. It records the exact environment the
    // supervisor handed it, then sleeps until the supervisor's graceful stop.
    let marker = root.path().join("worker-env.txt");
    let fake_worker = root.path().join("fake-lane-worker");
    let script = format!("#!/bin/sh\nenv > {}\nsleep 30\n", marker.display());
    std::fs::write(&fake_worker, script).unwrap();
    std::fs::set_permissions(&fake_worker, std::fs::Permissions::from_mode(0o755)).unwrap();

    for (backend, assignment, device_args, expected_env) in [
        (
            "metal",
            "backend = \"metal\"\ndevice_id = \"metal:lane-test-0\"\nprocess_local_device_index = 0\nshared_memory_limit_bytes = 2048",
            vec![
                "--metal-worker-binary",
                fake_worker.to_str().unwrap(),
                "--metal-devices",
                "metal:lane-test-0@0",
            ],
            vec![
                "IZWI_BACKEND=metal",
                "IZWI_WORKER_EXPECTED_DEVICE_ID=metal:lane-test-0",
                "IZWI_METAL_DEVICE_ORDINAL=0",
            ],
        ),
        (
            "cuda",
            "backend = \"cuda\"\ndevice_uuid = \"GPU-lane-test\"\nprocess_local_device_index = 0\ndevice_memory_limit_bytes = 1024\nhost_memory_limit_bytes = 1024",
            vec![
                "--cuda-worker-binary",
                fake_worker.to_str().unwrap(),
                "--cuda-devices",
                "GPU-lane-test@0@4096",
            ],
            vec![
                "IZWI_BACKEND=cuda",
                "IZWI_WORKER_EXPECTED_DEVICE_UUID=GPU-lane-test",
                "CUDA_VISIBLE_DEVICES=GPU-lane-test",
            ],
        ),
    ] {
        std::fs::remove_file(&marker).ok();
        let config = root.path().join(format!("node-{backend}.toml"));
        std::fs::write(
            &config,
            format!(
                r#"
schema_version = 2
node_id = "node-a"
working_directory = "{}"
runtime_directory = "{}"
host_memory_budget_bytes = 8192
max_parallel_model_loads = 1

[readiness]
startup_timeout_ms = 60000
poll_interval_ms = 100

[shutdown]
drain_grace_ms = 1000
cancellation_grace_ms = 500
termination_grace_ms = 1000

[[workers]]
worker_id = "lane-worker"
bind = "127.0.0.1:9470"
binary = "{}"
credential_id = "credential-lane"
bearer_token_env = "TEST_LANE_WORKER_TOKEN"
max_request_bytes = 4096

[workers.assignment]
{}

[workers.deployment]
deployment_id = "deployment-lane"
public_model = "model-lane"
artifact_revision = "revision-lane"
model_generation = 1
task = "chat"
backend = "{}"
precision = "gguf-q4_k_m"
execution_representation = "native-lfm2"
models_directory = "{}"

[workers.deployment.capability]
streaming = true
realtime = false
cancellation = "cooperative"
accepted_input_formats = ["chat_messages"]
output_formats = ["text"]
max_input_bytes = 4096
max_context_tokens = 32
max_output_tokens = 32
"#,
                root.path().display(),
                rundir.display(),
                backend,
                assignment,
                backend,
                models.display(),
            ),
        )
        .unwrap();

        let mut child = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"))
            .arg("--config")
            .arg(&config)
            .arg("--cpu-ids")
            .arg("0,1")
            .arg("--allocatable-host-memory-bytes")
            .arg("8192")
            .args(&device_args)
            .env("TEST_LANE_WORKER_TOKEN", "lane-worker-token")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("supervisor binary spawns");
        // Drain supervisor diagnostics so a restart loop cannot fill the pipe.
        let stderr = child.stderr.take().expect("supervisor stderr piped");
        std::thread::spawn(move || {
            for _ in BufReader::new(stderr).lines() {}
        });

        // The fake worker records its launch environment as startup evidence.
        let deadline = Instant::now() + Duration::from_secs(10);
        let env_snapshot = loop {
            if let Ok(snapshot) = std::fs::read_to_string(&marker) {
                break snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "{backend} lane worker never launched; supervisor may have rejected the lane"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
        for expected in &expected_env {
            assert!(
                env_snapshot.lines().any(|line| line == *expected),
                "{backend} lane launch environment missing {expected}; got:\n{env_snapshot}"
            );
        }
        assert!(
            env_snapshot
                .lines()
                .any(|line| line == "IZWI_WORKER_MODEL_LOAD_SLOTS=1"),
            "{backend} lane worker must receive the node's model-load slot bound"
        );

        // Graceful stop: SIGTERM reaps the supervisor and its lane worker.
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match child.try_wait().expect("supervisor waitable") {
                Some(status) => {
                    assert!(status.success(), "{backend} supervisor exit: {status}");
                    break;
                }
                None => {
                    assert!(
                        Instant::now() < deadline,
                        "{backend} supervisor did not stop gracefully"
                    );
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

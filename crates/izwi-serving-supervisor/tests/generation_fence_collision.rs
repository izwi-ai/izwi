//! DS5.4 / T25 process evidence: the supervisor generation fence holds
//! against real processes. A first supervisor's workers keep shared leases
//! on the generation fence for their whole lifetime; while any of them is
//! alive, a second supervisor on the same node configuration cannot get
//! past the fence — it exits with the contention error instead of racing
//! the old generation. Only once every fenced worker has exited can a new
//! supervisor start and launch its own workers. The node lease gives the
//! related live-collision guarantee: a second supervisor cannot even start
//! next to a live first supervisor.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
#[test]
fn supervisor_generation_fence_blocks_a_second_supervisor_until_workers_exit() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("temp dir");
    let rundir = root.path().join("run");
    let models = root.path().join("models");
    std::fs::create_dir_all(&rundir).unwrap();
    std::fs::create_dir_all(&models).unwrap();

    // The fake fenced worker records its launch environment, takes the
    // generation fence's SHARED lease exactly like the real worker does, and
    // then stays alive. The supervisor's own readiness probe will time out
    // long after this test finishes, so no restart loop interferes. The
    // worker is a python script so the lock holder IS the spawned process:
    // a shell wrapper would leave the flock stranded in an orphaned child
    // the test could not address by name.
    let env_marker = root.path().join("worker-env.txt");
    let locked_marker = root.path().join("worker-fenced.txt");
    let fake_worker = root.path().join("fake-fenced-worker");
    let script = format!(
        "#!/usr/bin/env python3\nimport fcntl, os, time\nwith open('{env}', 'a') as handle:\n    for key, value in sorted(os.environ.items()):\n        handle.write(key + '=' + value + '\\n')\nhandle = open(os.environ['IZWI_WORKER_GENERATION_FENCE_LOCK'], 'a+')\nfcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)\nopen('{locked}', 'w').write('locked')\ntime.sleep(300)\n",
        env = env_marker.display(),
        locked = locked_marker.display(),
    );
    std::fs::write(&fake_worker, script).unwrap();
    std::fs::set_permissions(&fake_worker, std::fs::Permissions::from_mode(0o755)).unwrap();

    // Distinct bind port so parallel supervisor test targets never collide.
    let config = root.path().join("node-fence.toml");
    std::fs::write(
        &config,
        format!(
            r#"
schema_version = 2
node_id = "node-fence"
working_directory = "{workdir}"
runtime_directory = "{rundir}"
host_memory_budget_bytes = 8589934592
max_parallel_model_loads = 1

[readiness]
startup_timeout_ms = 300000
poll_interval_ms = 100

[shutdown]
drain_grace_ms = 1000
cancellation_grace_ms = 500
termination_grace_ms = 1000

[[workers]]
worker_id = "fence-worker"
bind = "127.0.0.1:9655"
binary = "cpu"
credential_id = "credential-fence"
bearer_token_env = "TEST_FENCE_WORKER_TOKEN"
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
deployment_id = "deployment-fence"
public_model = "model-fence"
artifact_revision = "revision-fence"
model_generation = 1
task = "chat"
backend = "cpu"
precision = "gguf-q4_k_m"
execution_representation = "native-qwen38"
models_directory = "{models}"

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
            workdir = root.path().display(),
            rundir = rundir.display(),
            models = models.display(),
        ),
    )
    .unwrap();

    let supervisor_command = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"));
        command
            .arg("--config")
            .arg(&config)
            .arg("--cpu-worker-binary")
            .arg(&fake_worker)
            .arg("--cpu-ids")
            .arg("0")
            .arg("--allocatable-host-memory-bytes")
            .arg("8589934592")
            .env("TEST_FENCE_WORKER_TOKEN", "fence-worker-token")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        command
    };

    let mut first = supervisor_command()
        .spawn()
        .expect("first supervisor spawns");
    if let Some(stderr) = first.stderr.take() {
        std::thread::spawn(move || for _ in BufReader::new(stderr).lines() {});
    }

    // The first supervisor's worker holds the shared generation lease.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if locked_marker.exists() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the fenced worker never acquired the generation lease"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // Live collision: a second supervisor fails at the node lease while the
    // first supervisor is alive.
    let second_live = supervisor_command()
        .output()
        .expect("second supervisor runs to completion");
    assert!(
        !second_live.status.success(),
        "a second supervisor must not start next to a live first supervisor"
    );
    let live_stderr = String::from_utf8_lossy(&second_live.stderr);
    assert!(
        live_stderr.contains("Contended") && live_stderr.contains("supervisor-"),
        "the live collision must be the node lease, got: {live_stderr}"
    );

    // Kill the first supervisor without letting it stop its children: the
    // orphaned worker keeps the shared generation lease, which is exactly
    // the state a supervisor crash leaves behind.
    unsafe {
        libc::kill(first.id() as libc::pid_t, libc::SIGKILL);
    }
    let _ = first.wait();

    // A fresh supervisor acquires the node lease (its previous holder is
    // dead) but still fails at the generation barrier while the orphaned
    // worker holds the shared fence.
    let second_after_crash = supervisor_command()
        .output()
        .expect("post-crash supervisor runs to completion");
    assert!(
        !second_after_crash.status.success(),
        "the generation fence must block a replacement supervisor while fenced workers are alive"
    );
    let crash_stderr = String::from_utf8_lossy(&second_after_crash.stderr);
    assert!(
        crash_stderr.contains("Contended") && crash_stderr.contains("generation-"),
        "the post-crash collision must be the generation fence, got: {crash_stderr}"
    );

    // The orphaned workers exit. The fence frees only once every shared
    // lease is gone; probe it the same way a replacement supervisor would,
    // at the exact fence path the launcher handed the worker.
    let _ = Command::new("pkill")
        .arg("-f")
        .arg(fake_worker.to_str().unwrap())
        .output();
    let worker_env = std::fs::read_to_string(&env_marker).expect("worker env snapshot");
    let fence_path = worker_env
        .lines()
        .find_map(|line| line.strip_prefix("IZWI_WORKER_GENERATION_FENCE_LOCK="))
        .expect("the launcher passes the generation fence path to workers");
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let probe = Command::new("python3")
            .arg("-c")
            .arg(format!(
                "import fcntl\nhandle = open('{fence_path}', 'a+')\nfcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\nfcntl.flock(handle, fcntl.LOCK_UN)\n"
            ))
            .output()
            .expect("fence probe runs");
        if probe.status.success() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the generation fence never freed after the fenced workers exited"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // Recovery: with the fence free, a replacement supervisor starts and
    // launches its own workers.
    let mut replacement = supervisor_command()
        .spawn()
        .expect("replacement supervisor spawns");
    if let Some(stderr) = replacement.stderr.take() {
        std::thread::spawn(move || for _ in BufReader::new(stderr).lines() {});
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if env_marker.exists() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the replacement supervisor never launched a worker after the fence freed"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // The replacement stops gracefully and reaps its child. Let startup
    // settle first so the SIGTERM lands after the supervisor's own shutdown
    // listener is registered.
    std::thread::sleep(Duration::from_millis(750));
    unsafe {
        libc::kill(replacement.id() as libc::pid_t, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match replacement
            .try_wait()
            .expect("replacement supervisor waitable")
        {
            Some(status) => {
                assert!(
                    status.success(),
                    "the replacement supervisor must stop gracefully: {status}"
                );
                break;
            }
            None => {
                assert!(
                    Instant::now() < deadline,
                    "the replacement supervisor did not stop gracefully"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
    let _ = replacement.kill();
    let _ = replacement.wait();
    let _ = Command::new("pkill")
        .arg("-f")
        .arg(fake_worker.to_str().unwrap())
        .output();
}

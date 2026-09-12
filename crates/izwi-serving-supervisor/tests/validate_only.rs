#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, process::Command};

const TEST_SECRET: &str = "DO_NOT_PRINT_THIS_WORKER_SECRET_42";
const MAX_VALIDATION_DIAGNOSTIC_BYTES: usize = 16 * 1024;

#[test]
fn validate_only_redacts_credentials_and_never_launches_or_locks() {
    let directory = tempfile::tempdir().unwrap();
    let working_directory = directory.path().join("work");
    let models_directory = directory.path().join("models");
    let runtime_directory = directory.path().join("runtime-must-not-be-created");
    let launch_marker = directory.path().join("worker-launched");
    fs::create_dir(&working_directory).unwrap();
    fs::create_dir(&models_directory).unwrap();

    let worker_binary = directory.path().join("fake-worker");
    fs::write(
        &worker_binary,
        format!(
            "#!/bin/sh\nprintf launched > '{}'\n",
            launch_marker.display()
        ),
    )
    .unwrap();
    let mut permissions = fs::metadata(&worker_binary).unwrap().permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&worker_binary, permissions).unwrap();

    let config_path = directory.path().join("node.toml");
    fs::write(
        &config_path,
        format!(
            r#"schema_version = 1
node_id = "node-validate"
working_directory = "{}"
runtime_directory = "{}"
host_memory_budget_bytes = 1024

[[workers]]
worker_id = "worker-validate"
bind = "127.0.0.1:19470"
binary = "cpu"
credential_id = "credential-validate"
bearer_token_env = "IZWI_VALIDATE_ONLY_TEST_TOKEN"
max_active_invocations = 1
max_request_bytes = 1024
max_retained_attempts = 1
attempt_retention_secs = 60
streaming = true

[workers.assignment]
backend = "cpu"
thread_budget = 1
affinity = [0]
host_memory_limit_bytes = 1024

[workers.deployment]
deployment_id = "deployment-validate"
public_model = "model-validate"
artifact_revision = "revision-validate"
model_generation = 7
backend = "cpu"
models_directory = "{}"
"#,
            working_directory.display(),
            runtime_directory.display(),
            models_directory.display(),
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_izwi-serving-supervisor"))
        .arg("--config")
        .arg(&config_path)
        .arg("--cpu-worker-binary")
        .arg(&worker_binary)
        .arg("--cpu-ids")
        .arg("0")
        .arg("--allocatable-host-memory-bytes")
        .arg("1024")
        .arg("--validate-only")
        .env("IZWI_VALIDATE_ONLY_TEST_TOKEN", TEST_SECRET)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "validate-only failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= MAX_VALIDATION_DIAGNOSTIC_BYTES);
    let diagnostics = String::from_utf8(output.stdout).unwrap();
    assert!(diagnostics.contains("validation=ok mode=validate-only"));
    assert!(diagnostics.contains("worker=worker-validate"));
    assert!(diagnostics.contains("secret=redacted"));
    assert!(!diagnostics.contains(TEST_SECRET));
    assert!(!diagnostics.contains("IZWI_VALIDATE_ONLY_TEST_TOKEN"));
    assert!(!launch_marker.exists(), "validate-only launched the worker");
    assert!(
        !runtime_directory.exists(),
        "validate-only acquired the lock namespace"
    );
}

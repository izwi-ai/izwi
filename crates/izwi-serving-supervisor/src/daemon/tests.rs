use izwi_serving_protocol::{
    ArtifactRevision, CancellationBehavior, CredentialId, DeploymentId, DeviceAssignment, DeviceId,
    InputFormat, ModelAlias, ModelGeneration, NodeId, OutputFormat, TaskKind,
};
#[cfg(unix)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ffi::OsString;

#[cfg(unix)]
use super::autoscale_driver::poll_deployment;
use super::cli::{parse_cpu_ids, parse_cuda_devices, parse_metal_devices};
use super::diagnostics::{
    render_runtime_diagnostic, BoundedDiagnostic, SlotSnapshot, MAX_DIAGNOSTIC_RESPONSE_BYTES,
    TRUNCATED_DIAGNOSTIC_SUFFIX,
};
use super::*;
use crate::actor::{SupervisorMetrics, WorkerView};
use crate::{
    CapabilityProfileConfig, DeploymentConfig, ReadinessPolicy, RestartPolicy, ShutdownPolicy,
    WorkerConfig, DEFAULT_MODEL_LOAD_SLOTS, NODE_CONFIG_SCHEMA_VERSION,
};

fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value).unwrap()
}

fn config(assignment: DeviceAssignment, binary: WorkerBinaryFlavor) -> NodeConfig {
    NodeConfig {
        schema_version: NODE_CONFIG_SCHEMA_VERSION,
        node_id: id::<NodeId>("node-a"),
        working_directory: PathBuf::from("/work"),
        runtime_directory: PathBuf::from("/run"),
        host_memory_budget_bytes: 1024,
        max_parallel_model_loads: DEFAULT_MODEL_LOAD_SLOTS,
        workers: vec![WorkerConfig {
            worker_id: id("worker-a"),
            bind: "127.0.0.1:9470".parse().unwrap(),
            binary,
            credential_id: id::<CredentialId>("credential-a"),
            bearer_token_env: "WORKER_TOKEN".into(),
            assignment,
            deployment: DeploymentConfig {
                deployment_id: id::<DeploymentId>("deployment-a"),
                public_model: id::<ModelAlias>("model-a"),
                artifact_revision: id::<ArtifactRevision>("revision-a"),
                model_generation: ModelGeneration::new(1).unwrap(),
                task: TaskKind::Chat,
                backend: BackendKind::Cpu,
                precision: "gguf-q4_k_m".into(),
                execution_representation: "native-lfm2".into(),
                tokenizer_revision: None,
                capability: CapabilityProfileConfig {
                    streaming: true,
                    realtime: false,
                    cancellation: CancellationBehavior::Cooperative,
                    accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                    output_formats: BTreeSet::from([OutputFormat::Text]),
                    max_input_bytes: 1024,
                    max_context_tokens: Some(32),
                    max_output_tokens: Some(32),
                },
                models_directory: PathBuf::from("/models"),
            },
            max_active_invocations: Some(1),
            max_request_bytes: 1024,
            max_retained_attempts: 1,
            attempt_retention_secs: 60,
            streaming: true,
            host_kv_pool_budget_bytes: 0,
            environment: Default::default(),
        }],
        readiness: ReadinessPolicy::default(),
        liveness: Default::default(),
        restart: RestartPolicy::default(),
        shutdown: ShutdownPolicy::default(),
        autoscaling: None,
    }
}

#[test]
fn cli_requires_explicit_bounded_inventory() {
    let arguments = [
        "--config",
        "/config.toml",
        "--validate-only",
        "--cpu-worker-binary",
        "/worker",
        "--cpu-ids",
        "3,1",
        "--allocatable-host-memory-bytes",
        "4096",
    ]
    .map(OsString::from);
    let ParseOutcome::Run(options) = CliOptions::parse(arguments).unwrap() else {
        panic!("expected runnable options")
    };
    assert_eq!(options.cpu_ids, vec![1, 3]);
    assert_eq!(options.allocatable_host_memory_bytes, 4096);
    assert!(options.validate_only);
    // Worker binary paths are optional at the CLI level: the missing one is
    // reported against the flavor the node configuration actually references.
    assert_eq!(options.cpu_worker_binary, Some(PathBuf::from("/worker")));
    assert!(options.metal_worker_binary.is_none());
    assert!(options.cuda_worker_binary.is_none());
    assert!(matches!(
        CliOptions::parse([OsString::from("--config"), OsString::from("/x")]),
        Err(SupervisorError::MissingOption("--cpu-ids"))
    ));
    assert!(matches!(
        parse_cpu_ids("1,1"),
        Err(SupervisorError::InvalidCpuIds)
    ));
    assert!(matches!(
        CliOptions::parse(
            [
                "--validate-only",
                "--validate-only",
                "--config",
                "/config.toml",
                "--cpu-worker-binary",
                "/worker",
                "--cpu-ids",
                "1",
                "--allocatable-host-memory-bytes",
                "4096",
            ]
            .map(OsString::from)
        ),
        Err(SupervisorError::DuplicateOption(option)) if option == "--validate-only"
    ));
}

#[test]
fn diagnostic_buffer_is_hard_bounded() {
    let mut diagnostic = BoundedDiagnostic::new();
    diagnostic.push_line(&"x".repeat(MAX_DIAGNOSTIC_RESPONSE_BYTES * 2));
    diagnostic.push_line("must-not-appear");
    let output = diagnostic.finish();

    assert_eq!(output.len(), MAX_DIAGNOSTIC_RESPONSE_BYTES);
    assert!(output.ends_with(TRUNCATED_DIAGNOSTIC_SUFFIX));
    assert!(!output.contains("must-not-appear"));
}

#[test]
fn runtime_diagnostic_is_bounded_and_redacts_credentials() {
    let root = tempfile::tempdir().unwrap();
    let working_directory = root.path().join("work");
    let runtime_directory = root.path().join("run");
    let models_directory = root.path().join("models");
    std::fs::create_dir(&working_directory).unwrap();
    std::fs::create_dir(&runtime_directory).unwrap();
    std::fs::create_dir(&models_directory).unwrap();
    let worker_binary = root.path().join("worker");
    std::fs::write(&worker_binary, b"worker").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&worker_binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let assignment = DeviceAssignment::Cpu {
        thread_budget: 1,
        affinity: vec![1],
        host_memory_limit_bytes: 512,
    };
    let mut config = config(assignment, WorkerBinaryFlavor::Cpu);
    config.working_directory = working_directory;
    config.runtime_directory = runtime_directory;
    config.workers[0].deployment.models_directory = models_directory;
    let inventory = HostInventory {
        effective_cpu_ids: vec![1],
        allocatable_host_memory_bytes: 4096,
        metal_devices: Vec::new(),
        cuda_devices: Vec::new(),
    };
    let binaries = BinaryCatalog::new([(
        WorkerBinaryFlavor::Cpu,
        BinaryRecord {
            path: worker_binary,
            supported_backends: vec![BackendKind::Cpu],
        },
    )]);
    let node = config.validate(&inventory, &binaries).unwrap();
    let view = WorkerView {
        phase: WorkerPhase::RestartPending { at: Instant::now() },
        ready_launches: 1,
        failed_launches: 1,
        process_id: None,
        incarnation: None,
        client: None,
        exit_observation_failed: false,
    };
    let metrics = SupervisorMetrics::default();
    for (counter, value) in [
        (&metrics.launch_attempts, 2),
        (&metrics.readiness_successes, 1),
        (&metrics.readiness_failures, 1),
        (&metrics.restarts_scheduled, 1),
    ] {
        counter.store(value, std::sync::atomic::Ordering::Relaxed);
    }
    let worker_id = id("worker-a");
    let snapshot = SlotSnapshot {
        worker_id: &worker_id,
        config: node.worker(&worker_id).unwrap(),
        view: &view,
        autoscale: None,
    };

    let diagnostic =
        render_runtime_diagnostic(&node, std::iter::once(snapshot), &metrics, Instant::now());

    assert!(diagnostic.len() <= MAX_DIAGNOSTIC_RESPONSE_BYTES);
    assert!(diagnostic.contains("supervisor_status"));
    assert!(diagnostic.contains("launch_attempts_total=2"));
    assert!(diagnostic.contains("worker=worker-a state=restart-pending"));
    assert!(diagnostic.contains("backend=Cpu"));
    assert!(diagnostic.contains("secret=redacted"));
    for secret in ["private-worker-secret", "WORKER_TOKEN", "credential-a"] {
        assert!(!diagnostic.contains(secret));
    }
}

/// The runbook promises that autoscaling decisions hold while a worker
/// of the deployment waits for a restart: a crashed scale-up must not
/// be replaced by another standby before its own restart settles.
#[cfg(unix)]
#[tokio::test]
async fn autoscaling_holds_decisions_while_a_worker_is_restart_pending() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    for directory in ["work", "run", "models"] {
        std::fs::create_dir(root.path().join(directory)).unwrap();
    }
    // A worker that exits at once: every launch fails readiness.
    let worker_binary = root.path().join("worker");
    std::fs::write(&worker_binary, b"#!/bin/sh\nexit 1\n").unwrap();
    std::fs::set_permissions(&worker_binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let assignment = DeviceAssignment::Cpu {
        thread_budget: 1,
        affinity: vec![1],
        host_memory_limit_bytes: 512,
    };
    let mut config = config(assignment, WorkerBinaryFlavor::Cpu);
    config.working_directory = root.path().join("work");
    config.runtime_directory = root.path().join("run");
    config.workers[0].deployment.models_directory = root.path().join("models");
    config.restart.initial_backoff_ms = 60_000;
    config.restart.maximum_backoff_ms = 60_000;
    config.autoscaling = Some(crate::AutoscalingConfig {
        shared_approvals_path: root.path().join("approvals"),
        evaluation_interval_ms: 1_000,
        deployments: BTreeMap::from([(
            "deployment-a".to_string(),
            crate::DeploymentAutoscalingPolicy {
                min_workers: 1,
                max_workers: 1,
                scale_up_queue_depth: 1,
                scale_up_sustained_polls: 1,
                scale_up_stabilization_window_ms: 0,
                scale_down_stabilization_window_ms: 1_000,
            },
        )]),
    });
    let inventory = HostInventory {
        effective_cpu_ids: vec![1],
        allocatable_host_memory_bytes: 4096,
        metal_devices: Vec::new(),
        cuda_devices: Vec::new(),
    };
    let binaries = BinaryCatalog::new([(
        WorkerBinaryFlavor::Cpu,
        BinaryRecord {
            path: worker_binary,
            supported_backends: vec![BackendKind::Cpu],
        },
    )]);
    let node = Arc::new(config.validate(&inventory, &binaries).unwrap());
    let supervision = Arc::new(Supervision::new(
        LockNamespace::open(root.path().join("run")).unwrap(),
        BTreeMap::new(),
        1,
    ));
    let slot = WorkerSlot::new(
        WorkerActor::spawn(
            Arc::clone(&node),
            id("worker-a"),
            ResolvedWorkerSecret {
                bearer_token: ServiceBearerToken::new("private-worker-secret").unwrap(),
            },
            supervision,
        )
        .unwrap(),
    );
    let autoscaler = Autoscaler::from_config(&node).unwrap();
    let deployment = id::<DeploymentId>("deployment-a");
    let slots = std::slice::from_ref(&slot);
    let poll = || poll_deployment(&autoscaler, slots, &deployment, Duration::from_secs(1));
    assert!(poll().await.is_some(), "an idle deployment is observable");

    let before = slot.actor.view().launch_counts();
    slot.actor.start();
    assert_eq!(
        slot.actor.launch_outcome(before).await,
        LaunchOutcome::Failed
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while !matches!(slot.actor.view().phase, WorkerPhase::RestartPending { .. }) {
        assert!(
            Instant::now() < deadline,
            "the failed launch schedules a restart"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        poll().await.is_none(),
        "decisions hold while a worker of the deployment is restart-pending"
    );
    slot.actor.shut_down().await;
}

#[test]
fn ensure_flavor_binaries_requires_the_configured_lanes() {
    let metal = DeviceAssignment::Metal {
        device_id: id::<DeviceId>("metal:1"),
        process_local_device_index: 0,
        shared_memory_limit_bytes: 512,
    };
    let options = CliOptions {
        config: PathBuf::from("/x"),
        cpu_worker_binary: Some(PathBuf::from("/cpu-worker")),
        metal_worker_binary: None,
        cuda_worker_binary: None,
        cpu_ids: vec![0, 1],
        metal_devices: Vec::new(),
        cuda_devices: Vec::new(),
        allocatable_host_memory_bytes: 4294967296,
        validate_only: false,
        canary_worker_id: None,
        rollout_plan: None,
        rollout_abort: false,
        rollout_status: false,
        clear_quarantine: None,
    };
    assert!(matches!(
        ensure_flavor_binaries(&config(metal.clone(), WorkerBinaryFlavor::Metal), &options),
        Err(SupervisorError::MissingFlavorBinary {
            option: "--metal-worker-binary",
            binary: WorkerBinaryFlavor::Metal,
            ..
        })
    ));

    let mut with_metal = options.clone();
    with_metal.metal_worker_binary = Some(PathBuf::from("/metal-worker"));
    ensure_flavor_binaries(&config(metal, WorkerBinaryFlavor::Metal), &with_metal)
        .expect("a declared metal lane with its binary passes the flavor check");
}

#[test]
fn device_declarations_parse_and_fail_closed() {
    let metal = parse_metal_devices("metal:0x1@0,metal:0x2@1").unwrap();
    assert_eq!(
        metal,
        vec![
            (DeviceId::new("metal:0x1").unwrap(), 0),
            (DeviceId::new("metal:0x2").unwrap(), 1)
        ]
    );
    assert!(parse_metal_devices("metal:0x1").is_err());
    assert!(parse_metal_devices("metal:0x1@0,metal:0x1@0").is_err());
    assert!(parse_metal_devices("").is_err());

    let cuda = parse_cuda_devices("GPU-abc@0@8589934592").unwrap();
    assert_eq!(
        cuda,
        vec![(DeviceId::new("GPU-abc").unwrap(), 0, 8_589_934_592u64)]
    );
    assert!(parse_cuda_devices("GPU-abc@0").is_err());
    assert!(parse_cuda_devices("GPU-abc@0@0").is_err());
    assert!(parse_cuda_devices("GPU-abc@1@8589934592@extra").is_err());
}

#[test]
fn canary_worker_id_is_parsed_and_validated() {
    let ParseOutcome::Run(options) = CliOptions::parse([
        OsString::from("--config"),
        OsString::from("/x"),
        OsString::from("--cpu-worker-binary"),
        OsString::from("/w"),
        OsString::from("--cpu-ids"),
        OsString::from("0,1"),
        OsString::from("--allocatable-host-memory-bytes"),
        OsString::from("4294967296"),
        OsString::from("--canary-worker-id"),
        OsString::from("canary-worker-1"),
    ])
    .unwrap() else {
        panic!("expected Run outcome");
    };
    assert_eq!(
        options.canary_worker_id.as_ref().unwrap().as_str(),
        "canary-worker-1"
    );
}

#[test]
fn canary_worker_id_is_optional() {
    let ParseOutcome::Run(options) = CliOptions::parse([
        OsString::from("--config"),
        OsString::from("/x"),
        OsString::from("--cpu-worker-binary"),
        OsString::from("/w"),
        OsString::from("--cpu-ids"),
        OsString::from("0,1"),
        OsString::from("--allocatable-host-memory-bytes"),
        OsString::from("4294967296"),
    ])
    .unwrap() else {
        panic!("expected Run outcome");
    };
    assert!(options.canary_worker_id.is_none());
}

#[test]
fn duplicate_canary_worker_id_is_rejected() {
    let result = CliOptions::parse([
        OsString::from("--config"),
        OsString::from("/x"),
        OsString::from("--cpu-worker-binary"),
        OsString::from("/w"),
        OsString::from("--cpu-ids"),
        OsString::from("0,1"),
        OsString::from("--allocatable-host-memory-bytes"),
        OsString::from("4294967296"),
        OsString::from("--canary-worker-id"),
        OsString::from("canary-1"),
        OsString::from("--canary-worker-id"),
        OsString::from("canary-2"),
    ]);
    assert!(result.is_err());
    let error = result.err().unwrap().to_string();
    assert!(error.contains("repeated"));
}

use izwi_serving_protocol::{
    BackendKind, CancellationBehavior, DeviceAssignment, InputFormat, OutputFormat, TaskKind,
};
use izwi_serving_supervisor::{NodeConfig, WorkerBinaryFlavor, NODE_CONFIG_SCHEMA_VERSION};
use std::collections::BTreeSet;

const CPU_EXAMPLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/serving/examples/one-device-cpu.toml"
));
const METAL_EXAMPLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/serving/examples/one-device-metal.toml"
));
const CUDA_EXAMPLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/serving/examples/multi-device-cuda.toml"
));

fn parse(bytes: &[u8]) -> NodeConfig {
    NodeConfig::parse_bounded(bytes).expect("example must match the bounded node schema")
}

fn assert_explicit_chat_profile(worker: &izwi_serving_supervisor::WorkerConfig) {
    let deployment = &worker.deployment;
    assert_eq!(deployment.task, TaskKind::Chat);
    assert_eq!(deployment.precision, "gguf-q4_k_m");
    assert_eq!(deployment.execution_representation, "native-lfm2");
    assert_eq!(deployment.tokenizer_revision, None);
    assert_eq!(deployment.capability.streaming, worker.streaming);
    assert!(!deployment.capability.realtime);
    assert_eq!(
        deployment.capability.cancellation,
        CancellationBehavior::Cooperative
    );
    assert_eq!(
        deployment.capability.accepted_input_formats,
        BTreeSet::from([InputFormat::ChatMessages])
    );
    assert_eq!(
        deployment.capability.output_formats,
        BTreeSet::from([OutputFormat::Text])
    );
    assert_eq!(
        deployment.capability.max_input_bytes,
        worker.max_request_bytes as u64
    );
    assert_eq!(deployment.capability.max_context_tokens, Some(32));
    assert_eq!(deployment.capability.max_output_tokens, Some(32));
}

#[test]
fn one_device_examples_match_their_exact_backend_assignments() {
    let cpu = parse(CPU_EXAMPLE);
    assert_eq!(cpu.schema_version, NODE_CONFIG_SCHEMA_VERSION);
    assert_eq!(cpu.workers.len(), 1);
    assert_eq!(cpu.workers[0].binary, WorkerBinaryFlavor::Cpu);
    assert!(matches!(
        cpu.workers[0].assignment,
        DeviceAssignment::Cpu { .. }
    ));
    assert_eq!(cpu.workers[0].deployment.backend, BackendKind::Cpu);
    assert_explicit_chat_profile(&cpu.workers[0]);

    let metal = parse(METAL_EXAMPLE);
    assert_eq!(metal.schema_version, NODE_CONFIG_SCHEMA_VERSION);
    assert_eq!(metal.workers.len(), 1);
    assert_eq!(metal.workers[0].binary, WorkerBinaryFlavor::Metal);
    let DeviceAssignment::Metal {
        device_id,
        process_local_device_index,
        ..
    } = &metal.workers[0].assignment
    else {
        panic!("Metal example must use an exact Metal assignment");
    };
    assert!(device_id.as_str().starts_with("metal:REPLACE_"));
    assert_eq!(*process_local_device_index, 0);
    assert_eq!(metal.workers[0].deployment.backend, BackendKind::Metal);
    assert_explicit_chat_profile(&metal.workers[0]);
}

#[test]
fn cuda_example_assigns_one_unique_uuid_per_process() {
    let config = parse(CUDA_EXAMPLE);
    assert_eq!(config.schema_version, NODE_CONFIG_SCHEMA_VERSION);
    assert_eq!(config.workers.len(), 3);

    let mut device_uuids = BTreeSet::new();
    let mut binds = BTreeSet::new();
    for worker in &config.workers {
        assert_eq!(worker.binary, WorkerBinaryFlavor::Cuda);
        assert_eq!(worker.deployment.backend, BackendKind::Cuda);
        assert_explicit_chat_profile(worker);
        assert!(binds.insert(worker.bind));
        let DeviceAssignment::Cuda {
            device_uuid,
            process_local_device_index,
            ..
        } = &worker.assignment
        else {
            panic!("CUDA example worker must use an exact CUDA assignment");
        };
        assert_eq!(*process_local_device_index, 0);
        assert!(device_uuids.insert(device_uuid.as_str()));
    }
}

#[test]
fn cuda_example_distinguishes_replicas_from_an_independent_deployment() {
    let config = parse(CUDA_EXAMPLE);
    let deployment_ids = config
        .workers
        .iter()
        .map(|worker| worker.deployment.deployment_id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        deployment_ids
            .iter()
            .filter(|id| **id == "example-chat-cuda-v1")
            .count(),
        2
    );
    assert_eq!(
        deployment_ids
            .iter()
            .filter(|id| **id == "example-chat-cuda-specialized-v1")
            .count(),
        1
    );
}

#[test]
fn examples_keep_secrets_external_and_pin_existing_chat_capabilities() {
    for bytes in [CPU_EXAMPLE, METAL_EXAMPLE, CUDA_EXAMPLE] {
        let config = parse(bytes);
        assert!(config
            .workers
            .iter()
            .all(|worker| worker.bind.ip().is_loopback() && worker.bind.port() != 0));

        let text = std::str::from_utf8(bytes).expect("examples are UTF-8 TOML");
        assert!(text.contains("bearer_token_env = \"REPLACE_"));
        assert!(text.lines().any(|line| line.trim() == "task = \"chat\""));
        assert!(!text
            .lines()
            .any(|line| line.trim_start().starts_with("bearer_token =")));
    }
}

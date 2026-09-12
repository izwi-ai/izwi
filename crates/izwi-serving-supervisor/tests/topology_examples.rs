use izwi_serving_protocol::{BackendKind, DeviceAssignment};
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
fn examples_keep_secrets_external_and_do_not_invent_task_keys() {
    for bytes in [CPU_EXAMPLE, METAL_EXAMPLE, CUDA_EXAMPLE] {
        let config = parse(bytes);
        assert!(config
            .workers
            .iter()
            .all(|worker| worker.bind.ip().is_loopback() && worker.bind.port() != 0));

        let text = std::str::from_utf8(bytes).expect("examples are UTF-8 TOML");
        assert!(text.contains("bearer_token_env = \"REPLACE_"));
        assert!(!text.lines().any(|line| {
            let key = line.trim_start();
            key.starts_with("bearer_token =") || key.starts_with("task =")
        }));
    }
}

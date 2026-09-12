use izwi_serving_protocol::BackendKind;
use izwi_serving_supervisor::{NodeConfig, WorkerBinaryFlavor, NODE_CONFIG_SCHEMA_VERSION};

const EXAMPLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../config/serving/izwi-serving-node.example.toml"
));

#[test]
fn packaged_node_example_matches_the_bounded_schema() {
    let config = NodeConfig::parse_bounded(EXAMPLE).expect("packaged node example must parse");

    assert_eq!(config.schema_version, NODE_CONFIG_SCHEMA_VERSION);
    assert_eq!(config.workers.len(), 1);
    assert_eq!(config.workers[0].binary, WorkerBinaryFlavor::Cpu);
    assert_eq!(config.workers[0].assignment.backend(), BackendKind::Cpu);
}

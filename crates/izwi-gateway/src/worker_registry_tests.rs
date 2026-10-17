use super::*;
use izwi_serving_client::WorkerClientConfig;
use izwi_serving_protocol::{
    ArtifactRevision, Capability, CapacitySnapshot, CredentialId, DeviceAssignment, DeviceId,
    ServiceBearerToken, ServiceCredentials,
};

fn id<T>(value: &str) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_string()).unwrap()
}

fn client(port: u16) -> WorkerClient {
    WorkerClient::new(
        &format!("http://127.0.0.1:{port}"),
        ServiceCredentials {
            credential_id: id::<CredentialId>("gateway-key"),
            bearer_token: ServiceBearerToken::new("secret").unwrap(),
        },
        WorkerClientConfig::default(),
    )
    .unwrap()
}

fn descriptor(worker: &str, incarnation: &str, backend: BackendKind) -> WorkerDescriptor {
    let assignment = match backend {
        BackendKind::Cpu => DeviceAssignment::Cpu {
            thread_budget: 2,
            affinity: vec![],
            host_memory_limit_bytes: 1024,
        },
        BackendKind::Metal => DeviceAssignment::Metal {
            device_id: id::<DeviceId>("metal:1"),
            process_local_device_index: 0,
            shared_memory_limit_bytes: 1024,
        },
        BackendKind::Cuda => DeviceAssignment::Cuda {
            device_uuid: id::<DeviceId>("GPU-a"),
            process_local_device_index: 0,
            device_memory_limit_bytes: 1024,
            host_memory_limit_bytes: 1024,
        },
        BackendKind::Unknown => DeviceAssignment::Unknown,
    };
    WorkerDescriptor {
        schema_version: PROTOCOL_V1,
        supported_protocol_versions: vec![PROTOCOL_V1],
        worker_id: id::<WorkerId>(worker),
        node_id: id::<NodeId>("node-a"),
        incarnation_id: id::<IncarnationId>(incarnation),
        build_version: "test".into(),
        assignment,
        features: BTreeSet::from([WorkerFeature::Streaming, WorkerFeature::Cancellation]),
    }
}

fn deployment(name: &str, alias: &str, backend: BackendKind) -> LoadedDeployment {
    LoadedDeployment {
        deployment_id: id::<DeploymentId>(name),
        public_model: id::<ModelAlias>(alias),
        artifact_revision: id::<ArtifactRevision>("sha256:artifact"),
        model_generation: ModelGeneration::new(1).unwrap(),
        task: TaskKind::Chat,
        backend,
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
            max_input_bytes: 4096,
            max_context_tokens: Some(1024),
            max_output_tokens: Some(128),
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
    }
}

fn capacity(max: u32, active: u32, credits: u32) -> CapacitySnapshot {
    CapacitySnapshot {
        max_active_invocations: max,
        active_invocations: active,
        max_queued_invocations: 0,
        queued_invocations: 0,
        max_sessions: 0,
        reserved_sessions: 0,
        available_admission_credits: credits,
        outstanding_cost_units: 0,
    }
}

fn status(
    descriptor: &WorkerDescriptor,
    sequence: u64,
    deployments: Vec<LoadedDeployment>,
    capacity: CapacitySnapshot,
) -> WorkerStatus {
    WorkerStatus {
        schema_version: PROTOCOL_V1,
        worker_id: descriptor.worker_id.clone(),
        node_id: descriptor.node_id.clone(),
        incarnation_id: descriptor.incarnation_id.clone(),
        status_sequence: sequence,
        process_state: WorkerProcessState::Running,
        deployments,
        capacity,
    }
}

fn registration(
    worker: &str,
    incarnation: &str,
    backend: BackendKind,
    port: u16,
    validated_capacity: u32,
) -> ApprovedWorker {
    ApprovedWorker {
        descriptor: descriptor(worker, incarnation, backend),
        client: client(port),
        approved_deployments: {
            let deployment = deployment("chat-prod", "lfm2", backend);
            BTreeMap::from([(
                deployment.deployment_id.clone(),
                ApprovedDeployment::from_loaded(&deployment),
            )])
        },
        validated_capacity,
    }
}

fn selection() -> WorkerSelectionRequest {
    WorkerSelectionRequest {
        protocol_version: PROTOCOL_V1,
        deployment_id: id::<DeploymentId>("chat-prod"),
        public_model: id::<ModelAlias>("lfm2"),
        task: TaskKind::Chat,
        input_format: InputFormat::ChatMessages,
        output_format: OutputFormat::Text,
        streaming: true,
        realtime: false,
        cancellation: Some(CancellationBehavior::Cooperative),
        backend_policy: BackendPolicy::ANY,
        input_bytes: 128,
        context_tokens: Some(64),
        output_tokens: Some(32),
        required_features: BTreeSet::new(),
    }
}

#[test]
fn config_and_registration_limits_are_fail_closed() {
    let config = WorkerRegistryConfig {
        max_workers: 0,
        ..WorkerRegistryConfig::default()
    };
    assert_eq!(
        WorkerRegistry::new(config).unwrap_err(),
        WorkerRegistryError::InvalidConfig("max_workers is outside the supported range")
    );

    let config = WorkerRegistryConfig {
        circuit_failure_threshold: 0,
        ..WorkerRegistryConfig::default()
    };
    assert_eq!(
        WorkerRegistry::new(config).unwrap_err(),
        WorkerRegistryError::InvalidConfig(
            "circuit_failure_threshold is outside the supported range"
        )
    );

    let config = WorkerRegistryConfig {
        circuit_open_duration: MAX_CIRCUIT_OPEN_DURATION + Duration::from_secs(1),
        ..WorkerRegistryConfig::default()
    };
    assert_eq!(
        WorkerRegistry::new(config).unwrap_err(),
        WorkerRegistryError::InvalidConfig(
            "circuit_open_duration is outside the supported range"
        )
    );

    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        max_workers: 1,
        max_deployments_per_worker: 1,
        max_local_dispatches: 1,
        status_ttl: Duration::from_secs(10),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    registry
        .approve(registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1))
        .unwrap();
    assert_eq!(
        registry
            .approve(registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 1))
            .unwrap_err(),
        WorkerRegistryError::WorkerLimitReached
    );
}

#[test]
fn generation_gate_restricts_selection_to_the_eligible_generation() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let old = registration("worker-old", "inc-old", BackendKind::Cpu, 9101, 4);
    let old_descriptor = old.descriptor.clone();
    registry.approve(old).unwrap();

    let mut successor = registration("worker-new", "inc-new", BackendKind::Cpu, 9102, 4);
    let successor_descriptor = successor.descriptor.clone();
    let mut successor_deployment = deployment("chat-prod", "lfm2", BackendKind::Cpu);
    successor_deployment.model_generation = ModelGeneration::new(2).unwrap();
    successor.approved_deployments = BTreeMap::from([(
        successor_deployment.deployment_id.clone(),
        ApprovedDeployment::from_loaded(&successor_deployment),
    )]);
    registry.approve(successor).unwrap();

    let observed = Instant::now();
    registry
        .observe_status_at(
            status(
                &old_descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(4, 0, 4),
            ),
            observed,
        )
        .unwrap();
    registry
        .observe_status_at(
            status(
                &successor_descriptor,
                1,
                vec![successor_deployment],
                capacity(4, 0, 4),
            ),
            observed,
        )
        .unwrap();

    // Without the gate both generations are selectable; the deterministic
    // tie-break prefers the lexicographically smaller worker id.
    let selected = registry
        .select_and_reserve_at(&selection(), observed)
        .unwrap();
    assert_eq!(selected.key.worker_id, id::<WorkerId>("worker-new"));

    // With the rollout gate only the eligible generation is selectable;
    // the window's current generation serves until cutover completes.
    registry.set_generation_gate(Arc::new(|_, _| Some(ModelGeneration::new(1).unwrap())));
    let selected = registry
        .select_and_reserve_at(&selection(), observed)
        .unwrap();
    assert_eq!(selected.key.worker_id, id::<WorkerId>("worker-old"));
    assert_eq!(selected.model_generation, ModelGeneration::new(1).unwrap());

    // After the cutover swap the successor is the only eligible
    // generation and the drained predecessor is skipped.
    registry.set_generation_gate(Arc::new(|_, _| Some(ModelGeneration::new(2).unwrap())));
    let selected = registry
        .select_and_reserve_at(&selection(), observed)
        .unwrap();
    assert_eq!(selected.key.worker_id, id::<WorkerId>("worker-new"));
    assert_eq!(selected.model_generation, ModelGeneration::new(2).unwrap());
}

#[test]
fn replacement_incarnation_starts_unready_and_rejects_late_status() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let first = registration("worker-a", "inc-old", BackendKind::Cpu, 9101, 1);
    let old_descriptor = first.descriptor.clone();
    registry.approve(first).unwrap();
    registry
        .observe_status(status(
            &old_descriptor,
            1,
            vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
            capacity(1, 0, 1),
        ))
        .unwrap();

    let replacement = registration("worker-a", "inc-new", BackendKind::Cpu, 9102, 1);
    registry.approve(replacement).unwrap();
    assert_eq!(
        registry.select_and_reserve(&selection()).unwrap_err(),
        WorkerRegistryError::NoEligibleWorker
    );
    assert_eq!(
        registry
            .observe_status(status(
                &old_descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap_err(),
        WorkerRegistryError::UnknownOrStaleIncarnation
    );
}

#[test]
fn status_sequence_and_receiver_time_must_increase() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    let now = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            now,
        )
        .unwrap();
    assert_eq!(
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    2,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now + Duration::from_secs(1),
            )
            .unwrap_err(),
        WorkerRegistryError::StaleStatusSequence
    );
    assert_eq!(
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    3,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now - Duration::from_millis(1),
            )
            .unwrap_err(),
        WorkerRegistryError::ObservationTimeRegressed
    );
}

#[test]
fn freshness_expires_at_the_receiver_clock_boundary() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        status_ttl: Duration::from_secs(10),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    let observed = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            observed,
        )
        .unwrap();
    assert!(registry
        .select_and_reserve_at(&selection(), observed + Duration::from_secs(10))
        .is_err());
    let selected = registry
        .select_and_reserve_at(&selection(), observed + Duration::from_millis(9_999))
        .unwrap();
    drop(selected);
}

#[test]
fn partition_expiry_blocks_new_work_until_a_newer_authenticated_status() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        status_ttl: Duration::from_secs(10),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    let observed = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(2, 0, 2),
            ),
            observed,
        )
        .unwrap();

    let mut active = registry
        .select_and_reserve_at(&selection(), observed + Duration::from_secs(1))
        .unwrap();
    active.dispatch.mark_accepted().unwrap();
    assert_eq!(
        registry
            .select_and_reserve_at(&selection(), observed + Duration::from_secs(10))
            .unwrap_err(),
        WorkerRegistryError::NoEligibleWorker,
        "status expiry must stop only new selection"
    );
    {
        let inner = lock_recover(&registry.inner);
        assert_eq!(inner.dispatches.len(), 1, "active ownership is retained");
    }

    registry
        .observe_status_at(
            status(
                &descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(2, 1, 1),
            ),
            observed + Duration::from_secs(11),
        )
        .unwrap();
    let after_reconnect = registry
        .select_and_reserve_at(&selection(), observed + Duration::from_secs(11))
        .unwrap();
    assert_eq!(after_reconnect.key.incarnation_id.as_str(), "inc-a");
    drop(after_reconnect);
    drop(active);
}

#[test]
fn readiness_is_per_deployment_and_separate_from_saturation() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    let mut loading = deployment("chat-prod", "lfm2", BackendKind::Cpu);
    loading.readiness = ModelReadiness::Loading;
    let now = Instant::now();
    registry
        .observe_status_at(
            status(&descriptor, 1, vec![loading], capacity(1, 1, 0)),
            now,
        )
        .unwrap();
    let snapshot = registry.deployment_readiness_at(&id("chat-prod"), now);
    assert_eq!(snapshot.approved_workers, 1);
    assert_eq!(snapshot.fresh_running_workers, 1);
    assert_eq!(snapshot.ready_workers, 0);
    assert_eq!(snapshot.workers_with_observed_credit, 0);
}

#[test]
fn status_rejects_unapproved_duplicate_backend_and_capacity_claims() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();

    assert_eq!(
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("other", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ))
            .unwrap_err(),
        WorkerRegistryError::UnapprovedDeployment
    );
    let duplicate = deployment("chat-prod", "lfm2", BackendKind::Cpu);
    assert_eq!(
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![duplicate.clone(), duplicate],
                capacity(1, 0, 1),
            ))
            .unwrap_err(),
        WorkerRegistryError::DuplicateDeployment
    );
    assert_eq!(
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cuda)],
                capacity(1, 0, 1),
            ))
            .unwrap_err(),
        WorkerRegistryError::BackendMismatch
    );
    let mut wrong_generation = deployment("chat-prod", "lfm2", BackendKind::Cpu);
    wrong_generation.model_generation = ModelGeneration::new(2).unwrap();
    assert_eq!(
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![wrong_generation],
                capacity(1, 0, 1),
            ))
            .unwrap_err(),
        WorkerRegistryError::ApprovedDeploymentMismatch
    );
    assert_eq!(
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(2, 0, 2),
            ))
            .unwrap_err(),
        WorkerRegistryError::CapacityMismatch
    );
}

#[test]
fn exact_capability_and_backend_filters_fail_closed() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    registry
        .observe_status(status(
            &descriptor,
            1,
            vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
            capacity(1, 0, 1),
        ))
        .unwrap();

    let mutations: Vec<Box<dyn Fn(&mut WorkerSelectionRequest)>> = vec![
        Box::new(|request| request.public_model = id("other-model")),
        Box::new(|request| request.task = TaskKind::TextToSpeech),
        Box::new(|request| request.input_format = InputFormat::EncodedAudio),
        Box::new(|request| request.output_format = OutputFormat::Json),
        Box::new(|request| request.realtime = true),
        Box::new(|request| request.cancellation = Some(CancellationBehavior::TeardownRequired)),
        Box::new(|request| request.backend_policy = BackendPolicy::CUDA_ONLY),
        Box::new(|request| request.input_bytes = 4097),
        Box::new(|request| request.context_tokens = Some(1025)),
        Box::new(|request| request.output_tokens = Some(129)),
        Box::new(|request| request.protocol_version = SchemaVersion::new(2, 0)),
    ];
    for mutate in mutations {
        let mut request = selection();
        mutate(&mut request);
        assert_eq!(
            registry.select_and_reserve(&request).unwrap_err(),
            WorkerRegistryError::NoEligibleWorker
        );
    }
}

#[test]
fn mixed_backends_route_by_declared_policy_without_a_fixed_preference() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let workers = [
        ("worker-a-metal", "inc-metal", BackendKind::Metal, 9101),
        ("worker-b-cuda", "inc-cuda", BackendKind::Cuda, 9102),
        ("worker-c-cpu", "inc-cpu", BackendKind::Cpu, 9103),
    ];
    for (worker, incarnation, backend, port) in workers {
        let approved = registration(worker, incarnation, backend, port, 1);
        let descriptor = approved.descriptor.clone();
        registry.approve(approved).unwrap();
        registry
            .observe_status(status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", backend)],
                capacity(1, 0, 1),
            ))
            .unwrap();
    }

    // ANY uses the normal load/stable-identity ordering. It does not
    // silently prefer CUDA, Metal, or CPU.
    let selected = registry.select_and_reserve(&selection()).unwrap();
    assert_eq!(selected.backend, BackendKind::Metal);
    drop(selected);

    for (policy, expected) in [
        (BackendPolicy::CPU_ONLY, BackendKind::Cpu),
        (BackendPolicy::METAL_ONLY, BackendKind::Metal),
        (BackendPolicy::CUDA_ONLY, BackendKind::Cuda),
    ] {
        let mut request = selection();
        request.backend_policy = policy;
        let selected = registry.select_and_reserve(&request).unwrap();
        assert_eq!(selected.backend, expected);
        drop(selected);
    }
}

#[test]
fn selection_uses_capacity_weighted_score_then_stable_identity() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 4);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_b).unwrap();
    registry.approve(worker_a).unwrap();
    let now = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor_a,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(4, 2, 2),
            ),
            now,
        )
        .unwrap();
    registry
        .observe_status_at(
            status(
                &descriptor_b,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(2, 1, 1),
            ),
            now,
        )
        .unwrap();
    let selected = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(selected.key.worker_id.as_str(), "worker-a");
}

#[test]
fn minor_zero_worker_registers_observes_and_stays_selectable() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let worker = registration("worker-minor0", "inc-minor0", BackendKind::Cpu, 9103, 2);
    let mut descriptor = worker.descriptor.clone();
    descriptor.schema_version = SchemaVersion::new(1, 0);
    descriptor.supported_protocol_versions = vec![SchemaVersion::new(1, 0)];
    let minor_zero_worker = ApprovedWorker {
        descriptor: descriptor.clone(),
        ..worker
    };
    registry.approve(minor_zero_worker).unwrap();

    let mut minor_zero_status = status(
        &descriptor,
        1,
        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
        capacity(2, 0, 2),
    );
    minor_zero_status.schema_version = SchemaVersion::new(1, 0);
    registry
        .observe_status_at(minor_zero_status, Instant::now())
        .unwrap();

    let selected = registry
        .select_and_reserve_at(&selection(), Instant::now())
        .unwrap();
    assert_eq!(selected.key.worker_id.as_str(), "worker-minor0");
}

/// Two approved chat workers with the given capacity snapshots and
/// deployments, observed now.
fn two_workers(
    registry: &WorkerRegistry,
    a: (CapacitySnapshot, LoadedDeployment),
    b: (CapacitySnapshot, LoadedDeployment),
) -> Instant {
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9112, 8);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9113, 8);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_a).unwrap();
    registry.approve(worker_b).unwrap();
    let now = Instant::now();
    for (descriptor, (capacity, deployment)) in [(&descriptor_a, a), (&descriptor_b, b)] {
        registry
            .observe_status_at(status(descriptor, 1, vec![deployment], capacity), now)
            .unwrap();
    }
    now
}

fn chat_deployment() -> LoadedDeployment {
    deployment("chat-prod", "lfm2", BackendKind::Cpu)
}

/// Affinity that claims `blocks` leading blocks on one worker.
struct HeldBy(&'static str, usize);

impl PrefixAffinity for HeldBy {
    fn matched_blocks(&self, worker: &WorkerInstanceKey) -> usize {
        if worker.worker_id.as_str() == self.0 {
            self.1
        } else {
            0
        }
    }
}

fn routed(
    registry: &WorkerRegistry,
    affinity: Option<&dyn PrefixAffinity>,
    now: Instant,
) -> String {
    registry
        .select_and_reserve_routed(&selection(), None, affinity, &NoFleetCapacity, now)
        .unwrap()
        .key
        .worker_id
        .as_str()
        .to_string()
}

#[test]
fn prefix_holder_wins_within_the_balance_threshold_and_spills_beyond_it() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        routing: RoutingConfig {
            prefix_balance_threshold: 2,
            ..RoutingConfig::default()
        },
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 0, 8), chat_deployment()),
        (capacity(8, 2, 6), chat_deployment()),
    );
    // Without affinity the idle worker wins; holding the prefix, worker-b
    // wins while it is at most two requests busier.
    assert_eq!(routed(&registry, None, now), "worker-a");
    assert_eq!(
        routed(&registry, Some(&HeldBy("worker-b", 3)), now),
        "worker-b"
    );

    // A third outstanding request on worker-b crosses the threshold.
    let held = registry
        .select_and_reserve_routed(
            &selection(),
            None,
            Some(&HeldBy("worker-b", 3)),
            &NoFleetCapacity,
            now,
        )
        .unwrap();
    assert_eq!(held.key.worker_id.as_str(), "worker-b");
    assert_eq!(
        routed(&registry, Some(&HeldBy("worker-b", 3)), now),
        "worker-a"
    );
    drop(held);
    assert_eq!(
        routed(&registry, Some(&HeldBy("worker-b", 3)), now),
        "worker-b"
    );
}

#[test]
fn longer_prefix_beats_shorter_and_ties_fall_back_to_load() {
    struct Both;
    impl PrefixAffinity for Both {
        fn matched_blocks(&self, worker: &WorkerInstanceKey) -> usize {
            if worker.worker_id.as_str() == "worker-a" {
                2
            } else {
                5
            }
        }
    }
    struct Equal;
    impl PrefixAffinity for Equal {
        fn matched_blocks(&self, _worker: &WorkerInstanceKey) -> usize {
            3
        }
    }
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 0, 8), chat_deployment()),
        (capacity(8, 1, 7), chat_deployment()),
    );
    assert_eq!(routed(&registry, Some(&Both), now), "worker-b");
    assert_eq!(routed(&registry, Some(&Equal), now), "worker-a");
}

#[test]
fn kv_pressured_workers_are_chosen_only_when_every_worker_is() {
    let pressured = |usage| LoadedDeployment {
        kv_cache_usage_pct: Some(usage),
        ..chat_deployment()
    };
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 0, 8), pressured(95.0)),
        (capacity(8, 3, 5), pressured(40.0)),
    );
    // worker-a is idle but nearly full; worker-b is busier with headroom.
    // A prefix held on the pressured worker does not pull work onto it.
    assert_eq!(routed(&registry, None, now), "worker-b");
    assert_eq!(
        routed(&registry, Some(&HeldBy("worker-a", 9)), now),
        "worker-b"
    );

    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 0, 8), pressured(95.0)),
        (capacity(8, 3, 5), pressured(92.0)),
    );
    assert_eq!(routed(&registry, None, now), "worker-a");
}

#[test]
fn equal_load_prefers_the_higher_observed_output_rate() {
    let rated = |rate| LoadedDeployment {
        tokens_out_per_s_ema: Some(rate),
        ..chat_deployment()
    };
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 1, 7), rated(12.0)),
        (capacity(8, 1, 7), rated(48.0)),
    );
    assert_eq!(routed(&registry, None, now), "worker-b");

    // Load still comes first.
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let now = two_workers(
        &registry,
        (capacity(8, 1, 7), rated(12.0)),
        (capacity(8, 2, 6), rated(48.0)),
    );
    assert_eq!(routed(&registry, None, now), "worker-a");
}

#[test]
fn routing_config_bounds_are_fail_closed() {
    let invalid = |kv_pressure_pct: f64, prefix_balance_threshold: u32| {
        WorkerRegistry::new(WorkerRegistryConfig {
            routing: RoutingConfig {
                prefix_balance_threshold,
                kv_pressure_pct,
            },
            ..WorkerRegistryConfig::default()
        })
        .is_err()
    };
    assert!(invalid(f64::NAN, 4));
    assert!(invalid(0.0, 4));
    assert!(invalid(101.0, 4));
    assert!(invalid(90.0, MAX_PREFIX_BALANCE_THRESHOLD + 1));
    assert!(!invalid(90.0, 0));
    assert!(!invalid(100.0, MAX_PREFIX_BALANCE_THRESHOLD));
}

#[test]
fn randomized_tie_breaking_distributes_choices_across_equal_workers() {
    let config = WorkerRegistryConfig {
        randomized_tie_breaking: true,
        ..WorkerRegistryConfig::default()
    };
    let registry = WorkerRegistry::new(config).unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 10);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 10);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_b).unwrap();
    registry.approve(worker_a).unwrap();
    let now = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor_a,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(10, 0, 10),
            ),
            now,
        )
        .unwrap();
    registry
        .observe_status_at(
            status(
                &descriptor_b,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(10, 0, 10),
            ),
            now,
        )
        .unwrap();

    let mut a_selected = 0;
    let mut b_selected = 0;
    for _ in 0..100 {
        let selected = registry.select_and_reserve_at(&selection(), now).unwrap();
        if selected.key.worker_id.as_str() == "worker-a" {
            a_selected += 1;
        } else if selected.key.worker_id.as_str() == "worker-b" {
            b_selected += 1;
        }
        drop(selected);
    }
    assert!(a_selected > 0, "worker-a was never selected");
    assert!(b_selected > 0, "worker-b was never selected");
}

struct StubFleetCapacity {
    claims: BTreeMap<(String, String), u64>,
}

impl FleetCapacityView for StubFleetCapacity {
    fn cluster_claims(&self, worker: &WorkerInstanceKey) -> u64 {
        self.claims
            .get(&(
                worker.worker_id.as_str().to_string(),
                worker.incarnation_id.as_str().to_string(),
            ))
            .copied()
            .unwrap_or(0)
    }
}

#[test]
fn fleet_cluster_claims_steer_selection_away_from_peer_filled_workers() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 4);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 4);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_a).unwrap();
    registry.approve(worker_b).unwrap();
    let now = Instant::now();
    for descriptor in [&descriptor_a, &descriptor_b] {
        registry
            .observe_status_at(
                status(
                    descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(4, 0, 2),
                ),
                now,
            )
            .unwrap();
    }
    let peers_filled_a = StubFleetCapacity {
        claims: BTreeMap::from([(("worker-a".to_string(), "inc-a".to_string()), 2)]),
    };
    let selected = registry
        .select_and_reserve_with_fleet_at(&selection(), None, &peers_filled_a, now)
        .unwrap();
    assert_eq!(
        selected.key.worker_id.as_str(),
        "worker-b",
        "peer claims consume worker-a's observed credits"
    );
    drop(selected);
    let peers_filled_both = StubFleetCapacity {
        claims: BTreeMap::from([
            (("worker-a".to_string(), "inc-a".to_string()), 2),
            (("worker-b".to_string(), "inc-b".to_string()), 5),
        ]),
    };
    assert!(
        matches!(
            registry.select_and_reserve_with_fleet_at(
                &selection(),
                None,
                &peers_filled_both,
                now
            ),
            Err(WorkerRegistryError::NoEligibleWorker)
        ),
        "no worker is eligible when cluster claims exhaust all observed credits"
    );
}

#[tokio::test]
async fn circuit_opens_on_partition_and_recovers_through_half_open_probe() {
    // End-to-end partition behavior with real time: repeated transport
    // failures (a severed gateway/worker link) trip the breaker and stop
    // selection; cooldown alone does not restore eligibility; a newer
    // authenticated status admits exactly one half-open probe, whose
    // acceptance closes the circuit and resumes normal selection.
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        circuit_failure_threshold: 3,
        circuit_open_duration: Duration::from_millis(50),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let worker = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
    let descriptor = worker.descriptor.clone();
    registry.approve(worker).unwrap();
    let observe = |registry: &WorkerRegistry, sequence: u64| {
        registry
            .observe_status_at(
                status(
                    &descriptor,
                    sequence,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 0, 2),
                ),
                Instant::now(),
            )
            .unwrap()
    };
    observe(&registry, 1);
    let selected = registry
        .select_and_reserve_at(&selection(), Instant::now())
        .unwrap();
    let key = selected.key.clone();
    drop(selected);

    for _ in 0..3 {
        registry.report_worker_transport_failure(&key).unwrap();
    }
    assert!(
        matches!(
            registry.select_and_reserve_at(&selection(), Instant::now()),
            Err(WorkerRegistryError::NoEligibleWorker)
        ),
        "an open circuit must stop selection even with a fresh observation"
    );
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(
        matches!(
            registry.select_and_reserve_at(&selection(), Instant::now()),
            Err(WorkerRegistryError::NoEligibleWorker)
        ),
        "cooldown alone must not restore eligibility without a newer status"
    );

    observe(&registry, 2);
    let mut probe = registry
        .select_and_reserve_at(&selection(), Instant::now())
        .expect("a newer status must admit exactly one half-open probe");
    probe.dispatch.mark_accepted().unwrap();
    drop(probe);
    registry
        .select_and_reserve_at(&selection(), Instant::now())
        .expect("acceptance must close the circuit and resume selection");
}

#[test]
fn local_dispatches_are_bounded_and_affect_selection() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        max_local_dispatches: 2,
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_a).unwrap();
    registry.approve(worker_b).unwrap();
    let now = Instant::now();
    for descriptor in [&descriptor_a, &descriptor_b] {
        registry
            .observe_status_at(
                status(
                    descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 0, 2),
                ),
                now,
            )
            .unwrap();
    }

    let first = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(first.key.worker_id.as_str(), "worker-a");
    let second = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(second.key.worker_id.as_str(), "worker-b");
    assert_eq!(
        registry
            .select_and_reserve_at(&selection(), now)
            .unwrap_err(),
        WorkerRegistryError::LocalDispatchLimitReached
    );
    drop(first);
    let replacement = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(replacement.key.worker_id.as_str(), "worker-a");
}

#[test]
fn accepted_dispatch_reconciles_only_after_a_newer_status() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 2);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 2);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_a).unwrap();
    registry.approve(worker_b).unwrap();
    let now = Instant::now();
    for descriptor in [&descriptor_a, &descriptor_b] {
        registry
            .observe_status_at(
                status(
                    descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(2, 0, 2),
                ),
                now,
            )
            .unwrap();
    }

    let mut first = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(first.key.worker_id.as_str(), "worker-a");
    first.dispatch.mark_accepted().unwrap();
    {
        let inner = lock_recover(&registry.inner);
        assert_eq!(unreconciled_dispatches(&inner, &first.key), 1);
    }
    let before_update = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(before_update.key.worker_id.as_str(), "worker-b");
    drop(before_update);

    registry
        .observe_status_at(
            status(
                &descriptor_a,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(2, 1, 1),
            ),
            now + Duration::from_millis(1),
        )
        .unwrap();
    {
        let inner = lock_recover(&registry.inner);
        assert_eq!(unreconciled_dispatches(&inner, &first.key), 0);
    }
    let reconciled = registry
        .select_and_reserve_at(&selection(), now + Duration::from_millis(1))
        .unwrap();
    assert_eq!(reconciled.key.worker_id.as_str(), "worker-b");

    // Releasing reconciled and unreconciled dispatches empties the
    // per-worker index exactly.
    drop(first);
    drop(reconciled);
    let inner = lock_recover(&registry.inner);
    assert!(inner.dispatches.is_empty());
    assert!(inner.worker_dispatches.is_empty());
}

#[test]
fn exact_worker_exclusion_selects_one_bounded_alternate() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let worker_a = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let worker_b = registration("worker-b", "inc-b", BackendKind::Cpu, 9102, 1);
    let descriptor_a = worker_a.descriptor.clone();
    let descriptor_b = worker_b.descriptor.clone();
    registry.approve(worker_a).unwrap();
    registry.approve(worker_b).unwrap();
    let now = Instant::now();
    for descriptor in [&descriptor_a, &descriptor_b] {
        registry
            .observe_status_at(
                status(
                    descriptor,
                    1,
                    vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                    capacity(1, 0, 1),
                ),
                now,
            )
            .unwrap();
    }

    let first = registry.select_and_reserve_at(&selection(), now).unwrap();
    assert_eq!(first.key.worker_id.as_str(), "worker-a");
    let excluded = first.key.clone();
    drop(first);

    let alternate = registry
        .select_and_reserve_excluding_at(&selection(), Some(&excluded), now)
        .unwrap();
    assert_eq!(alternate.key.worker_id.as_str(), "worker-b");
}

#[test]
fn circuit_requires_cooldown_and_post_open_status_then_grants_one_probe() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        status_ttl: Duration::from_secs(30),
        circuit_failure_threshold: 2,
        circuit_open_duration: Duration::from_secs(3),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    let key = WorkerInstanceKey::from_descriptor(&descriptor);
    registry.approve(approved).unwrap();
    let now = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            now,
        )
        .unwrap();

    registry
        .report_worker_transport_failure_at(&key, now + Duration::from_secs(1))
        .unwrap();
    let after_one_strike = registry
        .select_and_reserve_at(&selection(), now + Duration::from_secs(1))
        .unwrap();
    drop(after_one_strike);
    registry
        .report_worker_transport_failure_at(&key, now + Duration::from_secs(2))
        .unwrap();

    assert_eq!(
        registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
            .unwrap_err(),
        WorkerRegistryError::NoEligibleWorker
    );
    registry
        .observe_status_at(
            status(
                &descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            now + Duration::from_secs(3),
        )
        .unwrap();
    assert_eq!(
        registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(4))
            .unwrap_err(),
        WorkerRegistryError::NoEligibleWorker
    );

    let mut probe = registry
        .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
        .unwrap();
    assert_eq!(
        registry
            .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
            .unwrap_err(),
        WorkerRegistryError::NoEligibleWorker
    );
    probe.dispatch.mark_accepted().unwrap();
    drop(probe);
    let closed = registry
        .select_and_reserve_at(&selection(), now + Duration::from_secs(5))
        .unwrap();
    drop(closed);
}

#[test]
fn stale_circuit_outcomes_cannot_affect_replacement_incarnation() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        circuit_failure_threshold: 1,
        circuit_open_duration: Duration::from_secs(1),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let old = registration("worker-a", "inc-old", BackendKind::Cpu, 9101, 1);
    let old_key = WorkerInstanceKey::from_descriptor(&old.descriptor);
    registry.approve(old).unwrap();
    registry.report_worker_transport_failure(&old_key).unwrap();

    let replacement = registration("worker-a", "inc-new", BackendKind::Cpu, 9102, 1);
    let replacement_descriptor = replacement.descriptor.clone();
    registry.approve(replacement).unwrap();
    registry
        .observe_status(status(
            &replacement_descriptor,
            1,
            vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
            capacity(1, 0, 1),
        ))
        .unwrap();

    assert_eq!(
        registry.report_worker_reachable(&old_key).unwrap_err(),
        WorkerRegistryError::UnknownOrStaleIncarnation
    );
    assert_eq!(
        registry
            .report_worker_transport_failure(&old_key)
            .unwrap_err(),
        WorkerRegistryError::UnknownOrStaleIncarnation
    );
    let selected = registry.select_and_reserve(&selection()).unwrap();
    assert_eq!(selected.key.incarnation_id.as_str(), "inc-new");
}

#[test]
fn authenticated_reachable_outcome_resets_consecutive_strikes() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        circuit_failure_threshold: 2,
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    let key = WorkerInstanceKey::from_descriptor(&descriptor);
    registry.approve(approved).unwrap();
    registry
        .observe_status(status(
            &descriptor,
            1,
            vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
            capacity(1, 0, 1),
        ))
        .unwrap();

    registry.report_worker_transport_failure(&key).unwrap();
    registry.report_worker_reachable(&key).unwrap();
    registry.report_worker_transport_failure(&key).unwrap();

    let selected = registry.select_and_reserve(&selection()).unwrap();
    assert_eq!(selected.key, key);
}

#[test]
fn abandoned_half_open_probe_reopens_the_circuit() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        circuit_failure_threshold: 1,
        circuit_open_duration: Duration::from_secs(1),
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    let key = WorkerInstanceKey::from_descriptor(&descriptor);
    registry.approve(approved).unwrap();
    let now = Instant::now();
    registry
        .observe_status_at(
            status(
                &descriptor,
                1,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            now,
        )
        .unwrap();
    registry
        .report_worker_transport_failure_at(&key, now + Duration::from_secs(1))
        .unwrap();
    registry
        .observe_status_at(
            status(
                &descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 0, 1),
            ),
            now + Duration::from_secs(2),
        )
        .unwrap();

    let probe = registry
        .select_and_reserve_at(&selection(), now + Duration::from_secs(3))
        .unwrap();
    drop(probe);

    let inner = lock_recover(&registry.inner);
    assert!(matches!(
        inner.workers.get(&key).expect("registered worker").circuit,
        WorkerCircuitState::Open { .. }
    ));
}

#[test]
fn draining_saturated_or_future_observations_are_ineligible() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    let approved = registration("worker-a", "inc-a", BackendKind::Cpu, 9101, 1);
    let descriptor = approved.descriptor.clone();
    registry.approve(approved).unwrap();
    let now = Instant::now();
    let mut draining = status(
        &descriptor,
        1,
        vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
        capacity(1, 0, 1),
    );
    draining.process_state = WorkerProcessState::Draining;
    registry.observe_status_at(draining, now).unwrap();
    assert!(registry.select_and_reserve_at(&selection(), now).is_err());

    registry
        .observe_status_at(
            status(
                &descriptor,
                2,
                vec![deployment("chat-prod", "lfm2", BackendKind::Cpu)],
                capacity(1, 1, 0),
            ),
            now + Duration::from_secs(1),
        )
        .unwrap();
    assert!(registry
        .select_and_reserve_at(&selection(), now + Duration::from_secs(1))
        .is_err());
    assert!(registry.select_and_reserve_at(&selection(), now).is_err());
}

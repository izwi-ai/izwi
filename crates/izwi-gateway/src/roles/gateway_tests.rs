//! The gateway role's configuration, worker adoption, serving and shutdown.

use super::*;
use clap::Parser;
use izwi_model_catalog::ModelVariant;
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    DeploymentId, IncarnationId, ModelAlias, ModelGeneration, NdjsonLimits, NodeId, TaskKind,
    WorkerDescriptor, WorkerId, WorkerStatus,
};
use izwi_serving_worker::test_support::{TestWorker, TestWorkerConfig};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;

use crate::cli::GatewayCli;
use crate::config::GatewayServeConfig;
use crate::roles::gateway_workers::{
    approve_gateway_worker, configured_worker_capacity, initial_gateway_worker_expectation,
    jittered_poll_interval, poll_jitter_seed, refresh_gateway_worker_status,
    validate_gateway_worker_endpoint_policy, GatewayWorkerExpectation,
};
use crate::test_support::env_lock;

/// The gateway flags in `argv`.
fn parse(argv: &[&str]) -> GatewayArgs {
    GatewayCli::try_parse_from(argv)
        .expect("arguments should parse")
        .gateway
}

/// A gateway role over `args` with the default listener, any worker
/// backend and the strict compatibility profile.
fn test_config(args: &GatewayArgs) -> GatewayConfig {
    GatewayConfig {
        args: args.clone(),
        serve: GatewayServeConfig::default(),
        backend: None,
        compat_profile: crate::api::openai::compat::OpenAiCompatibilityProfile::Strict,
    }
}

#[test]
fn jittered_poll_interval_is_deterministic_and_bounded() {
    let base = Duration::from_millis(2_000);
    for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX, 123_456_789] {
        let first = jittered_poll_interval(base, seed);
        let second = jittered_poll_interval(base, seed);
        assert_eq!(first, second, "seed {seed} must be deterministic");
        assert!(
            first >= Duration::from_millis(1_800) && first <= Duration::from_millis(2_200),
            "seed {seed} cadence {first:?} outside ±10% of the configured interval"
        );
    }
}

#[test]
fn jittered_poll_interval_never_collapses_and_desynchronizes_workers() {
    assert_eq!(
        jittered_poll_interval(Duration::from_millis(1), 7),
        Duration::from_millis(1),
        "sub-millisecond jittered cadences must clamp to 1ms, not a zero tokio interval"
    );
    let cadences: Vec<u64> = (0u64..16)
        .map(|seed| {
            let worker_id =
                WorkerId::new(format!("worker-{seed}").as_str()).expect("static worker id");
            let node_id = NodeId::new("node-a").expect("static node id");
            jittered_poll_interval(
                Duration::from_millis(2_000),
                poll_jitter_seed(&worker_id, &node_id),
            )
            .as_millis() as u64
        })
        .collect();
    let distinct: std::collections::BTreeSet<u64> = cadences.iter().copied().collect();
    assert!(
        distinct.len() >= 3,
        "co-configured workers must spread across at least 3 distinct cadences, got {distinct:?}"
    );
}

#[test]
fn status_cadence_validation_leaves_room_for_jitter() {
    validate_status_cadence(Duration::from_millis(10_000), Duration::from_millis(2_000))
        .expect("production defaults leave ample jitter headroom");
    validate_status_cadence(Duration::from_millis(10_000), Duration::from_millis(9_090))
        .expect("widest jittered cadence 9999ms stays below the 10s TTL");
    assert!(
        validate_status_cadence(Duration::from_millis(10_000), Duration::from_millis(0)).is_err()
    );
    assert!(
        validate_status_cadence(Duration::from_millis(10_000), Duration::from_millis(9_500))
            .is_err(),
        "a jittered 10450ms cadence would flap a 10s TTL expired"
    );
}

#[test]
fn fleet_topology_requires_versioned_identity_mtls_and_https() {
    const TEST_CERT_PEM: &[u8] =
        b"-----BEGIN CERTIFICATE-----\nMAECAQ==\n-----END CERTIFICATE-----\n";
    const TEST_KEY_PEM: &[u8] =
        b"-----BEGIN PRIVATE KEY-----\nMAECAQ==\n-----END PRIVATE KEY-----\n";
    let fleet: gateway_deployments::GatewayWorkerApproval =
        "v1|https://worker.example.test:9470|node-a|worker-a|chat|chat-model|chat-prod|7"
            .parse()
            .expect("versioned fleet approval");
    let legacy: gateway_deployments::GatewayWorkerApproval =
        "http://127.0.0.1:9470|chat|chat-model|chat-prod|7"
            .parse()
            .expect("standalone compatibility approval");
    let empty_tls = izwi_serving_client::WorkerClientTlsConfig::default();
    assert!(validate_gateway_topology_policy(
        GatewayTopology::FleetOneGateway,
        std::slice::from_ref(&fleet),
        &empty_tls,
    )
    .unwrap_err()
    .to_string()
    .contains("mutual TLS"));
    let ca_only_tls = izwi_serving_client::WorkerClientTlsConfig::from_pem(
        vec![TEST_CERT_PEM.to_vec()],
        None,
        None,
    )
    .expect("bounded test CA");
    assert!(validate_gateway_topology_policy(
        GatewayTopology::FleetOneGateway,
        std::slice::from_ref(&fleet),
        &ca_only_tls,
    )
    .unwrap_err()
    .to_string()
    .contains("mutual TLS"));

    let identity_tls = izwi_serving_client::WorkerClientTlsConfig::from_pem(
        Vec::new(),
        Some(TEST_CERT_PEM.to_vec()),
        Some(TEST_KEY_PEM.to_vec()),
    )
    .expect("bounded test identity");
    validate_gateway_topology_policy(
        GatewayTopology::FleetOneGateway,
        std::slice::from_ref(&fleet),
        &identity_tls,
    )
    .expect("versioned fleet policy");
    let mut duplicate = fleet.clone();
    duplicate.endpoint = "https://worker-b.example.test:9470".to_string();
    assert!(validate_gateway_topology_policy(
        GatewayTopology::FleetOneGateway,
        &[fleet.clone(), duplicate],
        &identity_tls,
    )
    .unwrap_err()
    .to_string()
    .contains("duplicate operator-approved"));
    assert!(validate_gateway_topology_policy(
        GatewayTopology::FleetOneGateway,
        std::slice::from_ref(&legacy),
        &identity_tls,
    )
    .unwrap_err()
    .to_string()
    .contains("versioned v1 approvals"));
    validate_gateway_worker_endpoint_policy(GatewayTopology::FleetOneGateway, true, false)
        .expect("HTTPS fleet endpoint");
    assert!(
        validate_gateway_worker_endpoint_policy(GatewayTopology::FleetOneGateway, false, true,)
            .is_err()
    );
    validate_gateway_worker_endpoint_policy(GatewayTopology::Standalone, false, true)
        .expect("standalone loopback compatibility");
    assert!(
        validate_gateway_worker_endpoint_policy(GatewayTopology::Standalone, true, false,).is_err()
    );

    let pinned_fleet = parse(&[
        "izwi-gateway",
        "--gateway-topology",
        "fleet-one-gateway",
        "--worker-endpoint",
        "https://worker.example.test:9470",
    ]);
    assert!(validate_gateway_topology_source(&pinned_fleet)
        .unwrap_err()
        .to_string()
        .contains("pinned and legacy"));
}

#[test]
fn gateway_topology_parses_from_cli_and_environment() {
    let cli = parse(&["izwi-gateway", "--gateway-topology", "fleet-one-gateway"]);
    assert_eq!(cli.gateway_topology, GatewayTopology::FleetOneGateway);

    let _guard = env_lock();
    std::env::set_var("IZWI_GATEWAY_TOPOLOGY", "fleet-one-gateway");
    let from_env = parse(&["izwi-gateway"]);
    std::env::remove_var("IZWI_GATEWAY_TOPOLOGY");
    assert_eq!(from_env.gateway_topology, GatewayTopology::FleetOneGateway);
}

#[test]
fn gateway_configuration_builds_without_a_runtime_service() {
    let args = parse(&[
        "izwi-gateway",
        "--worker-endpoint",
        "http://127.0.0.1:19091",
        "--worker-credential-id",
        "gateway-test-credential",
        "--worker-bearer-token",
        "gateway-test-secret",
        "--worker-incarnation",
        "worker-incarnation-1",
        "--worker-deployment",
        "qwen3-chat-v1",
        "--public-model",
        ModelVariant::Qwen354BGguf.dir_name(),
        "--worker-model-generation",
        "7",
    ]);
    // `--worker-endpoint` is the one-worker form of the legacy endpoint
    // list: one registry approval, discovered from the worker.
    let public_model =
        ModelAlias::new(ModelVariant::Qwen354BGguf.dir_name()).expect("static model alias");
    let (cli_approvals, approvals, shared_view) = {
        let _guard = env_lock();
        configured_gateway_worker_approvals(&args, &public_model)
            .expect("legacy worker endpoint should become an approval")
    };
    assert!(shared_view.is_none());
    assert_eq!(cli_approvals, approvals);
    assert_eq!(
        approvals,
        vec![gateway_deployments::GatewayWorkerApproval {
            endpoint: "http://127.0.0.1:19091".into(),
            identity:
                gateway_deployments::GatewayWorkerApprovalIdentity::DiscoverFromAuthenticatedEndpoint,
            task: TaskKind::Chat,
            public_model,
            deployment_id: DeploymentId::new("qwen3-chat-v1").expect("static deployment"),
            model_generation: ModelGeneration::new(7).expect("non-zero generation"),
        }]
    );
    assert_eq!(args.gateway_worker_queue_wait_ms, 250);
    assert_eq!(args.gateway_slow_consumer_timeout_ms, 5000);

    let client_config =
        gateway_worker_client_config(&args, izwi_serving_client::WorkerClientTlsConfig::default());
    assert_eq!(client_config.request_timeout, Duration::from_secs(10));
    assert_eq!(client_config.first_output_timeout, Duration::from_secs(60));
    assert_eq!(
        client_config.first_output_timeout_per_kib,
        Duration::from_secs(1)
    );
    assert_eq!(client_config.progress_timeout, Duration::from_secs(30));
    assert_eq!(
        client_config.ndjson_limits,
        NdjsonLimits {
            max_line_bytes: 1024 * 1024,
            max_total_bytes: 16 * 1024 * 1024,
            max_events: 8192,
            ..NdjsonLimits::default()
        }
    );
}

#[test]
fn gateway_stream_deadlines_are_configurable_and_bounded() {
    let args = parse(&[
        "izwi-gateway",
        "--gateway-worker-queue-wait-ms",
        "500",
        "--gateway-worker-admission-timeout-ms",
        "1500",
        "--gateway-worker-first-output-timeout-ms",
        "2500",
        "--gateway-worker-progress-idle-timeout-ms",
        "1200",
        "--gateway-slow-consumer-timeout-ms",
        "750",
    ]);
    validate_gateway_limits(&args).expect("bounded stream deadlines should validate");
    let config =
        gateway_worker_client_config(&args, izwi_serving_client::WorkerClientTlsConfig::default());
    assert_eq!(config.request_timeout, Duration::from_millis(1500));
    assert_eq!(config.first_output_timeout, Duration::from_millis(2500));
    assert_eq!(config.progress_timeout, Duration::from_millis(1200));

    for (flag, value) in [
        ("--gateway-worker-admission-timeout-ms", "60001"),
        ("--gateway-worker-first-output-timeout-ms", "3600001"),
        (
            "--gateway-worker-first-output-timeout-per-kib-ms",
            "3600001",
        ),
        ("--gateway-worker-progress-idle-timeout-ms", "3600001"),
        ("--gateway-slow-consumer-timeout-ms", "60001"),
    ] {
        let invalid = parse(&["izwi-gateway", flag, value]);
        assert!(
            validate_gateway_limits(&invalid).is_err(),
            "{flag} must be bounded"
        );
    }

    let invalid_queue = parse(&[
        "izwi-gateway",
        "--gateway-worker-queue-wait-ms",
        "10001",
        "--gateway-worker-admission-timeout-ms",
        "10000",
    ]);
    assert!(validate_gateway_limits(&invalid_queue).is_err());
}

#[test]
fn legacy_worker_endpoint_requires_its_deployment_and_generation() {
    let public_model =
        ModelAlias::new(ModelVariant::Qwen354BGguf.dir_name()).expect("static model alias");
    let _guard = env_lock();
    let args = parse(&[
        "izwi-gateway",
        "--worker-endpoint",
        "http://127.0.0.1:19091",
    ]);
    let error = configured_gateway_worker_approvals(&args, &public_model)
        .expect_err("gateway configuration must be complete");
    assert!(error.to_string().contains("--worker-deployment"));

    let args = parse(&[
        "izwi-gateway",
        "--worker-endpoint",
        "http://127.0.0.1:19091",
        "--worker-deployment",
        "qwen3-chat-v1",
    ]);
    let error = configured_gateway_worker_approvals(&args, &public_model)
        .expect_err("gateway configuration must be complete");
    assert!(error.to_string().contains("--worker-model-generation"));
}

#[tokio::test]
async fn worker_endpoint_and_incarnation_are_validated_before_any_worker_io() {
    let perimeter = || {
        GatewayPerimeterConfig::new_for_test("registry-test-api-key", 1024 * 1024)
            .expect("test perimeter should be valid")
    };
    for (flags, expected) in [
        (
            &[
                "--worker-endpoint",
                "http://127.0.0.1:19091",
                "--gateway-worker-endpoint",
                "http://127.0.0.1:19092",
            ][..],
            "cannot be combined",
        ),
        (
            &["--worker-incarnation", "worker-incarnation-1"][..],
            "applies only to the legacy --worker-endpoint form",
        ),
        (&[][..], "gateway mode requires a worker"),
    ] {
        let mut argv = vec![
            "izwi-gateway",
            "--public-model",
            ModelVariant::Qwen354BGguf.dir_name(),
        ];
        argv.extend_from_slice(flags);
        let args = parse(&argv);
        let error = {
            let _guard = env_lock();
            gateway_state(&test_config(&args), EnterpriseHooks::noop(), perimeter())
                .await
                .err()
                .expect("invalid worker configuration must fail")
        };
        assert!(error.to_string().contains(expected), "{flags:?}: {error}");
    }
}

#[test]
fn registry_gateway_endpoint_list_is_bounded() {
    let parsed = parse(&[
        "izwi-gateway",
        "--gateway-worker-endpoint",
        "http://127.0.0.1:19091",
        "--gateway-worker-endpoint",
        "http://127.0.0.1:19092",
    ]);
    assert_eq!(
        parsed.gateway_worker_endpoints,
        vec![
            "http://127.0.0.1:19091".to_string(),
            "http://127.0.0.1:19092".to_string()
        ]
    );

    let mut args = parse(&["izwi-gateway"]);
    args.gateway_worker_endpoints = (0..=MAX_CONFIGURED_GATEWAY_WORKERS)
        .map(|index| format!("http://127.0.0.1:{}", 20_000 + index))
        .collect();

    let error = validate_gateway_limits(&args).expect_err("worker list must be bounded");
    assert!(error
        .to_string()
        .contains("gateway worker endpoints or approvals"));
}

#[tokio::test]
async fn registry_gateway_configuration_approves_worker_without_runtime() {
    let model = ModelVariant::Qwen354BGguf;
    let public_model = ModelAlias::new(model.dir_name()).expect("static model alias");
    let first = TestWorker::spawn(TestWorkerConfig {
        worker_id: WorkerId::new("gateway-worker-a").expect("static worker id"),
        node_id: NodeId::new("gateway-node-a").expect("static node id"),
        incarnation_id: IncarnationId::new("gateway-incarnation-a").expect("static incarnation"),
        public_model: public_model.clone(),
        ..TestWorkerConfig::default()
    })
    .await
    .expect("test worker should bind");
    let credentials = first.config().credentials.clone();
    let mut args = parse(&[
        "izwi-gateway",
        "--worker-credential-id",
        credentials.credential_id.as_str(),
        "--worker-bearer-token",
        credentials.bearer_token.expose_secret(),
        "--worker-deployment",
        first.config().deployment_id.as_str(),
        "--public-model",
        model.dir_name(),
        "--worker-model-generation",
        "1",
    ]);
    args.gateway_worker_endpoints = vec![first.endpoint()];

    let (state, poller) = gateway_state(
        &test_config(&args),
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("registry-test-api-key", 1024 * 1024)
            .expect("test perimeter should be valid"),
    )
    .await
    .expect("registry gateway configuration should build");
    assert_eq!(poller.poller_count, 1);
    assert!(state.chat_dispatchers[0].readiness_check().is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gateway_shutdown_serves_through_prestop_and_ends_streams_with_a_terminal_event() {
    let model = ModelVariant::Qwen354BGguf;
    let worker = TestWorker::spawn(TestWorkerConfig {
        public_model: ModelAlias::new(model.dir_name()).unwrap(),
        script: izwi_serving_worker::test_support::WorkerScript::Hang,
        max_active_invocations: 4,
        cancellation_delay: Duration::from_millis(10),
        ..TestWorkerConfig::default()
    })
    .await
    .expect("test worker should bind");
    let dispatcher = crate::test_support::single_worker_dispatcher(
        model,
        &worker.endpoint(),
        worker.config().credentials.clone(),
    )
    .await;
    let api_key = "shutdown-test-api-key";
    let state = gateway::GatewayState::new(
        dispatcher,
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test(api_key, 1024 * 1024).unwrap(),
        30,
        4,
    );
    state.mark_ready();
    let app = gateway::create_gateway_router(state.clone(), &GatewayServeConfig::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (signal_tx, signal_rx) = oneshot::channel::<()>();
    let plan = ShutdownPlan {
        prestop: Duration::from_millis(400),
        budget: Duration::from_millis(1600),
    };
    let server = tokio::spawn(serve_gateway(
        listener,
        app,
        state,
        async move {
            let _ = signal_rx.await;
        },
        plan,
    ));

    let client = reqwest::Client::new();
    let stream = || {
        client
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(api_key)
            .json(&serde_json::json!({
                "model": model.dir_name(),
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_tokens": 8
            }))
            .send()
    };
    let open = stream().await.unwrap();
    assert_eq!(open.status(), reqwest::StatusCode::OK);

    signal_tx.send(()).unwrap();
    let signalled = Instant::now();
    tokio::time::sleep(Duration::from_millis(100)).await;
    // Pre-stop: unready, yet a new request is still served.
    let ready = client.get(format!("{base}/readyz")).send().await.unwrap();
    assert_eq!(ready.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    let during_prestop = stream().await.unwrap();
    assert_eq!(during_prestop.status(), reqwest::StatusCode::OK);

    // Both streams outlive the drain's notice point and end with a
    // terminal error event, not a cut connection.
    for response in [open, during_prestop] {
        let body = tokio::time::timeout(Duration::from_secs(3), response.text())
            .await
            .expect("the stream ends within the shutdown budget")
            .expect("the stream ends cleanly");
        assert!(body.contains("service_unavailable"), "{body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    }
    tokio::time::timeout(Duration::from_secs(3), server)
        .await
        .expect("the gateway stops within its budget")
        .unwrap()
        .unwrap();
    let elapsed = signalled.elapsed();
    assert!(
        elapsed >= plan.stream_notice() + plan.prestop
            && elapsed < plan.budget + Duration::from_millis(500),
        "{elapsed:?}"
    );
}

#[tokio::test]
async fn gateway_serves_every_approved_chat_model_primary_first() {
    let models = [ModelVariant::Qwen359BGguf, ModelVariant::Qwen354BGguf];
    let mut workers = Vec::new();
    for (index, model) in models.into_iter().enumerate() {
        workers.push(
            TestWorker::spawn(TestWorkerConfig {
                worker_id: WorkerId::new(format!("multi-model-worker-{index}")).unwrap(),
                incarnation_id: IncarnationId::new(format!("multi-model-inc-{index}")).unwrap(),
                deployment_id: DeploymentId::new(format!("multi-model-chat-{index}")).unwrap(),
                public_model: ModelAlias::new(model.dir_name()).unwrap(),
                ..TestWorkerConfig::default()
            })
            .await
            .expect("test worker should bind"),
        );
    }
    let credentials = workers[0].config().credentials.clone();
    let approvals: Vec<String> = workers
        .iter()
        .zip(models)
        .map(|(worker, model)| {
            format!(
                "{}|chat|{}|{}|1",
                worker.endpoint(),
                model.dir_name(),
                worker.config().deployment_id
            )
        })
        .collect();
    let env = crate::test_support::env_lock();
    // The second approval names the primary model.
    let args = parse(&[
        "izwi-gateway",
        "--worker-credential-id",
        credentials.credential_id.as_str(),
        "--worker-bearer-token",
        credentials.bearer_token.expose_secret(),
        "--public-model",
        models[1].dir_name(),
        "--gateway-worker-approval",
        &approvals[0],
        "--gateway-worker-approval",
        &approvals[1],
    ]);
    drop(env);
    let (state, _poller) = gateway_state(
        &test_config(&args),
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("registry-test-api-key", 1024 * 1024)
            .expect("test perimeter should be valid"),
    )
    .await
    .expect("multi-model gateway configuration should build");
    let served: Vec<ModelVariant> = state
        .chat_dispatchers
        .iter()
        .map(|dispatcher| dispatcher.public_model())
        .collect();
    assert_eq!(served, vec![models[1], models[0]]);
    assert!(state
        .chat_dispatchers
        .iter()
        .all(|dispatcher| dispatcher.readiness_check().is_ok()));
}

#[tokio::test]
async fn registry_gateway_boots_past_an_unavailable_worker_and_adopts_it_later() {
    let model = ModelVariant::Qwen354BGguf;
    let public_model = ModelAlias::new(model.dir_name()).expect("static model alias");
    let worker_config = |name: &str| TestWorkerConfig {
        worker_id: WorkerId::new(format!("gateway-worker-{name}")).expect("static worker id"),
        node_id: NodeId::new("gateway-node-a").expect("static node id"),
        incarnation_id: IncarnationId::new(format!("gateway-incarnation-{name}"))
            .expect("static incarnation"),
        public_model: public_model.clone(),
        ..TestWorkerConfig::default()
    };
    let first = TestWorker::spawn(worker_config("a"))
        .await
        .expect("test worker should bind");
    // The second approved worker is down at boot: nothing listens yet.
    let late_address = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("probe binds")
        .local_addr()
        .expect("probe address");
    let credentials = first.config().credentials.clone();
    // CLI defaults come from the environment, which other tests set
    // under this lock (IZWI_GATEWAY_TOPOLOGY, for one).
    let env = crate::test_support::env_lock();
    let mut args = parse(&[
        "izwi-gateway",
        "--worker-credential-id",
        credentials.credential_id.as_str(),
        "--worker-bearer-token",
        credentials.bearer_token.expose_secret(),
        "--worker-deployment",
        first.config().deployment_id.as_str(),
        "--public-model",
        model.dir_name(),
        "--worker-model-generation",
        "1",
    ]);
    drop(env);
    args.gateway_worker_endpoints = vec![first.endpoint(), format!("http://{late_address}/")];
    args.gateway_worker_status_poll_ms = 50;

    let (state, poller) = gateway_state(
        &test_config(&args),
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("registry-test-api-key", 1024 * 1024)
            .expect("test perimeter should be valid"),
    )
    .await
    .expect("one unavailable approved worker must not fail gateway boot");
    assert_eq!(poller.poller_count, 1);
    assert!(state.chat_dispatchers[0].readiness_check().is_ok());
    let dispatcher = &state.chat_dispatchers[0];
    let deployment_id = first.config().deployment_id.clone();
    assert_eq!(
        dispatcher
            .registry()
            .deployment_readiness(&deployment_id)
            .approved_workers,
        1
    );

    // Once the worker comes up, the gateway adopts it without a restart.
    let _late = TestWorker::spawn_at(worker_config("b"), late_address)
        .await
        .expect("late test worker should bind its approved address");
    let adopted = tokio::time::timeout(Duration::from_secs(10), async {
        while dispatcher
            .registry()
            .deployment_readiness(&deployment_id)
            .ready_workers
            < 2
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        adopted.is_ok(),
        "the gateway must adopt an approved worker that becomes available after boot"
    );
    drop(poller);
}

#[tokio::test]
async fn registry_gateway_boot_rejects_two_endpoints_for_one_logical_worker() {
    let model = ModelVariant::Qwen354BGguf;
    let config = TestWorkerConfig {
        worker_id: WorkerId::new("gateway-worker-a").expect("static worker id"),
        node_id: NodeId::new("gateway-node-a").expect("static node id"),
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        ..TestWorkerConfig::default()
    };
    let first = TestWorker::spawn(config.clone())
        .await
        .expect("test worker binds");
    let second = TestWorker::spawn(config).await.expect("test worker binds");
    let credentials = first.config().credentials.clone();
    let mut args = parse(&[
        "izwi-gateway",
        "--worker-credential-id",
        credentials.credential_id.as_str(),
        "--worker-bearer-token",
        credentials.bearer_token.expose_secret(),
        "--worker-deployment",
        first.config().deployment_id.as_str(),
        "--public-model",
        model.dir_name(),
        "--worker-model-generation",
        "1",
    ]);
    args.gateway_worker_endpoints = vec![first.endpoint(), second.endpoint()];
    let error = gateway_state(
        &test_config(&args),
        EnterpriseHooks::noop(),
        GatewayPerimeterConfig::new_for_test("registry-test-api-key", 1024 * 1024)
            .expect("test perimeter should be valid"),
    )
    .await
    .err()
    .expect("a duplicated logical worker is a configuration error");
    assert!(
        error.to_string().contains("distinct logical workers"),
        "{error:#}"
    );
}

#[tokio::test]
async fn restarted_worker_reapproval_pins_stable_identity_and_deployment() {
    let model = ModelVariant::Qwen354BGguf;
    let worker = TestWorker::spawn(TestWorkerConfig {
        worker_id: WorkerId::new("restart-worker").expect("static worker id"),
        node_id: NodeId::new("restart-node").expect("static node id"),
        incarnation_id: IncarnationId::new("restart-incarnation-1").expect("static incarnation"),
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        ..TestWorkerConfig::default()
    })
    .await
    .expect("test worker should bind");
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig::default(),
    )
    .expect("client should initialize");
    let descriptor = client.descriptor().await.expect("descriptor should load");
    let status = client.status().await.expect("status should load");
    let deployment_id = worker.config().deployment_id.clone();
    let approval = gateway_deployments::GatewayWorkerApproval {
        endpoint: worker.endpoint(),
        identity: gateway_deployments::GatewayWorkerApprovalIdentity::V1 {
            node_id: worker.config().node_id.clone(),
            worker_id: worker.config().worker_id.clone(),
        },
        task: TaskKind::Chat,
        public_model: ModelAlias::new(model.dir_name()).expect("static model alias"),
        deployment_id,
        model_generation: worker.config().model_generation,
    };
    let mut wrong_identity = approval.clone();
    wrong_identity.identity = gateway_deployments::GatewayWorkerApprovalIdentity::V1 {
        node_id: NodeId::new("unapproved-node").expect("static node id"),
        worker_id: worker.config().worker_id.clone(),
    };
    assert!(initial_gateway_worker_expectation(
        client.clone(),
        &descriptor,
        &status,
        &wrong_identity,
    )
    .err()
    .expect("unapproved node must fail")
    .to_string()
    .contains("operator-approved"));
    wrong_identity.identity = gateway_deployments::GatewayWorkerApprovalIdentity::V1 {
        node_id: worker.config().node_id.clone(),
        worker_id: WorkerId::new("unapproved-worker").expect("static worker id"),
    };
    assert!(initial_gateway_worker_expectation(
        client.clone(),
        &descriptor,
        &status,
        &wrong_identity,
    )
    .err()
    .expect("unapproved worker must fail")
    .to_string()
    .contains("operator-approved"));
    let expected = initial_gateway_worker_expectation(client, &descriptor, &status, &approval)
        .expect("initial worker should match configuration");
    let registry =
        worker_registry::WorkerRegistry::new(worker_registry::WorkerRegistryConfig::default())
            .expect("registry should initialize");
    approve_gateway_worker(&registry, &expected, descriptor, status.clone())
        .expect("initial worker should be approved");

    let next_incarnation =
        IncarnationId::new("restart-incarnation-2").expect("static replacement incarnation");
    let mut descriptor = expected
        .client
        .descriptor()
        .await
        .expect("descriptor should load");
    descriptor.incarnation_id = next_incarnation.clone();
    let mut restarted_status = status.clone();
    restarted_status.incarnation_id = next_incarnation;
    restarted_status.status_sequence = 1;
    approve_gateway_worker(
        &registry,
        &expected,
        descriptor.clone(),
        restarted_status.clone(),
    )
    .expect("matching replacement incarnation should be approved");
    assert_eq!(
        registry
            .observe_status(status)
            .expect_err("old incarnation must be fenced"),
        worker_registry::WorkerRegistryError::UnknownOrStaleIncarnation
    );

    descriptor.node_id = NodeId::new("unapproved-node").expect("static node id");
    restarted_status.node_id = descriptor.node_id.clone();
    assert!(
        approve_gateway_worker(&registry, &expected, descriptor, restarted_status)
            .expect_err("replacement node identity must stay pinned")
            .to_string()
            .contains("logical worker or node identity")
    );

    let mut descriptor = expected
        .client
        .descriptor()
        .await
        .expect("descriptor should load");
    descriptor.incarnation_id =
        IncarnationId::new("restart-incarnation-3").expect("static replacement incarnation");
    let mut changed_capacity = expected.client.status().await.expect("status should load");
    changed_capacity.incarnation_id = descriptor.incarnation_id.clone();
    changed_capacity.capacity.max_active_invocations += 1;
    assert!(
        approve_gateway_worker(&registry, &expected, descriptor, changed_capacity)
            .expect_err("replacement capacity must stay pinned")
            .to_string()
            .contains("validated capacity")
    );

    let mut descriptor = expected
        .client
        .descriptor()
        .await
        .expect("descriptor should load");
    descriptor.incarnation_id =
        IncarnationId::new("restart-incarnation-4").expect("static replacement incarnation");
    let mut changed_deployment = expected.client.status().await.expect("status should load");
    changed_deployment.incarnation_id = descriptor.incarnation_id.clone();
    changed_deployment.deployments[0].model_generation =
        ModelGeneration::new(2).expect("non-zero changed generation");
    assert!(
        approve_gateway_worker(&registry, &expected, descriptor, changed_deployment)
            .expect_err("replacement deployment must stay pinned")
            .to_string()
            .contains("deployment contract")
    );
}

async fn reconnect_expectation(
    config: izwi_serving_worker::test_support::TestWorkerConfig,
) -> (
    izwi_serving_worker::test_support::TestWorker,
    GatewayWorkerExpectation,
    WorkerDescriptor,
    WorkerStatus,
) {
    use izwi_serving_client::{WorkerClient, WorkerClientConfig};

    let worker = TestWorker::spawn(config).await.expect("test worker binds");
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig::default(),
    )
    .expect("client initializes");
    let descriptor = client.descriptor().await.expect("descriptor");
    let status = client.status().await.expect("status");
    let deployment = worker_registry::ApprovedDeployment::from_loaded(
        status.deployments.first().expect("deployment"),
    );
    let validated_capacity = configured_worker_capacity(&status).expect("valid capacity");
    let expected = GatewayWorkerExpectation {
        client,
        worker_id: descriptor.worker_id.clone(),
        node_id: descriptor.node_id.clone(),
        deployment,
        validated_capacity,
    };
    (worker, expected, descriptor, status)
}

#[tokio::test]
async fn status_refresh_never_waits_on_a_hung_fleet_store() {
    use izwi_serving_worker::test_support::TestWorkerConfig;

    let registry =
        worker_registry::WorkerRegistry::new(worker_registry::WorkerRegistryConfig::default())
            .expect("registry initializes");
    let (_worker, expected, descriptor, status) =
        reconnect_expectation(TestWorkerConfig::default()).await;
    approve_gateway_worker(&registry, &expected, descriptor, status.clone())
        .expect("initial approval");

    let root = tempfile::tempdir().expect("temp dir");
    let mut store = crate::gateway_fleet::FleetStore::new(crate::db::StoreDatabase::new(
        root.path().join("fleet.sqlite3"),
    ));
    let faults = Arc::new(crate::gateway_fleet::FleetFaults::default());
    store.set_faults_for_test(Arc::clone(&faults));
    faults.hang_fleet_calls();
    let coordinator = Arc::new(app::fleet_coordinator::FleetCoordinator::new(
        store,
        "gateway-test".to_string(),
    ));
    let publisher = coordinator.spawn_status_publisher();

    // A coordination store that stops answering must not delay the
    // observation that keeps the worker fresh: every poll still lands.
    for _ in 0..3 {
        let refreshed = tokio::time::timeout(
            Duration::from_secs(2),
            refresh_gateway_worker_status(&registry, &expected, Some(&publisher)),
        )
        .await
        .expect("a hung coordination store must not stall the status poll")
        .expect("status refresh succeeds");
        assert!(refreshed.status_sequence > status.status_sequence);
    }
    assert!(
        registry.observe_status(status).is_err(),
        "the registry recorded the newer polled statuses"
    );
    faults.answer_fleet_calls();
}

#[tokio::test]
async fn restarted_worker_reconnects_with_same_generation_and_fails_closed_on_drift() {
    use izwi_serving_protocol::{IncarnationId, ModelGeneration, WorkerId};
    use izwi_serving_worker::test_support::TestWorkerConfig;

    let registry =
        worker_registry::WorkerRegistry::new(worker_registry::WorkerRegistryConfig::default())
            .expect("registry initializes");
    let base = TestWorkerConfig {
        worker_id: WorkerId::new("test-worker-1").expect("test id"),
        node_id: NodeId::new("test-node-1").expect("test id"),
        ..TestWorkerConfig::default()
    };

    // Generation 1, incarnation 1: initial approval.
    let (worker_a, expected_a, descriptor_a, status_a) = reconnect_expectation(TestWorkerConfig {
        incarnation_id: IncarnationId::new("test-incarnation-1").expect("test id"),
        ..base.clone()
    })
    .await;
    approve_gateway_worker(&registry, &expected_a, descriptor_a, status_a.clone())
        .expect("initial approval");

    // The worker restarts with the same generation but a new incarnation.
    // The poller's refresh path re-approves it without operator action.
    drop(worker_a);
    let (_worker_b, expected_b, _, _) = reconnect_expectation(TestWorkerConfig {
        incarnation_id: IncarnationId::new("test-incarnation-2").expect("test id"),
        ..base.clone()
    })
    .await;
    refresh_gateway_worker_status(&registry, &expected_b, None)
        .await
        .expect("same-generation restart must reconnect");
    let expected_b_deployment = expected_b.deployment.clone();
    assert!(
        registry.observe_status(status_a).is_err(),
        "the dead incarnation's late statuses must stay fenced after replacement"
    );

    // A restart that changes the model generation fails closed: the
    // expectation pins the operator-approved generation, so a worker
    // advertising a new generation is never silently adopted.
    let (_worker_c, mut expected_c, _, _) = reconnect_expectation(TestWorkerConfig {
        incarnation_id: IncarnationId::new("test-incarnation-3").expect("test id"),
        model_generation: ModelGeneration::new(2).expect("non-zero"),
        ..base
    })
    .await;
    expected_c.deployment = expected_b_deployment;
    assert!(
        refresh_gateway_worker_status(&registry, &expected_c, None)
            .await
            .is_err(),
        "generation drift must fail closed instead of silently adopting"
    );
}

#![cfg(feature = "mock-worker")]

use izwi_serving_client::{
    mock::{MockFault, MockWorker, MockWorkerConfig},
    DeadlinePhase, WorkerClient, WorkerClientConfig, WorkerClientError,
};
use izwi_serving_protocol::*;
use std::{collections::BTreeSet, time::Duration};
use tokio::net::TcpListener;

fn id<T: TryFrom<String>>(prefix: &str, suffix: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(format!("{prefix}-{suffix}")).expect("test identity")
}

fn request(config: &MockWorkerConfig, suffix: &str) -> InvocationRequest {
    InvocationRequest {
        schema_version: PROTOCOL_V1,
        request_id: id("request", suffix),
        attempt_id: id("attempt", suffix),
        expected_worker_incarnation: config.incarnation_id.clone(),
        deployment_id: config.deployment_id.clone(),
        expected_model_generation: config.model_generation,
        caller: GatewayAttestedCallerContext {
            tenant_id: id("tenant", suffix),
            caller_id: id("caller", suffix),
            policy_revision: id("policy", suffix),
            permitted_actions: BTreeSet::from([
                PermittedAction::Invoke,
                PermittedAction::CancelOwnInvocation,
                PermittedAction::QueryOwnInvocation,
            ]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::Chat,
        service_class: ServiceClass::Interactive,
        remaining_time_ms: 2_000,
        max_queue_wait_ms: 25,
        output_limits: OutputLimits {
            max_tokens: 32,
            max_bytes: 4096,
        },
        requested_output_format: OutputFormat::Text,
        session_id: None,
        request_digest: id("digest", suffix),
        input: InvocationInput::Chat {
            input: ChatInput {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "hello mock".into(),
                }],
            },
            parameters: ChatParameters::default(),
        },
    }
}

fn client(worker: &MockWorker) -> WorkerClient {
    WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig::default(),
    )
    .expect("client")
}

fn client_for_endpoint(endpoint: &str, config: WorkerClientConfig) -> WorkerClient {
    WorkerClient::new(endpoint, MockWorkerConfig::default().credentials, config).expect("client")
}

async fn unused_loopback_endpoint() -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("reserve loopback address");
    let address = listener.local_addr().expect("loopback address");
    drop(listener);
    format!("http://{address}")
}

#[tokio::test]
async fn refused_connection_is_classified_as_not_established() {
    let endpoint = unused_loopback_endpoint().await;
    let config = WorkerClientConfig {
        connect_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(200),
        ..WorkerClientConfig::default()
    };
    let client = client_for_endpoint(&endpoint, config);

    assert!(matches!(
        client
            .invoke(request(&MockWorkerConfig::default(), "connect-refused"))
            .await,
        Err(WorkerClientError::ConnectionNotEstablished(_))
    ));
}

#[tokio::test]
async fn response_header_timeout_remains_acceptance_unknown() {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind silent worker");
    let endpoint = format!(
        "http://{}",
        listener.local_addr().expect("silent worker address")
    );
    let server = tokio::spawn(async move {
        let (_connection, _) = listener.accept().await.expect("accept invocation");
        std::future::pending::<()>().await;
    });
    let config = WorkerClientConfig {
        connect_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(30),
        ..WorkerClientConfig::default()
    };
    let client = client_for_endpoint(&endpoint, config);

    assert!(matches!(
        client
            .invoke(request(&MockWorkerConfig::default(), "header-timeout"))
            .await,
        Err(WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))
    ));
    server.abort();
}

#[tokio::test]
async fn generic_http_503_is_not_a_safe_rejection() {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind status worker");
    let endpoint = format!(
        "http://{}",
        listener.local_addr().expect("status worker address")
    );
    let app = axum::Router::new().route(
        INVOCATIONS_PATH,
        axum::routing::post(|| async {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "temporarily unavailable",
            )
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve status worker");
    });
    let client = client_for_endpoint(&endpoint, WorkerClientConfig::default());

    assert!(matches!(
        client
            .invoke(request(&MockWorkerConfig::default(), "generic-503"))
            .await,
        Err(WorkerClientError::HttpStatus {
            status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ..
        })
    ));
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn happy_path_uses_authenticated_real_socket_and_strict_event_identity() {
    let worker = MockWorker::spawn(MockWorkerConfig::default())
        .await
        .unwrap();
    let client = client(&worker);

    let descriptor = client.descriptor().await.unwrap();
    assert_eq!(descriptor.incarnation_id, worker.config().incarnation_id);
    assert_eq!(descriptor.assignment.backend(), BackendKind::Cpu);
    let status = client.status().await.unwrap();
    assert_eq!(status.capacity.available_admission_credits, 1);

    let invocation = request(worker.config(), "happy");
    let identity = AttemptIdentity::from(&invocation);
    let events = client.invoke_collect(invocation).await.unwrap();
    assert_eq!(events.len(), 3);
    assert!(matches!(
        events[0].event,
        InvocationEventKind::Accepted { .. }
    ));
    assert!(matches!(
        events[1].event,
        InvocationEventKind::TextDelta { .. }
    ));
    assert!(matches!(
        events[2].event,
        InvocationEventKind::Completed { .. }
    ));
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );

    let query = client.query_attempt(&identity).await.unwrap();
    assert_eq!(query.state, AttemptState::Completed);
    assert_eq!(query.last_sequence, Some(2));
    let cancel = client.cancel_attempt(&identity).await.unwrap();
    assert_eq!(cancel.disposition, CancelDisposition::AlreadyTerminal);
}

#[tokio::test]
async fn authentication_failure_does_not_expose_worker_control_data() {
    let worker = MockWorker::spawn(MockWorkerConfig::default())
        .await
        .unwrap();
    let wrong = ServiceCredentials {
        credential_id: CredentialId::new("wrong-credential").unwrap(),
        bearer_token: ServiceBearerToken::new("wrong-token").unwrap(),
    };
    let client =
        WorkerClient::new(&worker.endpoint(), wrong, WorkerClientConfig::default()).unwrap();
    assert!(matches!(
        client.status().await,
        Err(WorkerClientError::HttpStatus {
            status: reqwest::StatusCode::UNAUTHORIZED,
            ..
        })
    ));
}

#[tokio::test]
async fn incompatible_version_model_and_generation_are_rejected_before_execution() {
    let worker = MockWorker::spawn(MockWorkerConfig::default())
        .await
        .unwrap();
    let client = client(&worker);

    let mut wrong_version = request(worker.config(), "version");
    wrong_version.schema_version = SchemaVersion::new(2, 0);
    assert!(matches!(
        client.invoke(wrong_version).await,
        Err(WorkerClientError::InvalidInvocation(
            ContractValidationError::UnsupportedVersion { .. }
        ))
    ));

    // The client refuses an unsupported version before sending. Exercise the worker-side fence
    // separately over the same real HTTP socket to prove independently versioned peers are safe.
    let mut raw_wrong_version = request(worker.config(), "raw-version");
    raw_wrong_version.schema_version = SchemaVersion::new(2, 0);
    let response = reqwest::Client::new()
        .post(format!(
            "{}{}",
            worker.endpoint(),
            INVOCATIONS_PATH.trim_start_matches('/')
        ))
        .header(
            SERVICE_AUTHORIZATION_HEADER,
            format!(
                "{SERVICE_AUTH_SCHEME} {}",
                worker.config().credentials.bearer_token.expose_secret()
            ),
        )
        .header(
            SERVICE_CREDENTIAL_ID_HEADER,
            worker.config().credentials.credential_id.as_str(),
        )
        .json(&raw_wrong_version)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);
    let rejection: InvocationRejection = response.json().await.unwrap();
    assert_eq!(rejection.code, RejectionCode::UnsupportedProtocolVersion);

    let mut wrong_model = request(worker.config(), "model");
    wrong_model.deployment_id = DeploymentId::new("missing-deployment").unwrap();
    assert!(matches!(
        client.invoke(wrong_model).await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::UnknownDeployment
    ));

    let mut wrong_generation = request(worker.config(), "generation");
    wrong_generation.expected_model_generation = ModelGeneration::new(99).unwrap();
    assert!(matches!(
        client.invoke(wrong_generation).await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::WrongModelGeneration
    ));
    assert_eq!(worker.active_invocations(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_worker_capacity_rejects_a_second_gateway_view() {
    let config = MockWorkerConfig {
        fault: MockFault::Hang,
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let first_client = client(&worker);
    let second_client = client(&worker);

    let mut first = first_client
        .invoke(request(worker.config(), "capacity-a"))
        .await
        .unwrap();
    assert!(matches!(
        first.next_event().await.unwrap().unwrap().event,
        InvocationEventKind::Accepted { .. }
    ));
    assert_eq!(worker.active_invocations(), 1);

    assert!(matches!(
        second_client
            .invoke(request(worker.config(), "capacity-b"))
            .await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::CapacityExhausted
    ));
    drop(first);
}

#[tokio::test]
async fn malformed_oversized_and_accepted_then_eof_are_never_success() {
    for (suffix, fault, expected) in [
        ("malformed", MockFault::MalformedEvent, "ndjson"),
        (
            "oversized",
            MockFault::OversizedEvent { text_bytes: 2048 },
            "ndjson",
        ),
        ("eof", MockFault::AcceptedThenDisconnect, "interrupted"),
    ] {
        let config = MockWorkerConfig {
            fault,
            ..MockWorkerConfig::default()
        };
        let worker = MockWorker::spawn(config).await.unwrap();
        let mut client_config = WorkerClientConfig::default();
        client_config.ndjson_limits.max_line_bytes = 1024;
        client_config.ndjson_limits.max_total_bytes = 4096;
        let client = WorkerClient::new(
            &worker.endpoint(),
            worker.config().credentials.clone(),
            client_config,
        )
        .unwrap();
        let error = client
            .invoke_collect(request(worker.config(), suffix))
            .await
            .unwrap_err();
        match expected {
            "ndjson" => assert!(matches!(error, WorkerClientError::Ndjson(_))),
            "interrupted" => assert!(matches!(error, WorkerClientError::InterruptedUnknown)),
            _ => unreachable!(),
        }
    }
}

#[tokio::test]
async fn lost_acknowledgement_is_unknown_and_never_releases_capacity_early() {
    let config = MockWorkerConfig {
        fault: MockFault::AcceptedWithoutAcknowledgement,
        cancellation_delay: Duration::from_millis(150),
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let client = client(&worker);
    let invocation = request(worker.config(), "lost-ack");
    let identity = AttemptIdentity::from(&invocation);

    assert!(matches!(
        client.invoke(invocation).await,
        Err(WorkerClientError::InterruptedUnknown)
    ));
    assert_eq!(worker.active_invocations(), 1);
    assert!(matches!(
        client
            .invoke(request(worker.config(), "lost-ack-second"))
            .await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::CapacityExhausted
    ));
    assert!(matches!(
        client.query_attempt(&identity).await.unwrap().state,
        AttemptState::Running
            | AttemptState::CancellationRequested
            | AttemptState::ExecutionStopping
    ));

    tokio::time::sleep(Duration::from_millis(180)).await;
    assert_eq!(worker.active_invocations(), 0);
    assert_eq!(
        client.query_attempt(&identity).await.unwrap().state,
        AttemptState::Cancelled
    );
}

#[tokio::test]
async fn partial_output_then_disconnect_is_never_replayed_or_reported_as_success() {
    let config = MockWorkerConfig {
        fault: MockFault::PartialThenDisconnect,
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let client = client(&worker);
    let error = client
        .invoke_collect(request(worker.config(), "partial-eof"))
        .await
        .unwrap_err();

    assert!(matches!(error, WorkerClientError::InterruptedUnknown));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_timeout_requests_cancel_but_capacity_waits_for_teardown() {
    let config = MockWorkerConfig {
        fault: MockFault::Hang,
        cancellation_delay: Duration::from_millis(180),
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let client_config = WorkerClientConfig {
        progress_timeout: Duration::from_millis(30),
        request_timeout: Duration::from_millis(200),
        ..WorkerClientConfig::default()
    };
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        client_config,
    )
    .unwrap();

    let timed_out = request(worker.config(), "timeout");
    let timed_out_identity = AttemptIdentity::from(&timed_out);
    let error = client.invoke_collect(timed_out).await.unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::StreamProgress)
    ));

    // Cancellation is asynchronous and, even after it is observed, teardown is deliberately slow.
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(worker.active_invocations(), 1);
    assert!(matches!(
        client
            .invoke(request(worker.config(), "during-teardown"))
            .await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::CapacityExhausted
    ));
    let query = client.query_attempt(&timed_out_identity).await.unwrap();
    assert!(matches!(
        query.state,
        AttemptState::CancellationRequested | AttemptState::ExecutionStopping
    ));

    tokio::time::sleep(Duration::from_millis(180)).await;
    assert_eq!(worker.active_invocations(), 0);
    let query = client.query_attempt(&timed_out_identity).await.unwrap();
    assert_eq!(query.state, AttemptState::Cancelled);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn usage_trickle_does_not_reset_first_useful_output_deadline() {
    assert_non_output_trickle_times_out(MockFault::UsageTrickleWithoutOutput, "usage-trickle")
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_byte_trickle_does_not_reset_first_useful_output_deadline() {
    assert_non_output_trickle_times_out(MockFault::ByteTrickleWithoutEvent, "byte-trickle").await;
}

async fn assert_non_output_trickle_times_out(fault: MockFault, suffix: &str) {
    let config = MockWorkerConfig {
        fault,
        output_cadence: Duration::from_millis(10),
        cancellation_delay: Duration::from_millis(80),
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            progress_timeout: Duration::from_millis(35),
            request_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let invocation = request(worker.config(), suffix);
    let identity = AttemptIdentity::from(&invocation);

    let error = client.invoke_collect(invocation).await.unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::StreamProgress)
    ));
    assert_eq!(worker.active_invocations(), 1);
    tokio::time::sleep(Duration::from_millis(25)).await;
    assert!(matches!(
        client.query_attempt(&identity).await.unwrap().state,
        AttemptState::CancellationRequested | AttemptState::ExecutionStopping
    ));
    assert_eq!(worker.active_invocations(), 1);

    tokio::time::sleep(Duration::from_millis(90)).await;
    assert_eq!(worker.active_invocations(), 0);
    assert_eq!(
        client.query_attempt(&identity).await.unwrap().state,
        AttemptState::Cancelled
    );
}

#[tokio::test]
async fn text_delta_resets_progress_deadline_until_completion() {
    let config = MockWorkerConfig {
        output_cadence: Duration::from_millis(25),
        ..MockWorkerConfig::default()
    };
    let worker = MockWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            progress_timeout: Duration::from_millis(40),
            request_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();

    let events = client
        .invoke_collect(request(worker.config(), "progressing-output"))
        .await
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [
            InvocationEvent {
                event: InvocationEventKind::Accepted { .. },
                ..
            },
            InvocationEvent {
                event: InvocationEventKind::TextDelta { .. },
                ..
            },
            InvocationEvent {
                event: InvocationEventKind::Completed { .. },
                ..
            }
        ]
    ));
}

#[tokio::test]
async fn request_body_is_bounded_before_any_post() {
    let worker = MockWorker::spawn(MockWorkerConfig::default())
        .await
        .unwrap();
    let client_config = WorkerClientConfig {
        max_request_json_bytes: 128,
        ..WorkerClientConfig::default()
    };
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        client_config,
    )
    .unwrap();
    assert!(matches!(
        client.invoke(request(worker.config(), "too-large")).await,
        Err(WorkerClientError::RequestTooLarge { limit: 128, .. })
    ));
    assert_eq!(worker.active_invocations(), 0);
}

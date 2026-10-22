//! The private worker client against the real worker (`WorkerService` over a
//! scripted runtime, see `izwi_serving_worker::test_support`) on real
//! sockets. Behaviour the real worker cannot produce, because it breaks the
//! wire contract, comes from small raw peers that only exist to break it.

use izwi_serving_client::{DeadlinePhase, WorkerClient, WorkerClientConfig, WorkerClientError};
use izwi_serving_protocol::*;
use izwi_serving_worker::test_support::{TestWorker, TestWorkerConfig, WorkerScript};
use std::{collections::BTreeSet, time::Duration};
use tokio::net::TcpListener;

fn id<T: TryFrom<String>>(prefix: &str, suffix: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(format!("{prefix}-{suffix}")).expect("test identity")
}

fn request(config: &TestWorkerConfig, suffix: &str) -> InvocationRequest {
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
                    content: "hello worker".into(),
                }],
            },
            parameters: ChatParameters::default(),
        },
    }
}

fn client(worker: &TestWorker) -> WorkerClient {
    WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig::default(),
    )
    .expect("client")
}

fn client_for_endpoint(endpoint: &str, config: WorkerClientConfig) -> WorkerClient {
    WorkerClient::new(endpoint, TestWorkerConfig::default().credentials, config).expect("client")
}

/// Wire faults the real worker never produces; the client must still
/// classify them.
#[derive(Clone, Copy)]
enum WireFault {
    /// A 200 NDJSON response whose body never carries the Accepted event.
    OpenBodyWithoutAccepted,
    /// A 200 NDJSON response whose body ends before the Accepted event.
    EmptyBody,
    /// A line that is not a JSON event.
    MalformedEvent,
    /// Accepted, then usage events and never output.
    UsageTrickle,
    /// Accepted, then single bytes that never complete an event.
    ByteTrickle,
}

fn event_line(request: &InvocationRequest, sequence: u64, event: InvocationEventKind) -> Vec<u8> {
    let mut line = serde_json::to_vec(&InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence,
        event,
    })
    .expect("event encodes");
    line.push(b'\n');
    line
}

/// A peer that answers every invocation with `fault`, and its endpoint.
async fn broken_peer(fault: WireFault) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind broken peer");
    let endpoint = format!("http://{}/", listener.local_addr().expect("peer address"));
    let app = axum::Router::new().route(
        INVOCATIONS_PATH,
        axum::routing::post(move |body: axum::body::Bytes| async move {
            let request: InvocationRequest = serde_json::from_slice(&body).expect("invocation");
            let accepted = event_line(
                &request,
                0,
                InvocationEventKind::Accepted {
                    worker_id: TestWorkerConfig::default().worker_id,
                    node_id: TestWorkerConfig::default().node_id,
                    incarnation_id: request.expected_worker_incarnation.clone(),
                    deployment_id: request.deployment_id.clone(),
                    model_generation: request.expected_model_generation,
                },
            );
            let body = async_stream::stream! {
                match fault {
                    WireFault::OpenBodyWithoutAccepted => std::future::pending::<()>().await,
                    WireFault::EmptyBody => {}
                    WireFault::MalformedEvent => {
                        yield Ok::<_, std::convert::Infallible>(b"{malformed}\n".to_vec());
                    }
                    WireFault::UsageTrickle => {
                        yield Ok(accepted);
                        for sequence in 1.. {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                            yield Ok(event_line(
                                &request,
                                sequence,
                                InvocationEventKind::Usage {
                                    usage: Usage {
                                        input_tokens: 1,
                                        output_tokens: 0,
                                        cached_tokens: None,
                                    },
                                },
                            ));
                        }
                    }
                    WireFault::ByteTrickle => {
                        yield Ok(accepted);
                        loop {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                            yield Ok(b"{".to_vec());
                        }
                    }
                }
            };
            axum::response::Response::builder()
                .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
                .body(axum::body::Body::from_stream(body))
                .expect("static response")
        }),
    );
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (endpoint, server)
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
            .invoke(request(&TestWorkerConfig::default(), "connect-refused"))
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
            .invoke(request(&TestWorkerConfig::default(), "header-timeout"))
            .await,
        Err(WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))
    ));
    server.abort();
}

#[tokio::test]
async fn open_response_body_without_accepted_times_out_in_invocation_admission() {
    let (endpoint, server) = broken_peer(WireFault::OpenBodyWithoutAccepted).await;
    let client = client_for_endpoint(
        &endpoint,
        WorkerClientConfig {
            request_timeout: Duration::from_millis(30),
            first_output_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    );

    let error = client
        .invoke(request(
            &TestWorkerConfig::default(),
            "open-body-no-accepted",
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::InvocationAdmission)
    ));
    server.abort();
}

#[tokio::test]
async fn body_ending_before_accepted_is_unknown_not_a_rejection() {
    let (endpoint, server) = broken_peer(WireFault::EmptyBody).await;
    let client = client_for_endpoint(&endpoint, WorkerClientConfig::default());
    let error = client
        .invoke(request(&TestWorkerConfig::default(), "empty-body"))
        .await
        .unwrap_err();
    assert!(matches!(error, WorkerClientError::InterruptedUnknown));
    assert!(!error.proves_attempt_unaccepted());
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
            .invoke(request(&TestWorkerConfig::default(), "generic-503"))
            .await,
        Err(WorkerClientError::HttpStatus {
            status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ..
        })
    ));
    server.abort();
}

/// DP-F11: the client declares its protocol version, so a worker shows it
/// the minor-4 chat features that a gateway without the header (minor 3)
/// cannot decode; an older worker negotiates its own minor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn descriptor_negotiates_the_minor_and_its_features() {
    let worker = TestWorker::spawn(TestWorkerConfig::default())
        .await
        .unwrap();
    let descriptor = client(&worker).descriptor().await.unwrap();
    assert_eq!(descriptor.negotiated_version(), Some(PROTOCOL_V1));
    assert!(descriptor.features.contains(&WorkerFeature::ChatSeed));

    let _ = rustls::crypto::ring::default_provider().install_default();
    let undeclared: serde_json::Value = reqwest::Client::new()
        .get(format!(
            "{}{}",
            worker.endpoint(),
            WORKER_DESCRIPTOR_PATH.trim_start_matches('/')
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
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        undeclared["features"],
        serde_json::json!(["streaming", "cancellation", "attempt_query"])
    );

    let minor3 = TestWorker::spawn(TestWorkerConfig {
        protocol_version: SchemaVersion::new(1, 3),
        chat_features: BTreeSet::new(),
        ..TestWorkerConfig::default()
    })
    .await
    .unwrap();
    let descriptor = client(&minor3).descriptor().await.unwrap();
    assert_eq!(
        descriptor.negotiated_version(),
        Some(SchemaVersion::new(1, 3))
    );
    // A knob the worker does not advertise is refused, never ignored.
    let mut seeded = request(minor3.config(), "seeded");
    let InvocationInput::Chat { parameters, .. } = &mut seeded.input;
    parameters.seed = Some(7);
    assert!(matches!(
        client(&minor3).invoke_collect(seeded).await,
        Err(WorkerClientError::Rejected { rejection }) if rejection.code == RejectionCode::InvalidRequest
    ));
    assert_eq!(minor3.served_invocations(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn happy_path_uses_authenticated_real_socket_and_strict_event_identity() {
    let worker = TestWorker::spawn(TestWorkerConfig::default())
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
async fn cached_usage_reports_in_completed_usage_when_configured() {
    let worker = TestWorker::spawn(TestWorkerConfig {
        usage_cached_input_tokens: Some(7),
        ..TestWorkerConfig::default()
    })
    .await
    .unwrap();
    let client = client(&worker);

    let invocation = request(worker.config(), "cached");
    let events = client.invoke_collect(invocation).await.unwrap();
    let completed = events
        .iter()
        .find_map(|event| match &event.event {
            InvocationEventKind::Completed { usage, .. } => Some(usage.clone()),
            _ => None,
        })
        .expect("completed event present");
    let usage = completed.expect("completed usage present");
    assert_eq!(usage.input_tokens, 1);
    assert_eq!(usage.cached_tokens, Some(7));
}

#[tokio::test]
async fn configured_delta_logprobs_report_on_text_delta_events() {
    let worker = TestWorker::spawn(TestWorkerConfig {
        delta_logprobs: Some(vec![TokenLogprob {
            token: "x".into(),
            logprob: -1.25,
            bytes: b"x".to_vec(),
            top_logprobs: vec![TopTokenLogprob {
                token: "x".into(),
                logprob: -1.25,
                bytes: b"x".to_vec(),
            }],
        }]),
        ..TestWorkerConfig::default()
    })
    .await
    .unwrap();
    let client = client(&worker);

    let invocation = request(worker.config(), "logprobs");
    let events = client.invoke_collect(invocation).await.unwrap();
    let deltas = events
        .iter()
        .filter_map(|event| match &event.event {
            InvocationEventKind::TextDelta { logprobs, .. } => Some(logprobs.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!deltas.is_empty(), "expected at least one text delta");
    for logprobs in deltas {
        let entries = logprobs.expect("configured entries present");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].token, "x");
        assert_eq!(entries[0].logprob, -1.25);
    }
}

#[tokio::test]
async fn cached_usage_is_absent_on_the_wire_when_not_configured() {
    let worker = TestWorker::spawn(TestWorkerConfig::default())
        .await
        .unwrap();
    let client = client(&worker);

    let invocation = request(worker.config(), "uncached");
    let events = client.invoke_collect(invocation).await.unwrap();
    let completed = events
        .iter()
        .find_map(|event| match &event.event {
            InvocationEventKind::Completed { usage, .. } => Some(usage.clone()),
            _ => None,
        })
        .expect("completed event present");
    let usage = completed.expect("completed usage present");
    assert_eq!(usage.cached_tokens, None);
    let raw = serde_json::to_value(&usage).unwrap();
    assert!(
        raw.get("cached_tokens").is_none(),
        "unmeasured cache must omit the field, got {raw}"
    );
}

#[tokio::test]
async fn authentication_failure_does_not_expose_worker_control_data() {
    let worker = TestWorker::spawn(TestWorkerConfig::default())
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
    let worker = TestWorker::spawn(TestWorkerConfig::default())
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
    let _ = rustls::crypto::ring::default_provider().install_default();
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
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invocation_budget_includes_time_waiting_for_a_client_permit() {
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        max_active_invocations: 2,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            max_in_flight: 1,
            request_timeout: Duration::from_millis(500),
            first_output_timeout: Duration::from_millis(500),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let first = client
        .invoke(request(worker.config(), "permit-budget-holder"))
        .await
        .unwrap();

    let mut waiting_request = request(worker.config(), "permit-budget-forwarded");
    waiting_request.remaining_time_ms = 250;
    let waiting_identity = AttemptIdentity::from(&waiting_request);
    let waiting_client = client.clone();
    let waiting = tokio::spawn(async move { waiting_client.invoke(waiting_request).await });
    tokio::time::sleep(Duration::from_millis(40)).await;
    drop(first);

    let second = waiting.await.unwrap().unwrap();
    let forwarded_budget = worker
        .attempt_remaining_time_ms(&waiting_identity.attempt_id)
        .expect("second invocation reached worker");
    assert!(
        forwarded_budget < 250,
        "permit wait must be deducted from the forwarded budget: {forwarded_budget}"
    );
    drop(second);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invocation_total_budget_caps_wait_for_a_client_permit() {
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        max_active_invocations: 2,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            max_in_flight: 1,
            request_timeout: Duration::from_millis(500),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let first = client
        .invoke(request(worker.config(), "permit-deadline-holder"))
        .await
        .unwrap();
    let mut waiting_request = request(worker.config(), "permit-deadline-expired");
    waiting_request.remaining_time_ms = 30;

    let started = std::time::Instant::now();
    let error = client.invoke(waiting_request).await.unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::InFlightPermit)
    ));
    assert!(started.elapsed() < Duration::from_millis(200));
    drop(first);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_streams_never_starve_status_or_cancel_requests() {
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        max_active_invocations: 2,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            max_in_flight: 1,
            request_timeout: Duration::from_millis(500),
            first_output_timeout: Duration::from_secs(5),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    // The only invocation permit is held by a live stream.
    let streaming = request(worker.config(), "control-holder");
    let identity = AttemptIdentity::from(&streaming);
    let stream = client.invoke(streaming).await.unwrap();

    let started = std::time::Instant::now();
    client
        .status()
        .await
        .expect("status while a stream is live");
    client
        .descriptor()
        .await
        .expect("descriptor while a stream is live");
    client
        .cancel_attempt(&identity)
        .await
        .expect("cancel while a stream is live");
    assert!(started.elapsed() < Duration::from_millis(400));
    drop(stream);
}

#[tokio::test]
async fn malformed_oversized_and_accepted_then_eof_are_never_success() {
    let (endpoint, server) = broken_peer(WireFault::MalformedEvent).await;
    let error = client_for_endpoint(&endpoint, WorkerClientConfig::default())
        .invoke_collect(request(&TestWorkerConfig::default(), "malformed"))
        .await
        .unwrap_err();
    assert!(matches!(error, WorkerClientError::Ndjson(_)));
    server.abort();

    for (suffix, config, expected) in [
        (
            "oversized",
            TestWorkerConfig {
                output_text: "x".repeat(2048),
                ..TestWorkerConfig::default()
            },
            "ndjson",
        ),
        (
            "eof",
            TestWorkerConfig {
                script: WorkerScript::LoseAfterAccept,
                ..TestWorkerConfig::default()
            },
            "interrupted",
        ),
    ] {
        let worker = TestWorker::spawn(config).await.unwrap();
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

/// Runtime admission that outlasts the request's queue wait: the worker
/// cannot say whether it accepted, so the outcome is unknown, and the
/// capacity stays held until the late admission is cancelled and torn down.
#[tokio::test]
async fn lost_acknowledgement_is_unknown_and_never_releases_capacity_early() {
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        admission_delay: Duration::from_millis(100),
        cancellation_delay: Duration::from_millis(80),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = client(&worker);
    let invocation = request(worker.config(), "lost-ack");
    let identity = AttemptIdentity::from(&invocation);

    let error = client.invoke(invocation).await.unwrap_err();
    assert!(!error.proves_attempt_unaccepted(), "{error}");
    assert_eq!(worker.active_invocations(), 1);
    assert!(matches!(
        client
            .invoke(request(worker.config(), "lost-ack-second"))
            .await,
        Err(WorkerClientError::Rejected { rejection })
            if rejection.code == RejectionCode::CapacityExhausted
    ));
    assert!(!client
        .query_attempt(&identity)
        .await
        .unwrap()
        .state
        .is_terminal());

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while worker.active_invocations() != 0 {
        assert!(std::time::Instant::now() < deadline, "capacity never freed");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(client
        .query_attempt(&identity)
        .await
        .unwrap()
        .state
        .is_terminal());
}

#[tokio::test]
async fn partial_output_then_disconnect_is_never_replayed_or_reported_as_success() {
    let config = TestWorkerConfig {
        script: WorkerScript::LoseAfterPartial,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = client(&worker);
    let error = client
        .invoke_collect(request(worker.config(), "partial-eof"))
        .await
        .unwrap_err();

    assert!(matches!(error, WorkerClientError::InterruptedUnknown));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_output_timeout_requests_cancel_but_capacity_waits_for_teardown() {
    let config = TestWorkerConfig {
        script: WorkerScript::Hang,
        cancellation_delay: Duration::from_millis(180),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client_config = WorkerClientConfig {
        first_output_timeout: Duration::from_millis(30),
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
        WorkerClientError::Deadline(DeadlinePhase::FirstOutput)
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

/// A long prompt earns first-output time in proportion to its size, so its
/// slow prefill is not cancelled by the deadline sized for short prompts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn first_output_deadline_grows_with_the_prompt() {
    let config = TestWorkerConfig {
        script: WorkerScript::Complete,
        output_cadence: Duration::from_millis(150),
        cancellation_delay: Duration::from_millis(10),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            first_output_timeout: Duration::from_millis(40),
            first_output_timeout_per_kib: Duration::from_millis(40),
            request_timeout: Duration::from_millis(500),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();

    let error = client
        .invoke_collect(request(worker.config(), "short-prompt"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::FirstOutput)
    ));

    for _ in 0..100 {
        if worker.active_invocations() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // 8 KiB of prompt: 40 ms + 8 x 40 ms outlasts the 150 ms prefill.
    let mut long = request(worker.config(), "long-prompt");
    let InvocationInput::Chat { input, .. } = &mut long.input;
    input.messages[0].content = "x".repeat(8 * 1024);
    let events = client.invoke_collect(long).await.unwrap();
    assert!(matches!(
        events.last().map(|event| &event.event),
        Some(InvocationEventKind::Completed { .. })
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn usage_trickle_does_not_reset_first_useful_output_deadline() {
    assert_peer_trickle_times_out(WireFault::UsageTrickle, "usage-trickle").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_byte_trickle_does_not_reset_first_useful_output_deadline() {
    assert_peer_trickle_times_out(WireFault::ByteTrickle, "byte-trickle").await;
}

async fn assert_peer_trickle_times_out(fault: WireFault, suffix: &str) {
    let (endpoint, server) = broken_peer(fault).await;
    let client = client_for_endpoint(
        &endpoint,
        WorkerClientConfig {
            first_output_timeout: Duration::from_millis(35),
            progress_timeout: Duration::from_millis(20),
            request_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    );
    let error = client
        .invoke_collect(request(&TestWorkerConfig::default(), suffix))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::FirstOutput)
    ));
    server.abort();
}

/// Empty text deltas are real worker output (a runtime may emit them), but
/// not useful output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_delta_trickle_does_not_reset_first_useful_output_deadline() {
    let config = TestWorkerConfig {
        script: WorkerScript::EmptyDeltaTrickle,
        output_cadence: Duration::from_millis(10),
        cancellation_delay: Duration::from_millis(80),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            first_output_timeout: Duration::from_millis(35),
            progress_timeout: Duration::from_millis(20),
            request_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();
    let invocation = request(worker.config(), "empty-delta-trickle");
    let identity = AttemptIdentity::from(&invocation);

    let error = client.invoke_collect(invocation).await.unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::FirstOutput)
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_empty_output_switches_to_the_progress_idle_deadline() {
    let config = TestWorkerConfig {
        script: WorkerScript::TextDeltaThenHang,
        output_cadence: Duration::from_millis(25),
        cancellation_delay: Duration::from_millis(40),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        WorkerClientConfig {
            first_output_timeout: Duration::from_millis(80),
            progress_timeout: Duration::from_millis(20),
            request_timeout: Duration::from_millis(200),
            ..WorkerClientConfig::default()
        },
    )
    .unwrap();

    let error = client
        .invoke_collect(request(worker.config(), "post-output-stall"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        WorkerClientError::Deadline(DeadlinePhase::StreamProgress)
    ));
}

#[tokio::test]
async fn text_delta_resets_progress_deadline_until_completion() {
    let config = TestWorkerConfig {
        output_cadence: Duration::from_millis(25),
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
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

fn wide_chat_transport_config() -> WorkerClientConfig {
    let mut config = WorkerClientConfig::default();
    config.ndjson_limits.max_line_bytes = 1024 * 1024;
    config.ndjson_limits.max_total_bytes = 16 * 1024 * 1024;
    config.ndjson_limits.max_events = 8192;
    config
}

#[tokio::test]
async fn gateway_sized_event_budget_allows_4096_deltas_and_terminal_control_events() {
    let config = TestWorkerConfig {
        script: WorkerScript::ManyTextDeltas {
            count: 4096,
            text_bytes: 1,
        },
        output_cadence: Duration::ZERO,
        max_output_tokens: 4096,
        // Room for every delta, so the worker never merges them for a
        // lagging reader and the client sees 4096 separate events.
        event_channel_capacity: 4100,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        wide_chat_transport_config(),
    )
    .unwrap();
    let mut invocation = request(worker.config(), "maximum-delta-count");
    invocation.remaining_time_ms = 20_000;
    invocation.output_limits.max_tokens = 4096;
    invocation.output_limits.max_bytes = 512 * 1024;

    let events = client.invoke_collect(invocation).await.unwrap();
    assert_eq!(events.len(), 4098);
    assert!(matches!(
        events.last().map(|event| &event.event),
        Some(InvocationEventKind::Completed { .. })
    ));
}

#[tokio::test]
async fn gateway_sized_line_budget_allows_a_near_512_kib_text_delta() {
    let output_bytes = 512 * 1024 - 1024;
    let config = TestWorkerConfig {
        output_text: "x".repeat(output_bytes),
        output_cadence: Duration::ZERO,
        ..TestWorkerConfig::default()
    };
    let worker = TestWorker::spawn(config).await.unwrap();
    let client = WorkerClient::new(
        &worker.endpoint(),
        worker.config().credentials.clone(),
        wide_chat_transport_config(),
    )
    .unwrap();
    let mut invocation = request(worker.config(), "near-maximum-line");
    invocation.output_limits.max_bytes = 512 * 1024;

    let events = client.invoke_collect(invocation).await.unwrap();
    assert!(matches!(
        events.get(1).map(|event| &event.event),
        Some(InvocationEventKind::TextDelta { text, .. }) if text.len() == output_bytes
    ));
}

#[tokio::test]
async fn request_body_is_bounded_before_any_post() {
    let worker = TestWorker::spawn(TestWorkerConfig::default())
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

#[tokio::test]
async fn duplicate_attempt_returns_unretryable_conflict_response() {
    let worker = TestWorker::spawn(TestWorkerConfig {
        output_cadence: Duration::from_millis(50),
        cancellation_delay: Duration::from_millis(150),
        ..TestWorkerConfig::default()
    })
    .await
    .unwrap();
    let client = client(&worker);
    let invocation = request(worker.config(), "dup-attempt");
    let duplicate = invocation.clone();

    let first_task = tokio::spawn({
        let client = client.clone();
        async move { client.invoke_collect(invocation).await }
    });

    tokio::time::sleep(Duration::from_millis(20)).await;

    let dup_error = client.invoke_collect(duplicate).await.unwrap_err();
    assert!(matches!(
        dup_error,
        WorkerClientError::HttpStatus { status, ref body }
            if status == reqwest::StatusCode::CONFLICT && body.contains("attempt is already owned")
    ));
    assert!(!dup_error.proves_attempt_unaccepted());

    let first_result = first_task.await.unwrap();
    assert!(first_result.is_ok());
}

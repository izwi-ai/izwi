use super::*;
use izwi_serving_protocol::RejectionCode;
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use izwi_hooks::Principal;
use izwi_model_catalog::{ChatMessage, ChatRequestConfig, ChatRole};
use izwi_serving_client::{WorkerClient, WorkerClientConfig};
use izwi_serving_protocol::{
    ArtifactRevision, BackendKind, Capability, CapacitySnapshot, ChatContextPolicy,
    ChatParameters as WorkerChatParameters, ChatResponseFormat,
    ChatTruncation as WorkerChatTruncation, CredentialId, DeviceAssignment, FinishReason,
    IncarnationId, InvocationAdmissionUnknown, InvocationEvent, InvocationEventKind,
    InvocationInput, InvocationRejection, InvocationRequest, LoadedDeployment, ModelGeneration,
    ModelReadiness, NodeId, SchemaVersion, ServiceBearerToken, ServiceCredentials, Usage,
    WorkerDescriptor, WorkerFeature, WorkerId, WorkerProcessState, WorkerStatus, INVOCATIONS_PATH,
    NDJSON_MEDIA_TYPE,
};

use crate::db::StoreDatabase;
use crate::gateway_fleet::FleetStore;

use crate::worker_registry::{
    ApprovedDeployment, ApprovedWorker, RoutingConfig, WorkerRegistryConfig,
};

fn id<T>(value: &str) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_string()).unwrap()
}

fn credentials() -> ServiceCredentials {
    ServiceCredentials {
        credential_id: id::<CredentialId>("gateway-test-key"),
        bearer_token: ServiceBearerToken::new("gateway-test-secret").unwrap(),
    }
}

fn client(endpoint: &str) -> WorkerClient {
    WorkerClient::new(endpoint, credentials(), WorkerClientConfig::default()).unwrap()
}

#[derive(Clone)]
enum ScriptedResponse {
    Reject(RejectionCode),
    AdmissionUnknown,
    AcceptedWithoutAcknowledgement,
    PartialThenDisconnect,
    ManyDeltas(usize),
    Success(String),
    /// Accepts, then keeps the stream open without further events: an
    /// in-flight invocation that holds one worker credit.
    AcceptThenHold,
}

#[derive(Clone)]
struct ScriptedState {
    response: ScriptedResponse,
    delay: Duration,
    requests: Arc<Mutex<Vec<InvocationRequest>>>,
}

struct ScriptedWorker {
    address: SocketAddr,
    requests: Arc<Mutex<Vec<InvocationRequest>>>,
    server: tokio::task::JoinHandle<()>,
}

impl ScriptedWorker {
    async fn spawn(response: ScriptedResponse, delay: Duration) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let state = ScriptedState {
            response,
            delay,
            requests: Arc::clone(&requests),
        };
        let app = Router::new()
            .route(INVOCATIONS_PATH, post(scripted_invoke))
            .with_state(state);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self {
            address,
            requests,
            server,
        }
    }

    fn endpoint(&self) -> String {
        format!("http://{}/", self.address)
    }

    fn requests(&self) -> Vec<InvocationRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for ScriptedWorker {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn scripted_invoke(State(state): State<ScriptedState>, body: Bytes) -> Response<Body> {
    let request: InvocationRequest = serde_json::from_slice(&body).unwrap();
    state.requests.lock().unwrap().push(request.clone());
    if !state.delay.is_zero() {
        tokio::time::sleep(state.delay).await;
    }
    match state.response {
        ScriptedResponse::Reject(code) => Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_vec(&InvocationRejection::new(
                    request.request_id,
                    request.attempt_id,
                    code,
                    "scripted rejection",
                ))
                .unwrap(),
            ))
            .unwrap(),
        ScriptedResponse::AdmissionUnknown => Response::builder()
            .status(StatusCode::GATEWAY_TIMEOUT)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_vec(&InvocationAdmissionUnknown {
                    schema_version: PROTOCOL_V1,
                    request_id: request.request_id,
                    attempt_id: request.attempt_id,
                    message: "scripted admission timeout".into(),
                })
                .unwrap(),
            ))
            .unwrap(),
        ScriptedResponse::AcceptedWithoutAcknowledgement => Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
            .body(Body::empty())
            .unwrap(),
        ScriptedResponse::PartialThenDisconnect => accepted_response(
            &request,
            vec![InvocationEventKind::TextDelta {
                text: "partial".into(),
                logprobs: None,
            }],
        ),
        ScriptedResponse::ManyDeltas(count) => {
            let mut following = Vec::with_capacity(count.saturating_add(1));
            following.extend((0..count).map(|_| InvocationEventKind::TextDelta {
                text: "x".into(),
                logprobs: None,
            }));
            following.push(InvocationEventKind::Completed {
                finish_reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: 1,
                    output_tokens: u64::try_from(count).unwrap_or(u64::MAX),
                    cached_tokens: None,
                }),
            });
            accepted_response(&request, following)
        }
        ScriptedResponse::AcceptThenHold => {
            use futures::StreamExt as _;

            let accepted = accepted_response(&request, Vec::new());
            let first = axum::body::to_bytes(accepted.into_body(), usize::MAX)
                .await
                .unwrap();
            let held = futures::stream::once(async move { Ok::<_, std::io::Error>(first) })
                .chain(futures::stream::pending());
            Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
                .body(Body::from_stream(held))
                .unwrap()
        }
        ScriptedResponse::Success(text) => accepted_response(
            &request,
            vec![
                InvocationEventKind::TextDelta {
                    text,
                    logprobs: None,
                },
                InvocationEventKind::Completed {
                    finish_reason: FinishReason::Stop,
                    usage: Some(Usage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cached_tokens: None,
                    }),
                },
            ],
        ),
    }
}

fn accepted_response(
    request: &InvocationRequest,
    following: Vec<InvocationEventKind>,
) -> Response<Body> {
    let mut events = vec![InvocationEventKind::Accepted {
        worker_id: id::<WorkerId>("scripted-worker"),
        node_id: id::<NodeId>("node-a"),
        incarnation_id: request.expected_worker_incarnation.clone(),
        deployment_id: request.deployment_id.clone(),
        model_generation: request.expected_model_generation,
    }];
    events.extend(following);
    let mut encoded = Vec::new();
    for (sequence, event) in events.into_iter().enumerate() {
        encoded.extend(
            serde_json::to_vec(&InvocationEvent {
                schema_version: PROTOCOL_V1,
                request_id: request.request_id.clone(),
                attempt_id: request.attempt_id.clone(),
                sequence: sequence as u64,
                event,
            })
            .unwrap(),
        );
        encoded.push(b'\n');
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
        .body(Body::from(encoded))
        .unwrap()
}

async fn refused_endpoint() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}/")
}

fn deployment() -> LoadedDeployment {
    LoadedDeployment {
        deployment_id: id::<DeploymentId>("chat-prod"),
        public_model: ModelAlias::new(ModelVariant::Qwen354BGguf.dir_name()).unwrap(),
        artifact_revision: id::<ArtifactRevision>("artifact-v1"),
        model_generation: ModelGeneration::new(1).unwrap(),
        task: TaskKind::Chat,
        backend: BackendKind::Cpu,
        precision: "mock".into(),
        execution_representation: "deterministic-text".into(),
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
            max_context_tokens: Some(4096),
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

fn register(
    registry: &WorkerRegistry,
    worker_name: &str,
    incarnation_name: &str,
    worker_client: WorkerClient,
    status_sequence: u64,
) {
    register_with_credits(
        registry,
        worker_name,
        incarnation_name,
        worker_client,
        status_sequence,
        1,
    );
}

fn register_with_credits(
    registry: &WorkerRegistry,
    worker_name: &str,
    incarnation_name: &str,
    worker_client: WorkerClient,
    status_sequence: u64,
    credits: u32,
) {
    register_with_surface(
        registry,
        worker_name,
        incarnation_name,
        worker_client,
        status_sequence,
        credits,
        PROTOCOL_V1,
        izwi_serving_worker::test_support::minor4_chat_features(),
    );
}

/// A worker before minor 4: it advertises its own minor and no chat
/// control feature.
fn register_minor3(
    registry: &WorkerRegistry,
    worker_name: &str,
    incarnation_name: &str,
    worker_client: WorkerClient,
) {
    register_with_surface(
        registry,
        worker_name,
        incarnation_name,
        worker_client,
        1,
        1,
        SchemaVersion::new(1, 3),
        BTreeSet::new(),
    );
}

#[allow(clippy::too_many_arguments)]
fn register_with_surface(
    registry: &WorkerRegistry,
    worker_name: &str,
    incarnation_name: &str,
    worker_client: WorkerClient,
    status_sequence: u64,
    credits: u32,
    protocol_version: SchemaVersion,
    chat_features: BTreeSet<WorkerFeature>,
) {
    let loaded = deployment();
    let mut features = BTreeSet::from([WorkerFeature::Streaming, WorkerFeature::Cancellation]);
    features.extend(chat_features);
    let descriptor = WorkerDescriptor {
        schema_version: protocol_version,
        supported_protocol_versions: vec![protocol_version],
        worker_id: id::<WorkerId>(worker_name),
        node_id: id::<NodeId>("node-a"),
        incarnation_id: id(incarnation_name),
        build_version: "test".into(),
        assignment: DeviceAssignment::Cpu {
            thread_budget: 1,
            affinity: Vec::new(),
            host_memory_limit_bytes: 1024,
        },
        features,
    };
    registry
        .approve(ApprovedWorker {
            descriptor: descriptor.clone(),
            client: worker_client,
            approved_deployments: BTreeMap::from([(
                loaded.deployment_id.clone(),
                ApprovedDeployment::from_loaded(&loaded),
            )]),
            validated_capacity: credits,
        })
        .unwrap();
    registry
        .observe_status(worker_status(
            worker_name,
            incarnation_name,
            status_sequence,
            credits,
            0,
        ))
        .unwrap();
}

fn worker_status(
    worker_name: &str,
    incarnation_name: &str,
    status_sequence: u64,
    credits: u32,
    active: u32,
) -> WorkerStatus {
    WorkerStatus {
        schema_version: PROTOCOL_V1,
        worker_id: id::<WorkerId>(worker_name),
        node_id: id::<NodeId>("node-a"),
        incarnation_id: id(incarnation_name),
        status_sequence,
        process_state: WorkerProcessState::Running,
        deployments: vec![deployment()],
        capacity: CapacitySnapshot {
            max_active_invocations: credits,
            active_invocations: active,
            max_queued_invocations: 0,
            queued_invocations: 0,
            max_sessions: 0,
            reserved_sessions: 0,
            available_admission_credits: credits.saturating_sub(active),
            outstanding_cost_units: 0,
        },
    }
}

fn dispatcher(registry: WorkerRegistry) -> RemoteChatDispatcher {
    dispatcher_with_prefix(registry, None)
}

fn dispatcher_with_prefix(
    registry: WorkerRegistry,
    prefix_affinity: Option<PrefixAffinityConfig>,
) -> RemoteChatDispatcher {
    RemoteChatDispatcher::new(
        registry,
        RemoteChatDispatchConfig {
            public_model_variant: ModelVariant::Qwen354BGguf,
            deployment_id: id::<DeploymentId>("chat-prod"),
            policy_revision: id::<PolicyRevision>("policy-v1"),
            backend_policy: BackendPolicy::ANY,
            max_queue_wait: Duration::ZERO,
            max_output_tokens: 128,
            max_output_bytes: 4096,
            slow_consumer_timeout: Duration::from_millis(100),
            prefix_affinity,
        },
    )
    .unwrap()
}

fn request() -> ChatExecutionRequest {
    ChatExecutionRequest {
        variant: ModelVariant::Qwen354BGguf,
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        }],
        max_completion_tokens: None,
        max_tokens: Some(32),
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        presence_penalty: None,
        seed: None,
        stop: Vec::new(),
        logprobs: None,
        top_logprobs: None,
        response_format_json_object: false,
        skip_special_tokens: None,
        chat_config: ChatRequestConfig::default(),
        correlation_id: None,
    }
}

fn tenant_work() -> UnboundTenantWorkLease {
    crate::gateway_tenant_concurrency::GatewayTenantConcurrency::new(
        crate::gateway_tenant_concurrency::GatewayTenantConcurrencyConfig::new(1, 1).unwrap(),
    )
    .try_reserve([7; 32])
    .unwrap()
}

#[test]
fn local_reservation_balances_concurrent_dispatches() {
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client("http://127.0.0.1:19101"),
        1,
    );
    register(
        &registry,
        "worker-b",
        "inc-b",
        client("http://127.0.0.1:19102"),
        1,
    );
    let dispatcher = dispatcher(registry);
    let selection = dispatcher.selection_request(&request(), false).unwrap();

    let first = dispatcher.registry.select_and_reserve(&selection).unwrap();
    assert_eq!(first.key.worker_id.as_str(), "worker-a");
    let second = dispatcher.registry.select_and_reserve(&selection).unwrap();
    assert_eq!(second.key.worker_id.as_str(), "worker-b");

    drop(first);
    drop(second);
}

#[test]
fn retry_classifier_is_an_explicit_allowlist() {
    let rejection = |code| WorkerClientError::Rejected {
        rejection: InvocationRejection::new(id("request-1"), id("attempt-1"), code, "test"),
    };
    for code in [
        RejectionCode::CapacityExhausted,
        RejectionCode::QueueWaitExceeded,
        RejectionCode::WrongWorkerIncarnation,
        RejectionCode::WrongModelGeneration,
        RejectionCode::UnknownDeployment,
        RejectionCode::ModelNotReady,
        RejectionCode::WorkerDraining,
    ] {
        assert!(retryable_before_acceptance(&rejection(code)), "{code:?}");
    }
    for code in [
        RejectionCode::Unauthenticated,
        RejectionCode::Unauthorized,
        RejectionCode::UnsupportedProtocolVersion,
        RejectionCode::IncompatibleTask,
        RejectionCode::InvalidRequest,
        RejectionCode::DuplicateAttemptConflict,
        RejectionCode::PolicyDenied,
    ] {
        assert!(!retryable_before_acceptance(&rejection(code)), "{code:?}");
    }
    assert!(retryable_before_acceptance(&WorkerClientError::Deadline(
        DeadlinePhase::InFlightPermit,
    )));
    assert!(!retryable_before_acceptance(&WorkerClientError::Deadline(
        DeadlinePhase::ResponseHeaders,
    )));
    assert!(!retryable_before_acceptance(
        &WorkerClientError::HttpStatus {
            status: "503".parse().unwrap(),
            body: "generic".into(),
        },
    ));

    let mut invalid_accepted = match rejection(RejectionCode::CapacityExhausted) {
        WorkerClientError::Rejected { rejection } => rejection,
        _ => unreachable!(),
    };
    invalid_accepted.accepted = true;
    assert!(!retryable_before_acceptance(&WorkerClientError::Rejected {
        rejection: invalid_accepted,
    },));
}

#[test]
fn retry_backoff_honors_advice_with_bounded_stable_jitter() {
    let rejection = |retry_after_ms| {
        let mut rejection = InvocationRejection::new(
            id("request-1"),
            id("attempt-1"),
            RejectionCode::CapacityExhausted,
            "test",
        );
        rejection.retry_after_ms = retry_after_ms;
        WorkerClientError::Rejected { rejection }
    };

    let default_delay = retry_delay(&rejection(None), "request-1");
    assert!(default_delay >= Duration::from_millis(RETRY_BACKOFF_BASE_MS));
    assert!(
        default_delay <= Duration::from_millis(RETRY_BACKOFF_BASE_MS + RETRY_BACKOFF_JITTER_MS)
    );
    assert_eq!(
        default_delay,
        retry_delay(&rejection(None), "request-1"),
        "one request must use a stable retry delay"
    );

    let advised = retry_delay(&rejection(Some(95)), "request-1");
    assert!(advised >= Duration::from_millis(95));
    assert!(advised <= Duration::from_millis(RETRY_BACKOFF_MAX_MS));
    assert_eq!(
        retry_delay(&rejection(Some(u64::MAX)), "request-1"),
        Duration::from_millis(RETRY_BACKOFF_MAX_MS),
        "untrusted worker advice must be clamped"
    );
}

#[tokio::test]
async fn worker_receives_the_resolved_sampling_profile() {
    let worker =
        ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

    // Qwen3.5-4B thinks by default; its published profile is T 1.0 and
    // top-p 0.95, never the worker's greedy default. Stop strings travel
    // with it (the worker applies them in the engine).
    let mut request = request();
    request.stop = vec!["END".into()];
    dispatcher(registry)
        .generate(2, &context, request, tenant_work())
        .await
        .unwrap();
    let requests = worker.requests();
    let InvocationInput::Chat { parameters, .. } = &requests[0].input;
    assert_eq!(parameters.temperature, Some(1.0));
    assert_eq!(parameters.top_p, Some(0.95));
    assert_eq!(parameters.stop, ["END"]);
}

/// DP-F2: every control a request sets reaches a worker that advertises
/// it, with the negotiated version and the D8 context policy.
#[tokio::test]
async fn worker_receives_every_control_it_advertises() {
    let worker =
        ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    dispatcher(registry)
        .generate(2, &context, every_control(), tenant_work())
        .await
        .unwrap();
    let requests = worker.requests();
    assert_eq!(requests[0].schema_version, PROTOCOL_V1);
    let InvocationInput::Chat { parameters, .. } = &requests[0].input;
    assert_eq!(
        *parameters,
        WorkerChatParameters {
            temperature: Some(1.0),
            top_p: Some(0.95),
            seed: Some(7),
            stop: vec!["END".into()],
            top_k: Some(9),
            repetition_penalty: Some(1.25),
            presence_penalty: Some(0.5),
            logprobs: Some(2),
            response_format: Some(ChatResponseFormat::JsonObject),
            context: Some(ChatContextPolicy {
                max_tokens: Some(32),
                truncation: WorkerChatTruncation::Disabled,
            }),
            tools: vec![serde_json::json!({"type": "function", "function": {"name": "now"}})],
            skip_special_tokens: Some(false),
        }
    );
}

fn every_control() -> ChatExecutionRequest {
    let mut request = request();
    request.seed = Some(7);
    request.stop = vec!["END".into()];
    request.top_k = Some(9);
    request.repetition_penalty = Some(1.25);
    request.presence_penalty = Some(0.5);
    request.logprobs = Some(true);
    request.top_logprobs = Some(2);
    request.response_format_json_object = true;
    request.skip_special_tokens = Some(false);
    request.chat_config.tools =
        vec![serde_json::json!({"type": "function", "function": {"name": "now"}})];
    request
}

/// A control goes only to a worker that advertises it: a mixed fleet
/// routes it to the newer worker, and a fleet without one refuses it
/// with a 400 instead of letting a worker drop it. A plain request still
/// runs on an older worker, at its minor and without the context policy.
#[tokio::test]
async fn controls_route_only_to_workers_that_advertise_them() {
    let older = ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
    let newer = ScriptedWorker::spawn(ScriptedResponse::Success("ok".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register_minor3(&registry, "worker-a", "inc-a", client(&older.endpoint()));
    register(&registry, "worker-b", "inc-b", client(&newer.endpoint()), 1);
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    dispatcher(registry)
        .generate(2, &context, every_control(), tenant_work())
        .await
        .unwrap();
    assert!(older.requests().is_empty());
    assert_eq!(newer.requests().len(), 1);

    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register_minor3(&registry, "worker-a", "inc-a", client(&older.endpoint()));
    let dispatcher = dispatcher(registry);
    let cases: [(&str, fn(&mut ChatExecutionRequest)); 8] = [
        ("seed", |request| request.seed = Some(7)),
        ("stop", |request| request.stop = vec!["END".into()]),
        ("top_k", |request| request.top_k = Some(9)),
        ("presence_penalty", |request| {
            request.presence_penalty = Some(0.5)
        }),
        ("logprobs", |request| request.logprobs = Some(true)),
        ("response_format", |request| {
            request.response_format_json_object = true
        }),
        ("tools", |request| {
            request.chat_config.tools = vec![serde_json::json!({"type": "function"})]
        }),
        ("skip_special_tokens", |request| {
            request.skip_special_tokens = Some(false)
        }),
    ];
    for (parameter, apply) in cases {
        let mut unsupported = request();
        apply(&mut unsupported);
        let error = dispatcher
            .generate(2, &context, unsupported, tenant_work())
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST, "{parameter}");
        assert_eq!(
            error.detail.and_then(|detail| detail.param),
            Some(parameter)
        );
        assert_eq!(
            error.detail.and_then(|detail| detail.code),
            Some("unsupported_parameter")
        );
    }
    assert!(older.requests().is_empty());

    // An explicit `logprobs: false` or `skip_special_tokens: true` asks
    // for nothing the worker lacks.
    let mut plain = request();
    plain.logprobs = Some(false);
    plain.skip_special_tokens = Some(true);
    dispatcher
        .generate(2, &context, plain, tenant_work())
        .await
        .unwrap();
    let requests = older.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].schema_version, SchemaVersion::new(1, 3));
    let InvocationInput::Chat { parameters, .. } = &requests[0].input;
    assert_eq!(parameters.context, None);
    assert_eq!(parameters.skip_special_tokens, None);
    assert_eq!(parameters.temperature, Some(1.0));
}

#[tokio::test]
async fn unknown_admission_is_a_busy_worker_not_a_circuit_strike() {
    let worker = ScriptedWorker::spawn(ScriptedResponse::AdmissionUnknown, Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let dispatcher = dispatcher(registry);
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

    // More slow admissions than the circuit's failure threshold (3).
    for _ in 0..4 {
        let error = dispatcher
            .generate(2, &context, request(), tenant_work())
            .await
            .unwrap_err();
        assert_eq!(
            error.status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{}",
            error.message
        );
    }
    // An unknown outcome is never retried elsewhere, and the circuit
    // stays closed, so every request still reached the worker.
    assert_eq!(worker.requests().len(), 4);
}

#[tokio::test]
async fn connection_not_established_retries_once_for_streaming() {
    let worker_b =
        ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&refused_endpoint().await),
        1,
    );
    register(
        &registry,
        "worker-b",
        "inc-b",
        client(&worker_b.endpoint()),
        1,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

    let mut events = dispatcher(registry)
        .stream(2, &context, request(), tenant_work())
        .await
        .unwrap();
    assert!(matches!(
        events.recv().await,
        Some(ChatStreamEvent::Started)
    ));
    assert!(matches!(
        events.recv().await,
        Some(ChatStreamEvent::Delta { text, .. }) if text == "from-b"
    ));
    assert!(matches!(
        events.recv().await,
        Some(ChatStreamEvent::Completed(_))
    ));
    assert_eq!(worker_b.requests().len(), 1);
}

/// A worker burst three times the relay channel's capacity.
const BURST_DELTAS: usize = 192;

#[tokio::test]
async fn paused_public_reader_receives_every_worker_delta() {
    let count = BURST_DELTAS;
    let worker = ScriptedWorker::spawn(ScriptedResponse::ManyDeltas(count), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let mut public_events = dispatcher(registry)
        .stream(
            2,
            &RequestContext::new("test-request".into(), Principal::local_anonymous()),
            request(),
            tenant_work(),
        )
        .await
        .unwrap();

    // The worker's burst outruns both relay channels while the reader
    // pauses for less than the 100 ms slow-consumer timeout.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let mut text = String::new();
    let terminal = loop {
        match public_events.recv().await {
            Some(ChatStreamEvent::Delta { text: delta, .. }) => text.push_str(&delta),
            Some(ChatStreamEvent::Started) => {}
            terminal => break terminal,
        }
    };
    assert!(
        matches!(terminal, Some(ChatStreamEvent::Completed(_))),
        "{terminal:?}"
    );
    assert_eq!(text, "x".repeat(count));
}

#[tokio::test]
async fn non_draining_public_stream_cannot_retain_registry_dispatch() {
    let worker =
        ScriptedWorker::spawn(ScriptedResponse::ManyDeltas(BURST_DELTAS), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        max_local_dispatches: 1,
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let dispatcher = dispatcher(registry.clone());
    let selection = dispatcher.selection_request(&request(), true).unwrap();

    let mut public_events = dispatcher
        .stream(
            2,
            &RequestContext::new("test-request".into(), Principal::local_anonymous()),
            request(),
            tenant_work(),
        )
        .await
        .unwrap();
    let replacement = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            match registry.select_and_reserve(&selection) {
                Ok(selected) => break selected,
                Err(WorkerRegistryError::LocalDispatchLimitReached) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => panic!("unexpected replacement selection failure: {error}"),
            }
        }
    })
    .await
    .expect("slow-consumer forwarding timeout must release dispatch ownership");
    drop(replacement);

    let mut saw_completion = false;
    tokio::time::timeout(Duration::from_secs(1), async {
        while let Some(event) = public_events.recv().await {
            saw_completion |= matches!(event, ChatStreamEvent::Completed(_));
        }
    })
    .await
    .expect("timed-out bridge must close its public event channel");
    assert!(
        !saw_completion,
        "a truncated slow-consumer stream must never fabricate completion"
    );
    // `render_chat_stream` maps this non-terminal channel closure to an
    // explicit SSE error before [DONE]; its dedicated route-level test
    // covers that public encoding contract.
}

#[tokio::test]
async fn capacity_rejection_retries_with_same_request_and_fresh_attempt() {
    let worker_a = ScriptedWorker::spawn(
        ScriptedResponse::Reject(RejectionCode::CapacityExhausted),
        Duration::from_millis(25),
    )
    .await;
    let worker_b =
        ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker_a.endpoint()),
        1,
    );
    register(
        &registry,
        "worker-b",
        "inc-b",
        client(&worker_b.endpoint()),
        1,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

    let generation = dispatcher(registry)
        .generate(2, &context, request(), tenant_work())
        .await
        .unwrap();
    assert_eq!(generation.text, "from-b");
    let first = worker_a.requests();
    let alternate = worker_b.requests();
    assert_eq!(first.len(), 1);
    assert_eq!(alternate.len(), 1);
    assert_eq!(first[0].request_id, alternate[0].request_id);
    assert_ne!(first[0].attempt_id, alternate[0].attempt_id);
    assert!(alternate[0].remaining_time_ms < first[0].remaining_time_ms);
}

async fn fleet_coordinator() -> (
    Arc<crate::app::fleet_coordinator::FleetCoordinator>,
    tempfile::TempDir,
) {
    let root = tempfile::tempdir().unwrap();
    let store = FleetStore::new(StoreDatabase::new(root.path().join("fleet.sqlite3")));
    (
        Arc::new(
            crate::app::fleet_coordinator::FleetCoordinator::new(store, "gateway-test".to_string())
                .with_claim_timeout(crate::app::fleet_coordinator::TEST_FLEET_CLAIM_TIMEOUT)
                .opened()
                .await,
        ),
        root,
    )
}

#[tokio::test]
async fn fleet_claim_is_published_and_released_across_generate() {
    let worker = ScriptedWorker::spawn(
        ScriptedResponse::Success("from-fleet".into()),
        Duration::ZERO,
    )
    .await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let (coordinator, _root) = fleet_coordinator().await;
    let store = coordinator.store();
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    let generation = dispatcher(registry)
        .with_fleet_coordinator(Arc::clone(&coordinator))
        .generate(2, &context, request(), tenant_work())
        .await
        .unwrap();
    assert_eq!(generation.text, "from-fleet");
    for _ in 0..200 {
        if store.count_live_fleet_claims("worker-a").await.unwrap() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        store.count_live_fleet_claims("worker-a").await.unwrap(),
        0,
        "the dispatch claim guard must release its cluster claim when the request completes"
    );
}

#[tokio::test]
async fn accepted_work_releases_its_fleet_claim_so_free_credits_stay_dispatchable() {
    // Four credits, two accepted in-flight streams from gateway A: the
    // worker reports two free credits, which must not be shadowed again by
    // the accepted streams' claims, for A or for its peer gateway B.
    let worker = ScriptedWorker::spawn(ScriptedResponse::AcceptThenHold, Duration::ZERO).await;
    let root = tempfile::tempdir().unwrap();
    let fleet_db = root.path().join("fleet.sqlite3");
    let gateway = |gateway_id: &str| {
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register_with_credits(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker.endpoint()),
            1,
            4,
        );
        let coordinator = crate::app::fleet_coordinator::FleetCoordinator::new(
            FleetStore::new(StoreDatabase::new(fleet_db.clone())),
            gateway_id.to_string(),
        )
        .with_claim_timeout(crate::app::fleet_coordinator::TEST_FLEET_CLAIM_TIMEOUT);
        (registry, coordinator)
    };
    let open = |(registry, coordinator): (
        WorkerRegistry,
        crate::app::fleet_coordinator::FleetCoordinator,
    )| async move {
        let coordinator = Arc::new(coordinator.opened().await);
        let dispatcher =
            dispatcher(registry.clone()).with_fleet_coordinator(Arc::clone(&coordinator));
        (registry, coordinator, dispatcher)
    };
    let (registry_a, coordinator_a, dispatcher_a) = open(gateway("gateway-a")).await;
    let (registry_b, coordinator_b, dispatcher_b) = open(gateway("gateway-b")).await;
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    let mut held = Vec::new();
    for _ in 0..2 {
        held.push(
            dispatcher_a
                .stream(2, &context, request(), tenant_work())
                .await
                .unwrap(),
        );
    }
    // Guards release on a spawned task; give a released claim time to
    // land before the polls sample the store. A claim held to the end of
    // the stream never lands, and the dispatches below then shed.
    let store = coordinator_a.store();
    for _ in 0..200 {
        if store.count_live_fleet_claims("worker-a").await.unwrap() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // Both gateways' next status polls report the two accepted streams.
    let status = worker_status("worker-a", "inc-a", 2, 4, 2);
    for (registry, coordinator) in [(&registry_a, &coordinator_a), (&registry_b, &coordinator_b)] {
        registry.observe_status(status.clone()).unwrap();
        coordinator.refresh_peer_claims(&status).await;
    }

    for (name, dispatcher) in [
        ("gateway A", &dispatcher_a),
        ("peer gateway B", &dispatcher_b),
    ] {
        let next = dispatcher
            .stream(2, &context, request(), tenant_work())
            .await;
        assert!(
            next.is_ok(),
            "{name} must still dispatch to the two free credits: {:?}",
            next.err()
        );
        held.push(next.unwrap());
    }
    assert_eq!(worker.requests().len(), 4);
}

#[tokio::test]
async fn hung_fleet_store_costs_at_most_the_claim_timeout() {
    let worker = ScriptedWorker::spawn(
        ScriptedResponse::Success("from-fleet".into()),
        Duration::ZERO,
    )
    .await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let root = tempfile::tempdir().unwrap();
    let mut store = FleetStore::new(StoreDatabase::new(root.path().join("fleet.sqlite3")));
    let faults = Arc::new(crate::gateway_fleet::FleetFaults::default());
    store.set_faults_for_test(Arc::clone(&faults));
    faults.hang_fleet_calls();
    let coordinator = Arc::new(
        crate::app::fleet_coordinator::FleetCoordinator::new(store, "gateway-test".into())
            .with_claim_timeout(Duration::from_millis(50))
            .opened()
            .await,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    // The worker stays the admission arbiter: a store that never answers
    // the claim must not hold the request on its time-to-first-token.
    let generation = tokio::time::timeout(
        Duration::from_secs(2),
        dispatcher(registry)
            .with_fleet_coordinator(coordinator)
            .generate(2, &context, request(), tenant_work()),
    )
    .await
    .expect("a hung coordination store must not stall dispatch")
    .unwrap();
    assert_eq!(generation.text, "from-fleet");
    faults.answer_fleet_calls();
}

#[tokio::test]
async fn fleet_claim_loss_still_invokes_and_succeeds() {
    let worker = ScriptedWorker::spawn(
        ScriptedResponse::Success("from-fleet".into()),
        Duration::ZERO,
    )
    .await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker.endpoint()),
        1,
    );
    let (coordinator, _root) = fleet_coordinator().await;
    // A peer holds the only observable credit, but this gateway's cached
    // snapshot is stale (empty): selection proceeds, the claim loses, and
    // the worker arbitrates the invocation to success.
    let peer = coordinator
        .claim(
            &crate::worker_registry::WorkerInstanceKey {
                worker_id: id::<WorkerId>("worker-a"),
                incarnation_id: id::<IncarnationId>("inc-a"),
            },
            1,
        )
        .await
        .expect("peer claim");
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());
    let generation = dispatcher(registry)
        .with_fleet_coordinator(Arc::clone(&coordinator))
        .generate(2, &context, request(), tenant_work())
        .await
        .unwrap();
    assert_eq!(generation.text, "from-fleet");
    drop(peer);
}

#[tokio::test]
async fn uncertain_acceptance_and_partial_output_never_retry() {
    for first_response in [
        ScriptedResponse::AcceptedWithoutAcknowledgement,
        ScriptedResponse::PartialThenDisconnect,
    ] {
        let worker_a = ScriptedWorker::spawn(first_response, Duration::ZERO).await;
        let worker_b = ScriptedWorker::spawn(
            ScriptedResponse::Success("must-not-run".into()),
            Duration::ZERO,
        )
        .await;
        let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
        register(
            &registry,
            "worker-a",
            "inc-a",
            client(&worker_a.endpoint()),
            1,
        );
        register(
            &registry,
            "worker-b",
            "inc-b",
            client(&worker_b.endpoint()),
            1,
        );
        let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

        let error = dispatcher(registry)
            .generate(2, &context, request(), tenant_work())
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(worker_b.requests().is_empty());
    }
}

#[tokio::test]
async fn retry_is_bounded_to_one_alternate() {
    let worker_c = ScriptedWorker::spawn(
        ScriptedResponse::Success("must-not-run".into()),
        Duration::ZERO,
    )
    .await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&refused_endpoint().await),
        1,
    );
    register(
        &registry,
        "worker-b",
        "inc-b",
        client(&refused_endpoint().await),
        1,
    );
    register(
        &registry,
        "worker-c",
        "inc-c",
        client(&worker_c.endpoint()),
        1,
    );
    let context = RequestContext::new("test-request".into(), Principal::local_anonymous());

    let error = dispatcher(registry)
        .generate(2, &context, request(), tenant_work())
        .await
        .unwrap_err();
    assert_eq!(error.status, StatusCode::BAD_GATEWAY);
    assert!(worker_c.requests().is_empty());
}

fn conversation_turn(system: &str, turns: &[(&str, Option<&str>)]) -> ChatExecutionRequest {
    let mut messages = vec![ChatMessage {
        role: ChatRole::System,
        content: system.into(),
    }];
    for (user, assistant) in turns {
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: (*user).into(),
        });
        if let Some(assistant) = assistant {
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: (*assistant).into(),
            });
        }
    }
    ChatExecutionRequest {
        variant: ModelVariant::Qwen354BGguf,
        messages,
        max_completion_tokens: None,
        max_tokens: Some(32),
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        presence_penalty: None,
        seed: None,
        stop: Vec::new(),
        logprobs: None,
        top_logprobs: None,
        response_format_json_object: false,
        skip_special_tokens: None,
        chat_config: ChatRequestConfig::default(),
        correlation_id: None,
    }
}

fn turn_one() -> ChatExecutionRequest {
    conversation_turn(
        "You are a terse bearologist.",
        &[("Tell me about bears.", None)],
    )
}

fn turn_two() -> ChatExecutionRequest {
    conversation_turn(
        "You are a terse bearologist.",
        &[("Tell me about bears.", Some("Bears are large mammals."))],
    )
}

async fn two_ready_workers(
    routing: RoutingConfig,
) -> (ScriptedWorker, ScriptedWorker, WorkerRegistry) {
    let worker_a =
        ScriptedWorker::spawn(ScriptedResponse::Success("from-a".into()), Duration::ZERO).await;
    let worker_b =
        ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig {
        routing,
        ..WorkerRegistryConfig::default()
    })
    .unwrap();
    for (name, incarnation, worker) in [
        ("worker-a", "inc-a", &worker_a),
        ("worker-b", "inc-b", &worker_b),
    ] {
        register_with_credits(
            &registry,
            name,
            incarnation,
            client(&worker.endpoint()),
            1,
            4,
        );
    }
    (worker_a, worker_b, registry)
}

fn small_blocks() -> Option<PrefixAffinityConfig> {
    Some(PrefixAffinityConfig {
        block_bytes: 16,
        ..PrefixAffinityConfig::default()
    })
}

#[tokio::test]
async fn conversation_turns_follow_the_worker_holding_their_prefix() {
    let (worker_a, worker_b, registry) = two_ready_workers(RoutingConfig::default()).await;
    let dispatcher = dispatcher_with_prefix(registry.clone(), small_blocks());
    let context = RequestContext::new("prefix-request".into(), Principal::local_anonymous());

    // worker-a is busy, so turn one goes to the idle worker-b.
    let busy_a = registry
        .select_and_reserve(&dispatcher.selection_request(&turn_one(), false).unwrap())
        .unwrap();
    assert_eq!(busy_a.key.worker_id.as_str(), "worker-a");
    dispatcher
        .generate(2, &context, turn_one(), tenant_work())
        .await
        .unwrap();
    assert_eq!(worker_b.requests().len(), 1);
    drop(busy_a);

    // Both are idle again and worker-a holds the identity tie-break, but
    // turn two shares turn one's prefix, which worker-b holds.
    dispatcher
        .generate(2, &context, turn_two(), tenant_work())
        .await
        .unwrap();
    assert_eq!(worker_b.requests().len(), 2);
    assert!(worker_a.requests().is_empty());

    // A different conversation has no held prefix: load decides.
    dispatcher
        .generate(
            2,
            &context,
            conversation_turn("Ornithologist on duty.", &[("Owls?", None)]),
            tenant_work(),
        )
        .await
        .unwrap();
    assert_eq!(worker_a.requests().len(), 1);
}

#[tokio::test]
async fn prefix_affinity_spills_to_the_least_loaded_worker_past_the_balance_threshold() {
    for (threshold, expected) in [(1, "worker-a"), (0, "worker-b")] {
        let (worker_a, worker_b, registry) = two_ready_workers(RoutingConfig {
            prefix_balance_threshold: threshold,
            ..RoutingConfig::default()
        })
        .await;
        let dispatcher = dispatcher_with_prefix(registry.clone(), small_blocks());
        let context = RequestContext::new("prefix-spill".into(), Principal::local_anonymous());
        dispatcher
            .generate(2, &context, turn_one(), tenant_work())
            .await
            .unwrap();
        assert_eq!(worker_a.requests().len(), 1);

        // One more request in flight on the prefix holder.
        let busy_a = registry
            .select_and_reserve(&dispatcher.selection_request(&turn_one(), false).unwrap())
            .unwrap();
        assert_eq!(busy_a.key.worker_id.as_str(), "worker-a");
        dispatcher
            .generate(2, &context, turn_two(), tenant_work())
            .await
            .unwrap();
        let served_by = if worker_a.requests().len() == 2 {
            "worker-a"
        } else {
            assert_eq!(worker_b.requests().len(), 1);
            "worker-b"
        };
        assert_eq!(served_by, expected, "threshold {threshold}");
    }
}

#[tokio::test]
async fn prefix_recorded_through_failover_steers_the_next_turn() {
    let worker_a = ScriptedWorker::spawn(
        ScriptedResponse::Reject(RejectionCode::CapacityExhausted),
        Duration::ZERO,
    )
    .await;
    let worker_b =
        ScriptedWorker::spawn(ScriptedResponse::Success("from-b".into()), Duration::ZERO).await;
    let registry = WorkerRegistry::new(WorkerRegistryConfig::default()).unwrap();
    // worker-a refuses for capacity, so turn one fails over to worker-b,
    // which must record the prefix through the alternate-dispatch path.
    register(
        &registry,
        "worker-a",
        "inc-a",
        client(&worker_a.endpoint()),
        1,
    );
    register(
        &registry,
        "worker-b",
        "inc-b",
        client(&worker_b.endpoint()),
        1,
    );
    let dispatcher = dispatcher_with_prefix(registry, small_blocks());
    let context = RequestContext::new("prefix-failover".into(), Principal::local_anonymous());

    dispatcher
        .generate(2, &context, turn_one(), tenant_work())
        .await
        .unwrap();
    assert_eq!(worker_a.requests().len(), 1);
    assert_eq!(worker_b.requests().len(), 1);

    // The second turn goes straight to worker-b instead of first trying
    // the identity-preferred worker-a again.
    dispatcher
        .generate(2, &context, turn_two(), tenant_work())
        .await
        .unwrap();
    assert_eq!(worker_a.requests().len(), 1);
    assert_eq!(worker_b.requests().len(), 2);
}

//! Deterministic, accelerator-free worker used by real-socket contract tests.

use async_stream::stream;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_serving_protocol::*;
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit, Semaphore};

pub const DEFAULT_MOCK_REQUEST_LIMIT: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockFault {
    None,
    Hang,
    UsageTrickleWithoutOutput,
    ByteTrickleWithoutEvent,
    EmptyDeltaTrickleWithoutOutput,
    TextDeltaThenHang,
    ManyTextDeltas { count: usize, text_bytes: usize },
    OpenBodyWithoutAcknowledgement,
    AcceptedWithoutAcknowledgement,
    AcceptedThenDisconnect,
    PartialThenDisconnect,
    MalformedEvent,
    OversizedEvent { text_bytes: usize },
}

#[derive(Debug, Clone)]
pub struct MockWorkerConfig {
    pub worker_id: WorkerId,
    pub node_id: NodeId,
    pub incarnation_id: IncarnationId,
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub model_generation: ModelGeneration,
    pub credentials: ServiceCredentials,
    pub max_active_invocations: usize,
    pub max_request_bytes: usize,
    pub max_retained_attempts: usize,
    pub output_text: String,
    pub output_cadence: Duration,
    pub cancellation_delay: Duration,
    pub fault: MockFault,
    pub ready: bool,
}

impl Default for MockWorkerConfig {
    fn default() -> Self {
        fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
        where
            T::Error: std::fmt::Debug,
        {
            T::try_from(value).expect("static mock identity")
        }
        Self {
            worker_id: id("mock-worker-1"),
            node_id: id("mock-node-1"),
            incarnation_id: id("mock-incarnation-1"),
            deployment_id: id("mock-chat-v1"),
            public_model: id("mock-chat"),
            model_generation: ModelGeneration::new(1).expect("non-zero"),
            credentials: ServiceCredentials {
                credential_id: id("mock-credential-1"),
                bearer_token: ServiceBearerToken::new("mock-secret-token")
                    .expect("static mock token"),
            },
            max_active_invocations: 1,
            max_request_bytes: DEFAULT_MOCK_REQUEST_LIMIT,
            max_retained_attempts: 64,
            output_text: "deterministic mock response".into(),
            output_cadence: Duration::from_millis(5),
            cancellation_delay: Duration::from_millis(25),
            fault: MockFault::None,
            ready: true,
        }
    }
}

impl MockWorkerConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_active_invocations == 0 {
            return Err("max_active_invocations must be non-zero");
        }
        if self.max_request_bytes == 0 {
            return Err("max_request_bytes must be non-zero");
        }
        if self.max_retained_attempts < self.max_active_invocations {
            return Err("attempt retention must cover every active invocation");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AttemptRecord {
    identity: AttemptIdentity,
    digest: RequestDigest,
    state: AttemptState,
    last_sequence: Option<u64>,
    remaining_time_ms: u64,
    cancel: Option<watch::Sender<bool>>,
}

#[derive(Debug, Default)]
struct AttemptTable {
    records: HashMap<AttemptId, AttemptRecord>,
    order: VecDeque<AttemptId>,
}

struct MockState {
    config: MockWorkerConfig,
    capacity: Arc<Semaphore>,
    attempts: Mutex<AttemptTable>,
    status_sequence: AtomicU64,
}

impl MockState {
    fn authenticate(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get(SERVICE_AUTHORIZATION_HEADER)
            .and_then(|value| value.to_str().ok());
        let credential = headers
            .get(SERVICE_CREDENTIAL_ID_HEADER)
            .and_then(|value| value.to_str().ok());
        let presented_token = bearer.and_then(|value| {
            value
                .strip_prefix(SERVICE_AUTH_SCHEME)
                .and_then(|value| value.strip_prefix(' '))
        });
        credential == Some(self.config.credentials.credential_id.as_str())
            && presented_token.is_some_and(|token| {
                self.config
                    .credentials
                    .bearer_token
                    .matches_presented(token)
            })
    }

    fn update_attempt(&self, attempt_id: &AttemptId, state: AttemptState, sequence: Option<u64>) {
        let mut table = self.attempts.lock().expect("mock attempt table poisoned");
        if let Some(record) = table.records.get_mut(attempt_id) {
            record.state = state;
            if sequence.is_some() {
                record.last_sequence = sequence;
            }
            if state.is_terminal() {
                record.cancel = None;
            }
        }
    }

    fn next_sequence(&self, attempt_id: &AttemptId) -> u64 {
        self.attempts
            .lock()
            .expect("mock attempt table poisoned")
            .records
            .get(attempt_id)
            .and_then(|record| record.last_sequence)
            .map_or(0, |sequence| sequence.saturating_add(1))
    }

    fn insert_bounded(&self, attempt_id: AttemptId, record: AttemptRecord) -> bool {
        let mut table = self.attempts.lock().expect("mock attempt table poisoned");
        if table.records.contains_key(&attempt_id) {
            return false;
        }
        while table.records.len() >= self.config.max_retained_attempts {
            let removable = table.order.iter().position(|id| {
                table
                    .records
                    .get(id)
                    .is_some_and(|record| record.state.is_terminal() || record.cancel.is_none())
            });
            let Some(index) = removable else {
                return false;
            };
            let evicted = table.order.remove(index).expect("known retention index");
            table.records.remove(&evicted);
        }
        table.order.push_back(attempt_id.clone());
        table.records.insert(attempt_id, record);
        true
    }
}

/// A real TCP mock worker. Dropping it aborts the listener; active execution tasks remain governed
/// by their own permits until the Tokio runtime shuts down.
pub struct MockWorker {
    address: SocketAddr,
    state: Arc<MockState>,
    server: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for MockWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MockWorker")
            .field("address", &self.address)
            .field("config", &self.state.config)
            .finish_non_exhaustive()
    }
}

impl MockWorker {
    pub async fn spawn(config: MockWorkerConfig) -> Result<Self, std::io::Error> {
        config.validate().map_err(std::io::Error::other)?;
        let max_request_bytes = config.max_request_bytes;
        let state = Arc::new(MockState {
            capacity: Arc::new(Semaphore::new(config.max_active_invocations)),
            attempts: Mutex::new(AttemptTable::default()),
            status_sequence: AtomicU64::new(0),
            config,
        });
        let router = Router::new()
            .route(WORKER_DESCRIPTOR_PATH, get(descriptor))
            .route(WORKER_STATUS_PATH, get(status))
            .route(INVOCATIONS_PATH, post(invoke))
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}"),
                get(query_attempt),
            )
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}/cancel"),
                post(cancel_attempt),
            )
            .layer(DefaultBodyLimit::max(max_request_bytes))
            .with_state(Arc::clone(&state));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        Ok(Self {
            address,
            state,
            server,
        })
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}/", self.address)
    }

    pub fn config(&self) -> &MockWorkerConfig {
        &self.state.config
    }

    pub fn active_invocations(&self) -> usize {
        self.state.config.max_active_invocations - self.state.capacity.available_permits()
    }

    pub fn attempt_remaining_time_ms(&self, attempt_id: &AttemptId) -> Option<u64> {
        self.state
            .attempts
            .lock()
            .expect("mock attempt table poisoned")
            .records
            .get(attempt_id)
            .map(|record| record.remaining_time_ms)
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn descriptor(State(state): State<Arc<MockState>>, headers: HeaderMap) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(WorkerDescriptor {
        schema_version: PROTOCOL_V1,
        supported_protocol_versions: vec![PROTOCOL_V1],
        worker_id: state.config.worker_id.clone(),
        node_id: state.config.node_id.clone(),
        incarnation_id: state.config.incarnation_id.clone(),
        build_version: "deterministic-mock-v1".into(),
        assignment: DeviceAssignment::Cpu {
            thread_budget: 1,
            affinity: Vec::new(),
            host_memory_limit_bytes: 64 * 1024 * 1024,
        },
        features: BTreeSet::from([
            WorkerFeature::Streaming,
            WorkerFeature::Cancellation,
            WorkerFeature::AttemptQuery,
        ]),
    })
    .into_response()
}

async fn status(State(state): State<Arc<MockState>>, headers: HeaderMap) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let active = state.config.max_active_invocations - state.capacity.available_permits();
    Json(WorkerStatus {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.worker_id.clone(),
        node_id: state.config.node_id.clone(),
        incarnation_id: state.config.incarnation_id.clone(),
        status_sequence: state.status_sequence.fetch_add(1, Ordering::Relaxed) + 1,
        process_state: WorkerProcessState::Running,
        deployments: vec![deployment(&state.config)],
        capacity: CapacitySnapshot {
            max_active_invocations: state.config.max_active_invocations as u32,
            active_invocations: active as u32,
            max_queued_invocations: 0,
            queued_invocations: 0,
            max_sessions: 0,
            reserved_sessions: 0,
            available_admission_credits: state.capacity.available_permits() as u32,
            outstanding_cost_units: active as u64,
        },
    })
    .into_response()
}

fn deployment(config: &MockWorkerConfig) -> LoadedDeployment {
    LoadedDeployment {
        deployment_id: config.deployment_id.clone(),
        public_model: config.public_model.clone(),
        artifact_revision: ArtifactRevision::new("mock-artifact-v1").expect("static identity"),
        model_generation: config.model_generation,
        task: TaskKind::Chat,
        backend: BackendKind::Cpu,
        precision: "mock".into(),
        execution_representation: "deterministic-text".into(),
        tokenizer_revision: None,
        readiness: if config.ready {
            ModelReadiness::Ready
        } else {
            ModelReadiness::Loading
        },
        capability: Capability {
            task: TaskKind::Chat,
            streaming: true,
            realtime: false,
            cancellation: CancellationBehavior::Cooperative,
            accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
            output_formats: BTreeSet::from([OutputFormat::Text]),
            max_input_bytes: config.max_request_bytes as u64,
            max_context_tokens: Some(4096),
            max_output_tokens: Some(1024),
        },
    }
}

async fn invoke(State(state): State<Arc<MockState>>, headers: HeaderMap, body: Bytes) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let request: InvocationRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    if request.schema_version.major != PROTOCOL_V1.major {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::UnsupportedProtocolVersion,
            "unsupported protocol major version",
        );
    }
    if let Err(error) = request.validate() {
        return rejection(
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            error.to_string(),
        );
    }
    if request.expected_worker_incarnation != state.config.incarnation_id {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongWorkerIncarnation,
            "worker incarnation changed",
        );
    }
    if request.deployment_id != state.config.deployment_id {
        return rejection(
            &request,
            StatusCode::NOT_FOUND,
            RejectionCode::UnknownDeployment,
            "deployment is not loaded",
        );
    }
    if request.expected_model_generation != state.config.model_generation {
        return rejection(
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongModelGeneration,
            "model generation changed",
        );
    }
    if request.task != TaskKind::Chat {
        return rejection(
            &request,
            StatusCode::UNPROCESSABLE_ENTITY,
            RejectionCode::IncompatibleTask,
            "task is incompatible with deployment",
        );
    }
    if !state.config.ready {
        return rejection(
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::ModelNotReady,
            "deployment is not ready",
        );
    }

    {
        let table = state.attempts.lock().expect("mock attempt table poisoned");
        if let Some(existing) = table.records.get(&request.attempt_id) {
            let same = existing.identity.request_id == request.request_id
                && existing.digest == request.request_digest;
            return rejection(
                &request,
                StatusCode::CONFLICT,
                if same {
                    RejectionCode::CapacityExhausted
                } else {
                    RejectionCode::DuplicateAttemptConflict
                },
                if same {
                    "attempt is already owned by this worker"
                } else {
                    "attempt identity was reused with different content"
                },
            );
        }
    }

    let permit = match Arc::clone(&state.capacity).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return rejection(
                &request,
                StatusCode::TOO_MANY_REQUESTS,
                RejectionCode::CapacityExhausted,
                "worker capacity is exhausted",
            )
        }
    };
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let record = AttemptRecord {
        identity: AttemptIdentity::from(&request),
        digest: request.request_digest.clone(),
        state: AttemptState::Admitted,
        last_sequence: Some(0),
        remaining_time_ms: request.remaining_time_ms,
        cancel: Some(cancel_tx),
    };
    if !state.insert_bounded(request.attempt_id.clone(), record) {
        drop(permit);
        return rejection(
            &request,
            StatusCode::TOO_MANY_REQUESTS,
            RejectionCode::CapacityExhausted,
            "attempt retention is exhausted",
        );
    }

    let (tx, mut rx) = mpsc::channel::<Bytes>(8);
    tokio::spawn(run_invocation(
        Arc::clone(&state),
        request,
        permit,
        cancel_rx,
        tx,
    ));
    let output = stream! {
        while let Some(bytes) = rx.recv().await {
            yield Ok::<Bytes, Infallible>(bytes);
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
        .body(Body::from_stream(output))
        .expect("static mock response")
}

async fn run_invocation(
    state: Arc<MockState>,
    request: InvocationRequest,
    _permit: OwnedSemaphorePermit,
    mut cancel: watch::Receiver<bool>,
    tx: mpsc::Sender<Bytes>,
) {
    if matches!(
        &state.config.fault,
        MockFault::AcceptedWithoutAcknowledgement | MockFault::OpenBodyWithoutAcknowledgement
    ) {
        // Admission is authoritative even when the acknowledgement never reaches the gateway.
        // Keep or close the body according to the selected fault while retaining execution
        // capacity until exact-attempt cancellation completes.
        state.update_attempt(&request.attempt_id, AttemptState::Running, Some(0));
        let _open_response_body =
            (state.config.fault == MockFault::OpenBodyWithoutAcknowledgement).then_some(tx);
        let terminal_state = match cancel.changed().await {
            Ok(()) if *cancel.borrow() => {
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    Some(0),
                );
                tokio::time::sleep(state.config.cancellation_delay).await;
                AttemptState::Cancelled
            }
            _ => AttemptState::Failed,
        };
        state.update_attempt(&request.attempt_id, terminal_state, Some(0));
        drop(_open_response_body);
        return;
    }

    let accepted = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence: 0,
        event: InvocationEventKind::Accepted {
            worker_id: state.config.worker_id.clone(),
            node_id: state.config.node_id.clone(),
            incarnation_id: state.config.incarnation_id.clone(),
            deployment_id: state.config.deployment_id.clone(),
            model_generation: state.config.model_generation,
        },
    };
    let _ = tx.send(encode_event(&accepted)).await;
    state.update_attempt(&request.attempt_id, AttemptState::Running, Some(0));

    let state_for_work = Arc::clone(&state);
    let request_for_work = request.clone();
    let tx_for_work = tx.clone();
    let work = async move {
        tokio::time::sleep(state_for_work.config.output_cadence).await;
        match state_for_work.config.fault.clone() {
            MockFault::Hang => std::future::pending::<(AttemptState, Option<u64>)>().await,
            MockFault::UsageTrickleWithoutOutput => {
                let mut sequence = 1_u64;
                loop {
                    tokio::time::sleep(state_for_work.config.output_cadence).await;
                    let event = InvocationEvent {
                        schema_version: PROTOCOL_V1,
                        request_id: request_for_work.request_id.clone(),
                        attempt_id: request_for_work.attempt_id.clone(),
                        sequence,
                        event: InvocationEventKind::Usage {
                            usage: Usage {
                                input_tokens: 1,
                                output_tokens: 0,
                            },
                        },
                    };
                    let _ = tx_for_work.send(encode_event(&event)).await;
                    state_for_work.update_attempt(
                        &request_for_work.attempt_id,
                        AttemptState::Running,
                        Some(sequence),
                    );
                    sequence = sequence.saturating_add(1);
                }
            }
            MockFault::ByteTrickleWithoutEvent => loop {
                tokio::time::sleep(state_for_work.config.output_cadence).await;
                // This is an incomplete NDJSON record by design. Transport
                // activity must not count as useful invocation progress.
                let _ = tx_for_work.send(Bytes::from_static(b"{")).await;
            },
            MockFault::EmptyDeltaTrickleWithoutOutput => {
                let mut sequence = 1_u64;
                loop {
                    tokio::time::sleep(state_for_work.config.output_cadence).await;
                    let event = InvocationEvent {
                        schema_version: PROTOCOL_V1,
                        request_id: request_for_work.request_id.clone(),
                        attempt_id: request_for_work.attempt_id.clone(),
                        sequence,
                        event: InvocationEventKind::TextDelta {
                            text: String::new(),
                        },
                    };
                    let _ = tx_for_work.send(encode_event(&event)).await;
                    state_for_work.update_attempt(
                        &request_for_work.attempt_id,
                        AttemptState::Running,
                        Some(sequence),
                    );
                    sequence = sequence.saturating_add(1);
                }
            }
            MockFault::TextDeltaThenHang => {
                let event = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence: 1,
                    event: InvocationEventKind::TextDelta {
                        text: state_for_work.config.output_text.clone(),
                    },
                };
                let _ = tx_for_work.send(encode_event(&event)).await;
                state_for_work.update_attempt(
                    &request_for_work.attempt_id,
                    AttemptState::Running,
                    Some(1),
                );
                std::future::pending::<(AttemptState, Option<u64>)>().await
            }
            MockFault::ManyTextDeltas { count, text_bytes } => {
                let text = "x".repeat(text_bytes);
                for index in 0..count {
                    let sequence = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
                    let event = InvocationEvent {
                        schema_version: PROTOCOL_V1,
                        request_id: request_for_work.request_id.clone(),
                        attempt_id: request_for_work.attempt_id.clone(),
                        sequence,
                        event: InvocationEventKind::TextDelta { text: text.clone() },
                    };
                    if tx_for_work.send(encode_event(&event)).await.is_err() {
                        return (AttemptState::Failed, Some(sequence.saturating_sub(1)));
                    }
                    state_for_work.update_attempt(
                        &request_for_work.attempt_id,
                        AttemptState::Running,
                        Some(sequence),
                    );
                }
                let sequence = u64::try_from(count).unwrap_or(u64::MAX).saturating_add(1);
                let completed = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence,
                    event: InvocationEventKind::Completed {
                        finish_reason: FinishReason::Stop,
                        usage: Some(Usage {
                            input_tokens: 1,
                            output_tokens: u64::try_from(count).unwrap_or(u64::MAX),
                        }),
                    },
                };
                let _ = tx_for_work.send(encode_event(&completed)).await;
                (AttemptState::Completed, Some(sequence))
            }
            MockFault::OpenBodyWithoutAcknowledgement => {
                unreachable!("handled before response")
            }
            MockFault::AcceptedWithoutAcknowledgement => unreachable!("handled before response"),
            MockFault::AcceptedThenDisconnect => (AttemptState::Failed, Some(0)),
            MockFault::PartialThenDisconnect => {
                let event = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence: 1,
                    event: InvocationEventKind::TextDelta {
                        text: state_for_work.config.output_text.clone(),
                    },
                };
                let _ = tx_for_work.send(encode_event(&event)).await;
                state_for_work.update_attempt(
                    &request_for_work.attempt_id,
                    AttemptState::Running,
                    Some(1),
                );
                (AttemptState::Failed, Some(1))
            }
            MockFault::MalformedEvent => {
                let _ = tx_for_work.send(Bytes::from_static(b"{malformed}\n")).await;
                (AttemptState::Failed, Some(0))
            }
            MockFault::OversizedEvent { text_bytes } => {
                let event = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence: 1,
                    event: InvocationEventKind::TextDelta {
                        text: "x".repeat(text_bytes),
                    },
                };
                let _ = tx_for_work.send(encode_event(&event)).await;
                (AttemptState::Failed, Some(1))
            }
            MockFault::None => {
                let delta = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence: 1,
                    event: InvocationEventKind::TextDelta {
                        text: state_for_work.config.output_text.clone(),
                    },
                };
                let _ = tx_for_work.send(encode_event(&delta)).await;
                state_for_work.update_attempt(
                    &request_for_work.attempt_id,
                    AttemptState::Running,
                    Some(1),
                );
                tokio::time::sleep(state_for_work.config.output_cadence).await;
                let completed = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request_for_work.request_id.clone(),
                    attempt_id: request_for_work.attempt_id.clone(),
                    sequence: 2,
                    event: InvocationEventKind::Completed {
                        finish_reason: FinishReason::Stop,
                        usage: Some(Usage {
                            input_tokens: 1,
                            output_tokens: 3,
                        }),
                    },
                };
                let _ = tx_for_work.send(encode_event(&completed)).await;
                (AttemptState::Completed, Some(2))
            }
        }
    };
    tokio::pin!(work);

    let (terminal_state, sequence) = tokio::select! {
        terminal = &mut work => terminal,
        changed = cancel.changed() => {
            if changed.is_ok() && *cancel.borrow() {
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    None,
                );
                tokio::time::sleep(state.config.cancellation_delay).await;
                let cancelled_sequence = state.next_sequence(&request.attempt_id);
                let cancelled = InvocationEvent {
                    schema_version: PROTOCOL_V1,
                    request_id: request.request_id.clone(),
                    attempt_id: request.attempt_id.clone(),
                    sequence: cancelled_sequence,
                    event: InvocationEventKind::Cancelled { reason: Some("requested".into()) },
                };
                let _ = tx.send(encode_event(&cancelled)).await;
                (AttemptState::Cancelled, Some(cancelled_sequence))
            } else {
                (AttemptState::Failed, None)
            }
        }
    };
    state.update_attempt(&request.attempt_id, terminal_state, sequence);
}

fn encode_event(event: &InvocationEvent) -> Bytes {
    let mut bytes = serde_json::to_vec(event).expect("mock event serialization");
    bytes.push(b'\n');
    Bytes::from(bytes)
}

fn rejection(
    request: &InvocationRequest,
    status: StatusCode,
    code: RejectionCode,
    message: impl Into<String>,
) -> Response {
    (
        status,
        Json(InvocationRejection::new(
            request.request_id.clone(),
            request.attempt_id.clone(),
            code,
            message,
        )),
    )
        .into_response()
}

async fn query_attempt(
    State(state): State<Arc<MockState>>,
    headers: HeaderMap,
    Path(raw_attempt_id): Path<String>,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(attempt_id) = AttemptId::new(raw_attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let table = state.attempts.lock().expect("mock attempt table poisoned");
    let Some(record) = table.records.get(&attempt_id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    Json(AttemptQueryResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.worker_id.clone(),
        identity: record.identity.clone(),
        state: record.state,
        last_sequence: record.last_sequence,
    })
    .into_response()
}

async fn cancel_attempt(
    State(state): State<Arc<MockState>>,
    headers: HeaderMap,
    Path(raw_attempt_id): Path<String>,
    Json(request): Json<CancelAttemptRequest>,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(attempt_id) = AttemptId::new(raw_attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if request.schema_version.major != PROTOCOL_V1.major
        || request.identity.attempt_id != attempt_id
        || request.identity.incarnation_id != state.config.incarnation_id
    {
        return StatusCode::CONFLICT.into_response();
    }
    let disposition = {
        let mut table = state.attempts.lock().expect("mock attempt table poisoned");
        if let Some(record) = table.records.get_mut(&attempt_id) {
            if record.identity != request.identity {
                return StatusCode::CONFLICT.into_response();
            } else if record.state.is_terminal() {
                CancelDisposition::AlreadyTerminal
            } else if matches!(
                record.state,
                AttemptState::CancellationRequested | AttemptState::ExecutionStopping
            ) {
                CancelDisposition::AlreadyRequested
            } else if let Some(cancel) = &record.cancel {
                let _ = cancel.send(true);
                record.state = AttemptState::CancellationRequested;
                CancelDisposition::Requested
            } else {
                CancelDisposition::Unknown
            }
        } else {
            drop(table);
            let tombstone = AttemptRecord {
                identity: request.identity.clone(),
                digest: RequestDigest::new("cancel-tombstone").expect("static identity"),
                state: AttemptState::CancellationRequested,
                last_sequence: None,
                remaining_time_ms: 0,
                cancel: None,
            };
            let _ = state.insert_bounded(attempt_id.clone(), tombstone);
            CancelDisposition::Unknown
        }
    };
    Json(CancelAttemptResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.worker_id.clone(),
        identity: request.identity,
        disposition,
    })
    .into_response()
}

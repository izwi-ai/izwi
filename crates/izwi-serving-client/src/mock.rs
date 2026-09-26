//! Deterministic, accelerator-free worker used by real-socket contract tests.

use async_stream::stream;
use axum::{
    body::{Body, Bytes},
    extract::{
        ws::{CloseFrame, Message as WsMessage, Utf8Bytes, WebSocket, WebSocketUpgrade},
        DefaultBodyLimit, Path, State,
    },
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

/// Scripted realtime ASR stage behavior for the mock's
/// `izwi-realtime-v1` endpoint. Stale-identity fencing is inherent (the mock
/// applies the worker fencing checks); the knobs shape the happy path and
/// the owner-loss fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockRealtimeKnobs {
    /// Transcript delta emitted for every pushed audio frame.
    pub delta_per_frame: String,
    /// Final transcript emitted on finish.
    pub final_text: String,
    /// Silence between an audio push and its delta emission.
    pub push_cadence: Duration,
    /// When set, the socket is dropped abruptly after this many audio
    /// frames, simulating owner loss with no terminal event.
    pub disconnect_after_frames: Option<usize>,
}

impl Default for MockRealtimeKnobs {
    fn default() -> Self {
        Self {
            delta_per_frame: "partial ".into(),
            final_text: "mock realtime transcript".into(),
            push_cadence: Duration::from_millis(5),
            disconnect_after_frames: None,
        }
    }
}

/// Scripted realtime TTS stage behavior for the mock's `izwi-realtime-v1`
/// endpoint: the client accumulates `Input` frames, and one finish-commit
/// emits deterministic audio frames followed by the zero-payload final frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockTtsRealtimeKnobs {
    /// Payload bytes per emitted audio frame (deterministic pattern).
    pub chunk_bytes: usize,
    /// Payload frames emitted for one finish.
    pub chunk_count: usize,
    /// Silence before each emitted frame.
    pub push_cadence: Duration,
    /// Output sample rate announced in the `Admitted.output_audio` spec.
    pub output_sample_rate: u32,
    /// When set, the socket is dropped abruptly after this many emitted
    /// audio frames, simulating owner loss with no terminal event.
    pub disconnect_after_frames: Option<usize>,
}

impl Default for MockTtsRealtimeKnobs {
    fn default() -> Self {
        Self {
            chunk_bytes: 64,
            chunk_count: 3,
            push_cadence: Duration::from_millis(5),
            output_sample_rate: 24_000,
            disconnect_after_frames: None,
        }
    }
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
    pub max_output_tokens: u32,
    pub output_text: String,
    pub output_cadence: Duration,
    pub cancellation_delay: Duration,
    pub fault: MockFault,
    pub ready: bool,
    /// Optional per-deployment routing signals advertised on the status
    /// endpoint. `None` keeps the mock's minor-0 shape (signals absent).
    pub routing_signals: Option<MockRoutingSignals>,
    /// Realtime stage knobs. `None` keeps the mock a chat worker with no
    /// realtime surface; `Some` turns it into a realtime speech_to_text
    /// worker that rejects HTTP invocations and serves the WebSocket route.
    pub realtime: Option<MockRealtimeKnobs>,
    /// Realtime TTS-stage knobs. Mutually exclusive with `realtime`; `Some`
    /// turns the mock into a realtime text_to_speech worker.
    pub realtime_tts: Option<MockTtsRealtimeKnobs>,
}

/// Configurable engine-signal values a mock worker advertises so gateway
/// routing tests can exercise warm/cold/stale signal combinations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MockRoutingSignals {
    pub kv_cache_usage_pct: f64,
    pub prefix_hits_total: u64,
    pub prefix_queries_total: u64,
    pub prefix_evictions_total: u64,
    pub kv_host_pages: u64,
    pub kv_demotions_total: u64,
    pub kv_promotions_total: u64,
    pub kv_promotion_latency_avg_seconds: f64,
    pub tokens_out_per_s_ema: f64,
    pub observation_cost_units: u64,
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
            max_output_tokens: 1024,
            output_text: "deterministic mock response".into(),
            output_cadence: Duration::from_millis(5),
            cancellation_delay: Duration::from_millis(25),
            fault: MockFault::None,
            ready: true,
            routing_signals: None,
            realtime: None,
            realtime_tts: None,
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
        if self.realtime.is_some() && self.realtime_tts.is_some() {
            return Err("realtime and realtime_tts stages are mutually exclusive");
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
        let realtime_enabled = config.realtime.is_some() || config.realtime_tts.is_some();
        let state = Arc::new(MockState {
            capacity: Arc::new(Semaphore::new(config.max_active_invocations)),
            attempts: Mutex::new(AttemptTable::default()),
            status_sequence: AtomicU64::new(0),
            config,
        });
        let mut router = Router::new()
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
            );
        if realtime_enabled {
            router = router.route(REALTIME_WS_PATH, get(realtime_socket));
        }
        let router = router
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
        features: if realtime_enabled(&state.config) {
            BTreeSet::from([
                WorkerFeature::Streaming,
                WorkerFeature::Cancellation,
                WorkerFeature::AttemptQuery,
                WorkerFeature::RealtimeSocket,
            ])
        } else {
            BTreeSet::from([
                WorkerFeature::Streaming,
                WorkerFeature::Cancellation,
                WorkerFeature::AttemptQuery,
            ])
        },
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

/// True when the mock serves either realtime stage; realtime workers reject
/// HTTP invocations and advertise the socket feature.
fn realtime_enabled(config: &MockWorkerConfig) -> bool {
    config.realtime.is_some() || config.realtime_tts.is_some()
}

fn deployment(config: &MockWorkerConfig) -> LoadedDeployment {
    let task = if config.realtime.is_some() {
        TaskKind::SpeechToText
    } else if config.realtime_tts.is_some() {
        TaskKind::TextToSpeech
    } else {
        TaskKind::Chat
    };
    let (accepted_input_formats, output_formats, max_context_tokens, max_output_tokens, realtime) =
        if config.realtime.is_some() {
            (
                BTreeSet::from([InputFormat::PcmAudio]),
                BTreeSet::from([OutputFormat::Text]),
                None,
                None,
                true,
            )
        } else if config.realtime_tts.is_some() {
            (
                BTreeSet::from([InputFormat::Text]),
                BTreeSet::from([OutputFormat::PcmAudio]),
                None,
                None,
                true,
            )
        } else {
            (
                BTreeSet::from([InputFormat::ChatMessages]),
                BTreeSet::from([OutputFormat::Text]),
                Some(4096),
                Some(config.max_output_tokens),
                false,
            )
        };
    LoadedDeployment {
        deployment_id: config.deployment_id.clone(),
        public_model: config.public_model.clone(),
        artifact_revision: ArtifactRevision::new("mock-artifact-v1").expect("static identity"),
        model_generation: config.model_generation,
        task,
        backend: BackendKind::Cpu,
        precision: "mock".into(),
        execution_representation: if realtime {
            "deterministic-realtime".into()
        } else {
            "deterministic-text".into()
        },
        tokenizer_revision: None,
        readiness: if config.ready {
            ModelReadiness::Ready
        } else {
            ModelReadiness::Loading
        },
        capability: Capability {
            task,
            streaming: true,
            realtime,
            cancellation: CancellationBehavior::Cooperative,
            accepted_input_formats,
            output_formats,
            max_input_bytes: config.max_request_bytes as u64,
            max_context_tokens,
            max_output_tokens,
        },
        kv_cache_usage_pct: config.routing_signals.map(|s| s.kv_cache_usage_pct),
        prefix_hits_total: config.routing_signals.map(|s| s.prefix_hits_total),
        prefix_queries_total: config.routing_signals.map(|s| s.prefix_queries_total),
        prefix_evictions_total: config.routing_signals.map(|s| s.prefix_evictions_total),
        kv_host_pages: config.routing_signals.map(|s| s.kv_host_pages),
        kv_demotions_total: config.routing_signals.map(|s| s.kv_demotions_total),
        kv_promotions_total: config.routing_signals.map(|s| s.kv_promotions_total),
        kv_promotion_latency_avg_seconds: config
            .routing_signals
            .map(|s| s.kv_promotion_latency_avg_seconds),
        tokens_out_per_s_ema: config.routing_signals.map(|s| s.tokens_out_per_s_ema),
        observation_cost_units: config.routing_signals.map(|s| s.observation_cost_units),
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
    if realtime_enabled(&state.config) || request.task != TaskKind::Chat {
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
            if same {
                return (
                    StatusCode::CONFLICT,
                    "attempt is already owned; acceptance is unknown, query or cancel it",
                )
                    .into_response();
            } else {
                return rejection(
                    &request,
                    StatusCode::CONFLICT,
                    RejectionCode::DuplicateAttemptConflict,
                    "attempt identity was reused with different content",
                );
            }
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

const MOCK_REALTIME_ADMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// Mock-announced realtime bounds: the protocol caps themselves.
const MOCK_REALTIME_BOUNDS: RealtimeSessionBounds = RealtimeSessionBounds {
    max_frame_bytes: MAX_REALTIME_AUDIO_FRAME_BYTES,
    max_in_flight_frames: MAX_REALTIME_IN_FLIGHT_FRAMES,
    max_session_audio_bytes: MAX_REALTIME_SESSION_AUDIO_BYTES,
};

fn offers_realtime_subprotocol(headers: &HeaderMap) -> bool {
    headers
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|offered| {
            offered
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case(REALTIME_SUBPROTOCOL))
        })
}

async fn realtime_socket(
    State(state): State<Arc<MockState>>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !state.authenticate(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if !offers_realtime_subprotocol(&headers) {
        return (
            StatusCode::BAD_REQUEST,
            "the izwi-realtime-v1 subprotocol must be offered",
        )
            .into_response();
    }
    upgrade
        .protocols([REALTIME_SUBPROTOCOL])
        .on_upgrade(move |socket| run_mock_realtime_session(state, socket))
}

fn mock_close(code: RealtimeSessionCloseCode) -> WsMessage {
    WsMessage::Close(Some(CloseFrame {
        code: code.code(),
        reason: Utf8Bytes::from_static(code.reason()),
    }))
}

fn mock_event(
    state: &MockState,
    admit: &RealtimeSessionAdmit,
    event: InvocationEventKind,
) -> (u64, WsMessage) {
    let sequence = state.next_sequence(&admit.attempt_id);
    let event = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: admit.request_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        sequence,
        event,
    };
    let message = WsMessage::Text(
        serde_json::to_string(&RealtimeServerFrame::Event { event })
            .expect("mock event encodes")
            .into(),
    );
    (sequence, message)
}

/// One scripted realtime stage for the mock's `izwi-realtime-v1` endpoint.
#[derive(Debug, Clone)]
enum MockRealtimeStage {
    Asr(MockRealtimeKnobs),
    Tts(MockTtsRealtimeKnobs),
}

impl MockRealtimeStage {
    fn task(&self) -> TaskKind {
        match self {
            Self::Asr(_) => TaskKind::SpeechToText,
            Self::Tts(_) => TaskKind::TextToSpeech,
        }
    }

    fn output_audio_spec(&self) -> Option<RealtimeAudioSpec> {
        match self {
            Self::Asr(_) => None,
            Self::Tts(knobs) => Some(RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: knobs.output_sample_rate,
                channels: 1,
            }),
        }
    }
}

/// The mock's realtime session: the same fencing checks and shared attempt
/// table as the HTTP path, a scripted stage, and abrupt owner loss when the
/// disconnect knob fires.
async fn run_mock_realtime_session(state: Arc<MockState>, mut socket: WebSocket) {
    let stage = match (&state.config.realtime, &state.config.realtime_tts) {
        (Some(knobs), None) => MockRealtimeStage::Asr(knobs.clone()),
        (None, Some(knobs)) => MockRealtimeStage::Tts(knobs.clone()),
        _ => return,
    };

    // First frame must be the admit, within the session timeout.
    let first = match tokio::time::timeout(MOCK_REALTIME_ADMIT_TIMEOUT, socket.recv()).await {
        Ok(Some(Ok(message))) => message,
        _ => {
            let _ = socket
                .send(mock_close(RealtimeSessionCloseCode::ProtocolViolation))
                .await;
            return;
        }
    };
    let admit: RealtimeSessionAdmit = match first {
        WsMessage::Text(text) => match serde_json::from_str::<RealtimeClientFrame>(&text) {
            Ok(RealtimeClientFrame::Admit { admit }) => *admit,
            _ => {
                let _ = socket
                    .send(mock_close(RealtimeSessionCloseCode::ProtocolViolation))
                    .await;
                return;
            }
        },
        _ => {
            let _ = socket
                .send(mock_close(RealtimeSessionCloseCode::ProtocolViolation))
                .await;
            return;
        }
    };
    async fn reject_close(socket: &mut WebSocket, code: RealtimeSessionCloseCode) {
        let _ = socket.send(mock_close(code)).await;
    }

    // Fencing: identical outcome codes to the real worker.
    if admit.validate().is_err() {
        reject_close(&mut socket, RealtimeSessionCloseCode::ProtocolViolation).await;
        return;
    }
    if !admit
        .caller
        .permitted_actions
        .contains(&PermittedAction::Invoke)
    {
        reject_close(&mut socket, RealtimeSessionCloseCode::PolicyDenied).await;
        return;
    }
    if admit.expected_worker_incarnation != state.config.incarnation_id {
        reject_close(
            &mut socket,
            RealtimeSessionCloseCode::WrongWorkerIncarnation,
        )
        .await;
        return;
    }
    if admit.deployment_id != state.config.deployment_id {
        reject_close(&mut socket, RealtimeSessionCloseCode::UnknownDeployment).await;
        return;
    }
    if admit.expected_model_generation != state.config.model_generation {
        reject_close(&mut socket, RealtimeSessionCloseCode::WrongModelGeneration).await;
        return;
    }
    if admit.task != stage.task() {
        reject_close(&mut socket, RealtimeSessionCloseCode::IncompatibleTask).await;
        return;
    }
    if !state.config.ready {
        reject_close(&mut socket, RealtimeSessionCloseCode::ModelNotReady).await;
        return;
    }

    // Shared attempt table: duplicate ids conflict, HTTP query/cancel work.
    let digest = RequestDigest::new(format!(
        "mock-rt-{}-{}",
        admit.session_id.as_str(),
        admit.request_id.as_str()
    ))
    .expect("bounded mock digest");
    // Any live record under this attempt id fences a new session: identical
    // content is already-owned, different content is a conflict — both close
    // with DuplicateAttempt.
    let duplicate = {
        let table = state.attempts.lock().expect("mock attempt table poisoned");
        table.records.contains_key(&admit.attempt_id)
    };
    if duplicate {
        reject_close(&mut socket, RealtimeSessionCloseCode::DuplicateAttempt).await;
        return;
    }
    let permit = match Arc::clone(&state.capacity).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let _ = socket
                .send(mock_close(RealtimeSessionCloseCode::CapacityExhausted))
                .await;
            return;
        }
    };
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let identity = AttemptIdentity {
        request_id: admit.request_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        tenant_id: admit.caller.tenant_id.clone(),
        caller_id: admit.caller.caller_id.clone(),
        incarnation_id: admit.expected_worker_incarnation.clone(),
        deployment_id: admit.deployment_id.clone(),
        model_generation: admit.expected_model_generation,
    };
    let record = AttemptRecord {
        identity,
        digest,
        state: AttemptState::Admitted,
        last_sequence: None,
        remaining_time_ms: admit.remaining_time_ms,
        cancel: Some(cancel_tx),
    };
    if !state.insert_bounded(admit.attempt_id.clone(), record) {
        drop(permit);
        let _ = socket
            .send(mock_close(RealtimeSessionCloseCode::CapacityExhausted))
            .await;
        return;
    }

    let admitted = RealtimeServerFrame::Admitted {
        session_id: admit.session_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        worker_id: state.config.worker_id.clone(),
        node_id: state.config.node_id.clone(),
        incarnation_id: state.config.incarnation_id.clone(),
        deployment_id: state.config.deployment_id.clone(),
        model_generation: state.config.model_generation,
        output_audio: stage.output_audio_spec(),
        bounds: MOCK_REALTIME_BOUNDS,
    };
    if socket
        .send(WsMessage::Text(
            serde_json::to_string(&admitted)
                .expect("admitted encodes")
                .into(),
        ))
        .await
        .is_err()
    {
        return;
    }
    let accepted = InvocationEventKind::Accepted {
        worker_id: state.config.worker_id.clone(),
        node_id: state.config.node_id.clone(),
        incarnation_id: state.config.incarnation_id.clone(),
        deployment_id: state.config.deployment_id.clone(),
        model_generation: state.config.model_generation,
    };
    let (accepted_sequence, accepted_message) = mock_event(&state, &admit, accepted);
    if socket.send(accepted_message).await.is_err() {
        return;
    }
    state.update_attempt(
        &admit.attempt_id,
        AttemptState::Running,
        Some(accepted_sequence),
    );

    // Scripted stage loops. The socket is dropped on owner loss (no terminal
    // event, no close frame); every other path ends with one terminal event
    // followed by a clean close. A `None` terminal means the transport ended
    // without a terminal outcome.
    let terminal = match stage {
        MockRealtimeStage::Asr(knobs) => {
            run_mock_asr_stage(&state, &admit, &mut socket, knobs, cancel_rx).await
        }
        MockRealtimeStage::Tts(knobs) => {
            run_mock_tts_stage(&state, &admit, &mut socket, knobs, cancel_rx).await
        }
    };

    if let Some((attempt_state, event)) = terminal {
        let (sequence, message) = mock_event(&state, &admit, event);
        let _ = socket.send(message).await;
        state.update_attempt(&admit.attempt_id, attempt_state, Some(sequence));
        let _ = socket
            .send(WsMessage::Close(Some(CloseFrame {
                code: 1000,
                reason: Utf8Bytes::from_static("session complete"),
            })))
            .await;
    }
    drop(permit);
}

/// Scripted ASR stage: one transcript delta per pushed audio frame, one final
/// transcript on finish, and abrupt owner loss when `disconnect_after_frames`
/// fires.
async fn run_mock_asr_stage(
    state: &MockState,
    admit: &RealtimeSessionAdmit,
    socket: &mut WebSocket,
    knobs: MockRealtimeKnobs,
    mut cancel_rx: watch::Receiver<bool>,
) -> Option<(AttemptState, InvocationEventKind)> {
    let mut last_audio_sequence: Option<u32> = None;
    let mut frames_pushed: usize = 0;
    let mut terminal: Option<(AttemptState, InvocationEventKind)> = None;
    while terminal.is_none() {
        let message = tokio::select! {
            message = socket.recv() => message,
            changed = cancel_rx.changed() => {
                if changed.is_ok() && *cancel_rx.borrow() {
                    state.update_attempt(
                        &admit.attempt_id,
                        AttemptState::ExecutionStopping,
                        None,
                    );
                    tokio::time::sleep(state.config.cancellation_delay).await;
                    let cancelled = InvocationEventKind::Cancelled {
                        reason: Some("requested".into()),
                    };
                    terminal = Some((AttemptState::Cancelled, cancelled));
                }
                continue;
            }
        };
        let message = match message {
            Some(Ok(message)) => message,
            Some(Err(_)) | None => break, // transport error or client gone
        };
        match message {
            WsMessage::Text(text) => match serde_json::from_str::<RealtimeClientFrame>(&text) {
                Ok(RealtimeClientFrame::Finish) => {
                    let final_delta = InvocationEventKind::TextDelta {
                        text: knobs.final_text.clone(),
                    };
                    let (sequence, message) = mock_event(state, admit, final_delta);
                    if socket.send(message).await.is_err() {
                        break;
                    }
                    state.update_attempt(&admit.attempt_id, AttemptState::Running, Some(sequence));
                    terminal = Some((
                        AttemptState::Completed,
                        InvocationEventKind::Completed {
                            finish_reason: FinishReason::Stop,
                            usage: None,
                        },
                    ));
                }
                Ok(RealtimeClientFrame::Cancel) => {
                    state.update_attempt(&admit.attempt_id, AttemptState::ExecutionStopping, None);
                    tokio::time::sleep(state.config.cancellation_delay).await;
                    terminal = Some((
                        AttemptState::Cancelled,
                        InvocationEventKind::Cancelled {
                            reason: Some("requested".into()),
                        },
                    ));
                }
                Ok(RealtimeClientFrame::Ping) => {
                    let pong = WsMessage::Text(
                        serde_json::to_string(&RealtimeServerFrame::Pong)
                            .expect("pong encodes")
                            .into(),
                    );
                    let _ = socket.send(pong).await;
                }
                Ok(RealtimeClientFrame::Admit { .. }) => {
                    terminal = Some((
                        AttemptState::Failed,
                        InvocationEventKind::Error {
                            code: InvocationErrorCode::InvalidInput,
                            message: "duplicate admit".into(),
                        },
                    ));
                }
                Ok(RealtimeClientFrame::Input { .. }) => {
                    terminal = Some((
                        AttemptState::Failed,
                        InvocationEventKind::Error {
                            code: InvocationErrorCode::InvalidInput,
                            message: "text input is not valid for speech_to_text sessions".into(),
                        },
                    ));
                }
                Err(_) => {
                    terminal = Some((
                        AttemptState::Failed,
                        InvocationEventKind::Error {
                            code: InvocationErrorCode::InvalidInput,
                            message: "unparseable control frame".into(),
                        },
                    ));
                }
            },
            WsMessage::Binary(data) => {
                let Ok((header, _payload)) = decode_realtime_audio_frame(&data) else {
                    terminal = Some((
                        AttemptState::Failed,
                        InvocationEventKind::Error {
                            code: InvocationErrorCode::InvalidInput,
                            message: "malformed audio frame".into(),
                        },
                    ));
                    continue;
                };
                if last_audio_sequence.is_some_and(|last| header.sequence <= last) {
                    terminal = Some((
                        AttemptState::Failed,
                        InvocationEventKind::Error {
                            code: InvocationErrorCode::InvalidInput,
                            message: "audio frame sequence regressed".into(),
                        },
                    ));
                    continue;
                }
                last_audio_sequence = Some(header.sequence);
                frames_pushed = frames_pushed.saturating_add(1);
                if knobs
                    .disconnect_after_frames
                    .is_some_and(|limit| frames_pushed >= limit)
                {
                    // Owner loss: drop everything without a terminal event.
                    return None;
                }
                tokio::time::sleep(knobs.push_cadence).await;
                let delta = InvocationEventKind::TextDelta {
                    text: knobs.delta_per_frame.clone(),
                };
                let (sequence, message) = mock_event(state, admit, delta);
                if socket.send(message).await.is_err() {
                    break;
                }
                state.update_attempt(&admit.attempt_id, AttemptState::Running, Some(sequence));
            }
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
        }
    }
    terminal
}

/// Scripted TTS stage: `Input` frames accumulate the utterance, finish-commit
/// emits `chunk_count` deterministic payload frames plus the zero-payload
/// final frame, and the disconnect knob fires on emitted audio frames.
async fn run_mock_tts_stage(
    state: &MockState,
    admit: &RealtimeSessionAdmit,
    socket: &mut WebSocket,
    knobs: MockTtsRealtimeKnobs,
    mut cancel_rx: watch::Receiver<bool>,
) -> Option<(AttemptState, InvocationEventKind)> {
    let mut text = String::new();
    let mut text_bytes = 0u64;
    let mut finish_submitted = false;
    let mut audio_sequence = 0u32;
    let mut frames_emitted = 0usize;
    let mut terminal: Option<(AttemptState, InvocationEventKind)> = None;
    while terminal.is_none() {
        let message = tokio::select! {
            message = socket.recv() => message,
            changed = cancel_rx.changed() => {
                if changed.is_ok() && *cancel_rx.borrow() {
                    state.update_attempt(
                        &admit.attempt_id,
                        AttemptState::ExecutionStopping,
                        None,
                    );
                    tokio::time::sleep(state.config.cancellation_delay).await;
                    let cancelled = InvocationEventKind::Cancelled {
                        reason: Some("requested".into()),
                    };
                    terminal = Some((AttemptState::Cancelled, cancelled));
                }
                continue;
            }
        };
        let message = match message {
            Some(Ok(message)) => message,
            Some(Err(_)) | None => break, // transport error or client gone
        };
        match message {
            WsMessage::Text(text_frame) => {
                match serde_json::from_str::<RealtimeClientFrame>(&text_frame) {
                    Ok(RealtimeClientFrame::Finish) => {
                        if finish_submitted {
                            terminal = Some(mock_failure("finish was already submitted"));
                            continue;
                        }
                        finish_submitted = true;
                        let utterance = std::mem::take(&mut text);
                        if utterance.is_empty() {
                            // Mirror the worker: finishing without input is a
                            // successful empty result with no audio.
                            terminal = Some(mock_completed());
                            continue;
                        }
                        'emit: {
                            for index in 0..knobs.chunk_count {
                                if knobs
                                    .disconnect_after_frames
                                    .is_some_and(|limit| frames_emitted >= limit)
                                {
                                    // Owner loss: drop everything.
                                    return None;
                                }
                                // Emission runs inline, so cancellation is
                                // observed between frames here rather than
                                // through the stage select.
                                if *cancel_rx.borrow_and_update() {
                                    state.update_attempt(
                                        &admit.attempt_id,
                                        AttemptState::ExecutionStopping,
                                        None,
                                    );
                                    tokio::time::sleep(state.config.cancellation_delay).await;
                                    return Some((
                                        AttemptState::Cancelled,
                                        InvocationEventKind::Cancelled {
                                            reason: Some("requested".into()),
                                        },
                                    ));
                                }
                                tokio::time::sleep(knobs.push_cadence).await;
                                audio_sequence = audio_sequence.saturating_add(1);
                                frames_emitted = frames_emitted.saturating_add(1);
                                let payload = mock_tts_payload(index, knobs.chunk_bytes);
                                let frame =
                                    encode_realtime_audio_frame(audio_sequence, false, &payload)
                                        .expect("mock frame within caps");
                                if socket.send(WsMessage::Binary(frame.into())).await.is_err() {
                                    break 'emit;
                                }
                                state.update_attempt(
                                    &admit.attempt_id,
                                    AttemptState::Running,
                                    None,
                                );
                            }
                            if knobs
                                .disconnect_after_frames
                                .is_some_and(|limit| frames_emitted >= limit)
                            {
                                return None;
                            }
                            audio_sequence = audio_sequence.saturating_add(1);
                            let frame = encode_realtime_audio_frame(audio_sequence, true, &[])
                                .expect("final frame within caps");
                            if socket.send(WsMessage::Binary(frame.into())).await.is_err() {
                                break 'emit;
                            }
                            terminal = Some(mock_completed());
                        }
                    }
                    Ok(RealtimeClientFrame::Cancel) => {
                        state.update_attempt(
                            &admit.attempt_id,
                            AttemptState::ExecutionStopping,
                            None,
                        );
                        tokio::time::sleep(state.config.cancellation_delay).await;
                        terminal = Some((
                            AttemptState::Cancelled,
                            InvocationEventKind::Cancelled {
                                reason: Some("requested".into()),
                            },
                        ));
                    }
                    Ok(RealtimeClientFrame::Ping) => {
                        let pong = WsMessage::Text(
                            serde_json::to_string(&RealtimeServerFrame::Pong)
                                .expect("pong encodes")
                                .into(),
                        );
                        let _ = socket.send(pong).await;
                    }
                    Ok(RealtimeClientFrame::Admit { .. }) => {
                        terminal = Some(mock_failure("duplicate admit"));
                    }
                    Ok(RealtimeClientFrame::Input { text: incoming }) => {
                        if finish_submitted {
                            terminal = Some(mock_failure("input after finish"));
                        } else if incoming.len() > MAX_REALTIME_INPUT_TEXT_BYTES {
                            terminal = Some(mock_failure("input text exceeds bound"));
                        } else {
                            text_bytes = text_bytes.saturating_add(incoming.len() as u64);
                            if text_bytes > MOCK_REALTIME_BOUNDS.max_session_audio_bytes {
                                terminal = Some(mock_failure("session text budget exhausted"));
                            } else {
                                text.push_str(&incoming);
                            }
                        }
                    }
                    Err(_) => {
                        terminal = Some(mock_failure("unparseable control frame"));
                    }
                }
            }
            WsMessage::Binary(_) => {
                terminal = Some(mock_failure(
                    "audio input is not valid for text_to_speech sessions",
                ));
            }
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
        }
    }
    terminal
}

fn mock_failure(message: &str) -> (AttemptState, InvocationEventKind) {
    (
        AttemptState::Failed,
        InvocationEventKind::Error {
            code: InvocationErrorCode::InvalidInput,
            message: message.to_string(),
        },
    )
}

fn mock_completed() -> (AttemptState, InvocationEventKind) {
    (
        AttemptState::Completed,
        InvocationEventKind::Completed {
            finish_reason: FinishReason::Stop,
            usage: None,
        },
    )
}

/// Deterministic non-silent payload so tests can assert content.
fn mock_tts_payload(index: usize, chunk_bytes: usize) -> Vec<u8> {
    (0..chunk_bytes)
        .map(|position| ((index * 31 + position * 7) % 251 + 1) as u8)
        .collect()
}

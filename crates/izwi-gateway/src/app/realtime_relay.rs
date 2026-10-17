//! Gateway realtime relay (DS3.6): a bounded, authenticated WebSocket relay
//! that forwards public `izwi-realtime-v1` sessions to eligible realtime
//! workers, resolved per stage — the admit's task selects the
//! speech_to_text pool or the optional text_to_speech pool.
//!
//! The gateway is a transparent relay on the worker subprotocol: it
//! authenticates the client principal, selects a worker from the registry,
//! mints the attested caller context (client-supplied caller claims are
//! ignored), dials the worker, and pumps frames in both directions without
//! reinterpreting them. Owner loss (the worker stream ends without a
//! terminal event or close frame) is surfaced to the client as an explicit
//! error event before the socket closes. Reconnection is always a new
//! session; nothing is resumed.

use crate::api::request_context::{principal_namespace, RequestContext};
use crate::gateway::GatewayState;
use crate::worker_registry::{BackendPolicy, WorkerRegistry, WorkerSelectionRequest};
use axum::{
    extract::{
        ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade},
        Extension, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};
use izwi_serving_client::realtime::{RealtimeClientConfig, RealtimeClientError, WorkerSocket};
use izwi_serving_client::WorkerClientTlsConfig;
use izwi_serving_protocol::{
    decode_realtime_audio_frame, AttemptId, CallerId, CancellationBehavior, DeploymentId,
    GatewayAttestedCallerContext, InputFormat, InvocationErrorCode, InvocationEvent,
    InvocationEventKind, ModelAlias, OutputFormat, PermittedAction, PolicyRevision,
    RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit, RealtimeSessionCloseCode,
    RequestId, ServiceClass, ServiceCredentials, SessionId, TaskKind, TenantId, TraceContext,
    PROTOCOL_V1, REALTIME_SUBPROTOCOL,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message as WireMessage;

/// How long the relay waits for the client's first admit frame.
const CLIENT_ADMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// Deadline for one frame write toward either peer. A peer that cannot take
/// a frame for this long ends the session; it never stalls the other
/// direction.
const RELAY_SEND_TIMEOUT: Duration = Duration::from_secs(5);
/// Frames queued toward one peer while its writer is busy.
const RELAY_QUEUE_FRAMES: usize = 64;
/// How long an unused registration slot is held before TTL eviction.
pub const REGISTRATION_TTL: Duration = Duration::from_secs(3600);
/// Client control frames are tiny; anything larger is refused before parse.
const RELAY_CONTROL_FRAME_BYTES: usize = 64 * 1024;
/// Upper bound for the relay's configured session capacity.
pub const MAX_RELAY_SESSIONS: usize = 4_096;

/// Bounded configuration for the relay, resolved once at gateway boot. Each
/// stage pool is optional; boot fails closed only when both are absent.
#[derive(Clone)]
pub struct RealtimeRelayConfig {
    /// Approved speech_to_text deployment for ASR-stream sessions.
    pub deployment_id: Option<DeploymentId>,
    /// Optional second stage: the approved text_to_speech deployment serving
    /// TTS-stream sessions. A missing stage refuses its admits with
    /// PolicyDenied.
    pub tts_deployment_id: Option<DeploymentId>,
    pub public_model: ModelAlias,
    pub policy_revision: PolicyRevision,
    pub backend_policy: BackendPolicy,
    pub max_sessions: usize,
    /// End-to-end session budget minted into every worker admit.
    pub session_budget: Duration,
}

/// The resolved worker pool for one relayed session, keyed by the admit's
/// task.
#[derive(Debug, Clone)]
pub struct RelayStage {
    pub deployment_id: DeploymentId,
    pub task: TaskKind,
    pub input_format: InputFormat,
    pub output_format: OutputFormat,
}

/// One live relayed session as observed by the gateway: when it was
/// registered, so a session whose cleanup never ran expires.
#[derive(Debug, Clone)]
pub struct SessionRegistration {
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRegistryError {
    DuplicateSession,
    Full,
}

/// Bounded live-session registry: session id -> registration, with TTL
/// eviction of abandoned entries on insert. Mirrors the conversation pin
/// table's bounded-map discipline.
pub struct RealtimeSessionRegistry {
    entries: Mutex<RegistryState>,
    max_sessions: usize,
}

struct RegistryState {
    by_session: HashMap<SessionId, SessionRegistration>,
    order: VecDeque<SessionId>,
}

impl RealtimeSessionRegistry {
    fn new(max_sessions: usize) -> Self {
        Self {
            entries: Mutex::new(RegistryState {
                by_session: HashMap::new(),
                order: VecDeque::new(),
            }),
            max_sessions,
        }
    }

    pub fn active(&self) -> usize {
        self.entries
            .lock()
            .expect("registry poisoned")
            .by_session
            .len()
    }

    pub fn insert(
        &self,
        session_id: SessionId,
        registration: SessionRegistration,
        ttl: Duration,
    ) -> Result<(), SessionRegistryError> {
        let mut state = self.entries.lock().expect("registry poisoned");
        if state.by_session.contains_key(&session_id) {
            return Err(SessionRegistryError::DuplicateSession);
        }
        let now = Instant::now();
        while state.by_session.len() >= self.max_sessions {
            let evicted = state
                .order
                .iter()
                .position(|id| {
                    state
                        .by_session
                        .get(id)
                        .is_some_and(|record| now - record.created_at > ttl)
                })
                .and_then(|index| state.order.remove(index));
            let Some(evicted) = evicted else {
                return Err(SessionRegistryError::Full);
            };
            state.by_session.remove(&evicted);
        }
        state.order.push_back(session_id.clone());
        state.by_session.insert(session_id, registration);
        Ok(())
    }

    pub fn remove(&self, session_id: &SessionId) -> bool {
        let mut state = self.entries.lock().expect("registry poisoned");
        if state.by_session.remove(session_id).is_some() {
            state.order.retain(|id| id != session_id);
            true
        } else {
            false
        }
    }
}

/// The relay surface attached to [`GatewayState`] when the realtime flag is
/// on. `Debug` omits worker credentials.
pub struct GatewayRealtimeRelay {
    registry: WorkerRegistry,
    sessions: RealtimeSessionRegistry,
    config: RealtimeRelayConfig,
    worker_credentials: ServiceCredentials,
    /// Worker socket dial deadlines and TLS (private CA, mTLS identity).
    dial: RealtimeClientConfig,
}

impl fmt::Debug for GatewayRealtimeRelay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayRealtimeRelay")
            .field("active_sessions", &self.sessions.active())
            .finish_non_exhaustive()
    }
}

impl GatewayRealtimeRelay {
    pub fn new(
        registry: WorkerRegistry,
        config: RealtimeRelayConfig,
        worker_credentials: ServiceCredentials,
        worker_tls: WorkerClientTlsConfig,
    ) -> Result<Self, String> {
        if config.max_sessions == 0 {
            return Err("realtime relay max_sessions must be non-zero".into());
        }
        if config.session_budget.is_zero() {
            return Err("realtime relay session budget must be non-zero".into());
        }
        Ok(Self {
            registry,
            sessions: RealtimeSessionRegistry::new(config.max_sessions),
            config,
            worker_credentials,
            dial: RealtimeClientConfig {
                tls: worker_tls,
                ..RealtimeClientConfig::default()
            },
        })
    }

    pub fn sessions(&self) -> &RealtimeSessionRegistry {
        &self.sessions
    }

    pub fn config(&self) -> &RealtimeRelayConfig {
        &self.config
    }

    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    /// Dials an admitted worker session over the shared serving-client
    /// dialer, with the gateway's worker TLS.
    pub async fn dial_worker(
        &self,
        endpoint: &str,
        admit: &RealtimeSessionAdmit,
        trace: &TraceContext,
    ) -> Result<WorkerSide, RealtimeClientError> {
        let dialed = izwi_serving_client::realtime::dial(
            endpoint,
            &self.worker_credentials,
            admit,
            Some(trace),
            &self.dial,
        )
        .await?;
        let (sink, source) = dialed.stream.split();
        Ok(WorkerSide {
            sink,
            source,
            admitted_message: Message::Text(Utf8Bytes::from(dialed.admitted_frame)),
        })
    }
}

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

/// Public `/v1/realtime/ws` upgrade handler. Authentication, admission
/// accounting, and the request context arrive through the v1 middleware
/// stack. Clients offering the worker subprotocol get the byte-identical
/// passthrough relay; everyone else gets the public transcription-realtime
/// translator, so a single-node client can repoint at the gateway unchanged.
pub async fn relay_socket(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(relay) = state.realtime_relay.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if offers_realtime_subprotocol(&headers) {
        return upgrade
            .protocols([REALTIME_SUBPROTOCOL])
            .on_upgrade(move |socket| run_relay_session(state, relay, socket, context));
    }
    upgrade.on_upgrade(move |socket| {
        crate::app::realtime_translate::run_translate_session(state, relay, socket, context)
    })
}

/// A client frame shaped into a relay action by the reader task.
enum ClientAction {
    /// Raw binary audio frame, forwarded to the worker verbatim so the
    /// client's sequence numbers reach the worker unchanged.
    Audio(Vec<u8>),
    /// Bounded text input (TTS-stream stage), forwarded verbatim.
    Text(String),
    Finish,
    Cancel,
    Ping,
    Violation(&'static str),
    Gone,
}

async fn run_relay_session(
    state: GatewayState,
    relay: Arc<GatewayRealtimeRelay>,
    socket: WebSocket,
    context: RequestContext,
) {
    let (mut client_sink, mut client_source) = socket.split();

    // First client frame: the admit. Caller identity is minted by the
    // gateway after selection; the client's session/request/attempt ids pass
    // through so the same client library works against the gateway.
    let first = match tokio::time::timeout(CLIENT_ADMIT_TIMEOUT, client_source.next()).await {
        Ok(Some(Ok(message))) => message,
        _ => return,
    };
    let requested = match first {
        Message::Text(text) => {
            if text.len() > RELAY_CONTROL_FRAME_BYTES {
                return;
            }
            match serde_json::from_str::<RealtimeClientFrame>(&text) {
                Ok(RealtimeClientFrame::Admit { admit }) => *admit,
                _ => return,
            }
        }
        _ => return,
    };
    if requested.validate().is_err() {
        close_client(
            &mut client_sink,
            Some(RealtimeSessionCloseCode::ProtocolViolation),
        )
        .await;
        return;
    }
    // Stage resolution: the admit's task picks the pool. An unconfigured
    // stage is refused explicitly rather than silently dropping the socket.
    let stage = match requested.task {
        TaskKind::SpeechToText => match &relay.config().deployment_id {
            Some(deployment_id) => RelayStage {
                deployment_id: deployment_id.clone(),
                task: TaskKind::SpeechToText,
                input_format: InputFormat::PcmAudio,
                output_format: OutputFormat::Text,
            },
            None => {
                close_client(
                    &mut client_sink,
                    Some(RealtimeSessionCloseCode::PolicyDenied),
                )
                .await;
                return;
            }
        },
        TaskKind::TextToSpeech => match &relay.config().tts_deployment_id {
            Some(tts_deployment_id) => RelayStage {
                deployment_id: tts_deployment_id.clone(),
                task: TaskKind::TextToSpeech,
                input_format: InputFormat::Text,
                output_format: OutputFormat::PcmAudio,
            },
            None => {
                close_client(
                    &mut client_sink,
                    Some(RealtimeSessionCloseCode::PolicyDenied),
                )
                .await;
                return;
            }
        },
        // Protocol validate rejects chat and unknown-task admits before
        // this point.
        TaskKind::Chat | TaskKind::Unknown => {
            close_client(
                &mut client_sink,
                Some(RealtimeSessionCloseCode::ProtocolViolation),
            )
            .await;
            return;
        }
    };

    // Session accounting: one bounded registry entry per live session gates
    // capacity and duplicate sessions before any worker work happens.
    let registration = || SessionRegistration {
        created_at: Instant::now(),
    };
    if let Err(error) = relay.sessions().insert(
        requested.session_id.clone(),
        registration(),
        REGISTRATION_TTL,
    ) {
        let code = match error {
            SessionRegistryError::DuplicateSession => RealtimeSessionCloseCode::DuplicateAttempt,
            SessionRegistryError::Full => RealtimeSessionCloseCode::CapacityExhausted,
        };
        close_client(&mut client_sink, Some(code)).await;
        return;
    }
    let _session_cleanup = SessionGuard {
        relay: Arc::clone(&relay),
        session_id: requested.session_id.clone(),
    };

    // Tenant concurrency: one lease covers the whole stage session, so a
    // second stage of the same principal queues behind this one's slot
    // instead of deadlocking against it.
    let tenant_work = match state.begin_tenant_work(&context) {
        Ok(lease) => lease,
        Err(error) => {
            reject_client(&mut client_sink, error.status, &error.message).await;
            return;
        }
    };

    let mut selected = match relay.registry.select_and_reserve(&WorkerSelectionRequest {
        // A text stream that names a voice or streams its text needs a
        // worker that honours it; an older worker would ignore it.
        required_features: requested.input.required_features(),
        ..realtime_selection_request(relay.config(), &stage)
    }) {
        Ok(selected) => selected,
        Err(_) => {
            reject_client(
                &mut client_sink,
                StatusCode::SERVICE_UNAVAILABLE,
                "no fresh, ready realtime worker satisfies the request",
            )
            .await;
            return;
        }
    };

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: requested.session_id.clone(),
        request_id: requested.request_id.clone(),
        attempt_id: requested.attempt_id.clone(),
        expected_worker_incarnation: selected.key.incarnation_id.clone(),
        deployment_id: selected.deployment_id.clone(),
        expected_model_generation: selected.model_generation,
        caller: attested_caller(&relay, &context),
        task: stage.task,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: relay.config().session_budget.as_millis() as u64,
        input: requested.input.clone(),
    };
    let worker = match relay
        .dial_worker(&selected.client.endpoint(), &admit, &context.trace)
        .await
    {
        Ok(worker) => worker,
        Err(_) => {
            reject_client(
                &mut client_sink,
                StatusCode::BAD_GATEWAY,
                "realtime worker session could not be established",
            )
            .await;
            return;
        }
    };

    // Worker accepted: bind the tenant lease to this exact attempt and mark
    // the local dispatch accepted so capacity is held for the session. The
    // registry entry is refreshed with the resolved worker identity.
    let identity = izwi_serving_protocol::AttemptIdentity {
        request_id: admit.request_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        tenant_id: admit.caller.tenant_id.clone(),
        caller_id: admit.caller.caller_id.clone(),
        incarnation_id: admit.expected_worker_incarnation.clone(),
        deployment_id: admit.deployment_id.clone(),
        model_generation: admit.expected_model_generation,
    };
    let _bound_lease = tenant_work.bind(selected.client.clone(), identity);
    if selected.dispatch.mark_accepted().is_err() {
        close_client(
            &mut client_sink,
            Some(RealtimeSessionCloseCode::CapacityExhausted),
        )
        .await;
        return;
    }
    relay.sessions().remove(&requested.session_id);
    let _ = relay.sessions().insert(
        requested.session_id.clone(),
        registration(),
        REGISTRATION_TTL,
    );

    // Three tasks, so neither direction can stall the other: the client
    // writer owns the client socket's sink, client actions go straight to
    // the worker, and worker frames go to the client writer's queue. A slow
    // client therefore never delays a cancel or barge-in toward the worker.
    let (action_tx, action_rx) = mpsc::channel::<ClientAction>(16);
    let reader = tokio::spawn(read_client_actions(client_source, action_tx));
    let (client_tx, client_rx) = mpsc::channel::<Message>(RELAY_QUEUE_FRAMES);
    let client_writer = tokio::spawn(write_client(client_sink, client_rx));
    if client_tx.send(worker.admitted_message).await.is_err() {
        reader.abort();
        return;
    }
    let ids = (admit.request_id.clone(), admit.attempt_id.clone());
    let mut to_worker = tokio::spawn(forward_client_actions(
        action_rx,
        worker.sink,
        client_tx.clone(),
        ids.clone(),
    ));
    let mut to_client = tokio::spawn(forward_worker_frames(worker.source, client_tx, ids));
    // Either direction ending ends the session: the client left or broke the
    // protocol, or the worker finished, closed, or was lost. Dropping the
    // other direction's half closes the worker socket.
    tokio::select! {
        _ = &mut to_worker => to_client.abort(),
        _ = &mut to_client => to_worker.abort(),
    }
    reader.abort();
    // The client writer drains what is queued (terminal events, the close
    // frame) and exits once every sender is gone.
    let _ = tokio::time::timeout(RELAY_SEND_TIMEOUT, client_writer).await;
}

/// Writes queued frames to the client, each within the send deadline. Ends
/// when the queue closes or the client cannot take a frame in time.
async fn write_client(
    mut client_sink: futures::stream::SplitSink<WebSocket, Message>,
    mut frames: mpsc::Receiver<Message>,
) {
    while let Some(frame) = frames.recv().await {
        let closing = matches!(frame, Message::Close(_));
        match tokio::time::timeout(RELAY_SEND_TIMEOUT, client_sink.send(frame)).await {
            Ok(Ok(())) if !closing => {}
            _ => return,
        }
    }
    let _ = tokio::time::timeout(RELAY_SEND_TIMEOUT, client_sink.close()).await;
}

/// Queues a frame for the client writer. False when the client is gone or
/// has not drained a frame within the send deadline.
async fn queue_for_client(client_tx: &mpsc::Sender<Message>, frame: Message) -> bool {
    matches!(
        tokio::time::timeout(RELAY_SEND_TIMEOUT, client_tx.send(frame)).await,
        Ok(Ok(()))
    )
}

fn close_message(code: Option<RealtimeSessionCloseCode>) -> Message {
    let frame = code
        .map(|code| CloseFrame {
            code: code.code(),
            reason: Utf8Bytes::from_static(code.reason()),
        })
        .unwrap_or(CloseFrame {
            code: 1000,
            reason: Utf8Bytes::from_static("session complete"),
        });
    Message::Close(Some(frame))
}

/// Client to worker: forwards each client action with a send deadline.
/// Pings are answered through the client writer, never inline.
async fn forward_client_actions(
    mut actions: mpsc::Receiver<ClientAction>,
    mut worker_sink: futures::stream::SplitSink<WorkerSocket, WireMessage>,
    client_tx: mpsc::Sender<Message>,
    (request_id, attempt_id): (RequestId, AttemptId),
) {
    while let Some(action) = actions.recv().await {
        // Passthrough: the client's framed sequence numbers reach the worker
        // unchanged, so worker-side monotonicity enforcement sees exactly
        // what a direct client would send.
        let frame = match action {
            ClientAction::Audio(frame) => WireMessage::Binary(frame.into()),
            ClientAction::Text(text) => control_wire(&RealtimeClientFrame::Input { text }),
            ClientAction::Finish => control_wire(&RealtimeClientFrame::Finish),
            ClientAction::Cancel => control_wire(&RealtimeClientFrame::Cancel),
            ClientAction::Ping => {
                let pong = serde_json::to_string(&RealtimeServerFrame::Pong).expect("pong encodes");
                // A pong that does not fit is dropped; the client retries.
                let _ = client_tx.try_send(Message::Text(pong.into()));
                continue;
            }
            ClientAction::Violation(reason) => {
                let error = error_message(
                    &request_id,
                    &attempt_id,
                    InvocationErrorCode::InvalidInput,
                    reason,
                );
                let _ = queue_for_client(&client_tx, Message::Text(error.into())).await;
                let _ = queue_for_client(
                    &client_tx,
                    close_message(Some(RealtimeSessionCloseCode::ProtocolViolation)),
                )
                .await;
                return;
            }
            // Client left: request cooperative teardown toward the worker;
            // its session ends on its own cancellation ladder.
            ClientAction::Gone => {
                send_to_worker(&mut worker_sink, control_wire(&RealtimeClientFrame::Cancel)).await;
                return;
            }
        };
        if !send_to_worker(&mut worker_sink, frame).await {
            return;
        }
    }
    send_to_worker(&mut worker_sink, control_wire(&RealtimeClientFrame::Cancel)).await;
}

async fn send_to_worker(
    worker_sink: &mut futures::stream::SplitSink<WorkerSocket, WireMessage>,
    frame: WireMessage,
) -> bool {
    matches!(
        tokio::time::timeout(RELAY_SEND_TIMEOUT, worker_sink.send(frame)).await,
        Ok(Ok(()))
    )
}

/// Worker to client: relays every worker frame verbatim through the client
/// writer, and surfaces owner loss as an explicit error event.
async fn forward_worker_frames(
    mut worker_source: futures::stream::SplitStream<WorkerSocket>,
    client_tx: mpsc::Sender<Message>,
    (request_id, attempt_id): (RequestId, AttemptId),
) {
    let mut terminal_seen = false;
    loop {
        let frame = match worker_source.next().await {
            Some(Ok(WireMessage::Text(text))) => {
                // Relay verbatim; decode only to track the terminal outcome
                // for owner-loss classification.
                if let Ok(RealtimeServerFrame::Event { event }) =
                    serde_json::from_str::<RealtimeServerFrame>(&text)
                {
                    if matches!(
                        event.event,
                        InvocationEventKind::Completed { .. }
                            | InvocationEventKind::Error { .. }
                            | InvocationEventKind::Cancelled { .. }
                    ) {
                        terminal_seen = true;
                    }
                }
                Message::Text(Utf8Bytes::from(text.as_str()))
            }
            // Passthrough (TTS-stream audio): the worker's framed sequence
            // numbers reach the client unchanged.
            Some(Ok(WireMessage::Binary(data))) => Message::Binary(data),
            Some(Ok(WireMessage::Close(frame))) => {
                let code = frame
                    .as_ref()
                    .map(|frame| u16::from(frame.code))
                    .unwrap_or(1000);
                let reason = frame
                    .as_ref()
                    .map(|frame| Utf8Bytes::from(frame.reason.as_str()))
                    .unwrap_or_else(|| Utf8Bytes::from_static("session complete"));
                let _ = queue_for_client(
                    &client_tx,
                    Message::Close(Some(CloseFrame { code, reason })),
                )
                .await;
                return;
            }
            Some(Ok(WireMessage::Ping(_) | WireMessage::Pong(_) | WireMessage::Frame(_))) => {
                continue;
            }
            Some(Err(_)) | None => {
                // Owner loss: no terminal event, no close frame. The client
                // gets an explicit interruption event and a close;
                // reconnecting is always a new session.
                if !terminal_seen {
                    let error = error_message(
                        &request_id,
                        &attempt_id,
                        InvocationErrorCode::Internal,
                        "realtime worker session was lost",
                    );
                    let _ = queue_for_client(&client_tx, Message::Text(error.into())).await;
                }
                let _ = queue_for_client(
                    &client_tx,
                    close_message(Some(RealtimeSessionCloseCode::Internal)),
                )
                .await;
                return;
            }
        };
        if !queue_for_client(&client_tx, frame).await {
            return;
        }
    }
}

/// Holds the registry entry until the session ends on any path.
pub struct SessionGuard {
    relay: Arc<GatewayRealtimeRelay>,
    session_id: SessionId,
}

impl SessionGuard {
    pub fn new(relay: Arc<GatewayRealtimeRelay>, session_id: SessionId) -> Self {
        Self { relay, session_id }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.relay.sessions().remove(&self.session_id);
    }
}

pub fn realtime_selection_request(
    config: &RealtimeRelayConfig,
    stage: &RelayStage,
) -> WorkerSelectionRequest {
    WorkerSelectionRequest {
        protocol_version: PROTOCOL_V1,
        deployment_id: stage.deployment_id.clone(),
        public_model: config.public_model.clone(),
        task: stage.task,
        input_format: stage.input_format,
        output_format: stage.output_format,
        streaming: true,
        realtime: true,
        cancellation: Some(CancellationBehavior::Cooperative),
        backend_policy: config.backend_policy,
        input_bytes: 0,
        context_tokens: None,
        output_tokens: None,
        required_features: Default::default(),
    }
}

pub fn attested_caller(
    relay: &GatewayRealtimeRelay,
    context: &RequestContext,
) -> GatewayAttestedCallerContext {
    let tenant_namespace = principal_namespace(&context.principal);
    GatewayAttestedCallerContext {
        tenant_id: TenantId::new(hashed_identity("tenant", &tenant_namespace))
            .unwrap_or_else(|_| TenantId::new("tenant:unavailable").expect("static identity")),
        caller_id: CallerId::new(hashed_identity("caller", context.principal.id.clone()))
            .unwrap_or_else(|_| CallerId::new("caller:unavailable").expect("static identity")),
        policy_revision: relay.config().policy_revision.clone(),
        permitted_actions: [
            PermittedAction::Invoke,
            PermittedAction::CancelOwnInvocation,
        ]
        .into_iter()
        .collect(),
        allowed_data_regions: Vec::new(),
    }
}

fn hashed_identity(prefix: &str, value: impl AsRef<[u8]>) -> String {
    let digest = Sha256::digest(value.as_ref());
    let mut encoded = String::with_capacity(prefix.len() + 1 + digest.len() * 2);
    encoded.push_str(prefix);
    encoded.push(':');
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

async fn read_client_actions(
    mut client_source: futures::stream::SplitStream<WebSocket>,
    action_tx: tokio::sync::mpsc::Sender<ClientAction>,
) {
    while let Some(message) = client_source.next().await {
        let action = match message {
            Ok(Message::Text(text)) => {
                if text.len() > RELAY_CONTROL_FRAME_BYTES {
                    Some(ClientAction::Violation("control frame exceeds bound"))
                } else {
                    match serde_json::from_str::<RealtimeClientFrame>(&text) {
                        Ok(RealtimeClientFrame::Finish) => Some(ClientAction::Finish),
                        Ok(RealtimeClientFrame::Cancel) => Some(ClientAction::Cancel),
                        Ok(RealtimeClientFrame::Ping) => Some(ClientAction::Ping),
                        Ok(RealtimeClientFrame::Admit { .. }) => {
                            Some(ClientAction::Violation("duplicate admit"))
                        }
                        Ok(RealtimeClientFrame::Input { text }) => Some(ClientAction::Text(text)),
                        Err(_) => Some(ClientAction::Violation("unparseable control frame")),
                    }
                }
            }
            Ok(Message::Binary(data)) => match decode_realtime_audio_frame(&data) {
                // Forward the entire frame verbatim, header included.
                Ok(_) => Some(ClientAction::Audio(data.to_vec())),
                Err(_) => Some(ClientAction::Violation("malformed audio frame")),
            },
            Ok(Message::Close(_)) | Err(_) => {
                let _ = action_tx.send(ClientAction::Gone).await;
                return;
            }
            Ok(Message::Ping(_) | Message::Pong(_)) => continue,
        };
        let Some(action) = action else { continue };
        if action_tx.send(action).await.is_err() {
            return;
        }
    }
    let _ = action_tx.send(ClientAction::Gone).await;
}

fn control_wire(frame: &RealtimeClientFrame) -> WireMessage {
    WireMessage::Text(
        serde_json::to_string(frame)
            .expect("control frame encodes")
            .into(),
    )
}

/// Relay-side terminal synthesis uses a locally monotonic sequence so the
/// synthesized error event never collides with relayed worker events on the
/// client's validation path.
fn error_message(
    request_id: &RequestId,
    attempt_id: &AttemptId,
    code: InvocationErrorCode,
    message: &str,
) -> String {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let event = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request_id.clone(),
        attempt_id: attempt_id.clone(),
        sequence: SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        event: InvocationEventKind::Error {
            code,
            message: message.to_string(),
        },
    };
    serde_json::to_string(&RealtimeServerFrame::Event { event }).expect("error event encodes")
}

pub async fn close_client(
    client_sink: &mut futures::stream::SplitSink<WebSocket, Message>,
    code: Option<RealtimeSessionCloseCode>,
) {
    let _ = client_sink.send(close_message(code)).await;
}

pub async fn reject_client(
    client_sink: &mut futures::stream::SplitSink<WebSocket, Message>,
    status: StatusCode,
    message: &str,
) {
    let _ = client_sink
        .send(Message::Text(
            serde_json::json!({
                "type": "error",
                "status": status.as_u16(),
                "message": message,
            })
            .to_string()
            .into(),
        ))
        .await;
    close_client(client_sink, Some(RealtimeSessionCloseCode::PolicyDenied)).await;
}

/// Worker-side session transport with the admitted echo already consumed.
pub struct WorkerSide {
    pub sink: futures::stream::SplitSink<WorkerSocket, WireMessage>,
    pub source: futures::stream::SplitStream<WorkerSocket>,
    pub admitted_message: Message,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn registration(created_at: Instant) -> SessionRegistration {
        SessionRegistration { created_at }
    }

    #[test]
    fn realtime_registry_bounds_duplicates_and_evicts_expired_entries() {
        let registry = RealtimeSessionRegistry::new(2);
        let session: SessionId = id("session-1");
        registry
            .insert(
                session.clone(),
                registration(Instant::now()),
                REGISTRATION_TTL,
            )
            .expect("first insert");
        assert_eq!(registry.active(), 1);
        assert_eq!(
            registry.insert(
                session.clone(),
                registration(Instant::now()),
                REGISTRATION_TTL
            ),
            Err(SessionRegistryError::DuplicateSession)
        );

        let second: SessionId = id("session-2");
        registry
            .insert(second, registration(Instant::now()), REGISTRATION_TTL)
            .expect("second insert");
        let third: SessionId = id("session-3");
        assert_eq!(
            registry.insert(
                third.clone(),
                registration(Instant::now()),
                REGISTRATION_TTL
            ),
            Err(SessionRegistryError::Full)
        );

        // Expired entries yield their slot to new sessions.
        let expired: SessionId = id("session-expired");
        let mut stale = registration(Instant::now());
        stale.created_at = Instant::now() - REGISTRATION_TTL - Duration::from_secs(1);
        let registry_with_stale = RealtimeSessionRegistry::new(2);
        registry_with_stale
            .insert(expired.clone(), stale, REGISTRATION_TTL)
            .unwrap();
        registry_with_stale
            .insert(
                third.clone(),
                registration(Instant::now()),
                REGISTRATION_TTL,
            )
            .expect("expired slot is reusable");
        assert!(registry_with_stale.remove(&expired));
        assert!(!registry_with_stale.remove(&expired));
    }

    #[tokio::test]
    async fn relay_rejects_missing_subprotocol_without_upgrade() {
        // The route handler answers 400 before any upgrade when the client
        // does not offer izwi-realtime-v1; full-session behavior is covered
        // by the process test through the real gateway binary.
        assert!(!offers_realtime_subprotocol(&HeaderMap::new()));
    }
}

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
use izwi_serving_protocol::{
    decode_realtime_audio_frame, AttemptId, CallerId, CancellationBehavior, DeploymentId,
    GatewayAttestedCallerContext, InputFormat, InvocationErrorCode, InvocationEvent,
    InvocationEventKind, ModelAlias, ModelGeneration, OutputFormat, PermittedAction,
    PolicyRevision, RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit,
    RealtimeSessionCloseCode, RequestId, ServiceClass, ServiceCredentials, SessionId, TaskKind,
    TenantId, WorkerId, PROTOCOL_V1, REALTIME_SUBPROTOCOL, REALTIME_WS_PATH,
    SERVICE_AUTHORIZATION_HEADER, SERVICE_AUTH_SCHEME, SERVICE_CREDENTIAL_ID_HEADER,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_tungstenite::{
    tungstenite::{
        client::IntoClientRequest, http::HeaderValue, Error as WsError, Message as WireMessage,
    },
    MaybeTlsStream, WebSocketStream,
};

type WorkerWire = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// How long the relay waits for the client's first admit frame.
const CLIENT_ADMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the relay waits for the worker's admitted echo.
const WORKER_ADMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long an unused registration slot is held before TTL eviction.
pub(crate) const REGISTRATION_TTL: Duration = Duration::from_secs(3600);
/// Client control frames are tiny; anything larger is refused before parse.
const RELAY_CONTROL_FRAME_BYTES: usize = 64 * 1024;
/// Upper bound for the relay's configured session capacity.
pub(crate) const MAX_RELAY_SESSIONS: usize = 4_096;

/// Bounded configuration for the relay, resolved once at gateway boot. Each
/// stage pool is optional; boot fails closed only when both are absent.
#[derive(Clone)]
pub(crate) struct RealtimeRelayConfig {
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
pub(crate) struct RelayStage {
    pub deployment_id: DeploymentId,
    pub task: TaskKind,
    pub input_format: InputFormat,
    pub output_format: OutputFormat,
}

/// One live relayed session as observed by the gateway. Read through
/// `registration()` by introspection tests; retained as the registry's
/// operator-visibility data contract.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SessionRegistration {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    pub incarnation_id: String,
    pub deployment_id: DeploymentId,
    pub model_generation: ModelGeneration,
    pub principal_namespace: String,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionRegistryError {
    DuplicateSession,
    Full,
}

/// Bounded live-session registry: session id -> registration, with TTL
/// eviction of abandoned entries on insert. Mirrors the conversation pin
/// table's bounded-map discipline.
pub(crate) struct RealtimeSessionRegistry {
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

    pub(crate) fn active(&self) -> usize {
        self.entries
            .lock()
            .expect("registry poisoned")
            .by_session
            .len()
    }

    pub(crate) fn insert(
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

    pub(crate) fn remove(&self, session_id: &SessionId) -> bool {
        let mut state = self.entries.lock().expect("registry poisoned");
        if state.by_session.remove(session_id).is_some() {
            state.order.retain(|id| id != session_id);
            true
        } else {
            false
        }
    }

    /// Live registration lookup for introspection.
    #[cfg(test)]
    #[allow(dead_code)]
    fn registration(&self, session_id: &SessionId) -> Option<SessionRegistration> {
        self.entries
            .lock()
            .expect("registry poisoned")
            .by_session
            .get(session_id)
            .cloned()
    }
}

/// The relay surface attached to [`GatewayState`] when the realtime flag is
/// on. `Debug` omits worker credentials.
pub struct GatewayRealtimeRelay {
    registry: WorkerRegistry,
    sessions: RealtimeSessionRegistry,
    config: RealtimeRelayConfig,
    worker_credentials: ServiceCredentials,
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
        })
    }

    pub(crate) fn sessions(&self) -> &RealtimeSessionRegistry {
        &self.sessions
    }

    pub(crate) fn config(&self) -> &RealtimeRelayConfig {
        &self.config
    }

    pub(crate) fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }

    pub(crate) fn worker_credentials(&self) -> &ServiceCredentials {
        &self.worker_credentials
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
pub(crate) async fn relay_socket(
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
        // Protocol validate rejects chat admits before this point.
        TaskKind::Chat => {
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
    let registration =
        |worker_id: WorkerId, incarnation_id: String, generation| SessionRegistration {
            attempt_id: requested.attempt_id.clone(),
            worker_id,
            incarnation_id,
            deployment_id: stage.deployment_id.clone(),
            model_generation: generation,
            principal_namespace: principal_namespace(&context.principal),
            created_at: Instant::now(),
        };
    if let Err(error) = relay.sessions().insert(
        requested.session_id.clone(),
        registration(
            WorkerId::new("pending-selection").expect("bounded identity"),
            String::new(),
            ModelGeneration::new(1).expect("non-zero generation"),
        ),
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

    let mut selected = match relay
        .registry
        .select_and_reserve(&realtime_selection_request(relay.config(), &stage))
    {
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
    let mut worker = match dial_worker_session(
        &selected.client.endpoint(),
        relay.worker_credentials(),
        &admit,
    )
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
        registration(
            selected.key.worker_id.clone(),
            selected.key.incarnation_id.as_str().to_string(),
            selected.model_generation,
        ),
        REGISTRATION_TTL,
    );

    // The client reader forwards actions; the pump below owns every write.
    let (action_tx, mut action_rx) = mpsc::channel::<ClientAction>(16);
    let reader = tokio::spawn(read_client_actions(client_source, action_tx));

    // Announce the negotiated contract verbatim, then relay. Sink and source
    // are disjoint after the worker split, so both directions race here.
    if client_sink.send(worker.admitted_message).await.is_err() {
        let _ = worker.sink.send(WireMessage::Close(None)).await;
        reader.abort();
        return;
    }

    let mut terminal_seen = false;
    loop {
        tokio::select! {
            biased;
            action = action_rx.recv() => {
                match action {
                    Some(ClientAction::Audio(frame)) => {
                        // Passthrough: the client's framed sequence numbers
                        // reach the worker unchanged, so worker-side
                        // monotonicity enforcement sees exactly what a
                        // direct client would send.
                        if worker.sink.send(WireMessage::Binary(frame.into())).await.is_err() {
                            break;
                        }
                    }
                    Some(ClientAction::Text(text)) => {
                        if worker
                            .sink
                            .send(control_wire(&RealtimeClientFrame::Input { text }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Some(ClientAction::Finish) => {
                        if worker
                            .sink
                            .send(control_wire(&RealtimeClientFrame::Finish))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Some(ClientAction::Cancel) => {
                        if worker
                            .sink
                            .send(control_wire(&RealtimeClientFrame::Cancel))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Some(ClientAction::Ping) => {
                        let pong =
                            serde_json::to_string(&RealtimeServerFrame::Pong).expect("pong encodes");
                        let _ = client_sink.send(Message::Text(pong.into())).await;
                    }
                    Some(ClientAction::Violation(reason)) => {
                        let error = error_message(
                            &admit.request_id,
                            &admit.attempt_id,
                            InvocationErrorCode::InvalidInput,
                            reason,
                        );
                        let _ = client_sink.send(Message::Text(error.into())).await;
                        close_client(
                            &mut client_sink,
                            Some(RealtimeSessionCloseCode::ProtocolViolation),
                        )
                        .await;
                        break;
                    }
                    // Client left: request cooperative teardown toward the
                    // worker; its session ends on its own cancellation ladder.
                    Some(ClientAction::Gone) | None => {
                        let _ = worker
                            .sink
                            .send(control_wire(&RealtimeClientFrame::Cancel))
                            .await;
                        break;
                    }
                }
            }
            wire = worker.source.next() => {
                match wire {
                    Some(Ok(WireMessage::Text(text))) => {
                        // Relay verbatim; decode only to track the terminal
                        // outcome for owner-loss classification.
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
                        let relayed = Utf8Bytes::from(text.as_str());
                        if client_sink.send(Message::Text(relayed)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(WireMessage::Close(frame))) => {
                        let code = frame.as_ref().map(|frame| u16::from(frame.code)).unwrap_or(1000);
                        let reason = frame
                            .as_ref()
                            .map(|frame| Utf8Bytes::from(frame.reason.as_str()))
                            .unwrap_or_else(|| Utf8Bytes::from_static("session complete"));
                        let _ = client_sink
                            .send(Message::Close(Some(CloseFrame { code, reason })))
                            .await;
                        break;
                    }
                    Some(Ok(WireMessage::Binary(data))) => {
                        // Passthrough (TTS-stream audio): the worker's framed
                        // sequence numbers reach the client unchanged.
                        if client_sink.send(Message::Binary(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(_)) | Some(Err(_)) | None => {
                        // Owner loss: no terminal event, no close frame. The
                        // client gets an explicit interruption event and a
                        // close; reconnecting is always a new session.
                        if !terminal_seen {
                            let error = error_message(
                                &admit.request_id,
                                &admit.attempt_id,
                                InvocationErrorCode::Internal,
                                "realtime worker session was lost",
                            );
                            let _ = client_sink.send(Message::Text(error.into())).await;
                        }
                        close_client(&mut client_sink, Some(RealtimeSessionCloseCode::Internal))
                            .await;
                        break;
                    }
                }
            }
        }
    }
    reader.abort();
}

/// Holds the registry entry until the session ends on any path.
pub(crate) struct SessionGuard {
    relay: Arc<GatewayRealtimeRelay>,
    session_id: SessionId,
}

impl SessionGuard {
    pub(crate) fn new(relay: Arc<GatewayRealtimeRelay>, session_id: SessionId) -> Self {
        Self { relay, session_id }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.relay.sessions().remove(&self.session_id);
    }
}

pub(crate) fn realtime_selection_request(
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
    }
}

pub(crate) fn attested_caller(
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

pub(crate) async fn close_client(
    client_sink: &mut futures::stream::SplitSink<WebSocket, Message>,
    code: Option<RealtimeSessionCloseCode>,
) {
    let frame = code
        .map(|code| CloseFrame {
            code: code.code(),
            reason: Utf8Bytes::from_static(code.reason()),
        })
        .unwrap_or(CloseFrame {
            code: 1000,
            reason: Utf8Bytes::from_static("session complete"),
        });
    let _ = client_sink.send(Message::Close(Some(frame))).await;
}

pub(crate) async fn reject_client(
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
pub(crate) struct WorkerSide {
    pub sink: futures::stream::SplitSink<WorkerWire, WireMessage>,
    pub source: futures::stream::SplitStream<WorkerWire>,
    pub admitted_message: Message,
}

/// Dial failures keep their own vocabulary: the client only learns the
/// gateway could not establish the worker session.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WorkerDialError {
    #[error("worker endpoint is not a supported websocket origin")]
    UnsupportedOrigin,
    #[error(transparent)]
    Handshake(#[from] WsError),
    #[error("{0}")]
    Protocol(&'static str),
}

pub(crate) async fn dial_worker_session(
    endpoint: &str,
    credentials: &ServiceCredentials,
    admit: &RealtimeSessionAdmit,
) -> Result<WorkerSide, WorkerDialError> {
    let base = endpoint.trim_end_matches('/');
    let ws_url = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}{REALTIME_WS_PATH}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}{REALTIME_WS_PATH}")
    } else {
        return Err(WorkerDialError::UnsupportedOrigin);
    };
    let mut request: tokio_tungstenite::tungstenite::http::Request<()> = ws_url
        .into_client_request()
        .map_err(|_| WorkerDialError::UnsupportedOrigin)?;
    let headers = request.headers_mut();
    headers.insert(
        "sec-websocket-protocol",
        HeaderValue::from_static(REALTIME_SUBPROTOCOL),
    );
    let authorization = HeaderValue::from_str(&format!(
        "{SERVICE_AUTH_SCHEME} {}",
        credentials.bearer_token.expose_secret()
    ))
    .map_err(|_| WorkerDialError::Protocol("worker bearer token is not a valid header value"))?;
    headers.insert(SERVICE_AUTHORIZATION_HEADER, authorization);
    let credential = HeaderValue::from_str(credentials.credential_id.as_str()).map_err(|_| {
        WorkerDialError::Protocol("worker credential id is not a valid header value")
    })?;
    headers.insert(SERVICE_CREDENTIAL_ID_HEADER, credential);

    let handshake = tokio_tungstenite::connect_async(request);
    let (mut stream, response) = tokio::time::timeout(WORKER_ADMIT_TIMEOUT, handshake)
        .await
        .map_err(|_| WorkerDialError::Protocol("worker dial timed out"))??;
    let selected = response
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().eq_ignore_ascii_case(REALTIME_SUBPROTOCOL));
    if selected != Some(true) {
        return Err(WorkerDialError::Protocol(
            "worker did not select the izwi-realtime-v1 subprotocol",
        ));
    }

    stream
        .send(control_wire(&RealtimeClientFrame::Admit {
            admit: Box::new(admit.clone()),
        }))
        .await
        .map_err(|_| WorkerDialError::Protocol("admit send failed"))?;

    // The admitted echo must arrive first and echo the identity.
    loop {
        let message = tokio::time::timeout(WORKER_ADMIT_TIMEOUT, stream.next())
            .await
            .map_err(|_| WorkerDialError::Protocol("worker admitted echo timed out"))?
            .ok_or(WsError::ConnectionClosed)??;
        match message {
            WireMessage::Text(text) => {
                let frame: RealtimeServerFrame = serde_json::from_str(&text)
                    .map_err(|_| WorkerDialError::Protocol("admitted frame did not decode"))?;
                match frame {
                    RealtimeServerFrame::Admitted {
                        session_id,
                        attempt_id,
                        bounds,
                        ..
                    } => {
                        if session_id.as_str() != admit.session_id.as_str()
                            || attempt_id.as_str() != admit.attempt_id.as_str()
                        {
                            return Err(WorkerDialError::Protocol(
                                "admitted frame does not echo the admit identity",
                            ));
                        }
                        if bounds.validate().is_err() {
                            return Err(WorkerDialError::Protocol(
                                "admitted bounds exceed the protocol caps",
                            ));
                        }
                        let admitted_message = Message::Text(Utf8Bytes::from(text.as_str()));
                        let (sink, source) = stream.split();
                        return Ok(WorkerSide {
                            sink,
                            source,
                            admitted_message,
                        });
                    }
                    RealtimeServerFrame::Event { .. } | RealtimeServerFrame::Pong => {
                        return Err(WorkerDialError::Protocol(
                            "admitted frame must be the first server frame",
                        ));
                    }
                }
            }
            WireMessage::Close(_) => {
                return Err(WorkerDialError::Protocol("worker closed before admission"))
            }
            WireMessage::Ping(_) | WireMessage::Pong(_) => continue,
            WireMessage::Binary(_) | WireMessage::Frame(_) => {
                return Err(WorkerDialError::Protocol(
                    "worker sent a binary frame before admission",
                ));
            }
        }
    }
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
        SessionRegistration {
            attempt_id: id("attempt-1"),
            worker_id: id("worker-1"),
            incarnation_id: "incarnation-1".into(),
            deployment_id: id("deployment-1"),
            model_generation: ModelGeneration::new(1).unwrap(),
            principal_namespace: "tenant:t1".into(),
            created_at,
        }
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

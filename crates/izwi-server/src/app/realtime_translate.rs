//! Gateway realtime translator (DS3.4): serves the public
//! transcription-realtime surface (`transcription_realtime_v2` and the typed
//! `transcription_realtime` v3 envelopes) by translating it onto the internal
//! `izwi-realtime-v1` worker subprotocol.
//!
//! A client of the single-node `/v1/speech-to-text/realtime/ws` surface can
//! repoint at the gateway unchanged: the gateway negotiates the same wire
//! modes (legacy v2 JSON, typed v3 envelopes), translates the public ITRW
//! audio framing onto the worker's IRTA framing, and maps worker invocation
//! events back onto the public event vocabulary. Clients offering
//! `izwi-realtime-v1` keep the byte-identical passthrough relay in
//! [`crate::app::realtime_relay`]; this module only serves the public
//! transcription envelope.
//!
//! Admission is deferred to the first audio frame: the public envelope
//! declares the sample rate per frame, while the worker admit needs a
//! concrete `RealtimeAudioSpec`. A session that never streams audio finishes
//! locally without dialing a worker. Worker deltas accumulate into the
//! replaceable partial hypothesis; the last delta before the terminal event
//! carries the full final text and replaces the accumulation.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use izwi_serving_protocol::{
    encode_realtime_audio_frame, AttemptId, AttemptIdentity, InputFormat, InvocationErrorCode,
    InvocationEvent, InvocationEventKind, ModelGeneration, OutputFormat, RealtimeAudioCodec,
    RealtimeAudioSpec, RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit,
    RealtimeSessionCloseCode, RealtimeStageInput, RequestId, ServiceClass, SessionId, TaskKind,
    WorkerId, MAX_REALTIME_AUDIO_FRAME_BYTES, PROTOCOL_V1,
};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message as WireMessage;

use crate::api::request_context::{principal_namespace, RequestContext};
use crate::app::realtime_protocol::{
    RealtimeAudioGapAction, RealtimeClose, RealtimeCloseCode, RealtimeCloseReason,
    RealtimeErrorCode, RealtimeEventEnvelope, RealtimeProtocol, RealtimeServerEnvelope,
    RealtimeServerEvent, TRANSCRIPTION_REALTIME_VERSION,
};
use crate::app::realtime_relay::{
    attested_caller, close_client, dial_worker_session, realtime_selection_request, reject_client,
    GatewayRealtimeRelay, RelayStage, SessionGuard, SessionRegistration, SessionRegistryError,
    WorkerSide, REGISTRATION_TTL,
};
use crate::app::transcription_realtime::{
    negotiate_transcription_protocol, parse_binary_message, BinaryMessageKind, ClientEvent,
    TranscriptionWireProtocol, LEGACY_REALTIME_PROTOCOL,
};
use crate::gateway::GatewayState;
use crate::gateway_tenant_concurrency::BoundTenantWorkLease;

type ClientSink = futures::stream::SplitSink<WebSocket, Message>;

/// Public control frames are tiny; anything larger is refused before parse.
const PUBLIC_CONTROL_FRAME_BYTES: usize = 64 * 1024;
/// Bound on one public audio frame's PCM payload. Matches the single-node
/// transcription surface's frame cap, which equals the worker protocol's hard
/// frame cap.
const PUBLIC_FRAME_BYTES: usize = MAX_REALTIME_AUDIO_FRAME_BYTES;
/// Input command queue depth. The pump drains eagerly, so this only bounds
/// bursty client input the way the single-node socket does.
const TRANSLATE_ACTION_QUEUE_CAPACITY: usize = 512;

/// One parsed client message, shaped into a pump action by the reader task.
enum TranslateAction {
    /// The client's `model_id` hint is dropped here: the gateway routes by
    /// its approved deployment, never by the client's model string.
    Start {
        language: Option<String>,
        protocol: Option<String>,
        version: Option<u16>,
        resume_from_event_id: Option<u64>,
    },
    Stop,
    Ping {
        timestamp_ms: Option<u64>,
    },
    Audio {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
    /// Unparseable text frame. Mirrors the single-node surface, which answers
    /// parse failures with the legacy error shape regardless of wire mode and
    /// keeps the session open.
    InvalidText(String),
    /// Malformed binary audio frame. Mirrors the single-node session error
    /// path, which is wire-mode aware.
    InvalidBinary(String),
    Gone,
}

/// Gateway-minted invocation identity for the translated worker session.
struct TranslateIdentity {
    session_id: SessionId,
    request_id: RequestId,
    attempt_id: AttemptId,
}

/// Session-scoped translator state, owned by the pump task.
struct TranslateSession {
    wire: TranscriptionWireProtocol,
    started: bool,
    stop_requested: bool,
    language: Option<String>,
    identity: Option<TranslateIdentity>,
    typed: Option<TypedCounters>,
    transcript_sequence: u64,
    transcript: TranscriptAccumulator,
    sample_rate_lock: Option<u32>,
    last_accepted_frame: Option<u32>,
}

impl TranslateSession {
    fn new() -> Self {
        Self {
            wire: TranscriptionWireProtocol::LegacyV2,
            started: false,
            stop_requested: false,
            language: None,
            identity: None,
            typed: None,
            transcript_sequence: 0,
            transcript: TranscriptAccumulator::new(),
            sample_rate_lock: None,
            last_accepted_frame: None,
        }
    }

    fn start(
        &mut self,
        wire: TranscriptionWireProtocol,
        language: Option<String>,
        identity: TranslateIdentity,
    ) {
        self.wire = wire;
        self.started = true;
        self.language = language.filter(|value| !value.trim().is_empty());
        self.typed = match wire {
            TranscriptionWireProtocol::LegacyV2 => None,
            TranscriptionWireProtocol::TypedV3 => Some(TypedCounters::new(
                identity.session_id.as_str().to_string(),
                format!("gateway-{}", std::process::id()),
            )),
        };
        self.identity = Some(identity);
    }
}

/// Envelope counters for the typed v3 wire mode, mirroring the single-node
/// typed session: event ids start at 1, sequences at 0, one epoch.
struct TypedCounters {
    session_id: String,
    owner_instance_id: String,
    next_event_id: u64,
    next_sequence: u64,
    next_revision: u64,
}

impl TypedCounters {
    fn new(session_id: String, owner_instance_id: String) -> Self {
        Self {
            session_id,
            owner_instance_id,
            next_event_id: 1,
            next_sequence: 0,
            next_revision: 1,
        }
    }

    fn next_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }

    fn next_envelope(&mut self, event: RealtimeServerEvent) -> RealtimeServerEnvelope {
        let envelope = RealtimeEventEnvelope {
            protocol: RealtimeProtocol::TranscriptionRealtime,
            version: TRANSCRIPTION_REALTIME_VERSION,
            event_id: self.next_event_id,
            sequence: self.next_sequence,
            session_id: self.session_id.clone(),
            connection_epoch: 0,
            timestamp_ms: now_unix_millis(),
            utterance_id: None,
            turn_id: None,
            segment_id: None,
            event,
        };
        self.next_event_id = self.next_event_id.saturating_add(1);
        self.next_sequence = self.next_sequence.saturating_add(1);
        envelope
    }
}

/// Worker delta accumulation with one-delta hold-back.
///
/// The worker contract makes every delta an incremental fragment except the
/// last pre-terminal one, which carries the full final text and replaces the
/// accumulation. Because a delta's role is only known once the terminal
/// event (or the next delta) arrives, the newest delta is held back: pushing
/// delta N+1 folds delta N into the accumulated hypothesis and emits it as
/// the replaceable partial, and the terminal event resolves the held delta
/// as the authoritative final text.
#[derive(Default)]
struct TranscriptAccumulator {
    accumulated: String,
    held: Option<String>,
}

impl TranscriptAccumulator {
    fn new() -> Self {
        Self::default()
    }

    /// Records one worker delta. Returns the replaceable partial hypothesis
    /// to emit, or `None` while every received delta is still provisional.
    fn push(&mut self, delta: String) -> Option<String> {
        if let Some(previous) = self.held.take() {
            self.accumulated.push_str(&previous);
        }
        self.held = Some(delta);
        if self.accumulated.is_empty() {
            None
        } else {
            Some(self.accumulated.clone())
        }
    }

    /// Resolves the held delta as the worker's full final text.
    fn finish(&mut self) -> Option<String> {
        self.held.take()
    }
}

/// Worker invocation error codes mapped onto the public v3 error vocabulary.
fn map_worker_error_code(code: InvocationErrorCode) -> RealtimeErrorCode {
    match code {
        InvocationErrorCode::InvalidInput => RealtimeErrorCode::InvalidMessage,
        InvocationErrorCode::ExecutionFailed | InvocationErrorCode::DeadlineExceeded => {
            RealtimeErrorCode::InferenceFailed
        }
        InvocationErrorCode::OutputLimitExceeded => RealtimeErrorCode::BufferLimit,
        InvocationErrorCode::WorkerUnavailable => RealtimeErrorCode::ModelUnavailable,
        InvocationErrorCode::Internal => RealtimeErrorCode::Internal,
    }
}

/// Locks the session's input sample rate on the first accepted frame and
/// rejects mid-stream changes, mirroring the single-node ingest contract.
fn lock_sample_rate(lock: &mut Option<u32>, sample_rate: u32) -> Result<(), String> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(format!("Invalid input sample_rate {sample_rate}"));
    }
    match lock {
        Some(current) if *current != sample_rate => Err(format!(
            "Input sample rate changed mid-stream ({current} -> {sample_rate})"
        )),
        Some(_) => Ok(()),
        None => {
            *lock = Some(sample_rate);
            Ok(())
        }
    }
}

/// Detects a gap in the client's frame sequence. Stale and in-order frames
/// report no gap.
fn detect_audio_gap(last_accepted: Option<u32>, frame_seq: u32) -> Option<(u64, u64, u64)> {
    let expected = u64::from(last_accepted?) + 1;
    if u64::from(frame_seq) <= expected {
        return None;
    }
    Some((
        expected,
        u64::from(frame_seq),
        u64::from(frame_seq) - expected,
    ))
}

/// Entry point used by the gateway route when the client does not offer the
/// `izwi-realtime-v1` subprotocol: the socket speaks the public
/// transcription-realtime envelope instead.
pub(crate) async fn run_translate_session(
    state: GatewayState,
    relay: Arc<GatewayRealtimeRelay>,
    socket: WebSocket,
    context: RequestContext,
) {
    let (mut client_sink, client_source) = socket.split();

    // The public surface announces itself before any client frame, exactly
    // like the single-node socket; typed v3 clients see this legacy shape
    // first there too.
    let ready = serde_json::json!({
        "type": "session_ready",
        "protocol": LEGACY_REALTIME_PROTOCOL,
        "correlation_id": context.correlation_id.clone(),
    });
    let _ = client_sink
        .send(Message::Text(ready.to_string().into()))
        .await;

    let (action_tx, mut action_rx) =
        mpsc::channel::<TranslateAction>(TRANSLATE_ACTION_QUEUE_CAPACITY);
    let reader = tokio::spawn(read_public_actions(client_source, action_tx));

    let mut session = TranslateSession::new();
    // Held for its Drop: the registry entry lives until the session ends on
    // any path.
    let mut _session_guard: Option<SessionGuard> = None;
    let mut tenant_lease: Option<BoundTenantWorkLease> = None;

    // Phase one: negotiation and deferred admission. The worker is dialed by
    // the first accepted audio frame; everything before that is served
    // locally.
    let mut worker: Option<WorkerSide> = None;
    loop {
        let Some(action) = action_rx.recv().await else {
            finish_public_actions(&mut client_sink, &mut session, TranslateAction::Gone).await;
            break;
        };
        match action {
            TranslateAction::Start {
                language,
                protocol,
                version,
                resume_from_event_id,
            } => {
                if session.started {
                    let _ = client_sink
                        .send(error_text("session already started"))
                        .await;
                    continue;
                }
                let wire = match negotiate_transcription_protocol(
                    protocol.as_deref(),
                    version,
                    resume_from_event_id,
                ) {
                    Ok(wire) => wire,
                    Err(err) => {
                        let _ = client_sink.send(error_text(&err)).await;
                        continue;
                    }
                };
                // Stage availability gates the session the same way an
                // unconfigured stage refuses a v1 admit: fail closed before
                // any accounting.
                let Some(stage) = asr_stage(&relay) else {
                    close_client(
                        &mut client_sink,
                        Some(RealtimeSessionCloseCode::PolicyDenied),
                    )
                    .await;
                    break;
                };
                let identity = match mint_identity() {
                    Ok(identity) => identity,
                    Err(message) => {
                        let _ = client_sink.send(error_text(&message)).await;
                        continue;
                    }
                };
                let registration = SessionRegistration {
                    attempt_id: identity.attempt_id.clone(),
                    worker_id: WorkerId::new("pending-selection").expect("bounded identity"),
                    incarnation_id: String::new(),
                    deployment_id: stage.deployment_id.clone(),
                    model_generation: ModelGeneration::new(1).expect("non-zero generation"),
                    principal_namespace: principal_namespace(&context.principal),
                    created_at: std::time::Instant::now(),
                };
                match relay.sessions().insert(
                    identity.session_id.clone(),
                    registration,
                    REGISTRATION_TTL,
                ) {
                    Ok(()) => {}
                    Err(SessionRegistryError::DuplicateSession) => {
                        close_client(
                            &mut client_sink,
                            Some(RealtimeSessionCloseCode::DuplicateAttempt),
                        )
                        .await;
                        break;
                    }
                    Err(SessionRegistryError::Full) => {
                        close_client(
                            &mut client_sink,
                            Some(RealtimeSessionCloseCode::CapacityExhausted),
                        )
                        .await;
                        break;
                    }
                }
                _session_guard = Some(SessionGuard::new(
                    Arc::clone(&relay),
                    identity.session_id.clone(),
                ));
                session.start(wire, language, identity);
                announce_session_started(&mut client_sink, &mut session).await;
            }
            TranslateAction::Audio {
                frame_seq,
                sample_rate,
                payload,
            } => {
                if !accept_public_audio(
                    &mut client_sink,
                    &mut session,
                    frame_seq,
                    sample_rate,
                    &payload,
                )
                .await
                {
                    continue;
                }
                if worker.is_none() {
                    match dial_asr_worker(
                        &state,
                        &relay,
                        &context,
                        &session,
                        sample_rate,
                        &mut tenant_lease,
                    )
                    .await
                    {
                        Ok(dialed) => worker = Some(dialed),
                        Err(rejection) => {
                            match rejection {
                                WorkerRejection::Reject(status, message) => {
                                    reject_client(&mut client_sink, status, &message).await;
                                }
                                WorkerRejection::Close(code) => {
                                    close_client(&mut client_sink, Some(code)).await;
                                }
                            }
                            break;
                        }
                    }
                }
                let Some(dialed) = worker.as_mut() else {
                    break;
                };
                let frame = match encode_realtime_audio_frame(frame_seq, false, &payload) {
                    Ok(frame) => frame,
                    Err(_) => break,
                };
                if dialed
                    .sink
                    .send(WireMessage::Binary(frame.into()))
                    .await
                    .is_err()
                {
                    break;
                }
                // Admission is complete: the pump continues in phase two.
                break;
            }
            other => {
                let ended = session_ended(&other);
                finish_public_actions(&mut client_sink, &mut session, other).await;
                if ended {
                    break;
                }
            }
        }
    }

    let Some(mut worker) = worker else {
        reader.abort();
        return;
    };

    // Phase two: the worker session is live. Client audio is re-encoded onto
    // the worker subprotocol; worker events are translated back onto the
    // public vocabulary. Sink and source are disjoint after the worker split,
    // so both directions race here.
    loop {
        tokio::select! {
            biased;
            action = action_rx.recv() => {
                let Some(action) = action else {
                    let _ = worker
                        .sink
                        .send(control_wire(&RealtimeClientFrame::Cancel))
                        .await;
                    break;
                };
                match action {
                    TranslateAction::Start { .. } => {
                        let _ = client_sink
                            .send(error_text("session already started"))
                            .await;
                    }
                    TranslateAction::Audio { frame_seq, sample_rate, payload } => {
                        // Client input after session stop is not read by the
                        // single-node surface; the worker's Finish contract
                        // likewise ends its input stage.
                        if session.stop_requested {
                            continue;
                        }
                        if !accept_public_audio(
                            &mut client_sink,
                            &mut session,
                            frame_seq,
                            sample_rate,
                            &payload,
                        )
                        .await
                        {
                            continue;
                        }
                        let frame = match encode_realtime_audio_frame(frame_seq, false, &payload) {
                            Ok(frame) => frame,
                            Err(_) => break,
                        };
                        if worker
                            .sink
                            .send(WireMessage::Binary(frame.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    TranslateAction::Stop => {
                        if session.stop_requested {
                            continue;
                        }
                        session.stop_requested = true;
                        if worker
                            .sink
                            .send(control_wire(&RealtimeClientFrame::Finish))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    TranslateAction::Ping { timestamp_ms } => {
                        send_pong(&mut client_sink, &mut session, timestamp_ms).await;
                    }
                    TranslateAction::InvalidText(message) => {
                        let _ = client_sink.send(error_text(&message)).await;
                    }
                    TranslateAction::InvalidBinary(message) => {
                        send_session_error(&mut client_sink, &mut session, message).await;
                    }
                    TranslateAction::Gone => {
                        // Client left: request cooperative teardown toward the
                        // worker; its session ends on its own cancellation
                        // ladder.
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
                        let frame = serde_json::from_str::<RealtimeServerFrame>(&text);
                        match frame {
                            Ok(RealtimeServerFrame::Event { event }) => {
                                if handle_invocation_event(&mut client_sink, &mut session, event).await {
                                    break;
                                }
                            }
                            // The admitted echo was consumed by the dial; later
                            // echoes and worker pongs have no public meaning.
                            Ok(_) => {}
                            Err(_) => {
                                send_session_error(
                                    &mut client_sink,
                                    &mut session,
                                    "realtime worker sent an unparseable control frame",
                                )
                                .await;
                            }
                        }
                    }
                    // The ASR stage never emits audio toward the gateway.
                    Some(Ok(WireMessage::Binary(_))) => {
                        send_session_error(
                            &mut client_sink,
                            &mut session,
                            "realtime worker sent an unexpected binary frame",
                        )
                        .await;
                        close_client(
                            &mut client_sink,
                            Some(RealtimeSessionCloseCode::Internal),
                        )
                        .await;
                        break;
                    }
                    Some(Ok(WireMessage::Close(_))) | Some(Err(_)) | None => {
                        // Owner loss: no terminal event, no close frame. The
                        // client gets an explicit error and a close;
                        // reconnecting is always a new session.
                        send_session_error(
                            &mut client_sink,
                            &mut session,
                            "realtime worker session was lost",
                        )
                        .await;
                        close_client(
                            &mut client_sink,
                            Some(RealtimeSessionCloseCode::Internal),
                        )
                        .await;
                        break;
                    }
                    Some(Ok(WireMessage::Ping(_) | WireMessage::Pong(_) | WireMessage::Frame(_))) => {}
                }
            }
        }
    }
    reader.abort();
}

/// Whether the action ends the translated session outright.
fn session_ended(action: &TranslateAction) -> bool {
    matches!(action, TranslateAction::Stop | TranslateAction::Gone)
}

/// Actions that end the session locally during phase one: stop (no worker to
/// wait for) and client disconnect.
async fn finish_public_actions(
    client_sink: &mut ClientSink,
    session: &mut TranslateSession,
    action: TranslateAction,
) {
    match action {
        TranslateAction::Stop => finish_session_locally(client_sink, session).await,
        TranslateAction::Ping { timestamp_ms } => {
            send_pong(client_sink, session, timestamp_ms).await
        }
        TranslateAction::InvalidText(message) => {
            let _ = client_sink.send(error_text(&message)).await;
        }
        TranslateAction::InvalidBinary(message) => {
            send_session_error(client_sink, session, message).await;
        }
        TranslateAction::Start { .. } | TranslateAction::Audio { .. } => {}
        TranslateAction::Gone => {}
    }
}

/// Session stop with no worker dialed: the single-node surface finishes the
/// session locally, so the translator does too.
async fn finish_session_locally(client_sink: &mut ClientSink, session: &mut TranslateSession) {
    match session.wire {
        TranscriptionWireProtocol::LegacyV2 => {
            let _ = client_sink
                .send(json_text(serde_json::json!({ "type": "session_done" })))
                .await;
        }
        TranscriptionWireProtocol::TypedV3 => {
            if let Some(typed) = session.typed.as_mut() {
                // Mirrors the single-node typed finish: an empty final
                // transcript, then the closing ladder.
                let revision = typed.next_revision();
                emit_typed(
                    client_sink,
                    typed,
                    RealtimeServerEvent::TranscriptFinal {
                        text: session.transcript.finish().unwrap_or_default(),
                        revision,
                        language: session.language.clone(),
                    },
                )
                .await;
                let close = normal_close("transcription session stopped by client");
                emit_typed(
                    client_sink,
                    typed,
                    RealtimeServerEvent::Closing {
                        close: close.clone(),
                    },
                )
                .await;
                emit_typed(client_sink, typed, RealtimeServerEvent::Closed { close }).await;
            }
        }
    }
    close_client(client_sink, None).await;
}

/// Validates one public audio frame against the single-node ingest contract
/// and emits the typed v3 ingress events. Returns false for frames the
/// single-node surface ignores (stale, empty).
async fn accept_public_audio(
    client_sink: &mut ClientSink,
    session: &mut TranslateSession,
    frame_seq: u32,
    sample_rate: u32,
    payload: &[u8],
) -> bool {
    if !session.started {
        send_session_error(
            client_sink,
            session,
            "session_start is required before streaming audio",
        )
        .await;
        return false;
    }
    if payload.is_empty() {
        return false;
    }
    if payload.len() > PUBLIC_FRAME_BYTES {
        send_session_error(
            client_sink,
            session,
            format!(
                "Audio frame exceeded max size ({} > {})",
                payload.len(),
                PUBLIC_FRAME_BYTES
            ),
        )
        .await;
        return false;
    }
    if !payload.len().is_multiple_of(2) {
        send_session_error(client_sink, session, "PCM16 payload length must be even").await;
        return false;
    }
    if let Err(err) = lock_sample_rate(&mut session.sample_rate_lock, sample_rate) {
        send_session_error(client_sink, session, err).await;
        return false;
    }
    if session
        .last_accepted_frame
        .is_some_and(|last| frame_seq <= last)
    {
        return false;
    }

    if session.wire == TranscriptionWireProtocol::TypedV3 {
        if let Some((expected, received, missing)) =
            detect_audio_gap(session.last_accepted_frame, frame_seq)
        {
            if let Some(typed) = session.typed.as_mut() {
                emit_typed(
                    client_sink,
                    typed,
                    RealtimeServerEvent::AudioGap {
                        expected_frame_sequence: expected,
                        received_frame_sequence: received,
                        missing_frames: missing,
                        action: RealtimeAudioGapAction::Continue,
                    },
                )
                .await;
            }
        }
        if let Some(typed) = session.typed.as_mut() {
            emit_typed(
                client_sink,
                typed,
                RealtimeServerEvent::AudioAccepted {
                    frame_sequence: u64::from(frame_seq),
                    buffer_depth_samples: payload.len() / 2,
                    ingress_queue_depth: 0,
                },
            )
            .await;
        }
    }

    session.last_accepted_frame = Some(frame_seq);
    true
}

/// Translates one worker invocation event onto the public vocabulary.
/// Returns true when the session is over and the pump should stop.
async fn handle_invocation_event(
    client_sink: &mut ClientSink,
    session: &mut TranslateSession,
    event: InvocationEvent,
) -> bool {
    match event.event {
        InvocationEventKind::Accepted { .. } | InvocationEventKind::Usage { .. } => {}
        InvocationEventKind::TextDelta { text, .. } => {
            if let Some(partial) = session.transcript.push(text) {
                match session.wire {
                    TranscriptionWireProtocol::LegacyV2 => {
                        let payload = legacy_partial_json(
                            &mut session.transcript_sequence,
                            partial,
                            session.language.clone(),
                            false,
                        );
                        let _ = client_sink.send(json_text(payload)).await;
                    }
                    TranscriptionWireProtocol::TypedV3 => {
                        if let Some(typed) = session.typed.as_mut() {
                            let revision = typed.next_revision();
                            emit_typed(
                                client_sink,
                                typed,
                                RealtimeServerEvent::TranscriptPartial {
                                    text: partial,
                                    revision,
                                    language: session.language.clone(),
                                },
                            )
                            .await;
                        }
                    }
                }
            }
        }
        InvocationEventKind::Completed { .. } => {
            let final_text = session.transcript.finish().unwrap_or_default();
            match session.wire {
                TranscriptionWireProtocol::LegacyV2 => {
                    // The single-node legacy surface never emits a standalone
                    // empty final; session_done is the terminal shape.
                    if !final_text.is_empty() {
                        let payload = legacy_partial_json(
                            &mut session.transcript_sequence,
                            final_text,
                            session.language.clone(),
                            true,
                        );
                        let _ = client_sink.send(json_text(payload)).await;
                    }
                    let _ = client_sink
                        .send(json_text(serde_json::json!({ "type": "session_done" })))
                        .await;
                }
                TranscriptionWireProtocol::TypedV3 => {
                    if let Some(typed) = session.typed.as_mut() {
                        let revision = typed.next_revision();
                        emit_typed(
                            client_sink,
                            typed,
                            RealtimeServerEvent::TranscriptFinal {
                                text: final_text,
                                revision,
                                language: session.language.clone(),
                            },
                        )
                        .await;
                        let close = normal_close("transcription session stopped by client");
                        emit_typed(
                            client_sink,
                            typed,
                            RealtimeServerEvent::Closing {
                                close: close.clone(),
                            },
                        )
                        .await;
                        emit_typed(client_sink, typed, RealtimeServerEvent::Closed { close }).await;
                    }
                }
            }
            close_client(client_sink, None).await;
            return true;
        }
        InvocationEventKind::Error { code, message } => {
            match session.wire {
                TranscriptionWireProtocol::LegacyV2 => {
                    let _ = client_sink.send(error_text(&message)).await;
                }
                TranscriptionWireProtocol::TypedV3 => {
                    if let Some(typed) = session.typed.as_mut() {
                        emit_typed(
                            client_sink,
                            typed,
                            RealtimeServerEvent::FatalError {
                                code: map_worker_error_code(code),
                                message: message.clone(),
                                close: RealtimeClose {
                                    code: RealtimeCloseCode::InternalError,
                                    reason: RealtimeCloseReason::InternalError,
                                    message,
                                    retryable: false,
                                },
                            },
                        )
                        .await;
                    }
                }
            }
            close_client(client_sink, Some(RealtimeSessionCloseCode::Internal)).await;
            return true;
        }
        InvocationEventKind::Cancelled { .. } => {
            match session.wire {
                TranscriptionWireProtocol::LegacyV2 => {
                    let _ = client_sink
                        .send(json_text(serde_json::json!({ "type": "session_done" })))
                        .await;
                }
                TranscriptionWireProtocol::TypedV3 => {
                    if let Some(typed) = session.typed.as_mut() {
                        let close = normal_close("transcription session cancelled");
                        emit_typed(
                            client_sink,
                            typed,
                            RealtimeServerEvent::Closing {
                                close: close.clone(),
                            },
                        )
                        .await;
                        emit_typed(client_sink, typed, RealtimeServerEvent::Closed { close }).await;
                    }
                }
            }
            close_client(client_sink, None).await;
            return true;
        }
    }
    false
}

async fn announce_session_started(client_sink: &mut ClientSink, session: &mut TranslateSession) {
    match session.wire {
        TranscriptionWireProtocol::LegacyV2 => {
            let _ = client_sink
                .send(json_text(serde_json::json!({ "type": "session_started" })))
                .await;
        }
        TranscriptionWireProtocol::TypedV3 => {
            let Some(typed) = session.typed.as_mut() else {
                return;
            };
            let owner_instance_id = typed.owner_instance_id.clone();
            emit_typed(
                client_sink,
                typed,
                RealtimeServerEvent::SessionReady {
                    accepted_version: TRANSCRIPTION_REALTIME_VERSION,
                    owner_instance_id,
                    resumable: false,
                    resume_window_ms: 0,
                },
            )
            .await;
            emit_typed(client_sink, typed, RealtimeServerEvent::SessionStarted).await;
        }
    }
}

async fn send_pong(
    client_sink: &mut ClientSink,
    session: &mut TranslateSession,
    timestamp_ms: Option<u64>,
) {
    match session.wire {
        TranscriptionWireProtocol::LegacyV2 => {
            let _ = client_sink
                .send(json_text(serde_json::json!({
                    "type": "pong",
                    "timestamp_ms": timestamp_ms,
                })))
                .await;
        }
        TranscriptionWireProtocol::TypedV3 => {
            if let Some(typed) = session.typed.as_mut() {
                emit_typed(
                    client_sink,
                    typed,
                    RealtimeServerEvent::Pong {
                        client_timestamp_ms: timestamp_ms,
                        server_timestamp_ms: now_unix_millis(),
                    },
                )
                .await;
            }
        }
    }
}

/// Wire-mode aware session error, mirroring the single-node
/// `send_session_error`: legacy error JSON on v2, recoverable error envelope
/// on v3.
async fn send_session_error(
    client_sink: &mut ClientSink,
    session: &mut TranslateSession,
    message: impl Into<String>,
) {
    let message = message.into();
    match session.wire {
        TranscriptionWireProtocol::LegacyV2 => {
            let _ = client_sink.send(error_text(&message)).await;
        }
        TranscriptionWireProtocol::TypedV3 => {
            if let Some(typed) = session.typed.as_mut() {
                emit_typed(
                    client_sink,
                    typed,
                    RealtimeServerEvent::RecoverableError {
                        code: RealtimeErrorCode::InvalidMessage,
                        message,
                        retry_after_ms: None,
                    },
                )
                .await;
            }
        }
    }
}

async fn emit_typed(
    client_sink: &mut ClientSink,
    typed: &mut TypedCounters,
    event: RealtimeServerEvent,
) -> bool {
    let envelope = typed.next_envelope(event);
    match serde_json::to_string(&envelope) {
        Ok(text) => client_sink.send(Message::Text(text.into())).await.is_ok(),
        Err(_) => false,
    }
}

fn legacy_partial_json(
    transcript_sequence: &mut u64,
    text: String,
    language: Option<String>,
    is_final: bool,
) -> serde_json::Value {
    *transcript_sequence = transcript_sequence.saturating_add(1);
    let mut payload = serde_json::json!({
        "type": "transcript_partial",
        "sequence": *transcript_sequence,
        "text": text,
        "language": language,
        "audio_duration_secs": 0.0,
        "processing_time_ms": 0.0,
        "rtf": serde_json::Value::Null,
    });
    if is_final {
        payload["is_final"] = serde_json::Value::Bool(true);
    }
    payload
}

fn normal_close(message: &str) -> RealtimeClose {
    RealtimeClose {
        code: RealtimeCloseCode::Normal,
        reason: RealtimeCloseReason::ClientRequest,
        message: message.to_string(),
        retryable: false,
    }
}

fn error_text(message: &str) -> Message {
    json_text(serde_json::json!({ "type": "error", "message": message }))
}

fn json_text(value: serde_json::Value) -> Message {
    Message::Text(value.to_string().into())
}

fn control_wire(frame: &RealtimeClientFrame) -> WireMessage {
    WireMessage::Text(
        serde_json::to_string(frame)
            .expect("control frame encodes")
            .into(),
    )
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// The approved speech_to_text stage, refused explicitly when unconfigured.
fn asr_stage(relay: &GatewayRealtimeRelay) -> Option<RelayStage> {
    relay
        .config()
        .deployment_id
        .as_ref()
        .map(|deployment_id| RelayStage {
            deployment_id: deployment_id.clone(),
            task: TaskKind::SpeechToText,
            input_format: InputFormat::PcmAudio,
            output_format: OutputFormat::Text,
        })
}

fn mint_identity() -> Result<TranslateIdentity, String> {
    let session = uuid::Uuid::new_v4().to_string();
    let request = uuid::Uuid::new_v4().to_string();
    let attempt = uuid::Uuid::new_v4().to_string();
    Ok(TranslateIdentity {
        session_id: SessionId::new(format!("gateway-translate-{session}"))
            .map_err(|err| format!("minted session id was rejected: {err}"))?,
        request_id: RequestId::new(format!("gateway-translate-{request}"))
            .map_err(|err| format!("minted request id was rejected: {err}"))?,
        attempt_id: AttemptId::new(format!("gateway-translate-{attempt}"))
            .map_err(|err| format!("minted attempt id was rejected: {err}"))?,
    })
}

/// Worker dial failures keep their own vocabulary: pre-dial rejections are
/// either an HTTP-shaped rejection or a websocket close.
enum WorkerRejection {
    Reject(axum::http::StatusCode, String),
    Close(RealtimeSessionCloseCode),
}

/// Dials the approved ASR worker for the translated session, mirroring the
/// v1 relay's admission ladder: tenant lease, selection, dial, accept mark,
/// registry refresh.
async fn dial_asr_worker(
    state: &GatewayState,
    relay: &Arc<GatewayRealtimeRelay>,
    context: &RequestContext,
    session: &TranslateSession,
    sample_rate: u32,
    tenant_lease: &mut Option<BoundTenantWorkLease>,
) -> Result<WorkerSide, WorkerRejection> {
    let Some(stage) = asr_stage(relay) else {
        return Err(WorkerRejection::Close(
            RealtimeSessionCloseCode::PolicyDenied,
        ));
    };
    let Some(identity) = session.identity.as_ref() else {
        return Err(WorkerRejection::Close(
            RealtimeSessionCloseCode::ProtocolViolation,
        ));
    };

    let unbound = state
        .begin_tenant_work(context)
        .map_err(|error| WorkerRejection::Reject(error.status, error.message))?;

    let mut selected = match relay
        .registry()
        .select_and_reserve(&realtime_selection_request(relay.config(), &stage))
    {
        Ok(selected) => selected,
        Err(_) => {
            return Err(WorkerRejection::Reject(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "no fresh, ready realtime worker satisfies the request".to_string(),
            ));
        }
    };

    let admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: identity.session_id.clone(),
        request_id: identity.request_id.clone(),
        attempt_id: identity.attempt_id.clone(),
        expected_worker_incarnation: selected.key.incarnation_id.clone(),
        deployment_id: selected.deployment_id.clone(),
        expected_model_generation: selected.model_generation,
        caller: attested_caller(relay, context),
        task: stage.task,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: relay.config().session_budget.as_millis() as u64,
        input: RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate,
                channels: 1,
            },
            language: session.language.clone(),
        },
    };
    if admit.validate().is_err() {
        return Err(WorkerRejection::Close(
            RealtimeSessionCloseCode::ProtocolViolation,
        ));
    }

    let mut worker = match dial_worker_session(
        &selected.client.endpoint(),
        relay.worker_credentials(),
        &admit,
    )
    .await
    {
        Ok(worker) => worker,
        Err(_) => {
            return Err(WorkerRejection::Reject(
                axum::http::StatusCode::BAD_GATEWAY,
                "realtime worker session could not be established".to_string(),
            ));
        }
    };

    let attempt_identity = AttemptIdentity {
        request_id: admit.request_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        tenant_id: admit.caller.tenant_id.clone(),
        caller_id: admit.caller.caller_id.clone(),
        incarnation_id: admit.expected_worker_incarnation.clone(),
        deployment_id: admit.deployment_id.clone(),
        model_generation: admit.expected_model_generation,
    };
    *tenant_lease = Some(unbound.bind(selected.client.clone(), attempt_identity));
    if selected.dispatch.mark_accepted().is_err() {
        let _ = worker.sink.send(WireMessage::Close(None)).await;
        return Err(WorkerRejection::Close(
            RealtimeSessionCloseCode::CapacityExhausted,
        ));
    }

    // Refresh the registry entry with the resolved worker identity.
    relay.sessions().remove(&identity.session_id);
    let _ = relay.sessions().insert(
        identity.session_id.clone(),
        SessionRegistration {
            attempt_id: identity.attempt_id.clone(),
            worker_id: selected.key.worker_id.clone(),
            incarnation_id: selected.key.incarnation_id.as_str().to_string(),
            deployment_id: selected.deployment_id.clone(),
            model_generation: selected.model_generation,
            principal_namespace: principal_namespace(&context.principal),
            created_at: std::time::Instant::now(),
        },
        REGISTRATION_TTL,
    );

    Ok(worker)
}

async fn read_public_actions(
    mut client_source: futures::stream::SplitStream<WebSocket>,
    action_tx: mpsc::Sender<TranslateAction>,
) {
    while let Some(message) = client_source.next().await {
        let action = match message {
            Ok(Message::Text(text)) => {
                if text.len() > PUBLIC_CONTROL_FRAME_BYTES {
                    TranslateAction::InvalidText("control frame exceeds bound".to_string())
                } else {
                    match serde_json::from_str::<ClientEvent>(&text) {
                        Ok(ClientEvent::SessionStart {
                            model_id: _,
                            language,
                            protocol,
                            version,
                            resume_from_event_id,
                        }) => TranslateAction::Start {
                            language,
                            protocol,
                            version,
                            resume_from_event_id,
                        },
                        Ok(ClientEvent::SessionStop) => TranslateAction::Stop,
                        Ok(ClientEvent::Ping { timestamp_ms }) => {
                            TranslateAction::Ping { timestamp_ms }
                        }
                        Err(err) => TranslateAction::InvalidText(format!(
                            "Invalid realtime event payload: {err}"
                        )),
                    }
                }
            }
            Ok(Message::Binary(data)) => match parse_binary_message(&data) {
                Ok(BinaryMessageKind::ClientPcm16Frame {
                    frame_seq,
                    sample_rate,
                    payload,
                }) => TranslateAction::Audio {
                    frame_seq,
                    sample_rate,
                    payload,
                },
                Err(err) => TranslateAction::InvalidBinary(err),
            },
            Ok(Message::Close(_)) | Err(_) => {
                let _ = action_tx.send(TranslateAction::Gone).await;
                return;
            }
            Ok(Message::Ping(_) | Message::Pong(_)) => continue,
        };
        if action_tx.send(action).await.is_err() {
            return;
        }
    }
    let _ = action_tx.send(TranslateAction::Gone).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::realtime_protocol::RealtimeEventFinality;

    #[test]
    fn accumulator_holds_latest_delta_and_replaces_on_terminal() {
        let mut transcript = TranscriptAccumulator::new();
        assert_eq!(transcript.push("alpha ".to_string()), None);
        assert_eq!(transcript.push("beta ".to_string()), Some("alpha ".into()));
        assert_eq!(
            transcript.push("gamma".to_string()),
            Some("alpha beta ".into())
        );
        // The terminal resolves the held delta as the full final text,
        // replacing the accumulation rather than appending to it.
        assert_eq!(transcript.finish(), Some("gamma".into()));
        assert_eq!(transcript.finish(), None);
    }

    #[test]
    fn itrw_client_frame_reencodes_to_irta_worker_frame() {
        let payload = [0i16.to_le_bytes(), 1i16.to_le_bytes()].concat();
        let mut public_frame = Vec::new();
        public_frame.extend_from_slice(b"ITRW");
        public_frame.push(1);
        public_frame.push(1);
        public_frame.extend_from_slice(&0_u16.to_le_bytes());
        public_frame.extend_from_slice(&16_000_u32.to_le_bytes());
        public_frame.extend_from_slice(&7_u32.to_le_bytes());
        public_frame.extend_from_slice(&payload);

        let parsed = parse_binary_message(&public_frame).expect("public frame parses");
        let BinaryMessageKind::ClientPcm16Frame {
            frame_seq,
            sample_rate,
            payload: parsed_payload,
        } = parsed;
        assert_eq!(frame_seq, 7);
        assert_eq!(sample_rate, 16_000);
        assert_eq!(parsed_payload, payload);

        let worker_frame =
            encode_realtime_audio_frame(frame_seq, false, &parsed_payload).expect("re-encode");
        let (header, worker_payload) =
            izwi_serving_protocol::decode_realtime_audio_frame(&worker_frame).expect("irta");
        assert_eq!(header.sequence, frame_seq);
        assert!(!header.is_final);
        assert_eq!(worker_payload, payload.as_slice());
    }

    #[test]
    fn itrw_parser_rejects_malformed_frames() {
        assert!(parse_binary_message(&[0_u8; 8]).is_err(), "truncated");
        let mut bad_magic = vec![0_u8; 32];
        bad_magic[0] = b'X';
        assert!(parse_binary_message(&bad_magic).is_err(), "bad magic");
        let mut bad_version = vec![0_u8; 32];
        bad_version[..4].copy_from_slice(b"ITRW");
        bad_version[4] = 9;
        assert!(parse_binary_message(&bad_version).is_err(), "bad version");
        let mut bad_kind = vec![0_u8; 32];
        bad_kind[..4].copy_from_slice(b"ITRW");
        bad_kind[4] = 1;
        bad_kind[5] = 9;
        assert!(parse_binary_message(&bad_kind).is_err(), "bad kind");
    }

    #[test]
    fn sample_rate_lock_rejects_midstream_changes() {
        let mut lock = None;
        lock_sample_rate(&mut lock, 16_000).expect("first rate locks");
        assert_eq!(lock, Some(16_000));
        lock_sample_rate(&mut lock, 16_000).expect("same rate accepted");
        assert!(lock_sample_rate(&mut lock, 8_000)
            .expect_err("mid-stream change rejected")
            .contains("changed mid-stream"));
        assert!(lock_sample_rate(&mut None, 4_000)
            .expect_err("out-of-range rate rejected")
            .contains("Invalid input sample_rate"));
    }

    #[test]
    fn detect_audio_gap_reports_missing_span() {
        assert_eq!(detect_audio_gap(None, 1), None, "first frame has no gap");
        assert_eq!(detect_audio_gap(Some(1), 2), None, "in-order frame");
        assert_eq!(
            detect_audio_gap(Some(1), 4),
            Some((2, 4, 2)),
            "gap spans expected..received"
        );
        assert_eq!(detect_audio_gap(Some(4), 2), None, "stale frame");
    }

    #[test]
    fn typed_envelopes_carry_golden_shape_and_finality_order() {
        let mut typed = TypedCounters::new("session-1".to_string(), "gateway-42".to_string());

        let ready = typed.next_envelope(RealtimeServerEvent::SessionReady {
            accepted_version: TRANSCRIPTION_REALTIME_VERSION,
            owner_instance_id: typed.owner_instance_id.clone(),
            resumable: false,
            resume_window_ms: 0,
        });
        assert_eq!(ready.event_id, 1);
        assert_eq!(ready.sequence, 0);
        let mut ready = ready;
        ready.timestamp_ms = 1_725_000_000_123;
        assert_eq!(
            serde_json::to_string(&ready).expect("serialize typed SessionReady"),
            r#"{"protocol":"transcription_realtime","version":3,"event_id":1,"sequence":0,"session_id":"session-1","connection_epoch":0,"timestamp_ms":1725000000123,"type":"session_ready","data":{"accepted_version":3,"owner_instance_id":"gateway-42","resumable":false,"resume_window_ms":0}}"#
        );

        let started = typed.next_envelope(RealtimeServerEvent::SessionStarted);
        assert_eq!(started.event_id, 2);
        assert_eq!(started.sequence, 1);
        assert_eq!(started.validate_successor(&ready), Ok(()));

        let revision = typed.next_revision();
        let partial = typed.next_envelope(RealtimeServerEvent::TranscriptPartial {
            text: "hel".to_string(),
            revision,
            language: None,
        });
        assert_eq!(partial.validate_successor(&started), Ok(()));

        let revision = typed.next_revision();
        let final_event = typed.next_envelope(RealtimeServerEvent::TranscriptFinal {
            text: "hello".to_string(),
            revision,
            language: Some("en".to_string()),
        });
        assert_eq!(final_event.validate_successor(&partial), Ok(()));
        assert_eq!(
            final_event.event.finality(),
            RealtimeEventFinality::SegmentFinal
        );

        let close = normal_close("transcription session stopped by client");
        let closing = typed.next_envelope(RealtimeServerEvent::Closing {
            close: close.clone(),
        });
        assert_eq!(closing.validate_successor(&final_event), Ok(()));
        let closed = typed.next_envelope(RealtimeServerEvent::Closed { close });
        assert_eq!(closed.validate_successor(&closing), Ok(()));
        assert_eq!(closed.event.finality(), RealtimeEventFinality::SessionFinal);
        assert!(closed.event.requires_connection_close());
    }

    #[test]
    fn legacy_partial_json_shape_is_stable() {
        let mut sequence = 0;
        let partial = legacy_partial_json(
            &mut sequence,
            "hello".to_string(),
            Some("en".to_string()),
            false,
        );
        assert_eq!(partial["type"], "transcript_partial");
        assert_eq!(partial["sequence"], 1);
        assert_eq!(partial["text"], "hello");
        assert_eq!(partial["language"], "en");
        assert_eq!(partial["audio_duration_secs"], 0.0);
        assert_eq!(partial["processing_time_ms"], 0.0);
        assert!(partial["rtf"].is_null());
        assert!(partial.get("is_final").is_none());

        let final_partial = legacy_partial_json(&mut sequence, "hello".to_string(), None, true);
        assert_eq!(final_partial["sequence"], 2);
        assert_eq!(final_partial["is_final"], true);
    }

    #[test]
    fn negotiation_accepts_legacy_and_typed_v3_and_rejects_resume() {
        assert_eq!(
            negotiate_transcription_protocol(None, None, None),
            Ok(TranscriptionWireProtocol::LegacyV2)
        );
        assert_eq!(
            negotiate_transcription_protocol(Some("transcription_realtime"), Some(3), None),
            Ok(TranscriptionWireProtocol::TypedV3)
        );
        assert!(
            negotiate_transcription_protocol(Some("transcription_realtime"), Some(3), Some(9))
                .expect_err("v3 resume must be rejected")
                .contains("non-resumable")
        );
        assert!(
            negotiate_transcription_protocol(Some("other"), Some(1), None).is_err(),
            "unknown protocol rejected"
        );
    }

    #[test]
    fn worker_error_codes_map_onto_public_v3_vocabulary() {
        assert_eq!(
            map_worker_error_code(InvocationErrorCode::InvalidInput),
            RealtimeErrorCode::InvalidMessage
        );
        assert_eq!(
            map_worker_error_code(InvocationErrorCode::ExecutionFailed),
            RealtimeErrorCode::InferenceFailed
        );
        assert_eq!(
            map_worker_error_code(InvocationErrorCode::OutputLimitExceeded),
            RealtimeErrorCode::BufferLimit
        );
        assert_eq!(
            map_worker_error_code(InvocationErrorCode::WorkerUnavailable),
            RealtimeErrorCode::ModelUnavailable
        );
        assert_eq!(
            map_worker_error_code(InvocationErrorCode::Internal),
            RealtimeErrorCode::Internal
        );
    }
}

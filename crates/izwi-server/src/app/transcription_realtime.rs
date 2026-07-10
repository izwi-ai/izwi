//! Realtime transcription websocket endpoint for `/transcription`.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use izwi_core::{
    audio::{AudioEncoder, AudioFormat},
    RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream, RuntimeService, WorkloadClass,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::app::realtime_protocol::{
    RealtimeAudioGapAction, RealtimeClose, RealtimeCloseCode, RealtimeCloseReason,
    RealtimeErrorCode, RealtimeEventEnvelope, RealtimeProtocol, RealtimeServerEnvelope,
    RealtimeServerEvent, TRANSCRIPTION_REALTIME_VERSION,
};
use crate::state::AppState;

const LEGACY_REALTIME_PROTOCOL: &str = "transcription_realtime_v2";
const TYPED_REALTIME_PROTOCOL: &str = "transcription_realtime";

const WS_BIN_MAGIC: &[u8; 4] = b"ITRW";
const WS_BIN_VERSION: u8 = 1;
const WS_BIN_KIND_CLIENT_PCM16: u8 = 1;
const WS_BIN_CLIENT_HEADER_LEN: usize = 16;

const MAX_FRAME_BYTES: usize = 512 * 1024;
const MAX_STREAM_BUFFER_SECS: f32 = 32.0;
const INFERENCE_WINDOW_SECS: f32 = 14.0;
const INFERENCE_MIN_INTERVAL_MS: u64 = 350;
const MIN_INFERENCE_AUDIO_MS: u32 = 180;
const WS_OUTBOUND_QUEUE_CAPACITY: usize = 256;
const WORKER_COMMAND_QUEUE_CAPACITY: usize = 512;
const WS_WRITER_SEND_TIMEOUT: Duration = Duration::from_secs(5);
// LocalAgreement-2 style stabilization (as used in whisper_streaming):
// commit only the common prefix between the previous and current hypothesis.
// Text-only approximation of whisper_streaming's timestamp boundary filtering:
// allow a small leading drift while still anchoring overlap to committed tail.
const COMMITTED_OVERLAP_LOOKAHEAD_WORDS: usize = 6;
const COMMITTED_OVERLAP_MIN_WORDS: usize = 4;
const REPETITION_MIN_NGRAM_WORDS: usize = 3;
const REPETITION_MAX_NGRAM_WORDS: usize = 14;
const REPETITION_LOOKBACK_WORDS: usize = 48;
const REPETITION_MAX_GAP_WORDS: usize = 8;
const REPETITION_APPROX_MIN_WORDS: usize = 8;
const REPETITION_APPROX_MAX_WORDS: usize = 28;
const REPETITION_APPROX_MAX_GAP_WORDS: usize = 14;
const REPETITION_APPROX_MAX_LEN_DELTA: usize = 2;
const REPETITION_APPROX_MIN_LCS_RATIO: f32 = 0.84;
const REPETITION_APPROX_PREFIX_WINDOW_WORDS: usize = 3;
const REPETITION_APPROX_MIN_PREFIX_MATCH_WORDS: usize = 2;

#[derive(Clone)]
struct OutboundTx {
    tx: mpsc::Sender<Message>,
    runtime: Arc<RuntimeService>,
    label: &'static str,
}

impl OutboundTx {
    fn new(tx: mpsc::Sender<Message>, runtime: Arc<RuntimeService>, label: &'static str) -> Self {
        Self { tx, runtime, label }
    }

    fn send(&self, message: Message) -> bool {
        match self.tx.try_send(message) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.runtime.record_transcription_stream_backpressure();
                warn!(
                    "{} outbound websocket queue is full; dropping message",
                    self.label
                );
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        protocol: Option<String>,
        #[serde(default)]
        version: Option<u16>,
        #[serde(default)]
        resume_from_event_id: Option<u64>,
    },
    SessionStop,
    Ping {
        #[serde(default)]
        timestamp_ms: Option<u64>,
    },
}

#[derive(Debug)]
enum BinaryMessageKind {
    ClientPcm16Frame {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
}

#[derive(Debug)]
enum WorkerCommand {
    SessionStart {
        model_id: Option<String>,
        language: Option<String>,
        wire_protocol: TranscriptionWireProtocol,
    },
    AudioFrame {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
    Ping {
        timestamp_ms: Option<u64>,
    },
    ProtocolError(String),
    SessionStop,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionWireProtocol {
    LegacyV2,
    TypedV3,
}

struct TypedTranscriptionSession {
    session_id: String,
    owner_instance_id: String,
    next_event_id: u64,
    next_sequence: u64,
    next_revision: u64,
    last_stable_prefix_chars: usize,
    final_sent: bool,
}

impl TypedTranscriptionSession {
    fn process_owned() -> Self {
        Self::with_identity(
            uuid::Uuid::new_v4().to_string(),
            format!("process-{}", std::process::id()),
        )
    }

    fn with_identity(session_id: String, owner_instance_id: String) -> Self {
        Self {
            session_id,
            owner_instance_id,
            next_event_id: 1,
            next_sequence: 0,
            next_revision: 1,
            last_stable_prefix_chars: 0,
            final_sent: false,
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

    fn send(&mut self, out_tx: &OutboundTx, event: RealtimeServerEvent) -> bool {
        let envelope = self.next_envelope(event);
        match serde_json::to_string(&envelope) {
            Ok(text) => out_tx.send(Message::Text(text.into())),
            Err(err) => {
                warn!("failed to serialize typed transcription event: {err}");
                false
            }
        }
    }
}

struct PendingInference {
    sequence: u64,
    started_at: Instant,
    sample_rate: u32,
    sample_count: usize,
    receiver: oneshot::Receiver<Result<InferenceResult, String>>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RealtimeInferenceDiagnostics {
    route: &'static str,
    sequence: u64,
    sample_rate: u32,
    input_sample_count: usize,
    audio_duration_secs: f32,
    queue_wait_ms: f64,
    processing_time_ms: f64,
    rtf: Option<f64>,
    output_text_chars: usize,
    cancellation_reason: Option<&'static str>,
}

struct RealtimeSessionState {
    started: bool,
    model_id: Option<String>,
    language: Option<String>,
    sample_rate: Option<u32>,
    last_frame_seq: Option<u32>,
    samples_i16: Vec<i16>,
    pending_recompute: bool,
    in_flight: Option<PendingInference>,
    last_inference_started_at: Option<Instant>,
    inference_sequence: u64,
    last_emitted_sequence: u64,
    committed_text: String,
    trailing_text: String,
    native_stream_checked: bool,
    native_asr_stream: Option<RuntimeAsrRealtimeStream>,
    wire_protocol: TranscriptionWireProtocol,
    typed_session: Option<TypedTranscriptionSession>,
}

impl Default for RealtimeSessionState {
    fn default() -> Self {
        Self {
            started: false,
            model_id: None,
            language: None,
            sample_rate: None,
            last_frame_seq: None,
            samples_i16: Vec::new(),
            pending_recompute: false,
            in_flight: None,
            last_inference_started_at: None,
            inference_sequence: 0,
            last_emitted_sequence: 0,
            committed_text: String::new(),
            trailing_text: String::new(),
            native_stream_checked: false,
            native_asr_stream: None,
            wire_protocol: TranscriptionWireProtocol::LegacyV2,
            typed_session: None,
        }
    }
}

struct InferenceResult {
    text: String,
    language: Option<String>,
    duration_secs: f32,
    queue_wait_ms: f64,
}

impl RealtimeInferenceDiagnostics {
    fn success(
        pending: &PendingInference,
        output: &InferenceResult,
        processing_time_ms: f64,
        rtf: Option<f64>,
    ) -> Self {
        Self {
            route: "transcription.realtime",
            sequence: pending.sequence,
            sample_rate: pending.sample_rate,
            input_sample_count: pending.sample_count,
            audio_duration_secs: output.duration_secs,
            queue_wait_ms: output.queue_wait_ms,
            processing_time_ms,
            rtf,
            output_text_chars: output.text.chars().count(),
            cancellation_reason: None,
        }
    }

    fn failure(
        pending: &PendingInference,
        processing_time_ms: f64,
        cancellation_reason: &'static str,
    ) -> Self {
        let audio_duration_secs = if pending.sample_rate == 0 {
            0.0
        } else {
            pending.sample_count as f32 / pending.sample_rate as f32
        };

        Self {
            route: "transcription.realtime",
            sequence: pending.sequence,
            sample_rate: pending.sample_rate,
            input_sample_count: pending.sample_count,
            audio_duration_secs,
            queue_wait_ms: 0.0,
            processing_time_ms,
            rtf: None,
            output_text_chars: 0,
            cancellation_reason: Some(cancellation_reason),
        }
    }

    fn emit(&self) {
        info!(
            target: "izwi.audio",
            route = self.route,
            sequence = self.sequence,
            sample_rate = self.sample_rate,
            input_sample_count = self.input_sample_count,
            audio_duration_secs = self.audio_duration_secs,
            queue_wait_ms = self.queue_wait_ms,
            processing_time_ms = self.processing_time_ms,
            rtf = self.rtf.unwrap_or(0.0),
            has_rtf = self.rtf.is_some(),
            output_text_chars = self.output_text_chars,
            cancellation_reason = self.cancellation_reason.unwrap_or(""),
            "realtime transcription inference diagnostics"
        );
    }
}

pub async fn handle_socket(socket: WebSocket, state: AppState, correlation_id: String) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (raw_out_tx, mut out_rx) = mpsc::channel::<Message>(WS_OUTBOUND_QUEUE_CAPACITY);
    let out_tx = OutboundTx::new(raw_out_tx, state.runtime.clone(), "transcription realtime");
    let (worker_tx, worker_rx) = mpsc::channel::<WorkerCommand>(WORKER_COMMAND_QUEUE_CAPACITY);

    let writer = tokio::spawn(async move {
        while let Some(message) = out_rx.recv().await {
            if !matches!(
                tokio::time::timeout(WS_WRITER_SEND_TIMEOUT, ws_tx.send(message)).await,
                Ok(Ok(()))
            ) {
                break;
            }
        }
    });

    let worker = tokio::spawn(run_worker(
        state.clone(),
        correlation_id.clone(),
        out_tx.clone(),
        worker_rx,
    ));

    send_json(
        &out_tx,
        json!({
            "type": "session_ready",
            "protocol": LEGACY_REALTIME_PROTOCOL,
            "correlation_id": correlation_id,
        }),
    );

    while let Some(result) = ws_rx.next().await {
        let message = match result {
            Ok(message) => message,
            Err(err) => {
                warn!("transcription realtime websocket receive error: {err}");
                break;
            }
        };

        match message {
            Message::Text(text) => {
                if handle_text_message(&out_tx, &worker_tx, text.as_str()) {
                    break;
                }
            }
            Message::Binary(data) => match parse_binary_message(&data) {
                Ok(BinaryMessageKind::ClientPcm16Frame {
                    frame_seq,
                    sample_rate,
                    payload,
                }) => {
                    if !send_worker_command(
                        &worker_tx,
                        WorkerCommand::AudioFrame {
                            frame_seq,
                            sample_rate,
                            payload,
                        },
                    ) {
                        break;
                    }
                }
                Err(err) => {
                    let _ = send_worker_command(&worker_tx, WorkerCommand::ProtocolError(err));
                }
            },
            Message::Ping(payload) => {
                let _ = out_tx.send(Message::Pong(payload));
            }
            Message::Close(_) => break,
            Message::Pong(_) => {}
        }
    }

    let _ = send_worker_command(&worker_tx, WorkerCommand::Shutdown);
    let _ = worker.await;

    drop(out_tx);
    let _ = writer.await;
}

fn handle_text_message(
    out_tx: &OutboundTx,
    worker_tx: &mpsc::Sender<WorkerCommand>,
    text: &str,
) -> bool {
    let event: ClientEvent = match serde_json::from_str(text) {
        Ok(event) => event,
        Err(err) => {
            send_json(
                out_tx,
                json!({
                    "type": "error",
                    "message": format!("Invalid realtime event payload: {err}"),
                }),
            );
            return false;
        }
    };

    match event {
        ClientEvent::SessionStart {
            model_id,
            language,
            protocol,
            version,
            resume_from_event_id,
        } => {
            let wire_protocol = match negotiate_transcription_protocol(
                protocol.as_deref(),
                version,
                resume_from_event_id,
            ) {
                Ok(protocol) => protocol,
                Err(err) => {
                    send_json(out_tx, json!({ "type": "error", "message": err }));
                    return false;
                }
            };
            if !send_worker_command(
                worker_tx,
                WorkerCommand::SessionStart {
                    model_id,
                    language,
                    wire_protocol,
                },
            ) {
                return true;
            }
            false
        }
        ClientEvent::SessionStop => {
            let _ = send_worker_command(worker_tx, WorkerCommand::SessionStop);
            true
        }
        ClientEvent::Ping { timestamp_ms } => {
            let _ = send_worker_command(worker_tx, WorkerCommand::Ping { timestamp_ms });
            false
        }
    }
}

fn negotiate_transcription_protocol(
    protocol: Option<&str>,
    version: Option<u16>,
    resume_from_event_id: Option<u64>,
) -> Result<TranscriptionWireProtocol, String> {
    match (protocol.map(str::trim), version) {
        (None, None) => Ok(TranscriptionWireProtocol::LegacyV2),
        (Some(TYPED_REALTIME_PROTOCOL), Some(TRANSCRIPTION_REALTIME_VERSION)) => {
            if resume_from_event_id.is_some() {
                return Err(
                    "transcription_realtime v3 sessions are process-owned and non-resumable"
                        .to_string(),
                );
            }
            Ok(TranscriptionWireProtocol::TypedV3)
        }
        (Some(protocol), Some(version)) => Err(format!(
            "Unsupported realtime protocol negotiation `{protocol}` version {version}"
        )),
        _ => {
            Err("Realtime protocol negotiation requires both `protocol` and `version`".to_string())
        }
    }
}

async fn run_worker(
    state: AppState,
    correlation_id: String,
    out_tx: OutboundTx,
    mut worker_rx: mpsc::Receiver<WorkerCommand>,
) {
    let mut session = RealtimeSessionState::default();
    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        enum LoopEvent {
            Command(Option<WorkerCommand>),
            Tick,
            InferenceDone(Result<Result<InferenceResult, String>, oneshot::error::RecvError>),
        }

        let event = if let Some(in_flight) = session.in_flight.as_mut() {
            tokio::select! {
                command = worker_rx.recv() => LoopEvent::Command(command),
                _ = ticker.tick() => LoopEvent::Tick,
                result = &mut in_flight.receiver => LoopEvent::InferenceDone(result),
            }
        } else {
            tokio::select! {
                command = worker_rx.recv() => LoopEvent::Command(command),
                _ = ticker.tick() => LoopEvent::Tick,
            }
        };

        match event {
            LoopEvent::Command(Some(command)) => {
                let ingress_queue_depth = worker_rx.len();
                if handle_worker_command(
                    &state,
                    &correlation_id,
                    &out_tx,
                    &mut session,
                    command,
                    ingress_queue_depth,
                )
                .await
                {
                    break;
                }
            }
            LoopEvent::Command(None) => break,
            LoopEvent::Tick => {
                if let Err(err) =
                    maybe_schedule_inference(&state, &correlation_id, &mut session, false)
                {
                    send_session_error_with_code(
                        &out_tx,
                        &mut session,
                        RealtimeErrorCode::InferenceFailed,
                        err,
                    );
                }
            }
            LoopEvent::InferenceDone(result) => {
                let Some(pending) = session.in_flight.take() else {
                    continue;
                };
                handle_inference_result(&out_tx, &mut session, pending, result);
                if let Err(err) =
                    maybe_schedule_inference(&state, &correlation_id, &mut session, true)
                {
                    send_session_error_with_code(
                        &out_tx,
                        &mut session,
                        RealtimeErrorCode::InferenceFailed,
                        err,
                    );
                }
            }
        }
    }

    if let Some(in_flight) = session.in_flight.take() {
        in_flight.task.abort();
    }
}

async fn handle_worker_command(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    command: WorkerCommand,
    ingress_queue_depth: usize,
) -> bool {
    match command {
        WorkerCommand::SessionStart {
            model_id,
            language,
            wire_protocol,
        } => {
            if let Some(in_flight) = session.in_flight.take() {
                in_flight.task.abort();
            }

            session.started = true;
            session.model_id = model_id.filter(|value| !value.trim().is_empty());
            session.language = language.filter(|value| !value.trim().is_empty());
            session.sample_rate = None;
            session.last_frame_seq = None;
            session.samples_i16.clear();
            session.pending_recompute = false;
            session.last_inference_started_at = None;
            session.inference_sequence = 0;
            session.last_emitted_sequence = 0;
            session.committed_text.clear();
            session.trailing_text.clear();
            session.native_stream_checked = false;
            session.native_asr_stream = None;
            session.wire_protocol = wire_protocol;
            session.typed_session = match wire_protocol {
                TranscriptionWireProtocol::LegacyV2 => None,
                TranscriptionWireProtocol::TypedV3 => {
                    Some(TypedTranscriptionSession::process_owned())
                }
            };

            send_session_started(out_tx, session);
            false
        }
        WorkerCommand::AudioFrame {
            frame_seq,
            sample_rate,
            payload,
        } => {
            let previous_frame_seq = session.last_frame_seq;
            let frame_samples = match ingest_audio_frame(session, frame_seq, sample_rate, &payload)
            {
                Ok(samples) => samples,
                Err(err) => {
                    send_session_error(out_tx, session, err);
                    return false;
                }
            };

            if !frame_samples.is_empty() {
                send_audio_ingress_events(
                    out_tx,
                    session,
                    previous_frame_seq,
                    frame_seq,
                    ingress_queue_depth,
                );
            }

            match maybe_process_native_stream_frame(
                state,
                out_tx,
                session,
                &frame_samples,
                sample_rate,
            )
            .await
            {
                Ok(true) => {}
                Ok(false) => {
                    if let Err(err) =
                        maybe_schedule_inference(state, correlation_id, session, false)
                    {
                        send_session_error_with_code(
                            out_tx,
                            session,
                            RealtimeErrorCode::InferenceFailed,
                            err,
                        );
                    }
                }
                Err(err) => send_session_error_with_code(
                    out_tx,
                    session,
                    RealtimeErrorCode::InferenceFailed,
                    err,
                ),
            }
            false
        }
        WorkerCommand::Ping { timestamp_ms } => {
            send_session_pong(out_tx, session, timestamp_ms);
            false
        }
        WorkerCommand::ProtocolError(err) => {
            send_session_error(out_tx, session, err);
            false
        }
        WorkerCommand::SessionStop => {
            let had_native_stream = session.native_asr_stream.is_some();
            let finish_result = if had_native_stream {
                finish_native_stream_if_needed(state, out_tx, session).await
            } else {
                finish_fallback_stream(state, correlation_id, out_tx, session).await
            };
            if let Err(err) = finish_result {
                send_session_error_with_code(
                    out_tx,
                    session,
                    RealtimeErrorCode::InferenceFailed,
                    err,
                );
            }
            send_session_finished(out_tx, session);
            true
        }
        WorkerCommand::Shutdown => true,
    }
}

fn send_session_started(out_tx: &OutboundTx, session: &mut RealtimeSessionState) {
    match session.wire_protocol {
        TranscriptionWireProtocol::LegacyV2 => {
            send_json(out_tx, json!({ "type": "session_started" }));
        }
        TranscriptionWireProtocol::TypedV3 => {
            let Some(typed) = session.typed_session.as_mut() else {
                return;
            };
            let owner_instance_id = typed.owner_instance_id.clone();
            typed.send(
                out_tx,
                RealtimeServerEvent::SessionReady {
                    accepted_version: TRANSCRIPTION_REALTIME_VERSION,
                    owner_instance_id,
                    resumable: false,
                    resume_window_ms: 0,
                },
            );
            typed.send(out_tx, RealtimeServerEvent::SessionStarted);
        }
    }
}

fn send_session_error(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    message: impl Into<String>,
) {
    send_session_error_with_code(out_tx, session, RealtimeErrorCode::InvalidMessage, message);
}

fn send_session_error_with_code(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    code: RealtimeErrorCode,
    message: impl Into<String>,
) {
    let message = message.into();
    match session.wire_protocol {
        TranscriptionWireProtocol::LegacyV2 => {
            send_json(out_tx, json!({ "type": "error", "message": message }));
        }
        TranscriptionWireProtocol::TypedV3 => {
            if let Some(typed) = session.typed_session.as_mut() {
                typed.send(
                    out_tx,
                    RealtimeServerEvent::RecoverableError {
                        code,
                        message,
                        retry_after_ms: None,
                    },
                );
            }
        }
    }
}

fn send_session_pong(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    timestamp_ms: Option<u64>,
) {
    match session.wire_protocol {
        TranscriptionWireProtocol::LegacyV2 => {
            send_json(
                out_tx,
                json!({ "type": "pong", "timestamp_ms": timestamp_ms }),
            );
        }
        TranscriptionWireProtocol::TypedV3 => {
            if let Some(typed) = session.typed_session.as_mut() {
                typed.send(
                    out_tx,
                    RealtimeServerEvent::Pong {
                        client_timestamp_ms: timestamp_ms,
                        server_timestamp_ms: now_unix_millis(),
                    },
                );
            }
        }
    }
}

fn send_audio_ingress_events(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    previous_frame_seq: Option<u32>,
    frame_seq: u32,
    ingress_queue_depth: usize,
) {
    if session.wire_protocol != TranscriptionWireProtocol::TypedV3 {
        return;
    }
    let Some(typed) = session.typed_session.as_mut() else {
        return;
    };

    if let Some(previous) = previous_frame_seq {
        let expected = previous.saturating_add(1);
        if frame_seq > expected {
            typed.send(
                out_tx,
                RealtimeServerEvent::AudioGap {
                    expected_frame_sequence: expected as u64,
                    received_frame_sequence: frame_seq as u64,
                    missing_frames: frame_seq.saturating_sub(expected) as u64,
                    action: RealtimeAudioGapAction::Continue,
                },
            );
        }
    }

    typed.send(
        out_tx,
        RealtimeServerEvent::AudioAccepted {
            frame_sequence: frame_seq as u64,
            buffer_depth_samples: session.samples_i16.len(),
            ingress_queue_depth,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn send_transcript_partial(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    sequence: u64,
    text: String,
    language: Option<String>,
    audio_duration_secs: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
    stable_prefix_chars: usize,
) {
    match session.wire_protocol {
        TranscriptionWireProtocol::LegacyV2 => send_json(
            out_tx,
            json!({
                "type": "transcript_partial",
                "sequence": sequence,
                "text": text,
                "language": language,
                "audio_duration_secs": audio_duration_secs,
                "processing_time_ms": processing_time_ms,
                "rtf": rtf,
            }),
        ),
        TranscriptionWireProtocol::TypedV3 => {
            let Some(typed) = session.typed_session.as_mut() else {
                return;
            };
            let revision = typed.next_revision();
            typed.send(
                out_tx,
                RealtimeServerEvent::TranscriptPartial {
                    text: text.clone(),
                    revision,
                    language,
                },
            );

            if stable_prefix_chars > typed.last_stable_prefix_chars {
                typed.last_stable_prefix_chars = stable_prefix_chars;
                let revision = typed.next_revision();
                typed.send(
                    out_tx,
                    RealtimeServerEvent::TranscriptStable {
                        text,
                        revision,
                        stable_prefix_chars,
                    },
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn send_native_transcript(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    sequence: u64,
    text: String,
    language: Option<String>,
    audio_duration_secs: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
    is_final: bool,
) {
    match session.wire_protocol {
        TranscriptionWireProtocol::LegacyV2 => send_json(
            out_tx,
            json!({
                "type": "transcript_partial",
                "sequence": sequence,
                "text": text,
                "language": language,
                "audio_duration_secs": audio_duration_secs,
                "processing_time_ms": processing_time_ms,
                "rtf": rtf,
                "native_stream": true,
                "is_final": is_final,
            }),
        ),
        TranscriptionWireProtocol::TypedV3 => {
            let Some(typed) = session.typed_session.as_mut() else {
                return;
            };
            if is_final {
                if typed.final_sent {
                    return;
                }
                typed.final_sent = true;
                let revision = typed.next_revision();
                typed.send(
                    out_tx,
                    RealtimeServerEvent::TranscriptFinal {
                        text,
                        revision,
                        language,
                    },
                );
            } else {
                let stable_prefix_chars = text.chars().count();
                typed.last_stable_prefix_chars = stable_prefix_chars;
                let revision = typed.next_revision();
                typed.send(
                    out_tx,
                    RealtimeServerEvent::TranscriptStable {
                        text,
                        revision,
                        stable_prefix_chars,
                    },
                );
            }
        }
    }
}

fn send_session_finished(out_tx: &OutboundTx, session: &mut RealtimeSessionState) {
    if session.wire_protocol == TranscriptionWireProtocol::LegacyV2 {
        send_json(out_tx, json!({ "type": "session_done" }));
        return;
    }

    let final_text = concat_transcript(&session.committed_text, &session.trailing_text);
    let language = session.language.clone();
    let Some(typed) = session.typed_session.as_mut() else {
        return;
    };
    if !typed.final_sent {
        typed.final_sent = true;
        let revision = typed.next_revision();
        typed.send(
            out_tx,
            RealtimeServerEvent::TranscriptFinal {
                text: final_text,
                revision,
                language,
            },
        );
    }

    let close = RealtimeClose {
        code: RealtimeCloseCode::Normal,
        reason: RealtimeCloseReason::ClientRequest,
        message: "transcription session stopped by client".to_string(),
        retryable: false,
    };
    typed.send(
        out_tx,
        RealtimeServerEvent::Closing {
            close: close.clone(),
        },
    );
    typed.send(out_tx, RealtimeServerEvent::Closed { close });
}

fn ingest_audio_frame(
    session: &mut RealtimeSessionState,
    frame_seq: u32,
    sample_rate: u32,
    payload: &[u8],
) -> Result<Vec<i16>, String> {
    if !session.started {
        return Err("session_start is required before streaming audio".to_string());
    }
    if payload.is_empty() {
        return Ok(Vec::new());
    }
    if payload.len() > MAX_FRAME_BYTES {
        return Err(format!(
            "Audio frame exceeded max size ({} > {})",
            payload.len(),
            MAX_FRAME_BYTES
        ));
    }
    if payload.len() % 2 != 0 {
        return Err("PCM16 payload length must be even".to_string());
    }
    if sample_rate < 8_000 || sample_rate > 192_000 {
        return Err(format!("Invalid input sample_rate {sample_rate}"));
    }

    if let Some(last) = session.last_frame_seq {
        if frame_seq <= last {
            debug!(
                "transcription realtime stale frame ignored: frame_seq={} last_frame_seq={}",
                frame_seq, last
            );
            return Ok(Vec::new());
        }
    }
    session.last_frame_seq = Some(frame_seq);

    if let Some(current_sr) = session.sample_rate {
        if current_sr != sample_rate {
            return Err(format!(
                "Input sample rate changed mid-stream ({current_sr} -> {sample_rate})"
            ));
        }
    } else {
        session.sample_rate = Some(sample_rate);
    }

    let samples = pcm16_bytes_to_i16(payload);
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    session.samples_i16.extend_from_slice(&samples);
    session.pending_recompute = true;

    let max_samples = ((sample_rate as f32) * MAX_STREAM_BUFFER_SECS) as usize;
    if session.samples_i16.len() > max_samples {
        let drain = session.samples_i16.len() - max_samples;
        session.samples_i16.drain(0..drain);
    }

    Ok(samples)
}

async fn maybe_process_native_stream_frame(
    state: &AppState,
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    frame_samples: &[i16],
    sample_rate: u32,
) -> Result<bool, String> {
    if frame_samples.is_empty() {
        return Ok(session.native_asr_stream.is_some());
    }

    if !session.native_stream_checked {
        session.native_stream_checked = true;
        let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
        session.native_asr_stream = state
            .runtime
            .try_start_asr_realtime_stream(
                session.model_id.as_deref(),
                session.language.as_deref(),
                None,
            )
            .await
            .map_err(|err| err.to_string())?;
    }

    let Some(stream) = session.native_asr_stream.as_mut() else {
        return Ok(false);
    };

    let samples = pcm16_i16_to_f32(frame_samples);
    let started = Instant::now();
    let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    let events = state
        .runtime
        .push_asr_realtime_samples(stream, &samples, sample_rate)
        .map_err(|err| err.to_string())?;
    let processing_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    emit_native_stream_events(out_tx, session, events, sample_rate, processing_time_ms);
    session.pending_recompute = false;
    Ok(true)
}

async fn finish_native_stream_if_needed(
    state: &AppState,
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
) -> Result<(), String> {
    let Some(mut stream) = session.native_asr_stream.take() else {
        return Ok(());
    };

    let sample_rate = session.sample_rate.unwrap_or(16_000);
    let started = Instant::now();
    let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    let events = state
        .runtime
        .finish_asr_realtime_stream(&mut stream)
        .map_err(|err| err.to_string())?;
    let processing_time_ms = started.elapsed().as_secs_f64() * 1000.0;
    emit_native_stream_events(out_tx, session, events, sample_rate, processing_time_ms);
    Ok(())
}

async fn finish_fallback_stream(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
) -> Result<(), String> {
    loop {
        if session.in_flight.is_none() {
            maybe_schedule_inference(state, correlation_id, session, true)?;
        }
        let Some(mut pending) = session.in_flight.take() else {
            return Ok(());
        };

        let result = match tokio::time::timeout(
            Duration::from_secs(state.request_timeout_secs.max(1)),
            &mut pending.receiver,
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                pending.task.abort();
                return Err("Final realtime transcription timed out while draining".to_string());
            }
        };
        handle_inference_result(out_tx, session, pending, result);
        if !session.pending_recompute {
            return Ok(());
        }
    }
}

fn emit_native_stream_events(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    events: Vec<RuntimeAsrRealtimeEvent>,
    sample_rate: u32,
    processing_time_ms: f64,
) {
    let audio_duration_secs = if sample_rate > 0 {
        session.samples_i16.len() as f32 / sample_rate as f32
    } else {
        0.0
    };
    let rtf = (audio_duration_secs > 0.0)
        .then_some((processing_time_ms / 1000.0) / audio_duration_secs as f64);

    for event in events {
        if event.delta.is_empty() && !event.is_final {
            continue;
        }
        session.committed_text = event.text.clone();
        session.trailing_text.clear();
        session.last_emitted_sequence = event.chunk_index as u64;
        let language = session.language.clone();
        send_native_transcript(
            out_tx,
            session,
            event.chunk_index as u64,
            event.text,
            language,
            audio_duration_secs,
            processing_time_ms,
            rtf,
            event.is_final,
        );
    }
}

fn maybe_schedule_inference(
    state: &AppState,
    correlation_id: &str,
    session: &mut RealtimeSessionState,
    force_interval_check: bool,
) -> Result<(), String> {
    if session.native_asr_stream.is_some() {
        return Ok(());
    }
    if !session.started || session.in_flight.is_some() || !session.pending_recompute {
        return Ok(());
    }

    let sample_rate = match session.sample_rate {
        Some(sample_rate) => sample_rate,
        None => return Ok(()),
    };

    let min_samples = ((sample_rate as u64) * (MIN_INFERENCE_AUDIO_MS as u64) / 1000) as usize;
    if session.samples_i16.len() < min_samples {
        return Ok(());
    }

    if !force_interval_check {
        if let Some(last_started_at) = session.last_inference_started_at {
            if last_started_at.elapsed() < Duration::from_millis(INFERENCE_MIN_INTERVAL_MS) {
                return Ok(());
            }
        }
    }

    let window_samples = ((sample_rate as f32) * INFERENCE_WINDOW_SECS) as usize;
    let keep_samples = window_samples.max(min_samples);
    let start = session.samples_i16.len().saturating_sub(keep_samples);
    let inference_samples = session.samples_i16[start..].to_vec();
    let inference_sample_count = inference_samples.len();

    let sequence = session.inference_sequence.saturating_add(1);
    session.inference_sequence = sequence;
    session.pending_recompute = false;

    let started_at = Instant::now();
    session.last_inference_started_at = Some(started_at);

    let model_id = session.model_id.clone();
    let language = session.language.clone();
    let correlation_id = correlation_id.to_string();
    let state = state.clone();

    let (tx, rx) = oneshot::channel::<Result<InferenceResult, String>>();
    let task = tokio::spawn(async move {
        let result = run_inference(
            state,
            correlation_id,
            inference_samples,
            sample_rate,
            model_id,
            language,
        )
        .await;
        let _ = tx.send(result);
    });

    session.in_flight = Some(PendingInference {
        sequence,
        started_at,
        sample_rate,
        sample_count: inference_sample_count,
        receiver: rx,
        task,
    });

    Ok(())
}

fn handle_inference_result(
    out_tx: &OutboundTx,
    session: &mut RealtimeSessionState,
    pending: PendingInference,
    result: Result<Result<InferenceResult, String>, oneshot::error::RecvError>,
) {
    let sequence = pending.sequence;
    if sequence < session.last_emitted_sequence {
        return;
    }

    match result {
        Ok(Ok(output)) => {
            let merged_text = merge_online_transcript(
                &mut session.committed_text,
                &mut session.trailing_text,
                output.text.as_str(),
            );

            let processing_time_ms = pending.started_at.elapsed().as_secs_f64() * 1000.0;
            let rtf = if output.duration_secs > 0.0 {
                Some((processing_time_ms / 1000.0) / output.duration_secs as f64)
            } else {
                None
            };
            RealtimeInferenceDiagnostics::success(&pending, &output, processing_time_ms, rtf)
                .emit();

            let stable_prefix_chars = session.committed_text.chars().count();
            send_transcript_partial(
                out_tx,
                session,
                sequence,
                merged_text,
                output.language,
                output.duration_secs,
                processing_time_ms,
                rtf,
                stable_prefix_chars,
            );
            session.last_emitted_sequence = sequence;
        }
        Ok(Err(err)) => {
            RealtimeInferenceDiagnostics::failure(
                &pending,
                pending.started_at.elapsed().as_secs_f64() * 1000.0,
                "asr_failed",
            )
            .emit();
            send_session_error_with_code(
                out_tx,
                session,
                RealtimeErrorCode::InferenceFailed,
                format!("ASR failed: {err}"),
            );
        }
        Err(err) => {
            RealtimeInferenceDiagnostics::failure(
                &pending,
                pending.started_at.elapsed().as_secs_f64() * 1000.0,
                "task_failed",
            )
            .emit();
            send_session_error_with_code(
                out_tx,
                session,
                RealtimeErrorCode::InferenceFailed,
                format!("Realtime inference task failed: {err}"),
            );
        }
    }
}

async fn run_inference(
    state: AppState,
    correlation_id: String,
    samples_i16: Vec<i16>,
    sample_rate: u32,
    model_id: Option<String>,
    language: Option<String>,
) -> Result<InferenceResult, String> {
    let wav_bytes = wav_bytes_from_pcm16_mono(&samples_i16, sample_rate)?;

    let permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    let queue_wait_ms = permit.wait_ms();
    let output = state
        .runtime
        .asr_transcribe_bytes_with_runtime_context(
            wav_bytes.as_slice(),
            model_id.as_deref(),
            language.as_deref(),
            None,
            None,
            Some(correlation_id.as_str()),
            permit.runtime_context(),
        )
        .await
        .map_err(|err| err.to_string())?;

    debug!(
        "transcription realtime inference complete: correlation_id={} text_len={} duration_secs={}",
        correlation_id,
        output.text.len(),
        output.duration_secs
    );

    Ok(InferenceResult {
        text: output.text,
        language: output.language,
        duration_secs: output.duration_secs,
        queue_wait_ms,
    })
}

fn merge_online_transcript(
    committed: &mut String,
    trailing: &mut String,
    candidate: &str,
) -> String {
    let candidate = candidate.trim_start();
    if candidate.is_empty() {
        return concat_transcript(committed, trailing);
    }

    let mut candidate = collapse_unstable_repetition(candidate)
        .trim_start()
        .to_string();
    candidate =
        strip_suffix_prefix_overlap_with_lookahead_by_words(committed.as_str(), candidate.as_str());
    if candidate.is_empty() {
        trailing.clear();
        return concat_transcript(committed, trailing);
    }

    let commit_words = common_prefix_word_count(trailing.as_str(), candidate.as_str());
    if commit_words > 0 {
        let commit_bytes = byte_after_n_words(candidate.as_str(), commit_words);
        if commit_bytes > 0 {
            append_text(committed, &candidate[..commit_bytes]);
            candidate = drop_n_words(candidate.as_str(), commit_words).to_string();
        }
    }

    trailing.clear();
    trailing.push_str(candidate.as_str());

    concat_transcript(committed, trailing)
}

fn common_prefix_word_count(a: &str, b: &str) -> usize {
    let words_a = collect_word_spans(a);
    let words_b = collect_word_spans(b);
    let max = words_a.len().min(words_b.len());
    if max == 0 {
        return 0;
    }

    let mut shared = 0usize;
    while shared < max && words_a[shared].normalized == words_b[shared].normalized {
        shared += 1;
    }
    shared
}

fn strip_suffix_prefix_overlap_with_lookahead_by_words(reference: &str, candidate: &str) -> String {
    let candidate_words = collect_word_spans(candidate);
    if candidate_words.is_empty() {
        return String::new();
    }

    let max_shift = COMMITTED_OVERLAP_LOOKAHEAD_WORDS.min(candidate_words.len().saturating_sub(1));
    let mut best_match: Option<(usize, usize)> = None;

    for shift in 0..=max_shift {
        let start = candidate_words[shift].start;
        let shifted_candidate = candidate.get(start..).unwrap_or("");
        let overlap_words = longest_suffix_prefix_word_count(reference, shifted_candidate);
        if overlap_words == 0 {
            continue;
        }
        match best_match {
            None => best_match = Some((shift, overlap_words)),
            Some((best_shift, best_overlap)) => {
                if overlap_words > best_overlap
                    || (overlap_words == best_overlap && shift < best_shift)
                {
                    best_match = Some((shift, overlap_words));
                }
            }
        }
    }

    let Some((shift, overlap_words)) = best_match else {
        return candidate.trim_start().to_string();
    };

    let should_strip = if shift == 0 {
        true
    } else {
        overlap_words >= COMMITTED_OVERLAP_MIN_WORDS
    };
    if !should_strip {
        return candidate.trim_start().to_string();
    }

    let cut_words = shift.saturating_add(overlap_words);
    if cut_words == 0 {
        return candidate.trim_start().to_string();
    }
    let cut_idx = cut_words
        .saturating_sub(1)
        .min(candidate_words.len().saturating_sub(1));
    let cut_byte = candidate_words[cut_idx].end;
    candidate
        .get(cut_byte..)
        .unwrap_or("")
        .trim_start()
        .to_string()
}

fn concat_transcript(committed: &str, trailing: &str) -> String {
    if committed.is_empty() {
        return trailing.to_string();
    }
    if trailing.is_empty() {
        return committed.to_string();
    }
    let left_last = committed.chars().last().unwrap_or(' ');
    let right_first = trailing.chars().next().unwrap_or(' ');
    if is_word_char(left_last) && is_word_char(right_first) {
        format!("{committed} {trailing}")
    } else {
        format!("{committed}{trailing}")
    }
}

fn append_text(base: &mut String, segment: &str) {
    if segment.is_empty() {
        return;
    }
    if base.is_empty() {
        base.push_str(segment);
        return;
    }
    let left_last = base.chars().last().unwrap_or(' ');
    let right_first = segment.chars().next().unwrap_or(' ');
    if is_word_char(left_last) && is_word_char(right_first) {
        base.push(' ');
    }
    base.push_str(segment);
}

fn longest_suffix_prefix_word_count(a: &str, b: &str) -> usize {
    let words_a = collect_word_spans(a);
    let words_b = collect_word_spans(b);
    let max = words_a.len().min(words_b.len());
    if max == 0 {
        return 0;
    }

    for len in (1..=max).rev() {
        let a_start = words_a.len() - len;
        if words_a[a_start..]
            .iter()
            .zip(words_b[..len].iter())
            .all(|(lhs, rhs)| lhs.normalized == rhs.normalized)
        {
            return len;
        }
    }

    0
}

fn byte_after_n_words(text: &str, n_words: usize) -> usize {
    if n_words == 0 {
        return 0;
    }
    let words = collect_word_spans(text);
    if words.is_empty() {
        return 0;
    }
    let index = n_words.saturating_sub(1).min(words.len().saturating_sub(1));
    words[index].end
}

fn drop_n_words(text: &str, n_words: usize) -> &str {
    if n_words == 0 {
        return text;
    }
    let cut = byte_after_n_words(text, n_words);
    text.get(cut..).unwrap_or("").trim_start()
}

#[derive(Debug)]
struct WordSpan {
    normalized: String,
    start: usize,
    end: usize,
}

fn collect_word_spans(text: &str) -> Vec<WordSpan> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if is_word_char(ch) {
            if current_start.is_none() {
                current_start = Some(idx);
            }
        } else if let Some(start) = current_start.take() {
            let token = &text[start..idx];
            let normalized = normalize_word(token);
            if !normalized.is_empty() {
                spans.push(WordSpan {
                    normalized,
                    start,
                    end: idx,
                });
            }
        }
    }

    if let Some(start) = current_start {
        let token = &text[start..];
        let normalized = normalize_word(token);
        if !normalized.is_empty() {
            spans.push(WordSpan {
                normalized,
                start,
                end: text.len(),
            });
        }
    }

    spans
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '\''
}

fn collapse_unstable_repetition(text: &str) -> String {
    let mut collapsed = text.to_string();
    loop {
        let words = collect_word_spans(collapsed.as_str());
        if words.len() < REPETITION_MIN_NGRAM_WORDS * 2 {
            break;
        }

        let max_ngram = REPETITION_MAX_NGRAM_WORDS.min(words.len() / 2);
        let mut remove_range: Option<(usize, usize)> = None;

        'outer: for n in (REPETITION_MIN_NGRAM_WORDS..=max_ngram).rev() {
            let mut start = 0usize;
            while start + n <= words.len() {
                let lookback_start = start.saturating_sub(REPETITION_LOOKBACK_WORDS);
                let mut prev = lookback_start;
                while prev + n <= start {
                    let gap = start.saturating_sub(prev + n);
                    if gap <= REPETITION_MAX_GAP_WORDS
                        && words[prev..prev + n]
                            .iter()
                            .zip(words[start..start + n].iter())
                            .all(|(lhs, rhs)| lhs.normalized == rhs.normalized)
                    {
                        remove_range = Some((words[start].start, words[start + n - 1].end));
                        break 'outer;
                    }
                    prev += 1;
                }
                start += 1;
            }
        }

        let Some((remove_start, remove_end)) = remove_range else {
            break;
        };

        let left = collapsed[..remove_start].trim_end();
        let right = collapsed[remove_end..].trim_start();
        collapsed = concat_transcript(left, right);
    }

    loop {
        let Some((remove_start, remove_end)) = find_near_duplicate_span_range(collapsed.as_str())
        else {
            break;
        };

        let left = collapsed[..remove_start].trim_end();
        let right = collapsed[remove_end..].trim_start();
        collapsed = concat_transcript(left, right);
    }

    collapsed
}

fn find_near_duplicate_span_range(text: &str) -> Option<(usize, usize)> {
    let words = collect_word_spans(text);
    if words.len() < REPETITION_APPROX_MIN_WORDS * 2 {
        return None;
    }

    let max_len = REPETITION_APPROX_MAX_WORDS.min(words.len());
    for len_a in (REPETITION_APPROX_MIN_WORDS..=max_len).rev() {
        if len_a > words.len() {
            continue;
        }
        let min_len_b = len_a
            .saturating_sub(REPETITION_APPROX_MAX_LEN_DELTA)
            .max(REPETITION_APPROX_MIN_WORDS);
        let max_len_b = len_a
            .saturating_add(REPETITION_APPROX_MAX_LEN_DELTA)
            .min(max_len);

        for start_a in 0..=words.len().saturating_sub(len_a) {
            for len_b in (min_len_b..=max_len_b).rev() {
                if len_b > words.len() {
                    continue;
                }
                if start_a + len_b > words.len() {
                    continue;
                }

                let min_start_b = start_a.saturating_add(len_a.saturating_sub(2));
                let max_start_b = start_a
                    .saturating_add(len_a)
                    .saturating_add(REPETITION_APPROX_MAX_GAP_WORDS)
                    .min(words.len().saturating_sub(len_b));
                if min_start_b > max_start_b {
                    continue;
                }

                let span_a = &words[start_a..start_a + len_a];
                for start_b in min_start_b..=max_start_b {
                    let span_b = &words[start_b..start_b + len_b];
                    let prefix_match = prefix_word_match_count(
                        span_a,
                        span_b,
                        REPETITION_APPROX_PREFIX_WINDOW_WORDS,
                    );
                    if prefix_match < REPETITION_APPROX_MIN_PREFIX_MATCH_WORDS {
                        continue;
                    }

                    let lcs = lcs_word_len(span_a, span_b);
                    let similarity = (lcs as f32) / (len_a.max(len_b) as f32);
                    if similarity < REPETITION_APPROX_MIN_LCS_RATIO {
                        continue;
                    }

                    let remove_start = words[start_b].start;
                    let remove_end = words[start_b + len_b - 1].end;
                    return Some((remove_start, remove_end));
                }
            }
        }
    }

    None
}

fn prefix_word_match_count(a: &[WordSpan], b: &[WordSpan], window: usize) -> usize {
    let max = a.len().min(b.len()).min(window);
    let mut matched = 0usize;
    for idx in 0..max {
        if a[idx].normalized == b[idx].normalized {
            matched += 1;
        } else {
            break;
        }
    }
    matched
}

fn lcs_word_len(a: &[WordSpan], b: &[WordSpan]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let mut previous = vec![0usize; b.len() + 1];
    let mut current = vec![0usize; b.len() + 1];

    for left in a {
        for (idx, right) in b.iter().enumerate() {
            current[idx + 1] = if left.normalized == right.normalized {
                previous[idx] + 1
            } else {
                previous[idx + 1].max(current[idx])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }

    previous[b.len()]
}

fn normalize_word(token: &str) -> String {
    token
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .to_lowercase()
}

fn parse_binary_message(data: &[u8]) -> Result<BinaryMessageKind, String> {
    if data.len() < WS_BIN_CLIENT_HEADER_LEN || &data[..4] != WS_BIN_MAGIC {
        return Err("Unexpected binary message (missing transcription frame header)".to_string());
    }

    let version = data[4];
    if version != WS_BIN_VERSION {
        return Err(format!("Unsupported binary frame version {version}"));
    }

    let kind = data[5];
    match kind {
        WS_BIN_KIND_CLIENT_PCM16 => {
            let sample_rate = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            let frame_seq = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
            Ok(BinaryMessageKind::ClientPcm16Frame {
                frame_seq,
                sample_rate,
                payload: data[WS_BIN_CLIENT_HEADER_LEN..].to_vec(),
            })
        }
        other => Err(format!("Unsupported binary frame kind {other}")),
    }
}

fn pcm16_bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

fn pcm16_i16_to_f32(samples_i16: &[i16]) -> Vec<f32> {
    samples_i16
        .iter()
        .map(|sample| *sample as f32 / 32768.0)
        .collect()
}

fn wav_bytes_from_pcm16_mono(samples_i16: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    if sample_rate == 0 {
        return Err("Invalid sample rate 0".to_string());
    }
    let samples_f32 = pcm16_i16_to_f32(samples_i16);
    AudioEncoder::new(sample_rate, 1)
        .encode(&samples_f32, AudioFormat::Wav)
        .map_err(|err| format!("Failed to encode streamed WAV: {err}"))
}

fn send_json(tx: &OutboundTx, value: serde_json::Value) {
    let _ = tx.send(Message::Text(value.to_string().into()));
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn send_worker_command(tx: &mpsc::Sender<WorkerCommand>, command: WorkerCommand) -> bool {
    match tx.try_send(command) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(_)) => {
            warn!("transcription realtime worker command queue is full; dropping command");
            false
        }
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::extract::ws::Message;
    use izwi_core::{EngineConfig, RuntimeService};
    use tokio::sync::mpsc;

    use super::{
        collapse_unstable_repetition, ingest_audio_frame, merge_online_transcript,
        negotiate_transcription_protocol, pcm16_i16_to_f32, send_session_finished,
        strip_suffix_prefix_overlap_with_lookahead_by_words, OutboundTx,
        RealtimeInferenceDiagnostics, RealtimeSessionState, TranscriptionWireProtocol,
        TypedTranscriptionSession,
    };
    use crate::app::realtime_protocol::{
        RealtimeEventFinality, RealtimeServerEnvelope, RealtimeServerEvent,
        TRANSCRIPTION_REALTIME_VERSION,
    };

    #[test]
    fn realtime_protocol_negotiation_defaults_to_legacy_and_requires_explicit_v3() {
        assert_eq!(
            negotiate_transcription_protocol(None, None, None),
            Ok(TranscriptionWireProtocol::LegacyV2)
        );
        assert_eq!(
            negotiate_transcription_protocol(
                Some("transcription_realtime"),
                Some(TRANSCRIPTION_REALTIME_VERSION),
                None,
            ),
            Ok(TranscriptionWireProtocol::TypedV3)
        );
        assert!(negotiate_transcription_protocol(
            Some("transcription_realtime"),
            Some(TRANSCRIPTION_REALTIME_VERSION),
            Some(9),
        )
        .expect_err("v3 resume must be rejected")
        .contains("non-resumable"));
    }

    #[test]
    fn typed_session_ready_wire_shape_and_order_are_stable() {
        let mut session = TypedTranscriptionSession::with_identity(
            "session-1".to_string(),
            "process-42".to_string(),
        );
        let owner_instance_id = session.owner_instance_id.clone();
        let mut ready = session.next_envelope(RealtimeServerEvent::SessionReady {
            accepted_version: TRANSCRIPTION_REALTIME_VERSION,
            owner_instance_id,
            resumable: false,
            resume_window_ms: 0,
        });
        let started = session.next_envelope(RealtimeServerEvent::SessionStarted);

        assert_eq!(ready.event_id, 1);
        assert_eq!(ready.sequence, 0);
        assert_eq!(started.event_id, 2);
        assert_eq!(started.sequence, 1);
        assert_eq!(started.validate_successor(&ready), Ok(()));

        ready.timestamp_ms = 1_725_000_000_123;
        assert_eq!(
            serde_json::to_string(&ready).expect("serialize typed SessionReady"),
            r#"{"protocol":"transcription_realtime","version":3,"event_id":1,"sequence":0,"session_id":"session-1","connection_epoch":0,"timestamp_ms":1725000000123,"type":"session_ready","data":{"accepted_version":3,"owner_instance_id":"process-42","resumable":false,"resume_window_ms":0}}"#
        );
    }

    #[test]
    fn typed_session_finish_emits_final_then_closing_then_closed() {
        let runtime = Arc::new(RuntimeService::new(EngineConfig::default()).expect("runtime"));
        let (tx, mut rx) = mpsc::channel(8);
        let outbound = OutboundTx::new(tx, runtime, "transcription test");
        let mut session = RealtimeSessionState {
            started: true,
            wire_protocol: TranscriptionWireProtocol::TypedV3,
            typed_session: Some(TypedTranscriptionSession::with_identity(
                "session-1".to_string(),
                "process-42".to_string(),
            )),
            committed_text: "stable text".to_string(),
            ..RealtimeSessionState::default()
        };

        send_session_finished(&outbound, &mut session);

        let events = (0..3)
            .map(|_| {
                let Message::Text(text) = rx.try_recv().expect("typed terminal event") else {
                    panic!("expected a text event");
                };
                serde_json::from_str::<RealtimeServerEnvelope>(text.as_str())
                    .expect("typed event envelope")
            })
            .collect::<Vec<_>>();

        assert!(matches!(
            &events[0].event,
            RealtimeServerEvent::TranscriptFinal { .. }
        ));
        assert_eq!(
            events[0].event.finality(),
            RealtimeEventFinality::SegmentFinal
        );
        assert!(matches!(
            &events[1].event,
            RealtimeServerEvent::Closing { .. }
        ));
        assert!(matches!(
            &events[2].event,
            RealtimeServerEvent::Closed { .. }
        ));
        assert_eq!(events[1].validate_successor(&events[0]), Ok(()));
        assert_eq!(events[2].validate_successor(&events[1]), Ok(()));
        assert_eq!(
            events[2].event.finality(),
            RealtimeEventFinality::SessionFinal
        );
    }

    #[tokio::test]
    async fn outbound_tx_drops_full_queue_and_records_backpressure() {
        let runtime = Arc::new(RuntimeService::new(EngineConfig::default()).expect("runtime"));
        let before = runtime
            .telemetry_snapshot()
            .await
            .realtime
            .transcription_stream_backpressure_total;
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(Message::Text("occupied".into()))
            .expect("queue should accept first message");
        let outbound = OutboundTx::new(tx, runtime.clone(), "transcription test");

        assert!(!outbound.send(Message::Text("dropped".into())));

        let after = runtime
            .telemetry_snapshot()
            .await
            .realtime
            .transcription_stream_backpressure_total;
        assert_eq!(after, before + 1);
        assert!(rx.try_recv().is_ok());
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn realtime_inference_diagnostics_omit_transcript_content() {
        let diagnostics = RealtimeInferenceDiagnostics {
            route: "transcription.realtime",
            sequence: 7,
            sample_rate: 16_000,
            input_sample_count: 8_000,
            audio_duration_secs: 0.5,
            queue_wait_ms: 2.0,
            processing_time_ms: 125.0,
            rtf: Some(0.25),
            output_text_chars: "private words".chars().count(),
            cancellation_reason: None,
        };

        let value = serde_json::to_value(&diagnostics).expect("serialize diagnostics");

        assert_eq!(value["route"], "transcription.realtime");
        assert_eq!(value["sample_rate"], 16_000);
        assert_eq!(value["output_text_chars"], 13);
        assert!(value.get("text").is_none());
        assert!(value.get("transcript").is_none());
    }

    #[test]
    fn pcm16_i16_to_f32_uses_standard_pcm16_scale() {
        let converted = pcm16_i16_to_f32(&[-32768, -16384, 0, 16384, 32767]);

        assert_eq!(converted[0], -1.0);
        assert_eq!(converted[1], -0.5);
        assert_eq!(converted[2], 0.0);
        assert_eq!(converted[3], 0.5);
        assert!((converted[4] - 0.9999695).abs() < 1e-7);
    }

    #[test]
    fn ingest_audio_frame_returns_only_fresh_samples_for_native_streaming() {
        let mut session = RealtimeSessionState {
            started: true,
            ..RealtimeSessionState::default()
        };
        let payload = [0x00, 0x00, 0x00, 0x80, 0xff, 0x7f];

        let samples = ingest_audio_frame(&mut session, 1, 16_000, &payload).expect("fresh frame");

        assert_eq!(samples, vec![0, -32768, 32767]);
        assert_eq!(session.samples_i16, samples);
        assert_eq!(session.sample_rate, Some(16_000));
        assert_eq!(session.last_frame_seq, Some(1));
        assert!(session.pending_recompute);

        let stale_samples =
            ingest_audio_frame(&mut session, 1, 16_000, &payload).expect("stale frame ignored");

        assert!(stale_samples.is_empty());
        assert_eq!(session.samples_i16, samples);
    }

    #[test]
    fn merge_online_transcript_deduplicates_committed_prefix_restarts() {
        let mut committed = "So I want to ".to_string();
        let mut trailing = "test uh ".to_string();

        let merged = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "So I want to test uh all time streaming to see if it's working",
        );

        assert_eq!(
            merged,
            "So I want to test uh all time streaming to see if it's working"
        );
        assert!(!merged.contains("So I want to So I want to"));
    }

    #[test]
    fn merge_online_transcript_allows_tail_extension_after_restart() {
        let mut committed = "hello ".to_string();
        let mut trailing = "world".to_string();

        let merged = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "hello world from realtime model",
        );

        assert_eq!(merged, "hello world from realtime model");
    }

    #[test]
    fn merge_online_transcript_handles_punctuation_spacing_variation_without_duplication() {
        let mut committed = String::new();
        let mut trailing = "Hi,so we are going to start doing another test.".to_string();

        let merged = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Hi, so we are going to start doing another test to see if this is going to work.",
        );

        assert!(merged.ends_with("to see if this is going to work."));
        assert!(merged.contains("start doing another test"));
        assert!(!merged.contains("Hi,so we Hi, so we"));
    }

    #[test]
    fn collapse_unstable_repetition_removes_recent_duplicate_ngram() {
        let text = "Iran seemed to be bombing some other countries and I'm not Iran seemed to be bombing some other countries and I'm not sure what's going on";
        let collapsed = collapse_unstable_repetition(text);
        assert_eq!(
            collapsed,
            "Iran seemed to be bombing some other countries and I'm not sure what's going on"
        );
    }

    #[test]
    fn strip_suffix_prefix_overlap_with_lookahead_ignores_spacing_and_commas() {
        let stripped = strip_suffix_prefix_overlap_with_lookahead_by_words(
            "Hi,so we are going",
            "Hi, so we are going to test",
        );
        assert_eq!(stripped, "to test");
    }

    #[test]
    fn local_agreement_two_commits_common_prefix() {
        let mut committed = String::new();
        let mut trailing = String::new();

        let first = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Now so today is not a good day for me",
        );
        assert_eq!(committed, "");
        assert_eq!(first, "Now so today is not a good day for me");

        let second = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Now, so today is not a good day for me. I'm a little bit sad",
        );
        assert!(committed.contains("today is not a good day for me"));
        assert!(second.contains("I'm a little bit sad"));
        assert!(!second.contains("Now so today is not a good day for me Now"));
    }

    #[test]
    fn committed_overlap_lookahead_handles_changed_leading_words() {
        let stripped = strip_suffix_prefix_overlap_with_lookahead_by_words(
            "Now, so today is not a good day for me",
            "Yeah, so today is not a good day for me. I'm a little bit sad",
        );
        assert!(!stripped.contains("today is not a good day for me"));
        assert!(stripped.contains("I'm a little bit sad"));
    }

    #[test]
    fn collapse_unstable_repetition_handles_glued_punctuation_restarts() {
        let text = "Hi,so we Hi, so we are going to start doing another test.Hi,so we are going to start doing another test to see if this is going to work.";
        let collapsed = collapse_unstable_repetition(text);
        assert!(
            !collapsed.contains("Hi,so we Hi, so we"),
            "collapsed transcript still contains duplicated restart: {collapsed}"
        );
        assert!(collapsed.contains("to see if this is going to work"));
    }

    #[test]
    fn merge_online_transcript_handles_restart_with_changed_leading_words() {
        let mut committed = String::new();
        let mut trailing = String::new();

        let _ = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Now, so today is nota good day for me",
        );
        let second = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Yeah, so today is not a good day for me. I'm a little bit sad",
        );
        assert!(
            !second.contains("meYeah"),
            "restart should not be concatenated without boundary spacing"
        );
        assert!(
            !second.contains("Now, so today is nota good day for meYeah"),
            "stale prefix should remain mutable before local agreement commit"
        );

        let third = merge_online_transcript(
            &mut committed,
            &mut trailing,
            "Yeah, so today is not a good day for me. I'm a little bit sad Day is not a good day for me. I'm a little bit sad, so things are not good.",
        );
        assert!(
            !third.contains("Day is not a good day for me. I'm a little bit sad"),
            "restart repetition should be collapsed in unstable tail: {third}"
        );
    }

    #[test]
    fn local_agreement_two_stays_stable_over_longer_partial_sequence() {
        let mut committed = String::new();
        let mut trailing = String::new();

        let updates = [
            "Now, so today is nota good day for me",
            "Yeah, so today is not a good day for me. I'm a little bit sad",
            "Yeah, so today is not a good day for me. I'm a little bit sad, so things are not good.",
            "Yeah, so today is not a good day for me. I'm a little bit sad, so things are not good. I don't know.",
            "Yeah, so today is not a good day for me. I'm a little bit sad, so things are not good. I don't know. I don't really know what to do now.",
        ];

        let mut final_text = String::new();
        for update in updates {
            final_text = merge_online_transcript(&mut committed, &mut trailing, update);
        }

        assert!(final_text.contains("I don't really know what to do now"));
        assert!(!final_text.contains("good day for meYeah"));
        assert!(!final_text.contains("sad Day is not a good day for me"));
    }

    #[test]
    fn collapse_unstable_repetition_removes_near_duplicate_clause_burst() {
        let text = "Okay, so it seems the performance of this thing is a little bit better, but uh Okay, so it seems the performance of this thing is a little bit better, but um we can definitely improve it.";
        let collapsed = collapse_unstable_repetition(text);
        assert!(
            !collapsed.contains("better, but uh Okay, so it seems the performance"),
            "near-duplicate clause burst should be collapsed: {collapsed}"
        );
        assert!(collapsed.contains("we can definitely improve it"));
    }

    #[test]
    fn collapse_unstable_repetition_collapses_youre_vs_you_are_restart() {
        let text = "you know uh interpret uh what you're saying you know uh interpret uh what you are saying";
        let collapsed = collapse_unstable_repetition(text);
        assert!(
            !collapsed.contains("you're saying you know uh interpret uh what you are saying"),
            "contraction/wording variant restart should be collapsed: {collapsed}"
        );
    }
}

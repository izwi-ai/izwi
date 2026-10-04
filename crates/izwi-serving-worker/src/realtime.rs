//! Realtime WebSocket sessions (`izwi-realtime-v1`) on the worker boundary.
//!
//! One session binds exactly one realtime stage to this worker. Admission
//! reuses the atomic admission gate and the shared attempt table, so HTTP
//! `query_attempt`/`cancel_attempt` behave identically for realtime sessions
//! and draining fences new sessions exactly like new invocations. Cancellation
//! lands between audio pushes — the engine's input-quantum boundaries — and
//! capacity is released only after the stage's runtime stream has been
//! dropped (its job, session, and residency leases release in that drop; the
//! runtime's own watchdog additionally enforces absolute lifetime and idle
//! bounds on the stream itself).

use super::{
    InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner, RealtimeTtsStageStream,
    WorkerState, EVENT_ENVELOPE_ALLOWANCE,
};
use crate::runtime::map_execution_error;
use axum::{
    extract::{
        ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use izwi_serving_protocol::{
    decode_realtime_audio_frame, encode_realtime_audio_frame, AttemptState, InvocationErrorCode,
    InvocationEvent, InvocationEventKind, ModelReadiness, PermittedAction, RealtimeAudioCodec,
    RealtimeAudioSpec, RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit,
    RealtimeSessionBounds, RealtimeSessionCloseCode, RealtimeStageInput, RequestDigest, TaskKind,
    WorkerFeature, MAX_REALTIME_AUDIO_FRAME_BYTES, MAX_REALTIME_INPUT_TEXT_BYTES, PROTOCOL_V1,
    REALTIME_SUBPROTOCOL,
};
use std::{
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch, OwnedSemaphorePermit};

/// Bounded realtime session policy for this worker build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeRealtimeSessionLimits {
    /// Maximum binary audio frame payload accepted from the gateway.
    pub max_frame_bytes: usize,
    /// Audio frames the worker buffers for one session; the reader channel
    /// capacity equals this bound so overflow is TCP backpressure, never
    /// unbounded buffering.
    pub max_in_flight_frames: usize,
    /// Total audio bytes admitted for one session.
    pub max_session_audio_bytes: u64,
    /// How long the worker waits for the admit frame after the upgrade.
    pub admit_timeout: Duration,
    /// Outbound event/audio frame queue depth toward the gateway.
    pub max_outbound_queue: usize,
}

pub const REALTIME_SESSION_DEFAULT_LIMITS: RuntimeRealtimeSessionLimits =
    RuntimeRealtimeSessionLimits {
        max_frame_bytes: MAX_REALTIME_AUDIO_FRAME_BYTES,
        max_in_flight_frames: 16,
        max_session_audio_bytes: 256 * 1024 * 1024,
        admit_timeout: Duration::from_secs(10),
        max_outbound_queue: 64,
    };

impl RuntimeRealtimeSessionLimits {
    /// Fails closed unless every bound is non-zero and within the protocol's
    /// hard caps, so a misconfigured worker never negotiates beyond v1.
    pub fn validate(&self) -> Result<(), izwi_serving_protocol::RealtimeContractError> {
        RealtimeSessionBounds {
            max_frame_bytes: self.max_frame_bytes,
            max_in_flight_frames: self.max_in_flight_frames,
            max_session_audio_bytes: self.max_session_audio_bytes,
        }
        .validate()?;
        if self.admit_timeout.is_zero() || self.max_outbound_queue == 0 {
            return Err(izwi_serving_protocol::RealtimeContractError::InvalidBounds);
        }
        Ok(())
    }
}

/// Hard cap on any single WebSocket message (control JSON or audio frame).
const REALTIME_WS_MAX_MESSAGE_BYTES: usize = 1024 * 1024;
/// Cap on the JSON admit/control frame payload.
const REALTIME_CONTROL_FRAME_BYTES: usize = 64 * 1024;
const WRITER_SEND_TIMEOUT: Duration = Duration::from_secs(5);
const TERMINATION_JOIN_TIMEOUT: Duration = Duration::from_secs(10);

/// One inbound audio frame, decoded and bounds-checked at the reader.
struct AudioFrame {
    sequence: u32,
    payload: Vec<u8>,
}

enum ClientControl {
    Finish,
    Cancel,
    Ping,
    /// Bounded text input for the TTS-stream stage; the reader already
    /// enforced `MAX_REALTIME_INPUT_TEXT_BYTES`.
    Input(String),
    Violation(&'static str),
}

enum OutboundMessage {
    Text(String),
    /// One binary audio frame toward the gateway (TTS-stream stage).
    Binary(Vec<u8>),
    /// Normal session end (code 1000) or a pre-admission/close rejection.
    Close(Option<RealtimeSessionCloseCode>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventSendResult {
    Sent,
    Oversized,
    Unavailable,
}

struct AdmitFailure {
    close: RealtimeSessionCloseCode,
}

impl AdmitFailure {
    const fn close(close: RealtimeSessionCloseCode) -> Self {
        Self { close }
    }
}

fn offers_realtime_subprotocol(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::SEC_WEBSOCKET_PROTOCOL)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|offered| {
            offered
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case(REALTIME_SUBPROTOCOL))
        })
}

pub async fn realtime_socket<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if state.draining.load(Ordering::Acquire) {
        return (StatusCode::SERVICE_UNAVAILABLE, "worker is draining").into_response();
    }
    // Fail closed unless the deployment advertises realtime capability, the
    // descriptor advertises the socket surface, and the executor serves it.
    if !state.config.deployment.capability.realtime
        || !state
            .config
            .descriptor
            .features
            .contains(&WorkerFeature::RealtimeSocket)
        || !matches!(
            state.config.deployment.task,
            TaskKind::SpeechToText | TaskKind::TextToSpeech
        )
    {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(runner) = state.executor.realtime_runner() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if !offers_realtime_subprotocol(&headers) {
        return (
            StatusCode::BAD_REQUEST,
            "the izwi-realtime-v1 subprotocol must be offered",
        )
            .into_response();
    }
    upgrade
        .protocols([REALTIME_SUBPROTOCOL])
        .max_message_size(REALTIME_WS_MAX_MESSAGE_BYTES)
        .max_frame_size(REALTIME_WS_MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| run_realtime_session(state, runner, socket))
}

async fn run_realtime_session<E: InvocationExecutor>(
    state: Arc<WorkerState<E>>,
    runner: Arc<dyn RealtimeStageRunner>,
    socket: WebSocket,
) {
    let limits = state.config.realtime_session_limits;
    let (outbound_tx, outbound_rx) = mpsc::channel::<OutboundMessage>(limits.max_outbound_queue);
    let (sink, mut source) = socket.split();
    let writer = tokio::spawn(writer_task(sink, outbound_rx));

    let mut session = match admit_session(
        &state,
        &mut source,
        &outbound_tx,
        limits,
        runner.output_audio_spec(),
    )
    .await
    {
        Ok(session) => session,
        Err(failure) => {
            let _ = outbound_tx
                .send(OutboundMessage::Close(Some(failure.close)))
                .await;
            drop(outbound_tx);
            let _ = tokio::time::timeout(TERMINATION_JOIN_TIMEOUT, writer).await;
            return;
        }
    };

    let (control_tx, control_rx) = mpsc::channel::<ClientControl>(8);
    let (audio_tx, audio_rx) = mpsc::channel::<AudioFrame>(limits.max_in_flight_frames);
    let mut reader = tokio::spawn(reader_task(source, control_tx, audio_tx, limits));
    session.run(&runner, control_rx, audio_rx).await;
    session.finish();
    // Deliver a final close if the terminal publication could not queue one,
    // then let the writer drain and exit.
    let _ = outbound_tx.send(OutboundMessage::Close(None)).await;
    drop(outbound_tx);
    // Drain the reader before the socket closes: aborting it with inbound
    // frames still buffered would turn the orderly close into a TCP reset and
    // could cost the client its terminal event. The reader exits on the close
    // handshake, on a failed send (its receivers are gone), or on transport
    // end; the abort is only a backstop for a client that never completes the
    // handshake.
    if tokio::time::timeout(TERMINATION_JOIN_TIMEOUT, &mut reader)
        .await
        .is_err()
    {
        reader.abort();
    }
    let _ = tokio::time::timeout(TERMINATION_JOIN_TIMEOUT, writer).await;
}

async fn writer_task(
    mut sink: SplitSink<WebSocket, Message>,
    mut rx: mpsc::Receiver<OutboundMessage>,
) {
    while let Some(message) = rx.recv().await {
        let outbound = match message {
            OutboundMessage::Text(text) => Message::Text(Utf8Bytes::from(text)),
            OutboundMessage::Binary(bytes) => Message::Binary(bytes.into()),
            OutboundMessage::Close(code) => {
                let frame = code
                    .map(|code| CloseFrame {
                        code: code.code(),
                        reason: Utf8Bytes::from_static(code.reason()),
                    })
                    .unwrap_or(CloseFrame {
                        code: 1000,
                        reason: Utf8Bytes::from_static("session complete"),
                    });
                let _ = sink.send(Message::Close(Some(frame))).await;
                break;
            }
        };
        if tokio::time::timeout(WRITER_SEND_TIMEOUT, sink.send(outbound))
            .await
            .is_err()
        {
            break;
        }
    }
}

async fn reader_task(
    mut source: SplitStream<WebSocket>,
    control_tx: mpsc::Sender<ClientControl>,
    audio_tx: mpsc::Sender<AudioFrame>,
    limits: RuntimeRealtimeSessionLimits,
) {
    while let Some(message) = source.next().await {
        let control = match message {
            Ok(Message::Text(text)) => {
                if text.len() > REALTIME_CONTROL_FRAME_BYTES {
                    Some(ClientControl::Violation("control frame exceeds bound"))
                } else {
                    match serde_json::from_str::<RealtimeClientFrame>(&text) {
                        Ok(RealtimeClientFrame::Finish) => Some(ClientControl::Finish),
                        Ok(RealtimeClientFrame::Cancel) => Some(ClientControl::Cancel),
                        Ok(RealtimeClientFrame::Ping) => Some(ClientControl::Ping),
                        Ok(RealtimeClientFrame::Input { text }) => {
                            if text.len() <= MAX_REALTIME_INPUT_TEXT_BYTES {
                                Some(ClientControl::Input(text))
                            } else {
                                Some(ClientControl::Violation("input text exceeds bound"))
                            }
                        }
                        Ok(RealtimeClientFrame::Admit { .. }) => {
                            Some(ClientControl::Violation("duplicate admit"))
                        }
                        Err(_) => Some(ClientControl::Violation("unparseable control frame")),
                    }
                }
            }
            Ok(Message::Binary(data)) => match decode_realtime_audio_frame(&data) {
                Ok((header, payload)) if payload.len() <= limits.max_frame_bytes => {
                    if audio_tx
                        .send(AudioFrame {
                            sequence: header.sequence,
                            payload: payload.to_vec(),
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                Ok(_) => Some(ClientControl::Violation("audio frame exceeds bound")),
                Err(_) => Some(ClientControl::Violation("malformed audio frame")),
            },
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(_) | Message::Pong(_)) => continue,
            Err(_) => break,
        };
        if let Some(control) = control {
            if control_tx.send(control).await.is_err() {
                break;
            }
        }
    }
}

/// Session after admission: owns the admit identity, the capacity permit, the
/// cancellation watch, and the monotonic event sequence.
struct RealtimeSession<E> {
    state: Arc<WorkerState<E>>,
    admit: RealtimeSessionAdmit,
    permit: OwnedSemaphorePermit,
    cancel: watch::Receiver<bool>,
    remaining_time: Duration,
    started_at: Instant,
    sequence: u64,
    outbound: mpsc::Sender<OutboundMessage>,
    cancellation_started_at: Option<Instant>,
}

impl<E: InvocationExecutor> RealtimeSession<E> {
    fn send_server_frame(&self, frame: &RealtimeServerFrame) -> EventSendResult {
        let Ok(encoded) = serde_json::to_vec(frame) else {
            return EventSendResult::Oversized;
        };
        if encoded.len().saturating_add(EVENT_ENVELOPE_ALLOWANCE)
            > self.state.config.max_event_bytes
        {
            return EventSendResult::Oversized;
        }
        match self.outbound.try_send(OutboundMessage::Text(
            String::from_utf8(encoded).expect("serde json is utf8"),
        )) {
            Ok(()) => EventSendResult::Sent,
            Err(_) => EventSendResult::Unavailable,
        }
    }

    /// Emits one sequenced `InvocationEvent` and advances the sequence.
    fn emit_event(&mut self, event: InvocationEventKind) -> EventSendResult {
        let event = InvocationEvent {
            schema_version: PROTOCOL_V1,
            request_id: self.admit.request_id.clone(),
            attempt_id: self.admit.attempt_id.clone(),
            sequence: self.sequence,
            event,
        };
        let result = self.send_server_frame(&RealtimeServerFrame::Event { event });
        if result == EventSendResult::Sent {
            self.sequence = self.sequence.saturating_add(1);
        }
        result
    }

    /// Records execution and cancellation metrics and releases the capacity
    /// permit after the stage confirmed teardown by dropping its runtime
    /// stream.
    fn finish(self) {
        self.state
            .metrics
            .record_execution(self.started_at.elapsed());
        if let Some(started_at) = self.cancellation_started_at {
            self.state
                .metrics
                .record_cancellation_stopped(started_at.elapsed());
        }
        drop(self.permit);
    }

    fn mark_cancellation(&mut self) {
        if self.cancellation_started_at.is_none() {
            self.cancellation_started_at = Some(Instant::now());
            self.state.metrics.record_cancellation_started();
        }
        self.state.update_attempt(
            &self.admit.attempt_id,
            AttemptState::ExecutionStopping,
            None,
        );
    }

    async fn run(
        &mut self,
        runner: &Arc<dyn RealtimeStageRunner>,
        control_rx: mpsc::Receiver<ClientControl>,
        audio_rx: mpsc::Receiver<AudioFrame>,
    ) {
        // Contract parity with the HTTP path: the first event is the Accepted
        // event at sequence 0, and the session moves to Running.
        let accepted = InvocationEventKind::Accepted {
            worker_id: self.state.config.descriptor.worker_id.clone(),
            node_id: self.state.config.descriptor.node_id.clone(),
            incarnation_id: self.state.config.descriptor.incarnation_id.clone(),
            deployment_id: self.state.config.deployment.deployment_id.clone(),
            model_generation: self.state.config.deployment.model_generation,
        };
        if self.emit_event(accepted) == EventSendResult::Sent {
            self.state
                .update_attempt(&self.admit.attempt_id, AttemptState::Running, Some(0));
        }
        match runner.stage_task() {
            TaskKind::SpeechToText => self.run_asr_stage(runner, control_rx, audio_rx).await,
            TaskKind::TextToSpeech => self.run_tts_stage(runner, control_rx).await,
            // WorkerConfig::validate only admits the two stages above; this
            // arm keeps the session total if that ever changes.
            _ => {
                self.terminate_failed(
                    InvocationErrorCode::Internal,
                    "stage has no worker execution path",
                );
            }
        }
    }

    async fn run_asr_stage(
        &mut self,
        runner: &Arc<dyn RealtimeStageRunner>,
        mut control_rx: mpsc::Receiver<ClientControl>,
        mut audio_rx: mpsc::Receiver<AudioFrame>,
    ) {
        let spec = match &self.admit.input {
            izwi_serving_protocol::RealtimeStageInput::AudioStream { spec, .. } => *spec,
            _ => {
                self.terminate_failed(
                    InvocationErrorCode::InvalidInput,
                    "speech_to_text sessions require an audio-stream admission",
                );
                return;
            }
        };
        let RealtimeAudioSpec {
            codec: RealtimeAudioCodec::PcmI16Le,
            sample_rate,
            channels: _,
        } = spec;

        let mut stream: Option<Box<dyn RealtimeAsrStageStream>> = None;
        let mut last_audio_sequence: Option<u32> = None;
        let mut session_audio_bytes: u64 = 0;
        let mut cancellation_requested = *self.cancel.borrow();
        let mut timed_out = false;
        let mut terminal: Option<StageTerminal> = None;
        if cancellation_requested {
            self.mark_cancellation();
        }

        while terminal.is_none() && !cancellation_requested {
            let deadline = tokio::time::sleep_until(
                tokio::time::Instant::from_std(self.started_at) + self.remaining_time,
            );
            tokio::pin!(deadline);
            tokio::select! {
                biased;
                () = &mut deadline => {
                    timed_out = true;
                    cancellation_requested = true;
                    self.mark_cancellation();
                }
                _ = self.outbound.closed() => {
                    cancellation_requested = true;
                    self.mark_cancellation();
                }
                changed = self.cancel.changed(), if !cancellation_requested => {
                    if changed.is_ok() && *self.cancel.borrow() {
                        cancellation_requested = true;
                        self.mark_cancellation();
                    }
                }
                inbound = control_rx.recv() => {
                    match inbound {
                        Some(ClientControl::Finish) => {
                            terminal = self.finish_asr(&mut stream).await;
                        }
                        Some(ClientControl::Cancel) => {
                            cancellation_requested = true;
                            self.mark_cancellation();
                        }
                        Some(ClientControl::Ping) => {
                            let _ = self.send_server_frame(&RealtimeServerFrame::Pong);
                        }
                        Some(ClientControl::Input(_)) => {
                            terminal = Some(StageTerminal::Failed {
                                code: InvocationErrorCode::InvalidInput,
                                message: "text input is not valid for speech_to_text sessions".into(),
                            });
                        }
                        Some(ClientControl::Violation(reason)) => {
                            terminal = Some(StageTerminal::Failed {
                                code: InvocationErrorCode::InvalidInput,
                                message: reason.into(),
                            });
                        }
                        // Reader gone: the peer disconnected mid-session.
                        None => {
                            cancellation_requested = true;
                            self.mark_cancellation();
                        }
                    }
                }
                frame = audio_rx.recv() => {
                    match frame {
                        Some(frame) => {
                            if let Err(failure) = self
                                .push_audio(
                                    runner,
                                    &mut stream,
                                    &mut last_audio_sequence,
                                    &mut session_audio_bytes,
                                    frame,
                                    sample_rate,
                                )
                                .await
                            {
                                match failure {
                                    StageTerminal::Disconnected => {
                                        cancellation_requested = true;
                                        self.mark_cancellation();
                                    }
                                    terminal_failure => terminal = Some(terminal_failure),
                                }
                            }
                        }
                        None => {
                            cancellation_requested = true;
                            self.mark_cancellation();
                        }
                    }
                }
            }
        }

        // Teardown confirmation for the ASR stage: dropping the runtime
        // stream releases its job, session, and residency leases
        // synchronously. A stream that never started has nothing to confirm.
        drop(stream);
        if timed_out {
            self.terminate_failed(
                InvocationErrorCode::DeadlineExceeded,
                "session deadline elapsed; execution teardown is confirmed",
            );
            return;
        }
        if cancellation_requested {
            self.terminate_cancelled();
            return;
        }
        match terminal.expect("loop exit carries a terminal outcome") {
            StageTerminal::Completed { text } => {
                if let Some(text) = text.filter(|text| !text.is_empty()) {
                    match self.emit_event(InvocationEventKind::TextDelta {
                        text,
                        logprobs: None,
                    }) {
                        EventSendResult::Sent => {}
                        EventSendResult::Oversized => {
                            self.terminate_failed(
                                InvocationErrorCode::OutputLimitExceeded,
                                "encoded output event exceeded the worker limit",
                            );
                            return;
                        }
                        EventSendResult::Unavailable => {
                            self.terminate_cancelled();
                            return;
                        }
                    }
                }
                self.publish_terminal(
                    AttemptState::Completed,
                    InvocationEventKind::Completed {
                        finish_reason: izwi_serving_protocol::FinishReason::Stop,
                        usage: None,
                    },
                );
            }
            StageTerminal::Failed { code, message } => {
                self.terminate_failed(code, &message);
            }
            StageTerminal::Disconnected => {
                self.terminate_cancelled();
            }
        }
    }

    async fn push_audio(
        &mut self,
        runner: &Arc<dyn RealtimeStageRunner>,
        stream: &mut Option<Box<dyn RealtimeAsrStageStream>>,
        last_audio_sequence: &mut Option<u32>,
        session_audio_bytes: &mut u64,
        frame: AudioFrame,
        sample_rate: u32,
    ) -> Result<(), StageTerminal> {
        if last_audio_sequence.is_some_and(|last| frame.sequence <= last) {
            return Err(StageTerminal::Failed {
                code: InvocationErrorCode::InvalidInput,
                message: "audio frame sequence regressed".into(),
            });
        }
        *last_audio_sequence = Some(frame.sequence);
        *session_audio_bytes = session_audio_bytes.saturating_add(frame.payload.len() as u64);
        if *session_audio_bytes
            > self
                .state
                .config
                .realtime_session_limits
                .max_session_audio_bytes
        {
            return Err(StageTerminal::Failed {
                code: InvocationErrorCode::OutputLimitExceeded,
                message: "session audio budget exhausted".into(),
            });
        }
        if stream.is_none() {
            match runner.start_asr_stream(language_arg(&self.admit)).await {
                Ok(started) => *stream = Some(started),
                Err(error) => {
                    let failure = map_execution_error(error);
                    return Err(StageTerminal::Failed {
                        code: failure.code,
                        message: failure.message,
                    });
                }
            }
        }
        let active = stream
            .as_mut()
            .expect("stream is started before the first push");
        let samples: Vec<f32> = frame
            .payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| i16::from_le_bytes(*chunk) as f32 / 32768.0)
            .collect();
        let events = match active.push_samples(&samples, sample_rate).await {
            Ok(events) => events,
            Err(error) => {
                let failure = map_execution_error(error);
                return Err(StageTerminal::Failed {
                    code: failure.code,
                    message: failure.message,
                });
            }
        };
        for event in events {
            if event.delta.is_empty() {
                continue;
            }
            match self.emit_event(InvocationEventKind::TextDelta {
                text: event.delta,
                logprobs: None,
            }) {
                EventSendResult::Sent => {
                    self.state.update_attempt(
                        &self.admit.attempt_id,
                        AttemptState::Running,
                        Some(self.sequence.saturating_sub(1)),
                    );
                }
                EventSendResult::Oversized => {
                    return Err(StageTerminal::Failed {
                        code: InvocationErrorCode::OutputLimitExceeded,
                        message: "encoded output event exceeded the worker limit".into(),
                    });
                }
                EventSendResult::Unavailable => return Err(StageTerminal::Disconnected),
            }
        }
        Ok(())
    }

    async fn finish_asr(
        &mut self,
        stream: &mut Option<Box<dyn RealtimeAsrStageStream>>,
    ) -> Option<StageTerminal> {
        let Some(stream) = stream.as_mut() else {
            // No audio ever arrived: finish is a successful empty transcript.
            return Some(StageTerminal::Completed { text: None });
        };
        let events = match stream.finish().await {
            Ok(events) => events,
            Err(error) => {
                let failure = map_execution_error(error);
                return Some(StageTerminal::Failed {
                    code: failure.code,
                    message: failure.message,
                });
            }
        };
        let mut final_text = None;
        for event in events {
            if event.is_final {
                final_text = Some(event.text);
            } else if !event.delta.is_empty() {
                match self.emit_event(InvocationEventKind::TextDelta {
                    text: event.delta,
                    logprobs: None,
                }) {
                    EventSendResult::Sent => {
                        self.state.update_attempt(
                            &self.admit.attempt_id,
                            AttemptState::Running,
                            Some(self.sequence.saturating_sub(1)),
                        );
                    }
                    EventSendResult::Oversized => {
                        return Some(StageTerminal::Failed {
                            code: InvocationErrorCode::OutputLimitExceeded,
                            message: "encoded output event exceeded the worker limit".into(),
                        });
                    }
                    EventSendResult::Unavailable => return Some(StageTerminal::Disconnected),
                }
            }
        }
        Some(StageTerminal::Completed { text: final_text })
    }

    /// TTS-stream stage: the client accumulates the utterance with `Input`
    /// frames, `Finish` commits synthesis, and audio streams out as binary
    /// IRTA frames while the model generates. Cancellation semantics are
    /// identical to the ASR stage; teardown is confirmed when the synthesis
    /// forwarder exits after the session drops its chunk receiver.
    async fn run_tts_stage(
        &mut self,
        runner: &Arc<dyn RealtimeStageRunner>,
        mut control_rx: mpsc::Receiver<ClientControl>,
    ) {
        if !matches!(self.admit.input, RealtimeStageInput::TextStream) {
            self.terminate_failed(
                InvocationErrorCode::InvalidInput,
                "text_to_speech sessions require a text-stream admission",
            );
            return;
        }
        let limits = self.state.config.realtime_session_limits;
        let max_frame_bytes = limits.max_frame_bytes;
        let session_budget = limits.max_session_audio_bytes;

        // The sender moves into the synthesis forwarder on Finish, so the
        // receiver below returns `None` exactly when synthesis ends. Before
        // that, the open channel keeps this arm parked.
        let (chunks_tx, mut chunks_rx) =
            mpsc::channel::<Result<izwi_core::AudioChunk, izwi_core::Error>>(8);
        let mut chunks_tx = Some(chunks_tx);
        let mut synthesis: Option<tokio::task::JoinHandle<()>> = None;
        let mut text = String::new();
        let mut text_bytes = 0u64;
        let mut finish_submitted = false;
        let mut audio_sequence = 0u32;
        let mut audio_bytes_out = 0u64;
        let mut emitted_final = false;
        let mut cancellation_requested = *self.cancel.borrow();
        let mut timed_out = false;
        let mut terminal: Option<StageTerminal> = None;
        if cancellation_requested {
            self.mark_cancellation();
        }

        while terminal.is_none() && !cancellation_requested {
            let deadline = tokio::time::sleep_until(
                tokio::time::Instant::from_std(self.started_at) + self.remaining_time,
            );
            tokio::pin!(deadline);
            tokio::select! {
                biased;
                () = &mut deadline => {
                    timed_out = true;
                    cancellation_requested = true;
                    self.mark_cancellation();
                }
                _ = self.outbound.closed() => {
                    cancellation_requested = true;
                    self.mark_cancellation();
                }
                changed = self.cancel.changed(), if !cancellation_requested => {
                    if changed.is_ok() && *self.cancel.borrow() {
                        cancellation_requested = true;
                        self.mark_cancellation();
                    }
                }
                inbound = control_rx.recv() => {
                    match inbound {
                        Some(ClientControl::Finish) => {
                            if finish_submitted {
                                terminal = Some(StageTerminal::Failed {
                                    code: InvocationErrorCode::InvalidInput,
                                    message: "finish was already submitted".into(),
                                });
                            } else {
                                finish_submitted = true;
                                let utterance = std::mem::take(&mut text);
                                if utterance.is_empty() {
                                    // Mirror the ASR stage: finishing without
                                    // input is a successful empty result.
                                    terminal = Some(StageTerminal::Completed { text: None });
                                } else {
                                    self.state.update_attempt(
                                        &self.admit.attempt_id,
                                        AttemptState::Running,
                                        Some(0),
                                    );
                                    let deadline_at = tokio::time::Instant::from_std(
                                        self.started_at,
                                    ) + self.remaining_time;
                                    match runner
                                        .start_tts_stream(utterance, deadline_at.into_std())
                                        .await
                                    {
                                        Ok(stream) => {
                                            // Finish is processed at most once;
                                            // the take makes that hold for the
                                            // sender too.
                                            if let Some(chunks_tx) = chunks_tx.take() {
                                                synthesis = Some(tokio::spawn(
                                                    forward_synthesis(stream, chunks_tx),
                                                ));
                                            }
                                        }
                                        Err(error) => {
                                            let failure = map_execution_error(error);
                                            terminal = Some(StageTerminal::Failed {
                                                code: failure.code,
                                                message: failure.message,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        Some(ClientControl::Cancel) => {
                            cancellation_requested = true;
                            self.mark_cancellation();
                        }
                        Some(ClientControl::Ping) => {
                            let _ = self.send_server_frame(&RealtimeServerFrame::Pong);
                        }
                        Some(ClientControl::Input(incoming)) => {
                            if finish_submitted {
                                terminal = Some(StageTerminal::Failed {
                                    code: InvocationErrorCode::InvalidInput,
                                    message: "input after finish".into(),
                                });
                            } else {
                                text_bytes = text_bytes.saturating_add(incoming.len() as u64);
                                if text_bytes > session_budget {
                                    terminal = Some(StageTerminal::Failed {
                                        code: InvocationErrorCode::InvalidInput,
                                        message: "session text budget exhausted".into(),
                                    });
                                } else {
                                    text.push_str(&incoming);
                                }
                            }
                        }
                        Some(ClientControl::Violation(reason)) => {
                            terminal = Some(StageTerminal::Failed {
                                code: InvocationErrorCode::InvalidInput,
                                message: reason.into(),
                            });
                        }
                        // Reader gone: the peer disconnected mid-session.
                        None => {
                            cancellation_requested = true;
                            self.mark_cancellation();
                        }
                    }
                }
                chunk = chunks_rx.recv() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            if let Err(failure) = self.emit_audio_chunk(
                                &chunk,
                                max_frame_bytes,
                                session_budget,
                                &mut audio_sequence,
                                &mut audio_bytes_out,
                                &mut emitted_final,
                            ) {
                                match failure {
                                    StageTerminal::Disconnected => {
                                        cancellation_requested = true;
                                        self.mark_cancellation();
                                    }
                                    terminal_failure => terminal = Some(terminal_failure),
                                }
                            }
                        }
                        Some(Err(error)) => {
                            let failure = map_execution_error(error);
                            terminal = Some(StageTerminal::Failed {
                                code: failure.code,
                                message: failure.message,
                            });
                        }
                        // The forwarder only exits after the synthesis stream
                        // ended; guarantee the final-flagged frame the contract
                        // promises whenever audio was emitted.
                        None => {
                            if !emitted_final && audio_sequence > 0 {
                                let failure = self.emit_audio_frame(
                                    &[],
                                    true,
                                    max_frame_bytes,
                                    session_budget,
                                    &mut audio_sequence,
                                    &mut audio_bytes_out,
                                    &mut emitted_final,
                                );
                                if let Err(StageTerminal::Disconnected) = failure {
                                    cancellation_requested = true;
                                    self.mark_cancellation();
                                } else if let Err(terminal_failure) = failure {
                                    terminal = Some(terminal_failure);
                                }
                            }
                            if terminal.is_none() {
                                terminal = Some(StageTerminal::Completed { text: None });
                            }
                        }
                    }
                }
            }
        }

        // Teardown confirmation for the TTS stage: dropping the receiver makes
        // the forwarder's next send fail, which drops the synthesis stream and
        // releases the runtime's request leases. The abort backstop covers a
        // generation wedged outside a channel send.
        drop(chunks_rx);
        if let Some(synthesis) = synthesis.take() {
            if tokio::time::timeout(TERMINATION_JOIN_TIMEOUT, synthesis)
                .await
                .is_err()
            {
                self.state
                    .metrics
                    .inner
                    .unconfirmed_teardown
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if timed_out {
            self.terminate_failed(
                InvocationErrorCode::DeadlineExceeded,
                "session deadline elapsed; execution teardown is confirmed",
            );
            return;
        }
        if cancellation_requested {
            self.terminate_cancelled();
            return;
        }
        match terminal.expect("loop exit carries a terminal outcome") {
            StageTerminal::Completed { .. } => {
                self.publish_terminal(
                    AttemptState::Completed,
                    InvocationEventKind::Completed {
                        finish_reason: izwi_serving_protocol::FinishReason::Stop,
                        usage: None,
                    },
                );
            }
            StageTerminal::Failed { code, message } => {
                self.terminate_failed(code, &message);
            }
            StageTerminal::Disconnected => {
                self.terminate_cancelled();
            }
        }
    }

    /// Converts one synthesized chunk into bounded IRTA binary frames. The
    /// protocol's per-frame cap may split a chunk across several frames; only
    /// the terminal frame carries the final flag. The gateway is expected to
    /// drain audio continuously — a full outbound queue is treated as a lost
    /// peer so the cancellation ladder stays responsive.
    #[allow(clippy::too_many_arguments)]
    fn emit_audio_chunk(
        &mut self,
        chunk: &izwi_core::AudioChunk,
        max_frame_bytes: usize,
        session_budget: u64,
        audio_sequence: &mut u32,
        audio_bytes_out: &mut u64,
        emitted_final: &mut bool,
    ) -> Result<(), StageTerminal> {
        if chunk.samples.is_empty() && !chunk.is_final {
            return Ok(());
        }
        let mut payload = Vec::with_capacity(chunk.samples.len() * 2);
        for &sample in &chunk.samples {
            let quantized = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
            payload.extend_from_slice(&quantized.to_le_bytes());
        }
        let pieces: Vec<&[u8]> = if payload.is_empty() {
            vec![&payload]
        } else {
            payload.chunks(max_frame_bytes.max(1)).collect()
        };
        let last = pieces.len().saturating_sub(1);
        for (index, piece) in pieces.iter().enumerate() {
            let is_final = chunk.is_final && index == last;
            self.emit_audio_frame(
                piece,
                is_final,
                max_frame_bytes,
                session_budget,
                audio_sequence,
                audio_bytes_out,
                emitted_final,
            )?;
        }
        Ok(())
    }

    /// Emits one bounded audio frame on the session's outbound audio sequence.
    #[allow(clippy::too_many_arguments)]
    fn emit_audio_frame(
        &mut self,
        payload: &[u8],
        is_final: bool,
        _max_frame_bytes: usize,
        session_budget: u64,
        audio_sequence: &mut u32,
        audio_bytes_out: &mut u64,
        emitted_final: &mut bool,
    ) -> Result<(), StageTerminal> {
        *audio_sequence = audio_sequence.saturating_add(1);
        *audio_bytes_out = audio_bytes_out.saturating_add(payload.len() as u64);
        if *audio_bytes_out > session_budget {
            return Err(StageTerminal::Failed {
                code: InvocationErrorCode::OutputLimitExceeded,
                message: "session audio output budget exhausted".into(),
            });
        }
        let frame = encode_realtime_audio_frame(*audio_sequence, is_final, payload)
            .expect("worker frames respect the protocol caps");
        match self.outbound.try_send(OutboundMessage::Binary(frame)) {
            Ok(()) => {
                if is_final {
                    *emitted_final = true;
                }
                Ok(())
            }
            Err(_) => Err(StageTerminal::Disconnected),
        }
    }

    fn terminate_cancelled(&mut self) {
        self.publish_terminal(
            AttemptState::Cancelled,
            InvocationEventKind::Cancelled {
                reason: Some("requested".into()),
            },
        );
    }

    fn terminate_failed(&mut self, code: InvocationErrorCode, message: &str) {
        self.publish_terminal(
            AttemptState::Failed,
            InvocationEventKind::Error {
                code,
                message: message.to_string(),
            },
        );
    }

    fn publish_terminal(&mut self, attempt_state: AttemptState, event: InvocationEventKind) {
        match attempt_state {
            AttemptState::Completed => self
                .state
                .metrics
                .inner
                .completed
                .fetch_add(1, Ordering::Relaxed),
            AttemptState::Cancelled => self
                .state
                .metrics
                .inner
                .cancelled
                .fetch_add(1, Ordering::Relaxed),
            _ => self
                .state
                .metrics
                .inner
                .failed
                .fetch_add(1, Ordering::Relaxed),
        };
        let published = matches!(self.emit_event(event), EventSendResult::Sent);
        self.state.update_attempt(
            &self.admit.attempt_id,
            attempt_state,
            published.then_some(self.sequence.saturating_sub(1)),
        );
    }
}

enum StageTerminal {
    Completed {
        text: Option<String>,
    },
    Failed {
        code: InvocationErrorCode,
        message: String,
    },
    Disconnected,
}

/// Bridges the synthesis stream into the session loop so audio chunks can be
/// selected alongside control frames. The forwarder ends when the stream ends
/// or the session drops its receiver; the stream drop releases the runtime's
/// request leases in both cases.
async fn forward_synthesis(
    mut stream: Box<dyn RealtimeTtsStageStream>,
    chunks_tx: mpsc::Sender<Result<izwi_core::AudioChunk, izwi_core::Error>>,
) {
    loop {
        match stream.next_chunk().await {
            Ok(Some(chunk)) => {
                if chunks_tx.send(Ok(chunk)).await.is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(error) => {
                let _ = chunks_tx.send(Err(error)).await;
                break;
            }
        }
    }
}

fn language_arg(admit: &RealtimeSessionAdmit) -> Option<&str> {
    match &admit.input {
        izwi_serving_protocol::RealtimeStageInput::AudioStream { language, .. } => {
            language.as_deref()
        }
        _ => None,
    }
}

/// Ordered session admission: receive and validate the admit frame, apply the
/// same fencing checks as the HTTP invoke path, reserve in the shared attempt
/// table, then take the atomic admission gate and one capacity permit.
async fn admit_session<E: InvocationExecutor>(
    state: &Arc<WorkerState<E>>,
    source: &mut SplitStream<WebSocket>,
    outbound_tx: &mpsc::Sender<OutboundMessage>,
    limits: RuntimeRealtimeSessionLimits,
    output_audio: Option<RealtimeAudioSpec>,
) -> Result<RealtimeSession<E>, AdmitFailure> {
    let admit_deadline = tokio::time::sleep(limits.admit_timeout);
    tokio::pin!(admit_deadline);
    let first = tokio::select! {
        () = &mut admit_deadline => return Err(AdmitFailure::close(RealtimeSessionCloseCode::AdmitTimeout)),
        message = source.next() => match message {
            Some(Ok(message)) => message,
            _ => return Err(AdmitFailure::close(RealtimeSessionCloseCode::ProtocolViolation)),
        }
    };
    let Message::Text(text) = first else {
        return Err(AdmitFailure::close(
            RealtimeSessionCloseCode::ProtocolViolation,
        ));
    };
    if text.len() > REALTIME_CONTROL_FRAME_BYTES {
        return Err(AdmitFailure::close(
            RealtimeSessionCloseCode::ProtocolViolation,
        ));
    }
    let admit = match serde_json::from_str::<RealtimeClientFrame>(&text) {
        Ok(RealtimeClientFrame::Admit { admit }) => *admit,
        _ => {
            return Err(AdmitFailure::close(
                RealtimeSessionCloseCode::ProtocolViolation,
            ))
        }
    };
    if admit.validate().is_err() {
        return Err(AdmitFailure::close(
            RealtimeSessionCloseCode::ProtocolViolation,
        ));
    }
    macro_rules! reject {
        ($close:expr) => {{
            state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(AdmitFailure::close($close));
        }};
    }
    if !admit
        .caller
        .permitted_actions
        .contains(&PermittedAction::Invoke)
    {
        reject!(RealtimeSessionCloseCode::PolicyDenied);
    }
    if state.draining.load(Ordering::Acquire) {
        reject!(RealtimeSessionCloseCode::Draining);
    }
    if admit.expected_worker_incarnation != state.config.descriptor.incarnation_id {
        reject!(RealtimeSessionCloseCode::WrongWorkerIncarnation);
    }
    if admit.deployment_id != state.config.deployment.deployment_id {
        reject!(RealtimeSessionCloseCode::UnknownDeployment);
    }
    if admit.expected_model_generation != state.config.deployment.model_generation {
        reject!(RealtimeSessionCloseCode::WrongModelGeneration);
    }
    if admit.task != state.config.deployment.task {
        reject!(RealtimeSessionCloseCode::IncompatibleTask);
    }
    if state.config.deployment.readiness != ModelReadiness::Ready {
        reject!(RealtimeSessionCloseCode::ModelNotReady);
    }
    match state.reserve_realtime_session(&admit) {
        super::ReserveAttempt::Reserved => {}
        super::ReserveAttempt::AlreadyOwned | super::ReserveAttempt::Conflict => {
            reject!(RealtimeSessionCloseCode::DuplicateAttempt)
        }
        super::ReserveAttempt::Full => reject!(RealtimeSessionCloseCode::CapacityExhausted),
    }
    // Serialize accept-work versus begin-draining exactly like the HTTP path.
    let admission_guard = Arc::clone(&state.admission_gate).read_owned().await;
    if state.draining.load(Ordering::Acquire) {
        state.remove_reservation(&admit.attempt_id);
        drop(admission_guard);
        state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
        return Err(AdmitFailure::close(RealtimeSessionCloseCode::Draining));
    }
    let permit = match Arc::clone(&state.capacity).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            state.remove_reservation(&admit.attempt_id);
            drop(admission_guard);
            state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(AdmitFailure::close(
                RealtimeSessionCloseCode::CapacityExhausted,
            ));
        }
    };
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let cancel_was_requested = state.install_admission(&admit.attempt_id, cancel_tx.clone());
    if cancel_was_requested {
        let _ = cancel_tx.send(true);
    }
    state.metrics.inner.admitted.fetch_add(1, Ordering::Relaxed);
    drop(admission_guard);

    // Announce the negotiated session contract before any event flows.
    let bounds = RealtimeSessionBounds {
        max_frame_bytes: limits.max_frame_bytes,
        max_in_flight_frames: limits.max_in_flight_frames,
        max_session_audio_bytes: limits.max_session_audio_bytes,
    };
    let admitted = RealtimeServerFrame::Admitted {
        session_id: admit.session_id.clone(),
        attempt_id: admit.attempt_id.clone(),
        worker_id: state.config.descriptor.worker_id.clone(),
        node_id: state.config.descriptor.node_id.clone(),
        incarnation_id: state.config.descriptor.incarnation_id.clone(),
        deployment_id: state.config.deployment.deployment_id.clone(),
        model_generation: state.config.deployment.model_generation,
        output_audio,
        bounds,
    };
    let _ = outbound_tx
        .send(OutboundMessage::Text(
            serde_json::to_string(&admitted).expect("admitted frame encodes"),
        ))
        .await;

    let remaining_time = Duration::from_millis(admit.remaining_time_ms);
    Ok(RealtimeSession {
        state: Arc::clone(state),
        admit,
        permit,
        cancel: cancel_rx,
        remaining_time,
        started_at: Instant::now(),
        sequence: 0,
        outbound: outbound_tx.clone(),
        cancellation_started_at: None,
    })
}

/// Stable content digest for a realtime attempt reservation, derived from the
/// exact owner fields so attempt-id reuse with different content conflicts.
pub(crate) fn realtime_attempt_digest(admit: &RealtimeSessionAdmit) -> RequestDigest {
    fn mix(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    mix(&mut hash, admit.session_id.as_str().as_bytes());
    mix(&mut hash, admit.request_id.as_str().as_bytes());
    mix(&mut hash, admit.attempt_id.as_str().as_bytes());
    mix(
        &mut hash,
        admit.expected_worker_incarnation.as_str().as_bytes(),
    );
    mix(&mut hash, admit.deployment_id.as_str().as_bytes());
    mix(
        &mut hash,
        &admit.expected_model_generation.get().to_le_bytes(),
    );
    mix(&mut hash, format!("{:?}", admit.task).as_bytes());
    RequestDigest::new(format!("rt-{hash:016x}")).expect("bounded realtime digest")
}

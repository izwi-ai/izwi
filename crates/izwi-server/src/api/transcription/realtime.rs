//! Realtime transcription websocket endpoint for `/transcription`.

use std::time::{Duration, Instant};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, State,
    },
    response::Response,
    routing::get,
    Router,
};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use izwi_core::audio::{AudioEncoder, AudioFormat};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use crate::api::request_context::RequestContext;
use crate::state::AppState;

const REALTIME_PROTOCOL: &str = "transcription_realtime_v2";

const WS_BIN_MAGIC: &[u8; 4] = b"ITRW";
const WS_BIN_VERSION: u8 = 1;
const WS_BIN_KIND_CLIENT_PCM16: u8 = 1;
const WS_BIN_CLIENT_HEADER_LEN: usize = 16;

const MAX_FRAME_BYTES: usize = 512 * 1024;
const MAX_STREAM_BUFFER_SECS: f32 = 32.0;
const INFERENCE_WINDOW_SECS: f32 = 14.0;
const INFERENCE_MIN_INTERVAL_MS: u64 = 350;
const MIN_INFERENCE_AUDIO_MS: u32 = 180;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        language: Option<String>,
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
    },
    AudioFrame {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
    SessionStop,
    Shutdown,
}

struct PendingInference {
    sequence: u64,
    started_at: Instant,
    receiver: oneshot::Receiver<Result<InferenceResult, String>>,
    task: tokio::task::JoinHandle<()>,
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
        }
    }
}

struct InferenceResult {
    text: String,
    language: Option<String>,
    duration_secs: f32,
}

pub fn router() -> Router<AppState> {
    Router::new().route("/transcription/realtime/ws", get(ws_upgrade))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
) -> Response {
    let correlation_id = ctx.correlation_id;
    ws.on_upgrade(move |socket| handle_socket(socket, state, correlation_id))
}

async fn handle_socket(socket: WebSocket, state: AppState, correlation_id: String) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Message>();
    let (worker_tx, worker_rx) = mpsc::unbounded_channel::<WorkerCommand>();

    let writer = tokio::spawn(async move {
        while let Some(message) = out_rx.recv().await {
            if ws_tx.send(message).await.is_err() {
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
            "protocol": REALTIME_PROTOCOL,
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
                    if worker_tx
                        .send(WorkerCommand::AudioFrame {
                            frame_seq,
                            sample_rate,
                            payload,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                Err(err) => {
                    send_json(
                        &out_tx,
                        json!({
                            "type": "error",
                            "message": err,
                        }),
                    );
                }
            },
            Message::Ping(payload) => {
                let _ = out_tx.send(Message::Pong(payload));
            }
            Message::Close(_) => break,
            Message::Pong(_) => {}
        }
    }

    let _ = worker_tx.send(WorkerCommand::Shutdown);
    let _ = worker.await;

    drop(out_tx);
    let _ = writer.await;
}

fn handle_text_message(
    out_tx: &mpsc::UnboundedSender<Message>,
    worker_tx: &mpsc::UnboundedSender<WorkerCommand>,
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
        ClientEvent::SessionStart { model_id, language } => {
            if worker_tx
                .send(WorkerCommand::SessionStart { model_id, language })
                .is_err()
            {
                return true;
            }
            false
        }
        ClientEvent::SessionStop => {
            let _ = worker_tx.send(WorkerCommand::SessionStop);
            true
        }
        ClientEvent::Ping { timestamp_ms } => {
            send_json(
                out_tx,
                json!({
                    "type": "pong",
                    "timestamp_ms": timestamp_ms,
                }),
            );
            false
        }
    }
}

async fn run_worker(
    state: AppState,
    correlation_id: String,
    out_tx: mpsc::UnboundedSender<Message>,
    mut worker_rx: mpsc::UnboundedReceiver<WorkerCommand>,
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
                if handle_worker_command(&state, &correlation_id, &out_tx, &mut session, command)
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
                    send_json(&out_tx, json!({ "type": "error", "message": err }));
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
                    send_json(&out_tx, json!({ "type": "error", "message": err }));
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
    out_tx: &mpsc::UnboundedSender<Message>,
    session: &mut RealtimeSessionState,
    command: WorkerCommand,
) -> bool {
    match command {
        WorkerCommand::SessionStart { model_id, language } => {
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

            send_json(out_tx, json!({ "type": "session_started" }));
            false
        }
        WorkerCommand::AudioFrame {
            frame_seq,
            sample_rate,
            payload,
        } => {
            if let Err(err) = ingest_audio_frame(session, frame_seq, sample_rate, &payload) {
                send_json(out_tx, json!({ "type": "error", "message": err }));
                return false;
            }

            if let Err(err) = maybe_schedule_inference(state, correlation_id, session, false) {
                send_json(out_tx, json!({ "type": "error", "message": err }));
            }
            false
        }
        WorkerCommand::SessionStop => {
            send_json(out_tx, json!({ "type": "session_done" }));
            true
        }
        WorkerCommand::Shutdown => true,
    }
}

fn ingest_audio_frame(
    session: &mut RealtimeSessionState,
    frame_seq: u32,
    sample_rate: u32,
    payload: &[u8],
) -> Result<(), String> {
    if !session.started {
        return Err("session_start is required before streaming audio".to_string());
    }
    if payload.is_empty() {
        return Ok(());
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
            return Ok(());
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
        return Ok(());
    }

    session.samples_i16.extend_from_slice(&samples);
    session.pending_recompute = true;

    let max_samples = ((sample_rate as f32) * MAX_STREAM_BUFFER_SECS) as usize;
    if session.samples_i16.len() > max_samples {
        let drain = session.samples_i16.len() - max_samples;
        session.samples_i16.drain(0..drain);
    }

    Ok(())
}

fn maybe_schedule_inference(
    state: &AppState,
    correlation_id: &str,
    session: &mut RealtimeSessionState,
    force_interval_check: bool,
) -> Result<(), String> {
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
        receiver: rx,
        task,
    });

    Ok(())
}

fn handle_inference_result(
    out_tx: &mpsc::UnboundedSender<Message>,
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

            send_json(
                out_tx,
                json!({
                    "type": "transcript_partial",
                    "sequence": sequence,
                    "text": merged_text,
                    "language": output.language,
                    "audio_duration_secs": output.duration_secs,
                    "processing_time_ms": processing_time_ms,
                    "rtf": rtf,
                }),
            );
            session.last_emitted_sequence = sequence;
        }
        Ok(Err(err)) => {
            send_json(
                out_tx,
                json!({
                    "type": "error",
                    "message": format!("ASR failed: {err}"),
                }),
            );
        }
        Err(err) => {
            send_json(
                out_tx,
                json!({
                    "type": "error",
                    "message": format!("Realtime inference task failed: {err}"),
                }),
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
    let audio_base64 = base64::engine::general_purpose::STANDARD.encode(wav_bytes);

    let _permit = state.acquire_permit().await;
    let output = state
        .runtime
        .asr_transcribe(
            audio_base64.as_str(),
            model_id.as_deref(),
            language.as_deref(),
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
    })
}

fn merge_online_transcript(
    committed: &mut String,
    trailing: &mut String,
    candidate: &str,
) -> String {
    let candidate = candidate.trim_start();
    if candidate.is_empty() {
        return format!("{committed}{trailing}");
    }

    if trailing.is_empty() {
        trailing.clear();
        trailing.push_str(candidate);
        return format!("{committed}{trailing}");
    }

    let shared_prefix_len = longest_common_prefix_bytes(trailing, candidate);
    let commit_len = stable_commit_boundary(&trailing[..shared_prefix_len]);

    if commit_len > 0 {
        committed.push_str(&trailing[..commit_len]);
        trailing.clear();
        trailing.push_str(&candidate[commit_len..]);
    } else {
        trailing.clear();
        trailing.push_str(candidate);
    }

    format!("{committed}{trailing}")
}

fn longest_common_prefix_bytes(a: &str, b: &str) -> usize {
    let mut a_iter = a.char_indices();
    let mut b_iter = b.char_indices();
    let mut end = 0usize;

    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some((a_idx, a_char)), Some((b_idx, b_char))) => {
                if a_char != b_char || a_idx != b_idx {
                    break;
                }
                end = a_idx + a_char.len_utf8();
            }
            _ => break,
        }
    }

    end
}

fn stable_commit_boundary(prefix: &str) -> usize {
    let mut boundary = 0usize;
    for (idx, ch) in prefix.char_indices() {
        if ch.is_whitespace()
            || matches!(
                ch,
                '.' | ',' | '!' | '?' | ';' | ':' | '，' | '。' | '！' | '？' | '、'
            )
        {
            boundary = idx + ch.len_utf8();
        }
    }
    boundary
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

fn wav_bytes_from_pcm16_mono(samples_i16: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    if sample_rate == 0 {
        return Err("Invalid sample rate 0".to_string());
    }
    let samples_f32: Vec<f32> = samples_i16.iter().map(|s| *s as f32 / 32768.0).collect();
    AudioEncoder::new(sample_rate, 1)
        .encode(&samples_f32, AudioFormat::Wav)
        .map_err(|err| format!("Failed to encode streamed WAV: {err}"))
}

fn send_json(tx: &mpsc::UnboundedSender<Message>, value: serde_json::Value) {
    let _ = tx.send(Message::Text(value.to_string().into()));
}

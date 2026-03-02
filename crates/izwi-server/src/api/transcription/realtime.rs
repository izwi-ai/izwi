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

    collapsed
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

#[cfg(test)]
mod tests {
    use super::{
        collapse_unstable_repetition, merge_online_transcript,
        strip_suffix_prefix_overlap_with_lookahead_by_words,
    };

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
}

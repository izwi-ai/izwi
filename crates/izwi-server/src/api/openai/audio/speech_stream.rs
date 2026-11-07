//! `GET /v1/audio/speech/stream`: a text-in, audio-out WebSocket.
//!
//! A client configures a session once (`session.config`), then sends text as
//! it has it (`input.text`, e.g. an LLM's tokens). Text is cut into
//! segments by the voice loop's segmenter (the first sentence alone, then
//! complete sentences packed up to 240 characters, abbreviation-aware), and
//! each segment is synthesized while the one before it plays. Every segment
//! goes out as `audio.start`, its audio (binary frames, or base64
//! `audio.chunk` events), then `audio.done`. `input.commit` speaks what is
//! buffered now; `input.done` ends the utterance (`session.done`);
//! `input.cancel` drops it; `session.close` closes.
//!
//! It serves SGLang-Omni and vLLM-Omni clients too. Their protocols differ
//! on what `input.done` means, so `?dialect=sglang-omni` closes the socket
//! after `session.done` (SGLang-Omni) while `izwi` and `vllm-omni` keep it
//! open for the next utterance. `session.config` is accepted nested
//! (`{"type":"session.config","session":{...}}`) or flat, and may be resent
//! between utterances.
//!
//! Admission: a session takes a realtime session slot
//! (`IZWI_MAX_REALTIME_SESSIONS`); each segment takes a stage permit, and at
//! most two segments are in flight. A reader slower than the audio pauses
//! synthesis (the socket send is awaited); one stalled past
//! `IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS` is closed. Buffered text is capped
//! (`IZWI_SPEECH_STREAM_MAX_PENDING_CHARS`), as is a session's total text
//! (1 MiB) and one message (128 KiB); an idle session closes after
//! `IZWI_SPEECH_STREAM_IDLE_TIMEOUT_SECS`.

use std::collections::VecDeque;
use std::time::Duration;

use axum::{
    extract::{
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
        Extension, Query, State,
    },
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tracing::debug;

use super::speech::{
    build_generation_request, codec_api_error, validate_speech, SpeechRequest, ValidatedSpeech,
};
use crate::api::request_context::RequestContext;
use crate::app::voice_realtime::{acquire_stage_permit, AbortOnDropTask};
use crate::app::voice_reply::{SegmentJoiner, SpeechSegmenter};
use crate::error::ApiError;
use crate::speech_outcomes::{new_speech_id, SpeechCompletion, SpeechMode, SpeechUsage};
use crate::state::AppState;
use izwi_audio_codec::{Framing, StreamEncoder};
use izwi_core::{AudioChunk, ModelVariant, SessionModelLeases};

/// One message's size cap (a config message carries inline reference audio).
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
/// `input.text` cap.
const MAX_TEXT_MESSAGE_BYTES: usize = 128 * 1024;
/// Text one session may send in all.
const MAX_SESSION_TEXT_BYTES: usize = 1024 * 1024;
/// How long the first `session.config` may take.
const CONFIG_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dialect {
    Izwi,
    SglangOmni,
    VllmOmni,
}

impl Dialect {
    fn parse(value: Option<&str>) -> Result<Self, ApiError> {
        match value
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref()
        {
            None | Some("" | "izwi") => Ok(Self::Izwi),
            Some("sglang-omni" | "sglang") => Ok(Self::SglangOmni),
            Some("vllm-omni" | "vllm") => Ok(Self::VllmOmni),
            Some(other) => Err(ApiError::invalid_param(
                "dialect",
                format!("Unknown dialect `{other}`; use izwi, sglang-omni or vllm-omni"),
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Izwi => "izwi",
            Self::SglangOmni => "sglang-omni",
            Self::VllmOmni => "vllm-omni",
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct SpeechStreamQuery {
    #[serde(default)]
    dialect: Option<String>,
}

pub async fn speech_stream(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Query(query): Query<SpeechStreamQuery>,
) -> Response {
    let dialect = match Dialect::parse(query.dialect.as_deref()) {
        Ok(dialect) => dialect,
        Err(error) => return error.into_response(),
    };
    let Some(slot) = state.try_acquire_realtime_session() else {
        return ApiError::of(
            crate::error::ErrorClass::Unavailable,
            "Every realtime session slot is in use (IZWI_MAX_REALTIME_SESSIONS); try again shortly",
        )
        .into_response();
    };
    let tenant_key = ctx.tenant_key();
    let correlation_id = ctx.correlation_id;
    ws.max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| async move {
            let _slot = slot;
            let session = Session::new(state, socket, dialect, tenant_key, correlation_id);
            session.run().await;
        })
}

/// How text is cut into spoken segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Granularity {
    /// Complete sentences, the first alone (the default).
    Sentence,
    /// Sentences, further cut at clause breaks.
    Clause,
    /// Only `input.commit` and `input.done` cut.
    None,
}

/// A configured session.
struct StreamConfig {
    speech: ValidatedSpeech,
    stream_audio: bool,
    base64: bool,
    close_on_done: bool,
    granularity: Granularity,
}

/// One segment's synthesis, running ahead of its playback.
struct Segment {
    index: usize,
    text: String,
    chunks: mpsc::Receiver<AudioChunk>,
    task: AbortOnDropTask<Result<(), String>>,
    /// Set at its first audio.
    encoder: Option<Box<dyn StreamEncoder>>,
    /// stream_audio=false: the segment's PCM, encoded whole at the end.
    pcm: Vec<f32>,
    source_rate: Option<u32>,
    total_bytes: usize,
}

struct Session {
    state: AppState,
    socket: WebSocket,
    dialect: Dialect,
    tenant_key: Option<[u8; 32]>,
    correlation_id: String,
    session_id: String,
    config: Option<StreamConfig>,
    leases: SessionModelLeases,
    segmenter: SpeechSegmenter,
    joiner: SegmentJoiner,
    ready: VecDeque<String>,
    playing: Option<Segment>,
    lookahead: Option<Segment>,
    /// Text received this utterance (for the outcome) and this session.
    utterance_text: String,
    /// Characters of this utterance already handed out as segments.
    utterance_segment_chars: usize,
    session_text_bytes: usize,
    next_segment: usize,
    utterance_index: usize,
    utterance_segments: usize,
    utterance_speech_id: Option<String>,
    utterance_samples: usize,
    /// Model tokens the utterance's finished segments generated.
    utterance_tokens: usize,
    utterance_rate: u32,
    /// `input.done` (or `session.close`) arrived: finish, then end it.
    finishing: bool,
    close_after: bool,
    closed: bool,
}

fn error_event(message: impl Into<String>, error_type: &str, code: &str, fatal: bool) -> Value {
    json!({
        "type": "error",
        "message": message.into(),
        "error_type": error_type,
        "param": Value::Null,
        "code": code,
        "fatal": fatal,
    })
}

impl Session {
    fn new(
        state: AppState,
        socket: WebSocket,
        dialect: Dialect,
        tenant_key: Option<[u8; 32]>,
        correlation_id: String,
    ) -> Self {
        Self {
            state,
            socket,
            dialect,
            tenant_key,
            correlation_id,
            session_id: format!("sstream_{}", uuid::Uuid::new_v4().simple()),
            config: None,
            leases: SessionModelLeases::default(),
            segmenter: SpeechSegmenter::default(),
            joiner: SegmentJoiner::default(),
            ready: VecDeque::new(),
            playing: None,
            lookahead: None,
            utterance_text: String::new(),
            utterance_segment_chars: 0,
            session_text_bytes: 0,
            next_segment: 0,
            utterance_index: 0,
            utterance_segments: 0,
            utterance_speech_id: None,
            utterance_samples: 0,
            utterance_tokens: 0,
            utterance_rate: 0,
            finishing: false,
            close_after: false,
            closed: false,
        }
    }

    /// Sends one JSON event; a reader stalled past the stall deadline ends
    /// the session.
    async fn send_json(&mut self, value: Value) -> bool {
        self.send(Message::Text(value.to_string().into())).await
    }

    async fn send(&mut self, message: Message) -> bool {
        if self.closed {
            return false;
        }
        let stall = crate::env_config::tuning().audio_stream_stall_timeout;
        match tokio::time::timeout(stall, self.socket.send(message)).await {
            Ok(Ok(())) => true,
            _ => {
                self.closed = true;
                false
            }
        }
    }

    async fn fail(&mut self, message: impl Into<String>, code: &str) {
        let _ = self
            .send_json(error_event(message, "invalid_request_error", code, true))
            .await;
        self.close().await;
    }

    async fn close(&mut self) {
        if !self.closed {
            let _ = self
                .socket
                .send(Message::Close(Some(CloseFrame {
                    code: 1000,
                    reason: "".into(),
                })))
                .await;
        }
        self.closed = true;
    }

    fn idle(&self) -> bool {
        self.playing.is_none()
            && self.lookahead.is_none()
            && self.ready.is_empty()
            && self.utterance_text.is_empty()
    }

    async fn run(mut self) {
        // The first message configures the session.
        match tokio::time::timeout(CONFIG_TIMEOUT, self.socket.recv()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let parsed = serde_json::from_str::<Value>(&text).ok();
                let is_config = parsed
                    .as_ref()
                    .and_then(|value| value.get("type"))
                    .and_then(Value::as_str)
                    == Some("session.config");
                if !is_config {
                    return self
                        .fail(
                            "The first message must be session.config",
                            "missing_session_config",
                        )
                        .await;
                }
                if let Err(error) = self.configure(parsed.expect("parsed")).await {
                    return self.fail(error.message, "invalid_session_config").await;
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) | Ok(Some(Err(_))) => return,
            Ok(Some(Ok(_))) => {
                return self
                    .fail(
                        "The first message must be session.config",
                        "missing_session_config",
                    )
                    .await
            }
            Err(_) => {
                return self
                    .fail(
                        "No session.config within 10 seconds",
                        "session_config_timeout",
                    )
                    .await
            }
        }

        let idle_timeout = crate::env_config::tuning().speech_api.stream_idle_timeout;
        while !self.closed {
            self.advance().await;
            if self.closed {
                break;
            }
            if self.finishing && self.idle_except_text() {
                self.end_utterance(false).await;
                if self.close_after {
                    self.close().await;
                    break;
                }
                continue;
            }
            let idle = self.idle();
            tokio::select! {
                biased;
                message = self.socket.recv() => match message {
                    Some(Ok(message)) => self.on_message(message).await,
                    _ => break,
                },
                chunk = next_segment_chunk(self.playing.as_mut()) => self.on_chunk(chunk).await,
                _ = tokio::time::sleep(idle_timeout), if idle => {
                    let _ = self.send_json(error_event(
                        "The session was idle too long",
                        "timeout",
                        "idle_timeout",
                        true,
                    )).await;
                    self.close().await;
                }
            }
        }
        // Dropping the segments aborts their synthesis.
        if let Some(id) = self.utterance_speech_id.take() {
            self.state.speech_outcomes.cancel(&id);
        }
    }

    /// Nothing is left to say but text the segmenter still holds.
    fn idle_except_text(&self) -> bool {
        self.playing.is_none() && self.lookahead.is_none() && self.ready.is_empty()
    }

    /// Starts synthesis for ready segments (one playing, one ahead).
    async fn advance(&mut self) {
        if self.playing.is_none() {
            self.playing = self.lookahead.take();
        }
        while self.lookahead.is_none() || self.playing.is_none() {
            let Some(text) = self.ready.pop_front() else {
                break;
            };
            let segment = self.start_segment(text);
            if self.playing.is_none() {
                self.playing = Some(segment);
            } else {
                self.lookahead = Some(segment);
            }
        }
    }

    fn start_segment(&mut self, text: String) -> Segment {
        let config = self.config.as_mut().expect("configured");
        let index = self.next_segment;
        self.next_segment += 1;
        self.utterance_segments += 1;
        if self.utterance_speech_id.is_none() {
            let id = new_speech_id();
            self.state.speech_outcomes.begin(
                &id,
                self.tenant_key,
                config.speech.variant.dir_name(),
                SpeechMode::SocketUtterance,
                config.speech.output.label(),
            );
            self.utterance_speech_id = Some(id);
        }
        config.speech.req.input = text.clone();
        let request = build_generation_request(
            &config.speech.req,
            format!("{}-{index}", self.correlation_id),
            true,
            config.speech.variant,
        );
        let (chunk_tx, chunks) = mpsc::channel(32);
        let state = self.state.clone();
        let leases = self.leases.clone();
        let variant: ModelVariant = config.speech.variant;
        let task = AbortOnDropTask::new(tokio::spawn(async move {
            leases
                .hold(&state.runtime, variant)
                .await
                .map_err(|err| format!("Loading {variant} failed: {err}"))?;
            let permit = acquire_stage_permit(&state).await?;
            let mut request = request;
            request.runtime_context = permit.runtime_context();
            state
                .runtime
                .generate_streaming(request, chunk_tx)
                .await
                .map_err(|err| format!("TTS failed: {err}"))
        }));
        Segment {
            index,
            text,
            chunks,
            task,
            encoder: None,
            pcm: Vec::new(),
            source_rate: None,
            total_bytes: 0,
        }
    }

    async fn on_chunk(&mut self, chunk: Option<AudioChunk>) {
        let Some(chunk) = chunk else {
            return self.finish_segment().await;
        };
        // A final chunk's stats hold the segment's totals.
        if let Some(stats) = chunk.stats.as_ref().filter(|_| chunk.is_final) {
            self.utterance_tokens += stats.tokens_generated;
        }
        if chunk.samples.is_empty() {
            return;
        }
        let rate = chunk.sample_rate_or(24_000).max(1);
        if let Err(message) = self.segment_audio(&chunk.samples, rate).await {
            let _ = self
                .send_json(error_event(
                    message,
                    "server_error",
                    "synthesis_failed",
                    false,
                ))
                .await;
            self.cancel_utterance("synthesis_failed").await;
        }
    }

    /// Encodes and sends `samples` of the playing segment.
    async fn segment_audio(&mut self, samples: &[f32], rate: u32) -> Result<(), String> {
        let config = self.config.as_ref().expect("configured");
        let output = config.speech.output.clone();
        let stream_audio = config.stream_audio;
        let base64 = config.base64;
        let joined_first = self
            .playing
            .as_ref()
            .is_some_and(|segment| segment.source_rate.is_none());
        if joined_first {
            self.joiner.begin_segment();
        }
        let joined = self.joiner.push(samples, rate);
        let segment = self.playing.as_mut().expect("playing");
        if segment.source_rate.is_none() {
            segment.source_rate = Some(rate);
            let encoder = output.stream_encoder(rate).map_err(|error| error.message)?;
            let output_rate = encoder.output_rate();
            segment.encoder = Some(encoder);
            self.utterance_rate = output_rate;
            let start = json!({
                "type": "audio.start",
                "id": format!("{}-{}", self.session_id, segment.index),
                "utterance_index": self.utterance_index,
                "sentence_index": segment.index,
                "sentence_text": segment.text,
                "format": output.label(),
                "sample_rate": output_rate,
                "channels": 1,
            });
            if !self.send_json(start).await {
                return Ok(());
            }
        } else if segment.source_rate != Some(rate) {
            return Err(format!(
                "TTS audio changed sample rate mid-segment ({} Hz to {rate} Hz)",
                segment.source_rate.unwrap_or_default()
            ));
        }
        let segment = self.playing.as_mut().expect("playing");
        self.utterance_samples += samples.len();
        if !stream_audio {
            segment.pcm.extend_from_slice(&joined);
            return Ok(());
        }
        let bytes = segment
            .encoder
            .as_mut()
            .expect("encoder")
            .push(&joined)
            .map_err(|err| codec_api_error(err).message)?;
        self.send_audio(bytes, base64).await;
        Ok(())
    }

    async fn send_audio(&mut self, bytes: Vec<u8>, base64: bool) {
        if bytes.is_empty() {
            return;
        }
        let Some(segment) = self.playing.as_mut() else {
            return;
        };
        segment.total_bytes += bytes.len();
        let index = segment.index;
        if base64 {
            let event = json!({
                "type": "audio.chunk",
                "sentence_index": index,
                "utterance_index": self.utterance_index,
                "audio_b64": base64::engine::general_purpose::STANDARD.encode(&bytes),
            });
            self.send_json(event).await;
        } else {
            self.send(Message::Binary(bytes.into())).await;
        }
    }

    /// The playing segment's synthesis ended: flush its encoder and close it.
    async fn finish_segment(&mut self) {
        let Some(mut segment) = self.playing.take() else {
            return;
        };
        let result = std::mem::replace(
            &mut segment.task,
            AbortOnDropTask::new(tokio::spawn(async { Ok(()) })),
        )
        .join()
        .await
        .map_err(|err| err.to_string())
        .and_then(|result| result);
        let config = self.config.as_ref().expect("configured");
        let output = config.speech.output.clone();
        let (stream_audio, base64) = (config.stream_audio, config.base64);
        let mut error = result.err();
        if let Some(rate) = segment.source_rate {
            let tail = self.joiner.end_segment(rate);
            let bytes = if stream_audio {
                let encoder = segment.encoder.as_mut().expect("encoder");
                encoder
                    .push(&tail)
                    .and_then(|mut bytes| {
                        bytes.extend(encoder.finish()?);
                        Ok(bytes)
                    })
                    .map_err(|err| err.to_string())
            } else {
                segment.pcm.extend_from_slice(&tail);
                izwi_audio_codec::encode_file(&output.options(rate, Framing::File), &segment.pcm)
                    .map_err(|err| err.to_string())
            };
            match bytes {
                Ok(bytes) => {
                    self.playing = Some(segment);
                    self.send_audio(bytes, base64).await;
                    segment = self.playing.take().expect("restored");
                }
                Err(message) => error = error.or(Some(message)),
            }
        } else if error.is_none() {
            // A segment that made no audio still opens and closes.
            let start = json!({
                "type": "audio.start",
                "id": format!("{}-{}", self.session_id, segment.index),
                "utterance_index": self.utterance_index,
                "sentence_index": segment.index,
                "sentence_text": segment.text,
                "format": output.label(),
                "sample_rate": self.utterance_rate.max(output.output_rate(24_000)),
                "channels": 1,
            });
            self.send_json(start).await;
        }
        let done = json!({
            "type": "audio.done",
            "id": format!("{}-{}", self.session_id, segment.index),
            "utterance_index": self.utterance_index,
            "sentence_index": segment.index,
            "total_bytes": segment.total_bytes,
            "error": error.is_some(),
            "cancelled": false,
        });
        self.send_json(done).await;
        if let Some(message) = error {
            debug!(session = %self.session_id, "speech stream segment failed: {message}");
            let _ = self
                .send_json(error_event(
                    message,
                    "server_error",
                    "synthesis_failed",
                    false,
                ))
                .await;
        }
    }

    async fn on_message(&mut self, message: Message) {
        let text = match message {
            Message::Text(text) => text,
            Message::Close(_) => {
                self.closed = true;
                return;
            }
            Message::Ping(_) | Message::Pong(_) => return,
            Message::Binary(_) => {
                let _ = self
                    .send_json(error_event(
                        "Send text as JSON messages; binary frames are not accepted",
                        "invalid_request_error",
                        "binary_not_supported",
                        false,
                    ))
                    .await;
                return;
            }
        };
        let event: Value = match serde_json::from_str(&text) {
            Ok(event) => event,
            Err(err) => {
                let _ = self
                    .send_json(error_event(
                        format!("Invalid JSON: {err}"),
                        "invalid_request_error",
                        "invalid_json",
                        false,
                    ))
                    .await;
                return;
            }
        };
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "session.config" => {
                if !self.idle() {
                    let _ = self
                        .send_json(error_event(
                            "session.config may change only between utterances",
                            "invalid_request_error",
                            "utterance_in_progress",
                            false,
                        ))
                        .await;
                    return;
                }
                if let Err(error) = self.configure(event).await {
                    let _ = self
                        .send_json(error_event(
                            error.message,
                            "invalid_request_error",
                            "invalid_session_config",
                            false,
                        ))
                        .await;
                }
            }
            "input.text" => {
                let text = event
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.on_text(text).await;
            }
            "input.commit" => {
                self.flush();
                let committed = json!({
                    "type": "input.committed",
                    "session_id": self.session_id,
                    "utterance_index": self.utterance_index,
                    "segment_index": self.next_segment + self.ready.len(),
                    "total_sentences": self.utterance_segments + self.ready.len(),
                });
                self.send_json(committed).await;
            }
            "input.done" => {
                self.flush();
                self.finishing = true;
                self.close_after = self
                    .config
                    .as_ref()
                    .is_some_and(|config| config.close_on_done);
            }
            "input.cancel" => self.cancel_utterance("cancelled").await,
            "session.close" => {
                self.flush();
                self.finishing = true;
                self.close_after = true;
            }
            "ping" => {
                self.send_json(json!({"type": "pong"})).await;
            }
            other => {
                let _ = self
                    .send_json(error_event(
                        format!("Unknown message type `{other}`"),
                        "invalid_request_error",
                        "unknown_type",
                        false,
                    ))
                    .await;
            }
        }
    }

    async fn on_text(&mut self, text: &str) {
        if text.len() > MAX_TEXT_MESSAGE_BYTES {
            let _ = self
                .send_json(error_event(
                    "input.text is larger than 128 KiB",
                    "invalid_request_error",
                    "message_too_large",
                    false,
                ))
                .await;
            return;
        }
        let max_pending = crate::env_config::tuning()
            .speech_api
            .stream_max_pending_chars;
        let pending: usize = self
            .ready
            .iter()
            .map(|segment| segment.chars().count())
            .sum::<usize>()
            + self.segmenter_pending_chars();
        if pending + text.chars().count() > max_pending {
            let _ = self
                .send_json(error_event(
                    format!("More than {max_pending} characters are waiting to be spoken; wait for audio.done before sending more"),
                    "invalid_request_error",
                    "input_buffer_full",
                    false,
                ))
                .await;
            return;
        }
        if self.session_text_bytes + text.len() > MAX_SESSION_TEXT_BYTES {
            return self
                .fail(
                    "The session has sent 1 MiB of text; open a new one",
                    "session_text_budget",
                )
                .await;
        }
        self.session_text_bytes += text.len();
        self.utterance_text.push_str(text);
        self.segmenter.push(text);
        if self
            .config
            .as_ref()
            .is_some_and(|config| config.granularity != Granularity::None)
        {
            self.take_segments(false);
        }
    }

    /// Text the segmenter holds that is not yet in a segment.
    fn segmenter_pending_chars(&self) -> usize {
        self.utterance_text
            .chars()
            .count()
            .saturating_sub(self.utterance_segment_chars)
    }

    /// Moves ready segments out of the segmenter; `flush` takes the rest.
    fn take_segments(&mut self, flush: bool) {
        let clause = self
            .config
            .as_ref()
            .is_some_and(|config| config.granularity == Granularity::Clause);
        while let Some(segment) = self.segmenter.next_segment(flush) {
            self.utterance_segment_chars += segment.chars().count();
            if clause {
                self.ready.extend(clause_pieces(&segment));
            } else {
                self.ready.push_back(segment);
            }
        }
    }

    fn flush(&mut self) {
        self.take_segments(true);
    }

    /// Ends the utterance: `session.done`, its outcome, and a fresh segmenter.
    async fn end_utterance(&mut self, cancelled: bool) {
        let had_text = !self.utterance_text.trim().is_empty() || self.utterance_segments > 0;
        let done = json!({
            "type": "session.done",
            "session_id": self.session_id,
            "utterance_index": self.utterance_index,
            "total_sentences": self.utterance_segments,
            "speech_id": self.utterance_speech_id,
            "cancelled": cancelled,
        });
        if let Some(id) = self.utterance_speech_id.take() {
            if cancelled {
                self.state.speech_outcomes.cancel(&id);
            } else {
                self.state.speech_outcomes.complete(
                    &id,
                    SpeechCompletion {
                        sample_rate: Some(self.utterance_rate),
                        audio_duration_secs: (self.utterance_rate > 0)
                            .then(|| self.utterance_samples as f32 / self.utterance_rate as f32),
                        finish_reason: Some("stop"),
                        usage: Some(SpeechUsage::new(0, self.utterance_tokens)),
                        timing: None,
                    },
                );
            }
        }
        if had_text || cancelled {
            self.send_json(done).await;
            self.utterance_index += 1;
        }
        self.segmenter = SpeechSegmenter::default();
        self.joiner = SegmentJoiner::default();
        self.utterance_text.clear();
        self.utterance_segment_chars = 0;
        self.utterance_segments = 0;
        self.utterance_samples = 0;
        self.utterance_tokens = 0;
        self.finishing = false;
    }

    /// `input.cancel`: drops buffered text and the segments in flight.
    async fn cancel_utterance(&mut self, reason: &str) {
        self.ready.clear();
        self.lookahead = None;
        if let Some(segment) = self.playing.take() {
            let done = json!({
                "type": "audio.done",
                "id": format!("{}-{}", self.session_id, segment.index),
                "utterance_index": self.utterance_index,
                "sentence_index": segment.index,
                "total_bytes": segment.total_bytes,
                "error": reason != "cancelled",
                "cancelled": true,
            });
            drop(segment);
            self.send_json(done).await;
        }
        self.end_utterance(true).await;
    }

    async fn configure(&mut self, event: Value) -> Result<(), ApiError> {
        let fields = match event.get("session") {
            Some(Value::Object(session)) => session.clone(),
            _ => {
                let Value::Object(mut flat) = event else {
                    return Err(ApiError::bad_request("session.config must be an object"));
                };
                flat.remove("type");
                flat
            }
        };
        let config = parse_stream_config(&self.state, fields, self.dialect).await?;
        // Warm the model once, so the first segment does not pay for it.
        self.leases
            .hold(&self.state.runtime, config.speech.variant)
            .await
            .map_err(crate::error::core_api_error)?;
        let configured = json!({
            "type": "session.configured",
            "session_id": self.session_id,
            "dialect": self.dialect.as_str(),
            "model": config.speech.variant.dir_name(),
            "response_format": config.speech.output.label(),
            "sample_rate": config
                .speech
                .output
                .sample_rate
                .or(config.speech.output.format.forced_rate),
            "stream_audio": config.stream_audio,
            "split_granularity": match config.granularity {
                Granularity::Sentence => "sentence",
                Granularity::Clause => "clause",
                Granularity::None => "none",
            },
            "audio_transport": if config.base64 { "base64" } else { "binary" },
            "close_on_done": config.close_on_done,
        });
        self.config = Some(config);
        self.send_json(configured).await;
        Ok(())
    }
}

/// The audio of the playing segment, or never.
async fn next_segment_chunk(segment: Option<&mut Segment>) -> Option<AudioChunk> {
    match segment {
        Some(segment) => segment.chunks.recv().await,
        None => std::future::pending().await,
    }
}

/// `segment` cut at clause breaks (`,` `;` `:` before a space), each piece
/// with something to say.
fn clause_pieces(segment: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut chars = segment.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let breaks = matches!(ch, ',' | ';' | ':' | '，' | '；')
            && chars.peek().is_none_or(|(_, next)| next.is_whitespace());
        if breaks {
            let end = index + ch.len_utf8();
            pieces.push(segment[start..end].trim().to_string());
            start = end;
        }
    }
    pieces.push(segment[start..].trim().to_string());
    pieces
        .into_iter()
        .filter(|piece| piece.chars().any(char::is_alphanumeric))
        .collect()
}

fn take_bool(fields: &mut Map<String, Value>, name: &str, default: bool) -> Result<bool, ApiError> {
    match fields.remove(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(value),
        Some(_) => Err(ApiError::invalid_param(
            "session",
            format!("`{name}` must be a boolean"),
        )),
    }
}

/// A session configuration: the speech fields of `/v1/audio/speech` (minus
/// `input`), plus the stream's own.
async fn parse_stream_config(
    state: &AppState,
    mut fields: Map<String, Value>,
    dialect: Dialect,
) -> Result<StreamConfig, ApiError> {
    let stream_audio = take_bool(&mut fields, "stream_audio", true)?;
    let close_on_done = take_bool(&mut fields, "close_on_done", dialect == Dialect::SglangOmni)?;
    let granularity = match fields.remove("split_granularity") {
        None | Some(Value::Null) => Granularity::Sentence,
        Some(Value::String(value)) => match value.as_str() {
            "sentence" => Granularity::Sentence,
            "clause" => Granularity::Clause,
            "none" => Granularity::None,
            other => {
                return Err(ApiError::invalid_param(
                    "split_granularity",
                    format!("Unknown split_granularity `{other}`; use sentence, clause or none"),
                ))
            }
        },
        Some(_) => {
            return Err(ApiError::invalid_param(
                "split_granularity",
                "must be a string",
            ))
        }
    };
    let base64 = match fields.remove("audio_transport") {
        None | Some(Value::Null) => false,
        Some(Value::String(value)) if value == "binary" => false,
        Some(Value::String(value)) if value == "base64" => true,
        Some(_) => {
            return Err(ApiError::invalid_param(
                "audio_transport",
                "audio_transport is `binary` or `base64`",
            ))
        }
    };
    // vLLM-Omni's word-timestamp mode carries audio as JSON.
    let base64 = base64 || take_bool(&mut fields, "word_timestamps", false)?;
    for ignored in ["text_input_mode", "stream", "stream_format", "input"] {
        fields.remove(ignored);
    }
    if !fields.contains_key("response_format") {
        fields.insert("response_format".into(), Value::String("pcm".into()));
    }
    // Validated as a request would be; `input` stands in for the text to come.
    fields.insert("input".into(), Value::String("Hello.".into()));
    let req: SpeechRequest = serde_json::from_value(Value::Object(fields))
        .map_err(|err| ApiError::bad_request(format!("Invalid session.config: {err}")))?;
    let speech = validate_speech(state, req).await?;
    Ok(StreamConfig {
        speech,
        stream_audio,
        base64,
        close_on_done,
        granularity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialects_parse_and_name_themselves() {
        assert_eq!(Dialect::parse(None).unwrap(), Dialect::Izwi);
        assert_eq!(
            Dialect::parse(Some("SGLang-Omni")).unwrap(),
            Dialect::SglangOmni
        );
        assert_eq!(Dialect::parse(Some("vllm")).unwrap().as_str(), "vllm-omni");
        assert!(Dialect::parse(Some("openai")).is_err());
    }

    #[test]
    fn clause_mode_cuts_at_clause_breaks() {
        assert_eq!(
            clause_pieces("First, a pause; then the rest."),
            vec!["First,", "a pause;", "then the rest."]
        );
        assert_eq!(clause_pieces("1,000 apples."), vec!["1,000 apples."]);
        assert_eq!(clause_pieces("Done."), vec!["Done."]);
    }
}

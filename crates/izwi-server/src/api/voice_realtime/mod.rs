//! Realtime voice websocket endpoint for `/voice`.
//!
//! Frontend responsibilities:
//! - microphone capture
//! - simple local VAD (speech start/stop)
//! - audio playback
//!
//! Backend responsibilities:
//! - ASR -> agent -> TTS orchestration
//! - streaming assistant audio/text events
//! - interruption / barge-in cancellation

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use izwi_agent::{
    planner::{PlanningMode, SimplePlanner},
    AgentDefinition, AgentEngine, AgentSession, AgentTurnOptions, MemoryMessage, MemoryMessageMeta,
    MemoryMessageRole, MemoryStore, ModelBackend, ModelOutput, ModelRequest, NoopTool, TimeTool,
    ToolRegistry, TurnInput,
};
use izwi_core::{
    audio::{AudioEncoder, AudioFormat},
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant, ChatMessage, ChatRole,
    GenerationConfig, GenerationRequest,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::api::request_context::RequestContext;
use crate::chat_store::ChatStore;
use crate::state::{AppState, StoredAgentSessionRecord};

const DEFAULT_AGENT_ID: &str = "voice-agent";
const DEFAULT_AGENT_NAME: &str = "Voice Agent";
const DEFAULT_AGENT_SYSTEM_PROMPT: &str =
    "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Do not output <think> tags or internal reasoning. Return only the final spoken answer. Keep responses brief unless asked for details.";
const DEFAULT_CHAT_MODEL: &str = "Qwen3.5-0.8B";
const MAX_UTTERANCE_BYTES: usize = 16 * 1024 * 1024;
const WS_BIN_MAGIC: &[u8; 4] = b"IVWS";
const WS_BIN_VERSION: u8 = 1;
const WS_BIN_KIND_CLIENT_PCM16: u8 = 1;
const WS_BIN_KIND_ASSISTANT_PCM16: u8 = 2;
const WS_BIN_CLIENT_HEADER_LEN: usize = 16;
const WS_BIN_ASSISTANT_HEADER_LEN: usize = 24;
const DEFAULT_STREAM_VAD_THRESHOLD: f32 = 0.02;
const DEFAULT_STREAM_MIN_SPEECH_MS: u32 = 300;
const DEFAULT_STREAM_SILENCE_MS: u32 = 900;
const DEFAULT_STREAM_MAX_UTTERANCE_MS: u32 = 20_000;
const DEFAULT_STREAM_PRE_ROLL_MS: u32 = 160;

pub fn router() -> Router<AppState> {
    Router::new().route("/voice/realtime/ws", get(ws_upgrade))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
) -> Response {
    let correlation_id = ctx.correlation_id;
    ws.on_upgrade(move |socket| handle_socket(socket, state, correlation_id))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        system_prompt: Option<String>,
    },
    InputStreamStart {
        #[serde(default)]
        mode: RealtimeVoiceMode,
        #[serde(default)]
        asr_model_id: Option<String>,
        #[serde(default)]
        text_model_id: Option<String>,
        #[serde(default)]
        tts_model_id: Option<String>,
        #[serde(default)]
        s2s_model_id: Option<String>,
        #[serde(default)]
        speaker: Option<String>,
        #[serde(default)]
        asr_language: Option<String>,
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        max_output_tokens: Option<usize>,
        #[serde(default)]
        vad_threshold: Option<f32>,
        #[serde(default)]
        min_speech_ms: Option<u32>,
        #[serde(default)]
        silence_duration_ms: Option<u32>,
        #[serde(default)]
        max_utterance_ms: Option<u32>,
        #[serde(default)]
        pre_roll_ms: Option<u32>,
        #[serde(default)]
        input_sample_rate: Option<u32>,
    },
    InputStreamStop,
    Interrupt {
        #[serde(default)]
        reason: Option<String>,
    },
    Ping {
        #[serde(default)]
        timestamp_ms: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RealtimeVoiceMode {
    #[default]
    Modular,
    Unified,
}

#[derive(Debug, Clone)]
struct ModularVoiceTurnConfig {
    asr_model_id: String,
    text_model_id: String,
    tts_model_id: String,
    speaker: Option<String>,
    asr_language: Option<String>,
    max_output_tokens: usize,
}

#[derive(Debug, Clone)]
struct UnifiedVoiceTurnConfig {
    s2s_model_id: String,
    language: Option<String>,
    system_prompt: Option<String>,
}

#[derive(Debug, Clone)]
enum VoiceTurnConfig {
    Modular(ModularVoiceTurnConfig),
    Unified(UnifiedVoiceTurnConfig),
}

#[derive(Debug, Clone)]
struct PendingAudioCommit {
    utterance_id: String,
    utterance_seq: u64,
    turn_config: VoiceTurnConfig,
}

#[derive(Debug)]
struct ActiveTurn {
    utterance_id: String,
    utterance_seq: u64,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone)]
struct StreamingInputConfig {
    turn_config: VoiceTurnConfig,
    vad_threshold: f32,
    min_speech_ms: u32,
    silence_duration_ms: u32,
    max_utterance_ms: u32,
    pre_roll_ms: u32,
    input_sample_rate_hint: Option<u32>,
}

#[derive(Debug)]
struct StreamingActiveUtterance {
    utterance_id: String,
    utterance_seq: u64,
    samples_i16: Vec<i16>,
    voiced_ms: f32,
    total_ms: f32,
    silence_ms: f32,
}

#[derive(Debug)]
struct StreamingInputState {
    config: StreamingInputConfig,
    next_utterance_seq: u64,
    frame_seq_last: Option<u32>,
    current_sample_rate: Option<u32>,
    pre_roll: Vec<i16>,
    active: Option<StreamingActiveUtterance>,
}

#[derive(Debug)]
enum BinaryMessageKind {
    ClientPcm16Frame {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy)]
enum UtteranceEndReason {
    Silence,
    MaxDuration,
    StreamStopped,
}

impl UtteranceEndReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::MaxDuration => "max_duration",
            Self::StreamStopped => "stream_stopped",
        }
    }
}

struct ConnectionState {
    system_prompt: String,
    agent_session_id: Option<String>,
    agent_session_system_prompt: Option<String>,
    streaming_input: Option<StreamingInputState>,
    active_turn: Option<ActiveTurn>,
    started: bool,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_AGENT_SYSTEM_PROMPT.to_string(),
            agent_session_id: None,
            agent_session_system_prompt: None,
            streaming_input: None,
            active_turn: None,
            started: false,
        }
    }
}

#[derive(Debug)]
struct SpeechStartEvent {
    utterance_id: String,
    utterance_seq: u64,
}

#[derive(Debug)]
struct StreamingFrameResult {
    speech_start: Option<SpeechStartEvent>,
    finalized_utterance: Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>,
}

impl StreamingInputState {
    fn new(config: StreamingInputConfig) -> Self {
        Self {
            config,
            next_utterance_seq: 0,
            frame_seq_last: None,
            current_sample_rate: None,
            pre_roll: Vec::new(),
            active: None,
        }
    }

    fn handle_pcm16_frame(
        &mut self,
        frame_seq: u32,
        sample_rate: u32,
        payload: &[u8],
    ) -> Result<StreamingFrameResult, String> {
        if sample_rate < 8_000 || sample_rate > 192_000 {
            return Err(format!("Invalid input sample_rate {sample_rate}"));
        }
        if payload.is_empty() {
            return Ok(StreamingFrameResult {
                speech_start: None,
                finalized_utterance: None,
            });
        }
        if payload.len() % 2 != 0 {
            return Err("PCM16 payload length must be even".to_string());
        }

        if let Some(last) = self.frame_seq_last {
            if frame_seq <= last {
                debug!("voice ws input frame sequence non-increasing: {frame_seq} <= {last}");
            }
        }
        self.frame_seq_last = Some(frame_seq);

        if let Some(current_sr) = self.current_sample_rate {
            if current_sr != sample_rate {
                return Err(format!(
                    "Input stream sample rate changed mid-stream ({current_sr} -> {sample_rate})"
                ));
            }
        } else {
            self.current_sample_rate = Some(sample_rate);
        }

        let samples = pcm16_bytes_to_i16(payload);
        if samples.is_empty() {
            return Ok(StreamingFrameResult {
                speech_start: None,
                finalized_utterance: None,
            });
        }

        let rms = rms_i16(&samples);
        let frame_ms = (samples.len() as f32 * 1000.0) / (sample_rate as f32);
        let is_speech = rms >= self.config.vad_threshold;

        let mut result = StreamingFrameResult {
            speech_start: None,
            finalized_utterance: None,
        };

        if is_speech {
            if self.active.is_none() {
                let utterance_seq = self.next_utterance_seq.saturating_add(1);
                self.next_utterance_seq = utterance_seq;
                let utterance_id = format!("utt-{utterance_seq}");

                let mut capture = StreamingActiveUtterance {
                    utterance_id: utterance_id.clone(),
                    utterance_seq,
                    samples_i16: Vec::new(),
                    voiced_ms: 0.0,
                    total_ms: 0.0,
                    silence_ms: 0.0,
                };
                if !self.pre_roll.is_empty() {
                    capture.samples_i16.extend_from_slice(&self.pre_roll);
                }
                capture.samples_i16.extend_from_slice(&samples);
                capture.voiced_ms += frame_ms;
                capture.total_ms += frame_ms;
                self.active = Some(capture);

                result.speech_start = Some(SpeechStartEvent {
                    utterance_id,
                    utterance_seq,
                });
            } else if let Some(active) = self.active.as_mut() {
                active.samples_i16.extend_from_slice(&samples);
                active.voiced_ms += frame_ms;
                active.total_ms += frame_ms;
                active.silence_ms = 0.0;
            }
        } else if let Some(active) = self.active.as_mut() {
            active.samples_i16.extend_from_slice(&samples);
            active.total_ms += frame_ms;
            active.silence_ms += frame_ms;
        } else {
            self.push_pre_roll(&samples, sample_rate);
            return Ok(result);
        }

        let should_finalize = if let Some(active) = self.active.as_ref() {
            if active.total_ms >= self.config.max_utterance_ms as f32 {
                Some(UtteranceEndReason::MaxDuration)
            } else if active.voiced_ms >= self.config.min_speech_ms as f32
                && active.silence_ms >= self.config.silence_duration_ms as f32
            {
                Some(UtteranceEndReason::Silence)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(reason) = should_finalize {
            result.finalized_utterance = self.finalize_active_utterance(reason)?;
        }

        if !is_speech {
            self.push_pre_roll(&samples, sample_rate);
        }

        Ok(result)
    }

    fn finalize_active_utterance(
        &mut self,
        reason: UtteranceEndReason,
    ) -> Result<Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        let sample_rate = self
            .current_sample_rate
            .or(self.config.input_sample_rate_hint)
            .ok_or_else(|| "Missing input sample rate for streamed audio".to_string())?;

        if active.voiced_ms < self.config.min_speech_ms as f32 {
            return Ok(None);
        }

        let wav_bytes = wav_bytes_from_pcm16_mono(&active.samples_i16, sample_rate)?;
        if wav_bytes.len() > MAX_UTTERANCE_BYTES {
            return Err(format!(
                "Streamed utterance exceeded max encoded size ({} > {})",
                wav_bytes.len(),
                MAX_UTTERANCE_BYTES
            ));
        }

        let commit = PendingAudioCommit {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
            turn_config: self.config.turn_config.clone(),
        };

        Ok(Some((commit, wav_bytes, reason)))
    }

    fn push_pre_roll(&mut self, samples: &[i16], sample_rate: u32) {
        let max_samples = ((sample_rate as u64 * self.config.pre_roll_ms as u64) / 1000) as usize;
        if max_samples == 0 {
            self.pre_roll.clear();
            return;
        }

        self.pre_roll.extend_from_slice(samples);
        if self.pre_roll.len() > max_samples {
            let drain = self.pre_roll.len() - max_samples;
            self.pre_roll.drain(0..drain);
        }
    }
}

async fn finalize_stream_vad_utterance(
    state: &AppState,
    correlation_id: &str,
    out_tx: &mpsc::UnboundedSender<Message>,
    conn: &mut ConnectionState,
    commit: PendingAudioCommit,
    wav_bytes: Vec<u8>,
) -> Result<(), String> {
    interrupt_active_turn(out_tx, &mut conn.active_turn, "preempted_by_new_turn");

    let agent_session_id = match &commit.turn_config {
        VoiceTurnConfig::Modular(config) => Some(
            ensure_agent_session(
                state,
                &mut conn.agent_session_id,
                &mut conn.agent_session_system_prompt,
                &conn.system_prompt,
                &config.text_model_id,
            )
            .await?,
        ),
        VoiceTurnConfig::Unified(_) => None,
    };

    let task = spawn_turn_task(
        state.clone(),
        correlation_id.to_string(),
        out_tx.clone(),
        commit.clone(),
        wav_bytes,
        agent_session_id,
    );

    conn.active_turn = Some(ActiveTurn {
        utterance_id: commit.utterance_id,
        utterance_seq: commit.utterance_seq,
        task,
    });

    Ok(())
}

fn parse_binary_message(data: &[u8]) -> Result<BinaryMessageKind, String> {
    if data.len() < WS_BIN_CLIENT_HEADER_LEN || &data[..4] != WS_BIN_MAGIC {
        return Err("Unexpected binary message (missing voice realtime frame header)".to_string());
    }

    let version = data[4];
    if version != WS_BIN_VERSION {
        return Err(format!("Unsupported binary frame version {version}"));
    }

    let kind = data[5];
    match kind {
        WS_BIN_KIND_CLIENT_PCM16 => {
            if data.len() < WS_BIN_CLIENT_HEADER_LEN {
                return Err("Client PCM16 frame too short".to_string());
            }
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

fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &s in samples {
        let v = s as f64 / 32768.0f64;
        sum += v * v;
    }
    (sum / samples.len() as f64).sqrt() as f32
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

fn encode_assistant_audio_binary_frame(
    utterance_seq: u64,
    chunk_seq: u32,
    sample_rate: u32,
    is_final: bool,
    payload_pcm16: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(WS_BIN_ASSISTANT_HEADER_LEN + payload_pcm16.len());
    out.extend_from_slice(WS_BIN_MAGIC);
    out.push(WS_BIN_VERSION);
    out.push(WS_BIN_KIND_ASSISTANT_PCM16);
    let flags: u16 = if is_final { 1 } else { 0 };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&utterance_seq.to_le_bytes());
    out.extend_from_slice(&chunk_seq.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(payload_pcm16);
    out
}

async fn handle_socket(socket: WebSocket, state: AppState, correlation_id: String) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Message>();

    let writer = tokio::spawn(async move {
        while let Some(message) = out_rx.recv().await {
            if ws_tx.send(message).await.is_err() {
                break;
            }
        }
    });

    let mut conn = ConnectionState::default();
    send_json(
        &out_tx,
        json!({
            "type": "connected",
            "protocol": "voice_realtime_v1",
            "server_time_ms": now_unix_millis(),
        }),
    );

    while let Some(result) = ws_rx.next().await {
        let message = match result {
            Ok(message) => message,
            Err(err) => {
                warn!("voice realtime websocket receive error: {err}");
                break;
            }
        };

        match message {
            Message::Text(text) => {
                if let Err(err) =
                    handle_text_message(&state, &correlation_id, &out_tx, &mut conn, text.as_str())
                        .await
                {
                    send_error(&out_tx, None, None, err);
                }
            }
            Message::Binary(data) => {
                if let Err(err) = handle_binary_message(
                    &state,
                    &correlation_id,
                    &out_tx,
                    &mut conn,
                    data.to_vec(),
                )
                .await
                {
                    send_error(&out_tx, None, None, err);
                }
            }
            Message::Close(_) => break,
            Message::Ping(payload) => {
                let _ = out_tx.send(Message::Pong(payload));
            }
            Message::Pong(_) => {}
        }
    }

    interrupt_active_turn(&out_tx, &mut conn.active_turn, "socket_closed");
    drop(out_tx);
    let _ = writer.await;
}

async fn handle_text_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &mpsc::UnboundedSender<Message>,
    conn: &mut ConnectionState,
    text: &str,
) -> Result<(), String> {
    let event: ClientEvent =
        serde_json::from_str(text).map_err(|err| format!("Invalid websocket payload: {err}"))?;

    match event {
        ClientEvent::SessionStart { system_prompt } => {
            if let Some(prompt) = system_prompt
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
            {
                if conn.agent_session_system_prompt.as_deref() != Some(prompt.as_str()) {
                    conn.agent_session_id = None;
                }
                conn.system_prompt = prompt;
            }
            conn.started = true;
            send_json(
                out_tx,
                json!({
                    "type": "session_ready",
                    "protocol": "voice_realtime_v1",
                }),
            );
        }
        ClientEvent::InputStreamStart {
            mode,
            asr_model_id,
            text_model_id,
            tts_model_id,
            s2s_model_id,
            speaker,
            asr_language,
            language,
            max_output_tokens,
            vad_threshold,
            min_speech_ms,
            silence_duration_ms,
            max_utterance_ms,
            pre_roll_ms,
            input_sample_rate,
        } => {
            if !conn.started {
                return Err("Session not started. Send `session_start` first.".to_string());
            }

            let normalized_asr = asr_model_id
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_text = text_model_id
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_tts = tts_model_id
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_s2s = s2s_model_id
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_asr_language = asr_language
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_language = language
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());

            let turn_config = match mode {
                RealtimeVoiceMode::Modular => {
                    let Some(asr_model_id) = normalized_asr else {
                        return Err(
                            "Missing required model ids (`asr_model_id`, `text_model_id`, `tts_model_id`)."
                                .to_string(),
                        );
                    };
                    let Some(text_model_id) = normalized_text else {
                        return Err(
                            "Missing required model ids (`asr_model_id`, `text_model_id`, `tts_model_id`)."
                                .to_string(),
                        );
                    };
                    let Some(tts_model_id) = normalized_tts else {
                        return Err(
                            "Missing required model ids (`asr_model_id`, `text_model_id`, `tts_model_id`)."
                                .to_string(),
                        );
                    };

                    let _ = resolve_chat_model_id(Some(text_model_id.as_str()))?;
                    parse_tts_model_variant(tts_model_id.as_str())
                        .map_err(|err| format!("Unsupported TTS model: {err}"))?;

                    VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
                        asr_model_id,
                        text_model_id,
                        tts_model_id,
                        speaker: speaker
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty()),
                        asr_language: normalized_asr_language,
                        max_output_tokens: max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                    })
                }
                RealtimeVoiceMode::Unified => {
                    let Some(s2s_model_id) = normalized_s2s.or(normalized_tts).or(normalized_asr)
                    else {
                        return Err("Missing unified model id (`s2s_model_id`).".to_string());
                    };
                    let variant = parse_model_variant(&s2s_model_id)
                        .map_err(|err| format!("Unsupported speech-to-speech model: {err}"))?;
                    if !variant.is_lfm2() {
                        return Err(format!(
                            "Unsupported unified speech model '{}'. Supported: LFM2.5-Audio-1.5B, LFM2.5-Audio-1.5B-4bit",
                            s2s_model_id
                        ));
                    }

                    VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
                        s2s_model_id,
                        language: normalized_language.or(normalized_asr_language),
                        system_prompt: Some(conn.system_prompt.clone())
                            .filter(|s| !s.trim().is_empty()),
                    })
                }
            };

            conn.streaming_input = Some(StreamingInputState::new(StreamingInputConfig {
                turn_config,
                vad_threshold: vad_threshold
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or(DEFAULT_STREAM_VAD_THRESHOLD)
                    .clamp(0.0, 1.0),
                min_speech_ms: min_speech_ms
                    .unwrap_or(DEFAULT_STREAM_MIN_SPEECH_MS)
                    .clamp(50, 10_000),
                silence_duration_ms: silence_duration_ms
                    .unwrap_or(DEFAULT_STREAM_SILENCE_MS)
                    .clamp(50, 10_000),
                max_utterance_ms: max_utterance_ms
                    .unwrap_or(DEFAULT_STREAM_MAX_UTTERANCE_MS)
                    .clamp(1_000, 120_000),
                pre_roll_ms: pre_roll_ms
                    .unwrap_or(DEFAULT_STREAM_PRE_ROLL_MS)
                    .clamp(0, 2_000),
                input_sample_rate_hint: input_sample_rate
                    .filter(|sr| *sr >= 8_000 && *sr <= 192_000),
            }));

            send_json(
                out_tx,
                json!({
                    "type": "input_stream_ready",
                    "vad": {
                        "threshold": conn.streaming_input.as_ref().map(|s| s.config.vad_threshold),
                        "min_speech_ms": conn.streaming_input.as_ref().map(|s| s.config.min_speech_ms),
                        "silence_duration_ms": conn.streaming_input.as_ref().map(|s| s.config.silence_duration_ms),
                    }
                }),
            );
        }
        ClientEvent::InputStreamStop => {
            if let Some(mut streaming) = conn.streaming_input.take() {
                if let Some((commit, wav_bytes, end_reason)) =
                    streaming.finalize_active_utterance(UtteranceEndReason::StreamStopped)?
                {
                    send_json(
                        out_tx,
                        json!({
                            "type": "user_speech_end",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "reason": end_reason.as_str(),
                        }),
                    );
                    finalize_stream_vad_utterance(
                        state,
                        correlation_id,
                        out_tx,
                        conn,
                        commit,
                        wav_bytes,
                    )
                    .await?;
                }
            }
            send_json(out_tx, json!({ "type": "input_stream_stopped" }));
        }
        ClientEvent::Interrupt { reason } => {
            let reason = reason.unwrap_or_else(|| "client_interrupt".to_string());
            interrupt_active_turn(out_tx, &mut conn.active_turn, &reason);
        }
        ClientEvent::Ping { timestamp_ms } => {
            send_json(
                out_tx,
                json!({
                    "type": "pong",
                    "timestamp_ms": timestamp_ms,
                    "server_time_ms": now_unix_millis(),
                }),
            );
        }
    }

    // Silence unused parameters in some branches (kept for future per-message needs).
    let _ = (state, correlation_id);
    Ok(())
}

async fn handle_binary_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &mpsc::UnboundedSender<Message>,
    conn: &mut ConnectionState,
    audio_bytes: Vec<u8>,
) -> Result<(), String> {
    match parse_binary_message(&audio_bytes)? {
        BinaryMessageKind::ClientPcm16Frame {
            frame_seq,
            sample_rate,
            payload,
        } => {
            if !conn.started {
                return Err("Session not started. Send `session_start` first.".to_string());
            }
            let Some(streaming) = conn.streaming_input.as_mut() else {
                return Err(
                    "Received streaming audio frame before `input_stream_start`.".to_string(),
                );
            };

            let frame_result = streaming.handle_pcm16_frame(frame_seq, sample_rate, &payload)?;

            if let Some(evt) = frame_result.speech_start {
                if conn.active_turn.is_some() {
                    interrupt_active_turn(out_tx, &mut conn.active_turn, "barge_in");
                }
                send_json(
                    out_tx,
                    json!({
                        "type": "user_speech_start",
                        "utterance_id": evt.utterance_id,
                        "utterance_seq": evt.utterance_seq,
                    }),
                );
            }

            if let Some((commit, wav_bytes, end_reason)) = frame_result.finalized_utterance {
                send_json(
                    out_tx,
                    json!({
                        "type": "user_speech_end",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "reason": end_reason.as_str(),
                    }),
                );
                finalize_stream_vad_utterance(
                    state,
                    correlation_id,
                    out_tx,
                    conn,
                    commit,
                    wav_bytes,
                )
                .await?;
            }

            return Ok(());
        }
    }
}

fn spawn_turn_task(
    state: AppState,
    correlation_id: String,
    out_tx: mpsc::UnboundedSender<Message>,
    commit: PendingAudioCommit,
    audio_bytes: Vec<u8>,
    agent_session_id: Option<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let timeout_secs = state.request_timeout_secs.max(1);
        let timeout = Duration::from_secs(timeout_secs);

        let turn_future = async {
            let _permit = state
                .request_semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| "Server is shutting down".to_string())?;

            send_json(
                &out_tx,
                json!({
                    "type": "turn_processing",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );

            let audio_base64 = base64::engine::general_purpose::STANDARD.encode(audio_bytes);

            match &commit.turn_config {
                VoiceTurnConfig::Modular(config) => {
                    send_json(
                        &out_tx,
                        json!({
                            "type": "user_transcript_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    let transcript = {
                        let tx = out_tx.clone();
                        let utt_id = commit.utterance_id.clone();
                        let utt_seq = commit.utterance_seq;
                        let asr_model_id = config.asr_model_id.clone();
                        let asr_language = config.asr_language.clone();
                        state
                            .runtime
                            .asr_transcribe_streaming_with_correlation(
                                &audio_base64,
                                Some(&asr_model_id),
                                asr_language.as_deref(),
                                Some(&correlation_id),
                                move |delta| {
                                    if delta.is_empty() {
                                        return;
                                    }
                                    send_json(
                                        &tx,
                                        json!({
                                            "type": "user_transcript_delta",
                                            "utterance_id": utt_id,
                                            "utterance_seq": utt_seq,
                                            "delta": delta,
                                        }),
                                    );
                                },
                            )
                            .await
                            .map_err(|err| format!("ASR failed: {err}"))?
                    };

                    let user_text = transcript.text.trim().to_string();
                    send_json(
                        &out_tx,
                        json!({
                            "type": "user_transcript_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": user_text,
                            "language": transcript.language,
                            "audio_duration_secs": transcript.duration_secs,
                        }),
                    );

                    if user_text.is_empty() {
                        send_json(
                            &out_tx,
                            json!({
                                "type": "turn_done",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "status": "no_input",
                            }),
                        );
                        return Ok::<(), String>(());
                    }

                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_text_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    let Some(agent_session_id) = agent_session_id.as_deref() else {
                        return Err("Missing agent session for modular voice turn".to_string());
                    };
                    let assistant_raw = run_agent_turn(
                        &state,
                        agent_session_id,
                        &user_text,
                        &config.text_model_id,
                        config.max_output_tokens,
                        &correlation_id,
                    )
                    .await?;
                    let assistant_text = strip_think_tags(&assistant_raw);

                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_text_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": assistant_text,
                            "raw_text": assistant_raw,
                        }),
                    );

                    if assistant_text.is_empty() {
                        send_json(
                            &out_tx,
                            json!({
                                "type": "turn_done",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "status": "ok",
                            }),
                        );
                        return Ok(());
                    }

                    stream_tts_to_socket(
                        &state,
                        &out_tx,
                        &correlation_id,
                        &commit.utterance_id,
                        commit.utterance_seq,
                        &config.tts_model_id,
                        config.speaker.clone(),
                        assistant_text.as_str(),
                    )
                    .await?;

                    send_json(
                        &out_tx,
                        json!({
                            "type": "turn_done",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "status": "ok",
                        }),
                    );
                    Ok(())
                }
                VoiceTurnConfig::Unified(config) => {
                    send_json(
                        &out_tx,
                        json!({
                            "type": "user_transcript_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    let transcript_result = {
                        let tx = out_tx.clone();
                        let utt_id = commit.utterance_id.clone();
                        let utt_seq = commit.utterance_seq;
                        let model_id = config.s2s_model_id.clone();
                        let language = config.language.clone();
                        state
                            .runtime
                            .asr_transcribe_streaming_with_correlation(
                                &audio_base64,
                                Some(&model_id),
                                language.as_deref(),
                                Some(&correlation_id),
                                move |delta| {
                                    if delta.is_empty() {
                                        return;
                                    }
                                    send_json(
                                        &tx,
                                        json!({
                                            "type": "user_transcript_delta",
                                            "utterance_id": utt_id,
                                            "utterance_seq": utt_seq,
                                            "delta": delta,
                                        }),
                                    );
                                },
                            )
                            .await
                    };

                    let (user_text, user_language, user_audio_duration_secs) =
                        match transcript_result {
                            Ok(transcript) => {
                                let text = transcript.text.trim();
                                let final_text = if text.is_empty() {
                                    "User speech captured (transcription unavailable).".to_string()
                                } else {
                                    text.to_string()
                                };
                                (
                                    final_text,
                                    transcript.language,
                                    Some(transcript.duration_secs),
                                )
                            }
                            Err(err) => {
                                warn!("unified websocket transcript failed, continuing with speech generation: {err}");
                                (
                                    "User speech captured (transcription unavailable).".to_string(),
                                    None,
                                    None,
                                )
                            }
                        };
                    send_json(
                        &out_tx,
                        json!({
                            "type": "user_transcript_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": user_text,
                            "language": user_language,
                            "audio_duration_secs": user_audio_duration_secs,
                        }),
                    );

                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_text_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    let sample_rate = state.runtime.sample_rate().await;
                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_audio_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "sample_rate": sample_rate,
                            "audio_format": "pcm_i16",
                        }),
                    );

                    let audio_tx = out_tx.clone();
                    let utt_id = commit.utterance_id.clone();
                    let utt_seq = commit.utterance_seq;
                    let stream_result = state
                        .runtime
                        .lfm2_speech_to_speech_streaming_with_correlation(
                            &audio_base64,
                            config.language.as_deref(),
                            config.system_prompt.as_deref(),
                            None,
                            None,
                            Some(&correlation_id),
                            |_delta| {},
                            move |audio_chunk| {
                                if audio_chunk.samples.is_empty() && !audio_chunk.is_final {
                                    return;
                                }

                                let encoded = match AudioEncoder::new(sample_rate, 1)
                                    .encode(&audio_chunk.samples, AudioFormat::RawI16)
                                {
                                    Ok(bytes) => bytes,
                                    Err(err) => {
                                        send_error(
                                            &audio_tx,
                                            Some(utt_id.clone()),
                                            Some(utt_seq),
                                            format!("Failed to encode unified speech chunk: {err}"),
                                        );
                                        return;
                                    }
                                };

                                let chunk_seq =
                                    u32::try_from(audio_chunk.sequence).unwrap_or(u32::MAX);
                                let frame = encode_assistant_audio_binary_frame(
                                    utt_seq,
                                    chunk_seq,
                                    sample_rate,
                                    audio_chunk.is_final,
                                    &encoded,
                                );
                                let _ = audio_tx.send(Message::Binary(frame.into()));
                            },
                        )
                        .await
                        .map_err(|err| format!("Unified speech-to-speech failed: {err}"))?;

                    let assistant_text = strip_think_tags(stream_result.text.as_str());
                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_text_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": assistant_text,
                            "raw_text": stream_result.text,
                        }),
                    );

                    send_json(
                        &out_tx,
                        json!({
                            "type": "assistant_audio_done",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    send_json(
                        &out_tx,
                        json!({
                            "type": "turn_done",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "status": "ok",
                        }),
                    );
                    Ok(())
                }
            }
        };

        match tokio::time::timeout(timeout, turn_future).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                send_error(
                    &out_tx,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    err,
                );
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "error",
                    }),
                );
            }
            Err(_) => {
                send_error(
                    &out_tx,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    format!("Turn timed out after {timeout_secs} seconds"),
                );
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "timeout",
                    }),
                );
            }
        }
    })
}

async fn stream_tts_to_socket(
    state: &AppState,
    out_tx: &mpsc::UnboundedSender<Message>,
    correlation_id: &str,
    utterance_id: &str,
    utterance_seq: u64,
    tts_model_id: &str,
    speaker: Option<String>,
    text: &str,
) -> Result<(), String> {
    let tts_variant = parse_tts_model_variant(tts_model_id)
        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
    state
        .runtime
        .load_model(tts_variant)
        .await
        .map_err(|err| format!("Failed to load TTS model: {err}"))?;

    let sample_rate = state.runtime.sample_rate().await;
    let encoder = AudioEncoder::new(sample_rate, 1);

    send_json(
        out_tx,
        json!({
            "type": "assistant_audio_start",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "sample_rate": sample_rate,
            "audio_format": "pcm_i16",
        }),
    );

    let mut gen_config = GenerationConfig::default();
    gen_config.streaming = true;
    gen_config.options.max_tokens = 0;
    gen_config.options.speaker = speaker.clone();
    gen_config.options.voice = speaker;

    let gen_request = GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        correlation_id: Some(correlation_id.to_string()),
        text: text.to_string(),
        config: gen_config,
        language: None,
        reference_audio: None,
        reference_text: None,
        voice_description: None,
    };

    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<izwi_core::AudioChunk>(32);
    let runtime = state.runtime.clone();
    let generation_task =
        tokio::spawn(async move { runtime.generate_streaming(gen_request, chunk_tx).await });

    while let Some(chunk) = chunk_rx.recv().await {
        if chunk.samples.is_empty() && !chunk.is_final {
            continue;
        }

        let encoded = encoder
            .encode(&chunk.samples, AudioFormat::RawI16)
            .map_err(|err| format!("Failed to encode streamed TTS chunk: {err}"))?;

        let chunk_seq = u32::try_from(chunk.sequence).unwrap_or(u32::MAX);
        let frame = encode_assistant_audio_binary_frame(
            utterance_seq,
            chunk_seq,
            sample_rate,
            chunk.is_final,
            &encoded,
        );
        let _ = out_tx.send(Message::Binary(frame.into()));
    }

    match generation_task.await {
        Ok(Ok(())) => {
            send_json(
                out_tx,
                json!({
                    "type": "assistant_audio_done",
                    "utterance_id": utterance_id,
                    "utterance_seq": utterance_seq,
                }),
            );
            Ok(())
        }
        Ok(Err(err)) => Err(format!("TTS failed: {err}")),
        Err(err) => Err(format!("TTS streaming task failed: {err}")),
    }
}

fn interrupt_active_turn(
    out_tx: &mpsc::UnboundedSender<Message>,
    active_turn: &mut Option<ActiveTurn>,
    reason: &str,
) {
    if let Some(turn) = active_turn.take() {
        if turn.task.is_finished() {
            return;
        }
        turn.task.abort();
        send_json(
            out_tx,
            json!({
                "type": "turn_done",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
                "status": "interrupted",
                "reason": reason,
            }),
        );
    }
}

async fn ensure_agent_session(
    state: &AppState,
    agent_session_id: &mut Option<String>,
    agent_session_system_prompt: &mut Option<String>,
    system_prompt: &str,
    text_model_id: &str,
) -> Result<String, String> {
    if let Some(existing_id) = agent_session_id.as_ref() {
        if agent_session_system_prompt.as_deref() == Some(system_prompt) {
            return Ok(existing_id.clone());
        }
    }

    let model_id = resolve_chat_model_id(Some(text_model_id))?;
    let thread = state
        .chat_store
        .create_thread(Some("Voice Session".to_string()), Some(model_id.clone()))
        .await
        .map_err(|err| format!("Chat storage error: {err}"))?;

    let now = now_unix_millis();
    let session_id = format!("agent_sess_{}", uuid::Uuid::new_v4().simple());
    let record = StoredAgentSessionRecord {
        id: session_id.clone(),
        agent_id: DEFAULT_AGENT_ID.to_string(),
        thread_id: thread.id,
        model_id,
        system_prompt: system_prompt.to_string(),
        planning_mode: PlanningMode::Auto,
        created_at: now,
        updated_at: now,
    };

    state
        .agent_session_store
        .write()
        .await
        .insert(session_id.clone(), record);

    *agent_session_id = Some(session_id.clone());
    *agent_session_system_prompt = Some(system_prompt.to_string());
    Ok(session_id)
}

async fn run_agent_turn(
    state: &AppState,
    session_id: &str,
    input: &str,
    model_id: &str,
    max_output_tokens: usize,
    correlation_id: &str,
) -> Result<String, String> {
    let session_record = {
        let store = state.agent_session_store.read().await;
        store
            .get(session_id)
            .cloned()
            .ok_or_else(|| "Agent session not found".to_string())?
    };

    let resolved_model_id = resolve_chat_model_id(Some(model_id))?;

    let agent = AgentDefinition {
        id: session_record.agent_id.clone(),
        name: DEFAULT_AGENT_NAME.to_string(),
        system_prompt: session_record.system_prompt.clone(),
        default_model: session_record.model_id.clone(),
        capabilities: Default::default(),
        planning_mode: session_record.planning_mode,
    };
    let session = AgentSession {
        id: session_record.id.clone(),
        agent_id: session_record.agent_id.clone(),
        thread_id: session_record.thread_id.clone(),
        created_at: session_record.created_at,
        updated_at: session_record.updated_at,
    };

    let memory = ChatStoreMemory::new(state.chat_store.clone());
    let backend = IzwiRuntimeBackend {
        runtime: state.runtime.clone(),
        correlation_id: correlation_id.to_string(),
    };
    let planner = SimplePlanner;
    let mut tools = ToolRegistry::new();
    tools.register(NoopTool);
    tools.register(TimeTool);

    let result = AgentEngine
        .run_turn(
            &agent,
            &session,
            TurnInput {
                text: input.to_string(),
            },
            Some(resolved_model_id.clone()),
            &memory,
            &backend,
            &planner,
            &tools,
            AgentTurnOptions {
                max_output_tokens: max_output_tokens.clamp(1, 4096),
                max_tool_calls: 1,
            },
        )
        .await
        .map_err(|err| match err {
            izwi_agent::AgentError::InvalidInput(msg) => msg,
            other => other.to_string(),
        })?;

    {
        let mut store = state.agent_session_store.write().await;
        if let Some(record) = store.get_mut(session_id) {
            record.updated_at = now_unix_millis();
            record.model_id = resolved_model_id;
        }
    }

    Ok(result.assistant_text)
}

fn resolve_chat_model_id(raw: Option<&str>) -> Result<String, String> {
    let requested = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_CHAT_MODEL);
    let variant = parse_chat_model_variant(Some(requested))
        .map_err(|err| format!("Invalid chat model: {err}"))?;
    Ok(variant.dir_name().to_string())
}

fn send_json(out_tx: &mpsc::UnboundedSender<Message>, value: serde_json::Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send(Message::Text(text.into())).is_ok(),
        Err(err) => {
            warn!("failed to serialize voice ws event: {err}");
            false
        }
    }
}

fn send_error(
    out_tx: &mpsc::UnboundedSender<Message>,
    utterance_id: Option<String>,
    utterance_seq: Option<u64>,
    message: impl Into<String>,
) {
    let message = message.into();
    let _ = send_json(
        out_tx,
        json!({
            "type": "error",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "message": message,
        }),
    );
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn strip_think_tags(input: &str) -> String {
    let open_tag = "<think>";
    let close_tag = "</think>";
    let mut out = input.to_string();

    loop {
        let Some(start) = out.find(open_tag) else {
            break;
        };
        if let Some(end_rel) = out[start + open_tag.len()..].find(close_tag) {
            let end = start + open_tag.len() + end_rel;
            let mut next = String::with_capacity(out.len());
            next.push_str(&out[..start]);
            next.push_str(&out[end + close_tag.len()..]);
            out = next;
        } else {
            out.truncate(start);
            break;
        }
    }

    out.trim().to_string()
}

struct ChatStoreMemory {
    chat_store: Arc<ChatStore>,
}

impl ChatStoreMemory {
    fn new(chat_store: Arc<ChatStore>) -> Self {
        Self { chat_store }
    }
}

#[async_trait::async_trait]
impl MemoryStore for ChatStoreMemory {
    async fn load_messages(&self, thread_id: &str) -> izwi_agent::Result<Vec<MemoryMessage>> {
        let records = self
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;

        let mut out = Vec::with_capacity(records.len());
        for record in records {
            let role = match record.role.as_str() {
                "system" => MemoryMessageRole::System,
                "user" => MemoryMessageRole::User,
                "assistant" => MemoryMessageRole::Assistant,
                other => {
                    return Err(izwi_agent::AgentError::Memory(format!(
                        "Invalid stored chat role: {other}"
                    )))
                }
            };
            out.push(MemoryMessage {
                role,
                content: record.content,
            });
        }

        Ok(out)
    }

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> izwi_agent::Result<()> {
        self.chat_store
            .append_message(
                thread_id.to_string(),
                role.as_str().to_string(),
                content,
                None,
                meta.model_id,
                meta.tokens_generated,
                meta.generation_time_ms,
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;
        Ok(())
    }
}

struct IzwiRuntimeBackend {
    runtime: Arc<izwi_core::RuntimeService>,
    correlation_id: String,
}

#[async_trait::async_trait]
impl ModelBackend for IzwiRuntimeBackend {
    async fn generate(&self, request: ModelRequest) -> izwi_agent::Result<ModelOutput> {
        let variant = parse_chat_model_variant(Some(&request.model_id))
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        let mut runtime_messages = Vec::with_capacity(request.messages.len());
        for message in request.messages {
            let role = match message.role {
                MemoryMessageRole::System => ChatRole::System,
                MemoryMessageRole::User => ChatRole::User,
                MemoryMessageRole::Assistant => ChatRole::Assistant,
            };
            runtime_messages.push(ChatMessage {
                role,
                content: message.content,
            });
        }

        let generation = self
            .runtime
            .chat_generate_with_correlation(
                variant,
                runtime_messages,
                request.max_output_tokens.clamp(1, 4096),
                Some(&self.correlation_id),
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        Ok(ModelOutput {
            text: generation.text,
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        })
    }
}

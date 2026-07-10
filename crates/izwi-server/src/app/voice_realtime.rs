//! Realtime voice websocket endpoint for `/voice`.
//!
//! Frontend responsibilities:
//! - microphone capture
//! - level metering and audio framing
//! - audio playback
//!
//! Backend responsibilities:
//! - ASR -> agent -> TTS orchestration
//! - shared VAD and endpointing
//! - streaming assistant audio/text events
//! - interruption / barge-in cancellation

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
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
    GenerationConfig, GenerationParams, GenerationRequest, RuntimeRequestContext, RuntimeService,
    VoiceSession, WorkloadClass,
};
use izwi_vad::{
    sanitize_score_threshold, EndpointConfig, EndpointDetector, EndpointEndReason, EndpointEvent,
    VadScorer, DEFAULT_MAX_UTTERANCE_MS, DEFAULT_MIN_SPEECH_MS, DEFAULT_PRE_ROLL_MS,
    DEFAULT_SILENCE_MS, DEFAULT_SPEECH_THRESHOLD, VAD_FRAME_MS, VAD_SAMPLE_RATE,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;
use tracing::warn;

use crate::chat_store::ChatStore;
use crate::ids::new_uuid;
use crate::state::{AppState, StoredAgentSessionRecord};
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_NAME, DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
};
use crate::voice_memory::extract_observation_candidates;
use crate::voice_store::CreateVoiceTurnRequest;
const DEFAULT_CHAT_MODEL: &str = "Qwen3-1.7B-GGUF";
const MAX_UTTERANCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_UTTERANCE_PCM16_SAMPLES: usize = (MAX_UTTERANCE_BYTES - 44) / 2;
const WS_BIN_MAGIC: &[u8; 4] = b"IVWS";
const WS_BIN_VERSION: u8 = 1;
const WS_BIN_KIND_CLIENT_PCM16: u8 = 1;
const WS_BIN_KIND_ASSISTANT_PCM16: u8 = 2;
const WS_BIN_CLIENT_HEADER_LEN: usize = 16;
const WS_BIN_ASSISTANT_HEADER_LEN: usize = 24;
const DEFAULT_STREAM_VAD_THRESHOLD: f32 = DEFAULT_SPEECH_THRESHOLD;
const DEFAULT_STREAM_MIN_SPEECH_MS: u32 = DEFAULT_MIN_SPEECH_MS;
const DEFAULT_STREAM_SILENCE_MS: u32 = DEFAULT_SILENCE_MS;
const DEFAULT_STREAM_MAX_UTTERANCE_MS: u32 = DEFAULT_MAX_UTTERANCE_MS;
const DEFAULT_STREAM_PRE_ROLL_MS: u32 = DEFAULT_PRE_ROLL_MS;
const WS_OUTBOUND_QUEUE_CAPACITY: usize = 256;
const WS_OUTBOUND_AUDIO_MAX_BYTES: usize = 4 * 1024 * 1024;
const WS_OUTBOUND_AUDIO_MAX_MS: u64 = 5_000;
const WS_WRITER_SEND_TIMEOUT: Duration = Duration::from_secs(5);
const TURN_CANCELLATION_TIMEOUT: Duration = Duration::from_millis(750);
const SESSION_ACTOR_TICK: Duration = Duration::from_millis(100);

#[derive(Clone)]
struct OutboundTx {
    mailbox: Arc<OutboundMailbox>,
    runtime: Arc<RuntimeService>,
    label: &'static str,
}

impl OutboundTx {
    fn new(runtime: Arc<RuntimeService>, label: &'static str) -> Self {
        Self {
            mailbox: Arc::new(OutboundMailbox::new(OutboundLimits::from_env())),
            runtime,
            label,
        }
    }

    fn send(&self, message: Message) -> bool {
        self.enqueue(message, OutboundClass::Critical, None).is_ok()
    }

    fn send_turn(&self, generation: u64, message: Message) -> bool {
        self.enqueue(message, OutboundClass::Critical, Some(generation))
            .is_ok()
    }

    fn send_text_snapshot(&self, generation: u64, key: String, message: Message) -> bool {
        self.enqueue(
            message,
            OutboundClass::CoalescibleText(key),
            Some(generation),
        )
        .is_ok()
    }

    fn send_audio(
        &self,
        generation: u64,
        message: Message,
        bytes: usize,
        duration_ms: u64,
    ) -> Result<(), OutboundEnqueueError> {
        self.enqueue(
            message,
            OutboundClass::Audio { bytes, duration_ms },
            Some(generation),
        )
    }

    fn send_diagnostic(&self, message: Message) -> bool {
        self.enqueue(message, OutboundClass::Diagnostic, None)
            .is_ok()
    }

    fn enqueue(
        &self,
        message: Message,
        class: OutboundClass,
        turn_generation: Option<u64>,
    ) -> Result<(), OutboundEnqueueError> {
        let result = self.mailbox.enqueue(OutboundItem {
            message,
            class,
            turn_generation,
        });
        if let Err(err) = &result {
            if !matches!(err, OutboundEnqueueError::Cutoff) {
                self.runtime.record_voice_stream_backpressure();
                warn!(label = self.label, error = ?err, "voice outbound delivery rejected");
            }
        }
        result
    }

    fn cutoff_turn(&self, generation: u64) {
        self.mailbox.cutoff_turn(generation);
    }

    async fn next(&self) -> Option<MailboxOutput> {
        self.mailbox.next().await
    }

    fn close(&self) {
        self.mailbox.close();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutboundClass {
    Critical,
    CoalescibleText(String),
    Audio { bytes: usize, duration_ms: u64 },
    Diagnostic,
}

struct OutboundItem {
    message: Message,
    class: OutboundClass,
    turn_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct OutboundLimits {
    max_items: usize,
    max_audio_bytes: usize,
    max_audio_ms: u64,
}

impl OutboundLimits {
    fn from_env() -> Self {
        Self {
            max_items: env_usize("IZWI_VOICE_WS_OUTBOUND_ITEMS", WS_OUTBOUND_QUEUE_CAPACITY),
            max_audio_bytes: env_usize(
                "IZWI_VOICE_WS_OUTBOUND_AUDIO_BYTES",
                WS_OUTBOUND_AUDIO_MAX_BYTES,
            ),
            max_audio_ms: env_u64("IZWI_VOICE_WS_OUTBOUND_AUDIO_MS", WS_OUTBOUND_AUDIO_MAX_MS),
        }
    }
}

struct OutboundMailbox {
    limits: OutboundLimits,
    state: Mutex<OutboundMailboxState>,
    notify: Notify,
    minimum_turn_generation: AtomicU64,
}

#[derive(Default)]
struct OutboundMailboxState {
    queue: VecDeque<OutboundItem>,
    audio_bytes: usize,
    audio_ms: u64,
    close_reason: Option<String>,
    closed: bool,
}

enum MailboxOutput {
    Message(Message),
    Close(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutboundEnqueueError {
    Cutoff,
    AudioSaturated,
    QueueSaturated,
    Closed,
    DroppedDiagnostic,
}

impl OutboundMailbox {
    fn new(limits: OutboundLimits) -> Self {
        Self {
            limits,
            state: Mutex::new(OutboundMailboxState::default()),
            notify: Notify::new(),
            minimum_turn_generation: AtomicU64::new(0),
        }
    }

    fn enqueue(&self, item: OutboundItem) -> Result<(), OutboundEnqueueError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if item.turn_generation.is_some_and(|generation| {
            generation < self.minimum_turn_generation.load(Ordering::Acquire)
        }) {
            return Err(OutboundEnqueueError::Cutoff);
        }
        if state.closed {
            return Err(OutboundEnqueueError::Closed);
        }

        if let OutboundClass::CoalescibleText(key) = &item.class {
            if let Some(existing) = state.queue.iter_mut().find(|queued| {
                queued.turn_generation == item.turn_generation
                    && matches!(&queued.class, OutboundClass::CoalescibleText(existing_key) if existing_key == key)
            }) {
                existing.message = item.message;
                return Ok(());
            }
        }

        if let OutboundClass::Audio { bytes, duration_ms } = &item.class {
            if state.audio_bytes.saturating_add(*bytes) > self.limits.max_audio_bytes
                || state.audio_ms.saturating_add(*duration_ms) > self.limits.max_audio_ms
            {
                state.close_reason = Some("outbound_audio_saturated".to_string());
                self.notify.notify_one();
                return Err(OutboundEnqueueError::AudioSaturated);
            }
        }

        if state.queue.len() >= self.limits.max_items {
            match &item.class {
                OutboundClass::Critical => {
                    state.close_reason = Some("critical_outbound_queue_saturated".to_string());
                    self.notify.notify_one();
                    return Err(OutboundEnqueueError::QueueSaturated);
                }
                OutboundClass::Audio { .. } => {
                    state.close_reason = Some("outbound_audio_saturated".to_string());
                    self.notify.notify_one();
                    return Err(OutboundEnqueueError::AudioSaturated);
                }
                OutboundClass::Diagnostic => {
                    return Err(OutboundEnqueueError::DroppedDiagnostic);
                }
                OutboundClass::CoalescibleText(_) => {
                    return Err(OutboundEnqueueError::QueueSaturated);
                }
            }
        }

        if let OutboundClass::Audio { bytes, duration_ms } = &item.class {
            state.audio_bytes = state.audio_bytes.saturating_add(*bytes);
            state.audio_ms = state.audio_ms.saturating_add(*duration_ms);
        }
        state.queue.push_back(item);
        drop(state);
        self.notify.notify_one();
        Ok(())
    }

    fn cutoff_turn(&self, generation: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.minimum_turn_generation
            .fetch_max(generation.saturating_add(1), Ordering::AcqRel);
        let mut retained = VecDeque::with_capacity(state.queue.len());
        while let Some(item) = state.queue.pop_front() {
            if item.turn_generation == Some(generation) {
                subtract_audio_depth(&mut state, &item.class);
            } else {
                retained.push_back(item);
            }
        }
        state.queue = retained;
    }

    async fn next(&self) -> Option<MailboxOutput> {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                if let Some(reason) = state.close_reason.take() {
                    state.queue.clear();
                    state.audio_bytes = 0;
                    state.audio_ms = 0;
                    state.closed = true;
                    return Some(MailboxOutput::Close(reason));
                }
                if let Some(item) = state.queue.pop_front() {
                    subtract_audio_depth(&mut state, &item.class);
                    return Some(MailboxOutput::Message(item.message));
                }
                if state.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
    }
}

fn subtract_audio_depth(state: &mut OutboundMailboxState, class: &OutboundClass) {
    if let OutboundClass::Audio { bytes, duration_ms } = class {
        state.audio_bytes = state.audio_bytes.saturating_sub(*bytes);
        state.audio_ms = state.audio_ms.saturating_sub(*duration_ms);
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
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

impl RealtimeVoiceMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Modular => "modular",
            Self::Unified => "unified",
        }
    }
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
    speaker: Option<String>,
    max_output_tokens: usize,
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

struct ActiveTurn {
    utterance_id: String,
    utterance_seq: u64,
    turn_record_id: String,
    state: AppState,
    task: tokio::task::JoinHandle<()>,
    control: TurnControl,
    output_generation: u64,
}

#[derive(Clone)]
struct TurnControl {
    inner: Arc<TurnControlInner>,
}

struct TurnControlInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl TurnControl {
    fn new() -> Self {
        Self {
            inner: Arc::new(TurnControlInner {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::AcqRel) {
            self.inner.notify.notify_waiters();
        }
    }

    fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    async fn cancelled(&self) {
        loop {
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

struct AbortOnDropTask<T> {
    handle: Option<tokio::task::JoinHandle<T>>,
}

impl<T> AbortOnDropTask<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    async fn join(mut self) -> Result<T, tokio::task::JoinError> {
        self.handle.take().expect("task handle must exist").await
    }
}

impl<T> Drop for AbortOnDropTask<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.as_ref() {
            handle.abort();
        }
    }
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

impl StreamingInputConfig {
    fn endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            start_threshold: self.vad_threshold,
            end_threshold: (self.vad_threshold * 0.7).min(self.vad_threshold),
            min_speech_ms: self.min_speech_ms,
            silence_ms: self.silence_duration_ms,
            max_utterance_ms: self.max_utterance_ms,
        }
    }
}

#[derive(Debug)]
struct StreamingActiveUtterance {
    utterance_id: String,
    utterance_seq: u64,
    samples_i16: Vec<i16>,
    voiced_ms: f32,
    total_ms: f32,
    silence_ms: f32,
    started_at: Instant,
}

struct StreamingInputState {
    config: StreamingInputConfig,
    vad_scorer: VadScorer,
    endpoint: EndpointDetector,
    next_utterance_seq: u64,
    frame_seq_last: Option<u32>,
    current_sample_rate: Option<u32>,
    pre_roll: VecDeque<i16>,
    active: Option<StreamingActiveUtterance>,
    last_frame_at: Option<Instant>,
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
    ClientPause,
    StreamStopped,
}

impl UtteranceEndReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::MaxDuration => "max_duration",
            Self::ClientPause => "client_pause",
            Self::StreamStopped => "stream_stopped",
        }
    }
}

impl From<EndpointEndReason> for UtteranceEndReason {
    fn from(reason: EndpointEndReason) -> Self {
        match reason {
            EndpointEndReason::Silence => Self::Silence,
            EndpointEndReason::MaxDuration => Self::MaxDuration,
            EndpointEndReason::StreamStopped => Self::StreamStopped,
        }
    }
}

impl VoiceTurnConfig {
    fn mode_str(&self) -> &'static str {
        match self {
            Self::Modular(_) => RealtimeVoiceMode::Modular.as_str(),
            Self::Unified(_) => RealtimeVoiceMode::Unified.as_str(),
        }
    }

    fn speaker(&self) -> Option<String> {
        match self {
            Self::Modular(config) => config.speaker.clone(),
            Self::Unified(config) => config.speaker.clone(),
        }
    }

    fn model_ids(&self) -> VoiceTurnModelIds {
        match self {
            Self::Modular(config) => VoiceTurnModelIds {
                asr_model_id: Some(config.asr_model_id.clone()),
                text_model_id: Some(config.text_model_id.clone()),
                tts_model_id: Some(config.tts_model_id.clone()),
                s2s_model_id: None,
            },
            Self::Unified(config) => VoiceTurnModelIds {
                asr_model_id: None,
                text_model_id: None,
                tts_model_id: None,
                s2s_model_id: Some(config.s2s_model_id.clone()),
            },
        }
    }
}

#[derive(Debug, Clone)]
struct VoiceTurnModelIds {
    asr_model_id: Option<String>,
    text_model_id: Option<String>,
    tts_model_id: Option<String>,
    s2s_model_id: Option<String>,
}

struct ConnectionState {
    system_prompt: String,
    voice_profile_id: Option<String>,
    voice_session_id: Option<String>,
    agent_session_id: Option<String>,
    agent_session_system_prompt: Option<String>,
    streaming_input: Option<StreamingInputState>,
    active_turn: Option<ActiveTurn>,
    voice_session: VoiceSession,
    started: bool,
    transport_session_id: String,
    owner_instance_id: String,
    next_turn_generation: u64,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.to_string(),
            voice_profile_id: None,
            voice_session_id: None,
            agent_session_id: None,
            agent_session_system_prompt: None,
            streaming_input: None,
            active_turn: None,
            voice_session: VoiceSession::default(),
            started: false,
            transport_session_id: uuid::Uuid::new_v4().to_string(),
            owner_instance_id: process_owner_instance_id().to_string(),
            next_turn_generation: 0,
        }
    }
}

fn process_owner_instance_id() -> &'static str {
    static OWNER_INSTANCE_ID: OnceLock<String> = OnceLock::new();
    OWNER_INSTANCE_ID
        .get_or_init(|| format!("process-{}-{}", std::process::id(), uuid::Uuid::new_v4()))
        .as_str()
}

impl ConnectionState {
    fn clear_finished_turn(&mut self) {
        if self
            .active_turn
            .as_ref()
            .map(|turn| turn.task.is_finished())
            .unwrap_or(false)
        {
            self.active_turn = None;
        }
    }

    fn has_running_turn(&self) -> bool {
        self.active_turn
            .as_ref()
            .map(|turn| !turn.task.is_finished())
            .unwrap_or(false)
    }

    fn close_voice_session_if_idle(&mut self) -> Option<String> {
        self.clear_finished_turn();
        if self.has_running_turn() {
            return None;
        }
        self.voice_session.close();
        self.voice_session_id.take()
    }

    fn close_voice_session_now(&mut self) -> Option<String> {
        self.voice_session.close();
        self.voice_session_id.take()
    }
}

#[derive(Debug)]
struct SpeechStartEvent {
    utterance_id: String,
    utterance_seq: u64,
}

#[derive(Debug)]
struct StreamingFrameResult {
    sequence_gap: Option<(u32, u32)>,
    speech_start: Option<SpeechStartEvent>,
    speech_rejected: Option<SpeechStartEvent>,
    finalized_utterance: Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>,
}

impl StreamingInputState {
    fn new(config: StreamingInputConfig) -> Self {
        let endpoint = EndpointDetector::new(config.endpoint_config());
        Self {
            config,
            vad_scorer: VadScorer::new(),
            endpoint,
            next_utterance_seq: 0,
            frame_seq_last: None,
            current_sample_rate: None,
            pre_roll: VecDeque::new(),
            active: None,
            last_frame_at: None,
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
        let sequence_gap = if let Some(last) = self.frame_seq_last {
            if frame_seq <= last {
                return Err(format!(
                    "Stale or duplicate input frame sequence {frame_seq}; last accepted was {last}"
                ));
            }
            (frame_seq > last.saturating_add(1)).then_some((last.saturating_add(1), frame_seq))
        } else {
            None
        };
        self.frame_seq_last = Some(frame_seq);
        self.last_frame_at = Some(Instant::now());

        if payload.is_empty() {
            return Ok(StreamingFrameResult {
                sequence_gap,
                speech_start: None,
                speech_rejected: None,
                finalized_utterance: None,
            });
        }
        if payload.len() % 2 != 0 {
            return Err("PCM16 payload length must be even".to_string());
        }

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
                sequence_gap,
                speech_start: None,
                speech_rejected: None,
                finalized_utterance: None,
            });
        }

        let frame_ms = (samples.len() as f32 * 1000.0) / (sample_rate as f32);
        let vad_frames = self
            .vad_scorer
            .push_i16(&samples, sample_rate)
            .map_err(|err| format!("VAD scoring error: {err}"))?;
        let mut chunk_has_speech = false;
        let mut chunk_voiced_ms = 0.0f32;
        let mut speech_started = false;
        let mut noise_rejected = false;
        let mut end_reason: Option<UtteranceEndReason> = None;

        for vad_frame in &vad_frames {
            let decision = self.endpoint.process_score(vad_frame.score, VAD_FRAME_MS);
            if decision.is_speech {
                chunk_has_speech = true;
                chunk_voiced_ms += VAD_FRAME_MS;
            }
            for event in decision.events {
                match event {
                    EndpointEvent::SpeechStart => speech_started = true,
                    EndpointEvent::SpeechEnd(reason) => end_reason = Some(reason.into()),
                    EndpointEvent::NoiseRejected => noise_rejected = true,
                }
            }
        }

        let mut result = StreamingFrameResult {
            sequence_gap,
            speech_start: None,
            speech_rejected: None,
            finalized_utterance: None,
        };

        if speech_started && self.active.is_none() {
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
                started_at: Instant::now(),
            };
            if !self.pre_roll.is_empty() {
                capture.samples_i16.extend(self.pre_roll.iter().copied());
            }
            if capture.samples_i16.len().saturating_add(samples.len()) > MAX_UTTERANCE_PCM16_SAMPLES
            {
                return Err("Streamed utterance exceeded the PCM16 buffer limit".to_string());
            }
            capture.samples_i16.extend_from_slice(&samples);
            self.active = Some(capture);

            result.speech_start = Some(SpeechStartEvent {
                utterance_id,
                utterance_seq,
            });
        } else if let Some(active) = self.active.as_mut() {
            if active.samples_i16.len().saturating_add(samples.len()) > MAX_UTTERANCE_PCM16_SAMPLES
            {
                return Err("Streamed utterance exceeded the PCM16 buffer limit".to_string());
            }
            active.samples_i16.extend_from_slice(&samples);
        } else {
            self.push_pre_roll(&samples, sample_rate);
            return Ok(result);
        }

        if let Some(active) = self.active.as_mut() {
            active.voiced_ms += chunk_voiced_ms;
            active.total_ms += frame_ms;
            if chunk_has_speech {
                active.silence_ms = 0.0;
            } else {
                active.silence_ms += frame_ms;
            }
        }

        if let Some(reason) = end_reason {
            result.finalized_utterance = self.finalize_active_utterance(reason)?;
        } else if noise_rejected {
            result.speech_rejected = self.reject_active_utterance();
        }

        if !chunk_has_speech {
            self.push_pre_roll(&samples, sample_rate);
        }

        Ok(result)
    }

    fn reject_active_utterance(&mut self) -> Option<SpeechStartEvent> {
        let active = self.active.take()?;
        Some(SpeechStartEvent {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
        })
    }

    fn finish_stream(
        &mut self,
    ) -> Result<Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>, String> {
        match self.endpoint.finish() {
            Some(EndpointEvent::SpeechEnd(EndpointEndReason::StreamStopped)) => {
                self.finalize_active_utterance(UtteranceEndReason::StreamStopped)
            }
            Some(EndpointEvent::NoiseRejected) => {
                self.active = None;
                Ok(None)
            }
            _ => self.finalize_active_utterance(UtteranceEndReason::StreamStopped),
        }
    }

    fn on_tick(
        &mut self,
    ) -> Result<Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>, String> {
        let Some(active) = self.active.as_ref() else {
            return Ok(None);
        };
        if active.started_at.elapsed() >= Duration::from_millis(self.config.max_utterance_ms as u64)
        {
            return self.finalize_active_utterance(UtteranceEndReason::MaxDuration);
        }

        let pause_limit = Duration::from_millis(self.config.silence_duration_ms.max(250) as u64);
        if self
            .last_frame_at
            .is_some_and(|last_frame| last_frame.elapsed() >= pause_limit)
        {
            return self.finalize_active_utterance(UtteranceEndReason::ClientPause);
        }
        Ok(None)
    }

    fn finalize_active_utterance(
        &mut self,
        reason: UtteranceEndReason,
    ) -> Result<Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        self.endpoint = EndpointDetector::new(self.config.endpoint_config());
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

        self.pre_roll.extend(samples.iter().copied());
        if self.pre_roll.len() > max_samples {
            let drain = self.pre_roll.len() - max_samples;
            self.pre_roll.drain(0..drain);
        }
    }
}

async fn finalize_stream_vad_utterance(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    conn: &mut ConnectionState,
    commit: PendingAudioCommit,
    wav_bytes: Vec<u8>,
    end_reason: UtteranceEndReason,
) -> Result<(), String> {
    if interrupt_active_turn(out_tx, &mut conn.active_turn, "preempted_by_new_turn").await {
        state.runtime.record_voice_interruption();
        conn.voice_session.interrupt("preempted_by_new_turn");
    }

    let voice_session_id = conn
        .voice_session_id
        .clone()
        .ok_or_else(|| "Voice session not initialized".to_string())?;
    let voice_profile_id = conn.voice_profile_id.clone();
    let model_ids = commit.turn_config.model_ids();
    let turn_record = state
        .voice_store
        .create_turn(CreateVoiceTurnRequest {
            session_id: voice_session_id.clone(),
            utterance_id: commit.utterance_id.clone(),
            utterance_seq: commit.utterance_seq,
            mode: commit.turn_config.mode_str().to_string(),
            vad_end_reason: Some(end_reason.as_str().to_string()),
            asr_model_id: model_ids.asr_model_id,
            text_model_id: model_ids.text_model_id,
            tts_model_id: model_ids.tts_model_id,
            s2s_model_id: model_ids.s2s_model_id,
            speaker: commit.turn_config.speaker(),
        })
        .await
        .map_err(|err| format!("Voice storage error: {err}"))?;

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

    let turn_record_id = turn_record.id.clone();
    conn.next_turn_generation = conn.next_turn_generation.saturating_add(1);
    let output_generation = conn.next_turn_generation;
    let control = TurnControl::new();
    let task = spawn_turn_task(
        state.clone(),
        correlation_id.to_string(),
        out_tx.clone(),
        commit.clone(),
        wav_bytes,
        agent_session_id,
        voice_session_id,
        voice_profile_id,
        conn.system_prompt.clone(),
        turn_record_id.clone(),
        control.clone(),
        output_generation,
    );

    conn.active_turn = Some(ActiveTurn {
        utterance_id: commit.utterance_id,
        utterance_seq: commit.utterance_seq,
        turn_record_id: turn_record_id.clone(),
        state: state.clone(),
        task,
        control,
        output_generation,
    });
    conn.voice_session.begin_processing(turn_record_id);

    Ok(())
}

fn normalize_stream_vad_threshold(value: Option<f32>) -> f32 {
    value
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(sanitize_score_threshold)
        .unwrap_or(DEFAULT_STREAM_VAD_THRESHOLD)
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

pub async fn handle_socket(mut socket: WebSocket, state: AppState, correlation_id: String) {
    let Some(_session_permit) = state.try_acquire_realtime_session() else {
        let _ = socket
            .send(Message::Text(
                json!({
                    "type": "error",
                    "code": "realtime_session_capacity",
                    "message": "Realtime websocket session capacity is exhausted",
                    "fatal": true,
                })
                .to_string()
                .into(),
            ))
            .await;
        let _ = socket
            .send(Message::Close(Some(CloseFrame {
                code: 1013,
                reason: "realtime_session_capacity".into(),
            })))
            .await;
        return;
    };

    let (mut ws_tx, mut ws_rx) = socket.split();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice realtime");
    let writer_out_tx = out_tx.clone();

    let writer = tokio::spawn(async move {
        while let Some(output) = writer_out_tx.next().await {
            let message = match output {
                MailboxOutput::Message(message) => message,
                MailboxOutput::Close(reason) => Message::Close(Some(CloseFrame {
                    code: 1013,
                    reason: reason.into(),
                })),
            };
            match tokio::time::timeout(WS_WRITER_SEND_TIMEOUT, ws_tx.send(message)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => break,
            }
            if ws_tx.flush().await.is_err() {
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

    let mut actor_tick = tokio::time::interval(SESSION_ACTOR_TICK);
    actor_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    actor_tick.tick().await;
    loop {
        let result = tokio::select! {
            result = ws_rx.next() => result,
            _ = actor_tick.tick() => {
                let timed_utterance = match conn.streaming_input.as_mut() {
                    Some(streaming) => streaming.on_tick(),
                    None => Ok(None),
                };
                match timed_utterance {
                    Ok(Some((commit, wav_bytes, end_reason))) => {
                        send_json(
                            &out_tx,
                            json!({
                                "type": "user_speech_end",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "reason": end_reason.as_str(),
                            }),
                        );
                        if let Err(err) = finalize_stream_vad_utterance(
                            &state,
                            &correlation_id,
                            &out_tx,
                            &mut conn,
                            commit,
                            wav_bytes,
                            end_reason,
                        )
                        .await
                        {
                            send_error(&out_tx, None, None, err);
                        }
                    }
                    Ok(None) => {}
                    Err(err) => send_error(&out_tx, None, None, err),
                }
                continue;
            }
        };
        let Some(result) = result else {
            break;
        };
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
                let _ = out_tx.send_diagnostic(Message::Pong(payload));
            }
            Message::Pong(_) => {}
        }
    }

    if interrupt_active_turn(&out_tx, &mut conn.active_turn, "socket_closed").await {
        state.runtime.record_voice_interruption();
    }
    if let Some(session_id) = conn.close_voice_session_now() {
        if let Err(err) = state.voice_store.end_session(session_id).await {
            warn!("failed to end voice session on socket close: {err}");
        } else {
            state.runtime.record_voice_session_closed();
        }
    }
    out_tx.close();
    let _ = writer.await;
}

async fn handle_text_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    conn: &mut ConnectionState,
    text: &str,
) -> Result<(), String> {
    let event: ClientEvent =
        serde_json::from_str(text).map_err(|err| format!("Invalid websocket payload: {err}"))?;

    match event {
        ClientEvent::SessionStart { system_prompt } => {
            let profile = state
                .voice_store
                .get_default_profile()
                .await
                .map_err(|err| format!("Voice storage error: {err}"))?;
            conn.voice_profile_id = Some(profile.id.clone());

            if let Some(prompt) = system_prompt
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
            {
                if conn.agent_session_system_prompt.as_deref() != Some(prompt.as_str()) {
                    conn.agent_session_id = None;
                }
                conn.system_prompt = prompt;
            } else {
                conn.system_prompt = profile.system_prompt;
            }
            conn.started = true;
            let transport_session_id = conn.transport_session_id.clone();
            let owner_instance_id = conn.owner_instance_id.clone();
            send_json(
                out_tx,
                json!({
                    "type": "session_ready",
                    "protocol": "voice_realtime_v1",
                    "session_id": transport_session_id,
                    "owner_instance_id": owner_instance_id,
                    "connection_epoch": 0,
                    "resumable": false,
                    "resume_window_ms": 0,
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
            let normalized_speaker = speaker
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let normalized_asr_language = asr_language
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

                    let asr_variant = parse_model_variant(asr_model_id.as_str())
                        .map_err(|err| format!("Unsupported ASR model: {err}"))?;
                    if !asr_variant.is_asr() {
                        return Err(format!(
                            "Model `{asr_model_id}` is not a supported modular ASR model."
                        ));
                    }
                    let _ = resolve_chat_model_id(Some(text_model_id.as_str()))?;
                    parse_tts_model_variant(tts_model_id.as_str())
                        .map_err(|err| format!("Unsupported TTS model: {err}"))?;

                    VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
                        asr_model_id,
                        text_model_id,
                        tts_model_id,
                        speaker: normalized_speaker.clone(),
                        asr_language: normalized_asr_language,
                        max_output_tokens: max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                    })
                }
                RealtimeVoiceMode::Unified => {
                    let Some(s2s_model_id) = normalized_s2s else {
                        return Err(
                            "Missing required model id (`s2s_model_id`) for unified voice mode."
                                .to_string(),
                        );
                    };
                    let variant = parse_model_variant(s2s_model_id.as_str())
                        .map_err(|err| format!("Unsupported unified audio model: {err}"))?;
                    if !variant.is_audio_chat() {
                        return Err(format!(
                            "Model `{}` is not a supported unified audio-chat model.",
                            s2s_model_id
                        ));
                    }
                    VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
                        s2s_model_id,
                        speaker: normalized_speaker,
                        max_output_tokens: max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                    })
                }
            };

            if conn.voice_session_id.is_none() {
                let profile_id = conn.voice_profile_id.clone().ok_or_else(|| {
                    "Voice profile not initialized. Send `session_start` first.".to_string()
                })?;
                let session = state
                    .voice_store
                    .create_session(crate::voice_store::CreateVoiceSessionRequest {
                        profile_id,
                        mode: mode.as_str().to_string(),
                        system_prompt: conn.system_prompt.clone(),
                    })
                    .await
                    .map_err(|err| format!("Voice storage error: {err}"))?;
                conn.voice_session.start(session.id.clone());
                conn.voice_session_id = Some(session.id);
                state.runtime.record_voice_session_started();
            }

            conn.streaming_input = Some(StreamingInputState::new(StreamingInputConfig {
                turn_config,
                vad_threshold: normalize_stream_vad_threshold(vad_threshold),
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
            conn.voice_session.begin_listening();

            send_json(
                out_tx,
                json!({
                    "type": "input_stream_ready",
                    "vad": {
                        "backend": "earshot",
                        "threshold": conn.streaming_input.as_ref().map(|s| s.config.vad_threshold),
                        "score_sample_rate": VAD_SAMPLE_RATE,
                        "score_frame_ms": VAD_FRAME_MS,
                        "min_speech_ms": conn.streaming_input.as_ref().map(|s| s.config.min_speech_ms),
                        "silence_duration_ms": conn.streaming_input.as_ref().map(|s| s.config.silence_duration_ms),
                    }
                }),
            );
        }
        ClientEvent::InputStreamStop => {
            if let Some(mut streaming) = conn.streaming_input.take() {
                if let Some((commit, wav_bytes, end_reason)) = streaming.finish_stream()? {
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
                        end_reason,
                    )
                    .await?;
                }
            }
            send_json(out_tx, json!({ "type": "input_stream_stopped" }));
            if let Some(session_id) = conn.close_voice_session_if_idle() {
                state
                    .voice_store
                    .end_session(session_id)
                    .await
                    .map_err(|err| format!("Voice storage error: {err}"))?;
                state.runtime.record_voice_session_closed();
            }
        }
        ClientEvent::Interrupt { reason } => {
            let reason = reason.unwrap_or_else(|| "client_interrupt".to_string());
            if interrupt_active_turn(out_tx, &mut conn.active_turn, &reason).await {
                state.runtime.record_voice_interruption();
                conn.voice_session.interrupt(reason);
            }
        }
        ClientEvent::Ping { timestamp_ms } => {
            send_diagnostic_json(
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
    let _ = correlation_id;
    Ok(())
}

async fn handle_binary_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
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

            if let Some((expected, received)) = frame_result.sequence_gap {
                send_diagnostic_json(
                    out_tx,
                    json!({
                        "type": "input_sequence_gap",
                        "expected_frame_sequence": expected,
                        "received_frame_sequence": received,
                        "missing_frames": received.saturating_sub(expected),
                        "action": "continue",
                    }),
                );
            }

            if let Some(evt) = frame_result.speech_start {
                if interrupt_active_turn(out_tx, &mut conn.active_turn, "barge_in").await {
                    state.runtime.record_voice_interruption();
                    state.runtime.record_voice_barge_in();
                    conn.voice_session.interrupt("barge_in");
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
                    end_reason,
                )
                .await?;
            }

            if let Some(evt) = frame_result.speech_rejected {
                send_json(
                    out_tx,
                    json!({
                        "type": "user_speech_rejected",
                        "utterance_id": evt.utterance_id,
                        "utterance_seq": evt.utterance_seq,
                    }),
                );
            }

            return Ok(());
        }
    }
}

fn spawn_turn_task(
    state: AppState,
    correlation_id: String,
    out_tx: OutboundTx,
    commit: PendingAudioCommit,
    audio_bytes: Vec<u8>,
    agent_session_id: Option<String>,
    voice_session_id: String,
    voice_profile_id: Option<String>,
    system_prompt: String,
    turn_record_id: String,
    control: TurnControl,
    output_generation: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let timeout_secs = state.request_timeout_secs.max(1);
        let timeout = Duration::from_secs(timeout_secs);

        let turn_future = async {
            let permit = state
                .acquire_owned_workload_permit(WorkloadClass::Realtime)
                .await
                .map_err(|_| "Server is shutting down".to_string())?;
            let runtime_context = permit.runtime_context();

            send_turn_json(
                &out_tx,
                output_generation,
                json!({
                    "type": "turn_processing",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );
            match &commit.turn_config {
                VoiceTurnConfig::Modular(config) => {
                    state.runtime.record_modular_voice_pipeline_turn();
                    send_turn_json(
                        &out_tx,
                        output_generation,
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
                        let mut snapshot = String::new();
                        state
                            .runtime
                            .asr_transcribe_streaming_bytes_with_runtime_context(
                                audio_bytes.as_slice(),
                                Some(&asr_model_id),
                                asr_language.as_deref(),
                                None,
                                None,
                                Some(&correlation_id),
                                runtime_context,
                                move |delta| {
                                    if delta.is_empty() {
                                        return;
                                    }
                                    snapshot.push_str(&delta);
                                    send_turn_text_snapshot(
                                        &tx,
                                        output_generation,
                                        format!("user-transcript-{utt_seq}"),
                                        json!({
                                            "type": "user_transcript_snapshot",
                                            "utterance_id": utt_id,
                                            "utterance_seq": utt_seq,
                                            "text": snapshot,
                                        }),
                                    );
                                },
                            )
                            .await
                            .map_err(|err| format!("ASR failed: {err}"))?
                    };

                    let user_text = transcript.text.trim().to_string();
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "user_transcript_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": user_text.clone(),
                            "language": transcript.language.clone(),
                            "audio_duration_secs": transcript.duration_secs,
                        }),
                    );
                    state
                        .voice_store
                        .update_turn_transcript(
                            turn_record_id.clone(),
                            user_text.clone(),
                            transcript.language.clone(),
                            Some(transcript.duration_secs),
                        )
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;

                    if user_text.is_empty() {
                        state
                            .voice_store
                            .complete_turn(turn_record_id.clone(), "no_input", None)
                            .await
                            .map_err(|err| format!("Voice storage error: {err}"))?;
                        send_turn_json(
                            &out_tx,
                            output_generation,
                            json!({
                                "type": "turn_done",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "status": "no_input",
                            }),
                        );
                        return Ok::<(), String>(());
                    }

                    send_turn_json(
                        &out_tx,
                        output_generation,
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
                        voice_profile_id.as_deref(),
                        &correlation_id,
                        runtime_context,
                    )
                    .await?;
                    let assistant_text = strip_think_tags(&assistant_raw);

                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "assistant_text_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": assistant_text.clone(),
                            "raw_text": assistant_raw.clone(),
                        }),
                    );
                    state
                        .voice_store
                        .update_turn_assistant(
                            turn_record_id.clone(),
                            Some(assistant_text.clone()),
                            Some(assistant_raw.clone()),
                        )
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;

                    if assistant_text.is_empty() {
                        state
                            .voice_store
                            .complete_turn(turn_record_id.clone(), "ok", None)
                            .await
                            .map_err(|err| format!("Voice storage error: {err}"))?;
                        send_turn_json(
                            &out_tx,
                            output_generation,
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
                        runtime_context,
                        &control,
                        output_generation,
                    )
                    .await?;
                    state
                        .voice_store
                        .complete_turn(turn_record_id.clone(), "ok", None)
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;
                    maybe_spawn_observation_extraction(
                        state.clone(),
                        voice_profile_id.clone(),
                        voice_session_id.clone(),
                        turn_record_id.clone(),
                        config.text_model_id.clone(),
                        correlation_id.clone(),
                        user_text.clone(),
                        assistant_text.clone(),
                    );

                    send_turn_json(
                        &out_tx,
                        output_generation,
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
                    state.runtime.record_unified_voice_pipeline_turn();
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "user_transcript_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "assistant_text_start",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                        }),
                    );

                    let history_messages =
                        build_unified_history(&state, &voice_session_id, &turn_record_id).await?;
                    let generation = stream_unified_s2s_to_socket(
                        &state,
                        &out_tx,
                        &correlation_id,
                        &commit.utterance_id,
                        commit.utterance_seq,
                        &config.s2s_model_id,
                        config.speaker.clone(),
                        &system_prompt,
                        history_messages,
                        config.max_output_tokens,
                        audio_bytes.as_slice(),
                        runtime_context,
                        &control,
                        output_generation,
                    )
                    .await?;

                    let user_text = generation
                        .input_transcription
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "user_transcript_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": user_text.clone(),
                            "language": null,
                            "audio_duration_secs": null,
                        }),
                    );
                    state
                        .voice_store
                        .update_turn_transcript(
                            turn_record_id.clone(),
                            user_text.clone(),
                            None,
                            None,
                        )
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;

                    let assistant_text = strip_think_tags(generation.text.trim());
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "assistant_text_final",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": assistant_text.clone(),
                            "raw_text": generation.text.clone(),
                        }),
                    );
                    state
                        .voice_store
                        .update_turn_assistant(
                            turn_record_id.clone(),
                            Some(assistant_text.clone()),
                            Some(generation.text.clone()),
                        )
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;

                    if user_text.is_empty() && assistant_text.is_empty() {
                        state
                            .voice_store
                            .complete_turn(turn_record_id.clone(), "no_input", None)
                            .await
                            .map_err(|err| format!("Voice storage error: {err}"))?;
                        send_turn_json(
                            &out_tx,
                            output_generation,
                            json!({
                                "type": "turn_done",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "status": "no_input",
                            }),
                        );
                        return Ok(());
                    }

                    state
                        .voice_store
                        .complete_turn(turn_record_id.clone(), "ok", None)
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;
                    send_turn_json(
                        &out_tx,
                        output_generation,
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

        let turn_result = tokio::select! {
            _ = control.cancelled() => return,
            result = tokio::time::timeout(timeout, turn_future) => result,
        };

        match turn_result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                let _ = state
                    .voice_store
                    .complete_turn(turn_record_id.clone(), "error", Some(err.clone()))
                    .await;
                send_turn_error(
                    &out_tx,
                    output_generation,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    err,
                );
                send_turn_json(
                    &out_tx,
                    output_generation,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "error",
                    }),
                );
            }
            Err(_) => {
                let timeout_reason = format!("Turn timed out after {timeout_secs} seconds");
                let _ = state
                    .voice_store
                    .complete_turn(
                        turn_record_id.clone(),
                        "timeout",
                        Some(timeout_reason.clone()),
                    )
                    .await;
                send_turn_error(
                    &out_tx,
                    output_generation,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    timeout_reason,
                );
                send_turn_json(
                    &out_tx,
                    output_generation,
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
    out_tx: &OutboundTx,
    correlation_id: &str,
    utterance_id: &str,
    utterance_seq: u64,
    tts_model_id: &str,
    speaker: Option<String>,
    text: &str,
    runtime_context: RuntimeRequestContext,
    control: &TurnControl,
    output_generation: u64,
) -> Result<(), String> {
    if control.is_cancelled() {
        return Err("Turn cancelled".to_string());
    }
    let tts_variant = parse_tts_model_variant(tts_model_id)
        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
    state
        .runtime
        .load_model(tts_variant)
        .await
        .map_err(|err| format!("Failed to load TTS model: {err}"))?;

    let sample_rate = state.runtime.sample_rate().await;
    let encoder = AudioEncoder::new(sample_rate, 1);

    send_turn_json(
        out_tx,
        output_generation,
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
        model_variant: Some(tts_variant),
        correlation_id: Some(correlation_id.to_string()),
        runtime_context,
        text: text.to_string(),
        config: gen_config,
        language: None,
        reference_audio: None,
        reference_text: None,
        voice_description: None,
    };

    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<izwi_core::AudioChunk>(32);
    let runtime = state.runtime.clone();
    let generation_task = AbortOnDropTask::new(tokio::spawn(async move {
        runtime.generate_streaming(gen_request, chunk_tx).await
    }));

    loop {
        let chunk = tokio::select! {
            _ = control.cancelled() => return Err("Turn cancelled".to_string()),
            chunk = chunk_rx.recv() => chunk,
        };
        let Some(chunk) = chunk else {
            break;
        };
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
        let sample_count = encoded.len() / std::mem::size_of::<i16>();
        let duration_ms = (sample_count as u64)
            .saturating_mul(1_000)
            .checked_div(sample_rate.max(1) as u64)
            .unwrap_or(0);
        out_tx
            .send_audio(
                output_generation,
                Message::Binary(frame.into()),
                encoded.len(),
                duration_ms,
            )
            .map_err(|err| format!("Outbound audio delivery failed: {err:?}"))?;
    }

    match generation_task.join().await {
        Ok(Ok(())) => {
            send_turn_json(
                out_tx,
                output_generation,
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

async fn build_unified_history(
    state: &AppState,
    voice_session_id: &str,
    current_turn_id: &str,
) -> Result<Vec<ChatMessage>, String> {
    let session = state
        .voice_store
        .get_session(voice_session_id.to_string())
        .await
        .map_err(|err| format!("Voice storage error: {err}"))?
        .ok_or_else(|| "Voice session not found".to_string())?;

    let mut messages = Vec::new();
    for turn in session.turns {
        if turn.id == current_turn_id {
            continue;
        }
        if let Some(user_text) = turn.user_text.as_deref().map(str::trim) {
            if !user_text.is_empty() {
                messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: user_text.to_string(),
                });
            }
        }
        if let Some(assistant_text) = turn.assistant_text.as_deref().map(str::trim) {
            if !assistant_text.is_empty() {
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: assistant_text.to_string(),
                });
            }
        }
    }

    Ok(messages)
}

#[allow(clippy::too_many_arguments)]
async fn stream_unified_s2s_to_socket(
    state: &AppState,
    out_tx: &OutboundTx,
    correlation_id: &str,
    utterance_id: &str,
    utterance_seq: u64,
    s2s_model_id: &str,
    speaker: Option<String>,
    system_prompt: &str,
    history_messages: Vec<ChatMessage>,
    max_output_tokens: usize,
    audio_bytes: &[u8],
    runtime_context: RuntimeRequestContext,
    control: &TurnControl,
    output_generation: u64,
) -> Result<izwi_core::SpeechToSpeechGeneration, String> {
    if control.is_cancelled() {
        return Err("Turn cancelled".to_string());
    }
    let variant = parse_model_variant(s2s_model_id)
        .map_err(|err| format!("Unsupported unified audio model: {err}"))?;
    if !variant.is_audio_chat() {
        return Err(format!(
            "Model `{s2s_model_id}` is not a supported unified audio-chat model."
        ));
    }

    state
        .runtime
        .load_model(variant)
        .await
        .map_err(|err| format!("Failed to load unified audio model: {err}"))?;

    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "assistant_audio_start",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "sample_rate": 24_000,
            "audio_format": "pcm_i16",
        }),
    );

    let encoder = AudioEncoder::new(24_000, 1);
    let delivery_error = Arc::new(Mutex::new(None::<String>));
    let mut params = GenerationParams::default();
    params.max_tokens = max_output_tokens.clamp(1, 4096);
    params.speaker = speaker.clone();
    params.voice = speaker;

    let generation = state
        .runtime
        .speech_to_speech_generate_streaming_bytes_with_variant_and_runtime_context(
            variant,
            audio_bytes,
            history_messages,
            params,
            Some(system_prompt),
            Some(correlation_id),
            runtime_context,
            {
                let out_tx = out_tx.clone();
                let utterance_id = utterance_id.to_string();
                let control = control.clone();
                let delivery_error = delivery_error.clone();
                let mut text_snapshot = String::new();
                move |chunk| {
                    if control.is_cancelled() {
                        return;
                    }
                    if let Some(delta) = chunk.text.as_deref() {
                        if !delta.is_empty() {
                            text_snapshot.push_str(delta);
                            send_turn_text_snapshot(
                                &out_tx,
                                output_generation,
                                format!("assistant-text-{utterance_seq}"),
                                json!({
                                    "type": "assistant_text_snapshot",
                                    "utterance_id": utterance_id,
                                    "utterance_seq": utterance_seq,
                                    "text": text_snapshot,
                                }),
                            );
                        }
                    }

                    if !chunk.samples.is_empty() {
                        match encoder.encode(&chunk.samples, AudioFormat::RawI16) {
                            Ok(encoded) => {
                                let chunk_seq = u32::try_from(chunk.sequence).unwrap_or(u32::MAX);
                                let frame = encode_assistant_audio_binary_frame(
                                    utterance_seq,
                                    chunk_seq,
                                    chunk.sample_rate.max(1),
                                    chunk.is_final,
                                    &encoded,
                                );
                                let sample_count = encoded.len() / std::mem::size_of::<i16>();
                                let duration_ms = (sample_count as u64)
                                    .saturating_mul(1_000)
                                    .checked_div(chunk.sample_rate.max(1) as u64)
                                    .unwrap_or(0);
                                if let Err(err) = out_tx.send_audio(
                                    output_generation,
                                    Message::Binary(frame.into()),
                                    encoded.len(),
                                    duration_ms,
                                ) {
                                    *delivery_error
                                        .lock()
                                        .unwrap_or_else(|poison| poison.into_inner()) =
                                        Some(format!("Outbound audio delivery failed: {err:?}"));
                                    control.cancel();
                                }
                            }
                            Err(err) => {
                                send_turn_error(
                                    &out_tx,
                                    output_generation,
                                    Some(utterance_id.clone()),
                                    Some(utterance_seq),
                                    format!("Failed to encode unified streamed audio chunk: {err}"),
                                );
                            }
                        }
                    }
                }
            },
        )
        .await
        .map_err(|err| format!("Unified speech-to-speech failed: {err}"))?;

    if let Some(err) = delivery_error
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .take()
    {
        return Err(err);
    }

    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "assistant_audio_done",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
        }),
    );

    Ok(generation)
}

async fn interrupt_active_turn(
    out_tx: &OutboundTx,
    active_turn: &mut Option<ActiveTurn>,
    reason: &str,
) -> bool {
    if let Some(mut turn) = active_turn.take() {
        out_tx.cutoff_turn(turn.output_generation);
        if turn.task.is_finished() {
            let _ = turn.task.await;
            return false;
        }

        turn.control.cancel();
        send_json(
            out_tx,
            json!({
                "type": "turn_interrupted",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
                "reason": reason,
            }),
        );
        if tokio::time::timeout(TURN_CANCELLATION_TIMEOUT, &mut turn.task)
            .await
            .is_err()
        {
            turn.task.abort();
            let _ = turn.task.await;
        }

        let _ = tokio::time::timeout(
            TURN_CANCELLATION_TIMEOUT,
            turn.state.voice_store.complete_turn(
                turn.turn_record_id.clone(),
                "interrupted",
                Some(reason.to_string()),
            ),
        )
        .await;
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
        return true;
    }
    false
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
    let session_id = new_uuid();
    let record = StoredAgentSessionRecord {
        id: session_id.clone(),
        agent_id: DEFAULT_VOICE_AGENT_ID.to_string(),
        thread_id: thread.id,
        model_id,
        system_prompt: system_prompt.to_string(),
        planning_mode: PlanningMode::Auto,
        created_at: now,
        updated_at: now,
    };

    state.store_agent_session_record(record).await;

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
    voice_profile_id: Option<&str>,
    correlation_id: &str,
    runtime_context: RuntimeRequestContext,
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
        name: DEFAULT_VOICE_AGENT_NAME.to_string(),
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

    let memory_context = if let Some(profile_id) = voice_profile_id {
        match state
            .voice_store
            .get_profile(profile_id.to_string())
            .await
            .map_err(|err| format!("Voice storage error: {err}"))?
        {
            Some(profile) if profile.observational_memory_enabled => state
                .voice_observation_store
                .build_context(profile.id, 8)
                .await
                .map_err(|err| format!("Voice memory storage error: {err}"))?,
            _ => None,
        }
    } else {
        None
    };

    let memory = VoiceContextMemoryStore::new(state.chat_store.clone(), memory_context);
    let backend = IzwiRuntimeBackend {
        runtime: state.runtime.clone(),
        correlation_id: correlation_id.to_string(),
        runtime_context,
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

    state
        .touch_agent_session_record(session_id, now_unix_millis(), resolved_model_id)
        .await;

    Ok(result.assistant_text)
}

fn maybe_spawn_observation_extraction(
    state: AppState,
    voice_profile_id: Option<String>,
    _voice_session_id: String,
    turn_record_id: String,
    model_id: String,
    correlation_id: String,
    user_text: String,
    assistant_text: String,
) {
    let Some(profile_id) = voice_profile_id else {
        return;
    };
    if user_text.trim().is_empty() {
        return;
    }

    tokio::spawn(async move {
        let profile = match state.voice_store.get_profile(profile_id.clone()).await {
            Ok(Some(profile)) => profile,
            Ok(None) => return,
            Err(err) => {
                warn!("failed to load voice profile for memory extraction: {err}");
                return;
            }
        };
        if !profile.observational_memory_enabled {
            return;
        }

        let candidates = match extract_observation_candidates(
            &state,
            model_id.as_str(),
            correlation_id.as_str(),
            user_text.as_str(),
            assistant_text.as_str(),
        )
        .await
        {
            Ok(candidates) => candidates,
            Err(err) => {
                warn!("observation extraction failed: {err}");
                return;
            }
        };
        if candidates.is_empty() {
            return;
        }

        if let Err(err) = state
            .voice_observation_store
            .upsert_candidates(
                profile.id,
                Some(turn_record_id),
                Some(user_text),
                Some(assistant_text),
                candidates,
            )
            .await
        {
            warn!("failed to persist extracted voice observations: {err}");
        }
    });
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

fn send_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send(Message::Text(text.into())),
        Err(err) => {
            warn!("failed to serialize voice ws event: {err}");
            false
        }
    }
}

fn send_turn_json(out_tx: &OutboundTx, output_generation: u64, value: serde_json::Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send_turn(output_generation, Message::Text(text.into())),
        Err(err) => {
            warn!("failed to serialize voice ws turn event: {err}");
            false
        }
    }
}

fn send_turn_text_snapshot(
    out_tx: &OutboundTx,
    output_generation: u64,
    key: String,
    value: serde_json::Value,
) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send_text_snapshot(output_generation, key, Message::Text(text.into())),
        Err(err) => {
            warn!("failed to serialize voice ws text snapshot: {err}");
            false
        }
    }
}

fn send_diagnostic_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send_diagnostic(Message::Text(text.into())),
        Err(err) => {
            warn!("failed to serialize voice ws diagnostic: {err}");
            false
        }
    }
}

fn send_error(
    out_tx: &OutboundTx,
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

fn send_turn_error(
    out_tx: &OutboundTx,
    output_generation: u64,
    utterance_id: Option<String>,
    utterance_seq: Option<u64>,
    message: impl Into<String>,
) {
    let message = message.into();
    let _ = send_turn_json(
        out_tx,
        output_generation,
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

struct VoiceContextMemoryStore {
    chat_store: Arc<ChatStore>,
    memory_context: Option<String>,
}

impl VoiceContextMemoryStore {
    fn new(chat_store: Arc<ChatStore>, memory_context: Option<String>) -> Self {
        Self {
            chat_store,
            memory_context,
        }
    }
}

#[async_trait::async_trait]
impl MemoryStore for VoiceContextMemoryStore {
    async fn load_messages(&self, thread_id: &str) -> izwi_agent::Result<Vec<MemoryMessage>> {
        let records = self
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;

        let mut out =
            Vec::with_capacity(records.len() + usize::from(self.memory_context.is_some()));
        if let Some(memory_context) = self.memory_context.clone() {
            out.push(MemoryMessage {
                role: MemoryMessageRole::System,
                content: memory_context,
            });
        }
        for record in records {
            let role = match record.role.as_str() {
                "system" => MemoryMessageRole::System,
                "user" => MemoryMessageRole::User,
                "assistant" => MemoryMessageRole::Assistant,
                other => {
                    return Err(izwi_agent::AgentError::Memory(format!(
                        "Invalid stored chat role: {other}"
                    )));
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
    runtime_context: RuntimeRequestContext,
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
            .chat_generate_with_correlation_and_runtime_context(
                variant,
                runtime_messages,
                request.max_output_tokens.clamp(1, 4096),
                Some(&self.correlation_id),
                self.runtime_context,
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

#[cfg(test)]
mod tests {
    use axum::extract::ws::Message;

    use super::*;

    fn test_streaming_input() -> StreamingInputState {
        StreamingInputState::new(StreamingInputConfig {
            turn_config: VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
                s2s_model_id: "LFM2.5-Audio-1.5B".to_string(),
                speaker: None,
                max_output_tokens: 32,
            }),
            vad_threshold: DEFAULT_STREAM_VAD_THRESHOLD,
            min_speech_ms: 50,
            silence_duration_ms: 200,
            max_utterance_ms: 10_000,
            pre_roll_ms: 100,
            input_sample_rate_hint: Some(16_000),
        })
    }

    #[tokio::test]
    async fn outbound_mailbox_coalesces_text_snapshots() {
        let mailbox = OutboundMailbox::new(OutboundLimits {
            max_items: 4,
            max_audio_bytes: 100,
            max_audio_ms: 100,
        });

        for text in ["first", "latest"] {
            mailbox
                .enqueue(OutboundItem {
                    message: Message::Text(text.into()),
                    class: OutboundClass::CoalescibleText("transcript-1".to_string()),
                    turn_generation: Some(1),
                })
                .expect("snapshot should enqueue");
        }

        let Some(MailboxOutput::Message(Message::Text(text))) = mailbox.next().await else {
            panic!("latest text snapshot should be delivered");
        };
        assert_eq!(text.as_str(), "latest");
    }

    #[tokio::test]
    async fn outbound_mailbox_preserves_audio_order_and_closes_on_saturation() {
        let ordered = OutboundMailbox::new(OutboundLimits {
            max_items: 4,
            max_audio_bytes: 8,
            max_audio_ms: 100,
        });
        for byte in [1_u8, 2_u8] {
            ordered
                .enqueue(OutboundItem {
                    message: Message::Binary(vec![byte].into()),
                    class: OutboundClass::Audio {
                        bytes: 1,
                        duration_ms: 10,
                    },
                    turn_generation: Some(1),
                })
                .expect("audio should enqueue");
        }
        for expected in [1_u8, 2_u8] {
            let Some(MailboxOutput::Message(Message::Binary(bytes))) = ordered.next().await else {
                panic!("audio should be delivered in order");
            };
            assert_eq!(bytes.as_ref(), &[expected]);
        }

        let saturated = OutboundMailbox::new(OutboundLimits {
            max_items: 4,
            max_audio_bytes: 1,
            max_audio_ms: 10,
        });
        saturated
            .enqueue(OutboundItem {
                message: Message::Binary(vec![1].into()),
                class: OutboundClass::Audio {
                    bytes: 1,
                    duration_ms: 10,
                },
                turn_generation: Some(1),
            })
            .expect("first audio should enqueue");
        assert_eq!(
            saturated.enqueue(OutboundItem {
                message: Message::Binary(vec![2].into()),
                class: OutboundClass::Audio {
                    bytes: 1,
                    duration_ms: 1,
                },
                turn_generation: Some(1),
            }),
            Err(OutboundEnqueueError::AudioSaturated)
        );
        assert!(matches!(
            saturated.next().await,
            Some(MailboxOutput::Close(reason)) if reason == "outbound_audio_saturated"
        ));
    }

    #[tokio::test]
    async fn outbound_mailbox_drops_diagnostics_and_cuts_off_cancelled_turns() {
        let mailbox = OutboundMailbox::new(OutboundLimits {
            max_items: 2,
            max_audio_bytes: 100,
            max_audio_ms: 100,
        });
        mailbox
            .enqueue(OutboundItem {
                message: Message::Text("cancelled".into()),
                class: OutboundClass::Critical,
                turn_generation: Some(7),
            })
            .expect("cancelled turn event should initially enqueue");
        mailbox
            .enqueue(OutboundItem {
                message: Message::Text("next".into()),
                class: OutboundClass::Critical,
                turn_generation: Some(8),
            })
            .expect("next turn event should enqueue");
        assert_eq!(
            mailbox.enqueue(OutboundItem {
                message: Message::Text("diagnostic".into()),
                class: OutboundClass::Diagnostic,
                turn_generation: None,
            }),
            Err(OutboundEnqueueError::DroppedDiagnostic)
        );

        mailbox.cutoff_turn(7);
        assert_eq!(
            mailbox.enqueue(OutboundItem {
                message: Message::Text("late".into()),
                class: OutboundClass::Critical,
                turn_generation: Some(7),
            }),
            Err(OutboundEnqueueError::Cutoff)
        );
        let Some(MailboxOutput::Message(Message::Text(text))) = mailbox.next().await else {
            panic!("next turn output should remain queued");
        };
        assert_eq!(text.as_str(), "next");
    }

    #[tokio::test]
    async fn turn_control_wakes_cooperative_cleanup() {
        let control = TurnControl::new();
        let waiting_control = control.clone();
        let waiter = tokio::spawn(async move {
            waiting_control.cancelled().await;
            waiting_control.is_cancelled()
        });

        control.cancel();

        assert!(tokio::time::timeout(Duration::from_millis(50), waiter)
            .await
            .expect("cancellation should wake waiter")
            .expect("waiter should finish"));
    }

    #[test]
    fn stream_vad_threshold_uses_score_values_directly() {
        assert_eq!(normalize_stream_vad_threshold(Some(0.02)), 0.02);
    }

    #[test]
    fn stream_vad_threshold_accepts_score_values_directly() {
        assert_eq!(normalize_stream_vad_threshold(Some(0.62)), 0.62);
    }

    #[test]
    fn streaming_input_rejects_stale_frames_and_reports_gaps() {
        let mut input = test_streaming_input();
        let silence = vec![0_u8; 640];

        let first = input
            .handle_pcm16_frame(1, 16_000, &silence)
            .expect("first frame");
        assert_eq!(first.sequence_gap, None);
        let gap = input
            .handle_pcm16_frame(3, 16_000, &silence)
            .expect("gap frame");
        assert_eq!(gap.sequence_gap, Some((2, 3)));
        assert!(input
            .handle_pcm16_frame(3, 16_000, &silence)
            .expect_err("duplicate frame must fail")
            .contains("Stale or duplicate"));
    }

    #[test]
    fn streaming_pre_roll_is_a_bounded_ring() {
        let mut input = test_streaming_input();
        input.push_pre_roll(&vec![1_i16; 8_000], 16_000);

        assert_eq!(input.pre_roll.len(), 1_600);
        assert!(input.pre_roll.iter().all(|sample| *sample == 1));
    }
}

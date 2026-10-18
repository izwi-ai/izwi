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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures::StreamExt;
use izwi_agent::{
    planner::{PlanningMode, SimplePlanner},
    strip_reasoning, AgentDefinition, AgentEngine, AgentSession, AgentTurnOptions, HistoryBudget,
    MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore, ModelBackend, ModelOutput,
    ModelRequest, NoopTool, ReasoningFilter, TimeTool, ToolRegistry, TurnInput,
};
use izwi_core::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, ChatMessage, ChatRequestConfig, ChatRole, ChatTruncation,
    GenerationConfig, GenerationParams, GenerationRequest, RuntimeRequestContext, RuntimeService,
    SessionModelLeases, VoiceLatency, VoiceSession, VoiceSessionPhase, WorkloadClass,
};
use izwi_vad::{
    configured_snr_margin_db, sanitize_score_threshold, sanitize_snr_margin, EndpointConfig,
    EndpointDetector, EndpointEndReason, EndpointEvent, SpeechEvidence, SpeechEvidenceConfig,
    VadScorer, DEFAULT_MAX_UTTERANCE_MS, DEFAULT_MIN_SPEECH_MS, DEFAULT_PRE_ROLL_MS,
    DEFAULT_SILENCE_MS, DEFAULT_SPEECH_THRESHOLD, VAD_FRAME_MS, VAD_SAMPLE_RATE,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;
use tracing::warn;

use crate::app::realtime_codec::{AudioOutputEncoder, OutputAudioFormat};
use crate::app::realtime_protocol::{
    RealtimeAudioFormat, RealtimeAudioGapAction, RealtimeClose, RealtimeCloseCode,
    RealtimeCloseReason, RealtimeErrorCode, RealtimeEventEnvelope, RealtimeInterruptionReason,
    RealtimeProtocol, RealtimeServerEvent, RealtimeSpeechEndReason, RealtimeTurnStatus,
    VOICE_REALTIME_VERSION,
};
use crate::app::realtime_transport::{
    now_unix_millis, parse_client_pcm16_frame, pcm16_bytes_to_i16, spawn_writer,
    wav_bytes_from_pcm16_mono, ClientPcm16Frame, OutboundClass, OutboundEnqueueError, OutboundItem,
    OutboundLimits, OutboundMailbox,
};
use crate::app::voice_reply::{SegmentJoiner, SpeechSegmenter, SpokenLedger};
use crate::chat_store::ChatStore;
use crate::ids::new_uuid;
use crate::state::{AppState, OwnedAdmissionPermit, StoredAgentSessionRecord};
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_NAME, DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
};
use crate::voice_memory::extract_observation_candidates;
use crate::voice_store::CreateVoiceTurnRequest;
const DEFAULT_CHAT_MODEL: &str = "Qwen3.5-4B";
/// A spoken reply's default token budget: a few sentences, not an essay
/// (SPK-F2). Clients may raise it with `max_output_tokens`.
const VOICE_DEFAULT_MAX_OUTPUT_TOKENS: usize = 384;
/// The conversation history a voice turn's prompt carries (SPK-F13).
const VOICE_HISTORY_BUDGET: HistoryBudget = HistoryBudget {
    max_messages: 24,
    max_tokens: 3_000,
};
const MAX_UTTERANCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_UTTERANCE_PCM16_SAMPLES: usize = (MAX_UTTERANCE_BYTES - 44) / 2;
const WS_BIN_MAGIC: &[u8; 4] = b"IVWS";
const WS_BIN_KIND_ASSISTANT_PCM16: u8 = 2;
const WS_BIN_KIND_ASSISTANT_G711_ULAW: u8 = 3;
const WS_BIN_KIND_ASSISTANT_G711_ALAW: u8 = 4;
const WS_BIN_ASSISTANT_HEADER_LEN: usize = 24;
const DEFAULT_STREAM_VAD_THRESHOLD: f32 = DEFAULT_SPEECH_THRESHOLD;
const DEFAULT_STREAM_MIN_SPEECH_MS: u32 = DEFAULT_MIN_SPEECH_MS;
const DEFAULT_STREAM_SILENCE_MS: u32 = DEFAULT_SILENCE_MS;
const DEFAULT_STREAM_MAX_UTTERANCE_MS: u32 = DEFAULT_MAX_UTTERANCE_MS;
const DEFAULT_STREAM_PRE_ROLL_MS: u32 = DEFAULT_PRE_ROLL_MS;
/// Voiced speech that confirms a barge-in. A speech start alone does not
/// interrupt the assistant: a cough, a door or speaker echo also starts one.
const DEFAULT_BARGE_IN_MIN_SPEECH_MS: u32 = 300;
/// How long past the silence window frames may stop before the server
/// assumes the client paused without `input_stream_stop`. Silence is measured
/// in audio time, so this only has to cover network stalls and jitter.
const CLIENT_PAUSE_STALL_ALLOWANCE_MS: u64 = 1_500;
const TURN_CANCELLATION_TIMEOUT: Duration = Duration::from_millis(750);
const SESSION_ACTOR_TICK: Duration = Duration::from_millis(100);
/// Assistant audio sent this long after the audio already sent had finished
/// playing counts as a playback underrun.
const PLAYBACK_UNDERRUN_TOLERANCE: Duration = Duration::from_millis(20);
const LEGACY_VOICE_REALTIME_PROTOCOL: &str = "voice_realtime_v1";
const TYPED_VOICE_REALTIME_PROTOCOL: &str = "voice_realtime";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum VoiceWireProtocol {
    #[default]
    LegacyV1,
    TypedV2,
}

fn negotiate_voice_protocol(
    protocol: Option<&str>,
    version: Option<u16>,
    resume_from_event_id: Option<u64>,
) -> Result<VoiceWireProtocol, String> {
    match (protocol.map(str::trim), version) {
        (None, None) => Ok(VoiceWireProtocol::LegacyV1),
        (Some(TYPED_VOICE_REALTIME_PROTOCOL), Some(VOICE_REALTIME_VERSION)) => {
            if resume_from_event_id.is_some() {
                return Err(
                    "voice_realtime v2 sessions are process-owned and non-resumable".to_string(),
                );
            }
            Ok(VoiceWireProtocol::TypedV2)
        }
        (Some(LEGACY_VOICE_REALTIME_PROTOCOL), Some(1)) => Ok(VoiceWireProtocol::LegacyV1),
        (Some(protocol), Some(version)) => Err(format!(
            "Unsupported realtime voice protocol negotiation `{protocol}` version {version}"
        )),
        _ => Err(
            "Realtime voice protocol negotiation requires both `protocol` and `version`"
                .to_string(),
        ),
    }
}

struct TypedVoiceSession {
    session_id: String,
    owner_instance_id: String,
    connection_epoch: u64,
    next_event_id: u64,
    next_sequence: u64,
    next_revision: u64,
    last_audio_chunk_sequence: u64,
}

impl TypedVoiceSession {
    fn new(session_id: String, owner_instance_id: String) -> Self {
        Self {
            session_id,
            owner_instance_id,
            connection_epoch: 0,
            next_event_id: 1,
            next_sequence: 0,
            next_revision: 1,
            last_audio_chunk_sequence: 0,
        }
    }

    fn next_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }

    fn envelope(
        &mut self,
        event: RealtimeServerEvent,
        utterance_id: Option<String>,
        turn_id: Option<String>,
    ) -> RealtimeEventEnvelope<RealtimeServerEvent> {
        let envelope = RealtimeEventEnvelope {
            protocol: RealtimeProtocol::VoiceRealtime,
            version: VOICE_REALTIME_VERSION,
            event_id: self.next_event_id,
            sequence: self.next_sequence,
            session_id: self.session_id.clone(),
            connection_epoch: self.connection_epoch,
            timestamp_ms: now_unix_millis(),
            utterance_id,
            turn_id,
            segment_id: None,
            event,
        };
        self.next_event_id = self.next_event_id.saturating_add(1);
        self.next_sequence = self.next_sequence.saturating_add(1);
        envelope
    }
}

#[derive(Default)]
struct VoiceOutboundWireState {
    typed: Option<TypedVoiceSession>,
}

#[derive(Clone)]
struct OutboundTx {
    mailbox: Arc<OutboundMailbox>,
    wire: Arc<Mutex<VoiceOutboundWireState>>,
    runtime: Arc<RuntimeService>,
    label: &'static str,
}

impl OutboundTx {
    fn new(runtime: Arc<RuntimeService>, label: &'static str) -> Self {
        Self {
            mailbox: Arc::new(OutboundMailbox::new(voice_outbound_limits())),
            wire: Arc::new(Mutex::new(VoiceOutboundWireState::default())),
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
        self.enqueue(message, OutboundClass::Coalescible(key), Some(generation))
            .is_ok()
    }

    /// Queue assistant audio once the client has taken enough of the audio
    /// already queued (credit-based backpressure, SPK-F11).
    async fn send_audio(
        &self,
        generation: u64,
        message: Message,
        bytes: usize,
        duration_ms: u64,
    ) -> Result<(), OutboundEnqueueError> {
        self.note_audio_chunk(&message);
        let result = self
            .mailbox
            .enqueue_audio(OutboundItem {
                message,
                class: OutboundClass::Audio { bytes, duration_ms },
                turn_generation: Some(generation),
            })
            .await;
        self.report(result)
    }

    /// Remember the chunk sequence of an assistant audio frame for the typed
    /// protocol's `assistant_audio_completed`.
    fn note_audio_chunk(&self, message: &Message) {
        let Message::Binary(frame) = message else {
            return;
        };
        if frame.len() >= WS_BIN_ASSISTANT_HEADER_LEN
            && frame.get(..4) == Some(WS_BIN_MAGIC.as_slice())
            && frame.get(5).is_some_and(|kind| {
                (WS_BIN_KIND_ASSISTANT_PCM16..=WS_BIN_KIND_ASSISTANT_G711_ALAW).contains(kind)
            })
        {
            let chunk_sequence =
                u32::from_le_bytes([frame[16], frame[17], frame[18], frame[19]]) as u64;
            if let Some(typed) = self
                .wire
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .typed
                .as_mut()
            {
                typed.last_audio_chunk_sequence = chunk_sequence;
            }
        }
    }

    fn send_diagnostic(&self, message: Message) -> bool {
        self.enqueue(message, OutboundClass::Droppable, None)
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
        self.report(result)
    }

    fn report(&self, result: Result<(), OutboundEnqueueError>) -> Result<(), OutboundEnqueueError> {
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

    fn accepts_turn_generation(&self, generation: u64) -> bool {
        self.mailbox.accepts_turn_generation(generation)
    }

    fn configure_wire_protocol(
        &self,
        protocol: VoiceWireProtocol,
        session_id: String,
        owner_instance_id: String,
    ) {
        let mut wire = self
            .wire
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        wire.typed = match protocol {
            VoiceWireProtocol::LegacyV1 => None,
            VoiceWireProtocol::TypedV2 => {
                Some(TypedVoiceSession::new(session_id, owner_instance_id))
            }
        };
    }

    fn typed_protocol_enabled(&self) -> bool {
        self.wire
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .typed
            .is_some()
    }

    fn advance_typed_output_epoch(&self) {
        if let Some(typed) = self
            .wire
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .typed
            .as_mut()
        {
            typed.connection_epoch = typed.connection_epoch.saturating_add(1);
            typed.next_sequence = 0;
        }
    }

    fn encode_json_event(
        &self,
        value: &serde_json::Value,
    ) -> Result<Option<String>, serde_json::Error> {
        let mut wire = self
            .wire
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(typed) = wire.typed.as_mut() else {
            return serde_json::to_string(value).map(Some);
        };
        let Some((event, utterance_id, turn_id)) = map_typed_voice_event(value, typed) else {
            return Ok(None);
        };
        serde_json::to_string(&typed.envelope(event, utterance_id, turn_id)).map(Some)
    }

    fn close(&self) {
        self.mailbox.close();
    }
}

fn voice_outbound_limits() -> OutboundLimits {
    let tuning = crate::env_config::tuning();
    OutboundLimits {
        max_items: tuning.voice_ws_outbound_items,
        max_audio_bytes: tuning.voice_ws_outbound_audio_bytes,
        max_audio_ms: tuning.voice_ws_outbound_audio_ms,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        system_prompt: Option<String>,
        #[serde(default)]
        protocol: Option<String>,
        #[serde(default)]
        version: Option<u16>,
        #[serde(default)]
        resume_from_event_id: Option<u64>,
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
        barge_in_min_speech_ms: Option<u32>,
        /// Energy above the noise floor, in dB, a frame needs to start an
        /// utterance. Half of it keeps one going. `0` disables the check.
        #[serde(default)]
        vad_snr_margin_db: Option<f32>,
        #[serde(default)]
        input_sample_rate: Option<u32>,
        /// `pcm_i16` (default), `g711_ulaw` or `g711_alaw`.
        #[serde(default)]
        output_audio_format: Option<String>,
    },
    InputStreamStop,
    Interrupt {
        #[serde(default)]
        reason: Option<String>,
    },
    /// The client played only `audio_end_ms` of a reply's audio (OpenAI
    /// Realtime's `conversation.item.truncate`): the conversation remembers
    /// the reply only up to there.
    #[serde(alias = "conversation.item.truncate")]
    ConversationItemTruncate {
        utterance_seq: u64,
        audio_end_ms: u64,
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
    output_format: OutputAudioFormat,
}

#[derive(Debug, Clone)]
struct UnifiedVoiceTurnConfig {
    s2s_model_id: String,
    speaker: Option<String>,
    max_output_tokens: usize,
    output_format: OutputAudioFormat,
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
    /// When the endpoint decision ended the utterance; the turn's latencies
    /// are measured from here.
    endpointed_at: Instant,
    /// Trailing silence waited for before the decision, when silence (or a
    /// client pause) ended the utterance.
    endpointing_delay: Option<Duration>,
}

struct ActiveTurn {
    utterance_id: String,
    utterance_seq: u64,
    turn_record_id: String,
    state: AppState,
    conversation: Arc<VoiceConversation>,
    task: tokio::task::JoinHandle<()>,
    control: TurnControl,
    output_generation: u64,
}

/// The connection's spoken conversation: the latest assistant replies,
/// held until what the user heard of them is known, then stored in order
/// (SPK-F13). Replies are stored when the next turn needs the history, or
/// when the stream or socket ends; until then the client may truncate them.
#[derive(Default)]
struct VoiceConversation {
    pending: Mutex<VecDeque<(u64, TurnControl)>>,
}

impl VoiceConversation {
    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(u64, TurnControl)>> {
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Hold `utterance_seq`'s reply until it is stored.
    fn hold(&self, utterance_seq: u64, control: TurnControl) {
        let mut pending = self.lock();
        pending.retain(|(seq, _)| *seq != utterance_seq);
        pending.push_back((utterance_seq, control));
    }

    /// The held replies, oldest first.
    fn take(&self) -> Vec<TurnControl> {
        self.lock().drain(..).map(|(_, control)| control).collect()
    }

    /// The held reply of `utterance_seq`.
    fn reply_of(&self, utterance_seq: u64) -> Option<TurnControl> {
        self.lock()
            .iter()
            .find(|(seq, _)| *seq == utterance_seq)
            .map(|(_, control)| control.clone())
    }

    /// The user interrupted playback: the latest held reply was heard up to
    /// where the client's playback had got.
    fn interrupt_playback(&self) {
        if let Some((_, control)) = self.lock().back() {
            control.stop_playback();
        }
    }

    /// Store the held replies in order.
    async fn store(&self, state: &AppState) {
        for control in self.take() {
            store_reply(state, &control).await;
        }
    }

    /// Store the held replies in the background.
    fn store_in_background(self: &Arc<Self>, state: &AppState) {
        let conversation = self.clone();
        let state = state.clone();
        tokio::spawn(async move { conversation.store(&state).await });
    }
}

#[derive(Clone)]
struct TurnControl {
    inner: Arc<TurnControlInner>,
}

struct TurnControlInner {
    cancelled: AtomicBool,
    reason: Mutex<Option<String>>,
    notify: Notify,
    endpointed_at: Instant,
    /// Set once the turn has sent assistant audio.
    speaking: AtomicBool,
    playback: Mutex<PlaybackClock>,
    reply: Mutex<ReplyRecord>,
}

/// What the turn's assistant reply said, and how much of it was heard.
#[derive(Default)]
struct ReplyRecord {
    /// The agent session (thread) the reply belongs to.
    session: Option<AgentSession>,
    turn_record_id: Option<String>,
    /// The whole reply as generated, and without its reasoning.
    raw_text: Option<String>,
    text: Option<String>,
    meta: MemoryMessageMeta,
    ledger: SpokenLedger,
    /// How much of the reply's audio the user heard, when not all of it.
    heard_ms: Option<f64>,
}

impl ReplyRecord {
    /// The text to remember: all of it, or what was heard.
    fn remembered_text(&self) -> Option<String> {
        match self.heard_ms {
            Some(heard_ms) => Some(self.ledger.heard_text(heard_ms)),
            None => self.text.clone(),
        }
    }
}

impl TurnControl {
    fn new(endpointed_at: Instant) -> Self {
        Self {
            inner: Arc::new(TurnControlInner {
                cancelled: AtomicBool::new(false),
                reason: Mutex::new(None),
                notify: Notify::new(),
                endpointed_at,
                speaking: AtomicBool::new(false),
                playback: Mutex::new(PlaybackClock::default()),
                reply: Mutex::new(ReplyRecord::default()),
            }),
        }
    }

    fn reply(&self) -> std::sync::MutexGuard<'_, ReplyRecord> {
        self.inner
            .reply
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// The client stopped playing the reply now: it was heard as far as its
    /// playback had got, unless the client already said how far.
    fn stop_playback(&self) {
        let played = self
            .inner
            .playback
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .played(Instant::now());
        let mut reply = self.reply();
        if reply.heard_ms.is_none() && played.as_secs_f64() * 1000.0 < reply.ledger.audio_ms() {
            reply.heard_ms = Some(played.as_secs_f64() * 1000.0);
        }
    }

    /// The client says it played `audio_end_ms` of the reply; returns the
    /// text that covers.
    fn truncate(&self, audio_end_ms: u64) -> String {
        let mut reply = self.reply();
        reply.heard_ms = Some(audio_end_ms as f64);
        reply.ledger.heard_text(audio_end_ms as f64)
    }

    fn endpointed_at(&self) -> Instant {
        self.inner.endpointed_at
    }

    fn is_speaking(&self) -> bool {
        self.inner.speaking.load(Ordering::Acquire)
    }

    /// Account for `duration` of assistant audio just sent: the first audio
    /// is the turn's response latency, and a chunk that found the client's
    /// playback already drained is an underrun.
    fn audio_sent(&self, runtime: &RuntimeService, duration: Duration) {
        let now = Instant::now();
        if !self.inner.speaking.swap(true, Ordering::AcqRel) {
            runtime.record_voice_latency(
                VoiceLatency::TurnFirstAudio,
                now.saturating_duration_since(self.inner.endpointed_at),
            );
        }
        let underrun = self
            .inner
            .playback
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .sent(now, duration);
        if underrun {
            runtime.record_voice_egress_underrun();
        }
    }

    fn cancel(&self, reason: impl Into<String>) {
        let mut stored_reason = self
            .inner
            .reason
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.inner.cancelled.load(Ordering::Acquire) {
            return;
        }
        *stored_reason = Some(reason.into());
        self.inner.cancelled.store(true, Ordering::Release);
        drop(stored_reason);
        self.inner.notify.notify_waiters();
    }

    fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    fn reason(&self) -> Option<String> {
        self.inner
            .reason
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
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

/// The server's view of the client's playback: the audio sent so far finishes
/// playing at `playing_until`, assuming the client plays it as it arrives.
#[derive(Debug, Default)]
struct PlaybackClock {
    playing_until: Option<Instant>,
    sent: Duration,
}

impl PlaybackClock {
    /// Account for `duration` of audio sent at `now`; returns whether the
    /// audio sent before it had already finished playing (an underrun).
    fn sent(&mut self, now: Instant, duration: Duration) -> bool {
        let underrun = self
            .playing_until
            .is_some_and(|until| now > until + PLAYBACK_UNDERRUN_TOLERANCE);
        let starts = self
            .playing_until
            .filter(|until| *until > now)
            .unwrap_or(now);
        self.playing_until = Some(starts + duration);
        self.sent += duration;
        underrun
    }

    /// How much of the audio sent has played by `now`.
    fn played(&self, now: Instant) -> Duration {
        let queued = self
            .playing_until
            .map_or(Duration::ZERO, |until| until.saturating_duration_since(now));
        self.sent.saturating_sub(queued)
    }
}

/// Move the session phase along with its active turn: Speaking once the turn
/// has sent audio, Listening once it has ended. `turn` is the active turn's
/// record id, whether it is still running and whether it has spoken.
fn follow_turn_phase(session: &mut VoiceSession, turn: Option<(&str, bool, bool)>) {
    let in_turn = matches!(
        session.phase(),
        VoiceSessionPhase::Processing | VoiceSessionPhase::Speaking
    );
    match turn {
        Some((turn_id, true, true)) if session.phase() == VoiceSessionPhase::Processing => {
            session.begin_speaking(turn_id)
        }
        Some((_, true, _)) => {}
        Some((_, false, _)) | None if in_turn => session.begin_listening(),
        _ => {}
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
    barge_in_min_speech_ms: u32,
    snr_margin_db: f32,
    input_sample_rate_hint: Option<u32>,
}

impl StreamingInputConfig {
    fn evidence_config(&self) -> SpeechEvidenceConfig {
        SpeechEvidenceConfig {
            threshold: self.vad_threshold,
            snr_margin_db: self.snr_margin_db,
            ..SpeechEvidenceConfig::default()
        }
    }

    fn endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            start_threshold: self.vad_threshold,
            end_threshold: (self.vad_threshold * 0.7).min(self.vad_threshold),
            min_speech_ms: self.min_speech_ms,
            silence_ms: self.silence_duration_ms,
            max_utterance_ms: self.max_utterance_ms,
            snr_margin_db: self.snr_margin_db,
            continue_snr_margin_db: self.snr_margin_db * 0.5,
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
    barge_in_confirmed: bool,
    /// The offline no-speech gate's rule over this utterance's frames. An
    /// utterance without it never becomes a turn, for modular and unified
    /// (speech-to-speech) turns alike.
    evidence: SpeechEvidence,
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

    fn output_format(&self) -> OutputAudioFormat {
        match self {
            Self::Modular(config) => config.output_format,
            Self::Unified(config) => config.output_format,
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

/// The models a turn config runs, in pipeline order: (stage, model id,
/// variant).
fn turn_config_models(
    config: &VoiceTurnConfig,
) -> Result<Vec<(&'static str, String, izwi_core::ModelVariant)>, String> {
    Ok(match config {
        VoiceTurnConfig::Modular(config) => vec![
            (
                "asr",
                config.asr_model_id.clone(),
                resolve_asr_model_variant(Some(&config.asr_model_id)),
            ),
            (
                "text",
                config.text_model_id.clone(),
                parse_chat_model_variant(Some(&config.text_model_id))
                    .map_err(|err| format!("Invalid chat model: {err}"))?,
            ),
            (
                "tts",
                config.tts_model_id.clone(),
                parse_tts_model_variant(&config.tts_model_id)
                    .map_err(|err| format!("Unsupported TTS model: {err}"))?,
            ),
        ],
        VoiceTurnConfig::Unified(config) => vec![(
            "s2s",
            config.s2s_model_id.clone(),
            parse_model_variant(&config.s2s_model_id)
                .map_err(|err| format!("Unsupported unified audio model: {err}"))?,
        )],
    })
}

/// Load and lease an input stream's models when the stream starts, so its
/// first turn does not pay the cold load (SPK-F21), then report which are
/// ready in a `models_ready` event. A model that fails to load is reported
/// here; its turns fail with the same error.
async fn warm_session_models(
    runtime: Arc<RuntimeService>,
    leases: SessionModelLeases,
    models: Vec<(&'static str, String, izwi_core::ModelVariant)>,
    out_tx: OutboundTx,
) {
    let started = Instant::now();
    let mut statuses = Vec::with_capacity(models.len());
    for (stage, model_id, variant) in models {
        statuses.push(match leases.hold(&runtime, variant).await {
            Ok(()) => json!({ "stage": stage, "model_id": model_id, "ready": true }),
            Err(err) => json!({
                "stage": stage,
                "model_id": model_id,
                "ready": false,
                "error": err.to_string(),
            }),
        });
    }
    send_json(
        &out_tx,
        json!({
            "type": "models_ready",
            "models": statuses,
            "elapsed_ms": started.elapsed().as_millis() as u64,
        }),
    );
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
    agent_session: Arc<AgentSessionSlot>,
    conversation: Arc<VoiceConversation>,
    streaming_input: Option<StreamingInputState>,
    /// Keeps the input stream's models resident between turns without
    /// pinning them; turns in flight hold their own clone.
    model_leases: SessionModelLeases,
    /// Loads the input stream's models at its start (SPK-F21).
    model_warmup: Option<AbortOnDropTask<()>>,
    active_turn: Option<ActiveTurn>,
    voice_session: VoiceSession,
    voice_session_started_at: Option<Instant>,
    /// When the current utterance's speech started: a barge-in's latency
    /// runs from here.
    speech_started_at: Option<Instant>,
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
            agent_session: Arc::default(),
            conversation: Arc::default(),
            streaming_input: None,
            model_leases: SessionModelLeases::default(),
            model_warmup: None,
            active_turn: None,
            voice_session: VoiceSession::default(),
            voice_session_started_at: None,
            speech_started_at: None,
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
    fn sync_voice_phase(&mut self) {
        let turn = self.active_turn.as_ref().map(|turn| {
            (
                turn.turn_record_id.as_str(),
                !turn.task.is_finished(),
                turn.control.is_speaking(),
            )
        });
        follow_turn_phase(&mut self.voice_session, turn);
    }

    /// Record a closed session's duration, once.
    fn record_session_closed(&mut self, runtime: &RuntimeService) {
        runtime.record_voice_session_closed();
        if let Some(started) = self.voice_session_started_at.take() {
            runtime.record_voice_latency(VoiceLatency::SessionDuration, started.elapsed());
        }
    }

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

/// How an active utterance ended.
#[derive(Debug)]
enum UtteranceOutcome {
    /// Enough voiced speech: it becomes a turn.
    Finalized(PendingAudioCommit, Vec<u8>, UtteranceEndReason),
    /// Too little voiced speech, or no speech evidence: the client resumes
    /// whatever it ducked on `user_speech_start`.
    Rejected(SpeechStartEvent),
}

#[derive(Debug)]
struct StreamingFrameResult {
    sequence_gap: Option<(u32, u32)>,
    speech_start: Option<SpeechStartEvent>,
    /// The active utterance has just confirmed a barge-in.
    barge_in: bool,
    utterance_end: Option<UtteranceOutcome>,
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
        if !(8_000..=192_000).contains(&sample_rate) {
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
                barge_in: false,
                utterance_end: None,
            });
        }
        if !payload.len().is_multiple_of(2) {
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
                barge_in: false,
                utterance_end: None,
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
            // Voiced time, which confirms barge-in, counts only frames that
            // clear the SNR margin as well as the score threshold.
            let decision = self.endpoint.process_frame(vad_frame);
            if decision.is_speech {
                chunk_has_speech = true;
                chunk_voiced_ms += vad_frame.duration_ms();
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
            barge_in: false,
            utterance_end: None,
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
                barge_in_confirmed: false,
                evidence: SpeechEvidence::new(self.config.evidence_config()),
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
            for vad_frame in &vad_frames {
                active.evidence.push(vad_frame);
            }
            active.voiced_ms += chunk_voiced_ms;
            active.total_ms += frame_ms;
            if chunk_has_speech {
                active.silence_ms = 0.0;
            } else {
                active.silence_ms += frame_ms;
            }
            // An utterance that ends or is rejected in this frame never
            // barges in; finalizing one preempts the turn on its own.
            if !active.barge_in_confirmed
                && end_reason.is_none()
                && !noise_rejected
                && active.voiced_ms >= self.config.barge_in_min_speech_ms as f32
            {
                active.barge_in_confirmed = true;
                result.barge_in = true;
            }
        }

        if let Some(reason) = end_reason {
            result.utterance_end = self.finalize_active_utterance(reason)?;
        } else if noise_rejected {
            result.utterance_end = self.reject_active_utterance();
        }

        if !chunk_has_speech {
            self.push_pre_roll(&samples, sample_rate);
        }

        Ok(result)
    }

    fn reject_active_utterance(&mut self) -> Option<UtteranceOutcome> {
        let active = self.active.take()?;
        Some(UtteranceOutcome::Rejected(SpeechStartEvent {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
        }))
    }

    fn finish_stream(&mut self) -> Result<Option<UtteranceOutcome>, String> {
        match self.endpoint.finish() {
            Some(EndpointEvent::NoiseRejected) => Ok(self.reject_active_utterance()),
            _ => self.finalize_active_utterance(UtteranceEndReason::StreamStopped),
        }
    }

    /// Ends the active utterance when the client stops sending audio without
    /// `input_stream_stop`. Silence and the utterance cap are measured in
    /// audio time by the endpoint detector, so delayed but contiguous frames
    /// (a network stall) cannot end an utterance early.
    fn on_tick(&mut self) -> Result<Option<UtteranceOutcome>, String> {
        if self.active.is_none() {
            return Ok(None);
        }
        let pause_limit = Duration::from_millis(
            u64::from(self.config.silence_duration_ms) + CLIENT_PAUSE_STALL_ALLOWANCE_MS,
        );
        if self
            .last_frame_at
            .is_some_and(|last_frame| last_frame.elapsed() >= pause_limit)
        {
            return self.finalize_active_utterance(UtteranceEndReason::ClientPause);
        }
        Ok(None)
    }

    /// End the active utterance: a turn when it holds enough voiced speech,
    /// otherwise a rejection the client must hear about, since it already saw
    /// `user_speech_start`.
    fn finalize_active_utterance(
        &mut self,
        reason: UtteranceEndReason,
    ) -> Result<Option<UtteranceOutcome>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        self.endpoint = EndpointDetector::new(self.config.endpoint_config());
        if active.voiced_ms < self.config.min_speech_ms as f32 || !active.evidence.is_sufficient() {
            return Ok(Some(UtteranceOutcome::Rejected(SpeechStartEvent {
                utterance_id: active.utterance_id,
                utterance_seq: active.utterance_seq,
            })));
        }
        let sample_rate = self
            .current_sample_rate
            .or(self.config.input_sample_rate_hint)
            .ok_or_else(|| "Missing input sample rate for streamed audio".to_string())?;

        let wav_bytes = wav_bytes_from_pcm16_mono(&active.samples_i16, sample_rate)?;
        if wav_bytes.len() > MAX_UTTERANCE_BYTES {
            return Err(format!(
                "Streamed utterance exceeded max encoded size ({} > {})",
                wav_bytes.len(),
                MAX_UTTERANCE_BYTES
            ));
        }

        let trailing_silence =
            Duration::from_micros((active.silence_ms.max(0.0) * 1000.0).round() as u64);
        let endpointing_delay = match reason {
            UtteranceEndReason::Silence => Some(trailing_silence),
            UtteranceEndReason::ClientPause => Some(
                trailing_silence
                    + self
                        .last_frame_at
                        .map_or(Duration::ZERO, |last_frame| last_frame.elapsed()),
            ),
            UtteranceEndReason::MaxDuration | UtteranceEndReason::StreamStopped => None,
        };
        let commit = PendingAudioCommit {
            utterance_id: active.utterance_id,
            utterance_seq: active.utterance_seq,
            turn_config: self.config.turn_config.clone(),
            endpointed_at: Instant::now(),
            endpointing_delay,
        };

        Ok(Some(UtteranceOutcome::Finalized(commit, wav_bytes, reason)))
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

/// Tell the client how an utterance ended and, for a finished one, start its
/// turn.
fn dispatch_utterance_outcome(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    conn: &mut ConnectionState,
    outcome: UtteranceOutcome,
) -> Result<(), String> {
    match outcome {
        UtteranceOutcome::Finalized(commit, wav_bytes, end_reason) => {
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
        }
        UtteranceOutcome::Rejected(evt) => {
            state.runtime.record_voice_speech_rejected();
            send_json(
                out_tx,
                json!({
                    "type": "user_speech_rejected",
                    "utterance_id": evt.utterance_id,
                    "utterance_seq": evt.utterance_seq,
                }),
            );
            Ok(())
        }
    }
}

/// Start the turn for a finished utterance. Nothing here waits on storage
/// or on the turn it preempts (SPK-F19): the turn task creates its own
/// record, and a preempted turn is reaped in the background.
fn finalize_stream_vad_utterance(
    state: &AppState,
    correlation_id: &str,
    out_tx: &OutboundTx,
    conn: &mut ConnectionState,
    commit: PendingAudioCommit,
    wav_bytes: Vec<u8>,
    end_reason: UtteranceEndReason,
) -> Result<(), String> {
    if interrupt_active_turn(out_tx, &mut conn.active_turn, "preempted_by_new_turn") {
        state.runtime.record_voice_interruption();
        conn.voice_session.interrupt("preempted_by_new_turn");
    }
    if let Some(delay) = commit.endpointing_delay {
        state
            .runtime
            .record_voice_latency(VoiceLatency::Endpointing, delay);
    }

    let voice_session_id = conn
        .voice_session_id
        .clone()
        .ok_or_else(|| "Voice session not initialized".to_string())?;
    let model_ids = commit.turn_config.model_ids();
    let turn_record_id = new_uuid();
    let record = CreateVoiceTurnRequest {
        turn_id: Some(turn_record_id.clone()),
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
    };

    conn.next_turn_generation = conn.next_turn_generation.saturating_add(1);
    let output_generation = conn.next_turn_generation;
    let control = TurnControl::new(commit.endpointed_at);
    let task = spawn_turn_task(
        TurnContext {
            state: state.clone(),
            correlation_id: correlation_id.to_string(),
            out_tx: out_tx.clone(),
            commit: commit.clone(),
            turn_record_id: turn_record_id.clone(),
            voice_session_id,
            voice_profile_id: conn.voice_profile_id.clone(),
            system_prompt: conn.system_prompt.clone(),
            agent_session: conn.agent_session.clone(),
            conversation: conn.conversation.clone(),
            control: control.clone(),
            output_generation,
            model_leases: conn.model_leases.clone(),
        },
        record,
        wav_bytes,
    );

    conn.active_turn = Some(ActiveTurn {
        utterance_id: commit.utterance_id,
        utterance_seq: commit.utterance_seq,
        turn_record_id: turn_record_id.clone(),
        state: state.clone(),
        conversation: conn.conversation.clone(),
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

/// An assistant audio frame: the 24-byte header (magic, version, kind,
/// flags, utterance sequence, chunk sequence, sample rate) and the payload
/// in the session's output format, which the kind names.
fn encode_assistant_audio_binary_frame(
    format: OutputAudioFormat,
    utterance_seq: u64,
    chunk_seq: u32,
    sample_rate: u32,
    is_final: bool,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(WS_BIN_ASSISTANT_HEADER_LEN + payload.len());
    out.extend_from_slice(WS_BIN_MAGIC);
    out.push(crate::app::realtime_transport::WS_BIN_VERSION);
    out.push(match format {
        OutputAudioFormat::PcmI16 => WS_BIN_KIND_ASSISTANT_PCM16,
        OutputAudioFormat::G711Ulaw => WS_BIN_KIND_ASSISTANT_G711_ULAW,
        OutputAudioFormat::G711Alaw => WS_BIN_KIND_ASSISTANT_G711_ALAW,
    });
    let flags: u16 = if is_final { 1 } else { 0 };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&utterance_seq.to_le_bytes());
    out.extend_from_slice(&chunk_seq.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(payload);
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

    let (ws_tx, mut ws_rx) = socket.split();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice realtime");
    let writer = spawn_writer(ws_tx, out_tx.mailbox.clone(), 1013);

    let mut conn = ConnectionState::default();
    send_json(
        &out_tx,
        json!({
            "type": "connected",
            "protocol": LEGACY_VOICE_REALTIME_PROTOCOL,
            "server_time_ms": now_unix_millis(),
        }),
    );

    let mut actor_tick = tokio::time::interval(SESSION_ACTOR_TICK);
    actor_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    actor_tick.tick().await;
    loop {
        conn.sync_voice_phase();
        let result = tokio::select! {
            result = ws_rx.next() => result,
            _ = actor_tick.tick() => {
                let timed_utterance = match conn.streaming_input.as_mut() {
                    Some(streaming) => streaming.on_tick(),
                    None => Ok(None),
                };
                match timed_utterance {
                    Ok(Some(outcome)) => {
                        if let Err(err) = dispatch_utterance_outcome(
                            &state,
                            &correlation_id,
                            &out_tx,
                            &mut conn,
                            outcome,
                        ) {
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

    if interrupt_active_turn(&out_tx, &mut conn.active_turn, "socket_closed") {
        state.runtime.record_voice_interruption();
    }
    conn.conversation.store_in_background(&state);
    if let Some(session_id) = conn.close_voice_session_now() {
        if let Err(err) = state.voice_store.end_session(session_id).await {
            warn!("failed to end voice session on socket close: {err}");
        } else {
            conn.record_session_closed(&state.runtime);
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
        ClientEvent::SessionStart {
            system_prompt,
            protocol,
            version,
            resume_from_event_id,
        } => {
            let wire_protocol =
                negotiate_voice_protocol(protocol.as_deref(), version, resume_from_event_id)?;
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
                conn.system_prompt = prompt;
            } else {
                conn.system_prompt = profile.system_prompt;
            }
            conn.started = true;
            let transport_session_id = conn.transport_session_id.clone();
            let owner_instance_id = conn.owner_instance_id.clone();
            out_tx.configure_wire_protocol(
                wire_protocol,
                transport_session_id.clone(),
                owner_instance_id.clone(),
            );
            send_json(
                out_tx,
                json!({
                    "type": "session_ready",
                    "protocol": match wire_protocol {
                        VoiceWireProtocol::LegacyV1 => LEGACY_VOICE_REALTIME_PROTOCOL,
                        VoiceWireProtocol::TypedV2 => TYPED_VOICE_REALTIME_PROTOCOL,
                    },
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
            barge_in_min_speech_ms,
            vad_snr_margin_db,
            input_sample_rate,
            output_audio_format,
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
            let output_format = OutputAudioFormat::parse(output_audio_format.as_deref())?;
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
                    let tts_variant = parse_tts_model_variant(tts_model_id.as_str())
                        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
                    validate_realtime_tts_model(tts_variant)?;

                    VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
                        asr_model_id,
                        text_model_id,
                        tts_model_id,
                        speaker: normalized_speaker.clone(),
                        asr_language: normalized_asr_language,
                        max_output_tokens: max_output_tokens
                            .unwrap_or(VOICE_DEFAULT_MAX_OUTPUT_TOKENS)
                            .clamp(1, 4096),
                        output_format,
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
                        output_format,
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
                conn.voice_session_started_at = Some(Instant::now());
                state.runtime.record_voice_session_started();
            }

            let warm_models = turn_config_models(&turn_config)?;
            conn.model_leases = SessionModelLeases::default();
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
                barge_in_min_speech_ms: barge_in_min_speech_ms
                    .unwrap_or(DEFAULT_BARGE_IN_MIN_SPEECH_MS)
                    .min(5_000),
                snr_margin_db: sanitize_snr_margin(
                    vad_snr_margin_db.unwrap_or_else(configured_snr_margin_db),
                ),
                input_sample_rate_hint: input_sample_rate
                    .filter(|sr| *sr >= 8_000 && *sr <= 192_000),
            }));
            conn.voice_session.begin_listening();

            send_json(
                out_tx,
                json!({
                    "type": "input_stream_ready",
                    "output_audio_format": output_format.as_str(),
                    "vad": {
                        "backend": "earshot",
                        "threshold": conn.streaming_input.as_ref().map(|s| s.config.vad_threshold),
                        "score_sample_rate": VAD_SAMPLE_RATE,
                        "score_frame_ms": VAD_FRAME_MS,
                        "snr_margin_db": conn.streaming_input.as_ref().map(|s| s.config.snr_margin_db),
                        "min_speech_ms": conn.streaming_input.as_ref().map(|s| s.config.min_speech_ms),
                        "silence_duration_ms": conn.streaming_input.as_ref().map(|s| s.config.silence_duration_ms),
                        "barge_in_min_speech_ms": conn.streaming_input.as_ref().map(|s| s.config.barge_in_min_speech_ms),
                    }
                }),
            );
            conn.model_warmup = Some(AbortOnDropTask::new(tokio::spawn(warm_session_models(
                state.runtime.clone(),
                conn.model_leases.clone(),
                warm_models,
                out_tx.clone(),
            ))));
        }
        ClientEvent::InputStreamStop => {
            if let Some(mut streaming) = conn.streaming_input.take() {
                if let Some(outcome) = streaming.finish_stream()? {
                    dispatch_utterance_outcome(state, correlation_id, out_tx, conn, outcome)?;
                }
            }
            send_json(out_tx, json!({ "type": "input_stream_stopped" }));
            conn.conversation.store_in_background(state);
            conn.model_warmup = None;
            conn.model_leases = SessionModelLeases::default();
            if let Some(session_id) = conn.close_voice_session_if_idle() {
                conn.record_session_closed(&state.runtime);
                let store = state.voice_store.clone();
                tokio::spawn(async move {
                    if let Err(err) = store.end_session(session_id).await {
                        warn!("failed to end voice session: {err}");
                    }
                });
            }
        }
        ClientEvent::Interrupt { reason } => {
            let reason = reason.unwrap_or_else(|| "client_interrupt".to_string());
            if interrupt_active_turn(out_tx, &mut conn.active_turn, &reason) {
                state.runtime.record_voice_interruption();
                conn.voice_session.interrupt(reason);
            }
        }
        ClientEvent::ConversationItemTruncate {
            utterance_seq,
            audio_end_ms,
        } => {
            let control = conn
                .active_turn
                .as_ref()
                .filter(|turn| turn.utterance_seq == utterance_seq)
                .map(|turn| turn.control.clone())
                .or_else(|| conn.conversation.reply_of(utterance_seq))
                .ok_or_else(|| {
                    format!("No assistant reply of utterance {utterance_seq} to truncate")
                })?;
            let text = control.truncate(audio_end_ms);
            send_json(
                out_tx,
                json!({
                    "type": "conversation_item_truncated",
                    "utterance_seq": utterance_seq,
                    "audio_end_ms": audio_end_ms,
                    "text": text,
                }),
            );
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
    let ClientPcm16Frame {
        frame_seq,
        sample_rate,
        payload,
    } = parse_client_pcm16_frame(&audio_bytes, WS_BIN_MAGIC, "voice realtime")?;
    if !conn.started {
        return Err("Session not started. Send `session_start` first.".to_string());
    }
    let Some(streaming) = conn.streaming_input.as_mut() else {
        return Err("Received streaming audio frame before `input_stream_start`.".to_string());
    };

    let frame_result = streaming.handle_pcm16_frame(frame_seq, sample_rate, &payload)?;

    if let Some((expected, received)) = frame_result.sequence_gap {
        state
            .runtime
            .record_voice_ingress_dropped_frames(u64::from(received.saturating_sub(expected)));
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

    // Clients may duck playback on `user_speech_start` and resume it
    // on `user_speech_rejected`; the turn is cancelled only once the
    // speech is sustained.
    if let Some(evt) = frame_result.speech_start {
        conn.speech_started_at = Some(Instant::now());
        send_json(
            out_tx,
            json!({
                "type": "user_speech_start",
                "utterance_id": evt.utterance_id,
                "utterance_seq": evt.utterance_seq,
            }),
        );
    }

    if frame_result.barge_in {
        // A finished reply may still be playing on the client.
        conn.conversation.interrupt_playback();
    }
    if frame_result.barge_in && interrupt_active_turn(out_tx, &mut conn.active_turn, "barge_in") {
        state.runtime.record_voice_interruption();
        state.runtime.record_voice_barge_in();
        if let Some(started) = conn.speech_started_at {
            state
                .runtime
                .record_voice_latency(VoiceLatency::BargeIn, started.elapsed());
        }
        conn.voice_session.interrupt("barge_in");
    }

    if let Some(outcome) = frame_result.utterance_end {
        dispatch_utterance_outcome(state, correlation_id, out_tx, conn, outcome)?;
    }

    Ok(())
}

/// What one turn task works with.
struct TurnContext {
    state: AppState,
    correlation_id: String,
    out_tx: OutboundTx,
    commit: PendingAudioCommit,
    turn_record_id: String,
    voice_session_id: String,
    voice_profile_id: Option<String>,
    system_prompt: String,
    agent_session: Arc<AgentSessionSlot>,
    conversation: Arc<VoiceConversation>,
    control: TurnControl,
    output_generation: u64,
    model_leases: SessionModelLeases,
}

fn spawn_turn_task(
    context: TurnContext,
    record: CreateVoiceTurnRequest,
    audio_bytes: Vec<u8>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let TurnContext {
            state,
            correlation_id,
            out_tx,
            commit,
            turn_record_id,
            voice_session_id,
            voice_profile_id,
            system_prompt,
            agent_session,
            conversation,
            control,
            output_generation,
            model_leases,
        } = context;
        let timeout_secs = state.request_timeout_secs.max(1);
        let timeout = Duration::from_secs(timeout_secs);

        // The record exists before anything can complete it: a cancelled
        // turn is still recorded as interrupted.
        if let Err(err) = state.voice_store.create_turn(record).await {
            send_turn_error(
                &out_tx,
                output_generation,
                Some(commit.utterance_id.clone()),
                Some(commit.utterance_seq),
                format!("Voice storage error: {err}"),
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
            return;
        }

        let turn_future = async {
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

                    let transcript = with_stage_permit(&state, async |runtime_context| {
                        model_leases
                            .hold(
                                &state.runtime,
                                resolve_asr_model_variant(Some(&config.asr_model_id)),
                            )
                            .await
                            .map_err(|err| format!("Failed to load ASR model: {err}"))?;
                        let tx = out_tx.clone();
                        let utt_id = commit.utterance_id.clone();
                        let utt_seq = commit.utterance_seq;
                        let asr_model_id = config.asr_model_id.clone();
                        let asr_language = config.asr_language.clone();
                        let mut snapshot = String::new();
                        let runtime = state.runtime.clone();
                        let endpointed_at = control.endpointed_at();
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
                                    if snapshot.is_empty() {
                                        runtime.record_voice_latency(
                                            VoiceLatency::AsrFirstPartial,
                                            endpointed_at.elapsed(),
                                        );
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
                            .map_err(|err| format!("ASR failed: {err}"))
                    })
                    .await?;
                    let transcript_final_at = Instant::now();
                    state.runtime.record_voice_latency(
                        VoiceLatency::AsrFinal,
                        transcript_final_at.saturating_duration_since(control.endpointed_at()),
                    );

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

                    let agent_session_id = agent_session
                        .ensure(&state, &system_prompt, &config.text_model_id)
                        .await?;
                    let text_variant = parse_chat_model_variant(Some(&config.text_model_id))
                        .map_err(|err| format!("Invalid chat model: {err}"))?;
                    // The earlier replies join the history now, as far as
                    // they were heard.
                    conversation.store(&state).await;

                    // The reply streams into speech: its sentences are
                    // synthesized while the text model is still writing.
                    let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
                    let reply = async {
                        let raw_text = with_stage_permit(&state, async |runtime_context| {
                            model_leases
                                .hold(&state.runtime, text_variant)
                                .await
                                .map_err(|err| format!("Failed to load chat model: {err}"))?;
                            run_agent_turn(
                                &state,
                                &agent_session_id,
                                &user_text,
                                &config.text_model_id,
                                config.max_output_tokens,
                                voice_profile_id.as_deref(),
                                &correlation_id,
                                runtime_context,
                                &control,
                                &turn_record_id,
                                deltas,
                            )
                            .await
                        })
                        .await?;
                        let assistant_text = strip_reasoning(&raw_text);
                        send_turn_json(
                            &out_tx,
                            output_generation,
                            json!({
                                "type": "assistant_text_final",
                                "utterance_id": commit.utterance_id,
                                "utterance_seq": commit.utterance_seq,
                                "text": assistant_text.clone(),
                                "raw_text": raw_text.clone(),
                            }),
                        );
                        state
                            .voice_store
                            .update_turn_assistant(
                                turn_record_id.clone(),
                                Some(assistant_text.clone()),
                                Some(raw_text),
                            )
                            .await
                            .map_err(|err| format!("Voice storage error: {err}"))?;
                        Ok::<_, String>(assistant_text)
                    };
                    let tts_variant = parse_tts_model_variant(&config.tts_model_id)
                        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
                    let speech = speak_reply(
                        &state,
                        &out_tx,
                        &commit,
                        &control,
                        output_generation,
                        delta_rx,
                        transcript_final_at,
                        |text| {
                            SegmentSynthesis::start(
                                &state,
                                &correlation_id,
                                tts_variant,
                                config.speaker.clone(),
                                &model_leases,
                                text,
                            )
                        },
                    );
                    let (assistant_text, ()) = tokio::try_join!(reply, speech)?;
                    conversation.hold(commit.utterance_seq, control.clone());

                    state
                        .voice_store
                        .complete_turn(turn_record_id.clone(), "ok", None)
                        .await
                        .map_err(|err| format!("Voice storage error: {err}"))?;
                    if !assistant_text.is_empty() {
                        maybe_spawn_observation_extraction(
                            state.clone(),
                            voice_profile_id.clone(),
                            voice_session_id.clone(),
                            turn_record_id.clone(),
                            config.text_model_id.clone(),
                            correlation_id.clone(),
                            user_text.clone(),
                            assistant_text,
                        );
                    }

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
                    let generation = with_stage_permit(&state, async |runtime_context| {
                        stream_unified_s2s_to_socket(
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
                            config.output_format,
                            audio_bytes.as_slice(),
                            runtime_context,
                            &control,
                            output_generation,
                            &model_leases,
                        )
                        .await
                    })
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

                    let assistant_text = strip_reasoning(generation.text.trim());
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

        // A cancelled turn reports its interruption even when the cutoff
        // also failed the work it was doing.
        let turn_result = tokio::select! {
            biased;
            _ = control.cancelled() => {
                let reason = control.reason().unwrap_or_else(|| "cancelled".to_string());
                if matches!(
                    state
                        .voice_store
                        .complete_turn(
                            turn_record_id.clone(),
                            "interrupted",
                            Some(reason.clone()),
                        )
                        .await,
                    Ok(true)
                ) {
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "turn_interrupted",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "reason": reason,
                        }),
                    );
                    send_turn_json(
                        &out_tx,
                        output_generation,
                        json!({
                            "type": "turn_done",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "status": "interrupted",
                            "reason": reason,
                        }),
                    );
                }
                return;
            },
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

/// A realtime admission permit for one stage of a turn. A turn takes one per
/// stage (ASR, the reply, each TTS segment) instead of one for its whole
/// life, so the time it spends waiting between stages, or on the client,
/// does not hold a slot other traffic could use (SPK-F22).
async fn acquire_stage_permit(state: &AppState) -> Result<OwnedAdmissionPermit, String> {
    state
        .acquire_owned_workload_permit(WorkloadClass::Realtime)
        .await
        .map_err(|_| "Server is shutting down".to_string())
}

/// Run one turn stage under its own admission permit, released when the
/// stage ends.
async fn with_stage_permit<T>(
    state: &AppState,
    stage: impl AsyncFnOnce(RuntimeRequestContext) -> Result<T, String>,
) -> Result<T, String> {
    let permit = acquire_stage_permit(state).await?;
    stage(permit.runtime_context()).await
}

/// One reply segment being synthesized: its audio arrives on `chunks` while
/// the synthesis task, holding its own admission permit, runs ahead.
struct SegmentSynthesis {
    text: String,
    chunks: tokio::sync::mpsc::Receiver<izwi_core::AudioChunk>,
    task: AbortOnDropTask<Result<(), String>>,
}

impl SegmentSynthesis {
    fn start(
        state: &AppState,
        correlation_id: &str,
        variant: izwi_core::ModelVariant,
        speaker: Option<String>,
        model_leases: &SessionModelLeases,
        text: String,
    ) -> Self {
        let mut request = GenerationRequest {
            id: uuid::Uuid::new_v4().to_string(),
            model_variant: Some(variant),
            correlation_id: Some(correlation_id.to_string()),
            runtime_context: RuntimeRequestContext::default(),
            text: text.clone(),
            config: GenerationConfig {
                streaming: true,
                options: GenerationParams {
                    max_tokens: 0,
                    speaker: speaker.clone(),
                    voice: speaker,
                    ..Default::default()
                },
            },
            language: None,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
        };
        let (chunk_tx, chunks) = tokio::sync::mpsc::channel(32);
        let state = state.clone();
        let model_leases = model_leases.clone();
        let task = AbortOnDropTask::new(tokio::spawn(async move {
            model_leases
                .hold(&state.runtime, variant)
                .await
                .map_err(|err| format!("Failed to load TTS model: {err}"))?;
            let permit = acquire_stage_permit(&state).await?;
            request.runtime_context = permit.runtime_context();
            state
                .runtime
                .generate_streaming(request, chunk_tx)
                .await
                .map_err(|err| format!("TTS failed: {err}"))
        }));
        Self { text, chunks, task }
    }

    async fn finish(self) -> Result<(), String> {
        drop(self.chunks);
        self.task
            .join()
            .await
            .map_err(|err| format!("TTS streaming task failed: {err}"))?
    }
}

/// The next audio chunk of the segment playing, if one is.
async fn next_chunk(segment: Option<&mut SegmentSynthesis>) -> Option<izwi_core::AudioChunk> {
    match segment {
        Some(segment) => segment.chunks.recv().await,
        None => std::future::pending().await,
    }
}

/// Assistant audio of one turn as it goes out: the rate it is announced
/// at, its frame sequence, and the turn's latency accounting.
struct ReplyAudio<'a> {
    state: &'a AppState,
    out_tx: &'a OutboundTx,
    control: &'a TurnControl,
    utterance_id: &'a str,
    utterance_seq: u64,
    output_generation: u64,
    format: OutputAudioFormat,
    encoder: Option<AudioOutputEncoder>,
    announced_rate: Option<u32>,
    chunk_seq: u32,
    first_text_at: Option<Instant>,
}

impl ReplyAudio<'_> {
    /// The rate `chunk_rate` audio goes out at, announcing the first.
    fn rate(&mut self, chunk_rate: u32) -> Result<u32, String> {
        let (sample_rate, first) = realtime_chunk_rate(&mut self.announced_rate, chunk_rate)?;
        if first {
            self.encoder = Some(AudioOutputEncoder::new(self.format, sample_rate)?);
            send_turn_json(
                self.out_tx,
                self.output_generation,
                json!({
                    "type": "assistant_audio_start",
                    "utterance_id": self.utterance_id,
                    "utterance_seq": self.utterance_seq,
                    "sample_rate": self.format.wire_rate(sample_rate),
                    "audio_format": self.format.as_str(),
                }),
            );
        }
        Ok(sample_rate)
    }

    async fn send(&mut self, samples: &[f32], is_final: bool) -> Result<(), String> {
        let Some(sample_rate) = self.announced_rate else {
            return Ok(());
        };
        if samples.is_empty() && !is_final {
            return Ok(());
        }
        let Some(encoder) = self.encoder.as_mut() else {
            return Ok(());
        };
        let mut encoded = encoder.encode(samples)?;
        if is_final {
            encoded.extend(encoder.finish()?);
        } else if encoded.is_empty() {
            // The resampler holds this audio for the next chunk.
            return Ok(());
        }
        let frame = encode_assistant_audio_binary_frame(
            self.format,
            self.utterance_seq,
            self.chunk_seq,
            self.format.wire_rate(sample_rate),
            is_final,
            &encoded,
        );
        self.chunk_seq = self.chunk_seq.saturating_add(1);
        let duration = Duration::from_secs_f64(samples.len() as f64 / sample_rate.max(1) as f64);
        self.out_tx
            .send_audio(
                self.output_generation,
                Message::Binary(frame.into()),
                encoded.len(),
                duration.as_millis() as u64,
            )
            .await
            .map_err(|err| format!("Outbound audio delivery failed: {err:?}"))?;
        if !samples.is_empty() {
            if let Some(first_text_at) = self.first_text_at.take() {
                self.state
                    .runtime
                    .record_voice_latency(VoiceLatency::TtsFirstAudio, first_text_at.elapsed());
            }
            self.control
                .reply()
                .ledger
                .audio_sent(duration.as_secs_f64() * 1000.0);
            self.control.audio_sent(&self.state.runtime, duration);
        }
        Ok(())
    }
}

/// Speak a reply while the text model writes it (SPK-F2): its text, minus
/// any reasoning, is cut into segments as sentences complete, and the next
/// segment is synthesized while the one before it plays. Segments are
/// joined with the long-form pause and fades (SPK-F20), and each spoken
/// segment is entered in the turn's ledger, so an interruption knows what
/// was heard (SPK-F13).
#[allow(clippy::too_many_arguments)]
async fn speak_reply(
    state: &AppState,
    out_tx: &OutboundTx,
    commit: &PendingAudioCommit,
    control: &TurnControl,
    output_generation: u64,
    mut deltas: tokio::sync::mpsc::UnboundedReceiver<String>,
    text_requested_at: Instant,
    synthesize: impl Fn(String) -> SegmentSynthesis,
) -> Result<(), String> {
    let fallback_sample_rate = state.runtime.sample_rate().await;
    let mut reasoning = ReasoningFilter::default();
    let mut segmenter = SpeechSegmenter::default();
    let mut joiner = SegmentJoiner::default();
    let mut visible = String::new();
    let mut text_done = false;
    let mut spoke = false;
    let mut playing: Option<SegmentSynthesis> = None;
    let mut lookahead: Option<SegmentSynthesis> = None;
    let mut audio = ReplyAudio {
        state,
        out_tx,
        control,
        utterance_id: &commit.utterance_id,
        utterance_seq: commit.utterance_seq,
        output_generation,
        format: commit.turn_config.output_format(),
        encoder: None,
        announced_rate: None,
        chunk_seq: 0,
        first_text_at: None,
    };

    loop {
        if lookahead.is_none() {
            if let Some(text) = segmenter.next_segment(text_done) {
                if !spoke {
                    spoke = true;
                    state.runtime.record_voice_latency(
                        VoiceLatency::LlmFirstToken,
                        text_requested_at.elapsed(),
                    );
                    audio.first_text_at = Some(Instant::now());
                }
                lookahead = Some(synthesize(text));
            }
        }
        if playing.is_none() {
            if let Some(next) = lookahead.take() {
                joiner.begin_segment();
                control.reply().ledger.begin_segment(&next.text);
                playing = Some(next);
                continue;
            }
            if text_done {
                break;
            }
        }

        tokio::select! {
            biased;
            _ = control.cancelled() => return Err("Turn cancelled".to_string()),
            chunk = next_chunk(playing.as_mut()) => match chunk {
                Some(chunk) => {
                    if chunk.samples.is_empty() {
                        continue;
                    }
                    let sample_rate = audio.rate(chunk.sample_rate_or(fallback_sample_rate))?;
                    let samples = joiner.push(&chunk.samples, sample_rate);
                    audio.send(&samples, false).await?;
                }
                None => {
                    if let Some(finished) = playing.take() {
                        finished.finish().await?;
                    }
                    if let Some(sample_rate) = audio.announced_rate {
                        let tail = joiner.end_segment(sample_rate);
                        audio.send(&tail, false).await?;
                    }
                }
            },
            delta = deltas.recv(), if !text_done => {
                let text = match delta {
                    Some(delta) => reasoning.push(&delta),
                    None => {
                        text_done = true;
                        reasoning.finish()
                    }
                };
                if !text.is_empty() {
                    visible.push_str(&text);
                    segmenter.push(&text);
                    send_turn_text_snapshot(
                        out_tx,
                        output_generation,
                        format!("assistant-text-{}", commit.utterance_seq),
                        json!({
                            "type": "assistant_text_snapshot",
                            "utterance_id": commit.utterance_id,
                            "utterance_seq": commit.utterance_seq,
                            "text": visible.trim_start(),
                        }),
                    );
                }
            }
        }
    }

    if audio.announced_rate.is_none() {
        if !spoke {
            // Nothing to say.
            return Ok(());
        }
        audio.rate(fallback_sample_rate)?;
    }
    audio.send(&[], true).await?;
    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "assistant_audio_done",
            "utterance_id": commit.utterance_id,
            "utterance_seq": commit.utterance_seq,
        }),
    );
    Ok(())
}

/// Realtime voice speaks with a built-in voice and streams its audio: a
/// reference-only model (Qwen Base, VibeVoice, Fish) has no voice to speak
/// with here, and one that does not stream would delay the first audio to
/// the end (TPIPE-21).
fn validate_realtime_tts_model(variant: izwi_core::ModelVariant) -> Result<(), String> {
    let capabilities = variant
        .speech_capabilities()
        .ok_or_else(|| format!("Model `{variant}` is not a text-to-speech model."))?;
    if !capabilities.supports_builtin_voices {
        return Err(format!(
            "Model `{variant}` speaks only with a reference voice; realtime voice needs a model with built-in voices."
        ));
    }
    if !capabilities.supports_streaming {
        return Err(format!(
            "Model `{variant}` does not stream audio; realtime voice needs a streaming model."
        ));
    }
    Ok(())
}

/// The rate a realtime utterance's audio is labelled with: its first chunk's
/// (`true` when this chunk announces it). A later chunk at another rate is
/// refused rather than mislabelled.
fn realtime_chunk_rate(
    announced: &mut Option<u32>,
    chunk_rate: u32,
) -> Result<(u32, bool), String> {
    let chunk_rate = chunk_rate.max(1);
    match *announced {
        None => {
            *announced = Some(chunk_rate);
            Ok((chunk_rate, true))
        }
        Some(rate) if rate == chunk_rate => Ok((rate, false)),
        Some(rate) => Err(format!(
            "TTS audio changed sample rate mid-utterance ({rate} Hz to {chunk_rate} Hz)"
        )),
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

    let mut history = Vec::new();
    for turn in session.turns {
        if turn.id == current_turn_id {
            continue;
        }
        for (role, text) in [
            (MemoryMessageRole::User, turn.user_text),
            (MemoryMessageRole::Assistant, turn.assistant_text),
        ] {
            if let Some(content) = text.map(|text| text.trim().to_string()) {
                if !content.is_empty() {
                    history.push(MemoryMessage { role, content });
                }
            }
        }
    }

    Ok(VOICE_HISTORY_BUDGET
        .apply(history)
        .into_iter()
        .map(|message| ChatMessage {
            role: match message.role {
                MemoryMessageRole::System => ChatRole::System,
                MemoryMessageRole::User => ChatRole::User,
                MemoryMessageRole::Assistant => ChatRole::Assistant,
            },
            content: message.content,
        })
        .collect())
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
    output_format: OutputAudioFormat,
    audio_bytes: &[u8],
    runtime_context: RuntimeRequestContext,
    control: &TurnControl,
    output_generation: u64,
    model_leases: &SessionModelLeases,
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

    model_leases
        .hold(&state.runtime, variant)
        .await
        .map_err(|err| format!("Failed to load unified audio model: {err}"))?;

    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "assistant_audio_start",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "sample_rate": output_format.wire_rate(24_000),
            "audio_format": output_format.as_str(),
        }),
    );

    let mut encoder: Option<AudioOutputEncoder> = None;
    let delivery_error = Arc::new(Mutex::new(None::<String>));
    let params = GenerationParams {
        max_tokens: max_output_tokens.clamp(1, 4096),
        speaker: speaker.clone(),
        voice: speaker,
        ..Default::default()
    };

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
                let runtime = state.runtime.clone();
                move |chunk| {
                    // A cancelled turn stops the reply at its next token
                    // instead of letting it run to its end (SPK-F5).
                    let cancelled =
                        || izwi_core::Error::Cancelled("unified voice turn cancelled".to_string());
                    if control.is_cancelled() {
                        return futures::future::Either::Left(std::future::ready(Err(cancelled())));
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

                    let mut frame = None;
                    if !chunk.samples.is_empty() {
                        let encoded = match encoder.as_mut() {
                            Some(encoder) => encoder.encode(&chunk.samples),
                            None => {
                                AudioOutputEncoder::new(output_format, chunk.sample_rate.max(1))
                                    .and_then(|created| {
                                        encoder.insert(created).encode(&chunk.samples)
                                    })
                            }
                        };
                        match encoded {
                            Ok(encoded) => {
                                let chunk_seq = u32::try_from(chunk.sequence).unwrap_or(u32::MAX);
                                let model_rate = chunk.sample_rate.max(1);
                                let duration_ms = (chunk.samples.len() as u64)
                                    .saturating_mul(1_000)
                                    .checked_div(u64::from(model_rate))
                                    .unwrap_or(0);
                                frame = Some((
                                    encode_assistant_audio_binary_frame(
                                        output_format,
                                        utterance_seq,
                                        chunk_seq,
                                        output_format.wire_rate(model_rate),
                                        chunk.is_final,
                                        &encoded,
                                    ),
                                    encoded.len(),
                                    duration_ms,
                                ));
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
                    let Some((frame, bytes, duration_ms)) = frame else {
                        return futures::future::Either::Left(std::future::ready(Ok(())));
                    };
                    // Audio waits for the client's playback credit, as the
                    // cascaded pipeline's does (SPK-F11): a reply generated
                    // faster than real time is held back, not dropped.
                    let out_tx = out_tx.clone();
                    let control = control.clone();
                    let delivery_error = delivery_error.clone();
                    let runtime = runtime.clone();
                    futures::future::Either::Right(async move {
                        match out_tx
                            .send_audio(
                                output_generation,
                                Message::Binary(frame.into()),
                                bytes,
                                duration_ms,
                            )
                            .await
                        {
                            Ok(()) => {
                                control.audio_sent(&runtime, Duration::from_millis(duration_ms));
                                Ok(())
                            }
                            Err(err) => {
                                *delivery_error
                                    .lock()
                                    .unwrap_or_else(|poison| poison.into_inner()) =
                                    Some(format!("Outbound audio delivery failed: {err:?}"));
                                control.cancel("socket_closed");
                                Err(cancelled())
                            }
                        }
                    })
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

/// Interrupt the active turn without waiting for it (SPK-F19): its output
/// is cut off and the interruption announced at once, and a background
/// reaper gives the task [`TURN_CANCELLATION_TIMEOUT`] to stop on its own,
/// aborts it otherwise, and records the turn as interrupted. Returns
/// whether a running turn was interrupted.
fn interrupt_active_turn(
    out_tx: &OutboundTx,
    active_turn: &mut Option<ActiveTurn>,
    reason: &str,
) -> bool {
    let Some(turn) = active_turn.take() else {
        return false;
    };
    out_tx.cutoff_turn(turn.output_generation);
    if turn.task.is_finished() {
        return false;
    }

    turn.control.cancel(reason);
    // The reply was heard up to where playback had got; it is stored with
    // the next turn's history.
    turn.control.stop_playback();
    turn.conversation
        .hold(turn.utterance_seq, turn.control.clone());
    out_tx.advance_typed_output_epoch();
    send_json(
        out_tx,
        json!({
            "type": "turn_interrupted",
            "utterance_id": turn.utterance_id,
            "utterance_seq": turn.utterance_seq,
            "reason": reason,
        }),
    );
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
    tokio::spawn(reap_interrupted_turn(turn, reason.to_string()));
    true
}

async fn reap_interrupted_turn(mut turn: ActiveTurn, reason: String) {
    if tokio::time::timeout(TURN_CANCELLATION_TIMEOUT, &mut turn.task)
        .await
        .is_err()
    {
        turn.task.abort();
        let _ = turn.task.await;
    }
    // A turn that stopped on its own has already recorded this; the record
    // only moves out of `processing` once.
    if let Err(err) = turn
        .state
        .voice_store
        .complete_turn(turn.turn_record_id, "interrupted", Some(reason))
        .await
    {
        warn!("failed to record an interrupted voice turn: {err}");
    }
}

/// The connection's agent session, created by the first modular turn that
/// needs it (off the receive loop) and replaced when the system prompt
/// changes.
#[derive(Default)]
struct AgentSessionSlot {
    /// The session's id and the system prompt it was created with.
    binding: tokio::sync::Mutex<Option<(String, String)>>,
}

impl AgentSessionSlot {
    async fn ensure(
        &self,
        state: &AppState,
        system_prompt: &str,
        text_model_id: &str,
    ) -> Result<String, String> {
        let mut binding = self.binding.lock().await;
        if let Some((session_id, prompt)) = binding.as_ref() {
            if prompt == system_prompt {
                return Ok(session_id.clone());
            }
        }

        let model_id = resolve_chat_model_id(Some(text_model_id))?;
        let thread = state
            .chat_store
            .create_thread(Some("Voice Session".to_string()), Some(model_id.clone()))
            .await
            .map_err(|err| format!("Chat storage error: {err:#}"))?;

        let now = now_unix_millis();
        let session_id = new_uuid();
        state
            .store_agent_session_record(StoredAgentSessionRecord {
                id: session_id.clone(),
                agent_id: DEFAULT_VOICE_AGENT_ID.to_string(),
                thread_id: thread.id,
                model_id,
                system_prompt: system_prompt.to_string(),
                planning_mode: PlanningMode::Auto,
                created_at: now,
                updated_at: now,
            })
            .await;

        *binding = Some((session_id.clone(), system_prompt.to_string()));
        Ok(session_id)
    }
}

/// Run the agent's reply to `input`, streaming its text into `deltas`. The
/// reply is not stored here: the turn's record holds it until what the user
/// heard of it is known (SPK-F13). Returns the raw reply.
#[allow(clippy::too_many_arguments)]
async fn run_agent_turn(
    state: &AppState,
    session_id: &str,
    input: &str,
    model_id: &str,
    max_output_tokens: usize,
    voice_profile_id: Option<&str>,
    correlation_id: &str,
    runtime_context: RuntimeRequestContext,
    control: &TurnControl,
    turn_record_id: &str,
    deltas: tokio::sync::mpsc::UnboundedSender<String>,
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
    {
        let mut reply = control.reply();
        reply.session = Some(session.clone());
        reply.turn_record_id = Some(turn_record_id.to_string());
    }

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
        deltas,
        usage: Mutex::new(None),
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
                history: VOICE_HISTORY_BUDGET,
                persist_assistant_reply: false,
            },
        )
        .await
        .map_err(|err| match err {
            izwi_agent::AgentError::InvalidInput(msg) => msg,
            other => other.to_string(),
        })?;

    {
        let mut reply = control.reply();
        reply.text = Some(strip_reasoning(&result.assistant_text));
        reply.raw_text = Some(result.assistant_text.clone());
        reply.meta = MemoryMessageMeta {
            model_id: Some(resolved_model_id.clone()),
            ..backend.usage()
        };
    }
    state
        .touch_agent_session_record(session_id, now_unix_millis(), resolved_model_id)
        .await;

    Ok(result.assistant_text)
}

/// Store a turn's reply in its conversation thread: what the user heard of
/// it, without reasoning. A reply cut short also records the heard text on
/// its voice turn.
async fn store_reply(state: &AppState, control: &TurnControl) {
    let (session, text, meta, truncated, turn_record_id, raw_text) = {
        let reply = control.reply();
        let (Some(session), Some(text)) = (reply.session.clone(), reply.remembered_text()) else {
            return;
        };
        (
            session,
            text,
            reply.meta.clone(),
            reply.heard_ms.is_some(),
            reply.turn_record_id.clone(),
            reply.raw_text.clone(),
        )
    };
    let memory = VoiceContextMemoryStore::new(state.chat_store.clone(), None);
    if let Err(err) = AgentEngine
        .commit_assistant_reply(&memory, &session, &text, meta)
        .await
    {
        warn!("failed to store a voice reply: {err}");
    }
    if let (true, Some(turn_record_id)) = (truncated, turn_record_id) {
        if let Err(err) = state
            .voice_store
            .update_turn_assistant(turn_record_id, Some(text), raw_text)
            .await
        {
            warn!("failed to record a truncated voice reply: {err}");
        }
    }
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

fn map_typed_voice_event(
    value: &serde_json::Value,
    typed: &mut TypedVoiceSession,
) -> Option<(RealtimeServerEvent, Option<String>, Option<String>)> {
    let event_type = value.get("type")?.as_str()?;
    let utterance_id = value
        .get("utterance_id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let turn_id = value
        .get("utterance_seq")
        .and_then(serde_json::Value::as_u64)
        .map(|sequence| format!("turn-{sequence}"));
    let event = match event_type {
        "session_ready" => RealtimeServerEvent::SessionReady {
            accepted_version: VOICE_REALTIME_VERSION,
            owner_instance_id: typed.owner_instance_id.clone(),
            resumable: false,
            resume_window_ms: 0,
        },
        "input_stream_ready" => RealtimeServerEvent::SessionStarted,
        "user_speech_start" => RealtimeServerEvent::SpeechStarted,
        "user_speech_end" => RealtimeServerEvent::SpeechEnded {
            reason: match value
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("silence")
            {
                "max_duration" => RealtimeSpeechEndReason::MaxDuration,
                "client_pause" => RealtimeSpeechEndReason::ClientPause,
                "stream_stopped" => RealtimeSpeechEndReason::StreamStopped,
                _ => RealtimeSpeechEndReason::Silence,
            },
        },
        "input_sequence_gap" => {
            let expected_frame_sequence = value
                .get("expected_frame_sequence")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            let received_frame_sequence = value
                .get("received_frame_sequence")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default();
            RealtimeServerEvent::AudioGap {
                expected_frame_sequence,
                received_frame_sequence,
                missing_frames: value
                    .get("missing_frames")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_else(|| {
                        received_frame_sequence.saturating_sub(expected_frame_sequence)
                    }),
                action: RealtimeAudioGapAction::Continue,
            }
        }
        "user_speech_rejected" => RealtimeServerEvent::TurnCompleted {
            status: RealtimeTurnStatus::NoInput,
        },
        "user_transcript_snapshot" => RealtimeServerEvent::TranscriptPartial {
            text: value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            revision: typed.next_revision(),
            language: None,
        },
        "user_transcript_final" => RealtimeServerEvent::TranscriptFinal {
            text: value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            revision: typed.next_revision(),
            language: value
                .get("language")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        },
        "assistant_text_start" => RealtimeServerEvent::AssistantTextStarted,
        "assistant_text_snapshot" => RealtimeServerEvent::AssistantTextPartial {
            text: value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            revision: typed.next_revision(),
        },
        "assistant_text_final" => RealtimeServerEvent::AssistantTextFinal {
            text: value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "assistant_audio_start" => RealtimeServerEvent::AssistantAudioStarted {
            sample_rate: value
                .get("sample_rate")
                .and_then(serde_json::Value::as_u64)
                .and_then(|sample_rate| u32::try_from(sample_rate).ok())
                .unwrap_or(24_000),
            channels: 1,
            format: match value
                .get("audio_format")
                .and_then(serde_json::Value::as_str)
            {
                Some("pcm_f32") => RealtimeAudioFormat::PcmF32,
                Some("g711_ulaw") => RealtimeAudioFormat::G711Ulaw,
                Some("g711_alaw") => RealtimeAudioFormat::G711Alaw,
                _ => RealtimeAudioFormat::PcmI16,
            },
        },
        "assistant_audio_done" => RealtimeServerEvent::AssistantAudioCompleted {
            last_chunk_sequence: typed.last_audio_chunk_sequence,
        },
        "turn_interrupted" => RealtimeServerEvent::Interruption {
            reason: match value
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("client_request")
            {
                "barge_in" => RealtimeInterruptionReason::BargeIn,
                "preempted_by_new_turn" => RealtimeInterruptionReason::PreemptedByNewTurn,
                "outbound_audio_saturated" => RealtimeInterruptionReason::Backpressure,
                "socket_closed" => RealtimeInterruptionReason::SessionClosing,
                _ => RealtimeInterruptionReason::ClientRequest,
            },
            cutoff_event_id: typed.next_event_id.saturating_sub(1),
            cutoff_sequence: typed.next_sequence.saturating_sub(1),
        },
        "turn_done" => RealtimeServerEvent::TurnCompleted {
            status: match value
                .get("status")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("error")
            {
                "ok" => RealtimeTurnStatus::Ok,
                "no_input" => RealtimeTurnStatus::NoInput,
                "interrupted" => RealtimeTurnStatus::Interrupted,
                "timeout" => RealtimeTurnStatus::Timeout,
                _ => RealtimeTurnStatus::Error,
            },
        },
        "error" => {
            let message = value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("Voice realtime error")
                .to_string();
            let code = match value
                .get("code")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
            {
                "realtime_session_capacity" => RealtimeErrorCode::Backpressure,
                "buffer_limit" => RealtimeErrorCode::BufferLimit,
                "model_unavailable" => RealtimeErrorCode::ModelUnavailable,
                _ => RealtimeErrorCode::InferenceFailed,
            };
            if value
                .get("fatal")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                RealtimeServerEvent::FatalError {
                    code,
                    message,
                    close: RealtimeClose {
                        code: RealtimeCloseCode::InternalError,
                        reason: RealtimeCloseReason::InternalError,
                        message: "voice realtime session failed".to_string(),
                        retryable: false,
                    },
                }
            } else {
                RealtimeServerEvent::RecoverableError {
                    code,
                    message,
                    retry_after_ms: None,
                }
            }
        }
        "pong" => RealtimeServerEvent::Pong {
            client_timestamp_ms: value
                .get("timestamp_ms")
                .and_then(serde_json::Value::as_u64),
            server_timestamp_ms: value
                .get("server_time_ms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(now_unix_millis),
        },
        // These legacy-only state markers either have no typed equivalent or would encode
        // append-only deltas. Typed clients reconstruct state from full snapshots/finals.
        "input_stream_stopped"
        | "models_ready"
        | "conversation_item_truncated"
        | "turn_processing"
        | "user_transcript_start"
        | "user_transcript_delta"
        | "assistant_text_delta" => return None,
        _ => return None,
    };
    Some((event, utterance_id, turn_id))
}

fn send_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match out_tx.encode_json_event(&value) {
        Ok(Some(text)) => out_tx.send(Message::Text(text.into())),
        Ok(None) => true,
        Err(err) => {
            warn!("failed to serialize voice ws event: {err}");
            false
        }
    }
}

fn send_turn_json(out_tx: &OutboundTx, output_generation: u64, value: serde_json::Value) -> bool {
    if !out_tx.accepts_turn_generation(output_generation) {
        return false;
    }
    match out_tx.encode_json_event(&value) {
        Ok(Some(text)) => out_tx.send_turn(output_generation, Message::Text(text.into())),
        Ok(None) => true,
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
    if !out_tx.accepts_turn_generation(output_generation) {
        return false;
    }
    match out_tx.encode_json_event(&value) {
        Ok(Some(text)) if out_tx.typed_protocol_enabled() => {
            out_tx.send_turn(output_generation, Message::Text(text.into()))
        }
        Ok(Some(text)) => {
            out_tx.send_text_snapshot(output_generation, key, Message::Text(text.into()))
        }
        Ok(None) => true,
        Err(err) => {
            warn!("failed to serialize voice ws text snapshot: {err}");
            false
        }
    }
}

fn send_diagnostic_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match out_tx.encode_json_event(&value) {
        Ok(Some(text)) if out_tx.typed_protocol_enabled() => {
            out_tx.send(Message::Text(text.into()))
        }
        Ok(Some(text)) => out_tx.send_diagnostic(Message::Text(text.into())),
        Ok(None) => true,
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
    /// Receives the reply's text as it is generated.
    deltas: tokio::sync::mpsc::UnboundedSender<String>,
    usage: Mutex<Option<(usize, f64)>>,
}

impl IzwiRuntimeBackend {
    fn usage(&self) -> MemoryMessageMeta {
        let usage = *self
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        MemoryMessageMeta {
            model_id: None,
            tokens_generated: usage.map(|(tokens, _)| tokens),
            generation_time_ms: usage.map(|(_, ms)| ms),
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for IzwiRuntimeBackend {
    /// A streamed reply without reasoning: voice cannot speak thinking, and
    /// waiting for it would hold back the first audio (SPK-F2).
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

        let deltas = self.deltas.clone();
        let generation = self
            .runtime
            .chat_generate_streaming_with_runtime_context(
                variant,
                runtime_messages,
                GenerationParams {
                    max_tokens: request.max_output_tokens.clamp(1, 4096),
                    ..Default::default()
                },
                ChatRequestConfig {
                    enable_thinking: Some(false),
                    truncation: ChatTruncation::Auto,
                    ..ChatRequestConfig::default()
                },
                Some(&self.correlation_id),
                self.runtime_context,
                move |delta| {
                    let _ = deltas.send(delta);
                },
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        *self
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) =
            Some((generation.tokens_generated, generation.generation_time_ms));
        Ok(ModelOutput {
            text: generation.text,
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An app state on a temporary database with `realtime` realtime
    /// admission permits.
    fn test_app_state(realtime: usize) -> (tempfile::TempDir, AppState) {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let models_dir = temp_dir.path().join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir");
        let _env = crate::test_support::env_lock();
        std::env::set_var("IZWI_DB_PATH", temp_dir.path().join("izwi.sqlite3"));
        std::env::set_var("IZWI_MEDIA_DIR", temp_dir.path().join("media"));
        std::env::set_var("IZWI_MAX_REALTIME_REQUESTS", realtime.to_string());
        let serve_config = izwi_core::ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            models_dir,
            ..izwi_core::ServeRuntimeConfig::default()
        };
        let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
        let state = AppState::new(runtime, &serve_config).expect("state");
        for key in [
            "IZWI_DB_PATH",
            "IZWI_MEDIA_DIR",
            "IZWI_MAX_REALTIME_REQUESTS",
        ] {
            std::env::remove_var(key);
        }
        (temp_dir, state)
    }

    fn realtime_permits_available(state: &AppState) -> usize {
        state.request_admission_snapshot().realtime.available
    }

    /// SPK-F22: a turn stage holds a realtime permit only while it runs, so
    /// a turn idle between stages holds none, and a stage waiting for one
    /// starts as soon as another stage ends.
    #[tokio::test]
    async fn turn_stages_hold_a_permit_only_while_they_run() {
        let (_dir, state) = test_app_state(1);
        assert_eq!(realtime_permits_available(&state), 1);
        let inside = with_stage_permit(&state, async |_| Ok(realtime_permits_available(&state)))
            .await
            .expect("stage");
        assert_eq!(inside, 0, "the stage holds the permit");
        assert_eq!(
            realtime_permits_available(&state),
            1,
            "the turn holds no permit between stages"
        );

        // With one permit, a second stage waits for the first to end.
        let (release, released) = tokio::sync::oneshot::channel::<()>();
        let first = with_stage_permit(&state, async |_| {
            released.await.ok();
            Ok(())
        });
        let second = async {
            tokio::task::yield_now().await;
            with_stage_permit(&state, async |_| Ok(())).await
        };
        let second_waits = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            release.send(()).ok();
        };
        let (first, second, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(first, second, second_waits)
        })
        .await
        .expect("the second stage runs once the first ends");
        first.expect("first stage");
        second.expect("second stage");
        assert_eq!(realtime_permits_available(&state), 1);
    }

    /// A stored voice session with one processing turn, `turn_id`.
    async fn stored_turn(state: &AppState, turn_id: &str) -> String {
        let profile = state
            .voice_store
            .get_default_profile()
            .await
            .expect("profile");
        let session = state
            .voice_store
            .create_session(crate::voice_store::CreateVoiceSessionRequest {
                profile_id: profile.id,
                mode: "modular".to_string(),
                system_prompt: profile.system_prompt,
            })
            .await
            .expect("session");
        state
            .voice_store
            .create_turn(CreateVoiceTurnRequest {
                turn_id: Some(turn_id.to_string()),
                session_id: session.id.clone(),
                utterance_id: "utt-1".to_string(),
                utterance_seq: 1,
                mode: "modular".to_string(),
                vad_end_reason: None,
                asr_model_id: None,
                text_model_id: None,
                tts_model_id: None,
                s2s_model_id: None,
                speaker: None,
            })
            .await
            .expect("turn");
        session.id
    }

    async fn turn_status(state: &AppState, session_id: &str) -> String {
        let session = state
            .voice_store
            .get_session(session_id.to_string())
            .await
            .expect("session")
            .expect("stored session");
        session.turns[0].status.clone()
    }

    async fn drain_text(out_tx: &OutboundTx) -> Vec<serde_json::Value> {
        let mut events = Vec::new();
        out_tx.close();
        while let Some(output) = out_tx.mailbox.next().await {
            if let crate::app::realtime_transport::MailboxOutput::Message(Message::Text(text)) =
                output
            {
                events.push(serde_json::from_str(text.as_str()).expect("json"));
            }
        }
        events
    }

    /// SPK-F19: interrupting a turn that does not stop on its own returns at
    /// once with the interruption announced; a background reaper aborts the
    /// turn and records it as interrupted.
    #[tokio::test]
    async fn interrupting_a_turn_does_not_wait_for_it() {
        let (_dir, state) = test_app_state(1);
        let session_id = stored_turn(&state, "turn-1").await;
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let stubborn = tokio::spawn(tokio::time::sleep(Duration::from_secs(30)));
        let mut active = Some(ActiveTurn {
            utterance_id: "utt-1".to_string(),
            utterance_seq: 1,
            turn_record_id: "turn-1".to_string(),
            state: state.clone(),
            conversation: Arc::default(),
            task: stubborn,
            control: TurnControl::new(Instant::now()),
            output_generation: 1,
        });

        let started = Instant::now();
        assert!(interrupt_active_turn(&out_tx, &mut active, "barge_in"));
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "the receive loop is not held for the turn ({:?})",
            started.elapsed()
        );
        assert!(active.is_none());
        let deadline = Instant::now() + Duration::from_secs(5);
        while turn_status(&state, &session_id).await != "interrupted" {
            assert!(
                Instant::now() < deadline,
                "the reaper records the interruption"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let types: Vec<String> = drain_text(&out_tx)
            .await
            .iter()
            .map(|event| event["type"].as_str().unwrap_or_default().to_string())
            .collect();
        assert_eq!(types, ["turn_interrupted", "turn_done"]);
    }

    /// SPK-F21: starting an input stream loads and leases its models in the
    /// background and reports their readiness, before any turn.
    #[tokio::test]
    async fn input_stream_start_warms_the_session_models() {
        let (_dir, state) = test_app_state(1);
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let mut conn = ConnectionState::default();
        for message in [
            json!({ "type": "session_start" }),
            json!({
                "type": "input_stream_start",
                "asr_model_id": "Parakeet-TDT-0.6B-v3",
                "text_model_id": "LFM2.5-1.2B-Instruct-GGUF",
                "tts_model_id": "Kokoro-82M",
            }),
        ] {
            handle_text_message(&state, "corr", &out_tx, &mut conn, &message.to_string())
                .await
                .expect("message");
        }
        conn.model_warmup
            .take()
            .expect("the stream warms its models")
            .join()
            .await
            .expect("warm-up");
        let events = drain_text(&out_tx).await;
        let ready = events
            .iter()
            .find(|event| event["type"] == "models_ready")
            .expect("readiness is reported");
        let stages: Vec<&str> = ready["models"]
            .as_array()
            .expect("models")
            .iter()
            .map(|model| model["stage"].as_str().expect("stage"))
            .collect();
        assert_eq!(stages, ["asr", "text", "tts"]);
        // The test models directory is empty, so none can load; each says why.
        assert!(ready["models"]
            .as_array()
            .expect("models")
            .iter()
            .all(|model| model["ready"] == false && model["error"].is_string()));
    }

    /// A turn control whose reply spoke "One two." over its first second of
    /// audio and "Three four five six." over the next two, in `thread_id`.
    fn spoken_reply(thread_id: &str) -> TurnControl {
        let control = TurnControl::new(Instant::now());
        let mut reply = control.reply();
        reply.session = Some(AgentSession {
            id: "agent-session".to_string(),
            agent_id: DEFAULT_VOICE_AGENT_ID.to_string(),
            thread_id: thread_id.to_string(),
            created_at: 0,
            updated_at: 0,
        });
        reply.text = Some("One two. Three four five six.".to_string());
        reply.ledger.begin_segment("One two.");
        reply.ledger.audio_sent(1_000.0);
        reply.ledger.begin_segment("Three four five six.");
        reply.ledger.audio_sent(2_000.0);
        drop(reply);
        control
    }

    async fn thread_text(state: &AppState, thread_id: &str) -> Vec<String> {
        state
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .expect("messages")
            .into_iter()
            .map(|message| message.content)
            .collect()
    }

    /// SPK-F13: a client's truncate makes the conversation remember the
    /// reply only up to what was played, and is acknowledged with that text.
    #[tokio::test]
    async fn a_truncated_reply_is_remembered_up_to_what_was_played() {
        let (_dir, state) = test_app_state(1);
        let thread = state
            .chat_store
            .create_thread(Some("Voice Session".to_string()), None)
            .await
            .expect("thread");
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let mut conn = ConnectionState::default();
        conn.conversation.hold(4, spoken_reply(&thread.id));

        let truncate = json!({
            "type": "conversation.item.truncate",
            "utterance_seq": 4,
            "audio_end_ms": 2_000,
        });
        handle_text_message(&state, "corr", &out_tx, &mut conn, &truncate.to_string())
            .await
            .expect("truncate");
        assert!(handle_text_message(
            &state,
            "corr",
            &out_tx,
            &mut conn,
            &json!({"type": "conversation_item_truncate", "utterance_seq": 9, "audio_end_ms": 1})
                .to_string(),
        )
        .await
        .expect_err("no such reply")
        .contains("utterance 9"));
        conn.conversation.store(&state).await;

        assert_eq!(
            thread_text(&state, &thread.id).await,
            ["One two. Three four"]
        );
        let events = drain_text(&out_tx).await;
        assert_eq!(events[0]["type"], "conversation_item_truncated");
        assert_eq!(events[0]["text"], "One two. Three four");
    }

    /// SPK-F13: a reply that played out is remembered whole; one interrupted
    /// while playing is remembered up to where the client's playback had got.
    #[tokio::test]
    async fn an_interrupted_reply_is_remembered_up_to_its_playback() {
        let (_dir, state) = test_app_state(1);
        let thread = state
            .chat_store
            .create_thread(Some("Voice Session".to_string()), None)
            .await
            .expect("thread");
        let conversation = Arc::new(VoiceConversation::default());

        conversation.hold(1, spoken_reply(&thread.id));
        conversation.store(&state).await;

        let interrupted = spoken_reply(&thread.id);
        *interrupted.inner.playback.lock().expect("playback") = PlaybackClock {
            playing_until: Some(Instant::now() + Duration::from_millis(1_000)),
            sent: Duration::from_millis(3_000),
        };
        conversation.hold(2, interrupted);
        conversation.interrupt_playback();
        conversation.store(&state).await;

        assert_eq!(
            thread_text(&state, &thread.id).await,
            ["One two. Three four five six.", "One two. Three four"]
        );
    }

    /// A fake TTS for `speak_reply`: segment `n` (from 1) is 3 chunks of 100
    /// samples of value `n / 10` at 1 kHz; the first segment is slow (30 ms
    /// between chunks), later ones instant. Records each segment's start and
    /// the moment its synthesis ends.
    fn fake_synthesis(
        log: Arc<Mutex<Vec<(String, Instant, Option<Instant>)>>>,
    ) -> impl Fn(String) -> SegmentSynthesis {
        move |text| {
            let index = {
                let mut log = log.lock().expect("log");
                log.push((text.clone(), Instant::now(), None));
                log.len()
            };
            let (tx, chunks) = tokio::sync::mpsc::channel(32);
            let log = log.clone();
            let task = AbortOnDropTask::new(tokio::spawn(async move {
                for sequence in 0..3 {
                    if index == 1 {
                        tokio::time::sleep(Duration::from_millis(30)).await;
                    }
                    let chunk = izwi_core::AudioChunk::new(
                        format!("segment-{index}"),
                        sequence,
                        vec![index as f32 / 10.0; 100],
                    )
                    .with_sample_rate(1_000);
                    tx.send(chunk).await.map_err(|err| err.to_string())?;
                }
                log.lock().expect("log")[index - 1].2 = Some(Instant::now());
                Ok(())
            }));
            SegmentSynthesis { text, chunks, task }
        }
    }

    /// The PCM16 samples of the assistant audio frames queued on `out_tx`.
    async fn assistant_samples(out_tx: &OutboundTx) -> Vec<i16> {
        out_tx.close();
        let mut samples = Vec::new();
        while let Some(output) = out_tx.mailbox.next().await {
            if let crate::app::realtime_transport::MailboxOutput::Message(Message::Binary(frame)) =
                output
            {
                samples.extend(pcm16_bytes_to_i16(&frame[WS_BIN_ASSISTANT_HEADER_LEN..]));
            }
        }
        samples
    }

    /// SPK-F2/F20: speech starts from the first sentence while the text is
    /// still arriving, the next segment synthesizes while the first plays,
    /// and the audio goes out in segment order with a pause between
    /// segments, even when a later segment synthesizes faster.
    #[tokio::test]
    async fn the_reply_is_spoken_while_written_with_one_segment_ahead() {
        const REPLY: &str = "Hello there, it is lovely to hear from you today. The second \
            sentence is a little longer than the first one. Third and last.";
        let (_dir, state) = test_app_state(4);
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let control = TurnControl::new(Instant::now());
        let commit = PendingAudioCommit {
            utterance_id: "utt-1".to_string(),
            utterance_seq: 1,
            turn_config: test_streaming_input().config.turn_config,
            endpointed_at: Instant::now(),
            endpointing_delay: None,
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
        let writer = async {
            for delta in [
                "<think>plan</think>Hello there, it is lovely to hear from you today.",
                " The second sentence",
                " is a little longer than the first one.",
            ] {
                deltas.send(delta.to_string()).expect("delta");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            deltas.send(" Third and last.".to_string()).expect("delta");
            drop(deltas);
        };
        let speech = speak_reply(
            &state,
            &out_tx,
            &commit,
            &control,
            1,
            delta_rx,
            Instant::now(),
            fake_synthesis(log.clone()),
        );
        let ((), spoken) = tokio::join!(writer, speech);
        spoken.expect("spoken");

        let log = log.lock().expect("log").clone();
        let texts: Vec<&str> = log.iter().map(|(text, _, _)| text.as_str()).collect();
        assert_eq!(
            texts[0], "Hello there, it is lovely to hear from you today.",
            "the first sentence is spoken alone"
        );
        assert_eq!(texts.join(" "), REPLY);
        let first_done = log[0].2.expect("first segment finished");
        assert!(
            log[1].1 < first_done,
            "the second segment starts while the first is still synthesizing"
        );

        let samples = assistant_samples(&out_tx).await;
        let pause = crate::api::tts_long_form::segment_pause_samples(1_000);
        let segments = texts.len();
        assert_eq!(samples.len(), 300 * segments + pause * (segments - 1));
        // Each segment's audio, in segment order, with the pause between.
        for segment in 0..segments {
            let start = segment * (300 + pause);
            let block = &samples[start..start + 300];
            let level = (32_767.0 * (segment + 1) as f32 / 10.0) as i16;
            assert_eq!(block.iter().max(), Some(&level), "segment {segment}");
            if segment + 1 < segments {
                assert!(samples[start + 300..start + 300 + pause]
                    .iter()
                    .all(|sample| *sample == 0));
            }
        }
        assert_eq!(control.reply().ledger.spoken_text(), REPLY);
    }

    /// SPK-F11: a session that asked for G.711 gets μ-law frames at 8 kHz,
    /// announced as such, one byte a sample.
    #[tokio::test]
    async fn the_reply_goes_out_in_the_session_codec() {
        let (_dir, state) = test_app_state(4);
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let control = TurnControl::new(Instant::now());
        let commit = PendingAudioCommit {
            utterance_id: "utt-1".to_string(),
            utterance_seq: 1,
            turn_config: VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
                asr_model_id: "Parakeet-TDT-0.6B-v3".to_string(),
                text_model_id: "LFM2.5-1.2B-Instruct-GGUF".to_string(),
                tts_model_id: "Kokoro-82M".to_string(),
                speaker: None,
                asr_language: None,
                max_output_tokens: 64,
                output_format: OutputAudioFormat::G711Ulaw,
            }),
            endpointed_at: Instant::now(),
            endpointing_delay: None,
        };
        let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
        deltas.send("Hello there.".to_string()).expect("delta");
        drop(deltas);
        speak_reply(
            &state,
            &out_tx,
            &commit,
            &control,
            1,
            delta_rx,
            Instant::now(),
            fake_synthesis(Arc::default()),
        )
        .await
        .expect("spoken");

        out_tx.close();
        let (mut start, mut payload, mut kinds) = (None, 0, Vec::new());
        while let Some(output) = out_tx.mailbox.next().await {
            match output {
                crate::app::realtime_transport::MailboxOutput::Message(Message::Text(text)) => {
                    let event: serde_json::Value =
                        serde_json::from_str(text.as_str()).expect("json");
                    if event["type"] == "assistant_audio_start" {
                        start = Some(event);
                    }
                }
                crate::app::realtime_transport::MailboxOutput::Message(Message::Binary(frame)) => {
                    kinds.push(frame[5]);
                    assert_eq!(&frame[20..24], &8_000_u32.to_le_bytes());
                    payload += frame.len() - WS_BIN_ASSISTANT_HEADER_LEN;
                }
                _ => {}
            }
        }
        let start = start.expect("audio start");
        assert_eq!(start["audio_format"], "g711_ulaw");
        assert_eq!(start["sample_rate"], 8_000);
        assert!(kinds
            .iter()
            .all(|kind| *kind == WS_BIN_KIND_ASSISTANT_G711_ULAW));
        // 300 samples at 1 kHz is 2 400 at 8 kHz, one byte each.
        assert!((payload as i64 - 2_400).abs() <= 16, "{payload} bytes");
    }

    fn test_streaming_input() -> StreamingInputState {
        StreamingInputState::new(StreamingInputConfig {
            turn_config: VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
                s2s_model_id: "LFM2.5-Audio-1.5B".to_string(),
                speaker: None,
                max_output_tokens: 32,
                output_format: OutputAudioFormat::PcmI16,
            }),
            vad_threshold: DEFAULT_STREAM_VAD_THRESHOLD,
            min_speech_ms: 50,
            silence_duration_ms: 200,
            max_utterance_ms: 10_000,
            pre_roll_ms: 100,
            barge_in_min_speech_ms: DEFAULT_BARGE_IN_MIN_SPEECH_MS,
            snr_margin_db: izwi_vad::DEFAULT_SNR_MARGIN_DB,
            input_sample_rate_hint: Some(16_000),
        })
    }

    /// Every frame voiced, whatever its content: the score threshold drops
    /// to the minimum and the SNR check is off, since digital silence never
    /// rises over its own noise floor.
    fn always_voiced(input: &mut StreamingInputState) {
        input.config.vad_threshold = 0.01;
        input.config.snr_margin_db = 0.0;
        input.endpoint = EndpointDetector::new(input.config.endpoint_config());
    }

    fn finalized(
        outcome: Option<UtteranceOutcome>,
    ) -> Option<(PendingAudioCommit, Vec<u8>, UtteranceEndReason)> {
        match outcome {
            Some(UtteranceOutcome::Finalized(commit, wav_bytes, reason)) => {
                Some((commit, wav_bytes, reason))
            }
            _ => None,
        }
    }

    /// Realtime voice refuses a reference-only TTS model at session
    /// configuration and takes every built-in-voice model, all of which
    /// stream.
    #[test]
    fn realtime_tts_needs_a_built_in_streaming_voice() {
        use izwi_core::ModelVariant;
        assert!(validate_realtime_tts_model(ModelVariant::Kokoro82M).is_ok());
        assert!(validate_realtime_tts_model(ModelVariant::Qwen3Tts12Hz06BCustomVoice).is_ok());
        assert!(
            validate_realtime_tts_model(ModelVariant::Qwen3Tts12Hz06BBase)
                .unwrap_err()
                .contains("reference voice")
        );
        // Voxtral and LFM2.5 Audio stream committed audio now.
        assert!(validate_realtime_tts_model(ModelVariant::Voxtral4BTts2603).is_ok());
        assert!(validate_realtime_tts_model(ModelVariant::Lfm25Audio15BGguf).is_ok());
    }

    /// TPIPE-22: an utterance is labelled with its first chunk's rate (44.1
    /// kHz for Fish), and a chunk at another rate is refused.
    #[test]
    fn realtime_audio_is_labelled_with_the_models_rate() {
        let mut announced = None;
        assert_eq!(
            realtime_chunk_rate(&mut announced, 44_100),
            Ok((44_100, true))
        );
        assert_eq!(
            realtime_chunk_rate(&mut announced, 44_100),
            Ok((44_100, false))
        );
        assert!(realtime_chunk_rate(&mut announced, 24_000).is_err());
    }

    #[test]
    fn voice_protocol_negotiation_preserves_legacy_and_requires_explicit_v2() {
        assert_eq!(
            negotiate_voice_protocol(None, None, None),
            Ok(VoiceWireProtocol::LegacyV1)
        );
        assert_eq!(
            negotiate_voice_protocol(
                Some(TYPED_VOICE_REALTIME_PROTOCOL),
                Some(VOICE_REALTIME_VERSION),
                None,
            ),
            Ok(VoiceWireProtocol::TypedV2)
        );
        assert!(negotiate_voice_protocol(
            Some(TYPED_VOICE_REALTIME_PROTOCOL),
            Some(VOICE_REALTIME_VERSION),
            Some(7),
        )
        .expect_err("resume must be rejected")
        .contains("non-resumable"));
    }

    #[test]
    fn typed_voice_session_ready_has_a_stable_golden_shape() {
        let mut typed =
            TypedVoiceSession::new("voice-session-1".to_string(), "process-42".to_string());
        let (event, utterance_id, turn_id) = map_typed_voice_event(
            &json!({
                "type": "session_ready",
                "session_id": "voice-session-1",
                "owner_instance_id": "process-42",
            }),
            &mut typed,
        )
        .expect("typed session ready mapping");
        let mut envelope = typed.envelope(event, utterance_id, turn_id);
        envelope.timestamp_ms = 1_725_000_000_123;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../ui/src/features/voice/realtime/voice_v2_session_ready.fixture.json"
        ))
        .expect("shared voice v2 SessionReady fixture");
        assert_eq!(
            serde_json::to_value(&envelope).expect("serialize typed voice SessionReady"),
            fixture
        );
    }

    #[test]
    fn typed_voice_output_cutoff_starts_a_contiguous_new_epoch() {
        let runtime =
            Arc::new(RuntimeService::new(izwi_core::EngineConfig::default()).expect("runtime"));
        let outbound = OutboundTx::new(runtime, "voice typed test");
        outbound.configure_wire_protocol(
            VoiceWireProtocol::TypedV2,
            "voice-session-1".to_string(),
            "process-42".to_string(),
        );
        let first: RealtimeEventEnvelope<RealtimeServerEvent> = serde_json::from_str(
            outbound
                .encode_json_event(&json!({"type": "input_stream_ready"}))
                .expect("encode first event")
                .expect("mapped first event")
                .as_str(),
        )
        .expect("first envelope");
        outbound.advance_typed_output_epoch();
        let next: RealtimeEventEnvelope<RealtimeServerEvent> = serde_json::from_str(
            outbound
                .encode_json_event(&json!({
                    "type": "turn_interrupted",
                    "utterance_id": "utterance-1",
                    "utterance_seq": 1,
                    "reason": "barge_in",
                }))
                .expect("encode cutoff event")
                .expect("mapped cutoff event")
                .as_str(),
        )
        .expect("cutoff envelope");

        assert_eq!(first.connection_epoch, 0);
        assert_eq!(first.sequence, 0);
        assert_eq!(next.connection_epoch, 1);
        assert_eq!(next.sequence, 0);
        assert_eq!(next.validate_successor(&first), Ok(()));
    }

    #[tokio::test]
    async fn turn_control_wakes_cooperative_cleanup() {
        let control = TurnControl::new(Instant::now());
        let waiting_control = control.clone();
        let waiter = tokio::spawn(async move {
            waiting_control.cancelled().await;
            waiting_control.is_cancelled()
        });

        control.cancel("test_cancel");

        assert!(tokio::time::timeout(Duration::from_millis(50), waiter)
            .await
            .expect("cancellation should wake waiter")
            .expect("waiter should finish"));
        assert_eq!(control.reason().as_deref(), Some("test_cancel"));
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

    fn pcm16_frame(samples: &[i16]) -> Vec<u8> {
        samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    }

    /// SPK-F3: a speech start alone does not barge in. Sustained speech
    /// confirms exactly one barge-in once it reaches the configured voiced
    /// duration.
    #[test]
    fn barge_in_waits_for_sustained_speech() {
        let mut input = test_streaming_input();
        // Each frame is voiced, so the utterance is pure sustained speech.
        always_voiced(&mut input);
        let frame = pcm16_frame(&[0; 320]);

        let mut barge_ins = Vec::new();
        let mut voiced_before = 0.0_f32;
        for frame_seq in 1..=40_u32 {
            let result = input
                .handle_pcm16_frame(frame_seq, 16_000, &frame)
                .expect("frame");
            if frame_seq == 1 {
                assert!(result.speech_start.is_some());
            }
            let voiced = input.active.as_ref().expect("active utterance").voiced_ms;
            if result.barge_in {
                barge_ins.push(frame_seq);
                assert!(voiced >= DEFAULT_BARGE_IN_MIN_SPEECH_MS as f32);
                assert!(voiced_before < DEFAULT_BARGE_IN_MIN_SPEECH_MS as f32);
            }
            voiced_before = voiced;
        }
        assert_eq!(barge_ins.len(), 1, "one barge-in per utterance");
        assert!(barge_ins[0] > 1, "the first speech frame must not barge in");
    }

    /// SPK-F3: a short noise burst (a cough or a door) never barges in or
    /// becomes a turn; any speech start it causes is rejected. A burst out of
    /// silence clears any noise floor, so this holds on the model score: at
    /// the default threshold, not a lowered one where its mid-range scores
    /// sustain an utterance.
    #[test]
    fn short_noise_burst_never_barges_in() {
        let mut input = StreamingInputState::new(StreamingInputConfig {
            min_speech_ms: DEFAULT_STREAM_MIN_SPEECH_MS,
            ..test_streaming_input().config
        });
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let burst: Vec<i16> = (0..8_000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20_000.0) as i16
            })
            .collect();
        let silence = vec![0_i16; 320];
        let frames = std::iter::repeat_n(&silence[..], 15)
            .chain(burst.chunks(320))
            .chain(std::iter::repeat_n(&silence[..], 30));

        let (mut starts, mut rejections) = (0, 0);
        for (index, frame) in frames.enumerate() {
            let result = input
                .handle_pcm16_frame(index as u32 + 1, 16_000, &pcm16_frame(frame))
                .expect("frame");
            starts += usize::from(result.speech_start.is_some());
            rejections += usize::from(matches!(
                result.utterance_end,
                Some(UtteranceOutcome::Rejected(_))
            ));
            assert!(!result.barge_in, "a short burst must not barge in");
            assert!(finalized(result.utterance_end).is_none());
        }
        assert_eq!(rejections, starts, "every start is rejected as noise");
    }

    /// SPK-F4: a network stall longer than the silence window, followed by
    /// the delayed but contiguous frames, continues the same utterance. Only
    /// a client that stops sending for the stall allowance is paused.
    #[test]
    fn network_stall_does_not_end_an_utterance() {
        let mut input = test_streaming_input();
        always_voiced(&mut input);
        let frame = pcm16_frame(&[0; 320]);
        for frame_seq in 1..=10_u32 {
            input
                .handle_pcm16_frame(frame_seq, 16_000, &frame)
                .expect("frame");
        }
        let utterance_id = input.active.as_ref().expect("speech").utterance_id.clone();
        let silence_ms = u64::from(input.config.silence_duration_ms);

        input.last_frame_at = Some(Instant::now() - Duration::from_millis(silence_ms + 900));
        assert!(input.on_tick().expect("tick").is_none());
        for frame_seq in 11..=20_u32 {
            let result = input
                .handle_pcm16_frame(frame_seq, 16_000, &frame)
                .expect("late frame");
            assert!(result.speech_start.is_none());
            assert!(result.utterance_end.is_none());
        }
        assert_eq!(
            input.active.as_ref().expect("same utterance").utterance_id,
            utterance_id
        );

        input.last_frame_at = Some(
            Instant::now() - Duration::from_millis(silence_ms + CLIENT_PAUSE_STALL_ALLOWANCE_MS),
        );
        let (commit, _, reason) = finalized(input.on_tick().expect("tick"))
            .expect("a client that stopped sending is paused");
        assert_eq!(commit.utterance_id, utterance_id);
        assert!(matches!(reason, UtteranceEndReason::ClientPause));
        // SPK-F6: the endpointing delay a pause reports covers the stall.
        assert!(
            commit.endpointing_delay.expect("a pause endpoints")
                >= Duration::from_millis(silence_ms + CLIENT_PAUSE_STALL_ALLOWANCE_MS)
        );
    }

    /// SPK-F6: an utterance ended by silence reports the trailing silence the
    /// detector waited for; a forced end reports none.
    #[test]
    fn silence_endpointing_reports_the_trailing_silence() {
        let mut input = test_streaming_input();
        always_voiced(&mut input);
        let frame = pcm16_frame(&[0; 320]);
        for frame_seq in 1..=10_u32 {
            input
                .handle_pcm16_frame(frame_seq, 16_000, &frame)
                .expect("frame");
        }
        input.active.as_mut().expect("speech").silence_ms = 240.0;
        let (commit, _, _) = finalized(
            input
                .finalize_active_utterance(UtteranceEndReason::Silence)
                .expect("finalize"),
        )
        .expect("voiced utterance");
        assert_eq!(commit.endpointing_delay, Some(Duration::from_millis(240)));

        for frame_seq in 11..=20_u32 {
            input
                .handle_pcm16_frame(frame_seq, 16_000, &frame)
                .expect("frame");
        }
        let (commit, _, _) = finalized(
            input
                .finalize_active_utterance(UtteranceEndReason::MaxDuration)
                .expect("finalize"),
        )
        .expect("voiced utterance");
        assert_eq!(commit.endpointing_delay, None);
    }

    /// An utterance that ends without enough voiced speech is rejected, never
    /// dropped silently: the client already saw its `user_speech_start`.
    #[test]
    fn short_utterance_is_rejected_on_every_forced_end() {
        let frame = pcm16_frame(&[0; 320]);
        let start = |input: &mut StreamingInputState| {
            always_voiced(input);
            input.config.min_speech_ms = 300;
            input.endpoint = EndpointDetector::new(input.config.endpoint_config());
            for frame_seq in 1..=5_u32 {
                input
                    .handle_pcm16_frame(frame_seq, 16_000, &frame)
                    .expect("frame");
            }
            input.active.as_ref().expect("speech").utterance_id.clone()
        };
        let rejected = |outcome: Option<UtteranceOutcome>| match outcome {
            Some(UtteranceOutcome::Rejected(evt)) => evt.utterance_id,
            other => panic!("expected a rejection, got {other:?}"),
        };

        let mut input = test_streaming_input();
        let utterance_id = start(&mut input);
        let outcome = input
            .finalize_active_utterance(UtteranceEndReason::MaxDuration)
            .expect("finalize");
        assert_eq!(rejected(outcome), utterance_id);

        let mut input = test_streaming_input();
        let utterance_id = start(&mut input);
        input.last_frame_at = Some(
            Instant::now()
                - Duration::from_millis(
                    u64::from(input.config.silence_duration_ms) + CLIENT_PAUSE_STALL_ALLOWANCE_MS,
                ),
        );
        assert_eq!(rejected(input.on_tick().expect("tick")), utterance_id);

        let mut input = test_streaming_input();
        let utterance_id = start(&mut input);
        assert_eq!(rejected(input.finish_stream().expect("stop")), utterance_id);
        assert!(input.active.is_none());
    }

    /// 20 ms PCM16 frames of 16 kHz audio through the session input, counting
    /// finished turns and confirmed barge-ins.
    fn stream_turns(input: &mut StreamingInputState, samples: &[f32]) -> (usize, usize) {
        let (mut turns, mut barge_ins) = (0, 0);
        for (index, chunk) in samples.chunks(320).enumerate() {
            let pcm: Vec<i16> = chunk
                .iter()
                .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
                .collect();
            let result = input
                .handle_pcm16_frame(index as u32 + 1, 16_000, &pcm16_frame(&pcm))
                .expect("frame");
            barge_ins += usize::from(result.barge_in);
            turns += usize::from(finalized(result.utterance_end).is_some());
        }
        turns += usize::from(finalized(input.finish_stream().expect("stop")).is_some());
        (turns, barge_ins)
    }

    /// Steady background noise, which Earshot alone scores as speech, never
    /// becomes a turn or a barge-in; speech spoken over it becomes one turn.
    #[test]
    fn steady_noise_never_becomes_a_turn_but_speech_over_it_does() {
        use izwi_vad::test_signals::{fan_noise, mains_hum, pink_noise};

        let fox = {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
            let mut reader = hound::WavReader::open(path).expect("speech fixture");
            let rate = reader.spec().sample_rate;
            let samples: Vec<f32> = reader
                .samples::<i16>()
                .map(|sample| sample.expect("sample") as f32 / 32768.0)
                .collect();
            izwi_vad::resample_linear_f32(&samples, rate, 16_000).expect("resample")
        };
        let rms = |samples: &[f32]| {
            (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
                .sqrt()
        };
        let len = 16_000 * 10;
        for (name, noise) in [
            ("fan", fan_noise(len, 0.3, 7)),
            ("hum", mains_hum(len, 0.3)),
            ("pink", pink_noise(len, 0.3, 7)),
        ] {
            let mut input = test_streaming_input();
            input.config.min_speech_ms = DEFAULT_STREAM_MIN_SPEECH_MS;
            input.config.silence_duration_ms = DEFAULT_STREAM_SILENCE_MS;
            input.endpoint = EndpointDetector::new(input.config.endpoint_config());
            assert_eq!(stream_turns(&mut input, &noise), (0, 0), "{name} alone");

            // Fox 3 s in, 10 dB over the noise.
            let gain = rms(&fox) / rms(&noise) * 10f32.powf(-10.0 / 20.0);
            let mut mixed: Vec<f32> = noise.iter().map(|sample| sample * gain).collect();
            for (idx, sample) in fox.iter().enumerate() {
                mixed[16_000 * 3 + idx] += sample;
            }
            let mut input = test_streaming_input();
            input.config.min_speech_ms = DEFAULT_STREAM_MIN_SPEECH_MS;
            input.config.silence_duration_ms = DEFAULT_STREAM_SILENCE_MS;
            input.endpoint = EndpointDetector::new(input.config.endpoint_config());
            let (turns, barge_ins) = stream_turns(&mut input, &mixed);
            assert_eq!((turns, barge_ins), (1, 1), "{name} with speech");
        }

        // A short reply ("the quick", about half a second) still clears the
        // per-utterance speech evidence.
        let mut short = vec![0.0f32; 16_000 / 2];
        short.extend_from_slice(&fox[..16_000 * 3 / 4]);
        short.extend(vec![0.0f32; 16_000 * 3 / 2]);
        let mut input = test_streaming_input();
        input.config.min_speech_ms = DEFAULT_STREAM_MIN_SPEECH_MS;
        input.config.silence_duration_ms = DEFAULT_STREAM_SILENCE_MS;
        input.endpoint = EndpointDetector::new(input.config.endpoint_config());
        assert_eq!(stream_turns(&mut input, &short).0, 1, "short reply");
    }

    /// SPK-F6: audio sent after the client finished playing what it had is
    /// an underrun; audio sent while playback still has queued audio is not.
    #[test]
    fn playback_clock_counts_drained_playback_as_an_underrun() {
        let start = Instant::now();
        let mut clock = PlaybackClock::default();
        let chunk = Duration::from_millis(200);
        assert!(!clock.sent(start, chunk), "the first chunk starts playback");
        assert!(!clock.sent(start + Duration::from_millis(150), chunk));
        // Queued until 400 ms; arriving within the tolerance is in time.
        assert!(!clock.sent(start + Duration::from_millis(410), chunk));
        // Queued until 610 ms; arriving at 700 ms found playback drained.
        assert!(clock.sent(start + Duration::from_millis(700), chunk));
        // Playback restarted at 700 ms, so 850 ms is in time again.
        assert!(!clock.sent(start + Duration::from_millis(850), chunk));
    }

    /// SPK-F6: the session reaches Speaking once its turn sends audio and
    /// returns to Listening when the turn ends.
    #[test]
    fn session_phase_follows_the_turn_through_speaking() {
        let mut session = VoiceSession::default();
        session.start("session");
        session.begin_listening();
        session.begin_processing("turn-1");

        follow_turn_phase(&mut session, Some(("turn-1", true, false)));
        assert_eq!(session.phase(), VoiceSessionPhase::Processing);
        follow_turn_phase(&mut session, Some(("turn-1", true, true)));
        assert_eq!(session.phase(), VoiceSessionPhase::Speaking);
        assert_eq!(session.active_turn_id(), Some("turn-1"));
        follow_turn_phase(&mut session, Some(("turn-1", false, true)));
        assert_eq!(session.phase(), VoiceSessionPhase::Listening);
        follow_turn_phase(&mut session, None);
        assert_eq!(session.phase(), VoiceSessionPhase::Listening);

        // An interrupted or closed session is not moved by its turn.
        session.begin_processing("turn-2");
        session.interrupt("barge_in");
        follow_turn_phase(&mut session, Some(("turn-2", true, true)));
        assert_eq!(session.phase(), VoiceSessionPhase::Interrupted);
        session.close();
        follow_turn_phase(&mut session, None);
        assert_eq!(session.phase(), VoiceSessionPhase::Closed);
    }

    /// SPK-F6: a turn records its response latency at its first audio only,
    /// and an underrun when it falls behind playback.
    #[tokio::test]
    async fn turn_audio_records_first_audio_once_and_underruns() {
        let runtime = RuntimeService::new(izwi_core::EngineConfig::default()).expect("runtime");
        let control = TurnControl::new(Instant::now() - Duration::from_millis(300));
        assert!(!control.is_speaking());
        control.audio_sent(&runtime, Duration::from_millis(10));
        assert!(control.is_speaking());
        tokio::time::sleep(Duration::from_millis(80)).await;
        control.audio_sent(&runtime, Duration::from_millis(10));

        let payload = runtime.telemetry_prometheus().await;
        let counts = izwi_core::runtime::prometheus::check_histograms(&payload).unwrap();
        assert_eq!(counts["izwi_voice_turn_first_audio_seconds"], 1);
        assert!(payload.contains("izwi_voice_audio_egress_underruns_total 1"));
    }

    #[test]
    fn streaming_pre_roll_is_a_bounded_ring() {
        let mut input = test_streaming_input();
        input.push_pre_roll(&vec![1_i16; 8_000], 16_000);

        assert_eq!(input.pre_roll.len(), 1_600);
        assert!(input.pre_roll.iter().all(|sample| *sample == 1));
    }
}

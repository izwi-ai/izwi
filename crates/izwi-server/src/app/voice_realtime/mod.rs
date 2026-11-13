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
    strip_reasoning, AgentDefinition, AgentEngine, AgentEvent, AgentSession, AgentSet,
    AgentTurnOptions, BackgroundResults, HistoryBudget, MemoryMessage, MemoryMessageMeta,
    MemoryMessageRole, ReasoningFilter, TurnEvents, TurnInput,
};
use izwi_core::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, ChatMessage, ChatRole, GenerationConfig, GenerationParams,
    GenerationRequest, RuntimeRequestContext, RuntimeService, SessionModelLeases, VoiceLatency,
    VoiceSession, VoiceSessionPhase, WorkloadClass,
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

use crate::api::openai::tool_calls::{parse_answer, ToolCallFormat, ToolCallParser};
use crate::app::agent_runtime::{
    cancel_follows, cancel_with, drive_turn, finish_name, ChatAgentBackend, ConversationSummary,
    ThreadMemory, TurnTrace,
};
use crate::app::mcp::{McpAccess, McpServerConfig, McpTools};
use crate::app::realtime_codec::{AudioOutputEncoder, OutputAudioFormat};
use crate::app::realtime_protocol::{
    RealtimeAudioFormat, RealtimeAudioGapAction, RealtimeClose, RealtimeCloseCode,
    RealtimeCloseReason, RealtimeErrorCode, RealtimeEventEnvelope, RealtimeFunctionCallStatus,
    RealtimeFunctionExecutor, RealtimeInterruptionReason, RealtimeProtocol, RealtimeServerEvent,
    RealtimeSpeechEndReason, RealtimeTurnStatus, VOICE_REALTIME_VERSION,
};
use crate::app::realtime_transport::{
    now_unix_millis, parse_client_pcm16_frame, pcm16_bytes_to_i16, spawn_writer,
    wav_bytes_from_pcm16_mono, ClientPcm16Frame, OutboundClass, OutboundEnqueueError, OutboundItem,
    OutboundLimits, OutboundMailbox, OutboundMessage,
};
use crate::app::voice_reply::{SegmentJoiner, SpeechSegmenter, SpokenLedger};
use crate::app::voice_tools::{function_call_message, PendingFunctionCalls, VoiceTools};
use crate::ids::new_uuid;
use crate::state::{AppState, OwnedAdmissionPermit};
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_NAME, DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
};
use crate::voice_memory::extract_observation_candidates;
use crate::voice_store::CreateVoiceTurnRequest;
mod input;
mod speak;
mod turn;

pub(crate) use input::*;
pub(crate) use speak::*;
pub(crate) use turn::*;

const DEFAULT_CHAT_MODEL: &str = "Qwen3.5-4B";
/// A spoken reply's default token budget: a few sentences, not an essay
/// (SPK-F2). Clients may raise it with `max_output_tokens`.
pub(crate) const VOICE_DEFAULT_MAX_OUTPUT_TOKENS: usize = 384;
const VOICE_EMPTY_REPLY_ERROR: &str =
    "The model gave no answer to speak. Try again, or choose another chat model.";
/// The conversation history a voice turn's prompt carries (SPK-F13).
const VOICE_HISTORY_BUDGET: HistoryBudget = HistoryBudget {
    max_messages: 24,
    max_tokens: 3_000,
};
pub(crate) const MAX_UTTERANCE_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MAX_UTTERANCE_PCM16_SAMPLES: usize = (MAX_UTTERANCE_BYTES - 44) / 2;
const WS_BIN_MAGIC: &[u8; 4] = b"IVWS";
const WS_BIN_KIND_ASSISTANT_PCM16: u8 = 2;
const WS_BIN_KIND_ASSISTANT_G711_ULAW: u8 = 3;
const WS_BIN_KIND_ASSISTANT_G711_ALAW: u8 = 4;
const WS_BIN_ASSISTANT_HEADER_LEN: usize = 24;
pub(crate) const DEFAULT_STREAM_VAD_THRESHOLD: f32 = DEFAULT_SPEECH_THRESHOLD;
pub(crate) const DEFAULT_STREAM_MIN_SPEECH_MS: u32 = DEFAULT_MIN_SPEECH_MS;
pub(crate) const DEFAULT_STREAM_SILENCE_MS: u32 = DEFAULT_SILENCE_MS;
pub(crate) const DEFAULT_STREAM_MAX_UTTERANCE_MS: u32 = DEFAULT_MAX_UTTERANCE_MS;
pub(crate) const DEFAULT_STREAM_PRE_ROLL_MS: u32 = DEFAULT_PRE_ROLL_MS;
/// Voiced speech that confirms a barge-in. A speech start alone does not
/// interrupt the assistant: a cough, a door or speaker echo also starts one.
pub(crate) const DEFAULT_BARGE_IN_MIN_SPEECH_MS: u32 = 300;
/// How long past the silence window frames may stop before the server
/// assumes the client paused without `input_stream_stop`. Silence is measured
/// in audio time, so this only has to cover network stalls and jitter.
pub(crate) const CLIENT_PAUSE_STALL_ALLOWANCE_MS: u64 = 1_500;
pub(crate) const TURN_CANCELLATION_TIMEOUT: Duration = Duration::from_millis(750);
pub(crate) const SESSION_ACTOR_TICK: Duration = Duration::from_millis(100);
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
    /// The epoch events queued from now on belong to; a turn cutoff
    /// advances it.
    connection_epoch: u64,
    /// The epoch of the last event sent. Events are numbered when the writer
    /// sends them, so an event the mailbox drops never leaves a gap.
    sent_epoch: u64,
    next_event_id: u64,
    next_sequence: u64,
    last_sent_event_id: u64,
    last_sent_sequence: u64,
    next_revision: u64,
    last_audio_chunk_sequence: u64,
}

impl TypedVoiceSession {
    fn new(session_id: String, owner_instance_id: String) -> Self {
        Self {
            session_id,
            owner_instance_id,
            connection_epoch: 0,
            sent_epoch: 0,
            next_event_id: 1,
            next_sequence: 0,
            last_sent_event_id: 0,
            last_sent_sequence: 0,
            next_revision: 1,
            last_audio_chunk_sequence: 0,
        }
    }

    fn next_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }

    /// Number `event`, queued in `epoch`, as the next event on the wire.
    fn envelope(
        &mut self,
        mut event: RealtimeServerEvent,
        epoch: u64,
        utterance_id: Option<String>,
        turn_id: Option<String>,
    ) -> RealtimeEventEnvelope<RealtimeServerEvent> {
        if let RealtimeServerEvent::Interruption {
            cutoff_event_id,
            cutoff_sequence,
            ..
        } = &mut event
        {
            *cutoff_event_id = self.last_sent_event_id;
            *cutoff_sequence = self.last_sent_sequence;
        }
        if epoch > self.sent_epoch {
            self.sent_epoch = epoch;
            self.next_sequence = 0;
        }
        let envelope = RealtimeEventEnvelope {
            protocol: RealtimeProtocol::VoiceRealtime,
            version: VOICE_REALTIME_VERSION,
            event_id: self.next_event_id,
            sequence: self.next_sequence,
            session_id: self.session_id.clone(),
            connection_epoch: self.sent_epoch,
            timestamp_ms: now_unix_millis(),
            utterance_id,
            turn_id,
            segment_id: None,
            event,
        };
        self.last_sent_event_id = self.next_event_id;
        self.last_sent_sequence = self.next_sequence;
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
pub(crate) struct OutboundTx {
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

    fn send(&self, message: impl Into<OutboundMessage>) -> bool {
        self.enqueue(message, OutboundClass::Critical, None).is_ok()
    }

    fn send_turn(&self, generation: u64, message: impl Into<OutboundMessage>) -> bool {
        self.enqueue(message, OutboundClass::Critical, Some(generation))
            .is_ok()
    }

    fn send_text_snapshot(
        &self,
        generation: u64,
        key: String,
        message: impl Into<OutboundMessage>,
    ) -> bool {
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
                message: message.into(),
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

    fn send_diagnostic(&self, message: impl Into<OutboundMessage>) -> bool {
        self.enqueue(message, OutboundClass::Droppable, None)
            .is_ok()
    }

    fn enqueue(
        &self,
        message: impl Into<OutboundMessage>,
        class: OutboundClass,
        turn_generation: Option<u64>,
    ) -> Result<(), OutboundEnqueueError> {
        let result = self.mailbox.enqueue(OutboundItem {
            message: message.into(),
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
        }
    }

    /// The wire message for one event: legacy JSON as is, or for the typed
    /// protocol the mapped event, numbered when the writer sends it. `None`
    /// when the typed protocol has no equivalent.
    fn encode_json_event(
        &self,
        value: &serde_json::Value,
    ) -> Result<Option<OutboundMessage>, serde_json::Error> {
        let mut wire = self
            .wire
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(typed) = wire.typed.as_mut() else {
            return serde_json::to_string(value)
                .map(|text| Some(Message::Text(text.into()).into()));
        };
        let Some((event, utterance_id, turn_id)) = map_typed_voice_event(value, typed) else {
            return Ok(None);
        };
        let epoch = typed.connection_epoch;
        let wire = Arc::clone(&self.wire);
        Ok(Some(OutboundMessage::Deferred(Box::new(move || {
            let mut wire = wire.lock().unwrap_or_else(|poison| poison.into_inner());
            let typed = wire.typed.as_mut()?;
            let envelope = typed.envelope(event, epoch, utterance_id, turn_id);
            match serde_json::to_string(&envelope) {
                Ok(text) => Some(Message::Text(text.into())),
                Err(err) => {
                    warn!("failed to serialize typed voice ws event: {err}");
                    None
                }
            }
        }))))
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
    InputStreamStart(Box<InputStreamStartEvent>),
    InputStreamStop,
    /// The output of a call the server sent as `function_call`.
    FunctionCallOutput {
        call_id: String,
        output: serde_json::Value,
    },
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

/// `input_stream_start`: how the session listens and answers.
#[derive(Debug, Deserialize)]
struct InputStreamStartEvent {
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
    /// Let the silence that ends an utterance follow the speaker's own
    /// pauses, between `adaptive_silence_floor_ms` (default 450) and
    /// `silence_duration_ms`. Off by default.
    #[serde(default)]
    adaptive_silence: Option<bool>,
    #[serde(default)]
    adaptive_silence_floor_ms: Option<u32>,
    /// Modular mode: start transcribing an utterance at a short pause (a
    /// third of the end-of-utterance silence, 200 to 400 ms), so a pause
    /// that does end it finds the transcript already under way. A pause
    /// the speaker talks through wastes that transcription. Off by
    /// default.
    #[serde(default)]
    early_transcription: Option<bool>,
    /// Modular mode: also start writing the reply at that pause (this
    /// turns `early_transcription` on). The reply stores nothing and runs
    /// no tool until the pause has ended the utterance; if the speaker
    /// goes on, it is dropped. Off by default.
    #[serde(default)]
    preemptive_generation: Option<bool>,
    /// Pause the reply when the user seems to interrupt it, and play it on
    /// if the sound turns out to be no speech (noise, a cough) or, in
    /// modular mode, speech with no words. Off, any sustained speech
    /// cancels the reply at once. The client must honour `turn_paused` and
    /// `turn_resumed`.
    #[serde(default)]
    resume_false_interruptions: Option<bool>,
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
    /// Function tools a turn may call; the client answers their calls
    /// with `function_call_output`. They need a chat model with a
    /// tool-calling template, or LFM3-Audio in unified mode.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// Whether those turns may call the server's built-in tools
    /// (default `false`).
    #[serde(default)]
    builtin_tools: Option<bool>,
    /// How long a turn waits for the client's output of a call.
    #[serde(default)]
    function_call_timeout_ms: Option<u64>,
    /// What a modular reply says when a tool has kept it waiting and it
    /// has said nothing yet. Default "One moment." for English sessions and
    /// nothing otherwise; an empty string turns it off.
    #[serde(default)]
    tool_wait_phrase: Option<String>,
    /// MCP servers whose tools a turn may call (`label`, `url`, optional
    /// `headers` and `allowed_tools`). The server connects only to hosts in
    /// `IZWI_MCP_ALLOWED_HOSTS`.
    #[serde(default)]
    mcp_servers: Option<Vec<McpServerConfig>>,
    /// Modular mode: the agents of the session. The first holds the
    /// conversation at the start; an agent hands it to another by calling
    /// `transfer_to_<id>`, and records its task by calling
    /// `complete_<task name>`. Without `agents` the session is one agent
    /// with the session's system prompt.
    #[serde(default)]
    agents: Option<Vec<VoiceAgentConfig>>,
}

/// One agent of a voice session.
#[derive(Debug, Clone, Deserialize)]
struct VoiceAgentConfig {
    id: String,
    #[serde(default)]
    name: Option<String>,
    /// The agent's system prompt; the session's when absent.
    #[serde(default)]
    instructions: Option<String>,
    /// The session tools the agent may call; all of them when absent.
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    handoffs: Vec<String>,
    #[serde(default)]
    handoff_description: Option<String>,
    #[serde(default)]
    task: Option<izwi_agent::TaskSpec>,
}

/// The agents of a session as the runtime's, each on `model_id`. An agent
/// without instructions keeps an empty prompt and takes the session's at
/// its turn.
fn voice_agent_set(
    agents: Option<Vec<VoiceAgentConfig>>,
    model_id: &str,
    tools: Option<&VoiceTools>,
) -> Result<Option<AgentSet>, String> {
    let Some(agents) = agents.filter(|agents| !agents.is_empty()) else {
        return Ok(None);
    };
    let offered = tools
        .map(|tools| {
            tools
                .registry(&PendingFunctionCalls::default())
                .tool_names()
        })
        .unwrap_or_default();
    let agents = agents
        .into_iter()
        .map(|agent| {
            for tool in agent.tools.iter().flatten() {
                if !offered.contains(tool) {
                    return Err(format!(
                        "Agent `{}` names the tool `{tool}`, which the session does not offer",
                        agent.id
                    ));
                }
            }
            Ok(AgentDefinition {
                name: agent.name.unwrap_or_else(|| agent.id.clone()),
                tools: agent.tools,
                handoffs: agent.handoffs,
                handoff_description: agent.handoff_description,
                task: agent.task,
                ..AgentDefinition::new(
                    agent.id,
                    "",
                    agent.instructions.unwrap_or_default(),
                    model_id,
                )
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    AgentSet::new(agents).map(Some).map_err(|err| match err {
        izwi_agent::AgentError::InvalidAgent(message) => {
            format!("Invalid `agents`: {message}")
        }
        other => other.to_string(),
    })
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
pub(crate) struct ModularVoiceTurnConfig {
    asr_model_id: String,
    text_model_id: String,
    tts_model_id: String,
    speaker: Option<String>,
    asr_language: Option<String>,
    max_output_tokens: usize,
    output_format: OutputAudioFormat,
    /// The tools its turns may call, when the chat model can call tools.
    tools: Option<VoiceTools>,
    /// Said when a tool keeps a silent reply waiting.
    tool_wait_phrase: Option<String>,
    /// The session's agents, when it has more than the default one.
    agents: Option<AgentSet>,
}

#[derive(Debug, Clone)]
pub(crate) struct UnifiedVoiceTurnConfig {
    s2s_model_id: String,
    speaker: Option<String>,
    max_output_tokens: usize,
    output_format: OutputAudioFormat,
    /// The tools its turns may call (LFM3-Audio).
    tools: Option<VoiceTools>,
}

#[derive(Debug, Clone)]
pub(crate) enum VoiceTurnConfig {
    Modular(ModularVoiceTurnConfig),
    Unified(UnifiedVoiceTurnConfig),
}

#[derive(Debug, Clone)]
pub(crate) struct PendingAudioCommit {
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

pub(crate) struct AbortOnDropTask<T> {
    handle: Option<tokio::task::JoinHandle<T>>,
}

impl<T> AbortOnDropTask<T> {
    pub(crate) fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    pub(crate) async fn join(mut self) -> Result<T, tokio::task::JoinError> {
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
    /// How the input stream's turns are answered.
    turn_config: Option<VoiceTurnConfig>,
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
    /// The client's tool calls the session's turns await.
    function_calls: PendingFunctionCalls,
    /// The results of background tools, for the next turn to tell.
    background_results: BackgroundResults,
    /// A transcript started at a pause, for the turn that pause may begin.
    early_transcript: Option<EarlyTranscript>,
    /// Whether a reply is started at the pause that may end an utterance.
    preemptive_generation: bool,
    /// Whether speech over a reply pauses it until it is known to be an
    /// interruption.
    resume_false_interruptions: bool,
    /// The reply paused that way, if any.
    held_reply: HeldReply,
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
            turn_config: None,
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
            function_calls: PendingFunctionCalls::default(),
            background_results: BackgroundResults::default(),
            early_transcript: None,
            preemptive_generation: false,
            resume_false_interruptions: false,
            held_reply: HeldReply::default(),
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
        self.held_reply.release_finished();
    }

    fn has_running_turn(&self) -> bool {
        self.active_turn
            .as_ref()
            .map(|turn| !turn.task.is_finished())
            .unwrap_or(false)
            || self.held_reply.is_running()
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
        UtteranceOutcome::Finalized(ended, end_reason) => {
            let turn_config = conn
                .turn_config
                .clone()
                .ok_or_else(|| "Received an utterance before `input_stream_start`.".to_string())?;
            let wav_bytes = ended.wav_bytes;
            let commit = PendingAudioCommit {
                utterance_id: ended.utterance_id,
                utterance_seq: ended.utterance_seq,
                turn_config,
                endpointed_at: ended.endpointed_at,
                endpointing_delay: ended.endpointing_delay,
            };
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
            // The sound that paused a reply was no speech: it plays on.
            conn.held_reply.resume(out_tx);
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
    // A modular turn that may resume the reply it spoke over decides once
    // it has its transcript: the reply stays paused until then. Any other
    // turn interrupts it now.
    let decides_later = conn.resume_false_interruptions
        && matches!(commit.turn_config, VoiceTurnConfig::Modular(_));
    let paused_reply = if decides_later {
        let running = conn
            .active_turn
            .take_if(|turn| !turn.task.is_finished() || turn.control.is_speaking());
        conn.held_reply.pause(out_tx, running, "barge_in");
        conn.held_reply
            .holds_turn()
            .then(|| conn.held_reply.clone())
    } else {
        if conn.held_reply.interrupt(out_tx, "barge_in") {
            state.runtime.record_voice_interruption();
            state.runtime.record_voice_barge_in();
        }
        None
    };
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
            function_calls: conn.function_calls.clone(),
            background_results: conn.background_results.clone(),
            early_transcript: conn
                .early_transcript
                .take()
                .filter(|early| early.utterance_seq == commit.utterance_seq)
                .map(|early| early.task),
            paused_reply,
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

    if conn.held_reply.interrupt(&out_tx, "socket_closed") {
        state.runtime.record_voice_interruption();
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
        ClientEvent::InputStreamStart(start) => {
            let InputStreamStartEvent {
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
                adaptive_silence,
                adaptive_silence_floor_ms,
                early_transcription,
                preemptive_generation,
                resume_false_interruptions,
                barge_in_min_speech_ms,
                vad_snr_margin_db,
                input_sample_rate,
                output_audio_format,
                tools,
                builtin_tools,
                function_call_timeout_ms,
                tool_wait_phrase,
                mcp_servers,
                agents,
            } = *start;
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

            let normalized_asr_language_for_phrase = normalized_asr_language.clone();
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
                    if !asr_variant.serves_live_asr() {
                        return Err(format!(
                            "Model `{asr_model_id}` is not a supported modular ASR model."
                        ));
                    }
                    let text_variant = parse_chat_model_variant(Some(text_model_id.as_str()))
                        .map_err(|err| format!("Invalid chat model: {err}"))?;
                    validate_realtime_text_model(text_variant)?;
                    let tts_variant = parse_tts_model_variant(tts_model_id.as_str())
                        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
                    validate_realtime_tts_model(tts_variant)?;
                    let tools = VoiceTools::parse(
                        tools,
                        builtin_tools,
                        function_call_timeout_ms,
                        ToolCallFormat::of(text_variant).is_some(),
                    )?;
                    let tools = connect_mcp_tools(
                        tools,
                        mcp_servers,
                        ToolCallFormat::of(text_variant).is_some(),
                    )
                    .await?;

                    if agents.is_some() && ToolCallFormat::of(text_variant).is_none() {
                        return Err("`agents` need a chat model that can call tools".to_string());
                    }
                    let agents = voice_agent_set(agents, &text_model_id, tools.as_ref())?;

                    VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
                        asr_model_id,
                        text_model_id,
                        tts_model_id,
                        speaker: normalized_speaker.clone(),
                        asr_language: normalized_asr_language.clone(),
                        max_output_tokens: max_output_tokens
                            .unwrap_or(VOICE_DEFAULT_MAX_OUTPUT_TOKENS)
                            .clamp(1, 4096),
                        output_format,
                        tools,
                        tool_wait_phrase: resolve_tool_wait_phrase(
                            tool_wait_phrase,
                            normalized_asr_language_for_phrase.as_deref(),
                        ),
                        agents,
                    })
                }
                RealtimeVoiceMode::Unified => {
                    if agents.is_some() {
                        return Err("`agents` need modular mode".to_string());
                    }
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
                    let tools = VoiceTools::parse(
                        tools,
                        builtin_tools,
                        function_call_timeout_ms,
                        variant.family() == izwi_core::ModelFamily::Lfm3Audio,
                    )?;
                    let tools = connect_mcp_tools(
                        tools,
                        mcp_servers,
                        variant.family() == izwi_core::ModelFamily::Lfm3Audio,
                    )
                    .await?;
                    VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
                        s2s_model_id,
                        speaker: normalized_speaker,
                        max_output_tokens: max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                        output_format,
                        tools,
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
            conn.turn_config = Some(turn_config);
            conn.streaming_input = Some(StreamingInputState::new(StreamingInputConfig {
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
                adaptive_silence_floor_ms: adaptive_silence
                    .unwrap_or(false)
                    .then(|| adaptive_silence_floor_ms.unwrap_or(450).clamp(200, 5_000)),
                early_transcription: (early_transcription.unwrap_or(false)
                    || preemptive_generation.unwrap_or(false))
                    && mode == RealtimeVoiceMode::Modular,
            }));
            conn.voice_session.begin_listening();
            conn.resume_false_interruptions = resume_false_interruptions.unwrap_or(false);
            conn.preemptive_generation =
                preemptive_generation.unwrap_or(false) && mode == RealtimeVoiceMode::Modular;

            let mut ready = json!({
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
            });
            if conn.resume_false_interruptions {
                ready["resume_false_interruptions"] = json!(true);
            }
            if let Some(floor_ms) = conn
                .streaming_input
                .as_ref()
                .and_then(|input| input.config.adaptive_silence_floor_ms)
            {
                ready["vad"]["adaptive_silence_floor_ms"] = json!(floor_ms);
            }
            send_json(out_tx, ready);
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
            if conn.held_reply.interrupt(out_tx, &reason) {
                state.runtime.record_voice_interruption();
            }
            if interrupt_active_turn(out_tx, &mut conn.active_turn, &reason) {
                state.runtime.record_voice_interruption();
                conn.voice_session.interrupt(reason);
            }
        }
        ClientEvent::FunctionCallOutput { call_id, output } => {
            if !conn.function_calls.resolve(&call_id, &output) {
                return Err(format!(
                    "No tool call `{call_id}` awaits an output (unknown, settled or timed out)"
                ));
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
                .or_else(|| conn.held_reply.control_of(utterance_seq))
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

    // With `resume_false_interruptions` sustained speech only pauses the
    // reply: whether it was an interruption is known when the utterance
    // ends (and, in modular mode, has been transcribed).
    if frame_result.barge_in && conn.resume_false_interruptions {
        let running = conn
            .active_turn
            .take_if(|turn| !turn.task.is_finished() || turn.control.is_speaking());
        if conn.held_reply.pause(out_tx, running, "barge_in") {
            if let Some(started) = conn.speech_started_at {
                state
                    .runtime
                    .record_voice_latency(VoiceLatency::BargeIn, started.elapsed());
            }
        }
    } else if frame_result.barge_in {
        // A finished reply may still be playing on the client.
        conn.conversation.interrupt_playback();
    }
    if frame_result.barge_in
        && !conn.resume_false_interruptions
        && interrupt_active_turn(out_tx, &mut conn.active_turn, "barge_in")
    {
        state.runtime.record_voice_interruption();
        state.runtime.record_voice_barge_in();
        if let Some(started) = conn.speech_started_at {
            state
                .runtime
                .record_voice_latency(VoiceLatency::BargeIn, started.elapsed());
        }
        conn.voice_session.interrupt("barge_in");
    }

    if frame_result.speech_resumed {
        conn.early_transcript = None;
    }
    if let Some(early) = frame_result.early {
        // A reply is started ahead only when the conversation is settled:
        // no earlier reply still being written, or held for a decision.
        let reply_plan = match (conn.turn_config.as_ref(), conn.voice_session_id.as_ref()) {
            (Some(VoiceTurnConfig::Modular(config)), Some(voice_session_id))
                if conn.preemptive_generation
                    && !conn.has_running_turn()
                    && !conn.held_reply.holds_turn() =>
            {
                Some(AgentReplyPlan {
                    state: state.clone(),
                    correlation_id: correlation_id.to_string(),
                    voice_session_id: voice_session_id.clone(),
                    voice_profile_id: conn.voice_profile_id.clone(),
                    system_prompt: conn.system_prompt.clone(),
                    agent_session: conn.agent_session.clone(),
                    conversation: conn.conversation.clone(),
                    model_leases: conn.model_leases.clone(),
                    function_calls: conn.function_calls.clone(),
                    background_results: conn.background_results.clone(),
                    config: config.clone(),
                })
            }
            _ => None,
        };
        if let Some(VoiceTurnConfig::Modular(config)) = conn.turn_config.as_ref() {
            conn.early_transcript = Some(EarlyTranscript {
                utterance_seq: early.utterance_seq,
                task: spawn_early_transcript(
                    state,
                    correlation_id,
                    &conn.model_leases,
                    config,
                    early.wav_bytes,
                    reply_plan,
                ),
            });
        }
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
    function_calls: PendingFunctionCalls,
    /// The results of background tools, for the next turn to tell.
    background_results: BackgroundResults,
    early_transcript: Option<EarlyTranscriptTask>,
    /// The reply this turn's speech paused; the turn resumes or interrupts
    /// it once it knows whether the speech had words.
    paused_reply: Option<HeldReply>,
}

/// What was done ahead of a turn: the utterance's transcript, and with
/// preemptive generation the reply to it, unconfirmed.
struct EarlyTurnWork {
    transcript: izwi_core::AsrTranscription,
    reply: Option<AgentReply>,
}

type EarlyTranscriptTask = AbortOnDropTask<Result<EarlyTurnWork, String>>;

/// The transcript of an utterance, started at a pause in it. Dropping it
/// stops the work.
struct EarlyTranscript {
    utterance_seq: u64,
    task: EarlyTranscriptTask,
}

/// Transcribe `wav_bytes` under its own admission permit, beside the
/// session, and with a `reply_plan` start the reply to what was said.
fn spawn_early_transcript(
    state: &AppState,
    correlation_id: &str,
    model_leases: &SessionModelLeases,
    config: &ModularVoiceTurnConfig,
    wav_bytes: Vec<u8>,
    reply_plan: Option<AgentReplyPlan>,
) -> EarlyTranscriptTask {
    let (state, model_leases) = (state.clone(), model_leases.clone());
    let correlation_id = correlation_id.to_string();
    let (asr_model_id, asr_language) = (config.asr_model_id.clone(), config.asr_language.clone());
    AbortOnDropTask::new(tokio::spawn(async move {
        let transcript = with_stage_permit(&state, async |runtime_context| {
            model_leases
                .hold(
                    &state.runtime,
                    resolve_asr_model_variant(Some(&asr_model_id)),
                )
                .await
                .map_err(|err| model_load_error("ASR", err))?;
            state
                .runtime
                .asr_transcribe_streaming_bytes_with_runtime_context(
                    wav_bytes.as_slice(),
                    Some(&asr_model_id),
                    asr_language.as_deref(),
                    None,
                    izwi_core::AsrTaskOptions::default(),
                    None,
                    Some(&correlation_id),
                    runtime_context,
                    |_| {},
                )
                .await
                .map_err(|err| format!("ASR failed: {err}"))
        })
        .await?;
        let said = transcript.text.trim();
        let reply = match reply_plan {
            Some(plan) if !said.is_empty() => {
                // A reply that cannot start is started again by the turn.
                start_agent_reply(&plan, said, izwi_agent::TurnGate::closed())
                    .await
                    .ok()
            }
            _ => None,
        };
        Ok(EarlyTurnWork { transcript, reply })
    }))
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
            function_calls,
            background_results,
            early_transcript,
            paused_reply,
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

                    // A transcript started at the pause that ended the
                    // utterance is the utterance's: only silence followed.
                    let early = match early_transcript {
                        Some(task) => task.join().await.ok().and_then(Result::ok),
                        None => None,
                    };
                    let (early, early_reply) = match early {
                        Some(work) => (Some(work.transcript), work.reply),
                        None => (None, None),
                    };
                    let transcript = match early {
                        Some(transcript) => {
                            state.runtime.record_voice_latency(
                                VoiceLatency::AsrFirstPartial,
                                control.endpointed_at().elapsed(),
                            );
                            transcript
                        }
                        None => {
                            with_stage_permit(&state, async |runtime_context| {
                                model_leases
                                    .hold(
                                        &state.runtime,
                                        resolve_asr_model_variant(Some(&config.asr_model_id)),
                                    )
                                    .await
                                    .map_err(|err| model_load_error("ASR", err))?;
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
                                        izwi_core::AsrTaskOptions::default(),
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
                            .await?
                        }
                    };
                    let transcript_final_at = Instant::now();
                    state.runtime.record_voice_latency(
                        VoiceLatency::AsrFinal,
                        transcript_final_at.saturating_duration_since(control.endpointed_at()),
                    );

                    let user_text = transcript.text.trim().to_string();
                    // Speech with words interrupts the reply it paused;
                    // a sound with none lets it play on.
                    let mut resumed_reply = false;
                    if let Some(paused) = paused_reply.as_ref() {
                        if is_wordless(&user_text) {
                            resumed_reply = paused.resume(&out_tx);
                        } else if paused.interrupt(&out_tx, "barge_in") {
                            state.runtime.record_voice_interruption();
                            state.runtime.record_voice_barge_in();
                        }
                    }
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

                    // A sound that let the paused reply play on is not
                    // answered.
                    if user_text.is_empty() || resumed_reply {
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

                    let plan = AgentReplyPlan {
                        state: state.clone(),
                        correlation_id: correlation_id.clone(),
                        voice_session_id: voice_session_id.clone(),
                        voice_profile_id: voice_profile_id.clone(),
                        system_prompt: system_prompt.clone(),
                        agent_session: agent_session.clone(),
                        conversation: conversation.clone(),
                        model_leases: model_leases.clone(),
                        function_calls: function_calls.clone(),
                        background_results: background_results.clone(),
                        config: config.clone(),
                    };

                    // The reply streams into speech: its sentences are
                    // synthesized while the text model is still writing.
                    let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
                    let reply = async {
                        // A reply started at the pause that ended the
                        // utterance answers the same words.
                        let started = match early_reply.filter(|early| early.input == user_text) {
                            Some(early) => early,
                            None => {
                                start_agent_reply(&plan, &user_text, izwi_agent::TurnGate::opened())
                                    .await?
                            }
                        };
                        let raw_text = finish_agent_reply(
                            &plan,
                            started,
                            &control,
                            &turn_record_id,
                            &|event: &AgentEvent| {
                                if let Some(message) = function_call_message(
                                    event,
                                    &commit.utterance_id,
                                    commit.utterance_seq,
                                ) {
                                    send_turn_json(&out_tx, output_generation, message);
                                }
                            },
                            deltas,
                        )
                        .await?;
                        let assistant_text = strip_reasoning(&raw_text);
                        // Silence is not an answer: the turn fails and says so.
                        if assistant_text.is_empty() {
                            return Err(VOICE_EMPTY_REPLY_ERROR.to_string());
                        }
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
                    let mut sink = VoiceSocketReplySink::new(&out_tx, &commit, output_generation);
                    let speech = speak_reply(
                        &state,
                        &mut sink,
                        &control,
                        delta_rx,
                        transcript_final_at,
                        |text| {
                            SegmentSynthesis::start(
                                &state,
                                &correlation_id,
                                tts_variant,
                                TtsVoice::speaker(config.speaker.clone()),
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
                    let tools = config.tools.as_ref().map(|tools| VoiceTurnTools {
                        tools,
                        pending: &function_calls,
                        background: &background_results,
                        voice_session_id: &voice_session_id,
                    });
                    let instructions = unified_turn_instructions(
                        &system_prompt,
                        &izwi_agent::current_time_instruction(),
                    );
                    // What the user said is shown and stored as soon as it
                    // is transcribed, usually while the reply still streams,
                    // so it reads before the reply and an interrupted turn
                    // keeps it.
                    let record_transcript = |text: String| {
                        record_unified_transcript(
                            &state,
                            &out_tx,
                            output_generation,
                            &commit,
                            &turn_record_id,
                            text,
                        )
                    };
                    let (heard_tx, heard_rx) = tokio::sync::oneshot::channel();
                    let heard = async {
                        match heard_rx.await {
                            Ok(text) => record_transcript(text).await.map(Some),
                            Err(_) => Ok(None),
                        }
                    };
                    let (generation, heard) = tokio::join!(
                        stream_unified_s2s_to_socket(
                            &state,
                            &out_tx,
                            &correlation_id,
                            &commit.utterance_id,
                            commit.utterance_seq,
                            &config.s2s_model_id,
                            config.speaker.clone(),
                            &instructions,
                            history_messages,
                            config.max_output_tokens,
                            config.output_format,
                            audio_bytes.as_slice(),
                            tools,
                            heard_tx,
                            &control,
                            output_generation,
                            &model_leases,
                        ),
                        heard,
                    );
                    let generation = generation?;
                    let user_text = match heard? {
                        Some(text) => text,
                        // A model that reports the transcript with its reply.
                        None => {
                            record_transcript(
                                generation.input_transcription.clone().unwrap_or_default(),
                            )
                            .await?
                        }
                    };

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
pub(crate) async fn acquire_stage_permit(state: &AppState) -> Result<OwnedAdmissionPermit, String> {
    state
        .acquire_owned_workload_permit(WorkloadClass::Realtime)
        .await
        .map_err(|_| "Server is shutting down".to_string())
}

/// Run one turn stage under its own admission permit, released when the
/// stage ends.
pub(crate) async fn with_stage_permit<T>(
    state: &AppState,
    stage: impl AsyncFnOnce(RuntimeRequestContext) -> Result<T, String>,
) -> Result<T, String> {
    let permit = acquire_stage_permit(state).await?;
    stage(permit.runtime_context()).await
}

/// Realtime voice speaks with a built-in voice and streams its audio: a
/// reference-only model (Qwen Base, VibeVoice, Fish) has no voice to speak
/// with here, and one that does not stream would delay the first audio to
/// the end (TPIPE-21). The admin `voice_realtime_modular_tts` flag uses the
/// same check.
/// The voice chat model must answer without reasoning: a model that always
/// reasons (LFM2.5 Thinking) spends the reply on thoughts no one hears.
pub(crate) fn validate_realtime_text_model(variant: izwi_core::ModelVariant) -> Result<(), String> {
    let always_reasons = variant
        .chat_capabilities()
        .is_some_and(|caps| caps.supports_thinking && !caps.thinking_toggleable);
    if always_reasons {
        return Err(format!(
            "`{}` always thinks before it answers, so it can't hold a voice conversation. Choose another chat model.",
            variant.dir_name()
        ));
    }
    Ok(())
}

pub(crate) fn validate_realtime_tts_model(variant: izwi_core::ModelVariant) -> Result<(), String> {
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

/// Show and store what the user said in a unified turn; returns it
/// trimmed.
async fn record_unified_transcript(
    state: &AppState,
    out_tx: &OutboundTx,
    output_generation: u64,
    commit: &PendingAudioCommit,
    turn_record_id: &str,
    text: String,
) -> Result<String, String> {
    let text = text.trim().to_string();
    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "user_transcript_final",
            "utterance_id": commit.utterance_id,
            "utterance_seq": commit.utterance_seq,
            "text": text.clone(),
            "language": null,
            "audio_duration_secs": null,
        }),
    );
    state
        .voice_store
        .update_turn_transcript(turn_record_id.to_string(), text.clone(), None, None)
        .await
        .map_err(|err| format!("Voice storage error: {err}"))?;
    Ok(text)
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
                    history.push(MemoryMessage::new(role, content));
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
                MemoryMessageRole::Tool => ChatRole::Tool,
            },
            content: message.content,
        })
        .collect())
}

/// What a spoken turn's model rounds share while they stream.
#[derive(Default)]
struct UnifiedReplyStream {
    encoder: Option<AudioOutputEncoder>,
    /// Added to a round's own chunk numbers, so they count on across rounds.
    chunk_offset: u32,
    next_chunk: u32,
    /// The reply's visible text so far: tool-call markup is left out.
    text: String,
    /// The round's tool-call parser, when the turn offers tools.
    parser: Option<ToolCallParser>,
}

/// The tools a spoken turn offers and how their calls reach the client.
struct VoiceTurnTools<'a> {
    tools: &'a VoiceTools,
    pending: &'a PendingFunctionCalls,
    /// Where background tools leave their results for the next turn.
    background: &'a BackgroundResults,
    voice_session_id: &'a str,
}

/// Connect to a session's MCP servers and add their tools to its tools.
async fn connect_mcp_tools(
    tools: Option<VoiceTools>,
    servers: Option<Vec<McpServerConfig>>,
    calls_tools: bool,
) -> Result<Option<VoiceTools>, String> {
    let servers = servers.unwrap_or_default();
    if servers.is_empty() {
        return Ok(tools);
    }
    if !calls_tools {
        return Err(
            "MCP servers need a chat model that can call tools, or unified mode with an LFM3-Audio `s2s_model_id`"
                .to_string(),
        );
    }
    let mcp = McpTools::connect(&servers, &McpAccess::from_env()).await?;
    VoiceTools::with_mcp(tools, mcp)
}

/// How long a tool may keep a reply silent before the reply says so.
const TOOL_WAIT_PHRASE_AFTER: Duration = Duration::from_millis(1_500);

/// What a reply says while a tool keeps it waiting: the session's phrase,
/// or "One moment." when the session speaks English (or names no language).
fn resolve_tool_wait_phrase(requested: Option<String>, language: Option<&str>) -> Option<String> {
    match requested {
        Some(phrase) => Some(phrase.trim().to_string()).filter(|phrase| !phrase.is_empty()),
        None => language
            .is_none_or(|language| language.to_ascii_lowercase().starts_with("en"))
            .then(|| "One moment.".to_string()),
    }
}

/// A unified turn's instructions: the session's, then the current local
/// date and time (`now`, see [`izwi_agent::current_time_instruction`]). The
/// model then knows them without a `time` tool, which LFM3-Audio calls on a
/// plain greeting. A unified reply reads its whole prompt every turn, so a
/// line that changes from turn to turn costs nothing.
fn unified_turn_instructions(system_prompt: &str, now: &str) -> String {
    match system_prompt.trim() {
        "" => now.to_string(),
        instructions => format!("{instructions}\n\n{now}"),
    }
}

/// A unified spoken turn: the model's reply streamed to the socket. With
/// tools, the agent runtime follows a round that calls them with one that
/// hears their results, within its round and call limits; each round holds
/// its own admission permit, so none is held while a tool runs.
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
    tools: Option<VoiceTurnTools<'_>>,
    heard: tokio::sync::oneshot::Sender<String>,
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
        .map_err(|err| model_load_error("unified audio", err))?;

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

    let registry = tools
        .as_ref()
        .map(|tools| tools.tools.registry(tools.pending))
        .unwrap_or_default();
    let rounds = UnifiedRounds {
        state,
        out_tx,
        correlation_id,
        utterance_id,
        utterance_seq,
        variant,
        speaker,
        system_prompt,
        max_output_tokens,
        output_format,
        audio_bytes,
        control,
        output_generation,
        definitions: registry.function_definitions(None),
        stream: Arc::new(Mutex::new(UnifiedReplyStream::default())),
        delivery_error: Arc::new(Mutex::new(None)),
        heard: Mutex::new(Some(heard)),
        progress: Mutex::new(UnifiedProgress {
            messages: history_messages,
            reply: izwi_core::SpeechToSpeechGeneration {
                text: String::new(),
                samples: Vec::new(),
                sample_rate: 24_000,
                input_transcription: None,
                generation_time_ms: 0.0,
            },
        }),
    };

    let agents = AgentSet::single(AgentDefinition::new("voice", "voice", "", s2s_model_id));
    let session = AgentSession {
        id: tools
            .as_ref()
            .map(|tools| tools.voice_session_id.to_string())
            .unwrap_or_default(),
        agent_id: "voice".to_string(),
        thread_id: utterance_id.to_string(),
        created_at: 0,
        updated_at: 0,
    };
    // The turn keeps its own conversation; nothing is read from the store.
    let memory = ThreadMemory::new(state.chat_store.clone(), None);
    let (cancel, _cancel_link) = cancel_with(control);
    let (events, heard) = tokio::sync::mpsc::unbounded_channel();
    let turn = izwi_agent::TurnContext {
        agents: &agents,
        memory: &memory,
        model: &rounds,
        tools: &registry,
        events: TurnEvents::new(events),
        cancel,
        background: tools
            .as_ref()
            .map(|tools| tools.background.clone())
            .unwrap_or_default(),
        confirmed: izwi_agent::TurnGate::opened(),
    };
    let defaults = AgentTurnOptions::default();
    let run = AgentEngine.run_conversation(
        &turn,
        &session,
        Vec::new(),
        vec![MemoryMessage::user(izwi_core::CHAT_AUDIO_PLACEHOLDER)],
        "",
        None,
        AgentTurnOptions {
            max_output_tokens,
            tool_timeout: tools
                .as_ref()
                .map_or(defaults.tool_timeout, |tools| tools.tools.timeout),
            persist_assistant_reply: false,
            ..defaults
        },
    );
    let mut trace = TurnTrace::start("voice_unified", correlation_id);
    let result = drive_turn(run, heard, |event| {
        trace.on_event(&event);
        if let Some(message) = function_call_message(&event, utterance_id, utterance_seq) {
            send_turn_json(out_tx, output_generation, message);
        }
    })
    .await;
    trace.finish(match &result {
        Ok(result) => finish_name(result.finish),
        Err(_) => "error",
    });
    let result = result.map_err(|err| match err {
        izwi_agent::AgentError::Model(message) | izwi_agent::AgentError::ToolLoop(message) => {
            message
        }
        other => other.to_string(),
    })?;
    if result.finish == izwi_agent::TurnFinish::Cancelled {
        return Err("Turn cancelled".to_string());
    }

    let mut reply = rounds
        .progress
        .into_inner()
        .unwrap_or_else(|poison| poison.into_inner())
        .reply;
    reply.text = lock_reply_stream(&rounds.stream).text.clone();

    send_turn_json(
        out_tx,
        output_generation,
        json!({
            "type": "assistant_audio_done",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
        }),
    );

    Ok(reply)
}

/// What a unified turn's rounds have built so far.
struct UnifiedProgress {
    /// The prompt's messages: the history, then each round's raw reply and
    /// the results of its calls.
    messages: Vec<ChatMessage>,
    reply: izwi_core::SpeechToSpeechGeneration,
}

/// The rounds of a unified spoken turn as the agent runtime's model: each
/// round hears the user's speech in the context so far and streams its
/// speech and text to the socket.
struct UnifiedRounds<'a> {
    state: &'a AppState,
    out_tx: &'a OutboundTx,
    correlation_id: &'a str,
    utterance_id: &'a str,
    utterance_seq: u64,
    variant: izwi_core::ModelVariant,
    speaker: Option<String>,
    system_prompt: &'a str,
    max_output_tokens: usize,
    output_format: OutputAudioFormat,
    audio_bytes: &'a [u8],
    control: &'a TurnControl,
    output_generation: u64,
    definitions: Vec<serde_json::Value>,
    stream: Arc<Mutex<UnifiedReplyStream>>,
    delivery_error: Arc<Mutex<Option<String>>>,
    /// Where the first round sends the turn's transcript once it has it.
    heard: Mutex<Option<tokio::sync::oneshot::Sender<String>>>,
    progress: Mutex<UnifiedProgress>,
}

impl UnifiedRounds<'_> {
    fn progress(&self) -> std::sync::MutexGuard<'_, UnifiedProgress> {
        self.progress
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn parser(&self) -> Option<ToolCallParser> {
        (!self.definitions.is_empty())
            .then(|| ToolCallParser::new(ToolCallFormat::Lfm3Xml).with_tools(&self.definitions))
    }
}

#[async_trait::async_trait]
impl izwi_agent::ModelBackend for UnifiedRounds<'_> {
    async fn generate(
        &self,
        request: izwi_agent::ModelRequest,
        _events: &TurnEvents,
    ) -> izwi_agent::Result<izwi_agent::ModelOutput> {
        let failed = izwi_agent::AgentError::Model;
        let round = request.round;
        // The results of the round before close the prompt.
        let messages = {
            let results = request
                .messages
                .iter()
                .rev()
                .take_while(|message| message.role == MemoryMessageRole::Tool)
                .collect::<Vec<_>>();
            let mut progress = self.progress();
            progress
                .messages
                .extend(results.into_iter().rev().map(|message| ChatMessage {
                    role: ChatRole::Tool,
                    content: message.content.clone(),
                }));
            progress.messages.clone()
        };
        {
            let mut stream = lock_reply_stream(&self.stream);
            stream.chunk_offset = stream.next_chunk;
            stream.parser = self.parser();
        }
        let params = GenerationParams {
            max_tokens: self.max_output_tokens.clamp(1, 4096),
            speaker: self.speaker.clone(),
            voice: self.speaker.clone(),
            ..Default::default()
        };
        let heard = self
            .heard
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        let permit = acquire_stage_permit(self.state).await.map_err(failed)?;
        let generation = self
            .state
            .runtime
            .speech_to_speech_round_streaming_bytes(
                self.variant,
                self.audio_bytes,
                messages,
                self.definitions.clone(),
                round == 0,
                params,
                Some(self.system_prompt),
                Some(self.correlation_id),
                permit.runtime_context(),
                move |text: &str| {
                    if let Some(heard) = heard {
                        // The turn may have ended already; nobody listens.
                        let _ = heard.send(text.to_string());
                    }
                },
                unified_chunk_sink(
                    self.out_tx,
                    self.utterance_id,
                    self.utterance_seq,
                    self.output_format,
                    self.control,
                    self.output_generation,
                    self.state.runtime.clone(),
                    self.stream.clone(),
                    self.delivery_error.clone(),
                ),
            )
            .await
            .map_err(|err| failed(format!("Unified speech-to-speech failed: {err}")))?;
        drop(permit);
        if let Some(err) = self
            .delivery_error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
        {
            return Err(failed(err));
        }

        // The parser holds back text that could still open a call.
        let tail = {
            let mut stream = lock_reply_stream(&self.stream);
            let tail = stream
                .parser
                .take()
                .map(|mut parser| parser.finish().content)
                .unwrap_or_default();
            stream.text.push_str(&tail);
            (!tail.is_empty()).then(|| stream.text.clone())
        };
        if let Some(text) = tail {
            send_turn_text_snapshot(
                self.out_tx,
                self.output_generation,
                format!("assistant-text-{}", self.utterance_seq),
                json!({
                    "type": "assistant_text_snapshot",
                    "utterance_id": self.utterance_id,
                    "utterance_seq": self.utterance_seq,
                    "text": text,
                }),
            );
        }

        let calls = match self.parser() {
            Some(parser) => parse_answer(parser, &generation.text).1,
            None => Vec::new(),
        };
        let mut progress = self.progress();
        if round == 0 {
            progress.reply.input_transcription = generation.input_transcription.clone();
        }
        if !generation.samples.is_empty() {
            progress.reply.sample_rate = generation.sample_rate;
            progress
                .reply
                .samples
                .extend_from_slice(&generation.samples);
        }
        progress.reply.generation_time_ms += generation.generation_time_ms;
        if !calls.is_empty() {
            if round == 0 {
                progress.messages.push(ChatMessage {
                    role: ChatRole::User,
                    content: izwi_core::CHAT_AUDIO_PLACEHOLDER.to_string(),
                });
            }
            progress.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: generation.text.clone(),
            });
        }
        Ok(izwi_agent::ModelOutput {
            tool_calls: calls
                .into_iter()
                .map(|call| izwi_agent::ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: call.function.arguments,
                })
                .collect(),
            generation_time_ms: generation.generation_time_ms,
            ..Default::default()
        })
    }
}

fn lock_reply_stream(
    stream: &Mutex<UnifiedReplyStream>,
) -> std::sync::MutexGuard<'_, UnifiedReplyStream> {
    stream.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// The sink of one round's stream: visible text as snapshots, speech as
/// binary frames held back for the client's playback credit (SPK-F11).
#[allow(clippy::too_many_arguments)]
fn unified_chunk_sink(
    out_tx: &OutboundTx,
    utterance_id: &str,
    utterance_seq: u64,
    output_format: OutputAudioFormat,
    control: &TurnControl,
    output_generation: u64,
    runtime: Arc<RuntimeService>,
    stream: Arc<Mutex<UnifiedReplyStream>>,
    delivery_error: Arc<Mutex<Option<String>>>,
) -> impl FnMut(
    izwi_core::StreamingOutput,
) -> futures::future::Either<
    std::future::Ready<izwi_core::Result<()>>,
    std::pin::Pin<Box<dyn std::future::Future<Output = izwi_core::Result<()>> + Send>>,
> + Send
       + 'static {
    let out_tx = out_tx.clone();
    let utterance_id = utterance_id.to_string();
    let control = control.clone();
    move |chunk| {
        // A cancelled turn stops the reply at its next token instead of
        // letting it run to its end (SPK-F5).
        let cancelled = || izwi_core::Error::Cancelled("unified voice turn cancelled".to_string());
        if control.is_cancelled() {
            return futures::future::Either::Left(std::future::ready(Err(cancelled())));
        }
        let mut frame = None;
        {
            let mut stream = lock_reply_stream(&stream);
            if let Some(delta) = chunk.text.as_deref().filter(|delta| !delta.is_empty()) {
                let visible = match stream.parser.as_mut() {
                    Some(parser) => parser.push(delta).content,
                    None => delta.to_string(),
                };
                if !visible.is_empty() {
                    stream.text.push_str(&visible);
                    send_turn_text_snapshot(
                        &out_tx,
                        output_generation,
                        format!("assistant-text-{utterance_seq}"),
                        json!({
                            "type": "assistant_text_snapshot",
                            "utterance_id": utterance_id,
                            "utterance_seq": utterance_seq,
                            "text": stream.text,
                        }),
                    );
                }
            }

            if !chunk.samples.is_empty() {
                let model_rate = chunk.sample_rate.max(1);
                let encoded = match stream.encoder.as_mut() {
                    Some(encoder) => encoder.encode(&chunk.samples),
                    None => AudioOutputEncoder::new(output_format, model_rate)
                        .and_then(|created| stream.encoder.insert(created).encode(&chunk.samples)),
                };
                match encoded {
                    Ok(encoded) => {
                        let chunk_seq = stream
                            .chunk_offset
                            .saturating_add(u32::try_from(chunk.sequence).unwrap_or(u32::MAX));
                        stream.next_chunk = chunk_seq.saturating_add(1);
                        // A round that calls tools is not the reply's last.
                        let is_final = chunk.is_final
                            && !stream
                                .parser
                                .as_ref()
                                .is_some_and(ToolCallParser::saw_tool_calls);
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
                                is_final,
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
        }
        let Some((frame, bytes, duration_ms)) = frame else {
            return futures::future::Either::Left(std::future::ready(Ok(())));
        };
        // Audio waits for the client's playback credit, as the cascaded
        // pipeline's does (SPK-F11): a reply generated faster than real time
        // is held back, not dropped.
        let out_tx = out_tx.clone();
        let control = control.clone();
        let delivery_error = delivery_error.clone();
        let runtime = runtime.clone();
        futures::future::Either::Right(Box::pin(async move {
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
        }))
    }
}

/// A reply that speech may have interrupted: paused on the client, still
/// generating here, until the speech proves to be an interruption (the
/// reply is cancelled) or not (it plays on). The connection and the turn
/// that transcribes the speech share it.
#[derive(Clone, Default)]
pub(crate) struct HeldReply {
    state: Arc<Mutex<HeldReplyState>>,
}

#[derive(Default)]
struct HeldReplyState {
    turn: Option<ActiveTurn>,
    paused: bool,
}

impl HeldReply {
    fn lock(&self) -> std::sync::MutexGuard<'_, HeldReplyState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Whether a reply is held, paused or playing on.
    fn holds_turn(&self) -> bool {
        self.lock().turn.is_some()
    }

    fn is_running(&self) -> bool {
        self.lock()
            .turn
            .as_ref()
            .is_some_and(|turn| !turn.task.is_finished())
    }

    fn control_of(&self, utterance_seq: u64) -> Option<TurnControl> {
        self.lock()
            .turn
            .as_ref()
            .filter(|turn| turn.utterance_seq == utterance_seq)
            .map(|turn| turn.control.clone())
    }

    /// Pause `turn`'s reply (or the one already held) and tell the client.
    fn pause(&self, out_tx: &OutboundTx, turn: Option<ActiveTurn>, reason: &str) -> bool {
        let mut state = self.lock();
        if let Some(turn) = turn {
            // A reply held before this one has had its say.
            if let Some(earlier) = state.turn.replace(turn) {
                drop(state);
                interrupt_active_turn(out_tx, &mut Some(earlier), "preempted_by_new_turn");
                state = self.lock();
            }
            state.paused = false;
        }
        let Some(turn) = state.turn.as_ref() else {
            return false;
        };
        if state.paused {
            return false;
        }
        turn.control.pause_playback();
        send_json(
            out_tx,
            json!({
                "type": "turn_paused",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
                "reason": reason,
            }),
        );
        state.paused = true;
        true
    }

    /// The speech was no interruption: the reply plays on. It stays held,
    /// so a later interruption still finds it.
    pub(crate) fn resume(&self, out_tx: &OutboundTx) -> bool {
        let mut state = self.lock();
        if !state.paused {
            return false;
        }
        state.paused = false;
        let Some(turn) = state.turn.as_ref() else {
            return false;
        };
        turn.control.resume_playback();
        send_json(
            out_tx,
            json!({
                "type": "turn_resumed",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
            }),
        );
        true
    }

    /// The speech was an interruption: cancel the held reply. Returns
    /// whether a running turn was interrupted.
    pub(crate) fn interrupt(&self, out_tx: &OutboundTx, reason: &str) -> bool {
        let turn = {
            let mut state = self.lock();
            state.paused = false;
            state.turn.take()
        };
        let mut turn = turn;
        interrupt_active_turn(out_tx, &mut turn, reason)
    }

    /// Give back a reply that has finished and is not paused: nothing more
    /// can interrupt its generation, and its playback is the client's.
    fn release_finished(&self) {
        let mut state = self.lock();
        if !state.paused
            && state
                .turn
                .as_ref()
                .is_some_and(|turn| turn.task.is_finished())
        {
            state.turn = None;
        }
    }
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
        // The reply was all sent and its `turn_done` too, but the client may
        // still be playing it: tell it to stop now instead of when the user
        // stops talking.
        send_json(
            out_tx,
            json!({
                "type": "turn_interrupted",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
                "reason": reason,
            }),
        );
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
    binding: tokio::sync::Mutex<Option<VoiceAgentBinding>>,
    /// The agent that holds the conversation, once a turn has run.
    active_agent: Mutex<Option<String>>,
    /// Notes on the part of the conversation the prompt no longer carries.
    summary: ConversationSummary,
}

/// A voice connection's agent session: its conversation thread, and the
/// system prompt and chat model it was created with.
#[derive(Debug, Clone)]
struct VoiceAgentBinding {
    session: AgentSession,
    system_prompt: String,
    model_id: String,
}

impl AgentSessionSlot {
    async fn ensure(
        &self,
        state: &AppState,
        voice_session_id: &str,
        system_prompt: &str,
        text_model_id: &str,
    ) -> Result<VoiceAgentBinding, String> {
        let mut binding = self.binding.lock().await;
        if let Some(bound) = binding.as_ref() {
            if bound.system_prompt == system_prompt {
                return Ok(bound.clone());
            }
        }

        let model_id = resolve_chat_model_id(Some(text_model_id))?;
        // The thread is the agent's memory for this voice session: not a
        // Chat thread, and deleted with the session.
        let thread = state
            .chat_store
            .create_voice_thread(voice_session_id.to_string(), Some(model_id.clone()))
            .await
            .map_err(|err| format!("Chat storage error: {err:#}"))?;

        let now = now_unix_millis();
        let bound = VoiceAgentBinding {
            session: AgentSession {
                id: new_uuid(),
                agent_id: DEFAULT_VOICE_AGENT_ID.to_string(),
                thread_id: thread.id,
                created_at: now,
                updated_at: now,
            },
            system_prompt: system_prompt.to_string(),
            model_id,
        };

        // A new thread starts the conversation over.
        self.summary.reset();
        *binding = Some(bound.clone());
        Ok(bound)
    }
}

/// What a modular reply is made from. Owned, so a reply can start beside
/// the session before its turn exists.
#[derive(Clone)]
struct AgentReplyPlan {
    state: AppState,
    correlation_id: String,
    voice_session_id: String,
    voice_profile_id: Option<String>,
    system_prompt: String,
    agent_session: Arc<AgentSessionSlot>,
    conversation: Arc<VoiceConversation>,
    model_leases: SessionModelLeases,
    function_calls: PendingFunctionCalls,
    background_results: BackgroundResults,
    config: ModularVoiceTurnConfig,
}

/// A reply being written by the agent. Until its turn confirms it
/// ([`finish_agent_reply`]) it has stored nothing and run no tool; dropping
/// it stops it.
struct AgentReply {
    /// What the user said, as the reply read it.
    input: String,
    session: AgentSession,
    model_id: String,
    events: tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
    task: tokio::task::JoinHandle<izwi_agent::Result<izwi_agent::AgentTurnResult>>,
    cancel: izwi_agent::TurnCancel,
    confirmed: izwi_agent::TurnGate,
}

impl Drop for AgentReply {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}

/// Start the agent's reply to `input`. Each model round takes its own
/// admission permit, so none is held while a tool runs. With `confirmed`
/// closed the reply is speculative: it writes its text, and waits at its
/// first tool call.
async fn start_agent_reply(
    plan: &AgentReplyPlan,
    input: &str,
    confirmed: izwi_agent::TurnGate,
) -> Result<AgentReply, String> {
    let state = &plan.state;
    let config = &plan.config;
    let binding = plan
        .agent_session
        .ensure(
            state,
            &plan.voice_session_id,
            &plan.system_prompt,
            &config.text_model_id,
        )
        .await?;
    let text_variant = parse_chat_model_variant(Some(&config.text_model_id))
        .map_err(|err| format!("Invalid chat model: {err}"))?;
    // The earlier replies join the history now, as far as they were heard.
    plan.conversation.store(state).await;
    plan.model_leases
        .hold(&state.runtime, text_variant)
        .await
        .map_err(|err| model_load_error("chat", err))?;
    let model_id = resolve_chat_model_id(Some(&config.text_model_id))?;

    // The session's prompt leads each agent's own instructions.
    let agents = match &config.agents {
        Some(agents) => agents.with_system_prompts(|agent| {
            [binding.system_prompt.trim(), agent.system_prompt.trim()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        }),
        None => AgentSet::single(AgentDefinition::new(
            binding.session.agent_id.clone(),
            DEFAULT_VOICE_AGENT_NAME,
            binding.system_prompt.clone(),
            binding.model_id.clone(),
        )),
    };
    let holder = config
        .agents
        .as_ref()
        .and_then(|_| {
            plan.agent_session
                .active_agent
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .clone()
        })
        .unwrap_or_else(|| agents.first().id.clone());
    let session = AgentSession {
        agent_id: holder,
        ..binding.session.clone()
    };

    let memory_context = if let Some(profile_id) = plan.voice_profile_id.clone() {
        match state
            .voice_store
            .get_profile(profile_id)
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

    let cancel = izwi_agent::TurnCancel::new();
    let (events, heard) = tokio::sync::mpsc::unbounded_channel();
    let task = {
        let plan = plan.clone();
        let (session, model_id) = (session.clone(), model_id.clone());
        let (cancel, confirmed) = (cancel.clone(), confirmed.clone());
        let input = input.to_string();
        tokio::spawn(async move {
            let config = &plan.config;
            let memory = ThreadMemory::new(plan.state.chat_store.clone(), memory_context)
                .with_summary(plan.agent_session.summary.clone());
            let backend = ChatAgentBackend::new(&plan.state, plan.correlation_id.clone());
            let tools = config
                .tools
                .as_ref()
                .map(|tools| tools.registry(&plan.function_calls))
                .unwrap_or_default();
            let turn = izwi_agent::TurnContext {
                agents: &agents,
                memory: &memory,
                model: &backend,
                tools: &tools,
                events: TurnEvents::new(events),
                cancel,
                background: plan.background_results.clone(),
                confirmed,
            };
            let defaults = AgentTurnOptions::default();
            AgentEngine
                .run_turn(
                    &turn,
                    &session,
                    TurnInput { text: input },
                    Some(model_id),
                    AgentTurnOptions {
                        max_output_tokens: config.max_output_tokens.clamp(1, 4096),
                        tool_timeout: config
                            .tools
                            .as_ref()
                            .map_or(defaults.tool_timeout, |tools| tools.timeout),
                        tool_soft_timeout: config
                            .tool_wait_phrase
                            .as_ref()
                            .map(|_| TOOL_WAIT_PHRASE_AFTER),
                        history: VOICE_HISTORY_BUDGET,
                        // The turn stores both once it is confirmed and,
                        // for the reply, heard.
                        persist_assistant_reply: false,
                        persist_user_message: false,
                        ..defaults
                    },
                )
                .await
        })
    };
    Ok(AgentReply {
        input: input.to_string(),
        session,
        model_id,
        events: heard,
        task,
        cancel,
        confirmed,
    })
}

/// Confirm `reply` as its turn's and see it through: the user's words are
/// stored, the reply's tools may run, and its text streams into `deltas`
/// (what it wrote before now first). The reply itself is not stored here:
/// the turn's record holds it until what the user heard of it is known
/// (SPK-F13). Returns the raw reply.
async fn finish_agent_reply(
    plan: &AgentReplyPlan,
    mut reply: AgentReply,
    control: &TurnControl,
    turn_record_id: &str,
    on_event: &(dyn Fn(&AgentEvent) + Sync),
    deltas: tokio::sync::mpsc::UnboundedSender<String>,
) -> Result<String, String> {
    let state = &plan.state;
    {
        let mut record = control.reply();
        record.session = Some(reply.session.clone());
        record.turn_record_id = Some(turn_record_id.to_string());
    }
    AgentEngine
        .commit_user_message(
            &ThreadMemory::new(state.chat_store.clone(), None),
            &reply.session,
            &reply.input,
        )
        .await
        .map_err(|err| err.to_string())?;
    let _cancel_link = cancel_follows(control, reply.cancel.clone());
    reply.confirmed.open();

    // A reply that has said nothing while a tool runs says it is waiting,
    // once.
    let tool_wait_phrase = plan.config.tool_wait_phrase.as_deref();
    let mut spoke = false;
    let mut trace = TurnTrace::start("voice_modular", &plan.correlation_id);
    let mut hear = |event: AgentEvent| {
        trace.on_event(&event);
        on_event(&event);
        match event {
            AgentEvent::TextDelta { text } => {
                spoke |= !text.trim().is_empty();
                let _ = deltas.send(text);
            }
            AgentEvent::ToolCallSlow { .. } if !spoke => {
                if let Some(phrase) = tool_wait_phrase {
                    spoke = true;
                    let _ = deltas.send(format!("{phrase} "));
                }
            }
            _ => {}
        }
    };
    let joined = loop {
        tokio::select! {
            biased;
            Some(event) = reply.events.recv() => hear(event),
            joined = &mut reply.task => break joined,
        }
    };
    while let Ok(event) = reply.events.try_recv() {
        hear(event);
    }
    let result = match joined {
        Ok(result) => result.map_err(|err| match err {
            izwi_agent::AgentError::InvalidInput(msg) => msg,
            other => other.to_string(),
        }),
        Err(_) => Err("Turn cancelled".to_string()),
    };
    trace.finish(match &result {
        Ok(result) => finish_name(result.finish),
        Err(_) => "error",
    });
    let result = result?;
    if plan.config.agents.is_some() {
        *plan
            .agent_session
            .active_agent
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(result.agent_id.clone());
    }
    {
        let mut record = control.reply();
        record.text = Some(strip_reasoning(&result.assistant_text));
        record.raw_text = Some(result.assistant_text.clone());
        record.meta = MemoryMessageMeta {
            model_id: Some(reply.model_id.clone()),
            tokens_generated: Some(result.tokens_generated),
            generation_time_ms: Some(result.generation_time_ms),
            // Kept with the reply: a later turn knows what was looked up.
            tool_calls: result.tool_calls.clone(),
        };
    }
    if result.finish == izwi_agent::TurnFinish::Cancelled {
        return Err("Turn cancelled".to_string());
    }
    plan.agent_session.summary.refresh(
        state,
        &reply.session.thread_id,
        &reply.model_id,
        &plan.correlation_id,
        VOICE_HISTORY_BUDGET,
    );
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
    let memory = ThreadMemory::new(state.chat_store.clone(), None);
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
        "models_ready" => RealtimeServerEvent::ModelsStatus {
            models: value
                .get("models")
                .cloned()
                .and_then(|models| serde_json::from_value(models).ok())
                .unwrap_or_default(),
        },
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
        "function_call" => RealtimeServerEvent::FunctionCall {
            call_id: json_str(value, "call_id"),
            name: json_str(value, "name"),
            arguments: json_str(value, "arguments"),
            executor: match value.get("executor").and_then(serde_json::Value::as_str) {
                // An MCP tool runs on the server's side of the socket.
                Some("server" | "mcp") => Some(RealtimeFunctionExecutor::Server),
                Some("client") => Some(RealtimeFunctionExecutor::Client),
                _ => None,
            },
        },
        "function_call_completed" => RealtimeServerEvent::FunctionCallCompleted {
            call_id: json_str(value, "call_id"),
            name: json_str(value, "name"),
            status: match value.get("status").and_then(serde_json::Value::as_str) {
                Some("ok") => RealtimeFunctionCallStatus::Ok,
                Some("timeout") => RealtimeFunctionCallStatus::Timeout,
                Some("started") => RealtimeFunctionCallStatus::Started,
                _ => RealtimeFunctionCallStatus::Error,
            },
            output: json_str(value, "output"),
        },
        "agent_handoff" => RealtimeServerEvent::AgentHandoff {
            from: json_str(value, "from"),
            to: json_str(value, "to"),
        },
        "task_completed" => RealtimeServerEvent::TaskCompleted {
            name: json_str(value, "name"),
            result: value
                .get("result")
                .map(serde_json::Value::to_string)
                .unwrap_or_default(),
        },
        "turn_paused" => RealtimeServerEvent::TurnPaused,
        "turn_resumed" => RealtimeServerEvent::TurnResumed,
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
        | "conversation_item_truncated"
        | "turn_processing"
        | "user_transcript_start"
        | "user_transcript_delta"
        | "assistant_text_delta" => return None,
        _ => return None,
    };
    Some((event, utterance_id, turn_id))
}

/// A string field of a legacy event, empty when absent.
fn json_str(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn send_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match out_tx.encode_json_event(&value) {
        Ok(Some(message)) => out_tx.send(message),
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
        Ok(Some(message)) => out_tx.send_turn(output_generation, message),
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
        Ok(Some(message)) if out_tx.typed_protocol_enabled() => {
            out_tx.send_turn(output_generation, message)
        }
        Ok(Some(message)) => out_tx.send_text_snapshot(output_generation, key, message),
        Ok(None) => true,
        Err(err) => {
            warn!("failed to serialize voice ws text snapshot: {err}");
            false
        }
    }
}

fn send_diagnostic_json(out_tx: &OutboundTx, value: serde_json::Value) -> bool {
    match out_tx.encode_json_event(&value) {
        Ok(Some(message)) if out_tx.typed_protocol_enabled() => out_tx.send(message),
        Ok(Some(message)) => out_tx.send_diagnostic(message),
        Ok(None) => true,
        Err(err) => {
            warn!("failed to serialize voice ws diagnostic: {err}");
            false
        }
    }
}

const MODEL_LOAD_ERROR_PREFIX: &str = "Failed to load ";

/// The error a turn reports when one of its models cannot be loaded.
pub(crate) fn model_load_error(stage: &str, err: impl std::fmt::Display) -> String {
    format!("{MODEL_LOAD_ERROR_PREFIX}{stage} model: {err}")
}

/// The machine-readable code of a voice error message, so clients can tell
/// a missing model (fix it in settings) from a failed inference (retry).
fn voice_error_code(message: &str) -> Option<&'static str> {
    message
        .starts_with(MODEL_LOAD_ERROR_PREFIX)
        .then_some("model_unavailable")
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
            "code": voice_error_code(&message),
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
            "code": voice_error_code(&message),
            "message": message,
        }),
    );
}

#[cfg(test)]
mod golden_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// An app state on a temporary database with `realtime` realtime
    /// admission permits.
    pub(super) fn test_app_state(realtime: usize) -> (tempfile::TempDir, AppState) {
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

    /// A unified LFM3-Audio turn's tool rounds on real weights: a built-in
    /// call, then a client call answered over the socket, each followed by a
    /// spoken answer. Set `IZWI_LFM3_AUDIO_MODELS_DIR` (and
    /// `IZWI_LFM3_AUDIO_BACKEND=metal` with the `metal` feature) to run it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unified_turns_call_builtin_and_client_tools_if_env_set() {
        let Some(models_dir) = std::env::var_os("IZWI_LFM3_AUDIO_MODELS_DIR") else {
            eprintln!("IZWI_LFM3_AUDIO_MODELS_DIR unset; not run");
            return;
        };
        let backend = std::env::var("IZWI_LFM3_AUDIO_BACKEND")
            .ok()
            .and_then(|name| izwi_core::backends::BackendPreference::parse(&name))
            .unwrap_or(izwi_core::backends::BackendPreference::Cpu);
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let state = {
            let _env = crate::test_support::env_lock();
            std::env::set_var("IZWI_DB_PATH", temp_dir.path().join("izwi.sqlite3"));
            std::env::set_var("IZWI_MEDIA_DIR", temp_dir.path().join("media"));
            let serve_config = izwi_core::ServeRuntimeConfig {
                backend,
                ui_enabled: false,
                models_dir: models_dir.into(),
                ..izwi_core::ServeRuntimeConfig::default()
            };
            let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
            let state = AppState::new(runtime, &serve_config).expect("state");
            for key in ["IZWI_DB_PATH", "IZWI_MEDIA_DIR"] {
                std::env::remove_var(key);
            }
            state
        };
        let variant = izwi_core::ModelVariant::Lfm3Audio2BGguf;
        state
            .runtime
            .load_model(variant)
            .await
            .expect("load LFM3-Audio");
        let tools = VoiceTools::parse(
            Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            })]),
            Some(true),
            None,
            true,
        )
        .unwrap()
        .unwrap();
        let pending = PendingFunctionCalls::default();
        for (question, tool, executor, client_output) in [
            ("What time is it right now?", "time", "server", None),
            (
                "What is the weather in Paris right now?",
                "get_weather",
                "client",
                Some(json!({"temperature_c": 21, "sky": "clear"})),
            ),
        ] {
            let mut request = GenerationRequest::new(question);
            request.model_variant = Some(variant);
            request.config.options.voice = Some("female".into());
            request.config.options.speech.seed = Some(3);
            request.language = Some("en-US".into());
            let speech = state.runtime.generate(request).await.expect("speak");
            let audio = izwi_core::audio::AudioEncoder::new(speech.sample_rate, 1)
                .encode(&speech.samples, izwi_core::audio::AudioFormat::Wav)
                .expect("wav");

            let out_tx = OutboundTx::new(state.runtime.clone(), "voice tools test");
            let reader = {
                let out_tx = out_tx.clone();
                let pending = pending.clone();
                tokio::spawn(async move {
                    let (mut events, mut chunks) = (Vec::new(), Vec::new());
                    while let Some(output) = out_tx.mailbox.next().await {
                        let crate::app::realtime_transport::MailboxOutput::Message(message) =
                            output
                        else {
                            break;
                        };
                        match message {
                            Message::Text(text) => {
                                let event: serde_json::Value =
                                    serde_json::from_str(text.as_str()).expect("json");
                                if event["type"] == "function_call" && event["executor"] == "client"
                                {
                                    let output = client_output.clone().expect("a client call");
                                    let call_id = event["call_id"].as_str().unwrap();
                                    assert!(pending.resolve(call_id, &output));
                                }
                                events.push(event);
                            }
                            Message::Binary(frame) => chunks.push((
                                u32::from_le_bytes(frame[16..20].try_into().unwrap()),
                                frame[6] & 1 == 1,
                            )),
                            _ => {}
                        }
                    }
                    (events, chunks)
                })
            };
            let started = Instant::now();
            let (heard_tx, heard_rx) = tokio::sync::oneshot::channel();
            let reply = {
                let control = TurnControl::new(Instant::now());
                let background = BackgroundResults::default();
                let turn_tools = VoiceTurnTools {
                    tools: &tools,
                    pending: &pending,
                    background: &background,
                    voice_session_id: "voice-session-1",
                };
                stream_unified_s2s_to_socket(
                    &state,
                    &out_tx,
                    "voice-tools-test",
                    "utterance-1",
                    1,
                    &variant.to_string(),
                    None,
                    "You are a helpful voice assistant.",
                    Vec::new(),
                    400,
                    OutputAudioFormat::parse(None).unwrap(),
                    &audio,
                    Some(turn_tools),
                    heard_tx,
                    &control,
                    0,
                    &SessionModelLeases::default(),
                )
                .await
                .expect("a turn with tools")
            };
            assert_eq!(heard_rx.await.ok(), reply.input_transcription);
            out_tx.close();
            let (events, chunks) = reader.await.unwrap();
            let seconds = reply.samples.len() as f64 / f64::from(reply.sample_rate.max(1));
            eprintln!(
                "{question:?} heard {:?} -> {:?} ({seconds:.2}s of speech, {} chunks) in {:.2}s",
                reply.input_transcription,
                reply.text,
                chunks.len(),
                started.elapsed().as_secs_f64()
            );
            let of_type = |kind: &str| {
                events
                    .iter()
                    .filter(|event| event["type"] == kind)
                    .collect::<Vec<_>>()
            };
            let calls = of_type("function_call");
            assert_eq!(calls.len(), 1, "{events:?}");
            assert_eq!(calls[0]["name"], tool);
            assert_eq!(calls[0]["executor"], executor);
            let completed = of_type("function_call_completed");
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0]["status"], "ok", "{:?}", completed[0]);
            assert_eq!(completed[0]["call_id"], calls[0]["call_id"]);
            assert_eq!(of_type("assistant_audio_start").len(), 1);
            assert_eq!(of_type("assistant_audio_done").len(), 1);
            assert!(!reply.text.contains("<tool_call>") && !reply.text.trim().is_empty());
            assert!(of_type("assistant_text_snapshot")
                .iter()
                .all(|event| !event["text"].as_str().unwrap().contains("<tool_call>")));
            assert!(reply.input_transcription.is_some());
            assert!(seconds > 0.5, "spoke {seconds:.2}s");
            // Chunk numbers count on across rounds; at most the last is
            // final.
            assert!(
                chunks.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "{chunks:?}"
            );
            assert!(chunks[..chunks.len() - 1].iter().all(|(_, last)| !last));
        }
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

    /// A barge-in on a reply whose turn already finished (all of it sent,
    /// still playing on the client) stops the client's playback; the turn's
    /// `turn_done` was already sent, so only the interruption is.
    #[tokio::test]
    async fn barge_in_stops_a_finished_reply_still_playing() {
        let (_dir, state) = test_app_state(1);
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let finished = tokio::spawn(async {});
        while !finished.is_finished() {
            tokio::task::yield_now().await;
        }
        let mut active = Some(ActiveTurn {
            utterance_id: "utt-1".to_string(),
            utterance_seq: 1,
            turn_record_id: "turn-1".to_string(),
            state: state.clone(),
            conversation: Arc::default(),
            task: finished,
            control: TurnControl::new(Instant::now()),
            output_generation: 1,
        });

        assert!(!interrupt_active_turn(&out_tx, &mut active, "barge_in"));
        assert!(active.is_none());
        let events = drain_text(&out_tx).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "turn_interrupted");
        assert_eq!(events[0]["utterance_seq"], 1);
        assert_eq!(events[0]["reason"], "barge_in");
    }

    #[test]
    fn hums_and_coughs_as_recognizers_write_them_are_wordless() {
        for sound in [
            "",
            "  ",
            "Uh",
            "Uh uh.",
            "Uh oh.",
            "Oh",
            "Mm-hmm?",
            "um, er...",
        ] {
            assert!(is_wordless(sound), "{sound:?}");
        }
        for words in [
            "Stop",
            "Oh no",
            "Uh, wait",
            "No.",
            "Hm, what about Friday?",
            "2",
        ] {
            assert!(!is_wordless(words), "{words:?}");
        }
    }

    /// A paused client's playback stands still: audio sent meanwhile queues
    /// behind what it holds, and it plays on from where it stopped.
    #[test]
    fn a_paused_playback_clock_stands_still() {
        let start = Instant::now();
        let ms = Duration::from_millis;
        let mut clock = PlaybackClock::default();
        clock.sent(start, ms(2_000));
        clock.pause(start + ms(500));
        assert_eq!(clock.played(start + ms(500)), ms(500));
        // Time passes and more audio arrives; nothing more is played.
        assert!(!clock.sent(start + ms(900), ms(1_000)));
        assert_eq!(clock.played(start + ms(5_000)), ms(500));
        clock.pause(start + ms(6_000));
        assert_eq!(
            clock.played(start + ms(6_000)),
            ms(500),
            "pausing twice changes nothing"
        );
        clock.resume(start + ms(6_000));
        assert_eq!(clock.played(start + ms(6_400)), ms(900));
        assert_eq!(clock.played(start + ms(20_000)), ms(3_000));
        clock.resume(start + ms(21_000));
        assert_eq!(clock.played(start + ms(21_000)), ms(3_000));
    }

    fn running_turn(state: &AppState, utterance_seq: u64) -> ActiveTurn {
        ActiveTurn {
            utterance_id: format!("utt-{utterance_seq}"),
            utterance_seq,
            turn_record_id: format!("turn-{utterance_seq}"),
            state: state.clone(),
            conversation: Arc::default(),
            task: tokio::spawn(tokio::time::sleep(Duration::from_secs(30))),
            control: TurnControl::new(Instant::now()),
            output_generation: utterance_seq,
        }
    }

    /// A held reply is paused once, plays on when the sound was no
    /// interruption, and stays held so that a real one still cancels it.
    #[tokio::test]
    async fn a_held_reply_is_paused_then_resumed_or_interrupted() {
        let (_dir, state) = test_app_state(1);
        let out_tx = OutboundTx::new(state.runtime.clone(), "voice test");
        let types = |events: Vec<serde_json::Value>| -> Vec<String> {
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap_or_default().to_string())
                .collect()
        };
        let held = HeldReply::default();
        assert!(!held.pause(&out_tx, None, "barge_in"), "nothing to pause");
        assert!(!held.resume(&out_tx));

        let turn = running_turn(&state, 1);
        let control = turn.control.clone();
        control.audio_sent(&state.runtime, Duration::from_secs(4));
        {
            let mut reply = control.reply();
            reply.ledger.begin_segment("One two three four.");
            reply.ledger.audio_sent(4_000.0);
        }
        assert!(held.pause(&out_tx, Some(turn), "barge_in"));
        assert!(!held.pause(&out_tx, None, "barge_in"), "already paused");
        assert!(held.is_running() && held.holds_turn());
        assert!(held.control_of(1).is_some() && held.control_of(2).is_none());
        assert!(!control.is_cancelled(), "a paused reply keeps generating");

        assert!(held.resume(&out_tx));
        assert!(!held.resume(&out_tx));
        assert!(
            held.holds_turn(),
            "a resumed reply can still be interrupted"
        );

        // Speech with words: paused again, then cancelled where it stood.
        assert!(held.pause(&out_tx, None, "barge_in"));
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(held.interrupt(&out_tx, "barge_in"));
        assert!(control.is_cancelled());
        assert!(!held.holds_turn());
        let heard_ms = control.reply().heard_ms.expect("heard up to the pause");
        assert!(
            heard_ms < 50.0,
            "playback stood still while paused: {heard_ms} ms"
        );
        assert!(
            !held.interrupt(&out_tx, "barge_in"),
            "nothing left to interrupt"
        );

        // A newer reply replaces a held one, which is then over.
        let first = running_turn(&state, 2);
        let first_control = first.control.clone();
        assert!(held.pause(&out_tx, Some(first), "barge_in"));
        assert!(held.pause(&out_tx, Some(running_turn(&state, 3)), "barge_in"));
        assert!(first_control.is_cancelled());
        assert!(held.control_of(3).is_some());
        held.interrupt(&out_tx, "socket_closed");

        assert_eq!(
            types(drain_text(&out_tx).await),
            [
                // Paused, played on, paused again and interrupted.
                "turn_paused",
                "turn_resumed",
                "turn_paused",
                "turn_interrupted",
                "turn_done",
                // A held reply gives way to a newer one.
                "turn_paused",
                "turn_interrupted",
                "turn_done",
                "turn_paused",
                // The socket closes.
                "turn_interrupted",
                "turn_done",
            ]
        );
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
            paused_with: None,
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
    pub(super) fn fake_synthesis(
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
            turn_config: unified_test_config(),
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
        let mut sink = VoiceSocketReplySink::new(&out_tx, &commit, 1);
        let speech = speak_reply(
            &state,
            &mut sink,
            &control,
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
                tools: None,
                tool_wait_phrase: None,
                agents: None,
            }),
            endpointed_at: Instant::now(),
            endpointing_delay: None,
        };
        let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
        deltas.send("Hello there.".to_string()).expect("delta");
        drop(deltas);
        speak_reply(
            &state,
            &mut VoiceSocketReplySink::new(&out_tx, &commit, 1),
            &control,
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
            vad_threshold: DEFAULT_STREAM_VAD_THRESHOLD,
            min_speech_ms: 50,
            silence_duration_ms: 200,
            max_utterance_ms: 10_000,
            pre_roll_ms: 100,
            barge_in_min_speech_ms: DEFAULT_BARGE_IN_MIN_SPEECH_MS,
            snr_margin_db: izwi_vad::DEFAULT_SNR_MARGIN_DB,
            input_sample_rate_hint: Some(16_000),
            adaptive_silence_floor_ms: None,
            early_transcription: false,
        })
    }

    /// Every frame voiced, whatever its content: the score threshold drops
    /// to the minimum and the SNR check is off, since digital silence never
    /// rises over its own noise floor.
    #[test]
    fn the_silence_window_follows_the_speakers_pauses() {
        let mut window = AdaptiveSilence::new(450, 900);
        // Until four pauses are known, the configured silence stands.
        for pause in [200.0, 220.0, 180.0] {
            window.observe(pause);
            assert_eq!(window.window_ms(), 900);
        }
        // Gaps inside words teach nothing.
        window.observe(60.0);
        assert_eq!(window.window_ms(), 900);
        // A fluent speaker is answered sooner, never under the floor.
        window.observe(240.0);
        assert_eq!(window.window_ms(), 450);
        // One long pause is an outlier; a second moves the window.
        window.observe(500.0);
        assert_eq!(window.window_ms(), 450);
        window.observe(520.0);
        assert_eq!(window.window_ms(), 750);
        // A slow speaker stays at the configured silence.
        for _ in 0..4 {
            window.observe(800.0);
        }
        assert_eq!(window.window_ms(), 900);
        // Old pauses are forgotten.
        for _ in 0..12 {
            window.observe(200.0);
        }
        assert_eq!(window.window_ms(), 450);
    }

    /// A speaker cut off mid-thought speaks again at once: the silence
    /// that split the thought counts as a pause, and the window widens.
    #[test]
    fn speech_right_after_an_endpoint_widens_the_window() {
        let mut input = StreamingInputState::new(StreamingInputConfig {
            silence_duration_ms: 900,
            adaptive_silence_floor_ms: Some(300),
            ..test_streaming_input().config
        });
        always_voiced(&mut input);
        for pause in [200.0, 210.0, 190.0, 205.0] {
            input.adaptive_silence.as_mut().unwrap().observe(pause);
        }
        assert_eq!(input.silence_window_ms(), 308);

        input.last_silence_end = Some((Instant::now(), 308));
        input
            .handle_pcm16_frame(1, 16_000, &pcm16_frame(&[0; 320]))
            .expect("frame");
        assert!(input.active.is_some(), "speech started");
        assert!(input.last_silence_end.is_none());
        // 308 ms of silence plus the gap is the newest pause; with a
        // second one like it the window follows.
        input.adaptive_silence.as_mut().unwrap().observe(320.0);
        assert!(
            input.silence_window_ms() >= 460,
            "{}",
            input.silence_window_ms()
        );

        // Speech long after an endpoint is a new thought.
        let mut input = StreamingInputState::new(StreamingInputConfig {
            adaptive_silence_floor_ms: Some(300),
            ..test_streaming_input().config
        });
        always_voiced(&mut input);
        input.last_silence_end = Some((Instant::now() - RESUMED_SPEECH_WITHIN * 2, 300));
        input
            .handle_pcm16_frame(1, 16_000, &pcm16_frame(&[0; 320]))
            .expect("frame");
        assert!(input.adaptive_silence.as_ref().unwrap().window_ms() == 200);
    }

    fn always_voiced(input: &mut StreamingInputState) {
        input.config.vad_threshold = 0.01;
        input.config.snr_margin_db = 0.0;
        input.endpoint = EndpointDetector::new(input.config.endpoint_config(None));
    }

    fn finalized(
        outcome: Option<UtteranceOutcome>,
    ) -> Option<(EndedUtterance, Vec<u8>, UtteranceEndReason)> {
        match outcome {
            Some(UtteranceOutcome::Finalized(ended, reason)) => {
                let wav_bytes = ended.wav_bytes.clone();
                Some((ended, wav_bytes, reason))
            }
            _ => None,
        }
    }

    fn unified_test_config() -> VoiceTurnConfig {
        VoiceTurnConfig::Unified(UnifiedVoiceTurnConfig {
            s2s_model_id: "LFM2.5-Audio-1.5B".to_string(),
            speaker: None,
            max_output_tokens: 32,
            output_format: OutputAudioFormat::PcmI16,
            tools: None,
        })
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
    fn a_waiting_reply_speaks_english_by_default_and_the_sessions_phrase_otherwise() {
        let phrase = |requested: Option<&str>, language: Option<&str>| {
            resolve_tool_wait_phrase(requested.map(str::to_string), language)
        };
        assert_eq!(phrase(None, None).as_deref(), Some("One moment."));
        assert_eq!(phrase(None, Some("en-US")).as_deref(), Some("One moment."));
        assert_eq!(phrase(None, Some("zu")), None);
        assert_eq!(
            phrase(Some(" Ngicela ulinde. "), Some("zu")).as_deref(),
            Some("Ngicela ulinde.")
        );
        assert_eq!(phrase(Some(""), None), None);
    }

    #[test]
    fn unified_turns_are_told_the_date_and_time_after_the_instructions() {
        let now = "The current local date and time is Friday 13 November 2026, 19:45.";
        assert_eq!(
            unified_turn_instructions(" Be brief. ", now),
            format!("Be brief.\n\n{now}")
        );
        assert_eq!(unified_turn_instructions("  ", now), now);
    }

    #[test]
    fn a_session_declares_agents_that_hand_over_and_collect_tasks() {
        let start: ClientEvent = serde_json::from_value(json!({
            "type": "input_stream_start",
            "agents": [
                {"id": "front", "instructions": "Greet callers.", "handoffs": ["billing"],
                 "tools": ["time"]},
                {"id": "billing", "name": "Billing", "handoff_description": "For invoices.",
                 "task": {"name": "refund", "description": "Record a refund.",
                          "parameters": {"type": "object", "required": ["invoice"]},
                          "then": "front"}},
            ],
        }))
        .unwrap();
        let ClientEvent::InputStreamStart(start) = start else {
            panic!("an input stream start");
        };
        let tools = VoiceTools::parse(None, Some(true), None, true).unwrap();
        let agents = voice_agent_set(start.agents.clone(), "Qwen3.5-4B", tools.as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(agents.first().id, "front");
        assert_eq!(agents.first().name, "front");
        assert_eq!(agents.first().system_prompt, "Greet callers.");
        let billing = agents.get("billing").unwrap();
        assert_eq!(billing.name, "Billing");
        assert_eq!(billing.default_model, "Qwen3.5-4B");
        assert_eq!(
            billing.task.as_ref().unwrap().then.as_deref(),
            Some("front")
        );

        // A tool the session does not offer, and a handoff to nobody.
        let err = voice_agent_set(start.agents.clone(), "m", None).unwrap_err();
        assert!(err.contains("`time`"), "{err}");
        let mut lost = start.agents.clone().unwrap();
        lost.truncate(1);
        let err = voice_agent_set(Some(lost), "m", tools.as_ref()).unwrap_err();
        assert!(err.starts_with("Invalid `agents`:"), "{err}");
        assert_eq!(voice_agent_set(None, "m", None), Ok(None));
        assert_eq!(voice_agent_set(Some(Vec::new()), "m", None), Ok(None));
    }

    #[test]
    fn voice_sessions_declare_tools_and_answer_calls() {
        let start: ClientEvent = serde_json::from_value(json!({
            "type": "input_stream_start",
            "mode": "unified",
            "s2s_model_id": "LFM3-Audio-2B-GGUF",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "builtin_tools": false,
            "function_call_timeout_ms": 5000
        }))
        .unwrap();
        let ClientEvent::InputStreamStart(start) = start else {
            panic!("an input stream start");
        };
        let InputStreamStartEvent {
            tools,
            builtin_tools,
            function_call_timeout_ms,
            ..
        } = *start;
        assert_eq!(tools.map(|tools| tools.len()), Some(1));
        assert_eq!(builtin_tools, Some(false));
        assert_eq!(function_call_timeout_ms, Some(5000));
        let output: ClientEvent = serde_json::from_value(json!({
            "type": "function_call_output",
            "call_id": "call_1",
            "output": {"temperature_c": 21}
        }))
        .unwrap();
        assert!(matches!(
            output,
            ClientEvent::FunctionCallOutput { ref call_id, .. } if call_id == "call_1"
        ));
    }

    /// The typed v2 protocol reports tool calls with their turn.
    #[test]
    fn typed_voice_events_report_tool_calls() {
        let mut typed =
            TypedVoiceSession::new("voice-session-1".to_string(), "process-42".to_string());
        let (call, _, turn_id) = map_typed_voice_event(
            &json!({
                "type": "function_call",
                "utterance_id": "utterance-3",
                "utterance_seq": 3,
                "call_id": "call_1",
                "name": "time",
                "arguments": "{}",
                "executor": "server"
            }),
            &mut typed,
        )
        .expect("a typed function call");
        assert_eq!(turn_id.as_deref(), Some("turn-3"));
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"type": "function_call", "data": {
                "call_id": "call_1", "name": "time", "arguments": "{}", "executor": "server"
            }})
        );
        let (completed, _, _) = map_typed_voice_event(
            &json!({
                "type": "function_call_completed",
                "utterance_id": "utterance-3",
                "utterance_seq": 3,
                "call_id": "call_2",
                "name": "lookup",
                "status": "timeout",
                "output": "The tool did not answer within 10000 ms."
            }),
            &mut typed,
        )
        .expect("a typed completion");
        assert_eq!(
            completed,
            RealtimeServerEvent::FunctionCallCompleted {
                call_id: "call_2".to_string(),
                name: "lookup".to_string(),
                status: RealtimeFunctionCallStatus::Timeout,
                output: "The tool did not answer within 10000 ms.".to_string(),
            }
        );
        let (unknown, _, _) = map_typed_voice_event(
            &json!({"type": "function_call", "call_id": "call_3", "name": "x",
                    "arguments": "{}", "executor": null}),
            &mut typed,
        )
        .expect("a call to an unknown tool");
        assert!(serde_json::to_value(&unknown).unwrap()["data"]
            .get("executor")
            .is_none());
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
        let mut envelope = typed.envelope(event, 0, utterance_id, turn_id);
        envelope.timestamp_ms = 1_725_000_000_123;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../ui/src/features/voice/realtime/voice_v2_session_ready.fixture.json"
        ))
        .expect("shared voice v2 SessionReady fixture");
        assert_eq!(
            serde_json::to_value(&envelope).expect("serialize typed voice SessionReady"),
            fixture
        );
    }

    fn typed_outbound() -> OutboundTx {
        let runtime =
            Arc::new(RuntimeService::new(izwi_core::EngineConfig::default()).expect("runtime"));
        let outbound = OutboundTx::new(runtime, "voice typed test");
        outbound.configure_wire_protocol(
            VoiceWireProtocol::TypedV2,
            "voice-session-1".to_string(),
            "process-42".to_string(),
        );
        outbound
    }

    async fn drain_envelopes(
        outbound: &OutboundTx,
    ) -> Vec<RealtimeEventEnvelope<RealtimeServerEvent>> {
        drain_text(outbound)
            .await
            .into_iter()
            .map(|value| serde_json::from_value(value).expect("typed envelope"))
            .collect()
    }

    fn assert_contiguous(envelopes: &[RealtimeEventEnvelope<RealtimeServerEvent>]) {
        assert_eq!(envelopes[0].sequence, 0);
        assert_eq!(envelopes[0].connection_epoch, 0);
        for pair in envelopes.windows(2) {
            assert_eq!(pair[1].validate_successor(&pair[0]), Ok(()), "{pair:?}");
        }
    }

    #[tokio::test]
    async fn typed_voice_output_cutoff_starts_a_contiguous_new_epoch() {
        let outbound = typed_outbound();
        assert!(send_json(&outbound, json!({"type": "input_stream_ready"})));
        outbound.advance_typed_output_epoch();
        assert!(send_json(
            &outbound,
            json!({
                "type": "turn_interrupted",
                "utterance_id": "utterance-1",
                "utterance_seq": 1,
                "reason": "barge_in",
            }),
        ));

        let envelopes = drain_envelopes(&outbound).await;
        assert_eq!(envelopes.len(), 2);
        assert_eq!(envelopes[1].connection_epoch, 1);
        assert_eq!(envelopes[1].sequence, 0);
        assert!(matches!(
            envelopes[1].event,
            RealtimeServerEvent::Interruption {
                cutoff_event_id: 1,
                cutoff_sequence: 0,
                ..
            }
        ));
        assert_contiguous(&envelopes);
    }

    /// Typed clients learn which models failed to load before anyone speaks,
    /// and a model-load turn error is coded so they can tell it from a
    /// failed inference.
    #[tokio::test]
    async fn typed_voice_reports_model_status_and_coded_model_errors() {
        let outbound = typed_outbound();
        assert!(send_json(
            &outbound,
            json!({
                "type": "models_ready",
                "models": [
                    {"stage": "asr", "model_id": "VibeVoice-ASR", "ready": false,
                     "error": "Model VibeVoice-ASR not downloaded. Please download it first."},
                    {"stage": "text", "model_id": "Qwen3.5-4B", "ready": true},
                ],
                "elapsed_ms": 12,
            }),
        ));
        send_error(
            &outbound,
            None,
            None,
            model_load_error("ASR", "not downloaded"),
        );
        send_error(&outbound, None, None, "Chat generation failed");

        let envelopes = drain_envelopes(&outbound).await;
        let RealtimeServerEvent::ModelsStatus { models } = &envelopes[0].event else {
            panic!("expected models_status, got {:?}", envelopes[0].event);
        };
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].stage, "asr");
        assert!(!models[0].ready);
        assert!(models[0]
            .error
            .as_deref()
            .unwrap()
            .contains("not downloaded"));
        assert!(models[1].ready && models[1].error.is_none());
        assert!(matches!(
            envelopes[1].event,
            RealtimeServerEvent::RecoverableError {
                code: RealtimeErrorCode::ModelUnavailable,
                ..
            }
        ));
        assert!(matches!(
            envelopes[2].event,
            RealtimeServerEvent::RecoverableError {
                code: RealtimeErrorCode::InferenceFailed,
                ..
            }
        ));
    }

    /// A cut-off turn's queued events are dropped before they are sent, so
    /// they must not use up sequence numbers: the client closes the socket on
    /// any gap. This is the finished-turn barge-in, where no new epoch starts.
    #[tokio::test]
    async fn typed_voice_events_dropped_by_a_cutoff_leave_no_sequence_gap() {
        let outbound = typed_outbound();
        assert!(send_json(&outbound, json!({"type": "input_stream_ready"})));
        let turn_event = |event_type: &str| json!({"type": event_type, "utterance_id": "utterance-1", "utterance_seq": 1});
        assert!(send_turn_json(
            &outbound,
            1,
            turn_event("assistant_text_start")
        ));
        assert!(send_turn_json(
            &outbound,
            1,
            turn_event("assistant_audio_done")
        ));
        outbound.cutoff_turn(1);
        assert!(!send_turn_json(&outbound, 1, turn_event("turn_done")));
        assert!(send_json(
            &outbound,
            json!({"type": "user_speech_start", "utterance_id": "utterance-2", "utterance_seq": 2}),
        ));

        let envelopes = drain_envelopes(&outbound).await;
        assert_eq!(envelopes.len(), 2);
        assert!(matches!(
            envelopes[1].event,
            RealtimeServerEvent::SpeechStarted
        ));
        assert_contiguous(&envelopes);
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
            input.endpoint = EndpointDetector::new(input.config.endpoint_config(None));
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

    /// With early transcription an utterance is handed out at each pause
    /// long enough, taken back when the speaker goes on, and the last one
    /// handed out is of the utterance that ends.
    #[test]
    fn an_utterance_is_handed_out_at_a_pause_and_taken_back_when_speech_goes_on() {
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
        // Speech, half a second of silence, speech, then silence to the end.
        let mut samples = vec![0.0f32; 8_000];
        samples.extend_from_slice(&fox);
        samples.extend(vec![0.0f32; 8_000]);
        samples.extend_from_slice(&fox);
        samples.extend(vec![0.0f32; 48_000]);

        let run = |early_transcription: bool| {
            let mut input = StreamingInputState::new(StreamingInputConfig {
                min_speech_ms: DEFAULT_STREAM_MIN_SPEECH_MS,
                // Longer than the pause in the middle, the recording's own
                // leading and trailing silence included.
                silence_duration_ms: 2_000,
                early_transcription,
                ..test_streaming_input().config
            });
            // What happened, in order: e(arly), r(esumed), t(urn).
            let mut story = String::new();
            let mut last_early = None;
            for (index, chunk) in samples.chunks(320).enumerate() {
                let pcm: Vec<i16> = chunk
                    .iter()
                    .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .collect();
                let result = input
                    .handle_pcm16_frame(index as u32 + 1, 16_000, &pcm16_frame(&pcm))
                    .expect("frame");
                if result.speech_resumed {
                    story.push('r');
                }
                if let Some(early) = result.early {
                    story.push('e');
                    last_early = Some((early.utterance_seq, early.wav_bytes.len()));
                }
                if let Some((ended, wav, _)) = finalized(result.utterance_end) {
                    story.push('t');
                    let (seq, early_len) = last_early.take().unwrap_or((0, 0));
                    if early_transcription {
                        assert_eq!(seq, ended.utterance_seq);
                        // The same audio, short of the trailing silence.
                        assert!(early_len < wav.len());
                        assert!(early_len > wav.len() / 2, "{early_len} of {}", wav.len());
                    }
                }
            }
            story
        };
        assert_eq!(run(false), "t");
        let story = run(true);
        assert!(story.ends_with("et"), "{story}");
        assert!(
            story.contains("er"),
            "the middle pause was talked through: {story}"
        );
        assert_eq!(story.matches('t').count(), 1, "{story}");
        // Every hand-out but the last was taken back.
        assert_eq!(
            story.matches('e').count(),
            story.matches('r').count() + 1,
            "{story}"
        );
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
            input.endpoint = EndpointDetector::new(input.config.endpoint_config(None));
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
            input.endpoint = EndpointDetector::new(input.config.endpoint_config(None));
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
        input.endpoint = EndpointDetector::new(input.config.endpoint_config(None));
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

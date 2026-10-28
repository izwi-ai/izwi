//! Request types and processing for the inference engine.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::config::EngineCoreConfig;
use super::executor::PlannedAsrChunks;
use super::metrics::record_engine_stream_backpressure;
use super::output::StreamingOutput;
use super::types::{GenerationParams, Priority, RequestId, TaskType, TokenId};
use super::{
    BatchId, BatchLaneKey, ClockedStateProjection, ClockedStateSpan, InputRange, OutputVisibility,
    PlanId, RealtimeOperationId, ResourceAmount, ResourceVector, SessionKey, StageDescriptor,
    StageId, WorkCost,
};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::chatterbox::retained::ChatterboxPreparedArtifact;
use crate::models::architectures::chatterbox::ChatterboxGenerationParams;
use crate::models::architectures::fish_s2::{FishS2GenerationParams, FishS2PreparedArtifact};
use crate::models::architectures::granite_speech::asr::GraniteSpeechPreparedPromptArtifact;
use crate::models::architectures::kokoro::KokoroPreparedRequest;
use crate::models::architectures::lfm25_audio::{
    lfm25_audio_tts_system_prompt, model::Lfm25AudioPreparedAsrArtifact,
    tts_retained::Lfm25AudioPreparedTtsArtifact, Lfm25AudioGenerationConfig,
};
use crate::models::architectures::lfm3_audio::retained::Lfm3AudioPreparedArtifact;
use crate::models::architectures::maya1::retained::Maya1PreparedArtifact;
use crate::models::architectures::maya1::Maya1GenerationParams;
use crate::models::architectures::maya2::retained::Maya2PreparedArtifact;
use crate::models::architectures::maya2::Maya2GenerationParams;
use crate::models::architectures::parakeet::asr::ParakeetPreparedEncoderArtifact;
use crate::models::architectures::qwen3::asr::Qwen3AsrPreparedAudio;
use crate::models::architectures::qwen3::tts::{
    Qwen3TtsModel, SpeakerReference, TtsGenerationParams, TtsSessionCacheRequest,
};
use crate::models::architectures::vibevoice::asr::VibeVoiceAsrPreparedArtifact;
use crate::models::architectures::vibevoice::tts::{
    VibeVoiceTtsGenerationParams, VibeVoiceTtsPreparedArtifact,
};
use crate::models::architectures::voxtral::tts::retained::VoxtralTtsPreparedArtifact;
use crate::models::architectures::voxtral::tts::VoxtralTtsGenerationParams;
use crate::models::architectures::whisper::asr::WhisperPreparedWindow;
use crate::models::registry::{
    AsrModelLease, ChatModelLease, ChatterboxTtsModelLease, FishS2TtsModelLease,
    KokoroTtsModelLease, Lfm25AudioModelLease, Maya1TtsModelLease, Maya2TtsModelLease,
    NativeAsrModel, NativeChatModel, NativeChatPreparedPrompt, QwenTtsModelLease,
    VibeVoiceTtsModelLease, VoxtralTtsModelLease,
};
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig, ChatRole};
use crate::runtime::audio_io::{
    validate_base64_audio_retained_size, validate_base64_audio_source_input,
    MAX_AUDIO_SOURCE_BYTES, MAX_REFERENCE_SOURCE_BYTES,
};

mod asr_preparation;
mod chat_preparation;
mod constructors;
mod payloads;
mod realtime;
mod state_binding;
mod streaming;
mod tts_preparation;

/// Status of a request in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Request is waiting to be scheduled
    Waiting,
    /// Request is currently being processed
    Running,
    /// Request has completed successfully
    Finished,
    /// Request was aborted
    Aborted,
    /// Request failed with an error
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EngineStreamPolicy {
    #[default]
    FailOnFull,
    BlockWithDeadline {
        timeout_ms: u64,
    },
    /// Drop the newly produced output when the bounded queue is full.
    DropNewest,
}

impl EngineStreamPolicy {
    fn validate(self) -> Result<()> {
        if let Self::BlockWithDeadline { timeout_ms } = self {
            if timeout_ms > STREAM_BACKPRESSURE_MAX_TIMEOUT_MS {
                return Err(Error::InvalidInput(format!(
                    "stream backpressure timeout {timeout_ms}ms exceeds the {}ms limit",
                    STREAM_BACKPRESSURE_MAX_TIMEOUT_MS
                )));
            }
        }
        Ok(())
    }
}

/// Coarse workload class used by admission and latency-aware scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum WorkloadClass {
    /// Realtime voice or transcription work where first output latency matters most.
    Realtime,
    /// Interactive non-streaming user work.
    Interactive,
    /// User-visible streaming work such as chat or TTS over SSE/websocket.
    Streaming,
    /// Default online API work.
    #[default]
    Online,
    /// Offline batch jobs.
    Batch,
    /// Opportunistic background work.
    Background,
}

impl WorkloadClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Interactive => "interactive",
            Self::Streaming => "streaming",
            Self::Online => "online",
            Self::Batch => "batch",
            Self::Background => "background",
        }
    }

    pub fn is_latency_sensitive(self) -> bool {
        matches!(self, Self::Realtime | Self::Interactive | Self::Streaming)
    }

    /// Relative share of service under weighted fair scheduling: of
    /// scheduler quanta, and of device time between execution lanes.
    pub(crate) fn service_weight(self) -> u64 {
        match self {
            Self::Realtime => 8,
            Self::Interactive => 6,
            Self::Streaming => 5,
            Self::Online => 4,
            Self::Batch => 2,
            Self::Background => 1,
        }
    }

    pub fn prefers_single_token_decode(self) -> bool {
        self.is_latency_sensitive()
    }

    pub fn adaptive_score_boost(self) -> f64 {
        match self {
            Self::Realtime => 3.0,
            Self::Interactive => 2.0,
            Self::Streaming => 1.75,
            Self::Online => 0.0,
            Self::Batch => -0.35,
            Self::Background => -0.75,
        }
    }

    pub fn deadline_scale(self) -> f64 {
        match self {
            Self::Realtime => 0.45,
            Self::Interactive => 0.65,
            Self::Streaming => 0.75,
            Self::Online => 1.0,
            Self::Batch => 1.75,
            Self::Background => 2.50,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EngineAudioInput {
    Base64(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct TtsEngineInput {
    pub text: String,
    pub reference_audio: Option<String>,
    pub reference_text: Option<String>,
    pub voice_description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AsrEngineInput {
    pub audio: EngineAudioInput,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

/// What an ASR request asks for beyond transcription in the spoken language.
/// Only families that report the capability accept a non-empty value
/// (`ModelVariant::supports_asr_translation`, `supports_asr_keyword_biasing`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AsrTaskOptions {
    /// Translate the speech into this language instead of transcribing it.
    pub translate_to: Option<String>,
    /// Terms to bias recognition toward.
    pub keywords: Vec<String>,
}

impl AsrTaskOptions {
    /// Options with a blank target and blank keywords dropped.
    pub fn new(translate_to: Option<String>, keywords: Vec<String>) -> Self {
        Self {
            translate_to: translate_to
                .map(|target| target.trim().to_string())
                .filter(|target| !target.is_empty()),
            keywords: keywords
                .into_iter()
                .map(|keyword| keyword.trim().to_string())
                .filter(|keyword| !keyword.is_empty())
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.translate_to.is_none() && self.keywords.is_empty()
    }

    /// Host bytes the options keep alive while the request is retained.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.translate_to.as_ref().map_or(0, String::capacity)
            + self.keywords.capacity() * std::mem::size_of::<String>()
            + self.keywords.iter().map(String::capacity).sum::<usize>()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum RealtimeAsrOperationPayload {
    Push {
        samples: Arc<[f32]>,
        sample_rate: u32,
    },
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeAsrOperationKind {
    Push,
    Finish,
}

impl RealtimeAsrOperationPayload {
    const fn kind(&self) -> RealtimeAsrOperationKind {
        match self {
            Self::Push { .. } => RealtimeAsrOperationKind::Push,
            Self::Finish => RealtimeAsrOperationKind::Finish,
        }
    }

    fn accepted_samples(&self) -> usize {
        match self {
            Self::Push { samples, .. } => samples.len(),
            Self::Finish => 0,
        }
    }
}

/// Durable acknowledgement for one exact externally allocated operation ID.
/// The acknowledgement intentionally carries no payload owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeAsrOperationAck {
    operation_id: RealtimeOperationId,
    kind: RealtimeAsrOperationKind,
    accepted_samples: usize,
}

impl RealtimeAsrOperationAck {
    #[cfg(test)]
    pub(crate) const fn for_test(
        operation_id: RealtimeOperationId,
        kind: RealtimeAsrOperationKind,
        accepted_samples: usize,
    ) -> Self {
        Self {
            operation_id,
            kind,
            accepted_samples,
        }
    }

    pub const fn operation_id(&self) -> RealtimeOperationId {
        self.operation_id
    }

    pub const fn kind(&self) -> RealtimeAsrOperationKind {
        self.kind
    }

    pub const fn accepted_samples(&self) -> usize {
        self.accepted_samples
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RealtimeAsrTerminalOutcome {
    Completed,
    Cancelled,
    TimedOut,
    Unloaded,
    Failed(Arc<str>),
}

pub(crate) type RealtimeAsrOperationOutcome =
    std::result::Result<RealtimeAsrOperationAck, RealtimeAsrTerminalOutcome>;
pub(crate) type RealtimeAsrOperationWaiter = oneshot::Receiver<RealtimeAsrOperationOutcome>;

#[derive(Debug)]
struct RealtimeAsrIngressOperation {
    payload: RealtimeAsrOperationPayload,
    preparation_cost: WorkCost,
    completion: oneshot::Sender<RealtimeAsrOperationOutcome>,
}

#[derive(Debug, Default)]
struct RealtimeAsrIngressState {
    operations: std::collections::HashMap<RealtimeOperationId, RealtimeAsrIngressOperation>,
    highest_operation_id: Option<RealtimeOperationId>,
    terminal: Option<RealtimeAsrTerminalOutcome>,
}

#[derive(Debug, Default)]
pub(super) struct RealtimeAsrIngress {
    state: Mutex<RealtimeAsrIngressState>,
}

impl Drop for RealtimeAsrIngress {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        let terminal = state
            .terminal
            .clone()
            .unwrap_or(RealtimeAsrTerminalOutcome::Cancelled);
        for (_, operation) in state.operations.drain() {
            let _ = operation.completion.send(Err(terminal.clone()));
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatEngineInput {
    pub messages: Vec<ChatMessage>,
    pub chat_config: ChatRequestConfig,
    pub prompt_tokens: Vec<TokenId>,
}

#[derive(Debug, Clone)]
pub struct AudioChatEngineInput {
    pub audio: EngineAudioInput,
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub enum EngineTask {
    Tts(TtsEngineInput),
    Asr(AsrEngineInput),
    Chat(ChatEngineInput),
    SpeechToSpeech(AudioChatEngineInput),
}

impl EngineTask {
    pub fn task_type(&self) -> TaskType {
        match self {
            Self::Tts(_) => TaskType::TTS,
            Self::Asr(_) => TaskType::ASR,
            Self::Chat(_) => TaskType::Chat,
            Self::SpeechToSpeech(_) => TaskType::SpeechToSpeech,
        }
    }
}

/// Opaque proof that the exact chat payload was tokenized by a loaded model
/// before scheduler admission. Public request fields cannot construct this
/// marker, and the fingerprint catches any internal mutation after preparation.
#[derive(Debug, Clone)]
pub(super) struct ChatExecutionReady {
    model_variant: ModelVariant,
    model: PreparedChatModel,
    fingerprint: u64,
    prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
    context_limit: usize,
    core_validated: bool,
}

/// The exact loaded model instance that produced the prepared prompt. Holding
/// the `Arc` in the opaque preparation marker prevents a concurrent registry
/// unload/reload from switching model instances between tokenization and
/// execution.
#[derive(Clone)]
enum PreparedChatModel {
    Exact(ChatModelLease),
    #[cfg(test)]
    ValidationOnly,
}

/// The exact registry model selected for an incremental non-chat request.
/// The lease stays inside the immutable core request from admission through
/// terminal cleanup, so every decode quantum sees the same model instance.
#[derive(Clone)]
enum PreparedIncrementalModel {
    Asr(AsrModelLease),
    Lfm25AudioAsr(Lfm25AudioModelLease),
    Lfm25AudioTts(Lfm25AudioModelLease),
    QwenTts(QwenTtsModelLease),
    VibeVoiceTts(VibeVoiceTtsModelLease),
    FishS2Tts(FishS2TtsModelLease),
    VoxtralTts(VoxtralTtsModelLease),
    ChatterboxTts(ChatterboxTtsModelLease),
    Maya1Tts(Maya1TtsModelLease),
    Maya2Tts(Maya2TtsModelLease),
    /// LFM3-Audio speech as a retained row (the audio-chat model's lease).
    Lfm3Audio(Lfm25AudioModelLease),
    KokoroTts(KokoroTtsModelLease),
}

impl fmt::Debug for PreparedIncrementalModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asr(model) => write!(formatter, "Asr({:p})", &**model),
            Self::Lfm25AudioAsr(model) => write!(formatter, "Lfm25AudioAsr({:p})", &**model),
            Self::Lfm25AudioTts(model) => write!(formatter, "Lfm25AudioTts({:p})", &**model),
            Self::QwenTts(model) => write!(formatter, "QwenTts({:p})", &**model),
            Self::VibeVoiceTts(model) => write!(formatter, "VibeVoiceTts({:p})", &**model),
            Self::FishS2Tts(model) => write!(formatter, "FishS2Tts({:p})", &**model),
            Self::VoxtralTts(model) => write!(formatter, "VoxtralTts({:p})", &**model),
            Self::ChatterboxTts(model) => write!(formatter, "ChatterboxTts({:p})", &**model),
            Self::Maya1Tts(model) => write!(formatter, "Maya1Tts({:p})", &**model),
            Self::Maya2Tts(model) => write!(formatter, "Maya2Tts({:p})", &**model),
            Self::Lfm3Audio(model) => write!(formatter, "Lfm3Audio({:p})", &**model),
            Self::KokoroTts(model) => write!(formatter, "KokoroTts({:p})", &**model),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct IncrementalModelExecutionReady {
    model_variant: ModelVariant,
    model: PreparedIncrementalModel,
    qwen_tts: Option<PreparedQwenTtsInput>,
    lfm25_audio_tts: Option<Arc<Lfm25AudioPreparedTtsArtifact>>,
    vibevoice_tts: Option<Arc<VibeVoiceTtsPreparedArtifact>>,
    vibevoice_tts_params: Option<VibeVoiceTtsGenerationParams>,
    fish_s2_tts: Option<Arc<FishS2PreparedArtifact>>,
    fish_s2_tts_params: Option<FishS2GenerationParams>,
    voxtral_tts: Option<Arc<VoxtralTtsPreparedArtifact>>,
    voxtral_tts_params: Option<VoxtralTtsGenerationParams>,
    chatterbox_tts: Option<Arc<ChatterboxPreparedArtifact>>,
    chatterbox_tts_params: Option<ChatterboxGenerationParams>,
    maya1_tts: Option<Arc<Maya1PreparedArtifact>>,
    maya1_tts_params: Option<Maya1GenerationParams>,
    maya2_tts: Option<Arc<Maya2PreparedArtifact>>,
    maya2_tts_params: Option<Maya2GenerationParams>,
    lfm3_audio: Option<Arc<Lfm3AudioPreparedArtifact>>,
    kokoro_tts: Option<Arc<KokoroPreparedText>>,
}

/// Speech-rate range every TTS request is clamped to. Clients may send
/// OpenAI's 0.25–4.0; the engine supports 0.5–2.0 (as `izwi tts --speed`
/// documents).
pub(crate) const TTS_SPEED_RANGE: (f32, f32) = crate::catalog::SpeechModelCapabilities::SPEED_RANGE;

pub(crate) fn clamp_tts_speed(speed: f32) -> f32 {
    speed.clamp(TTS_SPEED_RANGE.0, TTS_SPEED_RANGE.1)
}

/// What an LFM3-Audio voice-chat reply is prepared from: the text turns
/// before the spoken one, its instructions, its slot budget and sampling.
#[derive(Debug, Clone)]
pub(crate) struct Lfm3AudioReplySettings {
    pub(crate) history: Vec<crate::models::shared::chat::ChatMessage>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) max_slots: usize,
    pub(crate) text_sampling: crate::models::architectures::lfm3_audio::config::Lfm3TextSampling,
    pub(crate) audio_sampling: crate::models::architectures::lfm3_audio::config::Lfm3AudioSampling,
    pub(crate) seed: u64,
}

/// A Kokoro request prepared for synthesis. The voice pack conditions at
/// most 510 phonemes, so longer text is split by the chunk planner into
/// pieces that each fit (one piece for most requests); the pieces are
/// synthesized in order and joined with a short pause.
#[derive(Debug, Clone)]
pub(crate) struct KokoroPreparedText {
    pub(crate) source_text: String,
    pub(crate) requested_speaker: Option<String>,
    pub(crate) requested_language: Option<String>,
    pub(crate) requested_speed: f32,
    pub(crate) chunks: Vec<KokoroPreparedRequest>,
}

impl KokoroPreparedText {
    /// Model input tokens across every chunk (each chunk's phoneme tokens
    /// plus its two boundary tokens).
    pub(crate) fn input_tokens(&self) -> Result<usize> {
        self.chunks
            .iter()
            .try_fold(0usize, |total, chunk| {
                total.checked_add(chunk.token_ids.len().checked_add(2)?)
            })
            .ok_or_else(|| Error::InvalidInput("Kokoro prepared token count overflowed".into()))
    }

    fn longest_chunk_tokens(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.token_ids.len().saturating_add(2))
            .max()
            .unwrap_or(0)
    }

    fn is_well_formed(&self) -> bool {
        !self.chunks.is_empty()
            && self.chunks.iter().all(|chunk| {
                !chunk.token_ids.is_empty()
                    && !chunk.phonemes.is_empty()
                    && chunk.speed.is_finite()
                    && chunk.requested_speed.to_bits() == self.requested_speed.to_bits()
            })
    }

    fn matches_request(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> bool {
        self.source_text == text
            && self.requested_speaker.as_deref() == speaker
            && self.requested_language.as_deref() == language
            && self.requested_speed.to_bits() == speed.to_bits()
    }

    pub(crate) fn retained_tensor_bytes(&self) -> Result<u64> {
        self.chunks.iter().try_fold(0u64, |total, chunk| {
            total
                .checked_add(chunk.retained_tensor_bytes()?)
                .ok_or_else(|| Error::Overloaded("Kokoro prepared bytes overflowed".into()))
        })
    }

    pub(crate) fn retained_host_bytes(&self) -> Result<u64> {
        let own = [
            self.source_text.capacity(),
            self.requested_speaker.as_ref().map_or(0, String::capacity),
            self.requested_language.as_ref().map_or(0, String::capacity),
        ]
        .into_iter()
        .sum::<usize>();
        self.chunks.iter().try_fold(own as u64, |total, chunk| {
            total
                .checked_add(chunk.retained_host_bytes()?)
                .ok_or_else(|| Error::Overloaded("Kokoro prepared bytes overflowed".into()))
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedQwenTtsInput {
    pub(crate) reference: Option<Arc<SpeakerReference>>,
    pub(crate) speaker: Option<String>,
    pub(crate) prefill_tokens: usize,
    pub(crate) max_frames: usize,
}

/// Exact, host-prepared cost for one loaded adapter stage. Model preparation
/// may publish shape and transient collation requirements here, but it must not
/// allocate device tensors before physical-batch admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreparedStageCost {
    stage_id: StageId,
    cost: WorkCost,
}

pub(super) const STREAM_PROGRESS_QUEUE_CAPACITY: usize = 64;
pub(super) const STREAM_PROGRESS_MAX_BUFFERED_BYTES: usize = 16 * 1024 * 1024;
const STREAM_PROGRESS_MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const STREAM_BACKPRESSURE_MAX_TIMEOUT_MS: u64 = 60_000;

pub(super) fn checked_stream_backpressure_deadline(timeout_ms: u64) -> Result<Instant> {
    EngineStreamPolicy::BlockWithDeadline { timeout_ms }.validate()?;
    let timeout = Duration::from_millis(timeout_ms.max(1));
    Instant::now().checked_add(timeout).ok_or_else(|| {
        Error::InvalidInput(format!(
            "stream backpressure timeout {timeout_ms}ms exceeds the supported deadline range"
        ))
    })
}
const STREAM_STAGED_MAX_EVENTS: usize = 4096;

#[derive(Debug)]
pub(super) struct StreamProgressBudget {
    max_bytes: usize,
    state: Mutex<usize>,
    available: Condvar,
}

#[derive(Debug)]
pub(super) struct StreamProgressPermit {
    budget: Arc<StreamProgressBudget>,
    bytes: usize,
}

/// Candidate output credit reserved before codec materialization. The staging
/// outbox and caller share ownership until commit transfers it to delivery.
#[derive(Debug, Clone)]
pub(crate) struct AudioOutputCredit {
    _permits: Arc<Vec<StreamProgressPermit>>,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct FencedStreamProgress {
    pub(super) batch_id: BatchId,
    pub(super) lane: BatchLaneKey,
    pub(super) plan_id: PlanId,
    pub(super) session: SessionKey,
    pub(super) output: StreamingOutput,
    pub(super) budget_permit: StreamProgressPermit,
}

#[derive(Debug, Clone)]
struct IncrementalStreamBinding {
    batch_id: BatchId,
    lane: BatchLaneKey,
    plan_id: PlanId,
    session: SessionKey,
    progress_tx: mpsc::Sender<FencedStreamProgress>,
    budget: Arc<StreamProgressBudget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamPushOutcome {
    Accepted,
    Dropped,
}

#[derive(Debug, Default)]
struct StreamBufferState {
    staged: Vec<StreamingOutput>,
    staged_bytes: usize,
    binding: Option<IncrementalStreamBinding>,
    audio_credits: Vec<AudioOutputCredit>,
    audio_budget: Option<Arc<StreamProgressBudget>>,
}

/// Bounded stream outbox and transaction-scoped progress binding. Clones
/// intentionally share state because requests are wrapped in `Arc` before
/// scheduling.
#[derive(Debug, Clone, Default)]
pub(super) struct StreamStagingBuffer {
    state: Arc<Mutex<StreamBufferState>>,
}

#[derive(Debug)]
pub(super) struct StreamBindingGuard {
    buffer: StreamStagingBuffer,
    plan_id: PlanId,
    session: SessionKey,
}

impl fmt::Debug for PreparedChatModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(model) => write!(formatter, "Exact({:p})", &**model),
            #[cfg(test)]
            Self::ValidationOnly => formatter.write_str("ValidationOnly"),
        }
    }
}

struct FingerprintWriter<'a>(&'a mut std::collections::hash_map::DefaultHasher);

impl Write for FingerprintWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn chat_execution_fingerprint(
    model_variant: ModelVariant,
    messages: &[ChatMessage],
    chat_config: &ChatRequestConfig,
    prompt_tokens: &[TokenId],
) -> Result<u64> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    model_variant.hash(&mut hasher);
    prompt_tokens.hash(&mut hasher);
    serde_json::to_writer(FingerprintWriter(&mut hasher), &(messages, chat_config))?;
    Ok(hasher.finish())
}

const DIRECT_CHAT_INPUT_BYTES_PER_CONTEXT_TOKEN: usize = 64;
const DIRECT_CHAT_MIN_INPUT_BYTES: usize = 64 * 1024;
const DIRECT_CHAT_MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;
const DIRECT_CHAT_MAX_TOOL_JSON_DEPTH: usize = 64;
const DIRECT_CHAT_JSON_NODE_BYTES: usize = 32;
const DIRECT_TTS_REFERENCE_TEXT_CHARS_PER_CONTEXT_TOKEN: usize = 8;

#[derive(Debug, Clone, Copy)]
struct DirectPayloadLimits {
    audio_source_bytes: usize,
    reference_source_bytes: usize,
    reference_text_chars: usize,
    metadata_bytes: usize,
    collection_items: usize,
}

impl DirectPayloadLimits {
    fn production(max_seq_len: usize) -> Result<Self> {
        let collection_items = max_seq_len.max(1);
        let metadata_bytes = collection_items
            .checked_mul(DIRECT_TTS_REFERENCE_TEXT_CHARS_PER_CONTEXT_TOKEN)
            .ok_or_else(|| {
                Error::InvalidInput("Direct Engine metadata limit overflow".to_string())
            })?;
        Ok(Self {
            audio_source_bytes: MAX_AUDIO_SOURCE_BYTES,
            reference_source_bytes: MAX_REFERENCE_SOURCE_BYTES,
            reference_text_chars: metadata_bytes,
            metadata_bytes,
            collection_items,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum BorrowedEngineAudioInput<'a> {
    Base64(&'a String),
    Bytes(&'a Vec<u8>),
}

/// Which upload field a digest was computed from, and that buffer's address
/// and length. Replacing the upload through a public field changes it, so
/// the accessors then hash the new bytes in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AsrSourceAudioIdentity {
    base64: bool,
    address: usize,
    len: usize,
}

impl AsrSourceAudioIdentity {
    fn of(audio: BorrowedEngineAudioInput<'_>) -> Self {
        match audio {
            BorrowedEngineAudioInput::Base64(value) => Self {
                base64: true,
                address: value.as_ptr() as usize,
                len: value.len(),
            },
            BorrowedEngineAudioInput::Bytes(value) => Self {
                base64: false,
                address: value.as_ptr() as usize,
                len: value.len(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AsrSourceAudioDigest {
    identity: AsrSourceAudioIdentity,
    digest: u64,
}

/// Hash of an encoded upload's bytes and representation.
fn asr_source_audio_digest(audio: BorrowedEngineAudioInput<'_>) -> u64 {
    #[cfg(test)]
    ASR_SOURCE_AUDIO_FULL_HASHES.with(|count| count.set(count.get() + 1));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match audio {
        BorrowedEngineAudioInput::Base64(value) => {
            0_u8.hash(&mut hasher);
            value.as_bytes().hash(&mut hasher);
        }
        BorrowedEngineAudioInput::Bytes(value) => {
            1_u8.hash(&mut hasher);
            value.as_slice().hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(test)]
thread_local! {
    /// Full upload hashes on this thread, for tests that count them.
    static ASR_SOURCE_AUDIO_FULL_HASHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl EngineAudioInput {
    fn is_empty(&self) -> bool {
        match self {
            Self::Base64(value) => value.is_empty(),
            Self::Bytes(value) => value.is_empty(),
        }
    }
}

fn add_bounded_chat_input_bytes(
    total: &mut usize,
    amount: usize,
    limit: usize,
    label: &str,
) -> Result<()> {
    *total = total
        .checked_add(amount)
        .ok_or_else(|| Error::InvalidInput(format!("Direct chat {label} size overflow")))?;
    if *total > limit {
        return Err(Error::InvalidInput(format!(
            "Direct chat preparation input exceeds the {limit} byte limit"
        )));
    }
    Ok(())
}

fn escaped_json_string_upper_bound(value: &str) -> Result<usize> {
    // A JSON control character can expand to a six-byte `\\u00XX` escape.
    value
        .len()
        .checked_mul(6)
        .ok_or_else(|| Error::InvalidInput("Direct chat JSON string size overflow".to_string()))
}

/// Exact Qwen3 ASR execution shape resolved from decoded media before
/// scheduler admission. Retained sequence rows use managed KV; long-form rows
/// use only their invocation-scoped atomic pipeline workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedAsrExecutionShape {
    RetainedSequence { input_tokens: usize },
    LongFormAtomic,
}

#[derive(Clone)]
pub(super) struct PreparedAsrAudio {
    model_variant: ModelVariant,
    samples: Arc<[f32]>,
    sample_rate: u32,
    source_fingerprint: u64,
}

#[derive(Clone)]
pub(super) struct PreparedAsrEncoderArtifact {
    model_variant: ModelVariant,
    artifact: PreparedAsrEncoderArtifactValue,
    source_fingerprint: u64,
}

#[derive(Clone)]
enum PreparedAsrEncoderArtifactValue {
    Qwen3(Arc<Qwen3AsrPreparedAudio>),
    Whisper(Arc<WhisperPreparedWindow>),
    VibeVoice(Arc<VibeVoiceAsrPreparedArtifact>),
    GraniteSpeech(Arc<GraniteSpeechPreparedPromptArtifact>),
    Lfm25Audio(Arc<Lfm25AudioPreparedAsrArtifact>),
    Parakeet(Arc<ParakeetPreparedEncoderArtifact>),
}

impl fmt::Debug for PreparedAsrEncoderArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedAsrEncoderArtifact")
            .field("model_variant", &self.model_variant)
            .field("artifact", &self.artifact.family_name())
            .field("source_fingerprint", &self.source_fingerprint)
            .finish()
    }
}

impl PreparedAsrEncoderArtifactValue {
    const fn family_name(&self) -> &'static str {
        match self {
            Self::Qwen3(_) => "qwen3",
            Self::Whisper(_) => "whisper",
            Self::VibeVoice(_) => "vibevoice",
            Self::GraniteSpeech(_) => "granite_speech",
            Self::Lfm25Audio(_) => "lfm25_audio",
            Self::Parakeet(_) => "parakeet",
        }
    }

    fn resident_resources(&self) -> Result<(u64, u64)> {
        match self {
            Self::Qwen3(value) => Ok((0, value.resident_tensor_bytes()?)),
            Self::Whisper(value) => Ok((0, value.resident_tensor_bytes()?)),
            Self::VibeVoice(value) => {
                Ok((value.resident_host_bytes(), value.resident_tensor_bytes()))
            }
            Self::GraniteSpeech(value) => {
                Ok((value.resident_host_bytes(), value.resident_tensor_bytes()?))
            }
            Self::Lfm25Audio(value) => {
                Ok((value.retained_host_bytes, value.retained_resident_bytes))
            }
            Self::Parakeet(value) => Ok((0, value.resident_tensor_bytes()?)),
        }
    }

    fn clocked_state_projections(&self) -> &[ClockedStateProjection] {
        match self {
            Self::VibeVoice(value) => value.tokenizer_state_projections(),
            Self::Qwen3(_)
            | Self::Whisper(_)
            | Self::GraniteSpeech(_)
            | Self::Lfm25Audio(_)
            | Self::Parakeet(_) => &[],
        }
    }
}

impl fmt::Debug for PreparedAsrAudio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedAsrAudio")
            .field("model_variant", &self.model_variant)
            .field("samples", &self.samples.len())
            .field("sample_rate", &self.sample_rate)
            .field("source_fingerprint", &self.source_fingerprint)
            .finish()
    }
}

/// A request to the engine core.
#[derive(Debug, Clone)]
pub struct EngineCoreRequest {
    /// Unique request ID
    pub id: RequestId,
    /// Typed task payload used by engine internals.
    ///
    /// Request constructors defer large audio/reference buffers to the legacy
    /// compatibility fields below so constructing a request does not duplicate
    /// caller-owned storage. Processing validates and moves those buffers here,
    /// making this the canonical execution representation.
    pub task: EngineTask,
    /// Task type (TTS, ASR, AudioChat)
    pub task_type: TaskType,
    /// Specific model variant to route to.
    pub model_variant: Option<ModelVariant>,
    /// Exact authoritative load generation. Requests without this lifecycle
    /// fence remain eligible only for scalar compatibility execution.
    pub(super) model_instance_id: Option<super::ModelInstanceId>,
    /// Exact loaded capability adapter selected by the runtime. Direct engine
    /// callers without a lifecycle bundle remain on compatibility dispatch.
    pub(super) execution_adapter_binding: Option<super::ExecutionAdapterBinding>,
    /// ABI-v2 state truth resolved by the lifecycle before the request can
    /// enter scheduler execution.
    pub(super) v2_state_descriptor: Option<crate::kv::v2::CapabilityStateDescriptorV2>,
    pub(super) v2_state_fingerprint: Option<[u8; 32]>,
    /// Exact load-sealed runtime proof for stateless/zero-workspace ABI-v2
    /// capabilities. Stateful variants replace this proof only when they own
    /// backend allocations and completion tracking.
    pub(super) v2_state_runtime: Option<Arc<crate::kv::v2::CapabilityStateRuntimeV2>>,
    /// Immutable model-level physical KV runtime installed by the engine.
    pub(super) managed_cache_runtime: Option<Arc<super::cache::managed::ManagedKvModelRuntime>>,
    /// DS1.5: prefill start cursor probed from the managed prefix index at
    /// admission. Tokens below the cursor are already resident as shared
    /// pages plus a forked tensor snapshot; the scheduler plans the first
    /// prefill span from this cursor and the managed prepare re-verifies it
    /// transactionally before any attach is staged.
    pub(super) managed_prefix_cursor: Option<u32>,
    /// Phase 5a: token count of the prompt's stable history, declared by the
    /// prepared prompt. A later request continuing the same conversation
    /// shares exactly these tokens, so a hybrid model snapshots its state at
    /// the interval-aligned boundary at or below it (see the scheduler's span
    /// clip and the manager's snapshot publication). `None` keeps the
    /// first-chunk-only publication.
    pub(super) managed_snapshot_boundary: Option<u32>,
    /// The end of the prompt's first turn (its system prompt, when it has
    /// one), declared alongside the history boundary. Requests sharing a
    /// system prompt but not their user turns share exactly these tokens, so
    /// a hybrid model also snapshots at its interval floor.
    pub(super) managed_preamble_boundary: Option<u32>,
    /// Request-specific shape/workspace facts produced by the exact loaded
    /// model. The engine remains model-neutral and keys these facts by the
    /// opaque stage identity from the loaded adapter contract.
    pub(super) prepared_stage_costs: Vec<PreparedStageCost>,
    /// Exact logical input span produced by model-specific multimodal
    /// preprocessing before scheduler admission. Text models derive this from
    /// `prompt_tokens`; speech and future multimodal adapters publish it here
    /// instead of manufacturing placeholder token IDs.
    pub(super) prepared_sequence_input_tokens: Option<usize>,
    pub(super) prepared_asr_execution_shape: Option<PreparedAsrExecutionShape>,
    /// Segmentation a long-form row executes, planned once at preparation.
    pub(super) prepared_asr_route_plan: Option<Arc<PlannedAsrChunks>>,
    pub(super) prepared_asr_audio: Option<PreparedAsrAudio>,
    /// Digest of the encoded upload, keyed by the buffer it was computed
    /// from. Preparation hashes the upload once; the per-step accessors reuse
    /// the digest while the same buffer is in place.
    pub(super) asr_source_audio_digest: Option<AsrSourceAudioDigest>,
    pub(super) prepared_asr_encoder_artifact: Option<PreparedAsrEncoderArtifact>,
    /// Executor-produced stream events remain invisible until their exact
    /// execution report has committed.
    pub(super) stream_staging: StreamStagingBuffer,
    /// Retain whole-wave samples for callers requesting a complete audio result.
    pub(crate) collect_audio_samples: bool,
    pub(crate) tenant_key: Option<[u8; 32]>,
    pub(crate) audio_stream_external_queue_capacity: usize,
    pub(crate) audio_stream_engine_queue_capacity: Option<usize>,
    /// Input text (for TTS)
    pub text: Option<String>,
    /// Chat input messages.
    pub chat_messages: Option<Vec<ChatMessage>>,
    /// Chat-specific prompt/runtime controls.
    pub chat_config: ChatRequestConfig,
    /// Optional language hint for multilingual generation.
    pub language: Option<String>,
    /// Request correlation ID propagated from API/runtime boundaries.
    pub correlation_id: Option<String>,
    /// Input audio (base64 encoded, primarily for OpenAI-compatible routes).
    /// Processing may move this buffer into [`Self::task`] and clear this field.
    pub audio_input: Option<String>,
    /// Input audio bytes for first-party routes that already parsed uploads.
    /// Processing may move this buffer into [`Self::task`] and clear this field.
    pub audio_bytes: Option<Vec<u8>>,
    /// Optional ASR initial prompt/context hint.
    pub asr_prompt: Option<String>,
    /// Whether the caller left ASR max_tokens unset, so `params.max_tokens` is
    /// a default or model-specific ceiling that duration-aware families narrow.
    pub asr_auto_max_tokens: bool,
    /// The caller wants model-native word timestamps. A family that emits
    /// them (Granite Speech Plus) runs its timestamp task instead of plain
    /// ASR; the others ignore it and leave timing to the forced aligner.
    pub asr_native_word_timestamps: bool,
    /// Translation target and keyword list, for families that support them.
    pub asr_task: AsrTaskOptions,
    /// Reference audio for voice cloning (base64 encoded). Processing may move
    /// this buffer into [`Self::task`] and clear this field.
    pub reference_audio: Option<String>,
    /// Reference text for voice cloning. Processing may move this buffer into
    /// [`Self::task`] and clear this field.
    pub reference_text: Option<String>,
    /// The language the reference audio is spoken in, for families that clone
    /// across languages (Maya 2 Native); absent, the family infers it.
    pub reference_language: Option<String>,
    /// Voice description for voice design
    pub voice_description: Option<String>,
    /// Optional system prompt (e.g. speech-to-speech system instruction).
    pub system_prompt: Option<String>,
    /// Generation parameters
    pub params: GenerationParams,
    /// Request priority
    pub priority: Priority,
    /// Coarse latency/throughput class for scheduling and admission.
    pub workload_class: WorkloadClass,
    /// Time spent waiting for server-side admission before entering the runtime.
    pub admission_ms: Option<f64>,
    /// Arrival timestamp
    pub arrival_time: Instant,
    /// When the runtime coordinator admitted this request. Input
    /// preparation, model residency and lane waits run between it and engine
    /// insertion; `None` for direct engine callers, whose queue starts at
    /// insertion.
    pub runtime_admitted_at: Option<Instant>,
    /// Optional absolute end-to-end deadline supplied by the ingress layer.
    pub deadline: Option<Instant>,
    /// Prompt token IDs (set by processor)
    pub prompt_tokens: Vec<TokenId>,
    /// Private authorization installed only after exact model prompt preparation.
    pub(super) chat_execution_ready: Option<ChatExecutionReady>,
    /// Exact ASR/Qwen-TTS model identity retained for standalone incremental
    /// execution. Public fields cannot manufacture this lifecycle fence.
    pub(super) incremental_model_execution_ready: Option<IncrementalModelExecutionReady>,
    /// Engine-owned, session-fenced realtime operation payloads. The immutable
    /// request owns the mailbox; only Engine ingress can install/remove rows.
    pub(super) realtime_asr_ingress: Option<Arc<RealtimeAsrIngress>>,
    /// Enable streaming output
    pub streaming: bool,
    /// Backpressure behavior for streaming output.
    pub stream_policy: EngineStreamPolicy,
    /// Channel for streaming output (internal use)
    pub(crate) streaming_tx: Option<mpsc::Sender<StreamingOutput>>,
    /// Cooperative cancellation signal set without waiting for the core write lock.
    pub(crate) cancellation: Option<Arc<AtomicBool>>,
}

impl EngineCoreRequest {
    pub(crate) fn set_cancellation_signal(&mut self, signal: Arc<AtomicBool>) {
        self.cancellation = Some(signal);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|signal| signal.load(Ordering::Acquire))
    }

    /// Get number of prompt tokens.
    pub fn num_prompt_tokens(&self) -> usize {
        if let Some(input_tokens) = self.prepared_sequence_input_tokens {
            input_tokens
        } else if !self.prompt_tokens.is_empty() {
            self.prompt_tokens.len()
        } else if let Some(prompt) = self.asr_prompt_for_execution() {
            (prompt.len() / 4).max(1)
        } else if let Some(messages) = match self.task_type {
            TaskType::SpeechToSpeech => Some(self.speech_messages_for_execution()),
            _ => self.chat_messages.as_deref(),
        } {
            (messages
                .iter()
                .map(|message| message.content.len())
                .sum::<usize>()
                / 4)
            .max(1)
        } else {
            // Estimate from text length (rough approximation)
            self.text.as_ref().map(|t| t.len() / 4).unwrap_or(0).max(1)
        }
    }

    /// Time since request arrival.
    pub fn waiting_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// Request processor - validates and preprocesses requests.
pub struct RequestProcessor {
    config: EngineCoreConfig,
}

impl RequestProcessor {
    /// Create a new request processor.
    pub fn new(config: EngineCoreConfig) -> Self {
        Self { config }
    }

    /// Process and validate a request.
    pub fn process(&self, mut request: EngineCoreRequest) -> Result<EngineCoreRequest> {
        // Standalone RequestProcessor callers receive the same preflight as
        // Engine admission. This is idempotent when Engine already moved the
        // compatibility payload into its canonical typed owner.
        request.canonicalize_direct_payloads(self.config.max_seq_len)?;
        self.process_canonicalized(request)
    }

    /// Complete request validation after Engine admission performed the direct
    /// payload preflight and canonicalization on its bounded blocking pool.
    pub(crate) fn process_canonicalized(
        &self,
        mut request: EngineCoreRequest,
    ) -> Result<EngineCoreRequest> {
        // Validate request based on task type
        match request.task_type {
            TaskType::TTS => {
                if request.text.is_none()
                    || request.text.as_ref().map(|t| t.is_empty()).unwrap_or(true)
                {
                    return Err(Error::InvalidInput("TTS request requires text".into()));
                }
            }
            TaskType::ASR => {
                if request.selected_audio_input().is_none() {
                    return Err(Error::InvalidInput(
                        "ASR request requires audio input".into(),
                    ));
                }
            }
            TaskType::Chat => {
                if request
                    .chat_messages
                    .as_ref()
                    .map(|m| m.is_empty())
                    .unwrap_or(true)
                {
                    return Err(Error::InvalidInput(
                        "Chat request requires at least one message".into(),
                    ));
                }
            }
            TaskType::SpeechToSpeech => {
                if request.selected_audio_input().is_none() {
                    return Err(Error::InvalidInput(
                        "Speech-to-speech request requires audio input".into(),
                    ));
                }
            }
        }

        request.stream_policy.validate()?;

        // Parakeet preparation already bounded max_tokens by the input's
        // emittable labels (E × max_symbols). Long audio exceeds the token
        // context, which limits nothing in Parakeet's decode, so only an
        // unprepared Parakeet request keeps the context clamp.
        let output_bound_prepared = request.task_type == TaskType::ASR
            && request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::ParakeetAsr)
            && request.prepared_sequence_input_tokens.is_some();

        // Validate and clamp parameters
        self.validate_params(
            request.task_type,
            request.model_variant,
            output_bound_prepared,
            &mut request.params,
        )?;

        // Empty prompt_tokens means the runtime only has an estimate. Never invent
        // placeholder IDs: equal-length unrelated prompts would become false prefix hits.

        request.sync_task_from_fields();
        Ok(request)
    }

    /// Validate and clamp generation parameters.
    fn validate_params(
        &self,
        task_type: TaskType,
        model_variant: Option<crate::model::ModelVariant>,
        output_bound_prepared: bool,
        params: &mut GenerationParams,
    ) -> Result<()> {
        // Clamp temperature
        params.temperature = params.temperature.clamp(0.0, 2.0);

        // Clamp top_p
        params.top_p = params.top_p.clamp(0.0, 1.0);

        // Clamp max_tokens
        if params.max_tokens == 0 && !matches!(task_type, TaskType::TTS) {
            params.max_tokens = 2048;
        }
        if params.max_tokens > 0 {
            params.max_tokens = match task_type {
                TaskType::TTS => {
                    let tts_limit = model_variant
                        .map(|variant| {
                            super::tts_explicit_output_limit(
                                self.config.backend,
                                variant,
                                self.config.max_seq_len,
                            )
                        })
                        .unwrap_or(self.config.max_seq_len);
                    params.max_tokens.min(tts_limit)
                }
                _ if output_bound_prepared => params.max_tokens,
                _ => params.max_tokens.min(self.config.max_seq_len),
            };
        }

        // Clamp speed (already applied before TTS preparation, so the
        // prepared artifact and execution see the same value).
        params.speed = clamp_tts_speed(params.speed);
        params.speech.validate()?;

        // Validate repetition penalty
        if params.repetition_penalty < 1.0 {
            params.repetition_penalty = 1.0;
        }

        // Clamp presence penalty to the OpenAI-compatible range.
        params.presence_penalty = params.presence_penalty.clamp(-2.0, 2.0);

        Ok(())
    }
}

/// Builder for creating requests with a fluent API.
#[cfg(test)]
pub struct RequestBuilder {
    request: EngineCoreRequest,
}

#[cfg(test)]
impl RequestBuilder {
    /// Create a new TTS request builder.
    #[cfg(test)]
    pub fn tts(text: impl Into<String>) -> Self {
        Self {
            request: EngineCoreRequest::tts(text),
        }
    }

    /// Set the voice.
    #[cfg(test)]
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.request.params.voice = Some(voice.into());
        self
    }

    /// Set temperature.
    #[cfg(test)]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.request.params.temperature = temp;
        self
    }

    /// Set max tokens.
    #[cfg(test)]
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.request.params.max_tokens = max;
        self
    }

    /// Set workload class.
    #[cfg(test)]
    pub fn workload_class(mut self, workload_class: WorkloadClass) -> Self {
        self.request.workload_class = workload_class;
        self
    }

    /// Enable streaming.
    #[cfg(test)]
    pub fn streaming(mut self) -> Self {
        self.request.streaming = true;
        if self.request.workload_class == WorkloadClass::Online {
            self.request.workload_class = WorkloadClass::Streaming;
        }
        self
    }

    /// Build the request.
    #[cfg(test)]
    pub fn build(mut self) -> EngineCoreRequest {
        self.request.sync_task_from_fields();
        self.request
    }
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;

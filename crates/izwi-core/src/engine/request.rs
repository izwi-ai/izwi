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
use crate::models::architectures::fish_s2::{FishS2GenerationParams, FishS2PreparedArtifact};
use crate::models::architectures::granite_speech::asr::GraniteSpeechPreparedPromptArtifact;
use crate::models::architectures::kokoro::KokoroPreparedRequest;
use crate::models::architectures::lfm25_audio::{
    lfm25_audio_tts_system_prompt, model::Lfm25AudioPreparedAsrArtifact,
    tts_retained::Lfm25AudioPreparedTtsArtifact, Lfm25AudioGenerationConfig,
};
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
    AsrModelLease, ChatModelLease, FishS2TtsModelLease, KokoroTtsModelLease, Lfm25AudioModelLease,
    NativeAsrModel, NativeChatModel, NativeChatPreparedPrompt, QwenTtsModelLease,
    VibeVoiceTtsModelLease, VoxtralTtsModelLease,
};
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig, ChatRole};
use crate::runtime::audio_io::{
    validate_base64_audio_retained_size, validate_base64_audio_source_input,
    MAX_AUDIO_SOURCE_BYTES, MAX_REFERENCE_SOURCE_BYTES,
};

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
    kokoro_tts: Option<Arc<KokoroPreparedText>>,
}

/// Speech-rate range every TTS request is clamped to. Clients may send
/// OpenAI's 0.25–4.0; the engine supports 0.5–2.0 (as `izwi tts --speed`
/// documents).
pub(crate) const TTS_SPEED_RANGE: (f32, f32) = crate::catalog::SpeechModelCapabilities::SPEED_RANGE;

pub(crate) fn clamp_tts_speed(speed: f32) -> f32 {
    speed.clamp(TTS_SPEED_RANGE.0, TTS_SPEED_RANGE.1)
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

impl StreamProgressBudget {
    pub(super) fn new(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            max_bytes: max_bytes.max(1),
            state: Mutex::new(0),
            available: Condvar::new(),
        })
    }

    fn reserve(
        self: &Arc<Self>,
        bytes: usize,
        policy: EngineStreamPolicy,
    ) -> Result<Option<StreamProgressPermit>> {
        if bytes > STREAM_PROGRESS_MAX_EVENT_BYTES || bytes > self.max_bytes {
            return Err(Error::Overloaded(format!(
                "stream progress event requires {bytes} bytes, exceeding its bounded budget"
            )));
        }

        let mut used = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream progress budget mutex poisoned".into()))?;
        let admits = |used: usize| {
            used.checked_add(bytes)
                .is_some_and(|next| next <= self.max_bytes)
        };

        match policy {
            EngineStreamPolicy::FailOnFull => {
                if !admits(*used) {
                    record_engine_stream_backpressure();
                    return Err(Error::Overloaded(
                        "stream progress byte budget is full".to_string(),
                    ));
                }
            }
            EngineStreamPolicy::DropNewest => {
                if !admits(*used) {
                    record_engine_stream_backpressure();
                    return Ok(None);
                }
            }
            EngineStreamPolicy::BlockWithDeadline { timeout_ms } => {
                let deadline = checked_stream_backpressure_deadline(timeout_ms)?;
                while !admits(*used) {
                    let now = Instant::now();
                    if now >= deadline {
                        record_engine_stream_backpressure();
                        return Err(Error::Overloaded(
                            "stream progress byte-budget deadline elapsed".to_string(),
                        ));
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next, result) = self.available.wait_timeout(used, wait).map_err(|_| {
                        Error::InferenceError("stream progress budget mutex poisoned".into())
                    })?;
                    used = next;
                    if result.timed_out() && !admits(*used) {
                        record_engine_stream_backpressure();
                        return Err(Error::Overloaded(
                            "stream progress byte-budget deadline elapsed".to_string(),
                        ));
                    }
                }
            }
        }

        *used = used.checked_add(bytes).ok_or_else(|| {
            Error::Overloaded("stream progress byte accounting overflowed".to_string())
        })?;
        Ok(Some(StreamProgressPermit {
            budget: self.clone(),
            bytes,
        }))
    }
}

#[derive(Debug)]
pub(super) struct StreamProgressPermit {
    budget: Arc<StreamProgressBudget>,
    bytes: usize,
}

impl Drop for StreamProgressPermit {
    fn drop(&mut self) {
        let Ok(mut used) = self.budget.state.lock() else {
            return;
        };
        *used = used.saturating_sub(self.bytes);
        self.budget.available.notify_all();
    }
}

/// Candidate output credit reserved before codec materialization. The staging
/// outbox and caller share ownership until commit transfers it to delivery.
#[derive(Debug, Clone)]
pub(crate) struct AudioOutputCredit {
    _permits: Arc<Vec<StreamProgressPermit>>,
    bytes: usize,
}

fn committed_audio_budget() -> &'static Arc<StreamProgressBudget> {
    static BUDGET: OnceLock<Arc<StreamProgressBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| StreamProgressBudget::new(64 * 1024 * 1024))
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

impl IncrementalStreamBinding {
    fn send(
        &self,
        output: StreamingOutput,
        policy: EngineStreamPolicy,
    ) -> Result<StreamPushOutcome> {
        let payload_bytes = stream_payload_bytes(&output)?;
        let Some(budget_permit) = self.budget.reserve(payload_bytes, policy)? else {
            return Ok(StreamPushOutcome::Dropped);
        };
        let mut progress = FencedStreamProgress {
            batch_id: self.batch_id,
            lane: self.lane.clone(),
            plan_id: self.plan_id,
            session: self.session.clone(),
            output,
            budget_permit,
        };

        match policy {
            EngineStreamPolicy::FailOnFull => self
                .progress_tx
                .try_send(progress)
                .map(|_| StreamPushOutcome::Accepted)
                .map_err(stream_progress_send_error),
            EngineStreamPolicy::DropNewest => match self.progress_tx.try_send(progress) {
                Ok(()) => Ok(StreamPushOutcome::Accepted),
                Err(mpsc::error::TrySendError::Closed(progress)) => Err(
                    stream_progress_send_error(mpsc::error::TrySendError::Closed(progress)),
                ),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    record_engine_stream_backpressure();
                    Ok(StreamPushOutcome::Dropped)
                }
            },
            EngineStreamPolicy::BlockWithDeadline { timeout_ms } => {
                let deadline = checked_stream_backpressure_deadline(timeout_ms)?;
                loop {
                    match self.progress_tx.try_send(progress) {
                        Ok(()) => return Ok(StreamPushOutcome::Accepted),
                        Err(mpsc::error::TrySendError::Closed(progress)) => {
                            return Err(stream_progress_send_error(
                                mpsc::error::TrySendError::Closed(progress),
                            ));
                        }
                        Err(mpsc::error::TrySendError::Full(returned)) => {
                            progress = returned;
                            let now = Instant::now();
                            if now >= deadline {
                                record_engine_stream_backpressure();
                                return Err(Error::Overloaded(
                                    "stream progress queue deadline elapsed".to_string(),
                                ));
                            }
                            std::thread::sleep(
                                deadline
                                    .saturating_duration_since(now)
                                    .min(Duration::from_millis(1)),
                            );
                        }
                    }
                }
            }
        }
    }
}

fn stream_progress_send_error(err: mpsc::error::TrySendError<FencedStreamProgress>) -> Error {
    match err {
        mpsc::error::TrySendError::Closed(_) => {
            Error::InferenceError("stream progress channel closed".to_string())
        }
        mpsc::error::TrySendError::Full(_) => {
            record_engine_stream_backpressure();
            Error::Overloaded("stream progress queue is full".to_string())
        }
    }
}

fn stream_payload_bytes(output: &StreamingOutput) -> Result<usize> {
    let sample_bytes = output
        .samples
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Overloaded("stream sample byte size overflowed".to_string()))?;
    output
        .request_id
        .len()
        .checked_add(output.text.as_ref().map_or(0, String::len))
        .and_then(|bytes| bytes.checked_add(sample_bytes))
        .and_then(|bytes| bytes.checked_add(256))
        .ok_or_else(|| Error::Overloaded("stream payload byte size overflowed".to_string()))
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

impl StreamStagingBuffer {
    #[cfg(test)]
    pub(super) fn clear(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        state.staged.clear();
        state.audio_credits.clear();
        state.staged_bytes = 0;
        state.binding = None;
        Ok(())
    }

    pub(super) fn bind_quantum(
        &self,
        batch_id: BatchId,
        lane: BatchLaneKey,
        plan_id: PlanId,
        session: SessionKey,
        visibility: OutputVisibility,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        budget: Arc<StreamProgressBudget>,
    ) -> Result<StreamBindingGuard> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        if state.binding.is_some() {
            return Err(Error::InferenceError(
                "stream progress binding is already active".to_string(),
            ));
        }
        state.staged.clear();
        state.audio_credits.clear();
        state.staged_bytes = 0;
        if visibility == OutputVisibility::IncrementalCommitted {
            state.binding = Some(IncrementalStreamBinding {
                batch_id,
                lane,
                plan_id,
                session: session.clone(),
                progress_tx,
                budget,
            });
        }
        Ok(StreamBindingGuard {
            buffer: self.clone(),
            plan_id,
            session,
        })
    }

    pub(super) fn push_with_policy(
        &self,
        output: StreamingOutput,
        policy: EngineStreamPolicy,
    ) -> Result<StreamPushOutcome> {
        let binding = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
            if output.is_final || state.binding.is_none() {
                let payload_bytes = stream_payload_bytes(&output)?;
                let next_bytes =
                    state
                        .staged_bytes
                        .checked_add(payload_bytes)
                        .ok_or_else(|| {
                            Error::Overloaded("stream staging size overflowed".to_string())
                        })?;
                if state.staged.len() >= STREAM_STAGED_MAX_EVENTS
                    || next_bytes > STREAM_PROGRESS_MAX_BUFFERED_BYTES
                {
                    record_engine_stream_backpressure();
                    if policy == EngineStreamPolicy::DropNewest && !output.is_final {
                        return Ok(StreamPushOutcome::Dropped);
                    }
                    return Err(Error::Overloaded(
                        "stream staging outbox exceeded its bounded capacity".to_string(),
                    ));
                }
                if !state.audio_credits.is_empty()
                    && next_bytes
                        > state
                            .audio_credits
                            .iter()
                            .map(|credit| credit.bytes)
                            .sum::<usize>()
                {
                    return Err(Error::Overloaded(
                        "Audio output exceeded its pre-dispatch credit".into(),
                    ));
                }
                state.staged.push(output);
                state.staged_bytes = next_bytes;
                return Ok(StreamPushOutcome::Accepted);
            }
            state.binding.clone().expect("binding checked above")
        };
        binding.send(output, policy)
    }

    pub(super) fn take(&self) -> Result<Vec<StreamingOutput>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        state.staged_bytes = 0;
        Ok(std::mem::take(&mut state.staged))
    }

    #[cfg(test)]
    pub(super) fn has_incremental_binding(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.binding.is_some())
            .unwrap_or(false)
    }

    fn clear_binding(&self, plan_id: PlanId, session: &SessionKey) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state
            .binding
            .as_ref()
            .is_some_and(|binding| binding.plan_id == plan_id && &binding.session == session)
        {
            state.binding = None;
        }
    }
}

impl Drop for StreamBindingGuard {
    fn drop(&mut self) {
        self.buffer.clear_binding(self.plan_id, &self.session);
    }
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
    /// Reference audio for voice cloning (base64 encoded). Processing may move
    /// this buffer into [`Self::task`] and clear this field.
    pub reference_audio: Option<String>,
    /// Reference text for voice cloning. Processing may move this buffer into
    /// [`Self::task`] and clear this field.
    pub reference_text: Option<String>,
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
    pub(crate) fn enable_realtime_asr_ingress(&mut self) -> Result<()> {
        if self.task_type != TaskType::ASR || self.realtime_asr_ingress.is_some() {
            return Err(Error::InvalidInput(format!(
                "request {} cannot enable realtime ASR ingress",
                self.id
            )));
        }
        self.streaming = true;
        self.realtime_asr_ingress = Some(Arc::new(RealtimeAsrIngress::default()));
        Ok(())
    }

    pub(crate) fn is_realtime_asr_session(&self) -> bool {
        self.realtime_asr_ingress.is_some()
    }

    #[cfg(test)]
    pub(crate) fn install_realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
        payload: RealtimeAsrOperationPayload,
    ) -> Result<RealtimeAsrOperationWaiter> {
        let logical_units = match &payload {
            RealtimeAsrOperationPayload::Push { samples, .. } => {
                u64::try_from(samples.len()).unwrap_or(u64::MAX).max(1)
            }
            RealtimeAsrOperationPayload::Finish => 1,
        };
        self.install_realtime_asr_operation_with_cost(
            operation_id,
            payload,
            WorkCost::new(logical_units, logical_units, 0),
        )
    }

    pub(crate) fn install_realtime_asr_operation_with_cost(
        &self,
        operation_id: RealtimeOperationId,
        payload: RealtimeAsrOperationPayload,
        preparation_cost: WorkCost,
    ) -> Result<RealtimeAsrOperationWaiter> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let mut state = ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.terminal.is_some() {
            return Err(Error::InferenceError(
                "realtime ASR session is already terminal".into(),
            ));
        }
        if state
            .highest_operation_id
            .is_some_and(|highest| operation_id <= highest)
        {
            return Err(Error::InferenceError(format!(
                "realtime ASR operation identity {} is duplicate or stale",
                operation_id.get()
            )));
        }
        let (completion, waiter) = oneshot::channel();
        match state.operations.entry(operation_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(RealtimeAsrIngressOperation {
                    payload,
                    preparation_cost,
                    completion,
                });
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(Error::InferenceError(
                    "realtime ASR operation identity was installed twice".into(),
                ));
            }
        }
        state.highest_operation_id = Some(operation_id);
        Ok(waiter)
    }

    pub(crate) fn realtime_asr_preparation_cost(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<WorkCost> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .get(&operation_id)
            .map(|operation| operation.preparation_cost)
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} has no retained preparation cost",
                    operation_id.get()
                ))
            })
    }

    pub(crate) fn realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<RealtimeAsrOperationPayload> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .get(&operation_id)
            .map(|operation| operation.payload.clone())
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} has no retained payload",
                    operation_id.get()
                ))
            })
    }

    /// Atomically retires one payload before publishing its acknowledgement.
    /// A repeated or unknown identity is rejected and cannot acknowledge a
    /// later operation that happens to have the same payload shape.
    pub(super) fn complete_realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<()> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let operation = ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .remove(&operation_id)
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} is unknown, stale, or already completed",
                    operation_id.get()
                ))
            })?;
        let acknowledgement = RealtimeAsrOperationAck {
            operation_id,
            kind: operation.payload.kind(),
            accepted_samples: operation.payload.accepted_samples(),
        };
        // Sending may fail when an external waiter was cancelled. The payload
        // is still authoritatively retired and must not be resurrected.
        let _ = operation.completion.send(Ok(acknowledgement));
        Ok(())
    }

    /// Fences future installation, retires every retained payload, and wakes
    /// all operation waiters with the same authoritative terminal outcome.
    pub(super) fn terminate_realtime_asr_ingress(
        &self,
        terminal: RealtimeAsrTerminalOutcome,
    ) -> Result<usize> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let operations = {
            let mut state = ingress
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if state.terminal.is_some() {
                return Ok(0);
            }
            state.terminal = Some(terminal.clone());
            std::mem::take(&mut state.operations)
        };
        let retired = operations.len();
        for (_, operation) in operations {
            let _ = operation.completion.send(Err(terminal.clone()));
        }
        Ok(retired)
    }

    fn sync_task_from_fields(&mut self) {
        if self.task_type == TaskType::Chat {
            self.task = EngineTask::Chat(ChatEngineInput {
                messages: self.chat_messages.clone().unwrap_or_default(),
                chat_config: self.chat_config.clone(),
                prompt_tokens: self.prompt_tokens.clone(),
            });
        }
    }

    fn empty_audio_for_compatibility_fields(
        audio_input: Option<&String>,
        audio_bytes: Option<&Vec<u8>>,
    ) -> EngineAudioInput {
        if audio_bytes.is_some() {
            EngineAudioInput::Bytes(Vec::new())
        } else if audio_input.is_some() {
            EngineAudioInput::Base64(String::new())
        } else {
            EngineAudioInput::Bytes(Vec::new())
        }
    }

    fn selected_audio_input(&self) -> Option<BorrowedEngineAudioInput<'_>> {
        if let Some(audio) = self.audio_bytes.as_ref() {
            return Some(BorrowedEngineAudioInput::Bytes(audio));
        }
        if let Some(audio) = self.audio_input.as_ref() {
            return Some(BorrowedEngineAudioInput::Base64(audio));
        }
        let audio = match (&self.task, self.task_type) {
            (EngineTask::Asr(input), TaskType::ASR) => &input.audio,
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => &input.audio,
            _ => return None,
        };
        (!audio.is_empty()).then_some(match audio {
            EngineAudioInput::Base64(value) => BorrowedEngineAudioInput::Base64(value),
            EngineAudioInput::Bytes(value) => BorrowedEngineAudioInput::Bytes(value),
        })
    }

    fn selected_tts_reference(&self) -> (Option<&String>, Option<&String>) {
        if self.reference_audio.is_some() || self.reference_text.is_some() {
            return (self.reference_audio.as_ref(), self.reference_text.as_ref());
        }
        match (&self.task, self.task_type) {
            (EngineTask::Tts(input), TaskType::TTS) => (
                input.reference_audio.as_ref(),
                input.reference_text.as_ref(),
            ),
            _ => (None, None),
        }
    }

    fn selected_asr_language(&self) -> Option<&String> {
        self.language
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::Asr(input), TaskType::ASR) => input.language.as_ref(),
                _ => None,
            })
    }

    fn selected_asr_prompt(&self) -> Option<&String> {
        self.asr_prompt
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::Asr(input), TaskType::ASR) => input.prompt.as_ref(),
                _ => None,
            })
    }

    fn selected_speech_messages(&self) -> &[ChatMessage] {
        if let Some(messages) = self.chat_messages.as_deref() {
            return messages;
        }
        match (&self.task, self.task_type) {
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => &input.messages,
            _ => &[],
        }
    }

    fn selected_speech_messages_capacity(&self) -> usize {
        if let Some(messages) = self.chat_messages.as_ref() {
            return messages.capacity();
        }
        match (&self.task, self.task_type) {
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                input.messages.capacity()
            }
            _ => 0,
        }
    }

    fn selected_speech_system_prompt(&self) -> Option<&String> {
        self.system_prompt
            .as_ref()
            .or(match (&self.task, self.task_type) {
                (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                    input.system_prompt.as_ref()
                }
                _ => None,
            })
    }

    pub(crate) fn audio_bytes_for_execution(&self) -> Option<&[u8]> {
        match self.selected_audio_input()? {
            BorrowedEngineAudioInput::Bytes(value) => Some(value.as_slice()),
            BorrowedEngineAudioInput::Base64(_) => None,
        }
    }

    pub(crate) fn audio_base64_for_execution(&self) -> Option<&str> {
        match self.selected_audio_input()? {
            BorrowedEngineAudioInput::Base64(value) => Some(value.as_str()),
            BorrowedEngineAudioInput::Bytes(_) => None,
        }
    }

    pub(crate) fn tts_reference_audio_for_execution(&self) -> Option<&str> {
        self.selected_tts_reference().0.map(String::as_str)
    }

    pub(crate) fn tts_text_for_execution(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub(crate) fn tts_speaker_for_execution(&self) -> Option<&str> {
        self.params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref())
    }

    pub(crate) fn tts_reference_text_for_execution(&self) -> Option<&str> {
        self.selected_tts_reference().1.map(String::as_str)
    }

    pub(crate) fn has_tts_reference_for_execution(&self) -> bool {
        let (audio, text) = self.selected_tts_reference();
        audio.is_some() || text.is_some()
    }

    pub(crate) fn asr_language_for_execution(&self) -> Option<&str> {
        self.selected_asr_language().map(String::as_str)
    }

    pub(crate) fn asr_prompt_for_execution(&self) -> Option<&str> {
        self.selected_asr_prompt().map(String::as_str)
    }

    /// Identity of an ASR request's routed source: model, encoded upload,
    /// language, prompt and timestamp option. Prepared artifacts record it
    /// and every executor accessor re-checks it. Hashing a large upload is
    /// the expensive part, so accessors reuse the digest that preparation
    /// computed while the same upload buffer is in place, and hash in full
    /// only when the buffer was replaced (or on a clone of the request).
    fn asr_execution_source_fingerprint(&self, model_variant: ModelVariant) -> Result<u64> {
        let audio = self.asr_execution_source_audio(model_variant)?;
        let identity = AsrSourceAudioIdentity::of(audio);
        let digest = match self.asr_source_audio_digest {
            Some(cached) if cached.identity == identity => cached.digest,
            _ => asr_source_audio_digest(audio),
        };
        Ok(self.asr_execution_source_fingerprint_with(model_variant, digest))
    }

    /// Hash the upload in full, cache the digest for the accessors, and
    /// return the source fingerprint. Every preparation install calls this,
    /// so an artifact always records the fingerprint of the bytes as they
    /// were when it was prepared.
    fn refresh_asr_execution_source_fingerprint(
        &mut self,
        model_variant: ModelVariant,
    ) -> Result<u64> {
        let audio = self.asr_execution_source_audio(model_variant)?;
        let cached = AsrSourceAudioDigest {
            identity: AsrSourceAudioIdentity::of(audio),
            digest: asr_source_audio_digest(audio),
        };
        self.asr_source_audio_digest = Some(cached);
        Ok(self.asr_execution_source_fingerprint_with(model_variant, cached.digest))
    }

    fn asr_execution_source_audio(
        &self,
        model_variant: ModelVariant,
    ) -> Result<BorrowedEngineAudioInput<'_>> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} source identity does not match its routed task/model",
                self.id
            )));
        }
        self.selected_audio_input().ok_or_else(|| {
            Error::InvalidInput(format!("ASR request {} is missing audio input", self.id))
        })
    }

    fn asr_execution_source_fingerprint_with(
        &self,
        model_variant: ModelVariant,
        audio_digest: u64,
    ) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model_variant.hash(&mut hasher);
        audio_digest.hash(&mut hasher);
        self.asr_language_for_execution().hash(&mut hasher);
        self.asr_prompt_for_execution().hash(&mut hasher);
        self.asr_native_word_timestamps.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn speech_messages_for_execution(&self) -> &[ChatMessage] {
        self.selected_speech_messages()
    }

    pub(crate) fn speech_system_prompt_for_execution(&self) -> Option<&str> {
        self.selected_speech_system_prompt().map(String::as_str)
    }

    /// Apply backend-independent source limits and collapse compatibility and
    /// typed payload views into one owner before request processing can clone
    /// the request or an executor can decode it.
    pub(crate) fn canonicalize_direct_payloads(&mut self, max_seq_len: usize) -> Result<()> {
        self.canonicalize_direct_payloads_with_limits(DirectPayloadLimits::production(max_seq_len)?)
    }

    fn canonicalize_direct_payloads_with_limits(
        &mut self,
        limits: DirectPayloadLimits,
    ) -> Result<()> {
        let dynamic_metadata_bytes = self.validate_direct_dynamic_allocations(limits)?;
        match self.task_type {
            TaskType::ASR => {
                let language = self.selected_asr_language();
                let prompt = self.selected_asr_prompt();
                Self::validate_direct_metadata(
                    "ASR",
                    dynamic_metadata_bytes,
                    [language, prompt]
                        .into_iter()
                        .flatten()
                        .map(String::capacity),
                    limits.metadata_bytes,
                )?;
                if let Some(audio) = self.selected_audio_input() {
                    Self::validate_direct_audio(audio, limits.audio_source_bytes, "input audio")?;
                }
                let already_canonical = self.audio_input.is_none()
                    && self.audio_bytes.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::Asr(input)
                            if !input.audio.is_empty()
                                && input.language.as_deref() == language.map(String::as_str)
                                && input.prompt.as_deref() == prompt.map(String::as_str)
                    );
                if !already_canonical {
                    let language = language.map(String::as_str).map(str::to_owned);
                    let prompt = prompt.map(String::as_str).map(str::to_owned);
                    let audio = if self.selected_audio_input().is_some() {
                        self.take_selected_audio_input()
                            .expect("validated selected audio must remain available")
                    } else {
                        Self::empty_audio_for_compatibility_fields(
                            self.audio_input.as_ref(),
                            self.audio_bytes.as_ref(),
                        )
                    };
                    self.task = EngineTask::Asr(AsrEngineInput {
                        audio,
                        language,
                        prompt,
                    });
                }
            }
            TaskType::SpeechToSpeech => {
                let messages = self.selected_speech_messages();
                let system_prompt = self.selected_speech_system_prompt();
                if messages.len() > limits.collection_items {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine speech-to-speech request contains {} messages, exceeding the {}-item context-derived limit",
                        messages.len(), limits.collection_items
                    )));
                }
                let message_storage = self
                    .selected_speech_messages_capacity()
                    .checked_mul(std::mem::size_of::<ChatMessage>())
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "Direct Engine speech-to-speech metadata size overflow".to_string(),
                        )
                    })?;
                Self::validate_direct_metadata(
                    "speech-to-speech",
                    dynamic_metadata_bytes,
                    std::iter::once(message_storage)
                        .chain(messages.iter().map(|message| message.content.capacity()))
                        .chain(system_prompt.map(String::capacity)),
                    limits.metadata_bytes,
                )?;
                if let Some(audio) = self.selected_audio_input() {
                    Self::validate_direct_audio(audio, limits.audio_source_bytes, "input audio")?;
                }
                let already_canonical = self.audio_input.is_none()
                    && self.audio_bytes.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::SpeechToSpeech(input)
                            if !input.audio.is_empty()
                                && Self::chat_messages_match(&input.messages, messages)
                                && input.system_prompt.as_deref()
                                    == system_prompt.map(String::as_str)
                    );
                if !already_canonical {
                    let canonical_messages = messages.to_vec();
                    let system_prompt = system_prompt.map(String::as_str).map(str::to_owned);
                    let audio = if self.selected_audio_input().is_some() {
                        self.take_selected_audio_input()
                            .expect("validated selected audio must remain available")
                    } else {
                        Self::empty_audio_for_compatibility_fields(
                            self.audio_input.as_ref(),
                            self.audio_bytes.as_ref(),
                        )
                    };
                    self.task = EngineTask::SpeechToSpeech(AudioChatEngineInput {
                        audio,
                        messages: canonical_messages,
                        system_prompt,
                    });
                }
            }
            TaskType::TTS => {
                let (reference_audio, reference_text) = self.selected_tts_reference();
                Self::validate_direct_metadata(
                    "TTS",
                    dynamic_metadata_bytes,
                    [
                        self.text.as_ref(),
                        reference_text,
                        self.voice_description.as_ref(),
                        self.language.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(String::capacity),
                    limits.metadata_bytes,
                )?;
                if reference_audio.is_some() || reference_text.is_some() {
                    let reference_audio = reference_audio.ok_or_else(|| {
                        Error::InvalidInput(
                            "reference_audio and reference_text must both be provided".to_string(),
                        )
                    })?;
                    let reference_text = reference_text.ok_or_else(|| {
                        Error::InvalidInput(
                            "reference_audio and reference_text must both be provided".to_string(),
                        )
                    })?;
                    if reference_text.trim().is_empty() {
                        return Err(Error::InvalidInput(
                            "reference_text cannot be empty".to_string(),
                        ));
                    }
                    validate_base64_audio_retained_size(
                        reference_audio.capacity(),
                        limits.reference_source_bytes,
                    )?;
                    Self::validate_direct_base64(
                        reference_audio,
                        limits.reference_source_bytes,
                        "reference audio",
                    )?;
                    let reference_chars = reference_text.chars().count();
                    if reference_chars > limits.reference_text_chars {
                        return Err(Error::InvalidInput(format!(
                            "TTS reference_text contains {reference_chars} characters, exceeding the {}-character direct Engine limit",
                            limits.reference_text_chars
                        )));
                    }
                }
                let already_canonical = self.reference_audio.is_none()
                    && self.reference_text.is_none()
                    && matches!(
                        &self.task,
                        EngineTask::Tts(input)
                            if input.text == self.text.as_deref().unwrap_or_default()
                                && input.voice_description == self.voice_description
                    );
                if !already_canonical {
                    let (reference_audio, reference_text) = self.take_selected_tts_reference();
                    self.task = EngineTask::Tts(TtsEngineInput {
                        text: self.text.clone().unwrap_or_default(),
                        reference_audio,
                        reference_text,
                        voice_description: self.voice_description.clone(),
                    });
                }
            }
            TaskType::Chat => {}
        }
        self.sync_task_from_fields();
        Ok(())
    }

    fn validate_direct_dynamic_allocations(&self, limits: DirectPayloadLimits) -> Result<usize> {
        for (label, count) in [
            ("stop sequences", self.params.stop_sequences.len()),
            ("stop token IDs", self.params.stop_token_ids.len()),
        ] {
            if count > limits.collection_items {
                return Err(Error::InvalidInput(format!(
                    "Direct Engine request contains {count} {label}, exceeding the {}-item context-derived limit",
                    limits.collection_items
                )));
            }
        }
        if self.task_type != TaskType::Chat && self.prompt_tokens.len() > limits.collection_items {
            return Err(Error::InvalidInput(format!(
                "Direct Engine request contains {} prompt tokens, exceeding the {}-item context-derived limit",
                self.prompt_tokens.len(),
                limits.collection_items
            )));
        }

        let stop_sequence_storage = self
            .params
            .stop_sequences
            .capacity()
            .checked_mul(std::mem::size_of::<String>())
            .ok_or_else(|| {
                Error::InvalidInput("Direct Engine stop-sequence storage size overflow".to_string())
            })?;
        let stop_token_storage = self
            .params
            .stop_token_ids
            .capacity()
            .checked_mul(std::mem::size_of::<TokenId>())
            .ok_or_else(|| {
                Error::InvalidInput("Direct Engine stop-token storage size overflow".to_string())
            })?;
        let prompt_token_storage = if self.task_type == TaskType::Chat {
            0
        } else {
            self.prompt_tokens
                .capacity()
                .checked_mul(std::mem::size_of::<TokenId>())
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Direct Engine prompt-token storage size overflow".to_string(),
                    )
                })?
        };
        let fixed = [
            self.params
                .speaker
                .as_ref()
                .map(String::capacity)
                .unwrap_or(0),
            self.params
                .voice
                .as_ref()
                .map(String::capacity)
                .unwrap_or(0),
            stop_sequence_storage,
            stop_token_storage,
            prompt_token_storage,
        ];
        Self::validate_direct_metadata(
            "request",
            0,
            fixed
                .into_iter()
                .chain(self.params.stop_sequences.iter().map(String::capacity)),
            limits.metadata_bytes,
        )
    }

    fn chat_messages_match(left: &[ChatMessage], right: &[ChatMessage]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.role == right.role && left.content == right.content)
    }

    fn validate_direct_metadata(
        task: &str,
        initial: usize,
        fields: impl IntoIterator<Item = usize>,
        limit: usize,
    ) -> Result<usize> {
        if initial > limit {
            return Err(Error::InvalidInput(format!(
                "Direct Engine {task} metadata is {initial} bytes, exceeding the {limit}-byte context-derived limit"
            )));
        }
        let mut total = initial;
        for bytes in fields {
            total = total.checked_add(bytes).ok_or_else(|| {
                Error::InvalidInput(format!("Direct Engine {task} metadata size overflow"))
            })?;
            if total > limit {
                return Err(Error::InvalidInput(format!(
                    "Direct Engine {task} metadata is {total} bytes, exceeding the {limit}-byte context-derived limit"
                )));
            }
        }
        Ok(total)
    }

    fn validate_direct_audio(
        audio: BorrowedEngineAudioInput<'_>,
        max_source_bytes: usize,
        label: &str,
    ) -> Result<()> {
        match audio {
            BorrowedEngineAudioInput::Base64(value) => {
                validate_base64_audio_retained_size(value.capacity(), max_source_bytes)?;
                Self::validate_direct_base64(value, max_source_bytes, label)
            }
            BorrowedEngineAudioInput::Bytes(value) => {
                if value.is_empty() {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine {label} cannot be empty"
                    )));
                }
                if value.capacity() > max_source_bytes {
                    return Err(Error::InvalidInput(format!(
                        "Direct Engine {label} retains {} bytes, exceeding the {max_source_bytes}-byte source limit",
                        value.capacity()
                    )));
                }
                Ok(())
            }
        }
    }

    fn validate_direct_base64(value: &str, max_source_bytes: usize, label: &str) -> Result<()> {
        if value.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Direct Engine {label} cannot be empty"
            )));
        }
        validate_base64_audio_source_input(value, max_source_bytes)
    }

    fn take_selected_audio_input(&mut self) -> Option<EngineAudioInput> {
        if let Some(audio) = self.audio_bytes.take() {
            self.audio_input = None;
            return Some(EngineAudioInput::Bytes(audio));
        }
        if let Some(audio) = self.audio_input.take() {
            return Some(EngineAudioInput::Base64(audio));
        }
        match (&mut self.task, self.task_type) {
            (EngineTask::Asr(input), TaskType::ASR) => Self::take_nonempty_audio(&mut input.audio),
            (EngineTask::SpeechToSpeech(input), TaskType::SpeechToSpeech) => {
                Self::take_nonempty_audio(&mut input.audio)
            }
            _ => None,
        }
    }

    fn take_nonempty_audio(audio: &mut EngineAudioInput) -> Option<EngineAudioInput> {
        if audio.is_empty() {
            return None;
        }
        Some(match audio {
            EngineAudioInput::Base64(value) => EngineAudioInput::Base64(std::mem::take(value)),
            EngineAudioInput::Bytes(value) => EngineAudioInput::Bytes(std::mem::take(value)),
        })
    }

    fn take_selected_tts_reference(&mut self) -> (Option<String>, Option<String>) {
        if self.reference_audio.is_some() || self.reference_text.is_some() {
            return (self.reference_audio.take(), self.reference_text.take());
        }
        match (&mut self.task, self.task_type) {
            (EngineTask::Tts(input), TaskType::TTS) => {
                (input.reference_audio.take(), input.reference_text.take())
            }
            _ => (None, None),
        }
    }

    pub(crate) fn has_chat_execution_preparation(&self) -> bool {
        self.chat_execution_ready.is_some()
    }

    /// Bound an unprepared direct chat request before it is cloned or passed
    /// to a tokenizer. Runtime-prepared requests are resource-admitted before
    /// this boundary; public `Engine` callers otherwise have no admission
    /// lease protecting the blocking tokenizer pool from oversized payloads.
    pub(crate) fn validate_direct_chat_preparation_input(&self, max_seq_len: usize) -> Result<()> {
        if self.task_type != TaskType::Chat || self.has_chat_execution_preparation() {
            return Ok(());
        }
        if !self.chat_config.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Direct Engine multimodal chat is not supported; use RuntimeService so media preparation is resource-admitted"
                    .to_string(),
            ));
        }

        let context_limit = max_seq_len.max(1);
        let input_limit = context_limit
            .saturating_mul(DIRECT_CHAT_INPUT_BYTES_PER_CONTEXT_TOKEN)
            .clamp(DIRECT_CHAT_MIN_INPUT_BYTES, DIRECT_CHAT_MAX_INPUT_BYTES);
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        if messages.len() > context_limit {
            return Err(Error::InvalidInput(format!(
                "Direct chat request contains {} messages, exceeding the {context_limit}-token context guard",
                messages.len()
            )));
        }

        let mut total = 0usize;
        for message in messages {
            add_bounded_chat_input_bytes(
                &mut total,
                message.content.len(),
                input_limit,
                "message",
            )?;
        }

        if self.chat_config.tools.len() > context_limit {
            return Err(Error::InvalidInput(format!(
                "Direct chat request contains {} tools, exceeding the {context_limit}-token context guard",
                self.chat_config.tools.len()
            )));
        }
        add_bounded_chat_input_bytes(
            &mut total,
            self.chat_config
                .tools
                .len()
                .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                .ok_or_else(|| {
                    Error::InvalidInput("Direct chat tool count overflow".to_string())
                })?,
            input_limit,
            "tool",
        )?;

        // Walk JSON iteratively so an adversarially nested tool schema cannot
        // overflow this thread's stack before serde reaches its own guard.
        let mut pending = self
            .chat_config
            .tools
            .iter()
            .map(|value| (value, 1usize))
            .collect::<Vec<_>>();
        while let Some((value, depth)) = pending.pop() {
            if depth > DIRECT_CHAT_MAX_TOOL_JSON_DEPTH {
                return Err(Error::InvalidInput(format!(
                    "Direct chat tool JSON exceeds the {DIRECT_CHAT_MAX_TOOL_JSON_DEPTH}-level nesting limit"
                )));
            }
            match value {
                serde_json::Value::Null | serde_json::Value::Bool(_) => {}
                serde_json::Value::Number(_) => add_bounded_chat_input_bytes(
                    &mut total,
                    DIRECT_CHAT_JSON_NODE_BYTES,
                    input_limit,
                    "tool number",
                )?,
                serde_json::Value::String(value) => add_bounded_chat_input_bytes(
                    &mut total,
                    escaped_json_string_upper_bound(value)?,
                    input_limit,
                    "tool string",
                )?,
                serde_json::Value::Array(values) => {
                    add_bounded_chat_input_bytes(
                        &mut total,
                        values
                            .len()
                            .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                            .ok_or_else(|| {
                                Error::InvalidInput(
                                    "Direct chat tool array size overflow".to_string(),
                                )
                            })?,
                        input_limit,
                        "tool array",
                    )?;
                    pending.extend(values.iter().map(|value| (value, depth + 1)));
                }
                serde_json::Value::Object(values) => {
                    add_bounded_chat_input_bytes(
                        &mut total,
                        values
                            .len()
                            .checked_mul(DIRECT_CHAT_JSON_NODE_BYTES)
                            .ok_or_else(|| {
                                Error::InvalidInput(
                                    "Direct chat tool object size overflow".to_string(),
                                )
                            })?,
                        input_limit,
                        "tool object",
                    )?;
                    for (key, value) in values {
                        add_bounded_chat_input_bytes(
                            &mut total,
                            escaped_json_string_upper_bound(key)?,
                            input_limit,
                            "tool key",
                        )?;
                        pending.push((value, depth + 1));
                    }
                }
            }
        }
        Ok(())
    }

    /// Enforce the configured context contract after exact tokenization. The
    /// prompt must leave room for at least one output token, and the requested
    /// output budget is reduced to the remaining context instead of allowing
    /// the scheduler/model cache to exceed the configured sequence length.
    pub(crate) fn enforce_chat_context_window(&mut self, max_seq_len: usize) -> Result<()> {
        if self.task_type != TaskType::Chat {
            return Ok(());
        }
        let context_limit = self
            .chat_execution_ready
            .as_ref()
            .map(|ready| ready.context_limit)
            .unwrap_or(max_seq_len);
        let prompt_tokens = self.prompt_tokens.len();
        if context_limit == 0 || prompt_tokens >= context_limit {
            return Err(Error::InvalidInput(format!(
                "Chat request {} exact prompt has {prompt_tokens} tokens and leaves no output capacity in the resolved {context_limit}-token context",
                self.id
            )));
        }
        self.params.max_tokens = self
            .params
            .max_tokens
            .max(1)
            .min(context_limit - prompt_tokens);
        Ok(())
    }

    /// Install the private scheduler/executor authorization for an exact chat
    /// prompt. Callers must supply tokens produced from this request's current
    /// messages/configuration by the loaded model adapter.
    pub(crate) fn install_chat_execution_preparation_with_model(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        model: ChatModelLease,
        context_limit: usize,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_chat_prompt,
            PreparedChatModel::Exact(model),
            context_limit,
        )
    }

    #[cfg(test)]
    pub(crate) fn install_chat_execution_preparation(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        context_limit: usize,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_chat_prompt,
            PreparedChatModel::ValidationOnly,
            context_limit,
        )
    }

    fn install_chat_execution_preparation_inner(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_chat_prompt: Option<NativeChatPreparedPrompt>,
        model: PreparedChatModel,
        context_limit: usize,
    ) -> Result<()> {
        self.chat_execution_ready = None;
        if self.task_type != TaskType::Chat {
            return Err(Error::InvalidInput(format!(
                "Request {} is not a chat request",
                self.id
            )));
        }
        if self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation model does not match its routed model",
                self.id
            )));
        }
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        if messages.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} has no messages to prepare",
                self.id
            )));
        }
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation produced no prompt tokens",
                self.id
            )));
        }
        if context_limit == 0 {
            return Err(Error::InvalidInput(format!(
                "Chat request {} preparation has a zero context limit",
                self.id
            )));
        }

        let family = model_variant.family();
        let needs_prepared_prompt =
            matches!(family, ModelFamily::Qwen35Chat | ModelFamily::Qwen38Chat);
        match prepared_chat_prompt.as_ref() {
            Some(prepared) if prepared.family() != family => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} routed a mismatched prepared prompt artifact to {model_variant}",
                    self.id
                )));
            }
            Some(prepared) if prepared.prompt_ids() != prompt_tokens => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} prepared prompt artifact does not match its exact prompt tokens",
                    self.id
                )));
            }
            None if needs_prepared_prompt => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared hybrid-Qwen prompt artifact",
                    self.id
                )));
            }
            _ => {}
        }

        if prepared_chat_prompt.is_some() {
            // The opaque artifact owns all decoded/encoded vision state needed
            // by execution. Do not retain or repeatedly fingerprint multi-MB
            // data URLs (or signed remote URLs) on the scheduler hot path.
            for media in &mut self.chat_config.media_inputs {
                media.source.clear();
                media.source.shrink_to_fit();
            }
        }

        let prepared_continuous_cost =
            Self::continuous_chat_stage_cost(self.execution_adapter_binding.as_ref(), &model)?;

        let fingerprint =
            chat_execution_fingerprint(model_variant, messages, &self.chat_config, &prompt_tokens)?;
        self.prepared_stage_costs.clear();
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        self.managed_snapshot_boundary = prepared_chat_prompt
            .as_ref()
            .and_then(NativeChatPreparedPrompt::managed_snapshot_boundary)
            .filter(|boundary| *boundary <= prompt_tokens.len())
            .and_then(|boundary| u32::try_from(boundary).ok());
        self.managed_preamble_boundary = prepared_chat_prompt
            .as_ref()
            .and_then(NativeChatPreparedPrompt::managed_preamble_boundary)
            .filter(|boundary| *boundary <= prompt_tokens.len())
            .and_then(|boundary| u32::try_from(boundary).ok());
        self.prompt_tokens = prompt_tokens;
        self.sync_task_from_fields();
        self.chat_execution_ready = Some(ChatExecutionReady {
            model_variant,
            model,
            fingerprint,
            prepared_chat_prompt,
            context_limit,
            core_validated: false,
        });
        Ok(())
    }

    fn continuous_chat_stage_cost(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &PreparedChatModel,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        #[allow(clippy::infallible_destructuring_match)]
        let model = match model {
            PreparedChatModel::Exact(model) => model,
            #[cfg(test)]
            PreparedChatModel::ValidationOnly => return Ok(None),
        };
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible chat model"
                    .to_string(),
            ));
        }
        let accelerator_bytes = model.continuous_decode_batch_workspace_per_row_bytes()?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            super::continuous_chat_workspace_per_row(accelerator_bytes)?,
        );
        let required_bytes = cost.workspace.workspace_bytes()?;
        if required_bytes > stage.max_workspace_bytes {
            return Err(Error::Overloaded(format!(
                "continuous decode workspace exceeds its loaded adapter budget (required {} bytes, available {} bytes)",
                required_bytes, stage.max_workspace_bytes,
            )));
        }
        Ok(Some((stage.id, cost)))
    }

    /// Validate the private preparation marker and both legacy/typed payload
    /// views before any scheduler or executor trusts prompt-token accounting.
    pub(crate) fn validate_chat_execution_preparation(&self) -> Result<()> {
        if self.task_type != TaskType::Chat {
            return Err(Error::InvalidInput(format!(
                "Request {} carries chat preparation for a non-chat task",
                self.id
            )));
        }
        let ready = self.chat_execution_ready.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chat request {} is missing exact model prompt preparation",
                self.id
            ))
        })?;
        let model_variant = self.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chat request {} is missing its routed model",
                self.id
            ))
        })?;
        if ready.model_variant != model_variant {
            return Err(Error::InvalidInput(format!(
                "Chat request {} changed model after prompt preparation",
                self.id
            )));
        }
        let messages = self.chat_messages.as_deref().ok_or_else(|| {
            Error::InvalidInput(format!("Chat request {} is missing messages", self.id))
        })?;
        let field_fingerprint = chat_execution_fingerprint(
            model_variant,
            messages,
            &self.chat_config,
            &self.prompt_tokens,
        )?;
        if field_fingerprint != ready.fingerprint {
            return Err(Error::InvalidInput(format!(
                "Chat request {} changed after exact prompt preparation",
                self.id
            )));
        }

        let EngineTask::Chat(input) = &self.task else {
            return Err(Error::InvalidInput(format!(
                "Chat request {} has a mismatched typed task payload",
                self.id
            )));
        };
        let task_fingerprint = chat_execution_fingerprint(
            model_variant,
            &input.messages,
            &input.chat_config,
            &input.prompt_tokens,
        )?;
        if task_fingerprint != ready.fingerprint {
            return Err(Error::InvalidInput(format!(
                "Chat request {} typed payload does not match its exact prompt preparation",
                self.id
            )));
        }

        match ready.prepared_chat_prompt.as_ref() {
            Some(prepared)
                if prepared.family() != model_variant.family()
                    || prepared.prompt_ids() != self.prompt_tokens =>
            {
                Err(Error::InvalidInput(format!(
                    "Chat request {} has a mismatched prepared prompt artifact",
                    self.id
                )))
            }
            None if matches!(
                model_variant.family(),
                ModelFamily::Qwen35Chat | ModelFamily::Qwen38Chat
            ) =>
            {
                Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared hybrid-Qwen prompt artifact",
                    self.id
                )))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn validate_execution_preparation(&self) -> Result<()> {
        let chat_result = if self.task_type == TaskType::Chat {
            self.validate_chat_execution_preparation()
        } else if self.chat_execution_ready.is_some() {
            Err(Error::InvalidInput(format!(
                "Non-chat request {} carries chat execution preparation",
                self.id
            )))
        } else {
            Ok(())
        };
        chat_result?;
        if self.task_type == TaskType::Chat && self.incremental_model_execution_ready.is_some() {
            return Err(Error::InvalidInput(format!(
                "Chat request {} carries non-chat incremental model preparation",
                self.id
            )));
        }
        self.validate_incremental_model_execution_preparation()?;
        self.validate_prepared_stage_costs()
    }

    /// Validate once at the mutable core admission boundary. Requests are
    /// immutable after insertion, so executor dispatch can then check this
    /// sealed state without re-hashing a long prompt on every decode quantum.
    pub(crate) fn seal_execution_preparation(&mut self) -> Result<()> {
        self.validate_execution_preparation()?;
        if let Some(ready) = self.chat_execution_ready.as_mut() {
            ready.core_validated = true;
        }
        Ok(())
    }

    pub(crate) fn validate_chat_execution_for_executor(&self) -> Result<()> {
        if self
            .chat_execution_ready
            .as_ref()
            .is_some_and(|ready| ready.core_validated)
        {
            if self.task_type != TaskType::Chat {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries sealed chat preparation for a non-chat task",
                    self.id
                )));
            }
            return Ok(());
        }
        self.validate_chat_execution_preparation()
    }

    pub(crate) fn prepared_chat_prompt_for_executor(
        &self,
    ) -> Result<Option<&NativeChatPreparedPrompt>> {
        self.validate_chat_execution_for_executor()?;
        Ok(self
            .chat_execution_ready
            .as_ref()
            .and_then(|ready| ready.prepared_chat_prompt.as_ref()))
    }

    pub(crate) fn prepared_chat_model_for_executor(&self) -> Result<Arc<NativeChatModel>> {
        self.validate_chat_execution_for_executor()?;
        match &self
            .chat_execution_ready
            .as_ref()
            .expect("validated chat preparation must exist")
            .model
        {
            PreparedChatModel::Exact(model) => Ok(model.model_arc()),
            #[cfg(test)]
            PreparedChatModel::ValidationOnly => Err(Error::InferenceError(
                "Test-only chat preparation does not carry a model instance".to_string(),
            )),
        }
    }

    pub(crate) fn install_asr_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: AsrModelLease,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let prepared_continuous_cost = self
            .uses_asr_retained_sequence()
            .then(|| {
                Self::continuous_asr_stage_cost(self.execution_adapter_binding.as_ref(), &model)
            })
            .transpose()?
            .flatten();
        self.prepared_stage_costs.clear();
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Asr(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn install_lfm25_audio_asr_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Lfm25AudioModelLease,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio ASR request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        // The service path installs the encoder artifact before the model
        // lease, when the stage costs cannot be priced yet; price them here
        // from the artifact's prompt so the stages are never admitted as free
        // (LFA-2).
        let prepared_costs = match self.prepared_lfm25_audio_asr_artifact_for_executor()? {
            Some(artifact) => Self::lfm25_audio_asr_stage_costs(
                self.execution_adapter_binding.as_ref(),
                &model,
                artifact.prompt_tokens,
                self.params.max_tokens,
            )?,
            None => Vec::new(),
        };
        self.prepared_stage_costs.clear();
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Lfm25AudioAsr(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        for (stage_id, cost) in prepared_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    fn continuous_asr_stage_cost(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &NativeAsrModel,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        if matches!(model, NativeAsrModel::Parakeet(_)) {
            let cost = WorkCost::new(
                1,
                1,
                crate::models::architectures::parakeet::asr::PARAKEET_RETAINED_WORKSPACE_PER_ROW_BYTES,
            );
            if !cost
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::Overloaded(
                    "Parakeet retained decode workspace exceeds its loaded adapter budget"
                        .to_string(),
                ));
            }
            return Ok(Some((stage.id, cost)));
        }
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible ASR model"
                    .to_string(),
            ));
        }
        let accelerator_bytes = model.continuous_decode_batch_workspace_per_row_bytes()?;
        let host_bytes = super::continuous_asr_host_workspace_per_row_bytes()?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if super::continuous_asr_workspace_per_row_bytes(accelerator_bytes)?
            > stage.max_workspace_bytes
        {
            return Err(Error::Overloaded(
                "continuous ASR decode workspace exceeds its loaded adapter budget".to_string(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    fn lfm25_audio_asr_stage_costs(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &crate::models::registry::NativeAudioChatModel,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<(StageId, WorkCost)>> {
        Self::lfm25_audio_asr_stage_costs_from_envelopes(binding, |selector| {
            // Reserve a complete prompt span even when the scheduler divides
            // it into smaller quanta. This also bounds singleton span prefill;
            // the per-row cost is aggregated once by physical batch admission.
            if selector == super::StageWorkSelector::SequencePrefill {
                model.lfm25_audio_asr_prefill_resource_envelope(0, prompt_tokens, prompt_tokens)
            } else {
                let position = prompt_tokens
                    .checked_add(max_new_tokens.max(1))
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        Error::Overloaded("LFM2.5 Audio ASR decode position overflowed".into())
                    })?;
                model.lfm25_audio_asr_decode_resource_envelope(position)
            }
        })
    }

    fn lfm25_audio_asr_stage_costs_from_envelopes(
        binding: Option<&super::ExecutionAdapterBinding>,
        mut envelope_for: impl FnMut(
            super::StageWorkSelector,
        ) -> Result<
            crate::models::architectures::lfm25_audio::model::Lfm25AudioAsrStepResourceEnvelope,
        >,
    ) -> Result<Vec<(StageId, WorkCost)>> {
        let Some(binding) = binding else {
            return Ok(Vec::new());
        };
        let mut costs = Vec::new();
        for stage in binding.stages.iter().filter(|stage| {
            matches!(
                stage.selector,
                super::StageWorkSelector::SequencePrefill
                    | super::StageWorkSelector::SequenceDecode
            )
        }) {
            let envelope = envelope_for(stage.selector)?;
            let cost = WorkCost::with_workspace(
                envelope.work_units,
                envelope.materialized_tensor_elements,
                ResourceVector {
                    host_bytes: ResourceAmount::Known(envelope.host_workspace_bytes),
                    device_bytes: ResourceAmount::Known(envelope.device_workspace_bytes),
                    unified_bytes: ResourceAmount::Known(envelope.unified_workspace_bytes),
                    ..ResourceVector::zero()
                },
            );
            if !cost
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::Overloaded(
                    "LFM2.5 Audio ASR workspace exceeds its loaded adapter budget".into(),
                ));
            }
            costs.push((stage.id, cost));
        }
        Ok(costs)
    }

    fn lfm25_audio_tts_stage_costs(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &crate::models::registry::NativeAudioChatModel,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<(StageId, WorkCost)>> {
        let Some(binding) = binding else {
            return Ok(Vec::new());
        };
        let mut costs = Vec::new();
        for stage in binding.stages.iter().filter(|stage| {
            matches!(
                stage.selector,
                super::StageWorkSelector::SequencePrefill
                    | super::StageWorkSelector::SequenceDecode
            )
        }) {
            let envelope = if stage.selector == super::StageWorkSelector::SequencePrefill {
                model.lfm25_audio_tts_prefill_resource_envelope(0, prompt_tokens, prompt_tokens)?
            } else {
                let position = prompt_tokens
                    .checked_add(max_new_tokens.max(1))
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        Error::Overloaded("LFM2.5 Audio TTS position overflowed".into())
                    })?;
                model.lfm25_audio_tts_decode_resource_envelope(position, true)?
            };
            let cost = WorkCost::with_workspace(
                envelope.work_units,
                envelope.materialized_tensor_elements,
                ResourceVector {
                    host_bytes: ResourceAmount::Known(envelope.host_workspace_bytes),
                    device_bytes: ResourceAmount::Known(envelope.device_workspace_bytes),
                    unified_bytes: ResourceAmount::Known(envelope.unified_workspace_bytes),
                    ..ResourceVector::zero()
                },
            );
            if !cost
                .workspace
                .workspace_bytes()
                .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::Overloaded(
                    "LFM2.5 Audio TTS workspace exceeds its loaded adapter budget".into(),
                ));
            }
            costs.push((stage.id, cost));
        }
        Ok(costs)
    }

    fn continuous_tts_stage_cost(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &Qwen3TtsModel,
        prefill_tokens: usize,
        max_frames: usize,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding
                .stages
                .iter()
                .find(|stage| stage.batch_mode == super::NativeBatchMode::Continuous)
        }) else {
            return Ok(None);
        };
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded adapter selected continuous decode for an incompatible TTS model".into(),
            ));
        }
        let (host_bytes, accelerator_bytes) =
            model.continuous_decode_bounded_workspace_per_row_bytes(prefill_tokens, max_frames)?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if !cost
            .workspace
            .workspace_bytes()
            .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
        {
            return Err(Error::Overloaded(
                "continuous TTS decode workspace exceeds its loaded adapter budget".into(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    fn kokoro_tts_stage_cost(
        binding: Option<&super::ExecutionAdapterBinding>,
        prepared: &KokoroPreparedText,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) = binding.and_then(|binding| {
            binding.stages.iter().find(|stage| {
                stage.selector == super::StageWorkSelector::Atomic
                    && stage.batch_mode == super::NativeBatchMode::Static
            })
        }) else {
            return Ok(None);
        };
        let tensor_elements = prepared
            .chunks
            .iter()
            .try_fold(0usize, |total, chunk| {
                total
                    .checked_add(chunk.token_ids.len())?
                    .checked_add(chunk.ref_style.elem_count())
            })
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| Error::Overloaded("Kokoro prepared tensor size overflowed".into()))?;
        Ok(Some((stage.id, WorkCost::new(1, tensor_elements, 0))))
    }

    fn native_prepared_stage(
        binding: Option<&super::ExecutionAdapterBinding>,
        selector: super::StageWorkSelector,
    ) -> Option<&super::StageDescriptor> {
        binding.and_then(|binding| {
            binding.stages.iter().find(|stage| {
                stage.selector == selector && stage.batch_mode != super::NativeBatchMode::None
            })
        })
    }

    fn fit_qwen_tts_layout_to_loaded_context(
        mut layout: crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout,
        max_sequence_tokens: usize,
        request_id: &str,
    ) -> Result<crate::models::architectures::qwen3::tts::Qwen3TtsPhysicalSessionLayout> {
        let output_capacity = max_sequence_tokens
            .checked_sub(layout.prefill_tokens)
            .filter(|capacity| *capacity > 0)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Qwen TTS request {request_id} exact prefill has {} tokens and leaves no output capacity in the configured {max_sequence_tokens}-token context",
                    layout.prefill_tokens
                ))
            })?;
        layout.max_frames = layout.max_frames.min(output_capacity);
        Ok(layout)
    }

    fn resumable_tts_prefill_stage_cost(
        binding: Option<&super::ExecutionAdapterBinding>,
        model: &Qwen3TtsModel,
        prefill_tokens: usize,
        max_frames: usize,
        has_reference: bool,
    ) -> Result<Option<(StageId, WorkCost)>> {
        let Some(stage) =
            Self::native_prepared_stage(binding, super::StageWorkSelector::SequencePrefill)
        else {
            return Ok(None);
        };
        let (host_bytes, accelerator_bytes) =
            model.resumable_prefill_workspace_bytes(prefill_tokens, max_frames, has_reference)?;
        let cost = WorkCost::with_workspace(
            1,
            1,
            ResourceVector {
                host_bytes: ResourceAmount::Known(host_bytes),
                temporary_bytes: ResourceAmount::Known(accelerator_bytes),
                ..ResourceVector::zero()
            },
        );
        if !cost
            .workspace
            .workspace_bytes()
            .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
        {
            return Err(Error::Overloaded(
                "resumable TTS prefill workspace exceeds its loaded adapter budget".into(),
            ));
        }
        Ok(Some((stage.id, cost)))
    }

    pub(crate) fn install_prepared_asr_audio(
        &mut self,
        model_variant: ModelVariant,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Result<()> {
        if sample_rate == 0 || samples.is_empty() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} decoded to an empty or zero-rate audio artifact",
                self.id
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} decoded to non-finite audio samples",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if let Some(current) = self.prepared_asr_audio.as_ref() {
            if current.model_variant == model_variant
                && current.sample_rate == sample_rate
                && current.source_fingerprint == source_fingerprint
                && current.samples.as_ref() == samples.as_slice()
            {
                return Ok(());
            }
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared decoded-audio artifact",
                self.id
            )));
        }
        self.prepared_asr_audio = Some(PreparedAsrAudio {
            model_variant,
            samples: Arc::from(samples.into_boxed_slice()),
            sample_rate,
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_asr_audio_for_executor(&self) -> Result<Option<(Arc<[f32]>, u32)>> {
        let Some(prepared) = self.prepared_asr_audio.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after decoded-audio preparation",
                self.id
            )));
        }
        Ok(Some((prepared.samples.clone(), prepared.sample_rate)))
    }

    /// Duration of the canonical input audio decoded at preparation. ASR
    /// executors report generated audio, which is empty for transcription, so
    /// this is the request's record of how much audio it transcribes.
    pub(crate) fn prepared_asr_input_duration_secs(&self) -> Option<f32> {
        self.prepared_asr_audio
            .as_ref()
            .map(|prepared| (prepared.samples.len() as f64 / prepared.sample_rate as f64) as f32)
    }

    pub(crate) fn prepared_asr_audio_retained_bytes(&self) -> Result<usize> {
        self.prepared_asr_audio.as_ref().map_or(Ok(0), |prepared| {
            prepared
                .samples
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    Error::Overloaded("prepared ASR decoded-audio storage overflowed".to_string())
                })
        })
    }

    pub(crate) fn install_prepared_asr_encoder_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<Qwen3AsrPreparedAudio>,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR || self.model_variant != Some(model_variant) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} encoder artifact does not match its routed task/model",
                self.id
            )));
        }
        if self.uses_asr_long_form_atomic() {
            return Err(Error::InvalidInput(format!(
                "long-form ASR request {} cannot retain a normal-route encoder artifact",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if let Some(current) = self.prepared_asr_encoder_artifact.as_ref() {
            if current.model_variant == model_variant
                && current.source_fingerprint == source_fingerprint
                && matches!(&current.artifact, PreparedAsrEncoderArtifactValue::Qwen3(value) if Arc::ptr_eq(value, &artifact))
            {
                return Ok(());
            }
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Qwen3(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_asr_encoder_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Qwen3AsrPreparedAudio>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Qwen3(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Qwen3 accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_whisper_window(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<WhisperPreparedWindow>,
    ) -> Result<()> {
        if model_variant.family() != crate::catalog::ModelFamily::WhisperAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Whisper artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Whisper(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_whisper_window_for_executor(
        &self,
    ) -> Result<Option<Arc<WhisperPreparedWindow>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Whisper encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Whisper(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Whisper accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_vibevoice_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<VibeVoiceAsrPreparedArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::VibeVoiceAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} VibeVoice artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::VibeVoice(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_vibevoice_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VibeVoiceAsrPreparedArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after VibeVoice encoder preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::VibeVoice(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the VibeVoice accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_granite_speech_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<GraniteSpeechPreparedPromptArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::GraniteSpeechAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Granite Speech artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::GraniteSpeech(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_granite_speech_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<GraniteSpeechPreparedPromptArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Granite Speech preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::GraniteSpeech(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::Lfm25Audio(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Granite Speech accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_lfm25_audio_asr_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<Lfm25AudioPreparedAsrArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} LFM2.5 Audio artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        let prepared_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model),
                _ => None,
            })
            .map(|model| {
                Self::lfm25_audio_asr_stage_costs(
                    self.execution_adapter_binding.as_ref(),
                    model,
                    artifact.prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Lfm25Audio(artifact),
            source_fingerprint,
        });
        for (stage_id, cost) in prepared_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn prepared_lfm25_audio_asr_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Lfm25AudioPreparedAsrArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after LFM2.5 Audio preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Lfm25Audio(value) => Ok(Some(value.clone())),
            PreparedAsrEncoderArtifactValue::Qwen3(_)
            | PreparedAsrEncoderArtifactValue::Whisper(_)
            | PreparedAsrEncoderArtifactValue::VibeVoice(_)
            | PreparedAsrEncoderArtifactValue::GraniteSpeech(_)
            | PreparedAsrEncoderArtifactValue::Parakeet(_) => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the LFM2.5 Audio accessor".into(),
            )),
        }
    }

    pub(crate) fn install_prepared_parakeet_artifact(
        &mut self,
        model_variant: ModelVariant,
        artifact: Arc<ParakeetPreparedEncoderArtifact>,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::ParakeetAsr
            || self.task_type != TaskType::ASR
            || self.model_variant != Some(model_variant)
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} Parakeet artifact does not match its normal routed model",
                self.id
            )));
        }
        let source_fingerprint = self.refresh_asr_execution_source_fingerprint(model_variant)?;
        if self.prepared_asr_encoder_artifact.is_some() {
            return Err(Error::InvalidInput(format!(
                "ASR request {} already owns an encoder artifact",
                self.id
            )));
        }
        self.prepared_asr_encoder_artifact = Some(PreparedAsrEncoderArtifact {
            model_variant,
            artifact: PreparedAsrEncoderArtifactValue::Parakeet(artifact),
            source_fingerprint,
        });
        Ok(())
    }

    pub(crate) fn prepared_parakeet_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<ParakeetPreparedEncoderArtifact>>> {
        let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() else {
            return Ok(None);
        };
        if self.model_variant != Some(prepared.model_variant)
            || self.asr_execution_source_fingerprint(prepared.model_variant)?
                != prepared.source_fingerprint
            || self.uses_asr_long_form_atomic()
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed after Parakeet preparation",
                self.id
            )));
        }
        match &prepared.artifact {
            PreparedAsrEncoderArtifactValue::Parakeet(value) => Ok(Some(value.clone())),
            _ => Err(Error::InvalidInput(
                "foreign encoder artifact was requested through the Parakeet accessor".into(),
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn prepared_asr_encoder_artifact_retained_bytes(&self) -> Result<u64> {
        self.prepared_asr_encoder_artifact
            .as_ref()
            .map_or(Ok(0), |prepared| {
                prepared
                    .artifact
                    .resident_resources()
                    .map(|(_, device)| device)
            })
    }

    pub(crate) fn prepared_asr_encoder_artifact_retained_resources(&self) -> Result<(u64, u64)> {
        self.prepared_asr_encoder_artifact
            .as_ref()
            .map_or(Ok((0, 0)), |prepared| {
                prepared.artifact.resident_resources()
            })
    }

    /// Derive independently clocked retained-state work from immutable
    /// preparation metadata. This method never chooses a model or a stage:
    /// the already-bound stage authorizes exact group/clock pairs, while Core
    /// authenticates the returned spans against the physical state plan.
    pub(crate) fn project_auxiliary_state_spans(
        &self,
        stage: &StageDescriptor,
        input: InputRange,
    ) -> Result<Option<Arc<[ClockedStateSpan]>>> {
        let Some(authorized) = stage.retained_state_selections.as_deref() else {
            return Ok(None);
        };
        if self
            .incremental_model_execution_ready
            .as_ref()
            .is_some_and(|ready| {
                matches!(ready.model, PreparedIncrementalModel::VibeVoiceTts(_))
                    && ready.vibevoice_tts.is_some()
            })
        {
            if authorized.is_empty() {
                return Ok(Some(Arc::from([])));
            }
            let selection = super::ClockedStateSelection::new(
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_TOKENIZER_GROUP,
                crate::kv::v2::StateClock::CodecFrames,
            )?;
            if authorized != [selection.clone()] {
                return Err(Error::InvalidInput(format!(
                    "stage {} does not authorize the VibeVoice TTS tokenizer clock",
                    stage.name
                )));
            }
            let prompt_tokens = self.prepared_sequence_input_tokens.ok_or_else(|| {
                Error::InferenceError(
                    "prepared VibeVoice TTS request lost its prompt boundary".into(),
                )
            })?;
            let start = input.start.checked_sub(prompt_tokens).ok_or_else(|| {
                Error::InvalidInput(
                    "VibeVoice TTS tokenizer state cannot advance during prompt prefill".into(),
                )
            })?;
            let end = input.end.checked_sub(prompt_tokens).ok_or_else(|| {
                Error::InvalidInput(
                    "VibeVoice TTS tokenizer state crossed its prompt boundary".into(),
                )
            })?;
            return Ok(Some(Arc::from([ClockedStateSpan::new(
                selection.group(),
                selection.clock().clone(),
                InputRange::new(start, end)?,
            )?])));
        }
        let projections = if let Some(prepared) = self.prepared_asr_encoder_artifact.as_ref() {
            if self.model_variant != Some(prepared.model_variant)
                || self.asr_execution_source_fingerprint(prepared.model_variant)?
                    != prepared.source_fingerprint
                || self.uses_asr_long_form_atomic()
            {
                return Err(Error::InvalidInput(format!(
                    "ASR request {} changed after auxiliary state projection preparation",
                    self.id
                )));
            }
            prepared.artifact.clocked_state_projections()
        } else {
            &[]
        };

        let mut previous = None;
        let mut spans = Vec::new();
        for projection in projections {
            let group = projection.selection().group();
            let primary = projection.primary();
            if let Some((previous_group, previous_clock, previous_end)) = previous.as_ref() {
                if group.get() < *previous_group
                    || (group.get() == *previous_group
                        && (projection.selection().clock() != previous_clock
                            || primary.start < *previous_end))
                {
                    return Err(Error::InvalidInput(format!(
                        "request {} has non-canonical clocked state projections",
                        self.id
                    )));
                }
            }
            previous = Some((
                group.get(),
                projection.selection().clock().clone(),
                primary.end,
            ));

            let Some(span) = projection.project(input)? else {
                continue;
            };
            if !authorized
                .iter()
                .any(|selection| selection == projection.selection())
            {
                return Err(Error::InvalidInput(format!(
                    "stage {} does not authorize request {} auxiliary group {} {:?}",
                    stage.name,
                    self.id,
                    group.get(),
                    projection.selection().clock()
                )));
            }
            spans.push(span);
        }

        Ok(Some(Arc::from(spans)))
    }

    pub(crate) fn project_paged_state_input(
        &self,
        domain: crate::kv::CacheDomainId,
        phase: super::SequencePhase,
        input: InputRange,
    ) -> Result<InputRange> {
        let is_vibevoice_tts =
            self.incremental_model_execution_ready
                .as_ref()
                .is_some_and(|ready| {
                    matches!(ready.model, PreparedIncrementalModel::VibeVoiceTts(_))
                        && ready.vibevoice_tts.is_some()
                });
        if !is_vibevoice_tts
            || domain != crate::models::architectures::vibevoice::VIBEVOICE_TTS_NEGATIVE_DOMAIN
        {
            return Ok(input);
        }
        match phase {
            super::SequencePhase::Prefill if input.start == 0 => InputRange::new(0, 1),
            super::SequencePhase::Prefill => Err(Error::InferenceError(
                "VibeVoice TTS requires its exact prompt prefill in one quantum".into(),
            )),
            super::SequencePhase::Decode => {
                let prompt_tokens = self.prepared_sequence_input_tokens.ok_or_else(|| {
                    Error::InferenceError(
                        "prepared VibeVoice TTS request lost its prompt boundary".into(),
                    )
                })?;
                let start = input
                    .start
                    .checked_sub(prompt_tokens)
                    .and_then(|frames| frames.checked_add(1))
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "VibeVoice TTS negative cache start crossed its prompt boundary".into(),
                        )
                    })?;
                let end = input
                    .end
                    .checked_sub(prompt_tokens)
                    .and_then(|frames| frames.checked_add(1))
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "VibeVoice TTS negative cache end crossed its prompt boundary".into(),
                        )
                    })?;
                InputRange::new(start, end)
            }
        }
    }

    pub(crate) fn install_prepared_sequence_input_tokens(
        &mut self,
        input_tokens: usize,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR {
            return Err(Error::InvalidInput(format!(
                "Request {} cannot install multimodal sequence shape for {:?}",
                self.id, self.task_type
            )));
        }
        if self.prepared_asr_execution_shape == Some(PreparedAsrExecutionShape::LongFormAtomic) {
            return Err(Error::InvalidInput(format!(
                "ASR request {} cannot change its prepared long-form route to retained sequence execution",
                self.id
            )));
        }
        if input_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "ASR request {} has an empty exact prefill",
                self.id
            )));
        }
        if input_tokens >= max_sequence_tokens {
            // The load fitted a context too small for this request's audio:
            // a capacity refusal (503); longer audio is chunked upstream.
            return Err(Error::Overloaded(format!(
                "ASR request {} exact prefill has {input_tokens} tokens and leaves no output capacity in the configured {max_sequence_tokens}-token context",
                self.id
            )));
        }
        if self
            .prepared_sequence_input_tokens
            .is_some_and(|current| current != input_tokens)
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} changed its prepared multimodal sequence shape",
                self.id
            )));
        }
        self.params.max_tokens = self
            .params
            .max_tokens
            .max(1)
            .min(max_sequence_tokens - input_tokens);
        self.prepared_sequence_input_tokens = Some(input_tokens);
        self.prepared_asr_execution_shape =
            Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens });
        Ok(())
    }

    /// Route the request to the long-form atomic executor, which runs `plan`
    /// as planned here.
    pub(crate) fn install_prepared_asr_long_form_atomic(
        &mut self,
        plan: PlannedAsrChunks,
    ) -> Result<()> {
        if self.task_type != TaskType::ASR {
            return Err(Error::InvalidInput(format!(
                "Request {} cannot install an ASR long-form route for {:?}",
                self.id, self.task_type
            )));
        }
        if self.prepared_sequence_input_tokens.is_some()
            || matches!(
                self.prepared_asr_execution_shape,
                Some(PreparedAsrExecutionShape::RetainedSequence { .. })
            )
        {
            return Err(Error::InvalidInput(format!(
                "ASR request {} cannot change its prepared retained sequence route to long-form execution",
                self.id
            )));
        }
        self.prepared_asr_execution_shape = Some(PreparedAsrExecutionShape::LongFormAtomic);
        self.prepared_asr_route_plan = Some(Arc::new(plan));
        Ok(())
    }

    /// The segmentation preparation planned for a long-form row.
    pub(crate) fn prepared_asr_route_plan(&self) -> Option<Arc<PlannedAsrChunks>> {
        self.prepared_asr_route_plan.clone()
    }

    pub(crate) fn prepared_asr_execution_shape(&self) -> Option<PreparedAsrExecutionShape> {
        self.prepared_asr_execution_shape
    }

    pub(crate) fn uses_asr_retained_sequence(&self) -> bool {
        matches!(
            self.prepared_asr_execution_shape,
            Some(PreparedAsrExecutionShape::RetainedSequence { .. })
        )
    }

    pub(crate) fn uses_asr_long_form_atomic(&self) -> bool {
        self.prepared_asr_execution_shape == Some(PreparedAsrExecutionShape::LongFormAtomic)
    }

    /// Clamp TTS generation parameters once, before any preparation reads
    /// them. Preparation records the speed it prepared for and execution
    /// checks it against the request, so a later clamp (engine admission)
    /// failed every request outside the range with "changed after
    /// preparation" (KOK-20).
    pub(crate) fn normalize_tts_params_before_preparation(&mut self) -> Result<()> {
        if self.task_type == TaskType::TTS {
            self.params.speed = clamp_tts_speed(self.params.speed);
            self.params.speech.validate()?;
        }
        Ok(())
    }

    pub(crate) fn lfm25_audio_tts_messages_for_preparation(&self) -> Result<Vec<ChatMessage>> {
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio TTS request is missing text".into())
            })?;
        let speaker = self
            .params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref());
        Ok(vec![
            ChatMessage {
                role: ChatRole::System,
                content: lfm25_audio_tts_system_prompt(speaker).to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: text.trim().to_string(),
            },
        ])
    }

    pub(crate) fn install_lfm25_audio_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: Lfm25AudioModelLease,
        artifact: Arc<Lfm25AudioPreparedTtsArtifact>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if model_variant.family() != ModelFamily::Lfm25Audio
            || self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio TTS request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let expected = self.lfm25_audio_tts_messages_for_preparation()?;
        if artifact.source_messages.len() != expected.len()
            || artifact
                .source_messages
                .iter()
                .zip(&expected)
                .any(|(actual, expected)| {
                    actual.role != expected.role || actual.content != expected.content
                })
            || artifact.prompt_tokens == 0
            || artifact.prompt_tokens >= max_sequence_tokens
        {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio TTS request {} prepared prompt does not match its exact input/context",
                self.id
            )));
        }
        let available = max_sequence_tokens - artifact.prompt_tokens;
        self.params.max_tokens = if self.params.max_tokens == 0 {
            1_024.min(available)
        } else {
            self.params.max_tokens.min(available)
        };
        if self.params.max_tokens == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS request leaves no output capacity".into(),
            ));
        }
        let stage_costs = Self::lfm25_audio_tts_stage_costs(
            self.execution_adapter_binding.as_ref(),
            &model,
            artifact.prompt_tokens,
            self.params.max_tokens,
        )?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Lfm25AudioTts(model),
            qwen_tts: None,
            lfm25_audio_tts: Some(artifact),
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        for (stage, cost) in stage_costs {
            self.install_prepared_stage_cost(stage, cost)?;
        }
        Ok(())
    }

    pub(crate) fn lfm25_audio_tts_generation_config(&self) -> Lfm25AudioGenerationConfig {
        // Unset fields keep the reference 0.8 / top-k 64; the seed defaults
        // to one derived from the request id, as audio chat does.
        let speech = &self.params.speech;
        Lfm25AudioGenerationConfig::tts(
            self.params.audio_temperature.or(speech.temperature),
            self.params.audio_top_k.or(speech.top_k),
            speech
                .seed
                .unwrap_or_else(|| Self::chat_request_seed(&self.id)),
        )
    }

    pub(crate) fn install_vibevoice_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: VibeVoiceTtsModelLease,
        artifact: Arc<VibeVoiceTtsPreparedArtifact>,
        params: VibeVoiceTtsGenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::VibeVoiceTts
            || artifact.prompt_tokens() == 0
            || artifact.prompt_tokens() >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "VibeVoice TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens());
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::VibeVoiceTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: Some(artifact),
            vibevoice_tts_params: Some(params),
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_fish_s2_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: FishS2TtsModelLease,
        artifact: Arc<FishS2PreparedArtifact>,
        params: FishS2GenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::FishS2Tts
            || artifact.prompt_tokens() == 0
            || artifact.prompt_tokens() >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "Fish S2 TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        // Validate exact reference + text + full output room before admission.
        crate::models::architectures::fish_s2::effective_frame_budget(
            artifact.prompt_tokens(),
            max_sequence_tokens,
            max_sequence_tokens,
            params.max_frames,
        )?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens());
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::FishS2Tts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: Some(artifact),
            fish_s2_tts_params: Some(params),
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_voxtral_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: VoxtralTtsModelLease,
        artifact: Arc<VoxtralTtsPreparedArtifact>,
        params: VoxtralTtsGenerationParams,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::VoxtralTts
            || artifact.prompt_tokens == 0
            || artifact.prompt_tokens >= max_sequence_tokens
            || params.max_frames == 0
        {
            return Err(Error::InvalidInput(format!(
                "Voxtral TTS request {} preparation does not match its routed model/context",
                self.id
            )));
        }
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(artifact.prompt_tokens);
        self.params.max_tokens = params.max_frames;
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::VoxtralTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: Some(artifact),
            voxtral_tts_params: Some(params),
            kokoro_tts: None,
        });
        Ok(())
    }

    pub(crate) fn install_kokoro_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: KokoroTtsModelLease,
        artifact: Arc<KokoroPreparedText>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Kokoro TTS request is missing text".into()))?;
        let speaker = self
            .params
            .speaker
            .as_deref()
            .or(self.params.voice.as_deref());
        let input_tokens = artifact.input_tokens()?;
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::KokoroTts
            || !artifact.matches_request(text, speaker, self.language.as_deref(), self.params.speed)
            || !artifact.is_well_formed()
            || artifact.longest_chunk_tokens() > max_sequence_tokens
        {
            return Err(Error::InvalidInput(format!(
                "Kokoro TTS request {} preparation does not match its routed input/model/context",
                self.id
            )));
        }
        let prepared_stage_cost =
            Self::kokoro_tts_stage_cost(self.execution_adapter_binding.as_ref(), &artifact)?;
        self.prepared_stage_costs.clear();
        self.prepared_sequence_input_tokens = Some(input_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::KokoroTts(model),
            qwen_tts: None,
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: Some(artifact),
        });
        if let Some((stage, cost)) = prepared_stage_cost {
            self.install_prepared_stage_cost(stage, cost)?;
        }
        Ok(())
    }

    pub(crate) fn install_qwen_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: QwenTtsModelLease,
        reference: Option<Arc<SpeakerReference>>,
        max_sequence_tokens: usize,
    ) -> Result<()> {
        if self.task_type != TaskType::TTS
            || self.model_variant != Some(model_variant)
            || model_variant.family() != ModelFamily::Qwen3Tts
        {
            return Err(Error::InvalidInput(format!(
                "Qwen TTS request {} model preparation does not match its routed task/model",
                self.id
            )));
        }
        let model_arc = model.model_arc();
        self.prepared_stage_costs.clear();
        let text = self
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("Qwen TTS request is missing text".to_string()))?;
        let speakers = model_arc.available_speakers();
        let speaker = if reference.is_some() || speakers.is_empty() {
            None
        } else {
            let requested = self
                .params
                .speaker
                .as_deref()
                .or(self.params.voice.as_deref())
                .filter(|speaker| !speaker.trim().is_empty())
                .unwrap_or_else(|| speakers[0].as_str());
            let resolved = speakers
                .iter()
                .find(|speaker| speaker.eq_ignore_ascii_case(requested))
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Unknown speaker '{requested}'. Available speakers: {}",
                        speakers
                            .iter()
                            .map(|speaker| speaker.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
            Some((*resolved).clone())
        };
        let params = self.qwen_tts_generation_params();
        let layout = model_arc.physical_session_layout(TtsSessionCacheRequest {
            text,
            reference: reference.as_deref(),
            language: self.language.as_deref(),
            instruct: self.voice_description.as_deref(),
            uses_preset_speaker: speaker.is_some(),
            max_frames: params.max_frames,
        })?;
        let layout =
            Self::fit_qwen_tts_layout_to_loaded_context(layout, max_sequence_tokens, &self.id)?;
        let prepared_continuous_cost = Self::continuous_tts_stage_cost(
            self.execution_adapter_binding.as_ref(),
            &model_arc,
            layout.prefill_tokens,
            layout.max_frames,
        )?;
        let prepared_prefill_cost = Self::resumable_tts_prefill_stage_cost(
            self.execution_adapter_binding.as_ref(),
            &model_arc,
            layout.prefill_tokens,
            layout.max_frames,
            reference.is_some(),
        )?;
        if layout.prefill_tokens == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen TTS request {} exact prefill is empty",
                self.id
            )));
        }
        self.params.max_tokens = layout.max_frames;
        self.prepared_sequence_input_tokens = Some(layout.prefill_tokens);
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::QwenTts(model),
            qwen_tts: Some(PreparedQwenTtsInput {
                reference,
                speaker,
                prefill_tokens: layout.prefill_tokens,
                max_frames: layout.max_frames,
            }),
            lfm25_audio_tts: None,
            vibevoice_tts: None,
            vibevoice_tts_params: None,
            fish_s2_tts: None,
            fish_s2_tts_params: None,
            voxtral_tts: None,
            voxtral_tts_params: None,
            kokoro_tts: None,
        });
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_prefill_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn qwen_tts_generation_params(&self) -> TtsGenerationParams {
        let model_max_frames = self
            .model_variant
            .and_then(|variant| variant.tts_max_output_frames_hint())
            .unwrap_or(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        let prepared_max_frames = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.qwen_tts.as_ref())
            .map(|prepared| prepared.max_frames);
        let max_frames = if let Some(max_frames) = prepared_max_frames {
            max_frames
        } else if self.params.max_tokens == 0 {
            model_max_frames
        } else {
            self.params.max_tokens.clamp(16, model_max_frames.max(16))
        };
        TtsGenerationParams::for_request(&self.params.speech, max_frames)
    }

    fn validate_incremental_model_execution_preparation(&self) -> Result<()> {
        let Some(ready) = self.incremental_model_execution_ready.as_ref() else {
            if self.prepared_sequence_input_tokens.is_some()
                || self.prepared_asr_execution_shape.is_some()
                || self.prepared_asr_audio.is_some()
                || self.prepared_asr_encoder_artifact.is_some()
            {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries prepared multimodal input without an exact loaded model",
                    self.id
                )));
            }
            return Ok(());
        };
        if self.model_variant != Some(ready.model_variant) {
            return Err(Error::InvalidInput(format!(
                "Request {} changed model after incremental model preparation",
                self.id
            )));
        }
        if matches!(&ready.model, PreparedIncrementalModel::Asr(_))
            && matches!(
                ready.model_variant.family(),
                ModelFamily::Qwen3Asr
                    | ModelFamily::WhisperAsr
                    | ModelFamily::VibeVoiceAsr
                    | ModelFamily::GraniteSpeechAsr
                    | ModelFamily::ParakeetAsr
            )
        {
            if self.prepared_asr_audio_for_executor()?.is_none() {
                return Err(Error::InvalidInput(format!(
                    "Qwen3 ASR request {} is missing its exact decoded-audio artifact",
                    self.id
                )));
            }
            match self.prepared_asr_execution_shape {
                Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens })
                    if self.prepared_sequence_input_tokens == Some(input_tokens)
                        && match ready.model_variant.family() {
                            ModelFamily::Qwen3Asr => {
                                self.prepared_asr_encoder_artifact_for_executor()?.is_some()
                            }
                            ModelFamily::WhisperAsr => {
                                self.prepared_whisper_window_for_executor()?.is_some()
                            }
                            ModelFamily::VibeVoiceAsr => {
                                self.prepared_vibevoice_artifact_for_executor()?.is_some()
                            }
                            ModelFamily::GraniteSpeechAsr => self
                                .prepared_granite_speech_artifact_for_executor()?
                                .is_some(),
                            ModelFamily::ParakeetAsr => {
                                self.prepared_parakeet_artifact_for_executor()?.is_some()
                            }
                            _ => false,
                        } => {}
                Some(PreparedAsrExecutionShape::LongFormAtomic)
                    if self.prepared_sequence_input_tokens.is_none()
                        && self.prepared_asr_encoder_artifact.is_none() => {}
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "ASR request {} is missing a consistent exact media execution shape",
                        self.id
                    )));
                }
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::Lfm25AudioAsr(_)) {
            if ready.model_variant.family() != ModelFamily::Lfm25Audio
                || self.prepared_asr_audio_for_executor()?.is_none()
            {
                return Err(Error::InvalidInput(format!(
                    "LFM2.5 Audio ASR request {} is missing its exact decoded-audio artifact",
                    self.id
                )));
            }
            match self.prepared_asr_execution_shape {
                Some(PreparedAsrExecutionShape::RetainedSequence { input_tokens })
                    if self.prepared_sequence_input_tokens == Some(input_tokens)
                        && self
                            .prepared_lfm25_audio_asr_artifact_for_executor()?
                            .is_some() => {}
                Some(PreparedAsrExecutionShape::LongFormAtomic)
                    if self.prepared_sequence_input_tokens.is_none()
                        && self.prepared_asr_encoder_artifact.is_none() => {}
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "LFM2.5 Audio ASR request {} is missing a consistent exact media execution shape",
                        self.id
                    )));
                }
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::Lfm25AudioTts(_)) {
            let expected = self.lfm25_audio_tts_messages_for_preparation()?;
            let prepared = ready.lfm25_audio_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "LFM2.5 Audio TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::Lfm25Audio
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens)
                || prepared.source_messages.len() != expected.len()
                || prepared
                    .source_messages
                    .iter()
                    .zip(&expected)
                    .any(|(actual, expected)| {
                        actual.role != expected.role || actual.content != expected.content
                    })
            {
                return Err(Error::InvalidInput(format!(
                    "LFM2.5 Audio TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::VibeVoiceTts(_)) {
            let prepared = ready.vibevoice_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "VibeVoice TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.vibevoice_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "VibeVoice TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::VibeVoiceTts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens())
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "VibeVoice TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::FishS2Tts(_)) {
            let prepared = ready.fish_s2_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Fish S2 TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.fish_s2_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Fish S2 TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::FishS2Tts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens())
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "Fish S2 TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::VoxtralTts(_)) {
            let prepared = ready.voxtral_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Voxtral TTS request {} is missing its prepared prompt",
                    self.id
                ))
            })?;
            let params = ready.voxtral_tts_params.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Voxtral TTS request {} is missing its sealed generation geometry",
                    self.id
                ))
            })?;
            if ready.model_variant.family() != ModelFamily::VoxtralTts
                || self.prepared_sequence_input_tokens != Some(prepared.prompt_tokens)
                || self.params.max_tokens != params.max_frames
            {
                return Err(Error::InvalidInput(format!(
                    "Voxtral TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        if matches!(&ready.model, PreparedIncrementalModel::KokoroTts(_)) {
            let prepared = ready.kokoro_tts.as_ref().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Kokoro TTS request {} is missing its prepared input",
                    self.id
                ))
            })?;
            let text = self.text.as_deref().unwrap_or_default();
            let speaker = self
                .params
                .speaker
                .as_deref()
                .or(self.params.voice.as_deref());
            let input_tokens = prepared.input_tokens()?;
            if ready.model_variant.family() != ModelFamily::KokoroTts
                || !prepared.matches_request(
                    text,
                    speaker,
                    self.language.as_deref(),
                    self.params.speed,
                )
                || !prepared.is_well_formed()
                || self.prepared_sequence_input_tokens != Some(input_tokens)
            {
                return Err(Error::InvalidInput(format!(
                    "Kokoro TTS request {} changed after preparation",
                    self.id
                )));
            }
        }
        match (&ready.model, self.task_type) {
            (PreparedIncrementalModel::Asr(_), TaskType::ASR) if ready.qwen_tts.is_none() => Ok(()),
            (PreparedIncrementalModel::Lfm25AudioAsr(_), TaskType::ASR)
                if ready.qwen_tts.is_none() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::Lfm25AudioTts(_), TaskType::TTS)
                if ready.qwen_tts.is_none() && ready.lfm25_audio_tts.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::QwenTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::Qwen3Tts
                    && ready.qwen_tts.as_ref().is_some_and(|prepared| {
                        self.prepared_sequence_input_tokens == Some(prepared.prefill_tokens)
                            && self.params.max_tokens == prepared.max_frames
                    }) =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::VibeVoiceTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::VibeVoiceTts
                    && ready.vibevoice_tts.is_some()
                    && ready.vibevoice_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::FishS2Tts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::FishS2Tts
                    && ready.fish_s2_tts.is_some()
                    && ready.fish_s2_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::VoxtralTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::VoxtralTts
                    && ready.voxtral_tts.is_some()
                    && ready.voxtral_tts_params.is_some() =>
            {
                Ok(())
            }
            (PreparedIncrementalModel::KokoroTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::KokoroTts
                    && ready.kokoro_tts.is_some() =>
            {
                Ok(())
            }
            _ => Err(Error::InvalidInput(format!(
                "Request {} carries incremental preparation for a different task",
                self.id
            ))),
        }
    }

    pub(crate) fn prepared_asr_model_for_executor(&self) -> Result<Option<Arc<NativeAsrModel>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model.model_arc()),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_asr_model_lease_for_executor(&self) -> Result<Option<AsrModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model.clone()),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_lfm25_audio_asr_model_lease_for_executor(
        &self,
    ) -> Result<Option<Lfm25AudioModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model.clone()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_lfm25_audio_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<Lfm25AudioModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_lfm25_audio_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<Lfm25AudioPreparedTtsArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.lfm25_audio_tts.clone()))
    }

    pub(crate) fn prepared_vibevoice_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<VibeVoiceTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::VibeVoiceTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_vibevoice_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VibeVoiceTtsPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.vibevoice_tts.clone()))
    }

    pub(crate) fn vibevoice_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<VibeVoiceTtsGenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.vibevoice_tts_params.clone()))
    }

    pub(crate) fn prepared_fish_s2_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<FishS2TtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::FishS2Tts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_fish_s2_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<FishS2PreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.fish_s2_tts.clone()))
    }

    pub(crate) fn fish_s2_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<FishS2GenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.fish_s2_tts_params.clone()))
    }

    pub(crate) fn prepared_voxtral_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<VoxtralTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::VoxtralTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_voxtral_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<VoxtralTtsPreparedArtifact>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.voxtral_tts.clone()))
    }

    pub(crate) fn voxtral_tts_generation_params_for_executor(
        &self,
    ) -> Result<Option<VoxtralTtsGenerationParams>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.voxtral_tts_params))
    }

    pub(crate) fn prepared_kokoro_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<KokoroTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::KokoroTts(model) => Some(model.clone()),
                _ => None,
            }))
    }

    pub(crate) fn prepared_kokoro_tts_artifact_for_executor(
        &self,
    ) -> Result<Option<Arc<KokoroPreparedText>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.kokoro_tts.clone()))
    }

    pub(crate) fn prepared_qwen_tts_model_for_executor(
        &self,
    ) -> Result<Option<Arc<Qwen3TtsModel>>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::QwenTts(model) => Some(model.model_arc()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_qwen_tts_model_lease_for_executor(
        &self,
    ) -> Result<Option<QwenTtsModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::QwenTts(model) => Some(model.clone()),
                PreparedIncrementalModel::Asr(_)
                | PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            }))
    }

    pub(crate) fn prepared_qwen_tts_input_for_executor(&self) -> Result<&PreparedQwenTtsInput> {
        self.validate_incremental_model_execution_preparation()?;
        self.incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| ready.qwen_tts.as_ref())
            .ok_or_else(|| {
                Error::InferenceError(
                    "Qwen TTS request is missing exact host preparation".to_string(),
                )
            })
    }

    pub(crate) fn chat_generation_config(&self) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: self.params.temperature.max(0.0),
            top_p: self.params.top_p.clamp(0.0, 1.0),
            top_k: self.params.top_k,
            repetition_penalty: self.params.repetition_penalty.max(1.0),
            presence_penalty: self.params.presence_penalty.clamp(-2.0, 2.0),
            stop_token_ids: self.params.stop_token_ids.clone(),
            seed: self
                .params
                .seed
                .map_or_else(|| Self::chat_request_seed(&self.id), Self::chat_user_seed),
            request: self.chat_config.clone(),
            logprobs: self.params.logprobs || self.params.top_logprobs > 0,
            top_logprobs: self.params.top_logprobs,
            constrain_json_object: self.params.constrain_json_object,
            skip_special_tokens: self.params.skip_special_tokens,
        }
    }

    pub(crate) fn chat_request_seed(request_id: &str) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET_BASIS;
        for byte in request_id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// The sampler seed for a caller's `seed`. The chat samplers seed from
    /// the clock on 0, so a caller's seed is mixed (SplitMix64's finalizer,
    /// a bijection) rather than passed through, and a 0 result becomes 1.
    pub(crate) fn chat_user_seed(seed: u64) -> u64 {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)).max(1)
    }

    /// Create a new TTS request.
    pub fn tts(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Tts(TtsEngineInput {
                // Large direct metadata is validated before the canonical
                // typed view is populated during request processing.
                text: String::new(),
                reference_audio: None,
                reference_text: None,
                voice_description: None,
            }),
            task_type: TaskType::TTS,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: Some(text),
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new ASR request.
    pub fn asr(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                // The compatibility field below owns the caller buffer until
                // bounded processing moves it into this typed representation.
                audio: EngineAudioInput::Base64(String::new()),
                language: None,
                prompt: None,
            }),
            task_type: TaskType::ASR,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new ASR request from already-decoded audio bytes.
    pub fn asr_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                // Keep construction allocation-free beyond the caller-owned
                // buffer; processing installs it here after source preflight.
                audio: EngineAudioInput::Bytes(Vec::new()),
                language: None,
                prompt: None,
            }),
            task_type: TaskType::ASR,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new chat request.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Chat(ChatEngineInput {
                messages: messages.clone(),
                chat_config: ChatRequestConfig::default(),
                prompt_tokens: Vec::new(),
            }),
            task_type: TaskType::Chat,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: Some(messages),
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new speech-to-speech request.
    pub fn speech_to_speech(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Base64(String::new()),
                messages: Vec::new(),
                system_prompt: None,
            }),
            task_type: TaskType::SpeechToSpeech,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    /// Create a new speech-to-speech request from already-decoded audio bytes.
    pub fn speech_to_speech_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Bytes(Vec::new()),
                messages: Vec::new(),
                system_prompt: None,
            }),
            task_type: TaskType::SpeechToSpeech,
            model_variant: None,
            model_instance_id: None,
            execution_adapter_binding: None,
            v2_state_descriptor: None,
            v2_state_fingerprint: None,
            v2_state_runtime: None,
            managed_cache_runtime: None,
            managed_prefix_cursor: None,
            managed_snapshot_boundary: None,
            managed_preamble_boundary: None,
            prepared_stage_costs: Vec::new(),
            prepared_sequence_input_tokens: None,
            prepared_asr_execution_shape: None,
            prepared_asr_route_plan: None,
            prepared_asr_audio: None,
            asr_source_audio_digest: None,
            prepared_asr_encoder_artifact: None,
            stream_staging: StreamStagingBuffer::default(),
            collect_audio_samples: true,
            tenant_key: None,
            audio_stream_external_queue_capacity: 0,
            audio_stream_engine_queue_capacity: None,
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            asr_native_word_timestamps: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            runtime_admitted_at: None,
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            realtime_asr_ingress: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    pub(crate) fn set_cancellation_signal(&mut self, signal: Arc<AtomicBool>) {
        self.cancellation = Some(signal);
    }

    pub(crate) fn bind_model_instance(
        &mut self,
        model_instance_id: super::ModelInstanceId,
    ) -> Result<()> {
        if self
            .model_instance_id
            .is_some_and(|current| current != model_instance_id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different model instance".to_string(),
            ));
        }
        self.model_instance_id = Some(model_instance_id);
        Ok(())
    }

    pub(crate) fn model_instance_id(&self) -> Option<super::ModelInstanceId> {
        self.model_instance_id
    }

    pub(crate) fn bind_execution_adapter(
        &mut self,
        binding: super::ExecutionAdapterBinding,
    ) -> Result<()> {
        binding.validate()?;
        if self.model_variant != Some(binding.model_variant) {
            return Err(Error::InvalidInput(
                "engine request model does not match its execution adapter".to_string(),
            ));
        }
        self.bind_model_instance(binding.model_instance_id)?;
        if self
            .execution_adapter_binding
            .as_ref()
            .is_some_and(|current| current != &binding)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different execution adapter".to_string(),
            ));
        }
        let prepared_continuous_cost = self
            .chat_execution_ready
            .as_ref()
            .map(|ready| Self::continuous_chat_stage_cost(Some(&binding), &ready.model))
            .transpose()?
            .flatten();
        let prepared_asr_continuous_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model),
                PreparedIncrementalModel::Lfm25AudioAsr(_)
                | PreparedIncrementalModel::Lfm25AudioTts(_)
                | PreparedIncrementalModel::QwenTts(_) => None,
                PreparedIncrementalModel::VibeVoiceTts(_)
                | PreparedIncrementalModel::FishS2Tts(_)
                | PreparedIncrementalModel::VoxtralTts(_)
                | PreparedIncrementalModel::KokoroTts(_) => None,
            })
            .filter(|_| self.uses_asr_retained_sequence())
            .map(|model| Self::continuous_asr_stage_cost(Some(&binding), model))
            .transpose()?
            .flatten();
        let prepared_lfm25_asr_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Lfm25AudioAsr(model) => Some(model),
                _ => None,
            })
            .zip(self.prepared_asr_encoder_artifact.as_ref())
            .and_then(|(model, prepared)| match &prepared.artifact {
                PreparedAsrEncoderArtifactValue::Lfm25Audio(artifact) => {
                    Some((model, artifact.prompt_tokens))
                }
                _ => None,
            })
            .filter(|_| self.uses_asr_retained_sequence())
            .map(|(model, prompt_tokens)| {
                Self::lfm25_audio_asr_stage_costs(
                    Some(&binding),
                    model,
                    prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let prepared_lfm25_tts_costs = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(
                |ready| match (&ready.model, ready.lfm25_audio_tts.as_ref()) {
                    (PreparedIncrementalModel::Lfm25AudioTts(model), Some(artifact)) => {
                        Some((model, artifact.prompt_tokens))
                    }
                    _ => None,
                },
            )
            .map(|(model, prompt_tokens)| {
                Self::lfm25_audio_tts_stage_costs(
                    Some(&binding),
                    model,
                    prompt_tokens,
                    self.params.max_tokens,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let prepared_tts_continuous_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.qwen_tts.as_ref()) {
                (PreparedIncrementalModel::QwenTts(model), Some(prepared)) => Some((
                    model.model_arc(),
                    prepared.prefill_tokens,
                    prepared.max_frames,
                )),
                (
                    PreparedIncrementalModel::Asr(_) | PreparedIncrementalModel::Lfm25AudioAsr(_),
                    _,
                ) => None,
                _ => None,
            })
            .map(|(model, prefill_tokens, max_frames)| {
                Self::continuous_tts_stage_cost(Some(&binding), &model, prefill_tokens, max_frames)
            })
            .transpose()?
            .flatten();
        let prepared_tts_prefill_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.qwen_tts.as_ref()) {
                (PreparedIncrementalModel::QwenTts(model), Some(prepared)) => Some((
                    model.model_arc(),
                    prepared.prefill_tokens,
                    prepared.max_frames,
                    prepared.reference.is_some(),
                )),
                _ => None,
            })
            .map(|(model, prefill_tokens, max_frames, has_reference)| {
                Self::resumable_tts_prefill_stage_cost(
                    Some(&binding),
                    &model,
                    prefill_tokens,
                    max_frames,
                    has_reference,
                )
            })
            .transpose()?
            .flatten();
        let prepared_kokoro_tts_cost = self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match (&ready.model, ready.kokoro_tts.as_ref()) {
                (PreparedIncrementalModel::KokoroTts(_), Some(prepared)) => Some(prepared),
                _ => None,
            })
            .map(|prepared| Self::kokoro_tts_stage_cost(Some(&binding), prepared))
            .transpose()?
            .flatten();
        self.execution_adapter_binding = Some(binding);
        if let Some((stage_id, cost)) = prepared_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_asr_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        for (stage_id, cost) in prepared_lfm25_asr_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        for (stage_id, cost) in prepared_lfm25_tts_costs {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_tts_continuous_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_tts_prefill_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        if let Some((stage_id, cost)) = prepared_kokoro_tts_cost {
            self.install_prepared_stage_cost(stage_id, cost)?;
        }
        Ok(())
    }

    pub(crate) fn execution_adapter_binding(&self) -> Option<&super::ExecutionAdapterBinding> {
        self.execution_adapter_binding.as_ref()
    }

    pub(crate) fn bind_v2_state_descriptor(
        &mut self,
        descriptor: crate::kv::v2::CapabilityStateDescriptorV2,
        fingerprint: [u8; 32],
    ) -> Result<()> {
        if self
            .v2_state_descriptor
            .as_ref()
            .is_some_and(|current| current != &descriptor)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 contract".to_string(),
            ));
        }
        if self
            .v2_state_fingerprint
            .is_some_and(|current| current != fingerprint)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 fingerprint"
                    .to_string(),
            ));
        }
        if let Some(execution) = &self.execution_adapter_binding {
            let expected = descriptor.fingerprint(&execution.stages)?;
            if expected != fingerprint {
                return Err(Error::InvalidInput(
                    "state ABI v2 fingerprint does not match the bound execution stages"
                        .to_string(),
                ));
            }
        }
        self.v2_state_descriptor = Some(descriptor);
        self.v2_state_fingerprint = Some(fingerprint);
        Ok(())
    }

    pub(crate) fn bind_v2_state_runtime(
        &mut self,
        runtime: Arc<crate::kv::v2::CapabilityStateRuntimeV2>,
        fingerprint: [u8; 32],
        backend: crate::backends::BackendKind,
    ) -> Result<()> {
        if runtime.state_fingerprint != fingerprint {
            return Err(Error::InvalidInput(
                "state ABI v2 runtime fingerprint does not match its capability binding"
                    .to_string(),
            ));
        }
        let execution = self.execution_adapter_binding.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "state ABI v2 runtime requires a bound execution adapter".to_string(),
            )
        })?;
        runtime.validate_against(backend, execution)?;
        if self
            .v2_state_runtime
            .as_ref()
            .is_some_and(|current| current.id != runtime.id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different state ABI v2 runtime".to_string(),
            ));
        }
        if self
            .v2_state_runtime
            .as_ref()
            .is_some_and(|current| current.id == runtime.id)
        {
            return Ok(());
        }
        self.bind_v2_state_descriptor(runtime.descriptor.clone(), fingerprint)?;
        self.v2_state_runtime = Some(runtime);
        Ok(())
    }

    pub(crate) fn v2_state_runtime(&self) -> Option<&Arc<crate::kv::v2::CapabilityStateRuntimeV2>> {
        self.v2_state_runtime.as_ref()
    }

    pub(crate) fn v2_state_descriptor(
        &self,
    ) -> Option<&crate::kv::v2::CapabilityStateDescriptorV2> {
        self.v2_state_descriptor.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn v2_state_fingerprint(&self) -> Option<[u8; 32]> {
        self.v2_state_fingerprint
    }

    pub(crate) fn install_managed_cache_runtime(
        &mut self,
        runtime: Arc<super::cache::managed::ManagedKvModelRuntime>,
    ) -> Result<()> {
        if self.model_instance_id != Some(runtime.plan().model_instance) {
            return Err(Error::InvalidInput(
                "managed-cache runtime does not match the request model instance".to_string(),
            ));
        }
        if self
            .managed_cache_runtime
            .as_ref()
            .is_some_and(|current| current.plan().id != runtime.plan().id)
        {
            return Err(Error::InvalidInput(
                "engine request is already bound to a different managed-cache plan".to_string(),
            ));
        }
        self.managed_cache_runtime = Some(runtime);
        Ok(())
    }

    /// Records the admission-time managed prefix cursor for the first prefill
    /// span. Must be called before the request enters the scheduler.
    pub(crate) fn set_managed_prefix_cursor(&mut self, cursor: Option<u32>) {
        self.managed_prefix_cursor = cursor;
    }

    pub(crate) fn managed_prefix_cursor(&self) -> Option<u32> {
        self.managed_prefix_cursor
    }

    /// Phase 5a: the prepared prompt's stable-history boundary (tokens).
    pub(crate) fn managed_snapshot_boundary(&self) -> Option<u32> {
        self.managed_snapshot_boundary
    }

    pub(crate) fn managed_preamble_boundary(&self) -> Option<u32> {
        self.managed_preamble_boundary
    }

    #[cfg(test)]
    pub(crate) fn set_managed_snapshot_boundary_for_test(&mut self, boundary: Option<u32>) {
        self.managed_snapshot_boundary = boundary;
    }

    pub(crate) fn managed_cache_runtime(
        &self,
    ) -> Option<&Arc<super::cache::managed::ManagedKvModelRuntime>> {
        self.managed_cache_runtime.as_ref()
    }

    fn validate_prepared_stage_costs(&self) -> Result<()> {
        if self.prepared_stage_costs.is_empty() {
            return Ok(());
        }
        let binding = self.execution_adapter_binding.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} carries prepared stage costs without a loaded adapter binding",
                self.id
            ))
        })?;
        let mut stage_ids = std::collections::HashSet::new();
        for prepared in &self.prepared_stage_costs {
            if !stage_ids.insert(prepared.stage_id) {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries duplicate prepared costs for one stage",
                    self.id
                )));
            }
            let stage = binding
                .stages
                .iter()
                .find(|stage| stage.id == prepared.stage_id)
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "Request {} prepared an unknown adapter stage",
                        self.id
                    ))
                })?;
            if stage.batch_mode == super::NativeBatchMode::None
                || prepared.cost.logical_units == 0
                || prepared.cost.tensor_elements == 0
                || !prepared
                    .cost
                    .workspace
                    .workspace_bytes()
                    .is_ok_and(|bytes| bytes <= stage.max_workspace_bytes)
            {
                return Err(Error::InvalidInput(format!(
                    "Request {} carries an invalid prepared tensor-stage cost",
                    self.id
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn install_prepared_stage_cost(
        &mut self,
        stage_id: StageId,
        cost: WorkCost,
    ) -> Result<()> {
        self.prepared_stage_costs
            .retain(|prepared| prepared.stage_id != stage_id);
        self.prepared_stage_costs
            .push(PreparedStageCost { stage_id, cost });
        self.validate_prepared_stage_costs()
    }

    pub(crate) fn prepared_stage_cost(&self, stage_id: StageId) -> Option<WorkCost> {
        self.prepared_stage_costs
            .iter()
            .find(|prepared| prepared.stage_id == stage_id)
            .map(|prepared| prepared.cost)
    }

    pub(crate) fn mark_audio_streaming_only(&mut self) {
        self.collect_audio_samples = false;
        if self.audio_stream_engine_queue_capacity.is_none() {
            self.audio_stream_engine_queue_capacity =
                Some(super::Engine::streaming_queue_capacity(self));
        }
    }

    /// Price all independently queued PCM, not only the credited codec outbox.
    pub(crate) fn streaming_audio_buffer_bytes(&self, max_chunk_samples: usize) -> Result<u64> {
        let engine_rows = self.audio_stream_engine_queue_capacity.ok_or_else(|| {
            Error::InvalidInput(
                "streaming audio queue capacity was not frozen before admission".into(),
            )
        })?;
        let chunk_bytes = max_chunk_samples
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(self.id.len()))
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| Error::Overloaded("streaming chunk reservation overflow".into()))?;
        let rows = engine_rows
            .checked_add(self.audio_stream_external_queue_capacity)
            .and_then(|rows| rows.checked_add(3))
            .ok_or_else(|| Error::Overloaded("streaming queue reservation overflow".into()))?;
        let bytes = rows
            .checked_mul(chunk_bytes)
            .and_then(|bytes| bytes.checked_add(STREAM_PROGRESS_MAX_BUFFERED_BYTES))
            .ok_or_else(|| Error::Overloaded("streaming PCM reservation overflow".into()))?;
        u64::try_from(bytes)
            .map_err(|_| Error::Overloaded("streaming PCM reservation exceeds u64".into()))
    }

    /// Nonblocking pre-codec admission. None means this row should yield while
    /// peers progress; no codec history or samples may be materialized yet.
    pub(crate) fn try_reserve_audio_output(
        &self,
        sample_bytes: usize,
    ) -> Result<Option<AudioOutputCredit>> {
        let bytes = sample_bytes
            .checked_add(self.id.len())
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or_else(|| Error::Overloaded("Audio output credit overflow".into()))?;
        let mut state = self
            .stream_staging
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".into()))?;
        let budget = state
            .audio_budget
            .get_or_insert_with(|| StreamProgressBudget::new(4 * 1024 * 1024))
            .clone();
        let Some(local) = budget.reserve(bytes, EngineStreamPolicy::DropNewest)? else {
            return Ok(None);
        };
        let Some(global) =
            committed_audio_budget().reserve(bytes, EngineStreamPolicy::DropNewest)?
        else {
            return Ok(None);
        };
        let credit = AudioOutputCredit {
            _permits: Arc::new(vec![local, global]),
            bytes,
        };
        state.audio_credits.push(credit.clone());
        Ok(Some(credit))
    }

    pub(super) fn take_audio_output_credits(&self) -> Vec<AudioOutputCredit> {
        let mut state = self
            .stream_staging
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        std::mem::take(&mut state.audio_credits)
    }

    #[cfg(test)]
    pub(super) fn begin_stream_staging(&self) -> Result<()> {
        self.stream_staging.clear()
    }

    pub(super) fn bind_stream_quantum(
        &self,
        batch_id: BatchId,
        lane: BatchLaneKey,
        plan_id: PlanId,
        session: SessionKey,
        visibility: OutputVisibility,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        budget: Arc<StreamProgressBudget>,
    ) -> Result<StreamBindingGuard> {
        self.stream_staging.bind_quantum(
            batch_id,
            lane,
            plan_id,
            session,
            visibility,
            progress_tx,
            budget,
        )
    }

    pub(super) fn stream_staging_buffer(&self) -> StreamStagingBuffer {
        self.stream_staging.clone()
    }

    pub(super) fn take_staged_stream_outputs(&self) -> Result<Vec<StreamingOutput>> {
        self.stream_staging.take()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|signal| signal.load(Ordering::Acquire))
    }

    /// Set model variant.
    pub fn with_model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.model_variant = Some(model_variant);
        self
    }

    /// Set generation parameters.
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set an absolute end-to-end deadline.
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Set workload class.
    pub fn with_workload_class(mut self, workload_class: WorkloadClass) -> Self {
        self.workload_class = workload_class;
        self
    }

    /// Enable streaming.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        if streaming && self.workload_class == WorkloadClass::Online {
            self.workload_class = WorkloadClass::Streaming;
        }
        self
    }

    /// Set streaming backpressure policy.
    pub fn with_stream_policy(mut self, stream_policy: EngineStreamPolicy) -> Self {
        self.stream_policy = stream_policy;
        self
    }

    /// Set voice/speaker.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.params.voice = Some(voice.into());
        self
    }

    /// Set reference audio for voice cloning.
    pub fn with_reference(mut self, audio: impl Into<String>, text: impl Into<String>) -> Self {
        self.reference_audio = Some(audio.into());
        self.reference_text = Some(text.into());
        self.sync_task_from_fields();
        self
    }

    /// Set voice description.
    pub fn with_voice_description(mut self, description: impl Into<String>) -> Self {
        self.voice_description = Some(description.into());
        self.sync_task_from_fields();
        self
    }

    /// Set language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self.sync_task_from_fields();
        self
    }

    /// Ask for model-native word timestamps where the model emits them.
    pub fn with_asr_native_word_timestamps(mut self, enabled: bool) -> Self {
        self.asr_native_word_timestamps = enabled;
        self
    }

    /// Set ASR initial prompt/context hint.
    pub fn with_asr_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        let trimmed = prompt.trim();
        self.asr_prompt = (!trimmed.is_empty()).then_some(trimmed.to_string());
        self.sync_task_from_fields();
        self
    }

    /// Set request correlation ID.
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }

    /// Set speech-to-speech system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self.sync_task_from_fields();
        self
    }

    /// Set chat-specific prompt/runtime configuration.
    pub fn with_chat_config(mut self, chat_config: ChatRequestConfig) -> Self {
        self.chat_config = chat_config;
        self.sync_task_from_fields();
        self
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

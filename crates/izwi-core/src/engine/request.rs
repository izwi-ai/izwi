//! Request types and processing for the inference engine.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::config::EngineCoreConfig;
use super::output::StreamingOutput;
use super::types::{GenerationParams, Priority, RequestId, TaskType, TokenId};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::architectures::qwen35::chat::Qwen35PreparedPrompt;
use crate::models::registry::{
    AsrModelLease, ChatModelLease, NativeAsrModel, NativeChatModel, QwenTtsModelLease,
};
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage, ChatRequestConfig};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineStreamPolicy {
    FailOnFull,
    BlockWithDeadline {
        timeout_ms: u64,
    },
    /// Drop the newly produced output when the bounded queue is full.
    DropNewest,
}

impl Default for EngineStreamPolicy {
    fn default() -> Self {
        Self::FailOnFull
    }
}

/// Coarse workload class used by admission and latency-aware scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadClass {
    /// Realtime voice or transcription work where first output latency matters most.
    Realtime,
    /// Interactive non-streaming user work.
    Interactive,
    /// User-visible streaming work such as chat or TTS over SSE/websocket.
    Streaming,
    /// Default online API work.
    Online,
    /// Offline batch jobs.
    Batch,
    /// Opportunistic background work.
    Background,
}

impl Default for WorkloadClass {
    fn default() -> Self {
        Self::Online
    }
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
    prepared_qwen35_prompt: Option<Qwen35PreparedPrompt>,
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
    QwenTts(QwenTtsModelLease),
}

impl fmt::Debug for PreparedIncrementalModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asr(model) => write!(formatter, "Asr({:p})", &**model),
            Self::QwenTts(model) => write!(formatter, "QwenTts({:p})", &**model),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct IncrementalModelExecutionReady {
    model_variant: ModelVariant,
    model: PreparedIncrementalModel,
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
    /// Whether ASR max_tokens was filled by a model-specific automatic heuristic.
    pub asr_auto_max_tokens: bool,
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
    /// Optional absolute end-to-end deadline supplied by the ingress layer.
    pub deadline: Option<Instant>,
    /// Prompt token IDs (set by processor)
    pub prompt_tokens: Vec<TokenId>,
    /// Private authorization installed only after exact model prompt preparation.
    pub(super) chat_execution_ready: Option<ChatExecutionReady>,
    /// Exact ASR/Qwen-TTS model identity retained for standalone incremental
    /// execution. Public fields cannot manufacture this lifecycle fence.
    pub(super) incremental_model_execution_ready: Option<IncrementalModelExecutionReady>,
    /// Enable streaming output
    pub streaming: bool,
    /// Backpressure behavior for streaming output.
    pub stream_policy: EngineStreamPolicy,
    /// Channel for streaming output (internal use)
    #[allow(dead_code)]
    pub(crate) streaming_tx: Option<mpsc::Sender<StreamingOutput>>,
    /// Cooperative cancellation signal set without waiting for the core write lock.
    pub(crate) cancellation: Option<Arc<AtomicBool>>,
}

impl EngineCoreRequest {
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
        (!audio.is_empty()).then(|| match audio {
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
            .or_else(|| match (&self.task, self.task_type) {
                (EngineTask::Asr(input), TaskType::ASR) => input.language.as_ref(),
                _ => None,
            })
    }

    fn selected_asr_prompt(&self) -> Option<&String> {
        self.asr_prompt
            .as_ref()
            .or_else(|| match (&self.task, self.task_type) {
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
            .or_else(|| match (&self.task, self.task_type) {
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
        let prompt_tokens = self.prompt_tokens.len();
        if max_seq_len == 0 || prompt_tokens >= max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Chat request {} exact prompt has {prompt_tokens} tokens and leaves no output capacity in the configured {max_seq_len}-token context",
                self.id
            )));
        }
        self.params.max_tokens = self
            .params
            .max_tokens
            .max(1)
            .min(max_seq_len - prompt_tokens);
        Ok(())
    }

    /// Install the private scheduler/executor authorization for an exact chat
    /// prompt. Callers must supply tokens produced from this request's current
    /// messages/configuration by the loaded model adapter.
    pub(crate) fn install_chat_execution_preparation_with_model(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_qwen35_prompt: Option<Qwen35PreparedPrompt>,
        model: ChatModelLease,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_qwen35_prompt,
            PreparedChatModel::Exact(model),
        )
    }

    #[cfg(test)]
    pub(crate) fn install_chat_execution_preparation(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_qwen35_prompt: Option<Qwen35PreparedPrompt>,
    ) -> Result<()> {
        self.install_chat_execution_preparation_inner(
            model_variant,
            prompt_tokens,
            prepared_qwen35_prompt,
            PreparedChatModel::ValidationOnly,
        )
    }

    fn install_chat_execution_preparation_inner(
        &mut self,
        model_variant: ModelVariant,
        prompt_tokens: Vec<TokenId>,
        prepared_qwen35_prompt: Option<Qwen35PreparedPrompt>,
        model: PreparedChatModel,
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

        let is_qwen35 = model_variant.family() == ModelFamily::Qwen35Chat;
        match prepared_qwen35_prompt.as_ref() {
            Some(_) if !is_qwen35 => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} routed a Qwen3.5 prompt artifact to {model_variant}",
                    self.id
                )));
            }
            Some(prepared) if prepared.prompt_ids() != prompt_tokens => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} Qwen3.5 prompt artifact does not match its exact prompt tokens",
                    self.id
                )));
            }
            None if is_qwen35 => {
                return Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared Qwen3.5 prompt artifact",
                    self.id
                )));
            }
            _ => {}
        }

        if prepared_qwen35_prompt.is_some() {
            // The opaque artifact owns all decoded/encoded vision state needed
            // by execution. Do not retain or repeatedly fingerprint multi-MB
            // data URLs (or signed remote URLs) on the scheduler hot path.
            for media in &mut self.chat_config.media_inputs {
                media.source.clear();
                media.source.shrink_to_fit();
            }
        }

        let fingerprint =
            chat_execution_fingerprint(model_variant, messages, &self.chat_config, &prompt_tokens)?;
        self.prompt_tokens = prompt_tokens;
        self.sync_task_from_fields();
        self.chat_execution_ready = Some(ChatExecutionReady {
            model_variant,
            model,
            fingerprint,
            prepared_qwen35_prompt,
            core_validated: false,
        });
        Ok(())
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

        match ready.prepared_qwen35_prompt.as_ref() {
            Some(prepared)
                if model_variant.family() != ModelFamily::Qwen35Chat
                    || prepared.prompt_ids() != self.prompt_tokens =>
            {
                Err(Error::InvalidInput(format!(
                    "Chat request {} has a mismatched Qwen3.5 prompt artifact",
                    self.id
                )))
            }
            None if model_variant.family() == ModelFamily::Qwen35Chat => {
                Err(Error::InvalidInput(format!(
                    "Chat request {} is missing its prepared Qwen3.5 prompt artifact",
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
        self.validate_incremental_model_execution_preparation()
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

    pub(crate) fn prepared_qwen35_prompt_for_executor(
        &self,
    ) -> Result<Option<&Qwen35PreparedPrompt>> {
        self.validate_chat_execution_for_executor()?;
        Ok(self
            .chat_execution_ready
            .as_ref()
            .and_then(|ready| ready.prepared_qwen35_prompt.as_ref()))
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
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::Asr(model),
        });
        Ok(())
    }

    pub(crate) fn install_qwen_tts_execution_model(
        &mut self,
        model_variant: ModelVariant,
        model: QwenTtsModelLease,
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
        self.incremental_model_execution_ready = Some(IncrementalModelExecutionReady {
            model_variant,
            model: PreparedIncrementalModel::QwenTts(model),
        });
        Ok(())
    }

    fn validate_incremental_model_execution_preparation(&self) -> Result<()> {
        let Some(ready) = self.incremental_model_execution_ready.as_ref() else {
            return Ok(());
        };
        if self.model_variant != Some(ready.model_variant) {
            return Err(Error::InvalidInput(format!(
                "Request {} changed model after incremental model preparation",
                self.id
            )));
        }
        match (&ready.model, self.task_type) {
            (PreparedIncrementalModel::Asr(_), TaskType::ASR) => Ok(()),
            (PreparedIncrementalModel::QwenTts(_), TaskType::TTS)
                if ready.model_variant.family() == ModelFamily::Qwen3Tts =>
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
                PreparedIncrementalModel::QwenTts(_) => None,
            }))
    }

    pub(crate) fn prepared_asr_model_lease_for_executor(&self) -> Result<Option<AsrModelLease>> {
        self.validate_incremental_model_execution_preparation()?;
        Ok(self
            .incremental_model_execution_ready
            .as_ref()
            .and_then(|ready| match &ready.model {
                PreparedIncrementalModel::Asr(model) => Some(model.clone()),
                PreparedIncrementalModel::QwenTts(_) => None,
            }))
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
                PreparedIncrementalModel::Asr(_) => None,
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
                PreparedIncrementalModel::Asr(_) => None,
            }))
    }

    pub(crate) fn chat_generation_config(&self) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: self.params.temperature.max(0.0),
            top_p: self.params.top_p.clamp(0.0, 1.0),
            top_k: self.params.top_k,
            repetition_penalty: self.params.repetition_penalty.max(1.0),
            presence_penalty: self.params.presence_penalty.clamp(-2.0, 2.0),
            stop_sequences: self.params.stop_sequences.clone(),
            stop_token_ids: self.params.stop_token_ids.clone(),
            seed: Self::chat_request_seed(&self.id),
            request: self.chat_config.clone(),
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
            text: Some(text),
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
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
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
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
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
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
            text: None,
            chat_messages: Some(messages),
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
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
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64),
            audio_bytes: None,
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
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
            text: None,
            chat_messages: None,
            chat_config: ChatRequestConfig::default(),
            language: None,
            correlation_id: None,
            audio_input: None,
            audio_bytes: Some(audio_bytes),
            asr_prompt: None,
            asr_auto_max_tokens: false,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            arrival_time: Instant::now(),
            deadline: None,
            prompt_tokens: Vec::new(),
            chat_execution_ready: None,
            incremental_model_execution_ready: None,
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
            cancellation: None,
        }
    }

    pub(crate) fn set_cancellation_signal(&mut self, signal: Arc<AtomicBool>) {
        self.cancellation = Some(signal);
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
        if !self.prompt_tokens.is_empty() {
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

        // Validate and clamp parameters
        self.validate_params(
            request.task_type,
            request.model_variant,
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
                    if let Some(tts_limit) =
                        model_variant.and_then(|variant| variant.tts_max_output_frames_hint())
                    {
                        params.max_tokens.min(tts_limit)
                    } else {
                        params.max_tokens.min(self.config.max_seq_len)
                    }
                }
                _ => params.max_tokens.min(self.config.max_seq_len),
            };
        }

        // Clamp speed
        params.speed = params.speed.clamp(0.5, 2.0);

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
pub struct RequestBuilder {
    request: EngineCoreRequest,
}

impl RequestBuilder {
    /// Create a new TTS request builder.
    pub fn tts(text: impl Into<String>) -> Self {
        Self {
            request: EngineCoreRequest::tts(text),
        }
    }

    /// Create a new ASR request builder.
    pub fn asr(audio_base64: impl Into<String>) -> Self {
        Self {
            request: EngineCoreRequest::asr(audio_base64),
        }
    }

    /// Create a new ASR request builder from audio bytes.
    pub fn asr_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            request: EngineCoreRequest::asr_bytes(audio_bytes),
        }
    }

    /// Create a new chat request builder.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            request: EngineCoreRequest::chat(messages),
        }
    }

    /// Create a new speech-to-speech request builder.
    pub fn speech_to_speech(audio_base64: impl Into<String>) -> Self {
        Self {
            request: EngineCoreRequest::speech_to_speech(audio_base64),
        }
    }

    /// Create a new speech-to-speech request builder from audio bytes.
    pub fn speech_to_speech_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            request: EngineCoreRequest::speech_to_speech_bytes(audio_bytes),
        }
    }

    /// Set the request ID.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.request.id = id.into();
        self
    }

    /// Set model variant.
    pub fn model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.request.model_variant = Some(model_variant);
        self
    }

    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.request.params.voice = Some(voice.into());
        self
    }

    /// Set the speaker (alias for voice).
    pub fn speaker(mut self, speaker: impl Into<String>) -> Self {
        self.request.params.speaker = Some(speaker.into());
        self
    }

    /// Set reference audio and text for voice cloning.
    pub fn reference(mut self, audio: impl Into<String>, text: impl Into<String>) -> Self {
        self.request.reference_audio = Some(audio.into());
        self.request.reference_text = Some(text.into());
        self
    }

    /// Set voice description.
    pub fn voice_description(mut self, description: impl Into<String>) -> Self {
        self.request.voice_description = Some(description.into());
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.request.params.temperature = temp;
        self
    }

    /// Set top_p.
    pub fn top_p(mut self, p: f32) -> Self {
        self.request.params.top_p = p;
        self
    }

    /// Set top_k.
    pub fn top_k(mut self, k: usize) -> Self {
        self.request.params.top_k = k;
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.request.params.max_tokens = max;
        self
    }

    /// Set audio temperature.
    pub fn audio_temperature(mut self, temp: f32) -> Self {
        self.request.params.audio_temperature = Some(temp);
        self
    }

    /// Set audio top_k.
    pub fn audio_top_k(mut self, k: usize) -> Self {
        self.request.params.audio_top_k = Some(k);
        self
    }

    /// Set priority.
    pub fn priority(mut self, priority: Priority) -> Self {
        self.request.priority = priority;
        self
    }

    /// Set workload class.
    pub fn workload_class(mut self, workload_class: WorkloadClass) -> Self {
        self.request.workload_class = workload_class;
        self
    }

    /// Enable streaming.
    pub fn streaming(mut self) -> Self {
        self.request.streaming = true;
        if self.request.workload_class == WorkloadClass::Online {
            self.request.workload_class = WorkloadClass::Streaming;
        }
        self
    }

    /// Set audio input (for ASR/chat).
    pub fn audio_input(mut self, audio: impl Into<String>) -> Self {
        self.request.audio_input = Some(audio.into());
        self.request.audio_bytes = None;
        self.request.sync_task_from_fields();
        self
    }

    /// Set audio input bytes for first-party ASR/speech routes.
    pub fn audio_bytes(mut self, audio: impl Into<Vec<u8>>) -> Self {
        self.request.audio_bytes = Some(audio.into());
        self.request.audio_input = None;
        self.request.sync_task_from_fields();
        self
    }

    /// Set text input (for chat).
    pub fn text_input(mut self, text: impl Into<String>) -> Self {
        self.request.text = Some(text.into());
        self.request.sync_task_from_fields();
        self
    }

    /// Set chat messages.
    pub fn chat_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.request.chat_messages = Some(messages);
        self.request.sync_task_from_fields();
        self
    }

    /// Set language hint.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.request.language = Some(language.into());
        self.request.sync_task_from_fields();
        self
    }

    /// Set ASR initial prompt/context hint.
    pub fn asr_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        let trimmed = prompt.trim();
        self.request.asr_prompt = (!trimmed.is_empty()).then_some(trimmed.to_string());
        self.request.sync_task_from_fields();
        self
    }

    /// Set speech-to-speech system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.system_prompt = Some(prompt.into());
        self.request.sync_task_from_fields();
        self
    }

    /// Build the request.
    pub fn build(mut self) -> EngineCoreRequest {
        self.request.sync_task_from_fields();
        self.request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelVariant;
    use crate::models::shared::chat::ChatRole;

    #[test]
    fn test_tts_request() {
        let request = EngineCoreRequest::tts("Hello, world!");
        assert_eq!(request.task_type, TaskType::TTS);
        assert_eq!(request.workload_class, WorkloadClass::Online);
        assert_eq!(request.text.as_deref(), Some("Hello, world!"));
        match &request.task {
            EngineTask::Tts(input) => assert!(input.text.is_empty()),
            other => panic!("unexpected task payload: {other:?}"),
        }

        let processed = RequestProcessor::new(EngineCoreConfig::default())
            .process(request)
            .expect("TTS request should canonicalize");
        match &processed.task {
            EngineTask::Tts(input) => assert_eq!(input.text, "Hello, world!"),
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn direct_request_constructors_move_large_inputs_without_cloning() {
        let mut audio = Vec::with_capacity(64);
        audio.extend_from_slice(&[1, 2, 3]);
        let audio_ptr = audio.as_ptr();
        let audio_capacity = audio.capacity();
        let request = EngineCoreRequest::asr_bytes(audio);
        let retained_audio = request.audio_bytes.as_ref().expect("compatibility audio");
        assert_eq!(retained_audio.as_ptr(), audio_ptr);
        assert_eq!(retained_audio.capacity(), audio_capacity);
        assert!(matches!(
            &request.task,
            EngineTask::Asr(AsrEngineInput {
                audio: EngineAudioInput::Bytes(bytes),
                ..
            }) if bytes.capacity() == 0
        ));

        let mut text = String::with_capacity(64);
        text.push_str("bounded later");
        let text_ptr = text.as_ptr();
        let request = EngineCoreRequest::tts(text);
        assert_eq!(
            request.text.as_ref().expect("compatibility text").as_ptr(),
            text_ptr
        );
        assert!(matches!(
            &request.task,
            EngineTask::Tts(TtsEngineInput { text, .. }) if text.capacity() == 0
        ));

        let mut reference = String::with_capacity(64);
        reference.push_str("AQID");
        let reference_ptr = reference.as_ptr();
        let request = EngineCoreRequest::tts("hello").with_reference(reference, "transcript");
        assert_eq!(
            request
                .reference_audio
                .as_ref()
                .expect("compatibility reference")
                .as_ptr(),
            reference_ptr
        );
        assert!(matches!(
            &request.task,
            EngineTask::Tts(TtsEngineInput {
                reference_audio: None,
                ..
            })
        ));
    }

    #[test]
    fn direct_payload_preflight_rejects_before_moving_or_cloning() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 3,
            reference_source_bytes: 3,
            reference_text_chars: 64,
            metadata_bytes: 64,
            collection_items: 64,
        };
        let mut asr = EngineCoreRequest::asr_bytes(vec![1, 2, 3, 4]);
        let asr_ptr = asr.audio_bytes.as_ref().unwrap().as_ptr();
        let error = asr
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("oversized bytes must fail before canonicalization");
        assert!(error.to_string().contains("source limit"));
        assert_eq!(asr.audio_bytes.as_ref().unwrap().as_ptr(), asr_ptr);
        assert!(matches!(
            &asr.task,
            EngineTask::Asr(AsrEngineInput {
                audio: EngineAudioInput::Bytes(bytes),
                ..
            }) if bytes.is_empty()
        ));

        let mut qwen_reference = EngineCoreRequest::tts("hello")
            .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
            .with_reference("AAAAAAAA", "reference transcript");
        let reference_ptr = qwen_reference.reference_audio.as_ref().unwrap().as_ptr();
        let error = qwen_reference
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("oversized Qwen reference must fail before canonicalization");
        assert!(error.to_string().contains("3-byte limit"));
        assert_eq!(
            qwen_reference.reference_audio.as_ref().unwrap().as_ptr(),
            reference_ptr
        );
        assert!(matches!(
            &qwen_reference.task,
            EngineTask::Tts(TtsEngineInput {
                reference_audio: None,
                ..
            })
        ));
    }

    #[test]
    fn direct_payload_preflight_rejects_whitespace_amplification_for_audio_and_reference() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 3,
            reference_source_bytes: 3,
            reference_text_chars: 64,
            metadata_bytes: 64,
            collection_items: 64,
        };
        // Three decoded bytes permit four base64 bytes plus the explicit
        // 1-KiB metadata/whitespace allowance, never unbounded whitespace.
        let amplified = " ".repeat(4 + 1024 + 1);
        let mut asr = EngineCoreRequest::asr(amplified.clone());
        let error = asr
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("inference audio whitespace amplification must fail");
        assert!(error.to_string().contains("encoded input limit"));

        let mut qwen_reference = EngineCoreRequest::tts("hello")
            .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
            .with_reference(amplified, "reference transcript");
        let error = qwen_reference
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("reference audio whitespace amplification must fail");
        assert!(error.to_string().contains("encoded input limit"));
    }

    #[test]
    fn direct_payload_canonicalization_preserves_legacy_precedence_and_single_ownership() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 64,
            metadata_bytes: 128,
            collection_items: 64,
        };
        let mut legacy_wins = EngineCoreRequest::asr("AQID");
        legacy_wins.language = Some("legacy-language".to_string());
        legacy_wins.asr_prompt = Some("legacy prompt".to_string());
        legacy_wins.task = EngineTask::Asr(AsrEngineInput {
            audio: EngineAudioInput::Bytes(vec![9, 9, 9]),
            language: Some("typed-language".to_string()),
            prompt: Some("typed prompt".to_string()),
        });
        legacy_wins
            .canonicalize_direct_payloads_with_limits(limits)
            .expect("legacy base64 should canonicalize");
        assert!(legacy_wins.audio_input.is_none());
        assert!(legacy_wins.audio_bytes.is_none());
        assert_eq!(legacy_wins.audio_base64_for_execution(), Some("AQID"));
        assert!(matches!(
            &legacy_wins.task,
            EngineTask::Asr(AsrEngineInput {
                audio: EngineAudioInput::Base64(value),
                language: Some(language),
                prompt: Some(prompt),
                ..
            }) if value == "AQID"
                && language == "legacy-language"
                && prompt == "legacy prompt"
        ));

        let mut bytes_win = EngineCoreRequest::asr("AQID");
        bytes_win.audio_bytes = Some(vec![1, 2, 3]);
        bytes_win
            .canonicalize_direct_payloads_with_limits(limits)
            .expect("legacy bytes should have historical precedence");
        assert!(bytes_win.audio_input.is_none());
        assert!(bytes_win.audio_bytes.is_none());
        assert_eq!(bytes_win.audio_bytes_for_execution(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn direct_payload_canonicalization_preserves_typed_only_asr_and_speech_metadata() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 64,
            metadata_bytes: 256,
            collection_items: 64,
        };
        let mut asr = EngineCoreRequest::asr_bytes(vec![1]);
        asr.audio_bytes = None;
        asr.task = EngineTask::Asr(AsrEngineInput {
            audio: EngineAudioInput::Bytes(vec![1, 2, 3]),
            language: Some("typed-language".to_string()),
            prompt: Some("typed prompt".to_string()),
        });
        asr.canonicalize_direct_payloads_with_limits(limits)
            .expect("typed-only ASR request should canonicalize");
        assert_eq!(asr.asr_language_for_execution(), Some("typed-language"));
        assert_eq!(asr.asr_prompt_for_execution(), Some("typed prompt"));
        assert_eq!(asr.audio_bytes_for_execution(), Some(&[1, 2, 3][..]));

        let typed_messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "typed history".to_string(),
        }];
        let mut speech = EngineCoreRequest::speech_to_speech_bytes(vec![1]);
        speech.audio_bytes = None;
        speech.task = EngineTask::SpeechToSpeech(AudioChatEngineInput {
            audio: EngineAudioInput::Bytes(vec![4, 5, 6]),
            messages: typed_messages.clone(),
            system_prompt: Some("typed system".to_string()),
        });
        speech
            .canonicalize_direct_payloads_with_limits(limits)
            .expect("typed-only speech request should canonicalize");
        assert!(EngineCoreRequest::chat_messages_match(
            speech.speech_messages_for_execution(),
            &typed_messages
        ));
        assert_eq!(
            speech.speech_system_prompt_for_execution(),
            Some("typed system")
        );

        speech.chat_messages = Some(vec![ChatMessage {
            role: ChatRole::Assistant,
            content: "legacy history".to_string(),
        }]);
        speech.system_prompt = Some("legacy system".to_string());
        speech
            .canonicalize_direct_payloads_with_limits(limits)
            .expect("legacy speech metadata should retain precedence");
        assert_eq!(
            speech.speech_messages_for_execution()[0].content,
            "legacy history"
        );
        assert_eq!(
            speech.speech_system_prompt_for_execution(),
            Some("legacy system")
        );
    }

    #[test]
    fn qwen_reference_canonicalization_synchronizes_typed_execution_view() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 64,
            metadata_bytes: 128,
            collection_items: 64,
        };
        let mut request = EngineCoreRequest::tts("hello")
            .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
            .with_reference("AQID", "reference transcript");
        request
            .canonicalize_direct_payloads_with_limits(limits)
            .expect("Qwen reference should canonicalize");
        assert!(request.reference_audio.is_none());
        assert!(request.reference_text.is_none());
        assert_eq!(request.tts_reference_audio_for_execution(), Some("AQID"));
        assert_eq!(
            request.tts_reference_text_for_execution(),
            Some("reference transcript")
        );
        match &request.task {
            EngineTask::Tts(input) => {
                assert_eq!(input.text, "hello");
                assert_eq!(input.reference_audio.as_deref(), Some("AQID"));
                assert_eq!(
                    input.reference_text.as_deref(),
                    Some("reference transcript")
                );
            }
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn direct_non_chat_metadata_has_one_checked_context_derived_bound() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 8,
            metadata_bytes: 8,
            collection_items: 1,
        };

        let mut tts = EngineCoreRequest::tts("123456789");
        assert!(tts
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("oversized TTS metadata must fail")
            .to_string()
            .contains("TTS metadata"));

        let mut asr = EngineCoreRequest::asr_bytes(vec![1])
            .with_language("12345")
            .with_asr_prompt("6789");
        assert!(asr
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("aggregate ASR metadata must fail")
            .to_string()
            .contains("ASR metadata"));

        let mut speech = EngineCoreRequest::speech_to_speech_bytes(vec![1]);
        speech.chat_messages = Some(vec![ChatMessage {
            role: ChatRole::User,
            content: "123456789".to_string(),
        }]);
        assert!(speech
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("aggregate speech metadata must fail")
            .to_string()
            .contains("speech-to-speech metadata"));
    }

    #[test]
    fn direct_dynamic_allocations_have_aggregate_and_count_bounds() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 64,
            metadata_bytes: 64,
            collection_items: 2,
        };

        let mut oversized_voice = EngineCoreRequest::tts("ok");
        oversized_voice.params.voice = Some("v".repeat(65));
        assert!(oversized_voice
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("voice allocation must share the metadata bound")
            .to_string()
            .contains("request metadata"));

        let mut excessive_stops = EngineCoreRequest::tts("ok");
        excessive_stops.params.stop_sequences = vec![String::new(); 3];
        assert!(excessive_stops
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("stop-sequence count must be bounded")
            .to_string()
            .contains("stop sequences"));

        let mut excessive_prompt_tokens = EngineCoreRequest::asr_bytes(vec![1]);
        excessive_prompt_tokens.prompt_tokens = vec![1, 2, 3];
        assert!(excessive_prompt_tokens
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("non-chat prompt-token count must be bounded")
            .to_string()
            .contains("prompt tokens"));
    }

    #[test]
    fn direct_preflight_rejects_small_payloads_with_oversized_retained_capacity() {
        let limits = DirectPayloadLimits {
            audio_source_bytes: 64,
            reference_source_bytes: 64,
            reference_text_chars: 64,
            metadata_bytes: 64,
            collection_items: 64,
        };

        let mut text = String::with_capacity(1024);
        text.push_str("ok");
        let mut request = EngineCoreRequest::tts(text);
        assert!(request
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("small text with oversized retained capacity must fail")
            .to_string()
            .contains("TTS metadata"));

        let mut audio = Vec::with_capacity(1024);
        audio.push(1);
        let mut request = EngineCoreRequest::asr_bytes(audio);
        assert!(request
            .canonicalize_direct_payloads_with_limits(limits)
            .expect_err("small audio with oversized retained capacity must fail")
            .to_string()
            .contains("retains"));
    }

    #[test]
    fn streaming_request_defaults_to_streaming_workload_class() {
        let request = EngineCoreRequest::tts("Hello").with_streaming(true);
        assert_eq!(request.workload_class, WorkloadClass::Streaming);

        let request = RequestBuilder::tts("Hello")
            .workload_class(WorkloadClass::Realtime)
            .streaming()
            .build();
        assert_eq!(request.workload_class, WorkloadClass::Realtime);
    }

    #[test]
    fn test_request_builder() {
        let request = RequestBuilder::tts("Hello")
            .voice("us_female")
            .temperature(0.8)
            .max_tokens(1024)
            .streaming()
            .build();

        assert!(request.streaming);
        assert_eq!(request.params.temperature, 0.8);
        assert_eq!(request.params.max_tokens, 1024);
    }

    #[test]
    fn test_request_processor() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let request = EngineCoreRequest::tts("Test");
        let processed = processor.process(request);
        assert!(processed.is_ok());
    }

    #[test]
    fn test_request_processor_preserves_tts_auto_max_tokens() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::tts("Test");
        request.params.max_tokens = 0;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.params.max_tokens, 0);
    }

    #[test]
    fn test_request_processor_clamps_tts_to_model_native_limit() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::tts("Test");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        request.params.max_tokens = 20_000;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(
            processed.params.max_tokens,
            ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES
        );
    }

    #[test]
    fn test_request_processor_keeps_tts_above_engine_seq_len_when_model_allows() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::tts("Test");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
        request.params.max_tokens = 5000;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.params.max_tokens, 5000);
    }

    #[test]
    fn test_request_processor_defaults_chat_max_tokens() {
        let config = EngineCoreConfig::default();
        let expected_default = 2048usize.min(config.max_seq_len);
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.params.max_tokens = 0;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.params.max_tokens, expected_default);
    }

    #[test]
    fn direct_chat_preparation_rejects_oversized_and_deep_inputs_before_tokenization() {
        let oversized = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "x".repeat(DIRECT_CHAT_MIN_INPUT_BYTES + 1),
        }]);
        assert!(matches!(
            oversized.validate_direct_chat_preparation_input(8),
            Err(Error::InvalidInput(message)) if message.contains("preparation input exceeds")
        ));

        let mut nested = serde_json::Value::Null;
        for _ in 0..=DIRECT_CHAT_MAX_TOOL_JSON_DEPTH {
            nested = serde_json::Value::Array(vec![nested]);
        }
        let deep = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "tool call".to_string(),
        }])
        .with_chat_config(ChatRequestConfig {
            tools: vec![nested],
            ..ChatRequestConfig::default()
        });
        assert!(matches!(
            deep.validate_direct_chat_preparation_input(4096),
            Err(Error::InvalidInput(message)) if message.contains("nesting limit")
        ));
    }

    #[test]
    fn exact_chat_context_rejects_full_prompts_and_clamps_output_to_remaining_tokens() {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.params.max_tokens = 100;
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3], None)
            .unwrap();
        request
            .enforce_chat_context_window(5)
            .expect("two output tokens remain");
        assert_eq!(request.params.max_tokens, 2);

        let mut full = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        full.install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3, 4, 5], None)
            .unwrap();
        assert!(matches!(
            full.enforce_chat_context_window(5),
            Err(Error::InvalidInput(message)) if message.contains("leaves no output capacity")
        ));
    }

    #[test]
    fn test_request_processor_syncs_but_does_not_authorize_public_prompt_tokens() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.prompt_tokens = vec![41, 42, 43];

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.prompt_tokens, vec![41, 42, 43]);
        match &processed.task {
            EngineTask::Chat(input) => assert_eq!(input.prompt_tokens, vec![41, 42, 43]),
            other => panic!("unexpected task payload: {other:?}"),
        }
        assert!(processed
            .validate_chat_execution_preparation()
            .expect_err("public tokens must not create the private marker")
            .to_string()
            .contains("missing exact model prompt preparation"));
    }

    #[test]
    fn exact_chat_preparation_overwrites_forged_token_views() {
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.prompt_tokens = vec![999];
        request.task = EngineTask::Chat(ChatEngineInput {
            messages: vec![ChatMessage {
                role: ChatRole::Assistant,
                content: "forged".to_string(),
            }],
            chat_config: ChatRequestConfig::default(),
            prompt_tokens: vec![888],
        });

        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![41, 42, 43], None)
            .expect("exact preparation should install");
        request
            .validate_chat_execution_preparation()
            .expect("installed preparation should validate");
        assert_eq!(request.prompt_tokens, vec![41, 42, 43]);
        match &request.task {
            EngineTask::Chat(input) => {
                assert_eq!(input.prompt_tokens, vec![41, 42, 43]);
                assert_eq!(input.messages[0].content, "Hello");
            }
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn exact_chat_preparation_detects_field_and_typed_payload_mutation() {
        let prepared_request = || {
            let mut request = EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
            }])
            .with_model_variant(ModelVariant::Qwen306B);
            request
                .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![41, 42, 43], None)
                .unwrap();
            request
        };

        let mut field_mutation = prepared_request();
        field_mutation.prompt_tokens[0] = 99;
        assert!(field_mutation
            .validate_chat_execution_preparation()
            .expect_err("field mutation must invalidate preparation")
            .to_string()
            .contains("changed after exact prompt preparation"));

        let mut typed_mutation = prepared_request();
        let EngineTask::Chat(input) = &mut typed_mutation.task else {
            panic!("expected chat task");
        };
        input.prompt_tokens[0] = 99;
        assert!(typed_mutation
            .validate_chat_execution_preparation()
            .expect_err("typed task mutation must invalidate preparation")
            .to_string()
            .contains("typed payload does not match"));
    }

    #[test]
    fn test_request_processor_accepts_audio_bytes_for_asr() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
        let processed = processor.process(request);
        assert!(processed.is_ok());
        let processed = processed.expect("processed");
        assert!(processed.audio_input.is_none());
        assert!(processed.audio_bytes.is_none());
        match processed.task {
            EngineTask::Asr(input) => match input.audio {
                EngineAudioInput::Bytes(bytes) => assert_eq!(bytes, vec![1, 2, 3]),
                other => panic!("unexpected audio input: {other:?}"),
            },
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn test_request_processor_carries_asr_prompt() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let request =
            EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_asr_prompt("spell Izwi correctly");
        let processed = processor.process(request).expect("request should process");

        assert_eq!(
            processed.asr_prompt.as_deref(),
            Some("spell Izwi correctly")
        );
        assert!(processed.num_prompt_tokens() >= 1);
        match processed.task {
            EngineTask::Asr(input) => {
                assert_eq!(input.prompt.as_deref(), Some("spell Izwi correctly"));
            }
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn test_request_processor_defaults_asr_max_tokens() {
        let config = EngineCoreConfig::default();
        let expected_default = 2048usize.min(config.max_seq_len);
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::asr("UklGRg==");
        request.params.max_tokens = 0;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.params.max_tokens, expected_default);
    }

    #[test]
    fn test_request_processor_defaults_speech_to_speech_max_tokens() {
        let config = EngineCoreConfig::default();
        let expected_default = 2048usize.min(config.max_seq_len);
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::speech_to_speech("UklGRg==");
        request.params.max_tokens = 0;

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.params.max_tokens, expected_default);
    }
}

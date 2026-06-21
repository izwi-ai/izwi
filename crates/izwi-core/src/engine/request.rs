//! Request types and processing for the inference engine.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::config::EngineCoreConfig;
use super::output::StreamingOutput;
use super::types::{GenerationParams, Priority, RequestId, TaskType, TokenId};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRequestConfig};

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
    BlockWithDeadline,
    DropOldest,
    Coalesce,
    Sample,
}

impl Default for EngineStreamPolicy {
    fn default() -> Self {
        Self::FailOnFull
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

/// A request to the engine core.
#[derive(Debug, Clone)]
pub struct EngineCoreRequest {
    /// Unique request ID
    pub id: RequestId,
    /// Typed task payload used by new engine internals.
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
    pub audio_input: Option<String>,
    /// Input audio bytes for first-party routes that already parsed uploads.
    pub audio_bytes: Option<Vec<u8>>,
    /// Optional ASR initial prompt/context hint.
    pub asr_prompt: Option<String>,
    /// Whether ASR max_tokens was filled by a model-specific automatic heuristic.
    pub asr_auto_max_tokens: bool,
    /// Reference audio for voice cloning (base64 encoded)
    pub reference_audio: Option<String>,
    /// Reference text for voice cloning
    pub reference_text: Option<String>,
    /// Voice description for voice design
    pub voice_description: Option<String>,
    /// Optional system prompt (e.g. speech-to-speech system instruction).
    pub system_prompt: Option<String>,
    /// Generation parameters
    pub params: GenerationParams,
    /// Request priority
    pub priority: Priority,
    /// Arrival timestamp
    pub arrival_time: Instant,
    /// Prompt token IDs (set by processor)
    pub prompt_tokens: Vec<TokenId>,
    /// Enable streaming output
    pub streaming: bool,
    /// Backpressure behavior for streaming output.
    pub stream_policy: EngineStreamPolicy,
    /// Channel for streaming output (internal use)
    #[allow(dead_code)]
    pub(crate) streaming_tx: Option<mpsc::Sender<StreamingOutput>>,
}

impl EngineCoreRequest {
    fn task_from_fields(
        task_type: TaskType,
        text: Option<String>,
        chat_messages: Option<Vec<ChatMessage>>,
        chat_config: ChatRequestConfig,
        language: Option<String>,
        audio_input: Option<String>,
        audio_bytes: Option<Vec<u8>>,
        asr_prompt: Option<String>,
        reference_audio: Option<String>,
        reference_text: Option<String>,
        voice_description: Option<String>,
        system_prompt: Option<String>,
        prompt_tokens: Vec<TokenId>,
    ) -> EngineTask {
        let audio = || {
            audio_bytes
                .clone()
                .map(EngineAudioInput::Bytes)
                .or_else(|| audio_input.clone().map(EngineAudioInput::Base64))
                .unwrap_or_else(|| EngineAudioInput::Bytes(Vec::new()))
        };

        match task_type {
            TaskType::TTS => EngineTask::Tts(TtsEngineInput {
                text: text.unwrap_or_default(),
                reference_audio,
                reference_text,
                voice_description,
            }),
            TaskType::ASR => EngineTask::Asr(AsrEngineInput {
                audio: audio(),
                language,
                prompt: asr_prompt,
            }),
            TaskType::Chat => EngineTask::Chat(ChatEngineInput {
                messages: chat_messages.unwrap_or_default(),
                chat_config,
                prompt_tokens,
            }),
            TaskType::SpeechToSpeech => EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: audio(),
                messages: chat_messages.unwrap_or_default(),
                system_prompt,
            }),
        }
    }

    fn sync_task_from_fields(&mut self) {
        self.task = Self::task_from_fields(
            self.task_type,
            self.text.clone(),
            self.chat_messages.clone(),
            self.chat_config.clone(),
            self.language.clone(),
            self.audio_input.clone(),
            self.audio_bytes.clone(),
            self.asr_prompt.clone(),
            self.reference_audio.clone(),
            self.reference_text.clone(),
            self.voice_description.clone(),
            self.system_prompt.clone(),
            self.prompt_tokens.clone(),
        );
    }

    /// Create a new TTS request.
    pub fn tts(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Tts(TtsEngineInput {
                text: text.clone(),
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
        }
    }

    /// Create a new ASR request.
    pub fn asr(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                audio: EngineAudioInput::Base64(audio_base64.clone()),
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
        }
    }

    /// Create a new ASR request from already-decoded audio bytes.
    pub fn asr_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::Asr(AsrEngineInput {
                audio: EngineAudioInput::Bytes(audio_bytes.clone()),
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
        }
    }

    /// Create a new speech-to-speech request.
    pub fn speech_to_speech(audio_base64: impl Into<String>) -> Self {
        let audio_base64 = audio_base64.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Base64(audio_base64.clone()),
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
        }
    }

    /// Create a new speech-to-speech request from already-decoded audio bytes.
    pub fn speech_to_speech_bytes(audio_bytes: impl Into<Vec<u8>>) -> Self {
        let audio_bytes = audio_bytes.into();
        Self {
            id: Uuid::new_v4().to_string(),
            task: EngineTask::SpeechToSpeech(AudioChatEngineInput {
                audio: EngineAudioInput::Bytes(audio_bytes.clone()),
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
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            stream_policy: EngineStreamPolicy::default(),
            streaming_tx: None,
        }
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

    /// Enable streaming.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
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
        } else if let Some(prompt) = &self.asr_prompt {
            (prompt.len() / 4).max(1)
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
                if request.audio_input.is_none() && request.audio_bytes.is_none() {
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
                if request.audio_input.is_none() && request.audio_bytes.is_none() {
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

        // Preserve exact prompt tokens when the caller already computed them.
        if request.prompt_tokens.is_empty() {
            // Tokenize text input (simplified - actual tokenization would be more complex)
            if let Some(text) = &request.text {
                let estimated_tokens = (text.len() / 4).max(1);
                request.prompt_tokens = (0..estimated_tokens as u32).collect();
            } else if let Some(prompt) = &request.asr_prompt {
                let estimated_tokens = (prompt.len() / 4).max(1);
                request.prompt_tokens = (0..estimated_tokens as u32).collect();
            } else if let Some(messages) = &request.chat_messages {
                let estimated_tokens =
                    (messages.iter().map(|m| m.content.len()).sum::<usize>() / 4).max(1);
                request.prompt_tokens = (0..estimated_tokens as u32).collect();
            }
        }

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

    /// Enable streaming.
    pub fn streaming(mut self) -> Self {
        self.request.streaming = true;
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
        assert_eq!(request.text.as_deref(), Some("Hello, world!"));
        match &request.task {
            EngineTask::Tts(input) => assert_eq!(input.text, "Hello, world!"),
            other => panic!("unexpected task payload: {other:?}"),
        }
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
    fn test_request_processor_preserves_precomputed_prompt_tokens() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.prompt_tokens = vec![41, 42, 43];

        let processed = processor.process(request).expect("request should process");
        assert_eq!(processed.prompt_tokens, vec![41, 42, 43]);
        match processed.task {
            EngineTask::Chat(input) => assert_eq!(input.prompt_tokens, vec![41, 42, 43]),
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn test_request_processor_accepts_audio_bytes_for_asr() {
        let config = EngineCoreConfig::default();
        let processor = RequestProcessor::new(config);

        let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
        let processed = processor.process(request);
        assert!(processed.is_ok());
        match processed.expect("processed").task {
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

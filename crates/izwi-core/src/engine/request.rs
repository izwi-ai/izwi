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
use crate::models::shared::chat::ChatMessage;

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

/// A request to the engine core.
#[derive(Debug, Clone)]
pub struct EngineCoreRequest {
    /// Unique request ID
    pub id: RequestId,
    /// Task type (TTS, ASR, AudioChat)
    pub task_type: TaskType,
    /// Specific model variant to route to.
    pub model_variant: Option<ModelVariant>,
    /// Input text (for TTS)
    pub text: Option<String>,
    /// Chat input messages.
    pub chat_messages: Option<Vec<ChatMessage>>,
    /// Optional language hint for multilingual generation.
    pub language: Option<String>,
    /// Request correlation ID propagated from API/runtime boundaries.
    pub correlation_id: Option<String>,
    /// Input audio (base64 encoded, for ASR/chat)
    pub audio_input: Option<String>,
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
    /// Channel for streaming output (internal use)
    #[allow(dead_code)]
    pub(crate) streaming_tx: Option<mpsc::UnboundedSender<StreamingOutput>>,
}

impl EngineCoreRequest {
    /// Create a new TTS request.
    pub fn tts(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task_type: TaskType::TTS,
            model_variant: None,
            text: Some(text.into()),
            chat_messages: None,
            language: None,
            correlation_id: None,
            audio_input: None,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            streaming_tx: None,
        }
    }

    /// Create a new ASR request.
    pub fn asr(audio_base64: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task_type: TaskType::ASR,
            model_variant: None,
            text: None,
            chat_messages: None,
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64.into()),
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            streaming_tx: None,
        }
    }

    /// Create a new chat request.
    pub fn chat(messages: Vec<ChatMessage>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task_type: TaskType::Chat,
            model_variant: None,
            text: None,
            chat_messages: Some(messages),
            language: None,
            correlation_id: None,
            audio_input: None,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
            streaming_tx: None,
        }
    }

    /// Create a new speech-to-speech request.
    pub fn speech_to_speech(audio_base64: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task_type: TaskType::SpeechToSpeech,
            model_variant: None,
            text: None,
            chat_messages: None,
            language: None,
            correlation_id: None,
            audio_input: Some(audio_base64.into()),
            reference_audio: None,
            reference_text: None,
            voice_description: None,
            system_prompt: None,
            params: GenerationParams::default(),
            priority: Priority::Normal,
            arrival_time: Instant::now(),
            prompt_tokens: Vec::new(),
            streaming: false,
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

    /// Set voice/speaker.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.params.voice = Some(voice.into());
        self
    }

    /// Set reference audio for voice cloning.
    pub fn with_reference(mut self, audio: impl Into<String>, text: impl Into<String>) -> Self {
        self.reference_audio = Some(audio.into());
        self.reference_text = Some(text.into());
        self
    }

    /// Set voice description.
    pub fn with_voice_description(mut self, description: impl Into<String>) -> Self {
        self.voice_description = Some(description.into());
        self
    }

    /// Set language hint.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
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
        self
    }

    /// Get number of prompt tokens.
    pub fn num_prompt_tokens(&self) -> usize {
        if !self.prompt_tokens.is_empty() {
            self.prompt_tokens.len()
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
                if request.audio_input.is_none() {
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
                if request.audio_input.is_none() {
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
            } else if let Some(messages) = &request.chat_messages {
                let estimated_tokens =
                    (messages.iter().map(|m| m.content.len()).sum::<usize>() / 4).max(1);
                request.prompt_tokens = (0..estimated_tokens as u32).collect();
            }
        }

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
        self
    }

    /// Set text input (for chat).
    pub fn text_input(mut self, text: impl Into<String>) -> Self {
        self.request.text = Some(text.into());
        self
    }

    /// Set chat messages.
    pub fn chat_messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.request.chat_messages = Some(messages);
        self
    }

    /// Set language hint.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.request.language = Some(language.into());
        self
    }

    /// Set speech-to-speech system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.system_prompt = Some(prompt.into());
        self
    }

    /// Build the request.
    pub fn build(self) -> EngineCoreRequest {
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

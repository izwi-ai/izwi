//! Typed runtime request wrappers.
//!
//! These types sit between public runtime requests and the broad engine request
//! shape. They keep task-specific validation and model identity close to the
//! capability that needs them while preserving the existing engine contract.

use crate::engine::{
    AsrTaskOptions, EngineCoreRequest, EngineStreamPolicy,
    GenerationParams as CoreGenerationParams, Priority, WorkloadClass,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRequestConfig};
use crate::runtime::types::{GenerationRequest, RuntimeRequestContext};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RuntimeStreamPolicy {
    #[default]
    FailOnFull,
    BlockWithDeadline {
        timeout_ms: u64,
    },
}

impl From<RuntimeStreamPolicy> for EngineStreamPolicy {
    fn from(policy: RuntimeStreamPolicy) -> Self {
        match policy {
            RuntimeStreamPolicy::FailOnFull => Self::FailOnFull,
            RuntimeStreamPolicy::BlockWithDeadline { timeout_ms } => {
                Self::BlockWithDeadline { timeout_ms }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RequestEnvelope {
    pub(crate) request_id: String,
    pub(crate) model_variant: ModelVariant,
    pub(crate) correlation_id: Option<String>,
    pub(crate) priority: Priority,
    pub(crate) deadline: Option<Instant>,
    pub(crate) stream_policy: RuntimeStreamPolicy,
    pub(crate) workload_class: WorkloadClass,
    pub(crate) admission_ms: Option<f64>,
    pub(crate) tenant_key: Option<[u8; 32]>,
}

impl RequestEnvelope {
    pub(crate) fn new(model_variant: ModelVariant) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            model_variant,
            correlation_id: None,
            priority: Priority::default(),
            deadline: None,
            stream_policy: RuntimeStreamPolicy::default(),
            workload_class: WorkloadClass::Online,
            admission_ms: None,
            tenant_key: None,
        }
    }

    pub(crate) fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = request_id.into();
        self
    }

    pub(crate) fn with_correlation_id(mut self, correlation_id: Option<String>) -> Self {
        self.correlation_id = correlation_id;
        self
    }

    pub(crate) fn with_stream_policy(mut self, stream_policy: RuntimeStreamPolicy) -> Self {
        self.stream_policy = stream_policy;
        self
    }

    pub(crate) fn with_runtime_context(mut self, context: RuntimeRequestContext) -> Self {
        self.workload_class = context.workload_class;
        self.admission_ms = context.admission_ms;
        self.tenant_key = context.tenant_key;
        self.priority = context.priority;
        self.deadline = context.deadline;
        self
    }

    fn apply_to(self, request: &mut EngineCoreRequest) {
        request.id = self.request_id;
        request.model_variant = Some(self.model_variant);
        request.correlation_id = self.correlation_id;
        request.priority = self.priority;
        request.deadline = self.deadline;
        request.stream_policy = self.stream_policy.into();
        request.workload_class = self.workload_class;
        request.admission_ms = self.admission_ms;
        request.tenant_key = self.tenant_key;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeAudioInput {
    Base64(String),
    Bytes(Vec<u8>),
}

impl RuntimeAudioInput {
    fn validate(&self, capability: &str) -> Result<()> {
        match self {
            Self::Base64(value) if value.is_empty() => Err(Error::InvalidInput(format!(
                "{capability} request missing audio input"
            ))),
            Self::Bytes(value) if value.is_empty() => Err(Error::InvalidInput(format!(
                "{capability} request missing audio bytes"
            ))),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TtsRuntimeRequest {
    envelope: RequestEnvelope,
    text: String,
    language: Option<String>,
    reference_audio: Option<String>,
    reference_text: Option<String>,
    reference_language: Option<String>,
    voice_description: Option<String>,
}

/// How long a TTS stream may wait for its reader before failing: the same
/// 30 s stall deadline the HTTP streams apply. A slow reader is waited for
/// rather than failed when the engine's small queue fills (TPIPE-13).
const TTS_STREAM_SEND_DEADLINE_MS: u64 = 30_000;

/// Speech-to-speech audio waits for its reader like TTS (SPK-F5): a voice
/// session's audio queue grants credit as the client plays it back, so a
/// reply generated faster than real time is held back, not failed.
const SPEECH_TO_SPEECH_STREAM_POLICY: RuntimeStreamPolicy =
    RuntimeStreamPolicy::BlockWithDeadline {
        timeout_ms: TTS_STREAM_SEND_DEADLINE_MS,
    };

impl TtsRuntimeRequest {
    pub(crate) fn from_generation(
        request: GenerationRequest,
        model_variant: ModelVariant,
    ) -> Result<Self> {
        if request.text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_request_id(request.id)
                .with_correlation_id(request.correlation_id)
                .with_runtime_context(request.runtime_context)
                .with_stream_policy(RuntimeStreamPolicy::BlockWithDeadline {
                    timeout_ms: TTS_STREAM_SEND_DEADLINE_MS,
                }),
            text: request.text,
            language: request.language,
            reference_audio: request.reference_audio,
            reference_text: request.reference_text,
            reference_language: request.reference_language,
            voice_description: request.voice_description,
        })
    }

    pub(crate) fn into_engine_request(self, params: CoreGenerationParams) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::tts(self.text);
        self.envelope.apply_to(&mut request);
        request.language = self.language;
        request.reference_audio = self.reference_audio;
        request.reference_text = self.reference_text;
        request.reference_language = self.reference_language;
        request.voice_description = self.voice_description;
        request.params = params;
        request
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AsrRuntimeRequest {
    envelope: RequestEnvelope,
    audio: RuntimeAudioInput,
    language: Option<String>,
    prompt: Option<String>,
    task: AsrTaskOptions,
    native_word_timestamps: bool,
}

impl AsrRuntimeRequest {
    pub(crate) fn from_base64(
        audio_base64: impl Into<String>,
        model_variant: ModelVariant,
        language: Option<String>,
        correlation_id: Option<String>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Base64(audio_base64.into());
        audio.validate("ASR")?;
        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_correlation_id(correlation_id)
                .with_runtime_context(runtime_context),
            audio,
            language,
            prompt: None,
            task: AsrTaskOptions::default(),
            native_word_timestamps: false,
        })
    }

    pub(crate) fn from_bytes(
        audio_bytes: impl Into<Vec<u8>>,
        model_variant: ModelVariant,
        language: Option<String>,
        correlation_id: Option<String>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("ASR")?;
        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_correlation_id(correlation_id)
                .with_runtime_context(runtime_context),
            audio,
            language,
            prompt: None,
            task: AsrTaskOptions::default(),
            native_word_timestamps: false,
        })
    }

    pub(crate) fn with_prompt(mut self, prompt: Option<String>) -> Self {
        self.prompt = prompt.and_then(|prompt| {
            let trimmed = prompt.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });
        self
    }

    pub(crate) fn with_task(mut self, task: AsrTaskOptions) -> Self {
        self.task = task;
        self
    }

    pub(crate) fn with_native_word_timestamps(mut self, enabled: bool) -> Self {
        self.native_word_timestamps = enabled;
        self
    }

    pub(crate) fn into_engine_request(self) -> EngineCoreRequest {
        let mut request = match self.audio {
            RuntimeAudioInput::Base64(audio_base64) => EngineCoreRequest::asr(audio_base64),
            RuntimeAudioInput::Bytes(audio_bytes) => EngineCoreRequest::asr_bytes(audio_bytes),
        };
        self.envelope.apply_to(&mut request);
        if let Some(language) = self.language {
            request = request.with_language(language);
        }
        if let Some(prompt) = self.prompt {
            request = request.with_asr_prompt(prompt);
        }
        request
            .with_asr_task(self.task)
            .with_asr_native_word_timestamps(self.native_word_timestamps)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ChatRuntimeRequest {
    envelope: RequestEnvelope,
    messages: Vec<ChatMessage>,
    params: CoreGenerationParams,
    chat_config: ChatRequestConfig,
    prompt_tokens: Vec<u32>,
}

impl ChatRuntimeRequest {
    pub(crate) fn from_messages(
        model_variant: ModelVariant,
        messages: Vec<ChatMessage>,
        params: CoreGenerationParams,
        chat_config: ChatRequestConfig,
        prompt_tokens: Vec<u32>,
        correlation_id: Option<String>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<Self> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request missing messages".to_string(),
            ));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_correlation_id(correlation_id)
                .with_runtime_context(runtime_context),
            messages,
            params,
            chat_config,
            prompt_tokens,
        })
    }

    pub(crate) fn into_engine_request(self) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::chat(self.messages);
        self.envelope.apply_to(&mut request);
        request.params = self.params;
        request.chat_config = self.chat_config;
        request.prompt_tokens = self.prompt_tokens;
        request
    }

    /// The request as an LFM3-Audio chat completion, whose prompt the
    /// model's reply row prepares (see [`EngineCoreRequest::lfm3_audio_chat`]).
    pub(crate) fn into_lfm3_audio_engine_request(self) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::lfm3_audio_chat(self.messages);
        self.envelope.apply_to(&mut request);
        request.params = self.params;
        request.chat_config = self.chat_config;
        request
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AudioChatRuntimeRequest {
    envelope: RequestEnvelope,
    audio: RuntimeAudioInput,
    messages: Vec<ChatMessage>,
    params: CoreGenerationParams,
    system_prompt: Option<String>,
}

impl AudioChatRuntimeRequest {
    pub(crate) fn speech_to_speech_bytes(
        model_variant: ModelVariant,
        audio_bytes: impl Into<Vec<u8>>,
        messages: Vec<ChatMessage>,
        params: CoreGenerationParams,
        system_prompt: Option<String>,
        correlation_id: Option<String>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("speech-to-speech")?;
        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_correlation_id(correlation_id)
                .with_runtime_context(runtime_context)
                .with_stream_policy(SPEECH_TO_SPEECH_STREAM_POLICY),
            audio,
            messages,
            params,
            system_prompt,
        })
    }

    pub(crate) fn speech_to_speech_base64(
        model_variant: ModelVariant,
        audio_base64: impl Into<String>,
        messages: Vec<ChatMessage>,
        params: CoreGenerationParams,
        system_prompt: Option<String>,
        correlation_id: Option<String>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Base64(audio_base64.into());
        audio.validate("speech-to-speech")?;
        Ok(Self {
            envelope: RequestEnvelope::new(model_variant)
                .with_correlation_id(correlation_id)
                .with_runtime_context(runtime_context)
                .with_stream_policy(SPEECH_TO_SPEECH_STREAM_POLICY),
            audio,
            messages,
            params,
            system_prompt,
        })
    }

    pub(crate) fn into_engine_request(self) -> EngineCoreRequest {
        let mut request = match self.audio {
            RuntimeAudioInput::Base64(audio_base64) => {
                EngineCoreRequest::speech_to_speech(audio_base64)
            }
            RuntimeAudioInput::Bytes(audio_bytes) => {
                EngineCoreRequest::speech_to_speech_bytes(audio_bytes)
            }
        };
        self.envelope.apply_to(&mut request);
        request.chat_messages = (!self.messages.is_empty()).then_some(self.messages);
        request.system_prompt = self
            .system_prompt
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        request.params = self.params;
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ModelVariant;
    use crate::engine::{EngineTask, GenerationParams};
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use crate::runtime::types::GenerationRequest;

    /// TPIPE-13: a TTS stream waits up to the stall deadline for a slow
    /// reader instead of failing when the engine queue is full.
    #[test]
    fn tts_streams_wait_for_a_slow_reader_up_to_the_stall_deadline() {
        let request = TtsRuntimeRequest::from_generation(
            GenerationRequest::new("hello"),
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        )
        .unwrap()
        .into_engine_request(CoreGenerationParams::default());
        assert_eq!(
            request.stream_policy,
            EngineStreamPolicy::BlockWithDeadline { timeout_ms: 30_000 }
        );
    }

    #[test]
    fn tts_runtime_request_builds_core_request_with_model_identity() {
        let generation_request = GenerationRequest::new("hello")
            .with_model_variant(ModelVariant::Qwen3Tts12Hz06BBase)
            .with_speaker("alice");
        let params = GenerationParams {
            speaker: Some("alice".to_string()),
            ..GenerationParams::default()
        };

        let runtime_request = TtsRuntimeRequest::from_generation(
            generation_request,
            ModelVariant::Qwen3Tts12Hz06BBase,
        )
        .expect("valid runtime request");
        let core_request = runtime_request.into_engine_request(params);

        assert_eq!(
            core_request.model_variant,
            Some(ModelVariant::Qwen3Tts12Hz06BBase)
        );
        assert_eq!(core_request.text.as_deref(), Some("hello"));
        assert_eq!(core_request.params.speaker.as_deref(), Some("alice"));
    }

    #[test]
    fn tts_runtime_request_rejects_empty_text() {
        let err = TtsRuntimeRequest::from_generation(
            GenerationRequest::new(""),
            ModelVariant::Qwen3Tts12Hz06BBase,
        )
        .expect_err("empty text should be rejected");

        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn asr_runtime_request_builds_core_request_with_audio_bytes() {
        let deadline = Instant::now();
        let runtime_request = AsrRuntimeRequest::from_bytes(
            vec![1, 2, 3],
            ModelVariant::WhisperLargeV3Turbo,
            Some("en".to_string()),
            Some("corr-asr".to_string()),
            RuntimeRequestContext::new(WorkloadClass::Batch)
                .with_admission_ms(3.5)
                .with_tenant_key([7; 32])
                .with_priority(Priority::Critical)
                .with_deadline(deadline),
        )
        .expect("valid ASR request");

        let core_request = runtime_request.into_engine_request();

        assert_eq!(
            core_request.model_variant,
            Some(ModelVariant::WhisperLargeV3Turbo)
        );
        assert_eq!(core_request.audio_bytes.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(core_request.language.as_deref(), Some("en"));
        assert_eq!(core_request.tenant_key, Some([7; 32]));
        assert_eq!(core_request.correlation_id.as_deref(), Some("corr-asr"));
        assert_eq!(core_request.workload_class, WorkloadClass::Batch);
        assert_eq!(core_request.admission_ms, Some(3.5));
        assert_eq!(core_request.priority, Priority::Critical);
        assert_eq!(core_request.deadline, Some(deadline));
    }

    #[test]
    fn asr_runtime_request_carries_initial_prompt() {
        let runtime_request = AsrRuntimeRequest::from_bytes(
            vec![1, 2, 3],
            ModelVariant::WhisperLargeV3Turbo,
            None,
            None,
            RuntimeRequestContext::default(),
        )
        .expect("valid ASR request")
        .with_prompt(Some("  spell Izwi correctly  ".to_string()));

        let core_request = runtime_request.into_engine_request();

        assert_eq!(
            core_request.asr_prompt.as_deref(),
            Some("spell Izwi correctly")
        );
        // Large direct fields remain in their compatibility owner until the
        // Engine/RequestProcessor preflight establishes the typed task view.
        assert!(matches!(core_request.task, EngineTask::Asr(_)));
        assert!(core_request.asr_task.is_empty());
    }

    #[test]
    fn asr_runtime_request_carries_translation_and_keywords() {
        let core_request = AsrRuntimeRequest::from_bytes(
            vec![1, 2, 3],
            ModelVariant::GraniteSpeech502B,
            None,
            None,
            RuntimeRequestContext::default(),
        )
        .expect("valid ASR request")
        .with_task(AsrTaskOptions::new(
            Some(" German ".to_string()),
            vec![" Izwi ".to_string(), " ".to_string(), "Granite".to_string()],
        ))
        .into_engine_request();

        assert_eq!(
            core_request.asr_task.translate_to.as_deref(),
            Some("German")
        );
        assert_eq!(core_request.asr_task.keywords, vec!["Izwi", "Granite"]);
    }

    #[test]
    fn asr_runtime_request_rejects_empty_audio() {
        let err = AsrRuntimeRequest::from_base64(
            "",
            ModelVariant::WhisperLargeV3Turbo,
            None,
            None,
            RuntimeRequestContext::default(),
        )
        .expect_err("empty audio should be rejected");

        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn chat_runtime_request_builds_core_request_with_prompt_tokens() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }];
        let params = GenerationParams {
            max_tokens: 12,
            ..Default::default()
        };
        let chat_config = ChatRequestConfig {
            enable_thinking: Some(false),
            ..Default::default()
        };

        let runtime_request = ChatRuntimeRequest::from_messages(
            ModelVariant::Qwen359BGguf,
            messages,
            params,
            chat_config,
            vec![10, 11],
            Some("corr-chat".to_string()),
            RuntimeRequestContext::default(),
        )
        .expect("valid chat request");
        let core_request = runtime_request.into_engine_request();

        assert_eq!(core_request.model_variant, Some(ModelVariant::Qwen359BGguf));
        assert_eq!(core_request.params.max_tokens, 12);
        assert_eq!(core_request.prompt_tokens, vec![10, 11]);
        assert_eq!(core_request.correlation_id.as_deref(), Some("corr-chat"));
        assert_eq!(core_request.chat_config.enable_thinking, Some(false));
        assert_eq!(
            core_request
                .chat_messages
                .as_ref()
                .and_then(|messages| messages.first())
                .map(|message| message.content.as_str()),
            Some("hello")
        );
    }

    #[test]
    fn chat_runtime_request_rejects_empty_messages() {
        let err = ChatRuntimeRequest::from_messages(
            ModelVariant::Qwen359BGguf,
            Vec::new(),
            GenerationParams::default(),
            ChatRequestConfig::default(),
            Vec::new(),
            None,
            RuntimeRequestContext::default(),
        )
        .expect_err("empty chat messages should be rejected");

        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn audio_chat_runtime_request_builds_speech_to_speech_core_request() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "respond briefly".to_string(),
        }];

        let runtime_request = AudioChatRuntimeRequest::speech_to_speech_bytes(
            ModelVariant::Lfm25Audio15BGguf,
            vec![4, 5, 6],
            messages,
            GenerationParams::default(),
            Some("  helpful voice  ".to_string()),
            Some("corr-audio".to_string()),
            RuntimeRequestContext::default(),
        )
        .expect("valid audio-chat request");

        let core_request = runtime_request.into_engine_request();

        assert_eq!(
            core_request.model_variant,
            Some(ModelVariant::Lfm25Audio15BGguf)
        );
        assert_eq!(core_request.audio_bytes.as_deref(), Some(&[4, 5, 6][..]));
        assert_eq!(core_request.system_prompt.as_deref(), Some("helpful voice"));
        assert_eq!(core_request.correlation_id.as_deref(), Some("corr-audio"));
        assert_eq!(core_request.chat_messages.as_ref().map(Vec::len), Some(1));
        // SPK-F5: a slow reader holds the reply back instead of failing it.
        assert_eq!(
            core_request.stream_policy,
            EngineStreamPolicy::BlockWithDeadline {
                timeout_ms: TTS_STREAM_SEND_DEADLINE_MS
            }
        );
    }
}

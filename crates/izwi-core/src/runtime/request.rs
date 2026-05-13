//! Typed runtime request wrappers.
//!
//! These types sit between public runtime requests and the broad engine request
//! shape. They keep task-specific validation and model identity close to the
//! capability that needs them while preserving the existing engine contract.

use crate::engine::{
    EngineCoreRequest, EngineStreamPolicy, GenerationParams as CoreGenerationParams,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRequestConfig};
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::types::{DiarizationConfig, GenerationRequest};
use std::time::Instant;
use uuid::Uuid;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestPriority {
    Background,
    Normal,
    Interactive,
}

impl Default for RequestPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeStreamPolicy {
    FailOnFull,
    BlockWithDeadline,
    DropOldest,
    Coalesce,
    Sample,
}

impl Default for RuntimeStreamPolicy {
    fn default() -> Self {
        Self::FailOnFull
    }
}

impl From<RuntimeStreamPolicy> for EngineStreamPolicy {
    fn from(policy: RuntimeStreamPolicy) -> Self {
        match policy {
            RuntimeStreamPolicy::FailOnFull => Self::FailOnFull,
            RuntimeStreamPolicy::BlockWithDeadline => Self::BlockWithDeadline,
            RuntimeStreamPolicy::DropOldest => Self::DropOldest,
            RuntimeStreamPolicy::Coalesce => Self::Coalesce,
            RuntimeStreamPolicy::Sample => Self::Sample,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct RequestEnvelope {
    pub(crate) request_id: String,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) correlation_id: Option<String>,
    pub(crate) priority: RequestPriority,
    pub(crate) deadline: Option<Instant>,
    pub(crate) stream_policy: RuntimeStreamPolicy,
}

impl RequestEnvelope {
    pub(crate) fn new(capability: CapabilityKind, model_variant: ModelVariant) -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            capability,
            model_variant,
            correlation_id: None,
            priority: RequestPriority::default(),
            deadline: None,
            stream_policy: RuntimeStreamPolicy::default(),
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

    pub(crate) fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    pub(crate) fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    pub(crate) fn with_stream_policy(mut self, stream_policy: RuntimeStreamPolicy) -> Self {
        self.stream_policy = stream_policy;
        self
    }
}

#[allow(dead_code)]
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
    voice_description: Option<String>,
}

impl TtsRuntimeRequest {
    pub(crate) fn from_generation(
        request: GenerationRequest,
        model_variant: ModelVariant,
    ) -> Result<Self> {
        if request.text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::Tts, model_variant)
                .with_request_id(request.id)
                .with_correlation_id(request.correlation_id),
            text: request.text,
            language: request.language,
            reference_audio: request.reference_audio,
            reference_text: request.reference_text,
            voice_description: request.voice_description,
        })
    }

    pub(crate) fn into_engine_request(self, params: CoreGenerationParams) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::tts(self.text);
        request.id = self.envelope.request_id;
        request.model_variant = Some(self.envelope.model_variant);
        request.correlation_id = self.envelope.correlation_id;
        request.stream_policy = self.envelope.stream_policy.into();
        request.language = self.language;
        request.reference_audio = self.reference_audio;
        request.reference_text = self.reference_text;
        request.voice_description = self.voice_description;
        request.params = params;
        request
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct AsrRuntimeRequest {
    envelope: RequestEnvelope,
    audio: RuntimeAudioInput,
    language: Option<String>,
    prompt: Option<String>,
}

impl AsrRuntimeRequest {
    pub(crate) fn from_base64(
        audio_base64: impl Into<String>,
        model_variant: ModelVariant,
        language: Option<String>,
        correlation_id: Option<String>,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Base64(audio_base64.into());
        audio.validate("ASR")?;
        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::Asr, model_variant)
                .with_correlation_id(correlation_id),
            audio,
            language,
            prompt: None,
        })
    }

    pub(crate) fn from_bytes(
        audio_bytes: impl Into<Vec<u8>>,
        model_variant: ModelVariant,
        language: Option<String>,
        correlation_id: Option<String>,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("ASR")?;
        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::Asr, model_variant)
                .with_correlation_id(correlation_id),
            audio,
            language,
            prompt: None,
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

    pub(crate) fn into_engine_request(self) -> EngineCoreRequest {
        let mut request = match self.audio {
            RuntimeAudioInput::Base64(audio_base64) => EngineCoreRequest::asr(audio_base64),
            RuntimeAudioInput::Bytes(audio_bytes) => EngineCoreRequest::asr_bytes(audio_bytes),
        };
        request.id = self.envelope.request_id;
        request.model_variant = Some(self.envelope.model_variant);
        request.correlation_id = self.envelope.correlation_id;
        request.stream_policy = self.envelope.stream_policy.into();
        if let Some(language) = self.language {
            request = request.with_language(language);
        }
        if let Some(prompt) = self.prompt {
            request = request.with_asr_prompt(prompt);
        }
        request
    }
}

#[allow(dead_code)]
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
    ) -> Result<Self> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request missing messages".to_string(),
            ));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::Chat, model_variant)
                .with_correlation_id(correlation_id),
            messages,
            params,
            chat_config,
            prompt_tokens,
        })
    }

    pub(crate) fn into_engine_request(self) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::chat(self.messages);
        request.id = self.envelope.request_id;
        request.model_variant = Some(self.envelope.model_variant);
        request.correlation_id = self.envelope.correlation_id;
        request.stream_policy = self.envelope.stream_policy.into();
        request.params = self.params;
        request.chat_config = self.chat_config;
        request.prompt_tokens = self.prompt_tokens;
        request
    }
}

#[allow(dead_code)]
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
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("speech-to-speech")?;
        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::SpeechToSpeech, model_variant)
                .with_correlation_id(correlation_id),
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
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Base64(audio_base64.into());
        audio.validate("speech-to-speech")?;
        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::SpeechToSpeech, model_variant)
                .with_correlation_id(correlation_id),
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
        request.id = self.envelope.request_id;
        request.model_variant = Some(self.envelope.model_variant);
        request.correlation_id = self.envelope.correlation_id;
        request.stream_policy = self.envelope.stream_policy.into();
        request.chat_messages = (!self.messages.is_empty()).then_some(self.messages);
        request.system_prompt = self
            .system_prompt
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        request.params = self.params;
        request
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct DiarizationRuntimeRequest {
    pub(crate) envelope: RequestEnvelope,
    pub(crate) audio: RuntimeAudioInput,
    pub(crate) diarization_model_id: Option<String>,
    pub(crate) asr_model_id: Option<String>,
    pub(crate) aligner_model_id: Option<String>,
    pub(crate) llm_model_id: Option<String>,
    pub(crate) config: DiarizationConfig,
    pub(crate) enable_llm_refinement: bool,
}

impl DiarizationRuntimeRequest {
    pub(crate) fn from_bytes(
        model_variant: ModelVariant,
        audio_bytes: impl Into<Vec<u8>>,
        config: DiarizationConfig,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("diarization")?;
        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::Diarization, model_variant),
            audio,
            diarization_model_id: None,
            asr_model_id: None,
            aligner_model_id: None,
            llm_model_id: None,
            config,
            enable_llm_refinement: false,
        })
    }

    pub(crate) fn with_pipeline_models(
        mut self,
        diarization_model_id: Option<String>,
        asr_model_id: Option<String>,
        aligner_model_id: Option<String>,
        llm_model_id: Option<String>,
    ) -> Self {
        self.diarization_model_id = diarization_model_id;
        self.asr_model_id = asr_model_id;
        self.aligner_model_id = aligner_model_id;
        self.llm_model_id = llm_model_id;
        self
    }

    pub(crate) fn with_llm_refinement(mut self, enable_llm_refinement: bool) -> Self {
        self.enable_llm_refinement = enable_llm_refinement;
        self
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct AlignmentRuntimeRequest {
    pub(crate) envelope: RequestEnvelope,
    pub(crate) audio: RuntimeAudioInput,
    pub(crate) transcript: String,
    pub(crate) language: Option<String>,
}

impl AlignmentRuntimeRequest {
    pub(crate) fn from_bytes(
        model_variant: ModelVariant,
        audio_bytes: impl Into<Vec<u8>>,
        transcript: impl Into<String>,
        language: Option<String>,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("forced alignment")?;
        let transcript = transcript.into();
        if transcript.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Forced alignment request missing transcript".to_string(),
            ));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(CapabilityKind::ForcedAlignment, model_variant),
            audio,
            transcript,
            language,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct VadRuntimeRequest {
    pub(crate) envelope: RequestEnvelope,
    pub(crate) audio: RuntimeAudioInput,
    pub(crate) sample_rate: u32,
    pub(crate) endpointing: bool,
}

impl VadRuntimeRequest {
    pub(crate) fn from_pcm_bytes(
        model_variant: ModelVariant,
        audio_bytes: impl Into<Vec<u8>>,
        sample_rate: u32,
        endpointing: bool,
    ) -> Result<Self> {
        let audio = RuntimeAudioInput::Bytes(audio_bytes.into());
        audio.validate("VAD")?;
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "VAD request missing sample rate".to_string(),
            ));
        }

        Ok(Self {
            envelope: RequestEnvelope::new(
                if endpointing {
                    CapabilityKind::Endpointing
                } else {
                    CapabilityKind::Vad
                },
                model_variant,
            ),
            audio,
            sample_rate,
            endpointing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ModelVariant;
    use crate::engine::{EngineTask, GenerationParams};
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use crate::runtime::types::GenerationRequest;

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
    fn request_envelope_carries_capability_model_and_rollout_controls() {
        let deadline = Instant::now();
        let envelope = RequestEnvelope::new(CapabilityKind::Chat, ModelVariant::Qwen38BGguf)
            .with_request_id("req-1")
            .with_correlation_id(Some("corr-1".to_string()))
            .with_priority(RequestPriority::Interactive)
            .with_deadline(Some(deadline))
            .with_stream_policy(RuntimeStreamPolicy::Coalesce);

        assert_eq!(envelope.request_id, "req-1");
        assert_eq!(envelope.capability, CapabilityKind::Chat);
        assert_eq!(envelope.model_variant, ModelVariant::Qwen38BGguf);
        assert_eq!(envelope.correlation_id.as_deref(), Some("corr-1"));
        assert_eq!(envelope.priority, RequestPriority::Interactive);
        assert_eq!(envelope.deadline, Some(deadline));
        assert_eq!(envelope.stream_policy, RuntimeStreamPolicy::Coalesce);
    }

    #[test]
    fn asr_runtime_request_builds_core_request_with_audio_bytes() {
        let runtime_request = AsrRuntimeRequest::from_bytes(
            vec![1, 2, 3],
            ModelVariant::WhisperLargeV3Turbo,
            Some("en".to_string()),
            Some("corr-asr".to_string()),
        )
        .expect("valid ASR request");

        let core_request = runtime_request.into_engine_request();

        assert_eq!(
            core_request.model_variant,
            Some(ModelVariant::WhisperLargeV3Turbo)
        );
        assert_eq!(core_request.audio_bytes.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(core_request.language.as_deref(), Some("en"));
        assert_eq!(core_request.correlation_id.as_deref(), Some("corr-asr"));
    }

    #[test]
    fn asr_runtime_request_carries_initial_prompt() {
        let runtime_request = AsrRuntimeRequest::from_bytes(
            vec![1, 2, 3],
            ModelVariant::WhisperLargeV3Turbo,
            None,
            None,
        )
        .expect("valid ASR request")
        .with_prompt(Some("  spell Izwi correctly  ".to_string()));

        let core_request = runtime_request.into_engine_request();

        assert_eq!(
            core_request.asr_prompt.as_deref(),
            Some("spell Izwi correctly")
        );
        match core_request.task {
            EngineTask::Asr(input) => {
                assert_eq!(input.prompt.as_deref(), Some("spell Izwi correctly"));
            }
            other => panic!("unexpected task payload: {other:?}"),
        }
    }

    #[test]
    fn asr_runtime_request_rejects_empty_audio() {
        let err = AsrRuntimeRequest::from_base64("", ModelVariant::WhisperLargeV3Turbo, None, None)
            .expect_err("empty audio should be rejected");

        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn chat_runtime_request_builds_core_request_with_prompt_tokens() {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        }];
        let mut params = GenerationParams::default();
        params.max_tokens = 12;
        let mut chat_config = ChatRequestConfig::default();
        chat_config.enable_thinking = Some(false);

        let runtime_request = ChatRuntimeRequest::from_messages(
            ModelVariant::Qwen38BGguf,
            messages,
            params,
            chat_config,
            vec![10, 11],
            Some("corr-chat".to_string()),
        )
        .expect("valid chat request");
        let core_request = runtime_request.into_engine_request();

        assert_eq!(core_request.model_variant, Some(ModelVariant::Qwen38BGguf));
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
            ModelVariant::Qwen38BGguf,
            Vec::new(),
            GenerationParams::default(),
            ChatRequestConfig::default(),
            Vec::new(),
            None,
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
    }

    #[test]
    fn diarization_runtime_request_captures_pipeline_model_ids() {
        let request = DiarizationRuntimeRequest::from_bytes(
            ModelVariant::DiarStreamingSortformer4SpkV21,
            vec![1, 2],
            DiarizationConfig::default(),
        )
        .expect("valid diarization request")
        .with_pipeline_models(
            Some("diar".to_string()),
            Some("asr".to_string()),
            Some("aligner".to_string()),
            Some("llm".to_string()),
        )
        .with_llm_refinement(true);

        assert_eq!(request.envelope.capability, CapabilityKind::Diarization);
        assert_eq!(
            request.envelope.model_variant,
            ModelVariant::DiarStreamingSortformer4SpkV21
        );
        assert_eq!(request.asr_model_id.as_deref(), Some("asr"));
        assert!(request.enable_llm_refinement);
    }

    #[test]
    fn alignment_runtime_request_requires_transcript() {
        let err = AlignmentRuntimeRequest::from_bytes(
            ModelVariant::Qwen3ForcedAligner06B,
            vec![1],
            "  ",
            Some("en".to_string()),
        )
        .expect_err("empty transcript should be rejected");

        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn vad_runtime_request_marks_endpointing_capability() {
        let request = VadRuntimeRequest::from_pcm_bytes(
            ModelVariant::VoxtralMini4BRealtime2602,
            vec![1],
            16_000,
            true,
        )
        .expect("valid endpointing request");

        assert_eq!(request.envelope.capability, CapabilityKind::Endpointing);
        assert_eq!(request.sample_rate, 16_000);
        assert!(request.endpointing);
    }
}

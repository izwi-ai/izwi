//! Typed runtime request wrappers.
//!
//! These types sit between public runtime requests and the broad engine request
//! shape. They keep task-specific validation and model identity close to the
//! capability that needs them while preserving the existing engine contract.

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenerationParams};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::types::GenerationRequest;

#[derive(Debug, Clone)]
pub(crate) struct TtsRuntimeRequest {
    id: String,
    model_variant: ModelVariant,
    correlation_id: Option<String>,
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
            id: request.id,
            model_variant,
            correlation_id: request.correlation_id,
            text: request.text,
            language: request.language,
            reference_audio: request.reference_audio,
            reference_text: request.reference_text,
            voice_description: request.voice_description,
        })
    }

    pub(crate) fn into_engine_request(self, params: CoreGenerationParams) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::tts(self.text);
        request.id = self.id;
        request.model_variant = Some(self.model_variant);
        request.correlation_id = self.correlation_id;
        request.language = self.language;
        request.reference_audio = self.reference_audio;
        request.reference_text = self.reference_text;
        request.voice_description = self.voice_description;
        request.params = params;
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ModelVariant;
    use crate::engine::GenerationParams;
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
}

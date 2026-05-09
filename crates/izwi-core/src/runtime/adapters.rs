//! Runtime capability adapter registry.
//!
//! The registry is intentionally metadata-first for now. It gives runtime
//! orchestration one stable place to ask whether a model can satisfy a
//! capability before dispatch reaches concrete model-family code.

use std::collections::HashMap;

use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CapabilityKind {
    Asr,
    RealtimeAsr,
    Tts,
    StreamingTts,
    Chat,
    AudioChat,
    SpeechToSpeech,
    Diarization,
    ForcedAlignment,
    Vad,
    Endpointing,
    Tokenizer,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamingMode {
    None,
    Chunked,
    Realtime,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterMetadata {
    pub(crate) id: &'static str,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_mode: StreamingMode,
}

pub(crate) trait ModelCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata>;
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeAdapterRegistry {
    adapters: HashMap<(CapabilityKind, ModelVariant), AdapterMetadata>,
}

impl RuntimeAdapterRegistry {
    pub(crate) fn built_in() -> Self {
        let mut registry = Self::default();
        registry.register_adapter(TtsCapabilityAdapter);
        registry
    }

    pub(crate) fn require(
        &self,
        capability: CapabilityKind,
        model_variant: ModelVariant,
    ) -> Result<&AdapterMetadata> {
        self.adapters
            .get(&(capability, model_variant))
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Model {model_variant} does not support runtime capability {capability:?}"
                ))
            })
    }

    fn register_adapter<A>(&mut self, adapter: A)
    where
        A: ModelCapabilityAdapter,
    {
        for variant in ModelVariant::all().iter().copied() {
            if let Some(metadata) = adapter.metadata_for(variant) {
                self.adapters
                    .insert((metadata.capability, metadata.model_variant), metadata);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TtsCapabilityAdapter;

impl ModelCapabilityAdapter for TtsCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        let capabilities = model_variant.speech_capabilities()?;
        Some(AdapterMetadata {
            id: "builtin.tts",
            capability: CapabilityKind::Tts,
            model_variant,
            streaming_mode: if capabilities.supports_streaming {
                StreamingMode::Chunked
            } else {
                StreamingMode::None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_registry_resolves_tts_models() {
        let registry = RuntimeAdapterRegistry::built_in();

        let qwen = registry
            .require(CapabilityKind::Tts, ModelVariant::Qwen3Tts12Hz06BBase)
            .expect("qwen tts adapter");
        assert_eq!(qwen.id, "builtin.tts");
        assert_eq!(qwen.streaming_mode, StreamingMode::Chunked);

        let lfm = registry
            .require(CapabilityKind::Tts, ModelVariant::Lfm25Audio15BGguf)
            .expect("lfm audio tts adapter");
        assert_eq!(lfm.streaming_mode, StreamingMode::None);
    }

    #[test]
    fn built_in_registry_rejects_non_tts_models() {
        let registry = RuntimeAdapterRegistry::built_in();

        let err = registry
            .require(CapabilityKind::Tts, ModelVariant::Qwen38BGguf)
            .expect_err("chat model should not satisfy TTS");

        assert!(matches!(err, Error::InvalidInput(_)));
    }
}

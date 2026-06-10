//! Runtime capability adapter registry.
//!
//! The registry is intentionally metadata-first for now. It gives runtime
//! orchestration one stable place to ask whether a model can satisfy a
//! capability before dispatch reaches concrete model-family code.

use std::collections::HashMap;

use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
pub(crate) enum ExecutionTargetKind {
    TokenEngine,
    BatchRunner,
    RealtimeRunner,
    PipelineRunner,
    DirectModel,
    Artifact,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterMetadata {
    pub(crate) id: &'static str,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_mode: StreamingMode,
    pub(crate) execution_target: ExecutionTargetKind,
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
        registry.register_adapter(StreamingTtsCapabilityAdapter);
        registry.register_adapter(AsrCapabilityAdapter);
        registry.register_adapter(RealtimeAsrCapabilityAdapter);
        registry.register_adapter(ChatCapabilityAdapter);
        registry.register_adapter(AudioChatCapabilityAdapter);
        registry.register_adapter(SpeechToSpeechCapabilityAdapter);
        registry.register_adapter(DiarizationCapabilityAdapter);
        registry.register_adapter(ForcedAlignmentCapabilityAdapter);
        registry.register_adapter(TokenizerCapabilityAdapter);
        registry
    }

    pub(crate) fn capabilities_for(&self, model_variant: ModelVariant) -> Vec<AdapterMetadata> {
        let mut capabilities = self
            .adapters
            .iter()
            .filter_map(|((_, variant), metadata)| (*variant == model_variant).then_some(*metadata))
            .collect::<Vec<_>>();
        capabilities.sort_by_key(|metadata| metadata.capability);
        capabilities
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

fn tts_execution_target(model_variant: ModelVariant) -> ExecutionTargetKind {
    if model_variant.is_kokoro()
        || model_variant.is_lfm25_audio_gguf()
        || matches!(
            model_variant.family(),
            crate::catalog::ModelFamily::VoxtralTts | crate::catalog::ModelFamily::VibeVoiceTts
        )
    {
        ExecutionTargetKind::DirectModel
    } else {
        ExecutionTargetKind::TokenEngine
    }
}

fn tts_streaming_mode(model_variant: ModelVariant) -> StreamingMode {
    let Some(capabilities) = model_variant.speech_capabilities() else {
        return StreamingMode::None;
    };
    if capabilities.supports_streaming
        || model_variant.is_lfm25_audio_gguf()
        || matches!(
            model_variant.family(),
            crate::catalog::ModelFamily::VoxtralTts | crate::catalog::ModelFamily::VibeVoiceTts
        )
    {
        StreamingMode::Chunked
    } else {
        StreamingMode::None
    }
}

fn asr_execution_target(model_variant: ModelVariant) -> ExecutionTargetKind {
    if model_variant.is_audio_chat() || model_variant.is_voxtral() {
        ExecutionTargetKind::DirectModel
    } else {
        ExecutionTargetKind::BatchRunner
    }
}

#[derive(Debug, Clone, Copy)]
struct TtsCapabilityAdapter;

impl ModelCapabilityAdapter for TtsCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.speech_capabilities()?;
        Some(AdapterMetadata {
            id: "builtin.tts",
            capability: CapabilityKind::Tts,
            model_variant,
            streaming_mode: tts_streaming_mode(model_variant),
            execution_target: tts_execution_target(model_variant),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct StreamingTtsCapabilityAdapter;

impl ModelCapabilityAdapter for StreamingTtsCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        let capabilities = model_variant.speech_capabilities()?;
        capabilities.supports_streaming.then_some(AdapterMetadata {
            id: "builtin.streaming_tts",
            capability: CapabilityKind::StreamingTts,
            model_variant,
            streaming_mode: StreamingMode::Chunked,
            execution_target: tts_execution_target(model_variant),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct AsrCapabilityAdapter;

impl ModelCapabilityAdapter for AsrCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        (model_variant.is_asr() || model_variant.is_voxtral() || model_variant.is_audio_chat())
            .then_some(AdapterMetadata {
                id: "builtin.asr",
                capability: CapabilityKind::Asr,
                model_variant,
                streaming_mode: if model_variant.is_audio_chat() {
                    StreamingMode::Chunked
                } else {
                    StreamingMode::None
                },
                execution_target: asr_execution_target(model_variant),
            })
    }
}

#[derive(Debug, Clone, Copy)]
struct RealtimeAsrCapabilityAdapter;

impl ModelCapabilityAdapter for RealtimeAsrCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        // Native Voxtral realtime stays hidden until the Candle realtime runner exists.
        let _ = model_variant;
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct ChatCapabilityAdapter;

impl ModelCapabilityAdapter for ChatCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.is_chat().then_some(AdapterMetadata {
            id: "builtin.chat",
            capability: CapabilityKind::Chat,
            model_variant,
            streaming_mode: StreamingMode::Chunked,
            execution_target: ExecutionTargetKind::TokenEngine,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct AudioChatCapabilityAdapter;

impl ModelCapabilityAdapter for AudioChatCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.is_audio_chat().then_some(AdapterMetadata {
            id: "builtin.audio_chat",
            capability: CapabilityKind::AudioChat,
            model_variant,
            streaming_mode: StreamingMode::Chunked,
            execution_target: ExecutionTargetKind::TokenEngine,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SpeechToSpeechCapabilityAdapter;

impl ModelCapabilityAdapter for SpeechToSpeechCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.is_audio_chat().then_some(AdapterMetadata {
            id: "builtin.speech_to_speech",
            capability: CapabilityKind::SpeechToSpeech,
            model_variant,
            streaming_mode: StreamingMode::Chunked,
            execution_target: ExecutionTargetKind::TokenEngine,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DiarizationCapabilityAdapter;

impl ModelCapabilityAdapter for DiarizationCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.is_diarization().then_some(AdapterMetadata {
            id: "builtin.diarization",
            capability: CapabilityKind::Diarization,
            model_variant,
            streaming_mode: StreamingMode::None,
            execution_target: ExecutionTargetKind::PipelineRunner,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ForcedAlignmentCapabilityAdapter;

impl ModelCapabilityAdapter for ForcedAlignmentCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant
            .is_forced_aligner()
            .then_some(AdapterMetadata {
                id: "builtin.forced_alignment",
                capability: CapabilityKind::ForcedAlignment,
                model_variant,
                streaming_mode: StreamingMode::None,
                execution_target: ExecutionTargetKind::BatchRunner,
            })
    }
}

#[derive(Debug, Clone, Copy)]
struct TokenizerCapabilityAdapter;

impl ModelCapabilityAdapter for TokenizerCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant.is_tokenizer().then_some(AdapterMetadata {
            id: "builtin.tokenizer",
            capability: CapabilityKind::Tokenizer,
            model_variant,
            streaming_mode: StreamingMode::None,
            execution_target: ExecutionTargetKind::Artifact,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn expected_capabilities(model_variant: ModelVariant) -> BTreeSet<CapabilityKind> {
        let mut expected = BTreeSet::new();

        if model_variant.speech_capabilities().is_some() {
            expected.insert(CapabilityKind::Tts);
        }
        if model_variant
            .speech_capabilities()
            .is_some_and(|capabilities| capabilities.supports_streaming)
        {
            expected.insert(CapabilityKind::StreamingTts);
        }
        if model_variant.is_asr() || model_variant.is_voxtral() || model_variant.is_audio_chat() {
            expected.insert(CapabilityKind::Asr);
        }
        if model_variant.is_chat() {
            expected.insert(CapabilityKind::Chat);
        }
        if model_variant.is_audio_chat() {
            expected.insert(CapabilityKind::AudioChat);
            expected.insert(CapabilityKind::SpeechToSpeech);
        }
        if model_variant.is_diarization() {
            expected.insert(CapabilityKind::Diarization);
        }
        if model_variant.is_forced_aligner() {
            expected.insert(CapabilityKind::ForcedAlignment);
        }
        if model_variant.is_tokenizer() {
            expected.insert(CapabilityKind::Tokenizer);
        }

        expected
    }

    fn registry_capabilities(
        registry: &RuntimeAdapterRegistry,
        model_variant: ModelVariant,
    ) -> BTreeSet<CapabilityKind> {
        registry
            .capabilities_for(model_variant)
            .into_iter()
            .map(|metadata| metadata.capability)
            .collect()
    }

    #[test]
    fn built_in_registry_resolves_tts_models() {
        let registry = RuntimeAdapterRegistry::built_in();

        let qwen = registry
            .require(CapabilityKind::Tts, ModelVariant::Qwen3Tts12Hz06BBase)
            .expect("qwen tts adapter");
        assert_eq!(qwen.id, "builtin.tts");
        assert_eq!(qwen.streaming_mode, StreamingMode::Chunked);
        assert_eq!(qwen.execution_target, ExecutionTargetKind::TokenEngine);

        let lfm = registry
            .require(CapabilityKind::Tts, ModelVariant::Lfm25Audio15BGguf)
            .expect("lfm audio tts adapter");
        assert_eq!(lfm.streaming_mode, StreamingMode::Chunked);
        assert_eq!(lfm.execution_target, ExecutionTargetKind::DirectModel);
    }

    #[test]
    fn built_in_registry_covers_every_model_variant_capability() {
        let registry = RuntimeAdapterRegistry::built_in();

        for variant in ModelVariant::all().iter().copied() {
            assert_eq!(
                registry_capabilities(&registry, variant),
                expected_capabilities(variant),
                "capability registry mismatch for {variant:?}"
            );
        }
    }

    #[test]
    fn built_in_registry_resolves_non_tts_capabilities() {
        let registry = RuntimeAdapterRegistry::built_in();

        assert_eq!(
            registry
                .require(CapabilityKind::Asr, ModelVariant::WhisperLargeV3Turbo)
                .expect("whisper asr adapter")
                .execution_target,
            ExecutionTargetKind::BatchRunner
        );
        assert_eq!(
            registry
                .require(CapabilityKind::Chat, ModelVariant::Qwen38BGguf)
                .expect("qwen chat adapter")
                .execution_target,
            ExecutionTargetKind::TokenEngine
        );
        assert_eq!(
            registry
                .require(
                    CapabilityKind::SpeechToSpeech,
                    ModelVariant::Lfm25Audio15BGguf
                )
                .expect("lfm audio speech-to-speech adapter")
                .execution_target,
            ExecutionTargetKind::TokenEngine
        );
        assert_eq!(
            registry
                .require(
                    CapabilityKind::Diarization,
                    ModelVariant::DiarStreamingSortformer4SpkV21
                )
                .expect("sortformer diarization adapter")
                .execution_target,
            ExecutionTargetKind::PipelineRunner
        );
        assert_eq!(
            registry
                .require(
                    CapabilityKind::ForcedAlignment,
                    ModelVariant::Qwen3ForcedAligner06B
                )
                .expect("forced alignment adapter")
                .execution_target,
            ExecutionTargetKind::BatchRunner
        );
    }

    #[test]
    fn built_in_registry_marks_audio_chat_as_direct_asr_but_token_s2s() {
        let registry = RuntimeAdapterRegistry::built_in();

        assert_eq!(
            registry
                .require(CapabilityKind::Asr, ModelVariant::Lfm25Audio15BGguf)
                .expect("lfm audio asr adapter")
                .execution_target,
            ExecutionTargetKind::DirectModel
        );
        assert_eq!(
            registry
                .require(CapabilityKind::Asr, ModelVariant::Lfm25Audio15BGguf)
                .expect("lfm audio asr adapter")
                .streaming_mode,
            StreamingMode::Chunked
        );
        assert_eq!(
            registry
                .require(CapabilityKind::AudioChat, ModelVariant::Lfm25Audio15BGguf)
                .expect("lfm audio-chat adapter")
                .execution_target,
            ExecutionTargetKind::TokenEngine
        );
    }

    #[test]
    fn built_in_registry_exposes_voxtral_only_as_direct_asr_for_now() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::VoxtralMini4BRealtime2602;

        assert_eq!(
            registry
                .require(CapabilityKind::Asr, variant)
                .expect("voxtral asr adapter")
                .execution_target,
            ExecutionTargetKind::DirectModel
        );
        assert_eq!(
            registry
                .require(CapabilityKind::Asr, variant)
                .expect("voxtral asr adapter")
                .streaming_mode,
            StreamingMode::None
        );
        assert!(
            registry
                .require(CapabilityKind::RealtimeAsr, variant)
                .is_err()
        );
        assert!(
            registry
                .require(CapabilityKind::AudioChat, variant)
                .is_err()
        );
        assert!(
            registry
                .require(CapabilityKind::SpeechToSpeech, variant)
                .is_err()
        );
    }

    #[test]
    fn built_in_registry_marks_granite_speech_as_batch_asr_only() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::GraniteSpeech412BPlus;

        let adapter = registry
            .require(CapabilityKind::Asr, variant)
            .expect("granite speech asr adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::BatchRunner);
        assert_eq!(adapter.streaming_mode, StreamingMode::None);
        assert!(
            registry
                .require(CapabilityKind::RealtimeAsr, variant)
                .is_err()
        );
        assert!(registry.require(CapabilityKind::AudioChat, variant).is_err());
    }

    #[test]
    fn built_in_registry_marks_voxtral_tts_as_direct_tts_with_final_only_streaming() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Voxtral4BTts2603;

        let adapter = registry
            .require(CapabilityKind::Tts, variant)
            .expect("voxtral tts adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::DirectModel);
        assert_eq!(adapter.streaming_mode, StreamingMode::Chunked);
        assert!(
            registry
                .require(CapabilityKind::StreamingTts, variant)
                .is_err()
        );
    }

    #[test]
    fn built_in_registry_marks_vibevoice_tts_as_direct_tts_with_final_only_streaming() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::VibeVoice15BTts;

        let adapter = registry
            .require(CapabilityKind::Tts, variant)
            .expect("vibevoice tts adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::DirectModel);
        assert_eq!(adapter.streaming_mode, StreamingMode::Chunked);
        assert!(
            registry
                .require(CapabilityKind::StreamingTts, variant)
                .is_err()
        );
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

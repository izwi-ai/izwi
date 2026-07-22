//! Runtime capability adapters and loaded-model execution bindings.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::engine::{
    CacheMode, CancellationGranularity, ExecutionMode, ExecutionProfile, NativeBatchMode,
    PrefillMode, TaskType,
};
use crate::error::{Error, Result};

mod loaded;

pub(crate) use loaded::{
    CapabilityStateBinding, LoadedCapabilityBinding, LoadedExecutionContract, LoadedModelBundle,
    LoadedModelBundleDraft, LoadedStatePublication, StreamingRequirements,
};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapabilityKind {
    Asr,
    SpeakerAttributedAsr,
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

impl CapabilityKind {
    pub(crate) const fn for_engine_task(task_type: TaskType) -> Self {
        match task_type {
            TaskType::TTS => Self::Tts,
            TaskType::ASR => Self::Asr,
            TaskType::Chat => Self::Chat,
            TaskType::SpeechToSpeech => Self::SpeechToSpeech,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::SpeakerAttributedAsr => "speaker_attributed_asr",
            Self::RealtimeAsr => "realtime_asr",
            Self::Tts => "tts",
            Self::StreamingTts => "streaming_tts",
            Self::Chat => "chat",
            Self::AudioChat => "audio_chat",
            Self::SpeechToSpeech => "speech_to_speech",
            Self::Diarization => "diarization",
            Self::ForcedAlignment => "forced_alignment",
            Self::Vad => "vad",
            Self::Endpointing => "endpointing",
            Self::Tokenizer => "tokenizer",
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StreamingMode {
    None,
    FinalOnly,
    Chunked,
    Realtime,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionTargetKind {
    TokenEngine,
    BatchRunner,
    RealtimeRunner,
    PipelineRunner,
    DirectModel,
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SequenceExecutionMode {
    None,
    Always,
    StreamingOnly,
}

/// Capability-authored lifetime truth for mutable inference data. This is
/// independent of whether the current compatibility execution profile happens
/// to expose a scheduler cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceStateRequirement {
    Stateless,
    Invocation,
    Retained,
    RetainedAndInvocation,
}

impl InferenceStateRequirement {
    pub(crate) const fn requires_retained(self) -> bool {
        matches!(self, Self::Retained | Self::RetainedAndInvocation)
    }

    pub(crate) const fn requires_invocation(self) -> bool {
        matches!(self, Self::Invocation | Self::RetainedAndInvocation)
    }
}

impl SequenceExecutionMode {
    const fn enabled(self, streaming_required: bool) -> bool {
        match self {
            Self::None => false,
            Self::Always => true,
            Self::StreamingOnly => streaming_required,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdapterMetadata {
    pub(crate) id: &'static str,
    pub(crate) capability: CapabilityKind,
    pub(crate) model_variant: ModelVariant,
    pub(crate) streaming_mode: StreamingMode,
    pub(crate) execution_target: ExecutionTargetKind,
    pub(crate) sequence_execution: SequenceExecutionMode,
    pub(crate) state_requirement: InferenceStateRequirement,
}

pub(crate) trait ModelCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata>;
}

#[derive(Debug)]
pub(crate) struct RuntimeAdapterRegistry {
    adapters: HashMap<(CapabilityKind, ModelVariant), AdapterMetadata>,
    max_tensor_batch_size: usize,
    request_parallelism: usize,
    loaded_adapter_factories: Vec<Arc<dyn loaded::LoadedExecutionAdapterFactory>>,
}

impl RuntimeAdapterRegistry {
    pub(crate) fn built_in() -> Self {
        Self::built_in_with_execution_limits(1, 1)
            .expect("the built-in native adapter registry must be unambiguous")
    }

    pub(crate) fn built_in_with_execution_limits(
        max_tensor_batch_size: usize,
        request_parallelism: usize,
    ) -> Result<Self> {
        let mut registry = Self {
            adapters: HashMap::new(),
            max_tensor_batch_size: max_tensor_batch_size.max(1),
            request_parallelism: request_parallelism.max(1),
            loaded_adapter_factories: loaded::built_in_loaded_adapter_factories(),
        };
        registry.register_adapter(TtsCapabilityAdapter);
        registry.register_adapter(StreamingTtsCapabilityAdapter);
        registry.register_adapter(AsrCapabilityAdapter);
        registry.register_adapter(SpeakerAttributedAsrCapabilityAdapter);
        registry.register_adapter(RealtimeAsrCapabilityAdapter);
        registry.register_adapter(ChatCapabilityAdapter);
        registry.register_adapter(AudioChatCapabilityAdapter);
        registry.register_adapter(SpeechToSpeechCapabilityAdapter);
        registry.register_adapter(DiarizationCapabilityAdapter);
        registry.register_adapter(ForcedAlignmentCapabilityAdapter);
        registry.register_adapter(TokenizerCapabilityAdapter);
        registry.validate_loaded_adapter_factories()?;
        Ok(registry)
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

    pub(crate) fn max_tensor_batch_size(&self) -> usize {
        self.max_tensor_batch_size
    }

    pub(crate) fn request_parallelism(&self) -> usize {
        self.request_parallelism
    }

    pub(crate) fn static_tensor_batch_variants(
        &self,
        backend_kind: BackendKind,
    ) -> HashSet<ModelVariant> {
        self.loaded_native_variants(backend_kind, NativeBatchMode::Static)
    }

    pub(crate) fn continuous_tensor_batch_variants(
        &self,
        backend_kind: BackendKind,
    ) -> HashSet<ModelVariant> {
        self.loaded_native_variants(backend_kind, NativeBatchMode::Continuous)
    }

    fn validate_loaded_adapter_factories(&self) -> Result<()> {
        const BACKENDS: [BackendKind; 3] =
            [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

        for metadata in self.adapters.values().copied() {
            for backend_kind in BACKENDS {
                self.loaded_adapter_factory(metadata, backend_kind)?;
            }
        }
        Ok(())
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

pub(crate) fn compatibility_execution_profile(
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
    streaming_required: bool,
) -> ExecutionProfile {
    let variant = metadata.model_variant;
    let sequence = metadata.sequence_execution.enabled(streaming_required);
    let mode = if sequence {
        ExecutionMode::Sequence
    } else {
        match metadata.execution_target {
            ExecutionTargetKind::RealtimeRunner => ExecutionMode::Realtime,
            ExecutionTargetKind::PipelineRunner => ExecutionMode::Pipeline,
            ExecutionTargetKind::Artifact => ExecutionMode::Artifact,
            ExecutionTargetKind::TokenEngine
            | ExecutionTargetKind::BatchRunner
            | ExecutionTargetKind::DirectModel => ExecutionMode::Atomic,
        }
    };
    let mut profile = ExecutionProfile::fail_closed(backend_kind, Some(variant), mode);
    if sequence {
        profile.prefill = PrefillMode::Full;
        profile.incremental_decode = true;
        profile.cache_mode = CacheMode::OpaqueModelOwned;
    }
    if metadata.capability == CapabilityKind::Asr {
        profile.cancellation = CancellationGranularity::OperationBoundary;
    }
    profile.prefill_batch = NativeBatchMode::None;
    profile.decode_batch = NativeBatchMode::None;
    profile.max_batch_size = 1;
    profile.compute_dtype = "loaded_model_default".to_string();
    profile.kv_dtype = if sequence {
        "loaded_model_default".to_string()
    } else {
        "none".to_string()
    };
    profile.cache_namespace =
        sequence.then(|| format!("{}:{}:opaque", variant, backend_kind.as_str()));
    profile
}

fn tts_execution_target(model_variant: ModelVariant) -> ExecutionTargetKind {
    if model_variant.is_kokoro()
        || model_variant.is_lfm25_audio_gguf()
        || matches!(
            model_variant.family(),
            crate::catalog::ModelFamily::VoxtralTts
                | crate::catalog::ModelFamily::VibeVoiceTts
                | crate::catalog::ModelFamily::FishS2Tts
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
    if capabilities.supports_streaming {
        StreamingMode::Chunked
    } else if matches!(
        model_variant.family(),
        crate::catalog::ModelFamily::Lfm25Audio
            | crate::catalog::ModelFamily::VoxtralTts
            | crate::catalog::ModelFamily::VibeVoiceTts
            | crate::catalog::ModelFamily::FishS2Tts
    ) {
        StreamingMode::FinalOnly
    } else {
        StreamingMode::None
    }
}

fn asr_execution_target(model_variant: ModelVariant) -> ExecutionTargetKind {
    if model_variant.is_audio_chat() {
        ExecutionTargetKind::DirectModel
    } else {
        ExecutionTargetKind::TokenEngine
    }
}

fn chat_sequence_execution(model_variant: ModelVariant) -> SequenceExecutionMode {
    if matches!(model_variant.family(), ModelFamily::Qwen35Chat)
        || matches!(
            model_variant,
            ModelVariant::Qwen306B
                | ModelVariant::Qwen306B4Bit
                | ModelVariant::Qwen317B
                | ModelVariant::Qwen317B4Bit
        )
    {
        SequenceExecutionMode::Always
    } else {
        SequenceExecutionMode::None
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
            sequence_execution: if model_variant.family() == ModelFamily::Qwen3Tts {
                SequenceExecutionMode::Always
            } else {
                SequenceExecutionMode::None
            },
            state_requirement: match model_variant.family() {
                ModelFamily::Qwen3Tts => InferenceStateRequirement::RetainedAndInvocation,
                ModelFamily::KokoroTts => InferenceStateRequirement::Stateless,
                _ => InferenceStateRequirement::Invocation,
            },
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
            sequence_execution: if model_variant.family() == ModelFamily::Qwen3Tts {
                SequenceExecutionMode::Always
            } else {
                SequenceExecutionMode::None
            },
            state_requirement: match model_variant.family() {
                ModelFamily::Qwen3Tts => InferenceStateRequirement::RetainedAndInvocation,
                ModelFamily::KokoroTts => InferenceStateRequirement::Stateless,
                _ => InferenceStateRequirement::Invocation,
            },
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
                streaming_mode: if model_variant.is_audio_chat()
                    || model_variant.is_voxtral()
                    || model_variant.family() == crate::catalog::ModelFamily::Qwen3Asr
                {
                    StreamingMode::Chunked
                } else {
                    StreamingMode::None
                },
                execution_target: asr_execution_target(model_variant),
                sequence_execution: if model_variant.family() == ModelFamily::Qwen3Asr {
                    SequenceExecutionMode::StreamingOnly
                } else {
                    SequenceExecutionMode::None
                },
                state_requirement: if model_variant.family() == ModelFamily::Qwen3Asr {
                    InferenceStateRequirement::Retained
                } else {
                    InferenceStateRequirement::Invocation
                },
            })
    }
}

#[derive(Debug, Clone, Copy)]
struct SpeakerAttributedAsrCapabilityAdapter;

impl ModelCapabilityAdapter for SpeakerAttributedAsrCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant
            .supports_speaker_attributed_asr()
            .then_some(AdapterMetadata {
                id: "builtin.speaker_attributed_asr",
                capability: CapabilityKind::SpeakerAttributedAsr,
                model_variant,
                streaming_mode: StreamingMode::None,
                execution_target: ExecutionTargetKind::PipelineRunner,
                sequence_execution: SequenceExecutionMode::None,
                state_requirement: InferenceStateRequirement::Invocation,
            })
    }
}

#[derive(Debug, Clone, Copy)]
struct RealtimeAsrCapabilityAdapter;

impl ModelCapabilityAdapter for RealtimeAsrCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        (model_variant == ModelVariant::Nemotron35AsrStreaming06B).then_some(AdapterMetadata {
            id: "builtin.realtime_asr",
            capability: CapabilityKind::RealtimeAsr,
            model_variant,
            streaming_mode: StreamingMode::Realtime,
            execution_target: ExecutionTargetKind::RealtimeRunner,
            sequence_execution: SequenceExecutionMode::None,
            state_requirement: InferenceStateRequirement::RetainedAndInvocation,
        })
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
            sequence_execution: chat_sequence_execution(model_variant),
            state_requirement: InferenceStateRequirement::Retained,
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
            sequence_execution: SequenceExecutionMode::None,
            state_requirement: InferenceStateRequirement::Invocation,
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
            sequence_execution: SequenceExecutionMode::None,
            state_requirement: InferenceStateRequirement::Invocation,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DiarizationCapabilityAdapter;

impl ModelCapabilityAdapter for DiarizationCapabilityAdapter {
    fn metadata_for(&self, model_variant: ModelVariant) -> Option<AdapterMetadata> {
        model_variant
            .supports_diarization_records()
            .then_some(AdapterMetadata {
                id: "builtin.diarization",
                capability: CapabilityKind::Diarization,
                model_variant,
                streaming_mode: StreamingMode::None,
                execution_target: ExecutionTargetKind::PipelineRunner,
                sequence_execution: SequenceExecutionMode::None,
                state_requirement: InferenceStateRequirement::Invocation,
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
                sequence_execution: SequenceExecutionMode::None,
                state_requirement: InferenceStateRequirement::Stateless,
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
            sequence_execution: SequenceExecutionMode::None,
            state_requirement: InferenceStateRequirement::Stateless,
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
        if model_variant.supports_speaker_attributed_asr() {
            expected.insert(CapabilityKind::SpeakerAttributedAsr);
        }
        if model_variant == ModelVariant::Nemotron35AsrStreaming06B {
            expected.insert(CapabilityKind::RealtimeAsr);
        }
        if model_variant.is_chat() {
            expected.insert(CapabilityKind::Chat);
        }
        if model_variant.is_audio_chat() {
            expected.insert(CapabilityKind::AudioChat);
            expected.insert(CapabilityKind::SpeechToSpeech);
        }
        if model_variant.supports_diarization_records() {
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
        assert_eq!(lfm.streaming_mode, StreamingMode::FinalOnly);
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
    fn capability_metadata_owns_compatibility_sequence_semantics() {
        let registry = RuntimeAdapterRegistry::built_in();
        let qwen_chat = *registry
            .require(CapabilityKind::Chat, ModelVariant::Qwen306B)
            .unwrap();
        let gemma_chat = *registry
            .require(CapabilityKind::Chat, ModelVariant::Gemma31BIt)
            .unwrap();
        let qwen_tts = *registry
            .require(CapabilityKind::Tts, ModelVariant::Qwen3Tts12Hz06BBase)
            .unwrap();
        let qwen_asr = *registry
            .require(CapabilityKind::Asr, ModelVariant::Qwen3Asr06BGguf)
            .unwrap();

        assert_eq!(qwen_chat.sequence_execution, SequenceExecutionMode::Always);
        assert_eq!(qwen_tts.sequence_execution, SequenceExecutionMode::Always);
        assert_eq!(
            qwen_asr.sequence_execution,
            SequenceExecutionMode::StreamingOnly
        );
        assert_eq!(gemma_chat.sequence_execution, SequenceExecutionMode::None);
        assert_eq!(
            compatibility_execution_profile(qwen_asr, BackendKind::Cpu, false).mode,
            ExecutionMode::Atomic
        );
        assert_eq!(
            compatibility_execution_profile(qwen_asr, BackendKind::Cpu, true).mode,
            ExecutionMode::Sequence
        );
    }

    #[test]
    fn physical_qwen_tts_factory_does_not_publish_the_removed_static_route() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();

        assert_eq!(registry.max_tensor_batch_size(), 4);
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let variants = registry.static_tensor_batch_variants(backend);
            assert!(!variants.contains(&variant));
            assert!(!variants.contains(&ModelVariant::Qwen306B));
        }
    }

    #[test]
    fn native_continuous_factories_publish_supported_variants_on_every_backend() {
        let variant = ModelVariant::Qwen306B;
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let variants = registry.continuous_tensor_batch_variants(backend);
            assert!(variants.contains(&variant));
            assert!(!variants.contains(&ModelVariant::Qwen3508BGguf));
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
            ExecutionTargetKind::TokenEngine
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
    fn built_in_registry_routes_voxtral_streaming_through_the_token_engine() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::VoxtralMini4BRealtime2602;

        assert_eq!(
            registry
                .require(CapabilityKind::Asr, variant)
                .expect("voxtral asr adapter")
                .execution_target,
            ExecutionTargetKind::TokenEngine
        );
        assert_eq!(
            registry
                .require(CapabilityKind::Asr, variant)
                .expect("voxtral asr adapter")
                .streaming_mode,
            StreamingMode::Chunked
        );
        assert!(registry
            .require(CapabilityKind::RealtimeAsr, variant)
            .is_err());
        assert!(registry
            .require(CapabilityKind::AudioChat, variant)
            .is_err());
        assert!(registry
            .require(CapabilityKind::SpeechToSpeech, variant)
            .is_err());
    }

    #[test]
    fn built_in_registry_separates_granite_asr_and_speaker_attribution_execution() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::GraniteSpeech412BPlus;

        let adapter = registry
            .require(CapabilityKind::Asr, variant)
            .expect("granite speech asr adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::TokenEngine);
        assert_eq!(adapter.streaming_mode, StreamingMode::None);
        assert_eq!(
            registry
                .require(CapabilityKind::SpeakerAttributedAsr, variant)
                .expect("granite speaker-attributed ASR adapter")
                .execution_target,
            ExecutionTargetKind::PipelineRunner
        );
        assert!(registry
            .require(CapabilityKind::Diarization, variant)
            .is_err());
        assert!(registry
            .require(CapabilityKind::RealtimeAsr, variant)
            .is_err());
        assert!(registry
            .require(CapabilityKind::AudioChat, variant)
            .is_err());
    }

    #[test]
    fn built_in_registry_marks_voxtral_tts_as_direct_tts_with_final_only_streaming() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::Voxtral4BTts2603;

        let adapter = registry
            .require(CapabilityKind::Tts, variant)
            .expect("voxtral tts adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::DirectModel);
        assert_eq!(adapter.streaming_mode, StreamingMode::FinalOnly);
        assert!(registry
            .require(CapabilityKind::StreamingTts, variant)
            .is_err());
    }

    #[test]
    fn built_in_registry_marks_vibevoice_tts_as_direct_tts_with_final_only_streaming() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::VibeVoice15BTts;

        let adapter = registry
            .require(CapabilityKind::Tts, variant)
            .expect("vibevoice tts adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::DirectModel);
        assert_eq!(adapter.streaming_mode, StreamingMode::FinalOnly);
        assert!(registry
            .require(CapabilityKind::StreamingTts, variant)
            .is_err());
    }

    #[test]
    fn built_in_registry_marks_fish_s2_as_direct_final_only_tts() {
        let registry = RuntimeAdapterRegistry::built_in();
        let variant = ModelVariant::FishAudioS2Pro;

        let adapter = registry
            .require(CapabilityKind::Tts, variant)
            .expect("Fish S2 TTS adapter");
        assert_eq!(adapter.execution_target, ExecutionTargetKind::DirectModel);
        assert_eq!(adapter.streaming_mode, StreamingMode::FinalOnly);
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

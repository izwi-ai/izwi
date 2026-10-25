//! The runtime's single capability truth.
//!
//! One row per model family and capability (narrowed to some of the family's
//! variants where its checkpoints differ). The adapter registry's metadata,
//! the loaded-adapter factory each backend binds, the coordinator lane, the
//! realtime ASR path and the conformance manifest are all read from these
//! rows; nothing else lists families per capability.

use crate::backends::BackendKind;
use crate::catalog::{ModelFamily, ModelVariant};

use super::{
    AdapterMetadata, CapabilityKind, ExecutionTargetKind, InferenceStateRequirement,
    SequenceExecutionMode, StreamingMode,
};

/// The built-in loaded-adapter factory that executes a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadedAdapterKind {
    QwenTtsSequence,
    NemotronRealtime,
    VoxtralRealtime,
    ContinuousChat,
    Qwen3AsrContinuous,
    WhisperAsr,
    VibeVoiceAsr,
    GraniteSpeechAsr,
    Lfm25AudioAsr,
    ParakeetAsr,
    Lfm25AudioTts,
    VibeVoiceTts,
    FishS2Tts,
    VoxtralTts,
    KokoroStaticTts,
    Scalar,
}

impl LoadedAdapterKind {
    pub(crate) const fn factory_id(self) -> &'static str {
        match self {
            Self::QwenTtsSequence => "builtin.qwen_tts.physical_sequence",
            Self::NemotronRealtime => "builtin.nemotron_realtime.physical_tensor",
            Self::VoxtralRealtime => "builtin.voxtral_realtime.physical_paged",
            Self::ContinuousChat => "builtin.physical_chat.tensor_continuous",
            Self::Qwen3AsrContinuous => "builtin.qwen3_asr.tensor_continuous",
            Self::WhisperAsr => "builtin.whisper_asr.physical_sequence",
            Self::VibeVoiceAsr => "builtin.vibevoice_asr.physical_sequence",
            Self::GraniteSpeechAsr => "builtin.granite_speech_asr.physical_sequence",
            Self::Lfm25AudioAsr => "builtin.lfm25_audio_asr.physical_sequence",
            Self::ParakeetAsr => "builtin.parakeet_asr.physical_recurrent",
            Self::Lfm25AudioTts => "builtin.lfm25_audio_tts.physical_sequence",
            Self::VibeVoiceTts => "builtin.vibevoice_tts.physical_sequence",
            Self::FishS2Tts => "builtin.fish_s2_tts.physical_sequence",
            Self::VoxtralTts => "builtin.voxtral_tts.physical_sequence",
            Self::KokoroStaticTts => "builtin.kokoro_tts.tensor_static",
            Self::Scalar => "builtin.scalar",
        }
    }
}

/// Where a realtime ASR session executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RealtimeAsrPath {
    /// The engine's realtime session admission, which needs a paged managed
    /// state runtime.
    Engine,
    /// The model's direct native streaming path.
    Direct,
}

const ALL_BACKENDS: &[BackendKind] = &[BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapabilityRow {
    pub(crate) family: ModelFamily,
    pub(crate) capability: CapabilityKind,
    /// The family's variants this row serves; `None` serves all of them.
    pub(crate) variants: Option<&'static [ModelVariant]>,
    pub(crate) streaming_mode: StreamingMode,
    pub(crate) execution_target: ExecutionTargetKind,
    pub(crate) sequence_execution: SequenceExecutionMode,
    pub(crate) state_requirement: InferenceStateRequirement,
    pub(crate) loaded_adapter: LoadedAdapterKind,
    /// Backends the loaded adapter serves.
    pub(crate) backends: &'static [BackendKind],
    /// The coordinator may run this capability in its resumable lane.
    pub(crate) resumable: bool,
    /// Realtime ASR rows only: where a session executes.
    pub(crate) realtime_path: Option<RealtimeAsrPath>,
}

impl CapabilityRow {
    const fn new(
        family: ModelFamily,
        capability: CapabilityKind,
        loaded_adapter: LoadedAdapterKind,
        state_requirement: InferenceStateRequirement,
    ) -> Self {
        Self {
            family,
            capability,
            variants: None,
            streaming_mode: StreamingMode::None,
            execution_target: ExecutionTargetKind::TokenEngine,
            sequence_execution: SequenceExecutionMode::None,
            state_requirement,
            loaded_adapter,
            backends: ALL_BACKENDS,
            resumable: false,
            realtime_path: None,
        }
    }

    const fn only(mut self, variants: &'static [ModelVariant]) -> Self {
        self.variants = Some(variants);
        self
    }

    const fn chunked(mut self) -> Self {
        self.streaming_mode = StreamingMode::Chunked;
        self
    }

    const fn target(mut self, target: ExecutionTargetKind) -> Self {
        self.execution_target = target;
        self
    }

    const fn sequence(mut self) -> Self {
        self.sequence_execution = SequenceExecutionMode::Always;
        self
    }

    const fn resumable(mut self) -> Self {
        self.resumable = true;
        self
    }

    const fn realtime(mut self, path: RealtimeAsrPath) -> Self {
        self.streaming_mode = StreamingMode::Realtime;
        self.execution_target = ExecutionTargetKind::RealtimeRunner;
        self.realtime_path = Some(path);
        self
    }

    pub(crate) fn serves(&self, variant: ModelVariant) -> bool {
        variant.family() == self.family
            && self
                .variants
                .is_none_or(|variants| variants.contains(&variant))
    }

    pub(crate) fn metadata_for(&self, model_variant: ModelVariant) -> AdapterMetadata {
        AdapterMetadata {
            id: self.capability.adapter_id(),
            capability: self.capability,
            model_variant,
            streaming_mode: self.streaming_mode,
            execution_target: self.execution_target,
            sequence_execution: self.sequence_execution,
            state_requirement: self.state_requirement,
        }
    }
}

/// Qwen3 dense checkpoints with resumable chunked prefill; the 14B GGUF
/// prefills in one step.
const QWEN3_RESUMABLE_CHAT: &[ModelVariant] = &[
    ModelVariant::Qwen306B,
    ModelVariant::Qwen306B4Bit,
    ModelVariant::Qwen317B,
    ModelVariant::Qwen317B4Bit,
];
const QWEN3_ATOMIC_CHAT: &[ModelVariant] = &[ModelVariant::Qwen314BGguf];
/// Only plus emits speaker turns; the base checkpoint serves plain ASR.
const GRANITE_SPEAKER_ATTRIBUTED: &[ModelVariant] = &[ModelVariant::GraniteSpeech412BPlus];

pub(crate) const CAPABILITY_TABLE: &[CapabilityRow] = {
    use CapabilityKind as C;
    use ExecutionTargetKind as Target;
    use InferenceStateRequirement::{Invocation, Retained, RetainedAndInvocation, Stateless};
    use LoadedAdapterKind as Loaded;
    use ModelFamily as F;
    const fn row(
        family: ModelFamily,
        capability: CapabilityKind,
        loaded_adapter: LoadedAdapterKind,
        state_requirement: InferenceStateRequirement,
    ) -> CapabilityRow {
        CapabilityRow::new(family, capability, loaded_adapter, state_requirement)
    }
    const fn chat(family: ModelFamily) -> CapabilityRow {
        row(family, C::Chat, Loaded::ContinuousChat, Retained)
            .chunked()
            .sequence()
    }

    &[
        // Text to speech.
        row(
            F::Qwen3Tts,
            C::Tts,
            Loaded::QwenTtsSequence,
            RetainedAndInvocation,
        )
        .chunked()
        .sequence()
        .resumable(),
        row(
            F::Qwen3Tts,
            C::StreamingTts,
            Loaded::QwenTtsSequence,
            RetainedAndInvocation,
        )
        .chunked()
        .sequence(),
        row(F::KokoroTts, C::Tts, Loaded::KokoroStaticTts, Stateless).chunked(),
        row(F::KokoroTts, C::StreamingTts, Loaded::Scalar, Stateless)
            .chunked()
            .target(Target::DirectModel),
        row(
            F::VibeVoiceTts,
            C::Tts,
            Loaded::VibeVoiceTts,
            RetainedAndInvocation,
        )
        .chunked(),
        row(
            F::VoxtralTts,
            C::Tts,
            Loaded::VoxtralTts,
            RetainedAndInvocation,
        )
        .chunked(),
        row(
            F::FishS2Tts,
            C::Tts,
            Loaded::FishS2Tts,
            RetainedAndInvocation,
        )
        .chunked(),
        row(
            F::FishS2Tts,
            C::StreamingTts,
            Loaded::FishS2Tts,
            RetainedAndInvocation,
        )
        .chunked(),
        row(
            F::Lfm25Audio,
            C::Tts,
            Loaded::Lfm25AudioTts,
            RetainedAndInvocation,
        )
        .chunked()
        .resumable(),
        // Speech recognition.
        row(
            F::ParakeetAsr,
            C::Asr,
            Loaded::ParakeetAsr,
            RetainedAndInvocation,
        ),
        row(
            F::WhisperAsr,
            C::Asr,
            Loaded::WhisperAsr,
            RetainedAndInvocation,
        )
        .sequence()
        .resumable(),
        row(
            F::Qwen3Asr,
            C::Asr,
            Loaded::Qwen3AsrContinuous,
            RetainedAndInvocation,
        )
        .chunked()
        .sequence()
        .resumable(),
        row(
            F::VibeVoiceAsr,
            C::Asr,
            Loaded::VibeVoiceAsr,
            RetainedAndInvocation,
        )
        .sequence()
        .resumable(),
        row(F::NemotronAsr, C::Asr, Loaded::Scalar, Invocation),
        row(
            F::NemotronAsr,
            C::RealtimeAsr,
            Loaded::NemotronRealtime,
            RetainedAndInvocation,
        )
        .realtime(RealtimeAsrPath::Direct),
        row(
            F::GraniteSpeechAsr,
            C::Asr,
            Loaded::GraniteSpeechAsr,
            RetainedAndInvocation,
        )
        .sequence()
        .resumable(),
        row(
            F::GraniteSpeechAsr,
            C::SpeakerAttributedAsr,
            Loaded::Scalar,
            Invocation,
        )
        .only(GRANITE_SPEAKER_ATTRIBUTED)
        .target(Target::PipelineRunner),
        // One atomic invocation per chunk on leased decoder pages; long
        // audio dispatches one chunk at a time.
        row(F::GraniteSpeech5Asr, C::Asr, Loaded::Scalar, Invocation).chunked(),
        // Long audio is chunked inside the dispatch.
        row(F::Voxtral, C::Asr, Loaded::Scalar, Invocation).chunked(),
        row(
            F::Voxtral,
            C::RealtimeAsr,
            Loaded::VoxtralRealtime,
            RetainedAndInvocation,
        )
        .realtime(RealtimeAsrPath::Engine),
        row(
            F::Lfm25Audio,
            C::Asr,
            Loaded::Lfm25AudioAsr,
            RetainedAndInvocation,
        )
        .chunked()
        .sequence()
        .resumable(),
        // Audio chat.
        row(F::Lfm25Audio, C::AudioChat, Loaded::Scalar, Invocation).chunked(),
        row(F::Lfm25Audio, C::SpeechToSpeech, Loaded::Scalar, Invocation).chunked(),
        // Chat.
        chat(F::Qwen3Chat).only(QWEN3_RESUMABLE_CHAT).resumable(),
        chat(F::Qwen3Chat).only(QWEN3_ATOMIC_CHAT),
        chat(F::Qwen3MoeChat),
        chat(F::Qwen35Chat).resumable(),
        chat(F::Qwen35MoeChat),
        chat(F::Qwen38Chat).resumable(),
        chat(F::Lfm2Chat),
        chat(F::Gemma3Chat),
        // Pipelines and artifacts.
        row(
            F::SortformerDiarization,
            C::Diarization,
            Loaded::Scalar,
            Invocation,
        )
        .target(Target::PipelineRunner),
        row(
            F::Qwen3ForcedAligner,
            C::ForcedAlignment,
            Loaded::Scalar,
            Stateless,
        )
        .target(Target::BatchRunner),
        row(F::Tokenizer, C::Tokenizer, Loaded::Scalar, Stateless).target(Target::Artifact),
    ]
};

/// The row that serves `capability` on `variant`, if the model has it.
pub(crate) fn capability_row(
    capability: CapabilityKind,
    variant: ModelVariant,
) -> Option<&'static CapabilityRow> {
    CAPABILITY_TABLE
        .iter()
        .find(|row| row.capability == capability && row.serves(variant))
}

/// Every row that serves `variant`.
pub(crate) fn capability_rows(
    variant: ModelVariant,
) -> impl Iterator<Item = &'static CapabilityRow> {
    CAPABILITY_TABLE
        .iter()
        .filter(move |row| row.serves(variant))
}

/// Whether the table binds `metadata` on `backend_kind` to the factory of
/// `kind`; each built-in factory selects exactly its rows this way.
pub(crate) fn table_selects(
    kind: LoadedAdapterKind,
    metadata: AdapterMetadata,
    backend_kind: BackendKind,
) -> bool {
    capability_row(metadata.capability, metadata.model_variant)
        .is_some_and(|row| row.loaded_adapter == kind && row.backends.contains(&backend_kind))
}

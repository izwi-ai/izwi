//! Capability conformance contracts for runtime architecture migrations.
//!
//! These contracts are intentionally lightweight. They give integration tests
//! and future fake adapters a stable checklist before execution behavior is
//! moved behind new broker, adapter, or pipeline boundaries.

use crate::catalog::{CudaSupportLevel, ModelFamily, ModelVariant};

use super::adapters::{CapabilityKind, RuntimeAdapterRegistry};

pub(crate) const EXPECTED_CATALOG_VARIANT_COUNT: usize = 50;
pub(crate) const EXPECTED_CATALOG_CAPABILITY_BINDING_COUNT: usize = 72;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConformanceCapability {
    Tts,
    StreamingTts,
    Asr,
    SpeakerAttributedAsr,
    RealtimeAsr,
    Chat,
    AudioChat,
    SpeechToSpeech,
    Diarization,
    ForcedAlignment,
    Vad,
    Endpointing,
    Tokenizer,
}

impl ConformanceCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tts => "tts",
            Self::StreamingTts => "streaming_tts",
            Self::Asr => "asr",
            Self::SpeakerAttributedAsr => "speaker_attributed_asr",
            Self::RealtimeAsr => "realtime_asr",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceExecutionClass {
    Scheduled,
    Streaming,
    Realtime,
    Batch,
    Pipeline,
    Artifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityConformanceCase {
    pub id: &'static str,
    pub capability: ConformanceCapability,
    pub execution_class: ConformanceExecutionClass,
    pub fixture: &'static str,
}

impl CapabilityConformanceCase {
    pub const fn new(
        id: &'static str,
        capability: ConformanceCapability,
        execution_class: ConformanceExecutionClass,
        fixture: &'static str,
    ) -> Self {
        Self {
            id,
            capability,
            execution_class,
            fixture,
        }
    }
}

pub const REQUIRED_CONFORMANCE_CAPABILITIES: &[ConformanceCapability] = &[
    ConformanceCapability::Tts,
    ConformanceCapability::StreamingTts,
    ConformanceCapability::Asr,
    ConformanceCapability::SpeakerAttributedAsr,
    ConformanceCapability::RealtimeAsr,
    ConformanceCapability::Chat,
    ConformanceCapability::AudioChat,
    ConformanceCapability::SpeechToSpeech,
    ConformanceCapability::Diarization,
    ConformanceCapability::ForcedAlignment,
    ConformanceCapability::Vad,
    ConformanceCapability::Endpointing,
    ConformanceCapability::Tokenizer,
];

pub const CAPABILITY_CONFORMANCE_CASES: &[CapabilityConformanceCase] = &[
    CapabilityConformanceCase::new(
        "tts.short_text.binary",
        ConformanceCapability::Tts,
        ConformanceExecutionClass::Scheduled,
        "short text with explicit model, voice option, and binary response format",
    ),
    CapabilityConformanceCase::new(
        "streaming_tts.short_text",
        ConformanceCapability::StreamingTts,
        ConformanceExecutionClass::Streaming,
        "short text with first audio chunk, multiple chunks, and terminal event",
    ),
    CapabilityConformanceCase::new(
        "asr.short_wav.transcript",
        ConformanceCapability::Asr,
        ConformanceExecutionClass::Batch,
        "short wav input with transcript text, language hint, and format mapping",
    ),
    CapabilityConformanceCase::new(
        "speaker_attributed_asr.short_multispeaker",
        ConformanceCapability::SpeakerAttributedAsr,
        ConformanceExecutionClass::Pipeline,
        "short multi-speaker audio producing attributed transcript turns",
    ),
    CapabilityConformanceCase::new(
        "realtime_asr.partial_final",
        ConformanceCapability::RealtimeAsr,
        ConformanceExecutionClass::Realtime,
        "pcm frames with partial update, final update, cancellation, and close",
    ),
    CapabilityConformanceCase::new(
        "chat.single_prompt.streaming",
        ConformanceCapability::Chat,
        ConformanceExecutionClass::Scheduled,
        "single user prompt with non-streaming and streaming delta responses",
    ),
    CapabilityConformanceCase::new(
        "audio_chat.audio_prompt.response",
        ConformanceCapability::AudioChat,
        ConformanceExecutionClass::Scheduled,
        "audio bytes plus optional text prompt returning text and/or audio output",
    ),
    CapabilityConformanceCase::new(
        "speech_to_speech.audio_stream",
        ConformanceCapability::SpeechToSpeech,
        ConformanceExecutionClass::Streaming,
        "audio request to streaming audio response with cancellation",
    ),
    CapabilityConformanceCase::new(
        "diarization.short_multispeaker",
        ConformanceCapability::Diarization,
        ConformanceExecutionClass::Pipeline,
        "short multi-speaker fixture with speaker labels and ASR attribution",
    ),
    CapabilityConformanceCase::new(
        "forced_alignment.words",
        ConformanceCapability::ForcedAlignment,
        ConformanceExecutionClass::Batch,
        "transcript plus audio fixture producing ordered word timestamps",
    ),
    CapabilityConformanceCase::new(
        "voice.vad.speech_events",
        ConformanceCapability::Vad,
        ConformanceExecutionClass::Realtime,
        "pcm frames producing speech start and speech end events",
    ),
    CapabilityConformanceCase::new(
        "voice.endpointing.turn_boundary",
        ConformanceCapability::Endpointing,
        ConformanceExecutionClass::Realtime,
        "speech activity stream producing stable turn boundary decisions",
    ),
    CapabilityConformanceCase::new(
        "tokenizer.model_artifact.round_trip",
        ConformanceCapability::Tokenizer,
        ConformanceExecutionClass::Artifact,
        "tokenizer artifact load with deterministic encode/decode smoke check",
    ),
];

pub fn capability_conformance_cases() -> &'static [CapabilityConformanceCase] {
    CAPABILITY_CONFORMANCE_CASES
}

pub fn required_conformance_capabilities() -> &'static [ConformanceCapability] {
    REQUIRED_CONFORMANCE_CAPABILITIES
}

/// Target retained-state ownership recorded before the physical-state
/// migration starts. This is descriptive scaffolding only; it does not select
/// an execution path.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetainedStateExpectation {
    Managed,
    Stateless,
}

/// Invocation workspace requirement recorded independently from retained
/// state. Every capability must eventually publish a workspace contract, while
/// atomic capabilities with internal iterative state require a non-empty one.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkspaceExpectation {
    DeclaredMayBeEmpty,
    InvocationStateRequired,
}

/// Frozen product expectation for an exact model/capability/backend cell.
/// `RequiredWhenCompiled` still permits normal runtime hardware admission to
/// reject an unavailable or unsupported physical device.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendConformanceExpectation {
    RequiredWhenCompiled,
    Unsupported,
    CatalogDisabled,
    NotApplicable,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackendConformanceExpectations {
    pub(crate) cpu: BackendConformanceExpectation,
    pub(crate) metal: BackendConformanceExpectation,
    pub(crate) cuda: BackendConformanceExpectation,
}

/// One entry per runtime-registry model/capability binding. The binding set is
/// derived from `ModelVariant::all()` and the built-in adapter registry so a
/// catalog addition cannot silently escape conformance coverage.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CatalogStateConformanceCase {
    pub(crate) variant: ModelVariant,
    pub(crate) capability: ConformanceCapability,
    pub(crate) catalog_enabled: bool,
    pub(crate) retained_state: RetainedStateExpectation,
    pub(crate) workspace: WorkspaceExpectation,
    pub(crate) backends: BackendConformanceExpectations,
    pub(crate) fixture_id: &'static str,
}

/// Build the Phase-0 catalog/state/backend manifest without affecting model
/// loading, routing, or execution.
#[allow(dead_code)]
pub(crate) fn catalog_state_conformance_manifest() -> Vec<CatalogStateConformanceCase> {
    let registry = RuntimeAdapterRegistry::built_in();
    let mut manifest = Vec::with_capacity(EXPECTED_CATALOG_CAPABILITY_BINDING_COUNT);

    for variant in ModelVariant::all().iter().copied() {
        for metadata in registry.capabilities_for(variant) {
            let capability = conformance_capability(metadata.capability);
            let retained_state = retained_state_expectation(variant, capability);

            manifest.push(CatalogStateConformanceCase {
                variant,
                capability,
                catalog_enabled: variant.is_enabled(),
                retained_state,
                workspace: workspace_expectation(variant, capability),
                backends: backend_expectations(variant),
                fixture_id: fixture_id_for(capability),
            });
        }
    }

    manifest
}

fn workspace_expectation(
    variant: ModelVariant,
    capability: ConformanceCapability,
) -> WorkspaceExpectation {
    use ConformanceCapability as Capability;
    use WorkspaceExpectation::{DeclaredMayBeEmpty, InvocationStateRequired};

    match (variant.family(), capability) {
        // The tokenizer is an artifact operation, not a tensor execution
        // capability. It must still publish a workspace contract, but an empty
        // resolved contract is valid.
        (ModelFamily::Tokenizer, Capability::Tokenizer)
        | (ModelFamily::Qwen3ForcedAligner, Capability::ForcedAlignment)
        | (ModelFamily::KokoroTts, Capability::Tts | Capability::StreamingTts) => {
            DeclaredMayBeEmpty
        }

        // Qwen TTS has a frame-local code-predictor cache in addition to its
        // retained talker state. Nemotron realtime has chunk-local staging in
        // addition to its retained streaming state. These entries keep the
        // retained-state and invocation-workspace axes independent.
        (ModelFamily::Qwen3Tts, Capability::Tts | Capability::StreamingTts)
        | (ModelFamily::NemotronAsr, Capability::RealtimeAsr) => InvocationStateRequired,

        // Current atomic model implementations keep their cache-like numeric
        // state within one invocation. Their final stateless declaration is
        // therefore paired with an explicit physical workspace contract.
        (_, _)
            if retained_state_expectation(variant, capability)
                == RetainedStateExpectation::Stateless =>
        {
            InvocationStateRequired
        }

        // Other managed capabilities may still resolve ordinary kernel
        // scratch, but do not require a cache-like invocation-state domain.
        _ => DeclaredMayBeEmpty,
    }
}

fn conformance_capability(capability: CapabilityKind) -> ConformanceCapability {
    match capability {
        CapabilityKind::Asr => ConformanceCapability::Asr,
        CapabilityKind::SpeakerAttributedAsr => ConformanceCapability::SpeakerAttributedAsr,
        CapabilityKind::RealtimeAsr => ConformanceCapability::RealtimeAsr,
        CapabilityKind::Tts => ConformanceCapability::Tts,
        CapabilityKind::StreamingTts => ConformanceCapability::StreamingTts,
        CapabilityKind::Chat => ConformanceCapability::Chat,
        CapabilityKind::AudioChat => ConformanceCapability::AudioChat,
        CapabilityKind::SpeechToSpeech => ConformanceCapability::SpeechToSpeech,
        CapabilityKind::Diarization => ConformanceCapability::Diarization,
        CapabilityKind::ForcedAlignment => ConformanceCapability::ForcedAlignment,
        CapabilityKind::Vad => ConformanceCapability::Vad,
        CapabilityKind::Endpointing => ConformanceCapability::Endpointing,
        CapabilityKind::Tokenizer => ConformanceCapability::Tokenizer,
    }
}

fn fixture_id_for(capability: ConformanceCapability) -> &'static str {
    capability_conformance_cases()
        .iter()
        .find(|case| case.capability == capability)
        .map(|case| case.id)
        .unwrap_or_else(|| {
            panic!(
                "runtime capability {} has no conformance fixture",
                capability.as_str()
            )
        })
}

fn retained_state_expectation(
    variant: ModelVariant,
    capability: ConformanceCapability,
) -> RetainedStateExpectation {
    use ConformanceCapability as Capability;
    use RetainedStateExpectation::{Managed, Stateless};

    match variant.family() {
        ModelFamily::Qwen3Tts => match capability {
            Capability::Tts | Capability::StreamingTts => Managed,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::Qwen3Asr => match capability {
            Capability::Asr => Managed,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::Qwen3Chat
        | ModelFamily::Qwen35Chat
        | ModelFamily::Lfm2Chat
        | ModelFamily::Gemma3Chat => match capability {
            Capability::Chat => Managed,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::NemotronAsr => match capability {
            Capability::RealtimeAsr => Managed,
            Capability::Asr => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::KokoroTts => match capability {
            Capability::Tts | Capability::StreamingTts => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::VibeVoiceTts | ModelFamily::FishS2Tts => match capability {
            Capability::Tts => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::VoxtralTts => match capability {
            Capability::Tts => Managed,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::ParakeetAsr | ModelFamily::WhisperAsr | ModelFamily::VibeVoiceAsr => {
            match capability {
                Capability::Asr => Stateless,
                _ => unexpected_capability(variant, capability),
            }
        }
        ModelFamily::Voxtral => match capability {
            Capability::Asr => Managed,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::GraniteSpeechAsr => match capability {
            Capability::Asr | Capability::SpeakerAttributedAsr => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::SortformerDiarization => match capability {
            Capability::Diarization => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::Lfm25Audio => match capability {
            Capability::Asr
            | Capability::Tts
            | Capability::AudioChat
            | Capability::SpeechToSpeech => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::Qwen3ForcedAligner => match capability {
            Capability::ForcedAlignment => Stateless,
            _ => unexpected_capability(variant, capability),
        },
        ModelFamily::Tokenizer => match capability {
            Capability::Tokenizer => Stateless,
            _ => unexpected_capability(variant, capability),
        },
    }
}

fn unexpected_capability(
    variant: ModelVariant,
    capability: ConformanceCapability,
) -> RetainedStateExpectation {
    panic!(
        "runtime registry published unexpected capability {} for {variant}",
        capability.as_str()
    )
}

fn backend_expectations(variant: ModelVariant) -> BackendConformanceExpectations {
    use BackendConformanceExpectation as Expectation;

    if !variant.is_enabled() {
        return BackendConformanceExpectations {
            cpu: Expectation::CatalogDisabled,
            metal: Expectation::CatalogDisabled,
            cuda: Expectation::CatalogDisabled,
        };
    }

    if variant.family() == ModelFamily::Tokenizer {
        return BackendConformanceExpectations {
            cpu: Expectation::NotApplicable,
            metal: Expectation::NotApplicable,
            cuda: Expectation::NotApplicable,
        };
    }

    let cuda = match variant.cuda_support_level() {
        CudaSupportLevel::NativeCuda | CudaSupportLevel::CandleCudaGeneric => {
            Expectation::RequiredWhenCompiled
        }
        CudaSupportLevel::CpuOnly => Expectation::Unsupported,
        CudaSupportLevel::Disabled => Expectation::CatalogDisabled,
        CudaSupportLevel::Unknown => {
            panic!("enabled catalog variant {variant} has unknown CUDA support")
        }
    };

    BackendConformanceExpectations {
        cpu: Expectation::RequiredWhenCompiled,
        metal: Expectation::RequiredWhenCompiled,
        cuda,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    fn manifest_case(
        manifest: &[CatalogStateConformanceCase],
        variant: ModelVariant,
        capability: ConformanceCapability,
    ) -> CatalogStateConformanceCase {
        manifest
            .iter()
            .find(|case| case.variant == variant && case.capability == capability)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "missing catalog conformance case for {variant} / {}",
                    capability.as_str()
                )
            })
    }

    #[test]
    fn conformance_cases_cover_every_required_capability() {
        let covered = capability_conformance_cases()
            .iter()
            .map(|case| case.capability)
            .collect::<BTreeSet<_>>();

        for capability in required_conformance_capabilities() {
            assert!(
                covered.contains(capability),
                "missing conformance case for {}",
                capability.as_str()
            );
        }
    }

    #[test]
    fn conformance_case_ids_are_unique() {
        let mut ids = BTreeSet::new();
        for case in capability_conformance_cases() {
            assert!(
                ids.insert(case.id),
                "duplicate conformance case {}",
                case.id
            );
        }
    }

    #[test]
    fn catalog_state_manifest_covers_every_registry_binding() {
        let registry = RuntimeAdapterRegistry::built_in();
        let manifest = catalog_state_conformance_manifest();

        assert_eq!(
            ModelVariant::all().len(),
            EXPECTED_CATALOG_VARIANT_COUNT,
            "update the frozen Phase-0 variant count deliberately when the catalog changes"
        );
        assert_eq!(
            manifest.len(),
            EXPECTED_CATALOG_CAPABILITY_BINDING_COUNT,
            "update the frozen Phase-0 binding count deliberately when capabilities change"
        );

        let expected = ModelVariant::all()
            .iter()
            .copied()
            .flat_map(|variant| {
                registry
                    .capabilities_for(variant)
                    .into_iter()
                    .map(move |metadata| (variant, conformance_capability(metadata.capability)))
            })
            .collect::<HashSet<_>>();
        let actual = manifest
            .iter()
            .map(|case| (case.variant, case.capability))
            .collect::<HashSet<_>>();

        assert_eq!(actual.len(), manifest.len(), "duplicate manifest binding");
        assert_eq!(actual, expected);
        for variant in ModelVariant::all() {
            assert!(
                manifest.iter().any(|case| case.variant == *variant),
                "catalog variant {variant} has no state conformance binding"
            );
        }
    }

    #[test]
    fn catalog_state_manifest_freezes_state_and_backend_expectations() {
        use BackendConformanceExpectation as Backend;
        use RetainedStateExpectation::{Managed, Stateless};
        use WorkspaceExpectation::{DeclaredMayBeEmpty, InvocationStateRequired};

        let manifest = catalog_state_conformance_manifest();

        for case in &manifest {
            assert_eq!(case.catalog_enabled, case.variant.is_enabled());
            assert_eq!(
                case.workspace,
                workspace_expectation(case.variant, case.capability)
            );

            if !case.catalog_enabled {
                assert_eq!(case.backends.cpu, Backend::CatalogDisabled);
                assert_eq!(case.backends.metal, Backend::CatalogDisabled);
                assert_eq!(case.backends.cuda, Backend::CatalogDisabled);
            } else if case.variant.family() == ModelFamily::Tokenizer {
                assert_eq!(case.backends.cpu, Backend::NotApplicable);
                assert_eq!(case.backends.metal, Backend::NotApplicable);
                assert_eq!(case.backends.cuda, Backend::NotApplicable);
            } else {
                assert_eq!(case.backends.cpu, Backend::RequiredWhenCompiled);
                assert_eq!(case.backends.metal, Backend::RequiredWhenCompiled);
                assert_eq!(case.backends.cuda, Backend::RequiredWhenCompiled);
            }

            assert!(
                capability_conformance_cases()
                    .iter()
                    .any(|fixture| fixture.id == case.fixture_id
                        && fixture.capability == case.capability),
                "invalid fixture {} for {} / {}",
                case.fixture_id,
                case.variant,
                case.capability.as_str()
            );
        }

        assert_eq!(
            manifest
                .iter()
                .filter(|case| case.retained_state == Managed)
                .count(),
            52
        );
        assert_eq!(
            manifest
                .iter()
                .filter(|case| case.retained_state == Stateless)
                .count(),
            20
        );

        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen306BGguf,
                ConformanceCapability::Chat,
            )
            .retained_state,
            Managed
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen3Tts12Hz06BBase,
                ConformanceCapability::Tts,
            )
            .workspace,
            InvocationStateRequired
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen306BGguf,
                ConformanceCapability::Chat,
            )
            .workspace,
            DeclaredMayBeEmpty
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Nemotron35AsrStreaming06B,
                ConformanceCapability::RealtimeAsr,
            )
            .retained_state,
            Managed
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Nemotron35AsrStreaming06B,
                ConformanceCapability::Asr,
            )
            .retained_state,
            Stateless
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen3TtsTokenizer12Hz,
                ConformanceCapability::Tokenizer,
            )
            .workspace,
            DeclaredMayBeEmpty
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::GraniteSpeech412BPlus,
                ConformanceCapability::SpeakerAttributedAsr,
            )
            .retained_state,
            Stateless
        );
    }
}

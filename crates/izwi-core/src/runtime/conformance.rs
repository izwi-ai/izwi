//! Capability conformance contracts for runtime architecture migrations.
//!
//! These contracts are intentionally lightweight. They give integration tests
//! and future fake adapters a stable checklist before execution behavior is
//! moved behind new broker, adapter, or pipeline boundaries.

use super::adapters::CapabilityKind;
#[cfg(test)]
use super::adapters::{capability_row, RuntimeAdapterRegistry};
#[cfg(test)]
use crate::backends::BackendKind;
#[cfg(test)]
use crate::catalog::{CudaSupportLevel, ModelFamily, ModelVariant};

#[cfg(test)]
pub(crate) const EXPECTED_CATALOG_VARIANT_COUNT: usize = 47;
#[cfg(test)]
pub(crate) const EXPECTED_CATALOG_CAPABILITY_BINDING_COUNT: usize = 67;

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
    pub capability: CapabilityKind,
    pub execution_class: ConformanceExecutionClass,
    pub fixture: &'static str,
}

impl CapabilityConformanceCase {
    pub const fn new(
        id: &'static str,
        capability: CapabilityKind,
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

pub const REQUIRED_CONFORMANCE_CAPABILITIES: &[CapabilityKind] = &[
    CapabilityKind::Tts,
    CapabilityKind::StreamingTts,
    CapabilityKind::Asr,
    CapabilityKind::SpeakerAttributedAsr,
    CapabilityKind::RealtimeAsr,
    CapabilityKind::Chat,
    CapabilityKind::AudioChat,
    CapabilityKind::SpeechToSpeech,
    CapabilityKind::Diarization,
    CapabilityKind::ForcedAlignment,
    CapabilityKind::Vad,
    CapabilityKind::Endpointing,
    CapabilityKind::Tokenizer,
];

pub const CAPABILITY_CONFORMANCE_CASES: &[CapabilityConformanceCase] = &[
    CapabilityConformanceCase::new(
        "tts.short_text.binary",
        CapabilityKind::Tts,
        ConformanceExecutionClass::Scheduled,
        "short text with explicit model, voice option, and binary response format",
    ),
    CapabilityConformanceCase::new(
        "streaming_tts.short_text",
        CapabilityKind::StreamingTts,
        ConformanceExecutionClass::Streaming,
        "short text with first audio chunk, multiple chunks, and terminal event",
    ),
    CapabilityConformanceCase::new(
        "asr.short_wav.transcript",
        CapabilityKind::Asr,
        ConformanceExecutionClass::Batch,
        "short wav input with transcript text, language hint, and format mapping",
    ),
    CapabilityConformanceCase::new(
        "speaker_attributed_asr.short_multispeaker",
        CapabilityKind::SpeakerAttributedAsr,
        ConformanceExecutionClass::Pipeline,
        "short multi-speaker audio producing attributed transcript turns",
    ),
    CapabilityConformanceCase::new(
        "realtime_asr.partial_final",
        CapabilityKind::RealtimeAsr,
        ConformanceExecutionClass::Realtime,
        "pcm frames with partial update, final update, cancellation, and close",
    ),
    CapabilityConformanceCase::new(
        "chat.single_prompt.streaming",
        CapabilityKind::Chat,
        ConformanceExecutionClass::Scheduled,
        "single user prompt with non-streaming and streaming delta responses",
    ),
    CapabilityConformanceCase::new(
        "audio_chat.audio_prompt.response",
        CapabilityKind::AudioChat,
        ConformanceExecutionClass::Scheduled,
        "audio bytes plus optional text prompt returning text and/or audio output",
    ),
    CapabilityConformanceCase::new(
        "speech_to_speech.audio_stream",
        CapabilityKind::SpeechToSpeech,
        ConformanceExecutionClass::Streaming,
        "audio request to streaming audio response with cancellation",
    ),
    CapabilityConformanceCase::new(
        "diarization.short_multispeaker",
        CapabilityKind::Diarization,
        ConformanceExecutionClass::Pipeline,
        "short multi-speaker fixture with speaker labels and ASR attribution",
    ),
    CapabilityConformanceCase::new(
        "forced_alignment.words",
        CapabilityKind::ForcedAlignment,
        ConformanceExecutionClass::Batch,
        "transcript plus audio fixture producing ordered word timestamps",
    ),
    CapabilityConformanceCase::new(
        "voice.vad.speech_events",
        CapabilityKind::Vad,
        ConformanceExecutionClass::Realtime,
        "pcm frames producing speech start and speech end events",
    ),
    CapabilityConformanceCase::new(
        "voice.endpointing.turn_boundary",
        CapabilityKind::Endpointing,
        ConformanceExecutionClass::Realtime,
        "speech activity stream producing stable turn boundary decisions",
    ),
    CapabilityConformanceCase::new(
        "tokenizer.model_artifact.round_trip",
        CapabilityKind::Tokenizer,
        ConformanceExecutionClass::Artifact,
        "tokenizer artifact load with deterministic encode/decode smoke check",
    ),
];

pub fn capability_conformance_cases() -> &'static [CapabilityConformanceCase] {
    CAPABILITY_CONFORMANCE_CASES
}

pub fn required_conformance_capabilities() -> &'static [CapabilityKind] {
    REQUIRED_CONFORMANCE_CAPABILITIES
}

/// Retained-state ownership of a binding: managed when its capability table
/// row requires retained state, which load-time state publication enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum RetainedStateExpectation {
    Managed,
    Stateless,
}

/// Invocation workspace requirement, independent of retained state: required
/// when the binding's row requires invocation state, which load-time state
/// publication enforces as a non-empty workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum WorkspaceExpectation {
    DeclaredMayBeEmpty,
    InvocationStateRequired,
}

/// Frozen product expectation for an exact model/capability/backend cell.
/// `RequiredWhenCompiled` still permits normal runtime hardware admission to
/// reject an unavailable or unsupported physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum BackendConformanceExpectation {
    RequiredWhenCompiled,
    Unsupported,
    CatalogDisabled,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct BackendConformanceExpectations {
    pub(crate) cpu: BackendConformanceExpectation,
    pub(crate) metal: BackendConformanceExpectation,
    pub(crate) cuda: BackendConformanceExpectation,
}

/// One entry per runtime-registry model/capability binding. The binding set is
/// derived from `ModelVariant::all()` and the built-in adapter registry so a
/// catalog addition cannot silently escape conformance coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct CatalogStateConformanceCase {
    pub(crate) variant: ModelVariant,
    pub(crate) capability: CapabilityKind,
    pub(crate) catalog_enabled: bool,
    pub(crate) retained_state: RetainedStateExpectation,
    pub(crate) workspace: WorkspaceExpectation,
    pub(crate) backends: BackendConformanceExpectations,
    pub(crate) fixture_id: &'static str,
}

/// Build the catalog/state/backend manifest from the capability table without
/// affecting model loading, routing, or execution.
#[cfg(test)]
pub(crate) fn catalog_state_conformance_manifest() -> Vec<CatalogStateConformanceCase> {
    let registry = RuntimeAdapterRegistry::built_in();
    let mut manifest = Vec::with_capacity(EXPECTED_CATALOG_CAPABILITY_BINDING_COUNT);

    for variant in ModelVariant::all().iter().copied() {
        for metadata in registry.capabilities_for(variant) {
            let capability = metadata.capability;
            let row = capability_row(capability, variant)
                .expect("registry bindings come from the capability table");
            let state = row.state_requirement;

            manifest.push(CatalogStateConformanceCase {
                variant,
                capability,
                catalog_enabled: variant.is_enabled(),
                retained_state: if state.requires_retained() {
                    RetainedStateExpectation::Managed
                } else {
                    RetainedStateExpectation::Stateless
                },
                workspace: if state.requires_invocation() {
                    WorkspaceExpectation::InvocationStateRequired
                } else {
                    WorkspaceExpectation::DeclaredMayBeEmpty
                },
                backends: backend_expectations(variant, row.backends),
                fixture_id: fixture_id_for(capability),
            });
        }
    }

    manifest
}

#[cfg(test)]
fn fixture_id_for(capability: CapabilityKind) -> &'static str {
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

#[cfg(test)]
fn backend_expectations(
    variant: ModelVariant,
    row_backends: &[BackendKind],
) -> BackendConformanceExpectations {
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

    let served = |backend: BackendKind| {
        if row_backends.contains(&backend) {
            Expectation::RequiredWhenCompiled
        } else {
            Expectation::Unsupported
        }
    };
    let cuda = match variant.cuda_support_level() {
        CudaSupportLevel::NativeCuda | CudaSupportLevel::CandleCudaGeneric => {
            served(BackendKind::Cuda)
        }
        CudaSupportLevel::CpuOnly => Expectation::Unsupported,
        CudaSupportLevel::Disabled => Expectation::CatalogDisabled,
        CudaSupportLevel::Unknown => {
            panic!("enabled catalog variant {variant} has unknown CUDA support")
        }
    };

    BackendConformanceExpectations {
        cpu: served(BackendKind::Cpu),
        metal: served(BackendKind::Metal),
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
        capability: CapabilityKind,
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
                    .map(move |metadata| (variant, metadata.capability))
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
            54
        );
        assert_eq!(
            manifest
                .iter()
                .filter(|case| case.retained_state == Stateless)
                .count(),
            13
        );

        assert_eq!(
            manifest_case(&manifest, ModelVariant::Qwen3827BFp8, CapabilityKind::Chat,)
                .retained_state,
            Managed
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen3Tts12Hz06BBase,
                CapabilityKind::Tts,
            )
            .workspace,
            InvocationStateRequired
        );
        assert_eq!(
            manifest_case(&manifest, ModelVariant::Qwen3827BFp8, CapabilityKind::Chat,).workspace,
            DeclaredMayBeEmpty
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Nemotron35AsrStreaming06B,
                CapabilityKind::RealtimeAsr,
            )
            .retained_state,
            Managed
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Nemotron35AsrStreaming06B,
                CapabilityKind::Asr,
            )
            .retained_state,
            Stateless
        );
        let voxtral_realtime = manifest_case(
            &manifest,
            ModelVariant::VoxtralMini4BRealtime2602,
            CapabilityKind::RealtimeAsr,
        );
        assert_eq!(voxtral_realtime.retained_state, Managed);
        assert_eq!(voxtral_realtime.workspace, InvocationStateRequired);
        assert_eq!(voxtral_realtime.backends.cpu, Backend::RequiredWhenCompiled);
        assert_eq!(
            voxtral_realtime.backends.metal,
            Backend::RequiredWhenCompiled
        );
        assert_eq!(
            voxtral_realtime.backends.cuda,
            Backend::RequiredWhenCompiled
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::Qwen3TtsTokenizer12Hz,
                CapabilityKind::Tokenizer,
            )
            .workspace,
            DeclaredMayBeEmpty
        );
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::GraniteSpeech412BPlus,
                CapabilityKind::SpeakerAttributedAsr,
            )
            .retained_state,
            Stateless
        );
        // The recurrent and sequence ASR/TTS adapters publish retained physical
        // state; load-time publication rejects any other declaration. The
        // pre-table manifest called these stateless.
        for (variant, capability) in [
            (ModelVariant::FishAudioS2Pro, CapabilityKind::Tts),
            (ModelVariant::FishAudioS2Pro, CapabilityKind::StreamingTts),
            (ModelVariant::WhisperLargeV3Turbo, CapabilityKind::Asr),
            (ModelVariant::VibeVoiceAsr, CapabilityKind::Asr),
            (ModelVariant::ParakeetTdt06BV3, CapabilityKind::Asr),
            (ModelVariant::GraniteSpeech412B, CapabilityKind::Asr),
            (ModelVariant::Lfm25Audio15BGguf, CapabilityKind::Asr),
            (ModelVariant::Lfm25Audio15BGguf, CapabilityKind::Tts),
        ] {
            let case = manifest_case(&manifest, variant, capability);
            assert_eq!(case.retained_state, Managed, "{variant} {capability:?}");
            assert_eq!(case.workspace, InvocationStateRequired);
        }
        // Voxtral's offline ASR keeps only invocation state; its retained
        // state belongs to the realtime capability.
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::VoxtralMini4BRealtime2602,
                CapabilityKind::Asr,
            )
            .retained_state,
            Stateless
        );
        for capability in [CapabilityKind::Asr, CapabilityKind::SpeakerAttributedAsr] {
            assert_eq!(
                manifest_case(&manifest, ModelVariant::GraniteSpeech412BPlus, capability,)
                    .workspace,
                InvocationStateRequired
            );
        }
        // The base checkpoint serves plain ASR only; speaker turns are plus's.
        assert_eq!(
            manifest_case(
                &manifest,
                ModelVariant::GraniteSpeech412B,
                CapabilityKind::Asr,
            )
            .workspace,
            InvocationStateRequired
        );
        assert!(!manifest
            .iter()
            .any(|case| case.variant == ModelVariant::GraniteSpeech412B
                && case.capability == CapabilityKind::SpeakerAttributedAsr));
        // Granite Speech 5.0 runs one atomic invocation on leased pages and
        // retains nothing between requests; it serves plain ASR only.
        // Its GGUF release is the same model with the same state.
        for variant in [
            ModelVariant::GraniteSpeech502B,
            ModelVariant::GraniteSpeech502BGguf,
        ] {
            let granite_5 = manifest_case(&manifest, variant, CapabilityKind::Asr);
            assert_eq!(granite_5.retained_state, Stateless);
            assert_eq!(granite_5.workspace, InvocationStateRequired);
            assert_eq!(
                manifest
                    .iter()
                    .filter(|case| case.variant == variant)
                    .count(),
                1
            );
        }
    }
}

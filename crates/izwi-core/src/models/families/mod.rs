//! Model-family registration metadata.
//!
//! This module is intentionally metadata-only. It gives contributors a stable
//! index of native model-family ownership before implementation files are
//! reorganized more deeply.

use crate::catalog::{ModelFamily, ModelVariant};
use crate::runtime::ConformanceCapability;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyRegistration {
    pub family: ModelFamily,
    pub module_path: &'static str,
    pub variants: &'static [ModelVariant],
    pub capabilities: &'static [ConformanceCapability],
    pub fixture_ids: &'static [&'static str],
}

impl FamilyRegistration {
    pub fn contains_variant(&self, variant: ModelVariant) -> bool {
        self.variants.contains(&variant)
    }
}

const QWEN3_TTS_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen3Tts12Hz06BBase,
    ModelVariant::Qwen3Tts12Hz06BBase4Bit,
    ModelVariant::Qwen3Tts12Hz06BBase8Bit,
    ModelVariant::Qwen3Tts12Hz06BBaseBf16,
    ModelVariant::Qwen3Tts12Hz06BCustomVoice,
    ModelVariant::Qwen3Tts12Hz06BCustomVoice4Bit,
    ModelVariant::Qwen3Tts12Hz06BCustomVoice8Bit,
    ModelVariant::Qwen3Tts12Hz06BCustomVoiceBf16,
    ModelVariant::Qwen3Tts12Hz17BBase,
    ModelVariant::Qwen3Tts12Hz17BBase4Bit,
    ModelVariant::Qwen3Tts12Hz17BCustomVoice,
    ModelVariant::Qwen3Tts12Hz17BCustomVoice4Bit,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesign4Bit,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesign8Bit,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesignBf16,
];
const KOKORO_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::Kokoro82M];
const PARAKEET_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::ParakeetTdt06BV3];
const WHISPER_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::WhisperLargeV3Turbo];
const QWEN3_ASR_VARIANTS: &[ModelVariant] =
    &[ModelVariant::Qwen3Asr06BGguf, ModelVariant::Qwen3Asr17BGguf];
const SORTFORMER_DIARIZATION_VARIANTS: &[ModelVariant] =
    &[ModelVariant::DiarStreamingSortformer4SpkV21];
const QWEN3_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen306B,
    ModelVariant::Qwen306B4Bit,
    ModelVariant::Qwen306BGguf,
    ModelVariant::Qwen317B,
    ModelVariant::Qwen317B4Bit,
    ModelVariant::Qwen317BGguf,
    ModelVariant::Qwen34BGguf,
    ModelVariant::Qwen38BGguf,
    ModelVariant::Qwen314BGguf,
];
const QWEN35_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen3508BGguf,
    ModelVariant::Qwen352BGguf,
    ModelVariant::Qwen354BGguf,
    ModelVariant::Qwen359BGguf,
];
const LFM2_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Lfm2512BInstructGguf,
    ModelVariant::Lfm2512BThinkingGguf,
];
const LFM25_AUDIO_VARIANTS: &[ModelVariant] = &[ModelVariant::Lfm25Audio15BGguf];
const GEMMA3_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Gemma31BIt, ModelVariant::Gemma34BIt];
const QWEN3_FORCED_ALIGNER_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen3ForcedAligner06B,
    ModelVariant::Qwen3ForcedAligner06B4Bit,
];
const VOXTRAL_VARIANTS: &[ModelVariant] = &[ModelVariant::VoxtralMini4BRealtime2602];
const TOKENIZER_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen3TtsTokenizer12Hz];

const TTS_STREAMING_CAPABILITIES: &[ConformanceCapability] = &[
    ConformanceCapability::Tts,
    ConformanceCapability::StreamingTts,
];
const TTS_STREAMING_FIXTURES: &[&str] = &["tts.short_text.binary", "streaming_tts.short_text"];
const ASR_CAPABILITIES: &[ConformanceCapability] = &[ConformanceCapability::Asr];
const ASR_FIXTURES: &[&str] = &["asr.short_wav.transcript"];
const DIARIZATION_CAPABILITIES: &[ConformanceCapability] = &[ConformanceCapability::Diarization];
const DIARIZATION_FIXTURES: &[&str] = &["diarization.short_multispeaker"];
const CHAT_CAPABILITIES: &[ConformanceCapability] = &[ConformanceCapability::Chat];
const CHAT_FIXTURES: &[&str] = &["chat.single_prompt.streaming"];
const LFM25_AUDIO_CAPABILITIES: &[ConformanceCapability] = &[
    ConformanceCapability::Tts,
    ConformanceCapability::Asr,
    ConformanceCapability::AudioChat,
    ConformanceCapability::SpeechToSpeech,
];
const LFM25_AUDIO_FIXTURES: &[&str] = &[
    "tts.short_text.binary",
    "asr.short_wav.transcript",
    "audio_chat.audio_prompt.response",
    "speech_to_speech.audio_stream",
];
const FORCED_ALIGNMENT_CAPABILITIES: &[ConformanceCapability] =
    &[ConformanceCapability::ForcedAlignment];
const FORCED_ALIGNMENT_FIXTURES: &[&str] = &["forced_alignment.words"];
const VOXTRAL_CAPABILITIES: &[ConformanceCapability] = &[ConformanceCapability::Asr];
const VOXTRAL_FIXTURES: &[&str] = &["asr.short_wav.transcript"];
const TOKENIZER_CAPABILITIES: &[ConformanceCapability] = &[ConformanceCapability::Tokenizer];
const TOKENIZER_FIXTURES: &[&str] = &["tokenizer.model_artifact.round_trip"];

pub const MODEL_FAMILY_REGISTRATIONS: &[FamilyRegistration] = &[
    FamilyRegistration {
        family: ModelFamily::Qwen3Tts,
        module_path: "crate::models::architectures::qwen3::tts",
        variants: QWEN3_TTS_VARIANTS,
        capabilities: TTS_STREAMING_CAPABILITIES,
        fixture_ids: TTS_STREAMING_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::KokoroTts,
        module_path: "crate::models::architectures::kokoro",
        variants: KOKORO_TTS_VARIANTS,
        capabilities: TTS_STREAMING_CAPABILITIES,
        fixture_ids: TTS_STREAMING_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::ParakeetAsr,
        module_path: "crate::models::architectures::parakeet::asr",
        variants: PARAKEET_ASR_VARIANTS,
        capabilities: ASR_CAPABILITIES,
        fixture_ids: ASR_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::WhisperAsr,
        module_path: "crate::models::architectures::whisper::asr",
        variants: WHISPER_ASR_VARIANTS,
        capabilities: ASR_CAPABILITIES,
        fixture_ids: ASR_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3Asr,
        module_path: "crate::models::architectures::qwen3::asr",
        variants: QWEN3_ASR_VARIANTS,
        capabilities: ASR_CAPABILITIES,
        fixture_ids: ASR_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::SortformerDiarization,
        module_path: "crate::models::architectures::sortformer::diarization",
        variants: SORTFORMER_DIARIZATION_VARIANTS,
        capabilities: DIARIZATION_CAPABILITIES,
        fixture_ids: DIARIZATION_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3Chat,
        module_path: "crate::models::architectures::qwen3::chat",
        variants: QWEN3_CHAT_VARIANTS,
        capabilities: CHAT_CAPABILITIES,
        fixture_ids: CHAT_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen35Chat,
        module_path: "crate::models::architectures::qwen35::chat",
        variants: QWEN35_CHAT_VARIANTS,
        capabilities: CHAT_CAPABILITIES,
        fixture_ids: CHAT_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm2Chat,
        module_path: "crate::models::architectures::lfm2::chat",
        variants: LFM2_CHAT_VARIANTS,
        capabilities: CHAT_CAPABILITIES,
        fixture_ids: CHAT_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm25Audio,
        module_path: "crate::models::architectures::lfm25_audio",
        variants: LFM25_AUDIO_VARIANTS,
        capabilities: LFM25_AUDIO_CAPABILITIES,
        fixture_ids: LFM25_AUDIO_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Gemma3Chat,
        module_path: "crate::models::architectures::gemma3::chat",
        variants: GEMMA3_CHAT_VARIANTS,
        capabilities: CHAT_CAPABILITIES,
        fixture_ids: CHAT_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3ForcedAligner,
        module_path: "crate::models::architectures::qwen3::asr",
        variants: QWEN3_FORCED_ALIGNER_VARIANTS,
        capabilities: FORCED_ALIGNMENT_CAPABILITIES,
        fixture_ids: FORCED_ALIGNMENT_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Voxtral,
        module_path: "crate::models::architectures::voxtral",
        variants: VOXTRAL_VARIANTS,
        capabilities: VOXTRAL_CAPABILITIES,
        fixture_ids: VOXTRAL_FIXTURES,
    },
    FamilyRegistration {
        family: ModelFamily::Tokenizer,
        module_path: "crate::models::architectures::qwen3::tts::speech_tokenizer",
        variants: TOKENIZER_VARIANTS,
        capabilities: TOKENIZER_CAPABILITIES,
        fixture_ids: TOKENIZER_FIXTURES,
    },
];

pub fn model_family_registrations() -> &'static [FamilyRegistration] {
    MODEL_FAMILY_REGISTRATIONS
}

pub fn registration_for_variant(variant: ModelVariant) -> Option<&'static FamilyRegistration> {
    model_family_registrations()
        .iter()
        .find(|registration| registration.contains_variant(variant))
}

pub fn registrations_for_capability(
    capability: ConformanceCapability,
) -> impl Iterator<Item = &'static FamilyRegistration> {
    model_family_registrations()
        .iter()
        .filter(move |registration| registration.capabilities.contains(&capability))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::capability_conformance_cases;
    use std::collections::BTreeSet;

    fn expected_capabilities(variant: ModelVariant) -> BTreeSet<ConformanceCapability> {
        let mut expected = BTreeSet::new();

        if variant.speech_capabilities().is_some() {
            expected.insert(ConformanceCapability::Tts);
        }
        if variant
            .speech_capabilities()
            .is_some_and(|capabilities| capabilities.supports_streaming)
        {
            expected.insert(ConformanceCapability::StreamingTts);
        }
        if variant.is_asr() || variant.is_voxtral() || variant.is_audio_chat() {
            expected.insert(ConformanceCapability::Asr);
        }
        if variant.is_chat() {
            expected.insert(ConformanceCapability::Chat);
        }
        if variant.is_audio_chat() {
            expected.insert(ConformanceCapability::AudioChat);
            expected.insert(ConformanceCapability::SpeechToSpeech);
        }
        if variant.is_diarization() {
            expected.insert(ConformanceCapability::Diarization);
        }
        if variant.is_forced_aligner() {
            expected.insert(ConformanceCapability::ForcedAlignment);
        }
        if variant.is_tokenizer() {
            expected.insert(ConformanceCapability::Tokenizer);
        }

        expected
    }

    #[test]
    fn every_model_variant_has_exactly_one_family_registration() {
        let mut covered = Vec::new();

        for registration in model_family_registrations() {
            assert!(!registration.variants.is_empty());
            assert!(!registration.module_path.is_empty());
            assert!(!registration.capabilities.is_empty());
            assert!(!registration.fixture_ids.is_empty());

            for variant in registration.variants {
                assert_eq!(
                    variant.family(),
                    registration.family,
                    "variant {variant:?} is registered under the wrong family"
                );
                assert!(
                    !covered.contains(variant),
                    "variant {variant:?} is registered more than once"
                );
                covered.push(*variant);
            }
        }

        for variant in ModelVariant::all() {
            assert!(
                covered.contains(variant),
                "missing family registration for {variant:?}"
            );
            assert!(
                registration_for_variant(*variant).is_some(),
                "registration_for_variant failed for {variant:?}"
            );
        }
        assert_eq!(covered.len(), ModelVariant::all().len());
    }

    #[test]
    fn family_capabilities_match_catalog_variant_helpers() {
        for registration in model_family_registrations() {
            let registered = registration
                .capabilities
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();

            for variant in registration.variants {
                assert_eq!(
                    registered,
                    expected_capabilities(*variant),
                    "capability registration mismatch for {variant:?}"
                );
            }
        }
    }

    #[test]
    fn fixture_ids_are_backed_by_conformance_cases() {
        let case_ids = capability_conformance_cases()
            .iter()
            .map(|case| case.id)
            .collect::<BTreeSet<_>>();

        for registration in model_family_registrations() {
            for fixture_id in registration.fixture_ids {
                assert!(
                    case_ids.contains(fixture_id),
                    "unknown conformance fixture id {fixture_id}"
                );
            }
        }
    }

    #[test]
    fn capability_lookup_finds_registered_families() {
        let chat_families = registrations_for_capability(ConformanceCapability::Chat)
            .map(|registration| registration.family)
            .collect::<Vec<_>>();

        assert!(chat_families.contains(&ModelFamily::Qwen3Chat));
        assert!(chat_families.contains(&ModelFamily::Qwen35Chat));
        assert!(chat_families.contains(&ModelFamily::Lfm2Chat));
        assert!(chat_families.contains(&ModelFamily::Gemma3Chat));
    }
}

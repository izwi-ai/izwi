//! Model-family registration metadata.
//!
//! This module is intentionally metadata-only. It gives contributors a stable
//! index of native model-family ownership before implementation files are
//! reorganized more deeply.

use crate::catalog::{ModelFamily, ModelVariant};
use crate::runtime::{capability_conformance_cases, capability_rows, CapabilityKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyRegistration {
    pub family: ModelFamily,
    pub module_path: &'static str,
    pub variants: &'static [ModelVariant],
}

impl FamilyRegistration {
    pub fn contains_variant(&self, variant: ModelVariant) -> bool {
        self.variants.contains(&variant)
    }

    /// Capabilities any of the family's variants serve, read from the
    /// runtime's capability table.
    pub fn capabilities(&self) -> Vec<CapabilityKind> {
        let mut capabilities = self
            .variants
            .iter()
            .flat_map(|variant| capability_rows(*variant).map(|row| row.capability))
            .collect::<Vec<_>>();
        capabilities.sort_unstable();
        capabilities.dedup();
        capabilities
    }

    /// The conformance fixtures of those capabilities.
    pub fn fixture_ids(&self) -> Vec<&'static str> {
        let capabilities = self.capabilities();
        capability_conformance_cases()
            .iter()
            .filter(|case| capabilities.contains(&case.capability))
            .map(|case| case.id)
            .collect()
    }
}

const QWEN3_TTS_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen3Tts12Hz06BBase,
    ModelVariant::Qwen3Tts12Hz06BBase8Bit,
    ModelVariant::Qwen3Tts12Hz06BBaseBf16,
    ModelVariant::Qwen3Tts12Hz06BCustomVoice,
    ModelVariant::Qwen3Tts12Hz06BCustomVoice8Bit,
    ModelVariant::Qwen3Tts12Hz06BCustomVoiceBf16,
    ModelVariant::Qwen3Tts12Hz17BBase,
    ModelVariant::Qwen3Tts12Hz17BCustomVoice,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesign8Bit,
    ModelVariant::Qwen3Tts12Hz17BVoiceDesignBf16,
];
const KOKORO_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::Kokoro82M];
const VOXTRAL_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::Voxtral4BTts2603];
const CHATTERBOX_TTS_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Chatterbox500M,
    ModelVariant::ChatterboxMultilingual500M,
];
const MAYA1_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::Maya13B, ModelVariant::Maya13BGguf];
const MAYA2_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::Maya2Native25B];
const VIBEVOICE_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::VibeVoice15BTts];
const FISH_S2_TTS_VARIANTS: &[ModelVariant] = &[ModelVariant::FishAudioS2Pro];
const PARAKEET_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::ParakeetTdt06BV3];
const WHISPER_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::WhisperLargeV3Turbo];
const QWEN3_ASR_VARIANTS: &[ModelVariant] =
    &[ModelVariant::Qwen3Asr06BGguf, ModelVariant::Qwen3Asr17BGguf];
const VIBEVOICE_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::VibeVoiceAsr];
const NEMOTRON_ASR_VARIANTS: &[ModelVariant] = &[ModelVariant::Nemotron35AsrStreaming06B];
const GRANITE_SPEECH_ASR_VARIANTS: &[ModelVariant] = &[
    ModelVariant::GraniteSpeech412B,
    ModelVariant::GraniteSpeech412BPlus,
];
const GRANITE_SPEECH5_ASR_VARIANTS: &[ModelVariant] = &[
    ModelVariant::GraniteSpeech502B,
    ModelVariant::GraniteSpeech502BGguf,
];
const NEMOTRON_TRANSCRIBE_ASR_VARIANTS: &[ModelVariant] =
    &[ModelVariant::Nemotron35Transcribe30BA4BFp8];
const SORTFORMER_DIARIZATION_VARIANTS: &[ModelVariant] =
    &[ModelVariant::DiarStreamingSortformer4SpkV21, ModelVariant::Nemotron3Diarization];
const QWEN3_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen306B,
    ModelVariant::Qwen306B4Bit,
    ModelVariant::Qwen317B,
    ModelVariant::Qwen317B4Bit,
    ModelVariant::Qwen314BGguf,
];
const QWEN3_MOE_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen3Moe30bA3bGguf];
const QWEN35_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Qwen3508BGguf,
    ModelVariant::Qwen352BGguf,
    ModelVariant::Qwen354BGguf,
    ModelVariant::Qwen359BGguf,
];
const QWEN38_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen3827BFp8];
const QWEN35_MOE_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen36Moe35BA3BFp8];
const LFM2_CHAT_VARIANTS: &[ModelVariant] = &[
    ModelVariant::Lfm2512BInstructGguf,
    ModelVariant::Lfm2512BThinkingGguf,
];
const LFM3_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Lfm317BGguf, ModelVariant::Lfm317BQ8Gguf];
const LFM25_AUDIO_VARIANTS: &[ModelVariant] = &[ModelVariant::Lfm25Audio15BGguf];
const LFM3_AUDIO_VARIANTS: &[ModelVariant] =
    &[ModelVariant::Lfm3Audio2BGguf, ModelVariant::Lfm3Audio2BQ8Gguf];
const GEMMA3_CHAT_VARIANTS: &[ModelVariant] = &[ModelVariant::Gemma34BIt];
const QWEN3_FORCED_ALIGNER_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen3ForcedAligner06B];
const VOXTRAL_VARIANTS: &[ModelVariant] = &[ModelVariant::VoxtralMini4BRealtime2602];
const TOKENIZER_VARIANTS: &[ModelVariant] = &[ModelVariant::Qwen3TtsTokenizer12Hz];

pub const MODEL_FAMILY_REGISTRATIONS: &[FamilyRegistration] = &[
    FamilyRegistration {
        family: ModelFamily::Qwen3Tts,
        module_path: "crate::models::architectures::qwen3::tts",
        variants: QWEN3_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::KokoroTts,
        module_path: "crate::models::architectures::kokoro",
        variants: KOKORO_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::VoxtralTts,
        module_path: "crate::models::architectures::voxtral::tts",
        variants: VOXTRAL_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::VibeVoiceTts,
        module_path: "crate::models::architectures::vibevoice::tts",
        variants: VIBEVOICE_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::FishS2Tts,
        module_path: "crate::models::architectures::fish_s2",
        variants: FISH_S2_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::ChatterboxTts,
        module_path: "crate::models::architectures::chatterbox",
        variants: CHATTERBOX_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Maya1Tts,
        module_path: "crate::models::architectures::maya1",
        variants: MAYA1_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Maya2Tts,
        module_path: "crate::models::architectures::maya2",
        variants: MAYA2_TTS_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::ParakeetAsr,
        module_path: "crate::models::architectures::parakeet::asr",
        variants: PARAKEET_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::WhisperAsr,
        module_path: "crate::models::architectures::whisper::asr",
        variants: WHISPER_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3Asr,
        module_path: "crate::models::architectures::qwen3::asr",
        variants: QWEN3_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::VibeVoiceAsr,
        module_path: "crate::models::architectures::vibevoice::asr",
        variants: VIBEVOICE_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::NemotronAsr,
        module_path: "crate::models::architectures::nemotron::asr",
        variants: NEMOTRON_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::GraniteSpeechAsr,
        module_path: "crate::models::architectures::granite_speech::asr",
        variants: GRANITE_SPEECH_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::GraniteSpeech5Asr,
        module_path: "crate::models::architectures::granite_speech5::asr",
        variants: GRANITE_SPEECH5_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::NemotronTranscribeAsr,
        module_path: "crate::models::architectures::nemotron_transcribe",
        variants: NEMOTRON_TRANSCRIBE_ASR_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::SortformerDiarization,
        module_path: "crate::models::architectures::sortformer::diarization",
        variants: SORTFORMER_DIARIZATION_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3Chat,
        module_path: "crate::models::architectures::qwen3::chat",
        variants: QWEN3_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3MoeChat,
        module_path: "crate::models::architectures::qwen3::chat",
        variants: QWEN3_MOE_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen35Chat,
        module_path: "crate::models::architectures::qwen35::chat",
        variants: QWEN35_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen38Chat,
        module_path: "crate::models::architectures::qwen38::chat",
        variants: QWEN38_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen35MoeChat,
        module_path: "crate::models::architectures::qwen36moe::chat",
        variants: QWEN35_MOE_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm2Chat,
        module_path: "crate::models::architectures::lfm2::chat",
        variants: LFM2_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm3Chat,
        module_path: "crate::models::architectures::lfm2::chat",
        variants: LFM3_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm25Audio,
        module_path: "crate::models::architectures::lfm25_audio",
        variants: LFM25_AUDIO_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Lfm3Audio,
        module_path: "crate::models::architectures::lfm3_audio",
        variants: LFM3_AUDIO_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Gemma3Chat,
        module_path: "crate::models::architectures::gemma3::chat",
        variants: GEMMA3_CHAT_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Qwen3ForcedAligner,
        module_path: "crate::models::architectures::qwen3::asr",
        variants: QWEN3_FORCED_ALIGNER_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Voxtral,
        module_path: "crate::models::architectures::voxtral",
        variants: VOXTRAL_VARIANTS,
    },
    FamilyRegistration {
        family: ModelFamily::Tokenizer,
        module_path: "crate::models::architectures::qwen3::tts::speech_tokenizer",
        variants: TOKENIZER_VARIANTS,
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
    capability: CapabilityKind,
) -> impl Iterator<Item = &'static FamilyRegistration> {
    model_family_registrations()
        .iter()
        .filter(move |registration| registration.capabilities().contains(&capability))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_model_variant_has_exactly_one_family_registration() {
        let mut covered = Vec::new();

        for registration in model_family_registrations() {
            assert!(!registration.variants.is_empty());
            assert!(!registration.module_path.is_empty());
            assert!(!registration.capabilities().is_empty());
            assert!(!registration.fixture_ids().is_empty());

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
    fn fixture_ids_are_backed_by_conformance_cases() {
        let case_ids = capability_conformance_cases()
            .iter()
            .map(|case| case.id)
            .collect::<BTreeSet<_>>();

        for registration in model_family_registrations() {
            for fixture_id in registration.fixture_ids() {
                assert!(
                    case_ids.contains(fixture_id),
                    "unknown conformance fixture id {fixture_id}"
                );
            }
        }
    }

    #[test]
    fn capability_lookup_finds_registered_families() {
        let chat_families = registrations_for_capability(CapabilityKind::Chat)
            .map(|registration| registration.family)
            .collect::<Vec<_>>();

        assert!(chat_families.contains(&ModelFamily::Qwen3Chat));
        assert!(chat_families.contains(&ModelFamily::Qwen35Chat));
        assert!(chat_families.contains(&ModelFamily::Lfm2Chat));
        assert!(chat_families.contains(&ModelFamily::Gemma3Chat));

        // Read from the capability table, so the realtime families and
        // Granite's speaker turns are listed (the old hand-written lists
        // omitted them).
        let realtime_families = registrations_for_capability(CapabilityKind::RealtimeAsr)
            .map(|registration| registration.family)
            .collect::<Vec<_>>();
        assert_eq!(
            realtime_families,
            [
                ModelFamily::NemotronAsr,
                ModelFamily::Lfm3Audio,
                ModelFamily::Voxtral
            ]
        );
        assert!(
            registrations_for_capability(CapabilityKind::SpeakerAttributedAsr)
                .any(|registration| registration.family == ModelFamily::GraniteSpeechAsr)
        );
    }
}

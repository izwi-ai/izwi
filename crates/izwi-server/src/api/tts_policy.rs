use izwi_core::catalog::ModelFamily;
use izwi_core::runtime_models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
use izwi_core::runtime_models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use izwi_core::ModelVariant;

pub(crate) fn resolve_tts_output_frames(
    variant: ModelVariant,
    text: &str,
    requested_frames: Option<usize>,
) -> Option<usize> {
    let model_max_frames = variant.tts_max_output_frames_hint()?;
    match requested_frames {
        Some(value) if value > 0 => Some(value.clamp(1, model_max_frames)),
        Some(0) | None if variant == ModelVariant::Voxtral4BTts2603 => {
            Some(voxtral_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::VibeVoice15BTts => {
            Some(vibevoice_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::FishAudioS2Pro => {
            Some(fish_s2_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::Qwen3Tts => {
            Some(qwen_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None => Some(model_max_frames),
        Some(_) => unreachable!("positive requested frames handled above"),
    }
}

pub(crate) fn qwen_tts_auto_max_frames_for_text(text: &str) -> usize {
    const MIN_AUDIO_SECS: f32 = 4.0;
    const MAX_AUDIO_SECS_PER_REQUEST: f32 = 120.0;
    const WORDS_PER_SECOND: f32 = 2.5;
    const CHARS_PER_WORD: usize = 5;
    const END_PADDING_SECS: f32 = 2.0;

    let word_count = text.split_whitespace().count();
    let char_word_equivalent = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .count()
        .div_ceil(CHARS_PER_WORD);
    let estimated_words = word_count.max(char_word_equivalent).max(1);
    let estimated_secs = ((estimated_words as f32) / WORDS_PER_SECOND + END_PADDING_SECS)
        .clamp(MIN_AUDIO_SECS, MAX_AUDIO_SECS_PER_REQUEST);
    let frames = (estimated_secs * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ).ceil() as usize;
    frames.clamp(
        (MIN_AUDIO_SECS * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ) as usize,
        (MAX_AUDIO_SECS_PER_REQUEST * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ) as usize,
    )
}

fn fish_s2_tts_auto_max_frames_for_text(text: &str) -> usize {
    // Whitespace alone treats an entire CJK paragraph as one word. Include a
    // character estimate and count CJK syllabic characters separately; leave
    // time for pauses and EOS rather than truncating at the speaking estimate.
    let non_space = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let cjk = text
        .chars()
        .filter(|ch| {
            matches!(*ch as u32,
                0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff |
                0xac00..=0xd7af | 0x20000..=0x3134f
            )
        })
        .count();
    let estimated_words = text
        .split_whitespace()
        .count()
        .max(non_space.saturating_sub(cjk).div_ceil(5));
    let estimated_secs = (estimated_words as f32 / 2.6 + cjk as f32 / 4.0 + 2.0).clamp(4.0, 120.0);
    let frames = (estimated_secs * ModelVariant::FISH_S2_PRO_FRAME_RATE_HZ).ceil() as usize;
    frames.clamp(96, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_auto_budget_is_text_sized_and_bounded() {
        let short =
            resolve_tts_output_frames(ModelVariant::Qwen3Tts12Hz06BCustomVoice, "hello", None)
                .expect("Qwen frame hint");
        let longer = resolve_tts_output_frames(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            "This sentence is intentionally much longer than hello and needs more speech frames.",
            None,
        )
        .expect("Qwen frame hint");

        assert!(short < ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        assert!(longer > short);
        assert!(longer <= ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn fish_s2_auto_budget_handles_unspaced_multilingual_text() {
        let short = fish_s2_tts_auto_max_frames_for_text("Hello");
        for text in [
            "你好世界".repeat(30),
            "こんにちは世界".repeat(20),
            "안녕하세요".repeat(24),
            "longunspacedtext".repeat(20),
        ] {
            let budget = fish_s2_tts_auto_max_frames_for_text(&text);
            assert!(budget > short, "{text}");
            assert!(budget <= ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES);
        }
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(10)),
            Some(10)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(usize::MAX)),
            Some(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(0)),
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", None)
        );
    }

    #[test]
    fn explicit_budget_is_clamped_for_every_route() {
        assert_eq!(
            resolve_tts_output_frames(
                ModelVariant::Qwen3Tts12Hz06BCustomVoice,
                "hello",
                Some(usize::MAX),
            ),
            Some(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        );
    }
}

//! Output-frame budgets.
//!
//! The prompt, the reference frames and the generated frames share the slow
//! transformer's 2,048 positions, and a request that ends at its budget
//! without an end-of-speech token fails, so the budget of a request that
//! names none is a generous estimate of its text's speaking time.

use crate::catalog::ModelVariant;

/// Shortest and longest speaking estimates, in seconds. The upper bound is
/// the frame cap.
const MIN_SECONDS: f32 = 6.0;
const MIN_FRAMES: usize = 96;

/// Whether `ch` is a CJK ideograph, kana or Hangul syllable: scripts spoken
/// one character at a time and written without spaces.
fn is_cjk_syllabic(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff |
        0xac00..=0xd7af | 0x20000..=0x3134f
    )
}

/// The frame budget of a request that names none: words (or five non-space,
/// non-CJK characters to a word, whichever is more, so unspaced scripts such
/// as Thai are counted) at 1.8 a second, CJK characters at 2.5 a second, and
/// 4 s for pauses and the end-of-speech token, between 96 frames and the
/// model's cap.
pub fn auto_frame_budget(text: &str) -> usize {
    let non_space = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let cjk = text.chars().filter(|ch| is_cjk_syllabic(*ch)).count();
    let words = text
        .split_whitespace()
        .count()
        .max(non_space.saturating_sub(cjk).div_ceil(5));
    let max_seconds =
        ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES as f32 / ModelVariant::AUDIO8_FRAME_RATE_HZ;
    let seconds = (words as f32 / 1.8 + cjk as f32 / 2.5 + 4.0).clamp(MIN_SECONDS, max_seconds);
    ((seconds * ModelVariant::AUDIO8_FRAME_RATE_HZ).ceil() as usize)
        .clamp(MIN_FRAMES, ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_grows_with_text_and_stays_inside_the_cap() {
        let short = auto_frame_budget("Hello");
        assert!((MIN_FRAMES..200).contains(&short), "{short}");
        let sentence = auto_frame_budget(&"word ".repeat(30));
        assert!(sentence > short);
        for text in ["word ".repeat(2_000), "字".repeat(5_000), "ก".repeat(5_000)] {
            assert_eq!(
                auto_frame_budget(&text),
                ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES
            );
        }
    }

    #[test]
    fn unspaced_scripts_are_not_one_word() {
        assert!(auto_frame_budget(&"字".repeat(60)) > auto_frame_budget("字"));
        assert!(auto_frame_budget(&"ก".repeat(200)) > auto_frame_budget("ก"));
    }
}

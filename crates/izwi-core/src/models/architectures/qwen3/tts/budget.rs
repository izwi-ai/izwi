//! Qwen3-TTS request budgets: the automatic output-frame budget for a text,
//! and the retained talker context every request needs.

use crate::catalog::ModelVariant;

use super::MAX_VOICE_CLONE_REFERENCE_FRAMES;

/// Longest output one automatic Qwen3-TTS budget grants. Longer text is
/// split by long form.
pub const QWEN3_TTS_AUTO_MAX_AUDIO_SECS: f32 = 120.0;
const AUTO_MIN_AUDIO_SECS: f32 = 4.0;
const AUTO_WORDS_PER_SECOND: f32 = 2.5;
const AUTO_CHARS_PER_WORD: usize = 5;
const AUTO_END_PADDING_SECS: f32 = 2.0;

/// Talker positions one request's prompt may take before its first frame.
///
/// A clone prompt is about nine role and codec-prefix tokens plus the larger
/// of the reference codes (at most [`MAX_VOICE_CLONE_REFERENCE_FRAMES`]) and
/// the reference transcript with the target text. The target text that fills
/// the automatic budget is about 300 English words (about 400 tokens); a
/// VoiceDesign instruction adds its own tokens. 1,024 covers these.
pub(crate) const QWEN3_TTS_STATE_PROMPT_TOKENS: usize = 1_024;

const _: () = assert!(QWEN3_TTS_STATE_PROMPT_TOKENS > MAX_VOICE_CLONE_REFERENCE_FRAMES + 16);

/// The automatic output-frame budget for `text`: its estimated speaking time
/// plus padding, between 4 s and [`QWEN3_TTS_AUTO_MAX_AUDIO_SECS`].
pub fn qwen3_tts_auto_max_frames_for_text(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    let char_word_equivalent = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .count()
        .div_ceil(AUTO_CHARS_PER_WORD);
    let estimated_words = word_count.max(char_word_equivalent).max(1);
    let estimated_secs = ((estimated_words as f32) / AUTO_WORDS_PER_SECOND + AUTO_END_PADDING_SECS)
        .clamp(AUTO_MIN_AUDIO_SECS, QWEN3_TTS_AUTO_MAX_AUDIO_SECS);
    let frames = (estimated_secs * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ).ceil() as usize;
    frames.clamp(
        (AUTO_MIN_AUDIO_SECS * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ) as usize,
        qwen3_tts_auto_max_output_frames(),
    )
}

/// The largest automatic output-frame budget.
pub fn qwen3_tts_auto_max_output_frames() -> usize {
    (QWEN3_TTS_AUTO_MAX_AUDIO_SECS * ModelVariant::QWEN3_TTS_FRAME_RATE_HZ).ceil() as usize
}

/// The retained talker context every Qwen3-TTS request needs: one prompt
/// plus the largest automatic output budget. Load admission guarantees it
/// or refuses the load; live memory caps only growth above it.
pub(crate) fn qwen3_tts_state_floor_tokens() -> usize {
    QWEN3_TTS_STATE_PROMPT_TOKENS + qwen3_tts_auto_max_output_frames()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_floor_covers_a_full_clone_prompt_and_the_largest_auto_budget() {
        let longest = "word ".repeat(1_000);
        assert_eq!(
            qwen3_tts_auto_max_frames_for_text(&longest),
            qwen3_tts_auto_max_output_frames()
        );
        assert!(
            qwen3_tts_state_floor_tokens()
                >= MAX_VOICE_CLONE_REFERENCE_FRAMES + qwen3_tts_auto_max_output_frames()
        );
    }
}

//! Qwen3-TTS request budgets: the automatic output-frame budget for a text,
//! and the retained talker context every request needs.

use crate::catalog::ModelVariant;

use super::MAX_VOICE_CLONE_REFERENCE_FRAMES;

/// Longest output one automatic Qwen3-TTS budget grants. Longer text is
/// split by long form.
pub const QWEN3_TTS_AUTO_MAX_AUDIO_SECS: f32 = 120.0;
const AUTO_MIN_AUDIO_SECS: f32 = 4.0;
/// The expected speaking pace: Qwen3-TTS voices speak about 2.5 English
/// words per second (the `ryan` voice 2.35 live).
const AUTO_WORDS_PER_SECOND: f32 = 2.5;
/// Mandarin and Japanese speech runs about 4-5 characters per second.
const AUTO_CJK_CHARS_PER_SECOND: f32 = 4.5;
/// Sampled pacing varies widely around the expected pace: 16 sampled runs
/// of one 46-word paragraph ended between 222 and 328 frames. The budget
/// grants 1.5 times the expected speech, which covers the slowest of them,
/// so only a runaway that never reaches end of speech is cut (`length`).
const AUTO_PACE_MARGIN: f32 = 1.5;
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

/// The automatic output-frame budget for `text`: 1.5 times its speaking time
/// at the expected pace (per language) plus padding, between 4 s and
/// [`QWEN3_TTS_AUTO_MAX_AUDIO_SECS`].
pub fn qwen3_tts_auto_max_frames_for_text(text: &str) -> usize {
    let estimated_secs = (crate::models::shared::speech_duration::estimated_speech_secs(
        text,
        AUTO_WORDS_PER_SECOND,
        AUTO_CJK_CHARS_PER_SECOND,
    ) * AUTO_PACE_MARGIN
        + AUTO_END_PADDING_SECS)
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

/// Text positions one request's retained encoder memory may hold: the
/// prompt plus the trailing text the talker reads one per frame. Both fit
/// in the state floor, so the retained tensor rows are sized from it, not
/// from the talker's position embeddings (32,768: 1.34 GB of rows at 0.6B).
pub(crate) fn qwen3_tts_retained_memory_tokens(max_position_embeddings: usize) -> usize {
    qwen3_tts_state_floor_tokens().min(max_position_embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QTTS-21: the frame rate is the published codec's, not the "12Hz" in
    /// the variant name.
    #[test]
    fn frame_rate_matches_the_published_speech_tokenizer_config() {
        // Top-level keys of Qwen3-TTS-12Hz-*/speech_tokenizer/config.json.
        let published: serde_json::Value = serde_json::from_str(
            r#"{"model_type": "qwen3_tts_tokenizer_12hz", "input_sample_rate": 24000,
                "output_sample_rate": 24000, "decode_upsample_rate": 1920,
                "encode_downsample_rate": 1920}"#,
        )
        .unwrap();
        let rate = published["output_sample_rate"].as_f64().unwrap()
            / published["decode_upsample_rate"].as_f64().unwrap();
        assert_eq!(f64::from(ModelVariant::QWEN3_TTS_FRAME_RATE_HZ), rate);
        assert_eq!(rate, 12.5);
    }

    /// TPIPE-2: the live 46-word paragraph spoken by `ryan` (2.35 words/s)
    /// reached end of speech at 252 frames; the old 12 Hz, 2.5 words/s
    /// budget of 245 frames cut its last eight words.
    #[test]
    fn automatic_budget_leaves_room_for_a_slow_voice() {
        let paragraph = "Good morning. Today we are testing a speech system that reads \
            longer passages aloud, and we want to hear whether every sentence comes out \
            clearly without the voice drifting, rushing, or stopping before the final word \
            of the paragraph has been spoken.";
        let words = paragraph.split_whitespace().count();
        let needed = (words as f32 / 2.35 * 12.5).ceil() as usize;
        assert!(qwen3_tts_auto_max_frames_for_text(paragraph) >= needed + needed / 10);
        // Unspaced Mandarin is budgeted by characters, not as one word.
        let mandarin = "今天我们正在测试一个语音系统".repeat(4);
        assert!(qwen3_tts_auto_max_frames_for_text(&mandarin) >= 56 * 125 / 10 / 4);
    }

    /// Lane B, Metal: 16 sampled runs of a 46-word English paragraph ended
    /// between 222 and 328 frames, and its 313-frame budget (2.0 words/s plus
    /// 2 s) cut the slowest. The budget for 46 words now covers the slowest
    /// sampled ending with room to spare, while a greedy run that never
    /// emitted end of speech (400 frames and counting) is still cut as
    /// `length`.
    #[test]
    fn automatic_budget_covers_the_slowest_sampled_pacing() {
        let paragraph = "spoken ".repeat(46);
        let budget = qwen3_tts_auto_max_frames_for_text(&paragraph);
        assert!(budget >= 328 * 11 / 10, "budget {budget} frames");
        assert!(budget >= 222 * 3 / 2, "budget {budget} frames");
        assert!(budget < 400 * 12 / 10, "budget {budget} frames");
    }

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

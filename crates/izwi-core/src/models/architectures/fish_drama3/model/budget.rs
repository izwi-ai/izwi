//! Frame budgets, position limits and the stop contract (§7.5).
//!
//! A chunk's budget is either the request's `max_tokens` (at most 4,096,
//! the codec stream's position capacity) or an automatic one: Fish S2's
//! CJK-aware speaking estimate (`izwi-server`'s `tts_policy.rs`, ported here
//! so the family owns its numbers) times 2.5, the report's RL cap, at most
//! 1,024 frames. Running out of budget before `<|im_end|>` is an error,
//! never truncated success: a cut-off take sounds like a finished one.
//!
//! A budget of N allows N frames: the draw after the N-th must be eos, else
//! the request stops with [`FishDrama3StopReason::FrameLimit`]. Fish S2
//! stops at its N-th frame without drawing again (so a take of exactly N
//! frames fails there) and reserves one more position for eos; here the
//! extra draw reads the head at the last frame's row, which the N frames'
//! positions already hold.

use super::super::codec::FishDrama3DacConfig;
use super::super::prompt::MAX_PROMPT_ROWS;
use crate::error::{Error, Result};

/// Codec frames per second (44,100 / 2,048 ≈ 21.5), as Fish S2 rounds it.
pub const FISH_DRAMA3_FRAME_RATE_HZ: f32 = 21.5;
/// The automatic budget over the speaking estimate.
pub const AUTO_BUDGET_FACTOR: f32 = 2.5;
pub const MAX_AUTO_FRAMES: usize = 1_024;
/// An explicit `max_tokens` is at most the codec stream's capacity.
pub const MAX_EXPLICIT_FRAMES: usize = FishDrama3DacConfig::MAX_QUANTIZER_FRAMES;
/// Positions one request may occupy: the longest prompt and the largest
/// budget.
pub const MAX_REQUEST_POSITIONS: usize = MAX_PROMPT_ROWS + MAX_EXPLICIT_FRAMES;

/// The bounds Fish S2 clamps its estimate to (`tts_policy.rs:476-479`).
const S2_MIN_SECONDS: f32 = 6.0;
const S2_MAX_SECONDS: f32 = 120.0;
const S2_MIN_FRAMES: usize = 96;
const S2_MAX_FRAMES: usize = 4_096;

/// Fish S2's speaking estimate in frames: words (or five non-space,
/// non-CJK characters to a word, whichever is more) at 1.8 per second, CJK
/// syllabic characters at 2.5 per second, plus 4 s for pauses and eos.
fn speaking_frames(text: &str) -> usize {
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
    let words = text
        .split_whitespace()
        .count()
        .max(non_space.saturating_sub(cjk).div_ceil(5));
    let seconds =
        (words as f32 / 1.8 + cjk as f32 / 2.5 + 4.0).clamp(S2_MIN_SECONDS, S2_MAX_SECONDS);
    ((seconds * FISH_DRAMA3_FRAME_RATE_HZ).ceil() as usize).clamp(S2_MIN_FRAMES, S2_MAX_FRAMES)
}

/// The automatic budget of a chunk's text.
pub fn auto_frame_budget(text: &str) -> usize {
    ((speaking_frames(text) as f32 * AUTO_BUDGET_FACTOR).ceil() as usize).min(MAX_AUTO_FRAMES)
}

/// A chunk's budget: `max_tokens` when the request names one (1 to
/// [`MAX_EXPLICIT_FRAMES`]), else [`auto_frame_budget`].
pub fn frame_budget(text: &str, max_tokens: Option<usize>) -> Result<usize> {
    match max_tokens {
        None => Ok(auto_frame_budget(text)),
        Some(frames) if (1..=MAX_EXPLICIT_FRAMES).contains(&frames) => Ok(frames),
        Some(frames) => Err(Error::InvalidInput(format!(
            "Fish Audio Drama 3 max_tokens must be 1 to {MAX_EXPLICIT_FRAMES}; got {frames}"
        ))),
    }
}

/// Checks that a `prompt_rows` prompt and a `max_frames` budget fit
/// `positions`: the trunk ends at row `prompt_rows + max_frames` (the last
/// frame's row) and the tower one row short of it.
pub(crate) fn check_positions(
    prompt_rows: usize,
    max_frames: usize,
    positions: usize,
) -> Result<()> {
    if prompt_rows == 0 || prompt_rows > MAX_PROMPT_ROWS {
        return Err(Error::ContextLengthExceeded(format!(
            "Fish Audio Drama 3 prompt has {prompt_rows} rows; it needs 1 to {MAX_PROMPT_ROWS}"
        )));
    }
    if max_frames == 0 || max_frames > MAX_EXPLICIT_FRAMES {
        return Err(Error::InvalidInput(format!(
            "Fish Audio Drama 3 frame budget must be 1 to {MAX_EXPLICIT_FRAMES}; got {max_frames}"
        )));
    }
    let needed = prompt_rows + max_frames;
    let limit = positions.min(MAX_REQUEST_POSITIONS);
    if needed > limit {
        return Err(Error::ContextLengthExceeded(format!(
            "Fish Audio Drama 3 prompt of {prompt_rows} rows and {max_frames} frames needs \
             {needed} positions; at most {limit}"
        )));
    }
    Ok(())
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3StopReason {
    /// The model drew `<|im_end|>`.
    EndOfSpeech,
    /// The budget ran out first: the speech is incomplete.
    FrameLimit,
}

impl FishDrama3StopReason {
    /// Only an eos stop is a finished take.
    pub fn require_complete(self) -> Result<()> {
        match self {
            Self::EndOfSpeech => Ok(()),
            Self::FrameLimit => Err(Error::InferenceError(
                "Fish Audio Drama 3 generation incomplete: frame_limit reached before end of \
                 speech"
                    .into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_auto_budget_is_s2s_estimate_times_2_5_capped_at_1024() {
        // Short text: S2's 6 s floor, 129 frames, times 2.5.
        assert_eq!(speaking_frames("Hello"), 129);
        assert_eq!(auto_frame_budget("Hello"), 323);
        // 36 words: 36 / 1.8 + 4 = 24 s, 516 frames; capped.
        let long = "word ".repeat(36);
        assert_eq!(speaking_frames(&long), 516);
        assert_eq!(auto_frame_budget(&long), MAX_AUTO_FRAMES);
        // CJK counts per character: 30 characters at 2.5 per second, one
        // word, 4 s; 16.6 s.
        let cjk = "你".repeat(30);
        assert_eq!(speaking_frames(&cjk), 356);
        assert!(auto_frame_budget(&cjk) > auto_frame_budget("Hello"));
        assert_eq!(frame_budget("Hello", None).unwrap(), 323);
        assert_eq!(frame_budget("Hello", Some(7)).unwrap(), 7);
        assert_eq!(frame_budget("Hello", Some(4_096)).unwrap(), 4_096);
        for bad in [0, 4_097] {
            assert!(matches!(
                frame_budget("Hello", Some(bad)),
                Err(Error::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn positions_hold_the_prompt_and_every_frame() {
        assert_eq!(MAX_REQUEST_POSITIONS, 12_288);
        check_positions(10, 6, 16).unwrap();
        assert!(matches!(
            check_positions(10, 7, 16),
            Err(Error::ContextLengthExceeded(_))
        ));
        assert!(matches!(
            check_positions(8_193, 1, usize::MAX),
            Err(Error::ContextLengthExceeded(_))
        ));
        assert!(check_positions(0, 1, 16).is_err());
        assert!(check_positions(1, 0, 16).is_err());
        check_positions(8_192, 4_096, usize::MAX).unwrap();
        assert!(FishDrama3StopReason::EndOfSpeech.require_complete().is_ok());
        let err = FishDrama3StopReason::FrameLimit
            .require_complete()
            .unwrap_err();
        assert!(err.to_string().contains("frame_limit"), "{err}");
    }
}

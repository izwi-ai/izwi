//! Request-shaped admission prices for Kokoro synthesis (KOK-8, KOK-9).
//!
//! The model's hard contract allows up to 50 duration bins per token and
//! 4,096 frames per chunk, and admission used to reserve that ceiling for
//! every request: a 45-word stream reserved about 3.3 GB on Metal, 2.3 GB of
//! it for 4,096 frames of decoder workspace, plus the whole text's output.
//! Real speech runs about 2.7 duration frames (25 ms each) per input
//! character at speed 1. Admission now prices one chunk at a slow planning
//! pace, and the stream holds one chunk's output at a time.
//!
//! Characters alone under-priced short and numeric text: every chunk speaks
//! about 30-60 frames of boundary silence whatever its length ("Sure." is 52
//! frames against a 20-frame price), and the G2P reads digits and symbols as
//! words ("$1,234,567.89" is about 250 frames). The price adds a per-chunk
//! allowance and weighs characters by what they expand to, calibrated on the
//! real checkpoint across six voices (worst measured chunk 78% of its price).

use crate::backends::BackendKind;
use crate::error::{Error, Result};

use super::{
    kokoro_peak_workspace, KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK,
    KOKORO_MAX_SAMPLES_PER_DURATION_FRAME, KOKORO_MAX_SPEED, KOKORO_MIN_SPEED,
};

/// Planning pace: duration frames per input character at speed 1. Four is
/// ten characters per second, against about 16 measured live (`af_heart`,
/// 45 words in 15.5 s). A chunk that runs longer fails its sample contract
/// instead of outgrowing its reservation.
pub(crate) const KOKORO_PLANNING_FRAMES_PER_CHAR: u32 = 4;

/// Planning frames every chunk adds at speed 1 for its boundary tokens and
/// the silence spoken around any text: a one-character chunk speaks 29-58
/// frames.
pub(crate) const KOKORO_PLANNING_FRAMES_PER_CHUNK: u32 = 64;

/// Planning frames of one input character at speed 1, by what the G2P
/// reads it as.
fn kokoro_planning_char_frames(c: char) -> u32 {
    match c {
        // A digit reads as a word or more: "1,234,567" is "one million two
        // hundred thirty-four thousand ...".
        '0'..='9' => 28,
        // A symbol reads as a word ("dollars", "percent", "at").
        '$' | '%' | '&' | '@' | '#' | '+' | '=' | '/' | '*' | '<' | '>' | '^' | '~' | '|'
        | '\\' => 16,
        // Acronyms are spelled letter by letter.
        'A'..='Z' => 8,
        c if c.is_ascii() => KOKORO_PLANNING_FRAMES_PER_CHAR,
        // A CJK character is a syllable.
        c if c.is_alphabetic() => 10,
        _ => KOKORO_PLANNING_FRAMES_PER_CHAR,
    }
}

/// Admission price of one synthesized chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KokoroChunkPrice {
    /// Planned duration frames.
    pub frames: u64,
    /// Output samples of those frames.
    pub output_samples: u64,
    /// Harmonic-source host buffers.
    pub host_bytes: u64,
    /// Decoder workspace on the CPU, including concurrent im2col buffers.
    pub cpu_tensor_bytes: u64,
    /// Decoder workspace on an accelerator, including the im2col buffer
    /// (rounded up to a power of two on Metal).
    pub accelerator_tensor_bytes: u64,
}

/// The planned duration frames of the heaviest chunk of at most
/// `max_chunk_chars` characters that `text` can be cut into (the whole text
/// when it is shorter), never above the model's per-chunk cap.
pub(crate) fn kokoro_chunk_frame_bound(text: &str, max_chunk_chars: usize, speed: f32) -> usize {
    let speed = if speed.is_finite() {
        speed.clamp(KOKORO_MIN_SPEED, KOKORO_MAX_SPEED)
    } else {
        1.0
    };
    let weights: Vec<u32> = text
        .trim()
        .chars()
        .map(kokoro_planning_char_frames)
        .collect();
    let window = max_chunk_chars.clamp(1, weights.len().max(1));
    let mut sum: u32 = weights.iter().take(window).sum();
    let mut heaviest = sum;
    for (entering, leaving) in weights.iter().skip(window).zip(&weights) {
        sum = sum + entering - leaving;
        heaviest = heaviest.max(sum);
    }
    let frames = (KOKORO_PLANNING_FRAMES_PER_CHUNK + heaviest) as f32 / speed;
    (frames.ceil() as usize).clamp(1, KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK)
}

/// The admission price of one chunk of `frames` duration frames on
/// `backend`.
pub(crate) fn kokoro_chunk_price(frames: usize, backend: BackendKind) -> Result<KokoroChunkPrice> {
    let overflow = || Error::Overloaded("Kokoro chunk price overflowed".to_string());
    let frames = u64::try_from(frames).map_err(|_| overflow())?;
    let workspace = kokoro_peak_workspace(frames)?;
    // The decoder tiles its convs to 16 MiB each, so a chunk holds one
    // tile's im2col (KOK-9), not the whole stage's.
    let (cpu_im2col, accelerator_im2col) = super::decoder::kokoro_decoder_im2col_bytes(frames)?;
    let accelerator_im2col = if backend == BackendKind::Metal {
        accelerator_im2col
            .checked_next_power_of_two()
            .ok_or_else(overflow)?
    } else {
        accelerator_im2col
    };
    Ok(KokoroChunkPrice {
        frames,
        output_samples: frames
            .checked_mul(KOKORO_MAX_SAMPLES_PER_DURATION_FRAME as u64)
            .ok_or_else(overflow)?,
        host_bytes: workspace.host_bytes,
        cpu_tensor_bytes: workspace
            .cpu_tensor_bytes
            .checked_add(cpu_im2col)
            .ok_or_else(overflow)?,
        accelerator_tensor_bytes: workspace
            .accelerator_tensor_bytes
            .checked_add(accelerator_im2col)
            .ok_or_else(overflow)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_frames_follow_text_and_speed_under_the_model_cap() {
        let words = "ab ".repeat(40);
        assert_eq!(
            kokoro_chunk_frame_bound(&words, usize::MAX, 1.0),
            64 + 119 * 4
        );
        assert_eq!(
            kokoro_chunk_frame_bound(&words, usize::MAX, 0.5),
            2 * (64 + 119 * 4)
        );
        assert_eq!(
            kokoro_chunk_frame_bound(&"a".repeat(100_000), usize::MAX, 1.0),
            4_096
        );
        assert_eq!(kokoro_chunk_frame_bound("", usize::MAX, 1.0), 64);
        // A stream prices its heaviest chunk-sized window.
        let mixed = format!("{} 1234567890", "a".repeat(200));
        assert_eq!(kokoro_chunk_frame_bound(&mixed, 11, 1.0), 64 + 4 + 10 * 28);
        assert_eq!(kokoro_chunk_frame_bound(&mixed, 5, 1.0), 64 + 5 * 28);
        let metal = kokoro_chunk_price(500, BackendKind::Metal).unwrap();
        let cuda = kokoro_chunk_price(500, BackendKind::Cuda).unwrap();
        assert!(metal.accelerator_tensor_bytes >= cuda.accelerator_tensor_bytes);
        assert_eq!(metal.output_samples, 500 * 600);
    }

    /// Duration frames the real checkpoint spoke (CPU, speed 1) for short
    /// and expanded text, the slowest of `af_heart`, `af_bella`, `am_adam`,
    /// `am_michael`, `bf_emma` and `bm_george`. Each fits its chunk price
    /// with room to spare; the character-only price admitted "Sure." at 20
    /// frames, so the voice pipeline failed it (34,200 > 2,400 samples).
    #[test]
    fn short_and_expanded_text_fits_its_planned_frames() {
        let spoken = [
            ("A", 33),
            ("1", 52),
            ("OK.", 58),
            ("Sure.", 53),
            ("Yes.", 54),
            ("No problem.", 72),
            ("Wait, really?", 70),
            ("Call 911.", 112),
            ("It is 3:45 pm.", 119),
            ("Room 404.", 98),
            ("That costs $1,234,567.89.", 309),
            ("Version 3.14.159 shipped on 2026-10-17.", 311),
            ("Dr. Lee, 221B Baker St.", 172),
            ("The FBI and NASA met at 9.", 138),
            ("50% off & free @ 5/6.", 155),
            (
                "Of course! Here is a quick summary of what we covered today.",
                176,
            ),
            (
                "Call me at 555-867-5309 or 1-800-273-8255 before 11:59 PM on 12/31/2026, \
                 or wire $9,999,999.99 to account 0042.",
                1124,
            ),
        ];
        for (text, frames) in spoken {
            let bound = kokoro_chunk_frame_bound(text, usize::MAX, 1.0);
            assert!(
                frames * 5 <= bound * 4,
                "{text:?}: spoke {frames} of {bound} frames"
            );
        }
    }
}

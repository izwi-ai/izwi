//! Request-shaped admission prices for Kokoro synthesis (KOK-8, KOK-9).
//!
//! The model's hard contract allows up to 50 duration bins per token and
//! 4,096 frames per chunk, and admission used to reserve that ceiling for
//! every request: a 45-word stream reserved about 3.3 GB on Metal, 2.3 GB of
//! it for 4,096 frames of decoder workspace, plus the whole text's output.
//! Real speech runs about 2.7 duration frames (25 ms each) per input
//! character at speed 1. Admission now prices one chunk at a slow planning
//! pace, and the stream holds one chunk's output at a time.

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
pub(crate) const KOKORO_PLANNING_FRAMES_PER_CHAR: f32 = 4.0;

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

/// The planned duration frames of a chunk of `chars` input characters,
/// never above the model's per-chunk cap.
pub(crate) fn kokoro_chunk_frame_bound(chars: usize, speed: f32) -> usize {
    let speed = if speed.is_finite() {
        speed.clamp(KOKORO_MIN_SPEED, KOKORO_MAX_SPEED)
    } else {
        1.0
    };
    ((chars.max(1) as f32 * KOKORO_PLANNING_FRAMES_PER_CHAR / speed).ceil() as usize)
        .clamp(1, KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK)
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
        assert_eq!(kokoro_chunk_frame_bound(100, 1.0), 400);
        assert_eq!(kokoro_chunk_frame_bound(100, 0.5), 800);
        assert_eq!(kokoro_chunk_frame_bound(100_000, 1.0), 4_096);
        assert_eq!(kokoro_chunk_frame_bound(0, 1.0), 4);
        let metal = kokoro_chunk_price(500, BackendKind::Metal).unwrap();
        let cuda = kokoro_chunk_price(500, BackendKind::Cuda).unwrap();
        assert!(metal.accelerator_tensor_bytes >= cuda.accelerator_tensor_bytes);
        assert_eq!(metal.output_samples, 500 * 600);
    }
}

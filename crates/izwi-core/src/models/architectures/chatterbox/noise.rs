//! Portable, seeded randomness for Chatterbox.
//!
//! Every random draw a request makes (T3 sampling, the flow's initial noise,
//! the vocoder's harmonic phases and noise) comes from a host-side
//! SplitMix64 stream derived from the request seed
//! ([`crate::models::shared::noise`]), never from a device RNG.
//! A request's audio therefore depends only on its own seed, not on which
//! requests share its batch or which backend runs it, and the reference
//! fixture (`scripts/fixtures/generate_chatterbox_reference.py`) regenerates
//! the same noise in numpy to drive upstream's PyTorch code.

use crate::models::shared::noise::{mix, stream_seed as shared_stream_seed};
pub use crate::models::shared::noise::{resolve_seed, SplitMix64};

/// Independent streams of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseStream {
    /// T3 speech-token sampling.
    Sampling = 1,
    /// The flow's initial Gaussian noise.
    FlowNoise = 2,
    /// The vocoder's random harmonic phases.
    HarmonicPhase = 3,
    /// The vocoder's additive source noise.
    SourceNoise = 4,
}

impl From<NoiseStream> for u64 {
    fn from(stream: NoiseStream) -> Self {
        stream as u64
    }
}

/// The stream seed for `stream` of a request seeded `seed`.
pub fn stream_seed(seed: u64, stream: NoiseStream) -> u64 {
    shared_stream_seed(seed, stream.into())
}

/// Flow noise for absolute mel frames `first..first + frames` of a streamed
/// utterance, [bins, frames] row-major as `S3GenFlow::begin` takes it. Each
/// frame draws its `bins` values from a stream of its own, so a frame gets
/// the same noise in every window that renders it.
pub fn frame_flow_noise(seed: u64, first: usize, frames: usize, bins: usize) -> Vec<f32> {
    let base = stream_seed(seed, NoiseStream::FlowNoise);
    let mut noise = vec![0f32; bins * frames];
    for frame in 0..frames {
        let index = (first + frame) as u64;
        let values = SplitMix64::new(mix(base ^ mix(index.wrapping_add(1)))).standard_normals(bins);
        for (bin, value) in values.into_iter().enumerate() {
            noise[bin * frames + frame] = value;
        }
    }
    noise
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_independent_and_repeatable() {
        let a = SplitMix64::for_stream(7, NoiseStream::FlowNoise).standard_normals(5);
        let b = SplitMix64::for_stream(7, NoiseStream::FlowNoise).standard_normals(5);
        let c = SplitMix64::for_stream(7, NoiseStream::SourceNoise).standard_normals(5);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn frame_noise_does_not_depend_on_the_window() {
        let bins = 3;
        let wide = frame_flow_noise(7, 10, 6, bins);
        let narrow = frame_flow_noise(7, 12, 2, bins);
        for bin in 0..bins {
            assert_eq!(narrow[bin * 2], wide[bin * 6 + 2]);
            assert_eq!(narrow[bin * 2 + 1], wide[bin * 6 + 3]);
        }
        assert_ne!(frame_flow_noise(8, 12, 2, bins), narrow);
    }
}

//! Maya1's seeded randomness: the token sampler's stream and the SNAC
//! decoder's noise, both derived from the request seed on the host
//! ([`crate::models::shared::noise`]).
//!
//! SNAC adds `noise · conv1x1(x)` in each of its four decoder blocks. The
//! noise is drawn per block and per absolute frame, so a frame gets the same
//! values in every window that decodes it and in a whole-utterance decode.
//! The reference fixture regenerates it in numpy and patches it into
//! upstream's `NoiseBlock`.

use super::frames::SnacCodes;
use super::snac::{SnacConfig, SnacNoise};
use crate::models::shared::noise::{mix, stream_seed, SplitMix64};

/// Independent streams of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maya1Stream {
    /// Next-token sampling.
    Sampling = 1,
    /// SNAC's decoder-block noise.
    CodecNoise = 2,
}

impl From<Maya1Stream> for u64 {
    fn from(stream: Maya1Stream) -> Self {
        stream as u64
    }
}

/// Noise for decoder block `block` over absolute frames
/// `first..first + frames`: each frame's `steps_per_frame` values from a
/// stream of its own.
pub fn block_noise(
    seed: u64,
    block: usize,
    first: usize,
    frames: usize,
    steps_per_frame: usize,
) -> Vec<f32> {
    let base = stream_seed(seed, Maya1Stream::CodecNoise.into());
    let mut noise = Vec::with_capacity(frames * steps_per_frame);
    for frame in first..first + frames {
        let key = ((block as u64) << 40) ^ (frame as u64 + 1);
        noise.extend(SplitMix64::new(mix(base ^ mix(key))).standard_normals(steps_per_frame));
    }
    noise
}

/// SNAC noise for a batch of windows: row `i` covers absolute frames
/// `windows[i].0..` of `windows[i].1`'s length, seeded `windows[i].2`.
pub fn snac_noise(cfg: &SnacConfig, windows: &[(usize, &SnacCodes, u64)]) -> SnacNoise {
    std::array::from_fn(|block| {
        windows
            .iter()
            .flat_map(|&(first, codes, seed)| {
                block_noise(
                    seed,
                    block,
                    first,
                    codes.frames(),
                    cfg.block_steps_per_frame(block),
                )
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_noise_depends_on_the_frame_not_the_window() {
        let wide = block_noise(9, 2, 4, 3, 8);
        let narrow = block_noise(9, 2, 5, 1, 8);
        assert_eq!(&wide[8..16], narrow.as_slice());
        assert_ne!(block_noise(9, 1, 5, 1, 8), narrow);
        assert_ne!(block_noise(10, 2, 5, 1, 8), narrow);
    }
}

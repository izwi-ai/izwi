//! Portable, seeded randomness for Chatterbox.
//!
//! Every random draw a request makes (T3 sampling, the flow's initial noise,
//! the vocoder's harmonic phases and noise) comes from a host-side
//! SplitMix64 stream derived from the request seed, never from a device RNG.
//! A request's audio therefore depends only on its own seed, not on which
//! requests share its batch or which backend runs it, and the reference
//! fixture (`scripts/fixtures/generate_chatterbox_reference.py`) regenerates
//! the same noise in numpy to drive upstream's PyTorch code.

/// SplitMix64's increment (the 64-bit golden ratio).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

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

/// The stream seed for `stream` of a request seeded `seed`.
pub fn stream_seed(seed: u64, stream: NoiseStream) -> u64 {
    mix(seed ^ mix((stream as u64).wrapping_mul(GOLDEN_GAMMA)))
}

/// SplitMix64's output function.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn for_stream(seed: u64, stream: NoiseStream) -> Self {
        Self::new(stream_seed(seed, stream))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        mix(self.state)
    }

    /// Uniform in the open interval (0, 1): the top 53 bits, centred.
    pub fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// `count` standard normals by Box–Muller, two per pair of uniforms
    /// (`r cos θ` then `r sin θ`), computed in F64 and rounded once.
    pub fn standard_normals(&mut self, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count + 1);
        while out.len() < count {
            let radius = (-2.0 * self.next_open01().ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * self.next_open01();
            out.push((radius * angle.cos()) as f32);
            out.push((radius * angle.sin()) as f32);
        }
        out.truncate(count);
        out
    }

    /// `count` uniforms in (0, 1), rounded to F32.
    pub fn uniforms(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_open01() as f32).collect()
    }
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

/// A request seed: the caller's when it is positive, else one drawn from the
/// clock, as every izwi sampler treats seed 0.
pub fn resolve_seed(seed: Option<u64>) -> u64 {
    match seed {
        Some(seed) if seed != 0 => seed,
        _ => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or(0x5EED);
            mix(nanos).max(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64's published reference outputs for seed 1234567.
    #[test]
    fn splitmix64_matches_the_reference_sequence() {
        let mut rng = SplitMix64::new(1_234_567);
        let expected = [
            6_457_827_717_110_365_317u64,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
            4_593_380_528_125_082_431,
            16_408_922_859_458_223_821,
        ];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

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
    fn normals_have_unit_scale() {
        let values = SplitMix64::new(42).standard_normals(200_000);
        let n = values.len() as f64;
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
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

    #[test]
    fn seed_zero_draws_a_fresh_positive_seed() {
        assert_eq!(resolve_seed(Some(9)), 9);
        assert_ne!(resolve_seed(Some(0)), 0);
        assert_ne!(resolve_seed(None), 0);
    }
}

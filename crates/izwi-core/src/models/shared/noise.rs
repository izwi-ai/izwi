//! Portable, seeded randomness for generative audio models.
//!
//! A request's random draws (token sampling, codec or flow noise, vocoder
//! phases) come from host-side SplitMix64 streams derived from the request
//! seed, never from a device RNG. Its audio therefore depends only on its own
//! seed, not on which requests share its batch or which backend runs it, and
//! a reference fixture can regenerate the same values in numpy to drive the
//! upstream PyTorch code. Families name their own streams and pass each one's
//! number to [`SplitMix64::for_stream`].

/// SplitMix64's increment (the 64-bit golden ratio).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The seed of stream number `stream` of a request seeded `seed`.
pub fn stream_seed(seed: u64, stream: u64) -> u64 {
    mix(seed ^ mix(stream.wrapping_mul(GOLDEN_GAMMA)))
}

/// SplitMix64's output function, also used to derive per-position streams.
pub fn mix(mut z: u64) -> u64 {
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

    /// The generator of stream `stream` of a request seeded `seed`.
    pub fn for_stream(seed: u64, stream: impl Into<u64>) -> Self {
        Self::new(stream_seed(seed, stream.into()))
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
        let a = SplitMix64::for_stream(7, 2u64).standard_normals(5);
        let b = SplitMix64::for_stream(7, 2u64).standard_normals(5);
        let c = SplitMix64::for_stream(7, 4u64).standard_normals(5);
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
    fn seed_zero_draws_a_fresh_positive_seed() {
        assert_eq!(resolve_seed(Some(9)), 9);
        assert_ne!(resolve_seed(Some(0)), 0);
        assert_ne!(resolve_seed(None), 0);
    }
}

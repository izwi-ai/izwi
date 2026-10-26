//! Test helpers shared by the Chatterbox fixture tests: the synthetic
//! reference signal the fixture scripts regenerate bit for bit, and the
//! strided sample points they record.

use super::noise::SplitMix64;

#[derive(serde::Deserialize)]
pub(super) struct Sample {
    pub(super) shape: Vec<usize>,
    pub(super) indices: Vec<usize>,
    pub(super) values: Vec<f32>,
    pub(super) rms: f32,
}

/// The fixture script's `reference_signal`, operation for operation: a
/// pulse train with a scheduled integer period through two resonators,
/// SplitMix64 noise between syllables, silent margins; F64, rounded once.
pub(super) fn reference_signal() -> Vec<f32> {
    const LEAD: usize = 6_000;
    const SYLLABLE: usize = 6_000;
    const GAP: usize = 1_200;
    const PERIODS: [usize; 8] = [160, 166, 172, 178, 172, 166, 154, 148];
    const AMPLITUDES: [f64; 8] = [0.9, 0.6, 1.0, 0.7, 0.8, 0.5, 0.95, 0.65];
    const SYLLABLES: usize = 14;
    const F1: (f64, f64) = (1.7832, -0.8836);
    const F2: (f64, f64) = (1.5174, -0.8464);
    let mut out = vec![0f64; LEAD];
    let mut noise = SplitMix64::new(20261026);
    let (mut y1, mut y2) = ([0f64; 2], [0f64; 2]);
    for syllable in 0..SYLLABLES {
        let period = PERIODS[syllable % PERIODS.len()];
        let amplitude = AMPLITUDES[syllable % AMPLITUDES.len()];
        for n in 0..SYLLABLE + GAP {
            let x = if n < SYLLABLE {
                if n % period == 0 {
                    amplitude
                } else {
                    0.0
                }
            } else {
                (noise.next_open01() - 0.5) * 0.08
            };
            let a = x + F1.0 * y1[0] + F1.1 * y1[1];
            y1 = [a, y1[0]];
            let b = a + F2.0 * y2[0] + F2.1 * y2[1];
            y2 = [b, y2[0]];
            out.push(b * 0.05);
        }
    }
    out.extend([0.0; LEAD]);
    out.into_iter().map(|value| value as f32).collect()
}

/// Max |izwi − upstream| at the sample points over the upstream RMS.
pub(super) fn sample_error(name: &str, actual: &[f32], shape: &[usize], sample: &Sample) -> f32 {
    assert_eq!(shape, sample.shape.as_slice(), "{name} shape");
    let error = sample
        .indices
        .iter()
        .zip(&sample.values)
        .map(|(&index, &expected)| {
            let diff = (actual[index] - expected).abs();
            if diff.is_finite() {
                diff
            } else {
                f32::INFINITY
            }
        })
        .fold(0f32, f32::max);
    error / sample.rms.max(f32::EPSILON)
}

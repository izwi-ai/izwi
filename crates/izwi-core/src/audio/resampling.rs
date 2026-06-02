//! Shared high-quality audio resampling primitives.
//!
//! This module is intentionally a primitive, not a blanket replacement for
//! model-specific preprocessing. Model adapters should opt into it one call site
//! at a time with fixture-backed parity tests.

use rubato::{FftFixedInOut, Resampler};

use crate::error::{Error, Result};

pub fn resample_mono_high_quality(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Resampling sample rates must be greater than zero".to_string(),
        ));
    }
    if samples.is_empty() || src_rate == dst_rate {
        return Ok(samples.to_vec());
    }

    let target_len = target_sample_count(samples.len(), src_rate, dst_rate);
    if target_len == 0 {
        return Ok(Vec::new());
    }

    let mut resampler =
        FftFixedInOut::<f32>::new(src_rate as usize, dst_rate as usize, samples.len(), 1)
            .map_err(|err| Error::AudioError(format!("Failed to construct resampler: {err}")))?;
    let input_frames = resampler.input_frames_next();
    let mut input = samples.to_vec();
    input.resize(input_frames, 0.0);

    let mut output = resampler
        .process(&[input], None)
        .map_err(|err| Error::AudioError(format!("Failed to resample audio: {err}")))?
        .into_iter()
        .next()
        .unwrap_or_default();
    output.truncate(target_len);

    if output.len() < target_len {
        output.resize(target_len, 0.0);
    }
    for sample in &mut output {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }

    Ok(output)
}

pub fn target_sample_count(input_samples: usize, src_rate: u32, dst_rate: u32) -> usize {
    if src_rate == 0 || dst_rate == 0 || input_samples == 0 {
        return 0;
    }
    ((input_samples as u128 * dst_rate as u128 + (src_rate as u128 / 2)) / src_rate as u128)
        as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn identity_resampling_preserves_samples() {
        let input = vec![0.0, 0.25, -0.5, 0.75];
        let output = resample_mono_high_quality(&input, 24_000, 24_000)
            .expect("identity resampling should pass");

        assert_eq!(output, input);
    }

    #[test]
    fn resampling_preserves_duration_sample_count() {
        let input = sine(440.0, 48_000, 4_800);
        let output =
            resample_mono_high_quality(&input, 48_000, 16_000).expect("resampling should pass");

        assert_eq!(output.len(), 1_600);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn target_sample_count_rounds_to_nearest_duration() {
        assert_eq!(target_sample_count(4_800, 48_000, 16_000), 1_600);
        assert_eq!(target_sample_count(44_100, 44_100, 16_000), 16_000);
        assert_eq!(target_sample_count(0, 44_100, 16_000), 0);
    }

    #[test]
    fn downsampling_attenuates_above_target_nyquist() {
        let high_frequency = sine(12_000.0, 48_000, 4_800);
        let linear = resample_linear_for_test(&high_frequency, 48_000, 16_000);
        let high_quality = resample_mono_high_quality(&high_frequency, 48_000, 16_000)
            .expect("resampling should pass");

        assert!(
            rms(&high_quality) < rms(&linear) * 0.5,
            "high-quality RMS {} should be clearly below linear RMS {}",
            rms(&high_quality),
            rms(&linear)
        );
    }

    #[test]
    fn rejects_zero_sample_rates() {
        let err =
            resample_mono_high_quality(&[0.0], 0, 16_000).expect_err("zero input rate should fail");
        assert!(err.to_string().contains("greater than zero"));
    }

    fn sine(frequency_hz: f32, sample_rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| (TAU * frequency_hz * index as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples
            .iter()
            .map(|sample| (*sample as f64) * (*sample as f64))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt() as f32
    }

    fn resample_linear_for_test(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        let target_len = target_sample_count(samples.len(), src_rate, dst_rate);
        if target_len == 0 {
            return Vec::new();
        }
        (0..target_len)
            .map(|index| {
                let src_pos = index as f64 * src_rate as f64 / dst_rate as f64;
                let left = src_pos.floor() as usize;
                let right = (left + 1).min(samples.len().saturating_sub(1));
                let frac = (src_pos - left as f64) as f32;
                samples[left] * (1.0 - frac) + samples[right] * frac
            })
            .collect()
    }
}

//! Shared high-quality audio resampling primitives.
//!
//! This module is intentionally a primitive, not a blanket replacement for
//! model-specific preprocessing. Model adapters should opt into it one call site
//! at a time with fixture-backed parity tests.

use rubato::{FftFixedInOut, Resampler};

use crate::error::{Error, Result};

const DEFAULT_FFT_RESAMPLE_CHUNK_FRAMES: usize = 4096;

/// Incremental high-quality mono resampler.
///
/// Feed native-rate mono samples through [`Self::push`] and collect
/// canonical-rate output; [`Self::finish`] flushes the resampler's internal
/// tail. The first `output_delay` produced samples are held back so the
/// concatenated stream matches the whole-buffer [`resample_mono_high_quality`]
/// contract: the caller aligns the final buffer with
/// [`align_resampled_length`] once the total input length is known.
pub(crate) struct HighQualityResampler {
    resampler: FftFixedInOut<f32>,
    outbuffer: Vec<Vec<f32>>,
    input_tail: Vec<f32>,
    delay_holdback: Vec<f32>,
    delay_holdback_open: bool,
    produced_total: usize,
    finished: bool,
}

impl HighQualityResampler {
    pub(crate) fn new(src_rate: u32, dst_rate: u32) -> Result<Self> {
        if src_rate == 0 || dst_rate == 0 {
            return Err(Error::InvalidInput(
                "Resampling sample rates must be greater than zero".to_string(),
            ));
        }
        let resampler = FftFixedInOut::<f32>::new(
            src_rate as usize,
            dst_rate as usize,
            DEFAULT_FFT_RESAMPLE_CHUNK_FRAMES,
            1,
        )
        .map_err(|err| Error::AudioError(format!("Failed to construct resampler: {err}")))?;
        Ok(Self {
            outbuffer: vec![vec![0.0f32; resampler.output_frames_max()]; 1],
            resampler,
            input_tail: Vec::new(),
            delay_holdback: Vec::new(),
            delay_holdback_open: true,
            produced_total: 0,
            finished: false,
        })
    }

    /// Push native-rate mono samples, appending produced output to `out`.
    pub(crate) fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) -> Result<()> {
        debug_assert!(!self.finished, "push after finish");
        if samples.is_empty() {
            return Ok(());
        }
        self.input_tail.extend_from_slice(samples);
        while self.input_tail.len() >= self.resampler.input_frames_next() {
            let input = [&self.input_tail[..]];
            let (consumed, written) = self
                .resampler
                .process_into_buffer(&input, &mut self.outbuffer, None)
                .map_err(|err| Error::AudioError(format!("Failed to resample audio: {err}")))?;
            if consumed == 0 {
                break;
            }
            self.emit(written, out);
            self.input_tail.drain(..consumed);
        }
        Ok(())
    }

    /// Flush the retained input tail and the resampler's internal pipeline.
    ///
    /// `target_len` is the expected total canonical output length for the
    /// whole input stream (see [`target_sample_count`]). rubato keeps
    /// returning buffered frames from `process_partial(None)`, so the flush
    /// must be bounded by the expected output length exactly like the
    /// whole-buffer implementation — an unbounded drain spins forever.
    pub(crate) fn finish(&mut self, out: &mut Vec<f32>, target_len: usize) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let output_delay = self.resampler.output_delay();
        let flush_limit = target_len.saturating_add(output_delay);
        if !self.input_tail.is_empty() {
            let input = [&self.input_tail[..]];
            let (_consumed, written) = self
                .resampler
                .process_partial_into_buffer(Some(&input), &mut self.outbuffer, None)
                .map_err(|err| Error::AudioError(format!("Failed to resample audio: {err}")))?;
            self.emit(written, out);
            self.input_tail.clear();
        }
        while self.produced_total < flush_limit {
            let (_consumed, written) = self
                .resampler
                .process_partial_into_buffer::<Vec<f32>, Vec<f32>>(None, &mut self.outbuffer, None)
                .map_err(|err| Error::AudioError(format!("Failed to resample audio: {err}")))?;
            if written == 0 {
                break;
            }
            self.emit(written, out);
        }
        Ok(())
    }

    fn emit(&mut self, written: usize, out: &mut Vec<f32>) {
        if written == 0 {
            return;
        }
        self.produced_total = self.produced_total.saturating_add(written);
        let produced = &self.outbuffer[0][..written];
        if !self.delay_holdback_open {
            out.extend_from_slice(produced);
            return;
        }
        let delay = self.resampler.output_delay();
        self.delay_holdback.extend_from_slice(produced);
        if self.delay_holdback.len() >= delay {
            out.extend_from_slice(&self.delay_holdback[delay..]);
            self.delay_holdback.clear();
            self.delay_holdback_open = false;
        }
    }
}

/// Align streamed resampler output to the exact duration contract of the
/// whole-buffer API: truncate or zero-pad to the target length, then sanitize
/// non-finite samples.
pub(crate) fn align_resampled_length(output: &mut Vec<f32>, target_len: usize) {
    output.truncate(target_len);
    if output.len() < target_len {
        output.resize(target_len, 0.0);
    }
    for sample in output.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
}

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

    let mut resampler = HighQualityResampler::new(src_rate, dst_rate)?;
    let mut output = Vec::with_capacity(target_len);
    resampler.push(samples, &mut output)?;
    resampler.finish(&mut output, target_len)?;
    align_resampled_length(&mut output, target_len);
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
    fn resampling_trims_fft_output_delay_for_short_clips() {
        let mut impulse = vec![0.0f32; 86_400];
        impulse[120] = 1.0;

        let delayed = resample_one_fft_chunk_for_test(&impulse, 24_000, 16_000);
        let corrected =
            resample_mono_high_quality(&impulse, 24_000, 16_000).expect("resampling should pass");

        assert_eq!(
            corrected.len(),
            target_sample_count(impulse.len(), 24_000, 16_000)
        );
        assert!(
            max_abs_index(&delayed) > corrected.len() / 3,
            "single whole-clip FFT resampling should retain a large delay"
        );
        assert!(
            max_abs_index(&corrected) < 2_000,
            "corrected resampling should keep the impulse near the start"
        );
    }

    #[test]
    fn rejects_zero_sample_rates() {
        let err =
            resample_mono_high_quality(&[0.0], 0, 16_000).expect_err("zero input rate should fail");
        assert!(err.to_string().contains("greater than zero"));
    }

    #[test]
    fn streaming_resampler_matches_whole_buffer_output() {
        let input = sine(220.0, 44_100, 96_000);
        let whole = resample_mono_high_quality(&input, 44_100, 16_000).expect("whole-buffer");
        let target = target_sample_count(input.len(), 44_100, 16_000);
        for chunk_frames in [1usize, 333, 4_096, 12_257] {
            let mut streamed = Vec::new();
            let mut resampler = HighQualityResampler::new(44_100, 16_000).expect("resampler");
            for chunk in input.chunks(chunk_frames) {
                resampler.push(chunk, &mut streamed).expect("push");
            }
            resampler.finish(&mut streamed, target).expect("finish");
            align_resampled_length(&mut streamed, target);
            assert_eq!(streamed.len(), whole.len(), "chunk size {chunk_frames}");
            let max_delta = streamed
                .iter()
                .zip(&whole)
                .map(|(streamed, whole)| (streamed - whole).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                max_delta, 0.0,
                "streamed output diverged at chunk size {chunk_frames}"
            );
        }
    }

    #[test]
    fn streaming_resampler_preserves_short_clip_duration() {
        let input = sine(440.0, 44_100, 22_050);
        let target = target_sample_count(input.len(), 44_100, 16_000);
        let mut streamed = Vec::new();
        let mut resampler = HighQualityResampler::new(44_100, 16_000).expect("resampler");
        for chunk in input.chunks(1_024) {
            resampler.push(chunk, &mut streamed).expect("push");
        }
        resampler.finish(&mut streamed, target).expect("finish");
        align_resampled_length(&mut streamed, target);
        assert_eq!(streamed.len(), 8_000);
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

    fn max_abs_index(samples: &[f32]) -> usize {
        samples
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.abs().total_cmp(&rhs.abs()))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn resample_one_fft_chunk_for_test(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        let target_len = target_sample_count(samples.len(), src_rate, dst_rate);
        let mut resampler =
            FftFixedInOut::<f32>::new(src_rate as usize, dst_rate as usize, samples.len(), 1)
                .unwrap();
        let input_frames = resampler.input_frames_next();
        let mut input = samples.to_vec();
        input.resize(input_frames, 0.0);
        let mut output = resampler.process(&[input], None).unwrap().remove(0);
        output.truncate(target_len);
        output
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

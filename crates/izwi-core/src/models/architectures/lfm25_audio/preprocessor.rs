use std::sync::Arc;

use candle_core::{Device, Tensor};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};

use super::config::{
    LFM25_AUDIO_INPUT_HOP_LENGTH, LFM25_AUDIO_INPUT_N_FFT, LFM25_AUDIO_INPUT_SAMPLE_RATE,
    LFM25_AUDIO_INPUT_WIN_LENGTH,
};

const NUM_MEL_BINS: usize = 128;
const F_MIN: f32 = 0.0;
const F_MAX: f32 = 8_000.0;
const NORMALIZE_EPS: f32 = 1e-5;

pub struct Lfm25AudioPreprocessor {
    mel_filterbank_spans: Vec<MelFilterSpan>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

struct MelFilterSpan {
    start: usize,
    weights: Vec<f32>,
}

impl Lfm25AudioPreprocessor {
    pub fn load() -> Result<Self> {
        let mel_filterbank_spans = sparse_mel_filterbank(&create_mel_filterbank(
            LFM25_AUDIO_INPUT_N_FFT / 2 + 1,
            NUM_MEL_BINS,
            LFM25_AUDIO_INPUT_SAMPLE_RATE as f32,
            F_MIN,
            F_MAX,
        ));
        let window = hann_window_padded(LFM25_AUDIO_INPUT_N_FFT, LFM25_AUDIO_INPUT_WIN_LENGTH);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(LFM25_AUDIO_INPUT_N_FFT);
        Ok(Self {
            mel_filterbank_spans,
            window,
            fft,
        })
    }

    pub fn compute_features(&self, waveform: &[f32], device: &Device) -> Result<(Tensor, usize)> {
        if waveform.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let effective_frames = effective_frame_count(waveform.len());
        let padded = reflect_pad_center(waveform, LFM25_AUDIO_INPUT_N_FFT / 2);
        let total_frames = if padded.len() >= LFM25_AUDIO_INPUT_N_FFT {
            (padded.len() - LFM25_AUDIO_INPUT_N_FFT) / LFM25_AUDIO_INPUT_HOP_LENGTH + 1
        } else {
            1
        };

        let mut log_mel = Vec::with_capacity(total_frames * NUM_MEL_BINS);
        let mut frame = vec![Complex::<f32>::new(0.0, 0.0); LFM25_AUDIO_INPUT_N_FFT];
        for frame_idx in 0..total_frames {
            let start = frame_idx * LFM25_AUDIO_INPUT_HOP_LENGTH;
            let frame_len = padded
                .len()
                .saturating_sub(start)
                .min(LFM25_AUDIO_INPUT_N_FFT);
            for (sample_idx, slot) in frame.iter_mut().enumerate() {
                if sample_idx < frame_len {
                    slot.re = padded[start + sample_idx] * self.window[sample_idx];
                } else {
                    slot.re = 0.0;
                }
                slot.im = 0.0;
            }

            self.fft.process(&mut frame);
            for filter in &self.mel_filterbank_spans {
                let mut sum = 0.0f32;
                for (offset, weight) in filter.weights.iter().copied().enumerate() {
                    sum += frame[filter.start + offset].norm_sqr() * weight;
                }
                log_mel.push(sum.max(1e-10).ln());
            }
        }
        normalize_flat_per_feature(&mut log_mel, total_frames, effective_frames, NUM_MEL_BINS);

        let mut flat = Vec::with_capacity(NUM_MEL_BINS * total_frames);
        for mel_idx in 0..NUM_MEL_BINS {
            for frame_idx in 0..total_frames {
                flat.push(log_mel[frame_idx * NUM_MEL_BINS + mel_idx]);
            }
        }

        let features = Tensor::from_vec(flat, (1, NUM_MEL_BINS, total_frames), device)?;
        Ok((features, effective_frames.min(total_frames)))
    }
}

pub fn effective_frame_count(num_samples: usize) -> usize {
    (num_samples / LFM25_AUDIO_INPUT_HOP_LENGTH).max(1)
}

fn normalize_per_feature(log_mel: &mut [Vec<f32>], effective_frames: usize) {
    if log_mel.is_empty() || log_mel[0].is_empty() {
        return;
    }

    let usable_frames = effective_frames.min(log_mel.len()).max(1);
    let mel_bins = log_mel[0].len();

    for mel_idx in 0..mel_bins {
        let mean = log_mel[..usable_frames]
            .iter()
            .map(|frame| frame[mel_idx])
            .sum::<f32>()
            / usable_frames as f32;

        let variance = if usable_frames > 1 {
            log_mel[..usable_frames]
                .iter()
                .map(|frame| {
                    let delta = frame[mel_idx] - mean;
                    delta * delta
                })
                .sum::<f32>()
                / (usable_frames - 1) as f32
        } else {
            0.0
        };
        let inv_std = 1.0 / (variance + NORMALIZE_EPS).sqrt();

        for frame in &mut log_mel[..usable_frames] {
            frame[mel_idx] = (frame[mel_idx] - mean) * inv_std;
        }
        for frame in &mut log_mel[usable_frames..] {
            frame[mel_idx] = 0.0;
        }
    }
}

fn normalize_flat_per_feature(
    log_mel: &mut [f32],
    total_frames: usize,
    effective_frames: usize,
    mel_bins: usize,
) {
    if total_frames == 0 || mel_bins == 0 {
        return;
    }

    let usable_frames = effective_frames.min(total_frames).max(1);
    for mel_idx in 0..mel_bins {
        let mean = (0..usable_frames)
            .map(|frame_idx| log_mel[frame_idx * mel_bins + mel_idx])
            .sum::<f32>()
            / usable_frames as f32;

        let variance = if usable_frames > 1 {
            (0..usable_frames)
                .map(|frame_idx| {
                    let delta = log_mel[frame_idx * mel_bins + mel_idx] - mean;
                    delta * delta
                })
                .sum::<f32>()
                / (usable_frames - 1) as f32
        } else {
            0.0
        };
        let inv_std = 1.0 / (variance + NORMALIZE_EPS).sqrt();

        for frame_idx in 0..usable_frames {
            let idx = frame_idx * mel_bins + mel_idx;
            log_mel[idx] = (log_mel[idx] - mean) * inv_std;
        }
        for frame_idx in usable_frames..total_frames {
            log_mel[frame_idx * mel_bins + mel_idx] = 0.0;
        }
    }
}

fn hann_window_padded(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0; n_fft];
    let left_pad = (n_fft.saturating_sub(win_length)) / 2;
    for idx in 0..win_length {
        let value =
            0.5 * (1.0 - f32::cos(2.0 * std::f32::consts::PI * idx as f32 / win_length as f32));
        window[left_pad + idx] = value;
    }
    window
}

fn create_mel_filterbank(
    n_freqs: usize,
    n_mels: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|idx| (sample_rate / 2.0) * idx as f32 / (n_freqs - 1) as f32)
        .collect();

    let mel_min = hertz_to_mel(f_min);
    let mel_max = hertz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|idx| mel_min + (mel_max - mel_min) * idx as f32 / (n_mels + 1) as f32)
        .collect();
    let filter_freqs: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hertz(mel)).collect();

    let mut filters = vec![vec![0.0; n_freqs]; n_mels];
    for mel_idx in 0..n_mels {
        let lower = filter_freqs[mel_idx];
        let center = filter_freqs[mel_idx + 1];
        let upper = filter_freqs[mel_idx + 2];

        for (freq_idx, &freq) in fft_freqs.iter().enumerate() {
            let up = if center > lower {
                (freq - lower) / (center - lower)
            } else {
                0.0
            };
            let down = if upper > center {
                (upper - freq) / (upper - center)
            } else {
                0.0
            };
            filters[mel_idx][freq_idx] = up.min(down).max(0.0);
        }

        let norm = if upper > lower {
            2.0 / (upper - lower)
        } else {
            0.0
        };
        for weight in &mut filters[mel_idx] {
            *weight *= norm;
        }
    }

    filters
}

fn sparse_mel_filterbank(filters: &[Vec<f32>]) -> Vec<MelFilterSpan> {
    filters
        .iter()
        .map(|filter| {
            let start = filter.iter().position(|weight| *weight != 0.0);
            let end = filter.iter().rposition(|weight| *weight != 0.0);
            match (start, end) {
                (Some(start), Some(end)) if start <= end => MelFilterSpan {
                    start,
                    weights: filter[start..=end].to_vec(),
                },
                _ => MelFilterSpan {
                    start: 0,
                    weights: Vec::new(),
                },
            }
        })
        .collect()
}

fn reflect_pad_center(waveform: &[f32], pad: usize) -> Vec<f32> {
    if waveform.is_empty() {
        return vec![0.0; pad * 2];
    }
    if waveform.len() == 1 {
        return std::iter::repeat_n(waveform[0], pad)
            .chain(std::iter::once(waveform[0]))
            .chain(std::iter::repeat_n(waveform[0], pad))
            .collect();
    }

    let mut out = Vec::with_capacity(waveform.len() + pad * 2);
    for idx in 0..pad {
        let reflected = (pad - idx).min(waveform.len() - 1);
        out.push(waveform[reflected]);
    }
    out.extend_from_slice(waveform);
    for idx in 0..pad {
        let reflected = waveform.len().saturating_sub(2 + idx);
        out.push(waveform[reflected]);
    }
    out
}

fn hertz_to_mel(freq: f32) -> f32 {
    let min_log_hertz = 1_000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / (6.4f32).ln();

    if freq < min_log_hertz {
        3.0 * freq / 200.0
    } else {
        min_log_mel + (freq / min_log_hertz).ln() * logstep
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    let min_log_hertz = 1_000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / (6.4f32).ln();

    if mel < min_log_mel {
        mel * 200.0 / 3.0
    } else {
        min_log_hertz * ((mel - min_log_mel) / logstep).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_frame_count, normalize_per_feature};

    #[test]
    fn effective_frame_count_uses_floor_hop_with_minimum_one() {
        assert_eq!(effective_frame_count(1), 1);
        assert_eq!(effective_frame_count(159), 1);
        assert_eq!(effective_frame_count(160), 1);
        assert_eq!(effective_frame_count(16_000), 100);
        assert_eq!(effective_frame_count(16_159), 100);
    }

    #[test]
    fn per_feature_normalization_zeroes_padded_tail() {
        let mut features = vec![vec![1.0, 10.0], vec![3.0, 14.0], vec![9.0, 99.0]];
        normalize_per_feature(&mut features, 2);

        assert!((features[0][0] + 0.70710677).abs() < 1e-4);
        assert!((features[1][0] - 0.70710677).abs() < 1e-4);
        assert!((features[0][1] + 0.70710677).abs() < 1e-4);
        assert!((features[1][1] - 0.70710677).abs() < 1e-4);
        assert_eq!(features[2], vec![0.0, 0.0]);
    }
}

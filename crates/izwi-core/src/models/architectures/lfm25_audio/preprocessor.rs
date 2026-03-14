use candle_core::{Device, Tensor};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

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
    mel_filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl Lfm25AudioPreprocessor {
    pub fn load() -> Result<Self> {
        let mel_filterbank = create_mel_filterbank(
            LFM25_AUDIO_INPUT_N_FFT / 2 + 1,
            NUM_MEL_BINS,
            LFM25_AUDIO_INPUT_SAMPLE_RATE as f32,
            F_MIN,
            F_MAX,
        );
        let window =
            hann_window_padded(LFM25_AUDIO_INPUT_N_FFT, LFM25_AUDIO_INPUT_WIN_LENGTH);
        Ok(Self {
            mel_filterbank,
            window,
        })
    }

    pub fn compute_features(&self, waveform: &[f32], device: &Device) -> Result<(Tensor, usize)> {
        if waveform.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let effective_frames = effective_frame_count(waveform.len());
        let stft = self.stft(waveform)?;
        let power = self.power_spectrogram(&stft);
        let mel = self.apply_mel_filterbank(&power);
        let mut log_mel = self.log_mel(&mel);
        normalize_per_feature(&mut log_mel, effective_frames);

        let total_frames = log_mel.len().max(1);
        let mut flat = Vec::with_capacity(NUM_MEL_BINS * total_frames);
        for mel_idx in 0..NUM_MEL_BINS {
            for frame in &log_mel {
                flat.push(frame[mel_idx]);
            }
        }

        let features = Tensor::from_vec(flat, (1, NUM_MEL_BINS, total_frames), device)?;
        Ok((features, effective_frames.min(total_frames)))
    }

    fn stft(&self, waveform: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(LFM25_AUDIO_INPUT_N_FFT);

        let padded = reflect_pad_center(waveform, LFM25_AUDIO_INPUT_N_FFT / 2);
        let num_frames = if padded.len() >= LFM25_AUDIO_INPUT_N_FFT {
            (padded.len() - LFM25_AUDIO_INPUT_N_FFT) / LFM25_AUDIO_INPUT_HOP_LENGTH + 1
        } else {
            1
        };

        let mut frames = Vec::with_capacity(num_frames);
        for frame_idx in 0..num_frames {
            let start = frame_idx * LFM25_AUDIO_INPUT_HOP_LENGTH;
            let end = (start + LFM25_AUDIO_INPUT_N_FFT).min(padded.len());

            let mut frame: Vec<Complex<f32>> = padded[start..end]
                .iter()
                .zip(self.window.iter())
                .map(|(&sample, &window)| Complex::new(sample * window, 0.0))
                .collect();
            frame.resize(LFM25_AUDIO_INPUT_N_FFT, Complex::new(0.0, 0.0));
            fft.process(&mut frame);
            frames.push(frame);
        }

        Ok(frames)
    }

    fn power_spectrogram(&self, stft: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
        stft.iter()
            .map(|frame| {
                frame[..LFM25_AUDIO_INPUT_N_FFT / 2 + 1]
                    .iter()
                    .map(|value| value.norm_sqr())
                    .collect()
            })
            .collect()
    }

    fn apply_mel_filterbank(&self, power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        power_spec
            .iter()
            .map(|frame| {
                self.mel_filterbank
                    .iter()
                    .map(|mel_filter| {
                        frame
                            .iter()
                            .zip(mel_filter.iter())
                            .map(|(&power, &weight)| power * weight)
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect()
    }

    fn log_mel(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mel_spec
            .iter()
            .map(|frame| frame.iter().map(|&x| x.max(1e-10).ln()).collect())
            .collect()
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

fn hann_window_padded(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0; n_fft];
    let left_pad = (n_fft.saturating_sub(win_length)) / 2;
    for idx in 0..win_length {
        let value = 0.5
            * (1.0
                - f32::cos(2.0 * std::f32::consts::PI * idx as f32 / win_length as f32));
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
        let mut features = vec![
            vec![1.0, 10.0],
            vec![3.0, 14.0],
            vec![9.0, 99.0],
        ];
        normalize_per_feature(&mut features, 2);

        assert!((features[0][0] + 0.70710677).abs() < 1e-4);
        assert!((features[1][0] - 0.70710677).abs() < 1e-4);
        assert!((features[0][1] + 0.70710677).abs() < 1e-4);
        assert!((features[1][1] - 0.70710677).abs() < 1e-4);
        assert_eq!(features[2], vec![0.0, 0.0]);
    }
}

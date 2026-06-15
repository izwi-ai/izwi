//! Audio preprocessing utilities for ASR.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: Option<usize>,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub normalize: bool,
    pub mel_scale: MelScale,
    pub mel_norm: MelNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Slaney,
    Htk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    Slaney,
    None,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 400, // Whisper and the retained Qwen forced-aligner stack use 25 ms windows
            win_length: None,
            hop_length: 160, // 10ms at 16kHz
            n_mels: 128,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: true,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        }
    }
}

pub struct MelSpectrogram {
    config: MelConfig,
    mel_filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new(config: MelConfig) -> Result<Self> {
        let mel_filterbank = Self::create_mel_filterbank(
            config.n_fft / 2 + 1,
            config.n_mels,
            config.sample_rate as f32,
            config.f_min,
            config.f_max,
            config.mel_scale,
            config.mel_norm,
        );
        let window =
            Self::hann_window_padded(config.n_fft, config.win_length.unwrap_or(config.n_fft));

        Ok(Self {
            config,
            mel_filterbank,
            window,
        })
    }

    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    pub fn compute(&self, waveform: &[f32]) -> Result<Vec<Vec<f32>>> {
        let stft = self.stft(waveform)?;
        let power_spec = self.power_spectrogram(&stft);
        let mel_spec = self.apply_mel_filterbank(&power_spec);
        let mut log_mel = self.log_mel(&mel_spec);

        // NOTE: The retained Qwen forced-aligner path may need all frames including the last one
        // Previously we did: if !log_mel.is_empty() { log_mel.pop(); }

        if self.config.normalize {
            Self::whisper_normalize(&mut log_mel);
        }

        Ok(log_mel)
    }

    fn stft(&self, waveform: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);

        let padded = self.reflect_pad_center(waveform);
        let num_frames = if padded.len() >= self.config.n_fft {
            (padded.len() - self.config.n_fft) / self.config.hop_length + 1
        } else {
            1
        };

        let mut result = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_length;
            let end = (start + self.config.n_fft).min(padded.len());

            let mut frame: Vec<Complex<f32>> = padded[start..end]
                .iter()
                .zip(&self.window[..end - start])
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            frame.resize(self.config.n_fft, Complex::new(0.0, 0.0));
            fft.process(&mut frame);
            result.push(frame);
        }

        Ok(result)
    }

    fn power_spectrogram(&self, stft: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
        stft.iter()
            .map(|frame| {
                frame[..self.config.n_fft / 2 + 1]
                    .iter()
                    .map(|c| c.norm_sqr())
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
                    .map(|mel_filter| frame.iter().zip(mel_filter).map(|(&p, &m)| p * m).sum())
                    .collect()
            })
            .collect()
    }

    fn log_mel(&self, mel_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        mel_spec
            .iter()
            .map(|frame| frame.iter().map(|&x| (x.max(1e-10)).log10()).collect())
            .collect()
    }

    fn whisper_normalize(log_mel: &mut [Vec<f32>]) {
        let mut max_val = f32::NEG_INFINITY;
        for frame in log_mel.iter() {
            for &v in frame.iter() {
                if v > max_val {
                    max_val = v;
                }
            }
        }

        let clamp_min = max_val - 8.0;
        for frame in log_mel.iter_mut() {
            for v in frame.iter_mut() {
                if *v < clamp_min {
                    *v = clamp_min;
                }
                *v = (*v + 4.0) / 4.0;
            }
        }
    }

    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - f32::cos(2.0 * std::f32::consts::PI * i as f32 / size as f32)))
            .collect()
    }

    fn hann_window_padded(n_fft: usize, win_length: usize) -> Vec<f32> {
        let win_length = win_length.clamp(1, n_fft.max(1));
        if win_length == n_fft {
            return Self::hann_window(n_fft);
        }
        let mut window = vec![0.0; n_fft];
        let offset = (n_fft - win_length) / 2;
        for (idx, value) in Self::hann_window(win_length).into_iter().enumerate() {
            window[offset + idx] = value;
        }
        window
    }

    fn create_mel_filterbank(
        n_freqs: usize,
        n_mels: usize,
        sample_rate: f32,
        f_min: f32,
        f_max: f32,
        mel_scale: MelScale,
        mel_norm: MelNorm,
    ) -> Vec<Vec<f32>> {
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| (sample_rate / 2.0) * (i as f32) / (n_freqs - 1) as f32)
            .collect();

        let mel_min = hertz_to_mel(f_min, mel_scale);
        let mel_max = hertz_to_mel(f_max, mel_scale);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let filter_freqs: Vec<f32> = mel_points
            .iter()
            .map(|&m| mel_to_hertz(m, mel_scale))
            .collect();

        // Shape: [n_mels, n_freqs] - each row is a mel bin's weights across frequencies
        let mut mel_filters = vec![vec![0.0; n_freqs]; n_mels];
        for mel_idx in 0..n_mels {
            let lower = filter_freqs[mel_idx];
            let center = filter_freqs[mel_idx + 1];
            let upper = filter_freqs[mel_idx + 2];

            for freq_idx in 0..n_freqs {
                let freq = fft_freqs[freq_idx];

                let down = if center > lower {
                    (freq - lower) / (center - lower)
                } else {
                    0.0
                };
                let up = if upper > center {
                    (upper - freq) / (upper - center)
                } else {
                    0.0
                };
                let val = down.min(up).max(0.0);
                mel_filters[mel_idx][freq_idx] = val;
            }
        }

        if mel_norm == MelNorm::Slaney {
            // Slaney-style normalization (constant energy per channel).
            let mut norms = vec![0.0; n_mels];
            for mel_idx in 0..n_mels {
                let low = filter_freqs[mel_idx];
                let high = filter_freqs[mel_idx + 2];
                norms[mel_idx] = if high > low { 2.0 / (high - low) } else { 0.0 };
            }
            for mel_idx in 0..n_mels {
                for freq_idx in 0..n_freqs {
                    mel_filters[mel_idx][freq_idx] *= norms[mel_idx];
                }
            }
        }

        mel_filters
    }

    fn reflect_pad_center(&self, waveform: &[f32]) -> Vec<f32> {
        let pad = self.config.n_fft / 2;
        if waveform.is_empty() {
            return vec![0.0; pad * 2];
        }
        let n = waveform.len();
        if n == 1 {
            let mut out = Vec::with_capacity(n + pad * 2);
            out.extend(std::iter::repeat(waveform[0]).take(pad));
            out.push(waveform[0]);
            out.extend(std::iter::repeat(waveform[0]).take(pad));
            return out;
        }

        let mut out = Vec::with_capacity(n + pad * 2);
        for i in 0..pad {
            let idx = (pad - i).min(n - 1);
            out.push(waveform[idx]);
        }
        out.extend_from_slice(waveform);
        for i in 0..pad {
            let idx = n.saturating_sub(2 + i);
            out.push(waveform[idx]);
        }
        out
    }
}

fn hertz_to_mel(freq: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Slaney => {
            let min_log_hertz = 1000.0;
            let min_log_mel = 15.0;
            let logstep = 27.0 / (6.4f32).ln();

            if freq < min_log_hertz {
                3.0 * freq / 200.0
            } else {
                min_log_mel + (freq / min_log_hertz).ln() * logstep
            }
        }
        MelScale::Htk => 2595.0 * (1.0 + freq / 700.0).log10(),
    }
}

fn mel_to_hertz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Slaney => {
            let min_log_hertz = 1000.0;
            let min_log_mel = 15.0;
            let logstep = (6.4f32).ln() / 27.0;

            if mel < min_log_mel {
                200.0 * mel / 3.0
            } else {
                min_log_hertz * ((mel - min_log_mel) * logstep).exp()
            }
        }
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htk_mel_scale_matches_standard_reference_points() {
        let mel_1000 = hertz_to_mel(1000.0, MelScale::Htk);
        assert!((mel_1000 - 999.9855).abs() < 0.001);

        let hz = mel_to_hertz(mel_1000, MelScale::Htk);
        assert!((hz - 1000.0).abs() < 0.01);
    }

    #[test]
    fn default_mel_config_preserves_slaney_filterbank_behavior() {
        let cfg = MelConfig::default();
        assert_eq!(cfg.mel_scale, MelScale::Slaney);
        assert_eq!(cfg.mel_norm, MelNorm::Slaney);
    }

    #[test]
    fn mel_filterbank_can_disable_slaney_area_normalization() {
        let slaney = MelSpectrogram::create_mel_filterbank(
            257,
            80,
            16_000.0,
            0.0,
            8_000.0,
            MelScale::Slaney,
            MelNorm::Slaney,
        );
        let htk_no_norm = MelSpectrogram::create_mel_filterbank(
            257,
            80,
            16_000.0,
            0.0,
            8_000.0,
            MelScale::Htk,
            MelNorm::None,
        );

        let slaney_sum: f32 = slaney.iter().flatten().copied().sum();
        let htk_sum: f32 = htk_no_norm.iter().flatten().copied().sum();
        assert!(htk_sum > slaney_sum);
        assert!((htk_sum - slaney_sum).abs() > 1.0);
    }
}

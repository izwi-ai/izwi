//! Audio preprocessing utilities for ASR.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub normalize: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 400,      // Whisper and the retained Qwen forced-aligner stack use 25 ms windows
            hop_length: 160, // 10ms at 16kHz
            n_mels: 128,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: true,
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
        );
        let window = Self::hann_window(config.n_fft);

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

    fn create_mel_filterbank(
        n_freqs: usize,
        n_mels: usize,
        sample_rate: f32,
        f_min: f32,
        f_max: f32,
    ) -> Vec<Vec<f32>> {
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| (sample_rate / 2.0) * (i as f32) / (n_freqs - 1) as f32)
            .collect();

        let mel_min = hertz_to_mel(f_min);
        let mel_max = hertz_to_mel(f_max);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let filter_freqs: Vec<f32> = mel_points.iter().map(|&m| mel_to_hertz(m)).collect();

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

fn hertz_to_mel(freq: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / (6.4f32).ln();

    if freq < min_log_hertz {
        3.0 * freq / 200.0
    } else {
        min_log_mel + (freq / min_log_hertz).ln() * logstep
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        200.0 * mel / 3.0
    } else {
        min_log_hertz * ((mel - min_log_mel) * logstep).exp()
    }
}

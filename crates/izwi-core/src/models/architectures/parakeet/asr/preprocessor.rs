use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};

const PREEMPH: f32 = 0.97;
const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const DEFAULT_N_MELS: usize = 128;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct ParakeetPreprocessor {
    device: Device,
    _window: Vec<f32>,       // [win_length]
    padded_window: Vec<f32>, // [n_fft]
    fb: Vec<f32>,            // [n_mels * (n_fft/2+1)]
    n_mels: usize,
    n_freqs: usize,
}

impl ParakeetPreprocessor {
    pub fn load(vb: &VarBuilder, model_dir: &Path) -> Result<Self> {
        let device = vb.device().clone();
        let preproc_vb = vb.pp("preprocessor.featurizer");

        let window = match preproc_vb.get_unchecked_dtype("window", DType::F32) {
            Ok(window_tensor) => {
                let window = window_tensor.to_vec1::<f32>()?;
                if window.len() != WIN_LENGTH {
                    return Err(Error::ModelLoadError(format!(
                        "Unexpected Parakeet window length: expected {}, got {}",
                        WIN_LENGTH,
                        window.len()
                    )));
                }
                window
            }
            Err(err) if is_missing_tensor_error(&err) => hann_window(WIN_LENGTH),
            Err(err) => {
                return Err(Error::ModelLoadError(format!(
                    "Failed to load Parakeet preprocessor window tensor: {err}"
                )));
            }
        };

        let (fb, n_mels, n_freqs) = match preproc_vb.get_unchecked_dtype("fb", DType::F32) {
            Ok(fb_tensor) => {
                let (_, n_mels, n_freqs) = fb_tensor.dims3().map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Unexpected Parakeet filterbank tensor shape: {e}"
                    ))
                })?;
                let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
                (fb, n_mels, n_freqs)
            }
            Err(err) if is_missing_tensor_error(&err) => {
                let n_mels = infer_mel_bins_from_config(model_dir).unwrap_or(DEFAULT_N_MELS);
                (
                    mel_filterbank(SAMPLE_RATE, N_FFT, n_mels, 0.0, SAMPLE_RATE as f32 / 2.0),
                    n_mels,
                    N_FFT / 2 + 1,
                )
            }
            Err(err) => {
                return Err(Error::ModelLoadError(format!(
                    "Failed to load Parakeet preprocessor filterbank tensor: {err}"
                )));
            }
        };

        if n_freqs != (N_FFT / 2 + 1) {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Parakeet filterbank bins: expected {}, got {}",
                N_FFT / 2 + 1,
                n_freqs
            )));
        }
        if fb.len() != n_mels * n_freqs {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Parakeet filterbank length: expected {}, got {}",
                n_mels * n_freqs,
                fb.len()
            )));
        }

        let mut padded_window = vec![0f32; N_FFT];
        let offset = (N_FFT - WIN_LENGTH) / 2;
        padded_window[offset..offset + WIN_LENGTH].copy_from_slice(&window);

        Ok(Self {
            device,
            _window: window,
            padded_window,
            fb,
            n_mels,
            n_freqs,
        })
    }

    pub fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        // torch.stft(center=True, pad_mode="constant")
        let center_pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat(0.0).take(center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(center_pad));

        let frame_count = if padded.len() >= N_FFT {
            (padded.len() - N_FFT) / HOP_LENGTH + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];

        for frame_idx in 0..frame_count {
            let start = frame_idx * HOP_LENGTH;
            let slice = &padded[start..start + N_FFT];

            for i in 0..N_FFT {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }

            fft.process(&mut buffer);

            for k in 0..self.n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * self.n_freqs + k] = mag * mag; // mag_power=2
            }
        }

        // Mel projection and log.
        let mut mel = vec![0f32; self.n_mels * frame_count];
        for m in 0..self.n_mels {
            for t in 0..frame_count {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * self.n_freqs..(t + 1) * self.n_freqs];
                let fb_row = &self.fb[m * self.n_freqs..(m + 1) * self.n_freqs];
                for f in 0..self.n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frame_count + t] = (acc + LOG_GUARD).ln();
            }
        }

        // NeMo get_seq_len for center=True case: floor(seq_len / hop_length)
        let valid_frames = audio.len() / HOP_LENGTH;

        // normalize per_feature (along time)
        normalize_per_feature(
            &mut mel,
            self.n_mels,
            frame_count,
            valid_frames.min(frame_count),
        );

        // mask padded frames to zero
        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
                }
            }
        }

        let features = Tensor::from_vec(mel, (1, self.n_mels, frame_count), &self.device)?;

        Ok((features, valid_frames.min(frame_count)))
    }
}

fn is_missing_tensor_error(err: &candle_core::Error) -> bool {
    err.to_string().contains("cannot find tensor")
}

fn infer_mel_bins_from_config(model_dir: &Path) -> Option<usize> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(config_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    value
        .get("preprocessor")
        .and_then(|preprocessor| preprocessor.get("features"))
        .and_then(|features| features.as_u64())
        .map(|features| features as usize)
}

fn hann_window(win_length: usize) -> Vec<f32> {
    if win_length <= 1 {
        return vec![1.0; win_length.max(1)];
    }

    (0..win_length)
        .map(|i| {
            let x = (2.0 * std::f32::consts::PI * i as f32) / (win_length as f32 - 1.0);
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.min(nyquist).max(fmin));

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| nyquist * i as f32 / (n_freqs.saturating_sub(1).max(1)) as f32)
        .collect();

    let mut fb = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let lower_width = (center - left).max(1e-12);
        let upper_width = (right - center).max(1e-12);
        let enorm = if right > left {
            2.0 / (right - left)
        } else {
            0.0
        };

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_width;
            let upper = (right - freq) / upper_width;
            fb[m * n_freqs + k] = lower.min(upper).max(0.0) * enorm;
        }
    }

    fb
}

fn preemphasis(x: &mut [f32], preemph: f32) {
    if x.len() < 2 {
        return;
    }

    let mut prev = x[0];
    for sample in x.iter_mut().skip(1) {
        let cur = *sample;
        *sample = cur - preemph * prev;
        prev = cur;
    }
}

fn normalize_per_feature(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];

        let mean = row[..valid_frames].iter().copied().sum::<f32>() / valid_frames as f32;

        let var = if valid_frames > 1 {
            row[..valid_frames]
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / (valid_frames as f32 - 1.0)
        } else {
            0.0
        };

        let std = var.sqrt() + NORMALIZE_EPS;
        for v in row[..valid_frames].iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

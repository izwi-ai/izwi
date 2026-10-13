use std::ops::Range;
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

    /// A NeMo-default preprocessor (Hann window, Slaney filterbank) for tests
    /// that never read weights.
    #[cfg(test)]
    pub(super) fn for_tests(device: Device) -> Self {
        let window = hann_window(WIN_LENGTH);
        let mut padded_window = vec![0f32; N_FFT];
        let offset = (N_FFT - WIN_LENGTH) / 2;
        padded_window[offset..offset + WIN_LENGTH].copy_from_slice(&window);
        Self {
            device,
            _window: window,
            padded_window,
            fb: mel_filterbank(
                SAMPLE_RATE,
                N_FFT,
                DEFAULT_N_MELS,
                0.0,
                SAMPLE_RATE as f32 / 2.0,
            ),
            n_mels: DEFAULT_N_MELS,
            n_freqs: N_FFT / 2 + 1,
        }
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// Host log-mel spectrogram, before per_feature normalization.
    pub(super) fn compute_log_mel(&self, audio: &[f32]) -> Result<ParakeetLogMel> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        // torch.stft(center=True, pad_mode="constant")
        let center_pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat_n(0.0, center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat_n(0.0, center_pad));

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

        Ok(ParakeetLogMel {
            mel,
            n_mels: self.n_mels,
            frames: frame_count,
            valid_frames: valid_feature_frames(audio.len()).min(frame_count),
        })
    }
}

/// Row-major `[n_mels, frames]` log-mel spectrogram before per_feature
/// normalization. The STFT emits more frames than `valid_frames`; NeMo masks
/// those padding frames to zero after normalizing.
pub(super) struct ParakeetLogMel {
    mel: Vec<f32>,
    n_mels: usize,
    frames: usize,
    valid_frames: usize,
}

impl ParakeetLogMel {
    #[cfg(test)]
    pub(super) fn from_parts(
        mel: Vec<f32>,
        n_mels: usize,
        frames: usize,
        valid_frames: usize,
    ) -> Self {
        assert_eq!(mel.len(), n_mels * frames);
        assert!(valid_frames <= frames);
        Self {
            mel,
            n_mels,
            frames,
            valid_frames,
        }
    }

    pub(super) const fn frames(&self) -> usize {
        self.frames
    }

    pub(super) const fn valid_frames(&self) -> usize {
        self.valid_frames
    }

    /// Normalizes the `columns` slice per feature over its own valid frames,
    /// zeroes its padding frames and uploads it as `[1, n_mels, columns]`.
    /// The full range reproduces NeMo's whole-utterance normalization.
    pub(super) fn normalize_and_upload(
        &self,
        columns: Range<usize>,
        device: &Device,
    ) -> Result<Tensor> {
        let width = columns.len();
        if width == 0 || columns.end > self.frames {
            return Err(Error::InvalidInput(format!(
                "Parakeet feature columns {columns:?} are outside {} frames",
                self.frames
            )));
        }
        let mut window = Vec::with_capacity(self.n_mels * width);
        for row in self.mel.chunks_exact(self.frames) {
            window.extend_from_slice(&row[columns.clone()]);
        }
        let valid = self
            .valid_frames
            .min(columns.end)
            .saturating_sub(columns.start);

        // normalize per_feature (along time)
        normalize_per_feature(&mut window, self.n_mels, width, valid);

        // mask padded frames to zero
        if valid < width {
            for row in window.chunks_exact_mut(width) {
                row[valid..].fill(0.0);
            }
        }

        Tensor::from_vec(window, (1, self.n_mels, width), device).map_err(Error::from)
    }
}

/// NeMo `get_seq_len` for `center=True`: `floor(samples / hop_length)`. The
/// STFT emits one more (padding) frame, which is masked to zero.
pub(super) const fn valid_feature_frames(samples: usize) -> usize {
    samples / HOP_LENGTH
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(len: usize, mut seed: u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 22) as f32 - 2.0
            })
            .collect()
    }

    /// The pre-split preprocessor: normalize the whole utterance, then mask.
    fn whole_utterance_normalized(
        mel: &[f32],
        n_mels: usize,
        frames: usize,
        valid: usize,
    ) -> Vec<f32> {
        let mut mel = mel.to_vec();
        normalize_per_feature(&mut mel, n_mels, frames, valid.min(frames));
        if valid < frames {
            for m in 0..n_mels {
                for t in valid..frames {
                    mel[m * frames + t] = 0.0;
                }
            }
        }
        mel
    }

    fn columns(mel: &[f32], frames: usize, range: Range<usize>) -> Vec<f32> {
        mel.chunks_exact(frames)
            .flat_map(|row| row[range.clone()].iter().copied())
            .collect()
    }

    fn uploaded(log_mel: &ParakeetLogMel, range: Range<usize>) -> Vec<f32> {
        log_mel
            .normalize_and_upload(range, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn log_mel_frames_follow_nemo_seq_len() {
        let preprocessor = ParakeetPreprocessor::for_tests(Device::Cpu);
        let audio = pseudo_random(16_000 + 77, 5);
        let log_mel = preprocessor.compute_log_mel(&audio).unwrap();

        assert_eq!(log_mel.valid_frames(), 100);
        assert_eq!(log_mel.frames(), 101);
        assert_eq!(log_mel.mel.len(), DEFAULT_N_MELS * 101);
    }

    #[test]
    fn full_range_normalization_matches_whole_utterance_normalization() {
        let (n_mels, frames, valid) = (8, 37, 36);
        let mel = pseudo_random(n_mels * frames, 3);
        let log_mel = ParakeetLogMel::from_parts(mel.clone(), n_mels, frames, valid);

        assert_eq!(
            uploaded(&log_mel, 0..frames),
            whole_utterance_normalized(&mel, n_mels, frames, valid)
        );
    }

    #[test]
    fn window_normalization_uses_window_statistics_and_zeroes_padding() {
        let (n_mels, frames, valid) = (4, 40, 33);
        let mel = pseudo_random(n_mels * frames, 11);
        let log_mel = ParakeetLogMel::from_parts(mel.clone(), n_mels, frames, valid);

        // A window inside the audio is normalized over its own columns.
        assert_eq!(
            uploaded(&log_mel, 8..24),
            whole_utterance_normalized(&columns(&mel, frames, 8..24), n_mels, 16, 16)
        );
        // A window reaching the STFT padding normalizes its valid columns
        // and keeps the padding at zero.
        let tail = uploaded(&log_mel, 24..40);
        assert_eq!(
            tail,
            whole_utterance_normalized(&columns(&mel, frames, 24..40), n_mels, 16, 9)
        );
        assert!((0..n_mels)
            .flat_map(|m| &tail[m * 16 + 9..(m + 1) * 16])
            .all(|value| *value == 0.0));

        assert!(log_mel.normalize_and_upload(30..41, &Device::Cpu).is_err());
        assert!(log_mel.normalize_and_upload(5..5, &Device::Cpu).is_err());
    }
}

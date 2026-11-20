//! The log-mel front end and the 30 s window plan.
//!
//! The reference pads the waveform to a whole number of 30 s windows and
//! extracts Whisper log-mel features **over the whole padded waveform in one
//! call**: the `max - 8` clamp sees every window, and the STFT frames at a
//! window boundary read samples from both sides. Only then is the result cut
//! into one row of 3000 frames per window. A per-window extraction would
//! normalize a quiet window against its own maximum and differ from what the
//! model was trained on.
//!
//! Windows that are mostly (or wholly) padding are kept: every window
//! becomes 375 audio tokens whatever it holds.

use std::sync::Arc;

use candle_core::{Device, Tensor};

use super::config::SovFrontendConfig;
use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::error::{Error, Result};

/// A clip's normalized log-mel features, frame-major: window `w`, frame `t`,
/// bin `m` is `data[(w * frames + t) * n_mels + m]`.
#[derive(Clone)]
pub struct SovMel {
    pub windows: usize,
    pub frames_per_window: usize,
    pub n_mels: usize,
    data: Arc<Vec<f32>>,
}

impl std::fmt::Debug for SovMel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovMel")
            .field("windows", &self.windows)
            .field("frames_per_window", &self.frames_per_window)
            .finish()
    }
}

impl SovMel {
    pub fn host_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// The frame-major values of every window.
    pub fn values(&self) -> &[f32] {
        &self.data
    }

    /// Windows `first..first + count` as the encoder's input
    /// `[count, n_mels, frames]` F32 on `device`.
    pub fn window_tensor(&self, first: usize, count: usize, device: &Device) -> Result<Tensor> {
        if count == 0 || first + count > self.windows {
            return Err(Error::InvalidInput(format!(
                "windows {first}+{count} are outside a {}-window clip",
                self.windows
            )));
        }
        let (frames, bins) = (self.frames_per_window, self.n_mels);
        let mut out = vec![0f32; count * bins * frames];
        for window in 0..count {
            let source = &self.data[(first + window) * frames * bins..][..frames * bins];
            let target = &mut out[window * bins * frames..][..bins * frames];
            for (frame, values) in source.chunks_exact(bins).enumerate() {
                for (bin, value) in values.iter().enumerate() {
                    target[bin * frames + frame] = *value;
                }
            }
        }
        Ok(Tensor::from_vec(out, (count, bins, frames), device)?)
    }
}

pub struct SovMelFrontend {
    config: SovFrontendConfig,
    mel: MelSpectrogram,
}

impl SovMelFrontend {
    pub fn new(config: SovFrontendConfig) -> Result<Self> {
        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            win_length: None,
            hop_length: config.hop_length,
            n_mels: config.n_mels,
            f_min: 0.0,
            f_max: (config.sample_rate / 2) as f32,
            // Normalized below, after the extra STFT frame is dropped.
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })?;
        Ok(Self { config, mel })
    }

    pub fn config(&self) -> &SovFrontendConfig {
        &self.config
    }

    /// Features of `samples` (mono, at the front end's rate).
    pub fn compute(&self, samples: &[f32]) -> Result<SovMel> {
        let c = &self.config;
        let windows = samples.len().div_ceil(c.window_samples).max(1);
        let mut padded = samples.to_vec();
        padded.resize(windows * c.window_samples, 0.0);
        let (mut flat, frames) = self.mel.compute_flat(&padded)?;
        let wanted = windows * c.window_frames;
        if frames < wanted {
            return Err(Error::InferenceError(format!(
                "the STFT produced {frames} frames for {wanted}"
            )));
        }
        // `WhisperFeatureExtractor` drops the STFT's last frame.
        flat.truncate(wanted * c.n_mels);
        MelSpectrogram::whisper_normalize_flat(&mut flat);
        Ok(SovMel {
            windows,
            frames_per_window: c.window_frames,
            n_mels: c.n_mels,
            data: Arc::new(flat),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend() -> SovMelFrontend {
        SovMelFrontend::new(super::super::config::tests::published().frontend).unwrap()
    }

    fn tone(samples: usize, amplitude: f32) -> Vec<f32> {
        (0..samples)
            .map(|i| amplitude * (i as f32 * 0.05).sin())
            .collect()
    }

    #[test]
    fn a_clip_pads_to_whole_windows() {
        let frontend = frontend();
        for (samples, windows) in [
            (1, 1),
            (16_000, 1),
            (480_000, 1),
            (480_001, 2),
            (1_000_000, 3),
        ] {
            let mel = frontend.compute(&tone(samples, 0.3)).unwrap();
            assert_eq!(mel.windows, windows, "{samples} samples");
            assert_eq!(mel.values().len(), windows * 3000 * 128);
        }
    }

    #[test]
    fn normalization_spans_the_whole_clip() {
        // A loud first window and a near-silent second one: the second
        // window's floor is the clip's `max - 8`, not its own.
        let frontend = frontend();
        let mut samples = tone(480_000, 0.8);
        samples.extend(tone(240_000, 1e-4));
        let whole = frontend.compute(&samples).unwrap();
        let alone = frontend.compute(&samples[480_000..]).unwrap();
        let second = &whole.values()[3000 * 128..];
        let floor = whole.values().iter().cloned().fold(f32::INFINITY, f32::min);
        let alone_floor = alone.values().iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(second.iter().any(|v| (*v - floor).abs() < 1e-6));
        assert!(floor > alone_floor + 0.5, "{floor} vs {alone_floor}");
    }

    #[test]
    fn a_window_tensor_is_channel_major() {
        let frontend = frontend();
        let mel = frontend.compute(&tone(600_000, 0.3)).unwrap();
        let tensor = mel.window_tensor(1, 1, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 128, 3000]);
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (frame, bin) = (17, 5);
        assert_eq!(
            values[bin * 3000 + frame],
            mel.values()[(3000 + frame) * 128 + bin]
        );
        assert!(mel.window_tensor(1, 2, &Device::Cpu).is_err());
    }
}

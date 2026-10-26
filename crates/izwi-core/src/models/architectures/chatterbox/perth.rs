//! PerTh (PERceptual THreshold) watermarking, ported natively: the implicit
//! watermarker Chatterbox marks every utterance with (resemble-ai/Perth at
//! ff1c8ac5, MIT; checkpoint `perth_net_250000.pth.tar`).
//!
//! At 32 kHz: a centred STFT (n_fft 2048, hop 320, periodic Hann), the
//! magnitude in dB normalised against a −180 dB floor with 15 dB of
//! headroom, a residual that seven 1-D convolutions write into the lowest
//! 128 bins (up to 2 kHz) of every frame holding at least 5% of the loudest
//! frame's level, and the inverse STFT with the original phase. Only the
//! encoder is ported; upstream's decoder is the detector the tests use.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Module, VarBuilder};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::reference_dsp::{hann_periodic, reflect_pad, sinc_resample};
use super::weights::{conv1d, conv_cfg, host_f32};
use crate::error::{Error, Result};

/// The checkpoint, read from the model directory.
pub const PERTH_CHECKPOINT_FILE: &str = "perth_net_250000.pth.tar";

const SAMPLE_RATE: u32 = 32_000;
const N_FFT: usize = 2048;
const HOP: usize = 320;
const BINS: usize = N_FFT / 2 + 1;
/// Bins below 2 kHz: round(1025 × 2000 / 16000).
const SUBBAND: usize = 128;
const HIDDEN: usize = 256;
/// 20·log10(1e-9), the magnitude floor.
const MIN_DB: f64 = -180.0;
const HEADROOM_DB: f64 = 15.0;
/// A frame is marked when its level is above this share of the loudest.
const MASK_SHARE: f64 = 0.05;
const LEAKY_SLOPE: f64 = 0.01;

pub struct PerthWatermarker {
    /// Each convolution and whether a LeakyReLU follows it.
    layers: Vec<(Conv1d, bool)>,
    device: Device,
}

impl PerthWatermarker {
    pub fn load(checkpoint: &Path, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_pth_with_state(checkpoint, DType::F32, "model", device).map_err(
            |error| {
                Error::ModelLoadError(format!(
                    "Cannot read the PerTh watermarker {}: {error}",
                    checkpoint.display()
                ))
            },
        )?;
        let encoder = vb.pp("encoder.layers");
        let kernels = [1, 7, 7, 7, 7, 7, 1];
        let layers = kernels
            .iter()
            .enumerate()
            .map(|(index, &kernel)| {
                let conv = conv1d(
                    &encoder.pp(format!("{index}.conv")),
                    conv_cfg((kernel - 1) / 2, 1, 1),
                )?;
                Ok((conv, index + 1 < kernels.len()))
            })
            .collect::<Result<Vec<_>>>()?;
        let expected = [
            (HIDDEN, SUBBAND),
            (HIDDEN, HIDDEN),
            (HIDDEN, HIDDEN),
            (HIDDEN, HIDDEN),
            (HIDDEN, HIDDEN),
            (HIDDEN, HIDDEN),
            (SUBBAND, HIDDEN),
        ];
        for ((conv, _), (out_channels, in_channels)) in layers.iter().zip(expected) {
            let (out, input, _) = conv.weight().dims3()?;
            if (out, input) != (out_channels, in_channels) {
                return Err(Error::ModelLoadError(format!(
                    "PerTh encoder layer is {out}×{input}, expected {out_channels}×{in_channels}"
                )));
            }
        }
        Ok(Self {
            layers,
            device: device.clone(),
        })
    }

    /// Watermark mono `samples` at `sample_rate`, through 32 kHz when the
    /// rate differs. Like upstream, the result ends at the last full hop at
    /// 32 kHz, up to 10 ms short of the input.
    pub fn apply(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let resampled = sample_rate != SAMPLE_RATE;
        let signal = if resampled {
            sinc_resample(samples, sample_rate, SAMPLE_RATE)
        } else {
            samples.to_vec()
        };
        let (mut magspec, phase, frames) = stft_magphase(&signal)?;
        let mask = frame_mask(&magspec, frames);
        let residual = self.residual(&magspec, frames)?;
        for bin in 0..SUBBAND {
            for frame in (0..frames).filter(|&frame| mask[frame]) {
                magspec[bin * frames + frame] += residual[bin * frames + frame];
            }
        }
        let watermarked = istft(&magspec, &phase, frames);
        Ok(if resampled {
            sinc_resample(&watermarked, SAMPLE_RATE, sample_rate)
        } else {
            watermarked
        })
    }

    #[cfg(test)]
    pub(crate) fn residual_for_tests(&self, magspec: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.residual(magspec, frames)
    }

    /// The encoder's residual for the lowest bins, [SUBBAND × frames].
    fn residual(&self, magspec: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut x = Tensor::from_slice(
            &magspec[..SUBBAND * frames],
            (1, SUBBAND, frames),
            &self.device,
        )?;
        for (conv, activation) in &self.layers {
            x = conv.forward(&x)?;
            if *activation {
                x = x.maximum(&x.affine(LEAKY_SLOPE, 0.0)?)?;
            }
        }
        host_f32(&x)
    }
}

/// The normalised dB magnitude and the phase of the centred STFT, each
/// [BINS × frames] row-major (torchaudio's `Spectrogram`, `power=None`).
pub(crate) fn stft_magphase(signal: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    if signal.len() <= N_FFT / 2 {
        return Err(Error::InvalidInput(
            "audio is too short for the PerTh watermark".into(),
        ));
    }
    let padded = reflect_pad(signal, N_FFT / 2);
    let frames = 1 + signal.len() / HOP;
    let window = hann_periodic(N_FFT);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let mut magspec = vec![0f32; BINS * frames];
    let mut phase = vec![0f32; BINS * frames];
    let mut buffer = vec![Complex::new(0.0, 0.0); N_FFT];
    for frame in 0..frames {
        let start = frame * HOP;
        for (n, value) in buffer.iter_mut().enumerate() {
            *value = Complex::new(f64::from(padded[start + n]) * window[n], 0.0);
        }
        fft.process(&mut buffer);
        for (bin, value) in buffer.iter().take(BINS).enumerate() {
            let db = 20.0 * value.norm().max(1e-9).log10();
            magspec[bin * frames + frame] = ((db - MIN_DB) / (HEADROOM_DB - MIN_DB)) as f32;
            phase[bin * frames + frame] = value.im.atan2(value.re) as f32;
        }
    }
    Ok((magspec, phase, frames))
}

/// Frames whose summed normalised level exceeds 5% of the loudest frame's.
pub(crate) fn frame_mask(magspec: &[f32], frames: usize) -> Vec<bool> {
    let levels = (0..frames)
        .map(|frame| {
            (0..BINS)
                .map(|bin| f64::from(magspec[bin * frames + frame]))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let threshold = levels.iter().copied().fold(f64::MIN, f64::max) * MASK_SHARE;
    levels.into_iter().map(|level| level > threshold).collect()
}

/// The inverse of [`stft_magphase`] (torch's `istft`, centred, no length):
/// window-weighted overlap-add over the squared-window envelope, trimmed to
/// `HOP × (frames − 1)` samples.
fn istft(magspec: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    let window = hann_periodic(N_FFT);
    let ifft = FftPlanner::<f64>::new().plan_fft_inverse(N_FFT);
    let total = N_FFT + HOP * (frames - 1);
    let mut sum = vec![0f64; total];
    let mut envelope = vec![0f64; total];
    let mut buffer = vec![Complex::new(0.0, 0.0); N_FFT];
    for frame in 0..frames {
        for bin in 0..BINS {
            let db = f64::from(magspec[bin * frames + frame]) * (HEADROOM_DB - MIN_DB) + MIN_DB;
            let magnitude = 10f64.powf((db / 20.0).min(10.0));
            buffer[bin] = Complex::from_polar(magnitude, f64::from(phase[bin * frames + frame]));
        }
        // A real inverse transform reads only the real part of the DC and
        // Nyquist bins.
        buffer[0].im = 0.0;
        buffer[N_FFT / 2].im = 0.0;
        for bin in 1..N_FFT / 2 {
            buffer[N_FFT - bin] = buffer[bin].conj();
        }
        ifft.process(&mut buffer);
        let start = frame * HOP;
        for n in 0..N_FFT {
            sum[start + n] += buffer[n].re / N_FFT as f64 * window[n];
            envelope[start + n] += window[n] * window[n];
        }
    }
    let start = N_FFT / 2;
    (start..start + HOP * (frames - 1))
        .map(|i| {
            if envelope[i] > 1e-11 {
                (sum[i] / envelope[i]) as f32
            } else {
                0.0
            }
        })
        .collect()
}

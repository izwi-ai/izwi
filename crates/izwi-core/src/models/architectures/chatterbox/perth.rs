//! PerTh (PERceptual THreshold) watermarking, ported natively: the implicit
//! watermarker Chatterbox marks every utterance with (resemble-ai/Perth at
//! ff1c8ac5, MIT; checkpoint `perth_net_250000.pth.tar`).
//!
//! At 32 kHz: a centred STFT (n_fft 2048, hop 320, periodic Hann), the
//! magnitude in dB normalised against a −180 dB floor with 15 dB of
//! headroom, a residual that seven 1-D convolutions write into the lowest
//! 128 bins (up to 2 kHz) of every frame holding at least 5% of the loudest
//! frame's level, and the inverse STFT with the original phase. A streamed
//! utterance is watermarked piece by piece over overlapping segments
//! ([`PerthStream`]). The decoder, upstream's detector, is ported for tests.

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
/// Context each side of a streamed segment, in seconds: more than the
/// resamplers' filters, the STFT window (32 ms) and the encoder's ±15-frame
/// reach (150 ms) together need, so a settled sample is marked as it would
/// be in the whole utterance (but for the loudness threshold).
const STREAM_CONTEXT_SECONDS: f64 = 0.25;

/// A streamed utterance's watermark state: what has arrived, what has been
/// emitted, and the loudest frame so far, which sets the mask threshold.
pub struct PerthStream {
    sample_rate: u32,
    /// Unmarked samples from `offset` on.
    history: Vec<f32>,
    offset: usize,
    emitted: usize,
    loudest: f64,
}

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
        Ok(Self {
            layers: conv_stack(&vb.pp("encoder.layers"), SUBBAND)?,
            device: device.clone(),
        })
    }

    /// A stream for an utterance at `sample_rate`, watermarked as it arrives.
    pub fn stream(&self, sample_rate: u32) -> PerthStream {
        PerthStream {
            sample_rate,
            history: Vec::new(),
            offset: 0,
            emitted: 0,
            loudest: 0.0,
        }
    }

    /// Append `samples` to `stream` and return the newly settled
    /// watermarked samples: all but the last [`STREAM_CONTEXT_SECONDS`],
    /// whose right context has not arrived, unless `last`. Each segment
    /// carries the same context on its left, and the mask threshold follows
    /// the loudest frame so far rather than the whole utterance's.
    pub fn push(&self, stream: &mut PerthStream, samples: &[f32], last: bool) -> Result<Vec<f32>> {
        stream.history.extend_from_slice(samples);
        let context = (STREAM_CONTEXT_SECONDS * f64::from(stream.sample_rate)) as usize;
        let total = stream.offset + stream.history.len();
        let end = if last {
            total
        } else {
            total.saturating_sub(context)
        };
        if end <= stream.emitted {
            return Ok(Vec::new());
        }
        let start = stream.emitted.saturating_sub(context).max(stream.offset);
        let (marked, loudest) = self.apply_with_floor(
            &stream.history[start - stream.offset..],
            stream.sample_rate,
            stream.loudest,
        )?;
        stream.loudest = loudest;
        // The inverse STFT stops at the last full hop, so the very end of
        // the utterance can come back a few milliseconds short.
        let to = (end - start).min(marked.len());
        let from = stream.emitted - start;
        let out = marked
            .get(from..to)
            .map(<[f32]>::to_vec)
            .unwrap_or_default();
        stream.emitted = if last { end } else { start + to.max(from) };
        let keep_from = stream.emitted.saturating_sub(context).max(stream.offset);
        stream.history.drain(..keep_from - stream.offset);
        stream.offset = keep_from;
        Ok(out)
    }

    /// Watermark mono `samples` at `sample_rate`, through 32 kHz when the
    /// rate differs. Like upstream, the result ends at the last full hop at
    /// 32 kHz, up to 10 ms short of the input.
    pub fn apply(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        self.apply_with_floor(samples, sample_rate, 0.0)
            .map(|(marked, _)| marked)
    }

    /// [`Self::apply`], with the mask threshold taken from the louder of
    /// `level_floor` and this signal's loudest frame; also returns that
    /// level.
    fn apply_with_floor(
        &self,
        samples: &[f32],
        sample_rate: u32,
        level_floor: f64,
    ) -> Result<(Vec<f32>, f64)> {
        let resampled = sample_rate != SAMPLE_RATE;
        let signal = if resampled {
            sinc_resample(samples, sample_rate, SAMPLE_RATE)
        } else {
            samples.to_vec()
        };
        let (mut magspec, phase, frames) = stft_magphase(&signal)?;
        let (mask, loudest) = frame_mask_with_floor(&magspec, frames, level_floor);
        let residual = self.residual(&magspec, frames)?;
        for bin in 0..SUBBAND {
            for frame in (0..frames).filter(|&frame| mask[frame]) {
                magspec[bin * frames + frame] += residual[bin * frames + frame];
            }
        }
        let watermarked = istft(&magspec, &phase, frames);
        let watermarked = if resampled {
            sinc_resample(&watermarked, SAMPLE_RATE, sample_rate)
        } else {
            watermarked
        };
        Ok((watermarked, loudest))
    }

    #[cfg(test)]
    pub(crate) fn residual_for_tests(&self, magspec: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.residual(magspec, frames)
    }

    /// The encoder's residual for the lowest bins, [SUBBAND × frames].
    fn residual(&self, magspec: &[f32], frames: usize) -> Result<Vec<f32>> {
        let x = Tensor::from_slice(
            &magspec[..SUBBAND * frames],
            (1, SUBBAND, frames),
            &self.device,
        )?;
        host_f32(&run_stack(&self.layers, x)?)
    }
}

/// PerTh's seven-convolution stack (`Conv`: kernel 1, five of 7, then 1,
/// LeakyReLU after all but the last) from `vb`, `inputs` channels wide.
fn conv_stack(vb: &VarBuilder, inputs: usize) -> Result<Vec<(Conv1d, bool)>> {
    let kernels = [1, 7, 7, 7, 7, 7, 1];
    let layers = kernels
        .iter()
        .enumerate()
        .map(|(index, &kernel)| {
            let conv = conv1d(
                &vb.pp(format!("{index}.conv")),
                conv_cfg((kernel - 1) / 2, 1, 1),
            )?;
            Ok((conv, index + 1 < kernels.len()))
        })
        .collect::<Result<Vec<_>>>()?;
    let (_, first_inputs, _) = layers[0].0.weight().dims3()?;
    let (_, hidden, _) = layers[1].0.weight().dims3()?;
    if first_inputs != inputs || hidden != HIDDEN {
        return Err(Error::ModelLoadError(format!(
            "PerTh stack takes {first_inputs} channels with {hidden} hidden, expected {inputs} and {HIDDEN}"
        )));
    }
    Ok(layers)
}

fn run_stack(layers: &[(Conv1d, bool)], mut x: Tensor) -> Result<Tensor> {
    for (conv, activation) in layers {
        x = conv.forward(&x)?;
        if *activation {
            x = x.maximum(&x.affine(LEAKY_SLOPE, 0.0)?)?;
        }
    }
    Ok(x)
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
    frame_mask_with_floor(magspec, frames, f64::MIN).0
}

/// [`frame_mask`] against the louder of `floor` and the loudest frame here;
/// also returns that level.
fn frame_mask_with_floor(magspec: &[f32], frames: usize, floor: f64) -> (Vec<bool>, f64) {
    let levels = (0..frames)
        .map(|frame| {
            (0..BINS)
                .map(|bin| f64::from(magspec[bin * frames + frame]))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let loudest = levels.iter().copied().fold(floor, f64::max);
    let threshold = loudest * MASK_SHARE;
    (
        levels.into_iter().map(|level| level > threshold).collect(),
        loudest,
    )
}

/// Upstream's detector (`Decoder`): the same stack three times, over the
/// low bins resampled in time to ×1.25, ×1 and ×0.75, each giving a
/// masked mean score and attention; the attention-weighted score, clipped
/// to [0, 1], is about 1 for a watermarked signal and 0 for a clean one.
#[cfg(test)]
pub(crate) struct PerthDetector {
    branches: [(f64, Vec<(Conv1d, bool)>); 3],
    device: Device,
}

#[cfg(test)]
impl PerthDetector {
    pub(crate) fn load(checkpoint: &Path, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_pth_with_state(checkpoint, DType::F32, "model", device)?;
        let decoder = vb.pp("decoder");
        let stack = |name: &str| conv_stack(&decoder.pp(name), SUBBAND);
        Ok(Self {
            branches: [
                (1.25, stack("slow_layers")?),
                (1.0, stack("normal_layers")?),
                (0.75, stack("fast_layers")?),
            ],
            device: device.clone(),
        })
    }

    /// The watermark score of `samples` at `sample_rate`.
    pub(crate) fn detect(&self, samples: &[f32], sample_rate: u32) -> Result<f32> {
        let signal = if sample_rate == SAMPLE_RATE {
            samples.to_vec()
        } else {
            sinc_resample(samples, sample_rate, SAMPLE_RATE)
        };
        let (magspec, _, frames) = stft_magphase(&signal)?;
        self.detect_magspec(&magspec, frames)
    }

    pub(crate) fn detect_magspec(&self, magspec: &[f32], frames: usize) -> Result<f32> {
        let mask = frame_mask(magspec, frames);
        let mut scores = Vec::new();
        let mut attentions = Vec::new();
        for (scale, layers) in &self.branches {
            // F.interpolate(linear, align_corners=True) along time.
            let length = (frames as f64 * scale) as usize;
            let mut subband = vec![0f32; SUBBAND * length];
            for t in 0..length {
                let position = if length > 1 {
                    t as f64 * (frames - 1) as f64 / (length - 1) as f64
                } else {
                    0.0
                };
                let left = (position.floor() as usize).min(frames - 1);
                let right = (left + 1).min(frames - 1);
                let weight = (position - left as f64) as f32;
                for bin in 0..SUBBAND {
                    let a = magspec[bin * frames + left];
                    let b = magspec[bin * frames + right];
                    subband[bin * length + t] = a + (b - a) * weight;
                }
            }
            let x = Tensor::from_vec(subband, (1, SUBBAND, length), &self.device)?;
            let out = host_f32(&run_stack(layers, x)?)?;
            // F.interpolate(nearest) of the mask to the branch length.
            let (mut sum_attention, mut sum_score, mut count) = (0f64, 0f64, 0f64);
            for t in 0..length {
                let source =
                    ((t as f64 * frames as f64 / length as f64).floor() as usize).min(frames - 1);
                if mask[source] {
                    sum_attention += f64::from(out[t]);
                    sum_score += f64::from(out[length + t]);
                    count += 1.0;
                }
            }
            attentions.push(sum_attention / count);
            scores.push(sum_score / count);
        }
        let peak = attentions.iter().copied().fold(f64::MIN, f64::max);
        let weights = attentions
            .iter()
            .map(|a| (a - peak).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let score = scores
            .iter()
            .zip(&weights)
            .map(|(score, weight)| score * weight / total)
            .sum::<f64>();
        Ok(score.clamp(0.0, 1.0) as f32)
    }
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

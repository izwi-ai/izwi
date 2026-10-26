//! HiFT: S3Gen's neural-source-filter iSTFT vocoder (`s3gen/hifigan.py`,
//! `s3gen/f0_predictor.py`; Apache-2.0, derived from CosyVoice).
//!
//! A conv F0 predictor reads the mels; the harmonic source (nine harmonics,
//! random start phases, additive noise, a learned mix) is synthesised on the
//! host at 24 kHz from the request's seeded streams, exactly as upstream's
//! `SineGen` computes it (phase accumulated in F64 like torch's CPU
//! `cumsum`, then rounded to F32). The generator upsamples the mels ×8 ×5
//! ×3, adds the source's STFT at each scale, and predicts a 16-point
//! magnitude/phase spectrum that an iSTFT (hop 4) turns into samples. The
//! STFT and iSTFT are fixed-basis convolutions, so they run on every
//! backend.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder};

use super::config::HiftConfig;
use super::noise::{NoiseStream, SplitMix64};
use super::weights::{
    conv1d, conv_cfg, host_f32, weight_norm_conv1d, weight_norm_conv_transpose1d,
};
use crate::error::{Error, Result};

pub struct HiftVocoder {
    cfg: HiftConfig,
    dtype: DType,
    device: Device,
    f0_convs: Vec<Conv1d>,
    f0_classifier: Linear,
    /// The source module's harmonic mix, `l_linear` (9 → 1).
    source_weights: Vec<f32>,
    source_bias: f32,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    /// STFT analysis kernel [2·bins, 1, n_fft]: windowed cos, then −sin.
    stft_kernel: Tensor,
    /// iSTFT synthesis kernel [2·bins, 1, n_fft] for `conv_transpose1d`.
    istft_kernel: Tensor,
    window_sq: Vec<f32>,
}

struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    /// Snake alphas, [1, channels, 1] each.
    alphas1: Vec<Tensor>,
    alphas2: Vec<Tensor>,
}

/// torch's `get_window("hann", n, fftbins=True)`: the periodic Hann window.
fn periodic_hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

impl HiftVocoder {
    pub fn load(cfg: HiftConfig, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let f0 = vb.pp("f0_predictor");
        let f0_convs = (0..5)
            .map(|i| weight_norm_conv1d(&f0.pp(format!("condnet.{}", 2 * i)), conv_cfg(1, 1, 1)))
            .collect::<Result<Vec<_>>>()?;
        let f0_classifier = candle_nn::linear(512, 1, f0.pp("classifier"))?;

        let source = vb.pp("m_source.l_linear");
        let source_weights = host_f32(&source.get((1, cfg.harmonics + 1), "weight")?)?;
        let source_bias = host_f32(&source.get(1, "bias")?)?[0];

        let conv_pre = weight_norm_conv1d(&vb.pp("conv_pre"), conv_cfg(3, 1, 1))?;
        let ups = cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes)
            .enumerate()
            .map(|(i, (&rate, kernel))| {
                weight_norm_conv_transpose1d(
                    &vb.pp(format!("ups.{i}")),
                    ConvTranspose1dConfig {
                        padding: (kernel - rate) / 2,
                        output_padding: 0,
                        stride: rate,
                        dilation: 1,
                        groups: 1,
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Source downsampling to each stage's rate: 15, 3, 1 for [8, 5, 3].
        let rates = &cfg.upsample_rates;
        let down_rates = [rates[1] * rates[2], rates[2], 1];
        let source_downs = down_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                let cfg = if rate == 1 {
                    conv_cfg(0, 1, 1)
                } else {
                    conv_cfg(rate / 2, rate, 1)
                };
                conv1d(&vb.pp(format!("source_downs.{i}")), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let channels = |i: usize| cfg.base_channels >> (i + 1);
        let source_resblocks = cfg
            .source_resblock_kernel_sizes
            .iter()
            .enumerate()
            .map(|(i, &kernel)| {
                ResBlock::load(
                    vb.pp(format!("source_resblocks.{i}")),
                    channels(i),
                    kernel,
                    &cfg.resblock_dilations,
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut resblocks = Vec::with_capacity(9);
        for stage in 0..cfg.upsample_rates.len() {
            for (j, &kernel) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let index = stage * cfg.resblock_kernel_sizes.len() + j;
                resblocks.push(ResBlock::load(
                    vb.pp(format!("resblocks.{index}")),
                    channels(stage),
                    kernel,
                    &cfg.resblock_dilations,
                    dtype,
                )?);
            }
        }
        let conv_post = weight_norm_conv1d(&vb.pp("conv_post"), conv_cfg(3, 1, 1))?;

        let n_fft = cfg.istft_n_fft;
        let bins = n_fft / 2 + 1;
        let window = periodic_hann(n_fft);
        let mut analysis = vec![0f32; 2 * bins * n_fft];
        let mut synthesis = vec![0f32; 2 * bins * n_fft];
        for k in 0..bins {
            // irfft weights: DC and Nyquist once, the others twice.
            let weight = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 } / n_fft as f64;
            for n in 0..n_fft {
                let angle = 2.0 * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
                analysis[k * n_fft + n] = (window[n] * angle.cos()) as f32;
                analysis[(bins + k) * n_fft + n] = (-window[n] * angle.sin()) as f32;
                synthesis[k * n_fft + n] = (weight * angle.cos() * window[n]) as f32;
                synthesis[(bins + k) * n_fft + n] = (-weight * angle.sin() * window[n]) as f32;
            }
        }
        let kernel = |values: Vec<f32>| -> Result<Tensor> {
            Ok(
                Tensor::from_vec(values, (2 * bins, 1, n_fft), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(&device)?,
            )
        };
        Ok(Self {
            stft_kernel: kernel(analysis)?,
            istft_kernel: kernel(synthesis)?,
            window_sq: window.iter().map(|w| (w * w) as f32).collect(),
            f0_convs,
            f0_classifier,
            source_weights,
            source_bias,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            cfg,
            dtype,
            device,
        })
    }

    pub fn config(&self) -> &HiftConfig {
        &self.cfg
    }

    /// F0 in Hz per mel frame, for `mel` [80, frames].
    pub fn predict_f0(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let mut h = mel.unsqueeze(0)?.to_dtype(self.dtype)?;
        for conv in &self.f0_convs {
            h = conv.forward(&h)?.elu(1.0)?;
        }
        let f0 = self
            .f0_classifier
            .forward(&h.squeeze(0)?.t()?)?
            .abs()?
            .squeeze(1)?;
        host_f32(&f0)
    }

    /// The harmonic source at the output rate for per-frame `f0`, from the
    /// request's phase and noise streams.
    pub fn harmonic_source(&self, f0: &[f32], seed: u64) -> Vec<f32> {
        let cfg = &self.cfg;
        let factor = cfg.upsample_factor();
        let len = f0.len() * factor;
        let harmonics = cfg.harmonics + 1;
        let sample_rate = super::config::S3GEN_SAMPLE_RATE as f32;
        let mut phases = SplitMix64::for_stream(seed, NoiseStream::HarmonicPhase)
            .uniforms(harmonics)
            .into_iter()
            .map(|u| -std::f32::consts::PI + 2.0 * std::f32::consts::PI * u)
            .collect::<Vec<_>>();
        phases[0] = 0.0;
        let noise = SplitMix64::for_stream(seed, NoiseStream::SourceNoise)
            .standard_normals(harmonics * len);
        let unvoiced_amp = cfg.nsf_alpha / 3.0;
        let two_pi = 2.0 * std::f64::consts::PI as f32;
        let mut mixed = vec![self.source_bias; len];
        for h in 0..harmonics {
            let weight = self.source_weights[h];
            let multiple = (h + 1) as f32;
            let mut cumulative = 0f64;
            for (n, slot) in mixed.iter_mut().enumerate() {
                let f0 = f0[n / factor];
                cumulative += f64::from(f0 * multiple / sample_rate);
                let theta = two_pi * (cumulative as f32).rem_euclid(1.0);
                let voiced = f0 > cfg.nsf_voiced_threshold;
                let noise_amp = if voiced { cfg.nsf_sigma } else { unvoiced_amp };
                let sine = if voiced {
                    cfg.nsf_alpha * (theta + phases[h]).sin()
                } else {
                    0.0
                };
                *slot += weight * (sine + noise_amp * noise[h * len + n]);
            }
        }
        for value in &mut mixed {
            *value = value.tanh();
        }
        mixed
    }

    /// Samples for `mel` [80, frames] with the request seed.
    pub fn vocode(&self, mel: &Tensor, seed: u64) -> Result<Vec<f32>> {
        let f0 = self.predict_f0(mel)?;
        let source = self.harmonic_source(&f0, seed);
        self.decode(mel, &source)
    }

    /// The generator and iSTFT for `mel` [80, frames] and its `source`.
    pub fn decode(&self, mel: &Tensor, source: &[f32]) -> Result<Vec<f32>> {
        let n_fft = self.cfg.istft_n_fft;
        let hop = self.cfg.istft_hop;
        let pad = n_fft / 2;
        if source.len() <= pad {
            return Err(Error::InvalidInput(
                "Chatterbox vocoder source is too short".into(),
            ));
        }
        // torch.stft(center=True): reflect-pad n_fft / 2 on both sides.
        let mut padded = Vec::with_capacity(source.len() + 2 * pad);
        padded.extend((1..=pad).rev().map(|i| source[i]));
        padded.extend_from_slice(source);
        padded.extend((0..pad).map(|i| source[source.len() - 2 - i]));
        let padded = Tensor::from_vec(padded, (1, 1, source.len() + 2 * pad), &Device::Cpu)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        let source_stft = padded.conv1d(&self.stft_kernel, 0, hop, 1, 1)?;

        let mut x = self
            .conv_pre
            .forward(&mel.unsqueeze(0)?.to_dtype(self.dtype)?)?;
        let kernels = self.cfg.resblock_kernel_sizes.len();
        let stages = self.ups.len();
        for stage in 0..stages {
            x = candle_nn::ops::leaky_relu(&x, self.cfg.lrelu_slope)?;
            x = self.ups[stage].forward(&x)?;
            if stage == stages - 1 {
                // ReflectionPad1d((1, 0)).
                x = Tensor::cat(&[&x.narrow(2, 1, 1)?, &x], 2)?;
            }
            let si = self.source_downs[stage].forward(&source_stft)?;
            let si = self.source_resblocks[stage].forward(&si)?;
            x = x.add(&si)?;
            let mut sum = self.resblocks[stage * kernels].forward(&x)?;
            for j in 1..kernels {
                sum = sum.add(&self.resblocks[stage * kernels + j].forward(&x)?)?;
            }
            x = sum.affine(1.0 / kernels as f64, 0.0)?;
        }
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?;
        let bins = n_fft / 2 + 1;
        let magnitude = x.narrow(1, 0, bins)?.exp()?.clamp(f32::MIN, 100f32)?;
        let phase = x.narrow(1, bins, bins)?.sin()?;
        let spectrum = Tensor::cat(
            &[magnitude.mul(&phase.cos()?)?, magnitude.mul(&phase.sin()?)?],
            1,
        )?;
        let frames = spectrum.dim(2)?;
        let overlapped = spectrum
            .conv_transpose1d(&self.istft_kernel, 0, 0, hop, 1, 1)?
            .flatten_all()?;
        let overlapped = host_f32(&overlapped)?;

        // Divide by the summed squared window, then drop the centre padding.
        let total = (frames - 1) * hop + n_fft;
        let mut envelope = vec![0f32; total];
        for frame in 0..frames {
            for (n, w) in self.window_sq.iter().enumerate() {
                envelope[frame * hop + n] += w;
            }
        }
        let length = (frames - 1) * hop;
        let limit = self.cfg.audio_limit;
        let samples = (pad..pad + length)
            .map(|i| {
                if envelope[i] <= 1e-11 {
                    0.0
                } else {
                    (overlapped[i] / envelope[i]).clamp(-limit, limit)
                }
            })
            .collect::<Vec<_>>();
        Ok(samples)
    }
}

impl ResBlock {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel: usize,
        dilations: &[usize; 3],
        dtype: DType,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut alphas1 = Vec::with_capacity(3);
        let mut alphas2 = Vec::with_capacity(3);
        for (i, &dilation) in dilations.iter().enumerate() {
            convs1.push(weight_norm_conv1d(
                &vb.pp(format!("convs1.{i}")),
                conv_cfg((kernel * dilation - dilation) / 2, 1, dilation),
            )?);
            convs2.push(weight_norm_conv1d(
                &vb.pp(format!("convs2.{i}")),
                conv_cfg((kernel - 1) / 2, 1, 1),
            )?);
            let alpha = |name: String| -> Result<Tensor> {
                Ok(vb
                    .get_unchecked_dtype(&name, dtype)?
                    .reshape((1, channels, 1))?)
            };
            alphas1.push(alpha(format!("activations1.{i}.alpha"))?);
            alphas2.push(alpha(format!("activations2.{i}.alpha"))?);
        }
        Ok(Self {
            convs1,
            convs2,
            alphas1,
            alphas2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = snake(&x, &self.alphas1[i])?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = snake(&xt, &self.alphas2[i])?;
            let xt = self.convs2[i].forward(&xt)?;
            x = xt.add(&x)?;
        }
        Ok(x)
    }
}

/// Snake: `x + sin²(αx) / (α + 1e-9)`.
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let sin = x.broadcast_mul(alpha)?.sin()?;
    let inverse = alpha.affine(1.0, 1e-9)?.recip()?;
    Ok(x.add(&sin.sqr()?.broadcast_mul(&inverse)?)?)
}

/// Silence the first 20 ms and fade in over the next 20 ms
/// (`S3Token2Wav.trim_fade`), against spillover from the reference.
pub fn apply_trim_fade(samples: &mut [f32]) {
    let n_trim = super::config::S3GEN_SAMPLE_RATE as usize / 50;
    for (i, sample) in samples.iter_mut().take(2 * n_trim).enumerate() {
        let gain = if i < n_trim {
            0.0
        } else {
            // torch.linspace(π, 0, n_trim) in F32, then (cos + 1) / 2.
            let k = (i - n_trim) as f64;
            let angle = std::f64::consts::PI * (1.0 - k / (n_trim - 1) as f64);
            ((angle.cos() + 1.0) / 2.0) as f32
        };
        *sample *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_matches_scipy() {
        let w = periodic_hann(16);
        assert_eq!(w[0], 0.0);
        assert!((w[8] - 1.0).abs() < 1e-12);
        assert!((w[4] - 0.5).abs() < 1e-12);
        assert!((w[1] - w[15]).abs() < 1e-12);
    }

    #[test]
    fn trim_fade_silences_then_ramps() {
        let mut samples = vec![1.0f32; 1200];
        apply_trim_fade(&mut samples);
        assert!(samples[..480].iter().all(|&s| s == 0.0));
        assert_eq!(samples[480], 0.0);
        assert!((samples[959] - 1.0).abs() < 1e-6);
        assert!(samples[700] > 0.0 && samples[700] < 1.0);
        assert!(samples[960..].iter().all(|&s| s == 1.0));
    }
}

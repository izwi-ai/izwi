//! SpeechBrain's ECAPA-TDNN speaker encoder (`speechbrain/spkrec-ecapa-voxceleb`),
//! the voice-clone speaker embedding Maya2 conditions on.
//!
//! Upstream (`maya3/api.py`, class `Ecapa`; speechbrain 1.1.2, scipy ≥ 1.13)
//! embeds a 24 kHz mono reference clip as `normalize(ECAPA_TDNN(norm(fbank(
//! resample_poly(w24, 2, 3)))))`, with `Fbank(n_mels=80)` and
//! `InputNormalization("sentence", std_norm=False)`. Each stage is ported
//! here, on the host or on Candle's CPU device in F32 (it runs once per new
//! reference clip):
//!
//! 1. [`resample_poly`]: scipy's polyphase resampler, 24 → 16 kHz: a
//!    Kaiser-windowed (β = 5) `firwin` low-pass of `20·max(up, down) + 1`
//!    taps scaled by `up`, zero padding beyond the clip, and scipy's output
//!    alignment and length (`⌈n·up/down⌉`).
//! 2. [`fbank`]: SpeechBrain's `Fbank`: `torch.stft` (n_fft 400, a 25 ms
//!    periodic Hamming window, 10 ms hop, centred with zero padding, so
//!    `1 + n/160` frames), the power spectrum, 80 triangular mel filters over
//!    0–8 kHz, and `10·log10` floored 80 dB below the clip's loudest bin.
//! 3. Sentence mean normalisation: each mel band minus its mean over the clip.
//! 4. ECAPA-TDNN in eval mode (channels 1024 ×4 and 3072, kernels 5/3/3/3/1,
//!    dilations 1/2/3/4/1, Res2Net scale 8, squeeze-excitation 128, attentive
//!    statistics pooling with global context over 128 attention channels):
//!    every TDNN block is conv → ReLU → BatchNorm (running statistics),
//!    convolutions pad "same" by reflection, and the pooled statistics pass
//!    `asp_bn` and `fc` to 192 dimensions.
//! 5. L2 normalisation.
//!
//! Only `embedding_model.ckpt` is read (a `torch.save` F32 state dict); the
//! repository's classifier and global embedding normaliser are unused
//! upstream. The front end reproduces torch's F32 arithmetic where that is
//! cheap (window, filterbank, dB floor, resampler accumulation order) and
//! computes the FFT and mel sums in F64, so it differs from upstream only at
//! F32 rounding level.

use std::f64::consts::PI;
use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::config::SAMPLE_RATE;
use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::batch_norm_affine;
use crate::models::shared::weights::pytorch::PthTensorMap;

/// The speaker embedding width.
pub const EMBEDDING_DIM: usize = 192;
/// Upstream rejects reference clips shorter than 0.1 s at 24 kHz.
pub const MIN_INPUT_SAMPLES: usize = 2_400;

/// The rate the encoder was trained at.
const ENCODER_SAMPLE_RATE: u32 = 16_000;
/// `Fbank(n_mels=80)`.
pub(crate) const N_MELS: usize = 80;
const N_FFT: usize = 400;
const HOP: usize = 160;
const FFT_BINS: usize = N_FFT / 2 + 1;
/// `Filterbank`'s `amin` and `top_db`.
const AMIN: f32 = 1e-10;
const TOP_DB: f32 = 80.0;
/// `resample_poly`'s default window, `('kaiser', 5.0)`.
const KAISER_BETA: f64 = 5.0;

const CHANNELS: [usize; 5] = [1024, 1024, 1024, 1024, 3072];
const KERNEL_SIZES: [usize; 5] = [5, 3, 3, 3, 1];
const DILATIONS: [usize; 5] = [1, 2, 3, 4, 1];
const ATTENTION_CHANNELS: usize = 128;
const RES2NET_SCALE: usize = 8;
const SE_CHANNELS: usize = 128;
const BATCH_NORM_EPS: f64 = 1e-5;
const POOLING_EPS: f64 = 1e-12;
/// The state dict's tensors besides the 31 `num_batches_tracked` counters:
/// a TDNN block holds 6 (conv weight and bias, four BatchNorm tensors), an
/// SE-Res2Net block three TDNN blocks' worth plus seven more and the
/// squeeze-excitation's 4, then `mfa`, `asp.tdnn`, `asp.conv`, `asp_bn` and
/// `fc`. Upstream loads it with `strict=True`; with every name below
/// fetched by shape, an equal count means there is nothing else.
const PARAMETER_TENSORS: usize =
    6 + (CHANNELS.len() - 2) * (6 * (RES2NET_SCALE + 1) + 4) + 6 + 6 + 2 + 4 + 2;

/// The ECAPA-TDNN speaker encoder with its SpeechBrain front end, on the CPU
/// in F32.
pub struct Maya2Ecapa {
    input: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatisticsPooling,
    asp_bn: ChannelAffine,
    fc: SameConv1d,
}

impl Maya2Ecapa {
    /// Load `embedding_model.ckpt` (a `torch.save` state dict) onto the CPU
    /// in F32, folding each BatchNorm's running statistics into a
    /// per-channel scale and shift.
    pub fn load(checkpoint: &Path) -> Result<Self> {
        if !checkpoint.is_file() {
            return Err(Error::ModelLoadError(format!(
                "Maya2 ECAPA weights {} are missing",
                checkpoint.display()
            )));
        }
        let archive = PthTensorMap::open(checkpoint, None)?;
        let tensors = archive.read_where(&Device::Cpu, Some(DType::F32), |name| {
            !name.ends_with(".num_batches_tracked")
        })?;
        if tensors.len() != PARAMETER_TENSORS {
            return Err(Error::ModelLoadError(format!(
                "Maya2 ECAPA weights {} hold {} parameter tensors, expected {PARAMETER_TENSORS}",
                checkpoint.display(),
                tensors.len()
            )));
        }
        let vb = VarBuilder::from_tensors(tensors.into_iter().collect(), DType::F32, &Device::Cpu);
        Self::from_var_builder(&vb).map_err(|error| {
            Error::ModelLoadError(format!(
                "Maya2 ECAPA weights {}: {error}",
                checkpoint.display()
            ))
        })
    }

    fn from_var_builder(vb: &VarBuilder) -> Result<Self> {
        let layers = CHANNELS.len();
        let input = TdnnBlock::load(
            N_MELS,
            CHANNELS[0],
            KERNEL_SIZES[0],
            DILATIONS[0],
            vb.pp("blocks.0"),
        )?;
        let blocks = (1..layers - 1)
            .map(|index| {
                SeRes2NetBlock::load(
                    CHANNELS[index - 1],
                    CHANNELS[index],
                    KERNEL_SIZES[index],
                    DILATIONS[index],
                    vb.pp(format!("blocks.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let last = CHANNELS[layers - 1];
        let mfa = TdnnBlock::load(
            CHANNELS[1..layers - 1].iter().sum(),
            last,
            KERNEL_SIZES[layers - 1],
            DILATIONS[layers - 1],
            vb.pp("mfa"),
        )?;
        let asp = AttentiveStatisticsPooling {
            tdnn: TdnnBlock::load(3 * last, ATTENTION_CHANNELS, 1, 1, vb.pp("asp.tdnn"))?,
            conv: SameConv1d::load(ATTENTION_CHANNELS, last, 1, 1, vb.pp("asp.conv"))?,
        };
        Ok(Self {
            input,
            blocks,
            mfa,
            asp,
            asp_bn: ChannelAffine::load(2 * last, vb.pp("asp_bn.norm"))?,
            fc: SameConv1d::load(2 * last, EMBEDDING_DIM, 1, 1, vb.pp("fc"))?,
        })
    }

    /// L2-normalised 192-d embedding of 24 kHz mono `samples` (>= 0.1 s).
    pub fn embed_24k(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() < MIN_INPUT_SAMPLES {
            return Err(Error::InvalidInput(
                "Maya2 voice-clone reference audio is too short (need at least 0.1 s; \
                 3-10 s recommended)"
                    .into(),
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(
                "Maya2 voice-clone reference audio has non-finite samples".into(),
            ));
        }
        let resampled = resample_poly(samples, ENCODER_SAMPLE_RATE as usize, SAMPLE_RATE as usize)?;
        let features = fbank(&resampled)?;
        // InputNormalization("sentence", std_norm=False): the mean only.
        let features = features.broadcast_sub(&features.mean_keepdim(0)?)?;
        let embedding = self.forward(&features.t()?.unsqueeze(0)?.contiguous()?)?;
        let embedding = embedding.flatten_all()?.to_vec1::<f32>()?;
        // F.normalize(dim=0): x / max(‖x‖₂, 1e-12).
        let norm = embedding
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        Ok(embedding
            .iter()
            .map(|&value| (f64::from(value) / norm) as f32)
            .collect())
    }

    /// Normalised features `[1, N_MELS, frames]` to the raw embedding
    /// `[1, EMBEDDING_DIM, 1]`.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut hidden = self.input.forward(features)?;
        let mut aggregated = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
            aggregated.push(hidden.clone());
        }
        let hidden = self.mfa.forward(&Tensor::cat(&aggregated, 1)?)?;
        let pooled = self.asp.forward(&hidden)?;
        self.fc.forward(&self.asp_bn.forward(&pooled)?)
    }
}

/// `scipy.signal.resample_poly(x, up, down)` with its defaults (Kaiser
/// window β = 5, constant zero padding), for an F32 signal: the filter is
/// rounded to F32 and scaled by `up` there, and each output accumulates its
/// products in F32 in ascending input order, as `upfirdn` does.
pub(crate) fn resample_poly(x: &[f32], up: usize, down: usize) -> Result<Vec<f32>> {
    if up == 0 || down == 0 {
        return Err(Error::InvalidInput(format!(
            "resample_poly needs positive factors, got {up}/{down}"
        )));
    }
    let common = gcd(up, down);
    let (up, down) = (up / common, down / common);
    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    if n_in == 0 {
        return Ok(Vec::new());
    }
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let gain = up as f32;
    let taps: Vec<f32> = firwin_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, KAISER_BETA)
        .into_iter()
        .map(|tap| tap as f32 * gain)
        .collect();
    // scipy prepends `pre_pad` zeros to the filter and drops the first
    // `pre_remove` outputs of the full upsample-filter-downsample, so output
    // `j` is Σᵢ x[i]·h_padded[(j + pre_remove)·down − i·up] (which centres
    // the filter on input j·down/up); zero-padding taps and samples add 0.
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let last_tap = taps.len() - 1;
    Ok((0..n_out)
        .map(|j| {
            // The tap index of input 0 in the unpadded filter.
            let origin = (j + pre_remove) * down - pre_pad;
            let first = origin.saturating_sub(last_tap).div_ceil(up);
            let last = (origin / up).min(n_in - 1);
            let mut acc = 0.0f32;
            for (i, &sample) in x.iter().enumerate().take(last + 1).skip(first) {
                acc += sample * taps[origin - i * up];
            }
            acc
        })
        .collect())
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta))`: a
/// low-pass with `cutoff` relative to Nyquist, scaled to unit DC gain.
fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps - 1) as f64;
    let i0_beta = bessel_i0(beta);
    let mut taps: Vec<f64> = (0..numtaps)
        .map(|n| {
            let m = n as f64 - alpha;
            let ratio = m / alpha;
            let window = bessel_i0(beta * (1.0 - ratio * ratio).sqrt()) / i0_beta;
            cutoff * sinc(cutoff * m) * window
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

/// `numpy.sinc`: `sin(πx) / (πx)`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let y = PI * x;
        y.sin() / y
    }
}

/// The modified Bessel function of the first kind, order 0, by its power
/// series (converged to F64 precision for the window's arguments, ≤ β).
fn bessel_i0(x: f64) -> f64 {
    let quarter_square = 0.25 * x * x;
    let (mut term, mut sum, mut k) = (1.0f64, 1.0f64, 1.0f64);
    loop {
        term *= quarter_square / (k * k);
        sum += term;
        if term <= sum * f64::EPSILON * 0.5 {
            return sum;
        }
        k += 1.0;
    }
}

/// SpeechBrain's `Fbank(n_mels=80)` on 16 kHz `samples`: log-mel features
/// `[1 + len/160, N_MELS]` (F32, CPU), before sentence normalisation.
pub(crate) fn fbank(samples: &[f32]) -> Result<Tensor> {
    if samples.is_empty() {
        return Err(Error::InvalidInput(
            "Maya2 ECAPA features need at least one sample".into(),
        ));
    }
    let frames = 1 + samples.len() / HOP;
    let pad = N_FFT / 2;
    // torch.hamming_window(400), periodic, in torch's F32 op order:
    // cos(n·(2π/400))·(−0.46) + 0.54.
    let step = (2.0 * PI / N_FFT as f64) as f32;
    let window: Vec<f32> = (0..N_FFT)
        .map(|n| (n as f32 * step).cos() * -0.46 + 0.54)
        .collect();
    let filters = mel_filterbank();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
    let mut buffer = vec![Complex::new(0.0f64, 0.0); N_FFT];
    let mut power = [0.0f32; FFT_BINS];
    let mut features = Vec::with_capacity(frames * N_MELS);
    for frame in 0..frames {
        // Frame `t` covers padded samples [160t, 160t + 400); the padding is
        // 200 zeros on each side.
        let start = frame * HOP;
        for (n, value) in buffer.iter_mut().enumerate() {
            let sample = (start + n)
                .checked_sub(pad)
                .and_then(|index| samples.get(index))
                .copied()
                .unwrap_or(0.0);
            *value = Complex::new(f64::from(sample * window[n]), 0.0);
        }
        fft.process(&mut buffer);
        for (bin, value) in power.iter_mut().enumerate() {
            *value = buffer[bin].norm_sqr() as f32;
        }
        for filter in &filters {
            let energy: f64 = filter
                .iter()
                .zip(&power)
                .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                .sum();
            features.push(10.0 * (energy as f32).max(AMIN).log10());
        }
    }
    let floor = features.iter().copied().fold(f32::NEG_INFINITY, f32::max) - TOP_DB;
    for value in &mut features {
        *value = value.max(floor);
    }
    Tensor::from_vec(features, (frames, N_MELS), &Device::Cpu).map_err(Error::from)
}

/// SpeechBrain's frozen triangular `Filterbank` over 0–8 kHz, `[N_MELS][FFT_BINS]`,
/// with torch's F32 arithmetic: mel points `linspace(0, mel(8000), 82)`,
/// filter `m` centred on point `m + 1`, a symmetric triangle whose sides
/// both span the gap down to point `m` (SpeechBrain's `band`).
fn mel_filterbank() -> Vec<[f32; FFT_BINS]> {
    let nyquist = (ENCODER_SAMPLE_RATE / 2) as f32;
    let mel_max = (2595.0 * (1.0 + f64::from(ENCODER_SAMPLE_RATE / 2) / 700.0).log10()) as f32;
    let hz: Vec<f32> = linspace_f32(0.0, mel_max, N_MELS + 2)
        .into_iter()
        .map(|mel| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0))
        .collect();
    let freqs = linspace_f32(0.0, nyquist, FFT_BINS);
    (0..N_MELS)
        .map(|mel| {
            let center = hz[mel + 1];
            let band = hz[mel + 1] - hz[mel];
            let mut filter = [0.0f32; FFT_BINS];
            for (weight, &freq) in filter.iter_mut().zip(&freqs) {
                let slope = (freq - center) / band;
                *weight = (slope + 1.0).min(-slope + 1.0).max(0.0);
            }
            filter
        })
        .collect()
}

/// `torch.linspace` in F32 (its scalar CPU path: the first half counts up
/// from `start`, the second down from `end`).
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|index| {
            if index < halfway {
                start + step * index as f32
            } else {
                end - step * (steps - index - 1) as f32
            }
        })
        .collect()
}

/// An eval-mode `BatchNorm1d` as `x·scale + shift`, both `[1, C, 1]`.
struct ChannelAffine {
    scale: Tensor,
    shift: Tensor,
}

impl ChannelAffine {
    /// The `nn.BatchNorm1d` at `vb` (`weight`, `bias`, running statistics).
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(channels, "weight")?;
        let beta = vb.get(channels, "bias")?;
        let (scale, shift) = batch_norm_affine(
            &vb.get(channels, "running_mean")?,
            &vb.get(channels, "running_var")?,
            BATCH_NORM_EPS,
            Some((&gamma, &beta)),
        )?;
        Ok(Self {
            scale: scale.reshape((1, channels, 1))?,
            shift: shift.reshape((1, channels, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)
            .map_err(Error::from)
    }
}

/// SpeechBrain's `Conv1d(padding="same", padding_mode="reflect")`, stride 1:
/// `dilation·(kernel − 1)/2` reflected frames on each side.
struct SameConv1d {
    conv: Conv1d,
    pad: usize,
}

impl SameConv1d {
    /// The SpeechBrain wrapper at `vb`; its `nn.Conv1d` is `vb.conv`.
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("conv");
        let conv = Conv1d::new(
            vb.get((out_channels, in_channels, kernel_size), "weight")?,
            Some(vb.get(out_channels, "bias")?),
            Conv1dConfig {
                dilation,
                ..Default::default()
            },
        );
        Ok(Self {
            conv,
            pad: dilation * (kernel_size - 1) / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = reflect_pad_time(x, self.pad)?;
        self.conv.forward(&x).map_err(Error::from)
    }
}

/// `F.pad(mode="reflect")` along the time axis of `[batch, channels, time]`
/// (the edge frame is not repeated).
fn reflect_pad_time(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let len = x.dim(2)?;
    if len <= pad {
        return Err(Error::InvalidInput(
            "Maya2 voice-clone reference audio is too short for the speaker encoder".into(),
        ));
    }
    let mut indices = Vec::with_capacity(len + 2 * pad);
    indices.extend((1..=pad).rev().map(|offset| offset as u32));
    indices.extend((0..len).map(|index| index as u32));
    indices.extend((0..pad).map(|offset| (len - 2 - offset) as u32));
    let indices = Tensor::from_vec(indices, (len + 2 * pad,), x.device())?;
    x.contiguous()?
        .index_select(&indices, 2)
        .map_err(Error::from)
}

/// `TDNNBlock`: conv → ReLU → BatchNorm.
struct TdnnBlock {
    conv: SameConv1d,
    norm: ChannelAffine,
}

impl TdnnBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            conv: SameConv1d::load(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                vb.pp("conv"),
            )?,
            norm: ChannelAffine::load(out_channels, vb.pp("norm.norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.norm.forward(&self.conv.forward(x)?.relu()?)
    }
}

/// `SERes2NetBlock`: TDNN → Res2Net → TDNN → squeeze-excitation, plus the
/// input (all blocks keep 1024 channels, so there is no shortcut conv).
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Vec<TdnnBlock>,
    tdnn2: TdnnBlock,
    se_conv1: SameConv1d,
    se_conv2: SameConv1d,
}

impl SeRes2NetBlock {
    fn load(
        in_channels: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if in_channels != channels {
            return Err(Error::ModelLoadError(format!(
                "ECAPA SE-Res2Net block {} would need a shortcut ({in_channels} → {channels})",
                vb.prefix()
            )));
        }
        let part = channels / RES2NET_SCALE;
        let res2net = (0..RES2NET_SCALE - 1)
            .map(|index| {
                TdnnBlock::load(
                    part,
                    part,
                    kernel_size,
                    dilation,
                    vb.pp(format!("res2net_block.blocks.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tdnn1: TdnnBlock::load(in_channels, channels, 1, 1, vb.pp("tdnn1"))?,
            res2net,
            tdnn2: TdnnBlock::load(channels, channels, 1, 1, vb.pp("tdnn2"))?,
            se_conv1: SameConv1d::load(channels, SE_CHANNELS, 1, 1, vb.pp("se_block.conv1"))?,
            se_conv2: SameConv1d::load(SE_CHANNELS, channels, 1, 1, vb.pp("se_block.conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.tdnn1.forward(x)?;
        // Res2Net: chunk 0 passes through, chunk i > 0 takes chunk i plus the
        // previous output (from chunk 2 on) through its own TDNN block.
        let parts = hidden.chunk(RES2NET_SCALE, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(RES2NET_SCALE);
        for (index, part) in parts.iter().enumerate() {
            let output = match index {
                0 => part.clone(),
                1 => self.res2net[0].forward(part)?,
                _ => self.res2net[index - 1].forward(&(part + &outputs[index - 1])?)?,
            };
            outputs.push(output);
        }
        let hidden = self.tdnn2.forward(&Tensor::cat(&outputs, 1)?)?;
        // Squeeze-excitation over the clip's mean.
        let gate = self.se_conv1.forward(&hidden.mean_keepdim(2)?)?.relu()?;
        let gate = candle_nn::ops::sigmoid(&self.se_conv2.forward(&gate)?)?;
        (hidden.broadcast_mul(&gate)? + x).map_err(Error::from)
    }
}

/// Attentive statistics pooling with global context: `[1, C, T]` to the
/// attention-weighted mean and std `[1, 2C, 1]`.
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock,
    conv: SameConv1d,
}

impl AttentiveStatisticsPooling {
    /// Weighted mean and std over time; `weights` sum to one per channel.
    fn statistics(x: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.broadcast_mul(weights)?.sum_keepdim(2)?;
        let variance = x
            .broadcast_sub(&mean)?
            .sqr()?
            .broadcast_mul(weights)?
            .sum_keepdim(2)?;
        Ok((mean, variance.clamp(POOLING_EPS, f64::MAX)?.sqrt()?))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, time) = x.dims3()?;
        let uniform = Tensor::full(1.0f32 / time as f32, (batch, 1, time), x.device())?;
        let (mean, std) = Self::statistics(x, &uniform)?;
        let context = Tensor::cat(
            &[
                x.clone(),
                mean.broadcast_as((batch, channels, time))?,
                std.broadcast_as((batch, channels, time))?,
            ],
            1,
        )?;
        let attention = self.conv.forward(&self.tdnn.forward(&context)?.tanh()?)?;
        let attention = candle_nn::ops::softmax(&attention, D::Minus1)?;
        let (mean, std) = Self::statistics(x, &attention)?;
        Tensor::cat(&[mean, std], 1).map_err(Error::from)
    }
}

#[cfg(test)]
// Goldens are printed from scipy 1.17.0 (`resample_poly`, `firwin`) and from
// a line-by-line torch 2.9.1 transcription of speechbrain 1.1.2's `Fbank`
// (`torch.stft`, `spectral_magnitude`, `Filterbank`) at full precision.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// A 64-bit LCG's top 24 bits in [-1, 1): exact in F32, so the golden
    /// generator reproduces it bit for bit.
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn tone(len: usize, freq: f64, rate: f64) -> Vec<f32> {
        (0..len)
            .map(|n| (0.5 * (2.0 * PI * freq * n as f64 / rate).sin()) as f32)
            .collect()
    }

    /// `firwin(61, 1/3, window=('kaiser', 5.0))`, the 24 → 16 kHz filter.
    const FIRWIN_GOLDEN: [(usize, f64); 7] = [
        (0, -4.7730704295658725e-19),
        (1, -0.0005075758102200313),
        (2, -0.0007171615080117002),
        (27, 1.2713991644429615e-17),
        (28, 0.13655221514239765),
        (29, 0.2751477226789338),
        (30, 0.3335353911845924),
    ];

    #[test]
    fn firwin_matches_scipy() {
        let taps = firwin_kaiser(61, 1.0 / 3.0, KAISER_BETA);
        for (index, expected) in FIRWIN_GOLDEN {
            assert!(
                (taps[index] - expected).abs() < 1e-15,
                "tap {index}: {} vs {expected}",
                taps[index]
            );
        }
        for index in 0..61 {
            assert_eq!(taps[index], taps[60 - index], "symmetric at {index}");
        }
        assert!((taps.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn resample_length_follows_scipy() {
        for (n_in, n_out) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (2_400, 1_600),
            (2_401, 1_601),
            (2_402, 1_602),
        ] {
            assert_eq!(
                resample_poly(&vec![0.5; n_in], 2, 3).unwrap().len(),
                n_out,
                "{n_in}"
            );
        }
        // Rates reduce by their gcd, and equal rates copy.
        let x = noise(100, 1);
        assert_eq!(
            resample_poly(&x, 16_000, 24_000).unwrap(),
            resample_poly(&x, 2, 3).unwrap()
        );
        assert_eq!(resample_poly(&x, 5, 5).unwrap(), x);
        assert!(resample_poly(&x, 0, 3).is_err());
    }

    /// scipy centres output j's filter on input 1.5·j, so an impulse at
    /// input 0 lands on output 0 as 2·h[30] (the filter's zeros at multiples
    /// of 3 taps silence the other outputs), and one at input 1 reads taps
    /// 28, 31, 34, … at outputs 0, 1, 2, ….
    #[test]
    fn resample_places_taps_as_scipy_does() {
        let h = |index: usize| {
            (FIRWIN_GOLDEN.iter().find(|(i, _)| *i == index).unwrap().1 * 2.0) as f32
        };
        let mut impulse = vec![0.0f32; 60];
        impulse[0] = 1.0;
        let y = resample_poly(&impulse, 2, 3).unwrap();
        assert!((y[0] - h(30)).abs() < 1e-7, "{}", y[0]);
        for (j, value) in y.iter().enumerate().take(11).skip(1) {
            assert!(value.abs() < 1e-7, "output {j}: {value}");
        }
        impulse.swap(0, 1);
        let y = resample_poly(&impulse, 2, 3).unwrap();
        assert!((y[0] - h(28)).abs() < 1e-7, "{}", y[0]);
        assert!((y[1] - h(29)).abs() < 1e-7, "{}", y[1]); // tap 31 = tap 29
    }

    #[test]
    fn resample_matches_scipy() {
        let y = resample_poly(&noise(2_405, 7), 2, 3).unwrap();
        assert_eq!(y.len(), 1_604);
        let golden = [
            (0, 0.10966508835554123),
            (1, 0.9861419796943665),
            (2, -0.29267972707748413),
            (3, -0.5799150466918945),
            (10, -0.38266608119010925),
            (500, 0.5263014435768127),
            (801, -0.15079794824123383),
            (1_600, -0.986931562423706),
            (1_601, -0.40984559059143066),
            (1_602, 0.21506264805793762),
            (1_603, 0.2734487056732178),
        ];
        for (index, expected) in golden {
            assert!(
                (f64::from(y[index]) - expected).abs() < 2e-6,
                "y[{index}] = {} vs {expected}",
                y[index]
            );
        }
        let sum: f64 = y.iter().map(|&value| f64::from(value).abs()).sum();
        assert!((sum - 599.8484269140754).abs() < 1e-3, "{sum}");
    }

    #[test]
    fn resample_keeps_dc_and_tones() {
        // Away from the edges (the filter spans 30 taps at 48 kHz), DC and a
        // 1 kHz tone pass unchanged up to the Kaiser passband ripple.
        let dc = resample_poly(&vec![1.0f32; 2_400], 2, 3).unwrap();
        for value in &dc[20..1_580] {
            assert!((value - 1.0).abs() < 1e-3, "{value}");
        }
        let y = resample_poly(&tone(12_000, 1_000.0, 24_000.0), 2, 3).unwrap();
        assert_eq!(y.len(), 8_000);
        let expected = tone(8_000, 1_000.0, 16_000.0);
        for j in 20..7_980 {
            assert!(
                (y[j] - expected[j]).abs() < 1e-3,
                "{j}: {} vs {}",
                y[j],
                expected[j]
            );
        }
    }

    #[test]
    fn fbank_frames_follow_the_centred_stft() {
        for (len, frames) in [(1, 1), (159, 1), (160, 2), (16_000, 101), (16_159, 101)] {
            let features = fbank(&noise(len, 2)).unwrap();
            assert_eq!(features.dims(), &[frames, N_MELS], "{len}");
            let values = features.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(values.iter().all(|value| value.is_finite()), "{len}");
        }
        // Silence sits at the amplitude floor: 10·log10(1e-10).
        let silence = fbank(&[0.0; 1_600])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(silence.iter().all(|&value| (value + 100.0).abs() < 1e-4));
        assert!(fbank(&[]).is_err());
    }

    #[test]
    fn fbank_matches_the_torch_reference() {
        let mut x = noise(3_205, 11);
        for value in &mut x[1_600..] {
            *value *= 1e-5;
        }
        let features = fbank(&x).unwrap();
        assert_eq!(features.dims(), &[21, N_MELS]);
        let rows = features.to_vec2::<f32>().unwrap();
        let golden = [
            (0, 0, 8.129216194152832),
            (0, 40, 13.903635025024414),
            (0, 79, 20.17289161682129),
            (5, 10, 6.039702415466309),
            (9, 33, 20.772417068481445),
            (10, 79, 17.005876541137695),
            // The quiet half sits at the 80 dB floor below the peak.
            (12, 5, -50.603275299072266),
            (15, 60, -50.603275299072266),
            (20, 0, -50.603275299072266),
            (20, 79, -50.603275299072266),
        ];
        for (frame, mel, expected) in golden {
            let value = f64::from(rows[frame][mel]);
            assert!(
                (value - expected).abs() < 1e-4,
                "[{frame}][{mel}] = {value} vs {expected}"
            );
        }
        let normalized = features
            .broadcast_sub(&features.mean_keepdim(0).unwrap())
            .unwrap();
        let rows = normalized.to_vec2::<f32>().unwrap();
        for (frame, mel, expected) in [
            (0, 0, 27.787765502929688),
            (10, 40, 31.054523468017578),
            (20, 79, -41.27900314331055),
        ] {
            let value = f64::from(rows[frame][mel]);
            assert!(
                (value - expected).abs() < 1e-4,
                "norm [{frame}][{mel}] = {value} vs {expected}"
            );
        }
    }

    /// The whole front end, 24 kHz in: scipy's resampler, then the torch
    /// transcription of `Fbank` and the sentence mean.
    #[test]
    fn front_end_matches_the_reference() {
        let w24: Vec<f32> = noise(4_800, 3)
            .into_iter()
            .map(|value| value * 0.5)
            .collect();
        let w16 = resample_poly(&w24, 2, 3).unwrap();
        let features = fbank(&w16).unwrap();
        let features = features
            .broadcast_sub(&features.mean_keepdim(0).unwrap())
            .unwrap();
        assert_eq!(features.dims(), &[21, N_MELS]);
        let rows = features.to_vec2::<f32>().unwrap();
        for (frame, mel, expected) in [
            (0, 0, -6.144469261169434),
            (7, 21, -0.7952260971069336),
            (13, 50, 3.183309555053711),
            (20, 79, -2.0084047317504883),
        ] {
            let value = f64::from(rows[frame][mel]);
            assert!(
                (value - expected).abs() < 1e-4,
                "[{frame}][{mel}] = {value} vs {expected}"
            );
        }
    }

    /// A tone at a filter's centre frequency peaks in that filter.
    #[test]
    fn fbank_tone_peaks_in_its_mel_band() {
        let mel_max = 2595.0 * (1.0f64 + 8_000.0 / 700.0).log10();
        for band in [10, 20, 60] {
            let mel = mel_max * (band + 1) as f64 / (N_MELS + 1) as f64;
            let freq = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
            let features = fbank(&tone(16_000, freq, 16_000.0)).unwrap();
            let mean = features.mean(0).unwrap().to_vec1::<f32>().unwrap();
            let peak = mean
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(peak, band, "{freq:.1} Hz");
        }
    }

    #[test]
    fn parameter_count_matches_the_checkpoint_layout() {
        // 231 tensors in embedding_model.ckpt, 31 of them num_batches_tracked.
        assert_eq!(PARAMETER_TENSORS, 200);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    #[ignore = "needs the Maya2 ECAPA checkpoint (IZWI_MAYA2_MODEL_DIR)"]
    fn real_checkpoint_embeds_reference_clips() {
        let Some(model_dir) = std::env::var_os("IZWI_MAYA2_MODEL_DIR") else {
            eprintln!("IZWI_MAYA2_MODEL_DIR unset; not run");
            return;
        };
        let checkpoint = Path::new(&model_dir).join(super::super::config::ECAPA_WEIGHTS_FILE);
        let ecapa = Maya2Ecapa::load(&checkpoint).unwrap();

        // A buzzy 120 Hz voice-like harmonic series and band-limited noise.
        let voiced: Vec<f32> = (0..24_000)
            .map(|n| {
                let t = n as f64 / 24_000.0;
                let value: f64 = (1..=20)
                    .map(|k| (2.0 * PI * 120.0 * k as f64 * t).sin() / k as f64)
                    .sum();
                (0.2 * value) as f32
            })
            .collect();
        let hiss = resample_poly(&noise(36_000, 5), 2, 3).unwrap();

        let a = ecapa.embed_24k(&voiced).unwrap();
        let b = ecapa.embed_24k(&hiss).unwrap();
        for embedding in [&a, &b] {
            assert_eq!(embedding.len(), EMBEDDING_DIM);
            assert!(embedding.iter().all(|value| value.is_finite()));
            let norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        }
        let different = cosine(&a, &b);
        assert!(different < 0.99, "distinct signals: cosine {different}");
        assert_eq!(ecapa.embed_24k(&voiced).unwrap(), a, "deterministic");
        // The sentence mean cancels a gain change in the log-mel domain.
        let quieter: Vec<f32> = voiced.iter().map(|value| value * 0.25).collect();
        let same = cosine(&a, &ecapa.embed_24k(&quieter).unwrap());
        assert!(same > 0.999, "gain change: cosine {same}");
        eprintln!("cosine(voiced, hiss) = {different}, cosine(voiced, quieter) = {same}");

        assert!(ecapa.embed_24k(&voiced[..MIN_INPUT_SAMPLES - 1]).is_err());
        assert!(ecapa.embed_24k(&voiced[..MIN_INPUT_SAMPLES]).is_ok());
    }
}

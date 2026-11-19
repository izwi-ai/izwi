//! The speaker encoder: CAM++ (3D-Speaker, Apache-2.0) over mean-normalised
//! Kaldi fbanks, giving the 192-number voice embedding Ming conditions a
//! cloned voice on.
//!
//! Upstream runs `campplus.onnx` through onnxruntime. izwi loads the same
//! network from its published PyTorch checkpoint (`funasr/campplus`,
//! `campplus_cn_common.bin`): the ONNX file is that checkpoint with each
//! batch norm folded into the convolution before it, which was checked
//! tensor by tensor on 2026-11-19. Batch norms are folded here at load too.
//!
//! Each stage matches the upstream function it replaces:
//!
//! * [`sinc_resample`]: torchaudio's `Resample` (`sinc_interp_hann`, width
//!   6, rolloff 0.99), kernel built in F64.
//! * [`kaldi_fbank`]: torchaudio's `compliance.kaldi.fbank` (80 bins, no
//!   dither, Povey window, pre-emphasis 0.97) on samples in [-1, 1], each
//!   bin's mean over the clip subtracted.
//! * [`CampPlus::embed`]: the network in eval mode, F32 on the host device.

use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Module, VarBuilder};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::models::shared::weights::pytorch::PthTensorMap;

/// The embedding's width.
pub const SPEAKER_EMBEDDING_DIM: usize = 192;
/// The rate the encoder hears.
pub const SPEAKER_SAMPLE_RATE: u32 = 16_000;

fn conv1d(vb: &VarBuilder, cfg: Conv1dConfig) -> Result<Conv1d> {
    let dtype = vb.dtype();
    let weight = vb.get_unchecked_dtype("weight", dtype)?.contiguous()?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get_unchecked_dtype("bias", dtype)?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, cfg))
}

fn conv_cfg(padding: usize, stride: usize, dilation: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding,
        stride,
        dilation,
        groups: 1,
        cudnn_fwd_algo: None,
    }
}

fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// torchaudio `Resample(orig, new)` of one mono signal.
pub fn sinc_resample(input: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new || input.is_empty() {
        return input.to_vec();
    }
    let g = gcd(u64::from(orig), u64::from(new));
    let orig = (u64::from(orig) / g) as usize;
    let new = (u64::from(new) / g) as usize;
    const WIDTH_ZEROS: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let base = orig.min(new) as f64 * ROLLOFF;
    let width = (WIDTH_ZEROS * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // kernel[phase][tap], in F64 then F32 as torchaudio rounds it.
    let mut kernel = vec![0f32; new * taps];
    for phase in 0..new {
        for tap in 0..taps {
            let idx = (tap as f64 - width as f64) / orig as f64;
            let t = (-(phase as f64) / new as f64 + idx) * base;
            let t = t.clamp(-WIDTH_ZEROS, WIDTH_ZEROS);
            let window = (t * std::f64::consts::PI / WIDTH_ZEROS / 2.0).cos().powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[phase * taps + tap] = (sinc * window * base / orig as f64) as f32;
        }
    }
    // Pad `width` before and `width + orig` after, then a strided
    // convolution: output block `b` reads padded[b * orig ..][..taps].
    let length = input.len();
    let mut padded = vec![0f32; width + length + width + orig];
    padded[width..width + length].copy_from_slice(input);
    let blocks = (padded.len() - taps) / orig + 1;
    let target = (new as u128 * length as u128).div_ceil(orig as u128) as usize;
    let mut output = Vec::with_capacity(blocks * new);
    for block in 0..blocks {
        let window = &padded[block * orig..block * orig + taps];
        for phase in 0..new {
            let row = &kernel[phase * taps..(phase + 1) * taps];
            let mut acc = 0f32;
            for (x, k) in window.iter().zip(row) {
                acc += x * k;
            }
            output.push(acc);
        }
    }
    output.truncate(target);
    output
}

/// torchaudio `compliance.kaldi.fbank(num_mel_bins=80)` at 16 kHz with its
/// defaults (25 ms frames, 10 ms shift, snip edges, DC removal,
/// pre-emphasis 0.97, Povey window, 512-point FFT, power, Kaldi mel bins
/// from 20 Hz to Nyquist, natural log floored at F32 epsilon), then each
/// bin's mean over time subtracted (CAMPPlus' `extract_feature`).
/// Returns [frames, 80].
pub fn kaldi_fbank(samples: &[f32]) -> (Vec<f32>, usize) {
    const FRAME: usize = 400;
    const SHIFT: usize = 160;
    const PADDED: usize = 512;
    const BINS: usize = 80;
    const PREEMPH: f32 = 0.97;
    if samples.len() < FRAME {
        return (Vec::new(), 0);
    }
    let frames = 1 + (samples.len() - FRAME) / SHIFT;
    // Povey window: symmetric Hann to the power 0.85.
    let window = (0..FRAME)
        .map(|i| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (FRAME - 1) as f64).cos();
            hann.powf(0.85) as f32
        })
        .collect::<Vec<_>>();
    // Kaldi mel banks on the 257 positive bins (the Nyquist column is zero).
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let nyquist = 8_000.0;
    let (low, high) = (mel(20.0), mel(nyquist));
    let delta = (high - low) / (BINS + 1) as f64;
    let fft_bin_width = 16_000.0 / PADDED as f64;
    let mut banks = vec![0f32; BINS * (PADDED / 2 + 1)];
    for b in 0..BINS {
        let left = low + b as f64 * delta;
        let center = left + delta;
        let right = center + delta;
        for k in 0..PADDED / 2 {
            let m = mel(fft_bin_width * k as f64);
            let up = (m - left) / (center - left);
            let down = (right - m) / (right - center);
            banks[b * (PADDED / 2 + 1) + k] = up.min(down).max(0.0) as f32;
        }
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(PADDED);
    let mut buffer = vec![Complex::new(0f32, 0f32); PADDED];
    let mut features = vec![0f32; frames * BINS];
    for frame in 0..frames {
        let raw = &samples[frame * SHIFT..frame * SHIFT + FRAME];
        let mean = raw.iter().sum::<f32>() / FRAME as f32;
        let centred = raw.iter().map(|&x| x - mean).collect::<Vec<_>>();
        for i in 0..PADDED {
            buffer[i] = if i < FRAME {
                let previous = centred[i.saturating_sub(1)];
                Complex::new((centred[i] - PREEMPH * previous) * window[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buffer);
        for b in 0..BINS {
            let row = &banks[b * (PADDED / 2 + 1)..(b + 1) * (PADDED / 2 + 1)];
            let energy = row
                .iter()
                .zip(&buffer)
                .map(|(w, c)| w * c.norm_sqr())
                .sum::<f32>();
            features[frame * BINS + b] = energy.max(f32::EPSILON).ln();
        }
    }
    for b in 0..BINS {
        let mean = (0..frames).map(|t| features[t * BINS + b]).sum::<f32>() / frames as f32;
        for t in 0..frames {
            features[t * BINS + b] -= mean;
        }
    }
    (features, frames)
}

/// Eval-mode batch norm folded to a per-channel scale and shift.
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn load(vb: VarBuilder, channels: usize, affine: bool) -> Result<Self> {
        let host = vb.clone().set_device(Device::Cpu);
        let mean = host_f32(&host.get(channels, "running_mean")?)?;
        let var = host_f32(&host.get(channels, "running_var")?)?;
        let (weight, bias) = if affine {
            (
                host_f32(&host.get(channels, "weight")?)?,
                host_f32(&host.get(channels, "bias")?)?,
            )
        } else {
            (vec![1.0; channels], vec![0.0; channels])
        };
        let mut scale = Vec::with_capacity(channels);
        let mut shift = Vec::with_capacity(channels);
        for c in 0..channels {
            let s = weight[c] / (var[c] + 1e-5).sqrt();
            scale.push(s);
            shift.push(bias[c] - mean[c] * s);
        }
        let device = vb.device();
        Ok(Self {
            scale: Tensor::from_vec(scale, channels, device)?,
            shift: Tensor::from_vec(shift, channels, device)?,
        })
    }

    /// `x` with channels on dim 1 ([B, C, ...]).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut shape = vec![1usize; x.rank()];
        shape[1] = self.scale.dim(0)?;
        Ok(x.broadcast_mul(&self.scale.reshape(shape.as_slice())?)?
            .broadcast_add(&self.shift.reshape(shape.as_slice())?)?)
    }
}

/// A 2-D convolution with stride (s, 1): stride 1, then every s-th row.
struct StridedConv2d {
    conv: Conv2d,
    row_stride: usize,
}

impl StridedConv2d {
    fn load(
        vb: VarBuilder,
        input: usize,
        output: usize,
        kernel: usize,
        padding: usize,
        row_stride: usize,
    ) -> Result<Self> {
        let weight = vb.get((output, input, kernel, kernel), "weight")?;
        Ok(Self {
            conv: Conv2d::new(
                weight,
                None,
                Conv2dConfig {
                    padding,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            row_stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        if self.row_stride == 1 {
            return Ok(y);
        }
        let rows = y.dim(2)?;
        let keep = (rows - 1) / self.row_stride + 1;
        let index = Tensor::arange_step(
            0u32,
            (keep * self.row_stride) as u32,
            self.row_stride as u32,
            x.device(),
        )?;
        Ok(y.index_select(&index, 2)?)
    }
}

struct ResBlock2d {
    conv1: StridedConv2d,
    bn1: BatchNorm,
    conv2: StridedConv2d,
    bn2: BatchNorm,
    shortcut: Option<(StridedConv2d, BatchNorm)>,
}

impl ResBlock2d {
    fn load(vb: VarBuilder, stride: usize) -> Result<Self> {
        Ok(Self {
            conv1: StridedConv2d::load(vb.pp("conv1"), 32, 32, 3, 1, stride)?,
            bn1: BatchNorm::load(vb.pp("bn1"), 32, true)?,
            conv2: StridedConv2d::load(vb.pp("conv2"), 32, 32, 3, 1, 1)?,
            bn2: BatchNorm::load(vb.pp("bn2"), 32, true)?,
            shortcut: (stride != 1)
                .then(|| -> Result<_> {
                    Ok((
                        StridedConv2d::load(vb.pp("shortcut.0"), 32, 32, 1, 0, stride)?,
                        BatchNorm::load(vb.pp("shortcut.1"), 32, true)?,
                    ))
                })
                .transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let skip = match &self.shortcut {
            Some((conv, bn)) => bn.forward(&conv.forward(x)?)?,
            None => x.clone(),
        };
        Ok(out.add(&skip)?.relu()?)
    }
}

struct DenseTdnnLayer {
    bn1: BatchNorm,
    linear1: Conv1d,
    bn2: BatchNorm,
    local: Conv1d,
    cam1: Conv1d,
    cam2: Conv1d,
}

impl DenseTdnnLayer {
    fn load(vb: VarBuilder, input: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            bn1: BatchNorm::load(vb.pp("nonlinear1.batchnorm"), input, true)?,
            linear1: conv1d(&vb.pp("linear1"), conv_cfg(0, 1, 1))?,
            bn2: BatchNorm::load(vb.pp("nonlinear2.batchnorm"), 128, true)?,
            local: conv1d(
                &vb.pp("cam_layer.linear_local"),
                conv_cfg(dilation, 1, dilation),
            )?,
            cam1: conv1d(&vb.pp("cam_layer.linear1"), conv_cfg(0, 1, 1))?,
            cam2: conv1d(&vb.pp("cam_layer.linear2"), conv_cfg(0, 1, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(&self.bn1.forward(x)?.relu()?)?;
        let h = self.bn2.forward(&h)?.relu()?;
        let y = self.local.forward(&h)?;
        let context = h
            .mean_keepdim(D::Minus1)?
            .broadcast_add(&segment_average(&h, 100)?)?;
        let gate =
            candle_nn::ops::sigmoid(&self.cam2.forward(&self.cam1.forward(&context)?.relu()?)?)?;
        Ok(y.mul(&gate)?)
    }
}

/// `avg_pool1d(kernel=stride=seg, ceil_mode=True)` expanded back to every
/// frame: each frame gets the mean of its segment's frames.
fn segment_average(x: &Tensor, seg: usize) -> Result<Tensor> {
    let (batch, channels, frames) = x.dims3()?;
    let segments = frames.div_ceil(seg);
    let padded = x.pad_with_zeros(2, 0, segments * seg - frames)?;
    let sums = padded
        .reshape((batch, channels, segments, seg))?
        .sum_keepdim(3)?;
    let counts = (0..segments)
        .map(|s| (seg.min(frames - s * seg)) as f32)
        .collect::<Vec<_>>();
    let counts = Tensor::from_vec(counts, (1, 1, segments, 1), x.device())?.to_dtype(x.dtype())?;
    Ok(sums
        .broadcast_div(&counts)?
        .repeat((1, 1, 1, seg))?
        .reshape((batch, channels, segments * seg))?
        .narrow(2, 0, frames)?)
}

pub struct CampPlus {
    conv1: StridedConv2d,
    bn1: BatchNorm,
    layers: Vec<ResBlock2d>,
    conv2: StridedConv2d,
    bn2: BatchNorm,
    tdnn: Conv1d,
    tdnn_bn: BatchNorm,
    blocks: Vec<(Vec<DenseTdnnLayer>, BatchNorm, Conv1d)>,
    out_bn: BatchNorm,
    dense: Conv1d,
    dense_bn: BatchNorm,
}

impl CampPlus {
    /// Load the network from the published torch archive at `path`, in F32
    /// on the host.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(Error::ModelLoadError(format!(
                "Ming speaker encoder weights {} are missing",
                path.display()
            )));
        }
        let archive = PthTensorMap::open(path, None)?;
        let tensors = archive
            .read_where(&Device::Cpu, Some(DType::F32), |name| {
                !name.ends_with("num_batches_tracked")
            })?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        Self::load(VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu))
    }

    pub(crate) fn load(vb: VarBuilder) -> Result<Self> {
        let head = vb.pp("head");
        let xv = vb.pp("xvector");
        let mut layers = Vec::new();
        for (name, stride) in [
            ("layer1.0", 2),
            ("layer1.1", 1),
            ("layer2.0", 2),
            ("layer2.1", 1),
        ] {
            layers.push(ResBlock2d::load(head.pp(name), stride)?);
        }
        let mut channels = 128;
        let mut blocks = Vec::new();
        for (index, (count, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)]
            .into_iter()
            .enumerate()
        {
            let block = xv.pp(format!("block{}", index + 1));
            let dense = (0..count)
                .map(|i| {
                    DenseTdnnLayer::load(
                        block.pp(format!("tdnnd{}", i + 1)),
                        channels + i * 32,
                        dilation,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            channels += count * 32;
            let transit = xv.pp(format!("transit{}", index + 1));
            blocks.push((
                dense,
                BatchNorm::load(transit.pp("nonlinear.batchnorm"), channels, true)?,
                conv1d(&transit.pp("linear"), conv_cfg(0, 1, 1))?,
            ));
            channels /= 2;
        }
        Ok(Self {
            conv1: StridedConv2d::load(head.pp("conv1"), 1, 32, 3, 1, 1)?,
            bn1: BatchNorm::load(head.pp("bn1"), 32, true)?,
            layers,
            conv2: StridedConv2d::load(head.pp("conv2"), 32, 32, 3, 1, 2)?,
            bn2: BatchNorm::load(head.pp("bn2"), 32, true)?,
            tdnn: conv1d(&xv.pp("tdnn.linear"), conv_cfg(2, 2, 1))?,
            tdnn_bn: BatchNorm::load(xv.pp("tdnn.nonlinear.batchnorm"), 128, true)?,
            blocks,
            out_bn: BatchNorm::load(xv.pp("out_nonlinear.batchnorm"), channels, true)?,
            dense: conv1d(&xv.pp("dense.linear"), conv_cfg(0, 1, 1))?,
            dense_bn: BatchNorm::load(xv.pp("dense.nonlinear.batchnorm"), 192, false)?,
        })
    }

    /// The 192-number embedding of mono `samples` at `sample_rate`.
    pub fn embed_clip(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let wav16 = sinc_resample(samples, sample_rate, SPEAKER_SAMPLE_RATE);
        let embedding = self.embed(&wav16, &Device::Cpu)?;
        if embedding.len() != SPEAKER_EMBEDDING_DIM
            || embedding.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InferenceError(
                "Ming speaker encoder produced an invalid embedding".into(),
            ));
        }
        Ok(embedding)
    }

    /// The 192-dim x-vector of 16 kHz `samples`.
    pub fn embed(&self, samples: &[f32], device: &Device) -> Result<Vec<f32>> {
        let (features, frames) = kaldi_fbank(samples);
        if frames < 2 {
            return Err(Error::InvalidInput(
                "Ming reference audio is too short for the speaker encoder".into(),
            ));
        }
        // [T, 80] → [1, 1, 80, T].
        let x = Tensor::from_vec(features, (frames, 80), device)?
            .t()?
            .contiguous()?
            .reshape((1, 1, 80, frames))?;
        let mut x = self.bn1.forward(&self.conv1.forward(&x)?)?.relu()?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x = self.bn2.forward(&self.conv2.forward(&x)?)?.relu()?;
        let (_, c, f, t) = x.dims4()?;
        let x = x.reshape((1, c * f, t))?;
        let mut x = self.tdnn_bn.forward(&self.tdnn.forward(&x)?)?.relu()?;
        for (dense, bn, transit) in &self.blocks {
            for layer in dense {
                let out = layer.forward(&x)?;
                x = Tensor::cat(&[&x, &out], 1)?;
            }
            x = transit.forward(&bn.forward(&x)?.relu()?)?;
        }
        let x = self.out_bn.forward(&x)?.relu()?;
        // Statistics pooling: mean and unbiased std over time.
        let frames = x.dim(2)? as f64;
        let mean = x.mean_keepdim(2)?;
        let var = x
            .broadcast_sub(&mean)?
            .sqr()?
            .sum_keepdim(2)?
            .affine(1.0 / (frames - 1.0).max(1.0), 0.0)?;
        let stats = Tensor::cat(&[mean, var.sqrt()?], 1)?;
        let embedding = self.dense_bn.forward(&self.dense.forward(&stats)?)?;
        host_f32(&embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_length_ratio_and_a_low_tone() {
        let input = (0..44_100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin())
            .collect::<Vec<_>>();
        let out = sinc_resample(&input, 44_100, 16_000);
        assert_eq!(out.len(), 16_000);
        for (i, value) in out.iter().enumerate().skip(100).take(15_800) {
            let expected = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            assert!(
                (value - expected).abs() < 2e-3,
                "{i}: {value} vs {expected}"
            );
        }
        assert_eq!(sinc_resample(&input, 44_100, 44_100), input);
    }

    #[test]
    fn kaldi_fbank_frames_and_mean_normalises() {
        let samples = (0..16_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect::<Vec<_>>();
        let (features, frames) = kaldi_fbank(&samples);
        assert_eq!(frames, 1 + (16_000 - 400) / 160);
        for b in 0..80 {
            let mean = (0..frames).map(|t| features[t * 80 + b]).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-3, "bin {b} mean {mean}");
        }
    }

    #[test]
    fn segment_average_matches_ceil_mode_pooling() {
        let x = Tensor::arange(0f32, 5.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 5))
            .unwrap();
        let out = host_f32(&segment_average(&x, 2).unwrap()).unwrap();
        assert_eq!(out, vec![0.5, 0.5, 2.5, 2.5, 4.0]);
    }
}

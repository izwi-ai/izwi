//! ECAPA-TDNN speaker encoder (the x-vector) for Qwen3-TTS voice cloning.
//!
//! A port of the reference `Qwen3TTSSpeakerEncoder` and `mel_spectrogram`
//! (QwenLM/Qwen3-TTS `modeling_qwen3_tts.py`). The Base checkpoints ship it as
//! `speaker_encoder.*`; the reference puts its output into the codec prefix of
//! every voice-clone prompt, between the think/language tags and pad/BOS.

use std::f64::consts::PI;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::Deserialize;

use crate::error::{Error, Result};

/// `speaker_encoder_config` with the reference defaults
/// (`Qwen3TTSSpeakerEncoderConfig`); the checkpoints only set `enc_dim` and
/// `sample_rate`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

fn default_mel_dim() -> usize {
    128
}
fn default_enc_dim() -> usize {
    1024
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_sample_rate() -> u32 {
    24_000
}

/// The reference `extract_speaker_embedding` mel front end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerMelParams {
    pub n_fft: usize,
    pub num_mels: usize,
    pub sample_rate: u32,
    pub hop_size: usize,
    pub fmin: f64,
    pub fmax: f64,
}

impl SpeakerMelParams {
    /// n_fft 1024, 128 mels at 24 kHz, hop 256, 0-12 kHz.
    pub const REFERENCE: Self = Self {
        n_fft: 1024,
        num_mels: 128,
        sample_rate: 24_000,
        hop_size: 256,
        fmin: 0.0,
        fmax: 12_000.0,
    };
}

/// The reference `mel_spectrogram`: reflect-pad `(n_fft - hop) / 2` on both
/// sides, periodic Hann STFT without centering, magnitude
/// `sqrt(re^2 + im^2 + 1e-9)`, librosa's Slaney mel filterbank, then
/// `ln(max(x, 1e-5))`. Returns `[frames][num_mels]`.
pub fn speaker_mel_spectrogram(samples: &[f32], params: SpeakerMelParams) -> Result<Vec<Vec<f32>>> {
    let n_fft = params.n_fft;
    let pad = (n_fft - params.hop_size) / 2;
    if samples.len() <= pad {
        return Err(Error::InvalidInput(
            "Voice-cloning reference audio is too short for the speaker encoder".into(),
        ));
    }
    let padded = reflect_pad(samples, pad);
    if padded.len() < n_fft {
        return Err(Error::InvalidInput(
            "Voice-cloning reference audio is too short for the speaker encoder".into(),
        ));
    }
    let frames = (padded.len() - n_fft) / params.hop_size + 1;
    let window: Vec<f64> = (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / n_fft as f64).cos())
        .collect();
    let filters = slaney_mel_filterbank(params);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n_fft);
    let bins = n_fft / 2 + 1;
    let mut buffer = vec![Complex::new(0.0f64, 0.0); n_fft];
    let mut magnitude = vec![0.0f64; bins];
    let mut mel = Vec::with_capacity(frames);
    for frame in 0..frames {
        let start = frame * params.hop_size;
        for (index, value) in buffer.iter_mut().enumerate() {
            *value = Complex::new(f64::from(padded[start + index]) * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for (bin, value) in magnitude.iter_mut().enumerate() {
            *value = (buffer[bin].norm_sqr() + 1e-9).sqrt();
        }
        mel.push(
            filters
                .iter()
                .map(|filter| {
                    let energy: f64 = filter.iter().zip(&magnitude).map(|(w, m)| w * m).sum();
                    energy.max(1e-5).ln() as f32
                })
                .collect(),
        );
    }
    Ok(mel)
}

/// NumPy/PyTorch `reflect` padding (the edge sample is not repeated).
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let len = samples.len();
    let mut padded = Vec::with_capacity(len + 2 * pad);
    padded.extend((1..=pad).rev().map(|offset| samples[offset]));
    padded.extend_from_slice(samples);
    padded.extend((0..pad).map(|offset| samples[len - 2 - offset]));
    padded
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` with its defaults:
/// Slaney mel scale and Slaney area normalization, computed in f64 and
/// rounded to f32 as librosa returns it.
fn slaney_mel_filterbank(params: SpeakerMelParams) -> Vec<Vec<f64>> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= MIN_LOG_HZ {
            min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
        } else {
            hz / F_SP
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel >= min_log_mel {
            MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
        } else {
            F_SP * mel
        }
    };
    let bins = params.n_fft / 2 + 1;
    let nyquist = f64::from(params.sample_rate) / 2.0;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|bin| nyquist * bin as f64 / (bins - 1) as f64)
        .collect();
    let (low, high) = (hz_to_mel(params.fmin), hz_to_mel(params.fmax));
    let points = params.num_mels + 2;
    let mel_freqs: Vec<f64> = (0..points)
        .map(|index| mel_to_hz(low + (high - low) * index as f64 / (points - 1) as f64))
        .collect();
    (0..params.num_mels)
        .map(|mel| {
            let (left, center, right) = (mel_freqs[mel], mel_freqs[mel + 1], mel_freqs[mel + 2]);
            let norm = 2.0 / (right - left);
            fft_freqs
                .iter()
                .map(|&freq| {
                    let lower = (freq - left) / (center - left);
                    let upper = (right - freq) / (right - center);
                    f64::from((lower.min(upper).max(0.0) * norm) as f32)
                })
                .collect()
        })
        .collect()
}

/// `nn.Conv1d(padding="same", padding_mode="reflect")`.
struct SameConv1d {
    conv: Conv1d,
    left: usize,
    right: usize,
}

impl SameConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = candle_nn::conv1d(
            in_channels,
            out_channels,
            kernel_size,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            },
            vb,
        )?;
        let total = dilation * (kernel_size - 1);
        Ok(Self {
            conv,
            left: total / 2,
            right: total - total / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = reflect_pad_time(x, self.left, self.right)?;
        self.conv.forward(&x).map_err(Error::from)
    }
}

/// Reflect padding along the last (time) axis of `[batch, channels, time]`.
fn reflect_pad_time(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let len = x.dim(2)?;
    if len <= left.max(right) {
        return Err(Error::InvalidInput(
            "Voice-cloning reference audio is too short for the speaker encoder".into(),
        ));
    }
    let mut indices = Vec::with_capacity(left + len + right);
    indices.extend((1..=left).rev().map(|offset| offset as u32));
    indices.extend((0..len).map(|index| index as u32));
    indices.extend((0..right).map(|offset| (len - 2 - offset) as u32));
    let indices = Tensor::from_vec(indices, (left + len + right,), x.device())?;
    x.contiguous()?
        .index_select(&indices, 2)
        .map_err(Error::from)
}

/// Conv + ReLU (`TimeDelayNetBlock`).
struct TdnnBlock {
    conv: SameConv1d,
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
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)?.relu().map_err(Error::from)
    }
}

struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn load(
        channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if scale == 0 || !channels.is_multiple_of(scale) {
            return Err(Error::ModelLoadError(format!(
                "speaker encoder Res2Net scale {scale} does not divide {channels} channels"
            )));
        }
        let part = channels / scale;
        let blocks = (0..scale - 1)
            .map(|index| {
                TdnnBlock::load(
                    part,
                    part,
                    kernel_size,
                    dilation,
                    vb.pp(format!("blocks.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let parts = x.chunk(self.scale, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        for (index, part) in parts.iter().enumerate() {
            let output = match index {
                0 => part.clone(),
                1 => self.blocks[0].forward(part)?,
                _ => self.blocks[index - 1].forward(&(part + &outputs[index - 1])?)?,
            };
            outputs.push(output);
        }
        Tensor::cat(&outputs, 1).map_err(Error::from)
    }
}

struct SqueezeExcitation {
    conv1: SameConv1d,
    conv2: SameConv1d,
}

impl SqueezeExcitation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(2)?;
        let gate = self.conv1.forward(&mean)?.relu()?;
        let gate = candle_nn::ops::sigmoid(&self.conv2.forward(&gate)?)?;
        x.broadcast_mul(&gate).map_err(Error::from)
    }
}

struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SqueezeExcitation,
}

impl SeRes2NetBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.tdnn1.forward(x)?;
        let hidden = self.res2net.forward(&hidden)?;
        let hidden = self.tdnn2.forward(&hidden)?;
        let hidden = self.se.forward(&hidden)?;
        (hidden + x).map_err(Error::from)
    }
}

/// Attentive statistics pooling over time: `[batch, 2 x channels, 1]`.
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock,
    conv: SameConv1d,
}

impl AttentiveStatisticsPooling {
    const EPS: f64 = 1e-12;

    /// Weighted mean and std over time; `weights` sum to one per channel.
    fn statistics(x: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.broadcast_mul(weights)?.sum_keepdim(2)?;
        let variance = x
            .broadcast_sub(&mean)?
            .sqr()?
            .broadcast_mul(weights)?
            .sum_keepdim(2)?;
        Ok((mean, variance.clamp(Self::EPS, f64::MAX)?.sqrt()?))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, time) = x.dims3()?;
        // Built in F32: Metal has no F64 storage, so an F64 fill on the
        // device fails with "unsupported const-set f64".
        let uniform =
            Tensor::full(1.0f32 / time as f32, (batch, 1, time), x.device())?.to_dtype(x.dtype())?;
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

/// `Qwen3TTSSpeakerEncoder`: mel `[batch, frames, mel_dim]` to
/// `[batch, enc_dim]`.
pub struct SpeakerEncoder {
    input: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatisticsPooling,
    fc: SameConv1d,
    config: SpeakerEncoderConfig,
}

impl SpeakerEncoder {
    pub fn load(config: SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let layers = config.enc_channels.len();
        if layers < 2
            || config.enc_kernel_sizes.len() != layers
            || config.enc_dilations.len() != layers
        {
            return Err(Error::ModelLoadError(
                "speaker encoder enc_channels, enc_kernel_sizes and enc_dilations must match"
                    .into(),
            ));
        }
        let channels = &config.enc_channels;
        let input = TdnnBlock::load(
            config.mel_dim,
            channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            vb.pp("blocks.0"),
        )?;
        let blocks = (1..layers - 1)
            .map(|index| {
                let vb = vb.pp(format!("blocks.{index}"));
                let out = channels[index];
                Ok(SeRes2NetBlock {
                    tdnn1: TdnnBlock::load(channels[index - 1], out, 1, 1, vb.pp("tdnn1"))?,
                    res2net: Res2NetBlock::load(
                        out,
                        config.enc_res2net_scale,
                        config.enc_kernel_sizes[index],
                        config.enc_dilations[index],
                        vb.pp("res2net_block"),
                    )?,
                    tdnn2: TdnnBlock::load(out, out, 1, 1, vb.pp("tdnn2"))?,
                    se: SqueezeExcitation {
                        conv1: SameConv1d::load(
                            out,
                            config.enc_se_channels,
                            1,
                            1,
                            vb.pp("se_block.conv1"),
                        )?,
                        conv2: SameConv1d::load(
                            config.enc_se_channels,
                            out,
                            1,
                            1,
                            vb.pp("se_block.conv2"),
                        )?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let last = channels[layers - 1];
        let aggregated: usize = channels[1..layers - 1].iter().sum();
        let mfa = TdnnBlock::load(
            aggregated,
            last,
            config.enc_kernel_sizes[layers - 1],
            config.enc_dilations[layers - 1],
            vb.pp("mfa"),
        )?;
        let asp = AttentiveStatisticsPooling {
            tdnn: TdnnBlock::load(
                last * 3,
                config.enc_attention_channels,
                1,
                1,
                vb.pp("asp.tdnn"),
            )?,
            conv: SameConv1d::load(config.enc_attention_channels, last, 1, 1, vb.pp("asp.conv"))?,
        };
        let fc = SameConv1d::load(last * 2, config.enc_dim, 1, 1, vb.pp("fc"))?;
        Ok(Self {
            input,
            blocks,
            mfa,
            asp,
            fc,
            config,
        })
    }

    pub fn config(&self) -> &SpeakerEncoderConfig {
        &self.config
    }

    /// `[batch, frames, mel_dim]` to `[batch, enc_dim]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut hidden = mel.transpose(1, 2)?.contiguous()?;
        hidden = self.input.forward(&hidden)?;
        let mut aggregated = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
            aggregated.push(hidden.clone());
        }
        let hidden = self.mfa.forward(&Tensor::cat(&aggregated, 1)?)?;
        let hidden = self.asp.forward(&hidden)?;
        self.fc.forward(&hidden)?.squeeze(2).map_err(Error::from)
    }

    /// The reference `extract_speaker_embedding` on 24 kHz mono samples:
    /// `[enc_dim]` F32.
    pub fn embed_samples(&self, samples_24k: &[f32], device: &Device) -> Result<Tensor> {
        let mel = speaker_mel_spectrogram(samples_24k, SpeakerMelParams::REFERENCE)?;
        let frames = mel.len();
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let mel = Tensor::from_vec(flat, (1, frames, self.config.mel_dim), device)?;
        self.forward(&mel)?
            .i(0)?
            .to_dtype(DType::F32)
            .map_err(Error::from)
    }
}

#[cfg(test)]
// Goldens are printed from the PyTorch reference at full f64 precision.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

    /// Name-keyed xorshift weights: biases scaled 0.1, kernels by
    /// 1/sqrt(fan-in). The golden generator uses the same values.
    struct XorshiftWeights;

    impl candle_nn::var_builder::SimpleBackend for XorshiftWeights {
        fn get(
            &self,
            shape: candle_core::Shape,
            name: &str,
            _: candle_nn::Init,
            dtype: DType,
            device: &Device,
        ) -> candle_core::Result<Tensor> {
            let seed = format!("speaker_encoder.{name}")
                .bytes()
                .fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
                });
            let scale = if name.ends_with("bias") {
                0.1
            } else {
                let fan_in = shape.dims().iter().skip(1).product::<usize>().max(1);
                1.0 / (fan_in as f32).sqrt()
            };
            xorshift_tensor(shape.dims(), seed, scale, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)
        }

        fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
            candle_core::bail!("unexpected unchecked tensor request {name}")
        }

        fn contains_tensor(&self, _: &str) -> bool {
            false
        }
    }

    /// QTTS-7: a small ECAPA encoder (channels 16-16-16-32, kernels 5-3-3-1,
    /// dilations 1-2-3-1, Res2Net scale 4) matches the reference
    /// `Qwen3TTSSpeakerEncoder` run in PyTorch on the same xorshift weights
    /// and input (golden generated on the dev Mac).
    #[test]
    fn speaker_encoder_matches_the_pytorch_reference() {
        let config = SpeakerEncoderConfig {
            mel_dim: 8,
            enc_dim: 6,
            enc_channels: vec![16, 16, 16, 32],
            enc_kernel_sizes: vec![5, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 4,
            enc_se_channels: 8,
            sample_rate: 24_000,
        };
        let vb = VarBuilder::from_backend(Box::new(XorshiftWeights), DType::F32, Device::Cpu);
        let encoder = SpeakerEncoder::load(config, vb).unwrap();
        let mel = xorshift_tensor(&[1, 20, 8], 99, 1.0, &Device::Cpu).unwrap();
        let output = host_f32(&encoder.forward(&mel).unwrap()).unwrap();
        let golden = [
            5.626647267e-03f32,
            -2.657077089e-02,
            2.953370288e-02,
            5.000360310e-03,
            -1.108159572e-01,
            4.901552945e-02,
        ];
        let error = max_abs_diff(&output, &golden);
        assert!(error < 1e-5, "x-vector max abs error {error}: {output:?}");
    }

    /// The x-vector also runs on Metal, which has no F64 storage: the same
    /// small encoder on the device matches the PyTorch golden.
    #[cfg(feature = "metal")]
    #[test]
    fn speaker_encoder_runs_on_metal() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("speaker_encoder_runs_on_metal")
        else {
            return;
        };
        let config = SpeakerEncoderConfig {
            mel_dim: 8,
            enc_dim: 6,
            enc_channels: vec![16, 16, 16, 32],
            enc_kernel_sizes: vec![5, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 4,
            enc_se_channels: 8,
            sample_rate: 24_000,
        };
        let vb = VarBuilder::from_backend(Box::new(XorshiftWeights), DType::F32, device.clone());
        let encoder = SpeakerEncoder::load(config, vb).unwrap();
        let mel = xorshift_tensor(&[1, 20, 8], 99, 1.0, &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let output = host_f32(&encoder.forward(&mel).unwrap()).unwrap();
        let golden = [
            5.626647267e-03f32,
            -2.657077089e-02,
            2.953370288e-02,
            5.000360310e-03,
            -1.108159572e-01,
            4.901552945e-02,
        ];
        let error = max_abs_diff(&output, &golden);
        assert!(error < 1e-4, "Metal x-vector max abs error {error}: {output:?}");
    }

    /// QTTS-7: the mel front end matches the reference `mel_spectrogram`
    /// (librosa Slaney filterbank, torch STFT) at small parameters: n_fft 64,
    /// hop 16, 6 mels at 1.6 kHz up to 800 Hz, 300 xorshift samples.
    #[test]
    fn speaker_mel_matches_the_pytorch_reference() {
        let samples = crate::kernels::numeric_check::xorshift_values(300, 5, 0.5);
        let params = SpeakerMelParams {
            n_fft: 64,
            num_mels: 6,
            sample_rate: 1_600,
            hop_size: 16,
            fmin: 0.0,
            fmax: 800.0,
        };
        let mel = speaker_mel_spectrogram(&samples, params).unwrap();
        assert_eq!((mel.len(), mel[0].len()), (18, 6));
        // Reference layout is [mels, frames].
        let actual: Vec<f32> = (0..6)
            .flat_map(|m| mel.iter().map(move |frame| frame[m]))
            .collect();
        let error = max_abs_diff(&actual, &MEL_GOLDEN);
        assert!(error < 1e-4, "mel max abs error {error}");
    }

    /// Real-weight check of the 0.6B Base x-vector against the reference run
    /// in PyTorch (BF16 weights widened to F32 on both sides) on 3 s of
    /// xorshift audio. Set IZWI_TEST_QWEN_TTS_BASE_DIR or keep the default
    /// izwi model path.
    #[test]
    #[ignore = "needs a local Qwen3-TTS-12Hz-0.6B-Base checkpoint"]
    fn published_speaker_encoder_matches_the_pytorch_reference() {
        let dir = {
            let _env = crate::env_test_lock().lock().unwrap();
            std::env::var_os("IZWI_TEST_QWEN_TTS_BASE_DIR")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(
                            "Library/Application Support/izwi/models/Qwen3-TTS-12Hz-0.6B-Base",
                        )
                    })
                })
                .unwrap()
        };
        let config: crate::models::architectures::qwen3::tts::config::Qwen3TtsConfig =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &Device::Cpu,
            )
            .unwrap()
        };
        let encoder = SpeakerEncoder::load(
            config.speaker_encoder_config.unwrap(),
            vb.pp("speaker_encoder"),
        )
        .unwrap();
        let audio = crate::kernels::numeric_check::xorshift_values(72_000, 7, 0.5);
        let embedding = host_f32(&encoder.embed_samples(&audio, &Device::Cpu).unwrap()).unwrap();
        let first8 = [
            0.1541769802570343f32,
            0.05539964139461517,
            0.12661632895469666,
            -0.057967036962509155,
            -0.023948341608047485,
            0.18634825944900513,
            0.020080700516700745,
            -0.12960506975650787,
        ];
        let error = max_abs_diff(&embedding[..8], &first8);
        let norm = embedding
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let probe = crate::kernels::numeric_check::xorshift_values(embedding.len(), 11, 1.0);
        let dot: f64 = embedding
            .iter()
            .zip(&probe)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        eprintln!("first8 max abs {error:e}, norm {norm}, dot {dot}");
        assert!(error < 1e-3, "first8 max abs {error}");
        assert!(
            (norm - 8.355371369005166).abs() < 1e-3 * 8.36,
            "norm {norm}"
        );
        assert!((dot - -5.637133586759392).abs() < 1e-2, "dot {dot}");
    }

    /// The reference `mel_spectrogram` output for the small-parameter test,
    /// `[6 mels][18 frames]` flattened.
    const MEL_GOLDEN: [f32; 108] = [
        -3.4578812e+00f32,
        -3.3677592e+00,
        -3.1540494e+00,
        -3.2378840e+00,
        -3.8319767e+00,
        -3.4549792e+00,
        -3.1899269e+00,
        -2.9626172e+00,
        -2.8387544e+00,
        -2.6908669e+00,
        -3.0871861e+00,
        -2.9130530e+00,
        -2.7552047e+00,
        -3.0851374e+00,
        -2.7777443e+00,
        -2.8123019e+00,
        -2.8275290e+00,
        -3.0236497e+00,
        -2.8327434e+00,
        -2.8693118e+00,
        -3.4093027e+00,
        -3.3444870e+00,
        -3.1675756e+00,
        -2.9097950e+00,
        -2.8025773e+00,
        -2.7548575e+00,
        -3.1145658e+00,
        -3.4157677e+00,
        -3.1765592e+00,
        -2.6606488e+00,
        -2.9956834e+00,
        -2.9103405e+00,
        -2.6501203e+00,
        -2.9539819e+00,
        -3.2228177e+00,
        -3.4919453e+00,
        -3.0492196e+00,
        -2.8189523e+00,
        -3.1389699e+00,
        -3.3505921e+00,
        -2.7622011e+00,
        -2.7853332e+00,
        -3.2655127e+00,
        -3.2011943e+00,
        -3.6065481e+00,
        -3.3146842e+00,
        -2.9431996e+00,
        -2.8744407e+00,
        -2.6695557e+00,
        -2.6586349e+00,
        -2.9178863e+00,
        -3.0655684e+00,
        -2.9280856e+00,
        -3.3707581e+00,
        -3.4492691e+00,
        -3.6340911e+00,
        -3.4657311e+00,
        -3.0673287e+00,
        -2.8107283e+00,
        -3.2933686e+00,
        -3.1604440e+00,
        -3.0262141e+00,
        -3.3554537e+00,
        -3.0633786e+00,
        -3.0374491e+00,
        -3.1452582e+00,
        -2.8111033e+00,
        -3.1837101e+00,
        -2.9326749e+00,
        -3.0551679e+00,
        -3.1382270e+00,
        -2.8852663e+00,
        -3.6994200e+00,
        -2.9562604e+00,
        -2.9691689e+00,
        -3.2834718e+00,
        -3.0194435e+00,
        -3.4524462e+00,
        -3.4893503e+00,
        -3.3008039e+00,
        -3.5042703e+00,
        -3.0303576e+00,
        -2.9219790e+00,
        -2.8385322e+00,
        -2.8322141e+00,
        -2.8946669e+00,
        -2.8293791e+00,
        -3.1254711e+00,
        -3.0956879e+00,
        -3.1282170e+00,
        -3.0071249e+00,
        -2.7815602e+00,
        -2.4986317e+00,
        -2.7885580e+00,
        -2.9060507e+00,
        -3.1144633e+00,
        -3.0400686e+00,
        -3.1672008e+00,
        -3.4534600e+00,
        -3.3458271e+00,
        -2.7188559e+00,
        -2.6116292e+00,
        -2.7772162e+00,
        -3.1740336e+00,
        -3.1607914e+00,
        -3.0149748e+00,
        -3.3558159e+00,
        -3.1724038e+00,
    ];
}

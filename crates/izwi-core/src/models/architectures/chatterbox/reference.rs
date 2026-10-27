//! Voice cloning: the three reference encoders and upstream's
//! `prepare_conditionals`, from reference audio alone (no transcript).
//!
//! * T3's speaker vector: the LSTM voice encoder (`voice_encoder/`, from
//!   Real-Time-Voice-Cloning, MIT) over 160-frame partials of the trimmed
//!   16 kHz reference, averaged and normalised.
//! * T3's prompt and S3Gen's prompt tokens: the S3 speech tokenizer V2
//!   (xingchensong/S3Tokenizer, Apache-2.0): log-mel, two strided convs, six
//!   FSMN attention blocks with RoPE, finite scalar quantisation to 6561
//!   codes at 25 Hz.
//! * S3Gen's speaker x-vector: CAMPPlus (3D-Speaker/FunASR, MIT) over
//!   mean-normalised Kaldi fbanks.
//! * S3Gen's prompt mel: [`reference_dsp::mel_s3gen_reference`].
//!
//! Upstream reads the reference at 24 kHz with librosa's SoXR resampler and
//! resamples again to 16 kHz with SoXR (T3) and torchaudio (S3Gen); izwi
//! uses the exact torchaudio resampler throughout.

use candle_core::{Device, Tensor, D};
use candle_nn::rnn::{LSTMConfig, LSTM, RNN};
use candle_nn::{Conv1d, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

use super::conds::{ChatterboxVoice, S3GenVoiceReference, T3VoiceConditioning};
use super::config::{S3GEN_SAMPLE_RATE, S3_TOKENIZER_SAMPLE_RATE};
use super::reference_dsp::{
    kaldi_fbank, log_mel_s3tokenizer, mel_s3gen_reference, mel_voice_encoder, sinc_resample,
    trim_silence,
};
use super::weights::{conv1d, conv_cfg, host_f32};
use crate::error::{Error, Result};

/// S3Gen's reference crop (10 s at 24 kHz) and T3's (6 s at 16 kHz).
const S3GEN_REFERENCE_SAMPLES: usize = 10 * S3GEN_SAMPLE_RATE as usize;
const T3_REFERENCE_SAMPLES: usize = 6 * S3_TOKENIZER_SAMPLE_RATE as usize;
/// Shortest reference izwi accepts: one second.
const MIN_REFERENCE_SECONDS: f32 = 1.0;

pub struct ChatterboxReferenceEncoders {
    voice_encoder: VoiceEncoder,
    tokenizer: S3SpeechTokenizer,
    campplus: CampPlus,
    device: Device,
}

impl ChatterboxReferenceEncoders {
    /// `ve` is `ve.safetensors`; `s3gen` the S3Gen file (its `tokenizer.*`
    /// and `speaker_encoder.*` tensors).
    pub fn load(ve: VarBuilder, s3gen: VarBuilder) -> Result<Self> {
        Ok(Self {
            device: ve.device().clone(),
            voice_encoder: VoiceEncoder::load(ve)?,
            tokenizer: S3SpeechTokenizer::load(s3gen.pp("tokenizer"))?,
            campplus: CampPlus::load(s3gen.pp("speaker_encoder"))?,
        })
    }

    /// Upstream's `prepare_conditionals` for mono `samples` at
    /// `sample_rate`.
    pub fn prepare_voice(&self, samples: &[f32], sample_rate: u32) -> Result<ChatterboxVoice> {
        if sample_rate == 0 || samples.is_empty() {
            return Err(Error::InvalidInput(
                "Chatterbox reference audio is empty".into(),
            ));
        }
        if (samples.len() as f32) < MIN_REFERENCE_SECONDS * sample_rate as f32 {
            return Err(Error::InvalidInput(format!(
                "Chatterbox needs at least {MIN_REFERENCE_SECONDS} s of reference audio"
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(
                "Chatterbox reference audio is not finite".into(),
            ));
        }
        let wav24 = sinc_resample(samples, sample_rate, S3GEN_SAMPLE_RATE);
        let wav16 = sinc_resample(&wav24, S3GEN_SAMPLE_RATE, S3_TOKENIZER_SAMPLE_RATE);

        // S3Gen: the first 10 s.
        let ref24 = &wav24[..wav24.len().min(S3GEN_REFERENCE_SAMPLES)];
        let (mut prompt_mel, mut frames) = mel_s3gen_reference(ref24);
        let ref16 = sinc_resample(ref24, S3GEN_SAMPLE_RATE, S3_TOKENIZER_SAMPLE_RATE);
        let speaker_xvector = self.campplus.embed(&ref16, &self.device)?;
        let mut prompt_tokens = self.tokenizer.tokenize(&ref16, None, &self.device)?;
        // Upstream trims the tokens to half the mel frames; izwi also trims
        // the mels to twice the tokens, so the two always pair.
        prompt_tokens.truncate(frames / 2);
        frames = frames.min(2 * prompt_tokens.len());
        prompt_mel.truncate(frames * 80);

        // T3: the first 6 s for the prompt, the whole reference for the
        // speaker vector.
        let t3_prompt = self.tokenizer.tokenize(
            &wav16[..wav16.len().min(T3_REFERENCE_SAMPLES)],
            Some(150),
            &self.device,
        )?;
        let speaker_embedding = self.voice_encoder.embed(&wav16, &self.device)?;

        let voice = ChatterboxVoice {
            t3: T3VoiceConditioning {
                speaker_embedding,
                prompt_speech_tokens: t3_prompt,
            },
            s3gen: S3GenVoiceReference {
                prompt_tokens,
                prompt_mel,
                prompt_mel_frames: frames,
                speaker_xvector,
            },
        };
        voice.validate()?;
        Ok(voice)
    }

    pub fn voice_encoder(&self) -> &VoiceEncoder {
        &self.voice_encoder
    }

    pub fn tokenizer(&self) -> &S3SpeechTokenizer {
        &self.tokenizer
    }

    pub fn campplus(&self) -> &CampPlus {
        &self.campplus
    }
}

// ----------------------------------------------------------- voice encoder

pub struct VoiceEncoder {
    layers: Vec<LSTM>,
    proj: Linear,
}

impl VoiceEncoder {
    const PARTIAL_FRAMES: usize = 160;
    /// `get_frame_step(rate=1.3)`: round(16000 / 1.3 / 160).
    const FRAME_STEP: usize = 77;
    const MIN_COVERAGE: f64 = 0.8;

    fn load(vb: VarBuilder) -> Result<Self> {
        let lstm = vb.pp("lstm");
        let layers = (0..3)
            .map(|layer| {
                let input = if layer == 0 { 40 } else { 256 };
                Ok(candle_nn::lstm(
                    input,
                    256,
                    LSTMConfig {
                        layer_idx: layer,
                        ..Default::default()
                    },
                    lstm.clone(),
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            proj: candle_nn::linear(256, 256, vb.pp("proj"))?,
        })
    }

    /// Upstream's `get_num_wins` and padded length for `frames` mel frames.
    fn partials(frames: usize) -> (usize, usize) {
        let (step, win) = (Self::FRAME_STEP, Self::PARTIAL_FRAMES);
        let span = (frames + step).saturating_sub(win);
        let (mut count, remainder) = (span / step, span % step);
        if count == 0 || (remainder + (win - step)) as f64 / win as f64 >= Self::MIN_COVERAGE {
            count += 1;
        }
        (count, win + step * (count - 1))
    }

    /// `embeds_from_wavs` for one 16 kHz reference: trimmed, power mel,
    /// partial embeddings averaged and normalised. Returns [256].
    pub fn embed(&self, wav16: &[f32], device: &Device) -> Result<Vec<f32>> {
        let trimmed = trim_silence(wav16, 20.0);
        if trimmed.len() < 2 {
            return Err(Error::InvalidInput(
                "Chatterbox reference audio is silent".into(),
            ));
        }
        let (mel, frames) = mel_voice_encoder(trimmed);
        let (count, target) = Self::partials(frames);
        let mut padded = mel;
        padded.resize(target * 40, 0.0);
        let mut partials = Vec::with_capacity(count * Self::PARTIAL_FRAMES * 40);
        for i in 0..count {
            let start = i * Self::FRAME_STEP * 40;
            partials.extend_from_slice(&padded[start..start + Self::PARTIAL_FRAMES * 40]);
        }
        let mut x = Tensor::from_vec(partials, (count, Self::PARTIAL_FRAMES, 40), device)?;
        let mut last = None;
        for layer in &self.layers {
            let states = layer.seq(&x)?;
            last = states.last().map(|state| state.h().clone());
            x = layer.states_to_tensor(&states)?;
        }
        let hidden =
            last.ok_or_else(|| Error::InferenceError("voice encoder ran no steps".into()))?;
        let raw = self.proj.forward(&hidden)?.relu()?;
        let normed = raw.broadcast_div(&raw.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        let mean = normed.mean(0)?;
        let embedding = mean.broadcast_div(&mean.sqr()?.sum_all()?.sqrt()?)?;
        host_f32(&embedding)
    }
}

// --------------------------------------------------------- speech tokenizer

pub struct S3SpeechTokenizer {
    mel_filters: Vec<f32>,
    conv1: Conv1d,
    conv2: Conv1d,
    blocks: Vec<TokenizerBlock>,
    project_down: Linear,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

struct TokenizerBlock {
    attn_ln: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    /// Depthwise FSMN kernel [1280, 1, 31].
    fsmn: Tensor,
    mlp_ln: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
}

impl S3SpeechTokenizer {
    const STATE: usize = 1280;
    const HEADS: usize = 20;
    const HEAD_DIM: usize = 64;

    fn load(vb: VarBuilder) -> Result<Self> {
        let encoder = vb.pp("encoder");
        let blocks = (0..6)
            .map(|i| {
                let b = encoder.pp(format!("blocks.{i}"));
                let attn = b.pp("attn");
                Ok(TokenizerBlock {
                    attn_ln: candle_nn::layer_norm(Self::STATE, 1e-5, b.pp("attn_ln"))?,
                    query: candle_nn::linear(Self::STATE, Self::STATE, attn.pp("query"))?,
                    key: candle_nn::linear_no_bias(Self::STATE, Self::STATE, attn.pp("key"))?,
                    value: candle_nn::linear(Self::STATE, Self::STATE, attn.pp("value"))?,
                    out: candle_nn::linear(Self::STATE, Self::STATE, attn.pp("out"))?,
                    fsmn: attn.get((Self::STATE, 1, 31), "fsmn_block.weight")?,
                    mlp_ln: candle_nn::layer_norm(Self::STATE, 1e-5, b.pp("mlp_ln"))?,
                    mlp_in: candle_nn::linear(Self::STATE, 4 * Self::STATE, b.pp("mlp.0"))?,
                    mlp_out: candle_nn::linear(4 * Self::STATE, Self::STATE, b.pp("mlp.2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // precompute_freqs_cis(64, 2048), θ 10000, both halves.
        let half = Self::HEAD_DIM / 2;
        let mut cos = Vec::with_capacity(2048 * half);
        let mut sin = Vec::with_capacity(2048 * half);
        for position in 0..2048 {
            for i in 0..half {
                let freq = 1.0 / 10000f32.powf((2 * i) as f32 / Self::HEAD_DIM as f32);
                let angle = position as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let device = vb.device();
        Ok(Self {
            mel_filters: host_f32(&vb.get((128, 201), "_mel_filters")?)?,
            conv1: conv1d(&encoder.pp("conv1"), conv_cfg(1, 2, 1))?,
            conv2: conv1d(&encoder.pp("conv2"), conv_cfg(1, 2, 1))?,
            blocks,
            project_down: candle_nn::linear(
                Self::STATE,
                8,
                vb.pp("quantizer._codebook.project_down"),
            )?,
            rope_cos: Tensor::from_vec(cos, (2048, half), device)?,
            rope_sin: Tensor::from_vec(sin, (2048, half), device)?,
        })
    }

    /// Speech tokens for 16 kHz `samples`; `max_len` keeps the first
    /// `max_len` tokens' mel frames (4 per token).
    pub fn tokenize(
        &self,
        samples: &[f32],
        max_len: Option<usize>,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let (mel, mut frames) = log_mel_s3tokenizer(samples, &self.mel_filters);
        if frames == 0 {
            return Err(Error::InvalidInput(
                "Chatterbox reference audio is too short to tokenize".into(),
            ));
        }
        let full = frames;
        if let Some(max_len) = max_len {
            frames = frames.min(4 * max_len);
        }
        let mel = Tensor::from_vec(mel, (128, full), device)?.narrow(1, 0, frames)?;
        let x = self.conv1.forward(&mel.unsqueeze(0)?)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let mut x = x.squeeze(0)?.t()?.contiguous()?;
        let tokens = x.dim(0)?;
        if tokens > 2048 {
            return Err(Error::InvalidInput(
                "Chatterbox reference is too long to tokenize".into(),
            ));
        }
        let cos = self.rope_cos.narrow(0, 0, tokens)?;
        let sin = self.rope_sin.narrow(0, 0, tokens)?;
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }
        // FSQ: tanh, ×0.999, round half to even, +1, base-3 digits.
        let h = self.project_down.forward(&x)?.tanh()?;
        let h = host_f32(&h)?;
        Ok(h.chunks(8)
            .map(|digits| {
                digits
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        let level = (value * 0.999).round_ties_even() + 1.0;
                        level as u32 * 3u32.pow(i as u32)
                    })
                    .sum()
            })
            .collect())
    }
}

impl TokenizerBlock {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (tokens, state) = x.dims2()?;
        let (heads, head_dim) = (S3SpeechTokenizer::HEADS, S3SpeechTokenizer::HEAD_DIM);
        let normed = self.attn_ln.forward(x)?;
        let split =
            |t: Tensor| -> candle_core::Result<Tensor> { t.reshape((tokens, heads, head_dim)) };
        let q = split(self.query.forward(&normed)?)?;
        let k = split(self.key.forward(&normed)?)?;
        let v = self.value.forward(&normed)?;
        let rotate = |t: &Tensor| -> Result<Tensor> {
            let half = head_dim / 2;
            let (a, b) = (t.narrow(2, 0, half)?, t.narrow(2, half, half)?);
            let (c, s) = (cos.unsqueeze(1)?, sin.unsqueeze(1)?);
            Ok(Tensor::cat(
                &[
                    a.broadcast_mul(&c)?.sub(&b.broadcast_mul(&s)?)?,
                    b.broadcast_mul(&c)?.add(&a.broadcast_mul(&s)?)?,
                ],
                2,
            )?)
        };
        let scale = (head_dim as f64).powf(-0.25);
        let q = rotate(&q)?
            .transpose(0, 1)?
            .affine(scale, 0.0)?
            .contiguous()?;
        let k = rotate(&k)?
            .transpose(0, 1)?
            .affine(scale, 0.0)?
            .contiguous()?;
        let vh = v
            .reshape((tokens, heads, head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let weights = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?)?;
        let attended = weights
            .matmul(&vh)?
            .transpose(0, 1)?
            .reshape((tokens, state))?;
        // FSMN memory: depthwise conv (31 taps, centred) over v, plus v.
        let memory = v
            .t()?
            .unsqueeze(0)?
            .pad_with_zeros(2, 15, 15)?
            .conv1d(&self.fsmn, 0, 1, 1, state)?
            .squeeze(0)?
            .t()?
            .add(&v)?;
        let x = x.add(&self.out.forward(&attended)?.add(&memory)?)?;
        let mlp = self
            .mlp_out
            .forward(&self.mlp_in.forward(&self.mlp_ln.forward(&x)?)?.gelu_erf()?)?;
        Ok(x.add(&mlp)?)
    }
}

// ----------------------------------------------------------------- CAMPPlus

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
    fn load(vb: VarBuilder) -> Result<Self> {
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

    /// The 192-dim x-vector of 16 kHz `samples`.
    pub fn embed(&self, samples: &[f32], device: &Device) -> Result<Vec<f32>> {
        let (features, frames) = kaldi_fbank(samples);
        if frames < 2 {
            return Err(Error::InvalidInput(
                "Chatterbox reference audio is too short for the speaker encoder".into(),
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
    fn voice_encoder_partials_follow_upstream_windows() {
        // A short clip still yields one full partial.
        assert_eq!(VoiceEncoder::partials(50), (1, 160));
        // Exactly one window.
        assert_eq!(VoiceEncoder::partials(160), (1, 160));
        // 300 frames: two partials (160 + 77 = 237), and the 63-frame
        // remainder covers (63 + 83) / 160 = 0.91 ≥ 0.8 of a third.
        assert_eq!(VoiceEncoder::partials(300), (3, 314));
    }

    #[test]
    fn segment_average_matches_ceil_mode_pooling() {
        let x = Tensor::arange(0f32, 5.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 5))
            .unwrap();
        let out = segment_average(&x, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Segments [0,1] → 0.5, [2,3] → 2.5, [4] → 4.
        assert_eq!(out, vec![0.5, 0.5, 2.5, 2.5, 4.0]);
    }
}

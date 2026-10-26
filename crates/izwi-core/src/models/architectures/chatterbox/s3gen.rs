//! S3Gen's token-to-mel flow (`s3gen/flow.py`, `flow_matching.py`,
//! `decoder.py`, `transformer/upsample_encoder.py`; Apache-2.0, derived from
//! CosyVoice).
//!
//! The reference's speech tokens and the generated ones are embedded
//! together, encoded by a Conformer (six layers at 25 Hz, upsampled ×2, four
//! more at 50 Hz) into `mu`, and a causal U-Net estimator integrates a
//! flow-matching ODE from Gaussian noise to 80-bin mels in ten
//! cosine-scheduled Euler steps. Each step runs the estimator on a
//! conditional row (with `mu`, the speaker vector and the reference mels)
//! and an unconditional row (all three zero), blended with rate 0.7. The
//! reference's frames are generated too and then dropped.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv1d, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::conds::S3GenVoiceReference;
use super::config::S3GenConfig;
use super::weights::{conv1d, conv_cfg};
use crate::error::{Error, Result};

pub struct S3GenFlow {
    cfg: S3GenConfig,
    dtype: DType,
    device: Device,
    input_embedding: Embedding,
    speaker_affine: Linear,
    encoder: UpsampleConformerEncoder,
    encoder_proj: Linear,
    estimator: CausalUnetEstimator,
}

/// A voice's reference, ready on the device for every request that uses it.
#[derive(Debug, Clone)]
pub struct S3GenPreparedVoice {
    prompt_tokens: Vec<u32>,
    /// [80, prompt frames].
    prompt_mel: Tensor,
    prompt_frames: usize,
    /// Normalised, projected speaker vector, [80].
    speaker: Tensor,
}

impl S3GenPreparedVoice {
    pub fn prompt_frames(&self) -> usize {
        self.prompt_frames
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens.len()
    }

    pub fn device_bytes(&self) -> usize {
        (self.prompt_mel.elem_count() + self.speaker.elem_count())
            * self.prompt_mel.dtype().size_in_bytes()
    }
}

/// Inputs of one flow run after the encoder: everything the Euler steps
/// reuse.
pub struct S3GenFlowState {
    /// Conditional estimator inputs, [80, frames] each.
    mu: Tensor,
    cond: Tensor,
    speaker: Tensor,
    /// Current sample, [80, frames].
    x: Tensor,
    prompt_frames: usize,
    step: usize,
}

impl S3GenFlowState {
    pub fn frames(&self) -> usize {
        self.x.dim(1).unwrap_or(0)
    }

    pub fn steps_done(&self) -> usize {
        self.step
    }

    /// `mu` or the current sample `x`, [80, frames], for parity probes.
    #[cfg(test)]
    pub(crate) fn probe(&self, which: &str) -> Result<Tensor> {
        match which {
            "mu" => Ok(self.mu.clone()),
            "x" => Ok(self.x.clone()),
            other => Err(Error::InvalidInput(format!("no flow probe {other}"))),
        }
    }
}

impl S3GenFlow {
    pub fn load(cfg: S3GenConfig, vb: VarBuilder) -> Result<Self> {
        let flow = vb.pp("flow");
        Ok(Self {
            dtype: vb.dtype(),
            device: vb.device().clone(),
            input_embedding: candle_nn::embedding(
                cfg.vocab_size,
                cfg.encoder_dim,
                flow.pp("input_embedding"),
            )?,
            speaker_affine: candle_nn::linear(
                cfg.speaker_embed_size,
                cfg.mel_bins,
                flow.pp("spk_embed_affine_layer"),
            )?,
            encoder: UpsampleConformerEncoder::load(&cfg, flow.pp("encoder"))?,
            encoder_proj: candle_nn::linear(
                cfg.encoder_dim,
                cfg.mel_bins,
                flow.pp("encoder_proj"),
            )?,
            estimator: CausalUnetEstimator::load(&cfg, flow.pp("decoder.estimator"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &S3GenConfig {
        &self.cfg
    }

    pub fn prepare_voice(&self, voice: &S3GenVoiceReference) -> Result<S3GenPreparedVoice> {
        let bins = self.cfg.mel_bins;
        let prompt_mel = Tensor::from_slice(
            &voice.prompt_mel,
            (voice.prompt_mel_frames, bins),
            &self.device,
        )?
        .to_dtype(self.dtype)?
        .t()?
        .contiguous()?;
        // F.normalize over the vector: x / max(||x||, 1e-12).
        let norm = voice
            .speaker_xvector
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        let normalized = voice
            .speaker_xvector
            .iter()
            .map(|&v| (f64::from(v) / norm) as f32)
            .collect::<Vec<_>>();
        let xvector = Tensor::from_vec(normalized, (1, voice.speaker_xvector.len()), &self.device)?
            .to_dtype(self.dtype)?;
        let speaker = self.speaker_affine.forward(&xvector)?.squeeze(0)?;
        Ok(S3GenPreparedVoice {
            prompt_tokens: voice.prompt_tokens.clone(),
            prompt_mel,
            prompt_frames: voice.prompt_mel_frames,
            speaker,
        })
    }

    /// Encode the reference and generated tokens and start the ODE from
    /// `noise` (row-major [80, frames], frames = 2 × all tokens, minus the
    /// 3-token lookahead when not `finalize`).
    pub fn begin(
        &self,
        voice: &S3GenPreparedVoice,
        tokens: &[u32],
        noise: &[f32],
        finalize: bool,
    ) -> Result<S3GenFlowState> {
        if tokens.is_empty() {
            return Err(Error::InvalidInput(
                "Chatterbox S3Gen needs at least one speech token".into(),
            ));
        }
        if let Some(&code) = tokens
            .iter()
            .find(|&&code| code as usize >= self.cfg.vocab_size)
        {
            return Err(Error::InvalidInput(format!(
                "Chatterbox S3Gen speech token {code} is outside its {} codes",
                self.cfg.vocab_size
            )));
        }
        let all = voice
            .prompt_tokens
            .iter()
            .chain(tokens)
            .copied()
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(all.clone(), all.len(), &self.device)?;
        let embedded = self.input_embedding.forward(&ids)?;
        let mut h = self.encoder.forward(&embedded)?;
        if !finalize {
            let keep = h
                .dim(0)?
                .checked_sub(self.cfg.pre_lookahead_len * 2)
                .filter(|&keep| keep > voice.prompt_frames)
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Chatterbox S3Gen chunk is shorter than its lookahead".into(),
                    )
                })?;
            h = h.narrow(0, 0, keep)?;
        }
        let frames = h.dim(0)?;
        if frames <= voice.prompt_frames {
            return Err(Error::InferenceError(
                "Chatterbox S3Gen produced no frames beyond the reference".into(),
            ));
        }
        let bins = self.cfg.mel_bins;
        if noise.len() != bins * frames {
            return Err(Error::InvalidInput(format!(
                "Chatterbox S3Gen noise holds {} values, expected {}",
                noise.len(),
                bins * frames
            )));
        }
        let mu = self.encoder_proj.forward(&h)?.t()?.contiguous()?;
        let cond = Tensor::cat(
            &[
                voice.prompt_mel.clone(),
                Tensor::zeros(
                    (bins, frames - voice.prompt_frames),
                    self.dtype,
                    &self.device,
                )?,
            ],
            1,
        )?;
        let x = Tensor::from_slice(noise, (bins, frames), &self.device)?.to_dtype(self.dtype)?;
        Ok(S3GenFlowState {
            mu,
            cond,
            speaker: voice.speaker.clone(),
            x,
            prompt_frames: voice.prompt_frames,
            step: 0,
        })
    }

    /// The cosine schedule's time at `step` of `steps`.
    fn schedule(step: usize, steps: usize) -> f64 {
        let linear = step as f64 / steps as f64;
        1.0 - (linear * 0.5 * std::f64::consts::PI).cos()
    }

    /// Advance up to `steps` Euler steps; returns true when the ODE is done.
    pub fn step(&self, state: &mut S3GenFlowState, steps: usize) -> Result<bool> {
        let total = self.cfg.cfm_steps;
        let rate = f64::from(self.cfg.cfm_cfg_rate);
        for _ in 0..steps {
            if state.step >= total {
                break;
            }
            // Upstream builds t_span in the tensor's dtype (F32).
            let t = Self::schedule(state.step, total) as f32;
            let r = Self::schedule(state.step + 1, total) as f32;
            let zeros = Tensor::zeros_like(&state.mu)?;
            let x = Tensor::stack(&[&state.x, &state.x], 0)?;
            let mu = Tensor::stack(&[&state.mu, &zeros], 0)?;
            let cond = Tensor::stack(&[&state.cond, &zeros], 0)?;
            let speaker = Tensor::stack(&[&state.speaker, &state.speaker.zeros_like()?], 0)?;
            let dxdt = self
                .estimator
                .forward(&x, &mu, &speaker, &cond, f64::from(t))?;
            let guided = dxdt
                .i(0)?
                .affine(1.0 + rate, 0.0)?
                .sub(&dxdt.i(1)?.affine(rate, 0.0)?)?;
            let dt = f64::from(r - t);
            state.x = state.x.add(&guided.affine(dt, 0.0)?)?;
            state.step += 1;
        }
        Ok(state.step >= total)
    }

    /// The generated mels after the last step, [80, frames − reference].
    pub fn finish(&self, state: &S3GenFlowState) -> Result<Tensor> {
        if state.step < self.cfg.cfm_steps {
            return Err(Error::InferenceError(
                "Chatterbox S3Gen flow read before its last step".into(),
            ));
        }
        let frames = state.x.dim(1)?;
        Ok(state
            .x
            .narrow(1, state.prompt_frames, frames - state.prompt_frames)?
            .contiguous()?)
    }

    /// The flow end to end: encode, integrate, drop the reference frames.
    pub fn generate_mel(
        &self,
        voice: &S3GenPreparedVoice,
        tokens: &[u32],
        noise: &[f32],
    ) -> Result<Tensor> {
        let mut state = self.begin(voice, tokens, noise, true)?;
        self.step(&mut state, self.cfg.cfm_steps)?;
        self.finish(&state)
    }

    /// Mel frames the flow produces for `tokens` after `voice`'s reference.
    pub fn total_frames(voice_prompt_tokens: usize, tokens: usize) -> usize {
        2 * (voice_prompt_tokens + tokens)
    }
}

// ---------------------------------------------------------------- encoder

/// ESPnet's linear input layer: Linear, LayerNorm, then ×√d and the relative
/// position table.
struct LinearInput {
    linear: Linear,
    norm: LayerNorm,
    scale: f64,
}

impl LinearInput {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(dim, dim, vb.pp("out.0"))?,
            norm: candle_nn::layer_norm(dim, 1e-5, vb.pp("out.1"))?,
            scale: (dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .norm
            .forward(&self.linear.forward(x)?)?
            .affine(self.scale, 0.0)?)
    }
}

/// `EspnetRelPositionalEncoding`'s table for `len` frames: relative
/// positions `len-1 .. -(len-1)`, [2·len − 1, dim].
fn relative_positions(len: usize, dim: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let rows = 2 * len - 1;
    let mut table = vec![0f32; rows * dim];
    // torch computes div_term in F32 from the F64 scalar −ln(10000)/d.
    let rate = -(10000f64.ln() / dim as f64) as f32;
    for row in 0..rows {
        let position = (len - 1) as f32 - row as f32;
        for pair in 0..dim / 2 {
            let div = ((2 * pair) as f32 * rate).exp();
            let angle = position * div;
            table[row * dim + 2 * pair] = angle.sin();
            table[row * dim + 2 * pair + 1] = angle.cos();
        }
    }
    Ok(Tensor::from_vec(table, (rows, dim), device)?.to_dtype(dtype)?)
}

struct RelPositionAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    /// [heads, 1, head_dim] each.
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    heads: usize,
}

impl RelPositionAttention {
    fn load(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / heads;
        Ok(Self {
            q: candle_nn::linear(dim, dim, vb.pp("linear_q"))?,
            k: candle_nn::linear(dim, dim, vb.pp("linear_k"))?,
            v: candle_nn::linear(dim, dim, vb.pp("linear_v"))?,
            out: candle_nn::linear(dim, dim, vb.pp("linear_out"))?,
            pos: candle_nn::linear_no_bias(dim, dim, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((heads, head_dim), "pos_bias_u")?.unsqueeze(1)?,
            pos_bias_v: vb.get((heads, head_dim), "pos_bias_v")?.unsqueeze(1)?,
            heads,
        })
    }

    /// `x` [frames, dim], `positions` [2·frames − 1, dim]; full context.
    fn forward(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let (frames, dim) = x.dims2()?;
        let head_dim = dim / self.heads;
        let split = |t: Tensor, len: usize| -> candle_core::Result<Tensor> {
            t.reshape((len, self.heads, head_dim))?.transpose(0, 1)
        };
        let q = split(self.q.forward(x)?, frames)?;
        let k = split(self.k.forward(x)?, frames)?.contiguous()?;
        let v = split(self.v.forward(x)?, frames)?.contiguous()?;
        let p = split(self.pos.forward(positions)?, positions.dim(0)?)?.contiguous()?;
        let q_u = q.broadcast_add(&self.pos_bias_u)?.contiguous()?;
        let q_v = q.broadcast_add(&self.pos_bias_v)?.contiguous()?;
        let content = q_u.matmul(&k.t()?)?;
        let position = rel_shift(&q_v.matmul(&p.t()?)?)?;
        let scores = content
            .add(&position)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((frames, dim))?;
        Ok(self.out.forward(&attended)?)
    }
}

/// ESPnet's `rel_shift` for scores [heads, frames, 2·frames − 1]: row `i`
/// keeps columns `frames − 1 − i ..`, so column `j` scores relative
/// position `i − j`.
fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (heads, frames, width) = x.dims3()?;
    let padded = x.pad_with_zeros(2, 1, 0)?;
    let shifted = padded
        .reshape((heads, width + 1, frames))?
        .narrow(1, 1, width)?
        .reshape((heads, frames, width))?
        .narrow(2, 0, frames)?;
    Ok(shifted.contiguous()?)
}

struct ConformerLayer {
    norm_mha: LayerNorm,
    attn: RelPositionAttention,
    norm_ff: LayerNorm,
    w1: Linear,
    w2: Linear,
}

impl ConformerLayer {
    fn load(cfg: &S3GenConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.encoder_dim;
        Ok(Self {
            norm_mha: candle_nn::layer_norm(dim, 1e-12, vb.pp("norm_mha"))?,
            attn: RelPositionAttention::load(dim, cfg.encoder_heads, vb.pp("self_attn"))?,
            norm_ff: candle_nn::layer_norm(dim, 1e-12, vb.pp("norm_ff"))?,
            w1: candle_nn::linear(dim, cfg.encoder_ffn, vb.pp("feed_forward.w_1"))?,
            w2: candle_nn::linear(cfg.encoder_ffn, dim, vb.pp("feed_forward.w_2"))?,
        })
    }

    fn forward(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attn.forward(&self.norm_mha.forward(x)?, positions)?)?;
        let ff = self
            .w2
            .forward(&self.w1.forward(&self.norm_ff.forward(&x)?)?.silu()?)?;
        Ok(x.add(&ff)?)
    }
}

struct UpsampleConformerEncoder {
    embed: LinearInput,
    lookahead_conv1: Conv1d,
    lookahead_conv2: Conv1d,
    pre_lookahead: usize,
    layers: Vec<ConformerLayer>,
    up_conv: Conv1d,
    up_embed: LinearInput,
    up_layers: Vec<ConformerLayer>,
    after_norm: LayerNorm,
    dim: usize,
}

impl UpsampleConformerEncoder {
    fn load(cfg: &S3GenConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.encoder_dim;
        Ok(Self {
            embed: LinearInput::load(dim, vb.pp("embed"))?,
            lookahead_conv1: conv1d(&vb.pp("pre_lookahead_layer.conv1"), conv_cfg(0, 1, 1))?,
            lookahead_conv2: conv1d(&vb.pp("pre_lookahead_layer.conv2"), conv_cfg(0, 1, 1))?,
            pre_lookahead: cfg.pre_lookahead_len,
            layers: (0..cfg.encoder_layers)
                .map(|i| ConformerLayer::load(cfg, vb.pp(format!("encoders.{i}"))))
                .collect::<Result<Vec<_>>>()?,
            up_conv: conv1d(&vb.pp("up_layer.conv"), conv_cfg(0, 1, 1))?,
            up_embed: LinearInput::load(dim, vb.pp("up_embed"))?,
            up_layers: (0..cfg.up_encoder_layers)
                .map(|i| ConformerLayer::load(cfg, vb.pp(format!("up_encoders.{i}"))))
                .collect::<Result<Vec<_>>>()?,
            after_norm: candle_nn::layer_norm(dim, 1e-5, vb.pp("after_norm"))?,
            dim,
        })
    }

    /// `x` [tokens, dim] → [2·tokens, dim].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        let mut h = self.embed.forward(x)?;
        // Pre-lookahead: look 3 tokens ahead, then a causal kernel of 3.
        let channels = h.t()?.unsqueeze(0)?;
        let ahead =
            self.lookahead_conv1
                .forward(&channels.pad_with_zeros(2, 0, self.pre_lookahead)?)?;
        let ahead = candle_nn::ops::leaky_relu(&ahead, 0.01)?;
        let ahead = self
            .lookahead_conv2
            .forward(&ahead.pad_with_zeros(2, 2, 0)?)?;
        h = h.add(&ahead.squeeze(0)?.t()?)?;
        let positions = relative_positions(tokens, self.dim, h.dtype(), h.device())?;
        for layer in &self.layers {
            h = layer.forward(&h, &positions)?;
        }
        // ×2 nearest upsampling, then a causal kernel of 5.
        let up = h
            .t()?
            .unsqueeze(2)?
            .repeat((1, 1, 2))?
            .reshape((self.dim, 2 * tokens))?
            .unsqueeze(0)?
            .pad_with_zeros(2, 4, 0)?;
        let up = self.up_conv.forward(&up)?.squeeze(0)?.t()?;
        let mut h = self.up_embed.forward(&up)?;
        let positions = relative_positions(2 * tokens, self.dim, h.dtype(), h.device())?;
        for layer in &self.up_layers {
            h = layer.forward(&h, &positions)?;
        }
        Ok(self.after_norm.forward(&h)?)
    }
}

// --------------------------------------------------------------- estimator

fn mish(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = max(x, 0) + ln(1 + e^{−|x|}), stable for large |x|.
    let softplus = x
        .relu()?
        .add(&x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
    Ok(x.mul(&softplus.tanh()?)?)
}

/// CausalConv1d: left padding of kernel − 1.
struct CausalConv {
    conv: Conv1d,
    left: usize,
}

impl CausalConv {
    fn load(vb: &VarBuilder, kernel: usize) -> Result<Self> {
        Ok(Self {
            conv: conv1d(vb, conv_cfg(0, 1, 1))?,
            left: kernel - 1,
        })
    }

    /// `x` [batch, channels, frames].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(&x.pad_with_zeros(2, self.left, 0)?)?)
    }
}

/// Causal conv, LayerNorm over channels, Mish.
struct CausalBlock {
    conv: CausalConv,
    norm: LayerNorm,
}

impl CausalBlock {
    fn load(vb: VarBuilder, dim_out: usize) -> Result<Self> {
        Ok(Self {
            conv: CausalConv::load(&vb.pp("block.0"), 3)?,
            norm: candle_nn::layer_norm(dim_out, 1e-5, vb.pp("block.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = self.norm.forward(&h.transpose(1, 2)?)?.transpose(1, 2)?;
        mish(&h)
    }
}

struct CausalResnet {
    block1: CausalBlock,
    block2: CausalBlock,
    time: Linear,
    res: Conv1d,
}

impl CausalResnet {
    fn load(dim_out: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            block1: CausalBlock::load(vb.pp("block1"), dim_out)?,
            block2: CausalBlock::load(vb.pp("block2"), dim_out)?,
            time: candle_nn::linear(time_dim, dim_out, vb.pp("mlp.1"))?,
            res: conv1d(&vb.pp("res_conv"), conv_cfg(0, 1, 1))?,
        })
    }

    /// `time` is the Mish of the time embedding, [batch, time_dim].
    fn forward(&self, x: &Tensor, time: &Tensor) -> Result<Tensor> {
        let h = self.block1.forward(x)?;
        let h = h.broadcast_add(&self.time.forward(time)?.unsqueeze(2)?)?;
        let h = self.block2.forward(&h)?;
        Ok(h.add(&self.res.forward(x)?)?)
    }
}

/// diffusers' `BasicTransformerBlock` as the estimator configures it:
/// pre-norm self-attention (no projection bias in) and a GELU feed-forward.
struct TransformerBlock {
    norm1: LayerNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl TransformerBlock {
    fn load(cfg: &S3GenConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.estimator_channels;
        let inner = cfg.estimator_heads * cfg.estimator_head_dim;
        let attn = vb.pp("attn1");
        Ok(Self {
            norm1: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            to_q: candle_nn::linear_no_bias(dim, inner, attn.pp("to_q"))?,
            to_k: candle_nn::linear_no_bias(dim, inner, attn.pp("to_k"))?,
            to_v: candle_nn::linear_no_bias(dim, inner, attn.pp("to_v"))?,
            to_out: candle_nn::linear(inner, dim, attn.pp("to_out.0"))?,
            norm3: candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff_in: candle_nn::linear(dim, 4 * dim, vb.pp("ff.net.0.proj"))?,
            ff_out: candle_nn::linear(4 * dim, dim, vb.pp("ff.net.2"))?,
            heads: cfg.estimator_heads,
            head_dim: cfg.estimator_head_dim,
        })
    }

    /// `x` [batch, frames, dim], every frame valid.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, frames, _) = x.dims3()?;
        let normed = self.norm1.forward(x)?;
        let split = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((batch, frames, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        // The scale goes on q, not on the [heads, frames, frames] scores.
        let q = split(
            self.to_q
                .forward(&normed)?
                .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?,
        )?;
        let k = split(self.to_k.forward(&normed)?)?;
        let v = split(self.to_v.forward(&normed)?)?;
        let weights = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?)?;
        let attended = weights.matmul(&v)?.transpose(1, 2)?.reshape((
            batch,
            frames,
            self.heads * self.head_dim,
        ))?;
        let x = x.add(&self.to_out.forward(&attended)?)?;
        let ff = self
            .ff_out
            .forward(&self.ff_in.forward(&self.norm3.forward(&x)?)?.gelu_erf()?)?;
        Ok(x.add(&ff)?)
    }
}

struct UnetStage {
    resnet: CausalResnet,
    transformers: Vec<TransformerBlock>,
    /// Down/up stages end in a causal conv of 3; mid stages do not.
    tail: Option<CausalConv>,
}

impl UnetStage {
    fn load(cfg: &S3GenConfig, vb: VarBuilder, tail: bool) -> Result<Self> {
        Ok(Self {
            resnet: CausalResnet::load(cfg.estimator_channels, cfg.time_embed_dim, vb.pp("0"))?,
            transformers: (0..cfg.estimator_blocks)
                .map(|i| TransformerBlock::load(cfg, vb.pp(format!("1.{i}"))))
                .collect::<Result<Vec<_>>>()?,
            tail: tail.then(|| CausalConv::load(&vb.pp("2"), 3)).transpose()?,
        })
    }

    /// Resnet and transformers; the tail is applied by the caller.
    fn forward(&self, x: &Tensor, time: &Tensor) -> Result<Tensor> {
        let mut h = self.resnet.forward(x, time)?.transpose(1, 2)?;
        for block in &self.transformers {
            h = block.forward(&h)?;
        }
        Ok(h.transpose(1, 2)?.contiguous()?)
    }
}

/// `ConditionalDecoder(causal=True, channels=[256])`: one down stage (whose
/// "downsample" is a causal conv), twelve mid stages, one up stage over the
/// skip connection, a final causal block and a 1×1 projection to 80 bins.
struct CausalUnetEstimator {
    time_linear1: Linear,
    time_linear2: Linear,
    down: UnetStage,
    mid: Vec<UnetStage>,
    up: UnetStage,
    final_block: CausalBlock,
    final_proj: Conv1d,
    in_channels: usize,
}

impl CausalUnetEstimator {
    fn load(cfg: &S3GenConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            time_linear1: candle_nn::linear(
                cfg.estimator_in_channels,
                cfg.time_embed_dim,
                vb.pp("time_mlp.linear_1"),
            )?,
            time_linear2: candle_nn::linear(
                cfg.time_embed_dim,
                cfg.time_embed_dim,
                vb.pp("time_mlp.linear_2"),
            )?,
            down: UnetStage::load(cfg, vb.pp("down_blocks.0"), true)?,
            mid: (0..cfg.estimator_mid_blocks)
                .map(|i| UnetStage::load(cfg, vb.pp(format!("mid_blocks.{i}")), false))
                .collect::<Result<Vec<_>>>()?,
            up: UnetStage::load(cfg, vb.pp("up_blocks.0"), true)?,
            final_block: CausalBlock::load(vb.pp("final_block"), cfg.estimator_channels)?,
            final_proj: conv1d(&vb.pp("final_proj"), conv_cfg(0, 1, 1))?,
            in_channels: cfg.estimator_in_channels,
        })
    }

    /// Matcha's `SinusoidalPosEmb` at scale 1000, then the time MLP, Mished
    /// for the resnets: [batch, time_dim].
    fn time_embedding(
        &self,
        t: f64,
        batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let half = self.in_channels / 2;
        // torch: an F32 arange times the F64 scalar ln(10000) / (half − 1).
        let step = (10000f64.ln() / (half as f64 - 1.0)) as f32;
        let t = t as f32;
        let mut row = Vec::with_capacity(self.in_channels);
        let args = (0..half)
            .map(|i| 1000.0 * t * (i as f32 * -step).exp())
            .collect::<Vec<_>>();
        row.extend(args.iter().map(|a| a.sin()));
        row.extend(args.iter().map(|a| a.cos()));
        let emb = Tensor::from_vec(row, (1, self.in_channels), device)?
            .to_dtype(dtype)?
            .repeat((batch, 1))?;
        let hidden = self.time_linear1.forward(&emb)?.silu()?;
        mish(&self.time_linear2.forward(&hidden)?)
    }

    /// Velocity for `x`, `mu`, `cond` [batch, 80, frames], `speaker`
    /// [batch, 80]. Every frame is valid: a call carries one request's two
    /// guidance rows, which share their length, so upstream's padding mask
    /// is all ones and its multiplies and key bias are left out (the
    /// broadcasts cost more than the rest of the attention on Metal).
    fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        speaker: &Tensor,
        cond: &Tensor,
        t: f64,
    ) -> Result<Tensor> {
        let (batch, _, frames) = x.dims3()?;
        let time = self.time_embedding(t, batch, x.dtype(), x.device())?;
        let speaker = speaker.unsqueeze(2)?.repeat((1, 1, frames))?;
        let h = Tensor::cat(&[x, mu, &speaker, cond], 1)?;

        let skip = self.down.forward(&h, &time)?;
        let mut h = self
            .down
            .tail
            .as_ref()
            .expect("down stage has a tail")
            .forward(&skip)?;
        for stage in &self.mid {
            h = stage.forward(&h, &time)?;
        }
        let h = Tensor::cat(&[&h.narrow(2, 0, skip.dim(2)?)?, &skip], 1)?;
        let h = self.up.forward(&h, &time)?;
        let h = self
            .up
            .tail
            .as_ref()
            .expect("up stage has a tail")
            .forward(&h)?;
        let h = self.final_block.forward(&h)?;
        Ok(self.final_proj.forward(&h)?)
    }
}

#[cfg(test)]
impl S3GenFlow {
    /// One estimator call on zeros of `[batch, 80, frames]`, waited for on
    /// the device: the cost of one Euler step's network (`batch` counts
    /// both guidance rows).
    pub(crate) fn probe_estimator(&self, batch: usize, frames: usize) -> Result<()> {
        let bins = self.cfg.mel_bins;
        let zeros = Tensor::zeros((batch, bins, frames), self.dtype, &self.device)?;
        let speaker = Tensor::zeros((batch, bins), self.dtype, &self.device)?;
        self.estimator
            .forward(&zeros, &zeros, &speaker, &zeros, 0.5)?;
        self.device.synchronize()?;
        Ok(())
    }

    /// Milliseconds per call of one resnet and one transformer block on
    /// `[batch, frames]` random activations, each waited for on the device.
    pub(crate) fn probe_estimator_parts(
        &self,
        batch: usize,
        frames: usize,
        runs: usize,
    ) -> Result<Vec<(&'static str, f64)>> {
        let stage = &self.estimator.mid[0];
        let dim = self.cfg.estimator_channels;
        let channels_first = Tensor::randn(0f32, 1.0, (batch, dim, frames), &self.device)?;
        let tokens = channels_first.transpose(1, 2)?.contiguous()?;
        let time = Tensor::randn(0f32, 1.0, (batch, self.cfg.time_embed_dim), &self.device)?;
        let time_it = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
            f()?;
            self.device.synchronize()?;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                f()?;
            }
            self.device.synchronize()?;
            Ok(started.elapsed().as_secs_f64() * 1000.0 / runs as f64)
        };
        Ok(vec![
            (
                "resnet",
                time_it(&|| stage.resnet.forward(&channels_first, &time))?,
            ),
            (
                "transformer block",
                time_it(&|| stage.transformers[0].forward(&tokens))?,
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_shift_scores_relative_position_i_minus_j() {
        let frames = 4usize;
        let width = 2 * frames - 1;
        // Column c of the unshifted scores encodes relative position
        // frames − 1 − c; fill each with that position.
        let values = (0..frames)
            .flat_map(|_| (0..width).map(|c| (frames as f32 - 1.0) - c as f32))
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(values, (1, frames, width), &Device::Cpu).unwrap();
        let shifted = rel_shift(&x).unwrap().squeeze(0).unwrap();
        let rows = shifted.to_vec2::<f32>().unwrap();
        for (i, row) in rows.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_eq!(*value, i as f32 - j as f32, "({i}, {j})");
            }
        }
    }

    #[test]
    fn mish_matches_its_definition_and_saturates() {
        let x = Tensor::new(&[-30f32, -1.0, 0.0, 0.5, 30.0], &Device::Cpu).unwrap();
        let out = mish(&x).unwrap().to_vec1::<f32>().unwrap();
        for (input, actual) in [-30f32, -1.0, 0.0, 0.5, 30.0].iter().zip(out) {
            let expected = input * ((1.0 + input.exp()).ln()).tanh();
            assert!(
                (actual - expected).abs() < 1e-6,
                "{input}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn relative_position_table_is_centred_on_zero() {
        let table = relative_positions(3, 4, DType::F32, &Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(table.len(), 5);
        // Row 2 is relative position 0: sin 0 = 0, cos 0 = 1.
        assert_eq!(table[2], vec![0.0, 1.0, 0.0, 1.0]);
        // Rows 0 and 4 are positions +2 and −2: sines negate, cosines agree.
        assert!((table[0][0] + table[4][0]).abs() < 1e-7);
        assert!((table[0][1] - table[4][1]).abs() < 1e-7);
    }

    #[test]
    fn cosine_schedule_spans_zero_to_one() {
        assert_eq!(S3GenFlow::schedule(0, 10), 0.0);
        assert!((S3GenFlow::schedule(10, 10) - 1.0).abs() < 1e-12);
        assert!(S3GenFlow::schedule(1, 10) < 0.1 / 2.0);
    }
}

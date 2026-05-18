//! Speech Tokenizer Decoder for Qwen3-TTS.
//!
//! Converts multi-codebook RVQ codec tokens to waveform audio.
//! Implementation mirrors the official 12Hz decoder architecture:
//! RVQ projection -> pre-conv -> pre-transformer -> upsample stack -> decoder blocks -> waveform.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, LayerNormConfig,
    Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::Deserialize;
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};

/// Speech Tokenizer Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub decoder_config: DecoderConfig,
    pub encoder_config: Option<serde_json::Value>,
}

/// Encoder configuration used for reference-audio tokenization.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub attention_bias: bool,
    pub audio_channels: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub compress: usize,
    pub dilation_growth_rate: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub layer_scale_initial_scale: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_filters: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_residual_layers: usize,
    pub num_semantic_quantizers: usize,
    pub pad_mode: String,
    pub residual_kernel_size: usize,
    pub rope_theta: f64,
    pub upsampling_ratios: Vec<usize>,
    pub use_causal_conv: bool,
    pub use_conv_shortcut: bool,
    pub vector_quantization_hidden_dimension: usize,
    #[serde(rename = "_frame_rate")]
    pub frame_rate: f64,
    #[serde(default = "default_encodec_frame_rate")]
    pub encodec_frame_rate: f64,
}

fn default_encodec_frame_rate() -> f64 {
    25.0
}

/// Decoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub semantic_codebook_size: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

/// Causal Conv1d with left padding only.
struct CausalConv1d {
    conv: Conv1d,
    causal_padding: usize,
}

impl CausalConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if with_bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, config, vb)?
        };
        Ok(Self {
            conv,
            causal_padding: dilation * (kernel_size - 1),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.causal_padding > 0 {
            x.pad_with_zeros(2, self.causal_padding, 0)?
        } else {
            x.clone()
        };
        self.conv.forward(&x).map_err(Error::from)
    }
}

/// Causal transposed conv that trims the right side to preserve exact stride upsampling.
struct CausalTransConv1d {
    conv: ConvTranspose1d,
    right_trim: usize,
}

impl CausalTransConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, config, vb)?;
        Ok(Self {
            conv,
            right_trim: kernel_size.saturating_sub(stride),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(x)?;
        if self.right_trim == 0 {
            return Ok(out);
        }
        let out_len = out.dim(2)?;
        let keep = out_len.saturating_sub(self.right_trim);
        out.narrow(2, 0, keep).map_err(Error::from)
    }
}

#[derive(Clone, Copy)]
enum PadMode {
    Constant,
    Replicate,
}

impl PadMode {
    fn from_str(value: &str) -> Self {
        match value {
            "replicate" => Self::Replicate,
            _ => Self::Constant,
        }
    }
}

/// Mimi-style 1D conv with causal/extra padding logic used by the tokenizer encoder.
struct MimiConv1d {
    conv: Conv1d,
    causal: bool,
    pad_mode: PadMode,
    effective_kernel: usize,
    stride: usize,
    padding_total: usize,
    padding_left: usize,
    padding_right: usize,
}

impl MimiConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
        causal: bool,
        pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if with_bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };
        let effective_kernel = dilation * (kernel_size - 1) + 1;
        let padding_total = effective_kernel.saturating_sub(stride);
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        Ok(Self {
            conv,
            causal,
            pad_mode,
            effective_kernel,
            stride,
            padding_total,
            padding_left,
            padding_right,
        })
    }

    fn extra_padding(&self, input_len: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        let n_frames = ((input_len as f64 - self.effective_kernel as f64
            + self.padding_total as f64)
            / self.stride as f64
            + 1.0)
            .ceil()
            .max(1.0);
        let ideal_len = ((n_frames as usize - 1) * self.stride)
            + (self.effective_kernel.saturating_sub(self.padding_total));
        ideal_len.saturating_sub(input_len)
    }

    fn pad_replicate(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
        if left == 0 && right == 0 {
            return Ok(x.clone());
        }
        let len = x.dim(2)?;
        if len == 0 {
            return x.pad_with_zeros(2, left, right).map_err(Error::from);
        }

        let mut chunks: Vec<Tensor> = Vec::with_capacity(3);
        if left > 0 {
            let first = x.narrow(2, 0, 1)?;
            let mut left_parts = Vec::with_capacity(left);
            for _ in 0..left {
                left_parts.push(first.clone());
            }
            chunks.push(Tensor::cat(&left_parts, 2)?);
        }
        chunks.push(x.clone());
        if right > 0 {
            let last = x.narrow(2, len - 1, 1)?;
            let mut right_parts = Vec::with_capacity(right);
            for _ in 0..right {
                right_parts.push(last.clone());
            }
            chunks.push(Tensor::cat(&right_parts, 2)?);
        }
        Tensor::cat(&chunks, 2).map_err(Error::from)
    }

    fn pad_input(&self, x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
        match self.pad_mode {
            PadMode::Constant => x.pad_with_zeros(2, left, right).map_err(Error::from),
            PadMode::Replicate => Self::pad_replicate(x, left, right),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_len = x.dim(2)?;
        let extra = self.extra_padding(input_len);
        let (left, right) = if self.causal {
            (self.padding_total, extra)
        } else {
            (self.padding_left, self.padding_right + extra)
        };
        let x = self.pad_input(x, left, right)?;
        self.conv.forward(&x).map_err(Error::from)
    }
}

/// Mimi encoder residual block.
struct MimiResnetBlock {
    conv1: MimiConv1d,
    conv2: MimiConv1d,
    shortcut: Option<MimiConv1d>,
}

impl MimiResnetBlock {
    fn load(
        cfg: &EncoderConfig,
        dim: usize,
        dilations: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = dim / cfg.compress.max(1);
        let pad_mode = PadMode::from_str(&cfg.pad_mode);
        Ok(Self {
            conv1: MimiConv1d::load(
                dim,
                hidden,
                cfg.residual_kernel_size,
                1,
                dilations.0,
                1,
                true,
                cfg.use_causal_conv,
                pad_mode,
                vb.pp("block.1.conv"),
            )?,
            conv2: MimiConv1d::load(
                hidden,
                dim,
                1,
                1,
                dilations.1,
                1,
                true,
                cfg.use_causal_conv,
                pad_mode,
                vb.pp("block.3.conv"),
            )?,
            shortcut: if cfg.use_conv_shortcut {
                Some(MimiConv1d::load(
                    dim,
                    dim,
                    1,
                    1,
                    1,
                    1,
                    true,
                    cfg.use_causal_conv,
                    pad_mode,
                    vb.pp("shortcut.conv"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(shortcut) = &self.shortcut {
            shortcut.forward(x)?
        } else {
            x.clone()
        };
        let hidden = x.elu(1.0)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = hidden.elu(1.0)?;
        let hidden = self.conv2.forward(&hidden)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

enum EncoderModule {
    Conv(MimiConv1d),
    Resnet(MimiResnetBlock),
    Elu,
}

struct SpeechTokenizerEncoderNet {
    layers: Vec<EncoderModule>,
}

impl SpeechTokenizerEncoderNet {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut layer_idx = 0usize;
        let pad_mode = PadMode::from_str(&cfg.pad_mode);

        layers.push(EncoderModule::Conv(MimiConv1d::load(
            cfg.audio_channels,
            cfg.num_filters,
            cfg.kernel_size,
            1,
            1,
            1,
            true,
            cfg.use_causal_conv,
            pad_mode,
            vb.pp(format!("encoder.layers.{layer_idx}.conv")),
        )?));
        layer_idx += 1;

        let mut scaling = 1usize;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let current_scale = scaling * cfg.num_filters;
            for j in 0..cfg.num_residual_layers {
                layers.push(EncoderModule::Resnet(MimiResnetBlock::load(
                    cfg,
                    current_scale,
                    (cfg.dilation_growth_rate.pow(j as u32), 1),
                    vb.pp(format!("encoder.layers.{layer_idx}")),
                )?));
                layer_idx += 1;
            }
            layers.push(EncoderModule::Elu);
            layer_idx += 1;
            layers.push(EncoderModule::Conv(MimiConv1d::load(
                current_scale,
                current_scale * 2,
                ratio * 2,
                ratio,
                1,
                1,
                true,
                cfg.use_causal_conv,
                pad_mode,
                vb.pp(format!("encoder.layers.{layer_idx}.conv")),
            )?));
            layer_idx += 1;
            scaling *= 2;
        }

        layers.push(EncoderModule::Elu);
        layer_idx += 1;
        layers.push(EncoderModule::Conv(MimiConv1d::load(
            scaling * cfg.num_filters,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            1,
            1,
            true,
            cfg.use_causal_conv,
            pad_mode,
            vb.pp(format!("encoder.layers.{layer_idx}.conv")),
        )?));

        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = match layer {
                EncoderModule::Conv(conv) => conv.forward(&hidden)?,
                EncoderModule::Resnet(block) => block.forward(&hidden)?,
                EncoderModule::Elu => hidden.elu(1.0)?,
            };
        }
        Ok(hidden)
    }
}

struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl EncoderAttention {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = if cfg.attention_bias {
            candle_nn::linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("self_attn.q_proj"),
            )?
        } else {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("self_attn.q_proj"),
            )?
        };
        let k_proj = if cfg.attention_bias {
            candle_nn::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.k_proj"),
            )?
        } else {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.k_proj"),
            )?
        };
        let v_proj = if cfg.attention_bias {
            candle_nn::linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.v_proj"),
            )?
        } else {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.v_proj"),
            )?
        };
        let o_proj = if cfg.attention_bias {
            candle_nn::linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("self_attn.o_proj"),
            )?
        } else {
            candle_nn::linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("self_attn.o_proj"),
            )?
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn apply_rope(&self, x: Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            0,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;

        let cos = Tensor::cat(&[cos.clone(), cos], 1)?;
        let sin = Tensor::cat(&[sin.clone(), sin], 1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let neg = Tensor::new(-1.0f32, x.device())?.to_dtype(x.dtype())?;
        let rotated = Tensor::cat(&[x2.broadcast_mul(&neg)?, x1], 3)?;

        x.broadcast_mul(&cos)?
            .broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;

        q = self.apply_rope(q)?;
        k = self.apply_rope(k)?;

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let total_len = k.dim(2)?;
        if flash_attention_requested() && total_len == seq_len {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, true)?
            {
                let out = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&out).map_err(Error::from);
            }
        }
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;
        let mask = causal_mask(seq_len, total_len, 0, att.device(), att.dtype())?;
        att = att.broadcast_add(&mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

struct EncoderMlp {
    fc1: Linear,
    fc2: Linear,
}

impl EncoderMlp {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.fc1"),
            )?,
            fc2: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.fc2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?;
        let hidden = hidden.gelu_erf()?;
        self.fc2.forward(&hidden).map_err(Error::from)
    }
}

struct EncoderTransformerLayer {
    input_layernorm: LayerNorm,
    self_attn: EncoderAttention,
    self_attn_layer_scale: Tensor,
    post_attention_layernorm: LayerNorm,
    mlp: EncoderMlp,
    mlp_layer_scale: Tensor,
}

impl EncoderTransformerLayer {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            LayerNormConfig {
                eps: cfg.norm_eps,
                ..Default::default()
            },
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            LayerNormConfig {
                eps: cfg.norm_eps,
                ..Default::default()
            },
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            input_layernorm,
            self_attn: EncoderAttention::load(cfg, vb.clone())?,
            self_attn_layer_scale: vb
                .pp("self_attn_layer_scale")
                .get((cfg.hidden_size,), "scale")?,
            post_attention_layernorm,
            mlp: EncoderMlp::load(cfg, vb.clone())?,
            mlp_layer_scale: vb.pp("mlp_layer_scale").get((cfg.hidden_size,), "scale")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed)?;
        let attn = attn.broadcast_mul(&self.self_attn_layer_scale)?;
        let x = x.broadcast_add(&attn)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = mlp_out.broadcast_mul(&self.mlp_layer_scale)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

struct EncoderTransformer {
    layers: Vec<EncoderTransformerLayer>,
}

impl EncoderTransformer {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(EncoderTransformerLayer::load(
                cfg,
                vb.pp(format!("encoder_transformer.layers.{idx}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

/// SnakeBeta activation.
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((channels,), "alpha")?,
            beta: vb.get((channels,), "beta")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let sin2 = x.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let eps = Tensor::new(1e-9f32, x.device())?.to_dtype(beta.dtype())?;
        let inv_beta = beta.broadcast_add(&eps)?.recip()?;
        x.broadcast_add(&sin2.broadcast_mul(&inv_beta)?)
            .map_err(Error::from)
    }
}

/// Residual unit used inside decoder blocks.
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::load(channels, vb.pp("act1"))?,
            conv1: CausalConv1d::load(
                channels,
                channels,
                7,
                dilation,
                1,
                true,
                vb.pp("conv1.conv"),
            )?,
            act2: SnakeBeta::load(channels, vb.pp("act2"))?,
            conv2: CausalConv1d::load(channels, channels, 1, 1, 1, true, vb.pp("conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = self.act1.forward(x)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

/// BigVGAN-style decoder block.
struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalTransConv1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            snake: SnakeBeta::load(in_channels, vb.pp("block.0"))?,
            upsample: CausalTransConv1d::load(
                in_channels,
                out_channels,
                upsample_rate * 2,
                upsample_rate,
                vb.pp("block.1.conv"),
            )?,
            res1: ResidualUnit::load(out_channels, 1, vb.pp("block.2"))?,
            res2: ResidualUnit::load(out_channels, 3, vb.pp("block.3"))?,
            res3: ResidualUnit::load(out_channels, 9, vb.pp("block.4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.snake.forward(x)?;
        let hidden = self.upsample.forward(&hidden)?;
        let hidden = self.res1.forward(&hidden)?;
        let hidden = self.res2.forward(&hidden)?;
        self.res3.forward(&hidden)
    }
}

/// ConvNeXt block used after upsample transposed conv.
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::load(dim, dim, 7, 1, dim, true, vb.pp("dwconv.conv"))?;
        let norm = candle_nn::layer_norm(
            dim,
            LayerNormConfig {
                eps: 1e-6,
                ..Default::default()
            },
            vb.pp("norm"),
        )?;
        let pwconv1 = candle_nn::linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get((dim,), "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = self.dwconv.forward(x)?;
        let hidden = hidden.transpose(1, 2)?;
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.pwconv1.forward(&hidden)?;
        let hidden = hidden.gelu_erf()?;
        let hidden = self.pwconv2.forward(&hidden)?;
        let hidden = hidden.broadcast_mul(&self.gamma)?;
        let hidden = hidden.transpose(1, 2)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

/// Upsample stage: transposed conv + ConvNeXt block.
struct UpsampleStage {
    trans_conv: CausalTransConv1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn load(channels: usize, ratio: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            trans_conv: CausalTransConv1d::load(channels, channels, ratio, ratio, vb.pp("0.conv"))?,
            convnext: ConvNeXtBlock::load(channels, vb.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.trans_conv.forward(x)?;
        self.convnext.forward(&hidden)
    }
}

/// Multi-head attention for pre-transformer.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl Attention {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: candle_nn::linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn apply_rope(&self, x: Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            0,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;

        let cos = Tensor::cat(&[cos.clone(), cos], 1)?;
        let sin = Tensor::cat(&[sin.clone(), sin], 1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let neg = Tensor::new(-1.0f32, x.device())?.to_dtype(x.dtype())?;
        let rotated = Tensor::cat(&[x2.broadcast_mul(&neg)?, x1], 3)?;

        x.broadcast_mul(&cos)?
            .broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;

        q = self.apply_rope(q)?;
        k = self.apply_rope(k)?;

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let total_len = k.dim(2)?;
        if flash_attention_requested() && total_len == seq_len {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, true)?
            {
                let out = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&out).map_err(Error::from);
            }
        }

        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;

        let mask = causal_mask(seq_len, total_len, 0, att.device(), att.dtype())?;
        att = att.broadcast_add(&mask)?;

        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&out).map_err(Error::from)
    }
}

/// SwiGLU MLP block.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Pre-transformer layer with official layer scales.
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    self_attn_layer_scale: Tensor,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    mlp_layer_scale: Tensor,
}

impl DecoderLayer {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::load(cfg, vb.pp("self_attn"))?,
            self_attn_layer_scale: vb
                .pp("self_attn_layer_scale")
                .get((cfg.hidden_size,), "scale")?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            mlp_layer_scale: vb.pp("mlp_layer_scale").get((cfg.hidden_size,), "scale")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let attn_out = attn_out.broadcast_mul(&self.self_attn_layer_scale)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = mlp_out.broadcast_mul(&self.mlp_layer_scale)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Encoder-side normalized codebook.
struct EncoderCodebook {
    embeddings: Vec<f32>,
    norms: Vec<f32>,
    codebook_size: usize,
    codebook_dim: usize,
}

impl EncoderCodebook {
    fn load(vb: VarBuilder, codebook_size: usize, codebook_dim: usize) -> Result<Self> {
        let embedding_sum = vb.get((codebook_size, codebook_dim), "codebook.embed_sum")?;
        let cluster_usage = vb
            .get((codebook_size,), "codebook.cluster_usage")?
            .clamp(1e-7f64, f64::MAX)?;
        let embeddings = embedding_sum.broadcast_div(&cluster_usage.unsqueeze(1)?)?;
        let embeddings_2d = embeddings.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut flat = Vec::with_capacity(codebook_size * codebook_dim);
        let mut norms = Vec::with_capacity(codebook_size);
        for row in &embeddings_2d {
            norms.push(row.iter().map(|v| v * v).sum());
            flat.extend_from_slice(row);
        }
        Ok(Self {
            embeddings: flat,
            norms,
            codebook_size,
            codebook_dim,
        })
    }

    fn embedding(&self, index: usize) -> &[f32] {
        let start = index * self.codebook_dim;
        &self.embeddings[start..start + self.codebook_dim]
    }

    fn nearest_index(&self, frame: &[f32]) -> usize {
        let frame_norm: f32 = frame.iter().map(|v| v * v).sum();
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for idx in 0..self.codebook_size {
            let emb = self.embedding(idx);
            let dot: f32 = frame.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
            let dist = frame_norm + self.norms[idx] - 2.0 * dot;
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx
    }
}

/// Encoder residual vector quantizer encode path.
struct EncoderResidualVectorQuantizer {
    input_proj: Option<Conv1d>,
    codebooks: Vec<EncoderCodebook>,
    projected_dim: usize,
}

impl EncoderResidualVectorQuantizer {
    fn load(cfg: &EncoderConfig, num_quantizers: usize, vb: VarBuilder) -> Result<Self> {
        let projected_dim = cfg.vector_quantization_hidden_dimension;
        let input_proj = if projected_dim != cfg.hidden_size {
            Some(candle_nn::conv1d_no_bias(
                cfg.hidden_size,
                projected_dim,
                1,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                },
                vb.pp("input_proj"),
            )?)
        } else {
            None
        };

        let mut codebooks = Vec::with_capacity(num_quantizers);
        for idx in 0..num_quantizers {
            codebooks.push(EncoderCodebook::load(
                vb.pp(format!("layers.{idx}")),
                cfg.codebook_size,
                cfg.codebook_dim,
            )?);
        }

        Ok(Self {
            input_proj,
            codebooks,
            projected_dim,
        })
    }

    fn encode(&self, embeddings: &Tensor, num_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        let mut residual = if let Some(input_proj) = &self.input_proj {
            input_proj.forward(embeddings)?
        } else {
            embeddings.clone()
        };

        let (batch, dim, seq_len) = residual.dims3()?;
        if batch != 1 {
            return Err(Error::InferenceError(
                "Speech tokenizer encoder currently supports batch_size=1".to_string(),
            ));
        }
        if dim != self.projected_dim {
            return Err(Error::InferenceError(format!(
                "Unexpected encoder projection dim: got {}, expected {}",
                dim, self.projected_dim
            )));
        }

        let mut all_indices = Vec::with_capacity(num_quantizers);
        for layer in self.codebooks.iter().take(num_quantizers) {
            let residual_bt = residual.i(0)?.transpose(0, 1)?;
            let frames = residual_bt.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let mut indices = Vec::with_capacity(seq_len);
            let mut quantized = vec![0f32; dim * seq_len];

            for (t, frame) in frames.iter().enumerate() {
                let idx = layer.nearest_index(frame);
                indices.push(idx as u32);
                let emb = layer.embedding(idx);
                for d in 0..dim {
                    quantized[d * seq_len + t] = emb[d];
                }
            }

            let quantized = Tensor::from_vec(quantized, (1, dim, seq_len), residual.device())?
                .to_dtype(residual.dtype())?;
            residual = residual.broadcast_sub(&quantized)?;
            all_indices.push(indices);
        }
        Ok(all_indices)
    }
}

/// Split RVQ encode path (semantic + acoustic branches).
struct EncoderSplitResidualVectorQuantizer {
    semantic: EncoderResidualVectorQuantizer,
    acoustic: EncoderResidualVectorQuantizer,
    num_semantic_quantizers: usize,
    max_num_quantizers: usize,
}

impl EncoderSplitResidualVectorQuantizer {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let num_semantic = cfg.num_semantic_quantizers;
        let num_acoustic = cfg.num_quantizers.saturating_sub(num_semantic);
        Ok(Self {
            semantic: EncoderResidualVectorQuantizer::load(
                cfg,
                num_semantic,
                vb.pp("semantic_residual_vector_quantizer"),
            )?,
            acoustic: EncoderResidualVectorQuantizer::load(
                cfg,
                num_acoustic,
                vb.pp("acoustic_residual_vector_quantizer"),
            )?,
            num_semantic_quantizers: num_semantic,
            max_num_quantizers: cfg.num_quantizers,
        })
    }

    fn encode(&self, embeddings: &Tensor, num_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        if num_quantizers < self.num_semantic_quantizers || num_quantizers > self.max_num_quantizers
        {
            return Err(Error::InferenceError(format!(
                "Invalid quantizer count {} (valid range: {}..={})",
                num_quantizers, self.num_semantic_quantizers, self.max_num_quantizers
            )));
        }

        let mut codes = self
            .semantic
            .encode(embeddings, self.num_semantic_quantizers)?;
        if num_quantizers > self.num_semantic_quantizers {
            let acoustic = self
                .acoustic
                .encode(embeddings, num_quantizers - self.num_semantic_quantizers)?;
            codes.extend(acoustic);
        }
        Ok(codes)
    }
}

struct SpeechTokenizerEncoder {
    encoder_net: SpeechTokenizerEncoderNet,
    encoder_transformer: EncoderTransformer,
    downsample: Option<MimiConv1d>,
    quantizer: EncoderSplitResidualVectorQuantizer,
    config: EncoderConfig,
    device: Device,
    dtype: DType,
}

impl SpeechTokenizerEncoder {
    fn load(config: EncoderConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        let encoder_net = SpeechTokenizerEncoderNet::load(&config, vb.clone())?;
        let encoder_transformer = EncoderTransformer::load(&config, vb.clone())?;

        let downsample = if (config.encodec_frame_rate - config.frame_rate).abs() > f64::EPSILON {
            let ratio = (config.encodec_frame_rate / config.frame_rate)
                .round()
                .max(1.0) as usize;
            Some(MimiConv1d::load(
                config.hidden_size,
                config.hidden_size,
                2 * ratio,
                2,
                1,
                1,
                false,
                config.use_causal_conv,
                PadMode::Replicate,
                vb.pp("downsample.conv"),
            )?)
        } else {
            None
        };

        let quantizer = EncoderSplitResidualVectorQuantizer::load(&config, vb.pp("quantizer"))?;
        Ok(Self {
            encoder_net,
            encoder_transformer,
            downsample,
            quantizer,
            config,
            device,
            dtype: vb.dtype(),
        })
    }

    fn encode(&self, audio_samples: &[f32], num_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        if audio_samples.is_empty() {
            return Ok(vec![Vec::new(); num_quantizers]);
        }

        let mut hidden = Tensor::from_vec(
            audio_samples.to_vec(),
            (1, self.config.audio_channels, audio_samples.len()),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        hidden = self.encoder_net.forward(&hidden)?;
        hidden = hidden.transpose(1, 2)?;
        hidden = self.encoder_transformer.forward(&hidden)?;
        hidden = hidden.transpose(1, 2)?;
        if let Some(downsample) = &self.downsample {
            hidden = downsample.forward(&hidden)?;
        }
        self.quantizer.encode(&hidden, num_quantizers)
    }
}

/// Normalized RVQ codebook (embedding_sum / cluster_usage).
struct RVQCodebook {
    embeddings: Tensor,
    codebook_size: usize,
    codebook_dim: usize,
}

impl RVQCodebook {
    fn load(
        vb: VarBuilder,
        layer_idx: usize,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let embedding_sum = vb.get(
            (codebook_size, codebook_dim),
            &format!("vq.layers.{layer_idx}._codebook.embedding_sum"),
        )?;
        let cluster_usage = vb.get(
            (codebook_size,),
            &format!("vq.layers.{layer_idx}._codebook.cluster_usage"),
        )?;
        let cluster_usage = cluster_usage.clamp(1e-7f64, f64::MAX)?;
        let embeddings = embedding_sum.broadcast_div(&cluster_usage.unsqueeze(1)?)?;
        Ok(Self {
            embeddings,
            codebook_size,
            codebook_dim,
        })
    }

    fn lookup(&self, indices: &Tensor) -> Result<Tensor> {
        let original_shape = indices.dims().to_vec();
        let flat_indices = indices.flatten(0, original_shape.len() - 1)?;
        let embeddings = self.embeddings.index_select(&flat_indices, 0)?;
        let mut target_shape = original_shape;
        target_shape.push(self.codebook_dim);
        embeddings
            .reshape(target_shape.as_slice())
            .map_err(Error::from)
    }
}

/// First (semantic) quantizer path.
struct RVQFirstQuantizer {
    codebook: RVQCodebook,
    output_proj: Conv1d,
}

impl RVQFirstQuantizer {
    fn load(
        vb: VarBuilder,
        codebook_size: usize,
        codebook_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let codebook = RVQCodebook::load(vb.clone(), 0, codebook_size, codebook_dim)?;
        let output_proj = candle_nn::conv1d_no_bias(
            codebook_dim,
            hidden_size,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebook,
            output_proj,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let embeddings = self.codebook.lookup(codes)?;
        let embeddings =
            embeddings.reshape((codes.dim(0)?, codes.dim(1)?, self.codebook.codebook_dim))?;
        let embeddings = embeddings.transpose(1, 2)?;
        self.output_proj.forward(&embeddings).map_err(Error::from)
    }
}

/// Rest (acoustic) quantizer path over 15 residual codebooks.
struct RVQRestQuantizer {
    codebooks: Vec<RVQCodebook>,
    output_proj: Conv1d,
}

impl RVQRestQuantizer {
    fn load(
        vb: VarBuilder,
        num_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(num_codebooks);
        for i in 0..num_codebooks {
            codebooks.push(RVQCodebook::load(
                vb.clone(),
                i,
                codebook_size,
                codebook_dim,
            )?);
        }
        let output_proj = candle_nn::conv1d_no_bias(
            codebook_dim,
            hidden_size,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    fn decode(&self, codec_tokens: &[Vec<u32>], seq_len: usize, device: &Device) -> Result<Tensor> {
        let codebook = &self.codebooks[0];
        let codebook_dim = codebook.codebook_dim;
        let mut rest_embed = Tensor::zeros(
            (1, seq_len, codebook_dim),
            codebook.embeddings.dtype(),
            device,
        )?;
        let mut normalized_codes = Vec::with_capacity(self.codebooks.len() * seq_len);
        for (idx, codebook) in self.codebooks.iter().enumerate() {
            let group_tokens = codec_tokens.get(idx + 1);
            normalized_codes.extend(normalized_codec_indices(
                group_tokens.map(Vec::as_slice),
                seq_len,
                codebook.codebook_size,
            ));
        }
        let normalized_codes =
            Tensor::from_vec(normalized_codes, (self.codebooks.len(), seq_len), device)?;

        for (idx, codebook) in self.codebooks.iter().enumerate() {
            let codes = normalized_codes.i(idx)?.unsqueeze(0)?;
            let embed = codebook.lookup(&codes)?;
            rest_embed = rest_embed.broadcast_add(&embed)?;
        }
        let rest_embed = rest_embed.transpose(1, 2)?;
        self.output_proj.forward(&rest_embed).map_err(Error::from)
    }
}

/// Speech tokenizer decoder model.
pub struct SpeechTokenizerDecoder {
    encoder: Option<SpeechTokenizerEncoder>,
    rvq_first: RVQFirstQuantizer,
    rvq_rest: RVQRestQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer_input_proj: Linear,
    pre_transformer_layers: Vec<DecoderLayer>,
    pre_transformer_norm: RmsNorm,
    pre_transformer_output_proj: Linear,
    upsample_stages: Vec<UpsampleStage>,
    decoder_init_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv1d,
    device: Device,
    encoder_valid_num_quantizers: usize,
    input_sample_rate: usize,
    encode_downsample_rate: usize,
    decode_upsample_rate: usize,
    config: DecoderConfig,
}

impl SpeechTokenizerDecoder {
    /// Load decoder from speech_tokenizer directory.
    pub fn load(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        info!("Loading speech tokenizer decoder from {:?}", model_dir);

        let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let speech_cfg: SpeechTokenizerConfig = serde_json::from_str(&config_str)?;
        let encoder_valid_num_quantizers = speech_cfg.encoder_valid_num_quantizers;
        let input_sample_rate = speech_cfg.input_sample_rate;
        let encode_downsample_rate = speech_cfg.encode_downsample_rate;
        let decode_upsample_rate = speech_cfg.decode_upsample_rate;
        let decoder_config = speech_cfg.decoder_config;
        let encoder_config: Option<EncoderConfig> = speech_cfg
            .encoder_config
            .map(serde_json::from_value)
            .transpose()?;

        let weights_path = model_dir.join("model.safetensors");
        let root_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
        let vb = root_vb.pp("decoder");

        // Official checkpoints store RVQ embeddings at 256 dims.
        let rvq_codebook_dim = decoder_config.codebook_dim / 2;

        let rvq_first = RVQFirstQuantizer::load(
            vb.pp("quantizer.rvq_first"),
            decoder_config.codebook_size,
            rvq_codebook_dim,
            decoder_config.hidden_size,
        )?;
        let rvq_rest = RVQRestQuantizer::load(
            vb.pp("quantizer.rvq_rest"),
            decoder_config.num_quantizers.saturating_sub(1),
            decoder_config.codebook_size,
            rvq_codebook_dim,
            decoder_config.hidden_size,
        )?;

        let pre_conv = CausalConv1d::load(
            decoder_config.hidden_size,
            decoder_config.latent_dim,
            3,
            1,
            1,
            true,
            vb.pp("pre_conv.conv"),
        )?;
        let pre_transformer_input_proj = candle_nn::linear(
            decoder_config.latent_dim,
            decoder_config.hidden_size,
            vb.pp("pre_transformer.input_proj"),
        )?;

        let mut pre_transformer_layers = Vec::with_capacity(decoder_config.num_hidden_layers);
        for idx in 0..decoder_config.num_hidden_layers {
            pre_transformer_layers.push(DecoderLayer::load(
                &decoder_config,
                vb.pp(format!("pre_transformer.layers.{idx}")),
            )?);
        }
        let pre_transformer_norm = candle_nn::rms_norm(
            decoder_config.hidden_size,
            decoder_config.rms_norm_eps,
            vb.pp("pre_transformer.norm"),
        )?;
        let pre_transformer_output_proj = candle_nn::linear(
            decoder_config.hidden_size,
            decoder_config.latent_dim,
            vb.pp("pre_transformer.output_proj"),
        )?;

        let mut upsample_stages = Vec::with_capacity(decoder_config.upsampling_ratios.len());
        for (idx, &ratio) in decoder_config.upsampling_ratios.iter().enumerate() {
            upsample_stages.push(UpsampleStage::load(
                decoder_config.latent_dim,
                ratio,
                vb.pp(format!("upsample.{idx}")),
            )?);
        }

        let decoder_init_conv = CausalConv1d::load(
            decoder_config.latent_dim,
            decoder_config.decoder_dim,
            7,
            1,
            1,
            true,
            vb.pp("decoder.0.conv"),
        )?;

        let mut decoder_blocks = Vec::with_capacity(decoder_config.upsample_rates.len());
        let mut in_channels = decoder_config.decoder_dim;
        for (idx, &rate) in decoder_config.upsample_rates.iter().enumerate() {
            let out_channels = in_channels / 2;
            decoder_blocks.push(DecoderBlock::load(
                in_channels,
                out_channels,
                rate,
                vb.pp(format!("decoder.{}", idx + 1)),
            )?);
            in_channels = out_channels;
        }

        let final_snake = SnakeBeta::load(in_channels, vb.pp("decoder.5"))?;
        let final_conv =
            CausalConv1d::load(in_channels, 1, 7, 1, 1, true, vb.pp("decoder.6.conv"))?;

        let encoder = if let Some(cfg) = encoder_config {
            info!("Loading speech tokenizer encoder...");
            Some(SpeechTokenizerEncoder::load(
                cfg,
                root_vb.pp("encoder"),
                device.clone(),
            )?)
        } else {
            None
        };

        Ok(Self {
            encoder,
            rvq_first,
            rvq_rest,
            pre_conv,
            pre_transformer_input_proj,
            pre_transformer_layers,
            pre_transformer_norm,
            pre_transformer_output_proj,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
            device,
            encoder_valid_num_quantizers,
            input_sample_rate,
            encode_downsample_rate,
            decode_upsample_rate,
            config: decoder_config,
        })
    }

    /// Encode reference waveform into speech-tokenizer codec groups.
    pub fn encode_reference_audio(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Vec<u32>>> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            Error::ModelError(
                "Speech tokenizer encoder is unavailable for voice cloning".to_string(),
            )
        })?;
        if audio_samples.is_empty() {
            return Ok(Vec::new());
        }

        let target_rate = self.input_sample_rate as u32;
        let resampled = resample_audio_linear(audio_samples, sample_rate, target_rate)?;
        let num_quantizers = self.encoder_valid_num_quantizers.max(1);
        let mut codes = encoder.encode(&resampled, num_quantizers)?;
        if codes.is_empty() {
            return Ok(codes);
        }

        let target_frames = ceil_div(resampled.len(), self.encode_downsample_rate).max(1);
        for group in &mut codes {
            group.truncate(target_frames);
        }
        debug!(
            "Encoded reference audio to {} groups x {} frames ({} -> {} Hz, {} samples)",
            codes.len(),
            codes.first().map(|g| g.len()).unwrap_or(0),
            sample_rate,
            target_rate,
            resampled.len()
        );
        Ok(codes)
    }

    /// Decode codec tokens to audio waveform samples.
    pub fn decode(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codec_tokens.is_empty() || codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = codec_tokens[0].len();
        let first_codes = normalized_codec_indices(
            Some(codec_tokens[0].as_slice()),
            seq_len,
            self.rvq_first.codebook.codebook_size,
        );
        let first_tensor = Tensor::from_vec(first_codes, (1, seq_len), &self.device)?;

        // RVQ decode (first + residual codebooks) and project to hidden features.
        let first_proj = self.rvq_first.decode(&first_tensor)?;
        let rest_proj = self.rvq_rest.decode(codec_tokens, seq_len, &self.device)?;
        let quantized = first_proj.broadcast_add(&rest_proj)?;

        // Pre-conv + pre-transformer.
        let hidden = self.pre_conv.forward(&quantized)?;
        let mut hidden = hidden.transpose(1, 2)?;
        hidden = self.pre_transformer_input_proj.forward(&hidden)?;
        for layer in &self.pre_transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden = self.pre_transformer_norm.forward(&hidden)?;
        hidden = self.pre_transformer_output_proj.forward(&hidden)?;

        // Upsample stack + decoder blocks.
        let mut hidden = hidden.transpose(1, 2)?;
        for stage in &self.upsample_stages {
            hidden = stage.forward(&hidden)?;
        }
        hidden = self.decoder_init_conv.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.final_snake.forward(&hidden)?;
        let audio = self.final_conv.forward(&hidden)?;

        // Clamp to audio range and flatten to mono sample vector.
        let audio = audio.clamp(-1.0f32, 1.0f32)?;
        audio
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
            .map_err(Error::from)
    }
}

/// Repeat KV heads for GQA.
fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidInput(format!(
            "Invalid GQA head config: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )));
    }
    let repeats = num_heads / num_kv_heads;
    let x = x.transpose(1, 2)?;
    let out = candle_repeat_kv(x, repeats)?;
    out.transpose(1, 2).map_err(Error::from)
}

/// Build standard RoPE cos/sin cache.
fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }

    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for &inv in &inv_freq {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Causal mask for self-attention.
fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * total_len];
    for i in 0..seq_len {
        let limit = start_pos + i;
        for j in 0..total_len {
            if j > limit {
                data[i * total_len + j] = -1e4;
            }
        }
    }
    Tensor::from_vec(data, (1, seq_len, total_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn ceil_div(n: usize, d: usize) -> usize {
    if d == 0 {
        return 0;
    }
    n.div_ceil(d)
}

fn normalized_codec_index(token: u32, codebook_size: usize) -> i64 {
    if codebook_size == 0 {
        return 0;
    }
    (token as i64).rem_euclid(codebook_size as i64)
}

fn normalized_codec_indices(
    tokens: Option<&[u32]>,
    seq_len: usize,
    codebook_size: usize,
) -> Vec<i64> {
    let mut values = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let token = tokens.and_then(|tokens| tokens.get(t)).copied().unwrap_or(0);
        values.push(normalized_codec_index(token, codebook_size));
    }
    values
}

fn resample_audio_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = *audio.get(idx).unwrap_or(&0.0);
        let s1 = *audio.get(idx + 1).unwrap_or(&s0);
        out.push((s0 as f64 * (1.0 - frac) + s1 as f64 * frac) as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_codec_indices_wrap_and_pad_tokens() {
        assert_eq!(normalized_codec_index(0, 8), 0);
        assert_eq!(normalized_codec_index(9, 8), 1);
        assert_eq!(normalized_codec_index(17, 8), 1);

        assert_eq!(
            normalized_codec_indices(Some(&[0, 9]), 4, 8),
            vec![0, 1, 0, 0]
        );
        assert_eq!(normalized_codec_indices(None, 3, 8), vec![0, 0, 0]);
    }
}

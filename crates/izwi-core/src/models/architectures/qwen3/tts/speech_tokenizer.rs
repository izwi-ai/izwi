//! Speech Tokenizer Decoder for Qwen3-TTS.
//!
//! Converts multi-codebook RVQ codec tokens to waveform audio.
//! Implementation mirrors the official 12Hz decoder architecture:
//! RVQ projection -> pre-conv -> pre-transformer -> upsample stack -> decoder blocks -> waveform.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, LayerNormConfig,
    Linear, Module, RmsNorm, VarBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};
use crate::models::shared::attention::gqa::{
    banded_sdpa_bhsd, compact_gqa_sdpa_bhsd, AttentionBand, CompactGqaMask, BANDED_QUERY_BLOCK,
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

/// The acoustic codebooks a reference encode reads. It encodes with
/// `quantizers` quantizers, the semantic ones first, so the codebooks a
/// checkpoint ships past those are never read: Qwen3-TTS-12Hz has 32 and
/// encodes with 16.
pub(crate) fn encoder_acoustic_codebooks(quantizers: usize, cfg: &EncoderConfig) -> Result<usize> {
    if quantizers < cfg.num_semantic_quantizers || quantizers > cfg.num_quantizers {
        return Err(Error::ModelError(format!(
            "speech-tokenizer encoder_valid_num_quantizers {quantizers} is outside {}..={}",
            cfg.num_semantic_quantizers, cfg.num_quantizers
        )));
    }
    Ok(quantizers - cfg.num_semantic_quantizers)
}

/// The acoustic codebooks the reference encoder of the speech tokenizer in
/// `tokenizer_dir` loads, or `None` when it ships no encoder.
pub(crate) fn read_encoder_acoustic_codebooks(
    tokenizer_dir: &std::path::Path,
) -> Result<Option<usize>> {
    let speech_cfg: SpeechTokenizerConfig =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_dir.join("config.json"))?)?;
    let Some(encoder) = speech_cfg.encoder_config else {
        return Ok(None);
    };
    let encoder: EncoderConfig = serde_json::from_value(encoder)?;
    encoder_acoustic_codebooks(speech_cfg.encoder_valid_num_quantizers.max(1), &encoder).map(Some)
}

/// Whether the reference encoder reads speech-tokenizer tensor `name` (an
/// `encoder.*` name) when it loads `acoustic_codebooks` acoustic codebooks.
pub(crate) fn encoder_reads_tensor(name: &str, acoustic_codebooks: usize) -> bool {
    name.strip_prefix("encoder.quantizer.acoustic_residual_vector_quantizer.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|layer| layer.parse::<usize>().ok())
        .is_none_or(|layer| layer < acoustic_codebooks)
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
    /// Every pre-transformer layer is `sliding_attention` over this many keys
    /// (the reference config's `layer_types` and `sliding_window`).
    #[serde(default = "default_decoder_sliding_window")]
    pub sliding_window: usize,
}

fn default_decoder_sliding_window() -> usize {
    72
}

/// Codec frames per upsampler/vocoder window in [`SpeechTokenizerDecoder::decode`].
/// The vocoder's activations run at up to 24 kHz (about 8 MiB of live F32
/// tensors per frame at the last stage), so the window, not the utterance,
/// sets the decode's working set: 32 frames (2.56 s) plus left context.
const VOCODER_WINDOW_FRAMES: usize = 32;

/// Kernel of the ConvNeXt depthwise conv after each upsampling stage.
const CONVNEXT_KERNEL: usize = 7;
/// Kernel of the first and last vocoder convs (`decoder.0`, `decoder.6`).
const VOCODER_EDGE_KERNEL: usize = 7;
/// Kernel and dilations of the three residual units in each decoder block.
const RESIDUAL_KERNEL: usize = 7;
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];

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
        let mask = causal_mask(seq_len, total_len, 0, q.device(), q.dtype())?;
        let out = compact_gqa_sdpa_bhsd(
            &q,
            &k,
            &v,
            Some(CompactGqaMask::Additive(&mask)),
            1.0 / (self.head_dim as f64).sqrt(),
        )?;
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
                RESIDUAL_KERNEL,
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
            res1: ResidualUnit::load(out_channels, RESIDUAL_DILATIONS[0], vb.pp("block.2"))?,
            res2: ResidualUnit::load(out_channels, RESIDUAL_DILATIONS[1], vb.pp("block.3"))?,
            res3: ResidualUnit::load(out_channels, RESIDUAL_DILATIONS[2], vb.pp("block.4"))?,
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
        let dwconv = CausalConv1d::load(
            dim,
            dim,
            CONVNEXT_KERNEL,
            1,
            dim,
            true,
            vb.pp("dwconv.conv"),
        )?;
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
    sliding_window: usize,
}

impl Attention {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.sliding_window == 0 {
            return Err(Error::ModelError(
                "speech-tokenizer decoder sliding_window must be non-zero".into(),
            ));
        }
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
            sliding_window: cfg.sliding_window,
        })
    }

    fn apply_rope(&self, x: Tensor, start_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            start_pos,
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
        self.forward_at(x, 0)
    }

    /// Attention over `x` whose first row is frame `start_pos`.
    fn forward_at(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
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

        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Plain causal attention equals the sliding window while every query
        // sees at most `sliding_window` keys, so the fused path stays valid.
        if flash_attention_requested() && seq_len <= self.sliding_window {
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

        // Query blocks against only the keys their band reaches: no T x T
        // score or mask tensor.
        let out = banded_sdpa_bhsd(
            &q,
            &k,
            &v,
            AttentionBand::causal_window(self.sliding_window),
            1.0 / (self.head_dim as f64).sqrt(),
        )?;
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
        self.forward_at(x, 0)
    }

    /// The layer over `x` whose first row is frame `start_pos`.
    fn forward_at(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_at(&normed, start_pos)?;
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
    embeddings: Tensor,
    norms: Tensor,
    codebook_size: usize,
    codebook_dim: usize,
}

impl EncoderCodebook {
    fn load(vb: VarBuilder, codebook_size: usize, codebook_dim: usize) -> Result<Self> {
        let embedding_sum = vb.get((codebook_size, codebook_dim), "codebook.embed_sum")?;
        let cluster_usage = vb
            .get((codebook_size,), "codebook.cluster_usage")?
            .clamp(1e-7f64, f64::MAX)?;
        let embeddings = embedding_sum
            .broadcast_div(&cluster_usage.unsqueeze(1)?)?
            .to_dtype(DType::F32)?;
        Self::from_embeddings(embeddings, codebook_size, codebook_dim)
    }

    fn from_embeddings(
        embeddings: Tensor,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        if embeddings.dims2()? != (codebook_size, codebook_dim) {
            return Err(Error::ModelLoadError(format!(
                "Unexpected speech tokenizer codebook shape {:?}; expected ({codebook_size}, {codebook_dim})",
                embeddings.dims()
            )));
        }
        let embeddings = embeddings.to_dtype(DType::F32)?;
        let norms = embeddings.sqr()?.sum(1)?;
        Ok(Self {
            embeddings,
            norms,
            codebook_size,
            codebook_dim,
        })
    }

    fn quantize(&self, frames: &Tensor) -> Result<(Tensor, Tensor)> {
        let (frame_count, frame_dim) = frames.dims2()?;
        if frame_dim != self.codebook_dim {
            return Err(Error::InferenceError(format!(
                "Speech tokenizer RVQ frame dim {frame_dim} does not match codebook dim {}",
                self.codebook_dim
            )));
        }
        if frame_count == 0 {
            return Ok((
                Tensor::zeros((0,), DType::U32, frames.device())?,
                Tensor::zeros((0, self.codebook_dim), DType::F32, frames.device())?,
            ));
        }
        let frames = frames.to_dtype(DType::F32)?;
        let frame_norms = frames.sqr()?.sum_keepdim(1)?;
        let distances = frame_norms
            .broadcast_add(&self.norms.unsqueeze(0)?)?
            .broadcast_sub(&(frames.matmul(&self.embeddings.t()?)? * 2.0)?)?;
        let indices = distances.argmin(1)?;
        if indices.dims1()? != frame_count {
            return Err(Error::InferenceError(
                "Speech tokenizer RVQ argmin returned an unexpected shape".to_string(),
            ));
        }
        let quantized = self.embeddings.index_select(&indices, 0)?;
        Ok((indices, quantized))
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
            if layer.codebook_size == 0 || layer.codebook_dim != dim {
                return Err(Error::InferenceError(format!(
                    "Speech tokenizer RVQ codebook geometry ({}, {}) does not match residual dim {dim}",
                    layer.codebook_size, layer.codebook_dim
                )));
            }
            crate::models::shared::telemetry::record_layout_copy();
            let residual_bt = residual.i(0)?.transpose(0, 1)?.contiguous()?;
            let (indices, quantized) = layer.quantize(&residual_bt)?;
            crate::models::shared::telemetry::record_host_read(DType::U32, seq_len);
            let indices_host = indices.to_vec1::<u32>()?;
            let quantized = quantized.transpose(0, 1)?.unsqueeze(0)?;
            let quantized = if quantized.dtype() == residual.dtype() {
                quantized
            } else {
                crate::models::shared::telemetry::record_dtype_cast();
                quantized.to_dtype(residual.dtype())?
            };
            residual = residual.broadcast_sub(&quantized)?;
            all_indices.push(indices_host);
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
    /// Load the codebooks of the first `quantizers` quantizers (semantic
    /// first): the encode never reads the rest.
    fn load(cfg: &EncoderConfig, quantizers: usize, vb: VarBuilder) -> Result<Self> {
        let num_semantic = cfg.num_semantic_quantizers;
        let num_acoustic = encoder_acoustic_codebooks(quantizers, cfg)?;
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
            max_num_quantizers: quantizers,
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
    /// Load the encoder for reference encodes with `quantizers` quantizers.
    fn load(
        config: EncoderConfig,
        quantizers: usize,
        vb: VarBuilder,
        device: Device,
    ) -> Result<Self> {
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

        let quantizer =
            EncoderSplitResidualVectorQuantizer::load(&config, quantizers, vb.pp("quantizer"))?;
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
    /// Left context per vocoder window: the vocoder's causal receptive field.
    vocoder_context_frames: usize,
    /// Upsampler/vocoder windows run since load (diagnostics).
    vocoder_windows: AtomicU64,
    /// Largest window, context included, the vocoder has run on (diagnostics).
    vocoder_peak_window_frames: AtomicUsize,
}

/// Incremental decode state of [`SpeechTokenizerDecoder::decode_stream_step`]:
/// the frames consumed and the bounded causal contexts. Tensors are
/// immutable, so cloning the state is a cheap, exact checkpoint.
#[derive(Debug, Clone)]
pub struct SpeechTokenizerStream {
    frames: usize,
    pre_conv_tail: Option<Tensor>,
    layer_tails: Vec<Option<Tensor>>,
    latent_tail: Option<Tensor>,
}

impl SpeechTokenizerStream {
    /// Codec frames decoded so far.
    pub fn frames(&self) -> usize {
        self.frames
    }
}

/// Windowed codec-decode counters for diagnostics.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpeechTokenizerDecodeStats {
    pub window_frames: usize,
    pub context_frames: usize,
    pub windows: u64,
    pub peak_window_frames: usize,
}

impl SpeechTokenizerDecoder {
    /// Load decoder from speech_tokenizer directory.
    /// Load the decoder, and the reference encoder when `load_encoder`
    /// (voice cloning is its only user).
    pub fn load(
        model_dir: &std::path::Path,
        device: Device,
        dtype: DType,
        load_encoder: bool,
    ) -> Result<Self> {
        info!("Loading speech tokenizer decoder from {:?}", model_dir);

        let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let mut speech_cfg: SpeechTokenizerConfig = serde_json::from_str(&config_str)?;
        if !load_encoder {
            speech_cfg.encoder_config = None;
        }
        let weights_path = model_dir.join("model.safetensors");
        // Host-cast, exact-size uploads with periodic synchronization on
        // Metal (`staged`).
        let root_vb = unsafe {
            crate::models::shared::weights::staged::metal_staged_safetensors(
                &[weights_path],
                dtype,
                &device,
            )?
        };
        Self::from_config(speech_cfg, root_vb, device)
    }

    fn from_config(
        speech_cfg: SpeechTokenizerConfig,
        root_vb: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let encoder_valid_num_quantizers = speech_cfg.encoder_valid_num_quantizers;
        let input_sample_rate = speech_cfg.input_sample_rate;
        let encode_downsample_rate = speech_cfg.encode_downsample_rate;
        let decode_upsample_rate = speech_cfg.decode_upsample_rate;
        let decoder_config = speech_cfg.decoder_config;
        let encoder_config: Option<EncoderConfig> = speech_cfg
            .encoder_config
            .map(serde_json::from_value)
            .transpose()?;
        let vocoder_upsample = vocoder_upsample_factor(&decoder_config);
        if vocoder_upsample != decode_upsample_rate {
            return Err(Error::ModelError(format!(
                "speech-tokenizer decode_upsample_rate {decode_upsample_rate} does not match \
                 the decoder's upsampling product {vocoder_upsample}"
            )));
        }
        let vocoder_context_frames = vocoder_receptive_field_frames(&decoder_config);
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
            VOCODER_EDGE_KERNEL,
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
        let final_conv = CausalConv1d::load(
            in_channels,
            1,
            VOCODER_EDGE_KERNEL,
            1,
            1,
            true,
            vb.pp("decoder.6.conv"),
        )?;

        let encoder = if let Some(cfg) = encoder_config {
            info!("Loading speech tokenizer encoder...");
            Some(SpeechTokenizerEncoder::load(
                cfg,
                encoder_valid_num_quantizers.max(1),
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
            vocoder_context_frames,
            vocoder_windows: AtomicU64::new(0),
            vocoder_peak_window_frames: AtomicUsize::new(0),
        })
    }

    pub fn decode_stats(&self) -> SpeechTokenizerDecodeStats {
        SpeechTokenizerDecodeStats {
            window_frames: VOCODER_WINDOW_FRAMES,
            context_frames: self.vocoder_context_frames,
            windows: self.vocoder_windows.load(Ordering::Relaxed),
            peak_window_frames: self.vocoder_peak_window_frames.load(Ordering::Relaxed),
        }
    }

    /// Maximum codec frames produced from this exact input shape.
    pub fn reference_frame_upper_bound(
        &self,
        input_samples: usize,
        sample_rate: u32,
    ) -> Result<usize> {
        if input_samples == 0 {
            return Ok(0);
        }
        let resampled_samples =
            resampled_audio_len(input_samples, sample_rate, self.input_sample_rate as u32)?;
        Ok(ceil_div(resampled_samples, self.encode_downsample_rate).max(1))
    }

    pub(crate) fn decoded_sample_upper_bound(&self, codec_frames: usize) -> Result<usize> {
        if self.decode_upsample_rate == 0 {
            return Err(Error::ModelError(
                "speech-tokenizer decode upsample rate must be non-zero".into(),
            ));
        }
        codec_frames
            .checked_mul(self.decode_upsample_rate)
            .ok_or_else(|| Error::Overloaded("speech-tokenizer sample bound overflow".into()))
    }

    pub(crate) fn reference_encode_temporary_upper_bound_bytes(
        &self,
        max_seconds: usize,
        element_bytes: usize,
    ) -> Result<u64> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            Error::ModelError("speech-tokenizer reference encoder is unavailable".into())
        })?;
        reference_encode_workspace_upper_bound_bytes(
            &encoder.config,
            self.input_sample_rate,
            self.encode_downsample_rate,
            max_seconds,
            element_bytes,
        )
    }

    /// Conservative backend-temporary peak for a terminal codec/vocoder pass
    /// over `codec_frames` frames, following [`Self::decode`]: the frame-rate
    /// stages scale with the frame count (sliding-window attention in bounded
    /// query blocks), the upsampler and vocoder with one window.
    pub(crate) fn decode_temporary_upper_bound_bytes(
        &self,
        codec_frames: usize,
        element_bytes: usize,
    ) -> Result<u64> {
        decode_workspace_upper_bound_bytes(
            &self.config,
            self.vocoder_context_frames,
            codec_frames,
            element_bytes,
        )
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
    ///
    /// The frame-rate stages (RVQ, pre-conv, the sliding-window transformer)
    /// run once over every frame; their activations are a few KiB per frame.
    /// The upsampler and vocoder, whose activations run at up to 24 kHz and
    /// dominate memory, run in windows of [`VOCODER_WINDOW_FRAMES`] frames.
    /// Each window carries enough left context to cover their causal
    /// receptive field and drops it afterwards, so the output equals a
    /// whole-utterance pass, and the device is synchronized after each window
    /// so its pooled buffers are released before the next one.
    pub fn decode(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decode_windowed(codec_tokens, VOCODER_WINDOW_FRAMES)
    }

    fn decode_windowed(&self, codec_tokens: &[Vec<u32>], window_frames: usize) -> Result<Vec<f32>> {
        if codec_tokens.is_empty() || codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }
        let latents = self.decode_latents(codec_tokens)?;
        let frames = latents.dim(2)?;
        self.vocode_range(&latents, 0, frames, window_frames)
    }

    /// Samples for latent frames `from..to` of `latents`, vocoded in windows
    /// of `window_frames` with the vocoder's causal left context from the
    /// frames before each window (frames before `from` serve as context
    /// only), clamped to [-1, 1].
    fn vocode_range(
        &self,
        latents: &Tensor,
        from: usize,
        to: usize,
        window_frames: usize,
    ) -> Result<Vec<f32>> {
        let window_frames = window_frames.max(1);
        let upsample = self.decode_upsample_rate;
        let mut audio = Vec::with_capacity(to.saturating_sub(from).saturating_mul(upsample));
        let mut start = from;
        while start < to {
            let end = start.saturating_add(window_frames).min(to);
            let context = self.vocoder_context_frames.min(start);
            let span = end - start + context;
            let samples = {
                let window = latents.narrow(2, start - context, span)?;
                let wav = self.vocode(&window)?;
                let produced = wav.dim(2)?;
                if produced != span * upsample {
                    return Err(Error::InferenceError(format!(
                        "speech-tokenizer vocoder produced {produced} samples for {span} \
                         frames, expected {}",
                        span * upsample
                    )));
                }
                wav.narrow(2, context * upsample, (end - start) * upsample)?
                    .clamp(-1.0f32, 1.0f32)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?
            };
            audio.extend_from_slice(&samples);
            self.vocoder_windows.fetch_add(1, Ordering::Relaxed);
            self.vocoder_peak_window_frames
                .fetch_max(span, Ordering::Relaxed);
            // The window's tensors are dropped; synchronizing lets the
            // backend release their pooled buffers before the next window.
            self.device.synchronize()?;
            start = end;
        }
        Ok(audio)
    }

    /// A fresh incremental decode (QTTS-5).
    pub fn start_stream(&self) -> SpeechTokenizerStream {
        SpeechTokenizerStream {
            frames: 0,
            pre_conv_tail: None,
            layer_tails: vec![None; self.pre_transformer_layers.len()],
            latent_tail: None,
        }
    }

    /// Decode the next codec frames, `new_tokens[group][frame]` (the frames
    /// after those `stream` has consumed), and return exactly the samples a
    /// whole-utterance [`Self::decode`] produces for them (clamped,
    /// unnormalized). The
    /// decoder is causal, so earlier output never changes: the pre-conv keeps
    /// its kernel's tail, each pre-transformer layer its last
    /// `sliding_window - 1` input frames (and absolute RoPE positions), and
    /// the vocoder its receptive-field latents. Each step costs its own
    /// frames plus those bounded contexts, not the prefix.
    pub fn decode_stream_step(
        &self,
        stream: &mut SpeechTokenizerStream,
        new_tokens: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        let frames = new_tokens.first().map_or(0, Vec::len);
        if frames == 0 {
            return Ok(Vec::new());
        }
        let latents = self.stream_latents(stream, new_tokens)?;
        let (latents, context) = match &stream.latent_tail {
            Some(tail) => (Tensor::cat(&[tail, &latents], 2)?, tail.dim(2)?),
            None => (latents, 0),
        };
        let samples =
            self.vocode_range(&latents, context, context + frames, VOCODER_WINDOW_FRAMES)?;
        let latent_len = latents.dim(2)?;
        let kept = self.vocoder_context_frames.min(latent_len);
        stream.latent_tail = (kept > 0)
            .then(|| latents.narrow(2, latent_len - kept, kept))
            .transpose()?;
        Ok(samples)
    }

    /// Like [`Self::decode_stream_step`]: the next frames' latents
    /// (`[1, latent_dim, frames]`), exactly as [`Self::decode_latents`]
    /// computes them for the whole utterance.
    fn stream_latents(
        &self,
        stream: &mut SpeechTokenizerStream,
        new_tokens: &[Vec<u32>],
    ) -> Result<Tensor> {
        let frames = new_tokens.first().map_or(0, Vec::len);
        let start = stream.frames;
        let total = start + frames;
        let slice = new_tokens;
        if frames == 0 {
            return Err(Error::InvalidInput(
                "speech-tokenizer stream step has no frames".into(),
            ));
        }
        let first_codes = normalized_codec_indices(
            Some(slice[0].as_slice()),
            frames,
            self.rvq_first.codebook.codebook_size,
        );
        let first_tensor = Tensor::from_vec(first_codes, (1, frames), &self.device)?;
        let quantized = self
            .rvq_first
            .decode(&first_tensor)?
            .broadcast_add(&self.rvq_rest.decode(slice, frames, &self.device)?)?;

        // Pre-conv: its causal zero padding is the initial tail.
        let pad = self.pre_conv.causal_padding;
        let tail = match &stream.pre_conv_tail {
            Some(tail) => tail.clone(),
            None => Tensor::zeros(
                (quantized.dim(0)?, quantized.dim(1)?, pad),
                quantized.dtype(),
                &self.device,
            )?,
        };
        let joined = Tensor::cat(&[&tail, &quantized], 2)?;
        let hidden = self.pre_conv.conv.forward(&joined)?;
        let joined_len = joined.dim(2)?;
        stream.pre_conv_tail = Some(joined.narrow(2, joined_len - pad, pad)?);

        let mut hidden = self
            .pre_transformer_input_proj
            .forward(&hidden.transpose(1, 2)?)?;
        let keep = self.config.sliding_window.saturating_sub(1);
        for (layer, tail) in self
            .pre_transformer_layers
            .iter()
            .zip(stream.layer_tails.iter_mut())
        {
            let (full, cached) = match tail.as_ref() {
                Some(tail) => (Tensor::cat(&[tail, &hidden], 1)?, tail.dim(1)?),
                None => (hidden.clone(), 0),
            };
            let full_len = full.dim(1)?;
            let kept = keep.min(full_len);
            *tail = (kept > 0)
                .then(|| full.narrow(1, full_len - kept, kept))
                .transpose()?;
            hidden = layer
                .forward_at(&full, start - cached)?
                .narrow(1, cached, frames)?;
        }
        let hidden = self.pre_transformer_norm.forward(&hidden)?;
        let latents = self
            .pre_transformer_output_proj
            .forward(&hidden)?
            .transpose(1, 2)?;
        stream.frames = total;
        Ok(latents)
    }

    /// RVQ, pre-conv and the pre-transformer over every frame:
    /// `[1, latent_dim, frames]`.
    fn decode_latents(&self, codec_tokens: &[Vec<u32>]) -> Result<Tensor> {
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
        hidden.transpose(1, 2).map_err(Error::from)
    }

    /// Upsample stack + vocoder on `[1, latent_dim, frames]`: unclamped
    /// `[1, 1, frames * decode_upsample_rate]` audio.
    fn vocode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut hidden = latents.clone();
        for stage in &self.upsample_stages {
            hidden = stage.forward(&hidden)?;
        }
        hidden = self.decoder_init_conv.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.final_snake.forward(&hidden)?;
        self.final_conv.forward(&hidden)
    }
}

/// Samples per codec frame produced by the upsampler and vocoder.
fn vocoder_upsample_factor(cfg: &DecoderConfig) -> usize {
    cfg.upsampling_ratios
        .iter()
        .chain(&cfg.upsample_rates)
        .product::<usize>()
        .max(1)
}

/// Left context, in codec frames, covering the causal receptive field of
/// everything after the pre-transformer (upsampler, ConvNeXt blocks, vocoder),
/// so a windowed pass reproduces a whole-utterance pass. Every stage is
/// causal; each contributes the samples its kernel reaches back at its own
/// rate, plus one sample for mapping a final-rate sample onto its grid.
fn vocoder_receptive_field_frames(cfg: &DecoderConfig) -> usize {
    let total = vocoder_upsample_factor(cfg);
    let mut reach = 0usize;
    let mut add = |taps: usize, rate: usize| reach += (taps + 1) * (total / rate.max(1));
    let mut rate = 1usize;
    for &ratio in &cfg.upsampling_ratios {
        // Transposed conv with kernel == stride: no overlap between frames.
        rate *= ratio;
        add(CONVNEXT_KERNEL - 1, rate);
    }
    add(VOCODER_EDGE_KERNEL - 1, rate);
    let residual_taps = (RESIDUAL_KERNEL - 1) * RESIDUAL_DILATIONS.iter().sum::<usize>();
    for &upsample in &cfg.upsample_rates {
        // Transposed conv with kernel 2r and stride r: one input sample back.
        add(1, rate);
        rate *= upsample;
        add(residual_taps, rate);
    }
    add(VOCODER_EDGE_KERNEL - 1, rate);
    reach.div_ceil(total)
}

/// Backend-temporary upper bound for [`SpeechTokenizerDecoder::decode`] over
/// `codec_frames` frames with `context_frames` of vocoder left context.
fn decode_workspace_upper_bound_bytes(
    cfg: &DecoderConfig,
    context_frames: usize,
    codec_frames: usize,
    element_bytes: usize,
) -> Result<u64> {
    let overflow = || Error::Overloaded("speech-tokenizer workspace overflow".into());
    let to_u64 = |value: usize| u64::try_from(value).map_err(|_| overflow());
    let mul = |left: u64, right: u64| left.checked_mul(right).ok_or_else(overflow);
    let add = |left: u64, right: u64| left.checked_add(right).ok_or_else(overflow);

    // Frame-rate stages over every frame: one query block of sliding-window
    // scores, the transformer's linear activations, and the pre-conv/latent
    // tensors that stay live while the vocoder windows run.
    let frames = to_u64(codec_frames)?;
    let block = codec_frames.min(BANDED_QUERY_BLOCK);
    let keys = codec_frames.min(block + cfg.sliding_window.max(1) - 1);
    let attention = mul(
        mul(to_u64(cfg.num_attention_heads)?, to_u64(block)?)?,
        to_u64(keys)?,
    )?;
    let transformer_linear = mul(mul(to_u64(cfg.hidden_size)?, frames)?, 12)?;
    let frame_rate_width = cfg.latent_dim.max(cfg.hidden_size.saturating_mul(3));
    let frame_rate = mul(mul(to_u64(frame_rate_width)?, frames)?, 2)?;

    // Upsampler and vocoder over one window, context included.
    let window = codec_frames.min(VOCODER_WINDOW_FRAMES.saturating_add(context_frames));
    let mut time = to_u64(window)?;
    let latent = to_u64(cfg.latent_dim)?;
    let mut peak_activation = mul(latent, time)?;
    for ratio in &cfg.upsampling_ratios {
        time = mul(time, to_u64(*ratio)?)?;
        peak_activation = peak_activation.max(mul(latent, time)?);
    }
    let mut channels = to_u64(cfg.decoder_dim)?;
    peak_activation = peak_activation.max(mul(channels, time)?);
    for rate in &cfg.upsample_rates {
        time = mul(time, to_u64(*rate)?)?;
        channels = (channels / 2).max(1);
        peak_activation = peak_activation.max(mul(channels, time)?);
    }
    // The widest stage's 7-tap convs hold an im2col copy (7 widths) next to
    // the input, its padded copy, the output, the residual and two SnakeBeta
    // temporaries: 13 activation widths.
    let vocoder = mul(peak_activation, 13)?;
    let elements = add(
        add(attention, transformer_linear)?,
        add(frame_rate, vocoder)?,
    )?;
    mul(elements, to_u64(element_bytes)?)
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

/// Backend temporaries of one reference encode of `max_seconds` of audio,
/// following [`SpeechTokenizerEncoderNet`] and the encoder transformer: the
/// conv stack's largest stage (its input, ELU, padded copy, the conv's
/// im2col of `kernel x channels x length` and its output live together),
/// the transformer at the conv stack's output rate, and the quantizer at
/// the codec frame rate. Each downsampling stage doubles the channels and
/// divides the length by its ratio (at least 2), so the full-rate stage
/// with `num_filters` channels dominates. The previous bound charged every
/// sample at the widest transformer width (`intermediate_size`) eight
/// times: 47 GB for 30 s on Qwen3-TTS-0.6B-Base, which refused the load.
pub(crate) fn reference_encode_workspace_upper_bound_bytes(
    cfg: &EncoderConfig,
    input_sample_rate: usize,
    encode_downsample_rate: usize,
    max_seconds: usize,
    element_bytes: usize,
) -> Result<u64> {
    let overflow = || Error::Overloaded("speech-tokenizer reference workspace overflow".into());
    let mul = |left: usize, right: usize| left.checked_mul(right).ok_or_else(overflow);
    let samples = mul(input_sample_rate, max_seconds)?;
    // First conv: input, padded input, im2col and output.
    let mut peak = mul(
        mul(cfg.audio_channels.max(1), samples)?,
        cfg.kernel_size.saturating_add(2),
    )?
    .checked_add(mul(cfg.num_filters, samples)?)
    .ok_or_else(overflow)?;
    let mut channels = cfg.num_filters.max(1);
    let mut length = samples;
    for &ratio in cfg.upsampling_ratios.iter().rev() {
        let activation = mul(channels, length)?;
        // Residual units: input, ELU, padded copy, conv output and sum
        // (five activations) beside the widest unit's im2col.
        let residual = mul(activation, cfg.residual_kernel_size.saturating_add(5))?;
        // Downsampling conv (kernel 2 x ratio, stride ratio): ELU, padded
        // copy and input, an im2col of 2 activations, output of 2/ratio.
        let downsample = mul(activation, 6)?;
        peak = peak.max(residual).max(downsample);
        channels = mul(channels, 2)?;
        length = ceil_div(length, ratio.max(1));
    }
    let activation = mul(channels, length)?;
    peak = peak.max(
        mul(activation, cfg.last_kernel_size.saturating_add(3))?
            .checked_add(mul(cfg.hidden_size, length)?)
            .ok_or_else(overflow)?,
    );
    // Encoder transformer at the conv output rate: attention scores and
    // their softmax, plus the layer activations.
    let attention = mul(mul(mul(cfg.num_attention_heads, length)?, length)?, 2)?;
    let transformer_width = mul(cfg.hidden_size, 8)?
        .checked_add(mul(cfg.intermediate_size, 3)?)
        .ok_or_else(overflow)?;
    let transformer = mul(length, transformer_width)?;
    let frames = ceil_div(samples, encode_downsample_rate.max(1));
    let quantizer_width = cfg
        .codebook_size
        .checked_add(mul(cfg.vector_quantization_hidden_dimension, 4)?)
        .ok_or_else(overflow)?;
    let quantizer = mul(frames, quantizer_width)?;
    let elements = [peak, attention, transformer, quantizer]
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
        .ok_or_else(overflow)?;
    u64::try_from(mul(elements, element_bytes)?).map_err(|_| overflow())
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
        let token = tokens
            .and_then(|tokens| tokens.get(t))
            .copied()
            .unwrap_or(0);
        values.push(normalized_codec_index(token, codebook_size));
    }
    values
}

fn resampled_audio_len(input_samples: usize, src_rate: u32, dst_rate: u32) -> Result<usize> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "Audio sample rates must be greater than zero".to_string(),
        ));
    }
    if input_samples == 0 || src_rate == dst_rate {
        return Ok(input_samples);
    }

    let numerator = (input_samples as u128)
        .checked_mul(dst_rate as u128)
        .ok_or_else(|| Error::Overloaded("Resampled audio length overflow".to_string()))?;
    let rounded = numerator
        .checked_add(src_rate as u128 - 1)
        .ok_or_else(|| Error::Overloaded("Resampled audio length overflow".to_string()))?
        / src_rate as u128;
    usize::try_from(rounded)
        .map_err(|_| Error::Overloaded("Resampled audio length overflow".to_string()))
}

fn resample_audio_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = resampled_audio_len(audio.len(), src_rate, dst_rate)?;
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

/// The encoder of the Qwen3-TTS-12Hz speech tokenizer as shipped (0.6B and
/// 1.7B Base share it): `speech_tokenizer/config.json` `encoder_config`.
#[cfg(test)]
pub(crate) const QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON: &str = r#"{"_frame_rate": 12.5,
    "attention_bias": false, "audio_channels": 1, "codebook_dim": 256,
    "codebook_size": 2048, "compress": 2, "dilation_growth_rate": 2, "head_dim": 64,
    "hidden_size": 512, "intermediate_size": 2048, "kernel_size": 7, "last_kernel_size": 3,
    "layer_scale_initial_scale": 0.01, "max_position_embeddings": 8000, "norm_eps": 1e-05,
    "num_attention_heads": 8, "num_filters": 64, "num_hidden_layers": 8,
    "num_key_value_heads": 8, "num_quantizers": 32, "num_residual_layers": 1,
    "num_semantic_quantizers": 1, "pad_mode": "constant", "residual_kernel_size": 3,
    "rope_theta": 10000.0, "sampling_rate": 24000, "sliding_window": 250,
    "upsampling_ratios": [8, 6, 5, 4], "use_causal_conv": true, "use_conv_shortcut": false,
    "vector_quantization_hidden_dimension": 256}"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A 30 s reference on the shipped encoder: the full-rate residual
    /// unit (64 channels x 720,000 samples, five activations beside a
    /// kernel-3 im2col) dominates, about 1.5 GB of F32 temporaries. The old
    /// bound charged every sample at the 2,048-wide transformer width eight
    /// times, 47 GB.
    #[test]
    fn reference_encode_bound_follows_the_shipped_encoder() {
        let cfg: EncoderConfig = serde_json::from_str(QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON).unwrap();
        let bytes =
            reference_encode_workspace_upper_bound_bytes(&cfg, 24_000, 1_920, 30, 4).unwrap();
        let full_rate_activation = 64u64 * 720_000 * 4;
        assert!(bytes >= full_rate_activation * 8, "{bytes}");
        assert!(bytes < 2_000_000_000, "{bytes}");
        // Shorter references cost proportionally less.
        let ten = reference_encode_workspace_upper_bound_bytes(&cfg, 24_000, 1_920, 10, 4).unwrap();
        assert!(ten * 2 < bytes);
    }

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

    #[test]
    fn resampled_audio_len_is_exact_and_fails_closed() {
        assert_eq!(resampled_audio_len(16_000, 16_000, 24_000).unwrap(), 24_000);
        assert_eq!(resampled_audio_len(1, 44_100, 24_000).unwrap(), 1);
        assert!(matches!(
            resampled_audio_len(1, 0, 24_000),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            resampled_audio_len(usize::MAX, 1, u32::MAX),
            Err(Error::Overloaded(_))
        ));
    }

    #[test]
    fn encoder_codebook_quantizes_frames_with_candle_ops() {
        let device = &Device::Cpu;
        let codebook = EncoderCodebook::from_embeddings(
            Tensor::from_vec(
                vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0],
                (4, 2),
                device,
            )
            .expect("embeddings"),
            4,
            2,
        )
        .expect("codebook");
        let frames = Tensor::from_vec(vec![0.9f32, 0.1, 0.1, 0.8], (2, 2), device).expect("frames");

        let (indices, quantized) = codebook.quantize(&frames).expect("quantize");
        assert_eq!(indices.to_vec1::<u32>().expect("indices"), vec![1, 2]);
        assert_eq!(
            quantized.to_vec2::<f32>().expect("quantized"),
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
    }

    #[test]
    fn encoder_loads_only_the_codebooks_its_encode_reads() {
        let cfg: EncoderConfig = serde_json::from_str(QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON).unwrap();
        // 16 of 32 quantizers, one semantic: 15 acoustic codebooks.
        assert_eq!(encoder_acoustic_codebooks(16, &cfg).unwrap(), 15);
        assert!(encoder_acoustic_codebooks(0, &cfg).is_err());
        assert!(encoder_acoustic_codebooks(33, &cfg).is_err());
        let acoustic = "encoder.quantizer.acoustic_residual_vector_quantizer.layers";
        assert!(encoder_reads_tensor(&format!("{acoustic}.14.codebook.embed_sum"), 15));
        assert!(!encoder_reads_tensor(&format!("{acoustic}.15.codebook.embed_sum"), 15));
        assert!(!encoder_reads_tensor(&format!("{acoustic}.31.codebook.initialized"), 15));
        assert!(encoder_reads_tensor(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            15
        ));
        assert!(encoder_reads_tensor("encoder.encoder.layers.0.conv.weight", 15));
    }

    #[test]
    fn encoder_codebook_argmin_preserves_first_index_on_ties() {
        let device = &Device::Cpu;
        let codebook = EncoderCodebook::from_embeddings(
            Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], (2, 2), device).expect("embeddings"),
            2,
            2,
        )
        .expect("codebook");
        let frames = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), device).expect("frames");

        let (indices, _) = codebook.quantize(&frames).expect("quantize");
        assert_eq!(indices.to_vec1::<u32>().expect("indices"), vec![0]);
    }

    /// Deterministic xorshift weights keyed by tensor name: norm gains near
    /// one, matrices and kernels scaled by 1/sqrt(fan-in), so stacked blocks
    /// stay O(1).
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
            let seed = name.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
            });
            let gain = name.ends_with("norm.weight");
            let fan_in = shape.dims().iter().skip(1).product::<usize>().max(1);
            let scale = if gain {
                0.1
            } else {
                (1.0 / (fan_in as f32).sqrt()).min(0.3)
            };
            let values = crate::kernels::numeric_check::xorshift_tensor(
                shape.dims(),
                seed,
                scale,
                &Device::Cpu,
            )?;
            let values = if gain { (values + 1.0)? } else { values };
            values.to_dtype(dtype)?.to_device(device)
        }

        fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
            candle_core::bail!("unexpected unchecked tensor request {name}")
        }

        fn contains_tensor(&self, _: &str) -> bool {
            false
        }
    }

    fn xorshift_vb(device: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(Box::new(XorshiftWeights), DType::F32, device.clone())
    }

    /// The published 12 Hz decoder geometry shrunk to a few channels: two
    /// pre-transformer layers with a 6-frame sliding window, the same
    /// upsampling topology at x2 x2 then x3 x2 (24 samples per frame).
    fn tiny_speech_tokenizer_config() -> SpeechTokenizerConfig {
        serde_json::from_str(
            r#"{"architectures": ["Qwen3TTSTokenizerV2Model"],
            "model_type": "qwen3_tts_tokenizer_12hz", "encoder_valid_num_quantizers": 4,
            "input_sample_rate": 24000, "output_sample_rate": 24000,
            "decode_upsample_rate": 24, "encode_downsample_rate": 24,
            "decoder_config": {"latent_dim": 16, "codebook_dim": 16, "codebook_size": 32,
                "decoder_dim": 16, "hidden_size": 16, "intermediate_size": 32,
                "hidden_act": "silu", "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5,
                "rope_theta": 10000, "num_quantizers": 4, "num_semantic_quantizers": 1,
                "semantic_codebook_size": 32, "sliding_window": 6,
                "upsample_rates": [3, 2], "upsampling_ratios": [2, 2],
                "vector_quantization_hidden_dimension": 16}}"#,
        )
        .unwrap()
    }

    /// QTTS-9: every pre-transformer layer attends to at most
    /// `sliding_window` keys, so a change to frame 0 cannot reach frames at
    /// or past the window through one layer (masked weights are exactly zero).
    #[test]
    fn decoder_attention_ignores_keys_outside_its_sliding_window() {
        use crate::kernels::numeric_check::{host_f32, xorshift_tensor};

        let config = tiny_speech_tokenizer_config().decoder_config;
        let window = config.sliding_window;
        let attention = Attention::load(&config, xorshift_vb(&Device::Cpu)).unwrap();
        let seq = 3 * window;
        let x = xorshift_tensor(&[1, seq, config.hidden_size], 5, 1.0, &Device::Cpu).unwrap();
        let bump = Tensor::ones((1, 1, config.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let perturbed = Tensor::cat(
            &[
                (x.narrow(1, 0, 1).unwrap() + bump).unwrap(),
                x.narrow(1, 1, seq - 1).unwrap(),
            ],
            1,
        )
        .unwrap();
        let base = host_f32(&attention.forward(&x).unwrap()).unwrap();
        let moved = host_f32(&attention.forward(&perturbed).unwrap()).unwrap();
        let width = config.hidden_size;
        let frame_changed =
            |t: usize| base[t * width..(t + 1) * width] != moved[t * width..(t + 1) * width];
        assert!(
            (0..window).all(frame_changed),
            "frames inside the window must see frame 0"
        );
        assert!(
            (window..seq).all(|t| !frame_changed(t)),
            "frames past the window attended to frame 0"
        );
    }

    fn tiny_decoder() -> SpeechTokenizerDecoder {
        SpeechTokenizerDecoder::from_config(
            tiny_speech_tokenizer_config(),
            xorshift_vb(&Device::Cpu),
            Device::Cpu,
        )
        .unwrap()
    }

    /// Unstructured codes for every quantizer group.
    fn xorshift_codes(groups: usize, frames: usize, codebook_size: usize) -> Vec<Vec<u32>> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..groups)
            .map(|_| {
                (0..frames)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state % codebook_size as u64) as u32
                    })
                    .collect()
            })
            .collect()
    }

    /// QTTS-3: the windowed upsampler/vocoder reproduces a whole-utterance
    /// pass across window boundaries, never runs on more than one window
    /// plus its context, and is bit-identical when one window covers it all.
    #[test]
    fn windowed_codec_decode_matches_whole_utterance_decode() {
        use crate::kernels::numeric_check::max_abs_diff;

        let decoder = tiny_decoder();
        let context = decoder.vocoder_context_frames;
        assert!(context > 0);
        let codes = xorshift_codes(4, 37, 32);
        let window = 5;
        let windowed = decoder.decode_windowed(&codes, window).unwrap();
        let stats = decoder.decode_stats();
        assert_eq!(stats.windows, 37usize.div_ceil(window) as u64);
        assert_eq!(stats.peak_window_frames, window + context);

        let whole = decoder.decode_windowed(&codes, usize::MAX).unwrap();
        assert_eq!(whole.len(), 37 * 24);
        assert!(
            whole.iter().filter(|v| v.abs() < 0.99).count() > whole.len() / 2,
            "decoder output is saturated; the comparison would be vacuous"
        );
        let error = max_abs_diff(&windowed, &whole);
        assert!(error < 1e-5, "windowed decode diverges by {error}");

        // The default window covers 20 frames in one pass: same op shapes.
        let short = xorshift_codes(4, 20, 32);
        assert_eq!(
            decoder.decode(&short).unwrap(),
            decoder.decode_windowed(&short, usize::MAX).unwrap()
        );
    }

    /// QTTS-5: decoding a stream in uneven steps (single frames, steps
    /// shorter and longer than the sliding window and the vocoder window)
    /// reproduces the whole-utterance decode sample for sample, so a
    /// stream never re-decodes its prefix and its chunks join seamlessly.
    #[test]
    fn stream_steps_reproduce_the_whole_decode() {
        use crate::kernels::numeric_check::max_abs_diff;

        let decoder = tiny_decoder();
        let codes = xorshift_codes(4, 61, 32);
        let whole = decoder.decode_windowed(&codes, usize::MAX).unwrap();
        let mut stream = decoder.start_stream();
        let mut streamed = Vec::new();
        let mut frames = 0;
        for step in [1usize, 5, 3, 9, 1, 19, 2, 21] {
            let end = (frames + step).min(61);
            let next = codes
                .iter()
                .map(|g| g[frames..end].to_vec())
                .collect::<Vec<_>>();
            let chunk = decoder.decode_stream_step(&mut stream, &next).unwrap();
            assert_eq!(chunk.len(), (end - frames) * 24);
            streamed.extend(chunk);
            frames = end;
        }
        assert_eq!(stream.frames(), 61);
        let none = vec![Vec::new(); 4];
        assert!(decoder
            .decode_stream_step(&mut stream, &none)
            .unwrap()
            .is_empty());
        assert_eq!(streamed.len(), whole.len());
        let error = max_abs_diff(&streamed, &whole);
        assert!(error < 1e-5, "streamed decode diverges by {error}");
    }

    /// QTTS-5: the streamed pre-transformer latents equal the whole
    /// utterance's where the attention context matters: each layer keeps
    /// `sliding_window - 1` input frames (one frame less changes the latents
    /// of every step's first frames about 20x past the tolerance).
    #[test]
    fn stream_latents_keep_each_layers_sliding_window_context() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff};

        // Sharp attention (large q/k projections, unit layer scales), so a
        // frame's latent depends visibly on its window's earlier frames.
        struct SharpAttentionWeights;
        impl candle_nn::var_builder::SimpleBackend for SharpAttentionWeights {
            fn get(
                &self,
                shape: candle_core::Shape,
                name: &str,
                init: candle_nn::Init,
                dtype: DType,
                device: &Device,
            ) -> candle_core::Result<Tensor> {
                let base = XorshiftWeights.get(shape, name, init, dtype, device)?;
                if name.contains("layer_scale") {
                    base.ones_like()
                } else if name.contains("q_proj") || name.contains("k_proj") {
                    base * 8.0
                } else if name.contains("v_proj")
                    || name.contains("o_proj")
                    || name.contains("output_proj")
                {
                    base * 4.0
                } else {
                    Ok(base)
                }
            }

            fn get_unchecked(
                &self,
                name: &str,
                _: DType,
                _: &Device,
            ) -> candle_core::Result<Tensor> {
                candle_core::bail!("unexpected unchecked tensor request {name}")
            }

            fn contains_tensor(&self, _: &str) -> bool {
                false
            }
        }
        let decoder = SpeechTokenizerDecoder::from_config(
            tiny_speech_tokenizer_config(),
            VarBuilder::from_backend(Box::new(SharpAttentionWeights), DType::F32, Device::Cpu),
            Device::Cpu,
        )
        .unwrap();
        let codes = xorshift_codes(4, 40, 32);
        let whole = host_f32(&decoder.decode_latents(&codes).unwrap()).unwrap();
        let mut stream = decoder.start_stream();
        let mut parts = Vec::new();
        let mut frames = 0;
        for step in [3usize, 1, 7, 2, 11, 16] {
            let end = frames + step;
            let next = codes
                .iter()
                .map(|g| g[frames..end].to_vec())
                .collect::<Vec<_>>();
            parts.push(decoder.stream_latents(&mut stream, &next).unwrap());
            frames = end;
        }
        let window = decoder.config.sliding_window;
        assert!(stream
            .layer_tails
            .iter()
            .all(|tail| tail.as_ref().unwrap().dim(1).unwrap() == window - 1));
        let streamed = host_f32(&Tensor::cat(&parts, 2).unwrap()).unwrap();
        let error = max_abs_diff(&streamed, &whole);
        // Float reassociation gives about 1e-6; one frame of missing
        // context gives about 2e-5.
        assert!(error < 5e-6, "streamed latents diverge by {error}");
    }

    /// The computed vocoder context really covers its receptive field: a
    /// change to latent frame 0 must not reach any sample at or past
    /// `context` frames.
    #[test]
    fn vocoder_context_covers_the_measured_receptive_field() {
        use crate::kernels::numeric_check::{host_f32, xorshift_tensor};

        let decoder = tiny_decoder();
        let config = &decoder.config;
        let context = decoder.vocoder_context_frames;
        let frames = context + 8;
        let latents =
            xorshift_tensor(&[1, config.latent_dim, frames], 9, 1.0, &Device::Cpu).unwrap();
        let bump = Tensor::ones((1, config.latent_dim, 1), DType::F32, &Device::Cpu).unwrap();
        let perturbed = Tensor::cat(
            &[
                (latents.narrow(2, 0, 1).unwrap() + bump).unwrap(),
                latents.narrow(2, 1, frames - 1).unwrap(),
            ],
            2,
        )
        .unwrap();
        let base = host_f32(&decoder.vocode(&latents).unwrap()).unwrap();
        let moved = host_f32(&decoder.vocode(&perturbed).unwrap()).unwrap();
        let last_changed = base
            .iter()
            .zip(&moved)
            .rposition(|(a, b)| a != b)
            .expect("frame 0 must reach its own samples");
        assert!(
            last_changed < context * decoder.decode_upsample_rate,
            "frame 0 reached sample {last_changed}, past {context} frames of context"
        );

        // Published geometry: 11 frames (0.88 s) of left context.
        let published: DecoderConfig = serde_json::from_str(PUBLISHED_DECODER_CONFIG).unwrap();
        assert_eq!(vocoder_receptive_field_frames(&published), 11);
    }

    /// `decoder_config` of Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer
    /// (identical across the published 12 Hz checkpoints).
    const PUBLISHED_DECODER_CONFIG: &str = r#"{"attention_bias": false,
        "attention_dropout": 0.0, "latent_dim": 1024, "codebook_dim": 512,
        "codebook_size": 2048, "decoder_dim": 1536, "hidden_act": "silu",
        "hidden_size": 512, "intermediate_size": 1024, "layer_scale_initial_scale": 0.01,
        "max_position_embeddings": 8000, "head_dim": 64, "num_attention_heads": 16,
        "num_hidden_layers": 8, "num_key_value_heads": 16, "num_quantizers": 16,
        "num_semantic_quantizers": 1, "rms_norm_eps": 1e-05, "rope_theta": 10000,
        "semantic_codebook_size": 4096, "sliding_window": 72, "upsample_rates": [8, 5, 4, 3],
        "upsampling_ratios": [2, 2], "vector_quantization_hidden_dimension": 512}"#;

    /// The decode price follows the windowed decode: the vocoder term is one
    /// window whatever the utterance length, so 120 s of audio prices close
    /// to 4 s instead of growing ~30x.
    #[test]
    fn codec_decode_price_is_one_vocoder_window() {
        let published: DecoderConfig = serde_json::from_str(PUBLISHED_DECODER_CONFIG).unwrap();
        let context = vocoder_receptive_field_frames(&published);
        let price = |frames| decode_workspace_upper_bound_bytes(&published, context, frames, 4);
        let window = price(VOCODER_WINDOW_FRAMES + context).unwrap();
        let long = price(1_500).unwrap();
        assert!(price(1).unwrap() < window);
        assert!(long >= window);
        // Only the frame-rate stages grow past one window: under 64 MiB more
        // at 1,500 frames (120 s).
        assert!(
            long - window < 64 << 20,
            "frame-rate growth {}",
            long - window
        );
        // One F32 window of the published vocoder: 43 frames at 24 kHz with
        // 96 channels and 13 activation widths, about 0.4 GiB.
        assert!(
            window > 300 << 20 && window < 512 << 20,
            "window price {window}"
        );
    }

    /// Real-weight check of the windowed decode against one whole-utterance
    /// pass (CPU, F32, about 1.5 GB). Set IZWI_TEST_QWEN_TTS_SPEECH_TOKENIZER
    /// to a `speech_tokenizer` directory, or keep the default izwi model path.
    #[test]
    #[ignore = "needs a local Qwen3-TTS speech_tokenizer checkpoint"]
    fn published_windowed_decode_matches_whole_decode() {
        use crate::kernels::numeric_check::{max_abs_diff, relative_l2};

        let dir = {
            let _env = crate::env_test_lock().lock().unwrap();
            std::env::var_os("IZWI_TEST_QWEN_TTS_SPEECH_TOKENIZER")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(
                            "Library/Application Support/izwi/models/\
                             Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer",
                        )
                    })
                })
                .unwrap()
        };
        let decoder = SpeechTokenizerDecoder::load(&dir, Device::Cpu, DType::F32, true).unwrap();
        let frames = 72;
        let codes = xorshift_codes(16, frames, 2048);
        let started = std::time::Instant::now();
        let windowed = decoder.decode(&codes).unwrap();
        let windowed_ms = started.elapsed().as_millis();
        let started = std::time::Instant::now();
        let whole = decoder.decode_windowed(&codes, usize::MAX).unwrap();
        let whole_ms = started.elapsed().as_millis();
        let stats = decoder.decode_stats();
        eprintln!(
            "windowed {windowed_ms} ms ({} windows, peak {} frames), whole {whole_ms} ms",
            stats.windows, stats.peak_window_frames
        );
        assert_eq!(windowed.len(), frames * 1920);
        let error = max_abs_diff(&windowed, &whole);
        let rel = relative_l2(&windowed, &whole);
        eprintln!("max abs diff {error:e}, relative L2 {rel:e}");
        assert!(error < 1e-4 && rel < 1e-5, "max abs {error}, rel {rel}");
    }

    /// QTTS-10/11 parity gate for an F16 speech-tokenizer decoder on Metal:
    /// the same codes (72 frames, xorshift, all 16 codebooks) decoded in F16
    /// and in F32 on Metal. Opt-in (ignored, real weights, about 1.5 GB);
    /// the decoder stays F32 by default unless this gate passes.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "needs a local Qwen3-TTS speech_tokenizer checkpoint and a Metal device"]
    fn metal_f16_decoder_parity_gate() {
        use crate::kernels::numeric_check::{max_abs_diff, relative_l2};

        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_f16_decoder_parity_gate")
        else {
            return;
        };
        let dir = {
            let _env = crate::env_test_lock().lock().unwrap();
            std::env::var_os("IZWI_TEST_QWEN_TTS_SPEECH_TOKENIZER")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        std::path::PathBuf::from(home).join(
                            "Library/Application Support/izwi/models/\
                             Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer",
                        )
                    })
                })
                .unwrap()
        };
        let codes = xorshift_codes(16, 72, 2048);
        let decode = |dtype| {
            SpeechTokenizerDecoder::load(&dir, metal.clone(), dtype, false)
                .unwrap()
                .decode(&codes)
                .unwrap()
        };
        let full = decode(DType::F32);
        let half = decode(DType::F16);
        let rel = relative_l2(&half, &full);
        let error = max_abs_diff(&half, &full);
        let snr_db = -20.0 * rel.log10();
        eprintln!("F16 vs F32 decoder: relative L2 {rel:e}, SNR {snr_db:.1} dB, max abs {error:e}");
        assert!(half.iter().all(|v| v.is_finite()));
        assert!(
            snr_db >= 30.0,
            "F16 decoder SNR {snr_db:.1} dB is below the 30 dB gate"
        );
    }
}

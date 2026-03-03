//! Native Qwen3.5 text-chat model loader and generation.
//!
//! Qwen3.5 chat checkpoints use a hybrid architecture (full-attention + linear-attention)
//! with `model_type = qwen3_5` and text weights under `model.language_model.*`.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{ops, Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::{build_rope_cache, causal_mask};
use crate::models::architectures::qwen35::vision::Qwen35VisionRuntime;
use crate::models::shared::chat::{
    parse_qwen35_multimodal_control_content, parse_qwen35_thinking_control_content,
    parse_qwen35_tools_control_content, ChatMessage, ChatRole, Qwen35MultimodalInput,
    Qwen35MultimodalKind,
};
use crate::models::shared::device::DeviceProfile;
use crate::models::shared::weights::mlx;
use crate::tokenizer::Tokenizer;

const QWEN_VISION_START_TOKEN: &str = "<|vision_start|>";
const QWEN_IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
const QWEN_VIDEO_PAD_TOKEN: &str = "<|video_pad|>";

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    cache: Qwen35Cache,
    logits: Tensor,
    pos: usize,
    generated_ids: Vec<u32>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    im_start: u32,
    im_end: u32,
    image_pad: u32,
    video_pad: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl ChatTokenizer {
    fn load(model_dir: &Path, expected_vocab_size: Option<usize>) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, expected_vocab_size)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

        let id_for = |token: &str| -> Option<u32> {
            config.added_tokens_decoder.iter().find_map(|(id, entry)| {
                if entry.content == token {
                    id.parse().ok()
                } else {
                    None
                }
            })
        };

        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let eos = config
            .eos_token
            .as_deref()
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");
        let image_pad = id_for(QWEN_IMAGE_PAD_TOKEN).ok_or_else(|| {
            Error::TokenizationError("Missing <|image_pad|> token id".to_string())
        })?;
        let video_pad = id_for(QWEN_VIDEO_PAD_TOKEN).ok_or_else(|| {
            Error::TokenizationError("Missing <|video_pad|> token id".to_string())
        })?;

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                image_pad,
                video_pad,
                eos,
                eos_alt,
            },
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawQwen35TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    rms_norm_eps: f64,
    vocab_size: usize,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    linear_key_head_dim: Option<usize>,
    #[serde(default)]
    linear_value_head_dim: Option<usize>,
    #[serde(default)]
    linear_num_key_heads: Option<usize>,
    #[serde(default)]
    linear_num_value_heads: Option<usize>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    full_attention_interval: Option<usize>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    rope_theta: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone)]
struct Qwen35TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    tie_word_embeddings: bool,
    attention_bias: bool,
    rope_theta: f64,
    partial_rotary_factor: f64,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    layer_types: Vec<LayerType>,
}

impl Qwen35TextConfig {
    fn from_raw(raw: RawQwen35TextConfig) -> Result<Self> {
        let hidden_act = raw.hidden_act.as_deref().unwrap_or("silu");
        if hidden_act != "silu" {
            return Err(Error::InvalidInput(format!(
                "Unsupported Qwen3.5 hidden_act `{hidden_act}`; expected `silu`"
            )));
        }

        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads.max(1));

        let rope_theta = raw
            .rope_parameters
            .as_ref()
            .and_then(|p| p.rope_theta)
            .or(raw.rope_theta)
            .unwrap_or(10_000_000.0);

        let partial_rotary_factor = raw
            .rope_parameters
            .as_ref()
            .and_then(|p| p.partial_rotary_factor)
            .unwrap_or(0.25);

        let layer_types = if let Some(layer_types) = raw.layer_types {
            if layer_types.len() != raw.num_hidden_layers {
                return Err(Error::InvalidInput(format!(
                    "Qwen3.5 layer_types length {} does not match num_hidden_layers {}",
                    layer_types.len(),
                    raw.num_hidden_layers
                )));
            }
            layer_types
                .into_iter()
                .map(|kind| match kind.as_str() {
                    "full_attention" => Ok(LayerType::FullAttention),
                    "linear_attention" => Ok(LayerType::LinearAttention),
                    other => Err(Error::InvalidInput(format!(
                        "Unsupported Qwen3.5 layer type: {other}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let interval = raw.full_attention_interval.unwrap_or(4).max(1);
            (0..raw.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect()
        };

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_attention_heads: raw.num_attention_heads,
            num_hidden_layers: raw.num_hidden_layers,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            vocab_size: raw.vocab_size,
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            attention_bias: raw.attention_bias.unwrap_or(false),
            rope_theta,
            partial_rotary_factor,
            linear_conv_kernel_dim: raw.linear_conv_kernel_dim.unwrap_or(4).max(1),
            linear_key_head_dim: raw.linear_key_head_dim.unwrap_or(128),
            linear_value_head_dim: raw.linear_value_head_dim.unwrap_or(128),
            linear_num_key_heads: raw.linear_num_key_heads.unwrap_or(16).max(1),
            linear_num_value_heads: raw.linear_num_value_heads.unwrap_or(32).max(1),
            layer_types,
        })
    }

    fn rotary_dim(&self) -> usize {
        let mut rotary = ((self.head_dim as f64) * self.partial_rotary_factor).round() as usize;
        rotary = rotary.clamp(2, self.head_dim);
        if rotary % 2 == 1 {
            rotary = rotary.saturating_sub(1);
        }
        rotary
    }
}

fn parse_qwen35_config(config_str: &str) -> Result<Qwen35TextConfig> {
    let value: Value = serde_json::from_str(config_str)?;
    let text_config = value.get("text_config").cloned().unwrap_or(value);
    let raw: RawQwen35TextConfig = serde_json::from_value(text_config)?;
    Qwen35TextConfig::from_raw(raw)
}

enum Qwen35LayerCache {
    Full {
        k: Option<Tensor>,
        v: Option<Tensor>,
    },
    Linear {
        conv_state: Option<Tensor>,
        recurrent_state: Option<Tensor>,
    },
}

struct Qwen35Cache {
    layers: Vec<Qwen35LayerCache>,
}

impl Qwen35Cache {
    fn new(layer_types: &[LayerType]) -> Self {
        let layers = layer_types
            .iter()
            .map(|layer_type| match layer_type {
                LayerType::FullAttention => Qwen35LayerCache::Full { k: None, v: None },
                LayerType::LinearAttention => Qwen35LayerCache::Linear {
                    conv_state: None,
                    recurrent_state: None,
                },
            })
            .collect();
        Self { layers }
    }

    fn layer_mut(&mut self, idx: usize) -> Result<&mut Qwen35LayerCache> {
        self.layers.get_mut(idx).ok_or_else(|| {
            Error::InferenceError(format!("Invalid Qwen3.5 cache layer index: {idx}"))
        })
    }
}

struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
    one_centered: bool,
}

impl Qwen35RmsNorm {
    fn from_weight(dim: usize, eps: f64, one_centered: bool, weight: Tensor) -> Result<Self> {
        let got = weight.dims1()?;
        if got != dim {
            return Err(Error::ModelLoadError(format!(
                "RMSNorm weight mismatch: expected {dim}, got {got}"
            )));
        }
        Ok(Self {
            weight,
            eps,
            one_centered,
        })
    }

    fn load(dim: usize, eps: f64, one_centered: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked_dtype("weight", vb.dtype())?;
        Self::from_weight(dim, eps, one_centered, weight)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps as f32, x.device())?;
        let denom = variance.broadcast_add(&eps)?.sqrt()?;
        let normalized = x_f32.broadcast_div(&denom)?;

        let scale = if self.one_centered {
            let ones = Tensor::ones(self.weight.dims1()?, DType::F32, self.weight.device())?;
            self.weight.to_dtype(DType::F32)?.broadcast_add(&ones)?
        } else {
            self.weight.to_dtype(DType::F32)?
        };

        normalized
            .broadcast_mul(&scale)?
            .to_dtype(x_dtype)
            .map_err(Error::from)
    }
}

struct Qwen35RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNormGated {
    fn from_weight(dim: usize, eps: f64, weight: Tensor) -> Result<Self> {
        let got = weight.dims1()?;
        if got != dim {
            return Err(Error::ModelLoadError(format!(
                "RMSNormGated weight mismatch: expected {dim}, got {got}"
            )));
        }
        Ok(Self { weight, eps })
    }

    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked_dtype("weight", vb.dtype())?;
        Self::from_weight(dim, eps, weight)
    }

    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states_f32 = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps as f32, hidden_states.device())?;
        let denom = variance.broadcast_add(&eps)?.sqrt()?;
        let normalized = hidden_states_f32.broadcast_div(&denom)?;

        let weighted = normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let gated = weighted.broadcast_mul(&ops::silu(&gate.to_dtype(DType::F32)?)?)?;
        Ok(gated.to_dtype(input_dtype)?)
    }
}

struct Qwen35Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen35Mlp {
    fn load(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            mlx::load_linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(ops::silu(&gate)? * up)?)?)
    }
}

struct Qwen35FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

impl Qwen35FullAttention {
    fn load(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = if cfg.attention_bias {
            mlx::load_linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim * 2,
                vb.pp("q_proj"),
            )?
        } else {
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim * 2,
                vb.pp("q_proj"),
            )?
        };
        let k_proj = if cfg.attention_bias {
            mlx::load_linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?
        } else {
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?
        };
        let v_proj = if cfg.attention_bias {
            mlx::load_linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?
        } else {
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?
        };
        let o_proj = if cfg.attention_bias {
            mlx::load_linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?
        } else {
            mlx::load_linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?
        };
        let q_norm = Qwen35RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, true, vb.pp("q_norm"))?;
        let k_norm = Qwen35RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, true, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q_with_gate = self.q_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query = q_with_gate.narrow(3, 0, self.head_dim)?;
        let gate = q_with_gate.narrow(3, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((batch_size, seq_len, ()))?;

        let query = self.q_norm.forward(&query)?;
        let key = self.k_norm.forward(&self.k_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?)?;
        let value = self.v_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let mut query = query.transpose(1, 2)?.contiguous()?;
        let mut key = key.transpose(1, 2)?.contiguous()?;
        let value = value.transpose(1, 2)?.contiguous()?;

        if self.rotary_dim > 0 {
            let (cos_half, sin_half) = build_rope_cache(
                seq_len,
                self.rotary_dim,
                start_pos,
                rope_theta,
                x.device(),
                query.dtype(),
            )?;
            let cos = Tensor::cat(&[cos_half.clone(), cos_half], 1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            let sin = Tensor::cat(&[sin_half.clone(), sin_half], 1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            query = apply_partial_rotary(&query, &cos, &sin, self.rotary_dim)?.contiguous()?;
            key = apply_partial_rotary(&key, &cos, &sin, self.rotary_dim)?.contiguous()?;
        }

        let (k_all, v_all) = if let Some(layer_cache) = layer_cache {
            match layer_cache {
                Qwen35LayerCache::Full { k, v } => {
                    let next_k = if let Some(prev) = k.as_ref() {
                        Tensor::cat(&[prev.clone(), key.clone()], 2)?
                    } else {
                        key.clone()
                    };
                    let next_v = if let Some(prev) = v.as_ref() {
                        Tensor::cat(&[prev.clone(), value.clone()], 2)?
                    } else {
                        value.clone()
                    };
                    *k = Some(next_k.clone());
                    *v = Some(next_v.clone());
                    (next_k, next_v)
                }
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 full-attention layer received a linear-attention cache entry"
                            .to_string(),
                    ));
                }
            }
        } else {
            (key, value)
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let key = repeat_kv_bhsd(&k_all, n_rep)?.contiguous()?;
        let value = repeat_kv_bhsd(&v_all, n_rep)?.contiguous()?;

        let mut att = query.matmul(&key.transpose(2, 3)?.contiguous()?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), x.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;

        let total_len = key.dim(2)?;
        if seq_len > 1 || total_len > seq_len {
            let mask = causal_mask(seq_len, total_len, start_pos, x.device(), att.dtype())?
                .unsqueeze(1)?;
            att = att.broadcast_add(&mask)?;
        }

        let att = ops::softmax(&att.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(query.dtype())?;
        let att_out = att.matmul(&value)?;
        let att_out = att_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, ()))?;

        let att_out = att_out.broadcast_mul(&ops::sigmoid(&gate)?)?;
        Ok(self.o_proj.forward(&att_out)?)
    }
}

struct Qwen35LinearAttention {
    hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv: Conv1d,
    conv_weight: Tensor,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen35RmsNormGated,
    out_proj: Linear,
}

impl Qwen35LinearAttention {
    fn load(cfg: &Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;

        let conv_dim = key_dim * 2 + value_dim;
        let projection_size_qkv = key_dim * 2 + value_dim;

        let linear_vb = vb.pp("linear_attn");
        let in_proj_qkv = mlx::load_linear_no_bias(
            cfg.hidden_size,
            projection_size_qkv,
            linear_vb.pp("in_proj_qkv"),
        )?;
        let in_proj_z =
            mlx::load_linear_no_bias(cfg.hidden_size, value_dim, linear_vb.pp("in_proj_z"))?;
        let in_proj_b =
            mlx::load_linear_no_bias(cfg.hidden_size, num_v_heads, linear_vb.pp("in_proj_b"))?;
        let in_proj_a =
            mlx::load_linear_no_bias(cfg.hidden_size, num_v_heads, linear_vb.pp("in_proj_a"))?;

        let conv_weight =
            load_depthwise_conv_weight(conv_dim, conv_kernel_size, linear_vb.pp("conv1d"))?;
        let conv = Conv1d::new(
            conv_weight.unsqueeze(1)?,
            None,
            Conv1dConfig {
                groups: conv_dim,
                padding: conv_kernel_size.saturating_sub(1),
                ..Default::default()
            },
        );

        let dt_bias = linear_vb.get_unchecked_dtype("dt_bias", linear_vb.dtype())?;
        let a_log = linear_vb.get_unchecked_dtype("A_log", linear_vb.dtype())?;
        let norm = Qwen35RmsNormGated::load(head_v_dim, cfg.rms_norm_eps, linear_vb.pp("norm"))?;
        let out_proj =
            mlx::load_linear_no_bias(value_dim, cfg.hidden_size, linear_vb.pp("out_proj"))?;

        Ok(Self {
            hidden_size: cfg.hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv,
            conv_weight,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            dt_bias,
            a_log,
            norm,
            out_proj,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 linear-attention hidden size mismatch: expected {}, got {}",
                self.hidden_size, hidden_size
            )));
        }

        let (conv_state, recurrent_state) = if let Some(layer_cache) = layer_cache {
            match layer_cache {
                Qwen35LayerCache::Linear {
                    conv_state,
                    recurrent_state,
                } => (Some(conv_state), Some(recurrent_state)),
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 linear-attention layer received a full-attention cache entry"
                            .to_string(),
                    ));
                }
            }
        } else {
            (None, None)
        };

        let projected_qkv = self.in_proj_qkv.forward(hidden_states)?;
        let z = self.in_proj_z.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;
        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;

        let mut mixed_qkv = projected_qkv.transpose(1, 2)?.contiguous()?;

        let use_precomputed_states =
            conv_state.as_ref().and_then(|slot| slot.as_ref()).is_some() && seq_len == 1;

        mixed_qkv = if use_precomputed_states {
            let prev_state = conv_state
                .as_ref()
                .and_then(|slot| slot.as_ref())
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Missing Qwen3.5 linear conv state for decode step".to_string(),
                    )
                })?;
            let (out, next_state) = self.depthwise_conv_step(&mixed_qkv, prev_state)?;
            if let Some(slot) = conv_state {
                *slot = Some(next_state);
            }
            out
        } else {
            if let Some(slot) = conv_state {
                *slot = Some(self.build_conv_state(&mixed_qkv)?);
            }
            let out = self.conv.forward(&mixed_qkv.contiguous()?)?;
            let out = out.narrow(2, 0, seq_len)?;
            ops::silu(&out)?
        };

        mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let query = mixed_qkv.narrow(2, 0, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv
            .narrow(2, self.key_dim * 2, self.value_dim)?
            .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let beta = ops::sigmoid(&b)?;
        let dt = a.broadcast_add(&self.dt_bias.reshape((1, 1, self.num_v_heads))?)?;
        let softplus = stable_softplus(&dt)?;
        let a_exp =
            self.a_log
                .to_dtype(softplus.dtype())?
                .exp()?
                .reshape((1, 1, self.num_v_heads))?;
        let g = softplus.broadcast_mul(&a_exp)?.neg()?;

        let repeats = self.num_v_heads / self.num_k_heads;
        let query = if repeats > 1 {
            repeat_kv_bshd(&query, repeats)?
        } else {
            query
        };
        let key = if repeats > 1 {
            repeat_kv_bshd(&key, repeats)?
        } else {
            key
        };

        let initial_state = recurrent_state
            .as_ref()
            .and_then(|slot| slot.as_ref())
            .map(|t| t.clone());

        let (core_attn_out, last_recurrent_state) = self.recurrent_gated_delta_rule(
            &query,
            &key,
            &value,
            &g,
            &beta,
            initial_state.as_ref(),
        )?;

        if let Some(slot) = recurrent_state {
            *slot = Some(last_recurrent_state);
        }

        let core_flat = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_norm = self.norm.forward(&core_flat, &z_flat)?;
        let core_norm = core_norm.reshape((batch_size, seq_len, self.value_dim))?;

        Ok(self.out_proj.forward(&core_norm)?)
    }

    fn build_conv_state(&self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let (batch, channels, seq_len) = mixed_qkv.dims3()?;
        if seq_len >= self.conv_kernel_size {
            mixed_qkv
                .narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)
                .map_err(Error::from)
        } else {
            let pad = Tensor::zeros(
                (batch, channels, self.conv_kernel_size - seq_len),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            Tensor::cat(&[pad, mixed_qkv.clone()], 2).map_err(Error::from)
        }
    }

    fn depthwise_conv_step(
        &self,
        mixed_qkv: &Tensor,
        prev_state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_batch, channels, seq_len) = mixed_qkv.dims3()?;
        if seq_len != 1 {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 linear conv step expects seq_len=1, got {seq_len}"
            )));
        }
        if prev_state.dim(1)? != channels || prev_state.dim(2)? != self.conv_kernel_size {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.5 conv state shape: expected [batch,{channels},{}]",
                self.conv_kernel_size
            )));
        }

        let next_state = if self.conv_kernel_size > 1 {
            let tail = prev_state.narrow(2, 1, self.conv_kernel_size - 1)?;
            Tensor::cat(&[tail, mixed_qkv.clone()], 2)?
        } else {
            mixed_qkv.clone()
        };

        let weight = self.conv_weight.unsqueeze(0)?;
        let conv = next_state
            .broadcast_mul(&weight)?
            .sum_keepdim(2)
            .map_err(Error::from)?;
        let conv = ops::silu(&conv)?;
        Ok((conv, next_state))
    }

    fn recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let initial_dtype = query.dtype();
        let (batch_size, seq_len, num_heads, _head_k_dim) = query.dims4()?;

        let query = l2norm_last_dim(&query.to_dtype(DType::F32)?)?;
        let key = l2norm_last_dim(&key.to_dtype(DType::F32)?)?;
        let value = value.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?.exp()?;

        let scale = Tensor::new((self.head_k_dim as f32).sqrt(), query.device())?
            .to_dtype(query.dtype())?;
        let query = query.broadcast_div(&scale)?;

        let mut state = if let Some(state) = initial_state {
            state.to_dtype(DType::F32)?
        } else {
            Tensor::zeros(
                (batch_size, num_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                query.device(),
            )?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for idx in 0..seq_len {
            let q_t = query.narrow(1, idx, 1)?.squeeze(1)?;
            let k_t = key.narrow(1, idx, 1)?.squeeze(1)?;
            let v_t = value.narrow(1, idx, 1)?.squeeze(1)?;
            let beta_t = beta.narrow(1, idx, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, idx, 1)?.squeeze(1)?;

            let g_scale = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            state = state.broadcast_mul(&g_scale)?;

            let kv_mem = state
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)
                .map_err(Error::from)?;
            let delta = v_t
                .broadcast_sub(&kv_mem)?
                .broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?;
            let update = k_t
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?;
            state = state.broadcast_add(&update)?;

            let out_t = state
                .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)
                .map_err(Error::from)?;
            outputs.push(out_t.unsqueeze(1)?);
        }

        let output = Tensor::cat(&outputs, 1)?.to_dtype(initial_dtype)?;
        Ok((output, state))
    }
}

enum Qwen35TokenMixer {
    Full(Qwen35FullAttention),
    Linear(Qwen35LinearAttention),
}

struct Qwen35Layer {
    mixer: Qwen35TokenMixer,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
    mlp: Qwen35Mlp,
}

impl Qwen35Layer {
    fn load(cfg: &Qwen35TextConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let mixer = match cfg.layer_types[layer_idx] {
            LayerType::FullAttention => {
                Qwen35TokenMixer::Full(Qwen35FullAttention::load(cfg, vb.pp("self_attn"))?)
            }
            LayerType::LinearAttention => {
                Qwen35TokenMixer::Linear(Qwen35LinearAttention::load(cfg, vb.clone())?)
            }
        };

        let input_layernorm = Qwen35RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Qwen35RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Qwen35Mlp::load(cfg, vb.pp("mlp"))?;

        Ok(Self {
            mixer,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let x = match &self.mixer {
            Qwen35TokenMixer::Full(attn) => attn.forward(&x, start_pos, rope_theta, layer_cache)?,
            Qwen35TokenMixer::Linear(linear) => linear.forward(&x, layer_cache)?,
        };

        let x = x.broadcast_add(residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.broadcast_add(residual).map_err(Error::from)
    }
}

struct Qwen35Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen35Layer>,
    norm: Qwen35RmsNorm,
    lm_head: Linear,
    cfg: Qwen35TextConfig,
}

impl Qwen35Model {
    fn load(cfg: Qwen35TextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = mlx::load_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.language_model.embed_tokens"),
        )?;

        let lm_head = if vb.contains_tensor("lm_head.weight") {
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            return Err(Error::ModelLoadError(
                "Qwen3.5 checkpoint is missing lm_head.weight and tie_word_embeddings=false"
                    .to_string(),
            ));
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Qwen35Layer::load(
                &cfg,
                idx,
                vb.pp(format!("model.language_model.layers.{idx}")),
            )?;
            layers.push(layer);
        }

        let norm = Qwen35RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            vb.pp("model.language_model.norm"),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache)
    }

    fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();

        let mut cache = cache;
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = if let Some(cache_ref) = cache.as_deref_mut() {
                Some(cache_ref.layer_mut(idx)?)
            } else {
                None
            };
            x = layer.forward(&x, start_pos, self.cfg.rope_theta, layer_cache)?;
        }

        let x = self.norm.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    fn new_cache(&self) -> Qwen35Cache {
        Qwen35Cache::new(&self.cfg.layer_types)
    }
}

fn load_gguf_qmatmul<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &DeviceProfile,
    src: &str,
) -> Result<QMatMul> {
    let qtensor = content.tensor(reader, src, &device.device).map_err(|e| {
        Error::ModelLoadError(format!("Failed to load GGUF quantized tensor `{src}`: {e}"))
    })?;
    QMatMul::from_qtensor(qtensor).map_err(|e| {
        Error::ModelLoadError(format!("Failed to build quantized matmul for `{src}`: {e}"))
    })
}

fn load_gguf_transformed_qmatmul<R, F>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &DeviceProfile,
    dtype: DType,
    src: &str,
    transform: F,
) -> Result<QMatMul>
where
    R: std::io::Read + std::io::Seek,
    F: FnOnce(Tensor) -> Result<Tensor>,
{
    let source = content.tensor(reader, src, &device.device).map_err(|e| {
        Error::ModelLoadError(format!("Failed to load GGUF quantized tensor `{src}`: {e}"))
    })?;
    let source_dtype = source.dtype();
    let tensor = dequantize_qtensor_to_dtype(&source, device, dtype).map_err(|e| {
        Error::ModelLoadError(format!("Failed to dequantize GGUF tensor `{src}`: {e}"))
    })?;
    let tensor = transform(tensor)?;
    let tensor = if tensor.dtype() != dtype {
        tensor.to_dtype(dtype)?
    } else {
        tensor
    };
    let tensor = tensor.contiguous()?;
    let qtensor = QTensor::quantize(&tensor, source_dtype)
        .or_else(|_| QTensor::quantize(&tensor, GgmlDType::F16))
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to requantize transformed GGUF tensor `{src}`: {e}"
            ))
        })?;
    QMatMul::from_qtensor(qtensor).map_err(|e| {
        Error::ModelLoadError(format!("Failed to build quantized matmul for `{src}`: {e}"))
    })
}

struct Qwen35QuantizedMlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Qwen35QuantizedMlp {
    fn load<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &DeviceProfile,
        layer_src_prefix: &str,
    ) -> Result<Self> {
        let gate_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.ffn_gate.weight"),
        )?;
        let up_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.ffn_up.weight"),
        )?;
        let down_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.ffn_down.weight"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gated = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.down_proj.forward(&gated).map_err(Error::from)
    }
}

struct Qwen35QuantizedFullAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

impl Qwen35QuantizedFullAttention {
    fn load<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &DeviceProfile,
        dtype: DType,
        cfg: &Qwen35TextConfig,
        layer_src_prefix: &str,
    ) -> Result<Self> {
        let q_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.attn_q.weight"),
        )?;
        let k_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.attn_k.weight"),
        )?;
        let v_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.attn_v.weight"),
        )?;
        let o_proj = load_gguf_qmatmul(
            content,
            reader,
            device,
            &format!("{layer_src_prefix}.attn_output.weight"),
        )?;
        let q_norm = Qwen35RmsNorm::from_weight(
            cfg.head_dim,
            cfg.rms_norm_eps,
            true,
            load_gguf_one_centered_norm_as_delta(
                content,
                reader,
                device,
                dtype,
                &format!("{layer_src_prefix}.attn_q_norm.weight"),
            )?,
        )?;
        let k_norm = Qwen35RmsNorm::from_weight(
            cfg.head_dim,
            cfg.rms_norm_eps,
            true,
            load_gguf_one_centered_norm_as_delta(
                content,
                reader,
                device,
                dtype,
                &format!("{layer_src_prefix}.attn_k_norm.weight"),
            )?,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q_with_gate = self.q_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        ))?;
        let query = q_with_gate.narrow(3, 0, self.head_dim)?;
        let gate = q_with_gate.narrow(3, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((batch_size, seq_len, ()))?;

        let query = self.q_norm.forward(&query)?;
        let key = self.k_norm.forward(&self.k_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?)?;
        let value = self.v_proj.forward(x)?.reshape((
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let mut query = query.transpose(1, 2)?.contiguous()?;
        let mut key = key.transpose(1, 2)?.contiguous()?;
        let value = value.transpose(1, 2)?.contiguous()?;

        if self.rotary_dim > 0 {
            let (cos_half, sin_half) = build_rope_cache(
                seq_len,
                self.rotary_dim,
                start_pos,
                rope_theta,
                x.device(),
                query.dtype(),
            )?;
            let cos = Tensor::cat(&[cos_half.clone(), cos_half], 1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            let sin = Tensor::cat(&[sin_half.clone(), sin_half], 1)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            query = apply_partial_rotary(&query, &cos, &sin, self.rotary_dim)?.contiguous()?;
            key = apply_partial_rotary(&key, &cos, &sin, self.rotary_dim)?.contiguous()?;
        }

        let (k_all, v_all) = if let Some(layer_cache) = layer_cache {
            match layer_cache {
                Qwen35LayerCache::Full { k, v } => {
                    let next_k = if let Some(prev) = k.as_ref() {
                        Tensor::cat(&[prev.clone(), key.clone()], 2)?
                    } else {
                        key.clone()
                    };
                    let next_v = if let Some(prev) = v.as_ref() {
                        Tensor::cat(&[prev.clone(), value.clone()], 2)?
                    } else {
                        value.clone()
                    };
                    *k = Some(next_k.clone());
                    *v = Some(next_v.clone());
                    (next_k, next_v)
                }
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 full-attention layer received a linear-attention cache entry"
                            .to_string(),
                    ))
                }
            }
        } else {
            (key, value)
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let key = repeat_kv_bhsd(&k_all, n_rep)?.contiguous()?;
        let value = repeat_kv_bhsd(&v_all, n_rep)?.contiguous()?;

        let mut att = query.matmul(&key.transpose(2, 3)?.contiguous()?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), x.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;

        let total_len = key.dim(2)?;
        if seq_len > 1 || total_len > seq_len {
            let mask = causal_mask(seq_len, total_len, start_pos, x.device(), att.dtype())?
                .unsqueeze(1)?;
            att = att.broadcast_add(&mask)?;
        }

        let att = ops::softmax(&att.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(query.dtype())?;
        let att_out = att.matmul(&value)?;
        let att_out = att_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, ()))?;

        let att_out = att_out.broadcast_mul(&ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&att_out).map_err(Error::from)
    }
}

struct Qwen35QuantizedLinearAttention {
    hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv: Conv1d,
    conv_weight: Tensor,
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen35RmsNormGated,
    out_proj: QMatMul,
}

impl Qwen35QuantizedLinearAttention {
    fn load<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &DeviceProfile,
        dtype: DType,
        cfg: &Qwen35TextConfig,
        layer_src_prefix: &str,
    ) -> Result<Self> {
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let qkv_src = format!("{layer_src_prefix}.attn_qkv.weight");
        let in_proj_qkv =
            load_gguf_transformed_qmatmul(content, reader, device, dtype, &qkv_src, |tensor| {
                untile_linear_qkv_weight(tensor, cfg, &qkv_src)
            })?;

        let z_src = format!("{layer_src_prefix}.attn_gate.weight");
        let in_proj_z =
            load_gguf_transformed_qmatmul(content, reader, device, dtype, &z_src, |tensor| {
                untile_linear_v_rows(
                    tensor,
                    cfg.linear_num_key_heads,
                    cfg.linear_num_value_heads,
                    cfg.linear_value_head_dim,
                    &z_src,
                )
            })?;

        let beta_src = format!("{layer_src_prefix}.ssm_beta.weight");
        let in_proj_b =
            load_gguf_transformed_qmatmul(content, reader, device, dtype, &beta_src, |tensor| {
                untile_linear_v_rows(
                    tensor,
                    cfg.linear_num_key_heads,
                    cfg.linear_num_value_heads,
                    1,
                    &beta_src,
                )
            })?;

        let alpha_src = format!("{layer_src_prefix}.ssm_alpha.weight");
        let in_proj_a =
            load_gguf_transformed_qmatmul(content, reader, device, dtype, &alpha_src, |tensor| {
                untile_linear_v_rows(
                    tensor,
                    cfg.linear_num_key_heads,
                    cfg.linear_num_value_heads,
                    1,
                    &alpha_src,
                )
            })?;

        let dt_src = format!("{layer_src_prefix}.ssm_dt.bias");
        let dt_bias = untile_linear_v_vector(
            load_gguf_tensor(content, reader, device, dtype, &dt_src)?,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            &dt_src,
        )?;

        let ssm_a_src = format!("{layer_src_prefix}.ssm_a");
        let a_log = load_gguf_tensor(content, reader, device, dtype, &ssm_a_src)?;
        let a_log = a_log
            .to_dtype(DType::F32)?
            .neg()?
            .clamp(1e-30f64, f64::MAX)?
            .log()?;
        let a_log = if a_log.dtype() != dtype {
            a_log.to_dtype(dtype)?
        } else {
            a_log
        };
        let a_log = untile_linear_v_vector(
            a_log,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            &ssm_a_src,
        )?;

        let norm_weight = load_gguf_tensor(
            content,
            reader,
            device,
            dtype,
            &format!("{layer_src_prefix}.ssm_norm.weight"),
        )?;
        let norm = Qwen35RmsNormGated::from_weight(head_v_dim, cfg.rms_norm_eps, norm_weight)?;

        let out_proj_src = format!("{layer_src_prefix}.ssm_out.weight");
        let out_proj = load_gguf_transformed_qmatmul(
            content,
            reader,
            device,
            dtype,
            &out_proj_src,
            |tensor| untile_linear_out_proj_weight(tensor, cfg, &out_proj_src),
        )?;

        let conv_src = format!("{layer_src_prefix}.ssm_conv1d.weight");
        let conv_weight = untile_linear_conv_weight(
            load_gguf_tensor(content, reader, device, dtype, &conv_src)?,
            cfg,
            &conv_src,
        )?;
        let conv = Conv1d::new(
            conv_weight.unsqueeze(1)?,
            None,
            Conv1dConfig {
                groups: conv_dim,
                padding: conv_kernel_size.saturating_sub(1),
                ..Default::default()
            },
        );

        Ok(Self {
            hidden_size: cfg.hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv,
            conv_weight,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            dt_bias,
            a_log,
            norm,
            out_proj,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if hidden_size != self.hidden_size {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 linear-attention hidden size mismatch: expected {}, got {}",
                self.hidden_size, hidden_size
            )));
        }

        let (conv_state, recurrent_state) = if let Some(layer_cache) = layer_cache {
            match layer_cache {
                Qwen35LayerCache::Linear {
                    conv_state,
                    recurrent_state,
                } => (Some(conv_state), Some(recurrent_state)),
                _ => {
                    return Err(Error::InferenceError(
                        "Qwen3.5 linear-attention layer received a full-attention cache entry"
                            .to_string(),
                    ));
                }
            }
        } else {
            (None, None)
        };

        let projected_qkv = self.in_proj_qkv.forward(hidden_states)?;
        let z = self.in_proj_z.forward(hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;
        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;

        let mut mixed_qkv = projected_qkv.transpose(1, 2)?.contiguous()?;
        let use_precomputed_states =
            conv_state.as_ref().and_then(|slot| slot.as_ref()).is_some() && seq_len == 1;

        mixed_qkv = if use_precomputed_states {
            let prev_state = conv_state
                .as_ref()
                .and_then(|slot| slot.as_ref())
                .ok_or_else(|| {
                    Error::InferenceError(
                        "Missing Qwen3.5 linear conv state for decode step".to_string(),
                    )
                })?;
            let (out, next_state) = self.depthwise_conv_step(&mixed_qkv, prev_state)?;
            if let Some(slot) = conv_state {
                *slot = Some(next_state);
            }
            out
        } else {
            if let Some(slot) = conv_state {
                *slot = Some(self.build_conv_state(&mixed_qkv)?);
            }
            let out = self.conv.forward(&mixed_qkv.contiguous()?)?;
            let out = out.narrow(2, 0, seq_len)?;
            ops::silu(&out)?
        };

        mixed_qkv = mixed_qkv.transpose(1, 2)?;

        let query = mixed_qkv.narrow(2, 0, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let value = mixed_qkv
            .narrow(2, self.key_dim * 2, self.value_dim)?
            .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let beta = ops::sigmoid(&b)?;
        let dt = a.broadcast_add(&self.dt_bias.reshape((1, 1, self.num_v_heads))?)?;
        let softplus = stable_softplus(&dt)?;
        let a_exp =
            self.a_log
                .to_dtype(softplus.dtype())?
                .exp()?
                .reshape((1, 1, self.num_v_heads))?;
        let g = softplus.broadcast_mul(&a_exp)?.neg()?;

        let repeats = self.num_v_heads / self.num_k_heads;
        let query = if repeats > 1 {
            repeat_kv_bshd(&query, repeats)?
        } else {
            query
        };
        let key = if repeats > 1 {
            repeat_kv_bshd(&key, repeats)?
        } else {
            key
        };

        let initial_state = recurrent_state
            .as_ref()
            .and_then(|slot| slot.as_ref())
            .cloned();
        let (core_attn_out, last_recurrent_state) = self.recurrent_gated_delta_rule(
            &query,
            &key,
            &value,
            &g,
            &beta,
            initial_state.as_ref(),
        )?;

        if let Some(slot) = recurrent_state {
            *slot = Some(last_recurrent_state);
        }

        let core_flat = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_norm = self.norm.forward(&core_flat, &z_flat)?;
        let core_norm = core_norm.reshape((batch_size, seq_len, self.value_dim))?;
        self.out_proj.forward(&core_norm).map_err(Error::from)
    }

    fn build_conv_state(&self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let (batch, channels, seq_len) = mixed_qkv.dims3()?;
        if seq_len >= self.conv_kernel_size {
            mixed_qkv
                .narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)
                .map_err(Error::from)
        } else {
            let pad = Tensor::zeros(
                (batch, channels, self.conv_kernel_size - seq_len),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            Tensor::cat(&[pad, mixed_qkv.clone()], 2).map_err(Error::from)
        }
    }

    fn depthwise_conv_step(
        &self,
        mixed_qkv: &Tensor,
        prev_state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_batch, channels, seq_len) = mixed_qkv.dims3()?;
        if seq_len != 1 {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 linear conv step expects seq_len=1, got {seq_len}"
            )));
        }
        if prev_state.dim(1)? != channels || prev_state.dim(2)? != self.conv_kernel_size {
            return Err(Error::InferenceError(format!(
                "Invalid Qwen3.5 conv state shape: expected [batch,{channels},{}]",
                self.conv_kernel_size
            )));
        }

        let next_state = if self.conv_kernel_size > 1 {
            let tail = prev_state.narrow(2, 1, self.conv_kernel_size - 1)?;
            Tensor::cat(&[tail, mixed_qkv.clone()], 2)?
        } else {
            mixed_qkv.clone()
        };

        let weight = self.conv_weight.unsqueeze(0)?;
        let conv = next_state
            .broadcast_mul(&weight)?
            .sum_keepdim(2)
            .map_err(Error::from)?;
        let conv = ops::silu(&conv)?;
        Ok((conv, next_state))
    }

    fn recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let initial_dtype = query.dtype();
        let (batch_size, seq_len, num_heads, _head_k_dim) = query.dims4()?;

        let query = l2norm_last_dim(&query.to_dtype(DType::F32)?)?;
        let key = l2norm_last_dim(&key.to_dtype(DType::F32)?)?;
        let value = value.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?.exp()?;

        let scale = Tensor::new((self.head_k_dim as f32).sqrt(), query.device())?
            .to_dtype(query.dtype())?;
        let query = query.broadcast_div(&scale)?;

        let mut state = if let Some(state) = initial_state {
            state.to_dtype(DType::F32)?
        } else {
            Tensor::zeros(
                (batch_size, num_heads, self.head_k_dim, self.head_v_dim),
                DType::F32,
                query.device(),
            )?
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for idx in 0..seq_len {
            let q_t = query.narrow(1, idx, 1)?.squeeze(1)?;
            let k_t = key.narrow(1, idx, 1)?.squeeze(1)?;
            let v_t = value.narrow(1, idx, 1)?.squeeze(1)?;
            let beta_t = beta.narrow(1, idx, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, idx, 1)?.squeeze(1)?;

            let g_scale = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            state = state.broadcast_mul(&g_scale)?;

            let kv_mem = state
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)
                .map_err(Error::from)?;
            let delta = v_t
                .broadcast_sub(&kv_mem)?
                .broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?;
            let update = k_t
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?;
            state = state.broadcast_add(&update)?;

            let out_t = state
                .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)
                .map_err(Error::from)?;
            outputs.push(out_t.unsqueeze(1)?);
        }

        let output = Tensor::cat(&outputs, 1)?.to_dtype(initial_dtype)?;
        Ok((output, state))
    }
}

enum Qwen35QuantizedTokenMixer {
    Full(Qwen35QuantizedFullAttention),
    Linear(Qwen35QuantizedLinearAttention),
}

struct Qwen35QuantizedLayer {
    mixer: Qwen35QuantizedTokenMixer,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
    mlp: Qwen35QuantizedMlp,
}

impl Qwen35QuantizedLayer {
    fn load<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &DeviceProfile,
        dtype: DType,
        cfg: &Qwen35TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let src = format!("blk.{layer_idx}");
        let input_layernorm = Qwen35RmsNorm::from_weight(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            load_gguf_one_centered_norm_as_delta(
                content,
                reader,
                device,
                dtype,
                &format!("{src}.attn_norm.weight"),
            )?,
        )?;
        let post_attention_layernorm = Qwen35RmsNorm::from_weight(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            load_gguf_one_centered_norm_as_delta(
                content,
                reader,
                device,
                dtype,
                &format!("{src}.post_attention_norm.weight"),
            )?,
        )?;

        let mixer = match cfg.layer_types[layer_idx] {
            LayerType::FullAttention => Qwen35QuantizedTokenMixer::Full(
                Qwen35QuantizedFullAttention::load(content, reader, device, dtype, cfg, &src)?,
            ),
            LayerType::LinearAttention => Qwen35QuantizedTokenMixer::Linear(
                Qwen35QuantizedLinearAttention::load(content, reader, device, dtype, cfg, &src)?,
            ),
        };
        let mlp = Qwen35QuantizedMlp::load(content, reader, device, &src)?;

        Ok(Self {
            mixer,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        layer_cache: Option<&mut Qwen35LayerCache>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match &self.mixer {
            Qwen35QuantizedTokenMixer::Full(attn) => {
                attn.forward(&x, start_pos, rope_theta, layer_cache)?
            }
            Qwen35QuantizedTokenMixer::Linear(linear) => linear.forward(&x, layer_cache)?,
        };
        let x = x.broadcast_add(residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.broadcast_add(residual).map_err(Error::from)
    }
}

struct Qwen35QuantizedModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen35QuantizedLayer>,
    norm: Qwen35RmsNorm,
    lm_head: QMatMul,
    cfg: Qwen35TextConfig,
}

impl Qwen35QuantizedModel {
    fn load_gguf<R: std::io::Read + std::io::Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &DeviceProfile,
        dtype: DType,
        cfg: Qwen35TextConfig,
    ) -> Result<Self> {
        let embed_q = content
            .tensor(reader, "token_embd.weight", &device.device)
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load GGUF tensor `token_embd.weight`: {e}"
                ))
            })?;
        let embed_weight = dequantize_qtensor_to_dtype(&embed_q, device, dtype).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to dequantize GGUF tensor `token_embd.weight`: {e}"
            ))
        })?;
        let embed_tokens = Embedding::new(embed_weight, cfg.hidden_size);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(Qwen35QuantizedLayer::load(
                content, reader, device, dtype, &cfg, idx,
            )?);
        }

        let norm = Qwen35RmsNorm::from_weight(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            true,
            load_gguf_one_centered_norm_as_delta(
                content,
                reader,
                device,
                dtype,
                "output_norm.weight",
            )?,
        )?;

        let lm_head = if content.tensor_infos.contains_key("output.weight") {
            load_gguf_qmatmul(content, reader, device, "output.weight")?
        } else {
            load_gguf_qmatmul(content, reader, device, "token_embd.weight")?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache)
    }

    fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();
        let mut cache = cache;
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = if let Some(cache_ref) = cache.as_deref_mut() {
                Some(cache_ref.layer_mut(idx)?)
            } else {
                None
            };
            x = layer.forward(&x, start_pos, self.cfg.rope_theta, layer_cache)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x).map_err(Error::from)
    }

    fn new_cache(&self) -> Qwen35Cache {
        Qwen35Cache::new(&self.cfg.layer_types)
    }
}

enum Qwen35VisionSupport {
    MissingProjector,
    Ready {
        path: PathBuf,
        runtime: Qwen35VisionRuntime,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35GgufBackend {
    Quantized,
    Native,
}

enum Qwen35TextBackend {
    Dense {
        text_model: Qwen35Model,
    },
    QuantizedGguf {
        text_model: Mutex<Qwen35QuantizedModel>,
    },
}

#[derive(Debug, Clone)]
struct PromptBuildOutput {
    ids: Vec<u32>,
    multimodal: Vec<Qwen35MultimodalInput>,
}

pub struct Qwen35ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    backend: Qwen35TextBackend,
    vision_support: Qwen35VisionSupport,
}

impl Qwen35ChatModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        if let Some(gguf_path) = find_preferred_gguf(model_dir)? {
            return Self::load_gguf(model_dir, &gguf_path, device);
        }
        Self::load_safetensors(model_dir, device)
    }

    fn load_safetensors(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config = parse_qwen35_config(&config_str)?;
        let projection_dim = config.hidden_size;

        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;

        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = match dtype_override.as_deref().map(str::trim) {
            Some(raw) if !raw.is_empty() => device.select_dtype(Some(raw)),
            _ if device.kind.is_metal() => DType::F16,
            _ => device.select_dtype(None),
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| model_dir.join(f)).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? }
        } else {
            let fallback = find_single_safetensor(model_dir)?;
            unsafe { VarBuilder::from_mmaped_safetensors(&[fallback], dtype, &device.device)? }
        };

        let text_model = Qwen35Model::load(config, vb)?;
        let vision_support = if let Some(mmproj_path) = find_mmproj_gguf(model_dir)? {
            let runtime = Qwen35VisionRuntime::load(&mmproj_path, &device, dtype, projection_dim)?;
            info!(
                "Loaded Qwen3.5 multimodal projector runtime from {}",
                mmproj_path.display()
            );
            Qwen35VisionSupport::Ready {
                path: mmproj_path,
                runtime,
            }
        } else {
            Qwen35VisionSupport::MissingProjector
        };

        info!(
            "Loaded Qwen3.5 chat model on {:?} with dtype {:?}",
            device.kind, dtype
        );

        Ok(Self {
            device,
            tokenizer,
            backend: Qwen35TextBackend::Dense { text_model },
            vision_support,
        })
    }

    fn load_gguf(model_dir: &Path, gguf_path: &Path, device: DeviceProfile) -> Result<Self> {
        match select_qwen35_gguf_backend()? {
            Qwen35GgufBackend::Quantized => Self::load_gguf_quantized(model_dir, gguf_path, device),
            Qwen35GgufBackend::Native => Self::load_gguf_dense(model_dir, gguf_path, device),
        }
    }

    fn load_gguf_quantized(
        model_dir: &Path,
        gguf_path: &Path,
        device: DeviceProfile,
    ) -> Result<Self> {
        let mut reader = BufReader::new(fs::File::open(gguf_path)?);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF header: {e}")))?;
        let config = parse_qwen35_gguf_config(&content)?;
        let projection_dim = config.hidden_size;

        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;
        let dtype = select_qwen35_gguf_quantized_dtype(&content, &device);

        let text_model =
            Qwen35QuantizedModel::load_gguf(&content, &mut reader, &device, dtype, config)?;
        let vision_support = if let Some(mmproj_path) = find_mmproj_gguf(model_dir)? {
            let runtime = Qwen35VisionRuntime::load(&mmproj_path, &device, dtype, projection_dim)?;
            info!(
                "Loaded Qwen3.5 multimodal projector runtime from {}",
                mmproj_path.display()
            );
            Qwen35VisionSupport::Ready {
                path: mmproj_path,
                runtime,
            }
        } else {
            Qwen35VisionSupport::MissingProjector
        };
        info!(
            "Loaded Qwen3.5 quantized GGUF chat model on {:?} from {} with dtype {:?}",
            device.kind,
            gguf_path.display(),
            dtype
        );

        Ok(Self {
            device,
            tokenizer,
            backend: Qwen35TextBackend::QuantizedGguf {
                text_model: Mutex::new(text_model),
            },
            vision_support,
        })
    }

    fn load_gguf_dense(model_dir: &Path, gguf_path: &Path, device: DeviceProfile) -> Result<Self> {
        let mut reader = BufReader::new(fs::File::open(gguf_path)?);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF header: {e}")))?;
        let config = parse_qwen35_gguf_config(&content)?;
        let projection_dim = config.hidden_size;

        let tokenizer = ChatTokenizer::load(model_dir, Some(config.vocab_size))?;

        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = match dtype_override.as_deref().map(str::trim) {
            Some(raw) if !raw.is_empty() => device.select_dtype(Some(raw)),
            _ if device.kind.is_metal() => DType::F16,
            _ => device.select_dtype(None),
        };

        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        insert_gguf_tensor(
            &mut tensors,
            &content,
            &mut reader,
            &device,
            dtype,
            "token_embd.weight",
            "model.language_model.embed_tokens.weight",
        )?;
        let output_norm = load_gguf_one_centered_norm_as_delta(
            &content,
            &mut reader,
            &device,
            dtype,
            "output_norm.weight",
        )?;
        tensors.insert("model.language_model.norm.weight".to_string(), output_norm);
        if content.tensor_infos.contains_key("output.weight") {
            insert_gguf_tensor(
                &mut tensors,
                &content,
                &mut reader,
                &device,
                dtype,
                "output.weight",
                "lm_head.weight",
            )?;
        }

        for (idx, layer_type) in config.layer_types.iter().enumerate() {
            let src = format!("blk.{idx}");
            let dst = format!("model.language_model.layers.{idx}");

            let input_norm_src = format!("{src}.attn_norm.weight");
            let input_norm = load_gguf_one_centered_norm_as_delta(
                &content,
                &mut reader,
                &device,
                dtype,
                &input_norm_src,
            )?;
            tensors.insert(format!("{dst}.input_layernorm.weight"), input_norm);

            let post_attn_norm_src = format!("{src}.post_attention_norm.weight");
            let post_attn_norm = load_gguf_one_centered_norm_as_delta(
                &content,
                &mut reader,
                &device,
                dtype,
                &post_attn_norm_src,
            )?;
            tensors.insert(
                format!("{dst}.post_attention_layernorm.weight"),
                post_attn_norm,
            );
            insert_gguf_tensor(
                &mut tensors,
                &content,
                &mut reader,
                &device,
                dtype,
                &format!("{src}.ffn_gate.weight"),
                &format!("{dst}.mlp.gate_proj.weight"),
            )?;
            insert_gguf_tensor(
                &mut tensors,
                &content,
                &mut reader,
                &device,
                dtype,
                &format!("{src}.ffn_up.weight"),
                &format!("{dst}.mlp.up_proj.weight"),
            )?;
            insert_gguf_tensor(
                &mut tensors,
                &content,
                &mut reader,
                &device,
                dtype,
                &format!("{src}.ffn_down.weight"),
                &format!("{dst}.mlp.down_proj.weight"),
            )?;

            match layer_type {
                LayerType::FullAttention => {
                    insert_gguf_tensor(
                        &mut tensors,
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &format!("{src}.attn_q.weight"),
                        &format!("{dst}.self_attn.q_proj.weight"),
                    )?;
                    insert_gguf_tensor(
                        &mut tensors,
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &format!("{src}.attn_k.weight"),
                        &format!("{dst}.self_attn.k_proj.weight"),
                    )?;
                    insert_gguf_tensor(
                        &mut tensors,
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &format!("{src}.attn_v.weight"),
                        &format!("{dst}.self_attn.v_proj.weight"),
                    )?;
                    insert_gguf_tensor(
                        &mut tensors,
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &format!("{src}.attn_output.weight"),
                        &format!("{dst}.self_attn.o_proj.weight"),
                    )?;
                    let q_norm_src = format!("{src}.attn_q_norm.weight");
                    let q_norm = load_gguf_one_centered_norm_as_delta(
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &q_norm_src,
                    )?;
                    tensors.insert(format!("{dst}.self_attn.q_norm.weight"), q_norm);

                    let k_norm_src = format!("{src}.attn_k_norm.weight");
                    let k_norm = load_gguf_one_centered_norm_as_delta(
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &k_norm_src,
                    )?;
                    tensors.insert(format!("{dst}.self_attn.k_norm.weight"), k_norm);
                }
                LayerType::LinearAttention => {
                    let qkv_src = format!("{src}.attn_qkv.weight");
                    let qkv = load_gguf_tensor(&content, &mut reader, &device, dtype, &qkv_src)?;
                    let qkv = untile_linear_qkv_weight(qkv, &config, &qkv_src)?;
                    tensors.insert(format!("{dst}.linear_attn.in_proj_qkv.weight"), qkv);

                    let z_src = format!("{src}.attn_gate.weight");
                    let z = load_gguf_tensor(&content, &mut reader, &device, dtype, &z_src)?;
                    let z = untile_linear_v_rows(
                        z,
                        config.linear_num_key_heads,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                        &z_src,
                    )?;
                    tensors.insert(format!("{dst}.linear_attn.in_proj_z.weight"), z);

                    let beta_src = format!("{src}.ssm_beta.weight");
                    let beta = load_gguf_tensor(&content, &mut reader, &device, dtype, &beta_src)?;
                    let beta = untile_linear_v_rows(
                        beta,
                        config.linear_num_key_heads,
                        config.linear_num_value_heads,
                        1,
                        &beta_src,
                    )?;
                    tensors.insert(format!("{dst}.linear_attn.in_proj_b.weight"), beta);

                    let alpha_src = format!("{src}.ssm_alpha.weight");
                    let alpha =
                        load_gguf_tensor(&content, &mut reader, &device, dtype, &alpha_src)?;
                    let alpha = untile_linear_v_rows(
                        alpha,
                        config.linear_num_key_heads,
                        config.linear_num_value_heads,
                        1,
                        &alpha_src,
                    )?;
                    tensors.insert(format!("{dst}.linear_attn.in_proj_a.weight"), alpha);

                    let dt_src = format!("{src}.ssm_dt.bias");
                    let dt_bias = load_gguf_tensor(&content, &mut reader, &device, dtype, &dt_src)?;
                    let dt_bias = untile_linear_v_vector(
                        dt_bias,
                        config.linear_num_key_heads,
                        config.linear_num_value_heads,
                        &dt_src,
                    )?;
                    tensors.insert(format!("{dst}.linear_attn.dt_bias"), dt_bias);

                    let ssm_a_src = format!("{src}.ssm_a");
                    let a_log =
                        load_gguf_tensor(&content, &mut reader, &device, dtype, &ssm_a_src)?;
                    // In Qwen3.5 GGUF exports this tensor is stored as `-exp(A_log)`.
                    let a_log = a_log
                        .to_dtype(DType::F32)?
                        .neg()?
                        .clamp(1e-30f64, f64::MAX)?
                        .log()?;
                    let a_log = if a_log.dtype() != dtype {
                        a_log.to_dtype(dtype)?
                    } else {
                        a_log
                    };
                    let a_log = untile_linear_v_vector(
                        a_log,
                        config.linear_num_key_heads,
                        config.linear_num_value_heads,
                        &ssm_a_src,
                    )?;
                    tensors.insert(format!("{dst}.linear_attn.A_log"), a_log);

                    insert_gguf_tensor(
                        &mut tensors,
                        &content,
                        &mut reader,
                        &device,
                        dtype,
                        &format!("{src}.ssm_norm.weight"),
                        &format!("{dst}.linear_attn.norm.weight"),
                    )?;
                    let out_proj_src = format!("{src}.ssm_out.weight");
                    let out_proj =
                        load_gguf_tensor(&content, &mut reader, &device, dtype, &out_proj_src)?;
                    let out_proj = untile_linear_out_proj_weight(out_proj, &config, &out_proj_src)?;
                    tensors.insert(format!("{dst}.linear_attn.out_proj.weight"), out_proj);

                    let conv_src = format!("{src}.ssm_conv1d.weight");
                    let conv = load_gguf_tensor(&content, &mut reader, &device, dtype, &conv_src)?;
                    let conv = untile_linear_conv_weight(conv, &config, &conv_src)?;
                    tensors.insert(format!("{dst}.linear_attn.conv1d.weight"), conv);
                }
            }
        }

        let vb = VarBuilder::from_tensors(tensors, dtype, &device.device);
        let text_model = Qwen35Model::load(config, vb)?;
        let vision_support = if let Some(mmproj_path) = find_mmproj_gguf(model_dir)? {
            let runtime = Qwen35VisionRuntime::load(&mmproj_path, &device, dtype, projection_dim)?;
            info!(
                "Loaded Qwen3.5 multimodal projector runtime from {}",
                mmproj_path.display()
            );
            Qwen35VisionSupport::Ready {
                path: mmproj_path,
                runtime,
            }
        } else {
            Qwen35VisionSupport::MissingProjector
        };

        info!(
            "Loaded Qwen3.5 GGUF chat model on {:?} from {} with dtype {:?}",
            device.kind,
            gguf_path.display(),
            dtype
        );

        Ok(Self {
            device,
            tokenizer,
            backend: Qwen35TextBackend::Dense { text_model },
            vision_support,
        })
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_with_callback(messages, max_new_tokens, &mut no_op)
    }

    pub fn generate_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        let mut state = self.start_decode(messages, max_new_tokens)?;
        loop {
            let step = self.decode_step(&mut state)?;
            if !step.delta.is_empty() {
                for ch in step.delta.chars() {
                    let mut buf = [0u8; 4];
                    on_delta(ch.encode_utf8(&mut buf));
                }
            }
            if step.finished {
                return Ok(ChatGenerationOutput {
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                });
            }
        }
    }

    pub fn start_decode(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatDecodeState> {
        let prompt = self.build_prompt(messages)?;
        let mut cache = self.new_cache()?;
        let logits = if prompt.multimodal.is_empty() {
            let input_ids = Tensor::from_vec(
                prompt.ids.clone(),
                (1, prompt.ids.len()),
                &self.device.device,
            )?;
            self.forward_text(&input_ids, 0, Some(&mut cache))?
        } else {
            self.prefill_with_multimodal(&prompt.ids, &prompt.multimodal, &mut cache)?
        };
        let pos = logits.dim(1)?;

        Ok(ChatDecodeState {
            cache,
            logits,
            pos,
            generated_ids: Vec::new(),
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let logits = state.logits.i((0, state.logits.dim(1)? - 1))?;
        let next = argmax(&logits)?;

        if next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next)
        {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        state.generated_ids.push(next);
        let decoded = self.tokenizer.decode_text(&state.generated_ids)?;
        let delta = text_delta(&state.assembled, &decoded);
        state.assembled = decoded;

        if has_token_repetition_loop(&state.generated_ids) {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta,
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        state.logits = self.forward_text(&next_tensor, state.pos, Some(&mut state.cache))?;
        state.pos += 1;

        if state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
        }

        Ok(ChatDecodeStep {
            delta,
            text: state.assembled.trim().to_string(),
            tokens_generated: state.generated_ids.len(),
            finished: state.finished,
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        true
    }

    fn new_cache(&self) -> Result<Qwen35Cache> {
        match &self.backend {
            Qwen35TextBackend::Dense { text_model } => Ok(text_model.new_cache()),
            Qwen35TextBackend::QuantizedGguf { text_model, .. } => {
                let model = text_model.lock().map_err(|_| {
                    Error::InferenceError("Qwen3.5 quantized GGUF model mutex poisoned".to_string())
                })?;
                Ok(model.new_cache())
            }
        }
    }

    fn forward_text(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        match &self.backend {
            Qwen35TextBackend::Dense { text_model } => {
                text_model.forward(input_ids, start_pos, cache)
            }
            Qwen35TextBackend::QuantizedGguf { text_model, .. } => {
                let model = text_model.lock().map_err(|_| {
                    Error::InferenceError("Qwen3.5 quantized GGUF model mutex poisoned".to_string())
                })?;
                model.forward(input_ids, start_pos, cache)
            }
        }
    }

    fn text_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        match &self.backend {
            Qwen35TextBackend::Dense { text_model } => text_model.embeddings(input_ids),
            Qwen35TextBackend::QuantizedGguf { text_model, .. } => {
                let model = text_model.lock().map_err(|_| {
                    Error::InferenceError("Qwen3.5 quantized GGUF model mutex poisoned".to_string())
                })?;
                model.embeddings(input_ids)
            }
        }
    }

    fn forward_text_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen35Cache>,
    ) -> Result<Tensor> {
        match &self.backend {
            Qwen35TextBackend::Dense { text_model } => {
                text_model.forward_with_embeds(embeds, start_pos, cache)
            }
            Qwen35TextBackend::QuantizedGguf { text_model, .. } => {
                let model = text_model.lock().map_err(|_| {
                    Error::InferenceError("Qwen3.5 quantized GGUF model mutex poisoned".to_string())
                })?;
                model.forward_with_embeds(embeds, start_pos, cache)
            }
        }
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<PromptBuildOutput> {
        let prompt = build_qwen35_prompt_ids(&self.tokenizer, messages)?;
        self.ensure_multimodal_ready(&prompt)?;
        Ok(prompt)
    }

    fn ensure_multimodal_ready(&self, prompt: &PromptBuildOutput) -> Result<()> {
        let placeholder_count = prompt
            .ids
            .iter()
            .filter(|&&id| {
                id == self.tokenizer.specials.image_pad || id == self.tokenizer.specials.video_pad
            })
            .count();
        let media_count = prompt.multimodal.len();

        if placeholder_count == 0 && media_count == 0 {
            return Ok(());
        }
        if placeholder_count == 0 && media_count > 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 multimodal control payloads were provided, but no <|image_pad|>/<|video_pad|> placeholders were found in the prompt."
                    .to_string(),
            ));
        }
        if placeholder_count > 0 && media_count == 0 {
            return Err(Error::InvalidInput(
                "Qwen3.5 multimodal placeholders are present, but no media payloads were provided. Submit image/video parts (not placeholders only)."
                    .to_string(),
            ));
        }
        if placeholder_count != media_count {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 multimodal placeholder/media mismatch: prompt has {placeholder_count} placeholders, but {media_count} media payloads were provided."
            )));
        }

        match &self.vision_support {
            Qwen35VisionSupport::MissingProjector => Err(Error::InvalidInput(
                "Qwen3.5 multimodal input requested, but projector weights are missing. Download `mmproj-F16.gguf` for this model variant first.".to_string(),
            )),
            Qwen35VisionSupport::Ready { .. } => Ok(()),
        }
    }

    fn prefill_with_multimodal(
        &self,
        prompt_ids: &[u32],
        multimodal: &[Qwen35MultimodalInput],
        cache: &mut Qwen35Cache,
    ) -> Result<Tensor> {
        let runtime = match &self.vision_support {
            Qwen35VisionSupport::Ready { runtime, .. } => runtime,
            Qwen35VisionSupport::MissingProjector => {
                return Err(Error::InvalidInput(
                    "Qwen3.5 multimodal input requested, but projector weights are missing."
                        .to_string(),
                ))
            }
        };

        let input_ids = Tensor::from_vec(
            prompt_ids.to_vec(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;
        let input_embeds = self.text_embeddings(&input_ids)?;

        let placeholder_positions = prompt_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| {
                if id == self.tokenizer.specials.image_pad {
                    Some((idx, Qwen35MultimodalKind::Image))
                } else if id == self.tokenizer.specials.video_pad {
                    Some((idx, Qwen35MultimodalKind::Video))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if placeholder_positions.len() != multimodal.len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3.5 multimodal placeholder/media mismatch during prefill: placeholders={}, media={}",
                placeholder_positions.len(),
                multimodal.len()
            )));
        }

        let mut visual_embeds = Vec::with_capacity(multimodal.len());
        for media in multimodal {
            visual_embeds.push(runtime.encode_media(media)?);
        }

        let mut merged = Vec::with_capacity(placeholder_positions.len() * 2 + 1);
        let mut start = 0usize;
        for ((position, placeholder_kind), (media, visual)) in placeholder_positions
            .iter()
            .zip(multimodal.iter().zip(visual_embeds.iter()))
        {
            if *placeholder_kind != media.kind {
                return Err(Error::InvalidInput(format!(
                    "Qwen3.5 multimodal kind mismatch at placeholder {}: prompt expects {:?}, payload is {:?}",
                    position, placeholder_kind, media.kind
                )));
            }
            if *position > start {
                merged.push(input_embeds.narrow(1, start, position - start)?);
            }
            merged.push(visual.unsqueeze(0)?);
            start = position + 1;
        }
        if start < prompt_ids.len() {
            merged.push(input_embeds.narrow(1, start, prompt_ids.len() - start)?);
        }

        let fused_embeds = Tensor::cat(&merged, 1)?;
        self.forward_text_with_embeds(&fused_embeds, 0, Some(cache))
    }
}

fn build_qwen35_prompt_ids(
    tokenizer: &ChatTokenizer,
    messages: &[ChatMessage],
) -> Result<PromptBuildOutput> {
    if messages.is_empty() {
        return Err(Error::InvalidInput(
            "Chat request must include at least one message".to_string(),
        ));
    }

    let mut enable_thinking = None;
    let mut tools: Option<Vec<Value>> = None;
    let mut pending_multimodal = VecDeque::<Vec<Qwen35MultimodalInput>>::new();
    let mut prompt_messages = Vec::with_capacity(messages.len());

    #[derive(Clone)]
    struct PromptMessageEntry {
        message: ChatMessage,
        multimodal: Vec<Qwen35MultimodalInput>,
    }

    for message in messages {
        if matches!(message.role, ChatRole::System) {
            if let Some(control) = parse_qwen35_thinking_control_content(&message.content) {
                enable_thinking = Some(control);
                continue;
            }
            if let Some(control) = parse_qwen35_tools_control_content(&message.content) {
                if !control.is_empty() {
                    tools = Some(control);
                }
                continue;
            }
            if let Some(control) = parse_qwen35_multimodal_control_content(&message.content) {
                pending_multimodal.push_back(control);
                continue;
            }
        }
        let multimodal = pending_multimodal.pop_front().unwrap_or_default();
        prompt_messages.push(PromptMessageEntry {
            message: message.clone(),
            multimodal,
        });
    }
    if !pending_multimodal.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3.5 multimodal control payloads were provided without a following chat message."
                .to_string(),
        ));
    }

    let mut ids = Vec::new();
    let mut multimodal = Vec::new();
    let mut consumed_first_system = false;
    if let Some(tools) = tools.as_ref().filter(|entries| !entries.is_empty()) {
        let first_system = prompt_messages.first().and_then(|entry| {
            if matches!(entry.message.role, ChatRole::System) {
                consumed_first_system = true;
                Some(entry.message.content.trim().to_string())
            } else {
                None
            }
        });

        let tools_system_content = qwen35_tools_system_content(tools, first_system.as_deref());
        if !tools_system_content.trim().is_empty() {
            ids.push(tokenizer.specials.im_start);
            ids.extend(tokenizer.encode_text("system\n")?);
            ids.extend(tokenizer.encode_text(&tools_system_content)?);
            ids.push(tokenizer.specials.im_end);
            ids.extend(tokenizer.encode_text("\n")?);
        }
    } else if !matches!(
        prompt_messages.first().map(|m| &m.message.role),
        Some(ChatRole::System)
    ) {
        prompt_messages.insert(
            0,
            PromptMessageEntry {
                message: ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful assistant.".to_string(),
                },
                multimodal: Vec::new(),
            },
        );
    }

    let iter_start = if consumed_first_system { 1 } else { 0 };
    for entry in prompt_messages.iter().skip(iter_start) {
        let content = if matches!(entry.message.role, ChatRole::Assistant) {
            strip_think_blocks(entry.message.content.trim())
        } else {
            entry.message.content.trim().to_string()
        };

        if content.is_empty() {
            if !entry.multimodal.is_empty() {
                return Err(Error::InvalidInput(
                    "Qwen3.5 multimodal control payload is attached to an empty message."
                        .to_string(),
                ));
            }
            continue;
        }

        ids.push(tokenizer.specials.im_start);
        ids.extend(tokenizer.encode_text(&format!("{}\n", entry.message.role.as_prompt_role()))?);
        ids.extend(tokenizer.encode_text(&content)?);
        ids.push(tokenizer.specials.im_end);
        ids.extend(tokenizer.encode_text("\n")?);
        multimodal.extend(entry.multimodal.iter().cloned());
    }

    ids.push(tokenizer.specials.im_start);
    ids.extend(tokenizer.encode_text("assistant\n")?);
    if enable_thinking.unwrap_or(false) {
        ids.extend(tokenizer.encode_text("<think>\n")?);
    } else {
        ids.extend(tokenizer.encode_text("<think>\n\n</think>\n\n")?);
    }

    Ok(PromptBuildOutput { ids, multimodal })
}

fn qwen35_tools_system_content(tools: &[Value], first_system_content: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        out.push('\n');
        let tool_json = serde_json::to_string(tool).unwrap_or_else(|_| "{}".to_string());
        out.push_str(&tool_json);
    }
    out.push_str("\n</tools>");
    out.push_str(
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
    );

    if let Some(content) = first_system_content
        .map(str::trim)
        .filter(|content| !content.is_empty())
    {
        out.push_str("\n\n");
        out.push_str(content);
    }

    out
}

fn messages_contain_vision_placeholders(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|message| {
        let content = message.content.as_str();
        content.contains(QWEN_VISION_START_TOKEN)
            || content.contains(QWEN_IMAGE_PAD_TOKEN)
            || content.contains(QWEN_VIDEO_PAD_TOKEN)
    })
}

fn load_depthwise_conv_weight(
    conv_dim: usize,
    kernel_size: usize,
    vb: VarBuilder,
) -> Result<Tensor> {
    let mut weight = vb.get_unchecked_dtype("weight", vb.dtype())?;

    if let Ok(dims) = weight.dims3() {
        let expected_c1k = (conv_dim, 1, kernel_size);
        let expected_ck1 = (conv_dim, kernel_size, 1);
        if dims == expected_ck1 {
            weight = weight.permute((0, 2, 1))?;
        } else if dims != expected_c1k {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 conv weight shape mismatch: got={dims:?}, expected {expected_c1k:?} or {expected_ck1:?}"
            )));
        }
        return weight.squeeze(1).map_err(Error::from);
    }

    if let Ok(dims2) = weight.dims2() {
        let expected = (conv_dim, kernel_size);
        if dims2 != expected {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 conv weight shape mismatch: got={dims2:?}, expected {expected:?}"
            )));
        }
        return Ok(weight);
    }

    Err(Error::ModelLoadError(
        "Qwen3.5 conv weight must be a 2D/3D tensor".to_string(),
    ))
}

fn l2norm_last_dim(x: &Tensor) -> Result<Tensor> {
    let sq = x.sqr()?;
    let sum = sq.sum_keepdim(D::Minus1)?;
    let eps = Tensor::new(1e-6f32, x.device())?;
    let denom = sum.broadcast_add(&eps)?.sqrt()?;
    x.broadcast_div(&denom).map_err(Error::from)
}

fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_pos = x_f32.clamp(0.0f64, f64::MAX)?;
    let x_neg_abs = x_f32.abs()?.neg()?;
    let exp = x_neg_abs.exp()?;
    let one = Tensor::ones(exp.shape(), exp.dtype(), exp.device())?;
    let log1p = exp.broadcast_add(&one)?.log()?;
    x_pos.broadcast_add(&log1p).map_err(Error::from)
}

// HF parity helper for `repeat_kv` in full-attention path:
// input/output layout is [batch, heads, seq_len, head_dim].
fn repeat_kv_bhsd(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    candle_repeat_kv(xs.clone(), n_rep).map_err(Error::from)
}

// Repeat heads for tensors laid out as [batch, seq_len, heads, head_dim].
fn repeat_kv_bshd(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let xs_bhsd = xs.transpose(1, 2)?;
    repeat_kv_bhsd(&xs_bhsd, n_rep)?
        .transpose(1, 2)
        .map_err(Error::from)
}

fn apply_partial_rotary(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Ok(x.clone());
    }

    let x_rot = x.narrow(3, 0, rotary_dim)?;
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(3, rotary_dim, head_dim - rotary_dim)?)
    } else {
        None
    };

    let half = rotary_dim / 2;
    let x1 = x_rot.narrow(3, 0, half)?;
    let x2 = x_rot.narrow(3, half, half)?;
    let neg_one = Tensor::new(-1.0f32, x.device())?.to_dtype(x.dtype())?;
    let rotated = Tensor::cat(&[x2.broadcast_mul(&neg_one)?, x1], 3)?;

    let x_rot = x_rot.broadcast_mul(cos)?;
    let x_rot = x_rot.broadcast_add(&rotated.broadcast_mul(sin)?)?;

    if let Some(pass) = x_pass {
        Tensor::cat(&[x_rot, pass], 3).map_err(Error::from)
    } else {
        Ok(x_rot)
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

    if let Some(close_idx) = output.find(close) {
        let has_open_before_close = output[..close_idx].find(open).is_some();
        if !has_open_before_close {
            let start = close_idx + close.len();
            output = output[start..].to_string();
        }
    }

    loop {
        let Some(start) = output.find(open) else {
            break;
        };

        let search_from = start + open.len();
        if let Some(end_rel) = output[search_from..].find(close) {
            let end = search_from + end_rel + close.len();
            output.replace_range(start..end, "");
            continue;
        }

        output.truncate(start);
        break;
    }

    output.replace(close, " ").trim().to_string()
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let values = match logits.rank() {
        1 => logits.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
            )));
        }
    };

    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| Error::InferenceError("Empty logits".to_string()))?;

    Ok(idx as u32)
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }

    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

fn has_suffix_repeat(ids: &[u32], span: usize, repeats: usize) -> bool {
    if span == 0 || repeats < 2 || ids.len() < span * repeats {
        return false;
    }
    let tail_start = ids.len() - span;
    let tail = &ids[tail_start..];
    (2..=repeats).all(|rep| {
        let start = ids.len() - (span * rep);
        &ids[start..start + span] == tail
    })
}

fn has_token_repetition_loop(ids: &[u32]) -> bool {
    // Catch common degenerate loops from greedy decode where the same token span
    // is emitted repeatedly (frequent in tiny reasoning models).
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

fn gguf_md_get<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing GGUF metadata key: {key}")))
}

fn gguf_md_u32(content: &gguf_file::Content, key: &str) -> Result<usize> {
    gguf_md_get(content, key)?
        .to_u32()
        .map(|v| v as usize)
        .map_err(|e| Error::ModelLoadError(format!("Invalid GGUF metadata value for {key}: {e}")))
}

fn gguf_md_f64(content: &gguf_file::Content, key: &str) -> Result<f64> {
    gguf_md_get(content, key)?
        .to_f32()
        .map(|v| v as f64)
        .map_err(|e| Error::ModelLoadError(format!("Invalid GGUF metadata value for {key}: {e}")))
}

fn gguf_md_u32_opt(content: &gguf_file::Content, key: &str) -> Option<usize> {
    content
        .metadata
        .get(key)
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
}

fn parse_qwen35_gguf_config(content: &gguf_file::Content) -> Result<Qwen35TextConfig> {
    let arch = gguf_md_get(content, "general.architecture")?
        .to_string()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Invalid GGUF metadata value for general.architecture: {e}"
            ))
        })?;
    if arch != "qwen35" {
        return Err(Error::ModelLoadError(format!(
            "Unsupported GGUF architecture `{arch}` for Qwen3.5 loader; expected `qwen35`"
        )));
    }

    let hidden_size = gguf_md_u32(content, "qwen35.embedding_length")?;
    let num_hidden_layers = gguf_md_u32(content, "qwen35.block_count")?;
    let token_embd = content
        .tensor_infos
        .get("token_embd.weight")
        .ok_or_else(|| Error::ModelLoadError("Missing token_embd.weight tensor".to_string()))?;
    let token_embd_dims = token_embd.shape.dims();
    let vocab_size = match token_embd_dims {
        [first, second] if *first == hidden_size => *second,
        [first, _second] => *first,
        _ => {
            return Err(Error::ModelLoadError(format!(
                "Unexpected token_embd.weight shape in GGUF: {:?}",
                token_embd_dims
            )));
        }
    };

    let mut layer_types = Vec::with_capacity(num_hidden_layers);
    let mut first_linear_idx = None;
    for idx in 0..num_hidden_layers {
        let prefix = format!("blk.{idx}");
        let has_full = content
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_q.weight"));
        let has_linear = content
            .tensor_infos
            .contains_key(&format!("{prefix}.attn_qkv.weight"));
        match (has_full, has_linear) {
            (true, false) => layer_types.push("full_attention".to_string()),
            (false, true) => {
                if first_linear_idx.is_none() {
                    first_linear_idx = Some(idx);
                }
                layer_types.push("linear_attention".to_string())
            }
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unable to classify GGUF Qwen3.5 layer {idx}: expected either full-attention tensors or linear-attention tensors"
                )));
            }
        }
    }

    let (linear_key_head_dim, linear_value_head_dim, linear_num_key_heads, linear_num_value_heads) =
        if let Some(idx) = first_linear_idx {
            let prefix = format!("blk.{idx}");
            let qkv_info = content
                .tensor_infos
                .get(&format!("{prefix}.attn_qkv.weight"))
                .ok_or_else(|| {
                    Error::ModelLoadError(format!("Missing tensor {}.attn_qkv.weight", prefix))
                })?;
            let gate_info = content
                .tensor_infos
                .get(&format!("{prefix}.attn_gate.weight"))
                .ok_or_else(|| {
                    Error::ModelLoadError(format!("Missing tensor {}.attn_gate.weight", prefix))
                })?;
            let beta_info = content
                .tensor_infos
                .get(&format!("{prefix}.ssm_beta.weight"))
                .ok_or_else(|| {
                    Error::ModelLoadError(format!("Missing tensor {}.ssm_beta.weight", prefix))
                })?;

            let infer_out_features = |shape: &[usize]| -> Result<usize> {
                match shape {
                    [out, inp] if *inp == hidden_size => Ok(*out),
                    [inp, out] if *inp == hidden_size => Ok(*out),
                    [out, _inp] => Ok(*out),
                    dims => Err(Error::ModelLoadError(format!(
                        "Unexpected linear tensor shape in GGUF: {dims:?}"
                    ))),
                }
            };

            let qkv_out = infer_out_features(qkv_info.shape.dims())?;
            let value_dim = infer_out_features(gate_info.shape.dims())?;
            let beta_out = infer_out_features(beta_info.shape.dims())?;
            if qkv_out <= value_dim || (qkv_out - value_dim) % 2 != 0 {
                return Err(Error::ModelLoadError(format!(
                "Invalid qwen35 linear projection dims from GGUF: attn_qkv_out={qkv_out}, attn_gate_out={value_dim}"
            )));
            }
            let key_dim = (qkv_out - value_dim) / 2;

            let md_state_size = gguf_md_u32_opt(content, "qwen35.ssm.state_size");
            let md_group_count = gguf_md_u32_opt(content, "qwen35.ssm.group_count");
            let md_time_step_rank = gguf_md_u32_opt(content, "qwen35.ssm.time_step_rank");
            let md_inner_size = gguf_md_u32_opt(content, "qwen35.ssm.inner_size");

            let linear_num_value_heads = md_time_step_rank.unwrap_or(beta_out);
            if linear_num_value_heads == 0 || value_dim % linear_num_value_heads != 0 {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5 linear value dim {value_dim} is not divisible by value head count {linear_num_value_heads}"
                )));
            }
            if beta_out != linear_num_value_heads {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5 linear beta projection out dim {beta_out} does not match linear_num_value_heads {linear_num_value_heads}"
                )));
            }
            if let Some(inner_size) = md_inner_size {
                if inner_size != value_dim {
                    return Err(Error::ModelLoadError(format!(
                        "Qwen3.5 GGUF ssm.inner_size metadata mismatch: expected {value_dim}, got {inner_size}"
                    )));
                }
            }
            let linear_value_head_dim = value_dim / linear_num_value_heads;

            let linear_num_key_heads = md_group_count
                .or_else(|| gguf_md_u32_opt(content, "qwen35.attention.head_count_kv"))
                .unwrap_or(linear_num_value_heads);
            if linear_num_key_heads == 0 || key_dim % linear_num_key_heads != 0 {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5 linear key dim {key_dim} is not divisible by key head count {linear_num_key_heads}"
                )));
            }
            let inferred_key_head_dim = key_dim / linear_num_key_heads;
            let linear_key_head_dim = if let Some(state_size) = md_state_size {
                if state_size != inferred_key_head_dim {
                    return Err(Error::ModelLoadError(format!(
                        "Qwen3.5 GGUF ssm.state_size metadata mismatch: expected {inferred_key_head_dim}, got {state_size}"
                    )));
                }
                state_size
            } else {
                inferred_key_head_dim
            };

            (
                linear_key_head_dim,
                linear_value_head_dim,
                linear_num_key_heads,
                linear_num_value_heads,
            )
        } else {
            let linear_num_key_heads = gguf_md_u32_opt(content, "qwen35.ssm.group_count")
                .or_else(|| gguf_md_u32_opt(content, "qwen35.attention.head_count_kv"))
                .unwrap_or(1);
            let linear_num_value_heads = gguf_md_u32_opt(content, "qwen35.ssm.time_step_rank")
                .or_else(|| gguf_md_u32_opt(content, "qwen35.ssm.group_count"))
                .unwrap_or(1);
            let linear_key_head_dim = gguf_md_u32_opt(content, "qwen35.ssm.state_size")
                .unwrap_or(gguf_md_u32(content, "qwen35.attention.key_length")?);
            let linear_value_head_dim = gguf_md_u32_opt(content, "qwen35.ssm.inner_size")
                .map(|inner| inner / linear_num_value_heads.max(1))
                .unwrap_or(gguf_md_u32(content, "qwen35.attention.value_length")?);
            (
                linear_key_head_dim,
                linear_value_head_dim,
                linear_num_key_heads,
                linear_num_value_heads,
            )
        };

    let raw = RawQwen35TextConfig {
        hidden_size,
        intermediate_size: gguf_md_u32(content, "qwen35.feed_forward_length")?,
        num_attention_heads: gguf_md_u32(content, "qwen35.attention.head_count")?,
        num_hidden_layers,
        num_key_value_heads: gguf_md_u32(content, "qwen35.attention.head_count_kv")?,
        head_dim: Some(gguf_md_u32(content, "qwen35.attention.key_length")?),
        rms_norm_eps: gguf_md_f64(content, "qwen35.attention.layer_norm_rms_epsilon")?,
        vocab_size,
        tie_word_embeddings: Some(!content.tensor_infos.contains_key("output.weight")),
        attention_bias: Some(false),
        hidden_act: Some("silu".to_string()),
        linear_conv_kernel_dim: Some(gguf_md_u32(content, "qwen35.ssm.conv_kernel")?),
        linear_key_head_dim: Some(linear_key_head_dim),
        linear_value_head_dim: Some(linear_value_head_dim),
        linear_num_key_heads: Some(linear_num_key_heads),
        linear_num_value_heads: Some(linear_num_value_heads),
        layer_types: Some(layer_types),
        full_attention_interval: gguf_md_u32_opt(content, "qwen35.full_attention_interval"),
        rope_parameters: None,
        rope_theta: Some(gguf_md_f64(content, "qwen35.rope.freq_base")?),
    };

    Qwen35TextConfig::from_raw(raw)
}

fn dequantize_qtensor_to_dtype(
    qtensor: &QTensor,
    device: &DeviceProfile,
    dtype: DType,
) -> Result<Tensor> {
    let mut tensor = if dtype == DType::F16 {
        qtensor.dequantize_f16(&device.device)?
    } else {
        qtensor.dequantize(&device.device)?
    };
    if tensor.dtype() != dtype {
        tensor = tensor.to_dtype(dtype)?;
    }
    Ok(tensor)
}

fn gguf_general_dtype(content: &gguf_file::Content) -> Option<DType> {
    let value = content.metadata.get("general.dtype")?.to_u32().ok()?;
    match value {
        0 => Some(DType::F32),
        1 => Some(DType::F16),
        _ => None,
    }
}

fn select_qwen35_gguf_quantized_dtype(
    content: &gguf_file::Content,
    device: &DeviceProfile,
) -> DType {
    let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
        .ok()
        .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
    let requested = match dtype_override.as_deref().map(str::trim) {
        Some(raw) if !raw.is_empty() => device.select_dtype(Some(raw)),
        _ => gguf_general_dtype(content).unwrap_or(DType::F16),
    };
    // Candle's quantized Metal kernel requires F32 activation input for QTensor matmul.
    // Keep quantized GGUF Qwen3.5 on Metal in F32 to avoid runtime dtype assertion panics.
    if device.kind.is_metal() {
        DType::F32
    } else {
        requested
    }
}

fn load_gguf_tensor<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &DeviceProfile,
    dtype: DType,
    src: &str,
) -> Result<Tensor> {
    let qtensor = content
        .tensor(&mut *reader, src, &device.device)
        .map_err(|e| {
            Error::ModelLoadError(format!("Missing or invalid GGUF tensor `{src}`: {e}"))
        })?;
    let tensor = dequantize_qtensor_to_dtype(&qtensor, device, dtype).map_err(|e| {
        Error::ModelLoadError(format!("Failed to dequantize GGUF tensor `{src}`: {e}"))
    })?;
    Ok(tensor)
}

fn insert_gguf_tensor<R: std::io::Read + std::io::Seek>(
    tensors: &mut HashMap<String, Tensor>,
    content: &gguf_file::Content,
    reader: &mut R,
    device: &DeviceProfile,
    dtype: DType,
    src: &str,
    dst: &str,
) -> Result<()> {
    let tensor = load_gguf_tensor(content, reader, device, dtype, src)?;
    tensors.insert(dst.to_string(), tensor);
    Ok(())
}

fn load_gguf_one_centered_norm_as_delta<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &DeviceProfile,
    dtype: DType,
    src: &str,
) -> Result<Tensor> {
    let weight = load_gguf_tensor(content, reader, device, dtype, src)?;
    let ones = Tensor::ones(weight.shape(), weight.dtype(), weight.device())?;
    weight.broadcast_sub(&ones).map_err(Error::from)
}

fn untile_linear_v_rows(
    weight: Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    head_v_dim: usize,
    tensor_name: &str,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(weight);
    }
    if num_k_heads == 0 || num_v_heads == 0 || num_v_heads % num_k_heads != 0 {
        return Err(Error::ModelLoadError(format!(
            "Invalid linear head mapping for `{tensor_name}`: num_k_heads={num_k_heads}, num_v_heads={num_v_heads}"
        )));
    }

    let (rows, cols) = weight.dims2()?;
    let expected_rows = num_v_heads * head_v_dim;
    if rows != expected_rows {
        return Err(Error::ModelLoadError(format!(
            "Unexpected row count for `{tensor_name}`: expected {expected_rows}, got {rows}"
        )));
    }

    let num_v_per_k = num_v_heads / num_k_heads;
    let weight = weight.reshape((num_k_heads, num_v_per_k, head_v_dim, cols))?;
    let weight = weight.permute((1, 0, 2, 3))?;
    weight.reshape((rows, cols)).map_err(Error::from)
}

fn untile_linear_v_vector(
    tensor: Tensor,
    num_k_heads: usize,
    num_v_heads: usize,
    tensor_name: &str,
) -> Result<Tensor> {
    if num_k_heads == num_v_heads {
        return Ok(tensor);
    }
    if num_k_heads == 0 || num_v_heads == 0 || num_v_heads % num_k_heads != 0 {
        return Err(Error::ModelLoadError(format!(
            "Invalid linear head mapping for `{tensor_name}`: num_k_heads={num_k_heads}, num_v_heads={num_v_heads}"
        )));
    }

    let len = tensor.dims1()?;
    if len != num_v_heads {
        return Err(Error::ModelLoadError(format!(
            "Unexpected length for `{tensor_name}`: expected {num_v_heads}, got {len}"
        )));
    }

    let num_v_per_k = num_v_heads / num_k_heads;
    let tensor = tensor.reshape((num_k_heads, num_v_per_k, 1))?;
    let tensor = tensor.permute((1, 0, 2))?;
    tensor.reshape((len,)).map_err(Error::from)
}

fn untile_linear_qkv_weight(
    weight: Tensor,
    cfg: &Qwen35TextConfig,
    tensor_name: &str,
) -> Result<Tensor> {
    if cfg.linear_num_key_heads == cfg.linear_num_value_heads {
        return Ok(weight);
    }

    let (rows, _cols) = weight.dims2()?;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let qk_rows = key_dim * 2;
    let expected_rows = qk_rows + value_dim;
    if rows != expected_rows {
        return Err(Error::ModelLoadError(format!(
            "Unexpected shape for `{tensor_name}`: expected rows={expected_rows}, got rows={rows}"
        )));
    }

    let qk = weight.narrow(0, 0, qk_rows)?;
    let v = weight.narrow(0, qk_rows, value_dim)?;
    let v = untile_linear_v_rows(
        v,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        tensor_name,
    )?;
    Tensor::cat(&[qk, v], 0).map_err(Error::from)
}

fn untile_linear_out_proj_weight(
    weight: Tensor,
    cfg: &Qwen35TextConfig,
    tensor_name: &str,
) -> Result<Tensor> {
    if cfg.linear_num_key_heads == cfg.linear_num_value_heads {
        return Ok(weight);
    }

    let (_out_dim, in_dim) = weight.dims2()?;
    let expected_in = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    if in_dim != expected_in {
        return Err(Error::ModelLoadError(format!(
            "Unexpected input dim for `{tensor_name}`: expected {expected_in}, got {in_dim}"
        )));
    }

    let transposed = weight.transpose(0, 1)?;
    let reordered = untile_linear_v_rows(
        transposed,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        tensor_name,
    )?;
    reordered.transpose(0, 1).map_err(Error::from)
}

fn untile_linear_conv_weight(
    weight: Tensor,
    cfg: &Qwen35TextConfig,
    tensor_name: &str,
) -> Result<Tensor> {
    if cfg.linear_num_key_heads == cfg.linear_num_value_heads {
        return Ok(weight);
    }

    let weight = if let Ok(_dims2) = weight.dims2() {
        weight
    } else if let Ok((_, c1, c2)) = weight.dims3() {
        if c1 == 1 {
            weight.squeeze(1)?
        } else if c2 == 1 {
            weight.squeeze(2)?
        } else {
            return Err(Error::ModelLoadError(format!(
                "Unsupported conv tensor rank/shape for `{tensor_name}`; expected 2D or 3D with singleton axis"
            )));
        }
    } else {
        return Err(Error::ModelLoadError(format!(
            "Unsupported conv tensor rank for `{tensor_name}`; expected 2D or 3D"
        )));
    };

    let (channels, _kernel) = weight.dims2()?;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let qk_channels = key_dim * 2;
    let expected_channels = qk_channels + value_dim;
    if channels != expected_channels {
        return Err(Error::ModelLoadError(format!(
            "Unexpected channel count for `{tensor_name}`: expected {expected_channels}, got {channels}"
        )));
    }

    let qk = weight.narrow(0, 0, qk_channels)?;
    let v = weight.narrow(0, qk_channels, value_dim)?;
    let v = untile_linear_v_rows(
        v,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        tensor_name,
    )?;
    Tensor::cat(&[qk, v], 0).map_err(Error::from)
}

fn select_qwen35_gguf_backend() -> Result<Qwen35GgufBackend> {
    let raw = std::env::var("IZWI_QWEN35_GGUF_BACKEND").unwrap_or_default();
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "quantized" => Ok(Qwen35GgufBackend::Quantized),
        "native" | "dense" => Ok(Qwen35GgufBackend::Native),
        other => Err(Error::InvalidInput(format!(
            "Invalid IZWI_QWEN35_GGUF_BACKEND value `{other}`. Expected `quantized` or `native`."
        ))),
    }
}

fn find_preferred_gguf(model_dir: &Path) -> Result<Option<std::path::PathBuf>> {
    let mut candidates = std::fs::read_dir(model_dir)
        .map_err(Error::from)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|name| {
                    name.ends_with(".gguf")
                        && !name.contains("mmproj")
                        && !name.starts_with("tokenizer")
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return Ok(None);
    }

    candidates.sort();
    if let Some(preferred) = candidates.iter().find(|path| {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|name| name.ends_with("Q4_K_M.gguf"))
            .unwrap_or(false)
    }) {
        return Ok(Some(preferred.clone()));
    }

    if candidates.len() == 1 {
        return Ok(Some(candidates.remove(0)));
    }

    let names = candidates
        .iter()
        .filter_map(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(Error::ModelLoadError(format!(
        "Multiple GGUF files found in {} but no Q4_K_M file could be selected: {names}",
        model_dir.display()
    )))
}

fn find_mmproj_gguf(model_dir: &Path) -> Result<Option<PathBuf>> {
    if let Ok(explicit) = std::env::var("IZWI_QWEN35_MMPROJ_PATH") {
        let explicit = explicit.trim();
        if !explicit.is_empty() {
            let path = PathBuf::from(explicit);
            if path.exists() {
                return Ok(Some(path));
            }
            return Err(Error::ModelLoadError(format!(
                "IZWI_QWEN35_MMPROJ_PATH is set but file was not found: {}",
                path.display()
            )));
        }
    }

    let preferred_names = ["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"];
    for name in preferred_names {
        let candidate = model_dir.join(name);
        if candidate.exists() {
            return Ok(Some(candidate));
        }
    }

    let mut candidates = std::fs::read_dir(model_dir)
        .map_err(Error::from)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|name| name.ends_with(".gguf") && name.contains("mmproj"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    candidates.sort();
    Ok(candidates.into_iter().next())
}

fn find_single_safetensor(model_dir: &Path) -> Result<std::path::PathBuf> {
    let direct = model_dir.join("model.safetensors");
    if direct.exists() {
        return Ok(direct);
    }

    let mut candidates = std::fs::read_dir(model_dir)
        .map_err(Error::from)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| {
                    name.starts_with("model.safetensors-") && name.ends_with(".safetensors")
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    candidates.sort();
    if candidates.len() == 1 {
        Ok(candidates.remove(0))
    } else {
        Err(Error::ModelLoadError(
            "Unable to resolve Qwen3.5 safetensor checkpoint; expected model.safetensors or one model.safetensors-*-of-*.safetensors file"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::chat::{
        qwen35_multimodal_control_content, qwen35_thinking_control_content, Qwen35MultimodalInput,
        Qwen35MultimodalKind,
    };
    use crate::models::shared::device::DeviceSelector;
    use serde_json::json;

    #[test]
    fn qwen35_tools_system_content_includes_schema_and_instructions() {
        let rendered = qwen35_tools_system_content(
            &[json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }
            })],
            Some("Use tools when needed."),
        );

        assert!(rendered.contains("# Tools"));
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("\"get_weather\""));
        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("<IMPORTANT>"));
        assert!(rendered.contains("Use tools when needed."));
    }

    #[test]
    fn detects_vision_placeholders_in_messages() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::User,
                content: format!("look at this {QWEN_VISION_START_TOKEN}{QWEN_IMAGE_PAD_TOKEN}"),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "ok".to_string(),
            },
        ];
        assert!(messages_contain_vision_placeholders(&messages));
    }

    #[test]
    fn qwen35_local_load_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_QWEN35_CHAT_MODEL_DIR") else {
            return;
        };

        let preferred_device = std::env::var("IZWI_QWEN35_SMOKE_DEVICE").ok();
        let device = DeviceSelector::detect_with_preference(preferred_device.as_deref())
            .expect("detect device for qwen3.5 smoke");
        let model = Qwen35ChatModel::load(Path::new(&model_dir), device)
            .expect("load local qwen3.5 chat model");
        assert!(model.supports_incremental_decode());
    }

    #[test]
    fn qwen35_prompt_thinking_toggle_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_QWEN35_CHAT_MODEL_DIR") else {
            return;
        };

        let tokenizer =
            ChatTokenizer::load(Path::new(&model_dir), None).expect("load qwen3.5 tokenizer");

        let user_message = ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        };

        let no_control_prompt =
            build_qwen35_prompt_ids(&tokenizer, std::slice::from_ref(&user_message))
                .expect("build prompt without control");

        let enabled_prompt = build_qwen35_prompt_ids(
            &tokenizer,
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: qwen35_thinking_control_content(true),
                },
                user_message.clone(),
            ],
        )
        .expect("build prompt with thinking enabled");

        let disabled_prompt = build_qwen35_prompt_ids(
            &tokenizer,
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: qwen35_thinking_control_content(false),
                },
                user_message,
            ],
        )
        .expect("build prompt with thinking disabled");

        let mut enabled_suffix = vec![tokenizer.specials.im_start];
        enabled_suffix.extend(
            tokenizer
                .encode_text("assistant\n<think>\n")
                .expect("encode enabled suffix"),
        );

        let mut disabled_suffix = vec![tokenizer.specials.im_start];
        disabled_suffix.extend(
            tokenizer
                .encode_text("assistant\n<think>\n\n</think>\n\n")
                .expect("encode disabled suffix"),
        );

        assert!(enabled_prompt.ids.ends_with(&enabled_suffix));
        assert!(disabled_prompt.ids.ends_with(&disabled_suffix));
        assert!(no_control_prompt.ids.ends_with(&disabled_suffix));
    }

    #[test]
    fn qwen35_prompt_multimodal_control_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_QWEN35_CHAT_MODEL_DIR") else {
            return;
        };
        let tokenizer =
            ChatTokenizer::load(Path::new(&model_dir), None).expect("load qwen3.5 tokenizer");

        let multimodal = vec![Qwen35MultimodalInput {
            kind: Qwen35MultimodalKind::Image,
            source: "https://example.com/cat.png".to_string(),
        }];
        let control =
            qwen35_multimodal_control_content(&multimodal).expect("build multimodal control");
        let prompt = build_qwen35_prompt_ids(
            &tokenizer,
            &[
                ChatMessage {
                    role: ChatRole::System,
                    content: control,
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: format!(
                        "Describe this image {QWEN_VISION_START_TOKEN}{QWEN_IMAGE_PAD_TOKEN}<|vision_end|>"
                    ),
                },
            ],
        )
        .expect("build prompt with multimodal control");

        assert_eq!(prompt.multimodal.len(), 1);
        assert_eq!(prompt.multimodal[0].kind, Qwen35MultimodalKind::Image);
        assert_eq!(prompt.multimodal[0].source, "https://example.com/cat.png");
        let image_pad_count = prompt
            .ids
            .iter()
            .filter(|&&id| id == tokenizer.specials.image_pad)
            .count();
        assert_eq!(image_pad_count, 1);
    }

    #[test]
    fn strip_think_blocks_handles_implicit_qwen35_open_tag() {
        let input = "reasoning line one\nreasoning line two</think>\n\nFinal answer.";
        let stripped = strip_think_blocks(input);
        assert_eq!(stripped, "Final answer.");
    }

    #[test]
    fn detects_token_repetition_loop() {
        let mut ids = Vec::new();
        let phrase = vec![1, 2, 3, 4, 5, 6, 7, 8];
        for _ in 0..5 {
            ids.extend(phrase.iter().copied());
        }
        // Pad a little prefix so the minimum-length guard is satisfied.
        ids.splice(0..0, vec![42; 16]);
        assert!(has_token_repetition_loop(&ids));
    }

    #[test]
    fn does_not_flag_short_sequences_as_loop() {
        let ids: Vec<u32> = (1..30).collect();
        assert!(!has_token_repetition_loop(&ids));
    }
}

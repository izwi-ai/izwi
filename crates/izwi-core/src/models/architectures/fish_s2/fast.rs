//! Fast codebook decoder for Fish S2 DualAR generation.

use candle_core::{IndexOp, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::contracts::semantic_code_from_token_id;
use crate::models::architectures::qwen3::core::{build_rope_cache, causal_mask, repeat_kv};

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2FastConfig {
    pub input_hidden_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2GeneratedFrame {
    pub semantic_token_id: u32,
    pub codebooks: Vec<u32>,
}

#[derive(Debug)]
pub struct FishS2Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: StdRng,
}

#[derive(Debug, Default)]
pub struct FishS2FastCache {
    layers: Vec<FishS2FastLayerCache>,
}

#[derive(Debug, Default)]
struct FishS2FastLayerCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

pub struct FishS2FastDecoder {
    cfg: FishS2FastConfig,
    project_in: FishS2FastProjectIn,
    embeddings: Embedding,
    layers: Vec<FishS2FastLayer>,
    norm: RmsNorm,
    output: Linear,
}

enum FishS2FastProjectIn {
    Identity,
    Linear(Linear),
}

struct FishS2FastLayer {
    input_layernorm: RmsNorm,
    self_attn: FishS2FastAttention,
    post_attention_layernorm: RmsNorm,
    mlp: FishS2FastMlp,
}

struct FishS2FastAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

struct FishS2FastMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FishS2FastConfig {
    pub fn from_config(config: &FishS2Config) -> Result<Self> {
        let audio = &config.audio_decoder_config;
        let text = &config.text_config;
        let head_dim = audio
            .head_dim
            .unwrap_or_else(|| audio.hidden_size / audio.num_attention_heads);
        let intermediate_size = audio
            .intermediate_size
            .unwrap_or_else(|| audio.hidden_size * 3);
        Ok(Self {
            input_hidden_size: text.hidden_size,
            hidden_size: audio.hidden_size,
            intermediate_size,
            num_hidden_layers: audio.num_hidden_layers,
            num_attention_heads: audio.num_attention_heads,
            num_key_value_heads: audio.num_key_value_heads,
            head_dim,
            codebook_size: config.codebook_size,
            num_codebooks: config.num_codebooks,
            semantic_start_token_id: config.semantic_start_token_id,
            semantic_end_token_id: config.semantic_end_token_id,
            rope_theta: text.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: text.rms_norm_eps.unwrap_or(1e-6),
        })
    }

    fn q_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

impl FishS2Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature: temperature.max(0.0),
            top_p: top_p.clamp(0.0, 1.0),
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl FishS2FastCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| FishS2FastLayerCache::default())
                .collect(),
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers
            .first()
            .and_then(|layer| layer.key.as_ref())
            .and_then(|key| key.dims().get(2).copied())
            .unwrap_or(0)
    }
}

impl FishS2FastDecoder {
    pub fn load(cfg: FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        let project_in = if cfg.input_hidden_size == cfg.hidden_size {
            FishS2FastProjectIn::Identity
        } else {
            let prefix = if vb.contains_tensor("fast_project_in.weight") {
                "fast_project_in"
            } else if vb.contains_tensor("project_in.weight") {
                "project_in"
            } else {
                "fast_project_in"
            };
            FishS2FastProjectIn::Linear(candle_nn::linear(
                cfg.input_hidden_size,
                cfg.hidden_size,
                vb.pp(prefix),
            )?)
        };
        let embeddings =
            candle_nn::embedding(cfg.codebook_size, cfg.hidden_size, vb.pp("fast_embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(FishS2FastLayer::load(
                &cfg,
                vb.pp(format!("fast_layers.{idx}")),
            )?);
        }
        let norm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["fast_norm", "norm"],
        )?;
        let output =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.codebook_size, vb.pp("fast_output"))?;
        Ok(Self {
            cfg,
            project_in,
            embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn config(&self) -> &FishS2FastConfig {
        &self.cfg
    }

    pub fn new_cache(&self) -> FishS2FastCache {
        FishS2FastCache::new(self.layers.len())
    }

    pub fn project_slow_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.project_in {
            FishS2FastProjectIn::Identity => Ok(hidden.clone()),
            FishS2FastProjectIn::Linear(linear) => linear.forward(hidden).map_err(Error::from),
        }
    }

    pub fn codebook_embedding(&self, code: u32) -> Result<Tensor> {
        if code as usize >= self.cfg.codebook_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast code {code} exceeds codebook size {}",
                self.cfg.codebook_size
            )));
        }
        let ids = Tensor::from_vec(vec![code], (1, 1), self.embeddings.embeddings().device())?;
        self.embeddings.forward(&ids).map_err(Error::from)
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut FishS2FastCache,
    ) -> Result<Tensor> {
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, input_pos, &mut cache.layers[idx])?;
        }
        let out = self.norm.forward(&hidden)?;
        self.output.forward(&out).map_err(Error::from)
    }

    pub fn generate_frame(
        &self,
        semantic_token_id: u32,
        slow_hidden: &Tensor,
        sampler: &mut FishS2Sampler,
    ) -> Result<FishS2GeneratedFrame> {
        let semantic_code =
            semantic_code_from_token_id_from_fast_config(&self.cfg, semantic_token_id)?;
        let mut cache = self.new_cache();
        let hidden = self.project_slow_hidden(slow_hidden)?;
        let _ = self.forward_step(&hidden, 0, &mut cache)?;

        let mut codebooks = vec![semantic_code];
        let mut current = self.codebook_embedding(semantic_code)?;
        for codebook_idx in 1..self.cfg.num_codebooks {
            let logits = self.forward_step(&current, codebook_idx, &mut cache)?;
            let row = logits.i((0, 0))?;
            let code = sample_logits(&row, sampler)?;
            codebooks.push(code);
            current = self.codebook_embedding(code)?;
        }
        Ok(FishS2GeneratedFrame {
            semantic_token_id,
            codebooks,
        })
    }
}

impl FishS2FastLayer {
    fn load(cfg: &FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = FishS2FastAttention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["post_attention_layernorm", "ffn_norm"],
        )?;
        let mlp = FishS2FastMlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut FishS2FastLayerCache,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed, input_pos, cache)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl FishS2FastAttention {
    fn load(cfg: &FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        Ok(Self {
            qkv_proj: candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("qkv_proj"))?,
            o_proj: candle_nn::linear_no_bias(cfg.q_size(), cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut FishS2FastLayerCache,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv
            .narrow(2, 0, q_size)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = qkv.narrow(2, q_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            input_pos,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;
        let q = apply_rope(&q, &cos, &sin)?.transpose(1, 2)?;
        let mut k = apply_rope(&k, &cos, &sin)?.transpose(1, 2)?;
        let mut v = v.transpose(1, 2)?;

        if let (Some(prev_k), Some(prev_v)) = (&cache.key, &cache.value) {
            k = Tensor::cat(&[prev_k, &k], 2)?;
            v = Tensor::cat(&[prev_v, &v], 2)?;
        }
        cache.key = Some(k.clone());
        cache.value = Some(v.clone());

        let total_len = k.dim(2)?;
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;
        let mask = causal_mask(seq_len, total_len, input_pos, att.device(), att.dtype())?;
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

impl FishS2FastMlp {
    fn load(cfg: &FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
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
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

fn sample_logits(row: &Tensor, sampler: &mut FishS2Sampler) -> Result<u32> {
    let values = row.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    if values.is_empty() {
        return Err(Error::InferenceError(
            "Fish S2 fast sampler received empty logits".to_string(),
        ));
    }
    if sampler.temperature <= 1e-5 || sampler.top_p <= 0.0 {
        return argmax_values(&values);
    }

    let temp = sampler.temperature.max(1e-5);
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value / temp));
    let mut probs = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, ((*value / temp) - max).exp()))
        .collect::<Vec<_>>();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_values(&values);
    }
    for (_, prob) in &mut probs {
        *prob /= sum;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0f32;
    let mut kept = Vec::new();
    for item in probs {
        cumulative += item.1;
        kept.push(item);
        if cumulative >= sampler.top_p.max(1e-6) {
            break;
        }
    }
    let kept_sum: f32 = kept.iter().map(|(_, p)| *p).sum();
    let mut draw = sampler.rng.r#gen::<f32>() * kept_sum;
    for (idx, prob) in kept {
        if draw <= prob {
            return u32::try_from(idx).map_err(|_| {
                Error::InferenceError("Fish S2 sampled index overflowed".to_string())
            });
        }
        draw -= prob;
    }
    argmax_values(&values)
}

fn argmax_values(values: &[f32]) -> Result<u32> {
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    u32::try_from(best_idx)
        .map_err(|_| Error::InferenceError("Fish S2 argmax index overflowed".to_string()))
}

fn semantic_code_from_token_id_from_fast_config(
    cfg: &FishS2FastConfig,
    token_id: u32,
) -> Result<u32> {
    let config = FishS2Config {
        architectures: vec!["DualARTransformer".to_string()],
        model_type: "fish_qwen3_omni".to_string(),
        torch_dtype: None,
        text_config: crate::models::architectures::fish_s2::config::FishS2TextConfig {
            hidden_size: cfg.input_hidden_size,
            num_hidden_layers: 1,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: Some(cfg.head_dim),
            vocab_size: cfg.semantic_end_token_id as usize + 1,
            max_seq_len: 16,
            rope_theta: Some(cfg.rope_theta),
            rms_norm_eps: Some(cfg.rms_norm_eps),
            intermediate_size: Some(cfg.intermediate_size),
            hidden_act: None,
        },
        audio_decoder_config:
            crate::models::architectures::fish_s2::config::FishS2AudioDecoderConfig {
                hidden_size: cfg.hidden_size,
                num_hidden_layers: cfg.num_hidden_layers,
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim: Some(cfg.head_dim),
                intermediate_size: Some(cfg.intermediate_size),
                max_seq_len: Some(cfg.num_codebooks + 1),
            },
        num_codebooks: cfg.num_codebooks,
        codebook_size: cfg.codebook_size,
        max_seq_len: 16,
        bos_token_id: 0,
        eos_token_id: 0,
        pad_token_id: 0,
        audio_pad_token_id: 0,
        semantic_start_token_id: cfg.semantic_start_token_id,
        semantic_end_token_id: cfg.semantic_end_token_id,
        sample_rate: None,
    };
    semantic_code_from_token_id(&config, token_id)
}

fn apply_rope(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let out_first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out_second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_first, &out_second], 3).map_err(Error::from)
}

fn load_rms_norm_alias(dim: usize, eps: f64, vb: &VarBuilder, aliases: &[&str]) -> Result<RmsNorm> {
    for alias in aliases {
        if vb.contains_tensor(&format!("{alias}.weight")) {
            return candle_nn::rms_norm(dim, eps, vb.pp(*alias)).map_err(Error::from);
        }
    }
    candle_nn::rms_norm(dim, eps, vb.pp(aliases[0])).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};
    use std::collections::HashMap;

    fn tiny_cfg() -> FishS2FastConfig {
        FishS2FastConfig {
            input_hidden_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            codebook_size: 8,
            num_codebooks: 3,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    fn tiny_decoder(device: &Device) -> FishS2FastDecoder {
        let cfg = tiny_cfg();
        let mut tensors = HashMap::new();
        tensors.insert(
            "fast_project_in.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.input_hidden_size), 0.01),
        );
        tensors.insert(
            "fast_project_in.bias".to_string(),
            tensor(device, (cfg.hidden_size,), 0.0),
        );
        tensors.insert(
            "fast_embeddings.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.02),
        );
        tensors.insert(
            "fast_norm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_output.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.03),
        );
        tensors.insert(
            "fast_layers.0.input_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.post_attention_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.self_attn.qkv_proj.weight".to_string(),
            tensor(
                device,
                (cfg.q_size() + 2 * cfg.kv_size(), cfg.hidden_size),
                0.01,
            ),
        );
        tensors.insert(
            "fast_layers.0.self_attn.o_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.q_size()), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.gate_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.up_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.down_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.intermediate_size), 0.01),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        FishS2FastDecoder::load(cfg, vb).unwrap()
    }

    #[test]
    fn sampler_argmax_is_deterministic_at_zero_temperature() {
        let device = Device::Cpu;
        let row = Tensor::from_vec(vec![0.1f32, 2.0, 1.0], (3,), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 7);
        assert_eq!(sample_logits(&row, &mut sampler).unwrap(), 1);
    }

    #[test]
    fn fast_decoder_generates_full_codebook_frame() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let slow_hidden = Tensor::full(0.5f32, (1, 1, 3), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 11);
        let frame = decoder
            .generate_frame(22, &slow_hidden, &mut sampler)
            .expect("frame");
        assert_eq!(frame.semantic_token_id, 22);
        assert_eq!(frame.codebooks.len(), 3);
        assert_eq!(frame.codebooks[0], 2);
        assert!(frame.codebooks[1] < 8);
        assert!(frame.codebooks[2] < 8);
    }
}

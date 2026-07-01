//! Slow semantic transformer for Fish S2 DualAR generation.

use candle_core::{Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::contracts::build_semantic_allowed_mask;
use crate::models::architectures::fish_s2::tokenizer::FishS2ConditioningPrompt;
use crate::models::architectures::qwen3::core::{build_rope_cache, causal_mask, repeat_kv};

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2SlowConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub scale_codebook_embeddings: bool,
}

#[derive(Debug, Clone)]
pub struct FishS2SlowOutput {
    pub logits: Tensor,
    pub hidden_states: Tensor,
}

#[derive(Debug, Default)]
pub struct FishS2SlowCache {
    layers: Vec<FishS2LayerCache>,
}

#[derive(Debug, Default)]
struct FishS2LayerCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

pub struct FishS2SlowTransformer {
    cfg: FishS2SlowConfig,
    embeddings: Embedding,
    codebook_embeddings: Embedding,
    layers: Vec<FishS2SlowLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

struct FishS2SlowLayer {
    input_layernorm: RmsNorm,
    self_attn: FishS2PackedAttention,
    post_attention_layernorm: RmsNorm,
    mlp: FishS2Mlp,
}

struct FishS2PackedAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

struct FishS2Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FishS2SlowConfig {
    pub fn from_config(config: &FishS2Config) -> Result<Self> {
        let text = &config.text_config;
        let head_dim = text
            .head_dim
            .unwrap_or_else(|| text.hidden_size / text.num_attention_heads);
        let intermediate_size = text.intermediate_size.ok_or_else(|| {
            Error::ConfigError("Fish S2 text_config missing intermediate_size".to_string())
        })?;
        Ok(Self {
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim,
            max_seq_len: text.max_seq_len,
            rope_theta: text.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: text.rms_norm_eps.unwrap_or(1e-6),
            codebook_size: config.codebook_size,
            num_codebooks: config.num_codebooks,
            semantic_start_token_id: config.semantic_start_token_id,
            semantic_end_token_id: config.semantic_end_token_id,
            scale_codebook_embeddings: true,
        })
    }

    fn semantic_contains(&self, token_id: u32) -> bool {
        token_id >= self.semantic_start_token_id && token_id <= self.semantic_end_token_id
    }

    fn q_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

impl FishS2SlowCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| FishS2LayerCache::default())
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

impl FishS2SlowTransformer {
    pub fn load(cfg: FishS2SlowConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let codebook_embeddings = candle_nn::embedding(
            cfg.codebook_size * cfg.num_codebooks,
            cfg.hidden_size,
            vb.pp("codebook_embeddings"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(FishS2SlowLayer::load(&cfg, vb.pp(format!("layers.{idx}")))?);
        }
        let norm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["norm", "model.norm"],
        )?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(embeddings.embeddings().clone(), None)
        };
        Ok(Self {
            cfg,
            embeddings,
            codebook_embeddings,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &FishS2SlowConfig {
        &self.cfg
    }

    pub fn new_cache(&self) -> FishS2SlowCache {
        FishS2SlowCache::new(self.layers.len())
    }

    pub fn embed_prompt(&self, prompt: &FishS2ConditioningPrompt) -> Result<Tensor> {
        if prompt.values.len() != self.cfg.num_codebooks + 1 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 prompt has {} rows, expected {}",
                prompt.values.len(),
                self.cfg.num_codebooks + 1
            )));
        }
        if prompt.prompt_length == 0 || prompt.values[0].len() != prompt.prompt_length {
            return Err(Error::InvalidInput(
                "Fish S2 prompt has invalid prompt_length".to_string(),
            ));
        }
        for row in &prompt.values {
            if row.len() != prompt.prompt_length {
                return Err(Error::InvalidInput(
                    "Fish S2 prompt rows must all have the same length".to_string(),
                ));
            }
        }

        let device = self.embeddings.embeddings().device();
        let row0 = Tensor::from_vec(prompt.values[0].clone(), (1, prompt.prompt_length), device)?;
        let mut x = self.embeddings.forward(&row0)?;

        let mut vq_sum: Option<Tensor> = None;
        for codebook_idx in 0..self.cfg.num_codebooks {
            let offset = u32::try_from(
                codebook_idx
                    .checked_mul(self.cfg.codebook_size)
                    .ok_or_else(|| {
                        Error::ConfigError(
                            "Fish S2 codebook embedding offset overflowed".to_string(),
                        )
                    })?,
            )
            .map_err(|_| {
                Error::ConfigError("Fish S2 codebook embedding offset exceeds u32".to_string())
            })?;
            let ids = prompt.values[codebook_idx + 1]
                .iter()
                .map(|code| {
                    code.checked_add(offset).ok_or_else(|| {
                        Error::ConfigError("Fish S2 codebook id overflowed".to_string())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let ids = Tensor::from_vec(ids, (1, prompt.prompt_length), device)?;
            let emb = self.codebook_embeddings.forward(&ids)?;
            vq_sum = Some(match vq_sum {
                Some(sum) => sum.broadcast_add(&emb)?,
                None => emb,
            });
        }

        if let Some(vq_sum) = vq_sum {
            let mask = prompt.values[0]
                .iter()
                .map(|token| {
                    if self.cfg.semantic_contains(*token) {
                        1.0f32
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let mask = Tensor::from_vec(mask, (1, prompt.prompt_length, 1), device)?
                .to_dtype(x.dtype())?;
            x = x.broadcast_add(&vq_sum.broadcast_mul(&mask)?)?;

            if self.cfg.scale_codebook_embeddings {
                let scale = 1.0f32 / ((self.cfg.num_codebooks + 1) as f32).sqrt();
                let scales = prompt.values[0]
                    .iter()
                    .map(|token| {
                        if self.cfg.semantic_contains(*token) {
                            scale
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<_>>();
                let scales = Tensor::from_vec(scales, (1, prompt.prompt_length, 1), device)?
                    .to_dtype(x.dtype())?;
                x = x.broadcast_mul(&scales)?;
            }
        }

        Ok(x)
    }

    pub fn forward_embeds(
        &self,
        x: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut FishS2SlowCache>,
        return_all: bool,
    ) -> Result<FishS2SlowOutput> {
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_deref_mut().map(|cache| &mut cache.layers[idx]);
            hidden = layer.forward(&hidden, start_pos, layer_cache)?;
        }

        let hidden_for_fast = self.norm.forward(&hidden)?;
        let logits_input = if return_all {
            hidden_for_fast.clone()
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        let logits = self.lm_head.forward(&logits_input)?;
        let hidden_states = if return_all {
            hidden_for_fast
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        Ok(FishS2SlowOutput {
            logits,
            hidden_states,
        })
    }

    pub fn forward_prompt(
        &self,
        prompt: &FishS2ConditioningPrompt,
        cache: Option<&mut FishS2SlowCache>,
        return_all: bool,
    ) -> Result<FishS2SlowOutput> {
        let x = self.embed_prompt(prompt)?;
        self.forward_embeds(&x, 0, cache, return_all)
    }

    pub fn semantic_allowed_mask(&self, im_end_token_id: u32) -> Result<Vec<bool>> {
        let config = FishS2Config {
            architectures: vec!["DualARTransformer".to_string()],
            model_type: "fish_qwen3_omni".to_string(),
            torch_dtype: None,
            text_config: crate::models::architectures::fish_s2::config::FishS2TextConfig {
                hidden_size: self.cfg.hidden_size,
                num_hidden_layers: self.cfg.num_hidden_layers,
                num_attention_heads: self.cfg.num_attention_heads,
                num_key_value_heads: self.cfg.num_key_value_heads,
                head_dim: Some(self.cfg.head_dim),
                vocab_size: self.cfg.vocab_size,
                max_seq_len: self.cfg.max_seq_len,
                rope_theta: Some(self.cfg.rope_theta),
                rms_norm_eps: Some(self.cfg.rms_norm_eps),
                intermediate_size: Some(self.cfg.intermediate_size),
                hidden_act: None,
            },
            audio_decoder_config:
                crate::models::architectures::fish_s2::config::FishS2AudioDecoderConfig {
                    hidden_size: self.cfg.hidden_size,
                    num_hidden_layers: 1,
                    num_attention_heads: self.cfg.num_attention_heads,
                    num_key_value_heads: self.cfg.num_key_value_heads,
                    head_dim: Some(self.cfg.head_dim),
                    intermediate_size: Some(self.cfg.intermediate_size),
                    max_seq_len: Some(self.cfg.num_codebooks + 1),
                    num_codebooks: Some(self.cfg.num_codebooks),
                    vocab_size: Some(self.cfg.codebook_size),
                },
            num_codebooks: self.cfg.num_codebooks,
            codebook_size: self.cfg.codebook_size,
            max_seq_len: self.cfg.max_seq_len,
            bos_token_id: 0,
            eos_token_id: im_end_token_id,
            pad_token_id: 0,
            audio_pad_token_id: 0,
            semantic_start_token_id: self.cfg.semantic_start_token_id,
            semantic_end_token_id: self.cfg.semantic_end_token_id,
            sample_rate: None,
        };
        build_semantic_allowed_mask(self.cfg.vocab_size, &config, im_end_token_id)
    }
}

impl FishS2SlowLayer {
    fn load(cfg: &FishS2SlowConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = FishS2PackedAttention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["post_attention_layernorm", "ffn_norm"],
        )?;
        let mlp = FishS2Mlp::load(cfg, vb.pp("mlp"))?;
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
        start_pos: usize,
        cache: Option<&mut FishS2LayerCache>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed, start_pos, cache)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl FishS2PackedAttention {
    fn load(cfg: &FishS2SlowConfig, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        Ok(Self {
            qkv_proj: candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("qkv_proj"))?,
            o_proj: candle_nn::linear_no_bias(cfg.q_size(), cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: if vb.contains_tensor("q_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("q_norm"),
                )?)
            } else {
                None
            },
            k_norm: if vb.contains_tensor("k_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("k_norm"),
                )?)
            } else {
                None
            },
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut FishS2LayerCache>,
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
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            start_pos,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;
        let q = apply_rope(&q, &cos, &sin)?;
        let k = apply_rope(&k, &cos, &sin)?;
        let q = q.transpose(1, 2)?;
        let mut k = k.transpose(1, 2)?;
        let mut v = v.transpose(1, 2)?;

        if let Some(cache) = cache {
            if let (Some(prev_k), Some(prev_v)) = (&cache.key, &cache.value) {
                k = Tensor::cat(&[prev_k, &k], 2)?;
                v = Tensor::cat(&[prev_v, &v], 2)?;
            }
            cache.key = Some(k.clone());
            cache.value = Some(v.clone());
        }

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
        let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
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

impl FishS2Mlp {
    fn load(cfg: &FishS2SlowConfig, vb: VarBuilder) -> Result<Self> {
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

    fn tiny_cfg() -> FishS2SlowConfig {
        FishS2SlowConfig {
            vocab_size: 32,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            codebook_size: 8,
            num_codebooks: 2,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            scale_codebook_embeddings: true,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    fn tiny_model(device: &Device) -> FishS2SlowTransformer {
        let cfg = tiny_cfg();
        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            tensor(device, (cfg.vocab_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "codebook_embeddings.weight".to_string(),
            tensor(
                device,
                (cfg.codebook_size * cfg.num_codebooks, cfg.hidden_size),
                0.02,
            ),
        );
        tensors.insert(
            "norm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.input_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.post_attention_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "layers.0.self_attn.qkv_proj.weight".to_string(),
            tensor(
                device,
                (cfg.q_size() + 2 * cfg.kv_size(), cfg.hidden_size),
                0.01,
            ),
        );
        tensors.insert(
            "layers.0.self_attn.q_norm.weight".to_string(),
            tensor(device, (cfg.head_dim,), 1.0),
        );
        tensors.insert(
            "layers.0.self_attn.k_norm.weight".to_string(),
            tensor(device, (cfg.head_dim,), 1.0),
        );
        tensors.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.q_size()), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.gate_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.up_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "layers.0.mlp.down_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.intermediate_size), 0.01),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        FishS2SlowTransformer::load(cfg, vb).unwrap()
    }

    #[test]
    fn embeds_prompt_with_codebooks_only_on_semantic_tokens() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        let prompt = FishS2ConditioningPrompt {
            values: vec![vec![1, 20, 21], vec![0, 3, 4], vec![0, 5, 6]],
            vq_mask: vec![false, true, true],
            prompt_length: 3,
        };
        let embeds = model.embed_prompt(&prompt).unwrap();
        assert_eq!(embeds.dims(), &[1, 3, 4]);
        let values = embeds.to_vec3::<f32>().unwrap();
        assert!(values[0][1][0] > values[0][0][0]);
    }

    #[test]
    fn slow_transformer_forward_returns_logits_and_hidden_tail() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        let prompt = FishS2ConditioningPrompt {
            values: vec![vec![1, 20, 21], vec![0, 3, 4], vec![0, 5, 6]],
            vq_mask: vec![false, true, true],
            prompt_length: 3,
        };
        let mut cache = model.new_cache();
        let output = model
            .forward_prompt(&prompt, Some(&mut cache), false)
            .unwrap();
        assert_eq!(output.logits.dims(), &[1, 1, 32]);
        assert_eq!(output.hidden_states.dims(), &[1, 1, 4]);
        assert_eq!(cache.current_len(), 3);
    }
}

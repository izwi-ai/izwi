//! Slow semantic transformer for Fish S2 DualAR generation.

use candle_core::{DType, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::rotary::FishS2RotaryCache;
use crate::models::architectures::fish_s2::tokenizer::FishS2ConditioningPrompt;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

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

pub struct FishS2SlowTransformer {
    cfg: FishS2SlowConfig,
    embeddings: Embedding,
    codebook_embeddings: Embedding,
    layers: Vec<FishS2SlowLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    semantic_head: Option<(Linear, u32)>,
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
    rotary: FishS2RotaryCache,
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

impl FishS2SlowTransformer {
    pub fn load(cfg: FishS2SlowConfig, vb: VarBuilder) -> Result<Self> {
        let rotary = FishS2RotaryCache::new(
            cfg.max_seq_len,
            cfg.head_dim,
            cfg.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        let embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let codebook_embeddings = candle_nn::embedding(
            cfg.codebook_size * cfg.num_codebooks,
            cfg.hidden_size,
            vb.pp("codebook_embeddings"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(FishS2SlowLayer::load(
                &cfg,
                &rotary,
                vb.pp(format!("layers.{idx}")),
            )?);
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
            semantic_head: None,
        })
    }

    /// Gather the only output rows Fish samples: EOS followed by the semantic
    /// vocabulary. Input embeddings remain the complete checkpoint matrix.
    pub(crate) fn configure_semantic_head(&mut self, eos: u32) -> Result<()> {
        if eos as usize >= self.cfg.vocab_size
            || (self.cfg.semantic_start_token_id..=self.cfg.semantic_end_token_id).contains(&eos)
        {
            return Err(Error::ModelLoadError(
                "Fish S2 EOS must be outside its semantic vocabulary".into(),
            ));
        }
        let weight = self.lm_head.weight();
        let eos_weight = weight.narrow(0, eos as usize, 1)?;
        let semantic_weight = weight.narrow(
            0,
            self.cfg.semantic_start_token_id as usize,
            (self.cfg.semantic_end_token_id - self.cfg.semantic_start_token_id + 1) as usize,
        )?;
        self.semantic_head = Some((
            Linear::new(Tensor::cat(&[&eos_weight, &semantic_weight], 0)?, None),
            eos,
        ));
        Ok(())
    }

    pub(crate) fn eos_logit_index(&self, eos: u32) -> u32 {
        if self.semantic_head.is_some() {
            0
        } else {
            eos
        }
    }

    pub(crate) fn token_id_from_logit(&self, index: u32) -> Result<u32> {
        if let Some((_, eos)) = &self.semantic_head {
            if index == 0 {
                return Ok(*eos);
            }
            let token = self
                .cfg
                .semantic_start_token_id
                .checked_add(index - 1)
                .filter(|id| *id <= self.cfg.semantic_end_token_id)
                .ok_or_else(|| {
                    Error::InferenceError("Fish S2 compact head index is out of range".into())
                })?;
            Ok(token)
        } else if (index as usize) < self.cfg.vocab_size {
            Ok(index)
        } else {
            Err(Error::InferenceError(
                "Fish S2 head index is out of range".into(),
            ))
        }
    }

    fn project_logits(&self, input: &Tensor) -> Result<Tensor> {
        self.semantic_head
            .as_ref()
            .map(|(head, _)| head)
            .unwrap_or(&self.lm_head)
            .forward(input)
            .map_err(Error::from)
    }

    pub fn config(&self) -> &FishS2SlowConfig {
        &self.cfg
    }

    /// Persistent RoPE table bytes, shared by every slow layer.
    pub fn rotary_cache_bytes(&self) -> u64 {
        self.layers
            .first()
            .map(|layer| layer.self_attn.rotary.storage_bytes())
            .unwrap_or(0)
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
        cache: &mut PhysicalPagedKvCache,
        return_all: bool,
    ) -> Result<FishS2SlowOutput> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 slow physical paging expects [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        let end_pos = start_pos.checked_add(sequence_len).ok_or_else(|| {
            Error::InvalidInput("Fish S2 slow physical context length overflow".into())
        })?;
        if end_pos > self.cfg.max_seq_len || end_pos > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Fish S2 slow physical append ends at {end_pos}, beyond model/cache capacity {}/{}",
                self.cfg.max_seq_len,
                cache.capacity_tokens()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        )?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, start_pos, cache, &mut prepared, idx)?;
        }

        let hidden_for_fast = self.norm.forward(&hidden)?;
        let logits_input = if return_all {
            hidden_for_fast.clone()
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        let logits = self.project_logits(&logits_input)?;
        let hidden_states = if return_all {
            hidden_for_fast
        } else {
            let seq_len = hidden_for_fast.dim(1)?;
            hidden_for_fast.narrow(1, seq_len - 1, 1)?
        };
        let output = FishS2SlowOutput {
            logits,
            hidden_states,
        };
        cache.commit_prepared(prepared)?;
        Ok(output)
    }

    pub fn forward_prompt(
        &self,
        prompt: &FishS2ConditioningPrompt,
        cache: &mut PhysicalPagedKvCache,
        return_all: bool,
    ) -> Result<FishS2SlowOutput> {
        let x = self.embed_prompt(prompt)?;
        self.forward_embeds(&x, 0, cache, return_all)
    }

    pub fn semantic_allowed_mask(&self, im_end_token_id: u32) -> Result<Vec<bool>> {
        if let Some((head, eos)) = &self.semantic_head {
            if *eos != im_end_token_id {
                return Err(Error::ModelLoadError(
                    "Fish S2 compact EOS changed after load".into(),
                ));
            }
            return Ok(vec![true; head.weight().dim(0)?]);
        }
        if self.cfg.semantic_end_token_id as usize >= self.cfg.vocab_size
            || im_end_token_id as usize >= self.cfg.vocab_size
        {
            return Err(Error::ModelLoadError(
                "Fish S2 semantic tokens exceed vocabulary".into(),
            ));
        }
        let mut mask = vec![false; self.cfg.vocab_size];
        mask[self.cfg.semantic_start_token_id as usize..=self.cfg.semantic_end_token_id as usize]
            .fill(true);
        mask[im_end_token_id as usize] = true;
        Ok(mask)
    }
}

impl FishS2SlowLayer {
    fn load(cfg: &FishS2SlowConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = FishS2PackedAttention::load(cfg, rotary, vb.pp("self_attn"))?;
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
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self
            .self_attn
            .forward(&normed, start_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl FishS2PackedAttention {
    fn load(cfg: &FishS2SlowConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        if !vb.contains_tensor("q_norm.weight") || !vb.contains_tensor("k_norm.weight") {
            return Err(Error::ModelLoadError(
                "Fish S2 slow attention requires Q/K normalization weights".into(),
            ));
        }
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
            rotary: rotary.clone(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Fish S2 slow physical paged attention expects one sequence".into(),
            ));
        }
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

        let q = self.rotary.apply(&q, start_pos)?;
        let k = self.rotary.apply(&k, start_pos)?;
        let q = q.squeeze(0)?;
        let k = k.squeeze(0)?;
        let v = v.squeeze(0)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
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

    fn check_compact_head(device: &Device) {
        let mut model = tiny_model(device);
        let cfg = model.cfg.clone();
        let weights: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 0.3) * 0.17).sin())
            .collect();
        model.lm_head = Linear::new(
            Tensor::from_vec(weights, (cfg.vocab_size, cfg.hidden_size), device).unwrap(),
            None,
        );
        let input = Tensor::from_vec(
            vec![0.2f32, -0.7, 1.3, 0.4, -1.2, 0.3, 0.7, 1.1],
            (1, 2, 4),
            device,
        )
        .unwrap();
        let full = model.project_logits(&input).unwrap();
        let expected = Tensor::cat(
            &[
                &full.narrow(2, 1, 1).unwrap(),
                &full.narrow(2, 20, 8).unwrap(),
            ],
            2,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        model.configure_semantic_head(1).unwrap();
        let compact = model.project_logits(&input).unwrap();
        assert_eq!(compact.dims(), &[1, 2, 9]);
        let actual = compact.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        assert_eq!(model.token_id_from_logit(0).unwrap(), 1);
        assert_eq!(model.token_id_from_logit(1).unwrap(), 20);
        assert_eq!(model.token_id_from_logit(8).unwrap(), 27);
        assert!(model.token_id_from_logit(9).is_err());
        assert_eq!(model.semantic_allowed_mask(1).unwrap(), vec![true; 9]);
        assert_eq!(
            model.embeddings.embeddings().dims(),
            &[cfg.vocab_size, cfg.hidden_size]
        );
    }

    #[test]
    fn compact_head_matches_gathered_full_logits_and_preserves_token_ids() {
        check_compact_head(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an available Metal device; never falls back to CPU"]
    fn metal_compact_head_matches_full_projection() {
        check_compact_head(&crate::backends::metal_device_if_available(0).expect("Metal device"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an available CUDA device; never falls back to CPU"]
    fn cuda_compact_head_matches_full_projection() {
        check_compact_head(&Device::new_cuda(0).expect("CUDA device"));
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
        let cfg = model.config();
        let mut cache = super::super::physical::test_physical_cache(
            201,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.max_seq_len,
        );
        let output = model.forward_prompt(&prompt, &mut cache, false).unwrap();
        assert_eq!(output.logits.dims(), &[1, 1, 32]);
        assert_eq!(output.hidden_states.dims(), &[1, 1, 4]);
        assert_eq!(cache.context_len(), 3);
    }
}

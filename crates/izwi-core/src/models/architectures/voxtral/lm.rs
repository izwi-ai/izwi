//! Voxtral Language Model - Mistral-style architecture variant of Qwen3.
//!
//! This module provides the Voxtral-specific language model loading and inference,
//! which uses different tensor naming conventions (wq/wk/wv/wo, w1/w2/w3) and
//! root-level layer structure compared to standard Qwen3.

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::kv::v2::{
    InferenceStateContract, PositionSemantics, PrefixPolicy, StateClock, StateDomainId,
    StateDomainSpec, StateGroupId, StateGroupSpec, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::{
    build_mrope_cache, build_rope_cache, qwen3_decoder_cache_domain, Qwen3Config,
    Qwen3DecoderCacheGeometry,
};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

use super::layers::linear_forward_last_dim;

const EMBEDDING_WEIGHT_CANDIDATES: &[&str] = &[
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "mm_audio_embeddings.tok_embeddings.weight",
    "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
];

const LM_HEAD_WEIGHT_CANDIDATES: &[&str] = &[
    "lm_head.weight",
    "output.weight",
    "model.lm_head.weight",
    "model.output.weight",
    "language_model.lm_head.weight",
    "language_model.output.weight",
];

pub struct VoxtralLM {
    embed_tokens: Embedding,
    layers: Vec<VoxtralLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    cfg: Qwen3Config,
    use_mrope: bool,
}

struct VoxtralLayer {
    input_layernorm: RmsNorm,
    self_attn: VoxtralAttention,
    post_attention_layernorm: RmsNorm,
    ada_rms_norm: Option<VoxtralAdaRmsNorm>,
    mlp: VoxtralMlp,
}

struct VoxtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_mrope: bool,
    mrope_section: Option<Vec<usize>>,
    rope_theta: f64,
    sliding_window: Option<usize>,
}

struct VoxtralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct VoxtralAdaRmsNorm {
    down: Linear,
    up: Linear,
}

impl VoxtralLM {
    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = load_embedding_from_candidates(&vb, &cfg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = VoxtralLayer::load(&cfg, vb.pp(format!("layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let lm_head = load_lm_head_from_candidates(&vb, &cfg, &embed_tokens)?;

        let use_mrope = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                scaling.mrope_interleaved.unwrap_or(false) || scaling.interleaved.unwrap_or(false)
            })
            .unwrap_or(false);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            cfg,
            use_mrope,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn physical_context_limit(&self) -> Option<usize> {
        self.cfg.context_length().map(|context| {
            self.cfg
                .sliding_window()
                .map_or(context, |window| context.min(window))
        })
    }

    pub(crate) fn model_context_limit(&self) -> Option<usize> {
        self.cfg.context_length()
    }

    pub(crate) fn managed_inference_state_contract(
        &self,
        domain: StateDomainId,
        storage_dtype: DType,
        preferred_page_tokens: usize,
    ) -> Result<InferenceStateContract> {
        let cache_domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
            domain,
            clock: StateClock::DecoderTokens,
            num_layers: self.cfg.num_hidden_layers,
            num_query_heads: self.cfg.num_attention_heads,
            num_kv_heads: self.cfg.num_key_value_heads,
            key_head_dim: self.cfg.head_dim(),
            value_head_dim: self.cfg.head_dim(),
            sliding_window: self.cfg.sliding_window(),
            storage_dtype,
            preferred_page_tokens,
            prefix: PrefixPolicy::CommittedPages {
                positions: PositionSemantics::Absolute,
            },
        })?;
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::PagedAttention(cache_domain)],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(domain.get()),
                domains: vec![domain],
                prefix_shareable: true,
            }],
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    pub(crate) fn forward_managed_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden, prepared) = self.prepare_managed_hidden_with_embeds(
            embeds,
            start_pos,
            cache,
            position_ids,
            t_cond,
        )?;
        let logits = self.logits_from_hidden(&hidden)?;
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    pub(crate) fn forward_managed_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (hidden, prepared) = self.prepare_managed_hidden_with_embeds(
            embeds,
            start_pos,
            cache,
            position_ids,
            t_cond,
        )?;
        cache.commit_prepared(prepared)?;
        Ok(hidden)
    }

    fn prepare_managed_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        position_ids: Option<&Tensor>,
        t_cond: Option<&Tensor>,
    ) -> Result<(Tensor, PreparedPhysicalPagedStep)> {
        let (batch_size, sequence_len, hidden_size) = embeds.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Voxtral managed embeddings expect [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        let end_pos = start_pos
            .checked_add(sequence_len)
            .ok_or_else(|| Error::InvalidInput("Voxtral managed context length overflow".into()))?;
        if self.cfg.context_length().is_some_and(|limit| end_pos > limit) {
            return Err(Error::InvalidInput(format!(
                "Voxtral sequence ends at {end_pos}, beyond the loaded model context"
            )));
        }
        if let Some(window) = self.cfg.sliding_window() {
            cache.advance_sliding_window_for_append(start_pos, sequence_len, window)?;
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        cache.slots_for_append(start_pos, sequence_len)?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let mut x = embeds.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_managed(
                &x,
                start_pos,
                position_ids,
                cache,
                &mut prepared,
                idx,
                t_cond,
            )?;
        }
        let hidden = self.norm.forward(&x)?;
        Ok((hidden, prepared))
    }

    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        linear_forward_last_dim(&self.lm_head, hidden)
    }
}

fn load_embedding_from_candidates(vb: &VarBuilder, cfg: &Qwen3Config) -> Result<Embedding> {
    let root = vb.root();
    for candidate in EMBEDDING_WEIGHT_CANDIDATES {
        if root.contains_tensor(candidate) {
            let embeddings = root.get((cfg.vocab_size, cfg.hidden_size), candidate)?;
            return Ok(Embedding::new(embeddings, cfg.hidden_size));
        }
    }

    Err(Error::ModelLoadError(format!(
        "Voxtral checkpoint is missing token embedding weights; tried {}",
        EMBEDDING_WEIGHT_CANDIDATES.join(", ")
    )))
}

fn load_lm_head_from_candidates(
    vb: &VarBuilder,
    cfg: &Qwen3Config,
    embed_tokens: &Embedding,
) -> Result<Linear> {
    let root = vb.root();
    for candidate in LM_HEAD_WEIGHT_CANDIDATES {
        if root.contains_tensor(candidate) {
            let weight = root.get((cfg.vocab_size, cfg.hidden_size), candidate)?;
            return Ok(Linear::new(weight, None));
        }
    }

    if cfg.tie_word_embeddings {
        return Ok(Linear::new(embed_tokens.embeddings().clone(), None));
    }

    Err(Error::ModelLoadError(format!(
        "Voxtral checkpoint is missing LM head weights and tie_word_embeddings is false; tried {}",
        LM_HEAD_WEIGHT_CANDIDATES.join(", "),
    )))
}

impl VoxtralLayer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attention_norm"))?;
        let self_attn = VoxtralAttention::load(cfg, vb.pp("attention"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;
        let ada_rms_norm = VoxtralAdaRmsNorm::load(cfg, vb.clone())?;
        let mlp = VoxtralMlp::load(cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            ada_rms_norm,
            mlp,
        })
    }

    fn forward_managed(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_managed(&normed, start_pos, position_ids, cache, prepared, layer_idx)
            .map_err(|err| {
                Error::InferenceError(format!(
                    "Voxtral LM layer {layer_idx} attention failed: {err}"
                ))
            })?;
        let x = x.broadcast_add(&attn_out)?;

        let mut normed = self.post_attention_layernorm.forward(&x)?;
        if let Some(ada_rms_norm) = &self.ada_rms_norm {
            let t_cond = t_cond.ok_or_else(|| {
                Error::InferenceError(
                    "Voxtral LM requires delay conditioning for Ada RMSNorm".to_string(),
                )
            })?;
            let mut scale = ada_rms_norm.forward(t_cond)?;
            if scale.dtype() != normed.dtype() {
                scale = scale.to_dtype(normed.dtype())?;
            }
            let one = Tensor::ones(scale.shape(), scale.dtype(), scale.device())?;
            let scale = scale.broadcast_add(&one)?;
            normed = normed.broadcast_mul(&scale)?;
        }
        let mlp_out = self.mlp.forward(&normed).map_err(|err| {
            Error::InferenceError(format!(
                "Voxtral LM layer {layer_idx} feed-forward failed: {err}"
            ))
        })?;
        let x = x.broadcast_add(&mlp_out)?;
        Ok(x)
    }
}

impl VoxtralAttention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("wq"),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("wk"),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("wv"),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("wo"),
        )?;

        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm")).ok();
        let (use_mrope, mrope_section) = cfg
            .rope_scaling
            .as_ref()
            .map(|scaling| {
                let use_mrope = scaling.mrope_interleaved.unwrap_or(false)
                    || scaling.interleaved.unwrap_or(false);
                (use_mrope, scaling.mrope_section.clone())
            })
            .unwrap_or((false, None));

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            use_mrope,
            mrope_section,
            rope_theta: cfg.rope_theta,
            sliding_window: cfg.sliding_window(),
        })
    }

    fn forward_managed(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Voxtral physical paged attention expects one sequence".into(),
            ));
        }

        let mut q = linear_forward_last_dim(&self.q_proj, x)?.reshape((
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim,
        ))?;
        let mut k = linear_forward_last_dim(&self.k_proj, x)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = linear_forward_last_dim(&self.v_proj, x)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;

        q = self.apply_rope(q, start_pos, position_ids)?;
        k = self.apply_rope(k, start_pos, position_ids)?;

        let q = q.squeeze(0)?;
        let k = k.squeeze(0)?;
        let v = v.squeeze(0)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = match self.sliding_window {
            Some(window) => cache.write_and_attend_with_window(
                layer_idx, prepared, &q, &k, &v, scale, window,
            )?,
            None => cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?,
        };
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        linear_forward_last_dim(&self.o_proj, &out)
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        norm: &Option<RmsNorm>,
        heads: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if let Some(norm) = norm {
            let bsz = x.dim(0)?;
            let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
            let normed = norm.forward(&reshaped)?;
            normed
                .reshape((bsz, seq_len, heads, self.head_dim))
                .map_err(Error::from)
        } else {
            Ok(x)
        }
    }

    fn apply_rope(
        &self,
        x: Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = x.dim(1)?;

        let (cos, sin) = if self.use_mrope {
            let position_ids = if let Some(position_ids) = position_ids {
                position_ids.clone()
            } else {
                let mut data = Vec::with_capacity(3 * seq_len);
                let base = start_pos as i64;
                for _axis in 0..3 {
                    for idx in 0..seq_len {
                        data.push(base + idx as i64);
                    }
                }
                Tensor::from_vec(data, (3, seq_len), x.device())?
            };
            build_mrope_cache(
                seq_len,
                self.head_dim,
                self.rope_theta,
                x.device(),
                x.dtype(),
                &position_ids,
                self.mrope_section.as_deref().unwrap_or(&[]),
            )?
        } else {
            build_rope_cache(
                seq_len,
                self.head_dim,
                start_pos,
                self.rope_theta,
                x.device(),
                x.dtype(),
            )?
        };

        apply_interleaved_rotary_emb(&x, &cos, &sin)
    }
}

fn apply_interleaved_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let bsz = x.dim(0)?;
    let seq_len = x.dim(1)?;
    let heads = x.dim(2)?;
    let head_dim = x.dim(3)?;
    let half_dim = head_dim / 2;
    let x = x.reshape((bsz, seq_len, heads, half_dim, 2))?;
    let x1 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x2 = x.narrow(4, 1, 1)?.squeeze(4)?;

    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
    let rot1 = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let rot2 = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    let rot1 = rot1.unsqueeze(4)?;
    let rot2 = rot2.unsqueeze(4)?;
    Tensor::cat(&[rot1, rot2], 4)?
        .reshape((bsz, seq_len, heads, head_dim))
        .map_err(Error::from)
}

impl VoxtralAdaRmsNorm {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Option<Self>> {
        if !cfg.ada_rms_norm_t_cond {
            return Ok(None);
        }
        let cond_dim = cfg.ada_rms_norm_t_cond_dim;
        if cond_dim == 0 {
            return Err(Error::ModelLoadError(
                "Voxtral ada_rms_norm_t_cond is enabled but ada_rms_norm_t_cond_dim is missing"
                    .to_string(),
            ));
        }

        for (down_path, up_path) in [
            ("ada_rms_norm_t_cond.0", "ada_rms_norm_t_cond.2"),
            ("ada_rms_norm.linear1", "ada_rms_norm.linear2"),
        ] {
            if vb.contains_tensor(&format!("{down_path}.weight"))
                && vb.contains_tensor(&format!("{up_path}.weight"))
            {
                let down = candle_nn::linear_no_bias(cfg.hidden_size, cond_dim, vb.pp(down_path))?;
                let up = candle_nn::linear_no_bias(cond_dim, cfg.hidden_size, vb.pp(up_path))?;
                return Ok(Some(Self { down, up }));
            }
        }

        Err(Error::ModelLoadError(
            "Voxtral checkpoint is missing ada_rms_norm_t_cond weights; tried \
             ada_rms_norm_t_cond.0/2 and ada_rms_norm.linear1/linear2"
                .to_string(),
        ))
    }

    fn forward(&self, t_cond: &Tensor) -> Result<Tensor> {
        let hidden = self.down.forward(t_cond)?.gelu()?;
        self.up.forward(&hidden).map_err(Error::from)
    }
}

impl VoxtralMlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w1"))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("w3"))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("w2"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear_forward_last_dim(&self.gate_proj, x)?;
        let up = linear_forward_last_dim(&self.up_proj, x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        let out = linear_forward_last_dim(&self.down_proj, &hidden)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    fn tiny_cfg() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 0,
            num_key_value_heads: 1,
            max_position_embeddings: None,
            head_dim: Some(2),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            vocab_size: 3,
            lm_head_size: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            ada_rms_norm_t_cond: false,
            ada_rms_norm_t_cond_dim: 0,
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |max, (left, right)| max.max((left - right).abs()))
    }

    #[test]
    fn voxtral_lm_loads_mistral_embedding_and_output_aliases() {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let embed_weight = Tensor::from_vec(
            (0..12).map(|v| v as f32).collect::<Vec<_>>(),
            (3, 4),
            &device,
        )
        .unwrap();
        let output_weight = Tensor::from_vec(
            (0..12).map(|v| (v as f32) / 10.0).collect::<Vec<_>>(),
            (3, 4),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(
            HashMap::from([
                ("tok_embeddings.weight".to_string(), embed_weight.clone()),
                ("output.weight".to_string(), output_weight.clone()),
            ]),
            DType::F32,
            &device,
        );

        let embeddings = load_embedding_from_candidates(&vb, &cfg).unwrap();
        let head = load_lm_head_from_candidates(&vb, &cfg, &embeddings).unwrap();

        assert_eq!(embeddings.embeddings().dims(), &[3, 4]);
        assert_eq!(head.weight().dims(), &[3, 4]);
    }

    #[test]
    fn voxtral_lm_missing_embedding_or_head_is_loader_error() {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &device);

        let embed_err = load_embedding_from_candidates(&vb, &cfg).unwrap_err();
        let embed_weight = Tensor::zeros((3, 4), DType::F32, &device).unwrap();
        let embeddings = Embedding::new(embed_weight, cfg.hidden_size);
        let head_err = load_lm_head_from_candidates(&vb, &cfg, &embeddings).unwrap_err();

        assert!(format!("{embed_err}").contains("missing token embedding weights"));
        assert!(format!("{head_err}").contains("missing LM head weights"));
    }

    #[test]
    fn voxtral_lm_ties_missing_head_to_embeddings_when_configured() {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.tie_word_embeddings = true;
        let embed_weight = Tensor::from_vec(
            (0..12).map(|v| v as f32).collect::<Vec<_>>(),
            (3, 4),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(
            HashMap::from([("tok_embeddings.weight".to_string(), embed_weight.clone())]),
            DType::F32,
            &device,
        );
        let embeddings = load_embedding_from_candidates(&vb, &cfg).unwrap();

        let head = load_lm_head_from_candidates(&vb, &cfg, &embeddings).unwrap();

        assert_eq!(head.weight().dims(), &[3, 4]);
        assert_eq!(max_abs_diff(head.weight(), &embed_weight), 0.0);
    }

    #[test]
    fn voxtral_lm_loads_ada_rms_norm_t_cond_aliases() {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.ada_rms_norm_t_cond = true;
        cfg.ada_rms_norm_t_cond_dim = 2;

        let down = Tensor::from_vec(
            (0..8).map(|v| (v as f32) / 10.0).collect::<Vec<_>>(),
            (2, 4),
            &device,
        )
        .unwrap();
        let up = Tensor::from_vec(
            (0..8).map(|v| (v as f32) / 20.0).collect::<Vec<_>>(),
            (4, 2),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(
            HashMap::from([
                ("layers.0.ada_rms_norm_t_cond.0.weight".to_string(), down),
                ("layers.0.ada_rms_norm_t_cond.2.weight".to_string(), up),
            ]),
            DType::F32,
            &device,
        );

        let ada = VoxtralAdaRmsNorm::load(&cfg, vb.pp("layers.0"))
            .unwrap()
            .unwrap();
        let t_cond = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        let out = ada.forward(&t_cond).unwrap();

        assert_eq!(out.dims(), &[1, 4]);
    }

    #[test]
    fn voxtral_lm_requires_configured_ada_rms_norm_weights() {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.ada_rms_norm_t_cond = true;
        cfg.ada_rms_norm_t_cond_dim = 2;
        let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &device);

        let err = VoxtralAdaRmsNorm::load(&cfg, vb.pp("layers.0"))
            .err()
            .unwrap();

        assert!(format!("{err}").contains("missing ada_rms_norm_t_cond weights"));
    }

    #[test]
    fn voxtral_text_rope_uses_interleaved_pairs() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 10.0, 2.0, 20.0], (1, 1, 1, 4), &device).unwrap();
        let cos = Tensor::from_vec(vec![0.5f32, 0.25], (1, 2), &device).unwrap();
        let sin = Tensor::from_vec(vec![0.1f32, 0.2], (1, 2), &device).unwrap();

        let rotated = apply_interleaved_rotary_emb(&x, &cos, &sin).unwrap();

        assert_eq!(
            rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                1.0 * 0.5 - 10.0 * 0.1,
                1.0 * 0.1 + 10.0 * 0.5,
                2.0 * 0.25 - 20.0 * 0.2,
                2.0 * 0.2 + 20.0 * 0.25,
            ]
        );
    }
}

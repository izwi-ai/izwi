//! Voxtral Language Model - Mistral-style architecture variant of Qwen3.
//!
//! This module provides the Voxtral-specific language model loading and inference,
//! which uses different tensor naming conventions (wq/wk/wv/wo, w1/w2/w3) and
//! root-level layer structure compared to standard Qwen3.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    InferenceStateContract, PositionSemantics, PrefixPolicy, StateClock, StateDomainId,
    StateDomainSpec, StateGroupId, StateGroupSpec, CURRENT_INFERENCE_STATE_ABI,
};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen3::core::{
    build_mrope_cache, build_rope_cache, qwen3_decoder_cache_domain, Qwen3Config,
    Qwen3DecoderCacheGeometry,
};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::telemetry::{record_decode_attention_path, DecodeAttentionPath};

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
        cfg.attention_geometry()?;
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
        let attention = self.cfg.attention_geometry()?;
        let cache_domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
            domain,
            clock: StateClock::DecoderTokens,
            num_layers: self.cfg.num_hidden_layers,
            num_query_heads: self.cfg.num_attention_heads,
            num_kv_heads: self.cfg.num_key_value_heads,
            key_head_dim: attention.key_head_dim(),
            value_head_dim: attention.value_head_dim(),
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

    /// Decode one token for every retained row while preserving each row's
    /// absolute position and rotating block-table view.
    pub(crate) fn forward_managed_decode_batch_with_embeds(
        &self,
        embeds: &Tensor,
        start_positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_managed_decode_batch_impl(embeds, start_positions, caches, t_cond, true)
    }

    /// Decode one embedded token per retained row and return the normalized
    /// last hidden state instead of projecting vocabulary logits.
    pub(crate) fn forward_managed_decode_batch_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_managed_decode_batch_impl(embeds, start_positions, caches, t_cond, false)
    }

    fn forward_managed_decode_batch_impl(
        &self,
        embeds: &Tensor,
        start_positions: &[usize],
        caches: &mut [&mut PhysicalPagedKvCache],
        t_cond: Option<&Tensor>,
        project_logits: bool,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, hidden_size) = embeds.dims3()?;
        if batch_size == 0
            || sequence_len != 1
            || hidden_size != self.cfg.hidden_size
            || start_positions.len() != batch_size
            || caches.len() != batch_size
            || t_cond.is_some_and(|condition| {
                condition.dims3().ok() != Some((batch_size, 1, self.cfg.hidden_size))
            })
        {
            return Err(Error::InvalidInput(format!(
                "Voxtral managed decode expects matching [batch,1,{}] rows, got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        if batch_size == 1 {
            return if project_logits {
                self.forward_managed_with_embeds(
                    embeds,
                    start_positions[0],
                    caches[0],
                    None,
                    t_cond,
                )
            } else {
                self.forward_managed_hidden_with_embeds(
                    embeds,
                    start_positions[0],
                    caches[0],
                    None,
                    t_cond,
                )
            };
        }

        let attention = self.cfg.attention_geometry()?;
        for (row, cache) in caches.iter().enumerate() {
            let end = start_positions[row].checked_add(1).ok_or_else(|| {
                Error::InvalidInput("Voxtral managed decode position overflow".into())
            })?;
            if self.cfg.context_length().is_some_and(|limit| end > limit) {
                return Err(Error::InvalidInput(format!(
                    "Voxtral decode row {row} ends at {end}, beyond the loaded model context"
                )));
            }
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                attention.key_head_dim(),
            )?;
            if cache.context_len() != start_positions[row] {
                return Err(Error::InvalidInput(format!(
                    "Voxtral decode row {row} starts at {}, but its cache cursor is {}",
                    start_positions[row],
                    cache.context_len()
                )));
            }
        }
        let first = &*caches[0];
        let arena = first.arena.clone();
        for cache in caches.iter().skip(1) {
            if !Arc::ptr_eq(&cache.arena, &first.arena) {
                return Err(Error::InvalidInput(
                    "Voxtral managed decode rows must share one physical arena".into(),
                ));
            }
        }
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let binding = first.layer_binding(layer_idx)?;
            for cache in caches.iter().skip(1) {
                if cache.layer_binding(layer_idx)? != binding {
                    return Err(Error::InvalidInput(
                        "Voxtral managed decode rows must share every layer binding".into(),
                    ));
                }
            }
        }

        let checkpoints = caches
            .iter()
            .map(|cache| cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let execution = (|| -> Result<Tensor> {
            if let Some(window) = self.cfg.sliding_window() {
                for (row, cache) in caches.iter_mut().enumerate() {
                    cache.advance_sliding_window_for_append(start_positions[row], 1, window)?;
                }
            }
            let combined_slots = caches
                .iter()
                .enumerate()
                .map(|(row, cache)| {
                    cache
                        .slots_for_append(start_positions[row], 1)
                        .map(|slots| slots[0])
                })
                .collect::<Result<Vec<_>>>()?;
            let lowered = arena.lower_slots(&combined_slots)?;
            let metadata = KvDecodeBatchMetadata {
                sequences: caches
                    .iter()
                    .enumerate()
                    .map(|(row, cache)| {
                        let context_len = start_positions[row].checked_add(1).ok_or_else(|| {
                            Error::InvalidInput("Voxtral decode position overflow".into())
                        })?;
                        match self.cfg.sliding_window() {
                            Some(window) => cache.sequence_table_with_window(context_len, window),
                            None => cache.sequence_table(context_len),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let mut completions =
                KvWriteCompletionCollector::new(arena.config(), lowered.logical_slots())?;
            let layer_result = (|| -> Result<Tensor> {
                let mut x = embeds.clone();
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let cache_refs = caches
                        .iter()
                        .map(|cache| &**cache)
                        .collect::<Vec<&PhysicalPagedKvCache>>();
                    x = layer.forward_managed_decode_batch(
                        &x,
                        start_positions,
                        &cache_refs,
                        lowered.as_ref(),
                        &metadata,
                        &mut completions,
                        layer_idx,
                        t_cond,
                    )?;
                }
                let hidden = self.norm.forward(&x)?;
                if project_logits {
                    self.logits_from_hidden(&hidden)
                } else {
                    Ok(hidden)
                }
            })();
            let logits = match layer_result {
                Ok(logits) => logits,
                Err(error) => {
                    return match completions.drain() {
                        Ok(()) => Err(error),
                        Err(drain) => Err(Error::InferenceError(format!(
                            "Voxtral decode batch failed: {error}; write-fence drain also failed: {drain}"
                        ))),
                    };
                }
            };
            let completion = Arc::new(completions.seal()?);
            for (committed, cache) in caches.iter_mut().enumerate() {
                cache.commit_shared_completion(
                    start_positions[committed],
                    1,
                    completion.clone(),
                )?;
            }
            Ok(logits)
        })();
        match execution {
            Ok(logits) => Ok(logits),
            Err(error) => {
                let mut rollback_error = None;
                for (cache, checkpoint) in caches.iter_mut().zip(checkpoints) {
                    if let Err(rollback) = cache.restore_logical_checkpoint(checkpoint) {
                        rollback_error.get_or_insert(rollback);
                    }
                }
                match rollback_error {
                    Some(rollback) => Err(Error::InferenceError(format!(
                        "Voxtral decode batch failed: {error}; rollback also failed: {rollback}"
                    ))),
                    None => Err(error),
                }
            }
        }
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
        if self
            .cfg
            .context_length()
            .is_some_and(|limit| end_pos > limit)
        {
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
            self.cfg.attention_geometry()?.key_head_dim(),
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

    #[allow(clippy::too_many_arguments)]
    fn forward_managed_decode_batch(
        &self,
        x: &Tensor,
        start_positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
        t_cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_managed_decode_batch(
                &normed,
                start_positions,
                caches,
                slots,
                metadata,
                completions,
                layer_idx,
            )
            .map_err(|err| {
                Error::InferenceError(format!(
                    "Voxtral LM layer {layer_idx} batched attention failed: {err}"
                ))
            })?;
        let x = x.broadcast_add(&attn_out)?;
        let mut normed = self.post_attention_layernorm.forward(&x)?;
        if let Some(ada_rms_norm) = &self.ada_rms_norm {
            let t_cond = t_cond.ok_or_else(|| {
                Error::InferenceError(
                    "Voxtral LM requires delay conditioning for Ada RMSNorm".into(),
                )
            })?;
            let mut scale = ada_rms_norm.forward(t_cond)?;
            if scale.dtype() != normed.dtype() {
                scale = scale.to_dtype(normed.dtype())?;
            }
            let one = Tensor::ones(scale.shape(), scale.dtype(), scale.device())?;
            normed = normed.broadcast_mul(&scale.broadcast_add(&one)?)?;
        }
        let mlp_out = self.mlp.forward(&normed).map_err(|err| {
            Error::InferenceError(format!(
                "Voxtral LM layer {layer_idx} batched feed-forward failed: {err}"
            ))
        })?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

impl VoxtralAttention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let geometry = cfg.attention_geometry()?;
        let head_dim = geometry.key_head_dim();
        let q_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, geometry.query_width(), vb.pp("wq"))?;
        let k_proj = candle_nn::linear_no_bias(cfg.hidden_size, geometry.key_width(), vb.pp("wk"))?;
        let v_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, geometry.value_width(), vb.pp("wv"))?;
        let o_proj =
            candle_nn::linear_no_bias(geometry.query_width(), cfg.hidden_size, vb.pp("wo"))?;

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
            Some(window) => cache
                .write_and_attend_with_window(layer_idx, prepared, &q, &k, &v, scale, window)?,
            None => cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?,
        };
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        linear_forward_last_dim(&self.o_proj, &out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_managed_decode_batch(
        &self,
        x: &Tensor,
        start_positions: &[usize],
        caches: &[&PhysicalPagedKvCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, sequence_len, _) = x.dims3()?;
        if batch == 0
            || sequence_len != 1
            || start_positions.len() != batch
            || caches.len() != batch
            || metadata.sequences.len() != batch
        {
            return Err(Error::InvalidInput(
                "Voxtral managed decode batch dimensions do not match".into(),
            ));
        }
        let first = caches[0];
        let binding = first.layer_binding(layer_idx)?;
        for cache in caches.iter().skip(1) {
            if !Arc::ptr_eq(&cache.arena, &first.arena)
                || cache.layer_binding(layer_idx)? != binding
            {
                return Err(Error::InvalidInput(
                    "Voxtral managed decode rows must share one arena and layer binding".into(),
                ));
            }
        }
        if slots.arena_id() != first.arena.id() || slots.len() != batch {
            return Err(Error::InvalidInput(
                "Voxtral managed decode received an incompatible slot map".into(),
            ));
        }

        let mut q = linear_forward_last_dim(&self.q_proj, x)?.reshape((
            batch,
            1,
            self.num_heads,
            self.head_dim,
        ))?;
        let mut k = linear_forward_last_dim(&self.k_proj, x)?.reshape((
            batch,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = linear_forward_last_dim(&self.v_proj, x)?.reshape((
            batch,
            1,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, 1)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, 1)?;

        let mut query_rows = Vec::with_capacity(batch);
        let mut key_rows = Vec::with_capacity(batch);
        let mut value_rows = Vec::with_capacity(batch);
        for (row, position) in start_positions.iter().copied().enumerate() {
            let q_row = self.apply_rope(q.i(row)?.unsqueeze(0)?, position, None)?;
            let k_row = self.apply_rope(k.i(row)?.unsqueeze(0)?, position, None)?;
            query_rows.push(q_row.reshape((self.num_heads, self.head_dim))?);
            key_rows.push(k_row.reshape((self.num_kv_heads, self.head_dim))?);
            value_rows.push(v.i(row)?.reshape((self.num_kv_heads, self.head_dim))?);
        }
        let queries = Tensor::stack(&query_rows.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        let keys = Tensor::stack(&key_rows.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        let values = Tensor::stack(&value_rows.iter().collect::<Vec<_>>(), 0)?.contiguous()?;
        let completion = first.arena.write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let (out, completion) = submit_ordered_after_write(completion, || {
            first.arena.paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: metadata,
                    softmax_scale: scale,
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        record_decode_attention_path(DecodeAttentionPath::Paged);
        let out = out.reshape((batch, 1, self.num_heads * self.head_dim))?;
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
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
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

    fn tiny_decode_model(device: &Device, sliding_window: Option<usize>) -> VoxtralLM {
        let mut cfg = tiny_cfg();
        cfg.num_hidden_layers = 1;
        cfg.vocab_size = 16;
        cfg.max_position_embeddings = Some(32);
        cfg.sliding_window = sliding_window;
        cfg.use_sliding_window = sliding_window.is_some();
        let values = |count: usize, offset: usize| {
            (0..count)
                .map(|index| (((index * 7 + offset) % 19) as f32 - 9.0) / 23.0)
                .collect::<Vec<_>>()
        };
        let mut tensors = HashMap::new();
        tensors.insert(
            "tok_embeddings.weight".into(),
            Tensor::from_vec(values(64, 1), (16, 4), device).unwrap(),
        );
        for name in [
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "norm.weight",
        ] {
            tensors.insert(name.into(), Tensor::ones(4, DType::F32, device).unwrap());
        }
        for (name, shape, offset) in [
            ("layers.0.attention.wq.weight", (4, 4), 2),
            ("layers.0.attention.wk.weight", (2, 4), 3),
            ("layers.0.attention.wv.weight", (2, 4), 4),
            ("layers.0.attention.wo.weight", (4, 4), 5),
            ("layers.0.feed_forward.w1.weight", (8, 4), 6),
            ("layers.0.feed_forward.w3.weight", (8, 4), 7),
            ("layers.0.feed_forward.w2.weight", (4, 8), 8),
            ("output.weight", (16, 4), 9),
        ] {
            tensors.insert(
                name.into(),
                Tensor::from_vec(values(shape.0 * shape.1, offset), shape, device).unwrap(),
            );
        }
        VoxtralLM::load(cfg, VarBuilder::from_tensors(tensors, DType::F32, device)).unwrap()
    }

    fn shared_decode_caches(
        rows: usize,
        model_instance: u64,
        pages_per_row: usize,
    ) -> (Arc<CpuKvArena>, Vec<PhysicalPagedKvCache>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: KvArenaId {
                    model_instance: ModelInstanceId::new(model_instance),
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    generation: 1,
                },
                group: KvGroupId::new(0),
                page_tokens: 2,
                capacity_pages: u32::try_from(rows * pages_per_row).unwrap(),
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .unwrap(),
        );
        let caches = (0..rows)
            .map(|row| {
                let blocks = (row * pages_per_row..(row + 1) * pages_per_row)
                    .map(|index| CacheBlockRef {
                        arena: arena.id(),
                        group: arena.config().group,
                        index: u32::try_from(index).unwrap(),
                        slot_generation: 1,
                    })
                    .collect();
                let physical: Arc<dyn KvArena> = arena.clone();
                PhysicalPagedKvCache::new(physical, vec![binding], blocks, 0).unwrap()
            })
            .collect();
        (arena, caches)
    }

    fn deterministic_embeds(len: usize, offset: usize, device: &Device) -> Tensor {
        Tensor::from_vec(
            (0..len * 4)
                .map(|index| ((index + offset) as f32 - 5.0) / 11.0)
                .collect::<Vec<_>>(),
            (1, len, 4),
            device,
        )
        .unwrap()
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        assert!(
            max_abs_diff(left, right) < 1e-4,
            "tensor mismatch: {}",
            max_abs_diff(left, right)
        );
    }

    #[test]
    fn managed_hidden_batch_width_one_matches_scalar_and_commits_once() {
        let device = Device::Cpu;
        let model = tiny_decode_model(&device, None);
        let (_, mut caches) = shared_decode_caches(2, 900, 16);
        let mut scalar = caches.remove(0);
        let mut batch = caches.remove(0);
        let prompt = deterministic_embeds(2, 1, &device);
        for cache in [&mut scalar, &mut batch] {
            model
                .forward_managed_hidden_with_embeds(&prompt, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let step = deterministic_embeds(1, 13, &device);
        let scalar_hidden = model
            .forward_managed_hidden_with_embeds(&step, 2, &mut scalar, None, None)
            .unwrap();
        let batch_hidden = model
            .forward_managed_decode_batch_hidden_with_embeds(&step, &[2], &mut [&mut batch], None)
            .unwrap();

        assert_close(&scalar_hidden, &batch_hidden);
        assert_eq!(batch.context_len(), 3);
        assert_eq!(batch.take_completed_writes().len(), 1);
    }

    #[test]
    fn hidden_batch_preserves_ragged_row_identity_and_shared_fence() {
        let device = Device::Cpu;
        let model = tiny_decode_model(&device, None);
        let (arena, mut caches) = shared_decode_caches(4, 905, 16);
        let mut scalar_a = caches.remove(0);
        let mut scalar_b = caches.remove(0);
        let mut batch_a = caches.remove(0);
        let mut batch_b = caches.remove(0);
        let prompt_a = deterministic_embeds(2, 1, &device);
        let prompt_b = deterministic_embeds(3, 7, &device);
        for cache in [&mut scalar_a, &mut batch_a] {
            model
                .forward_managed_hidden_with_embeds(&prompt_a, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        for cache in [&mut scalar_b, &mut batch_b] {
            model
                .forward_managed_hidden_with_embeds(&prompt_b, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let step_a = deterministic_embeds(1, 17, &device);
        let step_b = deterministic_embeds(1, 23, &device);
        let scalar_a_hidden = model
            .forward_managed_hidden_with_embeds(&step_a, 2, &mut scalar_a, None, None)
            .unwrap();
        let scalar_b_hidden = model
            .forward_managed_hidden_with_embeds(&step_b, 3, &mut scalar_b, None, None)
            .unwrap();
        let before = arena.operation_stats().paged_decode_dispatches;
        let batch_hidden = model
            .forward_managed_decode_batch_hidden_with_embeds(
                &Tensor::cat(&[&step_a, &step_b], 0).unwrap(),
                &[2, 3],
                &mut [&mut batch_a, &mut batch_b],
                None,
            )
            .unwrap();

        assert_close(
            &scalar_a_hidden,
            &batch_hidden.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_b_hidden,
            &batch_hidden.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_eq!(arena.operation_stats().paged_decode_dispatches - before, 1);
        assert_eq!((batch_a.context_len(), batch_b.context_len()), (3, 4));
        let completions_a = batch_a.take_completed_writes();
        let completions_b = batch_b.take_completed_writes();
        assert_eq!((completions_a.len(), completions_b.len()), (1, 1));
        assert!(Arc::ptr_eq(&completions_a[0], &completions_b[0]));
    }

    #[test]
    fn ragged_decode_matches_scalar_at_unequal_positions_and_dispatches_once() {
        let device = Device::Cpu;
        let model = tiny_decode_model(&device, None);
        let (arena, mut caches) = shared_decode_caches(4, 901, 16);
        let mut scalar_a = caches.remove(0);
        let mut scalar_b = caches.remove(0);
        let mut batch_a = caches.remove(0);
        let mut batch_b = caches.remove(0);
        let prompt_a = deterministic_embeds(2, 1, &device);
        let prompt_b = deterministic_embeds(3, 7, &device);
        for cache in [&mut scalar_a, &mut batch_a] {
            model
                .forward_managed_with_embeds(&prompt_a, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        for cache in [&mut scalar_b, &mut batch_b] {
            model
                .forward_managed_with_embeds(&prompt_b, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let step_a = deterministic_embeds(1, 17, &device);
        let step_b = deterministic_embeds(1, 23, &device);
        let scalar_a_logits = model
            .forward_managed_with_embeds(&step_a, 2, &mut scalar_a, None, None)
            .unwrap();
        let scalar_b_logits = model
            .forward_managed_with_embeds(&step_b, 3, &mut scalar_b, None, None)
            .unwrap();
        let embeds = Tensor::cat(&[&step_a, &step_b], 0).unwrap();
        let before = arena.operation_stats().paged_decode_dispatches;
        let batch_logits = model
            .forward_managed_decode_batch_with_embeds(
                &embeds,
                &[2, 3],
                &mut [&mut batch_a, &mut batch_b],
                None,
            )
            .unwrap();

        assert_close(
            &scalar_a_logits,
            &batch_logits.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_b_logits,
            &batch_logits.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_eq!(arena.operation_stats().paged_decode_dispatches - before, 1);
        assert_eq!((batch_a.context_len(), batch_b.context_len()), (3, 4));
        assert_eq!(batch_a.take_completed_writes().len(), 1);
        assert_eq!(batch_b.take_completed_writes().len(), 1);
    }

    #[test]
    fn ragged_decode_rotates_each_sliding_window_like_scalar() {
        let device = Device::Cpu;
        let model = tiny_decode_model(&device, Some(3));
        let (_, mut caches) = shared_decode_caches(4, 902, 2);
        let mut scalar_a = caches.remove(0);
        let mut scalar_b = caches.remove(0);
        let mut batch_a = caches.remove(0);
        let mut batch_b = caches.remove(0);
        let prompt = deterministic_embeds(4, 3, &device);
        for cache in [&mut scalar_a, &mut scalar_b, &mut batch_a, &mut batch_b] {
            model
                .forward_managed_with_embeds(&prompt, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let step_a = deterministic_embeds(1, 31, &device);
        let step_b = deterministic_embeds(1, 37, &device);
        let scalar_a_logits = model
            .forward_managed_with_embeds(&step_a, 4, &mut scalar_a, None, None)
            .unwrap();
        let scalar_b_logits = model
            .forward_managed_with_embeds(&step_b, 4, &mut scalar_b, None, None)
            .unwrap();
        let batch_logits = model
            .forward_managed_decode_batch_with_embeds(
                &Tensor::cat(&[&step_a, &step_b], 0).unwrap(),
                &[4, 4],
                &mut [&mut batch_a, &mut batch_b],
                None,
            )
            .unwrap();

        assert_close(
            &scalar_a_logits,
            &batch_logits.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_b_logits,
            &batch_logits.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_eq!((batch_a.window_start(), batch_b.window_start()), (2, 2));
    }

    #[test]
    fn ragged_decode_rejects_mixed_arenas_without_advancing_rows() {
        let device = Device::Cpu;
        let model = tiny_decode_model(&device, None);
        let (_, mut left) = shared_decode_caches(1, 903, 16);
        let (_, mut right) = shared_decode_caches(1, 904, 16);
        let mut cache_a = left.remove(0);
        let mut cache_b = right.remove(0);
        let prompt = deterministic_embeds(2, 1, &device);
        for cache in [&mut cache_a, &mut cache_b] {
            model
                .forward_managed_with_embeds(&prompt, 0, cache, None, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let embeds = Tensor::cat(
            &[
                &deterministic_embeds(1, 11, &device),
                &deterministic_embeds(1, 13, &device),
            ],
            0,
        )
        .unwrap();
        assert!(model
            .forward_managed_decode_batch_hidden_with_embeds(
                &embeds,
                &[2, 2],
                &mut [&mut cache_a, &mut cache_b],
                None,
            )
            .is_err());
        assert_eq!((cache_a.context_len(), cache_b.context_len()), (2, 2));
        assert!(cache_a.take_completed_writes().is_empty());
        assert!(cache_b.take_completed_writes().is_empty());
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

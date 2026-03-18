//! Qwen3-TTS Talker model implementation.
//!
//! The talker is the main LLM component that generates speech tokens from text input.
//! It uses a Qwen3 architecture with MRoPE (Multi-modal Rotary Position Embeddings)
//! to handle both text and audio modalities.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::config::TalkerConfig;
use crate::models::shared::attention::batched::{
    batched_scaled_dot_product_attention, BatchedAttentionConfig, BatchedAttentionInput,
};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};
use crate::models::shared::attention::paged::{
    append_to_pages, default_kv_page_size, default_kv_quantization, materialize_pages,
    paged_decode_attention, repeat_kv, KvCacheQuantization, KvPage,
};
use crate::models::shared::weights::mlx;

/// KV Cache for the talker model
pub struct TalkerCache {
    k_pages: Vec<Vec<KvPage>>,
    v_pages: Vec<Vec<KvPage>>,
    page_size: usize,
    quantization: KvCacheQuantization,
}

impl TalkerCache {
    /// Create a new cache for the specified number of layers
    pub fn new(num_layers: usize) -> Self {
        Self::with_page_size_and_quantization(
            num_layers,
            default_kv_page_size(),
            default_kv_quantization(),
        )
    }

    /// Create a new cache with explicit page size.
    pub fn with_page_size(num_layers: usize, page_size: usize) -> Self {
        Self::with_page_size_and_quantization(num_layers, page_size, default_kv_quantization())
    }

    pub fn with_page_size_and_quantization(
        num_layers: usize,
        page_size: usize,
        quantization: KvCacheQuantization,
    ) -> Self {
        Self {
            k_pages: vec![Vec::new(); num_layers],
            v_pages: vec![Vec::new(); num_layers],
            page_size: page_size.max(1),
            quantization,
        }
    }

    /// Append new k, v to paged cache.
    fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<()> {
        append_to_pages(
            self.page_size,
            &mut self.k_pages[layer],
            &k,
            self.quantization,
        )?;
        append_to_pages(
            self.page_size,
            &mut self.v_pages[layer],
            &v,
            self.quantization,
        )?;
        Ok(())
    }

    fn pages(&self, layer: usize) -> Option<(&[KvPage], &[KvPage])> {
        let k = self.k_pages.get(layer)?;
        let v = self.v_pages.get(layer)?;
        if k.is_empty() || v.is_empty() {
            None
        } else {
            Some((k.as_slice(), v.as_slice()))
        }
    }

    fn materialize(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let k = self.k_pages.get(layer).ok_or_else(|| {
            Error::InferenceError(format!("Invalid TalkerCache layer index: {layer}"))
        })?;
        let v = self.v_pages.get(layer).ok_or_else(|| {
            Error::InferenceError(format!("Invalid TalkerCache layer index: {layer}"))
        })?;
        Ok((materialize_pages(k)?, materialize_pages(v)?))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for i in 0..self.k_pages.len() {
            self.k_pages[i].clear();
            self.v_pages[i].clear();
        }
    }
}

/// Multi-head attention with optional Q/K normalization
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    use_mrope: bool,
    mrope_section: Vec<usize>,
}

impl Attention {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();

        let q_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = mlx::load_linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = mlx::load_linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        // Q/K normalization (optional, for Qwen3)
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm")).ok();

        let use_mrope = cfg.uses_mrope();
        let mrope_section = cfg.mrope_section();

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
            rope_theta: cfg.rope_theta,
            use_mrope,
            mrope_section,
        })
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
        let half_dim = self.head_dim / 2;

        let (cos, sin) = if self.use_mrope {
            let position_ids = if let Some(position_ids) = position_ids {
                position_ids.clone()
            } else {
                // Default: create standard position IDs replicated for 3 dims
                let mut data = Vec::with_capacity(3 * seq_len);
                let base = start_pos as i64;
                for _ in 0..3 {
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
                &self.mrope_section,
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

        // Qwen RoPE uses rotate_half(x) over [first_half, second_half].
        let cos = Tensor::cat(&[cos.clone(), cos], 1)?;
        let sin = Tensor::cat(&[sin.clone(), sin], 1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let minus_one = Tensor::from_vec(vec![-1.0f32], (1,), x.device())?.to_dtype(x.dtype())?;
        let neg_x2 = x2.broadcast_mul(&minus_one)?;
        let rotated = Tensor::cat(&[neg_x2, x1], 3)?;

        let out = x.broadcast_mul(&cos)?;
        out.broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: Option<&mut TalkerCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let use_batched = cache.is_none() && start_pos == 0 && bsz > 1;

        // Project to Q, K, V
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

        // Apply Q/K normalization if present
        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;

        // Apply RoPE
        q = self.apply_rope(q, start_pos, position_ids)?;
        k = self.apply_rope(k, start_pos, position_ids)?;

        // Repeat K/V for GQA
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        // Update cache and resolve KV view
        let (mut k, mut v) = if let Some(cache) = cache {
            cache.append(layer_idx, k, v)?;

            // Decode path hot loop: for single-token decode, avoid rematerializing full KV.
            if seq_len == 1 && start_pos > 0 {
                if let Some((k_pages, v_pages)) = cache.pages(layer_idx) {
                    let out = paged_decode_attention(
                        &q,
                        k_pages,
                        v_pages,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                    )?;
                    let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
                    return self.o_proj.forward(&out).map_err(Error::from);
                }
            }

            cache.materialize(layer_idx)?
        } else {
            (k, v)
        };

        if use_batched {
            let q = q.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
            k = k.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
            v = v.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
            let attention_mask = if seq_len > 1 {
                Some(causal_mask(
                    seq_len,
                    seq_len,
                    start_pos,
                    q.device(),
                    q.dtype(),
                )?)
            } else {
                None
            };
            let input = BatchedAttentionInput {
                queries: q,
                keys: k,
                values: v,
                attention_mask,
                seq_lengths: vec![seq_len; bsz],
            };
            let config = BatchedAttentionConfig::new(self.num_heads, self.head_dim);
            let out = batched_scaled_dot_product_attention(&input, &config)?;
            return self.o_proj.forward(&out).map_err(Error::from);
        }

        // Transpose for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let total_len = k.dim(2)?;
        if seq_len == 1 {
            let scale = 1.0f32 / (self.head_dim as f32).sqrt();
            if let Ok(sdpa_out) = ops::sdpa(&q, &k, &v, None, false, scale, 1.0) {
                let out = sdpa_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&out).map_err(Error::from);
            }
        }
        if flash_attention_requested() && start_pos == 0 && total_len == seq_len {
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

        // Reshape for batch matrix multiply
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        // Compute attention
        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scale_t =
            Tensor::from_vec(vec![scale as f32], (1,), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale_t)?;

        // Apply causal mask
        if seq_len > 1 {
            let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        // Softmax and apply to values
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        // Reshape back
        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Transformer layer
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

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
        position_ids: Option<&Tensor>,
        cache: Option<&mut TalkerCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out =
            self.self_attn
                .forward(&normed, start_pos, position_ids, cache, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Text projection MLP to project text embeddings to model hidden size
struct TextProjection {
    linear_fc1: Linear,
    linear_fc2: Linear,
}

impl TextProjection {
    fn load(text_hidden_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_fc1 = mlx::load_linear(text_hidden_size, text_hidden_size, vb.pp("linear_fc1"))?;
        let linear_fc2 = mlx::load_linear(text_hidden_size, hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_fc1.forward(x)?;
        let x = ops::silu(&x)?;
        self.linear_fc2.forward(&x).map_err(Error::from)
    }
}

/// Qwen3-TTS Talker model
pub struct TalkerModel {
    text_embedding: Embedding,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    cfg: TalkerConfig,
    use_mrope: bool,
}

impl TalkerModel {
    /// Load the talker model from VarBuilder
    pub fn load(cfg: TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = mlx::load_embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb.pp("model.text_embedding"),
        )?;
        let text_projection = TextProjection::load(
            cfg.text_hidden_size,
            cfg.hidden_size,
            vb.pp("text_projection"),
        )?;
        let codec_embedding = mlx::load_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.codec_embedding"),
        )?;
        let lm_head =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("codec_head"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let use_mrope = cfg.uses_mrope();

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            cfg,
            use_mrope,
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass starting from token IDs
    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut TalkerCache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache, None)
    }

    /// Get embeddings for token IDs
    /// Uses text_embedding for text tokens and codec_embedding for codec tokens
    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        let ids = input_ids.to_vec2::<u32>()?;

        // Fast path: all text ids.
        if ids.iter().all(|row| {
            row.iter()
                .all(|id| (*id as usize) < self.cfg.text_vocab_size)
        }) {
            let text_embeds = self.text_embedding.forward(input_ids)?;
            return self.text_projection.forward(&text_embeds);
        }

        // Mixed text/codec path:
        // - text ids: [0, text_vocab_size)
        // - codec/control ids: text_vocab_size + codec_id
        let mut batch_embeds = Vec::with_capacity(ids.len());
        for row in ids.iter() {
            if row.is_empty() {
                return Err(Error::InvalidInput(
                    "Empty token row in talker embeddings".to_string(),
                ));
            }

            let mut token_embeds = Vec::with_capacity(row.len());
            for &token_id in row {
                let embed = if (token_id as usize) < self.cfg.text_vocab_size {
                    let token = Tensor::from_vec(vec![token_id], (1,), &self.device)?;
                    let text = self.text_embedding.forward(&token)?;
                    self.text_projection.forward(&text)?
                } else {
                    let codec_id = token_id - self.cfg.text_vocab_size as u32;
                    if (codec_id as usize) >= self.cfg.vocab_size {
                        return Err(Error::InvalidInput(format!(
                            "Codec token out of range: token_id={token_id}, codec_id={codec_id}, codec_vocab={}",
                            self.cfg.vocab_size
                        )));
                    }
                    let token = Tensor::from_vec(vec![codec_id], (1,), &self.device)?;
                    self.codec_embedding.forward(&token)?
                };
                token_embeds.push(embed);
            }

            let row_embed = Tensor::cat(&token_embeds, 0)?;
            batch_embeds.push(row_embed);
        }

        Tensor::stack(&batch_embeds, 0).map_err(Error::from)
    }

    /// Forward pass with pre-computed embeddings
    pub fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: Option<&mut TalkerCache>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_hidden, logits) =
            self.forward_with_embeds_and_hidden(embeds, start_pos, cache, position_ids)?;
        Ok(logits)
    }

    /// Forward pass with pre-computed embeddings, returning both hidden states and logits.
    pub fn forward_with_embeds_and_hidden(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut TalkerCache>,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut x = embeds.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            x = layer.forward(&x, start_pos, position_ids, cache_ref, idx)?;
        }
        let hidden = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((hidden, logits))
    }

    /// Prefill pass from externally assembled embeddings.
    /// Returns (last_hidden, last_logits), each shaped [1, 1, ...].
    pub fn prefill_with_embeds(
        &self,
        embeds: &Tensor,
        cache: &mut TalkerCache,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden, logits) =
            self.forward_with_embeds_and_hidden(embeds, 0, Some(cache), position_ids)?;
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let last_logits = logits.i((.., seq_len - 1..seq_len, ..))?;
        Ok((last_hidden, last_logits))
    }

    /// Incremental generation step from an externally assembled single-step embedding.
    /// Returns (hidden, logits) for the provided step; shapes are [1, 1, ...].
    pub fn generate_step_with_embed(
        &self,
        input_embed: &Tensor,
        cache: &mut TalkerCache,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_embeds_and_hidden(input_embed, offset, Some(cache), None)
    }

    /// Get projected text embeddings for a sequence of token IDs.
    /// Output shape: [1, seq_len, hidden_size].
    pub fn get_projected_text_embeddings(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Ok(Tensor::zeros(
                (1, 0, self.cfg.hidden_size),
                DType::F32,
                &self.device,
            )?);
        }
        let ids_tensor = Tensor::from_vec(token_ids.to_vec(), (token_ids.len(),), &self.device)?;
        let embeds = self.text_embedding.forward(&ids_tensor)?;
        let embeds = embeds.unsqueeze(0)?;
        self.text_projection.forward(&embeds)
    }

    /// Get projected text embedding for a single token ID.
    /// Output shape: [1, 1, hidden_size].
    pub fn get_projected_special_embed(&self, token_id: u32) -> Result<Tensor> {
        self.get_projected_text_embeddings(&[token_id])
    }

    /// Get codec embedding for a single codec token ID.
    /// Output shape: [1, 1, hidden_size].
    pub fn get_codec_embedding(&self, token_id: u32) -> Result<Tensor> {
        let token_tensor = Tensor::from_vec(vec![token_id], (1,), &self.device)?;
        let embed = self.codec_embedding.forward(&token_tensor)?;
        embed.unsqueeze(0).map_err(Error::from)
    }

    /// Get codec embeddings for a sequence of codec token IDs.
    /// Output shape: [1, seq_len, hidden_size].
    pub fn get_codec_embedding_batch(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Ok(Tensor::zeros(
                (1, 0, self.cfg.hidden_size),
                DType::F32,
                &self.device,
            )?);
        }
        let ids_tensor = Tensor::from_vec(token_ids.to_vec(), (token_ids.len(),), &self.device)?;
        let embed = self.codec_embedding.forward(&ids_tensor)?;
        embed.unsqueeze(0).map_err(Error::from)
    }

    /// Check if using MRoPE
    pub fn uses_mrope(&self) -> bool {
        self.use_mrope
    }
}

/// Build standard RoPE cache
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
        for &inv in inv_freq.iter() {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Build MRoPE cache for multi-modal position encoding
fn build_mrope_cache(
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;

    if mrope_section.len() < 3 {
        return build_rope_cache(seq_len, head_dim, 0, rope_theta, device, dtype);
    }

    let positions = position_ids.to_vec2::<i64>()?;
    if positions.len() != 3 || positions.iter().any(|axis| axis.len() < seq_len) {
        return build_rope_cache(seq_len, head_dim, 0, rope_theta, device, dtype);
    }

    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }

    // Match Qwen3 interleaved MRoPE layout.
    let h_limit = mrope_section[1].saturating_mul(3).min(half_dim);
    let w_limit = mrope_section[2].saturating_mul(3).min(half_dim);

    let mut cos_data = Vec::with_capacity(seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(seq_len * half_dim);
    for t in 0..seq_len {
        let p0 = positions[0][t] as f32;
        let p1 = positions[1][t] as f32;
        let p2 = positions[2][t] as f32;
        for (dim, &inv) in inv_freq.iter().enumerate() {
            let pos = if dim % 3 == 1 && dim < h_limit {
                p1
            } else if dim % 3 == 2 && dim < w_limit {
                p2
            } else {
                p0
            };
            let angle = pos * inv;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    let cos = Tensor::from_vec(cos_data, (seq_len, half_dim), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin_data, (seq_len, half_dim), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Create causal attention mask
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

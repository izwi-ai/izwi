//! Code Predictor for multi-codebook RVQ token generation.
//!
//! The code predictor generates the residual codebook tokens after the talker
//! has produced the first (semantic) codebook. It uses a smaller transformer
//! for efficient multi-token prediction.

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;
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

/// KV Cache for the code predictor
pub struct CodePredictorCache {
    k_pages: Vec<Vec<KvPage>>,
    v_pages: Vec<Vec<KvPage>>,
    page_size: usize,
    quantization: KvCacheQuantization,
}

impl CodePredictorCache {
    /// Create a new cache
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

    /// Append new k, v to cache
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
            Error::InferenceError(format!("Invalid CodePredictorCache layer index: {layer}"))
        })?;
        let v = self.v_pages.get(layer).ok_or_else(|| {
            Error::InferenceError(format!("Invalid CodePredictorCache layer index: {layer}"))
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

/// Code Predictor model
pub struct CodePredictor {
    codec_embeddings: Vec<Embedding>,
    small_to_mtp_projection: Option<Linear>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    device: Device,
    cfg: CodePredictorConfig,
    num_code_groups: usize,
}

impl CodePredictor {
    /// Load the code predictor from VarBuilder
    pub fn load(cfg: CodePredictorConfig, vb: VarBuilder, num_code_groups: usize) -> Result<Self> {
        // Use text_hidden_size for codec embeddings if specified, otherwise hidden_size
        let codec_embed_dim = cfg.text_hidden_size.unwrap_or(cfg.hidden_size);

        // Load codec embeddings (one per codebook, but weights only have 15)
        // The model has embeddings 0-14 (15 total), not 16
        let num_codec_embeddings = num_code_groups.min(15);
        let mut codec_embeddings = Vec::with_capacity(num_codec_embeddings);
        for idx in 0..num_codec_embeddings {
            let embed = mlx::load_embedding(
                cfg.vocab_size,
                codec_embed_dim,
                vb.pp(format!("model.codec_embedding.{idx}")),
            )?;
            codec_embeddings.push(embed);
        }

        let small_to_mtp_projection = if codec_embed_dim != cfg.hidden_size {
            Some(mlx::load_linear(
                codec_embed_dim,
                cfg.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Load output heads (one per code group, but weights only have 15)
        let num_lm_heads = num_code_groups.min(15);
        let mut lm_heads = Vec::with_capacity(num_lm_heads);
        for idx in 0..num_lm_heads {
            let head = mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp(format!("lm_head.{idx}")),
            )?;
            lm_heads.push(head);
        }

        Ok(Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            device: vb.device().clone(),
            cfg,
            num_code_groups,
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

    /// Get number of code groups
    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    /// Number of acoustic code groups predicted after the semantic codebook.
    pub fn num_acoustic_groups(&self) -> usize {
        self.codec_embeddings.len()
    }

    /// Forward pass to predict all code groups from first codebook
    pub fn forward(
        &self,
        first_codebook: &Tensor,
        start_pos: usize,
        cache: Option<&mut CodePredictorCache>,
    ) -> Result<Vec<Tensor>> {
        // Embed the first codebook tokens using the first codec embedding
        let mut x = self.codec_embeddings[0].forward(first_codebook)?;
        if let Some(proj) = &self.small_to_mtp_projection {
            x = proj.forward(&x)?;
        }

        // Pass through transformer layers
        let mut cache_ref = cache;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, start_pos, cache_ref.as_deref_mut(), idx)?;
        }

        // Final normalization
        let x = self.norm.forward(&x)?;

        // Generate logits for each code group
        let mut outputs = Vec::with_capacity(self.num_code_groups);
        for head in &self.lm_heads {
            let logits = head.forward(&x)?;
            outputs.push(logits);
        }

        Ok(outputs)
    }

    /// Generate all acoustic code groups autoregressively.
    ///
    /// The predictor consumes [talker_hidden, semantic_embed] as prefill context,
    /// then predicts 15 acoustic codes one-by-one using KV cache.
    pub fn generate_acoustic_codes(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cache: &mut CodePredictorCache,
    ) -> Result<Vec<u32>> {
        cache.clear();

        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;
        let mut hidden = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&input)?
        } else {
            input
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, 0, Some(cache), idx)?;
        }
        hidden = self.norm.forward(&hidden)?;

        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;

        let num_acoustic = self.lm_heads.len();
        if num_acoustic == 0 {
            return Ok(Vec::new());
        }

        let first_logits = self.lm_heads[0].forward(&last_hidden)?;
        let mut prev_code = argmax_token(&first_logits.i((0, 0))?)?;
        let mut all_codes = Vec::with_capacity(num_acoustic);
        all_codes.push(prev_code);

        let mut offset = seq_len;
        for group_idx in 1..num_acoustic {
            let prev_tensor = Tensor::from_vec(vec![prev_code], (1,), &self.device)?;
            let mut step_hidden = self.codec_embeddings[group_idx - 1].forward(&prev_tensor)?;
            step_hidden = step_hidden.unsqueeze(0)?;
            if let Some(proj) = &self.small_to_mtp_projection {
                step_hidden = proj.forward(&step_hidden)?;
            }

            for (idx, layer) in self.layers.iter().enumerate() {
                step_hidden = layer.forward(&step_hidden, offset, Some(cache), idx)?;
            }
            step_hidden = self.norm.forward(&step_hidden)?;

            let logits = self.lm_heads[group_idx].forward(&step_hidden)?;
            prev_code = argmax_token(&logits.i((0, 0))?)?;
            all_codes.push(prev_code);
            offset += 1;
        }

        Ok(all_codes)
    }

    /// Sum acoustic embeddings for the 15 generated acoustic codes.
    /// Returned tensor shape is [1, 1, codec_embed_dim].
    pub fn get_acoustic_embeddings_sum(&self, acoustic_codes: &[u32]) -> Result<Tensor> {
        if acoustic_codes.len() != self.codec_embeddings.len() {
            return Err(Error::InvalidInput(format!(
                "Expected {} acoustic codes, got {}",
                self.codec_embeddings.len(),
                acoustic_codes.len()
            )));
        }

        let first_code = Tensor::from_vec(vec![acoustic_codes[0]], (1,), &self.device)?;
        let mut sum = self.codec_embeddings[0]
            .forward(&first_code)?
            .unsqueeze(0)?;

        for (group_idx, &code) in acoustic_codes.iter().enumerate().skip(1) {
            let code_tensor = Tensor::from_vec(vec![code], (1,), &self.device)?;
            let embed = self.codec_embeddings[group_idx]
                .forward(&code_tensor)?
                .unsqueeze(0)?;
            sum = sum.broadcast_add(&embed)?;
        }

        Ok(sum)
    }
}

/// Transformer layer for code predictor
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
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
        cache: Option<&mut CodePredictorCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(&normed, start_pos, cache, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Multi-head attention for code predictor
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl Attention {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
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
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
        })
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        heads: usize,
        seq_len: usize,
        norm: &RmsNorm,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
        let normed = norm.forward(&reshaped)?;
        normed
            .reshape((bsz, seq_len, heads, self.head_dim))
            .map_err(Error::from)
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
        cache: Option<&mut CodePredictorCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let use_batched = cache.is_none() && start_pos == 0 && bsz > 1;

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

        q = self.apply_qk_norm(q, self.num_heads, seq_len, &self.q_norm)?;
        k = self.apply_qk_norm(k, self.num_kv_heads, seq_len, &self.k_norm)?;

        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

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

        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scale_t =
            Tensor::from_vec(vec![scale as f32], (1,), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale_t)?;

        if seq_len > 1 {
            let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

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
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
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

/// Build RoPE cache
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

fn argmax_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    Ok(max_idx as u32)
}

//! Qwen3-TTS Talker model implementation.
//!
//! The talker is the main LLM component that generates speech tokens from text input.
//! It uses a Qwen3 architecture with MRoPE (Multi-modal Rotary Position Embeddings)
//! to handle both text and audio modalities.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::config::TalkerConfig;
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window, duplicate_rope_window, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as TalkerPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::weights::mlx;

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
    rope_inv_freq: Vec<f32>,
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
            rope_inv_freq: build_rope_inv_freq(head_dim, cfg.rope_theta),
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
            if let Some(position_ids) = position_ids {
                build_mrope_cache(
                    seq_len,
                    x.device(),
                    x.dtype(),
                    position_ids,
                    &self.mrope_section,
                    &self.rope_inv_freq,
                )?
            } else {
                let position_ids = repeated_mrope_position_ids(seq_len, start_pos, x.device())?;
                build_mrope_cache(
                    seq_len,
                    x.device(),
                    x.dtype(),
                    &position_ids,
                    &self.mrope_section,
                    &self.rope_inv_freq,
                )?
            }
        } else {
            build_rope_window(
                seq_len,
                start_pos,
                &self.rope_inv_freq,
                x.device(),
                x.dtype(),
            )?
        };

        // Qwen RoPE uses rotate_half(x) over [first_half, second_half].
        let (cos, sin) = if cos.dim(1)? == half_dim {
            duplicate_rope_window(cos, sin)?
        } else {
            (cos, sin)
        };
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rotated = qwen_rotate_half(&x, half_dim)?;

        let out = x.broadcast_mul(&cos)?;
        out.broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &TalkerPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        if bsz != 1 || seq_len == 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker attention expects [1,sequence,hidden], got {:?}",
                x.dims()
            )));
        }

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

        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;
        q = self.apply_rope(q, start_pos, position_ids)?;
        k = self.apply_rope(k, start_pos, position_ids)?;

        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
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

    fn forward_physical(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: &TalkerPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_physical(
            &normed,
            start_pos,
            position_ids,
            cache,
            prepared,
            layer_idx,
        )?;
        let x = x.broadcast_add(&attn_out)?;

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

    /// Run pre-computed embeddings against retained physical pages.
    ///
    /// `start_pos` must equal the cache's authoritative cursor. Every layer
    /// writes the same prepared slots, and the cursor advances only after all
    /// layers, final normalization, and the language-model head succeed.
    pub fn forward_physical_with_embeds_and_hidden(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut TalkerPhysicalCache,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, sequence_len, hidden_size) = embeds.dims3()?;
        if batch_size != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker expects [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        if start_pos != cache.context_len() {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker starts at {start_pos}, expected retained cursor {}",
                cache.context_len()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
        )?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;

        let mut x = embeds.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_physical(&x, start_pos, position_ids, cache, &mut prepared, idx)?;
        }
        let hidden = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&hidden)?;
        cache.commit_prepared(prepared)?;
        Ok((hidden, logits))
    }

    /// Prefill a fresh retained physical talker cache.
    pub fn prefill_physical_with_embeds(
        &self,
        embeds: &Tensor,
        cache: &mut TalkerPhysicalCache,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker prefill requires cursor 0, got {}",
                cache.context_len()
            )));
        }
        let (hidden, logits) =
            self.forward_physical_with_embeds_and_hidden(embeds, 0, cache, position_ids)?;
        let seq_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let last_logits = logits.i((.., seq_len - 1..seq_len, ..))?;
        Ok((last_hidden, last_logits))
    }

    /// Append one generation token at the retained physical cursor.
    pub fn generate_physical_step_with_embed(
        &self,
        input_embed: &Tensor,
        cache: &mut TalkerPhysicalCache,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, sequence_len, hidden_size) = input_embed.dims3()?;
        if batch_size != 1 || sequence_len != 1 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker step expects [1,1,{}], got {:?}",
                self.cfg.hidden_size,
                input_embed.dims()
            )));
        }
        let start_pos = cache.context_len();
        self.forward_physical_with_embeds_and_hidden(input_embed, start_pos, cache, None)
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

/// Build MRoPE cache for multi-modal position encoding
fn build_mrope_cache(
    seq_len: usize,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
    inv_freq: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freq.len();

    if mrope_section.len() < 3 {
        return build_rope_window(seq_len, 0, inv_freq, device, dtype);
    }

    let positions = position_ids.to_vec2::<i64>()?;
    if positions.len() != 3 || positions.iter().any(|axis| axis.len() < seq_len) {
        return build_rope_window(seq_len, 0, inv_freq, device, dtype);
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

fn repeated_mrope_position_ids(
    seq_len: usize,
    start_pos: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(3 * seq_len);
    let base = start_pos as i64;
    for _ in 0..3 {
        for idx in 0..seq_len {
            data.push(base + idx as i64);
        }
    }
    Tensor::from_vec(data, (3, seq_len), device).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_mrope_positions_match_standard_rope_when_axes_equal() {
        let device = Device::Cpu;
        let seq_len = 3;
        let start_pos = 2;
        let inv_freq = build_rope_inv_freq(6, 10_000.0);
        let position_ids = repeated_mrope_position_ids(seq_len, start_pos, &device).unwrap();

        let (mrope_cos, mrope_sin) = build_mrope_cache(
            seq_len,
            &device,
            DType::F32,
            &position_ids,
            &[1, 1, 1],
            &inv_freq,
        )
        .unwrap();
        let (standard_cos, standard_sin) =
            build_rope_window(seq_len, start_pos, &inv_freq, &device, DType::F32).unwrap();

        assert_eq!(
            mrope_cos.to_vec2::<f32>().unwrap(),
            standard_cos.to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            mrope_sin.to_vec2::<f32>().unwrap(),
            standard_sin.to_vec2::<f32>().unwrap()
        );
    }
}

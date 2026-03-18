//! Voxtral Language Model - Mistral-style architecture variant of Qwen3.
//!
//! This module provides the Voxtral-specific language model loading and inference,
//! which uses different tensor naming conventions (wq/wk/wv/wo, w1/w2/w3) and
//! root-level layer structure compared to standard Qwen3.

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::{
    build_mrope_cache, build_rope_cache, causal_mask, repeat_kv, Qwen3Cache, Qwen3Config,
};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};
use crate::models::shared::attention::paged::paged_decode_attention;

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
}

struct VoxtralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VoxtralLM {
    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        // For Voxtral, embeddings are provided by the audio projection + text projection
        // We create placeholder embeddings that will be overridden by custom embeddings
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                .unwrap_or_else(|_| {
                    candle_nn::embedding(
                        cfg.vocab_size,
                        cfg.hidden_size,
                        VarBuilder::zeros(DType::F32, vb.device()),
                    )
                    .unwrap()
                });

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = VoxtralLayer::load(&cfg, vb.pp(format!("layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
            .unwrap_or_else(|_| {
                candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    VarBuilder::zeros(DType::F32, vb.device()),
                )
                .unwrap()
            });

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

    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache, None)
    }

    pub fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            x = layer.forward(&x, start_pos, position_ids, cache_ref, idx)?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}

impl VoxtralLayer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attention_norm"))?;
        let self_attn = VoxtralAttention::load(cfg, vb.pp("attention"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;
        let mlp = VoxtralMlp::load(cfg, vb.pp("feed_forward"))?;
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
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out =
            self.self_attn
                .forward(&normed, start_pos, position_ids, cache, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
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
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
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

        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;

        q = self.apply_rope(q, start_pos, position_ids)?;
        k = self.apply_rope(k, start_pos, position_ids)?;

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k, v)?;

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

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let total_len = k.dim(2)?;
        if flash_attention_requested() && start_pos == 0 && total_len == seq_len {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, true)?
            {
                let out = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                let out = self.o_proj.forward(&out)?;
                return Ok(out);
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

        if seq_len > 1 || start_pos == 0 {
            let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        let out = self.o_proj.forward(&out)?;
        Ok(out)
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
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let heads = x.dim(2)?;
        let half_dim = self.head_dim / 2;

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

        let x = x.reshape((bsz, seq_len, heads, half_dim, 2))?;
        let x1 = x.narrow(4, 0, 1)?.squeeze(4)?;
        let x2 = x.narrow(4, 1, 1)?.squeeze(4)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rot1 = x1.broadcast_mul(&cos)?;
        let rot1 = rot1.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let rot2 = x1.broadcast_mul(&sin)?;
        let rot2 = rot2.broadcast_add(&x2.broadcast_mul(&cos)?)?;

        let rot1 = rot1.unsqueeze(4)?;
        let rot2 = rot2.unsqueeze(4)?;
        let out = Tensor::cat(&[rot1, rot2], 4)?;
        out.reshape((bsz, seq_len, heads, self.head_dim))
            .map_err(Error::from)
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
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        let out = self.down_proj.forward(&hidden)?;
        Ok(out)
    }
}

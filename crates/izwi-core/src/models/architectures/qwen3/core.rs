//! Minimal Qwen3 decoder implementation for native inference.
//!
//! Adapted from the Qwen3 architecture to allow embedding overrides
//! (used for audio-conditioned ASR).

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::kernels::cuda::{try_cuda_binary_activation, try_cuda_rotary_pair, CudaKernelKind};
use crate::kernels::metal::try_fused_silu_mul;
use crate::models::shared::attention::batched::{
    batched_scaled_dot_product_attention, BatchedAttentionConfig, BatchedAttentionInput,
};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::paged::{
    append_to_pages, default_kv_page_size, default_kv_quantization, materialize_pages,
    paged_decode_attention, KvCacheQuantization, KvPage,
};
use crate::models::shared::telemetry::{record_rope_kernel, record_rope_manual};
use crate::models::shared::weights::mlx;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub lm_head_size: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

pub struct Qwen3Cache {
    k_pages: Vec<Vec<KvPage>>,
    v_pages: Vec<Vec<KvPage>>,
    page_size: usize,
    quantization: KvCacheQuantization,
}

impl Qwen3Cache {
    pub fn new(num_layers: usize) -> Self {
        Self::with_page_size_and_quantization(
            num_layers,
            default_kv_page_size(),
            default_kv_quantization(),
        )
    }

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

    pub fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<()> {
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

    pub fn pages(&self, layer: usize) -> Option<(&[KvPage], &[KvPage])> {
        let k = self.k_pages.get(layer)?;
        let v = self.v_pages.get(layer)?;
        if k.is_empty() || v.is_empty() {
            None
        } else {
            Some((k.as_slice(), v.as_slice()))
        }
    }

    pub fn materialize(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let k = self.k_pages.get(layer).ok_or_else(|| {
            Error::InferenceError(format!("Invalid Qwen3Cache layer index: {layer}"))
        })?;
        let v = self.v_pages.get(layer).ok_or_else(|| {
            Error::InferenceError(format!("Invalid Qwen3Cache layer index: {layer}"))
        })?;
        Ok((materialize_pages(k)?, materialize_pages(v)?))
    }
}

struct Qwen3Attention {
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
    rope_inv_freqs: Vec<f32>,
    rope_kernel_enabled: bool,
}

impl Qwen3Attention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
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
            rope_inv_freqs: build_rope_inv_freqs(head_dim, cfg.rope_theta),
            rope_kernel_enabled: qwen3_rope_kernel_enabled(vb.device()),
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

    fn apply_rope_pair(
        &self,
        q: Tensor,
        k: Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        if k.dim(1)? != seq_len {
            return Err(Error::InferenceError(format!(
                "Qwen3 RoPE sequence mismatch: q_seq={}, k_seq={}",
                seq_len,
                k.dim(1)?
            )));
        }
        let q = q.contiguous()?;
        let k = k.contiguous()?;

        let (cos_half, sin_half) = if self.use_mrope {
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
                Tensor::from_vec(data, (3, seq_len), q.device())?
            };
            build_mrope_cache_with_inv_freqs(
                seq_len,
                self.head_dim,
                q.device(),
                q.dtype(),
                &position_ids,
                self.mrope_section.as_deref().unwrap_or(&[]),
                &self.rope_inv_freqs,
            )?
        } else {
            build_rope_cache_with_inv_freqs(
                seq_len,
                start_pos,
                q.device(),
                q.dtype(),
                &self.rope_inv_freqs,
            )?
        };

        let cos = cos_half.unsqueeze(0)?.contiguous()?;
        let sin = sin_half.unsqueeze(0)?.contiguous()?;
        if q.device().is_cuda() {
            let kind = if self.use_mrope {
                CudaKernelKind::MRope
            } else {
                CudaKernelKind::Rope
            };
            if let Some((q_out, k_out)) = try_cuda_rotary_pair(&q, &k, &cos, &sin, kind)? {
                record_rope_kernel();
                record_rope_kernel();
                return Ok((q_out, k_out));
            }
        }
        if self.should_try_rope_kernel(q.dtype()) {
            if let Some((q_out, k_out)) = try_apply_rope_pair_thd(&q, &k, &cos, &sin)? {
                record_rope_kernel();
                record_rope_kernel();
                return Ok((q_out, k_out));
            }
        }
        record_rope_manual();
        record_rope_manual();

        Ok((
            apply_rotary_emb(&q, &cos_half, &sin_half)?,
            apply_rotary_emb(&k, &cos_half, &sin_half)?,
        ))
    }

    fn should_try_rope_kernel(&self, dtype: DType) -> bool {
        if !self.rope_kernel_enabled {
            return false;
        }
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
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

        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;

        (q, k) = self.apply_rope_pair(q, k, start_pos, position_ids)?;

        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k.clone(), v.clone())?;

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
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        if use_batched {
            let q = q.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
            let k = k.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
            let v = v.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
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

        let q = q.transpose(1, 2)?; // [b, h, s, d]
        let k = k.transpose(1, 2)?; // [b, h, t, d]
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
        if start_pos == 0 && total_len == seq_len {
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

        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }
}

struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
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
        let hidden = if gate.device().is_cuda() {
            if let Some(fused) = try_cuda_binary_activation(&gate, &up, CudaKernelKind::SiluMul)? {
                fused
            } else if let Some(fused) = try_fused_silu_mul(&gate, &up) {
                fused
            } else {
                let act = ops::silu(&gate)?;
                act.broadcast_mul(&up)?
            }
        } else if let Some(fused) = try_fused_silu_mul(&gate, &up) {
            fused
        } else {
            let act = ops::silu(&gate)?;
            act.broadcast_mul(&up)?
        };
        let out = self.down_proj.forward(&hidden)?;
        Ok(out)
    }
}

struct Qwen3Layer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Qwen3Mlp,
}

impl Qwen3Layer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Qwen3Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Qwen3Mlp::load(cfg, vb.pp("mlp"))?;
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

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    cfg: Qwen3Config,
    use_mrope: bool,
}

impl Qwen3Model {
    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let lm_head_size = cfg.lm_head_size.unwrap_or(cfg.vocab_size);
        let embed_tokens =
            mlx::load_embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            mlx::load_linear_no_bias(cfg.hidden_size, lm_head_size, vb.pp("lm_head"))?
        } else {
            // Some MLX checkpoints omit lm_head and tie it to token embeddings.
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but lm_head.weight is missing",
                    cfg.vocab_size
                )));
            }
            Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Qwen3Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
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

    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut Qwen3Cache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache, None)
    }

    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
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

    pub fn uses_mrope(&self) -> bool {
        self.use_mrope
    }
}

pub fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidInput(format!(
            "Invalid GQA head config: num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )));
    }
    let repeats = num_heads / num_kv_heads;
    let seq_len = x.dim(1)?;
    let x = x.transpose(1, 2)?; // [b, kv, seq, d]
    let repeated = candle_repeat_kv(x, repeats)?;
    match repeated.rank() {
        4 => repeated.transpose(1, 2).map_err(Error::from),
        5 => {
            let (b, _kv_or_rep0, a2, a3, d) = repeated.dims5()?;
            let repeated = if a2 == seq_len && a3 == repeats {
                repeated
            } else if a3 == seq_len && a2 == repeats {
                repeated.transpose(2, 3)?
            } else {
                return Err(Error::InferenceError(format!(
                    "Unexpected repeat_kv rank-5 shape: {:?}, seq_len={seq_len}, repeats={repeats}",
                    repeated.dims()
                )));
            };
            let repeated = repeated.transpose(1, 2)?; // [b, seq, kv, repeats, d]
            repeated
                .reshape((b, seq_len, num_heads, d))
                .map_err(Error::from)
        }
        rank => Err(Error::InferenceError(format!(
            "Unexpected repeat_kv rank {rank}: {:?}",
            repeated.dims()
        ))),
    }
}

fn build_rope_inv_freqs(head_dim: usize, rope_theta: f64) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }
    inv_freq
}

fn build_rope_cache_with_inv_freqs(
    seq_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
    inv_freqs: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freqs.len();
    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for &inv in inv_freqs {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    Ok((
        angles.cos()?.to_dtype(dtype)?,
        angles.sin()?.to_dtype(dtype)?,
    ))
}

fn build_mrope_cache_with_inv_freqs(
    seq_len: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
    inv_freqs: &[f32],
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    if mrope_section.len() < 3 {
        return build_rope_cache_with_inv_freqs(seq_len, 0, device, dtype, inv_freqs);
    }

    let positions = position_ids.to_vec2::<i64>()?;
    if positions.len() != 3 || positions.iter().any(|axis| axis.len() < seq_len) {
        return build_rope_cache_with_inv_freqs(seq_len, 0, device, dtype, inv_freqs);
    }

    // Match Qwen3 Omni's interleaved MRoPE layout:
    // T,H,W,T,H,W,... for the first 3*section dims, then T for the tail.
    let h_limit = mrope_section[1].saturating_mul(3).min(half_dim);
    let w_limit = mrope_section[2].saturating_mul(3).min(half_dim);

    let mut cos_data = Vec::with_capacity(seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(seq_len * half_dim);
    for t in 0..seq_len {
        let p0 = positions[0][t] as f32;
        let p1 = positions[1][t] as f32;
        let p2 = positions[2][t] as f32;
        for (dim, &inv) in inv_freqs.iter().enumerate() {
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

    Ok((
        Tensor::from_vec(cos_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, (seq_len, half_dim), device)?.to_dtype(dtype)?,
    ))
}

pub fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let inv_freqs = build_rope_inv_freqs(head_dim, rope_theta);
    build_rope_cache_with_inv_freqs(seq_len, start_pos, device, dtype, &inv_freqs)
}

pub fn build_mrope_cache(
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let inv_freqs = build_rope_inv_freqs(head_dim, rope_theta);
    build_mrope_cache_with_inv_freqs(
        seq_len,
        head_dim,
        device,
        dtype,
        position_ids,
        mrope_section,
        &inv_freqs,
    )
}

pub fn causal_mask(
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

fn apply_rotary_emb(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
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

fn try_apply_rope_pair_thd(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let kernel_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q_out = rotary_emb::rope_thd(q, cos, sin)?;
        let k_out = rotary_emb::rope_thd(k, cos, sin)?;
        candle_core::Result::<(Tensor, Tensor)>::Ok((q_out, k_out))
    }));
    match kernel_result {
        Ok(Ok((q_out, k_out))) => Ok(Some((q_out, k_out))),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}

fn qwen3_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn qwen3_rope_kernel_enabled(device: &Device) -> bool {
    device.is_metal() && qwen3_env_bool("IZWI_QWEN3_ROPE_KERNEL", true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::rotary_emb;

    #[test]
    fn repeat_kv_returns_rank4_heads_last() {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 63, 16, 128), DType::F32, &device).expect("tensor");
        let out = repeat_kv(&input, 32, 16).expect("repeat_kv");
        assert_eq!(out.rank(), 4);
        assert_eq!(out.dims4().expect("dims4"), (1, 63, 32, 128));
    }

    #[test]
    fn qwen3_manual_rotary_matches_rope_thd() {
        let device = Device::Cpu;
        let seq_len = 4usize;
        let head_dim = 8usize;
        let x = Tensor::from_vec(
            (0..(seq_len * 2 * head_dim))
                .map(|v| (v as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, seq_len, 2, head_dim),
            &device,
        )
        .expect("x");

        let (cos_half, sin_half) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");
        let kernel = rotary_emb::rope_thd(
            &x,
            &cos_half.unsqueeze(0).expect("cos"),
            &sin_half.unsqueeze(0).expect("sin"),
        )
        .expect("kernel");

        let half_dim = head_dim / 2;
        let cos = Tensor::cat(&[cos_half.clone(), cos_half], 1)
            .expect("cat cos")
            .unsqueeze(0)
            .expect("unsqueeze cos")
            .unsqueeze(2)
            .expect("unsqueeze cos head");
        let sin = Tensor::cat(&[sin_half.clone(), sin_half], 1)
            .expect("cat sin")
            .unsqueeze(0)
            .expect("unsqueeze sin")
            .unsqueeze(2)
            .expect("unsqueeze sin head");

        let x1 = x.narrow(3, 0, half_dim).expect("x1");
        let x2 = x.narrow(3, half_dim, half_dim).expect("x2");
        let minus_one = Tensor::from_vec(vec![-1.0f32], (1,), &device).expect("minus one");
        let neg_x2 = x2.broadcast_mul(&minus_one).expect("neg");
        let rotated = Tensor::cat(&[neg_x2, x1], 3).expect("rotated");
        let manual = x
            .broadcast_mul(&cos)
            .expect("mul")
            .broadcast_add(&rotated.broadcast_mul(&sin).expect("rot mul"))
            .expect("add");

        let kernel_vals = kernel
            .flatten_all()
            .expect("flatten kernel")
            .to_vec1::<f32>()
            .expect("kernel vals");
        let manual_vals = manual
            .flatten_all()
            .expect("flatten manual")
            .to_vec1::<f32>()
            .expect("manual vals");
        assert_eq!(kernel_vals.len(), manual_vals.len());
        for (lhs, rhs) in kernel_vals.iter().zip(manual_vals.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn qwen3_apply_rotary_emb_accepts_half_dim_cos_sin() {
        let device = Device::Cpu;
        let seq_len = 3usize;
        let head_dim = 8usize;
        let x = Tensor::from_vec(
            (0..(seq_len * 2 * head_dim))
                .map(|v| (v as f32) * 0.03 - 0.2)
                .collect::<Vec<_>>(),
            (1, seq_len, 2, head_dim),
            &device,
        )
        .expect("x");
        let (cos_half, sin_half) =
            build_rope_cache(seq_len, head_dim, 0, 10000.0, &device, DType::F32).expect("cache");

        let rotated = apply_rotary_emb(&x, &cos_half, &sin_half).expect("rotary");
        assert_eq!(rotated.dims4().expect("dims4"), (1, seq_len, 2, head_dim));
    }
}

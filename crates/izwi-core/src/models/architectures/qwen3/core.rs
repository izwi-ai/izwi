//! Minimal Qwen3 decoder implementation for native inference.
//!
//! Adapted from the Qwen3 architecture to allow embedding overrides
//! (used for audio-conditioned ASR).

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{ops, rotary_emb, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernels::try_fused_silu_mul;
use crate::models::shared::attention::batched::{
    batched_scaled_dot_product_attention, BatchedAttentionConfig, BatchedAttentionInput,
};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::paged::{
    append_to_pages, default_kv_page_size, default_kv_quantization, materialize_pages,
    paged_decode_attention, KvCacheQuantization, KvPage,
};
use crate::models::shared::telemetry::{
    record_decode_attention_path, record_rope_kernel, record_rope_manual, DecodeAttentionPath,
};
use crate::models::shared::weights::gguf::GgufLoader;
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
    dense_k_cache_h: Vec<Option<Tensor>>,
    dense_v_cache_h: Vec<Option<Tensor>>,
    dense_kv_tokens: Vec<usize>,
    dense_decode_overflowed: Vec<bool>,
    dense_decode_max_tokens: usize,
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

    pub fn with_page_size_and_dense_decode(
        num_layers: usize,
        page_size: usize,
        device: &Device,
    ) -> Self {
        let quantization = default_kv_quantization();
        let dense_decode_max_tokens =
            qwen3_dense_decode_max_tokens(device, page_size, quantization);
        Self::with_page_size_quantization_and_dense_decode_tokens(
            num_layers,
            page_size,
            quantization,
            dense_decode_max_tokens,
        )
    }

    pub fn with_page_size_and_quantization(
        num_layers: usize,
        page_size: usize,
        quantization: KvCacheQuantization,
    ) -> Self {
        Self::with_page_size_quantization_and_dense_decode_tokens(
            num_layers,
            page_size,
            quantization,
            0,
        )
    }

    pub fn with_page_size_quantization_and_dense_decode_tokens(
        num_layers: usize,
        page_size: usize,
        quantization: KvCacheQuantization,
        dense_decode_max_tokens: usize,
    ) -> Self {
        let dense_decode_max_tokens = if quantization == KvCacheQuantization::None {
            dense_decode_max_tokens
        } else {
            0
        };
        Self {
            k_pages: vec![Vec::new(); num_layers],
            v_pages: vec![Vec::new(); num_layers],
            dense_k_cache_h: vec![None; num_layers],
            dense_v_cache_h: vec![None; num_layers],
            dense_kv_tokens: vec![0; num_layers],
            dense_decode_overflowed: vec![false; num_layers],
            dense_decode_max_tokens,
            page_size: page_size.max(1),
            quantization,
        }
    }

    pub fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<()> {
        if self.should_append_dense(layer, k.dim(1)?) {
            self.append_dense(layer, &k, &v)?;
            return Ok(());
        }
        if self.has_dense_cache(layer) {
            self.materialize_dense_to_pages(layer)?;
            self.dense_decode_overflowed[layer] = true;
        }
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

    fn should_append_dense(&self, layer: usize, append_tokens: usize) -> bool {
        self.dense_decode_max_tokens > 0
            && !self.dense_decode_overflowed[layer]
            && self.k_pages[layer].is_empty()
            && self.v_pages[layer].is_empty()
            && self.dense_kv_tokens[layer].saturating_add(append_tokens)
                <= self.dense_decode_max_tokens
    }

    fn append_dense(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let append_tokens = k.dim(1)?;
        if append_tokens == 0 {
            return Ok(());
        }

        let k_h = k.transpose(1, 2)?.contiguous()?;
        let v_h = v.transpose(1, 2)?.contiguous()?;
        append_dense_kv_cache_h(&mut self.dense_k_cache_h[layer], &k_h)?;
        append_dense_kv_cache_h(&mut self.dense_v_cache_h[layer], &v_h)?;
        self.dense_kv_tokens[layer] = self.dense_kv_tokens[layer].saturating_add(append_tokens);
        Ok(())
    }

    fn has_dense_cache(&self, layer: usize) -> bool {
        self.dense_kv_tokens[layer] > 0
            && self.dense_k_cache_h[layer].is_some()
            && self.dense_v_cache_h[layer].is_some()
    }

    fn materialize_dense_to_pages(&mut self, layer: usize) -> Result<()> {
        if !self.has_dense_cache(layer) {
            return Ok(());
        }
        let k_cache_h = self.dense_k_cache_h[layer].take().ok_or_else(|| {
            Error::InferenceError("Qwen3 missing dense key cache during page migration".to_string())
        })?;
        let v_cache_h = self.dense_v_cache_h[layer].take().ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 missing dense value cache during page migration".to_string(),
            )
        })?;
        let k_dense = k_cache_h.transpose(1, 2)?.contiguous()?;
        let v_dense = v_cache_h.transpose(1, 2)?.contiguous()?;
        append_to_pages(
            self.page_size,
            &mut self.k_pages[layer],
            &k_dense,
            self.quantization,
        )?;
        append_to_pages(
            self.page_size,
            &mut self.v_pages[layer],
            &v_dense,
            self.quantization,
        )?;
        self.dense_kv_tokens[layer] = 0;
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
        if k.is_empty() && v.is_empty() {
            if let Some((k_dense_h, v_dense_h)) = self.dense_heads(layer) {
                return Ok((
                    k_dense_h.transpose(1, 2)?.contiguous()?,
                    v_dense_h.transpose(1, 2)?.contiguous()?,
                ));
            }
        }
        Ok((materialize_pages(k)?, materialize_pages(v)?))
    }

    pub fn dense_heads(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        if self.dense_decode_max_tokens == 0 || self.dense_decode_overflowed[layer] {
            return None;
        }
        let k = self.dense_k_cache_h.get(layer)?.as_ref()?;
        let v = self.dense_v_cache_h.get(layer)?.as_ref()?;
        Some((k, v))
    }
}

fn append_dense_kv_cache_h(cache: &mut Option<Tensor>, append: &Tensor) -> Result<()> {
    if append.dim(2)? == 0 {
        return Ok(());
    }
    let append = append.contiguous()?;
    match cache {
        Some(existing) => {
            let existing_ref: &Tensor = &*existing;
            *cache = Some(Tensor::cat(&[existing_ref, &append], 2)?);
        }
        None => {
            *cache = Some(append);
        }
    }
    Ok(())
}

fn qwen3_dense_decode_max_tokens(
    device: &Device,
    page_size: usize,
    quantization: KvCacheQuantization,
) -> usize {
    if quantization != KvCacheQuantization::None {
        return 0;
    }
    if !qwen3_use_dense_decode_attention_feature(device) {
        return 0;
    }
    page_size
        .max(1)
        .saturating_mul(qwen3_dense_decode_max_pages())
}

fn qwen3_use_dense_decode_attention_feature(device: &Device) -> bool {
    if !device.is_metal() {
        return false;
    }
    std::env::var("IZWI_QWEN3_DENSE_DECODE_ATTENTION")
        .ok()
        .and_then(|raw| parse_env_bool(&raw))
        .unwrap_or(true)
}

fn qwen3_dense_decode_max_pages() -> usize {
    std::env::var("IZWI_QWEN3_DENSE_DECODE_MAX_PAGES")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128)
}

fn parse_env_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

enum Qwen3Projection {
    Dense(Linear),
    Quantized(QMatMul),
}

impl Qwen3Projection {
    fn dense(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        mlx::load_linear_no_bias(in_dim, out_dim, vb)
            .map(Self::Dense)
            .map_err(Error::from)
    }

    fn quantized(loader: &GgufLoader, device: &Device, name: &str) -> Result<Self> {
        let weights = Arc::new(loader.load_qtensor(name, device)?);
        QMatMul::from_arc(weights)
            .map(Self::Quantized)
            .map_err(Error::from)
    }

    fn tied_dense(weight: Tensor) -> Self {
        Self::Dense(Linear::new(weight, None))
    }
}

impl Module for Qwen3Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::Quantized(qmatmul) => qmatmul.forward(x),
        }
    }
}

fn load_optional_rms_norm_from_gguf(
    loader: &GgufLoader,
    device: &Device,
    dtype: DType,
    name: &str,
    eps: f64,
) -> Result<Option<RmsNorm>> {
    if !loader.has_tensor(name) {
        return Ok(None);
    }
    let weight = loader.load_tensor(name, dtype, device)?;
    Ok(Some(RmsNorm::new(weight, eps)))
}

struct Qwen3Attention {
    q_proj: Qwen3Projection,
    k_proj: Qwen3Projection,
    v_proj: Qwen3Projection,
    o_proj: Qwen3Projection,
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
        let q_proj = Qwen3Projection::dense(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = Qwen3Projection::dense(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = Qwen3Projection::dense(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = Qwen3Projection::dense(
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

    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let q_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.q_proj.weight"))?;
        let k_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.k_proj.weight"))?;
        let v_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.v_proj.weight"))?;
        let o_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.o_proj.weight"))?;

        let q_norm = load_optional_rms_norm_from_gguf(
            loader,
            device,
            dtype,
            &format!("{prefix}.q_norm.weight"),
            cfg.rms_norm_eps,
        )?;
        let k_norm = load_optional_rms_norm_from_gguf(
            loader,
            device,
            dtype,
            &format!("{prefix}.k_norm.weight"),
            cfg.rms_norm_eps,
        )?;
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
            rope_kernel_enabled: qwen3_rope_kernel_enabled(device),
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

        if self.should_try_rope_kernel(q.dtype()) {
            let cos = cos_half.unsqueeze(0)?.contiguous()?;
            let sin = sin_half.unsqueeze(0)?.contiguous()?;
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
                if let Some((k_dense_h, v_dense_h)) = cache.dense_heads(layer_idx) {
                    record_decode_attention_path(DecodeAttentionPath::Dense);
                    let out = dense_decode_attention(
                        &q,
                        k_dense_h,
                        v_dense_h,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                    )?;
                    let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
                    return self.o_proj.forward(&out).map_err(Error::from);
                }
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
            let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
            let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;
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
        let k_kv = k.transpose(1, 2)?; // [b, kv_h, t, d]
        let v_kv = v.transpose(1, 2)?;

        let total_len = k_kv.dim(2)?;
        if start_pos == 0 && total_len == seq_len {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k_kv, &v_kv, None, self.head_dim, true)?
            {
                let out = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&out).map_err(Error::from);
            }
        }

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;
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
    gate_proj: Qwen3Projection,
    up_proj: Qwen3Projection,
    down_proj: Qwen3Projection,
}

impl Qwen3Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            Qwen3Projection::dense(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            Qwen3Projection::dense(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            Qwen3Projection::dense(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn load_gguf(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        let gate_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.gate_proj.weight"))?;
        let up_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.up_proj.weight"))?;
        let down_proj =
            Qwen3Projection::quantized(loader, device, &format!("{prefix}.down_proj.weight"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = if let Some(fused) = try_fused_silu_mul(&gate, &up) {
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

    fn load_gguf(
        loader: &GgufLoader,
        cfg: &Qwen3Config,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::new(
            loader.load_tensor(&format!("{prefix}.input_layernorm.weight"), dtype, device)?,
            cfg.rms_norm_eps,
        );
        let self_attn =
            Qwen3Attention::load_gguf(loader, cfg, device, dtype, &format!("{prefix}.self_attn"))?;
        let post_attention_layernorm = RmsNorm::new(
            loader.load_tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                dtype,
                device,
            )?,
            cfg.rms_norm_eps,
        );
        let mlp = Qwen3Mlp::load_gguf(loader, device, &format!("{prefix}.mlp"))?;
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
    lm_head: Qwen3Projection,
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
            Qwen3Projection::dense(cfg.hidden_size, lm_head_size, vb.pp("lm_head"))?
        } else {
            // Some MLX checkpoints omit lm_head and tie it to token embeddings.
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but lm_head.weight is missing",
                    cfg.vocab_size
                )));
            }
            Qwen3Projection::tied_dense(embed_tokens.embeddings().clone())
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

    pub fn load_gguf_text(
        cfg: Qwen3Config,
        loader: &GgufLoader,
        device: &Device,
        dtype: DType,
        prefix: &str,
    ) -> Result<Self> {
        let lm_head_size = cfg.lm_head_size.unwrap_or(cfg.vocab_size);
        let model_prefix = qwen3_join_prefix(prefix, "model");
        let embed_tokens = Embedding::new(
            loader.load_tensor(
                &qwen3_join_prefix(&model_prefix, "embed_tokens.weight"),
                dtype,
                device,
            )?,
            cfg.hidden_size,
        );
        let lm_head_name = qwen3_join_prefix(prefix, "lm_head.weight");
        let lm_head = if loader.has_tensor(&lm_head_name) {
            Qwen3Projection::quantized(loader, device, &lm_head_name)?
        } else {
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but lm_head.weight is missing",
                    cfg.vocab_size
                )));
            }
            Qwen3Projection::tied_dense(embed_tokens.embeddings().clone())
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer_prefix = qwen3_join_prefix(&model_prefix, &format!("layers.{idx}"));
            let layer = Qwen3Layer::load_gguf(loader, &cfg, device, dtype, &layer_prefix)?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(
            loader.load_tensor(
                &qwen3_join_prefix(&model_prefix, "norm.weight"),
                dtype,
                device,
            )?,
            cfg.rms_norm_eps,
        );
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
            device: device.clone(),
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

fn qwen3_join_prefix(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

fn dense_decode_attention(
    q: &Tensor,
    k_heads: &Tensor,
    v_heads: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let bsz = q.dim(0)?;
    let seq_len = q.dim(1)?;
    if seq_len != 1 {
        return Err(Error::InvalidInput(format!(
            "Qwen3 dense decode attention expects q_len=1, got {seq_len}"
        )));
    }

    let q_heads = q.transpose(1, 2)?.contiguous()?;
    if let Some(out) = try_fused_self_attention(&q_heads, k_heads, v_heads, None, head_dim, true)? {
        return out.transpose(1, 2).map_err(Error::from);
    }

    let k = k_heads.transpose(1, 2)?.contiguous()?;
    let v = v_heads.transpose(1, 2)?.contiguous()?;
    let k = repeat_kv(&k, num_heads, num_kv_heads)?;
    let v = repeat_kv(&v, num_heads, num_kv_heads)?;
    let k_heads = k.transpose(1, 2)?.contiguous()?;
    let v_heads = v.transpose(1, 2)?.contiguous()?;
    let k_heads_t = k_heads.transpose(2, 3)?.contiguous()?;
    let mut att = q_heads.matmul(&k_heads_t)?;
    att = (att / (head_dim as f64).sqrt())?;
    let att = ops::softmax(&att, D::Minus1)?;
    let out = att.matmul(&v_heads)?;
    out.transpose(1, 2)?
        .reshape((bsz, seq_len, num_heads, head_dim))
        .map_err(Error::from)
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

    #[test]
    fn qwen3_cache_dense_decode_accumulates_head_major_kv() {
        let device = Device::Cpu;
        let mut cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            4,
            KvCacheQuantization::None,
            4,
        );
        let k = Tensor::from_vec(
            (0..8).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 2, 1, 4),
            &device,
        )
        .expect("k");
        let v = Tensor::from_vec(
            (0..8).map(|value| (value as f32) * 0.5).collect::<Vec<_>>(),
            (1, 2, 1, 4),
            &device,
        )
        .expect("v");

        cache.append(0, k, v).expect("append");
        let (k_dense, v_dense) = cache.dense_heads(0).expect("dense heads");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 2, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 1, 2, 4));
        assert!(cache.pages(0).is_none());

        let k_next =
            Tensor::from_vec(vec![8.0f32, 9.0, 10.0, 11.0], (1, 1, 1, 4), &device).expect("k next");
        let v_next =
            Tensor::from_vec(vec![4.0f32, 4.5, 5.0, 5.5], (1, 1, 1, 4), &device).expect("v next");
        cache.append(0, k_next, v_next).expect("append next");
        let (k_dense, _) = cache.dense_heads(0).expect("dense heads after append");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 3, 4));
    }

    #[test]
    fn qwen3_cache_dense_decode_disables_after_threshold_or_quantization() {
        let device = Device::Cpu;
        let k = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("k");
        let v = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("v");
        let mut cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            4,
            KvCacheQuantization::None,
            2,
        );
        cache
            .append(0, k.clone(), v.clone())
            .expect("append beyond threshold");
        assert!(cache.dense_heads(0).is_none());

        let mut quantized_cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            4,
            KvCacheQuantization::Int8,
            8,
        );
        quantized_cache.append(0, k, v).expect("append quantized");
        assert!(quantized_cache.dense_heads(0).is_none());
    }

    #[test]
    fn qwen3_cache_dense_decode_migrates_to_pages_after_threshold() {
        let device = Device::Cpu;
        let mut cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            2,
            KvCacheQuantization::None,
            2,
        );
        let first_k = Tensor::from_vec(
            (0..8).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 2, 1, 4),
            &device,
        )
        .expect("first k");
        let first_v = Tensor::from_vec(
            (0..8).map(|value| value as f32 + 20.0).collect::<Vec<_>>(),
            (1, 2, 1, 4),
            &device,
        )
        .expect("first v");
        cache.append(0, first_k, first_v).expect("append first");
        assert!(cache.dense_heads(0).is_some());
        assert!(cache.pages(0).is_none());

        let next_k =
            Tensor::from_vec(vec![8.0f32, 9.0, 10.0, 11.0], (1, 1, 1, 4), &device).expect("next k");
        let next_v = Tensor::from_vec(vec![28.0f32, 29.0, 30.0, 31.0], (1, 1, 1, 4), &device)
            .expect("next v");
        cache.append(0, next_k, next_v).expect("append next");

        assert!(cache.dense_heads(0).is_none());
        let (k_pages, v_pages) = cache.pages(0).expect("pages after migration");
        assert_eq!(k_pages.len(), 2);
        assert_eq!(v_pages.len(), 2);
        let (k_dense, v_dense) = cache.materialize(0).expect("materialized");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 3, 1, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 3, 1, 4));
    }

    #[test]
    fn dense_decode_attention_matches_manual_gqa() {
        let device = Device::Cpu;
        let q = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.3, 0.4, -0.5, 0.6, 0.7, -0.2],
            (1, 1, 2, 4),
            &device,
        )
        .expect("q");
        let k = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.1, 0.3, -0.2, 0.5, 0.4, 0.1],
            (1, 2, 1, 4),
            &device,
        )
        .expect("k");
        let v = Tensor::from_vec(
            vec![0.7f32, -0.3, 0.2, 0.5, 0.1, 0.4, -0.6, 0.8],
            (1, 2, 1, 4),
            &device,
        )
        .expect("v");
        let k_heads = k
            .transpose(1, 2)
            .expect("k heads")
            .contiguous()
            .expect("k contig");
        let v_heads = v
            .transpose(1, 2)
            .expect("v heads")
            .contiguous()
            .expect("v contig");

        let dense = dense_decode_attention(&q, &k_heads, &v_heads, 2, 1, 4).expect("dense");
        let q_heads = q.transpose(1, 2).expect("q heads");
        let k_repeated = repeat_kv(&k, 2, 1).expect("repeat k");
        let v_repeated = repeat_kv(&v, 2, 1).expect("repeat v");
        let k_repeated = k_repeated.transpose(1, 2).expect("k repeated heads");
        let v_repeated = v_repeated.transpose(1, 2).expect("v repeated heads");
        let mut att = q_heads
            .matmul(&k_repeated.transpose(2, 3).expect("k t"))
            .expect("scores");
        att = (att / 2.0f64).expect("scale");
        let expected = ops::softmax(&att, D::Minus1)
            .expect("softmax")
            .matmul(&v_repeated)
            .expect("expected")
            .transpose(1, 2)
            .expect("expected transpose");

        let dense_vals = dense
            .flatten_all()
            .expect("dense flat")
            .to_vec1::<f32>()
            .expect("dense vals");
        let expected_vals = expected
            .flatten_all()
            .expect("expected flat")
            .to_vec1::<f32>()
            .expect("expected vals");
        assert_eq!(dense_vals.len(), expected_vals.len());
        for (lhs, rhs) in dense_vals.iter().zip(expected_vals.iter()) {
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} != {rhs}");
        }
    }
}

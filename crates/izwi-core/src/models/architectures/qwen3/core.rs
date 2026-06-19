//! Minimal Qwen3 decoder implementation for native inference.
//!
//! Adapted from the Qwen3 architecture to allow embedding overrides
//! (used for audio-conditioned ASR).

use candle_core::quantized::QMatMul;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, kv_cache::Cache, ops, rotary_emb};
use candle_transformers::utils::repeat_kv as candle_repeat_kv;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernels::try_fused_silu_mul;
use crate::models::shared::attention::batched::{
    BatchedAttentionConfig, BatchedAttentionInput, batched_scaled_dot_product_attention,
};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::paged::{
    KvCacheQuantization, KvPage, append_to_pages, default_kv_page_size, default_kv_quantization,
    materialize_pages, paged_decode_attention,
};
use crate::models::shared::telemetry::{
    DecodeAttentionPath, record_decode_attention_path, record_rope_kernel, record_rope_manual,
};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::mlx;
use std::time::Instant;

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
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond_dim: usize,
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn sliding_window(&self) -> Option<usize> {
        self.use_sliding_window
            .then_some(self.sliding_window)
            .flatten()
            .filter(|window| *window > 0)
    }
}

pub struct Qwen3Cache {
    k_pages: Vec<Vec<KvPage>>,
    v_pages: Vec<Vec<KvPage>>,
    dense_k_cache_h: Vec<Option<Cache>>,
    dense_v_cache_h: Vec<Option<Cache>>,
    dense_kv_tokens: Vec<usize>,
    dense_decode_overflowed: Vec<bool>,
    rope_cache: Vec<Qwen3RopeCacheEntry>,
    dense_decode_max_tokens: usize,
    dense_decode_initial_capacity: usize,
    page_size: usize,
    quantization: KvCacheQuantization,
}

struct Qwen3RopeCacheEntry {
    seq_len: usize,
    start_pos: usize,
    head_dim: usize,
    dtype: DType,
    use_mrope: bool,
    mrope_section: Vec<usize>,
    cos_half: Tensor,
    sin_half: Tensor,
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

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn dense_decode_max_tokens(&self) -> usize {
        self.dense_decode_max_tokens
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

    pub fn with_page_size_and_dense_decode_initial_capacity(
        num_layers: usize,
        page_size: usize,
        device: &Device,
        dense_decode_initial_capacity: usize,
    ) -> Self {
        let quantization = default_kv_quantization();
        let dense_decode_max_tokens =
            qwen3_dense_decode_max_tokens(device, page_size, quantization);
        Self::with_page_size_quantization_dense_decode_tokens_initial_capacity(
            num_layers,
            page_size,
            quantization,
            dense_decode_max_tokens,
            dense_decode_initial_capacity,
        )
    }

    pub fn with_page_size_and_dense_decode_tokens(
        num_layers: usize,
        page_size: usize,
        device: &Device,
        dense_decode_max_tokens: usize,
    ) -> Self {
        let quantization = default_kv_quantization();
        let dense_decode_max_tokens = if qwen3_use_dense_decode_attention_feature(device) {
            dense_decode_max_tokens
        } else {
            0
        };
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
        Self::with_page_size_quantization_dense_decode_tokens_initial_capacity(
            num_layers,
            page_size,
            quantization,
            dense_decode_max_tokens,
            0,
        )
    }

    fn with_page_size_quantization_dense_decode_tokens_initial_capacity(
        num_layers: usize,
        page_size: usize,
        quantization: KvCacheQuantization,
        dense_decode_max_tokens: usize,
        dense_decode_initial_capacity: usize,
    ) -> Self {
        let dense_decode_max_tokens = if quantization == KvCacheQuantization::None {
            dense_decode_max_tokens
        } else {
            0
        };
        let dense_decode_initial_capacity = if dense_decode_max_tokens > 0 {
            dense_decode_initial_capacity.min(dense_decode_max_tokens)
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
            rope_cache: Vec::new(),
            dense_decode_max_tokens,
            dense_decode_initial_capacity,
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
        let initial_capacity = dense_cache_initial_capacity(
            append_tokens,
            self.page_size,
            self.dense_decode_max_tokens,
            self.dense_decode_initial_capacity,
        );
        self.dense_k_cache_h[layer]
            .get_or_insert_with(|| Cache::new(2, initial_capacity))
            .append(&k_h)?;
        self.dense_v_cache_h[layer]
            .get_or_insert_with(|| Cache::new(2, initial_capacity))
            .append(&v_h)?;
        self.dense_kv_tokens[layer] = self.dense_kv_tokens[layer].saturating_add(append_tokens);
        Ok(())
    }

    fn has_dense_cache(&self, layer: usize) -> bool {
        self.dense_kv_tokens[layer] > 0
    }

    fn materialize_dense_to_pages(&mut self, layer: usize) -> Result<()> {
        if !self.has_dense_cache(layer) {
            return Ok(());
        }
        let k_cache = self.dense_k_cache_h[layer].take().ok_or_else(|| {
            Error::InferenceError("Qwen3 missing dense key cache during page migration".to_string())
        })?;
        let v_cache = self.dense_v_cache_h[layer].take().ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 missing dense value cache during page migration".to_string(),
            )
        })?;
        let k_cache_h = k_cache.current_data()?.ok_or_else(|| {
            Error::InferenceError("Qwen3 empty dense key cache during page migration".to_string())
        })?;
        let v_cache_h = v_cache.current_data()?.ok_or_else(|| {
            Error::InferenceError("Qwen3 empty dense value cache during page migration".to_string())
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
            if let Some((k_dense_h, v_dense_h)) = self.dense_heads(layer)? {
                return Ok((
                    k_dense_h.transpose(1, 2)?.contiguous()?,
                    v_dense_h.transpose(1, 2)?.contiguous()?,
                ));
            }
        }
        Ok((materialize_pages(k)?, materialize_pages(v)?))
    }

    pub fn dense_heads(&self, layer: usize) -> Result<Option<(Tensor, Tensor)>> {
        if self.dense_decode_max_tokens == 0 || self.dense_decode_overflowed[layer] {
            return Ok(None);
        }
        let Some(k_cache) = self
            .dense_k_cache_h
            .get(layer)
            .and_then(|cache| cache.as_ref())
        else {
            return Ok(None);
        };
        let Some(v_cache) = self
            .dense_v_cache_h
            .get(layer)
            .and_then(|cache| cache.as_ref())
        else {
            return Ok(None);
        };
        let Some(k) = k_cache.current_data()? else {
            return Ok(None);
        };
        let Some(v) = v_cache.current_data()? else {
            return Ok(None);
        };
        Ok(Some((k, v)))
    }

    fn cached_rope_pair(
        &mut self,
        seq_len: usize,
        start_pos: usize,
        head_dim: usize,
        device: &Device,
        dtype: DType,
        use_mrope: bool,
        mrope_section: &[usize],
        inv_freqs: &[f32],
    ) -> Result<(Tensor, Tensor)> {
        if let Some(entry) = self.rope_cache.iter().find(|entry| {
            entry.seq_len == seq_len
                && entry.start_pos == start_pos
                && entry.head_dim == head_dim
                && entry.dtype == dtype
                && entry.use_mrope == use_mrope
                && entry.mrope_section == mrope_section
        }) {
            return Ok((entry.cos_half.clone(), entry.sin_half.clone()));
        }

        let (cos_half, sin_half) = if use_mrope {
            let mut data = Vec::with_capacity(3 * seq_len);
            let base = start_pos as i64;
            for _axis in 0..3 {
                for idx in 0..seq_len {
                    data.push(base + idx as i64);
                }
            }
            let position_ids = Tensor::from_vec(data, (3, seq_len), device)?;
            build_mrope_cache_with_inv_freqs(
                seq_len,
                head_dim,
                device,
                dtype,
                &position_ids,
                mrope_section,
                inv_freqs,
            )?
        } else {
            build_rope_cache_with_inv_freqs(seq_len, start_pos, device, dtype, inv_freqs)?
        };

        self.rope_cache.push(Qwen3RopeCacheEntry {
            seq_len,
            start_pos,
            head_dim,
            dtype,
            use_mrope,
            mrope_section: mrope_section.to_vec(),
            cos_half: cos_half.clone(),
            sin_half: sin_half.clone(),
        });
        Ok((cos_half, sin_half))
    }
}

fn dense_cache_initial_capacity(
    append_tokens: usize,
    page_size: usize,
    dense_decode_max_tokens: usize,
    requested_initial_capacity: usize,
) -> usize {
    let page_size = page_size.max(1);
    requested_initial_capacity
        .max(append_tokens)
        .saturating_add(page_size - 1)
        .saturating_div(page_size)
        .saturating_mul(page_size)
        .max(append_tokens)
        .max(1)
        .min(dense_decode_max_tokens.max(1))
}

pub(crate) fn qwen3_dense_decode_max_tokens(
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
    let override_enabled = std::env::var("IZWI_QWEN3_DENSE_DECODE_ATTENTION")
        .ok()
        .and_then(|raw| parse_env_bool(&raw));
    qwen3_use_dense_decode_attention_policy(device.is_metal(), device.is_cuda(), override_enabled)
}

fn qwen3_use_dense_decode_attention_policy(
    is_metal: bool,
    is_cuda: bool,
    override_enabled: Option<bool>,
) -> bool {
    if !is_metal && !is_cuda {
        return false;
    }
    override_enabled.unwrap_or(true)
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

struct Qwen3DenseProjection {
    linear: Linear,
    has_bias: bool,
}

enum Qwen3Projection {
    Dense(Qwen3DenseProjection),
    Quantized(QMatMul),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3ProjectionDiagnostics {
    pub dense_projection_count: usize,
    pub dense_bias_projection_count: usize,
    pub quantized_projection_count: usize,
}

impl Qwen3ProjectionDiagnostics {
    fn add(self, other: Self) -> Self {
        Self {
            dense_projection_count: self
                .dense_projection_count
                .saturating_add(other.dense_projection_count),
            dense_bias_projection_count: self
                .dense_bias_projection_count
                .saturating_add(other.dense_bias_projection_count),
            quantized_projection_count: self
                .quantized_projection_count
                .saturating_add(other.quantized_projection_count),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Qwen3ForwardDiagnostics {
    pub input_norm_ms: f64,
    pub attention_ms: f64,
    pub attention_residual_ms: f64,
    pub post_attention_norm_ms: f64,
    pub mlp_ms: f64,
    pub mlp_residual_ms: f64,
    pub layers_total_ms: f64,
    pub final_norm_ms: f64,
    pub lm_head_ms: f64,
}

fn qwen3_elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

impl Qwen3Projection {
    fn dense(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let has_bias = vb.contains_tensor("bias");
        let linear = if has_bias {
            mlx::load_linear(in_dim, out_dim, vb)
        } else {
            mlx::load_linear_no_bias(in_dim, out_dim, vb)
        }?;
        Ok(Self::Dense(Qwen3DenseProjection { linear, has_bias }))
    }

    fn quantized(loader: &GgufLoader, device: &Device, name: &str) -> Result<Self> {
        let weights = Arc::new(loader.load_qtensor(name, device)?);
        QMatMul::from_arc(weights)
            .map(Self::Quantized)
            .map_err(Error::from)
    }

    fn tied_dense(weight: Tensor) -> Self {
        Self::Dense(Qwen3DenseProjection {
            linear: Linear::new(weight, None),
            has_bias: false,
        })
    }

    fn diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        match self {
            Self::Dense(dense) => Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: usize::from(dense.has_bias),
                quantized_projection_count: 0,
            },
            Self::Quantized(_) => Qwen3ProjectionDiagnostics {
                dense_projection_count: 0,
                dense_bias_projection_count: 0,
                quantized_projection_count: 1,
            },
        }
    }

    fn logits_for_tokens(&self, hidden: &Tensor, token_ids: &[u32]) -> Result<Vec<(u32, f32)>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let logits = match self {
            Self::Dense(dense) => dense_logits_for_tokens(dense, hidden, token_ids)?,
            Self::Quantized(_) => logits_for_tokens_from_full(&self.forward(hidden)?, token_ids)?,
        };
        let values = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(token_ids.iter().copied().zip(values).collect())
    }
}

fn qwen3_cuda_qmatmul_input_dtype(device_is_cuda: bool, activation_dtype: DType) -> Option<DType> {
    (device_is_cuda && activation_dtype != DType::F32).then_some(DType::F32)
}

impl Module for Qwen3Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(dense) => dense.linear.forward(x),
            Self::Quantized(qmatmul) => {
                let activation_dtype = x.dtype();
                let x = if let Some(dtype) =
                    qwen3_cuda_qmatmul_input_dtype(x.device().is_cuda(), activation_dtype)
                {
                    x.to_dtype(dtype)?
                } else {
                    x.clone()
                };
                let out = qmatmul.forward(&x)?;
                if out.dtype() != activation_dtype {
                    out.to_dtype(activation_dtype)
                } else {
                    Ok(out)
                }
            }
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

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.q_proj
            .diagnostics()
            .add(self.k_proj.diagnostics())
            .add(self.v_proj.diagnostics())
            .add(self.o_proj.diagnostics())
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
        cache: Option<&mut Qwen3Cache>,
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

        let (cos_half, sin_half) = if let Some(position_ids) = position_ids {
            if self.use_mrope {
                build_mrope_cache_with_inv_freqs(
                    seq_len,
                    self.head_dim,
                    q.device(),
                    q.dtype(),
                    position_ids,
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
            }
        } else if let Some(cache) = cache {
            cache.cached_rope_pair(
                seq_len,
                start_pos,
                self.head_dim,
                q.device(),
                q.dtype(),
                self.use_mrope,
                self.mrope_section.as_deref().unwrap_or(&[]),
                &self.rope_inv_freqs,
            )?
        } else if self.use_mrope {
            let mut data = Vec::with_capacity(3 * seq_len);
            let base = start_pos as i64;
            for _axis in 0..3 {
                for idx in 0..seq_len {
                    data.push(base + idx as i64);
                }
            }
            let position_ids = Tensor::from_vec(data, (3, seq_len), q.device())?;
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
        mut cache: Option<&mut Qwen3Cache>,
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

        (q, k) = self.apply_rope_pair(q, k, start_pos, position_ids, cache.as_deref_mut())?;

        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k.clone(), v.clone())?;

            // Decode path hot loop: for single-token decode, avoid rematerializing full KV.
            if seq_len == 1 && start_pos > 0 {
                if let Some((k_dense_h, v_dense_h)) = cache.dense_heads(layer_idx)? {
                    record_decode_attention_path(DecodeAttentionPath::Dense);
                    let out = dense_decode_attention(
                        &q,
                        &k_dense_h,
                        &v_dense_h,
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

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.gate_proj
            .diagnostics()
            .add(self.up_proj.diagnostics())
            .add(self.down_proj.diagnostics())
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

    fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.self_attn
            .projection_diagnostics()
            .add(self.mlp.projection_diagnostics())
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        self.forward_inner(x, start_pos, position_ids, cache, layer_idx, None)
    }

    fn forward_with_diagnostics(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
        diagnostics: &mut Qwen3ForwardDiagnostics,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            start_pos,
            position_ids,
            cache,
            layer_idx,
            Some(diagnostics),
        )
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        cache: Option<&mut Qwen3Cache>,
        layer_idx: usize,
        mut diagnostics: Option<&mut Qwen3ForwardDiagnostics>,
    ) -> Result<Tensor> {
        let norm_started = diagnostics.as_ref().map(|_| Instant::now());
        let normed = self.input_layernorm.forward(x)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), norm_started) {
            diag.input_norm_ms += qwen3_elapsed_ms(started);
        }

        let attention_started = diagnostics.as_ref().map(|_| Instant::now());
        let attn_out =
            self.self_attn
                .forward(&normed, start_pos, position_ids, cache, layer_idx)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), attention_started) {
            diag.attention_ms += qwen3_elapsed_ms(started);
        }

        let residual_started = diagnostics.as_ref().map(|_| Instant::now());
        let x = x.broadcast_add(&attn_out)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), residual_started) {
            diag.attention_residual_ms += qwen3_elapsed_ms(started);
        }

        let norm_started = diagnostics.as_ref().map(|_| Instant::now());
        let normed = self.post_attention_layernorm.forward(&x)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), norm_started) {
            diag.post_attention_norm_ms += qwen3_elapsed_ms(started);
        }

        let mlp_started = diagnostics.as_ref().map(|_| Instant::now());
        let mlp_out = self.mlp.forward(&normed)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), mlp_started) {
            diag.mlp_ms += qwen3_elapsed_ms(started);
        }

        let residual_started = diagnostics.as_ref().map(|_| Instant::now());
        let x = x.broadcast_add(&mlp_out)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), residual_started) {
            diag.mlp_residual_ms += qwen3_elapsed_ms(started);
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3WeightLayout {
    pub model_prefix: &'static str,
    pub lm_head_prefix: Option<&'static str>,
}

impl Qwen3WeightLayout {
    pub const STANDARD: Self = Self {
        model_prefix: "model",
        lm_head_prefix: Some("lm_head"),
    };

    pub const VIBEVOICE: Self = Self {
        model_prefix: "model.language_model",
        lm_head_prefix: Some("lm_head"),
    };
}

impl Default for Qwen3WeightLayout {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl Qwen3Model {
    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Self::load_with_layout(cfg, vb, Qwen3WeightLayout::STANDARD)
    }

    pub fn load_with_layout(
        cfg: Qwen3Config,
        vb: VarBuilder,
        layout: Qwen3WeightLayout,
    ) -> Result<Self> {
        let lm_head_size = cfg.lm_head_size.unwrap_or(cfg.vocab_size);
        let embed_tokens = mlx::load_embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp(qwen3_join_prefix(layout.model_prefix, "embed_tokens")),
        )?;
        let lm_head = if let Some(lm_head_prefix) = layout.lm_head_prefix {
            let lm_head_weight = qwen3_join_prefix(lm_head_prefix, "weight");
            if vb.contains_tensor(&lm_head_weight) {
                Qwen3Projection::dense(cfg.hidden_size, lm_head_size, vb.pp(lm_head_prefix))?
            } else {
                if lm_head_size != cfg.vocab_size {
                    return Err(Error::InvalidInput(format!(
                        "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but {} is missing",
                        cfg.vocab_size, lm_head_weight
                    )));
                }
                Qwen3Projection::tied_dense(embed_tokens.embeddings().clone())
            }
        } else {
            if lm_head_size != cfg.vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head_size ({lm_head_size}) differs from vocab_size ({}) but no lm_head prefix was configured",
                    cfg.vocab_size
                )));
            }
            Qwen3Projection::tied_dense(embed_tokens.embeddings().clone())
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Qwen3Layer::load(
                &cfg,
                vb.pp(qwen3_join_prefix(
                    layout.model_prefix,
                    &format!("layers.{idx}"),
                )),
            )?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(qwen3_join_prefix(layout.model_prefix, "norm")),
        )?;
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

    pub fn attention_head_dim(&self) -> usize {
        self.cfg.head_dim()
    }

    pub fn projection_diagnostics(&self) -> Qwen3ProjectionDiagnostics {
        self.layers
            .iter()
            .fold(self.lm_head.diagnostics(), |acc, layer| {
                acc.add(layer.projection_diagnostics())
            })
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
        let hidden =
            self.forward_hidden_with_embeds(embeds, start_pos, cache.as_deref_mut(), position_ids)?;
        self.logits_from_hidden(&hidden)
    }

    pub fn forward_with_embeds_profiled(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Qwen3ForwardDiagnostics)> {
        let mut diagnostics = Qwen3ForwardDiagnostics::default();
        let hidden = self.forward_hidden_with_embeds_inner(
            embeds,
            start_pos,
            cache.as_deref_mut(),
            position_ids,
            Some(&mut diagnostics),
        )?;
        let lm_head_started = Instant::now();
        let logits = self.logits_from_hidden(&hidden)?;
        diagnostics.lm_head_ms += qwen3_elapsed_ms(lm_head_started);
        Ok((logits, diagnostics))
    }

    pub fn forward_hidden_with_embeds_profiled(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        position_ids: Option<&Tensor>,
    ) -> Result<(Tensor, Qwen3ForwardDiagnostics)> {
        let mut diagnostics = Qwen3ForwardDiagnostics::default();
        let hidden = self.forward_hidden_with_embeds_inner(
            embeds,
            start_pos,
            cache.as_deref_mut(),
            position_ids,
            Some(&mut diagnostics),
        )?;
        Ok((hidden, diagnostics))
    }

    pub fn forward_hidden_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_hidden_with_embeds_inner(
            embeds,
            start_pos,
            cache.as_deref_mut(),
            position_ids,
            None,
        )
    }

    fn forward_hidden_with_embeds_inner(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut Qwen3Cache>,
        position_ids: Option<&Tensor>,
        mut diagnostics: Option<&mut Qwen3ForwardDiagnostics>,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();
        let layers_started = diagnostics.as_ref().map(|_| Instant::now());
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            x = match diagnostics.as_deref_mut() {
                Some(diag) => {
                    layer.forward_with_diagnostics(&x, start_pos, position_ids, cache_ref, idx, diag)?
                }
                None => layer.forward(&x, start_pos, position_ids, cache_ref, idx)?,
            };
        }
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), layers_started) {
            diag.layers_total_ms += qwen3_elapsed_ms(started);
        }

        let norm_started = diagnostics.as_ref().map(|_| Instant::now());
        let x = self.norm.forward(&x)?;
        if let (Some(diag), Some(started)) = (diagnostics.as_deref_mut(), norm_started) {
            diag.final_norm_ms += qwen3_elapsed_ms(started);
        }
        Ok(x)
    }

    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden).map_err(Error::from)
    }

    pub fn logits_from_hidden_for_tokens(
        &self,
        hidden: &Tensor,
        token_ids: &[u32],
    ) -> Result<Vec<(u32, f32)>> {
        self.lm_head.logits_for_tokens(hidden, token_ids)
    }

    pub fn uses_mrope(&self) -> bool {
        self.use_mrope
    }
}

fn dense_logits_for_tokens(
    dense: &Qwen3DenseProjection,
    hidden: &Tensor,
    token_ids: &[u32],
) -> Result<Tensor> {
    let token_indices = token_index_tensor(token_ids, hidden.device())?;
    let weight_rows = dense.linear.weight().index_select(&token_indices, 0)?;
    let mut logits = hidden.matmul(&weight_rows.t()?)?;
    if let Some(bias) = dense.linear.bias() {
        let bias_rows = bias.index_select(&token_indices, 0)?;
        logits = logits.broadcast_add(&bias_rows)?;
    }
    Ok(logits)
}

fn logits_for_tokens_from_full(logits: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
    let token_indices = token_index_tensor(token_ids, logits.device())?;
    match logits.dims() {
        [1, _] => logits
            .i(0)?
            .index_select(&token_indices, 0)
            .map_err(Error::from),
        [1, 1, _] => logits
            .i((0, 0))?
            .index_select(&token_indices, 0)
            .map_err(Error::from),
        _ => Err(Error::InferenceError(format!(
            "Qwen3 selected-token logits expected [1,vocab] or [1,1,vocab], got {:?}",
            logits.dims()
        ))),
    }
}

fn token_index_tensor(token_ids: &[u32], device: &Device) -> Result<Tensor> {
    let indices = token_ids
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    Tensor::from_vec(indices, (token_ids.len(),), device).map_err(Error::from)
}

fn qwen3_join_prefix(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

pub(crate) fn dense_decode_attention(
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
    if let Some(out) = try_fused_self_attention(&q_heads, k_heads, v_heads, None, head_dim, false)?
    {
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
    use std::collections::HashMap;

    #[test]
    fn qwen3_dense_projection_loads_optional_bias() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("weight"),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.5f32, -1.0], (2,), &device).expect("bias"),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let projection = Qwen3Projection::dense(2, 2, vb).expect("projection");
        assert_eq!(
            projection.diagnostics(),
            Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: 1,
                quantized_projection_count: 0,
            }
        );

        let input = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2), &device).expect("input");
        let out = projection
            .forward(&input)
            .expect("projection forward")
            .to_vec2::<f32>()
            .expect("output values");
        assert_eq!(out, vec![vec![8.5, 17.0]]);
    }

    #[test]
    fn selected_dense_projection_logits_match_full_projection() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.5, -1.0, 0.25, 2.0, -0.5, 1.5, 0.75, -0.25, 1.25, 0.5,
            ],
            (4, 3),
            &device,
        )
        .expect("weight");
        let bias = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, 0.4], (4,), &device).expect("bias");
        let projection = Qwen3Projection::Dense(Qwen3DenseProjection {
            linear: Linear::new(weight, Some(bias)),
            has_bias: true,
        });
        let hidden = Tensor::from_vec(vec![0.5f32, -1.0, 2.0], (1, 3), &device).expect("hidden");

        let selected = projection
            .logits_for_tokens(&hidden, &[3, 1])
            .expect("selected logits");
        let full = projection
            .forward(&hidden)
            .expect("full logits")
            .i(0)
            .expect("row")
            .to_vec1::<f32>()
            .expect("values");

        assert_eq!(selected.len(), 2);
        assert!((selected[0].1 - full[3]).abs() < 1e-6);
        assert!((selected[1].1 - full[1]).abs() < 1e-6);
    }

    #[test]
    fn qwen3_dense_projection_keeps_no_bias_path() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).expect("weight"),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let projection = Qwen3Projection::dense(2, 2, vb).expect("projection");
        assert_eq!(
            projection.diagnostics(),
            Qwen3ProjectionDiagnostics {
                dense_projection_count: 1,
                dense_bias_projection_count: 0,
                quantized_projection_count: 0,
            }
        );

        let input = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2), &device).expect("input");
        let out = projection
            .forward(&input)
            .expect("projection forward")
            .to_vec2::<f32>()
            .expect("output values");
        assert_eq!(out, vec![vec![8.0, 18.0]]);
    }

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
        let (k_dense, v_dense) = cache.dense_heads(0).expect("dense heads").expect("dense");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 2, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 1, 2, 4));
        assert!(cache.pages(0).is_none());

        let k_next =
            Tensor::from_vec(vec![8.0f32, 9.0, 10.0, 11.0], (1, 1, 1, 4), &device).expect("k next");
        let v_next =
            Tensor::from_vec(vec![4.0f32, 4.5, 5.0, 5.5], (1, 1, 1, 4), &device).expect("v next");
        cache.append(0, k_next, v_next).expect("append next");
        let (k_dense, _) = cache
            .dense_heads(0)
            .expect("dense heads after append")
            .expect("dense");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 3, 4));
    }

    #[test]
    fn qwen3_cache_dense_decode_uses_bounded_initial_capacity() {
        let device = Device::Cpu;
        let mut cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            4,
            KvCacheQuantization::None,
            16,
        );
        let k = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).expect("k");
        let v = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).expect("v");

        cache.append(0, k, v).expect("append");

        assert_eq!(
            cache.dense_k_cache_h[0]
                .as_ref()
                .expect("dense key cache")
                .max_seq_len(),
            4
        );
        assert_eq!(
            cache.dense_v_cache_h[0]
                .as_ref()
                .expect("dense value cache")
                .max_seq_len(),
            4
        );

        let k_next = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("next k");
        let v_next = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("next v");
        cache.append(0, k_next, v_next).expect("append next");

        assert_eq!(
            cache.dense_k_cache_h[0]
                .as_ref()
                .expect("dense key cache")
                .max_seq_len(),
            8
        );
        let (k_dense, v_dense) = cache.dense_heads(0).expect("dense heads").expect("dense");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 5, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 1, 5, 4));
    }

    #[test]
    fn qwen3_cache_dense_decode_honors_initial_capacity_hint() {
        let device = Device::Cpu;
        let mut cache =
            Qwen3Cache::with_page_size_quantization_dense_decode_tokens_initial_capacity(
                1,
                4,
                KvCacheQuantization::None,
                16,
                5,
            );
        let k = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).expect("k");
        let v = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).expect("v");

        cache.append(0, k, v).expect("append");

        assert_eq!(
            cache.dense_k_cache_h[0]
                .as_ref()
                .expect("dense key cache")
                .max_seq_len(),
            8
        );
        let k_next = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("next k");
        let v_next = Tensor::zeros((1, 3, 1, 4), DType::F32, &device).expect("next v");
        cache.append(0, k_next, v_next).expect("append next");

        assert_eq!(
            cache.dense_k_cache_h[0]
                .as_ref()
                .expect("dense key cache")
                .max_seq_len(),
            8
        );
        let (k_dense, v_dense) = cache.dense_heads(0).expect("dense heads").expect("dense");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 1, 5, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 1, 5, 4));
    }

    #[test]
    fn qwen3_cache_reuses_rope_windows() {
        let device = Device::Cpu;
        let mut cache = Qwen3Cache::new(1);
        let head_dim = 8usize;
        let inv_freqs = build_rope_inv_freqs(head_dim, 10000.0);

        let (first_cos, first_sin) = cache
            .cached_rope_pair(2, 4, head_dim, &device, DType::F32, false, &[], &inv_freqs)
            .expect("first rope cache");
        let (second_cos, second_sin) = cache
            .cached_rope_pair(2, 4, head_dim, &device, DType::F32, false, &[], &inv_freqs)
            .expect("cached rope cache");

        assert_eq!(cache.rope_cache.len(), 1);
        assert_eq!(
            first_cos.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second_cos.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            first_sin.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second_sin.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        cache
            .cached_rope_pair(2, 5, head_dim, &device, DType::F32, false, &[], &inv_freqs)
            .expect("new rope window");
        assert_eq!(cache.rope_cache.len(), 2);
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
        assert!(cache.dense_heads(0).expect("dense heads").is_none());

        let mut quantized_cache = Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
            1,
            4,
            KvCacheQuantization::Int8,
            8,
        );
        quantized_cache.append(0, k, v).expect("append quantized");
        assert!(
            quantized_cache
                .dense_heads(0)
                .expect("dense heads")
                .is_none()
        );
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
        assert!(cache.dense_heads(0).expect("dense heads").is_some());
        assert!(cache.pages(0).is_none());

        let next_k =
            Tensor::from_vec(vec![8.0f32, 9.0, 10.0, 11.0], (1, 1, 1, 4), &device).expect("next k");
        let next_v = Tensor::from_vec(vec![28.0f32, 29.0, 30.0, 31.0], (1, 1, 1, 4), &device)
            .expect("next v");
        cache.append(0, next_k, next_v).expect("append next");

        assert!(cache.dense_heads(0).expect("dense heads").is_none());
        let (k_pages, v_pages) = cache.pages(0).expect("pages after migration");
        assert_eq!(k_pages.len(), 2);
        assert_eq!(v_pages.len(), 2);
        let (k_dense, v_dense) = cache.materialize(0).expect("materialized");
        assert_eq!(k_dense.dims4().expect("k dims"), (1, 3, 1, 4));
        assert_eq!(v_dense.dims4().expect("v dims"), (1, 3, 1, 4));
    }

    #[test]
    fn qwen3_dense_decode_policy_enables_only_metal_and_cuda() {
        assert!(!qwen3_use_dense_decode_attention_policy(false, false, None));
        assert!(qwen3_use_dense_decode_attention_policy(true, false, None));
        assert!(qwen3_use_dense_decode_attention_policy(false, true, None));
        assert!(qwen3_use_dense_decode_attention_policy(true, true, None));
        assert!(!qwen3_use_dense_decode_attention_policy(
            true,
            false,
            Some(false)
        ));
        assert!(!qwen3_use_dense_decode_attention_policy(
            false,
            true,
            Some(false)
        ));
        assert!(!qwen3_use_dense_decode_attention_policy(
            false,
            false,
            Some(true)
        ));
    }

    #[test]
    fn qwen3_cuda_qmatmul_uses_f32_input_for_lower_precision_activations() {
        assert_eq!(
            qwen3_cuda_qmatmul_input_dtype(true, DType::BF16),
            Some(DType::F32)
        );
        assert_eq!(
            qwen3_cuda_qmatmul_input_dtype(true, DType::F16),
            Some(DType::F32)
        );
        assert_eq!(qwen3_cuda_qmatmul_input_dtype(true, DType::F32), None);
        assert_eq!(qwen3_cuda_qmatmul_input_dtype(false, DType::BF16), None);
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

    #[cfg(feature = "metal")]
    #[test]
    fn dense_decode_attention_matches_manual_gqa_on_metal_decode_window() {
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            _ => return,
        };
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 64usize;
        let kv_len = 4usize;
        let q_values = (0..(num_heads * head_dim))
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.025)
            .collect::<Vec<_>>();
        let k_values = (0..(kv_len * num_kv_heads * head_dim))
            .map(|idx| ((idx % 23) as f32 - 11.0) * 0.02)
            .collect::<Vec<_>>();
        let v_values = (0..(kv_len * num_kv_heads * head_dim))
            .map(|idx| ((idx % 19) as f32 - 9.0) * 0.03)
            .collect::<Vec<_>>();

        let q = Tensor::from_vec(q_values, (1, 1, num_heads, head_dim), &device).expect("q");
        let k =
            Tensor::from_vec(k_values, (1, kv_len, num_kv_heads, head_dim), &device).expect("k");
        let v =
            Tensor::from_vec(v_values, (1, kv_len, num_kv_heads, head_dim), &device).expect("v");
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

        let dense =
            dense_decode_attention(&q, &k_heads, &v_heads, num_heads, num_kv_heads, head_dim)
                .expect("dense");
        let q_heads = q.transpose(1, 2).expect("q heads");
        let k_repeated = repeat_kv(&k, num_heads, num_kv_heads).expect("repeat k");
        let v_repeated = repeat_kv(&v, num_heads, num_kv_heads).expect("repeat v");
        let k_repeated = k_repeated.transpose(1, 2).expect("k repeated heads");
        let v_repeated = v_repeated.transpose(1, 2).expect("v repeated heads");
        let mut att = q_heads
            .matmul(&k_repeated.transpose(2, 3).expect("k t"))
            .expect("scores");
        att = (att / (head_dim as f64).sqrt()).expect("scale");
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
            assert!((lhs - rhs).abs() < 5e-3, "{lhs} != {rhs}");
        }
    }
}

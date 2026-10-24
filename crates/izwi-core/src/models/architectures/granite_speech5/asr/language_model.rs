//! Granite Speech 5.0's language model: granite-4.0-1b-base extended with
//! YaRN to 16k positions. Granite's four multipliers apply as in 4.1
//! (embedding, residual, attention scale, logits), attention runs on the
//! paged invocation cache, and RoPE reads a host-built F64 cos/sin table
//! carrying the YaRN frequencies and attention scale. The GGUF release's
//! projections, head and token table stay quantized (`QMatMul`).

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias, rotary_emb, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

use crate::error::{Error, Result};
use crate::kernels::numeric_check::xorshift_values;
use crate::kernels::{
    try_fused_rms_norm, try_fused_rope_pair_bshd, try_fused_silu_mul_with_status,
};
use crate::models::shared::attention::geometry::AttentionGeometry;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};
use crate::models::shared::memory::exact::exact_concat_rows;
use crate::models::shared::rope::{rope_frequencies, RopeFrequencies, RopeScaling};
use crate::models::shared::telemetry::{
    record_decode_attention_path, record_rope_kernel, record_rope_manual, DecodeAttentionPath,
};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::models::shared::weights::token_table::QuantizedTokenTable;

use super::config::GraniteSpeech5TextConfig;
use super::gguf::{joined_qmatmul, QuantizedPart};
use super::precision::{self, LmExecutionPlan};

/// Attention geometry, checked before any head-dimension division.
pub(super) fn text_attention_geometry(
    config: &GraniteSpeech5TextConfig,
) -> Result<AttentionGeometry> {
    for (name, value) in [
        ("rope_theta", config.rope_theta),
        (
            "attention_multiplier",
            f64::from(config.attention_multiplier),
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 text attention geometry is invalid: {name} must be finite and positive, got {value}"
            )));
        }
    }
    let geometry = AttentionGeometry::from_hidden_size(
        "Granite Speech 5.0 text",
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        None,
    )?;
    geometry.validate_rotary_dim("Granite Speech 5.0 text", geometry.key_head_dim())?;
    Ok(geometry)
}

/// A bias-free projection: dense, or quantized as the GGUF release stores it.
#[derive(Clone, Debug)]
enum LinearNoBias {
    Dense {
        linear: Linear,
        /// Cached transposed weight (Metal), so each step skips the transpose.
        weight_t: Option<Tensor>,
    },
    /// Candle's quantized kernels take F32 input on Metal and CUDA (and
    /// quantize it on the CPU), so a quantized projection computes in F32.
    Quantized(QMatMul),
}

impl LinearNoBias {
    fn new(linear: Linear) -> Result<Self> {
        let weight_t = precision::cached_linear_t(linear.weight().device())
            .then(|| linear.weight().t())
            .transpose()?;
        Ok(Self::Dense { linear, weight_t })
    }

    fn from_weight(weight: Tensor) -> Result<Self> {
        Self::new(Linear::new(weight, None))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense {
                linear,
                weight_t: Some(weight_t),
            } => cached_linear_forward(x, weight_t)
                .or_else(|_| linear.forward(x).map_err(Error::from)),
            Self::Dense { linear, .. } => linear.forward(x).map_err(Error::from),
            Self::Quantized(matmul) => matmul.forward(x).map_err(Error::from),
        }
    }

    /// The dtype the projection reads its input in: the dense weight's, or
    /// F32 for a quantized one.
    fn compute_dtype(&self) -> DType {
        match self {
            Self::Dense { linear, .. } => linear.weight().dtype(),
            Self::Quantized(_) => DType::F32,
        }
    }

    /// Run in the compute dtype, returning `output_dtype`.
    fn forward_cast(&self, x: &Tensor, output_dtype: DType) -> Result<Tensor> {
        let compute_dtype = self.compute_dtype();
        let out = if x.dtype() == compute_dtype {
            self.forward(x)?
        } else {
            self.forward(&x.to_dtype(compute_dtype)?)?
        };
        if out.dtype() == output_dtype {
            Ok(out)
        } else {
            out.to_dtype(output_dtype).map_err(Error::from)
        }
    }
}

fn cached_linear_forward(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    match *x.dims() {
        [b1, b2, m, k] if x.is_contiguous() => x
            .reshape((b1 * b2 * m, k))?
            .matmul(weight_t)?
            .reshape((b1, b2, m, ()))
            .map_err(Error::from),
        [batch, m, k] if x.is_contiguous() => x
            .reshape((batch * m, k))?
            .matmul(weight_t)?
            .reshape((batch, m, ()))
            .map_err(Error::from),
        _ => x.matmul(weight_t).map_err(Error::from),
    }
}

/// The host-built RoPE table for a whole invocation, sliced per step.
#[derive(Debug, Clone)]
pub(super) struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    /// `[cos | sin]` per position, for Metal's pair kernel.
    packed: Tensor,
}

#[derive(Debug, Clone)]
struct RopeSlice {
    cos: Tensor,
    sin: Tensor,
    packed: Tensor,
}

impl RopeCache {
    fn build(
        frequencies: &RopeFrequencies,
        start: usize,
        len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let pairs = frequencies.inv_freq.len();
        let (cos, sin) = frequencies.cos_sin_table(start, len);
        let cos = Tensor::from_vec(cos, (len, pairs), device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin, (len, pairs), device)?.to_dtype(dtype)?;
        let packed = Tensor::cat(&[&cos, &sin], 1)?;
        Ok(Self { cos, sin, packed })
    }

    fn slice(&self, start: usize, len: usize) -> Result<RopeSlice> {
        if start.saturating_add(len) > self.cos.dim(0)? {
            return Err(Error::InferenceError(format!(
                "Granite Speech 5.0 RoPE cache too short for start={start}, len={len}"
            )));
        }
        Ok(RopeSlice {
            cos: self.cos.narrow(0, start, len)?,
            sin: self.sin.narrow(0, start, len)?,
            packed: self.packed.narrow(0, start, len)?,
        })
    }
}

/// The token table: dense, or the GGUF release's quantized table.
enum TokenTable {
    Dense(Embedding),
    Quantized(QuantizedTokenTable),
}

/// The token embedding in the model's base dtype.
struct TokenEmbedding {
    table: TokenTable,
    dtype: DType,
}

impl TokenEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let rows = match &self.table {
            TokenTable::Dense(table) => table.forward(ids)?,
            TokenTable::Quantized(table) => table.embed(ids)?,
        };
        if rows.dtype() == self.dtype {
            Ok(rows)
        } else {
            rows.to_dtype(self.dtype).map_err(Error::from)
        }
    }
}

pub(super) struct GraniteLanguageModel {
    embed_tokens: TokenEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearNoBias,
    cfg: GraniteSpeech5TextConfig,
    rope: RopeFrequencies,
    head_dim: usize,
    device: Device,
    residual_branches_prescaled: bool,
    plan: LmExecutionPlan,
}

impl GraniteLanguageModel {
    pub(super) fn load(
        config: &GraniteSpeech5TextConfig,
        rope: &RopeScaling,
        tie_word_embeddings: bool,
        vb: VarBuilder,
        plan: LmExecutionPlan,
    ) -> Result<Self> {
        let geometry = text_attention_geometry(config)?;
        let head_dim = geometry.key_head_dim();
        let rope = rope_frequencies(head_dim, config.rope_theta, rope)?;
        let shared_f16_head = tie_word_embeddings && plan.f16_lm_head;
        // A tied head served in F16 stores the table once in F16 and casts
        // gathered rows to the base dtype.
        let table_vb = if shared_f16_head {
            vb.pp("model.embed_tokens").to_dtype(DType::F16)
        } else {
            vb.pp("model.embed_tokens")
        };
        let table = embedding(config.vocab_size, config.hidden_size, table_vb)?;
        let residual_branches_prescaled = precision::residual_prescale(vb.device());
        let layers = (0..config.num_hidden_layers)
            .map(|idx| {
                DecoderLayer::load(
                    config,
                    head_dim,
                    vb.pp(format!("model.layers.{idx}")),
                    residual_branches_prescaled,
                    plan,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if tie_word_embeddings {
            Linear::new(table.embeddings().clone(), None)
        } else {
            let head_vb = if plan.f16_lm_head {
                vb.to_dtype(DType::F16)
            } else {
                vb.clone()
            };
            linear_no_bias(config.hidden_size, config.vocab_size, head_vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens: TokenEmbedding {
                table: TokenTable::Dense(table),
                dtype: vb.dtype(),
            },
            layers,
            norm,
            lm_head: LinearNoBias::new(lm_head)?,
            cfg: config.clone(),
            rope,
            head_dim,
            device: vb.device().clone(),
            residual_branches_prescaled,
            plan,
        })
    }

    /// The language model from the GGUF release: its quantized projections,
    /// head and token table stay quantized (Q/K rows put back in half-split
    /// RoPE order), norms load in `dtype`, and activations run in `dtype`.
    /// Quantized weights cannot carry the residual multiplier, so it scales
    /// each branch instead; `plan` may only change the attention core's dtype.
    pub(super) fn load_gguf(
        config: &GraniteSpeech5TextConfig,
        rope: &RopeScaling,
        gguf: &GgufLoader,
        device: &Device,
        dtype: DType,
        plan: LmExecutionPlan,
    ) -> Result<Self> {
        if plan.f16_lm_head || plan.f16_qkv || plan.f16_mlp || plan.f16_attention_output {
            return Err(Error::ModelLoadError(
                "Granite Speech 5.0 GGUF projections stay quantized; only the attention core may run in F16"
                    .to_string(),
            ));
        }
        let geometry = text_attention_geometry(config)?;
        let head_dim = geometry.key_head_dim();
        let rope = rope_frequencies(head_dim, config.rope_theta, rope)?;
        // Norms are read and converted on the host, then uploaded.
        let norm = |name: &str| -> Result<RmsNorm> {
            let weight = gguf.load_tensor(name, dtype, &Device::Cpu)?;
            Ok(RmsNorm::new(weight.to_device(device)?, config.rms_norm_eps))
        };
        let quantized = |names: &[(&str, Option<usize>)]| -> Result<LinearNoBias> {
            let parts = names
                .iter()
                .map(|(name, rope_heads)| QuantizedPart {
                    name,
                    rope_heads: *rope_heads,
                })
                .collect::<Vec<_>>();
            Ok(LinearNoBias::Quantized(joined_qmatmul(
                gguf, &parts, device,
            )?))
        };
        let layers = (0..config.num_hidden_layers)
            .map(|idx| {
                let name = |part: &str| format!("blk.{idx}.{part}.weight");
                let (q, k, v) = (name("attn_q"), name("attn_k"), name("attn_v"));
                let (gate, up) = (name("ffn_gate"), name("ffn_up"));
                Ok(DecoderLayer {
                    input_layernorm: norm(&name("attn_norm"))?,
                    self_attn: TextAttention {
                        qkv_proj: QkvProjection::Fused {
                            fused: quantized(&[
                                (&q, Some(config.num_attention_heads)),
                                (&k, Some(config.num_key_value_heads)),
                                (&v, None),
                            ])?,
                            q_out: config.num_attention_heads * head_dim,
                            k_out: config.num_key_value_heads * head_dim,
                        },
                        o_proj: quantized(&[(&name("attn_output"), None)])?,
                        f16_attention_core: plan.f16_attention_core,
                        num_heads: config.num_attention_heads,
                        num_kv_heads: config.num_key_value_heads,
                        head_dim,
                        attention_multiplier: config.attention_multiplier,
                    },
                    post_attention_layernorm: norm(&name("ffn_norm"))?,
                    mlp: TextMlp {
                        gate_up_proj: GateUpProjection::Fused {
                            fused: quantized(&[(&gate, None), (&up, None)])?,
                            intermediate_size: config.intermediate_size,
                        },
                        down_proj: quantized(&[(&name("ffn_down"), None)])?,
                        f16: false,
                    },
                    residual_multiplier: config.residual_multiplier,
                    residual_branches_prescaled: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let table = QuantizedTokenTable::new(gguf.load_qtensor("token_embd.weight", device)?)?;
        if table.vocab_size() != config.vocab_size || table.hidden_size() != config.hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 GGUF token table is {} x {}, but config.json gives {} x {}",
                table.vocab_size(),
                table.hidden_size(),
                config.vocab_size,
                config.hidden_size
            )));
        }
        Ok(Self {
            embed_tokens: TokenEmbedding {
                table: TokenTable::Quantized(table),
                dtype,
            },
            layers,
            norm: norm("output_norm.weight")?,
            lm_head: quantized(&[("output.weight", None)])?,
            cfg: config.clone(),
            rope,
            head_dim,
            device: device.clone(),
            residual_branches_prescaled: false,
            plan,
        })
    }

    pub(super) fn plan(&self) -> LmExecutionPlan {
        self.plan
    }

    pub(super) fn residual_branches_prescaled(&self) -> bool {
        self.residual_branches_prescaled
    }

    pub(super) fn rope_attention_scale(&self) -> f64 {
        self.rope.attention_scale
    }

    pub(super) fn kv_dtype(&self) -> DType {
        if self.plan.f16_attention_core {
            DType::F16
        } else {
            self.embed_tokens.dtype
        }
    }

    fn rope_dtype(&self) -> DType {
        if self.plan.f16_attention_core && precision::rope_cache_in_attention_dtype(&self.device) {
            DType::F16
        } else {
            self.embed_tokens.dtype
        }
    }

    /// The RoPE table for positions `0..total_len`, built once per invocation.
    pub(super) fn rope_cache(&self, total_len: usize) -> Result<RopeCache> {
        RopeCache::build(&self.rope, 0, total_len, &self.device, self.rope_dtype())
    }

    fn embeddings(&self, input_ids: &[u32]) -> Result<Tensor> {
        let ids = input_ids
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (1, input_ids.len()), &self.device)?;
        self.embed_tokens.forward(&ids)
    }

    /// Prompt embeddings with the projected audio spliced over the expanded
    /// audio placeholders, before the embedding multiplier (which then
    /// scales the audio too, as in transformers).
    pub(super) fn prompt_embeddings_with_audio(
        &self,
        input_ids: &[u32],
        audio_start: usize,
        audio_embeds: &Tensor,
    ) -> Result<Tensor> {
        let audio_len = audio_embeds.dim(1)?;
        let mut ids = input_ids.to_vec();
        for id in ids.iter_mut().skip(audio_start).take(audio_len) {
            *id = 0;
        }
        let embeds = self.embeddings(&ids)?;
        let (_, seq_len, hidden) = embeds.dims3()?;
        let after_start = audio_start + audio_len;
        if after_start > seq_len {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 audio span exceeds the prompt".into(),
            ));
        }
        let empty = || Tensor::zeros((1, 0, hidden), embeds.dtype(), embeds.device());
        let before = if audio_start > 0 {
            embeds.narrow(1, 0, audio_start)?
        } else {
            empty()?
        };
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            empty()?
        };
        let audio = audio_embeds.to_dtype(embeds.dtype())?;
        Tensor::cat(&[before, audio, after], 1).map_err(Error::from)
    }

    /// One decode step for a token tensor `[1, 1]`.
    pub(super) fn forward_token(
        &self,
        token: &Tensor,
        position: usize,
        cache: &mut PhysicalPagedKvCache,
        rope: &RopeCache,
    ) -> Result<Tensor> {
        let embeds = self.embed_tokens.forward(token)?;
        self.forward_embeds(&embeds, position, cache, Some(rope), None)
    }

    /// Runs `embeds` `[1, sequence, hidden]` from `start_pos`, appending to
    /// `cache`, and returns the last position's (unscaled) logits.
    /// Greedy decoding needs no `logits_scaling`: dividing by a positive
    /// constant keeps the argmax. `observe` sees each decoder layer's output.
    pub(super) fn forward_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        rope: Option<&RopeCache>,
        mut observe: Option<&mut dyn FnMut(usize, &Tensor) -> Result<()>>,
    ) -> Result<Tensor> {
        let (batch, sequence_len, hidden_size) = embeds.dims3()?;
        if batch != 1 || sequence_len == 0 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 decoder embeddings must have shape [1,sequence,{}], got {:?}",
                self.cfg.hidden_size,
                embeds.dims()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.head_dim,
        )?;
        cache.slots_for_append(start_pos, sequence_len)?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let rope = match rope {
            Some(cache) => cache.slice(start_pos, sequence_len)?,
            None => {
                let table = RopeCache::build(
                    &self.rope,
                    start_pos,
                    sequence_len,
                    &self.device,
                    self.rope_dtype(),
                )?;
                table.slice(0, sequence_len)?
            }
        };
        let mut x = (embeds * f64::from(self.cfg.embedding_multiplier))?;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, cache, &mut prepared, idx, &rope)?;
            if let Some(observe) = observe.as_mut() {
                observe(idx, &x)?;
            }
        }
        let hidden = rms_norm_forward(&self.norm, &x)?;
        let hidden = hidden.narrow(1, sequence_len - 1, 1)?;
        let logits = self
            .lm_head
            .forward_cast(&hidden, self.lm_head.compute_dtype())?;
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    /// Logits at transformers' scale (`logits / logits_scaling`, F32), for
    /// parity checks.
    pub(super) fn scale_logits(&self, logits: &Tensor) -> Result<Tensor> {
        (logits.to_dtype(DType::F32)? / f64::from(self.cfg.logits_scaling)).map_err(Error::from)
    }

    /// The load-time range probe: `tokens` xorshift-chosen token ids through
    /// every decoder layer and the head, on a private scratch cache in the
    /// LM's K/V dtype. Returns each layer's output and the logits.
    pub(super) fn numeric_probe(&self, tokens: usize) -> Result<Vec<(String, Tensor)>> {
        let vocab = self.cfg.vocab_size.max(1);
        let ids = xorshift_values(tokens, 0x9A11, 1.0)
            .into_iter()
            .map(|unit| (((unit + 1.0) * 0.5 * vocab as f32) as usize).min(vocab - 1) as u32)
            .collect::<Vec<_>>();
        let embeds = self.embeddings(&ids)?;
        let mut cache = PhysicalPagedKvCache::scratch_for_probe(
            &self.device,
            self.kv_dtype(),
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.head_dim,
            tokens,
        )?;
        let mut outputs = Vec::new();
        let mut observe = |index: usize, x: &Tensor| {
            outputs.push((format!("layers.{index}"), x.clone()));
            Ok(())
        };
        let logits = self.forward_embeds(&embeds, 0, &mut cache, None, Some(&mut observe))?;
        outputs.push(("logits".to_string(), logits));
        Ok(outputs)
    }
}

fn rms_norm_forward(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    if precision::fused_rms_norm(x.device()) {
        if let Some(out) = try_fused_rms_norm(x, norm.weight(), norm.eps()) {
            return Ok(out);
        }
    }
    norm.forward(x).map_err(Error::from)
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: TextAttention,
    post_attention_layernorm: RmsNorm,
    mlp: TextMlp,
    residual_multiplier: f32,
    residual_branches_prescaled: bool,
}

impl DecoderLayer {
    fn load(
        config: &GraniteSpeech5TextConfig,
        head_dim: usize,
        vb: VarBuilder,
        residual_branches_prescaled: bool,
        plan: LmExecutionPlan,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: TextAttention::load(
                config,
                head_dim,
                vb.pp("self_attn"),
                residual_branches_prescaled,
                plan,
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: TextMlp::load(config, vb.pp("mlp"), residual_branches_prescaled, plan)?,
            residual_multiplier: config.residual_multiplier,
            residual_branches_prescaled,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
        rope: &RopeSlice,
    ) -> Result<Tensor> {
        let normed = rms_norm_forward(&self.input_layernorm, x)?;
        let attn = self
            .self_attn
            .forward(&normed, cache, prepared, layer_idx, rope)?;
        let x = x.broadcast_add(&self.scale_residual_branch(attn)?)?;
        let normed = rms_norm_forward(&self.post_attention_layernorm, &x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&self.scale_residual_branch(mlp)?)
            .map_err(Error::from)
    }

    fn scale_residual_branch(&self, branch: Tensor) -> Result<Tensor> {
        if self.residual_branches_prescaled {
            Ok(branch)
        } else {
            (branch * f64::from(self.residual_multiplier)).map_err(Error::from)
        }
    }
}

/// Q/K/V as one projection (rows joined at their exact size, since a pooled
/// `Tensor::cat` result is rounded up to a power of two for the model's
/// life), or three.
enum QkvProjection {
    Fused {
        fused: LinearNoBias,
        q_out: usize,
        k_out: usize,
    },
    Separate {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
    },
}

impl QkvProjection {
    fn load(config: &GraniteSpeech5TextConfig, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let kv_out = config.num_key_value_heads * head_dim;
        let k_proj = linear_no_bias(config.hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(config.hidden_size, kv_out, vb.pp("v_proj"))?;
        if !precision::PROJECTION_FUSION {
            return Ok(Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            });
        }
        let weight = exact_concat_rows(&[q_proj.weight(), k_proj.weight(), v_proj.weight()])?;
        Ok(Self::Fused {
            fused: LinearNoBias::from_weight(weight)?,
            q_out: q_proj.weight().dim(0)?,
            k_out: kv_out,
        })
    }

    fn weight_dtype(&self) -> DType {
        match self {
            Self::Fused { fused, .. } => fused.compute_dtype(),
            Self::Separate { q_proj, .. } => q_proj.weight().dtype(),
        }
    }

    /// Projects in the weight's dtype and returns `output_dtype`.
    fn forward(&self, x: &Tensor, output_dtype: DType) -> Result<(Tensor, Tensor, Tensor)> {
        let weight_dtype = self.weight_dtype();
        let x = if x.dtype() == weight_dtype {
            x.clone()
        } else {
            x.to_dtype(weight_dtype)?
        };
        let (q, k, v) = match self {
            Self::Fused {
                fused,
                q_out,
                k_out,
            } => {
                let qkv = fused.forward(&x)?;
                let last = qkv.rank() - 1;
                (
                    qkv.narrow(last, 0, *q_out)?,
                    qkv.narrow(last, *q_out, *k_out)?,
                    qkv.narrow(last, q_out + k_out, *k_out)?,
                )
            }
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(&x)?,
                k_proj.forward(&x)?,
                v_proj.forward(&x)?,
            ),
        };
        let cast = |t: Tensor| -> Result<Tensor> {
            if t.dtype() == output_dtype {
                Ok(t)
            } else {
                t.to_dtype(output_dtype).map_err(Error::from)
            }
        };
        Ok((cast(q)?, cast(k)?, cast(v)?))
    }
}

struct TextAttention {
    qkv_proj: QkvProjection,
    o_proj: LinearNoBias,
    f16_attention_core: bool,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f32,
}

impl TextAttention {
    fn load(
        config: &GraniteSpeech5TextConfig,
        head_dim: usize,
        vb: VarBuilder,
        prescale_residual: bool,
        plan: LmExecutionPlan,
    ) -> Result<Self> {
        // A selective-F16 o_proj is read and pre-scaled at F16: an F32 read,
        // an F32 product and an F16 cast would leave two released full-size
        // F32 buffers per layer for Metal's pool to hold.
        let o_proj_vb = if plan.f16_attention_output {
            vb.to_dtype(DType::F16)
        } else {
            vb.clone()
        };
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            o_proj_vb.pp("o_proj"),
        )?;
        let o_proj =
            prescale_residual_linear(o_proj, config.residual_multiplier, prescale_residual)?;
        let qkv_vb = if plan.f16_qkv {
            vb.to_dtype(DType::F16)
        } else {
            vb
        };
        Ok(Self {
            qkv_proj: QkvProjection::load(config, head_dim, qkv_vb)?,
            o_proj: LinearNoBias::new(o_proj)?,
            f16_attention_core: plan.f16_attention_core,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            attention_multiplier: config.attention_multiplier,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
        rope: &RopeSlice,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let core_dtype = if self.f16_attention_core {
            DType::F16
        } else {
            x.dtype()
        };
        let (q, k, v) = self.qkv_proj.forward(x, core_dtype)?;
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let (q, k) = apply_rope(&q, &k, rope, self.head_dim)?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q.reshape((seq_len, self.num_heads, self.head_dim))?
                .contiguous()?,
            &k.reshape((seq_len, self.num_kv_heads, self.head_dim))?
                .contiguous()?,
            &v.reshape((seq_len, self.num_kv_heads, self.head_dim))?
                .contiguous()?,
            self.attention_multiplier,
        )?;
        record_decode_attention_path(DecodeAttentionPath::Paged);
        let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward_cast(&out, x.dtype())
    }
}

/// Rotates `q` and `k` (`[batch, seq, heads, dim]`, half-split pairs) by the
/// table, through the backend's kernel when it has one.
fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    rope: &RopeSlice,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let dtype = q.dtype();
    let cast = |t: &Tensor| -> Result<Tensor> {
        if t.dtype() == dtype {
            Ok(t.clone())
        } else {
            t.to_dtype(dtype).map_err(Error::from)
        }
    };
    let (cos, sin, packed) = (cast(&rope.cos)?, cast(&rope.sin)?, cast(&rope.packed)?);
    let device = q.device();
    if precision::rope_pair_bshd_kernel(device, dtype, head_dim) {
        if let Some(rotated) = try_fused_rope_pair_bshd(q, k, &packed) {
            record_rope_kernel();
            record_rope_kernel();
            return Ok(rotated);
        }
    }
    if precision::rope_bhtd_kernel(device, dtype, head_dim) {
        if let Some(rotated) = try_rope_kernel("granite_speech5.rope_bhtd", || {
            let q = rotary_emb::rope(&q.transpose(1, 2)?.contiguous()?, &cos, &sin)?;
            let k = rotary_emb::rope(&k.transpose(1, 2)?.contiguous()?, &cos, &sin)?;
            Ok((
                q.transpose(1, 2)?.contiguous()?,
                k.transpose(1, 2)?.contiguous()?,
            ))
        }) {
            record_rope_kernel();
            record_rope_kernel();
            return Ok(rotated);
        }
    } else if precision::rope_thd_kernel(device, dtype, head_dim) {
        let cos = cos.unsqueeze(0)?.contiguous()?;
        let sin = sin.unsqueeze(0)?.contiguous()?;
        if let Some(rotated) = try_rope_kernel("granite_speech5.rope_thd", || {
            Ok((
                rotary_emb::rope_thd(&q.contiguous()?, &cos, &sin)?,
                rotary_emb::rope_thd(&k.contiguous()?, &cos, &sin)?,
            ))
        }) {
            record_rope_kernel();
            record_rope_kernel();
            return Ok(rotated);
        }
    }
    record_rope_manual();
    record_rope_manual();
    Ok((
        apply_rotary_emb(q, &cos, &sin)?,
        apply_rotary_emb(k, &cos, &sin)?,
    ))
}

/// A Candle RoPE kernel call, `None` (reported) when it fails or panics.
fn try_rope_kernel(
    label: &'static str,
    kernel: impl FnOnce() -> candle_core::Result<(Tensor, Tensor)>,
) -> Option<(Tensor, Tensor)> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(kernel));
    match result {
        Ok(Ok(rotated)) => Some(rotated),
        failure => {
            crate::models::shared::telemetry::report_kernel_fallback(label, &failure);
            None
        }
    }
}

fn apply_rotary_emb(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[first, second], 3).map_err(Error::from)
}

/// Gate and up as one projection, or two.
enum GateUpProjection {
    Fused {
        fused: LinearNoBias,
        intermediate_size: usize,
    },
    Separate {
        gate_proj: Linear,
        up_proj: Linear,
    },
}

impl GateUpProjection {
    fn load(config: &GraniteSpeech5TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        if !precision::PROJECTION_FUSION {
            return Ok(Self::Separate { gate_proj, up_proj });
        }
        Ok(Self::Fused {
            fused: LinearNoBias::from_weight(exact_concat_rows(&[
                gate_proj.weight(),
                up_proj.weight(),
            ])?)?,
            intermediate_size: config.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused {
                fused,
                intermediate_size,
            } => {
                let gate_up = fused.forward(x)?;
                let last = gate_up.rank() - 1;
                Ok((
                    gate_up.narrow(last, 0, *intermediate_size)?,
                    gate_up.narrow(last, *intermediate_size, *intermediate_size)?,
                ))
            }
            Self::Separate { gate_proj, up_proj } => {
                Ok((gate_proj.forward(x)?, up_proj.forward(x)?))
            }
        }
    }
}

struct TextMlp {
    gate_up_proj: GateUpProjection,
    down_proj: LinearNoBias,
    /// Selective F16: activations cast to F16 for the MLP and back.
    f16: bool,
}

impl TextMlp {
    fn load(
        config: &GraniteSpeech5TextConfig,
        vb: VarBuilder,
        prescale_residual: bool,
        plan: LmExecutionPlan,
    ) -> Result<Self> {
        // As for o_proj, a selective-F16 down projection is read and
        // pre-scaled at F16.
        let vb = if plan.f16_mlp {
            vb.to_dtype(DType::F16)
        } else {
            vb
        };
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        let down_proj =
            prescale_residual_linear(down_proj, config.residual_multiplier, prescale_residual)?;
        Ok(Self {
            gate_up_proj: GateUpProjection::load(config, vb)?,
            down_proj: LinearNoBias::new(down_proj)?,
            f16: plan.f16_mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output_dtype = x.dtype();
        let x = if self.f16 {
            x.to_dtype(DType::F16)?
        } else {
            x.clone()
        };
        let (gate, up) = self.gate_up_proj.forward(&x)?;
        let hidden = match try_fused_silu_mul_with_status(&gate, &up) {
            Some(fused) => fused.tensor,
            None => gate.silu()?.broadcast_mul(&up)?,
        };
        self.down_proj.forward_cast(&hidden, output_dtype)
    }
}

/// Folds Granite's residual multiplier into a bias-free projection.
fn prescale_residual_linear(
    linear: Linear,
    residual_multiplier: f32,
    enabled: bool,
) -> Result<Linear> {
    if !enabled || residual_multiplier == 1.0 {
        return Ok(linear);
    }
    if linear.bias().is_some() {
        return Err(Error::InferenceError(
            "Granite residual pre-scaling only supports bias-free projections".to_string(),
        ));
    }
    Ok(Linear::new(
        (linear.weight() * f64::from(residual_multiplier))?,
        None,
    ))
}

/// Greedy next token of the last position as a `[1, 1]` U32 tensor (left on
/// the device until read).
pub(super) fn argmax_last_token(logits: &Tensor) -> Result<Tensor> {
    let last = match logits.dims() {
        [1, vocab] | [1, 1, vocab] => logits.reshape((*vocab,))?,
        [1, seq, vocab] if *seq > 0 => logits.narrow(1, seq - 1, 1)?.reshape((*vocab,))?,
        dims => {
            return Err(Error::InferenceError(format!(
                "Granite Speech 5.0 logits expected [1,vocab] or [1,seq,vocab], got {dims:?}"
            )));
        }
    };
    last.argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .reshape((1, 1))
        .map_err(Error::from)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;
    use std::collections::HashMap;

    pub(crate) fn tiny_text_config() -> GraniteSpeech5TextConfig {
        GraniteSpeech5TextConfig {
            attention_multiplier: 0.5,
            bos_token_id: 1,
            dtype: Some("float32".to_string()),
            embedding_multiplier: 12.0,
            eos_token_id: 2,
            hidden_size: 8,
            intermediate_size: 16,
            logits_scaling: 8.0,
            max_position_embeddings: 64,
            model_type: Some("granite".to_string()),
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            pad_token_id: 0,
            residual_multiplier: 0.22,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: Some(serde_json::json!({
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 16
            })),
            rope_parameters: None,
            tie_word_embeddings: Some(false),
            vocab_size: 24,
        }
    }

    pub(crate) fn tiny_text_tensors(
        config: &GraniteSpeech5TextConfig,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        let (hidden, inter, vocab) = (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
        );
        let head_dim = hidden / config.num_attention_heads;
        let kv = config.num_key_value_heads * head_dim;
        let mut tensors = HashMap::new();
        let mut seed = 70;
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| {
            seed += 1;
            let tensor =
                (xorshift_tensor(shape, seed, scale, device).unwrap() + f64::from(offset)).unwrap();
            tensors.insert(name, tensor);
        };
        put(
            "model.embed_tokens.weight".into(),
            &[vocab, hidden],
            0.5,
            0.0,
        );
        put("lm_head.weight".into(), &[vocab, hidden], 0.5, 0.0);
        put("model.norm.weight".into(), &[hidden], 0.1, 1.0);
        for layer in 0..config.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            put(format!("{p}.input_layernorm.weight"), &[hidden], 0.1, 1.0);
            put(
                format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
                0.1,
                1.0,
            );
            put(
                format!("{p}.self_attn.q_proj.weight"),
                &[hidden, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.self_attn.k_proj.weight"),
                &[kv, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.self_attn.v_proj.weight"),
                &[kv, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.self_attn.o_proj.weight"),
                &[hidden, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.mlp.gate_proj.weight"),
                &[inter, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.mlp.up_proj.weight"),
                &[inter, hidden],
                0.3,
                0.0,
            );
            put(
                format!("{p}.mlp.down_proj.weight"),
                &[hidden, inter],
                0.3,
                0.0,
            );
        }
        tensors
    }

    pub(crate) fn tiny_language_model(
        device: &Device,
        plan: LmExecutionPlan,
    ) -> GraniteLanguageModel {
        let config = tiny_text_config();
        let vb = VarBuilder::from_tensors(tiny_text_tensors(&config, device), DType::F32, device);
        GraniteLanguageModel::load(&config, &config.rope().unwrap(), false, vb, plan).unwrap()
    }

    pub(crate) fn tiny_cache(
        device: &Device,
        model: &GraniteLanguageModel,
        tokens: usize,
    ) -> PhysicalPagedKvCache {
        PhysicalPagedKvCache::scratch_for_probe(
            device,
            model.kv_dtype(),
            model.cfg.num_hidden_layers,
            model.cfg.num_key_value_heads,
            model.head_dim,
            tokens,
        )
        .unwrap()
    }

    #[test]
    fn rope_cache_carries_the_yarn_frequencies_and_scale() {
        let model = tiny_language_model(&Device::Cpu, LmExecutionPlan::BASE_DTYPE);
        let scale = model.rope_attention_scale();
        assert!((scale - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
        let cache = model.rope_cache(5).unwrap();
        let cos = cache.cos.to_vec2::<f32>().unwrap();
        // Position 0: cos = scale, sin = 0, for every pair.
        assert!(cos[0]
            .iter()
            .all(|value| (f64::from(*value) - scale).abs() < 1e-6));
        let sin = cache.sin.to_vec2::<f32>().unwrap();
        assert!(sin[0].iter().all(|value| *value == 0.0));
        // A slice matches a direct build at the same positions.
        let slice = cache.slice(2, 3).unwrap();
        let direct = RopeCache::build(&model.rope, 2, 3, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(
            slice.packed.to_vec2::<f32>().unwrap(),
            direct.packed.to_vec2::<f32>().unwrap()
        );
    }

    /// Chunked prefill (and prefill then token-by-token decode) must equal one
    /// full prefill: the RoPE positions and the cache cursor line up.
    #[test]
    fn decode_after_prefill_matches_a_single_full_prefill() {
        let device = Device::Cpu;
        let model = tiny_language_model(&device, LmExecutionPlan::BASE_DTYPE);
        let ids = [3u32, 7, 11, 5, 19, 2];
        let embeds = model.embeddings(&ids).unwrap();

        let mut full_cache = tiny_cache(&device, &model, 8);
        let full = model
            .forward_embeds(&embeds, 0, &mut full_cache, None, None)
            .unwrap();

        let mut stepped_cache = tiny_cache(&device, &model, 8);
        let rope = model.rope_cache(ids.len()).unwrap();
        model
            .forward_embeds(
                &embeds.narrow(1, 0, 4).unwrap(),
                0,
                &mut stepped_cache,
                Some(&rope),
                None,
            )
            .unwrap();
        model
            .forward_token(
                &Tensor::from_vec(vec![ids[4]], (1, 1), &device).unwrap(),
                4,
                &mut stepped_cache,
                &rope,
            )
            .unwrap();
        let stepped = model
            .forward_token(
                &Tensor::from_vec(vec![ids[5]], (1, 1), &device).unwrap(),
                5,
                &mut stepped_cache,
                &rope,
            )
            .unwrap();
        let full = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let stepped = stepped.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in full.iter().zip(&stepped) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn manual_rotary_matches_the_candle_kernels() {
        let device = Device::Cpu;
        let model = tiny_language_model(&device, LmExecutionPlan::BASE_DTYPE);
        let rope = model.rope_cache(5).unwrap().slice(0, 5).unwrap();
        let q = xorshift_tensor(&[1, 5, 2, 4], 3, 1.0, &device).unwrap();
        let manual = apply_rotary_emb(&q, &rope.cos, &rope.sin).unwrap();
        let thd = rotary_emb::rope_thd(
            &q,
            &rope.cos.unsqueeze(0).unwrap(),
            &rope.sin.unsqueeze(0).unwrap(),
        )
        .unwrap();
        let bhtd = rotary_emb::rope(
            &q.transpose(1, 2).unwrap().contiguous().unwrap(),
            &rope.cos,
            &rope.sin,
        )
        .unwrap()
        .transpose(1, 2)
        .unwrap();
        for other in [thd, bhtd] {
            let diff = (&manual - &other)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-5, "{diff}");
        }
    }

    #[test]
    fn untied_head_projects_with_its_own_weight_and_prescale_matches() {
        let device = Device::Cpu;
        let config = tiny_text_config();
        let tensors = tiny_text_tensors(&config, &device);
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
        let model = GraniteLanguageModel::load(
            &config,
            &config.rope().unwrap(),
            false,
            vb,
            LmExecutionPlan::BASE_DTYPE,
        )
        .unwrap();
        let LinearNoBias::Dense { linear: head, .. } = &model.lm_head else {
            panic!("a safetensors head loads dense");
        };
        assert_eq!(
            head.weight().to_vec2::<f32>().unwrap(),
            tensors["lm_head.weight"].to_vec2::<f32>().unwrap()
        );
        let linear = Linear::new(
            tensors["model.layers.0.self_attn.o_proj.weight"].clone(),
            None,
        );
        let x = xorshift_tensor(&[1, 3, 8], 9, 1.0, &device).unwrap();
        let expected = (linear.forward(&x).unwrap() * 0.22).unwrap();
        let prescaled = prescale_residual_linear(linear, 0.22, true).unwrap();
        let actual = prescaled.forward(&x).unwrap();
        let diff = (expected - actual)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn selective_f16_probe_tracks_f32_on_cpu() {
        let device = Device::Cpu;
        let base = tiny_language_model(&device, LmExecutionPlan::BASE_DTYPE);
        let f16 = tiny_language_model(
            &device,
            LmExecutionPlan {
                f16_lm_head: true,
                f16_qkv: true,
                f16_attention_core: true,
                f16_mlp: true,
                f16_attention_output: true,
            },
        );
        assert_eq!(f16.kv_dtype(), DType::F16);
        let reference = base.numeric_probe(6).unwrap();
        let probed = f16.numeric_probe(6).unwrap();
        let logits = |outputs: &[(String, Tensor)]| {
            outputs
                .last()
                .unwrap()
                .1
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for (a, b) in logits(&reference).iter().zip(logits(&probed)) {
            assert!((a - b).abs() < 5e-2, "{a} vs {b}");
        }
    }

    /// Prefill then two decode steps on `device` with `plan`, against the
    /// same weights in F32 on the CPU: the Metal policies (pair RoPE kernel
    /// over the YaRN table, fused RMSNorm, residual pre-scaling, selective
    /// F16) must track the portable path.
    fn assert_device_decode_tracks_cpu_f32(device: &Device, plan: LmExecutionPlan, tolerance: f32) {
        let config = tiny_text_config();
        let tensors = tiny_text_tensors(&config, &Device::Cpu);
        let run = |device: &Device, plan: LmExecutionPlan| {
            let device_tensors = tensors
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.to_device(device).unwrap()))
                .collect::<HashMap<_, _>>();
            let vb = VarBuilder::from_tensors(device_tensors, DType::F32, device);
            let model =
                GraniteLanguageModel::load(&config, &config.rope().unwrap(), false, vb, plan)
                    .unwrap();
            let ids = [3u32, 7, 11, 5, 19];
            let mut cache = tiny_cache(device, &model, 8);
            let rope = model.rope_cache(7).unwrap();
            let embeds = model.embeddings(&ids).unwrap();
            let mut logits = vec![model
                .forward_embeds(&embeds, 0, &mut cache, Some(&rope), None)
                .unwrap()];
            for (offset, token) in [13u32, 2].into_iter().enumerate() {
                let token = Tensor::from_vec(vec![token], (1, 1), device).unwrap();
                logits.push(
                    model
                        .forward_token(&token, ids.len() + offset, &mut cache, &rope)
                        .unwrap(),
                );
            }
            logits
                .into_iter()
                .flat_map(|logits| {
                    logits
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let reference = run(&Device::Cpu, LmExecutionPlan::BASE_DTYPE);
        let actual = run(device, plan);
        assert_eq!(reference.len(), actual.len());
        for (a, b) in reference.iter().zip(&actual) {
            let diff = (a - b).abs();
            assert!(diff.is_finite() && diff <= tolerance, "{a} vs {b}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_decode_tracks_cpu_f32() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_decode_tracks_cpu_f32")
        else {
            return;
        };
        assert_device_decode_tracks_cpu_f32(&device, LmExecutionPlan::BASE_DTYPE, 2e-3);
        assert_device_decode_tracks_cpu_f32(
            &device,
            LmExecutionPlan::for_device(&device, DType::F32),
            5e-2,
        );
    }

    #[test]
    fn argmax_reads_the_last_position_and_is_unchanged_by_logit_scaling() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.1f32, 0.9, 0.3, 2.0, -1.0, 0.5], (1, 2, 3), &device).unwrap();
        let token = argmax_last_token(&logits)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(token, 0);
        let scaled = (logits / 8.0).unwrap();
        let scaled_token = argmax_last_token(&scaled)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert_eq!(scaled_token, token);
        assert!(argmax_last_token(&Tensor::zeros((2, 3), DType::F32, &device).unwrap()).is_err());
    }
}

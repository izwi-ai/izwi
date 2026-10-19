//! Qwen3-TTS Talker model implementation.
//!
//! The talker is the main LLM component that generates speech tokens from text input.
//! It uses a Qwen3 architecture with MRoPE (Multi-modal Rotary Position Embeddings)
//! to handle both text and audio modalities.

use std::path::Path;
use std::sync::Arc;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::backends::kv::{
    submit_ordered_after_write, KvSlotMap, KvWriteArgs, KvWriteCompletionCollector,
    PagedKvDecodeArgs,
};
use crate::error::{Error, Result};
use crate::kernels::f16_weight_linear::{projection_var_builder, ProjectionLinear};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::architectures::qwen3::tts::config::TalkerConfig;
use crate::models::architectures::qwen3::tts::rope::{
    build_rope_inv_freq, build_rope_window, duplicate_rope_window, qwen_rotate_half,
};
pub use crate::models::shared::attention::physical::PhysicalPagedKvCache as TalkerPhysicalCache;
use crate::models::shared::attention::physical::PreparedPhysicalPagedStep;
use crate::models::shared::weights::mlx;

/// Multi-head attention with optional Q/K normalization
struct Attention {
    q_proj: ProjectionLinear,
    k_proj: ProjectionLinear,
    v_proj: ProjectionLinear,
    o_proj: ProjectionLinear,
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
    fn load(cfg: &TalkerConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let pvb = projection_var_builder(&vb, f16);
        let head_dim = cfg.head_dim();

        let q_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * head_dim,
                pvb.pp("q_proj"),
            )?,
            f16,
        )?;
        let k_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                pvb.pp("k_proj"),
            )?,
            f16,
        )?;
        let v_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * head_dim,
                pvb.pp("v_proj"),
            )?,
            f16,
        )?;
        let o_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(
                cfg.num_attention_heads * head_dim,
                cfg.hidden_size,
                pvb.pp("o_proj"),
            )?,
            f16,
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

    /// RoPE cos/sin for `seq_len` rows starting at each of `start_positions`
    /// (one start per batch row), shaped `[rows, seq_len, 1, head_dim]`.
    ///
    /// The TTS talker runs interleaved MRoPE with the same position on all
    /// three axes, which is plain 1-D RoPE: the tables are built on the host
    /// from the start positions (bit-identical to [`build_mrope_cache`] with
    /// equal axes) instead of uploading position ids and reading them back
    /// once per layer for q and once for k. Explicit position ids still take
    /// the MRoPE path, which reads them back once per forward.
    fn rope_tables(
        &self,
        seq_len: usize,
        start_positions: &[usize],
        position_ids: Option<&Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<TalkerRope> {
        let rows = start_positions.len();
        let (cos, sin) = match position_ids {
            Some(position_ids) if self.use_mrope && rows == 1 => build_mrope_cache(
                seq_len,
                device,
                dtype,
                position_ids,
                &self.mrope_section,
                &self.rope_inv_freq,
            )?,
            _ if self.use_mrope && self.mrope_section.len() < 3 => {
                // MRoPE without three sections counts positions from zero.
                let (cos, sin) = build_rope_window(seq_len, 0, &self.rope_inv_freq, device, dtype)?;
                (cos.repeat((rows, 1))?, sin.repeat((rows, 1))?)
            }
            _ if self.use_mrope => sequential_rope_host_tables(
                seq_len,
                start_positions,
                &self.rope_inv_freq,
                device,
                dtype,
            )?,
            _ => {
                let mut cos_rows = Vec::with_capacity(rows);
                let mut sin_rows = Vec::with_capacity(rows);
                for &start in start_positions {
                    let (cos, sin) =
                        build_rope_window(seq_len, start, &self.rope_inv_freq, device, dtype)?;
                    cos_rows.push(cos);
                    sin_rows.push(sin);
                }
                (Tensor::cat(&cos_rows, 0)?, Tensor::cat(&sin_rows, 0)?)
            }
        };
        let (cos, sin) = duplicate_rope_window(cos, sin)?;
        let head_dim = self.head_dim;
        Ok(TalkerRope {
            cos: cos.reshape((rows, seq_len, 1, head_dim))?,
            sin: sin.reshape((rows, seq_len, 1, head_dim))?,
        })
    }

    /// Qwen RoPE (rotate_half over [first half, second half]) on
    /// `[rows, seq, heads, head_dim]`.
    fn apply_rope(&self, x: Tensor, rope: &TalkerRope) -> Result<Tensor> {
        let rotated = qwen_rotate_half(&x, self.head_dim / 2)?;
        x.broadcast_mul(&rope.cos)?
            .broadcast_add(&rotated.broadcast_mul(&rope.sin)?)
            .map_err(Error::from)
    }

    fn forward_physical(
        &self,
        x: &Tensor,
        rope: &TalkerRope,
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
        q = self.apply_rope(q, rope)?;
        k = self.apply_rope(k, rope)?;

        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = k
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = v
            .reshape((seq_len, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let compute_dtype = x.dtype();
        let state_dtype = cache.arena().config().dtype;
        let q = q.to_dtype(state_dtype)?;
        let k = k.to_dtype(state_dtype)?;
        let v = v.to_dtype(state_dtype)?;
        let out = cache.write_and_attend(
            layer_idx,
            prepared,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        let out =
            out.to_dtype(compute_dtype)?
                .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }

    /// Execute one native ragged decode step for rows sharing one physical
    /// arena. Dense projections keep the real batch dimension while the paged
    /// backend consumes each row's independent block table.
    #[allow(clippy::too_many_arguments)]
    fn forward_physical_decode_batch(
        &self,
        x: &Tensor,
        start_positions: &[usize],
        rope: &TalkerRope,
        caches: &[&TalkerPhysicalCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, _) = x.dims3()?;
        if batch_size == 0
            || sequence_len != 1
            || start_positions.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker batch expects matching [batch,1,hidden] rows, got {:?}",
                x.dims()
            )));
        }
        let first = caches[0];
        if caches.iter().any(|cache| {
            !Arc::ptr_eq(cache.arena(), first.arena())
                || cache.layer_binding(layer_idx).ok() != first.layer_binding(layer_idx).ok()
        }) {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical talker batch rows must share one arena and layer binding"
                    .into(),
            ));
        }

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((batch_size, 1, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, 1)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, 1)?;

        // One table row per batch row (each at its own position).
        let queries = self
            .apply_rope(q, rope)?
            .reshape((batch_size, self.num_heads, self.head_dim))?
            .contiguous()?;
        let keys = self
            .apply_rope(k, rope)?
            .reshape((batch_size, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let values = v
            .reshape((batch_size, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        if slots.arena_id() != first.arena().id() || slots.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical talker batch received an incompatible slot map".into(),
            ));
        }
        let binding = first.layer_binding(layer_idx)?;
        let compute_dtype = x.dtype();
        let state_dtype = first.arena().config().dtype;
        let queries = queries.to_dtype(state_dtype)?;
        let keys = keys.to_dtype(state_dtype)?;
        let values = values.to_dtype(state_dtype)?;
        let completion = first.arena().write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots,
            },
        )?;
        let (out, completion) = submit_ordered_after_write(completion, || {
            first.arena().paged_decode(
                binding,
                PagedKvDecodeArgs {
                    queries: &queries,
                    batch: metadata,
                    softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    softcap: None,
                },
            )
        })?;
        completions.collect(completion)?;
        self.o_proj
            .forward(&out.to_dtype(compute_dtype)?.reshape((
                batch_size,
                1,
                self.num_heads * self.head_dim,
            ))?)
            .map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: ProjectionLinear,
    up_proj: ProjectionLinear,
    down_proj: ProjectionLinear,
}

impl Mlp {
    fn load(cfg: &TalkerConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let pvb = projection_var_builder(&vb, f16);
        let gate_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, pvb.pp("gate_proj"))?,
            f16,
        )?;
        let up_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, pvb.pp("up_proj"))?,
            f16,
        )?;
        let down_proj = ProjectionLinear::new(
            mlx::load_linear_no_bias(cfg.intermediate_size, cfg.hidden_size, pvb.pp("down_proj"))?,
            f16,
        )?;

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
    fn load(cfg: &TalkerConfig, vb: VarBuilder, f16: Option<&Device>) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"), f16)?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"), f16)?;

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
        rope: &TalkerRope,
        cache: &TalkerPhysicalCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_physical(&normed, rope, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_physical_decode_batch(
        &self,
        x: &Tensor,
        start_positions: &[usize],
        rope: &TalkerRope,
        caches: &[&TalkerPhysicalCache],
        slots: &dyn KvSlotMap,
        metadata: &KvDecodeBatchMetadata,
        completions: &mut KvWriteCompletionCollector,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_physical_decode_batch(
            &normed,
            start_positions,
            rope,
            caches,
            slots,
            metadata,
            completions,
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
    linear_fc1: ProjectionLinear,
    linear_fc2: ProjectionLinear,
}

impl TextProjection {
    fn load(
        text_hidden_size: usize,
        hidden_size: usize,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let pvb = projection_var_builder(&vb, f16);
        let linear_fc1 = ProjectionLinear::new(
            mlx::load_linear(text_hidden_size, text_hidden_size, pvb.pp("linear_fc1"))?,
            f16,
        )?;
        let linear_fc2 = ProjectionLinear::new(
            mlx::load_linear(text_hidden_size, hidden_size, pvb.pp("linear_fc2"))?,
            f16,
        )?;
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

/// The text embedding table (`text_vocab x text_hidden`, the talker's
/// largest tensor): dense, or stored in F16 on the host with gathered rows
/// widened to F32 and uploaded (weight-only F16, like the projections). Only
/// prompt text is embedded through it, a few hundred rows per request, so
/// keeping the 0.62 GB table off the device costs one small upload.
enum TextEmbeddingTable {
    Dense(Embedding),
    /// The host F16 table and the relative L2 of the conversion on sampled
    /// rows.
    HostF16(Tensor, f64),
}

impl TextEmbeddingTable {
    /// With `f16`, a float table is read straight from `checkpoint` at F16
    /// ([`read_f16_embedding`]); an MLX-quantized one is dequantized through
    /// `vb` at F32 and converted.
    fn load(
        vocab: usize,
        hidden: usize,
        vb: VarBuilder,
        f16: Option<&Device>,
        checkpoint: &Path,
    ) -> Result<Self> {
        if f16.is_none() {
            return Ok(Self::Dense(mlx::load_embedding(vocab, hidden, vb)?));
        }
        let name = format!("{}.weight", vb.prefix());
        if let Some(read) = read_f16_embedding(checkpoint, &name, vocab, hidden)? {
            return Ok(Self::HostF16(read.table, read.relative_l2));
        }
        let host = mlx::load_embedding(vocab, hidden, projection_var_builder(&vb, f16))?;
        let table = host.embeddings().to_dtype(DType::F32)?;
        let f16_table = table.to_dtype(DType::F16)?;
        // Check the conversion against the dequantized F32 rows before they
        // are dropped.
        let rows = sampled_embedding_rows(vocab);
        let rows = Tensor::from_vec(rows.clone(), rows.len(), &Device::Cpu)?;
        let original = table
            .index_select(&rows, 0)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let converted = f16_table
            .index_select(&rows, 0)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let relative = crate::kernels::numeric_check::relative_l2(&converted, &original);
        Ok(Self::HostF16(f16_table.to_device(&Device::Cpu)?, relative))
    }

    /// The F32 rows for `ids` on `device`, `[ids.len(), hidden]`.
    fn embed(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(embedding) => {
                let ids = Tensor::from_vec(ids.to_vec(), (ids.len(),), device)?;
                embedding.forward(&ids).map_err(Error::from)
            }
            Self::HostF16(table, _) => {
                let ids = Tensor::from_vec(ids.to_vec(), (ids.len(),), &Device::Cpu)?;
                table
                    .index_select(&ids, 0)?
                    .to_dtype(DType::F32)?
                    .to_device(device)
                    .map_err(Error::from)
            }
        }
    }

    /// The F16 conversion's relative L2 on sampled rows; `None` when dense.
    fn f16_relative_l2(&self) -> Option<f64> {
        match self {
            Self::Dense(_) => None,
            Self::HostF16(_, relative) => Some(*relative),
        }
    }
}

/// The rows a weight-only F16 embedding conversion is checked on: every
/// 257th (unstructured, cheap).
fn sampled_embedding_rows(rows: usize) -> Vec<u32> {
    (0..rows as u32).step_by(257).collect()
}

/// Whether a text embedding stored at `dtype` is read in place at F16 by
/// [`read_f16_embedding`] (any other form is read through the builder at
/// F32 first). The Qwen3-TTS residency plan prices the read from this.
pub(crate) fn text_embedding_reads_in_place(dtype: DType) -> bool {
    matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
}

/// An embedding table read from its checkpoint at F16.
pub(crate) struct F16EmbeddingRead {
    /// The `[rows, cols]` F16 table, on the host.
    pub(crate) table: Tensor,
    /// The conversion's relative L2 against the stored values on sampled
    /// rows (infinite when not finite).
    pub(crate) relative_l2: f64,
    /// Host memory the read allocated: the F16 table's buffer.
    pub(crate) host_bytes: u64,
}

/// Reads `name` (`[rows, cols]`) from the memory-mapped safetensors
/// `checkpoint` straight into an F16 host table: each stored value is
/// decoded and rounded to F16 in place, so the read allocates the F16 table
/// and nothing else. Reading it through a builder copied the stored table
/// and widened it to F32 first: a 1.87 GB host peak for 1.7B's 151936 x
/// 2048 embedding against the F16 table's 0.62 GB, which kept 1.7B from
/// loading on a 16 GB Mac.
///
/// The conversion is checked on [`sampled_embedding_rows`] of the finished
/// table against F32 references decoded from the stored bytes (exact for
/// BF16, F16 and F32), one row at a time. `Ok(None)` when the tensor is not
/// stored as a float the read decodes ([`text_embedding_reads_in_place`]).
pub(crate) fn read_f16_embedding(
    checkpoint: &Path,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Option<F16EmbeddingRead>> {
    // SAFETY: as every safetensors load here, the checkpoint must not be
    // modified while mapped.
    let mapped = unsafe { MmapedSafetensors::new(checkpoint)? };
    let view = mapped.get(name)?;
    let dtype = match DType::try_from(view.dtype()) {
        Ok(dtype) if text_embedding_reads_in_place(dtype) => dtype,
        _ => return Ok(None),
    };
    let width = dtype.size_in_bytes();
    let data = view.data();
    if view.shape() != [rows, cols] || data.len() != rows * cols * width {
        return Err(Error::ModelLoadError(format!(
            "{name}: expected [{rows}, {cols}] of {dtype:?}, the checkpoint holds {:?} ({} bytes)",
            view.shape(),
            data.len()
        )));
    }
    let decode: fn(&[u8]) -> f32 = match dtype {
        DType::BF16 => |b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        DType::F16 => |b| half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
        _ => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    };
    let mut values = Vec::with_capacity(rows * cols);
    values.extend(
        data.chunks_exact(width)
            .map(|bytes| half::f16::from_f32(decode(bytes))),
    );
    let host_bytes = (values.capacity() * std::mem::size_of::<half::f16>()) as u64;
    let table = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;

    let (mut error, mut norm) = (0f64, 0f64);
    for row in sampled_embedding_rows(rows) {
        let row = row as usize;
        let converted = table.i(row)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let stored = &data[row * cols * width..(row + 1) * cols * width];
        for (&value, bytes) in converted.iter().zip(stored.chunks_exact(width)) {
            let reference = f64::from(decode(bytes));
            let diff = f64::from(value) - reference;
            error += diff * diff;
            norm += reference * reference;
        }
    }
    let relative = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    Ok(Some(F16EmbeddingRead {
        table,
        relative_l2: if relative.is_finite() {
            relative
        } else {
            f64::INFINITY
        },
        host_bytes,
    }))
}

/// Qwen3-TTS Talker model
pub struct TalkerModel {
    text_embedding: TextEmbeddingTable,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    cfg: TalkerConfig,
    use_mrope: bool,
}

/// Normalized hidden states and semantic logits produced by one native talker
/// batch. Both tensors retain `[batch, 1, width]` row order.
pub struct TalkerPhysicalBatchOutput {
    pub hidden_states: Tensor,
    pub logits: Tensor,
}

impl TalkerModel {
    /// Load the talker. With `f16`, its projections and text embedding are
    /// stored in F16 on that device (activations stay F32); the embedding
    /// is read from `checkpoint`, the safetensors file `vb` reads.
    pub fn load(
        cfg: TalkerConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
        checkpoint: &Path,
    ) -> Result<Self> {
        let text_embedding = TextEmbeddingTable::load(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb.pp("model.text_embedding"),
            f16,
            checkpoint,
        )?;
        let text_projection = TextProjection::load(
            cfg.text_hidden_size,
            cfg.hidden_size,
            vb.pp("text_projection"),
            f16,
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
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")), f16)?;
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

    /// Whether every layer's attention and MLP projections are stored as
    /// weight-only F16.
    pub fn projections_f16(&self) -> bool {
        !self.layers.is_empty()
            && self.layers.iter().all(|layer| {
                [
                    &layer.self_attn.q_proj,
                    &layer.self_attn.k_proj,
                    &layer.self_attn.v_proj,
                    &layer.self_attn.o_proj,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                ]
                .iter()
                .all(|projection| projection.as_f16().is_some())
            })
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.cfg
    }

    /// Checks the weight-only F16 conversions against the original F32
    /// weights: the first layer's MLP gate projection (re-read from `vb`, the
    /// talker's builder) and the text embedding (sampled at load). A check
    /// that compares converted weights with themselves cannot see a bad
    /// conversion. Returns the worst relative L2; dense weights check nothing.
    pub(crate) fn check_f16_conversion(&self, vb: &VarBuilder) -> Result<f64> {
        let mut worst = 0f64;
        if let Some(gate) = self.layers.first().and_then(|l| l.mlp.gate_proj.as_f16()) {
            let host = projection_var_builder(vb, Some(&Device::Cpu));
            let original = mlx::load_linear_no_bias(
                self.cfg.hidden_size,
                self.cfg.intermediate_size,
                host.pp("model.layers.0.mlp.gate_proj"),
            )?;
            worst = worst.max(gate.check_against(original.weight(), None, 8)?);
        }
        if let Some(relative) = self.text_embedding.f16_relative_l2() {
            worst = worst.max(if relative.is_finite() {
                relative
            } else {
                f64::INFINITY
            });
        }
        Ok(worst)
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
        let sequence_len = self.validate_physical_append(embeds, start_pos, cache)?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let execution = (|| {
            let hidden = self.forward_physical_layers_with_prepared(
                embeds,
                start_pos,
                cache,
                position_ids,
                &mut prepared,
            )?;
            let hidden = self.norm.forward(&hidden)?;
            let logits = self.lm_head.forward(&hidden)?;
            Ok((hidden, logits))
        })();
        match execution {
            Ok(output) => {
                cache.commit_prepared(prepared)?;
                Ok(output)
            }
            Err(error) => match cache.abort_prepared(prepared) {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3-TTS physical talker failed: {error}; write-fence drain also failed: {drain}"
                ))),
            },
        }
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

    /// Append one exact span of already prepared prompt embeddings.
    ///
    /// Intermediate spans avoid retaining decoder outputs. The final span
    /// returns only its last normalized hidden state and semantic logits, which
    /// are the continuation tensors required by TTS decode.
    pub fn prefill_physical_span_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &mut TalkerPhysicalCache,
        position_ids: Option<&Tensor>,
        final_span: bool,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let sequence_len = self.validate_physical_append(embeds, start_pos, cache)?;
        let mut prepared = cache.prepare_append(start_pos, sequence_len)?;
        let execution = (|| {
            let hidden = self.forward_physical_layers_with_prepared(
                embeds,
                start_pos,
                cache,
                position_ids,
                &mut prepared,
            )?;
            if !final_span {
                return Ok(None);
            }
            // RMS normalization and the codec head are token-local, so the
            // final continuation is exactly preserved by projecting only the
            // last token instead of the full final span.
            let last_hidden = hidden.i((.., sequence_len - 1..sequence_len, ..))?;
            let last_hidden = self.norm.forward(&last_hidden)?;
            let last_logits = self.lm_head.forward(&last_hidden)?;
            Ok(Some((last_hidden, last_logits)))
        })();
        match execution {
            Ok(output) => {
                cache.commit_prepared(prepared)?;
                Ok(output)
            }
            Err(error) => match cache.abort_prepared(prepared) {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::InferenceError(format!(
                    "Qwen3-TTS physical prefill span failed: {error}; write-fence drain also failed: {drain}"
                ))),
            },
        }
    }

    fn validate_physical_append(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &TalkerPhysicalCache,
    ) -> Result<usize> {
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
        cache.slots_for_append(start_pos, sequence_len)?;
        Ok(sequence_len)
    }

    fn forward_physical_layers_with_prepared(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        cache: &TalkerPhysicalCache,
        position_ids: Option<&Tensor>,
        prepared: &mut PreparedPhysicalPagedStep,
    ) -> Result<Tensor> {
        let rope = self.rope_tables(embeds.dim(1)?, &[start_pos], position_ids, embeds)?;
        let mut hidden = embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_physical(&hidden, &rope, cache, prepared, layer_idx)?;
        }
        Ok(hidden)
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

    /// Append one generation embedding for a ragged set of retained sessions.
    ///
    /// A single row intentionally uses the scalar implementation. Wider calls
    /// batch projections/MLPs and issue one paged decode operation per layer.
    /// Every row must share the same arena and layer geometry, but may have a
    /// different retained context length. Logical cursor commits are all-or-none.
    pub fn generate_physical_step_batch_with_embeds(
        &self,
        input_embeds: &Tensor,
        caches: &mut [&mut TalkerPhysicalCache],
    ) -> Result<TalkerPhysicalBatchOutput> {
        let (batch_size, sequence_len, hidden_size) = input_embeds.dims3()?;
        if batch_size == 0
            || sequence_len != 1
            || hidden_size != self.cfg.hidden_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3-TTS physical talker batch expects [batch,1,{}] and one cache per row, got {:?} and {} caches",
                self.cfg.hidden_size,
                input_embeds.dims(),
                caches.len()
            )));
        }
        if batch_size == 1 {
            let (hidden_states, logits) =
                self.generate_physical_step_with_embed(input_embeds, caches[0])?;
            return Ok(TalkerPhysicalBatchOutput {
                hidden_states,
                logits,
            });
        }

        let start_positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        for (row, cache) in caches.iter().enumerate() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim(),
            )?;
            cache.slots_for_append(start_positions[row], 1)?;
        }
        let first = &*caches[0];
        if caches
            .iter()
            .any(|cache| !Arc::ptr_eq(cache.arena(), first.arena()))
        {
            return Err(Error::InvalidInput(
                "Qwen3-TTS physical talker batch rows must share one arena".into(),
            ));
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
        let lowered = first.arena().lower_slots(&combined_slots)?;
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let checkpoints = caches
            .iter()
            .map(|cache| cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;
        let execution = (|| -> Result<(Tensor, Tensor)> {
            let rope = self.rope_tables(1, &start_positions, None, input_embeds)?;
            let mut hidden = input_embeds.clone();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let cache_refs = caches
                    .iter()
                    .map(|cache| &**cache)
                    .collect::<Vec<&TalkerPhysicalCache>>();
                hidden = layer.forward_physical_decode_batch(
                    &hidden,
                    &start_positions,
                    &rope,
                    &cache_refs,
                    lowered.as_ref(),
                    &metadata,
                    &mut completions,
                    layer_idx,
                )?;
            }
            let hidden = self.norm.forward(&hidden)?;
            let logits = self.lm_head.forward(&hidden)?;
            Ok((hidden, logits))
        })();
        let (hidden_states, logits) = match execution {
            Ok(output) => output,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "Qwen3-TTS talker batch failed: {error}; write-fence drain also failed: {drain}"
                    ))),
                };
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (committed, row) in (0..batch_size).enumerate() {
            if let Err(error) =
                caches[row].commit_shared_completion(start_positions[row], 1, completion.clone())
            {
                let mut rollback_error = None;
                for rollback_row in 0..committed {
                    if let Err(rollback) = caches[rollback_row]
                        .restore_logical_checkpoint(checkpoints[rollback_row].clone())
                    {
                        rollback_error.get_or_insert(rollback);
                    }
                }
                return if let Some(rollback) = rollback_error {
                    Err(Error::InferenceError(format!(
                        "Qwen3-TTS talker batch commit failed: {error}; rollback also failed: {rollback}"
                    )))
                } else {
                    Err(error)
                };
            }
        }
        Ok(TalkerPhysicalBatchOutput {
            hidden_states,
            logits,
        })
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
        let embeds = self.text_embedding.embed(token_ids, &self.device)?;
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

    /// One set of RoPE tables per forward, shared by every layer (all layers
    /// have the same rotary geometry).
    fn rope_tables(
        &self,
        seq_len: usize,
        start_positions: &[usize],
        position_ids: Option<&Tensor>,
        like: &Tensor,
    ) -> Result<TalkerRope> {
        let layer = self
            .layers
            .first()
            .ok_or_else(|| Error::InferenceError("Qwen3-TTS talker has no layers".into()))?;
        layer.self_attn.rope_tables(
            seq_len,
            start_positions,
            position_ids,
            like.device(),
            like.dtype(),
        )
    }
}

/// RoPE cos/sin broadcastable over `[rows, seq, heads, head_dim]`.
struct TalkerRope {
    cos: Tensor,
    sin: Tensor,
}

/// Host cos/sin `[rows * seq_len, half_dim]` for sequential positions from
/// each row's start: what [`build_mrope_cache`] computes when all three axes
/// hold the same positions (`pos as f32 * inv`, host `cos`/`sin`).
fn sequential_rope_host_tables(
    seq_len: usize,
    start_positions: &[usize],
    inv_freq: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freq.len();
    let rows = start_positions.len();
    let mut cos_data = Vec::with_capacity(rows * seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(rows * seq_len * half_dim);
    for &start in start_positions {
        for pos in start..start + seq_len {
            let pos = pos as f32;
            for &inv in inv_freq {
                let angle = pos * inv;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
    }
    let shape = (rows * seq_len, half_dim);
    Ok((
        Tensor::from_vec(cos_data, shape, device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin_data, shape, device)?.to_dtype(dtype)?,
    ))
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

    super::record_qwen3_tts_host_readback();
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
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::architectures::qwen3::tts::config::CodePredictorConfig;

    fn test_linear(output: usize, input: usize, offset: usize, device: &Device) -> Linear {
        let values = (0..output * input)
            .map(|index| {
                let value = (index.saturating_mul(7).saturating_add(offset)) % 29;
                (value as f32 - 14.0) / 32.0
            })
            .collect::<Vec<_>>();
        Linear::new(
            Tensor::from_vec(values, (output, input), device).unwrap(),
            None,
        )
    }

    pub(crate) fn tiny_talker(device: &Device) -> TalkerModel {
        tiny_talker_with_mrope(device, false)
    }

    /// The tiny talker with interleaved MRoPE (three sections over its one
    /// rotary pair), as the published talkers run.
    fn tiny_talker_with_mrope(device: &Device, use_mrope: bool) -> TalkerModel {
        let predictor = CodePredictorConfig {
            model_type: "test-predictor".into(),
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 32,
            vocab_size: 8,
            num_code_groups: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            use_cache: true,
            layer_types: Vec::new(),
            text_hidden_size: None,
        };
        let cfg = TalkerConfig {
            model_type: "test-talker".into(),
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 32,
            vocab_size: 8,
            text_vocab_size: 8,
            text_hidden_size: 4,
            num_code_groups: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            hidden_act: "silu".into(),
            use_cache: true,
            position_id_per_seconds: 13,
            rope_scaling: None,
            sliding_window: None,
            code_predictor_config: predictor,
            spk_id: HashMap::new(),
            spk_is_dialect: HashMap::new(),
            codec_bos_id: 1,
            codec_eos_token_id: 2,
            codec_think_id: 3,
            codec_nothink_id: 4,
            codec_pad_id: 5,
            codec_think_bos_id: 6,
            codec_think_eos_id: 7,
            codec_language_id: HashMap::new(),
        };
        let attention = Attention {
            q_proj: test_linear(4, 4, 1, device).into(),
            k_proj: test_linear(2, 4, 2, device).into(),
            v_proj: test_linear(2, 4, 3, device).into(),
            o_proj: test_linear(4, 4, 4, device).into(),
            q_norm: None,
            k_norm: None,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_inv_freq: build_rope_inv_freq(2, cfg.rope_theta),
            use_mrope,
            mrope_section: if use_mrope { vec![1, 0, 0] } else { Vec::new() },
        };
        let layer = Layer {
            input_layernorm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            self_attn: attention,
            post_attention_layernorm: RmsNorm::new(
                Tensor::ones(4, DType::F32, device).unwrap(),
                1e-5,
            ),
            mlp: Mlp {
                gate_proj: test_linear(8, 4, 5, device).into(),
                up_proj: test_linear(8, 4, 6, device).into(),
                down_proj: test_linear(4, 8, 7, device).into(),
            },
        };
        let embedding_values = (0..32)
            .map(|index| ((index * 5 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        TalkerModel {
            text_embedding: TextEmbeddingTable::Dense(Embedding::new(
                Tensor::from_vec(embedding_values.clone(), (8, 4), device).unwrap(),
                4,
            )),
            text_projection: TextProjection {
                linear_fc1: test_linear(4, 4, 8, device).into(),
                linear_fc2: test_linear(4, 4, 9, device).into(),
            },
            codec_embedding: Embedding::new(
                Tensor::from_vec(embedding_values, (8, 4), device).unwrap(),
                4,
            ),
            layers: vec![layer],
            norm: RmsNorm::new(Tensor::ones(4, DType::F32, device).unwrap(), 1e-5),
            lm_head: test_linear(8, 4, 10, device),
            device: device.clone(),
            cfg,
            use_mrope,
        }
    }

    pub(crate) fn test_arena(instance: u64) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        test_arena_with_dtype(instance, DType::F32)
    }

    fn test_arena_with_dtype(
        instance: u64,
        dtype: DType,
    ) -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(instance),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 24,
            growth: None,
            dtype,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .unwrap();
        (Arc::new(arena), vec![binding])
    }

    pub(crate) fn test_cache(
        arena: Arc<dyn KvArena>,
        bindings: &[KvLayerBinding],
        first_page: u32,
    ) -> TalkerPhysicalCache {
        let blocks = (first_page..first_page + 6)
            .map(|index| CacheBlockRef {
                arena: arena.id(),
                group: arena.config().group,
                index,
                slot_generation: 1,
            })
            .collect();
        TalkerPhysicalCache::new(arena, bindings.to_vec(), blocks, 0).unwrap()
    }

    pub(crate) fn embeddings(tokens: usize, seed: usize, device: &Device) -> Tensor {
        let values = (0..tokens * 4)
            .map(|index| {
                let value = (index.saturating_mul(11).saturating_add(seed)) % 31;
                (value as f32 - 15.0) / 20.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, tokens, 4), device).unwrap()
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        assert_eq!(left.dims(), right.dims());
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() < 1e-4, "{left} != {right}");
        }
    }

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

        // Accelerate's vector trig and scalar Rust trig differ by a few F32
        // ULPs. Require numerical parity on both caches, not bit identity.
        for (mrope, standard) in [(mrope_cos, standard_cos), (mrope_sin, standard_sin)] {
            let mrope = mrope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let standard = standard.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(mrope.len(), standard.len());
            for (actual, expected) in mrope.iter().zip(standard) {
                assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn resumable_embedding_prefill_matches_one_shot_last_continuation() {
        let device = Device::Cpu;
        let model = tiny_talker(&device);
        let prompt = embeddings(4, 3, &device);
        let (full_arena, full_bindings) = test_arena(701);
        let (chunk_arena, chunk_bindings) = test_arena(702);
        let mut full_cache = test_cache(full_arena, &full_bindings, 0);
        let mut chunk_cache = test_cache(chunk_arena, &chunk_bindings, 0);

        let (full_hidden, full_logits) = model
            .prefill_physical_with_embeds(&prompt, &mut full_cache, None)
            .unwrap();
        let first = prompt.narrow(1, 0, 2).unwrap();
        let second = prompt.narrow(1, 2, 2).unwrap();
        assert!(model
            .prefill_physical_span_with_embeds(&first, 0, &mut chunk_cache, None, false)
            .unwrap()
            .is_none());
        let (chunk_hidden, chunk_logits) = model
            .prefill_physical_span_with_embeds(&second, 2, &mut chunk_cache, None, true)
            .unwrap()
            .unwrap();

        assert_close(&full_hidden, &chunk_hidden);
        assert_close(&full_logits, &chunk_logits);
        assert_eq!(full_cache.context_len(), 4);
        assert_eq!(chunk_cache.context_len(), 4);
    }

    #[test]
    fn f32_talker_compute_runs_against_f16_physical_kv() {
        let device = Device::Cpu;
        let model = tiny_talker(&device);
        let prompt = embeddings(4, 13, &device);
        let (arena, bindings) = test_arena_with_dtype(703, DType::F16);
        let mut cache = test_cache(arena, &bindings, 0);

        let (hidden, logits) = model
            .prefill_physical_with_embeds(&prompt, &mut cache, None)
            .unwrap();

        assert_eq!(hidden.dtype(), DType::F32);
        assert_eq!(logits.dtype(), DType::F32);
        assert_eq!(cache.context_len(), 4);
    }

    /// QTTS-4: host sequential tables equal the MRoPE path with repeated
    /// position ids bit for bit (the published head_dim 128, theta 1e6,
    /// sections [24, 20, 20], late positions).
    #[test]
    fn sequential_host_tables_match_mrope_with_equal_axes() {
        let device = Device::Cpu;
        let inv_freq = build_rope_inv_freq(128, 1_000_000.0);
        let (seq_len, start) = (5, 4_093);
        let ids = repeated_mrope_position_ids(seq_len, start, &device).unwrap();
        let (mrope_cos, mrope_sin) =
            build_mrope_cache(seq_len, &device, DType::F32, &ids, &[24, 20, 20], &inv_freq)
                .unwrap();
        let (host_cos, host_sin) =
            sequential_rope_host_tables(seq_len, &[start], &inv_freq, &device, DType::F32).unwrap();
        for (mrope, host) in [(mrope_cos, host_cos), (mrope_sin, host_sin)] {
            let mrope = mrope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let host = host.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(mrope
                .iter()
                .zip(&host)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// QTTS-4: an MRoPE talker step reads nothing back (it read the position
    /// ids back for q and for k in every layer), and matches the same talker
    /// with plain RoPE, which equal-axis MRoPE is.
    #[test]
    fn mrope_talker_step_reads_nothing_back_and_matches_plain_rope() {
        let device = Device::Cpu;
        let mut outputs = Vec::new();
        for use_mrope in [true, false] {
            let model = tiny_talker_with_mrope(&device, use_mrope);
            let (arena, bindings) = test_arena(720 + u64::from(use_mrope));
            let mut cache = test_cache(arena, &bindings, 0);
            model
                .prefill_physical_with_embeds(&embeddings(3, 5, &device), &mut cache, None)
                .unwrap();
            let before = super::super::thread_qwen3_tts_host_readbacks();
            let output = model
                .generate_physical_step_with_embed(&embeddings(1, 9, &device), &mut cache)
                .unwrap();
            assert_eq!(super::super::thread_qwen3_tts_host_readbacks(), before);
            outputs.push(output);
        }
        assert_close(&outputs[0].0, &outputs[1].0);
        assert_close(&outputs[0].1, &outputs[1].1);
    }

    #[test]
    fn ragged_talker_batch_matches_scalar_rows_and_shares_completion() {
        for use_mrope in [false, true] {
            ragged_talker_batch_matches_scalar_rows(use_mrope);
        }
    }

    fn ragged_talker_batch_matches_scalar_rows(use_mrope: bool) {
        let device = Device::Cpu;
        let model = tiny_talker_with_mrope(&device, use_mrope);
        let (scalar_arena, scalar_bindings) = test_arena(703);
        let (batch_arena, batch_bindings) = test_arena(704);
        let mut scalar_a = test_cache(scalar_arena.clone(), &scalar_bindings, 0);
        let mut scalar_b = test_cache(scalar_arena, &scalar_bindings, 6);
        let mut batch_a = test_cache(batch_arena.clone(), &batch_bindings, 0);
        let mut batch_b = test_cache(batch_arena, &batch_bindings, 6);
        let prefix_a = embeddings(2, 5, &device);
        let prefix_b = embeddings(3, 7, &device);
        for cache in [&mut scalar_a, &mut batch_a] {
            model
                .prefill_physical_with_embeds(&prefix_a, cache, None)
                .unwrap();
            cache.take_completed_writes();
        }
        for cache in [&mut scalar_b, &mut batch_b] {
            model
                .prefill_physical_with_embeds(&prefix_b, cache, None)
                .unwrap();
            cache.take_completed_writes();
        }
        let step_a = embeddings(1, 11, &device);
        let step_b = embeddings(1, 13, &device);
        let (scalar_hidden_a, scalar_logits_a) = model
            .generate_physical_step_with_embed(&step_a, &mut scalar_a)
            .unwrap();
        let (scalar_hidden_b, scalar_logits_b) = model
            .generate_physical_step_with_embed(&step_b, &mut scalar_b)
            .unwrap();
        let inputs = Tensor::cat(&[&step_a, &step_b], 0).unwrap();
        let mut caches = [&mut batch_a, &mut batch_b];
        let batch = model
            .generate_physical_step_batch_with_embeds(&inputs, &mut caches)
            .unwrap();

        assert_close(
            &scalar_hidden_a,
            &batch.hidden_states.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_hidden_b,
            &batch.hidden_states.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_logits_a,
            &batch.logits.i(0).unwrap().unsqueeze(0).unwrap(),
        );
        assert_close(
            &scalar_logits_b,
            &batch.logits.i(1).unwrap().unsqueeze(0).unwrap(),
        );
        assert_eq!((batch_a.context_len(), batch_b.context_len()), (3, 4));
        let completion_a = batch_a.take_completed_writes();
        let completion_b = batch_b.take_completed_writes();
        assert_eq!(completion_a.len(), 1);
        assert_eq!(completion_b.len(), 1);
        assert!(Arc::ptr_eq(&completion_a[0], &completion_b[0]));
    }

    #[test]
    fn incompatible_talker_batch_leaves_every_row_cursor_unchanged() {
        let device = Device::Cpu;
        let model = tiny_talker(&device);
        let (arena_a, bindings_a) = test_arena(705);
        let (arena_b, bindings_b) = test_arena(706);
        let mut cache_a = test_cache(arena_a, &bindings_a, 0);
        let mut cache_b = test_cache(arena_b, &bindings_b, 0);
        let prefix = embeddings(2, 17, &device);
        model
            .prefill_physical_with_embeds(&prefix, &mut cache_a, None)
            .unwrap();
        model
            .prefill_physical_with_embeds(&prefix, &mut cache_b, None)
            .unwrap();
        cache_a.take_completed_writes();
        cache_b.take_completed_writes();
        let inputs = Tensor::cat(
            &[&embeddings(1, 19, &device), &embeddings(1, 23, &device)],
            0,
        )
        .unwrap();
        let mut caches = [&mut cache_a, &mut cache_b];

        assert!(model
            .generate_physical_step_batch_with_embeds(&inputs, &mut caches)
            .is_err());
        assert_eq!((cache_a.context_len(), cache_b.context_len()), (2, 2));
        assert!(cache_a.take_completed_writes().is_empty());
        assert!(cache_b.take_completed_writes().is_empty());
    }

    /// The weight-only F16 text embedding is read in place from each float
    /// form: rounded exactly as a Candle F16 cast of the stored values,
    /// allocating only the F16 table, and checked on sampled rows against
    /// the stored values (an F16 overflow on a sampled row is infinite).
    /// Other forms are left to the builder; a wrong shape is an error.
    #[test]
    fn f16_embedding_reads_each_float_form_in_place() {
        use crate::kernels::numeric_check::xorshift_tensor;
        let device = Device::Cpu;
        let (rows, cols) = (300, 24);
        let values = xorshift_tensor(&[rows, cols], 0x5EED, 0.4, &device).unwrap();
        let mut overflow = values.to_vec2::<f32>().unwrap();
        overflow[257][3] = 1e6;
        let mut tensors = HashMap::new();
        for (name, dtype) in [
            ("bf16", DType::BF16),
            ("f16", DType::F16),
            ("f32", DType::F32),
        ] {
            tensors.insert(name.to_string(), values.to_dtype(dtype).unwrap());
        }
        tensors.insert("overflow".into(), Tensor::new(overflow, &device).unwrap());
        tensors.insert("u8".into(), values.to_dtype(DType::U8).unwrap());
        let dir =
            std::env::temp_dir().join(format!("izwi-qwen3-tts-embedding-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();

        for name in ["bf16", "f16", "f32"] {
            let read = read_f16_embedding(&path, name, rows, cols)
                .unwrap()
                .unwrap();
            assert_eq!(read.table.dtype(), DType::F16, "{name}");
            assert_eq!(read.table.dims(), [rows, cols], "{name}");
            assert_eq!(read.host_bytes, (rows * cols * 2) as u64, "{name}");
            let expected = tensors[name]
                .to_dtype(DType::F32)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = read
                .table
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got, expected, "{name}");
            assert!(read.relative_l2 < 1e-3, "{name}: {}", read.relative_l2);
            if name == "f32" {
                // F32 values really are rounded, and the check sees it.
                assert!(read.relative_l2 > 0.0);
            } else {
                // A BF16 or F16 value in F16 range survives exactly but
                // for the rare subnormal.
                assert!(read.relative_l2 < 1e-5, "{name}: {}", read.relative_l2);
            }
        }
        let overflow = read_f16_embedding(&path, "overflow", rows, cols)
            .unwrap()
            .unwrap();
        assert_eq!(overflow.relative_l2, f64::INFINITY);
        assert!(read_f16_embedding(&path, "u8", rows, cols)
            .unwrap()
            .is_none());
        assert!(read_f16_embedding(&path, "bf16", rows - 1, cols).is_err());
        assert!(!text_embedding_reads_in_place(DType::U32));
        std::fs::remove_dir_all(&dir).ok();
    }
}

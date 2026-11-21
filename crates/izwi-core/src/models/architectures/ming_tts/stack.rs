//! Ming's Qwen2 transformer stacks (`Qwen2Model`: GQA 14/2, head 64,
//! half-split RoPE at theta 1e6, biased q/k/v, SwiGLU), family-local.
//!
//! The same layer runs two ways:
//!
//! * [`MingTrunk`]: the language model on izwi's shared paged attention.
//!   One call carries rows from any requests; its inputs are embeddings,
//!   not token ids (the prompt splices speaker and audio rows in, and every
//!   decode step feeds an audio latent's embedding back), and it returns
//!   each row's last hidden state after the final norm.
//! * [`MingWindowStack`]: the VAE's stacks, one sequence at a time, causal
//!   with a sliding window. Each layer keeps the keys and values a later
//!   chunk can still see, so a sequence decoded in chunks is the sequence
//!   decoded at once.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

use super::config::MingStackConfig;
use crate::error::{Error, Result};
use crate::kernels::cuda::stretch_graphs::StretchGraphs;
use crate::kernels::residual_norm;
use crate::models::shared::attention::mixed::{
    MixedRowLayout, MixedRowSpan, PreparedPhysicalMixedStep,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::rope::{rope_frequencies, RopeFrequencies, RopeScaling};

/// The most decode rows one language-model forward takes.
pub(crate) const MAX_DECODE_GROUP: usize = 32;

/// RMSNorm with its statistics in F32 whatever the model dtype: `eps` 1e-6
/// is below F16's smallest normal number. The result has `x`'s dtype.
pub(crate) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let wide = x.to_dtype(DType::F32)?;
    let variance = wide.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = wide.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(normed
        .to_dtype(dtype)?
        .broadcast_mul(&weight.to_dtype(dtype)?)?)
}

/// `x` [rows, in] through `weight` [out, in] (+ `bias`), each row rounded
/// the same however many requests share the call, so a request sounds the
/// same under load as alone: Candle picks a matmul's kernel from its row
/// count and each rounds a row's sums differently. On the CPU and Metal a
/// decode-sized call runs row by row; on CUDA it is padded to
/// [`MAX_DECODE_GROUP`] rows, one shape for every batch. A longer span is
/// one request's prefill, whose shape is its own.
pub(crate) fn project_rows(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let rows = x.dim(0)?;
    let product = if rows > MAX_DECODE_GROUP {
        x.matmul(&weight.t()?)?
    } else if x.device().is_cuda() {
        let padded = x.pad_with_zeros(0, 0, MAX_DECODE_GROUP - rows)?;
        padded.matmul(&weight.t()?)?.narrow(0, 0, rows)?
    } else if rows == 1 {
        x.matmul(&weight.t()?)?
    } else {
        let weight_t = weight.t()?;
        let products = (0..rows)
            .map(|row| Ok(x.narrow(0, row, 1)?.matmul(&weight_t)?))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&products, 0)?
    };
    add_bias(product, bias)
}

/// Multiplies Ming's F32 matmuls (the VAE's; the language model's only when
/// it falls back to F32) on CUDA tensor cores in TF32 (a boolean; on when
/// unset). Read once.
pub(crate) const MING_TTS_F32_TF32_ENV: &str = "IZWI_MING_TTS_F32_TF32";

fn f32_tf32_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(MING_TTS_F32_TF32_ENV, true));
    ON.get()
}

/// `product + bias`: on CUDA in place in one launch with the same bits
/// (`bias_rows`), elsewhere Candle's broadcast add.
fn add_bias(product: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    use crate::kernels::cuda::bias_rows;
    match bias {
        Some(bias)
            if bias_rows::serves(product.device(), product.dtype()) && product.is_contiguous() =>
        {
            bias_rows::add_bias(&product, bias, false)?;
            Ok(product)
        }
        Some(bias) => Ok(product.broadcast_add(bias)?),
        None => Ok(product),
    }
}

/// `x` [rows, in] · `weight` [out, in]ᵀ (+ `bias`): an F32 product runs in
/// TF32 on CUDA tensor cores (F32 tensors and sums, the multiply's inputs
/// rounded to a 10-bit mantissa) unless `IZWI_MING_TTS_F32_TF32` is off;
/// everything else is Candle's matmul. The bias goes through [`add_bias`].
pub(crate) fn product(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    use crate::kernels::cuda::tf32_gemm;
    let product = if tf32_gemm::serves(x.device(), x.dtype())
        && weight.dtype() == DType::F32
        && f32_tf32_requested()
    {
        tf32_gemm::matmul_t(x, weight)?
    } else {
        x.matmul(&weight.t()?)?
    };
    add_bias(product, bias)
}

/// An RMSNorm gain in F32, the form the fused norm takes.
fn norm_gain(vb: VarBuilder, hidden: usize) -> Result<Tensor> {
    Ok(vb.get(hidden, "weight")?.to_dtype(DType::F32)?)
}

/// RMSNorm of `x` with an F32 `gain` in one launch, its statistics and
/// result in F32 (`residual_norm`): the input is never narrowed first, so
/// the language model's F32 residual stream, which carries values past
/// F16's range, normalizes exactly as wide as it is. The caller casts.
fn norm_f32(x: &Tensor, gain: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(residual_norm::rms_norm(x, gain, eps as f32, DType::F32)?)
}

fn project(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    product(x, weight, bias)
}

/// One layer's weights, each as the checkpoint stores it (the two norm
/// gains in F32, for the fused norm). Nothing is joined on the device: a
/// joined tensor is a device-written buffer, which Metal rounds up to a
/// power of two (the 35 MB gate-and-up pair of an F32 stack would take
/// 64 MB), and the load plan prices the stored tensors.
pub(crate) struct MingLayer {
    input_norm: Tensor,
    q: Tensor,
    q_bias: Tensor,
    k: Tensor,
    k_bias: Tensor,
    v: Tensor,
    v_bias: Tensor,
    o_proj: Tensor,
    post_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    /// The down projection, F32 under F16 weights (see [`Self::finish`]).
    down: Tensor,
}

impl MingLayer {
    fn load(cfg: &MingStackConfig, vb: &VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        let weight = |vb: VarBuilder, rows: usize, cols: usize| -> Result<Tensor> {
            Ok(vb.get((rows, cols), "weight")?)
        };
        let bias = |vb: VarBuilder, rows: usize| -> Result<Tensor> { Ok(vb.get(rows, "bias")?) };
        Ok(Self {
            input_norm: norm_gain(vb.pp("input_layernorm"), hidden)?,
            q: weight(attn.pp("q_proj"), cfg.q_out(), hidden)?,
            q_bias: bias(attn.pp("q_proj"), cfg.q_out())?,
            k: weight(attn.pp("k_proj"), cfg.kv_out(), hidden)?,
            k_bias: bias(attn.pp("k_proj"), cfg.kv_out())?,
            v: weight(attn.pp("v_proj"), cfg.kv_out(), hidden)?,
            v_bias: bias(attn.pp("v_proj"), cfg.kv_out())?,
            o_proj: weight(attn.pp("o_proj"), hidden, cfg.q_out())?,
            post_norm: norm_gain(vb.pp("post_attention_layernorm"), hidden)?,
            gate: weight(mlp.pp("gate_proj"), cfg.intermediate_size, hidden)?,
            up: weight(mlp.pp("up_proj"), cfg.intermediate_size, hidden)?,
            down: {
                let down = mlp.pp("down_proj");
                if vb.dtype() == DType::F16 {
                    down.get_with_hints_dtype(
                        (hidden, cfg.intermediate_size),
                        "weight",
                        Default::default(),
                        DType::F32,
                    )?
                } else {
                    weight(down, hidden, cfg.intermediate_size)?
                }
            },
        })
    }

    /// The dtype the layer's matmuls run in.
    fn dtype(&self) -> DType {
        self.q.dtype()
    }

    /// q [tokens, heads, dim], k and v [tokens, kv_heads, dim] in the
    /// layer's dtype, q and k rotated; `x` is the residual stream (F32).
    fn qkv(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cfg: &MingStackConfig,
        rows_independent: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let tokens = x.dim(0)?;
        let head_dim = cfg.head_dim;
        let normed = norm_f32(x, &self.input_norm, cfg.rms_norm_eps)?.to_dtype(self.dtype())?;
        let run = |weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            if rows_independent {
                project_rows(&normed, weight, Some(bias))
            } else {
                project(&normed, weight, Some(bias))
            }
        };
        // Callers build the tables in the layer's dtype; a cast is a no-op.
        let (cos, sin) = (&cos.to_dtype(self.dtype())?, &sin.to_dtype(self.dtype())?);
        let q = run(&self.q, &self.q_bias)?.reshape((tokens, cfg.num_attention_heads, head_dim))?;
        let k = run(&self.k, &self.k_bias)?.reshape((tokens, cfg.num_key_value_heads, head_dim))?;
        let v = run(&self.v, &self.v_bias)?
            .reshape((tokens, cfg.num_key_value_heads, head_dim))?
            .contiguous()?;
        Ok((rope_tokens(&q, cos, sin)?, rope_tokens(&k, cos, sin)?, v))
    }

    /// The residual branches after attention: `attn` is [tokens, q_out] in
    /// the layer's dtype, `x` and the result the residual stream (F32).
    ///
    /// The residual stream stays in F32 whatever the weights' dtype, and so
    /// do the gated product and the down projection: on the published
    /// weights layer 3's down projection writes values past F16's 65,504
    /// (its gated product already reaches 21,056 on a short prompt), a
    /// "massive activation" channel Qwen2.5 carries. In F16 they become
    /// infinities that reach every later layer and the flow head.
    fn finish(
        &self,
        x: &Tensor,
        attn: &Tensor,
        cfg: &MingStackConfig,
        rows_independent: bool,
    ) -> Result<Tensor> {
        let run = |x: &Tensor, weight: &Tensor| -> Result<Tensor> {
            if rows_independent {
                project_rows(x, weight, None)
            } else {
                project(x, weight, None)
            }
        };
        let wide = x.dtype();
        // The residual add and the post-attention norm are one launch; the
        // sum stays in the stream's dtype (F32).
        let (x, normed) = residual_norm::add_rms_norm(
            x,
            &run(attn, &self.o_proj)?,
            &self.post_norm,
            cfg.rms_norm_eps as f32,
            DType::F32,
        )?;
        let normed = normed.to_dtype(self.dtype())?;
        let down_dtype = self.down.dtype();
        let gate = run(&normed, &self.gate)?.to_dtype(down_dtype)?;
        let up = run(&normed, &self.up)?.to_dtype(down_dtype)?;
        let gated = match crate::kernels::try_fused_silu_mul(&gate, &up) {
            Some(gated) => gated,
            None => candle_nn::ops::silu(&gate)?.mul(&up)?,
        };
        let mlp = run(&gated, &self.down)?;
        Ok(x.add(&mlp.to_dtype(wide)?)?)
    }
}

/// The half-split rotation of `x` [tokens, heads, dim] by per-token
/// `cos`/`sin` [tokens, dim / 2]: Candle's fused `rope_thd` over the tokens
/// as one sequence.
fn rope_tokens(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::rotary_emb::rope_thd(
        &x.unsqueeze(0)?.contiguous()?,
        &cos.contiguous()?,
        &sin.contiguous()?,
    )?
    .squeeze(0)?)
}

/// The same rotation spelled out (the reference [`rope_tokens`] is tested
/// against).
#[cfg(test)]
fn rotate_half_split(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let first = x1.broadcast_mul(&cos)?.sub(&x2.broadcast_mul(&sin)?)?;
    let second = x1.broadcast_mul(&sin)?.add(&x2.broadcast_mul(&cos)?)?;
    Ok(Tensor::cat(&[first, second], D::Minus1)?.contiguous()?)
}

fn load_layers(cfg: &MingStackConfig, vb: &VarBuilder) -> Result<Vec<MingLayer>> {
    (0..cfg.num_hidden_layers)
        .map(|index| MingLayer::load(cfg, &vb.pp(format!("layers.{index}"))))
        .collect()
}

fn frequencies(cfg: &MingStackConfig) -> Result<RopeFrequencies> {
    rope_frequencies(cfg.head_dim, cfg.rope_theta, &RopeScaling::Default)
}

/// RoPE cos/sin for positions `start..start + len`, [len, head_dim / 2],
/// computed on the host in F64.
fn rope_table(
    freqs: &RopeFrequencies,
    start: usize,
    len: usize,
    pairs: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = freqs.cos_sin_table(start, len);
    let table = |values: Vec<f32>| -> Result<Tensor> {
        Ok(Tensor::from_vec(values, (len, pairs), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?)
    };
    Ok((table(cos)?, table(sin)?))
}

// ------------------------------------------------------------ language model

/// One row of a language-model forward: `embeds` [count, hidden] appended
/// to `cache`.
pub struct MingForwardRow<'a> {
    pub embeds: &'a Tensor,
    pub cache: &'a mut PhysicalPagedKvCache,
}

pub struct MingTrunk {
    cfg: MingStackConfig,
    dtype: DType,
    device: Device,
    /// The token table, [vocab, hidden].
    embed: Tensor,
    layers: Vec<MingLayer>,
    norm: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    max_positions: usize,
    /// On CUDA, the decode step's stretches as graphs (see
    /// [`Self::forward`]), or why there are none.
    graphs: std::result::Result<StretchGraphs, String>,
}

impl MingTrunk {
    /// Load the language model from `vb` (rooted at `model.model`) in its
    /// dtype.
    pub fn load(
        cfg: MingStackConfig,
        vocab_size: usize,
        vb: VarBuilder,
        max_positions: usize,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let embed = vb.get((vocab_size, cfg.hidden_size), "embed_tokens.weight")?;
        let layers = load_layers(&cfg, &vb)?;
        let norm = norm_gain(vb.pp("norm"), cfg.hidden_size)?;
        let (rope_cos, rope_sin) = rope_table(
            &frequencies(&cfg)?,
            0,
            max_positions,
            cfg.head_dim / 2,
            dtype,
            &device,
        )?;
        // Every CUDA decode step is padded to one width (see
        // `Self::forward`): one set of graphs, no per-row state to copy.
        let graphs = StretchGraphs::resolve(
            "Ming TTS language model",
            super::graphs::MING_TTS_DECODE_GRAPHS_ENV,
            &device,
            cfg.num_hidden_layers + 1,
        )
        .map(|graphs| graphs.with_widths(|rows| rows == MAX_DECODE_GROUP));
        Ok(Self {
            cfg,
            dtype,
            device,
            embed,
            layers,
            norm,
            rope_cos,
            rope_sin,
            max_positions,
            graphs,
        })
    }

    pub fn config(&self) -> &MingStackConfig {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The decode-step graphs' state and counters, for the diagnostics.
    pub fn decode_graphs_diagnostics(&self) -> serde_json::Value {
        match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Run decode steps eagerly (a test's reference for the graphs).
    #[cfg(test)]
    pub(crate) fn turn_off_graphs(&mut self) {
        self.graphs = Err("turned off by a test".into());
    }

    /// The embeddings of `ids`, [ids, hidden] in the model dtype.
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        if let Some(&id) = ids
            .iter()
            .find(|&&id| id as usize >= self.embed.dim(0).unwrap_or(0))
        {
            return Err(Error::InvalidInput(format!(
                "Ming token {id} is outside the vocabulary"
            )));
        }
        let ids = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        Ok(self.embed.index_select(&ids, 0)?)
    }

    /// Requests' rows as the engine batches them: each row's last hidden
    /// state after the final norm, [rows, hidden] F32 in row order, the
    /// bits each row gets alone. A prefill row runs by itself (its span
    /// alone sets its matmul shapes); single-position decode rows share
    /// forwards in groups of at most [`MAX_DECODE_GROUP`].
    pub(crate) fn forward_batch(&self, rows: &mut [MingForwardRow<'_>]) -> Result<Tensor> {
        let mut decode = true;
        for row in rows.iter() {
            decode &= row.embeds.dim(0)? == 1;
        }
        let group = if decode { MAX_DECODE_GROUP } else { 1 };
        let mut parts = rows
            .chunks_mut(group)
            .map(|chunk| self.forward(chunk))
            .collect::<Result<Vec<_>>>()?;
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one forward"));
        }
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Run `rows` through the model and return each row's final-norm hidden
    /// state at its last position, [rows, hidden] in row order (F32).
    pub fn forward(&self, rows: &mut [MingForwardRow<'_>]) -> Result<Tensor> {
        if rows.is_empty() {
            return Err(Error::InvalidInput("Ming forward has no rows".into()));
        }
        let cfg = &self.cfg;
        let mut spans = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let (count, width) = row.embeds.dims2()?;
            let start = row.cache.context_len();
            let end = start.saturating_add(count);
            if count == 0
                || width != cfg.hidden_size
                || end > self.max_positions
                || end > row.cache.capacity_tokens()
            {
                return Err(Error::InvalidInput(format!(
                    "Ming row of {count} positions at {start} exceeds its capacity ({} positions, cache {})",
                    self.max_positions,
                    row.cache.capacity_tokens()
                )));
            }
            row.cache.validate_model(
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )?;
            spans.push(MixedRowSpan {
                start_pos: start,
                q_len: count,
            });
        }

        // Rows grouped by arena, decode rows first within each group; the
        // flattened position order is group after group.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for index in 0..rows.len() {
            let arena = rows[index].cache.arena();
            match groups
                .iter_mut()
                .find(|group| Arc::ptr_eq(rows[group[0]].cache.arena(), arena))
            {
                Some(group) => group.push(index),
                None => groups.push(vec![index]),
            }
        }
        for group in &mut groups {
            group.sort_by_key(|&index| spans[index].q_len != 1);
        }
        let order = groups.iter().flatten().copied().collect::<Vec<_>>();

        let mut steps = Vec::with_capacity(groups.len());
        let mut group_tokens = Vec::with_capacity(groups.len());
        for group in &groups {
            let caches = group.iter().map(|&i| &*rows[i].cache).collect::<Vec<_>>();
            let decode_rows = group.iter().filter(|&&i| spans[i].q_len == 1).count();
            let layout =
                MixedRowLayout::new(group.iter().map(|&i| spans[i]).collect(), decode_rows)?;
            group_tokens.push(layout.total_tokens());
            steps.push(PreparedPhysicalMixedStep::new(&caches, layout)?);
        }

        let result = (|| -> Result<Tensor> {
            let inputs = order
                .iter()
                .map(|&i| rows[i].embeds.to_dtype(DType::F32))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let mut hidden = if inputs.len() == 1 {
                inputs.into_iter().next().expect("one row")
            } else {
                Tensor::cat(&inputs, 0)?
            };
            let rows_independent = hidden.dim(0)? <= MAX_DECODE_GROUP;
            let positions = order
                .iter()
                .flat_map(|&i| {
                    let span = spans[i];
                    (span.start_pos..span.start_pos + span.q_len).map(|p| p as u32)
                })
                .collect::<Vec<_>>();
            let positions = Tensor::from_vec(positions.clone(), positions.len(), &self.device)?;
            let cos = self.rope_cos.index_select(&positions, 0)?;
            let sin = self.rope_sin.index_select(&positions, 0)?;
            let caches = groups
                .iter()
                .map(|group| group.iter().map(|&i| &*rows[i].cache).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            let tokens = hidden.dim(0)?;
            // The layers run as `layers + 1` stretches around the paged
            // attention: the first layer's norm, projections and rotation;
            // then each layer's residual branches with the next layer's
            // projections; then the last layer's residual branches. On CUDA
            // a decode step (one position a row, at most
            // `MAX_DECODE_GROUP` rows) replays each stretch as a graph,
            // captured and checked bit for bit against the eager stretch
            // first; elsewhere the stretches run as they are, in the same
            // order.
            //
            // A graphed step's stretch inputs are padded with zero rows to
            // `MAX_DECODE_GROUP`, one shape for every width. The bits are
            // the unpadded step's: the projections already run at that
            // padded row count (`project_rows`), every other stretch op
            // keeps rows apart, and attention reads the real rows only.
            let decode_step = spans.iter().all(|span| span.q_len == 1);
            let mut run = match &self.graphs {
                Ok(graphs) if decode_step && tokens <= MAX_DECODE_GROUP => {
                    graphs.begin(MAX_DECODE_GROUP)
                }
                _ => None,
            };
            let width = if run.is_some() {
                MAX_DECODE_GROUP
            } else {
                tokens
            };
            let pad = |t: Tensor| -> Result<Tensor> {
                if width > tokens {
                    Ok(t.pad_with_zeros(0, 0, width - tokens)?)
                } else {
                    Ok(t)
                }
            };
            let (cos, sin) = (pad(cos)?, pad(sin)?);
            hidden = pad(hidden)?;
            let mut stretch = |index: usize,
                               inputs: Vec<Tensor>,
                               body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>|
             -> Result<Vec<Tensor>> {
                match run.as_mut() {
                    Some(run) => run.run(index, &inputs, body),
                    None => body(&inputs),
                }
            };
            let first = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                let (q, k, v) = self.layers[0].qkv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    cfg,
                    rows_independent,
                )?;
                Ok(vec![q, k, v])
            };
            let mut qkv = stretch(0, vec![hidden.clone(), cos.clone(), sin.clone()], &first)?;
            let last_layer = self.layers.len() - 1;
            for index in 0..self.layers.len() {
                let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
                let mut outputs = Vec::with_capacity(steps.len());
                let mut offset = 0usize;
                for ((step, group), &count) in steps.iter_mut().zip(&caches).zip(&group_tokens) {
                    let slice = |t: &Tensor| t.narrow(0, offset, count)?.contiguous();
                    outputs.push(step.write_and_attend(
                        group,
                        index,
                        &slice(q)?,
                        &slice(k)?,
                        &slice(v)?,
                        scale,
                        None,
                        None,
                    )?);
                    offset += count;
                }
                let attn = pad(if outputs.len() == 1 {
                    outputs.pop().expect("one group")
                } else {
                    Tensor::cat(&outputs, 0)?
                }
                .reshape((tokens, cfg.q_out()))?)?;
                let layer = &self.layers[index];
                if index == last_layer {
                    let tail = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                        Ok(vec![layer.finish(
                            &inputs[0],
                            &inputs[1],
                            cfg,
                            rows_independent,
                        )?])
                    };
                    hidden = stretch(index + 1, vec![hidden, attn], &tail)?.remove(0);
                    break;
                }
                let next = &self.layers[index + 1];
                let between = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                    let x = layer.finish(&inputs[0], &inputs[1], cfg, rows_independent)?;
                    let (q, k, v) = next.qkv(&x, &inputs[2], &inputs[3], cfg, rows_independent)?;
                    Ok(vec![x, q, k, v])
                };
                let mut outputs = stretch(
                    index + 1,
                    vec![hidden, attn, cos.clone(), sin.clone()],
                    &between,
                )?;
                hidden = outputs.remove(0);
                qkv = outputs;
            }
            // Last position of each row, back in row order.
            let mut offsets = vec![0usize; rows.len()];
            let mut offset = 0usize;
            for &i in &order {
                offset += spans[i].q_len;
                offsets[i] = offset - 1;
            }
            let last = Tensor::from_vec(
                offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
                offsets.len(),
                &self.device,
            )?;
            let tails = hidden.index_select(&last, 0)?;
            norm_f32(&tails, &self.norm, cfg.rms_norm_eps)
        })();

        let outcome = result
            .as_ref()
            .map(|_| ())
            .map_err(|error| error.to_string());
        let mut slots = rows
            .iter_mut()
            .map(|row| Some(&mut *row.cache))
            .collect::<Vec<_>>();
        let mut finish_error = None;
        for (group, step) in groups.iter().zip(steps) {
            let mut caches = group
                .iter()
                .map(|&i| slots[i].take().expect("each row belongs to one group"))
                .collect::<Vec<_>>();
            let status = outcome.clone().map_err(Error::InferenceError);
            if let Err(error) = step.finish(&mut caches, status) {
                finish_error.get_or_insert(error);
            }
        }
        let hidden = result?;
        if let Some(error) = finish_error {
            return Err(error);
        }
        Ok(hidden)
    }
}

// ------------------------------------------------------------- VAE stacks

/// The keys and values of one sequence's recent positions, per layer: what
/// a later chunk can still attend to.
#[derive(Clone, Default)]
pub struct MingWindowState {
    /// Positions consumed so far.
    position: usize,
    /// Per layer: (k, v), each [kept, kv_heads, head_dim].
    kept: Vec<Option<(Tensor, Tensor)>>,
    /// The fixed-shape path's keys and values ([`MingWindowStack::fixed_forward`]):
    /// every layer's in one tensor, [layers, 2, window_keys - 1, kv_heads,
    /// head_dim], the valid ones (the last `min(position, window_keys - 1)`)
    /// at the end.
    fixed: Option<Tensor>,
}

impl MingWindowState {
    pub fn position(&self) -> usize {
        self.position
    }

    /// The fixed-shape path's keys and values for the next chunk, taken out
    /// (fresh zeros from `stack` before the first).
    pub(crate) fn take_fixed(&mut self, stack: &MingWindowStack) -> Result<Tensor> {
        match self.fixed.take() {
            Some(kept) => Ok(kept),
            None => stack.fixed_empty(),
        }
    }

    /// A fixed-shape chunk of `count` positions ran and keeps `kept`.
    pub(crate) fn advance_fixed(&mut self, count: usize, kept: Tensor) {
        self.position += count;
        self.fixed = Some(kept);
    }
}

/// A causal Qwen2 stack with a sliding attention window, for one sequence.
pub struct MingWindowStack {
    cfg: MingStackConfig,
    dtype: DType,
    device: Device,
    layers: Vec<MingLayer>,
    norm: Tensor,
    freqs: RopeFrequencies,
    /// Keys a position attends to, itself included.
    window_keys: usize,
}

/// Positions one attention call takes: bounds the score tensor of a long
/// sequence run at once.
const WINDOW_CHUNK: usize = 256;

impl MingWindowStack {
    pub fn load(cfg: MingStackConfig, vb: VarBuilder, window_keys: usize) -> Result<Self> {
        if window_keys < 2 {
            return Err(Error::ModelLoadError(
                "Ming VAE attention window must keep at least two keys".into(),
            ));
        }
        Ok(Self {
            dtype: vb.dtype(),
            device: vb.device().clone(),
            layers: load_layers(&cfg, &vb)?,
            norm: norm_gain(vb.pp("norm"), cfg.hidden_size)?,
            freqs: frequencies(&cfg)?,
            cfg,
            window_keys,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &MingStackConfig {
        &self.cfg
    }

    pub fn window_keys(&self) -> usize {
        self.window_keys
    }

    /// Run `x` [positions, hidden] as the next positions of `state`'s
    /// sequence; the final-norm hidden states, [positions, hidden].
    pub fn forward(&self, x: &Tensor, state: &mut MingWindowState) -> Result<Tensor> {
        let (total, width) = x.dims2()?;
        if total == 0 || width != self.cfg.hidden_size {
            return Err(Error::InvalidInput(
                "Ming VAE stack input has the wrong shape".into(),
            ));
        }
        if state.kept.is_empty() {
            state.kept = vec![None; self.layers.len()];
        }
        let mut outputs = Vec::with_capacity(total.div_ceil(WINDOW_CHUNK));
        let mut offset = 0usize;
        while offset < total {
            let count = WINDOW_CHUNK.min(total - offset);
            outputs.push(self.forward_chunk(&x.narrow(0, offset, count)?, state)?);
            offset += count;
        }
        if outputs.len() == 1 {
            return Ok(outputs.pop().expect("one chunk"));
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    fn forward_chunk(&self, x: &Tensor, state: &mut MingWindowState) -> Result<Tensor> {
        let cfg = &self.cfg;
        let count = x.dim(0)?;
        let start = state.position;
        let (cos, sin) = rope_table(
            &self.freqs,
            start,
            count,
            cfg.head_dim / 2,
            self.dtype,
            &self.device,
        )?;
        let kept = state.kept[0]
            .as_ref()
            .map_or(0, |(k, _)| k.dim(0).unwrap_or(0));
        let mask = self.window_mask(kept, count)?;
        let mut hidden = x.to_dtype(DType::F32)?;
        for (layer, slot) in self.layers.iter().zip(state.kept.iter_mut()) {
            let (q, k, v) = layer.qkv(&hidden, &cos, &sin, cfg, false)?;
            let (keys, values) = match slot.take() {
                Some((kept_k, kept_v)) => (
                    Tensor::cat(&[&kept_k, &k], 0)?,
                    Tensor::cat(&[&kept_v, &v], 0)?,
                ),
                None => (k, v),
            };
            let span = keys.dim(0)?;
            let attn = self.window_attention(&q, &keys, &values, &mask)?;
            hidden = layer.finish(&hidden, &attn, cfg, false)?;
            // Later positions see at most `window_keys - 1` of these.
            let keep = span.min(self.window_keys - 1);
            *slot = Some((
                keys.narrow(0, span - keep, keep)?.contiguous()?,
                values.narrow(0, span - keep, keep)?.contiguous()?,
            ));
        }
        state.position = start + count;
        norm_f32(&hidden, &self.norm, cfg.rms_norm_eps)
    }

    /// Attention of `q` [count, heads, dim] over `keys` and `values` [span,
    /// kv_heads, dim] under the additive `mask` [count, span] (F32):
    /// [count, q_out] in the stack's dtype.
    fn window_attention(
        &self,
        q: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let count = q.dim(0)?;
        let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        // q [kv, groups, count, dim] against k, v [kv, 1, span, dim]:
        // query head h reads KV head h / groups.
        let q = q
            .reshape((count, cfg.num_key_value_heads, groups, cfg.head_dim))?
            .permute((1, 2, 0, 3))?
            .contiguous()?;
        let k = keys.permute((1, 0, 2))?.unsqueeze(1)?.contiguous()?;
        let v = values.permute((1, 0, 2))?.unsqueeze(1)?.contiguous()?;
        let scores = (q.broadcast_matmul(&k.transpose(2, 3)?)? * scale)?
            .to_dtype(DType::F32)?
            .broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(self.dtype)?;
        Ok(weights
            .broadcast_matmul(&v)?
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .reshape((count, cfg.q_out()))?)
    }

    /// Keys each layer keeps between fixed-shape chunks: all a later
    /// position can still see.
    pub(crate) fn fixed_kept(&self) -> usize {
        self.window_keys - 1
    }

    /// Columns of [`Self::fixed_tables`] for `queries` query slots: cos,
    /// sin, then the mask over the kept keys and the slots.
    pub(crate) fn fixed_table_width(&self, queries: usize) -> usize {
        self.cfg.head_dim + self.fixed_kept() + queries
    }

    /// The host-built inputs of a fixed-shape chunk ([`Self::fixed_forward`])
    /// of `count` positions from `start` in `queries` slots, [queries,
    /// [`Self::fixed_table_width`]] row-major F32: each slot's RoPE cos and
    /// sin (zeros past `count`), then its additive mask over the keys
    /// [kept | slots]. Key `j` holds position `start - kept + j`, so slot `s`
    /// sees keys `s..=kept + s` (its window) that hold a position (the last
    /// `min(start, kept)` kept ones, and the slots up to its own). A padding
    /// slot sees only itself, so its softmax stays finite; no real slot sees
    /// a padding slot, which comes after it.
    pub(crate) fn fixed_tables(&self, start: usize, count: usize, queries: usize) -> Vec<f32> {
        let pairs = self.cfg.head_dim / 2;
        let kept = self.fixed_kept();
        let span = kept + queries;
        let width = self.fixed_table_width(queries);
        let (cos, sin) = self.freqs.cos_sin_table(start, count);
        let held = start.min(kept);
        let mut table = vec![0f32; queries * width];
        for (slot, row) in table.chunks_exact_mut(width).enumerate() {
            let (rope, mask) = row.split_at_mut(2 * pairs);
            mask[..span].fill(f32::NEG_INFINITY);
            if slot < count {
                rope[..pairs].copy_from_slice(&cos[slot * pairs..(slot + 1) * pairs]);
                rope[pairs..].copy_from_slice(&sin[slot * pairs..(slot + 1) * pairs]);
                mask[slot.max(kept - held)..=kept + slot].fill(0.0);
            } else {
                mask[kept + slot] = 0.0;
            }
        }
        table
    }

    /// One chunk of `state`'s sequence at a fixed shape, whatever its length
    /// and position: the CUDA VAE decoder's form, which one CUDA graph runs.
    /// `x` [queries, hidden] holds the chunk's positions then zero padding;
    /// `tables` is [`Self::fixed_tables`] on the device; `gather` [kept]
    /// U32 picks the keys the next chunk keeps (`count..count + kept`);
    /// `kept` is the state's fixed keys and values. Returns the final-norm
    /// hidden states [queries, hidden] (the padding rows are not meaningful)
    /// and the keys and values to keep. A row's bits depend only on its own
    /// sequence, never on what shares a dispatch.
    pub(crate) fn fixed_forward(
        &self,
        x: &Tensor,
        tables: &Tensor,
        gather: &Tensor,
        kept: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let queries = x.dim(0)?;
        let pairs = cfg.head_dim / 2;
        let span = self.fixed_kept() + queries;
        let cos = tables
            .narrow(1, 0, pairs)?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let sin = tables
            .narrow(1, pairs, pairs)?
            .contiguous()?
            .to_dtype(self.dtype)?;
        let mask = tables.narrow(1, 2 * pairs, span)?.contiguous()?;
        let mut hidden = x.to_dtype(DType::F32)?;
        let mut keep = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let (q, k, v) = layer.qkv(&hidden, &cos, &sin, cfg, false)?;
            let held = kept.get(index)?;
            let keys = Tensor::cat(&[&held.get(0)?, &k], 0)?;
            let values = Tensor::cat(&[&held.get(1)?, &v], 0)?;
            let attn = self.window_attention(&q, &keys, &values, &mask)?;
            hidden = layer.finish(&hidden, &attn, cfg, false)?;
            keep.push(Tensor::stack(
                &[
                    keys.index_select(gather, 0)?,
                    values.index_select(gather, 0)?,
                ],
                0,
            )?);
        }
        Ok((
            norm_f32(&hidden, &self.norm, cfg.rms_norm_eps)?,
            Tensor::stack(&keep, 0)?,
        ))
    }

    /// Fresh fixed keys and values for a sequence's first chunk (zeros, all
    /// masked).
    pub(crate) fn fixed_empty(&self) -> Result<Tensor> {
        let cfg = &self.cfg;
        Ok(Tensor::zeros(
            (
                self.layers.len(),
                2,
                self.fixed_kept(),
                cfg.num_key_value_heads,
                cfg.head_dim,
            ),
            self.dtype,
            &self.device,
        )?)
    }

    /// [`Self::forward`] of one chunk from position 0, reporting for each
    /// layer the largest magnitude of its attention input, attention
    /// output, gate, up, gated product, MLP output and residual stream (a
    /// diagnostic for precision limits).
    #[cfg(test)]
    pub(crate) fn trace_layers(&self, x: &Tensor) -> Result<Vec<[f32; 7]>> {
        let cfg = &self.cfg;
        let count = x.dim(0)?;
        let max = |t: &Tensor| -> Result<f32> {
            Ok(t.to_dtype(DType::F32)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?)
        };
        let (cos, sin) = rope_table(
            &self.freqs,
            0,
            count,
            cfg.head_dim / 2,
            self.dtype,
            &self.device,
        )?;
        let mask = self.window_mask(0, count)?;
        let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let mut hidden = x.to_dtype(DType::F32)?;
        let mut rows = Vec::new();
        for layer in &self.layers {
            let (q, k, v) = layer.qkv(&hidden, &cos, &sin, cfg, false)?;
            let q = q
                .reshape((count, cfg.num_key_value_heads, groups, cfg.head_dim))?
                .permute((1, 2, 0, 3))?
                .contiguous()?;
            let k = k.permute((1, 0, 2))?.unsqueeze(1)?.contiguous()?;
            let v = v.permute((1, 0, 2))?.unsqueeze(1)?.contiguous()?;
            let scores = (q.broadcast_matmul(&k.transpose(2, 3)?)? * scale)?
                .to_dtype(DType::F32)?
                .broadcast_add(&mask)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(self.dtype)?;
            let attn = weights
                .broadcast_matmul(&v)?
                .permute((2, 0, 1, 3))?
                .contiguous()?
                .reshape((count, cfg.q_out()))?;
            let attn_in = max(&q)?;
            let o = project(&attn, &layer.o_proj, None)?;
            let mid = hidden.add(&o.to_dtype(DType::F32)?)?;
            let normed =
                rms_norm(&mid, &layer.post_norm, cfg.rms_norm_eps)?.to_dtype(self.dtype)?;
            let down_dtype = layer.down.dtype();
            let gate = project(&normed, &layer.gate, None)?.to_dtype(down_dtype)?;
            let up = project(&normed, &layer.up, None)?.to_dtype(down_dtype)?;
            let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
            let mlp = project(&act, &layer.down, None)?;
            hidden = mid.add(&mlp.to_dtype(DType::F32)?)?;
            rows.push([
                attn_in,
                max(&o)?,
                max(&gate)?,
                max(&up)?,
                max(&act)?,
                max(&mlp)?,
                max(&hidden)?,
            ]);
        }
        Ok(rows)
    }

    /// The additive mask of `count` new positions over `kept` earlier keys
    /// and themselves, [count, kept + count] F32: a query sees keys at or
    /// before it and fewer than `window_keys` positions back.
    fn window_mask(&self, kept: usize, count: usize) -> Result<Tensor> {
        let span = kept + count;
        let mut mask = vec![f32::NEG_INFINITY; count * span];
        for query in 0..count {
            let position = kept + query;
            let first = (position + 1).saturating_sub(self.window_keys);
            mask[query * span + first..=query * span + position].fill(0.0);
        }
        Ok(Tensor::from_vec(mask, (count, span), &self.device)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::models::shared::noise::SplitMix64;

    pub(crate) fn tiny_stack_config(layers: usize) -> MingStackConfig {
        MingStackConfig {
            hidden_size: 32,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 48,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    pub(crate) fn random(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> Tensor {
        let count = shape.iter().product::<usize>();
        let values = rng
            .standard_normals(count)
            .into_iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    /// Random weights of a stack under `prefix` (`"x."` or `""`).
    pub(crate) fn tiny_stack_weights(
        cfg: &MingStackConfig,
        prefix: &str,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        let hidden = cfg.hidden_size;
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| {
            let tensor = (random(rng, shape, scale) + offset as f64).unwrap();
            tensors.insert(format!("{prefix}{name}"), tensor);
        };
        put("norm.weight".into(), &[hidden], 0.05, 1.0);
        for layer in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("layers.{layer}.{suffix}");
            put(name("input_layernorm.weight"), &[hidden], 0.05, 1.0);
            put(
                name("post_attention_layernorm.weight"),
                &[hidden],
                0.05,
                1.0,
            );
            put(
                name("self_attn.q_proj.weight"),
                &[cfg.q_out(), hidden],
                0.2,
                0.0,
            );
            put(name("self_attn.q_proj.bias"), &[cfg.q_out()], 0.1, 0.0);
            put(
                name("self_attn.k_proj.weight"),
                &[cfg.kv_out(), hidden],
                0.2,
                0.0,
            );
            put(name("self_attn.k_proj.bias"), &[cfg.kv_out()], 0.1, 0.0);
            put(
                name("self_attn.v_proj.weight"),
                &[cfg.kv_out(), hidden],
                0.2,
                0.0,
            );
            put(name("self_attn.v_proj.bias"), &[cfg.kv_out()], 0.1, 0.0);
            put(
                name("self_attn.o_proj.weight"),
                &[hidden, cfg.q_out()],
                0.2,
                0.0,
            );
            put(
                name("mlp.gate_proj.weight"),
                &[cfg.intermediate_size, hidden],
                0.2,
                0.0,
            );
            put(
                name("mlp.up_proj.weight"),
                &[cfg.intermediate_size, hidden],
                0.2,
                0.0,
            );
            put(
                name("mlp.down_proj.weight"),
                &[hidden, cfg.intermediate_size],
                0.2,
                0.0,
            );
        }
    }

    fn host(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    pub(crate) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if d.is_finite() {
                    d
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    /// An independent dense reference of the stack: full causal attention
    /// with an explicit window mask, one position at a time semantics.
    fn dense_reference(
        cfg: &MingStackConfig,
        tensors: &HashMap<String, Tensor>,
        x: &Tensor,
        window_keys: Option<usize>,
    ) -> Tensor {
        let get = |name: &str| tensors[name].clone();
        let total = x.dim(0).unwrap();
        let freqs = frequencies(cfg).unwrap();
        let (cos, sin) =
            rope_table(&freqs, 0, total, cfg.head_dim / 2, DType::F32, &Device::Cpu).unwrap();
        let groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        let mut mask = vec![f32::NEG_INFINITY; total * total];
        for query in 0..total {
            for key in 0..=query {
                if window_keys.is_none_or(|keys| query - key < keys) {
                    mask[query * total + key] = 0.0;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (total, total), &Device::Cpu).unwrap();
        let mut hidden = x.clone();
        for layer in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("layers.{layer}.{suffix}");
            let lin = |x: &Tensor, w: &str, b: Option<&str>| {
                let y = x.matmul(&get(&name(w)).t().unwrap()).unwrap();
                match b {
                    Some(b) => y.broadcast_add(&get(&name(b))).unwrap(),
                    None => y,
                }
            };
            let normed = rms_norm(&hidden, &get(&name("input_layernorm.weight")), 1e-6).unwrap();
            let q = lin(
                &normed,
                "self_attn.q_proj.weight",
                Some("self_attn.q_proj.bias"),
            )
            .reshape((total, cfg.num_attention_heads, cfg.head_dim))
            .unwrap();
            let k = lin(
                &normed,
                "self_attn.k_proj.weight",
                Some("self_attn.k_proj.bias"),
            )
            .reshape((total, cfg.num_key_value_heads, cfg.head_dim))
            .unwrap();
            let v = lin(
                &normed,
                "self_attn.v_proj.weight",
                Some("self_attn.v_proj.bias"),
            )
            .reshape((total, cfg.num_key_value_heads, cfg.head_dim))
            .unwrap();
            let q = rotate_half_split(&q, &cos, &sin).unwrap();
            let k = rotate_half_split(&k, &cos, &sin).unwrap();
            let mut heads = Vec::new();
            for head in 0..cfg.num_attention_heads {
                let kv = head / groups;
                let qh = q.narrow(1, head, 1).unwrap().squeeze(1).unwrap();
                let kh = k.narrow(1, kv, 1).unwrap().squeeze(1).unwrap();
                let vh = v.narrow(1, kv, 1).unwrap().squeeze(1).unwrap();
                let scores = (qh.matmul(&kh.t().unwrap()).unwrap() / (cfg.head_dim as f64).sqrt())
                    .unwrap()
                    .add(&mask)
                    .unwrap();
                let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
                heads.push(weights.matmul(&vh).unwrap());
            }
            let attn = Tensor::cat(&heads, 1).unwrap();
            let x1 = hidden
                .add(&lin(&attn, "self_attn.o_proj.weight", None))
                .unwrap();
            let normed =
                rms_norm(&x1, &get(&name("post_attention_layernorm.weight")), 1e-6).unwrap();
            let gate = candle_nn::ops::silu(&lin(&normed, "mlp.gate_proj.weight", None)).unwrap();
            let up = lin(&normed, "mlp.up_proj.weight", None);
            hidden = x1
                .add(&lin(&gate.mul(&up).unwrap(), "mlp.down_proj.weight", None))
                .unwrap();
        }
        rms_norm(&hidden, &get("norm.weight"), 1e-6).unwrap()
    }

    fn window_stack(window_keys: usize) -> (MingWindowStack, HashMap<String, Tensor>) {
        let cfg = tiny_stack_config(3);
        let mut rng = SplitMix64::new(11);
        let mut tensors = HashMap::new();
        tiny_stack_weights(&cfg, "", &mut rng, &mut tensors);
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
        (
            MingWindowStack::load(cfg, vb, window_keys).unwrap(),
            tensors,
        )
    }

    #[test]
    fn window_stack_matches_a_dense_masked_reference() {
        let (stack, tensors) = window_stack(5);
        let x = random(&mut SplitMix64::new(3), &[23, 32], 1.0);
        let expected = dense_reference(stack.config(), &tensors, &x, Some(5));
        let got = stack.forward(&x, &mut MingWindowState::default()).unwrap();
        assert!(max_diff(&host(&got), &host(&expected)) < 2e-5);
        // The window matters: full causal attention gives something else.
        let full = dense_reference(stack.config(), &tensors, &x, None);
        assert!(max_diff(&host(&got), &host(&full)) > 1e-3);
    }

    #[test]
    fn a_sequence_in_chunks_is_the_sequence_at_once() {
        let (stack, _) = window_stack(6);
        let x = random(&mut SplitMix64::new(4), &[31, 32], 1.0);
        let whole = stack.forward(&x, &mut MingWindowState::default()).unwrap();
        let mut state = MingWindowState::default();
        let mut parts = Vec::new();
        let mut offset = 0;
        for count in [1usize, 4, 2, 9, 7, 8] {
            parts.push(
                stack
                    .forward(&x.narrow(0, offset, count).unwrap(), &mut state)
                    .unwrap(),
            );
            offset += count;
        }
        assert_eq!(offset, 31);
        assert_eq!(state.position(), 31);
        let chunked = Tensor::cat(&parts, 0).unwrap();
        assert!(max_diff(&host(&whole), &host(&chunked)) < 2e-5);
    }

    #[test]
    fn rms_norm_computes_in_f32_for_half_precision() {
        let x = Tensor::from_vec(vec![1e-3f32, -2e-3, 3e-3, 4e-3], (1, 4), &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let wide = host(&rms_norm(&x, &weight, 1e-6).unwrap());
        let half = host(
            &rms_norm(
                &x.to_dtype(DType::F16).unwrap(),
                &weight.to_dtype(DType::F16).unwrap(),
                1e-6,
            )
            .unwrap(),
        );
        assert!(max_diff(&wide, &half) < 2e-3, "{wide:?} vs {half:?}");
        assert!(wide.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn fused_rotation_is_the_half_split_rotation() {
        let x = random(&mut SplitMix64::new(11), &[5, 3, 8], 1.0);
        let freqs = rope_frequencies(8, 1e6, &RopeScaling::Default).unwrap();
        let (cos, sin) = rope_table(&freqs, 7, 5, 4, DType::F32, &Device::Cpu).unwrap();
        let fused = host(&rope_tokens(&x, &cos, &sin).unwrap());
        let spelled = host(&rotate_half_split(&x, &cos, &sin).unwrap());
        assert!(
            max_diff(&fused, &spelled) < 1e-6,
            "{fused:?} vs {spelled:?}"
        );
    }

    #[test]
    fn the_fused_norm_keeps_values_past_half_precision() {
        // The language model's residual stream reaches ~74,000 on the
        // published weights: the norm must not narrow it before its sums.
        let x = Tensor::from_vec(vec![74_000f32, -3.0, 2.0, 1.0], (1, 4), &Device::Cpu).unwrap();
        let gain = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let fused = host(&norm_f32(&x, &gain, 1e-6).unwrap());
        let spelled = host(&rms_norm(&x, &gain, 1e-6).unwrap());
        assert!(fused.iter().all(|value| value.is_finite()));
        assert!(
            max_diff(&fused, &spelled) < 1e-5,
            "{fused:?} vs {spelled:?}"
        );
    }
}

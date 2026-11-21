//! The Gemma 4 26B-A4B MoE decoder.
//!
//! Each layer, over the residual `x`:
//!
//! ```text
//! x  = x + post_attention_norm(attention(input_norm(x)))
//! m  = post_ff_norm_1(mlp(pre_ff_norm(x)))
//!    + post_ff_norm_2(experts(pre_ff_norm_2(x), router(x)))
//! x  = (x + post_ff_norm(m)) * layer_scalar
//! ```
//!
//! The head is the final norm, the tied embedding and a `tanh` softcap.
//! Text embeddings are scaled by `sqrt(hidden)` (rounded to the embedding's
//! dtype); audio embeddings are not.
//!
//! One function runs every pass ([`SovDecoder::forward_rows`]): each row
//! appends a span of tokens at its own cache's cursor, the dense maths runs
//! once over every row's tokens and the attention per row over its own
//! paged KV. A prefill is rows with long spans; a decode step is rows with
//! one-token spans.

pub mod attention;
mod batch;
pub mod moe;
pub mod norm;
pub mod rope;
pub mod state;

use candle_core::{DType, Device, Tensor};

use self::attention::SovAttention;
use self::moe::{ExpertPath, MoeBackend, SovFeedForward};
use self::norm::RmsNorm;
use self::rope::{Rope, RopeTables};
use super::checkpoint::{SovCheckpoint, DECODER_PREFIX, EMBEDDING};
use super::config::{SovDecoderConfig, SovTranscribeConfig};
use super::encoder::linear;
use super::precision::embedding_scale;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

struct Layer {
    input_norm: RmsNorm,
    attention: SovAttention,
    post_attention_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm_1: RmsNorm,
    pre_ff_norm_2: RmsNorm,
    post_ff_norm_2: RmsNorm,
    post_ff_norm: RmsNorm,
    feed_forward: SovFeedForward,
    /// `layer_scalar`, its value in the compute dtype (applied as a scalar
    /// multiply: a broadcast of a `[1]` tensor uploads its strides every
    /// launch, a scalar's bits are the same).
    scalar: f64,
}

/// What a traced forward records (tests and parity runs).
#[derive(Debug, Default, Clone)]
pub struct DecoderTrace {
    /// Per layer, each token's `(expert, weight)` routes.
    pub routes: Vec<Vec<Vec<(u32, f32)>>>,
    /// The residual after the listed layers, row-major.
    pub hidden: Vec<(usize, Vec<f32>)>,
}

#[derive(Debug, Default, Clone)]
pub struct TraceRequest {
    pub routes: bool,
    pub hidden_after_layers: Vec<usize>,
}

/// One row of a forward: `embeds [tokens, hidden]` appended at the cache's
/// cursor.
pub struct SpanRow<'a> {
    pub embeds: Tensor,
    pub cache: &'a mut PhysicalPagedKvCache,
}

pub struct SovDecoder {
    config: SovDecoderConfig,
    device: Device,
    dtype: DType,
    /// `[vocab, hidden]`, the tied input and output embedding.
    embedding: Tensor,
    embedding_scale: f64,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    /// Every context position's rotation, per attention geometry.
    rope_sliding: RopeTables,
    rope_full: RopeTables,
    resident_bytes: usize,
    moe_backend: MoeBackend,
    /// CUDA graphs of the decode step's stretches, or why there are none.
    graphs: std::result::Result<crate::kernels::cuda::stretch_graphs::StretchGraphs, String>,
}

/// Positions the resident rotation tables cover: the model's context.
pub const ROPE_POSITIONS: usize = super::model::MAX_CONTEXT_TOKENS;

/// A step's rotation rows: `(cos, sin)` of the sliding and the full
/// geometry for every token of the step.
pub(crate) struct StepRope {
    sliding: (Tensor, Tensor),
    full: (Tensor, Tensor),
}

impl StepRope {
    fn of(&self, attention: &SovAttention) -> (&Tensor, &Tensor) {
        let (cos, sin) = if attention.geometry().window.is_some() {
            &self.sliding
        } else {
            &self.full
        };
        (cos, sin)
    }
}

fn host_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?)
}

impl SovDecoder {
    pub fn load(
        checkpoint: &SovCheckpoint,
        config: &SovTranscribeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let d = config.decoder.clone();
        let eps = d.norm_eps;
        let embedding = checkpoint.tensor(EMBEDDING, device, dtype)?;
        let mut resident_bytes = embedding.elem_count() * dtype.size_in_bytes();
        let mut layers = Vec::with_capacity(d.layers());
        for layer in 0..d.layers() {
            let l = format!("{DECODER_PREFIX}.layers.{layer}");
            let norm =
                |name: &str| RmsNorm::load(checkpoint, &format!("{l}.{name}.weight"), device, eps);
            let attention = SovAttention::load(checkpoint, config, layer, device, dtype)?;
            let feed_forward = SovFeedForward::load(checkpoint, config, layer, device, dtype)?;
            let layer = Layer {
                input_norm: norm("input_layernorm")?,
                post_attention_norm: norm("post_attention_layernorm")?,
                pre_ff_norm: norm("pre_feedforward_layernorm")?,
                post_ff_norm_1: norm("post_feedforward_layernorm_1")?,
                pre_ff_norm_2: norm("pre_feedforward_layernorm_2")?,
                post_ff_norm_2: norm("post_feedforward_layernorm_2")?,
                post_ff_norm: norm("post_feedforward_layernorm")?,
                scalar: checkpoint
                    .tensor(&format!("{l}.layer_scalar"), device, dtype)?
                    .to_dtype(DType::F64)?
                    .flatten_all()?
                    .to_vec1::<f64>()?[0],
                attention,
                feed_forward,
            };
            resident_bytes += layer.attention.resident_bytes()
                + layer.feed_forward.resident_bytes()
                + dtype.size_in_bytes()
                + [
                    &layer.input_norm,
                    &layer.post_attention_norm,
                    &layer.pre_ff_norm,
                    &layer.post_ff_norm_1,
                    &layer.pre_ff_norm_2,
                    &layer.post_ff_norm_2,
                    &layer.post_ff_norm,
                ]
                .iter()
                .map(|norm| norm.resident_bytes())
                .sum::<usize>();
            layers.push(layer);
        }
        let final_norm = RmsNorm::load(
            checkpoint,
            &format!("{DECODER_PREFIX}.norm.weight"),
            device,
            eps,
        )?;
        resident_bytes += final_norm.resident_bytes();
        let decode_graphs = super::graphs::resolve_decode(device, d.layers() + 1);
        let rope_sliding = Rope::new(&d.sliding).resident(ROPE_POSITIONS, device, dtype)?;
        let rope_full = Rope::new(&d.full).resident(ROPE_POSITIONS, device, dtype)?;
        resident_bytes += rope_sliding.resident_bytes() + rope_full.resident_bytes();
        Ok(Self {
            embedding_scale: embedding_scale(d.hidden, checkpoint.is_bf16(EMBEDDING)?) as f64,
            final_norm,
            rope_sliding,
            rope_full,
            config: d,
            device: device.clone(),
            dtype,
            embedding,
            layers,
            resident_bytes,
            moe_backend: MoeBackend::from_env()?,
            graphs: decode_graphs,
        })
    }

    pub fn config(&self) -> &SovDecoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// The decode graphs' captures and replays, or why they are off.
    pub fn graph_diagnostics(&self) -> serde_json::Value {
        match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }

    pub fn moe_backend(&self) -> MoeBackend {
        self.moe_backend
    }

    /// Switches the expert path (tests, and the load-time check's fallback).
    pub(crate) fn set_moe_backend(&mut self, backend: MoeBackend) {
        self.moe_backend = backend;
    }

    /// The relative difference between `path` and the host-routed experts
    /// of layer 0 over `rows` probe tokens: the load check that keeps a
    /// device path only when it agrees.
    pub fn check_expert_path(&self, path: ExpertPath, rows: usize) -> Result<f64> {
        let hidden = self.config.hidden;
        let h =
            crate::kernels::numeric_check::xorshift_tensor(&[rows, hidden], 11, 1.0, &self.device)?
                .to_dtype(self.dtype)?;
        let ff = &self.layers[0].feed_forward;
        let normed = self.layers[0].pre_ff_norm_2.forward(&h)?;
        let host = host_values(&ff.sparse(&h, &normed)?)?;
        let other = host_values(&ff.sparse_on(path, &h, &normed)?)?;
        Ok(crate::kernels::numeric_check::relative_l2(&other, &host))
    }

    /// Whether the experts are stored in FP8 (see `moe::ExpertWeights`).
    pub fn experts_fp8(&self) -> bool {
        self.layers
            .first()
            .is_some_and(|layer| layer.feed_forward.experts_fp8())
    }

    /// Whether every layer's experts fit the fused kernels.
    pub fn fused_experts_supported(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.feed_forward.fused_supported())
    }

    /// `(model layer, KV heads, key dim, value dim)` of every layer, in
    /// physical order.
    pub fn kv_layers(&self) -> Vec<(u32, usize, usize, usize)> {
        (0..self.config.layers())
            .map(|layer| {
                let g = self.config.geometry(layer);
                (layer as u32, g.kv_heads, g.head_dim, g.head_dim)
            })
            .collect()
    }

    /// Activation bytes of one token in the widest layer: the residual and
    /// its norms, the projections of the widest attention, the dense MLP
    /// and the selected experts' gate/up rows.
    pub fn workspace_per_token_bytes(&self) -> u64 {
        let d = &self.config;
        let widest = d.sliding.heads.max(d.full.heads) * d.sliding.head_dim.max(d.full.head_dim);
        let values = 6 * d.hidden + 3 * widest + 3 * d.mlp_width + d.top_k * 3 * d.expert_width;
        (values * 4) as u64
    }

    /// The rotation rows of `positions`, gathered from the resident tables
    /// by one small upload.
    pub(crate) fn step_rope(&self, positions: &[usize]) -> Result<StepRope> {
        if positions.iter().any(|&position| position >= ROPE_POSITIONS) {
            return Err(Error::ContextLengthExceeded(format!(
                "SOV-Transcribe positions end at {ROPE_POSITIONS}"
            )));
        }
        let index = Tensor::from_vec(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            positions.len(),
            &self.device,
        )?;
        Ok(StepRope {
            sliding: self.rope_sliding.gather(&index)?,
            full: self.rope_full.gather(&index)?,
        })
    }

    /// Scaled text embeddings `[ids, hidden]`.
    pub fn embed_text(&self, ids: &[u32]) -> Result<Tensor> {
        let index = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        let rows = self.embedding.index_select(&index, 0)?;
        Ok((rows * self.embedding_scale)?)
    }

    fn validate_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        cache.validate_sparse_model_layers(&self.kv_layers())
    }

    /// Appends each row's span at its cache's cursor and returns the final
    /// normalized hidden state of each row's last token, `[rows, hidden]`
    /// F32. Every row's KV commits together; on an error none does.
    pub fn forward_rows(
        &self,
        rows: &mut [SpanRow<'_>],
        mut trace: Option<(&TraceRequest, &mut DecoderTrace)>,
    ) -> Result<Tensor> {
        if rows.is_empty() {
            return Err(Error::InvalidInput("a decoder forward needs a row".into()));
        }
        // A decode step (one token a row) attends every row at once and runs
        // its dense work as stretches between the layers' attention, through
        // CUDA graphs where they are on.
        if trace.is_none()
            && rows
                .iter()
                .all(|row| row.embeds.dim(0).is_ok_and(|n| n == 1))
        {
            return self.decode_rows(rows);
        }
        let mut prepared = Vec::with_capacity(rows.len());
        let mut lengths = Vec::with_capacity(rows.len());
        let mut positions = Vec::new();
        for row in rows.iter() {
            self.validate_cache(row.cache)?;
            let (tokens, hidden) = row.embeds.dims2()?;
            if tokens == 0 || hidden != self.config.hidden {
                return Err(Error::InvalidInput(format!(
                    "a decoder span is [tokens, {}], got {:?}",
                    self.config.hidden,
                    row.embeds.dims()
                )));
            }
            let start = row.cache.context_len();
            match row.cache.prepare_append(start, tokens) {
                Ok(step) => prepared.push(step),
                Err(error) => {
                    for (row, step) in rows.iter_mut().zip(prepared) {
                        let _ = row.cache.abort_prepared(step);
                    }
                    return Err(error);
                }
            }
            positions.extend(start..start + tokens);
            lengths.push(tokens);
        }
        let result = self.run_layers(rows, &mut prepared, &lengths, &positions, &mut trace);
        match result {
            Ok(hidden) => {
                for (row, step) in rows.iter_mut().zip(prepared) {
                    row.cache.commit_prepared(step)?;
                }
                Ok(hidden)
            }
            Err(error) => {
                for (row, step) in rows.iter_mut().zip(prepared) {
                    let _ = row.cache.abort_prepared(step);
                }
                Err(error)
            }
        }
    }

    fn run_layers(
        &self,
        rows: &[SpanRow<'_>],
        prepared: &mut [PreparedPhysicalPagedStep],
        lengths: &[usize],
        positions: &[usize],
        trace: &mut Option<(&TraceRequest, &mut DecoderTrace)>,
    ) -> Result<Tensor> {
        let embeds = rows
            .iter()
            .map(|row| row.embeds.to_dtype(self.dtype))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut x = Tensor::cat(&embeds, 0)?;
        let rope = self.step_rope(positions)?;
        let mut profile = PrefillProfile::start(&self.device)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let h = layer.input_norm.forward(&x)?;
            let (cos, sin) = rope.of(&layer.attention);
            let (q, k, v) = layer.attention.project(&h, cos, sin)?;
            profile.mark(&self.device, PrefillPhase::Projections)?;
            let mut attended = Vec::with_capacity(rows.len());
            let mut offset = 0;
            for (r, row) in rows.iter().enumerate() {
                let len = lengths[r];
                attended.push(layer.attention.attend(
                    row.cache,
                    &mut prepared[r],
                    index,
                    &q.narrow(0, offset, len)?,
                    &k.narrow(0, offset, len)?,
                    &v.narrow(0, offset, len)?,
                    self.dtype,
                )?);
                offset += len;
            }
            profile.mark(&self.device, PrefillPhase::Attention)?;
            let attended = layer.attention.output(&Tensor::cat(&attended, 0)?)?;
            // The residual add and the dense branch's norm in one launch.
            let (sum, dense_in) = layer
                .pre_ff_norm
                .add_forward(&x, &layer.post_attention_norm.forward(&attended)?)?;
            x = sum;

            let ff = &layer.feed_forward;
            let dense = layer.post_ff_norm_1.forward(&ff.dense(&dense_in)?)?;
            profile.mark(&self.device, PrefillPhase::Dense)?;
            if let Some((request, trace)) = trace.as_mut() {
                if request.routes {
                    trace.routes.push(ff.routes(&x)?);
                }
            }
            let sparse = layer.post_ff_norm_2.forward(&ff.sparse_with(
                self.moe_backend,
                &x,
                &layer.pre_ff_norm_2.forward(&x)?,
            )?)?;
            let (_, m) = layer.post_ff_norm.add_forward(&dense, &sparse)?;
            x = (x + m)?.affine(layer.scalar, 0.0)?;
            profile.mark(&self.device, PrefillPhase::Experts)?;
            if let Some((request, trace)) = trace.as_mut() {
                if request.hidden_after_layers.contains(&index) {
                    trace.hidden.push((index, host_values(&x)?));
                }
            }
        }
        let mut last = Vec::with_capacity(rows.len());
        let mut offset = 0;
        for len in lengths {
            last.push(x.narrow(0, offset + len - 1, 1)?);
            offset += len;
        }
        profile.finish_named("prefill", rows.len(), lengths.iter().sum());
        self.final_norm.forward_f32(&Tensor::cat(&last, 0)?)
    }

    /// Layer `index`'s input norm and projections: q, k (rotated) and v of
    /// `x`. `rope` is the step's sliding then full `(cos, sin)`.
    fn pre_attention(
        &self,
        index: usize,
        x: &Tensor,
        rope: &[Tensor],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let layer = &self.layers[index];
        let h = layer.input_norm.forward(x)?;
        let (cos, sin) = if layer.attention.geometry().window.is_some() {
            (&rope[0], &rope[1])
        } else {
            (&rope[2], &rope[3])
        };
        layer.attention.project(&h, cos, sin)
    }

    /// Layer `index` after its attention (`attended` `[rows, heads * d]`):
    /// the output projection, the dense and expert branches and the
    /// residual, as [`Self::run_layers`] computes them.
    fn post_attention(&self, index: usize, x: &Tensor, attended: &Tensor) -> Result<Tensor> {
        let layer = &self.layers[index];
        let attended = layer.attention.output(attended)?;
        let (sum, dense_in) = layer
            .pre_ff_norm
            .add_forward(x, &layer.post_attention_norm.forward(&attended)?)?;
        let ff = &layer.feed_forward;
        let dense = layer.post_ff_norm_1.forward(&ff.dense(&dense_in)?)?;
        let sparse = layer.post_ff_norm_2.forward(&ff.sparse_with(
            self.moe_backend,
            &sum,
            &layer.pre_ff_norm_2.forward(&sum)?,
        )?)?;
        let (_, m) = layer.post_ff_norm.add_forward(&dense, &sparse)?;
        Ok((sum + m)?.affine(layer.scalar, 0.0)?)
    }

    /// One decode step (each row one token) as `layers + 1` stretches: the
    /// first layer's projections, then per layer everything between its
    /// attention and the next layer's (or the final norm). The paged
    /// attention between them stays eager. On CUDA, with the experts routed
    /// on the device, each stretch replays its graph for this row count; the
    /// same bodies run directly everywhere else.
    fn decode_rows(&self, rows: &mut [SpanRow<'_>]) -> Result<Tensor> {
        let mut positions = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            self.validate_cache(row.cache)?;
            let hidden = row.embeds.dim(1)?;
            if hidden != self.config.hidden {
                return Err(Error::InvalidInput(format!(
                    "a decoder span is [tokens, {}], got {:?}",
                    self.config.hidden,
                    row.embeds.dims()
                )));
            }
            positions.push(row.cache.context_len());
        }
        let window = self
            .layers
            .iter()
            .find_map(|layer| layer.attention.geometry().window);
        let mut batch = {
            let caches = rows.iter().map(|row| &*row.cache).collect::<Vec<_>>();
            batch::SovDecodeBatch::new(&caches, window, self.layers.len())?
        };
        let result = self.decode_layers(rows, &mut batch, &positions);
        let mut caches = rows
            .iter_mut()
            .map(|row| &mut *row.cache)
            .collect::<Vec<_>>();
        batch.finish(&mut caches, result)
    }

    fn decode_layers(
        &self,
        rows: &[SpanRow<'_>],
        batch: &mut batch::SovDecodeBatch,
        positions: &[usize],
    ) -> Result<Tensor> {
        let count = rows.len();
        let embeds = rows
            .iter()
            .map(|row| row.embeds.to_dtype(self.dtype))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut x = Tensor::cat(&embeds, 0)?;
        // With graphs on, the step pads its rows to the next captured width
        // (zero rows, at position 0) so every width replays a graph; the
        // paged attention reads the real rows only, and the output keeps
        // them alone.
        let graphed = self.graphs.is_ok()
            && self
                .moe_backend
                .routes_on_device(&self.device, count, self.config.top_k);
        let width = if graphed {
            super::graphs::decode_graph_width(count).unwrap_or(count)
        } else {
            count
        };
        let mut padded_positions = positions.to_vec();
        if width > count {
            x = x.pad_with_zeros(0, 0, width - count)?;
            padded_positions.resize(width, 0);
        }
        let step = self.step_rope(&padded_positions)?;
        let rope = [
            step.sliding.0.clone(),
            step.sliding.1.clone(),
            step.full.0.clone(),
            step.full.1.clone(),
        ];
        let mut run = match &self.graphs {
            Ok(graphs) if graphed => graphs.begin(width),
            _ => None,
        };
        let real = |t: &Tensor| -> Result<Tensor> {
            Ok(if width > count {
                t.narrow(0, 0, count)?
            } else {
                t.clone()
            })
        };
        let pad = |t: Tensor| -> Result<Tensor> {
            Ok(if width > count {
                t.pad_with_zeros(0, 0, width - count)?
            } else {
                t
            })
        };
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
            let (q, k, v) = self.pre_attention(0, &inputs[0], &inputs[1..])?;
            Ok(vec![q, k, v])
        };
        let mut inputs = vec![x.clone()];
        inputs.extend(rope.iter().cloned());
        let mut profile = PrefillProfile::start(&self.device)?;
        let mut qkv = stretch(0, inputs, &first)?;
        profile.mark(&self.device, PrefillPhase::Projections)?;
        let last = self.layers.len() - 1;
        for index in 0..self.layers.len() {
            let attention = &self.layers[index].attention;
            let attended = pad(attention.attend_batch(
                batch,
                rows[0].cache,
                index,
                [&real(&qkv[0])?, &real(&qkv[1])?, &real(&qkv[2])?],
                self.dtype,
            )?)?;
            profile.mark(&self.device, PrefillPhase::Attention)?;
            if index == last {
                let tail = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                    let x = self.post_attention(index, &inputs[0], &inputs[1])?;
                    Ok(vec![self.final_norm.forward_f32(&x)?])
                };
                let out = real(&stretch(index + 1, vec![x, attended], &tail)?.remove(0))?;
                profile.mark(&self.device, PrefillPhase::Experts)?;
                profile.finish_named("decode", count, count);
                return Ok(out);
            }
            let between = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
                let x = self.post_attention(index, &inputs[0], &inputs[1])?;
                let (q, k, v) = self.pre_attention(index + 1, &x, &inputs[2..])?;
                Ok(vec![x, q, k, v])
            };
            let mut inputs = vec![x, attended];
            inputs.extend(rope.iter().cloned());
            let mut outputs = stretch(index + 1, inputs, &between)?;
            profile.mark(&self.device, PrefillPhase::Experts)?;
            x = outputs.remove(0);
            qkv = outputs;
        }
        unreachable!("the last layer returns")
    }

    /// Logits `[rows, vocab]` F32 of final hidden states, softcapped.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits =
            linear(&hidden.to_dtype(self.dtype)?, &self.embedding, None)?.to_dtype(DType::F32)?;
        Ok(match self.config.final_logit_softcap {
            Some(cap) => ((logits / cap as f64)?.tanh()? * cap as f64)?,
            None => logits,
        })
    }

    /// The greedy `(token, margin)` of each row of final hidden states.
    pub fn greedy(&self, hidden: &Tensor) -> Result<Vec<(u32, f32)>> {
        Ok(crate::kernels::greedy::greedy(&self.logits(hidden)?)?)
    }
}

#[cfg(test)]
impl SovDecoder {
    /// The final hidden state of every token of one sequence from position
    /// 0, recomputed without a KV cache: dense attention with an explicit
    /// causal and sliding-window mask. `[tokens, hidden]` F32, normalized.
    pub(crate) fn dense_reference(&self, embeds: &Tensor) -> Result<Tensor> {
        self.dense_reference_with(embeds, true)
    }

    /// [`Self::dense_reference`], the sliding layers' window applied or not.
    pub(crate) fn dense_reference_with(&self, embeds: &Tensor, windowed: bool) -> Result<Tensor> {
        let tokens = embeds.dim(0)?;
        let positions = (0..tokens).collect::<Vec<_>>();
        let mut x = embeds.to_dtype(self.dtype)?;
        for layer in &self.layers {
            let g = *layer.attention.geometry();
            let h = layer.input_norm.forward(&x)?;
            // The host-built tables, so the resident ones are checked too.
            let (cos, sin) = Rope::new(&g).tables(&positions, &self.device, self.dtype)?;
            let (q, k, v) = layer.attention.project(&h, &cos, &sin)?;
            let group = g.heads / g.kv_heads;
            let spread = Tensor::from_vec(
                (0..g.heads)
                    .map(|head| (head / group) as u32)
                    .collect::<Vec<_>>(),
                g.heads,
                &self.device,
            )?;
            let heads_first = |t: &Tensor| t.transpose(0, 1)?.contiguous();
            let q = heads_first(&q)?.to_dtype(DType::F32)?;
            let k = heads_first(&k.index_select(&spread, 1)?)?.to_dtype(DType::F32)?;
            let v = heads_first(&v.index_select(&spread, 1)?)?.to_dtype(DType::F32)?;
            let mut mask = vec![0f32; tokens * tokens];
            for query in 0..tokens {
                for key in 0..tokens {
                    let hidden =
                        key > query || (windowed && g.window.is_some_and(|w| key + w <= query));
                    if hidden {
                        mask[query * tokens + key] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec(mask, (1, tokens, tokens), &self.device)?;
            let scores = q
                .matmul(&k.transpose(1, 2)?.contiguous()?)?
                .broadcast_add(&mask)?;
            let attended = candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((tokens, g.heads * g.head_dim))?
                .to_dtype(self.dtype)?;
            let attended = layer.attention.output(&attended)?;
            let (sum, dense_in) = layer
                .pre_ff_norm
                .add_forward(&x, &layer.post_attention_norm.forward(&attended)?)?;
            x = sum;
            let ff = &layer.feed_forward;
            let dense = layer.post_ff_norm_1.forward(&ff.dense(&dense_in)?)?;
            let sparse = layer
                .post_ff_norm_2
                .forward(&ff.sparse(&x, &layer.pre_ff_norm_2.forward(&x)?)?)?;
            let (_, m) = layer.post_ff_norm.add_forward(&dense, &sparse)?;
            x = (x + m)?.affine(layer.scalar, 0.0)?;
        }
        self.final_norm.forward_f32(&x)
    }

    /// One layer's expert branch computed token by token on the host from
    /// the reference formulas, for the routing test.
    pub(crate) fn feed_forward(&self, layer: usize) -> &SovFeedForward {
        &self.layers[layer].feed_forward
    }
}

/// `IZWI_SOV_TRANSCRIBE_PROFILE`: when set (`1`, `on`, `true`), every packed
/// prefill synchronizes the device around its phases and logs where its
/// time went. A diagnostic: the synchronizations slow serving.
pub const PROFILE_ENV: &str = "IZWI_SOV_TRANSCRIBE_PROFILE";

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::env::flag_or(PROFILE_ENV, false))
}

#[derive(Clone, Copy)]
enum PrefillPhase {
    Projections,
    Attention,
    Dense,
    Experts,
}

/// Device-synchronized wall time per prefill phase, summed over layers.
struct PrefillProfile {
    last: Option<std::time::Instant>,
    started: Option<std::time::Instant>,
    seconds: [f64; 4],
}

impl PrefillProfile {
    fn start(device: &candle_core::Device) -> Result<Self> {
        if !profiling() {
            return Ok(Self {
                last: None,
                started: None,
                seconds: [0.0; 4],
            });
        }
        device.synchronize()?;
        let now = std::time::Instant::now();
        Ok(Self {
            last: Some(now),
            started: Some(now),
            seconds: [0.0; 4],
        })
    }

    fn mark(&mut self, device: &candle_core::Device, phase: PrefillPhase) -> Result<()> {
        let Some(last) = self.last else {
            return Ok(());
        };
        device.synchronize()?;
        let now = std::time::Instant::now();
        self.seconds[phase as usize] += (now - last).as_secs_f64();
        self.last = Some(now);
        Ok(())
    }

    /// Logs the phases. In a decode step the dense and expert work runs
    /// inside the graphed stretches, so "experts" there is each stretch
    /// (dense, experts, norms and the next layer's projections) and
    /// "projections" the first layer's.
    fn finish_named(&self, kind: &'static str, rows: usize, tokens: usize) {
        let Some(started) = self.started else {
            return;
        };
        let ms = |s: f64| (s * 1e4).round() / 10.0;
        tracing::info!(
            rows,
            tokens,
            total_ms = ms(started.elapsed().as_secs_f64()),
            projections_ms = ms(self.seconds[0]),
            attention_ms = ms(self.seconds[1]),
            dense_ms = ms(self.seconds[2]),
            experts_ms = ms(self.seconds[3]),
            kind,
            "SOV-Transcribe profile"
        );
    }
}

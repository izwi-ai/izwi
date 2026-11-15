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
pub mod moe;
pub mod norm;
pub mod rope;
pub mod state;

use candle_core::{DType, Device, Tensor};

use self::attention::SovAttention;
use self::moe::{MoeBackend, SovFeedForward};
use self::norm::RmsNorm;
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
    /// `layer_scalar`, in the compute dtype.
    scalar: Tensor,
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
    resident_bytes: usize,
    moe_backend: MoeBackend,
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
                scalar: checkpoint.tensor(&format!("{l}.layer_scalar"), device, dtype)?,
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
        Ok(Self {
            embedding_scale: embedding_scale(d.hidden, checkpoint.is_bf16(EMBEDDING)?) as f64,
            final_norm,
            config: d,
            device: device.clone(),
            dtype,
            embedding,
            layers,
            resident_bytes,
            moe_backend: MoeBackend::from_env()?,
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

    pub fn moe_backend(&self) -> MoeBackend {
        self.moe_backend
    }

    /// Switches the expert path (tests, and the load-time check's fallback).
    pub(crate) fn set_moe_backend(&mut self, backend: MoeBackend) {
        self.moe_backend = backend;
    }

    /// The largest relative difference between the device-routed and the
    /// host-routed experts of layer 0 over `rows` probe tokens: the load
    /// check that keeps the device path only when it agrees.
    pub fn check_device_routed_experts(&self, rows: usize) -> Result<f64> {
        let hidden = self.config.hidden;
        let h =
            crate::kernels::numeric_check::xorshift_tensor(&[rows, hidden], 11, 1.0, &self.device)?
                .to_dtype(self.dtype)?;
        let ff = &self.layers[0].feed_forward;
        let normed = self.layers[0].pre_ff_norm_2.forward(&h)?;
        let host = host_values(&ff.sparse(&h, &normed)?)?;
        let device = host_values(&ff.sparse_device(&h, &normed)?)?;
        Ok(crate::kernels::numeric_check::relative_l2(&device, &host))
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
        for (index, layer) in self.layers.iter().enumerate() {
            let h = layer.input_norm.forward(&x)?;
            let (q, k, v) = layer.attention.project(&h, positions)?;
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
            let attended = layer.attention.output(&Tensor::cat(&attended, 0)?)?;
            x = (x + layer.post_attention_norm.forward(&attended)?)?;

            let ff = &layer.feed_forward;
            let dense = layer
                .post_ff_norm_1
                .forward(&ff.dense(&layer.pre_ff_norm.forward(&x)?)?)?;
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
            let m = layer.post_ff_norm.forward(&(dense + sparse)?)?;
            x = (x + m)?.broadcast_mul(&layer.scalar)?;
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
        self.final_norm.forward_f32(&Tensor::cat(&last, 0)?)
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
        let tokens = embeds.dim(0)?;
        let positions = (0..tokens).collect::<Vec<_>>();
        let mut x = embeds.to_dtype(self.dtype)?;
        for layer in &self.layers {
            let g = *layer.attention.geometry();
            let h = layer.input_norm.forward(&x)?;
            let (q, k, v) = layer.attention.project(&h, &positions)?;
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
                    let hidden = key > query || g.window.is_some_and(|w| key + w <= query);
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
            x = (x + layer.post_attention_norm.forward(&attended)?)?;
            let ff = &layer.feed_forward;
            let dense = layer
                .post_ff_norm_1
                .forward(&ff.dense(&layer.pre_ff_norm.forward(&x)?)?)?;
            let sparse = layer
                .post_ff_norm_2
                .forward(&ff.sparse(&x, &layer.pre_ff_norm_2.forward(&x)?)?)?;
            let m = layer.post_ff_norm.forward(&(dense + sparse)?)?;
            x = (x + m)?.broadcast_mul(&layer.scalar)?;
        }
        self.final_norm.forward_f32(&x)
    }

    /// One layer's expert branch computed token by token on the host from
    /// the reference formulas, for the routing test.
    pub(crate) fn feed_forward(&self, layer: usize) -> &SovFeedForward {
        &self.layers[layer].feed_forward
    }
}

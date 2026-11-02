//! The Nemotron-H decoder: 52 pre-norm layers, each `h + mixer(RMSNorm(h))`
//! with a plain multiplicative RMSNorm (eps 1e-5, no `1 + w`), then
//! `norm_f` and the untied language-model head. No embedding scaling. The
//! residual stream is F32.
//!
//! A prefill span runs one row's tokens; a decode step runs one token of
//! every row in one forward: projections and experts batched over rows,
//! the SSM and convolution over the rows' stacked states, attention per row
//! over its own paged KV.

pub mod attention;
mod graphs;
pub mod mamba;
pub mod moe;
pub mod state;

use candle_core::{DType, Device, Tensor};
use half::bf16;

use super::checkpoint::NemotronCheckpoint;
use super::config::{NemotronDecoderConfig, NemotronLayerKind, NemotronTranscribeConfig};
use super::linear::{NtLinear, Pass, Residency};
use super::precision::Fp8Mode;
use super::speech_transformer::host_values;
use crate::error::{Error, Result};
use crate::kernels::residual_norm;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

use attention::AttentionMixer;
use mamba::MambaMixer;
use moe::{MoeMixer, Routing};
pub use state::NemotronMambaState;

enum Mixer {
    Mamba { index: usize, mixer: MambaMixer },
    Moe(MoeMixer),
    Attention { index: usize, mixer: AttentionMixer },
}

struct Layer {
    norm: Tensor,
    mixer: Mixer,
}

/// One piece of the fused decode step.
#[derive(Debug, PartialEq)]
enum Piece {
    /// Layers `layers`, none of them attention, as one pure function of the
    /// residual, the normed hidden and their Mamba states. The first stretch
    /// starts from the embeddings with layer 0's norm; the tail ends with the
    /// LM head and the device greedy pick.
    Stretch {
        layers: std::ops::Range<usize>,
        first: bool,
        tail: bool,
    },
    /// Attention layer `index`, eager: it writes and reads the paged KV.
    Attention(usize),
}

/// The fused step's pieces over layers whose `attention` flags are given:
/// maximal stretches between attention layers, always a first and a tail
/// stretch (either may hold no layer).
fn pieces(attention: impl ExactSizeIterator<Item = bool>) -> Vec<Piece> {
    let layers = attention.len();
    let mut pieces = Vec::new();
    let mut start = 0;
    for (i, attention) in attention.enumerate() {
        if attention {
            if start < i || pieces.is_empty() {
                pieces.push(Piece::Stretch {
                    layers: start..i,
                    first: pieces.is_empty(),
                    tail: false,
                });
            }
            pieces.push(Piece::Attention(i));
            start = i + 1;
        }
    }
    pieces.push(Piece::Stretch {
        layers: start..layers,
        first: pieces.is_empty(),
        tail: true,
    });
    pieces
}

/// The decoder's embedding table, kept BF16 on the host: a span gathers its
/// rows and uploads them, so the table costs no device memory.
pub struct HostEmbeddings {
    hidden: usize,
    vocab: usize,
    values: Vec<bf16>,
}

impl HostEmbeddings {
    /// `[ids.len(), hidden]` F32 rows on `device`.
    pub fn gather(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            if id >= self.vocab {
                // `<|audio|>` (one past the table) must never be looked up.
                return Err(Error::InferenceError(format!(
                    "token {id} is outside the {}-row embedding table",
                    self.vocab
                )));
            }
            out.extend(
                self.values[id * self.hidden..(id + 1) * self.hidden]
                    .iter()
                    .map(|v| v.to_f32()),
            );
        }
        Ok(Tensor::from_vec(out, (ids.len(), self.hidden), device)?)
    }

    pub fn host_bytes(&self) -> usize {
        self.values.len() * 2
    }
}

/// What a forward observed, for diagnostics and parity tests.
#[derive(Debug, Default, Clone)]
pub struct DecoderTrace {
    /// Router decisions of every MoE layer, in layer order.
    pub routes: Vec<Vec<Routing>>,
    /// Hidden states after selected layers `(layer, [tokens, hidden])`.
    pub hidden: Vec<(usize, Vec<f32>)>,
    /// The last prompt position's logits, once the prompt has prefilled.
    pub final_logits: Vec<f32>,
}

/// Which hidden states a forward records.
#[derive(Debug, Default, Clone)]
pub struct TraceRequest {
    pub routes: bool,
    pub hidden_after_layers: Vec<usize>,
}

/// One row of a decode step: its state, cache and input token.
pub struct DecodeRow<'a> {
    pub state: &'a mut NemotronMambaState,
    pub cache: &'a mut PhysicalPagedKvCache,
    pub token: u32,
}

/// Kill switch for the fused decode step (`0`, `off`, `false`): the step
/// then runs the eager composition. Read at load.
pub const FUSED_DECODE_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_FUSED_DECODE";

/// How decode steps run, and why.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodePath {
    /// Fused norms, Mamba step kernels and device greedy, with the load-time
    /// kernel check's worst relative error.
    Fused { check: f64 },
    /// The eager composition: switched off, or the check failed.
    Eager { reason: String },
}

pub struct NemotronDecoder {
    config: NemotronDecoderConfig,
    embeddings: HostEmbeddings,
    layers: Vec<Layer>,
    norm_f: Tensor,
    lm_head: NtLinear,
    attention_layers: Vec<u32>,
    device: Device,
    path: DecodePath,
    pieces: Vec<Piece>,
    /// CUDA graphs of the fused step's stretches, or why there are none.
    graphs: std::result::Result<crate::kernels::cuda::stretch_graphs::StretchGraphs, String>,
}

impl NemotronDecoder {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = &config.decoder;
        let norm = |name: &str| -> Result<Tensor> {
            Ok(Tensor::from_vec(
                checkpoint.f32_values(name, &[d.hidden])?,
                d.hidden,
                device,
            )?)
        };
        let mut layers = Vec::with_capacity(d.layers.len());
        let (mut mamba, mut attention, mut moe) = (0usize, 0usize, 0usize);
        for (index, kind) in d.layers.iter().enumerate() {
            let mixer = match kind {
                NemotronLayerKind::Mamba => {
                    mamba += 1;
                    Mixer::Mamba {
                        index: mamba - 1,
                        mixer: MambaMixer::load(checkpoint, config, index, device, mode)?,
                    }
                }
                NemotronLayerKind::Moe => {
                    moe += 1;
                    // The first MoE layer checks its stacked experts.
                    Mixer::Moe(MoeMixer::load(
                        checkpoint,
                        config,
                        index,
                        device,
                        mode,
                        moe == 1,
                    )?)
                }
                NemotronLayerKind::Attention => {
                    attention += 1;
                    Mixer::Attention {
                        index: attention - 1,
                        mixer: AttentionMixer::load(checkpoint, config, index, device, mode)?,
                    }
                }
            };
            layers.push(Layer {
                norm: norm(&format!("backbone.layers.{index}.norm.weight"))?,
                mixer,
            });
        }
        let head = checkpoint.f32_values("lm_head.weight", &[d.vocab, d.hidden])?;
        let lm_head = match Residency::of(device) {
            Residency::Cpu => NtLinear::q8_from_values(&head, d.vocab, d.hidden)?,
            Residency::Metal => {
                NtLinear::dense_from_values(head, d.vocab, d.hidden, device, DType::F16)?
            }
            Residency::Cuda => {
                NtLinear::dense_from_values(head, d.vocab, d.hidden, device, DType::BF16)?
            }
        };
        let path = if !crate::env::flag_or(FUSED_DECODE_ENV, true) {
            DecodePath::Eager {
                reason: format!("{FUSED_DECODE_ENV} is off"),
            }
        } else {
            match fused_kernels_check(device, d) {
                Ok(check) if check <= FUSED_CHECK_LIMIT => DecodePath::Fused { check },
                Ok(check) => DecodePath::Eager {
                    reason: format!("the fused kernels deviate by {check:.3e}"),
                },
                Err(error) => DecodePath::Eager {
                    reason: format!("the fused kernel check failed: {error}"),
                },
            }
        };
        if let DecodePath::Eager { reason } = &path {
            tracing::warn!("Nemotron 3.5 Transcribe decodes eagerly: {reason}");
        }
        let pieces = pieces(
            layers
                .iter()
                .map(|layer| matches!(layer.mixer, Mixer::Attention { .. })),
        );
        let stretches = pieces
            .iter()
            .filter(|piece| matches!(piece, Piece::Stretch { .. }))
            .count();
        let graphs = match &path {
            DecodePath::Fused { .. } => graphs::resolve(device, stretches),
            DecodePath::Eager { .. } => Err("the decode runs eagerly".into()),
        };
        Ok(Self {
            path,
            pieces,
            graphs,
            embeddings: HostEmbeddings {
                hidden: d.hidden,
                vocab: d.vocab,
                values: checkpoint
                    .bf16_values("backbone.embeddings.weight", &[d.vocab, d.hidden])?,
            },
            norm_f: norm("backbone.norm_f.weight")?,
            attention_layers: d
                .layers_of(NemotronLayerKind::Attention)
                .into_iter()
                .map(|l| l as u32)
                .collect(),
            layers,
            lm_head,
            config: d.clone(),
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &NemotronDecoderConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// How decode steps run.
    pub fn decode_path(&self) -> &DecodePath {
        &self.path
    }

    /// Runs every stretch eagerly, for tests that compare.
    #[cfg(test)]
    pub(crate) fn disable_graphs(&mut self) {
        self.graphs = Err("disabled by a test".into());
    }

    /// The stretch graphs' state and counters.
    pub fn graph_diagnostics(&self) -> serde_json::Value {
        match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }

    fn fused(&self) -> bool {
        matches!(self.path, DecodePath::Fused { .. })
    }

    /// Forces the decode path, for tests that compare the two.
    #[cfg(test)]
    pub(crate) fn force_decode_path(&mut self, path: DecodePath) {
        self.path = path;
    }

    pub fn embeddings(&self) -> &HostEmbeddings {
        &self.embeddings
    }

    /// Model-layer indices of the paged attention layers.
    pub fn attention_layers(&self) -> &[u32] {
        &self.attention_layers
    }

    pub fn resident_bytes(&self) -> usize {
        let layers = self
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Mixer::Mamba { mixer, .. } => mixer.resident_bytes(),
                Mixer::Moe(mixer) => mixer.resident_bytes(),
                Mixer::Attention { mixer, .. } => mixer.resident_bytes(),
            })
            .sum::<usize>();
        layers + self.lm_head.resident_bytes()
    }

    /// Whether the MoE layers run on the device-routed kernels, and the
    /// first one's stacked-expert check (relative L2) when they do.
    pub fn moe_kernels(&self) -> (bool, Option<f64>) {
        self.layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Moe(mixer) => Some((mixer.device_routed(), mixer.stacked_check())),
                _ => None,
            })
            .unwrap_or((false, None))
    }

    /// One projection of each kind (the first layer of each), for the
    /// load-time FP8 self-check: `(checkpoint module, rows, cols, layer)`.
    pub fn sample_projections(&self) -> Vec<(String, &NtLinear)> {
        let mut out = Vec::new();
        let (mut mamba, mut moe, mut attention) = (false, false, false);
        for (index, layer) in self.layers.iter().enumerate() {
            let x = format!("backbone.layers.{index}.mixer");
            match &layer.mixer {
                Mixer::Mamba { mixer, .. } if !mamba => {
                    mamba = true;
                    out.push((format!("{x}.in_proj"), mixer.in_proj()));
                    out.push((format!("{x}.out_proj"), mixer.out_proj()));
                }
                Mixer::Moe(mixer) if !moe => {
                    moe = true;
                    for (name, linear) in mixer.sample_projections() {
                        out.push((format!("{x}.{name}"), linear));
                    }
                }
                Mixer::Attention { mixer, .. } if !attention => {
                    attention = true;
                    for (name, linear) in ["q_proj", "k_proj", "v_proj", "o_proj"]
                        .into_iter()
                        .zip(mixer.projections())
                    {
                        out.push((format!("{x}.{name}"), linear));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn validate_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        let a = self.config.attention;
        cache.validate_sparse_model(&self.attention_layers, a.kv_heads, a.head_dim, a.head_dim)
    }

    fn kv_dtype(cache: &PhysicalPagedKvCache) -> DType {
        cache.arena().config().dtype
    }

    /// Prefills `embeds` `[tokens, hidden]` of one row at its cache's
    /// cursor. Returns the final normalized hidden state of the last token
    /// `[1, hidden]`.
    pub fn forward_span(
        &self,
        embeds: &Tensor,
        state: &mut NemotronMambaState,
        cache: &mut PhysicalPagedKvCache,
        trace: Option<(&TraceRequest, &mut DecoderTrace)>,
    ) -> Result<Tensor> {
        self.validate_cache(cache)?;
        let tokens = embeds.dim(0)?;
        let start = cache.context_len();
        if state.cursor != start as u64 {
            return Err(Error::InferenceError(
                "Nemotron 3.5 Transcribe Mamba and KV clocks diverged".into(),
            ));
        }
        let mut prepared = cache.prepare_append(start, tokens)?;
        let kv_dtype = Self::kv_dtype(cache);
        let mut next = state.clone();
        let mut trace = trace;
        let result = (|| -> Result<Tensor> {
            let eps = self.config.norm_eps as f32;
            let mut x = embeds.to_dtype(DType::F32)?;
            let mut normed = self
                .fused()
                .then(|| residual_norm::rms_norm(&x, &self.layers[0].norm, eps, DType::F32))
                .transpose()?;
            for (index, layer) in self.layers.iter().enumerate() {
                let h = match normed.take() {
                    Some(h) => h,
                    None => candle_nn::ops::rms_norm(&x, &layer.norm, eps)?,
                };
                let out = match &layer.mixer {
                    Mixer::Mamba { index, mixer } => {
                        let (out, conv, ssm) = mixer.forward(
                            &h,
                            1,
                            tokens,
                            &next.conv[*index],
                            &next.ssm[*index],
                            Pass::Span,
                        )?;
                        next.conv[*index] = conv;
                        next.ssm[*index] = ssm;
                        out
                    }
                    Mixer::Moe(mixer) => {
                        let want_routes = trace.as_ref().is_some_and(|(request, _)| request.routes);
                        let (out, routes) = mixer.forward(&h, Pass::Span, want_routes)?;
                        if let Some((request, trace)) = trace.as_mut() {
                            if request.routes {
                                trace.routes.push(routes);
                            }
                        }
                        out
                    }
                    Mixer::Attention { index, mixer } => {
                        let (q, k, v) = mixer.project(&h, Pass::Span)?;
                        let attended =
                            mixer.attend(cache, &mut prepared, *index, &q, &k, &v, kv_dtype)?;
                        mixer.output(&attended, Pass::Span)?
                    }
                };
                if self.fused() {
                    // The next layer's norm (or the final one) with the add.
                    let gain = self.layers.get(index + 1).map_or(&self.norm_f, |l| &l.norm);
                    let (sum, h) = residual_norm::add_rms_norm(&x, &out, gain, eps, DType::F32)?;
                    x = sum;
                    normed = Some(h);
                } else {
                    x = (x + out)?;
                }
                if let Some((request, trace)) = trace.as_mut() {
                    if request.hidden_after_layers.contains(&index) {
                        trace.hidden.push((index, host_values(&x)?));
                    }
                }
            }
            match normed {
                Some(h) => Ok(h.narrow(0, tokens - 1, 1)?),
                None => {
                    let last = x.narrow(0, tokens - 1, 1)?;
                    Ok(candle_nn::ops::rms_norm(&last, &self.norm_f, eps)?)
                }
            }
        })();
        match result {
            Ok(hidden) => {
                cache.commit_prepared(prepared)?;
                next.cursor = (start + tokens) as u64;
                *state = next;
                Ok(hidden)
            }
            Err(error) => {
                let _ = cache.abort_prepared(prepared);
                Err(error)
            }
        }
    }

    /// One decode step for every row: each row's `token` at its cursor, then
    /// each row's greedy pick `(id, margin)`. The picks are read back before
    /// the rows' KV and state commit, so the step drains the device once.
    pub fn decode_step(&self, rows: &mut [DecodeRow<'_>]) -> Result<Vec<(u32, f32)>> {
        let count = rows.len();
        if count == 0 {
            return Err(Error::InvalidInput("a decode step needs a row".into()));
        }
        let mut prepared = Vec::with_capacity(count);
        for row in rows.iter() {
            self.validate_cache(row.cache)?;
            let start = row.cache.context_len();
            if row.state.cursor != start as u64 {
                return Err(Error::InferenceError(
                    "Nemotron 3.5 Transcribe Mamba and KV clocks diverged".into(),
                ));
            }
            prepared.push(row.cache.prepare_append(start, 1)?);
        }
        let ids = rows.iter().map(|r| r.token).collect::<Vec<_>>();
        let mamba_layers = self.config.layers_of(NemotronLayerKind::Mamba).len();
        let mut conv = (0..mamba_layers)
            .map(|layer| {
                Tensor::cat(
                    &rows
                        .iter()
                        .map(|r| &r.state.conv[layer])
                        .collect::<Vec<_>>(),
                    0,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut ssm = (0..mamba_layers)
            .map(|layer| {
                Tensor::cat(
                    &rows.iter().map(|r| &r.state.ssm[layer]).collect::<Vec<_>>(),
                    0,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let result = (|| -> Result<Vec<(u32, f32)>> {
            let x = self.embeddings.gather(&ids, &self.device)?;
            if self.fused() {
                let picks = self.step_fused(x, rows, &mut prepared, &mut conv, &mut ssm)?;
                Ok(crate::kernels::greedy::unpack(
                    &picks.flatten_all()?.to_vec1::<u32>()?,
                ))
            } else {
                let hidden = self.step_layers_eager(x, rows, &mut prepared, &mut conv, &mut ssm)?;
                self.greedy(&hidden)
            }
        })();
        match result {
            Ok(picks) => {
                for ((row, step), r) in rows.iter_mut().zip(prepared).zip(0..) {
                    row.cache.commit_prepared(step)?;
                    for layer in 0..mamba_layers {
                        row.state.conv[layer] = conv[layer].narrow(0, r, 1)?;
                        row.state.ssm[layer] = ssm[layer].narrow(0, r, 1)?;
                    }
                    row.state.cursor += 1;
                }
                Ok(picks)
            }
            Err(error) => {
                for (row, step) in rows.iter_mut().zip(prepared) {
                    let _ = row.cache.abort_prepared(step);
                }
                Err(error)
            }
        }
    }

    /// One attention layer of a decode step: projections, each row over its
    /// own paged KV, the output projection.
    fn step_attention(
        &self,
        index: usize,
        mixer: &AttentionMixer,
        h: &Tensor,
        rows: &mut [DecodeRow<'_>],
        prepared: &mut [PreparedPhysicalPagedStep],
    ) -> Result<Tensor> {
        let (q, k, v) = mixer.project(h, Pass::Step)?;
        let mut attended = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter().enumerate() {
            attended.push(mixer.attend(
                row.cache,
                &mut prepared[r],
                index,
                &q.narrow(0, r, 1)?,
                &k.narrow(0, r, 1)?,
                &v.narrow(0, r, 1)?,
                Self::kv_dtype(row.cache),
            )?);
        }
        mixer.output(&Tensor::cat(&attended, 0)?, Pass::Step)
    }

    /// The decode layers as composed ops; returns the final normed hidden.
    fn step_layers_eager(
        &self,
        mut x: Tensor,
        rows: &mut [DecodeRow<'_>],
        prepared: &mut [PreparedPhysicalPagedStep],
        conv: &mut [Tensor],
        ssm: &mut [Tensor],
    ) -> Result<Tensor> {
        let count = rows.len();
        let eps = self.config.norm_eps as f32;
        for layer in &self.layers {
            let h = candle_nn::ops::rms_norm(&x, &layer.norm, eps)?;
            let out = match &layer.mixer {
                Mixer::Mamba { index, mixer } => {
                    let (out, c, s) =
                        mixer.forward(&h, count, 1, &conv[*index], &ssm[*index], Pass::Step)?;
                    conv[*index] = c;
                    ssm[*index] = s;
                    out
                }
                Mixer::Moe(mixer) => mixer.forward(&h, Pass::Step, false)?.0,
                Mixer::Attention { index, mixer } => {
                    self.step_attention(*index, mixer, &h, rows, prepared)?
                }
            };
            x = (x + out)?;
        }
        Ok(candle_nn::ops::rms_norm(&x, &self.norm_f, eps)?)
    }

    /// The decode layers through the fused kernels, piece by piece: each
    /// residual add with the next norm in one launch, each Mamba mixer
    /// through its step kernels, the head in its own dtype straight into the
    /// device greedy pick. On CUDA the stretches replay their graphs.
    /// Returns the packed picks `[rows, 2]` U32 on the device.
    fn step_fused(
        &self,
        mut x: Tensor,
        rows: &mut [DecodeRow<'_>],
        prepared: &mut [PreparedPhysicalPagedStep],
        conv: &mut [Tensor],
        ssm: &mut [Tensor],
    ) -> Result<Tensor> {
        let eps = self.config.norm_eps as f32;
        let mut graphs = self
            .graphs
            .as_ref()
            .ok()
            .and_then(|graphs| graphs.begin(rows.len()));
        let mut h = None;
        let mut stretch = 0;
        for piece in &self.pieces {
            match piece {
                Piece::Stretch {
                    layers,
                    first,
                    tail,
                } => {
                    let mamba = self.mamba_indices(layers.clone());
                    let mut inputs = vec![x.clone()];
                    if !first {
                        inputs.push(h.clone().ok_or_else(|| {
                            Error::InferenceError("a decode stretch has no normed input".into())
                        })?);
                    }
                    inputs.extend(mamba.iter().map(|&m| conv[m].clone()));
                    inputs.extend(mamba.iter().map(|&m| ssm[m].clone()));
                    let body = |inputs: &[Tensor]| {
                        self.run_stretch(layers.clone(), *first, *tail, &mamba, inputs)
                    };
                    let mut outputs = match graphs.as_mut() {
                        Some(run) => run.run(stretch, &inputs, &body)?,
                        None => body(&inputs)?,
                    }
                    .into_iter();
                    let mut next = || {
                        outputs.next().ok_or_else(|| {
                            Error::InferenceError(
                                "a decode stretch returned too few outputs".into(),
                            )
                        })
                    };
                    if *tail {
                        let picks = next()?;
                        for &m in &mamba {
                            conv[m] = next()?;
                        }
                        for &m in &mamba {
                            ssm[m] = next()?;
                        }
                        return Ok(picks);
                    }
                    x = next()?;
                    h = Some(next()?);
                    for &m in &mamba {
                        conv[m] = next()?;
                    }
                    for &m in &mamba {
                        ssm[m] = next()?;
                    }
                    stretch += 1;
                }
                Piece::Attention(i) => {
                    let Mixer::Attention { index, mixer } = &self.layers[*i].mixer else {
                        return Err(Error::InferenceError(format!(
                            "Nemotron 3.5 Transcribe layer {i} is not attention"
                        )));
                    };
                    let normed = h.as_ref().ok_or_else(|| {
                        Error::InferenceError("an attention layer has no normed input".into())
                    })?;
                    let out = self.step_attention(*index, mixer, normed, rows, prepared)?;
                    let gain = self.layers.get(i + 1).map_or(&self.norm_f, |l| &l.norm);
                    let (sum, normed) =
                        residual_norm::add_rms_norm(&x, &out, gain, eps, DType::F32)?;
                    x = sum;
                    h = Some(normed);
                }
            }
        }
        Err(Error::InferenceError(
            "Nemotron 3.5 Transcribe's decode pieces have no tail".into(),
        ))
    }

    /// The Mamba state indices of `layers`, in order.
    fn mamba_indices(&self, layers: std::ops::Range<usize>) -> Vec<usize> {
        self.layers[layers]
            .iter()
            .filter_map(|layer| match layer.mixer {
                Mixer::Mamba { index, .. } => Some(index),
                _ => None,
            })
            .collect()
    }

    /// One stretch, eagerly: `inputs` are the residual, the normed hidden
    /// (unless `first`), then the conv and the SSM states of `mamba` (the
    /// stretch's Mamba indices). Returns the residual and the normed hidden
    /// (or, at the tail, the packed greedy picks), then the new states.
    fn run_stretch(
        &self,
        layers: std::ops::Range<usize>,
        first: bool,
        tail: bool,
        mamba: &[usize],
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let eps = self.config.norm_eps as f32;
        let base = if first { 1 } else { 2 };
        if inputs.len() != base + 2 * mamba.len() {
            return Err(Error::InferenceError(format!(
                "a decode stretch took {} inputs for {} Mamba layers",
                inputs.len(),
                mamba.len()
            )));
        }
        let mut x = inputs[0].clone();
        let mut h = if first {
            residual_norm::rms_norm(&x, &self.layers[0].norm, eps, DType::F32)?
        } else {
            inputs[1].clone()
        };
        let mut conv = inputs[base..base + mamba.len()].to_vec();
        let mut ssm = inputs[base + mamba.len()..].to_vec();
        let mut at = 0;
        for i in layers {
            let out = match &self.layers[i].mixer {
                Mixer::Mamba { mixer, .. } => {
                    let (out, c, s) = mixer.forward_step_fused(&h, &conv[at], &ssm[at])?;
                    conv[at] = c;
                    ssm[at] = s;
                    at += 1;
                    out
                }
                Mixer::Moe(mixer) => mixer.forward(&h, Pass::Step, false)?.0,
                Mixer::Attention { .. } => {
                    return Err(Error::InferenceError(format!(
                        "decode stretch layer {i} is attention"
                    )))
                }
            };
            let gain = self.layers.get(i + 1).map_or(&self.norm_f, |l| &l.norm);
            (x, h) = residual_norm::add_rms_norm(&x, &out, gain, eps, DType::F32)?;
        }
        let mut outputs = if tail {
            vec![crate::kernels::greedy::greedy_packed(
                &self.lm_head.forward_native(&h)?,
            )?]
        } else {
            vec![x, h]
        };
        outputs.extend(conv);
        outputs.extend(ssm);
        Ok(outputs)
    }

    /// Logits `[rows, vocab]` of final hidden states `[rows, hidden]`.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden, Pass::Step)
    }

    /// The greedy token of each row (ties to the lowest id), with the margin
    /// to the runner-up.
    pub fn greedy(&self, hidden: &Tensor) -> Result<Vec<(u32, f32)>> {
        let logits = host_values(&self.logits(hidden)?)?;
        Ok(logits
            .chunks_exact(self.config.vocab)
            .map(|row| {
                let mut best = (0usize, f32::NEG_INFINITY);
                let mut second = f32::NEG_INFINITY;
                for (id, &v) in row.iter().enumerate() {
                    if v > best.1 {
                        second = best.1;
                        best = (id, v);
                    } else if v > second {
                        second = v;
                    }
                }
                (best.0 as u32, best.1 - second)
            })
            .collect())
    }
}

/// Largest relative L2 deviation of a device's fused decode kernels from
/// their CPU forms that keeps the fused path on.
const FUSED_CHECK_LIMIT: f64 = 1e-4;

/// The fused decode kernels on `device` against their CPU forms, in the
/// decoder's geometry: the residual add + norm, the Mamba2 convolution step
/// and gated norm, and the greedy pick (which must match exactly). Returns
/// the worst relative L2 error; the CPU's kernels are the reference, 0.
fn fused_kernels_check(device: &Device, d: &NemotronDecoderConfig) -> Result<f64> {
    use crate::kernels::mamba2;
    use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_tensor};
    if device.is_cpu() {
        return Ok(0.0);
    }
    let cpu = Device::Cpu;
    let on = |t: &Tensor| t.to_device(device);
    let mut worst = 0f64;
    let mut track = |a: &Tensor, b: &Tensor| -> Result<()> {
        worst = worst.max(relative_l2(&host_f32(a)?, &host_f32(b)?));
        Ok(())
    };
    let rows = 2;
    let (x, delta) = (
        xorshift_tensor(&[rows, d.hidden], 1, 4.0, &cpu)?,
        xorshift_tensor(&[rows, d.hidden], 2, 2.0, &cpu)?,
    );
    let gain = (xorshift_tensor(&[d.hidden], 3, 0.3, &cpu)? + 1.0)?;
    let eps = d.norm_eps as f32;
    let (sum, normed) = residual_norm::add_rms_norm(&x, &delta, &gain, eps, DType::F32)?;
    let (dsum, dnormed) =
        residual_norm::add_rms_norm(&on(&x)?, &on(&delta)?, &on(&gain)?, eps, DType::F32)?;
    track(&dsum, &sum)?;
    track(&dnormed, &normed)?;
    let m = d.mamba;
    let geometry = m.geometry();
    let layout = mamba2::Mamba2StepLayout {
        width: m.in_proj_rows(),
        z_offset: 0,
        xbc_offset: m.inner(),
        dt_offset: m.inner() + m.conv_channels(),
        taps: m.conv_kernel,
    };
    let c = m.conv_channels();
    let projected = xorshift_tensor(&[rows, layout.width], 4, 2.0, &cpu)?;
    let history = xorshift_tensor(&[rows, m.conv_kernel - 1, c], 5, 2.0, &cpu)?;
    let weight = xorshift_tensor(&[c, m.conv_kernel], 6, 0.5, &cpu)?;
    let bias = xorshift_tensor(&[c], 7, 0.5, &cpu)?;
    let y = xorshift_tensor(&[rows, m.inner()], 8, 2.0, &cpu)?;
    let norm = (xorshift_tensor(&[m.inner()], 9, 0.3, &cpu)? + 1.0)?;
    let (u, dt, next) =
        mamba2::conv_step_silu(geometry, layout, &projected, &history, &weight, &bias)?;
    let (du, ddt, dnext) = mamba2::conv_step_silu(
        geometry,
        layout,
        &on(&projected)?,
        &on(&history)?,
        &on(&weight)?,
        &on(&bias)?,
    )?;
    track(&du, &u)?;
    track(&ddt, &dt)?;
    track(&dnext, &next)?;
    let gated = mamba2::gated_group_rms_norm_step(
        &y,
        &projected,
        layout,
        &norm,
        m.norm_group(),
        d.norm_eps,
    )?;
    let dgated = mamba2::gated_group_rms_norm_step(
        &on(&y)?,
        &on(&projected)?,
        layout,
        &on(&norm)?,
        m.norm_group(),
        d.norm_eps,
    )?;
    track(&dgated, &gated)?;
    let logits = xorshift_tensor(&[rows, d.vocab.min(65_536)], 10, 8.0, &cpu)?;
    let picks = crate::kernels::greedy::greedy(&logits)?;
    let dpicks = crate::kernels::greedy::greedy(&on(&logits)?)?;
    if picks
        .iter()
        .zip(&dpicks)
        .any(|(a, b)| a.0 != b.0 || a.1.to_bits() != b.1.to_bits())
    {
        return Ok(f64::INFINITY);
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(pattern: &str) -> Vec<Piece> {
        pieces(
            pattern
                .chars()
                .map(|c| c == '*')
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn stretch(layers: std::ops::Range<usize>, first: bool, tail: bool) -> Piece {
        Piece::Stretch {
            layers,
            first,
            tail,
        }
    }

    /// Stretches cover every non-attention layer once, between the attention
    /// layers, and the first and tail stretches always exist.
    #[test]
    fn the_pieces_split_the_layers_at_attention() {
        assert_eq!(
            layout("ME*ME"),
            [
                stretch(0..2, true, false),
                Piece::Attention(2),
                stretch(3..5, false, true)
            ]
        );
        assert_eq!(
            layout("*M"),
            [
                stretch(0..0, true, false),
                Piece::Attention(0),
                stretch(1..2, false, true)
            ]
        );
        assert_eq!(
            layout("ME**"),
            [
                stretch(0..2, true, false),
                Piece::Attention(2),
                Piece::Attention(3),
                stretch(4..4, false, true)
            ]
        );
        assert_eq!(layout("MEME"), [stretch(0..4, true, true)]);
    }
}

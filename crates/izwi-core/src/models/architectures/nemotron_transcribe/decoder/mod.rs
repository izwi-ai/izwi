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
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

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

pub struct NemotronDecoder {
    config: NemotronDecoderConfig,
    embeddings: HostEmbeddings,
    layers: Vec<Layer>,
    norm_f: Tensor,
    lm_head: NtLinear,
    attention_layers: Vec<u32>,
    device: Device,
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
        let (mut mamba, mut attention) = (0usize, 0usize);
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
                    Mixer::Moe(MoeMixer::load(checkpoint, config, index, device, mode)?)
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
        Ok(Self {
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
                    let [up, down, shared_up, shared_down] = mixer.sample_projections();
                    out.push((format!("{x}.experts.0.up_proj"), up));
                    out.push((format!("{x}.experts.0.down_proj"), down));
                    out.push((format!("{x}.shared_experts.up_proj"), shared_up));
                    out.push((format!("{x}.shared_experts.down_proj"), shared_down));
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
            let mut x = embeds.to_dtype(DType::F32)?;
            for (index, layer) in self.layers.iter().enumerate() {
                let h = candle_nn::ops::rms_norm(&x, &layer.norm, self.config.norm_eps as f32)?;
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
                        let (out, routes) = mixer.forward(&h, Pass::Span)?;
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
                x = (x + out)?;
                if let Some((request, trace)) = trace.as_mut() {
                    if request.hidden_after_layers.contains(&index) {
                        trace.hidden.push((index, host_values(&x)?));
                    }
                }
            }
            let last = x.narrow(0, tokens - 1, 1)?;
            Ok(candle_nn::ops::rms_norm(
                &last,
                &self.norm_f,
                self.config.norm_eps as f32,
            )?)
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

    /// One decode step for every row: each row's `token` at its cursor.
    /// Returns the rows' final normalized hidden states `[rows, hidden]`.
    pub fn forward_step(&self, rows: &mut [DecodeRow<'_>]) -> Result<Tensor> {
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
        let result = (|| -> Result<Tensor> {
            let mut x = self.embeddings.gather(&ids, &self.device)?;
            for layer in &self.layers {
                let h = candle_nn::ops::rms_norm(&x, &layer.norm, self.config.norm_eps as f32)?;
                let out = match &layer.mixer {
                    Mixer::Mamba { index, mixer } => {
                        let (out, c, s) =
                            mixer.forward(&h, count, 1, &conv[*index], &ssm[*index], Pass::Step)?;
                        conv[*index] = c;
                        ssm[*index] = s;
                        out
                    }
                    Mixer::Moe(mixer) => mixer.forward(&h, Pass::Step)?.0,
                    Mixer::Attention { index, mixer } => {
                        let (q, k, v) = mixer.project(&h, Pass::Step)?;
                        let mut attended = Vec::with_capacity(count);
                        for (r, row) in rows.iter().enumerate() {
                            attended.push(mixer.attend(
                                row.cache,
                                &mut prepared[r],
                                *index,
                                &q.narrow(0, r, 1)?,
                                &k.narrow(0, r, 1)?,
                                &v.narrow(0, r, 1)?,
                                Self::kv_dtype(row.cache),
                            )?);
                        }
                        mixer.output(&Tensor::cat(&attended, 0)?, Pass::Step)?
                    }
                };
                x = (x + out)?;
            }
            Ok(candle_nn::ops::rms_norm(
                &x,
                &self.norm_f,
                self.config.norm_eps as f32,
            )?)
        })();
        match result {
            Ok(hidden) => {
                for ((row, step), r) in rows.iter_mut().zip(prepared).zip(0..) {
                    row.cache.commit_prepared(step)?;
                    for layer in 0..mamba_layers {
                        row.state.conv[layer] = conv[layer].narrow(0, r, 1)?;
                        row.state.ssm[layer] = ssm[layer].narrow(0, r, 1)?;
                    }
                    row.state.cursor += 1;
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

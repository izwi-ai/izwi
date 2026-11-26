//! The text trunk, composed from plain tensor operations in F32.
//!
//! This is the reference path: it follows transformers'
//! `Inkling2LoopTextModel` operation by operation, keeps its own per-sequence
//! state, and is what every faster path is checked against. It is also the
//! fallback when a fused path is switched off or fails its self-check.
//!
//! One token passes through [`Inkling2Config::applications`] layer
//! applications. What makes the model unusual, in the order the code meets
//! it:
//!
//! * **Depth residual.** There is no plain residual stream. Layers are
//!   grouped in blocks; the input of every attention and every MLP is an
//!   attention over the block inputs so far plus the current partial state
//!   (all RMS-normalised), with a learned query. This is per position and
//!   needs no state across tokens.
//! * **Loop.** Layers `loop_start..loop_end` run `loop_passes` times with the
//!   same weights. Passes after the first restart from the state before the
//!   loop and receive "messages" from the previous pass: a query-based read
//!   of that pass's block states, mixed in through low-rank tanh gates.
//!   Every pass has its own attention and convolution state.
//! * **Attention.** No rotary embedding. Position enters as an additive bias
//!   that each query computes from its own hidden state and a trained bank
//!   of bias-versus-distance profiles. Each query also has a sink logit that
//!   only absorbs probability. Queries and keys are L2-clipped, values are
//!   RMS-normalised without gain, and the output is gated by a sigmoid of
//!   the layer input.
//! * **Short convolutions.** Keys, values, the attention output and the MLP
//!   output each pass a causal depthwise convolution over the last few
//!   tokens. On full-attention layers the key and value taps are static
//!   weights plus per-token weights projected from the input.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};

use super::checkpoint::{layer_prefix, ExpertProjection, Inkling2Checkpoint, TEXT_PREFIX};
use super::config::{
    Inkling2Application, Inkling2Config, Inkling2Heads, Inkling2LayerKind,
    CONV_STATES_PER_APPLICATION,
};
use super::experts::Inkling2Moe;
use crate::error::{Error, Result};

/// Epsilon of the gain-free RMS norms that take none from the config (the
/// depth-residual and message reads).
const DEPTH_NORM_EPS: f64 = 1e-6;
const L2_CLIP_EPS: f32 = 1e-6;

fn trunk_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 trunk: {message}"))
}

/// `x * rsqrt(mean(x^2) + eps)` over the last dimension.
fn unweighted_rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&scale)?)
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(unweighted_rms_norm(x, eps)?.broadcast_mul(weight)?)
}

/// Scales each vector (last dimension) down to `threshold` when its L2 norm
/// is larger.
fn l2_clip(x: &Tensor, threshold: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let factor = ((norm + L2_CLIP_EPS as f64)?.recip()? * threshold)?.minimum(1.0f32)?;
    Ok(x.broadcast_mul(&factor)?)
}

/// A learned-query read over a stack of states (the reference's
/// `BlockAttentionResidual` and `MessageAttentionResidual`).
///
/// `queries` is `[n, hidden]`; `states` and `partial` are `[seq, hidden]`.
/// Returns `[seq, n, hidden]`. Keys and values are the RMS-normalised states.
fn depth_read(queries: &Tensor, states: &[Tensor], partial: &Tensor) -> Result<Tensor> {
    let hidden = partial.dim(D::Minus1)?;
    let mut stacked = states.to_vec();
    stacked.push(partial.clone());
    let stacked = unweighted_rms_norm(&Tensor::stack(&stacked, 1)?, DEPTH_NORM_EPS)?;
    // [seq, states, n]
    let scores = (stacked.broadcast_matmul(&queries.t()?)? * (hidden as f64).powf(-0.5))?;
    let weights = softmax(&scores, 1)?;
    Ok(weights.transpose(1, 2)?.matmul(&stacked)?)
}

/// Depthwise causal convolution taps, `[kernel, channels]`: row `j`
/// multiplies the token `kernel - 1 - j` steps back.
#[derive(Debug, Clone)]
struct ConvTaps {
    taps: Tensor,
    kernel: usize,
}

impl ConvTaps {
    /// From the stored conv1d weight `[channels, 1, kernel]`.
    fn load(checkpoint: &Inkling2Checkpoint, name: &str, device: &Device) -> Result<Self> {
        let shape = checkpoint.shape(name)?.to_vec();
        let (channels, kernel) = (shape[0], shape[2]);
        let stored = checkpoint.host_f32(name)?;
        let mut taps = vec![0.0f32; kernel * channels];
        for channel in 0..channels {
            for tap in 0..kernel {
                taps[tap * channels + channel] = stored[channel * kernel + tap];
            }
        }
        Ok(Self {
            taps: Tensor::from_vec(taps, (kernel, channels), device)?,
            kernel,
        })
    }
}

/// `[history, x]` and the history the next span starts from.
fn extend(history: &Tensor, x: &Tensor) -> Result<(Tensor, Tensor)> {
    let seq = x.dim(0)?;
    let extended = Tensor::cat(&[history, x], 0)?;
    let next = extended.narrow(0, seq, history.dim(0)?)?.copy()?;
    Ok((extended, next))
}

/// The reference's `ShortConvolution`: `(conv(x) + x) / sqrt(kernel + 1)`.
fn short_conv(x: &Tensor, conv: &ConvTaps, history: &mut Tensor) -> Result<Tensor> {
    let seq = x.dim(0)?;
    let (extended, next) = extend(history, x)?;
    let mut out = x.clone();
    for tap in 0..conv.kernel {
        out = (out
            + extended
                .narrow(0, tap, seq)?
                .broadcast_mul(&conv.taps.narrow(0, tap, 1)?)?)?;
    }
    *history = next;
    Ok((out * ((conv.kernel + 1) as f64).powf(-0.5))?)
}

/// Key or value convolution of a full-attention layer: static taps plus
/// per-token taps projected from the layer input, shared inside each head.
#[derive(Debug, Clone)]
struct DynamicConv {
    /// `[kernel, channels]`, row = backward distance.
    static_taps: Tensor,
    /// `[hidden, heads * kernel]`.
    projection: Tensor,
    kernel: usize,
    heads: usize,
}

impl DynamicConv {
    fn load(
        checkpoint: &Inkling2Checkpoint,
        prefix: &str,
        heads: usize,
        device: &Device,
    ) -> Result<Self> {
        let static_taps = checkpoint.tensor(&format!("{prefix}.static_w"), device, DType::F32)?;
        let kernel = static_taps.dim(0)?;
        Ok(Self {
            static_taps,
            projection: checkpoint
                .tensor(&format!("{prefix}.weight_proj"), device, DType::F32)?
                .t()?
                .contiguous()?,
            kernel,
            heads,
        })
    }

    /// `(sum_lag x[t - lag] * w[t, lag] + x) / sqrt(2 * kernel + 1)`.
    fn forward(&self, x: &Tensor, input: &Tensor, history: &mut Tensor) -> Result<Tensor> {
        let (seq, channels) = x.dims2()?;
        let head_dim = channels / self.heads;
        let (extended, next) = extend(history, x)?;
        let dynamic = input
            .matmul(&self.projection)?
            .reshape((seq, self.heads, self.kernel))?;
        let mut out = x.clone();
        for lag in 0..self.kernel {
            let per_head = dynamic
                .narrow(2, lag, 1)?
                .broadcast_as((seq, self.heads, head_dim))?
                .reshape((seq, channels))?;
            let weights = per_head.broadcast_add(&self.static_taps.narrow(0, lag, 1)?)?;
            out = (out + (extended.narrow(0, self.kernel - 1 - lag, seq)? * weights)?)?;
        }
        *history = next;
        Ok((out * ((2 * self.kernel + 1) as f64).powf(-0.5))?)
    }
}

#[derive(Debug, Clone)]
enum KvConv {
    Static {
        key: ConvTaps,
        value: ConvTaps,
    },
    Dynamic {
        key: DynamicConv,
        value: DynamicConv,
    },
}

#[derive(Debug, Clone)]
enum InputNorm {
    /// Sliding layers: one gain for query, key and value inputs.
    Shared(Tensor),
    /// Full layers: one gain each.
    Split {
        query: Tensor,
        key: Tensor,
        value: Tensor,
    },
}

/// One of the four low-rank gates that mix a loop message into a layer
/// input: `(1 - tanh(F g)) * hidden + tanh(G g) * message`, with
/// `g = D [norm(hidden), norm(message)]`.
#[derive(Debug, Clone)]
struct MessageGate {
    down: Tensor,
    forget: Tensor,
    gain: Tensor,
}

impl MessageGate {
    fn load(checkpoint: &Inkling2Checkpoint, prefix: &str, device: &Device) -> Result<Self> {
        let load = |name: &str| -> Result<Tensor> {
            Ok(checkpoint
                .tensor(&format!("{prefix}.{name}"), device, DType::F32)?
                .t()?
                .contiguous()?)
        };
        Ok(Self {
            down: load("down_weight")?,
            forget: load("forget_weight")?,
            gain: load("gain_weight")?,
        })
    }

    fn forward(&self, hidden: &Tensor, message: &Tensor, eps: f64) -> Result<Tensor> {
        let joined = Tensor::cat(
            &[
                unweighted_rms_norm(hidden, eps)?,
                unweighted_rms_norm(message, eps)?,
            ],
            D::Minus1,
        )?;
        let low = joined.matmul(&self.down)?;
        let forget = low.matmul(&self.forget)?.tanh()?;
        let gain = low.matmul(&self.gain)?.tanh()?;
        Ok(((hidden - (forget * hidden)?)? + (gain * message)?)?)
    }
}

/// Message gates of a loop layer, in message order.
#[derive(Debug, Clone)]
struct Controllers {
    query: MessageGate,
    key: MessageGate,
    value: MessageGate,
    mlp: MessageGate,
}

#[derive(Debug, Clone)]
struct Layer {
    kind: Inkling2LayerKind,
    heads: Inkling2Heads,
    boundary: bool,
    /// Depth-residual queries of the attention input (`None` on layer 0):
    /// one row, or query/key/value rows on full layers.
    attention_query: Option<Tensor>,
    mlp_query: Tensor,
    input_norm: InputNorm,
    /// Projections, stored transposed (`[in, out]`).
    query_proj: Tensor,
    key_proj: Tensor,
    value_proj: Tensor,
    output_proj: Tensor,
    output_gate: Tensor,
    rel_proj: Tensor,
    sink_proj: Tensor,
    query_scale: Tensor,
    rel_scale: Tensor,
    /// `[d_rel, rel_extent]`.
    rel_bank: Tensor,
    kv_conv: KvConv,
    attention_conv: ConvTaps,
    mlp_conv: ConvTaps,
    mlp_norm: Tensor,
    moe: Inkling2Moe,
    controllers: Option<Controllers>,
}

/// The four convolution histories of one layer application: the last
/// `kernel - 1` inputs of the key, value, attention-output and MLP-output
/// convolutions, `[kernel - 1, channels]` each, oldest first.
#[derive(Debug, Clone)]
pub struct Inkling2ConvHistory {
    pub key: Tensor,
    pub value: Tensor,
    pub attention: Tensor,
    pub mlp: Tensor,
}

impl Inkling2ConvHistory {
    pub fn tensors(&self) -> [&Tensor; CONV_STATES_PER_APPLICATION] {
        [&self.key, &self.value, &self.attention, &self.mlp]
    }

    pub fn from_tensors(tensors: [Tensor; CONV_STATES_PER_APPLICATION]) -> Self {
        let [key, value, attention, mlp] = tensors;
        Self {
            key,
            value,
            attention,
            mlp,
        }
    }
}

/// Convolution state of a sequence: one history set per layer application.
#[derive(Debug, Clone)]
pub struct Inkling2ConvState {
    pub histories: Vec<Inkling2ConvHistory>,
}

/// Where a sequence's keys and values live. The trunk appends a span's keys
/// and values and gets back the context its queries attend over.
pub trait Inkling2Context {
    /// Appends `keys` and `values` (`[seq, kv_heads, head_dim]`, F32) of
    /// layer application `application` at positions `first..first + seq`
    /// and returns `(keys, values, oldest)`: the context in F32, oldest
    /// first, ending with the span, and the position of its first row. With
    /// a `window`, the context holds at least the `window` keys the span's
    /// first query sees.
    fn append(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        window: Option<usize>,
        first: usize,
    ) -> Result<(Tensor, Tensor, usize)>;
}

/// Keys and values held as plain tensors (the reference state).
#[derive(Debug, Clone)]
pub struct Inkling2DenseContext {
    /// Per application: `[keys, kv_heads, head_dim]` keys and values.
    held: Vec<Option<(Tensor, Tensor)>>,
}

impl Inkling2Context for Inkling2DenseContext {
    fn append(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        window: Option<usize>,
        first: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        let seq = keys.dim(0)?;
        let slot = self
            .held
            .get_mut(application)
            .ok_or_else(|| trunk_error("the state belongs to another model"))?;
        let (mut keys, mut values) = match slot.as_ref() {
            Some((held_keys, held_values)) => (
                Tensor::cat(&[held_keys, keys], 0)?,
                Tensor::cat(&[held_values, values], 0)?,
            ),
            None => (keys.clone(), values.clone()),
        };
        // Drop what no later query can see: the span's first query sees
        // `window` keys, itself included.
        let held = keys.dim(0)?;
        if let Some(window) = window {
            let needed = (window + seq - 1).min(held);
            if needed < held {
                keys = keys.narrow(0, held - needed, needed)?;
                values = values.narrow(0, held - needed, needed)?;
            }
        }
        let oldest = first + seq - keys.dim(0)?;
        *slot = Some((keys.clone(), values.clone()));
        Ok((keys, values, oldest))
    }
}

/// Everything a sequence carries from one token to the next, held by the
/// trunk itself.
#[derive(Debug, Clone)]
pub struct Inkling2SequenceState {
    position: usize,
    conv: Inkling2ConvState,
    context: Inkling2DenseContext,
}

impl Inkling2SequenceState {
    /// Tokens the sequence has consumed.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Keys held by each layer application.
    pub fn held_keys(&self) -> Vec<usize> {
        self.context
            .held
            .iter()
            .map(|held| held.as_ref().map_or(0, |(keys, _)| keys.dims()[0]))
            .collect()
    }
}

/// Activations recorded for parity tests.
#[derive(Debug, Default)]
pub struct Inkling2Trace {
    pub entries: Vec<(String, Tensor)>,
}

#[derive(Debug, Clone)]
pub struct Inkling2Trunk {
    config: Inkling2Config,
    applications: Vec<Inkling2Application>,
    device: Device,
    embedding: Tensor,
    embedding_norm: Tensor,
    final_norm: Tensor,
    /// `[hidden, unpadded_vocab]`.
    unembedding: Tensor,
    output_query: Tensor,
    /// Per loop layer, `[4, hidden]`: query, key, value, mlp.
    message_queries: Vec<Tensor>,
    layers: Vec<Layer>,
}

impl Inkling2Trunk {
    pub fn load(
        checkpoint: &Inkling2Checkpoint,
        config: &Inkling2Config,
        device: &Device,
    ) -> Result<Self> {
        let dense = |name: &str| checkpoint.tensor(name, device, DType::F32);
        let transposed = |name: &str| -> Result<Tensor> { Ok(dense(name)?.t()?.contiguous()?) };
        let t = TEXT_PREFIX;
        let h = config.hidden;

        // Two stored tensors of `[2 * loop_layers, hidden]`: (query, mlp)
        // and (key, value) of each loop layer. Messages are used in the
        // order query, key, value, mlp.
        let loop_layers = config.loop_layers();
        let queries =
            dense(&format!("{t}.middle_cycle_cross_pass_queries"))?.reshape((loop_layers, 2, h))?;
        let kv_queries = dense(&format!("{t}.middle_cycle_cross_pass_kv_queries"))?.reshape((
            loop_layers,
            2,
            h,
        ))?;
        let message_queries = (0..loop_layers)
            .map(|index| -> Result<Tensor> {
                let pair = queries.narrow(0, index, 1)?.squeeze(0)?;
                let kv = kv_queries.narrow(0, index, 1)?.squeeze(0)?;
                Ok(Tensor::cat(
                    &[pair.narrow(0, 0, 1)?, kv, pair.narrow(0, 1, 1)?],
                    0,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let l = layer_prefix(layer);
            let heads = config.heads(layer);
            let kind = config.layer_kinds[layer];
            // `attn_queries.{i}` is the attention-input query of layer `i + 1`.
            let attention_query = if layer == 0 {
                None
            } else {
                let query = dense(&format!("{t}.attn_res.attn_queries.{}", layer - 1))?;
                let rows = query.elem_count() / h;
                Some(query.reshape((rows, h))?)
            };
            let (input_norm, kv_conv) = match kind {
                Inkling2LayerKind::Sliding => (
                    InputNorm::Shared(dense(&format!("{l}.attn_norm.weight"))?),
                    KvConv::Static {
                        key: ConvTaps::load(
                            checkpoint,
                            &format!("{l}.attn.k_sconv.weight"),
                            device,
                        )?,
                        value: ConvTaps::load(
                            checkpoint,
                            &format!("{l}.attn.v_sconv.weight"),
                            device,
                        )?,
                    },
                ),
                Inkling2LayerKind::Full => (
                    InputNorm::Split {
                        query: dense(&format!("{l}.q_input_norm.weight"))?,
                        key: dense(&format!("{l}.k_input_norm.weight"))?,
                        value: dense(&format!("{l}.v_input_norm.weight"))?,
                    },
                    KvConv::Dynamic {
                        key: DynamicConv::load(
                            checkpoint,
                            &format!("{l}.attn.k_sconv"),
                            heads.kv_heads,
                            device,
                        )?,
                        value: DynamicConv::load(
                            checkpoint,
                            &format!("{l}.attn.v_sconv"),
                            heads.kv_heads,
                            device,
                        )?,
                    },
                ),
            };
            let controllers = if config.is_loop_layer(layer) {
                let gate = |which: &str| {
                    MessageGate::load(checkpoint, &format!("{l}.controllers.{which}"), device)
                };
                Some(Controllers {
                    query: gate("attn")?,
                    key: gate("k")?,
                    value: gate("v")?,
                    mlp: gate("mlp")?,
                })
            } else {
                None
            };
            layers.push(Layer {
                kind,
                heads,
                boundary: config.is_block_boundary(layer),
                attention_query,
                mlp_query: dense(&format!("{t}.attn_res.mlp_queries.{layer}"))?.reshape((1, h))?,
                input_norm,
                query_proj: transposed(&format!("{l}.attn.wq_du.weight"))?,
                key_proj: transposed(&format!("{l}.attn.wk_dv.weight"))?,
                value_proj: transposed(&format!("{l}.attn.wv_dv.weight"))?,
                output_proj: transposed(&format!("{l}.attn.wo_ud.weight"))?,
                output_gate: transposed(&format!("{l}.attn.gate_attn.weight"))?,
                rel_proj: transposed(&format!("{l}.attn.wr_du.weight"))?,
                sink_proj: transposed(&format!("{l}.attn.attn_sink_proj.weight"))?,
                query_scale: dense(&format!("{l}.attn.q_l2_clip.scale"))?,
                rel_scale: dense(&format!("{l}.attn.rel_l2_clip.scale"))?,
                rel_bank: dense(&format!("{l}.attn.rel_logits_proj.proj"))?,
                kv_conv,
                attention_conv: ConvTaps::load(
                    checkpoint,
                    &format!("{l}.attn_sconv.weight"),
                    device,
                )?,
                mlp_conv: ConvTaps::load(checkpoint, &format!("{l}.mlp_sconv.weight"), device)?,
                mlp_norm: dense(&format!("{l}.mlp_norm.weight"))?,
                moe: Inkling2Moe::new(
                    config,
                    checkpoint.host_f32(&format!("{l}.mlp.gate.weight"))?,
                    checkpoint.host_f32(&format!("{l}.mlp.gate.bias"))?,
                    checkpoint.experts(layer, ExpertProjection::GateUp)?,
                    checkpoint.experts(layer, ExpertProjection::Down)?,
                )?,
                controllers,
            });
        }

        Ok(Self {
            applications: config.applications(),
            device: device.clone(),
            embedding: dense(&format!("{t}.embed.weight"))?,
            embedding_norm: dense(&format!("{t}.embed_norm.weight"))?,
            final_norm: dense(&format!("{t}.norm.weight"))?,
            // Stored padded to the embedding's row count; only the first
            // `unpadded_vocab` rows are the model's.
            unembedding: checkpoint
                .tensor_rows(
                    &format!("{t}.unembed.weight"),
                    config.unpadded_vocab,
                    device,
                    DType::F32,
                )?
                .t()?
                .contiguous()?,
            output_query: dense(&format!("{t}.attn_res.output_query"))?.reshape((1, h))?,
            message_queries,
            layers,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs every layer's experts on the trunk's device
    /// ([`Inkling2Moe::upload`]). On an error some layers may already be
    /// uploaded; [`Self::experts_to_host`] undoes that.
    pub fn experts_to_device(&mut self) -> Result<()> {
        let device = self.device.clone();
        for layer in &mut self.layers {
            layer.moe.upload(&device)?;
        }
        Ok(())
    }

    /// Runs every layer's experts on the host again.
    pub fn experts_to_host(&mut self) {
        for layer in &mut self.layers {
            layer.moe.drop_device();
        }
    }

    pub fn experts_on_device(&self) -> bool {
        self.layers.iter().all(|layer| layer.moe.on_device())
    }

    /// Bytes of the experts' device copies.
    pub fn expert_device_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| layer.moe.device_resident_bytes())
            .sum()
    }

    /// Compares the device experts with the host path on unstructured
    /// input, one layer of each stored format, and returns the worst error
    /// relative to the largest output. The host path reads the checkpoint's
    /// own bytes, so an upload that changed them shows here.
    pub fn check_device_experts(&self) -> Result<f32> {
        let mut seen = Vec::new();
        let mut worst = 0f32;
        for layer in &self.layers {
            let format = layer.moe.gate_up().format();
            if seen.contains(&format) {
                continue;
            }
            seen.push(format);
            let rows = 3usize;
            let mut state = 0x9E37_79B9u32;
            let values = (0..rows * self.config.hidden)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect::<Vec<_>>();
            let want = layer.moe.forward(&values);
            let input = Tensor::from_vec(values, (rows, self.config.hidden), &self.device)?;
            let got = layer
                .moe
                .forward_tensor(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let scale = want.iter().fold(f32::MIN_POSITIVE, |m, v| m.max(v.abs()));
            for (got, want) in got.iter().zip(&want) {
                let error = (got - want).abs() / scale;
                worst = if error.is_finite() {
                    worst.max(error)
                } else {
                    f32::INFINITY
                };
            }
        }
        Ok(worst)
    }

    /// Bytes the packed experts keep resident.
    pub fn expert_resident_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| layer.moe.resident_bytes())
            .sum()
    }

    /// Zeroed convolution histories (the state before the first token).
    pub fn new_conv_state(&self) -> Result<Inkling2ConvState> {
        let history = self.config.conv_kernel - 1;
        let zeros = |channels: usize| Tensor::zeros((history, channels), DType::F32, &self.device);
        let histories = self
            .applications
            .iter()
            .map(|application| -> Result<Inkling2ConvHistory> {
                let kv = self.config.heads(application.layer).kv_width();
                Ok(Inkling2ConvHistory {
                    key: zeros(kv)?,
                    value: zeros(kv)?,
                    attention: zeros(self.config.hidden)?,
                    mlp: zeros(self.config.hidden)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Inkling2ConvState { histories })
    }

    pub fn new_state(&self) -> Result<Inkling2SequenceState> {
        Ok(Inkling2SequenceState {
            position: 0,
            conv: self.new_conv_state()?,
            context: Inkling2DenseContext {
                held: vec![None; self.applications.len()],
            },
        })
    }

    /// `1 + alpha * ln(max((position + 1) / n_floor, 1))` per position, on
    /// the host in F64 (device `ln` is not exact on every backend).
    fn log_scale(&self, first: usize, seq: usize) -> Result<Tensor> {
        let values = (first..first + seq)
            .map(|position| {
                let ratio = ((position + 1) as f64 / self.config.log_scaling_n_floor).max(1.0);
                (1.0 + self.config.log_scaling_alpha * ratio.ln()) as f32
            })
            .collect::<Vec<_>>();
        Ok(Tensor::from_vec(values, (seq, 1, 1), &self.device)?)
    }

    /// Embeddings of `tokens`, `[seq, hidden]`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= self.config.vocab)
        {
            return Err(trunk_error(format!(
                "token {token} is outside the vocabulary"
            )));
        }
        let ids = Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?;
        rms_norm(
            &self.embedding.index_select(&ids, 0)?,
            &self.embedding_norm,
            self.config.rms_norm_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        index: usize,
        layer: &Layer,
        [query_input, key_input, value_input]: [&Tensor; 3],
        first: usize,
        history: &mut Inkling2ConvHistory,
        context: &mut dyn Inkling2Context,
    ) -> Result<Tensor> {
        let c = &self.config;
        let g = layer.heads;
        let seq = query_input.dim(0)?;
        let full = layer.kind == Inkling2LayerKind::Full;
        let log_scale = if full {
            Some(self.log_scale(first, seq)?)
        } else {
            None
        };

        let mut queries = query_input
            .matmul(&layer.query_proj)?
            .reshape((seq, g.heads, g.head_dim))?
            .broadcast_mul(&layer.query_scale)?;
        queries = l2_clip(&queries, c.qk_l2_clip_threshold)?;
        if let Some(scale) = &log_scale {
            queries = queries.broadcast_mul(scale)?;
        }

        let keys = key_input.matmul(&layer.key_proj)?;
        let values = value_input.matmul(&layer.value_proj)?;
        let (keys, values) = match &layer.kv_conv {
            KvConv::Static { key, value } => (
                short_conv(&keys, key, &mut history.key)?,
                short_conv(&values, value, &mut history.value)?,
            ),
            KvConv::Dynamic { key, value } => (
                key.forward(&keys, key_input, &mut history.key)?,
                value.forward(&values, value_input, &mut history.value)?,
            ),
        };
        let keys = l2_clip(
            &keys.reshape((seq, g.kv_heads, g.head_dim))?,
            c.qk_l2_clip_threshold,
        )?;
        let values = unweighted_rms_norm(
            &values.reshape((seq, g.kv_heads, g.head_dim))?,
            c.rms_norm_eps,
        )?;

        let (keys, values, oldest) = context.append(index, &keys, &values, g.window, first)?;
        let held_now = keys.dim(0)?;
        if values.dim(0)? != held_now || oldest + held_now != first + seq {
            return Err(trunk_error(
                "the attention context does not end at the span",
            ));
        }

        // Per (query, key): the bank column to read, whether the bias
        // applies, and whether the key is visible.
        let mut columns = Vec::with_capacity(seq * held_now);
        let mut inside = Vec::with_capacity(seq * held_now);
        let mut hidden_mask = Vec::with_capacity(seq * held_now);
        for query in 0..seq {
            let query_position = (first + query) as i64;
            for key in 0..held_now {
                let distance = query_position - (oldest + key) as i64;
                let visible =
                    distance >= 0 && g.window.is_none_or(|window| distance < window as i64);
                let biased = distance >= 0 && distance < g.rel_extent as i64;
                columns.push(distance.clamp(0, g.rel_extent as i64 - 1) as u32);
                inside.push(if biased { 1.0f32 } else { 0.0 });
                hidden_mask.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
            }
        }
        let shape = (1, seq, held_now);
        let columns = Tensor::from_vec(columns, shape, &self.device)?
            .broadcast_as((g.heads, seq, held_now))?
            .contiguous()?;
        let inside = Tensor::from_vec(inside, shape, &self.device)?;
        let hidden_mask = Tensor::from_vec(hidden_mask, shape, &self.device)?;

        let mut relative = query_input
            .matmul(&layer.rel_proj)?
            .reshape((seq, g.heads, c.d_rel))?
            .broadcast_mul(&layer.rel_scale)?;
        relative = l2_clip(&relative, c.rel_l2_clip_threshold)?;
        if let Some(scale) = &log_scale {
            relative = relative.broadcast_mul(scale)?;
        }
        // [heads, seq, rel_extent]
        let profiles = relative
            .broadcast_matmul(&layer.rel_bank)?
            .transpose(0, 1)?
            .contiguous()?;
        let bias = profiles.gather(&columns, 2)?.broadcast_mul(&inside)?;

        let groups = g.heads / g.kv_heads;
        let per_head = |kv: &Tensor| -> Result<Tensor> {
            Ok(kv
                .transpose(0, 1)?
                .unsqueeze(1)?
                .broadcast_as((g.kv_heads, groups, held_now, g.head_dim))?
                .reshape((g.heads, held_now, g.head_dim))?)
        };
        let key_heads = per_head(&keys)?;
        let value_heads = per_head(&values)?;
        let queries = queries.transpose(0, 1)?.contiguous()?;
        let scores = (queries.matmul(&key_heads.transpose(1, 2)?.contiguous()?)?
            * (g.head_dim as f64).powf(-0.5))?;
        let scores = (scores + bias)?.broadcast_add(&hidden_mask)?;
        // One extra logit per query whose value is zero.
        let sinks = query_input
            .matmul(&layer.sink_proj)?
            .t()?
            .unsqueeze(2)?
            .contiguous()?;
        let probabilities =
            softmax(&Tensor::cat(&[&scores, &sinks], 2)?, 2)?.narrow(2, 0, held_now)?;
        let attended = probabilities
            .contiguous()?
            .matmul(&value_heads)?
            .transpose(0, 1)?
            .reshape((seq, g.query_width()))?;

        let gate = sigmoid(&query_input.matmul(&layer.output_gate)?)?;
        Ok((attended * gate)?.matmul(&layer.output_proj)?)
    }

    fn moe(&self, layer: &Layer, input: &Tensor) -> Result<Tensor> {
        layer.moe.forward_tensor(input)
    }

    /// One layer application over a span. `messages` is `[seq, 4, hidden]`
    /// (query, key, value, mlp) on loop passes after the first.
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        index: usize,
        hidden: &Tensor,
        block_states: &[Tensor],
        messages: Option<&Tensor>,
        first: usize,
        history: &mut Inkling2ConvHistory,
        context: &mut dyn Inkling2Context,
        trace: &mut Option<&mut Inkling2Trace>,
    ) -> Result<Tensor> {
        let layer = &self.layers[self.applications[index].layer];
        let eps = self.config.rms_norm_eps;
        let message = |slot: usize| -> Result<Option<Tensor>> {
            messages
                .map(|messages| Ok(messages.narrow(1, slot, 1)?.squeeze(1)?))
                .transpose()
        };
        let controllers = || {
            layer
                .controllers
                .as_ref()
                .ok_or_else(|| trunk_error("a message reached a layer outside the loop"))
        };

        // Inputs of the query, key and value projections.
        let inputs = match &layer.attention_query {
            None => vec![hidden.clone()],
            Some(query) => {
                let read = depth_read(query, block_states, hidden)?;
                (0..query.dim(0)?)
                    .map(|row| Ok(read.narrow(1, row, 1)?.squeeze(1)?))
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let pick = |slot: usize| inputs[slot.min(inputs.len() - 1)].clone();
        let (mut query_input, mut key_input, mut value_input) = (pick(0), pick(1), pick(2));
        if let (Some(q), Some(k), Some(v)) = (message(0)?, message(1)?, message(2)?) {
            let gates = controllers()?;
            query_input = gates.query.forward(&query_input, &q, eps)?;
            key_input = gates.key.forward(&key_input, &k, eps)?;
            value_input = gates.value.forward(&value_input, &v, eps)?;
        }
        let (query_input, key_input, value_input) = match &layer.input_norm {
            InputNorm::Shared(weight) => (
                rms_norm(&query_input, weight, eps)?,
                rms_norm(&key_input, weight, eps)?,
                rms_norm(&value_input, weight, eps)?,
            ),
            InputNorm::Split { query, key, value } => (
                rms_norm(&query_input, query, eps)?,
                rms_norm(&key_input, key, eps)?,
                rms_norm(&value_input, value, eps)?,
            ),
        };

        let attended = self.attention(
            index,
            layer,
            [&query_input, &key_input, &value_input],
            first,
            history,
            context,
        )?;
        let attended = short_conv(&attended, &layer.attention_conv, &mut history.attention)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .entries
                .push((format!("app.{index}.attn"), attended.clone()));
        }

        // A block's first layer starts a fresh residual and adds its own
        // input to the states the MLP read sees.
        let mut states = block_states.to_vec();
        let residual = if layer.boundary {
            states.push(hidden.clone());
            attended
        } else {
            (hidden + attended)?
        };
        let mut mlp_input = depth_read(&layer.mlp_query, &states, &residual)?.squeeze(1)?;
        if let Some(mlp_message) = message(3)? {
            mlp_input = controllers()?.mlp.forward(&mlp_input, &mlp_message, eps)?;
        }
        let mlp = self.moe(layer, &rms_norm(&mlp_input, &layer.mlp_norm, eps)?)?;
        let mlp = short_conv(&mlp, &layer.mlp_conv, &mut history.mlp)?;
        let out = (residual + mlp)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .entries
                .push((format!("app.{index}.out"), out.clone()));
        }
        Ok(out)
    }

    /// Runs `embeddings` (`[seq, hidden]`, the next `seq` tokens of the
    /// sequence) through the trunk and returns the final hidden states.
    pub fn forward_embeddings(
        &self,
        embeddings: &Tensor,
        state: &mut Inkling2SequenceState,
        trace: Option<&mut Inkling2Trace>,
    ) -> Result<Tensor> {
        let seq = embeddings.dim(0)?;
        let out = self.forward_span(
            embeddings,
            state.position,
            &mut state.conv,
            &mut state.context,
            trace,
        )?;
        state.position += seq;
        Ok(out)
    }

    /// [`Self::forward_embeddings`] over state the caller owns: `conv` is
    /// advanced in place and the span's keys and values go to `context`.
    /// `first` is the sequence position of the span's first token. On an
    /// error `conv` may be part-way through the span; callers that need to
    /// retry run on a clone.
    pub fn forward_span(
        &self,
        embeddings: &Tensor,
        first: usize,
        conv: &mut Inkling2ConvState,
        context: &mut dyn Inkling2Context,
        mut trace: Option<&mut Inkling2Trace>,
    ) -> Result<Tensor> {
        let c = &self.config;
        let (seq, hidden_size) = embeddings.dims2()?;
        if seq == 0 || hidden_size != c.hidden {
            return Err(trunk_error("empty span or wrong embedding width"));
        }
        if conv.histories.len() != self.applications.len() {
            return Err(trunk_error("the state belongs to another model"));
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace
                .entries
                .push(("embed".to_string(), embeddings.clone()));
        }
        let mut hidden = embeddings.clone();
        let mut block_states: Vec<Tensor> = Vec::new();
        let mut index = 0usize;

        macro_rules! run {
            ($layer:expr, $messages:expr) => {{
                let out = self.apply(
                    index,
                    &hidden,
                    &block_states,
                    $messages,
                    first,
                    &mut conv.histories[index],
                    &mut *context,
                    &mut trace,
                )?;
                if self.layers[$layer].boundary {
                    block_states.push(hidden.clone());
                }
                hidden = out;
                index += 1;
            }};
        }

        for layer in 0..c.loop_start {
            run!(layer, None);
        }
        let before_loop = (hidden.clone(), block_states.clone());
        for layer in c.loop_start..c.loop_end {
            run!(layer, None);
        }
        for _ in 1..c.loop_passes {
            // The states the previous pass added, without its first (the
            // state the loop started from), plus that pass's output.
            let previous_states = block_states[before_loop.1.len() + 1..].to_vec();
            let previous_hidden = hidden.clone();
            hidden = before_loop.0.clone();
            block_states = before_loop.1.clone();
            for layer in c.loop_start..c.loop_end {
                let messages = depth_read(
                    &self.message_queries[layer - c.loop_start],
                    &previous_states,
                    &previous_hidden,
                )?;
                run!(layer, Some(&messages));
            }
        }
        for layer in c.loop_end..c.layers {
            run!(layer, None);
        }
        debug_assert_eq!(index, self.applications.len());

        let read = depth_read(&self.output_query, &block_states, &hidden)?.squeeze(1)?;
        let out = rms_norm(&read, &self.final_norm, c.rms_norm_eps)?;
        if let Some(trace) = trace {
            trace
                .entries
                .push(("final_hidden".to_string(), out.clone()));
        }
        Ok(out)
    }

    /// [`Self::forward_embeddings`] over token ids.
    pub fn forward(
        &self,
        tokens: &[u32],
        state: &mut Inkling2SequenceState,
        trace: Option<&mut Inkling2Trace>,
    ) -> Result<Tensor> {
        self.forward_embeddings(&self.embed(tokens)?, state, trace)
    }

    /// Soft-capped logits of final hidden states, `[seq, unpadded_vocab]`.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let cap = self.config.final_logit_softcap;
        let logits = hidden.matmul(&self.unembedding)?;
        Ok(((logits / cap)?.tanh()? * cap)?)
    }
}

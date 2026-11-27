//! The block drafter published with the model (`mtp.safetensors`): a small
//! dense transformer that proposes the next tokens of a sequence in one
//! pass, for the main model to verify (speculative decoding).
//!
//! This file is the drafter's own arithmetic and its loader. It does not
//! capture the main model's states, verify drafts or drive a decode loop:
//! its caller hands it the states and decides which tokens to keep.
//!
//! One round, for a sequence whose positions `0..n` have been run through
//! the main model and whose next token, the **anchor**, has been chosen but
//! not yet run (it sits at position `n`):
//!
//! * **Taps.** Every committed position contributes four of the main
//!   model's hidden states, `[4, hidden]`, in `target_layer_ids` order. An
//!   id below the number of layer applications is the state entering that
//!   application (the running partial block of the depth-residual stack,
//!   unmixed and unnormalised); the last id is the final-norm output, the
//!   unembedding's input.
//! * **Fusion.** `fuse_weight` has one row per consumer: one per draft
//!   layer, then one for the head. A row is a softmax over the taps; the
//!   weighted sum of the taps is RMS-normalised with that consumer's
//!   `dflare_norm` gain.
//! * **Context.** Each layer turns a committed position's fused state into
//!   one key and one value: `k = rotate(rms_per_head(k_proj_ctx(c)), p)`,
//!   `v = v_proj_ctx(c)`. They never change once written.
//! * **Block.** `block_size` rows at positions `n..n + block_size`: the
//!   anchor's embedding as the main model feeds it to its first layer
//!   (after its embedding norm), then the learned `mask_embed` in every
//!   other row. Pre-norm layers with RMS norms; the attention normalises
//!   queries and keys per head (a gain per head channel), rotates them by
//!   absolute position (frequencies `theta^(-2j/d)` over the whole head, a
//!   head's first half paired with its second), leaves values alone, and
//!   takes one softmax over the context's keys followed by the block's own,
//!   scaled by `head_dim^-0.5`. Rows see the whole block, and all of them
//!   the same context: the `sliding_window_size` positions before the
//!   anchor in `local_layer_ids` layers, everything in the others. A
//!   key/value head serves `heads / kv_heads` neighbouring query heads. The
//!   feed-forward is `down(silu(gate(x)) * up(x))`.
//! * **Head.** Each mask row predicts the token at its own position. Its
//!   base logits are the final-norm hidden state times the main model's
//!   unembedding. A gated recurrent cell, run in order over the mask rows,
//!   adds a low-rank correction: `rnn_proj([state | markov_w1[previous] |
//!   hidden | f_rnn])` splits into (gate, candidate, output); `state' =
//!   sigmoid(gate) * state + (1 - sigmoid(gate)) * tanh(candidate)`; the
//!   logits are `cap * tanh((base + unembed @ hidden_out(tanh(output))) /
//!   cap)`, cut to the unpadded vocabulary. `previous` starts as the anchor
//!   and is then whichever token the caller picked; the state starts at
//!   zero; `f_rnn` is the head's fusion of the last committed position.
//!
//! **Source.** No reference implementation of the drafter is published;
//! this follows the serving engines, which agree with one another on every
//! point above: vLLM at `1185d98ea797be7a526d98f4c262cbdf36abefc2`
//! (`vllm/models/inkling2/nvidia/dspark.py`,
//! `vllm/models/inkling2/nvidia/ops/dspark.py`,
//! `vllm/models/inkling2/speculator.py`, `vllm/models/inkling2/configs.py`
//! and the torch references in `tests/models/inkling2/test_dspark.py`),
//! cross-read against SGLang at `4c085b66d69ffadcd40d258a8181e1927157b25d`
//! (`python/sglang/srt/models/inkling2_mtp.py`) and llama.cpp at
//! `ca123ba76311290ec6ab041dd2058fed8d9f03d2` (`conversion/inkling2.py`,
//! `src/models/dspark.cpp`).
//!
//! **Verified.** Every step against a torch implementation written from
//! those sources, on a tiny random drafter in the published tensor layout
//! (`scripts/fixtures/generate_inkling2_drafter_golden.py`, tests in
//! `drafter_tests`), on the host and on the Metal device. That torch
//! implementation agrees with a second, line-by-line transcription of
//! vLLM's forward and with the reference expressions of vLLM's test.
//!
//! **Not verified.** Nothing here has run on the published weights or been
//! compared with an engine's output. The engines round activations to BF16
//! at fixed points (the weights are BF16); this file computes in F32 from
//! the same weight values, so its logits differ from theirs by that
//! rounding.
//!
//! Weights are held in F32 for now. Every weight is read through [`Linear`]
//! or as a plain gain, so the dtype they are held in can change in one
//! place.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};
use half::bf16;
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;

use super::checkpoint::MTP_FILE;
use super::config::CONFIG_FILE;
use crate::error::{Error, Result};

/// The only head the published drafter has, and the only one built here.
const MARKOV_HEAD_TYPE: &str = "rnn_hidden";

fn load_error(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Inkling 2 drafter: {message}"))
}

fn drafter_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 drafter: {message}"))
}

#[derive(Debug, Clone, Deserialize)]
struct RawMtp {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    rms_norm_eps: f64,
    rope_theta: f64,
    local_layer_ids: Vec<usize>,
    sliding_window_size: usize,
    block_size: usize,
    markov_rank: usize,
    markov_head_type: String,
    target_layer_ids: Vec<usize>,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawLoop {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct RawText {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default)]
    loop_config: Option<RawLoop>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    text_config: RawText,
    #[serde(default)]
    mtp_config: Option<RawMtp>,
}

/// The drafter's geometry: `mtp_config` of the model's `config.json`, with
/// the few facts about the main model the drafter depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2DrafterConfig {
    /// Width of the drafter's residual stream; the main model's too.
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Whether each layer sees only the window before the anchor.
    pub local: Vec<bool>,
    /// Committed positions a local layer sees before the anchor.
    pub window: usize,
    /// Rows of a block: the anchor and `block - 1` drafted positions.
    pub block: usize,
    pub markov_rank: usize,
    /// Which main-model states a position contributes, in order: the state
    /// entering each listed layer application, then (the last id, the
    /// number of applications) the final-norm output.
    pub taps: Vec<usize>,
    /// Rows of the unembedding and of `markov_w1`.
    pub padded_vocab: usize,
    /// Tokens the logits cover.
    pub vocab: usize,
    /// The main model's logit soft cap, which the drafter's logits share.
    pub softcap: Option<f64>,
    /// Positions the drafter was trained over, when the config says.
    pub max_positions: Option<usize>,
}

impl Inkling2DrafterConfig {
    /// Reads `<dir>/config.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|err| load_error(format!("{}: {err}", path.display())))?;
        Self::from_json(&text)
    }

    /// From the text of a `config.json`.
    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|err| load_error(format!("config: {err}")))?;
        let mtp = raw
            .mtp_config
            .ok_or_else(|| load_error("the config has no `mtp_config`"))?;
        let main = raw.text_config;
        let fail = |message: String| Err(load_error(format!("config: {message}")));

        if mtp.markov_head_type != MARKOV_HEAD_TYPE {
            return fail(format!(
                "head type `{}`, not `{MARKOV_HEAD_TYPE}`",
                mtp.markov_head_type
            ));
        }
        if mtp.q_bias || mtp.o_bias {
            return fail("attention biases are not part of this drafter".into());
        }
        if mtp.hidden_size != main.hidden_size {
            return fail(format!(
                "the drafter is {} wide and the main model {}",
                mtp.hidden_size, main.hidden_size
            ));
        }
        if [
            mtp.hidden_size,
            mtp.num_hidden_layers,
            mtp.num_attention_heads,
            mtp.num_key_value_heads,
            mtp.head_dim,
            mtp.intermediate_size,
            mtp.sliding_window_size,
            mtp.markov_rank,
        ]
        .contains(&0)
        {
            return fail("a size is zero".into());
        }
        if mtp.num_attention_heads % mtp.num_key_value_heads != 0 {
            return fail(format!(
                "{} query heads over {} key/value heads",
                mtp.num_attention_heads, mtp.num_key_value_heads
            ));
        }
        // The rotation pairs the first half of a head with its second.
        if mtp.head_dim % 2 != 0 {
            return fail(format!("an odd head width, {}", mtp.head_dim));
        }
        if mtp.block_size < 2 {
            return fail("a block needs the anchor and at least one drafted row".into());
        }
        if !(mtp.rms_norm_eps > 0.0 && mtp.rope_theta > 0.0) {
            return fail("the norm epsilon and the rotation base must be positive".into());
        }
        let mut local = vec![false; mtp.num_hidden_layers];
        for layer in &mtp.local_layer_ids {
            match local.get_mut(*layer) {
                Some(flag) => *flag = true,
                None => return fail(format!("local layer {layer} does not exist")),
            }
        }
        // The taps name layer applications in execution order, loop passes
        // counted, and end with the final-norm output.
        let applications = main.num_hidden_layers
            + main.loop_config.as_ref().map_or(0, |l| {
                l.num_passes.saturating_sub(1) * l.core_end.saturating_sub(l.core_start)
            });
        let taps = mtp.target_layer_ids;
        if taps.is_empty()
            || taps.windows(2).any(|pair| pair[0] >= pair[1])
            || taps.last() != Some(&applications)
        {
            return fail(format!(
                "taps {taps:?} must rise and end with the final-norm output, id {applications}"
            ));
        }
        if main.unpadded_vocab_size == 0 || main.unpadded_vocab_size > main.vocab_size {
            return fail(format!(
                "{} tokens in a vocabulary padded to {}",
                main.unpadded_vocab_size, main.vocab_size
            ));
        }
        Ok(Self {
            hidden: mtp.hidden_size,
            layers: mtp.num_hidden_layers,
            heads: mtp.num_attention_heads,
            kv_heads: mtp.num_key_value_heads,
            head_dim: mtp.head_dim,
            intermediate: mtp.intermediate_size,
            rms_norm_eps: mtp.rms_norm_eps,
            rope_theta: mtp.rope_theta,
            local,
            window: mtp.sliding_window_size,
            block: mtp.block_size,
            markov_rank: mtp.markov_rank,
            taps,
            padded_vocab: main.vocab_size,
            vocab: main.unpadded_vocab_size,
            softcap: main.final_logit_softcapping.filter(|cap| *cap > 0.0),
            max_positions: mtp.max_position_embeddings,
        })
    }

    /// Tokens one block drafts: every row but the anchor's.
    pub fn drafted(&self) -> usize {
        self.block - 1
    }

    /// The first committed position `layer` sees when the anchor sits at
    /// `anchor_position`. Every row of the block sees the same span: the
    /// window is anchored at the block's start and does not slide per row.
    pub fn first_visible(&self, layer: usize, anchor_position: usize) -> usize {
        if self.local[layer] {
            anchor_position.saturating_sub(self.window)
        } else {
            0
        }
    }

    /// Every tensor of `mtp.safetensors` and its shape.
    fn census(&self) -> Vec<(String, Vec<usize>)> {
        let (hidden, rank) = (self.hidden, self.markov_rank);
        let query = self.heads * self.head_dim;
        let key_value = self.kv_heads * self.head_dim;
        let mut census = vec![
            (
                "fuse_weight".to_string(),
                vec![self.layers + 1, self.taps.len()],
            ),
            ("mask_embed".to_string(), vec![hidden]),
            ("norm.weight".to_string(), vec![hidden]),
            ("markov_head.dflare_norm.weight".to_string(), vec![hidden]),
            (
                "markov_head.markov_w1.weight".to_string(),
                vec![self.padded_vocab, rank],
            ),
            (
                "markov_head.rnn_proj.weight".to_string(),
                vec![3 * rank, 2 * rank + 2 * hidden],
            ),
            ("markov_head.rnn_proj.bias".to_string(), vec![3 * rank]),
            (
                "markov_head.hidden_out.weight".to_string(),
                vec![hidden, rank],
            ),
            ("markov_head.hidden_out.bias".to_string(), vec![hidden]),
        ];
        for layer in 0..self.layers {
            let l = format!("layers.{layer}");
            for (name, shape) in [
                ("dflare_norm.weight", vec![hidden]),
                ("input_layernorm.weight", vec![hidden]),
                ("post_attention_layernorm.weight", vec![hidden]),
                ("self_attn.q_norm.weight", vec![self.head_dim]),
                ("self_attn.k_norm.weight", vec![self.head_dim]),
                ("self_attn.k_norm_ctx.weight", vec![self.head_dim]),
                ("self_attn.q_proj.weight", vec![query, hidden]),
                ("self_attn.k_proj.weight", vec![key_value, hidden]),
                ("self_attn.v_proj.weight", vec![key_value, hidden]),
                ("self_attn.k_proj_ctx.weight", vec![key_value, hidden]),
                ("self_attn.v_proj_ctx.weight", vec![key_value, hidden]),
                ("self_attn.o_proj.weight", vec![hidden, query]),
                ("mlp.gate_proj.weight", vec![self.intermediate, hidden]),
                ("mlp.up_proj.weight", vec![self.intermediate, hidden]),
                ("mlp.down_proj.weight", vec![hidden, self.intermediate]),
            ] {
                census.push((format!("{l}.{name}"), shape));
            }
        }
        census
    }
}

struct Located {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// The memory-mapped `mtp.safetensors`.
struct DrafterFile {
    map: Mmap,
    tensors: BTreeMap<String, Located>,
}

/// Bytes the drafter in `path` (its weight file) holds once loaded: every
/// tensor in F32, read from the header alone.
pub fn held_bytes(path: &Path) -> Result<u64> {
    let file = DrafterFile::open(path)?;
    Ok(file
        .tensors
        .values()
        .map(|located| located.shape.iter().product::<usize>() as u64 * 4)
        .sum())
}

impl DrafterFile {
    fn open(path: &Path) -> Result<Self> {
        let at = |err: &dyn std::fmt::Display| load_error(format!("{}: {err}", path.display()));
        let handle = File::open(path).map_err(|err| at(&err))?;
        // SAFETY: the model directory is not modified while it is loaded.
        let map = unsafe { Mmap::map(&handle) }.map_err(|err| at(&err))?;
        let (header_len, metadata) = SafeTensors::read_metadata(&map).map_err(|err| at(&err))?;
        let data_start = 8 + header_len;
        let mut tensors = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            let (start, end) = (data_start + start, data_start + end);
            if end > map.len() || start > end {
                return Err(at(&format!("`{name}` runs past the end of the file")));
            }
            tensors.insert(
                name,
                Located {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start,
                    end,
                },
            );
        }
        Ok(Self { map, tensors })
    }

    /// Holds the file to `census`: every tensor present with its shape in
    /// BF16 (as published) or F32, and nothing else in the file.
    fn check(&self, census: &[(String, Vec<usize>)]) -> Result<()> {
        let expected = census
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(name) = self
            .tensors
            .keys()
            .find(|name| !expected.contains(name.as_str()))
        {
            return Err(load_error(format!("`{name}` is not a drafter tensor")));
        }
        for (name, shape) in census {
            let located = self
                .tensors
                .get(name)
                .ok_or_else(|| load_error(format!("`{name}` is missing")))?;
            if &located.shape != shape {
                return Err(load_error(format!(
                    "`{name}` has shape {:?}, the config asks for {shape:?}",
                    located.shape
                )));
            }
            let width = match located.dtype {
                Dtype::BF16 => 2,
                Dtype::F32 => 4,
                other => return Err(load_error(format!("`{name}` has dtype {other:?}"))),
            };
            if located.end - located.start != shape.iter().product::<usize>() * width {
                return Err(load_error(format!("`{name}` has the wrong byte length")));
            }
        }
        Ok(())
    }

    /// `name` on `device` in F32. Call after [`Self::check`].
    fn tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let located = self
            .tensors
            .get(name)
            .ok_or_else(|| load_error(format!("`{name}` is missing")))?;
        let bytes = &self.map[located.start..located.end];
        let values = match located.dtype {
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| bf16::from_le_bytes(*pair).to_f32())
                .collect::<Vec<_>>(),
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|quad| f32::from_le_bytes(*quad))
                .collect(),
            other => return Err(load_error(format!("`{name}` has dtype {other:?}"))),
        };
        Ok(Tensor::from_vec(values, located.shape.as_slice(), device)?)
    }
}

/// A linear map. Every matrix of the drafter is applied through this, so
/// the dtype the weights are held in is decided here and nowhere else.
struct Linear {
    /// `[in, out]` in F32: the stored `[out, in]` weight, transposed once.
    weight: Tensor,
    /// `[out]` in F32.
    bias: Option<Tensor>,
}

impl Linear {
    /// From a stored `[out, in]` weight.
    fn new(weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            weight: weight.t()?.contiguous()?,
            bias,
        })
    }

    /// `x W^T (+ b)` for F32 `x` of `[rows, in]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // A row cut out of a larger tensor is strided, which the Metal
        // product does not take; this is free when `x` is already dense.
        let out = x.contiguous()?.matmul(&self.weight)?;
        Ok(match &self.bias {
            Some(bias) => out.broadcast_add(bias)?,
            None => out,
        })
    }
}

/// `x * rsqrt(mean(x^2) + eps) * gain` over the last dimension.
fn rms_norm(x: &Tensor, gain: &Tensor, eps: f64) -> Result<Tensor> {
    let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&scale)?.broadcast_mul(gain)?)
}

struct Layer {
    /// Gains, `[hidden]`.
    norm_attention: Tensor,
    norm_mlp: Tensor,
    /// Gains per head channel, `[head_dim]`.
    norm_query: Tensor,
    norm_key: Tensor,
    norm_context_key: Tensor,
    query: Linear,
    key: Linear,
    value: Linear,
    context_key: Linear,
    context_value: Linear,
    output: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// The fused states of some committed positions: for each, one state per
/// draft layer (what that layer's context key and value are made from) and
/// one for the head.
pub struct Fused {
    /// `[positions, layers + 1, hidden]`, the head's row last.
    states: Tensor,
}

impl Fused {
    pub fn positions(&self) -> usize {
        self.states.dims()[0]
    }

    /// The states `layer` builds its context from, `[positions, hidden]`.
    pub fn layer(&self, layer: usize) -> Result<Tensor> {
        let consumers = self.states.dim(1)?;
        if layer + 1 >= consumers {
            return Err(drafter_error(format!("there is no draft layer {layer}")));
        }
        Ok(self.states.narrow(1, layer, 1)?.squeeze(1)?)
    }

    /// The head's state of every position, `[positions, hidden]`.
    pub fn head(&self) -> Result<Tensor> {
        let consumers = self.states.dim(1)?;
        Ok(self.states.narrow(1, consumers - 1, 1)?.squeeze(1)?)
    }

    /// The head's state of the last of these positions, `[hidden]`: the
    /// recurrent head's `f_rnn` when that position is the last committed
    /// one.
    pub fn last_head(&self) -> Result<Tensor> {
        let (positions, consumers, hidden) = self.states.dims3()?;
        Ok(self
            .states
            .narrow(0, positions - 1, 1)?
            .narrow(1, consumers - 1, 1)?
            .reshape(hidden)?)
    }
}

/// Where a block reads the committed positions' keys and values from. The
/// drafter decides which span a layer sees ([`Inkling2DrafterConfig::first_visible`]);
/// the context only has to hold it. [`DenseDrafterContext`] keeps plain
/// tensors; a served model can answer from its own caches instead.
pub trait DrafterContext {
    /// Committed positions held: the position the next anchor sits at.
    fn committed(&self) -> usize;

    /// Keys and values of positions `first..committed()` for `layer`, each
    /// `[committed() - first, kv_heads, head_dim]` in F32 on the drafter's
    /// device, as [`Inkling2Drafter::context_kv`] made them. `None` when
    /// the span is empty.
    fn span(&self, layer: usize, first: usize) -> Result<Option<(Tensor, Tensor)>>;
}

/// A [`DrafterContext`] of plain tensors for one sequence.
///
/// Only committed positions are ever appended, so there is nothing to take
/// back when a draft is rejected: the block's own keys and values live for
/// one [`Inkling2Drafter::draft_block`] call and are not stored here. A
/// local layer keeps the last `window` positions and drops older ones.
pub struct DenseDrafterContext {
    committed: usize,
    /// Per layer: the position of the first row held, keys and values.
    layers: Vec<(usize, Option<(Tensor, Tensor)>)>,
    /// The head's fused state of the last committed position.
    last_head: Option<Tensor>,
    local: Vec<bool>,
    window: usize,
}

impl DenseDrafterContext {
    pub fn new(config: &Inkling2DrafterConfig) -> Self {
        Self {
            committed: 0,
            layers: (0..config.layers).map(|_| (0, None)).collect(),
            last_head: None,
            local: config.local.clone(),
            window: config.window,
        }
    }

    /// Commits the positions whose taps are given (`[positions, taps,
    /// hidden]`, in order, directly after those already held).
    pub fn extend(&mut self, drafter: &Inkling2Drafter, taps: &Tensor) -> Result<()> {
        if taps.dim(0)? == 0 {
            return Ok(());
        }
        let fused = drafter.fuse(taps)?;
        let mut made = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            made.push(drafter.context_kv(layer, &fused.layer(layer)?, self.committed)?);
        }
        let positions = fused.positions();
        for (layer, (keys, values)) in made.into_iter().enumerate() {
            self.append(layer, keys, values)?;
        }
        self.committed += positions;
        self.last_head = Some(fused.last_head()?);
        Ok(())
    }

    fn append(&mut self, layer: usize, keys: Tensor, values: Tensor) -> Result<()> {
        let (first, held) = &mut self.layers[layer];
        let (mut keys, mut values) = match held.take() {
            Some((old_keys, old_values)) => (
                Tensor::cat(&[&old_keys, &keys], 0)?,
                Tensor::cat(&[&old_values, &values], 0)?,
            ),
            None => (keys, values),
        };
        let rows = keys.dim(0)?;
        // No anchor after these positions reads further back than the
        // window in a local layer.
        if self.local[layer] && rows > self.window {
            let drop = rows - self.window;
            keys = keys.narrow(0, drop, self.window)?.contiguous()?;
            values = values.narrow(0, drop, self.window)?.contiguous()?;
            *first += drop;
        }
        *held = Some((keys, values));
        Ok(())
    }

    /// The recurrent head's `f_rnn`: the head's fused state of the last
    /// committed position, `[hidden]`.
    pub fn last_head(&self) -> Result<&Tensor> {
        self.last_head
            .as_ref()
            .ok_or_else(|| drafter_error("no position has been committed"))
    }

    /// Rows held for `layer`.
    pub fn held(&self, layer: usize) -> usize {
        self.committed - self.layers[layer].0
    }
}

impl DrafterContext for DenseDrafterContext {
    fn committed(&self) -> usize {
        self.committed
    }

    fn span(&self, layer: usize, first: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (held_from, held) = self
            .layers
            .get(layer)
            .ok_or_else(|| drafter_error(format!("there is no draft layer {layer}")))?;
        if first >= self.committed {
            return Ok(None);
        }
        let Some((keys, values)) = held else {
            return Ok(None);
        };
        if first < *held_from {
            return Err(drafter_error(format!(
                "layer {layer} was asked for position {first} and holds {held_from} onwards"
            )));
        }
        let (skip, rows) = (first - held_from, self.committed - first);
        Ok(Some((
            keys.narrow(0, skip, rows)?,
            values.narrow(0, skip, rows)?,
        )))
    }
}

/// The drafter on one device.
pub struct Inkling2Drafter {
    config: Inkling2DrafterConfig,
    device: Device,
    /// `softmax(fuse_weight)` over the taps, `[layers + 1, taps]`.
    fusion: Tensor,
    /// The consumers' `dflare_norm` gains, `[layers + 1, hidden]`, the
    /// head's last.
    fusion_gains: Tensor,
    /// `[1, hidden]`.
    mask: Tensor,
    layers: Vec<Layer>,
    /// Gain of the final norm, `[hidden]`.
    norm: Tensor,
    /// `[padded_vocab, rank]`, kept on the device; rows are gathered.
    previous_embedding: Tensor,
    /// `rnn_proj` by the columns of its input: the state, the previous
    /// token's embedding, the row's hidden state, and `f_rnn` (which
    /// carries the bias, being the same for a whole chain).
    recurrent_state: Linear,
    recurrent_previous: Linear,
    recurrent_hidden: Linear,
    recurrent_fused: Linear,
    hidden_out: Linear,
}

impl std::fmt::Debug for Inkling2Drafter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inkling2Drafter")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Inkling2Drafter {
    /// Loads `<dir>/mtp.safetensors` on `device`, by `<dir>/config.json`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let config = Inkling2DrafterConfig::load(dir)?;
        Self::load_with_config(dir, config, device)
    }

    /// [`Self::load`] with a config already read.
    pub fn load_with_config(
        dir: &Path,
        config: Inkling2DrafterConfig,
        device: &Device,
    ) -> Result<Self> {
        let file = DrafterFile::open(&dir.join(MTP_FILE))?;
        file.check(&config.census())?;
        let tensor = |name: String| file.tensor(&name, device);
        let linear = |name: String| Linear::new(&tensor(format!("{name}.weight"))?, None);

        let fusion = softmax(&tensor("fuse_weight".into())?, D::Minus1)?;
        let mut gains = Vec::with_capacity(config.layers + 1);
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let l = format!("layers.{layer}");
            gains.push(tensor(format!("{l}.dflare_norm.weight"))?);
            layers.push(Layer {
                norm_attention: tensor(format!("{l}.input_layernorm.weight"))?,
                norm_mlp: tensor(format!("{l}.post_attention_layernorm.weight"))?,
                norm_query: tensor(format!("{l}.self_attn.q_norm.weight"))?,
                norm_key: tensor(format!("{l}.self_attn.k_norm.weight"))?,
                norm_context_key: tensor(format!("{l}.self_attn.k_norm_ctx.weight"))?,
                query: linear(format!("{l}.self_attn.q_proj"))?,
                key: linear(format!("{l}.self_attn.k_proj"))?,
                value: linear(format!("{l}.self_attn.v_proj"))?,
                context_key: linear(format!("{l}.self_attn.k_proj_ctx"))?,
                context_value: linear(format!("{l}.self_attn.v_proj_ctx"))?,
                output: linear(format!("{l}.self_attn.o_proj"))?,
                gate: linear(format!("{l}.mlp.gate_proj"))?,
                up: linear(format!("{l}.mlp.up_proj"))?,
                down: linear(format!("{l}.mlp.down_proj"))?,
            });
        }
        gains.push(tensor("markov_head.dflare_norm.weight".into())?);

        // The input of `rnn_proj` is [state | previous | hidden | f_rnn].
        let (rank, hidden) = (config.markov_rank, config.hidden);
        let recurrent = tensor("markov_head.rnn_proj.weight".into())?;
        let columns = |first: usize, width: usize, bias: Option<Tensor>| {
            Linear::new(&recurrent.narrow(1, first, width)?, bias)
        };
        Ok(Self {
            fusion,
            fusion_gains: Tensor::stack(&gains, 0)?,
            mask: tensor("mask_embed".into())?.unsqueeze(0)?,
            layers,
            norm: tensor("norm.weight".into())?,
            previous_embedding: tensor("markov_head.markov_w1.weight".into())?,
            recurrent_state: columns(0, rank, None)?,
            recurrent_previous: columns(rank, rank, None)?,
            recurrent_hidden: columns(2 * rank, hidden, None)?,
            recurrent_fused: columns(
                2 * rank + hidden,
                hidden,
                Some(tensor("markov_head.rnn_proj.bias".into())?),
            )?,
            hidden_out: Linear::new(
                &tensor("markov_head.hidden_out.weight".into())?,
                Some(tensor("markov_head.hidden_out.bias".into())?),
            )?,
            device: device.clone(),
            config,
        })
    }

    pub fn config(&self) -> &Inkling2DrafterConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Fuses the taps of some positions, `[positions, taps, hidden]` in
    /// F32, for every consumer.
    pub fn fuse(&self, taps: &Tensor) -> Result<Fused> {
        let c = &self.config;
        let (positions, count, hidden) = taps.dims3()?;
        if positions == 0 || count != c.taps.len() || hidden != c.hidden {
            return Err(drafter_error(format!(
                "taps of shape {:?}, not [positions, {}, {}]",
                taps.dims(),
                c.taps.len(),
                c.hidden
            )));
        }
        // A row of the fusion weights per consumer. As one plain product,
        // [layers + 1, taps] x [taps, positions * hidden]: the Metal
        // backend does not take a broadcast operand.
        let by_tap = taps
            .transpose(0, 1)?
            .contiguous()?
            .reshape((count, positions * hidden))?;
        let mixed = self
            .fusion
            .matmul(&by_tap)?
            .reshape((c.layers + 1, positions, hidden))?
            .transpose(0, 1)?
            .contiguous()?;
        Ok(Fused {
            states: rms_norm(&mixed, &self.fusion_gains, c.rms_norm_eps)?,
        })
    }

    /// Cosines and sines that rotate a head at each of `positions`, each
    /// `[positions, 1, head_dim / 2]`.
    fn rotation(&self, positions: std::ops::Range<usize>) -> Result<(Tensor, Tensor)> {
        let (dim, theta) = (self.config.head_dim, self.config.rope_theta as f32);
        let half = dim / 2;
        // In F32 throughout, as the table the engines read is built.
        let frequencies = (0..half)
            .map(|j| 1.0 / theta.powf((2 * j) as f32 / dim as f32))
            .collect::<Vec<_>>();
        let rows = positions.len();
        let mut cos = Vec::with_capacity(rows * half);
        let mut sin = Vec::with_capacity(rows * half);
        for position in positions {
            for frequency in &frequencies {
                let angle = position as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (rows, 1, half), &self.device)?,
            Tensor::from_vec(sin, (rows, 1, half), &self.device)?,
        ))
    }

    /// Rotates `x` (`[rows, heads, head_dim]`) by the rows' positions: the
    /// first half of a head is paired with its second half.
    fn rotate(&self, x: &Tensor, first_position: usize) -> Result<Tensor> {
        let half = self.config.head_dim / 2;
        let rows = x.dim(0)?;
        let (cos, sin) = self.rotation(first_position..first_position + rows)?;
        let (a, b) = (x.narrow(2, 0, half)?, x.narrow(2, half, half)?);
        Ok(Tensor::cat(
            &[
                (a.broadcast_mul(&cos)? - b.broadcast_mul(&sin)?)?,
                (b.broadcast_mul(&cos)? + a.broadcast_mul(&sin)?)?,
            ],
            2,
        )?)
    }

    /// `x` (`[rows, heads * head_dim]`) as heads, each RMS-normalised with
    /// `gain` and rotated by its row's position.
    fn normed_rotated(
        &self,
        x: &Tensor,
        heads: usize,
        gain: &Tensor,
        first_position: usize,
    ) -> Result<Tensor> {
        let c = &self.config;
        let x = x.reshape((x.dim(0)?, heads, c.head_dim))?;
        self.rotate(&rms_norm(&x, gain, c.rms_norm_eps)?, first_position)
    }

    /// The keys and values `layer` keeps for committed positions
    /// `first_position..`, from that layer's fused states (`[positions,
    /// hidden]`, [`Fused::layer`]). Each is `[positions, kv_heads,
    /// head_dim]`; keys are normalised and rotated by their position,
    /// values are as projected. A position's key and value are written
    /// once and never change.
    pub fn context_kv(
        &self,
        layer: usize,
        fused: &Tensor,
        first_position: usize,
    ) -> Result<(Tensor, Tensor)> {
        let c = &self.config;
        let weights = self
            .layers
            .get(layer)
            .ok_or_else(|| drafter_error(format!("there is no draft layer {layer}")))?;
        let (positions, hidden) = fused.dims2()?;
        if hidden != c.hidden {
            return Err(drafter_error("fused states of the wrong width"));
        }
        let keys = self.normed_rotated(
            &weights.context_key.forward(fused)?,
            c.kv_heads,
            &weights.norm_context_key,
            first_position,
        )?;
        let values = weights
            .context_value
            .forward(fused)?
            .reshape((positions, c.kv_heads, c.head_dim))?;
        Ok((keys, values))
    }

    /// One layer's attention over the context's keys followed by the
    /// block's own. `x` is the normalised layer input, `[block, hidden]`.
    fn attention(
        &self,
        index: usize,
        layer: &Layer,
        x: &Tensor,
        anchor_position: usize,
        context: &dyn DrafterContext,
    ) -> Result<Tensor> {
        let c = &self.config;
        let (rows, kv_heads, dim) = (c.block, c.kv_heads, c.head_dim);
        let group = c.heads / kv_heads;
        let query = self.normed_rotated(
            &layer.query.forward(x)?,
            c.heads,
            &layer.norm_query,
            anchor_position,
        )?;
        let key = self.normed_rotated(
            &layer.key.forward(x)?,
            kv_heads,
            &layer.norm_key,
            anchor_position,
        )?;
        let value = layer.value.forward(x)?.reshape((rows, kv_heads, dim))?;

        let first = c.first_visible(index, anchor_position);
        let (keys, values) = match context.span(index, first)? {
            Some((context_keys, context_values)) => {
                let visible = anchor_position - first;
                if context_keys.dims() != [visible, kv_heads, dim]
                    || context_values.dims() != [visible, kv_heads, dim]
                {
                    return Err(drafter_error(format!(
                        "layer {index}'s context is {:?}, not [{visible}, {kv_heads}, {dim}]",
                        context_keys.dims()
                    )));
                }
                (
                    Tensor::cat(&[&context_keys, &key], 0)?,
                    Tensor::cat(&[&context_values, &value], 0)?,
                )
            }
            None if first == anchor_position => (key, value),
            None => {
                return Err(drafter_error(format!(
                    "layer {index} has no context for positions {first}..{anchor_position}"
                )))
            }
        };

        // A key/value head serves `group` neighbouring query heads, so the
        // queries are grouped under their key/value head:
        // [kv_heads, group * block, head_dim].
        let query = query
            .reshape((rows, kv_heads, group, dim))?
            .permute((1, 2, 0, 3))?
            .reshape((kv_heads, group * rows, dim))?;
        let keys = keys.permute((1, 2, 0))?.contiguous()?;
        let values = values.transpose(0, 1)?.contiguous()?;
        let scores = (query.matmul(&keys)? * (dim as f64).powf(-0.5))?;
        let attended = softmax(&scores, D::Minus1)?
            .matmul(&values)?
            .reshape((kv_heads, group, rows, dim))?
            .permute((2, 0, 1, 3))?
            .reshape((rows, c.heads * dim))?;
        layer.output.forward(&attended)
    }

    /// Runs one block: the anchor (`anchor_embedding`, `[hidden]` in F32,
    /// the main model's token embedding after its embedding norm) at
    /// `anchor_position`, followed by mask rows. `context` holds the keys
    /// and values of every committed position, `0..anchor_position`.
    ///
    /// Returns the final-norm hidden states of all `block` rows, `[block,
    /// hidden]`. Row 0 is the anchor's and predicts nothing; row `j`
    /// predicts the token at position `anchor_position + j`.
    pub fn draft_block(
        &self,
        anchor_embedding: &Tensor,
        anchor_position: usize,
        context: &dyn DrafterContext,
    ) -> Result<Tensor> {
        let c = &self.config;
        if anchor_embedding.dims() != [c.hidden] {
            return Err(drafter_error(format!(
                "an anchor embedding of shape {:?}",
                anchor_embedding.dims()
            )));
        }
        // The context is written for every committed position before a
        // block runs; a gap would silently shift what the block reads.
        if context.committed() != anchor_position {
            return Err(drafter_error(format!(
                "the anchor sits at {anchor_position} and {} positions are committed",
                context.committed()
            )));
        }
        let masks = self.mask.broadcast_as((c.block - 1, c.hidden))?;
        let mut states = Tensor::cat(&[&anchor_embedding.unsqueeze(0)?, &masks], 0)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let x = rms_norm(&states, &layer.norm_attention, c.rms_norm_eps)?;
            states = (states + self.attention(index, layer, &x, anchor_position, context)?)?;
            let x = rms_norm(&states, &layer.norm_mlp, c.rms_norm_eps)?;
            let gated = (layer.gate.forward(&x)?.silu()? * layer.up.forward(&x)?)?;
            states = (states + layer.down.forward(&gated)?)?;
        }
        rms_norm(&states, &self.norm, c.rms_norm_eps)
    }

    /// The mask rows of a block's hidden states: `[block - 1, hidden]`.
    pub fn mask_rows(&self, block_hidden: &Tensor) -> Result<Tensor> {
        let c = &self.config;
        if block_hidden.dims() != [c.block, c.hidden] {
            return Err(drafter_error(format!(
                "block states of shape {:?}",
                block_hidden.dims()
            )));
        }
        Ok(block_hidden.narrow(0, 1, c.block - 1)?)
    }

    fn check_unembed(&self, unembed: &Tensor) -> Result<()> {
        let c = &self.config;
        if unembed.dims() != [c.padded_vocab, c.hidden] || unembed.dtype() != DType::F32 {
            return Err(drafter_error(format!(
                "an unembedding of shape {:?} in {:?}, not [{}, {}] in F32",
                unembed.dims(),
                unembed.dtype(),
                c.padded_vocab,
                c.hidden
            )));
        }
        Ok(())
    }

    /// Base logits of `hidden_rows` (`[rows, hidden]`) over the padded
    /// vocabulary: the rows times the main model's unembedding
    /// (`[padded_vocab, hidden]` in F32). No soft cap: the head adds its
    /// correction first.
    pub fn base_logits(&self, hidden_rows: &Tensor, unembed: &Tensor) -> Result<Tensor> {
        self.check_unembed(unembed)?;
        Ok(hidden_rows.matmul(&unembed.t()?)?)
    }

    /// Starts the recurrent head for one block. `f_rnn` is the head's
    /// fused state of the last committed position ([`Fused::last_head`]),
    /// `anchor_token` the anchor's id.
    pub fn markov_chain(&self, f_rnn: &Tensor, anchor_token: u32) -> Result<MarkovChain<'_>> {
        MarkovChain::new(self, f_rnn, anchor_token)
    }

    /// Drafts a block's tokens by taking the best token at every step.
    /// Returns the `block - 1` tokens and their logits, `[block - 1,
    /// vocab]`.
    pub fn draft_greedy(
        &self,
        block_hidden: &Tensor,
        f_rnn: &Tensor,
        anchor_token: u32,
        unembed: &Tensor,
    ) -> Result<(Vec<u32>, Tensor)> {
        let hidden = self.mask_rows(block_hidden)?;
        let base = self.base_logits(&hidden, unembed)?;
        let mut chain = self.markov_chain(f_rnn, anchor_token)?;
        let mut tokens = Vec::with_capacity(self.config.drafted());
        let mut rows = Vec::with_capacity(self.config.drafted());
        for row in 0..self.config.drafted() {
            let logits = chain.step(row, &hidden.get(row)?, &base.get(row)?, unembed)?;
            let token = logits.argmax(0)?.to_scalar::<u32>()?;
            chain.advance(token)?;
            tokens.push(token);
            rows.push(logits);
        }
        Ok((tokens, Tensor::stack(&rows, 0)?))
    }
}

/// The recurrent head over one block's mask rows, in order: [`Self::step`]
/// gives a row's logits, the caller picks a token from them (greedily or by
/// sampling), and [`Self::advance`] feeds that token to the next step.
pub struct MarkovChain<'a> {
    drafter: &'a Inkling2Drafter,
    /// `rnn_proj`'s bias and its `f_rnn` columns: the same at every step.
    fixed: Tensor,
    /// `[1, rank]`.
    state: Tensor,
    previous: u32,
    /// The mask row the next [`Self::step`] takes.
    next_row: usize,
    /// A step's token has not been given yet.
    awaiting_token: bool,
}

impl<'a> MarkovChain<'a> {
    pub fn new(drafter: &'a Inkling2Drafter, f_rnn: &Tensor, anchor_token: u32) -> Result<Self> {
        let c = &drafter.config;
        if f_rnn.dims() != [c.hidden] {
            return Err(drafter_error(format!(
                "a head state of shape {:?}",
                f_rnn.dims()
            )));
        }
        if anchor_token as usize >= c.padded_vocab {
            return Err(drafter_error(format!("no token {anchor_token}")));
        }
        Ok(Self {
            drafter,
            fixed: drafter.recurrent_fused.forward(&f_rnn.unsqueeze(0)?)?,
            state: Tensor::zeros((1, c.markov_rank), DType::F32, &drafter.device)?,
            previous: anchor_token,
            next_row: 0,
            awaiting_token: false,
        })
    }

    /// Logits of mask row `row` (0-based: block row `row + 1`), `[vocab]`,
    /// from that row's hidden state (`[hidden]`) and base logits
    /// (`[padded_vocab]`). Rows are taken in order, each followed by
    /// [`Self::advance`].
    pub fn step(
        &mut self,
        row: usize,
        hidden_row: &Tensor,
        base_logits_row: &Tensor,
        unembed: &Tensor,
    ) -> Result<Tensor> {
        let d = self.drafter;
        let c = &d.config;
        if self.awaiting_token || row != self.next_row || row >= c.drafted() {
            return Err(drafter_error(format!(
                "row {row} is out of turn (next is {}{})",
                self.next_row,
                if self.awaiting_token {
                    ", after its token"
                } else {
                    ""
                }
            )));
        }
        if hidden_row.dims() != [c.hidden] || base_logits_row.dims() != [c.padded_vocab] {
            return Err(drafter_error("a row of the wrong shape"));
        }
        d.check_unembed(unembed)?;
        let rank = c.markov_rank;
        let previous = d
            .previous_embedding
            .index_select(&Tensor::new(&[self.previous], &d.device)?, 0)?;
        let raw = (((&self.fixed + d.recurrent_state.forward(&self.state)?)?
            + d.recurrent_previous.forward(&previous)?)?
            + d.recurrent_hidden.forward(&hidden_row.unsqueeze(0)?)?)?;
        // (gate, candidate, output). The output is read from the same
        // activation as the update, so from the state before this step.
        let gate = sigmoid(&raw.narrow(1, 0, rank)?)?;
        let candidate = raw.narrow(1, rank, rank)?.tanh()?;
        let latent = raw.narrow(1, 2 * rank, rank)?.tanh()?;
        let keep = (&gate * &self.state)?;
        self.state = (keep + (gate.affine(-1.0, 1.0)? * candidate)?)?;

        let correction = d.hidden_out.forward(&latent)?.matmul(&unembed.t()?)?;
        let logits = (base_logits_row.unsqueeze(0)? + correction)?;
        let logits = match c.softcap {
            Some(cap) => ((logits / cap)?.tanh()? * cap)?,
            None => logits,
        };
        self.awaiting_token = true;
        Ok(logits.narrow(1, 0, c.vocab)?.squeeze(0)?)
    }

    /// [`Self::step`] through the main model's own unembedding: `project`
    /// maps `[1, hidden]` to logits over at least the unpadded vocabulary
    /// (a served model's unembedding, held without its padded rows and in
    /// any dtype). The base logits and the head's correction are one
    /// projection of `hidden_row + hidden_out(latent)`, which is the sum of
    /// the two up to rounding.
    pub fn step_projected(
        &mut self,
        row: usize,
        hidden_row: &Tensor,
        project: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let d = self.drafter;
        let c = &d.config;
        if self.awaiting_token || row != self.next_row || row >= c.drafted() {
            return Err(drafter_error(format!(
                "row {row} is out of turn (next is {})",
                self.next_row
            )));
        }
        if hidden_row.dims() != [c.hidden] {
            return Err(drafter_error("a row of the wrong shape"));
        }
        let rank = c.markov_rank;
        let previous = d
            .previous_embedding
            .index_select(&Tensor::new(&[self.previous], &d.device)?, 0)?;
        let raw = (((&self.fixed + d.recurrent_state.forward(&self.state)?)?
            + d.recurrent_previous.forward(&previous)?)?
            + d.recurrent_hidden.forward(&hidden_row.unsqueeze(0)?)?)?;
        let gate = sigmoid(&raw.narrow(1, 0, rank)?)?;
        let candidate = raw.narrow(1, rank, rank)?.tanh()?;
        let latent = raw.narrow(1, 2 * rank, rank)?.tanh()?;
        let keep = (&gate * &self.state)?;
        self.state = (keep + (gate.affine(-1.0, 1.0)? * candidate)?)?;

        let corrected = (hidden_row.unsqueeze(0)? + d.hidden_out.forward(&latent)?)?;
        let logits = project(&corrected)?;
        if logits.rank() != 2 || logits.dim(0)? != 1 || logits.dim(1)? < c.vocab {
            return Err(drafter_error(format!(
                "the unembedding gave logits of shape {:?}",
                logits.dims()
            )));
        }
        let logits = logits.narrow(1, 0, c.vocab)?.to_dtype(DType::F32)?;
        let logits = match c.softcap {
            Some(cap) => ((logits / cap)?.tanh()? * cap)?,
            None => logits,
        };
        self.awaiting_token = true;
        Ok(logits.squeeze(0)?)
    }

    /// The token chosen from the last [`Self::step`]'s logits.
    pub fn advance(&mut self, token: u32) -> Result<()> {
        if !self.awaiting_token {
            return Err(drafter_error("a token without a step before it"));
        }
        if token as usize >= self.drafter.config.padded_vocab {
            return Err(drafter_error(format!("no token {token}")));
        }
        self.previous = token;
        self.next_row += 1;
        self.awaiting_token = false;
        Ok(())
    }
}

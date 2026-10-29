//! LFM3-Audio's MaskGIT depth decoder (`Lfm3AudioDepthDecoder`): the codes
//! of one speech frame from the language model's hidden state.
//!
//! Each of the 12 codebooks gets a slot: the frame's projected hidden state
//! (`depth_linear`, 2048 -> 12 x 1024) plus its known code's embedding or
//! the learned mask embedding. A 3-layer bidirectional LFM3 trunk (per-head
//! q/k RMSNorm, NEOX RoPE over positions 0..11, sigmoid output gates, no
//! sandwich norms) runs over the slots, and each codebook scores its 2,017
//! codes through its own embedding table (tied head) after an RMSNorm.
//! Codebooks are revealed in a random order, four per pass; a pass scores
//! only the four it reveals, straight from the Q8_0 tables.
//!
//! In the GGUF bundle the tensors are the mmproj's `a.gen.code.*`.

use std::collections::VecDeque;

use std::borrow::Cow;

use candle_core::quantized::{QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};

use crate::error::{Error, Result};
use crate::models::shared::noise::SplitMix64;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::{
    Lfm3AudioSampling, AUDIO_EOS_CODE, CODEBOOKS, CODEBOOK_SIZE, MASKGIT_SCHEDULE,
};
use super::sampling::{penalize_distinct, permutation, sample};

const HIDDEN: usize = 1_024;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 32;
const ROPE_THETA: f64 = 1_000_000.0;
const EPS: f64 = 1e-5;
const PREFIX: &str = "a.gen.code";

struct DepthLayer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    gate: QMatMul,
    q_norm: Tensor,
    k_norm: Tensor,
    /// `[q_norm | k_norm]`, the fused q/k RMSNorm's weights.
    qk_norm: Tensor,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
}

pub(crate) struct Lfm3DepthDecoder {
    proj_in: QMatMul,
    proj_in_bias: Tensor,
    mask_embedding: Tensor,
    /// `[CODEBOOKS * CODEBOOK_SIZE, HIDDEN]`: every codebook's code
    /// embeddings, as stored (Q8_0).
    embeddings: QTensor,
    /// Each codebook's output head: its own embedding table (tied).
    heads: Vec<QMatMul>,
    /// `[CODEBOOKS, HIDDEN]`.
    head_norms: Tensor,
    layers: Vec<DepthLayer>,
    norm: Tensor,
    /// `[HIDDEN]` ones: the heads' unit RMSNorm through the fused kernel.
    ones: Tensor,
    cos: Tensor,
    sin: Tensor,
    /// `[CODEBOOKS, HEAD_DIM]`: each slot's `[cos | sin]`, for the fused
    /// RoPE kernel.
    cos_sin: Tensor,
    offsets: Tensor,
}

#[cfg(test)]
thread_local! {
    /// Test-only switch that sends this thread's depth passes down the
    /// op-by-op path, for parity and A/B timing against the fused kernels.
    pub(crate) static FORCE_UNFUSED_DEPTH: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Whether a depth pass runs the fused RMSNorm, q/k RMSNorm, RoPE and
/// SiLU-gate kernels: on Metal with fused kernels on. CPU and CUDA run the
/// op-by-op pass.
fn depth_fusion_on(device: &Device) -> bool {
    #[cfg(test)]
    if FORCE_UNFUSED_DEPTH.with(std::cell::Cell::get) {
        return false;
    }
    device.is_metal() && crate::kernels::use_fused_kernels_for_device(device)
}

/// The frames a request has generated, for the per-codebook repetition
/// penalty (the last `window` frames). Each codebook's distinct codes are
/// kept as frames come and go, so a head's penalty reads them directly
/// instead of deduplicating up to `window` codes.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameHistory {
    frames: VecDeque<[u32; CODEBOOKS]>,
    /// How often each code is in each codebook's window,
    /// `[CODEBOOKS * CODEBOOK_SIZE]` once a frame arrives.
    counts: Vec<u32>,
    /// Each codebook's distinct codes in the window.
    distinct: [Vec<u32>; CODEBOOKS],
}

impl FrameHistory {
    pub(crate) fn push(&mut self, frame: [u32; CODEBOOKS], window: usize) {
        if window == 0 {
            return;
        }
        if self.counts.is_empty() {
            self.counts = vec![0; CODEBOOKS * CODEBOOK_SIZE];
        }
        while self.frames.len() >= window {
            let Some(old) = self.frames.pop_front() else {
                break;
            };
            for (codebook, &code) in old.iter().enumerate() {
                let Some(count) = self.count_mut(codebook, code) else {
                    continue;
                };
                *count -= 1;
                if *count == 0 {
                    let distinct = &mut self.distinct[codebook];
                    if let Some(index) = distinct.iter().position(|&seen| seen == code) {
                        distinct.swap_remove(index);
                    }
                }
            }
        }
        for (codebook, &code) in frame.iter().enumerate() {
            let Some(count) = self.count_mut(codebook, code) else {
                continue;
            };
            *count += 1;
            if *count == 1 {
                self.distinct[codebook].push(code);
            }
        }
        self.frames.push_back(frame);
    }

    /// `codebook`'s distinct codes in the window, in no particular order.
    fn codebook(&self, codebook: usize) -> impl Iterator<Item = u32> + '_ {
        self.distinct[codebook].iter().copied()
    }

    fn count_mut(&mut self, codebook: usize, code: u32) -> Option<&mut u32> {
        let code = usize::try_from(code)
            .ok()
            .filter(|&code| code < CODEBOOK_SIZE)?;
        self.counts.get_mut(codebook * CODEBOOK_SIZE + code)
    }
}

impl Lfm3DepthDecoder {
    pub(crate) fn load(loader: &GgufLoader, device: &Device) -> Result<Self> {
        let qmatmul = |name: &str| -> Result<QMatMul> {
            Ok(QMatMul::from_qtensor(
                loader.load_qtensor(&format!("{PREFIX}.{name}"), device)?,
            )?)
        };
        let dense = |name: &str| -> Result<Tensor> {
            loader.load_tensor(&format!("{PREFIX}.{name}"), DType::F32, device)
        };
        let layers = (0..3)
            .map(|layer| -> Result<DepthLayer> {
                let p = |name: &str| format!("blk.{layer}.{name}.weight");
                let q_norm = dense(&p("attn_q_norm"))?;
                let k_norm = dense(&p("attn_k_norm"))?;
                Ok(DepthLayer {
                    attn_norm: dense(&p("ln1"))?,
                    ffn_norm: dense(&p("ln2"))?,
                    q: qmatmul(&p("attn_q"))?,
                    k: qmatmul(&p("attn_k"))?,
                    v: qmatmul(&p("attn_v"))?,
                    o: qmatmul(&p("attn_out"))?,
                    gate: qmatmul(&p("attn_gate"))?,
                    qk_norm: Tensor::cat(&[&q_norm, &k_norm], 0)?,
                    q_norm,
                    k_norm,
                    ffn_gate: qmatmul(&p("ffn_gate"))?,
                    ffn_up: qmatmul(&p("ffn_up"))?,
                    ffn_down: qmatmul(&p("ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The tables are codebook-major, so each codebook's rows are one
        // contiguous run of blocks: split them into per-codebook heads
        // without dequantizing.
        let tables = loader.load_qtensor(&format!("{PREFIX}.embd.weight"), &Device::Cpu)?;
        if tables.shape().dims() != [CODEBOOKS, CODEBOOK_SIZE, HIDDEN] {
            return Err(Error::ModelLoadError(format!(
                "LFM3-Audio depth tables are {:?}, expected [{CODEBOOKS}, {CODEBOOK_SIZE}, {HIDDEN}]",
                tables.shape().dims()
            )));
        }
        let dtype = tables.dtype();
        let bytes = tables.data()?;
        let per_codebook = bytes.len() / CODEBOOKS;
        let heads = (0..CODEBOOKS)
            .map(|codebook| -> Result<QMatMul> {
                let rows = &bytes[codebook * per_codebook..(codebook + 1) * per_codebook];
                let storage = QStorage::from_data(Cow::Borrowed(rows), device, dtype)?;
                Ok(QMatMul::from_qtensor(QTensor::new(
                    storage,
                    (CODEBOOK_SIZE, HIDDEN),
                )?)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = QTensor::new(
            QStorage::from_data(Cow::Borrowed(&bytes), device, dtype)?,
            (CODEBOOKS * CODEBOOK_SIZE, HIDDEN),
        )?;
        let (cos, sin) = rope_tables(device)?;
        let cos_sin = Tensor::cat(&[&cos, &sin], 1)?;
        let offsets = Tensor::from_vec(
            (0..CODEBOOKS as u32)
                .map(|codebook| codebook * CODEBOOK_SIZE as u32)
                .collect::<Vec<_>>(),
            (1, CODEBOOKS),
            device,
        )?;
        Ok(Self {
            proj_in: qmatmul("proj_in.weight")?,
            proj_in_bias: dense("proj_in.bias")?,
            mask_embedding: dense("mask_embd")?,
            embeddings,
            heads,
            head_norms: dense("head_norm.weight")?.reshape((CODEBOOKS, HIDDEN))?,
            layers,
            norm: dense("output_norm.weight")?,
            ones: Tensor::ones(HIDDEN, DType::F32, device)?,
            cos,
            sin,
            cos_sin,
            offsets,
        })
    }

    /// The frame inputs `[rows, CODEBOOKS, HIDDEN]` of language-model hidden
    /// states `[rows, 2048]`: computed once per frame, shared by its passes.
    /// Each row projects as its own `[1, 1, 2048]` mat-vec, so its inputs do
    /// not depend on the rows beside it; the rest of the pass is row-local at
    /// any row count (a real-weight test holds each row's frames alone and
    /// four wide).
    pub(crate) fn frame_inputs(&self, hidden: &Tensor) -> Result<Tensor> {
        let rows = hidden.dim(0)?;
        Ok(self
            .proj_in
            .forward(&hidden.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .broadcast_add(&self.proj_in_bias)?
            .reshape((rows, CODEBOOKS, HIDDEN))?)
    }

    /// One pass: the logits `[heads, CODEBOOK_SIZE]` of each `(row,
    /// codebook)` in `heads`, given each row's codes so far and which of
    /// them are revealed.
    pub(crate) fn pass_logits(
        &self,
        frame_inputs: &Tensor,
        codes: &[[u32; CODEBOOKS]],
        revealed: &[[bool; CODEBOOKS]],
        heads: &[(usize, usize)],
    ) -> Result<Tensor> {
        let device = frame_inputs.device();
        let rows = codes.len();
        let ids = Tensor::from_vec(
            codes.iter().flatten().copied().collect::<Vec<_>>(),
            (rows, CODEBOOKS),
            device,
        )?
        .broadcast_add(&self.offsets)?
        .flatten_all()?;
        let embeds = self
            .embeddings
            .embedding(&ids)?
            .reshape((rows, CODEBOOKS, HIDDEN))?;
        let known = Tensor::from_vec(
            revealed
                .iter()
                .flatten()
                .map(|&known| if known { 1f32 } else { 0f32 })
                .collect::<Vec<_>>(),
            (rows, CODEBOOKS, 1),
            device,
        )?;
        let unknown = known.affine(-1.0, 1.0)?;
        let mut x = frame_inputs.add(&embeds.broadcast_mul(&known)?)?;
        x = x.add(&unknown.broadcast_mul(&self.mask_embedding)?)?;
        let fused = depth_fusion_on(device);
        let norm = |x: &Tensor, weight: &Tensor| -> Result<Tensor> {
            match fused
                .then(|| crate::kernels::try_fused_rms_norm(x, weight, EPS))
                .flatten()
            {
                Some(normed) => Ok(normed),
                None => rms(x, weight),
            }
        };
        for layer in &self.layers {
            let h = norm(&x, &layer.attn_norm)?;
            let attended = match fused
                .then(|| self.fused_attention(layer, &h))
                .transpose()?
                .flatten()
            {
                Some(attended) => attended,
                None => self.attention(layer, &h)?,
            };
            x = x.add(&attended)?;
            let h = norm(&x, &layer.ffn_norm)?;
            let (gate, up) = (layer.ffn_gate.forward(&h)?, layer.ffn_up.forward(&h)?);
            let gated = match fused
                .then(|| crate::kernels::try_fused_silu_mul(&gate, &up))
                .flatten()
            {
                Some(gated) => gated,
                None => candle_nn::ops::silu(&gate)?.mul(&up)?,
            };
            x = x.add(&layer.ffn_down.forward(&gated)?)?;
        }
        // Each codebook's head: an RMSNorm then its tied embedding table.
        let normed = norm(&x, &self.norm)?;
        let normed = match fused
            .then(|| crate::kernels::try_fused_rms_norm(&normed, &self.ones, EPS))
            .flatten()
        {
            Some(unit) => unit,
            None => unit_rms(&normed)?,
        }
        .broadcast_mul(&self.head_norms)?
        .reshape((rows * CODEBOOKS, HIDDEN))?;
        let slots = Tensor::from_vec(
            heads
                .iter()
                .map(|&(row, codebook)| (row * CODEBOOKS + codebook) as u32)
                .collect::<Vec<_>>(),
            heads.len(),
            device,
        )?;
        let normed = normed.index_select(&slots, 0)?;
        let logits = heads
            .iter()
            .enumerate()
            .map(|(index, &(_, codebook))| {
                self.heads[codebook].forward(&normed.i(index..index + 1)?)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Tensor::cat(&logits, 0)?)
    }

    fn attention(&self, layer: &DepthLayer, h: &Tensor) -> Result<Tensor> {
        let (rows, slots, _) = h.dims3()?;
        let heads = |m: &QMatMul, n: usize, norm: &Tensor| -> Result<Tensor> {
            let x = m.forward(h)?.reshape((rows, slots, n, HEAD_DIM))?;
            let x = rms(&x, norm)?.transpose(1, 2)?.contiguous()?;
            Ok(candle_nn::rotary_emb::rope(&x, &self.cos, &self.sin)?)
        };
        let q = heads(&layer.q, HEADS, &layer.q_norm)?;
        let k = heads(&layer.k, KV_HEADS, &layer.k_norm)?;
        let v = layer
            .v
            .forward(h)?
            .reshape((rows, slots, KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let repeat = |x: Tensor| -> Result<Tensor> {
            Ok(x.unsqueeze(2)?
                .broadcast_as((rows, KV_HEADS, HEADS / KV_HEADS, slots, HEAD_DIM))?
                .reshape((rows, HEADS, slots, HEAD_DIM))?)
        };
        let (k, v) = (repeat(k)?, repeat(v)?);
        // Bidirectional: every slot sees every slot of its frame.
        let scores = q
            .matmul(&k.t()?)?
            .affine(1.0 / (HEAD_DIM as f64).sqrt(), 0.0)?;
        let attended = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let gate =
            candle_nn::ops::sigmoid(&layer.gate.forward(h)?)?.reshape((rows, slots, HEADS, 1))?;
        let attended = attended.transpose(1, 2)?.broadcast_mul(&gate)?.reshape((
            rows,
            slots,
            HEADS * HEAD_DIM,
        ))?;
        Ok(layer.o.forward(&attended)?)
    }

    /// [`Self::attention`] through the fused q/k RMSNorm and RoPE kernels,
    /// each query head grouped with its key/value head instead of the keys
    /// and values copied per query head; `None` when a kernel declines.
    fn fused_attention(&self, layer: &DepthLayer, h: &Tensor) -> Result<Option<Tensor>> {
        let (rows, slots, _) = h.dims3()?;
        // The q/k kernel normalizes one position's heads per row.
        let q = layer
            .q
            .forward(h)?
            .reshape((rows * slots, 1, HEADS, HEAD_DIM))?;
        let k = layer
            .k
            .forward(h)?
            .reshape((rows * slots, 1, KV_HEADS, HEAD_DIM))?;
        let Some((q, k)) = crate::kernels::try_fused_qk_rms_norm(&q, &k, &layer.qk_norm, EPS)
        else {
            return Ok(None);
        };
        let Some((q, k)) = crate::kernels::try_fused_rope_pair_bshd(
            &q.reshape((rows, slots, HEADS, HEAD_DIM))?,
            &k.reshape((rows, slots, KV_HEADS, HEAD_DIM))?,
            &self.cos_sin,
        ) else {
            return Ok(None);
        };
        // Query head `h` reads key/value head `h / group`, so the queries
        // of one key/value head are one contiguous run of `group * slots`.
        let group = HEADS / KV_HEADS;
        let q =
            q.transpose(1, 2)?
                .contiguous()?
                .reshape((rows, KV_HEADS, group * slots, HEAD_DIM))?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = layer
            .v
            .forward(h)?
            .reshape((rows, slots, KV_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = q
            .matmul(&k.t()?)?
            .affine(1.0 / (HEAD_DIM as f64).sqrt(), 0.0)?;
        let attended = candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v)?
            .reshape((rows, HEADS, slots, HEAD_DIM))?;
        let gate =
            candle_nn::ops::sigmoid(&layer.gate.forward(h)?)?.reshape((rows, slots, HEADS, 1))?;
        let attended = attended.transpose(1, 2)?.broadcast_mul(&gate)?.reshape((
            rows,
            slots,
            HEADS * HEAD_DIM,
        ))?;
        Ok(Some(layer.o.forward(&attended)?))
    }

    /// Sample one frame per row: a random reveal order, then four codebooks
    /// per pass, each drawn from its logits after the repetition penalty
    /// over that codebook's history. A row that draws the end-of-audio code
    /// in any codebook ends speech: its frame is all end-of-audio codes.
    pub(crate) fn sample_frames(
        &self,
        hidden: &Tensor,
        samplings: &[&Lfm3AudioSampling],
        histories: &[&FrameHistory],
        rngs: &mut [&mut SplitMix64],
    ) -> Result<Vec<[u32; CODEBOOKS]>> {
        let rows = hidden.dim(0)?;
        if samplings.len() != rows || histories.len() != rows || rows == 0 || rngs.len() != rows {
            return Err(Error::InvalidInput(
                "LFM3-Audio depth rows do not match their histories".into(),
            ));
        }
        let inputs = self.frame_inputs(hidden)?;
        let orders = rngs
            .iter_mut()
            .map(|rng| permutation(CODEBOOKS, rng))
            .collect::<Vec<_>>();
        let mut codes = vec![[0u32; CODEBOOKS]; rows];
        let mut revealed = vec![[false; CODEBOOKS]; rows];
        let mut start = 0;
        for count in MASKGIT_SCHEDULE {
            let heads = (0..rows)
                .flat_map(|row| {
                    orders[row][start..start + count]
                        .iter()
                        .map(move |&codebook| (row, codebook))
                })
                .collect::<Vec<_>>();
            let mut logits = self
                .pass_logits(&inputs, &codes, &revealed, &heads)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?
                .into_iter();
            for row in 0..rows {
                for &codebook in &orders[row][start..start + count] {
                    let mut scores = logits.next().expect("one row of logits per head");
                    penalize_distinct(
                        &mut scores,
                        histories[row].codebook(codebook),
                        samplings[row].repetition_penalty,
                    );
                    codes[row][codebook] = sample(
                        &mut scores,
                        samplings[row].temperature,
                        samplings[row].top_k,
                        1.0,
                        rngs[row],
                    );
                    revealed[row][codebook] = true;
                }
            }
            start += count;
        }
        for frame in &mut codes {
            if frame.contains(&AUDIO_EOS_CODE) {
                *frame = [AUDIO_EOS_CODE; CODEBOOKS];
            }
        }
        Ok(codes)
    }
}

/// `x * rsqrt(mean(x^2) + eps) * weight` over the last dimension, in F32.
fn rms(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    Ok(unit_rms(x)?.broadcast_mul(weight)?)
}

fn unit_rms(x: &Tensor) -> Result<Tensor> {
    let mean = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&mean.affine(1.0, EPS)?.sqrt()?)?)
}

/// NEOX RoPE tables for the 12 slot positions, built on the host in F64.
fn rope_tables(device: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let mut cos = Vec::with_capacity(CODEBOOKS * half);
    let mut sin = Vec::with_capacity(CODEBOOKS * half);
    for position in 0..CODEBOOKS {
        for pair in 0..half {
            let inv_freq = 1.0 / ROPE_THETA.powf((2 * pair) as f64 / HEAD_DIM as f64);
            let (s, c) = (position as f64 * inv_freq).sin_cos();
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (CODEBOOKS, half), device)?,
        Tensor::from_vec(sin, (CODEBOOKS, half), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each codebook's distinct codes are those of the last `window` frames,
    /// as the frames come and go.
    #[test]
    fn frame_history_keeps_each_codebooks_distinct_window_codes() {
        let mut rng = SplitMix64::new(9);
        let mut history = FrameHistory::default();
        let mut frames = Vec::new();
        for step in 0..400 {
            let window = if step < 200 { 16 } else { 7 };
            let frame = std::array::from_fn(|codebook| {
                // Few codes, so they repeat; the end-of-audio code too.
                (rng.next_u64() % (3 + codebook as u64)) as u32 * 1_000 % 2_017
            });
            history.push(frame, window);
            frames.push(frame);
            let start = frames.len().saturating_sub(window);
            for codebook in 0..CODEBOOKS {
                let mut expected = frames[start..]
                    .iter()
                    .map(|frame| frame[codebook])
                    .collect::<Vec<_>>();
                expected.sort_unstable();
                expected.dedup();
                let mut kept = history.codebook(codebook).collect::<Vec<_>>();
                kept.sort_unstable();
                assert_eq!(kept, expected, "step {step} codebook {codebook}");
            }
        }
        let mut empty = FrameHistory::default();
        empty.push([1; CODEBOOKS], 0);
        assert_eq!(empty.codebook(0).count(), 0);
    }
}

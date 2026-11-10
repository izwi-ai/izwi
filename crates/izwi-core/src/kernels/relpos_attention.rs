//! Relative-position self-attention in one pass, for the conformer ASR
//! encoders: Transformer-XL scores with `u`/`v` biases over a projected
//! position table (Parakeet, Nemotron Streaming) and Shaw block-local scores
//! over a position embedding table (Granite Speech 4.1 and 5.0).
//!
//! The eager encoders build whole score planes. Parakeet and Nemotron
//! multiply `q + v` by all `Tq + Tk - 1` projected positions, then pad,
//! reshape and narrow the product (`rel_shift`), which keeps three
//! `[B, H, T, 2T]` planes alive; Granite gathers a `[C, C, Dh]` slice of its
//! table in every layer. Here query `i` scores only its visible keys and
//! reads the one position row each key needs: `rel_shift(x)[i][j]` is
//! `x[i][Tq - 1 - i + j]` (the row of distance `i - j`), and Granite's
//! `[C, C]` gather is `table[clamp(i - j, -C, C) + center]`.
//!
//! Every visibility rule the encoders use is a contiguous key interval per
//! query ([`KeyBand`] with the rows' key lengths), so a masked key is
//! skipped rather than given a large negative score. Eager gives such a key
//! exactly zero weight (`exp(-1e9 - max)` underflows), so skipping it is the
//! same for every query that sees at least one key, and a non-finite value
//! in a padded key can no longer reach a real query through `0 * NaN`. A
//! query that sees no key (only padded queries can) gets a zero output.
//!
//! [`reference`] is the CUDA kernel's arithmetic on the host, operation for
//! operation: 32 lanes each own `Dh / 32` contiguous dims, a dot product is
//! each lane's fused multiply-add chain followed by an xor butterfly, the
//! softmax is exact two-pass (max, [`det_exp`], lane-strided sums, a
//! division), and the value product is each lane's multiply-add chain over
//! the keys in order. Eager's roundings are kept where they cost one convert:
//! with a BF16 `core` the scores round to BF16 after each eager op and the
//! probabilities before the value product, as the BF16 encoders do. The dot
//! products and sums run in another order than cuBLAS and Candle's
//! reductions, so the result matches eager ([`composed`]) within a tolerance
//! ([`tolerance`]), never bit for bit.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Result, Tensor};
use half::bf16;

use crate::backends::BackendKind;

/// Lanes of one warp: each owns `Dh / LANES` dims of a dot product.
const LANES: usize = 32;
/// The most dims a lane holds (`Dh` up to 256).
pub const MAX_LANE_DIMS: usize = 8;

/// How relative positions enter a score.
#[derive(Debug, Clone, Copy)]
pub enum RelPosition<'a> {
    /// Transformer-XL: `((q + u)·k_j + (q + v)·pos[Tq - 1 - i + j]) · scale`.
    /// `pos` is `[Tq + Tk - 1, H, Dh]`, the projected embeddings of the
    /// distances from `Tq - 1 + offset` down to `offset - Tk + 1`; the biases
    /// are `[H, Dh]`.
    TransformerXl {
        bias_u: &'a Tensor,
        bias_v: &'a Tensor,
    },
    /// Shaw: `(q·k_j) · scale + (q·pos[clamp(i - j, -clamp, clamp) + center])
    /// · scale`, with the position product in F32 as Granite computes it.
    /// `pos` is the `[R, Dh]` embedding table itself.
    Shaw { clamp: usize, center: usize },
}

/// The keys a query sees, before the row's key length cuts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyBand {
    /// Every key.
    Full,
    /// NeMo's `chunked_limited` context: positions form chunks of `chunk`
    /// frames, and a query sees its own chunk and the `left_chunks` before
    /// it. Query `i` sits at position `query_origin + i` and key `j` at
    /// `key_origin + j` (a streaming cache puts keys before the chunk).
    Chunked {
        chunk: usize,
        left_chunks: usize,
        query_origin: i64,
        key_origin: i64,
    },
    /// Block-local attention: a query sees the keys of its own `size`-frame
    /// block.
    Block { size: usize },
}

/// How many leading keys of each row are real.
#[derive(Debug, Clone, Copy)]
pub enum KeyLengths<'a> {
    /// All `Tk`.
    All,
    /// The same count for every row.
    Uniform(usize),
    /// `[B]` U32 on the inputs' device.
    PerRow(&'a Tensor),
}

/// One relative-position attention call.
#[derive(Debug, Clone, Copy)]
pub struct RelPosAttention<'a> {
    /// `[B, Tq, H, Dh]`, any strides with contiguous dims.
    pub q: &'a Tensor,
    /// `[B, Tk, H, Dh]`.
    pub k: &'a Tensor,
    /// `[B, Tk, H, Dh]`.
    pub v: &'a Tensor,
    /// See [`RelPosition`].
    pub pos: &'a Tensor,
    pub position: RelPosition<'a>,
    pub band: KeyBand,
    pub key_lengths: KeyLengths<'a>,
    pub scale: f64,
    /// The dtype eager rounds its scores and probabilities to: F32 or BF16.
    pub core: DType,
    /// The output's dtype: F32 or BF16.
    pub out_dtype: DType,
}

/// The validated geometry of a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelPosDims {
    pub batch: usize,
    pub q_len: usize,
    pub k_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pos_rows: usize,
}

impl RelPosAttention<'_> {
    /// The call's geometry, or an error for inputs the op does not take.
    pub fn dims(&self) -> Result<RelPosDims> {
        let (batch, q_len, heads, head_dim) = self.q.dims4()?;
        let (k_batch, k_len, k_heads, k_dim) = self.k.dims4()?;
        if (k_batch, k_heads, k_dim) != (batch, heads, head_dim) || self.v.dims() != self.k.dims() {
            candle_core::bail!(
                "relative-position attention takes q {:?} with k and v {:?} / {:?} of the same rows, heads and head dim",
                self.q.dims(),
                self.k.dims(),
                self.v.dims()
            )
        }
        if batch == 0 || q_len == 0 || k_len == 0 || heads == 0 {
            candle_core::bail!("relative-position attention needs non-empty inputs")
        }
        if head_dim == 0 || head_dim % LANES != 0 || head_dim / LANES > MAX_LANE_DIMS {
            candle_core::bail!(
                "relative-position attention takes a head dim that is a multiple of {LANES} up to {}, got {head_dim}",
                LANES * MAX_LANE_DIMS
            )
        }
        for dtype in [self.core, self.out_dtype] {
            if !matches!(dtype, DType::F32 | DType::BF16) {
                candle_core::bail!(
                    "relative-position attention rounds to F32 or BF16, got {dtype:?}"
                )
            }
        }
        let pos_rows = match self.position {
            RelPosition::TransformerXl { bias_u, bias_v } => {
                let (rows, pos_heads, pos_dim) = self.pos.dims3()?;
                if rows != q_len + k_len - 1 || (pos_heads, pos_dim) != (heads, head_dim) {
                    candle_core::bail!(
                        "Transformer-XL positions must be [{}, {heads}, {head_dim}], got {:?}",
                        q_len + k_len - 1,
                        self.pos.dims()
                    )
                }
                for bias in [bias_u, bias_v] {
                    if bias.dims() != [heads, head_dim] {
                        candle_core::bail!(
                            "Transformer-XL biases must be [{heads}, {head_dim}], got {:?}",
                            bias.dims()
                        )
                    }
                }
                rows
            }
            RelPosition::Shaw { clamp, center } => {
                let (rows, pos_dim) = self.pos.dims2()?;
                if pos_dim != head_dim || center < clamp || center + clamp >= rows {
                    candle_core::bail!(
                        "Shaw table {:?} cannot serve distances ±{clamp} around row {center}",
                        self.pos.dims()
                    )
                }
                rows
            }
        };
        match self.band {
            KeyBand::Chunked { chunk: 0, .. } | KeyBand::Block { size: 0 } => {
                candle_core::bail!("relative-position attention needs a non-empty chunk or block")
            }
            _ => {}
        }
        if let KeyLengths::PerRow(lengths) = self.key_lengths {
            if lengths.dims() != [batch] || lengths.dtype() != DType::U32 {
                candle_core::bail!(
                    "per-row key lengths must be [{batch}] U32, got {:?} {:?}",
                    lengths.dims(),
                    lengths.dtype()
                )
            }
        }
        Ok(RelPosDims {
            batch,
            q_len,
            k_len,
            heads,
            head_dim,
            pos_rows,
        })
    }
}

/// The most keys any query of a call sees: the scores one query keeps.
pub fn max_span(band: KeyBand, k_len: usize) -> usize {
    match band {
        KeyBand::Full => k_len,
        KeyBand::Chunked {
            chunk, left_chunks, ..
        } => k_len.min(chunk.saturating_mul(left_chunks.saturating_add(1))),
        KeyBand::Block { size } => k_len.min(size),
    }
}

/// The keys `[lo, hi)` query `i` sees in a row whose first `key_len` keys
/// are real (`hi <= lo`: none). The CUDA kernel computes the same interval.
pub fn visible_keys(band: KeyBand, i: usize, key_len: usize) -> (usize, usize) {
    let clip = |value: i64| value.clamp(0, key_len as i64) as usize;
    match band {
        KeyBand::Full => (0, key_len),
        KeyBand::Chunked {
            chunk,
            left_chunks,
            query_origin,
            key_origin,
        } => {
            let chunk = chunk as i64;
            let query_chunk = (query_origin + i as i64).div_euclid(chunk);
            let lo = chunk * (query_chunk - left_chunks as i64) - key_origin;
            let hi = chunk * (query_chunk + 1) - key_origin;
            (clip(lo), clip(hi))
        }
        KeyBand::Block { size } => {
            let start = i / size * size;
            (start.min(key_len), (start + size).min(key_len))
        }
    }
}

fn shaw_row(i: usize, j: usize, clamp: usize, center: usize) -> usize {
    let distance = (i as i64 - j as i64).clamp(-(clamp as i64), clamp as i64);
    (center as i64 + distance) as usize
}

/// `scale` as eager multiplies by it in `core`: Candle's affine takes its
/// factor in the tensor's dtype.
fn core_scale(scale: f64, core: DType) -> f32 {
    if core == DType::BF16 {
        bf16::from_f64(scale).to_f32()
    } else {
        scale as f32
    }
}

/// `e^x` from F32 multiplies, fused multiply-adds and one `rint`, so the
/// host and the CUDA kernel (`kernels/cuda/relpos_attention.cu`) agree bit
/// for bit, unlike the platforms' `expf`. Cody-Waite range reduction to
/// `|r| <= ln2/2` and Cephes' `expf` polynomial: within 2 units in the last
/// place of `e^x` for normal results. `x <= -104` gives zero; NaN stays NaN.
/// The softmax only calls it with `x <= 0`; it is exact for `x` up to 88.
pub fn det_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    if x.is_nan() {
        return x;
    }
    if x <= -104.0 {
        return 0.0;
    }
    let n = (x * LOG2E).round_ties_even();
    let r = (-n).mul_add(LN2_HI, x);
    let r = (-n).mul_add(LN2_LO, r);
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_6e-1);
    p = p.mul_add(r, 0.5);
    let y = p.mul_add(r * r, r) + 1.0;
    scale_by_power_of_two(y, n as i32)
}

/// `y · 2^n` for `y` in `[0.5, 2)` and `n` in `[-151, 127]`: one exact or
/// one rounding multiply (two below the normal range, the second rounding).
fn scale_by_power_of_two(y: f32, n: i32) -> f32 {
    let power = |exponent: i32| f32::from_bits(((exponent + 127) as u32) << 23);
    if n >= -126 {
        y * power(n)
    } else {
        y * power(n + 64) * power(-64)
    }
}

/// The xor butterfly over 32 lanes (offsets 16, 8, 4, 2, 1), each lane
/// combining its value with its partner's as `combine(own, partner)`.
fn butterfly(lanes: &mut [f32; LANES], combine: impl Fn(f32, f32) -> f32) {
    for offset in [16, 8, 4, 2, 1] {
        let previous = *lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = combine(previous[lane], previous[lane ^ offset]);
        }
    }
}

/// A warp's dot product: lane `l` multiplies dims `l·n .. (l+1)·n` in
/// order (the first product rounded on its own, the rest fused), then the
/// butterfly sums the lanes. Every lane ends with the same bits.
fn warp_dot(a: &[f32], b: &[f32]) -> f32 {
    let per_lane = a.len() / LANES;
    let mut lanes = [0f32; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let dims = lane * per_lane..(lane + 1) * per_lane;
        let (a, b) = (&a[dims.clone()], &b[dims]);
        let mut part = a[0] * b[0];
        for d in 1..per_lane {
            part = a[d].mul_add(b[d], part);
        }
        *slot = part;
    }
    butterfly(&mut lanes, |own, partner| own + partner);
    lanes[0]
}

/// The kernel's softmax over one query's scores, in place: lane `t % 32`
/// takes the scores `t`, the lanes' maxima and sums meet in butterflies, and
/// each probability rounds through `round`.
fn softmax_in_place(scores: &mut [f32], round: impl Fn(f32) -> f32) {
    let mut max = [f32::NEG_INFINITY; LANES];
    for (t, score) in scores.iter().enumerate() {
        if *score > max[t % LANES] {
            max[t % LANES] = *score;
        }
    }
    butterfly(
        &mut max,
        |own, partner| if partner > own { partner } else { own },
    );
    let mut sum = [0f32; LANES];
    for (t, score) in scores.iter_mut().enumerate() {
        let lane = t % LANES;
        *score = det_exp(*score - max[lane]);
        sum[lane] += *score;
    }
    butterfly(&mut sum, |own, partner| own + partner);
    for score in scores.iter_mut() {
        *score = round(*score / sum[0]);
    }
}

/// A call's inputs on the host, as F32 in their logical layouts.
struct HostInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    pos: Vec<f32>,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    /// Each row's real keys, at most `Tk`.
    key_lens: Vec<usize>,
}

fn host_values(tensor: &Tensor) -> Result<Vec<f32>> {
    crate::kernels::numeric_check::host_f32(tensor)
}

impl HostInputs {
    fn read(spec: &RelPosAttention<'_>, dims: &RelPosDims) -> Result<Self> {
        let (bias_u, bias_v) = match spec.position {
            RelPosition::TransformerXl { bias_u, bias_v } => {
                (host_values(bias_u)?, host_values(bias_v)?)
            }
            RelPosition::Shaw { .. } => (Vec::new(), Vec::new()),
        };
        Ok(Self {
            q: host_values(spec.q)?,
            k: host_values(spec.k)?,
            v: host_values(spec.v)?,
            pos: host_values(spec.pos)?,
            bias_u,
            bias_v,
            key_lens: key_lens(spec, dims)?,
        })
    }

    fn q_row(&self, dims: &RelPosDims, b: usize, i: usize, h: usize) -> &[f32] {
        let start = ((b * dims.q_len + i) * dims.heads + h) * dims.head_dim;
        &self.q[start..start + dims.head_dim]
    }

    fn key_row<'s>(
        values: &'s [f32],
        dims: &RelPosDims,
        b: usize,
        j: usize,
        h: usize,
    ) -> &'s [f32] {
        let start = ((b * dims.k_len + j) * dims.heads + h) * dims.head_dim;
        &values[start..start + dims.head_dim]
    }

    fn pos_row(
        &self,
        spec: &RelPosAttention<'_>,
        dims: &RelPosDims,
        i: usize,
        j: usize,
        h: usize,
    ) -> &[f32] {
        let start = match spec.position {
            RelPosition::TransformerXl { .. } => {
                ((dims.q_len - 1 - i + j) * dims.heads + h) * dims.head_dim
            }
            RelPosition::Shaw { clamp, center } => shaw_row(i, j, clamp, center) * dims.head_dim,
        };
        &self.pos[start..start + dims.head_dim]
    }
}

fn key_lens(spec: &RelPosAttention<'_>, dims: &RelPosDims) -> Result<Vec<usize>> {
    Ok(match spec.key_lengths {
        KeyLengths::All => vec![dims.k_len; dims.batch],
        KeyLengths::Uniform(len) => vec![len.min(dims.k_len); dims.batch],
        KeyLengths::PerRow(lengths) => lengths
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|len| (len as usize).min(dims.k_len))
            .collect(),
    })
}

/// The CUDA kernel's result on the host, bit for bit: `[B, Tq, H·Dh]` in
/// the call's output dtype, on the CPU.
pub fn reference(spec: &RelPosAttention<'_>) -> Result<Tensor> {
    let dims = spec.dims()?;
    let inputs = HostInputs::read(spec, &dims)?;
    let RelPosDims {
        batch,
        q_len,
        heads,
        head_dim,
        ..
    } = dims;
    let round = spec.core == DType::BF16;
    let rb = |x: f32| if round { bf16::from_f32(x).to_f32() } else { x };
    let scale = core_scale(spec.scale, spec.core);
    let position_scale = spec.scale as f32;
    let mut out = vec![0f32; batch * q_len * heads * head_dim];
    let (mut qa, mut qb) = (vec![0f32; head_dim], vec![0f32; head_dim]);
    let mut scores = Vec::with_capacity(max_span(spec.band, dims.k_len));
    for b in 0..batch {
        for i in 0..q_len {
            let (lo, hi) = visible_keys(spec.band, i, inputs.key_lens[b]);
            if hi <= lo {
                continue;
            }
            for h in 0..heads {
                let q = inputs.q_row(&dims, b, i, h);
                match spec.position {
                    RelPosition::TransformerXl { .. } => {
                        let bias = h * head_dim..(h + 1) * head_dim;
                        for (d, value) in q.iter().enumerate() {
                            qa[d] = rb(value + inputs.bias_u[bias.start + d]);
                            qb[d] = rb(value + inputs.bias_v[bias.start + d]);
                        }
                    }
                    RelPosition::Shaw { .. } => {
                        qa.copy_from_slice(q);
                        qb.copy_from_slice(q);
                    }
                }
                scores.clear();
                for j in lo..hi {
                    let content = warp_dot(&qa, HostInputs::key_row(&inputs.k, &dims, b, j, h));
                    let position = warp_dot(&qb, inputs.pos_row(spec, &dims, i, j, h));
                    scores.push(match spec.position {
                        RelPosition::TransformerXl { .. } => {
                            rb(rb(rb(content) + rb(position)) * scale)
                        }
                        RelPosition::Shaw { .. } => {
                            rb(rb(rb(content) * scale) + rb(position * position_scale))
                        }
                    });
                }
                softmax_in_place(&mut scores, rb);
                let start = ((b * q_len + i) * heads + h) * head_dim;
                for (d, slot) in out[start..start + head_dim].iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for (t, probability) in scores.iter().enumerate() {
                        let v = HostInputs::key_row(&inputs.v, &dims, b, lo + t, h);
                        acc = probability.mul_add(v[d], acc);
                    }
                    *slot = acc;
                }
            }
        }
    }
    Tensor::from_vec(out, (batch, q_len, heads * head_dim), &Device::Cpu)?.to_dtype(spec.out_dtype)
}

/// The mathematical op in F64 over the call's input values (no
/// intermediate rounding): `[B, Tq, H, Dh]` flattened, and whether each
/// `(b, i, h)` query sees a key. A query that sees none is zero.
pub fn exact_reference(spec: &RelPosAttention<'_>) -> Result<(Vec<f64>, Vec<bool>)> {
    let dims = spec.dims()?;
    let inputs = HostInputs::read(spec, &dims)?;
    let RelPosDims {
        batch,
        q_len,
        heads,
        head_dim,
        ..
    } = dims;
    let dot = |a: &[f64], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * f64::from(*b)).sum::<f64>();
    let mut out = vec![0f64; batch * q_len * heads * head_dim];
    let mut visible = vec![false; batch * q_len * heads];
    for b in 0..batch {
        for i in 0..q_len {
            let (lo, hi) = visible_keys(spec.band, i, inputs.key_lens[b]);
            if hi <= lo {
                continue;
            }
            for h in 0..heads {
                visible[(b * q_len + i) * heads + h] = true;
                let q = inputs.q_row(&dims, b, i, h);
                let (qa, qb): (Vec<f64>, Vec<f64>) = match spec.position {
                    RelPosition::TransformerXl { .. } => q
                        .iter()
                        .enumerate()
                        .map(|(d, value)| {
                            let value = f64::from(*value);
                            (
                                value + f64::from(inputs.bias_u[h * head_dim + d]),
                                value + f64::from(inputs.bias_v[h * head_dim + d]),
                            )
                        })
                        .unzip(),
                    RelPosition::Shaw { .. } => {
                        let q = q.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
                        (q.clone(), q)
                    }
                };
                let scores = (lo..hi)
                    .map(|j| {
                        let content = dot(&qa, HostInputs::key_row(&inputs.k, &dims, b, j, h));
                        let position = dot(&qb, inputs.pos_row(spec, &dims, i, j, h));
                        (content + position) * spec.scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let total = weights.iter().sum::<f64>();
                let start = ((b * q_len + i) * heads + h) * head_dim;
                for (t, weight) in weights.iter().enumerate() {
                    let v = HostInputs::key_row(&inputs.v, &dims, b, lo + t, h);
                    for d in 0..head_dim {
                        out[start + d] += weight / total * f64::from(v[d]);
                    }
                }
            }
        }
    }
    Ok((out, visible))
}

/// The eager composition on the inputs' device, as the encoders compute it
/// in `core`: content and position products as tensor matmuls, the shifted
/// or clamped position rows gathered from the full product, an additive mask
/// (`-1e9`), an F32 softmax and the value product. Queries that see no key
/// are zeroed, as [`reference`] leaves them. `[B, Tq, H·Dh]` in the call's
/// output dtype.
pub fn composed(spec: &RelPosAttention<'_>) -> Result<Tensor> {
    let dims = spec.dims()?;
    let RelPosDims {
        batch,
        q_len,
        k_len,
        heads,
        head_dim,
        pos_rows,
    } = dims;
    let device = spec.q.device();
    let core = spec.core;
    let heads_first =
        |t: &Tensor| -> Result<Tensor> { t.to_dtype(core)?.transpose(1, 2)?.contiguous() };
    let (q, k, v) = (
        heads_first(spec.q)?,
        heads_first(spec.k)?,
        heads_first(spec.v)?,
    );
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let key_lens = key_lens(spec, &dims)?;
    let mut index = Vec::with_capacity(q_len * k_len);
    for i in 0..q_len {
        for j in 0..k_len {
            index.push(match spec.position {
                RelPosition::TransformerXl { .. } => (q_len - 1 - i + j) as u32,
                RelPosition::Shaw { clamp, center } => shaw_row(i, j, clamp, center) as u32,
            });
        }
    }
    let index = Tensor::from_vec(index, (1, 1, q_len, k_len), device)?
        .broadcast_as((batch, heads, q_len, k_len))?
        .contiguous()?;
    let scores = match spec.position {
        RelPosition::TransformerXl { bias_u, bias_v } => {
            let bias = |bias: &Tensor| bias.to_dtype(core)?.reshape((1, heads, 1, head_dim));
            let q_u = q.broadcast_add(&bias(bias_u)?)?;
            let q_v = q.broadcast_add(&bias(bias_v)?)?;
            let content = q_u.matmul(&k_t)?;
            let pos_t = spec
                .pos
                .to_dtype(core)?
                .permute((1, 2, 0))?
                .unsqueeze(0)?
                .broadcast_as((batch, heads, head_dim, pos_rows))?
                .contiguous()?;
            let position = q_v.matmul(&pos_t)?.gather(&index, 3)?;
            content.add(&position)?.affine(spec.scale, 0.0)?
        }
        RelPosition::Shaw { .. } => {
            let content = q.matmul(&k_t)?.affine(spec.scale, 0.0)?;
            let pos_t = spec
                .pos
                .to_dtype(DType::F32)?
                .t()?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((batch, heads, head_dim, pos_rows))?
                .contiguous()?;
            let position = q
                .to_dtype(DType::F32)?
                .matmul(&pos_t)?
                .gather(&index, 3)?
                .affine(spec.scale, 0.0)?
                .to_dtype(core)?;
            content.add(&position)?
        }
    };
    let mut mask = vec![-1e9f32; batch * q_len * k_len];
    let mut seen = vec![0f32; batch * q_len];
    for b in 0..batch {
        for i in 0..q_len {
            let (lo, hi) = visible_keys(spec.band, i, key_lens[b]);
            for j in lo..hi {
                mask[(b * q_len + i) * k_len + j] = 0.0;
            }
            if hi > lo {
                seen[b * q_len + i] = 1.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (batch, 1, q_len, k_len), device)?.to_dtype(core)?;
    let scores = scores
        .broadcast_add(&mask)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let probabilities = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(core)?;
    let seen = Tensor::from_vec(seen, (batch, q_len, 1), device)?;
    probabilities
        .matmul(&v)?
        .transpose(1, 2)?
        .reshape((batch, q_len, heads * head_dim))?
        .to_dtype(DType::F32)?
        .broadcast_mul(&seen)?
        .to_dtype(spec.out_dtype)
}

/// How a fused result compares with eager over the queries that see a key.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Agreement {
    pub values: usize,
    /// The largest `|exact|`.
    pub peak: f64,
    /// `max |fused - exact|`.
    pub fused_error: f64,
    /// `max |eager - exact|`.
    pub eager_error: f64,
    /// `max |fused - eager|`.
    pub fused_vs_eager: f64,
}

/// Bounds relative to the output's peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// How much further from exact than eager the fused result may be.
    pub slack: f64,
    /// The largest fused-to-eager difference.
    pub versus_eager: f64,
}

/// The bound for a call rounding to `core` with output `out_dtype`: F32
/// throughout, 1e-5 of the peak beyond eager's own error and 1e-4 from
/// eager; with BF16 anywhere, 2^-7 and 2^-4 (eager and fused round
/// near-equal scores to neighbouring BF16 values).
pub fn tolerance(core: DType, out_dtype: DType) -> Tolerance {
    if core == DType::F32 && out_dtype == DType::F32 {
        Tolerance {
            slack: 1e-5,
            versus_eager: 1e-4,
        }
    } else {
        Tolerance {
            slack: 2f64.powi(-7),
            versus_eager: 2f64.powi(-4),
        }
    }
}

impl Agreement {
    /// Compares `fused` and `eager` (`[B, Tq, H·Dh]` flattened) with
    /// [`exact_reference`]'s values over its visible queries. Non-finite
    /// differences count as infinite.
    pub fn measure(
        fused: &[f32],
        eager: &[f32],
        exact: &[f64],
        visible: &[bool],
        head_dim: usize,
    ) -> Self {
        let mut agreement = Self::default();
        let worst = |worst: f64, a: f64, b: f64| {
            let diff = (a - b).abs();
            if diff.is_finite() {
                worst.max(diff)
            } else {
                f64::INFINITY
            }
        };
        if fused.len() != exact.len() || eager.len() != exact.len() {
            agreement.fused_error = f64::INFINITY;
            return agreement;
        }
        for (index, value) in exact.iter().enumerate() {
            if !visible[index / head_dim] {
                continue;
            }
            let (fused, eager) = (f64::from(fused[index]), f64::from(eager[index]));
            agreement.values += 1;
            agreement.peak = agreement.peak.max(value.abs());
            agreement.fused_error = worst(agreement.fused_error, fused, *value);
            agreement.eager_error = worst(agreement.eager_error, eager, *value);
            agreement.fused_vs_eager = worst(agreement.fused_vs_eager, fused, eager);
        }
        agreement
    }

    /// Whether the fused result is within `tolerance` of eager.
    pub fn within(&self, tolerance: Tolerance) -> bool {
        let peak = self.peak.max(1e-6);
        self.fused_error <= self.eager_error + tolerance.slack * peak
            && self.fused_vs_eager <= tolerance.versus_eager * peak
    }
}

/// Kill switch for the fused CUDA kernel (any false boolean, e.g. `0`),
/// read once per process.
pub const CUDA_RELPOS_ATTENTION_ENV: &str = "IZWI_CUDA_RELPOS_ATTENTION";

static FUSED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of calls the fused CUDA kernel served.
pub fn fused_calls() -> u64 {
    FUSED_CALLS.load(Ordering::Relaxed)
}

/// Whether the fused kernel serves `device`. The first call for a CUDA
/// device runs the kernel's self-check (every check call, against
/// [`composed`] within [`tolerance`] and against [`reference`] bit for bit,
/// reported) and logs the verdict; later calls read it. The encoders' loaders
/// call this so the check runs at load, before admission prices the
/// encoder ([`fused_priced`]). False off CUDA, without the `cuda` feature,
/// or with `IZWI_CUDA_RELPOS_ATTENTION=0`.
pub fn prepare_cuda_kernel(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::enabled(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        false
    }
}

/// Why `device` runs the eager composition, or `None` when the fused kernel
/// serves it.
pub fn eager_reason(device: &Device) -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::eager_reason(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Some("not a CUDA build".to_string())
    }
}

/// The kernel's state on `device` for the encoders' diagnostics.
pub fn diagnostics(device: &Device) -> serde_json::Value {
    let reason = eager_reason(device);
    serde_json::json!({
        "path": if reason.is_none() { "fused" } else { "eager" },
        "eager_reason": reason,
        "fused_calls_total": fused_calls(),
    })
}

/// Whether admission may price an encoder on `backend` without eager's
/// score and position planes: CUDA, the switch on, and the kernel passed
/// its self-check on a device and failed it on none. Before a loader has
/// checked a device this is false, which prices eager's larger buffers.
pub fn fused_priced(backend: BackendKind) -> bool {
    #[cfg(feature = "cuda")]
    {
        backend == BackendKind::Cuda && cuda_impl::priced()
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = backend;
        false
    }
}

/// The fused op on the inputs' device, or `None` where the encoder keeps its
/// eager composition: off CUDA, with the kernel off for the device, or for
/// dtypes, layouts or sizes the kernel does not take. Never runs a
/// self-check: [`prepare_cuda_kernel`] decides at load, so a call inside a
/// CUDA graph capture only reads the verdict.
pub fn relpos_attention(spec: &RelPosAttention<'_>) -> Result<Option<Tensor>> {
    if !spec.q.device().is_cuda() {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        if !cuda_impl::decided_on(spec.q.device()) {
            return Ok(None);
        }
        let out = cuda_impl::run(spec)?;
        if out.is_some() {
            FUSED_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(None)
    }
}

/// The single-launch CUDA form and its per-device self-check.
#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, PoisonError};

    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Device, DeviceLocation, Layout, Result, Shape, Storage, Tensor};

    use super::{
        check, composed, core_scale, exact_reference, max_span, reference, tolerance, Agreement,
        KeyBand, KeyLengths, RelPosAttention, RelPosition, CUDA_RELPOS_ATTENTION_ENV,
    };
    use crate::kernels::numeric_check::host_f32;

    const MODULE: &str = "izwi_relpos_attention";
    const WARPS: usize = 8;
    const THREADS: u32 = 256;
    /// Dynamic shared memory without raising the function's limit: 1,536
    /// keys per query.
    const MAX_SHARED_BYTES: usize = 48 * 1024;
    /// Devices past this ordinal keep the composition.
    const MAX_DEVICES: usize = 16;
    const UNCHECKED: u8 = 0;
    const ON: u8 = 1;
    const OFF: u8 = 2;
    /// Each device's self-check verdict, and why it is off.
    static VERDICT: [AtomicU8; MAX_DEVICES] = [const { AtomicU8::new(UNCHECKED) }; MAX_DEVICES];
    static REASON: Mutex<[Option<String>; MAX_DEVICES]> = Mutex::new([const { None }; MAX_DEVICES]);
    static CHECKING: Mutex<()> = Mutex::new(());

    fn switched_on() -> bool {
        static SWITCH: crate::env::ReadOnce<bool> =
            crate::env::ReadOnce::new(|| crate::env::flag_or(CUDA_RELPOS_ATTENTION_ENV, true));
        SWITCH.get()
    }

    fn verdict(device: &Device) -> Option<(usize, &'static AtomicU8)> {
        let DeviceLocation::Cuda { gpu_id } = device.location() else {
            return None;
        };
        VERDICT.get(gpu_id).map(|verdict| (gpu_id, verdict))
    }

    /// Whether the kernel serves `device`, running its self-check first if
    /// this device has none yet.
    pub(super) fn enabled(device: &Device) -> bool {
        if !switched_on() {
            return false;
        }
        let Some((gpu_id, verdict)) = verdict(device) else {
            return false;
        };
        match verdict.load(Ordering::Acquire) {
            ON => true,
            OFF => false,
            _ => decide(device, gpu_id, verdict),
        }
    }

    /// Whether `device` passed its self-check (no check runs here).
    pub(super) fn decided_on(device: &Device) -> bool {
        switched_on()
            && verdict(device).is_some_and(|(_, verdict)| verdict.load(Ordering::Acquire) == ON)
    }

    pub(super) fn eager_reason(device: &Device) -> Option<String> {
        if !switched_on() {
            return Some(format!("{CUDA_RELPOS_ATTENTION_ENV} is off"));
        }
        let Some((gpu_id, verdict)) = verdict(device) else {
            return Some("not a checked CUDA device".to_string());
        };
        match verdict.load(Ordering::Acquire) {
            ON => None,
            OFF => Some(
                REASON.lock().unwrap_or_else(PoisonError::into_inner)[gpu_id]
                    .clone()
                    .unwrap_or_else(|| "self-check failed".to_string()),
            ),
            _ => Some("self-check not run yet".to_string()),
        }
    }

    pub(super) fn priced() -> bool {
        let verdicts = VERDICT
            .iter()
            .map(|verdict| verdict.load(Ordering::Acquire));
        let (mut on, mut off) = (false, false);
        for verdict in verdicts {
            on |= verdict == ON;
            off |= verdict == OFF;
        }
        switched_on() && on && !off
    }

    fn decide(device: &Device, gpu_id: usize, verdict: &AtomicU8) -> bool {
        let _checking = CHECKING.lock().unwrap_or_else(PoisonError::into_inner);
        match verdict.load(Ordering::Acquire) {
            ON => return true,
            OFF => return false,
            _ => {}
        }
        let reason = match self_check(device) {
            Ok(report) => match &report.failure {
                None => {
                    tracing::info!(
                        "relative-position attention CUDA kernel passed its self-check on \
                         cuda:{gpu_id}: {} calls, {} values, {} differ in bits from the host \
                         reference, worst fused-to-eager difference {:e} of the peak; one \
                         launch per conformer attention",
                        report.cases,
                        report.values,
                        report.differing,
                        report.worst_ratio,
                    );
                    None
                }
                Some(failure) => {
                    tracing::warn!(
                        "relative-position attention CUDA kernel failed its self-check on \
                         cuda:{gpu_id} ({failure}); keeping the eager attention"
                    );
                    Some(format!("self-check failed: {failure}"))
                }
            },
            Err(error) => {
                tracing::warn!(
                    %error,
                    "relative-position attention CUDA kernel self-check could not run on \
                     cuda:{gpu_id}; keeping the eager attention"
                );
                Some(format!("self-check could not run: {error}"))
            }
        };
        let passed = reason.is_none();
        REASON.lock().unwrap_or_else(PoisonError::into_inner)[gpu_id] = reason;
        verdict.store(if passed { ON } else { OFF }, Ordering::Release);
        passed
    }

    fn slice<T: CudaDType>(storage: &Storage) -> Result<&CudaSlice<T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("the CUDA relative-position attention needs CUDA tensors")
        };
        cuda.as_cuda_slice::<T>()
    }

    fn view<'a, T: CudaDType>(storage: &'a Storage, layout: &Layout) -> Result<CudaView<'a, T>> {
        Ok(slice::<T>(storage)?.slice(layout.start_offset()..))
    }

    fn stride(layout: &Layout, dim: usize) -> i64 {
        layout.stride()[dim] as i64
    }

    fn i32_of(value: usize) -> Option<i32> {
        i32::try_from(value).ok()
    }

    /// One launch, or `None` for what the kernel does not take: storage
    /// other than F32 or BF16 (or mixed), operands on another device, a dim
    /// that is not contiguous, more keys per query than its shared memory
    /// holds, or sizes past `i32`.
    pub(super) fn run(spec: &RelPosAttention<'_>) -> Result<Option<Tensor>> {
        let dims = spec.dims()?;
        let storage = spec.q.dtype();
        let s_tag = match storage {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            _ => return Ok(None),
        };
        let o_tag = if spec.out_dtype == DType::BF16 {
            "bf16"
        } else {
            "f32"
        };
        let operands = [spec.k, spec.v, spec.pos];
        let device = spec.q.device();
        if operands
            .iter()
            .any(|t| t.dtype() != storage || !t.device().same_device(device))
            || [spec.q, spec.k, spec.v, spec.pos]
                .iter()
                .any(|t| t.layout().stride().last() != Some(&1))
        {
            return Ok(None);
        }
        let span = max_span(spec.band, dims.k_len);
        let shared = WARPS * span * std::mem::size_of::<f32>();
        let (shaw, clamp, center) = match spec.position {
            RelPosition::TransformerXl { .. } => (0, 0, 0),
            RelPosition::Shaw { clamp, center } => (1, clamp, center),
        };
        let (band, band_size, left_chunks, query_origin, key_origin) = match spec.band {
            KeyBand::Full => (0, 0, 0, 0, 0),
            KeyBand::Chunked {
                chunk,
                left_chunks,
                query_origin,
                key_origin,
            } => (1, chunk, left_chunks, query_origin, key_origin),
            KeyBand::Block { size } => (2, size, 0, 0, 0),
        };
        let uniform = match spec.key_lengths {
            KeyLengths::Uniform(len) => len.min(dims.k_len),
            _ => dims.k_len,
        };
        let ints = [
            dims.q_len,
            dims.k_len,
            dims.heads,
            dims.head_dim,
            clamp,
            center,
            band_size,
            left_chunks,
            uniform,
            span,
        ]
        .map(i32_of);
        let Some(blocks) = u32::try_from(dims.q_len.div_ceil(WARPS)).ok() else {
            return Ok(None);
        };
        if shared > MAX_SHARED_BYTES
            || ints.iter().any(Option::is_none)
            || dims.heads > 65_535
            || dims.batch > 65_535
            || dims
                .q_len
                .checked_add(dims.k_len)
                .is_none_or(|len| i32_of(len).is_none())
        {
            return Ok(None);
        }
        let [q_len, k_len, heads, head_dim, clamp, center, band_size, left_chunks, uniform, span] =
            ints.map(|value| value.unwrap_or_default());
        let (bias_u, bias_v) = match spec.position {
            RelPosition::TransformerXl { bias_u, bias_v } => (
                Some(bias_u.to_dtype(DType::F32)?.contiguous()?),
                Some(bias_v.to_dtype(DType::F32)?.contiguous()?),
            ),
            RelPosition::Shaw { .. } => (None, None),
        };
        let key_lens = match spec.key_lengths {
            KeyLengths::PerRow(lengths) => {
                if !lengths.device().same_device(device) {
                    return Ok(None);
                }
                Some(lengths.contiguous()?)
            }
            KeyLengths::All | KeyLengths::Uniform(_) => None,
        };
        let total = dims.batch * dims.q_len * dims.heads * dims.head_dim;
        let cuda = device.as_cuda_device()?;
        let function = cuda.get_or_load_custom_func(
            &format!("izwi_relpos_attention_{s_tag}_{o_tag}"),
            MODULE,
            crate::kernels::cuda::cuda_ptx_relpos_attention(),
        )?;
        let (q_storage, q_layout) = spec.q.storage_and_layout();
        let (k_storage, k_layout) = spec.k.storage_and_layout();
        let (v_storage, v_layout) = spec.v.storage_and_layout();
        let (p_storage, p_layout) = spec.pos.storage_and_layout();
        let u_storage = bias_u.as_ref().map(Tensor::storage_and_layout);
        let w_storage = bias_v.as_ref().map(Tensor::storage_and_layout);
        let l_storage = key_lens.as_ref().map(Tensor::storage_and_layout);
        let (p_sr, p_sh) = match spec.position {
            RelPosition::TransformerXl { .. } => (stride(p_layout, 0), stride(p_layout, 1)),
            RelPosition::Shaw { .. } => (stride(p_layout, 0), 0),
        };
        let config = LaunchConfig {
            grid_dim: (blocks, dims.heads as u32, dims.batch as u32),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let null = 0u64;
        macro_rules! go {
            ($s:ty, $o:ty) => {{
                let qs = view::<$s>(&q_storage, q_layout)?;
                let ks = view::<$s>(&k_storage, k_layout)?;
                let vs = view::<$s>(&v_storage, v_layout)?;
                let ps = view::<$s>(&p_storage, p_layout)?;
                let us = u_storage
                    .as_ref()
                    .map(|(storage, layout)| view::<f32>(storage, layout))
                    .transpose()?;
                let ws = w_storage
                    .as_ref()
                    .map(|(storage, layout)| view::<f32>(storage, layout))
                    .transpose()?;
                let ls = l_storage
                    .as_ref()
                    .map(|(storage, layout)| view::<u32>(storage, layout))
                    .transpose()?;
                // SAFETY: the kernel writes every output element.
                let out = unsafe { cuda.alloc::<$o>(total)? };
                let mut builder = function.builder();
                builder.arg(&qs);
                builder.arg(&ks);
                builder.arg(&vs);
                builder.arg(&ps);
                match &us {
                    Some(us) => builder.arg(us),
                    None => builder.arg(&null),
                };
                match &ws {
                    Some(ws) => builder.arg(ws),
                    None => builder.arg(&null),
                };
                match &ls {
                    Some(ls) => builder.arg(ls),
                    None => builder.arg(&null),
                };
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    q_len,
                    k_len,
                    heads,
                    head_dim,
                    stride(q_layout, 0),
                    stride(q_layout, 1),
                    stride(q_layout, 2),
                    stride(k_layout, 0),
                    stride(k_layout, 1),
                    stride(k_layout, 2),
                    stride(v_layout, 0),
                    stride(v_layout, 1),
                    stride(v_layout, 2),
                    p_sr,
                    p_sh,
                    shaw,
                    clamp,
                    center,
                    band,
                    band_size,
                    left_chunks,
                    query_origin,
                    key_origin,
                    uniform,
                    core_scale(spec.scale, spec.core),
                    spec.scale as f32,
                    i32::from(spec.core == DType::BF16),
                    span
                );
                // SAFETY: argument order and types match the kernel. Every
                // read stays inside the operands: queries [B, Tq, H, Dh],
                // keys and values [B, Tk, H, Dh] through their strides,
                // position rows below `Tq + Tk - 1` (XL) or within the
                // validated clamp around `center` (Shaw), and one length
                // per row; it writes `total` outputs and at most `span`
                // scores per warp of shared memory.
                unsafe { builder.launch(config) }.w()?;
                Tensor::from_storage(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(out, cuda.clone())),
                    Shape::from((dims.batch, dims.q_len, dims.heads * dims.head_dim)),
                    BackpropOp::none(),
                    false,
                )
            }};
        }
        use half::bf16;
        Ok(Some(match (storage, spec.out_dtype) {
            (DType::F32, DType::F32) => go!(f32, f32),
            (DType::F32, _) => go!(f32, bf16),
            (_, DType::F32) => go!(bf16, f32),
            _ => go!(bf16, bf16),
        }))
    }

    /// What [`self_check`] compared.
    pub(super) struct SelfCheck {
        pub(super) cases: usize,
        pub(super) values: usize,
        /// Values whose bits differ from the host reference's.
        pub(super) differing: usize,
        /// The largest fused-to-eager difference over a call's peak.
        pub(super) worst_ratio: f64,
        /// The first call past its tolerance, if any.
        pub(super) failure: Option<String>,
    }

    /// The kernel on `device` over every check call, against the eager
    /// composition on the same device within [`tolerance`] and against the
    /// host reference (bits counted). Queries that see no key must be zero.
    /// Sets no verdict.
    pub(super) fn self_check(device: &Device) -> Result<SelfCheck> {
        let mut report = SelfCheck {
            cases: 0,
            values: 0,
            differing: 0,
            worst_ratio: 0.0,
            failure: None,
        };
        for shape in &check::SHAPES {
            let case = check::Case::build(shape, device)?;
            let spec = case.spec();
            let dims = spec.dims()?;
            let Some(fused) = run(&spec)? else {
                report.failure = Some(format!("{}: the kernel refused the call", shape.name));
                return Ok(report);
            };
            let fused = host_f32(&fused)?;
            let host = host_f32(&reference(&spec)?)?;
            let eager = host_f32(&composed(&spec)?)?;
            let (exact, visible) = exact_reference(&spec)?;
            let agreement = Agreement::measure(&fused, &eager, &exact, &visible, dims.head_dim);
            report.cases += 1;
            report.values += fused.len();
            report.differing += fused
                .iter()
                .zip(&host)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            report.worst_ratio = report
                .worst_ratio
                .max(agreement.fused_vs_eager / agreement.peak.max(1e-6));
            let unseen_nonzero = visible.iter().enumerate().any(|(query, seen)| {
                !seen
                    && fused[query * dims.head_dim..(query + 1) * dims.head_dim]
                        .iter()
                        .any(|value| *value != 0.0)
            });
            if report.failure.is_none()
                && (unseen_nonzero || !agreement.within(tolerance(spec.core, spec.out_dtype)))
            {
                report.failure = Some(format!(
                    "{}: {agreement:?}{}",
                    shape.name,
                    if unseen_nonzero {
                        ", a query without keys is not zero"
                    } else {
                        ""
                    }
                ));
            }
        }
        Ok(report)
    }
}

/// The calls the CPU tests and the CUDA self-check compare: the encoders'
/// shapes (Parakeet with padded rows, its BF16 tier's F32 core over BF16
/// storage, Nemotron's chunked band offline and with streaming cache keys,
/// Granite's block-local Shaw with padded rows and as one row), a band that
/// leaves late queries without a key, one frame, and `Dh = 256`.
#[cfg(any(test, feature = "cuda"))]
mod check {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;

    pub(super) const NEMOTRON_BAND: KeyBand = KeyBand::Chunked {
        chunk: 14,
        left_chunks: 4,
        query_origin: 0,
        key_origin: 0,
    };

    /// One check call's geometry and dtypes.
    pub(super) struct Shape {
        pub(super) name: &'static str,
        pub(super) seed: u64,
        /// `(B, Tq, Tk, H, Dh)`.
        pub(super) dims: (usize, usize, usize, usize, usize),
        /// Shaw `(clamp, center, table rows)`; `None` for Transformer-XL.
        pub(super) shaw: Option<(usize, usize, usize)>,
        pub(super) band: KeyBand,
        /// Per-row key lengths; empty for none.
        pub(super) lengths: &'static [u32],
        /// One key length for every row (`0`: none).
        pub(super) uniform: usize,
        pub(super) storage: DType,
        pub(super) core: DType,
        pub(super) out_dtype: DType,
    }

    const fn shape(
        name: &'static str,
        seed: u64,
        dims: (usize, usize, usize, usize, usize),
        shaw: Option<(usize, usize, usize)>,
        band: KeyBand,
        lengths: &'static [u32],
        [storage, core, out_dtype]: [DType; 3],
    ) -> Shape {
        Shape {
            name,
            seed,
            dims,
            shaw,
            band,
            lengths,
            uniform: 0,
            storage,
            core,
            out_dtype,
        }
    }

    const F32: [DType; 3] = [DType::F32; 3];
    const BF16: [DType; 3] = [DType::BF16; 3];
    const GRANITE: Option<(usize, usize, usize)> = Some((16, 40, 81));
    const BLOCK: KeyBand = KeyBand::Block { size: 16 };

    pub(super) const SHAPES: [Shape; 11] = [
        shape(
            "parakeet",
            1,
            (2, 37, 37, 2, 128),
            None,
            KeyBand::Full,
            &[37, 20],
            F32,
        ),
        shape(
            "parakeet bf16 tier",
            2,
            (2, 29, 29, 2, 128),
            None,
            KeyBand::Full,
            &[29, 11],
            [DType::BF16, DType::F32, DType::BF16],
        ),
        shape(
            "nemotron",
            3,
            (2, 61, 61, 2, 128),
            None,
            NEMOTRON_BAND,
            &[61, 33],
            F32,
        ),
        shape(
            "nemotron bf16",
            4,
            (2, 61, 61, 2, 128),
            None,
            NEMOTRON_BAND,
            &[61, 33],
            BF16,
        ),
        shape(
            "nemotron streaming",
            5,
            (1, 14, 84, 2, 128),
            None,
            KeyBand::Chunked {
                chunk: 14,
                left_chunks: 4,
                query_origin: 3,
                key_origin: 3 - 70,
            },
            &[],
            BF16,
        ),
        shape(
            "granite",
            6,
            (2, 50, 50, 2, 128),
            GRANITE,
            BLOCK,
            &[50, 21],
            F32,
        ),
        shape(
            "granite bf16",
            7,
            (2, 50, 50, 2, 128),
            GRANITE,
            BLOCK,
            &[50, 21],
            BF16,
        ),
        Shape {
            uniform: 45,
            ..shape(
                "granite one row",
                8,
                (1, 50, 50, 2, 128),
                GRANITE,
                BLOCK,
                &[],
                BF16,
            )
        },
        shape(
            "short band",
            9,
            (2, 40, 40, 1, 64),
            None,
            KeyBand::Chunked {
                chunk: 4,
                left_chunks: 1,
                query_origin: 0,
                key_origin: 0,
            },
            &[40, 5],
            F32,
        ),
        shape(
            "one frame",
            10,
            (1, 1, 1, 1, 64),
            None,
            KeyBand::Full,
            &[],
            F32,
        ),
        shape(
            "dh 256",
            11,
            (1, 9, 9, 1, 256),
            Some((4, 8, 17)),
            KeyBand::Block { size: 4 },
            &[],
            F32,
        ),
    ];

    /// A check call's owned inputs.
    pub(super) struct Case {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos: Tensor,
        bias: Option<(Tensor, Tensor)>,
        shaw: (usize, usize),
        band: KeyBand,
        lengths: Option<Tensor>,
        uniform: usize,
        core: DType,
        out_dtype: DType,
    }

    impl Case {
        /// Xorshift inputs for `shape` on `device`: queries at 3x the keys'
        /// scale so the softmax is neither flat nor one-hot, biases rounded
        /// through the core dtype as the BF16 encoders cast theirs.
        pub(super) fn build(shape: &Shape, device: &Device) -> Result<Self> {
            let (batch, q_len, k_len, heads, head_dim) = shape.dims;
            let seed = shape.seed * 16;
            let tensor = |dims: &[usize], seed: u64, scale: f32| {
                xorshift_tensor(dims, seed, scale, device)?.to_dtype(shape.storage)
            };
            let (pos, bias, shaw) = match shape.shaw {
                Some((clamp, center, rows)) => (
                    tensor(&[rows, head_dim], seed + 4, 1.0)?,
                    None,
                    (clamp, center),
                ),
                None => {
                    let bias = |seed| {
                        xorshift_tensor(&[heads, head_dim], seed, 0.5, device)?
                            .to_dtype(shape.core)?
                            .to_dtype(DType::F32)
                    };
                    (
                        tensor(&[q_len + k_len - 1, heads, head_dim], seed + 4, 1.0)?,
                        Some((bias(seed + 5)?, bias(seed + 6)?)),
                        (0, 0),
                    )
                }
            };
            Ok(Self {
                q: tensor(&[batch, q_len, heads, head_dim], seed + 1, 3.0)?,
                k: tensor(&[batch, k_len, heads, head_dim], seed + 2, 1.0)?,
                v: tensor(&[batch, k_len, heads, head_dim], seed + 3, 1.0)?,
                pos,
                bias,
                shaw,
                band: shape.band,
                lengths: (!shape.lengths.is_empty())
                    .then(|| Tensor::new(shape.lengths, device))
                    .transpose()?,
                uniform: shape.uniform,
                core: shape.core,
                out_dtype: shape.out_dtype,
            })
        }

        pub(super) fn spec(&self) -> RelPosAttention<'_> {
            RelPosAttention {
                q: &self.q,
                k: &self.k,
                v: &self.v,
                pos: &self.pos,
                position: match &self.bias {
                    Some((bias_u, bias_v)) => RelPosition::TransformerXl { bias_u, bias_v },
                    None => RelPosition::Shaw {
                        clamp: self.shaw.0,
                        center: self.shaw.1,
                    },
                },
                band: self.band,
                key_lengths: match (&self.lengths, self.uniform) {
                    (Some(lengths), _) => KeyLengths::PerRow(lengths),
                    (None, 0) => KeyLengths::All,
                    (None, len) => KeyLengths::Uniform(len),
                },
                scale: 1.0 / (self.q.dims()[3] as f64).sqrt(),
                core: self.core,
                out_dtype: self.out_dtype,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::check::{Case, NEMOTRON_BAND, SHAPES};
    use super::*;
    use crate::kernels::numeric_check::{host_f32, xorshift_tensor};

    #[test]
    fn reference_matches_the_exact_op_and_the_eager_composition() {
        for shape in &SHAPES {
            let (name, case) = (shape.name, Case::build(shape, &Device::Cpu).unwrap());
            let spec = case.spec();
            let dims = spec.dims().unwrap();
            let fused = host_f32(&reference(&spec).unwrap()).unwrap();
            let eager = host_f32(&composed(&spec).unwrap()).unwrap();
            let (exact, visible) = exact_reference(&spec).unwrap();
            let agreement = Agreement::measure(&fused, &eager, &exact, &visible, dims.head_dim);
            eprintln!("{name}: {agreement:?}");
            assert!(agreement.values > 0, "{name}");
            assert!(
                agreement.within(tolerance(spec.core, spec.out_dtype)),
                "{name}: {agreement:?}"
            );
            // Queries that see no key are zero, and only those.
            for (query, seen) in visible.iter().enumerate() {
                let row = &fused[query * dims.head_dim..(query + 1) * dims.head_dim];
                if !seen {
                    assert!(
                        row.iter().all(|value| *value == 0.0),
                        "{name} query {query}"
                    );
                }
            }
        }
    }

    /// A short band leaves the late queries of a row cut short without a
    /// key, and those come out zero.
    #[test]
    fn queries_without_keys_are_zero() {
        let shape = SHAPES
            .iter()
            .find(|shape| shape.name == "short band")
            .unwrap();
        let case = Case::build(shape, &Device::Cpu).unwrap();
        let spec = case.spec();
        let (_, visible) = exact_reference(&spec).unwrap();
        assert!(visible.iter().any(|seen| !seen));
        let fused = host_f32(&reference(&spec).unwrap()).unwrap();
        for (query, seen) in visible.iter().enumerate() {
            let row = &fused[query * 64..(query + 1) * 64];
            assert_eq!(
                row.iter().any(|value| *value != 0.0),
                *seen,
                "query {query}"
            );
        }
    }

    fn xl_spec<'a>(
        [q, k, v, pos]: [&'a Tensor; 4],
        (bias_u, bias_v): (&'a Tensor, &'a Tensor),
        band: KeyBand,
        key_lengths: KeyLengths<'a>,
    ) -> RelPosAttention<'a> {
        RelPosAttention {
            q,
            k,
            v,
            pos,
            position: RelPosition::TransformerXl { bias_u, bias_v },
            band,
            key_lengths,
            scale: 0.125,
            core: DType::F32,
            out_dtype: DType::F32,
        }
    }

    /// A row's outputs depend on its own real keys only: alone and unpadded,
    /// or padded beside another row with NaN in its padded keys and values,
    /// its real queries come out bit for bit the same (the position table
    /// holds the same rows for the same distances).
    #[test]
    fn a_row_ignores_its_batch_padding_and_padded_values() {
        let cpu = &Device::Cpu;
        let (heads, head_dim, real, padded) = (2, 64, 20, 33);
        // Distances -(padded - 1) ..= padded - 1, largest first.
        let table = xorshift_tensor(&[2 * padded - 1, heads, head_dim], 21, 1.0, cpu).unwrap();
        let positions = |len: usize| table.narrow(0, padded - len, 2 * len - 1).unwrap();
        let bias_u = xorshift_tensor(&[heads, head_dim], 22, 0.5, cpu).unwrap();
        let bias_v = xorshift_tensor(&[heads, head_dim], 23, 0.5, cpu).unwrap();
        let rows = |seed| xorshift_tensor(&[1, padded, heads, head_dim], seed, 1.0, cpu).unwrap();
        let (q, k, v) = (rows(24), rows(25), rows(26));
        let poison = |t: &Tensor| {
            let nan = Tensor::full(f32::NAN, (1, padded - real, heads, head_dim), cpu).unwrap();
            Tensor::cat(&[&t.narrow(1, 0, real).unwrap(), &nan], 1).unwrap()
        };
        let other = |seed| xorshift_tensor(&[1, padded, heads, head_dim], seed, 1.0, cpu).unwrap();
        let batch = |row: Tensor, seed| Tensor::cat(&[&row, &other(seed)], 0).unwrap();
        let (bq, bk, bv) = (
            batch(q.clone(), 27),
            batch(poison(&k), 28),
            batch(poison(&v), 29),
        );
        let lengths = Tensor::new(&[real as u32, padded as u32], cpu).unwrap();
        for band in [KeyBand::Full, NEMOTRON_BAND] {
            let alone_q = q.narrow(1, 0, real).unwrap();
            let alone_k = k.narrow(1, 0, real).unwrap();
            let alone_v = v.narrow(1, 0, real).unwrap();
            let alone_pos = positions(real);
            let bias = (&bias_u, &bias_v);
            let alone = xl_spec(
                [&alone_q, &alone_k, &alone_v, &alone_pos],
                bias,
                band,
                KeyLengths::All,
            );
            let alone = host_f32(&reference(&alone).unwrap()).unwrap();
            let padded_pos = positions(padded);
            let together = xl_spec(
                [&bq, &bk, &bv, &padded_pos],
                bias,
                band,
                KeyLengths::PerRow(&lengths),
            );
            let together = reference(&together)
                .unwrap()
                .narrow(0, 0, 1)
                .unwrap()
                .narrow(1, 0, real)
                .unwrap();
            let together = host_f32(&together).unwrap();
            assert_eq!(
                alone.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                together.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{band:?}"
            );
        }
    }

    /// The families' `rel_shift` (pad one zero column, view as `[2T, T]`,
    /// drop the first row, view as `[T, 2T - 1]`) is the gather
    /// `x[i][T - 1 - i + j]` this op reads positions with.
    #[test]
    fn rel_shift_is_the_toeplitz_gather() {
        let cpu = &Device::Cpu;
        for t in [1, 2, 5, 16] {
            let x = xorshift_tensor(&[1, 2, t, 2 * t - 1], t as u64, 1.0, cpu).unwrap();
            let shifted = x
                .pad_with_zeros(3, 1, 0)
                .unwrap()
                .reshape((1, 2, 2 * t, t))
                .unwrap()
                .narrow(2, 1, 2 * t - 1)
                .unwrap()
                .reshape((1, 2, t, 2 * t - 1))
                .unwrap()
                .narrow(3, 0, t)
                .unwrap();
            let (x, shifted) = (host_f32(&x).unwrap(), host_f32(&shifted).unwrap());
            for h in 0..2 {
                for i in 0..t {
                    for j in 0..t {
                        assert_eq!(
                            shifted[(h * t + i) * t + j],
                            x[(h * t + i) * (2 * t - 1) + t - 1 - i + j]
                        );
                    }
                }
            }
        }
    }

    /// The band reproduces NeMo's `chunked_limited` mask (Nemotron's
    /// `chunked_limited_visible`), cache keys before the origin included.
    #[test]
    fn chunked_band_is_the_chunked_limited_mask() {
        let rule = |query: i64, key: i64, left: i64, chunk: i64| {
            let diff = query.div_euclid(chunk) - key.div_euclid(chunk);
            diff >= 0 && diff <= left / chunk
        };
        for (chunk, left, query_origin, key_origin) in [
            (14, 56, 0, 0),
            (14, 56, 3, -67),
            (4, 9, 5, -2),
            (1, 0, 0, 0),
        ] {
            let band = KeyBand::Chunked {
                chunk,
                left_chunks: left / chunk,
                query_origin,
                key_origin,
            };
            for key_len in [0, 7, 90] {
                for i in 0..40 {
                    let (lo, hi) = visible_keys(band, i, key_len);
                    for j in 0..key_len {
                        let seen = rule(
                            query_origin + i as i64,
                            key_origin + j as i64,
                            left as i64,
                            chunk as i64,
                        );
                        assert_eq!((lo..hi).contains(&j), seen, "{band:?} i={i} j={j}");
                    }
                    assert!(hi.saturating_sub(lo) <= max_span(band, key_len));
                }
            }
        }
    }

    #[test]
    fn det_exp_is_within_two_ulps_of_exp() {
        let ulps = |got: f32, want: f64| {
            let want = want as f32;
            (got.to_bits() as i64 - want.to_bits() as i64).abs()
        };
        let mut x = -103.9f32;
        while x <= 88.0 {
            let want = f64::from(x).exp();
            let got = det_exp(x);
            if want >= f64::from(f32::MIN_POSITIVE) {
                assert!(ulps(got, want) <= 2, "x={x}: {got} vs {want}");
            } else {
                assert!(
                    (f64::from(got) - want).abs() <= 2.0 * 2f64.powi(-149),
                    "x={x}"
                );
            }
            x += 0.013_7;
        }
        assert_eq!(det_exp(0.0), 1.0);
        assert_eq!(det_exp(-0.0), 1.0);
        assert_eq!(det_exp(-104.0), 0.0);
        assert_eq!(det_exp(f32::NEG_INFINITY), 0.0);
        assert!(det_exp(f32::NAN).is_nan());
    }

    /// Writes every check call for the CUDA kernel's emulator, a C++ harness
    /// that runs `kernels/cuda/relpos_attention.cu` thread by thread (32
    /// threads per warp meeting at each shuffle) and compares its output
    /// with the host reference bit for bit. One file per call: 20 i64
    /// (B, Tq, Tk, H, Dh, position rows, Shaw, clamp, center, band, band
    /// size, left chunks, query origin, key origin, uniform key length,
    /// per-row lengths, BF16 storage, BF16 output, BF16 rounding, max span),
    /// the two f32 scales, then q, k, v, positions, `u`, `v` as f32, the
    /// lengths as u32 and the reference output as f32, little-endian.
    #[test]
    #[ignore = "writes the CUDA emulator's inputs; set IZWI_RELPOS_ATTENTION_EMULATOR_DIR"]
    fn writes_the_cuda_emulator_cases() {
        let Some(dir) = std::env::var_os("IZWI_RELPOS_ATTENTION_EMULATOR_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (index, shape) in SHAPES.iter().enumerate() {
            let case = Case::build(shape, &Device::Cpu).unwrap();
            let spec = case.spec();
            let dims = spec.dims().unwrap();
            let (shaw, clamp, center, bias) = match spec.position {
                RelPosition::TransformerXl { bias_u, bias_v } => (
                    0,
                    0,
                    0,
                    [host_f32(bias_u).unwrap(), host_f32(bias_v).unwrap()].concat(),
                ),
                RelPosition::Shaw { clamp, center } => {
                    (1, clamp, center, vec![0.0; 2 * dims.heads * dims.head_dim])
                }
            };
            let (band, band_size, left, query_origin, key_origin) = match spec.band {
                KeyBand::Full => (0, 0, 0, 0, 0),
                KeyBand::Chunked {
                    chunk,
                    left_chunks,
                    query_origin,
                    key_origin,
                } => (1, chunk, left_chunks, query_origin, key_origin),
                KeyBand::Block { size } => (2, size, 0, 0, 0),
            };
            let lens = key_lens(&spec, &dims).unwrap();
            let (uniform, per_row) = match spec.key_lengths {
                KeyLengths::PerRow(_) => (dims.k_len, 1),
                KeyLengths::Uniform(len) => (len.min(dims.k_len), 0),
                KeyLengths::All => (dims.k_len, 0),
            };
            let header = [
                dims.batch as i64,
                dims.q_len as i64,
                dims.k_len as i64,
                dims.heads as i64,
                dims.head_dim as i64,
                dims.pos_rows as i64,
                shaw,
                clamp as i64,
                center as i64,
                band,
                band_size as i64,
                left as i64,
                query_origin,
                key_origin,
                uniform as i64,
                per_row,
                i64::from(spec.q.dtype() == DType::BF16),
                i64::from(spec.out_dtype == DType::BF16),
                i64::from(spec.core == DType::BF16),
                max_span(spec.band, dims.k_len) as i64,
            ];
            let mut bytes = Vec::new();
            for value in header {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [core_scale(spec.scale, spec.core), spec.scale as f32] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let floats = [spec.q, spec.k, spec.v, spec.pos]
                .map(|tensor| host_f32(tensor).unwrap())
                .concat();
            let out = host_f32(&reference(&spec).unwrap()).unwrap();
            for value in floats.iter().chain(&bias) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for len in lens {
                bytes.extend_from_slice(&(len as u32).to_le_bytes());
            }
            for value in out {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(dir.join(format!("case{index:02}.bin")), bytes).unwrap();
        }
    }

    #[test]
    fn rejects_shapes_the_op_does_not_take() {
        use check::Shape;
        let base = || Shape {
            name: "rejected",
            seed: 31,
            dims: (1, 4, 4, 1, 64),
            shaw: None,
            band: KeyBand::Full,
            lengths: &[],
            uniform: 0,
            storage: DType::F32,
            core: DType::F32,
            out_dtype: DType::F32,
        };
        let rejected = |shape: Shape, why: &str| {
            let case = Case::build(&shape, &Device::Cpu).unwrap();
            assert!(case.spec().dims().is_err(), "{why}");
        };
        rejected(
            Shape {
                dims: (1, 4, 4, 1, 48),
                ..base()
            },
            "head dim 48",
        );
        rejected(
            Shape {
                band: KeyBand::Block { size: 0 },
                ..base()
            },
            "empty block",
        );
        rejected(
            Shape {
                shaw: Some((4, 2, 9)),
                ..base()
            },
            "table too short for the clamp",
        );
        rejected(
            Shape {
                core: DType::F16,
                ..base()
            },
            "F16 core",
        );
        let case = Case::build(&base(), &Device::Cpu).unwrap();
        assert!(case.spec().dims().is_ok());
    }
}

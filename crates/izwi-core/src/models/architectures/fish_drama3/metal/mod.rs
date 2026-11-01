//! The Metal residency (§6, Metal column; D5).
//!
//! | Part | Form on Metal |
//! |---|---|
//! | Residual stream | F32 |
//! | Dense projections (trunk, shared expert, tower, heads) | F16 weights; F16 activations in and out |
//! | Packing at load | trunk `q_proj` / `k_proj` / `v_proj` and `in_proj_qkv` / `in_proj_z` as one matrix each; the tower's contiguous `wqkv` repacked per head like the trunk's |
//! | Routed experts | raw E4M3 + F32 128×128 scales stacked per layer, the fused Qwen3.6 MoE kernels (F16 in and out) |
//! | Router, shared-expert gate, DeltaNet `a` / `b`, norm gains, conv, `A_log`, `dt_bias` | F32 |
//! | RMSNorm, q/k norm, attention gate | the Qwen3.6 F16 kernels with F32 gains (a norm reads the F32 residual through F16) |
//! | RoPE | host F64 tables rounded to F32, rotated in F32 (NeoX for the trunk, interleaved pairs for the tower) |
//! | DeltaNet | `gdn_conv` + `gdn_decode` (F32 gates) row by row in steps and spans, F32 state; heads other than 128-dim run the same rule as a Candle composition |
//! | Router | F32 GEMV, the fused routing kernel, weights rounded to BF16 |
//! | Paged KV (both domains) | F16, F32 after a breached load-time probe |
//! | Embedding and code tables | F16 device tables, gathered to F32 |
//! | Logits | F16 projections widened to F32 |
//! | Sampling | on the device, two readbacks a frame (`sampling.rs`) |
//!
//! Every output of an F16 projection is added into the F32 residual, so
//! the stream itself never rounds to 16 bits; what does is each layer's
//! activations. The load-time probe (`self_check.rs`) runs every layer on
//! xorshift input and moves a layer whose activations leave F16's safe
//! range to F32 activations (its F16 weights widened per call), and a
//! breached MoE layer to the host-routed expert loop.
//!
//! **Kernel phases (§7.4).** A step's rows run together layer by layer, but
//! no reduction ever spans two rows: each projection runs one GEMV per row,
//! attention reads each row's own paged view, the DeltaNet kernels each
//! row's own state, and the fused MoE and norm kernels compute every token
//! on its own. So a row's bits never depend on its batch. A span (one
//! request's prefill) runs one GEMM per projection and, from 32 rows, the
//! grouped expert kernels when the geometry allows them; the DeltaNet
//! walks the span's rows through the same per-row kernels, so prefill and
//! decode share one recurrence.

pub(crate) mod sampling;
pub(crate) mod self_check;
#[cfg(test)]
mod tests;
pub(crate) mod tower;
pub(crate) mod trunk;

use candle_core::{DType, Device, Tensor};

use super::config::{FishDrama3AttentionConfig, FishDrama3RopeLayout};
use super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::trunk::attention::FishDrama3Rope;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

/// The activation dtype of a healthy layer.
pub(crate) const HALF: DType = DType::F16;

/// The tensors a probed forward records, by name (`self_check.rs`).
pub(crate) type Probe<'a> = Option<&'a mut Vec<(String, Tensor)>>;

/// Records `tensor` under `name()` when the forward is probed.
pub(crate) fn record(probe: &mut Probe<'_>, name: impl FnOnce() -> String, tensor: &Tensor) {
    if let Some(probe) = probe.as_deref_mut() {
        probe.push((name(), tensor.clone()));
    }
}

/// One request's rows in a device forward: rows `rows` of the forward's,
/// in request order, at positions `start..start + rows`, with its paged
/// view of the stack's domain and the append prepared for them.
pub(crate) struct MetalRow<'a> {
    pub(crate) start: usize,
    pub(crate) rows: usize,
    pub(crate) cache: &'a PhysicalPagedKvCache,
    pub(crate) prepared: &'a mut PreparedPhysicalPagedStep,
}

/// The positions of every row of `rows`, in order.
pub(crate) fn positions(rows: &[MetalRow<'_>]) -> Vec<usize> {
    rows.iter()
        .flat_map(|row| row.start..row.start + row.rows)
        .collect()
}

/// An F32 vector on `device`.
pub(crate) fn vector(values: &[f32], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(values, values.len(), device)?)
}

/// `x * rsqrt(mean(x²) + eps) * gain` over the last dim: the Qwen3.6 F16
/// kernel with its F32 gain for an F16 layer (it reads the F32 residual
/// through F16, which the range probe keeps inside), Candle's F32 RMSNorm
/// for an F32 one. Returns `act`.
pub(crate) fn rms_norm(x: &Tensor, gain: &Tensor, eps: f32, act: DType) -> Result<Tensor> {
    if act == HALF {
        let x = x.to_dtype(HALF)?.contiguous()?;
        return Ok(crate::kernels::cuda::norm::rms_norm(&x, gain, eps)?);
    }
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    Ok(candle_nn::ops::rms_norm(&x, gain, eps)?)
}

/// `silu(gate) * up` in F32, returned in `act`.
pub(crate) fn swiglu(gate: &Tensor, up: &Tensor, act: DType) -> Result<Tensor> {
    let gate = gate.to_dtype(DType::F32)?;
    let up = up.to_dtype(DType::F32)?;
    Ok((candle_nn::ops::silu(&gate)? * up)?.to_dtype(act)?)
}

/// Rotates the first `rotary_dims` of every head of `x` (`[n, heads,
/// head_dim]`, row `i` at `positions[i]`) in `rope`'s layout, in F32 from
/// the family's host tables (F64 angles rounded once to F32): never device
/// trigonometry, whose fast-math error grows with the position. Returns
/// `x`'s dtype.
pub(crate) fn rope(x: &Tensor, rope: &FishDrama3Rope, positions: &[usize]) -> Result<Tensor> {
    let (n, _, d) = x.dims3()?;
    let rotary = rope.rotary_dims();
    if positions.len() != n || rotary == 0 || rotary > d || !rotary.is_multiple_of(2) {
        return Err(Error::InferenceError(format!(
            "Fish Audio Drama 3 RoPE over {:?} at {} positions, {rotary} rotary dims",
            x.dims(),
            positions.len()
        )));
    }
    let half = rotary / 2;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for &position in positions {
        let (c, s) = rope.cos_sin(position);
        cos.extend(c);
        sin.extend(s);
    }
    let device = x.device();
    let cos = Tensor::from_vec(cos, (n, half), device)?;
    let sin = Tensor::from_vec(sin, (n, half), device)?;
    let xf = x.to_dtype(DType::F32)?;
    let part = xf.narrow(2, 0, rotary)?;
    let rotated = match rope.layout() {
        // `[1, t, h, d]` against `[t, d / 2]`.
        FishDrama3RopeLayout::HalfSplit => {
            candle_nn::rotary_emb::rope_thd(&part.unsqueeze(0)?.contiguous()?, &cos, &sin)?
                .squeeze(0)?
        }
        // `[1, h, t, d]` against `[t, d / 2]`.
        FishDrama3RopeLayout::InterleavedPairs => candle_nn::rotary_emb::rope_i(
            &part.transpose(0, 1)?.unsqueeze(0)?.contiguous()?,
            &cos,
            &sin,
        )?
        .squeeze(0)?
        .transpose(0, 1)?,
    };
    let out = if rotary < d {
        Tensor::cat(&[&rotated, &xf.narrow(2, rotary, d - rotary)?], 2)?
    } else {
        rotated
    };
    Ok(out.to_dtype(x.dtype())?)
}

/// Gated attention on the device, the trunk's and the tower's alike:
///
/// ```text
/// [q_h | gate_h], K, V = qkv(x)              # one F16 matrix, q and gate interleaved per head
/// q, k = (1 + w)-RMSNorm per head; RoPE      # F16 kernel, then F32 host-table rotation
/// o    = paged GQA(q, k, v) per request      # K/V stored in the domain's dtype
/// out  = o_proj(o * sigmoid(gate))           # the F16 gate kernel
/// ```
pub(crate) struct MetalAttention {
    /// `[2 q_width + 2 kv_width, hidden]`: per head its query rows then its
    /// gate rows, then K, then V.
    qkv: FishDrama3Linear,
    o_proj: FishDrama3Linear,
    q_gain: Tensor,
    k_gain: Tensor,
    config: FishDrama3AttentionConfig,
    rope: FishDrama3Rope,
    eps: f32,
}

impl MetalAttention {
    /// From exact F32 projection values: `query_gate` `[2 q_width, hidden]`
    /// interleaved per head (`[q_h | gate_h]`, the trunk's layout), `keys`
    /// and `values` `[kv_width, hidden]`, `out` `[hidden, q_width]`, and the
    /// q/k norms' `(1 + w)` gains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        query_gate: Vec<f32>,
        keys: &[f32],
        values: &[f32],
        out: Vec<f32>,
        q_gain: &[f32],
        k_gain: &[f32],
        config: FishDrama3AttentionConfig,
        eps: f32,
        device: &Device,
    ) -> Result<Self> {
        let (qw, kvw) = (config.q_width(), config.kv_width());
        if query_gate.is_empty()
            || !query_gate.len().is_multiple_of(2 * qw)
            || keys.len() * 2 * qw != query_gate.len() * kvw
            || values.len() != keys.len()
            || q_gain.len() != config.head_dim
            || k_gain.len() != config.head_dim
        {
            return Err(Error::ModelLoadError(
                "Fish Audio Drama 3 Metal attention weights do not match the geometry".into(),
            ));
        }
        let hidden = query_gate.len() / (2 * qw);
        let mut packed = query_gate;
        packed.extend_from_slice(keys);
        packed.extend_from_slice(values);
        let rows = 2 * qw + 2 * kvw;
        Ok(Self {
            qkv: FishDrama3Linear::from_values(
                packed,
                rows,
                hidden,
                super::residency::FishDrama3WeightForm::F16,
                device,
            )?,
            o_proj: FishDrama3Linear::from_values(
                out,
                hidden,
                qw,
                super::residency::FishDrama3WeightForm::F16,
                device,
            )?,
            q_gain: vector(q_gain, device)?,
            k_gain: vector(k_gain, device)?,
            rope: FishDrama3Rope::new(&config),
            config,
            eps,
        })
    }

    /// `query`, `gate` `[q_width, hidden]` contiguous blocks (the tower's
    /// `wqkv` layout) interleaved per head into the trunk's `[q_h | gate_h]`.
    pub(crate) fn interleave(
        query: &[f32],
        gate: &[f32],
        head_dim: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let block = head_dim * hidden;
        query
            .chunks_exact(block)
            .zip(gate.chunks_exact(block))
            .flat_map(|(q, g)| q.iter().chain(g).copied())
            .collect()
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.qkv.resident_bytes() + self.o_proj.resident_bytes() + 2 * self.config.head_dim * 4
    }

    /// The mixer output `[N, hidden]` in `act` of the normed rows `x`
    /// (`[N, hidden]`, `rows` in order); each request's K/V go into layer
    /// `slot` of its view through its prepared append.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        rows: &mut [MetalRow<'_>],
        slot: usize,
        phase: FishDrama3KernelPhase,
        act: DType,
        probe: &mut Probe<'_>,
    ) -> Result<Tensor> {
        let a = &self.config;
        let (heads, kv_heads, d) = (a.heads, a.kv_heads, a.head_dim);
        let (qw, kvw) = (a.q_width(), a.kv_width());
        let n = x.dim(0)?;
        let packed = self.qkv.forward_device(x, phase, act)?;
        record(probe, || "qkv".into(), &packed);
        let query_gate = packed
            .narrow(1, 0, 2 * qw)?
            .contiguous()?
            .reshape((n, heads, 2 * d))?;
        let queries = query_gate.narrow(2, 0, d)?.contiguous()?;
        let keys = packed.narrow(1, 2 * qw, kvw)?.contiguous()?;
        let values = packed
            .narrow(1, 2 * qw + kvw, kvw)?
            .contiguous()?
            .reshape((n, kv_heads, d))?;
        let queries = rms_norm(
            &queries.reshape((n * heads, d))?,
            &self.q_gain,
            self.eps,
            act,
        )?
        .reshape((n, heads, d))?;
        let keys = rms_norm(
            &keys.reshape((n * kv_heads, d))?,
            &self.k_gain,
            self.eps,
            act,
        )?
        .reshape((n, kv_heads, d))?;
        let at = positions(rows);
        let queries = rope(&queries, &self.rope, &at)?;
        let keys = rope(&keys, &self.rope, &at)?;
        record(probe, || "keys".into(), &keys);
        record(probe, || "values".into(), &values);
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let mut attended = Vec::with_capacity(rows.len());
        let mut offset = 0;
        for row in rows.iter_mut() {
            let kv = row.cache.arena().config().dtype;
            let take = |t: &Tensor| -> Result<Tensor> {
                Ok(t.narrow(0, offset, row.rows)?.to_dtype(kv)?.contiguous()?)
            };
            let out = row.cache.write_and_attend(
                slot,
                row.prepared,
                &take(&queries)?,
                &take(&keys)?,
                &take(&values)?,
                scale,
            )?;
            attended.push(out.to_dtype(act)?.reshape((row.rows, qw))?);
            offset += row.rows;
        }
        if offset != n {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 attention got {n} rows for requests of {offset}"
            )));
        }
        let attended = Tensor::cat(&attended, 0)?;
        let gated = crate::kernels::cuda::gate::attn_gate(&attended, &query_gate, heads, d)?;
        record(probe, || "gated".into(), &gated);
        let out = self.o_proj.forward_device(&gated, phase, act)?;
        record(probe, || "attention_out".into(), &out);
        Ok(out)
    }
}

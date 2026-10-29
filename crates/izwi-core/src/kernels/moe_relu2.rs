//! Device-routed sparse experts for sigmoid-routed relu² MoE layers with one
//! FP8 scale per expert projection (Nemotron-H as ModelOpt quantizes it).
//!
//! A layer runs as a handful of launches with the expert ids on the device,
//! so a decode step never waits for the host inside the MoE block:
//!
//! 1. [`router_logits`]: `x @ gate.T` in F32, one warp per (token, expert).
//! 2. [`route`]: `s = sigmoid(logit)`; the top `k` of `s + bias` (F32, ties
//!    to the lower expert) are picked; their weights are `s / sum(s) *
//!    scale`. The routing tensor is F32 `[tokens, 2 * k]`: `k` expert ids
//!    (exact small integers), then `k` weights.
//! 3. [`experts`]: `y[t] = sum_k w_k * down_e(relu(up_e x[t])^2)`. Decode
//!    steps and short spans run warp kernels, per routed (token, slot) pair
//!    or expert-major; the two sum every dot product in the same order, so
//!    they agree bit for bit and a row's result does not depend on how many
//!    rows share the step. Long spans (prefill; see [`uses_gemm`]) read the
//!    routing once and run each touched expert as a dense GEMM over its
//!    tokens on that expert's transiently dequantized weights, which the warp
//!    kernels cannot match once each expert has many tokens.
//!
//! Experts are stacked per layer: `up` `[experts, inter, hidden]` and `down`
//! `[experts, hidden, inter]` E4M3FN bytes with one F32 scale per expert and
//! projection. Widths need only be multiples of 16 (one 16-byte load per
//! lane); Nemotron-H's 2688 and 1856 fit without padding. Activations are
//! F32 throughout.
//!
//! The qwen36moe kernels (`kernels::cuda::moe`, `kernels::metal_qwen36moe`)
//! hard-code softmax routing, SwiGLU gate+up experts and 128x128 block
//! scales; this module is their sibling for the other conventions, not a
//! mode of theirs, so neither changes the other's arithmetic. The CPU
//! functions are the reference the device tests and load checks compare
//! against, not a performance path.

use candle_core::{DType, Device, Result, Tensor};

use super::cuda::fp8::decode_e4m3fn;

/// Most routed experts the router kernel holds (16 per lane, 32 lanes).
pub const MAX_EXPERTS: usize = 512;
/// Most experts a token may route to.
pub const MAX_TOP_K: usize = 32;
/// Routed pairs from which the expert-major kernels replace the per-pair
/// ones: below this, a pair's own weight reads cost less than grouping.
pub const GROUPED_MIN_PAIRS: usize = 32;
/// Fewest tokens a call runs as dequantized GEMMs: above every decode width
/// (CUDA's largest is 32 rows), so decode steps always take the warp kernels.
pub const GEMM_MIN_TOKENS: usize = 64;
/// Routed pairs per expert from which the dequantized GEMMs beat the warp
/// kernels: the GEMM form pays one dequantization per touched expert, the
/// warp kernels a pass over the weights per pair. Measured on an M-series
/// GPU in the published geometry (64 experts: the warp kernels win at 128
/// tokens, about 9 pairs per expert, the GEMMs from 256, about 18).
pub const GEMM_MIN_PAIRS_PER_EXPERT: usize = 16;

/// Whether [`experts`] runs a call of `tokens` tokens as dequantized GEMMs.
pub fn uses_gemm(tokens: usize, top_k: usize, experts: usize) -> bool {
    tokens >= GEMM_MIN_TOKENS && tokens * top_k >= GEMM_MIN_PAIRS_PER_EXPERT * experts
}
/// Values one warp covers per sweep: 16 bytes per lane, 32 lanes.
const SWEEP: usize = 512;
/// Sweeps the grouped kernels hold in registers: hidden <= 4096 for the up
/// projection, expert width <= 2048 for the down projection.
const MAX_HIDDEN_SWEEPS: usize = 8;
const MAX_INTER_SWEEPS: usize = 4;

/// The router of one MoE layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidRouting {
    pub experts: usize,
    pub top_k: usize,
    /// Applied to the renormalized weights (Nemotron-H: 2.5).
    pub scale: f32,
}

impl SigmoidRouting {
    fn validate(&self) -> Result<()> {
        if self.experts == 0
            || self.experts > MAX_EXPERTS
            || self.top_k == 0
            || self.top_k > self.experts
            || self.top_k > MAX_TOP_K
            || !self.scale.is_finite()
        {
            candle_core::bail!("invalid sigmoid MoE routing {self:?}")
        }
        Ok(())
    }
}

/// One layer's routed experts, stacked.
#[derive(Debug, Clone)]
pub struct StackedExperts {
    up: Tensor,
    up_scale: Tensor,
    down: Tensor,
    down_scale: Tensor,
    experts: usize,
    hidden: usize,
    inter: usize,
}

impl StackedExperts {
    /// `up` `[experts, inter, hidden]` and `down` `[experts, hidden, inter]`
    /// U8 (E4M3FN bytes), scales F32 `[experts]`, all on one device.
    pub fn new(up: Tensor, up_scale: Tensor, down: Tensor, down_scale: Tensor) -> Result<Self> {
        let (experts, inter, hidden) = up.dims3()?;
        if up.dtype() != DType::U8
            || down.dtype() != DType::U8
            || down.dims() != [experts, hidden, inter]
            || up_scale.dtype() != DType::F32
            || down_scale.dtype() != DType::F32
            || up_scale.dims() != [experts]
            || down_scale.dims() != [experts]
            || experts == 0
            || experts > MAX_EXPERTS
            || hidden == 0
            || inter == 0
            || !hidden.is_multiple_of(16)
            || !inter.is_multiple_of(16)
        {
            candle_core::bail!(
                "invalid stacked relu² experts: up {:?} {:?}, down {:?} {:?}, scales {:?} {:?}",
                up.dims(),
                up.dtype(),
                down.dims(),
                down.dtype(),
                up_scale.dims(),
                down_scale.dims()
            )
        }
        Ok(Self {
            up: up.contiguous()?,
            up_scale: up_scale.contiguous()?,
            down: down.contiguous()?,
            down_scale: down_scale.contiguous()?,
            experts,
            hidden,
            inter,
        })
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn inter(&self) -> usize {
        self.inter
    }

    pub fn device(&self) -> &Device {
        self.up.device()
    }

    /// Resident bytes: one byte per weight plus the scales.
    pub fn resident_bytes(&self) -> usize {
        2 * self.experts * self.hidden * self.inter + 8 * self.experts
    }
}

/// Whether the device kernels serve this device and geometry. The CPU runs
/// only the reference.
pub fn supported(device: &Device, experts: usize, hidden: usize, inter: usize) -> bool {
    let geometry = experts > 0
        && experts <= MAX_EXPERTS
        && hidden > 0
        && inter > 0
        && hidden.is_multiple_of(16)
        && inter.is_multiple_of(16)
        && hidden.div_ceil(SWEEP) <= MAX_HIDDEN_SWEEPS
        && inter.div_ceil(SWEEP) <= MAX_INTER_SWEEPS;
    geometry
        && match device {
            Device::Cpu => false,
            Device::Cuda(_) => cfg!(feature = "cuda"),
            Device::Metal(_) => cfg!(feature = "metal"),
        }
}

/// Router logits: `x` `[tokens, hidden]` F32 against `gate` `[experts,
/// hidden]` F32 → `[tokens, experts]` F32, each a sequential-per-lane dot
/// product that does not depend on the other rows.
pub fn router_logits(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts, w_hidden) = gate.dims2()?;
    if w_hidden != hidden || x.dtype() != DType::F32 || gate.dtype() != DType::F32 {
        candle_core::bail!(
            "invalid relu² MoE router: x {:?} {:?}, gate {:?} {:?}",
            x.dims(),
            x.dtype(),
            gate.dims(),
            gate.dtype()
        )
    }
    let (x, gate) = (x.contiguous()?, gate.contiguous()?);
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        return metal_impl::router_logits(&x, &gate, tokens, experts, hidden);
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::router_logits(&x, &gate, tokens, experts, hidden);
    }
    let _ = (tokens, experts);
    x.matmul(&gate.t()?)
}

/// Routing `[tokens, 2 * top_k]` F32 from `logits` `[tokens, experts]` F32
/// and the selection `bias` `[experts]` F32, on the logits' device.
pub fn route(logits: &Tensor, bias: &Tensor, spec: SigmoidRouting) -> Result<Tensor> {
    spec.validate()?;
    let (tokens, experts) = logits.dims2()?;
    if experts != spec.experts
        || logits.dtype() != DType::F32
        || bias.dims() != [experts]
        || bias.dtype() != DType::F32
    {
        candle_core::bail!(
            "invalid relu² MoE routing input: logits {:?}, bias {:?}, {spec:?}",
            logits.dims(),
            bias.dims()
        )
    }
    let (logits, bias) = (logits.contiguous()?, bias.contiguous()?);
    #[cfg(feature = "metal")]
    if logits.device().is_metal() {
        return metal_impl::route(&logits, &bias, tokens, spec);
    }
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        return cuda_impl::route(&logits, &bias, tokens, spec);
    }
    let routing = route_reference(
        &logits.flatten_all()?.to_vec1::<f32>()?,
        &bias.to_vec1::<f32>()?,
        spec,
    );
    Tensor::from_vec(routing, (tokens, 2 * spec.top_k), logits.device())
}

/// The routed experts: `x` `[tokens, hidden]` F32, `routing` from [`route`]
/// → `[tokens, hidden]` F32.
pub fn experts(
    x: &Tensor,
    routing: &Tensor,
    top_k: usize,
    weights: &StackedExperts,
) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    if hidden != weights.hidden
        || x.dtype() != DType::F32
        || routing.dims() != [tokens, 2 * top_k]
        || routing.dtype() != DType::F32
        || top_k == 0
        || top_k > MAX_TOP_K
    {
        candle_core::bail!(
            "invalid relu² MoE experts input: x {:?} {:?}, routing {:?}, top_k {top_k}, hidden {}",
            x.dims(),
            x.dtype(),
            routing.dims(),
            weights.hidden
        )
    }
    let (x, routing) = (x.contiguous()?, routing.contiguous()?);
    if uses_gemm(tokens, top_k, weights.experts) && !x.device().is_cpu() {
        return experts_by_gemm(&x, &routing, top_k, weights);
    }
    let grouped = tokens * top_k >= GROUPED_MIN_PAIRS;
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        return metal_impl::experts(&x, &routing, top_k, weights, grouped);
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::experts(&x, &routing, top_k, weights, grouped);
    }
    let _ = grouped;
    let y = experts_reference(
        &x.flatten_all()?.to_vec1::<f32>()?,
        &routing.flatten_all()?.to_vec1::<f32>()?,
        top_k,
        &weights.up.flatten_all()?.to_vec1::<u8>()?,
        &weights.up_scale.to_vec1::<f32>()?,
        &weights.down.flatten_all()?.to_vec1::<u8>()?,
        &weights.down_scale.to_vec1::<f32>()?,
        weights.hidden,
        weights.inter,
    );
    Tensor::from_vec(y, (tokens, hidden), x.device())
}

/// Expert `expert` of a stacked projection (`[experts, rows, cols]` E4M3FN
/// bytes, one scale per expert) as a dense `[rows, cols]` matrix:
/// `decode(byte) * scale` in F32, rounded once to `dtype` (F32, or BF16 on
/// CUDA).
pub fn dequantize_expert(
    stacked: &Tensor,
    scales: &Tensor,
    expert: usize,
    dtype: DType,
) -> Result<Tensor> {
    let (experts, rows, cols) = stacked.dims3()?;
    if stacked.dtype() != DType::U8
        || scales.dims() != [experts]
        || scales.dtype() != DType::F32
        || expert >= experts
    {
        candle_core::bail!(
            "invalid stacked FP8 dequantization: {:?} {:?}, scales {:?}, expert {expert}",
            stacked.dims(),
            stacked.dtype(),
            scales.dims()
        )
    }
    #[cfg(feature = "metal")]
    if stacked.device().is_metal() {
        if dtype != DType::F32 {
            candle_core::bail!("Metal dequantizes stacked experts to F32")
        }
        return metal_impl::dequantize(stacked, scales, expert, rows, cols);
    }
    #[cfg(feature = "cuda")]
    if stacked.device().is_cuda() {
        return cuda_impl::dequantize(stacked, scales, expert, rows, cols, dtype);
    }
    let scale = scales.to_vec1::<f32>()?[expert];
    let values = stacked
        .narrow(0, expert, 1)?
        .flatten_all()?
        .to_vec1::<u8>()?
        .into_iter()
        .map(|b| decode_e4m3fn(b) * scale)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (rows, cols), stacked.device())?.to_dtype(dtype)
}

/// The long-span form of [`experts`]: the routing read once, then per
/// touched expert `down(relu(x_e @ up.T)^2) * w` as dense GEMMs on the
/// expert's transiently dequantized weights (F32 on Metal, BF16 on CUDA,
/// accumulated in F32), scattered back by token.
fn experts_by_gemm(
    x: &Tensor,
    routing: &Tensor,
    top_k: usize,
    weights: &StackedExperts,
) -> Result<Tensor> {
    let device = x.device();
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let host = routing.flatten_all()?.to_vec1::<f32>()?;
    let mut assigned = vec![Vec::<(u32, f32)>::new(); weights.experts];
    for (token, row) in host.chunks_exact(2 * top_k).enumerate() {
        for slot in 0..top_k {
            let id = row[slot];
            if id >= 0.0 && (id as usize) < weights.experts {
                assigned[id as usize].push((token as u32, row[top_k + slot]));
            }
        }
    }
    let mut out = x.zeros_like()?;
    for (e, tokens) in assigned.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        let ids = Tensor::from_vec(
            tokens.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            tokens.len(),
            device,
        )?;
        let combine = Tensor::from_vec(
            tokens.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
            (tokens.len(), 1),
            device,
        )?;
        let xe = x.index_select(&ids, 0)?.to_dtype(dtype)?;
        let up = dequantize_expert(&weights.up, &weights.up_scale, e, dtype)?;
        let h = xe.matmul(&up.t()?)?.to_dtype(DType::F32)?.relu()?.sqr()?;
        drop(up);
        let down = dequantize_expert(&weights.down, &weights.down_scale, e, dtype)?;
        let y = h
            .to_dtype(dtype)?
            .matmul(&down.t()?)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&combine)?;
        out = out.index_add(&ids, &y, 0)?;
    }
    Ok(out)
}

/// The reference router (the family's host `route`, row by row).
pub fn route_reference(logits: &[f32], bias: &[f32], spec: SigmoidRouting) -> Vec<f32> {
    let experts = spec.experts;
    let k = spec.top_k;
    let mut out = Vec::with_capacity(logits.len() / experts * 2 * k);
    for row in logits.chunks_exact(experts) {
        let scores = row
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect::<Vec<f32>>();
        let biased = scores
            .iter()
            .zip(bias)
            .map(|(s, b)| s + b)
            .collect::<Vec<f32>>();
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
        let picked = &order[..k];
        let total = picked.iter().map(|&e| scores[e]).sum::<f32>();
        out.extend(picked.iter().map(|&e| e as f32));
        out.extend(picked.iter().map(|&e| scores[e] / total * spec.scale));
    }
    out
}

/// The reference experts: per token, the weighted sum over its routed
/// experts of `down_e(relu(up_e x)^2)`, F32 with sequential sums.
#[allow(clippy::too_many_arguments)]
pub fn experts_reference(
    x: &[f32],
    routing: &[f32],
    top_k: usize,
    up: &[u8],
    up_scale: &[f32],
    down: &[u8],
    down_scale: &[f32],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let tokens = x.len() / hidden;
    let experts = up_scale.len();
    let mut y = vec![0f32; tokens * hidden];
    let mut act = vec![0f32; inter];
    for token in 0..tokens {
        let xt = &x[token * hidden..(token + 1) * hidden];
        let row = &routing[token * 2 * top_k..(token + 1) * 2 * top_k];
        for slot in 0..top_k {
            let id = row[slot];
            if !(id >= 0.0 && (id as usize) < experts) {
                continue;
            }
            let e = id as usize;
            let weight = row[top_k + slot];
            for (n, a) in act.iter_mut().enumerate() {
                let w = &up[(e * inter + n) * hidden..(e * inter + n + 1) * hidden];
                let dot = w
                    .iter()
                    .zip(xt)
                    .map(|(&b, &v)| decode_e4m3fn(b) * v)
                    .sum::<f32>();
                let r = (dot * up_scale[e]).max(0.0);
                *a = r * r;
            }
            for (h, out) in y[token * hidden..(token + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                let w = &down[(e * hidden + h) * inter..(e * hidden + h + 1) * inter];
                let dot = w
                    .iter()
                    .zip(&act)
                    .map(|(&b, &v)| decode_e4m3fn(b) * v)
                    .sum::<f32>();
                *out += weight * dot * down_scale[e];
            }
        }
    }
    y
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use candle_core::op::BackpropOp;
    use candle_core::{bail, DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};
    use objc2_metal::MTLSize;

    use super::{SigmoidRouting, StackedExperts};
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// E4M3FN byte -> value * 2^-8 through an F16 lane (exact for finite values).
inline float mr2_e4m3(uint b) {
    ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return float(as_type<half>(h));
}

inline float mr2_dot4(uint w, thread const float* x) {
    float acc = mr2_e4m3(w & 0xFFu) * x[0];
    acc = fma(mr2_e4m3((w >> 8) & 0xFFu), x[1], acc);
    acc = fma(mr2_e4m3((w >> 16) & 0xFFu), x[2], acc);
    return fma(mr2_e4m3(w >> 24), x[3], acc);
}

inline float mr2_dot16(uint4 q, device const float* x) {
    float v[16];
    for (uint i = 0; i < 16; ++i) {
        v[i] = x[i];
    }
    float acc = mr2_dot4(q.x, v);
    acc += mr2_dot4(q.y, v + 4);
    acc += mr2_dot4(q.z, v + 8);
    return acc + mr2_dot4(q.w, v + 12);
}

// dims: tokens, experts, hidden. One simdgroup per (token, expert);
// grid (experts / 8, tokens).
kernel void mr2_router_logits(
    device const float* x [[buffer(0)]], device const float* gate [[buffer(1)]],
    device float* logits [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint tokens = dims[0], experts = dims[1], hidden = dims[2];
    const uint e = tg.x * 8 + sg;
    const uint t = tg.y;
    if (e >= experts || t >= tokens) {
        return;
    }
    device const float* xr = x + (ulong)t * hidden;
    device const float* gr = gate + (ulong)e * hidden;
    float acc = 0.0f;
    for (uint k = lane; k < hidden; k += 32) {
        acc = fma(xr[k], gr[k], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        logits[(ulong)t * experts + e] = acc;
    }
}

// dims: tokens, experts, top_k. One simdgroup per token, four per
// threadgroup. Selection on sigmoid(logit) + bias in F32, ties to the lower
// expert; weights sigmoid / sum(picked sigmoids) * scale.
kernel void mr2_route(
    device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint tokens = dims[0], experts = dims[1], top_k = dims[2];
    const uint token = tg * 4 + sg;
    if (token >= tokens) {
        return;
    }
    device const float* row = logits + (ulong)token * experts;
    float s[16];
    float b[16];
    for (uint i = 0; i < 16; ++i) {
        const uint e = lane + 32 * i;
        if (e < experts) {
            s[i] = 1.0f / (1.0f + precise::exp(-row[e]));
            b[i] = s[i] + bias[e];
        } else {
            s[i] = 0.0f;
            b[i] = -INFINITY;
        }
    }
    device float* dst = out + (ulong)token * 2 * top_k;
    const uint none = 0xFFFFFFFFu;
    uint taken = 0;
    float picked_sum = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        float best_s = 0.0f;
        uint best_e = none;
        for (uint i = 0; i < 16; ++i) {
            const uint e = lane + 32 * i;
            const bool open = e < experts && ((taken >> i) & 1u) == 0u;
            if (open && (best_e == none || b[i] > best || (b[i] == best && e < best_e))) {
                best = b[i];
                best_s = s[i];
                best_e = e;
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float other = simd_shuffle_xor(best, o);
            const float other_s = simd_shuffle_xor(best_s, o);
            const uint other_e = simd_shuffle_xor(best_e, o);
            if (other_e != none && (best_e == none || other > best || (other == best && other_e < best_e))) {
                best = other;
                best_s = other_s;
                best_e = other_e;
            }
        }
        for (uint i = 0; i < 16; ++i) {
            if (lane + 32 * i == best_e) {
                taken |= 1u << i;
            }
        }
        picked_sum += best_s;
        if (lane == 0) {
            dst[k] = float(best_e);
            dst[top_k + k] = best_s;
        }
    }
    if (lane == 0) {
        for (uint k = 0; k < top_k; ++k) {
            dst[top_k + k] = dst[top_k + k] / picked_sum * scale;
        }
    }
}

// One routed pair's up projection channel, the same sum in both forms.
inline float mr2_up_dot(device const uchar* row, device const float* xt, uint hidden, uint lane) {
    float acc = 0.0f;
    for (uint k0 = lane * 16; k0 < hidden; k0 += 512) {
        acc += mr2_dot16(*((device const uint4*)(row + k0)), xt + k0);
    }
    return acc;
}

// dims: slots, hidden, inter, experts. act[pair, n] = relu(scale * up[e, n] . x)^2;
// grid (pairs, inter / 8).
kernel void mr2_up(
    device const float* x [[buffer(0)]], device const float* routing [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device const float* ws [[buffer(3)]],
    device float* act [[buffer(4)]], constant uint* dims [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint slots = dims[0], hidden = dims[1], inter = dims[2], experts = dims[3];
    const uint pair = tg.x;
    const uint token = pair / slots;
    const uint slot = pair - token * slots;
    const uint n = tg.y * 8 + sg;
    if (n >= inter) {
        return;
    }
    const float id = routing[(ulong)token * 2 * slots + slot];
    if (!(id >= 0.0f) || uint(id) >= experts) {
        if (lane == 0) {
            act[(ulong)pair * inter + n] = 0.0f;
        }
        return;
    }
    const uint e = uint(id);
    device const uchar* row = w + ((ulong)e * inter + n) * hidden;
    const float v = simd_sum(mr2_up_dot(row, x + (ulong)token * hidden, hidden, lane)) * 256.0f * ws[e];
    if (lane == 0) {
        const float r = max(v, 0.0f);
        act[(ulong)pair * inter + n] = r * r;
    }
}

// dims: slots, hidden, inter, experts. y[t, h] = sum_s weight_s * (scale *
// down[e_s, h] . act[pair_s]); grid (tokens, hidden / 8).
kernel void mr2_down(
    device const float* act [[buffer(0)]], device const float* routing [[buffer(1)]],
    device const uchar* w [[buffer(2)]], device const float* ws [[buffer(3)]],
    device float* y [[buffer(4)]], constant uint* dims [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint slots = dims[0], hidden = dims[1], inter = dims[2], experts = dims[3];
    const uint token = tg.x;
    const uint h = tg.y * 8 + sg;
    if (h >= hidden) {
        return;
    }
    float total = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        const float id = routing[(ulong)token * 2 * slots + s];
        if (!(id >= 0.0f) || uint(id) >= experts) {
            continue;
        }
        const uint e = uint(id);
        device const uchar* row = w + ((ulong)e * hidden + h) * inter;
        device const float* a = act + ((ulong)token * slots + s) * inter;
        float acc = 0.0f;
        for (uint k0 = lane * 16; k0 < inter; k0 += 512) {
            acc += mr2_dot16(*((device const uint4*)(row + k0)), a + k0);
        }
        const float p = simd_sum(acc) * 256.0f * ws[e];
        total = fma(routing[(ulong)token * 2 * slots + slots + s], p, total);
    }
    if (lane == 0) {
        y[(ulong)token * hidden + h] = total;
    }
}

// Pairs grouped by expert: offsets[e]..offsets[e + 1] index sorted_pairs.
kernel void mr2_group_pairs(
    device const float* routing [[buffer(0)]], device int* offsets [[buffer(1)]],
    device int* sorted_pairs [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    const uint tokens = dims[0], slots = dims[1], experts = dims[2];
    threadgroup atomic_int counts[512];
    threadgroup atomic_int fill[512];
    threadgroup int starts[512];
    for (uint e = tid; e < experts; e += threads) {
        atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
        atomic_store_explicit(&fill[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pairs = tokens * slots;
    for (uint p = tid; p < pairs; p += threads) {
        const uint token = p / slots;
        const float id = routing[(ulong)token * 2 * slots + (p - token * slots)];
        if (id >= 0.0f && uint(id) < experts) {
            atomic_fetch_add_explicit(&counts[uint(id)], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int running = 0;
        for (uint e = 0; e < experts; ++e) {
            starts[e] = running;
            offsets[e] = running;
            running += atomic_load_explicit(&counts[e], memory_order_relaxed);
        }
        offsets[experts] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < pairs; p += threads) {
        const uint token = p / slots;
        const float id = routing[(ulong)token * 2 * slots + (p - token * slots)];
        if (id >= 0.0f && uint(id) < experts) {
            const uint e = uint(id);
            const int index = atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
            sorted_pairs[starts[e] + index] = int(p);
        }
    }
}

// dims: slots, hidden, inter. One expert's up channel for all its pairs, the
// row held in registers (hidden <= 4096); grid (experts, inter / 8).
kernel void mr2_up_grouped(
    device const float* x [[buffer(0)]], device const int* offsets [[buffer(1)]],
    device const int* sorted_pairs [[buffer(2)]], device const uchar* w [[buffer(3)]],
    device const float* ws [[buffer(4)]], device float* act [[buffer(5)]],
    constant uint* dims [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint slots = dims[0], hidden = dims[1], inter = dims[2];
    const uint e = tg.x;
    const int begin = offsets[e];
    const int end = offsets[e + 1];
    const uint n = tg.y * 8 + sg;
    if (begin == end || n >= inter) {
        return;
    }
    device const uchar* row = w + ((ulong)e * inter + n) * hidden;
    const uint sweeps = (hidden + 511) / 512;
    uint4 q[8];
    for (uint c = 0; c < sweeps; ++c) {
        const uint k0 = lane * 16 + c * 512;
        q[c] = k0 < hidden ? *((device const uint4*)(row + k0)) : uint4(0);
    }
    const float scale = ws[e];
    for (int i = begin; i < end; ++i) {
        const uint pair = uint(sorted_pairs[i]);
        device const float* xt = x + (ulong)(pair / slots) * hidden;
        float acc = 0.0f;
        for (uint c = 0; c < sweeps; ++c) {
            const uint k0 = lane * 16 + c * 512;
            if (k0 < hidden) {
                acc += mr2_dot16(q[c], xt + k0);
            }
        }
        const float v = simd_sum(acc) * 256.0f * scale;
        if (lane == 0) {
            const float r = max(v, 0.0f);
            act[(ulong)pair * inter + n] = r * r;
        }
    }
}

// dims: hidden, inter. One expert's down channel for all its pairs, the row
// held in registers (inter <= 2048); partial[pair, h] = scale * dot.
kernel void mr2_down_grouped(
    device const float* act [[buffer(0)]], device const int* offsets [[buffer(1)]],
    device const int* sorted_pairs [[buffer(2)]], device const uchar* w [[buffer(3)]],
    device const float* ws [[buffer(4)]], device float* partial [[buffer(5)]],
    constant uint* dims [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint hidden = dims[0], inter = dims[1];
    const uint e = tg.x;
    const int begin = offsets[e];
    const int end = offsets[e + 1];
    const uint h = tg.y * 8 + sg;
    if (begin == end || h >= hidden) {
        return;
    }
    device const uchar* row = w + ((ulong)e * hidden + h) * inter;
    const uint sweeps = (inter + 511) / 512;
    uint4 q[4];
    for (uint c = 0; c < sweeps; ++c) {
        const uint k0 = lane * 16 + c * 512;
        q[c] = k0 < inter ? *((device const uint4*)(row + k0)) : uint4(0);
    }
    const float scale = ws[e];
    for (int i = begin; i < end; ++i) {
        const uint pair = uint(sorted_pairs[i]);
        device const float* a = act + (ulong)pair * inter;
        float acc = 0.0f;
        for (uint c = 0; c < sweeps; ++c) {
            const uint k0 = lane * 16 + c * 512;
            if (k0 < inter) {
                acc += mr2_dot16(q[c], a + k0);
            }
        }
        const float p = simd_sum(acc) * 256.0f * scale;
        if (lane == 0) {
            partial[(ulong)pair * hidden + h] = p;
        }
    }
}

// dims: expert, rows * cols. out[i] = decode(w[expert][i]) * scale[expert].
kernel void mr2_dequant(
    device const uchar* w [[buffer(0)]], device const float* ws [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint expert = dims[0], count = dims[1];
    if (gid >= count) {
        return;
    }
    out[gid] = mr2_e4m3(w[(ulong)expert * count + gid]) * 256.0f * ws[expert];
}

// dims: slots, hidden, experts. y[t, h] = sum_s weight_s * partial[pair_s, h]
// in slot order, as mr2_down sums.
kernel void mr2_combine(
    device const float* partial [[buffer(0)]], device const float* routing [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint slots = dims[0], hidden = dims[1], experts = dims[2];
    const uint h = gid.x;
    const uint token = gid.y;
    if (h >= hidden) {
        return;
    }
    device const float* row = routing + (ulong)token * 2 * slots;
    float total = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        const float id = row[s];
        if (!(id >= 0.0f) || uint(id) >= experts) {
            continue;
        }
        total = fma(row[slots + s], partial[((ulong)token * slots + s) * hidden + h], total);
    }
    y[(ulong)token * hidden + h] = total;
}
"#;

    fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
            OnceLock::new();
        let key = (device.registry_id(), name);
        let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(pipeline) = pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .get(&key)
            .cloned()
        {
            return Ok(pipeline);
        }
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let function = library
            .get_function(name, None)
            .map_err(candle_core::Error::wrap)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn grid(width: usize, height: usize) -> MTLSize {
        MTLSize {
            width,
            height,
            depth: 1,
        }
    }

    fn dims<const N: usize>(values: [usize; N]) -> Result<[u32; N]> {
        let mut out = [0u32; N];
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = u32::try_from(value).map_err(|_| {
                candle_core::Error::Msg(format!("relu² MoE dimension {value} exceeds u32"))
            })?;
        }
        Ok(out)
    }

    /// A contiguous Metal tensor's buffer and byte offset.
    struct Bound<'a> {
        guard: std::sync::RwLockReadGuard<'a, Storage>,
        offset: usize,
    }

    impl Bound<'_> {
        fn buffer(&self) -> Result<&Buffer> {
            match &*self.guard {
                Storage::Metal(storage) => Ok(storage.buffer()),
                _ => bail!("relu² MoE Metal kernels need Metal tensors"),
            }
        }
    }

    fn bind<'a>(tensor: &'a Tensor, name: &str, align: usize) -> Result<Bound<'a>> {
        let (guard, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            bail!("relu² MoE Metal input {name} must be contiguous")
        }
        let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
        if !offset.is_multiple_of(align) {
            bail!("relu² MoE Metal input {name} must start on a {align}-byte boundary")
        }
        Ok(Bound { guard, offset })
    }

    fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
        match tensor.device() {
            candle_core::Device::Metal(device) => Ok(device.clone()),
            _ => bail!("relu² MoE Metal kernels need Metal tensors"),
        }
    }

    fn wrap(device: &candle_core::MetalDevice, buffer: Arc<Buffer>, shape: Shape) -> Tensor {
        let count = shape.elem_count();
        Tensor::from_storage(
            Storage::Metal(MetalStorage::new(buffer, device.clone(), count, DType::F32)),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    pub(super) fn dequantize(
        stacked: &Tensor,
        scales: &Tensor,
        expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let device = metal_device(stacked)?;
        let (stacked, scales) = (stacked.contiguous()?, scales.contiguous()?);
        let wb = bind(&stacked, "stacked weights", 1)?;
        let sb = bind(&scales, "scales", 4)?;
        let count = rows * cols;
        let out = device.new_buffer(count, DType::F32, "mr2-dequant")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mr2-dequant");
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_dequant")?);
        encoder.set_input_buffer(0, Some(wb.buffer()?), wb.offset);
        encoder.set_input_buffer(1, Some(sb.buffer()?), sb.offset);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &dims([expert, count])?);
        encoder.dispatch_threads(grid(count, 1), grid(256, 1));
        drop(encoder);
        drop((wb, sb));
        Ok(wrap(&device, out, Shape::from((rows, cols))))
    }

    pub(super) fn router_logits(
        x: &Tensor,
        gate: &Tensor,
        tokens: usize,
        experts: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let device = metal_device(x)?;
        let xb = bind(x, "x", 4)?;
        let gb = bind(gate, "gate", 4)?;
        let out = device.new_buffer(tokens * experts, DType::F32, "mr2-router-logits")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mr2-router-logits");
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_router_logits")?);
        encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
        encoder.set_input_buffer(1, Some(gb.buffer()?), gb.offset);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &dims([tokens, experts, hidden])?);
        encoder.dispatch_thread_groups(grid(experts.div_ceil(8), tokens), grid(256, 1));
        drop(encoder);
        drop((xb, gb));
        Ok(wrap(&device, out, Shape::from((tokens, experts))))
    }

    pub(super) fn route(
        logits: &Tensor,
        bias: &Tensor,
        tokens: usize,
        spec: SigmoidRouting,
    ) -> Result<Tensor> {
        let device = metal_device(logits)?;
        let lb = bind(logits, "logits", 4)?;
        let bb = bind(bias, "bias", 4)?;
        let out = device.new_buffer(tokens * 2 * spec.top_k, DType::F32, "mr2-routing")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mr2-route");
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_route")?);
        encoder.set_input_buffer(0, Some(lb.buffer()?), lb.offset);
        encoder.set_input_buffer(1, Some(bb.buffer()?), bb.offset);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &dims([tokens, spec.experts, spec.top_k])?);
        encoder.set_bytes(4, &spec.scale);
        encoder.dispatch_thread_groups(grid(tokens.div_ceil(4), 1), grid(128, 1));
        drop(encoder);
        drop((lb, bb));
        Ok(wrap(&device, out, Shape::from((tokens, 2 * spec.top_k))))
    }

    pub(super) fn experts(
        x: &Tensor,
        routing: &Tensor,
        slots: usize,
        weights: &StackedExperts,
        grouped: bool,
    ) -> Result<Tensor> {
        let device = metal_device(x)?;
        let tokens = x.dim(0)?;
        let pairs = tokens * slots;
        let (experts, hidden, inter) = (weights.experts, weights.hidden, weights.inter);
        let xb = bind(x, "x", 4)?;
        let rb = bind(routing, "routing", 4)?;
        let ub = bind(&weights.up, "up", 16)?;
        let usb = bind(&weights.up_scale, "up scales", 4)?;
        let db = bind(&weights.down, "down", 16)?;
        let dsb = bind(&weights.down_scale, "down scales", 4)?;
        let act = device.new_buffer(pairs * inter, DType::F32, "mr2-act")?;
        let out = device.new_buffer(tokens * hidden, DType::F32, "mr2-out")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mr2-experts");
        if grouped {
            let offsets = device.new_buffer(experts + 1, DType::U32, "mr2-offsets")?;
            let sorted = device.new_buffer(pairs, DType::U32, "mr2-sorted")?;
            let partial = device.new_buffer(pairs * hidden, DType::F32, "mr2-partial")?;
            encoder
                .set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_group_pairs")?);
            encoder.set_input_buffer(0, Some(rb.buffer()?), rb.offset);
            encoder.set_output_buffer(1, Some(&offsets), 0);
            encoder.set_output_buffer(2, Some(&sorted), 0);
            encoder.set_bytes(3, &dims([tokens, slots, experts])?);
            encoder.dispatch_thread_groups(grid(1, 1), grid(1024, 1));
            encoder.insert_memory_barrier();
            encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_up_grouped")?);
            encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
            encoder.set_input_buffer(1, Some(&offsets), 0);
            encoder.set_input_buffer(2, Some(&sorted), 0);
            encoder.set_input_buffer(3, Some(ub.buffer()?), ub.offset);
            encoder.set_input_buffer(4, Some(usb.buffer()?), usb.offset);
            encoder.set_output_buffer(5, Some(&act), 0);
            encoder.set_bytes(6, &dims([slots, hidden, inter])?);
            encoder.dispatch_thread_groups(grid(experts, inter.div_ceil(8)), grid(256, 1));
            encoder.insert_memory_barrier();
            encoder
                .set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_down_grouped")?);
            encoder.set_input_buffer(0, Some(&act), 0);
            encoder.set_input_buffer(1, Some(&offsets), 0);
            encoder.set_input_buffer(2, Some(&sorted), 0);
            encoder.set_input_buffer(3, Some(db.buffer()?), db.offset);
            encoder.set_input_buffer(4, Some(dsb.buffer()?), dsb.offset);
            encoder.set_output_buffer(5, Some(&partial), 0);
            encoder.set_bytes(6, &dims([hidden, inter])?);
            encoder.dispatch_thread_groups(grid(experts, hidden.div_ceil(8)), grid(256, 1));
            encoder.insert_memory_barrier();
            encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_combine")?);
            encoder.set_input_buffer(0, Some(&partial), 0);
            encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
            encoder.set_output_buffer(2, Some(&out), 0);
            encoder.set_bytes(3, &dims([slots, hidden, experts])?);
            encoder.dispatch_threads(grid(hidden, tokens), grid(256, 1));
        } else {
            encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_up")?);
            encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
            encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
            encoder.set_input_buffer(2, Some(ub.buffer()?), ub.offset);
            encoder.set_input_buffer(3, Some(usb.buffer()?), usb.offset);
            encoder.set_output_buffer(4, Some(&act), 0);
            encoder.set_bytes(5, &dims([slots, hidden, inter, experts])?);
            encoder.dispatch_thread_groups(grid(pairs, inter.div_ceil(8)), grid(256, 1));
            encoder.insert_memory_barrier();
            encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mr2_down")?);
            encoder.set_input_buffer(0, Some(&act), 0);
            encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
            encoder.set_input_buffer(2, Some(db.buffer()?), db.offset);
            encoder.set_input_buffer(3, Some(dsb.buffer()?), dsb.offset);
            encoder.set_output_buffer(4, Some(&out), 0);
            encoder.set_bytes(5, &dims([slots, hidden, inter, experts])?);
            encoder.dispatch_thread_groups(grid(tokens, hidden.div_ceil(8)), grid(256, 1));
        }
        drop(encoder);
        drop((xb, rb, ub, usb, db, dsb));
        Ok(wrap(&device, out, Shape::from((tokens, hidden))))
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Shape, Storage, Tensor};

    use super::{SigmoidRouting, StackedExperts};

    const MODULE: &str = "izwi_moe_relu2";

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        tensor: &Tensor,
        name: &str,
        align: usize,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("relu² MoE {name} must be a CUDA tensor")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("relu² MoE {name} must be contiguous")
        };
        if !(start * tensor.dtype().size_in_bytes()).is_multiple_of(align) {
            candle_core::bail!("relu² MoE {name} must start on a {align}-byte boundary")
        }
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn wrap(slice: CudaSlice<f32>, device: &CudaDevice, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    fn i32_arg(value: usize, name: &str) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("relu² MoE {name} {value} exceeds i32")))
    }

    fn u32_grid(value: usize, name: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("relu² MoE {name} grid {value} exceeds u32"))
        })
    }

    fn config(grid: (u32, u32), threads: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn router_logits(
        x: &Tensor,
        gate: &Tensor,
        tokens: usize,
        experts: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let (xs, _) = x.storage_and_layout();
        let (gs, _) = gate.storage_and_layout();
        let xv = view::<f32>(&xs, x, "x", 4)?;
        let gv = view::<f32>(&gs, gate, "gate", 4)?;
        // SAFETY: the kernel writes every (token, expert) logit.
        let out = unsafe { device.alloc::<f32>(tokens * experts)? };
        let function = device.get_or_load_custom_func(
            "izwi_mr2_router_logits",
            MODULE,
            super::super::cuda::cuda_ptx_moe_relu2(),
        )?;
        let mut builder = function.builder();
        builder.arg(&xv);
        builder.arg(&gv);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(tokens, "tokens")?,
            i32_arg(experts, "experts")?,
            i32_arg(hidden, "hidden")?
        );
        let launch = config(
            (
                u32_grid(experts.div_ceil(8), "experts")?,
                u32_grid(tokens, "tokens")?,
            ),
            256,
        );
        // SAFETY: argument order and types match `izwi_mr2_router_logits`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, experts))))
    }

    pub(super) fn dequantize(
        stacked: &Tensor,
        scales: &Tensor,
        expert: usize,
        rows: usize,
        cols: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let device = stacked.device().as_cuda_device()?;
        let (stacked, scales) = (stacked.contiguous()?, scales.contiguous()?);
        let (ws, _) = stacked.storage_and_layout();
        let (ss, _) = scales.storage_and_layout();
        let wv = view::<u8>(&ws, &stacked, "stacked weights", 1)?;
        let sv = view::<f32>(&ss, &scales, "scales", 4)?;
        let count = rows * cols;
        let launch = config((u32_grid(count.div_ceil(256), "dequant")?, 1), 256);
        let ptx = super::super::cuda::cuda_ptx_moe_relu2();
        let shape = Shape::from((rows, cols));
        let expert = i32_arg(expert, "expert")?;
        let count_arg = count as i64;
        match dtype {
            DType::F32 => {
                // SAFETY: the kernel writes every element.
                let out = unsafe { device.alloc::<f32>(count)? };
                let function =
                    device.get_or_load_custom_func("izwi_mr2_dequant_f32", MODULE, ptx)?;
                let mut builder = function.builder();
                builder.arg(&wv);
                builder.arg(&sv);
                builder.arg(&out);
                candle_core::builder_arg!(builder, expert, count_arg);
                // SAFETY: argument order and types match `izwi_mr2_dequant_f32`.
                unsafe { builder.launch(launch) }.w()?;
                Ok(wrap(out, device, shape))
            }
            DType::BF16 => {
                // SAFETY: the kernel writes every element.
                let out = unsafe { device.alloc::<half::bf16>(count)? };
                let function =
                    device.get_or_load_custom_func("izwi_mr2_dequant_bf16", MODULE, ptx)?;
                let mut builder = function.builder();
                builder.arg(&wv);
                builder.arg(&sv);
                builder.arg(&out);
                candle_core::builder_arg!(builder, expert, count_arg);
                // SAFETY: argument order and types match `izwi_mr2_dequant_bf16`.
                unsafe { builder.launch(launch) }.w()?;
                Ok(Tensor::from_storage(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
                    shape,
                    BackpropOp::none(),
                    false,
                ))
            }
            other => candle_core::bail!("relu² MoE dequantizes to F32 or BF16, not {other:?}"),
        }
    }

    pub(super) fn route(
        logits: &Tensor,
        bias: &Tensor,
        tokens: usize,
        spec: SigmoidRouting,
    ) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (ls, _) = logits.storage_and_layout();
        let (bs, _) = bias.storage_and_layout();
        let lv = view::<f32>(&ls, logits, "logits", 4)?;
        let bv = view::<f32>(&bs, bias, "bias", 4)?;
        // SAFETY: the kernel writes every token's ids and weights.
        let out = unsafe { device.alloc::<f32>(tokens * 2 * spec.top_k)? };
        let function = device.get_or_load_custom_func(
            "izwi_mr2_route",
            MODULE,
            super::super::cuda::cuda_ptx_moe_relu2(),
        )?;
        let mut builder = function.builder();
        builder.arg(&lv);
        builder.arg(&bv);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(tokens, "tokens")?,
            i32_arg(spec.experts, "experts")?,
            i32_arg(spec.top_k, "top_k")?,
            spec.scale
        );
        let launch = config((u32_grid(tokens.div_ceil(4), "tokens")?, 1), 128);
        // SAFETY: argument order and types match `izwi_mr2_route`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, 2 * spec.top_k))))
    }

    pub(super) fn experts(
        x: &Tensor,
        routing: &Tensor,
        slots: usize,
        weights: &StackedExperts,
        grouped: bool,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let tokens = x.dim(0)?;
        let pairs = tokens * slots;
        let (experts, hidden, inter) = (weights.experts, weights.hidden, weights.inter);
        let (xs, _) = x.storage_and_layout();
        let (rs, _) = routing.storage_and_layout();
        let (us, _) = weights.up.storage_and_layout();
        let (uss, _) = weights.up_scale.storage_and_layout();
        let (ds, _) = weights.down.storage_and_layout();
        let (dss, _) = weights.down_scale.storage_and_layout();
        let xv = view::<f32>(&xs, x, "x", 4)?;
        let rv = view::<f32>(&rs, routing, "routing", 4)?;
        let uv = view::<u8>(&us, &weights.up, "up", 16)?;
        let usv = view::<f32>(&uss, &weights.up_scale, "up scales", 4)?;
        let dv = view::<u8>(&ds, &weights.down, "down", 16)?;
        let dsv = view::<f32>(&dss, &weights.down_scale, "down scales", 4)?;
        let ptx = super::super::cuda::cuda_ptx_moe_relu2();
        let (slots_i, hidden_i, inter_i, experts_i) = (
            i32_arg(slots, "slots")?,
            i32_arg(hidden, "hidden")?,
            i32_arg(inter, "inter")?,
            i32_arg(experts, "experts")?,
        );
        // SAFETY: the up kernels write every routed pair's activation row.
        let act = unsafe { device.alloc::<f32>(pairs * inter)? };
        // SAFETY: the down or combine kernel writes every (token, channel).
        let out = unsafe { device.alloc::<f32>(tokens * hidden)? };
        if grouped {
            // SAFETY: the grouping kernel writes every offset and routed pair.
            let offsets = unsafe { device.alloc::<i32>(experts + 1)? };
            let sorted = unsafe { device.alloc::<i32>(pairs)? };
            // SAFETY: the grouped down kernel writes every routed pair's row.
            let partial = unsafe { device.alloc::<f32>(pairs * hidden)? };
            let group = device.get_or_load_custom_func("izwi_mr2_group_pairs", MODULE, ptx)?;
            let mut builder = group.builder();
            builder.arg(&rv);
            builder.arg(&offsets);
            builder.arg(&sorted);
            candle_core::builder_arg!(builder, i32_arg(tokens, "tokens")?, slots_i, experts_i);
            // SAFETY: argument order matches `izwi_mr2_group_pairs`.
            unsafe { builder.launch(config((1, 1), 1024)) }.w()?;
            let up = device.get_or_load_custom_func("izwi_mr2_up_grouped", MODULE, ptx)?;
            let mut builder = up.builder();
            builder.arg(&xv);
            builder.arg(&offsets);
            builder.arg(&sorted);
            builder.arg(&uv);
            builder.arg(&usv);
            builder.arg(&act);
            candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i);
            let launch = config(
                (
                    u32_grid(experts, "experts")?,
                    u32_grid(inter.div_ceil(8), "inter")?,
                ),
                256,
            );
            // SAFETY: argument order matches `izwi_mr2_up_grouped`.
            unsafe { builder.launch(launch) }.w()?;
            let down = device.get_or_load_custom_func("izwi_mr2_down_grouped", MODULE, ptx)?;
            let mut builder = down.builder();
            builder.arg(&act);
            builder.arg(&offsets);
            builder.arg(&sorted);
            builder.arg(&dv);
            builder.arg(&dsv);
            builder.arg(&partial);
            candle_core::builder_arg!(builder, hidden_i, inter_i);
            let launch = config(
                (
                    u32_grid(experts, "experts")?,
                    u32_grid(hidden.div_ceil(8), "hidden")?,
                ),
                256,
            );
            // SAFETY: argument order matches `izwi_mr2_down_grouped`.
            unsafe { builder.launch(launch) }.w()?;
            let combine = device.get_or_load_custom_func("izwi_mr2_combine", MODULE, ptx)?;
            let mut builder = combine.builder();
            builder.arg(&partial);
            builder.arg(&rv);
            builder.arg(&out);
            candle_core::builder_arg!(builder, slots_i, hidden_i, experts_i);
            let launch = config(
                (
                    u32_grid(hidden.div_ceil(256), "hidden")?,
                    u32_grid(tokens, "tokens")?,
                ),
                256,
            );
            // SAFETY: argument order matches `izwi_mr2_combine`.
            unsafe { builder.launch(launch) }.w()?;
        } else {
            let up = device.get_or_load_custom_func("izwi_mr2_up", MODULE, ptx)?;
            let mut builder = up.builder();
            builder.arg(&xv);
            builder.arg(&rv);
            builder.arg(&uv);
            builder.arg(&usv);
            builder.arg(&act);
            candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i, experts_i);
            let launch = config(
                (
                    u32_grid(pairs, "pairs")?,
                    u32_grid(inter.div_ceil(8), "inter")?,
                ),
                256,
            );
            // SAFETY: argument order matches `izwi_mr2_up`.
            unsafe { builder.launch(launch) }.w()?;
            let down = device.get_or_load_custom_func("izwi_mr2_down", MODULE, ptx)?;
            let mut builder = down.builder();
            builder.arg(&act);
            builder.arg(&rv);
            builder.arg(&dv);
            builder.arg(&dsv);
            builder.arg(&out);
            candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i, experts_i);
            let launch = config(
                (
                    u32_grid(tokens, "tokens")?,
                    u32_grid(hidden.div_ceil(8), "hidden")?,
                ),
                256,
            );
            // SAFETY: argument order matches `izwi_mr2_down`.
            unsafe { builder.launch(launch) }.w()?;
        }
        Ok(wrap(out, device, Shape::from((tokens, hidden))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::fp8::encode_e4m3fn;
    use crate::kernels::numeric_check::xorshift_values;

    const HIDDEN: usize = 2688;
    const INTER: usize = 1856;

    /// Random experts in the published geometry (fewer of them), quantized
    /// to E4M3 with one scale per projection as ModelOpt does.
    fn stacked(
        experts: usize,
        device: &Device,
        seed: u64,
    ) -> (StackedExperts, Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
        let quantize = |rows: usize, cols: usize, seed: u64| {
            let mut bytes = Vec::with_capacity(experts * rows * cols);
            let mut scales = Vec::with_capacity(experts);
            for e in 0..experts {
                let values =
                    xorshift_values(rows * cols, seed + e as u64, (cols as f32).sqrt().recip());
                let amax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = amax / 448.0;
                bytes.extend(values.iter().map(|v| encode_e4m3fn(v / scale)));
                scales.push(scale);
            }
            (bytes, scales)
        };
        let (up, up_scale) = quantize(INTER, HIDDEN, seed);
        let (down, down_scale) = quantize(HIDDEN, INTER, seed + 1000);
        let weights = StackedExperts::new(
            Tensor::from_vec(up.clone(), (experts, INTER, HIDDEN), device).unwrap(),
            Tensor::from_vec(up_scale.clone(), experts, device).unwrap(),
            Tensor::from_vec(down.clone(), (experts, HIDDEN, INTER), device).unwrap(),
            Tensor::from_vec(down_scale.clone(), experts, device).unwrap(),
        )
        .unwrap();
        (weights, up, up_scale, down, down_scale)
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (num, den) = a.iter().zip(b).fold((0f64, 0f64), |(n, d), (&x, &y)| {
            let diff = f64::from(x) - f64::from(y);
            (n + diff * diff, d + f64::from(y) * f64::from(y))
        });
        if !num.is_finite() {
            return f64::INFINITY;
        }
        (num / den.max(1e-30)).sqrt()
    }

    #[test]
    fn the_reference_router_is_the_familys_rule() {
        let spec = SigmoidRouting {
            experts: 8,
            top_k: 2,
            scale: 2.5,
        };
        let logits = [0.0f32, 1.0, -1.0, 2.0, 0.5, -2.0, 0.1, 0.2];
        let mut bias = vec![40.69f32; 8];
        bias[5] += 1.0;
        let routing = route_reference(&logits, &bias, spec);
        assert_eq!(&routing[..2], &[5.0, 3.0]);
        let s = |v: f32| 1.0 / (1.0 + (-v).exp());
        let total = s(-2.0) + s(2.0);
        assert_eq!(routing[2], s(-2.0) / total * 2.5);
        assert_eq!(routing[3], s(2.0) / total * 2.5);
    }

    #[test]
    fn the_cpu_path_is_the_reference_and_skips_invalid_slots() {
        let (weights, up, up_scale, down, down_scale) = stacked(2, &Device::Cpu, 3);
        let x = xorshift_values(HIDDEN, 9, 1.0);
        // Slot 1 names no expert.
        let routing = [1.0f32, -1.0, 0.75, 0.25];
        let y = experts(
            &Tensor::from_vec(x.clone(), (1, HIDDEN), &Device::Cpu).unwrap(),
            &Tensor::from_vec(routing.to_vec(), (1, 4), &Device::Cpu).unwrap(),
            2,
            &weights,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let only = experts_reference(
            &x,
            &[1.0, 0.75],
            1,
            &up,
            &up_scale,
            &down,
            &down_scale,
            HIDDEN,
            INTER,
        );
        assert_eq!(y, only);
        assert!(y.iter().any(|v| *v != 0.0));
    }

    #[cfg(feature = "metal")]
    fn metal() -> Option<(Device, impl Sized)> {
        crate::backends::metal_test_device_or_skip("relu² MoE")
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_routing_matches_the_reference() {
        let Some((device, _lease)) = metal() else {
            return;
        };
        let spec = SigmoidRouting {
            experts: 128,
            top_k: 6,
            scale: 2.5,
        };
        let tokens = 37;
        let logits = xorshift_values(tokens * 128, 21, 3.0);
        let bias = xorshift_values(128, 22, 0.01)
            .into_iter()
            .map(|b| 40.69 + b)
            .collect::<Vec<_>>();
        let expected = route_reference(&logits, &bias, spec);
        let routing = route(
            &Tensor::from_vec(logits, (tokens, 128), &device).unwrap(),
            &Tensor::from_vec(bias, 128, &device).unwrap(),
            spec,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        for (t, (row, want)) in routing.chunks(12).zip(expected.chunks(12)).enumerate() {
            assert_eq!(&row[..6], &want[..6], "token {t} experts");
            for (got, want) in row[6..].iter().zip(&want[6..]) {
                assert!(
                    (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                    "token {t}: {got} vs {want}"
                );
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_router_logits_match_the_f32_product_and_ignore_other_rows() {
        let Some((device, _lease)) = metal() else {
            return;
        };
        let gate = xorshift_values(128 * HIDDEN, 31, 0.02);
        let x = xorshift_values(5 * HIDDEN, 32, 1.0);
        let gate_t = Tensor::from_vec(gate.clone(), (128, HIDDEN), &device).unwrap();
        let all = router_logits(
            &Tensor::from_vec(x.clone(), (5, HIDDEN), &device).unwrap(),
            &gate_t,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let one = router_logits(
            &Tensor::from_vec(x[2 * HIDDEN..3 * HIDDEN].to_vec(), (1, HIDDEN), &device).unwrap(),
            &gate_t,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert_eq!(&all[2 * 128..3 * 128], one.as_slice());
        let reference = x
            .chunks(HIDDEN)
            .flat_map(|xr| {
                gate.chunks(HIDDEN)
                    .map(|gr| {
                        xr.iter()
                            .zip(gr)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>() as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            rel_l2(&all, &reference) < 1e-5,
            "{}",
            rel_l2(&all, &reference)
        );
    }

    /// Both forms of the experts match the reference, agree with each other
    /// bit for bit, and give a row the same output alone as beside others.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_experts_match_the_reference_in_both_forms() {
        let Some((device, _lease)) = metal() else {
            return;
        };
        let experts_n = 6;
        let (weights, up, up_scale, down, down_scale) = stacked(experts_n, &device, 41);
        let spec = SigmoidRouting {
            experts: experts_n,
            top_k: 3,
            scale: 2.5,
        };
        let tokens = 13;
        let x = xorshift_values(tokens * HIDDEN, 42, 1.0);
        let logits = xorshift_values(tokens * experts_n, 43, 2.0);
        let routing = route_reference(&logits, &vec![0.0; experts_n], spec);
        let reference = experts_reference(
            &x,
            &routing,
            3,
            &up,
            &up_scale,
            &down,
            &down_scale,
            HIDDEN,
            INTER,
        );
        let xt = Tensor::from_vec(x.clone(), (tokens, HIDDEN), &device).unwrap();
        let rt = Tensor::from_vec(routing.clone(), (tokens, 6), &device).unwrap();
        let run = |grouped: bool| {
            metal_impl::experts(&xt, &rt, 3, &weights, grouped)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let (per_pair, grouped) = (run(false), run(true));
        assert_eq!(per_pair, grouped);
        assert!(
            rel_l2(&per_pair, &reference) < 1e-5,
            "{}",
            rel_l2(&per_pair, &reference)
        );
        // Token 4 alone, through the dispatcher (per-pair form at 3 pairs).
        let alone = experts(
            &Tensor::from_vec(x[4 * HIDDEN..5 * HIDDEN].to_vec(), (1, HIDDEN), &device).unwrap(),
            &Tensor::from_vec(routing[4 * 6..5 * 6].to_vec(), (1, 6), &device).unwrap(),
            3,
            &weights,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert_eq!(alone.as_slice(), &grouped[4 * HIDDEN..5 * HIDDEN]);
        // The long-span GEMM form computes the same experts (in a different
        // summation order).
        let gemm = experts_by_gemm(&xt, &rt, 3, &weights)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            rel_l2(&gemm, &reference) < 1e-5,
            "{}",
            rel_l2(&gemm, &reference)
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_dequantizes_one_stacked_expert_exactly() {
        let Some((device, _lease)) = metal() else {
            return;
        };
        let (weights, _, _, down, down_scale) = stacked(3, &device, 51);
        let got = dequantize_expert(&weights.down, &weights.down_scale, 2, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = down[2 * HIDDEN * INTER..3 * HIDDEN * INTER]
            .iter()
            .map(|&b| decode_e4m3fn(b) * down_scale[2])
            .collect::<Vec<_>>();
        assert_eq!(got, want);
    }

    #[test]
    fn long_spans_switch_to_gemms_but_decode_never_does() {
        // Published geometry: 128 experts, top 6.
        assert!(!uses_gemm(32, 6, 128));
        assert!(!uses_gemm(256, 6, 128));
        assert!(uses_gemm(512, 6, 128));
        assert!(uses_gemm(2048, 6, 128));
        // Few experts: still never below the decode floor.
        assert!(!uses_gemm(GEMM_MIN_TOKENS - 1, 6, 1));
    }

    /// Times the warp kernels against the dequantized-GEMM form across span
    /// lengths in the published geometry (64 of the 128 experts), the
    /// evidence for [`GEMM_MIN_TOKENS`]. Run with `--ignored --nocapture`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "Metal benchmark; run explicitly"]
    fn metal_warp_kernels_versus_the_gemm_form_across_span_lengths() {
        let Some((device, _lease)) = metal() else {
            return;
        };
        let (experts_n, top_k) = (64usize, 6usize);
        let bytes = |len: usize, seed: u64| {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let b = (state >> 24) as u8 & 0xF7;
                    if b & 0x7f == 0x7f {
                        0x30
                    } else {
                        b
                    }
                })
                .collect::<Vec<u8>>()
        };
        let weights = StackedExperts::new(
            Tensor::from_vec(
                bytes(experts_n * INTER * HIDDEN, 1),
                (experts_n, INTER, HIDDEN),
                &device,
            )
            .unwrap(),
            Tensor::from_vec(vec![1e-3f32; experts_n], experts_n, &device).unwrap(),
            Tensor::from_vec(
                bytes(experts_n * HIDDEN * INTER, 2),
                (experts_n, HIDDEN, INTER),
                &device,
            )
            .unwrap(),
            Tensor::from_vec(vec![1e-3f32; experts_n], experts_n, &device).unwrap(),
        )
        .unwrap();
        let spec = SigmoidRouting {
            experts: experts_n,
            top_k,
            scale: 2.5,
        };
        let time = |f: &dyn Fn() -> Tensor| {
            f();
            device.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..5 {
                f();
            }
            device.synchronize().unwrap();
            started.elapsed().as_secs_f64() * 1e3 / 5.0
        };
        for tokens in [8usize, 16, 32, 64, 128, 256, 512] {
            let x = Tensor::from_vec(
                xorshift_values(tokens * HIDDEN, 5, 1.0),
                (tokens, HIDDEN),
                &device,
            )
            .unwrap();
            let logits = xorshift_values(tokens * experts_n, 6, 2.0);
            let routing = Tensor::from_vec(
                route_reference(&logits, &vec![0.0; experts_n], spec),
                (tokens, 2 * top_k),
                &device,
            )
            .unwrap();
            let warp = time(&|| {
                metal_impl::experts(
                    &x,
                    &routing,
                    top_k,
                    &weights,
                    tokens * top_k >= GROUPED_MIN_PAIRS,
                )
                .unwrap()
            });
            let gemm = time(&|| experts_by_gemm(&x, &routing, top_k, &weights).unwrap());
            eprintln!("{tokens:>4} tokens: warp {warp:8.3} ms, gemm {gemm:8.3} ms");
        }
    }
}

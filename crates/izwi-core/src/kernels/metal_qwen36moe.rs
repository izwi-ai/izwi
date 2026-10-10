//! Metal kernels for Qwen3.6-MoE block-FP8 residency.
//!
//! Apple GPUs have no FP8 arithmetic, but the CUDA kernels' E4M3 decode is
//! plain integer work: move the byte into the high byte of an F16 lane and
//! shift the exponent/mantissa down one bit (value x 2^-8, exact for every
//! finite E4M3 value). That lets Metal keep the checkpoint's raw FP8 bytes
//! resident (1 B/element instead of 2 for the F16 expansion) and run the same
//! device-routed sparse-expert kernels as CUDA. Every function mirrors its CUDA
//! counterpart in `kernels::cuda::moe` / `kernels::cuda::fp8`; the CPU
//! references there define the expected results and the Metal tests compare
//! against them.
use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::{bail, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// E4M3FN byte -> value * 2^-8 through an F16 lane (exact for finite values).
inline float q36m_e4m3(uint b) {
    ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return float(as_type<half>(h));
}

inline float q36m_dot4(uint w, thread const float* x) {
    float acc = q36m_e4m3(w & 0xFFu) * x[0];
    acc = fma(q36m_e4m3((w >> 8) & 0xFFu), x[1], acc);
    acc = fma(q36m_e4m3((w >> 16) & 0xFFu), x[2], acc);
    return fma(q36m_e4m3(w >> 24), x[3], acc);
}

inline float q36m_dot16(uint4 q, thread const float* x) {
    float acc = q36m_dot4(q.x, x);
    acc += q36m_dot4(q.y, x + 4);
    acc += q36m_dot4(q.z, x + 8);
    return acc + q36m_dot4(q.w, x + 12);
}

template <typename T>
inline void q36m_load16(device const T* x, thread float* out) {
    for (uint i = 0; i < 16; ++i) {
        out[i] = float(x[i]);
    }
}

// Block-FP8 projection y[m, n] = sum_k x[m, k] * w[n, k] * scale[n/128, k/128].
// One simdgroup per output channel, eight per threadgroup; grid (N/8, M).
template <typename T>
inline void q36m_fp8_mv(
    device const T* x, device const uchar* w, device const float* s, device T* y,
    uint M, uint N, uint K, uint2 tg, uint sg, uint lane) {
    const uint n = tg.x * 8 + sg;
    const uint m = tg.y;
    if (n >= N || m >= M) {
        return;
    }
    device const uchar* row = w + (ulong)n * K;
    device const float* scales = s + (ulong)(n >> 7) * (K >> 7);
    device const T* xr = x + (ulong)m * K;
    float acc = 0.0f;
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {
        const uint4 q = *((device const uint4*)(row + k0));
        float xv[16];
        q36m_load16(xr + k0, xv);
        acc = fma(q36m_dot16(q, xv), scales[k0 >> 7], acc);
    }
    acc = simd_sum(acc) * 256.0f;
    if (lane == 0) {
        y[(ulong)m * N + n] = T(acc);
    }
}

kernel void q36m_fp8_mv_f16(
    device const half* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device const float* s [[buffer(2)]], device half* y [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& K [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    q36m_fp8_mv<half>(x, w, s, y, M, N, K, tg, sg, lane);
}

kernel void q36m_fp8_mv_f32(
    device const float* x [[buffer(0)]], device const uchar* w [[buffer(1)]],
    device const float* s [[buffer(2)]], device float* y [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& K [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    q36m_fp8_mv<float>(x, w, s, y, M, N, K, tg, sg, lane);
}

// Router: one simdgroup per token; see qwen36moe_route_f32 (CUDA).
kernel void q36m_route_f32(
    device const float* logits [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& tokens [[buffer(2)]], constant uint& num_experts [[buffer(3)]],
    constant uint& ldl [[buffer(4)]], constant uint& top_k [[buffer(5)]],
    constant uint& shared_mode [[buffer(6)]], constant uint& shared_slot_id [[buffer(7)]],
    constant uint& norm_topk [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = tg * 4 + sg;
    if (token >= tokens) {
        return;
    }
    device const float* row = logits + (ulong)token * ldl;
    float v[16];
    float local_max = -INFINITY;
    for (uint i = 0; i < 16; ++i) {
        const uint e = lane + 32 * i;
        v[i] = e < num_experts ? row[e] : -INFINITY;
        local_max = max(local_max, v[i]);
    }
    const float row_max = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        if (lane + 32 * i < num_experts) {
            local_sum += exp(v[i] - row_max);
        }
    }
    const float total = simd_sum(local_sum);
    const uint slots = top_k + (shared_mode != 0 ? 1 : 0);
    device float* dst = out + (ulong)token * 2 * slots;
    uint taken = 0;
    float picked_sum = 0.0f;
    const uint none = 0xFFFFFFFFu;
    for (uint k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        uint best_e = none;
        for (uint i = 0; i < 16; ++i) {
            const uint e = lane + 32 * i;
            const bool open = e < num_experts && ((taken >> i) & 1u) == 0u;
            if (open && (best_e == none || v[i] > best || (v[i] == best && e < best_e))) {
                best = v[i];
                best_e = e;
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float other = simd_shuffle_xor(best, o);
            const uint other_e = simd_shuffle_xor(best_e, o);
            if (other_e != none && (best_e == none || other > best || (other == best && other_e < best_e))) {
                best = other;
                best_e = other_e;
            }
        }
        for (uint i = 0; i < 16; ++i) {
            if (lane + 32 * i == best_e) {
                taken |= 1u << i;
            }
        }
        const float p = exp(best - row_max);
        picked_sum += p;
        if (lane == 0) {
            dst[k] = float(best_e);
            dst[slots + k] = p;
        }
    }
    if (lane == 0) {
        const float denom = norm_topk != 0 ? picked_sum : total;
        for (uint k = 0; k < top_k; ++k) {
            dst[slots + k] = dst[slots + k] / denom;
        }
        if (shared_mode != 0) {
            dst[top_k] = float(shared_slot_id);
            dst[slots + top_k] = shared_mode == 2 ? 1.0f / (1.0f + exp(-row[num_experts])) : 1.0f;
        }
    }
}

// Router GEMV: logits[t, r] = float(T(x[t] . w[r])). One simdgroup per row.
kernel void q36m_router_logits_f16(
    device const half* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x * 8 + sg;
    const uint token = tg.y;
    if (row >= rows) {
        return;
    }
    device const float* wr = w + (ulong)row * hidden;
    device const half* xt = x + (ulong)token * hidden;
    float acc = 0.0f;
    for (uint k = lane * 4; k < hidden; k += 128) {
        const float4 wv = *((device const float4*)(wr + k));
        acc = fma(float(xt[k]), wv.x, acc);
        acc = fma(float(xt[k + 1]), wv.y, acc);
        acc = fma(float(xt[k + 2]), wv.z, acc);
        acc = fma(float(xt[k + 3]), wv.w, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[(ulong)token * rows + row] = float(half(acc));
    }
}

// Grouped gate+up+SwiGLU per routed (token, slot) pair; grid (pairs, inter/8).
kernel void q36m_gate_up_f16(
    device const half* x [[buffer(0)]], device const float* routing [[buffer(1)]],
    device const uchar* w13 [[buffer(2)]], device const float* s13 [[buffer(3)]],
    device half* act [[buffer(4)]], constant uint& slots [[buffer(5)]],
    constant uint& hidden [[buffer(6)]], constant uint& inter [[buffer(7)]],
    constant uint& experts_total [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.x;
    const uint token = pair / slots;
    const uint slot = pair - token * slots;
    const uint n = tg.y * 8 + sg;
    if (n >= inter) {
        return;
    }
    const float expert_f = routing[(ulong)token * 2 * slots + slot];
    if (expert_f < 0.0f || uint(expert_f) >= experts_total) {
        if (lane == 0) {
            act[(ulong)pair * inter + n] = half(0.0f);
        }
        return;
    }
    const uint expert = uint(expert_f);
    const uint kblocks = hidden >> 7;
    device const uchar* wg = w13 + ((ulong)expert * 2 * inter + n) * hidden;
    device const uchar* wu = wg + (ulong)inter * hidden;
    device const float* expert_scales = s13 + (ulong)expert * ((2 * inter) >> 7) * kblocks;
    device const float* sgs = expert_scales + (ulong)(n >> 7) * kblocks;
    device const float* sus = expert_scales + (ulong)((inter + n) >> 7) * kblocks;
    device const half* xt = x + (ulong)token * hidden;
    float accg = 0.0f;
    float accu = 0.0f;
    for (uint k0 = lane * 16; k0 < hidden; k0 += 512) {
        const uint4 qg = *((device const uint4*)(wg + k0));
        const uint4 qu = *((device const uint4*)(wu + k0));
        float xv[16];
        q36m_load16(xt + k0, xv);
        const uint kb = k0 >> 7;
        accg = fma(q36m_dot16(qg, xv), sgs[kb], accg);
        accu = fma(q36m_dot16(qu, xv), sus[kb], accu);
    }
    accg = simd_sum(accg) * 256.0f;
    accu = simd_sum(accu) * 256.0f;
    if (lane == 0) {
        const float silu = accg / (1.0f + exp(-accg));
        act[(ulong)pair * inter + n] = half(silu * accu);
    }
}

// Grouped down + weighted combine; grid (tokens, hidden/8).
kernel void q36m_down_f16(
    device const half* act [[buffer(0)]], device const float* routing [[buffer(1)]],
    device const uchar* w2 [[buffer(2)]], device const float* s2 [[buffer(3)]],
    device half* y [[buffer(4)]], constant uint& slots [[buffer(5)]],
    constant uint& hidden [[buffer(6)]], constant uint& inter [[buffer(7)]],
    constant uint& experts_total [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = tg.x;
    const uint h = tg.y * 8 + sg;
    if (h >= hidden) {
        return;
    }
    const uint kblocks = inter >> 7;
    float total = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        const float expert_f = routing[(ulong)token * 2 * slots + s];
        const float weight = routing[(ulong)token * 2 * slots + slots + s];
        if (expert_f < 0.0f || uint(expert_f) >= experts_total) {
            continue;
        }
        const uint e = uint(expert_f);
        device const uchar* w = w2 + ((ulong)e * hidden + h) * inter;
        device const float* sc = s2 + ((ulong)e * (hidden >> 7) + (h >> 7)) * kblocks;
        device const half* a = act + ((ulong)token * slots + s) * inter;
        float acc = 0.0f;
        for (uint k0 = lane * 16; k0 < inter; k0 += 512) {
            const uint4 q = *((device const uint4*)(w + k0));
            float av[16];
            q36m_load16(a + k0, av);
            acc = fma(q36m_dot16(q, av), sc[k0 >> 7], acc);
        }
        total = fma(weight, acc, total);
    }
    total = simd_sum(total) * 256.0f;
    if (lane == 0) {
        y[(ulong)token * hidden + h] = half(total);
    }
}

// ---- Expert-major (grouped) MoE for prefill; see the CUDA kernels. ----
kernel void q36m_group_pairs(
    device const float* routing [[buffer(0)]], device int* offsets [[buffer(1)]],
    device int* sorted_pairs [[buffer(2)]], constant uint& tokens [[buffer(3)]],
    constant uint& slots [[buffer(4)]], constant uint& experts_total [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup atomic_int counts[520];
    threadgroup atomic_int fill[520];
    threadgroup int starts[521];
    for (uint e = tid; e < experts_total; e += threads) {
        atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
        atomic_store_explicit(&fill[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pairs = tokens * slots;
    for (uint p = tid; p < pairs; p += threads) {
        const uint token = p / slots;
        const float expert_f = routing[(ulong)token * 2 * slots + (p - token * slots)];
        if (expert_f >= 0.0f && uint(expert_f) < experts_total) {
            atomic_fetch_add_explicit(&counts[uint(expert_f)], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int running = 0;
        for (uint e = 0; e < experts_total; ++e) {
            starts[e] = running;
            offsets[e] = running;
            running += atomic_load_explicit(&counts[e], memory_order_relaxed);
        }
        offsets[experts_total] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < pairs; p += threads) {
        const uint token = p / slots;
        const float expert_f = routing[(ulong)token * 2 * slots + (p - token * slots)];
        if (expert_f >= 0.0f && uint(expert_f) < experts_total) {
            const uint e = uint(expert_f);
            const int index = atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
            sorted_pairs[starts[e] + index] = int(p);
        }
    }
}

// Requires hidden % 512 == 0 and hidden / 512 <= 8.
kernel void q36m_gate_up_grouped_f16(
    device const half* x [[buffer(0)]], device const int* offsets [[buffer(1)]],
    device const int* sorted_pairs [[buffer(2)]], device const uchar* w13 [[buffer(3)]],
    device const float* s13 [[buffer(4)]], device half* act [[buffer(5)]],
    constant uint& slots [[buffer(6)]], constant uint& hidden [[buffer(7)]],
    constant uint& inter [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg.x;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    const uint n = tg.y * 8 + sg;
    if (begin == end || n >= inter) {
        return;
    }
    const uint kblocks = hidden >> 7;
    const uint chunks = hidden >> 9;
    device const uchar* wg = w13 + ((ulong)expert * 2 * inter + n) * hidden;
    device const uchar* wu = wg + (ulong)inter * hidden;
    device const float* expert_scales = s13 + (ulong)expert * ((2 * inter) >> 7) * kblocks;
    device const float* sgs = expert_scales + (ulong)(n >> 7) * kblocks;
    device const float* sus = expert_scales + (ulong)((inter + n) >> 7) * kblocks;
    uint4 qg[8];
    uint4 qu[8];
    float scale_g[8];
    float scale_u[8];
    for (uint c = 0; c < chunks; ++c) {
        const uint k0 = lane * 16 + c * 512;
        qg[c] = *((device const uint4*)(wg + k0));
        qu[c] = *((device const uint4*)(wu + k0));
        scale_g[c] = sgs[k0 >> 7];
        scale_u[c] = sus[k0 >> 7];
    }
    for (int i = begin; i < end; ++i) {
        const uint pair = uint(sorted_pairs[i]);
        device const half* xt = x + (ulong)(pair / slots) * hidden;
        float accg = 0.0f;
        float accu = 0.0f;
        for (uint c = 0; c < chunks; ++c) {
            float xv[16];
            q36m_load16(xt + lane * 16 + c * 512, xv);
            accg = fma(q36m_dot16(qg[c], xv), scale_g[c], accg);
            accu = fma(q36m_dot16(qu[c], xv), scale_u[c], accu);
        }
        accg = simd_sum(accg) * 256.0f;
        accu = simd_sum(accu) * 256.0f;
        if (lane == 0) {
            const float silu = accg / (1.0f + exp(-accg));
            act[(ulong)pair * inter + n] = half(silu * accu);
        }
    }
}

// Requires inter % 512 == 0 and inter / 512 <= 4.
kernel void q36m_down_grouped_f16(
    device const half* act [[buffer(0)]], device const int* offsets [[buffer(1)]],
    device const int* sorted_pairs [[buffer(2)]], device const uchar* w2 [[buffer(3)]],
    device const float* s2 [[buffer(4)]], device float* partial [[buffer(5)]],
    constant uint& hidden [[buffer(6)]], constant uint& inter [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg.x;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    const uint h = tg.y * 8 + sg;
    if (begin == end || h >= hidden) {
        return;
    }
    const uint kblocks = inter >> 7;
    const uint chunks = inter >> 9;
    device const uchar* w = w2 + ((ulong)expert * hidden + h) * inter;
    device const float* sc = s2 + ((ulong)expert * (hidden >> 7) + (h >> 7)) * kblocks;
    uint4 q[4];
    float scale[4];
    for (uint c = 0; c < chunks; ++c) {
        const uint k0 = lane * 16 + c * 512;
        q[c] = *((device const uint4*)(w + k0));
        scale[c] = sc[k0 >> 7];
    }
    for (int i = begin; i < end; ++i) {
        const uint pair = uint(sorted_pairs[i]);
        device const half* a = act + (ulong)pair * inter;
        float acc = 0.0f;
        for (uint c = 0; c < chunks; ++c) {
            float av[16];
            q36m_load16(a + lane * 16 + c * 512, av);
            acc = fma(q36m_dot16(q[c], av), scale[c], acc);
        }
        acc = simd_sum(acc) * 256.0f;
        if (lane == 0) {
            partial[(ulong)pair * hidden + h] = acc;
        }
    }
}

kernel void q36m_combine_f16(
    device const float* partial [[buffer(0)]], device const float* routing [[buffer(1)]],
    device half* y [[buffer(2)]], constant uint& slots [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint token = gid.y;
    if (h >= hidden) {
        return;
    }
    device const float* weights = routing + (ulong)token * 2 * slots + slots;
    float total = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        total = fma(weights[s], partial[((ulong)token * slots + s) * hidden + h], total);
    }
    y[(ulong)token * hidden + h] = half(total);
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

fn u32_arg(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| candle_core::Error::Msg(format!("{name} {value} exceeds u32")))
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
            _ => bail!("Qwen3.6 Metal kernels need Metal tensors"),
        }
    }
}

fn bind<'a>(tensor: &'a Tensor, name: &str, align: usize) -> Result<Bound<'a>> {
    let (guard, layout) = tensor.storage_and_layout();
    if !layout.is_contiguous() {
        bail!("Qwen3.6 Metal kernel input {name} must be contiguous")
    }
    let offset = layout.start_offset() * tensor.dtype().size_in_bytes();
    if !offset.is_multiple_of(align) {
        bail!("Qwen3.6 Metal kernel input {name} must start on a {align}-byte boundary")
    }
    Ok(Bound { guard, offset })
}

fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
    match tensor.device() {
        candle_core::Device::Metal(device) => Ok(device.clone()),
        _ => bail!("Qwen3.6 Metal kernels need Metal tensors"),
    }
}

fn output(
    device: &candle_core::MetalDevice,
    shape: Shape,
    dtype: DType,
    label: &str,
) -> Result<(std::sync::Arc<Buffer>, Shape)> {
    let buffer = device.new_buffer(shape.elem_count(), dtype, label)?;
    Ok((buffer, shape))
}

fn wrap(
    device: &candle_core::MetalDevice,
    buffer: std::sync::Arc<Buffer>,
    shape: Shape,
    dtype: DType,
) -> Tensor {
    let count = shape.elem_count();
    Tensor::from_storage(
        Storage::Metal(MetalStorage::new(buffer, device.clone(), count, dtype)),
        shape,
        BackpropOp::none(),
        false,
    )
}

/// Block-FP8 projection on raw Metal storages (the `CustomOp3` path of
/// `kernels::cuda::fp8::block_fp8_projection`): `x` `[m, k]` (F16 or F32),
/// `w` `[n, k]` E4M3 bytes, `s` F32 block scales.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_projection(
    x: &MetalStorage,
    xl: &Layout,
    w: &MetalStorage,
    wl: &Layout,
    s: &MetalStorage,
    sl: &Layout,
    n: usize,
    k: usize,
) -> Result<(MetalStorage, Shape)> {
    let dtype = x.dtype();
    let name = match dtype {
        DType::F16 => "q36m_fp8_mv_f16",
        DType::F32 => "q36m_fp8_mv_f32",
        other => bail!("Metal block-FP8 projection does not support {other:?} activations"),
    };
    if !xl.is_contiguous() || !wl.is_contiguous() || !sl.is_contiguous() {
        bail!("Metal block-FP8 projection needs contiguous operands")
    }
    if !wl.start_offset().is_multiple_of(16) || !k.is_multiple_of(16) {
        bail!("Metal block-FP8 projection needs 16-byte aligned weight rows")
    }
    let m = xl.shape().elem_count() / k;
    let device = x.device().clone();
    let out = device.new_buffer(m * n, dtype, "q36m-fp8-projection")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-fp8-projection");
    let pipeline = pipeline(device.metal_device(), name)?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(
        0,
        Some(x.buffer()),
        xl.start_offset() * dtype.size_in_bytes(),
    );
    encoder.set_input_buffer(1, Some(w.buffer()), wl.start_offset());
    encoder.set_input_buffer(2, Some(s.buffer()), sl.start_offset() * 4);
    encoder.set_output_buffer(3, Some(&out), 0);
    encoder.set_bytes(4, &u32_arg(m, "rows")?);
    encoder.set_bytes(5, &u32_arg(n, "out")?);
    encoder.set_bytes(6, &u32_arg(k, "in")?);
    encoder.dispatch_thread_groups(grid(n.div_ceil(8), m), grid(256, 1));
    drop(encoder);
    let mut dims = xl.dims().to_vec();
    *dims.last_mut().expect("projection input has a last dim") = n;
    Ok((
        MetalStorage::new(out, device, m * n, dtype),
        Shape::from(dims),
    ))
}

pub(crate) fn route(
    logits: &Tensor,
    tokens: usize,
    spec: &crate::kernels::cuda::moe::RouteSpec,
) -> Result<Tensor> {
    let device = metal_device(logits)?;
    let slots = spec.slots();
    let input = bind(logits, "router logits", 4)?;
    let (out, shape) = output(
        &device,
        Shape::from((tokens, 2 * slots)),
        DType::F32,
        "q36m-route",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-route");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_route_f32")?);
    encoder.set_input_buffer(0, Some(input.buffer()?), input.offset);
    encoder.set_output_buffer(1, Some(&out), 0);
    encoder.set_bytes(2, &u32_arg(tokens, "tokens")?);
    encoder.set_bytes(3, &u32_arg(spec.num_experts, "experts")?);
    encoder.set_bytes(4, &u32_arg(spec.logit_columns(), "columns")?);
    encoder.set_bytes(5, &u32_arg(spec.top_k, "top_k")?);
    encoder.set_bytes(6, &(spec.shared.mode() as u32));
    encoder.set_bytes(7, &u32_arg(spec.shared_slot_id, "shared slot")?);
    encoder.set_bytes(8, &u32::from(spec.norm_topk));
    encoder.dispatch_thread_groups(grid(tokens.div_ceil(4), 1), grid(128, 1));
    drop(encoder);
    drop(input);
    Ok(wrap(&device, out, shape, DType::F32))
}

pub(crate) fn router_logits(
    x: &Tensor,
    weight: &Tensor,
    tokens: usize,
    rows: usize,
    hidden: usize,
) -> Result<Tensor> {
    if x.dtype() != DType::F16 {
        bail!(
            "Metal MoE router needs F16 activations, found {:?}",
            x.dtype()
        )
    }
    let device = metal_device(x)?;
    let xb = bind(x, "activations", 2)?;
    let wb = bind(weight, "router weight", 16)?;
    let (out, shape) = output(
        &device,
        Shape::from((tokens, rows)),
        DType::F32,
        "q36m-router",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-router");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_router_logits_f16")?);
    encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
    encoder.set_input_buffer(1, Some(wb.buffer()?), wb.offset);
    encoder.set_output_buffer(2, Some(&out), 0);
    encoder.set_bytes(3, &u32_arg(rows, "rows")?);
    encoder.set_bytes(4, &u32_arg(hidden, "hidden")?);
    encoder.dispatch_thread_groups(grid(rows.div_ceil(8), tokens), grid(256, 1));
    drop(encoder);
    drop((xb, wb));
    Ok(wrap(&device, out, shape, DType::F32))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_up(
    x: &Tensor,
    routing: &Tensor,
    slots: usize,
    w13: &Tensor,
    s13: &Tensor,
    experts_total: usize,
    hidden: usize,
    inter: usize,
) -> Result<Tensor> {
    if x.dtype() != DType::F16 {
        bail!(
            "Metal fused MoE needs F16 activations, found {:?}",
            x.dtype()
        )
    }
    let device = metal_device(x)?;
    let tokens = x.dim(0)?;
    let pairs = tokens * slots;
    let (x, routing, w13, s13) = (
        x.contiguous()?,
        routing.contiguous()?,
        w13.contiguous()?,
        s13.contiguous()?,
    );
    let xb = bind(&x, "activations", 2)?;
    let rb = bind(&routing, "routing", 4)?;
    let wb = bind(&w13, "w13", 16)?;
    let sb = bind(&s13, "s13", 4)?;
    let (out, shape) = output(
        &device,
        Shape::from((pairs, inter)),
        DType::F16,
        "q36m-gate-up",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-gate-up");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_gate_up_f16")?);
    encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
    encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
    encoder.set_input_buffer(2, Some(wb.buffer()?), wb.offset);
    encoder.set_input_buffer(3, Some(sb.buffer()?), sb.offset);
    encoder.set_output_buffer(4, Some(&out), 0);
    encoder.set_bytes(5, &u32_arg(slots, "slots")?);
    encoder.set_bytes(6, &u32_arg(hidden, "hidden")?);
    encoder.set_bytes(7, &u32_arg(inter, "inter")?);
    encoder.set_bytes(8, &u32_arg(experts_total, "experts")?);
    encoder.dispatch_thread_groups(grid(pairs, inter.div_ceil(8)), grid(256, 1));
    drop(encoder);
    drop((xb, rb, wb, sb));
    Ok(wrap(&device, out, shape, DType::F16))
}

/// Expert-major routed MoE for prefill (see `moe::fp8_moe_grouped`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped(
    x: &Tensor,
    routing: &Tensor,
    slots: usize,
    w13: &Tensor,
    s13: &Tensor,
    w2: &Tensor,
    s2: &Tensor,
    experts_total: usize,
    hidden: usize,
    inter: usize,
) -> Result<Tensor> {
    if x.dtype() != DType::F16 {
        bail!(
            "Metal grouped MoE needs F16 activations, found {:?}",
            x.dtype()
        )
    }
    let device = metal_device(x)?;
    let tokens = x.dim(0)?;
    let pairs = tokens * slots;
    let (x, routing, w13, s13, w2, s2) = (
        x.contiguous()?,
        routing.contiguous()?,
        w13.contiguous()?,
        s13.contiguous()?,
        w2.contiguous()?,
        s2.contiguous()?,
    );
    let xb = bind(&x, "activations", 32)?;
    let rb = bind(&routing, "routing", 4)?;
    let w13b = bind(&w13, "w13", 16)?;
    let s13b = bind(&s13, "s13", 4)?;
    let w2b = bind(&w2, "w2", 16)?;
    let s2b = bind(&s2, "s2", 4)?;
    let offsets = device.new_buffer(experts_total + 1, DType::U32, "q36m-offsets")?;
    let sorted = device.new_buffer(pairs, DType::U32, "q36m-sorted")?;
    let act = device.new_buffer(pairs * inter, DType::F16, "q36m-grouped-act")?;
    let partial = device.new_buffer(pairs * hidden, DType::F32, "q36m-grouped-partial")?;
    let (out, shape) = output(
        &device,
        Shape::from((tokens, hidden)),
        DType::F16,
        "q36m-grouped",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-grouped");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_group_pairs")?);
    encoder.set_input_buffer(0, Some(rb.buffer()?), rb.offset);
    encoder.set_output_buffer(1, Some(&offsets), 0);
    encoder.set_output_buffer(2, Some(&sorted), 0);
    encoder.set_bytes(3, &u32_arg(tokens, "tokens")?);
    encoder.set_bytes(4, &u32_arg(slots, "slots")?);
    encoder.set_bytes(5, &u32_arg(experts_total, "experts")?);
    encoder.dispatch_thread_groups(grid(1, 1), grid(1024, 1));
    encoder.insert_memory_barrier();
    encoder.set_compute_pipeline_state(&pipeline(
        device.metal_device(),
        "q36m_gate_up_grouped_f16",
    )?);
    encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
    encoder.set_input_buffer(1, Some(&offsets), 0);
    encoder.set_input_buffer(2, Some(&sorted), 0);
    encoder.set_input_buffer(3, Some(w13b.buffer()?), w13b.offset);
    encoder.set_input_buffer(4, Some(s13b.buffer()?), s13b.offset);
    encoder.set_output_buffer(5, Some(&act), 0);
    encoder.set_bytes(6, &u32_arg(slots, "slots")?);
    encoder.set_bytes(7, &u32_arg(hidden, "hidden")?);
    encoder.set_bytes(8, &u32_arg(inter, "inter")?);
    encoder.dispatch_thread_groups(grid(experts_total, inter.div_ceil(8)), grid(256, 1));
    encoder.insert_memory_barrier();
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_down_grouped_f16")?);
    encoder.set_input_buffer(0, Some(&act), 0);
    encoder.set_input_buffer(1, Some(&offsets), 0);
    encoder.set_input_buffer(2, Some(&sorted), 0);
    encoder.set_input_buffer(3, Some(w2b.buffer()?), w2b.offset);
    encoder.set_input_buffer(4, Some(s2b.buffer()?), s2b.offset);
    encoder.set_output_buffer(5, Some(&partial), 0);
    encoder.set_bytes(6, &u32_arg(hidden, "hidden")?);
    encoder.set_bytes(7, &u32_arg(inter, "inter")?);
    encoder.dispatch_thread_groups(grid(experts_total, hidden.div_ceil(8)), grid(256, 1));
    encoder.insert_memory_barrier();
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_combine_f16")?);
    encoder.set_input_buffer(0, Some(&partial), 0);
    encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
    encoder.set_output_buffer(2, Some(&out), 0);
    encoder.set_bytes(3, &u32_arg(slots, "slots")?);
    encoder.set_bytes(4, &u32_arg(hidden, "hidden")?);
    encoder.dispatch_threads(grid(hidden, tokens), grid(256, 1));
    drop(encoder);
    drop((xb, rb, w13b, s13b, w2b, s2b));
    Ok(wrap(&device, out, shape, DType::F16))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn down(
    act: &Tensor,
    routing: &Tensor,
    slots: usize,
    w2: &Tensor,
    s2: &Tensor,
    experts_total: usize,
    hidden: usize,
    inter: usize,
) -> Result<Tensor> {
    if act.dtype() != DType::F16 {
        bail!(
            "Metal fused MoE needs F16 activations, found {:?}",
            act.dtype()
        )
    }
    let device = metal_device(act)?;
    let tokens = act.dim(0)? / slots;
    let (act, routing, w2, s2) = (
        act.contiguous()?,
        routing.contiguous()?,
        w2.contiguous()?,
        s2.contiguous()?,
    );
    let ab = bind(&act, "expert activations", 2)?;
    let rb = bind(&routing, "routing", 4)?;
    let wb = bind(&w2, "w2", 16)?;
    let sb = bind(&s2, "s2", 4)?;
    let (out, shape) = output(
        &device,
        Shape::from((tokens, hidden)),
        DType::F16,
        "q36m-down",
    )?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-down");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_down_f16")?);
    encoder.set_input_buffer(0, Some(ab.buffer()?), ab.offset);
    encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
    encoder.set_input_buffer(2, Some(wb.buffer()?), wb.offset);
    encoder.set_input_buffer(3, Some(sb.buffer()?), sb.offset);
    encoder.set_output_buffer(4, Some(&out), 0);
    encoder.set_bytes(5, &u32_arg(slots, "slots")?);
    encoder.set_bytes(6, &u32_arg(hidden, "hidden")?);
    encoder.set_bytes(7, &u32_arg(inter, "inter")?);
    encoder.set_bytes(8, &u32_arg(experts_total, "experts")?);
    encoder.dispatch_thread_groups(grid(tokens, hidden.div_ceil(8)), grid(256, 1));
    drop(encoder);
    drop((ab, rb, wb, sb));
    Ok(wrap(&device, out, shape, DType::F16))
}

#[cfg(test)]
mod tests {
    use crate::kernels::cuda::fp8::block_fp8_projection;
    use crate::kernels::cuda::moe::{self, RouteSpec, SharedSlot};
    use candle_core::{DType, Device, Tensor};

    fn device() -> Option<Device> {
        crate::backends::metal_device_if_available(0)
    }

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| loop {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let b = state as u8;
                if b & 0x7f != 0x7f {
                    break b;
                }
            })
            .collect()
    }

    fn wave(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * scale)
            .collect()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}");
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol * scale,
                "{label} index {i}: {a} vs {e}"
            );
        }
    }

    #[test]
    fn metal_fp8_projection_matches_the_cpu_reference() {
        let Some(gpu) = device() else { return };
        for (m, n, k) in [(1usize, 64usize, 256usize), (3, 136, 512), (7, 512, 2048)] {
            let w = Tensor::from_vec(bytes(n * k, (n + k) as u64), (n, k), &Device::Cpu).unwrap();
            let s = Tensor::from_vec(
                wave(n.div_ceil(128) * k.div_ceil(128), 1.0, 0.001)
                    .iter()
                    .map(|v| v.abs() + 0.002)
                    .collect::<Vec<_>>(),
                (n.div_ceil(128), k.div_ceil(128)),
                &Device::Cpu,
            )
            .unwrap();
            let x = Tensor::from_vec(wave(m * k, 2.0, 2.0), (m, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let expected = host(&block_fp8_projection(&x, &w, &s).unwrap());
            let to = |t: &Tensor| t.to_device(&gpu).unwrap();
            let actual = host(&block_fp8_projection(&to(&x), &to(&w), &to(&s)).unwrap());
            assert_close(&actual, &expected, 2e-3, &format!("m={m} n={n} k={k}"));
        }
    }

    #[test]
    fn metal_grouped_prefill_matches_the_per_pair_kernels() {
        let Some(gpu) = device() else { return };
        let (hidden, inter, experts, tokens) = (1024usize, 512usize, 33usize, 48usize);
        let spec = RouteSpec {
            num_experts: experts,
            top_k: 4,
            shared: SharedSlot::Gated,
            shared_slot_id: experts,
            norm_topk: true,
        };
        let total = experts + 1;
        let to = |t: Tensor| t.to_device(&gpu).unwrap();
        let w13 = to(Tensor::from_vec(
            bytes(total * 2 * inter * hidden, 11),
            (total, 2 * inter, hidden),
            &Device::Cpu,
        )
        .unwrap());
        let s13 = to(Tensor::from_vec(
            wave(total * (2 * inter / 128) * (hidden / 128), 1.0, 0.001)
                .iter()
                .map(|v| v.abs() + 0.002)
                .collect::<Vec<_>>(),
            (total, 2 * inter / 128, hidden / 128),
            &Device::Cpu,
        )
        .unwrap());
        let w2 = to(Tensor::from_vec(
            bytes(total * hidden * inter, 12),
            (total, hidden, inter),
            &Device::Cpu,
        )
        .unwrap());
        let s2 = to(Tensor::from_vec(
            wave(total * (hidden / 128) * (inter / 128), 2.0, 0.001)
                .iter()
                .map(|v| v.abs() + 0.002)
                .collect::<Vec<_>>(),
            (total, hidden / 128, inter / 128),
            &Device::Cpu,
        )
        .unwrap());
        let logits = to(Tensor::from_vec(
            wave(tokens * (experts + 1), 3.0, 3.0),
            (tokens, experts + 1),
            &Device::Cpu,
        )
        .unwrap());
        let x = to(Tensor::from_vec(
            wave(tokens * hidden, 4.0, 1.5),
            (tokens, hidden),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap());
        let routing = moe::route(&logits, &spec).unwrap();
        let slots = spec.slots();
        let per_pair = host(
            &moe::fp8_down(
                &moe::fp8_gate_up(&x, &routing, slots, &w13, &s13).unwrap(),
                &routing,
                slots,
                &w2,
                &s2,
            )
            .unwrap(),
        );
        let grouped =
            host(&moe::fp8_moe_grouped(&x, &routing, slots, &w13, &s13, &w2, &s2).unwrap());
        assert_close(&grouped, &per_pair, 0.005, "grouped vs per-pair");
    }

    #[test]
    fn metal_fused_moe_matches_the_cpu_reference_at_trunk_geometry() {
        let Some(gpu) = device() else { return };
        let (hidden, inter, experts) = (2048usize, 512usize, 256usize);
        let spec = RouteSpec {
            num_experts: experts,
            top_k: 8,
            shared: SharedSlot::Gated,
            shared_slot_id: experts,
            norm_topk: true,
        };
        let total = experts + 1;
        let w13 = Tensor::from_vec(
            bytes(total * 2 * inter * hidden, 7),
            (total, 2 * inter, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let s13 = Tensor::from_vec(
            wave(total * (2 * inter / 128) * (hidden / 128), 3.0, 0.001)
                .iter()
                .map(|v| v.abs() + 0.002)
                .collect::<Vec<_>>(),
            (total, 2 * inter / 128, hidden / 128),
            &Device::Cpu,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            bytes(total * hidden * inter, 9),
            (total, hidden, inter),
            &Device::Cpu,
        )
        .unwrap();
        let s2 = Tensor::from_vec(
            wave(total * (hidden / 128) * (inter / 128), 4.0, 0.001)
                .iter()
                .map(|v| v.abs() + 0.002)
                .collect::<Vec<_>>(),
            (total, hidden / 128, inter / 128),
            &Device::Cpu,
        )
        .unwrap();
        let router = Tensor::from_vec(
            wave((experts + 1) * hidden, 5.0, 0.05),
            (experts + 1, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let to = |t: &Tensor| t.to_device(&gpu).unwrap();
        let (gw13, gs13, gw2, gs2, grouter) = (to(&w13), to(&s13), to(&w2), to(&s2), to(&router));
        for tokens in [1usize, 5] {
            let x = Tensor::from_vec(
                wave(tokens * hidden, 6.0 + tokens as f32, 1.5),
                (tokens, hidden),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            // Router GEMV: Metal vs the F16-rounded CPU projection.
            let cpu_logits = moe::router_logits(&x, &router).unwrap();
            let gpu_logits = moe::router_logits(&to(&x), &grouter).unwrap();
            assert_close(
                &host(&gpu_logits),
                &host(&cpu_logits),
                2e-3,
                "router logits",
            );
            // Routing on identical logits must agree exactly on ids.
            let cpu_routing = moe::route(&cpu_logits, &spec).unwrap();
            let gpu_routing = moe::route(&to(&cpu_logits), &spec).unwrap();
            let (cr, gr) = (host(&cpu_routing), host(&gpu_routing));
            let slots = spec.slots();
            for t in 0..tokens {
                assert_eq!(
                    cr[t * 2 * slots..t * 2 * slots + slots],
                    gr[t * 2 * slots..t * 2 * slots + slots],
                    "ids T={tokens}"
                );
            }
            assert_close(&gr, &cr, 1e-4, "routing weights");
            let expected = host(
                &moe::fp8_down(
                    &moe::fp8_gate_up(&x, &cpu_routing, slots, &w13, &s13).unwrap(),
                    &cpu_routing,
                    slots,
                    &w2,
                    &s2,
                )
                .unwrap(),
            );
            let act = moe::fp8_gate_up(&to(&x), &to(&cpu_routing), slots, &gw13, &gs13).unwrap();
            let actual = host(&moe::fp8_down(&act, &to(&cpu_routing), slots, &gw2, &gs2).unwrap());
            assert_close(&actual, &expected, 0.01, &format!("fused MoE T={tokens}"));
        }
    }
}

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

// ---- Gated DeltaNet single-token decode; see the CUDA kernels. ----
template <typename T>
inline void q36m_gdn_conv(
    device const T* x, device const float* w, device const float* h0, device const float* h1,
    device const float* h2, device float* out, device float* history, uint conv_dim, uint c) {
    if (c >= conv_dim) {
        return;
    }
    const float xc = float(x[c]);
    device const float* wc = w + (ulong)c * 4;
    const float p1 = h1[c], p2 = h2[c];
    float v = xc * wc[3];
    v = v + h0[c] * wc[0];
    v = v + p1 * wc[1];
    v = v + p2 * wc[2];
    out[c] = v / (1.0f + exp(-v));
    // The next step's history, oldest first: [h1, h2, x].
    history[c] = p1;
    history[(ulong)conv_dim + c] = p2;
    history[2 * (ulong)conv_dim + c] = xc;
}

kernel void q36m_gdn_conv_f16(
    device const half* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* h0 [[buffer(2)]], device const float* h1 [[buffer(3)]],
    device const float* h2 [[buffer(4)]], device float* out [[buffer(5)]],
    device float* history [[buffer(6)]], constant uint& conv_dim [[buffer(7)]],
    uint c [[thread_position_in_grid]]) {
    q36m_gdn_conv<half>(x, w, h0, h1, h2, out, history, conv_dim, c);
}

kernel void q36m_gdn_conv_f32(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* h0 [[buffer(2)]], device const float* h1 [[buffer(3)]],
    device const float* h2 [[buffer(4)]], device float* out [[buffer(5)]],
    device float* history [[buffer(6)]], constant uint& conv_dim [[buffer(7)]],
    uint c [[thread_position_in_grid]]) {
    q36m_gdn_conv<float>(x, w, h0, h1, h2, out, history, conv_dim, c);
}

// One 512-thread threadgroup per value head (128-dim heads).
template <typename T>
inline void q36m_gdn_decode(
    device const float* conv, device const T* z, device const T* beta_raw,
    device const T* alpha, device const float* dt_bias, device const float* a,
    device const float* norm_w, device const float* state_in, device float* state_out,
    device T* y, uint key_heads, uint value_heads, uint grouped, float norm_eps,
    uint h, uint tid, uint warp, uint lane,
    threadgroup float* qs, threadgroup float* ks, threadgroup float* red,
    threadgroup float* part, threadgroup float* stats) {
    const uint repeats = value_heads / key_heads;
    const uint kh = grouped != 0 ? h / repeats : h % key_heads;
    const uint col = (warp & 3) * 32 + lane;
    const uint rg = warp >> 2;
    const uint key_width = key_heads * 128;
    if (tid < 128) {
        qs[tid] = conv[kh * 128 + tid];
        ks[tid] = conv[key_width + kh * 128 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp < 8) {
        const float val = warp < 4 ? qs[warp * 32 + lane] : ks[(warp - 4) * 32 + lane];
        const float sq = simd_sum(val * val);
        if (lane == 0) {
            part[warp] = sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float qsum = part[0] + part[1] + part[2] + part[3];
        const float ksum = part[4] + part[5] + part[6] + part[7];
        stats[0] = 1.0f / (sqrt(qsum + 1e-6f) * sqrt(128.0f));
        stats[1] = 1.0f / sqrt(ksum + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qscale = stats[0];
    const float knorm = stats[1];
    const float gate_in = float(alpha[h]) + dt_bias[h];
    const float softplus = max(gate_in, 0.0f) + log(1.0f + exp(-fabs(gate_in)));
    const float decay = exp(softplus * a[h]);
    const float beta = 1.0f / (1.0f + exp(-float(beta_raw[h])));
    const ulong row0 = (ulong)h * 128 + (ulong)rg * 32;
    device const float* sin_ = state_in + row0 * 128 + col;
    float s[32];
    float recalled = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        s[i] = sin_[(ulong)i * 128] * decay;
        recalled = fma(ks[rg * 32 + i] * knorm, s[i], recalled);
    }
    red[rg * 128 + col] = recalled;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kv = red[col] + red[128 + col] + red[256 + col] + red[384 + col];
    const float delta = (conv[2 * key_width + h * 128 + col] - kv) * beta;
    device float* sout = state_out + row0 * 128 + col;
    float o = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        const float updated = fma(ks[rg * 32 + i] * knorm, delta, s[i]);
        sout[(ulong)i * 128] = updated;
        o = fma(qs[rg * 32 + i] * qscale, updated, o);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[rg * 128 + col] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float out = 0.0f;
    if (rg == 0) {
        out = red[col] + red[128 + col] + red[256 + col] + red[384 + col];
        const float sq = simd_sum(out * out);
        if (lane == 0) {
            part[warp] = sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg == 0) {
        const float mean_sq = (part[0] + part[1] + part[2] + part[3]) / 128.0f;
        const float zz = float(z[h * 128 + col]);
        const float gate = zz / (1.0f + exp(-zz));
        y[h * 128 + col] = T(out / sqrt(mean_sq + norm_eps) * norm_w[col] * gate);
    }
}

#define Q36M_GDN_DECODE(T, S)                                                                      \
kernel void q36m_gdn_decode_##S(                                                                   \
    device const float* conv [[buffer(0)]], device const T* z [[buffer(1)]],                      \
    device const T* beta_raw [[buffer(2)]], device const T* alpha [[buffer(3)]],                  \
    device const float* dt_bias [[buffer(4)]], device const float* a [[buffer(5)]],               \
    device const float* norm_w [[buffer(6)]], device const float* state_in [[buffer(7)]],         \
    device float* state_out [[buffer(8)]], device T* y [[buffer(9)]],                             \
    constant uint& key_heads [[buffer(10)]], constant uint& value_heads [[buffer(11)]],           \
    constant uint& grouped [[buffer(12)]], constant float& norm_eps [[buffer(13)]],               \
    uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],            \
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {      \
    threadgroup float qs[128];                                                                     \
    threadgroup float ks[128];                                                                     \
    threadgroup float red[512];                                                                    \
    threadgroup float part[16];                                                                    \
    threadgroup float stats[2];                                                                    \
    q36m_gdn_decode<T>(conv, z, beta_raw, alpha, dt_bias, a, norm_w, state_in, state_out, y,      \
                       key_heads, value_heads, grouped, norm_eps, h, tid, warp, lane, qs, ks,      \
                       red, part, stats);                                                          \
}

Q36M_GDN_DECODE(half, f16)
Q36M_GDN_DECODE(float, f32)

// ---- RMSNorm with an F32 gain (+ optional residual add); one threadgroup per row. ----
template <typename T>
inline void q36m_rms_norm(
    device const T* x, device const T* residual, device const float* w, device T* sum_out,
    device T* out, uint hidden, float eps, uint row, uint tid, uint threads, uint warp,
    uint lane, threadgroup float* part, threadgroup float* inv_shared) {
    const ulong base = (ulong)row * hidden;
    float ss = 0.0f;
    for (uint i = tid; i < hidden; i += threads) {
        float v = float(x[base + i]);
        if (residual) {
            const T sum = T(float(residual[base + i]) + v);
            if (sum_out) {
                sum_out[base + i] = sum;
            }
            v = float(sum);
        }
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        part[warp] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (threads + 31) / 32; ++i) {
            total += part[i];
        }
        inv_shared[0] = 1.0f / sqrt(total / float(hidden) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_shared[0];
    for (uint i = tid; i < hidden; i += threads) {
        float v = float(x[base + i]);
        if (residual) {
            v = float(T(float(residual[base + i]) + v));
        }
        out[base + i] = T(v * inv * w[i]);
    }
}

kernel void q36m_rms_norm_f16(
    device const half* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint& hidden [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float inv_shared[1];
    q36m_rms_norm<half>(x, nullptr, w, nullptr, out, hidden, eps, row, tid, threads, warp, lane,
                        part, inv_shared);
}

kernel void q36m_add_rms_norm_f16(
    device const half* x [[buffer(0)]], device const half* residual [[buffer(1)]],
    device const float* w [[buffer(2)]], device half* sum_out [[buffer(3)]],
    device half* out [[buffer(4)]], constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    threadgroup float inv_shared[1];
    q36m_rms_norm<half>(x, residual, w, sum_out, out, hidden, eps, row, tid, threads, warp, lane,
                        part, inv_shared);
}

// ---- q/k head RMSNorm + partial rotate-half M-RoPE for one token. ----
kernel void q36m_qk_norm_rope_f16(
    device const half* q_src [[buffer(0)]], device const half* k_src [[buffer(1)]],
    device const float* q_gain [[buffer(2)]], device const float* k_gain [[buffer(3)]],
    device const float* inv_freq [[buffer(4)]], device half* q_out [[buffer(5)]],
    device half* k_out [[buffer(6)]], constant uint& num_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]], constant uint& rope_dim [[buffer(9)]],
    constant uint& pos_t [[buffer(10)]], constant uint& pos_h [[buffer(11)]],
    constant uint& pos_w [[buffer(12)]], constant uint& sec_h [[buffer(13)]],
    constant uint& sec_w [[buffer(14)]], constant float& eps [[buffer(15)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float vals[1024];
    threadgroup float part[32];
    threadgroup float inv_shared[1];
    const bool is_query = head < num_heads;
    device const half* src = is_query ? q_src + (ulong)head * 2 * head_dim
                                      : k_src + (ulong)(head - num_heads) * head_dim;
    device const float* gain = is_query ? q_gain : k_gain;
    device half* dst = is_query ? q_out + (ulong)head * head_dim
                                : k_out + (ulong)(head - num_heads) * head_dim;
    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += threads) {
        const float v = float(src[i]);
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        part[warp] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (threads + 31) / 32; ++i) {
            total += part[i];
        }
        inv_shared[0] = 1.0f / sqrt(total / float(head_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_shared[0];
    for (uint i = tid; i < head_dim; i += threads) {
        vals[i] = float(half(float(src[i]) * inv * gain[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_dim = rope_dim >> 1;
    const bool sectioned = pos_t != pos_h || pos_t != pos_w;
    for (uint i = tid; i < head_dim; i += threads) {
        float out = vals[i];
        if (i < rope_dim) {
            const uint j = i < half_dim ? i : i - half_dim;
            uint pos = pos_t;
            if (sectioned) {
                if (j % 3 == 1 && j < 3 * sec_h) {
                    pos = pos_h;
                } else if (j % 3 == 2 && j < 3 * sec_w) {
                    pos = pos_w;
                }
            }
            const float angle = float(pos) * inv_freq[j];
            const float c = float(half(precise::cos(angle)));
            const float sn = float(half(precise::sin(angle)));
            out = i < half_dim ? vals[i] * c - vals[i + half_dim] * sn
                               : vals[i - half_dim] * sn + vals[i] * c;
        }
        dst[i] = half(out);
    }
}

// ---- Gated attention output: attn * sigmoid(gate), gate from the gated q_proj. ----
kernel void q36m_attn_gate_f16(
    device const half* attn [[buffer(0)]], device const half* q_proj [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]], constant uint& total [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    uint width = heads * head_dim;
    uint row = i / width, j = i - row * width;
    uint head = j / head_dim, d = j - head * head_dim;
    float g = float(q_proj[row * 2 * width + head * 2 * head_dim + head_dim + d]);
    half s = half(1.0f / (1.0f + precise::exp(-g)));
    out[i] = half(float(attn[i]) * float(s));
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

fn f16_or_f32(dtype: DType, name: &str) -> Result<&'static str> {
    match dtype {
        DType::F16 => Ok("f16"),
        DType::F32 => Ok("f32"),
        other => bail!("Metal {name} does not support {other:?}"),
    }
}

fn kernel_name(prefix: &str, suffix: &str) -> &'static str {
    match (prefix, suffix) {
        ("q36m_gdn_conv", "f16") => "q36m_gdn_conv_f16",
        ("q36m_gdn_conv", _) => "q36m_gdn_conv_f32",
        ("q36m_gdn_decode", "f16") => "q36m_gdn_decode_f16",
        (_, _) => "q36m_gdn_decode_f32",
    }
}

/// DeltaNet causal-conv decode step (see `kernels::cuda::gdn::conv_decode`).
pub(crate) fn gdn_conv(
    x: &Tensor,
    weight: &Tensor,
    history: [&Tensor; 3],
    conv_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(x)?;
    let suffix = f16_or_f32(x.dtype(), "DeltaNet conv")?;
    let (x, weight) = (x.contiguous()?, weight.contiguous()?);
    let h = [
        history[0].contiguous()?,
        history[1].contiguous()?,
        history[2].contiguous()?,
    ];
    let xb = bind(&x, "conv input", 2)?;
    let wb = bind(&weight, "conv weight", 4)?;
    let hb = [
        bind(&h[0], "history", 4)?,
        bind(&h[1], "history", 4)?,
        bind(&h[2], "history", 4)?,
    ];
    let out = device.new_buffer(conv_dim, DType::F32, "q36m-gdn-conv")?;
    let next = device.new_buffer(3 * conv_dim, DType::F32, "q36m-gdn-history")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-gdn-conv");
    encoder.set_compute_pipeline_state(&pipeline(
        device.metal_device(),
        kernel_name("q36m_gdn_conv", suffix),
    )?);
    encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
    encoder.set_input_buffer(1, Some(wb.buffer()?), wb.offset);
    for (index, bound) in hb.iter().enumerate() {
        encoder.set_input_buffer(2 + index, Some(bound.buffer()?), bound.offset);
    }
    encoder.set_output_buffer(5, Some(&out), 0);
    encoder.set_output_buffer(6, Some(&next), 0);
    encoder.set_bytes(7, &u32_arg(conv_dim, "conv dim")?);
    encoder.dispatch_threads(grid(conv_dim, 1), grid(256, 1));
    drop(encoder);
    drop((xb, wb, hb));
    Ok((
        wrap(&device, out, Shape::from(conv_dim), DType::F32),
        wrap(&device, next, Shape::from((3, conv_dim)), DType::F32),
    ))
}

/// DeltaNet recurrent decode step (see `kernels::cuda::gdn::recurrent_decode`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_decode(
    conv: &Tensor,
    z: &Tensor,
    beta_raw: &Tensor,
    alpha: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    norm_weight: &Tensor,
    state: &Tensor,
    spec: &crate::kernels::cuda::gdn::GdnDecodeSpec,
) -> Result<(Tensor, Tensor)> {
    let device = metal_device(conv)?;
    let dtype = z.dtype();
    let suffix = f16_or_f32(dtype, "DeltaNet decode")?;
    let d = crate::kernels::cuda::gdn::HEAD_DIM;
    let hv = spec.value_heads;
    let tensors =
        [conv, z, beta_raw, alpha, dt_bias, a, norm_weight, state].map(Tensor::contiguous);
    let [conv, z, beta_raw, alpha, dt_bias, a, norm_weight, state] = tensors;
    let (conv, z, beta_raw, alpha) = (conv?, z?, beta_raw?, alpha?);
    let (dt_bias, a, norm_weight, state) = (dt_bias?, a?, norm_weight?, state?);
    let bound = [
        bind(&conv, "conv", 4)?,
        bind(&z, "z", 2)?,
        bind(&beta_raw, "beta", 2)?,
        bind(&alpha, "alpha", 2)?,
        bind(&dt_bias, "dt_bias", 4)?,
        bind(&a, "a", 4)?,
        bind(&norm_weight, "norm", 4)?,
        bind(&state, "state", 4)?,
    ];
    let next = device.new_buffer(hv * d * d, DType::F32, "q36m-gdn-state")?;
    let y = device.new_buffer(hv * d, dtype, "q36m-gdn-y")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-gdn-decode");
    encoder.set_compute_pipeline_state(&pipeline(
        device.metal_device(),
        kernel_name("q36m_gdn_decode", suffix),
    )?);
    for (index, b) in bound.iter().enumerate() {
        encoder.set_input_buffer(index, Some(b.buffer()?), b.offset);
    }
    encoder.set_output_buffer(8, Some(&next), 0);
    encoder.set_output_buffer(9, Some(&y), 0);
    encoder.set_bytes(10, &u32_arg(spec.key_heads, "key heads")?);
    encoder.set_bytes(11, &u32_arg(hv, "value heads")?);
    encoder.set_bytes(12, &u32::from(spec.grouped));
    encoder.set_bytes(13, &spec.norm_eps);
    encoder.dispatch_thread_groups(grid(hv, 1), grid(512, 1));
    drop(encoder);
    drop(bound);
    Ok((
        wrap(&device, y, Shape::from(hv * d), dtype),
        wrap(&device, next, Shape::from((1, hv, d, d)), DType::F32),
    ))
}

/// RMSNorm with an F32 gain over F16 activations, optionally fused with the
/// residual add (see `kernels::cuda::norm`).
pub(crate) fn rms_norm(
    x: &Tensor,
    residual: Option<&Tensor>,
    weight: &Tensor,
    eps: f32,
    hidden: usize,
) -> Result<(Option<Tensor>, Tensor)> {
    if x.dtype() != DType::F16 {
        bail!(
            "Metal fused RMSNorm needs F16 activations, found {:?}",
            x.dtype()
        )
    }
    let device = metal_device(x)?;
    let shape = x.shape().clone();
    let elements = shape.elem_count();
    let rows = elements / hidden.max(1);
    let (x, weight) = (x.contiguous()?, weight.contiguous()?);
    let residual = residual.map(Tensor::contiguous).transpose()?;
    let xb = bind(&x, "activations", 2)?;
    let wb = bind(&weight, "gain", 4)?;
    let out = device.new_buffer(elements, DType::F16, "q36m-rms-norm")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-rms-norm");
    let sum = match &residual {
        None => {
            encoder
                .set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_rms_norm_f16")?);
            encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
            encoder.set_input_buffer(1, Some(wb.buffer()?), wb.offset);
            encoder.set_output_buffer(2, Some(&out), 0);
            encoder.set_bytes(3, &u32_arg(hidden, "hidden")?);
            encoder.set_bytes(4, &eps);
            None
        }
        Some(residual) => {
            let rb = bind(residual, "residual", 2)?;
            let sum = device.new_buffer(elements, DType::F16, "q36m-residual-sum")?;
            encoder.set_compute_pipeline_state(&pipeline(
                device.metal_device(),
                "q36m_add_rms_norm_f16",
            )?);
            encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
            encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
            encoder.set_input_buffer(2, Some(wb.buffer()?), wb.offset);
            encoder.set_output_buffer(3, Some(&sum), 0);
            encoder.set_output_buffer(4, Some(&out), 0);
            encoder.set_bytes(5, &u32_arg(hidden, "hidden")?);
            encoder.set_bytes(6, &eps);
            Some(sum)
        }
    };
    encoder.dispatch_thread_groups(grid(rows, 1), grid(256, 1));
    drop(encoder);
    drop((xb, wb));
    Ok((
        sum.map(|sum| wrap(&device, sum, shape.clone(), DType::F16)),
        wrap(&device, out, shape, DType::F16),
    ))
}

/// Gated attention output (see `kernels::cuda::gate`): `attn` `[.., heads *
/// head_dim]` and the gated `q_proj` `[.., heads, 2 * head_dim]`, both F16.
pub(crate) fn attn_gate(
    attn: &Tensor,
    q_proj: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if attn.dtype() != DType::F16 || q_proj.dtype() != DType::F16 {
        bail!(
            "Metal attention gate needs F16 operands, found {:?} and {:?}",
            attn.dtype(),
            q_proj.dtype()
        )
    }
    let device = metal_device(attn)?;
    let shape = attn.shape().clone();
    let total = shape.elem_count();
    let (attn, q_proj) = (attn.contiguous()?, q_proj.contiguous()?);
    let ab = bind(&attn, "attention output", 2)?;
    let qb = bind(&q_proj, "gated query projection", 2)?;
    let out = device.new_buffer(total, DType::F16, "q36m-attn-gate")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-attn-gate");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_attn_gate_f16")?);
    encoder.set_input_buffer(0, Some(ab.buffer()?), ab.offset);
    encoder.set_input_buffer(1, Some(qb.buffer()?), qb.offset);
    encoder.set_output_buffer(2, Some(&out), 0);
    encoder.set_bytes(3, &u32_arg(heads, "heads")?);
    encoder.set_bytes(4, &u32_arg(head_dim, "head_dim")?);
    encoder.set_bytes(5, &u32_arg(total, "elements")?);
    encoder.dispatch_thread_groups(grid(total.div_ceil(256), 1), grid(256, 1));
    drop(encoder);
    drop((ab, qb));
    Ok(wrap(&device, out, shape, DType::F16))
}

/// q/k head norm + partial M-RoPE for one token (see `kernels::cuda::rope`).
pub(crate) fn qk_norm_rope(
    q_proj: &Tensor,
    k_proj: &Tensor,
    q_gain: &Tensor,
    k_gain: &Tensor,
    inv_freq: &Tensor,
    spec: &crate::kernels::cuda::rope::QkNormRopeSpec,
    position: crate::kernels::cuda::rope::MropePosition,
) -> Result<(Tensor, Tensor)> {
    if q_proj.dtype() != DType::F16 || spec.head_dim > 1024 {
        bail!("Metal fused q/k norm + RoPE needs F16 activations and head_dim <= 1024")
    }
    let device = metal_device(q_proj)?;
    let d = spec.head_dim;
    let tensors = [q_proj, k_proj, q_gain, k_gain, inv_freq].map(Tensor::contiguous);
    let [q, k, qg, kg, inv] = tensors;
    let (q, k, qg, kg, inv) = (q?, k?, qg?, kg?, inv?);
    let bound = [
        bind(&q, "q_proj", 2)?,
        bind(&k, "k_proj", 2)?,
        bind(&qg, "q gain", 4)?,
        bind(&kg, "k gain", 4)?,
        bind(&inv, "inverse frequencies", 4)?,
    ];
    let q_out = device.new_buffer(spec.num_heads * d, DType::F16, "q36m-q")?;
    let k_out = device.new_buffer(spec.num_kv_heads * d, DType::F16, "q36m-k")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("q36m-qk-norm-rope");
    encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "q36m_qk_norm_rope_f16")?);
    for (index, b) in bound.iter().enumerate() {
        encoder.set_input_buffer(index, Some(b.buffer()?), b.offset);
    }
    encoder.set_output_buffer(5, Some(&q_out), 0);
    encoder.set_output_buffer(6, Some(&k_out), 0);
    encoder.set_bytes(7, &u32_arg(spec.num_heads, "heads")?);
    encoder.set_bytes(8, &u32_arg(d, "head dim")?);
    encoder.set_bytes(9, &u32_arg(spec.rope_dim, "rope dim")?);
    encoder.set_bytes(10, &u32_arg(position.temporal, "position")?);
    encoder.set_bytes(11, &u32_arg(position.height, "position")?);
    encoder.set_bytes(12, &u32_arg(position.width, "position")?);
    encoder.set_bytes(13, &u32_arg(position.height_section, "section")?);
    encoder.set_bytes(14, &u32_arg(position.width_section, "section")?);
    encoder.set_bytes(15, &spec.eps);
    encoder.dispatch_thread_groups(grid(spec.num_heads + spec.num_kv_heads, 1), grid(256, 1));
    drop(encoder);
    drop(bound);
    Ok((
        wrap(
            &device,
            q_out,
            Shape::from((1, 1, spec.num_heads, d)),
            DType::F16,
        ),
        wrap(
            &device,
            k_out,
            Shape::from((1, 1, spec.num_kv_heads, d)),
            DType::F16,
        ),
    ))
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
    fn metal_gdn_decode_matches_the_cpu_reference_at_trunk_geometry() {
        use crate::kernels::cuda::gdn::{conv_decode, recurrent_decode, GdnDecodeSpec, HEAD_DIM};
        let Some(gpu) = device() else { return };
        let d = HEAD_DIM;
        for grouped in [true, false] {
            let spec = GdnDecodeSpec {
                key_heads: 16,
                value_heads: 32,
                grouped,
                norm_eps: 1e-6,
            };
            let conv_dim = spec.conv_dim();
            let t =
                |v: Vec<f32>, shape: &[usize]| Tensor::from_vec(v, shape, &Device::Cpu).unwrap();
            let x = t(wave(conv_dim, 1.0, 2.0), &[1, 1, conv_dim])
                .to_dtype(DType::F16)
                .unwrap();
            let w = t(wave(conv_dim * 4, 2.0, 0.5), &[conv_dim, 4]);
            let h: Vec<Tensor> = (0..3)
                .map(|i| t(wave(conv_dim, 9.0 + i as f32, 1.0), &[conv_dim, 1]))
                .collect();
            let z = t(wave(32 * d, 5.0, 2.0), &[32 * d])
                .to_dtype(DType::F16)
                .unwrap();
            let beta = t(wave(32, 6.0, 2.0), &[32]).to_dtype(DType::F16).unwrap();
            let alpha = t(wave(32, 7.0, 3.0), &[32]).to_dtype(DType::F16).unwrap();
            let dt = t(wave(32, 8.0, 1.0), &[32]);
            let a = t(wave(32, 9.0, 1.0).iter().map(|v| -v.exp()).collect(), &[32]);
            let norm = t(wave(d, 10.0, 0.2).iter().map(|v| 1.0 + v).collect(), &[d]);
            let state = t(wave(32 * d * d, 11.0, 0.3), &[1, 32, d, d]);
            let (cpu_conv, cpu_cur) = conv_decode(&x, &w, [&h[0], &h[1], &h[2]]).unwrap();
            let (cpu_y, cpu_next) =
                recurrent_decode(&cpu_conv, &z, &beta, &alpha, &dt, &a, &norm, &state, &spec)
                    .unwrap();
            let g = |t: &Tensor| t.to_device(&gpu).unwrap();
            let (gpu_conv, gpu_cur) =
                conv_decode(&g(&x), &g(&w), [&g(&h[0]), &g(&h[1]), &g(&h[2])]).unwrap();
            let (gpu_y, gpu_next) = recurrent_decode(
                &gpu_conv,
                &g(&z),
                &g(&beta),
                &g(&alpha),
                &g(&dt),
                &g(&a),
                &g(&norm),
                &g(&state),
                &spec,
            )
            .unwrap();
            assert_close(&host(&gpu_conv), &host(&cpu_conv), 1e-5, "conv");
            assert_eq!(host(&gpu_cur), host(&cpu_cur), "conv history");
            assert_close(
                &host(&gpu_y),
                &host(&cpu_y),
                1e-2,
                &format!("y grouped={grouped}"),
            );
            assert_close(
                &host(&gpu_next),
                &host(&cpu_next),
                1e-5,
                &format!("state grouped={grouped}"),
            );
        }
    }

    #[test]
    fn metal_rms_norm_matches_the_composition() {
        use crate::kernels::cuda::norm::{add_rms_norm, rms_norm};
        let Some(gpu) = device() else { return };
        for (rows, hidden) in [(1usize, 2048usize), (5, 2048), (16, 256)] {
            let mk = |seed: f32| {
                Tensor::from_vec(wave(rows * hidden, seed, 3.0), (rows, hidden), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap()
            };
            let (r, dl) = (mk(1.0), mk(2.0));
            let w = Tensor::from_vec(
                wave(hidden, 3.0, 0.3)
                    .iter()
                    .map(|v| 1.0 + v)
                    .collect::<Vec<_>>(),
                hidden,
                &Device::Cpu,
            )
            .unwrap();
            let (cpu_sum, cpu_out) = add_rms_norm(&r, &dl, &w, 1e-6).unwrap();
            let g = |t: &Tensor| t.to_device(&gpu).unwrap();
            let (gpu_sum, gpu_out) = add_rms_norm(&g(&r), &g(&dl), &g(&w), 1e-6).unwrap();
            assert_eq!(host(&gpu_sum), host(&cpu_sum), "sum rows={rows}");
            assert_close(
                &host(&gpu_out),
                &host(&cpu_out),
                4e-3,
                &format!("add+norm rows={rows}"),
            );
            assert_close(
                &host(&rms_norm(&g(&r), &g(&w), 1e-6).unwrap()),
                &host(&rms_norm(&r, &w, 1e-6).unwrap()),
                4e-3,
                &format!("norm rows={rows}"),
            );
        }
    }

    #[test]
    fn metal_attn_gate_matches_the_composition() {
        use crate::kernels::cuda::gate::{attn_gate, reference};
        let Some(gpu) = device() else { return };
        for (rows, heads, head_dim) in [(1usize, 16usize, 256usize), (3, 2, 100)] {
            let width = heads * head_dim;
            let attn =
                Tensor::from_vec(wave(rows * width, 1.0, 4.0), (rows, 1, width), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
            let q_proj = Tensor::from_vec(
                wave(rows * 2 * width, 2.0, 8.0),
                (rows, 1, heads, 2 * head_dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
            let expected = reference(&attn, &q_proj, heads, head_dim).unwrap();
            let g = |t: &Tensor| t.to_device(&gpu).unwrap();
            let actual = attn_gate(&g(&attn), &g(&q_proj), heads, head_dim).unwrap();
            assert_eq!(actual.dims(), attn.dims());
            assert_close(
                &host(&actual),
                &host(&expected),
                4e-3,
                &format!("gate rows={rows}"),
            );
        }
    }

    #[test]
    fn metal_qk_norm_rope_matches_the_cpu_reference() {
        use crate::kernels::cuda::rope::{qk_norm_rope, MropePosition, QkNormRopeSpec};
        let Some(gpu) = device() else { return };
        let spec = QkNormRopeSpec {
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 256,
            rope_dim: 64,
            eps: 1e-6,
        };
        let q = Tensor::from_vec(wave(16 * 512, 1.0, 2.0), (1, 1, 16, 512), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k = Tensor::from_vec(wave(2 * 256, 2.0, 2.0), (1, 1, 2, 256), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let qg = Tensor::from_vec(
            wave(256, 3.0, 0.2)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            256,
            &Device::Cpu,
        )
        .unwrap();
        let kg = Tensor::from_vec(
            wave(256, 4.0, 0.2)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            256,
            &Device::Cpu,
        )
        .unwrap();
        let inv = Tensor::from_vec(
            (0..32)
                .map(|j| 10_000_000f32.powf(-2.0 * j as f32 / 64.0))
                .collect::<Vec<_>>(),
            32,
            &Device::Cpu,
        )
        .unwrap();
        let g = |t: &Tensor| t.to_device(&gpu).unwrap();
        for (t, hh, w) in [
            (5usize, 5usize, 5usize),
            (100_000, 100_000, 100_000),
            (7, 3, 11),
        ] {
            let position = MropePosition {
                temporal: t,
                height: hh,
                width: w,
                height_section: 11,
                width_section: 10,
            };
            let (cq, ck) = qk_norm_rope(&q, &k, &qg, &kg, &inv, &spec, position).unwrap();
            let (gq, gk) =
                qk_norm_rope(&g(&q), &g(&k), &g(&qg), &g(&kg), &g(&inv), &spec, position).unwrap();
            assert_close(&host(&gq), &host(&cq), 4e-3, &format!("q at {t}"));
            assert_close(&host(&gk), &host(&ck), 4e-3, &format!("k at {t}"));
        }
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

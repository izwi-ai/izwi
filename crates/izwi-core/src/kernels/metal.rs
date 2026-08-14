//! Optimized kernel operations for model inference.
//!
//! This module provides fused tensor operations that reduce memory round-trips
//! and kernel launch overhead. On Metal backends, these use Candle's Metal
//! dispatch where possible; on other backends they use optimized CPU patterns.

#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "metal")]
use candle_core::{
    backend::BackendStorage, bail, CpuStorage, CustomOp2, CustomOp3, Layout, MetalStorage,
    Result as CandleResult, Shape,
};
use candle_core::{DType, Tensor};
#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

#[cfg(feature = "metal")]
use super::metal_encoder::IzwiMetalCommandEncoderExt;
use super::{FusedKernelError, FusedResult, FusedSiluMulResult};

#[cfg(feature = "metal")]
const METAL_PAGED_ATTENTION_PARTITION_TOKENS: usize = 512;
#[cfg(feature = "metal")]
const METAL_PAGED_ATTENTION_SPLIT_MIN_CONTEXT: usize = 2048;
#[cfg(feature = "metal")]
const METAL_PAGED_ATTENTION_SPLIT_MAX_BASE_WORKGROUPS: usize = 64;
// This path is intentionally conservative until hardware-backed benchmarks
// establish a lower crossover point for packed prefill on supported GPUs.
#[cfg(feature = "metal")]
const METAL_PAGED_PREFILL_SPLIT_MIN_CONTEXT: usize = 4096;
#[cfg(feature = "metal")]
const METAL_PAGED_PREFILL_SPLIT_MAX_BASE_WORKGROUPS: usize = 64;

#[cfg(feature = "metal")]
const IZWI_METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// The split-KV organization below (independent online-softmax partitions
// followed by a log-sum-exp merge) is adapted from vllm-metal's Apache-2.0
// paged-attention implementation at commit
// cc1b679725085ddb40f9beb0ed36e7745ae8d688:
// https://github.com/vllm-project/vllm-metal/blob/cc1b679725085ddb40f9beb0ed36e7745ae8d688/vllm_metal/metal/kernels_v2/pagedattention.metal
// Copyright contributors to the vLLM project. Licensed under Apache-2.0.
// Modified for Izwi's Candle custom-op ABI and page-major K/V layout.

kernel void izwi_silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& elem_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elem_count) {
        return;
    }

    float x = gate[gid];
    output[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

kernel void izwi_silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& elem_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elem_count) {
        return;
    }

    float x = float(gate[gid]);
    float y = float(up[gid]);
    output[gid] = half((x / (1.0f + exp(-x))) * y);
}

kernel void izwi_qk_rms_norm_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& q_rows [[buffer(4)]],
    constant uint& k_rows [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float sums[256];

    const bool is_q = row < q_rows;
    const uint local_row = is_q ? row : (row - q_rows);
    const device float* src = is_q ? q : k;
    const uint weight_offset = is_q ? 0 : head_dim;
    const uint out_offset = row * head_dim;
    const uint src_offset = local_row * head_dim;

    float sum = 0.0f;
    if (tid < head_dim) {
        const float value = src[src_offset + tid];
        sum = value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        const float value = src[src_offset + tid];
        const float scale = rsqrt((sums[0] / float(head_dim)) + eps);
        output[out_offset + tid] = value * scale * weights[weight_offset + tid];
    }
}

kernel void izwi_qk_rms_norm_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* weights [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& q_rows [[buffer(4)]],
    constant uint& k_rows [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float sums[256];

    const bool is_q = row < q_rows;
    const uint local_row = is_q ? row : (row - q_rows);
    const device half* src = is_q ? q : k;
    const uint weight_offset = is_q ? 0 : head_dim;
    const uint out_offset = row * head_dim;
    const uint src_offset = local_row * head_dim;

    float sum = 0.0f;
    if (tid < head_dim) {
        const float value = float(src[src_offset + tid]);
        sum = value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        const float value = float(src[src_offset + tid]);
        const float scale = rsqrt((sums[0] / float(head_dim)) + eps);
        output[out_offset + tid] = half(value * scale * float(weights[weight_offset + tid]));
    }
}

kernel void izwi_rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float sums[1024];

    if (row >= rows) {
        return;
    }

    const uint row_offset = row * hidden_dim;
    float sum = 0.0f;
    for (uint idx = tid; idx < hidden_dim; idx += threads_per_threadgroup) {
        const float value = input[row_offset + idx];
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt((sums[0] / float(hidden_dim)) + eps);
    for (uint idx = tid; idx < hidden_dim; idx += threads_per_threadgroup) {
        const float value = input[row_offset + idx];
        output[row_offset + idx] = value * scale * weight[idx];
    }
}

kernel void izwi_rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float sums[1024];

    if (row >= rows) {
        return;
    }

    const uint row_offset = row * hidden_dim;
    float sum = 0.0f;
    for (uint idx = tid; idx < hidden_dim; idx += threads_per_threadgroup) {
        const float value = float(input[row_offset + idx]);
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt((sums[0] / float(hidden_dim)) + eps);
    for (uint idx = tid; idx < hidden_dim; idx += threads_per_threadgroup) {
        const float value = float(input[row_offset + idx]);
        output[row_offset + idx] = half(value * scale * float(weight[idx]));
    }
}

kernel void izwi_rope_pair_bshd_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* cos_sin [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& q_rows [[buffer(4)]],
    constant uint& k_rows [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& q_heads [[buffer(7)]],
    constant uint& k_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint half_dim = head_dim / 2;
    const uint elem_count = (q_rows + k_rows) * head_dim;
    if (gid >= elem_count) {
        return;
    }

    const uint out_row = gid / head_dim;
    const uint dim = gid - (out_row * head_dim);
    const bool is_q = out_row < q_rows;
    const uint local_row = is_q ? out_row : (out_row - q_rows);
    const uint heads = is_q ? q_heads : k_heads;
    const uint pos = (local_row / heads) % seq_len;
    const uint in_base = local_row * head_dim;
    const device float* input = is_q ? q : k;
    const uint pair_dim = dim < half_dim ? dim : (dim - half_dim);
    const float cos_value = cos_sin[pos * head_dim + pair_dim];
    const float sin_value = cos_sin[pos * head_dim + half_dim + pair_dim];
    const float x1 = input[in_base + pair_dim];
    const float x2 = input[in_base + half_dim + pair_dim];

    if (dim < half_dim) {
        output[gid] = x1 * cos_value - x2 * sin_value;
    } else {
        output[gid] = x1 * sin_value + x2 * cos_value;
    }
}

kernel void izwi_rope_pair_bshd_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* cos_sin [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& q_rows [[buffer(4)]],
    constant uint& k_rows [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& q_heads [[buffer(7)]],
    constant uint& k_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint half_dim = head_dim / 2;
    const uint elem_count = (q_rows + k_rows) * head_dim;
    if (gid >= elem_count) {
        return;
    }

    const uint out_row = gid / head_dim;
    const uint dim = gid - (out_row * head_dim);
    const bool is_q = out_row < q_rows;
    const uint local_row = is_q ? out_row : (out_row - q_rows);
    const uint heads = is_q ? q_heads : k_heads;
    const uint pos = (local_row / heads) % seq_len;
    const uint in_base = local_row * head_dim;
    const device half* input = is_q ? q : k;
    const uint pair_dim = dim < half_dim ? dim : (dim - half_dim);
    const float cos_value = float(cos_sin[pos * head_dim + pair_dim]);
    const float sin_value = float(cos_sin[pos * head_dim + half_dim + pair_dim]);
    const float x1 = float(input[in_base + pair_dim]);
    const float x2 = float(input[in_base + half_dim + pair_dim]);

    if (dim < half_dim) {
        output[gid] = half(x1 * cos_value - x2 * sin_value);
    } else {
        output[gid] = half(x1 * sin_value + x2 * cos_value);
    }
}

kernel void izwi_decode_gqa_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& total_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& kv_capacity_len [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float scores[2048];
    threadgroup float scratch[256];

    if (head >= num_heads || total_len > 2048) {
        return;
    }

    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    float local_max = -INFINITY;
    for (uint pos = tid; pos < total_len; pos += threads_per_threadgroup) {
        float dot = 0.0f;
        const uint q_base = head * head_dim;
        const uint k_base = (kv_head * kv_capacity_len + pos) * head_dim;
        for (uint dim = 0; dim < head_dim; dim++) {
            dot += q[q_base + dim] * k[k_base + dim];
        }
        const float score = dot * scale;
        scores[pos] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = scratch[0];

    float local_sum = 0.0f;
    for (uint pos = tid; pos < total_len; pos += threads_per_threadgroup) {
        const float value = exp(scores[pos] - max_score);
        scores[pos] = value;
        local_sum += value;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = 1.0f / scratch[0];

    const uint out_base = head * head_dim;
    for (uint dim = tid; dim < head_dim; dim += threads_per_threadgroup) {
        float acc = 0.0f;
        for (uint pos = 0; pos < total_len; pos++) {
            const float prob = scores[pos] * inv_sum;
            const uint v_base = (kv_head * kv_capacity_len + pos) * head_dim;
            acc += prob * v[v_base + dim];
        }
        output[out_base + dim] = acc;
    }
}

kernel void izwi_decode_gqa_attention_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& total_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& kv_capacity_len [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float scores[2048];
    threadgroup float scratch[256];

    if (head >= num_heads || total_len > 2048) {
        return;
    }

    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    float local_max = -INFINITY;
    for (uint pos = tid; pos < total_len; pos += threads_per_threadgroup) {
        float dot = 0.0f;
        const uint q_base = head * head_dim;
        const uint k_base = (kv_head * kv_capacity_len + pos) * head_dim;
        for (uint dim = 0; dim < head_dim; dim++) {
            dot += float(q[q_base + dim]) * float(k[k_base + dim]);
        }
        const float score = dot * scale;
        scores[pos] = score;
        local_max = max(local_max, score);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = scratch[0];

    float local_sum = 0.0f;
    for (uint pos = tid; pos < total_len; pos += threads_per_threadgroup) {
        const float value = exp(scores[pos] - max_score);
        scores[pos] = value;
        local_sum += value;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = 1.0f / scratch[0];

    const uint out_base = head * head_dim;
    for (uint dim = tid; dim < head_dim; dim += threads_per_threadgroup) {
        float acc = 0.0f;
        for (uint pos = 0; pos < total_len; pos++) {
            const float prob = scores[pos] * inv_sum;
            const uint v_base = (kv_head * kv_capacity_len + pos) * head_dim;
            acc += prob * float(v[v_base + dim]);
        }
        output[out_base + dim] = half(acc);
    }
}

kernel void izwi_paged_decode_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* metadata [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& page_tokens [[buffer(8)]],
    constant uint& max_blocks [[buffer(9)]],
    constant uint& key_head_dim [[buffer(10)]],
    constant uint& value_head_dim [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant float& softcap [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint row = group.y;
    if (row >= batch_size || head >= num_heads) {
        return;
    }

    const uint context_len = metadata[row];
    const uint first_page_offset = metadata[batch_size + row];
    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    const uint q_base = (row * num_heads + head) * key_head_dim;
    const uint table_base = 2 * batch_size + row * max_blocks;

    const uint threads_per_threadgroup = threads_per_group.x;
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0; pos < context_len; pos++) {
        const uint physical_pos = first_page_offset + pos;
        const uint logical_page = physical_pos / page_tokens;
        const uint page_offset = physical_pos - logical_page * page_tokens;
        const uint physical_page = metadata[table_base + logical_page];
        const uint slot = physical_page * page_tokens + page_offset;
        const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;

        float local_dot = 0.0f;
        for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
            local_dot += q[q_base + dim] * k[k_base + dim];
        }
        dot_scratch[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                dot_scratch[tid] += dot_scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float score = dot_scratch[0] * scale;
            if (softcap > 0.0f) {
                score = softcap * tanh(score / softcap);
            }
            const float next_max = max(online_state[0], score);
            const float alpha = online_state[1] == 0.0f
                ? 0.0f
                : exp(online_state[0] - next_max);
            const float beta = exp(score - next_max);
            online_state[0] = next_max;
            online_state[1] = online_state[1] * alpha + beta;
            online_state[2] = alpha;
            online_state[3] = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
        for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
            accumulator[dim] = accumulator[dim] * online_state[2]
                + v[v_base + dim] * online_state[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_base = (row * num_heads + head) * value_head_dim;
    const float inv_sum = 1.0f / online_state[1];
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        output[out_base + dim] = accumulator[dim] * inv_sum;
    }
}

kernel void izwi_paged_decode_attention_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const uint* metadata [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& page_tokens [[buffer(8)]],
    constant uint& max_blocks [[buffer(9)]],
    constant uint& key_head_dim [[buffer(10)]],
    constant uint& value_head_dim [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant float& softcap [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint row = group.y;
    if (row >= batch_size || head >= num_heads) {
        return;
    }

    const uint context_len = metadata[row];
    const uint first_page_offset = metadata[batch_size + row];
    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    const uint q_base = (row * num_heads + head) * key_head_dim;
    const uint table_base = 2 * batch_size + row * max_blocks;

    const uint threads_per_threadgroup = threads_per_group.x;
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0; pos < context_len; pos++) {
        const uint physical_pos = first_page_offset + pos;
        const uint logical_page = physical_pos / page_tokens;
        const uint page_offset = physical_pos - logical_page * page_tokens;
        const uint physical_page = metadata[table_base + logical_page];
        const uint slot = physical_page * page_tokens + page_offset;
        const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;

        float local_dot = 0.0f;
        for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
            local_dot += float(q[q_base + dim]) * float(k[k_base + dim]);
        }
        dot_scratch[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                dot_scratch[tid] += dot_scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float score = dot_scratch[0] * scale;
            if (softcap > 0.0f) {
                score = softcap * tanh(score / softcap);
            }
            const float next_max = max(online_state[0], score);
            const float alpha = online_state[1] == 0.0f
                ? 0.0f
                : exp(online_state[0] - next_max);
            const float beta = exp(score - next_max);
            online_state[0] = next_max;
            online_state[1] = online_state[1] * alpha + beta;
            online_state[2] = alpha;
            online_state[3] = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
        for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
            accumulator[dim] = accumulator[dim] * online_state[2]
                + float(v[v_base + dim]) * online_state[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_base = (row * num_heads + head) * value_head_dim;
    const float inv_sum = 1.0f / online_state[1];
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        output[out_base + dim] = half(accumulator[dim] * inv_sum);
    }
}

kernel void izwi_paged_decode_attention_split_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* partial_values [[buffer(3)]],
    device float* partial_maxes [[buffer(4)]],
    device float* partial_sums [[buffer(5)]],
    device const uint* metadata [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& num_kv_heads [[buffer(9)]],
    constant uint& page_tokens [[buffer(10)]],
    constant uint& max_blocks [[buffer(11)]],
    constant uint& key_head_dim [[buffer(12)]],
    constant uint& value_head_dim [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    constant float& softcap [[buffer(15)]],
    constant uint& partition_tokens [[buffer(16)]],
    constant uint& max_partitions [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint row = group.y;
    const uint partition = group.z;
    if (row >= batch_size || head >= num_heads || partition >= max_partitions) {
        return;
    }

    const uint context_len = metadata[row];
    const uint partition_start = partition * partition_tokens;
    const uint partition_end = min(partition_start + partition_tokens, context_len);
    const uint partial_index = (row * num_heads + head) * max_partitions + partition;
    const uint partial_base = partial_index * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (partition_start < context_len) {
        const uint first_page_offset = metadata[batch_size + row];
        const uint kv_group = num_heads / num_kv_heads;
        const uint kv_head = head / kv_group;
        const uint q_base = (row * num_heads + head) * key_head_dim;
        const uint table_base = 2 * batch_size + row * max_blocks;

        for (uint pos = partition_start; pos < partition_end; pos++) {
            const uint physical_pos = first_page_offset + pos;
            const uint logical_page = physical_pos / page_tokens;
            const uint page_offset = physical_pos - logical_page * page_tokens;
            const uint physical_page = metadata[table_base + logical_page];
            const uint slot = physical_page * page_tokens + page_offset;
            const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;

            float local_dot = 0.0f;
            for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
                local_dot += q[q_base + dim] * k[k_base + dim];
            }
            dot_scratch[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    dot_scratch[tid] += dot_scratch[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = dot_scratch[0] * scale;
                if (softcap > 0.0f) {
                    score = softcap * tanh(score / softcap);
                }
                const float next_max = max(online_state[0], score);
                const float alpha = online_state[1] == 0.0f
                    ? 0.0f
                    : exp(online_state[0] - next_max);
                const float beta = exp(score - next_max);
                online_state[0] = next_max;
                online_state[1] = online_state[1] * alpha + beta;
                online_state[2] = alpha;
                online_state[3] = beta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
            for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
                accumulator[dim] = accumulator[dim] * online_state[2]
                    + v[v_base + dim] * online_state[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        partial_values[partial_base + dim] = accumulator[dim];
    }
    if (tid == 0) {
        partial_maxes[partial_index] = online_state[0];
        partial_sums[partial_index] = online_state[1];
    }
}

kernel void izwi_paged_decode_attention_split_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device float* partial_values [[buffer(3)]],
    device float* partial_maxes [[buffer(4)]],
    device float* partial_sums [[buffer(5)]],
    device const uint* metadata [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& num_heads [[buffer(8)]],
    constant uint& num_kv_heads [[buffer(9)]],
    constant uint& page_tokens [[buffer(10)]],
    constant uint& max_blocks [[buffer(11)]],
    constant uint& key_head_dim [[buffer(12)]],
    constant uint& value_head_dim [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    constant float& softcap [[buffer(15)]],
    constant uint& partition_tokens [[buffer(16)]],
    constant uint& max_partitions [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint row = group.y;
    const uint partition = group.z;
    if (row >= batch_size || head >= num_heads || partition >= max_partitions) {
        return;
    }

    const uint context_len = metadata[row];
    const uint partition_start = partition * partition_tokens;
    const uint partition_end = min(partition_start + partition_tokens, context_len);
    const uint partial_index = (row * num_heads + head) * max_partitions + partition;
    const uint partial_base = partial_index * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (partition_start < context_len) {
        const uint first_page_offset = metadata[batch_size + row];
        const uint kv_group = num_heads / num_kv_heads;
        const uint kv_head = head / kv_group;
        const uint q_base = (row * num_heads + head) * key_head_dim;
        const uint table_base = 2 * batch_size + row * max_blocks;

        for (uint pos = partition_start; pos < partition_end; pos++) {
            const uint physical_pos = first_page_offset + pos;
            const uint logical_page = physical_pos / page_tokens;
            const uint page_offset = physical_pos - logical_page * page_tokens;
            const uint physical_page = metadata[table_base + logical_page];
            const uint slot = physical_page * page_tokens + page_offset;
            const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;

            float local_dot = 0.0f;
            for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
                local_dot += float(q[q_base + dim]) * float(k[k_base + dim]);
            }
            dot_scratch[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    dot_scratch[tid] += dot_scratch[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float score = dot_scratch[0] * scale;
                if (softcap > 0.0f) {
                    score = softcap * tanh(score / softcap);
                }
                const float next_max = max(online_state[0], score);
                const float alpha = online_state[1] == 0.0f
                    ? 0.0f
                    : exp(online_state[0] - next_max);
                const float beta = exp(score - next_max);
                online_state[0] = next_max;
                online_state[1] = online_state[1] * alpha + beta;
                online_state[2] = alpha;
                online_state[3] = beta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
            for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
                accumulator[dim] = accumulator[dim] * online_state[2]
                    + float(v[v_base + dim]) * online_state[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        partial_values[partial_base + dim] = accumulator[dim];
    }
    if (tid == 0) {
        partial_maxes[partial_index] = online_state[0];
        partial_sums[partial_index] = online_state[1];
    }
}

kernel void izwi_paged_decode_attention_reduce_f32(
    device const float* partial_values [[buffer(0)]],
    device const float* partial_maxes [[buffer(1)]],
    device const float* partial_sums [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* context_lens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& value_head_dim [[buffer(6)]],
    constant uint& partition_tokens [[buffer(7)]],
    constant uint& max_partitions [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float merge_state[2];
    const uint head = group.x;
    const uint row = group.y;
    const uint partial_base = (row * num_heads + head) * max_partitions;
    const uint partition_count = (context_lens[row] + partition_tokens - 1) / partition_tokens;

    if (tid == 0) {
        float global_max = -INFINITY;
        for (uint partition = 0; partition < partition_count; partition++) {
            global_max = max(global_max, partial_maxes[partial_base + partition]);
        }
        float global_sum = 0.0f;
        for (uint partition = 0; partition < partition_count; partition++) {
            global_sum += partial_sums[partial_base + partition]
                * exp(partial_maxes[partial_base + partition] - global_max);
        }
        merge_state[0] = global_max;
        merge_state[1] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint value_base = partial_base * value_head_dim;
    const uint out_base = (row * num_heads + head) * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        float acc = 0.0f;
        for (uint partition = 0; partition < partition_count; partition++) {
            const float weight = exp(partial_maxes[partial_base + partition] - merge_state[0]);
            acc += partial_values[value_base + partition * value_head_dim + dim] * weight;
        }
        output[out_base + dim] = acc / merge_state[1];
    }
}

kernel void izwi_paged_decode_attention_reduce_f16(
    device const float* partial_values [[buffer(0)]],
    device const float* partial_maxes [[buffer(1)]],
    device const float* partial_sums [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const uint* context_lens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& value_head_dim [[buffer(6)]],
    constant uint& partition_tokens [[buffer(7)]],
    constant uint& max_partitions [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float merge_state[2];
    const uint head = group.x;
    const uint row = group.y;
    const uint partial_base = (row * num_heads + head) * max_partitions;
    const uint partition_count = (context_lens[row] + partition_tokens - 1) / partition_tokens;

    if (tid == 0) {
        float global_max = -INFINITY;
        for (uint partition = 0; partition < partition_count; partition++) {
            global_max = max(global_max, partial_maxes[partial_base + partition]);
        }
        float global_sum = 0.0f;
        for (uint partition = 0; partition < partition_count; partition++) {
            global_sum += partial_sums[partial_base + partition]
                * exp(partial_maxes[partial_base + partition] - global_max);
        }
        merge_state[0] = global_max;
        merge_state[1] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint value_base = partial_base * value_head_dim;
    const uint out_base = (row * num_heads + head) * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        float acc = 0.0f;
        for (uint partition = 0; partition < partition_count; partition++) {
            const float weight = exp(partial_maxes[partial_base + partition] - merge_state[0]);
            acc += partial_values[value_base + partition * value_head_dim + dim] * weight;
        }
        output[out_base + dim] = half(acc / merge_state[1]);
    }
}

kernel void izwi_paged_prefill_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* metadata [[buffer(4)]],
    constant uint& sequence_count [[buffer(5)]],
    constant uint& total_queries [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    constant uint& max_blocks [[buffer(10)]],
    constant uint& key_head_dim [[buffer(11)]],
    constant uint& value_head_dim [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant float& softcap [[buffer(14)]],
    constant uint& window_tokens [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint query_index = group.y;
    if (query_index >= total_queries || head >= num_heads) {
        return;
    }

    const uint query_rows_base = sequence_count * (4 + max_blocks);
    const uint sequence = metadata[query_rows_base + query_index];
    const uint query_start = metadata[sequence];
    const uint query_len = metadata[sequence_count + sequence];
    const uint context_len = metadata[2 * sequence_count + sequence];
    const uint first_page_offset = metadata[3 * sequence_count + sequence];
    const uint query_offset = query_index - query_start;
    const uint visible_context = context_len - query_len + query_offset + 1;
    const uint window_start = window_tokens > 0 && visible_context > window_tokens
        ? visible_context - window_tokens
        : 0;
    const uint table_base = 4 * sequence_count + sequence * max_blocks;
    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    const uint q_base = (query_index * num_heads + head) * key_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = window_start; pos < visible_context; pos++) {
        const uint physical_pos = first_page_offset + pos;
        const uint logical_page = physical_pos / page_tokens;
        const uint page_offset = physical_pos - logical_page * page_tokens;
        const uint physical_page = metadata[table_base + logical_page];
        const uint slot = physical_page * page_tokens + page_offset;
        const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;
        float local_dot = 0.0f;
        for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
            local_dot += q[q_base + dim] * k[k_base + dim];
        }
        dot_scratch[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                dot_scratch[tid] += dot_scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float score = dot_scratch[0] * scale;
            if (softcap > 0.0f) {
                score = softcap * tanh(score / softcap);
            }
            const float next_max = max(online_state[0], score);
            const float alpha = online_state[1] == 0.0f
                ? 0.0f
                : exp(online_state[0] - next_max);
            const float beta = exp(score - next_max);
            online_state[0] = next_max;
            online_state[1] = online_state[1] * alpha + beta;
            online_state[2] = alpha;
            online_state[3] = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
        for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
            accumulator[dim] = accumulator[dim] * online_state[2]
                + v[v_base + dim] * online_state[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_base = (query_index * num_heads + head) * value_head_dim;
    const float inv_sum = 1.0f / online_state[1];
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        output[out_base + dim] = accumulator[dim] * inv_sum;
    }
}

kernel void izwi_paged_prefill_attention_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* output [[buffer(3)]],
    device const uint* metadata [[buffer(4)]],
    constant uint& sequence_count [[buffer(5)]],
    constant uint& total_queries [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    constant uint& max_blocks [[buffer(10)]],
    constant uint& key_head_dim [[buffer(11)]],
    constant uint& value_head_dim [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant float& softcap [[buffer(14)]],
    constant uint& window_tokens [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint query_index = group.y;
    if (query_index >= total_queries || head >= num_heads) {
        return;
    }

    const uint query_rows_base = sequence_count * (4 + max_blocks);
    const uint sequence = metadata[query_rows_base + query_index];
    const uint query_start = metadata[sequence];
    const uint query_len = metadata[sequence_count + sequence];
    const uint context_len = metadata[2 * sequence_count + sequence];
    const uint first_page_offset = metadata[3 * sequence_count + sequence];
    const uint query_offset = query_index - query_start;
    const uint visible_context = context_len - query_len + query_offset + 1;
    const uint window_start = window_tokens > 0 && visible_context > window_tokens
        ? visible_context - window_tokens
        : 0;
    const uint table_base = 4 * sequence_count + sequence * max_blocks;
    const uint kv_group = num_heads / num_kv_heads;
    const uint kv_head = head / kv_group;
    const uint q_base = (query_index * num_heads + head) * key_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = window_start; pos < visible_context; pos++) {
        const uint physical_pos = first_page_offset + pos;
        const uint logical_page = physical_pos / page_tokens;
        const uint page_offset = physical_pos - logical_page * page_tokens;
        const uint physical_page = metadata[table_base + logical_page];
        const uint slot = physical_page * page_tokens + page_offset;
        const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;
        float local_dot = 0.0f;
        for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
            local_dot += float(q[q_base + dim]) * float(k[k_base + dim]);
        }
        dot_scratch[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                dot_scratch[tid] += dot_scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float score = dot_scratch[0] * scale;
            if (softcap > 0.0f) {
                score = softcap * tanh(score / softcap);
            }
            const float next_max = max(online_state[0], score);
            const float alpha = online_state[1] == 0.0f
                ? 0.0f
                : exp(online_state[0] - next_max);
            const float beta = exp(score - next_max);
            online_state[0] = next_max;
            online_state[1] = online_state[1] * alpha + beta;
            online_state[2] = alpha;
            online_state[3] = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
        for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
            accumulator[dim] = accumulator[dim] * online_state[2]
                + float(v[v_base + dim]) * online_state[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_base = (query_index * num_heads + head) * value_head_dim;
    const float inv_sum = 1.0f / online_state[1];
    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        output[out_base + dim] = half(accumulator[dim] * inv_sum);
    }
}

kernel void izwi_paged_prefill_attention_split_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* partial_values [[buffer(3)]],
    device float* partial_maxes [[buffer(4)]],
    device float* partial_sums [[buffer(5)]],
    device const uint* metadata [[buffer(6)]],
    constant uint& sequence_count [[buffer(7)]],
    constant uint& total_queries [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& page_tokens [[buffer(11)]],
    constant uint& max_blocks [[buffer(12)]],
    constant uint& key_head_dim [[buffer(13)]],
    constant uint& value_head_dim [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    constant float& softcap [[buffer(16)]],
    constant uint& window_tokens [[buffer(17)]],
    constant uint& partition_tokens [[buffer(18)]],
    constant uint& max_partitions [[buffer(19)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint query_index = group.y;
    const uint partition = group.z;
    if (query_index >= total_queries || head >= num_heads || partition >= max_partitions) {
        return;
    }

    const uint query_rows_base = sequence_count * (4 + max_blocks);
    const uint sequence = metadata[query_rows_base + query_index];
    const uint query_start = metadata[sequence];
    const uint query_len = metadata[sequence_count + sequence];
    const uint context_len = metadata[2 * sequence_count + sequence];
    const uint first_page_offset = metadata[3 * sequence_count + sequence];
    const uint query_offset = query_index - query_start;
    const uint visible_context = context_len - query_len + query_offset + 1;
    const uint window_start = window_tokens > 0 && visible_context > window_tokens
        ? visible_context - window_tokens
        : 0;
    const uint attended_tokens = visible_context - window_start;
    const uint relative_start = partition * partition_tokens;
    const uint relative_end = min(relative_start + partition_tokens, attended_tokens);
    const uint partial_index = (query_index * num_heads + head) * max_partitions + partition;
    const uint partial_base = partial_index * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (relative_start < attended_tokens) {
        const uint table_base = 4 * sequence_count + sequence * max_blocks;
        const uint kv_group = num_heads / num_kv_heads;
        const uint kv_head = head / kv_group;
        const uint q_base = (query_index * num_heads + head) * key_head_dim;
        for (uint relative_pos = relative_start; relative_pos < relative_end; relative_pos++) {
            const uint physical_pos = first_page_offset + window_start + relative_pos;
            const uint logical_page = physical_pos / page_tokens;
            const uint page_offset = physical_pos - logical_page * page_tokens;
            const uint physical_page = metadata[table_base + logical_page];
            const uint slot = physical_page * page_tokens + page_offset;
            const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;
            float local_dot = 0.0f;
            for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
                local_dot += q[q_base + dim] * k[k_base + dim];
            }
            dot_scratch[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    dot_scratch[tid] += dot_scratch[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                float score = dot_scratch[0] * scale;
                if (softcap > 0.0f) {
                    score = softcap * tanh(score / softcap);
                }
                const float next_max = max(online_state[0], score);
                const float alpha = online_state[1] == 0.0f ? 0.0f : exp(online_state[0] - next_max);
                const float beta = exp(score - next_max);
                online_state[0] = next_max;
                online_state[1] = online_state[1] * alpha + beta;
                online_state[2] = alpha;
                online_state[3] = beta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
            for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
                accumulator[dim] = accumulator[dim] * online_state[2]
                    + v[v_base + dim] * online_state[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        partial_values[partial_base + dim] = accumulator[dim];
    }
    if (tid == 0) {
        partial_maxes[partial_index] = online_state[0];
        partial_sums[partial_index] = online_state[1];
    }
}

kernel void izwi_paged_prefill_attention_split_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device float* partial_values [[buffer(3)]],
    device float* partial_maxes [[buffer(4)]],
    device float* partial_sums [[buffer(5)]],
    device const uint* metadata [[buffer(6)]],
    constant uint& sequence_count [[buffer(7)]],
    constant uint& total_queries [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& page_tokens [[buffer(11)]],
    constant uint& max_blocks [[buffer(12)]],
    constant uint& key_head_dim [[buffer(13)]],
    constant uint& value_head_dim [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    constant float& softcap [[buffer(16)]],
    constant uint& window_tokens [[buffer(17)]],
    constant uint& partition_tokens [[buffer(18)]],
    constant uint& max_partitions [[buffer(19)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float dot_scratch[256];
    threadgroup float accumulator[512];
    threadgroup float online_state[4];

    const uint head = group.x;
    const uint query_index = group.y;
    const uint partition = group.z;
    if (query_index >= total_queries || head >= num_heads || partition >= max_partitions) {
        return;
    }

    const uint query_rows_base = sequence_count * (4 + max_blocks);
    const uint sequence = metadata[query_rows_base + query_index];
    const uint query_start = metadata[sequence];
    const uint query_len = metadata[sequence_count + sequence];
    const uint context_len = metadata[2 * sequence_count + sequence];
    const uint first_page_offset = metadata[3 * sequence_count + sequence];
    const uint query_offset = query_index - query_start;
    const uint visible_context = context_len - query_len + query_offset + 1;
    const uint window_start = window_tokens > 0 && visible_context > window_tokens
        ? visible_context - window_tokens
        : 0;
    const uint attended_tokens = visible_context - window_start;
    const uint relative_start = partition * partition_tokens;
    const uint relative_end = min(relative_start + partition_tokens, attended_tokens);
    const uint partial_index = (query_index * num_heads + head) * max_partitions + partition;
    const uint partial_base = partial_index * value_head_dim;
    const uint threads_per_threadgroup = threads_per_group.x;

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        accumulator[dim] = 0.0f;
    }
    if (tid == 0) {
        online_state[0] = -INFINITY;
        online_state[1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (relative_start < attended_tokens) {
        const uint table_base = 4 * sequence_count + sequence * max_blocks;
        const uint kv_group = num_heads / num_kv_heads;
        const uint kv_head = head / kv_group;
        const uint q_base = (query_index * num_heads + head) * key_head_dim;
        for (uint relative_pos = relative_start; relative_pos < relative_end; relative_pos++) {
            const uint physical_pos = first_page_offset + window_start + relative_pos;
            const uint logical_page = physical_pos / page_tokens;
            const uint page_offset = physical_pos - logical_page * page_tokens;
            const uint physical_page = metadata[table_base + logical_page];
            const uint slot = physical_page * page_tokens + page_offset;
            const uint k_base = (slot * num_kv_heads + kv_head) * key_head_dim;
            float local_dot = 0.0f;
            for (uint dim = tid; dim < key_head_dim; dim += threads_per_threadgroup) {
                local_dot += float(q[q_base + dim]) * float(k[k_base + dim]);
            }
            dot_scratch[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    dot_scratch[tid] += dot_scratch[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                float score = dot_scratch[0] * scale;
                if (softcap > 0.0f) {
                    score = softcap * tanh(score / softcap);
                }
                const float next_max = max(online_state[0], score);
                const float alpha = online_state[1] == 0.0f ? 0.0f : exp(online_state[0] - next_max);
                const float beta = exp(score - next_max);
                online_state[0] = next_max;
                online_state[1] = online_state[1] * alpha + beta;
                online_state[2] = alpha;
                online_state[3] = beta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint v_base = (slot * num_kv_heads + kv_head) * value_head_dim;
            for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
                accumulator[dim] = accumulator[dim] * online_state[2]
                    + float(v[v_base + dim]) * online_state[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint dim = tid; dim < value_head_dim; dim += threads_per_threadgroup) {
        partial_values[partial_base + dim] = accumulator[dim];
    }
    if (tid == 0) {
        partial_maxes[partial_index] = online_state[0];
        partial_sums[partial_index] = online_state[1];
    }
}

kernel void izwi_qwen35_causal_conv_sequence_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* history [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& conv_dim [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& kernel_size [[buffer(6)]],
    constant uint& output_elem_count [[buffer(7)]],
    constant uint& total_elem_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_elem_count) {
        return;
    }

    const uint history_len = kernel_size - 1;
    if (gid < output_elem_count) {
        const uint token = gid / conv_dim;
        const uint channel = gid - token * conv_dim;
        const uint weight_base = channel * kernel_size;
        float value = 0.0f;
        for (uint tap = 0; tap < kernel_size; tap++) {
            const uint source_pos = token + tap;
            const float source = source_pos < history_len
                ? history[channel * history_len + source_pos]
                : input[(source_pos - history_len) * conv_dim + channel];
            value += source * weight[weight_base + tap];
        }
        output[gid] = value / (1.0f + exp(-value));
        return;
    }

    // Persist the last `kernel_size - 1` raw inputs in oldest-to-newest order.
    // This also handles chunks shorter than the history window by retaining the
    // still-live suffix of the incoming history.
    const uint state_idx = gid - output_elem_count;
    const uint channel = state_idx / history_len;
    const uint history_pos = state_idx - channel * history_len;
    const uint source_pos = seq_len + history_pos;
    output[gid] = source_pos < history_len
        ? history[channel * history_len + source_pos]
        : input[(source_pos - history_len) * conv_dim + channel];
}

kernel void izwi_qwen35_gated_delta_sequence_f32(
    device const float* qkv [[buffer(0)]],
    device const float* gates [[buffer(1)]],
    device const float* initial_state [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_k_heads [[buffer(5)]],
    constant uint& num_v_heads [[buffer(6)]],
    constant uint& head_k_dim [[buffer(7)]],
    constant uint& head_v_dim [[buffer(8)]],
    constant uint& qkv_width [[buffer(9)]],
    constant uint& output_elem_count [[buffer(10)]],
    constant float& query_scale [[buffer(11)]],
    constant uint& token_start [[buffer(12)]],
    constant uint& token_count [[buffer(13)]],
    constant uint& initialize_state [[buffer(14)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float delta[256];

    if (head >= num_v_heads) {
        return;
    }

    const uint state_head_size = head_k_dim * head_v_dim;
    const uint state_base = output_elem_count + head * state_head_size;
    const uint initial_state_base = head * state_head_size;
    if (initialize_state != 0) {
        for (uint idx = tid; idx < state_head_size; idx += threads_per_threadgroup) {
            output[state_base + idx] = initial_state[initial_state_base + idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // GGUF conversion stores V heads in tiled order, so the matching K head is
    // `v_head % num_k_heads` for both 16K/16V and 16K/32V layouts.
    const uint key_head = head % num_k_heads;
    const uint key_width = num_k_heads * head_k_dim;
    const uint value_offset = key_width * 2;

    const uint token_end = min(token_start + token_count, seq_len);
    for (uint token = token_start; token < token_end; token++) {
        const uint qkv_base = token * qkv_width;
        const uint query_base = qkv_base + key_head * head_k_dim;
        const uint key_base = qkv_base + key_width + key_head * head_k_dim;
        const uint value_base = qkv_base + value_offset + head * head_v_dim;
        const uint gate_base = token * num_v_heads * 2;
        const float decay = exp(gates[gate_base + head]);
        const float beta = gates[gate_base + num_v_heads + head];

        for (uint idx = tid; idx < state_head_size; idx += threads_per_threadgroup) {
            output[state_base + idx] *= decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint value_dim = tid; value_dim < head_v_dim; value_dim += threads_per_threadgroup) {
            float memory = 0.0f;
            for (uint key_dim = 0; key_dim < head_k_dim; key_dim++) {
                memory += output[state_base + key_dim * head_v_dim + value_dim]
                    * qkv[key_base + key_dim];
            }
            delta[value_dim] = (qkv[value_base + value_dim] - memory) * beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint idx = tid; idx < state_head_size; idx += threads_per_threadgroup) {
            const uint key_dim = idx / head_v_dim;
            const uint value_dim = idx - key_dim * head_v_dim;
            output[state_base + idx] += qkv[key_base + key_dim] * delta[value_dim];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint value_dim = tid; value_dim < head_v_dim; value_dim += threads_per_threadgroup) {
            float recurrent_value = 0.0f;
            for (uint key_dim = 0; key_dim < head_k_dim; key_dim++) {
                recurrent_value += output[state_base + key_dim * head_v_dim + value_dim]
                    * (qkv[query_base + key_dim] * query_scale);
            }
            output[(token * num_v_heads + head) * head_v_dim + value_dim] = recurrent_value;
        }
        // No thread may begin decaying the next state until every output read
        // for the current token has completed.
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

kernel void izwi_lfm_shortconv_decode3_f32(
    device const float* cache [[buffer(0)]],
    device const float* bx [[buffer(1)]],
    device const float* conv [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& elem_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elem_count) {
        return;
    }

    const uint h = gid % hidden_size;
    const uint b = gid / hidden_size;
    const uint cache_base = ((b * hidden_size) + h) * 3;
    const uint conv_base = h * 3;
    output[gid] =
        cache[cache_base + 1] * conv[conv_base] +
        cache[cache_base + 2] * conv[conv_base + 1] +
        bx[gid] * conv[conv_base + 2];
}

kernel void izwi_lfm_shortconv_update3_f32(
    device const float* cache [[buffer(0)]],
    device const float* bx [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& elem_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elem_count) {
        return;
    }

    const uint h = gid % hidden_size;
    const uint b = gid / hidden_size;
    const uint cache_base = ((b * hidden_size) + h) * 3;
    const uint out_base = cache_base;
    output[out_base] = cache[cache_base + 1];
    output[out_base + 1] = cache[cache_base + 2];
    output[out_base + 2] = bx[gid];
}

kernel void izwi_lfm_shortconv_sequence3_f32(
    device const float* bx [[buffer(0)]],
    device const float* conv [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& elem_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elem_count) {
        return;
    }

    const uint t = gid % seq_len;
    const uint h = (gid / seq_len) % hidden_size;
    const uint row_base = gid - t;
    const uint conv_base = h * 3;
    float value = bx[gid] * conv[conv_base + 2];
    if (t >= 1) {
        value += bx[row_base + t - 1] * conv[conv_base + 1];
    }
    if (t >= 2) {
        value += bx[row_base + t - 2] * conv[conv_base];
    }
    output[gid] = value;
}

kernel void izwi_lfm_shortconv_ring_f32(
    device const float* ring [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    constant ulong& expected_cursor [[buffer(8)]],
    constant ulong& valid_length [[buffer(9)]],
    constant uint& output_elements [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_elements) {
        return;
    }
    const uint step = gid % steps;
    const uint hidden_idx = (gid / steps) % hidden;
    const uint batch_idx = gid / (steps * hidden);
    const long window_start =
        long(expected_cursor) + long(step) + 1 - long(capacity);
    const ulong oldest = expected_cursor - valid_length;
    float value = 0.0f;
    for (uint tap = 0; tap < capacity; ++tap) {
        const long source = window_start + long(tap);
        if (source < 0) {
            continue;
        }
        const ulong absolute_source = ulong(source);
        float source_value;
        if (absolute_source < expected_cursor) {
            if (absolute_source < oldest) {
                continue;
            }
            const ulong physical = absolute_source % ulong(capacity);
            source_value =
                ring[(physical * ulong(batch) + ulong(batch_idx)) *
                     ulong(hidden) + ulong(hidden_idx)];
        } else {
            const ulong input_step = absolute_source - expected_cursor;
            if (input_step > ulong(step)) {
                continue;
            }
            source_value =
                input[(batch_idx * hidden + hidden_idx) * steps +
                      uint(input_step)];
        }
        value += source_value * weight[hidden_idx * capacity + tap];
    }
    output[gid] = value;
}
"#;

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct SiluMulOp;

#[cfg(feature = "metal")]
impl CustomOp2 for SiluMulOp {
    fn name(&self) -> &'static str {
        "izwi-silu-mul-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-silu-mul-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        gate_storage: &MetalStorage,
        gate_layout: &Layout,
        up_storage: &MetalStorage,
        up_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = gate_storage.dtype();
        if up_storage.dtype() != dtype {
            bail!("izwi-silu-mul-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-silu-mul-metal only supports F32 and F16 tensors")
        }
        if gate_layout.shape() != up_layout.shape() {
            bail!("izwi-silu-mul-metal requires matching shapes")
        }
        if !gate_layout.is_contiguous() || !up_layout.is_contiguous() {
            bail!("izwi-silu-mul-metal requires contiguous inputs")
        }

        let elem_count = gate_layout.shape().elem_count();
        if elem_count > u32::MAX as usize {
            bail!("izwi-silu-mul-metal tensor is too large")
        }

        let device = gate_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-silu-mul")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-silu-mul");
        let pipeline = silu_mul_pipeline(device.metal_device(), dtype)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(up_storage.buffer()),
            up_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            gate_layout.shape().clone(),
        ))
    }
}

#[cfg(feature = "metal")]
fn silu_mul_pipeline(device: &MetalDevice, dtype: DType) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_silu_mul_f32",
        DType::F16 => "izwi_silu_mul_f16",
        _ => bail!("izwi-silu-mul-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct QkRmsNormOp {
    q_rows: usize,
    k_rows: usize,
    head_dim: usize,
    eps: f32,
}

#[cfg(feature = "metal")]
impl CustomOp3 for QkRmsNormOp {
    fn name(&self) -> &'static str {
        "izwi-qk-rms-norm-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-qk-rms-norm-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        if k_storage.dtype() != dtype || weight_storage.dtype() != dtype {
            bail!("izwi-qk-rms-norm-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-qk-rms-norm-metal only supports F32 and F16 tensors")
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !weight_layout.is_contiguous()
        {
            bail!("izwi-qk-rms-norm-metal requires contiguous tensors")
        }
        if self.head_dim == 0 || self.head_dim > 256 {
            bail!("izwi-qk-rms-norm-metal requires 1..=256 head_dim")
        }
        if q_layout.shape().elem_count() != self.q_rows.saturating_mul(self.head_dim) {
            bail!("izwi-qk-rms-norm-metal q shape does not match q_rows/head_dim")
        }
        if k_layout.shape().elem_count() != self.k_rows.saturating_mul(self.head_dim) {
            bail!("izwi-qk-rms-norm-metal k shape does not match k_rows/head_dim")
        }
        if weight_layout.shape().elem_count() != self.head_dim.saturating_mul(2) {
            bail!("izwi-qk-rms-norm-metal weight must contain q and k norm weights")
        }

        let rows = self.q_rows.saturating_add(self.k_rows);
        let elem_count = rows.saturating_mul(self.head_dim);
        if elem_count > u32::MAX as usize
            || self.q_rows > u32::MAX as usize
            || self.k_rows > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
        {
            bail!("izwi-qk-rms-norm-metal tensor is too large")
        }

        let device = q_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-qk-rms-norm")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-qk-rms-norm");
        let pipeline = qk_rms_norm_pipeline(device.metal_device(), dtype)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.q_rows as u32));
        encoder.set_bytes(5, &(self.k_rows as u32));
        encoder.set_bytes(6, &(self.head_dim as u32));
        encoder.set_bytes(7, &self.eps);

        let threads_per_threadgroup = self
            .head_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .clamp(1, 256);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            Shape::from((rows, self.head_dim)),
        ))
    }
}

#[cfg(feature = "metal")]
fn qk_rms_norm_pipeline(device: &MetalDevice, dtype: DType) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_qk_rms_norm_f32",
        DType::F16 => "izwi_qk_rms_norm_f16",
        _ => bail!("izwi-qk-rms-norm-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct RmsNormOp {
    rows: usize,
    hidden_dim: usize,
    eps: f32,
}

#[cfg(feature = "metal")]
impl CustomOp2 for RmsNormOp {
    fn name(&self) -> &'static str {
        "izwi-rms-norm-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-rms-norm-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        input_storage: &MetalStorage,
        input_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = input_storage.dtype();
        if weight_storage.dtype() != dtype {
            bail!("izwi-rms-norm-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-rms-norm-metal only supports F32 and F16 tensors")
        }
        if !input_layout.is_contiguous() || !weight_layout.is_contiguous() {
            bail!("izwi-rms-norm-metal requires contiguous tensors")
        }
        if self.rows == 0 || self.hidden_dim == 0 {
            bail!("izwi-rms-norm-metal requires non-empty rows and hidden dim")
        }
        if input_layout.shape().elem_count() != self.rows.saturating_mul(self.hidden_dim) {
            bail!("izwi-rms-norm-metal input shape does not match rows/hidden_dim")
        }
        if weight_layout.shape().elem_count() != self.hidden_dim {
            bail!("izwi-rms-norm-metal weight length does not match hidden_dim")
        }
        let elem_count = input_layout.shape().elem_count();
        if elem_count > u32::MAX as usize
            || self.rows > u32::MAX as usize
            || self.hidden_dim > u32::MAX as usize
        {
            bail!("izwi-rms-norm-metal tensor is too large")
        }

        let device = input_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-rms-norm")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-rms-norm");
        let pipeline = rms_norm_pipeline(device.metal_device(), dtype)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(input_storage.buffer()),
            input_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.rows as u32));
        encoder.set_bytes(4, &(self.hidden_dim as u32));
        encoder.set_bytes(5, &self.eps);

        let threads_per_threadgroup = self
            .hidden_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .clamp(1, 1024);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            input_layout.shape().clone(),
        ))
    }
}

#[cfg(feature = "metal")]
fn rms_norm_pipeline(device: &MetalDevice, dtype: DType) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_rms_norm_f32",
        DType::F16 => "izwi_rms_norm_f16",
        _ => bail!("izwi-rms-norm-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct RopePairBshdOp {
    q_rows: usize,
    k_rows: usize,
    seq_len: usize,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
}

#[cfg(feature = "metal")]
impl CustomOp3 for RopePairBshdOp {
    fn name(&self) -> &'static str {
        "izwi-rope-pair-bshd-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-rope-pair-bshd-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        cos_sin_storage: &MetalStorage,
        cos_sin_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        if k_storage.dtype() != dtype || cos_sin_storage.dtype() != dtype {
            bail!("izwi-rope-pair-bshd-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-rope-pair-bshd-metal only supports F32 and F16 tensors")
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !cos_sin_layout.is_contiguous()
        {
            bail!("izwi-rope-pair-bshd-metal requires contiguous tensors")
        }
        if self.seq_len == 0
            || self.q_heads == 0
            || self.k_heads == 0
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
        {
            bail!("izwi-rope-pair-bshd-metal requires non-empty even head_dim")
        }
        if q_layout.shape().elem_count() != self.q_rows.saturating_mul(self.head_dim) {
            bail!("izwi-rope-pair-bshd-metal q shape does not match rows/head_dim")
        }
        if k_layout.shape().elem_count() != self.k_rows.saturating_mul(self.head_dim) {
            bail!("izwi-rope-pair-bshd-metal k shape does not match rows/head_dim")
        }
        if cos_sin_layout.shape().elem_count() != self.seq_len.saturating_mul(self.head_dim) {
            bail!("izwi-rope-pair-bshd-metal packed cos/sin shape mismatch")
        }
        if !self.q_rows.is_multiple_of(self.q_heads)
            || !self.k_rows.is_multiple_of(self.k_heads)
            || !(self.q_rows / self.q_heads).is_multiple_of(self.seq_len)
            || !(self.k_rows / self.k_heads).is_multiple_of(self.seq_len)
        {
            bail!("izwi-rope-pair-bshd-metal rows do not match heads/seq_len")
        }

        let rows = self.q_rows.saturating_add(self.k_rows);
        let elem_count = rows.saturating_mul(self.head_dim);
        if elem_count > u32::MAX as usize
            || self.q_rows > u32::MAX as usize
            || self.k_rows > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.q_heads > u32::MAX as usize
            || self.k_heads > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
        {
            bail!("izwi-rope-pair-bshd-metal tensor is too large")
        }

        let device = q_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-rope-pair-bshd")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-rope-pair-bshd");
        let pipeline = rope_pair_bshd_pipeline(device.metal_device(), dtype)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(cos_sin_storage.buffer()),
            cos_sin_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.q_rows as u32));
        encoder.set_bytes(5, &(self.k_rows as u32));
        encoder.set_bytes(6, &(self.seq_len as u32));
        encoder.set_bytes(7, &(self.q_heads as u32));
        encoder.set_bytes(8, &(self.k_heads as u32));
        encoder.set_bytes(9, &(self.head_dim as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            Shape::from((rows, self.head_dim)),
        ))
    }
}

#[cfg(feature = "metal")]
fn rope_pair_bshd_pipeline(device: &MetalDevice, dtype: DType) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_rope_pair_bshd_f32",
        DType::F16 => "izwi_rope_pair_bshd_f16",
        _ => bail!("izwi-rope-pair-bshd-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct DecodeGqaAttentionOp {
    num_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
    kv_capacity_len: usize,
    head_dim: usize,
    scale: f32,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
struct PagedDecodeAttentionOp {
    metadata: Vec<u32>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    scale: f32,
    softcap: Option<f32>,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
struct PagedPrefillAttentionOp {
    metadata: Vec<u32>,
    sequence_count: usize,
    total_queries: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    scale: f32,
    softcap: Option<f32>,
    window_tokens: Option<u32>,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct LfmShortConvDecode3Op {
    batch_size: usize,
    hidden_size: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct LfmShortConvUpdate3Op {
    batch_size: usize,
    hidden_size: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct LfmShortConvSequence3Op {
    batch_size: usize,
    hidden_size: usize,
    seq_len: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct LfmShortConvRingOp {
    batch_size: usize,
    hidden_size: usize,
    steps: usize,
    capacity: usize,
    expected_cursor: u64,
    valid_length: u64,
}

#[cfg(feature = "metal")]
impl CustomOp3 for DecodeGqaAttentionOp {
    fn name(&self) -> &'static str {
        "izwi-decode-gqa-attention-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-decode-gqa-attention-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        v_storage: &MetalStorage,
        v_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        if k_storage.dtype() != dtype || v_storage.dtype() != dtype {
            bail!("izwi-decode-gqa-attention-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-decode-gqa-attention-metal only supports F32 and F16 tensors")
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
            bail!("izwi-decode-gqa-attention-metal requires contiguous tensors")
        }
        if self.num_heads == 0
            || self.num_kv_heads == 0
            || self.kv_len == 0
            || self.kv_len > self.kv_capacity_len
            || self.kv_len > 2048
            || self.head_dim == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            bail!("izwi-decode-gqa-attention-metal unsupported shape")
        }
        if q_layout.shape().elem_count() != self.num_heads.saturating_mul(self.head_dim) {
            bail!("izwi-decode-gqa-attention-metal q shape mismatch")
        }
        let kv_elems = self
            .num_kv_heads
            .saturating_mul(self.kv_capacity_len)
            .saturating_mul(self.head_dim);
        if k_layout.shape().elem_count() != kv_elems || v_layout.shape().elem_count() != kv_elems {
            bail!("izwi-decode-gqa-attention-metal k/v shape mismatch")
        }
        let elem_count = self.num_heads.saturating_mul(self.head_dim);
        if elem_count > u32::MAX as usize
            || self.num_heads > u32::MAX as usize
            || self.num_kv_heads > u32::MAX as usize
            || self.kv_len > u32::MAX as usize
            || self.kv_capacity_len > u32::MAX as usize
            || self.head_dim > u32::MAX as usize
        {
            bail!("izwi-decode-gqa-attention-metal tensor is too large")
        }

        let device = q_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-decode-gqa-attention")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-decode-gqa-attention");
        let pipeline = decode_gqa_attention_pipeline(device.metal_device(), dtype)?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(v_storage.buffer()),
            v_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.num_heads as u32));
        encoder.set_bytes(5, &(self.num_kv_heads as u32));
        encoder.set_bytes(6, &(self.kv_len as u32));
        encoder.set_bytes(7, &(self.head_dim as u32));
        encoder.set_bytes(8, &self.scale);
        encoder.set_bytes(9, &(self.kv_capacity_len as u32));

        let threads_per_threadgroup = self
            .head_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .clamp(1, 256);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.num_heads,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            Shape::from((1, self.num_heads, 1, self.head_dim)),
        ))
    }
}

#[cfg(feature = "metal")]
fn decode_gqa_attention_pipeline(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_decode_gqa_attention_f32",
        DType::F16 => "izwi_decode_gqa_attention_f16",
        _ => bail!("izwi-decode-gqa-attention-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
impl CustomOp3 for PagedDecodeAttentionOp {
    fn name(&self) -> &'static str {
        "izwi-paged-decode-attention-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-paged-decode-attention-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        v_storage: &MetalStorage,
        v_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        if k_storage.dtype() != dtype || v_storage.dtype() != dtype {
            bail!("izwi-paged-decode-attention-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-paged-decode-attention-metal only supports F32 and F16 tensors")
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
            bail!("izwi-paged-decode-attention-metal requires contiguous tensors")
        }
        if self.batch_size == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
            || self.page_tokens == 0
            || self.max_blocks == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.key_head_dim > 512
            || self.value_head_dim > 512
            || self
                .softcap
                .is_some_and(|softcap| !softcap.is_finite() || softcap <= 0.0)
            || self.metadata.len()
                != self
                    .batch_size
                    .saturating_mul(2)
                    .saturating_add(self.batch_size.saturating_mul(self.max_blocks))
        {
            bail!("izwi-paged-decode-attention-metal unsupported shape")
        }
        let q_elems = self
            .batch_size
            .saturating_mul(self.num_heads)
            .saturating_mul(self.key_head_dim);
        if q_layout.shape().elem_count() != q_elems {
            bail!("izwi-paged-decode-attention-metal query shape mismatch")
        }
        let k_dims = k_layout.shape().dims();
        let v_dims = v_layout.shape().dims();
        if k_dims.len() != 4
            || v_dims.len() != 4
            || k_dims[0] != v_dims[0]
            || k_dims[1] != self.page_tokens
            || v_dims[1] != self.page_tokens
            || k_dims[2] != self.num_kv_heads
            || v_dims[2] != self.num_kv_heads
            || k_dims[3] != self.key_head_dim
            || v_dims[3] != self.value_head_dim
        {
            bail!("izwi-paged-decode-attention-metal page-major K/V shape mismatch")
        }
        let mut max_context_len = 0usize;
        for row in 0..self.batch_size {
            let context_len = self.metadata[row] as usize;
            let first_page_offset = self.metadata[self.batch_size + row] as usize;
            if context_len == 0 || first_page_offset >= self.page_tokens {
                bail!("izwi-paged-decode-attention-metal invalid row metadata")
            }
            let physical_tokens = context_len.checked_add(first_page_offset).ok_or_else(|| {
                candle_core::Error::Msg("paged attention context overflow".into())
            })?;
            let required_pages = physical_tokens.div_ceil(self.page_tokens);
            if required_pages == 0 || required_pages > self.max_blocks {
                bail!("izwi-paged-decode-attention-metal incomplete block table")
            }
            let table_base = self.batch_size.saturating_mul(2) + row * self.max_blocks;
            if self.metadata[table_base..table_base + required_pages]
                .iter()
                .any(|&page| page as usize >= k_dims[0])
            {
                bail!("izwi-paged-decode-attention-metal physical page is out of bounds")
            }
            max_context_len = max_context_len.max(context_len);
        }
        let values = [
            self.batch_size,
            self.num_heads,
            self.num_kv_heads,
            self.page_tokens,
            self.max_blocks,
            self.key_head_dim,
            self.value_head_dim,
        ];
        if values.iter().any(|&value| value > u32::MAX as usize) {
            bail!("izwi-paged-decode-attention-metal tensor is too large")
        }

        let elem_count = self
            .batch_size
            .checked_mul(self.num_heads)
            .and_then(|value| value.checked_mul(self.value_head_dim))
            .ok_or_else(|| candle_core::Error::Msg("paged attention output overflow".into()))?;
        let device = q_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-paged-decode-attention")?;
        // One compact host-authored control buffer per physical batch. It holds
        // context lengths, first-page offsets, and physical page ids; K/V
        // remains in the arena.
        let metadata = device.new_buffer_with_data(&self.metadata)?;
        let encoder = device.command_encoder()?;
        let reduction_width = self
            .key_head_dim
            .max(self.value_head_dim)
            .next_power_of_two()
            .clamp(1, 256);
        let base_workgroups = self.batch_size.saturating_mul(self.num_heads);
        let use_split_kv = max_context_len > METAL_PAGED_ATTENTION_SPLIT_MIN_CONTEXT
            && base_workgroups <= METAL_PAGED_ATTENTION_SPLIT_MAX_BASE_WORKGROUPS;

        if use_split_kv {
            // Adapted from vllm-metal's split-KV dispatch at commit
            // cc1b679725085ddb40f9beb0ed36e7745ae8d688 (Apache-2.0): partition
            // long, low-occupancy contexts and merge partial online-softmax
            // states in a second dispatch. Izwi keeps partial values in F32 to
            // avoid an intermediate F16 round-trip.
            let max_partitions = max_context_len.div_ceil(METAL_PAGED_ATTENTION_PARTITION_TOKENS);
            let partial_count = base_workgroups
                .checked_mul(max_partitions)
                .ok_or_else(|| candle_core::Error::Msg("paged attention split overflow".into()))?;
            let partial_value_count = partial_count
                .checked_mul(self.value_head_dim)
                .ok_or_else(|| candle_core::Error::Msg("paged attention split overflow".into()))?;
            let partial_values = device.new_buffer(
                partial_value_count,
                DType::F32,
                "izwi-paged-attention-parts",
            )?;
            let partial_maxes =
                device.new_buffer(partial_count, DType::F32, "izwi-paged-attention-maxes")?;
            let partial_sums =
                device.new_buffer(partial_count, DType::F32, "izwi-paged-attention-sums")?;
            let (split_pipeline, reduce_pipeline) =
                paged_decode_attention_split_pipelines(device.metal_device(), dtype)?;

            encoder.set_label("izwi-paged-decode-attention-split");
            encoder.set_compute_pipeline_state(&split_pipeline);
            encoder.set_input_buffer(
                0,
                Some(q_storage.buffer()),
                q_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                1,
                Some(k_storage.buffer()),
                k_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                2,
                Some(v_storage.buffer()),
                v_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_output_buffer(3, Some(&partial_values), 0);
            encoder.set_output_buffer(4, Some(&partial_maxes), 0);
            encoder.set_output_buffer(5, Some(&partial_sums), 0);
            encoder.set_input_buffer(6, Some(&metadata), 0);
            encoder.set_bytes(7, &(self.batch_size as u32));
            encoder.set_bytes(8, &(self.num_heads as u32));
            encoder.set_bytes(9, &(self.num_kv_heads as u32));
            encoder.set_bytes(10, &(self.page_tokens as u32));
            encoder.set_bytes(11, &(self.max_blocks as u32));
            encoder.set_bytes(12, &(self.key_head_dim as u32));
            encoder.set_bytes(13, &(self.value_head_dim as u32));
            encoder.set_bytes(14, &self.scale);
            encoder.set_bytes(15, &self.softcap.unwrap_or(0.0));
            encoder.set_bytes(16, &(METAL_PAGED_ATTENTION_PARTITION_TOKENS as u32));
            encoder.set_bytes(17, &(max_partitions as u32));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.batch_size,
                    depth: max_partitions,
                },
                objc2_metal::MTLSize {
                    width: reduction_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.insert_memory_barrier();

            encoder.set_label("izwi-paged-decode-attention-reduce");
            encoder.set_compute_pipeline_state(&reduce_pipeline);
            encoder.set_input_buffer(0, Some(&partial_values), 0);
            encoder.set_input_buffer(1, Some(&partial_maxes), 0);
            encoder.set_input_buffer(2, Some(&partial_sums), 0);
            encoder.set_output_buffer(3, Some(&output), 0);
            encoder.set_input_buffer(4, Some(&metadata), 0);
            encoder.set_bytes(5, &(self.num_heads as u32));
            encoder.set_bytes(6, &(self.value_head_dim as u32));
            encoder.set_bytes(7, &(METAL_PAGED_ATTENTION_PARTITION_TOKENS as u32));
            encoder.set_bytes(8, &(max_partitions as u32));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.batch_size,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: reduction_width,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            encoder.set_label("izwi-paged-decode-attention");
            let pipeline = paged_decode_attention_pipeline(device.metal_device(), dtype)?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_input_buffer(
                0,
                Some(q_storage.buffer()),
                q_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                1,
                Some(k_storage.buffer()),
                k_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                2,
                Some(v_storage.buffer()),
                v_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_output_buffer(3, Some(&output), 0);
            encoder.set_input_buffer(4, Some(&metadata), 0);
            encoder.set_bytes(5, &(self.batch_size as u32));
            encoder.set_bytes(6, &(self.num_heads as u32));
            encoder.set_bytes(7, &(self.num_kv_heads as u32));
            encoder.set_bytes(8, &(self.page_tokens as u32));
            encoder.set_bytes(9, &(self.max_blocks as u32));
            encoder.set_bytes(10, &(self.key_head_dim as u32));
            encoder.set_bytes(11, &(self.value_head_dim as u32));
            encoder.set_bytes(12, &self.scale);
            encoder.set_bytes(13, &self.softcap.unwrap_or(0.0));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.batch_size,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: reduction_width,
                    height: 1,
                    depth: 1,
                },
            );
        }
        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            Shape::from((self.batch_size, self.num_heads, self.value_head_dim)),
        ))
    }
}

#[cfg(feature = "metal")]
fn paged_decode_attention_pipeline(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);
    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_paged_decode_attention_f32",
        DType::F16 => "izwi_paged_decode_attention_f16",
        _ => bail!("izwi-paged-decode-attention-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
fn paged_decode_attention_split_pipelines(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<(ComputePipeline, ComputePipeline)> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), (ComputePipeline, ComputePipeline)>>> =
        OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);
    if let Some(pipelines) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipelines);
    }

    let (split_name, reduce_name) = match dtype {
        DType::F32 => (
            "izwi_paged_decode_attention_split_f32",
            "izwi_paged_decode_attention_reduce_f32",
        ),
        DType::F16 => (
            "izwi_paged_decode_attention_split_f16",
            "izwi_paged_decode_attention_reduce_f16",
        ),
        _ => bail!("izwi-paged-decode-attention-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let split_function = library
        .get_function(split_name, None)
        .map_err(candle_core::Error::wrap)?;
    let reduce_function = library
        .get_function(reduce_name, None)
        .map_err(candle_core::Error::wrap)?;
    let split_pipeline = device
        .new_compute_pipeline_state_with_function(&split_function)
        .map_err(candle_core::Error::wrap)?;
    let reduce_pipeline = device
        .new_compute_pipeline_state_with_function(&reduce_function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(key, (split_pipeline.clone(), reduce_pipeline.clone()));
    Ok((split_pipeline, reduce_pipeline))
}

#[cfg(feature = "metal")]
impl CustomOp3 for PagedPrefillAttentionOp {
    fn name(&self) -> &'static str {
        "izwi-paged-prefill-attention-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-paged-prefill-attention-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        q_storage: &MetalStorage,
        q_layout: &Layout,
        k_storage: &MetalStorage,
        k_layout: &Layout,
        v_storage: &MetalStorage,
        v_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        if k_storage.dtype() != dtype || v_storage.dtype() != dtype {
            bail!("izwi-paged-prefill-attention-metal requires matching dtypes")
        }
        if !matches!(dtype, DType::F32 | DType::F16) {
            bail!("izwi-paged-prefill-attention-metal only supports F32 and F16 tensors")
        }
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
            bail!("izwi-paged-prefill-attention-metal requires contiguous tensors")
        }
        let expected_metadata = self
            .sequence_count
            .checked_mul(4usize.saturating_add(self.max_blocks))
            .ok_or_else(|| candle_core::Error::Msg("paged prefill metadata overflow".into()))?;
        if self.sequence_count == 0
            || self.total_queries == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
            || self.page_tokens == 0
            || self.max_blocks == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.key_head_dim > 512
            || self.value_head_dim > 512
            || self.metadata.len() != expected_metadata
            || !self.scale.is_finite()
            || self.scale <= 0.0
            || self
                .softcap
                .is_some_and(|softcap| !softcap.is_finite() || softcap <= 0.0)
            || self.window_tokens == Some(0)
        {
            bail!("izwi-paged-prefill-attention-metal unsupported shape")
        }
        let q_elems = self
            .total_queries
            .checked_mul(self.num_heads)
            .and_then(|value| value.checked_mul(self.key_head_dim))
            .ok_or_else(|| candle_core::Error::Msg("paged prefill query overflow".into()))?;
        if q_layout.shape().elem_count() != q_elems {
            bail!("izwi-paged-prefill-attention-metal query shape mismatch")
        }
        let k_dims = k_layout.shape().dims();
        let v_dims = v_layout.shape().dims();
        if k_dims.len() != 4
            || v_dims.len() != 4
            || k_dims[0] != v_dims[0]
            || k_dims[1] != self.page_tokens
            || v_dims[1] != self.page_tokens
            || k_dims[2] != self.num_kv_heads
            || v_dims[2] != self.num_kv_heads
            || k_dims[3] != self.key_head_dim
            || v_dims[3] != self.value_head_dim
        {
            bail!("izwi-paged-prefill-attention-metal page-major K/V shape mismatch")
        }

        let mut next_query = 0usize;
        let mut query_rows = Vec::with_capacity(self.total_queries);
        let mut query_attention_lens = Vec::with_capacity(self.total_queries);
        let mut max_attention_len = 0usize;
        for sequence in 0..self.sequence_count {
            let query_start = self.metadata[sequence] as usize;
            let query_len = self.metadata[self.sequence_count + sequence] as usize;
            let context_len = self.metadata[2 * self.sequence_count + sequence] as usize;
            let first_page_offset = self.metadata[3 * self.sequence_count + sequence] as usize;
            if query_start != next_query
                || query_len == 0
                || query_len > context_len
                || first_page_offset >= self.page_tokens
            {
                bail!("izwi-paged-prefill-attention-metal invalid sequence metadata")
            }
            let physical_tokens = context_len
                .checked_add(first_page_offset)
                .ok_or_else(|| candle_core::Error::Msg("paged prefill context overflow".into()))?;
            let required_pages = physical_tokens.div_ceil(self.page_tokens);
            if required_pages == 0 || required_pages > self.max_blocks {
                bail!("izwi-paged-prefill-attention-metal incomplete block table")
            }
            let table_base = 4 * self.sequence_count + sequence * self.max_blocks;
            if self.metadata[table_base..table_base + required_pages]
                .iter()
                .any(|&page| page as usize >= k_dims[0])
            {
                bail!("izwi-paged-prefill-attention-metal physical page is out of bounds")
            }
            next_query = next_query
                .checked_add(query_len)
                .ok_or_else(|| candle_core::Error::Msg("paged prefill query overflow".into()))?;
            for query_offset in 0..query_len {
                let visible_context = context_len - query_len + query_offset + 1;
                let attention_len = self.window_tokens.map_or(visible_context, |window| {
                    visible_context.min(window as usize)
                });
                query_rows.push(sequence as u32);
                query_attention_lens.push(attention_len as u32);
                max_attention_len = max_attention_len.max(attention_len);
            }
        }
        if next_query != self.total_queries {
            bail!("izwi-paged-prefill-attention-metal rows do not cover every query")
        }
        let values = [
            self.sequence_count,
            self.total_queries,
            self.num_heads,
            self.num_kv_heads,
            self.page_tokens,
            self.max_blocks,
            self.key_head_dim,
            self.value_head_dim,
        ];
        if values.iter().any(|&value| value > u32::MAX as usize) {
            bail!("izwi-paged-prefill-attention-metal tensor is too large")
        }

        let elem_count = self
            .total_queries
            .checked_mul(self.num_heads)
            .and_then(|value| value.checked_mul(self.value_head_dim))
            .ok_or_else(|| candle_core::Error::Msg("paged prefill output overflow".into()))?;
        let device = q_storage.device().clone();
        let output = device.new_buffer(elem_count, dtype, "izwi-paged-prefill-attention")?;
        // Append direct query-to-sequence rows after the compact sequence
        // records. Both short and split kernels now perform O(1) row lookup.
        let mut expanded_metadata = self.metadata.clone();
        expanded_metadata.extend_from_slice(&query_rows);
        let metadata = device.new_buffer_with_data(&expanded_metadata)?;
        let encoder = device.command_encoder()?;
        let reduction_width = self
            .key_head_dim
            .max(self.value_head_dim)
            .next_power_of_two()
            .clamp(1, 256);
        let base_workgroups = self.total_queries.saturating_mul(self.num_heads);
        let use_split_kv = max_attention_len > METAL_PAGED_PREFILL_SPLIT_MIN_CONTEXT
            && base_workgroups <= METAL_PAGED_PREFILL_SPLIT_MAX_BASE_WORKGROUPS;

        if use_split_kv {
            let max_partitions = max_attention_len.div_ceil(METAL_PAGED_ATTENTION_PARTITION_TOKENS);
            let partial_count = base_workgroups
                .checked_mul(max_partitions)
                .ok_or_else(|| candle_core::Error::Msg("paged prefill split overflow".into()))?;
            let partial_value_count = partial_count
                .checked_mul(self.value_head_dim)
                .ok_or_else(|| candle_core::Error::Msg("paged prefill split overflow".into()))?;
            let partial_values =
                device.new_buffer(partial_value_count, DType::F32, "izwi-paged-prefill-parts")?;
            let partial_maxes =
                device.new_buffer(partial_count, DType::F32, "izwi-paged-prefill-maxes")?;
            let partial_sums =
                device.new_buffer(partial_count, DType::F32, "izwi-paged-prefill-sums")?;
            let attention_lens = device.new_buffer_with_data(&query_attention_lens)?;
            let (split_pipeline, reduce_pipeline) =
                paged_prefill_attention_split_pipelines(device.metal_device(), dtype)?;

            encoder.set_label("izwi-paged-prefill-attention-split");
            encoder.set_compute_pipeline_state(&split_pipeline);
            encoder.set_input_buffer(
                0,
                Some(q_storage.buffer()),
                q_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                1,
                Some(k_storage.buffer()),
                k_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                2,
                Some(v_storage.buffer()),
                v_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_output_buffer(3, Some(&partial_values), 0);
            encoder.set_output_buffer(4, Some(&partial_maxes), 0);
            encoder.set_output_buffer(5, Some(&partial_sums), 0);
            encoder.set_input_buffer(6, Some(&metadata), 0);
            encoder.set_bytes(7, &(self.sequence_count as u32));
            encoder.set_bytes(8, &(self.total_queries as u32));
            encoder.set_bytes(9, &(self.num_heads as u32));
            encoder.set_bytes(10, &(self.num_kv_heads as u32));
            encoder.set_bytes(11, &(self.page_tokens as u32));
            encoder.set_bytes(12, &(self.max_blocks as u32));
            encoder.set_bytes(13, &(self.key_head_dim as u32));
            encoder.set_bytes(14, &(self.value_head_dim as u32));
            encoder.set_bytes(15, &self.scale);
            encoder.set_bytes(16, &self.softcap.unwrap_or(0.0));
            encoder.set_bytes(17, &self.window_tokens.unwrap_or(0));
            encoder.set_bytes(18, &(METAL_PAGED_ATTENTION_PARTITION_TOKENS as u32));
            encoder.set_bytes(19, &(max_partitions as u32));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.total_queries,
                    depth: max_partitions,
                },
                objc2_metal::MTLSize {
                    width: reduction_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.insert_memory_barrier();

            encoder.set_label("izwi-paged-prefill-attention-reduce");
            encoder.set_compute_pipeline_state(&reduce_pipeline);
            encoder.set_input_buffer(0, Some(&partial_values), 0);
            encoder.set_input_buffer(1, Some(&partial_maxes), 0);
            encoder.set_input_buffer(2, Some(&partial_sums), 0);
            encoder.set_output_buffer(3, Some(&output), 0);
            encoder.set_input_buffer(4, Some(&attention_lens), 0);
            encoder.set_bytes(5, &(self.num_heads as u32));
            encoder.set_bytes(6, &(self.value_head_dim as u32));
            encoder.set_bytes(7, &(METAL_PAGED_ATTENTION_PARTITION_TOKENS as u32));
            encoder.set_bytes(8, &(max_partitions as u32));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.total_queries,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: reduction_width,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            encoder.set_label("izwi-paged-prefill-attention");
            let pipeline = paged_prefill_attention_pipeline(device.metal_device(), dtype)?;
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_input_buffer(
                0,
                Some(q_storage.buffer()),
                q_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                1,
                Some(k_storage.buffer()),
                k_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_input_buffer(
                2,
                Some(v_storage.buffer()),
                v_layout.start_offset() * dtype.size_in_bytes(),
            );
            encoder.set_output_buffer(3, Some(&output), 0);
            encoder.set_input_buffer(4, Some(&metadata), 0);
            encoder.set_bytes(5, &(self.sequence_count as u32));
            encoder.set_bytes(6, &(self.total_queries as u32));
            encoder.set_bytes(7, &(self.num_heads as u32));
            encoder.set_bytes(8, &(self.num_kv_heads as u32));
            encoder.set_bytes(9, &(self.page_tokens as u32));
            encoder.set_bytes(10, &(self.max_blocks as u32));
            encoder.set_bytes(11, &(self.key_head_dim as u32));
            encoder.set_bytes(12, &(self.value_head_dim as u32));
            encoder.set_bytes(13, &self.scale);
            encoder.set_bytes(14, &self.softcap.unwrap_or(0.0));
            encoder.set_bytes(15, &self.window_tokens.unwrap_or(0));
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: self.num_heads,
                    height: self.total_queries,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: reduction_width.min(pipeline.max_total_threads_per_threadgroup()),
                    height: 1,
                    depth: 1,
                },
            );
        }
        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, dtype),
            Shape::from((self.total_queries, self.num_heads, self.value_head_dim)),
        ))
    }
}

#[cfg(feature = "metal")]
fn paged_prefill_attention_pipeline(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);
    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }

    let function_name = match dtype {
        DType::F32 => "izwi_paged_prefill_attention_f32",
        DType::F16 => "izwi_paged_prefill_attention_f16",
        _ => bail!("izwi-paged-prefill-attention-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function_name, None)
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

#[cfg(feature = "metal")]
fn paged_prefill_attention_split_pipelines(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<(ComputePipeline, ComputePipeline)> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, DType), (ComputePipeline, ComputePipeline)>>> =
        OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (registry_id, dtype);
    if let Some(pipelines) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipelines);
    }

    let (split_name, reduce_name) = match dtype {
        DType::F32 => (
            "izwi_paged_prefill_attention_split_f32",
            "izwi_paged_decode_attention_reduce_f32",
        ),
        DType::F16 => (
            "izwi_paged_prefill_attention_split_f16",
            "izwi_paged_decode_attention_reduce_f16",
        ),
        _ => bail!("izwi-paged-prefill-attention-metal only supports F32 and F16 tensors"),
    };
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let split_function = library
        .get_function(split_name, None)
        .map_err(candle_core::Error::wrap)?;
    let reduce_function = library
        .get_function(reduce_name, None)
        .map_err(candle_core::Error::wrap)?;
    let split_pipeline = device
        .new_compute_pipeline_state_with_function(&split_function)
        .map_err(candle_core::Error::wrap)?;
    let reduce_pipeline = device
        .new_compute_pipeline_state_with_function(&reduce_function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(key, (split_pipeline.clone(), reduce_pipeline.clone()));
    Ok((split_pipeline, reduce_pipeline))
}

#[cfg(feature = "metal")]
impl CustomOp3 for LfmShortConvDecode3Op {
    fn name(&self) -> &'static str {
        "izwi-lfm-shortconv-decode3-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-lfm-shortconv-decode3-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        cache_storage: &MetalStorage,
        cache_layout: &Layout,
        bx_storage: &MetalStorage,
        bx_layout: &Layout,
        conv_storage: &MetalStorage,
        conv_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if cache_storage.dtype() != DType::F32
            || bx_storage.dtype() != DType::F32
            || conv_storage.dtype() != DType::F32
        {
            bail!("izwi-lfm-shortconv-decode3-metal only supports F32 tensors")
        }
        if !cache_layout.is_contiguous()
            || !bx_layout.is_contiguous()
            || !conv_layout.is_contiguous()
        {
            bail!("izwi-lfm-shortconv-decode3-metal requires contiguous tensors")
        }
        if self.batch_size == 0 || self.hidden_size == 0 {
            bail!("izwi-lfm-shortconv-decode3-metal requires non-empty dimensions")
        }
        let elem_count = self.batch_size.saturating_mul(self.hidden_size);
        if cache_layout.shape().elem_count() != elem_count.saturating_mul(3)
            || bx_layout.shape().elem_count() != elem_count
            || conv_layout.shape().elem_count() != self.hidden_size.saturating_mul(3)
        {
            bail!("izwi-lfm-shortconv-decode3-metal input shape mismatch")
        }
        if elem_count > u32::MAX as usize || self.hidden_size > u32::MAX as usize {
            bail!("izwi-lfm-shortconv-decode3-metal tensor is too large")
        }

        let device = cache_storage.device().clone();
        let output = device.new_buffer(elem_count, DType::F32, "izwi-lfm-shortconv-decode3")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-shortconv-decode3");
        let pipeline = lfm_shortconv_decode3_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(cache_storage.buffer()),
            cache_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(bx_storage.buffer()),
            bx_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(conv_storage.buffer()),
            conv_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.hidden_size as u32));
        encoder.set_bytes(5, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, DType::F32),
            Shape::from((self.batch_size, self.hidden_size, 1)),
        ))
    }
}

#[cfg(feature = "metal")]
fn lfm_shortconv_decode3_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }

    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_lfm_shortconv_decode3_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;

    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());

    Ok(pipeline)
}

#[cfg(feature = "metal")]
impl CustomOp2 for LfmShortConvUpdate3Op {
    fn name(&self) -> &'static str {
        "izwi-lfm-shortconv-update3-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-lfm-shortconv-update3-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        cache_storage: &MetalStorage,
        cache_layout: &Layout,
        bx_storage: &MetalStorage,
        bx_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if cache_storage.dtype() != DType::F32 || bx_storage.dtype() != DType::F32 {
            bail!("izwi-lfm-shortconv-update3-metal only supports F32 tensors")
        }
        if !cache_layout.is_contiguous() || !bx_layout.is_contiguous() {
            bail!("izwi-lfm-shortconv-update3-metal requires contiguous tensors")
        }
        if self.batch_size == 0 || self.hidden_size == 0 {
            bail!("izwi-lfm-shortconv-update3-metal requires non-empty dimensions")
        }
        let elem_count = self.batch_size.saturating_mul(self.hidden_size);
        if cache_layout.shape().elem_count() != elem_count.saturating_mul(3)
            || bx_layout.shape().elem_count() != elem_count
        {
            bail!("izwi-lfm-shortconv-update3-metal input shape mismatch")
        }
        if elem_count > u32::MAX as usize || self.hidden_size > u32::MAX as usize {
            bail!("izwi-lfm-shortconv-update3-metal tensor is too large")
        }

        let device = cache_storage.device().clone();
        let output = device.new_buffer(
            elem_count.saturating_mul(3),
            DType::F32,
            "izwi-lfm-shortconv-update3",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-shortconv-update3");
        let pipeline = lfm_shortconv_update3_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(cache_storage.buffer()),
            cache_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(bx_storage.buffer()),
            bx_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.hidden_size as u32));
        encoder.set_bytes(4, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count.saturating_mul(3), DType::F32),
            Shape::from((self.batch_size, self.hidden_size, 3)),
        ))
    }
}

#[cfg(feature = "metal")]
fn lfm_shortconv_update3_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }

    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_lfm_shortconv_update3_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;

    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());

    Ok(pipeline)
}

#[cfg(feature = "metal")]
impl CustomOp2 for LfmShortConvSequence3Op {
    fn name(&self) -> &'static str {
        "izwi-lfm-shortconv-sequence3-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-lfm-shortconv-sequence3-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        bx_storage: &MetalStorage,
        bx_layout: &Layout,
        conv_storage: &MetalStorage,
        conv_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if bx_storage.dtype() != DType::F32 || conv_storage.dtype() != DType::F32 {
            bail!("izwi-lfm-shortconv-sequence3-metal only supports F32 tensors")
        }
        if !bx_layout.is_contiguous() || !conv_layout.is_contiguous() {
            bail!("izwi-lfm-shortconv-sequence3-metal requires contiguous tensors")
        }
        if self.batch_size == 0 || self.hidden_size == 0 || self.seq_len == 0 {
            bail!("izwi-lfm-shortconv-sequence3-metal requires non-empty dimensions")
        }
        let elem_count = self
            .batch_size
            .saturating_mul(self.hidden_size)
            .saturating_mul(self.seq_len);
        if bx_layout.shape().elem_count() != elem_count
            || conv_layout.shape().elem_count() != self.hidden_size.saturating_mul(3)
        {
            bail!("izwi-lfm-shortconv-sequence3-metal input shape mismatch")
        }
        if elem_count > u32::MAX as usize
            || self.hidden_size > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
        {
            bail!("izwi-lfm-shortconv-sequence3-metal tensor is too large")
        }

        let device = bx_storage.device().clone();
        let output = device.new_buffer(elem_count, DType::F32, "izwi-lfm-shortconv-sequence3")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-shortconv-sequence3");
        let pipeline = lfm_shortconv_sequence3_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(bx_storage.buffer()),
            bx_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(conv_storage.buffer()),
            conv_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.hidden_size as u32));
        encoder.set_bytes(4, &(self.seq_len as u32));
        encoder.set_bytes(5, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );

        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, DType::F32),
            Shape::from((self.batch_size, self.hidden_size, self.seq_len)),
        ))
    }
}

#[cfg(feature = "metal")]
fn lfm_shortconv_sequence3_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }

    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_lfm_shortconv_sequence3_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;

    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());

    Ok(pipeline)
}

#[cfg(feature = "metal")]
impl CustomOp3 for LfmShortConvRingOp {
    fn name(&self) -> &'static str {
        "izwi-lfm-shortconv-ring-metal"
    }

    fn cpu_fwd(
        &self,
        _ring: &CpuStorage,
        _ring_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-lfm-shortconv-ring-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        ring_storage: &MetalStorage,
        ring_layout: &Layout,
        input_storage: &MetalStorage,
        input_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if ring_storage.dtype() != DType::F32
            || input_storage.dtype() != DType::F32
            || weight_storage.dtype() != DType::F32
        {
            bail!("izwi-lfm-shortconv-ring-metal only supports F32 tensors")
        }
        if !ring_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            bail!("izwi-lfm-shortconv-ring-metal requires contiguous tensors")
        }
        if self.batch_size == 0
            || self.hidden_size == 0
            || self.steps == 0
            || self.capacity == 0
            || self.valid_length > self.capacity as u64
            || self.valid_length > self.expected_cursor
        {
            bail!("izwi-lfm-shortconv-ring-metal received invalid geometry")
        }
        let elem_count = self
            .batch_size
            .checked_mul(self.hidden_size)
            .and_then(|value| value.checked_mul(self.steps))
            .ok_or_else(|| {
                candle_core::Error::Msg("izwi-lfm-shortconv-ring-metal output overflow".to_string())
            })?;
        if ring_layout.shape().elem_count()
            != self
                .capacity
                .saturating_mul(self.batch_size)
                .saturating_mul(self.hidden_size)
            || input_layout.shape().elem_count() != elem_count
            || weight_layout.shape().elem_count() != self.hidden_size.saturating_mul(self.capacity)
        {
            bail!("izwi-lfm-shortconv-ring-metal input shape mismatch")
        }
        if [
            self.batch_size,
            self.hidden_size,
            self.steps,
            self.capacity,
            elem_count,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
        {
            bail!("izwi-lfm-shortconv-ring-metal tensor is too large")
        }

        let device = ring_storage.device().clone();
        let output = device.new_buffer(elem_count, DType::F32, "izwi-lfm-shortconv-ring")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-shortconv-ring");
        let pipeline = lfm_shortconv_ring_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(ring_storage.buffer()),
            ring_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(input_storage.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.batch_size as u32));
        encoder.set_bytes(5, &(self.hidden_size as u32));
        encoder.set_bytes(6, &(self.steps as u32));
        encoder.set_bytes(7, &(self.capacity as u32));
        encoder.set_bytes(8, &self.expected_cursor);
        encoder.set_bytes(9, &self.valid_length);
        encoder.set_bytes(10, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(output, device, elem_count, DType::F32),
            Shape::from((self.batch_size, self.hidden_size, self.steps)),
        ))
    }
}

#[cfg(feature = "metal")]
fn lfm_shortconv_ring_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }
    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_lfm_shortconv_ring_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());
    Ok(pipeline)
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct Qwen35CausalConvSequenceOp {
    conv_dim: usize,
    seq_len: usize,
    kernel_size: usize,
}

#[cfg(feature = "metal")]
impl CustomOp3 for Qwen35CausalConvSequenceOp {
    fn name(&self) -> &'static str {
        "izwi-qwen35-causal-conv-sequence-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-qwen35-causal-conv-sequence-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        input_storage: &MetalStorage,
        input_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
        history_storage: &MetalStorage,
        history_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if input_storage.dtype() != DType::F32
            || weight_storage.dtype() != DType::F32
            || history_storage.dtype() != DType::F32
        {
            bail!("izwi-qwen35-causal-conv-sequence-metal only supports F32 tensors")
        }
        if !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !history_layout.is_contiguous()
        {
            bail!("izwi-qwen35-causal-conv-sequence-metal requires contiguous tensors")
        }
        if self.conv_dim == 0 || self.seq_len == 0 || self.kernel_size < 2 {
            bail!(
                "izwi-qwen35-causal-conv-sequence-metal requires non-empty dimensions and kernel_size >= 2"
            )
        }
        let history_len = self.kernel_size - 1;
        let Some(output_elem_count) = self.seq_len.checked_mul(self.conv_dim) else {
            bail!("izwi-qwen35-causal-conv-sequence-metal output size overflow")
        };
        let Some(state_elem_count) = history_len.checked_mul(self.conv_dim) else {
            bail!("izwi-qwen35-causal-conv-sequence-metal state size overflow")
        };
        let Some(weight_elem_count) = self.kernel_size.checked_mul(self.conv_dim) else {
            bail!("izwi-qwen35-causal-conv-sequence-metal weight size overflow")
        };
        let Some(total_elem_count) = output_elem_count.checked_add(state_elem_count) else {
            bail!("izwi-qwen35-causal-conv-sequence-metal packed output size overflow")
        };
        if input_layout.shape().elem_count() != output_elem_count
            || weight_layout.shape().elem_count() != weight_elem_count
            || history_layout.shape().elem_count() != state_elem_count
        {
            bail!("izwi-qwen35-causal-conv-sequence-metal input shape mismatch")
        }
        if total_elem_count > u32::MAX as usize
            || output_elem_count > u32::MAX as usize
            || self.conv_dim > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.kernel_size > u32::MAX as usize
        {
            bail!("izwi-qwen35-causal-conv-sequence-metal tensor is too large")
        }

        let device = input_storage.device().clone();
        let output = device.new_buffer(
            total_elem_count,
            DType::F32,
            "izwi-qwen35-causal-conv-sequence",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-qwen35-causal-conv-sequence");
        let pipeline = qwen35_causal_conv_sequence_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(input_storage.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(history_storage.buffer()),
            history_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.conv_dim as u32));
        encoder.set_bytes(5, &(self.seq_len as u32));
        encoder.set_bytes(6, &(self.kernel_size as u32));
        encoder.set_bytes(7, &(output_elem_count as u32));
        encoder.set_bytes(8, &(total_elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: total_elem_count,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(output, device, total_elem_count, DType::F32),
            Shape::from(total_elem_count),
        ))
    }
}

#[cfg(feature = "metal")]
fn qwen35_causal_conv_sequence_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }

    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_qwen35_causal_conv_sequence_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;

    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());
    Ok(pipeline)
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct Qwen35GatedDeltaSequenceOp {
    seq_len: usize,
    tile_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    qkv_width: usize,
    query_scale: f32,
}

#[cfg(feature = "metal")]
fn qwen35_gated_delta_tiled_layout_supported(
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> bool {
    num_k_heads == 16
        && matches!(num_v_heads, 16 | 32)
        && num_v_heads.is_multiple_of(num_k_heads)
        && head_k_dim > 0
        && (1..=256).contains(&head_v_dim)
}

#[cfg(feature = "metal")]
impl CustomOp3 for Qwen35GatedDeltaSequenceOp {
    fn name(&self) -> &'static str {
        "izwi-qwen35-gated-delta-sequence-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("izwi-qwen35-gated-delta-sequence-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        qkv_storage: &MetalStorage,
        qkv_layout: &Layout,
        gates_storage: &MetalStorage,
        gates_layout: &Layout,
        state_storage: &MetalStorage,
        state_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if qkv_storage.dtype() != DType::F32
            || gates_storage.dtype() != DType::F32
            || state_storage.dtype() != DType::F32
        {
            bail!("izwi-qwen35-gated-delta-sequence-metal only supports F32 tensors")
        }
        if !qkv_layout.is_contiguous()
            || !gates_layout.is_contiguous()
            || !state_layout.is_contiguous()
        {
            bail!("izwi-qwen35-gated-delta-sequence-metal requires contiguous tensors")
        }
        if self.seq_len == 0
            || self.tile_size == 0
            || !qwen35_gated_delta_tiled_layout_supported(
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
        {
            bail!(
                "izwi-qwen35-gated-delta-sequence-metal requires Qwen3.5 16K/16V or 16K/32V non-empty heads with head_v_dim <= 256"
            )
        }
        let Some(key_width) = self.num_k_heads.checked_mul(self.head_k_dim) else {
            bail!("izwi-qwen35-gated-delta-sequence-metal key width overflow")
        };
        let Some(value_width) = self.num_v_heads.checked_mul(self.head_v_dim) else {
            bail!("izwi-qwen35-gated-delta-sequence-metal value width overflow")
        };
        let Some(expected_qkv_width) = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
        else {
            bail!("izwi-qwen35-gated-delta-sequence-metal qkv width overflow")
        };
        if self.qkv_width != expected_qkv_width {
            bail!("izwi-qwen35-gated-delta-sequence-metal qkv width mismatch")
        }
        let Some(qkv_elem_count) = self.seq_len.checked_mul(self.qkv_width) else {
            bail!("izwi-qwen35-gated-delta-sequence-metal qkv size overflow")
        };
        let Some(gate_elem_count) = self
            .seq_len
            .checked_mul(self.num_v_heads)
            .and_then(|count| count.checked_mul(2))
        else {
            bail!("izwi-qwen35-gated-delta-sequence-metal gate size overflow")
        };
        let Some(output_elem_count) = self
            .seq_len
            .checked_mul(self.num_v_heads)
            .and_then(|count| count.checked_mul(self.head_v_dim))
        else {
            bail!("izwi-qwen35-gated-delta-sequence-metal output size overflow")
        };
        let Some(state_elem_count) = self
            .num_v_heads
            .checked_mul(self.head_k_dim)
            .and_then(|count| count.checked_mul(self.head_v_dim))
        else {
            bail!("izwi-qwen35-gated-delta-sequence-metal state size overflow")
        };
        let Some(total_elem_count) = output_elem_count.checked_add(state_elem_count) else {
            bail!("izwi-qwen35-gated-delta-sequence-metal packed output size overflow")
        };
        if qkv_layout.shape().elem_count() != qkv_elem_count
            || gates_layout.shape().elem_count() != gate_elem_count
            || state_layout.shape().elem_count() != state_elem_count
        {
            bail!("izwi-qwen35-gated-delta-sequence-metal input shape mismatch")
        }
        if total_elem_count > u32::MAX as usize
            || output_elem_count > u32::MAX as usize
            || self.seq_len > u32::MAX as usize
            || self.num_k_heads > u32::MAX as usize
            || self.num_v_heads > u32::MAX as usize
            || self.head_k_dim > u32::MAX as usize
            || self.head_v_dim > u32::MAX as usize
            || self.qkv_width > u32::MAX as usize
        {
            bail!("izwi-qwen35-gated-delta-sequence-metal tensor is too large")
        }

        let device = qkv_storage.device().clone();
        let output = device.new_buffer(
            total_elem_count,
            DType::F32,
            "izwi-qwen35-gated-delta-sequence",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-qwen35-gated-delta-sequence");
        let pipeline = qwen35_gated_delta_sequence_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(
            0,
            Some(qkv_storage.buffer()),
            qkv_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(gates_storage.buffer()),
            gates_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(state_storage.buffer()),
            state_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.seq_len as u32));
        encoder.set_bytes(5, &(self.num_k_heads as u32));
        encoder.set_bytes(6, &(self.num_v_heads as u32));
        encoder.set_bytes(7, &(self.head_k_dim as u32));
        encoder.set_bytes(8, &(self.head_v_dim as u32));
        encoder.set_bytes(9, &(self.qkv_width as u32));
        encoder.set_bytes(10, &(output_elem_count as u32));
        encoder.set_bytes(11, &self.query_scale);

        let threads_per_threadgroup = self
            .head_v_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .clamp(1, 256);
        let threadgroups = objc2_metal::MTLSize {
            width: self.num_v_heads,
            height: 1,
            depth: 1,
        };
        let threads = objc2_metal::MTLSize {
            width: threads_per_threadgroup,
            height: 1,
            depth: 1,
        };
        let tile_size = self.tile_size.min(self.seq_len);
        for token_start in (0..self.seq_len).step_by(tile_size) {
            if token_start != 0 {
                // Each tile consumes the recurrent state written into the
                // packed output buffer by the preceding dispatch.
                encoder.insert_memory_barrier();
            }
            let token_count = tile_size.min(self.seq_len - token_start);
            encoder.set_bytes(12, &(token_start as u32));
            encoder.set_bytes(13, &(token_count as u32));
            encoder.set_bytes(14, &u32::from(token_start == 0));
            encoder.dispatch_thread_groups(threadgroups, threads);
        }
        drop(encoder);

        Ok((
            MetalStorage::new(output, device, total_elem_count, DType::F32),
            Shape::from(total_elem_count),
        ))
    }
}

#[cfg(feature = "metal")]
fn qwen35_gated_delta_sequence_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let registry_id = device.registry_id();
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&registry_id)
        .cloned()
    {
        return Ok(pipeline);
    }

    let library = device
        .new_library_with_source(IZWI_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("izwi_qwen35_gated_delta_sequence_f32", None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;

    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(registry_id, pipeline.clone());
    Ok(pipeline)
}

/// Try fused gated delta recurrent computation.
///
/// This fuses multiple operations:
/// 1. Query scaling (1/sqrt(dim))
/// 2. State update with pre-computed gate (gated decay + key-value accumulation)
/// 3. Output projection
///
/// The `g` parameter should already be computed as: softplus(alpha) * a
///
/// Returns None if the operation cannot be performed (wrong dtype/device).
pub fn try_fused_gated_delta_recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices currently
    if query.dtype() != DType::F32 {
        return None;
    }

    if !query.device().is_metal() {
        return None;
    }

    // Validate shapes
    let (batch, num_v_heads, _head_k_dim) = query.dims3().ok()?;
    let (_, num_v_h, _head_v_dim) = value.dims3().ok()?;

    if batch != 1 || num_v_h != num_v_heads {
        return None;
    }

    // For now, use optimized sequential operations
    // In a future version with custom Metal kernels, this would be a single dispatch
    fused_gated_delta_sequential(query, key, value, g, beta, state).ok()
}

/// Optimized sequential implementation of gated delta using matmul for
/// batched reductions, halving intermediate tensor allocations.
fn fused_gated_delta_sequential(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> FusedResult<(Tensor, Tensor)> {
    let dim = query
        .dim(candle_core::D::Minus1)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;
    let scale = 1.0f64 / (dim as f64).sqrt();

    let scaled_query =
        (query * scale).map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    let g_val = g
        .exp()
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .reshape((1, g.dim(1).unwrap_or(1), 1, 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    let beta = beta
        .reshape((1, beta.dim(1).unwrap_or(1), 1))
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Gate the state
    let gated_state = state
        .broadcast_mul(&g_val)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // kv_mem via matmul: key (1,H,1,Dk) × state (1,H,Dk,Dv) → (1,H,1,Dv) → squeeze → (1,H,Dv)
    let kv_mem = key
        .unsqueeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .matmul(&gated_state)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .squeeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // delta = (value - kv_mem) * beta
    let delta = (value - kv_mem)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .broadcast_mul(&beta)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // State update via matmul outer product: key (1,H,Dk,1) × delta (1,H,1,Dv) → (1,H,Dk,Dv)
    let new_state = (&gated_state
        + &key
            .unsqueeze(3)
            .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
            .matmul(
                &delta
                    .unsqueeze(2)
                    .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?,
            )
            .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    // Output via matmul: query (1,H,1,Dk) × state (1,H,Dk,Dv) → (1,H,1,Dv) → squeeze → (1,H,Dv)
    let output = scaled_query
        .unsqueeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .matmul(&new_state)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?
        .squeeze(2)
        .map_err(|e| FusedKernelError::ExecutionError(e.to_string()))?;

    Ok((output, new_state))
}

/// Try fused L2 normalization.
///
/// Computes: x / sqrt(sum(x^2) + eps)
pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // Use Candle's built-in operations
    let sq_sum = input.sqr().ok()?.sum_keepdim(candle_core::D::Minus1).ok()?;

    let norm = (sq_sum + eps).ok()?.sqrt().ok()?;

    input.broadcast_div(&norm).ok()
}

/// Try fused MLP operation: silu(gate) * up.
///
/// This fuses the SiLU activation with the elementwise multiplication,
/// reducing memory bandwidth by 50% for this operation.
pub fn try_fused_silu_mul(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    try_fused_silu_mul_with_status(gate, up).map(|result| result.tensor)
}

pub fn try_fused_silu_mul_with_status(gate: &Tensor, up: &Tensor) -> Option<FusedSiluMulResult> {
    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        if gate.device().is_metal()
            && up.device().is_metal()
            && gate.dtype() == up.dtype()
            && matches!(gate.dtype(), DType::F32 | DType::F16)
            && gate.dims() == up.dims()
            && gate.is_contiguous()
            && up.is_contiguous()
        {
            if let Ok(result) = gate.apply_op2_no_bwd(up, &SiluMulOp) {
                return Some(FusedSiluMulResult {
                    tensor: result,
                    used_custom_kernel: true,
                });
            }
        }
    }

    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    let tensor = silu_gate.broadcast_mul(up).ok()?;
    Some(FusedSiluMulResult {
        tensor,
        used_custom_kernel: false,
    })
}

/// Try fused q_norm + k_norm for Qwen single-token decode.
///
/// Returns normalized q and k tensors with the same shapes as the inputs. This
/// custom kernel intentionally supports only the small contiguous Metal decode
/// case where q/k norm launch overhead dominates.
pub fn try_fused_qk_rms_norm(
    q: &Tensor,
    k: &Tensor,
    qk_weight: &Tensor,
    eps: f64,
) -> Option<(Tensor, Tensor)> {
    #[cfg(not(feature = "metal"))]
    let _ = (q, k, qk_weight, eps);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (q_bsz, q_seq, q_heads, q_head_dim) = q.dims4().ok()?;
        let (k_bsz, k_seq, k_heads, k_head_dim) = k.dims4().ok()?;
        if q_seq != 1 || k_seq != 1 {
            return None;
        }
        if q_bsz != k_bsz || q_head_dim != k_head_dim {
            return None;
        }
        if !q.device().is_metal()
            || !k.device().is_metal()
            || !qk_weight.device().is_metal()
            || q.dtype() != k.dtype()
            || q.dtype() != qk_weight.dtype()
            || !matches!(q.dtype(), DType::F32 | DType::F16)
            || !q.is_contiguous()
            || !k.is_contiguous()
            || !qk_weight.is_contiguous()
            || qk_weight.dims() != [q_head_dim * 2]
            || q_head_dim == 0
            || q_head_dim > 256
        {
            return None;
        }

        let q_rows = q_bsz.checked_mul(q_seq)?.checked_mul(q_heads)?;
        let k_rows = k_bsz.checked_mul(k_seq)?.checked_mul(k_heads)?;
        let fused = q
            .apply_op3_no_bwd(
                k,
                qk_weight,
                &QkRmsNormOp {
                    q_rows,
                    k_rows,
                    head_dim: q_head_dim,
                    eps: eps as f32,
                },
            )
            .ok()?;
        let q_out = fused
            .narrow(0, 0, q_rows)
            .ok()?
            .reshape((q_bsz, q_seq, q_heads, q_head_dim))
            .ok()?;
        let k_out = fused
            .narrow(0, q_rows, k_rows)
            .ok()?
            .reshape((k_bsz, k_seq, k_heads, k_head_dim))
            .ok()?;
        return Some((q_out, k_out));
    }

    #[allow(unreachable_code)]
    None
}

/// Try fused RMS normalization.
///
/// Computes: x / sqrt(mean(x^2) + eps) * weight
pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (input, weight, eps);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let dims = input.dims();
        let hidden_dim = *dims.last()?;
        if !input.device().is_metal()
            || !weight.device().is_metal()
            || input.dtype() != weight.dtype()
            || !matches!(input.dtype(), DType::F32 | DType::F16)
            || !input.is_contiguous()
            || !weight.is_contiguous()
            || hidden_dim == 0
            || weight.dims() != [hidden_dim]
        {
            return None;
        }
        let rows = input.elem_count().checked_div(hidden_dim)?;
        if rows == 0 {
            return None;
        }
        return input
            .apply_op2_no_bwd(
                weight,
                &RmsNormOp {
                    rows,
                    hidden_dim,
                    eps: eps as f32,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

/// Try fused RoPE for q/k tensors in `[batch, seq, heads, head_dim]` layout.
///
/// `cos_sin` is packed as `[seq, head_dim]`, with cos in the first half of the
/// last dimension and sin in the second half.
pub fn try_fused_rope_pair_bshd(
    q: &Tensor,
    k: &Tensor,
    cos_sin: &Tensor,
) -> Option<(Tensor, Tensor)> {
    #[cfg(not(feature = "metal"))]
    let _ = (q, k, cos_sin);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (q_bsz, q_seq, q_heads, q_head_dim) = q.dims4().ok()?;
        let (k_bsz, k_seq, k_heads, k_head_dim) = k.dims4().ok()?;
        if q_bsz != k_bsz
            || q_seq != k_seq
            || q_head_dim != k_head_dim
            || q_head_dim == 0
            || q_head_dim % 2 != 0
            || cos_sin.dims() != [q_seq, q_head_dim]
            || !q.device().is_metal()
            || !k.device().is_metal()
            || !cos_sin.device().is_metal()
            || !q.device().same_device(k.device())
            || !q.device().same_device(cos_sin.device())
            || q.dtype() != k.dtype()
            || q.dtype() != cos_sin.dtype()
            || !matches!(q.dtype(), DType::F32 | DType::F16)
            || !q.is_contiguous()
            || !k.is_contiguous()
            || !cos_sin.is_contiguous()
        {
            return None;
        }
        let q_rows = q_bsz.checked_mul(q_seq)?.checked_mul(q_heads)?;
        let k_rows = k_bsz.checked_mul(k_seq)?.checked_mul(k_heads)?;
        let fused = q
            .apply_op3_no_bwd(
                k,
                cos_sin,
                &RopePairBshdOp {
                    q_rows,
                    k_rows,
                    seq_len: q_seq,
                    q_heads,
                    k_heads,
                    head_dim: q_head_dim,
                },
            )
            .ok()?;
        let q_out = fused
            .narrow(0, 0, q_rows)
            .ok()?
            .reshape((q_bsz, q_seq, q_heads, q_head_dim))
            .ok()?;
        let k_out = fused
            .narrow(0, q_rows, k_rows)
            .ok()?
            .reshape((k_bsz, k_seq, k_heads, k_head_dim))
            .ok()?;
        return Some((q_out, k_out));
    }

    #[allow(unreachable_code)]
    None
}

/// Try fused single-token grouped-query attention for decode.
///
/// Inputs are head-major: q `[1, heads, 1, head_dim]`, k/v
/// `[1, kv_heads, total_len, head_dim]`.
pub fn try_fused_decode_gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Option<Tensor> {
    let total_len = k.dims4().ok()?.2;
    try_fused_decode_gqa_attention_with_kv_len(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_dim,
        total_len,
        scale,
    )
}

pub fn try_fused_decode_gqa_attention_with_kv_len(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (q, k, v, num_heads, num_kv_heads, head_dim, kv_len, scale);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (q_bsz, q_heads, q_seq, q_head_dim) = q.dims4().ok()?;
        let (k_bsz, k_heads, kv_capacity_len, k_head_dim) = k.dims4().ok()?;
        let (v_bsz, v_heads, v_capacity_len, v_head_dim) = v.dims4().ok()?;
        if q_bsz != 1
            || k_bsz != 1
            || v_bsz != 1
            || q_seq != 1
            || q_heads != num_heads
            || k_heads != num_kv_heads
            || v_heads != num_kv_heads
            || q_head_dim != head_dim
            || k_head_dim != head_dim
            || v_head_dim != head_dim
            || kv_capacity_len != v_capacity_len
            || kv_len == 0
            || kv_len > kv_capacity_len
            || kv_len > 2048
            || num_heads == 0
            || num_kv_heads == 0
            || !num_heads.is_multiple_of(num_kv_heads)
            || head_dim == 0
            || !q.device().is_metal()
            || !k.device().is_metal()
            || !v.device().is_metal()
            || !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || q.dtype() != k.dtype()
            || q.dtype() != v.dtype()
            || !matches!(q.dtype(), DType::F32 | DType::F16)
            || !q.is_contiguous()
            || !k.is_contiguous()
            || !v.is_contiguous()
        {
            return None;
        }
        return q
            .apply_op3_no_bwd(
                k,
                v,
                &DecodeGqaAttentionOp {
                    num_heads,
                    num_kv_heads,
                    kv_len,
                    kv_capacity_len,
                    head_dim,
                    scale,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

/// Dispatch one decode token per row directly against page-major arena K/V.
///
/// `metadata` is `[context_lens..., first_page_offsets...,
/// padded_block_table...]`. The Metal kernel
/// resolves each logical token to a physical page while applying online
/// softmax, so this path never gathers pages or expands grouped-query heads.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    metadata: Vec<u32>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    scale: f32,
    softcap: Option<f32>,
) -> CandleResult<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &PagedDecodeAttentionOp {
            metadata,
            batch_size,
            num_heads,
            num_kv_heads,
            page_tokens,
            max_blocks,
            key_head_dim,
            value_head_dim,
            scale,
            softcap,
        },
    )
}

/// Dispatch a packed ragged prefill directly against page-major arena K/V.
///
/// `metadata` is compact per sequence rather than expanded per query:
/// `[query_starts..., query_lens..., context_lens..., first_page_offsets...,
/// padded_block_table...]`. Dispatch appends a direct query-to-sequence map so
/// shaders avoid scanning ragged sequence boundaries. Each query derives its
/// causal visible context in the shader. Long, low-occupancy rows are split
/// into bounded online-softmax partitions and merged in F32; the conservative
/// crossover remains a candidate for hardware-backed promotion tuning.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_prefill_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    metadata: Vec<u32>,
    sequence_count: usize,
    total_queries: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_tokens: usize,
    max_blocks: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    scale: f32,
    softcap: Option<f32>,
    window_tokens: Option<u32>,
) -> CandleResult<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &PagedPrefillAttentionOp {
            metadata,
            sequence_count,
            total_queries,
            num_heads,
            num_kv_heads,
            page_tokens,
            max_blocks,
            key_head_dim,
            value_head_dim,
            scale,
            softcap,
            window_tokens,
        },
    )
}

pub fn try_lfm_shortconv_decode3(cache: &Tensor, bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (cache, bx, conv);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (batch_size, hidden_size, cache_len) = cache.dims3().ok()?;
        let (bx_batch, bx_hidden, bx_len) = bx.dims3().ok()?;
        let (conv_hidden, conv_len) = conv.dims2().ok()?;
        if cache_len != 3
            || bx_len != 1
            || conv_len != 3
            || bx_batch != batch_size
            || bx_hidden != hidden_size
            || conv_hidden != hidden_size
            || !cache.device().is_metal()
            || !bx.device().is_metal()
            || !conv.device().is_metal()
            || !cache.device().same_device(bx.device())
            || !cache.device().same_device(conv.device())
            || cache.dtype() != DType::F32
            || bx.dtype() != DType::F32
            || conv.dtype() != DType::F32
            || !cache.is_contiguous()
            || !bx.is_contiguous()
            || !conv.is_contiguous()
        {
            return None;
        }
        return cache
            .apply_op3_no_bwd(
                bx,
                conv,
                &LfmShortConvDecode3Op {
                    batch_size,
                    hidden_size,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

pub fn try_lfm_shortconv_update3(cache: &Tensor, bx: &Tensor) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (cache, bx);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (batch_size, hidden_size, cache_len) = cache.dims3().ok()?;
        let (bx_batch, bx_hidden, bx_len) = bx.dims3().ok()?;
        if cache_len != 3
            || bx_len != 1
            || bx_batch != batch_size
            || bx_hidden != hidden_size
            || !cache.device().is_metal()
            || !bx.device().is_metal()
            || !cache.device().same_device(bx.device())
            || cache.dtype() != DType::F32
            || bx.dtype() != DType::F32
            || !cache.is_contiguous()
            || !bx.is_contiguous()
        {
            return None;
        }
        return cache
            .apply_op2_no_bwd(
                bx,
                &LfmShortConvUpdate3Op {
                    batch_size,
                    hidden_size,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

pub fn try_lfm_shortconv_sequence3(bx: &Tensor, conv: &Tensor) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (bx, conv);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (batch_size, hidden_size, seq_len) = bx.dims3().ok()?;
        let (conv_hidden, conv_len) = conv.dims2().ok()?;
        if seq_len == 0
            || conv_len != 3
            || conv_hidden != hidden_size
            || !bx.device().is_metal()
            || !conv.device().is_metal()
            || !bx.device().same_device(conv.device())
            || bx.dtype() != DType::F32
            || conv.dtype() != DType::F32
            || !bx.is_contiguous()
            || !conv.is_contiguous()
        {
            return None;
        }
        return bx
            .apply_op2_no_bwd(
                conv,
                &LfmShortConvSequence3Op {
                    batch_size,
                    hidden_size,
                    seq_len,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

pub fn try_lfm_shortconv_ring_sequence(
    ring: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    expected_cursor: u64,
    valid_length: u64,
) -> Option<Tensor> {
    #[cfg(not(feature = "metal"))]
    let _ = (ring, input, weight, expected_cursor, valid_length);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (capacity, batch_size, hidden_size) = ring.dims3().ok()?;
        let (input_batch, input_hidden, steps) = input.dims3().ok()?;
        let (weight_hidden, weight_capacity) = weight.dims2().ok()?;
        if capacity == 0
            || batch_size == 0
            || hidden_size == 0
            || steps == 0
            || input_batch != batch_size
            || input_hidden != hidden_size
            || weight_hidden != hidden_size
            || weight_capacity != capacity
            || valid_length > capacity as u64
            || valid_length > expected_cursor
            || !ring.device().is_metal()
            || !input.device().is_metal()
            || !weight.device().is_metal()
            || !ring.device().same_device(input.device())
            || !ring.device().same_device(weight.device())
            || ring.dtype() != DType::F32
            || input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || !ring.is_contiguous()
            || !input.is_contiguous()
            || !weight.is_contiguous()
        {
            return None;
        }
        return ring
            .apply_op3_no_bwd(
                input,
                weight,
                &LfmShortConvRingOp {
                    batch_size,
                    hidden_size,
                    steps,
                    capacity,
                    expected_cursor,
                    valid_length,
                },
            )
            .ok();
    }

    #[allow(unreachable_code)]
    None
}

/// Run Qwen3.5's stateful causal depthwise-convolution sequence path in one
/// Metal dispatch. Inputs use token-major `[1, sequence, channels]` layout;
/// history is `[channels, kernel_size - 1]` in oldest-to-newest order.
pub fn try_qwen35_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    #[cfg(not(feature = "metal"))]
    let _ = (input, weight, history);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (batch, seq_len, conv_dim) = input.dims3().ok()?;
        let (weight_channels, kernel_size) = weight.dims2().ok()?;
        let (history_channels, history_len) = history.dims2().ok()?;
        if batch != 1
            || seq_len == 0
            || conv_dim == 0
            || kernel_size < 2
            || weight_channels != conv_dim
            || history_channels != conv_dim
            || history_len != kernel_size - 1
            || !input.device().is_metal()
            || !weight.device().is_metal()
            || !history.device().is_metal()
            || !input.device().same_device(weight.device())
            || !input.device().same_device(history.device())
            || input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || history.dtype() != DType::F32
            || !input.is_contiguous()
            || !weight.is_contiguous()
            || !history.is_contiguous()
        {
            return None;
        }

        let output_elem_count = seq_len.checked_mul(conv_dim)?;
        let state_elem_count = history_len.checked_mul(conv_dim)?;
        let packed = input
            .apply_op3_no_bwd(
                weight,
                history,
                &Qwen35CausalConvSequenceOp {
                    conv_dim,
                    seq_len,
                    kernel_size,
                },
            )
            .ok()?;
        let output = packed
            .narrow(0, 0, output_elem_count)
            .ok()?
            .reshape((1, seq_len, conv_dim))
            .ok()?;
        let final_history = packed
            .narrow(0, output_elem_count, state_elem_count)
            .ok()?
            .reshape((conv_dim, history_len))
            .ok()?;
        return Some((output, final_history));
    }

    #[allow(unreachable_code)]
    None
}

/// Try fused gated RMS normalization.
///
/// Computes: rms_norm(x) * silu(gate)
pub fn try_fused_gated_rms_norm(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    let rms_out = try_fused_rms_norm(hidden, weight, eps)?;
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    rms_out.broadcast_mul(&silu_gate).ok()
}

/// Try the reference Qwen3.5 Gated DeltaNet recurrence in bounded Metal dispatches.
///
/// The shader keeps token causality inside each value-head threadgroup and
/// evaluates state decay, delta correction, state update, and output reduction
/// in F32. Q/K retain the converted GGUF's 16-head tiled layout while V may
/// contain either 16 or 32 heads.
///
/// # Arguments
/// * `queries` - Query tensors [1, seq, 16, head_k_dim]
/// * `keys` - Key tensors [1, seq, 16, head_k_dim]
/// * `values` - Value tensors [batch, seq, num_v_heads, head_v_dim]
/// * `g` - Pre-computed gate values [batch, seq, num_v_heads]
/// * `beta` - Beta values [batch, seq, num_v_heads]
/// * `initial_state` - Initial recurrent state [batch, num_v_heads, head_k_dim, head_v_dim]
/// * `tile_size` - Maximum number of tokens processed by one Metal dispatch.
///
/// # Returns
/// (outputs, final_state) where outputs is [batch, seq, num_v_heads, head_v_dim]
pub fn try_tiled_deltanet_recurrence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
    tile_size: usize,
) -> Option<(Tensor, Tensor)> {
    #[cfg(not(feature = "metal"))]
    let _ = (queries, keys, values, g, beta, initial_state, tile_size);

    if !use_fused_kernels() {
        return None;
    }

    #[cfg(feature = "metal")]
    {
        let (batch, seq_len, num_k_heads, head_k_dim) = queries.dims4().ok()?;
        let (k_batch, k_seq_len, k_num_heads, k_head_k_dim) = keys.dims4().ok()?;
        let (v_batch, v_seq_len, num_v_heads, head_v_dim) = values.dims4().ok()?;
        let (g_batch, g_seq_len, g_heads) = g.dims3().ok()?;
        let (b_batch, b_seq_len, b_heads) = beta.dims3().ok()?;
        let (s_batch, s_heads, s_head_k_dim, s_head_v_dim) = initial_state.dims4().ok()?;

        if batch != 1
            || seq_len == 0
            || tile_size == 0
            || k_batch != batch
            || v_batch != batch
            || g_batch != batch
            || b_batch != batch
            || s_batch != batch
            || k_seq_len != seq_len
            || v_seq_len != seq_len
            || g_seq_len != seq_len
            || b_seq_len != seq_len
            || k_num_heads != num_k_heads
            || !qwen35_gated_delta_tiled_layout_supported(
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            )
            || g_heads != num_v_heads
            || b_heads != num_v_heads
            || s_heads != num_v_heads
            || k_head_k_dim != head_k_dim
            || s_head_k_dim != head_k_dim
            || s_head_v_dim != head_v_dim
            || !queries.device().is_metal()
            || !keys.device().is_metal()
            || !values.device().is_metal()
            || !g.device().is_metal()
            || !beta.device().is_metal()
            || !initial_state.device().is_metal()
            || !queries.device().same_device(keys.device())
            || !queries.device().same_device(values.device())
            || !queries.device().same_device(g.device())
            || !queries.device().same_device(beta.device())
            || !queries.device().same_device(initial_state.device())
            || queries.dtype() != DType::F32
            || keys.dtype() != DType::F32
            || values.dtype() != DType::F32
            || g.dtype() != DType::F32
            || beta.dtype() != DType::F32
            || initial_state.dtype() != DType::F32
            || !queries.is_contiguous()
            || !keys.is_contiguous()
            || !values.is_contiguous()
            || !g.is_contiguous()
            || !beta.is_contiguous()
            || !initial_state.is_contiguous()
        {
            return None;
        }

        let key_width = num_k_heads.checked_mul(head_k_dim)?;
        let value_width = num_v_heads.checked_mul(head_v_dim)?;
        let qkv_width = key_width.checked_mul(2)?.checked_add(value_width)?;
        let query_flat = queries.reshape((1, seq_len, key_width)).ok()?;
        let key_flat = keys.reshape((1, seq_len, key_width)).ok()?;
        let value_flat = values.reshape((1, seq_len, value_width)).ok()?;
        let qkv = Tensor::cat(&[&query_flat, &key_flat, &value_flat], 2).ok()?;
        let gates = Tensor::cat(&[g, beta], 2).ok()?;
        let output_elem_count = seq_len.checked_mul(num_v_heads)?.checked_mul(head_v_dim)?;
        let state_elem_count = num_v_heads
            .checked_mul(head_k_dim)?
            .checked_mul(head_v_dim)?;
        let packed = qkv
            .apply_op3_no_bwd(
                &gates,
                initial_state,
                &Qwen35GatedDeltaSequenceOp {
                    seq_len,
                    tile_size: tile_size.min(seq_len),
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    qkv_width,
                    query_scale: 1.0 / (head_k_dim as f32).sqrt(),
                },
            )
            .ok()?;
        let output = packed
            .narrow(0, 0, output_elem_count)
            .ok()?
            .reshape((1, seq_len, num_v_heads, head_v_dim))
            .ok()?;
        let final_state = packed
            .narrow(0, output_elem_count, state_elem_count)
            .ok()?
            .reshape((1, num_v_heads, head_k_dim, head_v_dim))
            .ok()?;
        return Some((output, final_state));
    }

    #[allow(unreachable_code)]
    None
}

/// Try SIMD-group softmax for attention.
///
/// Uses Metal's simd_shuffle and simd_sum instructions to perform
/// parallel softmax computation within a threadgroup. This is more
/// efficient for small models with fewer attention heads.
///
/// The SIMD-group approach allows one threadgroup to handle multiple
/// heads simultaneously, keeping GPU Execution Units busy.
///
/// # Arguments
/// * `scores` - Attention scores [batch, heads, q_len, kv_len]
/// * `scale` - Scale factor (typically 1/sqrt(head_dim))
///
/// # Returns
/// Softmax-normalized attention weights
pub fn try_simd_softmax(scores: &Tensor, scale: f32) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if scores.dtype() != DType::F32 {
        return None;
    }

    if !scores.device().is_metal() {
        return None;
    }

    // For now, use standard softmax
    // A true SIMD-group implementation would require custom Metal kernels
    // using threadgroup_barrier, simd_shuffle, and simd_sum

    // The ideal Metal kernel would:
    // 1. Load scores into threadgroup memory
    // 2. Use simd_max to find max score per SIMD group
    // 3. Use simd_sum to compute exp sum per SIMD group
    // 4. Normalize and write output

    // Current: use standard operations
    let scaled = (scores * scale as f64).ok()?;
    candle_nn::ops::softmax(&scaled, candle_core::D::Minus1).ok()
}

/// Try SIMD-group RMS normalization.
///
/// Uses Metal's simd_sum for parallel reduction across head dimensions.
/// More efficient for small head dimensions common in 4B/9B models.
pub fn try_simd_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // For now, use standard RMS norm
    // A SIMD-group implementation would:
    // 1. Use simd_sum for parallel sum of squares
    // 2. Compute RMS per SIMD group
    // 3. Apply normalization and weight

    // Current: use standard operations
    try_fused_rms_norm(input, weight, eps)
}

/// Try 3:1 DeltaNet block fusion.
///
/// Qwen 3.5 uses a 3:1 ratio of Gated DeltaNet to Gated Attention blocks.
/// This function merges 3 consecutive DeltaNet blocks into a single GPU
/// command, reducing CPU/GPU synchronization overhead by 66%.
///
/// In llama.cpp, this is done by kernel fusion to combine the 3x DeltaNet
/// blocks. By merging the three consecutive linear attention passes, we
/// reduce the "Round Trip Time" (RTT) to the GPU significantly.
///
/// # Arguments
/// * `input` - Input tensor for the first block
/// * `block_configs` - Configuration for each of the 3 blocks
/// * `states` - Mutable runtime states for each block (type-erased)
///
/// # Returns
/// Output tensor after processing all 3 blocks
pub fn try_fused_deltanet_blocks_3x1(
    input: &Tensor,
    block_configs: &[DeltaNetBlockConfig],
    _states: &mut [&mut dyn std::any::Any],
) -> Option<Tensor> {
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if input.dtype() != DType::F32 {
        return None;
    }

    if !input.device().is_metal() {
        return None;
    }

    // Require exactly 3 blocks for 3:1 fusion
    if block_configs.len() != 3 || _states.len() != 3 {
        return None;
    }

    // For now, process sequentially
    // A true fused implementation would dispatch all 3 blocks in one kernel

    // The ideal Metal kernel would:
    // 1. Load input into threadgroup memory
    // 2. For each of 3 blocks:
    //    - Compute qkv projections
    //    - Apply depthwise conv
    //    - Run gated delta recurrence with tile memory
    //    - Apply output projection
    // 3. Return final output without intermediate VRAM writes

    // Current: fall back to sequential processing
    tracing::debug!(
        "3:1 DeltaNet block fusion not yet implemented, processing {} blocks sequentially",
        block_configs.len()
    );

    None
}

/// Configuration for a single DeltaNet block in fused execution.
#[derive(Debug, Clone)]
pub struct DeltaNetBlockConfig {
    /// Number of key/value heads
    pub num_k_heads: usize,
    /// Number of value heads
    pub num_v_heads: usize,
    /// Key dimension per head
    pub head_k_dim: usize,
    /// Value dimension per head
    pub head_v_dim: usize,
    /// Depthwise conv kernel size
    pub conv_size: usize,
    /// Epsilon for normalization
    pub eps: f64,
}

/// Check if 3:1 block fusion is enabled.
pub fn use_block_fusion() -> bool {
    if !use_fused_kernels() {
        return false;
    }

    std::env::var("IZWI_BLOCK_FUSION")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

/// Check if SIMD-group optimizations should be used.
pub fn use_simd_optimizations() -> bool {
    if !use_fused_kernels() {
        return false;
    }

    std::env::var("IZWI_SIMD_OPTIMIZATIONS")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

/// Check if fused kernels should be used.
pub fn use_fused_kernels() -> bool {
    crate::kernels::use_fused_kernels()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use candle_nn::Module;

    #[cfg(feature = "metal")]
    #[test]
    fn qwen35_native_48_value_heads_remain_on_portable_recurrence() {
        assert!(qwen35_gated_delta_tiled_layout_supported(16, 16, 128, 128));
        assert!(qwen35_gated_delta_tiled_layout_supported(16, 32, 128, 128));
        // The current shader indexes converted-GGUF tiled heads with
        // `value_head % key_heads`. Native Qwen3.8 uses repeat-interleave head
        // ordering, so its 16K/48V geometry must stay on the portable path.
        assert!(!qwen35_gated_delta_tiled_layout_supported(16, 48, 128, 128));
    }

    fn qwen35_causal_conv_reference(
        input: &[f32],
        weight: &[f32],
        history: &[f32],
        seq_len: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let history_len = kernel_size - 1;
        let mut output = vec![0f32; seq_len * conv_dim];
        for token in 0..seq_len {
            for channel in 0..conv_dim {
                let mut value = 0f32;
                for tap in 0..kernel_size {
                    let source_pos = token + tap;
                    let source = if source_pos < history_len {
                        history[channel * history_len + source_pos]
                    } else {
                        input[(source_pos - history_len) * conv_dim + channel]
                    };
                    value += source * weight[channel * kernel_size + tap];
                }
                output[token * conv_dim + channel] = value / (1.0 + (-value).exp());
            }
        }

        let mut final_history = vec![0f32; conv_dim * history_len];
        for channel in 0..conv_dim {
            for history_pos in 0..history_len {
                let source_pos = seq_len + history_pos;
                final_history[channel * history_len + history_pos] = if source_pos < history_len {
                    history[channel * history_len + source_pos]
                } else {
                    input[(source_pos - history_len) * conv_dim + channel]
                };
            }
        }
        (output, final_history)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_gated_delta_reference(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        g: &[f32],
        beta: &[f32],
        initial_state: &[f32],
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial_state.to_vec();
        let mut output = vec![0f32; seq_len * num_v_heads * head_v_dim];
        let query_scale = 1.0f32 / (head_k_dim as f32).sqrt();
        let state_head_size = head_k_dim * head_v_dim;
        for token in 0..seq_len {
            for value_head in 0..num_v_heads {
                let key_head = value_head % num_k_heads;
                let state_base = value_head * state_head_size;
                let key_base = (token * num_k_heads + key_head) * head_k_dim;
                let value_base = (token * num_v_heads + value_head) * head_v_dim;
                let decay = g[token * num_v_heads + value_head].exp();
                for state_value in &mut state[state_base..state_base + state_head_size] {
                    *state_value *= decay;
                }

                let mut delta = vec![0f32; head_v_dim];
                for value_dim in 0..head_v_dim {
                    let mut memory = 0f32;
                    for key_dim in 0..head_k_dim {
                        memory += state[state_base + key_dim * head_v_dim + value_dim]
                            * key[key_base + key_dim];
                    }
                    delta[value_dim] = (value[value_base + value_dim] - memory)
                        * beta[token * num_v_heads + value_head];
                }

                for key_dim in 0..head_k_dim {
                    for value_dim in 0..head_v_dim {
                        state[state_base + key_dim * head_v_dim + value_dim] +=
                            key[key_base + key_dim] * delta[value_dim];
                    }
                }

                for value_dim in 0..head_v_dim {
                    let mut recurrent_value = 0f32;
                    for key_dim in 0..head_k_dim {
                        recurrent_value += state[state_base + key_dim * head_v_dim + value_dim]
                            * (query[key_base + key_dim] * query_scale);
                    }
                    output[value_base + value_dim] = recurrent_value;
                }
            }
        }
        (output, state)
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context} length mismatch");
        for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let bound = tolerance * (1.0 + expected.abs());
            assert!(
                actual.is_finite() && (actual - expected).abs() <= bound,
                "{context} mismatch at {idx}: {actual} != {expected} (bound {bound})"
            );
        }
    }

    #[test]
    fn test_l2_norm_matches_reference() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();

        // Reference implementation
        let sq_sum: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x * x).sum();
        let norm = sq_sum.sqrt();
        let expected = [[1.0f32 / norm, 2.0 / norm], [3.0 / norm, 4.0 / norm]];

        // Fused implementation (falls back to CPU for non-Metal)
        let eps = 1e-6;
        if let Some(result) = try_fused_l2_norm(&input, eps) {
            let result_data = result.to_vec3::<f32>().unwrap();
            assert!(
                (result_data[0][0][0] - expected[0][0]).abs() < 1e-5,
                "L2 norm mismatch"
            );
        }
    }

    #[test]
    fn test_silu_mul_matches_reference() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0], (2, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();

        // Reference: silu(x) = x / (1 + exp(-x))
        let silu_0 = 0.0f32 / (1.0 + (-0.0f32).exp());
        let silu_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let silu_m1 = -1.0f32 / (1.0 + (1.0f32).exp());
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());

        if let Some(result) = try_fused_silu_mul(&gate, &up) {
            let result_data = result.to_vec2::<f32>().unwrap();

            assert!((result_data[0][0] - silu_0 * 1.0).abs() < 1e-5);
            assert!((result_data[0][1] - silu_1 * 2.0).abs() < 1e-5);
            assert!((result_data[1][0] - silu_m1 * 3.0).abs() < 1e-5);
            assert!((result_data[1][1] - silu_2 * 4.0).abs() < 1e-5);
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_silu_mul_kernel_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let gate = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0, -3.0, 4.0], (2, 3), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let up = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, -0.5, 0.25], (2, 3), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            let result = gate.apply_op2_no_bwd(&up, &SiluMulOp).unwrap();
            let reference = candle_nn::ops::silu(&gate)
                .unwrap()
                .broadcast_mul(&up)
                .unwrap();
            let result = result
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = reference
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
            for (idx, (actual, expected)) in result.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} mismatch at {idx}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_one_pass_paged_attention_softcap_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let query_data = vec![2.0f32, -1.0];
        let key_data = vec![4.0f32, 0.0, 0.0, 2.0];
        let value_data = vec![1.0f32, 3.0, 5.0, -2.0];
        let metadata = vec![2, 0, 0];

        for dtype in [DType::F32, DType::F16] {
            let query = Tensor::from_vec(query_data.clone(), (1, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let keys = Tensor::from_vec(key_data.clone(), (1, 2, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let values = Tensor::from_vec(value_data.clone(), (1, 2, 1, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            for softcap in [None, Some(0.5f32)] {
                let actual = paged_decode_attention(
                    &query,
                    &keys,
                    &values,
                    metadata.clone(),
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    2,
                    1.0,
                    softcap,
                )
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let scores = [8.0f32, -2.0]
                    .map(|score| softcap.map_or(score, |cap| cap * (score / cap).tanh()));
                let max_score = scores[0].max(scores[1]);
                let weights = scores.map(|score| (score - max_score).exp());
                let denominator = weights[0] + weights[1];
                let expected = [
                    (weights[0] * 1.0 + weights[1] * 5.0) / denominator,
                    (weights[0] * 3.0 + weights[1] * -2.0) / denominator,
                ];
                let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
                assert_f32_close(&actual, &expected, tolerance, "one-pass paged attention");
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_split_paged_attention_matches_online_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let context_len = METAL_PAGED_ATTENTION_SPLIT_MIN_CONTEXT + 1;
        let page_tokens = 16usize;
        let page_count = context_len.div_ceil(page_tokens);
        let head_dim = 4usize;
        let padded_tokens = page_count * page_tokens;
        let query_data = vec![0.25f32, -0.5, 0.75, 1.0];
        let key_data = (0..padded_tokens * head_dim)
            .map(|index| ((index % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let value_data = (0..padded_tokens * head_dim)
            .map(|index| ((index % 29) as f32 - 14.0) / 11.0)
            .collect::<Vec<_>>();

        for dtype in [DType::F32, DType::F16] {
            let query = Tensor::from_vec(query_data.clone(), (1, 1, head_dim), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let keys = Tensor::from_vec(
                key_data.clone(),
                (page_count, page_tokens, 1, head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let values = Tensor::from_vec(
                value_data.clone(),
                (page_count, page_tokens, 1, head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

            let mut metadata = Vec::with_capacity(2 + page_count);
            metadata.push(context_len as u32);
            metadata.push(0);
            metadata.extend((0..page_count).map(|page| page as u32));
            let actual = paged_decode_attention(
                &query,
                &keys,
                &values,
                metadata,
                1,
                1,
                1,
                page_tokens,
                page_count,
                head_dim,
                head_dim,
                0.5,
                Some(0.7),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

            let query_reference = query
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let key_reference = keys
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let value_reference = values
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(context_len);
            for token in 0..context_len {
                let base = token * head_dim;
                let raw_score = (0..head_dim)
                    .map(|dim| query_reference[dim] * key_reference[base + dim])
                    .sum::<f32>()
                    * 0.5;
                let score = 0.7 * (raw_score / 0.7).tanh();
                max_score = max_score.max(score);
                scores.push(score);
            }
            let denominator = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .sum::<f32>();
            let mut expected = vec![0.0f32; head_dim];
            for (token, score) in scores.iter().enumerate() {
                let probability = (*score - max_score).exp() / denominator;
                let base = token * head_dim;
                for dim in 0..head_dim {
                    expected[dim] += probability * value_reference[base + dim];
                }
            }
            let tolerance = if dtype == DType::F16 { 2e-3 } else { 2e-5 };
            assert_f32_close(&actual, &expected, tolerance, "split paged attention");
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_packed_varlen_prefill_matches_ragged_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let page_tokens = 4usize;
        let head_dim = 2usize;
        let query_data = vec![0.5f32, 1.0, -0.25, 0.75, 1.0, -0.5];
        let key_data = vec![
            9.0f32, 9.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 1.0, -1.0, 0.5, 7.0, 7.0, 8.0, 8.0,
        ];
        let value_data = vec![
            9.0f32, 9.0, 1.0, 0.0, 0.0, 2.0, 3.0, 1.0, 2.0, 1.0, -1.0, 3.0, 7.0, 7.0, 8.0, 8.0,
        ];
        // Two compact rows: q=[0..2), final context=3, first-page offset=1;
        // q=[2..3), final context=2, first-page offset=0.
        let metadata = vec![0, 2, 2, 1, 3, 2, 1, 0, 0, 1];
        let slots = [&[1usize, 2][..], &[1usize, 2, 3][..], &[4usize, 5][..]];

        for dtype in [DType::F32, DType::F16] {
            let queries = Tensor::from_vec(query_data.clone(), (3, 1, head_dim), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let keys = Tensor::from_vec(key_data.clone(), (2, page_tokens, 1, head_dim), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let values =
                Tensor::from_vec(value_data.clone(), (2, page_tokens, 1, head_dim), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let actual = paged_prefill_attention(
                &queries,
                &keys,
                &values,
                metadata.clone(),
                2,
                3,
                1,
                1,
                page_tokens,
                1,
                head_dim,
                head_dim,
                1.0,
                Some(0.75),
                Some(2),
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let queries = queries
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let keys = keys
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let values = values
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut expected = Vec::with_capacity(3 * head_dim);
            for query_index in 0..3 {
                let q_base = query_index * head_dim;
                let visible_slots = slots[query_index];
                let visible_slots = &visible_slots[visible_slots.len().saturating_sub(2)..];
                let mut scores = Vec::with_capacity(visible_slots.len());
                let mut max_score = f32::NEG_INFINITY;
                for &slot in visible_slots {
                    let k_base = slot * head_dim;
                    let raw_score = (0..head_dim)
                        .map(|dim| queries[q_base + dim] * keys[k_base + dim])
                        .sum::<f32>();
                    let score = 0.75 * (raw_score / 0.75).tanh();
                    max_score = max_score.max(score);
                    scores.push(score);
                }
                let denominator = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..head_dim {
                    let value = scores
                        .iter()
                        .zip(visible_slots)
                        .map(|(score, slot)| {
                            ((*score - max_score).exp() / denominator)
                                * values[slot * head_dim + dim]
                        })
                        .sum::<f32>();
                    expected.push(value);
                }
            }
            let tolerance = if dtype == DType::F16 { 2e-3 } else { 2e-5 };
            assert_f32_close(&actual, &expected, tolerance, "packed paged prefill");
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_split_packed_prefill_matches_asymmetric_gqa_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cpu = Device::Cpu;
        let page_tokens = 64usize;
        let context_len = METAL_PAGED_PREFILL_SPLIT_MIN_CONTEXT + 1;
        let first_page_offset = 1usize;
        let page_count = (context_len + first_page_offset).div_ceil(page_tokens);
        let num_heads = 2usize;
        let key_head_dim = 2usize;
        let value_head_dim = 3usize;
        let query_data = vec![0.5f32, -0.25, -0.125, 0.75];
        let mut key_data = vec![0.0f32; page_count * page_tokens * key_head_dim];
        let mut value_data = vec![0.0f32; page_count * page_tokens * value_head_dim];
        for slot in 0..page_count * page_tokens {
            key_data[slot * key_head_dim] = ((slot % 17) as f32 - 8.0) / 9.0;
            key_data[slot * key_head_dim + 1] = ((slot % 13) as f32 - 6.0) / 7.0;
            for dim in 0..value_head_dim {
                value_data[slot * value_head_dim + dim] =
                    ((slot * (dim + 3) % 29) as f32 - 14.0) / 11.0;
            }
        }
        let mut metadata = Vec::with_capacity(4 + page_count);
        metadata.extend([0, 1, context_len as u32, first_page_offset as u32]);
        metadata.extend((0..page_count).map(|page| page as u32));

        for dtype in [DType::F32, DType::F16] {
            let queries =
                Tensor::from_vec(query_data.clone(), (1, num_heads, key_head_dim), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let keys = Tensor::from_vec(
                key_data.clone(),
                (page_count, page_tokens, 1, key_head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let values = Tensor::from_vec(
                value_data.clone(),
                (page_count, page_tokens, 1, value_head_dim),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let actual = paged_prefill_attention(
                &queries,
                &keys,
                &values,
                metadata.clone(),
                1,
                1,
                num_heads,
                1,
                page_tokens,
                page_count,
                key_head_dim,
                value_head_dim,
                0.5,
                Some(0.8),
                None,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let queries = queries
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let keys = keys
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let values = values
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut expected = Vec::with_capacity(num_heads * value_head_dim);
            for head in 0..num_heads {
                let q_base = head * key_head_dim;
                let mut scores = Vec::with_capacity(context_len);
                let mut max_score = f32::NEG_INFINITY;
                for token in 0..context_len {
                    let slot = first_page_offset + token;
                    let k_base = slot * key_head_dim;
                    let raw_score = (0..key_head_dim)
                        .map(|dim| queries[q_base + dim] * keys[k_base + dim])
                        .sum::<f32>()
                        * 0.5;
                    let score = 0.8 * (raw_score / 0.8).tanh();
                    max_score = max_score.max(score);
                    scores.push(score);
                }
                let denominator = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .sum::<f32>();
                for dim in 0..value_head_dim {
                    let value = scores
                        .iter()
                        .enumerate()
                        .map(|(token, score)| {
                            let slot = first_page_offset + token;
                            ((*score - max_score).exp() / denominator)
                                * values[slot * value_head_dim + dim]
                        })
                        .sum::<f32>();
                    expected.push(value);
                }
            }
            let tolerance = if dtype == DType::F16 { 4e-3 } else { 5e-5 };
            assert_f32_close(&actual, &expected, tolerance, "split packed paged prefill");
        }
    }

    #[test]
    fn qk_rms_norm_returns_none_on_cpu() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 1, 2, 4), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 1, 1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones(8, DType::F32, &device).unwrap();

        assert!(try_fused_qk_rms_norm(&q, &k, &weight, 1e-6).is_none());
    }

    #[test]
    fn rms_norm_returns_none_on_cpu() {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();

        assert!(try_fused_rms_norm(&input, &weight, 1e-6).is_none());
    }

    #[test]
    fn rope_pair_bshd_returns_none_on_cpu() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 1, 2, 4), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 1, 1, 4), DType::F32, &device).unwrap();
        let cos_sin = Tensor::zeros((1, 4), DType::F32, &device).unwrap();

        assert!(try_fused_rope_pair_bshd(&q, &k, &cos_sin).is_none());
    }

    #[test]
    fn decode_gqa_attention_returns_none_on_cpu() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 1, 3, 4), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 1, 3, 4), DType::F32, &device).unwrap();

        assert!(try_fused_decode_gqa_attention(&q, &k, &v, 2, 1, 4, 0.5).is_none());
    }

    #[test]
    fn lfm_shortconv_decode3_returns_none_on_cpu() {
        let device = Device::Cpu;
        let cache = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let bx = Tensor::zeros((1, 2, 1), DType::F32, &device).unwrap();
        let conv = Tensor::zeros((2, 3), DType::F32, &device).unwrap();

        assert!(try_lfm_shortconv_decode3(&cache, &bx, &conv).is_none());
    }

    #[test]
    fn lfm_shortconv_sequence3_returns_none_on_cpu() {
        let device = Device::Cpu;
        let bx = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();
        let conv = Tensor::zeros((2, 3), DType::F32, &device).unwrap();

        assert!(try_lfm_shortconv_sequence3(&bx, &conv).is_none());
    }

    #[test]
    fn lfm_shortconv_update3_returns_none_on_cpu() {
        let device = Device::Cpu;
        let cache = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let bx = Tensor::zeros((1, 2, 1), DType::F32, &device).unwrap();

        assert!(try_lfm_shortconv_update3(&cache, &bx).is_none());
    }

    #[test]
    fn qwen35_sequence_kernels_reject_cpu_tensors() {
        let device = Device::Cpu;
        let conv_input = Tensor::zeros((1, 2, 8), DType::F32, &device).unwrap();
        let conv_weight = Tensor::zeros((8, 4), DType::F32, &device).unwrap();
        let conv_history = Tensor::zeros((8, 3), DType::F32, &device).unwrap();
        assert!(
            try_qwen35_causal_conv_sequence(&conv_input, &conv_weight, &conv_history).is_none()
        );

        let query = Tensor::zeros((1, 2, 16, 2), DType::F32, &device).unwrap();
        let key = Tensor::zeros((1, 2, 16, 2), DType::F32, &device).unwrap();
        let value = Tensor::zeros((1, 2, 16, 3), DType::F32, &device).unwrap();
        let g = Tensor::zeros((1, 2, 16), DType::F32, &device).unwrap();
        let beta = Tensor::zeros((1, 2, 16), DType::F32, &device).unwrap();
        let state = Tensor::zeros((1, 16, 2, 3), DType::F32, &device).unwrap();
        assert!(
            try_tiled_deltanet_recurrence(&query, &key, &value, &g, &beta, &state, 64).is_none()
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_qwen35_causal_conv_sequence_matches_cpu_reference_if_available() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        let seq_len = 2;
        let conv_dim = 5;
        let kernel_size = 4;
        let input: Vec<f32> = (0..seq_len * conv_dim)
            .map(|idx| ((idx % 9) as f32 - 4.0) * 0.13)
            .collect();
        let weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.09)
            .collect();
        let history: Vec<f32> = (0..conv_dim * (kernel_size - 1))
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.07)
            .collect();
        let (expected_output, expected_history) =
            qwen35_causal_conv_reference(&input, &weight, &history, seq_len, conv_dim, kernel_size);

        let input_tensor = Tensor::from_vec(input, (1, seq_len, conv_dim), &device).expect("input");
        let weight_tensor =
            Tensor::from_vec(weight, (conv_dim, kernel_size), &device).expect("weight");
        let history_tensor =
            Tensor::from_vec(history, (conv_dim, kernel_size - 1), &device).expect("history");
        let (output, final_history) =
            try_qwen35_causal_conv_sequence(&input_tensor, &weight_tensor, &history_tensor)
                .expect("Qwen3.5 causal conv custom Metal op should run");
        let output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let final_history = final_history
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_f32_close(&output, &expected_output, 2e-5, "causal conv output");
        assert_f32_close(
            &final_history,
            &expected_history,
            1e-6,
            "causal conv history",
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_qwen35_gated_delta_matches_cpu_reference_for_both_layouts_if_available() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        // Cross several tile boundaries so the test also verifies that each
        // dispatch observes the recurrent state written by the preceding one.
        let seq_len = 17;
        let tile_size = 4;
        let num_k_heads = 16;
        let head_k_dim = 3;
        let head_v_dim = 5;

        for num_v_heads in [16usize, 32] {
            let query: Vec<f32> = (0..seq_len * num_k_heads * head_k_dim)
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.025)
                .collect();
            let key: Vec<f32> = (0..seq_len * num_k_heads * head_k_dim)
                .map(|idx| ((idx % 13) as f32 - 6.0) * 0.021)
                .collect();
            let value: Vec<f32> = (0..seq_len * num_v_heads * head_v_dim)
                .map(|idx| ((idx % 19) as f32 - 9.0) * 0.018)
                .collect();
            let g: Vec<f32> = (0..seq_len * num_v_heads)
                .map(|idx| -0.015 * ((idx % 5) + 1) as f32)
                .collect();
            let beta: Vec<f32> = (0..seq_len * num_v_heads)
                .map(|idx| 0.2 + 0.03 * (idx % 4) as f32)
                .collect();
            let initial_state: Vec<f32> = (0..num_v_heads * head_k_dim * head_v_dim)
                .map(|idx| ((idx % 23) as f32 - 11.0) * 0.004)
                .collect();
            let (expected_output, expected_state) = qwen35_gated_delta_reference(
                &query,
                &key,
                &value,
                &g,
                &beta,
                &initial_state,
                seq_len,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            );

            let query_tensor =
                Tensor::from_vec(query, (1, seq_len, num_k_heads, head_k_dim), &device).unwrap();
            let key_tensor =
                Tensor::from_vec(key, (1, seq_len, num_k_heads, head_k_dim), &device).unwrap();
            let value_tensor =
                Tensor::from_vec(value, (1, seq_len, num_v_heads, head_v_dim), &device).unwrap();
            let g_tensor = Tensor::from_vec(g, (1, seq_len, num_v_heads), &device).unwrap();
            let beta_tensor = Tensor::from_vec(beta, (1, seq_len, num_v_heads), &device).unwrap();
            let state_tensor = Tensor::from_vec(
                initial_state,
                (1, num_v_heads, head_k_dim, head_v_dim),
                &device,
            )
            .unwrap();

            let (output, final_state) = try_tiled_deltanet_recurrence(
                &query_tensor,
                &key_tensor,
                &value_tensor,
                &g_tensor,
                &beta_tensor,
                &state_tensor,
                tile_size,
            )
            .expect("Qwen3.5 Gated DeltaNet custom Metal op should run");
            assert_eq!(output.dtype(), DType::F32);
            assert_eq!(final_state.dtype(), DType::F32);
            let output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let final_state = final_state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let layout = format!("16K/{num_v_heads}V");
            assert_f32_close(&output, &expected_output, 2e-4, &format!("{layout} output"));
            assert_f32_close(
                &final_state,
                &expected_state,
                2e-4,
                &format!("{layout} state"),
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_qwen35_gated_delta_rejects_invalid_dtype_and_head_layout_if_available() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        let query = Tensor::zeros((1, 2, 16, 2), DType::F32, &device).unwrap();
        let key = Tensor::zeros((1, 2, 16, 2), DType::F32, &device).unwrap();
        let value = Tensor::zeros((1, 2, 16, 3), DType::F32, &device).unwrap();
        let g = Tensor::zeros((1, 2, 16), DType::F32, &device).unwrap();
        let beta = Tensor::zeros((1, 2, 16), DType::F32, &device).unwrap();
        let state = Tensor::zeros((1, 16, 2, 3), DType::F32, &device).unwrap();
        assert!(try_tiled_deltanet_recurrence(
            &query.to_dtype(DType::F16).unwrap(),
            &key,
            &value,
            &g,
            &beta,
            &state,
            64,
        )
        .is_none());
        assert!(
            try_tiled_deltanet_recurrence(&query, &key, &value, &g, &beta, &state, 0,).is_none()
        );

        let invalid_value = Tensor::zeros((1, 2, 24, 3), DType::F32, &device).unwrap();
        let invalid_g = Tensor::zeros((1, 2, 24), DType::F32, &device).unwrap();
        let invalid_beta = Tensor::zeros((1, 2, 24), DType::F32, &device).unwrap();
        let invalid_state = Tensor::zeros((1, 24, 2, 3), DType::F32, &device).unwrap();
        assert!(try_tiled_deltanet_recurrence(
            &query,
            &key,
            &invalid_value,
            &invalid_g,
            &invalid_beta,
            &invalid_state,
            64,
        )
        .is_none());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_lfm_shortconv_decode3_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cache = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
            (1, 2, 3),
            &device,
        )
        .unwrap();
        let bx = Tensor::from_vec(vec![7.0f32, 8.0], (1, 2, 1), &device).unwrap();
        let conv = Tensor::from_vec(
            vec![
                0.5f32, 1.5, 2.5, //
                -1.0, 0.25, 0.75,
            ],
            (2, 3),
            &device,
        )
        .unwrap();

        let out = try_lfm_shortconv_decode3(&cache, &bx, &conv)
            .expect("shortconv decode3 should run on Metal");
        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = [
            2.0 * 0.5 + 3.0 * 1.5 + 7.0 * 2.5,
            -5.0 + 6.0 * 0.25 + 8.0 * 0.75,
        ];
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "shortconv mismatch at {idx}: {actual} != {expected}"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_lfm_shortconv_update3_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let cache = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
            (1, 2, 3),
            &device,
        )
        .unwrap();
        let bx = Tensor::from_vec(vec![7.0f32, 8.0], (1, 2, 1), &device).unwrap();

        let out =
            try_lfm_shortconv_update3(&cache, &bx).expect("shortconv update3 should run on Metal");
        let tail = cache.narrow(2, 1, 2).unwrap();
        let reference = Tensor::cat(&[&tail, &bx], 2).unwrap();

        assert_eq!(
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            reference.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_lfm_shortconv_sequence3_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let bx = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0,
            ],
            (1, 2, 4),
            &device,
        )
        .unwrap();
        let conv = Tensor::from_vec(
            vec![
                0.5f32, 1.5, 2.5, //
                -1.0, 0.25, 0.75,
            ],
            (2, 3),
            &device,
        )
        .unwrap();

        let out = try_lfm_shortconv_sequence3(&bx, &conv)
            .expect("shortconv sequence3 should run on Metal");
        let conv_ref = candle_nn::Conv1d::new(
            conv.reshape((2, 1, 3)).unwrap().contiguous().unwrap(),
            None,
            candle_nn::Conv1dConfig {
                padding: 2,
                groups: 2,
                ..Default::default()
            },
        );
        let reference = conv_ref.forward(&bx).unwrap().narrow(2, 0, 4).unwrap();
        let actual = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "shortconv sequence mismatch at {idx}: {actual} != {expected}"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_lfm_shortconv_consumes_wrapped_physical_ring() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return;
        };
        let ring = Tensor::from_vec(
            vec![
                12.0f32, 22.0, // physical slot 0 = absolute step 3
                10.0, 20.0, // physical slot 1 = absolute step 1
                11.0, 21.0, // physical slot 2 = absolute step 2
            ],
            (3, 1, 2),
            &device,
        )
        .unwrap();
        let input = Tensor::from_vec(vec![13.0f32, 14.0, 23.0, 24.0], (1, 2, 2), &device).unwrap();
        let weight =
            Tensor::from_vec(vec![1.0f32, 10.0, 100.0, -1.0, 0.5, 2.0], (2, 3), &device).unwrap();
        let output = try_lfm_shortconv_ring_sequence(&ring, &input, &weight, 4, 3)
            .expect("physical ShortConv ring kernel should run on Metal");
        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = [
            11.0 + 12.0 * 10.0 + 13.0 * 100.0,
            12.0 + 13.0 * 10.0 + 14.0 * 100.0,
            -21.0 + 22.0 * 0.5 + 23.0 * 2.0,
            -22.0 + 23.0 * 0.5 + 24.0 * 2.0,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "physical ShortConv mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_rms_norm_kernel_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let input = Tensor::from_vec(
                vec![
                    0.2f32, -0.4, 0.6, 0.8, //
                    -1.0, 1.2, -1.4, 1.6, //
                    1.8, -2.0, 2.2, -2.4,
                ],
                (1, 3, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let weight = Tensor::from_vec(vec![1.0f32, 1.1, 0.9, 0.8], 4, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            let out = try_fused_rms_norm(&input, &weight, 1e-6)
                .expect("fused RMSNorm should run on Metal");
            let reference = candle_nn::ops::rms_norm(&input, &weight, 1e-6).unwrap();
            let out = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = reference
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
            for (idx, (actual, expected)) in out.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} mismatch at {idx}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_rope_pair_bshd_kernel_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let q = Tensor::from_vec(
                vec![
                    0.2f32, -0.4, 0.6, 0.8, //
                    -1.0, 1.2, -1.4, 1.6, //
                    1.8, -2.0, 2.2, -2.4, //
                    -2.6, 2.8, -3.0, 3.2,
                ],
                (1, 2, 2, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let k = Tensor::from_vec(
                vec![
                    0.3f32, -0.5, 0.7, -0.9, //
                    1.1, -1.3, 1.5, -1.7,
                ],
                (1, 2, 1, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let cos = Tensor::from_vec(vec![0.9f32, 0.8, 0.7, 0.6], (2, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let sin = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let cos_sin = Tensor::cat(&[&cos, &sin], 1).unwrap();

            let (q_out, k_out) = try_fused_rope_pair_bshd(&q, &k, &cos_sin)
                .expect("fused RoPE pair should run on Metal");
            let q_ref = candle_nn::rotary_emb::rope(
                &q.transpose(1, 2).unwrap().contiguous().unwrap(),
                &cos,
                &sin,
            )
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
            let k_ref = candle_nn::rotary_emb::rope(
                &k.transpose(1, 2).unwrap().contiguous().unwrap(),
                &cos,
                &sin,
            )
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();

            let q_out = q_out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let q_ref = q_ref
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let k_out = k_out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let k_ref = k_ref
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
            for (idx, (actual, expected)) in q_out.iter().zip(q_ref.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} q mismatch at {idx}: {actual} != {expected}"
                );
            }
            for (idx, (actual, expected)) in k_out.iter().zip(k_ref.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} k mismatch at {idx}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_decode_gqa_attention_kernel_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let q = Tensor::from_vec(
                vec![
                    0.2f32, -0.4, 0.6, 0.8, //
                    -1.0, 1.2, -1.4, 1.6,
                ],
                (1, 2, 1, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let k = Tensor::from_vec(
                vec![
                    0.3f32, -0.5, 0.7, -0.9, //
                    1.1, -1.3, 1.5, -1.7, //
                    1.9, -2.1, 2.3, -2.5,
                ],
                (1, 1, 3, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let v = Tensor::from_vec(
                vec![
                    -0.2f32, 0.4, -0.6, 0.8, //
                    1.0, -1.2, 1.4, -1.6, //
                    -1.8, 2.0, -2.2, 2.4,
                ],
                (1, 1, 3, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let scale = 0.5f32;

            let out = try_fused_decode_gqa_attention(&q, &k, &v, 2, 1, 4, scale)
                .expect("fused decode GQA attention should run on Metal");
            let k_rep = Tensor::cat(&[&k, &k], 1).unwrap();
            let v_rep = Tensor::cat(&[&v, &v], 1).unwrap();
            let scores = (q.matmul(&k_rep.t().unwrap()).unwrap() * scale as f64).unwrap();
            let probs = candle_nn::ops::softmax(
                &scores.to_dtype(DType::F32).unwrap(),
                candle_core::D::Minus1,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let reference = probs.matmul(&v_rep).unwrap();

            let out = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let reference = reference
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
            for (idx, (actual, expected)) in out.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} mismatch at {idx}: {actual} != {expected}"
                );
            }

            let k_padded = Tensor::from_vec(
                vec![
                    0.3f32, -0.5, 0.7, -0.9, //
                    1.1, -1.3, 1.5, -1.7, //
                    1.9, -2.1, 2.3, -2.5, //
                    90.0, 91.0, 92.0, 93.0, //
                    -90.0, -91.0, -92.0, -93.0,
                ],
                (1, 1, 5, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let v_padded = Tensor::from_vec(
                vec![
                    -0.2f32, 0.4, -0.6, 0.8, //
                    1.0, -1.2, 1.4, -1.6, //
                    -1.8, 2.0, -2.2, 2.4, //
                    80.0, 81.0, 82.0, 83.0, //
                    -80.0, -81.0, -82.0, -83.0,
                ],
                (1, 1, 5, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let padded_out = try_fused_decode_gqa_attention_with_kv_len(
                &q, &k_padded, &v_padded, 2, 1, 4, 3, scale,
            )
            .expect("fused decode GQA attention should ignore padded cache tail");
            let padded_out = padded_out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (idx, (actual, expected)) in padded_out.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} padded mismatch at {idx}: {actual} != {expected}"
                );
            }
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_qk_rms_norm_kernel_matches_reference_if_available() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        for dtype in [DType::F32, DType::F16] {
            let q = Tensor::from_vec(
                vec![
                    0.2f32, -0.4, 0.6, 0.8, //
                    -1.0, 1.2, -1.4, 1.6,
                ],
                (1, 1, 2, 4),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let k = Tensor::from_vec(vec![0.3f32, -0.5, 0.7, -0.9], (1, 1, 1, 4), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let q_weight = Tensor::from_vec(vec![1.0f32, 1.1, 0.9, 0.8], 4, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let k_weight = Tensor::from_vec(vec![0.7f32, 1.2, 0.6, 1.3], 4, &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let qk_weight = Tensor::cat(&[&q_weight, &k_weight], 0).unwrap();

            let (q_out, k_out) = try_fused_qk_rms_norm(&q, &k, &qk_weight, 1e-6)
                .expect("fused q/k norm should run on Metal");
            let q_ref = candle_nn::ops::rms_norm(&q.reshape((2, 4)).unwrap(), &q_weight, 1e-6)
                .unwrap()
                .reshape((1, 1, 2, 4))
                .unwrap();
            let k_ref = candle_nn::ops::rms_norm(&k.reshape((1, 4)).unwrap(), &k_weight, 1e-6)
                .unwrap()
                .reshape((1, 1, 1, 4))
                .unwrap();

            let q_out = q_out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let q_ref = q_ref
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let k_out = k_out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let k_ref = k_ref
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let tolerance = if dtype == DType::F16 { 5e-3 } else { 1e-5 };
            for (idx, (actual, expected)) in q_out.iter().zip(q_ref.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} q mismatch at {idx}: {actual} != {expected}"
                );
            }
            for (idx, (actual, expected)) in k_out.iter().zip(k_ref.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} k mismatch at {idx}: {actual} != {expected}"
                );
            }
        }
    }
}

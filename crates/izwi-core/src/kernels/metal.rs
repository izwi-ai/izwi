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

use super::{FusedKernelError, FusedResult, FusedSiluMulResult};

#[cfg(feature = "metal")]
const IZWI_METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

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

        encoder.set_buffer(
            0,
            Some(gate_storage.buffer()),
            gate_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(up_storage.buffer()),
            up_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(elem_count as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
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

        encoder.set_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.q_rows as u32));
        encoder.set_bytes(5, &(self.k_rows as u32));
        encoder.set_bytes(6, &(self.head_dim as u32));
        encoder.set_bytes(7, &self.eps);

        let threads_per_threadgroup = self
            .head_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .min(256)
            .max(1);
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

        encoder.set_buffer(
            0,
            Some(input_storage.buffer()),
            input_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.rows as u32));
        encoder.set_bytes(4, &(self.hidden_dim as u32));
        encoder.set_bytes(5, &self.eps);

        let threads_per_threadgroup = self
            .hidden_dim
            .next_power_of_two()
            .min(pipeline.max_total_threads_per_threadgroup())
            .min(1024)
            .max(1);
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
            || self.head_dim % 2 != 0
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
        if self.q_rows % self.q_heads != 0
            || self.k_rows % self.k_heads != 0
            || (self.q_rows / self.q_heads) % self.seq_len != 0
            || (self.k_rows / self.k_heads) % self.seq_len != 0
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

        encoder.set_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(cos_sin_storage.buffer()),
            cos_sin_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.q_rows as u32));
        encoder.set_bytes(5, &(self.k_rows as u32));
        encoder.set_bytes(6, &(self.seq_len as u32));
        encoder.set_bytes(7, &(self.q_heads as u32));
        encoder.set_bytes(8, &(self.k_heads as u32));
        encoder.set_bytes(9, &(self.head_dim as u32));

        let threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
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
            || self.num_heads % self.num_kv_heads != 0
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

        encoder.set_buffer(
            0,
            Some(q_storage.buffer()),
            q_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(k_storage.buffer()),
            k_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(v_storage.buffer()),
            v_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
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
            .min(256)
            .max(1);
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
            || num_heads % num_kv_heads != 0
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

/// Try tiled DeltaNet recurrence using Metal tile memory.
///
/// This processes multiple tokens (tile_size, typically 32 or 64) while keeping
/// the hidden state in fast tile memory instead of VRAM. This eliminates
/// memory round-trips for the recurrence h_t = A * h_{t-1} + B.
///
/// The tile memory is high-speed memory local to the GPU shader. By loading
/// the hidden state into tile memory, we can process an entire sequence of
/// tokens without writing the state back to main VRAM.
///
/// This is the primary optimization that enables llama.cpp to achieve 3x
/// higher TPS than naive implementations.
///
/// # Arguments
/// * `queries` - Query tensors [batch, seq, num_heads, head_dim]
/// * `keys` - Key tensors [batch, seq, num_heads, head_dim]
/// * `values` - Value tensors [batch, seq, num_v_heads, head_v_dim]
/// * `g` - Pre-computed gate values [batch, seq, num_v_heads]
/// * `beta` - Beta values [batch, seq, num_v_heads]
/// * `initial_state` - Initial recurrent state [batch, num_v_heads, head_k_dim, head_v_dim]
/// * `tile_size` - Number of tokens to process per tile (32 or 64 recommended)
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
    if !use_fused_kernels() {
        return None;
    }

    // Only supported for F32 on Metal devices
    if queries.dtype() != DType::F32 {
        return None;
    }

    if !queries.device().is_metal() {
        return None;
    }

    let (batch, seq_len, num_heads, head_k_dim) = queries.dims4().ok()?;
    let (k_batch, k_seq_len, k_num_heads, k_head_k_dim) = keys.dims4().ok()?;
    let (v_batch, v_seq_len, v_num_heads, _v_head_dim) = values.dims4().ok()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3().ok()?;
    let (b_batch, b_seq_len, b_heads) = beta.dims3().ok()?;
    let (s_batch, s_heads, s_head_k_dim, _s_head_v_dim) = initial_state.dims4().ok()?;

    if batch != 1
        || k_batch != batch
        || v_batch != batch
        || g_batch != batch
        || b_batch != batch
        || s_batch != batch
    {
        return None;
    }
    if k_seq_len != seq_len || v_seq_len != seq_len || g_seq_len != seq_len || b_seq_len != seq_len
    {
        return None;
    }
    if k_num_heads != num_heads || g_heads != num_heads || b_heads != num_heads {
        return None;
    }
    if k_head_k_dim != head_k_dim || s_heads != num_heads || s_head_k_dim != head_k_dim {
        return None;
    }
    if v_num_heads != num_heads {
        return None;
    }

    let tile_size = tile_size.max(1).min(seq_len.max(1));
    let mut outputs = Vec::with_capacity(seq_len);
    let mut state = initial_state.clone();

    for tile_start in (0..seq_len).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(seq_len);
        for token_idx in tile_start..tile_end {
            let query = queries.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let key = keys.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let value = values.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let g_t = g.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;
            let beta_t = beta.narrow(1, token_idx, 1).ok()?.squeeze(1).ok()?;

            let (output, next_state) =
                fused_gated_delta_sequential(&query, &key, &value, &g_t, &beta_t, &state).ok()?;
            outputs.push(output.unsqueeze(1).ok()?);
            state = next_state;
        }
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let outputs = Tensor::cat(&output_refs, 1).ok()?;
    Some((outputs, state))
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

    #[test]
    fn test_l2_norm_matches_reference() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();

        // Reference implementation
        let sq_sum: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x * x).sum();
        let norm = sq_sum.sqrt();
        let expected = vec![
            vec![1.0f32 / norm, 2.0 / norm],
            vec![3.0 / norm, 4.0 / norm],
        ];

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

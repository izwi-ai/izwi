//! Simdgroup paged decode attention for Metal.
//!
//! One threadgroup serves one (row, KV head, context partition) and every
//! query head of that KV head's GQA group, so each K/V page is read once per
//! group. Four simdgroups take strided keys; within a simdgroup the 32 lanes
//! split the head dimension and reduce the dot product with `simd_sum`, so no
//! threadgroup barrier is needed per key (the scalar kernel needed about nine).
//! Online-softmax state and the value accumulators stay in registers until a
//! final per-head merge across simdgroups. Partitions along the context keep
//! the GPU occupied when a batch has few rows and KV heads; their F32 partial
//! states use the same layout as the split kernel and merge through its reduce
//! kernel.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::{
    bail, CpuStorage, DType, InplaceOp3, Layout, MetalStorage, Result as CandleResult, Tensor,
};
use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

/// Simdgroups per threadgroup; the kernel is written for 32-lane simdgroups.
pub(crate) const GQA_DECODE_SIMDGROUPS: usize = 4;
pub(crate) const GQA_DECODE_THREADS: usize = GQA_DECODE_SIMDGROUPS * 32;
/// Threadgroups the dispatch aims for before it stops adding partitions.
const GQA_DECODE_TARGET_THREADGROUPS: usize = 128;
/// Smallest context slice worth a partition of its own.
const GQA_DECODE_MIN_PARTITION_TOKENS: usize = 64;

/// The compiled specialization serving one decode geometry: head-dimension
/// elements per lane and the query heads one KV head serves (both padded up).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GqaDecodeVariant {
    pub(crate) dims_per_lane: usize,
    pub(crate) max_group: usize,
}

impl GqaDecodeVariant {
    /// The variant for this geometry, or `None` when the scalar kernels must
    /// serve it (unequal key/value dims, head dim above 256, or more than
    /// eight query heads per KV head).
    pub(crate) fn select(
        key_head_dim: usize,
        value_head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Option<Self> {
        if key_head_dim != value_head_dim
            || key_head_dim == 0
            || num_kv_heads == 0
            || !num_heads.is_multiple_of(num_kv_heads)
        {
            return None;
        }
        let dims_per_lane = match key_head_dim.div_ceil(32) {
            1 => 1,
            2 => 2,
            3 | 4 => 4,
            5..=8 => 8,
            _ => return None,
        };
        let max_group = match num_heads / num_kv_heads {
            1 => 1,
            2 => 2,
            3 | 4 => 4,
            5..=8 => 8,
            _ => return None,
        };
        Some(Self {
            dims_per_lane,
            max_group,
        })
    }

    fn function_name(self, dtype: DType) -> CandleResult<String> {
        let dtype = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            _ => bail!("izwi-paged-decode-gqa only supports F32 and F16"),
        };
        Ok(format!(
            "izwi_paged_decode_gqa_{dtype}_d{}_g{}",
            self.dims_per_lane, self.max_group
        ))
    }
}

/// The most partitions one row's context is split into: enough for a lone
/// row's KV heads to approach [`GQA_DECODE_TARGET_THREADGROUPS`].
pub(crate) fn gqa_max_row_partitions(num_kv_heads: usize) -> usize {
    GQA_DECODE_TARGET_THREADGROUPS
        .div_ceil(num_kv_heads.max(1))
        .max(1)
}

/// Tokens in each of one row's context partitions, from that row's own
/// context length alone, never slicing below
/// [`GQA_DECODE_MIN_PARTITION_TOKENS`]. A row therefore splits, and sums its
/// keys, in the same order alone or in any batch: splitting by the batch's
/// size or longest row made a row's attention, and so its greedy tokens,
/// depend on which other requests shared its step. The kernel computes the
/// same value (`izwi_gqa_row_partition_tokens`).
pub(crate) fn gqa_row_partition_tokens(context_len: usize, max_row_partitions: usize) -> usize {
    let by_length = context_len.div_ceil(GQA_DECODE_MIN_PARTITION_TOKENS).max(1);
    let partitions = max_row_partitions.max(1).min(by_length);
    context_len.div_ceil(partitions).max(1)
}

/// The launch's partition count (the most any row uses) and the per-row cap
/// the kernel and reduce derive each row's split from.
pub(crate) fn gqa_decode_partitions(
    num_kv_heads: usize,
    context_lens: impl IntoIterator<Item = usize>,
) -> (usize, usize) {
    let max_row_partitions = gqa_max_row_partitions(num_kv_heads);
    let partitions = context_lens
        .into_iter()
        .map(|context_len| {
            context_len.div_ceil(gqa_row_partition_tokens(context_len, max_row_partitions))
        })
        .max()
        .unwrap_or(1)
        .max(1);
    (partitions, max_row_partitions)
}

/// Pipeline for one variant and dtype, compiled from this module's own
/// library so its specializations do not slow every other pipeline compile.
pub(crate) fn gqa_decode_pipeline(
    device: &MetalDevice,
    dtype: DType,
    variant: GqaDecodeVariant,
) -> CandleResult<ComputePipeline> {
    type Key = (u64, DType, GqaDecodeVariant);
    static PIPELINES: OnceLock<Mutex<HashMap<Key, ComputePipeline>>> = OnceLock::new();
    let key = (device.registry_id(), dtype, variant);
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
        .new_library_with_source(GQA_DECODE_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(&variant.function_name(dtype)?, None)
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

/// The reduce that merges each row's own partitions of a split GQA decode.
pub(crate) fn gqa_decode_reduce_pipeline(
    device: &MetalDevice,
    dtype: DType,
) -> CandleResult<ComputePipeline> {
    type Key = (u64, DType);
    static PIPELINES: OnceLock<Mutex<HashMap<Key, ComputePipeline>>> = OnceLock::new();
    let key = (device.registry_id(), dtype);
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }
    let name = match dtype {
        DType::F32 => "izwi_paged_decode_gqa_reduce_f32",
        DType::F16 => "izwi_paged_decode_gqa_reduce_f16",
        _ => bail!("izwi-paged-decode-gqa only supports F32 and F16"),
    };
    let library = device
        .new_library_with_source(GQA_DECODE_SOURCE, None)
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

/// Whether the simdgroup decode path is enabled (`IZWI_METAL_GQA_DECODE=0`
/// restores the scalar kernels for every geometry).
pub(crate) fn gqa_decode_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_METAL_GQA_DECODE", true));
    ENABLED.get()
}

/// Copy `source` rows `[rows, row_elements]` into the `destination` rows
/// named by `slots`, in place and in one dispatch. Replaces scatter through a
/// broadcast index tensor, which materialized a u32 index as large as the
/// written K/V (twice its bytes in F16) and cost an extra copy dispatch.
/// Callers must have bounds-checked every slot against the destination rows.
pub(crate) fn write_kv_rows(
    destination: &Tensor,
    source: &Tensor,
    slots: &Tensor,
) -> CandleResult<()> {
    if destination.rank() == 0
        || source.rank() != destination.rank()
        || source.dims()[1..] != destination.dims()[1..]
        || source.dim(0)? != slots.elem_count()
    {
        bail!(
            "izwi-kv-write-rows-metal shape mismatch: destination {:?}, source {:?}, {} slots",
            destination.dims(),
            source.dims(),
            slots.elem_count()
        )
    }
    destination.inplace_op3(&source.contiguous()?, &slots.contiguous()?, &KvWriteRowsOp)
}

struct KvWriteRowsOp;

impl InplaceOp3 for KvWriteRowsOp {
    fn name(&self) -> &'static str {
        "izwi-kv-write-rows-metal"
    }

    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CandleResult<()> {
        bail!("izwi-kv-write-rows-metal requires Metal tensors")
    }

    fn metal_fwd(
        &self,
        destination: &mut MetalStorage,
        destination_layout: &Layout,
        source: &MetalStorage,
        source_layout: &Layout,
        slots: &MetalStorage,
        slots_layout: &Layout,
    ) -> CandleResult<()> {
        let dtype = destination.dtype();
        if source.dtype() != dtype || slots.dtype() != DType::U32 {
            bail!("izwi-kv-write-rows-metal requires matching K/V dtypes and u32 slots")
        }
        if !destination_layout.is_contiguous()
            || !source_layout.is_contiguous()
            || !slots_layout.is_contiguous()
        {
            bail!("izwi-kv-write-rows-metal requires contiguous tensors")
        }
        let rows = slots_layout.shape().elem_count();
        let source_elements = source_layout.shape().elem_count();
        if rows == 0 || source_elements == 0 || !source_elements.is_multiple_of(rows) {
            bail!("izwi-kv-write-rows-metal source does not split into slot rows")
        }
        let row_elements = source_elements / rows;
        if !destination_layout
            .shape()
            .elem_count()
            .is_multiple_of(row_elements)
            || rows > u32::MAX as usize
            || row_elements > u32::MAX as usize
        {
            bail!("izwi-kv-write-rows-metal destination rows do not match the source")
        }
        let function = match dtype.size_in_bytes() {
            2 => "izwi_kv_write_rows_b16",
            4 => "izwi_kv_write_rows_b32",
            _ => bail!("izwi-kv-write-rows-metal supports 16- and 32-bit elements"),
        };
        let device = destination.device().clone();
        let pipeline = kv_write_pipeline(device.metal_device(), function)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-kv-write-rows");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_output_buffer(
            0,
            Some(destination.buffer()),
            destination_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(source.buffer()),
            source_layout.start_offset() * dtype.size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(slots.buffer()),
            slots_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(3, &(rows as u32));
        encoder.set_bytes(4, &(row_elements as u32));
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: source_elements,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: pipeline.max_total_threads_per_threadgroup().clamp(1, 256),
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

fn kv_write_pipeline(device: &MetalDevice, function: &str) -> CandleResult<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> = OnceLock::new();
    let key = (device.registry_id(), function.to_string());
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
        .new_library_with_source(KV_WRITE_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function, None)
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

const KV_WRITE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

template <typename E>
kernel void izwi_kv_write_rows(
    device E* destination [[buffer(0)]],
    device const E* source [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& row_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row = gid / row_elements;
    if (row >= rows) {
        return;
    }
    const uint element = gid - row * row_elements;
    destination[slots[row] * row_elements + element] = source[gid];
}

template [[host_name("izwi_kv_write_rows_b16")]] kernel void izwi_kv_write_rows<ushort>(
    device ushort* destination [[buffer(0)]],
    device const ushort* source [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& row_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]);

template [[host_name("izwi_kv_write_rows_b32")]] kernel void izwi_kv_write_rows<uint>(
    device uint* destination [[buffer(0)]],
    device const uint* source [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& row_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]);
"#;

const GQA_DECODE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint IZWI_GQA_SIMDGROUPS = 4;
constant constexpr uint IZWI_GQA_MIN_PARTITION_TOKENS = 64;

// Tokens in each of a row's context partitions, from that row's own context
// alone, so a row splits the same way in any batch (host twin:
// `gqa_row_partition_tokens`).
inline uint izwi_gqa_row_partition_tokens(uint context_len, uint max_row_partitions) {
    const uint by_length = max(
        1u,
        (context_len + IZWI_GQA_MIN_PARTITION_TOKENS - 1) / IZWI_GQA_MIN_PARTITION_TOKENS);
    const uint partitions = min(max(max_row_partitions, 1u), by_length);
    return max(1u, (context_len + partitions - 1) / partitions);
}

template <typename T, uint DPL, uint GMAX>
kernel void izwi_paged_decode_gqa(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const T* v [[buffer(2)]],
    device T* output [[buffer(3)]],
    device float* partial_values [[buffer(4)]],
    device float* partial_maxes [[buffer(5)]],
    device float* partial_sums [[buffer(6)]],
    device const uint* metadata [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& num_kv_heads [[buffer(10)]],
    constant uint& page_tokens [[buffer(11)]],
    constant uint& max_blocks [[buffer(12)]],
    constant uint& head_dim [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    constant float& softcap [[buffer(15)]],
    constant uint& max_row_partitions [[buffer(16)]],
    constant uint& partitions [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float merge_values[IZWI_GQA_SIMDGROUPS * DPL * 32];
    threadgroup float merge_state[IZWI_GQA_SIMDGROUPS * GMAX * 2];

    const uint kv_head = group.x;
    const uint row = group.y;
    const uint partition = group.z;
    // Uniform per threadgroup, so no thread can miss a later barrier.
    if (row >= batch_size || kv_head >= num_kv_heads || partition >= partitions) {
        return;
    }

    const uint group_size = num_heads / num_kv_heads;
    const uint context_len = metadata[row];
    const uint partition_tokens = izwi_gqa_row_partition_tokens(context_len, max_row_partitions);
    // A row with fewer partitions than the launch has none here (uniform per
    // threadgroup, before any barrier).
    if (partition * partition_tokens >= context_len) {
        return;
    }
    const uint first_page_offset = metadata[batch_size + row];
    const uint table_base = 2 * batch_size + row * max_blocks;
    const uint start = partition * partition_tokens;
    const uint end = min(start + partition_tokens, context_len);

    float query[GMAX][DPL];
    float acc[GMAX][DPL];
    float running_max[GMAX];
    float running_sum[GMAX];
    for (uint g = 0; g < GMAX; g++) {
        running_max[g] = -INFINITY;
        running_sum[g] = 0.0f;
        const uint head = kv_head * group_size + g;
        for (uint i = 0; i < DPL; i++) {
            const uint d = lane + 32 * i;
            acc[g][i] = 0.0f;
            // The softmax scale is folded into the query once.
            query[g][i] = (g < group_size && d < head_dim)
                ? float(q[(row * num_heads + head) * head_dim + d]) * scale
                : 0.0f;
        }
    }

    for (uint pos = start + simd; pos < end; pos += IZWI_GQA_SIMDGROUPS) {
        const uint physical_pos = first_page_offset + pos;
        const uint logical_page = physical_pos / page_tokens;
        const uint page_offset = physical_pos - logical_page * page_tokens;
        const uint physical_page = metadata[table_base + logical_page];
        const uint base =
            ((physical_page * page_tokens + page_offset) * num_kv_heads + kv_head) * head_dim;
        float key[DPL];
        float value[DPL];
        for (uint i = 0; i < DPL; i++) {
            const uint d = lane + 32 * i;
            key[i] = d < head_dim ? float(k[base + d]) : 0.0f;
            value[i] = d < head_dim ? float(v[base + d]) : 0.0f;
        }
        // A constant trip count unrolls, so `query`, `acc` and the running
        // state stay in registers. An early `break` on the runtime group size
        // kept this loop rolled, with those arrays indexed in thread memory on
        // every key: decode took 1.8x as long (1.0 ms against 0.55 ms for
        // 32 query / 8 KV heads of 64 dims over 16K tokens).
        for (uint g = 0; g < GMAX; g++) {
            if (g < group_size) {
                float dot = 0.0f;
                for (uint i = 0; i < DPL; i++) {
                    dot += query[g][i] * key[i];
                }
                float score = simd_sum(dot);
                if (softcap > 0.0f) {
                    score = softcap * tanh(score / softcap);
                }
                const float next_max = max(running_max[g], score);
                const float alpha =
                    running_sum[g] == 0.0f ? 0.0f : exp(running_max[g] - next_max);
                const float beta = exp(score - next_max);
                running_sum[g] = running_sum[g] * alpha + beta;
                running_max[g] = next_max;
                for (uint i = 0; i < DPL; i++) {
                    acc[g][i] = acc[g][i] * alpha + value[i] * beta;
                }
            }
        }
    }

    if (lane == 0) {
        for (uint g = 0; g < GMAX; g++) {
            merge_state[(simd * GMAX + g) * 2] = running_max[g];
            merge_state[(simd * GMAX + g) * 2 + 1] = running_sum[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = 0; g < GMAX; g++) {
        if (g >= group_size) {
            break;
        }
        float global_max = -INFINITY;
        for (uint s = 0; s < IZWI_GQA_SIMDGROUPS; s++) {
            global_max = max(global_max, merge_state[(s * GMAX + g) * 2]);
        }
        float global_sum = 0.0f;
        for (uint s = 0; s < IZWI_GQA_SIMDGROUPS; s++) {
            const float part_sum = merge_state[(s * GMAX + g) * 2 + 1];
            if (part_sum > 0.0f) {
                global_sum += part_sum * exp(merge_state[(s * GMAX + g) * 2] - global_max);
            }
        }
        const float weight = running_sum[g] > 0.0f ? exp(running_max[g] - global_max) : 0.0f;
        for (uint i = 0; i < DPL; i++) {
            merge_values[(simd * DPL + i) * 32 + lane] = acc[g][i] * weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint head = kv_head * group_size + g;
        const uint partial_index = (row * num_heads + head) * partitions + partition;
        for (uint d = tid; d < head_dim; d += IZWI_GQA_SIMDGROUPS * 32) {
            const uint i = d / 32;
            const uint d_lane = d - i * 32;
            float merged = 0.0f;
            for (uint s = 0; s < IZWI_GQA_SIMDGROUPS; s++) {
                merged += merge_values[(s * DPL + i) * 32 + d_lane];
            }
            if (partitions == 1) {
                output[(row * num_heads + head) * head_dim + d] =
                    T(global_sum > 0.0f ? merged / global_sum : 0.0f);
            } else {
                partial_values[partial_index * head_dim + d] = merged;
            }
        }
        if (tid == 0 && partitions > 1) {
            partial_maxes[partial_index] = global_max;
            partial_sums[partial_index] = global_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define IZWI_GQA_DECODE(T, TNAME, DPL, GMAX)                                          \
template [[host_name("izwi_paged_decode_gqa_" #TNAME "_d" #DPL "_g" #GMAX)]]          \
kernel void izwi_paged_decode_gqa<T, DPL, GMAX>(                                       \
    device const T* q [[buffer(0)]],                                                   \
    device const T* k [[buffer(1)]],                                                   \
    device const T* v [[buffer(2)]],                                                   \
    device T* output [[buffer(3)]],                                                    \
    device float* partial_values [[buffer(4)]],                                        \
    device float* partial_maxes [[buffer(5)]],                                         \
    device float* partial_sums [[buffer(6)]],                                          \
    device const uint* metadata [[buffer(7)]],                                         \
    constant uint& batch_size [[buffer(8)]],                                           \
    constant uint& num_heads [[buffer(9)]],                                            \
    constant uint& num_kv_heads [[buffer(10)]],                                        \
    constant uint& page_tokens [[buffer(11)]],                                         \
    constant uint& max_blocks [[buffer(12)]],                                          \
    constant uint& head_dim [[buffer(13)]],                                            \
    constant float& scale [[buffer(14)]],                                              \
    constant float& softcap [[buffer(15)]],                                            \
    constant uint& max_row_partitions [[buffer(16)]],                                  \
    constant uint& partitions [[buffer(17)]],                                          \
    uint3 group [[threadgroup_position_in_grid]],                                      \
    uint lane [[thread_index_in_simdgroup]],                                           \
    uint simd [[simdgroup_index_in_threadgroup]],                                      \
    uint tid [[thread_index_in_threadgroup]]);

#define IZWI_GQA_DECODE_GROUPS(T, TNAME, DPL) \
    IZWI_GQA_DECODE(T, TNAME, DPL, 1)         \
    IZWI_GQA_DECODE(T, TNAME, DPL, 2)         \
    IZWI_GQA_DECODE(T, TNAME, DPL, 4)         \
    IZWI_GQA_DECODE(T, TNAME, DPL, 8)

#define IZWI_GQA_DECODE_DIMS(T, TNAME)  \
    IZWI_GQA_DECODE_GROUPS(T, TNAME, 1) \
    IZWI_GQA_DECODE_GROUPS(T, TNAME, 2) \
    IZWI_GQA_DECODE_GROUPS(T, TNAME, 4) \
    IZWI_GQA_DECODE_GROUPS(T, TNAME, 8)

IZWI_GQA_DECODE_DIMS(float, f32)
IZWI_GQA_DECODE_DIMS(half, f16)

// Merges each row's own partitions; the shared paged-attention reduce takes
// one partition size for the whole batch.
template <typename T>
kernel void izwi_paged_decode_gqa_reduce(
    device const float* partial_values [[buffer(0)]],
    device const float* partial_maxes [[buffer(1)]],
    device const float* partial_sums [[buffer(2)]],
    device T* output [[buffer(3)]],
    device const uint* context_lens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& value_head_dim [[buffer(6)]],
    constant uint& max_row_partitions [[buffer(7)]],
    constant uint& max_partitions [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float merge_state[2];
    const uint head = group.x;
    const uint row = group.y;
    const uint partial_base = (row * num_heads + head) * max_partitions;
    const uint context_len = context_lens[row];
    const uint partition_tokens = izwi_gqa_row_partition_tokens(context_len, max_row_partitions);
    const uint partition_count = (context_len + partition_tokens - 1) / partition_tokens;

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
        output[out_base + dim] = T(acc / merge_state[1]);
    }
}

#define IZWI_GQA_REDUCE(T, TNAME)                                                      \
template [[host_name("izwi_paged_decode_gqa_reduce_" #TNAME)]]                        \
kernel void izwi_paged_decode_gqa_reduce<T>(                                           \
    device const float* partial_values [[buffer(0)]],                                  \
    device const float* partial_maxes [[buffer(1)]],                                   \
    device const float* partial_sums [[buffer(2)]],                                    \
    device T* output [[buffer(3)]],                                                    \
    device const uint* context_lens [[buffer(4)]],                                     \
    constant uint& num_heads [[buffer(5)]],                                            \
    constant uint& value_head_dim [[buffer(6)]],                                       \
    constant uint& max_row_partitions [[buffer(7)]],                                   \
    constant uint& max_partitions [[buffer(8)]],                                       \
    uint tid [[thread_index_in_threadgroup]],                                          \
    uint2 group [[threadgroup_position_in_grid]],                                      \
    uint2 threads_per_group [[threads_per_threadgroup]]);

IZWI_GQA_REDUCE(float, f32)
IZWI_GQA_REDUCE(half, f16)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_cover_the_served_head_dims_and_groups() {
        let variant = |kd, vd, heads, kv| GqaDecodeVariant::select(kd, vd, heads, kv);
        assert_eq!(
            variant(128, 128, 16, 8),
            Some(GqaDecodeVariant {
                dims_per_lane: 4,
                max_group: 2
            })
        );
        assert_eq!(
            variant(256, 256, 24, 4).map(|v| (v.dims_per_lane, v.max_group)),
            Some((8, 8))
        );
        assert_eq!(
            variant(64, 64, 20, 20).map(|v| (v.dims_per_lane, v.max_group)),
            Some((2, 1))
        );
        assert_eq!(
            variant(32, 32, 28, 4).map(|v| (v.dims_per_lane, v.max_group)),
            Some((1, 8))
        );
        assert_eq!(variant(2, 3, 2, 1), None, "unequal key/value dims");
        assert_eq!(variant(512, 512, 8, 8), None, "head dim above 256");
        assert_eq!(variant(128, 128, 32, 2), None, "group of 16");
        assert_eq!(variant(128, 128, 6, 4), None, "uneven grouping");
    }

    #[test]
    fn a_lone_row_fills_the_gpu_without_slicing_short_contexts() {
        // One row, two KV heads, long context: split to approach 128 groups.
        assert_eq!(gqa_decode_partitions(2, [4096]), (64, 64));
        assert_eq!(gqa_row_partition_tokens(4096, 64), 64);
        // Short contexts get at most one partition per 64 tokens.
        assert_eq!(gqa_row_partition_tokens(100, 64), 50);
        assert_eq!(gqa_decode_partitions(2, [100]).0, 2);
        assert_eq!(gqa_decode_partitions(2, [10]).0, 1);
        assert_eq!(gqa_row_partition_tokens(10, 64), 10);
    }

    /// A row splits by its own context alone: the same partitions whatever
    /// else shares the batch, so its attention sums run in the same order.
    #[test]
    fn a_rows_partitions_do_not_depend_on_the_batch() {
        let heads = 8;
        let cap = gqa_max_row_partitions(heads);
        for context in [1, 63, 64, 65, 100, 135, 700, 4096, 40_000] {
            let alone = gqa_decode_partitions(heads, [context]);
            let tokens = gqa_row_partition_tokens(context, cap);
            for others in [vec![10], vec![130, 5000], vec![70; 11]] {
                let batch = gqa_decode_partitions(heads, others.iter().copied().chain([context]));
                assert_eq!(batch.1, alone.1, "the cap depends on the model only");
                assert!(
                    context.div_ceil(tokens) <= batch.0,
                    "the launch covers the row"
                );
            }
            assert_eq!(context.div_ceil(tokens), alone.0);
        }
    }
}

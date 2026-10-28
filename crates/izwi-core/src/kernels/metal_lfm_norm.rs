//! Metal fusions for LFM3's decode step, which is bound by the host's
//! per-dispatch encoding cost:
//! * the sandwich norm and its residual add, `residual + rms_norm(x) * w`,
//!   in one dispatch instead of two (twice per layer);
//! * the attention output gate, `heads * sigmoid(gate)` per head, in one
//!   dispatch instead of two (once per attention layer).
//!
//! In F32 both compute what the unfused ops compute: the norm sums squares
//! in the same tree, with the same thread count, as Izwi's fused RMSNorm.
//! The residual add may round once instead of twice, as the GPU contracts
//! it into a multiply-add.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::{
    bail, CpuStorage, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor,
};
use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

/// The thread count Izwi's RMSNorm reduces a row with: the power of two at
/// or above `hidden`, within the pipeline's limit and 1,024.
fn reduction_threads(hidden: usize, pipeline_limit: usize) -> usize {
    let bound = hidden
        .max(1)
        .next_power_of_two()
        .min(pipeline_limit.max(1))
        .min(1_024);
    1 << (usize::BITS - 1 - bound.leading_zeros())
}

fn kernel_suffix(dtype: DType) -> candle_core::Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        other => bail!("izwi-lfm-norm serves F32 and F16, not {other:?}"),
    })
}

/// `residual + rms_norm(input) * weight` for Metal tensors `[.., hidden]`
/// of one dtype, or `None` where the fusion does not apply.
pub(crate) fn try_rms_norm_add(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    let hidden = *input.dims().last()?;
    let dtype = input.dtype();
    if !input.device().is_metal()
        || !matches!(dtype, DType::F32 | DType::F16)
        || residual.dtype() != dtype
        || weight.dtype() != dtype
        || residual.dims() != input.dims()
        || weight.dims() != [hidden]
        || !input.is_contiguous()
        || !residual.is_contiguous()
        || !weight.is_contiguous()
        || hidden == 0
    {
        return None;
    }
    let rows = input.elem_count() / hidden;
    input
        .apply_op3_no_bwd(
            residual,
            weight,
            &RmsNormAdd {
                rows,
                hidden,
                eps: eps as f32,
            },
        )
        .ok()
}

/// `heads * sigmoid(gate)`, `heads` `[.., n_head * head_dim]` and `gate`
/// `[.., n_head]` Metal tensors of one dtype, or `None` where the fusion
/// does not apply.
pub(crate) fn try_sigmoid_gate(heads: &Tensor, gate: &Tensor, head_dim: usize) -> Option<Tensor> {
    let dtype = heads.dtype();
    if !heads.device().is_metal()
        || !matches!(dtype, DType::F32 | DType::F16)
        || gate.dtype() != dtype
        || head_dim == 0
        || heads.elem_count() != gate.elem_count() * head_dim
        || !heads.is_contiguous()
        || !gate.is_contiguous()
    {
        return None;
    }
    heads
        .apply_op2_no_bwd(
            gate,
            &SigmoidGate {
                elements: heads.elem_count(),
                head_dim,
            },
        )
        .ok()
}

struct RmsNormAdd {
    rows: usize,
    hidden: usize,
    eps: f32,
}

impl CustomOp3 for RmsNormAdd {
    fn name(&self) -> &'static str {
        "izwi-lfm-rms-norm-add"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        bail!("izwi-lfm-rms-norm-add runs on Metal only")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_layout: &Layout,
        residual: &MetalStorage,
        residual_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let dtype = input.dtype();
        let elements = self.rows * self.hidden;
        if self.rows > u32::MAX as usize || self.hidden > u32::MAX as usize {
            bail!("izwi-lfm-rms-norm-add tensor is too large")
        }
        let device = input.device().clone();
        let output = device.new_buffer(elements, dtype, "izwi-lfm-rms-norm-add")?;
        let pipeline = pipeline(
            device.metal_device(),
            &format!("izwi_lfm_rms_norm_add_{}", kernel_suffix(dtype)?),
        )?;
        let threads = reduction_threads(self.hidden, pipeline.max_total_threads_per_threadgroup());
        let size = dtype.size_in_bytes();
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-rms-norm-add");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(input.buffer()), input_layout.start_offset() * size);
        encoder.set_input_buffer(
            1,
            Some(residual.buffer()),
            residual_layout.start_offset() * size,
        );
        encoder.set_input_buffer(2, Some(weight.buffer()), weight_layout.start_offset() * size);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.hidden as u32));
        encoder.set_bytes(5, &self.eps);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.rows,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(output, device, elements, dtype),
            input_layout.shape().clone(),
        ))
    }
}

struct SigmoidGate {
    elements: usize,
    head_dim: usize,
}

impl CustomOp2 for SigmoidGate {
    fn name(&self) -> &'static str {
        "izwi-lfm-sigmoid-gate"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        bail!("izwi-lfm-sigmoid-gate runs on Metal only")
    }

    fn metal_fwd(
        &self,
        heads: &MetalStorage,
        heads_layout: &Layout,
        gate: &MetalStorage,
        gate_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let dtype = heads.dtype();
        if self.elements > u32::MAX as usize {
            bail!("izwi-lfm-sigmoid-gate tensor is too large")
        }
        let device = heads.device().clone();
        let output = device.new_buffer(self.elements, dtype, "izwi-lfm-sigmoid-gate")?;
        let pipeline = pipeline(
            device.metal_device(),
            &format!("izwi_lfm_sigmoid_gate_{}", kernel_suffix(dtype)?),
        )?;
        let size = dtype.size_in_bytes();
        let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-lfm-sigmoid-gate");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(heads.buffer()), heads_layout.start_offset() * size);
        encoder.set_input_buffer(1, Some(gate.buffer()), gate_layout.start_offset() * size);
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.elements as u32));
        encoder.set_bytes(4, &(self.head_dim as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.elements.div_ceil(threads),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(output, device, self.elements, dtype),
            heads_layout.shape().clone(),
        ))
    }
}

fn pipeline(device: &MetalDevice, function: &str) -> candle_core::Result<ComputePipeline> {
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
        .new_library_with_source(SOURCE, None)
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

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup per row: the RMSNorm tree reduction of Izwi's fused norm,
// then the residual add, in F32.
#define IZWI_LFM_RMS_NORM_ADD(NAME, T)                                          \
kernel void NAME(                                                                \
    device const T* input [[buffer(0)]],                                         \
    device const T* residual [[buffer(1)]],                                      \
    device const T* weight [[buffer(2)]],                                        \
    device T* output [[buffer(3)]],                                              \
    constant uint& hidden [[buffer(4)]],                                         \
    constant float& eps [[buffer(5)]],                                           \
    uint tid [[thread_index_in_threadgroup]],                                    \
    uint row [[threadgroup_position_in_grid]],                                   \
    uint threads [[threads_per_threadgroup]]                                     \
) {                                                                              \
    threadgroup float sums[1024];                                                \
    const uint offset = row * hidden;                                            \
    float sum = 0.0f;                                                            \
    for (uint i = tid; i < hidden; i += threads) {                               \
        const float value = float(input[offset + i]);                            \
        sum += value * value;                                                    \
    }                                                                            \
    sums[tid] = sum;                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    for (uint stride = threads >> 1; stride > 0; stride >>= 1) {                 \
        if (tid < stride) {                                                      \
            sums[tid] += sums[tid + stride];                                     \
        }                                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    }                                                                            \
    const float scale = rsqrt((sums[0] / float(hidden)) + eps);                  \
    for (uint i = tid; i < hidden; i += threads) {                               \
        const float normed = float(input[offset + i]) * scale * float(weight[i]); \
        output[offset + i] = T(float(residual[offset + i]) + normed);            \
    }                                                                            \
}

IZWI_LFM_RMS_NORM_ADD(izwi_lfm_rms_norm_add_f32, float)
IZWI_LFM_RMS_NORM_ADD(izwi_lfm_rms_norm_add_f16, half)

#define IZWI_LFM_SIGMOID_GATE(NAME, T)                                          \
kernel void NAME(                                                                \
    device const T* heads [[buffer(0)]],                                         \
    device const T* gate [[buffer(1)]],                                          \
    device T* output [[buffer(2)]],                                              \
    constant uint& elements [[buffer(3)]],                                       \
    constant uint& head_dim [[buffer(4)]],                                       \
    uint index [[thread_position_in_grid]]                                       \
) {                                                                              \
    if (index >= elements) {                                                     \
        return;                                                                  \
    }                                                                            \
    const float g = float(gate[index / head_dim]);                               \
    output[index] = T(float(heads[index]) * (1.0f / (1.0f + exp(-g))));          \
}

IZWI_LFM_SIGMOID_GATE(izwi_lfm_sigmoid_gate_f32, float)
IZWI_LFM_SIGMOID_GATE(izwi_lfm_sigmoid_gate_f16, half)
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn worst(lhs: &Tensor, rhs: &Tensor) -> f32 {
        let lhs = lhs
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = rhs
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        lhs.iter()
            .zip(&rhs)
            .map(|(a, b)| {
                let difference = (a - b).abs();
                if difference.is_finite() {
                    difference
                } else {
                    f32::INFINITY
                }
            })
            .fold(0f32, f32::max)
    }

    /// The fused norm-add equals the fused RMSNorm followed by an add to an
    /// ulp in F32 (LFM3's activation dtype), to F16 rounding in F16.
    #[test]
    fn rms_norm_add_matches_the_unfused_ops() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("rms_norm_add_matches_the_unfused_ops")
        else {
            return;
        };
        for (rows, hidden) in [(1, 2_048), (3, 512), (2, 768), (1, 7)] {
            let tensor = |seed: u64, shape: &[usize]| {
                Tensor::from_vec(
                    xorshift_values(shape.iter().product(), seed, 1.0),
                    shape,
                    &device,
                )
                .unwrap()
            };
            let input = tensor(1, &[rows, hidden]);
            let residual = tensor(2, &[rows, hidden]);
            let weight = tensor(3, &[hidden]);
            for (dtype, tolerance) in [(DType::F32, 2e-6), (DType::F16, 2e-3)] {
                let (input, residual, weight) = (
                    input.to_dtype(dtype).unwrap(),
                    residual.to_dtype(dtype).unwrap(),
                    weight.to_dtype(dtype).unwrap(),
                );
                let normed = crate::kernels::metal::try_fused_rms_norm(&input, &weight, 1e-5)
                    .expect("fused RMSNorm");
                let expected = (&residual + &normed).unwrap();
                let fused = try_rms_norm_add(&input, &residual, &weight, 1e-5).expect("fused");
                let difference = worst(&fused, &expected);
                assert!(
                    difference <= tolerance,
                    "{dtype:?} {rows}x{hidden}: {difference}"
                );
            }
        }
    }

    #[test]
    fn sigmoid_gate_matches_the_unfused_ops() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("sigmoid_gate_matches_the_unfused_ops")
        else {
            return;
        };
        let (rows, heads, head_dim) = (3, 32, 64);
        let values = Tensor::from_vec(
            xorshift_values(rows * heads * head_dim, 5, 2.0),
            (rows, heads * head_dim),
            &device,
        )
        .unwrap();
        let gate = Tensor::from_vec(xorshift_values(rows * heads, 6, 6.0), (rows, heads), &device)
            .unwrap();
        let expected = values
            .reshape((rows, heads, head_dim))
            .unwrap()
            .broadcast_mul(
                &candle_nn::ops::sigmoid(&gate)
                    .unwrap()
                    .reshape((rows, heads, 1))
                    .unwrap(),
            )
            .unwrap()
            .reshape((rows, heads * head_dim))
            .unwrap();
        let fused = try_sigmoid_gate(&values, &gate, head_dim).expect("fused");
        let difference = worst(&fused, &expected);
        assert!(difference <= 1e-6, "{difference}");
        // A gate that does not tile the heads is refused, not misread.
        assert!(try_sigmoid_gate(&values, &gate.narrow(1, 0, 31).unwrap(), head_dim).is_none());
    }
}

#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "metal")]
use candle_core::{
    backend::BackendStorage, bail, CpuStorage, CustomOp2, CustomOp3, DType, Layout, MetalStorage,
    Result as CandleResult, Shape, Tensor,
};

#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

#[cfg(feature = "metal")]
const NEMOTRON_METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void nemotron_depthwise_conv1d_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    constant uint& kernel_width [[buffer(7)]],
    constant uint& padding [[buffer(8)]],
    constant uint& stride [[buffer(9)]],
    constant uint& dilation [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch * channels * output_width;
    if (gid >= total) {
        return;
    }

    uint out_t = gid % output_width;
    uint channel = (gid / output_width) % channels;
    uint batch_idx = gid / (output_width * channels);

    float acc = 0.0f;
    int base_t = int(out_t * stride) - int(padding);
    uint input_batch_base = batch_idx * channels * input_width;
    uint input_channel_base = input_batch_base + channel * input_width;
    uint weight_channel_base = channel * kernel_width;

    for (uint k = 0; k < kernel_width; ++k) {
        int input_t = base_t + int(k * dilation);
        if (input_t >= 0 && input_t < int(input_width)) {
            acc += input[input_channel_base + uint(input_t)] * weight[weight_channel_base + k];
        }
    }

    output[gid] = acc;
}

kernel void nemotron_depthwise_conv2d_bias_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& input_height [[buffer(6)]],
    constant uint& input_width [[buffer(7)]],
    constant uint& output_height [[buffer(8)]],
    constant uint& output_width [[buffer(9)]],
    constant uint& kernel_height [[buffer(10)]],
    constant uint& kernel_width [[buffer(11)]],
    constant uint& padding [[buffer(12)]],
    constant uint& stride [[buffer(13)]],
    constant uint& dilation [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch * channels * output_height * output_width;
    if (gid >= total) {
        return;
    }

    uint out_w = gid % output_width;
    uint out_h = (gid / output_width) % output_height;
    uint channel = (gid / (output_width * output_height)) % channels;
    uint batch_idx = gid / (output_width * output_height * channels);

    int base_h = int(out_h * stride) - int(padding);
    int base_w = int(out_w * stride) - int(padding);
    uint input_batch_base = batch_idx * channels * input_height * input_width;
    uint input_channel_base = input_batch_base + channel * input_height * input_width;
    uint weight_channel_base = channel * kernel_height * kernel_width;

    float acc = bias[channel];
    for (uint kh = 0; kh < kernel_height; ++kh) {
        int input_h = base_h + int(kh * dilation);
        if (input_h < 0 || input_h >= int(input_height)) {
            continue;
        }
        for (uint kw = 0; kw < kernel_width; ++kw) {
            int input_w = base_w + int(kw * dilation);
            if (input_w >= 0 && input_w < int(input_width)) {
                uint input_idx = input_channel_base + uint(input_h) * input_width + uint(input_w);
                uint weight_idx = weight_channel_base + kh * kernel_width + kw;
                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[gid] = acc;
}
"#;

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct DepthwiseConv1dOp {
    padding: usize,
    stride: usize,
    dilation: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy)]
struct DepthwiseConv2dBiasOp {
    padding: usize,
    stride: usize,
    dilation: usize,
}

#[cfg(feature = "metal")]
impl CustomOp2 for DepthwiseConv1dOp {
    fn name(&self) -> &'static str {
        "nemotron-depthwise-conv1d-metal"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        bail!("nemotron-depthwise-conv1d-metal requires a Metal tensor")
    }

    fn metal_fwd(
        &self,
        input_storage: &MetalStorage,
        input_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if input_storage.dtype() != DType::F32 || weight_storage.dtype() != DType::F32 {
            bail!("nemotron-depthwise-conv1d-metal only supports F32 tensors")
        }
        if !input_layout.is_contiguous() || !weight_layout.is_contiguous() {
            bail!("nemotron-depthwise-conv1d-metal requires contiguous inputs")
        }

        let input_dims = input_layout.dims();
        let weight_dims = weight_layout.dims();
        if input_dims.len() != 3 || weight_dims.len() != 3 {
            bail!("nemotron-depthwise-conv1d-metal expects rank-3 tensors")
        }

        let batch = input_dims[0];
        let channels = input_dims[1];
        let input_width = input_dims[2];
        let out_channels = weight_dims[0];
        let in_per_group = weight_dims[1];
        let kernel_width = weight_dims[2];
        if out_channels != channels || in_per_group != 1 {
            bail!("nemotron-depthwise-conv1d-metal expects depthwise weights shaped [C,1,K]")
        }
        if self.stride == 0 || self.dilation == 0 {
            bail!("nemotron-depthwise-conv1d-metal received invalid stride/dilation")
        }
        if input_width + 2 * self.padding < self.dilation * (kernel_width - 1) + 1 {
            bail!("nemotron-depthwise-conv1d-metal output width would underflow")
        }

        let output_width =
            (input_width + 2 * self.padding - self.dilation * (kernel_width - 1) - 1) / self.stride
                + 1;
        let elem_count = batch * channels * output_width;

        let device = input_storage.device().clone();
        let output = device.new_buffer(elem_count, DType::F32, "nemotron-depthwise-conv1d")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-depthwise-conv1d");
        let pipeline = depthwise_conv1d_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(
            0,
            Some(input_storage.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(batch as u32));
        encoder.set_bytes(4, &(channels as u32));
        encoder.set_bytes(5, &(input_width as u32));
        encoder.set_bytes(6, &(output_width as u32));
        encoder.set_bytes(7, &(kernel_width as u32));
        encoder.set_bytes(8, &(self.padding as u32));
        encoder.set_bytes(9, &(self.stride as u32));
        encoder.set_bytes(10, &(self.dilation as u32));

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
            MetalStorage::new(output, device, elem_count, DType::F32),
            Shape::from_dims(&[batch, channels, output_width]),
        ))
    }
}

#[cfg(feature = "metal")]
impl CustomOp3 for DepthwiseConv2dBiasOp {
    fn name(&self) -> &'static str {
        "nemotron-depthwise-conv2d-bias-metal"
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
        bail!("nemotron-depthwise-conv2d-bias-metal requires a Metal tensor")
    }

    fn metal_fwd(
        &self,
        input_storage: &MetalStorage,
        input_layout: &Layout,
        weight_storage: &MetalStorage,
        weight_layout: &Layout,
        bias_storage: &MetalStorage,
        bias_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        if input_storage.dtype() != DType::F32
            || weight_storage.dtype() != DType::F32
            || bias_storage.dtype() != DType::F32
        {
            bail!("nemotron-depthwise-conv2d-bias-metal only supports F32 tensors")
        }
        if !input_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !bias_layout.is_contiguous()
        {
            bail!("nemotron-depthwise-conv2d-bias-metal requires contiguous inputs")
        }

        let input_dims = input_layout.dims();
        let weight_dims = weight_layout.dims();
        let bias_dims = bias_layout.dims();
        if input_dims.len() != 4 || weight_dims.len() != 4 || bias_dims.len() != 1 {
            bail!("nemotron-depthwise-conv2d-bias-metal expects ranks [4,4,1]")
        }

        let batch = input_dims[0];
        let channels = input_dims[1];
        let input_height = input_dims[2];
        let input_width = input_dims[3];
        let out_channels = weight_dims[0];
        let in_per_group = weight_dims[1];
        let kernel_height = weight_dims[2];
        let kernel_width = weight_dims[3];
        if out_channels != channels || in_per_group != 1 || bias_dims[0] != channels {
            bail!("nemotron-depthwise-conv2d-bias-metal expects [B,C,H,W], [C,1,Kh,Kw], [C]")
        }
        if self.stride == 0 || self.dilation == 0 {
            bail!("nemotron-depthwise-conv2d-bias-metal received invalid stride/dilation")
        }
        if input_height + 2 * self.padding < self.dilation * (kernel_height - 1) + 1
            || input_width + 2 * self.padding < self.dilation * (kernel_width - 1) + 1
        {
            bail!("nemotron-depthwise-conv2d-bias-metal output shape would underflow")
        }

        let output_height =
            (input_height + 2 * self.padding - self.dilation * (kernel_height - 1) - 1)
                / self.stride
                + 1;
        let output_width =
            (input_width + 2 * self.padding - self.dilation * (kernel_width - 1) - 1) / self.stride
                + 1;
        let elem_count = batch * channels * output_height * output_width;

        let device = input_storage.device().clone();
        let output = device.new_buffer(elem_count, DType::F32, "nemotron-depthwise-conv2d-bias")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("nemotron-depthwise-conv2d-bias");
        let pipeline = depthwise_conv2d_bias_pipeline(device.metal_device())?;
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(
            0,
            Some(input_storage.buffer()),
            input_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            1,
            Some(weight_storage.buffer()),
            weight_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(
            2,
            Some(bias_storage.buffer()),
            bias_layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(batch as u32));
        encoder.set_bytes(5, &(channels as u32));
        encoder.set_bytes(6, &(input_height as u32));
        encoder.set_bytes(7, &(input_width as u32));
        encoder.set_bytes(8, &(output_height as u32));
        encoder.set_bytes(9, &(output_width as u32));
        encoder.set_bytes(10, &(kernel_height as u32));
        encoder.set_bytes(11, &(kernel_width as u32));
        encoder.set_bytes(12, &(self.padding as u32));
        encoder.set_bytes(13, &(self.stride as u32));
        encoder.set_bytes(14, &(self.dilation as u32));

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
            MetalStorage::new(output, device, elem_count, DType::F32),
            Shape::from_dims(&[batch, channels, output_height, output_width]),
        ))
    }
}

#[cfg(feature = "metal")]
fn depthwise_conv1d_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
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
        .new_library_with_source(NEMOTRON_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("nemotron_depthwise_conv1d_f32", None)
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
fn depthwise_conv2d_bias_pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
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
        .new_library_with_source(NEMOTRON_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function("nemotron_depthwise_conv2d_bias_f32", None)
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
pub fn depthwise_conv1d_enabled_for_device(device: &candle_core::Device) -> bool {
    device.is_metal()
        && std::env::var("IZWI_NEMOTRON_METAL_DEPTHWISE_CONV1D")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(true)
}

#[cfg(feature = "metal")]
pub fn depthwise_conv2d_enabled_for_device(device: &candle_core::Device) -> bool {
    device.is_metal()
        && std::env::var("IZWI_NEMOTRON_METAL_DEPTHWISE_CONV2D")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(true)
}

#[cfg(feature = "metal")]
pub fn try_depthwise_conv1d(
    input: &Tensor,
    weight: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Option<Tensor> {
    if input.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        return None;
    }
    if !depthwise_conv1d_enabled_for_device(input.device()) || !weight.device().is_metal() {
        return None;
    }
    let (batch, channels, input_width) = input.dims3().ok()?;
    let (out_channels, in_per_group, kernel_width) = weight.dims3().ok()?;
    if batch == 0
        || channels == 0
        || input_width == 0
        || out_channels != channels
        || in_per_group != 1
        || kernel_width == 0
        || stride == 0
        || dilation == 0
    {
        return None;
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous().ok()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous().ok()?
    };

    input
        .apply_op2_no_bwd(
            &weight,
            &DepthwiseConv1dOp {
                padding,
                stride,
                dilation,
            },
        )
        .ok()
}

#[cfg(feature = "metal")]
pub fn try_depthwise_conv2d_bias(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Option<Tensor> {
    if input.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        return None;
    }
    if !depthwise_conv2d_enabled_for_device(input.device())
        || !weight.device().is_metal()
        || !bias.device().is_metal()
    {
        return None;
    }
    let (batch, channels, input_height, input_width) = input.dims4().ok()?;
    let (out_channels, in_per_group, kernel_height, kernel_width) = weight.dims4().ok()?;
    let bias_channels = bias.dims1().ok()?;
    if batch == 0
        || channels == 0
        || input_height == 0
        || input_width == 0
        || out_channels != channels
        || in_per_group != 1
        || bias_channels != channels
        || kernel_height == 0
        || kernel_width == 0
        || stride == 0
        || dilation == 0
    {
        return None;
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous().ok()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous().ok()?
    };
    let bias = if bias.is_contiguous() {
        bias.clone()
    } else {
        bias.contiguous().ok()?
    };

    input
        .apply_op3_no_bwd(
            &weight,
            &bias,
            &DepthwiseConv2dBiasOp {
                padding,
                stride,
                dilation,
            },
        )
        .ok()
}

#[cfg(not(feature = "metal"))]
pub fn try_depthwise_conv1d(
    _input: &candle_core::Tensor,
    _weight: &candle_core::Tensor,
    _padding: usize,
    _stride: usize,
    _dilation: usize,
) -> Option<candle_core::Tensor> {
    None
}

#[cfg(not(feature = "metal"))]
pub fn try_depthwise_conv2d_bias(
    _input: &candle_core::Tensor,
    _weight: &candle_core::Tensor,
    _bias: &candle_core::Tensor,
    _padding: usize,
    _stride: usize,
    _dilation: usize,
) -> Option<candle_core::Tensor> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn depthwise_conv1d_falls_back_on_cpu() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let weight = Tensor::from_vec(vec![0.25f32, 0.5, 0.25], (1, 1, 3), &device).unwrap();

        assert!(try_depthwise_conv1d(&input, &weight, 1, 1, 1).is_none());
    }

    #[test]
    fn depthwise_conv2d_bias_falls_back_on_cpu() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            (0..16).map(|idx| idx as f32).collect(),
            (1, 1, 4, 4),
            &device,
        )
        .unwrap();
        let weight = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], (1, 1, 2, 2), &device).unwrap();
        let bias = Tensor::from_vec(vec![0.125f32], 1, &device).unwrap();

        assert!(try_depthwise_conv2d_bias(&input, &weight, &bias, 0, 1, 1).is_none());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn depthwise_conv1d_metal_matches_candle_reference() -> candle_core::Result<()> {
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            _ => return Ok(()),
        };

        let input_values: Vec<f32> = (0..40).map(|idx| (idx as f32 - 11.0) * 0.125).collect();
        let weight_values: Vec<f32> = (0..15).map(|idx| (idx as f32 - 4.0) * 0.0625).collect();
        let input = Tensor::from_vec(input_values, (2, 2, 10), &device)?;
        let weight = Tensor::from_vec(weight_values, (2, 1, 5), &device)?;

        let reference = input.conv1d(&weight, 2, 1, 1, 2)?;
        let actual = try_depthwise_conv1d(&input, &weight, 2, 1, 1)
            .expect("metal depthwise conv1d should run");

        let reference = reference.to_vec3::<f32>()?;
        let actual = actual.to_vec3::<f32>()?;
        for (reference_b, actual_b) in reference.iter().zip(actual.iter()) {
            for (reference_c, actual_c) in reference_b.iter().zip(actual_b.iter()) {
                for (reference, actual) in reference_c.iter().zip(actual_c.iter()) {
                    assert!(
                        (reference - actual).abs() < 1e-4,
                        "reference={reference} actual={actual}"
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn depthwise_conv2d_bias_metal_matches_candle_reference() -> candle_core::Result<()> {
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            _ => return Ok(()),
        };

        let input_values: Vec<f32> = (0..150).map(|idx| (idx as f32 - 21.0) * 0.03125).collect();
        let weight_values: Vec<f32> = (0..27).map(|idx| (idx as f32 - 9.0) * 0.015625).collect();
        let bias_values: Vec<f32> = vec![-0.25, 0.125, 0.5];
        let input = Tensor::from_vec(input_values, (2, 3, 5, 5), &device)?;
        let weight = Tensor::from_vec(weight_values, (3, 1, 3, 3), &device)?;
        let bias = Tensor::from_vec(bias_values, 3, &device)?;

        let reference = input
            .conv2d(&weight, 1, 2, 1, 3)?
            .broadcast_add(&bias.reshape((1, 3, 1, 1))?)?;
        let actual = try_depthwise_conv2d_bias(&input, &weight, &bias, 1, 2, 1)
            .expect("metal depthwise conv2d bias should run");

        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        for (reference, actual) in reference.iter().zip(actual.iter()) {
            assert!(
                (reference - actual).abs() < 1e-4,
                "reference={reference} actual={actual}"
            );
        }

        Ok(())
    }
}

//! Mamba2 mixer kernels (the selective state space of Nemotron-H's Mamba2
//! layers), for CPU, Metal and CUDA.
//!
//! [`selective_scan`] runs the recurrence over a span of `T` tokens for `B`
//! independent rows, each from its own state, in F32:
//!
//! ```text
//! dt[t,h]  = softplus(dt_raw[t,h] + dt_bias[h])        (v > 20 gives v)
//! dA       = exp(A[h] * dt[t,h])                       (A = -exp(A_log))
//! S[h,p,n] = S[h,p,n] * dA + (B[t,g,n] * dt[t,h]) * x[t,h,p]
//! y[t,h,p] = sum_n S[h,p,n] * C[t,g,n] + D[h] * x[t,h,p]
//! ```
//!
//! Head `h` reads group `g = h / (heads / groups)` (contiguous groups, as
//! Nemotron-H's checkpoints lay them out). There is no dt clamp.
//!
//! Prefill and decode run the same recurrence. SSD's chunked form (what vLLM
//! prefills with) is algebraically equal; the tests check the chunked form
//! against the recurrence at chunk-boundary lengths, so one numeric path
//! serves every span and a row never changes arithmetic between prefill and
//! decode. The GPU kernels keep each head's state in registers: one
//! threadgroup (block) per (row, head), `head_dim * lanes` threads, each lane
//! holding `state / lanes` columns of one `p` row, reduced across its lanes
//! with SIMD shuffles. `exp` and `log` are the precise forms on every device.
//!
//! [`causal_conv_silu`] and [`gated_group_rms_norm`] are composed from Candle
//! ops, so they run on every backend unchanged.
use candle_core::{DType, Device, Result, Tensor, D};

/// Above this, softplus returns its input (PyTorch's threshold).
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;
/// Most state columns one GPU lane holds in registers.
const MAX_COLUMNS_PER_LANE: usize = 32;

/// One Mamba2 mixer's SSM geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Geometry {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state: usize,
}

impl Mamba2Geometry {
    pub fn validate(&self) -> Result<()> {
        if self.heads == 0
            || self.head_dim == 0
            || self.groups == 0
            || !self.heads.is_multiple_of(self.groups)
            || !self.state.is_power_of_two()
            || self.state < 8
            || self.state > 256
            || self.head_dim * self.lanes() > 1024
            || self.head_dim * self.heads > u32::MAX as usize
        {
            candle_core::bail!("unsupported Mamba2 SSM geometry {self:?}")
        }
        Ok(())
    }

    /// `heads * head_dim`: the width of `x`, `y` and `z`.
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    /// The convolved channels `x | B | C`.
    pub fn conv_channels(&self) -> usize {
        self.inner() + 2 * self.groups * self.state
    }

    /// The group head `head` reads `B` and `C` from.
    pub fn group_of(&self, head: usize) -> usize {
        head / (self.heads / self.groups)
    }

    /// F32 values of one row's SSM state.
    pub fn state_elements(&self) -> usize {
        self.heads * self.head_dim * self.state
    }

    /// State columns per GPU lane.
    fn columns_per_lane(&self) -> usize {
        self.state.min(MAX_COLUMNS_PER_LANE)
    }

    /// GPU lanes sharing one `p` row of a head's state.
    fn lanes(&self) -> usize {
        self.state / self.columns_per_lane()
    }
}

/// `softplus(v)`, PyTorch's form: `v` above the threshold, else
/// `log1p(exp(v))`.
pub fn softplus(v: f32) -> f32 {
    if v > SOFTPLUS_THRESHOLD {
        v
    } else {
        v.exp().ln_1p()
    }
}

/// The selective scan of `T` tokens for `B` rows.
///
/// * `xbc` `[B, T, conv_channels]`: the convolved `x | B | C`, after SiLU;
/// * `dt` `[B, T, heads]`: raw time steps (before `dt_bias` and softplus);
/// * `params` `[3, heads]`: `dt_bias`, `A` (= `-exp(A_log)`) and `D`;
/// * `state` `[B, heads, head_dim, state]`: the rows' states before the span.
///
/// All F32 and on one device. Returns `y` `[B, T, inner]` (without the gate)
/// and the states after the span. The input state is never written, so a
/// caller may keep it as a rollback checkpoint.
pub fn selective_scan(
    geometry: Mamba2Geometry,
    xbc: &Tensor,
    dt: &Tensor,
    params: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    geometry.validate()?;
    let (rows, tokens, channels) = xbc.dims3()?;
    let device = xbc.device();
    if channels != geometry.conv_channels()
        || dt.dims() != [rows, tokens, geometry.heads]
        || params.dims() != [3, geometry.heads]
        || state.dims() != [rows, geometry.heads, geometry.head_dim, geometry.state]
        || rows == 0
        || tokens == 0
        || [xbc, dt, params, state]
            .iter()
            .any(|t| t.dtype() != DType::F32 || !t.device().same_device(device))
    {
        candle_core::bail!("invalid Mamba2 selective-scan operands for {geometry:?}")
    }
    if rows
        .checked_mul(tokens)
        .and_then(|v| v.checked_mul(channels))
        .is_none_or(|v| v > u32::MAX as usize)
    {
        candle_core::bail!("Mamba2 selective scan is too large for one launch")
    }
    let xbc = xbc.contiguous()?;
    let dt = dt.contiguous()?;
    let params = params.contiguous()?;
    let state = state.contiguous()?;
    match device {
        Device::Cpu => cpu_scan(geometry, &xbc, &dt, &params, &state),
        #[cfg(feature = "metal")]
        Device::Metal(_) => metal_impl::scan(geometry, rows, tokens, &xbc, &dt, &params, &state),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::scan(geometry, rows, tokens, &xbc, &dt, &params, &state),
        #[allow(unreachable_patterns)]
        _ => candle_core::bail!("the Mamba2 selective scan has no kernel for this device"),
    }
}

fn cpu_scan(
    geometry: Mamba2Geometry,
    xbc: &Tensor,
    dt: &Tensor,
    params: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use rayon::prelude::*;
    let (rows, tokens, channels) = xbc.dims3()?;
    let xbc_values = xbc.flatten_all()?.to_vec1::<f32>()?;
    let dt_values = dt.flatten_all()?.to_vec1::<f32>()?;
    let params = params.flatten_all()?.to_vec1::<f32>()?;
    let mut states = state.flatten_all()?.to_vec1::<f32>()?;
    let Mamba2Geometry {
        heads,
        head_dim,
        groups,
        state: width,
    } = geometry;
    let inner = geometry.inner();
    let head_state = head_dim * width;
    // Each (row, head) owns its state slice and writes its own y columns.
    let outputs = states
        .par_chunks_mut(head_state)
        .enumerate()
        .map(|(index, s)| {
            let (row, head) = (index / heads, index % heads);
            let group = geometry.group_of(head);
            let (bias, a, d) = (params[head], params[heads + head], params[2 * heads + head]);
            let mut y = vec![0f32; tokens * head_dim];
            for t in 0..tokens {
                let base = (row * tokens + t) * channels;
                let step = softplus(dt_values[(row * tokens + t) * heads + head] + bias);
                let decay = (a * step).exp();
                let b = &xbc_values[base + inner + group * width..][..width];
                let c = &xbc_values[base + inner + groups * width + group * width..][..width];
                for p in 0..head_dim {
                    let x = xbc_values[base + head * head_dim + p];
                    let row_state = &mut s[p * width..(p + 1) * width];
                    let mut acc = 0f32;
                    for n in 0..width {
                        row_state[n] = row_state[n] * decay + (b[n] * step) * x;
                        acc += row_state[n] * c[n];
                    }
                    y[t * head_dim + p] = acc + d * x;
                }
            }
            y
        })
        .collect::<Vec<_>>();
    let mut y = vec![0f32; rows * tokens * inner];
    for (index, head_y) in outputs.into_iter().enumerate() {
        let (row, head) = (index / heads, index % heads);
        for t in 0..tokens {
            let dst = (row * tokens + t) * inner + head * head_dim;
            y[dst..dst + head_dim].copy_from_slice(&head_y[t * head_dim..(t + 1) * head_dim]);
        }
    }
    Ok((
        Tensor::from_vec(y, (rows, tokens, inner), &Device::Cpu)?,
        Tensor::from_vec(states, (rows, heads, head_dim, width), &Device::Cpu)?,
    ))
}

/// The causal depthwise convolution of a Mamba2 mixer, with bias and SiLU,
/// over `xbc` `[B, T, C]` (the pre-convolution rows) after `history`
/// `[B, K - 1, C]` (the previous span's last `K - 1` pre-convolution rows,
/// zeros at the start). `weight` is `[C, K]` (tap `K - 1` multiplies the
/// current token) and `bias` `[C]`, all F32. Returns the activations
/// `[B, T, C]` and the next history.
pub fn causal_conv_silu(
    xbc: &Tensor,
    history: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (rows, tokens, channels) = xbc.dims3()?;
    let (taps_channels, taps) = weight.dims2()?;
    if taps < 2
        || taps_channels != channels
        || history.dims() != [rows, taps - 1, channels]
        || bias.dims() != [channels]
    {
        candle_core::bail!("invalid Mamba2 convolution operands")
    }
    let padded = Tensor::cat(&[history, xbc], 1)?;
    let mut out = bias
        .reshape((1, 1, channels))?
        .broadcast_as((rows, tokens, channels))?;
    for tap in 0..taps {
        let w = weight.narrow(1, tap, 1)?.reshape((1, 1, channels))?;
        out = (out + padded.narrow(1, tap, tokens)?.broadcast_mul(&w)?)?;
    }
    let next = padded.narrow(1, tokens, taps - 1)?.contiguous()?;
    Ok((candle_nn::ops::silu(&out)?, next))
}

/// Mamba2's gated RMSNorm: `g = y * silu(z)` (the gate first), then RMS over
/// each `group_size` channels, times `weight`. `y` and `z` are `[.., inner]`
/// F32 and `weight` `[inner]`.
pub fn gated_group_rms_norm(
    y: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    group_size: usize,
    eps: f64,
) -> Result<Tensor> {
    let dims = y.dims().to_vec();
    let inner = *dims.last().unwrap_or(&0);
    if z.dims() != dims
        || weight.dims() != [inner]
        || group_size == 0
        || !inner.is_multiple_of(group_size)
    {
        candle_core::bail!("invalid Mamba2 gated norm operands")
    }
    let gated = (y * candle_nn::ops::silu(z)?)?;
    let mut grouped_dims = dims.clone();
    *grouped_dims.last_mut().expect("rank checked") = inner / group_size;
    grouped_dims.push(group_size);
    let grouped = gated.reshape(grouped_dims)?;
    let rms = (grouped.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    grouped
        .broadcast_div(&rms)?
        .reshape(dims)?
        .broadcast_mul(weight)
}

/// Compares the device scan with the CPU scan on unstructured inputs at
/// lengths around a 128-token chunk boundary (1, 127, 128, 129, 300), at a
/// small geometry of `state` columns. Returns the worst relative L2 error;
/// a load uses it as its self-check.
pub fn verify_device(device: &Device, state: usize) -> Result<f64> {
    use super::numeric_check::{host_f32, relative_l2, xorshift_tensor};
    let geometry = Mamba2Geometry {
        heads: 4,
        head_dim: 16,
        groups: 2,
        state,
    };
    let mut worst = 0f64;
    for (seed, tokens) in [1usize, 127, 128, 129, 300].into_iter().enumerate() {
        let seed = seed as u64 * 7 + 1;
        let xbc = xorshift_tensor(
            &[2, tokens, geometry.conv_channels()],
            seed,
            1.0,
            &Device::Cpu,
        )?;
        let dt = xorshift_tensor(&[2, tokens, geometry.heads], seed + 1, 3.0, &Device::Cpu)?;
        let a = (xorshift_tensor(&[geometry.heads], seed + 2, 1.0, &Device::Cpu)?.exp()? * -1.0)?;
        let params = Tensor::stack(
            &[
                xorshift_tensor(&[geometry.heads], seed + 3, 1.0, &Device::Cpu)?,
                a,
                xorshift_tensor(&[geometry.heads], seed + 4, 1.0, &Device::Cpu)?,
            ],
            0,
        )?;
        let state0 = xorshift_tensor(
            &[2, geometry.heads, geometry.head_dim, geometry.state],
            seed + 5,
            0.5,
            &Device::Cpu,
        )?;
        let (y_ref, s_ref) = selective_scan(geometry, &xbc, &dt, &params, &state0)?;
        let (y, s) = selective_scan(
            geometry,
            &xbc.to_device(device)?,
            &dt.to_device(device)?,
            &params.to_device(device)?,
            &state0.to_device(device)?,
        )?;
        worst = worst
            .max(relative_l2(&host_f32(&y)?, &host_f32(&y_ref)?))
            .max(relative_l2(&host_f32(&s)?, &host_f32(&s_ref)?));
    }
    Ok(worst)
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    use super::Mamba2Geometry;
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan(
        geometry: Mamba2Geometry,
        rows: usize,
        tokens: usize,
        xbc: &Tensor,
        dt: &Tensor,
        params: &Tensor,
        state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (xbc_storage, xbc_layout) = xbc.storage_and_layout();
        let (dt_storage, dt_layout) = dt.storage_and_layout();
        let (params_storage, params_layout) = params.storage_and_layout();
        let (state_storage, state_layout) = state.storage_and_layout();
        let (
            Storage::Metal(xbc_m),
            Storage::Metal(dt_m),
            Storage::Metal(params_m),
            Storage::Metal(state_m),
        ) = (
            &*xbc_storage,
            &*dt_storage,
            &*params_storage,
            &*state_storage,
        )
        else {
            candle_core::bail!("the Metal Mamba2 scan needs Metal tensors")
        };
        let device = xbc_m.device().clone();
        let inner = geometry.inner();
        let y_count = rows * tokens * inner;
        let state_count = rows * geometry.state_elements();
        let y = device.new_buffer(y_count, DType::F32, "izwi-mamba2-y")?;
        let next = device.new_buffer(state_count, DType::F32, "izwi-mamba2-state")?;
        let function = format!("izwi_mamba2_scan_c{}", geometry.columns_per_lane());
        let pipeline = pipeline(device.metal_device(), &function)?;
        let dims: [u32; 8] = [
            tokens as u32,
            geometry.heads as u32,
            geometry.head_dim as u32,
            geometry.groups as u32,
            geometry.state as u32,
            geometry.lanes() as u32,
            geometry.conv_channels() as u32,
            (geometry.heads / geometry.groups) as u32,
        ];
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-mamba2-scan");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(xbc_m.buffer()), xbc_layout.start_offset() * 4);
        encoder.set_input_buffer(1, Some(dt_m.buffer()), dt_layout.start_offset() * 4);
        encoder.set_input_buffer(2, Some(params_m.buffer()), params_layout.start_offset() * 4);
        encoder.set_input_buffer(3, Some(state_m.buffer()), state_layout.start_offset() * 4);
        encoder.set_output_buffer(4, Some(&y), 0);
        encoder.set_output_buffer(5, Some(&next), 0);
        encoder.set_bytes(6, &dims);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: geometry.heads,
                height: rows,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: geometry.head_dim * geometry.lanes(),
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        let wrap = |buffer, count, shape: Shape| {
            Tensor::from_storage(
                Storage::Metal(MetalStorage::new(buffer, device.clone(), count, DType::F32)),
                shape,
                BackpropOp::none(),
                false,
            )
        };
        Ok((
            wrap(y, y_count, Shape::from((rows, tokens, inner))),
            wrap(
                next,
                state_count,
                Shape::from((rows, geometry.heads, geometry.head_dim, geometry.state)),
            ),
        ))
    }

    fn pipeline(device: &MetalDevice, function: &str) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> =
            OnceLock::new();
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

// log1p(exp(v)) with PyTorch's threshold, in precise arithmetic: the fast-math
// default would round exp and log, and dt feeds an exponential decay.
static inline float izwi_mamba2_softplus(float v) {
    if (v > 20.0f) {
        return v;
    }
    const float u = precise::exp(v);
    // log1p: below 1e-4 the series is exact to F32; above it 1 + u keeps
    // every bit log needs.
    return u < 1.0e-4f ? u * (1.0f - u * (0.5f - u * (1.0f / 3.0f))) : precise::log(1.0f + u);
}

// dims: tokens, heads, head_dim, groups, state, lanes, conv channels,
// heads per group. One threadgroup per (head, row); thread = p * lanes + lane
// holds COLS state columns of row p.
#define IZWI_MAMBA2_SCAN(COLS)                                                    \
kernel void izwi_mamba2_scan_c##COLS(                                             \
    device const float* xbc [[buffer(0)]],                                        \
    device const float* dt [[buffer(1)]],                                         \
    device const float* params [[buffer(2)]],                                     \
    device const float* state_in [[buffer(3)]],                                   \
    device float* y [[buffer(4)]],                                                \
    device float* state_out [[buffer(5)]],                                        \
    constant uint* dims [[buffer(6)]],                                            \
    uint2 local [[thread_position_in_threadgroup]],                               \
    uint2 extent [[threads_per_threadgroup]],                                     \
    uint2 group [[threadgroup_position_in_grid]]                                  \
) {                                                                               \
    threadgroup float bc[512];                                                    \
    const uint tid = local.x, threads = extent.x;                                 \
    const uint tokens = dims[0], heads = dims[1], head_dim = dims[2];             \
    const uint groups = dims[3], width = dims[4], lanes = dims[5];                \
    const uint channels = dims[6], per_group = dims[7];                           \
    const uint head = group.x, row = group.y;                                     \
    const uint p = tid / lanes, lane = tid % lanes;                               \
    const uint g = head / per_group;                                              \
    const uint inner = heads * head_dim;                                          \
    const uint n0 = lane * COLS;                                                  \
    const ulong state_base = ((ulong(row) * heads + head) * head_dim + p) * width + n0; \
    float s[COLS];                                                                \
    for (uint i = 0; i < COLS; ++i) {                                             \
        s[i] = state_in[state_base + i];                                          \
    }                                                                             \
    const float bias = params[head];                                              \
    const float a = params[heads + head];                                         \
    const float d = params[2 * heads + head];                                     \
    for (uint t = 0; t < tokens; ++t) {                                           \
        const ulong token = ulong(row) * tokens + t;                              \
        const ulong base = token * channels;                                      \
        for (uint i = tid; i < 2 * width; i += threads) {                         \
            bc[i] = i < width ? xbc[base + inner + g * width + i]                 \
                              : xbc[base + inner + (groups + g) * width + (i - width)]; \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
        const float step = izwi_mamba2_softplus(dt[token * heads + head] + bias); \
        const float decay = precise::exp(a * step);                               \
        const float x = xbc[base + head * head_dim + p];                          \
        float acc = 0.0f;                                                         \
        for (uint i = 0; i < COLS; ++i) {                                         \
            s[i] = s[i] * decay + (bc[n0 + i] * step) * x;                        \
            acc += s[i] * bc[width + n0 + i];                                     \
        }                                                                         \
        for (uint offset = lanes >> 1; offset > 0; offset >>= 1) {                \
            acc += simd_shuffle_xor(acc, (ushort)offset);                         \
        }                                                                         \
        if (lane == 0) {                                                          \
            y[token * inner + head * head_dim + p] = acc + d * x;                 \
        }                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    }                                                                             \
    for (uint i = 0; i < COLS; ++i) {                                             \
        state_out[state_base + i] = s[i];                                         \
    }                                                                             \
}

IZWI_MAMBA2_SCAN(8)
IZWI_MAMBA2_SCAN(16)
IZWI_MAMBA2_SCAN(32)
"#;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Shape, Storage, Tensor};

    use super::Mamba2Geometry;

    const MODULE: &str = "izwi_mamba2";

    fn view<'a>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, f32>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("the CUDA Mamba2 scan needs CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("the CUDA Mamba2 scan needs contiguous tensors")
        };
        Ok(cuda.as_cuda_slice::<f32>()?.slice(start..end))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan(
        geometry: Mamba2Geometry,
        rows: usize,
        tokens: usize,
        xbc: &Tensor,
        dt: &Tensor,
        params: &Tensor,
        state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = xbc.device().as_cuda_device()?;
        let (xbc_storage, _) = xbc.storage_and_layout();
        let (dt_storage, _) = dt.storage_and_layout();
        let (params_storage, _) = params.storage_and_layout();
        let (state_storage, _) = state.storage_and_layout();
        let xbc_v = view(&xbc_storage, xbc)?;
        let dt_v = view(&dt_storage, dt)?;
        let params_v = view(&params_storage, params)?;
        let state_v = view(&state_storage, state)?;
        let inner = geometry.inner();
        let y_count = rows * tokens * inner;
        let state_count = rows * geometry.state_elements();
        // SAFETY: the kernel writes every element of both outputs.
        let y = unsafe { device.alloc::<f32>(y_count)? };
        let next = unsafe { device.alloc::<f32>(state_count)? };
        let name = format!("izwi_mamba2_scan_c{}", geometry.columns_per_lane());
        let function =
            device.get_or_load_custom_func(&name, MODULE, super::super::cuda::cuda_ptx_mamba2())?;
        let config = LaunchConfig {
            grid_dim: (geometry.heads as u32, rows as u32, 1),
            block_dim: ((geometry.head_dim * geometry.lanes()) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = function.builder();
        builder.arg(&xbc_v);
        builder.arg(&dt_v);
        builder.arg(&params_v);
        builder.arg(&state_v);
        builder.arg(&y);
        builder.arg(&next);
        candle_core::builder_arg!(
            builder,
            tokens as i32,
            geometry.heads as i32,
            geometry.head_dim as i32,
            geometry.groups as i32,
            geometry.state as i32,
            geometry.lanes() as i32,
            geometry.conv_channels() as i32
        );
        // SAFETY: argument order and types match the kernel signature.
        unsafe { builder.launch(config) }.w()?;
        let wrap = |slice, shape: Shape| {
            Tensor::from_storage(
                Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
                shape,
                BackpropOp::none(),
                false,
            )
        };
        Ok((
            wrap(y, Shape::from((rows, tokens, inner))),
            wrap(
                next,
                Shape::from((rows, geometry.heads, geometry.head_dim, geometry.state)),
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{max_abs_diff, relative_l2, xorshift_values};

    struct Inputs {
        geometry: Mamba2Geometry,
        rows: usize,
        tokens: usize,
        xbc: Vec<f32>,
        dt: Vec<f32>,
        params: Vec<f32>,
        state: Vec<f32>,
    }

    fn make_inputs(geometry: Mamba2Geometry, rows: usize, tokens: usize, seed: u64) -> Inputs {
        let heads = geometry.heads;
        let a = xorshift_values(heads, seed + 2, 1.0)
            .into_iter()
            .map(|v| -v.exp())
            .collect::<Vec<_>>();
        let mut params = xorshift_values(heads, seed + 3, 1.0);
        params.extend(a);
        params.extend(xorshift_values(heads, seed + 4, 1.0));
        Inputs {
            geometry,
            rows,
            tokens,
            xbc: xorshift_values(rows * tokens * geometry.conv_channels(), seed, 1.0),
            dt: xorshift_values(rows * tokens * heads, seed + 1, 3.0),
            params,
            state: xorshift_values(rows * geometry.state_elements(), seed + 5, 0.5),
        }
    }

    impl Inputs {
        fn tensors(&self, device: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
            let g = self.geometry;
            (
                Tensor::from_vec(
                    self.xbc.clone(),
                    (self.rows, self.tokens, g.conv_channels()),
                    device,
                )
                .unwrap(),
                Tensor::from_vec(self.dt.clone(), (self.rows, self.tokens, g.heads), device)
                    .unwrap(),
                Tensor::from_vec(self.params.clone(), (3, g.heads), device).unwrap(),
                Tensor::from_vec(
                    self.state.clone(),
                    (self.rows, g.heads, g.head_dim, g.state),
                    device,
                )
                .unwrap(),
            )
        }

        fn x(&self, row: usize, t: usize, h: usize, p: usize) -> f64 {
            let g = self.geometry;
            f64::from(self.xbc[(row * self.tokens + t) * g.conv_channels() + h * g.head_dim + p])
        }

        fn b(&self, row: usize, t: usize, group: usize, n: usize) -> f64 {
            let g = self.geometry;
            f64::from(
                self.xbc
                    [(row * self.tokens + t) * g.conv_channels() + g.inner() + group * g.state + n],
            )
        }

        fn c(&self, row: usize, t: usize, group: usize, n: usize) -> f64 {
            let g = self.geometry;
            f64::from(
                self.xbc[(row * self.tokens + t) * g.conv_channels()
                    + g.inner()
                    + (g.groups + group) * g.state
                    + n],
            )
        }

        fn step(&self, row: usize, t: usize, h: usize) -> f64 {
            let v = f64::from(self.dt[(row * self.tokens + t) * self.geometry.heads + h])
                + f64::from(self.params[h]);
            if v > 20.0 {
                v
            } else {
                v.exp().ln_1p()
            }
        }

        fn a(&self, h: usize) -> f64 {
            f64::from(self.params[self.geometry.heads + h])
        }

        fn d(&self, h: usize) -> f64 {
            f64::from(self.params[2 * self.geometry.heads + h])
        }

        fn state0(&self, row: usize, h: usize, p: usize, n: usize) -> f64 {
            let g = self.geometry;
            f64::from(self.state[((row * g.heads + h) * g.head_dim + p) * g.state + n])
        }

        /// The recurrence in F64: `(y [rows, tokens, inner], state)`.
        fn recurrence(&self) -> (Vec<f64>, Vec<f64>) {
            let g = self.geometry;
            let mut y = vec![0f64; self.rows * self.tokens * g.inner()];
            let mut states = vec![0f64; self.rows * g.state_elements()];
            for row in 0..self.rows {
                for h in 0..g.heads {
                    let group = g.group_of(h);
                    for p in 0..g.head_dim {
                        let mut s = (0..g.state)
                            .map(|n| self.state0(row, h, p, n))
                            .collect::<Vec<_>>();
                        for t in 0..self.tokens {
                            let step = self.step(row, t, h);
                            let decay = (self.a(h) * step).exp();
                            let x = self.x(row, t, h, p);
                            let mut acc = 0f64;
                            for (n, value) in s.iter_mut().enumerate() {
                                *value = *value * decay + self.b(row, t, group, n) * step * x;
                                acc += *value * self.c(row, t, group, n);
                            }
                            y[(row * self.tokens + t) * g.inner() + h * g.head_dim + p] =
                                acc + self.d(h) * x;
                        }
                        let base = ((row * g.heads + h) * g.head_dim + p) * g.state;
                        states[base..base + g.state].copy_from_slice(&s);
                    }
                }
            }
            (y, states)
        }

        /// SSD's chunked form (Mamba2's `ssd_minimal`) in F64: within a
        /// chunk the quadratic "attention" term, across chunks the decayed
        /// state.
        fn ssd_chunked(&self, chunk: usize) -> (Vec<f64>, Vec<f64>) {
            let g = self.geometry;
            let mut y = vec![0f64; self.rows * self.tokens * g.inner()];
            let mut states = vec![0f64; self.rows * g.state_elements()];
            for row in 0..self.rows {
                for h in 0..g.heads {
                    let group = g.group_of(h);
                    let mut s = (0..g.head_dim * g.state)
                        .map(|i| self.state0(row, h, i / g.state, i % g.state))
                        .collect::<Vec<_>>();
                    for start in (0..self.tokens).step_by(chunk) {
                        let len = chunk.min(self.tokens - start);
                        let mut cum = vec![0f64; len];
                        let mut total = 0f64;
                        for (i, value) in cum.iter_mut().enumerate() {
                            total += self.a(h) * self.step(row, start + i, h);
                            *value = total;
                        }
                        for i in 0..len {
                            let t = start + i;
                            for p in 0..g.head_dim {
                                // Off-diagonal: the state entering the chunk.
                                let mut acc = 0f64;
                                for n in 0..g.state {
                                    acc += self.c(row, t, group, n)
                                        * cum[i].exp()
                                        * s[p * g.state + n];
                                }
                                // Diagonal: tokens of this chunk up to t.
                                for j in 0..=i {
                                    let cb = (0..g.state)
                                        .map(|n| {
                                            self.c(row, t, group, n)
                                                * self.b(row, start + j, group, n)
                                        })
                                        .sum::<f64>();
                                    acc += cb
                                        * (cum[i] - cum[j]).exp()
                                        * self.step(row, start + j, h)
                                        * self.x(row, start + j, h, p);
                                }
                                y[(row * self.tokens + t) * g.inner() + h * g.head_dim + p] =
                                    acc + self.d(h) * self.x(row, t, h, p);
                            }
                        }
                        // The state leaving the chunk.
                        for p in 0..g.head_dim {
                            for n in 0..g.state {
                                let mut next = cum[len - 1].exp() * s[p * g.state + n];
                                for j in 0..len {
                                    next += (cum[len - 1] - cum[j]).exp()
                                        * self.step(row, start + j, h)
                                        * self.b(row, start + j, group, n)
                                        * self.x(row, start + j, h, p);
                                }
                                s[p * g.state + n] = next;
                            }
                        }
                    }
                    let base = (row * g.heads + h) * g.head_dim * g.state;
                    states[base..base + g.head_dim * g.state].copy_from_slice(&s);
                }
            }
            (y, states)
        }
    }

    fn f64_rel(values: &[f64], reference: &[f64]) -> f64 {
        let (error, norm) = values
            .iter()
            .zip(reference)
            .fold((0f64, 0f64), |(e, n), (&a, &b)| {
                (e + (a - b) * (a - b), n + b * b)
            });
        let r = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_finite() {
            r
        } else {
            f64::INFINITY
        }
    }

    fn to_f32(values: &[f64]) -> Vec<f32> {
        values.iter().map(|&v| v as f32).collect()
    }

    const SMALL: Mamba2Geometry = Mamba2Geometry {
        heads: 4,
        head_dim: 16,
        groups: 2,
        state: 8,
    };

    #[test]
    fn ssd_chunked_form_equals_the_recurrence_across_chunk_boundaries() {
        for tokens in [1, 7, 8, 9, 20] {
            let inputs = make_inputs(SMALL, 2, tokens, 11 + tokens as u64);
            let (y, s) = inputs.recurrence();
            let (y_chunked, s_chunked) = inputs.ssd_chunked(8);
            assert!(f64_rel(&y_chunked, &y) < 1e-12, "y at {tokens} tokens");
            assert!(f64_rel(&s_chunked, &s) < 1e-12, "state at {tokens} tokens");
        }
    }

    #[test]
    fn cpu_scan_matches_the_f64_recurrence() {
        for tokens in [1, 127, 128, 129, 300] {
            let inputs = make_inputs(SMALL, 2, tokens, tokens as u64);
            let (xbc, dt, params, state) = inputs.tensors(&Device::Cpu);
            let (y, s) = selective_scan(SMALL, &xbc, &dt, &params, &state).unwrap();
            let (y_ref, s_ref) = inputs.recurrence();
            let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let s = s.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(relative_l2(&y, &to_f32(&y_ref)) < 1e-5, "y at {tokens}");
            assert!(relative_l2(&s, &to_f32(&s_ref)) < 1e-5, "state at {tokens}");
            // The input state is untouched (a rollback checkpoint).
            assert_eq!(
                state.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                inputs.state
            );
        }
    }

    #[test]
    fn a_span_equals_its_tokens_one_at_a_time() {
        let inputs = make_inputs(SMALL, 1, 13, 5);
        let (xbc, dt, params, state) = inputs.tensors(&Device::Cpu);
        let (y_span, s_span) = selective_scan(SMALL, &xbc, &dt, &params, &state).unwrap();
        let mut s = state;
        let mut ys = Vec::new();
        for t in 0..13 {
            let (y, next) = selective_scan(
                SMALL,
                &xbc.narrow(1, t, 1).unwrap(),
                &dt.narrow(1, t, 1).unwrap(),
                &params,
                &s,
            )
            .unwrap();
            ys.push(y);
            s = next;
        }
        let y_steps = Tensor::cat(&ys, 1).unwrap();
        // The same arithmetic in the same order: exact.
        assert_eq!(
            y_steps.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            y_span.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            s.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            s_span.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// RED checks: each convention broken on purpose must move the output by
    /// far more than the tolerance the tests above use.
    #[test]
    fn broken_conventions_fail_by_far_more_than_the_tolerance() {
        let inputs = make_inputs(SMALL, 1, 40, 3);
        let (y_ref, _) = inputs.recurrence();
        let y_ref = to_f32(&y_ref);
        let run = |inputs: &Inputs, geometry: Mamba2Geometry| {
            let (xbc, dt, params, state) = inputs.tensors(&Device::Cpu);
            let (y, _) = selective_scan(geometry, &xbc, &dt, &params, &state).unwrap();
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        // Head -> group as h % groups instead of h / (heads / groups): swap
        // the B and C groups of half the heads by reordering the groups.
        let mut swapped = inputs.xbc.clone();
        let g = SMALL;
        for t in 0..40 {
            let base = t * g.conv_channels() + g.inner();
            for block in [0, g.groups] {
                for n in 0..g.state {
                    swapped.swap(base + block * g.state + n, base + (block + 1) * g.state + n);
                }
            }
        }
        let swapped = Inputs {
            xbc: swapped,
            ..make_inputs(SMALL, 1, 40, 3)
        };
        assert!(relative_l2(&run(&swapped, SMALL), &y_ref) > 1e-2);
        // A positive A (no minus sign on exp(A_log)).
        let mut flipped = make_inputs(SMALL, 1, 40, 3);
        for a in &mut flipped.params[SMALL.heads..2 * SMALL.heads] {
            *a = -*a;
        }
        assert!(relative_l2(&run(&flipped, SMALL), &y_ref) > 1e-2);
        // dt without dt_bias.
        let mut unbiased = make_inputs(SMALL, 1, 40, 3);
        for bias in &mut unbiased.params[..SMALL.heads] {
            *bias = 0.0;
        }
        assert!(relative_l2(&run(&unbiased, SMALL), &y_ref) > 1e-2);
    }

    #[test]
    fn softplus_follows_pytorch_on_both_sides_of_the_threshold() {
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        // log1p keeps tiny steps instead of rounding them to zero.
        let tiny = softplus(-30.0);
        assert!(tiny > 0.0 && (tiny - (-30f32).exp()).abs() < 1e-18);
    }

    #[test]
    fn convolution_holds_pre_convolution_history_and_matches_a_direct_sum() {
        let (rows, tokens, channels, taps) = (2, 5, 6, 4);
        let x = xorshift_values(rows * tokens * channels, 1, 1.0);
        let history = xorshift_values(rows * (taps - 1) * channels, 2, 1.0);
        let weight = xorshift_values(channels * taps, 3, 1.0);
        let bias = xorshift_values(channels, 4, 1.0);
        let t = |v: &Vec<f32>, shape: &[usize]| {
            Tensor::from_vec(v.clone(), shape, &Device::Cpu).unwrap()
        };
        let (out, next) = causal_conv_silu(
            &t(&x, &[rows, tokens, channels]),
            &t(&history, &[rows, taps - 1, channels]),
            &t(&weight, &[channels, taps]),
            &t(&bias, &[channels]),
        )
        .unwrap();
        let out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for row in 0..rows {
            for step in 0..tokens {
                for c in 0..channels {
                    let mut acc = f64::from(bias[c]);
                    for k in 0..taps {
                        // Tap k reads token step - (taps - 1) + k.
                        let source = step as isize - (taps as isize - 1) + k as isize;
                        let value = if source < 0 {
                            history[(row * (taps - 1) + (source + taps as isize - 1) as usize)
                                * channels
                                + c]
                        } else {
                            x[(row * tokens + source as usize) * channels + c]
                        };
                        acc += f64::from(weight[c * taps + k]) * f64::from(value);
                    }
                    let expected = acc / (1.0 + (-acc).exp());
                    let got = out[(row * tokens + step) * channels + c];
                    assert!((f64::from(got) - expected).abs() < 1e-5);
                }
            }
        }
        // The next history is the last K - 1 pre-convolution rows.
        let next = next.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for row in 0..rows {
            assert_eq!(
                &next[row * (taps - 1) * channels..(row + 1) * (taps - 1) * channels],
                &x[(row * tokens + tokens - (taps - 1)) * channels..(row + 1) * tokens * channels]
            );
        }
    }

    #[test]
    fn gated_norm_gates_before_the_grouped_rms() {
        let (inner, group) = (8, 4);
        let y = xorshift_values(2 * inner, 1, 2.0);
        let z = xorshift_values(2 * inner, 2, 2.0);
        let w = xorshift_values(inner, 3, 1.0);
        let t = |v: &Vec<f32>, shape: &[usize]| {
            Tensor::from_vec(v.clone(), shape, &Device::Cpu).unwrap()
        };
        let out = gated_group_rms_norm(
            &t(&y, &[2, inner]),
            &t(&z, &[2, inner]),
            &t(&w, &[inner]),
            group,
            1e-5,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let mut expected = vec![0f32; 2 * inner];
        for row in 0..2 {
            let gated = (0..inner)
                .map(|i| {
                    let z = f64::from(z[row * inner + i]);
                    f64::from(y[row * inner + i]) * z / (1.0 + (-z).exp())
                })
                .collect::<Vec<_>>();
            for start in (0..inner).step_by(group) {
                let ms = gated[start..start + group]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    / group as f64;
                for i in start..start + group {
                    expected[row * inner + i] =
                        (gated[i] / (ms + 1e-5).sqrt() * f64::from(w[i])) as f32;
                }
            }
        }
        assert!(max_abs_diff(&out, &expected) < 1e-5);
        // RED: norm before the gate differs by far more.
        let ungated_first = {
            let mut v = vec![0f32; 2 * inner];
            for row in 0..2 {
                for start in (0..inner).step_by(group) {
                    let ms = (start..start + group)
                        .map(|i| f64::from(y[row * inner + i]).powi(2))
                        .sum::<f64>()
                        / group as f64;
                    for i in start..start + group {
                        let z = f64::from(z[row * inner + i]);
                        v[row * inner + i] = (f64::from(y[row * inner + i]) / (ms + 1e-5).sqrt()
                            * f64::from(w[i])
                            * z
                            / (1.0 + (-z).exp()))
                            as f32;
                    }
                }
            }
            v
        };
        assert!(max_abs_diff(&ungated_first, &expected) > 1e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_scan_matches_the_cpu_scan_at_every_column_width() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        for state in [8, 16, 32, 128] {
            let error = verify_device(&device, state).unwrap();
            assert!(error < 1e-5, "state {state}: {error}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_scan_matches_the_cpu_scan_at_both_column_widths() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_scan_matches_the_cpu_scan_at_both_column_widths",
        ) else {
            return;
        };
        for state in [8, 32, 128] {
            let error = verify_device(&device, state).unwrap();
            assert!(error < 1e-5, "state {state}: {error}");
        }
        // The production geometry, one decode step and a span.
        let geometry = Mamba2Geometry {
            heads: 64,
            head_dim: 64,
            groups: 8,
            state: 128,
        };
        for tokens in [1, 37] {
            let inputs = make_inputs(geometry, 2, tokens, 9);
            let (xbc, dt, params, state) = inputs.tensors(&Device::Cpu);
            let (y_cpu, s_cpu) = selective_scan(geometry, &xbc, &dt, &params, &state).unwrap();
            let (xbc, dt, params, state) = inputs.tensors(&device);
            let (y, s) = selective_scan(geometry, &xbc, &dt, &params, &state).unwrap();
            let host = |t: &Tensor| {
                t.flatten_all()
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            assert!(relative_l2(&host(&y), &host(&y_cpu)) < 1e-5);
            assert!(relative_l2(&host(&s), &host(&s_cpu)) < 1e-5);
        }
    }
}

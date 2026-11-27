//! Device-routed sparse experts for sigmoid-routed MoE layers whose expert
//! weights stay in a packed quantized form, with a soft-capped SwiGLU gate
//! (Inkling 2's experts).
//!
//! Two stored forms, both read as published:
//!
//! * **NVFP4**: 4-bit E2M1 codes, two per byte (even element in the low
//!   nibble), one FP8 E4M3 scale per 16 values along a row, one F32 factor
//!   per expert. `value = code * scale * factor`.
//! * **MXFP8**: FP8 E4M3 values, one scale byte `s` per 32 values along a
//!   row. `value = e4m3 * 2^(s - 127)`.
//!
//! A layer runs with the expert ids on the device, so a decode step never
//! waits for the host inside the MoE block:
//!
//! 1. [`route`]: sigmoid scores; the top `k` experts by `score + bias` (ties
//!    to the lower expert); their raw scores normalised to sum to one and
//!    multiplied by the route scale. The routing tensor is F32
//!    `[tokens, 2 * k]`: `k` expert ids ascending (exact small integers),
//!    then `k` weights.
//! 2. [`experts`]: for each routed pair,
//!    `act = softcap(gate . x) * sigmoid(gate . x) * softcap(up . x)` over
//!    the interleaved gate and up rows, `partial = down . act`, and
//!    `y[t] = sum_k w_k * partial[t, k]` in slot order.
//!
//! Activations and every sum are F32. Each pair's dot products are summed in
//! the same order whatever else shares the call, so a row's result does not
//! depend on the batch.
//!
//! The CPU functions are the reference the CUDA and Metal kernels are checked
//! against (in tests and at model load) and run the same arithmetic on other
//! devices. `moe_gelu` and `moe_relu2` are the siblings for other routings
//! and weight forms; none changes another.

use candle_core::{DType, Device, Result, Tensor};

/// Most experts the routing kernel scans.
pub const MAX_EXPERTS: usize = 512;
/// Most experts a token may route to.
pub const MAX_TOP_K: usize = 16;

/// How a projection's weights are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedFormat {
    Nvfp4,
    Mxfp8,
}

impl PackedFormat {
    /// Values one scale covers.
    pub fn block(self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            Self::Mxfp8 => 32,
        }
    }

    /// Stored bytes of a row of `cols` values.
    pub fn code_bytes(self, cols: usize) -> usize {
        match self {
            Self::Nvfp4 => cols / 2,
            Self::Mxfp8 => cols,
        }
    }
}

/// One projection of every expert, packed: `codes` U8
/// `[experts, rows, code_bytes]`, `scales` U8 `[experts, rows, cols / block]`
/// and `factors` F32 `[experts]` (all ones for MXFP8).
#[derive(Debug, Clone)]
pub struct PackedWeights {
    pub format: PackedFormat,
    pub codes: Tensor,
    pub scales: Tensor,
    pub factors: Tensor,
}

impl PackedWeights {
    /// `(experts, rows, cols)`.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let (experts, rows, bytes) = self.codes.dims3()?;
        let cols = match self.format {
            PackedFormat::Nvfp4 => bytes * 2,
            PackedFormat::Mxfp8 => bytes,
        };
        Ok((experts, rows, cols))
    }

    /// Whether the tensors agree with each other and the format.
    pub fn validate(&self) -> Result<()> {
        let (experts, rows, cols) = self.dims3()?;
        if self.codes.dtype() != DType::U8
            || self.scales.dtype() != DType::U8
            || self.factors.dtype() != DType::F32
            || cols == 0
            || !cols.is_multiple_of(self.format.block())
            || self.scales.dims() != [experts, rows, cols / self.format.block()]
            || self.factors.dims() != [experts]
        {
            candle_core::bail!(
                "packed expert weights do not agree: {:?} codes {:?}, scales {:?}, factors {:?}",
                self.format,
                self.codes.dims(),
                self.scales.dims(),
                self.factors.dims()
            )
        }
        Ok(())
    }

    /// Bytes resident on the device.
    pub fn resident_bytes(&self) -> usize {
        self.codes.elem_count() + self.scales.elem_count() + self.factors.elem_count() * 4
    }
}

/// Whether the kernels take this geometry.
pub fn supported(hidden: usize, inter: usize, experts: usize, top_k: usize) -> bool {
    hidden > 0
        && inter > 0
        && hidden.is_multiple_of(32)
        && inter.is_multiple_of(32)
        && (1..=MAX_EXPERTS).contains(&experts)
        && (1..=MAX_TOP_K.min(experts)).contains(&top_k)
}

/// The value of an FP8 E4M3 byte (0x7F and 0xFF are NaN).
pub fn e4m3(byte: u8) -> f32 {
    let (e, m) = (i32::from((byte >> 3) & 15), f32::from(byte & 7));
    let value = if e == 0 {
        m * 2f32.powi(-9)
    } else if e == 15 && m == 7.0 {
        f32::NAN
    } else {
        (1.0 + m * 0.125) * 2f32.powi(e - 7)
    };
    if byte & 128 != 0 {
        -value
    } else {
        value
    }
}

/// The value of a 4-bit E2M1 code.
pub fn e2m1(code: u8) -> f32 {
    let (e, m) = (u32::from((code >> 1) & 3), f32::from(code & 1));
    let value = if e == 0 {
        0.5 * m
    } else {
        (1.0 + 0.5 * m) * (1u32 << (e - 1)) as f32
    };
    if code & 8 != 0 {
        -value
    } else {
        value
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn softcap(v: f32, cap: f32) -> f32 {
    (v / cap).tanh() * cap
}

/// One token's `(ids, weights)`, as the routing kernel computes them.
pub fn route_row(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    route_scale: f32,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids: Vec<u32> = Vec::with_capacity(top_k);
    let mut picked: Vec<f32> = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        let mut best: Option<(f32, usize, f32)> = None;
        for (expert, logit) in logits.iter().enumerate() {
            if ids.contains(&(expert as u32)) {
                continue;
            }
            let score = sigmoid(*logit);
            let key = score + bias[expert];
            if best.is_none_or(|(best_key, _, _)| key > best_key) {
                best = Some((key, expert, score));
            }
        }
        let (_, expert, score) = best.expect("top_k is at most the expert count");
        let at = ids.partition_point(|id| (*id as usize) < expert);
        ids.insert(at, expert as u32);
        picked.insert(at, score);
    }
    let total = picked.iter().sum::<f32>() + 1e-20;
    let weights = picked
        .iter()
        .map(|score| score / total * route_scale)
        .collect();
    (ids, weights)
}

/// The routing of `logits` (F32 `[tokens, experts]`) with the selection
/// `bias` (F32 `[experts]`): F32 `[tokens, 2 * top_k]`.
pub fn route(logits: &Tensor, bias: &Tensor, top_k: usize, route_scale: f32) -> Result<Tensor> {
    let (tokens, experts) = logits.dims2()?;
    if logits.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || bias.elem_count() != experts
        || !(1..=MAX_TOP_K.min(experts)).contains(&top_k)
        || experts > MAX_EXPERTS
    {
        candle_core::bail!(
            "sigmoid MoE routing takes F32 logits [tokens, <= {MAX_EXPERTS}] and an F32 bias per expert, got {:?} {:?} and {:?} for top {top_k}",
            logits.dims(),
            logits.dtype(),
            bias.dims()
        )
    }
    match logits.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::route(
            &logits.contiguous()?,
            &bias.contiguous()?,
            tokens,
            experts,
            top_k,
            route_scale,
        ),
        #[cfg(feature = "metal")]
        Device::Metal(_) => metal_impl::route(
            &logits.contiguous()?,
            &bias.contiguous()?,
            tokens,
            experts,
            top_k,
            route_scale,
        ),
        device => {
            let logits = logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let bias = bias
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let mut out = Vec::with_capacity(tokens * 2 * top_k);
            for row in &logits {
                let (ids, weights) = route_row(row, &bias, top_k, route_scale);
                out.extend(ids.iter().map(|&id| id as f32));
                out.extend(weights);
            }
            Tensor::from_vec(out, (tokens, 2 * top_k), &Device::Cpu)?.to_device(device)
        }
    }
}

/// `row . x` over packed bytes, summed as the kernels sum it: each block's
/// code sum, times the block scale, accumulated in block order.
pub fn row_dot(format: PackedFormat, codes: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    let mut total = 0f32;
    match format {
        PackedFormat::Nvfp4 => {
            for (block, scale) in scales.iter().enumerate() {
                let (q, xb) = (&codes[block * 8..][..8], &x[block * 16..][..16]);
                let mut dot = 0f32;
                for (i, byte) in q.iter().enumerate() {
                    dot = e2m1(byte & 15).mul_add(xb[2 * i], dot);
                    dot = e2m1(byte >> 4).mul_add(xb[2 * i + 1], dot);
                }
                total = dot.mul_add(e4m3(*scale), total);
            }
        }
        PackedFormat::Mxfp8 => {
            for (block, scale) in scales.iter().enumerate() {
                let (q, xb) = (&codes[block * 32..][..32], &x[block * 32..][..32]);
                let mut dot = 0f32;
                for (byte, x) in q.iter().zip(xb) {
                    dot = e4m3(*byte).mul_add(*x, dot);
                }
                total = dot.mul_add((f32::from(*scale) - 127.0).exp2(), total);
            }
        }
    }
    total
}

struct HostWeights {
    format: PackedFormat,
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    factors: Vec<f32>,
}

impl HostWeights {
    fn of(weights: &PackedWeights) -> Result<Self> {
        let (_, rows, cols) = weights.dims3()?;
        let bytes = |tensor: &Tensor| -> Result<Vec<u8>> {
            tensor
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<u8>()
        };
        Ok(Self {
            format: weights.format,
            rows,
            cols,
            codes: bytes(&weights.codes)?,
            scales: bytes(&weights.scales)?,
            factors: weights.factors.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
        })
    }

    fn dot(&self, expert: usize, row: usize, x: &[f32]) -> f32 {
        let code_bytes = self.format.code_bytes(self.cols);
        let blocks = self.cols / self.format.block();
        let at = expert * self.rows + row;
        row_dot(
            self.format,
            &self.codes[at * code_bytes..][..code_bytes],
            &self.scales[at * blocks..][..blocks],
            x,
        ) * self.factors[expert]
    }
}

/// `y[t] = sum_k w_k * down_k(act(gate_k . x_t, up_k . x_t))` for `x` F32
/// `[tokens, hidden]`, `routing` from [`route`], `gate_up`
/// `[experts, 2 * inter, hidden]` (gate and up rows interleaved) and `down`
/// `[experts, hidden, inter]`. Returns F32 `[tokens, hidden]`.
pub fn experts(
    x: &Tensor,
    routing: &Tensor,
    gate_up: &PackedWeights,
    down: &PackedWeights,
    gate_cap: f32,
    up_cap: f32,
) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts_count, rows, gate_hidden) = gate_up.dims3()?;
    let inter = rows / 2;
    let (routing_tokens, routing_width) = routing.dims2()?;
    let top_k = routing_width / 2;
    gate_up.validate()?;
    down.validate()?;
    if x.dtype() != DType::F32
        || routing.dtype() != DType::F32
        || rows != 2 * inter
        || gate_hidden != hidden
        || down.dims3()? != (experts_count, hidden, inter)
        || routing_tokens != tokens
        || routing_width != 2 * top_k
        || !supported(hidden, inter, experts_count, top_k)
    {
        candle_core::bail!(
            "packed MoE operands do not agree: x {:?}, routing {:?}, gate_up {:?}, down {:?}",
            x.dims(),
            routing.dims(),
            gate_up.codes.dims(),
            down.codes.dims()
        )
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::experts(
            &x.contiguous()?,
            &routing.contiguous()?,
            gate_up,
            down,
            top_k,
            gate_cap,
            up_cap,
        ),
        #[cfg(feature = "metal")]
        Device::Metal(_) => metal_impl::experts(
            &x.contiguous()?,
            &routing.contiguous()?,
            gate_up,
            down,
            top_k,
            gate_cap,
            up_cap,
        ),
        device => {
            let xs = x.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let routes = routing.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let (gate_up, down) = (HostWeights::of(gate_up)?, HostWeights::of(down)?);
            let mut out = vec![0f32; tokens * hidden];
            let mut act = vec![0f32; inter];
            for (t, route) in routes.iter().enumerate() {
                let xt = &xs[t * hidden..][..hidden];
                let row = &mut out[t * hidden..][..hidden];
                let mut partials = Vec::with_capacity(top_k);
                for slot in 0..top_k {
                    let id = route[slot];
                    if !(id >= 0.0 && (id as usize) < experts_count) {
                        partials.push(vec![0f32; hidden]);
                        continue;
                    }
                    let expert = id as usize;
                    for (i, a) in act.iter_mut().enumerate() {
                        let gate = gate_up.dot(expert, 2 * i, xt);
                        let up = gate_up.dot(expert, 2 * i + 1, xt);
                        *a = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap);
                    }
                    partials.push((0..hidden).map(|h| down.dot(expert, h, &act)).collect());
                }
                for (slot, partial) in partials.iter().enumerate() {
                    let weight = route[top_k + slot];
                    for (y, p) in row.iter_mut().zip(partial) {
                        *y = weight.mul_add(*p, *y);
                    }
                }
            }
            Tensor::from_vec(out, (tokens, hidden), &Device::Cpu)?.to_device(device)
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Shape, Storage, Tensor};

    use super::{PackedFormat, PackedWeights};

    const MODULE: &str = "izwi_moe_packed";
    const THREADS: u32 = 128;

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        tensor: &Tensor,
        name: &str,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("packed MoE {name} must be a CUDA tensor")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("packed MoE {name} must be contiguous")
        };
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
            .map_err(|_| candle_core::Error::Msg(format!("packed MoE {name} {value} exceeds i32")))
    }

    fn u32_grid(value: usize, name: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("packed MoE {name} grid {value} exceeds u32"))
        })
    }

    fn config(grid: (u32, u32)) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn route(
        logits: &Tensor,
        bias: &Tensor,
        tokens: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
    ) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (ls, _) = logits.storage_and_layout();
        let (bs, _) = bias.storage_and_layout();
        let lv = view::<f32>(&ls, logits, "logits")?;
        let bv = view::<f32>(&bs, bias, "bias")?;
        // SAFETY: the routing kernel writes every token's ids and weights.
        let out = unsafe { device.alloc::<f32>(tokens * 2 * top_k)? };
        let ptx = super::super::cuda::cuda_ptx_moe_packed();
        let function = device.get_or_load_custom_func("izwi_mpk_route", MODULE, ptx)?;
        let mut builder = function.builder();
        builder.arg(&lv);
        builder.arg(&bv);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(tokens, "tokens")?,
            i32_arg(experts, "experts")?,
            i32_arg(top_k, "top_k")?,
            route_scale
        );
        let launch = config((u32_grid(tokens.div_ceil(THREADS as usize), "tokens")?, 1));
        // SAFETY: argument order matches `izwi_mpk_route`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, 2 * top_k))))
    }

    pub(super) fn experts(
        x: &Tensor,
        routing: &Tensor,
        gate_up: &PackedWeights,
        down: &PackedWeights,
        slots: usize,
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let (tokens, hidden) = x.dims2()?;
        let (experts, rows, _) = gate_up.dims3()?;
        let inter = rows / 2;
        let pairs = tokens * slots;
        let gate_up_tensors = [
            gate_up.codes.contiguous()?,
            gate_up.scales.contiguous()?,
            gate_up.factors.contiguous()?,
        ];
        let down_tensors = [
            down.codes.contiguous()?,
            down.scales.contiguous()?,
            down.factors.contiguous()?,
        ];
        let (xs, _) = x.storage_and_layout();
        let (rs, _) = routing.storage_and_layout();
        let xv = view::<f32>(&xs, x, "x")?;
        let rv = view::<f32>(&rs, routing, "routing")?;
        let ptx = super::super::cuda::cuda_ptx_moe_packed();
        let (slots_i, hidden_i, inter_i, experts_i) = (
            i32_arg(slots, "slots")?,
            i32_arg(hidden, "hidden")?,
            i32_arg(inter, "inter")?,
            i32_arg(experts, "experts")?,
        );
        // SAFETY: the up kernel writes every routed pair's activations, the
        // down kernel every pair's row, the combine every (token, channel).
        let act = unsafe { device.alloc::<f32>(pairs * inter)? };
        let partial = unsafe { device.alloc::<f32>(pairs * hidden)? };
        let out = unsafe { device.alloc::<f32>(tokens * hidden)? };
        let pairs_grid = u32_grid(pairs, "pairs")?;

        {
            let name = match gate_up.format {
                PackedFormat::Nvfp4 => "izwi_mpk_up_nvfp4",
                PackedFormat::Mxfp8 => "izwi_mpk_up_mxfp8",
            };
            let (cs, _) = gate_up_tensors[0].storage_and_layout();
            let (ss, _) = gate_up_tensors[1].storage_and_layout();
            let (fs, _) = gate_up_tensors[2].storage_and_layout();
            let cv = view::<u8>(&cs, &gate_up_tensors[0], "gate/up codes")?;
            let sv = view::<u8>(&ss, &gate_up_tensors[1], "gate/up scales")?;
            let fv = view::<f32>(&fs, &gate_up_tensors[2], "gate/up factors")?;
            let function = device.get_or_load_custom_func(name, MODULE, ptx)?;
            let mut builder = function.builder();
            builder.arg(&xv);
            builder.arg(&rv);
            builder.arg(&cv);
            builder.arg(&sv);
            builder.arg(&fv);
            builder.arg(&act);
            candle_core::builder_arg!(
                builder, slots_i, hidden_i, inter_i, experts_i, gate_cap, up_cap
            );
            let launch = config((
                pairs_grid,
                u32_grid(inter.div_ceil(THREADS as usize), "inter")?,
            ));
            // SAFETY: argument order matches `izwi_mpk_up_*`.
            unsafe { builder.launch(launch) }.w()?;
        }
        {
            let name = match down.format {
                PackedFormat::Nvfp4 => "izwi_mpk_down_nvfp4",
                PackedFormat::Mxfp8 => "izwi_mpk_down_mxfp8",
            };
            let (cs, _) = down_tensors[0].storage_and_layout();
            let (ss, _) = down_tensors[1].storage_and_layout();
            let (fs, _) = down_tensors[2].storage_and_layout();
            let cv = view::<u8>(&cs, &down_tensors[0], "down codes")?;
            let sv = view::<u8>(&ss, &down_tensors[1], "down scales")?;
            let fv = view::<f32>(&fs, &down_tensors[2], "down factors")?;
            let function = device.get_or_load_custom_func(name, MODULE, ptx)?;
            let mut builder = function.builder();
            builder.arg(&act);
            builder.arg(&rv);
            builder.arg(&cv);
            builder.arg(&sv);
            builder.arg(&fv);
            builder.arg(&partial);
            candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i, experts_i);
            let launch = config((
                pairs_grid,
                u32_grid(hidden.div_ceil(THREADS as usize), "hidden")?,
            ));
            // SAFETY: argument order matches `izwi_mpk_down_*`.
            unsafe { builder.launch(launch) }.w()?;
        }
        let combine = device.get_or_load_custom_func("izwi_mpk_combine", MODULE, ptx)?;
        let mut builder = combine.builder();
        builder.arg(&partial);
        builder.arg(&rv);
        builder.arg(&out);
        candle_core::builder_arg!(builder, slots_i, hidden_i);
        let launch = config((
            u32_grid(tokens, "tokens")?,
            u32_grid(hidden.div_ceil(THREADS as usize), "hidden")?,
        ));
        // SAFETY: argument order matches `izwi_mpk_combine`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, hidden))))
    }
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use candle_core::op::BackpropOp;
    use candle_core::{bail, DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};
    use objc2_metal::{MTLCompileOptions, MTLSize};

    use super::{PackedFormat, PackedWeights};
    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    /// Threads in a group, `MPK_THREADS` in the source.
    const THREADS: usize = 128;

    // The kernels of `cuda/moe_packed.cu`, one for one: the same grids, one
    // thread per output value, the same block-wise sums with `fma` where the
    // CUDA source has `fmaf`.
    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define MPK_MAX_SLOTS 16
// Threads in a group; the launchers dispatch groups of this width.
#define MPK_THREADS 128

// The value of a 4-bit E2M1 code: sign, two exponent bits, one mantissa bit.
constant float mpk_e2m1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// The value of an FP8 E4M3 byte, built as F32 bits: a normal code is
// (1 + m / 8) * 2^(e - 7), a code with a zero exponent is m * 2^-9.
inline float mpk_e4m3(uint b) {
    const uint e = (b >> 3) & 15u;
    const uint m = b & 7u;
    const float v = e == 0u ? float(m) * 0.001953125f
                            : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (b & 128u) != 0u ? -v : v;
}

// 2^(s - 127) for a scale byte, as F32 bits (s = 0 is the subnormal 2^-127).
inline float mpk_exp2_scale(uint s) {
    return as_type<float>(s == 0u ? 0x00400000u : s << 23);
}

inline float mpk_sigmoid(float v) { return 1.0f / (1.0f + precise::exp(-v)); }

inline float mpk_softcap(float v, float cap) { return precise::tanh(v / cap) * cap; }

// row . x for an NVFP4 row of `cols` values (cols % 16 == 0): per 16-value
// block, the code sum times the block scale; the caller applies the factor.
// Codes are read a word at a time: nibble j of a word is element j of its
// eight, so the sum runs in element order.
inline float mpk_nvfp4_dot(device const uchar* codes, device const uchar* scales,
                           device const float* x, uint cols) {
    device const uint* q = (device const uint*)codes;
    float total = 0.0f;
    for (uint block = 0; block < cols / 16; ++block) {
        device const float* xb = x + block * 16;
        float dot = 0.0f;
        for (uint word = 0; word < 2; ++word) {
            const uint w = q[block * 2 + word];
            for (uint j = 0; j < 8; ++j) {
                dot = fma(mpk_e2m1[(w >> (4 * j)) & 15u], xb[word * 8 + j], dot);
            }
        }
        total = fma(dot, mpk_e4m3(scales[block]), total);
    }
    return total;
}

// row . x for an MXFP8 row of `cols` values (cols % 32 == 0).
inline float mpk_mxfp8_dot(device const uchar* values, device const uchar* scales,
                           device const float* x, uint cols) {
    device const uint* q = (device const uint*)values;
    float total = 0.0f;
    for (uint block = 0; block < cols / 32; ++block) {
        device const float* xb = x + block * 32;
        float dot = 0.0f;
        for (uint word = 0; word < 8; ++word) {
            const uint w = q[block * 8 + word];
            for (uint j = 0; j < 4; ++j) {
                dot = fma(mpk_e4m3((w >> (8 * j)) & 255u), xb[word * 4 + j], dot);
            }
        }
        total = fma(dot, mpk_exp2_scale(scales[block]), total);
    }
    return total;
}

// dims: tokens, experts, slots. One thread per token: sigmoid scores; the
// `slots` experts with the highest score + bias (ties to the lower expert);
// their raw scores normalised to sum to one and multiplied by `route_scale`.
// Ids are written ascending. Grid (ceil(tokens / 128)), 128 threads;
// experts <= 512, slots <= 16.
kernel void mpk_route(
    device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
    device float* routing [[buffer(2)]], constant uint* dims [[buffer(3)]],
    constant float& route_scale [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint tokens = dims[0], experts = dims[1], slots = dims[2];
    const uint token = tg * MPK_THREADS + tid;
    if (token >= tokens) {
        return;
    }
    device const float* row = logits + (ulong)token * experts;
    int ids[MPK_MAX_SLOTS];
    float picked[MPK_MAX_SLOTS];
    for (uint s = 0; s < slots; ++s) {
        float best = -INFINITY;
        int best_id = -1;
        float best_score = 0.0f;
        for (uint e = 0; e < experts; ++e) {
            bool taken = false;
            for (uint p = 0; p < s; ++p) {
                taken = taken || ids[p] == int(e);
            }
            if (taken) {
                continue;
            }
            const float score = mpk_sigmoid(row[e]);
            const float key = score + bias[e];
            // Ascending scan with a strict comparison: the lower expert wins a tie.
            if (best_id < 0 || key > best) {
                best = key;
                best_id = int(e);
                best_score = score;
            }
        }
        // Insert ascending by expert id.
        uint at = s;
        while (at > 0 && ids[at - 1] > best_id) {
            ids[at] = ids[at - 1];
            picked[at] = picked[at - 1];
            --at;
        }
        ids[at] = best_id;
        picked[at] = best_score;
    }
    float total = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        total += picked[s];
    }
    total += 1e-20f;
    device float* out = routing + (ulong)token * 2 * slots;
    for (uint s = 0; s < slots; ++s) {
        out[s] = float(ids[s]);
        out[slots + s] = picked[s] / total * route_scale;
    }
}

// dims: slots, hidden, inter, experts; caps: gate, up.
// act[pair, i] = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
// with gate = row 2i . x and up = row 2i + 1 . x (stored rows interleave gate
// and up). Grid (pairs, ceil(inter / 128)), 128 threads.
#define MPK_UP(NAME, DOT, CODE_BYTES, BLOCK)                                                     \
    kernel void NAME(                                                                            \
        device const float* x [[buffer(0)]], device const float* routing [[buffer(1)]],         \
        device const uchar* codes [[buffer(2)]], device const uchar* scales [[buffer(3)]],      \
        device const float* factors [[buffer(4)]], device float* act [[buffer(5)]],             \
        constant uint* dims [[buffer(6)]], constant float* caps [[buffer(7)]],                  \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {  \
        const uint slots = dims[0], hidden = dims[1], inter = dims[2], experts = dims[3];       \
        const uint i = tg.y * MPK_THREADS + tid;                                                 \
        if (i >= inter) {                                                                        \
            return;                                                                              \
        }                                                                                        \
        const uint pair = tg.x;                                                                  \
        const uint token = pair / slots;                                                         \
        const float id = routing[(ulong)token * 2 * slots + pair % slots];                      \
        device float* out = act + (ulong)pair * inter + i;                                      \
        if (!(id >= 0.0f) || uint(id) >= experts) {                                              \
            *out = 0.0f;                                                                         \
            return;                                                                              \
        }                                                                                        \
        const uint expert = uint(id);                                                            \
        device const float* xt = x + (ulong)token * hidden;                                     \
        const ulong code_row = (ulong)(CODE_BYTES);                                              \
        const ulong scale_row = (ulong)(hidden / (BLOCK));                                       \
        const ulong first = (ulong)expert * 2 * inter + 2 * i;                                   \
        const float factor = factors[expert];                                                    \
        const float gate =                                                                       \
            DOT(codes + first * code_row, scales + first * scale_row, xt, hidden) * factor;     \
        const float up = DOT(codes + (first + 1) * code_row, scales + (first + 1) * scale_row,  \
                             xt, hidden) *                                                       \
                         factor;                                                                 \
        *out = mpk_softcap(gate, caps[0]) * mpk_sigmoid(gate) * mpk_softcap(up, caps[1]);       \
    }

MPK_UP(mpk_up_nvfp4, mpk_nvfp4_dot, hidden / 2, 16)
MPK_UP(mpk_up_mxfp8, mpk_mxfp8_dot, hidden, 32)

// dims: slots, hidden, inter, experts. partial[pair, h] = down row h . act[pair].
// Grid (pairs, ceil(hidden / 128)), 128 threads.
#define MPK_DOWN(NAME, DOT, CODE_BYTES, BLOCK)                                                   \
    kernel void NAME(                                                                            \
        device const float* act [[buffer(0)]], device const float* routing [[buffer(1)]],       \
        device const uchar* codes [[buffer(2)]], device const uchar* scales [[buffer(3)]],      \
        device const float* factors [[buffer(4)]], device float* partial [[buffer(5)]],         \
        constant uint* dims [[buffer(6)]],                                                       \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {  \
        const uint slots = dims[0], hidden = dims[1], inter = dims[2], experts = dims[3];       \
        const uint h = tg.y * MPK_THREADS + tid;                                                 \
        if (h >= hidden) {                                                                       \
            return;                                                                              \
        }                                                                                        \
        const uint pair = tg.x;                                                                  \
        const uint token = pair / slots;                                                         \
        const float id = routing[(ulong)token * 2 * slots + pair % slots];                      \
        device float* out = partial + (ulong)pair * hidden + h;                                 \
        if (!(id >= 0.0f) || uint(id) >= experts) {                                              \
            *out = 0.0f;                                                                         \
            return;                                                                              \
        }                                                                                        \
        const uint expert = uint(id);                                                            \
        const ulong row = (ulong)expert * hidden + h;                                            \
        *out = DOT(codes + row * (ulong)(CODE_BYTES), scales + row * (ulong)(inter / (BLOCK)),  \
                   act + (ulong)pair * inter, inter) *                                           \
               factors[expert];                                                                  \
    }

MPK_DOWN(mpk_down_nvfp4, mpk_nvfp4_dot, inter / 2, 16)
MPK_DOWN(mpk_down_mxfp8, mpk_mxfp8_dot, inter, 32)

// dims: slots, hidden. y[token, h] = sum over slots, in slot order, of
// weight * partial. Grid (tokens, ceil(hidden / 128)), 128 threads.
kernel void mpk_combine(
    device const float* partial [[buffer(0)]], device const float* routing [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* dims [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint slots = dims[0], hidden = dims[1];
    const uint h = tg.y * MPK_THREADS + tid;
    if (h >= hidden) {
        return;
    }
    const uint token = tg.x;
    device const float* route = routing + (ulong)token * 2 * slots;
    float acc = 0.0f;
    for (uint s = 0; s < slots; ++s) {
        acc = fma(route[slots + s], partial[((ulong)token * slots + s) * hidden + h], acc);
    }
    y[(ulong)token * hidden + h] = acc;
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
        // Metal compiles with fast math unless told otherwise, which lets the
        // compiler reorder the sums and split the fused multiply-adds. These
        // kernels promise the reference's summation order, so it is off.
        let options = MTLCompileOptions::new();
        #[allow(deprecated)] // `mathMode` needs macOS 15; this covers every release we run on.
        options.setFastMathEnabled(false);
        let library = device
            .new_library_with_source(SOURCE, Some(&options))
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
                candle_core::Error::Msg(format!("packed MoE dimension {value} exceeds u32"))
            })?;
        }
        Ok(out)
    }

    /// A contiguous Metal tensor's buffer and byte offset.
    struct Bound<'a> {
        guard: candle_core::StorageRef<'a>,
        offset: usize,
    }

    impl Bound<'_> {
        fn buffer(&self) -> Result<&Buffer> {
            match &*self.guard {
                Storage::Metal(storage) => Ok(storage.buffer()),
                _ => bail!("packed MoE Metal kernels need Metal tensors"),
            }
        }
    }

    fn byte_offset(tensor: &Tensor) -> usize {
        tensor.layout().start_offset() * tensor.dtype().size_in_bytes()
    }

    fn bind<'a>(tensor: &'a Tensor, name: &str) -> Result<Bound<'a>> {
        let (guard, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            bail!("packed MoE Metal input {name} must be contiguous")
        }
        let offset = byte_offset(tensor);
        if !offset.is_multiple_of(4) {
            bail!("packed MoE Metal input {name} must start on a 4-byte boundary")
        }
        Ok(Bound { guard, offset })
    }

    /// `tensor` contiguous and starting on a word boundary. The kernels read
    /// codes a word at a time, so a view that starts mid-word is copied (the
    /// weights a model loads start at zero and are never copied).
    fn word_aligned(tensor: &Tensor) -> Result<Tensor> {
        let tensor = tensor.contiguous()?;
        if byte_offset(&tensor).is_multiple_of(4) {
            Ok(tensor)
        } else {
            tensor.force_contiguous()
        }
    }

    fn metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
        match tensor.device() {
            candle_core::Device::Metal(device) => Ok(device.clone()),
            _ => bail!("packed MoE Metal kernels need Metal tensors"),
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

    pub(super) fn route(
        logits: &Tensor,
        bias: &Tensor,
        tokens: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
    ) -> Result<Tensor> {
        let device = metal_device(logits)?;
        let lb = bind(logits, "logits")?;
        let bb = bind(bias, "bias")?;
        let out = device.new_buffer(tokens * 2 * top_k, DType::F32, "mpk-routing")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mpk-route");
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mpk_route")?);
        encoder.set_input_buffer(0, Some(lb.buffer()?), lb.offset);
        encoder.set_input_buffer(1, Some(bb.buffer()?), bb.offset);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &dims([tokens, experts, top_k])?);
        encoder.set_bytes(4, &route_scale);
        encoder.dispatch_thread_groups(grid(tokens.div_ceil(THREADS), 1), grid(THREADS, 1));
        drop(encoder);
        drop((lb, bb));
        Ok(wrap(&device, out, Shape::from((tokens, 2 * top_k))))
    }

    pub(super) fn experts(
        x: &Tensor,
        routing: &Tensor,
        gate_up: &PackedWeights,
        down: &PackedWeights,
        slots: usize,
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<Tensor> {
        let device = metal_device(x)?;
        let (tokens, hidden) = x.dims2()?;
        let (experts, rows, _) = gate_up.dims3()?;
        let inter = rows / 2;
        let pairs = tokens * slots;
        let gate_up_tensors = [
            word_aligned(&gate_up.codes)?,
            gate_up.scales.contiguous()?,
            gate_up.factors.contiguous()?,
        ];
        let down_tensors = [
            word_aligned(&down.codes)?,
            down.scales.contiguous()?,
            down.factors.contiguous()?,
        ];
        let xb = bind(x, "x")?;
        let rb = bind(routing, "routing")?;
        let gcb = bind(&gate_up_tensors[0], "gate/up codes")?;
        let gsb = bind(&gate_up_tensors[1], "gate/up scales")?;
        let gfb = bind(&gate_up_tensors[2], "gate/up factors")?;
        let dcb = bind(&down_tensors[0], "down codes")?;
        let dsb = bind(&down_tensors[1], "down scales")?;
        let dfb = bind(&down_tensors[2], "down factors")?;
        let act = device.new_buffer(pairs * inter, DType::F32, "mpk-act")?;
        let partial = device.new_buffer(pairs * hidden, DType::F32, "mpk-partial")?;
        let out = device.new_buffer(tokens * hidden, DType::F32, "mpk-out")?;
        let geometry = dims([slots, hidden, inter, experts])?;
        // The grid's first axis is one group per pair.
        dims([pairs])?;
        let encoder = device.command_encoder()?;
        encoder.set_label("mpk-experts");

        let up = match gate_up.format {
            PackedFormat::Nvfp4 => "mpk_up_nvfp4",
            PackedFormat::Mxfp8 => "mpk_up_mxfp8",
        };
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), up)?);
        encoder.set_input_buffer(0, Some(xb.buffer()?), xb.offset);
        encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
        encoder.set_input_buffer(2, Some(gcb.buffer()?), gcb.offset);
        encoder.set_input_buffer(3, Some(gsb.buffer()?), gsb.offset);
        encoder.set_input_buffer(4, Some(gfb.buffer()?), gfb.offset);
        encoder.set_output_buffer(5, Some(&act), 0);
        encoder.set_bytes(6, &geometry);
        encoder.set_bytes(7, &[gate_cap, up_cap]);
        encoder.dispatch_thread_groups(grid(pairs, inter.div_ceil(THREADS)), grid(THREADS, 1));
        encoder.insert_memory_barrier();

        let down_name = match down.format {
            PackedFormat::Nvfp4 => "mpk_down_nvfp4",
            PackedFormat::Mxfp8 => "mpk_down_mxfp8",
        };
        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), down_name)?);
        encoder.set_input_buffer(0, Some(&act), 0);
        encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
        encoder.set_input_buffer(2, Some(dcb.buffer()?), dcb.offset);
        encoder.set_input_buffer(3, Some(dsb.buffer()?), dsb.offset);
        encoder.set_input_buffer(4, Some(dfb.buffer()?), dfb.offset);
        encoder.set_output_buffer(5, Some(&partial), 0);
        encoder.set_bytes(6, &geometry);
        encoder.dispatch_thread_groups(grid(pairs, hidden.div_ceil(THREADS)), grid(THREADS, 1));
        encoder.insert_memory_barrier();

        encoder.set_compute_pipeline_state(&pipeline(device.metal_device(), "mpk_combine")?);
        encoder.set_input_buffer(0, Some(&partial), 0);
        encoder.set_input_buffer(1, Some(rb.buffer()?), rb.offset);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &dims([slots, hidden])?);
        encoder.dispatch_thread_groups(grid(tokens, hidden.div_ceil(THREADS)), grid(THREADS, 1));
        drop(encoder);
        drop((xb, rb, gcb, gsb, gfb, dcb, dsb, dfb));
        Ok(wrap(&device, out, Shape::from((tokens, hidden))))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn noise_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 9) as u8
            })
            .collect()
    }

    /// Random packed weights with scales in a sane range (no NaN codes).
    pub(crate) fn random_weights(
        format: PackedFormat,
        experts: usize,
        rows: usize,
        cols: usize,
        seed: u32,
        device: &Device,
    ) -> PackedWeights {
        let code_bytes = format.code_bytes(cols);
        let mut codes = noise_bytes(experts * rows * code_bytes, seed);
        let mut scales = noise_bytes(experts * rows * cols / format.block(), seed ^ 0x55AA);
        match format {
            // E4M3 scales around 1 (exponent 5..=8), never the NaN code.
            PackedFormat::Nvfp4 => {
                for scale in &mut scales {
                    *scale = 0x28 + (*scale & 0x1F);
                }
            }
            PackedFormat::Mxfp8 => {
                for code in &mut codes {
                    // Keep values small and finite: exponent at most 8.
                    *code = (*code & 0x80) | ((*code & 0x7F) % 0x48);
                }
                for scale in &mut scales {
                    *scale = 123 + (*scale & 3);
                }
            }
        }
        let factors = match format {
            PackedFormat::Nvfp4 => (0..experts).map(|e| 0.05 + 0.01 * e as f32).collect(),
            PackedFormat::Mxfp8 => vec![1f32; experts],
        };
        PackedWeights {
            format,
            codes: Tensor::from_vec(codes, (experts, rows, code_bytes), device).unwrap(),
            scales: Tensor::from_vec(scales, (experts, rows, cols / format.block()), device)
                .unwrap(),
            factors: Tensor::from_vec(factors, experts, device).unwrap(),
        }
    }

    #[test]
    fn the_code_tables_match_the_formats() {
        let e2m1_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in e2m1_values.iter().enumerate() {
            assert_eq!(e2m1(code as u8), *value);
            assert_eq!(e2m1(code as u8 | 8), -*value);
        }
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0xB8), -1.0);
        assert!(e4m3(0x7F).is_nan());
    }

    #[test]
    fn routing_picks_by_score_plus_bias_and_weighs_by_score() {
        let logits = [0.0f32, 2.0, -1.0, 1.0, 0.5, -3.0];
        let mut bias = [0.0f32; 6];
        let (ids, weights) = route_row(&logits, &bias, 2, 8.0);
        assert_eq!(ids, [1, 3]);
        let (a, b) = (sigmoid(2.0), sigmoid(1.0));
        assert!((weights[0] - a / (a + b) * 8.0).abs() < 1e-6);
        assert!((weights.iter().sum::<f32>() - 8.0).abs() < 1e-5);
        // A bias moves the choice; the weight is still the raw score's share.
        bias[5] = 10.0;
        let (ids, weights) = route_row(&logits, &bias, 2, 8.0);
        assert_eq!(ids, [1, 5]);
        let c = sigmoid(-3.0);
        assert!((weights[1] - c / (a + c) * 8.0).abs() < 1e-6);
        // Ties go to the lower expert.
        let (ids, _) = route_row(&[1.0, 1.0, 1.0], &[0.0; 3], 2, 1.0);
        assert_eq!(ids, [0, 1]);
    }

    /// The packed path against an F64 evaluation over the decoded weights,
    /// for both formats, and each row alone gives the same bits.
    #[test]
    fn packed_experts_match_a_dense_evaluation_on_the_cpu() {
        let (hidden, inter, experts_count, top_k, tokens) = (64usize, 32usize, 6usize, 2usize, 5);
        let (gate_cap, up_cap) = (4.0f32, 25.0f32);
        let device = Device::Cpu;
        for format in [PackedFormat::Nvfp4, PackedFormat::Mxfp8] {
            let gate_up = random_weights(format, experts_count, 2 * inter, hidden, 11, &device);
            let down = random_weights(format, experts_count, hidden, inter, 23, &device);
            let x = noise(tokens * hidden, 5);
            let logits = noise(tokens * experts_count, 9);
            let bias = noise(experts_count, 13);
            let routing = route(
                &Tensor::from_vec(logits, (tokens, experts_count), &device).unwrap(),
                &Tensor::from_vec(bias, experts_count, &device).unwrap(),
                top_k,
                8.0,
            )
            .unwrap();
            let x_tensor = Tensor::from_vec(x.clone(), (tokens, hidden), &device).unwrap();
            let got = experts(&x_tensor, &routing, &gate_up, &down, gate_cap, up_cap)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();

            let dense = |weights: &PackedWeights, expert: usize, row: usize| -> Vec<f64> {
                let host = HostWeights::of(weights).unwrap();
                let code_bytes = format.code_bytes(host.cols);
                let blocks = host.cols / format.block();
                let at = expert * host.rows + row;
                let codes = &host.codes[at * code_bytes..][..code_bytes];
                let scales = &host.scales[at * blocks..][..blocks];
                (0..host.cols)
                    .map(|col| {
                        let value = match format {
                            PackedFormat::Nvfp4 => {
                                let byte = codes[col / 2];
                                let code = if col.is_multiple_of(2) {
                                    byte & 15
                                } else {
                                    byte >> 4
                                };
                                f64::from(e2m1(code)) * f64::from(e4m3(scales[col / 16]))
                            }
                            PackedFormat::Mxfp8 => {
                                f64::from(e4m3(codes[col]))
                                    * 2f64.powi(i32::from(scales[col / 32]) - 127)
                            }
                        };
                        value * f64::from(host.factors[expert])
                    })
                    .collect()
            };
            let routes = routing.to_vec2::<f32>().unwrap();
            for t in 0..tokens {
                let xt = &x[t * hidden..][..hidden];
                let mut want = vec![0f64; hidden];
                for slot in 0..top_k {
                    let expert = routes[t][slot] as usize;
                    let weight = f64::from(routes[t][top_k + slot]);
                    let dot =
                        |row: &[f64], v: &[f64]| row.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
                    let xt64 = xt.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
                    let act = (0..inter)
                        .map(|i| {
                            let gate = dot(&dense(&gate_up, expert, 2 * i), &xt64);
                            let up = dot(&dense(&gate_up, expert, 2 * i + 1), &xt64);
                            (gate / 4.0).tanh() * 4.0 / (1.0 + (-gate).exp())
                                * ((up / 25.0).tanh() * 25.0)
                        })
                        .collect::<Vec<_>>();
                    for (h, want) in want.iter_mut().enumerate() {
                        *want += weight * dot(&dense(&down, expert, h), &act);
                    }
                }
                let scale = want.iter().fold(1f64, |m, v| m.max(v.abs()));
                for (got, want) in got[t].iter().zip(&want) {
                    assert!(
                        (f64::from(*got) - want).abs() <= 2e-5 * scale,
                        "{format:?} token {t}: {got} vs {want}"
                    );
                }
                assert!(
                    want.iter().any(|v| v.abs() > 1e-3),
                    "{format:?}: a zero output"
                );
                let alone = experts(
                    &x_tensor.narrow(0, t, 1).unwrap(),
                    &routing.narrow(0, t, 1).unwrap(),
                    &gate_up,
                    &down,
                    gate_cap,
                    up_cap,
                )
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
                assert_eq!(
                    alone[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    got[t].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn mismatched_operands_are_refused() {
        let device = Device::Cpu;
        let gate_up = random_weights(PackedFormat::Nvfp4, 4, 64, 64, 1, &device);
        let down = random_weights(PackedFormat::Nvfp4, 4, 64, 32, 2, &device);
        let x = Tensor::zeros((2, 64), DType::F32, &device).unwrap();
        let routing = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        assert!(experts(&x, &routing, &gate_up, &down, 4.0, 25.0).is_ok());
        // A down projection for another expert width.
        let wrong = random_weights(PackedFormat::Nvfp4, 4, 64, 64, 2, &device);
        assert!(experts(&x, &routing, &gate_up, &wrong, 4.0, 25.0).is_err());
        // Scales that do not cover the columns.
        let mut broken = gate_up.clone();
        broken.scales = Tensor::zeros((4, 64, 3), DType::U8, &device).unwrap();
        assert!(experts(&x, &routing, &broken, &down, 4.0, 25.0).is_err());
        assert!(!supported(48, 32, 4, 2));
        assert!(!supported(64, 32, 4, 5));
    }
}

/// On a CUDA device: the kernels against the CPU reference.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::tests::{noise, random_weights};
    use super::*;

    #[test]
    fn cuda_packed_experts_match_the_cpu_reference() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let (hidden, inter, experts_count, top_k) = (2304usize, 1152usize, 128usize, 8usize);
        let cpu = Device::Cpu;
        for format in [PackedFormat::Nvfp4, PackedFormat::Mxfp8] {
            for tokens in [1usize, 3, 40] {
                let gate_up_cpu =
                    random_weights(format, experts_count, 2 * inter, hidden, 11, &cpu);
                let down_cpu = random_weights(format, experts_count, hidden, inter, 23, &cpu);
                let to = |weights: &PackedWeights| PackedWeights {
                    format,
                    codes: weights.codes.to_device(&device).unwrap(),
                    scales: weights.scales.to_device(&device).unwrap(),
                    factors: weights.factors.to_device(&device).unwrap(),
                };
                let x =
                    Tensor::from_vec(noise(tokens * hidden, 5), (tokens, hidden), &cpu).unwrap();
                let logits = Tensor::from_vec(
                    noise(tokens * experts_count, 9),
                    (tokens, experts_count),
                    &cpu,
                )
                .unwrap();
                let bias = Tensor::from_vec(noise(experts_count, 13), experts_count, &cpu).unwrap();
                let want_routing = route(&logits, &bias, top_k, 8.0).unwrap();
                let routing = route(
                    &logits.to_device(&device).unwrap(),
                    &bias.to_device(&device).unwrap(),
                    top_k,
                    8.0,
                )
                .unwrap();
                let got_routing = routing.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
                for (got, want) in got_routing
                    .iter()
                    .zip(want_routing.to_vec2::<f32>().unwrap())
                {
                    assert_eq!(got[..top_k], want[..top_k], "{format:?} expert ids");
                    for (g, w) in got[top_k..].iter().zip(&want[top_k..]) {
                        assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0));
                    }
                }
                let want = experts(&x, &want_routing, &gate_up_cpu, &down_cpu, 4.0, 25.0)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let got = experts(
                    &x.to_device(&device).unwrap(),
                    &routing,
                    &to(&gate_up_cpu),
                    &to(&down_cpu),
                    4.0,
                    25.0,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
                let worst = got.iter().zip(&want).fold(0f32, |worst, (g, w)| {
                    let error = (g - w).abs();
                    if error.is_finite() {
                        worst.max(error)
                    } else {
                        f32::INFINITY
                    }
                });
                assert!(
                    worst <= 1e-4 * scale,
                    "{format:?} at {tokens} tokens: error {worst} at scale {scale}"
                );
            }
        }
    }
}

/// On a Metal device: the kernels against the CPU reference.
#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::tests::{noise, random_weights};
    use super::*;

    fn metal(test: &str) -> Option<(Device, impl Sized)> {
        crate::backends::metal_test_device_or_skip(test)
    }

    fn on(weights: &PackedWeights, device: &Device) -> PackedWeights {
        PackedWeights {
            format: weights.format,
            codes: weights.codes.to_device(device).unwrap(),
            scales: weights.scales.to_device(device).unwrap(),
            factors: weights.factors.to_device(device).unwrap(),
        }
    }

    fn host(tensor: &Tensor) -> Vec<f32> {
        tensor
            .flatten_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The largest `|got - want|` over the largest `|want|` (at least one).
    fn worst_error(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
        got.iter().zip(want).fold(0f32, |worst, (g, w)| {
            let error = (g - w).abs() / scale;
            if error.is_finite() {
                worst.max(error)
            } else {
                f32::INFINITY
            }
        })
    }

    /// `(hidden, inter, experts, top_k)`: the smallest the kernels take, widths
    /// that are not multiples of the thread group, and the published layer at
    /// a quarter of its widths.
    const GEOMETRIES: [(usize, usize, usize, usize); 4] = [
        (32, 32, 8, 2),
        (160, 96, 32, 4),
        (96, 288, 64, 6),
        (576, 288, 128, 8),
    ];

    #[test]
    fn metal_routing_matches_the_cpu_reference() {
        let Some((device, _lease)) = metal("packed MoE routing") else {
            return;
        };
        let cpu = Device::Cpu;
        for (experts_count, top_k) in [(8usize, 2usize), (128, 8), (512, 16), (5, 5)] {
            for tokens in [1usize, 3, 17, 300] {
                let logits = noise(tokens * experts_count, 9)
                    .into_iter()
                    .map(|v| v * 4.0)
                    .collect::<Vec<_>>();
                let logits = Tensor::from_vec(logits, (tokens, experts_count), &cpu).unwrap();
                let bias = Tensor::from_vec(noise(experts_count, 13), experts_count, &cpu).unwrap();
                let want = route(&logits, &bias, top_k, 8.0)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                let got = route(
                    &logits.to_device(&device).unwrap(),
                    &bias.to_device(&device).unwrap(),
                    top_k,
                    8.0,
                )
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
                for (t, (got, want)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(got[..top_k], want[..top_k], "token {t} expert ids");
                    for (g, w) in got[top_k..].iter().zip(&want[top_k..]) {
                        assert!(
                            (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                            "token {t} of {tokens}, {experts_count} experts: {g} vs {w}"
                        );
                    }
                }
            }
        }
        // Equal keys: the lower experts win, and the ids come out ascending
        // even when a later expert is picked first.
        let ties = Tensor::zeros((2, 6), DType::F32, &device).unwrap();
        let bias = Tensor::from_vec(vec![0f32, 0.0, 0.0, 0.5, 0.0, 0.0], 6, &device).unwrap();
        let got = route(&ties, &bias, 3, 1.0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for row in &got {
            assert_eq!(row[..3], [0.0, 1.0, 3.0]);
        }
    }

    #[test]
    fn metal_packed_experts_match_the_cpu_reference() {
        let Some((device, _lease)) = metal("packed MoE experts") else {
            return;
        };
        let cpu = Device::Cpu;
        for format in [PackedFormat::Nvfp4, PackedFormat::Mxfp8] {
            let mut worst = 0f32;
            for (hidden, inter, experts_count, top_k) in GEOMETRIES {
                let gate_up_cpu =
                    random_weights(format, experts_count, 2 * inter, hidden, 11, &cpu);
                let down_cpu = random_weights(format, experts_count, hidden, inter, 23, &cpu);
                let (gate_up, down) = (on(&gate_up_cpu, &device), on(&down_cpu, &device));
                for tokens in [1usize, 3, 17] {
                    let x = Tensor::from_vec(noise(tokens * hidden, 5), (tokens, hidden), &cpu)
                        .unwrap();
                    let logits = Tensor::from_vec(
                        noise(tokens * experts_count, 9),
                        (tokens, experts_count),
                        &cpu,
                    )
                    .unwrap();
                    let bias =
                        Tensor::from_vec(noise(experts_count, 13), experts_count, &cpu).unwrap();
                    let want_routing = route(&logits, &bias, top_k, 8.0).unwrap();
                    let routing = route(
                        &logits.to_device(&device).unwrap(),
                        &bias.to_device(&device).unwrap(),
                        top_k,
                        8.0,
                    )
                    .unwrap();
                    let got_routing = routing.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
                    for (got, want) in got_routing
                        .iter()
                        .zip(want_routing.to_vec2::<f32>().unwrap())
                    {
                        assert_eq!(got[..top_k], want[..top_k], "{format:?} expert ids");
                        for (g, w) in got[top_k..].iter().zip(&want[top_k..]) {
                            assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0));
                        }
                    }
                    let want = host(
                        &experts(&x, &want_routing, &gate_up_cpu, &down_cpu, 4.0, 25.0).unwrap(),
                    );
                    let got = host(
                        &experts(
                            &x.to_device(&device).unwrap(),
                            &routing,
                            &gate_up,
                            &down,
                            4.0,
                            25.0,
                        )
                        .unwrap(),
                    );
                    assert!(want.iter().any(|v| v.abs() > 1e-3), "{format:?}: zeros");
                    let error = worst_error(&got, &want);
                    assert!(
                        error <= 1e-5,
                        "{format:?} {hidden}x{inter}, {experts_count} experts, top {top_k}, {tokens} tokens: relative error {error}"
                    );
                    worst = worst.max(error);
                }
            }
            eprintln!("metal packed experts {format:?}: worst relative error {worst:e}");
        }
    }

    /// The two projections may be stored differently, and a slot whose id is
    /// not an expert adds nothing.
    #[test]
    fn metal_packed_experts_take_mixed_formats_and_skip_ids_out_of_range() {
        let Some((device, _lease)) = metal("packed MoE mixed formats") else {
            return;
        };
        let cpu = Device::Cpu;
        let (hidden, inter, experts_count, top_k, tokens) = (96usize, 160usize, 12usize, 4usize, 5);
        let x = Tensor::from_vec(noise(tokens * hidden, 5), (tokens, hidden), &cpu).unwrap();
        let mut routes = Vec::new();
        for t in 0..tokens {
            let ids = match t {
                1 => [-1.0, 3.0, 12.0, 7.0],
                3 => [f32::NAN, 0.0, 11.0, 1e9],
                _ => [0.0, t as f32, 6.0, 11.0],
            };
            routes.extend(ids);
            routes.extend([0.5f32, 1.5, 2.0, 4.0]);
        }
        let routing = Tensor::from_vec(routes, (tokens, 2 * top_k), &cpu).unwrap();
        for (up_format, down_format) in [
            (PackedFormat::Nvfp4, PackedFormat::Mxfp8),
            (PackedFormat::Mxfp8, PackedFormat::Nvfp4),
        ] {
            let gate_up = random_weights(up_format, experts_count, 2 * inter, hidden, 11, &cpu);
            let down = random_weights(down_format, experts_count, hidden, inter, 23, &cpu);
            let want = host(&experts(&x, &routing, &gate_up, &down, 4.0, 25.0).unwrap());
            let got = host(
                &experts(
                    &x.to_device(&device).unwrap(),
                    &routing.to_device(&device).unwrap(),
                    &on(&gate_up, &device),
                    &on(&down, &device),
                    4.0,
                    25.0,
                )
                .unwrap(),
            );
            let error = worst_error(&got, &want);
            assert!(error <= 1e-5, "{up_format:?} up: relative error {error}");
        }
    }

    /// A token alone gives the bits it gives inside a larger batch.
    #[test]
    fn metal_packed_experts_do_not_depend_on_the_batch() {
        let Some((device, _lease)) = metal("packed MoE batch invariance") else {
            return;
        };
        let (hidden, inter, experts_count, top_k, tokens) = (576usize, 288usize, 128usize, 8, 17);
        for format in [PackedFormat::Nvfp4, PackedFormat::Mxfp8] {
            let gate_up = random_weights(format, experts_count, 2 * inter, hidden, 11, &device);
            let down = random_weights(format, experts_count, hidden, inter, 23, &device);
            let x = Tensor::from_vec(noise(tokens * hidden, 5), (tokens, hidden), &device).unwrap();
            let logits = Tensor::from_vec(
                noise(tokens * experts_count, 9),
                (tokens, experts_count),
                &device,
            )
            .unwrap();
            let bias = Tensor::from_vec(noise(experts_count, 13), experts_count, &device).unwrap();
            let routing = route(&logits, &bias, top_k, 8.0).unwrap();
            let all = experts(&x, &routing, &gate_up, &down, 4.0, 25.0).unwrap();
            let (routing_rows, rows) = (
                routing.to_vec2::<f32>().unwrap(),
                all.to_vec2::<f32>().unwrap(),
            );
            for t in 0..tokens {
                let alone_routing =
                    route(&logits.narrow(0, t, 1).unwrap(), &bias, top_k, 8.0).unwrap();
                assert_eq!(
                    bits(&host(&alone_routing)),
                    bits(&routing_rows[t]),
                    "{format:?} token {t} routing"
                );
                // The views start `t` rows into their buffers.
                let alone = experts(
                    &x.narrow(0, t, 1).unwrap(),
                    &routing.narrow(0, t, 1).unwrap(),
                    &gate_up,
                    &down,
                    4.0,
                    25.0,
                )
                .unwrap();
                assert_eq!(bits(&host(&alone)), bits(&rows[t]), "{format:?} token {t}");
            }
        }
    }

    /// The published layer at full size (128 experts, 2304 wide, 1152 inside,
    /// top 8): about half a gigabyte of packed weights per format in single
    /// buffers. Slow in a debug build, so it runs on request.
    #[test]
    #[ignore = "allocates the published layer; run with --ignored"]
    fn metal_packed_experts_run_the_published_geometry() {
        let Some((device, _lease)) = metal("packed MoE published geometry") else {
            return;
        };
        let cpu = Device::Cpu;
        let (hidden, inter, experts_count, top_k) = (2304usize, 1152usize, 128usize, 8usize);
        for format in [PackedFormat::Nvfp4, PackedFormat::Mxfp8] {
            let gate_up_cpu = random_weights(format, experts_count, 2 * inter, hidden, 11, &cpu);
            let down_cpu = random_weights(format, experts_count, hidden, inter, 23, &cpu);
            let (gate_up, down) = (on(&gate_up_cpu, &device), on(&down_cpu, &device));
            for tokens in [1usize, 16, 64] {
                let x =
                    Tensor::from_vec(noise(tokens * hidden, 5), (tokens, hidden), &device).unwrap();
                let logits = Tensor::from_vec(
                    noise(tokens * experts_count, 9),
                    (tokens, experts_count),
                    &device,
                )
                .unwrap();
                let bias =
                    Tensor::from_vec(noise(experts_count, 13), experts_count, &device).unwrap();
                let run = || {
                    let routing = route(&logits, &bias, top_k, 8.0).unwrap();
                    let y = experts(&x, &routing, &gate_up, &down, 4.0, 25.0).unwrap();
                    (routing, host(&y))
                };
                // The first run compiles the pipelines.
                let (routing, got) = run();
                let started = std::time::Instant::now();
                let runs = 5;
                for _ in 0..runs {
                    run();
                }
                let each = started.elapsed() / runs;
                eprintln!("metal packed experts {format:?}, {tokens} tokens: {each:?} a call");
                if tokens == 1 {
                    let want = host(
                        &experts(
                            &x.to_device(&cpu).unwrap(),
                            &routing.to_device(&cpu).unwrap(),
                            &gate_up_cpu,
                            &down_cpu,
                            4.0,
                            25.0,
                        )
                        .unwrap(),
                    );
                    let error = worst_error(&got, &want);
                    eprintln!("metal packed experts {format:?}: relative error {error:e}");
                    assert!(error <= 1e-5, "{format:?}: relative error {error}");
                }
            }
        }
    }
}

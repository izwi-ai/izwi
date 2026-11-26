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
//! The CPU functions are the reference the CUDA kernels are checked against
//! (in tests and at model load) and run the same arithmetic on other
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

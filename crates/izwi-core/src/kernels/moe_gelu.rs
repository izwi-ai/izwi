//! Device-routed sparse experts for softmax-routed MoE layers with BF16
//! experts and GELU-tanh gating (Gemma 4, as SOV-Transcribe-1 uses it).
//!
//! A layer runs as five launches with the expert ids on the device, so a
//! decode step never waits for the host inside the MoE block:
//!
//! 1. [`route`]: softmax over every expert in F32, the top `k` probabilities
//!    (ties to the lower expert), renormalized over the kept ones and
//!    multiplied by each expert's scale. The routing tensor is F32
//!    `[tokens, 2 * k]`: `k` expert ids (exact small integers), then `k`
//!    weights.
//! 2. [`experts`]: the routed pairs grouped by expert, then each expert's
//!    gate/up rows (`[2 * inter, hidden]`, gate first) and down rows
//!    (`[hidden, inter]`) read once for all of its pairs:
//!    `act = gelu_tanh(gate . x) * (up . x)`, `partial = down . act`, and
//!    `y[t] = sum_k w_k * partial[t, k]` in slot order.
//!
//! Weights stay BF16 as the checkpoint stores them; activations and every
//! sum are F32. Each pair's dot products are summed in the same order
//! whatever else shares the call, so a row's result does not depend on the
//! batch. Widths must be multiples of 8 (one 16-byte load per lane);
//! Gemma 4's 2816 and 704 are.
//!
//! The CPU functions are the reference the CUDA kernels are checked against
//! (in tests and at model load) and run the same arithmetic on other
//! devices, not a performance path. `kernels::moe_relu2` is the sibling for
//! sigmoid-routed relu² FP8 experts; neither changes the other.

use candle_core::{DType, Device, Result, Tensor};

/// Most experts the routing kernel holds (16 per lane, 32 lanes).
pub const MAX_EXPERTS: usize = 512;
/// Most experts a token may route to.
pub const MAX_TOP_K: usize = 16;

/// Whether the kernels take this geometry.
pub fn supported(hidden: usize, inter: usize, experts: usize, top_k: usize) -> bool {
    hidden > 0
        && inter > 0
        && hidden.is_multiple_of(8)
        && inter.is_multiple_of(8)
        && (1..=MAX_EXPERTS).contains(&experts)
        && (1..=MAX_TOP_K.min(experts)).contains(&top_k)
}

/// The routing of `logits` (F32 `[tokens, experts]`) with per-expert
/// `scale` (F32 `[experts]`): F32 `[tokens, 2 * top_k]`.
pub fn route(logits: &Tensor, scale: &Tensor, top_k: usize) -> Result<Tensor> {
    let (tokens, experts) = logits.dims2()?;
    if logits.dtype() != DType::F32
        || scale.dtype() != DType::F32
        || scale.elem_count() != experts
        || !(1..=MAX_TOP_K.min(experts)).contains(&top_k)
        || experts > MAX_EXPERTS
    {
        candle_core::bail!(
            "softmax MoE routing takes F32 logits [tokens, <= {MAX_EXPERTS}] and an F32 scale per expert, got {:?} {:?} and {:?} for top {top_k}",
            logits.dims(),
            logits.dtype(),
            scale.dims()
        )
    }
    match logits.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::route(
            &logits.contiguous()?,
            &scale.contiguous()?,
            tokens,
            experts,
            top_k,
        ),
        device => {
            let logits = logits.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let scale = scale
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let mut out = Vec::with_capacity(tokens * 2 * top_k);
            for row in &logits {
                let (ids, weights) = route_row(row, &scale, top_k);
                out.extend(ids.iter().map(|&id| id as f32));
                out.extend(weights);
            }
            Tensor::from_vec(out, (tokens, 2 * top_k), &Device::Cpu)?.to_device(device)
        }
    }
}

/// One token's `(ids, weights)`, as the routing kernel computes them.
pub fn route_row(logits: &[f32], scale: &[f32], top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs = logits.iter().map(|v| (v - best).exp()).collect::<Vec<_>>();
    let sum = probs.iter().sum::<f32>();
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    order.truncate(top_k);
    let kept = order.iter().map(|&e| probs[e] / sum).collect::<Vec<_>>();
    let total = kept.iter().sum::<f32>();
    let weights = order
        .iter()
        .zip(&kept)
        .map(|(&e, p)| p / total * scale[e])
        .collect();
    (order.into_iter().map(|e| e as u32).collect(), weights)
}

fn gelu_tanh(v: f32) -> f32 {
    let inner = 0.797_884_6_f32 * (v + 0.044715 * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

/// `y[t] = sum_k w_k * down_k(gelu_tanh(gate_k . x_t) * (up_k . x_t))` for
/// `x` F32 `[tokens, hidden]`, `routing` from [`route`], `gate_up`
/// `[experts, 2 * inter, hidden]` and `down` `[experts, hidden, inter]`
/// (BF16 on CUDA; BF16 or F32 for the CPU reference). Returns F32
/// `[tokens, hidden]`.
pub fn experts(x: &Tensor, routing: &Tensor, gate_up: &Tensor, down: &Tensor) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts, rows, gate_hidden) = gate_up.dims3()?;
    let inter = rows / 2;
    let (down_experts, down_hidden, down_inter) = down.dims3()?;
    let (routing_tokens, routing_width) = routing.dims2()?;
    let top_k = routing_width / 2;
    if x.dtype() != DType::F32
        || routing.dtype() != DType::F32
        || !weight_dtype_supported(gate_up)
        || down.dtype() != gate_up.dtype()
        || rows != 2 * inter
        || gate_hidden != hidden
        || (down_experts, down_hidden, down_inter) != (experts, hidden, inter)
        || routing_tokens != tokens
        || routing_width != 2 * top_k
        || !supported(hidden, inter, experts, top_k)
    {
        candle_core::bail!(
            "BF16 GELU MoE operands do not agree: x {:?} {:?}, routing {:?}, gate_up {:?} {:?}, down {:?} {:?}",
            x.dims(),
            x.dtype(),
            routing.dims(),
            gate_up.dims(),
            gate_up.dtype(),
            down.dims(),
            down.dtype()
        )
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::experts(
            &x.contiguous()?,
            &routing.contiguous()?,
            &gate_up.contiguous()?,
            &down.contiguous()?,
            top_k,
        ),
        device => {
            let x = x.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let routing = routing.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let gate_up = host_f32(gate_up)?;
            let down = host_f32(down)?;
            let mut out = vec![0f32; tokens * hidden];
            let mut act = vec![0f32; inter];
            for (t, (xt, route)) in x.iter().zip(&routing).enumerate() {
                let mut partials = Vec::with_capacity(top_k);
                for slot in 0..top_k {
                    let id = route[slot];
                    if !(id >= 0.0 && (id as usize) < experts) {
                        partials.push(None);
                        continue;
                    }
                    let e = id as usize;
                    let rows = &gate_up[e * 2 * inter * hidden..][..2 * inter * hidden];
                    for (n, a) in act.iter_mut().enumerate() {
                        let g = dot(&rows[n * hidden..][..hidden], xt);
                        let u = dot(&rows[(inter + n) * hidden..][..hidden], xt);
                        *a = gelu_tanh(g) * u;
                    }
                    let rows = &down[e * hidden * inter..][..hidden * inter];
                    partials.push(Some(
                        (0..hidden)
                            .map(|h| dot(&rows[h * inter..][..inter], &act))
                            .collect::<Vec<_>>(),
                    ));
                }
                let row = &mut out[t * hidden..][..hidden];
                for (slot, partial) in partials.iter().enumerate() {
                    if let Some(partial) = partial {
                        let weight = route[top_k + slot];
                        for (y, p) in row.iter_mut().zip(partial) {
                            *y = weight.mul_add(*p, *y);
                        }
                    }
                }
            }
            Tensor::from_vec(out, (tokens, hidden), &Device::Cpu)?.to_device(device)
        }
    }
}

/// A dot product summed as the kernels sum it: 32 lanes each over every
/// 256th group of 8 values, the lanes then added in butterfly order.
fn dot(weights: &[f32], x: &[f32]) -> f32 {
    let mut lanes = [0f32; 32];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        let mut k0 = lane * 8;
        while k0 < weights.len() {
            let mut acc = weights[k0] * x[k0];
            for i in 1..8 {
                acc = weights[k0 + i].mul_add(x[k0 + i], acc);
            }
            *sum += acc;
            k0 += 256;
        }
    }
    let mut width = 16;
    while width > 0 {
        let previous = lanes;
        for (lane, sum) in lanes.iter_mut().enumerate() {
            *sum = previous[lane] + previous[lane ^ width];
        }
        width >>= 1;
    }
    lanes[0]
}

fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
}

/// BF16 expert weights everywhere; F32 too for the CPU reference.
pub fn weight_dtype_supported(weights: &Tensor) -> bool {
    weights.dtype() == DType::BF16 || (!weights.device().is_cuda() && weights.dtype() == DType::F32)
}

/// Expert weights as E4M3 bytes with one F32 scale per output row:
/// `q` U8 `[experts, rows, cols]` and `scale` F32 `[experts, rows]`, a row's
/// value `e4m3(q) * scale`. Half the bytes of BF16, so a decode step that
/// reads every expert reads half as much.
#[derive(Clone, Debug)]
pub struct Fp8Weights {
    pub q: Tensor,
    pub scale: Tensor,
}

impl Fp8Weights {
    /// `(experts, rows, cols)`.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.q.dims3()
    }

    /// Bytes resident on the device.
    pub fn resident_bytes(&self) -> usize {
        self.q.elem_count() + self.scale.elem_count() * 4
    }
}

/// The largest finite E4M3FN magnitude, the value a row's peak scales to.
pub const E4M3_MAX: f32 = 448.0;

/// Whether the FP8 kernels take this geometry: [`supported`], and rows that
/// split into 16-byte loads.
pub fn fp8_supported(hidden: usize, inter: usize, experts: usize, top_k: usize) -> bool {
    supported(hidden, inter, experts, top_k)
        && hidden.is_multiple_of(16)
        && inter.is_multiple_of(16)
}

/// Exact E4M3FN decoding (the kernels' table).
pub fn e4m3_to_f32(byte: u8) -> f32 {
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

/// Round-to-nearest-even, saturating E4M3FN encoding (the kernels'
/// `mgl_to_e4m3`); zero and NaN encode as +0.
pub fn f32_to_e4m3(v: f32) -> u8 {
    let sign = if v < 0.0 { 128 } else { 0 };
    let a = v.abs().min(E4M3_MAX);
    if a.is_nan() || a <= 0.0 {
        return 0;
    }
    // a = f * 2^e with f in [0.5, 1).
    let e = a.log2().floor() as i32 + 1;
    let e = if a >= 2f32.powi(e) {
        e + 1
    } else if a < 2f32.powi(e - 1) {
        e - 1
    } else {
        e
    };
    let exp = e - 1 + 7;
    let code = if exp <= 0 {
        (a * 512.0).round_ties_even() as u32
    } else {
        let m = ((a / 2f32.powi(e - 1) - 1.0) * 8.0).round_ties_even() as u32;
        ((exp as u32) << 3).saturating_add(m).min(126)
    };
    sign | code as u8
}

/// `weights` (BF16 `[experts, rows, cols]`) as E4M3 rows, each scaled so its
/// peak is [`E4M3_MAX`] (an all-zero row keeps scale 1). On CUDA one launch
/// a call; the CPU runs the same encoding.
pub fn quantize_rows(weights: &Tensor) -> Result<Fp8Weights> {
    let (experts, rows, cols) = weights.dims3()?;
    if weights.dtype() != DType::BF16 || cols == 0 {
        candle_core::bail!(
            "FP8 expert quantization takes BF16 [experts, rows, cols], got {:?} {:?}",
            weights.dims(),
            weights.dtype()
        )
    }
    match weights.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::quantize_rows(&weights.contiguous()?, experts, rows, cols),
        device => {
            let values = host_f32(weights)?;
            let mut q = Vec::with_capacity(values.len());
            let mut scale = Vec::with_capacity(experts * rows);
            for row in values.chunks_exact(cols) {
                let peak = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = if peak > 0.0 { peak / E4M3_MAX } else { 1.0 };
                scale.push(s);
                q.extend(row.iter().map(|v| f32_to_e4m3(v / s)));
            }
            Ok(Fp8Weights {
                q: Tensor::from_vec(q, (experts, rows, cols), &Device::Cpu)?.to_device(device)?,
                scale: Tensor::from_vec(scale, (experts, rows), &Device::Cpu)?.to_device(device)?,
            })
        }
    }
}

/// The listed experts of `weights` decoded to BF16 `[experts.len(), rows,
/// cols]`, in list order.
pub fn dequantize_experts(weights: &Fp8Weights, experts: &[u32]) -> Result<Tensor> {
    let (count, rows, cols) = weights.dims3()?;
    if experts.is_empty() || experts.iter().any(|&e| e as usize >= count) {
        candle_core::bail!("FP8 expert dequantization of {experts:?} out of {count} experts")
    }
    match weights.q.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::dequantize_experts(weights, experts, rows, cols),
        device => {
            let q = weights
                .q
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<u8>()?;
            let scale = host_f32(&weights.scale)?;
            let mut out = Vec::with_capacity(experts.len() * rows * cols);
            for &e in experts {
                let e = e as usize;
                for r in 0..rows {
                    let s = scale[e * rows + r];
                    let row = &q[(e * rows + r) * cols..][..cols];
                    out.extend(
                        row.iter()
                            .map(|&b| half::bf16::from_f32(e4m3_to_f32(b) * s)),
                    );
                }
            }
            Tensor::from_vec(out, (experts.len(), rows, cols), &Device::Cpu)?.to_device(device)
        }
    }
}

/// [`experts`] over FP8 weights: on CUDA the FP8 up and down kernels (half
/// the bytes read); elsewhere the BF16 reference over the decoded weights.
pub fn experts_fp8(
    x: &Tensor,
    routing: &Tensor,
    gate_up: &Fp8Weights,
    down: &Fp8Weights,
) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts_count, rows, gate_hidden) = gate_up.dims3()?;
    let inter = rows / 2;
    let (routing_tokens, routing_width) = routing.dims2()?;
    let top_k = routing_width / 2;
    if x.dtype() != DType::F32
        || routing.dtype() != DType::F32
        || rows != 2 * inter
        || gate_hidden != hidden
        || down.dims3()? != (experts_count, hidden, inter)
        || routing_tokens != tokens
        || !fp8_supported(hidden, inter, experts_count, top_k)
    {
        candle_core::bail!(
            "FP8 GELU MoE operands do not agree: x {:?}, routing {:?}, gate_up {:?}, down {:?}",
            x.dims(),
            routing.dims(),
            gate_up.q.dims(),
            down.q.dims()
        )
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::experts_fp8(
            &x.contiguous()?,
            &routing.contiguous()?,
            gate_up,
            down,
            top_k,
        ),
        _ => {
            let all = (0..experts_count as u32).collect::<Vec<_>>();
            experts(
                x,
                routing,
                &dequantize_experts(gate_up, &all)?,
                &dequantize_experts(down, &all)?,
            )
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

    const MODULE: &str = "izwi_moe_gelu";

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        tensor: &Tensor,
        name: &str,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("BF16 GELU MoE {name} must be a CUDA tensor")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("BF16 GELU MoE {name} must be contiguous")
        };
        if !(start * tensor.dtype().size_in_bytes()).is_multiple_of(16) {
            candle_core::bail!("BF16 GELU MoE {name} must start on a 16-byte boundary")
        }
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
        i32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("BF16 GELU MoE {name} {value} exceeds i32"))
        })
    }

    fn u32_grid(value: usize, name: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("BF16 GELU MoE {name} grid {value} exceeds u32"))
        })
    }

    fn config(grid: (u32, u32), threads: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn route(
        logits: &Tensor,
        scale: &Tensor,
        tokens: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let (ls, _) = logits.storage_and_layout();
        let (ss, _) = scale.storage_and_layout();
        let lv = view::<f32>(&ls, logits, "logits")?;
        let sv = view::<f32>(&ss, scale, "scale")?;
        // SAFETY: the routing kernel writes every token's ids and weights.
        let out = unsafe { device.alloc::<f32>(tokens * 2 * top_k)? };
        let ptx = super::super::cuda::cuda_ptx_moe_gelu();
        let function = device.get_or_load_custom_func("izwi_mgl_route", MODULE, ptx)?;
        let mut builder = function.builder();
        builder.arg(&lv);
        builder.arg(&sv);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(experts, "experts")?,
            i32_arg(top_k, "top_k")?
        );
        let launch = config((u32_grid(tokens, "tokens")?, 1), 32);
        // SAFETY: argument order matches `izwi_mgl_route`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, 2 * top_k))))
    }

    pub(super) fn experts(
        x: &Tensor,
        routing: &Tensor,
        gate_up: &Tensor,
        down: &Tensor,
        slots: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let (tokens, hidden) = x.dims2()?;
        let (experts, rows, _) = gate_up.dims3()?;
        let inter = rows / 2;
        let pairs = tokens * slots;
        let (xs, _) = x.storage_and_layout();
        let (rs, _) = routing.storage_and_layout();
        let (gs, _) = gate_up.storage_and_layout();
        let (ds, _) = down.storage_and_layout();
        let xv = view::<f32>(&xs, x, "x")?;
        let rv = view::<f32>(&rs, routing, "routing")?;
        let gv = view::<half::bf16>(&gs, gate_up, "gate_up")?;
        let dv = view::<half::bf16>(&ds, down, "down")?;
        let ptx = super::super::cuda::cuda_ptx_moe_gelu();
        let (slots_i, hidden_i, inter_i, experts_i) = (
            i32_arg(slots, "slots")?,
            i32_arg(hidden, "hidden")?,
            i32_arg(inter, "inter")?,
            i32_arg(experts, "experts")?,
        );
        // SAFETY: the grouping kernel writes every offset and routed pair;
        // the up and down kernels every routed pair's row; the combine
        // every (token, channel).
        let offsets = unsafe { device.alloc::<i32>(experts + 1)? };
        let sorted = unsafe { device.alloc::<i32>(pairs)? };
        let act = unsafe { device.alloc::<f32>(pairs * inter)? };
        let partial = unsafe { device.alloc::<f32>(pairs * hidden)? };
        let out = unsafe { device.alloc::<f32>(tokens * hidden)? };

        let group = device.get_or_load_custom_func("izwi_mgl_group_pairs", MODULE, ptx)?;
        let mut builder = group.builder();
        builder.arg(&rv);
        builder.arg(&offsets);
        builder.arg(&sorted);
        candle_core::builder_arg!(builder, i32_arg(tokens, "tokens")?, slots_i, experts_i);
        // SAFETY: argument order matches `izwi_mgl_group_pairs`.
        unsafe { builder.launch(config((1, 1), 1024)) }.w()?;

        let up = device.get_or_load_custom_func("izwi_mgl_up_grouped", MODULE, ptx)?;
        let mut builder = up.builder();
        builder.arg(&xv);
        builder.arg(&offsets);
        builder.arg(&sorted);
        builder.arg(&gv);
        builder.arg(&act);
        candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i);
        let launch = config(
            (
                u32_grid(experts, "experts")?,
                u32_grid(inter.div_ceil(8), "inter")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_up_grouped`.
        unsafe { builder.launch(launch) }.w()?;

        let down_fn = device.get_or_load_custom_func("izwi_mgl_down_grouped", MODULE, ptx)?;
        let mut builder = down_fn.builder();
        builder.arg(&act);
        builder.arg(&offsets);
        builder.arg(&sorted);
        builder.arg(&dv);
        builder.arg(&partial);
        candle_core::builder_arg!(builder, hidden_i, inter_i);
        let launch = config(
            (
                u32_grid(experts, "experts")?,
                u32_grid(hidden.div_ceil(16), "hidden")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_down_grouped`.
        unsafe { builder.launch(launch) }.w()?;

        let combine = device.get_or_load_custom_func("izwi_mgl_combine", MODULE, ptx)?;
        let mut builder = combine.builder();
        builder.arg(&partial);
        builder.arg(&rv);
        builder.arg(&out);
        candle_core::builder_arg!(builder, slots_i, hidden_i, experts_i);
        let launch = config(
            (
                u32_grid(hidden.div_ceil(256), "hidden")?,
                u32_grid(tokens, "tokens")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_combine`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, hidden))))
    }

    fn wrap_as<T: candle_core::cuda_backend::CudaDType + candle_core::WithDType>(
        slice: CudaSlice<T>,
        device: &CudaDevice,
        shape: Shape,
    ) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    pub(super) fn quantize_rows(
        weights: &Tensor,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<super::Fp8Weights> {
        let device = weights.device().as_cuda_device()?;
        let (ws, _) = weights.storage_and_layout();
        let wv = view::<half::bf16>(&ws, weights, "weights")?;
        let total_rows = experts * rows;
        // SAFETY: one block per row writes every byte of its row and its
        // scale.
        let q = unsafe { device.alloc::<u8>(total_rows * cols)? };
        let scale = unsafe { device.alloc::<f32>(total_rows)? };
        let ptx = super::super::cuda::cuda_ptx_moe_gelu();
        let func = device.get_or_load_custom_func("izwi_mgl_fp8_quantize_rows", MODULE, ptx)?;
        let mut builder = func.builder();
        builder.arg(&wv);
        builder.arg(&q);
        builder.arg(&scale);
        candle_core::builder_arg!(builder, i32_arg(cols, "cols")?);
        // SAFETY: argument order matches `izwi_mgl_fp8_quantize_rows`.
        unsafe { builder.launch(config((u32_grid(total_rows, "rows")?, 1), 256)) }.w()?;
        drop(ws);
        Ok(super::Fp8Weights {
            q: wrap_as(q, device, Shape::from((experts, rows, cols))),
            scale: wrap_as(scale, device, Shape::from((experts, rows))),
        })
    }

    pub(super) fn dequantize_experts(
        weights: &super::Fp8Weights,
        experts: &[u32],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        let device = weights.q.device().as_cuda_device()?;
        let (qs, _) = weights.q.storage_and_layout();
        let (ss, _) = weights.scale.storage_and_layout();
        let qv = view::<u8>(&qs, &weights.q, "q")?;
        let sv = view::<f32>(&ss, &weights.scale, "scale")?;
        let ids = experts.iter().map(|&e| e as i32).collect::<Vec<_>>();
        let ids = device.clone_htod(&ids)?;
        let total = experts.len() * rows * cols;
        // SAFETY: the kernel writes every element.
        let out = unsafe { device.alloc::<half::bf16>(total)? };
        let ptx = super::super::cuda::cuda_ptx_moe_gelu();
        let func = device.get_or_load_custom_func("izwi_mgl_fp8_dequant_experts", MODULE, ptx)?;
        let mut builder = func.builder();
        builder.arg(&qv);
        builder.arg(&sv);
        builder.arg(&ids);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(experts.len(), "experts")?,
            i32_arg(rows, "rows")?,
            i32_arg(cols, "cols")?
        );
        let blocks = u32_grid(total.div_ceil(256).min(65_535), "dequant")?;
        // SAFETY: argument order matches `izwi_mgl_fp8_dequant_experts`.
        unsafe { builder.launch(config((blocks, 1), 256)) }.w()?;
        Ok(wrap_as(
            out,
            device,
            Shape::from((experts.len(), rows, cols)),
        ))
    }

    pub(super) fn experts_fp8(
        x: &Tensor,
        routing: &Tensor,
        gate_up: &super::Fp8Weights,
        down: &super::Fp8Weights,
        slots: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let (tokens, hidden) = x.dims2()?;
        let (experts, rows, _) = gate_up.dims3()?;
        let inter = rows / 2;
        let pairs = tokens * slots;
        let (xs, _) = x.storage_and_layout();
        let (rs, _) = routing.storage_and_layout();
        let (gqs, _) = gate_up.q.storage_and_layout();
        let (gss, _) = gate_up.scale.storage_and_layout();
        let (dqs, _) = down.q.storage_and_layout();
        let (dss, _) = down.scale.storage_and_layout();
        let xv = view::<f32>(&xs, x, "x")?;
        let rv = view::<f32>(&rs, routing, "routing")?;
        let gq = view::<u8>(&gqs, &gate_up.q, "gate_up")?;
        let gs = view::<f32>(&gss, &gate_up.scale, "gate_up scale")?;
        let dq = view::<u8>(&dqs, &down.q, "down")?;
        let ds = view::<f32>(&dss, &down.scale, "down scale")?;
        let ptx = super::super::cuda::cuda_ptx_moe_gelu();
        let (slots_i, hidden_i, inter_i, experts_i) = (
            i32_arg(slots, "slots")?,
            i32_arg(hidden, "hidden")?,
            i32_arg(inter, "inter")?,
            i32_arg(experts, "experts")?,
        );
        // SAFETY: as in `experts`: every offset, pair, act row, partial row
        // and output is written by its kernel.
        let offsets = unsafe { device.alloc::<i32>(experts + 1)? };
        let sorted = unsafe { device.alloc::<i32>(pairs)? };
        let act = unsafe { device.alloc::<f32>(pairs * inter)? };
        let partial = unsafe { device.alloc::<f32>(pairs * hidden)? };
        let out = unsafe { device.alloc::<f32>(tokens * hidden)? };

        let group = device.get_or_load_custom_func("izwi_mgl_group_pairs", MODULE, ptx)?;
        let mut builder = group.builder();
        builder.arg(&rv);
        builder.arg(&offsets);
        builder.arg(&sorted);
        candle_core::builder_arg!(builder, i32_arg(tokens, "tokens")?, slots_i, experts_i);
        // SAFETY: argument order matches `izwi_mgl_group_pairs`.
        unsafe { builder.launch(config((1, 1), 1024)) }.w()?;

        let up = device.get_or_load_custom_func("izwi_mgl_up_grouped_fp8", MODULE, ptx)?;
        let mut builder = up.builder();
        builder.arg(&xv);
        builder.arg(&offsets);
        builder.arg(&sorted);
        builder.arg(&gq);
        builder.arg(&gs);
        builder.arg(&act);
        candle_core::builder_arg!(builder, slots_i, hidden_i, inter_i);
        let launch = config(
            (
                u32_grid(experts, "experts")?,
                u32_grid(inter.div_ceil(8), "inter")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_up_grouped_fp8`.
        unsafe { builder.launch(launch) }.w()?;

        let down_fn = device.get_or_load_custom_func("izwi_mgl_down_grouped_fp8", MODULE, ptx)?;
        let mut builder = down_fn.builder();
        builder.arg(&act);
        builder.arg(&offsets);
        builder.arg(&sorted);
        builder.arg(&dq);
        builder.arg(&ds);
        builder.arg(&partial);
        candle_core::builder_arg!(builder, hidden_i, inter_i);
        let launch = config(
            (
                u32_grid(experts, "experts")?,
                u32_grid(hidden.div_ceil(16), "hidden")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_down_grouped_fp8`.
        unsafe { builder.launch(launch) }.w()?;

        let combine = device.get_or_load_custom_func("izwi_mgl_combine", MODULE, ptx)?;
        let mut builder = combine.builder();
        builder.arg(&partial);
        builder.arg(&rv);
        builder.arg(&out);
        candle_core::builder_arg!(builder, slots_i, hidden_i, experts_i);
        let launch = config(
            (
                u32_grid(hidden.div_ceil(256), "hidden")?,
                u32_grid(tokens, "tokens")?,
            ),
            256,
        );
        // SAFETY: argument order matches `izwi_mgl_combine`.
        unsafe { builder.launch(launch) }.w()?;
        Ok(wrap(out, device, Shape::from((tokens, hidden))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{relative_l2, xorshift_values};
    use half::bf16;

    fn bf16_tensor(shape: &[usize], seed: u64, scale: f32, device: &Device) -> Tensor {
        let values = xorshift_values(shape.iter().product(), seed, scale)
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device).unwrap()
    }

    #[test]
    fn routing_is_softmax_top_k_renormalized_then_scaled() {
        let logits = [0.5f32, 2.0, -1.0, 2.0, 1.0];
        let scale = [1.0f32, 2.0, 1.0, 0.5, 1.0];
        let (ids, weights) = route_row(&logits, &scale, 3);
        // Ties to the lower expert: 1 before 3.
        assert_eq!(ids, vec![1, 3, 4]);
        let exp = |v: f32| (v - 2.0).exp();
        let kept = exp(2.0) + exp(2.0) + exp(1.0);
        assert!((weights[0] - exp(2.0) / kept * 2.0).abs() < 1e-6);
        assert!((weights[1] - exp(2.0) / kept * 0.5).abs() < 1e-6);
        assert!((weights[2] - exp(1.0) / kept).abs() < 1e-6);
        let tensor = route(
            &Tensor::new(&[logits], &Device::Cpu).unwrap(),
            &Tensor::new(&scale, &Device::Cpu).unwrap(),
            3,
        )
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
        assert_eq!(&tensor[0][..3], &[1.0, 3.0, 4.0]);
        assert!(!supported(2816, 700, 128, 8));
        assert!(supported(2816, 704, 128, 8));
    }

    #[test]
    fn the_reference_matches_composed_expert_products() {
        let device = Device::Cpu;
        let (tokens, hidden, inter, experts, top_k) = (3, 64, 24, 6, 2);
        let x = Tensor::from_vec(
            xorshift_values(tokens * hidden, 1, 1.0),
            (tokens, hidden),
            &device,
        )
        .unwrap();
        let gate_up = bf16_tensor(&[experts, 2 * inter, hidden], 2, 0.2, &device);
        let down = bf16_tensor(&[experts, hidden, inter], 3, 0.2, &device);
        let logits = Tensor::from_vec(
            xorshift_values(tokens * experts, 4, 2.0),
            (tokens, experts),
            &device,
        )
        .unwrap();
        let scale =
            Tensor::from_vec(vec![1.0f32, 0.5, 2.0, 1.5, 1.0, 0.8], experts, &device).unwrap();
        let routing = route(&logits, &scale, top_k).unwrap();
        let got = experts_fn(&x, &routing, &gate_up, &down);
        // The same products composed from tensor operations.
        let routes = routing.to_vec2::<f32>().unwrap();
        let mut expected = Vec::new();
        for t in 0..tokens {
            let xt = x.narrow(0, t, 1).unwrap();
            let mut row = Tensor::zeros((1, hidden), DType::F32, &device).unwrap();
            for slot in 0..top_k {
                let e = routes[t][slot] as usize;
                let w = routes[t][top_k + slot] as f64;
                let gu = gate_up.get(e).unwrap().to_dtype(DType::F32).unwrap();
                let h = xt.matmul(&gu.t().unwrap()).unwrap();
                let gate = h.narrow(1, 0, inter).unwrap().gelu().unwrap();
                let up = h.narrow(1, inter, inter).unwrap();
                let d = down.get(e).unwrap().to_dtype(DType::F32).unwrap();
                let y = (gate * up).unwrap().matmul(&d.t().unwrap()).unwrap();
                row = (row + (y * w).unwrap()).unwrap();
            }
            expected.extend(row.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        }
        let error = relative_l2(&got, &expected);
        assert!(error < 1e-5, "{error}");
    }

    fn experts_fn(x: &Tensor, routing: &Tensor, gate_up: &Tensor, down: &Tensor) -> Vec<f32> {
        experts(x, routing, gate_up, down)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The CUDA kernels against the CPU reference, in the published
    /// geometry (fewer experts), alone and batched.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_cuda_kernels_match_the_reference() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("no CUDA device; skipped");
            return;
        };
        let (hidden, inter, experts, top_k) = (2816, 704, 16, 8);
        let gate_up = bf16_tensor(&[experts, 2 * inter, hidden], 5, 0.05, &device);
        let down = bf16_tensor(&[experts, hidden, inter], 6, 0.05, &device);
        let scale = Tensor::from_vec(xorshift_values(experts, 7, 0.5), experts, &device)
            .unwrap()
            .affine(1.0, 1.0)
            .unwrap();
        for tokens in [1usize, 5, 32] {
            let x = Tensor::from_vec(
                xorshift_values(tokens * hidden, 8, 1.0),
                (tokens, hidden),
                &device,
            )
            .unwrap();
            let logits = Tensor::from_vec(
                xorshift_values(tokens * experts, 9, 3.0),
                (tokens, experts),
                &device,
            )
            .unwrap();
            let routing = route(&logits, &scale, top_k).unwrap();
            let cpu_routing = route(
                &logits.to_device(&Device::Cpu).unwrap(),
                &scale.to_device(&Device::Cpu).unwrap(),
                top_k,
            )
            .unwrap();
            let ids = |r: &Tensor| {
                r.narrow(1, 0, top_k)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
            };
            assert_eq!(ids(&routing), ids(&cpu_routing), "{tokens} tokens");
            let got = experts_fn(&x, &routing, &gate_up, &down);
            let reference = experts_fn(
                &x.to_device(&Device::Cpu).unwrap(),
                &cpu_routing,
                &gate_up.to_device(&Device::Cpu).unwrap(),
                &down.to_device(&Device::Cpu).unwrap(),
            );
            let error = relative_l2(&got, &reference);
            assert!(error < 1e-5, "{tokens} tokens: {error}");
            // The first row alone gives the first row's bits.
            if tokens > 1 {
                let alone = experts_fn(
                    &x.narrow(0, 0, 1).unwrap(),
                    &routing.narrow(0, 0, 1).unwrap(),
                    &gate_up,
                    &down,
                );
                assert_eq!(alone, got[..hidden].to_vec());
            }
        }
    }

    #[test]
    fn the_e4m3_encoder_matches_candles_conversion() {
        let values = xorshift_values(4096, 21, 600.0)
            .into_iter()
            .chain([
                0.0, -0.0, 448.0, -448.0, 1e-4, 0.0019531, 0.00195, 1000.0, 7.5, 0.875,
            ])
            .collect::<Vec<_>>();
        let candle = Tensor::from_vec(values.clone(), values.len(), &Device::Cpu)
            .unwrap()
            .clamp(-E4M3_MAX, E4M3_MAX)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (v, want) in values.iter().zip(&candle) {
            let got = e4m3_to_f32(f32_to_e4m3(*v));
            assert!(
                got == *want || (got == 0.0 && *want == 0.0),
                "{v}: {got} vs {want}"
            );
        }
        for byte in 0..=255u8 {
            let v = e4m3_to_f32(byte);
            if v.is_finite() && v != 0.0 {
                assert_eq!(f32_to_e4m3(v), byte, "{byte}");
            }
        }
    }

    /// The kernels decode E4M3 by moving its bits into FP16 (seee.emmm to
    /// s0eeee.mmm0000000), the code's value times 2^-8 for every finite code.
    #[test]
    fn e4m3_bits_moved_into_fp16_are_the_value_over_256() {
        for byte in 0..=255u8 {
            let value = e4m3_to_f32(byte);
            if !value.is_finite() {
                continue;
            }
            let b = u16::from(byte);
            let bits = ((b & 0x80) << 8) | ((b & 0x7F) << 7);
            let moved = half::f16::from_bits(bits).to_f32() * 256.0;
            assert_eq!(moved, value, "{byte:#04x}");
        }
    }

    #[test]
    fn fp8_rows_round_trip_within_e4m3_precision() {
        let w = bf16_tensor(&[3, 8, 64], 5, 1.0, &Device::Cpu);
        let q = quantize_rows(&w).unwrap();
        let back = dequantize_experts(&q, &[0, 1, 2]).unwrap();
        let error = relative_l2(
            &back
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &w.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        // E4M3 keeps 3 mantissa bits: about 2-3% relative per element.
        assert!(error < 0.04, "relative error {error}");
        let picked = dequantize_experts(&q, &[2, 0]).unwrap();
        assert_eq!(
            picked
                .get(0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            back.get(2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fp8_experts_match_the_reference() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let (tokens, hidden, inter, experts_count, top_k) = (6, 512, 64, 8, 3);
        let gate_up = bf16_tensor(&[experts_count, 2 * inter, hidden], 1, 0.05, &Device::Cpu);
        let down = bf16_tensor(&[experts_count, hidden, inter], 2, 0.05, &Device::Cpu);
        // Quantization on the device writes the host encoder's bytes.
        let host_gate = quantize_rows(&gate_up).unwrap();
        let cuda_gate = quantize_rows(&gate_up.to_device(&device).unwrap()).unwrap();
        assert_eq!(
            cuda_gate
                .q
                .flatten_all()
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec1::<u8>()
                .unwrap(),
            host_gate.q.flatten_all().unwrap().to_vec1::<u8>().unwrap()
        );
        let cuda_down = quantize_rows(&down.to_device(&device).unwrap()).unwrap();
        let decoded = dequantize_experts(&cuda_gate, &[3, 1]).unwrap();
        assert_eq!(
            decoded
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            dequantize_experts(&host_gate, &[3, 1])
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let x = Tensor::from_vec(
            xorshift_values(tokens * hidden, 3, 1.0),
            (tokens, hidden),
            &device,
        )
        .unwrap();
        let logits = Tensor::from_vec(
            xorshift_values(tokens * experts_count, 4, 2.0),
            (tokens, experts_count),
            &device,
        )
        .unwrap();
        let scale = Tensor::ones(experts_count, DType::F32, &device).unwrap();
        let routing = route(&logits, &scale, top_k).unwrap();
        let got = experts_fp8(&x, &routing, &cuda_gate, &cuda_down).unwrap();
        // The reference: the CPU kernels' arithmetic over the exact F32
        // weights (a BF16 decode would round each e4m3 * scale product).
        let exact = |w: &Fp8Weights| {
            let (e, r, c) = w.dims3().unwrap();
            let q =
                w.q.flatten_all()
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<u8>()
                    .unwrap();
            let s = w
                .scale
                .flatten_all()
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let values = q
                .iter()
                .enumerate()
                .map(|(i, &b)| e4m3_to_f32(b) * s[i / c])
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (e, r, c), &Device::Cpu).unwrap()
        };
        let want = experts(
            &x.to_device(&Device::Cpu).unwrap(),
            &routing.to_device(&Device::Cpu).unwrap(),
            &exact(&cuda_gate),
            &exact(&cuda_down),
        )
        .unwrap();
        let host = |t: &Tensor| {
            t.to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let error = relative_l2(&host(&got), &host(&want));
        assert!(error < 1e-4, "relative error {error}");
    }
}

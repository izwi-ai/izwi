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
                u32_grid(hidden.div_ceil(8), "hidden")?,
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
}

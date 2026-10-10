//! Device-routed sparse-expert kernels for 128x128 block-FP8 MoE layers.
//!
//! These replace the host-routed per-expert loop (`models::shared::moe`) with
//! three launches per MoE layer:
//!
//! 1. [`route`]: F32 softmax over the router logits, top-k, renormalization,
//!    and optionally a folded shared-expert slot.
//! 2. [`fp8_gate_up`]: grouped gate+up projection with the SwiGLU epilogue.
//! 3. [`fp8_down`]: grouped down projection with the weighted combine in F32.
//!
//! Expert ids never leave the device, so a decode step has no host
//! synchronization inside the MoE block. Experts are stacked per layer:
//!
//! - `w13`: `[experts_total, 2 * inter, hidden]` E4M3FN bytes, gate rows first,
//!   then up rows;
//! - `w2`: `[experts_total, hidden, inter]`;
//! - scales: one F32 per 128x128 block.
//!
//! The routing tensor is F32 `[tokens, 2 * slots]`: per row, `slots` expert ids
//! (exact small integers), then `slots` combine weights.
//!
//! The CPU implementations are the portable reference: tests and the
//! load-time self-check compare against them. They are not a performance path.
use candle_core::{DType, Device, Result, Tensor};

use super::fp8::decode_e4m3fn;

/// Most routing slots (top-k plus the shared slot) the kernels accept.
pub const MAX_SLOTS: usize = 32;
/// Most routed experts the router kernel accepts (16 per lane, 32 lanes).
pub const MAX_EXPERTS: usize = 512;
const BLOCK: usize = 128;
const MAX_DYNAMIC_SHARED_BYTES: usize = 48 * 1024;

/// How the always-on shared expert joins the routed combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedSlot {
    /// No folded shared slot; the caller adds the shared expert itself.
    None,
    /// Folded as a stacked expert with combine weight 1.
    Ungated,
    /// Folded as a stacked expert weighted by `sigmoid(logits[:, num_experts])`;
    /// the router logits then carry one extra trailing column.
    Gated,
}

impl SharedSlot {
    pub(crate) fn mode(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Ungated => 1,
            Self::Gated => 2,
        }
    }
}

/// Routing contract for one MoE layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteSpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub shared: SharedSlot,
    /// Stacked expert index of the shared expert (ignored when `shared` is `None`).
    pub shared_slot_id: usize,
    /// Renormalize the top-k weights to sum to one (Qwen3.5/3.6 MoE does).
    pub norm_topk: bool,
}

impl RouteSpec {
    /// Routing slots per token: top-k plus the folded shared slot, if any.
    pub fn slots(&self) -> usize {
        self.top_k + usize::from(self.shared != SharedSlot::None)
    }

    /// Router logit columns the spec expects.
    pub fn logit_columns(&self) -> usize {
        self.num_experts + usize::from(self.shared == SharedSlot::Gated)
    }

    fn validate(&self) -> Result<()> {
        if self.num_experts == 0
            || self.num_experts > MAX_EXPERTS
            || self.top_k == 0
            || self.top_k > self.num_experts
            || self.slots() > MAX_SLOTS
        {
            candle_core::bail!("invalid MoE routing spec {self:?}")
        }
        Ok(())
    }
}

/// Whether the fused path can serve this device, activation dtype and geometry.
/// CUDA needs SM80+ and F16/BF16 activations; the CPU reference accepts F32 too.
pub fn supported(device: &Device, dtype: DType, hidden: usize, inter: usize, slots: usize) -> bool {
    let geometry = hidden > 0
        && inter > 0
        && hidden.is_multiple_of(BLOCK)
        && inter.is_multiple_of(BLOCK)
        && slots > 0
        && slots <= MAX_SLOTS;
    if !geometry {
        return false;
    }
    match device {
        Device::Cpu => matches!(dtype, DType::F32 | DType::F16 | DType::BF16),
        Device::Cuda(_) => {
            matches!(dtype, DType::F16 | DType::BF16)
                && hidden * 4 <= MAX_DYNAMIC_SHARED_BYTES
                && slots * inter * 4 <= MAX_DYNAMIC_SHARED_BYTES
                && super::fp8::device_is_sm80_or_newer(device)
        }
        // Apple GPUs run the same kernels over raw FP8 bytes with F16
        // activations (`kernels::metal_qwen36moe`).
        Device::Metal(_) => cfg!(feature = "metal") && dtype == DType::F16,
    }
}

/// Routed tokens from which the expert-major grouped kernels replace the
/// per-(token, slot) GEMVs: below this, the GEMVs' extra weight reads are
/// cheaper than grouping.
pub const GROUPED_MIN_TOKENS: usize = 32;
const MAX_GROUPED_EXPERTS: usize = 520;

/// Whether the grouped prefill kernels support this geometry: hidden and
/// expert width multiples of 512 (one 16-byte chunk per lane per 512 values),
/// at most 8 hidden and 4 expert-width chunks held in registers.
pub fn grouped_supported(hidden: usize, inter: usize, experts_total: usize) -> bool {
    hidden.is_multiple_of(512)
        && hidden / 512 <= 8
        && inter.is_multiple_of(512)
        && inter / 512 <= 4
        && experts_total <= MAX_GROUPED_EXPERTS
}

/// Expert-major routed MoE for prefill: the same result as
/// [`fp8_gate_up`] followed by [`fp8_down`], streaming each expert's weights
/// once per call instead of once per routed token.
pub fn fp8_moe_grouped(
    x: &Tensor,
    routing: &Tensor,
    slots: usize,
    w13: &Tensor,
    s13: &Tensor,
    w2: &Tensor,
    s2: &Tensor,
) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts_total, rows, _) = w13.dims3()?;
    let inter = rows / 2;
    check_routing(routing, tokens, slots)?;
    if !grouped_supported(hidden, inter, experts_total)
        || w2.dims3()? != (experts_total, hidden, inter)
    {
        candle_core::bail!(
            "grouped MoE does not support hidden {hidden}, expert width {inter}, {experts_total} experts"
        )
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::grouped(
            x,
            routing,
            slots,
            w13,
            s13,
            w2,
            s2,
            experts_total,
            hidden,
            inter,
        );
    }
    let act = fp8_gate_up(x, routing, slots, w13, s13)?;
    fp8_down(&act, routing, slots, w2, s2)
}

/// Router projection: `x` `[tokens, hidden]` (16-bit activations; F32 on the
/// CPU reference) against F32 router rows `[rows, hidden]` → F32 logits
/// `[tokens, rows]`, rounded through the activation dtype so routing sees the
/// values a projection in that dtype would produce.
pub fn router_logits(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (rows, w_hidden) = weight.dims2()?;
    if w_hidden != hidden || weight.dtype() != DType::F32 || !hidden.is_multiple_of(4) {
        candle_core::bail!(
            "invalid MoE router contract: x {:?}, weight {:?} {:?}",
            x.dims(),
            weight.dims(),
            weight.dtype()
        )
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::router(x, weight, tokens, rows, hidden);
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        return crate::kernels::metal_qwen36moe::router_logits(x, weight, tokens, rows, hidden);
    }
    let _ = (tokens, rows);
    x.to_dtype(DType::F32)?
        .matmul(&weight.t()?)?
        .to_dtype(x.dtype())?
        .to_dtype(DType::F32)
}

/// Fused router: `logits` `[tokens, spec.logit_columns()]` (any float dtype,
/// computed in F32) → routing `[tokens, 2 * slots]` F32 on the same device.
pub fn route(logits: &Tensor, spec: &RouteSpec) -> Result<Tensor> {
    spec.validate()?;
    let (tokens, columns) = logits.dims2()?;
    if columns != spec.logit_columns() {
        candle_core::bail!(
            "MoE router logits have {columns} columns, routing spec expects {}",
            spec.logit_columns()
        )
    }
    let logits = logits.to_dtype(DType::F32)?.contiguous()?;
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        return cuda_impl::route(&logits, tokens, spec);
    }
    #[cfg(feature = "metal")]
    if logits.device().is_metal() {
        return crate::kernels::metal_qwen36moe::route(&logits, tokens, spec);
    }
    if !logits.device().is_cpu() {
        candle_core::bail!("fused MoE routing has no implementation for this device")
    }
    let values = logits.flatten_all()?.to_vec1::<f32>()?;
    let routing = route_reference(&values, tokens, spec);
    Tensor::from_vec(routing, (tokens, 2 * spec.slots()), logits.device())
}

/// Grouped gate+up projection with the SwiGLU epilogue: `x` `[tokens, hidden]`
/// → `[tokens * slots, inter]` in `x`'s dtype, row `token * slots + slot`.
pub fn fp8_gate_up(
    x: &Tensor,
    routing: &Tensor,
    slots: usize,
    w13: &Tensor,
    s13: &Tensor,
) -> Result<Tensor> {
    let (tokens, hidden) = x.dims2()?;
    let (experts_total, rows, w_hidden) = w13.dims3()?;
    let inter = rows / 2;
    if w_hidden != hidden
        || rows % 2 != 0
        || w13.dtype() != DType::U8
        || s13.dtype() != DType::F32
        || s13.dims() != [experts_total, rows / BLOCK, hidden / BLOCK]
        || !hidden.is_multiple_of(BLOCK)
        || !inter.is_multiple_of(BLOCK)
    {
        candle_core::bail!(
            "invalid fused MoE gate/up contract: x {:?}, w13 {:?}, s13 {:?}",
            x.dims(),
            w13.dims(),
            s13.dims()
        )
    }
    check_routing(routing, tokens, slots)?;
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::gate_up(x, routing, slots, w13, s13, experts_total, hidden, inter);
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        return crate::kernels::metal_qwen36moe::gate_up(
            x,
            routing,
            slots,
            w13,
            s13,
            experts_total,
            hidden,
            inter,
        );
    }
    if !x.device().is_cpu() {
        candle_core::bail!("fused MoE gate/up has no implementation for this device")
    }
    let act = gate_up_reference(
        &host_f32(x)?,
        &host_f32(routing)?,
        &host_u8(w13)?,
        &host_f32(s13)?,
        Geometry {
            tokens,
            slots,
            hidden,
            inter,
            experts_total,
        },
    );
    Tensor::from_vec(act, (tokens * slots, inter), x.device())?.to_dtype(x.dtype())
}

/// Grouped down projection with the weighted combine:
/// `act` `[tokens * slots, inter]` → `[tokens, hidden]` in `act`'s dtype.
pub fn fp8_down(
    act: &Tensor,
    routing: &Tensor,
    slots: usize,
    w2: &Tensor,
    s2: &Tensor,
) -> Result<Tensor> {
    let (rows, inter) = act.dims2()?;
    let (experts_total, hidden, w_inter) = w2.dims3()?;
    if slots == 0
        || rows % slots != 0
        || w_inter != inter
        || w2.dtype() != DType::U8
        || s2.dtype() != DType::F32
        || s2.dims() != [experts_total, hidden / BLOCK, inter / BLOCK]
        || !hidden.is_multiple_of(BLOCK)
        || !inter.is_multiple_of(BLOCK)
    {
        candle_core::bail!(
            "invalid fused MoE down contract: act {:?}, w2 {:?}, s2 {:?}",
            act.dims(),
            w2.dims(),
            s2.dims()
        )
    }
    let tokens = rows / slots;
    check_routing(routing, tokens, slots)?;
    #[cfg(feature = "cuda")]
    if act.device().is_cuda() {
        return cuda_impl::down(act, routing, slots, w2, s2, experts_total, hidden, inter);
    }
    #[cfg(feature = "metal")]
    if act.device().is_metal() {
        return crate::kernels::metal_qwen36moe::down(
            act,
            routing,
            slots,
            w2,
            s2,
            experts_total,
            hidden,
            inter,
        );
    }
    if !act.device().is_cpu() {
        candle_core::bail!("fused MoE down has no implementation for this device")
    }
    let y = down_reference(
        &host_f32(act)?,
        &host_f32(routing)?,
        &host_u8(w2)?,
        &host_f32(s2)?,
        Geometry {
            tokens,
            slots,
            hidden,
            inter,
            experts_total,
        },
    );
    Tensor::from_vec(y, (tokens, hidden), act.device())?.to_dtype(act.dtype())
}

fn check_routing(routing: &Tensor, tokens: usize, slots: usize) -> Result<()> {
    if slots == 0 || slots > MAX_SLOTS || routing.dtype() != DType::F32 {
        candle_core::bail!("fused MoE routing must be F32 with 1..={MAX_SLOTS} slots")
    }
    if routing.dims() != [tokens, 2 * slots] {
        candle_core::bail!(
            "fused MoE routing {:?} does not match {tokens} tokens x {slots} slots",
            routing.dims()
        )
    }
    Ok(())
}

fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn host_u8(t: &Tensor) -> Result<Vec<u8>> {
    t.flatten_all()?.to_vec1::<u8>()
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    tokens: usize,
    slots: usize,
    hidden: usize,
    inter: usize,
    experts_total: usize,
}

/// Portable routing reference. Ties rank the lower expert index first, matching
/// the CUDA kernel.
fn route_reference(logits: &[f32], tokens: usize, spec: &RouteSpec) -> Vec<f32> {
    let columns = spec.logit_columns();
    let slots = spec.slots();
    let mut out = vec![0f32; tokens * 2 * slots];
    for token in 0..tokens {
        let row = &logits[token * columns..(token + 1) * columns];
        let experts = &row[..spec.num_experts];
        let max = experts.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = experts.iter().map(|l| (l - max).exp()).sum();
        let mut ranked: Vec<usize> = (0..spec.num_experts).collect();
        ranked.sort_by(|&a, &b| experts[b].total_cmp(&experts[a]).then(a.cmp(&b)));
        let picked: Vec<f32> = ranked[..spec.top_k]
            .iter()
            .map(|&e| (experts[e] - max).exp())
            .collect();
        let denom = if spec.norm_topk {
            picked.iter().sum()
        } else {
            total
        };
        let dst = &mut out[token * 2 * slots..(token + 1) * 2 * slots];
        for (k, (&expert, weight)) in ranked.iter().zip(&picked).enumerate() {
            dst[k] = expert as f32;
            dst[slots + k] = weight / denom;
        }
        if spec.shared != SharedSlot::None {
            dst[spec.top_k] = spec.shared_slot_id as f32;
            dst[slots + spec.top_k] = match spec.shared {
                SharedSlot::Gated => 1.0 / (1.0 + (-row[spec.num_experts]).exp()),
                _ => 1.0,
            };
        }
    }
    out
}

/// Dot product of one block-FP8 weight row with `x`, applying each 128-column
/// block's scale once (the kernel's accumulation order).
fn fp8_row_dot(row: &[u8], scales: &[f32], x: &[f32]) -> f32 {
    row.chunks(BLOCK)
        .zip(x.chunks(BLOCK))
        .zip(scales)
        .fold(0f32, |acc, ((w, x), &scale)| {
            let part = w
                .iter()
                .zip(x)
                .fold(0f32, |p, (&b, &v)| decode_e4m3fn(b).mul_add(v, p));
            part.mul_add(scale, acc)
        })
}

fn gate_up_reference(x: &[f32], routing: &[f32], w13: &[u8], s13: &[f32], g: Geometry) -> Vec<f32> {
    let kblocks = g.hidden / BLOCK;
    let scale_rows = 2 * g.inter / BLOCK;
    let mut act = vec![0f32; g.tokens * g.slots * g.inter];
    for token in 0..g.tokens {
        let xt = &x[token * g.hidden..(token + 1) * g.hidden];
        for slot in 0..g.slots {
            let expert = routing[token * 2 * g.slots + slot] as usize;
            if expert >= g.experts_total {
                continue;
            }
            let dst = &mut act[(token * g.slots + slot) * g.inter..][..g.inter];
            for (n, out) in dst.iter_mut().enumerate() {
                let row = |r: usize| &w13[(expert * 2 * g.inter + r) * g.hidden..][..g.hidden];
                let scales =
                    |r: usize| &s13[(expert * scale_rows + r / BLOCK) * kblocks..][..kblocks];
                let gate = fp8_row_dot(row(n), scales(n), xt);
                let up = fp8_row_dot(row(g.inter + n), scales(g.inter + n), xt);
                *out = gate / (1.0 + (-gate).exp()) * up;
            }
        }
    }
    act
}

fn down_reference(act: &[f32], routing: &[f32], w2: &[u8], s2: &[f32], g: Geometry) -> Vec<f32> {
    let kblocks = g.inter / BLOCK;
    let mut y = vec![0f32; g.tokens * g.hidden];
    for token in 0..g.tokens {
        for slot in 0..g.slots {
            let expert = routing[token * 2 * g.slots + slot] as usize;
            let weight = routing[token * 2 * g.slots + g.slots + slot];
            if expert >= g.experts_total {
                continue;
            }
            let a = &act[(token * g.slots + slot) * g.inter..][..g.inter];
            for h in 0..g.hidden {
                let row = &w2[(expert * g.hidden + h) * g.inter..][..g.inter];
                let scales = &s2[(expert * (g.hidden / BLOCK) + h / BLOCK) * kblocks..][..kblocks];
                y[token * g.hidden + h] =
                    weight.mul_add(fp8_row_dot(row, scales, a), y[token * g.hidden + h]);
            }
        }
    }
    y
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{RouteSpec, MAX_DYNAMIC_SHARED_BYTES};
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaDType, CudaDevice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_qwen36moe";
    const WARPS: u32 = 8;

    fn view<'a, T: CudaDType>(
        storage: &'a Storage,
        layout: &Layout,
        name: &str,
    ) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(storage) = storage else {
            candle_core::bail!("fused MoE {name} must be a CUDA tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("fused MoE {name} must be contiguous")
        };
        Ok(storage.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn aligned(layout: &Layout, name: &str) -> Result<()> {
        // uint4 weight loads need 16-byte aligned rows; allocations are
        // 256-byte aligned and every row length is a multiple of 128 bytes.
        if !layout.start_offset().is_multiple_of(16) {
            candle_core::bail!("fused MoE {name} view must start on a 16-byte boundary")
        }
        Ok(())
    }

    fn wrap<T: CudaDType>(slice: CudaSlice<T>, device: &CudaDevice, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    fn i32_arg(value: usize, name: &str) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("fused MoE {name} {value} exceeds i32")))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn grouped(
        x: &Tensor,
        routing: &Tensor,
        slots: usize,
        w13: &Tensor,
        s13: &Tensor,
        w2: &Tensor,
        s2: &Tensor,
        experts_total: usize,
        hidden: usize,
        inter: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let tokens = x.dim(0)?;
        let pairs = tokens * slots;
        let (x, routing, w13, s13, w2, s2) = (
            x.contiguous()?,
            routing.contiguous()?,
            w13.contiguous()?,
            s13.contiguous()?,
            w2.contiguous()?,
            s2.contiguous()?,
        );
        let (x_storage, x_layout) = x.storage_and_layout();
        let (r_storage, r_layout) = routing.storage_and_layout();
        let (w13_storage, w13_layout) = w13.storage_and_layout();
        let (s13_storage, s13_layout) = s13.storage_and_layout();
        let (w2_storage, w2_layout) = w2.storage_and_layout();
        let (s2_storage, s2_layout) = s2.storage_and_layout();
        aligned(w13_layout, "w13")?;
        aligned(w2_layout, "w2")?;
        if !(x_layout.start_offset() * 2).is_multiple_of(16) {
            candle_core::bail!("grouped MoE activations must start on a 16-byte boundary")
        }
        let r = view::<f32>(&r_storage, r_layout, "routing")?;
        let w13v = view::<u8>(&w13_storage, w13_layout, "w13")?;
        let s13v = view::<f32>(&s13_storage, s13_layout, "s13")?;
        let w2v = view::<u8>(&w2_storage, w2_layout, "w2")?;
        let s2v = view::<f32>(&s2_storage, s2_layout, "s2")?;
        // SAFETY: the grouping kernel writes every offset and every routed pair
        // index (routing ids are in range by contract).
        let offsets = unsafe { device.alloc::<i32>(experts_total + 1)? };
        let sorted = unsafe { device.alloc::<i32>(pairs)? };
        // SAFETY: the down kernel writes every (pair, channel) partial.
        let partial = unsafe { device.alloc::<f32>(pairs * hidden)? };
        let group = device.get_or_load_custom_func(
            "qwen36moe_group_pairs",
            MODULE,
            super::super::cuda_ptx::QWEN36MOE,
        )?;
        let mut builder = group.builder();
        builder.arg(&r);
        builder.arg(&offsets);
        builder.arg(&sorted);
        candle_core::builder_arg!(
            builder,
            i32_arg(tokens, "tokens")?,
            slots as i32,
            experts_total as i32
        );
        let single_block = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order matches `qwen36moe_group_pairs`.
        unsafe { builder.launch(single_block) }.w()?;
        macro_rules! run {
            ($ty:ty, $gate_up:literal, $down:literal, $combine:literal) => {{
                let xv = view::<$ty>(&x_storage, x_layout, "activations")?;
                // SAFETY: every routed pair's activation row is written.
                let act = unsafe { device.alloc::<$ty>(pairs * inter)? };
                // SAFETY: every (token, channel) output is written.
                let out = unsafe { device.alloc::<$ty>(tokens * hidden)? };
                let gate_up = device.get_or_load_custom_func(
                    $gate_up,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = gate_up.builder();
                builder.arg(&xv);
                builder.arg(&offsets);
                builder.arg(&sorted);
                builder.arg(&w13v);
                builder.arg(&s13v);
                builder.arg(&act);
                candle_core::builder_arg!(builder, slots as i32, hidden as i32, inter as i32);
                let config = LaunchConfig {
                    grid_dim: (experts_total as u32, (inter as u32).div_ceil(WARPS), 1),
                    block_dim: (32 * WARPS, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: validated grouped geometry; argument order matches.
                unsafe { builder.launch(config) }.w()?;
                let down = device.get_or_load_custom_func(
                    $down,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = down.builder();
                builder.arg(&act);
                builder.arg(&offsets);
                builder.arg(&sorted);
                builder.arg(&w2v);
                builder.arg(&s2v);
                builder.arg(&partial);
                candle_core::builder_arg!(builder, hidden as i32, inter as i32);
                let config = LaunchConfig {
                    grid_dim: (experts_total as u32, (hidden as u32).div_ceil(WARPS), 1),
                    block_dim: (32 * WARPS, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: validated grouped geometry; argument order matches.
                unsafe { builder.launch(config) }.w()?;
                let combine = device.get_or_load_custom_func(
                    $combine,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = combine.builder();
                builder.arg(&partial);
                builder.arg(&r);
                builder.arg(&out);
                candle_core::builder_arg!(builder, slots as i32, hidden as i32);
                let config = LaunchConfig {
                    grid_dim: (
                        u32::try_from(tokens)
                            .map_err(|_| candle_core::Error::Msg("combine grid".into()))?,
                        (hidden as u32).div_ceil(256),
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: argument order matches the combine kernel.
                unsafe { builder.launch(config) }.w()?;
                out
            }};
        }
        let tensor = match x.dtype() {
            DType::BF16 => wrap(
                run!(
                    half::bf16,
                    "qwen36moe_gate_up_grouped_bf16",
                    "qwen36moe_down_grouped_bf16",
                    "qwen36moe_combine_bf16"
                ),
                device,
                Shape::from((tokens, hidden)),
            ),
            DType::F16 => wrap(
                run!(
                    half::f16,
                    "qwen36moe_gate_up_grouped_f16",
                    "qwen36moe_down_grouped_f16",
                    "qwen36moe_combine_f16"
                ),
                device,
                Shape::from((tokens, hidden)),
            ),
            other => {
                candle_core::bail!("grouped MoE CUDA activations must be F16/BF16, found {other:?}")
            }
        };
        Ok(tensor)
    }

    pub(super) fn router(
        x: &Tensor,
        weight: &Tensor,
        tokens: usize,
        rows: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let (x, weight) = (x.contiguous()?, weight.contiguous()?);
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        if !w_layout.start_offset().is_multiple_of(4) {
            candle_core::bail!("MoE router rows must start on a 16-byte boundary")
        }
        let w = view::<f32>(&w_storage, w_layout, "router weight")?;
        // SAFETY: one warp writes every (token, row) logit.
        let out = unsafe { device.alloc::<f32>(tokens * rows)? };
        let config = LaunchConfig {
            grid_dim: (
                (rows as u32).div_ceil(WARPS),
                u32::try_from(tokens).map_err(|_| candle_core::Error::Msg("router grid".into()))?,
                1,
            ),
            block_dim: (32 * WARPS, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! run {
            ($ty:ty, $name:literal) => {{
                let xv = view::<$ty>(&x_storage, x_layout, "activations")?;
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&xv);
                builder.arg(&w);
                builder.arg(&out);
                candle_core::builder_arg!(builder, rows as i32, i32_arg(hidden, "hidden")?);
                // SAFETY: validated shapes; argument order matches the kernel.
                unsafe { builder.launch(config) }.w()?;
            }};
        }
        match x.dtype() {
            DType::BF16 => run!(half::bf16, "qwen36moe_router_logits_bf16"),
            DType::F16 => run!(half::f16, "qwen36moe_router_logits_f16"),
            other => {
                candle_core::bail!("MoE router CUDA activations must be F16/BF16, found {other:?}")
            }
        }
        drop((x_storage, w_storage));
        Ok(wrap(out, device, Shape::from((tokens, rows))))
    }

    pub(super) fn route(logits: &Tensor, tokens: usize, spec: &RouteSpec) -> Result<Tensor> {
        let device = logits.device().as_cuda_device()?;
        let slots = spec.slots();
        let (storage, layout) = logits.storage_and_layout();
        let input = view::<f32>(&storage, layout, "router logits")?;
        // SAFETY: lane 0 of every token's warp writes all 2 * slots outputs.
        let out = unsafe { device.alloc::<f32>(tokens * 2 * slots)? };
        let function = device.get_or_load_custom_func(
            "qwen36moe_route_f32",
            MODULE,
            super::super::cuda_ptx::QWEN36MOE,
        )?;
        let mut builder = function.builder();
        builder.arg(&input);
        builder.arg(&out);
        candle_core::builder_arg!(
            builder,
            i32_arg(tokens, "tokens")?,
            spec.num_experts as i32,
            spec.logit_columns() as i32,
            spec.top_k as i32,
            spec.shared.mode(),
            spec.shared_slot_id as i32,
            i32::from(spec.norm_topk)
        );
        let config = LaunchConfig {
            grid_dim: (tokens.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match `qwen36moe_route_f32`.
        unsafe { builder.launch(config) }.w()?;
        drop(storage);
        Ok(wrap(out, device, Shape::from((tokens, 2 * slots))))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gate_up(
        x: &Tensor,
        routing: &Tensor,
        slots: usize,
        w13: &Tensor,
        s13: &Tensor,
        experts_total: usize,
        hidden: usize,
        inter: usize,
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let tokens = x.dim(0)?;
        let (x, routing, w13, s13) = (
            x.contiguous()?,
            routing.contiguous()?,
            w13.contiguous()?,
            s13.contiguous()?,
        );
        let shared_bytes = hidden * 4;
        if shared_bytes > MAX_DYNAMIC_SHARED_BYTES {
            candle_core::bail!("fused MoE hidden {hidden} exceeds the gate/up staging buffer")
        }
        let (x_storage, x_layout) = x.storage_and_layout();
        let (r_storage, r_layout) = routing.storage_and_layout();
        let (w_storage, w_layout) = w13.storage_and_layout();
        let (s_storage, s_layout) = s13.storage_and_layout();
        aligned(w_layout, "w13")?;
        let r = view::<f32>(&r_storage, r_layout, "routing")?;
        let w = view::<u8>(&w_storage, w_layout, "w13")?;
        let s = view::<f32>(&s_storage, s_layout, "s13")?;
        let rows = tokens * slots;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows)
                    .map_err(|_| candle_core::Error::Msg("fused MoE grid".into()))?,
                (inter as u32).div_ceil(WARPS),
                1,
            ),
            block_dim: (32 * WARPS, 1, 1),
            shared_mem_bytes: shared_bytes as u32,
        };
        macro_rules! run {
            ($ty:ty, $name:literal) => {{
                let xv = view::<$ty>(&x_storage, x_layout, "activations")?;
                // SAFETY: one warp writes every (row, channel) output element.
                let out = unsafe { device.alloc::<$ty>(rows * inter)? };
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&xv);
                builder.arg(&r);
                builder.arg(&w);
                builder.arg(&s);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    slots as i32,
                    i32_arg(hidden, "hidden")?,
                    i32_arg(inter, "inter")?,
                    i32_arg(experts_total, "experts")?
                );
                // SAFETY: validated stacked geometry, aligned rows, matching signature.
                unsafe { builder.launch(config) }.w()?;
                out
            }};
        }
        let tensor = match x.dtype() {
            DType::BF16 => wrap(
                run!(half::bf16, "qwen36moe_gate_up_bf16"),
                device,
                Shape::from((rows, inter)),
            ),
            DType::F16 => wrap(
                run!(half::f16, "qwen36moe_gate_up_f16"),
                device,
                Shape::from((rows, inter)),
            ),
            other => {
                candle_core::bail!("fused MoE CUDA activations must be F16/BF16, found {other:?}")
            }
        };
        Ok(tensor)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn down(
        act: &Tensor,
        routing: &Tensor,
        slots: usize,
        w2: &Tensor,
        s2: &Tensor,
        experts_total: usize,
        hidden: usize,
        inter: usize,
    ) -> Result<Tensor> {
        let device = act.device().as_cuda_device()?;
        let tokens = act.dim(0)? / slots;
        let (act, routing, w2, s2) = (
            act.contiguous()?,
            routing.contiguous()?,
            w2.contiguous()?,
            s2.contiguous()?,
        );
        let shared_bytes = slots * inter * 4;
        if shared_bytes > MAX_DYNAMIC_SHARED_BYTES {
            candle_core::bail!("fused MoE {slots} slots x {inter} exceed the down staging buffer")
        }
        let (a_storage, a_layout) = act.storage_and_layout();
        let (r_storage, r_layout) = routing.storage_and_layout();
        let (w_storage, w_layout) = w2.storage_and_layout();
        let (s_storage, s_layout) = s2.storage_and_layout();
        aligned(w_layout, "w2")?;
        let r = view::<f32>(&r_storage, r_layout, "routing")?;
        let w = view::<u8>(&w_storage, w_layout, "w2")?;
        let s = view::<f32>(&s_storage, s_layout, "s2")?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(tokens)
                    .map_err(|_| candle_core::Error::Msg("fused MoE grid".into()))?,
                (hidden as u32).div_ceil(WARPS),
                1,
            ),
            block_dim: (32 * WARPS, 1, 1),
            shared_mem_bytes: shared_bytes as u32,
        };
        macro_rules! run {
            ($ty:ty, $name:literal) => {{
                let av = view::<$ty>(&a_storage, a_layout, "expert activations")?;
                // SAFETY: one warp writes every (token, channel) output element.
                let out = unsafe { device.alloc::<$ty>(tokens * hidden)? };
                let function = device.get_or_load_custom_func(
                    $name,
                    MODULE,
                    super::super::cuda_ptx::QWEN36MOE,
                )?;
                let mut builder = function.builder();
                builder.arg(&av);
                builder.arg(&r);
                builder.arg(&w);
                builder.arg(&s);
                builder.arg(&out);
                candle_core::builder_arg!(
                    builder,
                    slots as i32,
                    i32_arg(hidden, "hidden")?,
                    i32_arg(inter, "inter")?,
                    i32_arg(experts_total, "experts")?
                );
                // SAFETY: validated stacked geometry, aligned rows, matching signature.
                unsafe { builder.launch(config) }.w()?;
                out
            }};
        }
        let tensor = match act.dtype() {
            DType::BF16 => wrap(
                run!(half::bf16, "qwen36moe_down_bf16"),
                device,
                Shape::from((tokens, hidden)),
            ),
            DType::F16 => wrap(
                run!(half::f16, "qwen36moe_down_f16"),
                device,
                Shape::from((tokens, hidden)),
            ),
            other => {
                candle_core::bail!("fused MoE CUDA activations must be F16/BF16, found {other:?}")
            }
        };
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random stream (no test-only RNG dependency).
    fn stream(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    fn finite_fp8(next: &mut impl FnMut() -> u64) -> u8 {
        loop {
            let b = next() as u8;
            if b & 0x7f != 0x7f {
                return b;
            }
        }
    }

    struct Fixture {
        x: Tensor,
        logits: Tensor,
        w13: Tensor,
        s13: Tensor,
        w2: Tensor,
        s2: Tensor,
    }

    fn fixture(
        tokens: usize,
        hidden: usize,
        inter: usize,
        experts_total: usize,
        columns: usize,
    ) -> Fixture {
        let mut next = stream((tokens * 31 + hidden + inter * 7 + experts_total) as u64);
        let mut uniform =
            |scale: f32| ((next() >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale;
        let x: Vec<f32> = (0..tokens * hidden).map(|_| uniform(2.0)).collect();
        let logits: Vec<f32> = (0..tokens * columns).map(|_| uniform(3.0)).collect();
        let s13: Vec<f32> = (0..experts_total * (2 * inter / BLOCK) * (hidden / BLOCK))
            .map(|_| 0.002 + uniform(0.001))
            .collect();
        let s2: Vec<f32> = (0..experts_total * (hidden / BLOCK) * (inter / BLOCK))
            .map(|_| 0.002 + uniform(0.001))
            .collect();
        let mut next = stream(99 + tokens as u64);
        let w13: Vec<u8> = (0..experts_total * 2 * inter * hidden)
            .map(|_| finite_fp8(&mut next))
            .collect();
        let w2: Vec<u8> = (0..experts_total * hidden * inter)
            .map(|_| finite_fp8(&mut next))
            .collect();
        let cpu = Device::Cpu;
        Fixture {
            x: Tensor::from_vec(x, (tokens, hidden), &cpu).unwrap(),
            logits: Tensor::from_vec(logits, (tokens, columns), &cpu).unwrap(),
            w13: Tensor::from_vec(w13, (experts_total, 2 * inter, hidden), &cpu).unwrap(),
            s13: Tensor::from_vec(
                s13,
                (experts_total, 2 * inter / BLOCK, hidden / BLOCK),
                &cpu,
            )
            .unwrap(),
            w2: Tensor::from_vec(w2, (experts_total, hidden, inter), &cpu).unwrap(),
            s2: Tensor::from_vec(s2, (experts_total, hidden / BLOCK, inter / BLOCK), &cpu).unwrap(),
        }
    }

    /// Per-expert reference composed from the existing compact-FP8 projection,
    /// independent of the grouped reference's indexing.
    fn per_expert_reference(f: &Fixture, routing: &[f32], slots: usize, inter: usize) -> Vec<f32> {
        let (tokens, hidden) = f.x.dims2().unwrap();
        let mut y = vec![0f32; tokens * hidden];
        for token in 0..tokens {
            let xt = f.x.narrow(0, token, 1).unwrap();
            for slot in 0..slots {
                let expert = routing[token * 2 * slots + slot] as usize;
                let weight = routing[token * 2 * slots + slots + slot];
                let w13 = f.w13.get(expert).unwrap();
                let s13 = f.s13.get(expert).unwrap();
                let rows = inter / BLOCK;
                let gate = crate::kernels::cuda::fp8::block_fp8_projection(
                    &xt,
                    &w13.narrow(0, 0, inter).unwrap(),
                    &s13.narrow(0, 0, rows).unwrap(),
                )
                .unwrap();
                let up = crate::kernels::cuda::fp8::block_fp8_projection(
                    &xt,
                    &w13.narrow(0, inter, inter).unwrap(),
                    &s13.narrow(0, rows, rows).unwrap(),
                )
                .unwrap();
                let act = (candle_nn::ops::silu(&gate).unwrap() * up).unwrap();
                let down = crate::kernels::cuda::fp8::block_fp8_projection(
                    &act,
                    &f.w2.get(expert).unwrap(),
                    &f.s2.get(expert).unwrap(),
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                for (h, value) in down.iter().enumerate() {
                    y[token * hidden + h] += weight * value;
                }
            }
        }
        y
    }

    fn assert_close(actual: &[f32], expected: &[f32], rel: f32, label: &str) {
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= rel * scale,
                "{label} index {index}: {a} vs {e} (scale {scale})"
            );
        }
    }

    #[test]
    fn route_matches_sorted_softmax_reference_with_shared_slots() {
        let logits = Tensor::from_vec(
            vec![
                0.5f32, 2.0, -1.0, 2.0, 0.25, 3.0, // token 0 (tie between experts 1 and 3)
                -0.5, -0.25, 1.5, 0.0, 4.0, -2.0, // token 1
            ],
            (2, 6),
            &Device::Cpu,
        )
        .unwrap();
        let spec = RouteSpec {
            num_experts: 5,
            top_k: 2,
            shared: SharedSlot::Gated,
            shared_slot_id: 5,
            norm_topk: true,
        };
        let routing = route(&logits, &spec).unwrap().to_vec2::<f32>().unwrap();
        // Token 0: the tie ranks expert 1 before expert 3; both weigh 0.5.
        assert_eq!(&routing[0][..3], &[1.0, 3.0, 5.0]);
        assert!((routing[0][3] - 0.5).abs() < 1e-6 && (routing[0][4] - 0.5).abs() < 1e-6);
        assert!((routing[0][5] - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-6);
        // Token 1: experts 4 then 2; renormalized softmax over the pair.
        assert_eq!(&routing[1][..3], &[4.0, 2.0, 5.0]);
        let (a, b) = (4.0f32.exp(), 1.5f32.exp());
        assert!((routing[1][3] - a / (a + b)).abs() < 1e-6);
        assert!((routing[1][4] - b / (a + b)).abs() < 1e-6);

        let raw = RouteSpec {
            shared: SharedSlot::None,
            norm_topk: false,
            ..spec
        };
        let routing = route(&logits.narrow(1, 0, 5).unwrap(), &raw)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let total: f32 = [-0.5f32, -0.25, 1.5, 0.0, 4.0]
            .iter()
            .map(|v| v.exp())
            .sum();
        assert!((routing[1][2] - 4.0f32.exp() / total).abs() < 1e-6);
    }

    #[test]
    fn router_logits_round_through_the_activation_dtype() {
        let x = Tensor::from_vec(
            (0..2 * 8)
                .map(|i| (i as f32 * 0.37).sin() * 2.0)
                .collect::<Vec<_>>(),
            (2, 8),
            &Device::Cpu,
        )
        .unwrap();
        let w = Tensor::from_vec(
            (0..3 * 8)
                .map(|i| (i as f32 * 0.61).cos() * 0.1)
                .collect::<Vec<_>>(),
            (3, 8),
            &Device::Cpu,
        )
        .unwrap();
        let exact = router_logits(&x, &w).unwrap().to_vec2::<f32>().unwrap();
        let bf16 = router_logits(&x.to_dtype(DType::BF16).unwrap(), &w)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (row, (e, b)) in exact.iter().zip(&bf16).enumerate() {
            for (a, c) in e.iter().zip(b) {
                assert!(
                    (a - c).abs() <= a.abs() / 64.0 + 1e-3,
                    "row {row}: {a} vs {c}"
                );
                assert_eq!(
                    *c,
                    half::bf16::from_f32(*c).to_f32(),
                    "BF16-representable logit"
                );
            }
        }
        assert!(router_logits(&x, &w.narrow(1, 0, 4).unwrap()).is_err());
    }

    #[test]
    fn grouped_prefill_matches_the_per_pair_path_and_gates_its_geometry() {
        assert!(grouped_supported(2048, 512, 257));
        assert!(
            !grouped_supported(256, 512, 257),
            "hidden must be a multiple of 512"
        );
        assert!(
            !grouped_supported(2048, 256, 257),
            "expert width must be a multiple of 512"
        );
        assert!(
            !grouped_supported(8192, 512, 257),
            "at most 8 hidden chunks per lane"
        );
        assert!(
            !grouped_supported(2048, 512, 600),
            "expert table fits shared memory"
        );
        let spec = RouteSpec {
            num_experts: 4,
            top_k: 2,
            shared: SharedSlot::Ungated,
            shared_slot_id: 4,
            norm_topk: true,
        };
        let f = fixture(3, 512, 512, 5, spec.logit_columns());
        let routing = route(&f.logits, &spec).unwrap();
        let slots = spec.slots();
        let per_pair = fp8_down(
            &fp8_gate_up(&f.x, &routing, slots, &f.w13, &f.s13).unwrap(),
            &routing,
            slots,
            &f.w2,
            &f.s2,
        )
        .unwrap();
        let grouped = fp8_moe_grouped(&f.x, &routing, slots, &f.w13, &f.s13, &f.w2, &f.s2).unwrap();
        assert_eq!(
            grouped.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            per_pair.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let narrow = fixture(2, 256, 128, 5, spec.logit_columns());
        let routing = route(&narrow.logits, &spec).unwrap();
        assert!(fp8_moe_grouped(
            &narrow.x,
            &routing,
            slots,
            &narrow.w13,
            &narrow.s13,
            &narrow.w2,
            &narrow.s2
        )
        .is_err());
    }

    #[test]
    fn route_rejects_mismatched_columns_and_oversized_specs() {
        let logits = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        let spec = RouteSpec {
            num_experts: 4,
            top_k: 2,
            shared: SharedSlot::Gated,
            shared_slot_id: 4,
            norm_topk: true,
        };
        assert!(
            route(&logits, &spec).is_err(),
            "gated spec needs a 5th column"
        );
        let too_many = RouteSpec {
            top_k: 5,
            shared: SharedSlot::None,
            ..spec
        };
        assert!(route(&logits, &too_many).is_err());
    }

    #[test]
    fn grouped_cpu_reference_matches_per_expert_projection_composition() {
        let (tokens, hidden, inter, experts) = (3usize, 256usize, 128usize, 6usize);
        let spec = RouteSpec {
            num_experts: experts,
            top_k: 2,
            shared: SharedSlot::Gated,
            shared_slot_id: experts,
            norm_topk: true,
        };
        let f = fixture(tokens, hidden, inter, experts + 1, spec.logit_columns());
        let routing = route(&f.logits, &spec).unwrap();
        let slots = spec.slots();
        let act = fp8_gate_up(&f.x, &routing, slots, &f.w13, &f.s13).unwrap();
        assert_eq!(act.dims(), [tokens * slots, inter]);
        let y = fp8_down(&act, &routing, slots, &f.w2, &f.s2).unwrap();
        assert_eq!(y.dims(), [tokens, hidden]);
        let routing = routing.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = per_expert_reference(&f, &routing, slots, inter);
        let actual = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_close(&actual, &expected, 1e-4, "grouped vs per-expert");
    }

    #[test]
    fn fused_path_preserves_activation_dtype_and_rejects_bad_geometry() {
        let spec = RouteSpec {
            num_experts: 4,
            top_k: 2,
            shared: SharedSlot::None,
            shared_slot_id: 0,
            norm_topk: true,
        };
        let f = fixture(2, 128, 128, 4, spec.logit_columns());
        let routing = route(&f.logits, &spec).unwrap();
        let x = f.x.to_dtype(DType::BF16).unwrap();
        let act = fp8_gate_up(&x, &routing, 2, &f.w13, &f.s13).unwrap();
        assert_eq!(act.dtype(), DType::BF16);
        let y = fp8_down(&act, &routing, 2, &f.w2, &f.s2).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        // Wrong slot count for the routing tensor.
        assert!(fp8_gate_up(&x, &routing, 3, &f.w13, &f.s13).is_err());
        // Scale grid that does not match the weight blocks.
        let bad_scales = f.s13.narrow(1, 0, 1).unwrap();
        assert!(fp8_gate_up(&x, &routing, 2, &f.w13, &bad_scales).is_err());
        assert!(!supported(&Device::Cpu, DType::BF16, 100, 128, 2));
        assert!(supported(&Device::Cpu, DType::BF16, 128, 128, 2));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_moe_matches_cpu_reference_at_trunk_geometry() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        // Real Qwen3.6-35B-A3B shapes: hidden 2048, expert width 512, 256
        // routed experts + the folded shared slot, top-8, gated shared expert.
        let (hidden, inter, experts) = (2048usize, 512usize, 256usize);
        let spec = RouteSpec {
            num_experts: experts,
            top_k: 8,
            shared: SharedSlot::Gated,
            shared_slot_id: experts,
            norm_topk: true,
        };
        for tokens in [1usize, 4, 9, 33] {
            let f = fixture(tokens, hidden, inter, experts + 1, spec.logit_columns());
            let cpu_routing = route(&f.logits, &spec).unwrap();
            let gpu_routing = route(&f.logits.to_device(&device).unwrap(), &spec).unwrap();
            assert_eq!(
                cpu_routing
                    .to_vec2::<f32>()
                    .unwrap()
                    .iter()
                    .map(|r| r[..spec.slots()].to_vec())
                    .collect::<Vec<_>>(),
                gpu_routing
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap()
                    .iter()
                    .map(|r| r[..spec.slots()].to_vec())
                    .collect::<Vec<_>>(),
                "expert selection must match exactly (T={tokens})"
            );
            let slots = spec.slots();
            for dtype in [DType::BF16, DType::F16] {
                let x = f.x.to_dtype(dtype).unwrap();
                let expected = fp8_down(
                    &fp8_gate_up(&x, &cpu_routing, slots, &f.w13, &f.s13).unwrap(),
                    &cpu_routing,
                    slots,
                    &f.w2,
                    &f.s2,
                )
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let (w13, s13, w2, s2) = (
                    f.w13.to_device(&device).unwrap(),
                    f.s13.to_device(&device).unwrap(),
                    f.w2.to_device(&device).unwrap(),
                    f.s2.to_device(&device).unwrap(),
                );
                let act = fp8_gate_up(
                    &x.to_device(&device).unwrap(),
                    &gpu_routing,
                    slots,
                    &w13,
                    &s13,
                )
                .unwrap();
                let actual = fp8_down(&act, &gpu_routing, slots, &w2, &s2)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_close(&actual, &expected, 0.02, &format!("{dtype:?} T={tokens}"));
            }
        }
    }
}

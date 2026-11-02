//! The trunk on a device: the CPU reference's layers (`trunk/`) in the
//! device forms of `device/mod.rs` (16-bit F16 on Metal, BF16 on CUDA).
//!
//! * Gated attention: [`DeviceAttention`] over the trunk's packed
//!   `q_proj | k_proj | v_proj`, NeoX RoPE.
//! * Gated DeltaNet: `in_proj_qkv | in_proj_z` packed 16-bit, `in_proj_a` /
//!   `in_proj_b` F32, then per row `gdn_conv` (causal conv + SiLU over the
//!   request's 3-column history) and `gdn_decode` (q/k L2 norm, the delta
//!   rule on the F32 state with grouped value heads, the plain-`w` gated
//!   norm), then `out_proj`. `gdn_decode` takes the published 128-dim
//!   heads; another head size (the L0 test geometry, the tiny fixture's)
//!   runs the same rule as a Candle composition. On CUDA a span of
//!   [`CHUNKED_MIN_ROWS`] rows or more runs the causal conv over the span
//!   and the chunked prefill instead, once the load verified it.
//! * MoE: the F32 router through the fused routing kernel (softmax, top-k,
//!   renormalised), the weights rounded to BF16 as upstream does, the
//!   stacked E4M3 experts through the fused gate/up and down kernels (or
//!   the grouped ones on a long span), the 16-bit shared expert gated by its
//!   F32 sigmoid gate. A layer Metal's range probe breached, or every layer
//!   under `IZWI_FISH_DRAMA3_MOE_BACKEND=loop`, routes on the host and runs
//!   each expert's rows through the FP8 GEMV (F32 activations on Metal,
//!   BF16 on CUDA).

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{expert, trunk_layer, FishDrama3Checkpoint, TRUNK_PREFIX};
use super::super::config::{FishDrama3Config, FishDrama3LayerKind};
use super::super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::super::residency::FishDrama3WeightForm;
use super::super::trunk::state::FishDrama3LinearState;
use super::super::trunk::{norm_gain, LayerTrace, NormGain};
use super::{
    add_residual, count_chunked_deltanet_span, count_grouped_moe_span, record, rms_norm, swiglu,
    vector, DeviceAttention, DeviceForms, DeviceRow, Probe,
};
use crate::error::{Error, Result};
use crate::kernels::cuda::gdn::{self, GdnDecodeSpec};
use crate::kernels::cuda::moe::{self, GroupedKernel, RouteSpec, SharedSlot};

/// Rows from which a CUDA span runs the chunked DeltaNet prefill (the
/// grouped MoE's threshold too): below it the per-row kernels cost less.
pub(crate) const CHUNKED_MIN_ROWS: usize = 32;

/// How a MoE layer runs its routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3MoeBackend {
    /// Device routing and the fused FP8 kernels (F16 activations).
    Fused,
    /// Host routing and a per-expert loop of FP8 GEMVs with F32
    /// activations: the breach fallback and a diagnosis switch.
    Loop,
}

impl FishDrama3MoeBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Loop => "loop",
        }
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Fish Audio Drama 3 on the device: {}",
        message.into()
    ))
}

/// The F32 gain of norm `name` on `device`.
pub(crate) fn gain(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    width: usize,
    kind: NormGain,
    device: &Device,
) -> Result<Tensor> {
    vector(&norm_gain(checkpoint, name, width, kind)?, device)
}

/// The BF16 projection `name` as a 16-bit layer in `forms` (cast on the
/// host to F16 on Metal, kept BF16 on CUDA; uploaded once).
pub(crate) fn half_linear(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    rows: usize,
    cols: usize,
    forms: DeviceForms,
    device: &Device,
) -> Result<FishDrama3Linear> {
    FishDrama3Linear::from_half(checkpoint.bf16_tensor(name, &[rows, cols], forms.half, device)?)
}

fn f32_linear(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<FishDrama3Linear> {
    FishDrama3Linear::from_values(
        checkpoint.bf16_to_f32(name, &[rows, cols])?,
        rows,
        cols,
        FishDrama3WeightForm::F32,
        device,
    )
}

/// Gated DeltaNet on the device.
pub(crate) struct DeviceDeltaNet {
    /// `[conv_channels + value_width, hidden]`: `in_proj_qkv` then `in_proj_z`.
    qkvz: FishDrama3Linear,
    in_proj_a: FishDrama3Linear,
    in_proj_b: FishDrama3Linear,
    /// `[channels, kernel]` F32, tap 0 on the oldest column.
    conv: Tensor,
    /// `-exp(A_log)`.
    neg_a: Tensor,
    dt_bias: Tensor,
    /// The output norm's plain gain.
    norm: Tensor,
    out_proj: FishDrama3Linear,
    spec: GdnDecodeSpec,
    /// Whether `gdn_decode` takes the heads (else the Candle composition).
    fused: bool,
    /// Whether a long span runs the chunked prefill (CUDA, once verified).
    pub(crate) chunked: bool,
    key_dim: usize,
    channels: usize,
    value_width: usize,
    state_shape: (usize, usize, usize),
}

impl DeviceDeltaNet {
    fn load(
        checkpoint: &FishDrama3Checkpoint,
        layer: usize,
        config: &FishDrama3Config,
        forms: DeviceForms,
        device: &Device,
    ) -> Result<Self> {
        let t = &config.trunk;
        let la = t.linear;
        let h = t.hidden;
        if la.conv_kernel != gdn::CONV_TAPS {
            return Err(load_error(format!(
                "the DeltaNet conv kernel takes {} taps; this checkpoint has {}",
                gdn::CONV_TAPS,
                la.conv_kernel
            )));
        }
        let x = format!("{}.linear_attn", trunk_layer(layer));
        let channels = la.conv_channels();
        let vw = la.value_width();
        let mut packed =
            checkpoint.bf16_to_f32(&format!("{x}.in_proj_qkv.weight"), &[channels, h])?;
        packed.extend(checkpoint.bf16_to_f32(&format!("{x}.in_proj_z.weight"), &[vw, h])?);
        let vector_of =
            |name: &str| checkpoint.bf16_to_f32(&format!("{x}.{name}"), &[la.value_heads]);
        let neg_a = vector_of("A_log")?
            .into_iter()
            .map(|v| -v.exp())
            .collect::<Vec<_>>();
        let conv = checkpoint.bf16_to_f32(
            &format!("{x}.conv1d.weight"),
            &[channels, 1, la.conv_kernel],
        )?;
        Ok(Self {
            qkvz: FishDrama3Linear::from_values(
                packed,
                channels + vw,
                h,
                forms.weight_form(),
                device,
            )?,
            in_proj_a: f32_linear(
                checkpoint,
                &format!("{x}.in_proj_a.weight"),
                la.value_heads,
                h,
                device,
            )?,
            in_proj_b: f32_linear(
                checkpoint,
                &format!("{x}.in_proj_b.weight"),
                la.value_heads,
                h,
                device,
            )?,
            conv: Tensor::from_vec(conv, (channels, la.conv_kernel), device)?,
            neg_a: vector(&neg_a, device)?,
            dt_bias: vector(&vector_of("dt_bias")?, device)?,
            norm: gain(
                checkpoint,
                &format!("{x}.norm.weight"),
                la.value_head_dim,
                NormGain::Plain,
                device,
            )?,
            out_proj: half_linear(
                checkpoint,
                &format!("{x}.out_proj.weight"),
                h,
                vw,
                forms,
                device,
            )?,
            spec: GdnDecodeSpec {
                key_heads: la.key_heads,
                value_heads: la.value_heads,
                grouped: true,
                norm_eps: t.rms_norm_eps as f32,
            },
            fused: gdn::supported(device, la.key_head_dim, la.value_head_dim, la.conv_kernel),
            chunked: false,
            key_dim: la.key_head_dim,
            channels,
            value_width: vw,
            state_shape: (la.value_heads, la.key_head_dim, la.value_head_dim),
        })
    }

    fn resident_bytes(&self) -> usize {
        [&self.qkvz, &self.in_proj_a, &self.in_proj_b, &self.out_proj]
            .iter()
            .map(|l| l.resident_bytes())
            .sum::<usize>()
            + [&self.conv, &self.neg_a, &self.dt_bias, &self.norm]
                .iter()
                .map(|t| t.elem_count() * 4)
                .sum::<usize>()
    }

    /// `gdn_decode`'s rule as a Candle composition for one row, at any head
    /// size: `conv` `[channels]` (after the conv's SiLU), `z` `[value_width]`,
    /// `beta_raw` and `alpha` `[value_heads]` F32, `state` `[value_heads,
    /// key_dim, value_dim]`. Returns the gated output `[value_width]` and the
    /// next state.
    fn recurrence(
        &self,
        conv: &Tensor,
        z: &Tensor,
        beta_raw: &Tensor,
        alpha: &Tensor,
        state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (hv, dk, dv) = self.state_shape;
        let hk = self.spec.key_heads;
        let kw = hk * dk;
        let device = conv.device();
        let conv = conv.flatten_all()?;
        let l2 = |x: Tensor| -> Result<Tensor> {
            let norm = (x.sqr()?.sum_keepdim(1)?
                + f64::from(super::super::trunk::deltanet::L2_NORM_EPS))?
            .sqrt()?;
            Ok(x.broadcast_div(&norm)?)
        };
        // Value head j reads key head j / (value_heads / key_heads).
        let group = Tensor::from_vec(
            (0..hv).map(|j| (j / (hv / hk)) as u32).collect::<Vec<_>>(),
            hv,
            device,
        )?;
        let q = (l2(conv.narrow(0, 0, kw)?.reshape((hk, dk))?)?
            * (1.0 / (self.key_dim as f64).sqrt()))?
        .index_select(&group, 0)?;
        let k = l2(conv.narrow(0, kw, kw)?.reshape((hk, dk))?)?.index_select(&group, 0)?;
        let v = conv.narrow(0, 2 * kw, hv * dv)?.reshape((hv, dv))?;
        // g = -exp(A_log) * softplus(a + dt_bias), softplus without overflow.
        let gate_in = (alpha.flatten_all()? + &self.dt_bias)?;
        let softplus = (gate_in.relu()? + (gate_in.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
        let decay = (softplus * &self.neg_a)?.exp()?.reshape((hv, 1, 1))?;
        let beta = candle_nn::ops::sigmoid(&beta_raw.flatten_all()?)?.reshape((hv, 1))?;
        let s = state.broadcast_mul(&decay)?;
        let memory = k.unsqueeze(1)?.matmul(&s)?.squeeze(1)?;
        let delta = (v - memory)?.broadcast_mul(&beta)?;
        let s = (s + k.unsqueeze(2)?.matmul(&delta.unsqueeze(1)?)?)?;
        let o = q.unsqueeze(1)?.matmul(&s)?.squeeze(1)?.contiguous()?;
        let o = candle_nn::ops::rms_norm(&o, &self.norm, self.spec.norm_eps)?;
        let gate = candle_nn::ops::silu(&z.reshape((hv, dv))?)?;
        Ok(((o * gate)?.reshape(hv * dv)?, s))
    }

    /// One request's span through the causal conv over the span and the
    /// chunked prefill (FLA's chunk rule, `kernels::cuda`): the per-row
    /// kernels' rule with the span's rows resolved a chunk at a time.
    /// `packed` is `[n, conv_channels + value_width]`, `alpha` and `beta`
    /// `[n, value_heads]` F32. Returns the gated output `[n, value_width]`
    /// F32 and advances `state`, or `None` (nothing moved) when the kernels
    /// do not take these tensors.
    fn chunked_span(
        &self,
        packed: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        state: &mut FishDrama3LinearState,
    ) -> Result<Option<Tensor>> {
        let (c, vw) = (self.channels, self.value_width);
        let (hv, dk, dv) = self.state_shape;
        let hk = self.spec.key_heads;
        let kw = hk * dk;
        let n = packed.dim(0)?;
        let mixed = packed
            .narrow(1, 0, c)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .reshape((1, n, c))?;
        // The ring oldest first, as `[channels, columns]`.
        let history = Tensor::stack(&state.conv.columns()?, 1)?;
        let Some((conv, final_history)) =
            crate::kernels::cuda::try_qwen35_causal_conv_sequence(&mixed, &self.conv, &history)
        else {
            return Ok(None);
        };
        let conv = conv.reshape((n, c))?;
        let l2 = |t: Tensor| -> Result<Tensor> {
            let norm = (t.sqr()?.sum_keepdim(2)?
                + f64::from(super::super::trunk::deltanet::L2_NORM_EPS))?
            .sqrt()?;
            Ok(t.broadcast_div(&norm)?)
        };
        // Value head j reads key head j / (value_heads / key_heads).
        let expand = |t: Tensor| -> Result<Tensor> {
            Ok(t.unsqueeze(2)?
                .broadcast_as((n, hk, hv / hk, dk))?
                .reshape((1, n, hv, dk))?
                .contiguous()?)
        };
        let q = expand(l2(conv
            .narrow(1, 0, kw)?
            .contiguous()?
            .reshape((n, hk, dk))?)?)?;
        let k = expand(l2(conv
            .narrow(1, kw, kw)?
            .contiguous()?
            .reshape((n, hk, dk))?)?)?;
        let v = conv
            .narrow(1, 2 * kw, hv * dv)?
            .contiguous()?
            .reshape((1, n, hv, dv))?;
        // g = -exp(A_log) * softplus(a + dt_bias), softplus without overflow;
        // beta = sigmoid(b).
        let gate_in = alpha.broadcast_add(&self.dt_bias)?;
        let softplus = (gate_in.relu()? + (gate_in.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
        let g = softplus
            .broadcast_mul(&self.neg_a)?
            .contiguous()?
            .reshape((1, n, hv))?;
        let b = candle_nn::ops::sigmoid(beta)?
            .contiguous()?
            .reshape((1, n, hv))?;
        let initial = state.recurrent().reshape((1, hv, dk, dv))?;
        let Some((out, next)) =
            crate::kernels::cuda::try_chunked_deltanet_recurrence(&q, &k, &v, &g, &b, &initial)
        else {
            return Ok(None);
        };
        let out = out.contiguous()?.reshape((n * hv, dv))?;
        let normed = candle_nn::ops::rms_norm(&out, &self.norm, self.spec.norm_eps)?;
        let z = packed
            .narrow(1, c, vw)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .reshape((n * hv, dv))?;
        let gated = (normed * candle_nn::ops::silu(&z)?)?.reshape((n, vw))?;
        // The ring's next columns are the span's last raw inputs; the state
        // is copied out of the kernels' packed output, so the span's outputs
        // are not kept alive with it.
        state.conv.set_history(&final_history.t()?.contiguous()?)?;
        state.set_recurrent_tensor(next.reshape(self.state_shape)?.force_contiguous()?)?;
        Ok(Some(gated))
    }

    /// The mixer output `[N, hidden]` in `act` of the normed rows `x`;
    /// `states[r]` is request `r`'s state of this layer, advanced row by row
    /// (or, on CUDA, one long span through [`Self::chunked_span`]).
    fn forward(
        &self,
        x: &Tensor,
        rows: &[DeviceRow<'_>],
        states: &mut [&mut FishDrama3LinearState],
        phase: FishDrama3KernelPhase,
        act: DType,
        probe: &mut Probe<'_>,
    ) -> Result<Tensor> {
        let (c, vw) = (self.channels, self.value_width);
        let packed = self.qkvz.forward_device(x, phase, act)?;
        record(probe, || "qkvz".into(), &packed);
        let alpha = self.in_proj_a.forward_device(x, phase, DType::F32)?;
        let beta = self.in_proj_b.forward_device(x, phase, DType::F32)?;
        if self.chunked
            && phase == FishDrama3KernelPhase::Span
            && rows.len() == 1
            && rows[0].rows >= CHUNKED_MIN_ROWS
        {
            if let Some(gated) = self.chunked_span(&packed, &alpha, &beta, states[0])? {
                count_chunked_deltanet_span();
                record(probe, || "deltanet_core".into(), &gated);
                let out = self.out_proj.forward_device(&gated, phase, act)?;
                record(probe, || "deltanet_out".into(), &out);
                return Ok(out);
            }
        }
        let mut outputs = Vec::with_capacity(x.dim(0)?);
        let mut t = 0;
        for (row, state) in rows.iter().zip(states.iter_mut()) {
            for _ in 0..row.rows {
                let at = packed.narrow(0, t, 1)?;
                let mixed = at.narrow(1, 0, c)?.contiguous()?;
                let z = at.narrow(1, c, vw)?.to_dtype(DType::F32)?.contiguous()?;
                let (conv, history) = gdn::conv_decode(&mixed, &self.conv, state.conv.columns()?)?;
                let (beta, alpha) = (
                    beta.narrow(0, t, 1)?.contiguous()?,
                    alpha.narrow(0, t, 1)?.contiguous()?,
                );
                let (y, next) = if self.fused {
                    gdn::recurrent_decode(
                        &conv,
                        &z,
                        &beta,
                        &alpha,
                        &self.dt_bias,
                        &self.neg_a,
                        &self.norm,
                        state.recurrent(),
                        &self.spec,
                    )?
                } else {
                    self.recurrence(&conv, &z, &beta, &alpha, state.recurrent())?
                };
                state.conv.set_history(&history)?;
                state.set_recurrent_tensor(next.reshape(self.state_shape)?)?;
                outputs.push(y.reshape((1, vw))?);
                t += 1;
            }
        }
        let gated = Tensor::cat(&outputs, 0)?;
        record(probe, || "deltanet_core".into(), &gated);
        let out = self.out_proj.forward_device(&gated, phase, act)?;
        record(probe, || "deltanet_out".into(), &out);
        Ok(out)
    }
}

/// The sparse MoE block on the device.
pub(crate) struct DeviceMoe {
    /// `[experts, hidden]` F32.
    router: FishDrama3Linear,
    /// `[1, hidden]` F32.
    shared_gate: FishDrama3Linear,
    shared_gate_proj: FishDrama3Linear,
    shared_up: FishDrama3Linear,
    shared_down: FishDrama3Linear,
    /// `[experts, 2 intermediate, hidden]` E4M3 bytes, each expert's gate
    /// rows then its up rows.
    pub(crate) w13: Tensor,
    /// `[experts, 2 intermediate / 128, hidden / 128]` F32.
    pub(crate) s13: Tensor,
    /// `[experts, hidden, intermediate]` E4M3 bytes.
    pub(crate) w2: Tensor,
    /// `[experts, hidden / 128, intermediate / 128]` F32.
    pub(crate) s2: Tensor,
    spec: RouteSpec,
    hidden: usize,
    intermediate: usize,
    pub(crate) backend: FishDrama3MoeBackend,
    /// The grouped span kernels CUDA runs: the warp kernels until the load
    /// verified the tensor-core tiles (Metal has one form).
    pub(crate) grouped: GroupedKernel,
    forms: DeviceForms,
}

impl DeviceMoe {
    fn load(
        checkpoint: &FishDrama3Checkpoint,
        layer: usize,
        config: &FishDrama3Config,
        backend: FishDrama3MoeBackend,
        forms: DeviceForms,
        device: &Device,
    ) -> Result<Self> {
        let t = &config.trunk;
        let moe_config = t.moe;
        let (h, inter, experts) = (t.hidden, moe_config.intermediate, moe_config.experts);
        let block = config.fp8_block;
        let spec = RouteSpec {
            num_experts: experts,
            top_k: moe_config.top_k,
            shared: SharedSlot::None,
            shared_slot_id: 0,
            norm_topk: true,
        };
        if block != 128 || !moe::supported(device, forms.half, h, inter, spec.slots()) {
            return Err(load_error(format!(
                "the fused MoE takes 128×128 FP8 blocks over hidden and expert widths that are \
                 multiples of 128 and at most {} slots; this checkpoint has {block}-blocks, \
                 hidden {h}, width {inter}, top-{}",
                moe::MAX_SLOTS,
                moe_config.top_k
            )));
        }
        // Stacked one projection kind at a time: the host holds two thirds
        // of one layer's experts at most (`fish_drama3_memory`'s stacking
        // allowance is four thirds).
        let stack = |kinds: &[(&str, usize, usize)]| -> Result<(Tensor, Tensor)> {
            let rows = kinds.iter().map(|(_, rows, _)| rows).sum::<usize>();
            let cols = kinds[0].2;
            let mut codes = Vec::with_capacity(experts * rows * cols);
            let mut scales = Vec::with_capacity(experts * rows / block * cols / block);
            for e in 0..experts {
                for (name, rows, cols) in kinds {
                    let weight = checkpoint.fp8_block_weight(
                        &format!("{}.{name}", expert(layer, e)),
                        *rows,
                        *cols,
                        block,
                    )?;
                    codes.extend_from_slice(weight.codes);
                    scales.extend_from_slice(&weight.scales);
                }
            }
            Ok((
                Tensor::from_vec(codes, (experts, rows, cols), device)?,
                Tensor::from_vec(scales, (experts, rows / block, cols / block), device)?,
            ))
        };
        let (w13, s13) = stack(&[("gate_proj", inter, h), ("up_proj", inter, h)])?;
        let (w2, s2) = stack(&[("down_proj", h, inter)])?;
        let m = format!("{}.mlp", trunk_layer(layer));
        let s = format!("{m}.shared_expert");
        let si = moe_config.shared_intermediate;
        Ok(Self {
            router: f32_linear(checkpoint, &format!("{m}.gate.weight"), experts, h, device)?,
            shared_gate: f32_linear(
                checkpoint,
                &format!("{m}.shared_expert_gate.weight"),
                1,
                h,
                device,
            )?,
            shared_gate_proj: half_linear(
                checkpoint,
                &format!("{s}.gate_proj.weight"),
                si,
                h,
                forms,
                device,
            )?,
            shared_up: half_linear(
                checkpoint,
                &format!("{s}.up_proj.weight"),
                si,
                h,
                forms,
                device,
            )?,
            shared_down: half_linear(
                checkpoint,
                &format!("{s}.down_proj.weight"),
                h,
                si,
                forms,
                device,
            )?,
            w13,
            s13,
            w2,
            s2,
            spec,
            hidden: h,
            intermediate: inter,
            backend,
            grouped: GroupedKernel::Warp,
            forms,
        })
    }

    fn resident_bytes(&self) -> usize {
        [
            &self.router,
            &self.shared_gate,
            &self.shared_gate_proj,
            &self.shared_up,
            &self.shared_down,
        ]
        .iter()
        .map(|l| l.resident_bytes())
        .sum::<usize>()
            + self.w13.elem_count()
            + self.w2.elem_count()
            + (self.s13.elem_count() + self.s2.elem_count()) * 4
    }

    pub(crate) fn intermediate(&self) -> usize {
        self.intermediate
    }

    /// The router's F32 logits `[N, experts]` of the normed rows `x`.
    fn router_logits(&self, x: &Tensor, phase: FishDrama3KernelPhase) -> Result<Tensor> {
        self.router.forward_device(x, phase, DType::F32)
    }

    /// `[N, 2 top_k]` F32 routing of the normed rows `x`: expert ids, then
    /// the renormalised weights rounded to BF16.
    fn routing(&self, x: &Tensor, phase: FishDrama3KernelPhase) -> Result<Tensor> {
        let logits = self.router_logits(x, phase)?;
        let routing = moe::route(&logits, &self.spec)?;
        let k = self.spec.top_k;
        let weights = routing
            .narrow(1, k, k)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        Ok(Tensor::cat(&[&routing.narrow(1, 0, k)?, &weights], 1)?.contiguous()?)
    }

    /// The block's output `[N, hidden]` F32 for the normed rows `x`.
    fn forward(
        &self,
        x: &Tensor,
        phase: FishDrama3KernelPhase,
        act: DType,
        probe: &mut Probe<'_>,
    ) -> Result<Tensor> {
        let n = x.dim(0)?;
        let routing = self.routing(x, phase)?;
        let k = self.spec.top_k;
        let routed = match self.backend {
            FishDrama3MoeBackend::Fused => {
                let x16 = x.to_dtype(self.forms.half)?.contiguous()?;
                // Never in a step: a step's rows must not share a kernel
                // choice that depends on how many there are.
                let y = if phase == FishDrama3KernelPhase::Span
                    && n >= moe::GROUPED_MIN_TOKENS
                    && moe::grouped_supported(self.hidden, self.intermediate, self.spec.num_experts)
                {
                    count_grouped_moe_span();
                    moe::fp8_moe_grouped_with(
                        &x16,
                        &routing,
                        k,
                        &self.w13,
                        &self.s13,
                        &self.w2,
                        &self.s2,
                        self.grouped,
                    )?
                } else {
                    let activations = moe::fp8_gate_up(&x16, &routing, k, &self.w13, &self.s13)?;
                    record(probe, || "expert_activations".into(), &activations);
                    moe::fp8_down(&activations, &routing, k, &self.w2, &self.s2)?
                };
                record(probe, || "routed".into(), &y);
                y.to_dtype(DType::F32)?
            }
            FishDrama3MoeBackend::Loop => {
                self.expert_loop(&x.to_dtype(self.forms.loop_activations())?, &routing)?
            }
        };
        let gate = self.shared_gate_proj.forward_device(x, phase, act)?;
        let up = self.shared_up.forward_device(x, phase, act)?;
        let activations = swiglu(&gate, &up, act)?;
        record(probe, || "shared_activations".into(), &activations);
        let shared = self.shared_down.forward_device(&activations, phase, act)?;
        record(probe, || "shared_out".into(), &shared);
        let shared_gate =
            candle_nn::ops::sigmoid(&self.shared_gate.forward_device(x, phase, DType::F32)?)?;
        Ok((routed + shared.to_dtype(DType::F32)?.broadcast_mul(&shared_gate)?)?)
    }

    /// Host routing and one FP8 GEMV pair per expert over its rows in `x`'s
    /// dtype (F32 on Metal, BF16 on CUDA); the outputs combine in F32 in
    /// rank order as the CPU's do.
    fn expert_loop(&self, x: &Tensor, routing: &Tensor) -> Result<Tensor> {
        let (n, h, inter) = (x.dim(0)?, self.hidden, self.intermediate);
        let k = self.spec.top_k;
        let device = x.device();
        let host = routing.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let mut pairs = vec![Vec::new(); self.spec.num_experts];
        for (row, values) in host.iter().enumerate() {
            for rank in 0..k {
                let e = values[rank] as usize;
                let list = pairs.get_mut(e).ok_or_else(|| {
                    Error::InferenceError(format!("the router picked expert {e}"))
                })?;
                list.push((row, rank));
            }
        }
        let mut outputs = Vec::new();
        let mut order = vec![0u32; n * k];
        let mut at = 0usize;
        for (e, list) in pairs.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let index = Tensor::from_vec(
                list.iter().map(|(row, _)| *row as u32).collect::<Vec<_>>(),
                list.len(),
                device,
            )?;
            let rows = x.index_select(&index, 0)?.contiguous()?;
            let gate_up = crate::kernels::cuda::fp8::block_fp8_projection(
                &rows,
                &self.w13.get(e)?,
                &self.s13.get(e)?,
            )?;
            let activations = swiglu(
                &gate_up.narrow(1, 0, inter)?,
                &gate_up.narrow(1, inter, inter)?,
                x.dtype(),
            )?
            .contiguous()?;
            outputs.push(crate::kernels::cuda::fp8::block_fp8_projection(
                &activations,
                &self.w2.get(e)?,
                &self.s2.get(e)?,
            )?);
            for (j, (row, rank)) in list.iter().enumerate() {
                order[row * k + rank] = (at + j) as u32;
            }
            at += list.len();
        }
        let y = Tensor::cat(&outputs, 0)?
            .to_dtype(DType::F32)?
            .index_select(&Tensor::from_vec(order, n * k, device)?, 0)?
            .reshape((n, k, h))?;
        let weights = Tensor::from_vec(
            host.iter()
                .flat_map(|values| values[k..2 * k].to_vec())
                .collect::<Vec<_>>(),
            (n, k, 1),
            device,
        )?;
        let mut out = y
            .narrow(1, 0, 1)?
            .broadcast_mul(&weights.narrow(1, 0, 1)?)?;
        for rank in 1..k {
            out = (out
                + y.narrow(1, rank, 1)?
                    .broadcast_mul(&weights.narrow(1, rank, 1)?)?)?;
        }
        Ok(out.reshape((n, h))?)
    }
}

pub(crate) enum DeviceMixer {
    Attention {
        slot: usize,
        attention: DeviceAttention,
    },
    DeltaNet {
        slot: usize,
        net: DeviceDeltaNet,
    },
}

pub(crate) struct DeviceLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    pub(crate) mixer: DeviceMixer,
    pub(crate) moe: DeviceMoe,
    /// F16, or F32 once the range probe breached the layer.
    pub(crate) act: DType,
}

impl DeviceLayer {
    fn resident_bytes(&self) -> usize {
        let mixer = match &self.mixer {
            DeviceMixer::Attention { attention, .. } => attention.resident_bytes(),
            DeviceMixer::DeltaNet { net, .. } => net.resident_bytes(),
        };
        mixer
            + self.moe.resident_bytes()
            + (self.input_norm.elem_count() + self.post_norm.elem_count()) * 4
    }
}

/// The trunk's weights on the device.
pub(crate) struct DeviceTrunk {
    pub(crate) layers: Vec<DeviceLayer>,
    final_norm: Tensor,
    /// The final norm's dtype (F32 once Metal's probe breached it).
    pub(crate) final_act: DType,
    eps: f32,
    pub(crate) forms: DeviceForms,
}

impl DeviceTrunk {
    pub(crate) fn load(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        moe_backend: FishDrama3MoeBackend,
        device: &Device,
    ) -> Result<Self> {
        let forms = DeviceForms::of(device)?;
        let t = &config.trunk;
        let h = t.hidden;
        let mut attention_slots = 0;
        let mut linear_slots = 0;
        let layers = t
            .layers
            .iter()
            .enumerate()
            .map(|(layer, kind)| {
                let l = trunk_layer(layer);
                let mixer = match kind {
                    FishDrama3LayerKind::FullAttention => {
                        attention_slots += 1;
                        let a = t.attention;
                        let x = format!("{l}.self_attn");
                        let read = |name: &str, rows: usize, cols: usize| {
                            checkpoint.bf16_to_f32(&format!("{x}.{name}.weight"), &[rows, cols])
                        };
                        let gain_of = |name: &str| {
                            norm_gain(
                                checkpoint,
                                &format!("{x}.{name}.weight"),
                                a.head_dim,
                                NormGain::ZeroCentred,
                            )
                        };
                        DeviceMixer::Attention {
                            slot: attention_slots - 1,
                            // `q_proj` is already `[q_h | gate_h]` per head.
                            attention: DeviceAttention::new(
                                read("q_proj", 2 * a.q_width(), h)?,
                                &read("k_proj", a.kv_width(), h)?,
                                &read("v_proj", a.kv_width(), h)?,
                                read("o_proj", h, a.q_width())?,
                                &gain_of("q_norm")?,
                                &gain_of("k_norm")?,
                                a,
                                t.rms_norm_eps as f32,
                                forms,
                                device,
                            )?,
                        }
                    }
                    FishDrama3LayerKind::LinearAttention => {
                        linear_slots += 1;
                        DeviceMixer::DeltaNet {
                            slot: linear_slots - 1,
                            net: DeviceDeltaNet::load(checkpoint, layer, config, forms, device)?,
                        }
                    }
                };
                Ok(DeviceLayer {
                    input_norm: gain(
                        checkpoint,
                        &format!("{l}.input_layernorm.weight"),
                        h,
                        NormGain::ZeroCentred,
                        device,
                    )?,
                    post_norm: gain(
                        checkpoint,
                        &format!("{l}.post_attention_layernorm.weight"),
                        h,
                        NormGain::ZeroCentred,
                        device,
                    )?,
                    mixer,
                    moe: DeviceMoe::load(checkpoint, layer, config, moe_backend, forms, device)?,
                    act: forms.half,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            final_norm: gain(
                checkpoint,
                &format!("{TRUNK_PREFIX}.norm.weight"),
                h,
                NormGain::ZeroCentred,
                device,
            )?,
            final_act: forms.half,
            eps: t.rms_norm_eps as f32,
            forms,
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(DeviceLayer::resident_bytes)
            .sum::<usize>()
            + self.final_norm.elem_count() * 4
    }

    /// Layer `index` over the residual `x` (`[N, hidden]`, F32 on Metal and
    /// BF16 on CUDA, `rows` in order), with `act` in place of the layer's
    /// own when given; returns the residual after it. `states[r]` holds request `r`'s DeltaNet
    /// states. `trace` records the layer as the CPU's traced forwards do.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn layer(
        &self,
        index: usize,
        x: &Tensor,
        rows: &mut [DeviceRow<'_>],
        states: &mut [&mut [FishDrama3LinearState]],
        phase: FishDrama3KernelPhase,
        act: Option<DType>,
        probe: &mut Probe<'_>,
        trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        let layer = &self.layers[index];
        let act = act.unwrap_or(layer.act);
        let normed = rms_norm(x, &layer.input_norm, self.eps, act)?;
        record(probe, || "input_normed".into(), &normed);
        let mixed = match &layer.mixer {
            DeviceMixer::Attention { slot, attention } => {
                attention.forward(&normed, rows, *slot, phase, act, probe)?
            }
            DeviceMixer::DeltaNet { slot, net } => {
                let mut layer_states = states
                    .iter_mut()
                    .map(|states| {
                        states.get_mut(*slot).ok_or_else(|| {
                            Error::InferenceError("a DeltaNet state is missing".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                net.forward(&normed, rows, &mut layer_states, phase, act, probe)?
            }
        };
        let mixed = mixed.to_dtype(DType::F32)?;
        let x = add_residual(x, &mixed)?;
        let normed = rms_norm(&x, &layer.post_norm, self.eps, act)?;
        record(probe, || "post_normed".into(), &normed);
        let moe = layer.moe.forward(&normed, phase, act, probe)?;
        let x = add_residual(&x, &moe)?;
        record(probe, || "residual".into(), &x);
        if let Some(trace) = trace {
            let host = |t: &Tensor| -> Result<Vec<f32>> {
                Ok(t.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?)
            };
            trace
                .router_logits
                .push(host(&layer.moe.router_logits(&normed, phase)?)?);
            trace.mixers.push(host(&mixed)?);
            trace.moes.push(host(&moe)?);
            trace.outputs.push(host(&x)?);
        }
        Ok(x)
    }

    /// The final norm of the residual `x`, F32.
    pub(crate) fn final_norm(&self, x: &Tensor, act: Option<DType>) -> Result<Tensor> {
        Ok(
            rms_norm(x, &self.final_norm, self.eps, act.unwrap_or(self.final_act))?
                .to_dtype(DType::F32)?,
        )
    }

    /// Every layer over the embeddings `x` (`[N, hidden]`, taken into the
    /// residual's dtype), then the final norm: the final-norm hidden `[N,
    /// hidden]` F32.
    pub(crate) fn run(
        &self,
        x: Tensor,
        rows: &mut [DeviceRow<'_>],
        states: &mut [&mut [FishDrama3LinearState]],
        phase: FishDrama3KernelPhase,
        mut trace: Option<&mut LayerTrace>,
    ) -> Result<Tensor> {
        let mut x = x.to_dtype(self.forms.residual)?;
        for index in 0..self.layers.len() {
            x = self.layer(
                index,
                &x,
                rows,
                states,
                phase,
                None,
                &mut None,
                trace.as_deref_mut(),
            )?;
        }
        self.final_norm(&x, None)
    }

    /// The layers' activation dtypes and expert backends, for diagnostics.
    pub(crate) fn layer_forms(&self) -> Vec<(DType, FishDrama3MoeBackend)> {
        self.layers.iter().map(|l| (l.act, l.moe.backend)).collect()
    }
}

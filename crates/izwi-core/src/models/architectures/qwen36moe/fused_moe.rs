//! Device-routed fused execution for the Qwen3.6-MoE sparse block.
//!
//! The per-expert loop in `SparseMoeDispatcher` reads the top-k routing back to
//! the host on every layer, then runs about ten small ops per selected expert.
//! On CUDA that is 80 blocking syncs and about 3,200 launches per decode token
//! for Qwen3.6-35B-A3B.
//!
//! At load this module stacks a layer's block-FP8 experts into contiguous
//! `w13`/`w2` tensors. The per-expert projections the legacy path uses become
//! views into them, so residency does not grow. The block then runs in four
//! launches: the router GEMM, [`moe::route`], [`moe::fp8_gate_up`] and
//! [`moe::fp8_down`].
//!
//! Qwen3.6's always-on shared expert has the routed width, so it is stacked as
//! one extra expert. Its gate row is appended to the router, and it is folded
//! into the combine with weight `sigmoid(shared_expert_gate · x)`.
//!
//! Selection: `IZWI_QWEN36_MOE_BACKEND=legacy` keeps the per-expert loop and
//! does not stack anything; any other value means `auto`. In `auto`, every
//! stacked block runs [`Qwen36MoeFusedExperts::self_check`] at load against the
//! portable router reference and the per-expert projections. A failure keeps
//! the block on the legacy path and is reported in the runtime diagnostics.
//!
//! The fused path does not feed `ExpertActivationCounters`: recording them
//! would need the device-to-host read this path exists to remove.

use candle_core::{DType, Device, Tensor};
use candle_nn::ops;

use crate::error::{Error, Result};
use crate::kernels::cuda::moe::{self, RouteSpec, SharedSlot};
use crate::models::architectures::qwen36moe::fast_path::{compare_values, legacy_value};
use crate::models::architectures::qwen36moe::sparse::{
    Qwen36MoeExpertWeights, Qwen36MoeLinear, Qwen36MoeSharedExpertWeights,
};
use crate::models::architectures::qwen36moe::text::Qwen36MoeFfnGeometry;

/// Environment switch for the sparse-expert execution path.
pub(crate) const BACKEND_ENV: &str = "IZWI_QWEN36_MOE_BACKEND";
const BLOCK: usize = 128;

/// Requested sparse-expert execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen36MoeBackendRequest {
    /// Fused kernels when the weights, device and self-check allow it.
    Auto,
    /// Always the host-routed per-expert loop; no expert stacking.
    Legacy,
}

impl Qwen36MoeBackendRequest {
    pub(crate) fn from_env() -> Self {
        Self::parse(std::env::var(BACKEND_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        if legacy_value(value) {
            Self::Legacy
        } else {
            Self::Auto
        }
    }
}

/// One layer's experts stacked for the fused kernels.
pub(crate) struct Qwen36MoeFusedExperts {
    /// F32 router rows, plus the shared-expert gate row when it is folded in.
    router: Tensor,
    spec: RouteSpec,
    hidden: usize,
    inter: usize,
    w13: Tensor,
    s13: Tensor,
    w2: Tensor,
    s2: Tensor,
    /// Expert-major kernels for prefill-sized calls (see
    /// [`moe::GROUPED_MIN_TOKENS`]); verified by the self-check.
    grouped: bool,
}

/// Result of [`Qwen36MoeFusedExperts::stack`]. Unsupported layouts hand the
/// original weights back unchanged.
pub(crate) enum Qwen36MoeStacking {
    Stacked {
        fused: Qwen36MoeFusedExperts,
        /// Per-expert views into the stacked tensors, for the legacy path.
        experts: Vec<Qwen36MoeExpertWeights>,
        shared: Qwen36MoeSharedExpertWeights,
    },
    Unsupported {
        experts: Vec<Qwen36MoeExpertWeights>,
        shared: Qwen36MoeSharedExpertWeights,
        reason: String,
    },
}

fn compact(linear: &Qwen36MoeLinear) -> Option<(&Tensor, &Tensor)> {
    match linear {
        Qwen36MoeLinear::CompactFp8 { weights, scales } => Some((weights, scales)),
        _ => None,
    }
}

/// Whether `linear` is a compact block-FP8 projection of `[rows, cols]`.
fn compact_of(linear: &Qwen36MoeLinear, rows: usize, cols: usize, device: &Device) -> bool {
    compact(linear).is_some_and(|(weights, scales)| {
        weights.dtype() == DType::U8
            && scales.dtype() == DType::F32
            && weights.dims() == [rows, cols]
            && scales.dims() == [rows / BLOCK, cols / BLOCK]
            && weights.device().same_device(device)
            && scales.device().same_device(device)
    })
}

fn dense_f32_of(linear: &Qwen36MoeLinear, rows: usize, cols: usize) -> Option<&Tensor> {
    match linear {
        Qwen36MoeLinear::Dense(tensor)
            if tensor.dtype() == DType::F32 && tensor.dims() == [rows, cols] =>
        {
            Some(tensor)
        }
        _ => None,
    }
}

/// Stack the weight (or scale) halves of `(weights, scales)` pairs on a new
/// leading expert axis.
fn stack_part(pairs: &[(&Tensor, &Tensor)], scales: bool) -> candle_core::Result<Tensor> {
    let parts = pairs
        .iter()
        .map(|(weights, block_scales)| {
            if scales {
                (*block_scales).clone()
            } else {
                (*weights).clone()
            }
        })
        .collect::<Vec<_>>();
    Tensor::stack(&parts, 0)
}

fn compact_view(weights: &Tensor, scales: &Tensor) -> Qwen36MoeLinear {
    Qwen36MoeLinear::CompactFp8 {
        weights: weights.clone(),
        scales: scales.clone(),
    }
}

impl Qwen36MoeFusedExperts {
    /// Stack a layer's experts when every routed projection is 128x128
    /// block-FP8 at the declared geometry and the device can run the kernels.
    pub(crate) fn stack(
        router: &Qwen36MoeLinear,
        experts: Vec<Qwen36MoeExpertWeights>,
        shared: Qwen36MoeSharedExpertWeights,
        geometry: &Qwen36MoeFfnGeometry,
    ) -> Qwen36MoeStacking {
        let unsupported = |experts, shared, reason: &str| Qwen36MoeStacking::Unsupported {
            experts,
            shared,
            reason: reason.to_string(),
        };
        let inter = geometry.expert_intermediate_size;
        let Some((first, _)) = experts.first().and_then(|expert| compact(&expert.gate)) else {
            return unsupported(experts, shared, "routed experts are not block-FP8 resident");
        };
        let (Ok((_, hidden)), device) = (first.dims2(), first.device().clone()) else {
            return unsupported(experts, shared, "routed expert weights are not 2-D");
        };
        let uniform = experts.len() == geometry.num_experts
            && experts.iter().all(|expert| {
                compact_of(&expert.gate, inter, hidden, &device)
                    && compact_of(&expert.up, inter, hidden, &device)
                    && compact_of(&expert.down, hidden, inter, &device)
            });
        if !uniform {
            return unsupported(
                experts,
                shared,
                "routed experts are not uniform 128x128 block-FP8 at the declared geometry",
            );
        }
        let Some(router_rows) = dense_f32_of(router, geometry.num_experts, hidden) else {
            return unsupported(
                experts,
                shared,
                "router is not a dense F32 [experts, hidden] projection",
            );
        };
        let shared_fits = compact_of(&shared.gate, inter, hidden, &device)
            && compact_of(&shared.up, inter, hidden, &device)
            && compact_of(&shared.down, hidden, inter, &device);
        let gate_row = shared
            .output_gate
            .as_ref()
            .map(|gate| dense_f32_of(gate, 1, hidden));
        let shared_slot = match (shared_fits, gate_row) {
            (true, None) => SharedSlot::Ungated,
            (true, Some(Some(_))) => SharedSlot::Gated,
            _ => SharedSlot::None,
        };
        let spec = RouteSpec {
            num_experts: geometry.num_experts,
            top_k: geometry.num_experts_per_tok,
            shared: shared_slot,
            shared_slot_id: geometry.num_experts,
            norm_topk: true,
        };
        let activation = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        if spec.top_k == 0
            || spec.top_k > spec.num_experts
            || spec.num_experts > moe::MAX_EXPERTS
            || !moe::supported(&device, activation, hidden, inter, spec.slots())
        {
            return unsupported(
                experts,
                shared,
                "device, dtype or geometry is unsupported by the fused MoE kernels",
            );
        }

        let built = Self::build(
            router_rows,
            gate_row.flatten(),
            &experts,
            &shared,
            spec,
            inter,
        );
        let (router_tensor, w13, s13, w2, s2, views, shared_view) = match built {
            Ok(built) => built,
            Err(error) => {
                return Qwen36MoeStacking::Unsupported {
                    experts,
                    shared,
                    reason: format!("expert stacking failed: {error}"),
                }
            }
        };
        // Dropping the originals releases the per-expert allocations; the
        // legacy path now reads views into the stacked tensors.
        drop(experts);
        let shared = match shared_view {
            Some(view) => Qwen36MoeSharedExpertWeights {
                gate: view.gate,
                up: view.up,
                down: view.down,
                output_gate: shared.output_gate,
            },
            None => shared,
        };
        Qwen36MoeStacking::Stacked {
            fused: Self {
                grouped: moe::grouped_supported(hidden, inter, w13.dim(0).unwrap_or(0)),
                router: router_tensor,
                spec,
                hidden,
                inter,
                w13,
                s13,
                w2,
                s2,
            },
            experts: views,
            shared,
        }
    }

    /// Build the fused router, the stacked tensors and the per-expert views
    /// without consuming the originals, so a failure leaves them usable.
    #[allow(clippy::type_complexity)]
    fn build(
        router_rows: &Tensor,
        gate_row: Option<&Tensor>,
        experts: &[Qwen36MoeExpertWeights],
        shared: &Qwen36MoeSharedExpertWeights,
        spec: RouteSpec,
        inter: usize,
    ) -> candle_core::Result<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<Qwen36MoeExpertWeights>,
        Option<Qwen36MoeExpertWeights>,
    )> {
        let router = match (spec.shared, gate_row) {
            (SharedSlot::Gated, Some(gate)) => Tensor::cat(&[router_rows, gate], 0)?,
            _ => router_rows.clone(),
        };
        let fold_shared = spec.shared != SharedSlot::None;
        let (w13, s13, w2, s2) = Self::stack_tensors(experts, shared, fold_shared)?;
        let views = (0..experts.len())
            .map(|expert| Self::expert_view(&w13, &s13, &w2, &s2, expert, inter))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let shared_view = fold_shared
            .then(|| Self::expert_view(&w13, &s13, &w2, &s2, experts.len(), inter))
            .transpose()?;
        Ok((router, w13, s13, w2, s2, views, shared_view))
    }

    /// `[S, 2*inter, hidden]` / `[S, 2*inter/128, hidden/128]` gate+up and
    /// `[S, hidden, inter]` / `[S, hidden/128, inter/128]` down, S = experts
    /// (+1 for the folded shared expert, stacked last).
    fn stack_tensors(
        experts: &[Qwen36MoeExpertWeights],
        shared: &Qwen36MoeSharedExpertWeights,
        fold_shared: bool,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor)> {
        let mut gates = Vec::new();
        let mut ups = Vec::new();
        let mut downs = Vec::new();
        let shared_view = fold_shared.then_some((&shared.gate, &shared.up, &shared.down));
        let routed = experts
            .iter()
            .map(|expert| (&expert.gate, &expert.up, &expert.down));
        for (gate, up, down) in routed.chain(shared_view) {
            let (Some(gate), Some(up), Some(down)) = (compact(gate), compact(up), compact(down))
            else {
                candle_core::bail!("expert projection is not block-FP8")
            };
            gates.push(gate);
            ups.push(up);
            downs.push(down);
        }
        let w13 = Tensor::cat(&[stack_part(&gates, false)?, stack_part(&ups, false)?], 1)?;
        let s13 = Tensor::cat(&[stack_part(&gates, true)?, stack_part(&ups, true)?], 1)?;
        let w2 = stack_part(&downs, false)?;
        let s2 = stack_part(&downs, true)?;
        Ok((w13, s13, w2, s2))
    }

    fn expert_view(
        w13: &Tensor,
        s13: &Tensor,
        w2: &Tensor,
        s2: &Tensor,
        expert: usize,
        inter: usize,
    ) -> candle_core::Result<Qwen36MoeExpertWeights> {
        let (w, s) = (w13.get(expert)?, s13.get(expert)?);
        let rows = inter / BLOCK;
        Ok(Qwen36MoeExpertWeights {
            gate: compact_view(&w.narrow(0, 0, inter)?, &s.narrow(0, 0, rows)?),
            up: compact_view(&w.narrow(0, inter, inter)?, &s.narrow(0, rows, rows)?),
            down: compact_view(&w2.get(expert)?, &s2.get(expert)?),
        })
    }

    /// Whether this call can take the fused path (non-empty input on the
    /// stacked device in a dtype the kernels accept).
    pub(crate) fn accepts(&self, flat: &Tensor) -> bool {
        let dtype_ok = match flat.device() {
            Device::Cpu => matches!(flat.dtype(), DType::F32 | DType::F16 | DType::BF16),
            Device::Cuda(_) => matches!(flat.dtype(), DType::F16 | DType::BF16),
            _ => false,
        };
        dtype_ok
            && flat.dim(0).is_ok_and(|tokens| tokens > 0)
            && flat.dim(1).is_ok_and(|hidden| hidden == self.hidden)
            && flat.device().same_device(self.w13.device())
    }

    /// Whether the shared expert is part of the fused combine; when false the
    /// caller adds the legacy shared-expert output.
    pub(crate) fn folds_shared(&self) -> bool {
        self.spec.shared != SharedSlot::None
    }

    /// Routed (plus folded shared) output for `flat` `[tokens, hidden]`.
    pub(crate) fn forward(&self, flat: &Tensor) -> Result<Tensor> {
        let logits = moe::router_logits(flat, &self.router)?;
        let routing = moe::route(&logits, &self.spec)?;
        self.combine(flat, &routing)
    }

    fn combine(&self, flat: &Tensor, routing: &Tensor) -> Result<Tensor> {
        let slots = self.spec.slots();
        if self.grouped && flat.dim(0)? >= moe::GROUPED_MIN_TOKENS {
            return moe::fp8_moe_grouped(
                flat, routing, slots, &self.w13, &self.s13, &self.w2, &self.s2,
            )
            .map_err(Error::from);
        }
        self.combine_per_pair(flat, routing)
    }

    fn combine_per_pair(&self, flat: &Tensor, routing: &Tensor) -> Result<Tensor> {
        let slots = self.spec.slots();
        let act = moe::fp8_gate_up(flat, routing, slots, &self.w13, &self.s13)?;
        moe::fp8_down(&act, routing, slots, &self.w2, &self.s2).map_err(Error::from)
    }

    /// Check the fused kernels on this device against independent references
    /// before serving with them:
    ///
    /// 1. the router GEMV against the dense router projection, then the device
    ///    router against the portable router on the same logits (identical
    ///    ids, weights within 1e-4);
    /// 2. the grouped expert kernels against `Σ weight · expert(x)`, computed
    ///    with the per-expert projections the legacy path uses, under the same
    ///    routing.
    ///
    /// Comparing the two halves separately avoids false failures from near-tie
    /// top-k selections, which BF16 router logits make common.
    pub(crate) fn self_check(
        &self,
        experts: &[Qwen36MoeExpertWeights],
        shared: &Qwen36MoeSharedExpertWeights,
    ) -> std::result::Result<(), String> {
        self.run_self_check(experts, shared)
            .map_err(|error| error.to_string())
    }

    fn run_self_check(
        &self,
        experts: &[Qwen36MoeExpertWeights],
        shared: &Qwen36MoeSharedExpertWeights,
    ) -> Result<()> {
        let device = self.w13.device().clone();
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let slots = self.spec.slots();
        for tokens in [1usize, 5] {
            let values = (0..tokens * self.hidden)
                .map(|i| {
                    let i = i as f32;
                    (i * 0.754_877_7).sin() * 1.5 + (i * 0.031).cos() * 0.25
                })
                .collect::<Vec<_>>();
            let x = Tensor::from_vec(values, (tokens, self.hidden), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?;
            // Router GEMV against the dense projection it replaces.
            let logits = moe::router_logits(&x, &self.router)?;
            let projected = Qwen36MoeLinear::from_dense(self.router.clone())
                .project(&x)?
                .to_dtype(DType::F32)?;
            compare_values(
                "fused MoE router logits",
                &logits
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?,
                &projected
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?,
                0.01,
                0.02,
            )?;
            let routing = moe::route(&logits, &self.spec)?;
            let actual = routing.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let expected =
                moe::route(&logits.to_device(&Device::Cpu)?, &self.spec)?.to_vec2::<f32>()?;
            for (token, (got, want)) in actual.iter().zip(&expected).enumerate() {
                let weights_close = got[slots..]
                    .iter()
                    .zip(&want[slots..])
                    .all(|(a, b)| (a - b).abs() <= 1e-4);
                if got[..slots] != want[..slots] || !weights_close {
                    return Err(Error::InferenceError(format!(
                        "router mismatch at token {token} (T={tokens}): {got:?} vs reference {want:?}"
                    )));
                }
            }

            let fused = self
                .combine(&x, &routing)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let mut reference = vec![vec![0f32; self.hidden]; tokens];
            for (token, row) in actual.iter().enumerate() {
                let xt = x.narrow(0, token, 1)?;
                for slot in 0..slots {
                    let expert = row[slot] as usize;
                    let output = if expert < experts.len() {
                        expert_forward(
                            &experts[expert].gate,
                            &experts[expert].up,
                            &experts[expert].down,
                            &xt,
                        )?
                    } else {
                        expert_forward(&shared.gate, &shared.up, &shared.down, &xt)?
                    };
                    let output = output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for (acc, value) in reference[token].iter_mut().zip(&output) {
                        *acc += row[slots + slot] * value;
                    }
                }
            }
            compare(&fused, &reference, tokens)?;
        }
        if self.grouped {
            // Prefill-sized call: the expert-major kernels against the
            // per-pair GEMVs verified above, on the same routing.
            let tokens = moe::GROUPED_MIN_TOKENS + 16;
            let values = (0..tokens * self.hidden)
                .map(|i| ((i as f32) * 0.618_034).sin() * 1.5)
                .collect::<Vec<_>>();
            let x = Tensor::from_vec(values, (tokens, self.hidden), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?;
            let routing = moe::route(&moe::router_logits(&x, &self.router)?, &self.spec)?;
            let host = |t: Tensor| -> Result<Vec<f32>> {
                Ok(t.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?)
            };
            compare_values(
                "grouped MoE prefill",
                &host(self.combine(&x, &routing)?)?,
                &host(self.combine_per_pair(&x, &routing)?)?,
                0.01,
                0.02,
            )?;
        }
        Ok(())
    }
}

/// Ungated SwiGLU expert through the per-expert projections.
fn expert_forward(
    gate: &Qwen36MoeLinear,
    up: &Qwen36MoeLinear,
    down: &Qwen36MoeLinear,
    x: &Tensor,
) -> Result<Tensor> {
    let gate = gate.project(x)?;
    let up = up.project(x)?;
    down.project(&(ops::silu(&gate)? * up)?)
}

/// The per-expert reference rounds each projection to the activation dtype
/// while the fused path accumulates the combine in F32, so allow a 3% relative
/// L2 error and 8% of the largest reference magnitude per element. A wrong
/// expert, scale block or row produces errors near 100%.
fn compare(fused: &[Vec<f32>], reference: &[Vec<f32>], tokens: usize) -> Result<()> {
    compare_values(
        &format!("fused MoE (T={tokens})"),
        &fused.concat(),
        &reference.concat(),
        0.03,
        0.08,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_request_parses_legacy_and_defaults_to_auto() {
        assert_eq!(
            Qwen36MoeBackendRequest::parse(Some(" Legacy ")),
            Qwen36MoeBackendRequest::Legacy
        );
        assert_eq!(
            Qwen36MoeBackendRequest::parse(Some("auto")),
            Qwen36MoeBackendRequest::Auto
        );
        assert_eq!(
            Qwen36MoeBackendRequest::parse(None),
            Qwen36MoeBackendRequest::Auto
        );
    }

    #[test]
    fn compare_rejects_wrong_outputs_and_accepts_rounding_noise() {
        let reference = vec![vec![1.0f32, -2.0, 0.5, 4.0]];
        let noisy = vec![vec![1.004f32, -2.01, 0.498, 4.02]];
        assert!(compare(&noisy, &reference, 1).is_ok());
        let wrong = vec![vec![-1.0f32, 2.0, 0.5, 4.0]];
        assert!(compare(&wrong, &reference, 1).is_err());
        let nan = vec![vec![f32::NAN, -2.0, 0.5, 4.0]];
        assert!(compare(&nan, &reference, 1).is_err());
    }
}

//! The sparse MoE block every trunk layer runs, the CPU reference.
//!
//! ```text
//! p        = softmax(router(x))                     # F32 logits and softmax
//! ids, w   = top_k(p); w /= sum(w); w = bf16(w)     # renormalised, then rounded to BF16
//! routed   = sum_k w_k * down(silu(gate_k(x)) * up_k(x))   # F32 sum, in rank order
//! out      = routed + sigmoid(shared_gate · x) * shared(x)
//! ```
//!
//! Upstream casts the renormalised weights to the activation dtype (BF16)
//! before the experts use them; the rounding is part of the model's
//! numerics, so it is kept although the activations here are F32. Ties in
//! the top-k go to the lower expert index.
//!
//! The routed experts are FP8 E4M3 with F32 128×128 block scales, multiplied
//! (`W = e4m3 · scale`); a CPU load dequantizes each one exactly to F32 and
//! stores it in its class form (Q8_0). Experts run per expert over the rows
//! routed to them. `models::shared::moe` routes without the BF16 rounding
//! and scatters in expert order, so the block keeps its own loop.

use candle_core::Device;
use half::bf16;

use super::super::checkpoint::{expert, trunk_layer, FishDrama3Checkpoint};
use super::super::config::FishDrama3TrunkConfig;
use super::super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::super::residency::{FishDrama3ClassForms, FishDrama3ProjectionClass};
use super::{load_linear, sigmoid, silu};
use crate::error::{Error, Result};

/// One row's routing decision.
#[derive(Debug, Clone, PartialEq)]
pub struct FishDrama3Route {
    /// Expert ids, highest probability first.
    pub experts: Vec<usize>,
    /// Their renormalised weights, rounded to BF16 (held in F32).
    pub weights: Vec<f32>,
}

/// F32 router logits of one row: `x · router[e]` for every expert.
pub(crate) fn router_logits(router: &[f32], x: &[f32]) -> Vec<f32> {
    router
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect()
}

/// Softmax over every expert in F32, the top `top_k`, renormalised over
/// them, rounded to BF16.
pub fn route(logits: &[f32], top_k: usize) -> FishDrama3Route {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let total = exps.iter().sum::<f32>();
    let probabilities = exps.iter().map(|v| v / total).collect::<Vec<_>>();
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    // Stable: equal probabilities keep the lower index first.
    order.sort_by(|a, b| probabilities[*b].total_cmp(&probabilities[*a]));
    order.truncate(top_k);
    let kept = order.iter().map(|e| probabilities[*e]).collect::<Vec<_>>();
    let sum = kept.iter().sum::<f32>();
    FishDrama3Route {
        experts: order,
        weights: kept
            .iter()
            .map(|w| bf16::from_f32(w / sum).to_f32())
            .collect(),
    }
}

/// A SwiGLU expert: `down(silu(gate(x)) * up(x))`.
pub(crate) struct FishDrama3Expert {
    gate: FishDrama3Linear,
    up: FishDrama3Linear,
    down: FishDrama3Linear,
}

impl FishDrama3Expert {
    fn forward(&self, x: &[f32], m: usize, phase: FishDrama3KernelPhase) -> Result<Vec<f32>> {
        let gate = self.gate.forward_host(x, m, phase)?;
        let up = self.up.forward_host(x, m, phase)?;
        let act = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| silu(*g) * u)
            .collect::<Vec<_>>();
        self.down.forward_host(&act, m, phase)
    }

    fn resident_bytes(&self) -> usize {
        self.projections().iter().map(|l| l.resident_bytes()).sum()
    }

    pub(crate) fn projections(&self) -> [&FishDrama3Linear; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

pub(crate) struct FishDrama3Moe {
    /// `[experts, hidden]` F32.
    router: Vec<f32>,
    experts: Vec<FishDrama3Expert>,
    shared: FishDrama3Expert,
    /// `[hidden]` F32.
    shared_gate: Vec<f32>,
    top_k: usize,
    hidden: usize,
}

impl FishDrama3Moe {
    pub(crate) fn load(
        checkpoint: &FishDrama3Checkpoint,
        layer: usize,
        trunk: &FishDrama3TrunkConfig,
        forms: &FishDrama3ClassForms,
        block: usize,
        device: &Device,
    ) -> Result<Self> {
        let moe = trunk.moe;
        let h = trunk.hidden;
        let m = format!("{}.mlp", trunk_layer(layer));
        let routed = forms.form(FishDrama3ProjectionClass::RoutedExpert);
        let fp8 = |module: String, rows: usize, cols: usize| -> Result<FishDrama3Linear> {
            let weight = checkpoint.fp8_block_weight(&module, rows, cols, block)?;
            FishDrama3Linear::from_values(weight.dequantized(), rows, cols, routed, device)
        };
        let experts = (0..moe.experts)
            .map(|e| {
                let x = expert(layer, e);
                Ok(FishDrama3Expert {
                    gate: fp8(format!("{x}.gate_proj"), moe.intermediate, h)?,
                    up: fp8(format!("{x}.up_proj"), moe.intermediate, h)?,
                    down: fp8(format!("{x}.down_proj"), h, moe.intermediate)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_form = forms.form(FishDrama3ProjectionClass::SharedExpert);
        let s = format!("{m}.shared_expert");
        let si = moe.shared_intermediate;
        let shared = FishDrama3Expert {
            gate: load_linear(
                checkpoint,
                &format!("{s}.gate_proj.weight"),
                si,
                h,
                shared_form,
                device,
            )?,
            up: load_linear(
                checkpoint,
                &format!("{s}.up_proj.weight"),
                si,
                h,
                shared_form,
                device,
            )?,
            down: load_linear(
                checkpoint,
                &format!("{s}.down_proj.weight"),
                h,
                si,
                shared_form,
                device,
            )?,
        };
        Ok(Self {
            router: checkpoint.bf16_to_f32(&format!("{m}.gate.weight"), &[moe.experts, h])?,
            experts,
            shared,
            shared_gate: checkpoint
                .bf16_to_f32(&format!("{m}.shared_expert_gate.weight"), &[1, h])?,
            top_k: moe.top_k,
            hidden: h,
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.experts
            .iter()
            .map(|e| e.resident_bytes())
            .sum::<usize>()
            + self.shared.resident_bytes()
            + (self.router.len() + self.shared_gate.len()) * 4
    }

    pub(crate) fn expert(&self, index: usize) -> &FishDrama3Expert {
        &self.experts[index]
    }

    pub(crate) fn shared_expert(&self) -> &FishDrama3Expert {
        &self.shared
    }

    /// The router's F32 logits `[n, experts]` of the `n` normed rows `h`
    /// (traced forwards).
    pub(crate) fn router_logits_rows(&self, h: &[f32], n: usize) -> Vec<f32> {
        h.chunks_exact(self.hidden)
            .take(n)
            .flat_map(|x| router_logits(&self.router, x))
            .collect()
    }

    /// The routing of each of the `n` rows of `h`.
    pub(crate) fn routes(&self, h: &[f32], n: usize) -> Vec<FishDrama3Route> {
        h.chunks_exact(self.hidden)
            .take(n)
            .map(|x| route(&router_logits(&self.router, x), self.top_k))
            .collect()
    }

    /// The block's output `[n, hidden]` for the normed rows `h`.
    pub(crate) fn forward(
        &self,
        h: &[f32],
        n: usize,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        let hidden = self.hidden;
        if h.len() != n * hidden {
            return Err(Error::InferenceError(format!(
                "the MoE block got {} values for {n} rows of {hidden}",
                h.len()
            )));
        }
        let routes = self.routes(h, n);
        // Each (row, rank) pair's expert output, gathered per expert.
        let mut pairs = vec![Vec::new(); self.experts.len()];
        for (row, route) in routes.iter().enumerate() {
            for (rank, &e) in route.experts.iter().enumerate() {
                pairs[e].push((row, rank));
            }
        }
        let mut outputs = vec![0f32; n * self.top_k * hidden];
        for (e, rows) in pairs.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let x = rows
                .iter()
                .flat_map(|(row, _)| h[row * hidden..(row + 1) * hidden].iter().copied())
                .collect::<Vec<_>>();
            let y = self.experts[e].forward(&x, rows.len(), phase)?;
            for ((row, rank), values) in rows.iter().zip(y.chunks_exact(hidden)) {
                let at = (row * self.top_k + rank) * hidden;
                outputs[at..at + hidden].copy_from_slice(values);
            }
        }
        let shared = self.shared.forward(h, n, phase)?;
        let mut out = vec![0f32; n * hidden];
        for (row, route) in routes.iter().enumerate() {
            let o = &mut out[row * hidden..(row + 1) * hidden];
            for (rank, weight) in route.weights.iter().enumerate() {
                let at = (row * self.top_k + rank) * hidden;
                for (value, y) in o.iter_mut().zip(&outputs[at..at + hidden]) {
                    *value += y * weight;
                }
            }
            let x = &h[row * hidden..(row + 1) * hidden];
            let gate = sigmoid(x.iter().zip(&self.shared_gate).map(|(a, b)| a * b).sum());
            for (value, s) in o.iter_mut().zip(&shared[row * hidden..(row + 1) * hidden]) {
                *value += gate * s;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    /// F64 logits, F64 softmax, top-k, renormalised, then rounded to BF16.
    fn reference(
        x: &[f32],
        router: &[f32],
        top_k: usize,
        bf16_accumulation: bool,
    ) -> FishDrama3Route {
        let logits = router
            .chunks_exact(x.len())
            .map(|row| {
                if bf16_accumulation {
                    // RED: a BF16 running sum (and BF16 logits).
                    row.iter()
                        .zip(x)
                        .fold(bf16::ZERO, |acc, (w, v)| {
                            bf16::from_f32(acc.to_f32() + bf16::from_f32(w * v).to_f32())
                        })
                        .to_f32() as f64
                } else {
                    row.iter()
                        .zip(x)
                        .map(|(w, v)| f64::from(*w) * f64::from(*v))
                        .sum()
                }
            })
            .collect::<Vec<f64>>();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total = logits.iter().map(|v| (v - max).exp()).sum::<f64>();
        let p = logits
            .iter()
            .map(|v| (v - max).exp() / total)
            .collect::<Vec<_>>();
        let mut order = (0..p.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
        order.truncate(top_k);
        let sum = order.iter().map(|e| p[*e]).sum::<f64>();
        FishDrama3Route {
            weights: order
                .iter()
                .map(|e| bf16::from_f64(p[*e] / sum).to_f32())
                .collect(),
            experts: order,
        }
    }

    /// The largest weight difference in BF16 ulps at 1.0 (2^-8), counting a
    /// different expert set as unbounded.
    fn worst_ulps(a: &FishDrama3Route, b: &FishDrama3Route) -> f64 {
        let mut worst = 0f64;
        for (e, w) in a.experts.iter().zip(&a.weights) {
            match b.experts.iter().position(|x| x == e) {
                Some(at) => worst = worst.max(f64::from((w - b.weights[at]).abs()) * 256.0),
                None => return f64::INFINITY,
            }
        }
        worst
    }

    /// Softmax top-8 over 256 experts at the published width, renormalised
    /// and rounded to BF16: the F32 router agrees with F64 to within a BF16
    /// ulp; a BF16-accumulated router is many ulps off (or picks other
    /// experts).
    #[test]
    fn top_k_renormalises_then_rounds_to_bf16_and_bf16_accumulation_is_far_off() {
        let (hidden, experts, top_k) = (2048, 256, 8);
        let router = xorshift_values(experts * hidden, 21, (3.0 / hidden as f32).sqrt() * 2.0);
        let mut far = 0;
        for row in 0..16u64 {
            let x = xorshift_values(hidden, 100 + row, 1.0);
            let ours = route(&router_logits(&router, &x), top_k);
            let expected = reference(&x, &router, top_k, false);
            assert_eq!(ours.experts, expected.experts, "row {row}");
            assert!(worst_ulps(&ours, &expected) <= 1.0, "row {row}");
            // Rounded to BF16 after the renormalisation.
            assert!(ours
                .weights
                .iter()
                .all(|w| bf16::from_f32(*w).to_f32() == *w));
            let sum = ours.weights.iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 8.0 * f32::from(bf16::EPSILON), "{sum}");
            assert!(ours.weights.windows(2).all(|w| w[0] >= w[1]));
            let red = reference(&x, &router, top_k, true);
            if worst_ulps(&ours, &red) >= 8.0 {
                far += 1;
            }
        }
        assert!(far >= 12, "{far} of 16 rows");
    }

    #[test]
    fn ties_route_to_the_lower_expert_and_weights_renormalise() {
        let r = route(&[0.0, 1.0, 1.0, 0.5], 2);
        assert_eq!(r.experts, vec![1, 2]);
        assert_eq!(r.weights, vec![0.5, 0.5]);
        let r = route(&[2.0, 0.0, 1.0], 1);
        assert_eq!(
            (r.experts.as_slice(), r.weights.as_slice()),
            (&[0][..], &[1.0][..])
        );
    }
}

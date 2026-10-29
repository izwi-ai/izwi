//! The Nemotron-H mixture of experts: a sigmoid router with an F32
//! selection bias, relu² experts without a gate projection, and one
//! unscaled shared expert.
//!
//! ```text
//! logits = f32(x) @ f32(W_gate).T
//! s      = sigmoid(logits)
//! picked = top_k(s + e_score_correction_bias)     (selection only, F32)
//! w      = s[picked] / sum(s[picked]) * routed_scaling_factor
//! out    = sum_e w_e * down_e(relu(up_e x)^2) + down_s(relu(up_s x)^2)
//! ```
//!
//! The bias is about +40.7 for every expert, so the add must be F32: BF16
//! resolution at 40 is 0.25 and would reorder the experts. Routing runs on
//! the host (one read of `[tokens, experts]` F32 logits per layer); experts
//! run per expert over the tokens routed to them. Ties order by expert index.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::NemotronCheckpoint;
use super::super::config::{MoeConfig, NemotronTranscribeConfig};
use super::super::linear::{NtLinear, Pass};
use super::super::precision::Fp8Mode;
use super::super::speech_transformer::host_values;
use crate::error::{Error, Result};

/// One token's routing: expert ids and combine weights (scaling folded in).
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
    /// The gap between the last selected and the first rejected biased
    /// score: below about 1e-5 a different rounding may pick another
    /// expert (a router near-tie).
    pub margin: f32,
}

/// Routes one token from its F32 logits.
pub fn route(logits: &[f32], bias: &[f32], config: &MoeConfig) -> Routing {
    let scores = logits
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect::<Vec<f32>>();
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    let biased = scores
        .iter()
        .zip(bias)
        .map(|(s, b)| s + b)
        .collect::<Vec<f32>>();
    order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
    let experts = order[..config.top_k].to_vec();
    let total = experts.iter().map(|&e| scores[e]).sum::<f32>();
    let weights = experts
        .iter()
        .map(|&e| scores[e] / total * config.routed_scaling)
        .collect();
    let margin = order.get(config.top_k).map_or(f32::INFINITY, |&next| {
        biased[order[config.top_k - 1]] - biased[next]
    });
    Routing {
        experts,
        weights,
        margin,
    }
}

struct Expert {
    up: NtLinear,
    down: NtLinear,
}

impl Expert {
    fn forward(&self, x: &Tensor, pass: Pass) -> Result<Tensor> {
        let h = self.up.forward_padded(x, pass)?.relu()?.sqr()?;
        self.down.forward(&h, pass)
    }
}

pub struct MoeMixer {
    config: MoeConfig,
    /// `[experts, hidden]` F32 (the router GEMM runs in F32).
    gate: Tensor,
    bias: Vec<f32>,
    experts: Vec<Expert>,
    shared: Expert,
}

impl MoeMixer {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        layer: usize,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = &config.decoder;
        let moe = d.moe;
        let x = format!("backbone.layers.{layer}.mixer");
        let gate = checkpoint.f32_values(&format!("{x}.gate.weight"), &[moe.experts, d.hidden])?;
        let bias =
            checkpoint.f32_values(&format!("{x}.gate.e_score_correction_bias"), &[moe.experts])?;
        let expert = |prefix: String, width: usize| -> Result<Expert> {
            let up = NtLinear::load_fp8(
                checkpoint,
                &format!("{prefix}.up_proj"),
                width,
                d.hidden,
                false,
                device,
                mode,
            )?;
            let down = NtLinear::load_fp8(
                checkpoint,
                &format!("{prefix}.down_proj"),
                d.hidden,
                width,
                false,
                device,
                mode,
            )?;
            if up.padded_out() != down.padded_in() {
                return Err(Error::ModelLoadError(format!(
                    "{prefix}: padded up ({}) and down ({}) widths differ",
                    up.padded_out(),
                    down.padded_in()
                )));
            }
            Ok(Expert { up, down })
        };
        let experts = (0..moe.experts)
            .map(|e| expert(format!("{x}.experts.{e}"), moe.intermediate))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: moe,
            gate: Tensor::from_vec(gate, (moe.experts, d.hidden), device)?,
            bias,
            shared: expert(format!("{x}.shared_experts"), moe.shared_intermediate)?,
            experts,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.experts
            .iter()
            .chain(std::iter::once(&self.shared))
            .map(|e| e.up.resident_bytes() + e.down.resident_bytes())
            .sum::<usize>()
            + self.gate.elem_count() * 4
    }

    /// The first expert's and the shared expert's projections, for
    /// load-time checks.
    pub fn sample_projections(&self) -> [&NtLinear; 4] {
        [
            &self.experts[0].up,
            &self.experts[0].down,
            &self.shared.up,
            &self.shared.down,
        ]
    }

    /// Router decisions for `x` `[tokens, hidden]`.
    pub fn routes(&self, x: &Tensor) -> Result<Vec<Routing>> {
        let logits = host_values(&x.to_dtype(DType::F32)?.matmul(&self.gate.t()?)?)?;
        Ok(logits
            .chunks_exact(self.config.experts)
            .map(|row| route(row, &self.bias, &self.config))
            .collect())
    }

    /// The mixer over `x` `[tokens, hidden]` F32. Returns the output and the
    /// tokens' routing.
    pub fn forward(&self, x: &Tensor, pass: Pass) -> Result<(Tensor, Vec<Routing>)> {
        let routes = self.routes(x)?;
        let mut assigned = vec![Vec::<(u32, f32)>::new(); self.config.experts];
        for (token, routing) in routes.iter().enumerate() {
            for (&e, &w) in routing.experts.iter().zip(&routing.weights) {
                assigned[e].push((token as u32, w));
            }
        }
        let device = x.device();
        let mut out = self.shared.forward(x, pass)?;
        for (expert, tokens) in self.experts.iter().zip(&assigned) {
            if tokens.is_empty() {
                continue;
            }
            let ids = Tensor::from_vec(
                tokens.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
                tokens.len(),
                device,
            )?;
            let weights = Tensor::from_vec(
                tokens.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
                (tokens.len(), 1),
                device,
            )?;
            let y = expert
                .forward(&x.index_select(&ids, 0)?, pass)?
                .broadcast_mul(&weights)?;
            out = out.index_add(&ids, &y, 0)?;
        }
        Ok((out, routes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(top_k: usize) -> MoeConfig {
        MoeConfig {
            experts: 8,
            top_k,
            intermediate: 32,
            shared_intermediate: 32,
            routed_scaling: 2.5,
        }
    }

    #[test]
    fn routing_selects_by_biased_score_and_weighs_by_the_unbiased_one() {
        let logits = [0.0f32, 1.0, -1.0, 2.0, 0.5, -2.0, 0.1, 0.2];
        // A bias that promotes expert 5 above every unbiased winner.
        let mut bias = vec![40.69f32; 8];
        bias[5] += 1.0;
        let r = route(&logits, &bias, &config(2));
        assert_eq!(r.experts, vec![5, 3]);
        let s = |v: f32| 1.0 / (1.0 + (-v).exp());
        let total = s(-2.0) + s(2.0);
        assert!((r.weights[0] - s(-2.0) / total * 2.5).abs() < 1e-6);
        assert!((r.weights[1] - s(2.0) / total * 2.5).abs() < 1e-6);
        assert!((r.weights.iter().sum::<f32>() - 2.5).abs() < 1e-5);
    }

    /// RED: the selection add in BF16 at a bias of 40.69 collapses scores a
    /// 16th apart into ties and changes the routed set.
    #[test]
    fn a_bf16_selection_bias_reorders_the_experts() {
        let logits = [0.3f32, 0.32, 0.34, 0.36, 0.0, -0.1, -0.2, -0.3];
        let bias = (0..8).map(|i| 40.69 + 0.001 * i as f32).collect::<Vec<_>>();
        let f32_route = route(&logits, &bias, &config(2));
        let bf16_biased = logits
            .iter()
            .zip(&bias)
            .map(|(l, b)| half::bf16::from_f32(1.0 / (1.0 + (-l).exp()) + b).to_f32())
            .collect::<Vec<_>>();
        let mut order = (0..8).collect::<Vec<_>>();
        order.sort_by(|&a, &b| bf16_biased[b].total_cmp(&bf16_biased[a]).then(a.cmp(&b)));
        assert_eq!(f32_route.experts, vec![3, 2]);
        assert_ne!(order[..2].to_vec(), f32_route.experts);
    }

    #[test]
    fn the_margin_flags_near_ties() {
        let logits = [1.0f32, 1.0, 0.0, -1.0];
        let r = route(
            &logits,
            &[0.0; 4],
            &MoeConfig {
                experts: 4,
                ..config(1)
            },
        );
        assert_eq!(r.experts, vec![0]);
        assert_eq!(r.margin, 0.0);
    }
}

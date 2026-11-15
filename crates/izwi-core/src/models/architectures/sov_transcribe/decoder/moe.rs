//! The feed-forward half of a Gemma 4 MoE layer: a dense MLP and a sparse
//! expert block that read the same residual and are summed.
//!
//! * Dense: `down(gelu_tanh(gate(x)) * up(x))` over
//!   `pre_feedforward_layernorm(h)`.
//! * Router, in F32 throughout: `proj(rmsnorm(h) * scale * hidden^-0.5)`
//!   over the **raw** residual (a norm without gain), softmax over all
//!   experts, top-k, renormalize the kept probabilities, then multiply by
//!   `per_expert_scale` (so the weights do not sum to one).
//! * Experts read `pre_feedforward_layernorm_2(h)`: `gate_up_proj[e]` is
//!   `[2 * width, hidden]` with the gate in its first `width` rows and the
//!   up projection in its last; `down_proj[e]` is `[hidden, width]`.
//!
//! This is the host-routed path: the router's probabilities are read to the
//! host once per layer and each selected expert runs as two matmuls over its
//! tokens (`models::shared::moe`). It is the reference any device path is
//! checked against.
//! `per_expert_scale` multiplies each expert's output, which is the same
//! product as scaling its routing weight.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{SovCheckpoint, DECODER_PREFIX};
use super::super::config::SovTranscribeConfig;
use super::super::encoder::linear;
use super::norm::RmsNorm;
use crate::error::Result;
use crate::models::shared::moe::{ExpertSet, SparseMoeConfig, SparseMoeDispatcher};

pub struct DenseMlp {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl DenseMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear(x, &self.gate, None)?.gelu()?;
        linear(&(gate * linear(x, &self.up, None)?)?, &self.down, None)
    }
}

pub struct Router {
    norm: RmsNorm,
    /// `scale * hidden^-0.5`, F32 `[hidden]`.
    scale: Tensor,
    /// F32 `[experts, hidden]`.
    proj: Tensor,
}

impl Router {
    /// Router logits `[n, experts]` F32 of the raw residual `h`.
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward_f32(h)?.broadcast_mul(&self.scale)?;
        linear(&x, &self.proj, None)
    }
}

pub struct Experts {
    /// `[experts, 2 * width, hidden]`.
    gate_up: Tensor,
    /// `[experts, hidden, width]`.
    down: Tensor,
    per_expert_scale: Vec<f32>,
    width: usize,
}

impl ExpertSet for Experts {
    fn num_experts(&self) -> usize {
        self.per_expert_scale.len()
    }

    fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor> {
        let gate_up = linear(tokens, &self.gate_up.get(expert)?, None)?;
        let gate = gate_up.narrow(1, 0, self.width)?.gelu()?;
        let up = gate_up.narrow(1, self.width, self.width)?;
        let out = linear(&(gate * up)?, &self.down.get(expert)?, None)?;
        Ok((out * self.per_expert_scale[expert] as f64)?)
    }
}

/// One layer's dense MLP, router and experts.
pub struct SovFeedForward {
    pub(crate) mlp: DenseMlp,
    pub(crate) router: Router,
    pub(crate) experts: Experts,
    dispatcher: SparseMoeDispatcher,
    resident_bytes: usize,
}

impl SovFeedForward {
    pub fn load(
        checkpoint: &SovCheckpoint,
        config: &SovTranscribeConfig,
        layer: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let d = &config.decoder;
        let l = format!("{DECODER_PREFIX}.layers.{layer}");
        let get =
            |name: &str, dtype: DType| checkpoint.tensor(&format!("{l}.{name}"), device, dtype);
        let mlp = DenseMlp {
            gate: get("mlp.gate_proj.weight", dtype)?,
            up: get("mlp.up_proj.weight", dtype)?,
            down: get("mlp.down_proj.weight", dtype)?,
        };
        let router = Router {
            norm: RmsNorm::without_scale(d.norm_eps),
            scale: (get("router.scale", DType::F32)? * (d.hidden as f64).powf(-0.5))?,
            proj: get("router.proj.weight", DType::F32)?,
        };
        let experts = Experts {
            gate_up: get("experts.gate_up_proj", dtype)?,
            down: get("experts.down_proj", dtype)?,
            per_expert_scale: checkpoint.host_f32(&format!("{l}.router.per_expert_scale"))?,
            width: d.expert_width,
        };
        let resident_bytes = [
            &mlp.gate,
            &mlp.up,
            &mlp.down,
            &experts.gate_up,
            &experts.down,
        ]
        .iter()
        .map(|t| t.elem_count() * dtype.size_in_bytes())
        .sum::<usize>()
            + (router.scale.elem_count() + router.proj.elem_count()) * 4;
        Ok(Self {
            mlp,
            router,
            experts,
            dispatcher: SparseMoeDispatcher::new(SparseMoeConfig {
                num_experts: d.experts,
                num_experts_per_tok: d.top_k,
                norm_topk_prob: true,
            })?,
            resident_bytes,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// The dense branch over `pre_feedforward_layernorm(h)`.
    pub fn dense(&self, normed: &Tensor) -> Result<Tensor> {
        self.mlp.forward(normed)
    }

    /// The expert branch: routed by the raw residual `h`, applied to
    /// `normed` (`pre_feedforward_layernorm_2(h)`).
    pub fn sparse(&self, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        let logits = self.router.logits(h)?;
        self.dispatcher.dispatch(normed, &logits, &self.experts)
    }

    /// Each token's `(expert, weight)` choices, weights as the reference
    /// applies them (renormalized, then `per_expert_scale`).
    pub fn routes(&self, h: &Tensor) -> Result<Vec<Vec<(u32, f32)>>> {
        let logits = self
            .router
            .logits(h)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let top_k = self.dispatcher.config().num_experts_per_tok;
        Ok(logits
            .iter()
            .map(|row| {
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps = row.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
                let sum = exps.iter().sum::<f32>();
                let mut order = (0..row.len()).collect::<Vec<_>>();
                order.sort_by(|a, b| exps[*b].total_cmp(&exps[*a]).then(a.cmp(b)));
                order.truncate(top_k);
                let kept = order.iter().map(|e| exps[*e] / sum).sum::<f32>();
                order
                    .iter()
                    .map(|&e| {
                        (
                            e as u32,
                            exps[e] / sum / kept * self.experts.per_expert_scale[e],
                        )
                    })
                    .collect()
            })
            .collect())
    }
}

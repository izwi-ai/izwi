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
//! Two paths compute the experts, with the same arithmetic:
//!
//! * **Host-routed** (the reference): the router's probabilities are read to
//!   the host once per layer and each selected expert runs as two matmuls
//!   over its tokens (`models::shared::moe`). It reads each used expert once,
//!   whatever the batch, so it serves prefill and wide decode steps.
//! * **Device-routed** ([`SovFeedForward::sparse_device`]): softmax, top-k
//!   and the weights stay on the device, and each (token, choice) gathers its
//!   expert's rows and runs as one batched matmul. Nothing waits for the host,
//!   so a decode step of a few rows launches its 30 layers back to back, but
//!   it copies each choice's expert weights, so it only pays below
//!   [`DEVICE_ROUTED_MAX_CHOICES`] choices.
//!
//! `per_expert_scale` multiplies each expert's output, which is the same
//! product as scaling its routing weight.

use candle_core::{DType, Device, Tensor, D};

use super::super::checkpoint::{SovCheckpoint, DECODER_PREFIX};
use super::super::config::SovTranscribeConfig;
use super::super::encoder::linear;
use super::norm::RmsNorm;
use crate::error::Result;
use crate::models::shared::moe::{ExpertSet, SparseMoeConfig, SparseMoeDispatcher};

/// `IZWI_SOV_TRANSCRIBE_MOE_BACKEND`: `auto` (the default: device-routed
/// for steps of at most [`DEVICE_ROUTED_MAX_CHOICES`] choices on CUDA,
/// host-routed otherwise), `host` (always host-routed) or `device` (always
/// device-routed, for tests).
pub const MOE_BACKEND_ENV: &str = "IZWI_SOV_TRANSCRIBE_MOE_BACKEND";

/// The most (token, expert) choices a device-routed call takes: 4 decode
/// rows of 8 experts. A choice copies its expert's 12 MiB of BF16 rows, so
/// past this a host-routed call, which reads each used expert once, moves
/// fewer bytes.
pub const DEVICE_ROUTED_MAX_CHOICES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    Auto,
    Host,
    Device,
}

impl MoeBackend {
    pub fn parse(value: &str) -> crate::error::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "host" | "legacy" => Ok(Self::Host),
            "device" => Ok(Self::Device),
            other => Err(crate::error::Error::InvalidInput(format!(
                "{MOE_BACKEND_ENV}={other:?}: expected auto, host or device"
            ))),
        }
    }

    pub fn from_env() -> crate::error::Result<Self> {
        match std::env::var(MOE_BACKEND_ENV) {
            Ok(value) => Self::parse(&value),
            Err(_) => Ok(Self::Auto),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Host => "host",
            Self::Device => "device",
        }
    }

    /// Whether a call of `tokens` tokens on `device` routes on the device.
    pub fn routes_on_device(self, device: &Device, tokens: usize, top_k: usize) -> bool {
        match self {
            Self::Host => false,
            Self::Device => true,
            Self::Auto => device.is_cuda() && tokens * top_k <= DEVICE_ROUTED_MAX_CHOICES,
        }
    }
}

/// The dense MLP, gate and up projections as one `[gate; up]` weight.
pub struct DenseMlp {
    gate_up: Tensor,
    down: Tensor,
    width: usize,
}

impl DenseMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = linear(x, &self.gate_up, None)?;
        let gate = gate_up.narrow(1, 0, self.width)?.gelu()?;
        let up = gate_up.narrow(1, self.width, self.width)?;
        linear(&(gate * up)?, &self.down, None)
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
    /// `per_expert_scale` on the device, F32 `[experts]`.
    per_expert_scale: Tensor,
    top_k: usize,
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
            gate_up: Tensor::cat(
                &[
                    get("mlp.gate_proj.weight", dtype)?,
                    get("mlp.up_proj.weight", dtype)?,
                ],
                0,
            )?,
            down: get("mlp.down_proj.weight", dtype)?,
            width: d.mlp_width,
        };
        let router = Router {
            norm: RmsNorm::without_scale(d.norm_eps, d.hidden, device)?,
            scale: (get("router.scale", DType::F32)? * (d.hidden as f64).powf(-0.5))?,
            proj: get("router.proj.weight", DType::F32)?,
        };
        let experts = Experts {
            gate_up: get("experts.gate_up_proj", dtype)?,
            down: get("experts.down_proj", dtype)?,
            per_expert_scale: checkpoint.host_f32(&format!("{l}.router.per_expert_scale"))?,
            width: d.expert_width,
        };
        let resident_bytes = [&mlp.gate_up, &mlp.down, &experts.gate_up, &experts.down]
            .iter()
            .map(|t| t.elem_count() * dtype.size_in_bytes())
            .sum::<usize>()
            + (router.scale.elem_count() + router.proj.elem_count()) * 4;
        let per_expert_scale =
            Tensor::from_vec(experts.per_expert_scale.clone(), d.experts, device)?;
        Ok(Self {
            mlp,
            router,
            experts,
            per_expert_scale,
            top_k: d.top_k,
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

    /// [`Self::sparse`] with routing and experts on the device: softmax over
    /// every expert in F32, the top-k probabilities renormalized and scaled
    /// by `per_expert_scale`, each choice's expert rows gathered and run as
    /// one batched matmul, the choices of a token summed. No host readback.
    pub fn sparse_device(&self, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        let (tokens, hidden) = normed.dims2()?;
        let k = self.top_k;
        let probs = candle_nn::ops::softmax_last_dim(&self.router.logits(h)?)?;
        let index = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?;
        let kept = probs.gather(&index, D::Minus1)?;
        let flat = index.flatten_all()?;
        let weights = kept.broadcast_div(&kept.sum_keepdim(D::Minus1)?)?.mul(
            &self
                .per_expert_scale
                .index_select(&flat, 0)?
                .reshape((tokens, k))?,
        )?;
        let e = &self.experts;
        // [tokens * k, 1, hidden], each token repeated for its choices.
        let x = normed
            .unsqueeze(1)?
            .broadcast_as((tokens, k, hidden))?
            .contiguous()?
            .reshape((tokens * k, 1, hidden))?;
        let gate_up = e.gate_up.index_select(&flat, 0)?;
        let gate_up = x.matmul(&gate_up.transpose(1, 2)?)?;
        let gate = gate_up.narrow(2, 0, e.width)?.gelu()?;
        let up = gate_up.narrow(2, e.width, e.width)?;
        let down = e.down.index_select(&flat, 0)?;
        let out = (gate * up)?.matmul(&down.transpose(1, 2)?)?;
        let out = out.reshape((tokens, k, hidden))?;
        let weighted = out.broadcast_mul(&weights.to_dtype(out.dtype())?.unsqueeze(2)?)?;
        Ok(weighted.sum(1)?)
    }

    /// The expert branch through `backend`'s path for this many tokens.
    pub fn sparse_with(&self, backend: MoeBackend, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        if backend.routes_on_device(normed.device(), normed.dim(0)?, self.top_k) {
            self.sparse_device(h, normed)
        } else {
            self.sparse(h, normed)
        }
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

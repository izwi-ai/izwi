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
//! Four paths compute the experts, with the same arithmetic:
//!
//! * **Host-routed** (the reference): the router's probabilities are read to
//!   the host once per layer and each selected expert runs as two matmuls
//!   over its tokens (`models::shared::moe`). Each used expert is read once
//!   and its tokens run as one GEMM, which suits prefill spans.
//! * **Fused** ([`SovFeedForward::sparse_fused`], `kernels::moe_gelu`):
//!   routing and experts in five launches with the expert ids on the
//!   device. Each used expert's BF16 rows are read once for all of its
//!   tokens, so it costs what the host path reads without the readback, and
//!   a row's result does not depend on the batch. It serves decode steps
//!   (up to [`FUSED_MAX_PAIRS`] routed pairs) on CUDA.
//! * **Grouped** ([`SovFeedForward::sparse_grouped`]): device routing read
//!   back once, each expert's tokens laid out in a padded `[experts, rows,
//!   hidden]` block, and every expert run as one batched matmul over its
//!   resident weights (two GEMMs and the scatter back: about a dozen launches
//!   a layer). It serves prefill spans on CUDA, where the host path's
//!   per-expert loop cost about 1,800 launches a layer (233 ms for a
//!   512-token span on the H100) and the fused kernels' warp-per-channel
//!   dots cannot use tensor cores.
//! * **Gathered** ([`SovFeedForward::sparse_device`]): device routing from
//!   composed tensor operations, each choice's expert rows gathered into one
//!   batched matmul. Portable but it copies the weights; kept for
//!   `IZWI_SOV_TRANSCRIBE_MOE_BACKEND=device`.
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

/// `IZWI_SOV_TRANSCRIBE_MOE_BACKEND`: `auto` (the default: on CUDA fused
/// for calls of at most [`FUSED_MAX_PAIRS`] routed pairs and grouped above;
/// host-routed off CUDA), `fused` (the fused kernels for every call; their
/// CPU reference off CUDA), `grouped` (always grouped), `host` (always
/// host-routed) or `device` (always gathered).
pub const MOE_BACKEND_ENV: &str = "IZWI_SOV_TRANSCRIBE_MOE_BACKEND";

/// The most routed pairs (tokens times 8) a fused call takes under `auto`:
/// every decode width up to 64 rows. Longer prefill spans run each expert's
/// tokens as GEMMs on the grouped path, which the per-channel warp kernels
/// cannot match once an expert has many tokens.
pub const FUSED_MAX_PAIRS: usize = 512;

/// The most tokens one grouped call pads its experts' blocks over: with 128
/// experts of the published geometry, the padded input, activations and
/// output stay under about 1 GiB of BF16.
pub const GROUPED_CHUNK_TOKENS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    Auto,
    Host,
    Device,
    Fused,
    Grouped,
}

/// The expert path one call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPath {
    Host,
    Gathered,
    Fused,
    Grouped,
}

impl MoeBackend {
    pub fn parse(value: &str) -> crate::error::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "host" | "legacy" => Ok(Self::Host),
            "device" | "gathered" => Ok(Self::Device),
            "fused" => Ok(Self::Fused),
            "grouped" => Ok(Self::Grouped),
            other => Err(crate::error::Error::InvalidInput(format!(
                "{MOE_BACKEND_ENV}={other:?}: expected auto, fused, grouped, host or device"
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
            Self::Fused => "fused",
            Self::Grouped => "grouped",
        }
    }

    /// The path a call of `tokens` tokens on `device` takes.
    pub fn path(self, device: &Device, tokens: usize, top_k: usize) -> ExpertPath {
        match self {
            Self::Host => ExpertPath::Host,
            Self::Device => ExpertPath::Gathered,
            Self::Fused => ExpertPath::Fused,
            Self::Grouped => ExpertPath::Grouped,
            Self::Auto if device.is_cuda() && tokens * top_k <= FUSED_MAX_PAIRS => {
                ExpertPath::Fused
            }
            Self::Auto if device.is_cuda() => ExpertPath::Grouped,
            Self::Auto => ExpertPath::Host,
        }
    }

    /// Whether a call of `tokens` tokens on `device` keeps routing on the
    /// device (no host readback): what a CUDA graph of the step needs.
    pub fn routes_on_device(self, device: &Device, tokens: usize, top_k: usize) -> bool {
        matches!(
            self.path(device, tokens, top_k),
            ExpertPath::Gathered | ExpertPath::Fused
        )
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
        self.sparse_on(
            backend.path(normed.device(), normed.dim(0)?, self.top_k),
            h,
            normed,
        )
    }

    /// The expert branch through `path`.
    pub fn sparse_on(&self, path: ExpertPath, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        match path {
            ExpertPath::Host => self.sparse(h, normed),
            ExpertPath::Gathered => self.sparse_device(h, normed),
            ExpertPath::Fused => self.sparse_fused(h, normed),
            ExpertPath::Grouped => self.sparse_grouped(h, normed),
        }
    }

    /// [`Self::sparse`] as two batched matmuls over every expert's resident
    /// weights: the routing (renormalized, `per_expert_scale` applied) is
    /// read back once, each expert's tokens are gathered into a block padded
    /// to the busiest expert's count, and each block row's output is scaled
    /// by its weight and summed into its token.
    /// Calls of more than [`GROUPED_CHUNK_TOKENS`] tokens run in chunks of
    /// that many, which bounds the padded blocks however skewed the routing:
    /// on tensor cores even every expert padded to the whole chunk costs
    /// about 2 ms a layer, where the host path's per-expert loop cost 9.
    pub fn sparse_grouped(&self, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        let tokens = normed.dim(0)?;
        if tokens <= GROUPED_CHUNK_TOKENS {
            return self.sparse_grouped_chunk(h, normed);
        }
        let mut outputs = Vec::with_capacity(tokens.div_ceil(GROUPED_CHUNK_TOKENS));
        let mut start = 0;
        while start < tokens {
            let len = GROUPED_CHUNK_TOKENS.min(tokens - start);
            outputs.push(
                self.sparse_grouped_chunk(
                    &h.narrow(0, start, len)?,
                    &normed.narrow(0, start, len)?,
                )?,
            );
            start += len;
        }
        Ok(Tensor::cat(&outputs, 0)?)
    }

    fn sparse_grouped_chunk(&self, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        let (tokens, hidden) = normed.dims2()?;
        let k = self.top_k;
        let experts = self.experts.per_expert_scale.len();
        let routing =
            crate::kernels::moe_gelu::route(&self.router.logits(h)?, &self.per_expert_scale, k)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
        // Each expert's tokens, and where each (token, slot) lands among the
        // padded block rows.
        let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); experts];
        let mut slot_rows = vec![(usize::MAX, 0usize); tokens * k];
        let mut weights = vec![0f32; tokens * k];
        for (token, row) in routing.iter().enumerate() {
            for slot in 0..k {
                let id = row[slot];
                if id >= 0.0 && (id as usize) < experts {
                    let expert = id as usize;
                    slot_rows[token * k + slot] = (expert, by_expert[expert].len());
                    by_expert[expert].push(token as u32);
                    weights[token * k + slot] = row[k + slot];
                }
            }
        }
        let rows = by_expert.iter().map(Vec::len).max().unwrap_or(0);
        if rows == 0 {
            return Ok(normed.zeros_like()?);
        }
        let mut ids = vec![0u32; experts * rows];
        for (expert, choices) in by_expert.iter().enumerate() {
            ids[expert * rows..expert * rows + choices.len()].copy_from_slice(choices);
        }
        // A slot without an expert reads block row 0 at weight zero.
        let gather = slot_rows
            .iter()
            .map(|&(expert, index)| {
                if expert == usize::MAX {
                    0
                } else {
                    (expert * rows + index) as u32
                }
            })
            .collect::<Vec<_>>();
        let device = normed.device();
        let e = &self.experts;
        let ids = Tensor::from_vec(ids, experts * rows, device)?;
        let x = normed
            .index_select(&ids, 0)?
            .reshape((experts, rows, hidden))?;
        let gate_up = x.matmul(&e.gate_up.transpose(1, 2)?)?;
        let gate = gate_up.narrow(2, 0, e.width)?.gelu()?;
        let up = gate_up.narrow(2, e.width, e.width)?;
        let out = (gate * up)?
            .matmul(&e.down.transpose(1, 2)?)?
            .reshape((experts * rows, hidden))?;
        // Each token gathers its slots' rows and sums them by weight (an
        // index_add over the padded rows scanned every row per column).
        let gather = Tensor::from_vec(gather, tokens * k, device)?;
        let weights = Tensor::from_vec(weights, (tokens, k, 1), device)?.to_dtype(out.dtype())?;
        Ok(out
            .index_select(&gather, 0)?
            .reshape((tokens, k, hidden))?
            .broadcast_mul(&weights)?
            .sum(1)?)
    }

    /// [`Self::sparse`] through `kernels::moe_gelu`: routing and experts on
    /// the device, each used expert's rows read once, F32 sums.
    pub fn sparse_fused(&self, h: &Tensor, normed: &Tensor) -> Result<Tensor> {
        let routing = crate::kernels::moe_gelu::route(
            &self.router.logits(h)?,
            &self.per_expert_scale,
            self.top_k,
        )?;
        let out = crate::kernels::moe_gelu::experts(
            &normed.to_dtype(DType::F32)?,
            &routing,
            &self.experts.gate_up,
            &self.experts.down,
        )?;
        Ok(out.to_dtype(normed.dtype())?)
    }

    /// Whether the fused kernels take this layer's geometry and dtype.
    pub fn fused_supported(&self) -> bool {
        let (experts, rows, hidden) = self.experts.gate_up.dims3().unwrap_or((0, 0, 0));
        crate::kernels::moe_gelu::weight_dtype_supported(&self.experts.gate_up)
            && crate::kernels::moe_gelu::supported(hidden, rows / 2, experts, self.top_k)
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

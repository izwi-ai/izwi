//! Qwen3.5 sparse-expert feed-forward block (Qwen3.5-35B-A3B MoE FFN).
//!
//! One block replaces the dense SwiGLU MLP on every layer of the shared
//! Qwen3.5 hybrid trunk: a dense router produces expert logits, the shared
//! [`SparseMoeDispatcher`] routes tokens through the top-k routed experts,
//! and an always-on shared expert is applied to every token and added to
//! the routed output.
//!
//! Routing mode is pinned from the transformers `qwen3_5_moe` modeling
//! source (checked 2026-09-29): F32 softmax over the router logits, top-k
//! selection, then the top-k weights are divided by their sum. There is no
//! sigmoid scoring and no `e_score_correction_bias`, and the config class
//! exposes no `norm_topk_prob`/`scoring_func` knobs — so the renormalized
//! softmax here is not configurable for this family.
//!
//! The shared expert is `down(silu(gate(x)) * up(x))`, scaled by
//! `sigmoid(shared_expert_gate(x))` when the checkpoint carries the optional
//! `[1, hidden]` gate projection, and added to the routed output
//! unconditionally (same modeling source).
//!
//! When every expert is resident as 128x128 block-FP8, the block runs the
//! device-routed fused kernels instead of the dispatcher loop (see
//! [`super::fused_moe`]); the dispatcher remains the path for quantized/dense
//! residencies, the `IZWI_QWEN36_MOE_BACKEND=legacy` switch, and any block
//! whose load-time self-check fails.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops;

use crate::error::{Error, Result};
use crate::kernels::try_fused_silu_mul;
use crate::models::shared::moe::{
    ExpertActivationCounters, ExpertSet, SparseMoeConfig, SparseMoeDispatcher,
};
use crate::models::shared::weights::gguf::GgufLoader;

use crate::models::architectures::qwen36moe::fused_moe::{
    Qwen36MoeBackend, Qwen36MoeBackendRequest, Qwen36MoeFusedExperts, Qwen36MoeStacking,
    BACKEND_ENV,
};
use crate::models::architectures::qwen36moe::text::Qwen36MoeFfnGeometry;

/// Persistent form of one projection inside the sparse block. `Quantized`
/// keeps quantized residency (GGUF tensors, CPU-packed Q8_0 requants);
/// `Dense` holds an expanded weight for backends without packed kernels;
/// `CompactFp8` keeps the checkpoint's raw block-FP8 bytes plus F32 block
/// scales resident and decodes per GEMM inside the CUDA fp8 projection
/// kernel.
#[derive(Clone)]
pub(crate) enum Qwen36MoeLinear {
    Dense(Tensor),
    Quantized(QMatMul),
    CompactFp8 { weights: Tensor, scales: Tensor },
}

impl Qwen36MoeLinear {
    pub(crate) fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Ok(Self::Quantized(
            QMatMul::from_arc(Arc::new(qtensor)).map_err(Error::from)?,
        ))
    }

    pub(crate) fn from_dense(weight: Tensor) -> Self {
        Self::Dense(weight)
    }

    /// Apply the projection to `[num_tokens, in]`, returning
    /// `[num_tokens, out]` in the activation's dtype. Quantized matmuls
    /// compute in F32; dense projections compute in their residency dtype;
    /// compact FP8 decodes to the activation's dtype inside the kernel.
    pub(crate) fn project(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let output = match self {
            Self::Quantized(qmatmul) => qmatmul.forward(&x.to_dtype(DType::F32)?)?,
            Self::Dense(weight) => x.to_dtype(weight.dtype())?.matmul(&weight.t()?)?,
            Self::CompactFp8 { weights, scales } => {
                crate::kernels::cuda::fp8::block_fp8_projection(x, weights, scales)?
            }
        };
        if output.dtype() == input_dtype {
            Ok(output)
        } else {
            output.to_dtype(input_dtype).map_err(Error::from)
        }
    }
}

/// Routed-expert projections: `[out, in]` gate/up/down per expert.
#[derive(Clone)]
pub(crate) struct Qwen36MoeExpertWeights {
    pub gate: Qwen36MoeLinear,
    pub up: Qwen36MoeLinear,
    pub down: Qwen36MoeLinear,
}

/// Always-on shared-expert projections plus the optional sigmoid output
/// gate (`shared_expert_gate`, `[1, hidden]`).
#[derive(Clone)]
pub(crate) struct Qwen36MoeSharedExpertWeights {
    pub gate: Qwen36MoeLinear,
    pub up: Qwen36MoeLinear,
    pub down: Qwen36MoeLinear,
    pub output_gate: Option<Qwen36MoeLinear>,
}

struct Qwen36MoeRoutedExperts(Vec<Qwen36MoeExpertWeights>);

impl ExpertSet for Qwen36MoeRoutedExperts {
    fn num_experts(&self) -> usize {
        self.0.len()
    }

    fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor> {
        let weights = self.0.get(expert).ok_or_else(|| {
            Error::InferenceError(format!("qwen36moe routed expert {expert} is out of range"))
        })?;
        let hidden = swiglu(&weights.gate, &weights.up, tokens)?;
        weights.down.project(&hidden)
    }
}

struct Qwen36MoeSharedExpert(Qwen36MoeSharedExpertWeights);

impl Qwen36MoeSharedExpert {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let hidden = swiglu(&self.0.gate, &self.0.up, tokens)?;
        let output = self.0.down.project(&hidden)?;
        match &self.0.output_gate {
            None => Ok(output),
            Some(gate) => {
                let logits = gate.project(tokens)?;
                output.broadcast_mul(&ops::sigmoid(&logits)?).map_err(Error::from)
            }
        }
    }
}

fn swiglu(gate: &Qwen36MoeLinear, up: &Qwen36MoeLinear, tokens: &Tensor) -> Result<Tensor> {
    let gate_out = gate.project(tokens)?;
    let up_out = up.project(tokens)?;
    if let Some(fused) = try_fused_silu_mul(&gate_out, &up_out) {
        return Ok(fused);
    }
    let activated = ops::silu(&gate_out)?;
    (&activated * &up_out).map_err(Error::from)
}

/// One layer's sparse-expert feed-forward: dense router → renormalized
/// softmax top-k routed experts + unconditional shared expert.
pub(crate) struct Qwen36MoeSparseMlp {
    dispatcher: SparseMoeDispatcher,
    router: Qwen36MoeLinear,
    experts: Qwen36MoeRoutedExperts,
    shared: Qwen36MoeSharedExpert,
    counters: Arc<ExpertActivationCounters>,
    fused: Option<Qwen36MoeFusedExperts>,
    backend: Qwen36MoeBackend,
}

impl Qwen36MoeSparseMlp {
    /// Assemble a block from source-materialized weights. Fails closed when
    /// the routed expert count disagrees with the declared geometry.
    pub(crate) fn from_weights(
        router: Qwen36MoeLinear,
        experts: Vec<Qwen36MoeExpertWeights>,
        shared: Qwen36MoeSharedExpertWeights,
        geometry: &Qwen36MoeFfnGeometry,
    ) -> Result<Self> {
        Self::from_weights_with_backend(
            router,
            experts,
            shared,
            geometry,
            Qwen36MoeBackendRequest::from_env(),
        )
    }

    /// [`Self::from_weights`] with an explicit execution-path request instead
    /// of the `IZWI_QWEN36_MOE_BACKEND` environment switch.
    pub(crate) fn from_weights_with_backend(
        router: Qwen36MoeLinear,
        experts: Vec<Qwen36MoeExpertWeights>,
        shared: Qwen36MoeSharedExpertWeights,
        geometry: &Qwen36MoeFfnGeometry,
        request: Qwen36MoeBackendRequest,
    ) -> Result<Self> {
        if experts.len() != geometry.num_experts {
            return Err(Error::ModelLoadError(format!(
                "qwen36moe sparse block declares {} routed experts but geometry expects {}",
                experts.len(),
                geometry.num_experts
            )));
        }
        let counters = Arc::new(ExpertActivationCounters::new(geometry.num_experts));
        let dispatcher = SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts: geometry.num_experts,
            num_experts_per_tok: geometry.num_experts_per_tok,
            // Pinned from transformers qwen3_5_moe: softmax top-k with
            // in-top-k renormalization; not config-exposed for this family.
            norm_topk_prob: true,
        })?
        .with_counters(counters.clone());

        let (experts, shared, fused, backend) = match request {
            Qwen36MoeBackendRequest::Legacy => (
                experts,
                shared,
                None,
                Qwen36MoeBackend::Legacy {
                    reason: format!("{BACKEND_ENV}=legacy"),
                },
            ),
            Qwen36MoeBackendRequest::Auto => {
                match Qwen36MoeFusedExperts::stack(&router, experts, shared, geometry) {
                    Qwen36MoeStacking::Stacked {
                        fused,
                        experts,
                        shared,
                    } => match fused.self_check(&experts, &shared) {
                        Ok(()) => (experts, shared, Some(fused), Qwen36MoeBackend::Fused),
                        Err(reason) => {
                            tracing::warn!(
                                reason,
                                "Qwen3.6-MoE fused expert self-check failed; using the per-expert path"
                            );
                            (
                                experts,
                                shared,
                                None,
                                Qwen36MoeBackend::Legacy {
                                    reason: format!("self-check failed: {reason}"),
                                },
                            )
                        }
                    },
                    Qwen36MoeStacking::Unsupported {
                        experts,
                        shared,
                        reason,
                    } => (experts, shared, None, Qwen36MoeBackend::Legacy { reason }),
                }
            }
        };

        Ok(Self {
            dispatcher,
            router,
            experts: Qwen36MoeRoutedExperts(experts),
            shared: Qwen36MoeSharedExpert(shared),
            counters,
            fused,
            backend,
        })
    }

    /// Execution path this block resolved to at load.
    pub(crate) fn backend(&self) -> &Qwen36MoeBackend {
        &self.backend
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, sequence, hidden) = match hidden_states.rank() {
            3 => hidden_states.dims3()?,
            2 => {
                let (tokens, hidden) = hidden_states.dims2()?;
                (1, tokens, hidden)
            }
            other => {
                return Err(Error::InvalidInput(format!(
                    "qwen36moe sparse block expects rank-2 or rank-3 input, found rank {other}"
                )))
            }
        };
        let tokens = batch * sequence;
        let flat = hidden_states.reshape((tokens, hidden))?.contiguous()?;

        let combined = match &self.fused {
            Some(fused) if fused.accepts(&flat) => {
                let routed = fused.forward(&flat)?;
                if fused.folds_shared() {
                    routed
                } else {
                    routed.broadcast_add(&self.shared.forward(&flat)?)?
                }
            }
            _ => {
                let router_logits = self.router.project(&flat)?;
                let routed = self
                    .dispatcher
                    .dispatch(&flat, &router_logits, &self.experts)?;
                let shared_output = self.shared.forward(&flat)?;
                routed.broadcast_add(&shared_output)?
            }
        };

        if rank3(hidden_states) {
            combined.reshape((batch, sequence, hidden)).map_err(Error::from)
        } else {
            Ok(combined)
        }
    }

    pub(crate) fn counters(&self) -> Arc<ExpertActivationCounters> {
        self.counters.clone()
    }
}

fn rank3(tensor: &Tensor) -> bool {
    tensor.rank() == 3
}

/// Build the sparse block from a GGUF fixture checkpoint's fused expert
/// tensors. The routed experts ride the llama.cpp-style fused layout
/// (`ffn_gate_exps` / `ffn_up_exps` `[n, ff, hidden]`,
/// `ffn_down_exps` `[n, hidden, ff]`); the shared expert uses dedicated
/// `ffn_{gate,up,down}_shexp` tensors and the optional shared-expert output
/// gate is `ffn_gate_inp_shexp` `[1, hidden]`. Quantized residency is
/// preserved through the fused-tensor byte split.
pub(crate) fn load_gguf_sparse_mlp(
    loader: &GgufLoader,
    layer: usize,
    geometry: &Qwen36MoeFfnGeometry,
    device: &Device,
) -> Result<Qwen36MoeSparseMlp> {
    let prefix = format!("blk.{layer}");
    let router = Qwen36MoeLinear::Quantized(
        QMatMul::from_arc(Arc::new(
            loader.load_qtensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?,
        ))
        .map_err(Error::from)?,
    );

    let (ff, hidden) = (geometry.expert_intermediate_size, loader_hidden(loader)?);
    let gate_experts = split_fused(
        loader,
        device,
        &format!("{prefix}.ffn_gate_exps.weight"),
        geometry.num_experts,
        ff,
        hidden,
    )?;
    let up_experts = split_fused(
        loader,
        device,
        &format!("{prefix}.ffn_up_exps.weight"),
        geometry.num_experts,
        ff,
        hidden,
    )?;
    let down_experts = split_fused(
        loader,
        device,
        &format!("{prefix}.ffn_down_exps.weight"),
        geometry.num_experts,
        hidden,
        ff,
    )?;
    let mut experts = Vec::with_capacity(geometry.num_experts);
    for ((gate_q, up_q), down_q) in gate_experts.into_iter().zip(up_experts).zip(down_experts) {
        experts.push(Qwen36MoeExpertWeights {
            gate: Qwen36MoeLinear::from_qtensor(gate_q)?,
            up: Qwen36MoeLinear::from_qtensor(up_q)?,
            down: Qwen36MoeLinear::from_qtensor(down_q)?,
        });
    }

    let shared_ff = geometry.shared_expert_intermediate_size;
    let shared_gate_shape = loader
        .tensor_shape(&format!("{prefix}.ffn_gate_shexp.weight"))
        .ok_or_else(|| {
            Error::ModelLoadError(format!(
                "qwen36moe GGUF fixture is missing `{prefix}.ffn_gate_shexp.weight`"
            ))
        })?;
    let shared_down_shape = loader
        .tensor_shape(&format!("{prefix}.ffn_down_shexp.weight"))
        .ok_or_else(|| {
            Error::ModelLoadError(format!(
                "qwen36moe GGUF fixture is missing `{prefix}.ffn_down_shexp.weight`"
            ))
        })?;
    if shared_gate_shape.first() != Some(&shared_ff) || shared_down_shape.last() != Some(&shared_ff)
    {
        return Err(Error::ModelLoadError(format!(
            "qwen36moe GGUF fixture shared-expert shapes {shared_gate_shape:?}/{shared_down_shape:?} disagree with the declared shared intermediate width {shared_ff}"
        )));
    }
    let shared = Qwen36MoeSharedExpertWeights {
        gate: gguf_linear(loader, device, &format!("{prefix}.ffn_gate_shexp.weight"))?,
        up: gguf_linear(loader, device, &format!("{prefix}.ffn_up_shexp.weight"))?,
        down: gguf_linear(loader, device, &format!("{prefix}.ffn_down_shexp.weight"))?,
        output_gate: loader
            .has_tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"))
            .then(|| gguf_linear(loader, device, &format!("{prefix}.ffn_gate_inp_shexp.weight")))
            .transpose()?,
    };

    Qwen36MoeSparseMlp::from_weights(router, experts, shared, geometry)
}

fn gguf_linear(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
) -> Result<Qwen36MoeLinear> {
    Ok(Qwen36MoeLinear::Quantized(
        QMatMul::from_arc(Arc::new(loader.load_qtensor(name, device)?))
            .map_err(Error::from)?,
    ))
}

fn split_fused(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    num_experts: usize,
    expert_rows: usize,
    expert_cols: usize,
) -> Result<Vec<QTensor>> {
    let fused = loader.load_qtensor(name, device)?;
    crate::models::shared::weights::gguf::split_fused_expert_qtensor(
        &fused,
        num_experts,
        expert_rows,
        expert_cols,
    )
}

fn loader_hidden(loader: &GgufLoader) -> Result<usize> {
    // The fused gate-expert tensor pins the hidden width in its last dim.
    loader
        .tensor_shape("blk.0.ffn_gate_exps.weight")
        .and_then(|shape| shape.last().copied())
        .ok_or_else(|| {
            Error::ModelLoadError(
                "qwen36moe GGUF fixture is missing `blk.0.ffn_gate_exps.weight`".into(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const HIDDEN: usize = 4;
    const FF: usize = 4;
    const NUM_EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const TOKENS: usize = 3;

    fn geometry(shared_ff: usize) -> Qwen36MoeFfnGeometry {
        Qwen36MoeFfnGeometry {
            num_experts: NUM_EXPERTS,
            num_experts_per_tok: TOP_K,
            expert_intermediate_size: FF,
            shared_expert_intermediate_size: shared_ff,
        }
    }

    /// Dense `[out, in]` weight from row-major values.
    fn dense_weight(rows: usize, cols: usize, values: &[f32]) -> Qwen36MoeLinear {
        Qwen36MoeLinear::from_dense(
            Tensor::from_vec(values.to_vec(), (rows, cols), &Device::Cpu).unwrap(),
        )
    }

    fn expert(id: usize) -> Qwen36MoeExpertWeights {
        // expert_out(x) = down · silu(gate · x) * (up · x), all [ff/hidden].
        // Deterministic pseudo-weights derived from the expert id so every
        // expert computes a distinct map.
        let base = (id + 1) as f32 * 0.25;
        Qwen36MoeExpertWeights {
            gate: dense_weight(FF, HIDDEN, &(0..FF * HIDDEN).map(|i| base + (i % 3) as f32 * 0.125 - 0.25).collect::<Vec<_>>()),
            up: dense_weight(FF, HIDDEN, &(0..FF * HIDDEN).map(|i| 0.5 + (i % 5) as f32 * 0.0625).collect::<Vec<_>>()),
            down: dense_weight(HIDDEN, FF, &(0..HIDDEN * FF).map(|i| -0.125 + (i % 4) as f32 * 0.1875).collect::<Vec<_>>()),
        }
    }

    fn shared(with_gate: bool) -> Qwen36MoeSharedExpertWeights {
        Qwen36MoeSharedExpertWeights {
            gate: dense_weight(FF, HIDDEN, &(0..FF * HIDDEN).map(|i| 0.375 - (i % 4) as f32 * 0.09375).collect::<Vec<_>>()),
            up: dense_weight(FF, HIDDEN, &(0..FF * HIDDEN).map(|i| 0.25 + (i % 2) as f32 * 0.5).collect::<Vec<_>>()),
            down: dense_weight(HIDDEN, FF, &(0..HIDDEN * FF).map(|i| 0.625 - (i % 3) as f32 * 0.15625).collect::<Vec<_>>()),
            output_gate: with_gate.then(|| dense_weight(1, HIDDEN, &[0.5, -0.25, 0.75, -1.0])),
        }
    }

    fn router_weight() -> Vec<f32> {
        // Fixed `[num_experts, hidden]` table chosen so each test token's
        // top-2 projection selects a different expert pair (no ranking ties).
        vec![
            0.25, -0.5, 1.0, 0.75, //
            -1.0, 0.875, -0.25, 0.5, //
            1.5, -0.75, 0.125, -1.25, //
            0.625, 1.25, -0.875, 0.375,
        ]
    }

    fn hidden_states() -> Tensor {
        Tensor::from_vec(
            vec![1.0_f32, -2.0, 0.5, 3.0, -1.5, 2.5, 0.125, -0.375, 2.0, 1.0, -1.0, 0.75],
            (TOKENS, HIDDEN),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn project_dense(weight: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0_f32; x.len() / cols * rows];
        for (token, x_row) in x.chunks(cols).enumerate() {
            for (o, w_row) in weight.chunks(cols).enumerate() {
                out[token * rows + o] = x_row
                    .iter()
                    .zip(w_row)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
        }
        out
    }

    fn reference_expert(weights: &Qwen36MoeExpertWeights, x: &[f32]) -> Vec<f32> {
        let as_values = |linear: &Qwen36MoeLinear| match linear {
            Qwen36MoeLinear::Dense(tensor) => tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            Qwen36MoeLinear::Quantized(_) => unreachable!("reference test uses dense weights"),
            Qwen36MoeLinear::CompactFp8 { .. } => unreachable!("reference test uses dense weights"),
        };
        let gate = as_values(&weights.gate);
        let up = as_values(&weights.up);
        let down = as_values(&weights.down);
        let gate_out = project_dense(&gate, FF, HIDDEN, x);
        let up_out = project_dense(&up, FF, HIDDEN, x);
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        project_dense(&down, HIDDEN, FF, &hidden)
    }

    fn reference_forward(
        experts: &[Qwen36MoeExpertWeights],
        shared: &Qwen36MoeSharedExpertWeights,
        with_gate: bool,
    ) -> Vec<f32> {
        let router_weight = dense_weight(NUM_EXPERTS, HIDDEN, &router_weight());
        let router_values = match &router_weight {
            Qwen36MoeLinear::Dense(tensor) => {
                tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            }
            _ => unreachable!("reference test uses dense weights"),
        };
        let hidden = hidden_states().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut output = vec![0_f32; TOKENS * HIDDEN];
        for token in 0..TOKENS {
            let x = &hidden[token * HIDDEN..(token + 1) * HIDDEN];
            // Router logits are the projection of the token, not raw weights.
            let token_logits = project_dense(&router_values, NUM_EXPERTS, HIDDEN, x);
            let max = token_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = token_logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let mut ranked: Vec<(usize, f32)> = exps
                .iter()
                .enumerate()
                .map(|(e, p)| (e, p / sum))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let selected = &ranked[..TOP_K];
            let norm: f32 = selected.iter().map(|(_, w)| *w).sum();

            let mut routed = [0_f32; HIDDEN];
            for &(expert, weight) in selected {
                let applied = reference_expert(&experts[expert], x);
                for (o, value) in applied.iter().enumerate() {
                    routed[o] += (weight / norm) * value;
                }
            }

            let mut shared_out = reference_expert_public(shared, x);
            if with_gate {
                let gate_weight = match &shared.output_gate {
                    Some(Qwen36MoeLinear::Dense(tensor)) => {
                        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                    }
                    _ => unreachable!(),
                };
                let logit: f32 = x.iter().zip(&gate_weight).map(|(a, b)| a * b).sum();
                shared_out = shared_out.iter().map(|v| v * (1.0 / (1.0 + (-logit).exp()))).collect();
            }

            for (o, value) in routed.iter().enumerate() {
                output[token * HIDDEN + o] = value + shared_out[o];
            }
        }
        output
    }

    fn reference_expert_public(
        weights: &Qwen36MoeSharedExpertWeights,
        x: &[f32],
    ) -> Vec<f32> {
        let as_values = |linear: &Qwen36MoeLinear| match linear {
            Qwen36MoeLinear::Dense(tensor) => tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            Qwen36MoeLinear::Quantized(_) => unreachable!(),
            Qwen36MoeLinear::CompactFp8 { .. } => unreachable!(),
        };
        let gate_out = project_dense(&as_values(&weights.gate), FF, HIDDEN, x);
        let up_out = project_dense(&as_values(&weights.up), FF, HIDDEN, x);
        let hidden: Vec<f32> = gate_out
            .iter()
            .zip(&up_out)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        project_dense(&as_values(&weights.down), HIDDEN, FF, &hidden)
    }

    fn run_block(with_gate: bool) -> Vec<f32> {
        let experts: Vec<_> = (0..NUM_EXPERTS).map(expert).collect();
        let shared = shared(with_gate);
        let block = Qwen36MoeSparseMlp::from_weights(
            dense_weight(NUM_EXPERTS, HIDDEN, &router_weight()),
            experts.clone(),
            Qwen36MoeSharedExpertWeights {
                gate: shared.gate.clone(),
                up: shared.up.clone(),
                down: shared.down.clone(),
                output_gate: shared.output_gate.clone(),
            },
            &geometry(FF),
        )
        .unwrap();
        let output = block.forward(&hidden_states()).unwrap();
        assert_eq!(output.dims(), [TOKENS, HIDDEN]);
        output.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn sparse_block_matches_reference_math_with_gated_shared_expert() {
        let output = run_block(true);
        let experts: Vec<_> = (0..NUM_EXPERTS).map(expert).collect();
        let reference = reference_forward(&experts, &shared(true), true);
        for (actual, expected) in output.iter().zip(&reference) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn sparse_block_matches_reference_math_without_shared_gate() {
        let output = run_block(false);
        let experts: Vec<_> = (0..NUM_EXPERTS).map(expert).collect();
        let reference = reference_forward(&experts, &shared(false), false);
        for (actual, expected) in output.iter().zip(&reference) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} vs {expected}"
            );
        }
    }

    #[test]
    fn sparse_block_records_full_routing_histogram() {
        let experts: Vec<_> = (0..NUM_EXPERTS).map(expert).collect();
        let block = Qwen36MoeSparseMlp::from_weights(
            dense_weight(NUM_EXPERTS, HIDDEN, &router_weight()),
            experts,
            shared(false),
            &geometry(FF),
        )
        .unwrap();
        block.forward(&hidden_states()).unwrap();
        // tokens × top-k selections land in the histogram.
        assert_eq!(block.counters().total_selections(), (TOKENS * TOP_K) as u64);
        assert_eq!(block.counters().num_experts(), NUM_EXPERTS);
    }

    #[test]
    fn sparse_block_rejects_geometry_mismatch() {
        let experts: Vec<_> = (0..NUM_EXPERTS - 1).map(expert).collect();
        let error = match Qwen36MoeSparseMlp::from_weights(
            dense_weight(NUM_EXPERTS, HIDDEN, &router_weight()),
            experts,
            shared(false),
            &geometry(FF),
        ) {
            Ok(_) => panic!("routed expert count must fail closed against geometry"),
            Err(error) => error,
        };
        assert!(format!("{error}").contains("geometry expects 4"));
    }

    mod fused {
        use super::super::*;
        use crate::models::architectures::qwen36moe::fused_moe::BACKEND_ENV;

        const H: usize = 256;
        const I: usize = 128;
        const E: usize = 6;
        const K: usize = 2;

        fn geometry(shared_ff: usize) -> Qwen36MoeFfnGeometry {
            Qwen36MoeFfnGeometry {
                num_experts: E,
                num_experts_per_tok: K,
                expert_intermediate_size: I,
                shared_expert_intermediate_size: shared_ff,
            }
        }

        fn stream(seed: u64) -> impl FnMut() -> u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            }
        }

        /// Raw block-FP8 projection `[rows, cols]` with finite E4M3 bytes and
        /// per-block scales, resident on the CPU.
        fn compact(rows: usize, cols: usize, seed: u64) -> Qwen36MoeLinear {
            let mut next = stream(seed);
            let bytes = (0..rows * cols)
                .map(|_| loop {
                    let byte = next() as u8;
                    if byte & 0x7f != 0x7f {
                        break byte;
                    }
                })
                .collect::<Vec<_>>();
            let scales = (0..(rows / 128) * (cols / 128))
                .map(|_| 0.002 + (next() % 1000) as f32 * 2e-6)
                .collect::<Vec<_>>();
            Qwen36MoeLinear::CompactFp8 {
                weights: Tensor::from_vec(bytes, (rows, cols), &Device::Cpu).unwrap(),
                scales: Tensor::from_vec(scales, (rows / 128, cols / 128), &Device::Cpu).unwrap(),
            }
        }

        fn experts() -> Vec<Qwen36MoeExpertWeights> {
            (0..E as u64)
                .map(|e| Qwen36MoeExpertWeights {
                    gate: compact(I, H, 10 + e * 3),
                    up: compact(I, H, 11 + e * 3),
                    down: compact(H, I, 12 + e * 3),
                })
                .collect()
        }

        fn shared(ff: usize, gated: bool) -> Qwen36MoeSharedExpertWeights {
            Qwen36MoeSharedExpertWeights {
                gate: compact(ff, H, 90),
                up: compact(ff, H, 91),
                down: compact(H, ff, 92),
                output_gate: gated.then(|| {
                    Qwen36MoeLinear::from_dense(
                        Tensor::from_vec(
                            (0..H)
                                .map(|i| ((i as f32) * 0.37).sin() * 0.05)
                                .collect::<Vec<_>>(),
                            (1, H),
                            &Device::Cpu,
                        )
                        .unwrap(),
                    )
                }),
            }
        }

        fn router() -> Qwen36MoeLinear {
            Qwen36MoeLinear::from_dense(
                Tensor::from_vec(
                    (0..E * H)
                        .map(|i| ((i as f32) * 0.618).sin() * 0.08)
                        .collect::<Vec<_>>(),
                    (E, H),
                    &Device::Cpu,
                )
                .unwrap(),
            )
        }

        fn input(tokens: usize) -> Tensor {
            Tensor::from_vec(
                (0..tokens * H)
                    .map(|i| ((i as f32) * 0.754_877_7).sin() * 1.5)
                    .collect::<Vec<_>>(),
                (tokens, H),
                &Device::Cpu,
            )
            .unwrap()
        }

        fn block(
            request: Qwen36MoeBackendRequest,
            shared_ff: usize,
            gated: bool,
        ) -> Qwen36MoeSparseMlp {
            Qwen36MoeSparseMlp::from_weights_with_backend(
                router(),
                experts(),
                shared(shared_ff, gated),
                &geometry(shared_ff),
                request,
            )
            .unwrap()
        }

        fn values(tensor: &Tensor) -> Vec<f32> {
            tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        }

        fn assert_matches_legacy(shared_ff: usize, gated: bool) {
            let fused = block(Qwen36MoeBackendRequest::Auto, shared_ff, gated);
            let legacy = block(Qwen36MoeBackendRequest::Legacy, shared_ff, gated);
            assert_eq!(fused.backend(), &Qwen36MoeBackend::Fused);
            assert!(matches!(legacy.backend(), Qwen36MoeBackend::Legacy { .. }));
            for tokens in [1, 4] {
                let x = input(tokens);
                let expected = values(&legacy.forward(&x).unwrap());
                let actual = values(&fused.forward(&x).unwrap());
                let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-4 * scale.max(1e-6),
                        "T={tokens} index {index}: fused {a} vs legacy {e}"
                    );
                }
            }
        }

        #[test]
        fn fused_block_matches_the_legacy_dispatcher_with_a_gated_shared_slot() {
            assert_matches_legacy(I, true);
        }

        #[test]
        fn fused_block_matches_the_legacy_dispatcher_with_an_ungated_shared_slot() {
            assert_matches_legacy(I, false);
        }

        #[test]
        fn fused_block_adds_a_shared_expert_it_cannot_fold() {
            // Shared width differs from the routed width: routed experts stay
            // fused, the shared expert runs separately and is added.
            assert_matches_legacy(2 * I, true);
        }

        #[test]
        fn fused_block_never_runs_the_host_dispatcher() {
            let fused = block(Qwen36MoeBackendRequest::Auto, I, true);
            fused.forward(&input(3)).unwrap();
            assert_eq!(
                fused.counters().total_selections(),
                0,
                "the fused path must not route through the host dispatcher"
            );
            let legacy = block(Qwen36MoeBackendRequest::Legacy, I, true);
            legacy.forward(&input(3)).unwrap();
            assert_eq!(legacy.counters().total_selections(), (3 * K) as u64);
        }

        #[test]
        fn legacy_switch_and_unsupported_residency_report_their_reason() {
            let legacy = block(Qwen36MoeBackendRequest::Legacy, I, true);
            assert!(
                matches!(legacy.backend(), Qwen36MoeBackend::Legacy { reason } if reason.contains(BACKEND_ENV))
            );
            let dense_experts = (0..E)
                .map(|_| Qwen36MoeExpertWeights {
                    gate: Qwen36MoeLinear::from_dense(
                        Tensor::zeros((I, H), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    up: Qwen36MoeLinear::from_dense(
                        Tensor::zeros((I, H), DType::F32, &Device::Cpu).unwrap(),
                    ),
                    down: Qwen36MoeLinear::from_dense(
                        Tensor::zeros((H, I), DType::F32, &Device::Cpu).unwrap(),
                    ),
                })
                .collect();
            let dense = Qwen36MoeSparseMlp::from_weights_with_backend(
                router(),
                dense_experts,
                shared(I, true),
                &geometry(I),
                Qwen36MoeBackendRequest::Auto,
            )
            .unwrap();
            assert!(
                matches!(dense.backend(), Qwen36MoeBackend::Legacy { reason } if reason.contains("block-FP8"))
            );
        }

        #[test]
        fn self_check_rejects_a_mismatched_expert_layout() {
            let Qwen36MoeStacking::Stacked {
                fused,
                experts,
                shared,
            } = Qwen36MoeFusedExperts::stack(&router(), experts(), shared(I, true), &geometry(I))
            else {
                panic!("block-FP8 experts must stack on the CPU");
            };
            assert!(fused.self_check(&experts, &shared).is_ok());
            // Rotating the reference experts makes every routed slot compare
            // against the wrong weights: the check must not be vacuous.
            let mut rotated = experts.clone();
            rotated.rotate_left(1);
            let error = fused
                .self_check(&rotated, &shared)
                .expect_err("a wrong expert layout must fail the self-check");
            assert!(error.contains("diverges"), "{error}");
        }
    }

    #[test]
    fn sparse_block_accepts_rank3_input() {
        let experts: Vec<_> = (0..NUM_EXPERTS).map(expert).collect();
        let block = Qwen36MoeSparseMlp::from_weights(
            dense_weight(NUM_EXPERTS, HIDDEN, &router_weight()),
            experts,
            shared(false),
            &geometry(FF),
        )
        .unwrap();
        let flat = run_block(false);
        let rank3_input = hidden_states().reshape((1, TOKENS, HIDDEN)).unwrap();
        let output = block.forward(&rank3_input).unwrap();
        assert_eq!(output.dims(), [1, TOKENS, HIDDEN]);
        let rank3_output = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in rank3_output.iter().zip(&flat) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

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
//! resolution at 40 is 0.25 and would reorder the experts. Ties order by
//! expert index.
//!
//! On Metal and CUDA (`w8a16`) the routed experts are stacked per layer and
//! run on the device-routed kernels (`kernels::moe_relu2`): router, top-k
//! and the experts never leave the device, and a row's output does not
//! depend on the other rows of the step. The CPU (Q8_0) and the `vllm-fp8`
//! parity mode route on the host (one read of `[tokens, experts]` F32
//! logits per layer) and run per expert over the tokens routed to them.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::NemotronCheckpoint;
use super::super::config::{MoeConfig, NemotronTranscribeConfig};
use super::super::linear::{NtLinear, Pass};
use super::super::precision::Fp8Mode;
use super::super::residency::{stacks_routed_experts, ResidencyBackend};
use super::super::speech_transformer::host_values;
use crate::error::{Error, Result};
use crate::kernels::moe_relu2::{self, StackedExperts};

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

/// How a layer's routed experts run.
enum RoutedExperts {
    /// One FP8 linear pair per expert, routed on the host: the CPU (Q8_0)
    /// and the `vllm-fp8` parity mode.
    PerExpert(Vec<Expert>),
    /// Stacked raw FP8 with one scale per projection, routed on the device
    /// (`kernels::moe_relu2`): Metal and CUDA in `w8a16`.
    Stacked {
        experts: StackedExperts,
        /// `[experts]` F32 selection bias on the device.
        bias: Tensor,
    },
}

/// Relative L2 bound of the load-time check of the stacked experts against
/// the checkpoint's own expert bytes.
const STACKED_CHECK_LIMIT: f64 = 1e-4;

pub struct MoeMixer {
    config: MoeConfig,
    /// `[experts, hidden]` F32 (the router GEMM runs in F32).
    gate: Tensor,
    bias: Vec<f32>,
    routed: RoutedExperts,
    shared: Expert,
    /// The stacked experts' load-time check (relative L2), when stacked and
    /// checked.
    stacked_check: Option<f64>,
}

impl MoeMixer {
    /// Loads layer `layer`'s mixer; `check` runs the stacked experts'
    /// load-time check (the decoder checks the first MoE layer).
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        layer: usize,
        device: &Device,
        mode: Fp8Mode,
        check: bool,
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
        let stack = stacks_routed_experts(residency_backend(device), mode)
            && moe_relu2::supported(device, moe.experts, d.hidden, moe.intermediate)
            && moe.top_k <= moe_relu2::MAX_TOP_K;
        let mut stacked_check = None;
        let routed = if stack {
            let (experts, check_error) =
                load_stacked(checkpoint, &x, moe, d.hidden, device, check)?;
            stacked_check = check_error;
            RoutedExperts::Stacked {
                experts,
                bias: Tensor::from_vec(bias.clone(), moe.experts, device)?,
            }
        } else {
            RoutedExperts::PerExpert(
                (0..moe.experts)
                    .map(|e| expert(format!("{x}.experts.{e}"), moe.intermediate))
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        Ok(Self {
            config: moe,
            gate: Tensor::from_vec(gate, (moe.experts, d.hidden), device)?,
            bias,
            shared: expert(format!("{x}.shared_experts"), moe.shared_intermediate)?,
            routed,
            stacked_check,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        let routed = match &self.routed {
            RoutedExperts::PerExpert(experts) => experts
                .iter()
                .map(|e| e.up.resident_bytes() + e.down.resident_bytes())
                .sum::<usize>(),
            RoutedExperts::Stacked { experts, .. } => experts.resident_bytes(),
        };
        routed
            + self.shared.up.resident_bytes()
            + self.shared.down.resident_bytes()
            + self.gate.elem_count() * 4
    }

    /// Whether the routed experts run on the device-routed kernels.
    pub fn device_routed(&self) -> bool {
        matches!(self.routed, RoutedExperts::Stacked { .. })
    }

    /// The stacked experts' load-time check, when it ran.
    pub fn stacked_check(&self) -> Option<f64> {
        self.stacked_check
    }

    /// Projections for the load-time FP8 checks: the first routed expert's
    /// (when per-expert) and the shared expert's, with their module names
    /// relative to the mixer.
    pub fn sample_projections(&self) -> Vec<(&'static str, &NtLinear)> {
        let mut out = Vec::with_capacity(4);
        if let RoutedExperts::PerExpert(experts) = &self.routed {
            out.push(("experts.0.up_proj", &experts[0].up));
            out.push(("experts.0.down_proj", &experts[0].down));
        }
        out.push(("shared_experts.up_proj", &self.shared.up));
        out.push(("shared_experts.down_proj", &self.shared.down));
        out
    }

    /// Router decisions for `x` `[tokens, hidden]`.
    pub fn routes(&self, x: &Tensor) -> Result<Vec<Routing>> {
        let logits = host_values(&x.to_dtype(DType::F32)?.matmul(&self.gate.t()?)?)?;
        Ok(self.routes_from(&logits))
    }

    fn routes_from(&self, logits: &[f32]) -> Vec<Routing> {
        logits
            .chunks_exact(self.config.experts)
            .map(|row| route(row, &self.bias, &self.config))
            .collect()
    }

    /// The mixer over `x` `[tokens, hidden]` F32. Returns the output and,
    /// when `want_routes` (traces), the tokens' routing; device-routed
    /// layers read their logits to the host only then.
    pub fn forward(
        &self,
        x: &Tensor,
        pass: Pass,
        want_routes: bool,
    ) -> Result<(Tensor, Vec<Routing>)> {
        let shared = self.shared.forward(x, pass)?;
        match &self.routed {
            RoutedExperts::Stacked { experts, bias } => {
                let x = x.to_dtype(DType::F32)?;
                let logits = moe_relu2::router_logits(&x, &self.gate)?;
                let spec = moe_relu2::SigmoidRouting {
                    experts: self.config.experts,
                    top_k: self.config.top_k,
                    scale: self.config.routed_scaling,
                };
                let routing = moe_relu2::route(&logits, bias, spec)?;
                let routed = moe_relu2::experts(&x, &routing, self.config.top_k, experts)?;
                let routes = if want_routes {
                    self.routes_from(&host_values(&logits)?)
                } else {
                    Vec::new()
                };
                Ok(((shared + routed)?, routes))
            }
            RoutedExperts::PerExpert(experts) => {
                let routes = self.routes(x)?;
                let mut assigned = vec![Vec::<(u32, f32)>::new(); self.config.experts];
                for (token, routing) in routes.iter().enumerate() {
                    for (&e, &w) in routing.experts.iter().zip(&routing.weights) {
                        assigned[e].push((token as u32, w));
                    }
                }
                let device = x.device();
                let mut out = shared;
                for (expert, tokens) in experts.iter().zip(&assigned) {
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
                Ok((out, if want_routes { routes } else { Vec::new() }))
            }
        }
    }
}

fn residency_backend(device: &Device) -> ResidencyBackend {
    match device {
        Device::Cpu => ResidencyBackend::Cpu,
        Device::Metal(_) => ResidencyBackend::Metal,
        Device::Cuda(_) => ResidencyBackend::Cuda,
    }
}

/// The layer's routed experts stacked on `device`: every expert's E4M3
/// bytes copied straight from the checkpoint, one scale per projection.
/// With `check`, two probe tokens routed to the first, middle and last
/// experts run on the device and are compared with the reference computed
/// from those experts' own checkpoint bytes, so a wrong stacking offset or
/// kernel fails the load.
fn load_stacked(
    checkpoint: &NemotronCheckpoint,
    mixer: &str,
    moe: MoeConfig,
    hidden: usize,
    device: &Device,
    check: bool,
) -> Result<(StackedExperts, Option<f64>)> {
    let inter = moe.intermediate;
    let mut up = Vec::with_capacity(moe.experts * inter * hidden);
    let mut down = Vec::with_capacity(moe.experts * hidden * inter);
    let mut up_scale = Vec::with_capacity(moe.experts);
    let mut down_scale = Vec::with_capacity(moe.experts);
    for e in 0..moe.experts {
        let u = checkpoint.fp8(&format!("{mixer}.experts.{e}.up_proj"), inter, hidden)?;
        up.extend_from_slice(u.bytes);
        up_scale.push(u.weight_scale);
        let w = checkpoint.fp8(&format!("{mixer}.experts.{e}.down_proj"), hidden, inter)?;
        down.extend_from_slice(w.bytes);
        down_scale.push(w.weight_scale);
    }
    let experts = StackedExperts::new(
        Tensor::from_vec(up, (moe.experts, inter, hidden), device)?,
        Tensor::from_vec(up_scale, moe.experts, device)?,
        Tensor::from_vec(down, (moe.experts, hidden, inter), device)?,
        Tensor::from_vec(down_scale, moe.experts, device)?,
    )?;
    if !check {
        return Ok((experts, None));
    }
    let probe = [0, moe.experts / 2, moe.experts - 1];
    let mut ref_up = Vec::new();
    let mut ref_down = Vec::new();
    let mut ref_up_scale = Vec::new();
    let mut ref_down_scale = Vec::new();
    for &e in &probe {
        let u = checkpoint.fp8(&format!("{mixer}.experts.{e}.up_proj"), inter, hidden)?;
        ref_up.extend_from_slice(u.bytes);
        ref_up_scale.push(u.weight_scale);
        let w = checkpoint.fp8(&format!("{mixer}.experts.{e}.down_proj"), hidden, inter)?;
        ref_down.extend_from_slice(w.bytes);
        ref_down_scale.push(w.weight_scale);
    }
    let x = crate::kernels::numeric_check::xorshift_values(2 * hidden, 0x6e74, 1.0);
    // Token 0 takes the first and last experts, token 1 the middle and
    // first; the reference numbers them 0..3 in probe order.
    let device_routing = [
        probe[0] as f32,
        probe[2] as f32,
        0.75,
        0.5,
        probe[1] as f32,
        probe[0] as f32,
        1.25,
        0.25,
    ];
    let reference_routing = [0.0, 2.0, 0.75, 0.5, 1.0, 0.0, 1.25, 0.25];
    let got = host_values(&moe_relu2::experts(
        &Tensor::from_vec(x.clone(), (2, hidden), device)?,
        &Tensor::from_vec(device_routing.to_vec(), (2, 4), device)?,
        2,
        &experts,
    )?)?;
    let want = moe_relu2::experts_reference(
        &x,
        &reference_routing,
        2,
        &ref_up,
        &ref_up_scale,
        &ref_down,
        &ref_down_scale,
        hidden,
        inter,
    );
    let error = crate::kernels::numeric_check::relative_l2(&got, &want);
    if error.is_nan() || error > STACKED_CHECK_LIMIT {
        return Err(Error::ModelLoadError(format!(
            "Nemotron 3.5 Transcribe: {mixer}'s stacked experts deviate from the checkpoint by {error:.3e}"
        )));
    }
    Ok((experts, Some(error)))
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

    /// Times the device-routed MoE against the per-expert loop it replaces,
    /// on the same raw FP8 experts in the published geometry (32 of the 128
    /// experts, to keep the setup short), for one decode token, an 8-row
    /// decode step and a 512-token prefill span. Run with `--ignored
    /// --nocapture` on a Metal machine.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "Metal benchmark; run explicitly"]
    fn device_routed_moe_versus_the_per_expert_loop_on_metal() {
        use super::super::super::checkpoint::Fp8Weight;
        use crate::kernels::numeric_check::xorshift_values;
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip("MoE benchmark")
        else {
            return;
        };
        let (experts, hidden, inter, top_k) = (32usize, 2688usize, 1856usize, 6usize);
        // Random E4M3 bytes without NaN (0x7f / 0xff) and a plausible scale.
        let bytes = |len: usize, seed: u64| {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let b = (state >> 24) as u8 & 0xF7;
                    if b & 0x7f == 0x7f {
                        0x30
                    } else {
                        b
                    }
                })
                .collect::<Vec<u8>>()
        };
        let up = bytes(experts * inter * hidden, 1);
        let down = bytes(experts * hidden * inter, 2);
        let scale = 1e-3f32;
        let stacked = StackedExperts::new(
            Tensor::from_vec(up.clone(), (experts, inter, hidden), &device).unwrap(),
            Tensor::from_vec(vec![scale; experts], experts, &device).unwrap(),
            Tensor::from_vec(down.clone(), (experts, hidden, inter), &device).unwrap(),
            Tensor::from_vec(vec![scale; experts], experts, &device).unwrap(),
        )
        .unwrap();
        let weight = |data: &[u8], rows: usize, cols: usize| {
            NtLinear::from_fp8(
                Fp8Weight {
                    bytes: data,
                    rows,
                    cols,
                    weight_scale: scale,
                    input_scale: 1.0,
                },
                rows,
                cols,
                &device,
                Fp8Mode::W8A16,
            )
            .unwrap()
        };
        let per_expert = (0..experts)
            .map(|e| Expert {
                up: weight(
                    &up[e * inter * hidden..(e + 1) * inter * hidden],
                    inter,
                    hidden,
                ),
                down: weight(
                    &down[e * hidden * inter..(e + 1) * hidden * inter],
                    hidden,
                    inter,
                ),
            })
            .collect::<Vec<_>>();
        let gate_values = xorshift_values(experts * hidden, 3, 0.02);
        let gate = Tensor::from_vec(gate_values, (experts, hidden), &device).unwrap();
        let bias_values = vec![40.69f32; experts];
        let bias = Tensor::from_vec(bias_values.clone(), experts, &device).unwrap();
        let config = MoeConfig {
            experts,
            top_k,
            intermediate: inter,
            shared_intermediate: inter,
            routed_scaling: 2.5,
        };
        let device_routed = |x: &Tensor| {
            let logits = moe_relu2::router_logits(x, &gate).unwrap();
            let spec = moe_relu2::SigmoidRouting {
                experts,
                top_k,
                scale: 2.5,
            };
            let routing = moe_relu2::route(&logits, &bias, spec).unwrap();
            moe_relu2::experts(x, &routing, top_k, &stacked).unwrap()
        };
        let looped = |x: &Tensor| {
            let logits = host_values(&x.matmul(&gate.t().unwrap()).unwrap()).unwrap();
            let routes = logits
                .chunks_exact(experts)
                .map(|row| route(row, &bias_values, &config))
                .collect::<Vec<_>>();
            let mut assigned = vec![Vec::<(u32, f32)>::new(); experts];
            for (token, routing) in routes.iter().enumerate() {
                for (&e, &w) in routing.experts.iter().zip(&routing.weights) {
                    assigned[e].push((token as u32, w));
                }
            }
            let mut out = x.zeros_like().unwrap();
            for (expert, tokens) in per_expert.iter().zip(&assigned) {
                if tokens.is_empty() {
                    continue;
                }
                let ids = Tensor::from_vec(
                    tokens.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
                    tokens.len(),
                    &device,
                )
                .unwrap();
                let weights = Tensor::from_vec(
                    tokens.iter().map(|(_, w)| *w).collect::<Vec<_>>(),
                    (tokens.len(), 1),
                    &device,
                )
                .unwrap();
                let pass = if tokens.len() > 1 {
                    Pass::Span
                } else {
                    Pass::Step
                };
                let y = expert
                    .forward(&x.index_select(&ids, 0).unwrap(), pass)
                    .unwrap()
                    .broadcast_mul(&weights)
                    .unwrap();
                out = out.index_add(&ids, &y, 0).unwrap();
            }
            out
        };
        let time = |f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor| {
            for _ in 0..2 {
                f(x);
            }
            device.synchronize().unwrap();
            let iterations = 10;
            let started = std::time::Instant::now();
            for _ in 0..iterations {
                f(x);
            }
            device.synchronize().unwrap();
            started.elapsed().as_secs_f64() * 1e3 / iterations as f64
        };
        for tokens in [1usize, 8, 512] {
            let x = Tensor::from_vec(
                xorshift_values(tokens * hidden, 4, 1.0),
                (tokens, hidden),
                &device,
            )
            .unwrap();
            let error = crate::kernels::numeric_check::relative_l2(
                &host_values(&device_routed(&x)).unwrap(),
                &host_values(&looped(&x)).unwrap(),
            );
            let (fast, slow) = (time(&device_routed, &x), time(&looped, &x));
            eprintln!(
                "{tokens:>4} tokens: device-routed {fast:8.3} ms, per-expert loop {slow:8.3} ms ({:.1}x), rel L2 {error:.2e}",
                slow / fast
            );
            assert!(error < 1e-3, "{error}");
        }
    }
}

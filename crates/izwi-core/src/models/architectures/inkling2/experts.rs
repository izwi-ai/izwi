//! The sparse expert feed-forward of one layer, on the host.
//!
//! Routing runs in F32: `scores = sigmoid(W x)`; the top `k` experts are
//! picked by `scores + bias`; their weights are the raw `scores`,
//! normalised to sum to one and multiplied by `route_scale`. (The bias only
//! moves the choice, never the weight.) Each chosen expert computes
//! `down(act(gate_up(x)))` straight from its packed weights, and the outputs
//! are summed in F32 in ascending expert order, so a row's result does not
//! depend on what else is in the batch.
//!
//! This is the reference expert path on every device and the serving path on
//! the CPU, and on Metal unless the device path is asked for.
//!
//! On CUDA and Metal the layer can run on the device instead
//! ([`Inkling2Moe::upload`]): the packed bytes are uploaded as they are and
//! `kernels::moe_packed` routes and multiplies there, so a step never reads
//! activations back for the experts. The model checks that path against this
//! one at load. It is the default on CUDA. On Metal it is opt-in: the upload
//! is a second copy of the experts in unified memory beside the mapped file,
//! and which path is faster there has not been measured on the real model.

use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

use super::config::Inkling2Config;
use super::quant::{ExpertFormat, PackedExperts};
use crate::error::{Error, Result};
use crate::kernels::moe_packed::{self, PackedFormat, PackedWeights};

/// A layer's routers and packed experts on a device.
#[derive(Debug, Clone)]
struct DeviceExperts {
    /// `[hidden, experts]`.
    router: Tensor,
    bias: Tensor,
    gate_up: PackedWeights,
    down: PackedWeights,
}

fn upload_projection(experts: &PackedExperts, device: &Device) -> Result<PackedWeights> {
    let format = match experts.format() {
        ExpertFormat::Nvfp4 => PackedFormat::Nvfp4,
        ExpertFormat::Mxfp8 => PackedFormat::Mxfp8,
    };
    let (count, rows, columns) = (experts.experts(), experts.rows(), experts.columns());
    let factors = match experts.format() {
        ExpertFormat::Nvfp4 => experts.factors().to_vec(),
        ExpertFormat::Mxfp8 => vec![1.0f32; count],
    };
    let weights = PackedWeights {
        format,
        // The stored bytes go up as they are: no host conversion.
        codes: Tensor::from_raw_buffer(
            experts.codes().as_slice(),
            DType::U8,
            &[count, rows, format.code_bytes(columns)],
            device,
        )?,
        scales: Tensor::from_raw_buffer(
            experts.scales().as_slice(),
            DType::U8,
            &[count, rows, columns / format.block()],
            device,
        )?,
        factors: Tensor::from_vec(factors, count, device)?,
    };
    weights.validate()?;
    Ok(weights)
}

/// The experts a token uses and their weights, ascending by expert.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Route {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Inkling2Moe {
    hidden: usize,
    width: usize,
    top_k: usize,
    gate_cap: f32,
    up_cap: f32,
    route_scale: f32,
    /// `[experts, hidden]`, row major.
    router_weight: Vec<f32>,
    /// Added to the scores for selection only.
    router_bias: Vec<f32>,
    gate_up: PackedExperts,
    down: PackedExperts,
    /// The layer on a device, when it runs there.
    device: Option<DeviceExperts>,
}

impl Inkling2Moe {
    pub fn new(
        config: &Inkling2Config,
        router_weight: Vec<f32>,
        router_bias: Vec<f32>,
        gate_up: PackedExperts,
        down: PackedExperts,
    ) -> Result<Self> {
        let (experts, hidden, width) = (config.experts, config.hidden, config.expert_width);
        let fits = router_weight.len() == experts * hidden
            && router_bias.len() == experts
            && (gate_up.experts(), gate_up.rows(), gate_up.columns())
                == (experts, 2 * width, hidden)
            && (down.experts(), down.rows(), down.columns()) == (experts, hidden, width);
        if !fits {
            return Err(Error::ModelLoadError(
                "Inkling 2 experts: tensor shapes disagree with the config".to_string(),
            ));
        }
        Ok(Self {
            hidden,
            width,
            top_k: config.experts_per_token,
            gate_cap: config.gate_softcap as f32,
            up_cap: config.up_softcap as f32,
            route_scale: config.route_scale as f32,
            router_weight,
            router_bias,
            gate_up,
            down,
            device: None,
        })
    }

    /// Whether `kernels::moe_packed` takes this layer's geometry.
    pub fn device_supported(&self) -> bool {
        moe_packed::supported(self.hidden, self.width, self.router_bias.len(), self.top_k)
    }

    /// Puts the routers and the packed experts on `device`; the layer then
    /// runs there ([`Self::forward_tensor`]).
    pub fn upload(&mut self, device: &Device) -> Result<()> {
        if !self.device_supported() {
            return Err(Error::ModelLoadError(
                "Inkling 2 experts: the device kernels do not take this geometry".to_string(),
            ));
        }
        let experts = self.router_bias.len();
        self.device = Some(DeviceExperts {
            router: Tensor::from_vec(self.router_weight.clone(), (experts, self.hidden), device)?
                .t()?
                .contiguous()?,
            bias: Tensor::from_vec(self.router_bias.clone(), experts, device)?,
            gate_up: upload_projection(&self.gate_up, device)?,
            down: upload_projection(&self.down, device)?,
        });
        Ok(())
    }

    /// Back to the host path; frees the device copy.
    pub fn drop_device(&mut self) {
        self.device = None;
    }

    pub fn on_device(&self) -> bool {
        self.device.is_some()
    }

    /// Bytes the device copy holds.
    pub fn device_resident_bytes(&self) -> u64 {
        self.device.as_ref().map_or(0, |device| {
            (device.gate_up.resident_bytes()
                + device.down.resident_bytes()
                + (device.router.elem_count() + device.bias.elem_count()) * 4) as u64
        })
    }

    /// The layer over `x` (`[rows, hidden]`, F32): on the device when
    /// uploaded, else on the host with one read and one upload.
    pub fn forward_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, hidden) = x.dims2()?;
        match &self.device {
            Some(device) => {
                let logits = x.matmul(&device.router)?;
                let routing =
                    moe_packed::route(&logits, &device.bias, self.top_k, self.route_scale)?;
                Ok(moe_packed::experts(
                    x,
                    &routing,
                    &device.gate_up,
                    &device.down,
                    self.gate_cap,
                    self.up_cap,
                )?)
            }
            None => {
                let host = x.flatten_all()?.to_vec1::<f32>()?;
                Ok(Tensor::from_vec(
                    self.forward(&host),
                    (rows, hidden),
                    x.device(),
                )?)
            }
        }
    }

    pub fn gate_up(&self) -> &PackedExperts {
        &self.gate_up
    }

    pub fn down(&self) -> &PackedExperts {
        &self.down
    }

    pub fn router_weight(&self) -> &[f32] {
        &self.router_weight
    }

    pub fn router_bias(&self) -> &[f32] {
        &self.router_bias
    }

    /// Bytes the expert weights keep resident.
    pub fn resident_bytes(&self) -> u64 {
        self.gate_up.resident_bytes() + self.down.resident_bytes()
    }

    /// The experts of one token.
    pub fn route(&self, x: &[f32]) -> Inkling2Route {
        let scores = self
            .router_weight
            .chunks_exact(self.hidden)
            .map(|row| {
                let logit = row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
                1.0 / (1.0 + (-logit).exp())
            })
            .collect::<Vec<_>>();
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        // Highest `score + bias` first; ties go to the lower expert.
        order.sort_by(|a, b| {
            let (left, right) = (
                scores[*a] + self.router_bias[*a],
                scores[*b] + self.router_bias[*b],
            );
            right.total_cmp(&left).then(a.cmp(b))
        });
        let mut experts = order[..self.top_k].to_vec();
        experts.sort_unstable();
        let total = experts.iter().map(|expert| scores[*expert]).sum::<f32>() + 1e-20;
        let weights = experts
            .iter()
            .map(|expert| scores[*expert] / total * self.route_scale)
            .collect();
        Inkling2Route { experts, weights }
    }

    /// One expert's weighted output for one token.
    fn expert_output(&self, expert: usize, weight: f32, x: &[f32]) -> Vec<f32> {
        let mut act = vec![0.0f32; self.width];
        self.gate_up
            .gate_up(expert, x, self.gate_cap, self.up_cap, &mut act);
        let mut out = vec![0.0f32; self.hidden];
        self.down.matvec(expert, &act, &mut out);
        for value in &mut out {
            *value *= weight;
        }
        out
    }

    /// `x` is `[rows, hidden]` row major; so is the result.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.hidden, 0);
        let rows = x.len() / self.hidden;
        let routes = x
            .chunks_exact(self.hidden)
            .map(|row| self.route(row))
            .collect::<Vec<_>>();
        let tasks = routes
            .iter()
            .enumerate()
            .flat_map(|(row, route)| {
                route
                    .experts
                    .iter()
                    .zip(&route.weights)
                    .map(move |(expert, weight)| (row, *expert, *weight))
            })
            .collect::<Vec<_>>();
        let outputs = tasks
            .par_iter()
            .map(|(row, expert, weight)| {
                self.expert_output(
                    *expert,
                    *weight,
                    &x[row * self.hidden..(row + 1) * self.hidden],
                )
            })
            .collect::<Vec<_>>();
        // Tasks are in (row, ascending expert) order: a fixed sum order.
        let mut out = vec![0.0f32; rows * self.hidden];
        for ((row, _, _), output) in tasks.iter().zip(&outputs) {
            for (total, value) in out[row * self.hidden..(row + 1) * self.hidden]
                .iter_mut()
                .zip(output)
            {
                *total += value;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::checkpoint::{ExpertProjection, Inkling2Checkpoint};
    use super::super::config::tests::{fixture, tiny};
    use super::super::quant::softcapped_swiglu;
    use super::*;

    fn moe(layer: usize) -> (Inkling2Config, Inkling2Moe) {
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
        let prefix = format!("model.llm.layers.{layer}.mlp");
        let moe = Inkling2Moe::new(
            &config,
            checkpoint
                .host_f32(&format!("{prefix}.gate.weight"))
                .unwrap(),
            checkpoint.host_f32(&format!("{prefix}.gate.bias")).unwrap(),
            checkpoint.experts(layer, ExpertProjection::GateUp).unwrap(),
            checkpoint.experts(layer, ExpertProjection::Down).unwrap(),
        )
        .unwrap();
        (config, moe)
    }

    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_bias_moves_the_choice_but_not_the_weights() {
        let (config, mut moe) = moe(0);
        let x = noise(config.hidden, 7);
        let plain = moe.route(&x);
        assert_eq!(plain.experts.len(), config.experts_per_token);
        assert!(plain.experts.windows(2).all(|pair| pair[0] < pair[1]));
        let total = plain.weights.iter().sum::<f32>();
        assert!((total - config.route_scale as f32).abs() < 1e-4, "{total}");

        // A large bias on an unchosen expert forces it in; its weight is
        // still its raw score's share.
        let forced = (0..config.experts)
            .find(|expert| !plain.experts.contains(expert))
            .unwrap();
        moe.router_bias[forced] = 100.0;
        let biased = moe.route(&x);
        assert!(biased.experts.contains(&forced));
        let slot = biased.experts.iter().position(|e| *e == forced).unwrap();
        assert!(biased.weights[slot] < config.route_scale as f32);
        let total = biased.weights.iter().sum::<f32>();
        assert!((total - config.route_scale as f32).abs() < 1e-4, "{total}");
    }

    /// The packed path equals a dense F64 evaluation of the same formula,
    /// for both expert formats, and a row's output does not depend on its
    /// neighbours.
    #[test]
    fn packed_experts_match_a_dense_evaluation() {
        for layer in [0, 7] {
            let (config, moe) = moe(layer);
            let rows = 5;
            let x = noise(rows * config.hidden, 11 + layer as u32);
            let got = moe.forward(&x);
            for row in 0..rows {
                let x = &x[row * config.hidden..(row + 1) * config.hidden];
                let route = moe.route(x);
                let mut want = vec![0.0f64; config.hidden];
                for (expert, weight) in route.experts.iter().zip(&route.weights) {
                    let w13 = moe.gate_up.dequantize_expert(*expert);
                    let w2 = moe.down.dequantize_expert(*expert);
                    let dot = |row: &[f32], x: &[f32]| {
                        row.iter()
                            .zip(x)
                            .map(|(w, x)| *w as f64 * *x as f64)
                            .sum::<f64>() as f32
                    };
                    let act = (0..config.expert_width)
                        .map(|i| {
                            let gate =
                                dot(&w13[2 * i * config.hidden..(2 * i + 1) * config.hidden], x);
                            let up = dot(
                                &w13[(2 * i + 1) * config.hidden..(2 * i + 2) * config.hidden],
                                x,
                            );
                            softcapped_swiglu(gate, up, 4.0, 25.0)
                        })
                        .collect::<Vec<_>>();
                    for (h, want) in want.iter_mut().enumerate() {
                        let row = &w2[h * config.expert_width..(h + 1) * config.expert_width];
                        *want += dot(row, &act) as f64 * *weight as f64;
                    }
                }
                let got_row = &got[row * config.hidden..(row + 1) * config.hidden];
                let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!(scale > 0.0);
                for (got, want) in got_row.iter().zip(&want) {
                    assert!(
                        (*got as f64 - want).abs() <= 2e-5 * scale.max(1.0),
                        "{got} vs {want}"
                    );
                }
                // Alone, the row gives the same bits.
                let alone = moe.forward(x);
                assert_eq!(
                    alone.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    got_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

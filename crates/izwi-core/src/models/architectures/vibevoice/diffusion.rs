//! VibeVoice diffusion prediction head and scheduler helpers.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::kernels::{try_fused_rms_norm, try_fused_silu_mul};
use crate::models::architectures::vibevoice::config::VibeVoiceDiffusionHeadConfig;
use crate::models::shared::weights::mlx;

pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    frequency_embedding_size: usize,
    frequencies: Tensor,
}

impl TimestepEmbedder {
    pub fn load(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let frequency_embedding_size = 256;
        let frequencies = timestep_frequencies(frequency_embedding_size, vb.device())?;
        let linear_1 =
            mlx::load_linear_no_bias(frequency_embedding_size, hidden_size, vb.pp("mlp.0"))?;
        let linear_2 = mlx::load_linear_no_bias(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            frequency_embedding_size,
            frequencies,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let emb = timestep_embedding(timesteps, self.frequency_embedding_size, &self.frequencies)?;
        let emb = self.linear_1.forward(&emb)?;
        let emb = ops::silu(&emb)?;
        self.linear_2.forward(&emb).map_err(Error::from)
    }
}

pub struct VibeVoiceDiffusionHead {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    t_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
    cfg: VibeVoiceDiffusionHeadConfig,
}

impl VibeVoiceDiffusionHead {
    pub fn load(cfg: VibeVoiceDiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let noisy_images_proj =
            mlx::load_linear_no_bias(cfg.latent_size, cfg.hidden_size, vb.pp("noisy_images_proj"))?;
        let cond_proj =
            mlx::load_linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("cond_proj"))?;
        let t_embedder = TimestepEmbedder::load(cfg.hidden_size, vb.pp("t_embedder"))?;
        let ffn_dim = ((cfg.hidden_size as f32) * cfg.head_ffn_ratio).round() as usize;
        let mut layers = Vec::with_capacity(cfg.head_layers);
        for idx in 0..cfg.head_layers {
            layers.push(HeadLayer::load(
                cfg.hidden_size,
                ffn_dim,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let final_layer = FinalLayer::load(
            cfg.hidden_size,
            cfg.latent_size,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("final_layer"),
        )?;
        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
            cfg,
        })
    }

    pub fn config(&self) -> &VibeVoiceDiffusionHeadConfig {
        &self.cfg
    }

    pub fn forward(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let timestep_embedding = self.timestep_embedding(timesteps)?;
        let condition = self.condition_projection(condition)?;
        self.forward_with_precomputed(noisy_latents, &timestep_embedding, &condition)
    }

    pub fn timestep_embedding(&self, timesteps: &Tensor) -> Result<Tensor> {
        self.t_embedder.forward(timesteps)
    }

    pub fn condition_projection(&self, condition: &Tensor) -> Result<Tensor> {
        self.cond_proj.forward(condition).map_err(Error::from)
    }

    pub fn forward_with_precomputed(
        &self,
        noisy_latents: &Tensor,
        timestep_embedding: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.noisy_images_proj.forward(noisy_latents)?;
        let mut t = timestep_embedding.clone();
        if condition.rank() == 3 && t.rank() == 2 {
            t = t.unsqueeze(1)?;
        }
        let c = condition.broadcast_add(&t)?;
        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }
        self.final_layer.forward(&x, &c)
    }

    pub fn forward_cfg_batched(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: &Tensor,
    ) -> Result<Tensor> {
        let timestep_embedding = self.timestep_embedding(timesteps)?;
        let condition = self.condition_projection(condition)?;
        let negative_condition = self.condition_projection(negative_condition)?;
        self.forward_cfg_batched_with_precomputed(
            noisy_latents,
            &timestep_embedding,
            &condition,
            &negative_condition,
            cfg_scale,
        )
    }

    pub fn forward_cfg_batched_with_precomputed(
        &self,
        noisy_latents: &Tensor,
        timestep_embedding: &Tensor,
        condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: &Tensor,
    ) -> Result<Tensor> {
        let latents = Tensor::cat(&[noisy_latents.clone(), noisy_latents.clone()], 0)?;
        let timestep_embeddings =
            Tensor::cat(&[timestep_embedding.clone(), timestep_embedding.clone()], 0)?;
        let conditions = Tensor::cat(&[condition.clone(), negative_condition.clone()], 0)?;
        self.forward_cfg_batched_with_prebatched(
            &latents,
            &timestep_embeddings,
            &conditions,
            cfg_scale,
        )
    }

    pub fn forward_cfg_batched_with_prebatched(
        &self,
        batched_latents: &Tensor,
        batched_timestep_embedding: &Tensor,
        batched_conditions: &Tensor,
        cfg_scale: &Tensor,
    ) -> Result<Tensor> {
        let output = self.forward_with_precomputed(
            batched_latents,
            batched_timestep_embedding,
            batched_conditions,
        )?;
        let positive = output.narrow(0, 0, 1)?;
        let negative = output.narrow(0, 1, 1)?;
        classifier_free_guidance(&positive, &negative, cfg_scale)
    }
}

struct FeedForwardNetwork {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForwardNetwork {
    fn load(embed_dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: mlx::load_linear_no_bias(embed_dim, ffn_dim, vb.pp("gate_proj"))?,
            up_proj: mlx::load_linear_no_bias(embed_dim, ffn_dim, vb.pp("up_proj"))?,
            down_proj: mlx::load_linear_no_bias(ffn_dim, embed_dim, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = if let Some(fused) = try_fused_silu_mul(&gate, &up) {
            fused
        } else {
            ops::silu(&gate)?.broadcast_mul(&up)?
        };
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

struct HeadLayer {
    ffn: FeedForwardNetwork,
    norm: RmsNorm,
    ada_ln: Linear,
    embed_dim: usize,
}

impl HeadLayer {
    fn load(
        embed_dim: usize,
        ffn_dim: usize,
        cond_dim: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            ffn: FeedForwardNetwork::load(embed_dim, ffn_dim, vb.pp("ffn"))?,
            norm: candle_nn::rms_norm(embed_dim, eps, vb.pp("norm"))?,
            ada_ln: mlx::load_linear_no_bias(cond_dim, 3 * embed_dim, vb.pp("adaLN_modulation.1"))?,
            embed_dim,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln.forward(&ops::silu(c)?)?;
        let shift = modulation.narrow(D::Minus1, 0, self.embed_dim)?;
        let scale = modulation.narrow(D::Minus1, self.embed_dim, self.embed_dim)?;
        let gate = modulation.narrow(D::Minus1, self.embed_dim * 2, self.embed_dim)?;
        let normed = self.norm.forward(x)?;
        let modulated = modulate(&normed, &shift, &scale)?;
        let update = self.ffn.forward(&modulated)?.broadcast_mul(&gate)?;
        x.broadcast_add(&update).map_err(Error::from)
    }
}

struct FinalLayer {
    linear: Linear,
    ada_ln: Linear,
    unit_norm_weight: Tensor,
    hidden_size: usize,
    eps: f64,
}

impl FinalLayer {
    fn load(
        hidden_size: usize,
        output_size: usize,
        cond_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let unit_norm_weight = Tensor::ones((hidden_size,), vb.dtype(), vb.device())?;
        Ok(Self {
            linear: mlx::load_linear_no_bias(hidden_size, output_size, vb.pp("linear"))?,
            ada_ln: mlx::load_linear_no_bias(
                cond_size,
                2 * hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?,
            unit_norm_weight,
            hidden_size,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln.forward(&ops::silu(c)?)?;
        let shift = modulation.narrow(D::Minus1, 0, self.hidden_size)?;
        let scale = modulation.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        let normed = rms_norm_no_affine(x, self.eps, &self.unit_norm_weight)?;
        let modulated = modulate(&normed, &shift, &scale)?;
        self.linear.forward(&modulated).map_err(Error::from)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one = Tensor::new(1f32, x.device())?.to_dtype(scale.dtype())?;
    let scaled = x.broadcast_mul(&scale.broadcast_add(&one)?)?;
    scaled.broadcast_add(shift).map_err(Error::from)
}

/// The final layer's `RMSNorm(elementwise_affine=False)`. The reference
/// normalizes `x.float()` and casts back (`_norm(x.float()).type_as(x)`),
/// and so does this: the final layer's input reaches about 480 on the
/// VibeVoice-1.5B checkpoint, so `x²` in F16 overflowed to infinity and
/// zeroed the normalized row, which made every F16 (Metal) speech latent
/// wrong.
fn rms_norm_no_affine(x: &Tensor, eps: f64, unit_weight: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        let unit_weight = unit_weight.to_device(x.device())?.to_dtype(x.dtype())?;
        if let Some(output) = try_fused_rms_norm(x, &unit_weight, eps) {
            return Ok(output);
        }
    }
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let denom = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&denom)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn classifier_free_guidance(
    positive_output: &Tensor,
    negative_output: &Tensor,
    cfg_scale: &Tensor,
) -> Result<Tensor> {
    let guidance = positive_output.broadcast_sub(negative_output)?;
    negative_output
        .broadcast_add(&guidance.broadcast_mul(cfg_scale)?)
        .map_err(Error::from)
}

fn timestep_frequencies(dim: usize, device: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut freqs = Vec::with_capacity(half);
    for idx in 0..half {
        freqs.push((-((10_000f32).ln()) * (idx as f32) / (half as f32)).exp());
    }
    Tensor::from_vec(freqs, (half,), device).map_err(Error::from)
}

fn timestep_embedding(t: &Tensor, dim: usize, frequencies: &Tensor) -> Result<Tensor> {
    let dtype = t.dtype();
    let t = t.flatten_all()?.to_dtype(DType::F32)?;
    let half = dim / 2;
    let frequencies = frequencies.narrow(0, 0, half)?;
    let args = t.unsqueeze(1)?.broadcast_mul(&frequencies.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let mut emb = Tensor::cat(&[&cos, &sin], 1)?;
    if dim % 2 == 1 {
        let zeros = Tensor::zeros((emb.dim(0)?, 1), DType::F32, t.device())?;
        emb = Tensor::cat(&[&emb, &zeros], 1)?;
    }
    emb.to_dtype(dtype).map_err(Error::from)
}

/// DPM-Solver++(2M), the reference's noise scheduler:
/// `DPMSolverMultistepScheduler(num_train_timesteps=1000,
/// beta_schedule="cosine", prediction_type="v_prediction")` with the
/// diffusers defaults (solver order 2, `dpmsolver++`, midpoint,
/// `lower_order_final`, `final_sigmas_type="zero"`, linspace spacing) as
/// vibevoice `schedule/dpm_solver.py` implements it.
///
/// Every step is affine in the current sample, the current data
/// prediction and the previous one, so the per-step scalars are computed
/// once on the host in f64 and each step is three fused multiply-adds in
/// F32 (the reference upcasts the sample to F32 for the update).
pub struct VibeVoiceDiffusionScheduler {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    coefficients: Vec<DpmSolverCoefficients>,
}

/// One step: `x0 = data_sample * x + data_model * v` (the v-prediction
/// data estimate), then `x' = sample * x + current * x0 + previous * x0_prev`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DpmSolverCoefficients {
    data_sample: f64,
    data_model: f64,
    sample: f64,
    current: f64,
    previous: f64,
}

pub struct VibeVoiceDiffusionStepTensors {
    pub timestep: usize,
    pub timestep_tensor: Tensor,
    data_sample: Tensor,
    data_model: Tensor,
    sample: Tensor,
    current: Tensor,
    previous: Option<Tensor>,
}

/// The multistep history of one sampling trajectory: the previous step's
/// data prediction (F32). Start each latent with a fresh state.
#[derive(Default)]
pub struct DpmSolverState {
    previous_data: Option<Tensor>,
}

impl VibeVoiceDiffusionScheduler {
    pub fn new(num_train_timesteps: usize, num_inference_steps: usize) -> Self {
        let train = num_train_timesteps.max(1);
        let alphas_cumprod = cosine_alphas_cumprod(train);
        let steps = num_inference_steps.max(1);
        // np.linspace(0, train - 1, steps + 1).round()[::-1][:-1]; numpy
        // rounds halves to even.
        let spacing = (train - 1) as f64 / steps as f64;
        let mut timesteps = (0..=steps)
            .map(|k| ((k as f64) * spacing).round_ties_even() as usize)
            .collect::<Vec<_>>();
        if let Some(last) = timesteps.last_mut() {
            *last = train - 1;
        }
        timesteps.reverse();
        timesteps.pop();
        // sigmas at the inference timesteps, then a final sigma of zero.
        let mut sigmas = timesteps
            .iter()
            .map(|&t| {
                let alpha = alphas_cumprod[t];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect::<Vec<f32>>();
        sigmas.push(0.0);
        let coefficients = dpm_solver_coefficients(&sigmas);
        Self {
            alphas_cumprod,
            timesteps,
            sigmas,
            coefficients,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn alpha_cumprod(&self, timestep: usize) -> f32 {
        self.alphas_cumprod
            .get(timestep.min(self.alphas_cumprod.len().saturating_sub(1)))
            .copied()
            .unwrap_or(1.0)
    }

    /// Per-step scalars and the timestep tensor (in `dtype`, for the
    /// prediction head); the solver scalars are F32.
    pub fn step_tensors(
        &self,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<VibeVoiceDiffusionStepTensors>> {
        let scalar = |value: f64| Tensor::new(value as f32, device).map_err(Error::from);
        self.timesteps
            .iter()
            .zip(&self.coefficients)
            .map(|(&timestep, coefficients)| {
                Ok(VibeVoiceDiffusionStepTensors {
                    timestep,
                    timestep_tensor: Tensor::from_vec(vec![timestep as f32], (1,), device)?
                        .to_dtype(dtype)?,
                    data_sample: scalar(coefficients.data_sample)?,
                    data_model: scalar(coefficients.data_model)?,
                    sample: scalar(coefficients.sample)?,
                    current: scalar(coefficients.current)?,
                    previous: (coefficients.previous != 0.0)
                        .then(|| scalar(coefficients.previous))
                        .transpose()?,
                })
            })
            .collect()
    }

    /// One DPM-Solver++ step from a v-prediction `model_output`, returning
    /// the next sample in the sample's dtype.
    pub fn step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        step: &VibeVoiceDiffusionStepTensors,
        state: &mut DpmSolverState,
    ) -> Result<Tensor> {
        let dtype = sample.dtype();
        let x = sample.to_dtype(DType::F32)?;
        let v = model_output.to_dtype(DType::F32)?;
        let data = x
            .broadcast_mul(&step.data_sample)?
            .broadcast_add(&v.broadcast_mul(&step.data_model)?)?;
        let mut next = x
            .broadcast_mul(&step.sample)?
            .broadcast_add(&data.broadcast_mul(&step.current)?)?;
        if let Some(previous) = &step.previous {
            let previous_data = state.previous_data.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "VibeVoice DPM-Solver second-order step has no previous prediction".into(),
                )
            })?;
            next = next.broadcast_add(&previous_data.broadcast_mul(previous)?)?;
        }
        state.previous_data = Some(data);
        next.to_dtype(dtype).map_err(Error::from)
    }
}

/// The reference `step` per index: the first step and the final step (to
/// sigma 0) are first order, every other step is the second-order
/// midpoint update with the previous data prediction.
fn dpm_solver_coefficients(sigmas: &[f32]) -> Vec<DpmSolverCoefficients> {
    let alpha_sigma = |sigma: f64| {
        let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
        (alpha, sigma * alpha)
    };
    let lambda = |(alpha, sigma): (f64, f64)| alpha.ln() - sigma.ln();
    let steps = sigmas.len() - 1;
    (0..steps)
        .map(|index| {
            let (alpha_s0, sigma_s0) = alpha_sigma(f64::from(sigmas[index]));
            let (alpha_t, sigma_t) = alpha_sigma(f64::from(sigmas[index + 1]));
            let data_sample = alpha_s0;
            let data_model = -sigma_s0;
            if sigma_t == 0.0 {
                // h = inf: x' = x0.
                return DpmSolverCoefficients {
                    data_sample,
                    data_model,
                    sample: 0.0,
                    current: alpha_t,
                    previous: 0.0,
                };
            }
            let h = lambda((alpha_t, sigma_t)) - lambda((alpha_s0, sigma_s0));
            let first = -alpha_t * ((-h).exp() - 1.0);
            let sample = sigma_t / sigma_s0;
            if index == 0 || index == steps - 1 {
                return DpmSolverCoefficients {
                    data_sample,
                    data_model,
                    sample,
                    current: first,
                    previous: 0.0,
                };
            }
            // D0 = m0, D1 = (m0 - m1) / r0; x' = s*x + first*D0 + 0.5*first*D1.
            let (alpha_s1, sigma_s1) = alpha_sigma(f64::from(sigmas[index - 1]));
            let h0 = lambda((alpha_s0, sigma_s0)) - lambda((alpha_s1, sigma_s1));
            let half = 0.5 * first * h / h0;
            DpmSolverCoefficients {
                data_sample,
                data_model,
                sample,
                current: first + half,
                previous: -half,
            }
        })
        .collect()
}

/// `alphas_cumprod` of the reference cosine schedule: betas from
/// `betas_for_alpha_bar(alpha_transform_type="cosine")` (f64, clamped to
/// 0.999, stored as F32), then the F32 running product.
fn cosine_alphas_cumprod(num_steps: usize) -> Vec<f32> {
    let alpha_bar = |t: f64| {
        ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2)
            .cos()
            .powi(2)
    };
    let mut running = 1.0f32;
    (0..num_steps)
        .map(|index| {
            let t1 = index as f64 / num_steps as f64;
            let t2 = (index + 1) as f64 / num_steps as f64;
            let beta = (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999) as f32;
            running *= 1.0 - beta;
            running
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use candle_nn::VarBuilder;

    /// VVT-4: timesteps, sigmas and a scripted v-prediction trajectory
    /// match the reference `DPMSolverMultistepScheduler(1000, "cosine",
    /// "v_prediction")` run in PyTorch (vibevoice `schedule/dpm_solver.py`)
    /// for 3, 10 and 20 steps.
    #[test]
    // Goldens are printed from the PyTorch reference at full precision.
    #[allow(clippy::excessive_precision)]
    fn dpm_solver_matches_the_reference_scheduler() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_values};

        let cases: [(usize, &[usize], [f32; 3], [f32; 8]); 3] = [
            (
                3,
                &[999, 666, 333],
                [20291.302734375, 1.750723123550415, 0.5897074341773987],
                [
                    1.189_619_3,
                    0.705_316_1,
                    -1.044_905_3,
                    -0.856_171_7,
                    -0.219_362_26,
                    0.462_077_68,
                    0.189_852_01,
                    0.192_192_7,
                ],
            ),
            (
                10,
                &[999, 899, 799, 699, 599, 500, 400, 300, 200, 100],
                [20291.302734375, 6.3645920753479, 3.103729724884033],
                [
                    0.764_613_9,
                    -0.105_061_57,
                    -0.594_315_96,
                    -0.679_934_56,
                    -0.694_521_8,
                    0.429_939_1,
                    -0.014_546_813,
                    0.399_899_36,
                ],
            ),
            (
                20,
                &[
                    999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250,
                    200, 150, 100, 50,
                ],
                [20291.302734375, 12.807271003723145, 6.3645920753479],
                [
                    0.770_184_3,
                    -0.123_555_36,
                    -0.425_158_95,
                    -0.553_862_8,
                    -0.918_555_4,
                    0.221_215_87,
                    -0.042_713_7,
                    0.423_581_2,
                ],
            ),
        ];
        let device = Device::Cpu;
        for (steps, timesteps, first_sigmas, golden) in cases {
            let scheduler = VibeVoiceDiffusionScheduler::new(1000, steps);
            assert_eq!(scheduler.timesteps(), timesteps);
            assert_eq!(scheduler.sigmas().len(), steps + 1);
            assert_eq!(*scheduler.sigmas().last().unwrap(), 0.0);
            let sigma_error = max_abs_diff(&scheduler.sigmas()[..3], &first_sigmas);
            assert!(
                sigma_error <= 1e-6 * 20291.3,
                "sigmas {:?}",
                &scheduler.sigmas()[..3]
            );
            let tensors = scheduler.step_tensors(&device, DType::F32).unwrap();
            let mut state = DpmSolverState::default();
            let mut x = Tensor::from_vec(xorshift_values(8, 1, 1.0), (2, 4), &device).unwrap();
            for (index, step) in tensors.iter().enumerate() {
                let v =
                    Tensor::from_vec(xorshift_values(8, 100 + index as u64, 1.0), (2, 4), &device)
                        .unwrap();
                x = scheduler.step(&v, &x, step, &mut state).unwrap();
            }
            let error = max_abs_diff(&host_f32(&x).unwrap(), &golden);
            assert!(error < 1e-5, "{steps} steps: max abs error {error}");
        }
    }

    #[test]
    fn cfg_batched_prediction_matches_two_independent_passes() {
        let device = Device::Cpu;
        let head = tiny_diffusion_head(&device);
        let speech = Tensor::from_vec(vec![0.1f32, -0.2], (1, 2), &device).unwrap();
        let timestep = Tensor::from_vec(vec![42.0f32], (1,), &device).unwrap();
        let positive_condition =
            Tensor::from_vec(vec![0.25f32, -0.5, 0.75, 0.1], (1, 4), &device).unwrap();
        let negative_condition =
            Tensor::from_vec(vec![-0.1f32, 0.2, -0.3, 0.4], (1, 4), &device).unwrap();
        let cfg_scale = Tensor::new(1.5f32, &device).unwrap();

        let positive = head
            .forward(&speech, &timestep, &positive_condition)
            .unwrap();
        let negative = head
            .forward(&speech, &timestep, &negative_condition)
            .unwrap();
        let expected = classifier_free_guidance(&positive, &negative, &cfg_scale).unwrap();
        let batched = head
            .forward_cfg_batched(
                &speech,
                &timestep,
                &positive_condition,
                &negative_condition,
                &cfg_scale,
            )
            .unwrap();
        let timestep_embedding = head.timestep_embedding(&timestep).unwrap();
        let positive_condition_projection = head.condition_projection(&positive_condition).unwrap();
        let negative_condition_projection = head.condition_projection(&negative_condition).unwrap();
        let precomputed_batched = head
            .forward_cfg_batched_with_precomputed(
                &speech,
                &timestep_embedding,
                &positive_condition_projection,
                &negative_condition_projection,
                &cfg_scale,
            )
            .unwrap();
        let precomputed_positive = head
            .forward_with_precomputed(&speech, &timestep_embedding, &positive_condition_projection)
            .unwrap();

        assert_tensor_close(&batched, &expected, 1e-5);
        assert_tensor_close(&precomputed_batched, &expected, 1e-5);
        assert_tensor_close(&precomputed_positive, &positive, 1e-5);
    }

    #[test]
    fn modulate_keeps_bf16_scalar_add_in_bf16() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, -2.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let shift = Tensor::from_vec(vec![0.5f32, -0.25], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let scale = Tensor::from_vec(vec![0.25f32, -0.5], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let output = modulate(&x, &shift, &scale).unwrap();

        assert_eq!(output.dtype(), DType::BF16);
        let output = output.to_dtype(DType::F32).unwrap();
        assert_tensor_close(
            &output,
            &Tensor::from_vec(vec![1.75f32, -1.25], (1, 2), &device).unwrap(),
            1e-2,
        );
    }

    #[test]
    fn rms_norm_no_affine_keeps_bf16_manual_path_in_bf16() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let unit_weight = Tensor::ones((2,), DType::BF16, &device).unwrap();

        let output = rms_norm_no_affine(&x, 0.0, &unit_weight).unwrap();

        assert_eq!(output.dtype(), DType::BF16);
        let output = output.to_dtype(DType::F32).unwrap();
        assert_tensor_close(
            &output,
            &Tensor::from_vec(vec![0.84852815f32, 1.1313709], (1, 2), &device).unwrap(),
            1e-2,
        );
    }

    /// The final-layer norm keeps its statistics in F32: an F16 row whose
    /// squares overflow F16 (|x| about 480, as the checkpoint's final-layer
    /// input) normalizes like the F32 row instead of collapsing.
    #[test]
    fn rms_norm_no_affine_does_not_overflow_in_f16() {
        use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_tensor};
        let device = Device::Cpu;
        let x = xorshift_tensor(&[2, 64], 11, 480.0, &device).unwrap();
        let f32_out =
            rms_norm_no_affine(&x, 1e-5, &Tensor::ones(64, DType::F32, &device).unwrap()).unwrap();
        let f16_out = rms_norm_no_affine(
            &x.to_dtype(DType::F16).unwrap(),
            1e-5,
            &Tensor::ones(64, DType::F16, &device).unwrap(),
        )
        .unwrap();
        assert_eq!(f16_out.dtype(), DType::F16);
        let relative = relative_l2(&host_f32(&f16_out).unwrap(), &host_f32(&f32_out).unwrap());
        assert!(relative < 1e-3, "relative L2 {relative}");
    }

    fn tiny_diffusion_head(device: &Device) -> VibeVoiceDiffusionHead {
        let hidden = 4;
        let latent = 2;
        let ffn = 4;
        let mut tensors = HashMap::new();
        insert_linear(
            &mut tensors,
            "noisy_images_proj.weight",
            hidden,
            latent,
            device,
            0.01,
        );
        insert_linear(
            &mut tensors,
            "cond_proj.weight",
            hidden,
            hidden,
            device,
            -0.015,
        );
        insert_linear(
            &mut tensors,
            "t_embedder.mlp.0.weight",
            hidden,
            256,
            device,
            0.002,
        );
        insert_linear(
            &mut tensors,
            "t_embedder.mlp.2.weight",
            hidden,
            hidden,
            device,
            0.02,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.gate_proj.weight",
            ffn,
            hidden,
            device,
            0.01,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.up_proj.weight",
            ffn,
            hidden,
            device,
            -0.0125,
        );
        insert_linear(
            &mut tensors,
            "layers.0.ffn.down_proj.weight",
            hidden,
            ffn,
            device,
            0.0175,
        );
        insert_linear(
            &mut tensors,
            "layers.0.adaLN_modulation.1.weight",
            3 * hidden,
            hidden,
            device,
            0.006,
        );
        tensors.insert(
            "layers.0.norm.weight".to_string(),
            Tensor::from_vec(vec![1.0f32; hidden], (hidden,), device).unwrap(),
        );
        insert_linear(
            &mut tensors,
            "final_layer.linear.weight",
            latent,
            hidden,
            device,
            -0.02,
        );
        insert_linear(
            &mut tensors,
            "final_layer.adaLN_modulation.1.weight",
            2 * hidden,
            hidden,
            device,
            0.0075,
        );
        let cfg = VibeVoiceDiffusionHeadConfig {
            hidden_size: hidden,
            head_layers: 1,
            head_ffn_ratio: 1.0,
            rms_norm_eps: 1e-5,
            latent_size: latent,
            speech_vae_dim: None,
            prediction_type: "v_prediction".to_string(),
            diffusion_type: "ddpm".to_string(),
            ddpm_num_steps: 1000,
            ddpm_num_inference_steps: 4,
            ddpm_beta_schedule: "cosine".to_string(),
            ddpm_batch_mul: 4,
        };
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        VibeVoiceDiffusionHead::load(cfg, vb).unwrap()
    }

    fn insert_linear(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        rows: usize,
        cols: usize,
        device: &Device,
        scale: f32,
    ) {
        let values = (0..rows * cols)
            .map(|idx| (idx as f32 + 1.0) * scale)
            .collect::<Vec<_>>();
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(values, (rows, cols), device).unwrap(),
        );
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, epsilon: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (*actual - *expected).abs() <= epsilon,
                "expected {actual} to be within {epsilon} of {expected}"
            );
        }
    }
}

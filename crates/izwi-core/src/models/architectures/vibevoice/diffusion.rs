//! VibeVoice diffusion prediction head and scheduler helpers.

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, ops};

use crate::error::{Error, Result};
use crate::kernels::try_fused_silu_mul;
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
        let mut x = self.noisy_images_proj.forward(noisy_latents)?;
        let mut t = self.t_embedder.forward(timesteps)?;
        let condition = self.cond_proj.forward(condition)?;
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
        let latents = Tensor::cat(&[noisy_latents.clone(), noisy_latents.clone()], 0)?;
        let timesteps = Tensor::cat(&[timesteps.clone(), timesteps.clone()], 0)?;
        let conditions = Tensor::cat(&[condition.clone(), negative_condition.clone()], 0)?;
        let output = self.forward(&latents, &timesteps, &conditions)?;
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
        Ok(Self {
            linear: mlx::load_linear_no_bias(hidden_size, output_size, vb.pp("linear"))?,
            ada_ln: mlx::load_linear_no_bias(
                cond_size,
                2 * hidden_size,
                vb.pp("adaLN_modulation.1"),
            )?,
            hidden_size,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln.forward(&ops::silu(c)?)?;
        let shift = modulation.narrow(D::Minus1, 0, self.hidden_size)?;
        let scale = modulation.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        let normed = rms_norm_no_affine(x, self.eps)?;
        let modulated = modulate(&normed, &shift, &scale)?;
        self.linear.forward(&modulated).map_err(Error::from)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scaled = x.broadcast_mul(&scale.broadcast_add(&Tensor::new(1f32, x.device())?)?)?;
    scaled.broadcast_add(shift).map_err(Error::from)
}

fn rms_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let squared = x.sqr()?;
    let mean = squared.mean_keepdim(D::Minus1)?;
    let eps = Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?;
    let denom = mean.broadcast_add(&eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&denom).map_err(Error::from)
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

pub struct VibeVoiceDiffusionScheduler {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<usize>,
}

pub struct VibeVoiceDiffusionStepTensors {
    pub timestep: usize,
    pub prev_timestep: Option<usize>,
    pub timestep_tensor: Tensor,
    sqrt_alpha: Tensor,
    sqrt_beta: Tensor,
    sqrt_prev_alpha: Tensor,
    sqrt_prev_beta: Tensor,
}

impl VibeVoiceDiffusionScheduler {
    pub fn new(num_train_timesteps: usize, num_inference_steps: usize) -> Self {
        let betas = cosine_beta_schedule(num_train_timesteps.max(1));
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        let mut running = 1.0f32;
        for beta in betas {
            running *= 1.0 - beta;
            alphas_cumprod.push(running.clamp(1e-8, 1.0));
        }
        let inference_steps = num_inference_steps.max(1);
        let last = num_train_timesteps.saturating_sub(1);
        let timesteps = (0..inference_steps)
            .map(|idx| {
                let frac = if inference_steps == 1 {
                    0.0
                } else {
                    idx as f32 / (inference_steps - 1) as f32
                };
                ((last as f32) * (1.0 - frac)).round() as usize
            })
            .collect();
        Self {
            alphas_cumprod,
            timesteps,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn alpha_cumprod(&self, timestep: usize) -> f32 {
        self.alphas_cumprod
            .get(timestep.min(self.alphas_cumprod.len().saturating_sub(1)))
            .copied()
            .unwrap_or(1.0)
    }

    pub fn step_tensors(
        &self,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<VibeVoiceDiffusionStepTensors>> {
        self.timesteps
            .iter()
            .enumerate()
            .map(|(idx, &timestep)| {
                let prev_timestep = self.timesteps.get(idx + 1).copied();
                let coeffs = self.step_coefficients(timestep, prev_timestep);
                Ok(VibeVoiceDiffusionStepTensors {
                    timestep,
                    prev_timestep,
                    timestep_tensor: Tensor::from_vec(vec![timestep as f32], (1,), device)?
                        .to_dtype(dtype)?,
                    sqrt_alpha: Tensor::new(coeffs.sqrt_alpha, device)?.to_dtype(dtype)?,
                    sqrt_beta: Tensor::new(coeffs.sqrt_beta, device)?.to_dtype(dtype)?,
                    sqrt_prev_alpha: Tensor::new(coeffs.sqrt_prev_alpha, device)?
                        .to_dtype(dtype)?,
                    sqrt_prev_beta: Tensor::new(coeffs.sqrt_prev_beta, device)?.to_dtype(dtype)?,
                })
            })
            .collect()
    }

    pub fn step_v_prediction(
        &self,
        model_output: &Tensor,
        timestep: usize,
        prev_timestep: Option<usize>,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let coeffs = self.step_coefficients(timestep, prev_timestep);
        let sqrt_alpha =
            Tensor::new(coeffs.sqrt_alpha, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_beta = Tensor::new(coeffs.sqrt_beta, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_prev_alpha =
            Tensor::new(coeffs.sqrt_prev_alpha, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_prev_beta =
            Tensor::new(coeffs.sqrt_prev_beta, sample.device())?.to_dtype(sample.dtype())?;
        step_v_prediction_with_tensors(
            model_output,
            sample,
            &sqrt_alpha,
            &sqrt_beta,
            &sqrt_prev_alpha,
            &sqrt_prev_beta,
        )
    }

    pub fn step_v_prediction_with_tensors(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        step: &VibeVoiceDiffusionStepTensors,
    ) -> Result<Tensor> {
        step_v_prediction_with_tensors(
            model_output,
            sample,
            &step.sqrt_alpha,
            &step.sqrt_beta,
            &step.sqrt_prev_alpha,
            &step.sqrt_prev_beta,
        )
    }

    fn step_coefficients(
        &self,
        timestep: usize,
        prev_timestep: Option<usize>,
    ) -> DiffusionStepCoefficients {
        let alpha = self.alpha_cumprod(timestep);
        let prev_alpha = prev_timestep.map(|t| self.alpha_cumprod(t)).unwrap_or(1.0);
        let beta = (1.0 - alpha).max(0.0);
        let prev_beta = (1.0 - prev_alpha).max(0.0);
        DiffusionStepCoefficients {
            sqrt_alpha: alpha.sqrt(),
            sqrt_beta: beta.sqrt(),
            sqrt_prev_alpha: prev_alpha.sqrt(),
            sqrt_prev_beta: prev_beta.sqrt(),
        }
    }
}

struct DiffusionStepCoefficients {
    sqrt_alpha: f32,
    sqrt_beta: f32,
    sqrt_prev_alpha: f32,
    sqrt_prev_beta: f32,
}

fn step_v_prediction_with_tensors(
    model_output: &Tensor,
    sample: &Tensor,
    sqrt_alpha: &Tensor,
    sqrt_beta: &Tensor,
    sqrt_prev_alpha: &Tensor,
    sqrt_prev_beta: &Tensor,
) -> Result<Tensor> {
    let pred_original = sample
        .broadcast_mul(sqrt_alpha)?
        .broadcast_sub(&model_output.broadcast_mul(sqrt_beta)?)?;
    let pred_epsilon = model_output
        .broadcast_mul(sqrt_alpha)?
        .broadcast_add(&sample.broadcast_mul(sqrt_beta)?)?;
    pred_original
        .broadcast_mul(sqrt_prev_alpha)?
        .broadcast_add(&pred_epsilon.broadcast_mul(sqrt_prev_beta)?)
        .map_err(Error::from)
}

fn cosine_beta_schedule(num_steps: usize) -> Vec<f32> {
    let s = 0.008f32;
    let mut alphas_cumprod = Vec::with_capacity(num_steps + 1);
    for idx in 0..=num_steps {
        let t = idx as f32 / num_steps as f32;
        let value = (((t + s) / (1.0 + s)) * std::f32::consts::FRAC_PI_2)
            .cos()
            .powi(2);
        alphas_cumprod.push(value);
    }
    let first = alphas_cumprod[0];
    for value in &mut alphas_cumprod {
        *value /= first;
    }
    (0..num_steps)
        .map(|idx| {
            let beta = 1.0 - (alphas_cumprod[idx + 1] / alphas_cumprod[idx]);
            beta.clamp(1e-8, 0.999)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use candle_nn::VarBuilder;

    #[test]
    fn scheduler_builds_descending_inference_timesteps() {
        let scheduler = VibeVoiceDiffusionScheduler::new(1000, 20);
        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.timesteps()[0], 999);
        assert_eq!(*scheduler.timesteps().last().unwrap(), 0);
        assert!(scheduler.alpha_cumprod(999) > 0.0);
    }

    #[test]
    fn scheduler_step_tensors_match_scalar_step_path() {
        let device = Device::Cpu;
        let scheduler = VibeVoiceDiffusionScheduler::new(1000, 4);
        let sample = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, -0.4], (1, 4), &device).unwrap();
        let model_output =
            Tensor::from_vec(vec![0.05f32, 0.1, -0.15, -0.2], (1, 4), &device).unwrap();
        let steps = scheduler
            .step_tensors(&device, DType::F32)
            .expect("step tensors");

        let scalar = scheduler
            .step_v_prediction(
                &model_output,
                steps[0].timestep,
                steps[0].prev_timestep,
                &sample,
            )
            .expect("scalar step");
        let cached = scheduler
            .step_v_prediction_with_tensors(&model_output, &sample, &steps[0])
            .expect("cached step");
        let scalar = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cached = cached.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (lhs, rhs) in scalar.iter().zip(cached.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
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

        assert_tensor_close(&batched, &expected, 1e-5);
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

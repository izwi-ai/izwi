//! VibeVoice diffusion prediction head and scheduler helpers.

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, ops};

use crate::error::{Error, Result};
use crate::models::architectures::vibevoice::config::VibeVoiceDiffusionHeadConfig;
use crate::models::shared::weights::mlx;

pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn load(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let frequency_embedding_size = 256;
        let linear_1 =
            mlx::load_linear_no_bias(frequency_embedding_size, hidden_size, vb.pp("mlp.0"))?;
        let linear_2 = mlx::load_linear_no_bias(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            frequency_embedding_size,
        })
    }

    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let emb = timestep_embedding(
            timesteps,
            self.frequency_embedding_size,
            timesteps.device(),
            timesteps.dtype(),
        )?;
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
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
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

fn timestep_embedding(t: &Tensor, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let t = t.flatten_all()?.to_dtype(DType::F32)?;
    let half = dim / 2;
    let mut freqs = Vec::with_capacity(half);
    for idx in 0..half {
        freqs.push((-((10_000f32).ln()) * (idx as f32) / (half as f32)).exp());
    }
    let freqs = Tensor::from_vec(freqs, (half,), device)?;
    let args = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let mut emb = Tensor::cat(&[&cos, &sin], 1)?;
    if dim % 2 == 1 {
        let zeros = Tensor::zeros((emb.dim(0)?, 1), DType::F32, device)?;
        emb = Tensor::cat(&[&emb, &zeros], 1)?;
    }
    emb.to_dtype(dtype).map_err(Error::from)
}

pub struct VibeVoiceDiffusionScheduler {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<usize>,
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

    pub fn step_v_prediction(
        &self,
        model_output: &Tensor,
        timestep: usize,
        prev_timestep: Option<usize>,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let alpha = self.alpha_cumprod(timestep);
        let prev_alpha = prev_timestep.map(|t| self.alpha_cumprod(t)).unwrap_or(1.0);
        let beta = (1.0 - alpha).max(0.0);
        let prev_beta = (1.0 - prev_alpha).max(0.0);
        let sqrt_alpha = alpha.sqrt();
        let sqrt_beta = beta.sqrt();
        let sqrt_prev_alpha = prev_alpha.sqrt();
        let sqrt_prev_beta = prev_beta.sqrt();

        let sqrt_alpha = Tensor::new(sqrt_alpha, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_beta = Tensor::new(sqrt_beta, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_prev_alpha =
            Tensor::new(sqrt_prev_alpha, sample.device())?.to_dtype(sample.dtype())?;
        let sqrt_prev_beta =
            Tensor::new(sqrt_prev_beta, sample.device())?.to_dtype(sample.dtype())?;

        let pred_original = sample
            .broadcast_mul(&sqrt_alpha)?
            .broadcast_sub(&model_output.broadcast_mul(&sqrt_beta)?)?;
        let pred_epsilon = model_output
            .broadcast_mul(&sqrt_alpha)?
            .broadcast_add(&sample.broadcast_mul(&sqrt_beta)?)?;
        pred_original
            .broadcast_mul(&sqrt_prev_alpha)?
            .broadcast_add(&pred_epsilon.broadcast_mul(&sqrt_prev_beta)?)
            .map_err(Error::from)
    }
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

    #[test]
    fn scheduler_builds_descending_inference_timesteps() {
        let scheduler = VibeVoiceDiffusionScheduler::new(1000, 20);
        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.timesteps()[0], 999);
        assert_eq!(*scheduler.timesteps().last().unwrap(), 0);
        assert!(scheduler.alpha_cumprod(999) > 0.0);
    }
}

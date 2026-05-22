//! Time embedding for Voxtral Realtime delay conditioning.
//!
//! Sinusoidal embedding for encoding time/delay tokens.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::Module;

/// Sinusoidal Time Embedding for encoding delay tokens
pub struct TimeEmbedding {
    dim: usize,
    theta: f32,
    inv_freq: Tensor,
}

impl TimeEmbedding {
    pub fn new(dim: usize, theta: f32, device: &candle_core::Device) -> Result<Self> {
        let half_dim = dim / 2;
        let log_theta = theta.ln();
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (-log_theta * i as f32 / half_dim as f32).exp())
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        Ok(Self {
            dim,
            theta,
            inv_freq,
        })
    }

    /// Forward pass
    /// t: (B,) -> (B, dim) or (B, T) -> (B, T, dim)
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let dtype = t.dtype();
        // Add dimension: (B,) -> (B, 1) or (B, T) -> (B, T, 1)
        let t = t.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;

        // Broadcast: (B, 1) x (dim/2,) -> (B, dim/2)
        let emb = t.broadcast_mul(&self.inv_freq)?;

        // cat([cos(emb), sin(emb)], dim=-1)
        let cos_emb = emb.cos()?;
        let sin_emb = emb.sin()?;
        let emb = Tensor::cat(&[&cos_emb, &sin_emb], D::Minus1)?;

        emb.to_dtype(dtype)
    }
}

/// Audio-Language Adapter for projecting audio features to LM dimension
pub struct AudioLanguageAdapter {
    w_in: candle_nn::Linear,
    w_out: candle_nn::Linear,
}

impl AudioLanguageAdapter {
    pub fn load(hidden_size: usize, dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Voxtral uses sequential indices: 0 for w_in, 2 for w_out
        let w_in = candle_nn::linear_no_bias(hidden_size, dim, vb.pp("0"))?;
        let w_out = candle_nn::linear_no_bias(dim, dim, vb.pp("2"))?;

        Ok(Self { w_in, w_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_in.forward(x)?;
        let x = gelu(&x)?;
        self.w_out.forward(&x)
    }
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let x3 = x_f32.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?;
    let inner = (&x_f32 + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?;
    let out = x_f32.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    out.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn time_embedding_preserves_selected_dtype() {
        let device = Device::Cpu;
        let embedding = TimeEmbedding::new(8, 10000.0, &device).unwrap();
        let t = Tensor::from_vec(vec![4.0f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out = embedding.forward(&t).unwrap();

        assert_eq!(out.dtype(), DType::F16);
        assert_eq!(out.dims(), &[1, 8]);
    }

    #[test]
    fn time_embedding_uses_vllm_frequency_schedule() {
        let device = Device::Cpu;
        let embedding = TimeEmbedding::new(8, 10000.0, &device).unwrap();
        let t = Tensor::from_vec(vec![1.0f32], (1,), &device).unwrap();

        let out = embedding.forward(&t).unwrap().to_vec2::<f32>().unwrap();
        let row = &out[0];

        assert!((row[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((row[1] - 0.1f32.cos()).abs() < 1e-6);
        assert!((row[2] - 0.01f32.cos()).abs() < 1e-6);
        assert!((row[3] - 0.001f32.cos()).abs() < 1e-6);
        assert!((row[4] - 1.0f32.sin()).abs() < 1e-6);
        assert!((row[5] - 0.1f32.sin()).abs() < 1e-6);
        assert!((row[6] - 0.01f32.sin()).abs() < 1e-6);
        assert!((row[7] - 0.001f32.sin()).abs() < 1e-6);
    }
}

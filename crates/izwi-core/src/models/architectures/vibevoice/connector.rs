//! VibeVoice speech connector layers.

use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::shared::weights::mlx;

pub struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    pub fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = mlx::load_linear(input_dim, output_dim, vb.pp("fc1"))?;
        let norm = candle_nn::rms_norm(output_dim, 1e-6, vb.pp("norm"))?;
        let fc2 = mlx::load_linear(output_dim, output_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(features)?;
        let x = self.norm.forward(&x)?;
        self.fc2.forward(&x).map_err(Error::from)
    }
}

//! Gemma 4's RMSNorm: `x * (mean(x²) + eps)^-0.5 * weight`, computed in F32
//! and cast back to the input's dtype. The gain multiplies as stored (it is
//! not `1 + weight`, the Gemma 3 form), and `v_norm` and the router norm
//! have no gain at all.

use candle_core::{DType, Device, Tensor, D};

use super::super::checkpoint::SovCheckpoint;
use crate::error::Result;

pub struct RmsNorm {
    /// F32 `[dim]`, or none for a norm without scale.
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub fn load(checkpoint: &SovCheckpoint, name: &str, device: &Device, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: Some(checkpoint.tensor(name, device, DType::F32)?),
            eps,
        })
    }

    /// Bytes of the gain on the device.
    pub fn resident_bytes(&self) -> usize {
        self.weight
            .as_ref()
            .map_or(0, |weight| weight.elem_count() * 4)
    }

    pub fn without_scale(eps: f64) -> Self {
        Self { weight: None, eps }
    }

    /// The normalized input in F32.
    pub fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean_square = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(mean_square + self.eps)?.sqrt()?)?;
        Ok(match &self.weight {
            Some(weight) => normed.broadcast_mul(weight)?,
            None => normed,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_f32(x)?.to_dtype(x.dtype())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gain_multiplies_as_stored() {
        let x = Tensor::new(&[[3f32, 4.0]], &Device::Cpu).unwrap();
        let rms = (12.5f32 + 1e-6).sqrt();
        let plain = RmsNorm::without_scale(1e-6).forward(&x).unwrap();
        assert_eq!(
            plain.to_vec2::<f32>().unwrap(),
            vec![vec![3.0 / rms, 4.0 / rms]]
        );
        let scaled = RmsNorm {
            weight: Some(Tensor::new(&[2f32, 0.0], &Device::Cpu).unwrap()),
            eps: 1e-6,
        }
        .forward(&x)
        .unwrap();
        assert_eq!(
            scaled.to_vec2::<f32>().unwrap(),
            vec![vec![2.0 * 3.0 / rms, 0.0]]
        );
    }
}

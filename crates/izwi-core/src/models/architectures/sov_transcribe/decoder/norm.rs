//! Gemma 4's RMSNorm: `x * (mean(x²) + eps)^-0.5 * weight`, computed in F32
//! and cast back to the input's dtype. The gain multiplies as stored (it is
//! not `1 + weight`, the Gemma 3 form), and `v_norm` and the router norm
//! have no gain at all.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::SovCheckpoint;
use crate::error::Result;
use crate::kernels::residual_norm;

/// One launch per norm on every backend (`kernels::residual_norm`): the
/// row's squares summed in F32 in a fixed order, so a row's bits never depend
/// on the rows beside it.
pub struct RmsNorm {
    /// F32 `[dim]`; ones for a norm without scale.
    weight: Tensor,
    eps: f32,
    scaled: bool,
}

impl RmsNorm {
    pub fn load(checkpoint: &SovCheckpoint, name: &str, device: &Device, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: checkpoint.tensor(name, device, DType::F32)?,
            eps: eps as f32,
            scaled: true,
        })
    }

    /// A norm without gain over rows of `dim` values.
    pub fn without_scale(eps: f64, dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(dim, DType::F32, device)?,
            eps: eps as f32,
            scaled: false,
        })
    }

    /// Bytes of the gain on the device (a scale-less norm's ones are not
    /// weights).
    pub fn resident_bytes(&self) -> usize {
        if self.scaled {
            self.weight.elem_count() * 4
        } else {
            0
        }
    }

    /// The normalized input in F32.
    pub fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        Ok(residual_norm::rms_norm(
            x,
            &self.weight,
            self.eps,
            DType::F32,
        )?)
    }

    /// The normalized input in its own dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(residual_norm::rms_norm(
            x,
            &self.weight,
            self.eps,
            x.dtype(),
        )?)
    }

    /// `residual + delta` in the residual's dtype, and this norm of the sum,
    /// in one launch.
    pub fn add_forward(&self, residual: &Tensor, delta: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok(residual_norm::add_rms_norm(
            residual,
            delta,
            &self.weight,
            self.eps,
            residual.dtype(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: Vec<Vec<f32>>, expected: [f32; 2]) {
        for (a, e) in actual[0].iter().zip(expected) {
            assert!((a - e).abs() <= 1e-6 * e.abs().max(1.0), "{a} vs {e}");
        }
    }

    #[test]
    fn the_gain_multiplies_as_stored() {
        let x = Tensor::new(&[[3f32, 4.0]], &Device::Cpu).unwrap();
        let rms = (12.5f32 + 1e-6).sqrt();
        let plain = RmsNorm::without_scale(1e-6, 2, &Device::Cpu)
            .unwrap()
            .forward(&x)
            .unwrap();
        close(plain.to_vec2::<f32>().unwrap(), [3.0 / rms, 4.0 / rms]);
        let scaled = RmsNorm {
            weight: Tensor::new(&[2f32, 0.0], &Device::Cpu).unwrap(),
            eps: 1e-6,
            scaled: true,
        };
        close(
            scaled.forward(&x).unwrap().to_vec2::<f32>().unwrap(),
            [2.0 * 3.0 / rms, 0.0],
        );
        // The fused add: the sum, and the norm of the sum.
        let delta = Tensor::new(&[[1f32, -1.0]], &Device::Cpu).unwrap();
        let (sum, normed) = scaled.add_forward(&x, &delta).unwrap();
        assert_eq!(sum.to_vec2::<f32>().unwrap(), vec![vec![4.0, 3.0]]);
        close(
            normed.to_vec2::<f32>().unwrap(),
            [2.0 * 4.0 / (12.5f32 + 1e-6).sqrt(), 0.0],
        );
    }
}

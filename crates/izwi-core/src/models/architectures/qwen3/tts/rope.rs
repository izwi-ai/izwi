use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};

pub(super) fn build_rope_inv_freq(head_dim: usize, rope_theta: f64) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }
    inv_freq
}

pub(super) fn build_rope_window(
    seq_len: usize,
    start_pos: usize,
    inv_freq: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freq.len();
    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for &inv in inv_freq.iter() {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

pub(super) fn build_rope_window_full(
    seq_len: usize,
    start_pos: usize,
    inv_freq: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = build_rope_window(seq_len, start_pos, inv_freq, device, dtype)?;
    duplicate_rope_window(cos, sin)
}

pub(super) fn duplicate_rope_window(cos: Tensor, sin: Tensor) -> Result<(Tensor, Tensor)> {
    let cos = Tensor::cat(&[cos.clone(), cos], 1)?;
    let sin = Tensor::cat(&[sin.clone(), sin], 1)?;
    Ok((cos, sin))
}

pub(super) fn qwen_rotate_half(x: &Tensor, half_dim: usize) -> Result<Tensor> {
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let neg_x2 = if x.device().is_cuda() {
        x2.neg()?
    } else {
        let minus_one = Tensor::from_vec(vec![-1.0f32], (1,), x.device())?.to_dtype(x.dtype())?;
        x2.broadcast_mul(&minus_one)?
    };
    Tensor::cat(&[neg_x2, x1], 3).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_rope_window_matches_full_prefix_slice() {
        let device = Device::Cpu;
        let inv_freq = build_rope_inv_freq(4, 10_000.0);
        let (direct_cos, direct_sin) =
            build_rope_window_full(2, 3, &inv_freq, &device, DType::F32).unwrap();
        let (prefix_cos, prefix_sin) =
            build_rope_window_full(5, 0, &inv_freq, &device, DType::F32).unwrap();
        let cached_cos = prefix_cos.narrow(0, 3, 2).unwrap();
        let cached_sin = prefix_sin.narrow(0, 3, 2).unwrap();

        assert_eq!(cached_cos.dim(1).unwrap(), 4);
        assert_eq!(
            cached_cos.to_vec2::<f32>().unwrap(),
            direct_cos.to_vec2::<f32>().unwrap()
        );
        assert_eq!(
            cached_sin.to_vec2::<f32>().unwrap(),
            direct_sin.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn qwen_rotate_half_matches_reference_layout() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[[1.0f32, 2.0, 3.0, 4.0]]]], &device).unwrap();
        let rotated = qwen_rotate_half(&x, 2).unwrap();

        assert_eq!(
            rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![-3.0, -4.0, 1.0, 2.0]
        );
    }
}

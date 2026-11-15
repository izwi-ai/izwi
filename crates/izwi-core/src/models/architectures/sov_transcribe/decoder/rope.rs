//! Rotary position embeddings, in the two forms Gemma 4 uses.
//!
//! Sliding layers rotate every pair of a 256-dim head with theta 1e4. Full
//! layers use the "proportional" form over a 512-dim head: only the first
//! `partial_rotary_factor * head_dim / 2` (64) pairs rotate, with theta 1e6
//! and the exponent taken over the whole head (`2j / 512`), and the other
//! 192 pairs pass through. Pair `j` is channels `j` and `j + head_dim / 2`
//! (`rotate_half`).

use candle_core::{DType, Device, Tensor};

use super::super::config::SovAttentionGeometry;
use crate::error::Result;

pub struct Rope {
    half: usize,
    /// One frequency per pair; zero for a pair that does not rotate.
    inv_freq: Vec<f32>,
}

impl Rope {
    pub fn new(geometry: &SovAttentionGeometry) -> Self {
        let half = geometry.head_dim / 2;
        let inv_freq = (0..half)
            .map(|pair| {
                if pair < geometry.rotary_pairs {
                    let exponent = (2 * pair) as f64 / geometry.rope_dim as f64;
                    (1.0 / geometry.rope_theta.powf(exponent)) as f32
                } else {
                    0.0
                }
            })
            .collect();
        Self { half, inv_freq }
    }

    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// `(cos, sin)` `[positions, 1, head_dim / 2]` in `dtype`, computed on
    /// the host (the reference for [`RopeTables`]).
    pub fn tables(
        &self,
        positions: &[usize],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let mut cos = Vec::with_capacity(positions.len() * self.half);
        let mut sin = Vec::with_capacity(positions.len() * self.half);
        for &position in positions {
            for &frequency in &self.inv_freq {
                let angle = position as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let shape = (positions.len(), 1, self.half);
        Ok((
            Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?,
            Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?,
        ))
    }

    /// Every position's `(cos, sin)` up to `positions`, resident on `device`.
    pub fn resident(&self, positions: usize, device: &Device, dtype: DType) -> Result<RopeTables> {
        let all = (0..positions).collect::<Vec<_>>();
        let (cos, sin) = self.tables(&all, device, dtype)?;
        Ok(RopeTables {
            half: self.half,
            cos: cos.squeeze(1)?,
            sin: sin.squeeze(1)?,
        })
    }
}

/// Rotates `x [positions, heads, head_dim]` by `(cos, sin)`
/// `[positions, 1, head_dim / 2]` (`rotate_half`).
pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = cos.dim(2)?;
    let first = x.narrow(2, 0, half)?;
    let second = x.narrow(2, half, half)?;
    let rotated_first = (first.broadcast_mul(cos)? - second.broadcast_mul(sin)?)?;
    let rotated_second = (second.broadcast_mul(cos)? + first.broadcast_mul(sin)?)?;
    Ok(Tensor::cat(&[&rotated_first, &rotated_second], 2)?)
}

/// One geometry's `(cos, sin)` for every position of the context, resident
/// on the device: a step gathers its rows by a positions tensor instead of
/// building and uploading tables per layer.
pub struct RopeTables {
    half: usize,
    /// `[positions, head_dim / 2]`.
    cos: Tensor,
    sin: Tensor,
}

impl RopeTables {
    /// `(cos, sin)` `[n, 1, head_dim / 2]` of `positions` (U32 `[n]`).
    pub fn gather(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = positions.dim(0)?;
        Ok((
            self.cos
                .index_select(positions, 0)?
                .reshape((n, 1, self.half))?,
            self.sin
                .index_select(positions, 0)?
                .reshape((n, 1, self.half))?,
        ))
    }

    pub fn resident_bytes(&self) -> usize {
        2 * self.cos.elem_count() * self.cos.dtype().size_in_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_tables_match_the_reference_formulas() {
        let config = super::super::super::config::tests::published();
        let sliding = Rope::new(&config.decoder.sliding);
        assert_eq!(sliding.inv_freq().len(), 128);
        assert_eq!(sliding.inv_freq()[0], 1.0);
        let expected = (1.0f64 / 10_000f64.powf(2.0 / 256.0)) as f32;
        assert_eq!(sliding.inv_freq()[1], expected);
        assert!(sliding.inv_freq().iter().all(|f| *f > 0.0));

        let full = Rope::new(&config.decoder.full);
        assert_eq!(full.inv_freq().len(), 256);
        let expected = (1.0f64 / 1_000_000f64.powf(2.0 / 512.0)) as f32;
        assert_eq!(full.inv_freq()[1], expected);
        assert!(full.inv_freq()[..64].iter().all(|f| *f > 0.0));
        assert!(full.inv_freq()[64..].iter().all(|f| *f == 0.0));
    }

    #[test]
    fn unrotated_pairs_pass_through() {
        let geometry = SovAttentionGeometry {
            heads: 1,
            kv_heads: 1,
            head_dim: 8,
            own_values: false,
            rope_theta: 100.0,
            rotary_pairs: 1,
            rope_dim: 8,
            window: None,
        };
        let rope = Rope::new(&geometry);
        let values = (1..=8).map(|v| v as f32).collect::<Vec<_>>();
        let x = Tensor::from_vec(values.clone(), (1, 1, 8), &Device::Cpu).unwrap();
        let (cos, sin) = rope.tables(&[3], &Device::Cpu, DType::F32).unwrap();
        // The resident tables gather the same rows.
        let resident = rope.resident(5, &Device::Cpu, DType::F32).unwrap();
        let (gathered, _) = resident
            .gather(&Tensor::new(&[3u32], &Device::Cpu).unwrap())
            .unwrap();
        assert_eq!(
            gathered.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            cos.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let out = apply(&x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (c, s) = (3f32.cos(), 3f32.sin());
        // Pair 0 is channels 0 and 4.
        assert!((out[0] - (values[0] * c - values[4] * s)).abs() < 1e-6);
        assert!((out[4] - (values[4] * c + values[0] * s)).abs() < 1e-6);
        for channel in [1, 2, 3, 5, 6, 7] {
            assert_eq!(out[channel], values[channel]);
        }
    }
}

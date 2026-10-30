//! The published codec geometry the DAC is built from.
//!
//! The small invariants the port depends on, from the `modded_dac` config
//! that ships with `codec.pth`, kept cheap to test without running the codec.

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3DacContract {
    pub sample_rate: u32,
    pub encoder_rates: &'static [usize],
    pub decoder_rates: &'static [usize],
    pub semantic_codebooks: usize,
    pub residual_codebooks: usize,
    pub semantic_codebook_size: usize,
    pub residual_codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
}

impl FishDrama3DacContract {
    pub const CURRENT: Self = Self {
        sample_rate: 44_100,
        encoder_rates: &[2, 4, 8, 8],
        decoder_rates: &[8, 8, 4, 2],
        semantic_codebooks: 1,
        residual_codebooks: 9,
        semantic_codebook_size: 4096,
        residual_codebook_size: 1024,
        codebook_dim: 8,
        latent_dim: 1024,
    };

    pub fn total_codebooks(&self) -> usize {
        self.semantic_codebooks + self.residual_codebooks
    }

    pub fn hop_length(&self) -> Result<usize> {
        checked_product(self.encoder_rates, "Fish Audio Drama 3 DAC encoder rates")
    }

    pub fn frame_length(&self) -> Result<usize> {
        self.hop_length()?.checked_mul(4).ok_or_else(|| {
            Error::ConfigError("Fish Audio Drama 3 DAC frame length overflowed".to_string())
        })
    }

    pub fn frame_rate_hz(&self) -> Result<f32> {
        Ok(self.sample_rate as f32 / self.frame_length()? as f32)
    }
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| Error::ConfigError(format!("{label} product overflowed")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dac_contract_matches_the_published_codec_config() {
        let contract = FishDrama3DacContract::CURRENT;
        assert_eq!(contract.sample_rate, 44_100);
        assert_eq!(contract.total_codebooks(), 10);
        assert_eq!(contract.hop_length().unwrap(), 512);
        assert_eq!(contract.frame_length().unwrap(), 2048);
        let frame_rate = contract.frame_rate_hz().unwrap();
        assert!((frame_rate - 21.533).abs() < 0.01);
        assert_eq!(contract.semantic_codebook_size, 4096);
        assert_eq!(contract.residual_codebook_size, 1024);
    }
}

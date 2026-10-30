//! FP8 semantics and the `vllm-fp8` parity mode (D5).
//!
//! The default, [`Fp8Mode::W8A16`], computes with the checkpoint's weights
//! exactly (`decode(byte) * weight_scale`), unquantized activations and a
//! dense KV cache: closer to the BF16 model, and the same arithmetic as
//! vLLM's Marlin path on SM80/86 GPUs.
//!
//! vLLM on SM89+ runs W8A8 instead. [`Fp8Mode::VllmFp8`] emulates it, to
//! tell an izwi bug from the W8A8/W8A16 difference when a transcript
//! disagrees with such a run:
//! * every FP8 linear quantizes its input to E4M3 with its static
//!   `input_scale` (clamp ±448, round to nearest even) first;
//! * attention q/k/v share the largest of their weight scales, each shard
//!   requantized to it, and the largest input scale (vLLM's fused QKV);
//! * K and V round-trip through E4M3 with `k_scale` / `v_scale` before the
//!   cache (vLLM's FP8 KV cache).
//!
//! It is a diagnostic mode: the activation quantization runs on the host.

use crate::error::{Error, Result};
use crate::kernels::cuda::fp8::{decode_e4m3fn, encode_e4m3fn};

/// `IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE`: `w8a16` (default) or `vllm-fp8`.
pub const FP8_MODE_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fp8Mode {
    #[default]
    W8A16,
    VllmFp8,
}

impl Fp8Mode {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "w8a16" => Ok(Self::W8A16),
            "vllm-fp8" | "vllm_fp8" | "w8a8" => Ok(Self::VllmFp8),
            other => Err(Error::InvalidInput(format!(
                "{FP8_MODE_ENV}={other:?}: expected w8a16 or vllm-fp8"
            ))),
        }
    }

    /// The mode the environment selects.
    pub fn from_env() -> Result<Self> {
        match std::env::var(FP8_MODE_ENV) {
            Ok(value) => Self::parse(&value),
            Err(_) => Ok(Self::W8A16),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::W8A16 => "w8a16",
            Self::VllmFp8 => "vllm-fp8",
        }
    }
}

/// One shard of vLLM's fused QKV: its bytes and scales.
#[derive(Debug, Clone)]
pub struct Fp8Shard {
    pub bytes: Vec<u8>,
    pub weight_scale: f32,
    pub input_scale: f32,
}

/// vLLM's `requantize_with_max_scale` over fused shards: each shard's bytes
/// requantized to the largest weight scale, and the largest input scale for
/// all.
pub fn requantize_to_shared_scale(shards: &[Fp8Shard]) -> Vec<Fp8Shard> {
    let weight_scale = shards.iter().fold(0f32, |m, s| m.max(s.weight_scale));
    let input_scale = shards.iter().fold(0f32, |m, s| m.max(s.input_scale));
    shards
        .iter()
        .map(|shard| Fp8Shard {
            bytes: shard
                .bytes
                .iter()
                .map(|&b| encode_e4m3fn(decode_e4m3fn(b) * shard.weight_scale / weight_scale))
                .collect(),
            weight_scale,
            input_scale,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse_and_default_to_w8a16() {
        assert_eq!(Fp8Mode::parse("").unwrap(), Fp8Mode::W8A16);
        assert_eq!(Fp8Mode::parse("W8A16").unwrap(), Fp8Mode::W8A16);
        assert_eq!(Fp8Mode::parse("vllm-fp8").unwrap(), Fp8Mode::VllmFp8);
        assert!(Fp8Mode::parse("int4").is_err());
    }

    #[test]
    fn shared_scale_requantization_coarsens_the_smaller_shard() {
        // A value exact on the fine grid lands on the coarse one.
        let fine = Fp8Shard {
            bytes: vec![encode_e4m3fn(1.125)],
            weight_scale: 0.001,
            input_scale: 0.02,
        };
        let coarse = Fp8Shard {
            bytes: vec![encode_e4m3fn(3.0)],
            weight_scale: 0.008,
            input_scale: 0.05,
        };
        let out = requantize_to_shared_scale(&[fine, coarse]);
        assert_eq!(out[0].weight_scale, 0.008);
        assert_eq!(out[0].input_scale, 0.05);
        // 1.125 * 0.001 / 0.008 = 0.140625 → E4M3 0.140625 exactly (2^-3 * 1.125).
        assert_eq!(decode_e4m3fn(out[0].bytes[0]), 0.140625);
        assert_eq!(decode_e4m3fn(out[1].bytes[0]), 3.0);
    }
}

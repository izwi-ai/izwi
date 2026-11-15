//! What a load holds, priced from the config's census before any weight is
//! read, so load admission can refuse a device that cannot hold the model
//! with a clear message instead of failing mid-load.
//!
//! | Weight | Resident as |
//! |---|---|
//! | Encoder blocks, projector, decoder projections, experts, embedding | the compute dtype (BF16 on CUDA) |
//! | Encoder stem convolutions | F32 |
//! | Decoder norm gains, router projection and scale | F32 |
//! | `per_expert_scale` | host F32 |
//! | `lm_head.weight` | not loaded (tied to the embedding) |
//! | Rotation tables (cos, sin for every context position) | the compute dtype (48 MiB in BF16) |

use super::checkpoint::{expected_census, LM_HEAD};
use super::config::SovTranscribeConfig;

/// A load's resident bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SovResidency {
    /// Weights the device holds.
    pub device_bytes: u64,
    /// The largest single tensor a load stages on the host at once.
    pub largest_tensor_bytes: u64,
}

/// The resident weights of `config` with `compute_bytes` bytes per value of
/// the compute dtype (2 for BF16, 4 for the F32 test reference).
pub fn residency(config: &SovTranscribeConfig, compute_bytes: u64) -> SovResidency {
    let mut out = SovResidency::default();
    for entry in expected_census(config) {
        let elements = entry.shape.iter().product::<usize>().max(1) as u64;
        let name = entry.name.as_str();
        let width = if name == LM_HEAD || name.ends_with("per_expert_scale") {
            0
        } else if entry.f32
            || name.contains(".conv1.")
            || name.contains(".conv2.")
            || (name.contains("language_model")
                && (name.ends_with("norm.weight")
                    || name.ends_with("layernorm.weight")
                    || name.ends_with("layernorm_1.weight")
                    || name.ends_with("layernorm_2.weight")))
        {
            4
        } else {
            compute_bytes
        };
        out.device_bytes += elements * width;
        // A tensor is read from the mapping in its stored dtype (BF16).
        if width > 0 {
            out.largest_tensor_bytes = out.largest_tensor_bytes.max(elements * 2);
        }
    }
    // The rotation tables, `(cos, sin)` per position for each geometry.
    let d = &config.decoder;
    out.device_bytes += (super::decoder::ROPE_POSITIONS
        * (d.sliding.head_dim / 2 + d.full.head_dim / 2)
        * 2) as u64
        * compute_bytes;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_model_holds_48_gib_in_bf16() {
        let config = super::super::config::tests::published();
        let priced = residency(&config, 2);
        // 53.28 GB stored, less the 1.48 GB copy of the tied head.
        let gib = priced.device_bytes as f64 / (1u64 << 30) as f64;
        assert!((48.2..48.4).contains(&gib), "{gib}");
        // The embedding table is the largest tensor: 262,144 × 2,816 BF16.
        assert_eq!(priced.largest_tensor_bytes, 262_144 * 2_816 * 2);
    }

    #[test]
    fn the_loaded_synthetic_model_holds_what_the_plan_prices() {
        let dir = super::super::synthetic::temp_checkpoint();
        let model = super::super::model::SovTranscribeModel::load_on(
            dir.path(),
            None,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let config = SovTranscribeConfig::load(dir.path()).unwrap();
        assert_eq!(
            model.resident_bytes() as u64,
            residency(&config, 4).device_bytes
        );
    }
}

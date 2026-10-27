//! Maya1's checkpoint files and their published tensors.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::config::{Maya1LmConfig, WEIGHT_SHARDS};
use crate::error::{Error, Result};
use crate::models::shared::weights::staged::metal_staged_safetensors;

/// Both shards memory-mapped as one `VarBuilder` producing `dtype` on
/// `device`; on Metal a tensor is cast on the host and uploaded at its final
/// size, so an F16 load never holds a second copy.
pub(crate) fn mmap_lm(
    model_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let paths = WEIGHT_SHARDS
        .iter()
        .map(|shard| model_dir.join(shard))
        .collect::<Vec<_>>();
    if let Some(missing) = paths.iter().find(|path| !path.is_file()) {
        return Err(Error::ModelLoadError(format!(
            "Maya1 weights {} are missing",
            missing.display()
        )));
    }
    // SAFETY: the files are read-only model artifacts izwi verified on
    // download and does not modify while loaded.
    unsafe { metal_staged_safetensors(&paths, dtype, device) }
        .map_err(|error| Error::ModelLoadError(format!("Failed to map Maya1 weights: {error}")))
}

/// Every tensor of the published checkpoint (name, shape), all BF16: the
/// embedding, nine per layer and the final norm; no `lm_head` (tied).
pub fn published_tensors(cfg: &Maya1LmConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = cfg.hidden_size;
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![cfg.vocab_size, hidden],
        ),
        ("model.norm.weight".to_string(), vec![hidden]),
    ];
    for layer in 0..cfg.num_hidden_layers {
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        tensors.extend([
            (name("input_layernorm.weight"), vec![hidden]),
            (name("post_attention_layernorm.weight"), vec![hidden]),
            (name("self_attn.q_proj.weight"), vec![cfg.q_out(), hidden]),
            (name("self_attn.k_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.v_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.o_proj.weight"), vec![hidden, cfg.q_out()]),
            (
                name("mlp.gate_proj.weight"),
                vec![cfg.intermediate_size, hidden],
            ),
            (
                name("mlp.up_proj.weight"),
                vec![cfg.intermediate_size, hidden],
            ),
            (
                name("mlp.down_proj.weight"),
                vec![hidden, cfg.intermediate_size],
            ),
        ]);
    }
    tensors.sort();
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The census of the published safetensors headers (2026-10-26,
    /// `tasks/maya1-evidence-2026-10-26/maya1-tensor-census.txt`): 254 BF16
    /// tensors. Counts, edge names and shapes pinned here.
    #[test]
    fn published_tensor_list_matches_the_census() {
        let tensors = published_tensors(&Maya1LmConfig::pinned());
        assert_eq!(tensors.len(), 254);
        let find = |name: &str| {
            tensors
                .iter()
                .find(|(tensor, _)| tensor == name)
                .map(|(_, shape)| shape.clone())
        };
        assert_eq!(
            find("model.embed_tokens.weight"),
            Some(vec![156_960, 3_072])
        );
        assert_eq!(
            find("model.layers.0.self_attn.k_proj.weight"),
            Some(vec![1_024, 3_072])
        );
        assert_eq!(
            find("model.layers.27.mlp.down_proj.weight"),
            Some(vec![3_072, 8_192])
        );
        assert_eq!(find("lm_head.weight"), None);
    }
}

//! Maya2's checkpoint files and their published tensors.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::config::{Maya2BackboneConfig, WEIGHT_SHARDS};
use crate::error::{Error, Result};
use crate::models::shared::weights::staged::metal_staged_safetensors;

/// Both shards memory-mapped as one `VarBuilder` producing `dtype` on
/// `device` (on Metal each tensor is cast on the host and uploaded at its
/// final size).
pub(crate) fn mmap_release(
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
            "Maya2 weights {} are missing",
            missing.display()
        )));
    }
    // SAFETY: the files are read-only model artifacts izwi verified on
    // download and does not modify while loaded.
    unsafe { metal_staged_safetensors(&paths, dtype, device) }
        .map_err(|error| Error::ModelLoadError(format!("Failed to map Maya2 weights: {error}")))
}

/// Every tensor the release stores (name, shape), all BF16: the backbone
/// (its `lm_head` is an alias of the embedding and not stored), the
/// codebook-0 head, the audio input table, the speaker anchor and the depth
/// decoder, whose CAM++ speaker projection the release never uses.
pub fn published_tensors(cfg: &Maya2BackboneConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = cfg.hidden_size;
    let mut tensors = vec![
        ("audio_in.aeos_in".to_string(), vec![hidden]),
        ("audio_in.norm.weight".to_string(), vec![hidden]),
        ("audio_in.table.weight".to_string(), vec![32 * 2048, hidden]),
        ("backbone.cb0_head.weight".to_string(), vec![2049, hidden]),
        (
            "backbone.lm.model.embed_tokens.weight".to_string(),
            vec![cfg.vocab_size, hidden],
        ),
        ("backbone.lm.model.norm.weight".to_string(), vec![hidden]),
        ("speaker_anchor.fc1.bias".to_string(), vec![512]),
        ("speaker_anchor.fc1.weight".to_string(), vec![512, 192]),
        ("speaker_anchor.fc2.bias".to_string(), vec![hidden]),
        ("speaker_anchor.fc2.weight".to_string(), vec![hidden, 512]),
        ("speaker_anchor.null_vec".to_string(), vec![192]),
        ("depth.code_lo.weight".to_string(), vec![31 * 2048, 128]),
        ("depth.code_up.weight".to_string(), vec![1024, 128]),
        ("depth.in_proj.weight".to_string(), vec![1024, hidden]),
        ("depth.level_emb.weight".to_string(), vec![32, 1024]),
        ("depth.norm.bias".to_string(), vec![1024]),
        ("depth.norm.weight".to_string(), vec![1024]),
        ("depth.null_mem".to_string(), vec![1, 1, 1024]),
        ("depth.spk_proj_campp.weight".to_string(), vec![1024, 1024]),
        ("depth.spk_proj_ecapa.weight".to_string(), vec![1024, 192]),
    ];
    for layer in 0..cfg.num_hidden_layers {
        let name = |suffix: &str| format!("backbone.lm.model.layers.{layer}.{suffix}");
        tensors.extend([
            (name("input_layernorm.weight"), vec![hidden]),
            (name("post_attention_layernorm.weight"), vec![hidden]),
            (name("self_attn.q_proj.weight"), vec![cfg.q_out(), hidden]),
            (name("self_attn.k_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.v_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.o_proj.weight"), vec![hidden, cfg.q_out()]),
            (name("self_attn.q_norm.weight"), vec![cfg.head_dim]),
            (name("self_attn.k_norm.weight"), vec![cfg.head_dim]),
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
    for layer in 0..4 {
        let name = |suffix: &str| format!("depth.cross.{layer}.{suffix}");
        tensors.extend([
            (name("kv.weight"), vec![2048, 1024]),
            (name("n.bias"), vec![1024]),
            (name("n.weight"), vec![1024]),
            (name("o.weight"), vec![1024, 1024]),
            (name("q.weight"), vec![1024, 1024]),
        ]);
    }
    for group in 0..11 {
        for layer in 0..4 {
            let name = |suffix: &str| format!("depth.groups.{group}.{layer}.{suffix}");
            tensors.extend([
                (name("f1.weight"), vec![3072, 1024]),
                (name("f2.weight"), vec![1024, 3072]),
                (name("n1.bias"), vec![1024]),
                (name("n1.weight"), vec![1024]),
                (name("n2.bias"), vec![1024]),
                (name("n2.weight"), vec![1024]),
                (name("o.weight"), vec![1024, 1024]),
                (name("qkv.weight"), vec![3072, 1024]),
            ]);
        }
    }
    for level in 0..31 {
        tensors.push((format!("depth.heads.{level}.weight"), vec![2048, 1024]));
    }
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published header (both shards, `1403b841`) holds 731 tensors and
    /// 2,502,398,656 parameters.
    #[test]
    fn inventory_matches_the_published_census() {
        let tensors = published_tensors(&Maya2BackboneConfig::pinned());
        assert_eq!(tensors.len(), 731);
        let params = tensors
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum::<usize>();
        assert_eq!(params, 2_502_398_656);
    }

    /// Every published tensor has the inventory's name and shape (ignored:
    /// needs the downloaded shards).
    #[test]
    #[ignore = "needs IZWI_MAYA2_MODEL_DIR"]
    fn inventory_matches_the_downloaded_shards() {
        let dir = std::path::PathBuf::from(
            std::env::var("IZWI_MAYA2_MODEL_DIR").expect("IZWI_MAYA2_MODEL_DIR"),
        );
        let mut found = std::collections::BTreeMap::new();
        for shard in WEIGHT_SHARDS {
            let tensors =
                unsafe { candle_core::safetensors::MmapedSafetensors::new(dir.join(shard)) }
                    .unwrap();
            for (name, view) in tensors.tensors() {
                assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name}");
                found.insert(name, view.shape().to_vec());
            }
        }
        let expected = published_tensors(&Maya2BackboneConfig::pinned())
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(found, expected);
    }
}

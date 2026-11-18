//! Rumble's checkpoint layout and its census check.
//!
//! The published file is one `model.safetensors` in the Transformers-native
//! layout: `model.audio_tower.*`, `model.multi_modal_projector.*` and
//! `model.language_model.*`, with the output head tied to the token table.
//! Before any weight is mapped the loader compares the file's header with
//! the tensors the config implies, so a re-laid-out or truncated release
//! fails with the tensor's name instead of a shape error deep in a layer,
//! and a tensor this loader would silently leave unread is refused.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

use super::config::RumbleConfig;

pub(crate) const RUMBLE_WEIGHTS_FILE: &str = "model.safetensors";
/// Files the loader reads besides the weights.
pub(crate) const RUMBLE_CONFIG_ARTIFACTS: &[&str] =
    &["config.json", "preprocessor_config.json", "tokenizer.json"];

/// Prefix of every tensor in the file.
pub(crate) const RUMBLE_MODEL_PREFIX: &str = "model";
pub(crate) const RUMBLE_AUDIO_TOWER_PREFIX: &str = "audio_tower";
pub(crate) const RUMBLE_PROJECTOR_PREFIX: &str = "multi_modal_projector";
pub(crate) const RUMBLE_LANGUAGE_MODEL_PREFIX: &str = "language_model";

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// The checkpoint's weights file, once every file the loader reads exists.
pub fn ensure_rumble_artifacts(model_dir: &Path) -> Result<PathBuf> {
    let weights = model_dir.join(RUMBLE_WEIGHTS_FILE);
    let configs = RUMBLE_CONFIG_ARTIFACTS
        .iter()
        .map(|file| model_dir.join(file));
    for path in configs.chain(std::iter::once(weights.clone())) {
        if !path.is_file() {
            return Err(Error::ModelNotFound(format!(
                "Missing Rumble artifact {}",
                path.display()
            )));
        }
    }
    Ok(weights)
}

/// Every tensor the loader reads, with its shape, for `config`.
pub(crate) fn rumble_tensor_plan(config: &RumbleConfig) -> BTreeMap<String, Vec<usize>> {
    let audio = &config.audio_config;
    let text = &config.text_config;
    let head_dim = config.decoder_head_dim();
    let mut plan = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>| {
        plan.insert(format!("{RUMBLE_MODEL_PREFIX}.{name}"), shape);
    };

    let tower = RUMBLE_AUDIO_TOWER_PREFIX;
    let channels = audio.downsample_hidden_size;
    for (index, in_channels) in [(1, 1), (2, channels), (3, channels)] {
        add(
            format!("{tower}.conv2d{index}.weight"),
            vec![channels, in_channels, 3, 3],
        );
        add(format!("{tower}.conv2d{index}.bias"), vec![channels]);
    }
    add(
        format!("{tower}.conv_out.weight"),
        vec![audio.d_model, channels * (audio.num_mel_bins / 8)],
    );
    for layer in 0..audio.encoder_layers {
        let base = format!("{tower}.layers.{layer}");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add(
                format!("{base}.self_attn.{projection}.weight"),
                vec![audio.d_model, audio.d_model],
            );
            add(
                format!("{base}.self_attn.{projection}.bias"),
                vec![audio.d_model],
            );
        }
        for norm in ["self_attn_layer_norm", "final_layer_norm"] {
            add(format!("{base}.{norm}.weight"), vec![audio.d_model]);
            add(format!("{base}.{norm}.bias"), vec![audio.d_model]);
        }
        add(
            format!("{base}.fc1.weight"),
            vec![audio.encoder_ffn_dim, audio.d_model],
        );
        add(format!("{base}.fc1.bias"), vec![audio.encoder_ffn_dim]);
        add(
            format!("{base}.fc2.weight"),
            vec![audio.d_model, audio.encoder_ffn_dim],
        );
        add(format!("{base}.fc2.bias"), vec![audio.d_model]);
    }
    add(format!("{tower}.ln_post.weight"), vec![audio.d_model]);
    add(format!("{tower}.ln_post.bias"), vec![audio.d_model]);

    let projector = RUMBLE_PROJECTOR_PREFIX;
    add(
        format!("{projector}.linear_1.weight"),
        vec![audio.d_model, audio.d_model],
    );
    add(format!("{projector}.linear_1.bias"), vec![audio.d_model]);
    add(
        format!("{projector}.linear_2.weight"),
        vec![audio.output_dim, audio.d_model],
    );
    add(format!("{projector}.linear_2.bias"), vec![audio.output_dim]);

    let lm = RUMBLE_LANGUAGE_MODEL_PREFIX;
    let query = text.num_attention_heads * head_dim;
    let key_value = text.num_key_value_heads * head_dim;
    add(
        format!("{lm}.embed_tokens.weight"),
        vec![text.vocab_size, text.hidden_size],
    );
    for layer in 0..text.num_hidden_layers {
        let base = format!("{lm}.layers.{layer}");
        add(
            format!("{base}.self_attn.q_proj.weight"),
            vec![query, text.hidden_size],
        );
        add(
            format!("{base}.self_attn.k_proj.weight"),
            vec![key_value, text.hidden_size],
        );
        add(
            format!("{base}.self_attn.v_proj.weight"),
            vec![key_value, text.hidden_size],
        );
        add(
            format!("{base}.self_attn.o_proj.weight"),
            vec![text.hidden_size, query],
        );
        add(format!("{base}.self_attn.q_norm.weight"), vec![head_dim]);
        add(format!("{base}.self_attn.k_norm.weight"), vec![head_dim]);
        add(
            format!("{base}.input_layernorm.weight"),
            vec![text.hidden_size],
        );
        add(
            format!("{base}.post_attention_layernorm.weight"),
            vec![text.hidden_size],
        );
        add(
            format!("{base}.mlp.gate_proj.weight"),
            vec![text.intermediate_size, text.hidden_size],
        );
        add(
            format!("{base}.mlp.up_proj.weight"),
            vec![text.intermediate_size, text.hidden_size],
        );
        add(
            format!("{base}.mlp.down_proj.weight"),
            vec![text.hidden_size, text.intermediate_size],
        );
    }
    add(format!("{lm}.norm.weight"), vec![text.hidden_size]);
    plan
}

/// One tensor of a safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RumbleHeaderTensor {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
}

/// The tensors a safetensors file declares, from its header alone.
pub(crate) fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, RumbleHeaderTensor>> {
    let invalid = |reason: String| {
        Error::ModelLoadError(format!(
            "Invalid safetensors header in {}: {reason}",
            path.display()
        ))
    };
    let mut file = File::open(path)?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(invalid(format!("header length {length}")));
    }
    let mut raw = vec![0_u8; length as usize];
    file.read_exact(&mut raw)?;
    let header: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|error| invalid(error.to_string()))?;
    let entries = header
        .as_object()
        .ok_or_else(|| invalid("not an object".into()))?;
    let mut tensors = BTreeMap::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry
            .get("dtype")
            .and_then(|value| value.as_str())
            .ok_or_else(|| invalid(format!("{name} has no dtype")))?
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| invalid(format!("{name} has no shape")))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|dim| usize::try_from(dim).ok())
                    .ok_or_else(|| invalid(format!("{name} has a non-integer dimension")))
            })
            .collect::<Result<Vec<_>>>()?;
        tensors.insert(name.clone(), RumbleHeaderTensor { dtype, shape });
    }
    Ok(tensors)
}

/// Check a file's declared tensors against the plan: every planned tensor
/// present with its shape in a float dtype, and nothing else in the file.
pub(crate) fn check_rumble_census(
    plan: &BTreeMap<String, Vec<usize>>,
    header: &BTreeMap<String, RumbleHeaderTensor>,
) -> Result<()> {
    let fail = |reason: String| {
        Err(Error::ModelLoadError(format!(
            "Rumble checkpoint does not match its config: {reason}"
        )))
    };
    for (name, shape) in plan {
        let Some(tensor) = header.get(name) else {
            return fail(format!("missing tensor {name}"));
        };
        if &tensor.shape != shape {
            return fail(format!(
                "tensor {name} has shape {:?}, expected {shape:?}",
                tensor.shape
            ));
        }
        if !matches!(tensor.dtype.as_str(), "BF16" | "F16" | "F32") {
            return fail(format!("tensor {name} has dtype {}", tensor.dtype));
        }
    }
    if let Some(extra) = header.keys().find(|name| !plan.contains_key(*name)) {
        return fail(format!(
            "tensor {extra} is not part of the layout this loader reads ({} unread in all)",
            header
                .keys()
                .filter(|name| !plan.contains_key(*name))
                .count()
        ));
    }
    Ok(())
}

/// Parameters in the planned tensors.
pub(crate) fn rumble_parameter_count(plan: &BTreeMap<String, Vec<usize>>) -> u64 {
    plan.values()
        .map(|shape| shape.iter().map(|dim| *dim as u64).product::<u64>())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUBLISHED_CONFIG: &str = include_str!("../../../../tests/fixtures/rumble/config.json");
    const PUBLISHED_CENSUS: &str =
        include_str!("../../../../tests/fixtures/rumble/tensor_census.json");

    fn published_header() -> BTreeMap<String, RumbleHeaderTensor> {
        let census: serde_json::Value = serde_json::from_str(PUBLISHED_CENSUS).unwrap();
        census["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    RumbleHeaderTensor {
                        dtype: entry["dtype"].as_str().unwrap().to_string(),
                        shape: entry["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|dim| dim.as_u64().unwrap() as usize)
                            .collect(),
                    },
                )
            })
            .collect()
    }

    /// The plan is checked against the PUBLISHED header (fetched from the
    /// pinned revision), not against a fixture generated from the plan.
    #[test]
    fn the_plan_matches_the_published_checkpoint_exactly() {
        let config = RumbleConfig::from_json(PUBLISHED_CONFIG).unwrap();
        let plan = rumble_tensor_plan(&config);
        let header = published_header();
        assert_eq!(header.len(), 457);
        assert_eq!(plan.len(), header.len());
        check_rumble_census(&plan, &header).unwrap();
        // 1,017,263,104 BF16 tensor bytes of the 1,017,318,776-byte file.
        assert_eq!(rumble_parameter_count(&plan), 508_631_552);
    }

    #[test]
    fn a_missing_reshaped_or_extra_tensor_is_named() {
        let config = RumbleConfig::from_json(PUBLISHED_CONFIG).unwrap();
        let plan = rumble_tensor_plan(&config);

        let mut header = published_header();
        header.remove("model.multi_modal_projector.linear_2.weight");
        let error = check_rumble_census(&plan, &header).unwrap_err().to_string();
        assert!(
            error.contains("multi_modal_projector.linear_2.weight"),
            "{error}"
        );

        let mut header = published_header();
        header
            .get_mut("model.language_model.layers.3.self_attn.q_proj.weight")
            .unwrap()
            .shape = vec![1024, 1024];
        let error = check_rumble_census(&plan, &header).unwrap_err().to_string();
        assert!(error.contains("layers.3.self_attn.q_proj"), "{error}");

        let mut header = published_header();
        header.insert(
            "lm_head.weight".into(),
            RumbleHeaderTensor {
                dtype: "BF16".into(),
                shape: vec![99_630, 1024],
            },
        );
        let error = check_rumble_census(&plan, &header).unwrap_err().to_string();
        assert!(error.contains("lm_head.weight"), "{error}");
    }
}

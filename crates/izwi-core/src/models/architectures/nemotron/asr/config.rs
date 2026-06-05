use serde_yaml::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NemotronConfigInventory {
    pub sample_rate: Option<usize>,
    pub features: Option<usize>,
    pub encoder_layers: Option<usize>,
    pub encoder_dim: Option<usize>,
    pub encoder_heads: Option<usize>,
    pub predictor_hidden: Option<usize>,
    pub joint_hidden: Option<usize>,
    pub vocab_size: Option<usize>,
    pub prompt_dim: Option<usize>,
    pub left_context_frames: Option<usize>,
    pub right_context_frames: Vec<usize>,
    pub model_target: Option<String>,
    pub output_vocabulary: Vec<String>,
}

impl NemotronConfigInventory {
    pub fn from_yaml_str(contents: &str) -> Result<Self> {
        let value: Value = serde_yaml::from_str(contents).map_err(|e| {
            Error::ModelLoadError(format!("Failed parsing Nemotron model_config.yaml: {e}"))
        })?;

        Ok(Self {
            sample_rate: find_usize_by_keys(&value, &["sample_rate", "sample_rate_hz"]),
            features: find_usize_by_keys(&value, &["features", "n_mels", "num_mels"]),
            encoder_layers: find_usize_by_keys(&value, &["n_layers", "num_layers", "layers"]),
            encoder_dim: find_usize_by_keys(&value, &["d_model", "feat_out", "encoder_dim"]),
            encoder_heads: find_usize_by_keys(&value, &["n_heads", "num_heads"]),
            predictor_hidden: find_usize_by_keys(
                &value,
                &["pred_hidden", "prediction_hidden", "pred_hidden_dim"],
            ),
            joint_hidden: find_usize_by_keys(&value, &["joint_hidden", "joint_hidden_dim"]),
            vocab_size: find_usize_by_keys(&value, &["vocab_size", "num_classes"]),
            prompt_dim: find_usize_by_keys(&value, &["prompt_dim", "prompt_size", "lang_dim"]),
            left_context_frames: att_context_size(&value).and_then(|pair| pair.first().copied()),
            right_context_frames: collect_right_context_frames(&value),
            model_target: find_string_by_keys(&value, &["_target_", "target"]),
            output_vocabulary: output_vocabulary(&value),
        })
    }
}

fn output_vocabulary(value: &Value) -> Vec<String> {
    find_non_empty_string_sequence_by_key(value, "labels")
        .or_else(|| find_non_empty_string_sequence_by_key(value, "vocabulary"))
        .unwrap_or_default()
}

fn collect_right_context_frames(value: &Value) -> Vec<usize> {
    let mut contexts = Vec::new();
    if let Some(pair) = att_context_size(value) {
        if let Some(right) = pair.get(1).copied() {
            contexts.push(right);
        }
    }

    collect_context_sequences(value, &mut contexts);
    contexts.sort_unstable();
    contexts.dedup();
    contexts
}

fn att_context_size(value: &Value) -> Option<Vec<usize>> {
    find_sequence_by_key(value, "att_context_size")
        .or_else(|| find_sequence_by_key(value, "att_context_size_all"))
        .map(|seq| seq.iter().filter_map(value_to_usize).collect())
}

fn collect_context_sequences(value: &Value, contexts: &mut Vec<usize>) {
    match value {
        Value::Sequence(items) => {
            if items.len() == 2 {
                let maybe_pair: Vec<_> = items.iter().filter_map(value_to_usize).collect();
                if maybe_pair.len() == 2 {
                    contexts.push(maybe_pair[1]);
                }
            }
            for item in items {
                collect_context_sequences(item, contexts);
            }
        }
        Value::Mapping(map) => {
            for (key, value) in map {
                if scalar_key(key)
                    .map(|key| key.contains("att_context_size"))
                    .unwrap_or(false)
                {
                    if let Some(seq) = value.as_sequence() {
                        if seq.len() == 2 {
                            if let Some(right) = seq.get(1).and_then(value_to_usize) {
                                contexts.push(right);
                            }
                        } else {
                            for child in seq {
                                collect_context_sequences(child, contexts);
                            }
                        }
                    }
                }
                collect_context_sequences(value, contexts);
            }
        }
        _ => {}
    }
}

fn find_sequence_by_key(value: &Value, target: &str) -> Option<Vec<Value>> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if scalar_key(key).is_some_and(|key| key == target) {
                    return child.as_sequence().cloned();
                }
                if let Some(found) = find_sequence_by_key(child, target) {
                    return Some(found);
                }
            }
            None
        }
        Value::Sequence(items) => items
            .iter()
            .find_map(|item| find_sequence_by_key(item, target)),
        _ => None,
    }
}

fn find_non_empty_string_sequence_by_key(value: &Value, target: &str) -> Option<Vec<String>> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if scalar_key(key).is_some_and(|key| key == target) {
                    if let Some(seq) = child.as_sequence() {
                        let tokens = seq
                            .iter()
                            .filter_map(value_to_string_scalar)
                            .collect::<Vec<_>>();
                        if !tokens.is_empty() {
                            return Some(tokens);
                        }
                    }
                }

                if let Some(found) = find_non_empty_string_sequence_by_key(child, target) {
                    return Some(found);
                }
            }
            None
        }
        Value::Sequence(items) => items
            .iter()
            .find_map(|item| find_non_empty_string_sequence_by_key(item, target)),
        _ => None,
    }
}

fn find_usize_by_keys(value: &Value, keys: &[&str]) -> Option<usize> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if let Some(name) = scalar_key(key) {
                    if keys.iter().any(|candidate| *candidate == name) {
                        if let Some(value) = value_to_usize(child) {
                            return Some(value);
                        }
                    }
                }
            }
            map.values()
                .find_map(|child| find_usize_by_keys(child, keys))
        }
        Value::Sequence(items) => items.iter().find_map(|item| find_usize_by_keys(item, keys)),
        _ => None,
    }
}

fn find_string_by_keys(value: &Value, keys: &[&str]) -> Option<String> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if let Some(name) = scalar_key(key) {
                    if keys.iter().any(|candidate| *candidate == name) {
                        if let Some(value) = child.as_str() {
                            return Some(value.to_string());
                        }
                    }
                }
            }
            map.values()
                .find_map(|child| find_string_by_keys(child, keys))
        }
        Value::Sequence(items) => items
            .iter()
            .find_map(|item| find_string_by_keys(item, keys)),
        _ => None,
    }
}

fn value_to_usize(value: &Value) -> Option<usize> {
    if let Some(num) = value.as_u64() {
        return usize::try_from(num).ok();
    }
    if let Some(num) = value.as_i64() {
        return usize::try_from(num).ok();
    }
    value.as_str()?.parse::<usize>().ok()
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    if let Some(value) = value.as_str() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_u64() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_i64() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_f64() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_bool() {
        return Some(value.to_string());
    }
    None
}

fn scalar_key(value: &Value) -> Option<&str> {
    value.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_inventory_extracts_nemotron_style_fields() {
        let yaml = r#"
model:
  _target_: nemo.collections.asr.models.EncDecRNNTBPEModel
  preprocessor:
    sample_rate: 16000
    features: 128
  encoder:
    n_layers: 24
    d_model: 512
    n_heads: 8
    att_context_size: [56, 13]
    att_context_size_all:
      - [56, 0]
      - [56, 1]
      - [56, 3]
      - [56, 6]
      - [56, 13]
  decoder:
    pred_hidden: 640
    vocab_size: 1024
  joint:
    joint_hidden: 512
    vocabulary:
      - <unk>
      - ▁joint
  prompt:
    prompt_dim: 128
labels:
  - <unk>
  - <en-US>
  - ▁Hello
  - world
"#;

        let inventory = NemotronConfigInventory::from_yaml_str(yaml).unwrap();

        assert_eq!(inventory.sample_rate, Some(16_000));
        assert_eq!(inventory.features, Some(128));
        assert_eq!(inventory.encoder_layers, Some(24));
        assert_eq!(inventory.encoder_dim, Some(512));
        assert_eq!(inventory.encoder_heads, Some(8));
        assert_eq!(inventory.predictor_hidden, Some(640));
        assert_eq!(inventory.joint_hidden, Some(512));
        assert_eq!(inventory.vocab_size, Some(1024));
        assert_eq!(inventory.prompt_dim, Some(128));
        assert_eq!(inventory.left_context_frames, Some(56));
        assert_eq!(inventory.right_context_frames, vec![0, 1, 3, 6, 13]);
        assert_eq!(
            inventory.output_vocabulary,
            vec!["<unk>", "<en-US>", "▁Hello", "world"]
        );
    }

    #[test]
    fn config_inventory_falls_back_to_non_empty_joint_vocabulary() {
        let yaml = r#"
model:
  tokenizer:
    vocabulary: []
  joint:
    num_classes: 3
    vocabulary:
      - <unk>
      - hello
      - world
"#;

        let inventory = NemotronConfigInventory::from_yaml_str(yaml).unwrap();

        assert_eq!(inventory.vocab_size, Some(3));
        assert_eq!(inventory.output_vocabulary, vec!["<unk>", "hello", "world"]);
    }
}

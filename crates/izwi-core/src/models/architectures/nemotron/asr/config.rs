use serde_yaml::Value;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NemotronConfigInventory {
    pub sample_rate: Option<usize>,
    pub features: Option<usize>,
    pub n_fft: Option<usize>,
    pub window_length: Option<usize>,
    pub hop_length: Option<usize>,
    pub normalize: Option<String>,
    pub encoder_layers: Option<usize>,
    pub encoder_dim: Option<usize>,
    pub encoder_heads: Option<usize>,
    pub subsampling_factor: Option<usize>,
    pub subsampling_conv_channels: Option<usize>,
    pub ff_expansion_factor: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub predictor_hidden: Option<usize>,
    pub predictor_layers: Option<usize>,
    pub joint_hidden: Option<usize>,
    pub vocab_size: Option<usize>,
    pub prompt_dim: Option<usize>,
    pub prompt_dictionary: Vec<(String, usize)>,
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

        let sample_rate = find_usize_by_keys(&value, &["sample_rate", "sample_rate_hz"]);

        Ok(Self {
            sample_rate,
            features: find_usize_by_keys(&value, &["features", "n_mels", "num_mels"]),
            n_fft: find_usize_by_keys(&value, &["n_fft", "fft_length"]),
            window_length: find_frame_samples(&value, sample_rate, &["win_length", "window_size"]),
            hop_length: find_frame_samples(&value, sample_rate, &["hop_length", "window_stride"]),
            normalize: find_string_by_keys(&value, &["normalize"]),
            encoder_layers: find_usize_by_keys(&value, &["n_layers", "num_layers", "layers"]),
            encoder_dim: find_usize_by_keys(&value, &["d_model", "feat_out", "encoder_dim"]),
            encoder_heads: find_usize_by_keys(&value, &["n_heads", "num_heads"]),
            subsampling_factor: find_usize_by_keys(&value, &["subsampling_factor"]),
            subsampling_conv_channels: find_usize_by_keys(&value, &["subsampling_conv_channels"]),
            ff_expansion_factor: find_usize_by_keys(&value, &["ff_expansion_factor"]),
            conv_kernel_size: find_usize_by_keys(&value, &["conv_kernel_size"]),
            predictor_hidden: find_usize_by_keys(
                &value,
                &["pred_hidden", "prediction_hidden", "pred_hidden_dim"],
            ),
            predictor_layers: find_usize_by_keys(&value, &["pred_rnn_layers", "prediction_layers"]),
            joint_hidden: find_usize_by_keys(&value, &["joint_hidden", "joint_hidden_dim"]),
            vocab_size: find_usize_by_keys(&value, &["vocab_size", "num_classes"]),
            prompt_dim: find_usize_by_keys(&value, &["prompt_dim", "prompt_size", "lang_dim"])
                .or_else(|| find_usize_by_keys(&value, &["num_prompts"])),
            prompt_dictionary: prompt_dictionary(&value),
            left_context_frames: att_context_size(&value).and_then(|pair| pair.first().copied()),
            right_context_frames: collect_right_context_frames(&value),
            model_target: find_string_by_keys(&value, &["_target_", "target"]),
            output_vocabulary: output_vocabulary(&value),
        })
    }
}

fn prompt_dictionary(value: &Value) -> Vec<(String, usize)> {
    let mut entries = find_usize_mapping_by_key(value, "prompt_dictionary").unwrap_or_default();
    entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    entries
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

fn find_frame_samples(value: &Value, sample_rate: Option<usize>, keys: &[&str]) -> Option<usize> {
    let raw = find_value_by_keys(value, keys)?;
    if let Some(samples) = value_to_usize(raw) {
        return Some(samples);
    }
    let seconds = value_to_f64(raw)?;
    let sample_rate = sample_rate?;
    Some((seconds * sample_rate as f64).round().max(1.0) as usize)
}

fn find_value_by_keys<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if let Some(name) = scalar_key(key) {
                    if keys.iter().any(|candidate| *candidate == name) {
                        return Some(child);
                    }
                }
            }
            map.values()
                .find_map(|child| find_value_by_keys(child, keys))
        }
        Value::Sequence(items) => items.iter().find_map(|item| find_value_by_keys(item, keys)),
        _ => None,
    }
}

fn find_usize_mapping_by_key(value: &Value, target: &str) -> Option<Vec<(String, usize)>> {
    match value {
        Value::Mapping(map) => {
            for (key, child) in map {
                if scalar_key(key).is_some_and(|key| key == target) {
                    if let Value::Mapping(child_map) = child {
                        let entries = child_map
                            .iter()
                            .filter_map(|(key, value)| {
                                Some((scalar_key(key)?.to_string(), value_to_usize(value)?))
                            })
                            .collect::<Vec<_>>();
                        if !entries.is_empty() {
                            return Some(entries);
                        }
                    }
                }

                if let Some(found) = find_usize_mapping_by_key(child, target) {
                    return Some(found);
                }
            }
            None
        }
        Value::Sequence(items) => items
            .iter()
            .find_map(|item| find_usize_mapping_by_key(item, target)),
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

fn value_to_f64(value: &Value) -> Option<f64> {
    if let Some(num) = value.as_f64() {
        return Some(num);
    }
    if let Some(num) = value.as_u64() {
        return Some(num as f64);
    }
    if let Some(num) = value.as_i64() {
        return Some(num as f64);
    }
    value.as_str()?.parse::<f64>().ok()
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
    n_fft: 512
    window_size: 0.025
    window_stride: 0.01
    normalize: NA
  encoder:
    n_layers: 24
    d_model: 512
    n_heads: 8
    subsampling_factor: 8
    subsampling_conv_channels: 256
    ff_expansion_factor: 4
    conv_kernel_size: 9
    att_context_size: [56, 13]
    att_context_size_all:
      - [56, 0]
      - [56, 1]
      - [56, 3]
      - [56, 6]
      - [56, 13]
  decoder:
    prednet:
      pred_hidden: 640
      pred_rnn_layers: 2
    vocab_size: 1024
  joint:
    jointnet:
      joint_hidden: 512
    vocabulary:
      - <unk>
      - ▁joint
  model_defaults:
    num_prompts: 128
    prompt_dictionary:
      en-US: 0
      auto: 101
labels:
  - <unk>
  - <en-US>
  - ▁Hello
  - world
"#;

        let inventory = NemotronConfigInventory::from_yaml_str(yaml).unwrap();

        assert_eq!(inventory.sample_rate, Some(16_000));
        assert_eq!(inventory.features, Some(128));
        assert_eq!(inventory.n_fft, Some(512));
        assert_eq!(inventory.window_length, Some(400));
        assert_eq!(inventory.hop_length, Some(160));
        assert_eq!(inventory.normalize.as_deref(), Some("NA"));
        assert_eq!(inventory.encoder_layers, Some(24));
        assert_eq!(inventory.encoder_dim, Some(512));
        assert_eq!(inventory.encoder_heads, Some(8));
        assert_eq!(inventory.subsampling_factor, Some(8));
        assert_eq!(inventory.subsampling_conv_channels, Some(256));
        assert_eq!(inventory.ff_expansion_factor, Some(4));
        assert_eq!(inventory.conv_kernel_size, Some(9));
        assert_eq!(inventory.predictor_hidden, Some(640));
        assert_eq!(inventory.predictor_layers, Some(2));
        assert_eq!(inventory.joint_hidden, Some(512));
        assert_eq!(inventory.vocab_size, Some(1024));
        assert_eq!(inventory.prompt_dim, Some(128));
        assert_eq!(
            inventory.prompt_dictionary,
            vec![("en-US".to_string(), 0), ("auto".to_string(), 101)]
        );
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

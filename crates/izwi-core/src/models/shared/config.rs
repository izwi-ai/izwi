//! Shared config parsing helpers for model loaders.

use candle_core::DType;
use serde_json::Value;

use crate::backends::parse_dtype_name;

pub fn checkpoint_dtype_from_config_json(config_str: &str) -> Option<DType> {
    let value: Value = serde_json::from_str(config_str).ok()?;
    checkpoint_dtype_from_config_value(&value)
}

pub fn checkpoint_dtype_from_config_value(value: &Value) -> Option<DType> {
    dtype_from_config_object(value)
        .or_else(|| value.get("text_config").and_then(dtype_from_config_object))
}

fn dtype_from_config_object(value: &Value) -> Option<DType> {
    ["torch_dtype", "dtype"]
        .iter()
        .filter_map(|key| value.get(key).and_then(Value::as_str))
        .find_map(parse_dtype_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_dtype_reads_root_torch_dtype() {
        let dtype = checkpoint_dtype_from_config_json(r#"{"torch_dtype":"torch.float16"}"#);
        assert_eq!(dtype, Some(DType::F16));
    }

    #[test]
    fn checkpoint_dtype_reads_nested_text_config_dtype() {
        let dtype = checkpoint_dtype_from_config_json(
            r#"{"model_type":"wrapper","text_config":{"dtype":"float32"}}"#,
        );
        assert_eq!(dtype, Some(DType::F32));
    }

    #[test]
    fn checkpoint_dtype_prefers_root_over_text_config() {
        let dtype = checkpoint_dtype_from_config_json(
            r#"{"torch_dtype":"bfloat16","text_config":{"torch_dtype":"float16"}}"#,
        );
        assert_eq!(dtype, Some(DType::BF16));
    }
}

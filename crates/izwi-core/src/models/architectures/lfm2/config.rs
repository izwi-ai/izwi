use candle_core::quantized::gguf_file::Value as GgufValue;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

/// Model-neutral LFM2 backbone geometry parsed from the loaded GGUF.
#[derive(Debug, Clone)]
pub struct Lfm2BackboneConfig {
    pub architecture: String,
    pub block_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub embedding_length_out: Option<usize>,
    pub feed_forward_length: Option<usize>,
    pub attention_head_count: usize,
    pub attention_head_count_kv: Vec<usize>,
    pub attention_layer_norm_rms_epsilon: f64,
    pub attention_sliding_window: Option<usize>,
    pub rope_freq_base: f64,
    pub shortconv_l_cache: usize,
}

pub(crate) fn parse_lfm2_backbone_config(loader: &GgufLoader) -> Result<Lfm2BackboneConfig> {
    let block_count = required_usize(loader, "lfm2.block_count")?;
    let attention_head_count = required_usize(loader, "lfm2.attention.head_count")?;
    Ok(Lfm2BackboneConfig {
        architecture: loader
            .get_metadata_string("general.architecture")
            .unwrap_or_else(|| "lfm2".to_string()),
        block_count,
        context_length: required_usize(loader, "lfm2.context_length")?,
        embedding_length: required_usize(loader, "lfm2.embedding_length")?,
        embedding_length_out: optional_usize(loader, "lfm2.embedding_length_out"),
        feed_forward_length: optional_usize(loader, "lfm2.feed_forward_length"),
        attention_head_count,
        attention_head_count_kv: required_usize_or_array(
            loader,
            "lfm2.attention.head_count_kv",
            block_count,
        )?,
        attention_layer_norm_rms_epsilon: required_f64(
            loader,
            "lfm2.attention.layer_norm_rms_epsilon",
        )?,
        attention_sliding_window: optional_usize(loader, "lfm2.attention.sliding_window")
            .filter(|value| *value > 0),
        rope_freq_base: loader
            .metadata_value("lfm2.rope.freq_base")
            .and_then(gguf_to_f64)
            .unwrap_or(1_000_000.0),
        shortconv_l_cache: required_usize(loader, "lfm2.shortconv.l_cache")?,
    })
}

fn required_usize(loader: &GgufLoader, key: &str) -> Result<usize> {
    optional_usize(loader, key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn optional_usize(loader: &GgufLoader, key: &str) -> Option<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn required_f64(loader: &GgufLoader, key: &str) -> Result<f64> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn required_usize_or_array(loader: &GgufLoader, key: &str, len: usize) -> Result<Vec<usize>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    match value {
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let raw = gguf_to_u64(item).ok_or_else(|| {
                    Error::ModelLoadError(format!("Expected integer array values for {key}"))
                })?;
                out.push(usize::try_from(raw).map_err(|_| {
                    Error::ModelLoadError(format!("Array value out of range for {key}: {raw}"))
                })?);
            }
            if out.len() == len {
                Ok(out)
            } else if out.len() == 1 {
                Ok(vec![out[0]; len])
            } else {
                Err(Error::ModelLoadError(format!(
                    "Unexpected GGUF metadata array length for {key}: expected {len}, found {}",
                    out.len()
                )))
            }
        }
        _ => {
            let value = gguf_to_u64(value).ok_or_else(|| {
                Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}"))
            })?;
            let value = usize::try_from(value).map_err(|_| {
                Error::ModelLoadError(format!("Metadata value out of range for {key}: {value}"))
            })?;
            Ok(vec![value; len])
        }
    }
}

fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U64(n) => Some(*n),
        GgufValue::I64(n) => Some(*n as u64),
        GgufValue::U32(n) => Some(*n as u64),
        GgufValue::I32(n) => Some(*n as u64),
        GgufValue::U16(n) => Some(*n as u64),
        GgufValue::I16(n) => Some(*n as u64),
        GgufValue::U8(n) => Some(*n as u64),
        GgufValue::I8(n) => Some(*n as u64),
        GgufValue::F32(n) => Some(*n as u64),
        GgufValue::F64(n) => Some(*n as u64),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(n) => Some(*n),
        GgufValue::F32(n) => Some(*n as f64),
        GgufValue::U64(n) => Some(*n as f64),
        GgufValue::I64(n) => Some(*n as f64),
        GgufValue::U32(n) => Some(*n as f64),
        GgufValue::I32(n) => Some(*n as f64),
        GgufValue::U16(n) => Some(*n as f64),
        GgufValue::I16(n) => Some(*n as f64),
        GgufValue::U8(n) => Some(*n as f64),
        GgufValue::I8(n) => Some(*n as f64),
        _ => None,
    }
}

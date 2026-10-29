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
    /// LFM3's additions to the hybrid block. `None` for LFM2 and LFM2.5, whose
    /// loading and forward passes stay exactly as they were.
    pub lfm3: Option<Lfm3TrunkFeatures>,
}

/// What an LFM3 trunk adds to the LFM2 hybrid block.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Lfm3TrunkFeatures {
    /// Per model layer: the window of a sliding-window attention layer, `None`
    /// for global attention and ShortConv layers. Empty when every attention
    /// layer follows `attention_sliding_window`.
    pub layer_sliding_windows: Vec<Option<usize>>,
    /// RoPE base of the sliding-window layers (`lfm3.rope.freq_base_swa`).
    /// `None` when they share `rope_freq_base`.
    pub rope_freq_base_swa: Option<f64>,
    /// Divisors of the global layers' RoPE frequencies (`rope_freqs.weight`).
    /// LFM3 rotates only the highest-frequency half of each head: the other
    /// half carries 1e30, which zeroes its angles.
    pub rope_freq_factors: Option<Vec<f64>>,
    /// Each attention head's output is scaled by `sigmoid(gate_proj(x))`
    /// before the output projection.
    pub attention_output_gate: bool,
    /// An RMSNorm after each operator and after each FFN, before the residual
    /// add (`post_attention_norm`, `post_ffw_norm`).
    pub sandwich_norm: bool,
}

impl Lfm2BackboneConfig {
    /// The sliding window of model layer `layer` if it is an attention layer
    /// that attends locally.
    pub fn layer_sliding_window(&self, layer: usize) -> Option<usize> {
        match &self.lfm3 {
            Some(features) if !features.layer_sliding_windows.is_empty() => {
                features.layer_sliding_windows.get(layer).copied().flatten()
            }
            _ => self.attention_sliding_window,
        }
    }

    /// Whether model layer `layer` takes the sliding-window RoPE base.
    pub fn layer_uses_swa_rope(&self, layer: usize) -> bool {
        self.lfm3
            .as_ref()
            .is_some_and(|features| features.rope_freq_base_swa.is_some())
            && self.layer_sliding_window(layer).is_some()
    }
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
        lfm3: None,
    })
}

pub(crate) fn required_usize(loader: &GgufLoader, key: &str) -> Result<usize> {
    optional_usize(loader, key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

pub(crate) fn optional_usize(loader: &GgufLoader, key: &str) -> Option<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
}

pub(crate) fn required_f64(loader: &GgufLoader, key: &str) -> Result<f64> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

pub(crate) fn required_usize_or_array(
    loader: &GgufLoader,
    key: &str,
    len: usize,
) -> Result<Vec<usize>> {
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

pub(crate) fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
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

pub(crate) fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
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

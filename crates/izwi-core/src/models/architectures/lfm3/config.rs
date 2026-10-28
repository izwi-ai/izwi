//! LFM3 trunk geometry from `lfm3.*` GGUF metadata.

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, Device};

use crate::error::{Error, Result};
use crate::models::architectures::lfm2::config::{
    optional_usize, required_f64, required_usize, required_usize_or_array, Lfm2BackboneConfig,
    Lfm3TrunkFeatures,
};
use crate::models::shared::weights::gguf::GgufLoader;

/// `rope_freqs.weight` marks each unrotated frequency of a global layer with
/// a factor of 1e30 (llama.cpp `conversion/lfm3.py`); a rotated one carries 1.
const UNROTATED_FACTOR_MIN: f64 = 1e29;

/// The LFM3 trunk of an `lfm3` GGUF (LFM3-1.7B, the LFM3-Audio-2B language
/// model) as an LFM2 backbone configuration with its LFM3 features.
pub(crate) fn parse_lfm3_backbone_config(loader: &GgufLoader) -> Result<Lfm2BackboneConfig> {
    let architecture = loader
        .get_metadata_string("general.architecture")
        .unwrap_or_default();
    if architecture != "lfm3" {
        return Err(Error::ModelLoadError(format!(
            "expected an lfm3 GGUF, found architecture {architecture:?}"
        )));
    }
    let block_count = required_usize(loader, "lfm3.block_count")?;
    let attention_head_count = required_usize(loader, "lfm3.attention.head_count")?;
    let attention_head_count_kv =
        required_usize_or_array(loader, "lfm3.attention.head_count_kv", block_count)?;
    let embedding_length = required_usize(loader, "lfm3.embedding_length")?;
    if attention_head_count == 0 || !embedding_length.is_multiple_of(attention_head_count) {
        return Err(Error::ModelLoadError(
            "LFM3 embedding width is not a multiple of its head count".into(),
        ));
    }
    let head_dim = embedding_length / attention_head_count;
    for key in ["lfm3.attention.key_length", "lfm3.rope.dimension_count"] {
        if let Some(value) = optional_usize(loader, key) {
            if value != head_dim {
                return Err(Error::ModelLoadError(format!(
                    "LFM3 {key} is {value}, but the head dimension is {head_dim}"
                )));
            }
        }
    }
    let window = required_usize(loader, "lfm3.attention.sliding_window")?;
    let pattern = bool_array(loader, "lfm3.attention.sliding_window_pattern", block_count)?;
    let layer_sliding_windows = (0..block_count)
        .map(|layer| (attention_head_count_kv[layer] > 0 && pattern[layer]).then_some(window))
        .collect::<Vec<_>>();
    if !layer_sliding_windows.iter().any(Option::is_some) {
        return Err(Error::ModelLoadError(
            "LFM3 requires sliding-window attention layers".into(),
        ));
    }
    let attention_layers = (0..block_count)
        .filter(|layer| attention_head_count_kv[*layer] > 0)
        .collect::<Vec<_>>();
    let attention_output_gate = all_or_none(
        loader,
        &attention_layers,
        |layer| format!("blk.{layer}.attn_gate.weight"),
        "attention output gates",
    )?;
    let all_layers = (0..block_count).collect::<Vec<_>>();
    let sandwich_norm = all_or_none(
        loader,
        &all_layers,
        |layer| format!("blk.{layer}.post_attention_norm.weight"),
        "post-operator norms",
    )? && all_or_none(
        loader,
        &all_layers,
        |layer| format!("blk.{layer}.post_ffw_norm.weight"),
        "post-FFN norms",
    )?;
    Ok(Lfm2BackboneConfig {
        architecture,
        block_count,
        context_length: required_usize(loader, "lfm3.context_length")?,
        embedding_length,
        embedding_length_out: None,
        feed_forward_length: optional_usize(loader, "lfm3.feed_forward_length"),
        attention_head_count,
        attention_head_count_kv,
        attention_layer_norm_rms_epsilon: required_f64(
            loader,
            "lfm3.attention.layer_norm_rms_epsilon",
        )?,
        attention_sliding_window: None,
        rope_freq_base: required_f64(loader, "lfm3.rope.freq_base")?,
        shortconv_l_cache: required_usize(loader, "lfm3.shortconv.l_cache")?,
        lfm3: Some(Lfm3TrunkFeatures {
            layer_sliding_windows,
            rope_freq_base_swa: Some(required_f64(loader, "lfm3.rope.freq_base_swa")?),
            rope_freq_factors: rope_freq_factors(loader, head_dim)?,
            attention_output_gate,
            sandwich_norm,
        }),
    })
}

/// The global layers' RoPE frequency factors, each 1 (rotated) or 1e30
/// (unrotated). Anything else is a scaling this loader does not implement.
fn rope_freq_factors(loader: &GgufLoader, head_dim: usize) -> Result<Option<Vec<f64>>> {
    const NAME: &str = "rope_freqs.weight";
    if !loader.has_tensor(NAME) {
        return Ok(None);
    }
    let factors = loader
        .load_tensor(NAME, DType::F32, &Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    if factors.len() != head_dim / 2 {
        return Err(Error::ModelLoadError(format!(
            "LFM3 {NAME} has {} factors for {} frequencies",
            factors.len(),
            head_dim / 2
        )));
    }
    if factors
        .iter()
        .any(|factor| *factor != 1.0 && *factor < UNROTATED_FACTOR_MIN)
    {
        return Err(Error::ModelLoadError(format!(
            "LFM3 {NAME} holds a RoPE scaling other than partial rotation"
        )));
    }
    Ok(Some(factors))
}

fn bool_array(loader: &GgufLoader, key: &str, len: usize) -> Result<Vec<bool>> {
    let Some(GgufValue::Array(items)) = loader.metadata_value(key) else {
        return Err(Error::ModelLoadError(format!(
            "Missing or invalid GGUF metadata: {key}"
        )));
    };
    let values = items
        .iter()
        .map(|item| match item {
            GgufValue::Bool(value) => Ok(*value),
            _ => Err(Error::ModelLoadError(format!(
                "Expected boolean array values for {key}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != len {
        return Err(Error::ModelLoadError(format!(
            "{key} has {} entries for {len} layers",
            values.len()
        )));
    }
    Ok(values)
}

/// Whether every layer in `layers` has the tensor `name(layer)`; a partial
/// set is a malformed checkpoint.
fn all_or_none(
    loader: &GgufLoader,
    layers: &[usize],
    name: impl Fn(usize) -> String,
    what: &str,
) -> Result<bool> {
    let present = layers
        .iter()
        .filter(|layer| loader.has_tensor(&name(**layer)))
        .count();
    match present {
        0 => Ok(false),
        count if count == layers.len() => Ok(true),
        count => Err(Error::ModelLoadError(format!(
            "LFM3 GGUF has {what} on {count} of {} layers",
            layers.len()
        ))),
    }
}

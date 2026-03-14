use candle_core::quantized::gguf_file::Value as GgufValue;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;

pub const LFM25_AUDIO_CODEBOOKS: usize = 8;
pub const LFM25_AUDIO_TEXT_TO_AUDIO_TOKEN: &str = "<|audio_start|>";
pub const LFM25_AUDIO_TEXT_END_TOKEN: &str = "<|text_end|>";
pub const LFM25_AUDIO_INPUT_SAMPLE_RATE: u32 = 16_000;
pub const LFM25_AUDIO_OUTPUT_SAMPLE_RATE: u32 = 24_000;
pub const LFM25_AUDIO_INPUT_N_FFT: usize = 512;
pub const LFM25_AUDIO_INPUT_WIN_LENGTH: usize = 400;
pub const LFM25_AUDIO_INPUT_HOP_LENGTH: usize = 160;
pub const LFM25_AUDIO_OUTPUT_N_FFT: usize = 1_280;
pub const LFM25_AUDIO_OUTPUT_HOP_LENGTH: usize = 320;
pub const LFM25_AUDIO_ENCODER_SUBSAMPLING_FACTOR: usize = 8;
pub const LFM25_AUDIO_ENCODER_SUBSAMPLING_CHANNELS: usize = 256;
pub const LFM25_AUDIO_DETOKENIZER_UPSAMPLE: usize = 6;
pub const LFM25_AUDIO_AUDIO_VOCAB_SIZE: usize = 2_049;
pub const LFM25_AUDIO_AUDIO_END_TOKEN_ID: u32 = 2_048;
pub const LFM25_AUDIO_INTERLEAVED_TEXT_TOKENS: usize = 6;
pub const LFM25_AUDIO_INTERLEAVED_AUDIO_TOKENS: usize = 12;

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

#[derive(Debug, Clone)]
pub struct Lfm25AudioEncoderConfig {
    pub projector_type: Option<String>,
    pub num_mel_bins: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub block_count: usize,
    pub projection_dim: usize,
    pub attention_head_count: usize,
    pub attention_layer_norm_epsilon: f64,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
}

#[derive(Debug, Clone)]
pub struct Lfm25AudioDecoderConfig {
    pub codebooks: usize,
    pub audio_vocab_size: usize,
    pub audio_end_token_id: u32,
    pub depthformer_layers: usize,
    pub depthformer_dim: usize,
    pub output_sample_rate: u32,
    pub output_n_fft: usize,
    pub output_hop_length: usize,
    pub detokenizer_upsample_factor: usize,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
}

pub fn parse_main_backbone_config(loader: &GgufLoader) -> Result<Lfm2BackboneConfig> {
    parse_lfm2_backbone_config(loader)
}

pub fn parse_detokenizer_config(loader: &GgufLoader) -> Result<Lfm2BackboneConfig> {
    parse_lfm2_backbone_config(loader)
}

pub fn parse_audio_encoder_config(loader: &GgufLoader) -> Result<Lfm25AudioEncoderConfig> {
    let embedding_length = required_usize(loader, "clip.audio.embedding_length")?;
    let metadata_feed_forward_length = required_usize(loader, "clip.audio.feed_forward_length")?;
    let feed_forward_length = infer_audio_encoder_feed_forward_length(loader, embedding_length)?
        .unwrap_or(metadata_feed_forward_length);
    if feed_forward_length != metadata_feed_forward_length {
        tracing::warn!(
            metadata_feed_forward_length,
            feed_forward_length,
            "LFM2.5 Audio GGUF metadata feed-forward length disagrees with encoder tensors; using inferred tensor width"
        );
    }

    Ok(Lfm25AudioEncoderConfig {
        projector_type: loader.get_metadata_string("clip.projector_type"),
        num_mel_bins: required_usize(loader, "clip.audio.num_mel_bins")?,
        embedding_length,
        feed_forward_length,
        block_count: required_usize(loader, "clip.audio.block_count")?,
        projection_dim: required_usize(loader, "clip.audio.projection_dim")?,
        attention_head_count: required_usize(loader, "clip.audio.attention.head_count")?,
        attention_layer_norm_epsilon: required_f64(
            loader,
            "clip.audio.attention.layer_norm_epsilon",
        )?,
        subsampling_factor: LFM25_AUDIO_ENCODER_SUBSAMPLING_FACTOR,
        subsampling_channels: LFM25_AUDIO_ENCODER_SUBSAMPLING_CHANNELS,
    })
}

fn infer_audio_encoder_feed_forward_length(
    loader: &GgufLoader,
    embedding_length: usize,
) -> Result<Option<usize>> {
    for name in [
        "a.blk.0.ffn_up.weight",
        "audio_encoder.layers.0.ff1.linear1.weight",
    ] {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        if shape.len() != 2 {
            return Err(Error::ModelLoadError(format!(
                "Audio encoder FFN tensor {name} must be rank 2, found shape {shape:?}"
            )));
        }
        let dims = (shape[0], shape[1]);
        if dims.1 == embedding_length {
            return Ok(Some(dims.0));
        }
        if dims.0 == embedding_length {
            return Ok(Some(dims.1));
        }
        return Err(Error::ModelLoadError(format!(
            "Audio encoder FFN tensor {name} has shape {dims:?}, but neither dimension matches embedding length {embedding_length}"
        )));
    }
    Ok(None)
}

pub fn parse_audio_decoder_config(loader: &GgufLoader) -> Result<Lfm25AudioDecoderConfig> {
    Ok(Lfm25AudioDecoderConfig {
        codebooks: LFM25_AUDIO_CODEBOOKS,
        audio_vocab_size: LFM25_AUDIO_AUDIO_VOCAB_SIZE,
        audio_end_token_id: LFM25_AUDIO_AUDIO_END_TOKEN_ID,
        depthformer_layers: required_usize(loader, "depthformer_n_layer")?,
        depthformer_dim: required_usize(loader, "depthformer_n_embd")?,
        output_sample_rate: LFM25_AUDIO_OUTPUT_SAMPLE_RATE,
        output_n_fft: LFM25_AUDIO_OUTPUT_N_FFT,
        output_hop_length: LFM25_AUDIO_OUTPUT_HOP_LENGTH,
        detokenizer_upsample_factor: LFM25_AUDIO_DETOKENIZER_UPSAMPLE,
        interleaved_n_text: LFM25_AUDIO_INTERLEAVED_TEXT_TOKENS,
        interleaved_n_audio: LFM25_AUDIO_INTERLEAVED_AUDIO_TOKENS,
    })
}

fn parse_lfm2_backbone_config(loader: &GgufLoader) -> Result<Lfm2BackboneConfig> {
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

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    use candle_core::quantized::gguf_file::{write as write_gguf, Value as GgufValue};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};
    use uuid::Uuid;

    use super::parse_audio_encoder_config;
    use crate::models::shared::weights::gguf::GgufLoader;

    fn make_temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("izwi-lfm25-audio-config-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_test_gguf(
        path: &Path,
        metadata: Vec<(&'static str, GgufValue)>,
        tensors: Vec<(&'static str, Tensor)>,
    ) {
        let quantized = tensors
            .into_iter()
            .map(|(name, tensor)| {
                Ok::<_, candle_core::Error>((name, QTensor::quantize(&tensor, GgmlDType::F32)?))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("quantize tensors");
        let metadata_refs = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let tensor_refs = quantized
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect::<Vec<_>>();
        let mut file = File::create(path).expect("create gguf");
        write_gguf(&mut file, &metadata_refs, &tensor_refs).expect("write gguf");
    }

    #[test]
    fn parse_audio_encoder_config_prefers_ffn_width_inferred_from_tensors() {
        let dir = make_temp_dir();
        let path = dir.join("mmproj.gguf");
        write_test_gguf(
            &path,
            vec![
                ("clip.audio.num_mel_bins", GgufValue::U32(128)),
                ("clip.audio.embedding_length", GgufValue::U32(512)),
                ("clip.audio.feed_forward_length", GgufValue::U32(512)),
                ("clip.audio.block_count", GgufValue::U32(1)),
                ("clip.audio.projection_dim", GgufValue::U32(2048)),
                ("clip.audio.attention.head_count", GgufValue::U32(8)),
                (
                    "clip.audio.attention.layer_norm_epsilon",
                    GgufValue::F32(1e-5),
                ),
            ],
            vec![(
                "a.blk.0.ffn_up.weight",
                Tensor::zeros((2048, 512), candle_core::DType::F32, &Device::Cpu)
                    .expect("ffn tensor"),
            )],
        );

        let loader = GgufLoader::from_path(&path).expect("load gguf");
        let config = parse_audio_encoder_config(&loader).expect("parse config");
        assert_eq!(config.embedding_length, 512);
        assert_eq!(config.feed_forward_length, 2048);

        let _ = fs::remove_dir_all(&dir);
    }
}

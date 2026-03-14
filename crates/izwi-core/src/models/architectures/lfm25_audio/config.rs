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
    Ok(Lfm25AudioEncoderConfig {
        projector_type: loader.get_metadata_string("clip.projector_type"),
        num_mel_bins: required_usize(loader, "clip.audio.num_mel_bins")?,
        embedding_length: required_usize(loader, "clip.audio.embedding_length")?,
        feed_forward_length: required_usize(loader, "clip.audio.feed_forward_length")?,
        block_count: required_usize(loader, "clip.audio.block_count")?,
        projection_dim: required_usize(loader, "clip.audio.projection_dim")?,
        attention_head_count: required_usize(loader, "clip.audio.attention.head_count")?,
        attention_layer_norm_epsilon: required_f64(
            loader,
            "clip.audio.attention.layer_norm_epsilon",
        )?,
        subsampling_factor: LFM25_AUDIO_ENCODER_SUBSAMPLING_FACTOR,
    })
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

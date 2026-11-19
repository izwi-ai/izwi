//! Audio8 TTS (`arktts`) Hugging Face config parsing.
//!
//! The published `config.json` is flat: one object carries the slow
//! transformer, the fast transformer, the token bands and the codec's post
//! module. It is parsed into the slow (`text_config`) and fast
//! (`audio_decoder_config`) halves the rest of the family reads.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer};

use crate::error::{Error, Result};

pub const AUDIO8_MODEL_TYPE: &str = "arktts";
pub const AUDIO8_ARCHITECTURE: &str = "ArkttsModel";

#[derive(Debug, Clone, PartialEq)]
pub struct Audio8Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub torch_dtype: Option<String>,
    pub text_config: Audio8TextConfig,
    pub audio_decoder_config: Audio8AudioDecoderConfig,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub max_seq_len: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub sample_rate: Option<u32>,
    /// Samples one model frame decodes to.
    pub codec_frame_size: usize,
    /// The codec's post-quantizer transformer. It is narrower than the
    /// pre-quantizer one, so it has its own geometry.
    pub codec_post: Audio8CodecPostConfig,
    /// Whether the summed codebook embeddings are scaled at semantic
    /// positions. The published checkpoint was trained without scaling.
    pub scale_codebook_embeddings: bool,
    /// Whether the fast transformer reads the normalized slow hidden state.
    pub norm_fastlayer_input: bool,
    pub ras_window_size: usize,
    pub ras_temperature: f32,
    pub ras_top_p: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Audio8CodecPostConfig {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub intermediate_size: usize,
}

impl Audio8CodecPostConfig {
    /// The post module of the published `codec.pth`.
    pub const PUBLISHED: Self = Self {
        layers: 8,
        heads: 16,
        kv_heads: 8,
        intermediate_size: 1216,
    };
}

#[derive(Debug, Deserialize)]
struct Audio8RawConfig {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
    #[serde(default, alias = "torch_dtype")]
    dtype: Option<String>,

    dim: usize,
    n_layer: usize,
    n_head: usize,
    n_local_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    intermediate_size: usize,
    max_seq_len: usize,
    #[serde(default)]
    rope_base: Option<f64>,
    #[serde(default)]
    norm_eps: Option<f64>,
    vocab_size: usize,
    #[serde(default = "default_true")]
    attention_qkv_bias: bool,
    #[serde(default)]
    attention_qk_norm: bool,
    #[serde(default)]
    attention_o_bias: bool,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,

    n_fast_layer: usize,
    #[serde(default)]
    fast_dim: Option<usize>,
    #[serde(default)]
    fast_n_head: Option<usize>,
    #[serde(default)]
    fast_n_local_heads: Option<usize>,
    #[serde(default)]
    fast_head_dim: Option<usize>,
    #[serde(default)]
    fast_intermediate_size: Option<usize>,
    #[serde(default)]
    fast_attention_qkv_bias: bool,
    #[serde(default)]
    fast_attention_qk_norm: bool,
    #[serde(default)]
    fast_attention_o_bias: bool,
    #[serde(default = "default_true")]
    norm_fastlayer_input: bool,
    #[serde(default)]
    scale_codebook_embeddings: bool,

    num_codebooks: usize,
    codebook_size: usize,
    semantic_begin_id: u32,
    semantic_end_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(default)]
    bos_token_id: Option<u32>,

    #[serde(default)]
    codec_sample_rate: Option<u32>,
    #[serde(default = "default_codec_frame_size")]
    codec_frame_size: usize,
    #[serde(default = "default_codec_post_n_layer")]
    codec_post_n_layer: usize,
    #[serde(default = "default_codec_post_n_head")]
    codec_post_n_head: usize,
    #[serde(default = "default_codec_post_n_local_heads")]
    codec_post_n_local_heads: usize,
    #[serde(default = "default_codec_post_intermediate_size")]
    codec_post_intermediate_size: usize,

    #[serde(default = "default_ras_window_size")]
    ras_window_size: usize,
    #[serde(default = "default_ras_temperature")]
    ras_temperature: f32,
    #[serde(default = "default_ras_top_p")]
    ras_top_p: f32,
}

fn default_true() -> bool {
    true
}
fn default_codec_frame_size() -> usize {
    2048
}
fn default_codec_post_n_layer() -> usize {
    Audio8CodecPostConfig::PUBLISHED.layers
}
fn default_codec_post_n_head() -> usize {
    Audio8CodecPostConfig::PUBLISHED.heads
}
fn default_codec_post_n_local_heads() -> usize {
    Audio8CodecPostConfig::PUBLISHED.kv_heads
}
fn default_codec_post_intermediate_size() -> usize {
    Audio8CodecPostConfig::PUBLISHED.intermediate_size
}
fn default_ras_window_size() -> usize {
    10
}
fn default_ras_temperature() -> f32 {
    1.0
}
fn default_ras_top_p() -> f32 {
    0.9
}

/// The slow transformer: one semantic token per audio frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio8TextConfig {
    pub hidden_size: usize,
    pub attention_qkv_bias: bool,
    pub attention_o_bias: bool,
    pub attention_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: Option<f64>,
    pub intermediate_size: Option<usize>,
}

/// The fast transformer: the frame's codebooks, one position each.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio8AudioDecoderConfig {
    pub hidden_size: usize,
    pub attention_qkv_bias: bool,
    pub attention_o_bias: bool,
    pub attention_qk_norm: bool,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_seq_len: Option<usize>,
    pub num_codebooks: Option<usize>,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub rms_norm_eps: Option<f64>,
}

impl<'de> Deserialize<'de> for Audio8Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Audio8RawConfig::deserialize(deserializer)?;
        let text_config = Audio8TextConfig {
            hidden_size: raw.dim,
            attention_qkv_bias: raw.attention_qkv_bias,
            attention_o_bias: raw.attention_o_bias,
            attention_qk_norm: raw.attention_qk_norm,
            tie_word_embeddings: raw.tie_word_embeddings,
            num_hidden_layers: raw.n_layer,
            num_attention_heads: raw.n_head,
            num_key_value_heads: raw.n_local_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            max_seq_len: raw.max_seq_len,
            rope_theta: raw.rope_base,
            rms_norm_eps: raw.norm_eps,
            intermediate_size: Some(raw.intermediate_size),
        };
        let audio_decoder_config = Audio8AudioDecoderConfig {
            hidden_size: raw.fast_dim.unwrap_or(raw.dim),
            attention_qkv_bias: raw.fast_attention_qkv_bias,
            attention_o_bias: raw.fast_attention_o_bias,
            attention_qk_norm: raw.fast_attention_qk_norm,
            num_hidden_layers: raw.n_fast_layer,
            num_attention_heads: raw.fast_n_head.unwrap_or(raw.n_head),
            num_key_value_heads: raw.fast_n_local_heads.unwrap_or(raw.n_local_heads),
            head_dim: raw.fast_head_dim.or(raw.head_dim),
            intermediate_size: Some(raw.fast_intermediate_size.unwrap_or(raw.intermediate_size)),
            max_seq_len: Some(raw.num_codebooks),
            num_codebooks: Some(raw.num_codebooks),
            vocab_size: Some(raw.codebook_size),
            rope_theta: raw.rope_base,
            rms_norm_eps: raw.norm_eps,
        };
        Ok(Self {
            architectures: raw.architectures,
            model_type: raw.model_type,
            torch_dtype: raw.dtype,
            max_seq_len: text_config.max_seq_len,
            text_config,
            audio_decoder_config,
            num_codebooks: raw.num_codebooks,
            codebook_size: raw.codebook_size,
            bos_token_id: raw.bos_token_id.unwrap_or(raw.pad_token_id),
            eos_token_id: raw.eos_token_id,
            pad_token_id: raw.pad_token_id,
            semantic_start_token_id: raw.semantic_begin_id,
            semantic_end_token_id: raw.semantic_end_id,
            sample_rate: raw.codec_sample_rate,
            codec_frame_size: raw.codec_frame_size,
            codec_post: Audio8CodecPostConfig {
                layers: raw.codec_post_n_layer,
                heads: raw.codec_post_n_head,
                kv_heads: raw.codec_post_n_local_heads,
                intermediate_size: raw.codec_post_intermediate_size,
            },
            scale_codebook_embeddings: raw.scale_codebook_embeddings,
            norm_fastlayer_input: raw.norm_fastlayer_input,
            ras_window_size: raw.ras_window_size,
            ras_temperature: raw.ras_temperature,
            ras_top_p: raw.ras_top_p,
        })
    }
}

impl Audio8Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", path.display()))
        })?;
        let config: Self = serde_json::from_str(&raw).map_err(|err| {
            Error::ModelLoadError(format!("Failed to parse {}: {err}", path.display()))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != AUDIO8_MODEL_TYPE {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Audio8 model_type `{}`",
                self.model_type
            )));
        }
        if !self.architectures.is_empty()
            && !self
                .architectures
                .iter()
                .any(|name| name == AUDIO8_ARCHITECTURE)
        {
            return Err(Error::ModelLoadError(format!(
                "Audio8 config must advertise {AUDIO8_ARCHITECTURE}"
            )));
        }
        if self.num_codebooks != 10
            || self.codebook_size != 4096
            || self.sample_rate.is_some_and(|rate| rate != 44_100)
            || self.codec_frame_size != 2048
        {
            return Err(Error::ModelLoadError(
                "Audio8 requires ten 4096-logit codebooks and a 44100 Hz, 2048-sample codec".into(),
            ));
        }
        if self.semantic_end_token_id <= self.semantic_start_token_id {
            return Err(Error::ModelLoadError(
                "Audio8 semantic token range is empty".to_string(),
            ));
        }
        if self.semantic_vocab_size() != self.codebook_size {
            return Err(Error::ModelLoadError(format!(
                "Audio8 semantic token range has {} entries but codebook_size is {}",
                self.semantic_vocab_size(),
                self.codebook_size
            )));
        }
        if self.max_seq_len == 0
            || self.max_seq_len > self.text_config.max_seq_len
            || self
                .audio_decoder_config
                .max_seq_len
                .is_some_and(|n| n < self.num_codebooks)
            || self
                .audio_decoder_config
                .num_codebooks
                .is_some_and(|n| n != self.num_codebooks)
            || self
                .audio_decoder_config
                .vocab_size
                .is_some_and(|n| n != self.codebook_size)
            || self.semantic_end_token_id as usize >= self.text_config.vocab_size
            || [self.eos_token_id, self.pad_token_id]
                .iter()
                .any(|id| *id as usize >= self.text_config.vocab_size)
            || (self.semantic_start_token_id..=self.semantic_end_token_id)
                .contains(&self.eos_token_id)
        {
            return Err(Error::ModelLoadError(
                "Audio8 context, vocabulary or codebook contracts disagree".into(),
            ));
        }
        // What the published checkpoint is, and what this family implements:
        // a biased fused QKV projection on the slow transformer only, no Q/K
        // normalization, a tied output embedding, unscaled codebook
        // embeddings, and a fast transformer the width of the slow one that
        // reads its normalized hidden state.
        if !self.text_config.attention_qkv_bias
            || self.text_config.attention_o_bias
            || self.text_config.attention_qk_norm
            || !self.text_config.tie_word_embeddings
            || self.audio_decoder_config.attention_qkv_bias
            || self.audio_decoder_config.attention_o_bias
            || self.audio_decoder_config.attention_qk_norm
            || self.scale_codebook_embeddings
            || !self.norm_fastlayer_input
            || self.audio_decoder_config.hidden_size != self.text_config.hidden_size
        {
            return Err(Error::ModelLoadError(
                "Unsupported Audio8 bias, normalization, embedding or projection contract".into(),
            ));
        }
        if self.ras_window_size == 0
            || !self.ras_temperature.is_finite()
            || self.ras_temperature <= 0.0
            || !self.ras_top_p.is_finite()
            || self.ras_top_p <= 0.0
            || self.ras_top_p > 1.0
        {
            return Err(Error::ModelLoadError(
                "Audio8 repetition-aware sampling parameters are invalid".into(),
            ));
        }
        self.codec_post.validate()?;
        // Codec memory and workspace are priced before a config is in hand,
        // from the published geometry: a config that names another post
        // module would be priced wrongly, so it is refused here.
        if self.codec_post != Audio8CodecPostConfig::PUBLISHED {
            return Err(Error::ModelLoadError(
                "Audio8 codec post-module geometry differs from the published codec".into(),
            ));
        }
        self.text_config.validate("slow transformer")?;
        self.audio_decoder_config.validate("fast transformer")?;
        Ok(())
    }

    pub fn semantic_vocab_size(&self) -> usize {
        (self.semantic_end_token_id - self.semantic_start_token_id + 1) as usize
    }
}

impl Audio8CodecPostConfig {
    fn validate(&self) -> Result<()> {
        if self.layers == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.intermediate_size == 0
            || !self.heads.is_multiple_of(self.kv_heads)
        {
            return Err(Error::ModelLoadError(
                "Audio8 codec post-module geometry is invalid".into(),
            ));
        }
        Ok(())
    }
}

impl Audio8TextConfig {
    fn validate(&self, label: &str) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
            || self.max_seq_len == 0
        {
            return Err(Error::ModelLoadError(format!(
                "Audio8 {label} dimensions must be non-zero"
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::ModelLoadError(format!(
                "Audio8 {label} attention heads must be divisible by KV heads"
            )));
        }
        validate_attention_geometry(
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim,
            self.rope_theta,
            self.rms_norm_eps,
            self.intermediate_size,
        )?;
        Ok(())
    }
}

impl Audio8AudioDecoderConfig {
    fn validate(&self, label: &str) -> Result<()> {
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
        {
            return Err(Error::ModelLoadError(format!(
                "Audio8 {label} dimensions must be non-zero"
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::ModelLoadError(format!(
                "Audio8 {label} attention heads must be divisible by KV heads"
            )));
        }
        validate_attention_geometry(
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim,
            self.rope_theta,
            self.rms_norm_eps,
            self.intermediate_size,
        )?;
        Ok(())
    }
}

fn validate_attention_geometry(
    dim: usize,
    heads: usize,
    explicit: Option<usize>,
    rope: Option<f64>,
    eps: Option<f64>,
    intermediate: Option<usize>,
) -> Result<()> {
    let head = explicit.unwrap_or(dim / heads);
    if head == 0
        || !head.is_multiple_of(2)
        || (explicit.is_none() && !dim.is_multiple_of(heads))
        || rope.is_some_and(|v| !v.is_finite() || v <= 0.0)
        || eps.is_some_and(|v| !v.is_finite() || v <= 0.0)
        || intermediate == Some(0)
    {
        return Err(Error::ModelLoadError(
            "Audio8 attention geometry, RoPE and normalization must be valid and finite".into(),
        ));
    }
    Ok(())
}

/// `config.json` of `Edge0/Audio8-TTS-Realistic-Air` at revision `6bf7a980`.
#[cfg(test)]
pub(crate) fn current_config_json() -> &'static str {
    r#"{
          "architectures": ["ArkttsModel"],
          "attention_o_bias": false,
          "attention_qk_norm": false,
          "attention_qkv_bias": true,
          "codebook_size": 4096,
          "codec_filename": "codec.pth",
          "codec_frame_size": 2048,
          "codec_post_intermediate_size": 1216,
          "codec_post_n_head": 16,
          "codec_post_n_layer": 8,
          "codec_post_n_local_heads": 8,
          "codec_sample_rate": 44100,
          "dim": 896,
          "dropout": 0.0,
          "dtype": "bfloat16",
          "eos_token_id": 151645,
          "fast_attention_o_bias": false,
          "fast_attention_qk_norm": false,
          "fast_attention_qkv_bias": false,
          "fast_dim": 896,
          "fast_head_dim": 64,
          "fast_intermediate_size": 4864,
          "fast_n_head": 14,
          "fast_n_local_heads": 2,
          "head_dim": 64,
          "hidden_size": 896,
          "initializer_range": 0.02,
          "intermediate_size": 4864,
          "max_position_embeddings": 2048,
          "max_seq_len": 2048,
          "model_type": "arktts",
          "n_fast_layer": 4,
          "n_head": 14,
          "n_layer": 24,
          "n_local_heads": 2,
          "norm_eps": 1e-06,
          "norm_fastlayer_input": true,
          "num_attention_heads": 14,
          "num_codebooks": 10,
          "num_hidden_layers": 24,
          "num_key_value_heads": 2,
          "pad_token_id": 151643,
          "ras_temperature": 1.0,
          "ras_top_p": 0.9,
          "ras_window_size": 10,
          "rms_norm_eps": 1e-06,
          "rope_base": 1000000.0,
          "rope_theta": 1000000.0,
          "scale_codebook_embeddings": false,
          "semantic_begin_id": 151678,
          "semantic_end_id": 155773,
          "transformers_version": "4.57.3",
          "use_gradient_checkpointing": false,
          "vocab_size": 155776,
          "tie_word_embeddings": true
        }"#
}

#[cfg(test)]
pub(crate) fn current_config() -> Audio8Config {
    serde_json::from_str(current_config_json()).expect("current Audio8 config")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_published_flat_config() {
        let config = current_config();
        config.validate().unwrap();
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.num_codebooks, 10);
        assert_eq!(config.codebook_size, 4096);
        assert_eq!(config.semantic_vocab_size(), 4096);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.bos_token_id, config.pad_token_id);
        assert_eq!(config.sample_rate, Some(44_100));
        let slow = &config.text_config;
        assert_eq!(
            (
                slow.hidden_size,
                slow.num_hidden_layers,
                slow.num_attention_heads,
                slow.num_key_value_heads,
                slow.head_dim,
                slow.intermediate_size,
            ),
            (896, 24, 14, 2, Some(64), Some(4864))
        );
        assert!(slow.attention_qkv_bias && slow.tie_word_embeddings && !slow.attention_qk_norm);
        let fast = &config.audio_decoder_config;
        assert_eq!(
            (
                fast.hidden_size,
                fast.num_hidden_layers,
                fast.num_attention_heads,
                fast.num_key_value_heads,
                fast.max_seq_len,
            ),
            (896, 4, 14, 2, Some(10))
        );
        assert!(!fast.attention_qkv_bias);
        assert_eq!(config.codec_post, Audio8CodecPostConfig::PUBLISHED);
        assert!(!config.scale_codebook_embeddings && config.norm_fastlayer_input);
        assert_eq!(config.ras_window_size, 10);
    }

    #[test]
    fn rejects_contracts_the_family_does_not_implement() {
        let base = current_config();
        let edits: [fn(&mut Audio8Config); 17] = [
            |c| c.text_config.attention_qk_norm = true,
            |c| c.text_config.attention_qkv_bias = false,
            |c| c.scale_codebook_embeddings = true,
            |c| c.norm_fastlayer_input = false,
            |c| c.text_config.tie_word_embeddings = false,
            |c| c.audio_decoder_config.attention_qkv_bias = true,
            |c| c.audio_decoder_config.attention_o_bias = true,
            |c| c.audio_decoder_config.hidden_size = 1024,
            |c| c.text_config.head_dim = Some(3),
            |c| c.audio_decoder_config.rms_norm_eps = Some(f64::NAN),
            |c| c.audio_decoder_config.num_codebooks = Some(11),
            |c| c.eos_token_id = c.semantic_start_token_id,
            |c| c.max_seq_len = c.text_config.max_seq_len + 1,
            |c| c.codec_post.kv_heads = 3,
            |c| c.codec_post.intermediate_size = 3072,
            |c| c.codec_frame_size = 1024,
            |c| c.ras_top_p = 0.0,
        ];
        for edit in edits {
            let mut config = base.clone();
            edit(&mut config);
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn rejects_wrong_model_type() {
        let mut config = current_config();
        config.model_type = "fish_qwen3_omni".to_string();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("model_type"));
    }

    #[test]
    fn nested_fish_layout_does_not_parse() {
        let raw = r#"{"model_type":"fish_qwen3_omni","text_config":{"dim":2560},
            "audio_decoder_config":{"dim":2560},"eos_token_id":1,"pad_token_id":2}"#;
        assert!(serde_json::from_str::<Audio8Config>(raw).is_err());
    }
}

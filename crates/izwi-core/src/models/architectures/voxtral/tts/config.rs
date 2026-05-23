//! Configuration parsing for `mistralai/Voxtral-4B-TTS-2603`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3Config;

pub const VOXTRAL_TTS_MODEL_TYPE: &str = "voxtral_tts";
pub const DEFAULT_N_DECODING_STEPS: usize = 7;
pub const DEFAULT_CFG_ALPHA: f32 = 1.2;

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsConfig {
    #[serde(rename = "dim")]
    pub text_dim: usize,
    #[serde(rename = "n_layers")]
    pub text_n_layers: usize,
    pub head_dim: usize,
    #[serde(rename = "hidden_dim")]
    pub text_hidden_dim: usize,
    #[serde(rename = "n_heads")]
    pub text_n_heads: usize,
    #[serde(rename = "n_kv_heads")]
    pub text_n_kv_heads: usize,
    #[serde(default)]
    pub use_biases: bool,
    #[serde(default = "default_true")]
    pub causal: bool,
    pub rope_theta: f64,
    #[serde(rename = "norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub model_parallel: usize,
    #[serde(default)]
    pub tied_embeddings: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub multimodal: VoxtralTtsMultimodalConfig,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub model_max_length: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub model_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsMultimodalConfig {
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(rename = "audio_model_args", alias = "audio_generation_args")]
    pub audio_model_args: VoxtralTtsAudioModelArgs,
    pub audio_tokenizer_args: VoxtralTtsAudioTokenizerArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioModelArgs {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub audio_encoding_args: VoxtralTtsAudioEncodingArgs,
    pub audio_token_id: u32,
    pub begin_audio_token_id: u32,
    #[serde(default)]
    pub input_embedding_concat_type: Option<String>,
    pub acoustic_transformer_args: VoxtralTtsAcousticTransformerArgs,
    #[serde(default)]
    pub p_uncond: Option<f32>,
    #[serde(default)]
    pub text_feature_bugged: bool,
    #[serde(default)]
    pub condition_dropped_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioEncodingArgs {
    #[serde(default)]
    pub codebook_pattern: Option<String>,
    #[serde(default)]
    pub interleave_audio_tokens_per_segment: Option<usize>,
    #[serde(default)]
    pub interleave_text_tokens_per_segment: Option<usize>,
    #[serde(default)]
    pub single_trailing_segment: bool,
    pub num_codebooks: usize,
    pub sampling_rate: usize,
    pub frame_rate: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAcousticTransformerArgs {
    pub input_dim: usize,
    pub dim: usize,
    #[serde(rename = "n_layers")]
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    #[serde(rename = "n_heads")]
    pub n_heads: usize,
    #[serde(rename = "n_kv_heads")]
    pub n_kv_heads: usize,
    #[serde(default)]
    pub use_biases: bool,
    pub rope_theta: f64,
    pub sigma: f32,
    #[serde(default)]
    pub sigma_max: Option<f32>,
    #[serde(default)]
    pub n_decoding_steps: Option<usize>,
    #[serde(default)]
    pub cfg_alpha: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTtsAudioTokenizerArgs {
    pub channels: usize,
    pub sampling_rate: usize,
    pub pretransform_patch_size: usize,
    pub patch_proj_kernel_size: usize,
    pub semantic_codebook_size: usize,
    pub semantic_dim: usize,
    pub acoustic_codebook_size: usize,
    pub acoustic_dim: usize,
    #[serde(default)]
    pub conv_weight_norm: bool,
    #[serde(default)]
    pub causal: bool,
    pub attn_sliding_window_size: usize,
    #[serde(default)]
    pub half_attn_window_upon_downsampling: bool,
    pub dim: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    #[serde(rename = "n_heads")]
    pub n_heads: usize,
    #[serde(rename = "n_kv_heads")]
    pub n_kv_heads: usize,
    pub qk_norm_eps: f64,
    #[serde(default)]
    pub qk_norm: bool,
    #[serde(default)]
    pub use_biases: bool,
    pub norm_eps: f64,
    #[serde(default)]
    pub layer_scale: bool,
    #[serde(default)]
    pub layer_scale_init: Option<f32>,
    pub decoder_transformer_lengths_str: String,
    pub decoder_convs_kernels_str: String,
    pub decoder_convs_strides_str: String,
    pub voice: BTreeMap<String, usize>,
}

impl VoxtralTtsConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("params.json");
        let json = std::fs::read_to_string(&path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {}", path.display(), err))
        })?;
        Self::from_json_str(&json)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.model_type.is_empty() && self.model_type != VOXTRAL_TTS_MODEL_TYPE {
            return Err(Error::ConfigError(format!(
                "Expected Voxtral TTS model_type `{}`, got `{}`",
                VOXTRAL_TTS_MODEL_TYPE, self.model_type
            )));
        }

        let audio = &self.multimodal.audio_model_args;
        let encoding = &audio.audio_encoding_args;
        if encoding.num_codebooks != audio.n_acoustic_codebook + 1 {
            return Err(Error::ConfigError(format!(
                "Voxtral TTS num_codebooks must equal n_acoustic_codebook + 1; got {} and {}",
                encoding.num_codebooks, audio.n_acoustic_codebook
            )));
        }
        if audio.semantic_codebook_size
            != self.multimodal.audio_tokenizer_args.semantic_codebook_size
        {
            return Err(Error::ConfigError(
                "Voxtral TTS semantic codebook size mismatch between audio model and tokenizer"
                    .to_string(),
            ));
        }
        if audio.acoustic_codebook_size
            != self.multimodal.audio_tokenizer_args.acoustic_codebook_size
        {
            return Err(Error::ConfigError(
                "Voxtral TTS acoustic codebook size mismatch between audio model and tokenizer"
                    .to_string(),
            ));
        }
        if encoding.sampling_rate != self.multimodal.audio_tokenizer_args.sampling_rate {
            return Err(Error::ConfigError(
                "Voxtral TTS sampling rate mismatch between audio model and tokenizer".to_string(),
            ));
        }
        if self.multimodal.audio_tokenizer_args.voice.is_empty() {
            return Err(Error::ConfigError(
                "Voxtral TTS config does not define any preset voices".to_string(),
            ));
        }
        Ok(())
    }

    pub fn text_config(&self) -> Qwen3Config {
        Qwen3Config {
            hidden_size: self.text_dim,
            intermediate_size: self.text_hidden_dim,
            num_attention_heads: self.text_n_heads,
            num_hidden_layers: self.text_n_layers,
            num_key_value_heads: self.text_n_kv_heads,
            head_dim: Some(self.head_dim),
            rms_norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            vocab_size: self.vocab_size,
            lm_head_size: None,
            tie_word_embeddings: self.tied_embeddings,
            rope_scaling: None,
            sliding_window: self.sliding_window.filter(|window| *window > 0),
            use_sliding_window: self.sliding_window.unwrap_or(0) > 0,
            ada_rms_norm_t_cond: false,
            ada_rms_norm_t_cond_dim: 0,
        }
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
            .or(self.model_max_length)
            .or(self.max_seq_len)
            .unwrap_or(65_536)
    }

    pub fn bos_token_id(&self) -> u32 {
        self.multimodal.bos_token_id.unwrap_or(1)
    }

    pub fn sample_rate(&self) -> usize {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .sampling_rate
    }

    pub fn frame_rate(&self) -> f32 {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .frame_rate
    }

    pub fn num_codebooks(&self) -> usize {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .num_codebooks
    }

    pub fn n_acoustic_codebooks(&self) -> usize {
        self.multimodal.audio_model_args.n_acoustic_codebook
    }

    pub fn semantic_codebook_size(&self) -> usize {
        self.multimodal.audio_model_args.semantic_codebook_size
    }

    pub fn acoustic_codebook_size(&self) -> usize {
        self.multimodal.audio_model_args.acoustic_codebook_size
    }

    pub fn codec_latent_dim(&self) -> usize {
        self.multimodal.audio_tokenizer_args.semantic_dim
            + self.multimodal.audio_tokenizer_args.acoustic_dim
    }

    pub fn audio_token_id(&self) -> u32 {
        self.multimodal.audio_model_args.audio_token_id
    }

    pub fn begin_audio_token_id(&self) -> u32 {
        self.multimodal.audio_model_args.begin_audio_token_id
    }

    pub fn condition_dropped_token_id(&self) -> Option<u32> {
        self.multimodal.audio_model_args.condition_dropped_token_id
    }

    pub fn n_decoding_steps(&self) -> usize {
        self.multimodal
            .audio_model_args
            .acoustic_transformer_args
            .n_decoding_steps
            .unwrap_or(DEFAULT_N_DECODING_STEPS)
    }

    pub fn cfg_alpha(&self) -> f32 {
        self.multimodal
            .audio_model_args
            .acoustic_transformer_args
            .cfg_alpha
            .unwrap_or(DEFAULT_CFG_ALPHA)
    }

    pub fn voice_names_by_id(&self) -> Vec<String> {
        let mut entries = self
            .multimodal
            .audio_tokenizer_args
            .voice
            .iter()
            .map(|(name, id)| (name.as_str(), *id))
            .collect::<Vec<_>>();
        entries.sort_by_key(|(_, id)| *id);
        entries
            .into_iter()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    pub fn decoder_transformer_lengths(&self) -> Result<Vec<usize>> {
        parse_usize_csv(
            "decoder_transformer_lengths_str",
            &self
                .multimodal
                .audio_tokenizer_args
                .decoder_transformer_lengths_str,
        )
    }

    pub fn decoder_conv_kernels(&self) -> Result<Vec<usize>> {
        parse_usize_csv(
            "decoder_convs_kernels_str",
            &self
                .multimodal
                .audio_tokenizer_args
                .decoder_convs_kernels_str,
        )
    }

    pub fn decoder_conv_strides(&self) -> Result<Vec<usize>> {
        parse_usize_csv(
            "decoder_convs_strides_str",
            &self
                .multimodal
                .audio_tokenizer_args
                .decoder_convs_strides_str,
        )
    }

    pub fn codec_downsample_factor(&self) -> Result<usize> {
        let stride_product =
            self.decoder_conv_strides()?
                .into_iter()
                .try_fold(1usize, |acc, stride| {
                    acc.checked_mul(stride).ok_or_else(|| {
                        Error::ConfigError(
                            "Voxtral TTS decoder stride product overflowed".to_string(),
                        )
                    })
                })?;
        self.multimodal
            .audio_tokenizer_args
            .pretransform_patch_size
            .checked_mul(stride_product)
            .ok_or_else(|| {
                Error::ConfigError("Voxtral TTS codec downsample factor overflowed".to_string())
            })
    }
}

fn parse_usize_csv(name: &str, raw: &str) -> Result<Vec<usize>> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.parse::<usize>().map_err(|err| {
                Error::ConfigError(format!(
                    "Failed to parse Voxtral TTS {name} value `{value}`: {err}"
                ))
            })
        })
        .collect()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
pub(crate) fn fixture_json() -> &'static str {
    r#"{
        "dim": 3072,
        "n_layers": 26,
        "head_dim": 128,
        "hidden_dim": 9216,
        "n_heads": 32,
        "n_kv_heads": 8,
        "use_biases": false,
        "causal": true,
        "rope_theta": 1000000.0,
        "norm_eps": 1e-05,
        "vocab_size": 131072,
        "model_parallel": 1,
        "tied_embeddings": true,
        "multimodal": {
            "bos_token_id": 1,
            "audio_model_args": {
                "semantic_codebook_size": 8192,
                "acoustic_codebook_size": 21,
                "n_acoustic_codebook": 36,
                "audio_encoding_args": {
                    "codebook_pattern": "parallel",
                    "interleave_audio_tokens_per_segment": 8192,
                    "interleave_text_tokens_per_segment": 8192,
                    "single_trailing_segment": false,
                    "num_codebooks": 37,
                    "sampling_rate": 24000,
                    "frame_rate": 12.5
                },
                "audio_token_id": 24,
                "begin_audio_token_id": 25,
                "input_embedding_concat_type": "sum",
                "acoustic_transformer_args": {
                    "input_dim": 3072,
                    "dim": 3072,
                    "n_layers": 3,
                    "head_dim": 128,
                    "hidden_dim": 9216,
                    "n_heads": 32,
                    "n_kv_heads": 8,
                    "use_biases": false,
                    "rope_theta": 10000.0,
                    "sigma": 1e-05,
                    "sigma_max": 1.0
                },
                "p_uncond": 0.0,
                "text_feature_bugged": false,
                "condition_dropped_token_id": 42
            },
            "audio_tokenizer_args": {
                "channels": 1,
                "sampling_rate": 24000,
                "pretransform_patch_size": 240,
                "patch_proj_kernel_size": 7,
                "semantic_codebook_size": 8192,
                "semantic_dim": 256,
                "acoustic_codebook_size": 21,
                "acoustic_dim": 36,
                "conv_weight_norm": true,
                "causal": true,
                "attn_sliding_window_size": 16,
                "half_attn_window_upon_downsampling": true,
                "dim": 1024,
                "hidden_dim": 4096,
                "head_dim": 128,
                "n_heads": 8,
                "n_kv_heads": 8,
                "qk_norm_eps": 1e-06,
                "qk_norm": true,
                "use_biases": false,
                "norm_eps": 0.01,
                "layer_scale": true,
                "layer_scale_init": 0.01,
                "decoder_transformer_lengths_str": "2,2,2,2",
                "decoder_convs_kernels_str": "3,4,4,4",
                "decoder_convs_strides_str": "1,2,2,2",
                "voice": {
                    "casual_female": 0,
                    "casual_male": 1,
                    "cheerful_female": 2,
                    "neutral_female": 3,
                    "neutral_male": 4,
                    "pt_male": 5,
                    "pt_female": 6,
                    "nl_male": 7,
                    "nl_female": 8,
                    "it_male": 9,
                    "it_female": 10,
                    "fr_male": 11,
                    "fr_female": 12,
                    "es_male": 13,
                    "es_female": 14,
                    "de_male": 15,
                    "de_female": 16,
                    "ar_male": 17,
                    "hi_male": 18,
                    "hi_female": 19
                }
            }
        },
        "max_seq_len": 65536,
        "model_type": "voxtral_tts",
        "max_position_embeddings": 128000
    }"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_params_shape() {
        let cfg = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        assert_eq!(cfg.text_dim, 3072);
        assert_eq!(cfg.num_codebooks(), 37);
        assert_eq!(cfg.n_acoustic_codebooks(), 36);
        assert_eq!(cfg.semantic_codebook_size(), 8192);
        assert_eq!(cfg.acoustic_codebook_size(), 21);
        assert_eq!(cfg.sample_rate(), 24_000);
        assert_eq!(cfg.frame_rate(), 12.5);
        assert_eq!(cfg.codec_latent_dim(), 292);
        assert_eq!(cfg.codec_downsample_factor().unwrap(), 1920);
        assert_eq!(cfg.n_decoding_steps(), DEFAULT_N_DECODING_STEPS);
        assert_eq!(cfg.cfg_alpha(), DEFAULT_CFG_ALPHA);
        assert_eq!(cfg.voice_names_by_id().len(), 20);
    }

    #[test]
    fn converts_text_backbone_to_qwen3_config() {
        let cfg = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let text = cfg.text_config();
        assert_eq!(text.hidden_size, 3072);
        assert_eq!(text.num_hidden_layers, 26);
        assert_eq!(text.num_attention_heads, 32);
        assert_eq!(text.num_key_value_heads, 8);
        assert_eq!(text.head_dim(), 128);
        assert_eq!(text.vocab_size, 131072);
        assert!(text.tie_word_embeddings);
        assert_eq!(cfg.max_position_embeddings(), 128000);
    }

    #[test]
    fn rejects_mismatched_codebook_counts() {
        let bad = fixture_json().replace("\"num_codebooks\": 37", "\"num_codebooks\": 36");
        let err = VoxtralTtsConfig::from_json_str(&bad).unwrap_err();
        assert!(err.to_string().contains("num_codebooks"));
    }
}

//! VibeVoice Hugging Face config parsing.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3Config;

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub decoder_config: Qwen3Config,
    pub acoustic_tokenizer_config: VibeVoiceTokenizerConfig,
    pub semantic_tokenizer_config: VibeVoiceTokenizerConfig,
    #[serde(default)]
    pub diffusion_head_config: Option<VibeVoiceDiffusionHeadConfig>,
    #[serde(default)]
    pub acoustic_vae_dim: Option<usize>,
    #[serde(default)]
    pub semantic_vae_dim: Option<usize>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl VibeVoiceConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path).map_err(|err| {
            Error::ModelLoadError(format!("Failed to read {}: {err}", path.display()))
        })?;
        serde_json::from_str(&raw).map_err(|err| {
            Error::ModelLoadError(format!("Failed to parse {}: {err}", path.display()))
        })
    }

    pub fn acoustic_vae_dim(&self) -> usize {
        self.acoustic_vae_dim
            .unwrap_or(self.acoustic_tokenizer_config.vae_dim)
    }

    pub fn semantic_vae_dim(&self) -> usize {
        self.semantic_vae_dim
            .unwrap_or(self.semantic_tokenizer_config.vae_dim)
    }

    pub fn is_asr(&self) -> bool {
        self.architectures
            .iter()
            .any(|architecture| architecture.to_ascii_lowercase().contains("asr"))
    }

    pub fn is_tts(&self) -> bool {
        if self.is_asr() {
            return false;
        }
        if self.architectures.iter().any(|architecture| {
            let normalized = architecture.to_ascii_lowercase();
            normalized.contains("conditionalgeneration")
                || normalized.contains("tts")
                || normalized.contains("texttospeech")
        }) {
            return true;
        }
        self.diffusion_head_config.is_some()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceTokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub corpus_normalize: f32,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_vae_dim")]
    pub vae_dim: usize,
    #[serde(default)]
    pub fix_std: f32,
    #[serde(default = "default_std_dist_type")]
    pub std_dist_type: String,
    #[serde(default = "default_mixer_layer")]
    pub mixer_layer: String,
    #[serde(default = "default_conv_norm")]
    pub conv_norm: String,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_layernorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_init_value: f32,
    #[serde(default = "default_weight_init")]
    pub weight_init_value: f32,
    #[serde(default = "default_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_depths")]
    pub encoder_depths: DepthSpec,
    #[serde(default = "default_filters")]
    pub decoder_n_filters: usize,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    #[serde(default)]
    pub decoder_depths: Option<DepthSpec>,
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    #[serde(default = "default_kernel_size")]
    pub last_kernel_size: usize,
    #[serde(default = "default_trim_right_ratio")]
    pub trim_right_ratio: f32,
}

impl VibeVoiceTokenizerConfig {
    pub fn encoder_depths_vec(&self) -> Result<Vec<usize>> {
        self.encoder_depths.to_vec()
    }

    pub fn decoder_depths_vec(&self) -> Result<Vec<usize>> {
        match &self.decoder_depths {
            Some(depths) => depths.to_vec(),
            None => {
                let mut depths = self.encoder_depths_vec()?;
                depths.reverse();
                Ok(depths)
            }
        }
    }

    pub fn decoder_ratios_vec(&self) -> Vec<usize> {
        self.decoder_ratios
            .clone()
            .unwrap_or_else(|| self.encoder_ratios.clone())
    }

    pub fn hop_length(&self) -> usize {
        self.encoder_ratios.iter().product::<usize>().max(1)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DepthSpec {
    String(String),
    Vec(Vec<usize>),
}

impl DepthSpec {
    pub fn to_vec(&self) -> Result<Vec<usize>> {
        match self {
            Self::Vec(values) => Ok(values.clone()),
            Self::String(raw) => raw
                .split('-')
                .map(|part| {
                    part.trim().parse::<usize>().map_err(|err| {
                        Error::ModelLoadError(format!(
                            "Invalid VibeVoice depth value '{part}' in '{raw}': {err}"
                        ))
                    })
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceDiffusionHeadConfig {
    #[serde(default = "default_diffusion_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_head_ffn_ratio")]
    pub head_ffn_ratio: f32,
    #[serde(default = "default_layernorm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vae_dim")]
    pub latent_size: usize,
    #[serde(default)]
    pub speech_vae_dim: Option<usize>,
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,
    #[serde(default = "default_diffusion_type")]
    pub diffusion_type: String,
    #[serde(default = "default_ddpm_num_steps")]
    pub ddpm_num_steps: usize,
    #[serde(default = "default_ddpm_inference_steps")]
    pub ddpm_num_inference_steps: usize,
    #[serde(default = "default_beta_schedule")]
    pub ddpm_beta_schedule: String,
    #[serde(default = "default_ddpm_batch_mul")]
    pub ddpm_batch_mul: usize,
}

fn default_channels() -> usize {
    1
}
fn default_true() -> bool {
    true
}
fn default_vae_dim() -> usize {
    64
}
fn default_std_dist_type() -> String {
    "fix".to_string()
}
fn default_mixer_layer() -> String {
    "depthwise_conv".to_string()
}
fn default_conv_norm() -> String {
    "none".to_string()
}
fn default_pad_mode() -> String {
    "constant".to_string()
}
fn default_layernorm() -> String {
    "RMSNorm".to_string()
}
fn default_layernorm_eps() -> f64 {
    1e-5
}
fn default_layer_scale() -> f32 {
    1e-6
}
fn default_weight_init() -> f32 {
    1e-2
}
fn default_filters() -> usize {
    32
}
fn default_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}
fn default_depths() -> DepthSpec {
    DepthSpec::String("3-3-3-3-3-3-8".to_string())
}
fn default_kernel_size() -> usize {
    7
}
fn default_trim_right_ratio() -> f32 {
    1.0
}
fn default_diffusion_hidden() -> usize {
    768
}
fn default_head_layers() -> usize {
    4
}
fn default_head_ffn_ratio() -> f32 {
    3.0
}
fn default_prediction_type() -> String {
    "v_prediction".to_string()
}
fn default_diffusion_type() -> String {
    "ddpm".to_string()
}
fn default_ddpm_num_steps() -> usize {
    1000
}
fn default_ddpm_inference_steps() -> usize {
    20
}
fn default_beta_schedule() -> String {
    "cosine".to_string()
}
fn default_ddpm_batch_mul() -> usize {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vibevoice_tts_config_shape_contract() {
        let raw = r#"{
            "architectures": ["VibeVoiceForConditionalGeneration"],
            "decoder_config": {
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_attention_heads": 12,
                "num_hidden_layers": 28,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "vocab_size": 151936,
                "tie_word_embeddings": true
            },
            "acoustic_tokenizer_config": {
                "vae_dim": 64,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "decoder_ratios": [8, 5, 5, 4, 2, 2],
                "fix_std": 0.5
            },
            "semantic_tokenizer_config": {
                "vae_dim": 128,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "fix_std": 0.0,
                "std_dist_type": "none"
            },
            "diffusion_head_config": {
                "hidden_size": 1536,
                "head_layers": 4,
                "latent_size": 64,
                "ddpm_num_inference_steps": 20
            }
        }"#;
        let cfg: VibeVoiceConfig = serde_json::from_str(raw).unwrap();
        assert!(cfg.is_tts());
        assert!(!cfg.is_asr());
        assert_eq!(cfg.acoustic_vae_dim(), 64);
        assert_eq!(cfg.semantic_vae_dim(), 128);
        assert_eq!(cfg.acoustic_tokenizer_config.hop_length(), 3200);
        assert_eq!(
            cfg.acoustic_tokenizer_config.encoder_depths_vec().unwrap(),
            vec![3, 3, 3, 3, 3, 3, 8]
        );
        assert_eq!(
            cfg.acoustic_tokenizer_config.decoder_depths_vec().unwrap(),
            vec![8, 3, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn parses_vibevoice_asr_config_with_diffusion_head_as_asr() {
        let raw = r#"{
            "architectures": ["VibeVoiceForASRTraining"],
            "decoder_config": {
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "vocab_size": 152064,
                "tie_word_embeddings": true
            },
            "acoustic_tokenizer_config": {
                "vae_dim": 64,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "decoder_ratios": [8, 5, 5, 4, 2, 2],
                "fix_std": 0.5
            },
            "semantic_tokenizer_config": {
                "vae_dim": 128,
                "encoder_ratios": [8, 5, 5, 4, 2, 2],
                "fix_std": 0.0,
                "std_dist_type": "none"
            },
            "diffusion_head_config": {
                "hidden_size": 3584,
                "head_layers": 4,
                "latent_size": 64,
                "ddpm_num_inference_steps": 20
            }
        }"#;
        let cfg: VibeVoiceConfig = serde_json::from_str(raw).unwrap();
        assert!(cfg.is_asr());
        assert!(!cfg.is_tts());
        assert!(cfg.diffusion_head_config.is_some());
        assert_eq!(cfg.decoder_config.hidden_size, 3584);
        assert_eq!(cfg.decoder_config.vocab_size, 152064);
    }
}

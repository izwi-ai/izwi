//! Mimi's configuration, as Kyutai publishes it for transformers'
//! `MimiModel` (`kyutai/mimi` `config.json`).

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct MimiConfig {
    pub audio_channels: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub compress: usize,
    pub dilation_growth_rate: usize,
    /// Codec frames per second (12.5).
    pub frame_rate: f64,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_filters: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    /// Codebooks of the split RVQ; the published config omits it and
    /// transformers defaults it to 32.
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
    pub num_residual_layers: usize,
    pub num_semantic_quantizers: usize,
    pub pad_mode: String,
    pub residual_kernel_size: usize,
    pub rope_theta: f64,
    pub sampling_rate: usize,
    /// Keys a transformer query sees, itself included.
    pub sliding_window: usize,
    pub trim_right_ratio: f64,
    pub upsample_groups: usize,
    pub upsampling_ratios: Vec<usize>,
    pub use_causal_conv: bool,
    pub use_conv_shortcut: bool,
    pub vector_quantization_hidden_dimension: usize,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_num_quantizers() -> usize {
    32
}

impl MimiConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Mimi config {} is unreadable: {error}",
                path.display()
            ))
        })?;
        let cfg: Self = serde_json::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Refuse a configuration this port does not implement: it streams, so
    /// every convolution must be causal with constant padding, the
    /// transposed convolutions must trim all their padding on the right,
    /// and the transformer must be plain multi-head attention.
    pub fn validate(&self) -> Result<()> {
        let unsupported = |what: &str| {
            Err(Error::ModelLoadError(format!(
                "Mimi config is not supported: {what}"
            )))
        };
        if !self.use_causal_conv || self.pad_mode != "constant" {
            return unsupported("convolutions must be causal with constant padding");
        }
        if (self.trim_right_ratio - 1.0).abs() > f64::EPSILON {
            return unsupported("trim_right_ratio must be 1");
        }
        if self.use_conv_shortcut || self.num_residual_layers != 1 {
            return unsupported("residual blocks must be single, without conv shortcuts");
        }
        if self.hidden_act != "gelu" || self.attention_bias {
            return unsupported("the transformer must use exact GELU and no attention bias");
        }
        if self.num_key_value_heads != self.num_attention_heads
            || self.num_attention_heads * self.head_dim != self.hidden_size
        {
            return unsupported("the transformer must be multi-head over the hidden size");
        }
        if self.upsample_groups != self.hidden_size {
            return unsupported("the upsample must be depthwise");
        }
        if self.encodec_frame_rate() != 2.0 * self.frame_rate {
            return unsupported("the transformer must run at twice the codec frame rate");
        }
        if self.num_semantic_quantizers == 0 || self.num_quantizers <= self.num_semantic_quantizers
        {
            return unsupported("the split RVQ needs semantic and acoustic codebooks");
        }
        if self.sliding_window == 0 || self.compress == 0 {
            return unsupported("sliding_window and compress must be positive");
        }
        Ok(())
    }

    /// Audio samples per transformer step (960 at 24 kHz).
    pub fn hop_length(&self) -> usize {
        self.upsampling_ratios.iter().product()
    }

    /// Transformer steps per second (25).
    pub fn encodec_frame_rate(&self) -> f64 {
        self.sampling_rate as f64 / self.hop_length() as f64
    }

    /// Audio samples per codec frame (1,920 at 24 kHz and 12.5 Hz).
    pub fn samples_per_frame(&self) -> usize {
        2 * self.hop_length()
    }

    pub fn acoustic_quantizers(&self) -> usize {
        self.num_quantizers - self.num_semantic_quantizers
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `kyutai/mimi` at 89091b3e.
    pub(crate) const PUBLISHED: &str = r#"{
      "architectures": ["MimiModel"], "attention_dropout": 0.0, "audio_channels": 1,
      "codebook_dim": 256, "codebook_size": 2048, "compress": 2, "dilation_growth_rate": 2,
      "frame_rate": 12.5, "head_dim": 64, "hidden_act": "gelu", "hidden_size": 512,
      "initializer_range": 0.02, "intermediate_size": 2048, "kernel_size": 7,
      "last_kernel_size": 3, "layer_scale_initial_scale": 0.01,
      "max_position_embeddings": 8000, "model_type": "mimi", "norm_eps": 1e-05,
      "normalize": false, "num_attention_heads": 8, "num_filters": 64,
      "num_hidden_layers": 8, "num_key_value_heads": 8, "num_residual_layers": 1,
      "num_semantic_quantizers": 1, "pad_mode": "constant", "residual_kernel_size": 3,
      "rope_theta": 10000.0, "sampling_rate": 24000, "sliding_window": 250,
      "torch_dtype": "float32", "transformers_version": "4.45.0.dev0",
      "trim_right_ratio": 1.0, "upsample_groups": 512, "upsampling_ratios": [8, 6, 5, 4],
      "use_cache": false, "use_causal_conv": true, "use_conv_shortcut": false,
      "vector_quantization_hidden_dimension": 256
    }"#;

    #[test]
    fn published_config_parses_with_its_derived_rates() {
        let cfg: MimiConfig = serde_json::from_str(PUBLISHED).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.num_quantizers, 32);
        assert_eq!(cfg.acoustic_quantizers(), 31);
        assert_eq!(cfg.hop_length(), 960);
        assert_eq!(cfg.encodec_frame_rate(), 25.0);
        assert_eq!(cfg.samples_per_frame(), 1920);
    }

    #[test]
    fn non_causal_configs_are_refused() {
        let mut cfg: MimiConfig = serde_json::from_str(PUBLISHED).unwrap();
        cfg.use_causal_conv = false;
        assert!(cfg.validate().is_err());
    }
}

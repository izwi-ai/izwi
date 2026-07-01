//! Source-backed Fish S2 parity contracts.
//!
//! These are the small invariants the native Rust port depends on. They mirror
//! the public Fish Speech S2 implementation without pulling model execution
//! into tests that should stay cheap.

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;

pub const IM_END_TOKEN: &str = "<|im_end|>";
pub const MODALITY_VOICE_TOKEN: &str = "<|voice|>";
pub const SPEAKER_ASSISTANT_TOKEN: &str = "<|speaker:assistant|>";
pub const SEMANTIC_TOKEN_TEMPLATE_PREFIX: &str = "<|semantic:";
pub const SEMANTIC_TOKEN_TEMPLATE_SUFFIX: &str = "|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishS2DacContract {
    pub sample_rate: u32,
    pub encoder_rates: &'static [usize],
    pub decoder_rates: &'static [usize],
    pub semantic_codebooks: usize,
    pub residual_codebooks: usize,
    pub semantic_codebook_size: usize,
    pub residual_codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
}

impl FishS2DacContract {
    pub const CURRENT: Self = Self {
        sample_rate: 44_100,
        encoder_rates: &[2, 4, 8, 8],
        decoder_rates: &[8, 8, 4, 2],
        semantic_codebooks: 1,
        residual_codebooks: 9,
        semantic_codebook_size: 4096,
        residual_codebook_size: 1024,
        codebook_dim: 8,
        latent_dim: 1024,
    };

    pub fn total_codebooks(&self) -> usize {
        self.semantic_codebooks + self.residual_codebooks
    }

    pub fn hop_length(&self) -> Result<usize> {
        checked_product(self.encoder_rates, "Fish S2 DAC encoder rates")
    }

    pub fn frame_length(&self) -> Result<usize> {
        self.hop_length()?
            .checked_mul(4)
            .ok_or_else(|| Error::ConfigError("Fish S2 DAC frame length overflowed".to_string()))
    }

    pub fn frame_rate_hz(&self) -> Result<f32> {
        Ok(self.sample_rate as f32 / self.frame_length()? as f32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishS2PromptTensorShape {
    pub rows: usize,
    pub seq_len: usize,
}

impl FishS2PromptTensorShape {
    pub fn from_config(config: &FishS2Config, seq_len: usize) -> Self {
        Self {
            rows: config.num_codebooks + 1,
            seq_len,
        }
    }

    pub fn validate(&self, config: &FishS2Config) -> Result<()> {
        let expected = config.num_codebooks + 1;
        if self.rows != expected {
            return Err(Error::InvalidInput(format!(
                "Fish S2 prompt tensor must have {expected} rows, got {}",
                self.rows
            )));
        }
        if self.seq_len == 0 {
            return Err(Error::InvalidInput(
                "Fish S2 prompt tensor sequence length cannot be zero".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn remap_fish_qwen3_omni_key(key: &str) -> String {
    if let Some(suffix) = key.strip_prefix("text_model.model.") {
        remap_fish_transformer_suffix(suffix)
    } else if let Some(suffix) = key.strip_prefix("text_model.") {
        remap_fish_transformer_suffix(suffix)
    } else if let Some(suffix) = key.strip_prefix("audio_decoder.") {
        if suffix.starts_with("codebook_embeddings.") {
            suffix.to_string()
        } else if suffix == "embeddings.weight" {
            "fast_embeddings.weight".to_string()
        } else if suffix == "norm.weight" {
            "fast_norm.weight".to_string()
        } else if suffix == "output.weight" {
            "fast_output.weight".to_string()
        } else {
            format!("fast_{}", remap_fish_transformer_suffix(suffix))
        }
    } else {
        key.to_string()
    }
}

fn remap_fish_transformer_suffix(suffix: &str) -> String {
    let suffix = match suffix {
        "embeddings.weight" => return "embed_tokens.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        _ => suffix,
    };

    suffix
        .replace(".attention.wqkv.", ".self_attn.qkv_proj.")
        .replace(".attention.wo.", ".self_attn.o_proj.")
        .replace(".attention.q_norm.", ".self_attn.q_norm.")
        .replace(".attention.k_norm.", ".self_attn.k_norm.")
        .replace(".attention_norm.", ".input_layernorm.")
        .replace(".ffn_norm.", ".post_attention_layernorm.")
        .replace(".feed_forward.w1.", ".mlp.gate_proj.")
        .replace(".feed_forward.w3.", ".mlp.up_proj.")
        .replace(".feed_forward.w2.", ".mlp.down_proj.")
}

pub fn semantic_token_id(config: &FishS2Config, semantic_code: u32) -> Result<u32> {
    let max_code = u32::try_from(config.codebook_size.saturating_sub(1))
        .map_err(|_| Error::ConfigError("Fish S2 codebook size does not fit in u32".to_string()))?;
    if semantic_code > max_code {
        return Err(Error::InvalidInput(format!(
            "Fish S2 semantic code {semantic_code} exceeds max code {max_code}"
        )));
    }
    config
        .semantic_start_token_id
        .checked_add(semantic_code)
        .ok_or_else(|| Error::ConfigError("Fish S2 semantic token id overflowed".to_string()))
}

pub fn semantic_code_from_token_id(config: &FishS2Config, token_id: u32) -> Result<u32> {
    if token_id < config.semantic_start_token_id || token_id > config.semantic_end_token_id {
        return Err(Error::InvalidInput(format!(
            "Fish S2 token id {token_id} is outside semantic range {}..={}",
            config.semantic_start_token_id, config.semantic_end_token_id
        )));
    }
    Ok(token_id - config.semantic_start_token_id)
}

pub fn build_semantic_allowed_mask(
    vocab_size: usize,
    config: &FishS2Config,
    im_end_token_id: u32,
) -> Result<Vec<bool>> {
    if config.semantic_end_token_id as usize >= vocab_size {
        return Err(Error::ConfigError(format!(
            "Fish S2 semantic end id {} exceeds vocab size {vocab_size}",
            config.semantic_end_token_id
        )));
    }
    if im_end_token_id as usize >= vocab_size {
        return Err(Error::ConfigError(format!(
            "Fish S2 im_end id {im_end_token_id} exceeds vocab size {vocab_size}"
        )));
    }

    let mut mask = vec![false; vocab_size];
    for token_id in config.semantic_start_token_id..=config.semantic_end_token_id {
        mask[token_id as usize] = true;
    }
    mask[im_end_token_id as usize] = true;
    Ok(mask)
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| Error::ConfigError(format!("{label} product overflowed")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FishS2Config {
        crate::models::architectures::fish_s2::config::current_config()
    }

    #[test]
    fn remaps_current_hf_weight_names_to_upstream_module_names() {
        assert_eq!(
            remap_fish_qwen3_omni_key("text_model.model.layers.0.self_attn.qkv_proj.weight"),
            "layers.0.self_attn.qkv_proj.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("text_model.model.layers.0.attention.wqkv.weight"),
            "layers.0.self_attn.qkv_proj.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("text_model.model.layers.0.feed_forward.w3.weight"),
            "layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("audio_decoder.layers.0.attention.wqkv.weight"),
            "fast_layers.0.self_attn.qkv_proj.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("audio_decoder.codebook_embeddings.weight"),
            "codebook_embeddings.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("audio_decoder.embeddings.weight"),
            "fast_embeddings.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("audio_decoder.output.weight"),
            "fast_output.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("text_model.output.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            remap_fish_qwen3_omni_key("lm_head.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn semantic_token_mapping_is_inclusive_and_contiguous() {
        let config = config();
        assert_eq!(
            semantic_token_id(&config, 0).unwrap(),
            config.semantic_start_token_id
        );
        assert_eq!(
            semantic_token_id(&config, 4095).unwrap(),
            config.semantic_end_token_id
        );
        assert_eq!(
            semantic_code_from_token_id(&config, config.semantic_start_token_id).unwrap(),
            0
        );
        assert_eq!(
            semantic_code_from_token_id(&config, config.semantic_end_token_id).unwrap(),
            4095
        );
        assert!(semantic_token_id(&config, 4096).is_err());
        assert!(semantic_code_from_token_id(&config, config.semantic_start_token_id - 1).is_err());
    }

    #[test]
    fn semantic_allowed_mask_admits_only_semantic_range_and_im_end() {
        let config = config();
        let im_end_id = config.eos_token_id;
        let mask = build_semantic_allowed_mask(config.text_config.vocab_size, &config, im_end_id)
            .expect("mask");
        let allowed = mask.iter().filter(|allowed| **allowed).count();
        assert_eq!(allowed, config.semantic_vocab_size() + 1);
        assert!(mask[config.semantic_start_token_id as usize]);
        assert!(mask[config.semantic_end_token_id as usize]);
        assert!(mask[im_end_id as usize]);
        assert!(!mask[(config.semantic_start_token_id - 1) as usize]);
    }

    #[test]
    fn prompt_tensor_shape_has_main_row_plus_codebook_rows() {
        let config = config();
        let shape = FishS2PromptTensorShape::from_config(&config, 17);
        assert_eq!(shape.rows, 11);
        assert_eq!(shape.seq_len, 17);
        shape.validate(&config).unwrap();
        assert!(FishS2PromptTensorShape {
            rows: 10,
            seq_len: 17
        }
        .validate(&config)
        .is_err());
        assert!(FishS2PromptTensorShape {
            rows: 11,
            seq_len: 0
        }
        .validate(&config)
        .is_err());
    }

    #[test]
    fn dac_contract_matches_current_upstream_s2_codec_config() {
        let contract = FishS2DacContract::CURRENT;
        assert_eq!(contract.sample_rate, 44_100);
        assert_eq!(contract.total_codebooks(), 10);
        assert_eq!(contract.hop_length().unwrap(), 512);
        assert_eq!(contract.frame_length().unwrap(), 2048);
        let frame_rate = contract.frame_rate_hz().unwrap();
        assert!((frame_rate - 21.533).abs() < 0.01);
        assert_eq!(contract.semantic_codebook_size, 4096);
        assert_eq!(contract.residual_codebook_size, 1024);
    }
}

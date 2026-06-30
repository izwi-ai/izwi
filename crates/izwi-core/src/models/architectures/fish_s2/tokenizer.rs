//! Fish S2 tokenizer and ChatML prompt helpers.

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishS2SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub audio_pad: u32,
    pub semantic_start: u32,
    pub semantic_end: u32,
}

impl From<&FishS2Config> for FishS2SpecialTokens {
    fn from(config: &FishS2Config) -> Self {
        Self {
            bos: config.bos_token_id,
            eos: config.eos_token_id,
            pad: config.pad_token_id,
            audio_pad: config.audio_pad_token_id,
            semantic_start: config.semantic_start_token_id,
            semantic_end: config.semantic_end_token_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2Prompt {
    pub rendered: String,
    pub input_ids: Vec<u32>,
}

pub struct FishS2PromptTokenizer {
    tokenizer: Tokenizer,
    specials: FishS2SpecialTokens,
    chat_template: String,
}

impl FishS2PromptTokenizer {
    pub fn load(model_dir: &Path, config: &FishS2Config) -> Result<Self> {
        let tokenizer = Tokenizer::from_path_with_expected_vocab(
            model_dir,
            Some(config.text_config.vocab_size),
        )?;
        let chat_template_path = model_dir.join("chat_template.jinja");
        let chat_template = fs::read_to_string(&chat_template_path).map_err(|err| {
            Error::ModelLoadError(format!(
                "Failed to read Fish S2 chat template {}: {err}",
                chat_template_path.display()
            ))
        })?;
        validate_chat_template(&chat_template)?;
        Ok(Self {
            tokenizer,
            specials: FishS2SpecialTokens::from(config),
            chat_template,
        })
    }

    pub fn specials(&self) -> FishS2SpecialTokens {
        self.specials
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    pub fn build_reference_tts_prompt(
        &self,
        reference_text: &str,
        target_text: &str,
    ) -> Result<FishS2Prompt> {
        let rendered = render_reference_tts_chatml(reference_text, target_text)?;
        let input_ids = self.tokenizer.encode(&rendered)?;
        Ok(FishS2Prompt {
            rendered,
            input_ids,
        })
    }
}

pub fn validate_chat_template(template: &str) -> Result<()> {
    if !template.contains("<|im_start|>") || !template.contains("<|im_end|>") {
        return Err(Error::ModelLoadError(
            "Fish S2 chat_template.jinja must contain ChatML im_start/im_end tokens".to_string(),
        ));
    }
    Ok(())
}

pub fn render_reference_tts_chatml(reference_text: &str, target_text: &str) -> Result<String> {
    let reference_text = reference_text.trim();
    let target_text = target_text.trim();
    if reference_text.is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 reference_text cannot be empty".to_string(),
        ));
    }
    if target_text.is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 target text cannot be empty".to_string(),
        ));
    }

    Ok(format!(
        "<|im_start|>user\nReference text:\n{reference_text}\n\nTarget text:\n{target_text}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_chatml_template_markers() {
        validate_chat_template("<|im_start|>{{ role }}<|im_end|>").unwrap();
        let err = validate_chat_template("plain text").unwrap_err();
        assert!(err.to_string().contains("ChatML"));
    }

    #[test]
    fn renders_reference_tts_chatml_prompt() {
        let prompt = render_reference_tts_chatml("I am the reference.", "Speak this.").unwrap();
        assert!(prompt.starts_with("<|im_start|>user\n"));
        assert!(prompt.contains("Reference text:\nI am the reference."));
        assert!(prompt.contains("Target text:\nSpeak this."));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn rejects_empty_reference_or_target_text() {
        assert!(render_reference_tts_chatml("", "hello").is_err());
        assert!(render_reference_tts_chatml("hello", "").is_err());
    }
}

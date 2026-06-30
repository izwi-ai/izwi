//! Fish S2 tokenizer and ChatML prompt helpers.

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::contracts::{
    semantic_token_id, FishS2DacContract, IM_END_TOKEN, MODALITY_VOICE_TOKEN,
};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2VqCodes {
    /// Codebook-major codes shaped `[num_codebooks][frames]`.
    pub codebooks: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2ConditioningPrompt {
    /// Upstream prompt tensor values shaped `[num_codebooks + 1][seq_len]`.
    pub values: Vec<Vec<u32>>,
    pub vq_mask: Vec<bool>,
    pub prompt_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FishS2PromptPart {
    Text(String),
    Vq(FishS2VqCodes),
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

    pub fn build_reference_voice_prompt(
        &self,
        config: &FishS2Config,
        reference_text: &str,
        reference_codes: FishS2VqCodes,
        target_text: &str,
    ) -> Result<FishS2ConditioningPrompt> {
        let parts = reference_voice_prompt_parts(reference_text, reference_codes, target_text)?;
        encode_prompt_parts_for_inference(&self.tokenizer, config, &parts)
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

fn reference_voice_prompt_parts(
    reference_text: &str,
    reference_codes: FishS2VqCodes,
    target_text: &str,
) -> Result<Vec<FishS2PromptPart>> {
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

    let reference_text = ensure_reference_speaker_tag(reference_text);
    Ok(vec![
        FishS2PromptPart::Text(
            "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n"
                .to_string(),
        ),
        FishS2PromptPart::Text(reference_text),
        FishS2PromptPart::Text("\n\nSpeech:\n".to_string()),
        FishS2PromptPart::Vq(reference_codes),
        FishS2PromptPart::Text(format!("{IM_END_TOKEN}\n")),
        FishS2PromptPart::Text(format!("<|im_start|>user\n{target_text}{IM_END_TOKEN}\n")),
        FishS2PromptPart::Text(format!("<|im_start|>assistant\n{MODALITY_VOICE_TOKEN}")),
    ])
}

fn ensure_reference_speaker_tag(reference_text: &str) -> String {
    if reference_text.contains("<|speaker:") {
        reference_text.to_string()
    } else {
        format!("<|speaker:0|>{reference_text}")
    }
}

fn encode_prompt_parts_for_inference(
    tokenizer: &Tokenizer,
    config: &FishS2Config,
    parts: &[FishS2PromptPart],
) -> Result<FishS2ConditioningPrompt> {
    encode_prompt_parts_for_inference_with(config, parts, |text| tokenizer.encode(text))
}

fn encode_prompt_parts_for_inference_with<F>(
    config: &FishS2Config,
    parts: &[FishS2PromptPart],
    mut encode_text: F,
) -> Result<FishS2ConditioningPrompt>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    let mut row0 = Vec::new();
    let mut vq_mask = Vec::new();
    let mut vq_segments = Vec::new();

    for part in parts {
        match part {
            FishS2PromptPart::Text(text) => {
                let ids = encode_text(text)?;
                let len = ids.len();
                row0.extend(ids);
                vq_mask.extend(std::iter::repeat(false).take(len));
            }
            FishS2PromptPart::Vq(codes) => {
                validate_vq_codes(config, codes)?;
                let frames = codes.frame_count();
                for frame_idx in 0..frames {
                    row0.push(semantic_token_id(config, codes.codebooks[0][frame_idx])?);
                    vq_mask.push(true);
                }
                vq_segments.push(codes.clone());
            }
        }
    }

    if row0.is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 prompt cannot be empty".to_string(),
        ));
    }

    let mut values = vec![vec![0u32; row0.len()]; config.num_codebooks + 1];
    values[0].clone_from(&row0);

    let mut cursor = 0usize;
    for segment in vq_segments {
        while cursor < vq_mask.len() && !vq_mask[cursor] {
            cursor += 1;
        }
        for frame_idx in 0..segment.frame_count() {
            let col = cursor + frame_idx;
            for codebook_idx in 0..config.num_codebooks {
                values[codebook_idx + 1][col] = segment.codebooks[codebook_idx][frame_idx];
            }
        }
        cursor += segment.frame_count();
    }

    Ok(FishS2ConditioningPrompt {
        values,
        vq_mask,
        prompt_length: row0.len(),
    })
}

fn validate_vq_codes(config: &FishS2Config, codes: &FishS2VqCodes) -> Result<()> {
    if codes.codebooks.len() != config.num_codebooks {
        return Err(Error::InvalidInput(format!(
            "Fish S2 VQ codes must contain {} codebooks, got {}",
            config.num_codebooks,
            codes.codebooks.len()
        )));
    }
    let frames = codes.frame_count();
    if frames == 0 {
        return Err(Error::InvalidInput(
            "Fish S2 VQ codes cannot be empty".to_string(),
        ));
    }
    for (idx, codebook) in codes.codebooks.iter().enumerate() {
        if codebook.len() != frames {
            return Err(Error::InvalidInput(format!(
                "Fish S2 VQ codebook {idx} has {} frames, expected {frames}",
                codebook.len()
            )));
        }
        let codebook_size = if idx == 0 {
            config.codebook_size
        } else {
            FishS2DacContract::CURRENT.residual_codebook_size
        };
        if let Some(code) = codebook
            .iter()
            .copied()
            .find(|code| *code as usize >= codebook_size)
        {
            return Err(Error::InvalidInput(format!(
                "Fish S2 VQ code {code} in codebook {idx} exceeds codebook size {codebook_size}"
            )));
        }
    }
    Ok(())
}

impl FishS2VqCodes {
    pub fn frame_count(&self) -> usize {
        self.codebooks.first().map(Vec::len).unwrap_or(0)
    }
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

    #[test]
    fn reference_voice_prompt_parts_match_upstream_message_structure() {
        let codes = FishS2VqCodes {
            codebooks: vec![vec![1, 2]; 10],
        };
        let parts = reference_voice_prompt_parts("hello", codes, "target").unwrap();
        assert!(matches!(
            parts.first().unwrap(),
            FishS2PromptPart::Text(text)
                if text.starts_with("<|im_start|>system\nconvert the provided text")
        ));
        assert!(matches!(
            &parts[1],
            FishS2PromptPart::Text(text) if text == "<|speaker:0|>hello"
        ));
        assert!(matches!(&parts[3], FishS2PromptPart::Vq(_)));
        assert!(matches!(
            parts.last().unwrap(),
            FishS2PromptPart::Text(text) if text == "<|im_start|>assistant\n<|voice|>"
        ));
    }

    #[test]
    fn validates_vq_code_shape_and_ranges() {
        let config = crate::models::architectures::fish_s2::config::current_config();
        let valid = FishS2VqCodes {
            codebooks: vec![vec![0, 1, 4095]]
                .into_iter()
                .chain((1..10).map(|_| vec![0, 1, 1023]))
                .collect(),
        };
        validate_vq_codes(&config, &valid).unwrap();

        let wrong_count = FishS2VqCodes {
            codebooks: vec![vec![0, 1]],
        };
        assert!(validate_vq_codes(&config, &wrong_count).is_err());

        let mut out_of_range = valid.clone();
        out_of_range.codebooks[1][0] = 1024;
        assert!(validate_vq_codes(&config, &out_of_range).is_err());
    }

    #[test]
    fn inference_prompt_tensor_places_vq_codes_on_semantic_columns() {
        let config = crate::models::architectures::fish_s2::config::current_config();
        let codes = FishS2VqCodes {
            codebooks: vec![vec![7, 8]]
                .into_iter()
                .chain((1..10).map(|idx| vec![idx, idx + 10]))
                .collect(),
        };
        let parts = vec![
            FishS2PromptPart::Text("a".to_string()),
            FishS2PromptPart::Vq(codes),
            FishS2PromptPart::Text("b".to_string()),
        ];
        let prompt = encode_prompt_parts_for_inference_with(&config, &parts, |text| {
            Ok(match text {
                "a" => vec![11],
                "b" => vec![12],
                _ => vec![99],
            })
        })
        .unwrap();

        assert_eq!(prompt.prompt_length, 4);
        assert_eq!(
            prompt.values[0],
            vec![
                11,
                config.semantic_start_token_id + 7,
                config.semantic_start_token_id + 8,
                12
            ]
        );
        assert_eq!(prompt.vq_mask, vec![false, true, true, false]);
        assert_eq!(prompt.values[1][1], 7);
        assert_eq!(prompt.values[1][2], 8);
        assert_eq!(prompt.values[2][1], 1);
        assert_eq!(prompt.values[2][2], 11);
        assert_eq!(prompt.values[10][1], 9);
        assert_eq!(prompt.values[10][2], 19);
        assert_eq!(prompt.values[1][0], 0);
        assert_eq!(prompt.values[1][3], 0);
    }
}

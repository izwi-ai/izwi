//! Audio8 tokenizer and prompt helpers.
//!
//! The prompt mirrors `ArkttsProcessor`: every literal part is tokenized on
//! its own, caller text has its whitespace collapsed, and the reference
//! frames sit between the transcript and the end of the system turn.

use std::path::Path;

use crate::error::{Error, Result};
use crate::models::architectures::audio8::config::Audio8Config;
use crate::models::architectures::audio8::contracts::{
    semantic_token_id, Audio8DacContract, IM_END_TOKEN, MODALITY_VOICE_TOKEN,
};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Audio8SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub semantic_start: u32,
    pub semantic_end: u32,
}

impl From<&Audio8Config> for Audio8SpecialTokens {
    fn from(config: &Audio8Config) -> Self {
        Self {
            bos: config.bos_token_id,
            eos: config.eos_token_id,
            pad: config.pad_token_id,
            semantic_start: config.semantic_start_token_id,
            semantic_end: config.semantic_end_token_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio8Prompt {
    pub rendered: String,
    pub input_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio8VqCodes {
    /// Codebook-major codes shaped `[num_codebooks][frames]`.
    pub codebooks: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio8ConditioningPrompt {
    /// The reference's prompt tensor values, shaped `[num_codebooks + 1][seq_len]`.
    pub values: Vec<Vec<u32>>,
    pub vq_mask: Vec<bool>,
    pub prompt_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Audio8PromptPart {
    Text(String),
    Vq(Audio8VqCodes),
}

pub struct Audio8PromptTokenizer {
    tokenizer: Tokenizer,
    specials: Audio8SpecialTokens,
}

impl Audio8PromptTokenizer {
    #[cfg(test)]
    pub(super) fn for_batch_test(config: &Audio8Config) -> Self {
        let tokenizer =
            tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        Self {
            tokenizer: Tokenizer::from_hf_json_bytes(
                tokenizer.to_string(false).unwrap().as_bytes(),
            )
            .unwrap(),
            specials: Audio8SpecialTokens::from(config),
        }
    }

    pub fn load(model_dir: &Path, config: &Audio8Config) -> Result<Self> {
        let tokenizer = Tokenizer::from_path_with_expected_vocab(
            model_dir,
            Some(config.text_config.vocab_size),
        )?;
        for (token, expected) in [
            (IM_END_TOKEN, config.eos_token_id),
            ("<|endoftext|>", config.pad_token_id),
        ] {
            if tokenizer.token_to_id(token) != Some(expected) {
                return Err(Error::ModelLoadError(format!(
                    "Audio8 tokenizer {token} must map to {expected}"
                )));
            }
        }
        for token in ["<|im_start|>", MODALITY_VOICE_TOKEN] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::ModelLoadError(format!(
                    "Audio8 tokenizer is missing {token}"
                )));
            }
        }
        for code in 0..config.codebook_size {
            let token = format!("<|semantic:{code}|>");
            let expected = config.semantic_start_token_id + code as u32;
            if tokenizer.token_to_id(&token) != Some(expected) {
                return Err(Error::ModelLoadError(format!(
                    "Audio8 tokenizer {token} must map to {expected}"
                )));
            }
        }
        Ok(Self {
            tokenizer,
            specials: Audio8SpecialTokens::from(config),
        })
    }

    pub fn specials(&self) -> Audio8SpecialTokens {
        self.specials
    }

    /// The prompt for `target_text`, in `reference`'s voice when one is
    /// given and in a voice the model picks otherwise.
    pub fn build_speech_prompt(
        &self,
        config: &Audio8Config,
        reference: Option<(&str, Audio8VqCodes)>,
        target_text: &str,
    ) -> Result<Audio8ConditioningPrompt> {
        self.build_speech_prompt_bounded(config, reference, target_text, config.max_seq_len)
    }

    pub(crate) fn build_speech_prompt_bounded(
        &self,
        config: &Audio8Config,
        reference: Option<(&str, Audio8VqCodes)>,
        target_text: &str,
        context_limit: usize,
    ) -> Result<Audio8ConditioningPrompt> {
        let parts = speech_prompt_parts(reference, target_text)?;
        encode_prompt_parts_bounded_with(config, &parts, context_limit, |text| {
            self.tokenizer.encode(text)
        })
    }

    /// The token ids of the prompt's text, before and after the place the
    /// reference frames go (everything is before it when there is no
    /// reference): what the published processor returns as its prefix and
    /// suffix.
    #[cfg(test)]
    pub(crate) fn speech_prompt_text_ids(
        &self,
        reference_text: Option<&str>,
        target_text: &str,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let parts = speech_prompt_text_parts(reference_text, target_text)?;
        let encode = |parts: &[String]| -> Result<Vec<u32>> {
            let mut ids = Vec::new();
            for part in parts {
                ids.extend(self.tokenizer.encode(part)?);
            }
            Ok(ids)
        };
        Ok((encode(&parts.before)?, encode(&parts.after)?))
    }
}

/// `ArkttsProcessor._clean_text`: trim, and collapse every run of whitespace
/// (newlines included) to one space.
pub(crate) fn clean_prompt_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

const SYSTEM_OPEN: &str = "<|im_start|>system\n";
const USER_OPEN: &str = "<|im_start|>user\n";
const PLAIN_INSTRUCTION: &str = "convert the provided text to speech";
const REFERENCE_INSTRUCTION: &str =
    "convert the provided text to speech reference to the following:\n\nText:\n";
const REFERENCE_SPEECH_HEADER: &str = "\n\nSpeech:\n";

/// The prompt's text, split where the reference frames go. Each entry is
/// tokenized on its own, as the published processor does.
struct Audio8PromptTextParts {
    before: Vec<String>,
    after: Vec<String>,
}

fn speech_prompt_text_parts(
    reference_text: Option<&str>,
    target_text: &str,
) -> Result<Audio8PromptTextParts> {
    let target_text = clean_prompt_text(target_text);
    if target_text.is_empty() {
        return Err(Error::InvalidInput(
            "Audio8 target text cannot be empty".to_string(),
        ));
    }
    let turn_end = format!("{IM_END_TOKEN}\n");
    let assistant = format!("<|im_start|>assistant\n{MODALITY_VOICE_TOKEN}");
    let Some(reference_text) = reference_text else {
        return Ok(Audio8PromptTextParts {
            before: vec![
                SYSTEM_OPEN.to_string(),
                PLAIN_INSTRUCTION.to_string(),
                turn_end.clone(),
                USER_OPEN.to_string(),
                target_text,
                turn_end,
                assistant,
            ],
            after: Vec::new(),
        });
    };
    let reference_text = clean_prompt_text(reference_text);
    if reference_text.is_empty() {
        return Err(Error::InvalidInput(
            "Audio8 reference_text cannot be empty".to_string(),
        ));
    }
    Ok(Audio8PromptTextParts {
        before: vec![
            SYSTEM_OPEN.to_string(),
            REFERENCE_INSTRUCTION.to_string(),
            ensure_reference_speaker_tag(&reference_text),
            REFERENCE_SPEECH_HEADER.to_string(),
        ],
        after: vec![
            turn_end.clone(),
            USER_OPEN.to_string(),
            target_text,
            turn_end,
            assistant,
        ],
    })
}

fn speech_prompt_parts(
    reference: Option<(&str, Audio8VqCodes)>,
    target_text: &str,
) -> Result<Vec<Audio8PromptPart>> {
    let (reference_text, reference_codes) = match reference {
        Some((text, codes)) => (Some(text), Some(codes)),
        None => (None, None),
    };
    let text = speech_prompt_text_parts(reference_text, target_text)?;
    let mut parts = text
        .before
        .into_iter()
        .map(Audio8PromptPart::Text)
        .collect::<Vec<_>>();
    if let Some(codes) = reference_codes {
        parts.push(Audio8PromptPart::Vq(codes));
    }
    parts.extend(text.after.into_iter().map(Audio8PromptPart::Text));
    Ok(parts)
}

fn ensure_reference_speaker_tag(reference_text: &str) -> String {
    if has_speaker_tag(reference_text) {
        reference_text.to_string()
    } else {
        format!("<|speaker:0|>{reference_text}")
    }
}

/// Whether `text` holds a `<|speaker:N|>` tag with a decimal `N`.
fn has_speaker_tag(text: &str) -> bool {
    text.match_indices("<|speaker:").any(|(start, open)| {
        let rest = &text[start + open.len()..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        digits > 0 && rest[digits..].starts_with("|>")
    })
}

#[cfg(test)]
fn encode_prompt_parts_for_inference_with<F>(
    config: &Audio8Config,
    parts: &[Audio8PromptPart],
    encode_text: F,
) -> Result<Audio8ConditioningPrompt>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    encode_prompt_parts_bounded_with(config, parts, config.max_seq_len, encode_text)
}

fn encode_prompt_parts_bounded_with<F>(
    config: &Audio8Config,
    parts: &[Audio8PromptPart],
    context_limit: usize,
    mut encode_text: F,
) -> Result<Audio8ConditioningPrompt>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    let limit = context_limit.min(config.max_seq_len);
    let mut length = 0usize;
    // Tokenize once and borrow VQ rows. Validate the complete shape before any
    // dense codebook/mask allocation; retain one position for generation.
    let mut text_ids = Vec::with_capacity(parts.len());
    for part in parts {
        let ids = match part {
            Audio8PromptPart::Text(text) => Some(encode_text(text)?),
            Audio8PromptPart::Vq(codes) => {
                validate_vq_codes(config, codes)?;
                None
            }
        };
        let count = match (part, &ids) {
            (_, Some(ids)) => ids.len(),
            (Audio8PromptPart::Vq(codes), None) => codes.frame_count(),
            _ => unreachable!(),
        };
        length = length
            .checked_add(count)
            .ok_or_else(|| Error::InvalidInput("Audio8 prompt length overflow".into()))?;
        if length >= limit {
            return Err(Error::InvalidInput(format!("Audio8 prompt length {length} exceeds effective context {limit} (one generation position required)")));
        }
        text_ids.push(ids);
    }
    if length == 0 {
        return Err(Error::InvalidInput("Audio8 prompt cannot be empty".into()));
    }
    let rows = config
        .num_codebooks
        .checked_add(1)
        .ok_or_else(|| Error::Overloaded("Audio8 prompt row count overflow".into()))?;
    rows.checked_mul(length)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::Overloaded("Audio8 prompt allocation overflow".into()))?;
    let mut values = vec![vec![0u32; length]; rows];
    let mut vq_mask = vec![false; length];
    let mut cursor = 0;
    for (part, ids) in parts.iter().zip(text_ids) {
        match part {
            Audio8PromptPart::Text(_) => {
                let ids = ids.expect("text was tokenized");
                values[0][cursor..cursor + ids.len()].copy_from_slice(&ids);
                cursor += ids.len();
            }
            Audio8PromptPart::Vq(codes) => {
                for frame in 0..codes.frame_count() {
                    let col = cursor + frame;
                    values[0][col] = semantic_token_id(config, codes.codebooks[0][frame])?;
                    vq_mask[col] = true;
                    for (dst, src) in values[1..].iter_mut().zip(&codes.codebooks) {
                        dst[col] = src[frame];
                    }
                }
                cursor += codes.frame_count();
            }
        }
    }
    Ok(Audio8ConditioningPrompt {
        values,
        vq_mask,
        prompt_length: length,
    })
}

fn validate_vq_codes(config: &Audio8Config, codes: &Audio8VqCodes) -> Result<()> {
    if codes.codebooks.len() != config.num_codebooks {
        return Err(Error::InvalidInput(format!(
            "Audio8 VQ codes must contain {} codebooks, got {}",
            config.num_codebooks,
            codes.codebooks.len()
        )));
    }
    let frames = codes.frame_count();
    if frames == 0 {
        return Err(Error::InvalidInput(
            "Audio8 VQ codes cannot be empty".to_string(),
        ));
    }
    for (idx, codebook) in codes.codebooks.iter().enumerate() {
        if codebook.len() != frames {
            return Err(Error::InvalidInput(format!(
                "Audio8 VQ codebook {idx} has {} frames, expected {frames}",
                codebook.len()
            )));
        }
        let codebook_size = if idx == 0 {
            config.codebook_size
        } else {
            Audio8DacContract::CURRENT.residual_codebook_size
        };
        if let Some(code) = codebook
            .iter()
            .copied()
            .find(|code| *code as usize >= codebook_size)
        {
            return Err(Error::InvalidInput(format!(
                "Audio8 VQ code {code} in codebook {idx} exceeds codebook size {codebook_size}"
            )));
        }
    }
    Ok(())
}

impl Audio8VqCodes {
    pub fn frame_count(&self) -> usize {
        self.codebooks.first().map(Vec::len).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_context_is_enforced_before_dense_prompt_construction() {
        let config = crate::models::architectures::audio8::config::current_config();
        let parts = [
            Audio8PromptPart::Text("a".into()),
            Audio8PromptPart::Text("b".into()),
        ];
        let mut calls = 0;
        let error = encode_prompt_parts_bounded_with(&config, &parts, 4, |_| {
            calls += 1;
            Ok(vec![1; 4])
        })
        .unwrap_err();
        assert!(error.to_string().contains("effective context 4"));
        assert_eq!(
            calls, 1,
            "oversize must stop before further tokenization/allocation"
        );
        for tokens in [1, 17, config.max_seq_len - 1] {
            let prompt = encode_prompt_parts_bounded_with(&config, &parts[..1], tokens + 1, |_| {
                Ok(vec![1; tokens])
            })
            .unwrap();
            assert_eq!(prompt.prompt_length, tokens);
            assert_eq!(prompt.vq_mask.capacity(), tokens);
            assert!(prompt.values.iter().all(|row| row.capacity() == tokens));
        }
        assert!(
            encode_prompt_parts_bounded_with(&config, &parts[..1], usize::MAX, |_| Ok(
                vec![1; config.max_seq_len]
            ))
            .is_err()
        );
    }

    fn text(part: &Audio8PromptPart) -> &str {
        match part {
            Audio8PromptPart::Text(text) => text,
            Audio8PromptPart::Vq(_) => panic!("expected text"),
        }
    }

    #[test]
    fn plain_prompt_matches_the_published_processor() {
        let parts = speech_prompt_parts(None, "  Hello,\n  world.  ").unwrap();
        let rendered = parts.iter().map(text).collect::<Vec<_>>();
        assert_eq!(
            rendered,
            [
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "Hello, world.",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        );
    }

    #[test]
    fn reference_prompt_matches_the_published_processor() {
        let codes = Audio8VqCodes {
            codebooks: vec![vec![1, 2]; 10],
        };
        let parts = speech_prompt_parts(Some((" hello\tthere ", codes)), "target").unwrap();
        assert_eq!(parts.len(), 10);
        let before = parts[..4].iter().map(text).collect::<Vec<_>>();
        assert_eq!(
            before,
            [
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:\n\nText:\n",
                "<|speaker:0|>hello there",
                "\n\nSpeech:\n",
            ]
        );
        assert!(matches!(&parts[4], Audio8PromptPart::Vq(_)));
        let after = parts[5..].iter().map(text).collect::<Vec<_>>();
        assert_eq!(
            after,
            [
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "target",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        );
    }

    #[test]
    fn reference_transcript_keeps_its_own_speaker_tag() {
        assert_eq!(ensure_reference_speaker_tag("hi"), "<|speaker:0|>hi");
        assert_eq!(
            ensure_reference_speaker_tag("<|speaker:3|>hi"),
            "<|speaker:3|>hi"
        );
        // Not a tag the processor recognizes: it has no number.
        assert_eq!(
            ensure_reference_speaker_tag("<|speaker:x|>hi"),
            "<|speaker:0|><|speaker:x|>hi"
        );
    }

    #[test]
    fn rejects_empty_reference_or_target_text() {
        assert!(speech_prompt_parts(None, " \n ").is_err());
        let codes = Audio8VqCodes {
            codebooks: vec![vec![1]; 10],
        };
        assert!(speech_prompt_parts(Some(("  ", codes)), "hello").is_err());
    }

    #[test]
    fn validates_vq_code_shape_and_ranges() {
        let config = crate::models::architectures::audio8::config::current_config();
        let valid = Audio8VqCodes {
            codebooks: vec![vec![0, 1, 4095]]
                .into_iter()
                .chain((1..10).map(|_| vec![0, 1, 1023]))
                .collect(),
        };
        validate_vq_codes(&config, &valid).unwrap();

        let wrong_count = Audio8VqCodes {
            codebooks: vec![vec![0, 1]],
        };
        assert!(validate_vq_codes(&config, &wrong_count).is_err());

        let mut out_of_range = valid.clone();
        out_of_range.codebooks[1][0] = 1024;
        assert!(validate_vq_codes(&config, &out_of_range).is_err());
    }

    #[test]
    fn inference_prompt_tensor_places_vq_codes_on_semantic_columns() {
        let config = crate::models::architectures::audio8::config::current_config();
        let codes = Audio8VqCodes {
            codebooks: vec![vec![7, 8]]
                .into_iter()
                .chain((1..10).map(|idx| vec![idx, idx + 10]))
                .collect(),
        };
        let parts = vec![
            Audio8PromptPart::Text("a".to_string()),
            Audio8PromptPart::Vq(codes),
            Audio8PromptPart::Text("b".to_string()),
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

//! VibeVoice Qwen-tokenizer prompt assembly.

use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

const TTS_SYSTEM_PROMPT: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
const ASR_SYSTEM_PROMPT: &str =
    "You are a speech recognition model. Transcribe the user's audio faithfully.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceSpecialTokens {
    pub endoftext: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub object_ref_start: u32,
    pub object_ref_end: u32,
    pub box_start: u32,
    pub speech_start: u32,
    pub speech_end: u32,
    pub speech_pad: u32,
    pub image_pad: u32,
}

impl Default for VibeVoiceSpecialTokens {
    fn default() -> Self {
        Self {
            endoftext: 151_643,
            im_start: 151_644,
            im_end: 151_645,
            object_ref_start: 151_646,
            object_ref_end: 151_647,
            box_start: 151_648,
            speech_start: 151_652,
            speech_end: 151_653,
            speech_pad: 151_654,
            image_pad: 151_655,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceTtsPrompt {
    pub input_ids: Vec<u32>,
    pub reference_voice_range: Option<Range<usize>>,
    pub text_token_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceAsrPrompt {
    pub input_ids: Vec<u32>,
    pub acoustic_input_range: Range<usize>,
    pub prompt_token_count: usize,
}

pub struct VibeVoicePromptTokenizer {
    tokenizer: Tokenizer,
    specials: VibeVoiceSpecialTokens,
}

impl VibeVoicePromptTokenizer {
    pub fn load(model_dir: &Path, expected_vocab_size: usize) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size))?;
        let specials = read_special_tokens(model_dir).unwrap_or_default();
        Ok(Self {
            tokenizer,
            specials,
        })
    }

    pub fn specials(&self) -> &VibeVoiceSpecialTokens {
        &self.specials
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    pub fn build_tts_prompt(
        &self,
        text: &str,
        speaker: &str,
        reference_text: Option<&str>,
        reference_frames: usize,
    ) -> Result<VibeVoiceTtsPrompt> {
        let speaker = sanitize_tts_speaker(speaker);
        let text = text.trim();
        let mut input_ids = Vec::new();
        input_ids.extend(self.encode_text(TTS_SYSTEM_PROMPT)?);

        let reference_voice_range = if reference_frames > 0 {
            input_ids.extend(self.encode_text(" Voice input:\n")?);
            input_ids.extend(self.encode_text(&format!(" {speaker}:"))?);
            input_ids.push(self.specials.speech_start);
            let start = input_ids.len();
            input_ids.extend(std::iter::repeat(self.specials.speech_pad).take(reference_frames));
            let end = input_ids.len();
            input_ids.push(self.specials.speech_end);
            input_ids.extend(self.encode_text("\n")?);
            Some(start..end)
        } else {
            None
        };

        input_ids.extend(self.encode_text(&build_tts_text_input(
            text,
            &speaker,
            reference_text,
            reference_frames > 0,
        )?)?);
        let text_token_count = self.encode_text(text)?.len();
        input_ids.extend(self.encode_text(" Speech output:\n")?);
        input_ids.push(self.specials.speech_start);

        Ok(VibeVoiceTtsPrompt {
            input_ids,
            reference_voice_range,
            text_token_count,
        })
    }

    pub fn build_asr_prompt(
        &self,
        audio_seconds: f32,
        acoustic_frames: usize,
        extra_instruction: Option<&str>,
    ) -> Result<VibeVoiceAsrPrompt> {
        let mut input_ids = Vec::new();
        self.push_chat_segment(&mut input_ids, "system", ASR_SYSTEM_PROMPT)?;
        input_ids.push(self.specials.im_end);
        input_ids.extend(self.encode_text("\n")?);
        self.push_role_header(&mut input_ids, "user")?;

        input_ids.push(self.specials.object_ref_start);
        let start = input_ids.len();
        input_ids.extend(std::iter::repeat(self.specials.box_start).take(acoustic_frames));
        let end = input_ids.len();
        input_ids.push(self.specials.object_ref_end);
        input_ids.extend(self.encode_text(&format!(
            "\nThis is a {:.1} seconds audio, ",
            audio_seconds.max(0.0)
        ))?);
        if let Some(extra) = extra_instruction.filter(|extra| !extra.trim().is_empty()) {
            input_ids.extend(self.encode_text(extra.trim())?);
            input_ids.extend(self.encode_text(" ")?);
        }
        input_ids.extend(self.encode_text(
            "Please transcribe it with these keys: Start time, End time, Speaker ID, Content",
        )?);
        input_ids.push(self.specials.im_end);
        input_ids.extend(self.encode_text("\n")?);
        self.push_role_header(&mut input_ids, "assistant")?;

        Ok(VibeVoiceAsrPrompt {
            prompt_token_count: input_ids.len(),
            input_ids,
            acoustic_input_range: start..end,
        })
    }

    fn push_role_header(&self, ids: &mut Vec<u32>, role: &str) -> Result<()> {
        ids.push(self.specials.im_start);
        ids.extend(self.encode_text(&format!("{role}\n"))?);
        Ok(())
    }

    fn push_chat_segment(&self, ids: &mut Vec<u32>, role: &str, text: &str) -> Result<()> {
        self.push_role_header(ids, role)?;
        ids.extend(self.encode_text(text)?);
        Ok(())
    }
}

fn build_tts_text_input(
    text: &str,
    speaker: &str,
    reference_text: Option<&str>,
    include_reference: bool,
) -> Result<String> {
    let mut input = String::from(" Text input:\n");
    if include_reference {
        let reference_text = reference_text.unwrap_or_default().trim();
        if reference_text.is_empty() {
            return Err(Error::InvalidInput(
                "VibeVoice TTS reference_text cannot be empty when reference audio is provided"
                    .to_string(),
            ));
        }
        input.push_str(&format!(" {speaker}:{reference_text}\n"));
    }
    input.push_str(&format!(" {speaker}:{}\n", text.trim()));
    Ok(input)
}

fn sanitize_tts_speaker(speaker: &str) -> String {
    let trimmed = speaker.trim();
    if trimmed.is_empty() {
        "Speaker 0".to_string()
    } else if trimmed.starts_with("Speaker ") {
        trimmed.to_string()
    } else {
        "Speaker 0".to_string()
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

fn read_special_tokens(model_dir: &Path) -> Result<VibeVoiceSpecialTokens> {
    let path = model_dir.join("tokenizer_config.json");
    let raw = fs::read_to_string(path)?;
    let config: TokenizerConfig = serde_json::from_str(&raw)?;
    let by_token = config
        .added_tokens_decoder
        .iter()
        .filter_map(|(id, token)| {
            id.parse::<u32>()
                .ok()
                .map(|id| (token.content.as_str(), id))
        })
        .collect::<HashMap<_, _>>();
    let defaults = VibeVoiceSpecialTokens::default();
    Ok(VibeVoiceSpecialTokens {
        endoftext: by_token
            .get("<|endoftext|>")
            .copied()
            .unwrap_or(defaults.endoftext),
        im_start: by_token
            .get("<|im_start|>")
            .copied()
            .unwrap_or(defaults.im_start),
        im_end: by_token
            .get("<|im_end|>")
            .copied()
            .unwrap_or(defaults.im_end),
        object_ref_start: by_token
            .get("<|object_ref_start|>")
            .copied()
            .unwrap_or(defaults.object_ref_start),
        object_ref_end: by_token
            .get("<|object_ref_end|>")
            .copied()
            .unwrap_or(defaults.object_ref_end),
        box_start: by_token
            .get("<|box_start|>")
            .copied()
            .unwrap_or(defaults.box_start),
        speech_start: by_token
            .get("<|vision_start|>")
            .copied()
            .unwrap_or(defaults.speech_start),
        speech_end: by_token
            .get("<|vision_end|>")
            .copied()
            .unwrap_or(defaults.speech_end),
        speech_pad: by_token
            .get("<|vision_pad|>")
            .copied()
            .unwrap_or(defaults.speech_pad),
        image_pad: by_token
            .get("<|image_pad|>")
            .copied()
            .unwrap_or(defaults.image_pad),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_text_input_pairs_reference_transcript_before_target_text() {
        let block = build_tts_text_input(
            "Generate this line.",
            "Speaker 1",
            Some("Reference transcript."),
            true,
        )
        .expect("text input");

        assert_eq!(
            block,
            " Text input:\n Speaker 1:Reference transcript.\n Speaker 1:Generate this line.\n"
        );
    }

    #[test]
    fn tts_text_input_omits_reference_line_without_reference_audio() {
        let block = build_tts_text_input("Generate this line.", "Speaker 0", None, false)
            .expect("text input");

        assert_eq!(block, " Text input:\n Speaker 0:Generate this line.\n");
    }
}

//! Prompt assembly for Granite Speech ASR modes.

use crate::error::{Error, Result};
use crate::models::architectures::granite_speech::asr::config::{
    GraniteSpeechConfig, GraniteSpeechProcessorConfig, GraniteSpeechTokenizerConfig,
};
use crate::tokenizer::Tokenizer;

pub const GRANITE_SPEECH_SYSTEM_PROMPT: &str = "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant";
pub const GRANITE_SPEECH_ASR_PROMPT: &str =
    "<|audio|> can you transcribe the speech into a written format?";
pub const GRANITE_SPEECH_SPEAKER_PROMPT: &str = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.";
pub const GRANITE_SPEECH_TIMESTAMP_PROMPT: &str = "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechTask {
    Asr,
    SpeakerAttributed,
    WordTimestamps,
}

impl GraniteSpeechTask {
    pub fn default_prompt(self) -> &'static str {
        match self {
            Self::Asr => GRANITE_SPEECH_ASR_PROMPT,
            Self::SpeakerAttributed => GRANITE_SPEECH_SPEAKER_PROMPT,
            Self::WordTimestamps => GRANITE_SPEECH_TIMESTAMP_PROMPT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechPromptOptions {
    pub task: GraniteSpeechTask,
    pub language: Option<String>,
    pub keywords: Vec<String>,
    pub prefix_text: Option<String>,
    pub custom_prompt: Option<String>,
}

impl Default for GraniteSpeechPromptOptions {
    fn default() -> Self {
        Self {
            task: GraniteSpeechTask::Asr,
            language: None,
            keywords: Vec::new(),
            prefix_text: None,
            custom_prompt: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechSpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub audio_token_id: u32,
    pub audio_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechPrompt {
    pub text: String,
    pub input_ids: Vec<u32>,
    pub audio_token_positions: Vec<usize>,
    pub prefix_text_token_count: usize,
}

pub struct GraniteSpeechPromptTokenizer {
    tokenizer: Tokenizer,
    special_tokens: GraniteSpeechSpecialTokens,
}

impl GraniteSpeechPromptTokenizer {
    pub fn load(
        model_dir: &std::path::Path,
        config: &GraniteSpeechConfig,
        processor: &GraniteSpeechProcessorConfig,
        tokenizer_config: &GraniteSpeechTokenizerConfig,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_path_with_expected_vocab(
            model_dir,
            Some(config.text_config.vocab_size),
        )?;
        let audio_token = if tokenizer_config.audio_token.trim().is_empty() {
            processor.audio_token.clone()
        } else {
            tokenizer_config.audio_token.clone()
        };
        let audio_token_id = tokenizer
            .token_to_id(&audio_token)
            .unwrap_or(config.audio_token_index);
        let special_tokens = GraniteSpeechSpecialTokens {
            bos_token_id: config.text_config.bos_token_id,
            eos_token_id: config.text_config.eos_token_id,
            pad_token_id: config.text_config.pad_token_id,
            audio_token_id,
            audio_token,
        };
        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }

    pub fn special_tokens(&self) -> &GraniteSpeechSpecialTokens {
        &self.special_tokens
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    pub fn build_prompt(
        &self,
        options: &GraniteSpeechPromptOptions,
    ) -> Result<GraniteSpeechPrompt> {
        let user_prompt = self.user_prompt(options);
        if !user_prompt.contains(&self.special_tokens.audio_token) {
            return Err(Error::InvalidInput(format!(
                "Granite Speech prompt must contain {}",
                self.special_tokens.audio_token
            )));
        }

        let prefix = options
            .prefix_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("");
        let text = format!(
            "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{prefix}",
            system = GRANITE_SPEECH_SYSTEM_PROMPT,
            user = user_prompt,
        );
        let input_ids = self.encode(&text)?;
        let audio_token_positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, token)| {
                (*token == self.special_tokens.audio_token_id).then_some(idx)
            })
            .collect::<Vec<_>>();
        if audio_token_positions.is_empty() {
            return Err(Error::TokenizationError(format!(
                "Granite Speech audio token {} did not tokenize to id {}",
                self.special_tokens.audio_token, self.special_tokens.audio_token_id
            )));
        }
        let prefix_text_token_count = if prefix.is_empty() {
            0
        } else {
            self.encode(prefix)?.len()
        };

        Ok(GraniteSpeechPrompt {
            text,
            input_ids,
            audio_token_positions,
            prefix_text_token_count,
        })
    }

    fn user_prompt(&self, options: &GraniteSpeechPromptOptions) -> String {
        let mut prompt = options
            .custom_prompt
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| options.task.default_prompt())
            .to_string();
        if !prompt.contains(&self.special_tokens.audio_token) {
            prompt = format!("{} {}", self.special_tokens.audio_token, prompt);
        }

        if let Some(language) = options
            .language
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            prompt.push_str(" Language: ");
            prompt.push_str(language);
            prompt.push('.');
        }

        let keywords = options
            .keywords
            .iter()
            .map(|keyword| keyword.trim())
            .filter(|keyword| !keyword.is_empty())
            .collect::<Vec<_>>();
        if !keywords.is_empty() {
            prompt.push_str(" Keywords: ");
            prompt.push_str(&keywords.join(", "));
            prompt.push('.');
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_task_prompts_include_audio_token() {
        for task in [
            GraniteSpeechTask::Asr,
            GraniteSpeechTask::SpeakerAttributed,
            GraniteSpeechTask::WordTimestamps,
        ] {
            assert!(task.default_prompt().contains("<|audio|>"));
        }
    }
}

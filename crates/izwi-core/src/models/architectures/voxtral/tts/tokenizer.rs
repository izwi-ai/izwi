//! Tekken tokenizer wrapper for Voxtral TTS.

use std::fmt::Display;
use std::ops::Range;
use std::path::Path;

use tekken::{SpecialTokenPolicy, Tekkenizer};

use crate::error::{Error, Result};

use super::config::VoxtralTtsConfig;

const EOS_TOKEN_CANDIDATES: &[&str] = &["</s>", "~~ "];
const END_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[END_AUDIO]"];
const TEXT_TO_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[NEXT_AUDIO_TEXT]"];
const AUDIO_TO_TEXT_TOKEN_CANDIDATES: &[&str] = &["[REPEAT_AUDIO_TEXT]"];
const INST_START_TOKEN_CANDIDATES: &[&str] = &["[INST]"];
const INST_END_TOKEN_CANDIDATES: &[&str] = &["[/INST]"];
const FALLBACK_AUDIO_TO_TEXT_TOKEN_ID: u32 = 35;
const FALLBACK_TEXT_TO_AUDIO_TOKEN_ID: u32 = 36;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralTtsSpecialTokens {
    pub bos: u32,
    pub eos: Option<u32>,
    pub audio: u32,
    pub begin_audio: u32,
    pub end_audio: Option<u32>,
    pub text_to_audio: Option<u32>,
    pub audio_to_text: Option<u32>,
    pub inst_start: Option<u32>,
    pub inst_end: Option<u32>,
}

impl VoxtralTtsSpecialTokens {
    pub fn from_config(config: &VoxtralTtsConfig) -> Self {
        Self {
            bos: config.bos_token_id(),
            eos: None,
            audio: config.audio_token_id(),
            begin_audio: config.begin_audio_token_id(),
            end_audio: None,
            text_to_audio: None,
            audio_to_text: None,
            inst_start: None,
            inst_end: None,
        }
    }

    fn with_tekken(mut self, tokenizer: &Tekkenizer) -> Self {
        self.eos = optional_control_token(tokenizer, EOS_TOKEN_CANDIDATES);
        self.end_audio = optional_control_token(tokenizer, END_AUDIO_TOKEN_CANDIDATES);
        self.text_to_audio = optional_control_token(tokenizer, TEXT_TO_AUDIO_TOKEN_CANDIDATES)
            .or(Some(FALLBACK_TEXT_TO_AUDIO_TOKEN_ID));
        self.audio_to_text = optional_control_token(tokenizer, AUDIO_TO_TEXT_TOKEN_CANDIDATES)
            .or(Some(FALLBACK_AUDIO_TO_TEXT_TOKEN_ID));
        self.inst_start = optional_control_token(tokenizer, INST_START_TOKEN_CANDIDATES);
        self.inst_end = optional_control_token(tokenizer, INST_END_TOKEN_CANDIDATES);
        self
    }

    pub fn is_tts_control_token(&self, token: u32) -> bool {
        token == self.bos
            || token == self.audio
            || token == self.begin_audio
            || self.eos == Some(token)
            || self.end_audio == Some(token)
            || self.text_to_audio == Some(token)
            || self.audio_to_text == Some(token)
            || self.inst_start == Some(token)
            || self.inst_end == Some(token)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralTtsPrompt {
    pub input_ids: Vec<u32>,
    pub text_token_count: usize,
    pub voice_token_range: Option<Range<usize>>,
}

pub struct VoxtralTtsTokenizer {
    inner: Tekkenizer,
    specials: VoxtralTtsSpecialTokens,
}

impl VoxtralTtsTokenizer {
    pub fn load(model_dir: &Path, config: &VoxtralTtsConfig) -> Result<Self> {
        let tekken_path = model_dir.join("tekken.json");
        let inner = Tekkenizer::from_file(&tekken_path)
            .map_err(|err| tokenization_error(format!("load {}", tekken_path.display()), err))?;
        let specials = VoxtralTtsSpecialTokens::from_config(config).with_tekken(&inner);
        Ok(Self { inner, specials })
    }

    pub fn specials(&self) -> &VoxtralTtsSpecialTokens {
        &self.specials
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false, false)
            .map_err(|err| tokenization_error("encode Voxtral TTS text", err))
    }

    /// Build the conservative text-to-audio prefix used before acoustic frames
    /// are generated. Full voice-conditioning prompt assembly lives in the
    /// generation phase because it depends on loaded preset embeddings.
    pub fn build_text_to_audio_prompt(&self, text: &str) -> Result<VoxtralTtsPrompt> {
        let text_tokens = self.encode_text(text)?;
        let mut input_ids = Vec::with_capacity(text_tokens.len() + 3);
        input_ids.push(self.specials.bos);
        input_ids.extend(text_tokens.iter().copied());
        if let Some(text_to_audio) = self.specials.text_to_audio {
            input_ids.push(text_to_audio);
        }
        input_ids.push(self.specials.begin_audio);
        Ok(VoxtralTtsPrompt {
            input_ids,
            text_token_count: text_tokens.len(),
            voice_token_range: None,
        })
    }

    pub fn build_speech_prompt(&self, text: &str, voice_frames: usize) -> Result<VoxtralTtsPrompt> {
        let text_tokens = self.encode_text(text)?;
        build_speech_prompt_ids(&self.specials, &text_tokens, voice_frames)
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        let text_tokens = tokens
            .iter()
            .copied()
            .filter(|token| !self.specials.is_tts_control_token(*token))
            .collect::<Vec<_>>();
        self.inner
            .decode(&text_tokens, SpecialTokenPolicy::Ignore)
            .map_err(|err| tokenization_error("decode Voxtral TTS text", err))
    }
}

fn optional_control_token(tokenizer: &Tekkenizer, candidates: &[&str]) -> Option<u32> {
    candidates
        .iter()
        .find_map(|candidate| tokenizer.get_control_token(candidate).ok())
}

fn tokenization_error(context: impl Display, err: impl Display) -> Error {
    Error::TokenizationError(format!("{context}: {err}"))
}

fn build_speech_prompt_ids(
    specials: &VoxtralTtsSpecialTokens,
    text_tokens: &[u32],
    voice_frames: usize,
) -> Result<VoxtralTtsPrompt> {
    let text_to_audio = specials.text_to_audio.ok_or_else(|| {
        Error::TokenizationError(
            "Voxtral TTS tokenizer is missing [NEXT_AUDIO_TEXT] token".to_string(),
        )
    })?;
    let audio_to_text = specials.audio_to_text.ok_or_else(|| {
        Error::TokenizationError(
            "Voxtral TTS tokenizer is missing [REPEAT_AUDIO_TEXT] token".to_string(),
        )
    })?;

    let mut input_ids = Vec::with_capacity(text_tokens.len() + voice_frames + 5);
    input_ids.push(specials.bos);
    input_ids.push(specials.begin_audio);
    let voice_start = input_ids.len();
    input_ids.extend(std::iter::repeat(specials.audio).take(voice_frames));
    let voice_end = input_ids.len();
    input_ids.push(audio_to_text);
    input_ids.extend(text_tokens.iter().copied());
    input_ids.push(text_to_audio);
    input_ids.push(specials.begin_audio);
    Ok(VoxtralTtsPrompt {
        input_ids,
        text_token_count: text_tokens.len(),
        voice_token_range: Some(voice_start..voice_end),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::config::{VoxtralTtsConfig, fixture_json};

    #[test]
    fn special_tokens_come_from_params_without_tekken() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let specials = VoxtralTtsSpecialTokens::from_config(&config);
        assert_eq!(specials.bos, 1);
        assert_eq!(specials.audio, 24);
        assert_eq!(specials.begin_audio, 25);
        assert!(specials.eos.is_none());
    }

    #[test]
    fn speech_prompt_matches_mistral_reference_layout() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let mut specials = VoxtralTtsSpecialTokens::from_config(&config);
        specials.text_to_audio = Some(FALLBACK_TEXT_TO_AUDIO_TOKEN_ID);
        specials.audio_to_text = Some(FALLBACK_AUDIO_TO_TEXT_TOKEN_ID);
        let prompt = build_speech_prompt_ids(&specials, &[100, 101], 3).unwrap();

        assert_eq!(
            prompt.input_ids,
            vec![1, 25, 24, 24, 24, 35, 100, 101, 36, 25]
        );
        assert_eq!(prompt.text_token_count, 2);
        assert_eq!(prompt.voice_token_range, Some(2..5));
    }

    #[test]
    fn speech_prompt_uses_text_audio_separators_even_when_inst_tokens_exist() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let mut specials = VoxtralTtsSpecialTokens::from_config(&config);
        specials.text_to_audio = Some(36);
        specials.audio_to_text = Some(35);
        specials.inst_start = Some(3);
        specials.inst_end = Some(4);

        let prompt = build_speech_prompt_ids(&specials, &[100, 101], 2).unwrap();

        assert_eq!(prompt.input_ids, vec![1, 25, 24, 24, 35, 100, 101, 36, 25]);
        assert!(!prompt.input_ids.contains(&3));
        assert!(!prompt.input_ids.contains(&4));
    }
}

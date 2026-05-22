//! Tokenizer for the Voxtral Realtime model.
//!
//! Voxtral uses Mistral's Tekken tokenizer. The model directory must contain
//! `tekken.json`, which is downloaded alongside `params.json` and
//! `consolidated.safetensors`.

use std::fmt::Display;
use std::path::Path;

use tekken::{SpecialTokenPolicy, Tekkenizer};

use crate::error::{Error, Result};

const AUDIO_TOKEN_CANDIDATES: &[&str] = &["[AUDIO]"];
const BEGIN_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[BEGIN_AUDIO]"];
const END_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[END_AUDIO]"];
const UNK_TOKEN_CANDIDATES: &[&str] = &["<unk>", ""];
const BOS_TOKEN_CANDIDATES: &[&str] = &["<s>", "~~"];
const EOS_TOKEN_CANDIDATES: &[&str] = &["</s>", "~~ "];
const PAD_TOKEN_CANDIDATES: &[&str] = &["<pad>", ""];
const TRANSCRIBE_TOKEN_CANDIDATES: &[&str] = &["[TRANSCRIBE]"];
const STREAMING_PAD_TOKEN_CANDIDATES: &[&str] = &["[STREAMING_PAD]"];
const STREAMING_WORD_TOKEN_CANDIDATES: &[&str] = &["[STREAMING_WORD]"];
const TEXT_TO_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[NEXT_AUDIO_TEXT]"];
const AUDIO_TO_TEXT_TOKEN_CANDIDATES: &[&str] = &["[REPEAT_AUDIO_TEXT]"];
const DEFAULT_STREAMING_LEFT_PAD_TOKENS: usize = 32;
pub(crate) const OFFLINE_STREAMING_BUFFER_TOKENS: usize = 10;

/// Special token IDs for Voxtral.
#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    pub audio: u32,
    pub begin_audio: u32,
    pub end_audio: Option<u32>,
    pub pad: u32,
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
    pub transcribe: Option<u32>,
    pub streaming_pad: Option<u32>,
    pub streaming_word: Option<u32>,
    pub text_to_audio: Option<u32>,
    pub audio_to_text: Option<u32>,
}

impl SpecialTokenIds {
    fn from_tekkenizer(tokenizer: &Tekkenizer) -> Result<Self> {
        Ok(Self {
            audio: required_control_token(tokenizer, "audio", AUDIO_TOKEN_CANDIDATES)?,
            begin_audio: required_control_token(
                tokenizer,
                "begin_audio",
                BEGIN_AUDIO_TOKEN_CANDIDATES,
            )?,
            end_audio: optional_control_token(tokenizer, END_AUDIO_TOKEN_CANDIDATES),
            pad: required_control_token(tokenizer, "pad", PAD_TOKEN_CANDIDATES)?,
            bos: required_control_token(tokenizer, "bos", BOS_TOKEN_CANDIDATES)?,
            eos: required_control_token(tokenizer, "eos", EOS_TOKEN_CANDIDATES)?,
            unk: required_control_token(tokenizer, "unk", UNK_TOKEN_CANDIDATES)?,
            transcribe: optional_control_token(tokenizer, TRANSCRIBE_TOKEN_CANDIDATES),
            streaming_pad: optional_control_token(tokenizer, STREAMING_PAD_TOKEN_CANDIDATES),
            streaming_word: optional_control_token(tokenizer, STREAMING_WORD_TOKEN_CANDIDATES),
            text_to_audio: optional_control_token(tokenizer, TEXT_TO_AUDIO_TOKEN_CANDIDATES),
            audio_to_text: optional_control_token(tokenizer, AUDIO_TO_TEXT_TOKEN_CANDIDATES),
        })
    }

    fn is_voxtral_special(&self, token: u32) -> bool {
        token == self.audio
            || token == self.begin_audio
            || token == self.pad
            || token == self.bos
            || token == self.eos
            || token == self.unk
            || self.end_audio == Some(token)
            || self.transcribe == Some(token)
            || self.streaming_pad == Some(token)
            || self.streaming_word == Some(token)
            || self.text_to_audio == Some(token)
            || self.audio_to_text == Some(token)
    }
}

/// Audio configuration for tokenization.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sampling_rate: usize,
    pub frame_rate: f32,
    pub window_size: usize,
    pub hop_length: usize,
    pub num_mel_bins: usize,
    pub n_delay_tokens: usize,
    pub streaming_left_pad_tokens: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            frame_rate: 12.5,
            window_size: 400,
            hop_length: 160,
            num_mel_bins: 128,
            n_delay_tokens: 0,
            streaming_left_pad_tokens: DEFAULT_STREAMING_LEFT_PAD_TOKENS,
        }
    }
}

impl AudioConfig {
    /// Compute number of audio tokens for a given raw audio length.
    pub fn num_audio_tokens(&self, audio_length: usize) -> usize {
        let samples_per_frame = self.sampling_rate / self.frame_rate as usize;
        (audio_length + samples_per_frame - 1) / samples_per_frame
    }

    pub fn raw_audio_length_per_tok(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate).max(1.0) as usize
    }

    pub fn streaming_prompt_pad_tokens(&self) -> usize {
        self.streaming_left_pad_tokens + self.n_delay_tokens
    }

    pub fn offline_right_pad_tokens(&self) -> usize {
        self.n_delay_tokens + 1 + OFFLINE_STREAMING_BUFFER_TOKENS
    }
}

/// Tokenizer for Voxtral Realtime.
pub struct VoxtralTokenizer {
    inner: Tekkenizer,
    specials: SpecialTokenIds,
    audio_config: AudioConfig,
}

impl VoxtralTokenizer {
    pub fn load(model_dir: &Path, audio_config: AudioConfig) -> Result<Self> {
        let tekken_path = model_dir.join("tekken.json");
        let tekken_str = std::fs::read_to_string(&tekken_path)
            .map_err(|err| tokenization_error(format!("read {}", tekken_path.display()), err))?;
        let audio_config = merge_tekken_audio_metadata(audio_config, &tekken_str)?;
        let inner = Tekkenizer::from_file(&tekken_path)
            .map_err(|err| tokenization_error(format!("load {}", tekken_path.display()), err))?;
        Self::from_tekkenizer(inner, audio_config)
    }

    fn from_tekkenizer(inner: Tekkenizer, audio_config: AudioConfig) -> Result<Self> {
        let specials = SpecialTokenIds::from_tekkenizer(&inner)?;
        Ok(Self {
            inner,
            specials,
            audio_config,
        })
    }

    /// Get special token IDs.
    pub fn specials(&self) -> &SpecialTokenIds {
        &self.specials
    }

    /// Get audio configuration.
    pub fn audio_config(&self) -> &AudioConfig {
        &self.audio_config
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Build the Mistral streaming transcription prefix:
    /// BOS followed by left-padding and delay STREAMING_PAD tokens.
    pub fn build_transcription_prompt(&self) -> Result<Vec<u32>> {
        let streaming_pad = self.specials.streaming_pad.ok_or_else(|| {
            Error::TokenizationError(
                "Voxtral streaming transcription requires [STREAMING_PAD]".to_string(),
            )
        })?;
        let mut tokens = Vec::with_capacity(self.audio_config.streaming_prompt_pad_tokens() + 1);
        tokens.push(self.specials.bos);
        tokens.extend(std::iter::repeat_n(
            streaming_pad,
            self.audio_config.streaming_prompt_pad_tokens(),
        ));
        Ok(tokens)
    }

    /// Decode generated tokens to text.
    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        let text_tokens = tokens
            .iter()
            .copied()
            .filter(|token| {
                !self.specials.is_voxtral_special(*token) && !self.inner.is_special_token(*token)
            })
            .collect::<Vec<_>>();
        self.inner
            .decode(&text_tokens, SpecialTokenPolicy::Ignore)
            .map_err(|err| tokenization_error("decode Tekken tokens", err))
    }
}

#[derive(Debug, Default, serde::Deserialize)]
struct TekkenAudioMetadata {
    #[serde(default)]
    audio: Option<TekkenAudioConfig>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct TekkenAudioConfig {
    streaming_n_left_pad_tokens: Option<usize>,
}

fn merge_tekken_audio_metadata(
    mut audio_config: AudioConfig,
    tekken_json: &str,
) -> Result<AudioConfig> {
    let metadata: TekkenAudioMetadata = serde_json::from_str(tekken_json)
        .map_err(|err| tokenization_error("parse Tekken audio metadata", err))?;
    if let Some(left_pad) = metadata
        .audio
        .and_then(|audio| audio.streaming_n_left_pad_tokens)
    {
        audio_config.streaming_left_pad_tokens = left_pad;
    }
    Ok(audio_config)
}

fn required_control_token(tokenizer: &Tekkenizer, name: &str, candidates: &[&str]) -> Result<u32> {
    for candidate in candidates {
        if let Ok(token_id) = tokenizer.get_control_token(candidate) {
            return Ok(token_id);
        }
    }

    Err(Error::TokenizationError(format!(
        "Voxtral Tekken tokenizer is missing required `{name}` control token; tried {}",
        candidates.join(", ")
    )))
}

fn optional_control_token(tokenizer: &Tekkenizer, candidates: &[&str]) -> Option<u32> {
    candidates
        .iter()
        .find_map(|candidate| tokenizer.get_control_token(candidate).ok())
}

fn tokenization_error(context: impl Display, err: impl Display) -> Error {
    Error::TokenizationError(format!("{context}: {err}"))
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine;
    use tekken::config::{TokenInfo, TokenizerVersion};
    use tekken::special_tokens::SpecialTokenInfo;

    use super::*;

    fn token(rank: usize, text: &str) -> TokenInfo {
        TokenInfo {
            rank,
            token_bytes: BASE64.encode(text.as_bytes()),
            token_str: Some(text.to_string()),
        }
    }

    fn byte_token(rank: usize) -> TokenInfo {
        let byte = rank as u8;
        TokenInfo {
            rank,
            token_bytes: BASE64.encode([byte]),
            token_str: None,
        }
    }

    fn special(rank: usize, text: &str) -> SpecialTokenInfo {
        SpecialTokenInfo {
            rank,
            token_str: text.to_string(),
            is_control: true,
        }
    }

    fn test_tekkenizer() -> Tekkenizer {
        let mut vocab = (0..=255).map(byte_token).collect::<Vec<_>>();
        vocab.push(token(256, "hello"));
        vocab.push(token(257, " world"));
        let special_tokens = vec![
            special(258, ""),
            special(259, "~~"),
            special(260, "~~ "),
            special(261, "[AUDIO]"),
            special(262, "[BEGIN_AUDIO]"),
            special(263, "[STREAMING_PAD]"),
            special(264, "[STREAMING_WORD]"),
        ];
        Tekkenizer::new(
            vocab,
            &special_tokens,
            r"(?s).+",
            265,
            special_tokens.len(),
            TokenizerVersion::V13,
            None,
        )
        .expect("test Tekkenizer")
    }

    #[test]
    fn derives_special_token_ids_from_tekken() {
        let tokenizer =
            VoxtralTokenizer::from_tekkenizer(test_tekkenizer(), AudioConfig::default())
                .expect("Voxtral tokenizer");

        assert_eq!(tokenizer.specials().audio, 261);
        assert_eq!(tokenizer.specials().begin_audio, 262);
        assert_eq!(tokenizer.specials().streaming_pad, Some(263));
        assert_eq!(tokenizer.specials().streaming_word, Some(264));
        assert_eq!(tokenizer.specials().end_audio, None);
    }

    #[test]
    fn reads_streaming_left_pad_tokens_from_tekken_audio_metadata() {
        let config = merge_tekken_audio_metadata(
            AudioConfig::default(),
            r#"{
                "audio": {
                    "streaming_n_left_pad_tokens": 12
                }
            }"#,
        )
        .expect("audio metadata");

        assert_eq!(config.streaming_left_pad_tokens, 12);
        assert_eq!(
            config.streaming_prompt_pad_tokens(),
            12 + config.n_delay_tokens
        );
    }

    #[test]
    fn transcription_prompt_uses_streaming_pad_schedule() {
        let tokenizer = VoxtralTokenizer::from_tekkenizer(
            test_tekkenizer(),
            AudioConfig {
                n_delay_tokens: 3,
                streaming_left_pad_tokens: 2,
                ..AudioConfig::default()
            },
        )
        .expect("Voxtral tokenizer");

        let prompt = tokenizer
            .build_transcription_prompt()
            .expect("transcription prompt");

        assert_eq!(
            prompt,
            vec![
                tokenizer.specials().bos,
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
            ]
        );
    }

    #[test]
    fn decode_text_uses_tekken_and_ignores_specials() {
        let tokenizer =
            VoxtralTokenizer::from_tekkenizer(test_tekkenizer(), AudioConfig::default())
                .expect("Voxtral tokenizer");

        let decoded = tokenizer
            .decode_text(&[
                tokenizer.specials().bos,
                b'h' as u32,
                b'e' as u32,
                b'l' as u32,
                b'l' as u32,
                b'o' as u32,
                tokenizer.specials().streaming_word.unwrap(),
                tokenizer.specials().eos,
            ])
            .expect("decoded text");

        assert!(!decoded.is_empty());
        assert!(!decoded.contains("[STREAMING_WORD]"));
    }
}

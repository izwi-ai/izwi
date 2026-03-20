//! Tokenizer wrapper for the retained Qwen speech/aligner stack.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_token: u32,
    pub timestamp: Option<u32>,
    pub asr_text: Option<u32>,
    pub fim_prefix: Option<u32>,
    pub fim_middle: Option<u32>,
    pub fim_suffix: Option<u32>,
    pub fim_pad: Option<u32>,
    pub eos: u32,
    pub eos_alt: Option<u32>,
    pub pad: u32,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    pad_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

pub struct AsrTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    timestamp_token_indices: HashMap<u32, u32>,
}

impl AsrTokenizer {
    pub fn load(model_dir: &Path, expected_vocab_size: usize) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size))?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

        let mut id_for = |token: &str| -> Option<u32> {
            config.added_tokens_decoder.iter().find_map(|(id, entry)| {
                if entry.content == token {
                    id.parse().ok()
                } else {
                    None
                }
            })
        };

        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let audio_start = id_for("<|audio_start|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_start|> token id".to_string())
        })?;
        let audio_end = id_for("<|audio_end|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_end|> token id".to_string())
        })?;
        let audio_token = id_for("<|audio_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_pad|> token id".to_string())
        })?;
        let timestamp = id_for("<timestamp>");
        let asr_text = id_for("<asr_text>");
        let fim_prefix = id_for("<|fim_prefix|>");
        let fim_middle = id_for("<|fim_middle|>");
        let fim_suffix = id_for("<|fim_suffix|>");
        let fim_pad = id_for("<|fim_pad|>");

        let eos = config
            .eos_token
            .as_deref()
            .and_then(&mut id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");
        let pad = config
            .pad_token
            .as_deref()
            .and_then(&mut id_for)
            .unwrap_or(eos);

        let timestamp_token_indices: HashMap<u32, u32> = config
            .added_tokens_decoder
            .iter()
            .filter_map(|(id, entry)| {
                let token_id = id.parse::<u32>().ok()?;
                let timestamp_idx = parse_timestamp_token_index(&entry.content)?;
                Some((token_id, timestamp_idx))
            })
            .collect();

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                audio_start,
                audio_end,
                audio_token,
                timestamp,
                asr_text,
                fim_prefix,
                fim_middle,
                fim_suffix,
                fim_pad,
                eos,
                eos_alt,
                pad,
            },
            timestamp_token_indices,
        })
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }

    pub fn decode_text_with_special_tokens(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode_with_special_tokens(&filtered)
    }

    pub fn timestamp_index_for_token(&self, token_id: u32) -> Option<u32> {
        self.timestamp_token_indices.get(&token_id).copied()
    }

    pub fn specials(&self) -> &SpecialTokenIds {
        &self.specials
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn parse_timestamp_token_index(token: &str) -> Option<u32> {
    let trimmed = token.trim();
    if !trimmed.starts_with("<|timestamp_") || !trimmed.ends_with("|>") {
        return None;
    }
    let inner = trimmed.strip_prefix("<|timestamp_")?.strip_suffix("|>")?;
    inner.parse::<u32>().ok()
}

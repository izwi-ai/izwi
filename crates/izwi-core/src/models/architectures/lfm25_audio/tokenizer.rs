use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value as GgufValue;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::Tokenizer;

use super::config::{LFM25_AUDIO_TEXT_END_TOKEN, LFM25_AUDIO_TEXT_TO_AUDIO_TOKEN};

#[derive(Debug, Clone)]
pub struct Lfm25SpecialTokenIds {
    pub bos: Option<u32>,
    pub im_start: u32,
    pub im_end: u32,
    pub eos: u32,
    pub eos_alt: Option<u32>,
    pub audio_start: u32,
    pub text_end: u32,
}

pub struct Lfm25TextTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    literal_special_tokens: Vec<(String, u32)>,
    specials: Lfm25SpecialTokenIds,
}

impl Lfm25TextTokenizer {
    pub fn load(loader: &GgufLoader) -> Result<Self> {
        let tokens = required_string_array(loader, "tokenizer.ggml.tokens")?;
        let merges = required_string_array(loader, "tokenizer.ggml.merges")?;
        let pre_tokenizer = loader.get_metadata_string("tokenizer.ggml.pre");
        let inner =
            Tokenizer::from_gguf_bpe(&tokens, &merges, pre_tokenizer.as_deref(), false)?;
        let vocab_size = inner.vocab_size();

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| {
                Error::TokenizationError(format!("GGUF tokenizer id out of range: {idx}"))
            })?;
            token_to_id.insert(token.clone(), id);
        }

        let id_for = |token: &str| token_to_id.get(token).copied();
        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let audio_start = id_for(LFM25_AUDIO_TEXT_TO_AUDIO_TOKEN).ok_or_else(|| {
            Error::TokenizationError(format!(
                "Missing {} token id",
                LFM25_AUDIO_TEXT_TO_AUDIO_TOKEN
            ))
        })?;
        let text_end = id_for(LFM25_AUDIO_TEXT_END_TOKEN).ok_or_else(|| {
            Error::TokenizationError(format!("Missing {} token id", LFM25_AUDIO_TEXT_END_TOKEN))
        })?;
        let bos = id_for("<|startoftext|>");
        let eos = loader
            .get_metadata_u64("tokenizer.ggml.eos_token_id")
            .and_then(|value| u32::try_from(value).ok())
            .or_else(|| id_for("<|endoftext|>"))
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        let mut literal_special_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter_map(|(token, id)| {
                (token.starts_with("<|") && token.ends_with("|>")).then_some((token.clone(), *id))
            })
            .collect();
        literal_special_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });

        Ok(Self {
            inner,
            vocab_size,
            literal_special_tokens,
            specials: Lfm25SpecialTokenIds {
                bos,
                im_start,
                im_end,
                eos,
                eos_alt,
                audio_start,
                text_end,
            },
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn specials(&self) -> &Lfm25SpecialTokenIds {
        &self.specials
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        if self.literal_special_tokens.is_empty() {
            return self.inner.encode(text);
        }

        let mut ids = Vec::new();
        let mut offset = 0usize;
        while offset < text.len() {
            let tail = &text[offset..];
            let mut next_match: Option<(usize, &str, u32)> = None;
            for (token, token_id) in &self.literal_special_tokens {
                if let Some(rel_idx) = tail.find(token) {
                    let candidate = (rel_idx, token.as_str(), *token_id);
                    match next_match {
                        None => next_match = Some(candidate),
                        Some((best_idx, best_token, _)) => {
                            if rel_idx < best_idx
                                || (rel_idx == best_idx && token.len() > best_token.len())
                            {
                                next_match = Some(candidate);
                            }
                        }
                    }
                }
            }

            let Some((rel_idx, matched_token, matched_id)) = next_match else {
                ids.extend(self.inner.encode(tail)?);
                break;
            };

            if rel_idx > 0 {
                ids.extend(self.inner.encode(&tail[..rel_idx])?);
            }
            ids.push(matched_id);
            offset += rel_idx + matched_token.len();
        }

        Ok(ids)
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }
}

fn required_string_array(loader: &GgufLoader, key: &str) -> Result<Vec<String>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let GgufValue::String(raw) = item else {
            return Err(Error::ModelLoadError(format!(
                "Expected string array values for {key}"
            )));
        };
        values.push(raw.clone());
    }
    Ok(values)
}

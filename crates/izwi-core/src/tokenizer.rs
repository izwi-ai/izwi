//! Text tokenization for Qwen3-TTS

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use uuid::Uuid;

use serde::Deserialize;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokenizerSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::AddedToken;
use tokenizers::SplitDelimiterBehavior;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub audio_start_id: Option<u32>,
    pub audio_end_id: Option<u32>,
}

pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: SpecialTokens,
}

const QWEN2_PRETOKENIZER_REGEX: &str =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

impl Tokenizer {
    pub fn from_path(model_dir: &Path) -> Result<Self> {
        Self::from_path_with_expected_vocab(model_dir, None)
    }

    pub fn from_path_with_expected_vocab(
        model_dir: &Path,
        expected_vocab_size: Option<usize>,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            match Self::from_tokenizer_json(&tokenizer_path) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(err) => {
                    warn!(
                        "Failed to parse tokenizer.json at {:?}: {}. Falling back to vocab.json + merges.txt",
                        tokenizer_path, err
                    );
                }
            }
        }

        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");

        if vocab_path.exists() && merges_path.exists() {
            return Self::from_vocab_merges(
                model_dir,
                &vocab_path,
                &merges_path,
                expected_vocab_size,
            );
        }

        Err(Error::TokenizationError(format!(
            "No tokenizer found in {:?}",
            model_dir
        )))
    }

    fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| Error::TokenizationError(e.to_string()))?;
        debug!("Loaded tokenizer from {:?}", path);
        Self::new_with_tokenizer(inner)
    }

    fn from_vocab_merges(
        model_dir: &Path,
        vocab_path: &Path,
        merges_path: &Path,
        expected_vocab_size: Option<usize>,
    ) -> Result<Self> {
        info!("Loading BPE tokenizer from vocab.json + merges.txt");
        let vocab_str = vocab_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid vocab path".to_string()))?;
        let merges_str = merges_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid merges path".to_string()))?;

        let bpe = BPE::from_file(vocab_str, merges_str)
            .byte_fallback(true)
            .build()
            .map_err(|e| Error::TokenizationError(format!("BPE build failed: {}", e)))?;

        let mut inner = HfTokenizer::new(bpe);

        let config = load_tokenizer_config(model_dir)?;
        let add_prefix_space = config
            .as_ref()
            .and_then(|cfg| cfg.add_prefix_space)
            .unwrap_or(true);
        let byte_level = ByteLevel::new(add_prefix_space, true, true);
        inner.with_pre_tokenizer(Some(byte_level.clone()));
        let decoder = DecoderWrapper::Sequence(DecoderSequence::new(vec![
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::ByteLevel(byte_level),
        ]));
        inner.with_decoder(Some(decoder));

        if let Some(cfg) = config {
            let mut added: Vec<(u32, AddedToken, bool)> = cfg
                .added_tokens_decoder
                .into_iter()
                .filter_map(|(id, entry)| {
                    id.parse::<u32>().ok().map(|id| {
                        let is_special = entry.special;
                        (id, entry.into_added_token(), is_special)
                    })
                })
                .collect();
            added.sort_by_key(|(id, _, _)| *id);

            // Preserve upstream token ids exactly by inserting in id order.
            // Grouping normal/special tokens changes insertion order and shifts ids
            // for control tokens like <asr_text>, breaking prompt semantics.
            for (expected_id, token, is_special) in added {
                let current_size = inner.get_vocab_size(true) as u32;
                if expected_id < current_size {
                    continue;
                }
                if expected_id > current_size {
                    let missing = (expected_id - current_size) as usize;
                    let mut fillers = Vec::with_capacity(missing);
                    for idx in 0..missing {
                        fillers.push(AddedToken::from(
                            format!("<|gap_{}|>", current_size + idx as u32),
                            false,
                        ));
                    }
                    inner.add_tokens(&fillers);
                }

                if is_special {
                    inner.add_special_tokens(&[token]);
                } else {
                    inner.add_tokens(&[token]);
                }
            }
        }

        if let Some(expected_vocab_size) = expected_vocab_size {
            let current_size = inner.get_vocab_size(true);
            if current_size < expected_vocab_size {
                let missing = expected_vocab_size - current_size;
                let mut byte_tokens = Vec::with_capacity(missing);
                for byte in 0..missing {
                    byte_tokens.push(AddedToken::from(format!("<0x{:02X}>", byte), false));
                }
                inner.add_tokens(&byte_tokens);
            }
        }

        debug!("Loaded BPE tokenizer with byte-level fallback");
        Self::new_with_tokenizer(inner)
    }

    pub fn from_gguf_bpe(
        tokens: &[String],
        merges: &[String],
        pre_tokenizer: Option<&str>,
        add_prefix_space: bool,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::TokenizationError(
                "Cannot build tokenizer from empty GGUF token list".to_string(),
            ));
        }

        let mut vocab = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| {
                Error::TokenizationError(format!("GGUF tokenizer id out of range: {idx}"))
            })?;
            vocab.insert(token.clone(), id);
        }

        let mut merge_lines = Vec::with_capacity(merges.len());
        for merge in merges {
            let merge = merge.trim();
            if merge.is_empty() || merge.starts_with('#') {
                continue;
            }
            merge_lines.push(merge.to_string());
        }

        let temp_dir = std::env::temp_dir().join(format!("izwi-gguf-tokenizer-{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).map_err(|e| {
            Error::TokenizationError(format!("Failed to create GGUF tokenizer temp dir: {e}"))
        })?;
        let vocab_path = temp_dir.join("vocab.json");
        let merges_path = temp_dir.join("merges.txt");
        fs::write(
            &vocab_path,
            serde_json::to_vec(&vocab).map_err(Error::from)?,
        )
        .map_err(|e| {
            Error::TokenizationError(format!("Failed to write GGUF vocab.json temp file: {e}"))
        })?;
        fs::write(&merges_path, merge_lines.join("\n")).map_err(|e| {
            Error::TokenizationError(format!("Failed to write GGUF merges.txt temp file: {e}"))
        })?;
        let vocab_str = vocab_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid temporary vocab path".to_string()))?;
        let merges_str = merges_path
            .to_str()
            .ok_or_else(|| Error::TokenizationError("Invalid temporary merges path".to_string()))?;

        let bpe = BPE::from_file(vocab_str, merges_str)
            .byte_fallback(true)
            .build()
            .map_err(|e| Error::TokenizationError(format!("BPE build failed: {e}")))?;
        let _ = fs::remove_file(&vocab_path);
        let _ = fs::remove_file(&merges_path);
        let _ = fs::remove_dir(&temp_dir);
        let mut inner = HfTokenizer::new(bpe);

        let byte_level = if matches!(pre_tokenizer, Some("qwen2")) {
            let split = Split::new(
                SplitPattern::Regex(QWEN2_PRETOKENIZER_REGEX.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| Error::TokenizationError(format!("Invalid Qwen2 split regex: {e}")))?;
            let byte_level = ByteLevel::new(add_prefix_space, false, false);
            let sequence = PreTokenizerSequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(byte_level.clone()),
            ]);
            inner.with_pre_tokenizer(Some(sequence));
            byte_level
        } else {
            let byte_level = ByteLevel::new(add_prefix_space, true, true);
            inner.with_pre_tokenizer(Some(byte_level.clone()));
            byte_level
        };

        let decoder = DecoderWrapper::Sequence(DecoderSequence::new(vec![
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::ByteLevel(byte_level),
        ]));
        inner.with_decoder(Some(decoder));

        debug!("Loaded BPE tokenizer from GGUF metadata");
        Self::new_with_tokenizer(inner)
    }

    fn new_with_tokenizer(inner: HfTokenizer) -> Result<Self> {
        let special_tokens = SpecialTokens::default();

        Ok(Self {
            inner,
            special_tokens,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::TokenizationError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::TokenizationError(e.to_string()))
    }

    pub fn decode_with_special_tokens(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| Error::TokenizationError(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    pub fn format_tts_prompt(&self, text: &str, speaker: Option<&str>) -> String {
        let speaker_tag = speaker.unwrap_or("default");
        format!("[speaker:{}] {}", speaker_tag, text)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default)]
    add_prefix_space: Option<bool>,
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

impl AddedTokenConfig {
    fn into_added_token(self) -> AddedToken {
        AddedToken::from(self.content, self.special)
            .single_word(self.single_word)
            .lstrip(self.lstrip)
            .rstrip(self.rstrip)
            .normalized(self.normalized)
    }
}

fn load_tokenizer_config(model_dir: &Path) -> Result<Option<TokenizerConfigFile>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_str = fs::read_to_string(config_path)?;
    let config: TokenizerConfigFile = serde_json::from_str(&config_str)?;
    Ok(Some(config))
}

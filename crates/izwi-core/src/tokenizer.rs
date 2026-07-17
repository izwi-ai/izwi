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
use tokenizers::tokenizer::step_decode_stream;
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

/// Owned state for incremental token decoding.
///
/// `tokenizers::DecodeStream` borrows its tokenizer, which makes it unsuitable
/// for storing beside the tokenizer in a long-lived model/session struct. This
/// state mirrors the upstream decoder's owned fields and additionally retains
/// the complete generated id list so [`Tokenizer::finish_incremental_decode`]
/// can exactly match a one-shot decode, including an incomplete final UTF-8
/// sequence.
#[derive(Debug, Clone)]
pub struct IncrementalDecoder {
    skip_special_tokens: bool,
    stream_ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    all_ids: Vec<u32>,
    emitted: String,
    finished: bool,
}

impl IncrementalDecoder {
    pub fn new(skip_special_tokens: bool) -> Self {
        Self {
            skip_special_tokens,
            stream_ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            all_ids: Vec::new(),
            emitted: String::new(),
            finished: false,
        }
    }
}

const QWEN2_PRETOKENIZER_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

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

    pub fn from_hf_json_bytes(bytes: &[u8]) -> Result<Self> {
        let inner =
            HfTokenizer::from_bytes(bytes).map_err(|e| Error::TokenizationError(e.to_string()))?;
        debug!("Loaded tokenizer from in-memory HuggingFace tokenizer JSON");
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
            if let Some(previous_id) = vocab.insert(token.clone(), id) {
                return Err(Error::TokenizationError(format!(
                    "Duplicate GGUF tokenizer token {:?} at ids {previous_id} and {id}",
                    token
                )));
            }
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

        let byte_level = if matches!(pre_tokenizer, Some("qwen2" | "qwen35")) {
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

    /// Register GGUF control (type 3) and user-defined (type 4) tokens as
    /// atomic added tokens while preserving their ids from the model vocab.
    ///
    /// Control tokens are special and therefore omitted by normal decoding;
    /// user-defined tokens remain visible. `tokenizers` reuses the BPE model id
    /// when the token content is already present, so this does not grow or
    /// reorder the GGUF vocabulary.
    pub fn register_gguf_token_types(
        &mut self,
        tokens: &[String],
        token_types: &[u32],
    ) -> Result<()> {
        if tokens.len() != token_types.len() {
            return Err(Error::TokenizationError(format!(
                "GGUF tokenizer token/type length mismatch: {} tokens for {} token types",
                tokens.len(),
                token_types.len()
            )));
        }

        let original_vocab_size = self.inner.get_vocab_size(true);
        let mut atomic_tokens = Vec::new();
        for (id, (token, token_type)) in tokens.iter().zip(token_types).enumerate() {
            if !matches!(token_type, 3 | 4) {
                continue;
            }

            let expected_id = u32::try_from(id).map_err(|_| {
                Error::TokenizationError(format!("GGUF tokenizer id out of range: {id}"))
            })?;
            let actual_id = self.inner.token_to_id(token);
            if actual_id != Some(expected_id) {
                return Err(Error::TokenizationError(format!(
                    "GGUF atomic token id mismatch for {:?}: expected {expected_id}, found {:?}",
                    token, actual_id
                )));
            }

            let special = *token_type == 3;
            atomic_tokens.push(AddedToken::from(token.clone(), special).normalized(false));
        }

        self.inner.add_tokens(&atomic_tokens);
        if self.inner.get_vocab_size(true) != original_vocab_size {
            return Err(Error::TokenizationError(format!(
                "Registering GGUF atomic tokens changed vocab size from {original_vocab_size} to {}",
                self.inner.get_vocab_size(true)
            )));
        }
        for (id, (token, token_type)) in tokens.iter().zip(token_types).enumerate() {
            if matches!(token_type, 3 | 4) && self.inner.token_to_id(token) != Some(id as u32) {
                return Err(Error::TokenizationError(format!(
                    "Registering GGUF atomic token {:?} changed its id",
                    token
                )));
            }
        }
        Ok(())
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

    /// Decode one token into the next complete UTF-8-safe text delta.
    ///
    /// An empty string means the decoder is buffering bytes until they form a
    /// complete decoded suffix. Call [`Self::finish_incremental_decode`] when
    /// generation finishes to emit any remaining one-shot-decoder suffix.
    pub fn decode_incrementally(
        &self,
        decoder: &mut IncrementalDecoder,
        token_id: u32,
    ) -> Result<String> {
        if decoder.finished {
            return Err(Error::TokenizationError(
                "Cannot decode another token after incremental decoding was finished".to_string(),
            ));
        }
        decoder.all_ids.push(token_id);
        let delta = step_decode_stream(
            &self.inner,
            vec![token_id],
            decoder.skip_special_tokens,
            &mut decoder.stream_ids,
            &mut decoder.prefix,
            &mut decoder.prefix_index,
        )
        .map_err(|e| Error::TokenizationError(e.to_string()))?
        .unwrap_or_default();
        decoder.emitted.push_str(&delta);
        Ok(delta)
    }

    /// Finish an incremental decode and return the not-yet-emitted suffix.
    ///
    /// This is idempotent and deliberately compares against a one-shot decode,
    /// ensuring streamed deltas plus this suffix have exactly the same decoder
    /// semantics even when generation ends part-way through a byte sequence.
    pub fn finish_incremental_decode(&self, decoder: &mut IncrementalDecoder) -> Result<String> {
        if decoder.finished {
            return Ok(String::new());
        }
        let decoded = self
            .inner
            .decode(&decoder.all_ids, decoder.skip_special_tokens)
            .map_err(|e| Error::TokenizationError(e.to_string()))?;
        let suffix = decoded.strip_prefix(&decoder.emitted).ok_or_else(|| {
            Error::TokenizationError(format!(
                "Incremental decode diverged from one-shot decode: emitted {:?}, decoded {:?}",
                decoder.emitted, decoded
            ))
        })?;
        let suffix = suffix.to_string();
        decoder.emitted.push_str(&suffix);
        decoder.finished = true;
        Ok(suffix)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn vocab(&self) -> HashMap<String, u32> {
        self.inner.get_vocab(true)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_level_char(byte: u8) -> char {
        let mut bytes: Vec<u8> = (b'!'..=b'~')
            .chain(b'\xA1'..=b'\xAC')
            .chain(b'\xAE'..=b'\xFF')
            .collect();
        let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
        let mut next = 0u32;
        for candidate in 0..=u8::MAX {
            if !bytes.contains(&candidate) {
                bytes.push(candidate);
                codepoints.push(256 + next);
                next += 1;
            }
        }
        let index = bytes
            .iter()
            .position(|candidate| *candidate == byte)
            .expect("byte-level alphabet contains every byte");
        char::from_u32(codepoints[index]).expect("byte-level codepoint is valid")
    }

    fn byte_level_piece(text: &str) -> String {
        text.as_bytes()
            .iter()
            .map(|byte| byte_level_char(*byte))
            .collect()
    }

    #[test]
    fn qwen35_uses_qwen_regex_for_combining_marks() {
        let combining = byte_level_piece("\u{301}");
        let mut combining_chars = combining.chars();
        let first = combining_chars.next().expect("first combining byte");
        let second = combining_chars.next().expect("second combining byte");
        assert!(combining_chars.next().is_none());

        let first_merge = format!("a{first}");
        let combined = format!("{first_merge}{second}");
        let tokens = vec![
            "a".to_string(),
            first.to_string(),
            second.to_string(),
            first_merge.clone(),
            combined,
        ];
        let merges = vec![format!("a {first}"), format!("{first_merge} {second}")];

        let qwen35 = Tokenizer::from_gguf_bpe(&tokens, &merges, Some("qwen35"), false)
            .expect("qwen35 tokenizer");
        let generic = Tokenizer::from_gguf_bpe(&tokens, &merges, Some("unknown"), false)
            .expect("generic tokenizer");

        assert_eq!(qwen35.encode("a\u{301}").expect("qwen35 encode"), vec![4]);
        assert_ne!(
            generic.encode("a\u{301}").expect("generic encode"),
            vec![4],
            "the generic GPT-2 regex splits the combining mark away from its letter"
        );
    }

    #[test]
    fn gguf_control_and_user_defined_tokens_are_atomic_without_id_changes() {
        let tokens = vec![
            "a".to_string(),
            "b".to_string(),
            "<|im_start|>".to_string(),
            "<think>".to_string(),
            "<tool_call>".to_string(),
        ];
        let mut tokenizer =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        tokenizer
            .register_gguf_token_types(&tokens, &[1, 1, 3, 4, 4])
            .expect("register token types");

        let ids = tokenizer
            .encode("a<|im_start|><think><tool_call>b")
            .expect("atomic encode");
        assert_eq!(ids, vec![0, 2, 3, 4, 1]);
        assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(2));
        assert_eq!(tokenizer.token_to_id("<think>"), Some(3));
        assert_eq!(tokenizer.token_to_id("<tool_call>"), Some(4));
        assert_eq!(tokenizer.vocab_size(), tokens.len());
        assert_eq!(
            tokenizer.decode(&ids).expect("skip control"),
            "a<think><tool_call>b"
        );
        assert_eq!(
            tokenizer
                .decode_with_special_tokens(&ids)
                .expect("include control"),
            "a<|im_start|><think><tool_call>b"
        );
    }

    #[test]
    fn gguf_bpe_rejects_duplicate_token_strings() {
        let tokens = vec!["a".to_string(), "a".to_string()];
        assert!(matches!(
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false),
            Err(Error::TokenizationError(message)) if message.contains("Duplicate GGUF tokenizer token")
        ));
    }

    #[test]
    fn gguf_token_type_registration_rejects_length_mismatch() {
        let tokens = vec!["a".to_string()];
        let mut tokenizer =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        assert!(matches!(
            tokenizer.register_gguf_token_types(&tokens, &[]),
            Err(Error::TokenizationError(message)) if message.contains("length mismatch")
        ));
    }

    #[test]
    fn incremental_decode_is_utf8_safe_and_matches_one_shot_with_flush() {
        let text = "café 中文 🌍 👨‍👩‍👧‍👦";
        let mut tokens = Vec::<String>::new();
        for byte in text.as_bytes() {
            let token = byte_level_char(*byte).to_string();
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        let tokenizer =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        let ids: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|byte| {
                let token = byte_level_char(*byte).to_string();
                tokenizer.token_to_id(&token).expect("byte token id")
            })
            .collect();

        let one_shot = tokenizer.decode(&ids).expect("one-shot decode");
        let mut decoder = IncrementalDecoder::new(true);
        let mut streamed = String::new();
        for id in &ids {
            let delta = tokenizer
                .decode_incrementally(&mut decoder, *id)
                .expect("incremental decode");
            assert!(!delta.ends_with('\u{fffd}'));
            streamed.push_str(&delta);
        }
        streamed.push_str(
            &tokenizer
                .finish_incremental_decode(&mut decoder)
                .expect("finish decode"),
        );

        assert_eq!(one_shot, text);
        assert_eq!(streamed, one_shot);
        assert_eq!(
            tokenizer
                .finish_incremental_decode(&mut decoder)
                .expect("idempotent finish"),
            ""
        );
        assert!(matches!(
            tokenizer.decode_incrementally(&mut decoder, ids[0]),
            Err(Error::TokenizationError(message)) if message.contains("after incremental decoding was finished")
        ));
    }

    #[test]
    fn incremental_flush_matches_incomplete_one_shot_decode() {
        let emoji = "🌍";
        let tokens: Vec<String> = emoji
            .as_bytes()
            .iter()
            .map(|byte| byte_level_char(*byte).to_string())
            .collect();
        let tokenizer =
            Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).expect("tokenizer");
        let incomplete_ids = vec![0, 1, 2];
        let mut decoder = IncrementalDecoder::new(true);
        let mut streamed = String::new();
        for id in &incomplete_ids {
            streamed.push_str(
                &tokenizer
                    .decode_incrementally(&mut decoder, *id)
                    .expect("incremental decode"),
            );
        }
        streamed.push_str(
            &tokenizer
                .finish_incremental_decode(&mut decoder)
                .expect("finish incomplete decode"),
        );

        assert_eq!(
            streamed,
            tokenizer
                .decode(&incomplete_ids)
                .expect("one-shot incomplete decode")
        );
    }
}

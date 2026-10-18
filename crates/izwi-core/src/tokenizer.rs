//! Text tokenization for Qwen3-TTS

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use serde::Deserialize;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokenizerSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::{step_decode_stream, DecodeStreamError};
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

#[derive(Clone)]
pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: SpecialTokens,
    /// Special tokens an incremental decode that skips special tokens
    /// leaves out of the text (see [`Self::skips_special_token`]).
    skipped_special_ids: Arc<HashSet<u32>>,
}

/// Markup the serving stack parses out of generated text: tool calls
/// (Qwen `<tool_call>` XML, LFM2 `<|tool_call_start|>` lists) and reasoning
/// blocks. A tokenizer that flags one of these special still writes it into
/// the text, so skipping special tokens never hides what a parser needs.
const PARSED_OUTPUT_MARKUP: [&str; 6] = [
    "<tool_call>",
    "</tool_call>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<think>",
    "</think>",
];

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
    /// The streamed text, or after a divergence the one-shot decode the
    /// stream continued from. Equals the one-shot decode once finished.
    emitted: String,
    divergences: u64,
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
            divergences: 0,
            finished: false,
        }
    }

    /// The one-shot decode of every token once
    /// [`Tokenizer::finish_incremental_decode`] has run.
    pub fn decoded_text(&self) -> &str {
        &self.emitted
    }

    /// Steps at which a decoder that is not prefix-stable rewrote text the
    /// stream had already emitted. Byte-level decoders never do.
    pub fn divergences(&self) -> u64 {
        self.divergences
    }
}

/// `current` after the longest character prefix it shares with `previous`.
fn text_after_shared_prefix(previous: &str, current: &str) -> String {
    let shared = previous
        .char_indices()
        .zip(current.chars())
        .find(|((_, emitted), next)| emitted != next)
        .map_or(previous.len().min(current.len()), |((at, _), _)| at);
    current.get(shared..).unwrap_or_default().to_owned()
}

/// The split regex transformers' `Qwen2Converter` gives `Qwen2TokenizerFast`
/// (unlike [`QWEN2_PRETOKENIZER_REGEX`], letters do not absorb marks).
const QWEN2_HF_PRETOKENIZER_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
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
                Ok(tokenizer) => return Ok(tokenizer.with_config_special_tokens(model_dir)),
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

    /// Like [`Self::from_path`], but a present `tokenizer.json` that fails to
    /// parse is an error instead of a fallback. The vocab.json + merges.txt
    /// fallback uses the GPT-2 ByteLevel split and no normalizer, so for a
    /// checkpoint whose `tokenizer.json` carries its own pre-tokenizer regex
    /// and NFC normalizer (Qwen) a silent fallback changes tokenization.
    pub fn from_path_requiring_tokenizer_json(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Self::from_tokenizer_json(&tokenizer_path)
                .map(|tokenizer| tokenizer.with_config_special_tokens(model_dir))
                .map_err(|err| {
                    Error::TokenizationError(format!(
                        "Failed to parse tokenizer.json at {}: {err}",
                        tokenizer_path.display()
                    ))
                });
        }
        Self::from_path(model_dir)
    }

    /// Also skip the tokens `tokenizer_config.json` flags special. A
    /// `tokenizer.json` need not flag them itself (a checkpoint can list its
    /// control tokens only in the config), and a skipping decode then wrote
    /// them into the reply (`<|image_pad|>`).
    fn with_config_special_tokens(mut self, model_dir: &Path) -> Self {
        let config = match load_tokenizer_config(model_dir) {
            Ok(Some(config)) => config,
            Ok(None) => return self,
            Err(err) => {
                warn!("Ignoring unreadable tokenizer_config.json in {model_dir:?}: {err}");
                return self;
            }
        };
        let flagged = config
            .added_tokens_decoder
            .iter()
            .filter(|(_, entry)| entry.special)
            .filter_map(|(id, entry)| {
                let id = id.parse::<u32>().ok()?;
                (self.inner.token_to_id(&entry.content) == Some(id)).then_some(id)
            })
            .collect::<Vec<_>>();
        self.skipped_special_ids = skipped_special_ids(&self.inner, flagged);
        self
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
        let qwen2 = config
            .as_ref()
            .and_then(|cfg| cfg.tokenizer_class.as_deref())
            == Some("Qwen2Tokenizer");
        let byte_level = if qwen2 {
            // As transformers builds Qwen2TokenizerFast from these files
            // (`Qwen2Converter`): NFC, then the Qwen2 split regex.
            inner.with_normalizer(Some(NFC));
            qwen_split_pre_tokenizer(&mut inner, QWEN2_HF_PRETOKENIZER_REGEX, add_prefix_space)?
        } else {
            let byte_level = ByteLevel::new(add_prefix_space, true, true);
            inner.with_pre_tokenizer(Some(byte_level));
            byte_level
        };
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
            qwen_split_pre_tokenizer(&mut inner, QWEN2_PRETOKENIZER_REGEX, add_prefix_space)?
        } else {
            let byte_level = ByteLevel::new(add_prefix_space, true, true);
            inner.with_pre_tokenizer(Some(byte_level));
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
        let kept = self.skipped_special_ids.iter().copied().collect::<Vec<_>>();
        self.skipped_special_ids = skipped_special_ids(&self.inner, kept);
        Ok(())
    }

    fn new_with_tokenizer(inner: HfTokenizer) -> Result<Self> {
        let special_tokens = SpecialTokens::default();
        let skipped_special_ids = skipped_special_ids(&inner, []);

        Ok(Self {
            inner,
            special_tokens,
            skipped_special_ids,
        })
    }

    /// Whether a decode that skips special tokens leaves `id` out of the
    /// text: a special token of the vocabulary, the GGUF control types or
    /// `tokenizer_config.json`, unless it is markup the serving stack parses
    /// (`PARSED_OUTPUT_MARKUP`).
    pub fn skips_special_token(&self, id: u32) -> bool {
        self.skipped_special_ids.contains(&id)
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
        if decoder.skip_special_tokens && self.skips_special_token(token_id) {
            return Ok(String::new());
        }
        // Skipped ids never enter the decoder, so every decode below keeps
        // the rest, parsed markup included.
        decoder.all_ids.push(token_id);
        let delta = match step_decode_stream(
            &self.inner,
            vec![token_id],
            false,
            &mut decoder.stream_ids,
            &mut decoder.prefix,
            &mut decoder.prefix_index,
        ) {
            Ok(delta) => delta.unwrap_or_default(),
            // ByteFallback decodes a run of byte tokens as one sequence, so a
            // stray byte turns a character the stream already emitted into
            // U+FFFD. Emitted text cannot be retracted: continue after the
            // longest prefix it shares with the one-shot decode, as
            // whole-sequence re-decoding streamed, and restart the window.
            Err(error) if error.downcast_ref::<DecodeStreamError>().is_some() => {
                let decoded = self
                    .inner
                    .decode(&decoder.all_ids, false)
                    .map_err(|e| Error::TokenizationError(e.to_string()))?;
                let delta = text_after_shared_prefix(&decoder.emitted, &decoded);
                decoder.emitted = decoded;
                decoder.stream_ids.clear();
                decoder.prefix.clear();
                decoder.prefix_index = 0;
                decoder.divergences += 1;
                return Ok(delta);
            }
            Err(error) => return Err(Error::TokenizationError(error.to_string())),
        };
        decoder.emitted.push_str(&delta);
        Ok(delta)
    }

    /// Finish an incremental decode and return the not-yet-emitted suffix.
    ///
    /// This is idempotent and deliberately compares against a one-shot decode,
    /// ensuring streamed deltas plus this suffix have exactly the same decoder
    /// semantics even when generation ends part-way through a byte sequence.
    /// A decoder that is not prefix-stable can have rewritten emitted text;
    /// the suffix then follows the longest shared prefix.
    pub fn finish_incremental_decode(&self, decoder: &mut IncrementalDecoder) -> Result<String> {
        if decoder.finished {
            return Ok(String::new());
        }
        let decoded = self
            .inner
            .decode(&decoder.all_ids, false)
            .map_err(|e| Error::TokenizationError(e.to_string()))?;
        let suffix = match decoded.strip_prefix(&decoder.emitted) {
            Some(suffix) => suffix.to_string(),
            None => {
                decoder.divergences += 1;
                text_after_shared_prefix(&decoder.emitted, &decoded)
            }
        };
        decoder.emitted = decoded;
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

    /// The vocabulary piece for `id`, before any decoder normalization.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    pub fn format_tts_prompt(&self, text: &str, speaker: Option<&str>) -> String {
        let speaker_tag = speaker.unwrap_or("default");
        format!("[speaker:{}] {}", speaker_tag, text)
    }
}

/// The vocabulary's special tokens plus `extra`, less parsed markup.
fn skipped_special_ids(
    inner: &HfTokenizer,
    extra: impl IntoIterator<Item = u32>,
) -> Arc<HashSet<u32>> {
    let parsed = |id: u32| {
        inner
            .id_to_token(id)
            .is_some_and(|token| PARSED_OUTPUT_MARKUP.contains(&token.as_str()))
    };
    let special = inner
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| token.special)
        .map(|(id, _)| id);
    Arc::new(special.chain(extra).filter(|id| !parsed(*id)).collect())
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default)]
    add_prefix_space: Option<bool>,
    #[serde(default)]
    tokenizer_class: Option<String>,
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

/// Pre-tokenize with a Qwen split `regex`, then byte-level without its own
/// regex. Returns the byte-level step for the decoder.
fn qwen_split_pre_tokenizer(
    inner: &mut HfTokenizer,
    regex: &str,
    add_prefix_space: bool,
) -> Result<ByteLevel> {
    let split = Split::new(
        SplitPattern::Regex(regex.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| Error::TokenizationError(format!("Invalid Qwen2 split regex: {e}")))?;
    let byte_level = ByteLevel::new(add_prefix_space, false, false);
    inner.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ])));
    Ok(byte_level)
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

    /// QFA-6: vocab.json + merges.txt with `tokenizer_class: Qwen2Tokenizer`
    /// (the aligner and Qwen3-TTS ship no tokenizer.json) tokenize like
    /// Qwen2TokenizerFast: NFC first, then the Qwen2 split, which keeps a
    /// leading `-` or `'T` with the letters after it. Without the class the
    /// GPT-2 split stays.
    #[test]
    fn qwen2_tokenizer_class_uses_the_qwen2_split_and_nfc() {
        let byte_ids = (0..=u8::MAX)
            .map(|byte| (byte_level_char(byte).to_string(), u32::from(byte)))
            .collect::<HashMap<_, _>>();
        let mut vocab = byte_ids.clone();
        vocab.insert("'T".to_string(), 256);
        vocab.insert("-k".to_string(), 257);
        let merges = "#version: 0.2\n' T\n- k\n";
        let bytes = |text: &str| {
            text.bytes()
                .map(|byte| byte_ids[&byte_level_char(byte).to_string()])
                .collect::<Vec<_>>()
        };
        let load = |class: Option<&str>| {
            let dir = std::env::temp_dir().join(format!("izwi-tokenizer-qwen2-{}", Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("vocab.json"), serde_json::to_vec(&vocab).unwrap()).unwrap();
            fs::write(dir.join("merges.txt"), merges).unwrap();
            let mut config = serde_json::json!({ "add_prefix_space": false });
            if let Some(class) = class {
                config["tokenizer_class"] = class.into();
            }
            fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
            let tokenizer = Tokenizer::from_path(&dir).expect("vocab + merges tokenizer");
            let _ = fs::remove_dir_all(&dir);
            tokenizer
        };

        let qwen2 = load(Some("Qwen2Tokenizer"));
        assert_eq!(
            qwen2.encode("DON'T").unwrap(),
            [bytes("DON"), vec![256]].concat()
        );
        assert_eq!(
            qwen2.encode("well-known").unwrap(),
            [bytes("well"), vec![257], bytes("nown")].concat()
        );
        assert_eq!(
            qwen2.encode("cafe\u{301}").unwrap(),
            bytes("caf\u{e9}"),
            "NFD input is composed first"
        );

        let generic = load(None);
        assert_eq!(generic.encode("DON'T").unwrap(), bytes("DON'T"));
        assert_eq!(generic.encode("well-known").unwrap(), bytes("well-known"));
        assert_eq!(generic.encode("cafe\u{301}").unwrap(), bytes("cafe\u{301}"));
    }

    #[test]
    fn requiring_tokenizer_json_rejects_a_corrupt_file_instead_of_falling_back() {
        let dir = std::env::temp_dir().join(format!("izwi-tokenizer-strict-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("tokenizer.json"), b"{ not json").unwrap();
        fs::write(dir.join("vocab.json"), br#"{"a": 0, "b": 1}"#).unwrap();
        fs::write(dir.join("merges.txt"), b"#version: 0.2\n").unwrap();

        let err = match Tokenizer::from_path_requiring_tokenizer_json(&dir) {
            Ok(_) => panic!("a corrupt tokenizer.json must not fall back"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("tokenizer.json"), "{err}");
        assert!(
            Tokenizer::from_path(&dir).is_ok(),
            "the lenient loader keeps its vocab+merges fallback"
        );
        let _ = fs::remove_dir_all(&dir);
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

    /// Every id streamed through an incremental decode, concatenated with
    /// its finish, and the decoder's one-shot text.
    fn incremental_text(tokenizer: &Tokenizer, ids: &[u32], skip: bool) -> (String, String) {
        let mut decoder = IncrementalDecoder::new(skip);
        let mut text = String::new();
        for &id in ids {
            text.push_str(&tokenizer.decode_incrementally(&mut decoder, id).unwrap());
        }
        text.push_str(&tokenizer.finish_incremental_decode(&mut decoder).unwrap());
        (text, decoder.decoded_text().to_string())
    }

    /// A tokenizer.json that lists its control tokens only as plain
    /// vocabulary, with the special flags in tokenizer_config.json (the
    /// serving fixture's Qwen3.8 checkpoint): a skipping decode wrote
    /// `<|image_pad|>` into the reply. Markup a parser needs stays even when
    /// flagged special (Gemma-style `<tool_call>`).
    #[test]
    fn skipping_decode_drops_config_special_tokens_and_keeps_parsed_markup() {
        let dir = std::env::temp_dir().join(format!("izwi-tokenizer-special-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let tokenizer_json = r#"{
            "version": "1.0",
            "added_tokens": [
                {"id": 4, "content": "<tool_call>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 5, "content": "<|im_end|>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"hi": 0, "there": 1, "<unk>": 2,
                "<|image_pad|>": 3, "<tool_call>": 4, "<|im_end|>": 5}, "unk_token": "<unk>"}
        }"#;
        fs::write(dir.join("tokenizer.json"), tokenizer_json).unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {
                "3": {"content": "<|image_pad|>", "special": true},
                "4": {"content": "<tool_call>", "special": true}
            }}"#,
        )
        .unwrap();
        let ids = [0, 3, 4, 1, 5];
        for tokenizer in [
            Tokenizer::from_path(&dir).unwrap(),
            Tokenizer::from_path_requiring_tokenizer_json(&dir).unwrap(),
        ] {
            assert!(tokenizer.skips_special_token(3));
            assert!(tokenizer.skips_special_token(5));
            assert!(
                !tokenizer.skips_special_token(4),
                "parsed markup is never skipped"
            );
            let (streamed, whole) = incremental_text(&tokenizer, &ids, true);
            assert_eq!(streamed, "hi <tool_call> there");
            assert_eq!(whole, streamed);
            let (streamed, whole) = incremental_text(&tokenizer, &ids, false);
            assert_eq!(streamed, "hi <|image_pad|> <tool_call> there <|im_end|>");
            assert_eq!(whole, streamed);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// GGUF control tokens are skipped and user-defined markup kept by the
    /// incremental decode as by the one-shot one.
    #[test]
    fn skipping_decode_follows_gguf_token_types() {
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
        let ids = [0, 2, 3, 4, 1];
        assert_eq!(
            incremental_text(&tokenizer, &ids, true).0,
            "a<think><tool_call>b"
        );
        assert_eq!(
            incremental_text(&tokenizer, &ids, false).0,
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
    fn incremental_decode_continues_when_byte_fallback_rewrites_emitted_text() {
        // SentencePiece-style byte fallback decodes a run of byte tokens as
        // one sequence: a stray continuation byte after "é" turns the whole
        // run, including the emitted "é", into U+FFFD.
        let tokenizer = Tokenizer::from_hf_json_bytes(
            br#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"WordLevel","unk_token":"b",
                "vocab":{"<0xC3>":0,"<0xA9>":1,"b":2}}
        }"#,
        )
        .expect("tokenizer");
        let ids = [0, 1, 1, 2, 0, 1];
        let mut decoder = IncrementalDecoder::new(true);
        let deltas = ids
            .iter()
            .map(|id| {
                tokenizer
                    .decode_incrementally(&mut decoder, *id)
                    .expect("a rewrite is not an error")
            })
            .collect::<Vec<_>>();
        let suffix = tokenizer
            .finish_incremental_decode(&mut decoder)
            .expect("finish after a rewrite");
        let one_shot = tokenizer.decode(&ids).expect("one-shot decode");
        assert_eq!(one_shot, "\u{FFFD}\u{FFFD}\u{FFFD}bé");
        // The stream continues after the longest prefix it shares with the
        // one-shot decode, as whole-sequence re-decoding streamed.
        assert_eq!(deltas, ["", "é", "", "\u{FFFD}\u{FFFD}\u{FFFD}b", "", "é"]);
        assert_eq!(suffix, "");
        assert_eq!(decoder.decoded_text(), one_shot);
        assert_eq!(decoder.divergences(), 1);
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

//! Tokenizer wrapper for the retained Qwen speech/aligner stack.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::models::shared::weights::gguf::GgufLoader;
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

        let id_for = |token: &str| -> Option<u32> {
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
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");
        let pad = config
            .pad_token
            .as_deref()
            .and_then(id_for)
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

    pub fn load_from_gguf(
        model_dir: &Path,
        loader: &GgufLoader,
        expected_vocab_size: usize,
    ) -> Result<Self> {
        if model_dir.join("tokenizer.json").exists()
            && model_dir.join("tokenizer_config.json").exists()
        {
            if let Ok(sidecar) = Self::load(model_dir, expected_vocab_size) {
                return Ok(sidecar);
            }
        }

        let tokenizer_json = loader
            .get_metadata_string("tokenizer.huggingface.json")
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Missing GGUF tokenizer metadata: tokenizer.huggingface.json".to_string(),
                )
            })?;
        let architecture = loader.get_metadata_string("general.architecture");
        Self::from_gguf_tokenizer_json(
            model_dir,
            &tokenizer_json,
            architecture.as_deref(),
            |key| metadata_token_id(loader, key),
            expected_vocab_size,
        )
    }

    fn from_gguf_tokenizer_json(
        model_dir: &Path,
        tokenizer_json: &str,
        architecture: Option<&str>,
        metadata_token_id: impl Fn(&str) -> Option<u32>,
        expected_vocab_size: usize,
    ) -> Result<Self> {
        let mut tokenizer_value: Value = serde_json::from_str(tokenizer_json).map_err(|e| {
            Error::TokenizationError(format!(
                "Failed to parse GGUF tokenizer.huggingface.json metadata: {e}"
            ))
        })?;
        let injected = architecture == Some("qwen3_asr")
            && inject_qwen3_asr_added_tokens(&mut tokenizer_value);
        let patched_json;
        let tokenizer_json = if injected {
            patched_json = tokenizer_value.to_string();
            patched_json.as_str()
        } else {
            tokenizer_json
        };

        let inner =
            match Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size)) {
                Ok(tokenizer) => tokenizer,
                Err(_) => Tokenizer::from_hf_json_bytes(tokenizer_json.as_bytes())?,
            };
        let vocab_size = inner.vocab_size();
        if injected && inner.token_to_id(QWEN3_ASR_TEXT_TOKEN) != Some(QWEN3_ASR_TEXT_TOKEN_ID) {
            return Err(Error::TokenizationError(format!(
                "Injected Qwen3-ASR added tokens did not resolve {QWEN3_ASR_TEXT_TOKEN} to \
                 {QWEN3_ASR_TEXT_TOKEN_ID}"
            )));
        }

        let token_to_id = collect_token_ids_from_hf_json(&tokenizer_value);
        if token_to_id.is_empty() {
            return Err(Error::TokenizationError(
                "GGUF tokenizer metadata did not include any token IDs".to_string(),
            ));
        }

        let id_for = |token: &str| token_to_id.get(token).copied();
        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let audio_start = id_for("<|audio_start|>")
            .or_else(|| metadata_token_id("qwen3_asr.audio_start_token_id"))
            .ok_or_else(|| {
                Error::TokenizationError(
                    "Missing <|audio_start|> token id (and qwen3_asr.audio_start_token_id metadata)"
                        .to_string(),
                )
            })?;
        let audio_end = id_for("<|audio_end|>")
            .or_else(|| metadata_token_id("qwen3_asr.audio_end_token_id"))
            .ok_or_else(|| {
                Error::TokenizationError(
                    "Missing <|audio_end|> token id (and qwen3_asr.audio_end_token_id metadata)"
                        .to_string(),
                )
            })?;
        let audio_token = id_for("<|audio_pad|>")
            .or_else(|| metadata_token_id("qwen3_asr.audio_token_id"))
            .ok_or_else(|| {
                Error::TokenizationError(
                    "Missing <|audio_pad|> token id (and qwen3_asr.audio_token_id metadata)"
                        .to_string(),
                )
            })?;
        let timestamp = id_for("<timestamp>");
        let asr_text = id_for("<asr_text>");
        let fim_prefix = id_for("<|fim_prefix|>");
        let fim_middle = id_for("<|fim_middle|>");
        let fim_suffix = id_for("<|fim_suffix|>");
        let fim_pad = id_for("<|fim_pad|>");

        let eos = read_token_literal(tokenizer_value.get("eos_token"))
            .as_deref()
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");
        let pad = read_token_literal(tokenizer_value.get("pad_token"))
            .as_deref()
            .and_then(id_for)
            .unwrap_or(eos);

        let timestamp_token_indices: HashMap<u32, u32> = token_to_id
            .iter()
            .filter_map(|(token, id)| parse_timestamp_token_index(token).map(|idx| (*id, idx)))
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

/// Added tokens 151669..=151704 of the official Qwen/Qwen3-ASR-0.6B (and
/// 1.7B) `tokenizer_config.json`, as (content, special). The published GGUFs
/// embed a Qwen3 text tokenizer whose added tokens stop at 151668, so
/// `<asr_text>` (151704, not special) is missing: the forced-language prompt
/// loses its marker and auto-language output cannot be split the way
/// upstream `parse_asr_output` splits it.
const QWEN3_ASR_ADDED_TOKENS: [(&str, bool); 36] = [
    ("<|audio_start|>", true),
    ("<|audio_end|>", true),
    ("<tts_pad>", true),
    ("<tts_text_bos>", true),
    ("<tts_text_eod>", true),
    ("<tts_text_bos_single>", true),
    ("<non_speech>", false),
    ("<|audio_pad|>", true),
    ("<blank1>", true),
    ("<blank2>", true),
    ("<blank3>", true),
    ("<blank4>", true),
    ("<blank5>", true),
    ("<blank6>", true),
    ("<blank7>", true),
    ("<blank8>", true),
    ("<blank9>", true),
    ("<blank10>", true),
    ("<blank11>", true),
    ("<blank12>", true),
    ("<blank13>", true),
    ("<blank14>", true),
    ("<blank15>", true),
    ("<blank16>", true),
    ("<blank17>", true),
    ("<blank18>", true),
    ("<blank19>", true),
    ("<blank20>", true),
    ("<blank21>", true),
    ("<blank22>", true),
    ("<blank23>", true),
    ("<blank24>", true),
    ("<blank25>", true),
    ("<blank26>", true),
    ("<blank27>", true),
    ("<asr_text>", false),
];
const QWEN3_ASR_FIRST_ADDED_TOKEN_ID: u32 = 151_669;
const QWEN3_ASR_TEXT_TOKEN: &str = "<asr_text>";
const QWEN3_ASR_TEXT_TOKEN_ID: u32 = 151_704;

/// Appends [`QWEN3_ASR_ADDED_TOKENS`] to a GGUF-embedded HF tokenizer JSON
/// whose ids stop right before them. HF `tokenizers` assigns added-token ids
/// in order, so the table is only appended when it continues the existing ids
/// exactly; a tokenizer that already has `<asr_text>` or a different id
/// layout is left untouched.
fn inject_qwen3_asr_added_tokens(tokenizer_value: &mut Value) -> bool {
    let token_to_id = collect_token_ids_from_hf_json(tokenizer_value);
    if token_to_id.contains_key(QWEN3_ASR_TEXT_TOKEN)
        || token_to_id.values().max().copied() != Some(QWEN3_ASR_FIRST_ADDED_TOKEN_ID - 1)
    {
        return false;
    }
    let Some(added) = tokenizer_value
        .get_mut("added_tokens")
        .and_then(Value::as_array_mut)
    else {
        return false;
    };
    for (offset, (content, special)) in QWEN3_ASR_ADDED_TOKENS.iter().enumerate() {
        added.push(serde_json::json!({
            "id": QWEN3_ASR_FIRST_ADDED_TOKEN_ID + offset as u32,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": special,
        }));
    }
    true
}

fn parse_timestamp_token_index(token: &str) -> Option<u32> {
    let trimmed = token.trim();
    if !trimmed.starts_with("<|timestamp_") || !trimmed.ends_with("|>") {
        return None;
    }
    let inner = trimmed.strip_prefix("<|timestamp_")?.strip_suffix("|>")?;
    inner.parse::<u32>().ok()
}

fn collect_token_ids_from_hf_json(tokenizer_value: &Value) -> HashMap<String, u32> {
    let mut token_to_id = HashMap::new();

    if let Some(object) = tokenizer_value
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    {
        for (id, entry) in object {
            let Some(token_id) = id.parse::<u32>().ok() else {
                continue;
            };
            let Some(content) = entry.get("content").and_then(Value::as_str) else {
                continue;
            };
            token_to_id.insert(content.to_string(), token_id);
        }
    }

    if let Some(entries) = tokenizer_value
        .get("added_tokens")
        .and_then(Value::as_array)
    {
        for entry in entries {
            let Some(token_id) = entry.get("id").and_then(value_to_u32) else {
                continue;
            };
            let Some(content) = entry.get("content").and_then(Value::as_str) else {
                continue;
            };
            token_to_id.insert(content.to_string(), token_id);
        }
    }

    if let Some(vocab) = tokenizer_value
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(Value::as_object)
    {
        for (token, id) in vocab {
            let Some(token_id) = value_to_u32(id) else {
                continue;
            };
            token_to_id.entry(token.clone()).or_insert(token_id);
        }
    }

    token_to_id
}

fn read_token_literal(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(raw) = value.as_str() {
        return Some(raw.to_string());
    }
    value
        .get("content")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn value_to_u32(value: &Value) -> Option<u32> {
    match value {
        Value::Number(number) => number.as_u64().and_then(|raw| u32::try_from(raw).ok()),
        Value::String(raw) => raw.parse::<u32>().ok(),
        _ => None,
    }
}

fn metadata_token_id(loader: &GgufLoader, key: &str) -> Option<u32> {
    loader
        .get_metadata_u64(key)
        .and_then(|raw| u32::try_from(raw).ok())
}

/// A tokenizer built from the published GGUF's embedded tokenizer JSON
/// (`tests/fixtures/qwen3-asr-gguf-tokenizer-skeleton.json`, vocab refilled
/// with the 256 byte-level symbols plus fillers up to id 151642).
#[cfg(test)]
pub(super) fn gguf_shaped_test_tokenizer(architecture: Option<&str>) -> Result<AsrTokenizer> {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen3-asr-gguf-tokenizer-skeleton.json");
    let mut value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, symbol) in alphabet.iter().enumerate() {
        vocab.insert(symbol.to_string(), id.into());
    }
    for id in alphabet.len()..151_643 {
        vocab.insert(format!("<filler_{id}>"), id.into());
    }
    value["model"]["vocab"] = Value::Object(vocab);
    // HF tokenizers rejects keys it does not know.
    value.as_object_mut().unwrap().remove("_comment");
    let metadata = |key: &str| match key {
        "qwen3_asr.audio_start_token_id" => Some(151_669),
        "qwen3_asr.audio_end_token_id" => Some(151_670),
        "qwen3_asr.audio_token_id" => Some(151_676),
        _ => None,
    };
    AsrTokenizer::from_gguf_tokenizer_json(
        Path::new("/nonexistent/izwi-qwen3-asr-gguf-without-sidecars"),
        &value.to_string(),
        architecture,
        metadata,
        151_936,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QFA-6 on the real aligner tokenizer (vocab.json + merges.txt, no
    /// tokenizer.json): ids equal Qwen2TokenizerFast's. Goldens from
    /// `tokenizers` 0.22.2 with transformers 4.57's `Qwen2Converter` pipeline
    /// (NFC, Qwen2 split regex, byte-level BPE) over the same files,
    /// 2026-10-15. The first six differed under the GPT-2 split.
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint"]
    fn aligner_tokenizer_matches_qwen2_tokenizer_fast() {
        let models = std::env::var("IZWI_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::data_local_dir()
                    .expect("data dir")
                    .join("izwi")
                    .join("models")
            });
        let tokenizer = AsrTokenizer::load(&models.join("Qwen3-ForcedAligner-0.6B"), 152_064)
            .expect("aligner tokenizer");
        let goldens: [(&str, &[u32]); 12] = [
            ("well-known", &[9157, 21309]),
            ("e-mail", &[68, 11468]),
            ("DON'T", &[84641, 17323]),
            ("IT'S", &[952, 13272]),
            ("rock'n'roll", &[20521, 43786, 6, 1100]),
            ("cafe\u{301}", &[924, 58858]),
            ("don't", &[15007, 944]),
            ("COVID-19", &[77963, 12, 16, 24]),
            ("2026", &[17, 15, 17, 21]),
            ("你好", &[108386]),
            ("안녕", &[126246, 144370]),
            ("AIautonomous", &[15469, 2717, 29846]),
        ];
        for (word, ids) in goldens {
            assert_eq!(tokenizer.encode_text(word).unwrap(), ids, "{word}");
        }
    }

    #[test]
    fn gguf_qwen3_asr_tokenizer_gains_the_official_added_tokens() {
        let tokenizer = gguf_shaped_test_tokenizer(Some("qwen3_asr")).unwrap();
        let specials = tokenizer.specials();
        assert_eq!(specials.asr_text, Some(151_704));
        assert_eq!(specials.audio_start, 151_669);
        assert_eq!(specials.audio_end, 151_670);
        assert_eq!(specials.audio_token, 151_676);
        assert_eq!(tokenizer.vocab_size(), 151_705);
        assert_eq!(tokenizer.inner.token_to_id("<asr_text>"), Some(151_704));
        assert_eq!(tokenizer.inner.token_to_id("<blank1>"), Some(151_677));
        // Non-special like upstream, so a skip-special decode keeps it.
        assert_eq!(tokenizer.decode_text(&[151_704]).unwrap(), "<asr_text>");
        assert_eq!(tokenizer.decode_text(&[151_669, 151_676]).unwrap(), "");
        let mut ids = tokenizer.encode_text("language English").unwrap();
        ids.push(151_704);
        ids.extend(tokenizer.encode_text("你好").unwrap());
        assert_eq!(
            tokenizer.decode_text(&ids).unwrap(),
            "language English<asr_text>你好"
        );
    }

    #[test]
    fn gguf_tokenizer_of_another_architecture_is_not_patched() {
        let tokenizer = gguf_shaped_test_tokenizer(Some("qwen3")).unwrap();
        assert_eq!(tokenizer.specials().asr_text, None);
        assert_eq!(tokenizer.vocab_size(), 151_669);
    }

    #[test]
    fn qwen3_asr_added_tokens_are_not_injected_twice_or_out_of_order() {
        let mut complete = serde_json::json!({
            "added_tokens": [{ "id": 151_704, "content": "<asr_text>" }],
            "model": { "vocab": { "a": 0 } }
        });
        assert!(!inject_qwen3_asr_added_tokens(&mut complete));
        let mut gapped = serde_json::json!({
            "added_tokens": [{ "id": 151_660, "content": "<|fim_middle|>" }],
            "model": { "vocab": { "a": 0 } }
        });
        assert!(!inject_qwen3_asr_added_tokens(&mut gapped));
    }

    #[test]
    fn read_token_literal_accepts_string_and_object() {
        let string_value = serde_json::json!("<|im_end|>");
        assert_eq!(
            read_token_literal(Some(&string_value)),
            Some("<|im_end|>".to_string())
        );

        let object_value = serde_json::json!({ "content": "<|audio_pad|>" });
        assert_eq!(
            read_token_literal(Some(&object_value)),
            Some("<|audio_pad|>".to_string())
        );
    }

    #[test]
    fn collect_token_ids_from_hf_json_reads_added_tokens_and_vocab() {
        let value = serde_json::json!({
            "added_tokens": [
                { "id": 151644, "content": "<|im_start|>" },
                { "id": 151645, "content": "<|im_end|>" },
                { "id": 151676, "content": "<|audio_pad|>" }
            ],
            "model": {
                "vocab": {
                    "<|timestamp_10|>": 151900
                }
            }
        });

        let map = collect_token_ids_from_hf_json(&value);
        assert_eq!(map.get("<|im_start|>").copied(), Some(151644));
        assert_eq!(map.get("<|im_end|>").copied(), Some(151645));
        assert_eq!(map.get("<|audio_pad|>").copied(), Some(151676));
        assert_eq!(map.get("<|timestamp_10|>").copied(), Some(151900));
    }

    #[test]
    fn parse_timestamp_token_index_parses_timestamp_ids() {
        assert_eq!(parse_timestamp_token_index("<|timestamp_42|>"), Some(42));
        assert_eq!(parse_timestamp_token_index("<|im_end|>"), None);
    }

    #[test]
    fn value_to_u32_supports_numbers_and_strings() {
        assert_eq!(value_to_u32(&serde_json::json!(42)), Some(42));
        assert_eq!(value_to_u32(&serde_json::json!("99")), Some(99));
        assert_eq!(value_to_u32(&serde_json::json!(null)), None);
    }
}

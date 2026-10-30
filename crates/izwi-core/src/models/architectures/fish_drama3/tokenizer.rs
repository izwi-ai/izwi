//! The Drama 3 tokenizer: the published `tokenizer.json`, checked against
//! the config, encoding one prompt literal at a time.
//!
//! Upstream tokenizes every literal of the prompt on its own
//! (`generate.py:107-108,177-183`): tokenizing the joined string changes ids
//! at the seams. Added tokens (`<|im_start|>`, `<|speaker:N|>`, …) are
//! recognised inside a literal, a space before one is its own token (220),
//! and nothing is added around it: the post-processor is a plain ByteLevel
//! one, so upstream's `encode(segment)` (special tokens on) and
//! [`FishDrama3Tokenizer::encode_segment`] (off) give the same ids. There is
//! no BOS.
//!
//! `chat_template.jinja` is the stock Qwen3.5 chat template, not the speech
//! layout (`prompt.rs` writes that), so it is never read.

use std::path::Path;

use tokenizers::tokenizer::PostProcessor;
use tokenizers::Tokenizer;

use super::config::{FishDrama3Config, FishDrama3ConfigPolicy, FishDrama3TokenIds, SPEAKER_TAGS};
use crate::error::{Error, Result};

pub const TOKENIZER_FILE: &str = "tokenizer.json";
/// `pad_token_id`, not S2's `<|pad|>`.
pub const PAD_TOKEN: &str = "<|endoftext|>";
pub const IM_START_TOKEN: &str = "<|im_start|>";
/// Eos: the only stop token.
pub const IM_END_TOKEN: &str = "<|im_end|>";
pub const VOICE_TOKEN: &str = "<|voice|>";
pub const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";
/// `<|im_start|>`'s id in the published vocabulary (config.json does not
/// carry it).
pub const PUBLISHED_IM_START: u32 = 248_045;

/// `<|speaker:index|>`.
pub fn speaker_token(index: usize) -> String {
    format!("<|speaker:{index}|>")
}

/// `<|semantic:code|>`.
pub fn semantic_token(code: usize) -> String {
    format!("<|semantic:{code}|>")
}

/// The checked tokenizer and what the prompt rows need from the config.
pub struct FishDrama3Tokenizer {
    tokenizer: Tokenizer,
    ids: FishDrama3TokenIds,
    im_start: u32,
    codebooks: usize,
}

impl std::fmt::Debug for FishDrama3Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Tokenizer")
            .field("ids", &self.ids)
            .field("im_start", &self.im_start)
            .field("codebooks", &self.codebooks)
            .finish_non_exhaustive()
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!("Fish Audio Drama 3 tokenizer: {}", message.into()))
}

impl FishDrama3Tokenizer {
    /// Reads `model_dir/tokenizer.json`.
    pub fn load(model_dir: &Path, config: &FishDrama3Config) -> Result<Self> {
        Self::from_file(&model_dir.join(TOKENIZER_FILE), config)
    }

    pub fn from_file(path: &Path, config: &FishDrama3Config) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|err| invalid(format!("cannot read {}: {err}", path.display())))?;
        Self::from_tokenizer(tokenizer, config)
    }

    /// Checks `tokenizer` against `config`: every id the prompt writes or
    /// the model emits must be the added token the config says it is, and
    /// nothing may be added around a literal.
    pub fn from_tokenizer(tokenizer: Tokenizer, config: &FishDrama3Config) -> Result<Self> {
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return Err(invalid(
                "tokenizer.json sets truncation or padding; a prompt literal must encode whole",
            ));
        }
        let added = tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false));
        if added != 0 {
            return Err(invalid(format!(
                "the post-processor adds {added} tokens to a sequence; this family has no BOS or EOS"
            )));
        }
        let vocab = tokenizer.get_vocab_size(true);
        if vocab > config.trunk.vocab {
            return Err(invalid(format!(
                "{vocab} tokens do not fit the {}-row embedding",
                config.trunk.vocab
            )));
        }

        let ids = config.ids;
        let decoder = tokenizer.get_added_tokens_decoder();
        // Each must be an added token, so a literal that spells it selects
        // it instead of being split by BPE.
        let expect = |token: &str, id: u32| -> Result<()> {
            let found = tokenizer.token_to_id(token);
            let added = decoder.get(&id).is_some_and(|entry| entry.content == token);
            if found == Some(id) && added {
                Ok(())
            } else {
                Err(invalid(format!(
                    "{token} must be the added token {id}; found {found:?}"
                )))
            }
        };
        expect(PAD_TOKEN, ids.pad)?;
        expect(IM_END_TOKEN, ids.eos)?;
        expect(VOICE_TOKEN, ids.voice)?;
        expect(AUDIO_PAD_TOKEN, ids.audio_pad)?;
        for index in 0..SPEAKER_TAGS {
            expect(&speaker_token(index), ids.speaker_first + index as u32)?;
        }
        // The band is contiguous: `<|semantic:N|>` is `semantic_start + N`.
        for code in 0..ids.semantic_codes() {
            expect(&semantic_token(code), ids.semantic_start + code as u32)?;
        }
        let im_start = match config.policy {
            FishDrama3ConfigPolicy::Published => PUBLISHED_IM_START,
            FishDrama3ConfigPolicy::Synthetic => tokenizer
                .token_to_id(IM_START_TOKEN)
                .ok_or_else(|| invalid(format!("{IM_START_TOKEN} is missing")))?,
        };
        // An added token's id has one content, so no two of these collide.
        expect(IM_START_TOKEN, im_start)?;

        Ok(Self {
            tokenizer,
            ids,
            im_start,
            codebooks: config.tower.codebooks,
        })
    }

    pub fn ids(&self) -> &FishDrama3TokenIds {
        &self.ids
    }

    pub fn im_start(&self) -> u32 {
        self.im_start
    }

    /// Codebooks per frame: a prompt row is `1 + codebooks` wide.
    pub fn codebooks(&self) -> usize {
        self.codebooks
    }

    /// The ids of one prompt literal, added tokens recognised, nothing added
    /// around it.
    pub fn encode_segment(&self, literal: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(literal, false).map_err(|err| {
            Error::TokenizationError(format!("Fish Audio Drama 3 prompt literal: {err}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::models::architectures::fish_drama3::config::tests::synthetic;

    /// Word ids of the synthetic tokenizer besides the control tokens.
    pub(crate) const WORDS: [(&str, u32); 6] = [
        ("system", 1),
        ("user", 6),
        ("assistant", 7),
        ("hello", 8),
        ("speech", 9),
        ("Hi", 10),
    ];

    /// A WordLevel tokenizer laid out like the synthetic config's ids: pad 2,
    /// audio pad 3, eos 4, `<|im_start|>` 5, `<|voice|>` 22, speakers
    /// 24..=39, the semantic band 40..=47.
    pub(crate) fn synthetic_json() -> Value {
        let mut controls: Vec<(String, u32)> = vec![
            (PAD_TOKEN.into(), 2),
            (AUDIO_PAD_TOKEN.into(), 3),
            (IM_END_TOKEN.into(), 4),
            (IM_START_TOKEN.into(), 5),
            (VOICE_TOKEN.into(), 22),
            ("<|interleave|>".into(), 23),
        ];
        controls.extend((0..16).map(|index| (speaker_token(index), 24 + index as u32)));
        controls.extend((0..8).map(|code| (semantic_token(code), 40 + code as u32)));
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".into(), 0.into());
        for (word, id) in WORDS {
            vocab.insert(word.into(), id.into());
        }
        for (token, id) in &controls {
            vocab.insert(token.clone(), (*id).into());
        }
        let added = controls
            .iter()
            .map(|(token, id)| {
                json!({
                    "id": id, "content": token, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect::<Vec<_>>();
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
        })
    }

    fn load(json: &Value) -> Result<FishDrama3Tokenizer> {
        let tokenizer = Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap();
        FishDrama3Tokenizer::from_tokenizer(tokenizer, &synthetic())
    }

    pub(crate) fn synthetic_tokenizer() -> FishDrama3Tokenizer {
        load(&synthetic_json()).unwrap()
    }

    #[test]
    fn a_tokenizer_laid_out_like_the_config_loads_and_recognises_added_tokens() {
        let tokenizer = synthetic_tokenizer();
        assert_eq!(tokenizer.im_start(), 5);
        assert_eq!(tokenizer.codebooks(), 3);
        assert_eq!(tokenizer.ids().voice, 22);
        // Added tokens split out of a literal; nothing is added around it.
        assert_eq!(
            tokenizer
                .encode_segment("<|im_start|>assistant\n<|voice|>")
                .unwrap(),
            vec![5, 7, 22]
        );
        assert_eq!(
            tokenizer.encode_segment("<|speaker:15|>hello").unwrap(),
            vec![39, 8]
        );
        assert_eq!(tokenizer.encode_segment("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn a_tokenizer_that_disagrees_with_the_config_fails_closed() {
        type Edit = fn(&mut Value);
        let edits: &[(&str, Edit)] = &[
            // `<|speaker:3|>` and `<|speaker:4|>` swapped.
            ("speaker order", |t| {
                t["model"]["vocab"]["<|speaker:3|>"] = 28.into();
                t["model"]["vocab"]["<|speaker:4|>"] = 27.into();
                t["added_tokens"][9]["content"] = "<|speaker:4|>".into();
                t["added_tokens"][10]["content"] = "<|speaker:3|>".into();
            }),
            // In the vocabulary but not an added token: BPE would split it.
            ("voice not added", |t| {
                t["added_tokens"].as_array_mut().unwrap().remove(4);
            }),
            ("semantic band gap", |t| {
                let vocab = t["model"]["vocab"].as_object_mut().unwrap();
                vocab.remove("<|semantic:5|>");
                vocab.insert("<|semantic:5|>".into(), 50.into());
                t["added_tokens"][27]["id"] = 50.into();
            }),
            // S2's pad.
            ("pad string", |t| {
                let vocab = t["model"]["vocab"].as_object_mut().unwrap();
                vocab.remove(PAD_TOKEN);
                vocab.insert("<|pad|>".into(), 2.into());
                t["added_tokens"][0]["content"] = "<|pad|>".into();
            }),
            ("im_start missing", |t| {
                let vocab = t["model"]["vocab"].as_object_mut().unwrap();
                vocab.remove(IM_START_TOKEN);
                t["added_tokens"].as_array_mut().unwrap().remove(3);
            }),
            ("a BOS template", |t| {
                t["post_processor"] = json!({
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": PAD_TOKEN, "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}}
                    ],
                    "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                    "special_tokens": {
                        PAD_TOKEN: {"id": PAD_TOKEN, "ids": [2], "tokens": [PAD_TOKEN]}
                    }
                })
            }),
            ("truncation", |t| {
                t["truncation"] = json!({
                    "direction": "Right", "max_length": 8, "strategy": "LongestFirst", "stride": 0
                })
            }),
            ("vocabulary larger than the embedding", |t| {
                for extra in 0..30u32 {
                    t["model"]["vocab"][format!("w{extra}")] = (48 + extra).into();
                }
            }),
        ];
        assert!(load(&synthetic_json()).is_ok());
        for (name, edit) in edits {
            let mut json = synthetic_json();
            edit(&mut json);
            let err = load(&json).expect_err(name);
            assert!(matches!(err, Error::ModelLoadError(_)), "{name}: {err}");
        }
    }

    /// The published `tokenizer.json`; on this Mac
    /// `tasks/fish-drama3-evidence-2026-10-29/hf-78e5ab79/tokenizer.json`.
    const TOKENIZER_PATH: &str = "IZWI_FISH_DRAMA3_TOKENIZER_PATH";

    pub(crate) fn published_tokenizer() -> FishDrama3Tokenizer {
        let path = std::env::var(TOKENIZER_PATH)
            .unwrap_or_else(|_| panic!("set {TOKENIZER_PATH} to the published tokenizer.json"));
        let config = crate::models::architectures::fish_drama3::config::tests::published();
        FishDrama3Tokenizer::from_file(Path::new(&path), &config).unwrap()
    }

    /// Ids from the Python `tokenizers` 0.22.2 library (the one upstream
    /// uses) on the pinned `tokenizer.json`; the first seven are evidence 03
    /// §5 / plan §2.3.
    pub(crate) const LITERAL_GOLDENS: &[(&str, &[u32])] = &[
        ("<|im_start|>system\n", &[248_045, 8678, 198]),
        (
            "convert the provided text to speech",
            &[13761, 279, 3766, 1414, 310, 8540],
        ),
        (
            "convert the provided text to speech reference to the following:",
            &[
                13761, 279, 3766, 1414, 310, 8540, 5604, 310, 279, 2614, 25,
            ],
        ),
        ("<|im_end|>\n", &[248_046, 198]),
        ("<|im_start|>user\n", &[248_045, 846, 198]),
        (
            "<|im_start|>assistant\n<|voice|>",
            &[248_045, 74455, 198, 248_081],
        ),
        (
            "<|speaker:0|> Hello, how are you?",
            &[248_083, 21251, 11, 1204, 513, 488, 30],
        ),
        // Evidence 03 §5 elides the middle; the README's `en0` transcript.
        (
            "\n\n[Reference 1]\nText: <|speaker:0|> We asked over twenty different people, and they all said it was his.\nSpeech: ",
            &[
                271, 58, 8589, 220, 16, 60, 198, 1138, 25, 220, 248_083, 1165, 4439, 888, 16570,
                2086, 1208, 11, 321, 781, 660, 1018, 424, 557, 780, 13, 198, 53757, 25, 220,
            ],
        ),
        (
            "\n\n[Reference 2]\nText: <|speaker:1|> Always.\nSpeech: ",
            &[
                271, 58, 8589, 220, 17, 60, 198, 1138, 25, 220, 248_084, 22516, 13, 198, 53757,
                25, 220,
            ],
        ),
        (
            "<|speaker:0|> 今天天气很好，我们去公园散步。",
            &[
                248_083, 220, 97375, 101618, 98692, 3709, 116450, 99192, 112869, 1710,
            ],
        ),
        (
            "<|speaker:1|> Ready? <|speaker:0|> Always.",
            &[248_084, 29965, 30, 220, 248_083, 22516, 13],
        ),
        (
            "<|speaker:0|> [sighs] <whisper>not now</whisper> Pi is 3.14.",
            &[
                248_083, 498, 82, 1055, 82, 60, 361, 1266, 26483, 29, 1856, 1381, 510, 1266,
                26483, 29, 20010, 369, 220, 18, 13, 16, 19, 13,
            ],
        ),
        ("<|speaker:0|>   reference", &[248_083, 256, 5604]),
        ("<|speaker:0|>hello", &[248_083, 14556]),
        ("", &[]),
    ];

    #[test]
    #[ignore = "needs the published tokenizer.json in IZWI_FISH_DRAMA3_TOKENIZER_PATH"]
    fn the_published_tokenizer_matches_the_golden_ids() {
        let tokenizer = published_tokenizer();
        assert_eq!(tokenizer.im_start(), 248_045);
        assert_eq!(tokenizer.codebooks(), 10);
        for (literal, ids) in LITERAL_GOLDENS {
            assert_eq!(
                tokenizer.encode_segment(literal).unwrap(),
                *ids,
                "{literal:?}"
            );
        }
        // The trained-but-never-emitted ids, and S2's pad, sit where plan
        // §2.3 says.
        for (token, id) in [
            ("<|fim_prefix|>", 248_060),
            ("<|fim_suffix|>", 248_062),
            ("<|pad|>", 248_077),
            ("<|text|>", 248_080),
            ("<|interleave|>", 248_082),
            ("<|endofturn|>", 252_195),
            ("<|audio_separator|>", 252_196),
        ] {
            assert_eq!(tokenizer.tokenizer.token_to_id(token), Some(id), "{token}");
        }
    }
}

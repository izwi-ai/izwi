//! Chatterbox's text frontends: upstream's `punc_norm`, then the BPE
//! tokenizer with spaces spelled `[SPACE]` (`tts.py`, `mtl_tts.py`,
//! `models/tokenizers/tokenizer.py`). The multilingual one also lowercases,
//! applies NFKD, decomposes Korean syllables and prefixes a `[lang]` tag.

use std::path::Path;

use tokenizers::normalizers::NFKD;
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

use crate::error::{Error, Result};

const SPACE_TOKEN: &str = "[SPACE]";
const EMPTY_TEXT_REPLACEMENT: &str = "You need to add some text for me to talk.";

/// Punctuation upstream rewrites, applied in this order.
const PUNCTUATION_REPLACEMENTS: [(&str, &str); 12] = [
    ("...", ", "),
    ("…", ", "),
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("“", "\""),
    ("”", "\""),
    ("‘", "'"),
    ("’", "'"),
];

const SENTENCE_ENDERS: [char; 5] = ['.', '!', '?', '-', ','];
/// The multilingual `punc_norm` also accepts CJK enders.
const MULTILINGUAL_SENTENCE_ENDERS: [char; 10] =
    ['.', '!', '?', '-', ',', '、', '，', '。', '？', '！'];

/// The multilingual checkpoint's languages (`SUPPORTED_LANGUAGES`), by code
/// and English name.
pub const MULTILINGUAL_LANGUAGES: [(&str, &str); 23] = [
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// Languages whose upstream frontend needs a word segmenter and Cangjie
/// table (zh, spacy-pkuseg) or a kanji reader (ja, pykakasi) izwi has not
/// ported; every other language matches upstream's default install, Hebrew
/// and Russian without the optional diacritics and stress models upstream
/// also skips when they are absent.
const UNPORTED_LANGUAGES: [&str; 2] = ["ja", "zh"];

/// The multilingual language tag for a request's `language`: a code ("fr")
/// or English name ("French") in any case. Absent or "auto" is English.
pub fn multilingual_language(language: Option<&str>) -> Result<&'static str> {
    let requested = language
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"));
    let Some(requested) = requested else {
        return Ok("en");
    };
    let (code, name) = MULTILINGUAL_LANGUAGES
        .iter()
        .find(|(code, name)| {
            requested.eq_ignore_ascii_case(code) || requested.eq_ignore_ascii_case(name)
        })
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Chatterbox Multilingual does not speak {requested:?}; use one of {}",
                supported_multilingual_languages().join(", ")
            ))
        })?;
    if UNPORTED_LANGUAGES.contains(code) {
        return Err(Error::InvalidInput(format!(
            "Chatterbox Multilingual's {name} needs a text frontend izwi does not have yet; use one of {}",
            supported_multilingual_languages().join(", ")
        )));
    }
    Ok(code)
}

/// The codes [`multilingual_language`] accepts.
pub fn supported_multilingual_languages() -> Vec<&'static str> {
    supported_multilingual_language_names()
        .into_iter()
        .map(|(code, _)| code)
        .collect()
}

/// The languages [`multilingual_language`] accepts, by code and English
/// name.
pub fn supported_multilingual_language_names() -> Vec<(&'static str, &'static str)> {
    MULTILINGUAL_LANGUAGES
        .iter()
        .copied()
        .filter(|(code, _)| !UNPORTED_LANGUAGES.contains(code))
        .collect()
}

/// Whether a request's `language` asks the English checkpoint for English
/// (or for nothing in particular).
pub fn english_language(language: Option<&str>) -> Result<()> {
    match language.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(()),
        Some(value)
            if ["en", "english", "auto"]
                .iter()
                .any(|accepted| value.eq_ignore_ascii_case(accepted)) =>
        {
            Ok(())
        }
        Some(value) => Err(Error::InvalidInput(format!(
            "Chatterbox-500M speaks English only, not {value:?}; use Chatterbox-Multilingual-500M"
        ))),
    }
}

/// Upstream's `punc_norm`: capitalise the first letter, collapse
/// whitespace, rewrite uncommon punctuation and end on punctuation.
pub fn normalize_punctuation(text: &str) -> String {
    normalize_punctuation_ending(text, &SENTENCE_ENDERS)
}

/// The multilingual text upstream tokenizes: `punc_norm` (CJK enders
/// accepted), lowercase, NFKD, Korean syllables as jamo, then the `[lang]`
/// tag (`mtl_tts.py`, `MTLTokenizer.encode`).
pub fn normalize_multilingual(text: &str, language: &str) -> Result<String> {
    let text = normalize_punctuation_ending(text, &MULTILINGUAL_SENTENCE_ENDERS).to_lowercase();
    let mut normalized = NormalizedString::from(text);
    NFKD.normalize(&mut normalized)
        .map_err(|error| Error::InvalidInput(format!("Chatterbox text normalization: {error}")))?;
    let mut text = normalized.get().to_string();
    if language == "ko" {
        text = korean_jamo(&text);
    }
    Ok(format!("[{language}]{text}"))
}

/// Upstream's `korean_normalize`: each Hangul syllable as its jamo, then
/// trimmed. After NFKD the syllables are already decomposed the same way.
fn korean_jamo(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 3);
    for c in text.chars() {
        let code = c as u32;
        if !(0xAC00..=0xD7AF).contains(&code) {
            out.push(c);
            continue;
        }
        let base = code - 0xAC00;
        out.push(char::from_u32(0x1100 + base / (21 * 28)).expect("initial jamo"));
        out.push(char::from_u32(0x1161 + (base % (21 * 28)) / 28).expect("medial jamo"));
        if !base.is_multiple_of(28) {
            out.push(char::from_u32(0x11A7 + base % 28).expect("final jamo"));
        }
    }
    out.trim().to_string()
}

fn normalize_punctuation_ending(text: &str, enders: &[char]) -> String {
    if text.is_empty() {
        return EMPTY_TEXT_REPLACEMENT.to_string();
    }
    let mut chars = text.chars();
    let first = chars.next().expect("text is not empty");
    let text = if first.is_lowercase() {
        first.to_uppercase().chain(chars).collect::<String>()
    } else {
        text.to_string()
    };
    // Python's `str.split()` separators: Unicode whitespace plus the ASCII
    // information separators.
    let mut text = text
        .split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    for (old, new) in PUNCTUATION_REPLACEMENTS {
        text = text.replace(old, new);
    }
    let trimmed_len = text.trim_end_matches(' ').len();
    text.truncate(trimmed_len);
    if !text.ends_with(enders) {
        text.push('.');
    }
    text
}

pub struct ChatterboxTextTokenizer {
    tokenizer: Tokenizer,
    start_text_token: u32,
    stop_text_token: u32,
    /// The multilingual frontend: language tags, lowercase, NFKD.
    multilingual: bool,
}

impl ChatterboxTextTokenizer {
    pub fn load(path: &Path, start_text_token: u32, stop_text_token: u32) -> Result<Self> {
        Self::load_frontend(path, start_text_token, stop_text_token, false)
    }

    /// The multilingual checkpoint's grapheme tokenizer, which must carry a
    /// tag for every language.
    pub fn load_multilingual(
        path: &Path,
        start_text_token: u32,
        stop_text_token: u32,
    ) -> Result<Self> {
        Self::load_frontend(path, start_text_token, stop_text_token, true)
    }

    fn load_frontend(
        path: &Path,
        start_text_token: u32,
        stop_text_token: u32,
        multilingual: bool,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Failed to load Chatterbox tokenizer {}: {error}",
                path.display()
            ))
        })?;
        for (name, id) in [("[START]", start_text_token), ("[STOP]", stop_text_token)] {
            if tokenizer.token_to_id(name) != Some(id) {
                return Err(Error::ModelLoadError(format!(
                    "Chatterbox tokenizer maps {name} to {:?}, expected {id}",
                    tokenizer.token_to_id(name)
                )));
            }
        }
        if tokenizer.token_to_id(SPACE_TOKEN).is_none() {
            return Err(Error::ModelLoadError(
                "Chatterbox tokenizer has no [SPACE] token".into(),
            ));
        }
        if multilingual {
            for (code, _) in MULTILINGUAL_LANGUAGES {
                if tokenizer.token_to_id(&format!("[{code}]")).is_none() {
                    return Err(Error::ModelLoadError(format!(
                        "Chatterbox multilingual tokenizer has no [{code}] tag"
                    )));
                }
            }
        }
        Ok(Self {
            tokenizer,
            start_text_token,
            stop_text_token,
            multilingual,
        })
    }

    pub fn multilingual(&self) -> bool {
        self.multilingual
    }

    /// Token ids of already-normalised text, without start/stop tokens.
    pub fn encode_normalized(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text.replace(' ', SPACE_TOKEN), true)
            .map_err(|error| Error::InvalidInput(format!("Chatterbox tokenizer: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The T3 text sequence for raw request text: normalised, tokenised and
    /// framed by `[START]` and `[STOP]`.
    pub fn text_sequence(&self, text: &str) -> Result<Vec<u32>> {
        self.text_sequence_for(text, "en")
    }

    /// [`Self::text_sequence`] in `language`, a code from
    /// [`multilingual_language`]; the English tokenizer ignores it.
    pub fn text_sequence_for(&self, text: &str, language: &str) -> Result<Vec<u32>> {
        let normalized = if self.multilingual {
            normalize_multilingual(text, language)?
        } else {
            normalize_punctuation(text)
        };
        let mut ids = Vec::with_capacity(normalized.len() + 2);
        ids.push(self.start_text_token);
        ids.extend(self.encode_normalized(&normalized)?);
        ids.push(self.stop_text_token);
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected outputs are upstream's `punc_norm` at the pin, double spaces
    /// and the uncapitalised leading-space case included.
    #[test]
    fn punctuation_normalization_follows_upstream() {
        let cases = [
            ("", EMPTY_TEXT_REPLACEMENT),
            ("hello world", "Hello world."),
            ("  spaced \t out\n text  ", "spaced out text."),
            ("Wait... what…", "Wait,  what,"),
            ("Note: one; two — three", "Note, one,  two - three."),
            ("Quote “this” and ‘that’", "Quote \"this\" and 'that'."),
            ("A - B", "A, B."),
            ("Ends with a dash -", "Ends with a dash -"),
            ("Question?", "Question?"),
            ("ßtraße", "SStraße."),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_punctuation(input), expected, "{input:?}");
        }
    }

    /// Upstream's multilingual preprocessing, worked by hand from
    /// `MTLTokenizer.encode`: `punc_norm` with CJK enders, lowercase, NFKD
    /// (é → e + U+0301), Hangul as jamo, then the tag.
    #[test]
    fn multilingual_text_follows_upstream() {
        assert_eq!(
            normalize_multilingual("Ça va très bien", "fr").unwrap(),
            "[fr]c\u{327}a va tre\u{300}s bien."
        );
        assert_eq!(
            normalize_multilingual("ΟΔΟΣ ΚΑΛΗ", "el").unwrap(),
            "[el]οδο\u{3c2} καλη."
        );
        assert_eq!(
            normalize_multilingual("你好。", "zh").unwrap(),
            "[zh]你好。"
        );
        // 한 = ᄒ + ᅡ + ᆫ; NFKD and the jamo step agree.
        assert_eq!(
            normalize_multilingual("한국", "ko").unwrap(),
            "[ko]\u{1112}\u{1161}\u{11ab}\u{1100}\u{116e}\u{11a8}."
        );
        assert_eq!(korean_jamo(" 한 "), "\u{1112}\u{1161}\u{11ab}");
    }

    #[test]
    fn languages_resolve_by_code_or_name() {
        assert_eq!(multilingual_language(None).unwrap(), "en");
        assert_eq!(multilingual_language(Some("Auto")).unwrap(), "en");
        assert_eq!(multilingual_language(Some("FR")).unwrap(), "fr");
        assert_eq!(multilingual_language(Some("swahili")).unwrap(), "sw");
        assert!(multilingual_language(Some("klingon")).is_err());
        let refused = multilingual_language(Some("Japanese"))
            .unwrap_err()
            .to_string();
        assert!(refused.contains("Japanese"), "{refused}");
        assert!(multilingual_language(Some("zh")).is_err());
        assert_eq!(supported_multilingual_languages().len(), 21);
        assert!(english_language(Some("English")).is_ok());
        assert!(english_language(None).is_ok());
        assert!(english_language(Some("fr")).is_err());
    }

    /// Every supported language's corpus sentences tokenize under the
    /// grapheme vocabulary with their tag first and no `[UNK]`.
    #[test]
    #[ignore = "needs grapheme_mtl_merged_expanded_v1.json in IZWI_CHATTERBOX_MULTILINGUAL_DIR"]
    fn multilingual_corpus_tokenizes_without_unknowns() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("IZWI_CHATTERBOX_MULTILINGUAL_DIR")
                .expect("IZWI_CHATTERBOX_MULTILINGUAL_DIR"),
        );
        let tokenizer = ChatterboxTextTokenizer::load_multilingual(
            &dir.join("grapheme_mtl_merged_expanded_v1.json"),
            255,
            0,
        )
        .unwrap();
        let corpus: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/chatterbox/multilingual_corpus.json"
        ))
        .unwrap();
        let unknown = tokenizer.tokenizer.token_to_id("[UNK]").unwrap();
        for code in supported_multilingual_languages() {
            let sentences = corpus["sentences"][code].as_array().unwrap();
            for sentence in sentences {
                let ids = tokenizer
                    .text_sequence_for(sentence.as_str().unwrap(), code)
                    .unwrap();
                let tag = tokenizer
                    .tokenizer
                    .token_to_id(&format!("[{code}]"))
                    .unwrap();
                assert_eq!(ids[1], tag, "{code}: {sentence}");
                assert!(!ids.contains(&unknown), "{code}: {sentence} → {ids:?}");
            }
        }
    }
}

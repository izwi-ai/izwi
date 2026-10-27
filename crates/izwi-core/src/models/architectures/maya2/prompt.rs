//! Maya2's text front end: language resolution and the prompt spans of the
//! training packer (`tts/data/packing_v2.py::PackerV2._span`, layout "v5").
//!
//! A span is `[<|lang:xx|>] tokens(text) <|audio|>`, with no BOS. A built-in
//! voice is selected by its name in the text, braces included:
//!
//! `<|lang:B|> tokens("{Ahana}: " + text) <|audio|>`
//!
//! and a cloned voice by a reference span ahead of the target's:
//!
//! `<|lang:A|> tokens(ref_text) <|audio|> [ref frames] [AEOS] [<|lang:B|>] tokens(text) <|audio|>`
//!
//! where the target's language tag appears only when it differs from the
//! reference's. The speaker anchor is added at every `<|audio|>` sentinel.

use std::path::Path;

use tokenizers::Tokenizer;

use super::config::{
    language_token, AUDIO_SENTINEL, AUDIO_SENTINEL_TEXT, CODEBOOKS, CODEBOOK_SIZE, END_OF_TEXT,
    FIRST_LANGUAGE_TOKEN,
};
use crate::catalog::MAYA2_LANGUAGES;
use crate::error::{Error, Result};

/// One prompt position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptItem {
    /// A text-vocabulary token.
    Text(u32),
    /// Reference frame `n`: the sum of its 32 codebook embeddings.
    Audio(usize),
    /// The end-of-audio input after a reference's frames.
    Aeos,
}

/// A request's prompt: its positions and where the speaker anchor goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Maya2Prompt {
    pub items: Vec<PromptItem>,
    /// Indices of the `<|audio|>` sentinels (the last is the final position).
    pub sentinels: Vec<usize>,
    /// Tokens the target text alone takes.
    pub text_tokens: usize,
}

/// A prepared reference clip: its transcript, language and Mimi codes
/// (frame-major, 32 per frame).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Maya2Reference {
    pub text: String,
    pub language: &'static str,
    pub codes: Vec<u32>,
}

impl Maya2Reference {
    pub fn frames(&self) -> usize {
        self.codes.len() / CODEBOOKS
    }
}

pub struct Maya2Tokenizer {
    tokenizer: Tokenizer,
    /// Every added token's text: input that spells one would select it.
    added: Vec<String>,
}

impl Maya2Tokenizer {
    /// Load `backbone/tokenizer.json` and check the control tokens sit where
    /// the port expects them.
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Maya2 tokenizer {} unreadable: {error}",
                path.display()
            ))
        })?;
        let added = tokenizer
            .get_added_tokens_decoder()
            .values()
            .map(|token| token.content.clone())
            .collect();
        let this = Self { tokenizer, added };
        this.check_layout()?;
        Ok(this)
    }

    fn check_layout(&self) -> Result<()> {
        let expect = |token: &str, id: u32| -> Result<()> {
            match self.tokenizer.token_to_id(token) {
                Some(found) if found == id => Ok(()),
                found => Err(Error::ModelLoadError(format!(
                    "Maya2 tokenizer maps {token} to {found:?}, expected {id}"
                ))),
            }
        };
        expect("<|endoftext|>", END_OF_TEXT)?;
        expect(AUDIO_SENTINEL_TEXT, AUDIO_SENTINEL)?;
        for (offset, language) in MAYA2_LANGUAGES.iter().enumerate() {
            expect(
                &format!("<|lang:{}|>", language.code),
                FIRST_LANGUAGE_TOKEN + offset as u32,
            )?;
        }
        Ok(())
    }

    /// Token ids of `text`, without added special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(token) = self
            .added
            .iter()
            .find(|token| text.contains(token.as_str()))
        {
            return Err(Error::InvalidInput(format!(
                "Maya2 text may not contain the control token `{token}`"
            )));
        }
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::InvalidInput(format!("Maya2 tokenizer failed: {error}")))?
            .get_ids()
            .to_vec())
    }

    /// One span: `[lang] tokens(text) <|audio|>`, appended to `items`.
    fn span(
        &self,
        items: &mut Vec<PromptItem>,
        language: Option<&str>,
        text: &str,
    ) -> Result<usize> {
        if let Some(code) = language {
            items.push(PromptItem::Text(language_token(code)?));
        }
        let ids = self.encode(text)?;
        let count = ids.len();
        items.extend(ids.into_iter().map(PromptItem::Text));
        items.push(PromptItem::Text(AUDIO_SENTINEL));
        Ok(count)
    }

    /// The prompt of built-in voice `voice` speaking `text` in `language`.
    pub fn named_prompt(&self, voice: &str, language: &str, text: &str) -> Result<Maya2Prompt> {
        let text = checked_text(text)?;
        let mut items = Vec::new();
        self.span(&mut items, Some(language), &format!("{{{voice}}}: {text}"))?;
        let text_tokens = self.encode(text)?.len();
        Ok(Maya2Prompt {
            sentinels: vec![items.len() - 1],
            items,
            text_tokens,
        })
    }

    /// The prompt of a voice cloned from `reference` speaking `text` in
    /// `language`.
    pub fn clone_prompt(
        &self,
        reference: &Maya2Reference,
        language: &str,
        text: &str,
    ) -> Result<Maya2Prompt> {
        let text = checked_text(text)?;
        let reference_text = checked_text(&reference.text)?;
        if reference.codes.is_empty()
            || !reference.codes.len().is_multiple_of(CODEBOOKS)
            || reference
                .codes
                .iter()
                .any(|&code| code as usize >= CODEBOOK_SIZE)
        {
            return Err(Error::InvalidInput(
                "Maya2 reference codes must be whole frames of 32 codes".into(),
            ));
        }
        let mut items = Vec::new();
        self.span(&mut items, Some(reference.language), reference_text)?;
        let mut sentinels = vec![items.len() - 1];
        items.extend((0..reference.frames()).map(PromptItem::Audio));
        items.push(PromptItem::Aeos);
        let cross_lingual = reference.language != language;
        let text_tokens = self.span(&mut items, cross_lingual.then_some(language), text)?;
        sentinels.push(items.len() - 1);
        Ok(Maya2Prompt {
            items,
            sentinels,
            text_tokens,
        })
    }
}

/// `text` trimmed, refused when empty.
fn checked_text(text: &str) -> Result<&str> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::InvalidInput("Maya2 cannot speak empty text".into()));
    }
    Ok(text)
}

/// Unicode blocks of the Indic scripts and the language upstream assumes
/// for each (Devanagari reads as Hindi, Bengali script as Bengali: Marathi
/// and Assamese must be named).
const SCRIPTS: [(u32, u32, &str); 9] = [
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
];

/// The language of `text`'s dominant script, as upstream's server picks it
/// for a request without one: ASCII letters count for English, each Indic
/// block for its language; ties go to the language seen first; no letters
/// at all is English.
pub fn detect_language(text: &str) -> &'static str {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for ch in text.chars() {
        let code = ch as u32;
        let language = if ch.is_ascii() {
            ch.is_ascii_alphabetic().then_some("en")
        } else {
            SCRIPTS
                .iter()
                .find(|(lo, hi, _)| (*lo..=*hi).contains(&code))
                .map(|(_, _, language)| *language)
        };
        if let Some(language) = language {
            match counts.iter_mut().find(|(seen, _)| *seen == language) {
                Some((_, count)) => *count += 1,
                None => counts.push((language, 1)),
            }
        }
    }
    let mut best: Option<(&'static str, usize)> = None;
    for (language, count) in counts {
        if best.is_none_or(|(_, top)| count > top) {
            best = Some((language, count));
        }
    }
    best.map_or("en", |(language, _)| language)
}

/// The language a request speaks: `requested` when it names a model
/// language, the dominant script of `text` when it names none or `auto`.
pub fn resolve_language(requested: Option<&str>, text: &str) -> Result<&'static str> {
    match requested.map(str::trim).filter(|value| !value.is_empty()) {
        None => Ok(detect_language(text)),
        Some(value) if value.eq_ignore_ascii_case("auto") => Ok(detect_language(text)),
        Some(value) => crate::catalog::maya2_language(value)
            .map(|language| language.code)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Maya 2 Native does not speak `{value}`; it speaks {}",
                    MAYA2_LANGUAGES
                        .iter()
                        .map(|language| language.code)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_script_picks_the_language() {
        assert_eq!(detect_language("Hello there."), "en");
        assert_eq!(detect_language("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।"), "hi");
        assert_eq!(detect_language("வணக்கம், எப்படி இருக்கிறீர்கள்?"), "ta");
        // Code-mixed: the script with more letters wins.
        assert_eq!(
            detect_language("12/08/2023 তারিখে আপনার 1,500 টাকার refund process হয়ে গেছে"),
            "bn"
        );
        assert_eq!(detect_language("12345"), "en");
        // A tie goes to the language seen first.
        assert_eq!(detect_language("ab கட"), "en");
    }

    #[test]
    fn requested_languages_are_validated() {
        assert_eq!(resolve_language(Some("MR"), "x").unwrap(), "mr");
        assert_eq!(resolve_language(Some("auto"), "ಹೌದಾ").unwrap(), "kn");
        assert_eq!(resolve_language(None, "ਸਤ ਸ੍ਰੀ").unwrap(), "pa");
        assert!(resolve_language(Some("fr"), "x").is_err());
    }
}

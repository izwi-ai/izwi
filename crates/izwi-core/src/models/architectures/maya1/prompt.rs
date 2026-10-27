//! Maya1's text front end: the voice description, inline emotion tags and
//! the prompt framing.
//!
//! Upstream frames a request three ways (the card's Transformers example,
//! its Space, which adds a second BOS, and a vLLM script using Llama 3's
//! dated chat template). Izwi uses the card's explicit framing:
//!
//! `SOH BOS <description="DESCRIPTION"> TEXT EOT EOH SOA SOS`
//!
//! tokenized without added special tokens, so generation starts on the
//! first audio code.

use std::path::Path;

use tokenizers::Tokenizer;

use super::config::{
    BOS, END_OF_HUMAN, MAX_DESCRIPTION_CHARS, MAYA1_INLINE_TAGS, START_OF_AI, START_OF_HUMAN,
    START_OF_SPEECH, TEXT_EOT,
};
use crate::error::{Error, Result};

/// The first and last ids of the emotion tags in the tokenizer.
const FIRST_TAG_ID: u32 = 156_940;

pub struct Maya1Tokenizer {
    tokenizer: Tokenizer,
}

impl Maya1Tokenizer {
    /// Load `tokenizer.json` and check that the control ids and every
    /// emotion tag are where the port expects them.
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Maya1 tokenizer {} unreadable: {error}",
                path.display()
            ))
        })?;
        let this = Self { tokenizer };
        this.check_layout()?;
        Ok(this)
    }

    fn check_layout(&self) -> Result<()> {
        let expect = |token: &str, id: u32| -> Result<()> {
            match self.tokenizer.token_to_id(token) {
                Some(found) if found == id => Ok(()),
                found => Err(Error::ModelLoadError(format!(
                    "Maya1 tokenizer maps {token} to {found:?}, expected {id}"
                ))),
            }
        };
        expect("<|begin_of_text|>", BOS)?;
        expect("<|eot_id|>", TEXT_EOT)?;
        expect("<custom_token_1>", START_OF_SPEECH)?;
        expect("<custom_token_3>", START_OF_HUMAN)?;
        expect("<custom_token_4>", END_OF_HUMAN)?;
        expect("<custom_token_5>", START_OF_AI)?;
        let mut tags = MAYA1_INLINE_TAGS.to_vec();
        tags.sort_unstable();
        for (offset, tag) in tags.iter().enumerate() {
            expect(tag, FIRST_TAG_ID + offset as u32)?;
        }
        Ok(())
    }

    /// Prompt ids for `description` and `text`, both already normalised.
    pub fn prompt_ids(&self, description: &str, text: &str) -> Result<Vec<u32>> {
        let content = format!("<description=\"{description}\"> {text}");
        let body = self.encode(&content)?;
        let mut ids = Vec::with_capacity(body.len() + 6);
        ids.extend([START_OF_HUMAN, BOS]);
        ids.extend(body);
        ids.extend([TEXT_EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]);
        Ok(ids)
    }

    /// Tokens `text` alone takes (generation budgets scale with it).
    pub fn text_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::InvalidInput(format!("Maya1 tokenization failed: {error}")))?
            .get_ids()
            .to_vec())
    }
}

/// A voice description as the model reads it: whitespace collapsed, double
/// quotes (which would close the description's wrapper) turned into single
/// ones, and a final full stop. Upstream users found the model reads terse
/// or unterminated descriptions aloud (maya1 discussion #25).
pub fn normalize_description(raw: &str) -> Result<String> {
    let mut description = raw
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('"', "'");
    if description.is_empty() {
        return Err(Error::InvalidInput(
            "Maya1 needs a voice description (or a built-in voice)".into(),
        ));
    }
    if description.chars().count() > MAX_DESCRIPTION_CHARS {
        return Err(Error::InvalidInput(format!(
            "Maya1 voice descriptions take at most {MAX_DESCRIPTION_CHARS} characters"
        )));
    }
    if !description.ends_with(['.', '!', '?']) {
        description.push('.');
    }
    Ok(description)
}

/// Text to speak, whitespace collapsed, with every `<tag>` checked against
/// the tokenizer's emotion tags: an unknown one would be read aloud.
pub fn normalize_text(raw: &str) -> Result<String> {
    let text = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Err(Error::InvalidInput("Maya1 needs non-empty text".into()));
    }
    validate_inline_tags(&text)?;
    Ok(text)
}

/// Refuse any `<word>` in `text` that is not one of [`MAYA1_INLINE_TAGS`].
pub fn validate_inline_tags(text: &str) -> Result<()> {
    let unknown = inline_tags(text)
        .filter(|tag| !MAYA1_INLINE_TAGS.contains(tag))
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(Error::InvalidInput(format!(
        "Maya1 does not know the tag{} {}; it understands {}",
        if unknown.len() == 1 { "" } else { "s" },
        unknown.join(", "),
        MAYA1_INLINE_TAGS.join(" ")
    )))
}

/// Every `<word>` in `text` (letters and underscores between angle brackets).
pub fn inline_tags(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut index = 0;
    std::iter::from_fn(move || {
        while index < bytes.len() {
            if bytes[index] != b'<' {
                index += 1;
                continue;
            }
            let start = index;
            let mut end = start + 1;
            while end < bytes.len() && (bytes[end].is_ascii_alphabetic() || bytes[end] == b'_') {
                end += 1;
            }
            if end > start + 1 && end < bytes.len() && bytes[end] == b'>' {
                index = end + 1;
                return Some(&text[start..=end]);
            }
            index = start + 1;
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_are_wrapped_safely_and_terminated() {
        assert_eq!(
            normalize_description("  casual speaker,\n male 40s ").unwrap(),
            "casual speaker, male 40s."
        );
        assert_eq!(
            normalize_description("A \"bright\" voice!").unwrap(),
            "A 'bright' voice!"
        );
        assert!(normalize_description(" \n ").is_err());
        assert!(normalize_description(&"a".repeat(MAX_DESCRIPTION_CHARS + 1)).is_err());
    }

    #[test]
    fn inline_tags_are_found_and_checked() {
        let text = "Ha! <laugh> That was <laugh_harder> fun, x < y > z, <3";
        assert_eq!(
            inline_tags(text).collect::<Vec<_>>(),
            vec!["<laugh>", "<laugh_harder>"]
        );
        assert!(validate_inline_tags(text).is_ok());
        let error = validate_inline_tags("Well <happy> and <Laugh>").unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("<happy>") && message.contains("<Laugh>"),
            "{message}"
        );
        assert!(message.contains("<whisper>"), "{message}");
    }

    #[test]
    fn text_is_collapsed_and_must_not_be_empty() {
        assert_eq!(normalize_text(" Hello\n\nthere ").unwrap(), "Hello there");
        assert!(normalize_text("   ").is_err());
        assert!(normalize_text("Hi <shout>").is_err());
    }

    #[test]
    fn tag_list_is_sorted_like_the_tokenizer_ids() {
        let mut sorted = MAYA1_INLINE_TAGS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, MAYA1_INLINE_TAGS.to_vec());
    }
}

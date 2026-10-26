//! Chatterbox's English text frontend: upstream's `punc_norm`, then the
//! BPE tokenizer with spaces spelled `[SPACE]` (`tts.py`,
//! `models/tokenizers/tokenizer.py`).

use std::path::Path;

use tokenizers::Tokenizer;

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

/// Upstream's `punc_norm`: capitalise the first letter, collapse
/// whitespace, rewrite uncommon punctuation and end on punctuation.
pub fn normalize_punctuation(text: &str) -> String {
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
    if !text.ends_with(SENTENCE_ENDERS) {
        text.push('.');
    }
    text
}

pub struct ChatterboxTextTokenizer {
    tokenizer: Tokenizer,
    start_text_token: u32,
    stop_text_token: u32,
}

impl ChatterboxTextTokenizer {
    pub fn load(path: &Path, start_text_token: u32, stop_text_token: u32) -> Result<Self> {
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
        Ok(Self {
            tokenizer,
            start_text_token,
            stop_text_token,
        })
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
        let normalized = normalize_punctuation(text);
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
}

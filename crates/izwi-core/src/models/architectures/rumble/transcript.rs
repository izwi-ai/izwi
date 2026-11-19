//! Rumble's output: `language {Name}<asr_text>{transcript}` when the model
//! names the language itself, or the transcript alone after a forced
//! language. The split is made on token ids, so transcript text that
//! happens to read "language ..." is never mistaken for the header.

use crate::error::Result;

use super::tokenizer::RumbleTokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RumbleOutput {
    /// The forced language, or the one the model named (`None` when it
    /// named none, as for audio without speech).
    pub language: Option<String>,
    pub text: String,
}

/// Split one invocation's generated ids into language and transcript.
pub fn split_rumble_output(
    tokenizer: &RumbleTokenizer,
    generated: &[u32],
    forced_language: Option<&str>,
) -> Result<RumbleOutput> {
    if let Some(language) = forced_language {
        return Ok(RumbleOutput {
            language: Some(language.to_string()),
            text: clean(&tokenizer.decode(generated)?),
        });
    }
    let asr_text = tokenizer.specials().asr_text;
    let Some(marker) = generated.iter().position(|id| *id == asr_text) else {
        // No header: the model stopped before the marker (or never wrote
        // one). Whatever it wrote is not a transcript header we can trust
        // as text, unless it does not look like one.
        let raw = clean(&tokenizer.decode(generated)?);
        return Ok(match parse_language_header(&raw) {
            Some(language) => RumbleOutput {
                language,
                text: String::new(),
            },
            None => RumbleOutput {
                language: None,
                text: raw,
            },
        });
    };
    let header = tokenizer.decode(&generated[..marker])?;
    let text = clean(&tokenizer.decode(&generated[marker + 1..])?);
    Ok(RumbleOutput {
        language: parse_language_header(&header).flatten(),
        text,
    })
}

/// The language a `language {Name}` header names: `Some(None)` for
/// `language None`, `None` when the text is not a header.
fn parse_language_header(header: &str) -> Option<Option<String>> {
    let name = header.trim().strip_prefix("language")?.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("none") {
        return Some(None);
    }
    if name.len() > 40
        || !name
            .chars()
            .all(|ch| ch.is_alphabetic() || ch == ' ' || ch == '-')
    {
        return None;
    }
    Some(Some(name.to_string()))
}

fn clean(text: &str) -> String {
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_parse_to_a_language_or_none() {
        assert_eq!(
            parse_language_header("language English"),
            Some(Some("English".into()))
        );
        assert_eq!(parse_language_header(" language None "), Some(None));
        assert_eq!(parse_language_header("language"), Some(None));
        assert_eq!(parse_language_header("hello there"), None);
        assert_eq!(parse_language_header("language 123"), None);
    }
}

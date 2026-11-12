//! Rules applied to a transcript before it is inserted: filler words out,
//! the user's replacements in, spacing tidied. No model is involved.

use serde::{Deserialize, Serialize};

/// Replace `from` (matched as whole words, ignoring case) with `to`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Replacement {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextRules {
    pub remove_fillers: bool,
    pub replacements: Vec<Replacement>,
    /// End the text with a space, so the next dictation does not run into it.
    pub trailing_space: bool,
}

const FILLERS: &[&str] = &["um", "umm", "uh", "uhh", "uhm", "erm", "hmm"];

/// The text to insert for `raw`. Empty when nothing is left to insert.
pub fn apply(raw: &str, rules: &TextRules) -> String {
    let started_upper = first_letter(raw).is_some_and(char::is_uppercase);
    let mut text = raw.to_string();
    if rules.remove_fillers {
        for filler in FILLERS {
            text = replace_words(&text, filler, "", true);
        }
    }
    for replacement in &rules.replacements {
        let from = replacement.from.trim();
        if !from.is_empty() {
            text = replace_words(&text, from, &replacement.to, false);
        }
    }
    let mut text = tidy(&text);
    if rules.remove_fillers && started_upper {
        text = capitalize_first(&text);
    }
    if text.is_empty() {
        return text;
    }
    if rules.trailing_space && !text.ends_with(char::is_whitespace) {
        text.push(' ');
    }
    text
}

fn first_letter(text: &str) -> Option<char> {
    text.chars().find(|ch| ch.is_alphabetic())
}

fn capitalize_first(text: &str) -> String {
    let Some((index, ch)) = text.char_indices().find(|(_, ch)| ch.is_alphabetic()) else {
        return text.to_string();
    };
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..index]);
    out.extend(ch.to_uppercase());
    out.push_str(&text[index + ch.len_utf8()..]);
    out
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '\'' || ch == '’'
}

/// Replace whole-word, case-insensitive matches of `from`. With
/// `swallow_comma`, a comma straight after a match goes too ("Um, hello").
fn replace_words(text: &str, from: &str, to: &str, swallow_comma: bool) -> String {
    let needle: Vec<char> = from.chars().flat_map(char::to_lowercase).collect();
    if needle.is_empty() {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        let end = match_at(&chars, index, &needle);
        let boundary_before = index == 0 || !is_word_char(chars[index - 1]);
        match end {
            Some(end) if boundary_before && chars.get(end).is_none_or(|ch| !is_word_char(*ch)) => {
                out.push_str(to);
                index = end;
                if swallow_comma && chars.get(index) == Some(&',') {
                    index += 1;
                }
            }
            _ => {
                out.push(chars[index]);
                index += 1;
            }
        }
    }
    out
}

/// The index after a case-insensitive match of `needle` starting at `start`.
fn match_at(chars: &[char], start: usize, needle: &[char]) -> Option<usize> {
    let mut index = start;
    let mut matched = 0;
    while matched < needle.len() {
        let ch = *chars.get(index)?;
        for lower in ch.to_lowercase() {
            if needle.get(matched) != Some(&lower) {
                return None;
            }
            matched += 1;
        }
        index += 1;
    }
    Some(index)
}

/// Collapse runs of spaces, drop spaces before punctuation and trim the ends
/// of each line. Line breaks are kept: a replacement may add them.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (number, line) in text.split('\n').enumerate() {
        if number > 0 {
            out.push('\n');
        }
        let mut pending_space = false;
        let mut line_started = false;
        for ch in line.chars() {
            if ch == ' ' || ch == '\t' {
                pending_space = line_started;
                continue;
            }
            if pending_space && !matches!(ch, ',' | '.' | '!' | '?' | ';' | ':') {
                out.push(' ');
            }
            pending_space = false;
            line_started = true;
            out.push(ch);
        }
    }
    // Punctuation left at the very start by a removed filler.
    out.trim_matches(|ch: char| ch == ' ' || ch == '\t')
        .trim_start_matches([',', ';', ':'])
        .trim_start()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> TextRules {
        TextRules::default()
    }

    fn replacement(from: &str, to: &str) -> Replacement {
        Replacement {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    #[test]
    fn plain_text_is_trimmed_and_tidied() {
        assert_eq!(apply("  Hello   world .  ", &rules()), "Hello world.");
        assert_eq!(apply("   ", &rules()), "");
    }

    #[test]
    fn trailing_space_is_added_once() {
        let rules = TextRules {
            trailing_space: true,
            ..rules()
        };
        assert_eq!(apply("Done.", &rules), "Done. ");
        assert_eq!(apply("", &rules), "");
    }

    #[test]
    fn fillers_are_removed_as_whole_words() {
        let rules = TextRules {
            remove_fillers: true,
            ..rules()
        };
        assert_eq!(
            apply("Um, I think, uh, we should umbrella this.", &rules),
            "I think, we should umbrella this."
        );
        assert_eq!(apply("Hmm.", &rules), ".");
        assert_eq!(apply("so um yes", &rules), "so yes");
        // Off by default.
        assert_eq!(apply("Um, yes", &TextRules::default()), "Um, yes");
    }

    #[test]
    fn replacements_match_whole_words_ignoring_case() {
        let rules = TextRules {
            replacements: vec![
                replacement("is we", "Izwi"),
                replacement("new line", "\n"),
                replacement("", "ignored"),
            ],
            ..rules()
        };
        assert_eq!(
            apply("Open Is We and this weekend", &rules),
            "Open Izwi and this weekend"
        );
        assert_eq!(apply("First new line second", &rules), "First\nsecond");
        // Not inside a longer word.
        assert_eq!(apply("this weird", &rules), "this weird");
    }

    #[test]
    fn replacements_handle_non_ascii_text() {
        let rules = TextRules {
            replacements: vec![replacement("straße", "Strasse")],
            ..rules()
        };
        assert_eq!(
            apply("Die STRASSE und die Straße", &rules),
            "Die STRASSE und die Strasse"
        );
    }
}

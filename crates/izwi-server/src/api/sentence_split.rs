//! Sentence segmentation shared by long-form TTS and Studio (P3.1, TPIPE-5).
//!
//! A sentence ends at `!`, `?`, a full stop or an ellipsis followed by
//! whitespace (and, for an ellipsis, a capitalized next word), an Indic
//! danda (`।`, `॥`) followed by whitespace, a CJK terminator, or a line
//! break. Trailing closing quotes and brackets stay with
//! their sentence. A full stop does not end a sentence inside a number
//! ("3.14"), after a title or a fixed abbreviation ("Dr.", "e.g.", "No."),
//! after an initial ("J. Smith"), or after an abbreviation such as "etc."
//! unless the next word is capitalized. Colons and semicolons never end a
//! sentence ("10:30"); they are clause breaks for splitting a sentence that
//! is too long to speak in one request.
//!
//! [`pack_sentences`] then packs whole sentences into segments up to a
//! window (the family's one-request capacity and a character cap), so a
//! long text becomes a few model-sized requests rather than one per
//! sentence.

use std::ops::Range;

/// Abbreviations after which a full stop never ends a sentence.
const FIXED_ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon", "capt", "col", "gen",
    "gov", "lt", "sgt", "messrs", "vs", "cf", "no", "nos", "vol", "fig", "figs", "eq", "e.g",
    "i.e", "approx", "dept", "ca", "op", "pp", "u.s", "u.k",
];

/// Abbreviations that may also end a sentence: one does when the next word
/// is capitalized.
const TRAILING_ABBREVIATIONS: &[&str] = &[
    "etc", "al", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "est",
    "ave", "blvd", "rd",
];

fn is_closing(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\'' | '\u{201d}' | '\u{2019}' | '\u{00bb}' | ')' | ']' | '}'
            // CJK closing quotes and brackets: 」』）】》〉〕］｝〙〛
            | '\u{300D}' | '\u{300F}' | '\u{FF09}' | '\u{3011}' | '\u{300B}' | '\u{3009}'
            | '\u{3015}' | '\u{FF3D}' | '\u{FF5D}' | '\u{3019}' | '\u{301B}'
    )
}

/// Full stops, exclamation and question marks of CJK text, which need no
/// space after them: 。！？．｡‼⁇⁈⁉
fn is_cjk_terminator(ch: char) -> bool {
    matches!(
        ch,
        '。' | '！' | '？' | '\u{FF0E}' | '\u{FF61}' | '‼' | '⁇' | '⁈' | '⁉'
    )
}

/// The danda and double danda that end sentences in Devanagari, Bengali,
/// Gurmukhi and Odia text (Hindi, Marathi, Bengali, Assamese, Punjabi, Odia).
fn is_danda(ch: char) -> bool {
    matches!(ch, '\u{0964}' | '\u{0965}')
}

/// Byte ranges of the sentences of `text`, trimmed, in order.
pub(crate) fn sentence_spans(text: &str) -> Vec<Range<usize>> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    let end_of = |index: usize| chars.get(index).map_or(text.len(), |(offset, _)| *offset);
    while index < chars.len() {
        let (_, ch) = chars[index];
        let boundary = match ch {
            '\n' => Some(index + 1),
            _ if is_cjk_terminator(ch) => Some(absorb_closing(&chars, index + 1)),
            _ if is_danda(ch) => {
                let mut run_end = index + 1;
                while run_end < chars.len() && is_danda(chars[run_end].1) {
                    run_end += 1;
                }
                let after = absorb_closing(&chars, run_end);
                let ends = after >= chars.len() || chars[after].1.is_whitespace();
                index = run_end - 1;
                ends.then_some(after)
            }
            '!' | '?' | '.' | '\u{2026}' => {
                // The run of terminators ("?!", "...", "…") and the closing
                // quotes and brackets after it belong to this sentence.
                let mut run_end = index + 1;
                while run_end < chars.len()
                    && matches!(chars[run_end].1, '!' | '?' | '.' | '\u{2026}')
                {
                    run_end += 1;
                }
                let after = absorb_closing(&chars, run_end);
                let ellipsis = ch == '\u{2026}'
                    || (ch == '.' && run_end - index >= 3)
                    || chars[index..run_end].iter().any(|(_, c)| *c == '\u{2026}');
                let terminal_mark = chars[index..run_end]
                    .iter()
                    .any(|(_, c)| matches!(c, '!' | '?'));
                let next = next_word_start(&chars, after);
                let at_end = after >= chars.len();
                let followed_by_space = at_end || chars[after].1.is_whitespace();
                let ends = if !followed_by_space {
                    false
                } else if terminal_mark {
                    true
                } else if ellipsis {
                    at_end || next.is_some_and(|word| starts_capitalized(&chars, word))
                } else {
                    full_stop_ends_sentence(&chars, index, next)
                };
                index = run_end - 1;
                ends.then_some(after)
            }
            _ => None,
        };
        if let Some(boundary) = boundary {
            push_trimmed(text, start, end_of(boundary), &mut spans);
            start = end_of(boundary);
            index = boundary;
            continue;
        }
        index += 1;
    }
    push_trimmed(text, start, text.len(), &mut spans);
    spans
}

fn absorb_closing(chars: &[(usize, char)], mut index: usize) -> usize {
    while index < chars.len() && is_closing(chars[index].1) {
        index += 1;
    }
    index
}

fn next_word_start(chars: &[(usize, char)], mut index: usize) -> Option<usize> {
    while index < chars.len() && chars[index].1.is_whitespace() {
        index += 1;
    }
    // Opening quotes and brackets do not count against capitalization.
    while index < chars.len()
        && matches!(
            chars[index].1,
            '"' | '\'' | '\u{201c}' | '\u{2018}' | '(' | '['
        )
    {
        index += 1;
    }
    (index < chars.len()).then_some(index)
}

fn starts_capitalized(chars: &[(usize, char)], index: usize) -> bool {
    let ch = chars[index].1;
    ch.is_uppercase() || ch.is_ascii_digit()
}

/// Whether the full stop at `dot` (followed by whitespace) ends a sentence:
/// not after a title, a fixed abbreviation or an initial, and after a
/// trailing abbreviation only before a capitalized word.
fn full_stop_ends_sentence(chars: &[(usize, char)], dot: usize, next: Option<usize>) -> bool {
    let mut word_start = dot;
    while word_start > 0 && !chars[word_start - 1].1.is_whitespace() {
        word_start -= 1;
    }
    let word: String = chars[word_start..dot]
        .iter()
        .map(|(_, ch)| *ch)
        .skip_while(|ch| !ch.is_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    if word.is_empty() {
        return true;
    }
    let letters = word.chars().filter(|ch| ch.is_alphabetic()).count();
    let initial = letters == 1 && word.chars().count() == 1;
    // "U.S", "e.g", "a.m": an interior full stop marks an abbreviation.
    let dotted = word.contains('.');
    if initial || FIXED_ABBREVIATIONS.contains(&word.as_str()) {
        return false;
    }
    if dotted || TRAILING_ABBREVIATIONS.contains(&word.as_str()) {
        return next.is_none_or(|word| starts_capitalized(chars, word));
    }
    true
}

fn push_trimmed(text: &str, start: usize, end: usize, spans: &mut Vec<Range<usize>>) {
    let slice = &text[start..end];
    let leading = slice.len() - slice.trim_start().len();
    let trimmed = slice.trim();
    if !trimmed.is_empty() {
        let begin = start + leading;
        spans.push(begin..begin + trimmed.len());
    }
}

/// Byte ranges of the clauses of one sentence: after a comma, semicolon,
/// colon or dash followed by whitespace (so "1,250" and "10:30" stay whole).
fn clause_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((offset, ch)) = chars.next() {
        if matches!(ch, ',' | ';' | ':' | '\u{2014}' | '\u{2013}')
            && chars.peek().is_some_and(|(_, next)| next.is_whitespace())
        {
            let end = offset + ch.len_utf8();
            push_trimmed(text, start, end, &mut spans);
            start = end;
        }
    }
    push_trimmed(text, start, text.len(), &mut spans);
    spans
}

/// Pack the units at `spans` of `text` greedily: a segment grows while it
/// stays within `max_chars` and `fits`. A unit that does not fit alone is
/// split by `split_unit`.
fn pack(
    text: &str,
    spans: Vec<Range<usize>>,
    max_chars: usize,
    fits: &dyn Fn(&str) -> bool,
    split_unit: &dyn Fn(&str) -> Vec<String>,
) -> Vec<String> {
    let within = |candidate: &str| candidate.chars().count() <= max_chars && fits(candidate);
    let mut segments = Vec::new();
    let mut current: Option<Range<usize>> = None;
    for span in spans {
        let unit = &text[span.clone()];
        if !within(unit) {
            if let Some(range) = current.take() {
                segments.push(text[range].to_string());
            }
            segments.extend(split_unit(unit));
            continue;
        }
        current = Some(match current.take() {
            Some(range) if within(&text[range.start..span.end]) => range.start..span.end,
            Some(range) => {
                segments.push(text[range].to_string());
                span
            }
            None => span,
        });
    }
    if let Some(range) = current {
        segments.push(text[range].to_string());
    }
    segments
}

/// The segments of `text`: whole sentences packed up to `max_chars` and
/// `fits`; a sentence too long alone is split at its clauses, then at
/// words, then at characters.
pub(crate) fn pack_sentences(
    text: &str,
    max_chars: usize,
    fits: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let split_sentence = |sentence: &str| {
        pack(
            sentence,
            clause_spans(sentence),
            max_chars,
            fits,
            &|clause: &str| split_words(clause, max_chars),
        )
    };
    pack(text, sentence_spans(text), max_chars, fits, &split_sentence)
}

/// A unit longer than `max_chars` with no clause break: whole words, then
/// characters for an unbroken run (a CJK block or a URL).
fn split_words(unit: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    for word in unit.split_whitespace() {
        let word_chars = word.chars().count();
        let separator = usize::from(!current.is_empty());
        if current_chars + separator + word_chars <= max_chars {
            if separator == 1 {
                current.push(' ');
            }
            current.push_str(word);
            current_chars += separator + word_chars;
            continue;
        }
        if !current.is_empty() {
            out.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        if word_chars <= max_chars {
            current.push_str(word);
            current_chars = word_chars;
            continue;
        }
        let characters: Vec<char> = word.chars().collect();
        for piece in characters.chunks(max_chars) {
            out.push(piece.iter().collect());
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hindi, Bengali and Punjabi sentences end at a danda followed by a
    /// space; a danda run and a closing quote stay with their sentence.
    #[test]
    fn dandas_end_indic_sentences() {
        let text = "नमस्ते। आप कैसे हैं? \"ठीक हूँ॥\" আমি ভালো আছি। ਧੰਨਵਾਦ";
        let sentences = sentence_spans(text)
            .into_iter()
            .map(|span| &text[span])
            .collect::<Vec<_>>();
        assert_eq!(
            sentences,
            ["नमस्ते।", "आप कैसे हैं?", "\"ठीक हूँ॥\"", "আমি ভালো আছি।", "ਧੰਨਵਾਦ"]
        );
        // No space after the danda: not a boundary.
        assert_eq!(sentence_spans("एक।दो").len(), 1);
    }

    fn sentences(text: &str) -> Vec<&str> {
        sentence_spans(text)
            .into_iter()
            .map(|span| &text[span])
            .collect()
    }

    #[test]
    fn cjk_sentences_keep_their_closing_quotes() {
        assert_eq!(
            sentences("「こんにちは。」次の文です！最後？"),
            vec!["「こんにちは。」", "次の文です！", "最後？"]
        );
        assert_eq!(
            sentences("他说：“好。”然后走了．再见"),
            vec!["他说：“好。”", "然后走了．", "再见"]
        );
    }

    #[test]
    fn decimals_times_abbreviations_and_initials_stay_inside_a_sentence() {
        assert_eq!(
            sentences(
                "Pi is 3.14 today. Dr. Smith met J. R. Tolkien at 10:30, e.g. after lunch. \
                 See Fig. 2 and No. 5."
            ),
            vec![
                "Pi is 3.14 today.",
                "Dr. Smith met J. R. Tolkien at 10:30, e.g. after lunch.",
                "See Fig. 2 and No. 5.",
            ]
        );
        // A trailing abbreviation ends a sentence before a capital only.
        assert_eq!(
            sentences("Bring pens, paper, etc. and ink. We sell paper, etc. Next week we close."),
            vec![
                "Bring pens, paper, etc. and ink.",
                "We sell paper, etc.",
                "Next week we close.",
            ]
        );
        assert_eq!(
            sentences("It moved to the U.S. last year. The U.S. Army came."),
            vec!["It moved to the U.S. last year.", "The U.S. Army came."]
        );
    }

    #[test]
    fn ellipses_quotes_and_marks_end_sentences_where_speech_does() {
        assert_eq!(
            sentences("Wait... what? \"Really?!\" she said. Then... Nothing happened… Done."),
            vec![
                "Wait... what?",
                "\"Really?!\"",
                "she said.",
                "Then...",
                "Nothing happened…",
                "Done.",
            ]
        );
        assert_eq!(
            sentences("(He left.) Line one\nLine two。第二句！"),
            vec!["(He left.)", "Line one", "Line two。", "第二句！"]
        );
        // A URL or a version number is not a sentence end.
        assert_eq!(
            sentences("Visit example.com/v1.2 now. Bye"),
            vec!["Visit example.com/v1.2 now.", "Bye"]
        );
    }

    #[test]
    fn sentences_pack_up_to_the_window_and_long_ones_split_at_clauses() {
        let text = "One two three. Four five six. Seven eight nine.";
        assert_eq!(
            pack_sentences(text, 32, &|_| true),
            vec!["One two three. Four five six.", "Seven eight nine."]
        );
        // The family's one-request capacity bounds a segment as well.
        assert_eq!(
            pack_sentences(text, 480, &|segment| segment.split_whitespace().count()
                <= 3),
            vec!["One two three.", "Four five six.", "Seven eight nine."]
        );
        let long = "The price was $1,250.75 at 10:30, which was high; nobody bought it.";
        assert_eq!(
            pack_sentences(long, 40, &|_| true),
            vec![
                "The price was $1,250.75 at 10:30,",
                "which was high; nobody bought it."
            ]
        );
        let unbroken = "a".repeat(10);
        assert_eq!(
            pack_sentences(&unbroken, 4, &|_| true),
            vec!["aaaa", "aaaa", "aa"]
        );
    }
}

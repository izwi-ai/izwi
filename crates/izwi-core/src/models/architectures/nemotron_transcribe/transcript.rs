//! Output post-processing: tags, speaker turns, the generation-loop
//! guardrail and the online loop detector.
//!
//! The model writes `<spk:N>` speaker tags (prompted), `<ev:*>` event and
//! `<sty:*>` style tags (rich transcription). The text a request returns has
//! the event and style tags stripped (D8); the raw text keeps them.
//!
//! The card's guardrail (adapted from Qwen3-ASR) runs once on the final
//! text, in order:
//!
//! ```text
//! re.sub(r"(.)\1{20,}", r"\1", text, flags=re.DOTALL)
//! re.sub(r"(.{1,4096}?)\1{19,}", r"\1", text, flags=re.DOTALL)
//! ```
//!
//! Rust's `regex` has no backreferences, so both are hand-written scanners
//! with Python's semantics: left to right, the lazy group trying the
//! shortest block first, each match consuming every repeat, over characters.
//!
//! The online detector stops decoding once the last generated tokens are
//! one block repeated [`LOOP_REPEATS`] times, the case the second
//! substitution would collapse anyway; it saves the runaway to the token
//! budget that one-hour files otherwise hit.

/// A character repeated more than this many extra times collapses.
pub const CHAR_REPEAT_MIN_EXTRA: usize = 20;
/// A block repeated at least this many extra times collapses.
pub const BLOCK_REPEAT_MIN_EXTRA: usize = 19;
pub const MAX_BLOCK_CHARS: usize = 4096;
/// Copies of one block (in tokens) that stop decoding.
pub const LOOP_REPEATS: usize = 20;
/// The longest token block the online detector looks for.
pub const LOOP_MAX_BLOCK_TOKENS: usize = 256;

/// Why decoding stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronFinishReason {
    /// The stop token.
    Stop,
    /// The token budget.
    Length,
    /// The online loop detector.
    Repetition,
    /// The request was cancelled.
    Cancelled,
}

impl NemotronFinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Repetition => "repetition",
            Self::Cancelled => "cancelled",
        }
    }
}

/// `(.)\1{20,}` → `\1`.
pub fn collapse_char_runs(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let mut run = 1;
        while i + run < chars.len() && chars[i + run] == chars[i] {
            run += 1;
        }
        if run > CHAR_REPEAT_MIN_EXTRA {
            out.push(chars[i]);
            i += run;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// `(.{1,4096}?)\1{19,}` → `\1`.
pub fn collapse_block_repeats(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    'scan: while i < chars.len() {
        let remaining = chars.len() - i;
        let max_block = MAX_BLOCK_CHARS.min(remaining / (BLOCK_REPEAT_MIN_EXTRA + 1));
        for block in 1..=max_block {
            let mut repeats = 0;
            let mut at = i + block;
            while at + block <= chars.len() && chars[at..at + block] == chars[i..i + block] {
                repeats += 1;
                at += block;
            }
            if repeats >= BLOCK_REPEAT_MIN_EXTRA {
                out.extend(&chars[i..i + block]);
                i = at;
                continue 'scan;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// The card's guardrail, both substitutions in order.
pub fn apply_guardrail(text: &str) -> String {
    collapse_block_repeats(&collapse_char_runs(text))
}

/// Whether `tokens` end in one block of at most [`LOOP_MAX_BLOCK_TOKENS`]
/// repeated [`LOOP_REPEATS`] times.
pub fn ends_in_loop(tokens: &[u32]) -> bool {
    let n = tokens.len();
    let max_block = LOOP_MAX_BLOCK_TOKENS.min(n / LOOP_REPEATS);
    (1..=max_block).any(|block| {
        let start = n - block * LOOP_REPEATS;
        (start + block..n).all(|i| tokens[i] == tokens[i - block])
    })
}

/// A tag the model writes, by its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputTag {
    Speaker(u32),
    Event(String),
    Style(String),
    Think,
}

fn parse_tag(tag: &str) -> Option<OutputTag> {
    let inner = tag.strip_prefix('<')?.strip_suffix('>')?;
    if let Some(n) = inner.strip_prefix("spk:") {
        return n.parse().ok().map(OutputTag::Speaker);
    }
    if let Some(e) = inner.strip_prefix("ev:") {
        return Some(OutputTag::Event(e.to_string()));
    }
    if let Some(s) = inner.strip_prefix("sty:") {
        return Some(OutputTag::Style(s.to_string()));
    }
    matches!(inner, "think" | "/think").then_some(OutputTag::Think)
}

/// Splits text into plain runs and recognized tags.
fn segments(text: &str) -> Vec<Result<OutputTag, String>> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        match parse_tag(&rest[open..open + close + 1]) {
            Some(tag) => {
                if open > 0 {
                    out.push(Err(rest[..open].to_string()));
                }
                out.push(Ok(tag));
            }
            None => out.push(Err(rest[..open + close + 1].to_string())),
        }
        rest = &rest[open + close + 1..];
    }
    if !rest.is_empty() {
        out.push(Err(rest.to_string()));
    }
    out
}

fn tidy(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One speaker's run of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerTurn {
    /// `None` for text before the first tag.
    pub speaker: Option<u32>,
    pub text: String,
}

/// The parsed transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedTranscript {
    /// The returned text: speaker, event, style and think tags removed.
    pub text: String,
    /// The text as decoded (with special tokens), after the guardrail.
    pub raw: String,
    pub turns: Vec<SpeakerTurn>,
}

/// Parses decoded text (special tokens kept), after the guardrail.
pub fn parse_transcript(decoded: &str) -> ParsedTranscript {
    let raw = apply_guardrail(decoded.trim());
    let mut text = String::new();
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    let mut speaker = None;
    let mut current = String::new();
    let flush = |turns: &mut Vec<SpeakerTurn>, speaker: Option<u32>, current: &mut String| {
        let t = tidy(current);
        if !t.is_empty() {
            turns.push(SpeakerTurn { speaker, text: t });
        }
        current.clear();
    };
    for segment in segments(&raw) {
        match segment {
            Err(plain) => {
                text.push_str(&plain);
                current.push_str(&plain);
            }
            Ok(OutputTag::Speaker(n)) => {
                flush(&mut turns, speaker, &mut current);
                speaker = Some(n);
                text.push(' ');
            }
            Ok(_) => text.push(' '),
        }
    }
    flush(&mut turns, speaker, &mut current);
    ParsedTranscript {
        text: tidy(&text),
        raw,
        turns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goldens computed with Python 3 `re` and the card's two patterns.
    #[test]
    fn the_guardrail_matches_pythons_substitutions() {
        let cases: &[(&str, &str)] = &[
            ("", ""),
            ("hello", "hello"),
            // 21 'a's: one plus 20 repeats collapses; 20 does not.
            (&"a".repeat(21), "a"),
            (&"a".repeat(20), "a"),
            ("xaaaaaaaaaaaaaaaaaaaay", "xay"),
            // A 3-char block 20 times collapses to one copy.
            (
                "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc!",
                "abc!",
            ),
            // 19 copies stay.
            (
                "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
                "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(&apply_guardrail(input), expected, "{input:?}");
        }
        // 20 'a's survive the first pattern but are a 1-char block repeated
        // 20 times for the second, as in Python.
        assert_eq!(collapse_char_runs(&"a".repeat(20)), "a".repeat(20));
        // Lazy: the shortest block wins ("ab" over "abab").
        let looped = "ab".repeat(40);
        assert_eq!(apply_guardrail(&looped), "ab");
        // Multi-byte characters count as one.
        assert_eq!(apply_guardrail(&"é".repeat(25)), "é");
        // A phrase loop with a tail.
        let phrase = "I will go. ".repeat(30) + "Done.";
        assert_eq!(apply_guardrail(&phrase), "I will go. Done.");
    }

    #[test]
    fn the_loop_detector_needs_twenty_copies_at_the_end() {
        let mut tokens = vec![1u32, 2, 3];
        for _ in 0..19 {
            tokens.extend([7, 8]);
        }
        assert!(!ends_in_loop(&tokens));
        tokens.extend([7, 8]);
        assert!(ends_in_loop(&tokens));
        tokens.push(9);
        assert!(!ends_in_loop(&tokens));
        assert!(ends_in_loop(&[5; 20]));
        assert!(!ends_in_loop(&[5; 19]));
    }

    #[test]
    fn tags_split_turns_and_leave_plain_text() {
        let parsed = parse_transcript(
            "<spk:0> Hello there. <ev:laughter> <spk:1>Hi! <sty:stt:flat>Fine.<sty:end:flat> <spk:0> Bye.",
        );
        assert_eq!(parsed.text, "Hello there. Hi! Fine. Bye.");
        assert_eq!(
            parsed.turns,
            vec![
                SpeakerTurn {
                    speaker: Some(0),
                    text: "Hello there.".into()
                },
                SpeakerTurn {
                    speaker: Some(1),
                    text: "Hi! Fine.".into()
                },
                SpeakerTurn {
                    speaker: Some(0),
                    text: "Bye.".into()
                },
            ]
        );
        assert!(parsed.raw.contains("<ev:laughter>"));
        let untagged = parse_transcript("Just text, a < b > c.");
        assert_eq!(untagged.text, "Just text, a < b > c.");
        assert_eq!(
            untagged.turns,
            vec![SpeakerTurn {
                speaker: None,
                text: "Just text, a < b > c.".into()
            }]
        );
    }
}

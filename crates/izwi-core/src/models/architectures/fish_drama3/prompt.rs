//! The `reference-bracket-v1` prompt (`generate.py:96-193`).
//!
//! ```text
//! "<|im_start|>system\n"
//! "convert the provided text to speech"                               no references
//!   | "convert the provided text to speech reference to the following:"
//! for i, reference (1-based):
//!   "\n\n[Reference {i}]\nText: {tagged transcript}\nSpeech: "  <reference frames>
//! "<|im_end|>\n"
//! for turn in history:                                                 completed earlier turns
//!   "<|im_start|>user\n"  with_speaker_tag(turn.text)  "<|im_end|>\n"
//!   "<|im_start|>assistant\n<|voice|>"  <turn frames>  "<|im_end|>\n"
//! "<|im_start|>user\n"  with_speaker_tag(text)  "<|im_end|>\n"
//! "<|im_start|>assistant\n<|voice|>"                                   generation starts here
//! ```
//!
//! Each literal is tokenized on its own. Every position is a row
//! `[main, c0, …, c9]`: text rows are `[token, 0, …, 0]`, audio rows
//! `[semantic_start + c0, c0, …, c9]`. Speakers come only from the
//! `<|speaker:N|>` tags inside transcripts and text, never from the
//! reference number.
//!
//! White space here is the Unicode `White_Space` set upstream spells out
//! (`generate.py:45-53`), not Python's `str.strip()` set (which adds
//! U+001C..U+001F) nor Rust's `char::is_whitespace` (the same 25 code points
//! today, but tied to the toolchain's Unicode version).
//!
//! Codes are validated against the codebook size (4096) only: history takes
//! are model output, and the residual heads may legitimately emit codes
//! ≥ 1024. Only the codec clamps them.

use std::borrow::Cow;

use super::chunking::FishDrama3History;
use super::tokenizer::FishDrama3Tokenizer;
use crate::error::{Error, Result};

pub const SYSTEM_OPEN: &str = "<|im_start|>system\n";
pub const TTS_INSTRUCTION: &str = "convert the provided text to speech";
pub const REFERENCE_INSTRUCTION: &str =
    "convert the provided text to speech reference to the following:";
pub const MESSAGE_END: &str = "<|im_end|>\n";
pub const USER_OPEN: &str = "<|im_start|>user\n";
pub const GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";
/// Only the builder writes these; any other special token in caller text is
/// content (`generate.py:33-41`).
pub const STRUCTURAL_TOKENS: [&str; 6] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
];
/// `<|speaker:0|>` .. `<|speaker:15|>`.
pub const MAX_SPEAKER_ID: usize = 15;
pub const MAX_PROMPT_ROWS: usize = 8192;
/// The Unicode `White_Space` property (`generate.py:47-53`).
pub const UNICODE_WHITE_SPACE: [char; 25] = [
    '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', '\u{20}', '\u{85}', '\u{a0}', '\u{1680}',
    '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}',
    '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}',
];

const SPEAKER_TAG_OPEN: &str = "<|speaker:";
const DEFAULT_SPEAKER_TAG: &str = "<|speaker:0|> ";

pub fn is_white_space(c: char) -> bool {
    UNICODE_WHITE_SPACE.contains(&c)
}

/// `text.strip(UNICODE_WHITE_SPACE)`.
pub fn strip_white_space(text: &str) -> &str {
    text.trim_matches(is_white_space)
}

/// `text.lstrip(UNICODE_WHITE_SPACE)`.
pub fn lstrip_white_space(text: &str) -> &str {
    text.trim_start_matches(is_white_space)
}

/// `SPEAKER_TAG_PATTERN.match(text)` (`<\|speaker:([0-9]+)\|>`, ASCII
/// digits): the tag's byte length and its digits.
pub fn match_speaker_tag(text: &str) -> Option<(usize, &str)> {
    let rest = text.strip_prefix(SPEAKER_TAG_OPEN)?;
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    (digits > 0 && rest[digits..].starts_with("|>"))
        .then(|| (SPEAKER_TAG_OPEN.len() + digits + 2, &rest[..digits]))
}

/// `SPEAKER_TAG_PATTERN.finditer(text)`: each tag's byte range and digits,
/// left to right.
pub fn speaker_tags<'a>(text: &'a str) -> impl Iterator<Item = (usize, usize, &'a str)> + 'a {
    text.match_indices(SPEAKER_TAG_OPEN)
        .filter_map(move |(start, _)| {
            let (len, digits) = match_speaker_tag(&text[start..])?;
            Some((start, start + len, digits))
        })
}

/// The speaker of canonical digits (`0` to `15`, no leading zero).
pub fn speaker_index(digits: &str) -> Option<usize> {
    let canonical = digits.len() <= 2
        && !digits.is_empty()
        && (digits == "0" || !digits.starts_with('0'))
        && digits.bytes().all(|b| b.is_ascii_digit());
    canonical
        .then(|| digits.parse::<usize>().ok())
        .flatten()
        .filter(|&index| index <= MAX_SPEAKER_ID)
}

/// `validate_drama3_text` (`generate.py:70-86`): not empty, no structural
/// token, and every `<|speaker` a canonical tag 0..=15.
pub fn validate_text(text: &str, location: &str) -> Result<()> {
    if strip_white_space(text).is_empty() {
        return Err(Error::InvalidInput(format!(
            "{location}: text must not be empty"
        )));
    }
    if let Some(token) = STRUCTURAL_TOKENS
        .iter()
        .find(|token| text.contains(**token))
    {
        return Err(Error::InvalidInput(format!(
            "{location}: {token} is written by the prompt builder"
        )));
    }
    for (start, _) in text.match_indices("<|speaker") {
        let tagged = match_speaker_tag(&text[start..])
            .and_then(|(_, digits)| speaker_index(digits))
            .is_some();
        if !tagged {
            return Err(Error::InvalidInput(format!(
                "{location}: speaker tags must be <|speaker:0|> to <|speaker:{MAX_SPEAKER_ID}|>; got {text:?}"
            )));
        }
    }
    Ok(())
}

/// `with_speaker_tag` (`generate.py:89-93`): `"<|speaker:0|> " + text`
/// unless a tag leads the text after white space.
pub fn with_speaker_tag(text: &str) -> Cow<'_, str> {
    if match_speaker_tag(lstrip_white_space(text)).is_some() {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{DEFAULT_SPEAKER_TAG}{text}"))
    }
}

/// A reference transcript as its `Text: ` line holds it
/// (`generate.py:150-155`): the tag first, then any white space that led it.
pub fn tagged_transcript(transcript: &str) -> String {
    let tagged = with_speaker_tag(transcript);
    let body = lstrip_white_space(&tagged);
    let leading = &tagged[..tagged.len() - body.len()];
    match match_speaker_tag(body) {
        Some((tag, _)) => format!("{}{leading}{}", &body[..tag], &body[tag..]),
        // `with_speaker_tag` always leaves a leading tag.
        None => tagged.into_owned(),
    }
}

/// Codec codes `[codebooks][frames]` (upstream's `[10, frames]`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FishDrama3Codes {
    pub codebooks: Vec<Vec<u32>>,
}

impl FishDrama3Codes {
    pub fn new(codebooks: Vec<Vec<u32>>) -> Self {
        Self { codebooks }
    }

    pub fn frames(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }
}

/// One reference clip: its transcript and its codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3Reference {
    pub transcript: String,
    pub codes: FishDrama3Codes,
}

/// A completed earlier turn (a chunk or a take): its text and the frames
/// generated for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3HistoryTurn {
    pub text: String,
    pub codes: FishDrama3Codes,
}

/// What a row holds besides text ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3RowLayout {
    /// Codes per frame.
    pub codebooks: usize,
    /// Every code is below this (the semantic band's size).
    pub codebook_size: usize,
    /// `<|semantic:0|>`: an audio row's main id is this plus its first code.
    pub semantic_start: u32,
}

impl FishDrama3RowLayout {
    pub fn of(tokenizer: &FishDrama3Tokenizer) -> Self {
        let ids = tokenizer.ids();
        Self {
            codebooks: tokenizer.codebooks(),
            codebook_size: ids.semantic_codes(),
            semantic_start: ids.semantic_start,
        }
    }

    /// Values per row: the main id and the codes.
    pub fn width(&self) -> usize {
        self.codebooks + 1
    }
}

/// A built prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3Prompt {
    /// `[1 + codebooks][rows]`: `values[0]` holds the main ids,
    /// `values[1 + k]` codebook `k` (zero on text rows).
    pub values: Vec<Vec<u32>>,
    /// Positions: text tokens plus audio frames.
    pub rows: usize,
}

impl FishDrama3Prompt {
    /// Row `t` as upstream lays it out, `[main, c0, …]`.
    pub fn row(&self, t: usize) -> Option<Vec<u32>> {
        self.values
            .iter()
            .map(|values| values.get(t).copied())
            .collect()
    }

    pub fn main_ids(&self) -> &[u32] {
        self.values.first().map_or(&[], Vec::as_slice)
    }
}

/// A chunk's prompt and the history turns it holds, oldest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3ChunkPrompt<'h> {
    pub prompt: FishDrama3Prompt,
    pub history: Vec<&'h FishDrama3HistoryTurn>,
}

enum Segment<'a> {
    Text(Cow<'a, str>),
    Audio(&'a FishDrama3Codes),
}

enum Encoded<'a> {
    Ids(Vec<u32>),
    Frames(&'a FishDrama3Codes),
}

impl Encoded<'_> {
    fn rows(&self) -> usize {
        match self {
            Self::Ids(ids) => ids.len(),
            Self::Frames(codes) => codes.frames(),
        }
    }
}

/// `check_codes` (`generate.py:124-139`).
fn check_codes(layout: FishDrama3RowLayout, codes: &FishDrama3Codes, location: &str) -> Result<()> {
    let frames = codes.frames();
    if codes.codebooks.len() != layout.codebooks
        || frames == 0
        || codes
            .codebooks
            .iter()
            .any(|codebook| codebook.len() != frames)
    {
        let lengths = codes.codebooks.iter().map(Vec::len).collect::<Vec<_>>();
        return Err(Error::InvalidInput(format!(
            "{location}: codes must be integer [{}, frames >= 1]; got codebooks of {lengths:?} frames",
            layout.codebooks
        )));
    }
    if let Some(code) = codes
        .codebooks
        .iter()
        .flatten()
        .find(|&&code| code as usize >= layout.codebook_size)
    {
        return Err(Error::InvalidInput(format!(
            "{location}: codes must be in [0, {}); got {code}",
            layout.codebook_size
        )));
    }
    Ok(())
}

/// The literals and frames of a prompt, validated (`generate.py:122-175`).
fn prompt_segments<'a>(
    layout: FishDrama3RowLayout,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
) -> Result<Vec<Segment<'a>>> {
    validate_text(text, "text")?;
    let instruction = if references.is_empty() {
        TTS_INSTRUCTION
    } else {
        REFERENCE_INSTRUCTION
    };
    let mut segments = vec![
        Segment::Text(SYSTEM_OPEN.into()),
        Segment::Text(instruction.into()),
    ];
    for (index, reference) in references.iter().enumerate() {
        let location = format!("reference {}", index + 1);
        validate_text(&reference.transcript, &location)?;
        check_codes(layout, &reference.codes, &location)?;
        segments.push(Segment::Text(
            format!(
                "\n\n[Reference {}]\nText: {}\nSpeech: ",
                index + 1,
                tagged_transcript(&reference.transcript)
            )
            .into(),
        ));
        segments.push(Segment::Audio(&reference.codes));
    }
    segments.push(Segment::Text(MESSAGE_END.into()));
    for (index, turn) in history.into_iter().enumerate() {
        let location = format!("history {}", index + 1);
        validate_text(&turn.text, &location)?;
        check_codes(layout, &turn.codes, &location)?;
        segments.extend([
            Segment::Text(USER_OPEN.into()),
            Segment::Text(with_speaker_tag(&turn.text)),
            Segment::Text(MESSAGE_END.into()),
            Segment::Text(GENERATION_START.into()),
            Segment::Audio(&turn.codes),
            Segment::Text(MESSAGE_END.into()),
        ]);
    }
    segments.extend([
        Segment::Text(USER_OPEN.into()),
        Segment::Text(with_speaker_tag(text)),
        Segment::Text(MESSAGE_END.into()),
        Segment::Text(GENERATION_START.into()),
    ]);
    Ok(segments)
}

/// Tokenizes every literal on its own; returns the pieces and their rows.
fn encode<'a>(
    segments: Vec<Segment<'a>>,
    mut encode_segment: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<(Vec<Encoded<'a>>, usize)> {
    let mut rows = 0usize;
    let mut encoded = Vec::with_capacity(segments.len());
    for segment in segments {
        let piece = match segment {
            Segment::Text(literal) => Encoded::Ids(encode_segment(&literal)?),
            Segment::Audio(codes) => Encoded::Frames(codes),
        };
        rows += piece.rows();
        encoded.push(piece);
    }
    Ok((encoded, rows))
}

fn too_long(rows: usize, max_rows: usize) -> Error {
    Error::ContextLengthExceeded(format!("prompt has {rows} rows; at most {max_rows}"))
}

fn planned_rows_with<'a>(
    layout: FishDrama3RowLayout,
    encode_segment: impl FnMut(&str) -> Result<Vec<u32>>,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
) -> Result<usize> {
    let segments = prompt_segments(layout, references, history, text)?;
    Ok(encode(segments, encode_segment)?.1)
}

fn build_with<'a>(
    layout: FishDrama3RowLayout,
    encode_segment: impl FnMut(&str) -> Result<Vec<u32>>,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
    max_rows: usize,
) -> Result<FishDrama3Prompt> {
    let segments = prompt_segments(layout, references, history, text)?;
    let (pieces, rows) = encode(segments, encode_segment)?;
    // `generate.py:189-192`, checked before the rows are allocated.
    if rows > max_rows {
        return Err(too_long(rows, max_rows));
    }
    let mut values = vec![vec![0u32; rows]; layout.width()];
    let mut cursor = 0;
    for piece in &pieces {
        match piece {
            Encoded::Ids(ids) => values[0][cursor..cursor + ids.len()].copy_from_slice(ids),
            Encoded::Frames(codes) => {
                for (frame, &first) in codes.codebooks[0].iter().enumerate() {
                    values[0][cursor + frame] = layout.semantic_start + first;
                }
                for (row, codebook) in values[1..].iter_mut().zip(&codes.codebooks) {
                    row[cursor..cursor + codebook.len()].copy_from_slice(codebook);
                }
            }
        }
        cursor += piece.rows();
    }
    Ok(FishDrama3Prompt { values, rows })
}

/// The prompt's row count, every input validated, without building it: the
/// 8,192-row check at preparation.
pub fn planned_rows<'a>(
    tokenizer: &FishDrama3Tokenizer,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
) -> Result<usize> {
    planned_rows_with(
        FishDrama3RowLayout::of(tokenizer),
        |literal| tokenizer.encode_segment(literal),
        references,
        history,
        text,
    )
}

/// `build_drama3_prompt` with the 8,192-row limit.
pub fn build_prompt<'a>(
    tokenizer: &FishDrama3Tokenizer,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
) -> Result<FishDrama3Prompt> {
    build_prompt_with_limit(tokenizer, references, history, text, MAX_PROMPT_ROWS)
}

/// `build_drama3_prompt` refusing more than `max_rows` rows.
pub fn build_prompt_with_limit<'a>(
    tokenizer: &FishDrama3Tokenizer,
    references: &'a [FishDrama3Reference],
    history: impl IntoIterator<Item = &'a FishDrama3HistoryTurn>,
    text: &'a str,
    max_rows: usize,
) -> Result<FishDrama3Prompt> {
    build_with(
        FishDrama3RowLayout::of(tokenizer),
        |literal| tokenizer.encode_segment(literal),
        references,
        history,
        text,
        max_rows,
    )
}

/// `build_drama3_chunk_prompt` (`inference.py:1021-1042`): the prompt for
/// one chunk with as much of `history` as fits `max_rows`.
pub fn build_chunk_prompt<'h>(
    tokenizer: &FishDrama3Tokenizer,
    references: &[FishDrama3Reference],
    history: &'h FishDrama3History,
    text: &str,
    max_rows: usize,
) -> Result<FishDrama3ChunkPrompt<'h>> {
    let turns = history.select(max_rows, |turns| {
        planned_rows(tokenizer, references, turns.iter().copied(), text)
    })?;
    let prompt =
        build_prompt_with_limit(tokenizer, references, turns.iter().copied(), text, max_rows)?;
    Ok(FishDrama3ChunkPrompt {
        prompt,
        history: turns,
    })
}

/// A dialogue script as the chunker writes it (`chunking.py:164,172`): one
/// `"<|speaker:S|> text"` line per entry, joined by `"\n"`. A line's speaker
/// comes from its field only, so its text may not carry a tag.
pub fn render_script(lines: &[(usize, &str)]) -> Result<String> {
    if lines.is_empty() {
        return Err(Error::InvalidInput("script: no lines".into()));
    }
    let mut rendered = Vec::with_capacity(lines.len());
    for (index, &(speaker, text)) in lines.iter().enumerate() {
        let location = format!("script line {}", index + 1);
        if speaker > MAX_SPEAKER_ID {
            return Err(Error::InvalidInput(format!(
                "{location}: speaker {speaker} is not 0 to {MAX_SPEAKER_ID}"
            )));
        }
        let text = strip_white_space(text);
        if text.is_empty() {
            return Err(Error::InvalidInput(format!(
                "{location}: text must not be empty"
            )));
        }
        if text.contains("<|speaker") {
            return Err(Error::InvalidInput(format!(
                "{location}: the speaker comes from the line's speaker field, not a tag"
            )));
        }
        rendered.push(format!("<|speaker:{speaker}|> {text}"));
    }
    Ok(rendered.join("\n"))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::RefCell;
    use std::path::Path;

    use serde_json::Value;

    use super::*;
    use crate::models::architectures::fish_drama3::tokenizer::tests::{
        published_tokenizer, synthetic_tokenizer, LITERAL_GOLDENS,
    };

    /// Upstream's tiny test config (`test_fish_omni_generate.py:60-103`):
    /// 3 codebooks of 8 codes, the band at 32.
    const TINY: FishDrama3RowLayout = FishDrama3RowLayout {
        codebooks: 3,
        codebook_size: 8,
        semantic_start: 32,
    };

    /// `RecordingTokenizer` (`test_fish_omni_generate.py:117-128`): records
    /// every literal; each becomes `[3, 4]`.
    #[derive(Default)]
    struct Recording {
        parts: RefCell<Vec<String>>,
    }

    impl Recording {
        fn encode(&self, literal: &str) -> Result<Vec<u32>> {
            self.parts.borrow_mut().push(literal.to_string());
            Ok(vec![3, 4])
        }

        fn parts(&self) -> Vec<String> {
            self.parts.borrow().clone()
        }

        fn build(
            &self,
            references: &[FishDrama3Reference],
            history: &[FishDrama3HistoryTurn],
            text: &str,
        ) -> Result<FishDrama3Prompt> {
            build_with(
                TINY,
                |literal| self.encode(literal),
                references,
                history,
                text,
                MAX_PROMPT_ROWS,
            )
        }
    }

    fn codes(codebooks: &[&[u32]]) -> FishDrama3Codes {
        FishDrama3Codes::new(codebooks.iter().map(|codes| codes.to_vec()).collect())
    }

    fn reference(transcript: &str, codes: FishDrama3Codes) -> FishDrama3Reference {
        FishDrama3Reference {
            transcript: transcript.into(),
            codes,
        }
    }

    fn turn(text: &str, codes: FishDrama3Codes) -> FishDrama3HistoryTurn {
        FishDrama3HistoryTurn {
            text: text.into(),
            codes,
        }
    }

    fn invalid_message(result: Result<impl std::fmt::Debug>) -> String {
        match result {
            Err(Error::InvalidInput(message)) => message,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn white_space_is_the_unicode_property_not_str_strip() {
        // `generate.py:45-53`: U+001C..U+001F are not White_Space.
        assert_eq!(strip_white_space("\u{3000}\u{85} a \u{2029}"), "a");
        assert_eq!(strip_white_space("\u{1c}a\u{1f}"), "\u{1c}a\u{1f}");
        assert_eq!(lstrip_white_space("\u{200b}a"), "\u{200b}a");
        let rust: Vec<char> = (0..=0x10FFFFu32)
            .filter_map(char::from_u32)
            .filter(|c| c.is_whitespace())
            .collect();
        assert_eq!(rust, UNICODE_WHITE_SPACE);
    }

    #[test]
    fn speaker_tags_match_the_upstream_pattern() {
        // `generate.py:42`: `<\|speaker:([0-9]+)\|>`.
        assert_eq!(match_speaker_tag("<|speaker:12|> hi"), Some((14, "12")));
        assert_eq!(match_speaker_tag("<|speaker:007|>"), Some((15, "007")));
        for text in [
            "<|speaker:|>",
            "<|speaker:1>",
            "<|speaker:x|>",
            " <|speaker:1|>",
        ] {
            assert_eq!(match_speaker_tag(text), None, "{text:?}");
        }
        let tags = speaker_tags("a<|speaker:<|speaker:3|>b<|speaker:15|>").collect::<Vec<_>>();
        assert_eq!(tags, vec![(11, 24, "3"), (25, 39, "15")]);
        // `generate.py:82`: `str(int(digits)) != digits or int(digits) > 15`.
        assert_eq!(speaker_index("0"), Some(0));
        assert_eq!(speaker_index("15"), Some(15));
        for digits in ["", "00", "01", "16", "100", "x"] {
            assert_eq!(speaker_index(digits), None, "{digits:?}");
        }
    }

    #[test]
    fn caller_text_is_validated_as_upstream_does() {
        // `test_fish_omni_generate.py:215-224` and `generate.py:72-86`.
        for text in [
            " ",
            "\u{3000}\n",
            "hello <|voice|>",
            "<|speaker:16|> hello",
            "<|speaker:00|> hi",
            "<|speaker:01|> hi",
            "<|speaker:x|> hi",
            "<|speaker:u0|> hi",
            "<|speaker hi",
            "<|speaker_change|> hi",
            "a <|semantic:12|>",
            "a <|semantic:",
            "<|im_start|>",
            "x<|im_end|>",
            "<|text|>",
            "<|interleave|>",
        ] {
            assert!(validate_text(text, "text").is_err(), "{text:?}");
        }
        // Other special tokens are content (`generate.py:33`).
        for text in [
            "<|speaker:15|> hi\n<|speaker:0|> yes",
            "<|fim_prefix|> hi",
            "<|endofturn|> hi",
            "\u{1c}",
            "[sighs] <whisper>hi</whisper>",
        ] {
            assert!(validate_text(text, "text").is_ok(), "{text:?}");
        }
        assert_eq!(
            invalid_message(validate_text("hi <|voice|>", "history 2")),
            "history 2: <|voice|> is written by the prompt builder"
        );
    }

    #[test]
    fn the_speaker_tag_rules_follow_upstream() {
        // `generate.py:89-93`.
        assert_eq!(with_speaker_tag("hello"), "<|speaker:0|> hello");
        assert_eq!(with_speaker_tag(" \t hi"), "<|speaker:0|>  \t hi");
        assert_eq!(with_speaker_tag("<|speaker:3|>hi"), "<|speaker:3|>hi");
        assert_eq!(
            with_speaker_tag("\u{2003}<|speaker:2|> x"),
            "\u{2003}<|speaker:2|> x"
        );
        // U+001C is not White_Space, so the tag does not lead.
        assert_eq!(
            with_speaker_tag("\u{1c}<|speaker:1|> x"),
            "<|speaker:0|> \u{1c}<|speaker:1|> x"
        );
        // `generate.py:150-155`, `test_fish_omni_generate.py:171-174`.
        assert_eq!(
            tagged_transcript("  <|speaker:0|> reference"),
            "<|speaker:0|>   reference"
        );
        assert_eq!(
            tagged_transcript("\u{3000}\u{a0}<|speaker:4|>x"),
            "<|speaker:4|>\u{3000}\u{a0}x"
        );
        assert_eq!(tagged_transcript("  plain"), "<|speaker:0|>   plain");
        assert_eq!(tagged_transcript("<|speaker:2|>ref"), "<|speaker:2|>ref");
    }

    #[test]
    fn a_reference_free_prompt_has_the_upstream_literals() {
        // `test_fish_omni_generate.py:139-155`.
        let recording = Recording::default();
        let prompt = recording.build(&[], &[], "hello").unwrap();
        assert_eq!(
            recording.parts(),
            [
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "<|speaker:0|> hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        );
        assert_eq!((prompt.rows, prompt.values.len()), (14, 4));
        assert!(prompt.values[1..]
            .iter()
            .all(|codes| codes.iter().all(|&c| c == 0)));
        assert_eq!(prompt.main_ids(), [3, 4].repeat(7));
    }

    #[test]
    fn reference_codes_become_audio_rows_and_white_space_moves_behind_the_tag() {
        // `test_fish_omni_generate.py:157-178`.
        let recording = Recording::default();
        let references = [reference(
            "  <|speaker:0|> reference",
            codes(&[&[1, 2], &[3, 4], &[5, 6]]),
        )];
        let prompt = recording
            .build(&references, &[], "<|speaker:0|>hello")
            .unwrap();
        let parts = recording.parts();
        assert_eq!(parts[1], REFERENCE_INSTRUCTION);
        assert_eq!(
            parts[2],
            "\n\n[Reference 1]\nText: <|speaker:0|>   reference\nSpeech: "
        );
        assert!(parts.contains(&"<|speaker:0|>hello".to_string()));
        // [3, 4] system, instruction, reference line, then the 2 frames.
        assert_eq!(prompt.row(6), Some(vec![33, 1, 3, 5]));
        assert_eq!(prompt.row(7), Some(vec![34, 2, 4, 6]));
        assert_eq!(prompt.row(8), Some(vec![3, 0, 0, 0]));
        assert_eq!(prompt.rows, 2 * 3 + 2 + 2 * 5);
    }

    #[test]
    fn speakers_zero_to_fifteen_are_kept_everywhere() {
        // `test_fish_omni_generate.py:180-200`.
        let recording = Recording::default();
        let one = codes(&[&[1], &[2], &[3]]);
        let text = "<|speaker:15|> hi\n<|speaker:0|> yes";
        recording
            .build(
                &[reference("<|speaker:2|>ref", one.clone())],
                &[turn("<|speaker:3|> before", one.clone())],
                text,
            )
            .unwrap();
        let parts = recording.parts();
        assert!(parts.contains(&text.to_string()));
        assert!(parts.contains(&"<|speaker:3|> before".to_string()));
        assert!(parts.contains(&"\n\n[Reference 1]\nText: <|speaker:2|>ref\nSpeech: ".to_string()));
        let err = invalid_message(recording.build(&[], &[turn("<|speaker:16|> no", one)], "hi"));
        assert!(err.starts_with("history 1: speaker tags"), "{err}");
    }

    #[test]
    fn bad_codes_are_refused_and_head_codes_above_the_codec_range_are_not() {
        // `test_fish_omni_generate.py:225-233`, `generate.py:124-139`.
        let recording = Recording::default();
        for (name, bad) in [
            ("two codebooks", codes(&[&[0, 0, 0], &[0, 0, 0]])),
            ("code 8", codes(&[&[8, 8], &[8, 8], &[8, 8]])),
            ("no frames", codes(&[&[], &[], &[]])),
            ("ragged", codes(&[&[1, 2], &[3], &[5, 6]])),
        ] {
            let err = invalid_message(recording.build(&[reference("ref", bad)], &[], "hi"));
            assert!(err.starts_with("reference 1: codes must"), "{name}: {err}");
        }
        // The published codebook size: a residual code of 3000 (a head
        // output the codec would clamp) is a valid history code.
        let layout = FishDrama3RowLayout {
            codebooks: 10,
            codebook_size: 4096,
            semantic_start: 248_099,
        };
        let take = turn(
            "before",
            FishDrama3Codes::new(
                vec![vec![4095]; 1]
                    .into_iter()
                    .chain(vec![vec![3000]; 9])
                    .collect(),
            ),
        );
        let prompt = build_with(
            layout,
            |_| Ok(vec![1]),
            &[],
            [&take],
            "now",
            MAX_PROMPT_ROWS,
        )
        .unwrap();
        let audio = (0..prompt.rows)
            .filter_map(|t| prompt.row(t))
            .find(|row| row[0] >= 248_099)
            .unwrap();
        assert_eq!(audio, [vec![252_194, 4095], vec![3000; 9]].concat());
    }

    #[test]
    fn history_turns_are_written_as_completed_turns() {
        // `test_fish_omni_generate.py:510-535`.
        let recording = Recording::default();
        let plain = recording.build(&[], &[], "now").unwrap();
        let history = [turn("before", codes(&[&[1, 2], &[3, 4], &[5, 6]]))];
        let prompt = recording.build(&[], &history, "now").unwrap();
        let parts = recording.parts();
        assert_eq!(
            parts[parts.len() - 9..],
            [
                "<|im_start|>user\n",
                "<|speaker:0|> before",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "<|speaker:0|> now",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        );
        // Five two-token literals plus the two audio frames.
        assert_eq!(prompt.rows - plain.rows, 12);
        let start = (0..prompt.rows)
            .find(|&t| prompt.row(t) == Some(vec![33, 1, 3, 5]))
            .unwrap();
        assert_eq!(prompt.row(start + 1), Some(vec![34, 2, 4, 6]));
    }

    #[test]
    fn the_row_limit_is_checked_before_building_and_planned_rows_agree() {
        // `generate.py:189-192`: more than the limit is refused.
        let recording = Recording::default();
        let history = [turn("before", codes(&[&[1, 2], &[3, 4], &[5, 6]]))];
        let rows = planned_rows_with(TINY, |l| recording.encode(l), &[], &history, "now").unwrap();
        assert_eq!(rows, 26);
        let fits = build_with(TINY, |l| recording.encode(l), &[], &history, "now", 26).unwrap();
        assert_eq!(fits.rows, 26);
        let err = build_with(TINY, |l| recording.encode(l), &[], &history, "now", 25).unwrap_err();
        assert!(
            matches!(&err, Error::ContextLengthExceeded(m) if m == "prompt has 26 rows; at most 25"),
            "{err}"
        );
        // Validation runs before counting.
        assert!(planned_rows_with(TINY, |l| recording.encode(l), &[], &history, " ").is_err());
    }

    #[test]
    fn chunk_prompts_carry_history_and_drop_it_when_too_long() {
        // `test_fish_omni_generate.py:572-579`: the first chunk is pinned and
        // only the latest other one is kept.
        let tokenizer = synthetic_tokenizer();
        let frame = FishDrama3Codes::new(vec![vec![1], vec![2], vec![3]]);
        let mut history = FishDrama3History::new();
        let mut used = Vec::new();
        for chunk in ["one", "two", "three"] {
            let built =
                build_chunk_prompt(&tokenizer, &[], &history, chunk, MAX_PROMPT_ROWS).unwrap();
            used.push(
                built
                    .history
                    .iter()
                    .map(|t| t.text.clone())
                    .collect::<Vec<_>>(),
            );
            history.push(turn(chunk, frame.clone()));
        }
        assert_eq!(used, [vec![], vec!["one"], vec!["one", "two"]]);

        // `test_fish_omni_generate.py:599-604`: a limit that fits only the
        // bare prompt drops the history.
        let bare = build_prompt(&tokenizer, &[], [], "x").unwrap().rows;
        let mut history = FishDrama3History::new();
        let mut used = Vec::new();
        for chunk in ["one", "two"] {
            let built = build_chunk_prompt(&tokenizer, &[], &history, chunk, bare).unwrap();
            assert_eq!(built.prompt.rows, bare);
            used.push(built.history.len());
            history.push(turn(chunk, frame.clone()));
        }
        assert_eq!(used, [0, 0]);
        // Nothing fits: the bare prompt's error.
        let err = build_chunk_prompt(&tokenizer, &[], &history, "two", bare - 1).unwrap_err();
        assert!(matches!(err, Error::ContextLengthExceeded(_)), "{err}");
        // Invalid text fails at once, not as "too long".
        let err = build_chunk_prompt(&tokenizer, &[], &history, "<|voice|>", bare).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    }

    #[test]
    fn scripts_render_as_the_chunker_tags_segments() {
        assert_eq!(
            render_script(&[(0, "Ready?"), (1, " Always. "), (15, "Fine.")]).unwrap(),
            "<|speaker:0|> Ready?\n<|speaker:1|> Always.\n<|speaker:15|> Fine."
        );
        let bad: [&[(usize, &str)]; 4] = [
            &[(16, "hi")],
            &[(0, " \u{3000}")],
            &[(0, "a <|speaker:1|> b")],
            &[],
        ];
        for bad in bad {
            assert!(render_script(bad).is_err(), "{bad:?}");
        }
    }

    fn literal_ids(literal: &str) -> &'static [u32] {
        LITERAL_GOLDENS
            .iter()
            .find(|(text, _)| *text == literal)
            .map(|(_, ids)| *ids)
            .unwrap_or_else(|| panic!("no golden for {literal:?}"))
    }

    #[test]
    #[ignore = "needs the published tokenizer.json in IZWI_FISH_DRAMA3_TOKENIZER_PATH"]
    fn the_published_tokenizer_builds_the_golden_prompts() {
        let tokenizer = published_tokenizer();
        // Plan §2.3: "Hello, how are you?" is a 27-row plain prompt.
        let plain = build_prompt(&tokenizer, &[], [], "Hello, how are you?").unwrap();
        let expected = [
            SYSTEM_OPEN,
            TTS_INSTRUCTION,
            MESSAGE_END,
            USER_OPEN,
            "<|speaker:0|> Hello, how are you?",
            MESSAGE_END,
            GENERATION_START,
        ]
        .iter()
        .flat_map(|literal| literal_ids(literal).to_vec())
        .collect::<Vec<_>>();
        assert_eq!(plain.rows, 27);
        assert_eq!(plain.main_ids(), expected);
        assert!(plain.values[1..]
            .iter()
            .all(|codes| codes.iter().all(|&c| c == 0)));

        // One reference: the `[Reference 1]` golden, then its frames.
        let frames = FishDrama3Codes::new(
            (0..10u32)
                .map(|k| (0..3u32).map(|f| (k * 131 + f * 977) % 1024).collect())
                .collect(),
        );
        let references = [reference(
            "We asked over twenty different people, and they all said it was his.",
            frames.clone(),
        )];
        assert_eq!(
            planned_rows(&tokenizer, &references, [], "Hello, how are you?").unwrap(),
            3 + 11 + 30 + 3 + 2 + 3 + 7 + 2 + 4
        );
        let prompt = build_prompt(&tokenizer, &references, [], "Hello, how are you?").unwrap();
        let line = "\n\n[Reference 1]\nText: <|speaker:0|> We asked over twenty different people, and they all said it was his.\nSpeech: ";
        let head = [SYSTEM_OPEN, REFERENCE_INSTRUCTION, line]
            .iter()
            .flat_map(|literal| literal_ids(literal).to_vec())
            .collect::<Vec<_>>();
        assert_eq!(prompt.main_ids()[..head.len()], head);
        for frame in 0..3 {
            let row = prompt.row(head.len() + frame).unwrap();
            assert_eq!(row[0], 248_099 + frames.codebooks[0][frame]);
            let codes = frames
                .codebooks
                .iter()
                .map(|c| c[frame])
                .collect::<Vec<_>>();
            assert_eq!(row[1..], codes);
        }
        assert_eq!(
            prompt.main_ids()[prompt.rows - 4..],
            [248_045, 74455, 198, 248_081]
        );
    }

    /// A fixture written by `scripts/fixtures/generate_fish_drama3_goldens.py`.
    pub(crate) fn golden(name: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fish_drama3")
            .join(name);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!(
                "{}: {err}; generate it with scripts/fixtures/generate_fish_drama3_goldens.py",
                path.display()
            )
        });
        serde_json::from_str(&text).unwrap()
    }

    pub(crate) fn golden_str(value: &Value) -> &str {
        value
            .as_str()
            .unwrap_or_else(|| panic!("not a string: {value}"))
    }

    pub(crate) fn golden_usize(value: &Value) -> usize {
        value
            .as_u64()
            .unwrap_or_else(|| panic!("not a count: {value}")) as usize
    }

    pub(crate) fn golden_strings(value: &Value) -> Vec<String> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| golden_str(item).to_string())
            .collect()
    }

    pub(crate) fn golden_ids(value: &Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|id| golden_usize(id) as u32)
            .collect()
    }

    pub(crate) fn golden_codes(value: &Value) -> FishDrama3Codes {
        FishDrama3Codes::new(value.as_array().unwrap().iter().map(golden_ids).collect())
    }

    pub(crate) fn golden_references(value: &Value) -> Vec<FishDrama3Reference> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|r| reference(golden_str(&r["transcript"]), golden_codes(&r["codes"])))
            .collect()
    }

    /// Upstream's exception class maps to the error variant, and its message
    /// to ours up to the first "; got" (Python's repr is not Rust's Debug).
    #[track_caller]
    pub(crate) fn assert_golden_error(expected: &Value, err: &Error, context: &str) {
        let message = match (golden_str(&expected["type"]), err) {
            ("Drama3PromptTooLongError", Error::ContextLengthExceeded(message)) => message,
            ("InvalidInputError", Error::InvalidInput(message)) => message,
            (kind, err) => panic!("{context}: upstream raised {kind}, izwi returned {err:?}"),
        };
        let cut = |m: &str| m.split("; got").next().unwrap_or(m).to_string();
        assert_eq!(
            cut(message),
            cut(golden_str(&expected["message"])),
            "{context}"
        );
    }

    #[test]
    #[ignore = "needs prompt_goldens.json from scripts/fixtures/generate_fish_drama3_goldens.py and IZWI_FISH_DRAMA3_TOKENIZER_PATH"]
    fn upstream_prompt_goldens() {
        let goldens = golden("prompt_goldens.json");
        assert_eq!(goldens["schema"], "izwi.fish_drama3.prompt_goldens");
        assert_eq!(goldens["schema_version"], 1);
        let constants = &goldens["constants"];
        assert_eq!(constants["tts_instruction"], TTS_INSTRUCTION);
        assert_eq!(constants["reference_instruction"], REFERENCE_INSTRUCTION);
        assert_eq!(constants["message_end"], MESSAGE_END);
        assert_eq!(constants["generation_start"], GENERATION_START);
        assert_eq!(
            golden_strings(&constants["structural_tokens"]),
            STRUCTURAL_TOKENS
        );
        assert_eq!(golden_usize(&constants["max_speaker_id"]), MAX_SPEAKER_ID);
        assert_eq!(golden_usize(&constants["max_prompt_rows"]), MAX_PROMPT_ROWS);
        let white_space = UNICODE_WHITE_SPACE.map(u32::from).to_vec();
        assert_eq!(golden_ids(&constants["unicode_white_space"]), white_space);

        let tokenizer = published_tokenizer();
        for literal in goldens["literals"].as_array().unwrap() {
            let text = golden_str(&literal["text"]);
            let ids = tokenizer.encode_segment(text).unwrap();
            assert_eq!(ids, golden_ids(&literal["ids"]), "{text:?}");
        }
        for case in goldens["validate"].as_array().unwrap() {
            let text = golden_str(&case["text"]);
            match validate_text(text, "text") {
                Ok(()) => assert!(case["error"].is_null(), "{text:?} passed"),
                Err(err) => assert_golden_error(&case["error"], &err, text),
            }
        }
        for case in goldens["with_speaker_tag"].as_array().unwrap() {
            let text = golden_str(&case["text"]);
            assert_eq!(
                with_speaker_tag(text),
                golden_str(&case["tagged"]),
                "{text:?}"
            );
        }

        let layout = FishDrama3RowLayout::of(&tokenizer);
        let cases = goldens["cases"].as_array().unwrap();
        assert!(cases.len() >= 12);
        for case in cases {
            let name = golden_str(&case["name"]);
            let text = golden_str(&case["text"]);
            let references = golden_references(&case["references"]);
            let history = case["history"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| turn(golden_str(&t["text"]), golden_codes(&t["codes"])))
                .collect::<Vec<_>>();
            let max_rows = golden_usize(&case["max_prompt_rows"]);
            let parts = RefCell::new(Vec::new());
            let built = build_with(
                layout,
                |literal| {
                    parts.borrow_mut().push(literal.to_string());
                    tokenizer.encode_segment(literal)
                },
                &references,
                &history,
                text,
                max_rows,
            );
            if !case["error"].is_null() {
                assert_golden_error(&case["error"], &built.unwrap_err(), name);
                continue;
            }
            let prompt = built.unwrap_or_else(|err| panic!("{name}: {err}"));
            let rows = (0..prompt.rows)
                .map(|t| prompt.row(t).unwrap())
                .collect::<Vec<_>>();
            let expected = case["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(golden_ids)
                .collect::<Vec<_>>();
            assert_eq!(rows, expected, "{name}");
            assert_eq!(
                parts.into_inner(),
                golden_strings(&case["segments"]),
                "{name}"
            );
            let planned = planned_rows(&tokenizer, &references, &history, text).unwrap();
            assert_eq!(planned, prompt.rows, "{name}");
        }
    }
}

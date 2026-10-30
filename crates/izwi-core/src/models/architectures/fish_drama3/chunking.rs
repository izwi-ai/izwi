//! Long-text chunking, a port of fish-speech's `fish_omni/chunking.py`
//! (`0f8a12e8`), and the history policy of its chunk loop
//! (`inference.py:912-1043`). Pure: no model, no device.
//!
//! Text is cut into chunks of about 300 UTF-8 bytes (a threshold, not a
//! cap: a protected span longer than that stays whole) with at most three
//! speakers, preferring sentence ends, then clause marks, then spaces, then
//! raw length. Decimals stay whole, and nothing splits inside `<|…|>`,
//! `<tag>…</tag>` or `[…]`. Each chunk is generated on its own; the first
//! chunk stays pinned in the next prompts, with the latest one, as completed
//! turns.
//!
//! Strings are walked by code point, as Python indexes them. Python's
//! Unicode classes are reproduced from the Unicode data (`icu_properties`):
//! `\w` is a letter (`L*`), a number (`N*`) or `_`, and `str.isdigit` is
//! `Numeric_Type` Decimal or Digit (superscripts and circled digits
//! included). The two can only disagree on code points one Unicode version
//! has and the other lacks.

use std::collections::{HashMap, VecDeque};

use icu_properties::props::{GeneralCategory, GeneralCategoryGroup, NumericType};
use icu_properties::CodePointMapData;

use super::prompt::{speaker_tags, strip_white_space, FishDrama3HistoryTurn};
use crate::error::{Error, Result};

/// A threshold, not a hard cap (`chunking.py:21-22`).
pub const CHUNK_BYTES: usize = 300;
/// A new chunk starts at the fourth distinct speaker.
pub const CHUNK_MAX_SPEAKERS: usize = 3;
/// Frames per chunk when the request asks for none (about 27.9 s).
pub const CHUNK_MAX_NEW_TOKENS: usize = 600;
/// History besides the pinned first chunk.
pub const HISTORY_RECENT_TURNS: usize = 1;

const SENTENCE_ENDS: &[char] = &['.', '!', '?', ';', '。', '！', '？', '；', '…', '\n'];
const CLAUSE_MARKS: &[char] = &[',', ':', '，', '：', '、'];
const SPACES: &[char] = &[' '];

/// Python's `\w`: `str.isalnum()` or `_`, that is `L*`, `N*` or `_`.
fn is_word(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    let category = CodePointMapData::<GeneralCategory>::new().get(c);
    GeneralCategoryGroup::Letter.contains(category)
        || GeneralCategoryGroup::Number.contains(category)
}

/// Python's `str.isdigit`.
fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    matches!(
        CodePointMapData::<NumericType>::new().get(c),
        NumericType::Decimal | NumericType::Digit
    )
}

/// A character of a paired tag's name, `[\w-]`.
fn is_name(c: char) -> bool {
    c == '-' || is_word(c)
}

/// `PROTECTED_SPAN` (`chunking.py:35-37`) as `finditer` runs it:
///
/// ```text
/// <\|[^|>]*\|>  |  <([A-Za-z][\w-]*)\b[^>]*>.*?</\1>  |  \[[^\[\]]*\]      (re.S)
/// ```
///
/// The first and last alternatives are deterministic. The paired tag tries
/// names longest first (the greedy `[\w-]*` backtracking), each needing a
/// `\b` after it, ends its open tag at the first `>`, and takes the nearest
/// `</name>` after that (the lazy `.*?`). Close tags are indexed once, so a
/// text full of unclosed tags stays linear.
struct ProtectedSpans<'c> {
    chars: &'c [char],
    /// The first `>` at or after each index (`len + 1` entries).
    next_gt: Vec<Option<usize>>,
    /// Each `</name>`'s start, ascending, by name.
    closes: HashMap<String, Vec<usize>>,
}

impl<'c> ProtectedSpans<'c> {
    fn new(chars: &'c [char]) -> Self {
        let mut next_gt = vec![None; chars.len() + 1];
        for index in (0..chars.len()).rev() {
            next_gt[index] = if chars[index] == '>' {
                Some(index)
            } else {
                next_gt[index + 1]
            };
        }
        let mut closes: HashMap<String, Vec<usize>> = HashMap::new();
        for start in 0..chars.len() {
            if chars[start] != '<'
                || chars.get(start + 1) != Some(&'/')
                || !chars.get(start + 2).is_some_and(char::is_ascii_alphabetic)
            {
                continue;
            }
            let end = (start + 3..chars.len())
                .find(|&index| !is_name(chars[index]))
                .unwrap_or(chars.len());
            if chars.get(end) == Some(&'>') {
                let name = chars[start + 2..end].iter().collect();
                closes.entry(name).or_default().push(start);
            }
        }
        Self {
            chars,
            next_gt,
            closes,
        }
    }

    /// The end of the span starting at `start`, if one does.
    fn match_at(&self, start: usize) -> Option<usize> {
        self.special(start)
            .or_else(|| self.paired(start))
            .or_else(|| self.bracketed(start))
    }

    /// `<\|[^|>]*\|>`.
    fn special(&self, start: usize) -> Option<usize> {
        let c = self.chars;
        if c.get(start) != Some(&'<') || c.get(start + 1) != Some(&'|') {
            return None;
        }
        let stop = (start + 2..c.len()).find(|&index| c[index] == '|' || c[index] == '>')?;
        (c[stop] == '|' && c.get(stop + 1) == Some(&'>')).then_some(stop + 2)
    }

    /// `<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>`.
    fn paired(&self, start: usize) -> Option<usize> {
        let c = self.chars;
        if c.get(start) != Some(&'<') || !c.get(start + 1).is_some_and(char::is_ascii_alphabetic) {
            return None;
        }
        let longest = (start + 2..c.len())
            .find(|&index| !is_name(c[index]))
            .unwrap_or(c.len());
        // Name characters are never `>`, so every name length ends its open
        // tag at the same `>`.
        let open_end = self.next_gt[longest]?;
        for end in (start + 2..=longest).rev() {
            let boundary = is_word(c[end - 1]) != c.get(end).is_some_and(|&next| is_word(next));
            if !boundary {
                continue;
            }
            let name: String = c[start + 1..end].iter().collect();
            let Some(positions) = self.closes.get(&name) else {
                continue;
            };
            let first = positions.partition_point(|&position| position <= open_end);
            if let Some(&close) = positions.get(first) {
                return Some(close + end - start + 2);
            }
        }
        None
    }

    /// `\[[^\[\]]*\]`.
    fn bracketed(&self, start: usize) -> Option<usize> {
        let c = self.chars;
        if c.get(start) != Some(&'[') {
            return None;
        }
        let stop = (start + 1..c.len()).find(|&index| c[index] == '[' || c[index] == ']')?;
        (c[stop] == ']').then_some(stop + 1)
    }
}

/// A run of text and whether it is a protected span.
type Piece = (String, bool);

/// `_protected_pieces` (`chunking.py:47-56`).
fn protected_pieces(text: &str) -> Vec<Piece> {
    let chars = text.chars().collect::<Vec<_>>();
    let spans = ProtectedSpans::new(&chars);
    let collect = |range: std::ops::Range<usize>| chars[range].iter().collect::<String>();
    let (mut pieces, mut last, mut index) = (Vec::new(), 0, 0);
    while index < chars.len() {
        match spans.match_at(index) {
            Some(end) => {
                if index > last {
                    pieces.push((collect(last..index), false));
                }
                pieces.push((collect(index..end), true));
                last = end;
                index = end;
            }
            None => index += 1,
        }
    }
    if last < chars.len() {
        pieces.push((collect(last..chars.len()), false));
    }
    pieces
}

/// `_breaks_at` (`chunking.py:59-68`): a mark, except a `.` between digits.
fn breaks_at(chars: &[char], index: usize, marks: &[char]) -> bool {
    let c = chars[index];
    if !marks.contains(&c) {
        return false;
    }
    !(c == '.'
        && 0 < index
        && index + 1 < chars.len()
        && is_digit(chars[index - 1])
        && is_digit(chars[index + 1]))
}

/// `_break_at_marks` (`chunking.py:71-85`): free pieces longer than
/// `length` bytes split after each mark, keeping it.
fn break_at_marks(pieces: Vec<Piece>, length: usize, marks: &[char]) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let chars = text.chars().collect::<Vec<_>>();
        let mut start = 0;
        for index in 0..chars.len() {
            if breaks_at(&chars, index, marks) {
                out.push((chars[start..=index].iter().collect(), false));
                start = index + 1;
            }
        }
        if start < chars.len() {
            out.push((chars[start..].iter().collect(), false));
        }
    }
    out
}

/// `_break_by_length` (`chunking.py:88-103`): free pieces still longer than
/// `length` split on character boundaries.
fn break_by_length(pieces: Vec<Piece>, length: usize) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let mut current = String::new();
        for c in text.chars() {
            if !current.is_empty() && current.len() + c.len_utf8() > length {
                out.push((std::mem::take(&mut current), false));
            }
            current.push(c);
        }
        out.push((current, false));
    }
    out
}

/// `split_by_length` (`chunking.py:106-133`): chunks of about `length`
/// bytes, packed sentence by sentence. Text that fits is returned whole and
/// unstripped; split chunks are stripped of white space.
pub fn split_by_length(text: &str, length: usize) -> Vec<String> {
    if strip_white_space(text).is_empty() {
        return Vec::new();
    }
    if text.len() <= length {
        return vec![text.to_string()];
    }
    let mut pieces = break_at_marks(protected_pieces(text), 0, SENTENCE_ENDS);
    for marks in [CLAUSE_MARKS, SPACES] {
        pieces = break_at_marks(pieces, length, marks);
    }
    let pieces = break_by_length(pieces, length);

    let mut chunks = Vec::new();
    let mut current = String::new();
    for (piece, _) in pieces {
        if !current.is_empty() && current.len() + piece.len() > length {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&piece);
    }
    chunks.push(current);
    chunks
        .iter()
        .map(|chunk| strip_white_space(chunk))
        .filter(|chunk| !chunk.is_empty())
        .map(str::to_string)
        .collect()
}

/// `parse_speaker_segments` (`chunking.py:136-143`): `(speaker digits,
/// text)` per tagged segment, unstripped; text before the first tag is
/// speaker `"0"`, and blank segments are dropped.
pub fn parse_speaker_segments(text: &str) -> Vec<(&str, &str)> {
    let tags = speaker_tags(text).collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(tags.len() + 1);
    let head = &text[..tags.first().map_or(text.len(), |&(start, _, _)| start)];
    if !strip_white_space(head).is_empty() {
        segments.push(("0", head));
    }
    for (index, &(_, end, speaker)) in tags.iter().enumerate() {
        let body_end = tags
            .get(index + 1)
            .map_or(text.len(), |&(start, _, _)| start);
        let body = &text[end..body_end];
        if !strip_white_space(body).is_empty() {
            segments.push((speaker, body));
        }
    }
    segments
}

/// `split_drama3_text` (`chunking.py:146-181`): the generation chunks of
/// `text`, in order.
///
/// Untagged text is split by length. Tagged text is split into speaker
/// segments, long ones cut further with each piece re-tagged
/// `"<|speaker:S|> piece"`, and regrouped (joined by `"\n"`): a new chunk
/// starts when a segment would bring in one speaker too many or exceed
/// `max_bytes`, so A, B, A can share a chunk.
pub fn split_text(text: &str, max_bytes: usize, max_speakers: usize) -> Vec<String> {
    if speaker_tags(text).next().is_none() {
        return split_by_length(text, max_bytes);
    }
    let mut segments = Vec::new();
    for (speaker, body) in parse_speaker_segments(text) {
        for piece in split_by_length(body, max_bytes) {
            segments.push((speaker, format!("<|speaker:{speaker}|> {piece}")));
        }
    }

    let mut chunks = Vec::new();
    let mut batch: Vec<String> = Vec::new();
    let mut speakers: Vec<&str> = Vec::new();
    let mut size = 0;
    for (speaker, segment) in segments {
        let mut new_speaker = !speakers.contains(&speaker);
        let too_many_speakers = new_speaker && speakers.len() >= max_speakers;
        let too_long = !batch.is_empty() && size + segment.len() > max_bytes;
        if too_many_speakers || too_long {
            chunks.push(batch.join("\n"));
            batch.clear();
            speakers.clear();
            size = 0;
            new_speaker = true;
        }
        size += segment.len();
        batch.push(segment);
        if new_speaker {
            speakers.push(speaker);
        }
    }
    if !batch.is_empty() {
        chunks.push(batch.join("\n"));
    }
    chunks
}

/// The chunks a request generates (`inference.py:939`): the whole text with
/// `single_chunk`, else [`split_text`] at the defaults, or the text itself
/// when that yields none (the prompt builder then refuses blank text).
pub fn plan_chunks(text: &str, single_chunk: bool) -> Vec<String> {
    if single_chunk {
        return vec![text.to_string()];
    }
    let chunks = split_text(text, CHUNK_BYTES, CHUNK_MAX_SPEAKERS);
    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

/// `Drama3History` (`chunking.py:184-206`): the context for the next chunk,
/// the first chunk pinned plus the latest `recent_turns` others.
///
/// The chunk loop pushes a chunk only when it produced frames
/// (`inference.py:1012-1017`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3History {
    pinned: Option<FishDrama3HistoryTurn>,
    recent: VecDeque<FishDrama3HistoryTurn>,
    recent_turns: usize,
}

impl Default for FishDrama3History {
    fn default() -> Self {
        Self::new()
    }
}

impl FishDrama3History {
    pub fn new() -> Self {
        Self::with_recent_turns(HISTORY_RECENT_TURNS)
    }

    pub fn with_recent_turns(recent_turns: usize) -> Self {
        Self {
            pinned: None,
            recent: VecDeque::with_capacity(recent_turns),
            recent_turns,
        }
    }

    /// Records a completed chunk: the first is pinned, later ones fill a
    /// window of `recent_turns` (a `deque(maxlen=…)`).
    pub fn push(&mut self, turn: FishDrama3HistoryTurn) {
        if self.pinned.is_none() {
            self.pinned = Some(turn);
            return;
        }
        if self.recent_turns == 0 {
            return;
        }
        if self.recent.len() == self.recent_turns {
            self.recent.pop_front();
        }
        self.recent.push_back(turn);
    }

    /// Histories to try, longest first: the oldest recent turns dropped one
    /// by one, then the pinned one, ending with none.
    pub fn candidates(&self) -> Vec<Vec<&FishDrama3HistoryTurn>> {
        let pinned = self.pinned.iter().collect::<Vec<_>>();
        let mut candidates = (0..=self.recent.len())
            .map(|start| {
                pinned
                    .iter()
                    .copied()
                    .chain(self.recent.iter().skip(start))
                    .collect()
            })
            .collect::<Vec<_>>();
        if !pinned.is_empty() {
            candidates.push(Vec::new());
        }
        candidates
    }

    /// `build_drama3_chunk_prompt`'s choice (`inference.py:1021-1042`): the
    /// first candidate whose prompt has at most `max_rows` rows, by
    /// `rows_of`. References are part of every candidate's prompt, so they
    /// are always kept. Errors from `rows_of` (invalid text) end the search;
    /// when nothing fits, the history-free prompt's length is the error.
    pub fn select<'h>(
        &'h self,
        max_rows: usize,
        mut rows_of: impl FnMut(&[&'h FishDrama3HistoryTurn]) -> Result<usize>,
    ) -> Result<Vec<&'h FishDrama3HistoryTurn>> {
        let mut rows = 0;
        for candidate in self.candidates() {
            rows = rows_of(&candidate)?;
            if rows <= max_rows {
                return Ok(candidate);
            }
        }
        Err(Error::ContextLengthExceeded(format!(
            "prompt has {rows} rows; at most {max_rows}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::fish_drama3::prompt::tests::{
        assert_golden_error, golden, golden_codes, golden_references, golden_str, golden_strings,
        golden_usize,
    };
    use crate::models::architectures::fish_drama3::prompt::{build_chunk_prompt, FishDrama3Codes};
    use crate::models::architectures::fish_drama3::tokenizer::tests::published_tokenizer;

    fn turn(text: &str) -> FishDrama3HistoryTurn {
        FishDrama3HistoryTurn {
            text: text.into(),
            codes: FishDrama3Codes::new(vec![vec![1]; 2]),
        }
    }

    #[track_caller]
    fn assert_pieces(text: &str, expected: &[(&str, bool)]) {
        let pieces = protected_pieces(text);
        let pieces = pieces
            .iter()
            .map(|(piece, protected)| (piece.as_str(), *protected))
            .collect::<Vec<_>>();
        assert_eq!(pieces, expected, "{text:?}");
    }

    #[test]
    fn python_character_classes_follow_the_unicode_data() {
        // `\w` is L*, N* or `_` (`chunking.py:36`).
        for c in ['a', 'Z', '0', '_', 'é', '中', '٣', '²', 'Ⅷ'] {
            assert!(is_word(c), "{c:?}");
        }
        // Marks are not, even Other_Alphabetic ones (U+0BBE, Tamil vowel sign
        // AA) that Rust's `is_alphanumeric` takes, nor circled letters.
        for c in ['-', ' ', '\u{301}', '\u{bbe}', 'Ⓐ', '‿'] {
            assert!(!is_word(c), "{c:?}");
        }
        // `str.isdigit`: decimal or digit, not other numerics (`chunking.py:66-67`).
        for c in ['7', '٣', '²', '①', '₉'] {
            assert!(is_digit(c), "{c:?}");
        }
        for c in ['½', 'Ⅷ', '一', 'a'] {
            assert!(!is_digit(c), "{c:?}");
        }
    }

    #[test]
    fn protected_spans_follow_the_upstream_pattern() {
        // `chunking.py:35-37,47-56`, worked by hand.
        assert_pieces(
            "a <|x|> [b] <w a=1>c</w> <a->d</a> e",
            &[
                ("a ", false),
                ("<|x|>", true),
                (" ", false),
                ("[b]", true),
                (" ", false),
                ("<w a=1>c</w>", true),
                (" ", false),
                // `a-` fails `\b` before `>`; `a` holds it.
                ("<a->d</a>", true),
                (" e", false),
            ],
        );
        // Unclosed tags and brackets stay free.
        assert_pieces("<b>x [y", &[("<b>x [y", false)]);
        // A bracket cannot hold a bracket.
        assert_pieces(
            "[a [b] c]",
            &[("[a ", false), ("[b]", true), (" c]", false)],
        );
        // The lazy `.*?` takes the nearest close; the back reference must
        // match the name exactly.
        assert_pieces("<i>a</i>b</i>", &[("<i>a</i>", true), ("b</i>", false)]);
        assert_pieces("<em>x</e>", &[("<em>x</e>", false)]);
        // `[^|>]*` stops at `>`.
        assert_pieces("<|a>b|>", &[("<|a>b|>", false)]);
        // Non-ASCII names: `é` is a word character, U+0BBE is not (so the
        // name is `a`, whose close never comes).
        assert_pieces("<aé>x</aé>", &[("<aé>x</aé>", true)]);
        assert_pieces(
            "<a\u{bbe}>x</a\u{bbe}>",
            &[("<a\u{bbe}>x</a\u{bbe}>", false)],
        );
        // `re.S`: the span crosses lines.
        assert_pieces(
            "<whisper>a.\nb.</whisper>!",
            &[("<whisper>a.\nb.</whisper>", true), ("!", false)],
        );
    }

    #[test]
    fn short_text_is_one_unchanged_chunk() {
        // `test_drama3_chunking.py:15-17`.
        assert_eq!(
            split_by_length("  Hello there.  ", 300),
            ["  Hello there.  "]
        );
        assert!(split_by_length(" \n ", 300).is_empty());
        // `chunking.py:113`: U+001C is not white space.
        assert_eq!(split_by_length("\u{1c}", 300), ["\u{1c}"]);
    }

    #[test]
    fn splits_prefer_sentence_ends() {
        // `test_drama3_chunking.py:19-33`.
        let text = (0..40)
            .map(|i| format!("Sentence number {i} is here."))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = split_by_length(&text, 100);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.len() <= 100 && c.ends_with('.')));
        assert_eq!(chunks.join(" "), text);

        let text = "今天天气很好，我们去公园散步。".repeat(20);
        let chunks = split_by_length(&text, 100);
        assert!(chunks.iter().all(|c| c.len() <= 100 && c.ends_with('。')));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn splits_fall_back_to_clauses_spaces_then_length() {
        // `test_drama3_chunking.py:35-49`.
        let clauses = vec!["这是一个很长的分句"; 20].join("，");
        let chunks = split_by_length(&clauses, 60);
        assert!(chunks[..chunks.len() - 1].iter().all(|c| c.ends_with('，')));
        assert_eq!(chunks.concat(), clauses);

        let words = vec!["word"; 100].join(" ");
        let chunks = split_by_length(&words, 50);
        assert!(chunks.iter().all(|c| c.len() <= 50));
        assert_eq!(chunks.join(" "), words);

        let unbroken = "字".repeat(100);
        let chunks = split_by_length(&unbroken, 50);
        assert!(chunks.iter().all(|c| c.len() <= 50));
        assert_eq!(chunks.concat(), unbroken);
    }

    #[test]
    fn worked_splits_match_the_upstream_passes() {
        // `chunking.py:117-133` by hand: sentence pass ["One.", " Two three,
        // four five six."], clause pass splits after ",", space pass after
        // each " ", then packing to 10 bytes and stripping.
        assert_eq!(
            split_by_length("One. Two three, four five six.", 10),
            ["One. Two", "three,", "four five", "six."]
        );
        // `chunking.py:63-68`: the "." of 3.14 is kept, the one after it
        // breaks.
        assert_eq!(split_by_length("Pi 3.14. Go", 6), ["Pi", "3.14.", "Go"]);
        // Arabic-Indic and superscript digits are digits too.
        // (Without them the sentence pass would give ["x ٣.", "١٤ y"] and
        // ["a².", "³ b"].)
        assert_eq!(split_by_length("x ٣.١٤ y", 8), ["x", "٣.١٤", "y"]);
        assert_eq!(split_by_length("a².³ b", 7), ["a².³", "b"]);
        // A protected span longer than the threshold stays whole.
        let whisper = format!("<whisper>{}</whisper>", "secret words. ".repeat(3));
        assert_eq!(
            split_by_length(&format!("Go. {whisper} Done."), 20),
            ["Go.", whisper.as_str(), "Done."]
        );
    }

    #[test]
    fn decimals_and_protected_spans_stay_whole() {
        // `test_drama3_chunking.py:51-58`.
        let whisper = format!("<whisper>{}</whisper>", "secret words. ".repeat(10));
        let text = format!(
            "{}{whisper} [laughing] done.",
            "Pi is 3.14159 roughly. ".repeat(10)
        );
        let chunks = split_by_length(&text, 60);
        assert!(chunks.contains(&whisper));
        assert!(whisper.len() > 60);
        assert!(!chunks.iter().any(|c| c.ends_with("3.")));
        assert!(chunks.iter().any(|c| c.contains("[laughing]")));
    }

    #[test]
    fn speaker_segments_follow_the_upstream_split() {
        // `test_drama3_chunking.py:62-66`.
        assert_eq!(
            parse_speaker_segments("intro <|speaker:1|>Hi<|speaker:2|> Yo"),
            [("0", "intro "), ("1", "Hi"), ("2", " Yo")]
        );
        // Blank bodies drop; non-canonical digits pass through unvalidated.
        assert_eq!(
            parse_speaker_segments(" <|speaker:01|>a<|speaker:3|> \n"),
            [("01", "a")]
        );
        assert_eq!(
            parse_speaker_segments("<|speaker:3|> \n<|speaker:x|>b"),
            [("3", " \n<|speaker:x|>b")]
        );
    }

    #[test]
    fn a_new_chunk_starts_at_the_fourth_speaker() {
        // `test_drama3_chunking.py:68-79`.
        let text = "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2<|speaker:2|>C1<|speaker:3|>D1";
        assert_eq!(
            split_text(text, 300, 3),
            [
                "<|speaker:0|> A1\n<|speaker:1|> B1\n<|speaker:0|> A2\n<|speaker:2|> C1",
                "<|speaker:3|> D1",
            ]
        );
        // `chunking.py:170`: or when the chunk would pass `max_bytes`
        // (18 + 18 > 20; separators are not counted).
        assert_eq!(
            split_text("<|speaker:0|>aaaa<|speaker:1|>bbbb", 20, 3),
            ["<|speaker:0|> aaaa", "<|speaker:1|> bbbb"]
        );
        assert_eq!(
            split_text("<|speaker:0|>aaaa<|speaker:1|>bbbb", 36, 3),
            ["<|speaker:0|> aaaa\n<|speaker:1|> bbbb"]
        );
        // `chunking.py:115-116,164`: a short body keeps its own leading space.
        assert_eq!(
            split_text("<|speaker:1|>Hi<|speaker:2|> Yo", 300, 3),
            ["<|speaker:1|> Hi\n<|speaker:2|>  Yo"]
        );
    }

    #[test]
    fn long_segments_are_cut_and_keep_their_tag() {
        // `test_drama3_chunking.py:81-87`.
        let long = "This sentence is long enough. ".repeat(10);
        let chunks = split_text(&format!("<|speaker:0|>{long}<|speaker:0|>Short."), 100, 3);
        assert!(chunks.len() > 2);
        assert!(chunks
            .iter()
            .flat_map(|c| c.split('\n'))
            .all(|s| s.starts_with("<|speaker:0|> ")));
        assert!(chunks.last().unwrap().ends_with("Short."));
    }

    #[test]
    fn untagged_text_is_split_by_length() {
        // `test_drama3_chunking.py:89-91`.
        assert_eq!(
            split_text("Hello.", CHUNK_BYTES, CHUNK_MAX_SPEAKERS),
            ["Hello."]
        );
        assert!(split_text("  ", CHUNK_BYTES, CHUNK_MAX_SPEAKERS).is_empty());
        // `inference.py:939`.
        assert_eq!(plan_chunks("  ", false), ["  "]);
        assert_eq!(plan_chunks("a. b.", true), ["a. b."]);
        assert_eq!(plan_chunks("Hello.", false), ["Hello."]);
    }

    #[test]
    fn the_history_pins_the_first_chunk_and_keeps_the_latest() {
        // `test_drama3_chunking.py:95-101`.
        let texts = |history: &FishDrama3History| {
            history
                .candidates()
                .iter()
                .map(|turns| turns.iter().map(|t| t.text.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let mut history = FishDrama3History::with_recent_turns(1);
        assert_eq!(texts(&history), [Vec::<String>::new()]);
        history.push(turn("chunk 0"));
        assert_eq!(texts(&history), [vec!["chunk 0"], vec![]]);
        for index in 1..3 {
            history.push(turn(&format!("chunk {index}")));
        }
        assert_eq!(
            texts(&history),
            [vec!["chunk 0", "chunk 2"], vec!["chunk 0"], vec![]]
        );
        // `chunking.py:188-206` with two recent turns, and `deque(maxlen=0)`.
        let mut history = FishDrama3History::with_recent_turns(2);
        let mut none = FishDrama3History::with_recent_turns(0);
        for index in 0..4 {
            history.push(turn(&format!("c{index}")));
            none.push(turn(&format!("c{index}")));
        }
        assert_eq!(
            texts(&history),
            [vec!["c0", "c2", "c3"], vec!["c0", "c3"], vec!["c0"], vec![]]
        );
        assert_eq!(texts(&none), [vec!["c0"], vec![]]);
    }

    #[test]
    fn selection_drops_the_oldest_history_first_then_the_pinned_chunk() {
        // `inference.py:1029-1042`, rows standing in for prompt lengths.
        let mut history = FishDrama3History::new();
        history.push(turn("pinned"));
        history.push(turn("recent"));
        let rows = |turns: &[&FishDrama3HistoryTurn]| -> Result<usize> {
            Ok(10
                + 5 * turns.iter().filter(|t| t.text == "pinned").count()
                + 7 * turns.iter().filter(|t| t.text == "recent").count())
        };
        let chosen = |max_rows| -> Result<Vec<String>> {
            Ok(history
                .select(max_rows, rows)?
                .iter()
                .map(|t| t.text.clone())
                .collect())
        };
        assert_eq!(chosen(22).unwrap(), ["pinned", "recent"]);
        assert_eq!(chosen(21).unwrap(), ["pinned"]);
        assert_eq!(chosen(14).unwrap(), Vec::<String>::new());
        let err = chosen(9).unwrap_err();
        assert!(
            matches!(&err, Error::ContextLengthExceeded(m) if m == "prompt has 10 rows; at most 9"),
            "{err}"
        );
        // A refusal is not "too long": it ends the search.
        let mut calls = 0;
        let err = history
            .select(100, |_| {
                calls += 1;
                Err(Error::InvalidInput("text: bad".into()))
            })
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert_eq!(calls, 1);
    }

    fn sorted(marks: &[char]) -> String {
        let mut marks = marks.to_vec();
        marks.sort_unstable();
        marks.into_iter().collect()
    }

    #[test]
    #[ignore = "needs chunk_goldens.json from scripts/fixtures/generate_fish_drama3_goldens.py"]
    fn upstream_chunk_goldens() {
        let goldens = golden("chunk_goldens.json");
        assert_eq!(goldens["schema"], "izwi.fish_drama3.chunk_goldens");
        assert_eq!(goldens["schema_version"], 1);
        let constants = &goldens["constants"];
        assert_eq!(golden_usize(&constants["chunk_bytes"]), CHUNK_BYTES);
        assert_eq!(golden_usize(&constants["max_speakers"]), CHUNK_MAX_SPEAKERS);
        assert_eq!(
            golden_usize(&constants["chunk_max_new_tokens"]),
            CHUNK_MAX_NEW_TOKENS
        );
        assert_eq!(
            golden_usize(&constants["recent_turns"]),
            HISTORY_RECENT_TURNS
        );
        assert_eq!(constants["sentence_ends"], sorted(SENTENCE_ENDS));
        assert_eq!(constants["clause_marks"], sorted(CLAUSE_MARKS));
        assert_eq!(constants["spaces"], sorted(SPACES));

        let corpus = &goldens["corpus"];
        for case in goldens["protected_pieces"].as_array().unwrap() {
            let text = golden_str(&case["text"]);
            let expected = case["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|piece| {
                    (
                        golden_str(&piece[0]).to_string(),
                        piece[1].as_bool().unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(protected_pieces(text), expected, "{text:?}");
        }
        for case in goldens["split_by_length"].as_array().unwrap() {
            let (name, length) = (golden_str(&case["name"]), golden_usize(&case["length"]));
            let chunks = split_by_length(golden_str(&corpus[name]), length);
            assert_eq!(
                chunks,
                golden_strings(&case["chunks"]),
                "{name} at {length}"
            );
        }
        for case in goldens["parse_speaker_segments"].as_array().unwrap() {
            let text = golden_str(&case["text"]);
            let expected = case["segments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| (golden_str(&s[0]), golden_str(&s[1])))
                .collect::<Vec<_>>();
            assert_eq!(parse_speaker_segments(text), expected, "{text:?}");
        }
        for case in goldens["split_text"].as_array().unwrap() {
            let name = golden_str(&case["name"]);
            let (max_bytes, max_speakers) = (
                golden_usize(&case["max_bytes"]),
                golden_usize(&case["max_speakers"]),
            );
            let chunks = split_text(golden_str(&corpus[name]), max_bytes, max_speakers);
            assert_eq!(
                chunks,
                golden_strings(&case["chunks"]),
                "{name} at {max_bytes} bytes, {max_speakers} speakers"
            );
        }
        for case in goldens["history_candidates"].as_array().unwrap() {
            let mut history =
                FishDrama3History::with_recent_turns(golden_usize(&case["recent_turns"]));
            for text in golden_strings(&case["pushed"]) {
                history.push(turn(&text));
            }
            let candidates = history
                .candidates()
                .iter()
                .map(|turns| turns.iter().map(|t| t.text.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let expected = case["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .map(golden_strings)
                .collect::<Vec<_>>();
            assert_eq!(candidates, expected, "{case}");
        }
    }

    #[test]
    #[ignore = "needs chunk_goldens.json from scripts/fixtures/generate_fish_drama3_goldens.py and IZWI_FISH_DRAMA3_TOKENIZER_PATH"]
    fn upstream_history_selection_goldens() {
        let goldens = golden("chunk_goldens.json");
        let tokenizer = published_tokenizer();
        let plans = goldens["history_selection"].as_array().unwrap();
        assert!(!plans.is_empty());
        for plan in plans {
            let name = golden_str(&plan["name"]);
            let references = golden_references(&plan["references"]);
            let chunks = plan["chunks"].as_array().unwrap();
            for run in plan["runs"].as_array().unwrap() {
                let max_rows = golden_usize(&run["max_prompt_rows"]);
                let steps = run["steps"].as_array().unwrap();
                assert_eq!(steps.len(), chunks.len(), "{name}");
                let mut history = FishDrama3History::new();
                for (index, (chunk, step)) in chunks.iter().zip(steps).enumerate() {
                    let context = format!("{name} at {max_rows} rows, chunk {index}");
                    let text = golden_str(&chunk["text"]);
                    match build_chunk_prompt(&tokenizer, &references, &history, text, max_rows) {
                        Ok(built) => {
                            assert!(step["error"].is_null(), "{context}: upstream refused it");
                            assert_eq!(built.prompt.rows, golden_usize(&step["rows"]), "{context}");
                            let kept = built
                                .history
                                .iter()
                                .map(|t| t.text.as_str())
                                .collect::<Vec<_>>();
                            let expected = step["history"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|i| golden_str(&chunks[golden_usize(i)]["text"]))
                                .collect::<Vec<_>>();
                            assert_eq!(kept, expected, "{context}");
                        }
                        Err(err) => assert_golden_error(&step["error"], &err, &context),
                    }
                    if !chunk["codes"].is_null() {
                        history.push(FishDrama3HistoryTurn {
                            text: text.to_string(),
                            codes: golden_codes(&chunk["codes"]),
                        });
                    }
                }
            }
        }
    }
}

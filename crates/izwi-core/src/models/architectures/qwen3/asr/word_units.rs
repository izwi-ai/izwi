//! The forced aligner's word units, as the reference
//! `Qwen3ForceAlignProcessor.encode_timestamp` (qwen-asr 0.0.6) builds them.
//!
//! - Korean: soynlp's `LTokenizer` with the reference's dictionary. A
//!   whitespace token longer than two characters splits after its longest
//!   prefix in the dictionary; a token with no such prefix stays whole.
//! - Japanese: the reference tags words with nagisa, a BiLSTM tagger that is
//!   too heavy to port; lindera with an embedded IPADIC grows a binary by
//!   about 48 MB. [`japanese_pieces`] approximates nagisa with script runs.
//! - Every other language: whitespace tokens, each cleaned, with every CJK
//!   ideograph a unit of its own.
//!
//! Cleaning (`clean_token`) keeps letters, numbers and ASCII `'` and drops
//! every other character, also inside a token: `We’re` becomes `Were` and
//! `AI—autonomous` becomes `AIautonomous`.
//!
//! Each unit also carries the byte range of the source text it stands for:
//! from the unit to the next unit in the same whitespace token, or to the
//! token's end. The first unit of a token starts at the token start. So
//! punctuation stays with its word (`Hello,`, `“quoted”`, `好，`).

use std::collections::HashSet;
use std::ops::Range;
use std::sync::OnceLock;

use regex::Regex;

/// One unit the aligner times: the cleaned text it puts in the prompt and
/// the byte range of the source text it stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentUnit {
    pub text: String,
    pub source: Range<usize>,
}

impl AlignmentUnit {
    /// The source text the unit stands for, punctuation included.
    pub fn surface<'a>(&self, text: &'a str) -> &'a str {
        &text[self.source.clone()]
    }
}

/// The word units of `text` in `language` (a name or code; anything other
/// than Japanese or Korean takes the whitespace rules).
pub fn alignment_units(text: &str, language: Option<&str>) -> Vec<AlignmentUnit> {
    let pieces = match Segmentation::of(language) {
        Segmentation::Japanese => japanese_pieces(text),
        Segmentation::Korean => korean_pieces(text),
        Segmentation::Whitespace => whitespace_pieces(text),
    };
    units_with_sources(text, pieces)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segmentation {
    Whitespace,
    Japanese,
    Korean,
}

impl Segmentation {
    fn of(language: Option<&str>) -> Self {
        let Some(language) = language else {
            return Self::Whitespace;
        };
        let language = language.trim().to_ascii_lowercase();
        let primary = language.split(['-', '_']).next().unwrap_or_default();
        match primary {
            "japanese" | "ja" | "jp" | "jpn" => Self::Japanese,
            "korean" | "ko" | "kor" => Self::Korean,
            _ => Self::Whitespace,
        }
    }
}

/// The kept characters of one unit: its text and the byte offset of its
/// first character in the source.
type Piece = (usize, String);

/// Python's `str.split()` whitespace, which adds the four ASCII separators
/// to Rust's.
fn is_split_whitespace(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// Byte ranges of the whitespace-separated tokens of `text`.
fn whitespace_tokens(text: &str) -> Vec<Range<usize>> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if is_split_whitespace(ch) {
            if let Some(start) = start.take() {
                tokens.push(start..idx);
            }
        } else {
            start.get_or_insert(idx);
        }
    }
    if let Some(start) = start {
        tokens.push(start..text.len());
    }
    tokens
}

/// The reference `is_kept_char`: ASCII `'` or a letter or number (general
/// category L* or N*). Combining marks are dropped.
fn is_kept_char(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_alphanumeric() || ch == '\'';
    }
    static LETTER_OR_NUMBER: OnceLock<Regex> = OnceLock::new();
    ch.is_alphanumeric()
        && LETTER_OR_NUMBER
            .get_or_init(|| Regex::new(r"^[\p{L}\p{N}]$").expect("letter-or-number class"))
            .is_match(ch.encode_utf8(&mut [0u8; 4]))
}

/// The reference `is_cjk_char`: CJK ideographs only, no kana or Hangul.
fn is_cjk_ideograph(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
    )
}

/// The cleaned piece of `text[range]`, if any character survives.
fn cleaned_piece(text: &str, range: Range<usize>) -> Option<Piece> {
    let mut piece: Option<Piece> = None;
    for (idx, ch) in text[range.clone()].char_indices() {
        if is_kept_char(ch) {
            piece
                .get_or_insert_with(|| (range.start + idx, String::new()))
                .1
                .push(ch);
        }
    }
    piece
}

/// `tokenize_space_lang`: each whitespace token cleaned, then split so that
/// every CJK ideograph is a unit and the characters between them form one.
fn whitespace_pieces(text: &str) -> Vec<Piece> {
    let mut pieces = Vec::new();
    for token in whitespace_tokens(text) {
        let mut run: Option<Piece> = None;
        for (idx, ch) in text[token.clone()].char_indices() {
            if !is_kept_char(ch) {
                continue;
            }
            let at = token.start + idx;
            if is_cjk_ideograph(ch) {
                pieces.extend(run.take());
                pieces.push((at, ch.to_string()));
            } else {
                run.get_or_insert_with(|| (at, String::new())).1.push(ch);
            }
        }
        pieces.extend(run);
    }
    pieces
}

fn korean_dictionary() -> &'static HashSet<&'static str> {
    static WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| {
        include_str!("korean_dict_words.txt")
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    })
}

/// `tokenize_korean`: soynlp `LTokenizer(scores=dictionary)` with every
/// dictionary word scoring 1 and every other prefix 0, then each part
/// cleaned. The highest score wins and the longer prefix breaks ties, so a
/// token splits after its longest dictionary prefix of at least two
/// characters, and stays whole when it has none.
fn korean_pieces(text: &str) -> Vec<Piece> {
    let dictionary = korean_dictionary();
    let mut pieces = Vec::new();
    for token in whitespace_tokens(text) {
        let ends = text[token.clone()]
            .char_indices()
            .map(|(idx, ch)| token.start + idx + ch.len_utf8())
            .collect::<Vec<_>>();
        let split = if ends.len() <= 2 {
            token.end
        } else {
            ends[1..]
                .iter()
                .rev()
                .copied()
                .find(|end| dictionary.contains(&text[token.start..*end]))
                .unwrap_or(token.end)
        };
        pieces.extend(cleaned_piece(text, token.start..split));
        pieces.extend(cleaned_piece(text, split..token.end));
    }
    pieces
}

/// Script of one kept character, for [`japanese_pieces`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Script {
    Kanji,
    Hiragana,
    Katakana,
    Digit,
    Other,
}

fn japanese_script(ch: char) -> Script {
    match ch as u32 {
        0x3005..=0x3007 => Script::Kanji, // 々 〆 〇
        0x3041..=0x309F => Script::Hiragana,
        0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Script::Katakana,
        _ if is_cjk_ideograph(ch) => Script::Kanji,
        _ if ch.is_numeric() => Script::Digit,
        _ => Script::Other,
    }
}

/// Hiragana that usually stand alone as particles after a noun; they are
/// never taken as okurigana.
const JAPANESE_PARTICLES: &str = "はがをにでとのもへやかねよ";

/// An approximation of `tokenize_japanese` (nagisa tagging, each word
/// cleaned): runs of one script, with
/// - a kanji run taking the hiragana character right after it as okurigana
///   unless it is a particle (`良い`, `飲み`, but `私` `は`),
/// - every digit a unit of its own, as nagisa tags numbers,
/// - characters that are not kept separating units.
///
/// On 20 sentences tagged by nagisa 0.3.0, 94 % of these unit boundaries are
/// nagisa boundaries and they find 72 % of nagisa's (178 units against
/// 234): the units are coarser, `東京大学` and `しています` where nagisa has
/// `東京` `大学` and `し` `て` `い` `ます`.
fn japanese_pieces(text: &str) -> Vec<Piece> {
    let mut runs: Vec<(Script, Piece)> = Vec::new();
    let mut previous_end = None;
    for (idx, ch) in text.char_indices() {
        if !is_kept_char(ch) {
            previous_end = None;
            continue;
        }
        let script = japanese_script(ch);
        let adjacent = previous_end == Some(idx);
        previous_end = Some(idx + ch.len_utf8());
        match runs.last_mut() {
            Some((last, (_, run))) if adjacent && *last == script && script != Script::Digit => {
                run.push(ch);
            }
            _ => runs.push((script, (idx, ch.to_string()))),
        }
    }

    let mut pieces: Vec<Piece> = Vec::with_capacity(runs.len());
    let mut kanji_end = None;
    for (script, (mut start, mut run)) in runs {
        if script == Script::Hiragana && kanji_end == Some(start) {
            let okurigana = run
                .chars()
                .next()
                .filter(|ch| !JAPANESE_PARTICLES.contains(*ch));
            if let (Some(ch), Some((_, kanji))) = (okurigana, pieces.last_mut()) {
                kanji.push(ch);
                run.drain(..ch.len_utf8());
                start += ch.len_utf8();
            }
        }
        kanji_end = (script == Script::Kanji).then_some(start + run.len());
        if !run.is_empty() {
            pieces.push((start, run));
        }
    }
    pieces
}

/// Attach to each piece the source range it stands for (see the module
/// docs). Every piece lies inside one whitespace token.
fn units_with_sources(text: &str, pieces: Vec<Piece>) -> Vec<AlignmentUnit> {
    let tokens = whitespace_tokens(text);
    let token_of = |offset: usize| {
        tokens
            .partition_point(|token| token.end <= offset)
            .min(tokens.len().saturating_sub(1))
    };
    let token_indices = pieces
        .iter()
        .map(|(offset, _)| token_of(*offset))
        .collect::<Vec<_>>();
    pieces
        .iter()
        .enumerate()
        .map(|(idx, (offset, piece))| {
            let token = &tokens[token_indices[idx]];
            let start = if idx > 0 && token_indices[idx - 1] == token_indices[idx] {
                *offset
            } else {
                token.start
            };
            let end = match pieces.get(idx + 1) {
                Some((next, _)) if token_indices[idx + 1] == token_indices[idx] => *next,
                _ => token.end,
            };
            AlignmentUnit {
                text: piece.clone(),
                source: start..end,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(text: &str, language: Option<&str>) -> Vec<String> {
        alignment_units(text, language)
            .into_iter()
            .map(|unit| unit.text)
            .collect()
    }

    fn surfaces(text: &str, language: Option<&str>) -> Vec<String> {
        alignment_units(text, language)
            .iter()
            .map(|unit| unit.surface(text).to_string())
            .collect()
    }

    // Goldens: qwen-asr 0.0.6 `Qwen3ForceAlignProcessor.encode_timestamp`
    // (nagisa 0.3.0, soynlp 0.0.493 `LTokenizer` with the shipped
    // dictionary), run on these sentences on 2026-10-15.

    #[test]
    fn english_units_match_the_reference() {
        assert_eq!(
            texts("We’re no longer", Some("English")),
            ["Were", "no", "longer"]
        );
        let test_md = include_str!("../../../../../../../data/test.md");
        let expected =
            "Were no longer just talking to AI were collaborating with it In 2026 the era \
            of simple chatbots is over Weve entered the age of Agentic AIautonomous systems that \
            dont just answer questions but execute complex goals From sovereign data networks to \
            physical robotics AI is becoming the invisible backbone of our world Its not about \
            replacing human potential its about amplifying it to solve the impossible";
        assert_eq!(
            texts(test_md, Some("English")),
            expected.split(' ').collect::<Vec<_>>()
        );
        assert_eq!(
            texts(
                "Hello, world. It’s well-known: e-mail, DON'T, rock'n'roll — café (naïve) “quoted”",
                None
            ),
            [
                "Hello",
                "world",
                "Its",
                "wellknown",
                "email",
                "DON'T",
                "rock'n'roll",
                "café",
                "naïve",
                "quoted"
            ]
        );
    }

    #[test]
    fn chinese_units_match_the_reference() {
        assert_eq!(
            texts(
                "你好，世界。我们正在测试AI模型，版本2.0很好！",
                Some("Chinese")
            ),
            [
                "你", "好", "世", "界", "我", "们", "正", "在", "测", "试", "AI", "模", "型", "版",
                "本", "20", "很", "好"
            ]
        );
        assert_eq!(
            texts("今天天气很好 我们去 Beijing看看", Some("Chinese")),
            [
                "今", "天", "天", "气", "很", "好", "我", "们", "去", "Beijing", "看", "看"
            ]
        );
    }

    #[test]
    fn korean_units_match_the_reference() {
        assert_eq!(
            texts(
                "안녕하세요, 저는 학생입니다. 오늘 날씨가 정말 좋네요! 대한민국의 수도는 서울입니다.",
                Some("Korean")
            ),
            [
                "안녕",
                "하세요",
                "저는",
                "학생",
                "입니다",
                "오늘",
                "날씨",
                "가",
                "정말",
                "좋네요",
                "대한민국",
                "의",
                "수도",
                "는",
                "서울",
                "입니다"
            ]
        );
        assert_eq!(
            texts("학교에서는 AI를 공부합니다", Some("ko")),
            ["학교", "에서는", "AI를", "공부", "합니다"]
        );
    }

    /// The approximation's units, and nagisa's (the reference) for the same
    /// sentences: the approximation mostly keeps nagisa's boundaries but
    /// merges compounds, verb endings and a particle before hiragana
    /// (`はとても`), and splits a few compound verbs (`走り` `回る`).
    #[test]
    fn japanese_units_approximate_nagisa() {
        let cases = [
            (
                "私は東京大学で日本語を勉強しています。",
                vec![
                    "私",
                    "は",
                    "東京大学",
                    "で",
                    "日本語",
                    "を",
                    "勉強し",
                    "ています",
                ],
                // nagisa: 私|は|東京|大学|で|日本|語|を|勉強|し|て|い|ます
            ),
            (
                "今日はとても良い天気ですね！コーヒーを飲みに行きましょう。",
                vec![
                    "今日",
                    "はとても",
                    "良い",
                    "天気",
                    "ですね",
                    "コーヒー",
                    "を",
                    "飲み",
                    "に",
                    "行き",
                    "ましょう",
                ],
                // nagisa: 今日|は|とても|良い|天気|です|ね|コーヒー|を|飲み|に|行き|ましょう
            ),
            (
                "AIの時代、2026年にロボットが走り回る。",
                vec![
                    "AI",
                    "の",
                    "時代",
                    "2",
                    "0",
                    "2",
                    "6",
                    "年",
                    "に",
                    "ロボット",
                    "が",
                    "走り",
                    "回る",
                ],
                // nagisa: AI|の|時代|2|0|2|6|年|に|ロボット|が|走り回る
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(texts(text, Some("Japanese")), expected, "{text}");
            assert_eq!(texts(text, Some("ja-JP")), expected, "{text}");
        }
    }

    #[test]
    fn units_keep_the_source_text_they_stand_for() {
        assert_eq!(
            surfaces("Hello, world. We’re “AI—autonomous”!", None),
            ["Hello,", "world.", "We’re", "“AI—autonomous”!"]
        );
        assert_eq!(
            surfaces("你好，世界。", Some("Chinese")),
            ["你", "好，", "世", "界。"]
        );
        assert_eq!(
            surfaces("안녕하세요, 저는", Some("Korean")),
            ["안녕", "하세요,", "저는"]
        );
        assert_eq!(
            surfaces("私は東京、です。", Some("Japanese")),
            ["私", "は", "東京、", "です。"]
        );
        // A token without letters stays in the gap between units.
        let text = "one — two";
        let units = alignment_units(text, None);
        assert_eq!(units[0].source, 0..3);
        assert_eq!(units[1].source, 8..11);
        assert!(alignment_units(" … — ", None).is_empty());
    }
}

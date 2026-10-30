//! The transcript side of the aligner: cleaning the decoded t-SOT text,
//! splitting it into speaker-tagged words, choosing alignment units
//! (characters for Chinese and Japanese) and the CTC ids of a character.
//!
//! Python semantics the port keeps: `str.split()` whitespace (Unicode
//! `White_Space` plus U+001C..U+001F), the speaker-tag regex
//! `<spk:(\d+)>` with `re.IGNORECASE` (so `ſ` matches `s` and the Kelvin
//! sign matches `k`) and `\d` as any Unicode decimal digit, `int()` of
//! those digits, and `unicodedata.name` prefixes for script detection
//! (tables below, from Python 3.13's Unicode 15.1 database).

use icu_properties::props::GeneralCategory;
use icu_properties::CodePointMapData;
use tokenizers::NormalizedString;

use super::tokenizer::CtcTokenizer;

/// One alignment unit of the transcript, in transcript order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SotWord {
    pub word: String,
    /// The `<spk:N>` tag as `str(int(N))`; `None` before the first tag; `"0"`
    /// throughout a transcript without tags.
    pub speaker_tag: Option<String>,
    /// How many tags precede the word, minus one; `None` before the first.
    pub turn_index: Option<usize>,
    /// A character unit (Chinese or Japanese), tokenized on its own.
    pub character: bool,
}

/// `MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript`: removes
/// every decoder special token except speaker (`<spk:N>`), event
/// (`<ev:…>`) and style (`<sty:…>`) tags, then joins the words with
/// single spaces. `special_tokens` are the decoding tokenizer's (the LLM's
/// `all_special_tokens`), not the CTC tokenizer's.
pub fn clean_sot_transcript(transcript: &str, special_tokens: &[&str]) -> String {
    let mut text = transcript.to_string();
    for token in special_tokens {
        let kept = speaker_tag_at(token, 0).is_some_and(|(end, _)| end == token.len())
            || is_event_or_style_tag(token);
        if !kept {
            text = text.replace(token, " ");
        }
    }
    py_split(&text).collect::<Vec<_>>().join(" ")
}

/// `re.fullmatch(r'<(?:ev|sty):[^<>]*>', token)`.
fn is_event_or_style_tag(token: &str) -> bool {
    let Some(body) = token
        .strip_prefix("<ev:")
        .or_else(|| token.strip_prefix("<sty:"))
    else {
        return false;
    };
    body.strip_suffix('>')
        .is_some_and(|inner| !inner.contains(['<', '>']))
}

/// `parse_sot_words`: em dashes become spaces; each `<spk:N>` starts a
/// turn; words are whitespace-separated.
pub(crate) fn parse_sot_words(transcript: &str) -> Vec<SotWord> {
    let transcript = transcript.replace('\u{2014}', " ");
    let mut parts: Vec<(&str, Option<String>)> = Vec::new();
    let mut part_start = 0;
    let mut tag: Option<String> = None;
    let mut offset = 0;
    while offset < transcript.len() {
        if let Some((end, value)) = speaker_tag_at(&transcript, offset) {
            parts.push((&transcript[part_start..offset], tag.take()));
            tag = Some(value);
            part_start = end;
            offset = end;
        } else {
            offset += transcript[offset..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
        }
    }
    parts.push((&transcript[part_start..], tag));

    let untagged = parts.len() == 1;
    let mut words = Vec::new();
    let mut speaker_tag = untagged.then(|| "0".to_string());
    let mut turn_index = None;
    for (turn, (text, tag)) in parts.into_iter().enumerate() {
        if let Some(tag) = tag {
            speaker_tag = Some(tag);
            turn_index = Some(turn - 1);
        }
        for word in py_split(text) {
            words.push(SotWord {
                word: word.to_string(),
                speaker_tag: speaker_tag.clone(),
                turn_index,
                character: false,
            });
        }
    }
    words
}

/// `<spk:(\d+)>` (case-insensitive) at byte `offset`: the end of the match
/// and `str(int(digits))`.
fn speaker_tag_at(text: &str, offset: usize) -> Option<(usize, String)> {
    let mut chars = text[offset..].char_indices();
    let mut expect = |accepts: &dyn Fn(char) -> bool| chars.next().filter(|&(_, c)| accepts(c));
    expect(&|c| c == '<')?;
    expect(&|c| matches!(c, 's' | 'S' | '\u{17F}'))?;
    expect(&|c| matches!(c, 'p' | 'P'))?;
    expect(&|c| matches!(c, 'k' | 'K' | '\u{212A}'))?;
    expect(&|c| c == ':')?;
    let mut digits = String::new();
    let mut count = 0;
    for (index, c) in chars {
        if let Some(value) = decimal_digit_value(c) {
            // `int()` drops leading zeros.
            if !(digits.is_empty() && value == 0) {
                digits.push(char::from(b'0' + value));
            }
            count += 1;
            continue;
        }
        if c == '>' && count > 0 {
            if digits.is_empty() {
                digits.push('0');
            }
            return Some((offset + index + 1, digits));
        }
        return None;
    }
    None
}

/// The value of a Unicode decimal digit (`Nd`): Unicode encodes each set
/// of decimal digits as a contiguous run from zero to nine.
fn decimal_digit_value(c: char) -> Option<u8> {
    if c.is_ascii_digit() {
        return Some(c as u8 - b'0');
    }
    if general_category(c) != GeneralCategory::DecimalNumber {
        return None;
    }
    let mut run = 0u32;
    let mut code = c as u32;
    while let Some(previous) = code.checked_sub(1).and_then(char::from_u32) {
        if general_category(previous) != GeneralCategory::DecimalNumber {
            break;
        }
        run += 1;
        code -= 1;
    }
    Some((run % 10) as u8)
}

fn general_category(c: char) -> GeneralCategory {
    CodePointMapData::<GeneralCategory>::new().get(c)
}

/// Python's `str.isspace()`, which `str.split()` splits on.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c)
}

/// Python's `str.split()` with no separator.
fn py_split(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_py_space).filter(|word| !word.is_empty())
}

/// `re.fullmatch(r'<[^<>]+>', word)`: an event or style tag (or any
/// bracketed token), kept as one unit.
pub(crate) fn is_bracketed(word: &str) -> bool {
    word.strip_prefix('<')
        .and_then(|rest| rest.strip_suffix('>'))
        .is_some_and(|inner| !inner.is_empty() && !inner.contains(['<', '>']))
}

/// Code-point ranges whose `unicodedata.name` starts with `HANGUL`.
const HANGUL: &[(u32, u32)] = &[
    (0x1100, 0x11FF),
    (0x302E, 0x302F),
    (0x3131, 0x318E),
    (0xA960, 0xA97C),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7C6),
    (0xD7CB, 0xD7FB),
];
/// Names starting with `HIRAGANA` or `KATAKANA`.
const KANA: &[(u32, u32)] = &[
    (0x3041, 0x3096),
    (0x309B, 0x30FF),
    (0x31F0, 0x31FF),
    (0x1AFF0, 0x1AFF3),
    (0x1AFF5, 0x1AFFB),
    (0x1AFFD, 0x1AFFE),
    (0x1B000, 0x1B001),
    (0x1B11F, 0x1B122),
    (0x1B132, 0x1B132),
    (0x1B150, 0x1B152),
    (0x1B155, 0x1B155),
    (0x1B164, 0x1B167),
];
/// Names starting with `CJK UNIFIED IDEOGRAPH` or `CJK COMPATIBILITY IDEOGRAPH`.
const CJK_IDEOGRAPHS: &[(u32, u32)] = &[
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B739),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
];
/// Names starting with `THAI`.
const THAI: &[(u32, u32)] = &[(0x0E01, 0x0E3A), (0x0E3F, 0x0E5B)];

/// The script tables in `_timestamp_units`' detection order.
pub(crate) const SCRIPT_TABLES: [(&str, &[(u32, u32)]); 4] = [
    ("ko", HANGUL),
    ("ja", KANA),
    ("zh", CJK_IDEOGRAPHS),
    ("th", THAI),
];

fn in_table(c: char, table: &[(u32, u32)]) -> bool {
    let code = c as u32;
    table
        .binary_search_by(|&(low, high)| {
            if high < code {
                std::cmp::Ordering::Less
            } else if low > code {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `_timestamp_units`' `auto` detection: the first of ko, ja, zh, th
/// whose script any character of a non-bracketed word belongs to.
pub(crate) fn detect_language(words: &[SotWord]) -> Option<&'static str> {
    let chars: Vec<char> = words
        .iter()
        .filter(|word| !is_bracketed(&word.word))
        .flat_map(|word| word.word.chars())
        .collect();
    SCRIPT_TABLES
        .iter()
        .find(|(_, table)| chars.iter().any(|&c| in_table(c, table)))
        .map(|(language, _)| *language)
}

/// `_timestamp_units`: Chinese and Japanese words become their lexical
/// characters, never across a turn or a bracketed tag; every other
/// language keeps its whitespace words. `language` is the aligner's
/// `timestamp_language`; `None` (what NeMo serves with) detects it.
///
/// Thai deviates: upstream segments it with pythainlp's `newmm`
/// dictionary segmenter, which izwi does not ship, so Thai keeps its
/// whitespace words.
pub(crate) fn timestamp_units(words: Vec<SotWord>, language: Option<&str>) -> Vec<SotWord> {
    let language = match language {
        None | Some("") => detect_language(&words),
        Some(language) => Some(language),
    };
    if !matches!(language, Some("zh" | "ja")) {
        return words;
    }
    let mut output = Vec::new();
    for word in words {
        if is_bracketed(&word.word) {
            output.push(word);
            continue;
        }
        for character in lexical_characters(&word.word) {
            output.push(SotWord {
                word: character.to_string(),
                speaker_tag: word.speaker_tag.clone(),
                turn_index: word.turn_index,
                character: true,
            });
        }
    }
    output
}

/// `_lexical_characters`: the letters, marks and numbers of the NFC form.
pub(crate) fn lexical_characters(text: &str) -> Vec<char> {
    let mut normalized = NormalizedString::from(text);
    normalized.nfc();
    normalized
        .get()
        .chars()
        .filter(|&c| is_lexical(c))
        .collect()
}

/// General category L*, M* or N*.
fn is_lexical(c: char) -> bool {
    use GeneralCategory as G;
    matches!(
        general_category(c),
        G::UppercaseLetter
            | G::LowercaseLetter
            | G::TitlecaseLetter
            | G::ModifierLetter
            | G::OtherLetter
            | G::NonspacingMark
            | G::SpacingMark
            | G::EnclosingMark
            | G::DecimalNumber
            | G::LetterNumber
            | G::OtherNumber
    )
}

/// `_character_token_ids`: the character's own piece, else its encoding
/// without bare `▁` pieces; `None` when that is empty, unknown, not below
/// `blank`, or does not decode back to the character.
pub(crate) fn character_token_ids(
    tokenizer: &CtcTokenizer,
    character: &str,
    blank: u32,
) -> Option<Vec<u32>> {
    let unk = tokenizer.unk_id();
    let own = tokenizer.piece_to_id(character);
    let ids: Vec<u32> = if own != unk {
        vec![own]
    } else {
        tokenizer
            .encode(character)
            .into_iter()
            .filter(|&id| tokenizer.id_to_piece(id) != Some("\u{2581}"))
            .collect()
    };
    let round_trip: String = lexical_characters(&tokenizer.decode(&ids))
        .into_iter()
        .collect();
    let valid = !ids.is_empty()
        && !ids.contains(&unk)
        && ids.iter().all(|&id| id < blank)
        && round_trip == character;
    valid.then_some(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(transcript: &str) -> Vec<(String, Option<String>, Option<usize>)> {
        parse_sot_words(transcript)
            .into_iter()
            .map(|w| (w.word, w.speaker_tag, w.turn_index))
            .collect()
    }

    fn w(
        word: &str,
        tag: Option<&str>,
        turn: Option<usize>,
    ) -> (String, Option<String>, Option<usize>) {
        (word.to_string(), tag.map(str::to_string), turn)
    }

    #[test]
    fn cleaning_keeps_speaker_event_and_style_tags() {
        let specials = [
            "<|im_end|>",
            "<spk:0>",
            "<SPK:12>",
            "<ev:laughter>",
            "<sty:stt:flat>",
            "<ev>",
            "<pad>",
        ];
        let text = "<spk:0> hi<|im_end|>there  <ev:laughter> <pad>\t<ev> x <sty:stt:flat>";
        assert_eq!(
            clean_sot_transcript(text, &specials),
            "<spk:0> hi there <ev:laughter> x <sty:stt:flat>"
        );
        // Python's str.split() also splits on U+001C..U+001F.
        assert_eq!(clean_sot_transcript("a\u{1F}b\u{3000}c", &[]), "a b c");
    }

    #[test]
    fn parsing_follows_speaker_turns() {
        assert_eq!(
            words("pre <spk:1> hello there <spk:0> hi\u{2014}yo <spk:1> again"),
            vec![
                w("pre", None, None),
                w("hello", Some("1"), Some(0)),
                w("there", Some("1"), Some(0)),
                w("hi", Some("0"), Some(1)),
                w("yo", Some("0"), Some(1)),
                w("again", Some("1"), Some(2)),
            ]
        );
        // No tag at all: speaker 0, no turn.
        assert_eq!(
            words(" a  b "),
            vec![w("a", Some("0"), None), w("b", Some("0"), None)]
        );
        // Tags glued to words, case-insensitive, leading zeros, ſ / Kelvin
        // sign, non-ASCII digits; malformed tags stay words.
        assert_eq!(
            words("x<SPK:007>y <\u{17F}p\u{212A}:\u{0663}>z <spk:>q <spk:1 >"),
            vec![
                w("x", None, None),
                w("y", Some("7"), Some(0)),
                w("z", Some("3"), Some(1)),
                w("<spk:>q", Some("3"), Some(1)),
                w("<spk:1", Some("3"), Some(1)),
                w(">", Some("3"), Some(1)),
            ]
        );
        assert!(words("").is_empty());
        assert!(words("<spk:2>").is_empty());
    }

    #[test]
    fn bracketed_words_match_the_upstream_pattern() {
        assert!(is_bracketed("<ev:laughter>"));
        assert!(is_bracketed("<x y>"));
        assert!(!is_bracketed("<>"));
        assert!(!is_bracketed("<a<b>"));
        assert!(!is_bracketed("hi<ev:x>"));
    }

    #[test]
    fn units_split_chinese_and_japanese_into_characters() {
        let parsed = parse_sot_words("<spk:0> 你好， 世界 <ev:laughter> ok <spk:1> 再见");
        assert_eq!(detect_language(&parsed), Some("zh"));
        let units = timestamp_units(parsed, None);
        let flat: Vec<(&str, Option<&str>, bool)> = units
            .iter()
            .map(|u| (u.word.as_str(), u.speaker_tag.as_deref(), u.character))
            .collect();
        assert_eq!(
            flat,
            vec![
                ("你", Some("0"), true),
                ("好", Some("0"), true),
                ("世", Some("0"), true),
                ("界", Some("0"), true),
                ("<ev:laughter>", Some("0"), false),
                ("o", Some("0"), true),
                ("k", Some("0"), true),
                ("再", Some("1"), true),
                ("见", Some("1"), true),
            ]
        );
        // Hangul wins over Han; Korean keeps whitespace words.
        let mixed = parse_sot_words("한국 中文");
        assert_eq!(detect_language(&mixed), Some("ko"));
        assert_eq!(timestamp_units(mixed.clone(), None), mixed);
        // Kana means Japanese; an explicit language overrides detection.
        assert_eq!(
            detect_language(&parse_sot_words("カタカナ 漢字")),
            Some("ja")
        );
        let english = parse_sot_words("ab");
        assert_eq!(timestamp_units(english.clone(), Some("en")), english);
        assert_eq!(timestamp_units(english.clone(), Some("ja")).len(), 2);
        // Thai keeps whitespace words (documented deviation).
        let thai = parse_sot_words("สวัสดี ครับ");
        assert_eq!(detect_language(&thai), Some("th"));
        assert_eq!(timestamp_units(thai.clone(), None), thai);
        // Bracketed words never trigger detection.
        assert_eq!(detect_language(&parse_sot_words("<中> a")), None);
    }

    #[test]
    fn lexical_characters_are_nfc_letters_marks_and_numbers() {
        // e + combining acute composes; punctuation and symbols drop;
        // a compatibility ideograph maps to its unified form.
        assert_eq!(
            lexical_characters("e\u{301}!²\u{F900}▁ ⁇"),
            vec!['é', '²', '\u{8C48}']
        );
    }
}

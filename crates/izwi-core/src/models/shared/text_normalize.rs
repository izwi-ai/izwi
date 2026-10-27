//! English text normalization for speech (`batch.tts.normalize_text`,
//! TPIPE-26): currency, percentages, ordinals, dates, times, years, units
//! and plain numbers become the words a speaker says. Kokoro read
//! "$1,250.75" as "dollar one thousand two hundred fifty point seven five"
//! and "March 3rd" as "March three"; it opts in by default. Families that
//! normalize inside the model stay off until measured
//! ([`family_normalizes_by_default`]); a request can turn it on or off.
//!
//! Only standalone numeric tokens change: digits inside a word ("MP3",
//! "4K", "v1.2") are left alone.

use crate::catalog::ModelFamily;

/// Whether `family` normalizes speech text when the request does not say.
/// Maya 2 Native was evaluated and is served upstream with its own
/// multilingual normaliser on.
pub(crate) fn family_normalizes_by_default(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::KokoroTts | ModelFamily::Maya2Tts)
}

/// Whether text in `language` (or, without one, for `speaker`) is English:
/// an `en*` language tag, or a Kokoro American/British voice (`a*`/`b*`).
pub(crate) fn is_english(language: Option<&str>, speaker: Option<&str>) -> bool {
    match language.map(|language| language.trim().to_ascii_lowercase()) {
        Some(language) if !language.is_empty() => {
            language.starts_with("en") || language == "a" || language == "b"
        }
        _ => speaker.is_none_or(|speaker| {
            let speaker = speaker.trim().to_ascii_lowercase();
            speaker.is_empty() || speaker.starts_with('a') || speaker.starts_with('b')
        }),
    }
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const MAGNITUDES: [&str; 4] = ["thousand", "million", "billion", "trillion"];

/// Units after a number: the abbreviation, its singular and its plural.
const UNITS: &[(&str, &str, &str)] = &[
    ("km/h", "kilometer per hour", "kilometers per hour"),
    ("km", "kilometer", "kilometers"),
    ("cm", "centimeter", "centimeters"),
    ("mm", "millimeter", "millimeters"),
    ("kg", "kilogram", "kilograms"),
    ("mg", "milligram", "milligrams"),
    ("lbs", "pounds", "pounds"),
    ("lb", "pound", "pounds"),
    ("oz", "ounce", "ounces"),
    ("ft", "foot", "feet"),
    ("mph", "mile per hour", "miles per hour"),
    ("kph", "kilometer per hour", "kilometers per hour"),
    ("ml", "milliliter", "milliliters"),
    ("°C", "degree Celsius", "degrees Celsius"),
    ("°F", "degree Fahrenheit", "degrees Fahrenheit"),
    ("°", "degree", "degrees"),
    ("TB", "terabyte", "terabytes"),
    ("GB", "gigabyte", "gigabytes"),
    ("MB", "megabyte", "megabytes"),
    ("KB", "kilobyte", "kilobytes"),
    ("GHz", "gigahertz", "gigahertz"),
    ("MHz", "megahertz", "megahertz"),
    ("kHz", "kilohertz", "kilohertz"),
    ("Hz", "hertz", "hertz"),
    ("ms", "millisecond", "milliseconds"),
];

/// Currencies: the symbol, the unit (singular, plural) and the hundredth.
const CURRENCIES: &[(char, &str, &str, &str, &str)] = &[
    ('$', "dollar", "dollars", "cent", "cents"),
    ('£', "pound", "pounds", "penny", "pence"),
    ('€', "euro", "euros", "cent", "cents"),
];

fn below_thousand(value: u64, out: &mut Vec<String>) {
    let hundreds = value / 100;
    let rest = value % 100;
    if hundreds > 0 {
        out.push(ONES[hundreds as usize].to_string());
        out.push("hundred".to_string());
    }
    if rest >= 20 {
        let tens = TENS[(rest / 10) as usize];
        out.push(match rest % 10 {
            0 => tens.to_string(),
            ones => format!("{tens}-{}", ONES[ones as usize]),
        });
    } else if rest > 0 || hundreds == 0 {
        out.push(ONES[rest as usize].to_string());
    }
}

/// `value` in words: "one thousand two hundred fifty".
pub(crate) fn cardinal(value: u64) -> String {
    if value == 0 {
        return "zero".to_string();
    }
    let mut words = Vec::new();
    let mut rest = value;
    for (scale, name) in SCALES {
        if rest >= scale {
            below_thousand(rest / scale, &mut words);
            words.push(name.to_string());
            rest %= scale;
        }
    }
    if rest > 0 {
        below_thousand(rest, &mut words);
    }
    words.join(" ")
}

/// `value` as an ordinal: "third", "twenty-first", "one hundredth".
pub(crate) fn ordinal(value: u64) -> String {
    let words = cardinal(value);
    let (head, last) = match words.rfind([' ', '-']) {
        Some(index) => words.split_at(index + 1),
        None => ("", words.as_str()),
    };
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        word if word.ends_with('y') => format!("{}ieth", &word[..word.len() - 1]),
        word => format!("{word}th"),
    };
    format!("{head}{last}")
}

/// A four-digit year as spoken: "nineteen ninety-nine", "two thousand
/// five", "twenty twenty-five", "nineteen hundred".
fn year(value: u64) -> String {
    let (high, low) = (value / 100, value % 100);
    if (2000..2010).contains(&value) || value.is_multiple_of(1000) {
        return cardinal(value);
    }
    match low {
        0 => format!("{} hundred", cardinal(high)),
        1..=9 => format!("{} oh {}", cardinal(high), cardinal(low)),
        _ => format!("{} {}", cardinal(high), cardinal(low)),
    }
}

fn digits_spoken(digits: &str) -> String {
    digits
        .chars()
        .filter_map(|ch| ch.to_digit(10))
        .map(|digit| ONES[digit as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// A numeric literal: its integer value, its decimal digits and whether it
/// used thousands separators.
#[derive(Debug, Clone)]
struct Number {
    integer: u64,
    fraction: Option<String>,
    grouped: bool,
    digits: usize,
    /// Characters consumed.
    len: usize,
}

fn parse_number(chars: &[char], start: usize) -> Option<Number> {
    let mut index = start;
    let mut integer: u64 = 0;
    let mut digits = 0usize;
    let mut grouped = false;
    while index < chars.len() {
        let ch = chars[index];
        if let Some(digit) = ch.to_digit(10) {
            integer = integer.checked_mul(10)?.checked_add(u64::from(digit))?;
            digits += 1;
            index += 1;
        } else if ch == ','
            && digits > 0
            && chars
                .get(index + 1..index + 4)
                .is_some_and(|group| group.iter().all(char::is_ascii_digit))
            && chars.get(index + 4).is_none_or(|c| !c.is_ascii_digit())
        {
            grouped = true;
            index += 1;
        } else {
            break;
        }
    }
    if digits == 0 {
        return None;
    }
    let mut fraction = None;
    if chars.get(index) == Some(&'.') && chars.get(index + 1).is_some_and(char::is_ascii_digit) {
        let mut end = index + 1;
        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }
        fraction = Some(chars[index + 1..end].iter().collect());
        index = end;
    }
    Some(Number {
        integer,
        fraction,
        grouped,
        digits,
        len: index - start,
    })
}

fn number_words(number: &Number) -> String {
    let mut words = cardinal(number.integer);
    if let Some(fraction) = &number.fraction {
        words.push_str(" point ");
        words.push_str(&digits_spoken(fraction));
    }
    words
}

fn matches_at(chars: &[char], index: usize, word: &str) -> bool {
    let word: Vec<char> = word.chars().collect();
    chars.len() >= index + word.len() && chars[index..index + word.len()] == word[..]
}

fn word_boundary(chars: &[char], index: usize) -> bool {
    chars.get(index).is_none_or(|ch| !ch.is_alphanumeric())
}

/// A following word (after one space) matching one of `words`,
/// case-insensitively: (the word, characters consumed).
fn following_word<'a>(chars: &[char], index: usize, words: &[&'a str]) -> Option<(&'a str, usize)> {
    if chars.get(index) != Some(&' ') {
        return None;
    }
    let start = index + 1;
    let mut end = start;
    while end < chars.len() && chars[end].is_alphabetic() {
        end += 1;
    }
    let word: String = chars[start..end].iter().collect::<String>().to_lowercase();
    words
        .iter()
        .find(|candidate| **candidate == word)
        .map(|candidate| (*candidate, end - index))
}

/// The meridiem after a time: " am", "pm", " a.m.", ...: (spoken, consumed).
fn meridiem(chars: &[char], index: usize) -> Option<(&'static str, usize)> {
    let space = usize::from(chars.get(index) == Some(&' '));
    let at = index + space;
    for (form, spoken) in [
        ("a.m.", "a m"),
        ("p.m.", "p m"),
        ("am", "a m"),
        ("pm", "p m"),
        ("AM", "a m"),
        ("PM", "p m"),
        ("A.M.", "a m"),
        ("P.M.", "p m"),
    ] {
        let len = form.chars().count();
        let boundary = if form.ends_with('.') {
            true
        } else {
            word_boundary(chars, at + len)
        };
        if matches_at(chars, at, form) && boundary {
            return Some((spoken, space + len));
        }
    }
    None
}

fn previous_word_is_month(out: &str) -> bool {
    let word: String = out
        .trim_end()
        .rsplit(|ch: char| !ch.is_alphabetic())
        .next()
        .unwrap_or("")
        .to_lowercase();
    out.ends_with(' ') && MONTHS.contains(&word.as_str())
}

/// Normalize English `text` for speech.
pub(crate) fn normalize_english(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + text.len() / 2);
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        // A numeric token starts after whitespace or an opening mark, never
        // inside a word or a dotted/version token ("MP3", "v1.2").
        let at_boundary = index == 0
            || chars[index - 1].is_whitespace()
            || matches!(
                chars[index - 1],
                '(' | '[' | '{' | '"' | '\'' | '\u{201c}' | '\u{2018}' | '\u{2014}' | '\u{2013}'
            );
        // Currency: "$1,250.75", "$5 million", "£3.50".
        if at_boundary {
            if let Some((_, one, many, hundredth_one, hundredth_many)) =
                CURRENCIES.iter().find(|currency| currency.0 == ch)
            {
                if let Some(number) = parse_number(&chars, index + 1) {
                    let end = index + 1 + number.len;
                    if word_boundary(&chars, end) {
                        if let Some((magnitude, consumed)) =
                            following_word(&chars, end, &MAGNITUDES)
                        {
                            out.push_str(&format!("{} {magnitude} {many}", number_words(&number)));
                            index = end + consumed;
                            continue;
                        }
                        let cents = number.fraction.as_deref().and_then(|fraction| {
                            (fraction.len() == 2)
                                .then(|| fraction.parse::<u64>().ok())
                                .flatten()
                        });
                        let whole = number.integer;
                        let mut spoken = Vec::new();
                        if whole > 0 || cents.is_none_or(|cents| cents == 0) {
                            spoken.push(format!(
                                "{} {}",
                                cardinal(whole),
                                if whole == 1 { one } else { many }
                            ));
                        }
                        match (cents, &number.fraction) {
                            (Some(0), _) => {}
                            (Some(cents), _) => spoken.push(format!(
                                "{} {}",
                                cardinal(cents),
                                if cents == 1 {
                                    hundredth_one
                                } else {
                                    hundredth_many
                                }
                            )),
                            (None, Some(_)) => {
                                spoken = vec![format!("{} {many}", number_words(&number))]
                            }
                            (None, None) => {}
                        }
                        out.push_str(&spoken.join(" and "));
                        index = end;
                        continue;
                    }
                }
            }
        }
        let negative = ch == '-'
            && at_boundary
            && (index == 0 || chars[index - 1].is_whitespace())
            && chars.get(index + 1).is_some_and(char::is_ascii_digit);
        let start = if negative { index + 1 } else { index };
        if (at_boundary || negative) && chars.get(start).is_some_and(char::is_ascii_digit) {
            if let Some(number) = parse_number(&chars, start) {
                let end = start + number.len;
                if let Some((spoken, consumed)) = spoken_number(&chars, end, &number, &out) {
                    if negative {
                        out.push_str("minus ");
                    }
                    out.push_str(&spoken);
                    index = end + consumed;
                    continue;
                }
                // Digits run into letters ("4K"): keep the token.
                let mut token_end = end;
                while token_end < chars.len() && chars[token_end].is_alphanumeric() {
                    token_end += 1;
                }
                out.extend(&chars[index..token_end]);
                index = token_end;
                continue;
            }
        }
        out.push(ch);
        index += 1;
    }
    out
}

/// The words for the number ending at `end`, with whatever suffix it takes
/// (a time, an ordinal suffix, a percent sign, a unit), or `None` when the
/// number runs into a word.
fn spoken_number(
    chars: &[char],
    end: usize,
    number: &Number,
    out: &str,
) -> Option<(String, usize)> {
    let plain = !number.grouped && number.fraction.is_none();
    // Times: "10:30", "9:05 pm", "12:00".
    if plain
        && number.integer <= 24
        && chars.get(end) == Some(&':')
        && chars.get(end + 1).is_some_and(char::is_ascii_digit)
        && chars.get(end + 2).is_some_and(char::is_ascii_digit)
        && word_boundary(chars, end + 3)
    {
        let minutes: u64 = chars[end + 1..end + 3]
            .iter()
            .collect::<String>()
            .parse()
            .ok()?;
        if minutes < 60 {
            let hours = cardinal(number.integer);
            let mut spoken = match minutes {
                0 => format!("{hours} o'clock"),
                1..=9 => format!("{hours} oh {}", cardinal(minutes)),
                _ => format!("{hours} {}", cardinal(minutes)),
            };
            let mut consumed = 3;
            if let Some((meridiem, length)) = meridiem(chars, end + 3) {
                if minutes == 0 {
                    spoken = cardinal(number.integer);
                }
                spoken.push(' ');
                spoken.push_str(meridiem);
                consumed += length;
            }
            return Some((spoken, consumed));
        }
    }
    // Ordinals: "3rd", "21st", "100th".
    if plain {
        for suffix in ["st", "nd", "rd", "th", "ST", "ND", "RD", "TH"] {
            if matches_at(chars, end, suffix) && word_boundary(chars, end + 2) {
                return Some((ordinal(number.integer), 2));
            }
        }
    }
    if chars.get(end) == Some(&'%') {
        return Some((format!("{} percent", number_words(number)), 1));
    }
    // Units, directly attached or after one space.
    for (abbreviation, one, many) in UNITS {
        let length = abbreviation.chars().count();
        for gap in [0usize, 1] {
            if gap == 1 && chars.get(end) != Some(&' ') {
                continue;
            }
            let at = end + gap;
            if matches_at(chars, at, abbreviation) && word_boundary(chars, at + length) {
                let singular = number.integer == 1 && number.fraction.is_none();
                return Some((
                    format!(
                        "{} {}",
                        number_words(number),
                        if singular { one } else { many }
                    ),
                    gap + length,
                ));
            }
        }
    }
    if chars.get(end).is_some_and(|ch| ch.is_alphanumeric()) {
        return None;
    }
    // A day after a month name: "March 3" is "March third".
    if plain && (1..=31).contains(&number.integer) && previous_word_is_month(out) {
        return Some((ordinal(number.integer), 0));
    }
    if plain && number.digits == 4 && (1100..=2099).contains(&number.integer) {
        return Some((year(number.integer), 0));
    }
    Some((number_words(number), 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers fixture: each input with the words a speaker says. It
    /// is the yardstick for opting a family in (measure the family's own
    /// reading of the inputs first).
    #[test]
    fn numbers_fixture_reads_as_spoken_english() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/tts_text_normalization.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let expected = case["expected"].as_str().unwrap();
            assert_eq!(normalize_english(input), expected, "{input}");
        }
    }

    #[test]
    fn cardinals_ordinals_and_years() {
        assert_eq!(cardinal(1_250), "one thousand two hundred fifty");
        assert_eq!(cardinal(1_000_005), "one million five");
        assert_eq!(ordinal(3), "third");
        assert_eq!(ordinal(21), "twenty-first");
        assert_eq!(ordinal(40), "fortieth");
        assert_eq!(ordinal(112), "one hundred twelfth");
        assert_eq!(year(1999), "nineteen ninety-nine");
        assert_eq!(year(2005), "two thousand five");
        assert_eq!(year(2025), "twenty twenty-five");
        assert_eq!(year(1905), "nineteen oh five");
        assert_eq!(year(1900), "nineteen hundred");
    }

    #[test]
    fn kokoro_and_maya2_opt_in_by_default() {
        assert!(family_normalizes_by_default(ModelFamily::KokoroTts));
        assert!(family_normalizes_by_default(ModelFamily::Maya2Tts));
        for family in [
            ModelFamily::Qwen3Tts,
            ModelFamily::FishS2Tts,
            ModelFamily::VibeVoiceTts,
            ModelFamily::VoxtralTts,
        ] {
            assert!(!family_normalizes_by_default(family), "{family:?}");
        }
        assert!(is_english(None, Some("af_heart")));
        assert!(is_english(None, Some("bm_george")));
        assert!(!is_english(None, Some("jf_alpha")));
        assert!(is_english(Some("en-US"), Some("jf_alpha")));
        assert!(!is_english(Some("fr"), None));
    }
}

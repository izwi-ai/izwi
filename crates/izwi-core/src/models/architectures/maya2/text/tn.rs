//! The vendored normaliser, `maya3/text/tn/normalize.py` (`validate_text`,
//! `Normalizer`, `normalize`), with `lexicon.py`'s L3 `_m_url` in place of
//! the vendored one (it is always installed when `rules=True`).
//!
//! `normalize(text, lang)`: validate (a rejected span is re-read character by
//! character and the text tried again) → protect e-mails / URLs → pre-clean →
//! run the 40 matchers in priority order and keep the earliest-priority,
//! leftmost, longest non-overlapping spans → glue native postpositions onto
//! the rendered spans → sweep the leftover symbols → read any digit left →
//! tidy spacing and punctuation.
//!
//! Default behaviour only: `FR3_TN_URL_SPELL`, `FR3_TN_LONG_DIGITS` and
//! `FR3_TN_STRICT` are off, `date_order` is DMY and `two_digit_year_base` is
//! unset (so a two-digit year that reaches the date matcher is the Python
//! `TypeError`, a [`Fail::Crash`]).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::lang::{fmt, language_table, resolve_language, Lang};
use super::pyre::{
    decimal_value, escape, is_alnum, is_alpha, is_control_or_private, is_decimal, is_digit,
    is_space, re, re_i, Match, Regex,
};
use super::tables as t;
use super::{chars, py_int, py_int_char, py_split, py_strip, sl, st, Fail, R};

/// A replacement: `(start, end, spoken)` in the matched text.
type Span = (usize, usize, String);

const PAUSE: &str = r"[.!?;\x{0964}\x{0965}\n]";
const INDIC_TOKEN: &str = r"[\w#\x{0900}-\x{0DFF}']+";

fn norm_err(code: &'static str, span: Option<(usize, usize)>) -> Fail {
    Fail::Norm {
        code,
        span,
        source: None,
    }
}

/// `_ascii_digits`: every Unicode decimal digit → ASCII.
pub(super) fn ascii_digits(text: &[char]) -> Vec<char> {
    text.iter()
        .map(|&c| match decimal_value(c) {
            Some(d) if !c.is_ascii_digit() => char::from_digit(d, 10).unwrap(),
            _ => c,
        })
        .collect()
}

/// `alt(words)` of `_build_regexes`: longest first, escaped; never-matching
/// when empty.
fn alt(words: &[String]) -> String {
    let mut uniq: Vec<&String> = Vec::new();
    for w in words {
        if !uniq.contains(&w) {
            uniq.push(w);
        }
    }
    uniq.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    if uniq.is_empty() {
        return "(?!x)x".into();
    }
    uniq.iter().map(|w| escape(w)).collect::<Vec<_>>().join("|")
}

/// `"|".join(sorted(map(re.escape, words), key=len, reverse=True))`.
pub(super) fn alt_escaped<S: AsRef<str>>(words: &[S]) -> String {
    let mut esc: Vec<String> = words.iter().map(|w| escape(w.as_ref())).collect();
    esc.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    esc.join("|")
}

fn month_lookup(lang: &Lang) -> HashMap<String, u32> {
    let mut map: HashMap<String, u32> = t::ENGLISH_MONTHS
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    for (i, name) in lang.months.iter().enumerate() {
        map.insert(name.to_lowercase(), i as u32 + 1);
    }
    for (name, i) in &lang.month_aliases {
        map.insert(name.to_lowercase(), *i);
    }
    map
}

/// `date(year, month, day)` validity.
fn calendar_ok(day: u128, month: u128, year: u128) -> bool {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) || day < 1 {
        return false;
    }
    let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let dim = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    day <= dim
}

fn calendar_check(day: u128, month: u128, year: u128, span: Option<(usize, usize)>) -> R<()> {
    if calendar_ok(day, month, year) {
        Ok(())
    } else {
        Err(norm_err("invalid_date", span))
    }
}

/// `_sentence_before(text, start)`.
fn sentence_before(text: &[char], start: usize) -> Vec<char> {
    let piece = &text[start.saturating_sub(100)..start];
    let parts = re!(PAUSE).split(piece);
    chars(parts.last().map_or("", String::as_str))
}

/// `_sentence_after(text, end)`.
fn sentence_after(text: &[char], end: usize) -> Vec<char> {
    let piece = sl(text, end, end + 100);
    let parts = re!(PAUSE).split(&piece);
    chars(&parts[0])
}

/// `_valid_grouping(number)`.
fn valid_grouping(number: &str) -> bool {
    let trimmed = number.trim_start_matches(['+', '-', '−']);
    let integer = trimmed.split('.').next().unwrap_or("");
    !integer.contains(',')
        || re!(r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}")
            .fullmatch(&chars(integer))
            .is_some()
}

fn lower(text: &[char]) -> Vec<char> {
    chars(&st(text).to_lowercase())
}

/// The per-language compiled patterns and keyword sets (`_build_regexes`).
pub(super) struct Normalizer {
    lang: &'static Lang,
    month_lookup: HashMap<String, u32>,
    id_keywords: HashSet<String>,
    label_words: HashSet<String>,
    ratio_kws: HashSet<String>,
    roman_ctx: HashSet<String>,
    year_words: HashSet<String>,
    time_pat: Regex,
    time_range: Regex,
    time_verbalized: Regex,
    date_range: Regex,
    date_day_month: Regex,
    date_month_day: Regex,
    currency_range: Regex,
    currency: Regex,
    amount_units: Regex,
    ordinal: Regex,
    label: Regex,
    unit_initialism: Regex,
    abbrev: Vec<(Regex, String)>,
    v_month_day: Regex,
    v_day_month: Regex,
    v_months: HashMap<String, u32>,
    v_tamil: Regex,
    v_rupee_paise: Regex,
    v_currency_percent: Regex,
}

/// `_engine(language, "DMY", None)`.
pub(super) fn engine(code: &str) -> Option<&'static Normalizer> {
    static ENGINES: OnceLock<Vec<Normalizer>> = OnceLock::new();
    let engines = ENGINES.get_or_init(|| {
        super::lang::CODES
            .iter()
            .map(|c| Normalizer::new(language_table(c).expect("table")))
            .collect()
    });
    engines.iter().find(|n| n.lang.code == code)
}

impl Normalizer {
    fn new(lang: &'static Lang) -> Normalizer {
        let l = lang;
        let period_re = alt(&l.period_detect.keys().map(String::from).collect::<Vec<_>>());
        let marker_re = alt(&l.time_markers);
        let month_lookup = month_lookup(l);
        let month_re = alt(&month_lookup.keys().cloned().collect::<Vec<_>>());
        let mut scales: Vec<String> = l.scale_words.keys().map(String::from).collect();
        for w in [
            "crore", "crores", "Crore", "Crores", "Cr", "cr", "lakh", "lakhs", "Lakh", "Lakhs",
            "thousand",
        ] {
            scales.push(w.into());
        }
        let mut en_scales: Vec<&str> = t::EN_SCALES.to_vec();
        en_scales.sort();
        scales.extend(en_scales.iter().map(|s| s.to_string()));
        let scale_re = alt(&scales);
        let mut ord: Vec<String> = l.ordinal_input_suffixes.clone();
        ord.extend(["st", "nd", "rd", "th"].map(String::from));
        let ord_suffix_re = alt(&ord);
        let native_currency: Vec<String> = l
            .currency_words
            .iter()
            .filter(|(w, k)| !w.is_ascii() && matches!(*k, "rupee" | "dollar" | "euro" | "pound"))
            .map(|(w, _)| w.to_string())
            .collect();
        let mut currency_re = r"₹|\$|€|£|Rs\.?|INR|USD|EUR|GBP".to_string();
        if !native_currency.is_empty() {
            currency_re += &format!("|(?:{})\\.?", alt(&native_currency));
        }
        let id_keywords: HashSet<String> = t::ID_KEYWORDS_EN
            .iter()
            .chain(t::OTP_INPUT_WORDS.iter())
            .map(|s| s.to_string())
            .chain(l.id_keywords.iter().map(|w| w.to_lowercase()))
            .collect();
        let label_words: HashSet<String> = t::LABEL_WORDS_EN
            .iter()
            .map(|s| s.to_string())
            .chain(l.label_words.iter().map(|w| w.to_lowercase()))
            .collect();
        let mut minutes = l.minute_words.clone();
        minutes.extend(["minute", "minutes", "min"].map(String::from));
        let minute_re = alt(&minutes);
        let tpl = re!(r"\{[ab]\}").sub(&chars(&l.ratio), " ");
        let mut ratio_kws: HashSet<String> = ["ratio", "odds", "proportion", "अनुपात"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        ratio_kws.extend(py_split(&st(&tpl)).into_iter().map(|w| w.to_lowercase()));
        let roman_ctx: HashSet<String> = label_words
            .iter()
            .cloned()
            .chain(t::ROMAN_CONTEXT.iter().map(|s| s.to_string()))
            .collect();
        let year_words: HashSet<String> = l
            .year_words
            .iter()
            .map(|w| w.to_lowercase())
            .chain(t::YEAR_CONTEXT.iter().map(|s| s.to_string()))
            .collect();

        let (p, mk) = (&period_re, &marker_re);
        let time_src = format!(
            r"(?:(?<!\w)({p})\s+)?(\d{{1,2}})(?=:\d{{2}}|\.\d{{2}}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])|\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)(?![A-Za-z])|\s*(?:{mk})(?!\w))(?:[.:](\d{{2}}))?(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?(?:\s*({mk})(?!\w))?(?:\s+({p})(?!\w))?"
        );
        let time_pat = Regex::new_i(&time_src);
        let time_range = Regex::new_i(&format!(r"({time_src})\s*[–\-]\s*({time_src})"));
        let time_verbalized = Regex::new(&format!(
            r"\b(\d{{1,2}})\s*(?:{marker_re})\s+(\d{{1,2}})\s*({minute_re})(?!\w)"
        ));
        let date_range = Regex::new_i(&format!(
            r"\b(\d{{1,2}})[-–](\d{{1,2}})\s+({month_re})(?!\w)(?:\s+(\d{{4}}))?"
        ));
        let date_day_month = Regex::new_i(&format!(
            r"\b(\d{{1,2}})(?:-?({ord_suffix_re}))?[\-\s]({month_re})(?!\w)(?:[\-\s,]+(\d{{4}}))?"
        ));
        let date_month_day = Regex::new_i(&format!(
            r"\b({month_re})\s+(\d{{1,2}})(?:-?({ord_suffix_re}))?(?:,?\s+(\d{{4}}))?(?!\d)"
        ));
        let currency_range = Regex::new_i(&format!(
            r"(?<!\w)([+-]?)({currency_re})\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)\s*[–\-]\s*([+-]?\d+(?:,\d{{2,3}})*(?:\.\d+)?)(?!\d|[.,]\d)"
        ));
        let currency = Regex::new_i(&format!(
            r"(?<!\d)([+-]?)\s*(₹|\$|€|£|(?<!\w)(?:{currency_re}))\s*([+-]?)(\d+(?:,\d{{2,3}})*(?:\.\d+)?)(?:\s*({scale_re}{}",
            t::CURRENCY_TAIL
        ));
        let units = format!(
            "{}|{}",
            t::AMOUNT_UNITS,
            l.currency_words
                .keys()
                .map(escape)
                .collect::<Vec<_>>()
                .join("|")
        );
        let amount_units = Regex::new_i(&format!(
            r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:{units})(?![A-Za-z]))"
        ));
        let ordinal = Regex::new_i(&format!(r"\b(\d+)-?({ord_suffix_re})(?![A-Za-z0-9])"));
        let mut labels: Vec<&String> = label_words.iter().collect();
        labels.sort();
        labels.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
        let lbl = labels
            .iter()
            .map(|w| escape(w))
            .collect::<Vec<_>>()
            .join("|");
        let label = Regex::new_i(&format!(
            r"\b({lbl})\s+(?:(\d+)(?![\d.]|,\d)(?!-[A-Za-z]{{2,}})(?:[-\s]?([A-Za-z])\b)?|([A-Za-z])\b(?:[-\s]?(\d+))?)"
        ));
        let mut unit_list: Vec<&str> = t::UNIT_INITIALISMS.to_vec();
        unit_list.sort_by_key(|w| std::cmp::Reverse(w.len()));
        let unit_initialism = Regex::new_i(&format!(
            r"(?<![\w.])(-?)\b(\d+(?:,\d{{2,3}})*(?:\.\d+)?)[\s\-]?({})(?:\s?/\s?([A-Za-z]{{1,4}}))?\b",
            unit_list.join("|")
        ));
        let abbrev = l
            .abbrev_units
            .iter()
            .map(|(ab, word)| {
                (
                    Regex::new_i(&format!(r"(?<![\w-]){ab}(?:\.(?=\s+\S))?(?![\w-])")),
                    word.to_string(),
                )
            })
            .collect();

        // validate_text's language-dependent patterns
        let mut v_months: HashMap<String, u32> = t::TAMIL_MONTHS
            .iter()
            .enumerate()
            .map(|(i, m)| (m.to_string(), i as u32 + 1))
            .collect();
        v_months.extend(month_lookup.iter().map(|(k, v)| (k.clone(), *v)));
        let v_month_re = alt_escaped(&v_months.keys().cloned().collect::<Vec<_>>());
        let mut v_ord: Vec<String> = Vec::new();
        for s in l.ordinal_input_suffixes.iter().map(String::as_str).chain([
            "st",
            "nd",
            "rd",
            "th",
            "ஆம்",
            "ம்",
        ]) {
            if !v_ord.iter().any(|x| x == s) {
                v_ord.push(s.to_string());
            }
        }
        let v_ord_re = alt_escaped(&v_ord);
        let v_day_month = Regex::new_i(&format!(
            r"(?<!\w)([0-9]{{1,2}})(?:-?(?:{v_ord_re}))?[ -]+({v_month_re})(?!\w)(?:[ ,\-]+([0-9]{{4}}))?"
        ));
        let v_month_day = Regex::new_i(&format!(
            r"(?<!\w)({v_month_re})[ -]+([0-9]{{1,2}})(?:-?(?:{v_ord_re}))?(?!\d)(?:,?\s+([0-9]{{4}}))?"
        ));
        let v_tamil = Regex::new(&format!(
            r"({})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?:\s*தேதி)?(?:\s+(\d{{4}}))?",
            t::TAMIL_MONTHS.join("|")
        ));
        let mut rupee_words: Vec<String> = l
            .currency_words
            .iter()
            .filter(|(_, k)| *k == "rupee")
            .map(|(w, _)| w.to_string())
            .collect();
        rupee_words.extend(["INR".to_string(), "Rs".to_string()]);
        let rupee_prefix = format!(r"(?:₹|(?<!\w)(?:{})\.?)", alt_escaped(&rupee_words));
        let mut paise: Vec<&str> = t::PAISA_INPUT_WORDS.to_vec();
        paise.sort();
        let minor_units = alt_escaped(&paise);
        let v_rupee_paise = Regex::new_i(&format!(
            r"{rupee_prefix}\s*[+-]?[0-9][0-9,.]*\s*(?:{minor_units})(?=$|[\s.,!?;\x{{0964}}\x{{0965}}-])"
        ));
        let v_currency_percent = Regex::new_i(&format!(
            r"(?:{rupee_prefix}|[$€£]|\b(?:USD|EUR|GBP))\s*[+-]?[0-9][0-9,.]*\s*%"
        ));

        Normalizer {
            lang,
            month_lookup,
            id_keywords,
            label_words,
            ratio_kws,
            roman_ctx,
            year_words,
            time_pat,
            time_range,
            time_verbalized,
            date_range,
            date_day_month,
            date_month_day,
            currency_range,
            currency,
            amount_units,
            ordinal,
            label,
            unit_initialism,
            abbrev,
            v_month_day,
            v_day_month,
            v_months,
            v_tamil,
            v_rupee_paise,
            v_currency_percent,
        }
    }

    // ------------------------------------------------------------------
    // validate_text
    // ------------------------------------------------------------------

    fn validate(&self, text: &[char]) -> R<()> {
        if text.len() > 20000 {
            return Err(norm_err("text_too_long", None));
        }
        let mut tx = ascii_digits(text);
        if let Some(m) = re!(r"<\s*(?:/?[A-Za-z_]|[!?])|[{}]|\$\{").search(&tx) {
            return Err(norm_err(
                "markup_or_placeholder",
                Some((m.start(), m.end())),
            ));
        }
        for (i, &c) in tx.iter().enumerate() {
            if is_control_or_private(c) && !(matches!(c, '\n' | '\r' | '\t')) {
                return Err(norm_err("control_character", Some((i, i + 1))));
            }
        }
        tx = re!(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}")
            .sub_with(&tx, |m| " ".repeat(m.end() - m.start()));
        let tx = tx;
        if let Some(m) = re!(r"[0-9]{31,}").search(&tx) {
            return Err(norm_err("number_too_long", Some((m.start(), m.end()))));
        }
        if let Some(m) = re_i!(r"\d{4}-\d{2}-\d{2}T\d").search(&tx) {
            return Err(norm_err(
                "unsupported_timestamp",
                Some((m.start(), m.end())),
            ));
        }
        for m in re!(r"(?<![\w.])[+-]?[0-9][0-9,]*(?:\.[0-9]+)?").find_iter(&tx) {
            let s = m.str(&tx, 0);
            if s.chars().filter(|c| c.is_ascii_digit()).count() > 30 {
                return Err(norm_err("number_too_long", Some((m.start(), m.end()))));
            }
            if !valid_grouping(s.trim_end_matches(',')) {
                return Err(norm_err("invalid_grouping", Some((m.start(), m.end()))));
            }
        }
        for m in re!(r"(?<![\w\d])([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?!\d)").find_iter(&tx) {
            let g = |i| py_int(&m.str(&tx, i));
            calendar_check(g(3)?, g(2)?, g(1)?, Some((m.start(), m.end())))?;
        }
        for m in
            re!(r"(?<![\w\d-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2,4})(?!\d)").find_iter(&tx)
        {
            let y4 = m.str(&tx, 4);
            if y4.chars().count() != 4 {
                // len 2 without a configured century, or len 3
                return Err(norm_err("ambiguous_year", Some((m.start(), m.end()))));
            }
            let (a, b) = (py_int(&m.str(&tx, 1))?, py_int(&m.str(&tx, 3))?);
            calendar_check(a, b, py_int(&y4)?, Some((m.start(), m.end())))?;
        }
        for (pat, reverse) in [(&self.v_day_month, false), (&self.v_month_day, true)] {
            for m in pat.find_iter(&tx) {
                let (day, month) = if reverse {
                    (m.str(&tx, 2), m.str(&tx, 1))
                } else {
                    (m.str(&tx, 1), m.str(&tx, 2))
                };
                let mo = *self
                    .v_months
                    .get(&month.to_lowercase())
                    .ok_or(Fail::Crash("KeyError: months"))?;
                let year = match m.get(&tx, 3) {
                    Some(y) if !y.is_empty() => py_int(&y)?,
                    _ => 2000,
                };
                calendar_check(py_int(&day)?, mo as u128, year, Some((m.start(), m.end())))?;
            }
        }
        for m in self.v_tamil.find_iter(&tx) {
            let mo = *self
                .v_months
                .get(&m.str(&tx, 1))
                .ok_or(Fail::Crash("KeyError: months"))?;
            let year = match m.get(&tx, 3) {
                Some(y) if !y.is_empty() => py_int(&y)?,
                _ => 2000,
            };
            calendar_check(
                py_int(&m.str(&tx, 2))?,
                mo as u128,
                year,
                Some((m.start(), m.end())),
            )?;
        }
        if let Some(m) =
            re_i!(r"\b(?:on|by|before|after|date|due)\s+[0-9]{1,2}/[0-9]{1,2}(?![/\d])").search(&tx)
        {
            return Err(norm_err("ambiguous_short_date", Some((m.start(), m.end()))));
        }
        if let Some(m) = re!(r"தேதி\s+[0-9]{1,2}\.\s*[0-9]{4}").search(&tx) {
            return Err(norm_err("conflicting_date", Some((m.start(), m.end()))));
        }
        if let Some(m) = self.v_rupee_paise.search(&tx) {
            return Err(norm_err(
                "conflicting_currency_unit",
                Some((m.start(), m.end())),
            ));
        }
        if let Some(m) = self.v_currency_percent.search(&tx) {
            return Err(norm_err(
                "conflicting_currency_unit",
                Some((m.start(), m.end())),
            ));
        }
        for m in re!(r"(?<![\w:])([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?![\d:])").find_iter(&tx) {
            if py_int(&m.str(&tx, 2))? > 59 || py_int(&m.str(&tx, 3))? > 59 {
                return Err(norm_err("invalid_duration", Some((m.start(), m.end()))));
            }
        }
        let no_durations =
            re!(r"[0-9]+:[0-9]{2}:[0-9]{2}").sub_with(&tx, |m| " ".repeat(m.end() - m.start()));
        let clock = re_i!(
            r"(?<![\w:])([0-9]{1,2})(?:[:.]([0-9]{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)|(?<![\w:])([0-9]{1,2}):([0-9]{2})(?![\d:])"
        );
        for m in clock.find_iter(&no_durations) {
            let valid = if m.span(1).is_some() {
                let h = py_int(&m.str(&no_durations, 1))?;
                let mm = match m.get(&no_durations, 2) {
                    Some(s) => py_int(&s)?,
                    None => 0,
                };
                (1..=12).contains(&h) && mm < 60
            } else {
                let before = sentence_before(&tx, m.start());
                if re_i!(r"\b(?:ratio|odds|proportion)\b")
                    .search(&before)
                    .is_some()
                {
                    continue;
                }
                py_int(&m.str(&no_durations, 4))? < 24 && py_int(&m.str(&no_durations, 5))? < 60
            };
            if !valid {
                return Err(norm_err("invalid_time", Some((m.start(), m.end()))));
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------

    fn month_index(&self, token: &str) -> Option<u32> {
        self.month_lookup
            .get(&py_strip(token).to_lowercase())
            .copied()
    }

    fn digits_spelled(&self, digits: &str, comma: bool) -> R<String> {
        let mut words = Vec::new();
        for c in digits.chars() {
            words.push(self.lang.digit(py_int_char(c)?).to_string());
        }
        Ok(words.join(if comma { ", " } else { " " }))
    }

    fn letters_spelled(&self, letters: &str) -> String {
        letters
            .chars()
            .map(|c| self.lang.letter(c))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn spell_alnum(&self, token: &str, comma: bool) -> R<String> {
        let mut parts = Vec::new();
        for ch in token.chars() {
            if is_digit(ch) {
                parts.push(self.lang.digit(py_int_char(ch)?).to_string());
            } else if is_alpha(ch) && ch.is_ascii() {
                parts.push(self.lang.letter(ch));
            }
        }
        Ok(parts.join(if comma { ", " } else { " " }))
    }

    fn spell_model_segment(&self, seg: &str) -> R<String> {
        let mut out = Vec::new();
        for run in re!(r"[A-Za-z]+|\d+").findall(&chars(seg)) {
            if py_isdigit(&run) {
                let n = py_int(&run)?;
                if run.chars().count() == 1 || (n % 100 == 0 && n >= 100) {
                    out.push(self.lang.cardinal(n));
                } else {
                    out.push(self.digits_spelled(&run, false)?);
                }
            } else {
                out.push(self.letters_spelled(&run));
            }
        }
        Ok(out.join(" "))
    }

    /// `_number_words`: a plain number string (commas allowed).
    fn number_words(&self, numstr: &str) -> R<String> {
        if let Some(rest) = numstr.strip_prefix('-') {
            return Ok(format!(
                "{} {}",
                self.lang.sym("minus"),
                self.number_words(rest)?
            ));
        }
        if let Some(rest) = numstr.strip_prefix('+') {
            return Ok(format!(
                "{} {}",
                self.lang.sym("plus"),
                self.number_words(rest)?
            ));
        }
        let numstr = numstr.replace(',', "");
        if let Some((ip, fp)) = numstr.split_once('.') {
            let ip = if ip.is_empty() { 0 } else { py_int(ip)? };
            return self.lang.decimal(ip, fp);
        }
        Ok(self.lang.cardinal(py_int(&numstr)?))
    }

    fn keyword_before(
        &self,
        text: &[char],
        start: usize,
        keywords: &HashSet<String>,
        window: usize,
    ) -> bool {
        let before: Vec<char> = sentence_before(text, start)
            .into_iter()
            .filter(|&c| c != '\u{200c}')
            .collect();
        let toks = re!(INDIC_TOKEN).findall(&lower(&before));
        let from = toks.len().saturating_sub(window);
        toks[from..]
            .iter()
            .any(|tok| keywords.contains(tok.trim_matches(['#', '\''])))
    }

    fn period_for_hour24(h: u32) -> &'static str {
        if (4..12).contains(&h) {
            "morning"
        } else if (12..16).contains(&h) {
            "afternoon"
        } else if (16..20).contains(&h) {
            "evening"
        } else {
            "night"
        }
    }

    fn time_words(&self, h: u32, m: u32, period: Option<&str>) -> String {
        let l = self.lang;
        if l.code != "en" && (h == 0 || h > 12) {
            let template = if m != 0 { &l.time_hm } else { &l.time_on_hour };
            return fmt(
                template,
                &[("h", &l.cardinal(h as u128)), ("m", &l.cardinal(m as u128))],
            );
        }
        l.time_phrase(h, m, period)
    }

    // ------------------------------------------------------------------
    // Pre-clean
    // ------------------------------------------------------------------

    fn preclean(&self, text: &[char]) -> Vec<char> {
        let mut text: Vec<char> = ascii_digits(text)
            .into_iter()
            .map(|c| if c == '−' { '-' } else { c })
            .filter(|c| !matches!(c, '\u{200b}' | '\u{200e}' | '\u{200f}' | '\u{feff}'))
            .collect();
        text = re!(r#"["“”„]+"#).sub(&text, "");
        text = re!(r"(?<=[\s(\x{0964},—–-])['‘’](?![A-Za-z]['‘’])(?=\S)").sub(&text, "");
        text = re!(r"^['‘’](?![A-Za-z]['‘’])(?=\S)").sub(&text, "");
        text = re!(r"(?<=\S)(?<!['‘’][A-Za-z])['‘’](?=[\s,\x{0964}.!?;:)]|$)").sub(&text, "");
        text = re!(r"#{2,}").sub(&text, " ");
        text = re!(r"\s*(?:…|‥|\.{2,})\s*").sub(&text, ", ");
        text = re!(r"\s*[()]\s*").sub(&text, ", ");
        text = re!(r"(?<!:)/{2,}").sub(&text, " ");
        if !self.lang.number_word.is_empty() {
            let word = format!("{} ", self.lang.number_word);
            text = re!(r"\b[Nn][Oo]s?\.\s*(?=\d)").sub(&text, &word);
        }
        for (pat, word) in &self.abbrev {
            text = pat.sub(&text, word);
        }
        text = re!(r"^[,\s]+").sub(&text, "");
        text = re!(concat!(
            "[\u{1F300}-\u{1FAFF}\u{2700}-\u{27BF}\u{2600}-\u{26FF}\u{1F1E6}-\u{1F1FF}",
            "\u{FE00}-\u{FE0F}\u{1F000}-\u{1F0FF}\u{2B00}-\u{2BFF}\u{2300}-\u{23FF}",
            "\u{2190}-\u{21FF}\u{25A0}-\u{25FF}\u{2022}\u{2605}\u{2606}\u{2122}\u{A9}\u{AE}",
            "`~|\\\\{}\\[\\]«»‹›„‚•·¤§†‡]+"
        ))
        .sub(&text, " ");
        text = re!(r"(?<=[\d])\s*[–—]\s*(?=[\d])").sub(&text, "–");
        text = re!(r"\s+[–—]\s+(?=\d)").sub(&text, " – ");
        text = re!(r"[—]").sub(&text, " ");
        text = re!(r"(?<![\d\s])[–](?![\d\s])").sub(&text, " ");
        re!(r"[ \t]{2,}").sub(&text, " ")
    }

    // ------------------------------------------------------------------
    // Matchers (priority = order in `gather`)
    // ------------------------------------------------------------------

    fn gather(&self, text: &[char]) -> R<Vec<(usize, Span)>> {
        type M = fn(&Normalizer, &[char]) -> R<Vec<Span>>;
        let matchers: [M; 40] = [
            Self::m_email,
            Self::m_url,
            Self::m_handle,
            Self::m_comparison,
            Self::m_time_range,
            Self::m_duration,
            Self::m_time_verbalized,
            Self::m_time,
            Self::m_tamil_date,
            Self::m_date,
            Self::m_currency_range,
            Self::m_currency,
            Self::m_bare_rupee_suffix,
            // _m_long_digit_runs: FR3_TN_LONG_DIGITS only
            Self::m_amount_context,
            Self::m_percent_range,
            Self::m_percent,
            Self::m_negative,
            Self::m_designation,
            Self::m_vehicle_plate,
            Self::m_ordinal,
            Self::m_code,
            Self::m_id_digits,
            Self::m_chem,
            Self::m_math,
            Self::m_num_range,
            Self::m_dimensions,
            Self::m_roman,
            Self::m_unit_initialism,
            Self::m_kmb_suffix,
            Self::m_fraction,
            Self::m_ratio,
            Self::m_label,
            Self::m_acronym_model,
            Self::m_num_hyphen_word,
            Self::m_decade,
            Self::m_year,
            Self::m_quoted_letter,
            Self::m_short_mixed_token,
            Self::m_decimal,
            Self::m_int,
        ];
        let mut out = Vec::new();
        for (prio, matcher) in matchers.iter().enumerate() {
            // _m_long_digit_runs sits at 13 upstream: keep the later priorities in order
            let prio = if prio >= 13 { prio + 1 } else { prio };
            for span in matcher(self, text)? {
                out.push((prio, span));
            }
        }
        Ok(out)
    }

    /// `_spell_uri_part`: word chunks stay, digits spelled, symbols to words.
    fn spell_uri_part(&self, part: &str) -> R<String> {
        let pc = chars(part);
        let mut out: Vec<String> = Vec::new();
        for run in re!(r"[^\W\d_]+|\d+|[^\w\s]|_").findall(&pc) {
            let word = if py_isdigit(&run) {
                self.digits_spelled(&run, false)?
            } else {
                match run.as_str() {
                    "." => self.lang.sym("dot").into(),
                    "_" => self.lang.sym("underscore").into(),
                    "@" => self.lang.sym("at").into(),
                    "/" => self.lang.sym("slash").into(),
                    "-" => "dash".into(),
                    "+" => self.lang.sym("plus").into(),
                    "?" => "question mark".into(),
                    "=" => "equals".into(),
                    "&" => "ampersand".into(),
                    "#" => "hash".into(),
                    "%" => "percent".into(),
                    ":" => "colon".into(),
                    "~" => "tilde".into(),
                    _ if run.to_lowercase() == "www" => self.letters_spelled("WWW"),
                    _ if run.chars().count() == 1
                        || self.lang.abbrev_units.contains(&run.to_uppercase()) =>
                    {
                        self.letters_spelled(&run.to_uppercase())
                    }
                    _ => run.clone(),
                }
            };
            out.push(word);
        }
        Ok(out
            .into_iter()
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn m_email(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").find_iter(text) {
            out.push((m.start(), m.end(), self.spell_uri_part(&m.str(text, 0))?));
        }
        Ok(out)
    }

    /// `lexicon._patch_url`'s `_m_url` (L3): hosts with sub-domains, a
    /// sentence-final bare domain, and UPI IDs with a dotted local part.
    fn m_url(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        let pat = re!(concat!(
            r"https?://[^\s<>]+|www\.[^\s<>]+",
            r"|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)",
            r"(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w]|\.\w)"
        ));
        for m in pat.find_iter(text) {
            let token = m.str(text, 0);
            let token = token.trim_end_matches(['.', ',', '!', '?', ';', ')']);
            let body = re!(r"^https?://").sub(&chars(token), "");
            out.push((
                m.start(),
                m.start() + token.chars().count(),
                self.spell_uri_part(&st(&body))?,
            ));
        }
        let upi =
            re!(r"(?<![\w.@])[A-Za-z][\w\-]*\.[\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}(?![\w@]|\.\w)");
        for m in upi.find_iter(text) {
            out.push((m.start(), m.end(), self.spell_uri_part(&m.str(text, 0))?));
        }
        Ok(out)
    }

    fn m_handle(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"(?<![\w.])@([A-Za-z](?:[\w.]*\w)?)").find_iter(text) {
            let spoken = format!(
                "{} {}",
                self.lang.sym("at"),
                self.spell_uri_part(&m.str(text, 1))?
            );
            out.push((m.start(), m.end(), spoken));
        }
        Ok(out)
    }

    // --- time ------------------------------------------------------------

    /// `_resolve_time(lead_p, h, mm, ampm, marker, trail_p)` → (h, m, period).
    fn resolve_time(&self, g: &[Option<String>]) -> R<(u32, u32, Option<String>)> {
        let l = self.lang;
        let h = py_int(g[1].as_deref().unwrap_or(""))? as u32;
        let m = match &g[2] {
            Some(mm) => py_int(mm)? as u32,
            None => 0,
        };
        let src_period = g[0].clone().or_else(|| g[5].clone());
        let mut period: Option<String> = src_period
            .as_deref()
            .and_then(|p| l.period_detect.get(p))
            .map(str::to_string);
        let ampm = g[3].as_deref();
        if l.code != "en" && ampm.is_none() && src_period.is_none() {
            return Ok((h, m, None));
        }
        let infer = l.ampm_words.0.is_empty();
        if let Some(a) = ampm {
            let a = a.to_lowercase().replace('.', "");
            let h24 = h % 12 + if a == "pm" { 12 } else { 0 };
            if infer && period.is_none() {
                period = Some(Self::period_for_hour24(h24).into());
            }
        } else if (h > 12 || h == 0) && infer && period.is_none() {
            period = Some(Self::period_for_hour24(h).into());
        }
        if !l.ampm_words.0.is_empty() && ampm.is_none() && (h > 12 || h == 0) {
            return Ok((h, m, period));
        }
        let h12 = if h.is_multiple_of(12) { 12 } else { h % 12 };
        Ok((h12, m, period))
    }

    fn groups(m: &Match, text: &[char], from: usize, n: usize) -> Vec<Option<String>> {
        (from..from + n).map(|g| m.get(text, g)).collect()
    }

    fn m_duration(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let sec_word = l.sym.get("second").unwrap_or("seconds");
        let mut out = Vec::new();
        for m in re!(r"\b(\d{1,3}):(\d{2}):(\d{2})\b").find_iter(text) {
            let (h, mm, ss) = (
                py_int(&m.str(text, 1))?,
                py_int(&m.str(text, 2))?,
                py_int(&m.str(text, 3))?,
            );
            if mm > 59 || ss > 59 {
                continue;
            }
            let hours = l.abbrev_units.get("HRS").unwrap_or("hours");
            let minutes = l.sym("minute");
            let parts: Vec<String> = [(h, hours), (mm, minutes), (ss, sec_word)]
                .iter()
                .filter(|(n, _)| *n != 0)
                .map(|(n, unit)| {
                    let unit = if *n != 1 || l.code != "en" {
                        unit
                    } else {
                        unit.trim_end_matches('s')
                    };
                    l.join_amount(&l.cardinal(*n), unit)
                })
                .collect();
            let rep = if parts.is_empty() {
                l.join_amount(&l.cardinal(0), sec_word)
            } else {
                parts.join(" ")
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_time_verbalized(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.time_verbalized.find_iter(text) {
            let (h, mm) = (py_int(&m.str(text, 1))?, py_int(&m.str(text, 2))?);
            if h > 23 || mm > 59 {
                continue;
            }
            let mut rep = self.time_words(h as u32, mm as u32, None);
            if let Some(sfx) = l.marker_suffixes.get(&m.str(text, 3)) {
                rep = l.attach_postposition(&rep, sfx).unwrap_or(rep);
            }
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_time(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.time_pat.find_iter(text) {
            let g = Self::groups(&m, text, 1, 6);
            let (lead_p, h, mm, ampm, marker, trail_p) = (&g[0], &g[1], &g[2], &g[3], &g[4], &g[5]);
            let Some(h) = h else { continue };
            let hh = py_int(h)?;
            let mm_v = match mm {
                Some(x) => Some(py_int(x)?),
                None => None,
            };
            let is_hhmm = mm_v.is_some_and(|v| v <= 59) && hh < 24;
            let src = lead_p.is_some() || trail_p.is_some();
            if !(ampm.is_some() || marker.is_some() || (src && (is_hhmm || mm.is_none())))
                && !(is_hhmm && mm.is_some() && hh > 12)
                && !(is_hhmm && !self.keyword_before(text, m.start(), &self.ratio_kws, 4))
            {
                continue;
            }
            if mm.is_none() && !(ampm.is_some() || marker.is_some()) {
                continue;
            }
            if mm_v.is_some_and(|v| v > 59) || hh > 23 {
                continue;
            }
            let (h12, mnt, period) = self.resolve_time(&g)?;
            let mut rep = match lead_p.as_ref().or(trail_p.as_ref()) {
                Some(src_period) => format!("{src_period} {}", l.time_phrase(h12, mnt, None)),
                None => self.time_words(h12, mnt, period.as_deref()),
            };
            if let Some(a) = ampm {
                if !l.ampm_words.0.is_empty() {
                    if let Some(w) = l.ampm_words.get(&a.to_lowercase().replace('.', "")) {
                        if mnt == 0 {
                            rep = l.two_digit(h12 as u128).to_string();
                        }
                        rep = format!("{rep} {w}");
                    }
                }
            }
            if let Some(sfx) = l.marker_suffixes.get(marker.as_deref().unwrap_or("")) {
                rep = l.attach_postposition(&rep, sfx).unwrap_or(rep);
            }
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_time_range(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.time_range.find_iter(text) {
            let gl = Self::groups(&m, text, 2, 6);
            let gr = Self::groups(&m, text, 9, 6);
            if gl[1].is_none() || gr[1].is_none() {
                continue;
            }
            let looks = |g: &[Option<String>]| g[3].is_some() || g[4].is_some() || g[2].is_some();
            if !(looks(&gl) && looks(&gr)) {
                continue;
            }
            let (h1, m1, p1) = self.resolve_time(&gl)?;
            let (h2, m2, p2) = self.resolve_time(&gr)?;
            let mut a = self.time_words(h1, m1, p1.as_deref());
            let mut b = self.time_words(h2, m2, p2.as_deref());
            if !l.ampm_words.0.is_empty() {
                let word = |ap: &str| {
                    l.ampm_words
                        .get(&ap.to_lowercase().replace('.', ""))
                        .map(str::to_string)
                        .ok_or(Fail::Crash("KeyError: AMPM_WORDS"))
                };
                if let Some(ap) = &gl[3] {
                    let head = if m1 == 0 {
                        l.two_digit(h1 as u128).to_string()
                    } else {
                        a
                    };
                    a = format!("{head} {}", word(ap)?);
                }
                if let Some(ap) = &gr[3] {
                    let head = if m2 == 0 {
                        l.two_digit(h2 as u128).to_string()
                    } else {
                        b
                    };
                    b = format!("{head} {}", word(ap)?);
                }
            }
            let mut rep = fmt(&l.range_to, &[("a", &a), ("b", &b)]);
            let tail = py_strip(l.range_to.rsplit("{b}").next().unwrap_or("")).to_string();
            if !tail.is_empty() {
                let check = Regex::new(&format!(r"\s*{}(?!\w)", escape(&tail)));
                if check.match_at(&sl(text, m.end(), text.len()), 0).is_some() {
                    let keep = rep.chars().count().saturating_sub(tail.chars().count());
                    rep = rep
                        .chars()
                        .take(keep)
                        .collect::<String>()
                        .trim_end()
                        .to_string();
                }
            }
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    // --- dates -----------------------------------------------------------

    fn m_tamil_date(&self, text: &[char]) -> R<Vec<Span>> {
        if !matches!(self.lang.code, "en" | "ta") {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let pat = re!(&format!(
            r"(?<!\w)({})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?![\dA-Za-z])(?:\s*தேதி)?",
            t::TAMIL_MONTHS.join("|")
        ));
        for m in pat.find_iter(text) {
            let d = py_int(&m.str(text, 2))? as usize;
            if (1..=31).contains(&d) {
                out.push((
                    m.start(),
                    m.end(),
                    format!("{} {} தேதி", m.str(text, 1), t::TAMIL_ORDINALS[d]),
                ));
            }
        }
        for m in re!(r"(?<!\w)(\d+)-?(?:ஆம்|ம்)(?![\w\x{0B80}-\x{0BFF}])").find_iter(text)
        {
            let d = py_int(&m.str(text, 1))?;
            if !(1..=31).contains(&d) {
                if self.lang.code == "en" {
                    return Err(Fail::Norm {
                        code: "unsupported_tamil_ordinal",
                        span: Some((m.start(), m.end())),
                        source: Some(text.to_vec()),
                    });
                }
                out.push((m.start(), m.end(), self.lang.ordinal(d, "ஆம்")));
            } else {
                out.push((
                    m.start(),
                    m.end(),
                    t::TAMIL_ORDINALS[d as usize].to_string(),
                ));
            }
        }
        Ok(out)
    }

    fn date_words(
        &self,
        day: u128,
        month: u32,
        year: Option<u128>,
        ordinal: Option<&str>,
        month_first: bool,
    ) -> R<String> {
        let l = self.lang;
        let out = if l.code == "ta" {
            let ord = t::TAMIL_ORDINALS
                .get(day as usize)
                .ok_or(Fail::Crash("IndexError: _TAMIL_ORDINALS"))?;
            format!("{} {ord} தேதி", l.month(month)?)
        } else {
            let ordinal = ordinal.filter(|o| !o.is_empty());
            let suffix = ordinal
                .filter(|o| l.ordinal_input_suffixes.iter().any(|s| s == o))
                .unwrap_or("");
            let day_words = if ordinal.is_some() {
                l.ordinal(day, suffix)
            } else {
                l.cardinal(day)
            };
            if month_first && l.code != "en" {
                format!("{} {day_words}", l.month(month)?)
            } else {
                format!("{day_words} {}", l.month(month)?)
            }
        };
        Ok(match year {
            Some(y) => format!("{out} {}", l.year(y)),
            None => out,
        })
    }

    fn opt_year(m: &Match, text: &[char], g: usize) -> R<Option<u128>> {
        match m.get(text, g) {
            Some(y) if !y.is_empty() => Ok(Some(py_int(&y)?)),
            _ => Ok(None),
        }
    }

    fn m_date(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.date_range.find_iter(text) {
            let mo = self
                .month_index(&m.str(text, 3))
                .ok_or(Fail::Crash("TypeError: month None"))?;
            let year = Self::opt_year(&m, text, 4)?;
            for g in [1, 2] {
                // _calendar_check without a span: the error is not recoverable
                calendar_check(
                    py_int(&m.str(text, g))?,
                    mo as u128,
                    year.unwrap_or(2000),
                    None,
                )?;
            }
            let days = fmt(
                &l.range,
                &[
                    ("a", &l.cardinal(py_int(&m.str(text, 1))?)),
                    ("b", &l.cardinal(py_int(&m.str(text, 2))?)),
                ],
            );
            let mut rep = format!("{days} {}", l.month(mo)?);
            if let Some(y) = year {
                rep = format!("{rep} {}", l.year(y));
            }
            out.push((m.start(), m.end(), rep));
        }
        for m in re!(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])").find_iter(text) {
            let rep = self.date_words(
                py_int(&m.str(text, 3))?,
                py_int(&m.str(text, 2))? as u32,
                Some(py_int(&m.str(text, 1))?),
                None,
                false,
            )?;
            out.push((m.start(), m.end(), rep));
        }
        for m in re!(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b(?![/A-Za-z-])").find_iter(text) {
            let (d, mo) = (py_int(&m.str(text, 1))?, py_int(&m.str(text, 3))?);
            if !((1..=31).contains(&d) && (1..=12).contains(&mo)) {
                continue;
            }
            let y4 = m.str(text, 4);
            if y4.chars().count() == 2 {
                // `y += self.two_digit_year_base` with the default None
                return Err(Fail::Crash("TypeError: two_digit_year_base is None"));
            }
            let rep = self.date_words(d, mo as u32, Some(py_int(&y4)?), None, false)?;
            out.push((m.start(), m.end(), rep));
        }
        for m in self.date_day_month.find_iter(text) {
            let d = py_int(&m.str(text, 1))?;
            let Some(mo) = self.month_index(&m.str(text, 3)) else {
                continue;
            };
            if !(1..=31).contains(&d) {
                continue;
            }
            let ord = m.get(text, 2);
            let rep =
                self.date_words(d, mo, Self::opt_year(&m, text, 4)?, ord.as_deref(), false)?;
            out.push((m.start(), m.end(), rep));
        }
        for m in self.date_month_day.find_iter(text) {
            let Some(mo) = self.month_index(&m.str(text, 1)) else {
                continue;
            };
            let d = py_int(&m.str(text, 2))?;
            if !(1..=31).contains(&d) {
                continue;
            }
            let ord = m.get(text, 3);
            let rep = self.date_words(d, mo, Self::opt_year(&m, text, 4)?, ord.as_deref(), true)?;
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    // --- currency --------------------------------------------------------

    fn cur_key(&self, symtok: &str) -> String {
        let t1 = py_strip(symtok).trim_end_matches('.');
        let key = py_strip(t1).to_uppercase();
        if let Some((_, v)) = t::CURRENCY_CODE_KEYS.iter().find(|(k, _)| *k == key) {
            return v.to_string();
        }
        let cw = &self.lang.currency_words;
        cw.get(&key)
            .or_else(|| cw.get(&format!("{key}.")))
            .unwrap_or("rupee")
            .to_string()
    }

    fn m_currency_range(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.currency_range.find_iter(text) {
            let (sign, amount) = (m.str(text, 1), m.str(text, 3));
            if !sign.is_empty() && (amount.starts_with('-') || amount.starts_with('+')) {
                return Err(Fail::Norm {
                    code: "conflicting_sign",
                    span: Some((m.start(), m.end())),
                    source: Some(text.to_vec()),
                });
            }
            let a = self.number_words(&format!("{sign}{amount}"))?;
            let b = self.number_words(&m.str(text, 4))?;
            let cw = l.sym(&self.cur_key(&m.str(text, 2))).to_string();
            out.push((
                m.start(),
                m.end(),
                fmt(&l.range, &[("a", &a), ("b", &l.join_amount(&b, &cw))]),
            ));
        }
        Ok(out)
    }

    fn m_bare_rupee_suffix(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"(?<![\w₹.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s?/-").find_iter(text) {
            let words = self.number_words(&m.str(text, 1).replace(',', ""))?;
            out.push((m.start(), m.end(), l.join_amount(&words, l.sym("rupee"))));
        }
        Ok(out)
    }

    fn m_currency(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.currency.find_iter(text) {
            let (sign1, sign2) = (m.str(text, 1), m.str(text, 3));
            if !sign1.is_empty() && !sign2.is_empty() {
                return Err(Fail::Norm {
                    code: "conflicting_sign",
                    span: Some((m.start(), m.end())),
                    source: Some(text.to_vec()),
                });
            }
            let key = self.cur_key(&m.str(text, 2));
            let num = m.str(text, 4).replace(',', "");
            let scale = m.get(text, 5).filter(|s| !s.is_empty());
            let cur_word = l.sym(&key).to_string();
            let frac2 = num
                .split_once('.')
                .is_some_and(|(_, f)| f.chars().count() == 2);
            let mut rep = if frac2 && scale.is_none() {
                let (ip, fp) = num.split_once('.').unwrap();
                let ipn = if ip.is_empty() { 0 } else { py_int(ip)? };
                let main = l.cardinal(ipn);
                let sub_word: Option<String> = match (key.as_str(), l.sym.get("paisa")) {
                    ("rupee", Some(p)) if !p.is_empty() => Some(p.into()),
                    ("rupee", _) => None,
                    _ => Some(l.sym("cent").into()),
                };
                if fp.trim_matches('0').is_empty() {
                    l.join_amount(&main, &cur_word)
                } else if let Some(sub) = sub_word.filter(|s| !s.is_empty()) {
                    format!(
                        "{} {}",
                        l.join_amount(&main, &cur_word),
                        l.join_amount(&l.cardinal(py_int(fp)?), &sub)
                    )
                } else {
                    l.join_amount(&l.decimal(ipn, fp)?, &cur_word)
                }
            } else {
                let words = self.number_words(&num)?;
                match &scale {
                    Some(scale) => {
                        let sl_ = scale.to_lowercase();
                        let canon = l.scale_words.get(scale).map(str::to_string);
                        let scale_word = if canon.is_none() && t::EN_SCALES.contains(&sl_.as_str())
                        {
                            scale.clone()
                        } else {
                            let canon = canon.unwrap_or_else(|| {
                                if sl_ == "cr" || sl_.contains("crore") {
                                    "crore".into()
                                } else if sl_.contains("lakh") {
                                    "lakh".into()
                                } else {
                                    "thousand".into()
                                }
                            });
                            l.scale_noun(&canon, &num)
                        };
                        let mut rep = l.join_amount(&l.join_amount(&words, &scale_word), &cur_word);
                        if let Some(case) = m.get(text, 6).filter(|c| !c.is_empty()) {
                            rep = l.attach_postposition(&rep, &case).unwrap_or(rep);
                        }
                        rep
                    }
                    None => l.join_amount(&words, &cur_word),
                }
            };
            let sign = if !sign1.is_empty() { sign1 } else { sign2 };
            if !sign.is_empty() {
                rep = format!(
                    "{} {rep}",
                    if sign == "-" {
                        l.sym("minus")
                    } else {
                        l.sym("plus")
                    }
                );
            }
            let whole = m.str(text, 0);
            let space = whole.chars().next().is_some_and(is_space)
                || (m.start() > 0 && is_alnum(text[m.start() - 1]));
            out.push((
                m.start(),
                m.end(),
                if space { format!(" {rep}") } else { rep },
            ));
        }
        Ok(out)
    }

    fn m_amount_context(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in self.amount_units.find_iter(text) {
            out.push((m.start(), m.end(), self.number_words(&m.str(text, 1))?));
        }
        let banned = ["number", "id", "code", "otp", "pin"];
        for m in re!(r"(?<![\w.,+/-])\d+(?:,\d{2,3})*(?!\w|[.,/-]\d)").find_iter(text) {
            let before = lower(&sentence_before(text, m.start()));
            let toks = re!(r"\w+").findall(&before);
            let toks = &toks[toks.len().saturating_sub(3)..];
            if toks.iter().any(|t| t::AMOUNT_LABELS.contains(&t.as_str()))
                && !toks.iter().any(|t| banned.contains(&t.as_str()))
            {
                out.push((m.start(), m.end(), self.number_words(&m.str(text, 0))?));
            }
        }
        Ok(out)
    }

    // --- percent ---------------------------------------------------------

    fn m_percent_range(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)\s*%").find_iter(text) {
            let (a, b) = (
                self.number_words(&m.str(text, 1))?,
                self.number_words(&m.str(text, 2))?,
            );
            out.push((
                m.start(),
                m.end(),
                format!(
                    "{} {}",
                    fmt(&l.range, &[("a", &a), ("b", &b)]),
                    l.sym("percent")
                ),
            ));
        }
        Ok(out)
    }

    fn m_percent(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"(-?)(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*%").find_iter(text) {
            let mut words = self.number_words(&m.str(text, 2))?;
            if !m.str(text, 1).is_empty() {
                words = format!("{} {words}", l.sym("minus"));
            }
            out.push((m.start(), m.end(), format!("{words} {}", l.sym("percent"))));
        }
        Ok(out)
    }

    // --- designations / codes -------------------------------------------

    fn m_designation(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"\b([A-Z][a-z]{2,})-([A-Za-z0-9]+)\b").find_iter(text) {
            let seg = m.str(text, 2);
            if !seg.chars().any(is_decimal) {
                continue;
            }
            if self.month_index(&seg).is_some() || self.month_index(&m.str(text, 1)).is_some() {
                continue;
            }
            out.push((
                m.start(),
                m.end(),
                format!("{} {}", m.str(text, 1), self.spell_model_segment(&seg)?),
            ));
        }
        for m in re!(r"\b([A-Z][a-z]{2,})\s+([A-Z]{1,4}-[A-Za-z0-9]+)\b").find_iter(text) {
            let brand = m.str(text, 1).to_lowercase();
            if self.id_keywords.contains(&brand) || self.label_words.contains(&brand) {
                continue;
            }
            let seg = m.str(text, 2);
            if !seg.chars().any(is_decimal) {
                continue;
            }
            let (s, e) = m.span(2).unwrap();
            out.push((s, e, self.spell_model_segment(&seg)?));
        }
        Ok(out)
    }

    fn m_code(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        let pat = re!(concat!(
            r"#?\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])",
            r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+(?![A-Za-z0-9])"
        ));
        for m in pat.find_iter(text) {
            let whole = m.str(text, 0);
            let tok = whole.trim_start_matches('#');
            let parts: Vec<String> = re!(r"[-/]").split(&chars(tok));
            let is_month = |p: &str| self.month_index(p).is_some();
            if parts.iter().any(|p| is_month(p))
                && parts.iter().filter(|p| !is_month(p)).all(|p| py_isdigit(p))
            {
                continue;
            }
            if parts.iter().any(|p| {
                re!(r"[a-z][a-z]+|[A-Z][a-z]{2,}")
                    .fullmatch(&chars(p))
                    .is_some()
            }) {
                continue;
            }
            if parts.len() == 2
                && py_isdigit(&parts[0])
                && py_isalpha(&parts[1])
                && parts[1].chars().count() >= 3
            {
                continue;
            }
            if parts.len() == 2
                && py_isdigit(&parts[0])
                && t::UNIT_INITIALISMS.contains(&parts[1].to_uppercase().as_str())
            {
                continue;
            }
            if re!(r"\d{1,3}-?[A-Za-z]").fullmatch(&chars(tok)).is_some()
                && self.keyword_before(text, m.start(), &self.label_words, 2)
            {
                continue;
            }
            if tok.chars().filter(|&c| is_alnum(c)).count() <= 2 {
                continue;
            }
            out.push((m.start(), m.end(), self.spell_alnum(tok, true)?));
        }
        let pat2 = re!(r"\b(?=(?:[A-Za-z]*\d){2})(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{5,}\b");
        for m in pat2.find_iter(text) {
            let tok = m.str(text, 0);
            let tc = chars(&tok);
            if re!(r"\d+s").fullmatch(&tc).is_some() || re!(r"\d+[x×]\d+").fullmatch(&tc).is_some()
            {
                continue;
            }
            if let Some(m2) = re!(r"\d+([A-Za-z]{1,4})").fullmatch(&tc) {
                if t::UNIT_INITIALISMS.contains(&m2.str(&tc, 1).to_uppercase().as_str()) {
                    continue;
                }
            }
            if re!(r"(?:[A-Z][a-z]?\d*)+").fullmatch(&tc).is_some() && chem_ok(&tok) {
                continue;
            }
            out.push((m.start(), m.end(), self.spell_alnum(&tok, true)?));
        }
        Ok(out)
    }

    fn m_vehicle_plate(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b([A-Z]{2})[ -](\d{1,2})[ -]([A-Z]{1,3})[ -](\d{3,4})\b").find_iter(text) {
            let mut words = Vec::new();
            for g in 1..=4 {
                for ch in m.str(text, g).chars() {
                    words.push(if is_digit(ch) {
                        l.digit(py_int_char(ch)?).to_string()
                    } else {
                        l.letter(ch)
                    });
                }
            }
            out.push((m.start(), m.end(), words.join(", ")));
        }
        Ok(out)
    }

    fn after_tokens(text: &[char], end: usize, n: usize) -> Vec<String> {
        let mut toks = re!(INDIC_TOKEN).findall(&lower(&sentence_after(text, end)));
        toks.truncate(n);
        toks
    }

    fn m_id_digits(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        let pat = re!(concat!(
            r"(\+?)(?:#?\b\d+(?:[-\s]\d+)+\b|#\s?\d{3,}\b",
            r"|(?<![\d.])(?<!\d,)\d{7,}(?![.,]?\d))"
        ));
        for m in pat.find_iter(text) {
            let has_plus = m.str(text, 1) == "+";
            let whole = m.str(text, 0);
            let tok = whole.trim_start_matches(['+', '#', ' ']);
            let groups: Vec<String> = re!(r"[-\s]").split(&chars(tok));
            let digits: String = groups.concat();
            let ndig = digits.chars().count();
            let is_id = (has_plus && ndig >= 8)
                || whole.starts_with('#')
                || (ndig >= 7 && !tok.contains(' '))
                || (groups.iter().any(|g| g.chars().count() >= 4)
                    && groups.len() >= 2
                    && tok.contains('-'))
                || (groups.len() >= 3 && tok.contains('-'))
                || self.keyword_before(text, m.start(), &self.id_keywords, 3)
                || Self::after_tokens(text, m.end(), 4)
                    .iter()
                    .any(|t| self.id_keywords.contains(t));
            if !is_id {
                continue;
            }
            let a_toks2 = Self::after_tokens(text, m.end(), 4);
            if tok.contains('-')
                && groups.len() == 2
                && groups.iter().all(|g| g.chars().count() <= 3)
                && py_int(&groups[0])? < py_int(&groups[1])?
                && !self.keyword_before(text, m.start(), &self.id_keywords, 3)
                && !a_toks2.iter().any(|t| self.id_keywords.contains(t))
            {
                continue;
            }
            let mut rep = self.digits_spelled(&digits, true)?;
            if has_plus {
                rep = format!("{} {rep}", l.sym("plus"));
            }
            out.push((m.start(), m.end(), rep));
        }
        let year_kw: HashSet<String> = l
            .year_words
            .iter()
            .map(|w| w.to_lowercase())
            .chain(std::iter::once("year".to_string()))
            .collect();
        let mut id_after: HashSet<String> =
            ["dial", "डायल"].iter().map(|s| s.to_string()).collect();
        id_after.extend(l.id_after_words.iter().cloned());
        for m in re!(r"(?<![\d.])(?<!\d,)\d{3,6}(?![.,]?\d)(?!s\b)").find_iter(text) {
            let after_toks = Self::after_tokens(text, m.end(), 6);
            let spelled = |s: &Self| s.digits_spelled(&m.str(text, 0), true);
            if after_toks.iter().any(|t| id_after.contains(t)) {
                out.push((m.start(), m.end(), spelled(self)?));
                continue;
            }
            if after_toks
                .iter()
                .take(4)
                .any(|t| self.id_keywords.contains(t))
                && !self.keyword_before(text, m.start(), &self.label_words, 2)
                && !self.keyword_before(text, m.start(), &year_kw, 2)
            {
                out.push((m.start(), m.end(), spelled(self)?));
                continue;
            }
            if self.keyword_before(text, m.start(), &self.id_keywords, 3) {
                if self.keyword_before(text, m.start(), &self.label_words, 2) {
                    continue;
                }
                if self.keyword_before(text, m.start(), &year_kw, 2) {
                    continue;
                }
                out.push((m.start(), m.end(), spelled(self)?));
            }
        }
        Ok(out)
    }

    fn m_chem(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(?:[A-Z][a-z]?\d*){1,8}\b").find_iter(text) {
            let tok = m.str(text, 0);
            if tok.chars().count() < 2
                || t::ACRONYM_WHITELIST.contains(&tok.to_uppercase().as_str())
            {
                continue;
            }
            if !chem_ok(&tok) {
                continue;
            }
            let mut words = Vec::new();
            for piece in re!(r"[A-Z][a-z]?|\d+").findall(&chars(&tok)) {
                if py_isdigit(&piece) {
                    words.push(l.cardinal(py_int(&piece)?));
                } else {
                    words.push(self.letters_spelled(&piece.to_uppercase()));
                }
            }
            out.push((m.start(), m.end(), words.join(" ")));
        }
        Ok(out)
    }

    // --- math ------------------------------------------------------------

    fn m_comparison(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let pat = re!(concat!(
            r"(?<![\w.])(?:(?P<left>(?:₹\s*)?[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*%)?)\s*)?",
            r"(?P<op><=|>=|≤|≥|<|>)\s*(?P<right>(?:₹\s*)?[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*%)?)"
        ));
        let words = |token: &str| -> R<String> {
            let s = py_strip(token);
            if let Some(rest) = s.strip_prefix('₹') {
                return Ok(l.join_amount(&self.number_words(py_strip(rest))?, l.sym("rupee")));
            }
            if let Some(rest) = s.strip_suffix('%') {
                return Ok(format!(
                    "{} {}",
                    self.number_words(py_strip(rest))?,
                    l.sym("percent")
                ));
            }
            self.number_words(s)
        };
        let mut out = Vec::new();
        for m in pat.find_iter(text) {
            let op = m.str(text, 2);
            let mut comparison = l
                .sym(if op.starts_with('<') || op == "≤" {
                    "lt"
                } else {
                    "gt"
                })
                .to_string();
            if op.chars().count() > 1 || op == "≤" || op == "≥" {
                if l.code == "en" {
                    comparison += " or equal to";
                } else {
                    let or = super::lang::lookup_str(t::COMPARISON_OR, l.code)
                        .ok_or(Fail::Crash("KeyError: ors"))?;
                    comparison = format!("{comparison} {or} {}", l.sym("equals"));
                }
            }
            let left = match m.get(text, 1) {
                Some(s) if !s.is_empty() => format!("{} ", words(&s)?),
                _ => String::new(),
            };
            let right = words(&m.str(text, 3))?;
            let rep = if l.code == "en" {
                format!("{left}{comparison} {right}")
            } else {
                format!("{left}{right} {comparison}")
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn operand_words(&self, v: &str) -> R<String> {
        if v.chars().next().is_some_and(is_digit) {
            self.number_words(v)
        } else {
            Ok(self.lang.letter(v.chars().next().unwrap_or(' ')))
        }
    }

    fn m_math(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"(\d+(?:\.\d+)?|[A-Za-z])\s*\^\s*(\d+|[A-Za-z])").find_iter(text) {
            let (bw, ew) = (
                self.operand_words(&m.str(text, 1))?,
                self.operand_words(&m.str(text, 2))?,
            );
            out.push((
                m.start(),
                m.end(),
                fmt(&l.power, &[("base", &bw), ("exp", &ew)]),
            ));
        }
        for m in re!(r"(\d+(?:\.\d+)?|[A-Za-z])²").find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                format!(
                    "{} {}",
                    self.operand_words(&m.str(text, 1))?,
                    l.sym("squared")
                ),
            ));
        }
        for m in re!(r"√\s*(\d+(?:\.\d+)?)").find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                format!("{} {}", l.sym("sqrt"), self.number_words(&m.str(text, 1))?),
            ));
        }
        let eq = re!(concat!(
            r"(?<![\w.])((?:[A-Za-z]{1,2}|\d+(?:\.\d+)?)",
            r"(?:\s*[=+\-−×*÷/^]\s*(?:[A-Za-z]{1,2}|\d+(?:\.\d+)?))+)"
        ));
        for m in eq.find_iter(text) {
            let span = m.str(text, 1);
            if re!(r"[=×÷^√]").search(&chars(&span)).is_none() {
                continue;
            }
            let (s, e) = m.span(1).unwrap();
            out.push((s, e, self.render_math_expr(&span)?));
        }
        let verbal =
            re!(r"\b([A-Za-z])\b(?=\s+(?:squared|cubed|plus|minus|times|equals|to the power)\b)");
        for m in verbal.find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                l.letter(m.str(text, 1).chars().next().unwrap_or(' ')),
            ));
        }
        Ok(out)
    }

    fn render_math_expr(&self, span: &str) -> R<String> {
        let l = self.lang;
        let op = |tok: &str| -> Option<&str> {
            Some(match tok {
                "=" => l.sym("equals"),
                "+" => l.sym("plus"),
                "-" | "−" => l.sym("minus"),
                "×" | "*" => l.sym("times"),
                "÷" | "/" => l.sym("divide"),
                _ => return None,
            })
        };
        let toks = re!(r"\d+(?:\.\d+)?|[A-Za-z]+|[=+\-−×*÷/^]").findall(&chars(span));
        let mut out: Vec<String> = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            let tok = &toks[i];
            if tok == "^" {
                let exp = toks.get(i + 1).map_or("", String::as_str);
                let ew = if exp.chars().next().is_some_and(is_digit) {
                    self.number_words(exp)?
                } else {
                    self.letters_spelled(&exp.to_uppercase())
                };
                let base = out.pop().unwrap_or_default();
                out.push(fmt(&l.power, &[("base", &base), ("exp", &ew)]));
                i += 2;
                continue;
            }
            if let Some(w) = op(tok) {
                out.push(w.to_string());
            } else if tok.chars().next().is_some_and(is_digit) {
                out.push(self.number_words(tok)?);
            } else {
                out.push(self.letters_spelled(&tok.to_uppercase()));
            }
            i += 1;
        }
        Ok(out.join(" "))
    }

    // --- ranges / dimensions / roman / K-M-B ------------------------------

    fn m_num_range(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"(\d+(?:\.\d+)?)\s*[–]\s*(\d+(?:\.\d+)?)").find_iter(text) {
            let (a, b) = (
                self.number_words(&m.str(text, 1))?,
                self.number_words(&m.str(text, 2))?,
            );
            out.push((m.start(), m.end(), fmt(&l.range, &[("a", &a), ("b", &b)])));
        }
        for m in re!(r"\b(\d{1,3})-(\d{1,3})\b").find_iter(text) {
            if self.keyword_before(text, m.start(), &self.id_keywords, 3) {
                continue;
            }
            let (a, b) = (py_int(&m.str(text, 1))?, py_int(&m.str(text, 2))?);
            let (ca, cb) = (l.cardinal(a), l.cardinal(b));
            let range = fmt(&l.range, &[("a", &ca), ("b", &cb)]);
            let rep = if a == b {
                format!("{ca} {cb}")
            } else if a > b {
                let unit = re_i!(r"\s+(?:days?|months?|years?|hours?|minutes?)\b");
                if unit.match_at(&sl(text, m.end(), text.len()), 0).is_some() {
                    range
                } else {
                    format!("{ca} {cb}")
                }
            } else {
                range
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_dimensions(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\b").find_iter(text) {
            if m.str(text, 1) == "0" {
                continue;
            }
            let rep = format!(
                "{} {} {}",
                self.number_words(&m.str(text, 1))?,
                l.sym("times"),
                self.number_words(&m.str(text, 2))?
            );
            out.push((m.start(), m.end(), rep));
        }
        for m in re!(r"(?<=\s)[x×](?=\s\d)").find_iter(text) {
            out.push((m.start(), m.end(), l.sym("times").to_string()));
        }
        Ok(out)
    }

    fn m_roman(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        let pat = re!(concat!(
            r"\b(?=[IVXLCDM]{2,7}\b)M{0,3}(?:CM|CD|D?C{0,3})",
            r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b"
        ));
        for m in pat.find_iter(text) {
            let tok = m.str(text, 0);
            if tok.is_empty() || !self.keyword_before(text, m.start(), &self.roman_ctx, 2) {
                continue;
            }
            out.push((m.start(), m.end(), self.lang.cardinal(roman_value(&tok))));
        }
        Ok(out)
    }

    fn m_kmb_suffix(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)([KMB])\b").find_iter(text) {
            let letter = self
                .lang
                .letter(m.str(text, 2).chars().next().unwrap_or(' '));
            out.push((
                m.start(),
                m.end(),
                format!("{} {letter}", self.number_words(&m.str(text, 1))?),
            ));
        }
        Ok(out)
    }

    fn m_ordinal(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in self.ordinal.find_iter(text) {
            let n = py_int(&m.str(text, 1))?;
            let suf = m.str(text, 2).to_lowercase();
            let rep = if matches!(suf.as_str(), "st" | "nd" | "rd" | "th") {
                self.lang.ordinal(n, "")
            } else {
                self.lang.ordinal(n, &suf)
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_fraction(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(\d{1,4}(?:\.\d+)?)/(\d{1,4}(?:\.\d+)?)\b").find_iter(text) {
            let (a, b) = (m.str(text, 1), m.str(text, 2));
            let rep = match (a.as_str(), b.as_str()) {
                ("1", "2") => l.sym("half").to_string(),
                ("1", "4") => l.sym("quarter").to_string(),
                ("3", "4") => l.sym("three_quarters").to_string(),
                _ => fmt(
                    &l.fraction,
                    &[
                        ("num", &self.number_words(&a)?),
                        ("den", &self.number_words(&b)?),
                    ],
                ),
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_ratio(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(\d{1,3}):(\d{1,2})\b").find_iter(text) {
            let mm = m.str(text, 2);
            let mmv = py_int(&mm)?;
            if mm.chars().count() == 2
                && mmv <= 59
                && !self.keyword_before(text, m.start(), &self.ratio_kws, 4)
            {
                continue;
            }
            let a = l.cardinal(py_int(&m.str(text, 1))?);
            let after = st(&sl(text, m.end(), m.end() + 20));
            let after = py_strip(&after);
            if self
                .ratio_kws
                .iter()
                .filter(|w| !w.is_ascii())
                .any(|w| after.starts_with(w.as_str()))
            {
                out.push((m.start(), m.end(), format!("{a} {}", l.cardinal(mmv))));
                continue;
            }
            out.push((
                m.start(),
                m.end(),
                fmt(&l.ratio, &[("a", &a), ("b", &l.cardinal(mmv))]),
            ));
        }
        Ok(out)
    }

    fn m_label(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.label.find_iter(text) {
            let mut words = vec![m.str(text, 1)];
            if let Some(n) = m.get(text, 2).filter(|s| !s.is_empty()) {
                words.push(l.cardinal(py_int(&n)?));
                if let Some(c) = m.get(text, 3).filter(|s| !s.is_empty()) {
                    words.push(l.letter(c.chars().next().unwrap()));
                }
            } else if let Some(c) = m.get(text, 4).filter(|s| !s.is_empty()) {
                if c.chars().count() == 1 {
                    words.push(l.letter(c.chars().next().unwrap()));
                } else {
                    words.push(c);
                }
                if let Some(n) = m.get(text, 5).filter(|s| !s.is_empty()) {
                    words.push(l.cardinal(py_int(&n)?));
                }
            }
            out.push((m.start(), m.end(), words.join(" ")));
        }
        Ok(out)
    }

    fn m_acronym_model(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"\b([A-Z]{2,4})\s+(\d{1,3})(?![\d.])").find_iter(text) {
            let tok = m.str(text, 1);
            if t::ACRONYM_WHITELIST.contains(&tok.as_str())
                || t::UNIT_INITIALISMS.contains(&tok.to_uppercase().as_str())
            {
                continue;
            }
            let low = tok.to_lowercase();
            if self.id_keywords.contains(&low) || self.label_words.contains(&low) {
                continue;
            }
            let rep = format!(
                "{} {}",
                self.letters_spelled(&tok),
                self.lang.cardinal(py_int(&m.str(text, 2))?)
            );
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_unit_initialism(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in self.unit_initialism.find_iter(text) {
            let unit = m.str(text, 3);
            if !t::UNIT_INITIALISMS.contains(&unit.to_uppercase().as_str()) {
                continue;
            }
            if unit.chars().count() == 1 && !py_isupper(&unit) {
                continue;
            }
            let mut num = self.number_words(&m.str(text, 2))?;
            if !m.str(text, 1).is_empty() {
                num = format!("{} {num}", l.sym("minus"));
            }
            let uk: String = unit
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_uppercase();
            let word = l
                .unit_words
                .get(&uk)
                .filter(|w| !w.is_empty())
                .map_or_else(|| self.letters_spelled(&uk), str::to_string);
            let mut rep = l.join_amount(&num, &word);
            if let Some(u2) = m.get(text, 4).filter(|s| !s.is_empty()) {
                let u2 = u2.to_uppercase();
                let w2 = l
                    .unit_words
                    .get(&u2)
                    .filter(|w| !w.is_empty())
                    .map_or_else(|| self.letters_spelled(&u2), str::to_string);
                rep = format!("{rep} {} {w2}", l.sym.get("per").unwrap_or("per"));
            }
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    fn m_num_hyphen_word(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in
            re!(r"\b(\d+(?:,\d{2,3})*(?:\.\d+)?)-([A-Za-z\x{0900}-\x{0D7F}]{2,})\b").find_iter(text)
        {
            let word = m.str(text, 2);
            if self.month_index(&word).is_some()
                || t::UNIT_INITIALISMS.contains(&word.to_uppercase().as_str())
            {
                continue;
            }
            out.push((
                m.start(),
                m.end(),
                format!("{} {word}", self.number_words(&m.str(text, 1))?),
            ));
        }
        Ok(out)
    }

    fn m_decade(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(19|20)(\d)0s\b").find_iter(text) {
            let rep = if m.str(text, 1) == "19" {
                l.decade(py_int(&m.str(text, 2))? * 10, true)
            } else {
                let y: String = m.str(text, 0).chars().take(4).collect();
                fmt(&l.decade_full, &[("tens", &l.year(py_int(&y)?))])
            };
            out.push((m.start(), m.end(), rep));
        }
        for m in re!(r"(?<![\w])'?(\d)0s\b").find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                l.decade(py_int(&m.str(text, 1))? * 10, false),
            ));
        }
        Ok(out)
    }

    fn m_year(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b(1[5-9]\d{2}|20\d{2})\b").find_iter(text) {
            let ctx = self.keyword_before(text, m.start(), &self.year_words, 2);
            let after = sl(text, m.end(), m.end() + 20);
            let suffixed = re!(r"^-?(?![\x{0964}\x{0965}])[\x{0900}-\x{0DFF}]")
                .match_at(&after, 0)
                .is_some();
            let after_s = st(&after);
            let next_tok = py_split(&after_s).first().map_or("", |s| *s).to_string();
            let follower = t::YEAR_FOLLOWERS.contains(&next_tok.as_str())
                || l.year_words
                    .iter()
                    .any(|w| !w.is_empty() && next_tok.starts_with(w.as_str()));
            if ctx || suffixed || follower {
                out.push((m.start(), m.end(), l.year(py_int(&m.str(text, 0))?)));
            }
        }
        Ok(out)
    }

    fn m_quoted_letter(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r#"['"‘“]([A-Za-z])['"’”]"#).find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                self.lang.letter(m.str(text, 1).chars().next().unwrap()),
            ));
        }
        Ok(out)
    }

    fn m_short_mixed_token(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        let pat = re!(concat!(
            r"(?<![\d.])\b(?=[A-Za-z0-9]{2,5}\b)",
            r"(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])",
            r"[A-Za-z0-9]+\b"
        ));
        for m in pat.find_iter(text) {
            let tok = m.str(text, 0);
            if re!(r"\d+(?:st|nd|rd|th|s)")
                .fullmatch(&chars(&tok))
                .is_some()
            {
                continue;
            }
            let mut words = Vec::new();
            for run in re!(r"[A-Za-z]+|\d+").findall(&chars(&tok)) {
                if py_isdigit(&run) {
                    words.push(self.lang.cardinal(py_int(&run)?));
                } else {
                    words.push(self.letters_spelled(&run.to_uppercase()));
                }
            }
            out.push((m.start(), m.end(), words.join(" ")));
        }
        Ok(out)
    }

    fn m_decimal(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d+)(?!\.?\d)").find_iter(text) {
            let ip = py_int(&m.str(text, 1).replace(',', ""))?;
            out.push((m.start(), m.end(), self.lang.decimal(ip, &m.str(text, 2))?));
        }
        Ok(out)
    }

    fn m_negative(&self, text: &[char]) -> R<Vec<Span>> {
        let mut out = Vec::new();
        for m in re!(r"(?<![\w.,-])-(\d+(?:\.\d+)?)").find_iter(text) {
            out.push((
                m.start(),
                m.end(),
                format!(
                    "{} {}",
                    self.lang.sym("minus"),
                    self.number_words(&m.str(text, 1))?
                ),
            ));
        }
        Ok(out)
    }

    fn m_int(&self, text: &[char]) -> R<Vec<Span>> {
        let l = self.lang;
        let mut out = Vec::new();
        for m in re!(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d+\b").find_iter(text) {
            let digits = m.str(text, 0).replace(',', "");
            let after = st(&sl(text, m.end(), m.end() + 30));
            let after = py_strip(&after).to_lowercase();
            let street = py_split(&after)
                .iter()
                .take(2)
                .any(|w| t::STREET_WORDS.contains(&w.trim_matches(['.', ','])));
            let rep = if digits.chars().count() == 4 && !digits.starts_with('0') && street {
                l.year(py_int(&digits)?)
            } else if digits.starts_with('0') && digits.chars().count() > 1 {
                self.digits_spelled(&digits, true)?
            } else {
                l.cardinal(py_int(&digits)?)
            };
            out.push((m.start(), m.end(), rep));
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Final sweep / digit fallback / normalize
    // ------------------------------------------------------------------

    fn final_sweep(&self, text: &[char]) -> Vec<char> {
        let s = |k| self.lang.sym(k);
        let mut text = re!(r"<=|≤").sub(text, " less than or equal to ");
        text = re!(r">=|≥").sub(&text, " greater than or equal to ");
        text = re!(r"°\s*C\b").sub(&text, &format!(" {} ", s("celsius")));
        text = re!(r"°\s*F\b").sub(&text, &format!(" {} ", s("fahrenheit")));
        let mut out = st(&text).replace('°', &format!(" {} ", s("degree")));
        for (sym, key) in [
            ("%", "percent"),
            ("@", "at"),
            ("&", "amp"),
            ("=", "equals"),
            ("+", "plus"),
            ("×", "times"),
            ("÷", "divide"),
            ("±", "plus_minus"),
            ("√", "sqrt"),
            ("_", "underscore"),
            ("#", "hash"),
        ] {
            out = out.replace(sym, &format!(" {} ", s(key)));
        }
        out = out
            .replace('<', &format!(" {} ", s("lt")))
            .replace('>', &format!(" {} ", s("gt")));
        let mut text = re!(r"[\^*]").sub(&chars(&out), " ");
        text = re!(r"\s*/\s*").sub(&text, " ");
        text = re!(r"\s*:\s*").sub(&text, ", ");
        text = re!(r"[₹$€£](?!\s?\d)").sub(&text, " ");
        re!(r"[–—]").sub(&text, " ")
    }

    fn digit_fallback(&self, text: &[char]) -> R<Vec<char>> {
        let mut err = None;
        let out = re!(r"\d+").sub_with(text, |m| {
            let d = m.str(text, 0);
            let r = if d.chars().count() >= 7 || d.starts_with('0') {
                self.digits_spelled(&d, true)
            } else {
                py_int(&d).map(|n| self.lang.cardinal(n))
            };
            r.unwrap_or_else(|e| {
                err.get_or_insert(e);
                String::new()
            })
        });
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    /// `Normalizer.normalize(text)`.
    fn normalize(&self, text: &[char]) -> R<String> {
        self.validate(text)?;
        let mut raw = ascii_digits(text);
        let mut protected: Vec<(char, String)> = Vec::new();
        let mut originals: Vec<(char, String)> = Vec::new();
        let mut spans: Vec<(usize, usize, char)> = Vec::new();
        for matcher in [
            Self::m_email as fn(&Self, &[char]) -> R<Vec<Span>>,
            Self::m_url,
        ] {
            for (s, e, spoken) in matcher(self, &raw)? {
                if !spans.iter().any(|&(a, b, _)| s < b && e > a) {
                    let key =
                        char::from_u32(0xF0000 + protected.len() as u32).unwrap_or('\u{10fffd}');
                    protected.push((key, spoken));
                    originals.push((key, sl(&raw, s, e).iter().collect()));
                    spans.push((s, e, key));
                }
            }
        }
        spans.sort_by(|a, b| b.cmp(a));
        for (s, e, key) in &spans {
            let mut next = sl(&raw, 0, *s);
            next.push(*key);
            next.extend(sl(&raw, *e, raw.len()));
            raw = next;
        }
        let text = self.preclean(&raw);
        let mut matches = match self.gather(&text) {
            Ok(m) => m,
            Err(Fail::Norm {
                code,
                span: Some((start, end)),
                ..
            }) => {
                let mut pieces = [
                    st(&sl(&text, 0, start)),
                    st(&sl(&text, start, end)),
                    st(&sl(&text, end, text.len())),
                ];
                for (key, original) in &originals {
                    for p in pieces.iter_mut() {
                        *p = p.replace(*key, original);
                    }
                }
                let (a, b) = (pieces[0].chars().count(), pieces[1].chars().count());
                return Err(Fail::Norm {
                    code,
                    span: Some((a, a + b)),
                    source: Some(chars(&pieces.concat())),
                });
            }
            Err(e) => return Err(e),
        };
        matches.sort_by_key(|(prio, (s, e, _))| (*prio, *s, std::cmp::Reverse(*e - *s)));
        let mut taken: Vec<(usize, usize)> = Vec::new();
        let mut accepted: Vec<Span> = Vec::new();
        for (_, (s, e, rep)) in matches {
            if taken.iter().any(|&(ts, te)| !(e <= ts || s >= te)) {
                continue;
            }
            taken.push((s, e));
            accepted.push((s, e, rep));
        }
        accepted.sort_by_key(|(s, _, _)| *s);
        let mut out: Vec<char> = Vec::new();
        let mut pos = 0usize;
        let next_pat = re!(r"[ \t]*-?([^\s\d,\x{0964}.!?;:()\-—–]{1,14})");
        for (s, e, mut rep) in accepted {
            out.extend(sl(&text, pos, s));
            pos = e;
            loop {
                let rest = sl(&text, pos, text.len());
                let Some(nxt) = next_pat.match_at(&rest, 0) else {
                    break;
                };
                if rest.get(nxt.end()) == Some(&'.') {
                    break;
                }
                match self.lang.attach_postposition(&rep, &nxt.str(&rest, 1)) {
                    Some(merged) => {
                        rep = merged;
                        pos += nxt.end();
                    }
                    None => break,
                }
            }
            out.extend(rep.chars());
        }
        out.extend(sl(&text, pos, text.len()));
        let result = self.final_sweep(&out);
        let result = self.digit_fallback(&result)?;
        let mut result = st(&result);
        for (key, spoken) in &protected {
            result = result.replace(*key, spoken);
        }
        let mut r = chars(&result);
        r = re!(r"(?<=[A-Za-z])(?=[\x{0900}-\x{0DFF}])").sub(&r, " ");
        r = re!(r"\s+([,\x{0964}.!?;])").sub_with(&r, |m| m.str(&r, 1));
        r = re!(r",\s*([\x{0964}.!?;,])").sub_with(&r, |m| m.str(&r, 1));
        r = re!(r",(?=\S)").sub(&r, ", ");
        r = re!(r"\s{2,}").sub(&r, " ");
        r = re!(r"\s+’").sub(&r, "’");
        if py_strip_end(&st(&r)).ends_with(',') {
            r = re!(r"[,\s]+$").sub(&r, ".");
        }
        if self.lang.code == "hi" {
            r = r
                .into_iter()
                .filter_map(
                    |c| match t::HINDI_NUKTA_FOLD.iter().find(|(k, _)| *k == c) {
                        Some((_, to)) => *to,
                        None => Some(c),
                    },
                )
                .collect();
        }
        Ok(py_strip(&st(&r)).to_string())
    }
}

fn py_strip_end(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `str.isdigit()` of a whole string.
pub(super) fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit)
}

fn py_isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_alpha)
}

/// `str.isupper()`: a cased character, and none lower-case.
fn py_isupper(s: &str) -> bool {
    s.chars().any(|c| c.is_uppercase()) && !s.chars().any(|c| c.is_lowercase())
}

fn chem_ok(tok: &str) -> bool {
    if t::CHEM_WHITELIST.contains(&tok) {
        return true;
    }
    let pieces = re!(r"[A-Z][a-z]?|\d+").findall(&chars(tok));
    let letters: Vec<&String> = pieces.iter().filter(|p| !py_isdigit(p)).collect();
    let digits = pieces.iter().any(|p| py_isdigit(p));
    if letters.is_empty() || letters.iter().any(|p| !t::ELEMENTS.contains(&p.as_str())) {
        return false;
    }
    (digits && letters.len() >= 2) || (digits && t::CHEM_WHITELIST.contains(&tok))
}

fn roman_value(s: &str) -> u128 {
    let val = |c: char| {
        t::ROMAN_VALS
            .iter()
            .find(|(k, _)| k.starts_with(c))
            .map_or(0, |(_, v)| *v as i64)
    };
    let cs: Vec<char> = s.chars().collect();
    let mut total: i64 = 0;
    for (i, &c) in cs.iter().enumerate() {
        let v = val(c);
        if i + 1 < cs.len() && val(cs[i + 1]) > v {
            total -= v;
        } else {
            total += v;
        }
    }
    total.max(0) as u128
}

/// Module-level `normalize(text, language)`: validation errors with a span
/// are re-read character by character and the text tried again.
pub(super) fn normalize(text: &str, language: &str) -> R<String> {
    let code =
        resolve_language(language).ok_or(Fail::Crash("unsupported_normalization_language"))?;
    let engine = engine(code).ok_or(Fail::Crash("unsupported_normalization_language"))?;
    let locale = engine.lang;
    let mut text = chars(text);
    let mut warning: Option<&'static str> = None;
    // Every retry replaces at least one rejected character; the cap only
    // guards against a pathological loop (upstream would spin).
    for _ in 0..(text.len() + 64) {
        let attempt = engine.validate(&text).and_then(|_| engine.normalize(&text));
        match attempt {
            Ok(result) => {
                if let Some(code) = warning {
                    tracing::debug!(code, "maya2 text normaliser fell back on a span");
                }
                return Ok(result);
            }
            Err(Fail::Norm {
                code,
                span: Some((start, end)),
                source,
            }) => {
                warning.get_or_insert(code);
                let source = source.unwrap_or_else(|| text.clone());
                let token = sl(&source, start, end);
                let mut replacement = String::new();
                for &c in &token {
                    if let Some(d) = decimal_value(c) {
                        replacement += &format!(" {} ", locale.digit(d));
                    } else if let Some(key) = match c {
                        '/' => Some("slash"),
                        '.' => Some("dot"),
                        '-' => Some("minus"),
                        '+' => Some("plus"),
                        '@' => Some("at"),
                        '_' => Some("underscore"),
                        '₹' => Some("rupee"),
                        '$' => Some("dollar"),
                        '%' => Some("percent"),
                        _ => None,
                    } {
                        replacement += &format!(" {} ", locale.sym(key));
                    } else if ",:;<>{}".contains(c) || is_control_or_private(c) {
                        replacement.push(' ');
                    } else {
                        replacement.push(c);
                    }
                }
                if replacement == st(&token) {
                    return Err(Fail::Norm {
                        code,
                        span: Some((start, end)),
                        source: Some(source),
                    });
                }
                let mut next = sl(&source, 0, start);
                next.push(' ');
                next.extend(replacement.chars());
                next.push(' ');
                next.extend(sl(&source, end, source.len()));
                text = next;
            }
            Err(other) => return Err(other),
        }
    }
    Err(Fail::Crash("normaliser fallback did not converge"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_language_engine_compiles() {
        for code in super::super::lang::CODES {
            assert!(engine(code).is_some(), "{code}");
        }
    }
}

//! The language classes of upstream's normaliser (`tn/normalize.py`
//! `LangBase` / `English`, `tn/indic_languages.py`): number, year, ordinal,
//! clock and amount composition per language, plus the postposition sandhi
//! hooks, over the tables in `tables.rs`.
//!
//! The tables are built as `normalise_text(..., rules=True)` sees them: after
//! `rules.install` (R9: a paise word for bn en gu kn ml mr pa ta) and
//! `lexicon.install` (L1: the Assamese lexicon; L2: the language's "per" word
//! and hour / second unit words).

use std::sync::OnceLock;

use super::tables::{self as t, LangData};
use super::{py_int_char, Fail, R};

/// A Python `dict[str, str]` (insertion-ordered).
#[derive(Clone, Debug, Default)]
pub(super) struct OMap(pub Vec<(String, String)>);

impl OMap {
    fn from(pairs: &[(&str, &str)]) -> Self {
        let mut map = OMap::default();
        for (k, v) in pairs {
            map.insert(k, v);
        }
        map
    }

    pub(super) fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub(super) fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// `d[key] = value` (an existing key keeps its position).
    pub(super) fn insert(&mut self, key: &str, value: &str) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.to_string(),
            None => self.0.push((key.to_string(), value.to_string())),
        }
    }

    pub(super) fn setdefault(&mut self, key: &str, value: &str) {
        if !self.contains(key) {
            self.insert(key, value);
        }
    }

    pub(super) fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(k, _)| k.as_str())
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

fn owned(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// `s[:-1]` (one code point).
pub(super) fn drop_last(s: &str) -> &str {
    match s.char_indices().last() {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `s[1:]` (one code point).
fn drop_first(s: &str) -> &str {
    let mut it = s.chars();
    it.next();
    it.as_str()
}

/// `str.format(**kw)` / `_fmt` on the tables' `{name}` templates.
pub(super) fn fmt(template: &str, kw: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in kw {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

/// One language table (a `LangBase` instance) with its patches applied.
pub(super) struct Lang {
    pub code: &'static str,
    pub ones: Vec<String>,
    pub hundred: String,
    pub thousand: String,
    pub lakh: String,
    pub crore: String,
    pub letters: OMap,
    pub months: Vec<String>,
    pub month_aliases: Vec<(String, u32)>,
    pub sym: OMap,
    pub time_on_hour: String,
    pub time_hm: String,
    pub range: String,
    pub range_to: String,
    pub ratio: String,
    pub power: String,
    pub fraction: String,
    pub decade_short: String,
    pub decade_full: String,
    pub periods: OMap,
    pub time_markers: Vec<String>,
    pub marker_suffixes: OMap,
    pub period_detect: OMap,
    pub year_words: Vec<String>,
    pub id_keywords: Vec<String>,
    pub id_after_words: Vec<String>,
    pub label_words: Vec<String>,
    pub minute_words: Vec<String>,
    pub scale_words: OMap,
    pub currency_words: OMap,
    pub ordinal_exceptions: Vec<(u128, String)>,
    pub ordinal_suffix: String,
    pub ordinal_input_suffixes: Vec<String>,
    pub unit_words: OMap,
    pub abbrev_units: OMap,
    pub number_word: String,
    pub ampm_words: OMap,
    pub count_one: Option<String>,
    pub singular_nouns: OMap,
}

/// The normaliser's language codes (`LANGUAGE_NAMES` keys, English as `en`).
pub(super) const CODES: [&str; 12] = [
    "en", "hi", "bn", "as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te",
];

/// `resolve_language`: a code, `<code>-IN`, an English name, or one of the
/// documented aliases (case-insensitive) → the table's code.
pub(super) fn resolve_language(language: &str) -> Option<&'static str> {
    let l = language.to_lowercase();
    for (code, name) in t::LANGUAGE_NAMES {
        let suffixed = if *code == "en-IN" {
            code.to_string()
        } else {
            format!("{code}-IN")
        };
        if l == code.to_lowercase() || l == name.to_lowercase() || l == suffixed.to_lowercase() {
            return Some(if *code == "en-IN" {
                "en"
            } else {
                CODES.iter().find(|c| *c == code)?
            });
        }
    }
    match l.as_str() {
        "en" | "english" | "en-us" | "en-gb" | "indian-english" => Some("en"),
        "oriya" | "od" => Some("or"),
        _ => None,
    }
}

/// `language_table(language)` with rules and lexicon installed.
pub(super) fn language_table(language: &str) -> Option<&'static Lang> {
    static TABLES: OnceLock<Vec<Lang>> = OnceLock::new();
    let code = resolve_language(language)?;
    let tables = TABLES.get_or_init(|| CODES.iter().map(|c| Lang::build(c)).collect());
    tables.iter().find(|l| l.code == code)
}

fn data(code: &str) -> &'static LangData {
    match code {
        "en" => &t::EN,
        "hi" => &t::HI,
        "bn" => &t::BN,
        "as" => &t::AS,
        "gu" => &t::GU,
        "kn" => &t::KN,
        "ml" => &t::ML,
        "mr" => &t::MR,
        "or" => &t::OR,
        "pa" => &t::PA,
        "ta" => &t::TA,
        _ => &t::TE,
    }
}

/// `normalize.py` `_two_digit`: 0..99 in English words, no hyphen.
fn en_two_digit(n: usize) -> String {
    if n < 20 {
        return t::ONES_0_19[n].to_string();
    }
    let (tens, o) = (n / 10, n % 10);
    if o == 0 {
        t::TENS[tens].to_string()
    } else {
        format!("{} {}", t::TENS[tens], t::ONES_0_19[o])
    }
}

fn en_irregular(n: usize) -> Option<&'static str> {
    t::ORDINAL_IRREGULAR
        .iter()
        .find(|(k, _)| *k as usize == n)
        .map(|(_, v)| *v)
}

/// `normalize.py` `_ordinal` for 0..99.
fn en_ordinal(n: usize) -> String {
    if let Some(w) = en_irregular(n) {
        return w.to_string();
    }
    let w = en_two_digit(n);
    let tail = n % 10;
    if n >= 20 {
        if let Some(irr) = en_irregular(tail) {
            return format!("{} {irr}", t::TENS[n / 10]);
        }
    }
    if let Some(stem) = w.strip_suffix('y') {
        return format!("{stem}ieth");
    }
    w + "th"
}

fn as_script(s: &str) -> String {
    s.replace(t::LEX_R, t::LEX_RR)
}

impl Lang {
    fn build(code: &'static str) -> Lang {
        let d = data(code);
        debug_assert_eq!(d.code, code);
        let mut lang = Lang {
            code,
            ones: owned(d.ones),
            hundred: d.hundred.into(),
            thousand: d.thousand.into(),
            lakh: d.lakh.into(),
            crore: d.crore.into(),
            letters: OMap::from(d.letters),
            months: owned(d.months),
            month_aliases: d
                .month_aliases
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            sym: OMap::from(d.sym),
            time_on_hour: d.time_on_hour.into(),
            time_hm: d.time_hm.into(),
            range: d.range.into(),
            range_to: d.range_to.into(),
            ratio: d.ratio.into(),
            power: d.power.into(),
            fraction: d.fraction.into(),
            decade_short: d.decade_short.into(),
            decade_full: d.decade_full.into(),
            periods: OMap::from(d.periods),
            time_markers: owned(d.time_markers),
            marker_suffixes: OMap::from(d.marker_suffixes),
            period_detect: OMap::from(d.period_detect),
            year_words: owned(d.year_words),
            id_keywords: owned(d.id_keywords),
            id_after_words: owned(d.id_after_words),
            label_words: owned(d.label_words),
            minute_words: owned(d.minute_words),
            scale_words: OMap::from(d.scale_words),
            currency_words: OMap::from(d.currency_words),
            ordinal_exceptions: d
                .ordinal_exceptions
                .iter()
                .map(|(k, v)| (*k as u128, v.to_string()))
                .collect(),
            ordinal_suffix: d.ordinal_suffix.into(),
            ordinal_input_suffixes: owned(d.ordinal_input_suffixes),
            unit_words: OMap::from(d.unit_words),
            abbrev_units: OMap::from(d.abbrev_units),
            number_word: d.number_word.into(),
            ampm_words: OMap::from(d.ampm_words),
            count_one: d.count_one.map(str::to_string),
            singular_nouns: OMap::from(d.singular_nouns),
        };
        if code == "en" {
            // ONES = [_two_digit(i) for i in range(100)]; LETTERS A..Z -> itself;
            // ORDINAL_EXCEPTIONS = {n: _ordinal(n) for n in range(1, 100)}
            lang.ones = (0..100).map(en_two_digit).collect();
            for c in 'A'..='Z' {
                lang.letters.insert(&c.to_string(), &c.to_string());
            }
            lang.ordinal_exceptions = (1..100).map(|n| (n as u128, en_ordinal(n))).collect();
        }
        // rules.install(True), R9: the paise word.
        if let Some((_, word)) = t::PAISE.iter().find(|(c, _)| *c == code) {
            lang.sym.setdefault("paisa", word);
        }
        // lexicon.install(True)
        if code == "as" {
            // L1: Assamese ones with ৰ, two malformed entries corrected
            let mut ones: Vec<String> = t::AS.ones.iter().map(|w| as_script(w)).collect();
            ones[45] = t::AS_ONES_FIX_45.into();
            ones[71] = t::AS_ONES_FIX_71.into();
            lang.ones = ones;
            lang.ordinal_exceptions = t::AS_PATCH_ORDINAL_EXCEPTIONS
                .iter()
                .map(|(k, v)| (*k as u128, v.to_string()))
                .collect();
            let mut letters = OMap::default();
            for (k, v) in lang.letters.iter() {
                letters.insert(k, &as_script(v));
            }
            for (k, v) in t::AS_PATCH_LETTERS {
                letters.insert(k, v);
            }
            lang.letters = letters;
            let scripted: Vec<(String, String)> = lang
                .sym
                .iter()
                .filter(|(_, v)| as_script(v) != *v)
                .map(|(k, v)| (k.to_string(), as_script(v)))
                .collect();
            for (k, v) in scripted {
                lang.sym.insert(&k, &v);
            }
            for (k, v) in t::AS_SYM {
                lang.sym.insert(k, v);
            }
            lang.periods = OMap::from(t::AS_PATCH_PERIODS);
            for (k, v) in t::AS_PERIOD_DETECT {
                lang.period_detect.insert(k, v);
            }
            lang.time_on_hour = t::AS_PATCH_TIME_ON_HOUR.into();
            lang.time_markers = owned(t::AS_PATCH_TIME_MARKERS);
            lang.marker_suffixes = OMap::from(t::AS_PATCH_MARKER_SUFFIXES);
            lang.power = t::AS_PATCH_POWER.into();
            lang.decade_short = t::AS_PATCH_DECADE_SHORT.into();
            lang.decade_full = t::AS_PATCH_DECADE_FULL.into();
            lang.abbrev_units = OMap::from(t::AS_PATCH_ABBREV_UNITS);
        }
        // L2: the language's "per" word, and hour / second as per-unit denominators
        if let Some((_, word)) = t::PER.iter().find(|(c, _)| *c == code) {
            lang.sym.setdefault("per", word);
        }
        if let Some((_, words)) = t::PER_UNITS.iter().find(|(c, _)| *c == code) {
            for (k, w) in words.iter() {
                lang.unit_words.setdefault(k, w);
            }
        }
        lang
    }

    /// `SYM[key]` (every table has the base keys; a missing one reads "").
    pub(super) fn sym(&self, key: &str) -> &str {
        self.sym.get(key).unwrap_or("")
    }

    pub(super) fn digit(&self, d: u32) -> &str {
        self.ones.get(d as usize).map_or("", String::as_str)
    }

    pub(super) fn two_digit(&self, n: u128) -> &str {
        self.ones.get(n as usize).map_or("", String::as_str)
    }

    fn exception(&self, n: u128) -> Option<&str> {
        self.ordinal_exceptions
            .iter()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| v.as_str())
    }

    fn hundreds(&self, h: u128, rest: u128) -> String {
        let hu = h as usize;
        let tail = |s: String| {
            if rest != 0 {
                format!("{s} {}", self.two_digit(rest))
            } else {
                s
            }
        };
        match self.code {
            "bn" => tail(format!("{}{}", self.two_digit(h), self.hundred)),
            "as" => tail(t::AS_HUNDREDS.get(hu).copied().unwrap_or("").to_string()),
            "gu" => tail(lookup(t::GU_HUNDREDS_FUSED, h).to_string()),
            "kn" => {
                if rest != 0 {
                    format!(
                        "{} {}",
                        t::KN_HUNDREDS_COMB.get(hu).copied().unwrap_or(""),
                        self.two_digit(rest)
                    )
                } else {
                    t::KN_HUNDREDS.get(hu).copied().unwrap_or("").to_string()
                }
            }
            "ml" => {
                if rest != 0 {
                    format!("{} {}", lookup(t::ML_H_COMB, h), self.two_digit(rest))
                } else {
                    lookup(t::ML_H_STAND, h).to_string()
                }
            }
            "mr" => {
                if h == 1 && rest == 0 {
                    return "शंभर".into();
                }
                tail(format!("{}{}", self.two_digit(h), self.hundred))
            }
            "ta" => {
                if rest != 0 {
                    ta_sandhi_join(&[
                        t::TA_HUNDREDS_COMB
                            .get(hu)
                            .copied()
                            .unwrap_or("")
                            .to_string(),
                        self.two_digit(rest).to_string(),
                    ])
                } else {
                    t::TA_HUNDREDS.get(hu).copied().unwrap_or("").to_string()
                }
            }
            "te" => {
                if h == 1 {
                    if rest != 0 {
                        format!("నూట {}", self.two_digit(rest))
                    } else {
                        "వంద".into()
                    }
                } else if rest != 0 {
                    format!("{} వందల {}", self.two_digit(h), self.two_digit(rest))
                } else {
                    format!("{} వందలు", self.two_digit(h))
                }
            }
            _ => tail(format!("{} {}", self.two_digit(h), self.hundred)),
        }
    }

    fn under_thousand(&self, n: u128) -> String {
        if n < 100 {
            return self.two_digit(n).to_string();
        }
        self.hundreds(n / 100, n % 100)
    }

    /// `cardinal(n)` for n >= 0 (Indian system).
    pub(super) fn cardinal(&self, n: u128) -> String {
        match self.code {
            "kn" => self.kn_cardinal(n),
            "ml" => self.ml_cardinal(n),
            "ta" => self.ta_cardinal(n),
            "te" => self.te_cardinal(n),
            _ => {
                if n < 1000 {
                    return self.under_thousand(n);
                }
                let (crore, rest) = (n / 10_000_000, n % 10_000_000);
                let (lakh, rest) = (rest / 100_000, rest % 100_000);
                let (thousand, rest) = (rest / 1000, rest % 1000);
                let mut parts = Vec::new();
                if crore != 0 {
                    parts.push(format!("{} {}", self.cardinal(crore), self.crore));
                }
                if lakh != 0 {
                    parts.push(format!("{} {}", self.two_digit(lakh), self.lakh));
                }
                if thousand != 0 {
                    parts.push(format!("{} {}", self.two_digit(thousand), self.thousand));
                }
                if rest != 0 {
                    parts.push(self.under_thousand(rest));
                }
                parts.join(" ")
            }
        }
    }

    fn kn_cardinal(&self, n: u128) -> String {
        if n < 1000 {
            return self.under_thousand(n);
        }
        let (crore, rest) = (n / 10_000_000, n % 10_000_000);
        let (lakh, rest) = (rest / 100_000, rest % 100_000);
        let (thousand, rest) = (rest / 1000, rest % 1000);
        let mut parts = Vec::new();
        if crore != 0 {
            parts.push(format!("{} {}", self.cardinal(crore), self.crore));
        }
        if lakh != 0 {
            let word = if thousand != 0 || rest != 0 {
                t::KN_LAKH_COMB
            } else {
                self.lakh.as_str()
            };
            parts.push(format!("{} {word}", self.two_digit(lakh)));
        }
        if thousand != 0 {
            let word = if rest != 0 {
                t::KN_THOUSAND_COMB
            } else {
                self.thousand.as_str()
            };
            if thousand == 1 && crore == 0 && lakh == 0 {
                parts.push(word.to_string());
            } else {
                parts.push(format!("{} {word}", self.under_thousand(thousand)));
            }
        }
        if rest != 0 {
            parts.push(self.under_thousand(rest));
        }
        parts.join(" ")
    }

    fn ml_thousand(&self, m: u128, combining: bool) -> String {
        let word = match t::ML_TH_STAND.iter().find(|(k, _)| *k as u128 == m) {
            Some((_, w)) => w.to_string(),
            None => {
                let base = self.two_digit(m);
                if base.ends_with(&format!("ത{}", t::ML_VIRAMA)) {
                    format!("{}ിനായിരം", drop_last(base))
                } else if base.ends_with(t::ML_VIRAMA) {
                    format!("{}ായിരം", drop_last(base))
                } else {
                    format!("{base} {}", self.thousand)
                }
            }
        };
        if combining {
            format!("{}ത്തി", drop_last(&word))
        } else {
            word
        }
    }

    fn ml_cardinal(&self, n: u128) -> String {
        if n < 100 {
            return self.two_digit(n).to_string();
        }
        if n < 1000 {
            return self.hundreds(n / 100, n % 100);
        }
        let (crore, rest) = (n / 10_000_000, n % 10_000_000);
        let (lakh, rest) = (rest / 100_000, rest % 100_000);
        let (thousand, rest) = (rest / 1000, rest % 1000);
        let mut parts = Vec::new();
        if crore != 0 {
            let mult = if crore == 1 {
                t::ML_ONE_MULT.to_string()
            } else {
                self.cardinal(crore)
            };
            parts.push(format!("{mult} {}", self.crore));
        }
        if lakh != 0 {
            let word = if thousand != 0 || rest != 0 {
                format!("{}ത്തി", drop_last(&self.lakh))
            } else {
                self.lakh.clone()
            };
            let mult = if lakh == 1 {
                t::ML_ONE_MULT
            } else {
                self.two_digit(lakh)
            };
            parts.push(format!("{mult} {word}"));
        }
        if thousand != 0 {
            parts.push(self.ml_thousand(thousand, rest != 0));
        }
        if rest != 0 {
            parts.push(self.under_thousand(rest));
        }
        parts.join(" ")
    }

    fn ta_thousands_word(&self, th: u128, combining: bool) -> String {
        let base = if combining {
            "ஆயிரத்து"
        } else {
            "ஆயிரம்"
        };
        if th == 1 {
            return base.into();
        }
        format!("{}ா{}", drop_last(self.two_digit(th)), drop_first(base))
    }

    fn ta_cardinal(&self, n: u128) -> String {
        if n == 0 {
            return self.two_digit(0).to_string();
        }
        let (crore, rest) = (n / 10_000_000, n % 10_000_000);
        let (lakh, rest) = (rest / 100_000, rest % 100_000);
        let (thousand, low) = (rest / 1000, rest % 1000);
        let mut parts = Vec::new();
        if crore != 0 {
            let mult = if crore == 1 {
                "ஒரு".to_string()
            } else {
                self.cardinal(crore)
            };
            let word = if lakh != 0 || thousand != 0 || low != 0 {
                "கோடியே"
            } else {
                "கோடி"
            };
            parts.push(format!("{mult} {word}"));
        }
        if lakh != 0 {
            let mult = if lakh == 1 {
                "ஒரு"
            } else {
                self.two_digit(lakh)
            };
            let word = if thousand != 0 || low != 0 {
                "லட்சத்து"
            } else {
                "லட்சம்"
            };
            parts.push(format!("{mult} {word}"));
        }
        if thousand != 0 {
            parts.push(self.ta_thousands_word(thousand, low != 0));
        }
        if low != 0 {
            parts.push(self.under_thousand(low));
        }
        ta_sandhi_join(&parts)
    }

    fn te_cardinal(&self, n: u128) -> String {
        if n < 1000 {
            return self.under_thousand(n);
        }
        let (crore, rest) = (n / 10_000_000, n % 10_000_000);
        let (lakh, rest) = (rest / 100_000, rest % 100_000);
        let (thousand, sub) = (rest / 1000, rest % 1000);
        let mut parts = Vec::new();
        if crore != 0 {
            let has_rest = lakh != 0 || thousand != 0 || sub != 0;
            parts.push(if crore == 1 {
                (if has_rest {
                    "కోటీ"
                } else {
                    "కోటి"
                })
                .to_string()
            } else {
                format!(
                    "{} {}",
                    self.cardinal(crore),
                    if has_rest {
                        "కోట్ల"
                    } else {
                        "కోట్లు"
                    }
                )
            });
        }
        if lakh != 0 {
            let has_rest = thousand != 0 || sub != 0;
            parts.push(if lakh == 1 {
                (if has_rest {
                    "లక్షా"
                } else {
                    "లక్ష"
                })
                .to_string()
            } else {
                format!(
                    "{} {}",
                    self.two_digit(lakh),
                    if has_rest {
                        "లక్షల"
                    } else {
                        "లక్షలు"
                    }
                )
            });
        }
        if thousand != 0 {
            parts.push(if thousand == 1 {
                "వెయ్యి".to_string()
            } else {
                format!(
                    "{} {}",
                    self.two_digit(thousand),
                    if sub != 0 {
                        "వేల"
                    } else {
                        "వేలు"
                    }
                )
            });
        }
        if sub != 0 {
            parts.push(self.under_thousand(sub));
        }
        parts.join(" ")
    }

    /// Year-style reading.
    pub(super) fn year(&self, y: u128) -> String {
        let (hi, lo) = (y / 100, y % 100);
        let in_range = (1100..=1999).contains(&y);
        match self.code {
            "as" | "kn" | "ml" | "ta" => self.cardinal(y),
            "bn" | "mr" if in_range => {
                let mut out = format!("{}{}", self.two_digit(hi), self.hundred);
                if lo != 0 {
                    out = format!("{out} {}", self.two_digit(lo));
                }
                out
            }
            "gu" if in_range => {
                let w = self.two_digit(hi);
                let w = match w.strip_suffix('સ') {
                    Some(stem) => format!("{stem}{}", self.hundred),
                    None => format!("{w}{}", self.hundred),
                };
                if lo != 0 {
                    format!("{w} {}", self.two_digit(lo))
                } else {
                    w
                }
            }
            "te" if in_range => {
                if lo != 0 {
                    format!("{} వందల {}", self.two_digit(hi), self.two_digit(lo))
                } else {
                    format!("{} వందలు", self.two_digit(hi))
                }
            }
            "bn" | "mr" | "gu" | "te" => self.cardinal(y),
            _ if in_range => {
                let mut out = format!("{} {}", self.two_digit(hi), self.hundred);
                if lo != 0 {
                    out = format!("{out} {}", self.two_digit(lo));
                }
                out
            }
            _ => self.cardinal(y),
        }
    }

    /// `letter(c)`: `LETTERS.get(c.upper(), c)`.
    pub(super) fn letter(&self, c: char) -> String {
        let up: String = c.to_uppercase().collect();
        self.letters
            .get(&up)
            .map_or_else(|| c.to_string(), str::to_string)
    }

    /// `month(i)` = `MONTHS[i - 1]` (Python indexing: 0 reads the last).
    pub(super) fn month(&self, i: u32) -> R<String> {
        let idx = if i == 0 {
            self.months.len().checked_sub(1)
        } else {
            Some(i as usize - 1)
        };
        idx.and_then(|k| self.months.get(k))
            .cloned()
            .ok_or(Fail::Crash("IndexError: MONTHS"))
    }

    fn base_clock(&self, h: u32, m: u32) -> String {
        let h12 = if !self.ampm_words.0.is_empty() && h > 12 {
            h
        } else if h.is_multiple_of(12) {
            12
        } else {
            h % 12
        };
        if m == 0 {
            fmt(&self.time_on_hour, &[("h", self.two_digit(h12 as u128))])
        } else {
            fmt(
                &self.time_hm,
                &[
                    ("h", self.two_digit(h12 as u128)),
                    ("m", self.two_digit(m as u128)),
                ],
            )
        }
    }

    /// `time_phrase(h, m, period_key)`.
    pub(super) fn time_phrase(&self, h: u32, m: u32, period: Option<&str>) -> String {
        let period_word = |p: &str| self.periods.get(p).unwrap_or("").to_string();
        match self.code {
            "en" => {
                let core = if h == 0 {
                    format!(
                        "zero {}",
                        if m == 0 {
                            "hundred"
                        } else {
                            self.two_digit(m as u128)
                        }
                    )
                } else {
                    self.base_clock(h, m)
                };
                match period {
                    Some(p) => format!("{core} {}", period_word(p)),
                    None => core,
                }
            }
            "te" => {
                let h12 = if h.is_multiple_of(12) { 12 } else { h % 12 };
                let hw = if h12 == 1 {
                    "ఒక"
                } else {
                    self.two_digit(h12 as u128)
                };
                let mut hnoun = if h12 == 1 {
                    "గంట"
                } else if m == 0 {
                    "గంటలు"
                } else {
                    "గంటల"
                };
                if h12 == 1 && m != 0 {
                    hnoun = "గంట";
                }
                let core = if m == 0 {
                    format!("{hw} {hnoun}")
                } else if m == 1 {
                    format!("{hw} {hnoun} ఒక నిమిషం")
                } else {
                    format!("{hw} {hnoun} {} నిమిషాలు", self.two_digit(m as u128))
                };
                match period {
                    Some(p) => format!("{} {core}", period_word(p)),
                    None => core,
                }
            }
            _ => {
                let core = self.base_clock(h, m);
                match period {
                    Some(p) => format!("{} {core}", period_word(p)),
                    None => core,
                }
            }
        }
    }

    /// `ordinal(n, src_suffix)`.
    pub(super) fn ordinal(&self, n: u128, src_suffix: &str) -> String {
        if src_suffix.is_empty() {
            if let Some(w) = self.exception(n) {
                return w.to_string();
            }
        }
        self.ordinal_compose(n, src_suffix)
    }

    fn ordinal_compose(&self, n: u128, src: &str) -> String {
        let exc = self.exception(n).map(str::to_string);
        match self.code {
            "en" => {
                if n < 100 {
                    return if n == 0 {
                        "zeroth".into()
                    } else {
                        en_ordinal(n as usize)
                    };
                }
                for (scale, word) in [
                    (10_000_000u128, &self.crore),
                    (100_000, &self.lakh),
                    (1000, &self.thousand),
                    (100, &self.hundred),
                ] {
                    if n >= scale {
                        let (count, rest) = (n / scale, n % scale);
                        let head = format!("{} {word}", self.cardinal(count));
                        return if rest != 0 {
                            format!("{head} {}", self.ordinal(rest, ""))
                        } else {
                            head + "th"
                        };
                    }
                }
                unreachable!()
            }
            "hi" => {
                if !src.is_empty() {
                    if let Some(w) = exc {
                        let ending = match src {
                            "वीं" => "ी",
                            "वें" | "वे" => "े",
                            _ => "ा",
                        };
                        return format!("{}{ending}", drop_last(&w));
                    }
                    return format!("{}{src}", self.cardinal(n));
                }
                exc.unwrap_or_else(|| format!("{}{}", self.cardinal(n), self.ordinal_suffix))
            }
            "bn" | "as" => {
                if matches!(src, "লা" | "রা" | "ঠা") {
                    if let Some(w) = lookup_opt(t::BN_DATE_ORDINALS, n) {
                        return w.to_string();
                    }
                }
                match src {
                    "ই" => return format!("{}ই", self.cardinal(n)),
                    "শে" => return format!("{}ে", self.cardinal(n)),
                    "তম" => return format!("{}তম", self.cardinal(n)),
                    _ => {}
                }
                exc.unwrap_or_else(|| format!("{}{}", self.cardinal(n), self.ordinal_suffix))
            }
            "gu" => {
                if let Some(stem) = lookup_opt(t::GU_ORDINAL_STEMS, n) {
                    if src.chars().count() > 1 {
                        return format!("{stem}{}", drop_first(src));
                    }
                    return exc.unwrap_or_default();
                }
                if src.starts_with('મ') {
                    return format!("{}{src}", self.cardinal(n));
                }
                format!("{}{}", self.cardinal(n), self.ordinal_suffix)
            }
            "kn" => {
                let base = self.cardinal(n);
                let base = base.strip_suffix('ು').unwrap_or(&base);
                format!("{base}{}", self.input_suffix_or_default(src))
            }
            "ml" => {
                let stem = self.cardinal(n);
                let stem = if stem.ends_with(t::ML_VIRAMA) || stem.ends_with(t::ML_ANUSVARA) {
                    drop_last(&stem).to_string()
                } else {
                    stem
                };
                format!("{stem}{}", self.input_suffix_or_default(src))
            }
            "mr" | "pa" => {
                if !src.is_empty() {
                    return format!("{}{src}", self.cardinal(n));
                }
                exc.unwrap_or_else(|| format!("{}{}", self.cardinal(n), self.ordinal_suffix))
            }
            "ta" => {
                let suffix = if matches!(src, "ஆம்" | "ம்") {
                    "ஆம்"
                } else {
                    "ஆவது"
                };
                if n == 1 {
                    return (if suffix == "ஆம்" {
                        "முதலாம்"
                    } else {
                        "முதலாவது"
                    })
                    .into();
                }
                let base = self.cardinal(n);
                let last = base.chars().last();
                if matches!(last, Some('ு') | Some('்')) {
                    return format!("{}ா{}", drop_last(&base), drop_first(suffix));
                }
                if matches!(last, Some('ி') | Some('ீ') | Some('ை')) {
                    return format!("{base}யா{}", drop_first(suffix));
                }
                base + suffix
            }
            "te" => {
                if src.is_empty() {
                    if let Some(w) = exc {
                        return w;
                    }
                }
                let word = self.cardinal(n);
                let stem = if word.ends_with('ు') || word.ends_with('ి') {
                    drop_last(&word).to_string()
                } else if word.ends_with('ై') {
                    format!("{}య్య", drop_last(&word))
                } else {
                    word
                };
                format!("{stem}{}", self.ordinal_suffix)
            }
            _ => {
                // LangBase._ordinal_compose
                if src.is_empty() {
                    if let Some(w) = exc {
                        return w;
                    }
                }
                format!("{}{}", self.cardinal(n), self.input_suffix_or_default(src))
            }
        }
    }

    fn input_suffix_or_default<'a>(&'a self, src: &'a str) -> &'a str {
        if self.ordinal_input_suffixes.iter().any(|s| s == src) {
            src
        } else {
            &self.ordinal_suffix
        }
    }

    /// Telugu `_oblique_last`: వందలు -> వందల.
    fn te_oblique_last(words: &str) -> &str {
        let last = words.rsplit(' ').next().unwrap_or("");
        if last.ends_with("లు") {
            drop_last(words)
        } else {
            words
        }
    }

    /// `decimal(int_part, frac_digits)`.
    pub(super) fn decimal(&self, int_part: u128, frac: &str) -> R<String> {
        let mut words = Vec::new();
        for c in frac.chars() {
            words.push(self.digit(py_int_char(c)?));
        }
        let head = self.cardinal(int_part);
        let head = if self.code == "te" {
            Self::te_oblique_last(&head).to_string()
        } else {
            head
        };
        Ok(format!("{head} {} {}", self.sym("point"), words.join(" ")))
    }

    /// `decade(tens, full)`.
    pub(super) fn decade(&self, tens: u128, full: bool) -> String {
        let tpl = if full {
            &self.decade_full
        } else {
            &self.decade_short
        };
        let w = self.two_digit(tens);
        let stem = match self.code {
            "kn" => w.strip_suffix('ು').unwrap_or(w).to_string(),
            "ml" => match w.strip_suffix(t::ML_VIRAMA) {
                Some(s) => format!("{s}ു"),
                None => w.to_string(),
            },
            _ => w.to_string(),
        };
        fmt(tpl, &[("tens", &stem)])
    }

    /// `join_amount(words, noun)`.
    pub(super) fn join_amount(&self, words: &str, noun: &str) -> String {
        if self.code == "te" {
            if words == "ఒకటి" {
                return format!(
                    "ఒక {}",
                    lookup_str(t::TE_NOUN_SINGULAR, noun).unwrap_or(noun)
                );
            }
            return format!("{} {noun}", Self::te_oblique_last(words));
        }
        if words == self.two_digit(1) {
            let one = self.count_one.as_deref().unwrap_or(words);
            return format!("{one} {}", self.singular_nouns.get(noun).unwrap_or(noun));
        }
        format!("{words} {noun}")
    }

    /// `scale_noun(canon, numstr)`.
    pub(super) fn scale_noun(&self, canon: &str, numstr: &str) -> String {
        if self.code == "te" {
            let one = numstr.replace(',', "") == "1";
            return match canon {
                "crore" => if one {
                    "కోటి"
                } else {
                    "కోట్లు"
                }
                .into(),
                "lakh" => if one {
                    "లక్ష"
                } else {
                    "లక్షలు"
                }
                .into(),
                "thousand" => if one {
                    "వెయ్యి"
                } else {
                    "వేలు"
                }
                .into(),
                other => other.into(),
            };
        }
        match canon {
            "crore" => self.crore.clone(),
            "lakh" => self.lakh.clone(),
            "thousand" => self.thousand.clone(),
            other => other.into(),
        }
    }

    /// `attach_postposition(rep, tok)`: the merged string, or `None` to leave
    /// the native token after a rendered span alone.
    pub(super) fn attach_postposition(&self, rep: &str, tok: &str) -> Option<String> {
        let last = rep.rsplit(' ').next().unwrap_or("");
        let base = &rep[..rep.len() - last.len()];
        let t = tok.trim_start_matches('-');
        let in_list = |list: &[&str], w: &str| list.contains(&w);
        match self.code {
            "bn" => {
                if !in_list(t::BN_ATTACH_NOUNS, last) {
                    return None;
                }
                if t == last {
                    return Some(rep.into());
                }
                let vowel_final = last.ends_with('া') || last.ends_with('ো') || last.ends_with('ে');
                if matches!(t, "এ" | "তে" | "\u{9af}\u{9bc}") {
                    return Some(if vowel_final {
                        format!("{rep}\u{9af}\u{9bc}")
                    } else {
                        format!("{rep}ে")
                    });
                }
                if matches!(t, "এর" | "র") {
                    return Some(if vowel_final {
                        format!("{rep}র")
                    } else {
                        format!("{rep}ের")
                    });
                }
                if matches!(t, "মধ্যে" | "নাগাদ" | "থেকে" | "পর্যন্ত" | "দি\u{9af}\u{9bc}ে")
                {
                    let g = if vowel_final {
                        format!("{rep}র")
                    } else {
                        format!("{rep}ের")
                    };
                    return Some(format!("{g} {t}"));
                }
                None
            }
            "as" => {
                // lexicon._as_attach
                if !in_list(t::AS_ATTACH, last) || !in_list(t::AS_PP, t) {
                    return None;
                }
                Some(format!("{rep}{t}"))
            }
            "gu" | "or" => {
                let (nouns, pps) = if self.code == "gu" {
                    (t::GU_ATTACH_NOUNS, t::GU_PP)
                } else {
                    (t::OR_ATTACH_NOUNS, t::OR_PP)
                };
                if t == last && in_list(nouns, last) {
                    return Some(rep.into());
                }
                if in_list(pps, t) && in_list(nouns, last) {
                    return Some(format!("{rep}{t}"));
                }
                None
            }
            "kn" => {
                let lc = last.chars().last()?;
                if !('\u{0c80}'..='\u{0cff}').contains(&lc) {
                    return None;
                }
                if t == last {
                    return Some(rep.into());
                }
                if !in_list(t::KN_PP, t) {
                    return None;
                }
                let dative = matches!(t, "ಗೆ" | "ಕ್ಕೆ");
                let ablative = matches!(t, "ಇಂದ" | "ದಿಂದ" | "ರಿಂದ");
                let locative = matches!(t, "ನಲ್ಲಿ" | "ಅಲ್ಲಿ" | "ರಲ್ಲಿ");
                if let Some(stem) = last.strip_suffix('\u{0cc1}') {
                    let suf = if dative {
                        "ಕ್ಕೆ"
                    } else if ablative {
                        "ರಿಂದ"
                    } else if locative {
                        "ರಲ್ಲಿ"
                    } else {
                        t
                    };
                    return Some(format!("{base}{stem}{suf}"));
                }
                if matches!(lc, '\u{0cc6}' | '\u{0cbf}' | '\u{0cc0}') {
                    let suf = if dative {
                        "ಗೆ"
                    } else if ablative {
                        "ಯಿಂದ"
                    } else if locative {
                        "ಯಲ್ಲಿ"
                    } else {
                        t
                    };
                    return Some(format!("{base}{last}{suf}"));
                }
                let suf = if dative {
                    "ಕ್ಕೆ"
                } else if ablative {
                    "ದಿಂದ"
                } else if locative {
                    "ದಲ್ಲಿ"
                } else {
                    t
                };
                Some(format!("{base}{last}{suf}"))
            }
            "ml" => {
                let lc = last.chars().last()?;
                if !('\u{0d00}'..='\u{0d7f}').contains(&lc) {
                    return None;
                }
                let virama = last.ends_with('്');
                let anusvara = last.ends_with('ം');
                let ivow = matches!(lc, 'ി' | 'ീ');
                let stem = drop_last(last);
                if matches!(t, "ന്റെ" | "ുടെ" | "യുടെ") {
                    if virama {
                        return Some(format!("{base}{stem}ിന്റെ"));
                    }
                    if anusvara {
                        return Some(format!("{base}{stem}ത്തിന്റെ"));
                    }
                    return Some(format!("{base}{last}യുടെ"));
                }
                if matches!(t, "ൽ" | "ിൽ" | "ലെ" | "ിലെ") {
                    let end = if matches!(t, "ലെ" | "ിലെ") {
                        "ിലെ"
                    } else {
                        "ിൽ"
                    };
                    if virama {
                        return Some(format!("{base}{stem}{end}"));
                    }
                    if anusvara {
                        return Some(format!("{base}{stem}ത്ത{end}"));
                    }
                    return Some(format!("{base}{last}യ{end}"));
                }
                if matches!(t, "ലേക്ക്" | "ിലേക്ക്") {
                    if virama {
                        return Some(format!("{base}{stem}ിലേക്ക്"));
                    }
                    if anusvara {
                        return Some(format!("{base}{stem}ത്തിലേക്ക്"));
                    }
                    return Some(format!("{base}{last}യിലേക്ക്"));
                }
                if matches!(t, "ന്" | "ിന്" | "ക്ക്" | "ിനാണ്") {
                    let tail = if t == "ിനാണ്" {
                        "ിനാണ്"
                    } else {
                        "ിന്"
                    };
                    if virama {
                        return Some(format!("{base}{stem}{tail}"));
                    }
                    if anusvara {
                        return Some(format!("{base}{stem}ത്ത{tail}"));
                    }
                    if ivow {
                        return Some(format!("{base}{last}ക്ക്"));
                    }
                    return Some(format!("{base}{last}യ്ക്ക്"));
                }
                None
            }
            "mr" => {
                let t = t.trim_start_matches(['ा', 'ं']);
                let case = lookup_str(t::MR_CASE, t)?;
                if !in_list(t::MR_ATTACH_NOUNS, last) {
                    return None;
                }
                let obl = match last.strip_suffix('े') {
                    Some(stem) => format!("{stem}ां"),
                    None => last.to_string(),
                };
                Some(format!("{base}{obl}{case}"))
            }
            "ta" => {
                const NUM_OBL: [(&str, &str); 11] = [
                    ("ஒன்று", "ஒன்ற"),
                    ("இரண்டு", "இரண்ட"),
                    ("மூன்று", "மூன்ற"),
                    ("நான்கு", "நான்க"),
                    ("ஐந்து", "ஐந்த"),
                    ("ஆறு", "ஆற"),
                    ("ஏழு", "ஏழ"),
                    ("எட்டு", "எட்ட"),
                    ("ஒன்பது", "ஒன்பத"),
                    ("பத்து", "பத்த"),
                    ("பூஜ்ஜியம்", "பூஜ்ஜியத்த"),
                ];
                if let Some(obl) = lookup_str(&NUM_OBL, last) {
                    return match t {
                        "இல்" | "ல்" => Some(format!("{base}{obl}ில்")),
                        "க்கு" | "கு" => Some(format!("{base}{obl}ிற்கு")),
                        "இலிருந்து" | "ல்இருந்து" => {
                            Some(format!("{base}{obl}ிலிருந்து"))
                        }
                        _ => None,
                    };
                }
                if !in_list(t::TA_ATTACH_NOUNS, last) {
                    return None;
                }
                if matches!(t, "க்கு" | "கு") {
                    if last == "நிமிடம்" {
                        return Some(format!("{base}நிமிடத்திற்கு"));
                    }
                    if last == "சதவீதம்" {
                        return Some(format!("{base}சதவீதத்திற்கு"));
                    }
                    return Some(format!("{base}{last}க்கு"));
                }
                if matches!(t, "இல்" | "ல்" | "இலிருந்து" | "ல்இருந்து")
                {
                    if let Some(stem) = last.strip_suffix("ம்") {
                        return Some(format!("{base}{stem}த்தில்"));
                    }
                    return Some(format!("{base}{last}இல்"));
                }
                None
            }
            "te" => self.te_attach(rep, tok),
            _ => None,
        }
    }

    fn te_attach(&self, rep: &str, tok: &str) -> Option<String> {
        let lower = tok.to_lowercase();
        let tok = lookup_str(t::TE_PP_ROMAN, &lower).unwrap_or(tok);
        let last = rep.rsplit(' ').next().unwrap_or("");
        let first = tok.chars().next();
        if rep == "ఒకటి" && first.is_some_and(|c| c.is_ascii() && super::pyre::is_alpha(c))
        {
            return Some(format!("ఒక {tok}"));
        }
        let lc = last.chars().last()?;
        if !('\u{0c00}'..='\u{0c7f}').contains(&lc) {
            return None;
        }
        if self.letters.0.iter().any(|(_, v)| v == last) {
            return None;
        }
        if tok == last {
            return Some(rep.into());
        }
        let plural = last.ends_with("లు");
        if t::TE_PP_NOMINATIVE.contains(&tok) {
            return Some(format!("{rep}{tok}"));
        }
        let mut base_tok = tok.replace('\u{200c}', "");
        for end in ["ను", "ని", "కు", "కి", "లో", "గా", "తో", "పై"]
        {
            if let Some(stem) = base_tok.strip_suffix(end) {
                if t::TE_COUNT_NOUNS.contains(&stem) {
                    base_tok = stem.to_string();
                    break;
                }
            }
        }
        if plural && t::TE_COUNT_NOUNS.contains(&base_tok.as_str()) {
            return Some(format!("{} {tok}", drop_last(rep)));
        }
        if rep == "ఒకటి" && t::TE_COUNT_NOUNS.contains(&tok) {
            return Some(format!(
                "ఒక {}",
                lookup_str(t::TE_NOUN_SINGULAR, tok).unwrap_or(tok)
            ));
        }
        if t::TE_PP_SPACED.contains(&tok) {
            return Some(format!(
                "{} {tok}",
                if plural { drop_last(rep) } else { rep }
            ));
        }
        if !t::TE_PP_GLUE.contains(&tok) {
            return None;
        }
        if last == "శాతం" && matches!(tok, "కి" | "కు") {
            return Some(format!("{}ానికి", drop_last(rep)));
        }
        if plural {
            if matches!(tok, "కి" | "కు") {
                return Some(format!("{}{tok}", drop_last(rep)));
            }
            if tok == "ని" {
                return Some(format!("{}ను", drop_last(rep)));
            }
            return Some(format!("{}{tok}", drop_last(rep)));
        }
        Some(format!("{rep}{tok}"))
    }
}

/// Tamil `_sandhi_join`: after -த்து / -ற்று a following க ச த ப doubles.
fn ta_sandhi_join(parts: &[String]) -> String {
    let mut out: Vec<String> = parts.iter().filter(|p| !p.is_empty()).cloned().collect();
    for i in 0..out.len().saturating_sub(1) {
        if out[i].ends_with("த்து") || out[i].ends_with("ற்று") {
            let first = out[i + 1]
                .chars()
                .next()
                .map(String::from)
                .unwrap_or_default();
            if let Some(d) = lookup_str(t::TA_DOUBLE, &first) {
                out[i].push_str(d);
            }
        }
    }
    out.join(" ")
}

fn lookup(table: &[(u32, &'static str)], n: u128) -> &'static str {
    lookup_opt(table, n).unwrap_or("")
}

fn lookup_opt(table: &[(u32, &'static str)], n: u128) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k as u128 == n).map(|(_, v)| *v)
}

pub(super) fn lookup_str<'a>(table: &[(&str, &'a str)], key: &str) -> Option<&'a str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_SYM: [&str; 31] = [
        "rupee",
        "dollar",
        "euro",
        "pound",
        "cent",
        "percent",
        "point",
        "minus",
        "plus",
        "equals",
        "times",
        "divide",
        "squared",
        "sqrt",
        "lt",
        "gt",
        "plus_minus",
        "at",
        "underscore",
        "dot",
        "slash",
        "hash",
        "amp",
        "half",
        "quarter",
        "three_quarters",
        "degree",
        "celsius",
        "fahrenheit",
        "minute",
        "rupee",
    ];

    #[test]
    fn every_table_is_complete() {
        for code in CODES {
            let l = language_table(code).unwrap();
            assert_eq!(l.ones.len(), 100, "{code}");
            assert_eq!(l.months.len(), 12, "{code}");
            assert_eq!(l.letters.0.len(), 26, "{code}");
            for key in BASE_SYM {
                assert!(l.sym.contains(key), "{code} SYM[{key}]");
            }
            for key in ["morning", "afternoon", "evening", "night"] {
                assert!(l.periods.contains(key), "{code} PERIODS[{key}]");
            }
        }
        assert_eq!(resolve_language("Hindi"), Some("hi"));
        assert_eq!(resolve_language("en-IN"), Some("en"));
        assert_eq!(resolve_language("od"), Some("or"));
        assert_eq!(resolve_language("fr"), None);
    }

    #[test]
    fn english_numbers() {
        let en = language_table("en").unwrap();
        assert_eq!(en.cardinal(125_000), "one lakh twenty five thousand");
        assert_eq!(en.cardinal(1_234_567_890), "one hundred twenty three crore forty five lakh sixty seven thousand eight hundred ninety");
        assert_eq!(en.year(1947), "nineteen hundred forty seven");
        assert_eq!(en.ordinal(21, ""), "twenty first");
        assert_eq!(en.ordinal(112, ""), "one hundred twelfth");
        assert_eq!(en.ordinal(100, ""), "one hundredth");
        assert_eq!(en.time_phrase(0, 30, None), "zero thirty");
    }
}

//! `maya3/text/rules.py`: the targeted pre / post rules around the vendored
//! normaliser (R9's paise words are installed in `lang.rs`).
//!
//! Pre (outside e-mail / URL spans, in this order): R10 percent ranges
//! written with two signs, R11 price per unit, R12 "p.a.", Izwi's own
//! English "<number> ms" as milliseconds (not upstream), R7 version
//! strings, R4 spaced card / phone digit groups, R1 dates with a two-digit
//! year, R2 card expiry next to an expiry / date word or a CVV, R3 financial
//! years, R5 a PIN keyword's digits, R6 bare six-digit PIN codes, R8 Tamil
//! "Month D, YYYY". Post: P1 a comma before an inline tag (off upstream:
//! `TAG_COMMA = False`), P2 the dash inside an e-mail / URL in the request
//! language, P3 acronym labels inside e-mails / URLs spelled letter by letter.

use std::collections::HashSet;
use std::sync::OnceLock;

use super::lang::{language_table, Lang};
use super::pyre::{escape, re, re_i, Regex};
use super::tables as t;
use super::tn::alt_escaped;
use super::{chars, py_int, py_int_char, py_split, sl, st, R};

/// P1 is off upstream ("no measured gain").
pub(super) const TAG_COMMA: bool = false;

fn uri_re() -> &'static Regex {
    re!(t::URI_RE)
}

fn expiry_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new_i(&alt_escaped(t::EXPIRY_WORDS)))
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new_i(&alt_escaped(t::DATE_WORDS)))
}

fn pin_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let words: Vec<String> = t::PIN_WORDS.iter().map(|w| escape(w)).collect();
        Regex::new_i(&format!(r"(?<![\w])(?:{}{}", words.join("|"), t::PIN_TAIL))
    })
}

/// `_tn_id_context`: an ID keyword among the last three tokens before a
/// number (the normaliser will then read it digit by digit itself).
fn tn_id_context(l: &Lang, before: &[char]) -> bool {
    let kws: HashSet<String> = t::ID_KEYWORDS_EN
        .iter()
        .chain(t::OTP_INPUT_WORDS.iter())
        .map(|s| s.to_string())
        .chain(l.id_keywords.iter().map(|w| w.to_lowercase()))
        .collect();
    let cleaned: String = st(before).replace('\u{200c}', "").to_lowercase();
    let toks = re!(r"[\w#ऀ-෿']+").findall(&chars(&cleaned));
    toks[toks.len().saturating_sub(3)..]
        .iter()
        .any(|t| kws.contains(t))
}

/// `_digits`: digit by digit, comma-separated.
fn digits(l: &Lang, s: &str) -> R<String> {
    let mut words = Vec::new();
    for c in s.chars() {
        words.push(l.digit(py_int_char(c)?));
    }
    Ok(words.join(", "))
}

fn year4(yy: u128) -> u128 {
    if yy <= 50 {
        2000 + yy
    } else {
        1900 + yy
    }
}

/// `_decimal_word`: the language's word between "one" and "two" in 1.2.
fn decimal_word(l: &Lang) -> R<String> {
    let w = l.decimal(1, "2")?;
    let (a, b) = (l.cardinal(1), l.digit(2).to_string());
    Ok(
        match w
            .strip_prefix(a.as_str())
            .and_then(|r| r.strip_suffix(b.as_str()))
        {
            Some(mid) if w.chars().count() >= a.chars().count() + b.chars().count() => {
                super::py_strip(mid).to_string()
            }
            _ => "point".into(),
        },
    )
}

/// `pre(text, lang)`: R1–R12 outside e-mail / URL spans.
pub(super) fn pre(text: &str, lang: &str) -> R<String> {
    let Some(l) = language_table(lang) else {
        return Ok(text.to_string());
    };
    let tc = chars(text);
    let mut out = String::new();
    let mut pos = 0;
    for m in uri_re().find_iter(&tc) {
        out += &pre_piece(&sl(&tc, pos, m.start()), lang, l)?;
        out += &m.str(&tc, 0);
        pos = m.end();
    }
    out += &pre_piece(&sl(&tc, pos, tc.len()), lang, l)?;
    Ok(out)
}

/// `re.sub` with a fallible callable.
fn sub_r(
    pat: &Regex,
    text: &[char],
    mut f: impl FnMut(&super::pyre::Match) -> R<String>,
) -> R<Vec<char>> {
    let mut err = None;
    let out = pat.sub_with(text, |m| {
        f(m).unwrap_or_else(|e| {
            err.get_or_insert(e);
            String::new()
        })
    });
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

fn pre_piece(text: &[char], lang: &str, l: &Lang) -> R<String> {
    let mut t_ = text.to_vec();

    // R10 3.5%-4% -> 3.5-4% (the normaliser's own percent range)
    let r10 = re!(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%\s?[-–]\s?(\d+(?:\.\d+)?)\s?%");
    let cur = t_.clone();
    t_ = r10.sub_with(&cur, |m| format!("{}-{}%", m.str(&cur, 1), m.str(&cur, 2)));

    // R11 ₹500/kg -> "<amount> <per word> K G"
    let r11 = re_i!(
        r"((?:[₹$€£]|\b(?:Rs\.?|INR)\s?)\s?\d[\d,]*(?:\.\d+)?)\s?/\s?(kg|km|gm|g|ml|ltr|l|day|month|night|year|person)\b"
    );
    let cur = t_.clone();
    t_ = r11.sub_with(&cur, |m| {
        let mut unit = m.str(&cur, 2);
        let per = l.sym.get("per").filter(|p| !p.is_empty()).unwrap_or("per");
        if lang != "en" && t::SPELL_UNITS.contains(&unit.to_lowercase().as_str()) {
            unit = unit
                .to_uppercase()
                .chars()
                .map(|c| l.letter(c))
                .collect::<Vec<_>>()
                .join(" ");
        }
        format!("{} {per} {unit}", m.str(&cur, 1))
    });

    // R12 "p.a." after a rate
    let cur = t_.clone();
    t_ = re!(r"(?<![\w.])[pP]\.\s?[aA]\.?(?![\w])").sub_with(&cur, |_| {
        if lang == "en" {
            "per annum".into()
        } else {
            format!("{} {}", l.letter('P'), l.letter('A'))
        }
    });

    // Not upstream: a number's "ms" is milliseconds. English's abbreviation
    // table reads every standalone "ms" as the honorific "miss" (upstream
    // too), which made "50 ms" "fifty miss".
    if lang == "en" {
        let cur = t_.clone();
        t_ = re!(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)[\s-]?(?:ms|MS)\b").sub_with(&cur, |m| {
            let n = m.str(&cur, 1);
            let unit = if n == "1" {
                "millisecond"
            } else {
                "milliseconds"
            };
            format!("{n} {unit}")
        });
    }

    // R7 v4.5.2 -> "v four point five point two"
    let cur = t_.clone();
    t_ = sub_r(
        re!(r"(?<![\w.])[vV](\d+(?:\.\d+){1,3})(?![\w]|\.\d)"),
        &cur,
        |m| {
            let dw = decimal_word(l)?;
            let mut parts = Vec::new();
            for p in m.str(&cur, 1).split('.') {
                parts.push(l.cardinal(py_int(p)?));
            }
            Ok(format!(
                "{} {}",
                l.letter('v'),
                parts.join(&format!(" {dw} "))
            ))
        },
    )?;

    // R4 space-separated digit groups (>= 10 digits) -> one run
    let cur = t_.clone();
    t_ = re!(r"(?<![\d,.])(?:\d{4,5}(?: \d{3,5}){1,3}|\d{2,4} \d{3,4} \d{4})(?![\d,]|\.\d)")
        .sub_with(&cur, |m| {
            let whole = m.str(&cur, 0);
            let groups = py_split(&whole);
            if groups.iter().map(|g| g.chars().count()).sum::<usize>() >= 10 {
                groups.concat()
            } else {
                whole
            }
        });

    // R1 DD-MM-YY / DD/MM/YY / DD.MM.YY -> four-digit year
    let cur = t_.clone();
    t_ = sub_r(
        re!(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{2})(?![\d/\-]|\.\d)"),
        &cur,
        |m| {
            let (d, sep, mo, yy) = (
                m.str(&cur, 1),
                m.str(&cur, 2),
                m.str(&cur, 3),
                m.str(&cur, 4),
            );
            let (dv, mv) = (py_int(&d)?, py_int(&mo)?);
            if !((1..=31).contains(&dv) && (1..=12).contains(&mv)) {
                return Ok(m.str(&cur, 0));
            }
            if sep == "-" && d.chars().count() == 1 && mo.chars().count() == 1 {
                return Ok(m.str(&cur, 0)); // 8-2-45: a door number
            }
            let before = sl(&cur, m.start().saturating_sub(16), m.start());
            if re_i!(t::NUMBER_WORD_BEFORE).search(&before).is_some() {
                return Ok(m.str(&cur, 0)); // "House No. 12-5-24"
            }
            Ok(format!("{d}{sep}{mo}{sep}{}", year4(py_int(&yy)?)))
        },
    )?;

    // R2 card expiry MM/YY next to an expiry / date word or a CVV
    let cur = t_.clone();
    t_ = sub_r(
        re!(r"(?<![\d/.\-])(0?[1-9]|1[0-2])/(\d{2}|20\d{2})(?![\d/]|\.\d)"),
        &cur,
        |m| {
            let (mo, y) = (py_int(&m.str(&cur, 1))?, m.str(&cur, 2));
            let before = sl(&cur, m.start().saturating_sub(40), m.start());
            let after = sl(&cur, m.end(), m.end() + 40);
            let around = chars(&format!("{} {}", st(&before), st(&after)));
            let strong = expiry_re().search(&before).is_some()
                || re_i!(r"\bcvv\b").search(&around).is_some();
            let weak =
                date_re().search(&before).is_some() && (y.chars().count() == 4 || py_int(&y)? > 12);
            if !(strong || weak) {
                return Ok(m.str(&cur, 0));
            }
            let year = if y.chars().count() == 4 {
                py_int(&y)?
            } else {
                2000 + py_int(&y)?
            };
            Ok(format!("{} {}", l.month(mo as u32)?, l.year(year)))
        },
    )?;

    // R3 financial-year ranges 2023-24
    let cur = t_.clone();
    t_ = sub_r(
        re!(r"(?<![\d\-])((?:19|20)\d{2})[-–](\d{2})(?![\d\-])"),
        &cur,
        |m| {
            let (y, yy) = (py_int(&m.str(&cur, 1))?, py_int(&m.str(&cur, 2))?);
            if (y + 1) % 100 != yy {
                return Ok(m.str(&cur, 0));
            }
            Ok(format!("{} {}", l.year(y), l.cardinal(yy)))
        },
    )?;

    // R5 PIN keyword + 4-6 digits -> digit by digit
    let cur = t_.clone();
    t_ = sub_r(pin_re(), &cur, |m| {
        let s1 = m.start_of(1).unwrap_or(m.start());
        if tn_id_context(l, &sl(&cur, s1.saturating_sub(60), s1)) {
            return Ok(m.str(&cur, 0));
        }
        let head = st(&sl(&cur, m.start(), s1));
        Ok(format!("{head}{}", digits(l, &m.str(&cur, 1))?))
    })?;

    // R6 a bare six-digit number (Indian PIN code) unless an amount / ID / maths
    let cur = t_.clone();
    t_ = sub_r(
        re!(r"(?<![\d,.₹$#\w])[1-9]\d{5}(?![\d,]|\.\d|\s*%)"),
        &cur,
        |m| {
            let before = sl(&cur, m.start().saturating_sub(8), m.start());
            let sentence = sl(&cur, m.start().saturating_sub(60), m.start());
            let clause = re!(r"[.!?;।॥\n]").split(&sentence);
            let keep = re_i!(t::CURRENCY_BEFORE).search(&before).is_some()
                || re_i!(t::AMOUNT_AFTER).match_at(&cur, m.end()).is_some()
                || re!(r"[A-Za-z0-9]\s?-\s?$|[#÷×*/+=<>]\s*$")
                    .search(&before)
                    .is_some()
                || re!(r"\s*[÷×*/+=<>^]")
                    .match_at(&sl(&cur, m.end(), cur.len()), 0)
                    .is_some()
                || tn_id_context(l, &chars(clause.last().map_or("", String::as_str)));
            if keep {
                return Ok(m.str(&cur, 0));
            }
            digits(l, &m.str(&cur, 0))
        },
    )?;

    // R8 Tamil "Month D, YYYY" -> "D Month YYYY"
    if lang == "ta" {
        static R8: OnceLock<Regex> = OnceLock::new();
        let r8 = R8.get_or_init(|| {
            Regex::new(&format!(
                r"(?<!\w)({})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)",
                t::TA_MONTHS.join("|")
            ))
        });
        let cur = t_.clone();
        t_ = r8.sub_with(&cur, |m| {
            format!("{} {} {}", m.str(&cur, 2), m.str(&cur, 1), m.str(&cur, 3))
        });
    }
    Ok(st(&t_))
}

/// `post(out, seg, lang, before_tag)`: P1 (when `tag_comma`), P2, P3.
pub(super) fn post(out: &str, seg: &str, lang: &str, before_tag: bool, tag_comma: bool) -> String {
    let mut out = out.to_string();
    if tag_comma && before_tag && seg.trim_end_matches(super::pyre::is_space).ends_with(',') {
        let trimmed = out.trim_end_matches(super::pyre::is_space);
        if let Some(stem) = trimmed.strip_suffix('.') {
            out = format!("{stem},");
        }
    }
    let sc = chars(seg);
    let uris: Vec<String> = uri_re()
        .find_iter(&sc)
        .iter()
        .map(|m| m.str(&sc, 0))
        .collect();
    if let Some((_, dash)) = t::DASH.iter().find(|(k, _)| *k == lang) {
        if !uris.is_empty() {
            out = st(&re!(r"(?<=\s)dash(?=\s)").sub(&chars(&out), dash));
        }
    }
    let mut labels: Vec<String> = Vec::new();
    for u in &uris {
        let body = re!(r"^https?://").sub(&chars(u), "");
        for w in re!(r"[a-z]+").findall(&body) {
            let hit = t::URI_ACRONYMS.contains(&w.as_str())
                || t::URI_ACR_PREFIX.iter().any(|p| w.starts_with(p))
                || t::URI_ACR_SUFFIX.iter().any(|s| w.ends_with(s));
            if hit && !labels.contains(&w) {
                labels.push(w);
            }
        }
    }
    if labels.is_empty() {
        return out;
    }
    let Some(l) = language_table(lang) else {
        return out;
    };
    let spell = |w: &str| {
        w.chars()
            .map(|c| l.letter(c.to_ascii_uppercase()))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let pat = Regex::new(&format!(
        r"(?<![\w.@/])(?:{})(?![\w]|[.@/\-]\w)",
        alt_escaped(&labels)
    ));
    let oc = chars(&out);
    let replaced = pat.sub_with(&oc, |m| {
        let w = m.str(&oc, 0);
        if t::URI_ACRONYMS.contains(&w.as_str()) {
            return spell(&w);
        }
        for a in t::URI_ACR_PREFIX {
            if w.starts_with(a) && w.len() - a.len() >= 3 {
                return format!("{} {}", spell(a), &w[a.len()..]);
            }
        }
        for a in t::URI_ACR_SUFFIX {
            if w.ends_with(a) && w.len() - a.len() >= 4 {
                return format!("{} {}", &w[..w.len() - a.len()], spell(a));
            }
        }
        w
    });
    st(&replaced)
}

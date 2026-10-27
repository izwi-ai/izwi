//! Maya2's text normaliser: numbers, currency, dates, times, OTPs and phone
//! numbers, URLs and e-mails turned into spoken words in the request language
//! (`as bn en gu hi kn ml mr or pa ta te`).
//!
//! A port of upstream's `maya3.text.normalise_text(text, lang, rules=True)`
//! (<https://github.com/MayaResearch/maya2-native> at `bbfdb152`,
//! `maya3/text/`, Apache-2.0), which wraps Maya Research's vendored
//! normaliser `tn/` (upstream commit `fe5e6ec`; its Assamese number words are
//! AI4Bharat indic-numtowords data, MIT). Ported by reading the source; the
//! language tables in `tables.rs` are generated from the Python AST.
//!
//! The pipeline, per call:
//!
//! 1. split on inline tags `[laughs]` (`(\[[a-z ]+\])`): tags pass through,
//!    the text between them is normalised piece by piece;
//! 2. per piece: ALL-CAPS Latin words that look like names (4+ letters with
//!    2+ vowels) are title-cased, short / vowel-poor acronyms stay;
//! 3. `rules.rs` pre-rules R1–R12 (two-digit-year dates, card expiry, FY
//!    ranges, spaced digit groups, PIN keyword / PIN codes, versions, Tamil
//!    "Month D, YYYY", percent ranges, price per unit, p.a.), never inside an
//!    e-mail or URL;
//! 4. `tn.rs`, the vendored normaliser (40 matchers over the language tables
//!    of `lang.rs`, patched as `rules.install` / `lexicon.install` leave them:
//!    paise words, the Assamese lexicon, "per" units, sub-domain URLs);
//! 5. `rules.rs` post-rules P2 / P3 (the URL dash and acronym labels spoken in
//!    the request language);
//! 6. a piece the normaliser rejects (upstream: any exception) is kept as
//!    written; each piece keeps its leading / trailing whitespace, and the
//!    result has whitespace runs collapsed to one space and is stripped.
//!
//! Patterns run on `pyre.rs`, a small matcher with Python `re` semantics
//! (look-around, back-references, Python's `\w` / `\b`), over `char` slices
//! as Python indexes strings by code point.

mod lang;
mod pyre;
mod rules;
mod tables;
mod tn;

#[cfg(test)]
mod tests;

use pyre::{is_space, re};

/// Why a piece could not be normalised.
#[derive(Debug)]
pub(crate) enum Fail {
    /// Upstream's `NormalizationError`: `span` (in `source`, else in the text
    /// being normalised) is re-read character by character and retried.
    Norm {
        code: &'static str,
        span: Option<(usize, usize)>,
        source: Option<Vec<char>>,
    },
    /// Any other Python exception (`KeyError`, `TypeError`, `ValueError`):
    /// the wrapper keeps the piece as written.
    Crash(&'static str),
}

pub(crate) type R<T> = Result<T, Fail>;

pub(crate) fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

pub(crate) fn st(text: &[char]) -> String {
    text.iter().collect()
}

/// `text[a:b]` with Python's clamping (an empty slice when `a >= b`).
pub(crate) fn sl(text: &[char], a: usize, b: usize) -> Vec<char> {
    let b = b.min(text.len());
    if a >= b {
        Vec::new()
    } else {
        text[a..b].to_vec()
    }
}

/// `str.strip()`.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.split()`.
pub(crate) fn py_split(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// `int(c)` for one character (a Unicode decimal digit).
pub(crate) fn py_int_char(c: char) -> R<u32> {
    pyre::decimal_value(c).ok_or(Fail::Crash("ValueError: int()"))
}

/// `int(s)` for a digit string (surrounding whitespace allowed, any script's
/// decimal digits); `ValueError` otherwise. Numbers past `u128` (39+ digits,
/// which validation keeps out of the matchers) are reported the same way.
pub(crate) fn py_int(s: &str) -> R<u128> {
    let s = py_strip(s);
    if s.is_empty() {
        return Err(Fail::Crash("ValueError: int()"));
    }
    let mut n: u128 = 0;
    for c in s.chars() {
        let d = py_int_char(c)?;
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(d as u128))
            .ok_or(Fail::Crash("int() past u128"))?;
    }
    Ok(n)
}

const TAG: &str = r"(\[[a-z ]+\])";

/// `_caps_fold`: an ALL-CAPS Latin word of 4+ letters with 2+ vowels reads as
/// a name (title case); `OTP`, `NEFT`, `HDFC`, `KYC` stay acronyms.
fn caps_fold(text: &[char]) -> Vec<char> {
    re!(r"\b[A-Z]{4,}\b").sub_with(text, |m| {
        let w = m.str(text, 0);
        let vowels = w.chars().filter(|c| "AEIOU".contains(*c)).count();
        if w.len() >= 4 && vowels >= 2 {
            let mut it = w.chars();
            let first = it.next().unwrap();
            std::iter::once(first)
                .chain(it.map(|c| c.to_ascii_lowercase()))
                .collect()
        } else {
            w
        }
    })
}

/// One piece between tags: rules, normaliser, post-rules.
fn normalise_piece(folded: &str, seg: &str, lang: &str, before_tag: bool) -> R<String> {
    let pre = rules::pre(folded, lang)?;
    let out = tn::normalize(&pre, lang)?;
    Ok(rules::post(&out, seg, lang, before_tag, rules::TAG_COMMA))
}

/// The wrapper with the per-piece step injectable (tests run it without the
/// panic guard so that a panic fails them).
fn normalise_with(
    text: &str,
    lang: &str,
    piece: impl Fn(&str, &str, &str, bool) -> String,
) -> String {
    let tag = re!(TAG);
    let segs = tag.split(&chars(text));
    let mut out = String::with_capacity(text.len() * 2);
    for (i, seg) in segs.iter().enumerate() {
        if seg.is_empty() || tag.fullmatch(&chars(seg)).is_some() {
            out.push_str(seg);
            continue;
        }
        let folded = st(&caps_fold(&chars(seg)));
        let before_tag = segs
            .get(i + 1)
            .is_some_and(|next| tag.fullmatch(&chars(next)).is_some());
        let s2 = piece(&folded, seg, lang, before_tag);
        let lead = &seg[..seg.len() - seg.trim_start_matches(is_space).len()];
        let trail = &seg[seg.trim_end_matches(is_space).len()..];
        out.push_str(lead);
        out.push_str(py_strip(&s2));
        out.push_str(trail);
    }
    let collapsed = re!(r"\s+").sub(&chars(&out), " ");
    py_strip(&st(&collapsed)).to_string()
}

/// Spoken-form text for `lang` (a Maya2 language code: `as bn en gu hi kn ml
/// mr or pa ta te`; upstream's aliases such as `en-IN` or `hindi` work too).
/// Never fails: a piece the normaliser cannot handle is kept as written (an
/// unknown language keeps every piece, with only the caps fold and the
/// whitespace collapse applied). Upstream's `"auto"` script detection is not
/// ported: it is treated as an unknown language.
pub fn normalise_text(text: &str, lang: &str) -> String {
    normalise_with(text, lang, |folded, seg, lang, before_tag| {
        pyre::take_exhausted();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            normalise_piece(folded, seg, lang, before_tag)
        }));
        if pyre::take_exhausted() {
            tracing::debug!("maya2 text normaliser ran out of match steps; piece kept as written");
            return folded.to_string();
        }
        match attempt {
            Ok(Ok(spoken)) => spoken,
            Ok(Err(fail)) => {
                // like upstream's observation line: the error code, never the text
                let code = match fail {
                    Fail::Norm { code, .. } | Fail::Crash(code) => code,
                };
                tracing::debug!(code, "maya2 text normaliser kept a piece as written");
                folded.to_string()
            }
            Err(_) => {
                tracing::warn!("maya2 text normaliser panicked; piece kept as written");
                folded.to_string()
            }
        }
    })
}

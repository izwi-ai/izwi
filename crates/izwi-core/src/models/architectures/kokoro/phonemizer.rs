//! Kokoro's grapheme-to-phoneme front end.
//!
//! Kokoro was trained on misaki phonemes (hexgrad/misaki @ fba12365). For
//! English, misaki looks words up in its gold/silver lexicons and falls back
//! to espeak-ng through `EspeakFallback`, which rewrites espeak's IPA into
//! misaki's symbols (diphthongs `A I O Q W Y`, `ɜɹ`, `ᵊ`, the US flap `T`,
//! no US length marks). Other languages use `EspeakG2P` (tied phonemes to
//! misaki symbols, language-switch flags and `-` removed). Both keep
//! punctuation as tokens, so pauses, questions and sentence ends reach the
//! model.
//!
//! This module implements the espeak-based half of that front end: every
//! word goes through espeak-ng with misaki's espeak-to-misaki (E2M) rules,
//! and punctuation is kept as tokens. The misaki lexicons are not vendored,
//! so in-lexicon English words get the fallback's phonemes rather than the
//! lexicon's. Japanese and Mandarin need misaki's own G2P and are rejected.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::{Error, Result};

/// Punctuation Kokoro's vocabulary has tokens for (misaki `PUNCTS`). These
/// pass through to the model; espeak-ng never sees them.
const KOKORO_PUNCTUATION: &[char] = &[
    ';', ':', ',', '.', '!', '?', '—', '…', '"', '(', ')', '“', '”',
];

/// Which misaki conventions a language's phonemes follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroG2p {
    /// misaki `en.G2P(british=False)`'s espeak fallback.
    EnglishUs,
    /// misaki `en.G2P(british=True)`'s espeak fallback.
    EnglishGb,
    /// misaki `EspeakG2P` (es, fr-fr, hi, it, pt-br).
    Espeak,
}

/// The espeak-ng voice and G2P conventions for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KokoroVoice {
    pub espeak: &'static str,
    pub g2p: KokoroG2p,
}

impl KokoroVoice {
    const US: Self = Self {
        espeak: "en-us",
        g2p: KokoroG2p::EnglishUs,
    };
    const GB: Self = Self {
        espeak: "en-gb",
        g2p: KokoroG2p::EnglishGb,
    };

    const fn espeak(espeak: &'static str) -> Self {
        Self {
            espeak,
            g2p: KokoroG2p::Espeak,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EspeakPhonemizer {
    bin_path: PathBuf,
}

impl EspeakPhonemizer {
    pub fn new(bin_path: impl Into<PathBuf>) -> Self {
        Self {
            bin_path: bin_path.into(),
        }
    }

    pub fn auto() -> Result<Self> {
        for candidate in [
            "/opt/homebrew/bin/espeak-ng",
            "/usr/local/bin/espeak-ng",
            "espeak-ng",
        ] {
            if let Some(path) = resolve_bin(candidate) {
                return Ok(Self::new(path));
            }
        }
        Err(Error::MissingDependency(
            "espeak-ng not found; install it to enable Kokoro phonemization".to_string(),
        ))
    }

    /// misaki-convention phonemes for `text`, punctuation kept as tokens.
    /// Usually one espeak-ng call for the whole text (see
    /// [`phonemize_with`]).
    pub fn phonemize(
        &self,
        text: &str,
        language: Option<&str>,
        speaker: Option<&str>,
    ) -> Result<String> {
        let voice = kokoro_voice_for(language, speaker)?;
        phonemize_with(text, voice, |input| self.run_espeak(voice.espeak, input))
    }

    /// `text`'s clauses with their phonemes, from one espeak-ng call for
    /// the whole text (see [`phonemize_units_with`]).
    pub fn phonemize_units(
        &self,
        text: &str,
        language: Option<&str>,
        speaker: Option<&str>,
    ) -> Result<Vec<KokoroPhonemeUnit>> {
        let voice = kokoro_voice_for(language, speaker)?;
        phonemize_units_with(text, voice, |input| self.run_espeak(voice.espeak, input))
    }

    /// One espeak-ng run: IPA with `^` ties, one output line per clause.
    fn run_espeak(&self, voice: &str, text: &str) -> Result<String> {
        let output = Command::new(&self.bin_path)
            .args(espeak_args(voice, text))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to run espeak-ng for Kokoro phonemization via {}: {}",
                    self.bin_path.display(),
                    e
                ))
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::InferenceError(format!(
                "espeak-ng phonemization failed for voice '{voice}': {}",
                stderr.trim()
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    pub fn bin_path(&self) -> &Path {
        &self.bin_path
    }
}

/// One piece of the input: words for espeak-ng, a punctuation token, or a
/// whitespace run.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Words(String),
    Punct(char),
    Space,
}

/// Split `text` into words, punctuation tokens and whitespace. `...` reads
/// as `…` and `--`/`–` as `—`. A `.` or `,` between digits (`3.50`,
/// `1,000`) and a `.` between letters (`U.S`) stay inside the words.
fn split_pieces(text: &str) -> Vec<Piece> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut pieces = Vec::new();
    let mut words = String::new();
    let flush = |words: &mut String, pieces: &mut Vec<Piece>| {
        if !words.is_empty() {
            pieces.push(Piece::Words(std::mem::take(words)));
        }
    };
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        let prev = i.checked_sub(1).map(|p| chars[p]);
        let next = chars.get(i + 1).copied();
        let (token, width) = if ch == '.' && chars.get(i + 1..i + 3) == Some(&['.', '.']) {
            (Some('…'), 3)
        } else if ch == '-' && next == Some('-') {
            (Some('—'), 2)
        } else if ch == '–' {
            (Some('—'), 1)
        } else if KOKORO_PUNCTUATION.contains(&ch) && !inside_word(ch, prev, next) {
            (Some(ch), 1)
        } else {
            (None, 1)
        };
        match token {
            Some(punct) => {
                flush(&mut words, &mut pieces);
                pieces.push(Piece::Punct(punct));
            }
            None if ch.is_whitespace() => {
                flush(&mut words, &mut pieces);
                if pieces.last() != Some(&Piece::Space) {
                    pieces.push(Piece::Space);
                }
            }
            None => words.push(ch),
        }
        i += width;
    }
    flush(&mut words, &mut pieces);
    // Word pieces separated only by whitespace merge into one clause, so it
    // reaches espeak whole (stress and linking depend on neighbours).
    let mut merged: Vec<Piece> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if let Piece::Words(next) = &piece {
            if let [.., Piece::Words(_), Piece::Space] = merged.as_slice() {
                merged.pop();
                if let Some(Piece::Words(words)) = merged.last_mut() {
                    words.push(' ');
                    words.push_str(next);
                }
                continue;
            }
        }
        merged.push(piece);
    }
    merged
}

/// A `.` or `,` between digits (`3.50`, `1,000`) or a `.` between letters
/// (`U.S`) belongs to the word, not the punctuation.
fn inside_word(ch: char, prev: Option<char>, next: Option<char>) -> bool {
    let between =
        |test: fn(&char) -> bool| prev.is_some_and(|c| test(&c)) && next.is_some_and(|c| test(&c));
    ((ch == '.' || ch == ',') && between(char::is_ascii_digit))
        || (ch == '.' && between(|c| c.is_alphabetic()))
}

/// How a clause of the text ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroBoundary {
    /// `.`, `!`, `?` or `…`.
    Sentence,
    /// `,`, `;`, `:` or `—`.
    Clause,
    /// The end of the text without closing punctuation.
    End,
}

/// One clause of the input: its text, its phonemes (punctuation tokens
/// included) and how it ends. Planning packs these into chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KokoroPhonemeUnit {
    pub text: String,
    pub phonemes: String,
    pub boundary: KokoroBoundary,
}

fn boundary_of(punct: char) -> Option<KokoroBoundary> {
    match punct {
        '.' | '!' | '?' | '…' => Some(KokoroBoundary::Sentence),
        ',' | ';' | ':' | '—' => Some(KokoroBoundary::Clause),
        _ => None,
    }
}

/// Phonemes for `text` in `voice`'s conventions, with `run` invoking
/// espeak-ng on a string: the clauses of [`phonemize_units_with`] joined
/// by spaces.
pub(crate) fn phonemize_with(
    text: &str,
    voice: KokoroVoice,
    run: impl FnMut(&str) -> Result<String>,
) -> Result<String> {
    Ok(phonemize_units_with(text, voice, run)?
        .into_iter()
        .map(|unit| unit.phonemes)
        .collect::<Vec<_>>()
        .join(" "))
}

/// `text`'s clauses with their phonemes, with `run` invoking espeak-ng on
/// a string. The word segments between punctuation tokens go to espeak-ng
/// in one call for the whole text, joined by commas: espeak-ng ends a clause
/// (and an output line) at every comma, so line `i` is segment `i`. If the
/// line count does not match (espeak-ng split or merged a clause on its
/// own), each segment is phonemized separately instead. Punctuation stays
/// where it was, and a clause ends after sentence or clause punctuation
/// (closing quotes and brackets stay with the clause they close).
pub(crate) fn phonemize_units_with(
    text: &str,
    voice: KokoroVoice,
    mut run: impl FnMut(&str) -> Result<String>,
) -> Result<Vec<KokoroPhonemeUnit>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidInput(
            "Kokoro TTS input text is empty".to_string(),
        ));
    }
    if trimmed.contains('\0') {
        return Err(Error::InvalidInput(
            "Kokoro TTS input text contains a NUL character".to_string(),
        ));
    }
    let pieces = split_pieces(trimmed);
    let segments = pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Words(words) => Some(words.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut phonemized = Vec::with_capacity(segments.len());
    if !segments.is_empty() {
        let lines = espeak_lines(&run(&segments.join(", "))?);
        if lines.len() == segments.len() {
            phonemized = lines;
        } else {
            for segment in &segments {
                phonemized.push(espeak_lines(&run(segment)?).join(" "));
            }
        }
    }
    let mut phonemized = phonemized
        .into_iter()
        .map(|line| espeak_to_misaki(&line, voice.g2p));

    let mut units = Vec::new();
    let mut unit_text = String::new();
    let mut unit_phonemes = String::new();
    let mut pending: Option<KokoroBoundary> = None;
    let mut close = |text: &mut String, phonemes: &mut String, boundary| {
        let unit = KokoroPhonemeUnit {
            text: std::mem::take(text).trim().to_string(),
            phonemes: std::mem::take(phonemes).trim().to_string(),
            boundary,
        };
        if !unit.phonemes.is_empty() {
            units.push(unit);
        }
    };
    for piece in &pieces {
        if let Some(boundary) = pending {
            let closes_quote = matches!(piece, Piece::Punct(')' | '”' | '"'));
            if !closes_quote {
                close(&mut unit_text, &mut unit_phonemes, boundary);
                pending = None;
            }
        }
        match piece {
            Piece::Words(words) => {
                unit_text.push_str(words);
                if let Some(phonemes) = phonemized.next() {
                    unit_phonemes.push_str(&phonemes);
                }
            }
            Piece::Punct(punct) => {
                unit_text.push(*punct);
                unit_phonemes.push(*punct);
                if let Some(boundary) = boundary_of(*punct) {
                    pending = Some(boundary);
                }
            }
            Piece::Space => {
                unit_text.push(' ');
                if !unit_phonemes.is_empty() && !unit_phonemes.ends_with(' ') {
                    unit_phonemes.push(' ');
                }
            }
        }
    }
    close(
        &mut unit_text,
        &mut unit_phonemes,
        pending.unwrap_or(KokoroBoundary::End),
    );
    if !units.iter().any(|unit| {
        unit.phonemes
            .chars()
            .any(|ch| !KOKORO_PUNCTUATION.contains(&ch) && ch != ' ')
    }) {
        return Err(Error::InferenceError(
            "espeak-ng returned empty phonemes for Kokoro request".to_string(),
        ));
    }
    Ok(units)
}

/// espeak-ng's output, one cleaned line per clause.
fn espeak_lines(raw: &str) -> Vec<String> {
    raw.lines()
        .map(normalize_espeak_ipa)
        .filter(|line| !line.is_empty())
        .collect()
}

/// espeak-ng IPA (with `^` ties) to misaki's symbols: misaki's
/// `EspeakFallback` for English, `EspeakG2P` otherwise.
pub(crate) fn espeak_to_misaki(ipa: &str, g2p: KokoroG2p) -> String {
    let mut ps = remove_language_flags(ipa);
    match g2p {
        KokoroG2p::EnglishUs | KokoroG2p::EnglishGb => {
            // misaki EspeakFallback.E2M, longest first.
            for (old, new) in [
                ("ʔˌn\u{329}", "tᵊn"),
                ("ʔn\u{329}", "tᵊn"),
                ("ʔn", "tᵊn"),
                ("a^ɪ", "I"),
                ("a^ʊ", "W"),
                ("d^ʒ", "ʤ"),
                ("e^ɪ", "A"),
                ("t^ʃ", "ʧ"),
                ("ɔ^ɪ", "Y"),
                ("ə^l", "ᵊl"),
                ("ʔ", "t"),
                ("r", "ɹ"),
                ("x", "k"),
                ("ç", "k"),
                ("ɐ", "ə"),
                ("ɚ", "əɹ"),
                ("ɬ", "l"),
                ("ʲ", ""),
            ] {
                ps = ps.replace(old, new);
            }
            ps = syllabic_to_schwa(&ps);
            if g2p == KokoroG2p::EnglishGb {
                ps = ps
                    .replace("e^ə", "ɛː")
                    .replace("iə", "ɪə")
                    .replace("ə^ʊ", "Q");
            } else {
                ps = ps
                    .replace("o^ʊ", "O")
                    .replace("ɜːɹ", "ɜɹ")
                    .replace("ɜː", "ɜɹ")
                    .replace("ɪə", "iə")
                    .replace('ː', "")
                    .replace('ɾ', "T");
            }
        }
        KokoroG2p::Espeak => {
            for (old, new) in [
                ("a^ɪ", "I"),
                ("a^ʊ", "W"),
                ("d^z", "ʣ"),
                ("d^ʒ", "ʤ"),
                ("e^ɪ", "A"),
                ("o^ʊ", "O"),
                ("ə^ʊ", "Q"),
                ("s^s", "S"),
                ("t^s", "ʦ"),
                ("t^ʃ", "ʧ"),
                ("ɔ^ɪ", "Y"),
            ] {
                ps = ps.replace(old, new);
            }
            ps = ps.replace('-', "");
        }
    }
    ps.replace('^', "")
}

/// misaki: a syllabic consonant `C̩` becomes `ᵊC`.
fn syllabic_to_schwa(ps: &str) -> String {
    let chars = ps.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(ps.len());
    let mut i = 0;
    while i < chars.len() {
        if chars.get(i + 1) == Some(&'\u{329}') && !chars[i].is_whitespace() {
            out.push('ᵊ');
            out.push(chars[i]);
            i += 2;
        } else {
            if chars[i] != '\u{329}' {
                out.push(chars[i]);
            }
            i += 1;
        }
    }
    out
}

/// Strip espeak-ng's language-switch flags such as `(en)` and `(fr)`.
fn remove_language_flags(ipa: &str) -> String {
    let mut out = String::with_capacity(ipa.len());
    let mut rest = ipa;
    while let Some(start) = rest.find('(') {
        let after = &rest[start + 1..];
        match after.find(')') {
            Some(end)
                if end > 0
                    && after[..end]
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c == '-') =>
            {
                out.push_str(&rest[..start]);
                rest = &after[end + 1..];
            }
            _ => {
                out.push_str(&rest[..=start]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The espeak-ng argument vector for one phonemization call.
///
/// User text is untrusted: espeak-ng parses its argv with getopt, so text such
/// as `-f/etc/passwd` would otherwise be read as an option (here: speak a local
/// file) and text such as `-5 degrees` would be rejected as an invalid option.
/// The `--` terminator makes getopt treat everything after it as an operand, so
/// the text is always phonemized literally. The text must stay the final
/// argument, after the terminator.
fn espeak_args<'a>(voice: &'a str, text: &'a str) -> [&'a OsStr; 7] {
    [
        OsStr::new("-q"),
        OsStr::new("--ipa"),
        OsStr::new("--tie=^"),
        OsStr::new("-v"),
        OsStr::new(voice),
        OsStr::new("--"),
        OsStr::new(text),
    ]
}

fn resolve_bin(candidate: &str) -> Option<PathBuf> {
    let path = PathBuf::from(candidate);
    if path.is_absolute() {
        return path.exists().then_some(path);
    }
    let out = Command::new("which").arg(candidate).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let resolved = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if resolved.is_empty() {
        None
    } else {
        Some(PathBuf::from(resolved))
    }
}

/// Whether Kokoro can phonemize for `speaker`: Japanese (`j*`) and Mandarin
/// (`z*`) voices need misaki's own G2P, which is not ported.
pub fn kokoro_speaker_supported(speaker: &str) -> bool {
    !matches!(
        speaker.chars().next().map(|c| c.to_ascii_lowercase()),
        Some('j') | Some('z')
    )
}

fn unsupported_language(name: &str) -> Error {
    Error::InvalidInput(format!(
        "Kokoro does not support {name} yet: it needs misaki's {name} G2P, which is not available; use an English, Spanish, French, Hindi, Italian or Portuguese voice"
    ))
}

/// The espeak-ng voice and G2P conventions for a request (misaki pipeline
/// `LANG_CODES`). An explicit British language (`en-gb`, `en_GB`, `en-uk`,
/// "british") is British English; an explicit `en-us` or "american" is US
/// English; a generic English (`en`, "english") follows the voice's prefix,
/// so British voices (`bf_`/`bm_`) get British phonemes. Portuguese is
/// Brazilian (`pt-br`). Japanese and Mandarin, by language or by voice
/// (`j*`/`z*`), are rejected as invalid input.
pub fn kokoro_voice_for(language: Option<&str>, speaker: Option<&str>) -> Result<KokoroVoice> {
    let prefix = speaker
        .and_then(|s| s.trim().chars().next())
        .map(|c| c.to_ascii_lowercase());
    match prefix {
        Some('j') => return Err(unsupported_language("Japanese")),
        Some('z') => return Err(unsupported_language("Mandarin")),
        _ => {}
    }
    let from_prefix = match prefix {
        Some('b') => KokoroVoice::GB,
        Some('e') => KokoroVoice::espeak("es"),
        Some('f') => KokoroVoice::espeak("fr-fr"),
        Some('h') => KokoroVoice::espeak("hi"),
        Some('i') => KokoroVoice::espeak("it"),
        Some('p') => KokoroVoice::espeak("pt-br"),
        _ => KokoroVoice::US,
    };
    let Some(language) = language
        .map(|l| l.trim().to_ascii_lowercase().replace('_', "-"))
        .filter(|l| !l.is_empty())
    else {
        return Ok(from_prefix);
    };
    let l = language.as_str();
    let voice = if matches!(l, "en-gb" | "en-uk" | "british" | "british english") {
        KokoroVoice::GB
    } else if matches!(l, "en-us" | "american" | "american english") {
        KokoroVoice::US
    } else if l.starts_with("en") || l == "english" {
        match from_prefix.g2p {
            KokoroG2p::EnglishGb => KokoroVoice::GB,
            _ => KokoroVoice::US,
        }
    } else if l.starts_with("ja") || l == "japanese" {
        return Err(unsupported_language("Japanese"));
    } else if l.starts_with("zh") || l.starts_with("cmn") || matches!(l, "chinese" | "mandarin") {
        return Err(unsupported_language("Mandarin"));
    } else if l.starts_with("es") || l == "spanish" {
        KokoroVoice::espeak("es")
    } else if l.starts_with("fr") || l == "french" {
        KokoroVoice::espeak("fr-fr")
    } else if l.starts_with("it") || l == "italian" {
        KokoroVoice::espeak("it")
    } else if l.starts_with("pt") || l == "portuguese" {
        KokoroVoice::espeak("pt-br")
    } else if l.starts_with("hi") || l == "hindi" {
        KokoroVoice::espeak("hi")
    } else if l.starts_with("de") || l == "german" {
        KokoroVoice::espeak("de")
    } else {
        from_prefix
    };
    Ok(voice)
}

pub fn normalize_espeak_ipa(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_was_space = false;

    for ch in raw.chars() {
        let mapped = match ch {
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FE0F}' => None,
            '\r' | '\n' | '\t' => Some(' '),
            // Common eSpeak combining tie; ties arrive as `^` (`--tie=^`).
            '\u{0361}' => None,
            _ if ch.is_control() => None,
            _ => Some(ch),
        };

        if let Some(ch) = mapped {
            if ch.is_whitespace() {
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
            } else {
                out.push(ch);
                last_was_space = false;
            }
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn espeak_args_terminate_options_before_user_text() {
        for text in [
            "-f/etc/passwd",
            "--version",
            "- First item",
            "-5 degrees",
            "hello",
        ] {
            let args = espeak_args("en-us", text);
            let n = args.len();
            assert_eq!(
                args[n - 1],
                OsStr::new(text),
                "text must be the last argument"
            );
            assert_eq!(args[n - 2], OsStr::new("--"), "`--` must precede the text");
            assert!(
                args[..n - 2].iter().all(|a| *a != OsStr::new(text)),
                "text must not appear among the options"
            );
        }
    }

    /// Runs only when espeak-ng is installed; otherwise the argv test above is
    /// the guard.
    fn installed_espeak() -> Option<EspeakPhonemizer> {
        match EspeakPhonemizer::auto() {
            Ok(p) => Some(p),
            Err(_) => {
                eprintln!("espeak-ng not installed; skipping live phonemizer check");
                None
            }
        }
    }

    #[test]
    fn leading_dash_text_is_phonemized_literally() {
        let Some(espeak) = installed_espeak() else {
            return;
        };
        let minus = espeak
            .phonemize("-5 degrees outside.", Some("en"), None)
            .expect("`-5 degrees` must phonemize");
        assert!(minus.starts_with("mˈInəs"), "got {minus:?}");
        let item = espeak
            .phonemize("- First item, then second.", Some("en"), None)
            .expect("`- First item` must phonemize");
        assert!(item.contains("fˈɜɹst"), "got {item:?}");
        let version = espeak
            .phonemize("--version", Some("en"), None)
            .expect("`--version` must phonemize");
        assert!(
            !version.contains("eSpeak") && !version.contains('/'),
            "got {version:?}"
        );
    }

    #[test]
    fn file_option_text_never_reads_the_file() {
        let Some(espeak) = installed_espeak() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("izwi-kokoro-argv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.txt");
        std::fs::write(&path, "zebra zebra zebra").unwrap();
        let input = format!("-f{}", path.display());
        let result = espeak.phonemize(&input, Some("en"), None);
        let file_phonemes = espeak.phonemize("zebra", Some("en"), None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let phonemes = result.expect("`-f<path>` must phonemize as text");
        assert!(
            !phonemes.contains(&file_phonemes),
            "espeak-ng spoke the file contents: {phonemes:?}"
        );
    }

    #[test]
    fn normalizes_zero_width_and_spacing() {
        let raw = "həlˈo\u{200d}ʊ\nwˈɜːld";
        assert_eq!(normalize_espeak_ipa(raw), "həlˈoʊ wˈɜːld");
    }

    /// KOK-4: golden espeak-ng 1.52 lines (`--ipa --tie=^`) through misaki's
    /// `EspeakFallback` rules, derived by hand from misaki @ fba12365
    /// `espeak.py` (the E2M table, the syllabic rule, then the US or GB
    /// rewrites). misaki itself gives `həlˈO wˈɜɹld` for "Hello world".
    #[test]
    fn english_espeak_ipa_maps_to_misaki_symbols() {
        let us = [
            ("həlˈo^ʊ wˈɜːld", "həlˈO wˈɜɹld"),
            ("nˈo^ʊ wˈɔːɾɚ", "nˈO wˈɔTəɹ"),
            ("nˈɜːs bˈʌʔn̩ lˈɪɾə^l", "nˈɜɹs bˈʌtᵊn lˈɪTᵊl"),
            ("wˈe^ɪt", "wˈAt"),
            ("a^ɪm fˈa^ɪn", "Im fˈIn"),
            ("hˈa^ʊ ɑːɹ juː", "hˈW ɑɹ ju"),
            ("bˈɔ^ɪ d^ʒˈʌmpt t^ʃˈiːz", "bˈY ʤˈʌmpt ʧˈiz"),
            ("ɹˈi^əli", "ɹˈiəli"),
            ("lˈɑːk (fr)lə(en)", "lˈɑk lə"),
        ];
        for (raw, expected) in us {
            assert_eq!(
                espeak_to_misaki(raw, KokoroG2p::EnglishUs),
                expected,
                "US {raw}"
            );
        }
        let gb = [
            ("həlˈə^ʊ wˈɜːld", "həlˈQ wˈɜːld"),
            ("nˈə^ʊ wˈɔːtə", "nˈQ wˈɔːtə"),
            ("ðˈe^ə", "ðˈɛː"),
            ("nˈiə", "nˈɪə"),
            ("fˈa^ɪn", "fˈIn"),
        ];
        for (raw, expected) in gb {
            assert_eq!(
                espeak_to_misaki(raw, KokoroG2p::EnglishGb),
                expected,
                "GB {raw}"
            );
        }
        let other = [
            ("t^ʃˈa^ʊ", "ʧˈW"),
            ("kˌomʊ vosˌe estˈa", "kˌomʊ vosˌe estˈa"),
            ("pit^sa d^zˈe-ɾo", "piʦa ʣˈeɾo"),
        ];
        for (raw, expected) in other {
            assert_eq!(
                espeak_to_misaki(raw, KokoroG2p::Espeak),
                expected,
                "espeak {raw}"
            );
        }
    }

    /// KOK-3: punctuation becomes tokens in its original place, the words
    /// between it reach espeak-ng in one call (line `i` is segment `i`), and
    /// a line-count mismatch falls back to one call per segment.
    #[test]
    fn punctuation_is_kept_and_segments_are_phonemized_in_one_call() {
        let calls = RefCell::new(Vec::new());
        let fake = |input: &str| -> Result<String> {
            calls.borrow_mut().push(input.to_string());
            Ok(input
                .split(", ")
                .map(|segment| format!("{}^", segment.to_lowercase()))
                .collect::<Vec<_>>()
                .join("\n"))
        };
        let out = phonemize_with(
            "Wait, really? Yes. No! He said “Hi” (twice) -- then... left; costs 3.50, 1,000 in the U.S. too",
            KokoroVoice::US,
            fake,
        )
        .unwrap();
        assert_eq!(
            out,
            "wait, ɹeally? yes. no! he said “hi” (twice) — then… left; costs 3.50, 1,000 in the u.s. too"
        );
        assert_eq!(calls.borrow().len(), 1, "{:?}", calls.borrow());
        assert_eq!(
            calls.borrow()[0],
            "Wait, really, Yes, No, He said, Hi, twice, then, left, costs 3.50, 1,000 in the U.S, too"
        );

        // espeak-ng merging two clauses: each segment is retried alone.
        let calls = RefCell::new(0usize);
        let merging = |input: &str| -> Result<String> {
            *calls.borrow_mut() += 1;
            Ok(input.replace(", ", " "))
        };
        let out = phonemize_with("One, two.", KokoroVoice::US, merging).unwrap();
        assert_eq!(out, "One, two.");
        assert_eq!(*calls.borrow(), 3);

        assert!(phonemize_with("?!", KokoroVoice::US, |_| Ok(String::new())).is_err());
        assert!(phonemize_with("   ", KokoroVoice::US, |_| Ok(String::new())).is_err());
    }

    /// KOK-11: a whole multi-sentence text is phonemized in one espeak-ng
    /// call and comes back as clauses with their boundaries; closing quotes
    /// stay with the sentence they close.
    #[test]
    fn units_split_at_clauses_from_one_espeak_call() {
        let calls = RefCell::new(0usize);
        let fake = |input: &str| -> Result<String> {
            *calls.borrow_mut() += 1;
            Ok(input
                .split(", ")
                .map(str::to_lowercase)
                .collect::<Vec<_>>()
                .join("\n"))
        };
        let units = phonemize_units_with(
            "He said “Hi.” Then, at last, we left! Done",
            KokoroVoice::US,
            fake,
        )
        .unwrap();
        assert_eq!(*calls.borrow(), 1);
        let summary = units
            .iter()
            .map(|u| (u.text.as_str(), u.phonemes.as_str(), u.boundary))
            .collect::<Vec<_>>();
        use KokoroBoundary::*;
        assert_eq!(
            summary,
            [
                ("He said “Hi.”", "he said “hi.”", Sentence),
                ("Then,", "then,", Clause),
                ("at last,", "at last,", Clause),
                ("we left!", "we left!", Sentence),
                ("Done", "done", End),
            ]
        );
    }

    /// KOK-3/KOK-4 live: the pinned espeak-ng 1.52 output through misaki's
    /// rules, punctuation kept. "No!" is `nˈO!`, not the split `oʊ` the
    /// model heard as "now".
    #[test]
    fn live_espeak_matches_misaki_fallback_goldens() {
        let Some(espeak) = installed_espeak() else {
            return;
        };
        let version = Command::new(espeak.bin_path())
            .arg("--version")
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default();
        if !version.contains("1.52") {
            eprintln!("skipping goldens pinned to espeak-ng 1.52: {version}");
            return;
        }
        for (text, language, speaker, expected) in [
            (
                "Wait, really? Yes. No!",
                None,
                Some("af_heart"),
                "wˈAt, ɹˈiəli? jˈɛs. nˈO!",
            ),
            (
                "Hello world.",
                Some("en"),
                Some("af_heart"),
                "həlˈO wˈɜɹld.",
            ),
            (
                "Hello world.",
                Some("en"),
                Some("bm_george"),
                "həlˈQ wˈɜːld.",
            ),
            (
                "Hello world.",
                Some("en-us"),
                Some("bm_george"),
                "həlˈO wˈɜɹld.",
            ),
            (
                "The water is better, isn't it?",
                None,
                Some("af_heart"),
                "ðə wˈɔTəɹɹ ɪz bˈɛTəɹ, ˈɪzənt ɪt?",
            ),
            (
                "Olá, como você está?",
                Some("pt"),
                Some("pf_dora"),
                "olˈa, kˌomʊ vosˌe estˈa?",
            ),
        ] {
            let got = espeak.phonemize(text, language, speaker).unwrap();
            assert_eq!(got, expected, "{text} ({language:?}, {speaker:?})");
        }
    }

    /// KOK-5/KOK-6/KOK-7: British voices get British English for a generic
    /// "en"; explicit variants win; Portuguese is Brazilian; Japanese and
    /// Mandarin, by language or voice, are invalid input.
    #[test]
    fn voice_resolution_follows_misaki_language_codes() {
        let voice = |language, speaker| kokoro_voice_for(language, speaker).unwrap();
        assert_eq!(voice(None, Some("af_heart")), KokoroVoice::US);
        assert_eq!(voice(None, Some("bf_lily")), KokoroVoice::GB);
        assert_eq!(voice(Some("en"), Some("bm_george")), KokoroVoice::GB);
        assert_eq!(voice(Some("English"), Some("bm_george")), KokoroVoice::GB);
        assert_eq!(voice(Some("en-GB"), Some("af_heart")), KokoroVoice::GB);
        assert_eq!(voice(Some("en_GB"), Some("af_heart")), KokoroVoice::GB);
        assert_eq!(voice(Some("en-us"), Some("bm_george")), KokoroVoice::US);
        assert_eq!(voice(Some("en"), Some("af_heart")), KokoroVoice::US);
        assert_eq!(voice(Some("pt"), None).espeak, "pt-br");
        assert_eq!(voice(None, Some("pf_dora")).espeak, "pt-br");
        assert_eq!(voice(Some("Spanish"), None).espeak, "es");
        assert_eq!(voice(None, Some("ff_siwis")).espeak, "fr-fr");
        for (language, speaker) in [
            (Some("ja"), None),
            (Some("zh"), Some("af_heart")),
            (Some("Mandarin"), None),
            (None, Some("jf_alpha")),
            (None, Some("zm_yunxi")),
            (Some("en"), Some("zf_xiaobei")),
        ] {
            let err = kokoro_voice_for(language, speaker).unwrap_err();
            assert!(matches!(err, Error::InvalidInput(_)), "{err}");
        }
        assert!(kokoro_speaker_supported("af_heart"));
        assert!(!kokoro_speaker_supported("jf_alpha"));
        assert!(!kokoro_speaker_supported("zm_yunxi"));
    }
}

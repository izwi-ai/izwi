use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Error, Result};

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

    pub fn phonemize(
        &self,
        text: &str,
        language: Option<&str>,
        speaker: Option<&str>,
    ) -> Result<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidInput(
                "Kokoro TTS input text is empty".to_string(),
            ));
        }

        let voice = espeak_voice_for(language, speaker);
        let output = Command::new(&self.bin_path)
            .arg("-q")
            .arg("--ipa=3")
            .arg("-v")
            .arg(&voice)
            .arg(trimmed)
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

        let raw = String::from_utf8_lossy(&output.stdout);
        let normalized = normalize_espeak_ipa(&raw);
        if normalized.is_empty() {
            return Err(Error::InferenceError(
                "espeak-ng returned empty phonemes for Kokoro request".to_string(),
            ));
        }
        Ok(normalized)
    }

    pub fn bin_path(&self) -> &Path {
        &self.bin_path
    }
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

pub fn espeak_voice_for(language: Option<&str>, speaker: Option<&str>) -> String {
    if let Some(lang) = language {
        let l = lang.trim().to_ascii_lowercase();
        if !l.is_empty() {
            if l.starts_with("en") {
                return "en-us".to_string();
            }
            if l.starts_with("ja") {
                return "ja".to_string();
            }
            if l.starts_with("zh") {
                return "cmn".to_string();
            }
            if l.starts_with("es") {
                return "es".to_string();
            }
            if l.starts_with("fr") {
                return "fr-fr".to_string();
            }
            if l.starts_with("it") {
                return "it".to_string();
            }
            if l.starts_with("pt") {
                return "pt".to_string();
            }
            if l.starts_with("hi") {
                return "hi".to_string();
            }
            if l.starts_with("de") {
                return "de".to_string();
            }
        }
    }

    let prefix = speaker
        .and_then(|s| s.split('_').next())
        .and_then(|p| p.chars().next())
        .map(|c| c.to_ascii_lowercase());

    match prefix {
        Some('j') => "ja".to_string(),
        Some('z') => "cmn".to_string(),
        Some('p') => "pt".to_string(),
        Some('i') => "it".to_string(),
        Some('f') => "fr-fr".to_string(),
        Some('e') => "es".to_string(),
        Some('h') => "hi".to_string(),
        Some('b') => "en-gb".to_string(),
        Some('a') | Some('m') | Some('u') => "en-us".to_string(),
        _ => "en-us".to_string(),
    }
}

pub fn normalize_espeak_ipa(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_was_space = false;

    for ch in raw.chars() {
        let mapped = match ch {
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FE0F}' => None,
            '\r' | '\n' | '\t' => Some(' '),
            // Common eSpeak combining tie; Kokoro vocab uses precomposed affricates too.
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

    let out = out
        .replace("dʒ", "ʤ")
        .replace("tʃ", "ʧ")
        .replace("d\u{200d}ʒ", "ʤ")
        .replace("t\u{200d}ʃ", "ʧ");

    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::{espeak_voice_for, normalize_espeak_ipa};

    #[test]
    fn normalizes_zero_width_and_spacing() {
        let raw = "həlˈo\u{200d}ʊ\nwˈɜːld";
        assert_eq!(normalize_espeak_ipa(raw), "həlˈoʊ wˈɜːld");
    }

    #[test]
    fn infers_espeak_voice_from_kokoro_speaker_prefix() {
        assert_eq!(espeak_voice_for(None, Some("af_heart")), "en-us");
        assert_eq!(espeak_voice_for(None, Some("bf_lily")), "en-gb");
        assert_eq!(espeak_voice_for(None, Some("jf_alpha")), "ja");
        assert_eq!(espeak_voice_for(None, Some("zf_xiaobei")), "cmn");
    }
}

//! Rumble's prompt contract (the published `chat_template.jinja`):
//!
//! ```text
//! <|im_start|>system\n{style}{context}<|im_end|>\n
//! <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
//! <|im_start|>assistant\n[language {Name}<asr_text>]
//! ```
//!
//! `{style}` is three control tokens: `<digits>` or `<nodigits>`, `<filler>`
//! or `<nofiller>`, `<verbatim>` or `<noverbatim>`. With a forced language
//! the assistant turn is prefilled and the model emits only the transcript;
//! without one it emits `language {Name}<asr_text>` first.

use crate::error::{Error, Result};

use super::tokenizer::{RumbleSpecialTokens, RumbleTokenizer};

/// How the transcript is written. The defaults are the model's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RumbleTranscriptStyle {
    /// Numbers as digits (`true`) or as words.
    pub digits: bool,
    /// Keep fillers such as "um" and "uh".
    pub fillers: bool,
    /// Keep repeated words (`true`) or collapse them.
    pub verbatim: bool,
}

impl Default for RumbleTranscriptStyle {
    fn default() -> Self {
        Self {
            digits: true,
            fillers: false,
            verbatim: true,
        }
    }
}

impl RumbleTranscriptStyle {
    /// The three control ids, in the template's order.
    pub fn control_ids(&self, specials: &RumbleSpecialTokens) -> [u32; 3] {
        [
            if self.digits {
                specials.digits
            } else {
                specials.no_digits
            },
            if self.fillers {
                specials.filler
            } else {
                specials.no_filler
            },
            if self.verbatim {
                specials.verbatim
            } else {
                specials.no_verbatim
            },
        ]
    }
}

/// Languages the checkpoint documents a forced-language name for
/// (`config.json` `support_languages`), as (ISO code, name).
pub const RUMBLE_FORCED_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

/// Languages the card lists without a documented forced-language name. The
/// model detects them; a request that forces one sends its English name.
pub const RUMBLE_DETECTED_LANGUAGES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("ceb", "Cebuano"),
    ("eu", "Basque"),
    ("hr", "Croatian"),
    ("is", "Icelandic"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mt", "Maltese"),
    ("nb", "Norwegian"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("uk", "Ukrainian"),
];

/// The forced-language name for a request's language: `None` for no
/// language or `auto`; the model's name for a known code or name (any
/// case, with or without a region such as `en-US`); any other value
/// title-cased and passed through, since the model accepts English language
/// names beyond its documented list.
pub fn resolve_rumble_language(language: Option<&str>) -> Option<String> {
    let raw = language?.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    let lowered = raw.to_ascii_lowercase();
    let base = lowered.split(['-', '_']).next().unwrap_or(lowered.as_str());
    let known = RUMBLE_FORCED_LANGUAGES
        .iter()
        .chain(RUMBLE_DETECTED_LANGUAGES)
        .find(|(code, name)| *code == lowered || name.eq_ignore_ascii_case(raw) || *code == base);
    if let Some((_, name)) = known {
        return Some((*name).to_string());
    }
    let mut name = String::with_capacity(raw.len());
    let mut start = true;
    for ch in raw.chars() {
        if ch.is_alphabetic() {
            if start {
                name.extend(ch.to_uppercase());
            } else {
                name.extend(ch.to_lowercase());
            }
            start = false;
        } else {
            name.push(ch);
            start = true;
        }
    }
    Some(name)
}

/// What a request puts in the prompt besides its audio.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RumblePromptOptions {
    /// The spoken language, when the caller knows it.
    pub language: Option<String>,
    /// Context text (names, terms) placed after the style tokens.
    pub context: Option<String>,
    pub style: RumbleTranscriptStyle,
}

/// A tokenized prompt with its audio placeholder span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RumblePrompt {
    pub ids: Vec<u32>,
    pub audio_start: usize,
    pub audio_tokens: usize,
    /// The language the assistant turn was prefilled with.
    pub forced_language: Option<String>,
    pub context_tokens: usize,
}

/// Longest context text, in tokens, a request may put in the system turn.
pub const RUMBLE_MAX_CONTEXT_TOKENS: usize = 256;

pub fn build_rumble_prompt(
    tokenizer: &RumbleTokenizer,
    audio_tokens: usize,
    options: &RumblePromptOptions,
) -> Result<RumblePrompt> {
    if audio_tokens == 0 {
        return Err(Error::InvalidInput(
            "Rumble prompt needs at least one audio token".into(),
        ));
    }
    let specials = tokenizer.specials();
    let mut ids = Vec::with_capacity(audio_tokens + 48);
    ids.push(specials.im_start);
    ids.extend(tokenizer.encode("system\n")?);
    ids.extend(options.style.control_ids(specials));
    let context = options
        .context
        .as_deref()
        .map(str::trim)
        .filter(|context| !context.is_empty());
    let context_tokens = match context {
        Some(context) => {
            let tokens = tokenizer.encode(context)?;
            if tokens.len() > RUMBLE_MAX_CONTEXT_TOKENS {
                return Err(Error::InvalidInput(format!(
                    "Rumble context prompt is {} tokens; at most {RUMBLE_MAX_CONTEXT_TOKENS} fit",
                    tokens.len()
                )));
            }
            let count = tokens.len();
            ids.extend(tokens);
            count
        }
        None => 0,
    };
    ids.push(specials.im_end);
    ids.extend(tokenizer.encode("\n")?);
    ids.push(specials.im_start);
    ids.extend(tokenizer.encode("user\n")?);
    ids.push(specials.audio_start);
    let audio_start = ids.len();
    ids.extend(std::iter::repeat_n(specials.audio_pad, audio_tokens));
    ids.push(specials.audio_end);
    ids.push(specials.im_end);
    ids.extend(tokenizer.encode("\n")?);
    ids.push(specials.im_start);
    ids.extend(tokenizer.encode("assistant\n")?);
    let forced_language = resolve_rumble_language(options.language.as_deref());
    if let Some(language) = forced_language.as_deref() {
        ids.extend(tokenizer.encode(&format!("language {language}"))?);
        ids.push(specials.asr_text);
    }
    Ok(RumblePrompt {
        ids,
        audio_start,
        audio_tokens,
        forced_language,
        context_tokens,
    })
}

/// Prompt tokens besides the audio and the context text: the turn markers,
/// the style tokens and a forced-language prefix. A bound, for sizing.
pub const RUMBLE_PROMPT_OVERHEAD_TOKENS: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_resolve_by_code_name_and_region() {
        assert_eq!(resolve_rumble_language(None), None);
        assert_eq!(resolve_rumble_language(Some("")), None);
        assert_eq!(resolve_rumble_language(Some("Auto")), None);
        for input in ["en", "EN", "english", "English", "en-US", "en_gb"] {
            assert_eq!(
                resolve_rumble_language(Some(input)).as_deref(),
                Some("English"),
                "{input}"
            );
        }
        assert_eq!(
            resolve_rumble_language(Some("nl")).as_deref(),
            Some("Dutch")
        );
        assert_eq!(
            resolve_rumble_language(Some("yue")).as_deref(),
            Some("Cantonese")
        );
        assert_eq!(
            resolve_rumble_language(Some("zh-CN")).as_deref(),
            Some("Chinese")
        );
        // On the card, without a documented name.
        assert_eq!(
            resolve_rumble_language(Some("uk")).as_deref(),
            Some("Ukrainian")
        );
        // Unknown values pass through title-cased.
        assert_eq!(
            resolve_rumble_language(Some("swahili")).as_deref(),
            Some("Swahili")
        );
    }

    #[test]
    fn the_language_tables_cover_the_card_once() {
        assert_eq!(RUMBLE_FORCED_LANGUAGES.len(), 30);
        assert_eq!(RUMBLE_DETECTED_LANGUAGES.len(), 15);
        let mut codes = RUMBLE_FORCED_LANGUAGES
            .iter()
            .chain(RUMBLE_DETECTED_LANGUAGES)
            .map(|(code, _)| *code)
            .collect::<Vec<_>>();
        codes.sort_unstable();
        let count = codes.len();
        codes.dedup();
        assert_eq!(codes.len(), count);
        // The forced names are the config's `support_languages`.
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/rumble/config.json"
        ))
        .unwrap();
        let mut published = config["support_languages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        published.sort();
        let mut ours = RUMBLE_FORCED_LANGUAGES
            .iter()
            .map(|(_, name)| (*name).to_string())
            .collect::<Vec<_>>();
        ours.sort();
        assert_eq!(ours, published);
    }

    #[test]
    fn the_default_style_is_the_templates() {
        let style = RumbleTranscriptStyle::default();
        assert!(style.digits && !style.fillers && style.verbatim);
        // The published template's defaults.
        let template = include_str!("../../../../tests/fixtures/rumble/chat_template.jinja");
        assert!(template.contains("digits if digits is defined else true"));
        assert!(template.contains("filler if filler is defined else false"));
        assert!(template.contains("verbatim if verbatim is defined else true"));
    }
}

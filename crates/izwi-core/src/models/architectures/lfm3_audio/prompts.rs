//! LFM3-Audio task prompts, rendered as the checkpoint's chat template
//! renders them (`continue_final_message` for a prefilled assistant turn),
//! and the languages and voices they name.

use crate::error::{Error, Result};

/// Speech-recognition languages: (ISO 639-1 code, the English name the
/// `Perform ASR in <language>.` prompt takes).
pub const LFM3_AUDIO_ASR_LANGUAGES: [(&str, &str); 18] = [
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// Speech-synthesis locales: (BCP 47 tag, the locale the
/// `Perform TTS in <locale> using a <gender> voice.` prompt takes). English
/// comes as American and British; the bare `en` is American.
pub const LFM3_AUDIO_TTS_LOCALES: [(&str, &str); 19] = [
    ("en-US", "American English"),
    ("en-GB", "British English"),
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// The built-in voices: the prompt's two genders.
pub const LFM3_AUDIO_VOICES: [&str; 2] = ["Female", "Male"];

/// The system prompt of an interleaved voice reply.
pub const LFM3_AUDIO_INTERLEAVED_SYSTEM_PROMPT: &str = "Respond with interleaved text and audio.";

/// A TTS request's voice: the prompt's gender and locale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lfm3AudioVoice {
    pub female: bool,
    /// The locale's prompt name, for example "British English".
    pub locale: &'static str,
}

impl Lfm3AudioVoice {
    pub const DEFAULT: Self = Self {
        female: true,
        locale: "American English",
    };
    /// The [`LFM3_AUDIO_TTS_LOCALES`] tag of [`Self::DEFAULT`]'s locale.
    pub const DEFAULT_LOCALE_TAG: &'static str = "en-US";

    /// Resolve a voice name and an optional language. The voice is `Female`
    /// or `Male` (any case), or an LFM2.5-style `US Female` / `UK Male` that
    /// also names the English locale; `language` is a tag from
    /// [`LFM3_AUDIO_TTS_LOCALES`] (case-insensitive, `_` or `-`), an English
    /// locale name, or a bare `en`.
    pub fn resolve(voice: Option<&str>, language: Option<&str>) -> Result<Self> {
        let mut resolved = Self::DEFAULT;
        if let Some(voice) = voice.map(str::trim).filter(|voice| !voice.is_empty()) {
            let lower = voice.to_ascii_lowercase();
            let (accent, gender) = match lower.split_once(' ') {
                Some((accent, gender)) => (Some(accent), gender),
                None => (None, lower.as_str()),
            };
            resolved.female = match gender {
                "female" => true,
                "male" => false,
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "LFM3-Audio voice must be Female or Male, got {voice:?}"
                    )))
                }
            };
            resolved.locale = match accent {
                None => resolved.locale,
                Some("us") => "American English",
                Some("uk") | Some("gb") => "British English",
                Some(_) => {
                    return Err(Error::InvalidInput(format!(
                        "LFM3-Audio voice must be Female or Male, got {voice:?}"
                    )))
                }
            };
        }
        if let Some(language) = language
            .map(str::trim)
            .filter(|language| !language.is_empty())
        {
            resolved.locale = tts_locale(language)?;
        }
        Ok(resolved)
    }

    fn gender(self) -> &'static str {
        if self.female {
            "female"
        } else {
            "male"
        }
    }
}

/// The prompt locale of a TTS language tag or name.
pub fn tts_locale(language: &str) -> Result<&'static str> {
    let normalized = language.trim().replace('_', "-").to_ascii_lowercase();
    if normalized == "en" || normalized == "english" {
        return Ok("American English");
    }
    LFM3_AUDIO_TTS_LOCALES
        .iter()
        .find(|(tag, name)| {
            tag.to_ascii_lowercase() == normalized || name.to_ascii_lowercase() == normalized
        })
        .map(|(_, name)| *name)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "LFM3-Audio speaks {}; unsupported language {language:?}",
                LFM3_AUDIO_TTS_LOCALES
                    .iter()
                    .map(|(tag, _)| *tag)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// The ASR prompt language of an ISO code or English name. Regional tags
/// (`en-GB`, `pt-BR`) take their base language.
pub fn asr_language(language: &str) -> Result<&'static str> {
    let normalized = language.trim().replace('_', "-").to_ascii_lowercase();
    let base = normalized.split('-').next().unwrap_or_default();
    LFM3_AUDIO_ASR_LANGUAGES
        .iter()
        .find(|(code, name)| *code == base || name.to_ascii_lowercase() == normalized)
        .map(|(_, name)| *name)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "LFM3-Audio transcribes {}; unsupported language {language:?}",
                LFM3_AUDIO_ASR_LANGUAGES
                    .iter()
                    .map(|(code, _)| *code)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// A prompt: text the tokenizer reads as one string, and runs of speech
/// placeholders the model fills with encoder frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PromptPart {
    Text(String),
    /// `<|audio|>` per encoder frame of the next input clip.
    Audio,
}

/// ChatML turns after `<|startoftext|>`, with `<|im_start|>assistant\n` and
/// `prefill` at the end (the template's `continue_final_message`).
pub(crate) struct PromptBuilder {
    parts: Vec<PromptPart>,
    text: String,
}

impl PromptBuilder {
    pub(crate) fn new() -> Self {
        Self {
            parts: Vec::new(),
            text: "<|startoftext|>".to_string(),
        }
    }

    pub(crate) fn turn(mut self, role: &str, content: &str) -> Self {
        self.text.push_str("<|im_start|>");
        self.text.push_str(role);
        self.text.push('\n');
        self.text.push_str(content);
        self.text.push_str("<|im_end|>\n");
        self
    }

    /// A user turn holding one speech clip and, after it, `text`.
    pub(crate) fn audio_turn(mut self, role: &str, text: &str) -> Self {
        self.text.push_str("<|im_start|>");
        self.text.push_str(role);
        self.text.push('\n');
        self.parts
            .push(PromptPart::Text(std::mem::take(&mut self.text)));
        self.parts.push(PromptPart::Audio);
        self.text.push_str(text);
        self.text.push_str("<|im_end|>\n");
        self
    }

    pub(crate) fn assistant(mut self, prefill: &str) -> Vec<PromptPart> {
        self.text.push_str("<|im_start|>assistant\n");
        self.text.push_str(prefill);
        self.parts.push(PromptPart::Text(self.text));
        self.parts
    }
}

/// `Perform ASR in <language>.`, or `Perform ASR.` to let the model pick.
pub(crate) fn asr_prompt(language: Option<&str>) -> Result<Vec<PromptPart>> {
    let system = match language
        .map(str::trim)
        .filter(|language| !language.is_empty())
    {
        Some(language) => format!("Perform ASR in {}.", asr_language(language)?),
        None => "Perform ASR.".to_string(),
    };
    Ok(PromptBuilder::new()
        .turn("system", &system)
        .audio_turn("user", "")
        .assistant(""))
}

/// `Perform live ASR.` (English only): a system turn and an open assistant
/// turn, which the stream's audio and text then fill.
pub(crate) fn live_asr_prompt() -> Vec<PromptPart> {
    PromptBuilder::new()
        .turn("system", "Perform live ASR.")
        .assistant("")
}

/// `Perform TTS in <locale> using a <gender> voice.`, prefilled with
/// `<|audio_start|>`.
pub(crate) fn tts_prompt(text: &str, voice: Lfm3AudioVoice) -> Vec<PromptPart> {
    let system = format!(
        "Perform TTS in {} using a {} voice.",
        voice.locale,
        voice.gender()
    );
    PromptBuilder::new()
        .turn("system", &system)
        .turn("user", text)
        .assistant("<|audio_start|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(parts: &[PromptPart]) -> String {
        parts
            .iter()
            .map(|part| match part {
                PromptPart::Text(text) => text.as_str(),
                PromptPart::Audio => "[AUDIO]",
            })
            .collect()
    }

    #[test]
    fn the_default_locale_tag_names_the_default_locale() {
        assert_eq!(
            tts_locale(Lfm3AudioVoice::DEFAULT_LOCALE_TAG).unwrap(),
            Lfm3AudioVoice::DEFAULT.locale
        );
    }

    #[test]
    fn live_asr_prompt_has_no_user_turn() {
        assert_eq!(
            text(&live_asr_prompt()),
            "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn asr_prompts_name_the_language_or_let_the_model_pick() {
        assert_eq!(
            text(&asr_prompt(Some("en")).unwrap()),
            "<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n\
             <|im_start|>user\n[AUDIO]<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            text(&asr_prompt(None).unwrap()),
            "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n\
             <|im_start|>user\n[AUDIO]<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(text(&asr_prompt(Some("pt-BR")).unwrap()).contains("Perform ASR in Portuguese."));
        assert!(asr_prompt(Some("sw")).is_err());
    }

    #[test]
    fn tts_prompts_name_the_locale_and_gender_and_open_speech() {
        let voice = Lfm3AudioVoice::resolve(Some("male"), Some("en-GB")).unwrap();
        assert_eq!(
            text(&tts_prompt("Hello there", voice)),
            "<|startoftext|><|im_start|>system\nPerform TTS in British English using a male voice.<|im_end|>\n\
             <|im_start|>user\nHello there<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
        );
        assert_eq!(
            Lfm3AudioVoice::resolve(None, None).unwrap(),
            Lfm3AudioVoice::DEFAULT
        );
        assert_eq!(
            Lfm3AudioVoice::resolve(Some("UK Female"), None).unwrap(),
            Lfm3AudioVoice {
                female: true,
                locale: "British English"
            }
        );
        assert_eq!(
            Lfm3AudioVoice::resolve(Some("Female"), Some("ja"))
                .unwrap()
                .locale,
            "Japanese"
        );
        assert!(Lfm3AudioVoice::resolve(Some("Ryan"), None).is_err());
        assert!(Lfm3AudioVoice::resolve(None, Some("sw")).is_err());
    }
}

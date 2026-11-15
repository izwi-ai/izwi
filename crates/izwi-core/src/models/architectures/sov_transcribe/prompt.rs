//! The prompt: instruction text, the Gemma 4 chat turn and the audio span.
//!
//! A request is one user turn holding the instruction and then the audio,
//! and the generation prompt with thinking disabled (the empty thought
//! channel is part of what the model was trained to continue):
//!
//! ```text
//! <bos><|turn>user\n{instruction}<|audio>{375 x <|audio|> per window}<audio|><turn|>\n
//! <|turn>model\n<|channel>thought\n<channel|>
//! ```
//!
//! Everything the model can be asked for is plain instruction text: the
//! language hint, keywords and transcript style all use the model card's
//! wording. The control tokens are placed by id, never parsed out of the
//! instruction; an instruction that itself contains one is refused.

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

/// The reference processor's default (`default_transcription_prompt`):
/// spoken words only, no punctuation.
pub const VERBATIM_INSTRUCTION: &str =
    "Transcribe the audio verbatim, outputting only spoken words in sequence.";
/// The card's automatic-language instruction: a punctuated, capitalized
/// transcript in whatever language is spoken. Izwi's default.
pub const AUTO_LANGUAGE_INSTRUCTION: &str = "Transcribe the audio verbatim in the language's standard writing system, using punctuation and capitalization. Output only the transcript.";

/// `(code, name)` of the languages the model transcribes.
pub const LANGUAGES: [(&str, &str); 7] = [
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
];

/// The English name of a supported language given its code or name.
pub fn language_name(language: &str) -> Option<&'static str> {
    let wanted = language.trim().to_ascii_lowercase();
    let code = wanted.split(['-', '_']).next().unwrap_or("");
    LANGUAGES
        .iter()
        .find(|(c, name)| *c == code || name.to_ascii_lowercase() == wanted)
        .map(|(_, name)| *name)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SovPromptOptions {
    /// A whole instruction to use as it is (the API's `prompt`).
    pub instruction: Option<String>,
    /// A language code or name; none lets the model detect it.
    pub language: Option<String>,
    /// Names and terms the audio may contain.
    pub keywords: Vec<String>,
}

/// The instruction text for `options`.
pub fn render_instruction(options: &SovPromptOptions) -> Result<String> {
    let language = options
        .language
        .as_deref()
        .map(str::trim)
        .filter(|language| !language.is_empty() && !language.eq_ignore_ascii_case("auto"))
        .map(|language| {
            language_name(language).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "SOV-Transcribe-1 does not transcribe {language:?}; it supports {}",
                    LANGUAGES.map(|(code, _)| code).join(", ")
                ))
            })
        })
        .transpose()?;
    let keywords = options
        .keywords
        .iter()
        .map(|keyword| keyword.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|keyword| !keyword.is_empty())
        .collect::<Vec<_>>();
    let custom = options
        .instruction
        .as_deref()
        .map(str::trim)
        .filter(|instruction| !instruction.is_empty());
    let task = match (custom, language) {
        (Some(custom), _) => custom.to_string(),
        (None, Some(language)) => format!(
            "Transcribe the {language} audio into text, ensuring all punctuation marks are included."
        ),
        (None, None) if !keywords.is_empty() => "Transcribe the audio into text.".to_string(),
        (None, None) => AUTO_LANGUAGE_INSTRUCTION.to_string(),
    };
    if keywords.is_empty() {
        return Ok(task);
    }
    Ok(format!(
        "This audio may contain the following words or phrases: {}. {task}",
        keywords.join(", ")
    ))
}

/// The control tokens of the chat turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SovControlTokens {
    pub bos: u32,
    pub turn_start: u32,
    pub turn_end: u32,
    pub channel_start: u32,
    pub channel_end: u32,
    pub audio: u32,
    pub audio_begin: u32,
    pub audio_end: u32,
}

impl SovControlTokens {
    pub fn resolve(tokenizer: &Tokenizer, audio: u32, bos: u32) -> Result<Self> {
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                Error::ModelLoadError(format!("the SOV-Transcribe tokenizer has no {token}"))
            })
        };
        let tokens = Self {
            bos,
            turn_start: id("<|turn>")?,
            turn_end: id("<turn|>")?,
            channel_start: id("<|channel>")?,
            channel_end: id("<channel|>")?,
            audio,
            audio_begin: id("<|audio>")?,
            audio_end: id("<audio|>")?,
        };
        if id("<|audio|>")? != audio || id("<bos>")? != bos {
            return Err(Error::ModelLoadError(
                "the SOV-Transcribe tokenizer and config disagree on special ids".into(),
            ));
        }
        Ok(tokens)
    }

    fn all(&self) -> [u32; 8] {
        [
            self.bos,
            self.turn_start,
            self.turn_end,
            self.channel_start,
            self.channel_end,
            self.audio,
            self.audio_begin,
            self.audio_end,
        ]
    }
}

/// A prompt's token layout: text ids around `audio_tokens` positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLayout {
    pub before: Vec<u32>,
    pub audio_tokens: usize,
    pub after: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSlot {
    Text(u32),
    Audio(usize),
}

impl PromptLayout {
    pub fn len(&self) -> usize {
        self.before.len() + self.audio_tokens + self.after.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The prompt's ids with every audio position as `audio_id`.
    pub fn expanded_ids(&self, audio_id: u32) -> Vec<u32> {
        let mut ids = self.before.clone();
        ids.extend(std::iter::repeat_n(audio_id, self.audio_tokens));
        ids.extend(&self.after);
        ids
    }

    /// What position `position` holds.
    pub fn slot(&self, position: usize) -> PromptSlot {
        let audio_start = self.before.len();
        if position < audio_start {
            PromptSlot::Text(self.before[position])
        } else if position < audio_start + self.audio_tokens {
            PromptSlot::Audio(position - audio_start)
        } else {
            PromptSlot::Text(self.after[position - audio_start - self.audio_tokens])
        }
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|err| Error::TokenizationError(format!("SOV-Transcribe prompt: {err}")))?
        .get_ids()
        .to_vec())
}

/// The prompt for `instruction` over `audio_tokens` audio positions.
pub fn tokenize_prompt(
    tokenizer: &Tokenizer,
    tokens: &SovControlTokens,
    instruction: &str,
    audio_tokens: usize,
) -> Result<PromptLayout> {
    if audio_tokens == 0 {
        return Err(Error::InvalidInput(
            "the audio produced no audio tokens".into(),
        ));
    }
    let user = encode(tokenizer, &format!("user\n{}", instruction.trim()))?;
    if user.iter().any(|id| tokens.all().contains(id)) {
        return Err(Error::InvalidInput(
            "the transcription prompt must not contain chat or audio control tokens".into(),
        ));
    }
    let mut before = vec![tokens.bos, tokens.turn_start];
    before.extend(user);
    before.push(tokens.audio_begin);
    let mut after = vec![tokens.audio_end, tokens.turn_end];
    after.extend(encode(tokenizer, "\n")?);
    after.push(tokens.turn_start);
    after.extend(encode(tokenizer, "model\n")?);
    after.push(tokens.channel_start);
    after.extend(encode(tokenizer, "thought\n")?);
    after.push(tokens.channel_end);
    Ok(PromptLayout {
        before,
        audio_tokens,
        after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_cards_automatic_language_instruction() {
        assert_eq!(
            render_instruction(&SovPromptOptions::default()).unwrap(),
            AUTO_LANGUAGE_INSTRUCTION
        );
        let auto = SovPromptOptions {
            language: Some("auto".into()),
            ..Default::default()
        };
        assert_eq!(
            render_instruction(&auto).unwrap(),
            AUTO_LANGUAGE_INSTRUCTION
        );
    }

    #[test]
    fn language_and_keywords_use_the_cards_wording() {
        let german = SovPromptOptions {
            language: Some("de-DE".into()),
            ..Default::default()
        };
        assert_eq!(
            render_instruction(&german).unwrap(),
            "Transcribe the German audio into text, ensuring all punctuation marks are included."
        );
        let keywords = SovPromptOptions {
            keywords: vec!["Izwi".into(), "  Ada   Lovelace ".into(), " ".into()],
            ..Default::default()
        };
        assert_eq!(
            render_instruction(&keywords).unwrap(),
            "This audio may contain the following words or phrases: Izwi, Ada Lovelace. Transcribe the audio into text."
        );
        let custom = SovPromptOptions {
            instruction: Some("  Write numbers as digits. ".into()),
            language: Some("fr".into()),
            keywords: vec!["Izwi".into()],
        };
        assert_eq!(
            render_instruction(&custom).unwrap(),
            "This audio may contain the following words or phrases: Izwi. Write numbers as digits."
        );
    }

    #[test]
    fn an_unsupported_language_is_refused() {
        let japanese = SovPromptOptions {
            language: Some("ja".into()),
            ..Default::default()
        };
        let error = render_instruction(&japanese).unwrap_err().to_string();
        assert!(error.contains("en, fr, de, es, it, pt, nl"), "{error}");
        assert_eq!(language_name("Portuguese"), Some("Portuguese"));
        assert_eq!(language_name("NL"), Some("Dutch"));
    }
}

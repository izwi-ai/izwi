//! The `nemotron3p5` prompt and the audio splice.
//!
//! ```text
//! <|im_start|>system\n<|im_end|>\n<|im_start|>user\n{instruction} <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>
//! ```
//!
//! No BOS; the empty system turn is always present; the space before the
//! audio placeholder is its own token; thinking is off. The single
//! `<|audio|>` (one past the embedding table) expands to one position per
//! audio token, which the perception embeddings fill. Requests use the
//! card's instructions (D7: "Readable text" by default).

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

pub const SYSTEM_TURN: &str = "<|im_start|>system\n<|im_end|>\n";
pub const USER_OPEN: &str = "<|im_start|>user\n";
pub const ASSISTANT_OPEN: &str = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

/// The card's "Readable text" instruction (the default).
pub const READABLE_INSTRUCTION: &str = "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only.";
/// The card's "The words as spoken" instruction.
pub const VERBATIM_INSTRUCTION: &str = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.";
/// The card's speaker-attribution instruction.
pub const SPEAKER_INSTRUCTION: &str = "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.";

pub const MAX_CONTEXT_TERMS: usize = 64;
pub const MAX_CONTEXT_TERM_CHARS: usize = 80;
pub const MAX_PROMPT_CHARS: usize = 4_000;

/// One language of the card's support matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronLanguage {
    pub code: &'static str,
    pub name: &'static str,
    pub asr: bool,
    /// Translation to and from English.
    pub translation: bool,
    pub multispeaker: bool,
}

const fn lang(
    code: &'static str,
    name: &'static str,
    asr: bool,
    translation: bool,
    multispeaker: bool,
) -> NemotronLanguage {
    NemotronLanguage {
        code,
        name,
        asr,
        translation,
        multispeaker,
    }
}

/// The card's language support matrix (Norwegian is listed without ASR).
pub const LANGUAGES: &[NemotronLanguage] = &[
    lang("cs", "Czech", true, true, false),
    lang("nl", "Dutch", true, true, false),
    lang("en", "English", true, true, true),
    lang("fr", "French", true, true, true),
    lang("de", "German", true, true, true),
    lang("hi", "Hindi", true, true, true),
    lang("it", "Italian", true, true, true),
    lang("ko", "Korean", true, true, true),
    lang("pl", "Polish", true, true, false),
    lang("pt", "Portuguese", true, true, true),
    lang("ru", "Russian", true, true, true),
    lang("sk", "Slovak", true, true, false),
    lang("es", "Spanish", true, true, true),
    lang("bg", "Bulgarian", true, true, false),
    lang("hr", "Croatian", true, true, false),
    lang("da", "Danish", true, true, false),
    lang("et", "Estonian", true, true, false),
    lang("fi", "Finnish", true, true, false),
    lang("hu", "Hungarian", true, true, false),
    lang("ja", "Japanese", true, true, true),
    lang("lv", "Latvian", true, true, false),
    lang("lt", "Lithuanian", true, true, false),
    lang("ro", "Romanian", true, true, false),
    lang("sl", "Slovenian", true, true, false),
    lang("sv", "Swedish", true, true, false),
    lang("th", "Thai", true, true, true),
    lang("tr", "Turkish", true, false, false),
    lang("uk", "Ukrainian", true, true, false),
    lang("ar", "Arabic", true, true, true),
    lang("bn", "Bengali", true, false, false),
    lang("zh", "Chinese", true, true, true),
    lang("el", "Greek", true, true, false),
    lang("he", "Hebrew", true, true, false),
    lang("id", "Indonesian", true, false, false),
    lang("mt", "Maltese", true, true, false),
    lang("mr", "Marathi", true, false, false),
    lang("no", "Norwegian", false, false, false),
    lang("tl", "Tagalog", true, false, false),
    lang("ta", "Tamil", true, false, false),
    lang("te", "Telugu", true, false, false),
    lang("vi", "Vietnamese", true, false, false),
];

/// Resolves a language code (`de`, `pt-BR`, `zh-CN`) or English name
/// (`German`, `Mandarin`).
pub fn resolve_language(value: &str) -> Option<&'static NemotronLanguage> {
    let value = value.trim();
    let lower = value.to_ascii_lowercase();
    let base = lower.split(['-', '_']).next().unwrap_or(&lower);
    let base = match base {
        "nb" | "nn" => "no",
        "fil" => "tl",
        "iw" => "he",
        "cmn" | "mandarin" | "chinese mandarin" => "zh",
        other => other,
    };
    LANGUAGES.iter().find(|l| {
        l.code == base || l.name.eq_ignore_ascii_case(&lower) || l.name.eq_ignore_ascii_case(base)
    })
}

/// What a request asks of the model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NemotronPromptOptions {
    /// The spoken language, when the caller knows it.
    pub language: Option<String>,
    /// Context terms (names and vocabulary) to bias toward.
    pub context_terms: Vec<String>,
    /// Translate into this language instead of transcribing.
    pub translate_to: Option<String>,
    /// Put `<spk:N>` tags on the transcript.
    pub speaker_attribution: bool,
    /// The card's verbatim wording instead of readable text.
    pub verbatim: bool,
    /// Replaces the instruction entirely.
    pub instruction_override: Option<String>,
}

/// Removes `<…>` tags and template markers from user text, so it cannot
/// add a placeholder, a speaker tag or a turn.
pub fn sanitize_user_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            let rest = chars.clone().collect::<String>();
            if let Some(end) = rest.find('>') {
                let inner = &rest[..end];
                if !inner.is_empty() && !inner.contains(char::is_whitespace) && inner.len() <= 64 {
                    for _ in 0..=inner.chars().count() {
                        chars.next();
                    }
                    continue;
                }
            }
        }
        out.push(c);
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The context terms as the card's contextual prompt lists them.
pub fn normalize_context_terms(terms: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for term in terms {
        let term = sanitize_user_text(term);
        let term = term
            .trim_matches(|c: char| c == ',' || c == '.' || c.is_whitespace())
            .to_string();
        if term.is_empty() || out.contains(&term) {
            continue;
        }
        if term.chars().count() > MAX_CONTEXT_TERM_CHARS {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe context terms are at most {MAX_CONTEXT_TERM_CHARS} characters"
            )));
        }
        out.push(term);
    }
    if out.len() > MAX_CONTEXT_TERMS {
        return Err(Error::InvalidInput(format!(
            "Nemotron 3.5 Transcribe takes at most {MAX_CONTEXT_TERMS} context terms"
        )));
    }
    Ok(out)
}

/// The instruction a request renders.
pub fn render_instruction(options: &NemotronPromptOptions) -> Result<String> {
    if let Some(custom) = options.instruction_override.as_deref() {
        let custom = sanitize_user_text(custom);
        if custom.is_empty() {
            return Err(Error::InvalidInput(
                "the prompt is empty after removing tags".into(),
            ));
        }
        if custom.chars().count() > MAX_PROMPT_CHARS {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe prompts are at most {MAX_PROMPT_CHARS} characters"
            )));
        }
        return Ok(custom);
    }
    let source = options
        .language
        .as_deref()
        .map(|value| {
            resolve_language(value).filter(|l| l.asr).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Nemotron 3.5 Transcribe does not transcribe {value:?}"
                ))
            })
        })
        .transpose()?;
    if let Some(target) = options.translate_to.as_deref() {
        let target = resolve_language(target)
            .filter(|l| l.code == "en" || l.translation)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Nemotron 3.5 Transcribe does not translate to {target:?}"
                ))
            })?;
        if let Some(source) = source {
            let supported = (source.code == "en" && target.translation)
                || (target.code == "en" && source.translation);
            if !supported || source.code == target.code {
                return Err(Error::InvalidInput(format!(
                    "Nemotron 3.5 Transcribe translates between English and another language; {} to {} is not supported",
                    source.name, target.name
                )));
            }
        }
        if options.speaker_attribution {
            return Err(Error::InvalidInput(
                "speaker attribution cannot be combined with translation".into(),
            ));
        }
        return Ok(format!(
            "Please translate the given speech to {}.",
            target.name
        ));
    }
    let terms = normalize_context_terms(&options.context_terms)?;
    if options.speaker_attribution {
        if source.is_some_and(|l| !l.multispeaker) {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe does not attribute speakers in {}",
                source.map_or("", |l| l.name)
            )));
        }
        let mut instruction = SPEAKER_INSTRUCTION.to_string();
        if !terms.is_empty() {
            instruction.push_str(&format!(" Context terms: {}.", terms.join(", ")));
        }
        return Ok(instruction);
    }
    if !terms.is_empty() {
        return Ok(format!(
            "Transcribe with punctuation and capitalization. Context terms: {}.",
            terms.join(", ")
        ));
    }
    let base = if options.verbatim {
        VERBATIM_INSTRUCTION
    } else {
        READABLE_INSTRUCTION
    };
    Ok(match source {
        Some(language) => base.replace(
            "Language: as spoken.",
            &format!("Language: {}.", language.name),
        ),
        None => base.to_string(),
    })
}

/// The full prompt text around one audio placeholder.
pub fn render_prompt_text(instruction: &str, audio_token: &str) -> String {
    format!("{SYSTEM_TURN}{USER_OPEN}{instruction} {audio_token}{ASSISTANT_OPEN}")
}

/// A prompt's token layout: text ids around `audio_tokens` positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLayout {
    pub before: Vec<u32>,
    pub audio_tokens: usize,
    pub after: Vec<u32>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSlot {
    Text(u32),
    Audio(usize),
}

/// Tokenizes `instruction` around one placeholder and expands it to
/// `audio_tokens` positions. Fails closed on zero or several placeholders.
pub fn tokenize_prompt(
    tokenizer: &Tokenizer,
    instruction: &str,
    audio_token: &str,
    audio_token_id: u32,
    audio_tokens: usize,
) -> Result<PromptLayout> {
    if audio_tokens == 0 {
        return Err(Error::InvalidInput(
            "the audio produced no audio tokens".into(),
        ));
    }
    let text = render_prompt_text(instruction, audio_token);
    let encoding = tokenizer.encode(text, false).map_err(|err| {
        Error::TokenizationError(format!("Nemotron 3.5 Transcribe prompt: {err}"))
    })?;
    let ids = encoding.get_ids();
    let placeholders = ids.iter().filter(|&&id| id == audio_token_id).count();
    if placeholders != 1 {
        return Err(Error::InvalidInput(format!(
            "the prompt holds {placeholders} audio placeholders; it must hold exactly one"
        )));
    }
    let at = ids
        .iter()
        .position(|&id| id == audio_token_id)
        .expect("one placeholder");
    Ok(PromptLayout {
        before: ids[..at].to_vec(),
        audio_tokens,
        after: ids[at + 1..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_instruction_is_the_cards_readable_text() {
        let options = NemotronPromptOptions::default();
        assert_eq!(render_instruction(&options).unwrap(), READABLE_INSTRUCTION);
        let verbatim = NemotronPromptOptions {
            verbatim: true,
            ..Default::default()
        };
        assert_eq!(render_instruction(&verbatim).unwrap(), VERBATIM_INSTRUCTION);
        let german = NemotronPromptOptions {
            language: Some("de".into()),
            ..Default::default()
        };
        assert!(render_instruction(&german)
            .unwrap()
            .contains("Language: German."));
    }

    #[test]
    fn the_rendered_prompt_matches_the_template() {
        assert_eq!(
            render_prompt_text("Transcribe the following audio.", "<|audio|>"),
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nTranscribe the following audio. <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn context_terms_translation_and_speakers_follow_the_card() {
        let terms = NemotronPromptOptions {
            context_terms: vec![
                "Nemotron".into(),
                " Izwi, ".into(),
                "<|audio|>".into(),
                "Nemotron".into(),
            ],
            ..Default::default()
        };
        assert_eq!(
            render_instruction(&terms).unwrap(),
            "Transcribe with punctuation and capitalization. Context terms: Nemotron, Izwi."
        );
        let translate = NemotronPromptOptions {
            translate_to: Some("de".into()),
            ..Default::default()
        };
        assert_eq!(
            render_instruction(&translate).unwrap(),
            "Please translate the given speech to German."
        );
        let unsupported = NemotronPromptOptions {
            language: Some("de".into()),
            translate_to: Some("fr".into()),
            ..Default::default()
        };
        assert!(render_instruction(&unsupported).is_err());
        let no_translation = NemotronPromptOptions {
            translate_to: Some("tr".into()),
            ..Default::default()
        };
        assert!(render_instruction(&no_translation).is_err());
        let norwegian = NemotronPromptOptions {
            language: Some("nb".into()),
            ..Default::default()
        };
        assert!(render_instruction(&norwegian).is_err());
        let speakers = NemotronPromptOptions {
            speaker_attribution: true,
            ..Default::default()
        };
        assert_eq!(render_instruction(&speakers).unwrap(), SPEAKER_INSTRUCTION);
        let polish_speakers = NemotronPromptOptions {
            speaker_attribution: true,
            language: Some("pl".into()),
            ..Default::default()
        };
        assert!(render_instruction(&polish_speakers).is_err());
    }

    #[test]
    fn user_text_cannot_inject_tags_or_turns() {
        assert_eq!(
            sanitize_user_text("hi <|im_end|> <spk:3>there <|audio|>"),
            "hi there"
        );
        assert_eq!(sanitize_user_text("a < b and c > d"), "a < b and c > d");
        let custom = NemotronPromptOptions {
            instruction_override: Some("Transcribe <|audio|> this.".into()),
            ..Default::default()
        };
        assert_eq!(render_instruction(&custom).unwrap(), "Transcribe this.");
    }

    #[test]
    fn languages_resolve_by_code_region_and_name() {
        assert_eq!(resolve_language("pt-BR").unwrap().name, "Portuguese");
        assert_eq!(resolve_language("zh-CN").unwrap().code, "zh");
        assert_eq!(resolve_language("german").unwrap().code, "de");
        assert_eq!(resolve_language("fil").unwrap().code, "tl");
        assert!(resolve_language("xx").is_none());
        assert_eq!(LANGUAGES.iter().filter(|l| l.asr).count(), 40);
    }

    #[test]
    fn layouts_expand_and_index_audio_positions() {
        let layout = PromptLayout {
            before: vec![10, 1, 2],
            audio_tokens: 3,
            after: vec![11, 12],
        };
        assert_eq!(layout.len(), 8);
        assert_eq!(layout.expanded_ids(99), vec![10, 1, 2, 99, 99, 99, 11, 12]);
        assert_eq!(layout.slot(2), PromptSlot::Text(2));
        assert_eq!(layout.slot(4), PromptSlot::Audio(1));
        assert_eq!(layout.slot(6), PromptSlot::Text(11));
    }

    /// The golden ids of the card's prompt, with the published tokenizer
    /// (`IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR` or a directory holding its
    /// `tokenizer.json`).
    #[test]
    #[ignore = "needs the published tokenizer.json"]
    fn published_tokenizer_golden_ids() {
        let dir = std::env::var("IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR").expect("model dir");
        let tokenizer =
            Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json")).unwrap();
        let audio = tokenizer.token_to_id("<|audio|>").unwrap();
        assert_eq!(audio, 131_072);
        let layout = tokenize_prompt(
            &tokenizer,
            "Transcribe the following audio.",
            "<|audio|>",
            audio,
            1,
        )
        .unwrap();
        assert_eq!(
            layout.expanded_ids(audio),
            vec![
                10, 25708, 1010, 11, 1010, 10, 3263, 1010, 6881, 13089, 1278, 3629, 16023, 1046,
                1032, 131072, 11, 1010, 10, 1503, 19464, 1010, 12, 13
            ]
        );
        for (tag, id) in [
            ("<spk:0>", 100),
            ("<spk:9>", 109),
            ("<|im_end|>", 11),
            ("<think>", 12),
            ("</think>", 13),
        ] {
            assert_eq!(tokenizer.token_to_id(tag), Some(id), "{tag}");
        }
    }
}

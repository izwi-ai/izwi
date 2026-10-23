//! Prompt assembly for Granite Speech 5.0, following the model card's
//! "Preferred prompt by task" table: raw or punctuated transcription, a
//! language-ID sentence for multilingual transcription, speech translation,
//! and keyword lists of up to 2,000 terms appended to either task.
//!
//! The chat template is `USER: {content}\n ASSISTANT:` with the audio token
//! leading the user content.

use crate::error::{Error, Result};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::config::{
    GraniteSpeech5Config, GraniteSpeech5ProcessorConfig, GraniteSpeech5TokenizerConfig,
};

/// Punctuated, capitalised transcription (the default).
pub const GRANITE_SPEECH5_PUNCTUATED_ASR_INSTRUCTION: &str =
    "transcribe the speech with proper punctuation and capitalization.";
/// Raw transcription, without punctuation or capitalization.
pub const GRANITE_SPEECH5_RAW_ASR_INSTRUCTION: &str =
    "can you transcribe the speech into a written format?";
/// The instruction a keyword list follows.
pub const GRANITE_SPEECH5_KEYWORD_ASR_INSTRUCTION: &str = "transcribe the speech to text.";
/// Most keywords one request may bias toward (the model card's limit).
pub const GRANITE_SPEECH5_MAX_KEYWORDS: usize = 2_000;
/// Longest keyword, in characters.
pub const GRANITE_SPEECH5_MAX_KEYWORD_CHARS: usize = 128;

/// Openings of the instructions the card uses. A request `prompt` that
/// starts with one (or with the audio token) replaces the default
/// instruction; other prompt text is context the model was not trained to
/// follow and is ignored.
const INSTRUCTION_STEMS: &[&str] = &[
    "transcribe the speech",
    "can you transcribe",
    "translate the speech",
    "the speech is in",
];
/// The `prompt` form for keyword lists (`keywords: Izwi, Granite`).
const KEYWORDS_PREFIX: &str = "keywords:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraniteSpeech5Language {
    English,
    French,
    German,
    Spanish,
    Portuguese,
    Japanese,
    Italian,
    Mandarin,
}

/// Languages 5.0 transcribes (and translates from).
pub const GRANITE_SPEECH5_SOURCE_LANGUAGES: &[GraniteSpeech5Language] = &[
    GraniteSpeech5Language::English,
    GraniteSpeech5Language::French,
    GraniteSpeech5Language::German,
    GraniteSpeech5Language::Spanish,
    GraniteSpeech5Language::Portuguese,
    GraniteSpeech5Language::Japanese,
];

/// Languages 5.0 translates into: English from the other source languages,
/// and English into those plus Italian and Mandarin.
pub const GRANITE_SPEECH5_TRANSLATION_TARGETS: &[GraniteSpeech5Language] = &[
    GraniteSpeech5Language::English,
    GraniteSpeech5Language::French,
    GraniteSpeech5Language::German,
    GraniteSpeech5Language::Spanish,
    GraniteSpeech5Language::Portuguese,
    GraniteSpeech5Language::Japanese,
    GraniteSpeech5Language::Italian,
    GraniteSpeech5Language::Mandarin,
];

impl GraniteSpeech5Language {
    /// ISO 639-1 and 639-2 codes and English names, case-insensitive.
    pub fn from_alias(raw: &str) -> Option<Self> {
        Some(match raw.trim().to_lowercase().as_str() {
            "en" | "eng" | "english" => Self::English,
            "fr" | "fra" | "fre" | "french" => Self::French,
            "de" | "deu" | "ger" | "german" => Self::German,
            "es" | "spa" | "spanish" => Self::Spanish,
            "pt" | "por" | "portuguese" => Self::Portuguese,
            "ja" | "jpn" | "japanese" => Self::Japanese,
            "it" | "ita" | "italian" => Self::Italian,
            "zh" | "zho" | "chi" | "chinese" | "mandarin" | "mandarin chinese" => Self::Mandarin,
            _ => return None,
        })
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::French => "fr",
            Self::German => "de",
            Self::Spanish => "es",
            Self::Portuguese => "pt",
            Self::Japanese => "ja",
            Self::Italian => "it",
            Self::Mandarin => "zh",
        }
    }

    /// The name the card's prompts use.
    pub const fn label(self) -> &'static str {
        match self {
            Self::English => "English",
            Self::French => "French",
            Self::German => "German",
            Self::Spanish => "Spanish",
            Self::Portuguese => "Portuguese",
            Self::Japanese => "Japanese",
            Self::Italian => "Italian",
            Self::Mandarin => "Mandarin",
        }
    }
}

fn supported_codes(languages: &[GraniteSpeech5Language]) -> String {
    let mut codes = languages
        .iter()
        .map(|language| language.code())
        .collect::<Vec<_>>();
    codes.sort_unstable();
    codes.join(", ")
}

/// A request's spoken language, which 5.0 must have been trained to
/// transcribe. Empty means none.
pub fn resolve_granite_speech5_source_language(
    raw: Option<&str>,
) -> Result<Option<GraniteSpeech5Language>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match GraniteSpeech5Language::from_alias(raw) {
        Some(language) if GRANITE_SPEECH5_SOURCE_LANGUAGES.contains(&language) => {
            Ok(Some(language))
        }
        _ => Err(Error::InvalidInput(format!(
            "Granite Speech 5.0 does not transcribe language '{raw}'; supported languages: {}",
            supported_codes(GRANITE_SPEECH5_SOURCE_LANGUAGES)
        ))),
    }
}

/// A translation target 5.0 supports, from the request's spoken language
/// when it names one: into English from the other source languages, or from
/// English into the rest.
pub fn resolve_granite_speech5_translation(
    source: Option<GraniteSpeech5Language>,
    raw_target: &str,
) -> Result<GraniteSpeech5Language> {
    let target = GraniteSpeech5Language::from_alias(raw_target)
        .filter(|language| GRANITE_SPEECH5_TRANSLATION_TARGETS.contains(language))
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Granite Speech 5.0 does not translate into '{}'; supported targets: {}",
                raw_target.trim(),
                supported_codes(GRANITE_SPEECH5_TRANSLATION_TARGETS)
            ))
        })?;
    let Some(source) = source else {
        return Ok(target);
    };
    if source == target {
        return Err(Error::InvalidInput(format!(
            "Granite Speech 5.0 translation source and target are both {}; omit the translation to transcribe",
            source.code()
        )));
    }
    let english = GraniteSpeech5Language::English;
    if source != english && target != english {
        return Err(Error::InvalidInput(format!(
            "Granite Speech 5.0 translates to or from English, not {}->{}",
            source.code(),
            target.code()
        )));
    }
    Ok(target)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeech5Task {
    Transcribe,
    Translate { target: GraniteSpeech5Language },
}

impl GraniteSpeech5Task {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate { .. } => "translate",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5PromptOptions {
    pub task: GraniteSpeech5Task,
    /// The spoken language. For transcription it adds the card's
    /// language-ID sentence; for translation it only checks the pair.
    pub source_language: Option<String>,
    /// Ask for punctuation and capitalization.
    pub punctuate: bool,
    pub keywords: Vec<String>,
    /// A request's free-form prompt, classified by
    /// [`classify_granite_speech5_prompt`].
    pub custom_prompt: Option<String>,
}

impl Default for GraniteSpeech5PromptOptions {
    fn default() -> Self {
        Self {
            task: GraniteSpeech5Task::Transcribe,
            source_language: None,
            punctuate: true,
            keywords: Vec::new(),
            custom_prompt: None,
        }
    }
}

/// What a request's free-form `prompt` means to 5.0. OpenAI's `prompt` is
/// context text (vocabulary, style, the previous segment), not an
/// instruction; replacing the trained instruction with it degrades the
/// transcript, so only recognisable instructions override it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraniteSpeech5PromptOverride {
    Instruction(String),
    Keywords(Vec<String>),
    Ignored,
}

pub fn classify_granite_speech5_prompt(
    raw: &str,
    audio_token: &str,
) -> GraniteSpeech5PromptOverride {
    let prompt = raw.trim();
    let starts_with = |prefix: &str| {
        prompt
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
    };
    if prompt.starts_with(audio_token) || INSTRUCTION_STEMS.iter().any(|stem| starts_with(stem)) {
        return GraniteSpeech5PromptOverride::Instruction(prompt.to_string());
    }
    if starts_with(KEYWORDS_PREFIX) {
        let keywords = prompt[KEYWORDS_PREFIX.len()..]
            .split([',', '\n'])
            .map(str::trim)
            .filter(|keyword| !keyword.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !keywords.is_empty() {
            return GraniteSpeech5PromptOverride::Keywords(keywords);
        }
    }
    GraniteSpeech5PromptOverride::Ignored
}

/// Collapse internal whitespace, drop empty entries and keep the first
/// occurrence of each keyword, then enforce the card's limits.
pub fn normalize_granite_speech5_keywords(
    keywords: &[String],
    audio_token: &str,
) -> Result<Vec<String>> {
    let mut normalized = Vec::<String>::new();
    for keyword in keywords {
        let cleaned = keyword.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.is_empty() || normalized.contains(&cleaned) {
            continue;
        }
        if cleaned.contains(audio_token) {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 keywords must not contain {audio_token}"
            )));
        }
        if cleaned.chars().count() > GRANITE_SPEECH5_MAX_KEYWORD_CHARS {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 keywords are at most {GRANITE_SPEECH5_MAX_KEYWORD_CHARS} characters; got one of {}",
                cleaned.chars().count()
            )));
        }
        normalized.push(cleaned);
    }
    if normalized.len() > GRANITE_SPEECH5_MAX_KEYWORDS {
        return Err(Error::InvalidInput(format!(
            "Granite Speech 5.0 accepts at most {GRANITE_SPEECH5_MAX_KEYWORDS} keywords, got {}",
            normalized.len()
        )));
    }
    Ok(normalized)
}

/// IBM's `add_keyword_bias_instruction`.
fn add_keyword_bias(instruction: &str, keywords: &[String]) -> String {
    if keywords.is_empty() {
        return instruction.to_string();
    }
    let suffix = format!("Keywords: {}", keywords.join(", "));
    let base = instruction.trim_end();
    if base.is_empty() {
        suffix
    } else if base.ends_with(['.', '!', '?']) {
        format!("{base} {suffix}")
    } else {
        format!("{base}. {suffix}")
    }
}

/// A rendered prompt before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5RenderedPrompt {
    pub text: String,
    /// The instruction after the audio token, keywords included.
    pub instruction: String,
    /// The request prompt was context text, not an instruction.
    pub prompt_ignored: bool,
    pub keyword_count: usize,
}

/// Render the full prompt text, rejecting a language, translation pair or
/// keyword list 5.0 was not trained for.
pub fn render_granite_speech5_prompt(
    audio_token: &str,
    options: &GraniteSpeech5PromptOptions,
) -> Result<GraniteSpeech5RenderedPrompt> {
    let source = resolve_granite_speech5_source_language(options.source_language.as_deref())?;
    let mut keywords = options.keywords.clone();
    let mut custom = None;
    let mut prompt_ignored = false;
    if let Some(raw) = options
        .custom_prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        match classify_granite_speech5_prompt(raw, audio_token) {
            GraniteSpeech5PromptOverride::Instruction(prompt) => custom = Some(prompt),
            GraniteSpeech5PromptOverride::Keywords(extra) => keywords.extend(extra),
            GraniteSpeech5PromptOverride::Ignored => prompt_ignored = true,
        }
    }
    let keywords = normalize_granite_speech5_keywords(&keywords, audio_token)?;
    let custom = custom
        .as_deref()
        .map(|prompt| {
            let prompt = prompt.trim();
            prompt
                .strip_prefix(audio_token)
                .map(str::trim)
                .unwrap_or(prompt)
        })
        .filter(|prompt| !prompt.is_empty());

    let instruction = match (custom, options.task) {
        (Some(prompt), _) if prompt.contains(audio_token) => {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 prompt may only place {audio_token} at its start"
            )));
        }
        (Some(_), GraniteSpeech5Task::Translate { .. }) => {
            return Err(Error::InvalidInput(
                "A Granite Speech 5.0 instruction prompt cannot be combined with a translation request; send keywords instead"
                    .to_string(),
            ));
        }
        // An explicit instruction is used as written.
        (Some(prompt), GraniteSpeech5Task::Transcribe) => prompt.to_string(),
        (None, GraniteSpeech5Task::Translate { target }) => {
            resolve_granite_speech5_translation(source, target.code())?;
            if options.punctuate && keywords.is_empty() {
                format!(
                    "translate the speech to {} with proper punctuation and capitalization.",
                    target.label()
                )
            } else {
                format!("translate the speech to {}.", target.label())
            }
        }
        (None, GraniteSpeech5Task::Transcribe) => {
            let base = if !keywords.is_empty() {
                GRANITE_SPEECH5_KEYWORD_ASR_INSTRUCTION
            } else if options.punctuate {
                GRANITE_SPEECH5_PUNCTUATED_ASR_INSTRUCTION
            } else {
                GRANITE_SPEECH5_RAW_ASR_INSTRUCTION
            };
            match source {
                Some(language) => format!(
                    "The speech is in {}. Transcribe it exactly; do not translate. {base}",
                    language.label()
                ),
                None => base.to_string(),
            }
        }
    };
    let instruction = add_keyword_bias(&instruction, &keywords);
    Ok(GraniteSpeech5RenderedPrompt {
        text: format!("USER: {audio_token}{instruction}\n ASSISTANT:"),
        instruction,
        prompt_ignored,
        keyword_count: keywords.len(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5SpecialTokens {
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub audio_token_id: u32,
    pub audio_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5Prompt {
    pub task: GraniteSpeech5Task,
    pub text: String,
    pub instruction: String,
    pub prompt_ignored: bool,
    pub keyword_count: usize,
    pub input_ids: Vec<u32>,
    /// Index of the single audio placeholder in `input_ids`.
    pub audio_token_position: usize,
}

pub struct GraniteSpeech5PromptTokenizer {
    tokenizer: Tokenizer,
    special_tokens: GraniteSpeech5SpecialTokens,
}

impl GraniteSpeech5PromptTokenizer {
    pub fn load(
        model_dir: &std::path::Path,
        config: &GraniteSpeech5Config,
        processor: &GraniteSpeech5ProcessorConfig,
        tokenizer_config: &GraniteSpeech5TokenizerConfig,
        eos_token_id: u32,
        pad_token_id: u32,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_path_with_expected_vocab(
            model_dir,
            Some(config.text_config.vocab_size),
        )?;
        let audio_token = if tokenizer_config.audio_token.trim().is_empty() {
            processor.audio_token.clone()
        } else {
            tokenizer_config.audio_token.clone()
        };
        let audio_token_id = tokenizer
            .token_to_id(&audio_token)
            .unwrap_or(config.audio_token_index);
        if audio_token_id != config.audio_token_index {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 tokenizer maps {audio_token} to {audio_token_id}, but the config's audio_token_index is {}",
                config.audio_token_index
            )));
        }
        Ok(Self {
            tokenizer,
            special_tokens: GraniteSpeech5SpecialTokens {
                eos_token_id,
                pad_token_id,
                audio_token_id,
                audio_token,
            },
        })
    }

    pub fn special_tokens(&self) -> &GraniteSpeech5SpecialTokens {
        &self.special_tokens
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    pub(crate) fn incremental_decoder(&self) -> IncrementalDecoder {
        IncrementalDecoder::new(true)
    }

    pub(crate) fn decode_incrementally(
        &self,
        decoder: &mut IncrementalDecoder,
        token: u32,
    ) -> Result<String> {
        self.tokenizer.decode_incrementally(decoder, token)
    }

    pub(crate) fn finish_incremental_decode(
        &self,
        decoder: &mut IncrementalDecoder,
    ) -> Result<String> {
        self.tokenizer.finish_incremental_decode(decoder)
    }

    pub fn build_prompt(
        &self,
        options: &GraniteSpeech5PromptOptions,
    ) -> Result<GraniteSpeech5Prompt> {
        let rendered = render_granite_speech5_prompt(&self.special_tokens.audio_token, options)?;
        let input_ids = self.encode(&rendered.text)?;
        let positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, token)| {
                (*token == self.special_tokens.audio_token_id).then_some(index)
            })
            .collect::<Vec<_>>();
        let [audio_token_position] = positions[..] else {
            return Err(Error::TokenizationError(format!(
                "Granite Speech 5.0 prompt must tokenize {} to exactly one id {}, found {}",
                self.special_tokens.audio_token,
                self.special_tokens.audio_token_id,
                positions.len()
            )));
        };
        Ok(GraniteSpeech5Prompt {
            task: options.task,
            text: rendered.text,
            instruction: rendered.instruction,
            prompt_ignored: rendered.prompt_ignored,
            keyword_count: rendered.keyword_count,
            input_ids,
            audio_token_position,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AUDIO: &str = "<|audio|>";

    fn render(options: GraniteSpeech5PromptOptions) -> Result<String> {
        render_granite_speech5_prompt(AUDIO, &options).map(|prompt| prompt.text)
    }

    fn options() -> GraniteSpeech5PromptOptions {
        GraniteSpeech5PromptOptions::default()
    }

    fn keywords(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    /// Expected strings are `chat_template.jinja` rendered by jinja2 with
    /// the card's prompts as the user content (`add_generation_prompt`
    /// changes nothing for this template).
    #[test]
    fn renders_the_model_card_prompts_through_the_chat_template() {
        assert_eq!(
            render(options()).unwrap(),
            "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                punctuate: false,
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                keywords: keywords(&["Izwi", "Granite"]),
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>transcribe the speech to text. Keywords: Izwi, Granite\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                source_language: Some("de".into()),
                punctuate: false,
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>The speech is in German. Transcribe it exactly; do not translate. can you transcribe the speech into a written format?\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                task: GraniteSpeech5Task::Translate {
                    target: GraniteSpeech5Language::French
                },
                punctuate: false,
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>translate the speech to French.\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                task: GraniteSpeech5Task::Translate {
                    target: GraniteSpeech5Language::English
                },
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>translate the speech to English with proper punctuation and capitalization.\n ASSISTANT:"
        );
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                task: GraniteSpeech5Task::Translate {
                    target: GraniteSpeech5Language::Mandarin
                },
                keywords: keywords(&["IBM"]),
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>translate the speech to Mandarin. Keywords: IBM\n ASSISTANT:"
        );
    }

    #[test]
    fn language_id_composes_with_the_selected_instruction_and_keywords() {
        let rendered = render_granite_speech5_prompt(
            AUDIO,
            &GraniteSpeech5PromptOptions {
                source_language: Some("Japanese".into()),
                keywords: keywords(&["  Izwi   AI ", "Izwi AI", ""]),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(
            rendered.instruction,
            "The speech is in Japanese. Transcribe it exactly; do not translate. transcribe the speech to text. Keywords: Izwi AI"
        );
        assert_eq!(rendered.keyword_count, 1);
        assert_eq!(
            render(GraniteSpeech5PromptOptions {
                source_language: Some("en".into()),
                ..options()
            })
            .unwrap(),
            "USER: <|audio|>The speech is in English. Transcribe it exactly; do not translate. transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
        );
    }

    #[test]
    fn request_prompts_override_only_as_instructions_or_keywords() {
        let instruction = render_granite_speech5_prompt(
            AUDIO,
            &GraniteSpeech5PromptOptions {
                custom_prompt: Some("<|audio|> Translate the speech to German.".into()),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(instruction.instruction, "Translate the speech to German.");
        assert!(!instruction.prompt_ignored);

        let listed = render_granite_speech5_prompt(
            AUDIO,
            &GraniteSpeech5PromptOptions {
                custom_prompt: Some("keywords: Izwi, Granite\nLennex".into()),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(
            listed.instruction,
            "transcribe the speech to text. Keywords: Izwi, Granite, Lennex"
        );

        let context = render_granite_speech5_prompt(
            AUDIO,
            &GraniteSpeech5PromptOptions {
                custom_prompt: Some("The previous segment ended with a question.".into()),
                ..options()
            },
        )
        .unwrap();
        assert!(context.prompt_ignored);
        assert_eq!(
            context.instruction,
            GRANITE_SPEECH5_PUNCTUATED_ASR_INSTRUCTION
        );

        assert!(render(GraniteSpeech5PromptOptions {
            custom_prompt: Some("transcribe the speech <|audio|> twice".into()),
            ..options()
        })
        .is_err());
        assert!(render(GraniteSpeech5PromptOptions {
            task: GraniteSpeech5Task::Translate {
                target: GraniteSpeech5Language::English
            },
            custom_prompt: Some("transcribe the speech to text.".into()),
            ..options()
        })
        .is_err());
    }

    #[test]
    fn languages_and_translation_pairs_follow_the_card() {
        use GraniteSpeech5Language::*;
        assert_eq!(
            resolve_granite_speech5_source_language(Some(" PT ")).unwrap(),
            Some(Portuguese)
        );
        assert_eq!(
            resolve_granite_speech5_source_language(Some("")).unwrap(),
            None
        );
        // Italian and Mandarin are translation targets only.
        for unsupported in ["it", "zh", "ko", "klingon"] {
            let err = resolve_granite_speech5_source_language(Some(unsupported))
                .unwrap_err()
                .to_string();
            assert!(err.contains("de, en, es, fr, ja, pt"), "{err}");
        }
        assert_eq!(
            resolve_granite_speech5_translation(None, "zh").unwrap(),
            Mandarin
        );
        assert_eq!(
            resolve_granite_speech5_translation(Some(Japanese), "english").unwrap(),
            English
        );
        assert_eq!(
            resolve_granite_speech5_translation(Some(English), "it").unwrap(),
            Italian
        );
        assert!(resolve_granite_speech5_translation(Some(French), "de")
            .unwrap_err()
            .to_string()
            .contains("to or from English"));
        assert!(resolve_granite_speech5_translation(Some(English), "en").is_err());
        assert!(resolve_granite_speech5_translation(None, "ko").is_err());

        // A translation request checks its pair against the spoken language.
        assert!(render(GraniteSpeech5PromptOptions {
            task: GraniteSpeech5Task::Translate { target: German },
            source_language: Some("fr".into()),
            ..options()
        })
        .is_err());
    }

    #[test]
    fn keyword_lists_are_normalized_and_bounded() {
        let many = (0..GRANITE_SPEECH5_MAX_KEYWORDS)
            .map(|index| format!("term {index}"))
            .collect::<Vec<_>>();
        assert_eq!(
            normalize_granite_speech5_keywords(&many, AUDIO)
                .unwrap()
                .len(),
            GRANITE_SPEECH5_MAX_KEYWORDS
        );
        let mut too_many = many.clone();
        too_many.push("one more".into());
        assert!(normalize_granite_speech5_keywords(&too_many, AUDIO)
            .unwrap_err()
            .to_string()
            .contains("at most 2000 keywords"));
        // Duplicates do not count toward the limit.
        let mut duplicated = many;
        duplicated.push("term   0".into());
        assert!(normalize_granite_speech5_keywords(&duplicated, AUDIO).is_ok());
        assert!(normalize_granite_speech5_keywords(&["x".repeat(129)], AUDIO).is_err());
        assert!(normalize_granite_speech5_keywords(&["a <|audio|>".into()], AUDIO).is_err());
    }
}

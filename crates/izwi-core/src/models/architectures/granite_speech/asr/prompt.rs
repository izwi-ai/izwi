//! Prompt assembly for Granite Speech 4.1, following IBM's reference library
//! (`ibm-granite/granite-speech`, `prompts.py` and `_models.py`) and the
//! model cards. The base and plus checkpoints take different templates and
//! instructions; each task runs on the checkpoint IBM trained it on.

use crate::error::{Error, Result};
use crate::models::architectures::granite_speech::asr::config::{
    GraniteSpeechCheckpointKind, GraniteSpeechConfig, GraniteSpeechProcessorConfig,
    GraniteSpeechTokenizerConfig,
};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Plus's training-time system prompt, verbatim. The dates are what the
/// model saw in training, not the current date.
pub const GRANITE_SPEECH_SYSTEM_PROMPT: &str = "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant";
/// Base's punctuated, capitalised ASR instruction (IBM's default).
pub const GRANITE_SPEECH_BASE_ASR_INSTRUCTION: &str =
    "transcribe the speech with proper punctuation and capitalization.";
/// Base's keyword-biased ASR instruction.
pub const GRANITE_SPEECH_KEYWORD_ASR_INSTRUCTION: &str = "transcribe the speech to text.";
/// Plus's ASR instruction; plus never punctuates.
pub const GRANITE_SPEECH_PLUS_ASR_INSTRUCTION: &str =
    "can you transcribe the speech into a written format?";
pub const GRANITE_SPEECH_SPEAKER_INSTRUCTION: &str = "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.";
pub const GRANITE_SPEECH_TIMESTAMP_INSTRUCTION: &str = "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";

/// Openings of the instructions IBM's cards and library use. A request
/// `prompt` that starts with one (or with the audio token) is a Granite
/// instruction, as in the base card's llama-server example.
const GRANITE_SPEECH_INSTRUCTION_STEMS: &[&str] = &[
    "transcribe the speech",
    "can you transcribe",
    "translate the speech",
    "Speaker attribution:",
    "Timestamps:",
];
/// izwi's documented keyword form (`keywords: Izwi, Granite Speech`).
const GRANITE_SPEECH_KEYWORDS_PREFIX: &str = "keywords:";

/// What a request's free-form `prompt` means to Granite. OpenAI's `prompt` is
/// context text (vocabulary, style, the previous segment), not an
/// instruction; replacing Granite's trained instruction with it degrades
/// the transcript, so only recognisable Granite instructions override it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraniteSpeechPromptOverride {
    /// A Granite instruction that replaces the default one.
    Instruction(String),
    /// Keywords for IBM's keyword-biasing prompt.
    Keywords(Vec<String>),
    /// Context the model was not trained to follow; the default instruction
    /// is kept and diagnostics report `prompt_ignored`.
    Ignored,
}

pub fn classify_granite_speech_prompt(raw: &str, audio_token: &str) -> GraniteSpeechPromptOverride {
    let prompt = raw.trim();
    let starts_with = |prefix: &str| {
        prompt
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
    };
    if prompt.starts_with(audio_token)
        || GRANITE_SPEECH_INSTRUCTION_STEMS
            .iter()
            .any(|stem| starts_with(stem))
    {
        return GraniteSpeechPromptOverride::Instruction(prompt.to_string());
    }
    if starts_with(GRANITE_SPEECH_KEYWORDS_PREFIX) {
        let keywords = prompt[GRANITE_SPEECH_KEYWORDS_PREFIX.len()..]
            .split([',', '\n'])
            .map(str::trim)
            .filter(|keyword| !keyword.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !keywords.is_empty() {
            return GraniteSpeechPromptOverride::Keywords(keywords);
        }
    }
    GraniteSpeechPromptOverride::Ignored
}

/// A language IBM's library knows, with its alias table and labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraniteSpeechLanguage {
    English,
    French,
    German,
    Spanish,
    Portuguese,
    Japanese,
    Italian,
    Mandarin,
}

impl GraniteSpeechLanguage {
    /// IBM's `LANGUAGE_ALIASES`: ISO 639-1 and 639-2 codes and English
    /// names, case-insensitive.
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

    /// IBM's `LANGUAGE_LABELS`, the form instructions name a language in.
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

const BASE_SOURCE_LANGUAGES: &[GraniteSpeechLanguage] = &[
    GraniteSpeechLanguage::German,
    GraniteSpeechLanguage::English,
    GraniteSpeechLanguage::Spanish,
    GraniteSpeechLanguage::French,
    GraniteSpeechLanguage::Japanese,
    GraniteSpeechLanguage::Portuguese,
];
const PLUS_SOURCE_LANGUAGES: &[GraniteSpeechLanguage] = &[
    GraniteSpeechLanguage::German,
    GraniteSpeechLanguage::English,
    GraniteSpeechLanguage::Spanish,
    GraniteSpeechLanguage::French,
    GraniteSpeechLanguage::Portuguese,
];

/// Base's translation pairs, IBM's `_BASE_TRANSLATION_PAIRS`: to and from
/// English for the ASR languages, plus English to Italian and Mandarin.
fn base_translation_pair_supported(
    source: GraniteSpeechLanguage,
    target: GraniteSpeechLanguage,
) -> bool {
    use GraniteSpeechLanguage::*;
    matches!(
        (source, target),
        (French | German | Spanish | Portuguese | Japanese, English)
            | (
                English,
                French | German | Spanish | Portuguese | Japanese | Italian | Mandarin
            )
    )
}

/// Which template and instruction set a checkpoint was trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechPromptProfile {
    /// `USER: <|audio|>{instruction}\n ASSISTANT:`, no system prompt.
    Base,
    /// Granite role tokens with the card's system prompt, `<|audio|> ` with a
    /// space, and an optional assistant prefix for incremental decoding.
    Plus,
}

impl GraniteSpeechPromptProfile {
    pub const fn for_checkpoint(kind: GraniteSpeechCheckpointKind) -> Self {
        match kind {
            GraniteSpeechCheckpointKind::Base => Self::Base,
            GraniteSpeechCheckpointKind::Plus => Self::Plus,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Plus => "plus",
        }
    }

    /// IBM's `source_languages` for the checkpoint.
    pub const fn source_languages(self) -> &'static [GraniteSpeechLanguage] {
        match self {
            Self::Base => BASE_SOURCE_LANGUAGES,
            Self::Plus => PLUS_SOURCE_LANGUAGES,
        }
    }

    /// Speaker turns and word timestamps are plus-only; translation is
    /// base-only.
    pub const fn supports_task(self, task: GraniteSpeechTask) -> bool {
        match task {
            GraniteSpeechTask::Asr => true,
            GraniteSpeechTask::SpeakerAttributed | GraniteSpeechTask::WordTimestamps => {
                matches!(self, Self::Plus)
            }
            GraniteSpeechTask::Translate { .. } => matches!(self, Self::Base),
        }
    }

    /// Incremental decoding through an assistant prefix is plus-only.
    pub const fn supports_prefix_text(self) -> bool {
        matches!(self, Self::Plus)
    }

    /// Resolve and validate a request's source language. Empty means none.
    pub fn source_language(self, raw: Option<&str>) -> Result<Option<GraniteSpeechLanguage>> {
        let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        match GraniteSpeechLanguage::from_alias(raw) {
            Some(language) if self.source_languages().contains(&language) => Ok(Some(language)),
            _ => {
                let mut supported = self
                    .source_languages()
                    .iter()
                    .map(|language| language.code())
                    .collect::<Vec<_>>();
                supported.sort_unstable();
                Err(Error::InvalidInput(format!(
                    "Granite Speech {} does not support source language '{raw}'; supported source languages: {}",
                    self.name(),
                    supported.join(", ")
                )))
            }
        }
    }

    const fn audio_separator(self) -> &'static str {
        match self {
            Self::Base => "",
            Self::Plus => " ",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechTask {
    Asr,
    SpeakerAttributed,
    WordTimestamps,
    Translate { target: GraniteSpeechLanguage },
}

impl GraniteSpeechTask {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::SpeakerAttributed => "speaker_attributed",
            Self::WordTimestamps => "word_timestamps",
            Self::Translate { .. } => "translate",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechPromptOptions {
    pub task: GraniteSpeechTask,
    pub language: Option<String>,
    pub keywords: Vec<String>,
    pub prefix_text: Option<String>,
    /// A request's free-form prompt, classified by
    /// [`classify_granite_speech_prompt`].
    pub custom_prompt: Option<String>,
}

impl Default for GraniteSpeechPromptOptions {
    fn default() -> Self {
        Self {
            task: GraniteSpeechTask::Asr,
            language: None,
            keywords: Vec::new(),
            prefix_text: None,
            custom_prompt: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechSpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub audio_token_id: u32,
    pub audio_token: String,
}

/// A rendered prompt before tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechRenderedPrompt {
    pub text: String,
    /// The instruction after the audio token, keywords included.
    pub instruction: String,
    /// The request prompt was context text, not a Granite instruction.
    pub prompt_ignored: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeechPrompt {
    pub profile: GraniteSpeechPromptProfile,
    pub task: GraniteSpeechTask,
    pub text: String,
    pub instruction: String,
    pub prompt_ignored: bool,
    pub input_ids: Vec<u32>,
    pub audio_token_positions: Vec<usize>,
    pub prefix_text_token_count: usize,
}

/// Render the full prompt text for `profile`, rejecting a task, language,
/// translation pair or prefix the checkpoint was not trained for.
pub fn render_granite_speech_prompt(
    profile: GraniteSpeechPromptProfile,
    audio_token: &str,
    options: &GraniteSpeechPromptOptions,
) -> Result<GraniteSpeechRenderedPrompt> {
    let language = profile.source_language(options.language.as_deref())?;
    validate_task(profile, options.task, language)?;
    let prefix = options
        .prefix_text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("");
    if !prefix.is_empty() && !profile.supports_prefix_text() {
        return Err(Error::InvalidInput(format!(
            "Granite Speech {} does not support prefix_text incremental decoding",
            profile.name()
        )));
    }

    let mut keywords = options.keywords.clone();
    let mut custom = None;
    let mut prompt_ignored = false;
    if let Some(raw) = options
        .custom_prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        match classify_granite_speech_prompt(raw, audio_token) {
            GraniteSpeechPromptOverride::Instruction(prompt) => custom = Some(prompt),
            GraniteSpeechPromptOverride::Keywords(extra) => keywords.extend(extra),
            GraniteSpeechPromptOverride::Ignored => prompt_ignored = true,
        }
    }
    let keywords = normalize_keywords(&keywords);
    let custom = custom
        .as_deref()
        .map(|prompt| strip_leading_audio_token(prompt, audio_token))
        .filter(|prompt| !prompt.is_empty());
    let instruction = match custom {
        Some(prompt) if prompt.contains(audio_token) => {
            return Err(Error::InvalidInput(format!(
                "Granite Speech prompt may only place {audio_token} at its start"
            )));
        }
        Some(prompt) => prompt.to_string(),
        None => default_instruction(profile, options.task, language, !keywords.is_empty()),
    };
    let instruction = add_keyword_bias(&instruction, &keywords);
    let content = format!("{audio_token}{}{instruction}", profile.audio_separator());
    let text = match profile {
        GraniteSpeechPromptProfile::Base => format!("USER: {content}\n ASSISTANT:"),
        GraniteSpeechPromptProfile::Plus => format!(
            "<|start_of_role|>system<|end_of_role|>{GRANITE_SPEECH_SYSTEM_PROMPT}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{content}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{prefix}"
        ),
    };
    Ok(GraniteSpeechRenderedPrompt {
        text,
        instruction,
        prompt_ignored,
    })
}

fn validate_task(
    profile: GraniteSpeechPromptProfile,
    task: GraniteSpeechTask,
    source: Option<GraniteSpeechLanguage>,
) -> Result<()> {
    if !profile.supports_task(task) {
        let checkpoint = match task {
            GraniteSpeechTask::Translate { .. } => "base",
            _ => "plus",
        };
        return Err(Error::InvalidInput(format!(
            "Granite Speech {} task requires the {checkpoint} checkpoint, not {}",
            task.name(),
            profile.name()
        )));
    }
    let GraniteSpeechTask::Translate { target } = task else {
        return Ok(());
    };
    let source = source.ok_or_else(|| {
        Error::InvalidInput(
            "Granite Speech translation requires an explicit source language".to_string(),
        )
    })?;
    if source == target {
        return Err(Error::InvalidInput(format!(
            "Granite Speech translation source and target are both {}",
            source.code()
        )));
    }
    if !base_translation_pair_supported(source, target) {
        return Err(Error::InvalidInput(format!(
            "Granite Speech does not support translation {}->{}",
            source.code(),
            target.code()
        )));
    }
    Ok(())
}

/// IBM's `default_instruction`, verbatim per checkpoint and task.
fn default_instruction(
    profile: GraniteSpeechPromptProfile,
    task: GraniteSpeechTask,
    language: Option<GraniteSpeechLanguage>,
    keyword_biased: bool,
) -> String {
    match (profile, task) {
        (_, GraniteSpeechTask::SpeakerAttributed) => GRANITE_SPEECH_SPEAKER_INSTRUCTION.into(),
        (_, GraniteSpeechTask::WordTimestamps) => GRANITE_SPEECH_TIMESTAMP_INSTRUCTION.into(),
        (_, GraniteSpeechTask::Translate { target }) if keyword_biased => {
            format!("translate the speech to {}.", target.label())
        }
        (_, GraniteSpeechTask::Translate { target }) => format!(
            "translate the speech to {} with proper punctuation and capitalization.",
            target.label()
        ),
        (GraniteSpeechPromptProfile::Plus, GraniteSpeechTask::Asr) => match language {
            None => GRANITE_SPEECH_PLUS_ASR_INSTRUCTION.into(),
            Some(language) => format!(
                "can you transcribe the {} speech into a written format?",
                language.label()
            ),
        },
        (GraniteSpeechPromptProfile::Base, GraniteSpeechTask::Asr) if keyword_biased => {
            GRANITE_SPEECH_KEYWORD_ASR_INSTRUCTION.into()
        }
        (GraniteSpeechPromptProfile::Base, GraniteSpeechTask::Asr) => match language {
            None => GRANITE_SPEECH_BASE_ASR_INSTRUCTION.into(),
            Some(language) => format!(
                "transcribe the {} speech with proper punctuation and capitalization.",
                language.label()
            ),
        },
    }
}

/// IBM's `_strip_audio_token`: an explicit prompt may carry a leading audio
/// token, which the template re-adds with the profile's separator.
fn strip_leading_audio_token<'a>(prompt: &'a str, audio_token: &str) -> &'a str {
    let prompt = prompt.trim();
    prompt
        .strip_prefix(audio_token)
        .map(str::trim)
        .unwrap_or(prompt)
}

/// IBM's `normalize_keyword_biases`: collapse internal whitespace and keep
/// the first occurrence of each keyword.
fn normalize_keywords(keywords: &[String]) -> Vec<String> {
    let mut normalized = Vec::<String>::new();
    for keyword in keywords {
        let cleaned = keyword.split_whitespace().collect::<Vec<_>>().join(" ");
        if !cleaned.is_empty() && !normalized.contains(&cleaned) {
            normalized.push(cleaned);
        }
    }
    normalized
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

pub struct GraniteSpeechPromptTokenizer {
    tokenizer: Tokenizer,
    special_tokens: GraniteSpeechSpecialTokens,
    profile: GraniteSpeechPromptProfile,
}

impl GraniteSpeechPromptTokenizer {
    pub fn load(
        model_dir: &std::path::Path,
        config: &GraniteSpeechConfig,
        processor: &GraniteSpeechProcessorConfig,
        tokenizer_config: &GraniteSpeechTokenizerConfig,
    ) -> Result<Self> {
        let profile = GraniteSpeechPromptProfile::for_checkpoint(config.checkpoint_kind()?);
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
        let special_tokens = GraniteSpeechSpecialTokens {
            bos_token_id: config.text_config.bos_token_id,
            eos_token_id: config.text_config.eos_token_id,
            pad_token_id: config.text_config.pad_token_id,
            audio_token_id,
            audio_token,
        };
        Ok(Self {
            tokenizer,
            special_tokens,
            profile,
        })
    }

    pub fn special_tokens(&self) -> &GraniteSpeechSpecialTokens {
        &self.special_tokens
    }

    pub fn profile(&self) -> GraniteSpeechPromptProfile {
        self.profile
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
        options: &GraniteSpeechPromptOptions,
    ) -> Result<GraniteSpeechPrompt> {
        let rendered =
            render_granite_speech_prompt(self.profile, &self.special_tokens.audio_token, options)?;
        let input_ids = self.encode(&rendered.text)?;
        let audio_token_positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, token)| {
                (*token == self.special_tokens.audio_token_id).then_some(idx)
            })
            .collect::<Vec<_>>();
        if audio_token_positions.len() != 1 {
            return Err(Error::TokenizationError(format!(
                "Granite Speech prompt must tokenize {} to exactly one id {}, found {}",
                self.special_tokens.audio_token,
                self.special_tokens.audio_token_id,
                audio_token_positions.len()
            )));
        }
        let prefix_text_token_count = match options
            .prefix_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(prefix) => self.encode(prefix)?.len(),
            None => 0,
        };

        Ok(GraniteSpeechPrompt {
            profile: self.profile,
            task: options.task,
            text: rendered.text,
            instruction: rendered.instruction,
            prompt_ignored: rendered.prompt_ignored,
            input_ids,
            audio_token_positions,
            prefix_text_token_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AUDIO: &str = "<|audio|>";

    fn render(
        profile: GraniteSpeechPromptProfile,
        options: GraniteSpeechPromptOptions,
    ) -> Result<String> {
        render_granite_speech_prompt(profile, AUDIO, &options).map(|prompt| prompt.text)
    }

    fn base(options: GraniteSpeechPromptOptions) -> Result<String> {
        render(GraniteSpeechPromptProfile::Base, options)
    }

    fn plus(options: GraniteSpeechPromptOptions) -> Result<String> {
        render(GraniteSpeechPromptProfile::Plus, options)
    }

    fn plus_template(user: &str, prefix: &str) -> String {
        format!(
            "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{prefix}"
        )
    }

    fn with_language(language: &str) -> GraniteSpeechPromptOptions {
        GraniteSpeechPromptOptions {
            language: Some(language.to_string()),
            ..GraniteSpeechPromptOptions::default()
        }
    }

    fn with_task(task: GraniteSpeechTask) -> GraniteSpeechPromptOptions {
        GraniteSpeechPromptOptions {
            task,
            ..GraniteSpeechPromptOptions::default()
        }
    }

    fn with_keywords(keywords: &[&str]) -> GraniteSpeechPromptOptions {
        GraniteSpeechPromptOptions {
            keywords: keywords.iter().map(|keyword| keyword.to_string()).collect(),
            ..GraniteSpeechPromptOptions::default()
        }
    }

    /// Goldens rendered from the published `chat_template.jinja` files with
    /// jinja2 (base de575db6, plus 1454e6e1 and e69efe37, whose templates are
    /// byte-identical) and IBM's `prompts.py`.
    #[test]
    fn base_prompts_match_the_published_template_and_ibm_instructions() {
        assert_eq!(
            base(GraniteSpeechPromptOptions::default()).unwrap(),
            "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
        );
        assert_eq!(
            base(with_language("fr")).unwrap(),
            "USER: <|audio|>transcribe the French speech with proper punctuation and capitalization.\n ASSISTANT:"
        );
        assert_eq!(
            base(with_keywords(&["Izwi", " Granite   Speech ", "Izwi", " "])).unwrap(),
            "USER: <|audio|>transcribe the speech to text. Keywords: Izwi, Granite Speech\n ASSISTANT:"
        );
        let translate = GraniteSpeechPromptOptions {
            task: GraniteSpeechTask::Translate {
                target: GraniteSpeechLanguage::English,
            },
            language: Some("de".to_string()),
            ..GraniteSpeechPromptOptions::default()
        };
        assert_eq!(
            base(translate.clone()).unwrap(),
            "USER: <|audio|>translate the speech to English with proper punctuation and capitalization.\n ASSISTANT:"
        );
        assert_eq!(
            base(GraniteSpeechPromptOptions {
                keywords: vec!["Bundestag".to_string()],
                ..translate
            })
            .unwrap(),
            "USER: <|audio|>translate the speech to English. Keywords: Bundestag\n ASSISTANT:"
        );
    }

    #[test]
    fn plus_prompts_match_the_published_template_and_card_instructions() {
        assert_eq!(
            plus(GraniteSpeechPromptOptions::default()).unwrap(),
            plus_template(
                "<|audio|> can you transcribe the speech into a written format?",
                ""
            )
        );
        assert_eq!(
            plus(with_language("German")).unwrap(),
            plus_template(
                "<|audio|> can you transcribe the German speech into a written format?",
                ""
            )
        );
        assert_eq!(
            plus(with_keywords(&["Izwi", "Granite"])).unwrap(),
            plus_template(
                "<|audio|> can you transcribe the speech into a written format? Keywords: Izwi, Granite",
                ""
            )
        );
        // Speaker attribution and timestamps ignore the source language.
        assert_eq!(
            plus(GraniteSpeechPromptOptions {
                language: Some("en".to_string()),
                ..with_task(GraniteSpeechTask::SpeakerAttributed)
            })
            .unwrap(),
            plus_template(
                "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.",
                ""
            )
        );
        assert_eq!(
            plus(GraniteSpeechPromptOptions {
                prefix_text: Some(" hello there ".to_string()),
                ..with_task(GraniteSpeechTask::WordTimestamps)
            })
            .unwrap(),
            plus_template(
                "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]",
                "hello there"
            )
        );
    }

    #[test]
    fn explicit_prompts_replace_the_instruction_with_the_profile_separator() {
        let raw = GraniteSpeechPromptOptions {
            custom_prompt: Some(
                "<|audio|>can you transcribe the speech into a written format?".into(),
            ),
            ..GraniteSpeechPromptOptions::default()
        };
        assert_eq!(
            base(raw.clone()).unwrap(),
            "USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:"
        );
        assert_eq!(
            plus(raw).unwrap(),
            plus_template(
                "<|audio|> can you transcribe the speech into a written format?",
                ""
            )
        );
        let misplaced = GraniteSpeechPromptOptions {
            custom_prompt: Some("transcribe the speech <|audio|> again".into()),
            ..GraniteSpeechPromptOptions::default()
        };
        assert!(base(misplaced).is_err());
    }

    #[test]
    fn request_prompts_override_only_with_a_granite_instruction() {
        let prompt = |text: &str| GraniteSpeechPromptOptions {
            custom_prompt: Some(text.to_string()),
            ..GraniteSpeechPromptOptions::default()
        };
        let rendered = |profile, text: &str| {
            render_granite_speech_prompt(profile, AUDIO, &prompt(text)).unwrap()
        };

        // The base card's llama-server example passes its instruction as
        // the OpenAI prompt.
        let card = rendered(
            GraniteSpeechPromptProfile::Base,
            "transcribe the speech with proper punctuation and capitalization.",
        );
        assert_eq!(card.instruction, GRANITE_SPEECH_BASE_ASR_INSTRUCTION);
        assert!(!card.prompt_ignored);
        let raw = rendered(
            GraniteSpeechPromptProfile::Base,
            "Can you transcribe the speech into a written format?",
        );
        assert_eq!(
            raw.text,
            "USER: <|audio|>Can you transcribe the speech into a written format?\n ASSISTANT:"
        );
        for instruction in [
            "Translate the speech to English.",
            "speaker attribution: Transcribe and denote who is speaking.",
            "TIMESTAMPS: Transcribe the speech.",
        ] {
            assert_eq!(
                rendered(GraniteSpeechPromptProfile::Plus, instruction).instruction,
                instruction
            );
        }

        // OpenAI-style context keeps the trained instruction.
        let context = rendered(
            GraniteSpeechPromptProfile::Base,
            "The following is a quarterly earnings call with Izwi.",
        );
        assert!(context.prompt_ignored);
        assert_eq!(
            context.text,
            "USER: <|audio|>transcribe the speech with proper punctuation and capitalization.\n ASSISTANT:"
        );

        // izwi's documented `keywords:` form becomes IBM keyword biasing.
        let keywords = rendered(
            GraniteSpeechPromptProfile::Base,
            "Keywords: Izwi, Granite Speech\nKokoro, ",
        );
        assert!(!keywords.prompt_ignored);
        assert_eq!(
            keywords.instruction,
            "transcribe the speech to text. Keywords: Izwi, Granite Speech, Kokoro"
        );
        assert_eq!(
            rendered(GraniteSpeechPromptProfile::Plus, "keywords: Izwi").instruction,
            "can you transcribe the speech into a written format? Keywords: Izwi"
        );
        assert_eq!(
            classify_granite_speech_prompt("keywords:  , ", AUDIO),
            GraniteSpeechPromptOverride::Ignored
        );
    }

    #[test]
    fn tasks_languages_and_prefixes_are_limited_to_the_trained_checkpoint() {
        for task in [
            GraniteSpeechTask::SpeakerAttributed,
            GraniteSpeechTask::WordTimestamps,
        ] {
            let err = base(with_task(task)).unwrap_err().to_string();
            assert!(err.contains("requires the plus checkpoint"), "{err}");
        }
        let translate = GraniteSpeechPromptOptions {
            language: Some("de".to_string()),
            ..with_task(GraniteSpeechTask::Translate {
                target: GraniteSpeechLanguage::English,
            })
        };
        assert!(plus(translate)
            .unwrap_err()
            .to_string()
            .contains("requires the base checkpoint"));
        assert!(base(GraniteSpeechPromptOptions {
            prefix_text: Some("earlier words".into()),
            ..GraniteSpeechPromptOptions::default()
        })
        .unwrap_err()
        .to_string()
        .contains("prefix_text"));

        // Japanese is a base-only source language; plus drops it.
        assert!(base(with_language("ja")).is_ok());
        let err = plus(with_language("Japanese")).unwrap_err().to_string();
        assert!(
            err.contains("supported source languages: de, en, es, fr, pt"),
            "{err}"
        );
        assert!(base(with_language("Chinese")).is_err());
        assert!(base(with_language("klingon")).is_err());
        assert_eq!(
            base(with_language("  ")).unwrap(),
            base(GraniteSpeechPromptOptions::default()).unwrap()
        );
    }

    #[test]
    fn translation_needs_a_source_and_a_published_pair() {
        let to = |target, source: Option<&str>| GraniteSpeechPromptOptions {
            task: GraniteSpeechTask::Translate { target },
            language: source.map(str::to_string),
            ..GraniteSpeechPromptOptions::default()
        };
        assert!(base(to(GraniteSpeechLanguage::English, None))
            .unwrap_err()
            .to_string()
            .contains("explicit source language"));
        assert!(base(to(GraniteSpeechLanguage::English, Some("en"))).is_err());
        assert!(base(to(GraniteSpeechLanguage::German, Some("fr")))
            .unwrap_err()
            .to_string()
            .contains("fr->de"));
        assert!(base(to(GraniteSpeechLanguage::Mandarin, Some("en"))).is_ok());
        assert!(base(to(GraniteSpeechLanguage::English, Some("ja"))).is_ok());
    }

    #[test]
    fn language_aliases_follow_ibm() {
        for (alias, code, label) in [
            ("en", "en", "English"),
            ("ENG", "en", "English"),
            (" german ", "de", "German"),
            ("fre", "fr", "French"),
            ("spa", "es", "Spanish"),
            ("Portuguese", "pt", "Portuguese"),
            ("jpn", "ja", "Japanese"),
            ("ita", "it", "Italian"),
            ("Mandarin Chinese", "zh", "Mandarin"),
        ] {
            let language = GraniteSpeechLanguage::from_alias(alias).unwrap();
            assert_eq!(
                (language.code(), language.label()),
                (code, label),
                "{alias}"
            );
        }
        assert_eq!(GraniteSpeechLanguage::from_alias("Deutsch"), None);
        assert_eq!(GraniteSpeechLanguage::from_alias("auto"), None);
    }
}

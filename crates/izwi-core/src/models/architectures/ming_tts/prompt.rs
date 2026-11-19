//! Ming's text front end: the prompt layout, the control captions and the
//! light text clean-up of the reference pipeline (`prompt_contract.py`,
//! `tts_tasks.py`, `frontend/toolkit.py`).
//!
//! The strings here are part of the model's input protocol. Each piece is
//! tokenized on its own and the ids are joined, so byte-pair merges never
//! cross a piece boundary; the caption's JSON keeps Python's key order and
//! separators. A prompt that differs still produces speech, only worse,
//! which is why every piece is pinned by a test.

use std::path::Path;

use tokenizers::Tokenizer;

use super::config::{
    PINNED_TOKENS, TOKEN_AUDIO_PATCH, TOKEN_AUDIO_START, TOKEN_SPEAKER_PAD, TOKEN_SPEAKER_START,
};
use crate::error::{Error, Result};

const SYSTEM_PREFIX: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const USER_PREFIX: &str = "<|im_start|>user\n";
const USER_SUFFIX: &str = "<|im_end|>\n";
const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
const TASK_PROMPT: &str = "Please generate speech based on the following description.\n";
const SPEAKER_LABEL: &str = "  speaker_1:";
const SPEAKER_START: &str = "<|vision_start|>";
const SPEAKER_PAD: &str = "<|vision_pad|>";
const SPEAKER_END: &str = "<|vision_end|>\n";
const TEXT_PREFIX: &str = " Text input:\n";
const INSTRUCTION_END: &str = "<|im_end|>";
const AUDIO_START: &str = "<audio>";

/// The inline sounds the model performs, written exactly like this in the
/// text.
pub const MING_INLINE_TAGS: &[&str] = &[
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
];

/// A level of speed, pitch or volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MingLevel {
    Low,
    Medium,
    High,
}

impl MingLevel {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" | "slow" | "低" | "慢速" => Some(Self::Low),
            "medium" | "normal" | "mid" | "中" | "中速" => Some(Self::Medium),
            "high" | "fast" | "高" | "快速" => Some(Self::High),
            _ => None,
        }
    }

    fn speed(self) -> &'static str {
        match self {
            Self::Low => "慢速",
            Self::Medium => "中速",
            Self::High => "快速",
        }
    }

    fn level(self) -> &'static str {
        match self {
            Self::Low => "低",
            Self::Medium => "中",
            Self::High => "高",
        }
    }
}

/// The emotions the model was trained to name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MingEmotion {
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Neutral,
}

impl MingEmotion {
    pub const ALL: [Self; 7] = [
        Self::Happy,
        Self::Sad,
        Self::Angry,
        Self::Fearful,
        Self::Disgusted,
        Self::Surprised,
        Self::Neutral,
    ];

    /// The English name requests use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Angry => "angry",
            Self::Fearful => "fearful",
            Self::Disgusted => "disgusted",
            Self::Surprised => "surprised",
            Self::Neutral => "neutral",
        }
    }

    /// The label the model reads.
    pub fn label(self) -> &'static str {
        match self {
            Self::Happy => "高兴",
            Self::Sad => "悲伤",
            Self::Angry => "愤怒",
            Self::Fearful => "恐惧",
            Self::Disgusted => "厌恶",
            Self::Surprised => "惊讶",
            Self::Neutral => "中性",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let lower = value.to_ascii_lowercase();
        Self::ALL.into_iter().find(|emotion| {
            emotion.name() == lower
                || emotion.label() == value
                || matches!(
                    (emotion, lower.as_str()),
                    (Self::Happy, "happiness" | "joy" | "joyful")
                        | (Self::Sad, "sadness")
                        | (Self::Angry, "anger")
                        | (Self::Fearful, "fear" | "scared")
                        | (Self::Disgusted, "disgust")
                        | (Self::Surprised, "surprise")
                )
        })
    }
}

/// The two Chinese dialects the model speaks on request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MingDialect {
    Sichuanese,
    Cantonese,
}

impl MingDialect {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sichuanese" | "sichuan" | "chuanyu" | "chuanyu dialect" | "四川话" | "川渝" => {
                Some(Self::Sichuanese)
            }
            "cantonese" | "yue" | "粤语" | "广东话" => Some(Self::Cantonese),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sichuanese => "chuanyu dialect",
            Self::Cantonese => "cantonese",
        }
    }
}

/// What the assistant-side caption asks for. Upstream trains each as its
/// own task, so one request carries one of them.
#[derive(Debug, Clone, PartialEq)]
pub enum MingControl {
    /// No caption.
    None,
    /// A built-in voice, by its exact name.
    BuiltInVoice(String),
    /// A described voice.
    VoiceDesign(String),
    /// A plain-language instruction, passed through as written.
    Freeform(String),
    Emotion(MingEmotion),
    Dialect(MingDialect),
    /// Speed, pitch and volume; at least one is set.
    Prosody {
        speed: Option<MingLevel>,
        pitch: Option<MingLevel>,
        volume: Option<MingLevel>,
    },
}

impl MingControl {
    /// The caption the model reads, or `None` for no caption.
    pub fn instruction(&self) -> Result<Option<String>> {
        let text = |value: &str, what: &str| -> Result<String> {
            let value = value.trim();
            if value.is_empty() {
                return Err(Error::InvalidInput(format!("Ming {what} is empty")));
            }
            Ok(value.to_string())
        };
        Ok(Some(match self {
            Self::None => return Ok(None),
            Self::Freeform(instruction) => text(instruction, "instruction")?,
            Self::BuiltInVoice(name) => caption(Caption {
                ip: Some(&text(name, "voice name")?),
                ..Caption::chinese()
            }),
            Self::VoiceDesign(style) => caption(Caption {
                style: Some(&text(style, "voice description")?),
                ..Caption::chinese()
            }),
            Self::Emotion(emotion) => caption(Caption {
                emotion: Some(emotion.label()),
                ..Caption::chinese()
            }),
            Self::Dialect(dialect) => caption(Caption {
                dialect: Some(dialect.label()),
                ..Caption::english()
            }),
            Self::Prosody {
                speed,
                pitch,
                volume,
            } => {
                if speed.is_none() && pitch.is_none() && volume.is_none() {
                    return Err(Error::InvalidInput(
                        "Ming prosody control sets none of speed, pitch and volume".into(),
                    ));
                }
                caption(Caption {
                    speed: speed.map(MingLevel::speed),
                    pitch: pitch.map(MingLevel::level),
                    volume: volume.map(MingLevel::level),
                    ..Caption::chinese()
                })
            }
        }))
    }

    /// Whether the text keeps its own punctuation and spelling untouched
    /// (the dialect task does, as upstream's front end leaves it alone).
    fn preserves_text(&self) -> bool {
        matches!(self, Self::Dialect(_))
    }
}

struct Caption<'a> {
    english: bool,
    dialect: Option<&'a str>,
    style: Option<&'a str>,
    speed: Option<&'a str>,
    pitch: Option<&'a str>,
    volume: Option<&'a str>,
    emotion: Option<&'a str>,
    ip: Option<&'a str>,
}

impl Caption<'_> {
    fn chinese() -> Self {
        Self {
            english: false,
            dialect: None,
            style: None,
            speed: None,
            pitch: None,
            volume: None,
            emotion: None,
            ip: None,
        }
    }

    fn english() -> Self {
        Self {
            english: true,
            ..Self::chinese()
        }
    }
}

/// `json.dumps(payload, ensure_ascii=False)` of upstream's ordered caption:
/// `", "` and `": "` separators, keys in training order.
fn caption(fields: Caption<'_>) -> String {
    let keys: [&str; 9] = if fields.english {
        [
            "id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion", "ip",
        ]
    } else {
        [
            "序号",
            "说话人",
            "方言",
            "风格",
            "语速",
            "基频",
            "音量",
            "情感",
            "IP",
        ]
    };
    let value = |value: Option<&str>| match value {
        Some(value) => serde_json::Value::String(value.to_string()).to_string(),
        None => "null".to_string(),
    };
    let bgm = ["Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"]
        .map(|key| format!("\"{key}\": null"))
        .join(", ");
    format!(
        "{{\"audio_sequence\": [{{\"{}\": 1, \"{}\": \"speaker_1\", \"{}\": {}, \"{}\": {}, \
         \"{}\": {}, \"{}\": {}, \"{}\": {}, \"{}\": {}, \"BGM\": {{{bgm}}}, \"{}\": {}}}]}}",
        keys[0],
        keys[1],
        keys[2],
        value(fields.dialect),
        keys[3],
        value(fields.style),
        keys[4],
        value(fields.speed),
        keys[5],
        value(fields.pitch),
        keys[6],
        value(fields.volume),
        keys[7],
        value(fields.emotion),
        keys[8],
        value(fields.ip),
    )
}

/// The name of a target language as the prompt writes it, from a code or a
/// name; `None` for a language the model was not trained on.
pub fn target_language_name(language: &str) -> Option<&'static str> {
    let language = language.trim().to_ascii_lowercase().replace('_', "-");
    let primary = language.split('-').next().unwrap_or("");
    Some(match (language.as_str(), primary) {
        ("chinese" | "mandarin" | "cantonese", _) | (_, "zh" | "cmn" | "yue") => "Chinese",
        ("english", _) | (_, "en") => "English",
        ("japanese", _) | (_, "ja") => "Japanese",
        ("korean", _) | (_, "ko") => "Korean",
        ("german", _) | (_, "de") => "German",
        ("french", _) | (_, "fr") => "French",
        ("spanish", _) | (_, "es") => "Spanish",
        _ => return None,
    })
}

/// Whether text in `language` (a prompt name) bypasses clean-up: every
/// language but Chinese and English does upstream.
fn language_preserves_text(language: Option<&str>) -> bool {
    language.is_some_and(|language| !matches!(language, "Chinese" | "English"))
}

/// Strip the Markdown wrappers a reader would not speak: code fences and
/// backticks, heading marks, bold marks and link targets. Inline sound tags
/// are left as written.
fn strip_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            continue;
        }
        let indent = line.len() - trimmed.len();
        let hashes = trimmed.chars().take_while(|&c| c == '#').count();
        let heading =
            indent <= 3 && (1..=6).contains(&hashes) && trimmed[hashes..].starts_with([' ', '\t']);
        out.push_str(if heading {
            trimmed[hashes..].trim_start()
        } else {
            line
        });
    }
    let out = out.replace("**", "").replace('`', "");
    // [label](target) → label, unless the bracket is an inline sound tag.
    let mut result = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(open) = rest.find('[') {
        let Some(close) = rest[open..].find("](").map(|at| open + at) else {
            break;
        };
        let label = &rest[open + 1..close];
        let tag = &rest[open..=close];
        let end = rest[close + 2..].find(')').map(|at| close + 2 + at);
        match end {
            Some(end)
                if !label.contains('\n')
                    && !rest[close + 2..end].contains('\n')
                    && !MING_INLINE_TAGS.contains(&tag) =>
            {
                result.push_str(&rest[..open]);
                result.push_str(label);
                rest = &rest[end + 1..];
            }
            _ => {
                result.push_str(&rest[..close + 2]);
                rest = &rest[close + 2..];
            }
        }
    }
    result.push_str(rest);
    result
}

/// `add_period`: a text ending in a letter or digit gets a full stop (the
/// Chinese one after a Chinese character), and a trailing weak mark becomes
/// one.
fn add_period(text: &str) -> String {
    let stripped = text.trim_end();
    let Some(last) = stripped.chars().last() else {
        return ".".to_string();
    };
    let is_han = |c: char| ('\u{4e00}'..='\u{9fff}').contains(&c);
    if "，,、;；".contains(last) {
        let body = &stripped[..stripped.len() - last.len_utf8()];
        let stop = if stripped.chars().any(is_han) {
            '。'
        } else {
            '.'
        };
        return format!("{body}{stop}");
    }
    if !last.is_alphanumeric() {
        return stripped.to_string();
    }
    let stop = if is_han(last) || ('\u{3400}'..='\u{4dbf}').contains(&last) {
        '。'
    } else {
        '.'
    };
    format!("{stripped}{stop}")
}

/// The text as the model reads it.
pub fn prepare_text(raw: &str, language: Option<&str>, control: &MingControl) -> Result<String> {
    let text = strip_markdown(raw);
    let text = text.trim();
    if !text.chars().any(char::is_alphanumeric) {
        return Err(Error::InvalidInput(
            "Ming needs text with at least one letter or digit to speak".into(),
        ));
    }
    if language_preserves_text(language) || control.preserves_text() {
        return Ok(text.to_string());
    }
    Ok(add_period(text))
}

/// How the prompt's speaker block is filled.
#[derive(Debug, Clone, PartialEq)]
pub enum MingSpeaker {
    /// No speaker block.
    Absent,
    /// A block whose row is zeros: built-in and designed voices.
    Zero,
    /// A block whose row is the projection of this 192-wide embedding.
    Embedding(Vec<f32>),
}

/// A prompt's token ids and the rows the model fills in afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MingPromptIds {
    pub ids: Vec<u32>,
    /// The speaker row, when the prompt has a speaker block.
    pub speaker_row: Option<usize>,
    /// The first reference-latent row, when the prompt carries a reference.
    pub latent_row: Option<usize>,
    /// Tokens the text alone takes (budgets scale with it).
    pub text_tokens: usize,
}

pub struct MingTokenizer {
    tokenizer: Tokenizer,
}

impl MingTokenizer {
    /// Load `tokenizer.json` and check the ids the prompt relies on.
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Ming tokenizer {} unreadable: {error}",
                path.display()
            ))
        })?;
        let this = Self { tokenizer };
        this.check_layout()?;
        Ok(this)
    }

    #[cfg(test)]
    pub(crate) fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Each pinned token must have its trained id and encode to itself: a
    /// tokenizer that splits `<audio>` still produces audio, conditioned on
    /// a sequence the model never saw.
    fn check_layout(&self) -> Result<()> {
        for (token, id) in PINNED_TOKENS {
            let found = self.tokenizer.token_to_id(token);
            if found != Some(id) {
                return Err(Error::ModelLoadError(format!(
                    "Ming tokenizer maps {token} to {found:?}, expected {id}"
                )));
            }
            if self.encode(token)? != [id] {
                return Err(Error::ModelLoadError(format!(
                    "Ming tokenizer splits {token}"
                )));
            }
        }
        Ok(())
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::InvalidInput(format!("Ming tokenization failed: {error}")))?
            .get_ids()
            .to_vec())
    }

    /// Tokens `text` takes on its own.
    pub fn text_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    /// The prompt for `text` (already prepared), piece by piece.
    ///
    /// `language` is a prompt language name; `reference` is the reference
    /// clip's transcript and its latent count.
    pub fn prompt_ids(
        &self,
        text: &str,
        language: Option<&str>,
        speaker: bool,
        instruction: Option<&str>,
        reference: Option<(&str, usize)>,
    ) -> Result<MingPromptIds> {
        let mut ids = self.encode(SYSTEM_PREFIX)?;
        ids.extend(self.encode(USER_PREFIX)?);
        let task = match language {
            Some(language) => format!("{TASK_PROMPT}Target language: {language}.\n"),
            None => TASK_PROMPT.to_string(),
        };
        ids.extend(self.encode(&task)?);
        if speaker {
            ids.extend(self.encode(SPEAKER_LABEL)?);
            ids.extend(self.encode(SPEAKER_START)?);
            ids.extend(self.encode(SPEAKER_PAD)?);
            ids.extend(self.encode(SPEAKER_END)?);
        }
        ids.extend(self.encode(TEXT_PREFIX)?);
        if let Some((transcript, latents)) = reference {
            if latents == 0 || transcript.trim().is_empty() {
                return Err(Error::InvalidInput(
                    "a Ming reference needs its transcript and at least one latent".into(),
                ));
            }
            ids.extend(self.encode(transcript)?);
        }
        let text_ids = self.encode(text)?;
        let text_tokens = text_ids.len();
        ids.extend(text_ids);
        ids.extend(self.encode(USER_SUFFIX)?);
        ids.extend(self.encode(ASSISTANT_PREFIX)?);
        if let Some(instruction) = instruction {
            ids.extend(self.encode(instruction)?);
            ids.extend(self.encode(INSTRUCTION_END)?);
        }
        ids.extend(self.encode(AUDIO_START)?);
        if ids.last() != Some(&TOKEN_AUDIO_START) {
            return Err(Error::InferenceError(
                "Ming prompt does not end on its audio start token".into(),
            ));
        }

        let starts = ids
            .iter()
            .enumerate()
            .filter_map(|(index, &id)| (id == TOKEN_SPEAKER_START).then_some(index + 1))
            .collect::<Vec<_>>();
        let speaker_row = match (speaker, starts.as_slice()) {
            (false, []) => None,
            (true, &[row]) if ids.get(row) == Some(&TOKEN_SPEAKER_PAD) => Some(row),
            _ => {
                return Err(Error::InvalidInput(
                    "Ming text may not contain the speaker slot tokens".into(),
                ))
            }
        };
        let latent_row = reference.map(|(_, latents)| {
            let row = ids.len();
            ids.extend(std::iter::repeat_n(TOKEN_AUDIO_PATCH, latents));
            row
        });
        Ok(MingPromptIds {
            ids,
            speaker_row,
            latent_row,
            text_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captions_match_python_json_dumps() {
        // tts_tasks.build_structured_instruction(ip="US_Emma").
        assert_eq!(
            MingControl::BuiltInVoice("US_Emma".into())
                .instruction()
                .unwrap()
                .unwrap(),
            "{\"audio_sequence\": [{\"序号\": 1, \"说话人\": \"speaker_1\", \"方言\": null, \
             \"风格\": null, \"语速\": null, \"基频\": null, \"音量\": null, \"情感\": null, \
             \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \
             \"ENV\": null, \"SNR\": null}, \"IP\": \"US_Emma\"}]}"
        );
        // The dialect task uses the English schema.
        assert_eq!(
            MingControl::Dialect(MingDialect::Cantonese)
                .instruction()
                .unwrap()
                .unwrap(),
            "{\"audio_sequence\": [{\"id\": 1, \"speaker\": \"speaker_1\", \"dialect\": \
             \"cantonese\", \"style\": null, \"speed\": null, \"pitch\": null, \"volume\": null, \
             \"emotion\": null, \"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \
             \"Theme\": null, \"ENV\": null, \"SNR\": null}, \"ip\": null}]}"
        );
        let prosody = MingControl::Prosody {
            speed: Some(MingLevel::Low),
            pitch: None,
            volume: Some(MingLevel::High),
        }
        .instruction()
        .unwrap()
        .unwrap();
        assert!(prosody.contains("\"语速\": \"慢速\", \"基频\": null, \"音量\": \"高\""));
        let emotion = MingControl::Emotion(MingEmotion::Happy)
            .instruction()
            .unwrap()
            .unwrap();
        assert!(emotion.contains("\"情感\": \"高兴\""));
    }

    #[test]
    fn caption_strings_are_json_escaped_and_keep_unicode() {
        let caption = MingControl::VoiceDesign("a \"calm\" voice\n温柔".into())
            .instruction()
            .unwrap()
            .unwrap();
        assert!(caption.contains("\"风格\": \"a \\\"calm\\\" voice\\n温柔\""));
        let parsed: serde_json::Value = serde_json::from_str(&caption).unwrap();
        assert_eq!(
            parsed["audio_sequence"][0]["风格"],
            "a \"calm\" voice\n温柔"
        );
        assert_eq!(MingControl::None.instruction().unwrap(), None);
        assert!(MingControl::Freeform("  ".into()).instruction().is_err());
        assert_eq!(
            MingControl::Freeform(" Speak slowly. ".into())
                .instruction()
                .unwrap()
                .unwrap(),
            "Speak slowly."
        );
        assert!(MingControl::Prosody {
            speed: None,
            pitch: None,
            volume: None
        }
        .instruction()
        .is_err());
    }

    #[test]
    fn languages_resolve_to_prompt_names() {
        assert_eq!(target_language_name("de"), Some("German"));
        assert_eq!(target_language_name("en-US"), Some("English"));
        assert_eq!(target_language_name("zh_CN"), Some("Chinese"));
        assert_eq!(target_language_name("Japanese"), Some("Japanese"));
        assert_eq!(target_language_name("yue"), Some("Chinese"));
        assert_eq!(target_language_name("it"), None);
    }

    #[test]
    fn text_gets_a_final_stop_unless_preserved() {
        let none = MingControl::None;
        assert_eq!(
            prepare_text("Hello there", None, &none).unwrap(),
            "Hello there."
        );
        assert_eq!(prepare_text("你好", None, &none).unwrap(), "你好。");
        assert_eq!(prepare_text("Wait,", None, &none).unwrap(), "Wait.");
        assert_eq!(prepare_text("等一下，", None, &none).unwrap(), "等一下。");
        assert_eq!(prepare_text("Done!", None, &none).unwrap(), "Done!");
        assert_eq!(
            prepare_text("Guten Tag", Some("German"), &none).unwrap(),
            "Guten Tag"
        );
        assert_eq!(
            prepare_text(
                "今日天气几好",
                None,
                &MingControl::Dialect(MingDialect::Cantonese)
            )
            .unwrap(),
            "今日天气几好"
        );
        assert!(prepare_text("  ...  ", None, &none).is_err());
    }

    #[test]
    fn markdown_is_stripped_and_sound_tags_survive() {
        let none = MingControl::None;
        assert_eq!(
            prepare_text("# Title\n**Bold** and `code`", None, &none).unwrap(),
            "Title\nBold and code."
        );
        assert_eq!(
            prepare_text("See [the docs](https://example.com) now", None, &none).unwrap(),
            "See the docs now."
        );
        assert_eq!(
            prepare_text("[Laughter](so funny) indeed", None, &none).unwrap(),
            "[Laughter](so funny) indeed."
        );
        assert_eq!(
            prepare_text("[Sigh] it rains again", None, &none).unwrap(),
            "[Sigh] it rains again."
        );
    }

    #[test]
    fn levels_emotions_and_dialects_parse() {
        assert_eq!(MingLevel::parse("Fast"), Some(MingLevel::High));
        assert_eq!(MingLevel::parse("慢速"), Some(MingLevel::Low));
        assert_eq!(MingLevel::parse("loud"), None);
        assert_eq!(MingEmotion::parse("Angry"), Some(MingEmotion::Angry));
        assert_eq!(MingEmotion::parse("悲伤"), Some(MingEmotion::Sad));
        assert_eq!(MingEmotion::parse("bored"), None);
        assert_eq!(
            MingDialect::parse("Cantonese"),
            Some(MingDialect::Cantonese)
        );
        assert_eq!(
            MingDialect::parse("sichuanese"),
            Some(MingDialect::Sichuanese)
        );
    }
}

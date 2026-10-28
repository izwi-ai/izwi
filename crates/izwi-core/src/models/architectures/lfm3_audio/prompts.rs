//! LFM3-Audio task prompts, rendered as the checkpoint's chat template
//! renders them (`continue_final_message` for a prefilled assistant turn),
//! and the languages and voices they name.

use izwi_model_catalog::chat::tools::{is_deferred_tool, tool_to_json};
use izwi_model_catalog::CHAT_AUDIO_PLACEHOLDER;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMessage, ChatRole};

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

/// The system prompt of an interleaved voice reply that may call tools.
pub const LFM3_AUDIO_TOOLS_SYSTEM_PROMPT: &str =
    "Respond with function calls, and interleaved text and audio.";

/// The line a voice reply's system prompt names the tools after
/// (LFM3-Audio's card).
const TOOL_LIST_HEADER: &str = "List of tools:";

/// What a reply writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm3AudioReplyOutput {
    /// Text only (chat): the reply never starts speech.
    Text,
    /// Interleaved text and speech (voice).
    Interleaved,
}

impl Lfm3AudioReplyOutput {
    /// The assistant turn's prefill: an interleaved reply opens
    /// `<|mixed_start|>`, unless it may call tools, when the model chooses
    /// between a call and speech itself.
    pub(crate) fn prefill(self, tools_offered: bool) -> &'static str {
        match self {
            Self::Interleaved if !tools_offered => "<|mixed_start|>",
            _ => "",
        }
    }
}

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

    /// A turn whose `content` holds a speech clip at each
    /// [`CHAT_AUDIO_PLACEHOLDER`].
    pub(crate) fn message_turn(mut self, role: &str, content: &str) -> Self {
        self.text.push_str("<|im_start|>");
        self.text.push_str(role);
        self.text.push('\n');
        let mut pieces = content.split(CHAT_AUDIO_PLACEHOLDER);
        self.text.push_str(pieces.next().unwrap_or_default());
        for piece in pieces {
            self.parts
                .push(PromptPart::Text(std::mem::take(&mut self.text)));
            self.parts.push(PromptPart::Audio);
            self.text.push_str(piece);
        }
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

/// The prompt of a reply to `messages`, rendered as LFM3-Audio's chat
/// template renders them, each [`CHAT_AUDIO_PLACEHOLDER`] in a user turn a
/// speech clip.
///
/// * The system turn: an interleaved reply's mode line
///   ([`LFM3_AUDIO_TOOLS_SYSTEM_PROMPT`] when it may call tools), the leading
///   system message, and for an interleaved reply with tools `List of
///   tools:`, blank lines apart; the template then appends `<tools>` with
///   one definition per line (deferred ones left out). A text reply's system
///   turn is the template's own.
/// * An assistant turn at or before the last user turn keeps only the text
///   after its last `</think>`, unless `preserve_thinking`; later ones are
///   kept whole.
/// * A tool turn is wrapped in `<tool_response>` unless it already holds one
///   or is empty (the gateway renders deferred-tool results itself).
/// * The assistant turn is prefilled as [`Lfm3AudioReplyOutput::prefill`].
pub(crate) fn reply_prompt(
    messages: &[ChatMessage],
    tools: &[Value],
    output: Lfm3AudioReplyOutput,
    preserve_thinking: bool,
) -> Vec<PromptPart> {
    let (system, turns) = match messages.split_first() {
        Some((first, rest)) if first.role == ChatRole::System => (first.content.trim(), rest),
        _ => ("", messages),
    };
    let mut lines = Vec::new();
    if output == Lfm3AudioReplyOutput::Interleaved {
        lines.push(if tools.is_empty() {
            LFM3_AUDIO_INTERLEAVED_SYSTEM_PROMPT
        } else {
            LFM3_AUDIO_TOOLS_SYSTEM_PROMPT
        });
    }
    if !system.is_empty() {
        lines.push(system);
    }
    let tool_lines = tools
        .iter()
        .filter(|tool| !is_deferred_tool(tool))
        .map(tool_to_json)
        .collect::<Vec<_>>();
    if output == Lfm3AudioReplyOutput::Interleaved && !tool_lines.is_empty() {
        lines.push(TOOL_LIST_HEADER);
    }
    let mut system = lines.join("\n\n");
    if !tool_lines.is_empty() {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str("<tools>\n");
        system.push_str(&tool_lines.join("\n"));
        system.push_str("\n</tools>");
    }

    let mut prompt = PromptBuilder::new();
    if !system.is_empty() {
        prompt = prompt.turn("system", &system);
    }
    let last_user = turns
        .iter()
        .rposition(|message| message.role == ChatRole::User);
    for (index, message) in turns.iter().enumerate() {
        let content = message.content.as_str();
        prompt = match message.role {
            ChatRole::System => prompt.turn("system", content),
            ChatRole::User => prompt.message_turn("user", content),
            ChatRole::Assistant => {
                let keep_thinking = preserve_thinking || last_user.is_none_or(|last| index > last);
                let content = match content.rsplit_once("</think>") {
                    Some((_, answer)) if !keep_thinking => answer.trim(),
                    _ => content,
                };
                prompt.turn("assistant", content)
            }
            ChatRole::Tool if content.is_empty() || content.contains("<tool_response>") => {
                prompt.turn("tool", content)
            }
            ChatRole::Tool => prompt.turn(
                "tool",
                &format!("<tool_response>\n{content}\n</tool_response>"),
            ),
        };
    }
    prompt.assistant(output.prefill(!tools.is_empty()))
}

/// The messages of a spoken turn's reply: the instructions, the text history
/// (its system messages left out, as a voice session keeps them in the
/// instructions) and the spoken turn. A history that already holds the
/// spoken turn (a tool round's continuation) is kept as it is.
pub(crate) fn voice_reply_messages(
    history: &[ChatMessage],
    system_prompt: Option<&str>,
) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(history.len() + 2);
    if let Some(system) = system_prompt.map(str::trim).filter(|text| !text.is_empty()) {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: system.to_string(),
        });
    }
    messages.extend(
        history
            .iter()
            .filter(|message| message.role != ChatRole::System)
            .cloned(),
    );
    if !history
        .iter()
        .any(|message| message.content.contains(CHAT_AUDIO_PLACEHOLDER))
    {
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: CHAT_AUDIO_PLACEHOLDER.to_string(),
        });
    }
    messages
}

/// How many speech clips a prompt reads.
pub(crate) fn audio_slots(prompt: &[PromptPart]) -> usize {
    prompt
        .iter()
        .filter(|part| matches!(part, PromptPart::Audio))
        .count()
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

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    /// The spoken turn's prompt as it was before replies took messages: the
    /// mode line and instructions, the text history (system messages left
    /// out), the clip, and `<|mixed_start|>`.
    #[test]
    fn voice_reply_prompts_render_the_spoken_turn_as_before() {
        let history = [
            message(ChatRole::System, "kept in the instructions"),
            message(ChatRole::User, "Hi"),
            message(ChatRole::Assistant, "Hello!"),
        ];
        let prompt = reply_prompt(
            &voice_reply_messages(&history, Some(" Be brief. ")),
            &[],
            Lfm3AudioReplyOutput::Interleaved,
            false,
        );
        assert_eq!(
            text(&prompt),
            "<|startoftext|><|im_start|>system\nRespond with interleaved text and audio.\n\n\
             Be brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\n[AUDIO]<|im_end|>\n\
             <|im_start|>assistant\n<|mixed_start|>"
        );
        assert_eq!(audio_slots(&prompt), 1);
        assert_eq!(
            text(&reply_prompt(
                &voice_reply_messages(&[], None),
                &[],
                Lfm3AudioReplyOutput::Interleaved,
                false
            )),
            "<|startoftext|><|im_start|>system\nRespond with interleaved text and audio.<|im_end|>\n\
             <|im_start|>user\n[AUDIO]<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>"
        );
        // A tool round's history already holds the spoken turn.
        let round = [
            message(ChatRole::User, CHAT_AUDIO_PLACEHOLDER),
            message(ChatRole::Assistant, "<tool_call>[f()]</tool_call>"),
            message(ChatRole::Tool, "21"),
        ];
        assert_eq!(voice_reply_messages(&round, None), round);
    }

    /// A chat reply is text: no mode line or prefill, each clip where its
    /// user turn places it, and earlier answers without their thinking.
    #[test]
    fn chat_reply_prompts_answer_in_text() {
        let messages = [
            message(ChatRole::System, "Be brief."),
            message(ChatRole::User, "<|audio|>What is said?"),
            message(ChatRole::Assistant, "<think>a fox</think>\n\nA fox jumps."),
            message(ChatRole::User, "And in <|audio|> and <|audio|>?"),
        ];
        let prompt = reply_prompt(&messages, &[], Lfm3AudioReplyOutput::Text, false);
        assert_eq!(
            text(&prompt),
            "<|startoftext|><|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\n[AUDIO]What is said?<|im_end|>\n\
             <|im_start|>assistant\nA fox jumps.<|im_end|>\n\
             <|im_start|>user\nAnd in [AUDIO] and [AUDIO]?<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(audio_slots(&prompt), 3);
        // `preserve_thinking` keeps an earlier answer's thinking.
        assert!(text(&reply_prompt(
            &messages,
            &[],
            Lfm3AudioReplyOutput::Text,
            true
        ))
        .contains("<|im_start|>assistant\n<think>a fox</think>\n\nA fox jumps.<|im_end|>"));
        assert_eq!(
            text(&reply_prompt(
                &[message(ChatRole::User, "Hi")],
                &[],
                Lfm3AudioReplyOutput::Text,
                false
            )),
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// With tools the system turn lists them (deferred ones left out; a text
    /// reply's as the template writes it), an interleaved reply switches to
    /// speech itself, and tool results are native tool turns.
    #[test]
    fn reply_prompts_list_tools_and_render_their_results() {
        let tools = [
            serde_json::json!({"type": "function", "function": {"name": "get_time"}}),
            serde_json::json!({
                "type": "function",
                "function": {"name": "search", "defer_loading": true}
            }),
        ];
        let messages = [
            message(ChatRole::System, "Be brief."),
            message(ChatRole::User, "<|audio|>"),
            message(ChatRole::Assistant, "<tool_call>[get_time()]</tool_call>"),
            message(ChatRole::Tool, "12:00"),
            message(ChatRole::Tool, ""),
            message(ChatRole::Tool, "<tool_response>\nkept\n</tool_response>"),
        ];
        let turns = "<|im_start|>user\n[AUDIO]<|im_end|>\n\
             <|im_start|>assistant\n<tool_call>[get_time()]</tool_call><|im_end|>\n\
             <|im_start|>tool\n<tool_response>\n12:00\n</tool_response><|im_end|>\n\
             <|im_start|>tool\n<|im_end|>\n\
             <|im_start|>tool\n<tool_response>\nkept\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n";
        assert_eq!(
            text(&reply_prompt(
                &messages,
                &tools,
                Lfm3AudioReplyOutput::Text,
                false
            )),
            format!(
                "<|startoftext|><|im_start|>system\nBe brief.\n\n\
                 <tools>\n{{\"name\": \"get_time\"}}\n</tools><|im_end|>\n{turns}"
            )
        );
        assert_eq!(
            text(&reply_prompt(
                &messages,
                &tools,
                Lfm3AudioReplyOutput::Interleaved,
                false
            )),
            format!(
                "<|startoftext|><|im_start|>system\n\
                 Respond with function calls, and interleaved text and audio.\n\n\
                 Be brief.\n\nList of tools:\n\n\
                 <tools>\n{{\"name\": \"get_time\"}}\n</tools><|im_end|>\n{turns}"
            )
        );
        // Only deferred tools: nothing to list.
        assert_eq!(
            text(&reply_prompt(
                &[message(ChatRole::User, "Hi")],
                &tools[1..],
                Lfm3AudioReplyOutput::Text,
                false
            )),
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        // Without a system message the tools are the system turn.
        assert_eq!(
            text(&reply_prompt(
                &[message(ChatRole::User, "Hi")],
                &tools,
                Lfm3AudioReplyOutput::Text,
                false
            )),
            "<|startoftext|><|im_start|>system\n<tools>\n{\"name\": \"get_time\"}\n</tools>\
             <|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}

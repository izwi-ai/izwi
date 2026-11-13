//! The session's conversation: items the client and the responses add, in
//! order, with OpenAI's insertion rules (`previous_item_id`, `"root"` for the
//! head), delete and truncate.
//!
//! The store lives as long as the socket. A user audio item keeps its PCM so a
//! unified model can hear it again (a tool follow-up), within a budget; older
//! items keep only their transcripts.

use std::sync::Arc;

use serde_json::{json, Value};

use super::events::RealtimeError;
use crate::app::voice_realtime::TurnControl;

/// The user audio the store keeps across all items, in samples (about 5
/// minutes at 24 kHz). Older items lose their audio first.
const AUDIO_BUDGET_SAMPLES: usize = 24_000 * 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn parse(value: Option<&str>) -> Result<Self, RealtimeError> {
        match value {
            Some("system" | "developer") => Ok(Self::System),
            Some("user") => Ok(Self::User),
            Some("assistant") => Ok(Self::Assistant),
            other => Err(RealtimeError::invalid(
                "invalid_value",
                format!("Invalid message role '{}'.", other.unwrap_or("")),
            )
            .with_param("item.role")),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One content part of a message.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ContentPart {
    InputText(String),
    /// Spoken input; its transcript once known.
    InputAudio {
        transcript: Option<String>,
    },
    /// An image the models here cannot see: kept, never prompted.
    InputImage,
    OutputText(String),
    /// Spoken output and its transcript (what was heard of it once
    /// truncated).
    OutputAudio {
        transcript: String,
    },
}

impl ContentPart {
    fn parse(value: &Value) -> Result<Self, RealtimeError> {
        let text = || {
            value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        match value.get("type").and_then(Value::as_str) {
            Some("input_text" | "text") => Ok(Self::InputText(text())),
            Some("output_text") => Ok(Self::OutputText(text())),
            Some("input_audio" | "audio") => Ok(Self::InputAudio {
                transcript: value
                    .get("transcript")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            }),
            Some("output_audio") => Ok(Self::OutputAudio {
                transcript: value
                    .get("transcript")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            }),
            Some("input_image") => Ok(Self::InputImage),
            other => Err(RealtimeError::invalid(
                "invalid_value",
                format!("Invalid content type '{}'.", other.unwrap_or("")),
            )
            .with_param("item.content")),
        }
    }

    /// The part as sent back: audio bytes are never echoed.
    fn to_json(&self) -> Value {
        match self {
            Self::InputText(text) => json!({"type": "input_text", "text": text}),
            Self::InputAudio { transcript } => {
                json!({"type": "input_audio", "transcript": transcript})
            }
            Self::InputImage => json!({"type": "input_image"}),
            Self::OutputText(text) => json!({"type": "output_text", "text": text}),
            Self::OutputAudio { transcript } => {
                json!({"type": "output_audio", "transcript": transcript})
            }
        }
    }

    /// The part's words, for the prompt.
    fn words(&self) -> Option<&str> {
        match self {
            Self::InputText(text) | Self::OutputText(text) => Some(text),
            Self::InputAudio { transcript } => transcript.as_deref(),
            Self::OutputAudio { transcript } => Some(transcript),
            Self::InputImage => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ItemKind {
    Message {
        role: Role,
        content: Vec<ContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    /// A call the server made to an MCP server's tool, and what came back.
    McpCall {
        server_label: String,
        name: String,
        arguments: String,
        output: Option<String>,
        /// The tool's or the protocol's error message.
        error: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

impl ItemStatus {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
        }
    }
}

/// User speech an item holds: samples at `sample_rate`.
#[derive(Debug, Clone)]
pub(crate) struct ItemAudio {
    pub(crate) samples: Arc<Vec<i16>>,
    pub(crate) sample_rate: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct Item {
    pub(crate) id: String,
    pub(crate) kind: ItemKind,
    pub(crate) status: ItemStatus,
    /// A user audio item's speech, until the budget drops it.
    pub(crate) audio: Option<ItemAudio>,
    /// An assistant audio reply's control: its spoken ledger answers
    /// `conversation.item.truncate`.
    pub(crate) reply: Option<TurnControl>,
    /// Milliseconds of audio a spoken reply sent.
    pub(crate) audio_ms: u64,
    /// Turns `true` once a user audio item's transcription has finished.
    pub(crate) transcript_ready: Option<tokio::sync::watch::Receiver<bool>>,
}

impl Item {
    pub(crate) fn new(id: String, kind: ItemKind, status: ItemStatus) -> Self {
        Self {
            id,
            kind,
            status,
            audio: None,
            reply: None,
            audio_ms: 0,
            transcript_ready: None,
        }
    }

    /// An item from a `conversation.item.create`. The client may name the
    /// id; `fresh_id` supplies one otherwise.
    pub(crate) fn from_client(
        value: &Value,
        fresh_id: impl FnOnce() -> String,
    ) -> Result<Self, RealtimeError> {
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(fresh_id);
        if id.len() > 64 {
            return Err(RealtimeError::invalid(
                "invalid_value",
                "Item ids are at most 64 characters.",
            )
            .with_param("item.id"));
        }
        let kind = match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
        {
            "message" => {
                let role = Role::parse(value.get("role").and_then(Value::as_str))?;
                let content = value
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .map(ContentPart::parse)
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                ItemKind::Message { role, content }
            }
            "function_call" => ItemKind::FunctionCall {
                call_id: value
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: value
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        RealtimeError::invalid(
                            "missing_required_parameter",
                            "A function_call item needs a 'name'.",
                        )
                        .with_param("item.name")
                    })?
                    .to_string(),
                arguments: match value.get("arguments") {
                    Some(Value::String(arguments)) => arguments.clone(),
                    None | Some(Value::Null) => "{}".to_string(),
                    Some(other) => other.to_string(),
                },
            },
            "function_call_output" => ItemKind::FunctionCallOutput {
                call_id: value
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|call_id| !call_id.is_empty())
                    .ok_or_else(|| {
                        RealtimeError::invalid(
                            "missing_required_parameter",
                            "A function_call_output item needs a 'call_id'.",
                        )
                        .with_param("item.call_id")
                    })?
                    .to_string(),
                output: match value.get("output") {
                    Some(Value::String(output)) => output.clone(),
                    None | Some(Value::Null) => String::new(),
                    Some(other) => other.to_string(),
                },
            },
            other => {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    format!("Item type '{other}' is not supported."),
                )
                .with_param("item.type"))
            }
        };
        Ok(Self::new(id, kind, ItemStatus::Completed))
    }

    /// The item as OpenAI sends it, without audio bytes.
    pub(crate) fn to_json(&self) -> Value {
        let mut item = match &self.kind {
            ItemKind::Message { role, content } => json!({
                "type": "message",
                "role": role.as_str(),
                "content": content.iter().map(ContentPart::to_json).collect::<Vec<_>>(),
            }),
            ItemKind::FunctionCall {
                call_id,
                name,
                arguments,
            } => json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }),
            ItemKind::FunctionCallOutput { call_id, output } => json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }),
            ItemKind::McpCall {
                server_label,
                name,
                arguments,
                output,
                error,
            } => json!({
                "type": "mcp_call",
                "server_label": server_label,
                "name": name,
                "arguments": arguments,
                "output": output,
                "error": error.as_ref().map(|message| json!({
                    "type": "tool_execution_error",
                    "message": message,
                })),
                "approval_request_id": null,
            }),
        };
        item["id"] = json!(self.id);
        item["object"] = json!("realtime.item");
        item["status"] = json!(self.status.as_str());
        item
    }

    /// The item's text for the prompt; `None` when it has none.
    pub(crate) fn text(&self) -> Option<String> {
        let ItemKind::Message { content, .. } = &self.kind else {
            return None;
        };
        let text = content
            .iter()
            .filter_map(ContentPart::words)
            .map(str::trim)
            .filter(|words| !words.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        (!text.is_empty()).then_some(text)
    }

    pub(crate) fn role(&self) -> Option<Role> {
        match &self.kind {
            ItemKind::Message { role, .. } => Some(*role),
            _ => None,
        }
    }

    /// Whether this is user speech still waiting for its transcript.
    pub(crate) fn awaits_transcript(&self) -> bool {
        matches!(&self.kind, ItemKind::Message { content, .. }
            if content.iter().any(|part| matches!(part, ContentPart::InputAudio { transcript: None })))
    }

    /// Set a user audio item's transcript.
    pub(crate) fn set_transcript(&mut self, text: &str) {
        if let ItemKind::Message { content, .. } = &mut self.kind {
            for part in content {
                if let ContentPart::InputAudio { transcript } = part {
                    *transcript = Some(text.to_string());
                }
            }
        }
    }
}

/// Where a `conversation.item.create` puts its item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Position {
    End,
    Root,
    After(String),
}

impl Position {
    pub(crate) fn parse(previous_item_id: Option<&str>) -> Self {
        match previous_item_id.map(str::trim) {
            None | Some("") => Self::End,
            Some("root") => Self::Root,
            Some(id) => Self::After(id.to_string()),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ItemStore {
    items: Vec<Item>,
}

impl ItemStore {
    /// Insert `item` at `position`; returns the id of the item before it.
    pub(crate) fn insert(
        &mut self,
        item: Item,
        position: &Position,
    ) -> Result<Option<String>, RealtimeError> {
        if self.index_of(&item.id).is_some() {
            return Err(RealtimeError::invalid(
                "item_create_duplicate_item_id",
                format!("Item with id '{}' already exists.", item.id),
            )
            .with_param("item.id"));
        }
        let index = match position {
            Position::End => self.items.len(),
            Position::Root => 0,
            Position::After(previous) => self
                .index_of(previous)
                .map(|index| index + 1)
                .ok_or_else(|| {
                    RealtimeError::invalid(
                        "previous_item_not_found",
                        format!("Previous item with id '{previous}' not found."),
                    )
                    .with_param("previous_item_id")
                })?,
        };
        self.items.insert(index, item);
        self.enforce_audio_budget();
        Ok(index
            .checked_sub(1)
            .map(|previous| self.items[previous].id.clone()))
    }

    pub(crate) fn delete(&mut self, id: &str) -> Option<Item> {
        let index = self.index_of(id)?;
        Some(self.items.remove(index))
    }

    pub(crate) fn get(&self, id: &str) -> Option<&Item> {
        self.index_of(id).map(|index| &self.items[index])
    }

    pub(crate) fn get_mut(&mut self, id: &str) -> Option<&mut Item> {
        self.index_of(id).map(|index| &mut self.items[index])
    }

    pub(crate) fn items(&self) -> &[Item] {
        &self.items
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.items.iter().position(|item| item.id == id)
    }

    /// Drop the audio of the oldest user items past the budget.
    fn enforce_audio_budget(&mut self) {
        let mut kept = 0usize;
        for item in self.items.iter_mut().rev() {
            if let Some(audio) = item.audio.as_ref() {
                kept += audio.samples.len();
                if kept > AUDIO_BUDGET_SAMPLES {
                    item.audio = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_item(id: &str, role: &str, text: &str) -> Item {
        Item::from_client(
            &json!({"id": id, "type": "message", "role": role,
                    "content": [{"type": "input_text", "text": text}]}),
            || unreachable!(),
        )
        .expect("item")
    }

    #[test]
    fn items_insert_after_their_previous_item_and_at_the_root() {
        let mut store = ItemStore::default();
        assert_eq!(
            store
                .insert(text_item("a", "user", "one"), &Position::End)
                .expect("a"),
            None
        );
        assert_eq!(
            store
                .insert(text_item("c", "assistant", "three"), &Position::End)
                .expect("c"),
            Some("a".to_string())
        );
        assert_eq!(
            store
                .insert(
                    text_item("b", "user", "two"),
                    &Position::After("a".to_string())
                )
                .expect("b"),
            Some("a".to_string())
        );
        assert_eq!(
            store
                .insert(text_item("z", "system", "zero"), &Position::Root)
                .expect("z"),
            None
        );
        let ids: Vec<&str> = store.items().iter().map(|item| item.id.as_str()).collect();
        assert_eq!(ids, ["z", "a", "b", "c"]);

        let err = store
            .insert(text_item("a", "user", "again"), &Position::End)
            .expect_err("duplicate");
        assert_eq!(err.code.as_deref(), Some("item_create_duplicate_item_id"));
        let err = store
            .insert(
                text_item("d", "user", "x"),
                &Position::After("nope".to_string()),
            )
            .expect_err("unknown previous");
        assert_eq!(err.param.as_deref(), Some("previous_item_id"));
        assert!(store.delete("b").is_some());
        assert!(store.delete("b").is_none());
    }

    #[test]
    fn client_items_parse_like_livekit_and_pipecat_send_them() {
        // Pipecat sends assistant text as `input_text`.
        let item = text_item("x", "assistant", "Hello");
        assert_eq!(item.text().as_deref(), Some("Hello"));
        let call = Item::from_client(
            &json!({"type": "function_call", "call_id": "call_1", "name": "get_weather",
                    "arguments": "{\"city\":\"Harare\"}"}),
            || "item_fresh".to_string(),
        )
        .expect("call");
        assert_eq!(call.id, "item_fresh");
        let output = Item::from_client(
            &json!({"type": "function_call_output", "call_id": "call_1", "output": {"temp": 21}}),
            || "item_out".to_string(),
        )
        .expect("output");
        assert!(
            matches!(output.kind, ItemKind::FunctionCallOutput { ref output, .. } if output == "{\"temp\":21}")
        );
        let audio = Item::from_client(
            &json!({"type": "message", "role": "user",
                    "content": [{"type": "input_audio", "audio": "AAA=", "transcript": "hi"}]}),
            || "item_a".to_string(),
        )
        .expect("audio");
        assert_eq!(audio.text().as_deref(), Some("hi"));
        let json = audio.to_json();
        assert_eq!(json["object"], "realtime.item");
        assert!(
            json["content"][0].get("audio").is_none(),
            "audio bytes are not echoed"
        );
        assert!(Item::from_client(&json!({"type": "mcp_call"}), || "i".to_string()).is_err());
    }

    #[test]
    fn old_user_audio_leaves_the_store_first() {
        let mut store = ItemStore::default();
        for index in 0..3 {
            let mut item = Item::new(
                format!("u{index}"),
                ItemKind::Message {
                    role: Role::User,
                    content: vec![ContentPart::InputAudio {
                        transcript: Some("x".into()),
                    }],
                },
                ItemStatus::Completed,
            );
            item.audio = Some(ItemAudio {
                samples: Arc::new(vec![0; AUDIO_BUDGET_SAMPLES / 2]),
                sample_rate: 24_000,
            });
            store.insert(item, &Position::End).expect("insert");
        }
        let kept: Vec<bool> = store
            .items()
            .iter()
            .map(|item| item.audio.is_some())
            .collect();
        assert_eq!(kept, [false, true, true]);
    }
}

use izwi_core::{ChatReasoningEffort, ChatTemplateKwargs, ChatTruncation};
use serde::{Deserialize, Serialize};

use crate::api::openai::convert::InboundContent;

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesCreateRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<ResponseInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// `-1` (vLLM) or `0` disables top-k sampling.
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ChatReasoningEffort>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// OpenAI `truncation`: `"disabled"` (the default) rejects input that
    /// does not fit the context; `"auto"` drops the oldest turns instead.
    #[serde(default)]
    pub truncation: Option<ChatTruncation>,
    /// vLLM's OpenAI extension: `false` keeps the tokenizer's special tokens
    /// in the reply text. Absent or `true` skips them.
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    One(Box<ResponseInputItem>),
    Many(Vec<ResponseInputItem>),
}

/// An input item: a message (the default), a `function_call` the model
/// made in an earlier turn, or the `function_call_output` answering it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponseInputItem {
    #[serde(default, rename = "type")]
    pub item_type: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<InboundContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// A `function_call`'s JSON-encoded arguments.
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
    /// A `function_call_output`'s result: a string, or content parts.
    #[serde(default)]
    pub output: Option<serde_json::Value>,
    /// On a tool result, for LFM3 models: deferred tools it reveals (their
    /// definitions, or a flag that the output holds them).
    #[serde(default)]
    pub additional_tools: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub usage: ResponseUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message {
        id: String,
        role: &'static str,
        content: Vec<ResponseOutputContent>,
    },
    /// A tool call the client runs and answers with a
    /// `function_call_output` input item carrying the same `call_id`.
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponseOutputContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    /// DS9.1: OpenAI-shape input token details. Always present with usage;
    /// `cached_tokens` is 0 when the serving runtime did not measure prefix
    /// reuse.
    pub input_tokens_details: ResponseInputTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseInputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseError {
    pub message: String,
    pub code: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ResponseDeletedObject {
    pub id: String,
    pub object: &'static str,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct ResponseInputItemsList {
    pub object: &'static str,
    /// Message items, and function call items as they were sent.
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResponseInputItemObject {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub role: String,
    pub content: Vec<ResponseInputItemContent>,
}

#[derive(Debug, Serialize)]
pub struct ResponseInputItemContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseStreamEnvelope<T>
where
    T: Serialize,
{
    #[serde(rename = "type")]
    pub event_type: &'static str,
    #[serde(flatten)]
    pub payload: T,
}

#[derive(Debug, Serialize)]
pub struct ResponseStreamCreatedPayload {
    pub response: ResponseObject,
}

#[derive(Debug, Serialize)]
pub struct ResponseStreamDeltaPayload {
    pub response_id: String,
    pub item_id: String,
    pub output_index: usize,
    pub content_index: usize,
    pub delta: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseStreamCompletedPayload {
    pub response: ResponseObject,
}

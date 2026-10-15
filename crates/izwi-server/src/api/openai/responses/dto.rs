use izwi_core::{ChatReasoningEffort, ChatTemplateKwargs, ChatTruncation};
use serde::{Deserialize, Serialize};

use crate::api::openai::convert::InboundContent;

#[allow(dead_code)]
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
    pub user: Option<String>,
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    One(ResponseInputItem),
    Many(Vec<ResponseInputItem>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseInputItem {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<InboundContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
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

#[derive(Debug, Clone, Serialize)]
pub struct ResponseOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ResponseOutputContent>,
}

#[derive(Debug, Clone, Serialize)]
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
    pub data: Vec<ResponseInputItemObject>,
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
    pub delta: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseStreamCompletedPayload {
    pub response: ResponseObject,
}

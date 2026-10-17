//! The OpenAI chat completion wire types: the request, the response and the stream chunk.

use serde::{Deserialize, Serialize};

use crate::api::openai::convert::InboundContent;
use crate::api::openai::tool_calls::{ToolCall, ToolCallDelta};
use izwi_model_catalog::{ChatReasoningEffort, ChatTemplateKwargs, ChatTruncation};

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiInboundMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub n: Option<usize>,
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
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Sampling seed (a signed 64-bit integer, as in OpenAI and vLLM).
    #[serde(default)]
    pub seed: Option<i64>,
    /// DS9.3: include per-token logprobs in the response.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// DS9.3: number of top alternatives per token (0-20).
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// DS9.2: structured output request. `json_object` constrains generation
    /// to one valid JSON value on models whose sampler enforces it.
    #[serde(default)]
    pub response_format: Option<OpenAiResponseFormat>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
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
    /// `"parsed"` moves `<think>` reasoning into `reasoning_content`;
    /// `"none"` (the default) leaves it inside `content`.
    #[serde(default)]
    pub reasoning_format: Option<String>,
    /// Izwi extension mirroring OpenAI Responses: `"disabled"` (the default)
    /// rejects a request that does not fit the context with
    /// `context_length_exceeded`; `"auto"` drops the oldest turns instead.
    #[serde(default)]
    pub truncation: Option<ChatTruncation>,
    /// vLLM's OpenAI extension: `false` keeps the tokenizer's special tokens
    /// in the reply text. Absent or `true` skips them.
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

/// DS9.2: OpenAI `response_format`. Only `json_object` is supported;
/// `json_schema` is rejected by `kind` with a documented error, and any
/// `json_schema` payload key is ignored by deserialization.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiInboundMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<InboundContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// On a `tool` message: the id of the call it answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: OpenAiUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub izwi_generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub izwi_timing: Option<izwi_model_catalog::LatencyBreakdown>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChoice {
    pub index: usize,
    pub message: OpenAiAssistantMessage,
    pub finish_reason: &'static str,
    /// DS9.3: per-token logprobs; absent unless the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<OpenAiChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiAssistantMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// DS9.1: OpenAI-shape prompt token details. Always present with usage;
    /// `cached_tokens` is 0 when the serving runtime did not measure prefix
    /// reuse.
    pub prompt_tokens_details: OpenAiPromptTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiPromptTokensDetails {
    pub cached_tokens: u64,
}

/// DS9.3: OpenAI-shape per-choice logprobs wrapper.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChoiceLogprobs {
    pub content: Vec<OpenAiTokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<OpenAiTopTokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiTopTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

/// DS9.3: map core logprob entries into the OpenAI response shape. `None`
/// when the request did not collect logprobs.
pub fn openai_logprobs(
    entries: &[izwi_model_catalog::TokenLogprob],
) -> Option<OpenAiChoiceLogprobs> {
    if entries.is_empty() {
        return None;
    }
    Some(OpenAiChoiceLogprobs {
        content: entries
            .iter()
            .map(|entry| OpenAiTokenLogprob {
                token: entry.token.clone(),
                logprob: entry.logprob,
                bytes: entry.bytes.clone(),
                top_logprobs: entry
                    .top_logprobs
                    .iter()
                    .map(|top| OpenAiTopTokenLogprob {
                        token: top.token.clone(),
                        logprob: top.logprob,
                        bytes: top.bytes.clone(),
                    })
                    .collect(),
            })
            .collect(),
    })
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChunkChoice>,
    /// With `stream_options.include_usage`: `null` on every chunk but the
    /// last, which carries the usage and no choices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<OpenAiUsage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub izwi_generation_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub izwi_timing: Option<izwi_model_catalog::LatencyBreakdown>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChunkChoice {
    pub index: usize,
    pub delta: OpenAiDelta,
    /// DS9.3: the logprobs of this chunk's tokens; absent unless requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<OpenAiChoiceLogprobs>,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InboundMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl OpenAiDelta {
    pub fn is_empty(&self) -> bool {
        self.role.is_none()
            && self.reasoning_content.is_none()
            && self.content.is_none()
            && self.tool_calls.is_none()
    }
}

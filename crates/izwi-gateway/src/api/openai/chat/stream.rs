//! Rendering a chat generation as OpenAI chat completion SSE chunks.

use std::convert::Infallible;

use axum::response::{sse::Event, Sse};
use futures::stream::Stream;

use super::completions::{finish_reason, now_unix_secs, AnswerFormat};
use super::dto::{
    openai_logprobs, ChatCompletionRequest, OpenAiChatChunk, OpenAiChunkChoice, OpenAiDelta,
    OpenAiPromptTokensDetails, OpenAiUsage,
};
use crate::api::openai::tool_calls::{ParsedOutput, ToolCallParser};
use crate::api::request_context::StreamOutcome;
use crate::app::chat::{ChatStreamEvent, ChatStreamObserver};
use crate::error::{ApiError, ErrorScope};
use crate::ids::new_prefixed_id;
use izwi_model_catalog::ChatGeneration;

const CHAT_STREAM_INTERRUPTED_ERROR: &str = "Chat stream ended before a terminal event";

/// An in-band error event: the OpenAI error object of the HTTP error the
/// request would have had before the stream started.
fn openai_chat_stream_error_payload(scope: &ErrorScope, error: &ApiError) -> String {
    scope.openai_body(error).to_string()
}

pub fn openai_chat_stream_interruption_payload(
    scope: &ErrorScope,
    saw_terminal: bool,
) -> Option<String> {
    (!saw_terminal).then(|| {
        openai_chat_stream_error_payload(scope, &ApiError::internal(CHAT_STREAM_INTERRUPTED_ERROR))
    })
}

/// Tells the request audit how a stream ended in-band.
pub struct AuditedStream<O> {
    pub observer: O,
    pub outcome: StreamOutcome,
}

impl<O: ChatStreamObserver> ChatStreamObserver for AuditedStream<O> {
    fn record_delta(&mut self) {
        self.observer.record_delta();
    }

    fn record_completion(&mut self, generation: &ChatGeneration) {
        self.outcome.record_success();
        self.observer.record_completion(generation);
    }

    fn record_failure(&mut self) {
        self.outcome.record_failure();
        self.observer.record_failure();
    }
}

/// Streamed answer text on its way to the client: reasoning is split off
/// first, then tool-call markup becomes `tool_calls`.
struct AnswerStream {
    splitter: Option<super::reasoning::ReasoningSplitter>,
    tool_calls: Option<ToolCallParser>,
}

impl AnswerStream {
    fn new(format: AnswerFormat) -> Self {
        Self {
            splitter: format
                .reasoning
                .map(super::reasoning::ReasoningSplitter::new),
            tool_calls: format.tool_calls.map(ToolCallParser::new),
        }
    }

    /// The delta for one piece of model text.
    fn push(&mut self, text: &str) -> OpenAiDelta {
        let (reasoning, content) = match self.splitter.as_mut() {
            Some(splitter) => splitter.push(text),
            None => (String::new(), text.to_string()),
        };
        let parsed = match self.tool_calls.as_mut() {
            Some(parser) => parser.push(&content),
            None => ParsedOutput {
                content,
                tool_calls: Vec::new(),
            },
        };
        answer_delta(reasoning, parsed)
    }

    /// The delta for the text held back at the end of the answer.
    fn finish(&mut self) -> OpenAiDelta {
        let (reasoning, content) = self
            .splitter
            .as_mut()
            .map(|splitter| splitter.finish())
            .unwrap_or_default();
        let parsed = match self.tool_calls.as_mut() {
            Some(parser) => {
                let mut parsed = parser.push(&content);
                parsed.extend(parser.finish());
                parsed
            }
            None => ParsedOutput {
                content,
                tool_calls: Vec::new(),
            },
        };
        answer_delta(reasoning, parsed)
    }

    fn saw_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(ToolCallParser::saw_tool_calls)
    }
}

fn answer_delta(reasoning: String, parsed: ParsedOutput) -> OpenAiDelta {
    OpenAiDelta {
        role: None,
        reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
        content: (!parsed.content.is_empty()).then_some(parsed.content),
        tool_calls: (!parsed.tool_calls.is_empty()).then_some(parsed.tool_calls),
    }
}

pub fn render_chat_stream(
    req: ChatCompletionRequest,
    model_id: String,
    mut event_rx: tokio::sync::mpsc::Receiver<ChatStreamEvent>,
    format: AnswerFormat,
    observer: impl ChatStreamObserver,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let completion_id = new_prefixed_id("chatcmpl-");
    let created = now_unix_secs();
    let chunk = move |choices: Vec<OpenAiChunkChoice>| OpenAiChatChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_id.clone(),
        choices,
        usage: include_usage.then_some(None),
        izwi_generation_time_ms: None,
        izwi_timing: None,
    };
    let choice = |delta, logprobs, finish_reason| {
        vec![OpenAiChunkChoice {
            index: 0,
            delta,
            logprobs,
            finish_reason,
        }]
    };

    // The body is polled after the handler returns, outside the request's
    // error scope; capture it here.
    let errors = ErrorScope::current();
    let stream = async_stream::stream! {
        let mut observer = observer;
        let mut answer = AnswerStream::new(format);
        let mut saw_terminal = false;
        while let Some(event) = event_rx.recv().await {
            let (payloads, terminal) = match event {
                ChatStreamEvent::Started => {
                    let delta = OpenAiDelta {
                        role: Some("assistant"),
                        reasoning_content: None,
                        content: None,
                        tool_calls: None,
                    };
                    (vec![chunk_json(&chunk(choice(delta, None, None)))], false)
                }
                ChatStreamEvent::Delta { text, logprobs } => {
                    if !text.is_empty() {
                        observer.record_delta();
                    }
                    let delta = answer.push(&text);
                    let logprobs = openai_logprobs(&logprobs);
                    if delta.is_empty() && logprobs.is_none() {
                        (Vec::new(), false)
                    } else {
                        (vec![chunk_json(&chunk(choice(delta, logprobs, None)))], false)
                    }
                }
                ChatStreamEvent::Completed(generation) => {
                    observer.record_completion(&generation);
                    // Text held back for a tag is final now.
                    let delta = answer.finish();
                    let finish = finish_reason(answer.saw_tool_calls(), generation.finish_reason);
                    let mut last = chunk(choice(delta, None, Some(finish)));
                    if format.compat_profile.is_relaxed() {
                        last.izwi_generation_time_ms = Some(generation.generation_time_ms);
                        last.izwi_timing = generation.latency_breakdown;
                    }
                    let mut payloads = vec![chunk_json(&last)];
                    if include_usage {
                        let mut usage = chunk(Vec::new());
                        usage.usage = Some(Some(OpenAiUsage {
                            prompt_tokens: generation.prompt_tokens,
                            completion_tokens: generation.tokens_generated,
                            total_tokens: generation.prompt_tokens + generation.tokens_generated,
                            prompt_tokens_details: OpenAiPromptTokensDetails {
                                cached_tokens: generation.cached_prompt_tokens.unwrap_or(0),
                            },
                        }));
                        payloads.push(chunk_json(&usage));
                    }
                    (payloads, true)
                }
                ChatStreamEvent::Failed(error) => {
                    observer.record_failure();
                    (vec![openai_chat_stream_error_payload(&errors, &error)], true)
                }
            };
            for payload in payloads {
                yield Ok(Event::default().data(payload));
            }
            if terminal {
                saw_terminal = true;
                break;
            }
        }
        if let Some(payload) = openai_chat_stream_interruption_payload(&errors, saw_terminal) {
            observer.record_failure();
            yield Ok(Event::default().data(payload));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

fn chunk_json(chunk: &OpenAiChatChunk) -> String {
    serde_json::to_string(chunk).unwrap_or_default()
}

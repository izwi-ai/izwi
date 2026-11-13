//! What the server gives `izwi-agent`'s turn runtime: a chat model backend
//! on the runtime service, the stored thread as memory, and a turn's
//! cancellation tied to a voice turn's control.

use std::sync::{Arc, Mutex};

use izwi_agent::{
    AgentError, MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore, ModelBackend,
    ModelOutput, ModelRequest, ToolCall, TurnCancel, TurnEvents,
};
use izwi_core::{
    parse_chat_model_variant, ChatFinishReason, ChatMessage, ChatRequestConfig, ChatTruncation,
    GenerationParams, ModelVariant, RuntimeRequestContext, RuntimeService,
};
use serde_json::Value;

use crate::api::openai::convert::InboundContent;
use crate::api::openai::responses::dto::{ResponseInput, ResponseInputItem};
use crate::api::openai::responses::handlers::build_input_messages;
use crate::api::openai::tool_calls::{parse_answer, ToolCallFormat, ToolChoice, ToolPlan};
use crate::app::voice_realtime::{acquire_stage_permit, TurnControl};
use crate::chat_store::ChatStore;
use crate::state::AppState;

/// Where a model round gets its admission.
pub(crate) enum RoundAdmission<'a> {
    /// The caller holds a permit for the whole turn.
    Held(RuntimeRequestContext),
    /// Each round takes its own realtime permit, so none is held while a
    /// tool runs or a client answers.
    PerRound(&'a AppState),
}

/// A chat model as the agent's backend: a streamed reply without reasoning
/// (voice cannot speak thinking, and waiting for it would hold back the
/// first audio), its visible text sent as it is generated and its tool
/// calls parsed by the model family's call format.
pub(crate) struct ChatAgentBackend<'a> {
    pub(crate) runtime: Arc<RuntimeService>,
    pub(crate) correlation_id: String,
    pub(crate) admission: RoundAdmission<'a>,
}

/// The tools a round offers `variant`: none for a family without a
/// tool-calling chat template.
pub(crate) fn tool_plan_for(variant: ModelVariant, tools: &[Value]) -> Option<ToolPlan> {
    if tools.is_empty() {
        return None;
    }
    let dialect = ToolCallFormat::of(variant)?;
    ToolPlan::new(dialect, tools.to_vec(), ToolChoice::Auto, true).ok()
}

#[async_trait::async_trait]
impl ModelBackend for ChatAgentBackend<'_> {
    async fn generate(
        &self,
        request: ModelRequest,
        events: &TurnEvents,
    ) -> izwi_agent::Result<ModelOutput> {
        let model_error = |err: String| AgentError::Model(err);
        let variant = parse_chat_model_variant(Some(&request.model_id))
            .map_err(|err| model_error(err.to_string()))?;
        let plan = tool_plan_for(variant, &request.tools);
        let tools = plan
            .as_ref()
            .map(|plan| plan.tools.clone())
            .unwrap_or_default();
        let messages = chat_messages(variant, request.messages, &tools).map_err(model_error)?;

        let permit;
        let runtime_context = match &self.admission {
            RoundAdmission::Held(context) => *context,
            RoundAdmission::PerRound(state) => {
                permit = acquire_stage_permit(state).await.map_err(model_error)?;
                permit.runtime_context()
            }
        };

        let parser = Arc::new(Mutex::new(plan.as_ref().map(ToolPlan::parser)));
        let (parser_sink, events_sink) = (parser.clone(), events.clone());
        let generation = self
            .runtime
            .chat_generate_streaming_with_runtime_context(
                variant,
                messages,
                GenerationParams {
                    max_tokens: request.max_output_tokens.clamp(1, 8_192),
                    ..Default::default()
                },
                ChatRequestConfig {
                    enable_thinking: Some(false),
                    tools,
                    truncation: ChatTruncation::Auto,
                    ..ChatRequestConfig::default()
                },
                Some(&self.correlation_id),
                runtime_context,
                move |delta| {
                    let visible = match parser_sink
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .as_mut()
                    {
                        Some(parser) => parser.push(&delta).content,
                        None => delta,
                    };
                    events_sink.text_delta(&visible);
                },
            )
            .await
            .map_err(|err| model_error(err.to_string()))?;

        // The parser holds back text that could still open a call.
        let tail = parser
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
            .map(|mut parser| parser.finish().content)
            .unwrap_or_default();
        events.text_delta(&tail);

        let (text, tool_calls) = match &plan {
            Some(plan) => {
                let (content, calls) = parse_answer(plan.parser(), &generation.text);
                let calls = calls
                    .into_iter()
                    .map(|call| ToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments: call.function.arguments,
                    })
                    .collect();
                (content.unwrap_or_default(), calls)
            }
            None => (generation.text, Vec::new()),
        };
        Ok(ModelOutput {
            text,
            tool_calls,
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
            truncated: generation.finish_reason == Some(ChatFinishReason::MaxTokens),
        })
    }
}

/// The agent's messages as the runtime's, through the Responses API's item
/// conversion: tool calls and results are rendered as the model's template
/// expects.
pub(crate) fn chat_messages(
    variant: ModelVariant,
    messages: Vec<MemoryMessage>,
    tools: &[Value],
) -> Result<Vec<ChatMessage>, String> {
    let mut system = None;
    let mut items = Vec::with_capacity(messages.len());
    for message in messages {
        match message.role {
            MemoryMessageRole::System => {
                let system: &mut String = system.get_or_insert_default();
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&message.content);
            }
            MemoryMessageRole::Tool => items.push(ResponseInputItem {
                item_type: Some("function_call_output".to_string()),
                call_id: message.tool_call_id,
                output: Some(Value::String(message.content)),
                ..ResponseInputItem::default()
            }),
            MemoryMessageRole::User | MemoryMessageRole::Assistant => {
                if !message.content.is_empty() || message.tool_calls.is_empty() {
                    items.push(ResponseInputItem {
                        role: Some(message.role.as_str().to_string()),
                        content: Some(InboundContent::Text(message.content)),
                        ..ResponseInputItem::default()
                    });
                }
                items.extend(
                    message
                        .tool_calls
                        .into_iter()
                        .map(|call| ResponseInputItem {
                            item_type: Some("function_call".to_string()),
                            call_id: Some(call.id),
                            name: Some(call.name),
                            arguments: Some(Value::String(call.arguments)),
                            ..ResponseInputItem::default()
                        }),
                );
            }
        }
    }
    build_input_messages(
        variant,
        system.as_deref(),
        Some(ResponseInput::Many(items)),
        tools,
    )
    .map(|(messages, _, _)| messages)
    .map_err(|err| err.message)
}

/// A stored chat thread as the agent's memory, with the context a voice
/// profile remembers about its user ahead of it.
pub(crate) struct ThreadMemory {
    chat_store: Arc<ChatStore>,
    memory_context: Option<String>,
}

impl ThreadMemory {
    pub(crate) fn new(chat_store: Arc<ChatStore>, memory_context: Option<String>) -> Self {
        Self {
            chat_store,
            memory_context,
        }
    }
}

#[async_trait::async_trait]
impl MemoryStore for ThreadMemory {
    async fn load_messages(&self, thread_id: &str) -> izwi_agent::Result<Vec<MemoryMessage>> {
        let records = self
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .map_err(|err| AgentError::Memory(err.to_string()))?;

        let mut out = Vec::with_capacity(records.len() + 1);
        if let Some(memory_context) = self.memory_context.clone() {
            out.push(MemoryMessage::system(memory_context));
        }
        for record in records {
            let role = match record.role.as_str() {
                "system" => MemoryMessageRole::System,
                "user" => MemoryMessageRole::User,
                "assistant" => MemoryMessageRole::Assistant,
                other => {
                    return Err(AgentError::Memory(format!(
                        "Invalid stored chat role: {other}"
                    )));
                }
            };
            out.push(MemoryMessage::new(role, record.content));
        }
        Ok(out)
    }

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> izwi_agent::Result<()> {
        self.chat_store
            .append_message(
                thread_id.to_string(),
                role.as_str().to_string(),
                content,
                None,
                meta.model_id,
                meta.tokens_generated,
                meta.generation_time_ms,
            )
            .await
            .map_err(|err| AgentError::Memory(err.to_string()))?;
        Ok(())
    }
}

/// An agent turn's cancellation that follows a voice turn's control until
/// the guard drops.
pub(crate) fn cancel_with(control: &TurnControl) -> (TurnCancel, CancelLink) {
    let cancel = TurnCancel::new();
    let link = {
        let (cancel, control) = (cancel.clone(), control.clone());
        tokio::spawn(async move {
            control.cancelled().await;
            cancel.cancel();
        })
    };
    (cancel, CancelLink(link))
}

pub(crate) struct CancelLink(tokio::task::JoinHandle<()>);

impl Drop for CancelLink {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::ChatRole;

    #[test]
    fn a_turns_calls_and_results_render_through_the_models_template() {
        let messages = vec![
            MemoryMessage::system("Be brief."),
            MemoryMessage::system("The user is Ada."),
            MemoryMessage::user("What time is it?"),
            MemoryMessage::assistant_calls(
                "",
                vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "time".to_string(),
                    arguments: "{}".to_string(),
                }],
            ),
            MemoryMessage::tool("call_1", "It is noon."),
        ];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "time", "parameters": {"type": "object", "properties": {}}}
        })];
        let rendered = chat_messages(ModelVariant::Qwen354BGguf, messages, &tools).unwrap();
        let roles: Vec<_> = rendered
            .iter()
            .map(|message| message.role.clone())
            .collect();
        assert_eq!(
            roles,
            [
                ChatRole::System,
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::Tool
            ]
        );
        assert_eq!(rendered[0].content, "Be brief.\n\nThe user is Ada.");
        assert!(
            rendered[2].content.contains("time"),
            "{}",
            rendered[2].content
        );
        assert_eq!(rendered[3].content, "It is noon.");
    }

    #[test]
    fn a_family_without_a_tool_template_is_offered_no_tools() {
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "time"}})];
        assert!(tool_plan_for(ModelVariant::Qwen354BGguf, &tools).is_some());
        assert!(tool_plan_for(ModelVariant::Qwen354BGguf, &[]).is_none());
        assert!(tool_plan_for(ModelVariant::Gemma34BIt, &tools).is_none());
    }
}

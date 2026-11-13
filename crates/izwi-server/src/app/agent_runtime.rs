//! What the server gives `izwi-agent`'s turn runtime: a chat model backend
//! on the runtime service, the stored thread as memory, and a turn's
//! cancellation tied to a voice turn's control.

use std::sync::{Arc, Mutex};

use izwi_agent::{
    AgentError, AgentEvent, MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore,
    ModelBackend, ModelOutput, ModelRequest, ToolCall, TurnCancel, TurnEvents,
};
use izwi_core::{
    parse_chat_model_variant, ChatFinishReason, ChatMessage, ChatRequestConfig, ChatTruncation,
    GenerationParams, ModelVariant,
};
use serde_json::Value;

use crate::api::openai::convert::InboundContent;
use crate::api::openai::responses::dto::{ResponseInput, ResponseInputItem};
use crate::api::openai::responses::handlers::build_input_messages;
use crate::api::openai::tool_calls::{parse_answer, ToolCallFormat, ToolChoice, ToolPlan};
use crate::app::voice_realtime::{acquire_stage_permit, TurnControl};
use crate::chat_store::ChatStore;
use crate::state::AppState;

/// A chat model as the agent's backend: a streamed reply without reasoning
/// (voice cannot speak thinking, and waiting for it would hold back the
/// first audio), its visible text sent as it is generated and its tool
/// calls parsed by the model family's call format. Each round takes its
/// own realtime admission permit, so none is held while a tool runs or a
/// client answers.
pub(crate) struct ChatAgentBackend<'a> {
    state: &'a AppState,
    correlation_id: String,
    /// What the first round must call; later rounds are free.
    choice: ToolChoice,
    usage: Mutex<ChatUsage>,
}

/// The tokens a turn's rounds read and wrote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ChatUsage {
    pub(crate) prompt_tokens: u64,
    pub(crate) cached_prompt_tokens: u64,
    pub(crate) output_tokens: u64,
}

impl<'a> ChatAgentBackend<'a> {
    pub(crate) fn new(state: &'a AppState, correlation_id: impl Into<String>) -> Self {
        Self {
            state,
            correlation_id: correlation_id.into(),
            choice: ToolChoice::Auto,
            usage: Mutex::new(ChatUsage::default()),
        }
    }

    /// Open the turn's first reply with a call: any tool (`Required`) or a
    /// named one.
    pub(crate) fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.choice = choice;
        self
    }

    pub(crate) fn usage(&self) -> ChatUsage {
        *self
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

/// How a round uses the tools it offers `variant`: `None` for a family
/// without a tool-calling chat template, or when `choice` names a tool that
/// is not offered.
pub(crate) fn tool_plan_for(
    variant: ModelVariant,
    tools: &[Value],
    choice: ToolChoice,
) -> Option<ToolPlan> {
    if tools.is_empty() {
        return None;
    }
    let dialect = ToolCallFormat::of(variant)?;
    ToolPlan::new(dialect, tools.to_vec(), choice, true).ok()
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
        let choice = if request.round == 0 {
            self.choice.clone()
        } else {
            ToolChoice::Auto
        };
        let plan = tool_plan_for(variant, &request.tools, choice);
        let tools = plan
            .as_ref()
            .map(|plan| plan.tools.clone())
            .unwrap_or_default();
        let messages = chat_messages(variant, request.messages, &tools).map_err(model_error)?;

        let permit = acquire_stage_permit(self.state)
            .await
            .map_err(model_error)?;
        let runtime_context = permit.runtime_context();

        let parser = Arc::new(Mutex::new(plan.as_ref().map(ToolPlan::parser)));
        let (parser_sink, events_sink) = (parser.clone(), events.clone());
        let generation = self
            .state
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
                    response_prefix: plan
                        .as_ref()
                        .and_then(ToolPlan::response_prefix)
                        .unwrap_or_default(),
                    // A required or named tool is held to the declared
                    // names and schemas where the sampler can.
                    tool_constraint: plan
                        .as_ref()
                        .filter(|_| izwi_core::chat_supports_tool_constraint(variant))
                        .and_then(|plan| plan.constraint(false))
                        .map(Box::new),
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

        {
            let mut usage = self
                .usage
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            usage.prompt_tokens += generation.prompt_tokens as u64;
            usage.cached_prompt_tokens += generation.cached_prompt_tokens.unwrap_or(0);
            usage.output_tokens += generation.tokens_generated as u64;
        }
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
    summary: Option<ConversationSummary>,
}

impl ThreadMemory {
    pub(crate) fn new(chat_store: Arc<ChatStore>, memory_context: Option<String>) -> Self {
        Self {
            chat_store,
            memory_context,
            summary: None,
        }
    }

    /// Carry `summary` for the part of the thread a trimmed prompt leaves
    /// out.
    pub(crate) fn with_summary(mut self, summary: ConversationSummary) -> Self {
        self.summary = Some(summary);
        self
    }
}

/// How many more messages must fall out of the prompt before the summary
/// is written again.
const SUMMARY_STEP: usize = 6;
const SUMMARY_SYSTEM_PROMPT: &str = "You keep notes on a conversation for the assistant taking part in it. Write the notes it needs to carry on once the earlier messages are gone: who the user is, what they asked for, what was decided or done, and what is still open. Plain sentences, at most 120 words, no preamble.";

/// A conversation's running summary of the messages its prompt no longer
/// carries, kept for the life of a voice connection and rewritten in the
/// background as more messages fall out.
#[derive(Clone, Default)]
pub(crate) struct ConversationSummary {
    state: Arc<Mutex<SummaryState>>,
}

#[derive(Default)]
struct SummaryState {
    text: Option<String>,
    /// How many of the thread's oldest messages the text covers.
    covers: usize,
    writing: bool,
}

impl ConversationSummary {
    fn lock(&self) -> std::sync::MutexGuard<'_, SummaryState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub(crate) fn text(&self) -> Option<String> {
        self.lock().text.clone()
    }

    /// Forget the summary: the conversation started over.
    pub(crate) fn reset(&self) {
        *self.lock() = SummaryState::default();
    }

    /// The range of dropped messages to add to the summary, when `dropped`
    /// messages no longer fit the prompt and enough of them are new.
    fn claim(&self, dropped: usize) -> Option<std::ops::Range<usize>> {
        let mut state = self.lock();
        if state.writing || dropped < state.covers + SUMMARY_STEP {
            return None;
        }
        state.writing = true;
        Some(state.covers..dropped)
    }

    /// Rewrite the summary in the background when enough of `thread_id`'s
    /// messages have fallen out of `budget`.
    pub(crate) fn refresh(
        &self,
        state: &AppState,
        thread_id: &str,
        model_id: &str,
        correlation_id: &str,
        budget: izwi_agent::HistoryBudget,
    ) {
        let (summary, state) = (self.clone(), state.clone());
        let (thread_id, model_id, correlation_id) = (
            thread_id.to_string(),
            model_id.to_string(),
            correlation_id.to_string(),
        );
        tokio::spawn(async move {
            let memory = ThreadMemory::new(state.chat_store.clone(), None);
            let Ok(messages) = memory.load_messages(&thread_id).await else {
                return;
            };
            let dropped = messages.len() - budget.apply(messages.clone()).len();
            let Some(range) = summary.claim(dropped) else {
                return;
            };
            let earlier = summary.text();
            let written = write_summary(
                &state,
                &model_id,
                &correlation_id,
                summary_request(earlier.as_deref(), &messages[range]),
            )
            .await;
            let mut summary = summary.lock();
            summary.writing = false;
            match written {
                Ok(text) if !text.is_empty() => {
                    summary.text = Some(text);
                    summary.covers = dropped;
                }
                Ok(_) => {}
                Err(err) => tracing::warn!("conversation summary failed: {err}"),
            }
        });
    }
}

/// What the summary's writer reads: the notes so far and the messages to
/// add to them.
fn summary_request(earlier: Option<&str>, messages: &[MemoryMessage]) -> String {
    let mut request = String::new();
    if let Some(earlier) = earlier {
        request.push_str("Notes so far:\n");
        request.push_str(earlier);
        request.push_str("\n\nWhat was said next:\n");
    }
    for message in messages {
        let speaker = match message.role {
            MemoryMessageRole::User => "User",
            MemoryMessageRole::Assistant => "Assistant",
            MemoryMessageRole::System | MemoryMessageRole::Tool => continue,
        };
        request.push_str(&format!("{speaker}: {}\n", message.content.trim()));
    }
    request
}

async fn write_summary(
    state: &AppState,
    model_id: &str,
    correlation_id: &str,
    request: String,
) -> Result<String, String> {
    let variant = parse_chat_model_variant(Some(model_id)).map_err(|err| err.to_string())?;
    let permit = state
        .acquire_workload_permit(izwi_core::WorkloadClass::Background)
        .await;
    let response = state
        .runtime
        .auxiliary_generate(
            variant,
            vec![
                ChatMessage {
                    role: izwi_core::ChatRole::System,
                    content: SUMMARY_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: izwi_core::ChatRole::User,
                    content: request,
                },
            ],
            GenerationParams {
                max_tokens: 256,
                ..Default::default()
            },
            Some(correlation_id),
            permit.runtime_context(),
        )
        .await
        .map_err(|err| err.to_string())?;
    Ok(izwi_agent::strip_reasoning(&response.text))
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

    async fn load_summary(
        &self,
        _thread_id: &str,
        _kept_messages: usize,
    ) -> izwi_agent::Result<Option<String>> {
        Ok(self.summary.as_ref().and_then(ConversationSummary::text))
    }
}

/// Run a turn to its end while handing its events to `on_event` as they
/// happen.
pub(crate) async fn drive_turn<T>(
    run: impl std::future::Future<Output = T>,
    mut heard: tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
    mut on_event: impl FnMut(AgentEvent),
) -> T {
    tokio::pin!(run);
    let result = loop {
        tokio::select! {
            biased;
            Some(event) = heard.recv() => on_event(event),
            result = &mut run => break result,
        }
    };
    while let Ok(event) = heard.try_recv() {
        on_event(event);
    }
    result
}

/// How a turn ended, as a trace's outcome.
pub(crate) fn finish_name(finish: izwi_agent::TurnFinish) -> &'static str {
    match finish {
        izwi_agent::TurnFinish::Completed => "completed",
        izwi_agent::TurnFinish::MaxTokens => "max_tokens",
        izwi_agent::TurnFinish::PendingCalls => "pending_calls",
        izwi_agent::TurnFinish::Cancelled => "cancelled",
    }
}

/// A turn's trace: an `agent_turn` span, and under it a span for each
/// model round and each tool call, built from the turn's events. The
/// spans are exported when `IZWI_OTLP_TRACES_ENDPOINT` is set and are
/// otherwise ordinary tracing spans.
pub(crate) struct TurnTrace {
    trace_id: String,
    span_id: String,
    turn: tracing::Span,
    round: Option<tracing::Span>,
    calls: Vec<(String, tracing::Span)>,
}

/// 32 random hex digits; the first 16 are a span id.
fn trace_hex() -> String {
    crate::ids::new_uuid().replace('-', "")
}

impl TurnTrace {
    /// Start the trace of a turn of `kind` (`voice_modular`,
    /// `voice_unified` or `realtime`).
    pub(crate) fn start(kind: &'static str, correlation_id: &str) -> Self {
        let trace_id = trace_hex();
        let span_id = trace_hex()[..16].to_string();
        let turn = tracing::info_span!(
            parent: None,
            "agent_turn",
            trace_id = %trace_id,
            span_id = %span_id,
            correlation_id = %correlation_id,
            kind,
            agent_id = tracing::field::Empty,
            outcome = tracing::field::Empty,
        );
        Self {
            trace_id,
            span_id,
            turn,
            round: None,
            calls: Vec::new(),
        }
    }

    /// Follow one of the turn's events.
    pub(crate) fn on_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::ModelRoundStarted { round } => {
                self.round = Some(tracing::info_span!(
                    parent: None,
                    "agent_model_round",
                    trace_id = %self.trace_id,
                    span_id = %&trace_hex()[..16],
                    parent_span_id = %self.span_id,
                    round = *round as u64,
                    tokens_generated = tracing::field::Empty,
                    tool_calls = tracing::field::Empty,
                ));
            }
            AgentEvent::ModelRoundCompleted {
                tokens_generated,
                tool_calls,
                ..
            } => {
                if let Some(round) = self.round.take() {
                    round.record("tokens_generated", *tokens_generated as u64);
                    round.record("tool_calls", *tool_calls as u64);
                }
            }
            AgentEvent::ToolCallStarted {
                call_id,
                name,
                executor,
                ..
            } => {
                let span = tracing::info_span!(
                    parent: None,
                    "agent_tool_call",
                    trace_id = %self.trace_id,
                    span_id = %&trace_hex()[..16],
                    parent_span_id = %self.span_id,
                    tool = %name,
                    call_id = %call_id,
                    executor = %executor,
                    outcome = tracing::field::Empty,
                );
                self.calls.push((call_id.clone(), span));
            }
            AgentEvent::ToolCallCompleted {
                call_id, status, ..
            } => {
                if let Some(index) = self.calls.iter().position(|(id, _)| id == call_id) {
                    let (_, span) = self.calls.swap_remove(index);
                    span.record("outcome", status.as_str());
                }
            }
            AgentEvent::Handoff { to, .. } => {
                self.turn.record("agent_id", to.as_str());
            }
            _ => {}
        }
    }

    /// End the trace with the turn's `outcome` (`completed`, `cancelled`,
    /// `error`, ...).
    pub(crate) fn finish(self, outcome: &str) {
        self.turn.record("outcome", outcome);
    }
}

/// An agent turn's cancellation that follows a voice turn's control until
/// the guard drops.
pub(crate) fn cancel_with(control: &TurnControl) -> (TurnCancel, CancelLink) {
    let cancel = TurnCancel::new();
    let link = cancel_follows(control, cancel.clone());
    (cancel, link)
}

/// Make `cancel` follow a voice turn's control until the guard drops.
pub(crate) fn cancel_follows(control: &TurnControl, cancel: TurnCancel) -> CancelLink {
    let control = control.clone();
    CancelLink(tokio::spawn(async move {
        control.cancelled().await;
        cancel.cancel();
    }))
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
    fn a_summary_is_rewritten_once_enough_more_messages_have_dropped() {
        let summary = ConversationSummary::default();
        assert_eq!(summary.claim(SUMMARY_STEP - 1), None);
        assert_eq!(summary.claim(SUMMARY_STEP), Some(0..SUMMARY_STEP));
        // One writer at a time.
        assert_eq!(summary.claim(40), None);
        {
            let mut state = summary.lock();
            state.writing = false;
            state.text = Some("They asked about trains.".to_string());
            state.covers = SUMMARY_STEP;
        }
        assert_eq!(summary.claim(SUMMARY_STEP + 2), None);
        assert_eq!(
            summary.claim(2 * SUMMARY_STEP),
            Some(SUMMARY_STEP..2 * SUMMARY_STEP)
        );
        summary.reset();
        assert_eq!(summary.text(), None);

        let request = summary_request(
            Some("They asked about trains."),
            &[
                MemoryMessage::user(" And buses? "),
                MemoryMessage::tool("c", "ignored"),
                MemoryMessage::assistant("Every hour."),
            ],
        );
        assert_eq!(
            request,
            "Notes so far:\nThey asked about trains.\n\nWhat was said next:\nUser: And buses?\nAssistant: Every hour.\n"
        );
    }

    #[test]
    fn a_turns_trace_follows_its_rounds_and_calls() {
        let mut trace = TurnTrace::start("voice_modular", "corr-1");
        assert_eq!(trace.trace_id.len(), 32);
        assert_eq!(trace.span_id.len(), 16);
        trace.on_event(&AgentEvent::ModelRoundStarted { round: 0 });
        assert!(trace.round.is_some());
        trace.on_event(&AgentEvent::ModelRoundCompleted {
            round: 0,
            tokens_generated: 12,
            tool_calls: 1,
        });
        assert!(trace.round.is_none());
        trace.on_event(&AgentEvent::ToolCallStarted {
            call_id: "c1".into(),
            name: "time".into(),
            arguments: "{}".into(),
            executor: "server".into(),
        });
        assert_eq!(trace.calls.len(), 1);
        trace.on_event(&AgentEvent::ToolCallCompleted {
            call_id: "c1".into(),
            name: "time".into(),
            status: "ok".into(),
            output: String::new(),
        });
        assert!(trace.calls.is_empty());
        trace.finish("completed");
    }

    #[test]
    fn a_family_without_a_tool_template_is_offered_no_tools() {
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "time"}})];
        assert!(tool_plan_for(ModelVariant::Qwen354BGguf, &tools, ToolChoice::Auto).is_some());
        assert!(tool_plan_for(ModelVariant::Qwen354BGguf, &[], ToolChoice::Auto).is_none());
        assert!(tool_plan_for(ModelVariant::Gemma34BIt, &tools, ToolChoice::Auto).is_none());
    }
}

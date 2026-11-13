use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent::{AgentDefinition, AgentSet};
use crate::cancel::TurnCancel;
use crate::errors::{AgentError, Result};
use crate::memory::{MemoryMessage, MemoryMessageMeta, MemoryMessageRole, MemoryStore};
use crate::model::{ModelBackend, ModelRequest};
use crate::reasoning::strip_reasoning;
use crate::session::{
    AgentEvent, AgentSession, AgentTurnResult, CompletedTask, ToolCallRecord, TurnEvents,
    TurnFinish, TurnInput,
};
use crate::tools::{
    function_definition, BackgroundResult, BackgroundResults, Tool, ToolCall, ToolContext,
    ToolExecution, ToolRegistry,
};

#[derive(Debug, Clone, Copy)]
pub struct AgentTurnOptions {
    pub max_output_tokens: usize,
    /// The most model rounds one turn runs: a round that calls tools is
    /// followed by one that reads their results.
    pub max_tool_rounds: usize,
    /// The most calls one turn makes.
    pub max_tool_calls: usize,
    /// How long the turn waits for one tool.
    pub tool_timeout: Duration,
    /// After this long a running call is reported as slow
    /// ([`AgentEvent::ToolCallSlow`]), once.
    pub tool_soft_timeout: Option<Duration>,
    /// How long a background tool may run.
    pub background_tool_timeout: Duration,
    /// Whether the calls of one round run at the same time.
    pub parallel_tool_calls: bool,
    /// How much of the thread's history the prompt carries.
    pub history: HistoryBudget,
    /// Whether the turn stores the assistant reply itself. A caller that
    /// learns later how much of the reply reached the user (voice: an
    /// interrupted reply was only partly heard) stores it with
    /// [`AgentEngine::commit_assistant_reply`] instead.
    pub persist_assistant_reply: bool,
}

impl Default for AgentTurnOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 1536,
            max_tool_rounds: 3,
            max_tool_calls: 8,
            tool_timeout: Duration::from_secs(10),
            tool_soft_timeout: None,
            background_tool_timeout: Duration::from_secs(120),
            parallel_tool_calls: false,
            history: HistoryBudget::default(),
            persist_assistant_reply: true,
        }
    }
}

/// The most conversation history a prompt carries: the newest messages that
/// fit both limits, starting at a user message. System messages at the head
/// of the history (memory context) are always kept. The token count is an
/// estimate (four characters a token); the model's own context check stays
/// authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryBudget {
    pub max_messages: usize,
    pub max_tokens: usize,
}

impl Default for HistoryBudget {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl HistoryBudget {
    pub const UNBOUNDED: Self = Self {
        max_messages: usize::MAX,
        max_tokens: usize::MAX,
    };

    /// The part of `history` this budget keeps.
    pub fn apply(self, mut history: Vec<MemoryMessage>) -> Vec<MemoryMessage> {
        let pinned = history
            .iter()
            .take_while(|message| message.role == MemoryMessageRole::System)
            .count();
        let conversation = history.split_off(pinned);
        let mut tokens = 0usize;
        let mut start = conversation.len();
        for (index, message) in conversation.iter().enumerate().rev() {
            tokens = tokens.saturating_add(estimate_tokens(&message.content));
            if conversation.len() - index > self.max_messages || tokens > self.max_tokens {
                break;
            }
            start = index;
        }
        // A window never opens on the reply to a user message it dropped.
        while conversation
            .get(start)
            .is_some_and(|message| message.role != MemoryMessageRole::User)
        {
            start += 1;
        }
        history.extend(conversation.into_iter().skip(start));
        history
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
}

/// What a turn runs with.
pub struct TurnContext<'a, M, B> {
    pub agents: &'a AgentSet,
    pub memory: &'a M,
    pub model: &'a B,
    pub tools: &'a ToolRegistry,
    /// Hears the turn as it runs: text, calls, handoffs.
    pub events: TurnEvents,
    pub cancel: TurnCancel,
    /// The session's background tool results.
    pub background: BackgroundResults,
}

#[derive(Debug, Clone, Default)]
pub struct AgentEngine;

impl AgentEngine {
    /// Run one turn on the session's stored thread: the user's `input`,
    /// then model rounds until the reply calls no more tools.
    pub async fn run_turn<M, B>(
        &self,
        ctx: &TurnContext<'_, M, B>,
        session: &AgentSession,
        input: TurnInput,
        model_id: Option<String>,
        options: AgentTurnOptions,
    ) -> Result<AgentTurnResult>
    where
        M: MemoryStore,
        B: ModelBackend,
    {
        let user_text = input.text.trim().to_string();
        if user_text.is_empty() {
            return Err(AgentError::InvalidInput(
                "Turn input text cannot be empty".to_string(),
            ));
        }

        let stored = ctx.memory.load_messages(&session.thread_id).await?;
        let stored_len = stored.len();
        let history = options.history.apply(stored);
        ctx.memory
            .append_message(
                &session.thread_id,
                MemoryMessageRole::User,
                user_text.clone(),
                MemoryMessageMeta::default(),
            )
            .await?;

        // One leading system message: chat templates such as Qwen3.5's reject
        // a system message anywhere else, so the memory context joins the
        // agent's prompt instead of following it.
        let mut context = Vec::new();
        let mut conversation = Vec::with_capacity(history.len() + 1);
        for message in history {
            if message.role == MemoryMessageRole::System {
                context.push(message.content);
            } else {
                conversation.push(message);
            }
        }
        let kept = conversation.len();
        if kept + context.len() < stored_len {
            if let Some(summary) = ctx.memory.load_summary(&session.thread_id, kept).await? {
                context.push(format!("Earlier in this conversation: {summary}"));
            }
        }
        conversation.push(MemoryMessage::user(user_text.clone()));

        let result = self
            .run_conversation(
                ctx,
                session,
                context,
                conversation,
                &user_text,
                model_id,
                options,
            )
            .await?;

        if options.persist_assistant_reply {
            self.commit_assistant_reply(
                ctx.memory,
                session,
                &result.assistant_text,
                MemoryMessageMeta {
                    model_id: Some(result.model_id.clone()),
                    tokens_generated: Some(result.tokens_generated),
                    generation_time_ms: Some(result.generation_time_ms),
                },
            )
            .await?;
        }
        Ok(result)
    }

    /// Run a turn's model rounds over a `conversation` the caller keeps:
    /// `context` joins the agent's system prompt, `conversation` ends with
    /// what the reply answers. Nothing is stored.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_conversation<M, B>(
        &self,
        ctx: &TurnContext<'_, M, B>,
        session: &AgentSession,
        mut context: Vec<String>,
        mut conversation: Vec<MemoryMessage>,
        user_text: &str,
        model_id: Option<String>,
        options: AgentTurnOptions,
    ) -> Result<AgentTurnResult>
    where
        M: MemoryStore,
        B: ModelBackend,
    {
        let mut agent = ctx
            .agents
            .get(&session.agent_id)
            .unwrap_or_else(|| ctx.agents.first());
        let model_id = model_id.unwrap_or_else(|| agent.default_model.clone());
        let mut log = EventLog::new(ctx.events.clone());
        log.emit(AgentEvent::TurnStarted {
            session_id: session.id.clone(),
            thread_id: session.thread_id.clone(),
        });

        let earlier = ctx.background.take();
        if !earlier.is_empty() {
            let lines = earlier
                .iter()
                .map(|result| format!("- {} ({}): {}", result.name, result.status, result.output))
                .collect::<Vec<_>>()
                .join("\n");
            context.push(format!(
                "Tools started earlier have finished. Tell the user what they found if it still matters:\n{lines}"
            ));
        }

        let max_rounds = options.max_tool_rounds.max(1);
        let mut turn = TurnState::default();
        let mut finish = TurnFinish::Completed;
        'rounds: for round in 0..max_rounds {
            if ctx.cancel.is_cancelled() {
                finish = TurnFinish::Cancelled;
                break;
            }
            // A listener hears the rounds as one reply.
            if !turn.text.is_empty() && !turn.text.ends_with(char::is_whitespace) {
                ctx.events.text_delta(" ");
            }
            let request = ModelRequest {
                model_id: model_id.clone(),
                messages: prompt(agent, &context, &conversation),
                tools: offered_tools(ctx.agents, agent, ctx.tools),
                max_output_tokens: options.max_output_tokens.max(1),
                round,
            };
            ctx.events.emit(AgentEvent::ModelRoundStarted { round });
            let output = tokio::select! {
                biased;
                () = ctx.cancel.cancelled() => {
                    finish = TurnFinish::Cancelled;
                    break;
                }
                output = ctx.model.generate(request, &ctx.events) => output?,
            };
            ctx.events.emit(AgentEvent::ModelRoundCompleted {
                round,
                tokens_generated: output.tokens_generated,
                tool_calls: output.tool_calls.len(),
            });
            turn.tokens_generated += output.tokens_generated;
            turn.generation_time_ms += output.generation_time_ms;
            join_text(&mut turn.text, &output.text);

            if output.tool_calls.is_empty() {
                if output.truncated {
                    finish = TurnFinish::MaxTokens;
                }
                break;
            }
            if round + 1 == max_rounds {
                return Err(AgentError::ToolLoop(format!(
                    "The reply called tools in each of its {max_rounds} rounds without answering"
                )));
            }
            turn.calls_made += output.tool_calls.len();
            if turn.calls_made > options.max_tool_calls {
                return Err(AgentError::ToolLoop(format!(
                    "The reply made more than {} tool calls in one turn",
                    options.max_tool_calls
                )));
            }
            conversation.push(MemoryMessage::assistant_calls(
                output.text.clone(),
                output.tool_calls.clone(),
            ));

            let routes = output
                .tool_calls
                .iter()
                .map(|call| route(ctx.agents, agent, ctx.tools, &call.name))
                .collect::<Vec<_>>();
            let together = options.parallel_tool_calls
                && routes.len() > 1
                && routes
                    .iter()
                    .all(|route| matches!(route, Route::Tool(tool) if tool.execution() == ToolExecution::Inline));
            if together {
                let runs = output.tool_calls.iter().zip(&routes).map(|(call, route)| {
                    let Route::Tool(tool) = route else {
                        unreachable!("only inline tools run together");
                    };
                    log.emit(started(call, tool.executor()));
                    run_inline(ctx, session, tool.clone(), call, user_text, options)
                });
                let outcomes = futures::future::join_all(runs).await;
                for (call, outcome) in output.tool_calls.iter().zip(outcomes) {
                    turn.settle(&mut log, &mut conversation, call, outcome);
                }
                if ctx.cancel.is_cancelled() {
                    finish = TurnFinish::Cancelled;
                    break 'rounds;
                }
                continue;
            }

            for (call, route) in output.tool_calls.iter().zip(routes) {
                let outcome = match route {
                    Route::External => {
                        log.emit(started(call, "client"));
                        turn.pending.push(call.clone());
                        continue;
                    }
                    Route::Tool(tool) => {
                        log.emit(started(call, tool.executor()));
                        match tool.execution() {
                            ToolExecution::Inline => {
                                run_inline(ctx, session, tool, call, user_text, options).await
                            }
                            ToolExecution::Background => {
                                start_background(ctx, session, tool, call, user_text, options)
                            }
                        }
                    }
                    Route::Handoff(target) => {
                        log.emit(started(call, "server"));
                        log.emit(AgentEvent::Handoff {
                            from: agent.id.clone(),
                            to: target.id.clone(),
                        });
                        agent = target;
                        Outcome::ok(handoff_note(target))
                    }
                    Route::Task => {
                        log.emit(started(call, "server"));
                        let task = agent.task.as_ref().expect("a task route has a task");
                        match serde_json::from_str::<Value>(&call.arguments) {
                            Err(err) => Outcome::error(format!("Invalid arguments: {err}")),
                            Ok(result) => {
                                let missing = task.missing_fields(&result);
                                if missing.is_empty() {
                                    log.emit(AgentEvent::TaskCompleted {
                                        name: task.name.clone(),
                                        result: result.clone(),
                                    });
                                    turn.completed_tasks.push(CompletedTask {
                                        name: task.name.clone(),
                                        result,
                                    });
                                    let next =
                                        task.then.as_deref().and_then(|id| ctx.agents.get(id));
                                    match next {
                                        Some(target) => {
                                            log.emit(AgentEvent::Handoff {
                                                from: agent.id.clone(),
                                                to: target.id.clone(),
                                            });
                                            agent = target;
                                            Outcome::ok(format!(
                                                "Recorded. {}",
                                                handoff_note(target)
                                            ))
                                        }
                                        None => Outcome::ok("Recorded."),
                                    }
                                } else {
                                    Outcome::error(format!(
                                        "Not recorded: ask the user for {}.",
                                        missing.join(", ")
                                    ))
                                }
                            }
                        }
                    }
                    Route::Unknown => {
                        log.emit(started(call, "none"));
                        Outcome::error(format!("No tool named `{}` is available.", call.name))
                    }
                };
                turn.settle(&mut log, &mut conversation, call, outcome);
                if ctx.cancel.is_cancelled() {
                    finish = TurnFinish::Cancelled;
                    break 'rounds;
                }
            }
            if !turn.pending.is_empty() {
                finish = TurnFinish::PendingCalls;
                break;
            }
        }

        if !turn.text.is_empty() {
            log.emit(AgentEvent::AssistantMessage {
                content: turn.text.clone(),
            });
        }
        log.emit(AgentEvent::TurnCompleted {
            session_id: session.id.clone(),
            thread_id: session.thread_id.clone(),
            finish,
        });
        tracing::debug!(
            session_id = %session.id,
            agent_id = %agent.id,
            ?finish,
            tool_calls = turn.records.len(),
            tokens = turn.tokens_generated,
            "agent turn finished"
        );

        Ok(AgentTurnResult {
            assistant_text: turn.text,
            model_id,
            finish,
            agent_id: agent.id.clone(),
            tool_calls: turn.records,
            pending_calls: turn.pending,
            completed_tasks: turn.completed_tasks,
            tokens_generated: turn.tokens_generated,
            generation_time_ms: turn.generation_time_ms,
            events: log.events,
        })
    }

    /// Store an assistant reply in the session's thread, without its
    /// reasoning. An empty reply is not stored.
    pub async fn commit_assistant_reply<M: MemoryStore>(
        &self,
        memory: &M,
        session: &AgentSession,
        reply: &str,
        meta: MemoryMessageMeta,
    ) -> Result<()> {
        let reply = strip_reasoning(reply);
        if reply.is_empty() {
            return Ok(());
        }
        memory
            .append_message(
                &session.thread_id,
                MemoryMessageRole::Assistant,
                reply,
                meta,
            )
            .await
    }
}

/// The events of a turn: each goes to the turn's listener and into its
/// result.
struct EventLog {
    listener: TurnEvents,
    events: Vec<AgentEvent>,
}

impl EventLog {
    fn new(listener: TurnEvents) -> Self {
        Self {
            listener,
            events: Vec::new(),
        }
    }

    fn emit(&mut self, event: AgentEvent) {
        self.listener.emit(event.clone());
        self.events.push(event);
    }
}

#[derive(Default)]
struct TurnState {
    text: String,
    calls_made: usize,
    records: Vec<ToolCallRecord>,
    pending: Vec<ToolCall>,
    completed_tasks: Vec<CompletedTask>,
    tokens_generated: usize,
    generation_time_ms: f64,
}

impl TurnState {
    /// Record a call's outcome and hand it to the model's next round.
    fn settle(
        &mut self,
        log: &mut EventLog,
        conversation: &mut Vec<MemoryMessage>,
        call: &ToolCall,
        outcome: Outcome,
    ) {
        log.emit(AgentEvent::ToolCallCompleted {
            call_id: call.id.clone(),
            name: call.name.clone(),
            status: outcome.status.to_string(),
            output: outcome.output.clone(),
        });
        conversation.push(MemoryMessage::tool(call.id.clone(), outcome.output.clone()));
        self.records.push(ToolCallRecord {
            call_id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            status: outcome.status.to_string(),
            output: outcome.output,
        });
    }
}

struct Outcome {
    status: &'static str,
    output: String,
}

impl Outcome {
    fn ok(output: impl Into<String>) -> Self {
        Self {
            status: "ok",
            output: output.into(),
        }
    }

    fn error(output: impl Into<String>) -> Self {
        Self {
            status: "error",
            output: output.into(),
        }
    }

    fn timeout(timeout: Duration) -> Self {
        Self {
            status: "timeout",
            output: format!("The tool did not answer within {} ms.", timeout.as_millis()),
        }
    }
}

/// Who answers a call.
enum Route<'a> {
    Tool(Arc<dyn Tool>),
    /// A declared tool the caller runs.
    External,
    Handoff(&'a AgentDefinition),
    Task,
    Unknown,
}

fn route<'a>(
    agents: &'a AgentSet,
    agent: &AgentDefinition,
    tools: &ToolRegistry,
    name: &str,
) -> Route<'a> {
    if let Some(target) = agent
        .handoffs
        .iter()
        .find(|id| AgentDefinition::transfer_tool_name(id) == name)
        .and_then(|id| agents.get(id))
    {
        return Route::Handoff(target);
    }
    if agent
        .task
        .as_ref()
        .is_some_and(|task| task.tool_name() == name)
    {
        return Route::Task;
    }
    if agent
        .tools
        .as_ref()
        .is_some_and(|allowed| !allowed.iter().any(|allowed| allowed == name))
    {
        return Route::Unknown;
    }
    match tools.find(name) {
        Some(tool) => Route::Tool(tool),
        None if tools.is_external(name) => Route::External,
        None => Route::Unknown,
    }
}

/// The tools a round offers: the agent's registry tools, a transfer tool
/// for each handoff and the tool that records its task.
fn offered_tools(agents: &AgentSet, agent: &AgentDefinition, tools: &ToolRegistry) -> Vec<Value> {
    let mut offered = tools.function_definitions(agent.tools.as_deref());
    for target in agent.handoffs.iter().filter_map(|id| agents.get(id)) {
        let description = match &target.handoff_description {
            Some(when) => format!("Hand the conversation to {}. {when}", target.name),
            None => format!("Hand the conversation to {}.", target.name),
        };
        offered.push(function_definition(
            &AgentDefinition::transfer_tool_name(&target.id),
            &description,
            json!({"type": "object", "properties": {}}),
        ));
    }
    if let Some(task) = &agent.task {
        offered.push(function_definition(
            &task.tool_name(),
            &task.description,
            task.parameters.clone(),
        ));
    }
    offered
}

fn handoff_note(target: &AgentDefinition) -> String {
    format!(
        "Done: you are now {}. Answer the user directly in that role, following your new \
         instructions. Do not announce the transfer.",
        target.name
    )
}

fn prompt(
    agent: &AgentDefinition,
    context: &[String],
    conversation: &[MemoryMessage],
) -> Vec<MemoryMessage> {
    let system = std::iter::once(agent.system_prompt.as_str())
        .chain(context.iter().map(String::as_str))
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut messages = Vec::with_capacity(conversation.len() + 1);
    if !system.is_empty() {
        messages.push(MemoryMessage::system(system));
    }
    messages.extend(conversation.iter().cloned());
    messages
}

/// Append a round's text to the reply so far.
fn join_text(reply: &mut String, round: &str) {
    if round.is_empty() {
        return;
    }
    if !reply.is_empty()
        && !reply.ends_with(char::is_whitespace)
        && !round.starts_with(char::is_whitespace)
    {
        reply.push(' ');
    }
    reply.push_str(round);
}

fn started(call: &ToolCall, executor: &str) -> AgentEvent {
    AgentEvent::ToolCallStarted {
        call_id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        executor: executor.to_string(),
    }
}

fn tool_context(
    session: &AgentSession,
    call: &ToolCall,
    user_text: &str,
    cancel: TurnCancel,
) -> std::result::Result<ToolContext, Outcome> {
    let arguments = match call.arguments.trim() {
        "" => json!({}),
        arguments => serde_json::from_str::<Value>(arguments)
            .map_err(|err| Outcome::error(format!("Invalid arguments: {err}")))?,
    };
    Ok(ToolContext {
        session_id: session.id.clone(),
        thread_id: session.thread_id.clone(),
        call_id: call.id.clone(),
        user_text: user_text.to_string(),
        arguments,
        cancel,
    })
}

/// Run a tool the turn waits for: until it answers, its timeout passes or
/// the turn is cancelled.
async fn run_inline<M, B>(
    ctx: &TurnContext<'_, M, B>,
    session: &AgentSession,
    tool: Arc<dyn Tool>,
    call: &ToolCall,
    user_text: &str,
    options: AgentTurnOptions,
) -> Outcome {
    let context = match tool_context(session, call, user_text, ctx.cancel.clone()) {
        Ok(context) => context,
        Err(outcome) => return outcome,
    };
    let invoke = tool.invoke(context);
    let deadline = tokio::time::sleep(options.tool_timeout);
    let slow = async {
        match options.tool_soft_timeout {
            Some(after) => tokio::time::sleep(after).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(invoke, deadline, slow);
    let mut reported_slow = false;
    loop {
        tokio::select! {
            result = &mut invoke => {
                return match result {
                    Ok(output) => Outcome::ok(output.text),
                    Err(err) => Outcome::error(err.to_string()),
                };
            }
            () = &mut deadline => return Outcome::timeout(options.tool_timeout),
            () = ctx.cancel.cancelled() => {
                return Outcome {
                    status: "cancelled",
                    output: "The turn was interrupted.".to_string(),
                };
            }
            () = &mut slow, if !reported_slow => {
                reported_slow = true;
                ctx.events.emit(AgentEvent::ToolCallSlow {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                });
            }
        }
    }
}

/// Start a tool that outlives the turn. Its result waits in the session's
/// background results for the next turn.
fn start_background<M, B>(
    ctx: &TurnContext<'_, M, B>,
    session: &AgentSession,
    tool: Arc<dyn Tool>,
    call: &ToolCall,
    user_text: &str,
    options: AgentTurnOptions,
) -> Outcome {
    // The turn's cancellation does not reach a background tool.
    let context = match tool_context(session, call, user_text, TurnCancel::new()) {
        Ok(context) => context,
        Err(outcome) => return outcome,
    };
    let (results, events) = (ctx.background.clone(), ctx.events.clone());
    let (call_id, name) = (call.id.clone(), call.name.clone());
    let timeout = options.background_tool_timeout;
    tokio::spawn(async move {
        let outcome = match tokio::time::timeout(timeout, tool.invoke(context)).await {
            Ok(Ok(output)) => Outcome::ok(output.text),
            Ok(Err(err)) => Outcome::error(err.to_string()),
            Err(_) => Outcome::timeout(timeout),
        };
        results.push(BackgroundResult {
            call_id: call_id.clone(),
            name: name.clone(),
            status: outcome.status.to_string(),
            output: outcome.output,
        });
        events.emit(AgentEvent::BackgroundToolCompleted {
            call_id,
            name,
            status: outcome.status.to_string(),
        });
    });
    Outcome {
        status: "started",
        output: "Started. The result will arrive later; tell the user it is under way.".to_string(),
    }
}

#[cfg(test)]
mod tests;

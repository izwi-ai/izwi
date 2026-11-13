use super::*;
use crate::agent::TaskSpec;
use crate::model::ModelOutput;
use crate::tools::ToolOutput;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;
use tokio::sync::mpsc;

#[derive(Default)]
struct TestMemory {
    messages: Mutex<Vec<MemoryMessage>>,
    summary: Option<String>,
}

#[async_trait]
impl MemoryStore for TestMemory {
    async fn load_messages(&self, _thread_id: &str) -> Result<Vec<MemoryMessage>> {
        Ok(self.messages.lock().unwrap().clone())
    }

    async fn append_message(
        &self,
        _thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        _meta: MemoryMessageMeta,
    ) -> Result<()> {
        self.messages
            .lock()
            .unwrap()
            .push(MemoryMessage::new(role, content));
        Ok(())
    }

    async fn load_summary(&self, _thread_id: &str, _kept: usize) -> Result<Option<String>> {
        Ok(self.summary.clone())
    }
}

/// A model that plays a script: one output a round. Past the script it
/// answers "done". It records every request.
#[derive(Default)]
struct ScriptedModel {
    script: Mutex<VecDeque<ModelOutput>>,
    requests: Mutex<Vec<ModelRequest>>,
    /// Never answers (a turn cancelled mid-reply).
    hangs: bool,
}

impl ScriptedModel {
    fn new(script: Vec<ModelOutput>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            ..Self::default()
        }
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ModelBackend for ScriptedModel {
    async fn generate(&self, request: ModelRequest, events: &TurnEvents) -> Result<ModelOutput> {
        self.requests.lock().unwrap().push(request);
        if self.hangs {
            events.text_delta("Well");
            std::future::pending::<()>().await;
        }
        let output = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| says("done"));
        events.text_delta(&output.text);
        Ok(output)
    }
}

fn says(text: &str) -> ModelOutput {
    ModelOutput {
        text: text.to_string(),
        tokens_generated: 3,
        generation_time_ms: 1.0,
        ..ModelOutput::default()
    }
}

fn calls(text: &str, calls: &[(&str, &str)]) -> ModelOutput {
    ModelOutput {
        tool_calls: calls
            .iter()
            .enumerate()
            .map(|(index, (name, arguments))| ToolCall {
                id: format!("call_{name}_{index}"),
                name: name.to_string(),
                arguments: arguments.to_string(),
            })
            .collect(),
        ..says(text)
    }
}

/// Answers with its arguments after `delay`.
struct EchoTool {
    name: &'static str,
    delay: Duration,
    execution: ToolExecution,
}

impl EchoTool {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            delay: Duration::ZERO,
            execution: ToolExecution::Inline,
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Echoes its arguments."
    }

    fn execution(&self) -> ToolExecution {
        self.execution
    }

    async fn invoke(&self, ctx: ToolContext) -> Result<ToolOutput> {
        tokio::time::sleep(self.delay).await;
        if ctx.arguments.get("fail").is_some() {
            return Err(AgentError::Tool("it broke".to_string()));
        }
        Ok(ToolOutput {
            text: format!("{} got {}", self.name, ctx.arguments),
        })
    }
}

struct Harness {
    agents: AgentSet,
    memory: TestMemory,
    model: ScriptedModel,
    tools: ToolRegistry,
    cancel: TurnCancel,
    background: BackgroundResults,
    events: mpsc::UnboundedReceiver<AgentEvent>,
    listener: TurnEvents,
    session: AgentSession,
}

impl Harness {
    fn new(script: Vec<ModelOutput>) -> Self {
        let (tx, events) = mpsc::unbounded_channel();
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool::new("lookup"));
        Self {
            agents: AgentSet::single(AgentDefinition::new("a", "Ada", "Be brief.", "model")),
            memory: TestMemory::default(),
            model: ScriptedModel::new(script),
            tools,
            cancel: TurnCancel::new(),
            background: BackgroundResults::default(),
            events,
            listener: TurnEvents::new(tx),
            session: AgentSession {
                id: "s".to_string(),
                agent_id: "a".to_string(),
                thread_id: "t".to_string(),
                created_at: 0,
                updated_at: 0,
            },
        }
    }

    async fn turn(&self, text: &str, options: AgentTurnOptions) -> Result<AgentTurnResult> {
        let ctx = TurnContext {
            agents: &self.agents,
            memory: &self.memory,
            model: &self.model,
            tools: &self.tools,
            events: self.listener.clone(),
            cancel: self.cancel.clone(),
            background: self.background.clone(),
        };
        AgentEngine
            .run_turn(
                &ctx,
                &self.session,
                TurnInput {
                    text: text.to_string(),
                },
                None,
                options,
            )
            .await
    }

    fn heard(&mut self) -> Vec<AgentEvent> {
        let mut heard = Vec::new();
        while let Ok(event) = self.events.try_recv() {
            heard.push(event);
        }
        heard
    }

    fn stored(&self) -> Vec<(MemoryMessageRole, String)> {
        self.memory
            .messages
            .lock()
            .unwrap()
            .iter()
            .map(|message| (message.role, message.content.clone()))
            .collect()
    }
}

fn message(role: MemoryMessageRole, content: &str) -> MemoryMessage {
    MemoryMessage::new(role, content)
}

#[test]
fn a_budget_keeps_the_newest_turns_and_the_memory_context() {
    use MemoryMessageRole::{Assistant, System, User};
    let history = vec![
        message(System, "context"),
        message(User, "one"),
        message(Assistant, "two"),
        message(User, "three"),
        message(Assistant, "four"),
    ];
    let kept = HistoryBudget {
        max_messages: 3,
        max_tokens: usize::MAX,
    }
    .apply(history.clone());
    // Three messages would open on a reply; the window starts at its user.
    let contents: Vec<_> = kept.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(contents, ["context", "three", "four"]);
    assert_eq!(HistoryBudget::UNBOUNDED.apply(history.clone()), history);
    let tight = HistoryBudget {
        max_messages: usize::MAX,
        max_tokens: 1,
    }
    .apply(history);
    assert_eq!(tight.len(), 1, "only the pinned context fits");
}

#[tokio::test]
async fn a_reply_without_calls_is_one_round_and_is_stored_without_reasoning() {
    let mut harness = Harness::new(vec![says("<think>hm</think>Hello there.")]);
    let result = harness
        .turn("  Hi  ", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.finish, TurnFinish::Completed);
    assert_eq!(result.assistant_text, "<think>hm</think>Hello there.");
    assert_eq!(result.tokens_generated, 3);
    assert_eq!(
        harness.stored(),
        [
            (MemoryMessageRole::User, "Hi".to_string()),
            (MemoryMessageRole::Assistant, "Hello there.".to_string()),
        ]
    );

    let requests = harness.model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].messages,
        [
            MemoryMessage::system("Be brief."),
            MemoryMessage::user("Hi")
        ]
    );
    assert_eq!(requests[0].tools[0]["function"]["name"], "lookup");

    let heard = harness.heard();
    assert!(heard.contains(&AgentEvent::TextDelta {
        text: "<think>hm</think>Hello there.".to_string()
    }));
    assert!(matches!(
        heard.last(),
        Some(AgentEvent::TurnCompleted {
            finish: TurnFinish::Completed,
            ..
        })
    ));
    // The result carries the events but not the deltas.
    assert!(!result
        .events
        .iter()
        .any(|event| matches!(event, AgentEvent::TextDelta { .. })));
    assert!(harness
        .turn("   ", AgentTurnOptions::default())
        .await
        .is_err());
}

#[tokio::test]
async fn a_round_that_calls_tools_is_followed_by_one_that_reads_them() {
    let mut harness = Harness::new(vec![
        calls(
            "Let me check.",
            &[("lookup", r#"{"q":"rain"}"#), ("nope", "{}")],
        ),
        says("It will rain."),
    ]);
    let result = harness
        .turn("Weather?", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.assistant_text, "Let me check. It will rain.");
    assert_eq!(result.tokens_generated, 6);
    let outcomes: Vec<_> = result
        .tool_calls
        .iter()
        .map(|call| {
            (
                call.name.as_str(),
                call.status.as_str(),
                call.output.as_str(),
            )
        })
        .collect();
    assert_eq!(
        outcomes,
        [
            ("lookup", "ok", r#"lookup got {"q":"rain"}"#),
            ("nope", "error", "No tool named `nope` is available."),
        ]
    );

    // The second round reads the call and its results, in order.
    let second = &harness.model.requests()[1];
    assert_eq!(second.round, 1);
    let tail = &second.messages[second.messages.len() - 3..];
    assert_eq!(tail[0].role, MemoryMessageRole::Assistant);
    assert_eq!(tail[0].tool_calls.len(), 2);
    assert_eq!(tail[1].tool_call_id.as_deref(), Some("call_lookup_0"));
    assert_eq!(tail[2].content, "No tool named `nope` is available.");
    // The thread keeps the user's words and the reply, not the calls.
    assert_eq!(harness.stored().len(), 2);

    let heard = harness.heard();
    let started = heard
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
        .count();
    assert_eq!(started, 2);
}

#[tokio::test]
async fn a_turn_stops_calling_at_its_round_and_call_limits() {
    let looping = || calls("", &[("lookup", "{}")]);
    let harness = Harness::new(vec![looping(), looping(), looping()]);
    let err = harness
        .turn("Go", AgentTurnOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "The reply called tools in each of its 3 rounds without answering"
    );
    assert_eq!(harness.model.requests().len(), 3);

    let harness = Harness::new(vec![calls(
        "",
        &[("lookup", "{}"), ("lookup", "{}"), ("lookup", "{}")],
    )]);
    let err = harness
        .turn(
            "Go",
            AgentTurnOptions {
                max_tool_calls: 2,
                ..AgentTurnOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "The reply made more than 2 tool calls in one turn"
    );
}

#[tokio::test(start_paused = true)]
async fn a_slow_tool_is_reported_once_then_times_out() {
    let mut harness = Harness::new(vec![calls("", &[("slow", "{}")]), says("Sorry.")]);
    harness.tools.register(EchoTool {
        delay: Duration::from_secs(30),
        ..EchoTool::new("slow")
    });
    let result = harness
        .turn(
            "Go",
            AgentTurnOptions {
                tool_soft_timeout: Some(Duration::from_secs(1)),
                tool_timeout: Duration::from_secs(5),
                ..AgentTurnOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.tool_calls[0].status, "timeout");
    assert_eq!(
        result.tool_calls[0].output,
        "The tool did not answer within 5000 ms."
    );
    assert_eq!(result.assistant_text, "Sorry.");
    let slow = harness
        .heard()
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolCallSlow { .. }))
        .count();
    assert_eq!(slow, 1);
}

#[tokio::test(start_paused = true)]
async fn failed_and_malformed_calls_reach_the_model_as_errors() {
    let harness = Harness::new(vec![
        calls("", &[("lookup", r#"{"fail":true}"#), ("lookup", "{")]),
        says("Hm."),
    ]);
    let result = harness
        .turn("Go", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.tool_calls[0].status, "error");
    assert_eq!(result.tool_calls[0].output, "Tool error: it broke");
    assert!(result.tool_calls[1]
        .output
        .starts_with("Invalid arguments:"));
}

#[tokio::test(start_paused = true)]
async fn the_calls_of_a_round_can_run_together() {
    let mut harness = Harness::new(vec![
        calls("", &[("slow", r#"{"n":1}"#), ("slow", r#"{"n":2}"#)]),
        says("Both."),
    ]);
    harness.tools.register(EchoTool {
        delay: Duration::from_secs(4),
        ..EchoTool::new("slow")
    });
    let started = tokio::time::Instant::now();
    let result = harness
        .turn(
            "Go",
            AgentTurnOptions {
                parallel_tool_calls: true,
                tool_timeout: Duration::from_secs(5),
                ..AgentTurnOptions::default()
            },
        )
        .await
        .unwrap();
    // One after another they would take eight seconds.
    assert!(started.elapsed() < Duration::from_secs(5));
    let outputs: Vec<_> = result
        .tool_calls
        .iter()
        .map(|c| c.output.as_str())
        .collect();
    assert_eq!(outputs, [r#"slow got {"n":1}"#, r#"slow got {"n":2}"#]);
}

#[tokio::test(start_paused = true)]
async fn cancelling_stops_a_reply_and_a_waiting_tool() {
    // Mid-reply.
    let mut harness = Harness::new(Vec::new());
    harness.model.hangs = true;
    let cancel = harness.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        cancel.cancel();
    });
    let result = harness
        .turn(
            "Go",
            AgentTurnOptions {
                persist_assistant_reply: false,
                ..AgentTurnOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.finish, TurnFinish::Cancelled);
    assert!(harness.heard().contains(&AgentEvent::TextDelta {
        text: "Well".to_string()
    }));

    // Mid-tool: the turn ends, the model is not asked again.
    let mut harness = Harness::new(vec![calls("", &[("slow", "{}")]), says("never")]);
    harness.tools.register(EchoTool {
        delay: Duration::from_secs(30),
        ..EchoTool::new("slow")
    });
    let cancel = harness.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        cancel.cancel();
    });
    let result = harness
        .turn("Go", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.finish, TurnFinish::Cancelled);
    assert_eq!(result.tool_calls[0].status, "cancelled");
    assert_eq!(harness.model.requests().len(), 1);
}

#[tokio::test]
async fn a_call_to_a_declared_tool_ends_the_turn_with_the_call_pending() {
    let mut harness = Harness::new(vec![calls(
        "",
        &[("lookup", "{}"), ("client_side", r#"{"x":1}"#)],
    )]);
    harness.tools.declare_external(function_definition(
        "client_side",
        "Runs at the caller.",
        json!({"type": "object"}),
    ));
    let result = harness
        .turn("Go", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.finish, TurnFinish::PendingCalls);
    assert_eq!(result.pending_calls.len(), 1);
    assert_eq!(result.pending_calls[0].name, "client_side");
    // The turn's own tool still ran.
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "lookup");
    assert_eq!(harness.model.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_background_tool_outlives_its_turn_and_reports_in_the_next() {
    let mut harness = Harness::new(vec![
        calls("", &[("report", r#"{"id":7}"#)]),
        says("I've started the report."),
        says("The report is ready."),
    ]);
    harness.tools.register(EchoTool {
        delay: Duration::from_secs(20),
        execution: ToolExecution::Background,
        ..EchoTool::new("report")
    });
    let result = harness
        .turn("Run it", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.tool_calls[0].status, "started");
    assert_eq!(result.assistant_text, "I've started the report.");
    assert!(harness.background.is_empty());

    tokio::time::sleep(Duration::from_secs(21)).await;
    assert!(harness
        .heard()
        .contains(&AgentEvent::BackgroundToolCompleted {
            call_id: "call_report_0".to_string(),
            name: "report".to_string(),
            status: "ok".to_string(),
        }));

    harness
        .turn("Anything yet?", AgentTurnOptions::default())
        .await
        .unwrap();
    let system = &harness.model.requests()[2].messages[0];
    assert_eq!(system.role, MemoryMessageRole::System);
    assert!(
        system
            .content
            .contains(r#"- report (ok): report got {"id":7}"#),
        "{}",
        system.content
    );
    assert!(harness.background.is_empty(), "a result is told once");
}

fn two_agents() -> AgentSet {
    let mut front = AgentDefinition::new("front", "Front desk", "Greet.", "model");
    front.handoffs = vec!["billing".to_string()];
    front.tools = Some(Vec::new());
    let mut billing = AgentDefinition::new("billing", "Billing", "Handle invoices.", "model");
    billing.handoff_description = Some("Use for invoices.".to_string());
    billing.task = Some(TaskSpec {
        name: "refund".to_string(),
        description: "Record a refund request.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"invoice": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["invoice", "reason"],
        }),
        then: Some("front".to_string()),
    });
    AgentSet::new(vec![front, billing]).unwrap()
}

#[tokio::test]
async fn a_handoff_changes_the_prompt_and_tools_of_the_next_round() {
    let mut harness = Harness::new(vec![
        calls("", &[("transfer_to_billing", "{}")]),
        says("Billing here."),
    ]);
    harness.agents = two_agents();
    harness.session.agent_id = "front".to_string();
    let result = harness
        .turn("My invoice is wrong", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(result.agent_id, "billing");
    assert!(harness.heard().contains(&AgentEvent::Handoff {
        from: "front".to_string(),
        to: "billing".to_string(),
    }));

    let requests = harness.model.requests();
    let names = |request: &ModelRequest| -> Vec<String> {
        request
            .tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_string())
            .collect()
    };
    // The front desk may call no registry tool, only hand over.
    assert_eq!(names(&requests[0]), ["transfer_to_billing"]);
    assert_eq!(requests[0].messages[0].content, "Greet.");
    assert_eq!(
        requests[0].tools[0]["function"]["description"],
        "Hand the conversation to Billing. Use for invoices."
    );
    assert_eq!(names(&requests[1]), ["lookup", "complete_refund"]);
    assert_eq!(requests[1].messages[0].content, "Handle invoices.");
}

#[tokio::test]
async fn a_task_is_recorded_once_its_required_fields_are_there() {
    let mut harness = Harness::new(vec![
        calls("", &[("complete_refund", r#"{"invoice":"A1"}"#)]),
        says("What is the reason?"),
        calls(
            "",
            &[(
                "complete_refund",
                r#"{"invoice":"A1","reason":"double charge"}"#,
            )],
        ),
        says("Done. Anything else?"),
    ]);
    harness.agents = two_agents();
    harness.session.agent_id = "billing".to_string();

    let first = harness
        .turn("Refund A1", AgentTurnOptions::default())
        .await
        .unwrap();
    assert!(first.completed_tasks.is_empty());
    assert_eq!(
        first.tool_calls[0].output,
        "Not recorded: ask the user for reason."
    );
    assert_eq!(first.agent_id, "billing");

    let second = harness
        .turn("Double charge", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(second.completed_tasks.len(), 1);
    assert_eq!(second.completed_tasks[0].name, "refund");
    assert_eq!(second.completed_tasks[0].result["reason"], "double charge");
    // The task hands back to the front desk.
    assert_eq!(second.agent_id, "front");
    assert_eq!(harness.model.requests()[3].messages[0].content, "Greet.");
}

#[tokio::test]
async fn a_trimmed_history_carries_the_stores_summary() {
    let mut harness = Harness::new(Vec::new());
    harness.memory.summary = Some("They asked about trains.".to_string());
    for index in 0..6 {
        let role = if index % 2 == 0 {
            MemoryMessageRole::User
        } else {
            MemoryMessageRole::Assistant
        };
        harness
            .memory
            .messages
            .lock()
            .unwrap()
            .push(MemoryMessage::new(role, format!("m{index}")));
    }
    let options = AgentTurnOptions {
        history: HistoryBudget {
            max_messages: 2,
            max_tokens: usize::MAX,
        },
        ..AgentTurnOptions::default()
    };
    harness.turn("And buses?", options).await.unwrap();
    let request = &harness.model.requests()[0];
    assert_eq!(
        request.messages[0].content,
        "Be brief.\n\nEarlier in this conversation: They asked about trains."
    );
    let contents: Vec<_> = request.messages[1..]
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(contents, ["m4", "m5", "And buses?"]);

    // A history that fits is not summarised.
    harness
        .turn("Thanks", AgentTurnOptions::default())
        .await
        .unwrap();
    assert_eq!(harness.model.requests()[1].messages[0].content, "Be brief.");
}

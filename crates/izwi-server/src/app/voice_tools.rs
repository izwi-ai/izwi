//! Tools a voice turn may call: the server's built-in tools, the function
//! tools a client declares on the voice websocket, and the client's calls a
//! session awaits.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use izwi_agent::{
    AgentError, AgentEvent, TimeTool, Tool, ToolContext, ToolExecution, ToolOutput, ToolRegistry,
};
use serde_json::{json, Value};

use crate::app::mcp::McpTools;
use tokio::sync::oneshot;

const MAX_CLIENT_TOOLS: usize = 64;
const DEFAULT_FUNCTION_CALL_TIMEOUT_MS: u64 = 10_000;
const FUNCTION_CALL_TIMEOUT_RANGE_MS: (u64, u64) = (1_000, 60_000);

/// The server's built-in tools, shared by agent turns and voice turns.
pub(crate) fn builtin_tool_registry() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(TimeTool);
    tools
}

/// The tools a voice session's turns may call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VoiceTools {
    builtin: bool,
    /// The client's function tools, in the chat completions form.
    client: Vec<Value>,
    /// The tools of the session's MCP servers.
    mcp: McpTools,
    pub(crate) timeout: Duration,
}

impl VoiceTools {
    /// The tools `input_stream_start` offers: the built-in tools unless
    /// `builtin_tools` is `false`, and the client's `tools` (OpenAI function
    /// tools, in the chat completions or the Responses form), whose calls
    /// the client answers within `function_call_timeout_ms` (default 10 s,
    /// 1 to 60 s). `calls_tools` is whether the session's model can call
    /// tools: a chat model with a tool-calling template, or LFM3-Audio.
    pub(crate) fn parse(
        tools: Option<Vec<Value>>,
        builtin_tools: Option<bool>,
        function_call_timeout_ms: Option<u64>,
        calls_tools: bool,
    ) -> Result<Option<Self>, String> {
        let tools = tools.unwrap_or_default();
        if !calls_tools {
            if !tools.is_empty()
                || builtin_tools == Some(true)
                || function_call_timeout_ms.is_some()
            {
                return Err(
                    "Voice tools need a chat model that can call tools, or unified mode with an LFM3-Audio `s2s_model_id`"
                        .to_string(),
                );
            }
            return Ok(None);
        }
        if tools.len() > MAX_CLIENT_TOOLS {
            return Err(format!(
                "A voice session declares at most {MAX_CLIENT_TOOLS} tools, not {}",
                tools.len()
            ));
        }
        let builtin = builtin_tools.unwrap_or(true);
        let registry = builtin_tool_registry();
        let mut names = std::collections::HashSet::new();
        let client = tools
            .into_iter()
            .map(|tool| {
                let tool = chat_function_tool(tool)?;
                let name = tool["function"]["name"].as_str().unwrap_or_default();
                if builtin && registry.find(name).is_some() {
                    return Err(format!(
                        "Tool `{name}` is a built-in tool; rename it or set `builtin_tools: false`"
                    ));
                }
                if !names.insert(name.to_string()) {
                    return Err(format!("Tool `{name}` is declared twice"));
                }
                Ok(tool)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if !builtin && client.is_empty() {
            return Ok(None);
        }
        let timeout_ms = function_call_timeout_ms
            .unwrap_or(DEFAULT_FUNCTION_CALL_TIMEOUT_MS)
            .clamp(
                FUNCTION_CALL_TIMEOUT_RANGE_MS.0,
                FUNCTION_CALL_TIMEOUT_RANGE_MS.1,
            );
        Ok(Some(Self {
            builtin,
            client,
            mcp: McpTools::default(),
            timeout: Duration::from_millis(timeout_ms),
        }))
    }

    /// Add the tools of the session's MCP servers to `tools` (which may be
    /// none so far). `Err` when one has the name of another tool.
    pub(crate) fn with_mcp(tools: Option<Self>, mcp: McpTools) -> Result<Option<Self>, String> {
        if mcp.is_empty() {
            return Ok(tools);
        }
        let mut tools = tools.unwrap_or(Self {
            builtin: false,
            client: Vec::new(),
            mcp: McpTools::default(),
            timeout: Duration::from_millis(DEFAULT_FUNCTION_CALL_TIMEOUT_MS),
        });
        let mut registry = tools.registry(&PendingFunctionCalls::default());
        mcp.register(&mut registry)?;
        tools.mcp = mcp;
        Ok(Some(tools))
    }

    /// The tools as a turn's registry: the built-in ones, then the client's,
    /// each of which waits in `pending` for the client's output.
    pub(crate) fn registry(&self, pending: &PendingFunctionCalls) -> ToolRegistry {
        let mut registry = if self.builtin {
            builtin_tool_registry()
        } else {
            ToolRegistry::new()
        };
        for tool in &self.client {
            let function = &tool["function"];
            registry.register(ClientTool {
                name: function["name"].as_str().unwrap_or_default().to_string(),
                description: function["description"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                parameters: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                background: tool["background"].as_bool() == Some(true),
                pending: pending.clone(),
            });
        }
        // Name clashes were refused when the session started.
        let _ = self.mcp.register(&mut registry);
        registry
    }
}

/// A function tool the voice client runs: the turn sends the call and waits
/// for the client's `function_call_output`.
struct ClientTool {
    name: String,
    description: String,
    parameters: Value,
    /// The reply does not wait for the client's output: it reaches the
    /// next turn.
    background: bool,
    pending: PendingFunctionCalls,
}

#[async_trait::async_trait]
impl Tool for ClientTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn executor(&self) -> &'static str {
        "client"
    }

    fn execution(&self) -> ToolExecution {
        if self.background {
            ToolExecution::Background
        } else {
            ToolExecution::Inline
        }
    }

    async fn invoke(&self, ctx: ToolContext) -> izwi_agent::Result<ToolOutput> {
        self.pending
            .wait(&ctx.call_id)
            .await
            .map(|text| ToolOutput { text })
            .ok_or_else(|| AgentError::Tool("The call was abandoned.".to_string()))
    }
}

/// A turn's event as the voice socket's message: `function_call` when a
/// call starts, `function_call_completed` when it ends, `agent_handoff`
/// when the conversation changes agent and `task_completed` when an agent
/// records its task.
pub(crate) fn function_call_message(
    event: &AgentEvent,
    utterance_id: &str,
    utterance_seq: u64,
) -> Option<Value> {
    match event {
        AgentEvent::ToolCallStarted {
            call_id,
            name,
            arguments,
            executor,
        } => Some(json!({
            "type": "function_call",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "executor": (executor != "none").then_some(executor),
        })),
        AgentEvent::ToolCallCompleted {
            call_id,
            name,
            status,
            output,
        } => Some(json!({
            "type": "function_call_completed",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "call_id": call_id,
            "name": name,
            // An interrupted call is an error on the wire; a background
            // call reports `started` and its result reaches the next turn.
            "status": if status == "cancelled" { "error" } else { status.as_str() },
            "output": output,
        })),
        AgentEvent::Handoff { from, to } => Some(json!({
            "type": "agent_handoff",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "from": from,
            "to": to,
        })),
        AgentEvent::TaskCompleted { name, result } => Some(json!({
            "type": "task_completed",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "name": name,
            "result": result,
        })),
        _ => None,
    }
}

/// A function tool in the chat completions form.
fn chat_function_tool(tool: Value) -> Result<Value, String> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err("Voice tools must be function tools (`\"type\": \"function\"`)".to_string());
    }
    let function = match tool.get("function") {
        Some(function) => function.clone(),
        // The Responses form keeps the function's fields on the tool.
        None => {
            let mut function = tool.clone();
            if let Some(map) = function.as_object_mut() {
                map.remove("type");
            }
            function
        }
    };
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "Voice tool name {name:?} must be letters, digits, `_` or `-`"
        ));
    }
    let background = tool.get("background").and_then(Value::as_bool) == Some(true)
        || function.get("background").and_then(Value::as_bool) == Some(true);
    let mut function = function;
    if let Some(function) = function.as_object_mut() {
        function.remove("background");
    }
    Ok(json!({"type": "function", "function": function, "background": background}))
}

/// The client's calls a voice session awaits, by call id. A turn registers
/// each call it sends; the client's `function_call_output` settles it once.
#[derive(Clone, Default)]
pub(crate) struct PendingFunctionCalls {
    calls: Arc<Mutex<HashMap<String, oneshot::Sender<String>>>>,
}

impl PendingFunctionCalls {
    /// The client's output for `call_id`, or `None` when the call is
    /// abandoned. A wait that is dropped (the turn timed out or was
    /// interrupted) no longer accepts an output.
    pub(crate) async fn wait(&self, call_id: &str) -> Option<String> {
        struct Forget<'a>(&'a PendingFunctionCalls, &'a str);
        impl Drop for Forget<'_> {
            fn drop(&mut self) {
                self.0.lock().remove(self.1);
            }
        }
        let (tx, rx) = oneshot::channel();
        self.lock().insert(call_id.to_string(), tx);
        let _forget = Forget(self, call_id);
        rx.await.ok()
    }

    /// Settle `call_id` with the client's `output` (text as it is, other
    /// JSON as JSON). Returns whether a turn awaited it.
    pub(crate) fn resolve(&self, call_id: &str, output: &Value) -> bool {
        let output = match output {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        self.lock()
            .remove(call_id)
            .is_some_and(|tx| tx.send(output).is_ok())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, oneshot::Sender<String>>> {
        self.calls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "parameters": {}}})
    }

    #[test]
    fn sessions_offer_builtin_and_client_tools_on_lfm3_audio_only() {
        let tools = VoiceTools::parse(Some(vec![weather()]), None, None, true)
            .unwrap()
            .unwrap();
        let names = tools
            .registry(&PendingFunctionCalls::default())
            .tool_names();
        assert_eq!(names, ["time", "get_weather"]);
        assert_eq!(tools.timeout, Duration::from_secs(10));
        let registry = tools.registry(&PendingFunctionCalls::default());
        assert_eq!(registry.find("time").unwrap().executor(), "server");
        assert_eq!(registry.find("get_weather").unwrap().executor(), "client");
        assert!(registry.find("nope").is_none());

        // The Responses form, no built-ins, a clamped timeout.
        let responses = json!({"type": "function", "name": "lookup", "parameters": {}});
        let tools = VoiceTools::parse(Some(vec![responses]), Some(false), Some(500), true)
            .unwrap()
            .unwrap();
        assert_eq!(
            tools
                .registry(&PendingFunctionCalls::default())
                .tool_names(),
            ["lookup"]
        );
        assert_eq!(tools.timeout, Duration::from_secs(1));
        assert_eq!(VoiceTools::parse(None, Some(false), None, true), Ok(None));

        // Other sessions take no tools.
        assert_eq!(VoiceTools::parse(None, None, None, false), Ok(None));
        assert_eq!(VoiceTools::parse(None, Some(false), None, false), Ok(None));
        for (tools, builtin, timeout) in [
            (Some(vec![weather()]), None, None),
            (None, Some(true), None),
            (None, None, Some(5_000)),
        ] {
            assert!(VoiceTools::parse(tools, builtin, timeout, false).is_err());
        }
    }

    #[test]
    fn client_tools_must_be_named_functions_that_do_not_clash() {
        let time = json!({"type": "function", "function": {"name": "time"}});
        for tools in [
            vec![time.clone()],
            vec![weather(), weather()],
            vec![json!({"type": "web_search"})],
            vec![json!({"type": "function", "function": {"name": "a b"}})],
        ] {
            assert!(VoiceTools::parse(Some(tools), None, None, true).is_err());
        }
        // Without the built-ins, `time` is the client's.
        let tools = VoiceTools::parse(Some(vec![time]), Some(false), None, true)
            .unwrap()
            .unwrap();
        let registry = tools.registry(&PendingFunctionCalls::default());
        assert_eq!(registry.find("time").unwrap().executor(), "client");
    }

    #[tokio::test]
    async fn client_calls_settle_once_and_a_dropped_wait_forgets_its_call() {
        let pending = PendingFunctionCalls::default();
        let waiting = {
            let pending = pending.clone();
            tokio::spawn(async move { pending.wait("call_1").await })
        };
        while !pending.lock().contains_key("call_1") {
            tokio::task::yield_now().await;
        }
        assert!(pending.resolve("call_1", &json!({"temperature_c": 21})));
        assert!(!pending.resolve("call_1", &json!("again")));
        assert_eq!(
            waiting.await.unwrap(),
            Some(r#"{"temperature_c":21}"#.to_string())
        );

        // A turn that stops waiting (timeout, interruption) drops the wait.
        let timed_out =
            tokio::time::timeout(Duration::from_millis(10), pending.wait("call_2")).await;
        assert!(timed_out.is_err());
        assert!(!pending.resolve("call_2", &json!("late")));
    }

    #[tokio::test]
    async fn a_client_tool_answers_with_the_clients_output() {
        let tools = VoiceTools::parse(Some(vec![weather()]), Some(false), None, true)
            .unwrap()
            .unwrap();
        let pending = PendingFunctionCalls::default();
        let tool = tools.registry(&pending).find("get_weather").unwrap();
        let call = tokio::spawn(async move {
            tool.invoke(ToolContext {
                session_id: "s".into(),
                thread_id: "t".into(),
                call_id: "call_9".into(),
                user_text: String::new(),
                arguments: json!({}),
                cancel: izwi_agent::TurnCancel::new(),
            })
            .await
        });
        while !pending.lock().contains_key("call_9") {
            tokio::task::yield_now().await;
        }
        assert!(pending.resolve("call_9", &json!("Sunny.")));
        assert_eq!(call.await.unwrap().unwrap().text, "Sunny.");
    }

    #[test]
    fn a_client_tool_marked_background_does_not_hold_the_reply() {
        let report = json!({
            "type": "function", "background": true,
            "function": {"name": "run_report", "parameters": {"type": "object"}}
        });
        let tools = VoiceTools::parse(Some(vec![report, weather()]), Some(false), None, true)
            .unwrap()
            .unwrap();
        let registry = tools.registry(&PendingFunctionCalls::default());
        assert_eq!(
            registry.find("run_report").unwrap().execution(),
            ToolExecution::Background
        );
        assert_eq!(
            registry.find("get_weather").unwrap().execution(),
            ToolExecution::Inline
        );
        // The flag is the server's, not part of what the model reads.
        let definitions = registry.function_definitions(None);
        assert!(definitions[0]["function"].get("background").is_none());
        assert!(definitions[0].get("background").is_none());
    }

    #[test]
    fn tool_events_read_as_the_sockets_function_call_messages() {
        let started = AgentEvent::ToolCallStarted {
            call_id: "c1".into(),
            name: "time".into(),
            arguments: "{}".into(),
            executor: "server".into(),
        };
        assert_eq!(
            function_call_message(&started, "utt-1", 1).unwrap(),
            json!({
                "type": "function_call", "utterance_id": "utt-1", "utterance_seq": 1,
                "call_id": "c1", "name": "time", "arguments": "{}", "executor": "server",
            })
        );
        let interrupted = AgentEvent::ToolCallCompleted {
            call_id: "c1".into(),
            name: "time".into(),
            status: "cancelled".into(),
            output: "The turn was interrupted.".into(),
        };
        let message = function_call_message(&interrupted, "utt-1", 1).unwrap();
        assert_eq!(message["type"], "function_call_completed");
        assert_eq!(message["status"], "error");
        let handoff = AgentEvent::Handoff {
            from: "front".into(),
            to: "billing".into(),
        };
        let message = function_call_message(&handoff, "utt-1", 1).unwrap();
        assert_eq!(message["type"], "agent_handoff");
        assert_eq!(message["to"], "billing");
        let task = AgentEvent::TaskCompleted {
            name: "refund".into(),
            result: json!({"invoice": "A1"}),
        };
        let message = function_call_message(&task, "utt-1", 1).unwrap();
        assert_eq!(message["result"]["invoice"], "A1");
        let other = AgentEvent::TextDelta { text: "hi".into() };
        assert!(function_call_message(&other, "utt-1", 1).is_none());
    }
}

//! Tools a unified voice turn may call (LFM3-Audio): the server's built-in
//! tools, the function tools a client declares on the voice websocket, and
//! the client's calls a session awaits.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use izwi_agent::{NoopTool, TimeTool, ToolContext, ToolRegistry};
use serde_json::{json, Value};
use tokio::sync::oneshot;

/// The most model rounds one spoken turn runs: a round that calls tools is
/// followed by one that hears their results.
pub(crate) const MAX_TOOL_ROUNDS: usize = 3;
/// The most calls one spoken turn makes.
pub(crate) const MAX_TURN_FUNCTION_CALLS: usize = 8;
const MAX_CLIENT_TOOLS: usize = 64;
const DEFAULT_FUNCTION_CALL_TIMEOUT_MS: u64 = 10_000;
const FUNCTION_CALL_TIMEOUT_RANGE_MS: (u64, u64) = (1_000, 60_000);

/// The server's built-in tools, shared by agent turns and voice turns.
pub(crate) fn builtin_tool_registry() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(NoopTool);
    tools.register(TimeTool);
    tools
}

/// The tools a voice session's turns may call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VoiceTools {
    builtin: bool,
    /// The client's function tools, in the chat completions form.
    client: Vec<Value>,
    pub(crate) timeout: Duration,
}

/// Who runs a call.
pub(crate) enum FunctionExecutor {
    Server(Arc<dyn izwi_agent::Tool>),
    Client,
}

impl FunctionExecutor {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Server(_) => "server",
            Self::Client => "client",
        }
    }
}

impl VoiceTools {
    /// The tools `input_stream_start` offers: the built-in tools unless
    /// `builtin_tools` is `false`, and the client's `tools` (OpenAI function
    /// tools, in the chat completions or the Responses form), whose calls
    /// the client answers within `function_call_timeout_ms` (default 10 s,
    /// 1 to 60 s). Only a unified session on LFM3-Audio calls tools.
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
                    "Voice tools need unified mode with an LFM3-Audio `s2s_model_id`".to_string(),
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
            timeout: Duration::from_millis(timeout_ms),
        }))
    }

    /// The tool definitions the model sees: the built-in ones, then the
    /// client's.
    pub(crate) fn definitions(&self) -> Vec<Value> {
        let mut definitions = if self.builtin {
            builtin_tool_registry().function_definitions()
        } else {
            Vec::new()
        };
        definitions.extend(self.client.iter().cloned());
        definitions
    }

    /// Who runs a call to `name`, or `None` for a tool the session does not
    /// offer.
    pub(crate) fn executor(&self, name: &str) -> Option<FunctionExecutor> {
        if self.builtin {
            if let Some(tool) = builtin_tool_registry().find(name) {
                return Some(FunctionExecutor::Server(tool));
            }
        }
        self.client
            .iter()
            .any(|tool| tool["function"]["name"].as_str() == Some(name))
            .then_some(FunctionExecutor::Client)
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
    Ok(json!({"type": "function", "function": function}))
}

/// The outcome of one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FunctionCallOutcome {
    /// `ok`, `error` or `timeout`.
    pub(crate) status: &'static str,
    /// What the model reads as the call's result.
    pub(crate) output: String,
}

impl FunctionCallOutcome {
    fn ok(output: String) -> Self {
        Self {
            status: "ok",
            output,
        }
    }

    pub(crate) fn error(message: impl Into<String>) -> Self {
        Self {
            status: "error",
            output: message.into(),
        }
    }

    fn timeout(timeout: Duration) -> Self {
        Self {
            status: "timeout",
            output: format!("The tool did not answer within {} ms.", timeout.as_millis()),
        }
    }
}

/// Run a built-in tool with the call's `arguments` (a JSON object as text).
pub(crate) async fn run_builtin_call(
    tool: Arc<dyn izwi_agent::Tool>,
    arguments: &str,
    context: ToolContext,
    timeout: Duration,
) -> FunctionCallOutcome {
    let arguments = match serde_json::from_str::<Value>(arguments) {
        Ok(arguments) => arguments,
        Err(err) => return FunctionCallOutcome::error(format!("Invalid arguments: {err}")),
    };
    let context = ToolContext {
        arguments,
        ..context
    };
    match tokio::time::timeout(timeout, tool.invoke(context)).await {
        Ok(Ok(output)) => FunctionCallOutcome::ok(output.text),
        Ok(Err(err)) => FunctionCallOutcome::error(err.to_string()),
        Err(_) => FunctionCallOutcome::timeout(timeout),
    }
}

/// The client's calls a voice session awaits, by call id. A turn registers
/// each call it sends; the client's `function_call_output` settles it once.
#[derive(Clone, Default)]
pub(crate) struct PendingFunctionCalls {
    calls: Arc<Mutex<HashMap<String, oneshot::Sender<String>>>>,
}

impl PendingFunctionCalls {
    /// Await the client's output for `call_id` for at most `timeout`, or
    /// until `cancelled` resolves.
    pub(crate) async fn await_output(
        &self,
        call_id: &str,
        timeout: Duration,
        cancelled: impl std::future::Future<Output = ()>,
    ) -> FunctionCallOutcome {
        let (tx, rx) = oneshot::channel();
        self.lock().insert(call_id.to_string(), tx);
        let outcome = tokio::select! {
            output = tokio::time::timeout(timeout, rx) => match output {
                Ok(Ok(output)) => FunctionCallOutcome::ok(output),
                Ok(Err(_)) => FunctionCallOutcome::error("The call was abandoned."),
                Err(_) => FunctionCallOutcome::timeout(timeout),
            },
            () = cancelled => FunctionCallOutcome::error("The turn was interrupted."),
        };
        self.lock().remove(call_id);
        outcome
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
            .definitions()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(names, ["time", "get_weather"]);
        assert_eq!(tools.timeout, Duration::from_secs(10));
        assert_eq!(tools.executor("time").unwrap().as_str(), "server");
        assert_eq!(tools.executor("get_weather").unwrap().as_str(), "client");
        assert!(tools.executor("noop").is_none());

        // The Responses form, no built-ins, a clamped timeout.
        let responses = json!({"type": "function", "name": "lookup", "parameters": {}});
        let tools = VoiceTools::parse(Some(vec![responses]), Some(false), Some(500), true)
            .unwrap()
            .unwrap();
        assert_eq!(tools.definitions()[0]["function"]["name"], "lookup");
        assert!(tools.executor("time").is_none());
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
        assert_eq!(tools.executor("time").unwrap().as_str(), "client");
    }

    #[tokio::test]
    async fn client_calls_settle_once_or_time_out() {
        let pending = PendingFunctionCalls::default();
        let waiting = {
            let pending = pending.clone();
            tokio::spawn(async move {
                pending
                    .await_output("call_1", Duration::from_secs(5), std::future::pending())
                    .await
            })
        };
        while !pending.lock().contains_key("call_1") {
            tokio::task::yield_now().await;
        }
        assert!(pending.resolve("call_1", &json!({"temperature_c": 21})));
        assert!(!pending.resolve("call_1", &json!("again")));
        assert_eq!(
            waiting.await.unwrap(),
            FunctionCallOutcome::ok(r#"{"temperature_c":21}"#.to_string())
        );

        let outcome = pending
            .await_output("call_2", Duration::from_millis(10), std::future::pending())
            .await;
        assert_eq!(outcome.status, "timeout");
        let outcome = pending
            .await_output("call_3", Duration::from_secs(5), std::future::ready(()))
            .await;
        assert_eq!(outcome.status, "error");
        assert!(!pending.resolve("call_3", &json!("late")));
    }

    #[tokio::test]
    async fn builtin_calls_run_with_their_arguments() {
        let tool = builtin_tool_registry().find("time").unwrap();
        let context = ToolContext {
            session_id: "s".into(),
            thread_id: "t".into(),
            user_text: String::new(),
            arguments: Value::Null,
        };
        let outcome =
            run_builtin_call(tool.clone(), "{}", context.clone(), Duration::from_secs(1)).await;
        assert_eq!(outcome.status, "ok");
        assert!(outcome.output.starts_with("It is "));
        let outcome = run_builtin_call(tool, "{", context, Duration::from_secs(1)).await;
        assert_eq!(outcome.status, "error");
    }
}

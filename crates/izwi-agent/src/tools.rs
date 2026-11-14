use crate::cancel::TurnCancel;
use crate::errors::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// A call a model made: the tool's name and its arguments, a JSON object
/// as text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ToolContext {
    pub session_id: String,
    pub thread_id: String,
    pub call_id: String,
    /// What the user said in the turn that made the call.
    pub user_text: String,
    /// The arguments the model called the tool with (an object).
    pub arguments: Value,
    /// Resolves when the turn is cancelled; a tool that waits on something
    /// (a client's answer) stops waiting then.
    pub cancel: TurnCancel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub text: String,
}

/// How a turn runs a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecution {
    /// The turn waits for the result and the model reads it in the next
    /// round.
    Inline,
    /// The tool keeps running after the turn; the model is told it started
    /// and reads the result in a later turn.
    Background,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    /// The JSON Schema of the arguments a model calls the tool with.
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }

    fn execution(&self) -> ToolExecution {
        ToolExecution::Inline
    }

    /// Who runs the tool, for the turn's events: `server`, `client` or
    /// `mcp`.
    fn executor(&self) -> &'static str {
        "server"
    }

    async fn invoke(&self, ctx: ToolContext) -> Result<ToolOutput>;
}

/// The tools a turn may call: tools the turn runs itself, and declared
/// tools the caller runs (a call to one ends the turn with the call
/// pending).
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    external: Vec<Value>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.push(Arc::new(tool));
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Declare a function tool (chat completions form) the caller runs.
    pub fn declare_external(&mut self, definition: Value) {
        self.external.push(definition);
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.external.is_empty()
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.name().to_string())
            .chain(
                self.external
                    .iter()
                    .filter_map(|tool| definition_name(tool).map(str::to_string)),
            )
            .collect()
    }

    /// The tool named `name` that the turn runs itself.
    pub fn find(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|tool| tool.name() == name).cloned()
    }

    /// Whether `name` is a declared tool the caller runs.
    pub fn is_external(&self, name: &str) -> bool {
        self.external
            .iter()
            .any(|tool| definition_name(tool) == Some(name))
    }

    /// The tools as OpenAI function tools; `allowed` narrows them to the
    /// names an agent may call.
    pub fn function_definitions(&self, allowed: Option<&[String]>) -> Vec<Value> {
        let allows = |name: &str| allowed.is_none_or(|names| names.iter().any(|n| n == name));
        self.tools
            .iter()
            .filter(|tool| allows(tool.name()))
            .map(|tool| function_definition(tool.name(), tool.description(), tool.parameters()))
            .chain(
                self.external
                    .iter()
                    .filter(|tool| definition_name(tool).is_some_and(allows))
                    .cloned(),
            )
            .collect()
    }
}

/// A function tool in the chat completions form.
pub fn function_definition(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

fn definition_name(definition: &Value) -> Option<&str> {
    definition["function"]["name"].as_str()
}

/// The result of a background tool, waiting for the session's next turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundResult {
    pub call_id: String,
    pub name: String,
    /// `ok`, `error` or `timeout`.
    pub status: String,
    pub output: String,
}

/// A session's background tool results: a turn's background tools add to
/// it when they finish, and the next turn takes what is there.
#[derive(Debug, Clone, Default)]
pub struct BackgroundResults {
    results: Arc<Mutex<Vec<BackgroundResult>>>,
}

impl BackgroundResults {
    pub fn push(&self, result: BackgroundResult) {
        self.lock().push(result);
    }

    pub fn take(&self) -> Vec<BackgroundResult> {
        std::mem::take(&mut *self.lock())
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<BackgroundResult>> {
        self.results
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str {
        "time"
    }

    fn description(&self) -> &str {
        "Returns the current local date and time."
    }

    async fn invoke(&self, _ctx: ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput {
            text: format!("It is {}.", readable_now(chrono::Local::now())),
        })
    }
}

/// "Monday 20 October 2026, 14:05 (UTC+02:00)": what a model can say aloud.
fn readable_now<Tz: chrono::TimeZone>(now: chrono::DateTime<Tz>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    now.format("%A %-d %B %Y, %H:%M (UTC%:z)").to_string()
}

/// "The current local date and time is Monday 20 October 2026, 14:05.": an
/// instructions line for a model that should know the date and time without
/// calling [`TimeTool`]. It leaves out the UTC offset, which a voice reads
/// out ("UTC plus two hours").
pub fn current_time_instruction() -> String {
    time_instruction(chrono::Local::now())
}

fn time_instruction<Tz: chrono::TimeZone>(now: chrono::DateTime<Tz>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    format!(
        "The current local date and time is {}.",
        now.format("%A %-d %B %Y, %H:%M")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_lists_its_own_and_declared_tools() {
        let mut tools = ToolRegistry::new();
        tools.register(TimeTool);
        tools.declare_external(function_definition("lookup", "Look it up.", json!({})));
        assert!(tools.find("time").is_some());
        assert!(tools.find("lookup").is_none());
        assert!(tools.is_external("lookup"));
        assert!(!tools.is_external("time"));
        assert_eq!(tools.tool_names(), ["time", "lookup"]);
        assert_eq!(
            tools.function_definitions(None)[0],
            json!({
                "type": "function",
                "function": {
                    "name": "time",
                    "description": "Returns the current local date and time.",
                    "parameters": {"type": "object", "properties": {}}
                }
            })
        );
        let only = ["lookup".to_string()];
        let narrowed = tools.function_definitions(Some(&only));
        assert_eq!(narrowed.len(), 1);
        assert_eq!(narrowed[0]["function"]["name"], "lookup");
    }

    #[test]
    fn the_time_reads_as_words() {
        use chrono::TimeZone;
        let now = chrono::FixedOffset::east_opt(2 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 20, 14, 5, 0)
            .unwrap();
        assert_eq!(
            readable_now(now),
            "Tuesday 20 October 2026, 14:05 (UTC+02:00)"
        );
    }

    #[test]
    fn the_time_instruction_has_no_offset() {
        use chrono::TimeZone;
        let now = chrono::FixedOffset::east_opt(2 * 3600)
            .unwrap()
            .with_ymd_and_hms(2026, 11, 13, 9, 5, 0)
            .unwrap();
        assert_eq!(
            time_instruction(now),
            "The current local date and time is Friday 13 November 2026, 09:05."
        );
    }
}

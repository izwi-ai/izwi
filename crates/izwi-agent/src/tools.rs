use crate::errors::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    pub session_id: String,
    pub thread_id: String,
    pub user_text: String,
    /// The arguments a model called the tool with (an object), or `null`
    /// when the tool runs on its own.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub text: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;

    fn description(&self) -> &'static str;

    fn should_auto_run(&self, _user_text: &str) -> bool {
        false
    }

    /// The JSON Schema of the arguments a model calls the tool with.
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    /// Whether a model may call the tool.
    fn model_callable(&self) -> bool {
        true
    }

    async fn invoke(&self, ctx: ToolContext) -> Result<ToolOutput>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.push(Arc::new(tool));
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.name().to_string())
            .collect()
    }

    /// The model-callable tool named `name`.
    pub fn find(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.model_callable() && tool.name() == name)
            .cloned()
    }

    /// The model-callable tools as OpenAI function tools.
    pub fn function_definitions(&self) -> Vec<serde_json::Value> {
        self.tools
            .iter()
            .filter(|tool| tool.model_callable())
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters(),
                    }
                })
            })
            .collect()
    }

    pub fn find_auto_tool(&self, user_text: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.should_auto_run(user_text))
            .cloned()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTool;

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn description(&self) -> &'static str {
        "A no-op tool used for testing and scaffolding."
    }

    fn model_callable(&self) -> bool {
        false
    }

    async fn invoke(&self, _ctx: ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput {
            text: "No-op tool executed.".to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &'static str {
        "time"
    }

    fn description(&self) -> &'static str {
        "Returns the current local date and time."
    }

    /// A question about the time or the date, by whole words: "sometimes",
    /// "update" or "times" do not run it.
    fn should_auto_run(&self, user_text: &str) -> bool {
        has_any_word(user_text, &["time", "date", "today"])
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

/// Whether `text` contains one of `words` as a whole word, ignoring case.
pub(crate) fn has_any_word(text: &str, words: &[&str]) -> bool {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .any(|word| words.contains(&word))
}

#[cfg(test)]
mod word_tests {
    use super::*;

    #[test]
    fn the_time_tool_runs_for_whole_words_only() {
        let tool = TimeTool;
        for asks in ["What time is it?", "What's the date", "What is today?"] {
            assert!(tool.should_auto_run(asks), "{asks}");
        }
        for chat in [
            "Sometimes I forget",
            "Any update on the candidate?",
            "Three times a week",
            "Nice lifetime story",
        ] {
            assert!(!tool.should_auto_run(chat), "{chat}");
        }
    }

    /// A model can call `time` (with no arguments), never the scaffolding
    /// `noop`.
    #[test]
    fn models_call_the_time_tool_but_not_noop() {
        let mut tools = ToolRegistry::new();
        tools.register(NoopTool);
        tools.register(TimeTool);
        assert!(tools.find("time").is_some());
        assert!(tools.find("noop").is_none());
        assert_eq!(
            tools.function_definitions(),
            vec![serde_json::json!({
                "type": "function",
                "function": {
                    "name": "time",
                    "description": "Returns the current local date and time.",
                    "parameters": {"type": "object", "properties": {}}
                }
            })]
        );
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
}

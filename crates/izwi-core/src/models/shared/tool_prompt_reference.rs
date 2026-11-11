//! Test support: tool-calling conversations rendered by the published Qwen
//! chat templates (`tests/fixtures/qwen_tool_prompt_reference.json`, written
//! by `scripts/fixtures/generate_qwen_tool_prompt_reference.py`), turned into
//! the messages a prompt builder receives.

use std::path::PathBuf;

use izwi_model_catalog::chat::tools::{
    append_assistant_tool_calls, assistant_content_with_reasoning, ToolDialect,
};
use izwi_model_catalog::chat::{ChatMessage, ChatRequestConfig, ChatRole};
use serde_json::Value;

/// One fixture case: the request as a prompt builder sees it, and the prompt
/// the family's chat template renders for it.
pub(crate) struct ReferenceCase {
    pub name: String,
    pub messages: Vec<ChatMessage>,
    pub request: ChatRequestConfig,
    pub prompt: String,
}

/// The asserted cases of `family` (`qwen3`, `qwen35`, `qwen36`, `qwen38`).
/// Messages are converted from their OpenAI form the way the chat routes do:
/// past calls in the family's markup after the text, reasoning in a
/// `<think>` block, tool results as tool messages.
pub(crate) fn reference_cases(family: &str, dialect: ToolDialect) -> Vec<ReferenceCase> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen_tool_prompt_reference.json");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture is readable"))
            .expect("fixture is JSON");
    let cases = fixture["families"][family]["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("fixture has no family {family}"));
    assert!(!cases.is_empty());
    cases
        .iter()
        .filter(|case| case["asserted"] == Value::Bool(true))
        .map(|case| ReferenceCase {
            name: case["name"].as_str().unwrap().to_string(),
            messages: case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|message| chat_message(dialect, message))
                .collect(),
            request: ChatRequestConfig {
                enable_thinking: case["enable_thinking"].as_bool(),
                tools: case["tools"].as_array().unwrap().clone(),
                ..ChatRequestConfig::default()
            },
            prompt: case["prompt"].as_str().unwrap().to_string(),
        })
        .collect()
}

fn chat_message(dialect: ToolDialect, message: &Value) -> ChatMessage {
    let mut content = message["content"].as_str().unwrap_or_default().to_string();
    let role = match message["role"].as_str().unwrap() {
        "system" => ChatRole::System,
        "user" => ChatRole::User,
        "tool" => ChatRole::Tool,
        "assistant" => {
            let calls = message["tool_calls"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default();
            append_assistant_tool_calls(dialect, &mut content, calls).unwrap();
            content =
                assistant_content_with_reasoning(message["reasoning_content"].as_str(), &content);
            ChatRole::Assistant
        }
        other => panic!("unexpected role {other}"),
    };
    ChatMessage { role, content }
}

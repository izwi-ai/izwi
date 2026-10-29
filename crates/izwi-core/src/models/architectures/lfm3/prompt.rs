//! The LFM3 chat template rendered in Rust: `chat_template.jinja` of
//! LiquidAI/LFM3-1.7B @ `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` (LFM3-Audio-2B
//! renders text turns the same way).
//!
//! The rendered string is tokenized once, as `apply_chat_template` does, so
//! special and added tokens split exactly where the template puts them.

use serde_json::Value;

use izwi_model_catalog::chat::tools::{is_deferred_tool, tool_to_json};

use crate::models::shared::chat::{ChatMessage, ChatRole};

pub(crate) const BOS: &str = "<|startoftext|>";
pub(crate) const IM_START: &str = "<|im_start|>";
pub(crate) const IM_END: &str = "<|im_end|>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const EMPTY_THINK: &str = "<think></think>";
const TOOL_RESPONSE_OPEN: &str = "<tool_response>";

/// Template switches (`apply_chat_template` keyword arguments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm3PromptOptions {
    /// The generation prompt opens `<think>` (default) or closes an empty
    /// `<think></think>`.
    pub(crate) enable_thinking: bool,
    /// Keep the reasoning of assistant turns before the last user turn
    /// (default off: they are reduced to their final answer).
    pub(crate) preserve_thinking: bool,
}

impl Default for Lfm3PromptOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            preserve_thinking: false,
        }
    }
}

/// Render `messages` and `tools` like the checkpoint template with
/// `add_generation_prompt=True`.
///
/// * BOS, then a leading system message as the system turn; tool definitions
///   (except deferred ones) append `<tools>` with one JSON object per line,
///   after a blank line when the system turn has text.
/// * Every turn is `<|im_start|>{role}\n{content}<|im_end|>\n`.
/// * An assistant turn at or before the last user turn keeps only the text
///   after its last `</think>`, trimmed, behind an empty `<think></think>`;
///   a later one keeps its content and gets `<think></think>` only when it
///   has no `</think>`. `preserve_thinking` keeps every turn whole.
/// * A tool turn is wrapped in `<tool_response>\n...\n</tool_response>` unless
///   it already holds one or is empty (rendered by the gateway, which
///   resolves deferred tools: `api/openai/tool_response.rs`).
/// * The generation prompt `<|im_start|>assistant\n<think>`, or
///   `...\n<think></think>` with thinking off.
///
/// Past assistant tool calls arrive already rendered after the turn's text
/// (the gateway writes them the template's way).
pub(crate) fn render_lfm3_chat_prompt(
    messages: &[ChatMessage],
    tools: &[Value],
    options: Lfm3PromptOptions,
) -> String {
    let mut out = String::from(BOS);
    let (system, turns) = match messages.split_first() {
        Some((first, rest)) if matches!(first.role, ChatRole::System) => {
            (first.content.clone(), rest)
        }
        _ => (String::new(), messages),
    };
    let mut system = system;
    let tool_lines = tools
        .iter()
        .filter(|tool| !is_deferred_tool(tool))
        .map(tool_to_json)
        .collect::<Vec<_>>();
    if !tool_lines.is_empty() {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str("<tools>\n");
        system.push_str(&tool_lines.join("\n"));
        system.push_str("\n</tools>");
    }
    if !system.is_empty() {
        push_turn(&mut out, "system", &system);
    }

    let last_user = turns
        .iter()
        .rposition(|message| matches!(message.role, ChatRole::User));
    for (index, message) in turns.iter().enumerate() {
        match message.role {
            ChatRole::Assistant => {
                let keep_thinking =
                    options.preserve_thinking || last_user.is_none_or(|last| index > last);
                let content = message.content.as_str();
                let rendered = if keep_thinking {
                    if content.contains(THINK_CLOSE) {
                        content.to_string()
                    } else {
                        format!("{EMPTY_THINK}{content}")
                    }
                } else {
                    let answer = match content.rsplit_once(THINK_CLOSE) {
                        Some((_, tail)) => tail.trim(),
                        None => content,
                    };
                    format!("{EMPTY_THINK}{answer}")
                };
                push_turn(&mut out, "assistant", &rendered);
            }
            ChatRole::Tool => {
                // A turn the gateway rendered already (`<tool_response>`
                // blocks, or the empty turn of a tool search that revealed
                // nothing) passes through as the template's own output.
                let content = &message.content;
                let rendered = if content.is_empty() || content.contains(TOOL_RESPONSE_OPEN) {
                    content.clone()
                } else {
                    format!("{TOOL_RESPONSE_OPEN}\n{content}\n</tool_response>")
                };
                push_turn(&mut out, "tool", &rendered);
            }
            ChatRole::System => push_turn(&mut out, "system", &message.content),
            ChatRole::User => push_turn(&mut out, "user", &message.content),
        }
    }

    out.push_str(IM_START);
    out.push_str("assistant\n");
    out.push_str(if options.enable_thinking {
        THINK_OPEN
    } else {
        EMPTY_THINK
    });
    out
}

fn push_turn(out: &mut String, role: &str, content: &str) {
    out.push_str(IM_START);
    out.push_str(role);
    out.push('\n');
    out.push_str(content);
    out.push_str(IM_END);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: ChatRole, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    #[test]
    fn renders_the_generation_prompt_with_thinking_on_and_off() {
        let messages = [message(ChatRole::User, "Hi")];
        assert_eq!(
            render_lfm3_chat_prompt(&messages, &[], Lfm3PromptOptions::default()),
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>"
        );
        let off = Lfm3PromptOptions {
            enable_thinking: false,
            ..Lfm3PromptOptions::default()
        };
        assert_eq!(
            render_lfm3_chat_prompt(&messages, &[], off),
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn appends_tools_to_the_system_turn_one_json_object_per_line() {
        let messages = [
            message(ChatRole::System, "Be brief."),
            message(ChatRole::User, "Weather?"),
        ];
        let tools = [
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}),
            json!({"type": "function", "defer_loading": true, "function": {"name": "hidden"}}),
            json!({"name": "search", "description": "Find tools"}),
        ];
        assert_eq!(
            render_lfm3_chat_prompt(&messages, &tools, Lfm3PromptOptions::default()),
            "<|startoftext|><|im_start|>system\nBe brief.\n\n<tools>\n\
             {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\"}}\n\
             {\"description\": \"Find tools\", \"name\": \"search\"}\n</tools><|im_end|>\n\
             <|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think>"
        );
        // Tools alone make the system turn.
        let rendered =
            render_lfm3_chat_prompt(&messages[1..], &tools[..1], Lfm3PromptOptions::default());
        assert!(rendered.starts_with("<|startoftext|><|im_start|>system\n<tools>\n{"));
    }

    #[test]
    fn past_assistant_turns_keep_only_their_answer() {
        let messages = [
            message(ChatRole::User, "a"),
            message(ChatRole::Assistant, "<think>plan</think>\n\n  answer  "),
            message(ChatRole::User, "b"),
        ];
        let rendered = render_lfm3_chat_prompt(&messages, &[], Lfm3PromptOptions::default());
        assert!(rendered.contains(
            "<|im_start|>assistant\n<think></think>answer<|im_end|>\n<|im_start|>user\nb"
        ));
        let preserved = render_lfm3_chat_prompt(
            &messages,
            &[],
            Lfm3PromptOptions {
                preserve_thinking: true,
                ..Lfm3PromptOptions::default()
            },
        );
        assert!(preserved
            .contains("<|im_start|>assistant\n<think>plan</think>\n\n  answer  <|im_end|>"));
    }

    #[test]
    fn the_current_tool_loop_keeps_its_turns_and_wraps_results() {
        let messages = [
            message(ChatRole::User, "Weather in Paris?"),
            message(
                ChatRole::Assistant,
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            ),
            message(ChatRole::Tool, "{\"temp\": 21}"),
        ];
        assert_eq!(
            render_lfm3_chat_prompt(&messages, &[], Lfm3PromptOptions::default()),
            "<|startoftext|><|im_start|>user\nWeather in Paris?<|im_end|>\n\
             <|im_start|>assistant\n<think></think><tool_call>\n<function=get_weather>\n\
             <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
             <|im_start|>tool\n<tool_response>\n{\"temp\": 21}\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn tool_turns_the_gateway_rendered_pass_through() {
        let rendered = "<tool_response>\n<additional_tools>\n{\"name\": \"a\"}\n</additional_tools>\n</tool_response>";
        let messages = [
            message(ChatRole::User, "q"),
            message(ChatRole::Tool, rendered),
            message(ChatRole::Tool, ""),
        ];
        assert_eq!(
            render_lfm3_chat_prompt(&messages, &[], Lfm3PromptOptions::default()),
            format!(
                "<|startoftext|><|im_start|>user\nq<|im_end|>\n<|im_start|>tool\n{rendered}<|im_end|>\n\
                 <|im_start|>tool\n<|im_end|>\n<|im_start|>assistant\n<think>"
            )
        );
    }

    #[test]
    fn a_later_system_message_is_an_ordinary_turn() {
        let messages = [
            message(ChatRole::User, "a"),
            message(ChatRole::System, "switch"),
        ];
        assert!(
            render_lfm3_chat_prompt(&messages, &[], Lfm3PromptOptions::default())
                .contains("<|im_start|>system\nswitch<|im_end|>\n")
        );
    }
}

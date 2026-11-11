//! Tool definitions, past tool calls and tool results as each family's chat
//! template writes them. The prompt builders in izwi-core and the request
//! conversions in the gateway and server share these, so the prompt a model
//! reads matches the template it was trained with
//! (`izwi-core/tests/fixtures/qwen_tool_prompt_reference.json`,
//! `lfm3_chat_reference.json`, `lfm25_chat_reference.json`).

use serde_json::{Map, Value};

use super::dialect::ToolDialect;
use super::json::{python_json, write_python_json};
use super::lfm;

/// What the Qwen3.5/3.6/3.8 templates write after the tool list.
const QWEN_XML_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The tools section of a Qwen3.5/3.6/3.8 system turn: the heading, each
/// tool as the template's `tojson` prints it, and the call instructions. The
/// system text follows it after a blank line.
pub fn qwen_xml_tools_prompt(tools: &[Value]) -> String {
    let mut out = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    push_tool_lines(&mut out, tools);
    out.push_str("\n</tools>");
    out.push_str(QWEN_XML_INSTRUCTIONS);
    out
}

/// The tools section of a Qwen3 system turn (the Hermes format). It follows
/// the system text after a blank line.
pub fn hermes_tools_prompt(tools: &[Value]) -> String {
    let mut out = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    push_tool_lines(&mut out, tools);
    out.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    out
}

/// One tool per line, as passed (the Qwen templates print the OpenAI
/// wrapper too).
fn push_tool_lines(out: &mut String, tools: &[Value]) {
    for tool in tools {
        out.push('\n');
        write_python_json(tool, out);
    }
}

/// One tool result in a ChatML prompt whose template reads results as
/// `<tool_response>` blocks in a user turn (the Qwen families). Consecutive
/// results share one turn: `previous_is_tool` and `next_is_tool` say whether
/// the neighbouring messages are tool results too.
pub fn push_chatml_tool_response(
    prompt: &mut String,
    content: &str,
    previous_is_tool: bool,
    next_is_tool: bool,
) {
    if !previous_is_tool {
        prompt.push_str("<|im_start|>user");
    }
    prompt.push_str("\n<tool_response>\n");
    prompt.push_str(content);
    prompt.push_str("\n</tool_response>");
    if !next_is_tool {
        prompt.push_str("<|im_end|>\n");
    }
}

/// An assistant turn's text with the reasoning a client sent back beside it
/// (`reasoning_content`), in the `<think>` block every prompt builder reads.
pub fn assistant_content_with_reasoning(reasoning: Option<&str>, content: &str) -> String {
    match reasoning
        .map(str::trim)
        .filter(|reasoning| !reasoning.is_empty())
    {
        // Text that already carries its own block keeps it.
        Some(reasoning) if !content.contains("</think>") => {
            format!("<think>\n{reasoning}\n</think>\n\n{content}")
        }
        _ => content.to_string(),
    }
}

/// Append an assistant turn's past `tool_calls` (OpenAI form) to its text,
/// in the format the model family writes them.
pub fn append_assistant_tool_calls(
    dialect: ToolDialect,
    content: &mut String,
    tool_calls: &[Value],
) -> Result<(), String> {
    if tool_calls.is_empty() {
        return Ok(());
    }
    let rendered = render_tool_calls(dialect, tool_calls)?;
    match dialect {
        // After text the first call starts on its own line.
        ToolDialect::HermesJson => {
            if !content.is_empty() {
                content.push('\n');
            }
        }
        // The Qwen XML templates put a blank line between text and calls.
        ToolDialect::QwenXml => {
            if !content.trim().is_empty() {
                content.push_str("\n\n");
            }
        }
        // The LFM templates write the calls right after the text.
        ToolDialect::Lfm3Xml | ToolDialect::LfmPythonic => {}
    }
    content.push_str(&rendered);
    Ok(())
}

/// OpenAI `tool_calls` in the family's call markup.
pub fn render_tool_calls(dialect: ToolDialect, tool_calls: &[Value]) -> Result<String, String> {
    if dialect == ToolDialect::LfmPythonic {
        return lfm::render_tool_calls(tool_calls);
    }
    let mut rendered = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        let function = tool_call.get("function").unwrap_or(tool_call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "Each assistant tool call must include `function.name`".to_string())?;
        let arguments = normalize_arguments(function.get("arguments"));
        rendered.push(match dialect {
            ToolDialect::HermesJson => format!(
                "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {}}}\n</tool_call>",
                python_json(&Value::Object(arguments))
            ),
            _ => xml_call(name, &arguments),
        });
    }
    Ok(rendered.join(match dialect {
        ToolDialect::Lfm3Xml => "\n\n",
        _ => "\n",
    }))
}

/// `<tool_call><function=NAME><parameter=K>V</parameter>…</function>
/// </tool_call>`, a value as Qwen3.6, Qwen3.8 and LFM3 write it: a string
/// raw, anything else as the template's `tojson`.
fn xml_call(name: &str, arguments: &Map<String, Value>) -> String {
    let mut block = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in arguments {
        block.push_str(&format!("<parameter={key}>\n"));
        match value {
            Value::String(text) => block.push_str(text),
            other => write_python_json(other, &mut block),
        }
        block.push_str("\n</parameter>\n");
    }
    block.push_str("</function>\n</tool_call>");
    block
}

/// OpenAI sends `arguments` as a JSON-encoded object; anything else is kept
/// under an `arguments` key.
fn normalize_arguments(raw: Option<&Value>) -> Map<String, Value> {
    let wrap = |value: Value| Map::from_iter([("arguments".to_string(), value)]);
    match raw {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(raw)) if raw.trim().is_empty() => Map::new(),
        Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(map)) => map,
            Ok(Value::Null) => Map::new(),
            Ok(value) => wrap(value),
            Err(_) => wrap(Value::String(raw.clone())),
        },
        Some(value) => wrap(value.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, arguments: Value) -> Value {
        json!({"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments.to_string()}})
    }

    #[test]
    fn qwen_xml_calls_keep_argument_order_and_write_values_like_tojson() {
        let mut content = String::from("Checking.");
        append_assistant_tool_calls(
            ToolDialect::QwenXml,
            &mut content,
            &[
                call("set", json!({"unit": "c", "city": "Harare", "on": true, "due": null, "tags": ["a", "b"], "meta": {"k": 1}})),
                call("now", json!({})),
            ],
        )
        .unwrap();
        assert_eq!(
            content,
            "Checking.\n\n<tool_call>\n<function=set>\n<parameter=unit>\nc\n</parameter>\n\
             <parameter=city>\nHarare\n</parameter>\n<parameter=on>\ntrue\n</parameter>\n\
             <parameter=due>\nnull\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n\
             <parameter=meta>\n{\"k\": 1}\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=now>\n</function>\n</tool_call>"
        );
        // Without text the calls open the turn.
        let mut content = String::new();
        append_assistant_tool_calls(
            ToolDialect::QwenXml,
            &mut content,
            &[call("now", json!({}))],
        )
        .unwrap();
        assert!(content.starts_with("<tool_call>"));
    }

    #[test]
    fn hermes_calls_are_json_objects_one_per_block() {
        let mut content = String::from("Checking.");
        append_assistant_tool_calls(
            ToolDialect::HermesJson,
            &mut content,
            &[
                call("get_weather", json!({"unit": "c", "city": "São Paulo"})),
                call("now", json!({})),
            ],
        )
        .unwrap();
        assert_eq!(
            content,
            "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": \"c\", \"city\": \"São Paulo\"}}\n</tool_call>\n\
             <tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>"
        );
    }

    #[test]
    fn lfm_calls_sit_right_after_the_text() {
        let calls = [
            call("get_weather", json!({"city": "Paris"})),
            json!({"function": {"name": "set", "arguments": {"on": true, "level": 2.5, "tags": ["a", "b"]}}}),
        ];
        let mut lfm3 = String::from("Checking.");
        append_assistant_tool_calls(ToolDialect::Lfm3Xml, &mut lfm3, &calls).unwrap();
        assert_eq!(
            lfm3,
            "Checking.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
             </function>\n</tool_call>\n\n<tool_call>\n<function=set>\n<parameter=on>\ntrue\n\
             </parameter>\n<parameter=level>\n2.5\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n\
             </parameter>\n</function>\n</tool_call>"
        );
        let mut lfm25 = String::new();
        append_assistant_tool_calls(ToolDialect::LfmPythonic, &mut lfm25, &calls[..1]).unwrap();
        assert_eq!(
            lfm25,
            "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"
        );
    }

    #[test]
    fn a_call_without_a_name_is_refused() {
        for dialect in [
            ToolDialect::HermesJson,
            ToolDialect::QwenXml,
            ToolDialect::Lfm3Xml,
            ToolDialect::LfmPythonic,
        ] {
            assert!(
                render_tool_calls(dialect, &[json!({"function": {"arguments": "{}"}})]).is_err()
            );
        }
    }

    #[test]
    fn consecutive_tool_results_share_one_user_turn() {
        let mut prompt = String::new();
        push_chatml_tool_response(&mut prompt, "24 C", false, true);
        push_chatml_tool_response(&mut prompt, "14:05", true, false);
        assert_eq!(
            prompt,
            "<|im_start|>user\n<tool_response>\n24 C\n</tool_response>\n<tool_response>\n14:05\n</tool_response><|im_end|>\n"
        );
    }

    #[test]
    fn reasoning_sent_back_is_wrapped_in_a_think_block() {
        assert_eq!(
            assistant_content_with_reasoning(Some(" why "), "answer"),
            "<think>\nwhy\n</think>\n\nanswer"
        );
        assert_eq!(assistant_content_with_reasoning(None, "answer"), "answer");
        assert_eq!(
            assistant_content_with_reasoning(Some(""), "answer"),
            "answer"
        );
        assert_eq!(
            assistant_content_with_reasoning(Some("why"), "<think>own</think>answer"),
            "<think>own</think>answer"
        );
    }

    #[test]
    fn tool_definitions_print_like_tojson_in_client_key_order() {
        let tool = json!({"type": "function", "function": {"name": "f", "description": "d — é", "parameters": {"type": "object", "properties": {}}}});
        let line = r#"{"type": "function", "function": {"name": "f", "description": "d — é", "parameters": {"type": "object", "properties": {}}}}"#;
        assert!(qwen_xml_tools_prompt(std::slice::from_ref(&tool))
            .starts_with(&format!("# Tools\n\nYou have access to the following functions:\n\n<tools>\n{line}\n</tools>\n\nIf you choose")));
        assert!(hermes_tools_prompt(&[tool]).contains(&format!(
            "<tools>\n{line}\n</tools>\n\nFor each function call"
        )));
    }
}

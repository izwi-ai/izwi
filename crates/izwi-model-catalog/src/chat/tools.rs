//! Tool definitions as the LFM chat templates print them: Python's
//! `json.dumps` spacing (the templates' `tojson`), and the deferred tools
//! (`defer_loading`) a template keeps out of the system turn's `<tools>`.

use serde_json::Value;

/// Serialize like Python's `json.dumps(value, ensure_ascii=False)`, which a
/// chat template's `tojson` uses: `", "` and `": "` separators, non-ASCII
/// kept. Object keys follow `serde_json`'s (sorted) map order.
pub fn write_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_python_json(item, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

/// [`write_python_json`] into a new string.
pub fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_python_json(value, &mut out);
    out
}

/// A tool's function object: `function` of an OpenAI tool, else the tool.
pub fn tool_function(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

/// Whether a tool is deferred (`defer_loading` on the tool or its function),
/// listed only once a tool search returns it.
pub fn is_deferred_tool(tool: &Value) -> bool {
    [tool, tool_function(tool)]
        .iter()
        .any(|value| value.get("defer_loading").and_then(Value::as_bool) == Some(true))
}

/// The templates' `tool_to_json`: the function object without
/// `defer_loading`, serialized like Python's `json.dumps`.
pub fn tool_to_json(tool: &Value) -> String {
    let function = tool_function(tool);
    let mut out = String::new();
    match function {
        Value::Object(map) => {
            out.push('{');
            let mut first = true;
            for (key, value) in map {
                if key == "defer_loading" {
                    continue;
                }
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(": ");
                write_python_json(value, &mut out);
            }
            out.push('}');
        }
        Value::String(text) => out.push_str(text),
        other => write_python_json(other, &mut out),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_spaces_like_json_dumps_and_keeps_non_ascii() {
        assert_eq!(
            python_json(&json!({"b": [1, true, null], "a": "é"})),
            r#"{"a": "é", "b": [1, true, null]}"#
        );
    }

    #[test]
    fn deferred_tools_are_marked_on_the_tool_or_its_function() {
        assert!(is_deferred_tool(
            &json!({"type": "function", "defer_loading": true, "function": {"name": "a"}})
        ));
        assert!(is_deferred_tool(
            &json!({"type": "function", "function": {"name": "a", "defer_loading": true}})
        ));
        assert!(!is_deferred_tool(
            &json!({"type": "function", "function": {"name": "a", "defer_loading": false}})
        ));
    }

    #[test]
    fn tool_to_json_prints_the_function_without_defer_loading() {
        assert_eq!(
            tool_to_json(&json!({
                "type": "function",
                "function": {"name": "get_weather", "defer_loading": true, "parameters": {}}
            })),
            r#"{"name": "get_weather", "parameters": {}}"#
        );
    }
}

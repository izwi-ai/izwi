//! Shared OpenAI-compatible API helpers.

use std::collections::HashSet;

use serde_json::Value;

use crate::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

pub(crate) fn parse_tool_choice(raw: Option<&Value>) -> Result<ToolChoice, ApiError> {
    let Some(raw) = raw else {
        return Ok(ToolChoice::Auto);
    };

    match raw {
        Value::Null => Ok(ToolChoice::Auto),
        Value::String(value) => parse_tool_choice_tag(value),
        Value::Object(map) => {
            if let Some(kind) = map.get("type").and_then(Value::as_str) {
                match kind.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(ToolChoice::Auto),
                    "none" => Ok(ToolChoice::None),
                    "required" => Ok(ToolChoice::Required),
                    "function" => {
                        let name = map
                            .get("function")
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|name| !name.is_empty())
                            .ok_or_else(|| {
                                ApiError::bad_request(
                                    "`tool_choice` with type `function` must include `function.name`",
                                )
                            })?;
                        Ok(ToolChoice::Function(name.to_string()))
                    }
                    _ => Err(ApiError::bad_request(
                        "Unsupported `tool_choice` type. Use `auto`, `none`, `required`, or `function`.",
                    )),
                }
            } else if let Some(name) = map
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
            {
                Ok(ToolChoice::Function(name.to_string()))
            } else {
                Err(ApiError::bad_request(
                    "Unsupported `tool_choice` object. Expected `{\"type\":\"auto|none|required|function\"}`.",
                ))
            }
        }
        _ => Err(ApiError::bad_request(
            "Unsupported `tool_choice` value. Use a string or object.",
        )),
    }
}

fn parse_tool_choice_tag(raw: &str) -> Result<ToolChoice, ApiError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "required" => Ok(ToolChoice::Required),
        _ => Err(ApiError::bad_request(
            "Unsupported `tool_choice` string. Use `auto`, `none`, or `required`.",
        )),
    }
}

pub(crate) fn should_inject_tools(choice: &ToolChoice) -> bool {
    !matches!(choice, ToolChoice::None)
}

pub(crate) fn tool_choice_instruction(
    choice: &ToolChoice,
    tools: &[Value],
) -> Result<Option<String>, ApiError> {
    match choice {
        ToolChoice::Auto => Ok(None),
        ToolChoice::None => Ok(Some(
            "Tool use is disabled for this response. Do not call any function.".to_string(),
        )),
        ToolChoice::Required => {
            if tools.is_empty() {
                return Err(ApiError::bad_request(
                    "`tool_choice` is `required` but no `tools` were provided",
                ));
            }
            Ok(Some(
                "You MUST call one of the provided functions in your next response.".to_string(),
            ))
        }
        ToolChoice::Function(name) => {
            if tools.is_empty() {
                return Err(ApiError::bad_request(
                    "`tool_choice` requests a specific function but no `tools` were provided",
                ));
            }

            let available = tool_names(tools);
            if !available.contains(name.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "`tool_choice` requested function `{name}` but it was not present in `tools`"
                )));
            }

            Ok(Some(format!(
                "You MUST call the function `{name}` in your next response. Do not call any other function."
            )))
        }
    }
}

fn tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(extract_tool_name)
        .map(str::to_string)
        .collect()
}

fn extract_tool_name(tool: &Value) -> Option<&str> {
    tool.get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| tool.get("name").and_then(Value::as_str))
}

#[cfg(test)]
mod tests {
    use super::{parse_tool_choice, should_inject_tools, tool_choice_instruction, ToolChoice};
    use serde_json::json;

    #[test]
    fn parses_tool_choice_string_variants() {
        assert_eq!(
            parse_tool_choice(Some(&json!("auto"))).expect("auto should parse"),
            ToolChoice::Auto
        );
        assert_eq!(
            parse_tool_choice(Some(&json!("none"))).expect("none should parse"),
            ToolChoice::None
        );
        assert_eq!(
            parse_tool_choice(Some(&json!("required"))).expect("required should parse"),
            ToolChoice::Required
        );
    }

    #[test]
    fn parses_tool_choice_function_object() {
        let parsed = parse_tool_choice(Some(&json!({
            "type": "function",
            "function": {"name": "get_weather"}
        })))
        .expect("function choice should parse");
        assert_eq!(parsed, ToolChoice::Function("get_weather".to_string()));
    }

    #[test]
    fn disables_tools_for_none_choice() {
        assert!(!should_inject_tools(&ToolChoice::None));
        let instruction =
            tool_choice_instruction(&ToolChoice::None, &[]).expect("none should be valid");
        assert!(instruction
            .as_deref()
            .is_some_and(|text| text.contains("disabled")));
    }

    #[test]
    fn required_choice_errors_without_tools() {
        let err = tool_choice_instruction(&ToolChoice::Required, &[])
            .expect_err("required without tools must fail");
        assert!(err.message.contains("required"));
    }

    #[test]
    fn function_choice_errors_when_function_is_missing() {
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "get_time"}
        })];

        let err = tool_choice_instruction(&ToolChoice::Function("get_weather".to_string()), &tools)
            .expect_err("missing function should fail");
        assert!(err.message.contains("get_weather"));
    }
}

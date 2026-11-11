//! OpenAI tool calling on the chat routes: how a request's `tools`,
//! `tool_choice` and `parallel_tool_calls` apply to its model. The formats,
//! prompt rendering and the parser live in
//! [`izwi_model_catalog::chat::tools`], shared with the prompt builders.

use std::sync::Arc;

use izwi_model_catalog::{chat_supports_tools, ModelVariant};
use serde_json::Value;

pub use izwi_model_catalog::chat::tools::{
    parse_answer, ParsedOutput, ToolCall, ToolCallDelta, ToolCallDeltaFunction, ToolCallFunction,
    ToolCallParser, ToolChoice, ToolDialect, ToolPlan,
};

use crate::error::ApiError;

/// How a model family writes tool calls.
pub type ToolCallFormat = ToolDialect;

/// How a request's `tools` and `tool_choice` apply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolPolicy {
    /// The tool definitions rendered into the prompt.
    pub tools: Vec<Value>,
    /// How the answer's calls are made and parsed; `None` leaves the answer
    /// text.
    pub plan: Option<Arc<ToolPlan>>,
}

/// Resolve OpenAI `tools`, `tool_choice` and `parallel_tool_calls` for
/// `variant`. `"auto"` (the default) offers the tools and parses calls;
/// `"none"` does neither. Forcing a call (`"required"` or a named function)
/// is refused.
pub fn resolve_tool_policy(
    variant: ModelVariant,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolPolicy, ApiError> {
    let tools = tools.unwrap_or_default();
    let choice = ToolChoice::parse(tool_choice)
        .map_err(|message| ApiError::invalid_param("tool_choice", message))?;
    if choice.is_forced() {
        return Err(ApiError::unsupported_param(
            "tool_choice",
            "`tool_choice` \"required\" and named functions are not supported; use \"auto\" or \"none\"",
        ));
    }
    if let Some(index) = tools.iter().position(|tool| !is_function_tool(tool)) {
        return Err(ApiError::invalid_param(
            "tools",
            format!("`tools[{index}]` must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"),
        ));
    }
    if tools.is_empty() || choice == ToolChoice::None {
        return Ok(ToolPolicy::default());
    }
    let dialect = ToolDialect::of(variant)
        .filter(|_| chat_supports_tools(variant))
        .ok_or_else(|| {
            ApiError::unsupported_param(
                "tools",
                format!("Model {variant} has no tool-calling chat template"),
            )
        })?;
    let plan = ToolPlan::new(
        dialect,
        tools.to_vec(),
        choice,
        parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|message| ApiError::invalid_param("tool_choice", message))?;
    Ok(ToolPolicy {
        tools: tools.to_vec(),
        plan: Some(Arc::new(plan)),
    })
}

fn is_function_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("function")
        && tool
            .pointer("/function/name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_choice_offers_parses_or_refuses() {
        let qwen = ModelVariant::Qwen359BGguf;
        let tools = [json!({"type": "function", "function": {"name": "now"}})];
        let policy = |variant, tools: &[Value], choice: Option<Value>| {
            resolve_tool_policy(variant, Some(tools), choice.as_ref(), None)
        };
        let dialect = |policy: ToolPolicy| policy.plan.map(|plan| plan.dialect);

        let auto = policy(qwen, &tools, None).unwrap();
        assert_eq!(auto.tools, tools);
        assert_eq!(dialect(auto.clone()), Some(ToolDialect::QwenXml));
        assert_eq!(policy(qwen, &tools, Some(json!("auto"))).unwrap(), auto);
        assert_eq!(
            dialect(policy(ModelVariant::Lfm2512BInstructGguf, &tools, None).unwrap()),
            Some(ToolDialect::LfmPythonic)
        );
        // "none" neither offers the tools nor parses calls.
        assert_eq!(
            policy(qwen, &tools, Some(json!("none"))).unwrap(),
            ToolPolicy::default()
        );
        assert_eq!(policy(qwen, &[], None).unwrap(), ToolPolicy::default());

        let refused = |variant, tools: &[Value], choice: Option<Value>| {
            let error = policy(variant, tools, choice).expect_err("refused");
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
            error.detail.unwrap()
        };
        for forced in [
            json!("required"),
            json!({"type": "function", "function": {"name": "now"}}),
        ] {
            let detail = refused(qwen, &tools, Some(forced));
            assert_eq!(detail.param, Some("tool_choice"));
            assert_eq!(detail.code, Some("unsupported_parameter"));
        }
        assert_eq!(
            refused(qwen, &tools, Some(json!("sometimes"))).code,
            Some("invalid_value")
        );
        assert_eq!(
            refused(qwen, &[json!({"name": "now"})], None).param,
            Some("tools")
        );
        // A model without a tool template refuses tools it would ignore.
        let detail = refused(ModelVariant::Gemma34BIt, &tools, None);
        assert_eq!(
            (detail.param, detail.code),
            (Some("tools"), Some("unsupported_parameter"))
        );
        assert_eq!(
            policy(ModelVariant::Gemma34BIt, &tools, Some(json!("none"))).unwrap(),
            ToolPolicy::default()
        );
    }

    #[test]
    fn parallel_tool_calls_off_limits_the_answer_to_one_call() {
        let tools = [json!({"type": "function", "function": {"name": "now"}})];
        let policy =
            resolve_tool_policy(ModelVariant::Qwen359BGguf, Some(&tools), None, Some(false))
                .unwrap();
        assert_eq!(policy.plan.unwrap().max_calls(), Some(1));
    }
}

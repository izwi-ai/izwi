//! OpenAI tool calling on the chat routes: how a request's `tools`,
//! `tool_choice` and `parallel_tool_calls` apply to its model. The formats,
//! prompt rendering and the parser live in
//! [`izwi_model_catalog::chat::tools`], shared with the prompt builders.

use std::sync::Arc;

use izwi_model_catalog::chat::tools::ToolConstraint;
use izwi_model_catalog::{chat_supports_tool_constraint, chat_supports_tools, ModelVariant};
use serde_json::Value;

pub use izwi_model_catalog::chat::tools::{
    parse_answer, ParsedOutput, ToolCall, ToolCallDelta, ToolCallDeltaFunction, ToolCallFunction,
    ToolCallParser, ToolChoice, ToolDialect, ToolPlan,
};

use crate::app::chat::ChatExecutionRequest;
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
    /// What the sampler holds the calls to: set when a tool is `strict` or
    /// the answer must make a call, on a model whose sampler can.
    pub constraint: Option<ToolConstraint>,
}

impl ToolPolicy {
    /// Apply the policy to the engine request: a forced call's opening goes
    /// into the prompt, and an answer limited to one call stops when that
    /// call closes.
    pub fn apply(&self, request: &mut ChatExecutionRequest) {
        let Some(plan) = &self.plan else {
            return;
        };
        if let Some(prefix) = plan.response_prefix() {
            request.chat_config.response_prefix = prefix;
        }
        request.chat_config.tool_constraint = self.constraint.clone().map(Box::new);
        if let Some(stop) = plan.stop_sequence() {
            if !request.stop.iter().any(|existing| existing == stop) {
                request.stop.push(stop.to_string());
            }
        }
    }

    /// Whether the answer must make a call, which turns thinking off for
    /// the turn.
    pub fn is_forced(&self) -> bool {
        self.plan
            .as_ref()
            .is_some_and(|plan| plan.choice.is_forced())
    }
}

/// Resolve OpenAI `tools`, `tool_choice` and `parallel_tool_calls` for
/// `variant`. `"auto"` (the default) offers the tools and parses calls;
/// `"none"` does neither; `"required"` and a named function make the answer
/// open with a call. Tools for a model without a tool template are refused
/// rather than ignored.
pub fn resolve_tool_policy(
    variant: ModelVariant,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolPolicy, ApiError> {
    let tools = tools.unwrap_or_default();
    let choice = ToolChoice::parse(tool_choice)
        .map_err(|message| ApiError::invalid_param("tool_choice", message))?;
    if let Some(index) = tools.iter().position(|tool| !is_function_tool(tool)) {
        return Err(ApiError::invalid_param(
            "tools",
            format!("`tools[{index}]` must be {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"),
        ));
    }
    if choice == ToolChoice::None {
        return Ok(ToolPolicy::default());
    }
    if tools.is_empty() {
        if choice.is_forced() {
            return Err(ApiError::invalid_param(
                "tool_choice",
                "`tool_choice` requires a tool call, but the request has no `tools`",
            ));
        }
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
    // The model reads the definitions without OpenAI's `strict` flag, which
    // is an instruction to the server.
    let strict = tools.iter().any(is_strict);
    let tools: Vec<Value> = tools.iter().map(without_strict).collect();
    let plan = ToolPlan::new(
        dialect,
        tools.clone(),
        choice,
        parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|message| ApiError::invalid_param("tool_choice", message))?;
    let constraint = tool_constraint(variant, &plan, strict)?;
    Ok(ToolPolicy {
        tools,
        plan: Some(Arc::new(plan)),
        constraint,
    })
}

/// The sampler constraint for `plan`. A `strict` tool needs one: a model
/// whose sampler has none, or a schema the grammar cannot express, is
/// refused. A forced call without `strict` gets one where it can, and is
/// otherwise opened by the prompt alone.
fn tool_constraint(
    variant: ModelVariant,
    plan: &ToolPlan,
    strict: bool,
) -> Result<Option<ToolConstraint>, ApiError> {
    if !chat_supports_tool_constraint(variant) {
        if strict {
            return Err(ApiError::unsupported_param(
                "tools",
                format!("Model {variant} cannot enforce `strict` tool arguments; remove `strict`"),
            ));
        }
        return Ok(None);
    }
    let Some(constraint) = plan.constraint(strict) else {
        return Ok(None);
    };
    match check_grammars(&constraint) {
        Ok(()) => Ok(Some(constraint)),
        Err(reason) if strict => Err(ApiError::invalid_param(
            "tools",
            format!("A `strict` tool's parameter schema cannot be enforced: {reason}"),
        )),
        Err(reason) => {
            tracing::warn!(%reason, "forced tool call runs without a constraint");
            Ok(None)
        }
    }
}

/// Compile a constraint's grammars, so a schema the grammar compiler does
/// not support is found before the request runs. The check is independent
/// of the model's tokenizer.
fn check_grammars(constraint: &ToolConstraint) -> Result<(), String> {
    use llguidance::api::TopLevelGrammar;
    use llguidance::toktrie::ApproximateTokEnv;
    use llguidance::ParserFactory;
    use std::sync::OnceLock;

    static FACTORY: OnceLock<Result<ParserFactory, String>> = OnceLock::new();
    let factory = FACTORY
        .get_or_init(|| {
            let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
                .map_err(|err| err.to_string())?;
            factory.quiet();
            Ok(factory)
        })
        .as_ref()
        .map_err(Clone::clone)?;
    for grammar in [Some(&constraint.call), constraint.forced.as_ref()]
        .into_iter()
        .flatten()
    {
        factory
            .create_parser(TopLevelGrammar::from_lark(grammar.clone()))
            .map_err(|err| first_line(&err.to_string()))?;
    }
    Ok(())
}

fn first_line(message: &str) -> String {
    message.lines().next().unwrap_or(message).to_string()
}

/// Whether a tool asks for `strict` arguments (on its function, as chat
/// completions sends it, or on the tool).
fn is_strict(tool: &Value) -> bool {
    [tool.pointer("/function/strict"), tool.get("strict")]
        .into_iter()
        .flatten()
        .any(|strict| strict.as_bool() == Some(true))
}

/// `tool` without the `strict` key of its function.
fn without_strict(tool: &Value) -> Value {
    let mut tool = tool.clone();
    if let Some(function) = tool.get_mut("function").and_then(Value::as_object_mut) {
        function.shift_remove("strict");
    }
    if let Some(tool) = tool.as_object_mut() {
        tool.shift_remove("strict");
    }
    tool
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
        // A forced call opens the answer; one tool makes "required" that tool.
        let named = json!({"type": "function", "function": {"name": "now"}});
        for forced in [json!("required"), named] {
            let policy = policy(qwen, &tools, Some(forced)).unwrap();
            assert!(policy.is_forced());
            assert_eq!(
                policy.plan.unwrap().response_prefix().as_deref(),
                Some("<tool_call>\n<function=now>\n")
            );
        }
        let missing = json!({"type": "function", "function": {"name": "later"}});
        assert_eq!(
            refused(qwen, &tools, Some(missing)).param,
            Some("tool_choice")
        );
        assert_eq!(
            refused(qwen, &[], Some(json!("required"))).param,
            Some("tool_choice")
        );
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
    fn strict_is_kept_out_of_the_prompt() {
        let tools = [
            json!({"type": "function", "function": {"name": "now", "strict": true, "parameters": {}}}),
        ];
        let policy =
            resolve_tool_policy(ModelVariant::Qwen359BGguf, Some(&tools), None, None).unwrap();
        assert_eq!(
            policy.tools,
            [json!({"type": "function", "function": {"name": "now", "parameters": {}}})]
        );
    }

    #[test]
    fn strict_tools_and_forced_calls_are_constrained_where_the_model_can() {
        let strict = [
            json!({"type": "function", "function": {"name": "now", "strict": true,
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}}}}),
        ];
        let plain = [json!({"type": "function", "function": {"name": "now"}})];
        let policy = |variant, tools: &[Value], choice: Option<Value>| {
            resolve_tool_policy(variant, Some(tools), choice.as_ref(), None)
        };
        let qwen = ModelVariant::Qwen359BGguf;
        // Unforced and not strict: the model is free.
        assert_eq!(policy(qwen, &plain, None).unwrap().constraint, None);
        let constraint = policy(qwen, &strict, None).unwrap().constraint.unwrap();
        assert!(constraint.strict && constraint.forced.is_none());
        let forced = policy(qwen, &plain, Some(json!("required"))).unwrap();
        let constraint = forced.constraint.unwrap();
        assert!(!constraint.strict && constraint.forced.is_some());

        // Qwen3.8's sampler has no grammar hook: a forced call is opened by
        // the prompt alone, and `strict` is refused.
        let qwen38 = ModelVariant::Qwen3827BFp8;
        let forced = policy(qwen38, &plain, Some(json!("required"))).unwrap();
        assert!(forced.is_forced() && forced.constraint.is_none());
        let error = policy(qwen38, &strict, None).expect_err("refused");
        assert_eq!(error.detail.unwrap().code, Some("unsupported_parameter"));

        // A schema the grammar cannot express is refused for a strict tool.
        let broken = [
            json!({"type": "function", "function": {"name": "now", "strict": true,
            "parameters": {"type": "object", "properties": {"n": {"type": "integer", "if": {}}}}}}),
        ];
        let error = policy(qwen, &broken, None).expect_err("refused");
        assert_eq!(error.detail.unwrap().param, Some("tools"));
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

//! A request's `tool_choice` and `parallel_tool_calls`, and what they mean
//! for the prompt, the generation and the parser.

use serde_json::Value;

use super::dialect::ToolDialect;
use super::json::tool_function;
use super::parse::ToolCallParser;

/// OpenAI `tool_choice`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The tools are neither offered nor parsed.
    None,
    /// The answer must call at least one tool.
    Required,
    /// The answer must call this tool.
    Named(String),
}

impl ToolChoice {
    /// Read `tool_choice`: `"auto"` (also when absent), `"none"`,
    /// `"required"`, or a named function in the chat completions form
    /// (`{"type": "function", "function": {"name": …}}`) or the Responses
    /// form (`{"type": "function", "name": …}`). `Err` for anything else.
    pub fn parse(value: Option<&Value>) -> Result<Self, &'static str> {
        const INVALID: &str =
            "`tool_choice` must be \"none\", \"auto\", \"required\" or a named function";
        match value {
            None | Some(Value::Null) => Ok(Self::Auto),
            Some(Value::String(choice)) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                _ => Err(INVALID),
            },
            Some(choice) => {
                if choice.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(INVALID);
                }
                choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .and_then(Value::as_str)
                    .filter(|name| !name.trim().is_empty())
                    .map(|name| Self::Named(name.to_string()))
                    .ok_or(INVALID)
            }
        }
    }

    /// Whether the answer must make a call.
    pub fn is_forced(&self) -> bool {
        matches!(self, Self::Required | Self::Named(_))
    }
}

/// How one request uses its tools: the definitions in the prompt, the
/// family's call format, and the limits `tool_choice` and
/// `parallel_tool_calls` set.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolPlan {
    pub dialect: ToolDialect,
    /// The tool definitions rendered into the prompt.
    pub tools: Vec<Value>,
    /// `Auto`, `Required` or `Named`; a `None` choice makes no plan.
    pub choice: ToolChoice,
    /// `parallel_tool_calls`: `false` keeps the answer to one call.
    pub parallel: bool,
}

impl ToolPlan {
    /// The plan for `tools` under `choice`. `Required` with a single tool is
    /// that tool by name. `Err` when a named tool is not in `tools`.
    pub fn new(
        dialect: ToolDialect,
        tools: Vec<Value>,
        choice: ToolChoice,
        parallel: bool,
    ) -> Result<Self, String> {
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool_function(tool).get("name").and_then(Value::as_str))
            .collect();
        let choice = match choice {
            ToolChoice::Named(name) if !names.contains(&name.as_str()) => {
                return Err(format!(
                    "`tool_choice` names the function {name:?}, which is not in `tools`"
                ));
            }
            ToolChoice::Required if names.len() == 1 => ToolChoice::Named(names[0].to_string()),
            choice => choice,
        };
        Ok(Self {
            dialect,
            tools,
            choice,
            parallel,
        })
    }

    /// The start of the assistant turn the prompt writes for the model when
    /// the answer must make a call: the call's opening markup, up to and
    /// including the function name when it is named. The model continues
    /// from there, so a call is certain. `None` when the model is free.
    pub fn response_prefix(&self) -> Option<String> {
        let name = match &self.choice {
            ToolChoice::Required => None,
            ToolChoice::Named(name) => Some(name.as_str()),
            ToolChoice::Auto | ToolChoice::None => return None,
        };
        Some(match (self.dialect, name) {
            (ToolDialect::QwenXml | ToolDialect::Lfm3Xml, None) => "<tool_call>\n<function=".into(),
            (ToolDialect::QwenXml | ToolDialect::Lfm3Xml, Some(name)) => {
                format!("<tool_call>\n<function={name}>\n")
            }
            (ToolDialect::HermesJson, None) => "<tool_call>\n{\"name\": \"".into(),
            (ToolDialect::HermesJson, Some(name)) => {
                format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": ")
            }
            (ToolDialect::LfmPythonic, None) => "<|tool_call_start|>[".into(),
            (ToolDialect::LfmPythonic, Some(name)) => format!("<|tool_call_start|>[{name}("),
        })
    }

    /// The most calls the answer may make.
    pub fn max_calls(&self) -> Option<usize> {
        (!self.parallel).then_some(1)
    }

    /// A stop sequence that ends generation when the only call the answer
    /// may make closes, so no tokens are spent on calls that would be
    /// dropped.
    pub fn stop_sequence(&self) -> Option<&'static str> {
        (!self.parallel).then(|| self.dialect.call_closer())
    }

    /// The parser for this request's answer.
    pub fn parser(&self) -> ToolCallParser {
        let parser = ToolCallParser::new(self.dialect)
            .with_tools(&self.tools)
            .with_max_calls(self.max_calls());
        match self.response_prefix() {
            Some(prefix) => parser.with_prefill(&prefix),
            None => parser,
        }
    }
}

/// A plan that only parses `dialect`: no declared tools, no limits.
impl From<ToolDialect> for ToolPlan {
    fn from(dialect: ToolDialect) -> Self {
        Self {
            dialect,
            tools: Vec::new(),
            choice: ToolChoice::Auto,
            parallel: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "get_weather"}}),
            json!({"type": "function", "function": {"name": "get_time"}}),
        ]
    }

    #[test]
    fn tool_choice_reads_every_openai_form() {
        assert_eq!(ToolChoice::parse(None), Ok(ToolChoice::Auto));
        assert_eq!(
            ToolChoice::parse(Some(&json!("auto"))),
            Ok(ToolChoice::Auto)
        );
        assert_eq!(
            ToolChoice::parse(Some(&json!("none"))),
            Ok(ToolChoice::None)
        );
        assert_eq!(
            ToolChoice::parse(Some(&json!("required"))),
            Ok(ToolChoice::Required)
        );
        for named in [
            json!({"type": "function", "function": {"name": "get_time"}}),
            json!({"type": "function", "name": "get_time"}),
        ] {
            assert_eq!(
                ToolChoice::parse(Some(&named)),
                Ok(ToolChoice::Named("get_time".into()))
            );
        }
        for invalid in [json!("sometimes"), json!({"type": "function"}), json!(3)] {
            assert!(ToolChoice::parse(Some(&invalid)).is_err());
        }
    }

    #[test]
    fn a_forced_call_is_opened_in_the_family_format() {
        let plan = |dialect, choice| ToolPlan::new(dialect, tools(), choice, true).unwrap();
        let named = || ToolChoice::Named("get_time".into());
        assert_eq!(
            plan(ToolDialect::QwenXml, ToolChoice::Auto).response_prefix(),
            None
        );
        assert_eq!(
            plan(ToolDialect::QwenXml, ToolChoice::Required)
                .response_prefix()
                .unwrap(),
            "<tool_call>\n<function="
        );
        assert_eq!(
            plan(ToolDialect::Lfm3Xml, named())
                .response_prefix()
                .unwrap(),
            "<tool_call>\n<function=get_time>\n"
        );
        assert_eq!(
            plan(ToolDialect::HermesJson, ToolChoice::Required)
                .response_prefix()
                .unwrap(),
            "<tool_call>\n{\"name\": \""
        );
        assert_eq!(
            plan(ToolDialect::HermesJson, named())
                .response_prefix()
                .unwrap(),
            "<tool_call>\n{\"name\": \"get_time\", \"arguments\": "
        );
        assert_eq!(
            plan(ToolDialect::LfmPythonic, ToolChoice::Required)
                .response_prefix()
                .unwrap(),
            "<|tool_call_start|>["
        );
        assert_eq!(
            plan(ToolDialect::LfmPythonic, named())
                .response_prefix()
                .unwrap(),
            "<|tool_call_start|>[get_time("
        );
    }

    #[test]
    fn required_with_one_tool_names_it_and_unknown_names_are_refused() {
        let one = vec![json!({"type": "function", "function": {"name": "now"}})];
        let plan = ToolPlan::new(ToolDialect::QwenXml, one, ToolChoice::Required, true).unwrap();
        assert_eq!(plan.choice, ToolChoice::Named("now".into()));
        assert!(ToolPlan::new(
            ToolDialect::QwenXml,
            tools(),
            ToolChoice::Named("missing".into()),
            true
        )
        .is_err());
    }

    #[test]
    fn a_forced_answer_parses_from_the_prefix_the_prompt_wrote() {
        let named = |dialect| {
            ToolPlan::new(dialect, tools(), ToolChoice::Named("get_time".into()), true).unwrap()
        };
        for (dialect, continuation) in [
            (
                ToolDialect::QwenXml,
                "<parameter=city>\nHarare\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                ToolDialect::HermesJson,
                "{\"city\": \"Harare\"}}\n</tool_call>",
            ),
            (ToolDialect::LfmPythonic, "city='Harare')]<|tool_call_end|>"),
        ] {
            let (content, calls) =
                super::super::parse_answer(named(dialect).parser(), continuation);
            assert_eq!(content, None, "{dialect:?}");
            assert_eq!(calls.len(), 1, "{dialect:?}");
            assert_eq!(calls[0].function.name, "get_time");
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                json!({"city": "Harare"})
            );
        }
    }

    #[test]
    fn one_call_at_most_when_parallel_calls_are_off() {
        let plan = ToolPlan::new(ToolDialect::QwenXml, tools(), ToolChoice::Auto, false).unwrap();
        assert_eq!(plan.stop_sequence(), Some("</tool_call>"));
        let text = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call> trailing";
        let (content, calls) = super::super::parse_answer(plan.parser(), text);
        assert_eq!((content, calls.len()), (None, 1));
        // The stop sequence removes the close tag; the call still stands.
        let (_, calls) = super::super::parse_answer(
            plan.parser(),
            "<tool_call>\n<function=get_time>\n</function>\n",
        );
        assert_eq!(calls.len(), 1);

        let plan =
            ToolPlan::new(ToolDialect::LfmPythonic, tools(), ToolChoice::Auto, false).unwrap();
        let (_, calls) = super::super::parse_answer(
            plan.parser(),
            "<|tool_call_start|>[get_time(), get_weather()]<|tool_call_end|>",
        );
        assert_eq!(calls.len(), 1);
    }
}

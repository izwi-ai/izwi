//! Grammars that hold a model's tool calls to the request's tool
//! definitions: only declared names, and arguments that satisfy each tool's
//! parameter schema, in the family's own call markup.
//!
//! The grammars are text in llguidance's Lark dialect; the engine compiles
//! them against the model's tokenizer and masks the sampler with them
//! (`izwi-core` `models::shared::tool_constraint`). A grammar describes the
//! text of one call *after* a given point: after the call's opening marker
//! (a call the model chose to make), or after the prefix the prompt wrote
//! for a forced call.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::choice::{ToolChoice, ToolPlan};
use super::dialect::ToolDialect;
use super::json::tool_function;

/// The most parameters a tool may have for its arguments to be constrained
/// one by one; a larger tool's arguments are left to the model.
const MAX_CONSTRAINED_PARAMETERS: usize = 48;

/// How the sampler constrains one request's tool calls.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolConstraint {
    /// The marker that opens a call. It is one token in every tool-calling
    /// vocabulary; sampling it puts the sampler under `call`.
    pub opener: String,
    /// Grammar of a call's text after `opener`, through its closing marker.
    pub call: String,
    /// Grammar of the answer's first call after the prefix the prompt wrote
    /// (`ChatRequestConfig::response_prefix`), when the answer must make a
    /// call. The sampler starts under it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forced: Option<String>,
    /// The most calls the answer may make; after the last one only the end
    /// of the turn can be sampled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_calls: Option<u32>,
    /// A tool asked for `strict` arguments, so the request must not run
    /// where the constraint cannot be applied. Without it the constraint
    /// only strengthens a forced call, and may be dropped.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

impl ToolPlan {
    /// The constraint for this plan: `Some` when the answer must make a call
    /// (`tool_choice` required or a named function), or when a tool asked
    /// for `strict` arguments (`strict`). Otherwise the model is left free,
    /// as it is on other servers, and its output is parsed leniently.
    pub fn constraint(&self, strict: bool) -> Option<ToolConstraint> {
        if !strict && !self.choice.is_forced() {
            return None;
        }
        let tools: Vec<Tool<'_>> = self.tools.iter().filter_map(Tool::new).collect();
        if tools.is_empty() {
            return None;
        }
        let named = match &self.choice {
            ToolChoice::Named(name) => tools.iter().position(|tool| tool.name == name),
            _ => None,
        };
        let forced = match (&self.choice, named) {
            (ToolChoice::Named(_), Some(index)) => Some(grammar(
                self.dialect,
                &tools,
                Start::AfterName(index),
                self.parallel,
            )),
            (ToolChoice::Required, _) => Some(grammar(
                self.dialect,
                &tools,
                Start::AfterCallOpening,
                self.parallel,
            )),
            _ => None,
        };
        Some(ToolConstraint {
            opener: self.dialect.call_opener().to_string(),
            call: grammar(self.dialect, &tools, Start::AfterOpener, self.parallel),
            forced,
            max_calls: self.max_calls().map(|max| max as u32),
            strict,
        })
    }
}

/// One declared tool: its name and parameter schema.
struct Tool<'a> {
    name: &'a str,
    /// The `parameters` schema, when it is an object.
    parameters: Option<&'a Map<String, Value>>,
}

impl<'a> Tool<'a> {
    fn new(tool: &'a Value) -> Option<Self> {
        let function = tool_function(tool);
        Some(Self {
            name: function.get("name")?.as_str()?,
            parameters: function.get("parameters").and_then(Value::as_object),
        })
    }

    /// The declared parameters in schema order as `(name, schema, required)`,
    /// or `None` when the arguments cannot be listed one by one (no
    /// `properties`, too many of them, or names the markup cannot carry).
    fn listed_parameters(&self) -> Option<Vec<(&'a str, &'a Value, bool)>> {
        let parameters = self.parameters?;
        let properties = parameters.get("properties")?.as_object()?;
        if properties.len() > MAX_CONSTRAINED_PARAMETERS
            || properties.keys().any(|name| !is_plain_name(name))
        {
            return None;
        }
        let required: Vec<&str> = parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        Some(
            properties
                .iter()
                .map(|(name, schema)| (name.as_str(), schema, required.contains(&name.as_str())))
                .collect(),
        )
    }
}

/// A name that can stand in `<parameter=NAME>` and `NAME=value` unquoted.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Where in a call a grammar starts.
#[derive(Clone, Copy)]
enum Start {
    /// Right after the opening marker (`<tool_call>`, `<|tool_call_start|>`).
    AfterOpener,
    /// After the prefix of a required call, up to where the name goes.
    AfterCallOpening,
    /// After the prefix of a call to the tool at this index, up to where its
    /// arguments go.
    AfterName(usize),
}

fn grammar(dialect: ToolDialect, tools: &[Tool<'_>], start: Start, parallel: bool) -> String {
    let mut rules = Rules::default();
    let start_rule = match dialect {
        ToolDialect::QwenXml | ToolDialect::Lfm3Xml => xml_start(&mut rules, tools, start),
        ToolDialect::HermesJson => hermes_start(&mut rules, tools, start),
        ToolDialect::LfmPythonic => pythonic_start(&mut rules, tools, start, parallel),
    };
    let mut out = format!("start: {start_rule}\n");
    out.push_str(&rules.text);
    out
}

/// The rules of one grammar, with the shared terminals added once.
#[derive(Default)]
struct Rules {
    text: String,
    shared: Vec<&'static str>,
    next_json: usize,
}

impl Rules {
    fn push(&mut self, name: &str, body: &str) {
        self.text.push_str(&format!("{name}: {body}\n"));
    }

    /// A rule shared by every tool, written the first time it is used.
    fn shared(&mut self, name: &'static str, definition: &'static str) -> &'static str {
        if !self.shared.contains(&name) {
            self.shared.push(name);
            self.text.push_str(definition);
            self.text.push('\n');
        }
        name
    }

    /// A rule for one JSON value satisfying `schema`.
    fn json(&mut self, schema: &Value) -> String {
        let name = format!("json_{}", self.next_json);
        self.next_json += 1;
        self.push(&name, &format!("%json {schema}"));
        name
    }
}

/// A Lark string literal for `text`.
fn literal(text: &str) -> String {
    Value::String(text.to_string()).to_string()
}

/// How one parameter's value is written, by its declared type.
enum ValueKind<'a> {
    /// A `string` parameter: free text.
    Text,
    /// A string `enum`: one of these.
    Choice(Vec<&'a str>),
    /// Anything else: JSON satisfying the schema.
    Json(&'a Value),
}

fn value_kind(schema: &Value) -> ValueKind<'_> {
    if let Some(choices) = schema.get("enum").and_then(Value::as_array) {
        let strings: Option<Vec<&str>> = choices.iter().map(Value::as_str).collect();
        if let Some(choices) = strings.filter(|choices| !choices.is_empty()) {
            return ValueKind::Choice(choices);
        }
    }
    match schema.get("type") {
        Some(Value::String(kind)) if kind == "string" => {
            // A string with further keywords (`pattern`, `format`,
            // `maxLength`) is still written as bare text in these formats,
            // where JSON string rules do not apply.
            ValueKind::Text
        }
        Some(Value::String(_)) => ValueKind::Json(schema),
        // No single type: the model writes it as it sees fit.
        _ => ValueKind::Text,
    }
}

// --- Qwen3.5 / LFM3: <function=NAME><parameter=K>V</parameter></function> ---

const XML_TEXT_VALUE: &str = "xml_text[lazy]: /(.|\\n)*\\n<\\/parameter>\\n/";
const XML_ANY_PARAMETERS: &str = "xml_any: (\"<parameter=\" /[^>\\n]+/ \">\\n\" xml_text)*";

fn xml_start(rules: &mut Rules, tools: &[Tool<'_>], start: Start) -> String {
    const END: &str = "\"</function>\\n</tool_call>\"";
    let bodies: Vec<String> = tools
        .iter()
        .enumerate()
        .map(|(index, tool)| xml_body(rules, index, tool))
        .collect();
    if let Start::AfterName(index) = start {
        return format!("{} {END}", bodies[index]);
    }
    let alternatives: Vec<String> = tools
        .iter()
        .zip(&bodies)
        .map(|(tool, body)| format!("{} {body}", literal(&format!("{}>\n", tool.name))))
        .collect();
    rules.push("xml_function", &alternatives.join(" | "));
    match start {
        Start::AfterOpener => format!("\"\\n<function=\" xml_function {END}"),
        _ => format!("xml_function {END}"),
    }
}

/// The parameters of one XML call, in schema order: a required one must
/// appear, an optional one may.
fn xml_body(rules: &mut Rules, index: usize, tool: &Tool<'_>) -> String {
    let Some(parameters) = tool.listed_parameters() else {
        rules.shared("xml_text", XML_TEXT_VALUE);
        return rules.shared("xml_any", XML_ANY_PARAMETERS).to_string();
    };
    let mut body = Vec::new();
    for (position, (name, schema, required)) in parameters.into_iter().enumerate() {
        let open = literal(&format!("<parameter={name}>\n"));
        let value = match value_kind(schema) {
            ValueKind::Text => rules.shared("xml_text", XML_TEXT_VALUE).to_string(),
            ValueKind::Choice(choices) => format!(
                "({}) \"\\n</parameter>\\n\"",
                choices
                    .iter()
                    .map(|c| literal(c))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            ValueKind::Json(schema) => format!("{} \"\\n</parameter>\\n\"", rules.json(schema)),
        };
        let rule = format!("xml_p_{index}_{position}");
        rules.push(&rule, &format!("{open} {value}"));
        body.push(if required { rule } else { format!("{rule}?") });
    }
    let rule = format!("xml_body_{index}");
    rules.push(
        &rule,
        &if body.is_empty() {
            "\"\"".to_string()
        } else {
            body.join(" ")
        },
    );
    rule
}

// --- Qwen3: {"name": NAME, "arguments": {...}} ---

fn hermes_start(rules: &mut Rules, tools: &[Tool<'_>], start: Start) -> String {
    const END: &str = "\"}\" /\\n?/ \"</tool_call>\"";
    let arguments: Vec<String> = tools
        .iter()
        .map(|tool| {
            let schema = match tool.parameters {
                Some(parameters) => Value::Object(parameters.clone()),
                None => serde_json::json!({"type": "object"}),
            };
            rules.json(&schema)
        })
        .collect();
    if let Start::AfterName(index) = start {
        return format!("{} {END}", arguments[index]);
    }
    let alternatives: Vec<String> = tools
        .iter()
        .zip(&arguments)
        .map(|(tool, arguments)| {
            format!(
                "{} {arguments}",
                literal(&format!("{}\", \"arguments\": ", tool.name))
            )
        })
        .collect();
    rules.push("hermes_call", &alternatives.join(" | "));
    match start {
        Start::AfterOpener => format!("/\\n?/ \"{{\\\"name\\\": \\\"\" hermes_call {END}"),
        _ => format!("hermes_call {END}"),
    }
}

// --- LFM2.5: [NAME(k='v', n=1), OTHER()] ---

const PY_STRING: &str = "PY_STRING: /'([^'\\\\\\n]|\\\\.)*'/ | /\"([^\"\\\\\\n]|\\\\.)*\"/";
const PY_ANY_ARGUMENTS: &str = "PY_ANY: /[^()\\n]*/";

fn pythonic_start(rules: &mut Rules, tools: &[Tool<'_>], start: Start, parallel: bool) -> String {
    const END: &str = "\"]<|tool_call_end|>\"";
    let arguments: Vec<String> = tools
        .iter()
        .enumerate()
        .map(|(index, tool)| pythonic_arguments(rules, index, tool))
        .collect();
    let alternatives: Vec<String> = tools
        .iter()
        .zip(&arguments)
        .map(|(tool, arguments)| {
            format!("{} {arguments} \")\"", literal(&format!("{}(", tool.name)))
        })
        .collect();
    rules.push("py_call", &alternatives.join(" | "));
    let more = if parallel { " (\", \" py_call)*" } else { "" };
    match start {
        Start::AfterOpener => format!("\"[\" py_call{more} {END}"),
        Start::AfterCallOpening => format!("py_call{more} {END}"),
        Start::AfterName(index) => format!("{} \")\"{more} {END}", arguments[index]),
    }
}

/// The arguments of one pythonic call as `k=v` pairs separated by `", "`, in
/// schema order: a required one must appear, an optional one may.
fn pythonic_arguments(rules: &mut Rules, index: usize, tool: &Tool<'_>) -> String {
    let Some(parameters) = tool.listed_parameters() else {
        return rules.shared("PY_ANY", PY_ANY_ARGUMENTS).to_string();
    };
    let mut pairs = Vec::new();
    for (position, (name, schema, required)) in parameters.into_iter().enumerate() {
        let value = match value_kind(schema) {
            ValueKind::Text => rules.shared("PY_STRING", PY_STRING).to_string(),
            ValueKind::Choice(choices) => format!(
                "({})",
                choices
                    .iter()
                    .map(|c| literal(&format!(
                        "'{}'",
                        c.replace('\\', "\\\\").replace('\'', "\\'")
                    )))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            ValueKind::Json(schema) => rules.json(schema),
        };
        let rule = format!("py_p_{index}_{position}");
        rules.push(&rule, &format!("{} {value}", literal(&format!("{name}="))));
        pairs.push((rule, required));
    }
    // What may follow an argument already written: each later one after a
    // separator.
    let tail = |from: usize| -> String {
        pairs[from..]
            .iter()
            .map(|(rule, required)| {
                if *required {
                    format!(" \", \" {rule}")
                } else {
                    format!(" (\", \" {rule})?")
                }
            })
            .collect()
    };
    // The first argument written is the first required one, or any optional
    // one before it.
    let mut alternatives = Vec::new();
    for (position, (rule, required)) in pairs.iter().enumerate() {
        alternatives.push(format!("{rule}{}", tail(position + 1)));
        if *required {
            break;
        }
    }
    if pairs.iter().all(|(_, required)| !required) {
        alternatives.push("\"\"".to_string());
    }
    let rule = format!("py_args_{index}");
    rules.push(&rule, &alternatives.join(" | "));
    rule
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer"}
                },
                "required": ["city"]
            }}}),
            json!({"type": "function", "function": {"name": "now"}}),
        ]
    }

    fn plan(dialect: ToolDialect, choice: ToolChoice, parallel: bool) -> ToolPlan {
        ToolPlan::new(dialect, tools(), choice, parallel).unwrap()
    }

    #[test]
    fn only_forced_or_strict_requests_are_constrained() {
        let auto = plan(ToolDialect::QwenXml, ToolChoice::Auto, true);
        assert_eq!(auto.constraint(false), None);
        let strict = auto.constraint(true).unwrap();
        assert_eq!(strict.opener, "<tool_call>");
        assert_eq!(strict.forced, None);
        assert_eq!(strict.max_calls, None);
        let required = plan(ToolDialect::QwenXml, ToolChoice::Required, false)
            .constraint(false)
            .unwrap();
        assert!(required.forced.is_some());
        assert_eq!(required.max_calls, Some(1));
    }

    #[test]
    fn xml_grammar_lists_names_and_typed_parameters() {
        let constraint = plan(
            ToolDialect::QwenXml,
            ToolChoice::Named("get_weather".into()),
            true,
        )
        .constraint(false)
        .unwrap();
        // A model-opened call chooses among the declared names.
        assert!(constraint
            .call
            .starts_with("start: \"\\n<function=\" xml_function \"</function>\\n</tool_call>\"\n"));
        assert!(constraint
            .call
            .contains("xml_function: \"get_weather>\\n\" xml_body_0 | \"now>\\n\" xml_any"));
        // Required first, optional after, each by its type.
        assert!(constraint
            .call
            .contains("xml_body_0: xml_p_0_0 xml_p_0_1? xml_p_0_2?"));
        assert!(constraint
            .call
            .contains("xml_p_0_0: \"<parameter=city>\\n\" xml_text"));
        assert!(constraint.call.contains(
            "xml_p_0_1: \"<parameter=unit>\\n\" (\"celsius\" | \"fahrenheit\") \"\\n</parameter>\\n\""
        ));
        assert!(constraint
            .call
            .contains("json_0: %json {\"type\":\"integer\"}"));
        // A tool that declares no parameters takes any.
        assert!(constraint.call.contains("xml_any: ("));
        // The forced call continues after the name the prompt wrote.
        assert!(constraint
            .forced
            .unwrap()
            .starts_with("start: xml_body_0 \"</function>\\n</tool_call>\"\n"));
    }

    #[test]
    fn hermes_grammar_wraps_each_tools_schema() {
        let required = plan(ToolDialect::HermesJson, ToolChoice::Required, true)
            .constraint(false)
            .unwrap();
        assert!(required
            .call
            .starts_with("start: /\\n?/ \"{\\\"name\\\": \\\"\" hermes_call"));
        assert!(required
            .call
            .contains("hermes_call: \"get_weather\\\", \\\"arguments\\\": \" json_0 | \"now\\\", \\\"arguments\\\": \" json_1"));
        assert!(required
            .forced
            .unwrap()
            .starts_with("start: hermes_call \"}\""));
        let named = plan(
            ToolDialect::HermesJson,
            ToolChoice::Named("now".into()),
            true,
        )
        .constraint(false)
        .unwrap();
        assert!(named.forced.unwrap().starts_with("start: json_1 \"}\""));
    }

    #[test]
    fn pythonic_grammar_orders_arguments_and_limits_calls() {
        let one = plan(ToolDialect::LfmPythonic, ToolChoice::Required, false)
            .constraint(false)
            .unwrap();
        assert!(one
            .call
            .starts_with("start: \"[\" py_call \"]<|tool_call_end|>\"\n"));
        let many = plan(
            ToolDialect::LfmPythonic,
            ToolChoice::Named("get_weather".into()),
            true,
        )
        .constraint(false)
        .unwrap();
        assert!(many
            .call
            .starts_with("start: \"[\" py_call (\", \" py_call)* \"]<|tool_call_end|>\"\n"));
        // city is required and first; unit and days may follow.
        assert!(many
            .call
            .contains("py_args_0: py_p_0_0 (\", \" py_p_0_1)? (\", \" py_p_0_2)?\n"));
        assert!(many.call.contains("py_p_0_0: \"city=\" PY_STRING"));
        assert!(many
            .forced
            .unwrap()
            .starts_with("start: py_args_0 \")\" (\", \" py_call)* \"]<|tool_call_end|>\"\n"));
    }
}

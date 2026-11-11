//! What a request's tool definitions say about names and parameter types,
//! for the parser: which tools exist, and how to read an argument a model
//! wrote as bare text.

use serde_json::{Map, Value};

use super::json::tool_function;

/// The declared tools of one request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolSchemas {
    /// `(name, parameters.properties)` per tool, in request order.
    tools: Vec<(String, Map<String, Value>)>,
}

impl ToolSchemas {
    pub fn new(tools: &[Value]) -> Self {
        let tools = tools
            .iter()
            .filter_map(|tool| {
                let function = tool_function(tool);
                let name = function.get("name")?.as_str()?.to_string();
                let properties = function
                    .pointer("/parameters/properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                Some((name, properties))
            })
            .collect();
        Self { tools }
    }

    /// No tools are known, so nothing can be checked against them.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Whether `name` is a declared tool. With no tools known, every name
    /// passes.
    pub fn declares(&self, name: &str) -> bool {
        self.is_empty() || self.tools.iter().any(|(tool, _)| tool == name)
    }

    fn property(&self, tool: &str, parameter: &str) -> Option<&Value> {
        self.tools
            .iter()
            .rfind(|(name, _)| name == tool)
            .and_then(|(_, properties)| properties.get(parameter))
    }

    /// The value of an argument written as bare text (the XML formats'
    /// `<parameter=k>` body), read by the parameter's declared type: a
    /// `string` parameter keeps its text even when the text is `123`, `true`
    /// or `null`; any other declared type is read as JSON (a boolean or null
    /// also in Python's spelling). Only an undeclared or untyped parameter is
    /// guessed (JSON when it parses, else text).
    pub fn text_argument(&self, tool: &str, parameter: &str, raw: &str) -> Value {
        let parsed = || serde_json::from_str::<Value>(raw).ok();
        let text = || Value::String(raw.to_string());
        let Some(kinds) = self.property(tool, parameter).and_then(declared_types) else {
            return parsed().unwrap_or_else(text);
        };
        let declares = |kind: &str| kinds.iter().any(|declared| declared == kind);
        // Python literals: Qwen3.5's own template writes booleans and null
        // this way in history, so the model writes them too.
        match raw {
            "null" | "None" if declares("null") => return Value::Null,
            "True" if declares("boolean") => return Value::Bool(true),
            "False" if declares("boolean") => return Value::Bool(false),
            _ => {}
        }
        // The JSON reading wins when it has one of the declared non-string
        // types; otherwise the text is the value when a string is allowed.
        let json_value = parsed();
        if let Some(value) = &json_value {
            if kinds
                .iter()
                .any(|kind| kind != "string" && has_type(value, kind))
            {
                return value.clone();
            }
        }
        if kinds.iter().any(|kind| kind == "string") {
            return text();
        }
        // Not the declared type: hand the client what the model wrote.
        json_value.unwrap_or_else(text)
    }
}

/// The JSON types a property schema declares: its `type` (a name or a list),
/// or `string` for an all-string `enum`. `None` when it declares none.
fn declared_types(schema: &Value) -> Option<Vec<String>> {
    match schema.get("type") {
        Some(Value::String(kind)) => return Some(vec![kind.clone()]),
        Some(Value::Array(kinds)) => {
            let kinds: Vec<String> = kinds
                .iter()
                .filter_map(|kind| kind.as_str().map(str::to_owned))
                .collect();
            if !kinds.is_empty() {
                return Some(kinds);
            }
        }
        _ => {}
    }
    let choices = schema.get("enum")?.as_array()?;
    (!choices.is_empty() && choices.iter().all(Value::is_string))
        .then(|| vec!["string".to_string()])
}

fn has_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schemas() -> ToolSchemas {
        ToolSchemas::new(&[json!({"type": "function", "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "on": {"type": "boolean"},
                "tags": {"type": "array"},
                "due": {"type": ["string", "null"]},
                "size": {"type": ["string", "integer"]},
                "unit": {"enum": ["c", "f"]},
                "any": {"description": "untyped"}
            }}
        }})])
    }

    #[test]
    fn string_parameters_keep_text_that_looks_like_json() {
        let schemas = schemas();
        for raw in [
            "00123",
            "4521",
            "true",
            "[1]",
            "{\"a\": 1}",
            "\"quoted\"",
            "plain",
        ] {
            assert_eq!(schemas.text_argument("f", "id", raw), json!(raw), "{raw}");
        }
        assert_eq!(schemas.text_argument("f", "id", "null"), json!("null"));
        assert_eq!(schemas.text_argument("f", "unit", "c"), json!("c"));
    }

    #[test]
    fn typed_parameters_are_read_as_json() {
        let schemas = schemas();
        assert_eq!(schemas.text_argument("f", "count", "5"), json!(5));
        assert_eq!(schemas.text_argument("f", "on", "true"), json!(true));
        assert_eq!(schemas.text_argument("f", "on", "True"), json!(true));
        assert_eq!(schemas.text_argument("f", "on", "False"), json!(false));
        assert_eq!(schemas.text_argument("f", "due", "None"), Value::Null);
        // A string parameter keeps the word.
        assert_eq!(schemas.text_argument("f", "id", "True"), json!("True"));
        assert_eq!(schemas.text_argument("f", "tags", "[\"a\"]"), json!(["a"]));
        // Text that is not the declared type stays text for the client to see.
        assert_eq!(schemas.text_argument("f", "count", "five"), json!("five"));
    }

    #[test]
    fn nullable_and_mixed_types_follow_their_declaration() {
        let schemas = schemas();
        assert_eq!(schemas.text_argument("f", "due", "null"), Value::Null);
        assert_eq!(schemas.text_argument("f", "due", "2026"), json!("2026"));
        assert_eq!(schemas.text_argument("f", "size", "12"), json!(12));
        assert_eq!(schemas.text_argument("f", "size", "large"), json!("large"));
    }

    #[test]
    fn undeclared_parameters_and_tools_are_guessed() {
        let schemas = schemas();
        assert_eq!(schemas.text_argument("f", "any", "3"), json!(3));
        assert_eq!(schemas.text_argument("f", "missing", "x y"), json!("x y"));
        assert_eq!(schemas.text_argument("g", "id", "3"), json!(3));
        assert_eq!(
            ToolSchemas::default().text_argument("f", "id", "3"),
            json!(3)
        );
    }

    #[test]
    fn names_are_checked_only_when_tools_are_known() {
        let schemas = schemas();
        assert!(schemas.declares("f"));
        assert!(!schemas.declares("g"));
        assert!(ToolSchemas::default().declares("anything"));
    }
}

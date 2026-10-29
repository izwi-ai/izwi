use super::*;

/// Tool turns rendered by the checkpoints' own `chat_template.jinja` (see
/// the file's `source`), with keys sorted as `serde_json` writes them.
const FIXTURES: &str = include_str!("tool_response_fixtures.json");

fn result(message: &Value) -> ToolResult<'_> {
    ToolResult {
        content: message.get("content"),
        additional_tools: message.get("additional_tools"),
        tool_call_id: message.get("tool_call_id").and_then(Value::as_str),
    }
}

fn render_all(
    dialect: ToolResponseDialect,
    tools: &[Value],
    messages: &[Value],
) -> Result<Vec<String>, ApiError> {
    let mut responses = ToolResponses::new(dialect, tools);
    let mut turns = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let next = messages.get(index + 1).map(result);
        if let Some(turn) = responses.render(result(message), next)? {
            turns.push(turn);
        }
    }
    Ok(turns)
}

#[test]
fn tool_turns_match_the_checkpoint_templates() {
    let fixtures: Value = serde_json::from_str(FIXTURES).unwrap();
    let cases = fixtures["cases"].as_array().unwrap();
    assert!(cases.len() >= 20);
    for case in cases {
        let dialect = match case["dialect"].as_str().unwrap() {
            "lfm3_chat" => ToolResponseDialect::Lfm3Chat,
            "lfm3_audio" => ToolResponseDialect::Lfm3Audio,
            other => panic!("unknown dialect {other}"),
        };
        let name = case["name"].as_str().unwrap();
        let turns = render_all(
            dialect,
            case["tools"].as_array().unwrap(),
            case["results"].as_array().unwrap(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let expected = case["turns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|turn| turn.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(turns, expected, "{dialect:?} {name}");
    }
}

#[test]
fn an_unknown_tool_reference_is_refused_as_the_template_does() {
    let error = render_all(
        ToolResponseDialect::Lfm3Chat,
        &[json!({"type": "function", "function": {"name": "search_tools"}})],
        &[json!({"role": "tool", "content": [{"type": "tool_reference", "name": "missing"}]})],
    )
    .unwrap_err();
    assert!(error.message.contains("Tool reference 'missing' not found"));
}

#[test]
fn lfm3_audio_points_tool_references_at_additional_tools() {
    let error = render_all(
        ToolResponseDialect::Lfm3Audio,
        &[json!({"type": "function", "function": {"name": "get_weather"}})],
        &[json!({"role": "tool", "content": [{"type": "tool_reference", "name": "get_weather"}]})],
    )
    .unwrap_err();
    assert!(error.message.contains("additional_tools"));
}

#[test]
fn additional_tools_beside_content_is_refused() {
    let error = render_all(
        ToolResponseDialect::Lfm3Chat,
        &[],
        &[json!({
            "role": "tool",
            "content": "ignored by the template",
            "additional_tools": [{"type": "function", "function": {"name": "a"}}]
        })],
    )
    .unwrap_err();
    assert!(error.message.contains("not both"));
}

#[test]
fn other_families_refuse_deferred_tool_results() {
    let references = json!([{"type": "tool_reference", "name": "a"}]);
    let flag = json!(true);
    for result in [
        ToolResult {
            content: Some(&references),
            ..ToolResult::default()
        },
        ToolResult {
            additional_tools: Some(&flag),
            ..ToolResult::default()
        },
    ] {
        assert!(reject_deferred_tool_results(ModelVariant::Qwen359BGguf, result).is_err());
        assert!(reject_deferred_tool_results(ModelVariant::Lfm317BGguf, result).is_ok());
    }
    let plain = json!("21");
    assert!(reject_deferred_tool_results(
        ModelVariant::Qwen359BGguf,
        ToolResult {
            content: Some(&plain),
            ..ToolResult::default()
        }
    )
    .is_ok());
}

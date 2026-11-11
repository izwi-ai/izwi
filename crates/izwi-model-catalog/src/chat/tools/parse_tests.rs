use super::*;
use serde_json::json;

/// Feed `text` in `chunk`-character deltas; returns the streamed text, the
/// calls assembled from their fragments, and the fragment count.
fn stream_with(
    mut parser: ToolCallParser,
    text: &str,
    chunk: usize,
) -> (String, Vec<ToolCall>, usize) {
    let chars: Vec<char> = text.chars().collect();
    let mut out = ParsedOutput::default();
    let mut fragments = 0;
    for piece in chars.chunks(chunk) {
        let parsed = parser.push(&piece.iter().collect::<String>());
        fragments += parsed.tool_calls.len();
        out.extend(parsed);
    }
    let tail = parser.finish();
    fragments += tail.tool_calls.len();
    out.extend(tail);
    let mut calls: Vec<ToolCall> = Vec::new();
    for delta in out.tool_calls {
        match calls.get_mut(delta.index) {
            Some(call) => {
                assert!(delta.id.is_none() && delta.function.name.is_none());
                call.function.arguments.push_str(&delta.function.arguments);
            }
            None => {
                assert_eq!(delta.index, calls.len(), "indexes are dense");
                assert!(delta
                    .id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("call_")));
                assert_eq!(delta.kind, Some("function"));
                calls.push(ToolCall {
                    id: delta.id.unwrap(),
                    kind: "function",
                    function: ToolCallFunction {
                        name: delta.function.name.expect("first fragment names the call"),
                        arguments: delta.function.arguments,
                    },
                });
            }
        }
    }
    (out.content, calls, fragments)
}

fn stream(dialect: ToolDialect, text: &str, chunk: usize) -> (String, Vec<ToolCall>, usize) {
    stream_with(ToolCallParser::new(dialect), text, chunk)
}

fn arguments(call: &ToolCall) -> Value {
    serde_json::from_str(&call.function.arguments).expect("arguments are JSON")
}

fn whole(dialect: ToolDialect, text: &str) -> (Option<String>, Vec<ToolCall>) {
    parse_answer(ToolCallParser::new(dialect), text)
}

const QWEN: &str = "Checking that now.\n<tool_call>\n<function=get_time>\n<parameter=timezone>\n\"Africa/Harare\"\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";

#[test]
fn qwen_calls_stream_without_markup_at_every_split() {
    for dialect in [ToolDialect::QwenXml, ToolDialect::Lfm3Xml] {
        for chunk in 1..=QWEN.len() {
            let (content, calls, _) = stream(dialect, QWEN, chunk);
            assert_eq!(content, "Checking that now.", "chunk {chunk}");
            assert_eq!(calls.len(), 1, "chunk {chunk}");
            assert_eq!(calls[0].function.name, "get_time");
            assert_eq!(
                arguments(&calls[0]),
                json!({"timezone": "Africa/Harare", "days": 2}),
                "chunk {chunk}"
            );
        }
    }
}

#[test]
fn qwen_arguments_stream_one_parameter_at_a_time() {
    // Character by character: the name, each parameter and the close arrive
    // as separate fragments.
    let (_, calls, fragments) = stream(ToolDialect::QwenXml, QWEN, 1);
    assert_eq!(calls.len(), 1);
    assert_eq!(fragments, 4);
    // The whole answer at once merges them.
    let (_, _, fragments) = stream(ToolDialect::QwenXml, QWEN, QWEN.len());
    assert_eq!(fragments, 1);
}

#[test]
fn qwen_parses_several_calls_json_bodies_and_empty_arguments() {
    let text = "<tool_call>\n<function=now>\n</function>\n</tool_call>\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    for chunk in [1, 3, 7, text.len()] {
        let (content, calls, _) = stream(ToolDialect::QwenXml, text, chunk);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].function.name.as_str(), arguments(&calls[0])),
            ("now", json!({}))
        );
        assert_eq!(
            (calls[1].function.name.as_str(), arguments(&calls[1])),
            ("weather", json!({"city": "Paris"}))
        );
    }
}

const HERMES: &str = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": \"celsius\", \"city\": \"São Paulo\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>";

#[test]
fn hermes_calls_stream_without_markup_at_every_split() {
    for chunk in 1..=HERMES.chars().count() {
        let (content, calls, _) = stream(ToolDialect::HermesJson, HERMES, chunk);
        assert_eq!(content, "Let me check.", "chunk {chunk}");
        assert_eq!(calls.len(), 2, "chunk {chunk}");
        assert_eq!(calls[0].function.name, "get_weather");
        // The arguments keep the order the model wrote them in.
        assert_eq!(
            calls[0].function.arguments,
            r#"{"unit":"celsius","city":"São Paulo"}"#
        );
        assert_eq!(arguments(&calls[1]), json!({}));
    }
}

#[test]
fn a_close_tag_inside_a_json_string_is_part_of_the_argument() {
    let text = "<tool_call>\n{\"name\": \"note\", \"arguments\": {\"body\": \"write </tool_call> twice: </tool_call>\"}}\n</tool_call> done";
    for chunk in 1..=text.len() {
        let (content, calls, _) = stream(ToolDialect::HermesJson, text, chunk);
        assert_eq!(content, " done", "chunk {chunk}");
        assert_eq!(calls.len(), 1, "chunk {chunk}");
        assert_eq!(
            arguments(&calls[0]),
            json!({"body": "write </tool_call> twice: </tool_call>"})
        );
    }
}

#[test]
fn text_that_only_looks_like_a_call_is_released_unchanged() {
    for text in [
        "a < b and <tool_ is not a tag",
        "keep <tool_call>not a call</tool_call> as text",
        "unterminated <tool_call>\n{\"name\": \"f\"",
        "  spaced  \n",
    ] {
        for dialect in [ToolDialect::QwenXml, ToolDialect::HermesJson] {
            for chunk in [1, 2, 5, text.len()] {
                let (content, calls, _) = stream(dialect, text, chunk);
                assert!(calls.is_empty(), "{text:?}");
                assert_eq!(content, text, "chunk {chunk}");
            }
        }
    }
}

#[test]
fn a_call_cut_off_inside_its_parameters_keeps_the_complete_ones() {
    let text = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2";
    let mut parser = ToolCallParser::new(ToolDialect::QwenXml);
    let mut parsed = parser.push(text);
    parsed.extend(parser.finish());
    assert!(parser.truncated());
    let (content, calls) = parsed.into_calls();
    assert_eq!(content, "");
    assert_eq!(arguments(&calls[0]), json!({"a": 1}));
    // A complete answer is not truncated.
    let mut parser = ToolCallParser::new(ToolDialect::QwenXml);
    parser.push(QWEN);
    parser.finish();
    assert!(!parser.truncated());
}

#[test]
fn a_whole_call_without_its_close_tag_still_stands() {
    // A stop sequence on the close tag, or a model that ends its turn early.
    let (_, calls) = whole(
        ToolDialect::HermesJson,
        "<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n",
    );
    assert_eq!(calls.len(), 1);
    let (_, calls) = whole(
        ToolDialect::QwenXml,
        "<tool_call>\n<function=now>\n</function>\n",
    );
    assert_eq!(calls.len(), 1);
    let (content, calls) = whole(ToolDialect::LfmPythonic, "<|tool_call_start|>[now()]");
    assert_eq!((content, calls.len()), (None, 1));
}

const LFM: &str =
    "Checking.<|tool_call_start|>[get_weather(city='Paris', days=2), now()]<|tool_call_end|> Done.";

#[test]
fn lfm_calls_stream_without_markup_at_every_split() {
    for chunk in 1..=LFM.len() {
        let (content, calls, _) = stream(ToolDialect::LfmPythonic, LFM, chunk);
        assert_eq!(content, "Checking. Done.", "chunk {chunk}");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(arguments(&calls[0]), json!({"city": "Paris", "days": 2}));
        assert_eq!(calls[1].function.name, "now");
        assert_eq!(arguments(&calls[1]), json!({}));
    }
}

#[test]
fn malformed_or_unterminated_lfm_blocks_stay_in_the_text() {
    for text in [
        "<|tool_call_start|>[f(a=)]<|tool_call_end|>",
        "<|tool_call_start|>not a list<|tool_call_end|>",
        "partial <|tool_call_start|>[f(a=1",
    ] {
        let (content, calls, _) = stream(ToolDialect::LfmPythonic, text, 3);
        assert!(calls.is_empty(), "{text}");
        assert_eq!(content, text);
    }
}

#[test]
fn whole_answers_parse_like_the_stream() {
    let (content, calls) = whole(ToolDialect::QwenXml, QWEN);
    assert_eq!(content.as_deref(), Some("Checking that now."));
    assert_eq!(
        arguments(&calls[0]),
        json!({"timezone": "Africa/Harare", "days": 2})
    );
    let (content, calls) = whole(
        ToolDialect::LfmPythonic,
        "<|tool_call_start|>[now()]<|tool_call_end|>",
    );
    assert_eq!((content, calls.len()), (None, 1));
    let (content, calls) = whole(ToolDialect::QwenXml, "  plain answer\n");
    assert_eq!(content.as_deref(), Some("  plain answer\n"));
    assert!(calls.is_empty());
}

fn typed_tools() -> Vec<Value> {
    vec![
        json!({"type": "function", "function": {"name": "lookup_order", "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "limit": {"type": "integer"},
                "notes": {"type": "string"}
            }
        }}}),
        json!({"type": "function", "function": {"name": "now"}}),
    ]
}

#[test]
fn bare_text_arguments_are_read_by_their_declared_type() {
    let text = "<tool_call>\n<function=lookup_order>\n<parameter=order_id>\n4521\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=notes>\n  indented\ntrue\n</parameter>\n</function>\n</tool_call>";
    for chunk in [1, 4, text.len()] {
        let parser = ToolCallParser::new(ToolDialect::QwenXml).with_tools(&typed_tools());
        let (_, calls, _) = stream_with(parser, text, chunk);
        assert_eq!(
            arguments(&calls[0]),
            json!({"order_id": "4521", "limit": 5, "notes": "  indented\ntrue"})
        );
    }
    // Without the definitions the parser can only guess.
    let (_, calls) = whole(ToolDialect::QwenXml, text);
    assert_eq!(arguments(&calls[0])["order_id"], json!(4521));
}

#[test]
fn calls_to_undeclared_tools_stay_text() {
    let parser = || ToolCallParser::new(ToolDialect::QwenXml).with_tools(&typed_tools());
    let xml = "<tool_call>\n<function=delete_everything>\n</function>\n</tool_call>";
    for chunk in [1, 6, xml.len()] {
        let (content, calls, _) = stream_with(parser(), xml, chunk);
        assert!(calls.is_empty());
        assert_eq!(content, xml);
    }
    let hermes = "<tool_call>\n{\"name\": \"delete_everything\", \"arguments\": {}}\n</tool_call>";
    let (content, calls, _) = stream_with(
        ToolCallParser::new(ToolDialect::HermesJson).with_tools(&typed_tools()),
        hermes,
        5,
    );
    assert!(calls.is_empty());
    assert_eq!(content, hermes);
    let lfm = "<|tool_call_start|>[now(), delete_everything()]<|tool_call_end|>";
    let (content, calls, _) = stream_with(
        ToolCallParser::new(ToolDialect::LfmPythonic).with_tools(&typed_tools()),
        lfm,
        5,
    );
    assert!(calls.is_empty());
    assert_eq!(content, lfm);
    // A declared tool next to it still parses.
    let (_, calls, _) = stream_with(
        parser(),
        "<tool_call>\n<function=now>\n</function>\n</tool_call>",
        3,
    );
    assert_eq!(calls.len(), 1);
}

/// The stream must agree with the whole-text parse on every input, well
/// formed or not, at every split.
#[test]
fn streaming_equals_whole_text_on_malformed_and_truncated_input() {
    let corpus: &[(ToolDialect, &str)] = &[
        (ToolDialect::QwenXml, QWEN),
        (ToolDialect::QwenXml, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>"),
        (ToolDialect::QwenXml, "<tool_call>\n<function=f>\n<parameter=a\n1"),
        (ToolDialect::QwenXml, "<tool_call>\n<function=>\n</function>\n</tool_call> after"),
        (ToolDialect::QwenXml, "<tool_call><function=f></function> text with no close"),
        (ToolDialect::QwenXml, "<tool_call>\n<function=f>\njunk <parameter=a>\nx\n</parameter>\n</tool_call>tail"),
        (ToolDialect::QwenXml, "<tool_call><tool_call>\n<function=f>\n</function>\n</tool_call>"),
        (ToolDialect::QwenXml, "é<tool_call>\n<function=f>\n<parameter=k>\nñ — ✓\n</parameter>\n</function>\n</tool_call>é"),
        (ToolDialect::HermesJson, HERMES),
        (ToolDialect::HermesJson, "<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}\n</tool_call>"),
        (ToolDialect::HermesJson, "<tool_call>\n{\"name\": \"f\", \"arguments\": [1]}\n</tool_call>"),
        (ToolDialect::HermesJson, "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": "),
        (ToolDialect::HermesJson, "<tool_call>{\"name\": \"f\"}</tool_call><tool_call>broken</tool_call>"),
        (ToolDialect::LfmPythonic, LFM),
        (ToolDialect::LfmPythonic, "<|tool_call_start|>[f(a='x', b=[1, {'k': None}])]<|tool_call_end|>"),
        (ToolDialect::LfmPythonic, "<|tool_call_start|>[f(a='x'"),
        (ToolDialect::LfmPythonic, "<|tool_call_start|>[{\"name\": \"f\", \"arguments\": {}}]<|tool_call_end|>x"),
        (ToolDialect::LfmPythonic, "<|tool_call_start|><|tool_call_start|>[f()]<|tool_call_end|>"),
    ];
    for &(dialect, text) in corpus {
        let (want_content, want_calls, _) = stream(dialect, text, text.chars().count().max(1));
        let want: Vec<(String, String)> = want_calls
            .iter()
            .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
            .collect();
        for chunk in 1..text.chars().count() {
            let (content, calls, _) = stream(dialect, text, chunk);
            let got: Vec<(String, String)> = calls
                .iter()
                .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
                .collect();
            assert_eq!(content, want_content, "{text:?} chunk {chunk}");
            assert_eq!(got, want, "{text:?} chunk {chunk}");
        }
        for (_, arguments) in &want {
            assert!(
                serde_json::from_str::<Value>(arguments).is_ok_and(|value| value.is_object()),
                "{text:?}: arguments {arguments:?} are a JSON object"
            );
        }
    }
}

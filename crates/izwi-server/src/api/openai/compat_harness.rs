use std::collections::HashSet;

use serde::Deserialize;

const CONTRACT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/openai-compatibility-contract.json"
));

const OPENAI_MOD_SRC: &str = include_str!("mod.rs");
const OPENAI_AUDIO_MOD_SRC: &str = include_str!("audio/mod.rs");
const OPENAI_CHAT_MOD_SRC: &str = include_str!("chat/mod.rs");
const OPENAI_RESPONSES_MOD_SRC: &str = include_str!("responses/mod.rs");
const OPENAI_CHAT_COMPLETIONS_SRC: &str = include_str!("chat/completions.rs");
const OPENAI_RESPONSES_HANDLERS_SRC: &str = include_str!("responses/handlers.rs");
const OPENAI_TRANSCRIPTIONS_SRC: &str = include_str!("audio/transcriptions.rs");
const OPENAI_SPEECH_SRC: &str = include_str!("audio/speech.rs");

#[derive(Debug, Deserialize)]
struct CompatibilityContract {
    scope: ScopeContract,
    streaming_contracts: StreamingContracts,
}

#[derive(Debug, Deserialize)]
struct ScopeContract {
    supported_endpoints: Vec<String>,
    out_of_scope_endpoints: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StreamingContracts {
    chat_completions: StreamContractSequence,
    responses: StreamContractSequence,
    audio_transcriptions: StreamContractEvents,
    audio_speech: StreamContractEvents,
}

#[derive(Debug, Deserialize)]
struct StreamContractSequence {
    required_sequence: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StreamContractEvents {
    required_events: Vec<String>,
}

fn parse_contract() -> CompatibilityContract {
    serde_json::from_str(CONTRACT_JSON).expect("compatibility contract should parse")
}

fn marker_matches_source(marker: &str, source: &str) -> bool {
    if marker == "[DONE]" {
        return source.contains("[DONE]");
    }
    let marker = marker
        .split('(')
        .next()
        .unwrap_or(marker)
        .trim();
    if marker.contains('|') {
        return marker.split('|').any(|option| source.contains(option.trim()));
    }
    source.contains(marker)
}

#[test]
fn contract_supported_scope_matches_expected_endpoints() {
    let contract = parse_contract();
    let actual: HashSet<&str> = contract
        .scope
        .supported_endpoints
        .iter()
        .map(String::as_str)
        .collect();
    let expected: HashSet<&str> = vec![
        "/v1/models",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/responses/:response_id",
        "/v1/responses/:response_id/cancel",
        "/v1/responses/:response_id/input_items",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ]
    .into_iter()
    .collect();
    assert_eq!(actual, expected);
}

#[test]
fn out_of_scope_endpoints_are_documented_and_not_routed() {
    let contract = parse_contract();
    let actual: HashSet<&str> = contract
        .scope
        .out_of_scope_endpoints
        .iter()
        .map(String::as_str)
        .collect();
    let expected: HashSet<&str> = vec![
        "/v1/audio/translations",
        "/v1/realtime/client_secrets",
        "/v1/realtime/sessions",
        "/v1/realtime/transcription_sessions",
    ]
    .into_iter()
    .collect();
    assert_eq!(actual, expected);

    let routed_src = format!(
        "{}\n{}\n{}\n{}",
        OPENAI_MOD_SRC, OPENAI_AUDIO_MOD_SRC, OPENAI_CHAT_MOD_SRC, OPENAI_RESPONSES_MOD_SRC
    );
    assert!(!routed_src.contains("/audio/translations"));
    assert!(!routed_src.contains("/realtime/client_secrets"));
    assert!(!routed_src.contains("/realtime/sessions"));
    assert!(!routed_src.contains("/realtime/transcription_sessions"));
}

#[test]
fn chat_stream_contract_markers_exist_in_source() {
    let contract = parse_contract();
    for marker in &contract.streaming_contracts.chat_completions.required_sequence {
        assert!(
            marker_matches_source(marker, OPENAI_CHAT_COMPLETIONS_SRC),
            "missing chat stream marker in source: {}",
            marker
        );
    }
}

#[test]
fn responses_stream_contract_markers_exist_in_source() {
    let contract = parse_contract();
    for marker in &contract.streaming_contracts.responses.required_sequence {
        assert!(
            marker_matches_source(marker, OPENAI_RESPONSES_HANDLERS_SRC),
            "missing responses stream marker in source: {}",
            marker
        );
    }
}

#[test]
fn transcription_stream_contract_markers_exist_in_source() {
    let contract = parse_contract();
    for marker in &contract.streaming_contracts.audio_transcriptions.required_events {
        assert!(
            marker_matches_source(marker, OPENAI_TRANSCRIPTIONS_SRC),
            "missing transcription stream marker in source: {}",
            marker
        );
    }
}

#[test]
fn speech_stream_contract_markers_exist_in_source() {
    let contract = parse_contract();
    for marker in &contract.streaming_contracts.audio_speech.required_events {
        assert!(
            marker_matches_source(marker, OPENAI_SPEECH_SRC),
            "missing speech stream marker in source: {}",
            marker
        );
    }
}

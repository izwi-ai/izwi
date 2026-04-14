use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{to_bytes, Body},
    http::{Method, Request, StatusCode},
    response::Response,
    Router,
};
use izwi_core::{RuntimeService, ServeRuntimeConfig};
use serde::Deserialize;
use tower::Service;

use crate::api::router::create_router;
use crate::state::AppState;

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

async fn send_request(mut app: Router, request: Request<Body>) -> Response {
    app.as_service::<Body>()
        .call(request)
        .await
        .expect("request should succeed")
}

fn build_json_request(method: Method, path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .expect("request should build")
}

fn test_ui_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be monotonic")
        .as_nanos();
    std::env::temp_dir().join(format!("izwi-openai-compat-{name}-{nanos}"))
}

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("environment lock should not be poisoned")
}

fn test_api_app(name: &str) -> (Router, TempDirGuard) {
    let temp_dir = test_ui_dir(name);
    let ui_dir = temp_dir.join("ui");
    let models_dir = temp_dir.join("models");
    std::fs::create_dir_all(&models_dir).expect("models dir should be created");

    let db_path = temp_dir.join("izwi.sqlite3");
    let media_dir = temp_dir.join("media");

    let _guard = env_lock();
    std::env::set_var("IZWI_DB_PATH", &db_path);
    std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

    let serve_config = ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        ui_dir,
        models_dir,
        ..ServeRuntimeConfig::default()
    };
    let runtime = with_suppressed_panic_hook(|| RuntimeService::new(serve_config.engine_config()))
        .expect("runtime should init");
    let state = AppState::new(runtime, &serve_config).expect("state should init");
    std::env::remove_var("IZWI_DB_PATH");
    std::env::remove_var("IZWI_MEDIA_DIR");

    (create_router(state, &serve_config), TempDirGuard(temp_dir))
}

fn with_suppressed_panic_hook<T>(f: impl FnOnce() -> T) -> T {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = f();
    std::panic::set_hook(default_hook);
    result
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

#[tokio::test]
async fn out_of_scope_endpoints_return_not_found() {
    let contract = parse_contract();
    let (app, _temp_dir) = test_api_app("out-of-scope-not-found");
    for endpoint in &contract.scope.out_of_scope_endpoints {
        let request = Request::builder()
            .method(Method::POST)
            .uri(endpoint)
            .body(Body::empty())
            .expect("request should build");
        let response = send_request(app.clone(), request).await;
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "expected 404 for out-of-scope endpoint {endpoint}"
        );
    }
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
fn audio_streaming_contract_lists_required_events() {
    let contract = parse_contract();
    assert_eq!(
        contract.streaming_contracts.audio_transcriptions.required_events,
        vec!["transcript.text.delta", "transcript.text.done"]
    );
    assert_eq!(
        contract.streaming_contracts.audio_speech.required_events,
        vec!["audio.chunk", "audio.done|audio.failed"]
    );
}

#[tokio::test]
async fn transcriptions_unknown_model_returns_openai_error_envelope() {
    let (app, _temp_dir) = test_api_app("transcriptions-unknown-model");
    let response = send_request(
        app,
        build_json_request(
            Method::POST,
            "/v1/audio/transcriptions",
            r#"{"audio_base64":"UklGRg==","model":"not-a-real-model"}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should be readable");
    let payload: serde_json::Value =
        serde_json::from_slice(&body).expect("error response should be json");
    assert_eq!(
        payload
            .get("error")
            .and_then(|error| error.get("type"))
            .and_then(|value| value.as_str()),
        Some("invalid_request_error")
    );
    assert!(
        payload
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .contains("Unsupported transcription model")
    );
}

#[tokio::test]
async fn speech_unknown_model_returns_openai_error_envelope() {
    let (app, _temp_dir) = test_api_app("speech-unknown-model");
    let response = send_request(
        app,
        build_json_request(
            Method::POST,
            "/v1/audio/speech",
            r#"{"model":"not-a-real-model","input":"hello world","voice":"alloy"}"#,
        ),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should be readable");
    let payload: serde_json::Value =
        serde_json::from_slice(&body).expect("error response should be json");
    assert_eq!(
        payload
            .get("error")
            .and_then(|error| error.get("type"))
            .and_then(|value| value.as_str()),
        Some("invalid_request_error")
    );
    assert!(
        payload
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .contains("Unsupported TTS model")
    );
}

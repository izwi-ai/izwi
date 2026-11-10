use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{to_bytes, Body},
    http::{Method, Request, StatusCode},
    response::Response,
    Router,
};
use base64::Engine as _;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::{RuntimeService, ServeRuntimeConfig};
use serde::Deserialize;
use tower::Service;

use crate::api::router::create_router;
use crate::state::AppState;
use crate::test_support::env_lock;

const CONTRACT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/openai-compatibility-contract.json"
));

const OPENAI_MOD_SRC: &str = include_str!("mod.rs");
const OPENAI_AUDIO_MOD_SRC: &str = include_str!("audio/mod.rs");
const OPENAI_REALTIME_SRC: &str = include_str!("realtime.rs");
const OPENAI_CHAT_MOD_SRC: &str = include_str!("chat/mod.rs");
const OPENAI_RESPONSES_MOD_SRC: &str = include_str!("responses/mod.rs");
// The chat stream renderer is izwi-gateway's, shared by both roles.
const OPENAI_CHAT_STREAM_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../izwi-gateway/src/api/openai/chat/stream.rs"
));
const OPENAI_RESPONSES_HANDLERS_SRC: &str = include_str!("responses/handlers.rs");
const OPENAI_SPEECH_SRC: &str = include_str!("audio/speech.rs");
const OPENAI_TRANSCRIPTIONS_SRC: &str = include_str!("audio/transcriptions.rs");

#[derive(Debug, Deserialize)]
struct CompatibilityContract {
    scope: ScopeContract,
    streaming_contracts: StreamingContracts,
    formats: FormatsContract,
}

#[derive(Debug, Deserialize)]
struct FormatsContract {
    audio_speech: AudioSpeechFormats,
}

#[derive(Debug, Deserialize)]
struct AudioSpeechFormats {
    response_format: Vec<String>,
    izwi_response_format: Vec<String>,
    refused_response_format: Vec<String>,
    stream_format: Vec<String>,
    pcm_sample_rate: u32,
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
    audio_translations: StreamContractEvents,
    audio_speech_stream: StreamContractEvents,
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
    let marker = marker.split('(').next().unwrap_or(marker).trim();
    if marker.contains('|') {
        return marker
            .split('|')
            .any(|option| source.contains(option.trim()));
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

fn valid_wav_base64() -> String {
    let wav = AudioEncoder::new(16_000, 1)
        .encode(&[0.0, 0.1, -0.1, 0.0], AudioFormat::Wav)
        .expect("wav should encode");
    base64::engine::general_purpose::STANDARD.encode(wav)
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
        "/v1/models/{model}",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/responses/:response_id",
        "/v1/responses/:response_id/cancel",
        "/v1/responses/:response_id/input_items",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/audio/speech",
        "/v1/audio/voices",
        "/v1/audio/voices/{voice_id}",
        "/v1/audio/voice_consents",
        "/v1/audio/voice_consents/{consent_id}",
        "/v1/realtime",
        "/v1/realtime/client_secrets",
        "/v1/files",
        "/v1/files/{file_id}",
        "/v1/files/{file_id}/content",
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
        "/v1/realtime/sessions",
        "/v1/realtime/transcription_sessions",
    ]
    .into_iter()
    .collect();
    assert_eq!(actual, expected);

    let routed_src = format!(
        "{}\n{}\n{}\n{}\n{}",
        OPENAI_MOD_SRC,
        OPENAI_AUDIO_MOD_SRC,
        OPENAI_CHAT_MOD_SRC,
        OPENAI_RESPONSES_MOD_SRC,
        OPENAI_REALTIME_SRC
    );
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
    for marker in &contract
        .streaming_contracts
        .chat_completions
        .required_sequence
    {
        assert!(
            marker_matches_source(marker, OPENAI_CHAT_STREAM_SRC),
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
        contract
            .streaming_contracts
            .audio_transcriptions
            .required_events,
        vec!["transcript.text.delta", "transcript.text.done"]
    );
    assert_eq!(
        contract.streaming_contracts.audio_speech.required_events,
        vec!["speech.audio.delta", "speech.audio.done|audio.failed"]
    );
    for marker in &contract.streaming_contracts.audio_speech.required_events {
        assert!(
            marker_matches_source(marker, OPENAI_SPEECH_SRC),
            "missing speech stream marker in source: {marker}"
        );
    }
    for marker in &contract
        .streaming_contracts
        .audio_translations
        .required_events
    {
        assert!(
            marker_matches_source(marker, OPENAI_TRANSCRIPTIONS_SRC),
            "missing translation stream marker in source: {marker}"
        );
    }
    assert_eq!(
        contract
            .streaming_contracts
            .audio_speech_stream
            .required_events,
        vec![
            "session.configured",
            "audio.start",
            "audio.done",
            "session.done"
        ]
    );
}

/// Every format the contract lists resolves to an encoder this build has;
/// every refused one is a 400 that says why.
#[test]
fn speech_formats_contract_matches_the_encoders() {
    use crate::api::openai::audio::speech::{parse_response_format, SpeechOutput};
    let formats = parse_contract().formats.audio_speech;
    for name in formats
        .response_format
        .iter()
        .chain(&formats.izwi_response_format)
    {
        let resolved = parse_response_format(name, false)
            .unwrap_or_else(|err| panic!("{name}: {}", err.message));
        assert!(resolved.fallback.is_none(), "{name} is native");
        assert!(resolved.format.is_available(), "{name}");
    }
    for name in &formats.refused_response_format {
        let error = parse_response_format(name, false).expect_err(name);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }
    let pcm = SpeechOutput::resolve(Some("pcm"), false, None, None).unwrap();
    assert_eq!(pcm.output_rate(44_100), formats.pcm_sample_rate);
    for stream_format in &formats.stream_format {
        assert!(
            OPENAI_SPEECH_SRC.contains(&format!("Some(\"{stream_format}\")")),
            "stream_format {stream_format} is resolved in speech.rs"
        );
    }
}

/// `/v1/audio/translations` is routed (it was out of scope until models that
/// translate speech joined the catalog): a model that cannot translate gets
/// the OpenAI error envelope, not a 404.
#[tokio::test]
async fn translations_refuse_a_model_that_does_not_translate() {
    let (app, _temp_dir) = test_api_app("translations-non-translating-model");
    let body = serde_json::json!({
        "audio_base64": valid_wav_base64(),
        "model": "Whisper-Large-v3-Turbo"
    })
    .to_string();
    let response = send_request(
        app,
        build_json_request(Method::POST, "/v1/audio/translations", body.as_str()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should be readable");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("error json");
    assert_eq!(json["error"]["param"], "model");
    assert!(json["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("Granite-Speech-5.0")));
}

#[tokio::test]
async fn transcriptions_unknown_model_returns_openai_error_envelope() {
    let (app, _temp_dir) = test_api_app("transcriptions-unknown-model");
    let audio_base64 = valid_wav_base64();
    let body = serde_json::json!({
        "audio_base64": audio_base64,
        "model": "not-a-real-model"
    })
    .to_string();
    let response = send_request(
        app,
        build_json_request(Method::POST, "/v1/audio/transcriptions", body.as_str()),
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
    assert!(payload
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .contains("Unsupported transcription model"));
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
    assert!(payload
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .contains("Unsupported TTS model"));
}

#[tokio::test]
async fn transcriptions_reject_unsupported_formats_before_asr() {
    let (app, _temp_dir) = test_api_app("transcriptions-unsupported-format");
    for (format, expected) in [
        ("diarized_json", "POST /v1/diarizations"),
        ("xml", "Unsupported response_format: xml"),
    ] {
        let body = serde_json::json!({
            "audio_base64": valid_wav_base64(),
            "model": "Parakeet-TDT-0.6B-v3",
            "response_format": format,
        })
        .to_string();
        let response = send_request(
            app.clone(),
            build_json_request(Method::POST, "/v1/audio/transcriptions", body.as_str()),
        )
        .await;

        // No model is downloaded here, so a response about the format proves
        // it was rejected before transcription ran.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{format}");
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let payload: serde_json::Value =
            serde_json::from_slice(&body).expect("error response should be json");
        let message = payload["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(expected), "{format}: {message}");
    }
}

/// A multipart body of text fields and one WAV file field.
fn multipart_request(
    path: &str,
    fields: &[(&str, &str)],
    file: Option<(&str, Vec<u8>)>,
) -> Request<Body> {
    let boundary = "izwi-test-boundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    if let Some((name, bytes)) = file {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("request should build")
}

fn one_second_wav() -> Vec<u8> {
    let tone: Vec<f32> = (0..16_000)
        .map(|index| (index as f32 * 220.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.3)
        .collect();
    AudioEncoder::new(16_000, 1)
        .encode(&tone, AudioFormat::Wav)
        .expect("wav should encode")
}

async fn json_body(response: Response) -> (StatusCode, serde_json::Value) {
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should be readable");
    (
        status,
        serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null),
    )
}

/// OpenAI's custom-voice flow end to end: a consent recording, a voice made
/// with it, lookups by id and by name (SGLang-Omni deletes by name), and the
/// consent's deletion revoking the voice.
#[tokio::test]
async fn voices_and_consents_follow_openais_resources() {
    let (app, _temp_dir) = test_api_app("voices-and-consents");
    let (status, consent) = json_body(
        send_request(
            app.clone(),
            multipart_request(
                "/v1/audio/voice_consents",
                &[("name", "Jane Doe"), ("language", "en-US")],
                Some(("recording", one_second_wav())),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{consent}");
    assert_eq!(consent["object"], "audio.voice_consent");
    let consent_id = consent["id"].as_str().expect("id").to_string();
    assert!(consent_id.starts_with("cons_"));
    assert_eq!(consent["language"], "en-US");

    let (status, list) = json_body(
        send_request(
            app.clone(),
            build_json_request(Method::GET, "/v1/audio/voice_consents?limit=5", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["object"], "list");
    assert_eq!(list["first_id"], consent_id.as_str());
    assert_eq!(list["has_more"], false);

    let (status, renamed) = json_body(
        send_request(
            app.clone(),
            build_json_request(
                Method::POST,
                &format!("/v1/audio/voice_consents/{consent_id}"),
                r#"{"name":"Jane D."}"#,
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(renamed["name"], "Jane D.");

    // OpenAI's built-in names are reserved.
    let (status, _) = json_body(
        send_request(
            app.clone(),
            multipart_request(
                "/v1/audio/voices",
                &[("name", "Alloy"), ("consent", consent_id.as_str())],
                Some(("audio_sample", one_second_wav())),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, voice) = json_body(
        send_request(
            app.clone(),
            multipart_request(
                "/v1/audio/voices",
                &[
                    ("name", "narrator"),
                    ("consent", consent_id.as_str()),
                    ("ref_text", "A calm reference sample."),
                ],
                Some(("audio_sample", one_second_wav())),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{voice}");
    assert_eq!(voice["object"], "audio.voice");
    assert_eq!(voice["type"], "audio_sample");
    assert_eq!(voice["consent"], consent_id.as_str());
    assert_eq!(voice["usable"], true);
    let voice_id = voice["id"].as_str().expect("voice id").to_string();

    // An attestation (free-text consent) works in attestation mode.
    let (status, attested) = json_body(
        send_request(
            app.clone(),
            multipart_request(
                "/v1/audio/voices",
                &[("name", "host"), ("consent", "my own voice")],
                Some(("audio_sample", one_second_wav())),
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{attested}");
    assert_eq!(attested["consent"], "my own voice");

    let (status, voices) = json_body(
        send_request(
            app.clone(),
            build_json_request(Method::GET, "/v1/audio/voices", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(voices["data"].as_array().map(Vec::len), Some(2));
    assert!(voices["voices"]
        .as_array()
        .is_some_and(|names| names.iter().any(|name| name == "narrator")));

    let (status, by_name) = json_body(
        send_request(
            app.clone(),
            build_json_request(Method::GET, "/v1/audio/voices/Narrator", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(by_name["id"], voice_id.as_str());

    // Deleting the consent revokes the voice made with it.
    let (status, deleted) = json_body(
        send_request(
            app.clone(),
            build_json_request(
                Method::DELETE,
                &format!("/v1/audio/voice_consents/{consent_id}"),
                "",
            ),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["deleted"], true);
    let (_, revoked) = json_body(
        send_request(
            app.clone(),
            build_json_request(Method::GET, &format!("/v1/audio/voices/{voice_id}"), ""),
        )
        .await,
    )
    .await;
    assert_eq!(revoked["usable"], false);

    let (status, gone) = json_body(
        send_request(
            app.clone(),
            build_json_request(Method::DELETE, "/v1/audio/voices/narrator", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(gone["object"], "audio.voice.deleted");
    let (status, _) = json_body(
        send_request(
            app,
            build_json_request(Method::DELETE, "/v1/audio/voices/narrator", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// A speech request's id is looked up afterwards; an unknown one is a 404.
#[tokio::test]
async fn speech_lookup_answers_unknown_ids_with_404() {
    let (app, _temp_dir) = test_api_app("speech-lookup");
    let (status, body) = json_body(
        send_request(
            app,
            build_json_request(Method::GET, "/v1/audio/speech/speech_nope", ""),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("speech_nope")));
}

/// The batch envelope SGLang-Omni and vLLM-Omni share: envelope errors fail
/// the request, item errors stay in their item.
#[tokio::test]
async fn speech_batches_keep_item_errors_in_their_items() {
    let (app, _temp_dir) = test_api_app("speech-batch");
    for (body, expect) in [
        (
            serde_json::json!({"model": "Kokoro-82M", "items": []}),
            "items",
        ),
        (
            serde_json::json!({"model": "Kokoro-82M", "stream": true, "items": [{"input": "a"}]}),
            "stream",
        ),
        (
            serde_json::json!({"model": "Kokoro-82M", "items": [{"input": "a", "stream_format": "sse"}]}),
            "items.0.stream_format",
        ),
    ] {
        let (status, json) = json_body(
            send_request(
                app.clone(),
                build_json_request(Method::POST, "/v1/audio/speech/batch", &body.to_string()),
            )
            .await,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            json["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(expect)),
            "{json}"
        );
    }
    let body = serde_json::json!({
        "model": "not-a-model",
        "response_format": "mp3",
        "items": [{"input": "One."}, {"input": "Two.", "speed": 1.5}],
    });
    let (status, json) = json_body(
        send_request(
            app,
            build_json_request(Method::POST, "/v1/audio/speech/batch", &body.to_string()),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["object"], "audio.speech.batch");
    assert_eq!(json["total"], 2);
    assert_eq!(json["failed"], 2);
    assert_eq!(json["results"][1]["index"], 1);
    assert_eq!(json["results"][1]["status"], "error");
    assert!(json["results"][0]["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("Unsupported TTS model")));
}

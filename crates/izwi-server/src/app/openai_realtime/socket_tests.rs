//! `/v1/realtime` over a real socket: the server on a loopback port, driven
//! by a WebSocket client the way stock OpenAI Realtime clients drive it.
//! No model weights are installed, so every model stage fails; these tests
//! pin the protocol around them (handshake, items, VAD turns, response
//! lifecycle, errors), event by event.

use std::net::SocketAddr;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use super::audio::{base64, pcm16_bytes};
use crate::state::AppState;

pub(crate) type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

pub(crate) struct Server {
    pub(crate) addr: SocketAddr,
    pub(crate) state: AppState,
    _dir: tempfile::TempDir,
}

pub(crate) async fn serve() -> Server {
    serve_with_hooks(izwi_hooks::EnterpriseHooks::noop()).await
}

pub(crate) async fn serve_with_hooks(hooks: izwi_hooks::EnterpriseHooks) -> Server {
    let dir = tempfile::tempdir().expect("temp dir");
    let models_dir = dir.path().join("models");
    std::fs::create_dir_all(&models_dir).expect("models dir");
    let serve_config = izwi_core::ServeRuntimeConfig {
        backend: izwi_core::backends::BackendPreference::Cpu,
        ui_enabled: false,
        models_dir,
        ..izwi_core::ServeRuntimeConfig::default()
    };
    let state = {
        let _env = crate::test_support::env_lock();
        std::env::set_var("IZWI_DB_PATH", dir.path().join("izwi.sqlite3"));
        std::env::set_var("IZWI_MEDIA_DIR", dir.path().join("media"));
        let runtime =
            izwi_core::RuntimeService::new(serve_config.engine_config()).expect("runtime");
        let state = AppState::with_enterprise_hooks(runtime, &serve_config, hooks).expect("state");
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
        state
    };
    state.lifecycle.mark_ready();
    let router = crate::api::create_router(state.clone(), &serve_config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    Server {
        addr,
        state,
        _dir: dir,
    }
}

async fn connect(server: &Server, query: &str) -> Client {
    let url = format!("ws://{}/v1/realtime{query}", server.addr);
    let (client, _) = tokio_tungstenite::connect_async(url)
        .await
        .expect("connect");
    client
}

async fn send(client: &mut Client, event: Value) {
    client
        .send(WsMessage::Text(event.to_string().into()))
        .await
        .expect("send");
}

/// The next server event.
async fn next(client: &mut Client) -> Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(20), client.next())
            .await
            .expect("a server event in time")
            .expect("open socket")
            .expect("message");
        if let WsMessage::Text(text) = message {
            let event: Value = serde_json::from_str(text.as_str()).expect("json event");
            assert!(
                event["event_id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("event_")),
                "every server event has an event_id: {event}"
            );
            return event;
        }
    }
}

/// Server events up to and including the first of `until`.
async fn until(client: &mut Client, until: &str) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = next(client).await;
        let done = event["type"] == until;
        events.push(event);
        if done {
            return events;
        }
    }
}

/// Server events until every type in `wanted` has arrived, in any order.
async fn until_all(client: &mut Client, wanted: &[&str]) -> Vec<Value> {
    let mut events = Vec::new();
    while !wanted
        .iter()
        .all(|kind| events.iter().any(|event: &Value| event["type"] == *kind))
    {
        events.push(next(client).await);
    }
    events
}

fn types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or(""))
        .collect()
}

fn position(events: &[Value], event_type: &str) -> usize {
    types(events)
        .iter()
        .position(|kind| *kind == event_type)
        .unwrap_or_else(|| panic!("no {event_type} in {:?}", types(events)))
}

async fn handshake(client: &mut Client) -> Value {
    let created = next(client).await;
    assert_eq!(created["type"], "session.created");
    let conversation = next(client).await;
    assert_eq!(conversation["type"], "conversation.created");
    assert_eq!(
        conversation["conversation"]["object"],
        "realtime.conversation"
    );
    created
}

fn pcm_chunk(samples: &[i16]) -> Value {
    json!({"type": "input_audio_buffer.append", "audio": base64(&pcm16_bytes(samples))})
}

fn tone(ms: usize, offset: usize) -> Vec<i16> {
    (0..24 * ms)
        .map(|index| {
            let t = (offset + index) as f32 / 24_000.0;
            ((t * 220.0 * std::f32::consts::TAU).sin() * 10_000.0) as i16
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stock_client_handshake_items_and_errors() {
    let server = serve().await;
    let mut client = connect(&server, "?model=gpt-realtime").await;
    let created = handshake(&mut client).await;
    let session = &created["session"];
    assert_eq!(session["type"], "realtime");
    assert_eq!(session["model"], "gpt-realtime");
    assert_eq!(
        session["audio"]["input"]["format"],
        json!({"type": "audio/pcm", "rate": 24000})
    );
    assert_eq!(
        session["audio"]["input"]["turn_detection"]["type"],
        "server_vad"
    );
    assert_eq!(session["audio"]["output"]["voice"], "marin");

    // LiveKit sends its whole session config first, then diffs.
    send(
        &mut client,
        json!({"type": "session.update", "session": {"type": "realtime", "instructions": "Be brief.",
               "audio": {"output": {"voice": "cedar"}}}}),
    )
    .await;
    let updated = next(&mut client).await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["instructions"], "Be brief.");
    assert_eq!(updated["session"]["audio"]["output"]["voice"], "cedar");

    // History sync with client ids, `root` and `previous_item_id`.
    send(
        &mut client,
        json!({"type": "conversation.item.create", "event_id": "c1", "previous_item_id": "root",
               "item": {"id": "lk_user_1", "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "Hi"}]}}),
    )
    .await;
    let added = next(&mut client).await;
    assert_eq!(added["type"], "conversation.item.added");
    assert_eq!(added["previous_item_id"], Value::Null);
    assert_eq!(added["item"]["id"], "lk_user_1");
    assert_eq!(next(&mut client).await["type"], "conversation.item.done");
    send(
        &mut client,
        json!({"type": "conversation.item.create", "previous_item_id": "lk_user_1",
               "item": {"id": "lk_assistant_1", "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello!"}]}}),
    )
    .await;
    let added = next(&mut client).await;
    assert_eq!(added["previous_item_id"], "lk_user_1");
    next(&mut client).await;

    send(
        &mut client,
        json!({"type": "conversation.item.retrieve", "item_id": "lk_assistant_1"}),
    )
    .await;
    let retrieved = next(&mut client).await;
    assert_eq!(retrieved["type"], "conversation.item.retrieved");
    assert_eq!(retrieved["item"]["content"][0]["text"], "Hello!");

    send(
        &mut client,
        json!({"type": "conversation.item.delete", "item_id": "lk_assistant_1"}),
    )
    .await;
    assert_eq!(next(&mut client).await["type"], "conversation.item.deleted");
    send(
        &mut client,
        json!({"type": "conversation.item.delete", "event_id": "del_2", "item_id": "nope"}),
    )
    .await;
    let error = next(&mut client).await;
    assert_eq!(error["type"], "error");
    assert_eq!(
        error["error"]["event_id"], "del_2",
        "LiveKit matches errors by event_id"
    );

    // A cancel with nothing running: both clients tolerate this error.
    send(&mut client, json!({"type": "response.cancel"})).await;
    let error = next(&mut client).await;
    assert_eq!(error["error"]["code"], "response_cancel_not_active");
    assert!(error["error"]["message"]
        .as_str()
        .is_some_and(|message| message.starts_with("Cancellation failed")));

    // A beta-shaped update is refused, and changes nothing.
    send(
        &mut client,
        json!({"type": "session.update", "session": {"modalities": ["text", "audio"], "input_audio_format": "pcm16"}}),
    )
    .await;
    assert_eq!(
        next(&mut client).await["error"]["code"],
        "unsupported_beta_session"
    );
    send(&mut client, json!({"type": "nope.nope"})).await;
    assert_eq!(next(&mut client).await["error"]["code"], "invalid_value");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_response_without_its_model_fails_with_its_lifecycle_intact() {
    let server = serve().await;
    let mut client = connect(&server, "").await;
    handshake(&mut client).await;
    send(
        &mut client,
        json!({"type": "conversation.item.create",
               "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}}),
    )
    .await;
    until(&mut client, "conversation.item.done").await;
    send(
        &mut client,
        json!({"type": "response.create", "event_id": "response_create_1",
               "response": {"instructions": null, "metadata": {"client_event_id": "response_create_1"}}}),
    )
    .await;
    let created = next(&mut client).await;
    assert_eq!(created["type"], "response.created");
    assert_eq!(created["response"]["status"], "in_progress");
    assert_eq!(
        created["response"]["metadata"]["client_event_id"], "response_create_1",
        "LiveKit binds generate_reply by the echoed metadata"
    );
    let events = until(&mut client, "response.done").await;
    let done = events.last().expect("done");
    assert_eq!(done["response"]["id"], created["response"]["id"]);
    assert_eq!(done["response"]["status"], "failed");
    assert!(
        done["response"]["status_details"]["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("not downloaded")),
        "{done}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_server_vad_turn_commits_transcribes_and_responds() {
    let server = serve().await;
    let mut client = connect(&server, "?model=izwi-voice").await;
    handshake(&mut client).await;
    send(
        &mut client,
        json!({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {
            "transcription": {"model": "gpt-4o-mini-transcribe"},
            "turn_detection": {"type": "server_vad", "threshold": 0.01, "silence_duration_ms": 300}}}}}),
    )
    .await;
    assert_eq!(next(&mut client).await["type"], "session.updated");

    // 300 ms of silence, 900 ms of tone, 1 s of silence, in 100 ms appends.
    let mut offset = 0;
    for chunk in 0..22 {
        let samples = if (3..12).contains(&chunk) {
            tone(100, offset)
        } else {
            vec![0; 2_400]
        };
        offset += samples.len();
        send(&mut client, pcm_chunk(&samples)).await;
    }
    // Transcription was asked for: it fails here (no ASR model), beside the
    // response, in either order.
    let events = until_all(
        &mut client,
        &[
            "response.done",
            "conversation.item.input_audio_transcription.failed",
        ],
    )
    .await;
    let started = &events[position(&events, "input_audio_buffer.speech_started")];
    let stopped = &events[position(&events, "input_audio_buffer.speech_stopped")];
    let committed = &events[position(&events, "input_audio_buffer.committed")];
    assert_eq!(started["item_id"], committed["item_id"]);
    assert_eq!(stopped["item_id"], committed["item_id"]);
    let start_ms = started["audio_start_ms"].as_u64().expect("start");
    // At threshold 0.01 even digital silence scores as speech, so the turn
    // may start at 0; it is dated where speech began, not where it was confirmed.
    assert!(
        start_ms <= 400,
        "dated back to the speech start: {start_ms}"
    );
    assert!(stopped["audio_end_ms"].as_u64().expect("end") > start_ms + 800);
    assert!(
        position(&events, "input_audio_buffer.speech_started")
            < position(&events, "input_audio_buffer.speech_stopped")
    );
    assert!(
        position(&events, "input_audio_buffer.speech_stopped")
            < position(&events, "input_audio_buffer.committed")
    );
    assert!(
        position(&events, "input_audio_buffer.committed")
            < position(&events, "conversation.item.added")
    );
    assert!(position(&events, "conversation.item.added") < position(&events, "response.created"));
    let added = &events[position(&events, "conversation.item.added")];
    assert_eq!(added["item"]["content"][0]["type"], "input_audio");
    assert_eq!(
        events[position(&events, "response.done")]["response"]["status"],
        "failed"
    );
    let failed = &events[position(
        &events,
        "conversation.item.input_audio_transcription.failed",
    )];
    assert_eq!(failed["item_id"], committed["item_id"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn manual_turns_commit_and_clear() {
    let server = serve().await;
    let mut client = connect(&server, "?model=izwi-voice-compact").await;
    handshake(&mut client).await;
    send(
        &mut client,
        json!({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"turn_detection": null}}}}),
    )
    .await;
    let updated = next(&mut client).await;
    assert_eq!(
        updated["session"]["audio"]["input"]["turn_detection"],
        Value::Null
    );
    send(&mut client, pcm_chunk(&tone(50, 0))).await;
    send(
        &mut client,
        json!({"type": "input_audio_buffer.commit", "event_id": "commit_1"}),
    )
    .await;
    let error = next(&mut client).await;
    assert_eq!(error["error"]["code"], "input_audio_buffer_commit_empty");
    assert_eq!(error["error"]["event_id"], "commit_1");
    send(&mut client, pcm_chunk(&tone(200, 1_200))).await;
    send(&mut client, json!({"type": "input_audio_buffer.commit"})).await;
    let committed = next(&mut client).await;
    assert_eq!(committed["type"], "input_audio_buffer.committed");
    assert_eq!(next(&mut client).await["type"], "conversation.item.added");
    assert_eq!(next(&mut client).await["type"], "conversation.item.done");
    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    send(&mut client, json!({"type": "input_audio_buffer.clear"})).await;
    assert_eq!(
        next(&mut client).await["type"],
        "input_audio_buffer.cleared"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upgrades_refuse_unknown_models_and_echo_the_realtime_subprotocol() {
    let server = serve().await;
    let url = format!("ws://{}/v1/realtime?model=no-such-model", server.addr);
    match tokio_tungstenite::connect_async(url).await {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
            assert_eq!(response.status(), 400)
        }
        other => panic!("expected HTTP 400, got {other:?}"),
    }
    let url = format!("ws://{}/v1/realtime?call_id=rtc_1", server.addr);
    assert!(tokio_tungstenite::connect_async(url).await.is_err());

    let mut request = format!("ws://{}/v1/realtime?model=izwi-voice", server.addr)
        .into_client_request()
        .expect("request");
    request.headers_mut().insert(
        "sec-websocket-protocol",
        "realtime".parse().expect("header"),
    );
    let (mut client, response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("connect");
    assert_eq!(
        response
            .headers()
            .get("sec-websocket-protocol")
            .and_then(|value| value.to_str().ok()),
        Some("realtime")
    );
    assert!(response
        .headers()
        .get("x-izwi-realtime-profile")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|profile| profile.contains("tts=Kokoro-82M")));
    handshake(&mut client).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_secrets_bind_a_session_config_and_capabilities_list_profiles() {
    let server = serve().await;
    let http = reqwest::Client::new();
    let minted: Value = http
        .post(format!("http://{}/v1/realtime/client_secrets", server.addr))
        .json(&json!({
            "expires_after": {"anchor": "created_at", "seconds": 120},
            "session": {"type": "realtime", "model": "izwi-voice-compact", "instructions": "From the secret.",
                        "audio": {"output": {"voice": "cedar"}}},
        }))
        .send()
        .await
        .expect("mint")
        .json()
        .await
        .expect("json");
    let key = minted["value"].as_str().expect("value");
    assert!(key.starts_with("ek_"), "{minted}");
    assert_eq!(minted["session"]["instructions"], "From the secret.");
    assert_eq!(minted["session"]["object"], "realtime.session");

    let bad = http
        .post(format!("http://{}/v1/realtime/client_secrets", server.addr))
        .json(&json!({"expires_after": {"seconds": 5}}))
        .send()
        .await
        .expect("request");
    assert_eq!(bad.status(), 400, "a lifetime under 10 s is refused");

    // A browser presents the key as a subprotocol and names no model.
    let mut request = format!("ws://{}/v1/realtime", server.addr)
        .into_client_request()
        .expect("request");
    request.headers_mut().insert(
        "sec-websocket-protocol",
        format!("realtime, openai-insecure-api-key.{key}")
            .parse()
            .expect("header"),
    );
    let (mut client, response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("connect");
    assert!(response
        .headers()
        .get("x-izwi-realtime-profile")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|profile| profile.starts_with("s2s=")));
    let created = handshake(&mut client).await;
    assert_eq!(created["session"]["instructions"], "From the secret.");
    assert_eq!(created["session"]["audio"]["output"]["voice"], "cedar");
    assert_eq!(created["session"]["model"], "izwi-voice-compact");

    let capabilities: Value = http
        .get(format!("http://{}/v1/realtime/capabilities", server.addr))
        .send()
        .await
        .expect("capabilities")
        .json()
        .await
        .expect("json");
    assert_eq!(capabilities["default_profile"], "izwi-voice");
    let ids: Vec<&str> = capabilities["profiles"]
        .as_array()
        .expect("profiles")
        .iter()
        .filter_map(|profile| profile["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"izwi-voice") && ids.contains(&"izwi-voice-compact"),
        "{ids:?}"
    );
}

/// `IZWI_REALTIME_PERSIST=1` records a session's turns in the voice history
/// (D5); without it nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn persisted_sessions_record_their_turns_in_the_voice_history() {
    let server = serve().await;
    let sessions_before = server
        .state
        .voice_store
        .list_sessions(100)
        .await
        .expect("sessions")
        .len();
    let mut client = {
        let _env = crate::test_support::env_lock();
        std::env::set_var("IZWI_REALTIME_PERSIST", "1");
        let mut client = connect(&server, "?model=izwi-voice").await;
        handshake(&mut client).await;
        std::env::remove_var("IZWI_REALTIME_PERSIST");
        client
    };
    send(
        &mut client,
        json!({"type": "conversation.item.create",
               "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Remember me."}]}}),
    )
    .await;
    until(&mut client, "conversation.item.done").await;
    send(&mut client, json!({"type": "response.create"})).await;
    until(&mut client, "response.done").await;
    client.close(None).await.expect("close");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let sessions = server
            .state
            .voice_store
            .list_sessions(100)
            .await
            .expect("sessions");
        if sessions.len() > sessions_before {
            let session = server
                .state
                .voice_store
                .get_session(sessions[0].id.clone())
                .await
                .expect("session")
                .expect("stored");
            // A turn is written in steps (created, transcript, reply, then
            // completed), so wait until it leaves `processing`.
            if let Some(turn) = session
                .turns
                .first()
                .filter(|turn| turn.status != "processing")
            {
                assert_eq!(turn.user_text.as_deref(), Some("Remember me."));
                assert_eq!(
                    turn.status, "error",
                    "the response failed without its model"
                );
                break;
            }
        }
        assert!(std::time::Instant::now() < deadline, "the turn is recorded");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A model with a native streaming session, which transcribes turns live.
pub(crate) const LIVE_ASR_MODEL: &str = "Voxtral-Mini-4B-Realtime-2602";

fn live_session(turn_detection: Value) -> Value {
    json!({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {
        "transcription": {"model": LIVE_ASR_MODEL},
        "turn_detection": turn_detection}}}})
}

/// Wait for every scripted native session to close.
pub(crate) async fn sessions_closed(script: &super::live::scripted::Script) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while script.open.load(std::sync::atomic::Ordering::SeqCst) > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the native session is released"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn deltas(events: &[Value]) -> String {
    events
        .iter()
        .filter(|event| event["type"] == "conversation.item.input_audio_transcription.delta")
        .filter_map(|event| event["delta"].as_str())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_model_transcribes_a_turn_while_it_is_spoken() {
    let server = serve().await;
    let script = super::live::scripted::install(&server.state);
    let mut client = connect(&server, "?model=izwi-voice").await;
    handshake(&mut client).await;
    send(
        &mut client,
        live_session(json!({"type": "server_vad", "threshold": 0.01, "silence_duration_ms": 300})),
    )
    .await;
    assert_eq!(next(&mut client).await["type"], "session.updated");

    // 300 ms of silence and 900 ms of tone: the turn is still open.
    let mut offset = 0;
    for chunk in 0..12 {
        let samples = if chunk < 3 {
            vec![0; 2_400]
        } else {
            tone(100, offset)
        };
        offset += samples.len();
        send(&mut client, pcm_chunk(&samples)).await;
    }
    let mut events = until(
        &mut client,
        "conversation.item.input_audio_transcription.delta",
    )
    .await;
    let started = events[position(&events, "input_audio_buffer.speech_started")].clone();
    assert!(
        !types(&events).contains(&"input_audio_buffer.speech_stopped"),
        "the first delta arrives while the user speaks: {:?}",
        types(&events)
    );
    assert_eq!(events.last().expect("delta")["item_id"], started["item_id"]);

    // A second of silence ends the turn.
    for _ in 0..10 {
        send(&mut client, pcm_chunk(&[0; 2_400])).await;
    }
    events.extend(
        until_all(
            &mut client,
            &[
                "response.done",
                "conversation.item.input_audio_transcription.completed",
            ],
        )
        .await,
    );
    let kinds = types(&events);
    assert!(
        !kinds.contains(&"error")
            && !kinds.contains(&"conversation.item.input_audio_transcription.failed"),
        "{kinds:?}"
    );
    let committed = &events[position(&events, "input_audio_buffer.committed")];
    assert_eq!(committed["item_id"], started["item_id"]);
    assert!(
        position(&events, "input_audio_buffer.speech_stopped")
            < position(&events, "input_audio_buffer.committed")
    );
    assert!(
        position(&events, "input_audio_buffer.committed")
            < position(
                &events,
                "conversation.item.input_audio_transcription.completed"
            )
    );
    let completed = &events[position(
        &events,
        "conversation.item.input_audio_transcription.completed",
    )];
    assert_eq!(completed["item_id"], started["item_id"]);
    let transcript = completed["transcript"].as_str().expect("transcript");
    assert!(
        transcript.starts_with("word1") && transcript.ends_with("end."),
        "{transcript}"
    );
    assert_eq!(deltas(&events), transcript, "the deltas add up to it");

    // The item carries the same transcript, for the reply and for history.
    send(
        &mut client,
        json!({"type": "conversation.item.retrieve", "item_id": started["item_id"]}),
    )
    .await;
    let retrieved = until(&mut client, "conversation.item.retrieved").await;
    assert_eq!(
        retrieved.last().expect("item")["item"]["content"][0]["transcript"],
        transcript
    );
    // One native session served the turn, and the commit closed it.
    assert_eq!(script.started.load(std::sync::atomic::Ordering::SeqCst), 1);
    sessions_closed(&script).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_live_turn_without_a_native_session_is_transcribed_at_its_commit() {
    let server = serve().await;
    let script = super::live::scripted::install(&server.state);
    script
        .unavailable
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let mut client = connect(&server, "?intent=transcription").await;
    assert_eq!(next(&mut client).await["type"], "session.created");
    send(&mut client, live_session(Value::Null)).await;
    assert_eq!(next(&mut client).await["type"], "session.updated");

    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    send(&mut client, json!({"type": "input_audio_buffer.commit"})).await;
    // The turn-based pass runs instead (and fails here, without weights):
    // no delta, and no `error` event for the missing session.
    let events = until(
        &mut client,
        "conversation.item.input_audio_transcription.failed",
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.failed"
        ]
    );
    assert_eq!(events[0]["item_id"], events[1]["item_id"]);
    assert_eq!(script.started.load(std::sync::atomic::Ordering::SeqCst), 0);

    // A session that fails mid-turn falls back the same way, and its
    // deltas are not sent a second time.
    script
        .unavailable
        .store(false, std::sync::atomic::Ordering::SeqCst);
    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    let delta = next(&mut client).await;
    assert_eq!(
        delta["type"],
        "conversation.item.input_audio_transcription.delta"
    );
    script
        .fail_pushes
        .store(true, std::sync::atomic::Ordering::SeqCst);
    send(&mut client, pcm_chunk(&tone(200, 4_800))).await;
    sessions_closed(&script).await;
    send(&mut client, json!({"type": "input_audio_buffer.commit"})).await;
    let events = until(
        &mut client,
        "conversation.item.input_audio_transcription.failed",
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.failed"
        ]
    );
    assert_eq!(events[0]["item_id"], delta["item_id"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clearing_or_closing_a_live_turn_releases_its_native_session() {
    let server = serve().await;
    let script = super::live::scripted::install(&server.state);
    let mut client = connect(&server, "?intent=transcription").await;
    assert_eq!(next(&mut client).await["type"], "session.created");
    send(&mut client, live_session(Value::Null)).await;
    assert_eq!(next(&mut client).await["type"], "session.updated");

    // A manual turn is transcribed from its first audio.
    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    let cleared_turn = next(&mut client).await;
    assert_eq!(
        cleared_turn["type"],
        "conversation.item.input_audio_transcription.delta"
    );
    assert_eq!(script.open.load(std::sync::atomic::Ordering::SeqCst), 1);
    send(&mut client, json!({"type": "input_audio_buffer.clear"})).await;
    assert_eq!(
        next(&mut client).await["type"],
        "input_audio_buffer.cleared"
    );
    sessions_closed(&script).await;

    // The next turn has its own session and its own item.
    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    let delta = next(&mut client).await;
    assert_ne!(delta["item_id"], cleared_turn["item_id"]);
    send(&mut client, json!({"type": "input_audio_buffer.commit"})).await;
    let events = until(
        &mut client,
        "conversation.item.input_audio_transcription.completed",
    )
    .await;
    assert_eq!(events[0]["type"], "input_audio_buffer.committed");
    assert_eq!(events[0]["item_id"], delta["item_id"]);
    let completed = events.last().expect("completed");
    assert_eq!(completed["item_id"], delta["item_id"]);
    assert_eq!(completed["transcript"], "word1 end.");
    assert_eq!(completed["usage"]["type"], "duration");
    sessions_closed(&script).await;

    // A socket that closes mid-turn leaves no session behind.
    send(&mut client, pcm_chunk(&tone(200, 0))).await;
    assert_eq!(
        next(&mut client).await["type"],
        "conversation.item.input_audio_transcription.delta"
    );
    assert_eq!(script.open.load(std::sync::atomic::Ordering::SeqCst), 1);
    client.close(None).await.expect("close");
    sessions_closed(&script).await;
    assert_eq!(script.started.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn capabilities_report_how_each_model_transcribes() {
    let server = serve().await;
    let capabilities: Value = reqwest::Client::new()
        .get(format!("http://{}/v1/realtime/capabilities", server.addr))
        .send()
        .await
        .expect("capabilities")
        .json()
        .await
        .expect("json");
    let mode = |model: &str| {
        capabilities["transcription_models"]
            .as_array()
            .expect("transcription models")
            .iter()
            .find(|entry| entry["id"] == model)
            .unwrap_or_else(|| panic!("no {model}"))["transcription_mode"]
            .clone()
    };
    assert_eq!(mode(LIVE_ASR_MODEL), "live");
    assert_eq!(mode("Parakeet-TDT-0.6B-v3"), "turn");
    for profile in capabilities["profiles"].as_array().expect("profiles") {
        // The built-in profiles transcribe with Parakeet, or reply with a
        // unified model: both commit a turn before transcribing it.
        assert_eq!(profile["transcription_mode"], "turn", "{profile}");
    }
}

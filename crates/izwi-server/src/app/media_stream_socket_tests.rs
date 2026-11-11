//! `/v1/realtime/media-stream` over a real socket, driven the way a carrier
//! drives it. No model weights are installed, so a reply fails at its model;
//! live transcripts come from the scripted native session of the
//! `/v1/realtime` socket tests. What a reply's speech becomes on the stream
//! is pinned by the unit tests in `media_stream`.

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::app::openai_realtime::audio::base64;
use crate::app::openai_realtime::live::scripted;
use crate::app::openai_realtime::socket_tests::{
    serve, serve_with_hooks, sessions_closed, Client, Server, LIVE_ASR_MODEL,
};
use crate::app::realtime_codec::linear_to_ulaw;

/// A carrier's connection, with `headers`.
async fn connect(server: &Server, query: &str, headers: &[(&'static str, String)]) -> Client {
    let mut request = format!("ws://{}/v1/realtime/media-stream{query}", server.addr)
        .into_client_request()
        .expect("request");
    for (name, value) in headers {
        request
            .headers_mut()
            .insert(*name, value.parse().expect("header"));
    }
    let (client, _) = tokio_tungstenite::connect_async(request)
        .await
        .expect("connect");
    client
}

async fn send(client: &mut Client, frame: Value) {
    client
        .send(WsMessage::Text(frame.to_string().into()))
        .await
        .expect("send");
}

/// `connected` and `start`, as a carrier opens a call.
async fn start(client: &mut Client, custom_parameters: Value) {
    send(
        client,
        json!({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
    )
    .await;
    send(
        client,
        json!({"event": "start", "sequenceNumber": "1", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "accountSid": "AC1", "callSid": "CA1", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            "customParameters": custom_parameters}}),
    )
    .await;
}

/// `ms` of audio in 20 ms μ-law frames: a 220 Hz tone, or silence.
async fn send_audio(client: &mut Client, ms: usize, tone: bool) {
    for frame in 0..ms / 20 {
        let payload: Vec<u8> = (0..160)
            .map(|index| {
                let t = (frame * 160 + index) as f32 / 8_000.0;
                let sample = (t * 220.0 * std::f32::consts::TAU).sin() * 10_000.0;
                linear_to_ulaw(if tone { sample as i16 } else { 0 })
            })
            .collect();
        send(
            client,
            json!({"event": "media", "streamSid": "MZ1", "media": {"track": "inbound",
                "chunk": frame.to_string(), "timestamp": (frame * 20).to_string(),
                "payload": base64(&payload)}}),
        )
        .await;
    }
}

/// The next frame of the stream; `Err` with the close code once it closes.
async fn next(client: &mut Client) -> Result<Value, Option<u16>> {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(20), client.next())
            .await
            .expect("a frame in time");
        match message {
            Some(Ok(WsMessage::Text(text))) => {
                let frame: Value = serde_json::from_str(text.as_str()).expect("json frame");
                assert_eq!(frame["streamSid"], "MZ1", "{frame}");
                return Ok(frame);
            }
            Some(Ok(WsMessage::Close(frame))) => return Err(frame.map(|frame| frame.code.into())),
            Some(Ok(_)) => {}
            Some(Err(_)) | None => return Err(None),
        }
    }
}

/// A client secret whose session cuts turns the way the `/v1/realtime`
/// socket tests do: a tone is no voice, so the VAD takes anything.
fn secret(session_type: &str, input: Value) -> Vec<(&'static str, String)> {
    let mut input = input;
    input["turn_detection"] =
        json!({"type": "server_vad", "threshold": 0.01, "silence_duration_ms": 300});
    let session = json!({"type": session_type, "audio": {"input": input}});
    let (key, _) = crate::app::openai_realtime::secrets::mint(session, 60).expect("mint");
    vec![("authorization", format!("Bearer {key}"))]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_transcription_call_streams_transcript_frames() {
    let server = serve().await;
    let script = scripted::install(&server.state);
    let headers = secret(
        "transcription",
        json!({"transcription": {"model": LIVE_ASR_MODEL}}),
    );
    let mut client = connect(&server, "", &headers).await;
    start(&mut client, json!({})).await;
    // Frames outside the framing, before and between the audio, are ignored.
    send(
        &mut client,
        json!({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"digit": "5"}}),
    )
    .await;
    send(
        &mut client,
        json!({"event": "something-new", "streamSid": "MZ1"}),
    )
    .await;
    client
        .send(WsMessage::Text("not json".into()))
        .await
        .expect("send");

    send_audio(&mut client, 300, false).await;
    send_audio(&mut client, 900, true).await;
    send(
        &mut client,
        json!({"event": "mark", "streamSid": "MZ1", "mark": {"name": "x"}}),
    )
    .await;
    send_audio(&mut client, 1_000, false).await;

    // Nothing but transcripts reaches the carrier: deltas, then the whole.
    let mut deltas = String::new();
    let completed = loop {
        let frame = next(&mut client).await.expect("a transcript frame");
        assert_eq!(frame["event"], "transcript", "{frame}");
        assert!(frame["item_id"].as_str().is_some(), "{frame}");
        match frame["type"].as_str() {
            Some("delta") => deltas.push_str(frame["text"].as_str().expect("text")),
            Some("completed") => break frame,
            other => panic!("unexpected transcript type {other:?}"),
        }
    };
    let transcript = completed["text"].as_str().expect("text");
    assert!(
        transcript.starts_with("word1") && transcript.ends_with("end."),
        "{transcript}"
    );
    assert_eq!(deltas, transcript, "the deltas add up to it");

    // `stop` ends the call and its session.
    send(
        &mut client,
        json!({"event": "stop", "streamSid": "MZ1", "stop": {"callSid": "CA1"}}),
    )
    .await;
    assert_eq!(next(&mut client).await, Err(Some(1000)));
    sessions_closed(&script).await;
    assert_eq!(server.state.realtime_session_admission_snapshot().active, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_conversation_call_answers_a_turn_and_keeps_session_events_off_the_stream() {
    let server = serve().await;
    let mut client = connect(&server, "?model=izwi-voice", &secret("realtime", json!({}))).await;
    // The carrier's custom parameters reach the session.
    start(
        &mut client,
        json!({"voice": "cedar", "transcripts": "false"}),
    )
    .await;
    send_audio(&mut client, 300, false).await;
    send_audio(&mut client, 900, true).await;
    send_audio(&mut client, 1_000, false).await;

    // The turn is answered; with no model installed the reply fails, which
    // is the first frame the carrier sees.
    let frame = next(&mut client).await.expect("a frame");
    assert_eq!(frame["event"], "error", "{frame}");
    assert!(
        frame["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("not downloaded")),
        "{frame}"
    );

    // The carrier hanging up ends the session.
    drop(client);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while server.state.realtime_session_admission_snapshot().active > 0 {
        assert!(std::time::Instant::now() < deadline, "the session ends");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn calls_that_cannot_run_are_refused() {
    let server = serve().await;
    // A bad query refuses the upgrade.
    for query in ["?model=no-such-model", "?intent=fax"] {
        let url = format!("ws://{}/v1/realtime/media-stream{query}", server.addr);
        match tokio_tungstenite::connect_async(url).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), 400, "{query}")
            }
            other => panic!("expected HTTP 400 for {query}, got {other:?}"),
        }
    }

    // A bad custom parameter, known only at `start`, closes the call.
    let mut client = connect(&server, "", &[]).await;
    start(&mut client, json!({"model": "no-such-model"})).await;
    let frame = next(&mut client).await.expect("an error frame");
    assert_eq!(frame["event"], "error", "{frame}");
    assert_eq!(frame["error"]["code"], "invalid_call");
    assert_eq!(next(&mut client).await, Err(Some(1008)));

    // So does audio the adapter does not carry.
    let mut client = connect(&server, "", &[]).await;
    send(
        &mut client,
        json!({"event": "start", "streamSid": "MZ1", "start": {"streamSid": "MZ1",
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 16000, "channels": 1}}}),
    )
    .await;
    let message = client.next().await.expect("frame").expect("message");
    let WsMessage::Text(text) = message else {
        panic!("expected an error frame, got {message:?}");
    };
    assert!(text.as_str().contains("Unsupported media format"), "{text}");
    assert_eq!(server.state.realtime_session_admission_snapshot().active, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_takes_a_realtime_session_slot() {
    let server = serve().await;
    let held: Vec<_> = std::iter::from_fn(|| server.state.try_acquire_realtime_session()).collect();
    assert!(!held.is_empty());
    let mut client = connect(&server, "", &[]).await;
    start(&mut client, json!({})).await;
    let frame = next(&mut client).await.expect("an error frame");
    assert_eq!(frame["event"], "error", "{frame}");
    assert_eq!(frame["error"]["code"], "session_capacity");
    assert_eq!(next(&mut client).await, Err(Some(1013)));

    // A freed slot admits the next call, which holds it until it ends.
    drop(held);
    let mut client = connect(&server, "", &[]).await;
    start(&mut client, json!({})).await;
    send_audio(&mut client, 20, false).await;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while server.state.realtime_session_admission_snapshot().active != 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "the call holds a slot"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_call_needs_the_api_key_or_a_client_secret() {
    const KEY: &str = "media-stream-test-api-key-123456";
    let server = serve_with_hooks(crate::local_security::with_api_key(
        izwi_hooks::EnterpriseHooks::noop(),
        Some(crate::local_security::LocalApiKey::new(KEY).expect("key")),
        false,
    ))
    .await;
    let url = format!("ws://{}/v1/realtime/media-stream", server.addr);
    for headers in [
        vec![],
        vec![("authorization", "Bearer wrong-key".to_string())],
        vec![("authorization", "Bearer ek_unknown".to_string())],
    ] {
        let mut request = url.as_str().into_client_request().expect("request");
        for (name, value) in headers {
            request
                .headers_mut()
                .insert(name, value.parse().expect("header"));
        }
        match tokio_tungstenite::connect_async(request).await {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), 401)
            }
            other => panic!("expected HTTP 401, got {other:?}"),
        }
    }
    connect(&server, "", &[("authorization", format!("Bearer {KEY}"))]).await;
    // A client secret, as a bearer token or as the browser subprotocol.
    connect(&server, "", &secret("realtime", json!({}))).await;
    let key = secret("realtime", json!({})).remove(0).1;
    let key = key.trim_start_matches("Bearer ");
    connect(
        &server,
        "",
        &[(
            "sec-websocket-protocol",
            format!("realtime, openai-insecure-api-key.{key}"),
        )],
    )
    .await;
}

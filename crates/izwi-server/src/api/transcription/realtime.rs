//! Realtime transcription websocket endpoint for `/transcription`.

use std::time::Instant;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, State,
    },
    response::Response,
    routing::get,
    Router,
};
use base64::Engine;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, warn};

use crate::api::request_context::RequestContext;
use crate::state::AppState;

const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
const REALTIME_PROTOCOL: &str = "transcription_realtime_v1";

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        language: Option<String>,
    },
    AudioSnapshot {
        sequence: u64,
        audio_base64: String,
    },
    SessionStop,
    Ping {
        #[serde(default)]
        timestamp_ms: Option<u64>,
    },
}

#[derive(Debug, Default)]
struct RealtimeSessionState {
    started: bool,
    model_id: Option<String>,
    language: Option<String>,
    last_sequence: Option<u64>,
}

pub fn router() -> Router<AppState> {
    Router::new().route("/transcription/realtime/ws", get(ws_upgrade))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
) -> Response {
    let correlation_id = ctx.correlation_id;
    ws.on_upgrade(move |socket| handle_socket(socket, state, correlation_id))
}

async fn handle_socket(mut socket: WebSocket, state: AppState, correlation_id: String) {
    let mut session = RealtimeSessionState::default();

    if send_json(
        &mut socket,
        json!({
            "type": "session_ready",
            "protocol": REALTIME_PROTOCOL,
            "correlation_id": correlation_id,
        }),
    )
    .await
    .is_err()
    {
        return;
    }

    while let Some(message) = socket.recv().await {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                warn!("transcription realtime websocket receive error: {err}");
                break;
            }
        };

        match message {
            Message::Text(text) => {
                if handle_text_message(
                    &mut socket,
                    &state,
                    &correlation_id,
                    &mut session,
                    text.as_str(),
                )
                .await
                .is_err()
                {
                    break;
                }
            }
            Message::Ping(payload) => {
                if socket.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            Message::Binary(_) => {
                if send_json(
                    &mut socket,
                    json!({
                        "type": "error",
                        "message": "Unexpected binary message",
                    }),
                )
                .await
                .is_err()
                {
                    break;
                }
            }
            _ => {}
        }
    }
}

async fn handle_text_message(
    socket: &mut WebSocket,
    state: &AppState,
    correlation_id: &str,
    session: &mut RealtimeSessionState,
    text: &str,
) -> Result<(), ()> {
    let event: ClientEvent = match serde_json::from_str(text) {
        Ok(event) => event,
        Err(err) => {
            send_json(
                socket,
                json!({
                    "type": "error",
                    "message": format!("Invalid realtime event payload: {err}"),
                }),
            )
            .await?;
            return Ok(());
        }
    };

    match event {
        ClientEvent::SessionStart { model_id, language } => {
            session.started = true;
            session.model_id = model_id;
            session.language = language;
            session.last_sequence = None;
            send_json(
                socket,
                json!({
                    "type": "session_started",
                }),
            )
            .await?;
        }
        ClientEvent::AudioSnapshot {
            sequence,
            audio_base64,
        } => {
            if !session.started {
                send_json(
                    socket,
                    json!({
                        "type": "error",
                        "message": "session_start is required before audio_snapshot",
                    }),
                )
                .await?;
                return Ok(());
            }

            if session
                .last_sequence
                .is_some_and(|last_sequence| sequence <= last_sequence)
            {
                debug!(
                    "transcription realtime stale snapshot ignored: sequence={} last_sequence={:?}",
                    sequence, session.last_sequence
                );
                return Ok(());
            }
            session.last_sequence = Some(sequence);

            let audio_bytes = match decode_audio_base64(audio_base64.as_str()) {
                Ok(bytes) => bytes,
                Err(message) => {
                    send_json(
                        socket,
                        json!({
                            "type": "error",
                            "message": message,
                        }),
                    )
                    .await?;
                    return Ok(());
                }
            };

            if audio_bytes.is_empty() {
                send_json(
                    socket,
                    json!({
                        "type": "error",
                        "message": "audio_snapshot payload is empty",
                    }),
                )
                .await?;
                return Ok(());
            }

            if audio_bytes.len() > MAX_SNAPSHOT_BYTES {
                send_json(
                    socket,
                    json!({
                        "type": "error",
                        "message": format!(
                            "audio_snapshot exceeds max size ({} > {})",
                            audio_bytes.len(),
                            MAX_SNAPSHOT_BYTES
                        ),
                    }),
                )
                .await?;
                return Ok(());
            }

            let _permit = state.acquire_permit().await;
            let started = Instant::now();

            let output = match state
                .runtime
                .asr_transcribe_streaming_with_correlation(
                    base64::engine::general_purpose::STANDARD
                        .encode(&audio_bytes)
                        .as_str(),
                    session.model_id.as_deref(),
                    session.language.as_deref(),
                    Some(correlation_id),
                    |_delta| {},
                )
                .await
            {
                Ok(output) => output,
                Err(err) => {
                    send_json(
                        socket,
                        json!({
                            "type": "error",
                            "message": format!("ASR failed: {err}"),
                        }),
                    )
                    .await?;
                    return Ok(());
                }
            };

            let processing_time_ms = started.elapsed().as_secs_f64() * 1000.0;
            let rtf = if output.duration_secs > 0.0 {
                Some((processing_time_ms / 1000.0) / output.duration_secs as f64)
            } else {
                None
            };

            send_json(
                socket,
                json!({
                    "type": "transcript_partial",
                    "sequence": sequence,
                    "text": output.text,
                    "language": output.language,
                    "audio_duration_secs": output.duration_secs,
                    "processing_time_ms": processing_time_ms,
                    "rtf": rtf,
                }),
            )
            .await?;
        }
        ClientEvent::SessionStop => {
            send_json(socket, json!({ "type": "session_done" })).await?;
            return Err(());
        }
        ClientEvent::Ping { timestamp_ms } => {
            send_json(
                socket,
                json!({
                    "type": "pong",
                    "timestamp_ms": timestamp_ms,
                }),
            )
            .await?;
        }
    }

    Ok(())
}

async fn send_json(socket: &mut WebSocket, value: serde_json::Value) -> Result<(), ()> {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .map_err(|_| ())
}

fn decode_audio_base64(input: &str) -> Result<Vec<u8>, String> {
    let payload = input
        .split_once(',')
        .map(|(_, value)| value)
        .unwrap_or(input)
        .trim();

    if payload.is_empty() {
        return Err("Audio payload is empty".to_string());
    }

    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| format!("Invalid base64 audio payload: {err}"))
}

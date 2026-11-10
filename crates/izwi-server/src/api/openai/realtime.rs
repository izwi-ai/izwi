//! OpenAI Realtime routes:
//!
//! - `GET /v1/realtime`: the Realtime (GA) WebSocket. `?model=` selects a
//!   realtime profile (see `app::openai_realtime::profiles`);
//!   `?intent=transcription` opens a transcription session; `?call_id=` (the
//!   WebRTC/SIP sideband) is not served. A client offering the `realtime`
//!   subprotocol gets it back, and a browser may authenticate with an
//!   ephemeral key as the `openai-insecure-api-key.<key>` subprotocol.
//! - `POST /v1/realtime/client_secrets`: mint such an ephemeral key, bound to
//!   a session config.
//! - `GET /v1/realtime/capabilities`: Izwi's realtime profiles, voices and
//!   formats, for integrators choosing `model` and `voice` (no OpenAI SDK
//!   reads it).

use axum::{
    extract::{ws::WebSocketUpgrade, Extension, Query, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::request_context::RequestContext;
use crate::app::openai_realtime::events::Ids;
use crate::app::openai_realtime::live::{transcribes_live, transcription_mode};
use crate::app::openai_realtime::profiles::{
    configured_specs, resolve_profile, transcribes_input, transcription_model, validate_spec,
    ProfileMode,
};
use crate::app::openai_realtime::secrets;
use crate::app::openai_realtime::session_config::SessionConfig;
use crate::app::openai_realtime::{handle_socket, SessionKind};
use crate::error::ApiError;
use crate::state::AppState;
use crate::voice_defaults::DEFAULT_VOICE_AGENT_SYSTEM_PROMPT;

/// The largest client message: a 15 MiB `input_audio_buffer.append` in base64.
const MAX_MESSAGE_BYTES: usize = 24 * 1024 * 1024;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/realtime", get(ws_upgrade))
        .route("/realtime/client_secrets", post(create_client_secret))
        .route("/realtime/capabilities", get(capabilities))
}

#[derive(Debug, Default, Deserialize)]
struct RealtimeQuery {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
}

/// The subprotocols the client offered.
fn offered_protocols(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(axum::http::header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|protocol| protocol.trim().to_string())
        .filter(|protocol| !protocol.is_empty())
        .collect()
}

/// The session config an ephemeral key on this request binds, if any.
fn bound_session(headers: &HeaderMap) -> Option<Value> {
    let pairs: Vec<(String, String)> = headers
        .iter()
        .filter_map(|(name, value)| {
            Some((
                name.as_str().to_ascii_lowercase(),
                value.to_str().ok()?.to_string(),
            ))
        })
        .collect();
    secrets::presented_key(
        pairs
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str())),
    )
    .and_then(|key| secrets::lookup(&key))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Query(query): Query<RealtimeQuery>,
    headers: HeaderMap,
) -> Response {
    if query.call_id.is_some() {
        return ApiError::bad_request(
            "This server does not serve WebRTC or SIP calls; connect with ?model= over WebSocket.",
        )
        .into_response();
    }
    let bound = bound_session(&headers);
    let bound_type = bound
        .as_ref()
        .and_then(|session| session.get("type"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let kind = match query
        .intent
        .as_deref()
        .map(str::trim)
        .or(bound_type.as_deref())
    {
        None | Some("" | "realtime" | "conversation") => SessionKind::Realtime,
        Some("transcription") => SessionKind::Transcription,
        Some(other) => {
            return ApiError::bad_request(format!(
                "Unsupported intent `{other}`; use `realtime` or `transcription`."
            ))
            .into_response()
        }
    };
    let model = query.model.clone().or_else(|| {
        bound
            .as_ref()
            .and_then(|session| session.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let profile = match resolve_profile(model.as_deref()) {
        Ok(profile) => profile,
        Err(err) => return ApiError::bad_request(err).into_response(),
    };
    let requested_model = model
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| profile.id.clone());
    let profile_header = profile.summary();
    let ws = ws
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES);
    let ws = if offered_protocols(&headers)
        .iter()
        .any(|protocol| protocol == "realtime")
    {
        ws.protocols(["realtime"])
    } else {
        ws
    };
    let correlation_id = ctx.correlation_id;
    // The session counts against the speech quota for as long as it is open.
    let speech_connection = ctx.speech_connection;
    let mut response = ws.on_upgrade(move |socket| async move {
        let _speech_connection = speech_connection;
        handle_socket(
            socket,
            state,
            correlation_id,
            profile,
            requested_model,
            kind,
            bound,
        )
        .await
    });
    if let Ok(value) = axum::http::HeaderValue::from_str(&profile_header) {
        response
            .headers_mut()
            .insert("x-izwi-realtime-profile", value);
    }
    response
}

#[derive(Debug, Default, Deserialize)]
struct ExpiresAfter {
    #[serde(default)]
    anchor: Option<String>,
    #[serde(default)]
    seconds: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct ClientSecretRequest {
    #[serde(default)]
    expires_after: Option<ExpiresAfter>,
    #[serde(default)]
    session: Option<Value>,
}

/// `POST /v1/realtime/client_secrets`.
async fn create_client_secret(
    Json(request): Json<ClientSecretRequest>,
) -> Result<Json<Value>, ApiError> {
    let ttl = match request.expires_after {
        None => secrets::DEFAULT_TTL_SECS,
        Some(ExpiresAfter { anchor, seconds }) => {
            if anchor
                .as_deref()
                .is_some_and(|anchor| anchor != "created_at")
            {
                return Err(ApiError::invalid_param(
                    "expires_after.anchor",
                    "The only anchor is `created_at`.",
                ));
            }
            let seconds = seconds.unwrap_or(secrets::DEFAULT_TTL_SECS);
            if !(secrets::MIN_TTL_SECS..=secrets::MAX_TTL_SECS).contains(&seconds) {
                return Err(ApiError::invalid_param(
                    "expires_after.seconds",
                    "A client secret lives from 10 to 7200 seconds.",
                ));
            }
            seconds
        }
    };
    let session = request
        .session
        .unwrap_or_else(|| json!({"type": "realtime"}));
    if !session.is_object() {
        return Err(ApiError::invalid_param(
            "session",
            "`session` must be an object.",
        ));
    }
    let session_type = session
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("realtime");
    if !matches!(session_type, "realtime" | "transcription") {
        return Err(ApiError::invalid_param(
            "session.type",
            "`session.type` is `realtime` or `transcription`.",
        ));
    }
    let model = session.get("model").and_then(Value::as_str);
    let profile = resolve_profile(model).map_err(ApiError::bad_request)?;
    let ids = Ids::new();
    // Validate the session config now, so a bad one fails here rather than
    // in the browser's session.
    let mut config = SessionConfig::new(
        ids.next("sess"),
        model.unwrap_or(&profile.id).to_string(),
        profile.default_voice.clone(),
        DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.to_string(),
        0,
    );
    let mut update = session.clone();
    if session_type == "transcription" {
        update["type"] = json!("realtime");
        if let Some(object) = update.as_object_mut() {
            object.retain(|key, _| key == "type" || key == "audio" || key == "include");
        }
    }
    config
        .apply_update(&update, false)
        .map_err(|err| ApiError::bad_request(err.message))?;
    let (value, expires_at) = secrets::mint(session.clone(), ttl).map_err(ApiError::bad_request)?;
    config.expires_at = expires_at;
    let mut effective = config.to_json();
    if session_type == "transcription" {
        effective = json!({
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": config.id,
            "expires_at": expires_at,
            "audio": {"input": effective["audio"]["input"]},
        });
    }
    Ok(Json(
        json!({"value": value, "expires_at": expires_at, "session": effective}),
    ))
}

/// `GET /v1/realtime/capabilities`.
async fn capabilities(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let specs = configured_specs().map_err(ApiError::internal)?;
    let default_profile = resolve_profile(None).ok().map(|profile| profile.id);
    let profiles: Vec<Value> = specs
        .iter()
        .map(|spec| match validate_spec(spec) {
            Ok(profile) => {
                let models = match &profile.mode {
                    ProfileMode::Modular {
                        asr_model_id,
                        text_model_id,
                        tts_model_id,
                        ..
                    } => json!({"asr": asr_model_id, "text": text_model_id, "tts": tts_model_id}),
                    ProfileMode::Unified { s2s_model_id, .. } => json!({"s2s": s2s_model_id}),
                };
                // How a conversation on this profile transcribes its turns
                // when the client names no transcription model.
                let live = transcription_model(&profile, None)
                    .is_ok_and(|model| transcribes_live(&profile, true, &model));
                json!({
                    "id": profile.id,
                    "mode": if profile.is_unified() { "unified" } else { "modular" },
                    "models": models,
                    "default_voice": profile.default_voice,
                    "transcription_mode": transcription_mode(live),
                })
            }
            Err(error) => json!({"id": spec.id, "error": error}),
        })
        .collect();
    // Every model `audio.input.transcription.model` may name.
    let transcription_models: Vec<Value> = izwi_core::ModelVariant::all()
        .iter()
        .filter(|variant| transcribes_input(**variant))
        .map(|variant| {
            json!({
                "id": variant.dir_name(),
                "transcription_mode": transcription_mode(izwi_core::asr_streams_live(*variant)),
            })
        })
        .collect();
    Ok(Json(json!({
        "object": "realtime.capabilities",
        "default_profile": default_profile,
        "profiles": profiles,
        "transcription_models": transcription_models,
        "model_aliases": {
            "openai": "gpt-realtime*, gpt-4o-realtime*, gpt-4o-mini-realtime* and gpt-audio* select the default profile",
            "audio_chat_model_ids": "a speech-to-speech model id selects that model",
            "chat_model_ids": "a chat model id selects the default ASR and TTS with that chat model",
        },
        "voices": {
            "openai": ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"],
            "custom": "{\"id\": \"<saved voice id>\"} for a TTS model that clones voices",
            "native": "any other name is passed to the model as its own voice",
        },
        "audio_formats": [
            {"type": "audio/pcm", "rate": 24000},
            {"type": "audio/pcmu"},
            {"type": "audio/pcma"},
        ],
        "turn_detection": ["server_vad", "semantic_vad", null],
        "output_modalities": [["audio"], ["text"]],
        "transports": ["websocket"],
        "limits": {
            "max_sessions": state.realtime_session_admission_snapshot().capacity,
            "max_append_bytes": crate::app::openai_realtime::audio::MAX_APPEND_BYTES,
            "session_lifetime_s": 3600,
        },
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offered_subprotocols_split_on_commas() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::SEC_WEBSOCKET_PROTOCOL,
            "realtime, openai-insecure-api-key.ek_123"
                .parse()
                .expect("header"),
        );
        assert_eq!(
            offered_protocols(&headers),
            ["realtime", "openai-insecure-api-key.ek_123"]
        );
    }
}

//! The gateway's OpenAPI document: the local server's document restricted to
//! the routes the gateway can serve. The schema annotations live on
//! izwi-server's handlers, so izwi-server generates this file and its
//! `gateway_openapi_document_is_current` test keeps it in step. The handler
//! drops the routes this gateway's configuration leaves off.

use axum::extract::State;
use axum::Json;
use serde_json::Value;
use std::sync::OnceLock;

use crate::gateway::GatewayState;

/// The checked-in document, `gateway-openapi.json`.
const GATEWAY_OPENAPI: &str = include_str!("gateway-openapi.json");

/// The document as JSON.
pub fn gateway_document() -> &'static Value {
    static DOCUMENT: OnceLock<Value> = OnceLock::new();
    DOCUMENT.get_or_init(|| {
        serde_json::from_str(GATEWAY_OPENAPI).expect("gateway-openapi.json is valid JSON")
    })
}

/// Whether this gateway serves `path`: `/v1/realtime` and its media-stream
/// framing need OpenAI Realtime upstreams, `/v1/audio/*` needs audio upstreams, and voices, consents,
/// files, batches and jobs also need their shared state.
fn serves(state: &GatewayState, path: &str) -> bool {
    use crate::app::audio_proxy::is_shared_state_route;
    if matches!(path, "/v1/realtime" | "/v1/realtime/media-stream") {
        return state.openai_realtime.is_some();
    }
    if !path.starts_with("/v1/audio/") && !is_shared_state_route(path) {
        return true;
    }
    state
        .audio_upstreams
        .as_ref()
        .is_some_and(|audio| audio.shared_state() || !is_shared_state_route(path))
}

pub async fn gateway_openapi_json(State(state): State<GatewayState>) -> Json<Value> {
    let mut document = gateway_document().clone();
    if let Some(paths) = document.get_mut("paths").and_then(Value::as_object_mut) {
        paths.retain(|path, _| serves(&state, path));
    }
    Json(document)
}

use axum::{
    extract::{ws::WebSocketUpgrade, Extension, State},
    response::Response,
    routing::get,
    Router,
};

use crate::api::request_context::RequestContext;
use crate::app::transcription_realtime;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/transcription/realtime/ws", get(ws_upgrade))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
) -> Response {
    let correlation_id = ctx.correlation_id;
    ws.on_upgrade(move |socket| {
        transcription_realtime::handle_socket(socket, state, correlation_id)
    })
}

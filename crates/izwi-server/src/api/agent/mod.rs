//! Minimal agent session + turn endpoints for first-party clients.

mod handlers;

use axum::{
    routing::{get, post},
    Router,
};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/agent/sessions", post(handlers::create_session))
        .route("/agent/sessions/{session_id}", get(handlers::get_session))
        .route(
            "/agent/sessions/{session_id}/turns",
            post(handlers::create_turn),
        )
}

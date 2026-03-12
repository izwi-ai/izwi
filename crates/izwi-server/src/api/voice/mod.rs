use axum::{routing::get, Router};

use crate::state::AppState;

mod handlers;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/voice/profile",
            get(handlers::get_voice_profile).patch(handlers::update_voice_profile),
        )
        .route("/voice/sessions", get(handlers::list_voice_sessions))
        .route(
            "/voice/sessions/:session_id",
            get(handlers::get_voice_session),
        )
}

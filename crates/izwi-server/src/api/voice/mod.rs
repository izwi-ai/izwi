use axum::{routing::get, Router};

use crate::state::AppState;

mod handlers;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/voice/profile",
            get(handlers::get_voice_profile).patch(handlers::update_voice_profile),
        )
        .route(
            "/voice/observations",
            get(handlers::list_voice_observations).delete(handlers::clear_voice_observations),
        )
        .route(
            "/voice/observations/:observation_id",
            axum::routing::delete(handlers::delete_voice_observation),
        )
        .route("/voice/sessions", get(handlers::list_voice_sessions))
        .route(
            "/voice/sessions/:session_id",
            get(handlers::get_voice_session),
        )
}

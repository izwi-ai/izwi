//! OpenAI-compatible API namespace.

pub mod audio;
pub mod chat;
pub mod models;
pub mod responses;

use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(audio::router())
        .merge(chat::router())
        .merge(models::router())
        .merge(responses::router())
}

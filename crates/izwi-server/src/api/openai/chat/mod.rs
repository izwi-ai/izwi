//! OpenAI-compatible chat resources.

pub mod completions;
pub use izwi_gateway::api::openai::chat::dto;

use axum::{extract::DefaultBodyLimit, routing::post, Router};

use crate::api::chat::CHAT_REQUEST_LIMIT_BYTES;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route(
        "/chat/completions",
        post(completions::completions).layer(DefaultBodyLimit::max(CHAT_REQUEST_LIMIT_BYTES)),
    )
}

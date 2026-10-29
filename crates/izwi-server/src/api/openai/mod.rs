//! OpenAI-compatible API namespace.

pub mod audio;
pub mod chat;
pub mod compat;
pub use izwi_gateway::api::openai::convert;
pub mod models;
pub mod responses;
pub(crate) use izwi_gateway::api::openai::tool_calls;
pub(crate) use izwi_gateway::api::openai::tool_response;

use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(audio::router())
        .merge(chat::router())
        .merge(models::router())
        .merge(responses::router())
}

#[cfg(test)]
mod compat_harness;

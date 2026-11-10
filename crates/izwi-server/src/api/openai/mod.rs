//! OpenAI-compatible API namespace.

pub mod audio;
pub mod chat;
pub mod compat;
pub mod files;
pub use izwi_gateway::api::openai::convert;
pub(crate) mod model_aliases;
pub mod models;
pub mod realtime;
pub mod responses;
pub(crate) mod service_tier;
pub(crate) use izwi_gateway::api::openai::tool_calls;
pub(crate) use izwi_gateway::api::openai::tool_response;

use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(audio::router())
        .merge(chat::router())
        .merge(files::router())
        .merge(models::router())
        .merge(realtime::router())
        .merge(responses::router())
}

#[cfg(test)]
mod compat_harness;

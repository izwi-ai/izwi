//! OpenAI-compatible API namespace.

pub mod audio;
pub mod chat;
pub mod compat;
pub mod convert;
mod lfm_tools;
pub mod models;
pub mod responses;
pub(crate) mod tool_calls;

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

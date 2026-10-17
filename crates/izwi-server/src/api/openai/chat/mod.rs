//! OpenAI-compatible chat resources.

pub mod completions;
pub mod dto;
mod reasoning;
mod stream;

use axum::{routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/chat/completions", post(completions::completions))
}

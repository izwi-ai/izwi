//! OpenAI-compatible chat resources.

pub mod completions;
pub use izwi_gateway::api::openai::chat::{dto, reasoning};
mod stream;

use axum::{routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/chat/completions", post(completions::completions))
}

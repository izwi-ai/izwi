//! OpenAI-compatible chat resources.

pub mod completions;
mod lfm_tools;
mod reasoning;

use axum::{routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/chat/completions", post(completions::completions))
}

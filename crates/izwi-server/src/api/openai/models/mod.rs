//! OpenAI-compatible model resources.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/models", get(handlers::list_models_openai))
        .route("/models/{model}", get(handlers::get_model_openai))
}

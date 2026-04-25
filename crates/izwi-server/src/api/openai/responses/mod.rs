//! OpenAI-compatible responses resources.

mod dto;
mod handlers;

use axum::{routing::get, routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/responses", post(handlers::create_response))
        .route(
            "/responses/{response_id}",
            get(handlers::get_response).delete(handlers::delete_response),
        )
        .route(
            "/responses/{response_id}/cancel",
            post(handlers::cancel_response),
        )
        .route(
            "/responses/{response_id}/input_items",
            get(handlers::list_response_input_items),
        )
}

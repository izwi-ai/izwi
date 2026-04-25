//! Thread-based chat resources for first-party UI clients.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/chat/threads",
            get(handlers::list_threads).post(handlers::create_thread),
        )
        .route(
            "/chat/threads/{thread_id}",
            get(handlers::get_thread)
                .patch(handlers::update_thread)
                .delete(handlers::delete_thread),
        )
        .route(
            "/chat/threads/{thread_id}/messages",
            get(handlers::list_thread_messages).post(handlers::create_thread_message),
        )
}

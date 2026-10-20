//! Thread-based chat resources for first-party UI clients.

mod handlers;
mod title;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, patch, post},
    Router,
};

use crate::state::AppState;

/// Chat requests carry images as base64 data URLs, so a single photo can
/// outgrow axum's 2 MB default. Each image is still checked against the
/// model's own limit when it is decoded.
pub(crate) const CHAT_REQUEST_LIMIT_BYTES: usize = 64 * 1024 * 1024;

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
            "/chat/threads/{thread_id}/title",
            post(title::generate_thread_title),
        )
        .route(
            "/chat/threads/{thread_id}/messages",
            get(handlers::list_thread_messages).post(handlers::create_thread_message),
        )
        .route(
            "/chat/threads/{thread_id}/messages/{message_id}",
            patch(handlers::edit_thread_message).delete(handlers::delete_thread_message),
        )
        .route(
            "/chat/threads/{thread_id}/messages/{message_id}/regenerate",
            post(handlers::regenerate_thread_message),
        )
        .layer(DefaultBodyLimit::max(CHAT_REQUEST_LIMIT_BYTES))
}

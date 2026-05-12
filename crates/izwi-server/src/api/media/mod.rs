//! Media file serving endpoints for persisted local chat attachments.

mod handlers;

use axum::{Router, routing::get};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/media",
            get(handlers::list_media).post(handlers::create_media),
        )
        .route(
            "/media/{*path}",
            get(handlers::get_media).delete(handlers::delete_media),
        )
}

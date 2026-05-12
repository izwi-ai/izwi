//! Media file serving endpoints for persisted local chat attachments.

mod handlers;

use axum::{Router, extract::DefaultBodyLimit, routing::get};

use crate::api::openai::audio::resolve_audio_upload_limit_bytes;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    let media_upload_limit_bytes = resolve_audio_upload_limit_bytes();

    Router::new()
        .route(
            "/media",
            get(handlers::list_media)
                .post(handlers::create_media)
                .layer(DefaultBodyLimit::max(media_upload_limit_bytes)),
        )
        .route(
            "/media/{*path}",
            get(handlers::get_media).delete(handlers::delete_media),
        )
}

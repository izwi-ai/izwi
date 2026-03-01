//! First-party transcription history routes for the desktop UI.

mod handlers;
mod realtime;

use axum::{extract::DefaultBodyLimit, routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    Router::new()
        .route(
            "/transcription/records",
            get(handlers::list_records)
                .post(handlers::create_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .merge(realtime::router())
        .route(
            "/transcription/records/:record_id",
            get(handlers::get_record).delete(handlers::delete_record),
        )
        .route(
            "/transcription/records/:record_id/audio",
            get(handlers::get_record_audio),
        )
}

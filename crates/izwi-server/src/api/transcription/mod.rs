//! First-party persisted transcription resource routes for the desktop UI.

mod handlers;
mod realtime;

use axum::{extract::DefaultBodyLimit, routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;
    const CANONICAL_COLLECTION: &str = "/transcriptions";
    const CANONICAL_MEMBER: &str = "/transcriptions/:record_id";
    const CANONICAL_AUDIO: &str = "/transcriptions/:record_id/audio";

    Router::new()
        .route(
            CANONICAL_COLLECTION,
            get(handlers::list_records)
                .post(handlers::create_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .merge(realtime::router())
        .route(
            CANONICAL_MEMBER,
            get(handlers::get_record).delete(handlers::delete_record),
        )
        .route(CANONICAL_AUDIO, get(handlers::get_record_audio))
}

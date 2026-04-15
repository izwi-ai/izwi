//! First-party persisted transcription resource routes for the desktop UI.

mod handlers;
mod realtime;
mod unified_read;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;
    const CANONICAL_COLLECTION: &str = "/transcriptions";
    const CANONICAL_MEMBER: &str = "/transcriptions/:record_id";
    const CANONICAL_AUDIO: &str = "/transcriptions/:record_id/audio";
    const CANONICAL_SUMMARY_REGENERATE: &str = "/transcriptions/:record_id/summary/regenerate";
    const CANONICAL_UNIFIED_COLLECTION: &str = "/transcriptions/jobs";
    const CANONICAL_UNIFIED_MEMBER: &str = "/transcriptions/jobs/:record_id";
    const CANONICAL_UNIFIED_AUDIO: &str = "/transcriptions/jobs/:record_id/audio";

    Router::new()
        .route(
            CANONICAL_COLLECTION,
            get(handlers::list_records)
                .post(handlers::create_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .merge(realtime::router())
        .route(CANONICAL_UNIFIED_COLLECTION, get(unified_read::list_jobs))
        .route(CANONICAL_UNIFIED_MEMBER, get(unified_read::get_job))
        .route(CANONICAL_UNIFIED_AUDIO, get(unified_read::get_job_audio))
        .route(
            CANONICAL_MEMBER,
            get(handlers::get_record).delete(handlers::delete_record),
        )
        .route(CANONICAL_AUDIO, get(handlers::get_record_audio))
        .route(
            CANONICAL_SUMMARY_REGENERATE,
            post(handlers::regenerate_summary),
        )
}

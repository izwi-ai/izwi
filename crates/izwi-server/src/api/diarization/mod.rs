//! First-party persisted diarization resource routes for the desktop UI.

pub mod handlers;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;
    const CANONICAL_COLLECTION: &str = "/diarizations";
    const CANONICAL_MEMBER: &str = "/diarizations/:record_id";
    const CANONICAL_AUDIO: &str = "/diarizations/:record_id/audio";
    const CANONICAL_RERUNS: &str = "/diarizations/:record_id/reruns";
    const CANONICAL_CANCEL: &str = "/diarizations/:record_id/cancel";
    const CANONICAL_SUMMARY_REGENERATE: &str = "/diarizations/:record_id/summary/regenerate";

    Router::new()
        .route(
            CANONICAL_COLLECTION,
            get(handlers::list_records)
                .post(handlers::create_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            CANONICAL_MEMBER,
            get(handlers::get_record)
                .patch(handlers::update_record)
                .put(handlers::update_record)
                .delete(handlers::delete_record),
        )
        .route(CANONICAL_AUDIO, get(handlers::get_record_audio))
        .route(
            CANONICAL_RERUNS,
            axum::routing::post(handlers::rerun_record),
        )
        .route(CANONICAL_CANCEL, post(handlers::cancel_record))
        .route(
            CANONICAL_SUMMARY_REGENERATE,
            post(handlers::regenerate_summary),
        )
}

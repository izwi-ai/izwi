//! First-party persisted speech-to-text job routes for the desktop UI.

mod handlers;
mod realtime;
mod unified_read;
mod unified_write;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};

use crate::state::AppState;

pub(crate) use crate::api::speech_text_upload::FIRST_PARTY_AUDIO_UPLOAD_LIMIT_BYTES as AUDIO_UPLOAD_LIMIT_BYTES;
pub(crate) use handlers::batch_asr_stage_executor;

pub fn router() -> Router<AppState> {
    const CANONICAL_UNIFIED_COLLECTION: &str = "/speech-to-text/jobs";
    const CANONICAL_UNIFIED_MEMBER: &str = "/speech-to-text/jobs/{record_id}";
    const CANONICAL_UNIFIED_AUDIO: &str = "/speech-to-text/jobs/{record_id}/audio";
    const CANONICAL_UNIFIED_RERUNS: &str = "/speech-to-text/jobs/{record_id}/reruns";
    const CANONICAL_UNIFIED_CANCEL: &str = "/speech-to-text/jobs/{record_id}/cancel";
    const CANONICAL_UNIFIED_SUMMARY_REGENERATE: &str =
        "/speech-to-text/jobs/{record_id}/summary/regenerate";

    Router::new()
        .merge(realtime::router())
        .route(
            CANONICAL_UNIFIED_COLLECTION,
            get(unified_read::list_jobs)
                .post(unified_write::create_job)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            CANONICAL_UNIFIED_MEMBER,
            get(unified_read::get_job)
                .patch(unified_write::update_job)
                .put(unified_write::update_job)
                .delete(unified_write::delete_job),
        )
        .route(CANONICAL_UNIFIED_AUDIO, get(unified_read::get_job_audio))
        .route(CANONICAL_UNIFIED_RERUNS, post(unified_write::rerun_job))
        .route(CANONICAL_UNIFIED_CANCEL, post(unified_write::cancel_job))
        .route(
            CANONICAL_UNIFIED_SUMMARY_REGENERATE,
            post(unified_write::regenerate_job_summary),
        )
}

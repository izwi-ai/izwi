//! OpenAI-compatible audio resources.

pub mod diarizations;
pub mod speech;
pub mod transcriptions;

use axum::{extract::DefaultBodyLimit, routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    let audio_upload_limit_bytes = resolve_audio_upload_limit_bytes();

    Router::new()
        .route(
            "/audio/speech",
            post(speech::speech).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        .route(
            "/audio/transcriptions",
            post(transcriptions::transcriptions)
                .layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        .route(
            "/audio/diarizations",
            post(diarizations::diarizations).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        // Legacy alias kept for older clients. Canonical route is /audio/diarizations.
        .route(
            "/audio/diarize",
            post(diarizations::diarizations).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
}

pub fn resolve_audio_upload_limit_bytes() -> usize {
    const MIB: usize = 1024 * 1024;
    const OPENAI_DEFAULT_MB: usize = 25;
    std::env::var("IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|mb| *mb > 0)
        .unwrap_or(OPENAI_DEFAULT_MB)
        .saturating_mul(MIB)
}

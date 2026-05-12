//! OpenAI-compatible audio resources.

pub mod speech;
pub mod transcriptions;

use axum::{Router, extract::DefaultBodyLimit, routing::post};

use crate::api::openai::compat::{OpenAiCompatibilityProfile, compatibility_profile};
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
}

pub fn resolve_audio_upload_limit_bytes() -> usize {
    const MIB: usize = 1024 * 1024;
    let default_mb = default_audio_upload_limit_mb(compatibility_profile());
    std::env::var("IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|mb| *mb > 0)
        .unwrap_or(default_mb)
        .saturating_mul(MIB)
}

fn default_audio_upload_limit_mb(profile: OpenAiCompatibilityProfile) -> usize {
    if profile.is_relaxed() { 64 } else { 25 }
}

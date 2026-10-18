//! OpenAI-compatible audio resources.

pub mod align;
pub mod speech;
pub mod transcriptions;

use axum::{extract::DefaultBodyLimit, routing::post, Router};

use crate::api::openai::compat::{compatibility_profile, OpenAiCompatibilityProfile};
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
            "/audio/align",
            post(align::align).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
}

pub fn resolve_audio_upload_limit_bytes() -> usize {
    const MIB: usize = 1024 * 1024;
    crate::env_config::tuning()
        .openai_audio_upload_limit_mb
        .unwrap_or_else(|| default_audio_upload_limit_mb(compatibility_profile()))
        .saturating_mul(MIB)
}

fn default_audio_upload_limit_mb(profile: OpenAiCompatibilityProfile) -> usize {
    if profile.is_relaxed() {
        64
    } else {
        25
    }
}

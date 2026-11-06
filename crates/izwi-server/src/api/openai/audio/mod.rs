//! OpenAI-compatible audio resources.

pub mod align;
pub mod speech;
pub mod speech_batch;
pub mod speech_stream;
pub mod transcriptions;
pub mod voices;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};

use crate::api::openai::compat::{compatibility_profile, OpenAiCompatibilityProfile};
use crate::state::AppState;

/// Multipart framing around a voice upload's 10 MiB of audio.
const MULTIPART_SLACK: usize = 1024 * 1024;

pub fn router() -> Router<AppState> {
    let audio_upload_limit_bytes = resolve_audio_upload_limit_bytes();

    Router::new()
        .route(
            "/audio/speech",
            post(speech::speech).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        .route(
            "/audio/speech/batch",
            post(speech_batch::speech_batch).layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        .route("/audio/speech/stream", get(speech_stream::speech_stream))
        .route("/audio/speech/{speech_id}", get(speech::get_speech))
        .route(
            "/audio/voices",
            get(voices::list_voices)
                .post(voices::create_voice)
                .layer(DefaultBodyLimit::max(
                    voices::VOICE_UPLOAD_LIMIT_BYTES + MULTIPART_SLACK,
                )),
        )
        .route(
            "/audio/voices/{voice_id}",
            get(voices::get_voice).delete(voices::delete_voice),
        )
        .route(
            "/audio/voice_consents",
            get(voices::list_voice_consents)
                .post(voices::create_voice_consent)
                .layer(DefaultBodyLimit::max(
                    voices::VOICE_UPLOAD_LIMIT_BYTES + MULTIPART_SLACK,
                )),
        )
        .route(
            "/audio/voice_consents/{consent_id}",
            get(voices::get_voice_consent)
                .post(voices::update_voice_consent)
                .delete(voices::delete_voice_consent),
        )
        .route("/audio/consent_phrases", get(voices::list_consent_phrases))
        .route(
            "/audio/transcriptions",
            post(transcriptions::transcriptions)
                .layer(DefaultBodyLimit::max(audio_upload_limit_bytes)),
        )
        .route(
            "/audio/translations",
            post(transcriptions::translations)
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

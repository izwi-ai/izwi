//! Persistent saved voice routes for reusable voice cloning references.

mod handlers;

use axum::{extract::DefaultBodyLimit, routing::get, Router};
use base64::Engine;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Clone)]
pub(crate) struct SavedVoiceReference {
    pub voice_id: String,
    pub reference_text: String,
    pub reference_audio_base64: String,
}

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    Router::new()
        .route(
            "/voices",
            get(handlers::list_saved_voices)
                .post(handlers::create_saved_voice)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/voices/:voice_id",
            get(handlers::get_saved_voice).delete(handlers::delete_saved_voice),
        )
        .route(
            "/voices/:voice_id/audio",
            get(handlers::get_saved_voice_audio),
        )
}

pub(crate) async fn resolve_saved_voice_reference(
    state: &AppState,
    voice_id: &str,
) -> Result<SavedVoiceReference, ApiError> {
    let saved_voice = state
        .saved_voice_store
        .get_voice(voice_id.to_string())
        .await
        .map_err(map_saved_voice_store_error)?
        .ok_or_else(|| ApiError::not_found("Saved voice not found"))?;
    let audio = state
        .saved_voice_store
        .get_audio(voice_id.to_string())
        .await
        .map_err(map_saved_voice_store_error)?
        .ok_or_else(|| ApiError::not_found("Saved voice audio not found"))?;

    Ok(SavedVoiceReference {
        voice_id: saved_voice.id,
        reference_text: saved_voice.reference_text,
        reference_audio_base64: base64::engine::general_purpose::STANDARD.encode(audio.audio_bytes),
    })
}

fn map_saved_voice_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Saved voice storage error: {err}"))
}

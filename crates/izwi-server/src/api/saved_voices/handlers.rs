use axum::{
    body::Body,
    extract::{Json, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde::{Deserialize, Serialize};

use crate::api::audio_payload::{
    decode_base64_audio_payload, inspect_audio_payload_with_diagnostics,
};
use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::error::{ApiError, ApiJson};
use crate::saved_voice_store::{
    NewSavedVoice, SavedVoice, SavedVoiceListCursor, SavedVoiceSourceRouteKind, SavedVoiceSummary,
    StoredSavedVoiceAudio,
};
use crate::state::AppState;

const SAVED_VOICE_LIST_LIMIT: usize = 500;

#[derive(Debug, Deserialize, Default)]
pub(crate) struct SavedVoiceAudioQuery {
    #[serde(default)]
    download: bool,
}

#[derive(Debug, Serialize)]
pub struct SavedVoiceListResponse {
    pub voices: Vec<SavedVoiceSummary>,
    pub pagination: CursorPagination,
}

#[derive(Debug, Serialize)]
pub struct DeleteSavedVoiceResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CreateSavedVoiceRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub reference_text: Option<String>,
    #[serde(default)]
    pub audio_base64: Option<String>,
    #[serde(default)]
    pub audio_mime_type: Option<String>,
    #[serde(default)]
    pub audio_filename: Option<String>,
    #[serde(default)]
    pub source_route_kind: Option<SavedVoiceSourceRouteKind>,
    #[serde(default)]
    pub source_record_id: Option<String>,
    /// The caller confirms it may clone this voice: it is theirs, or its
    /// owner agreed. Required for every voice that is not designed.
    #[serde(default)]
    pub consent: Option<bool>,
}

/// Fields spelled out: `serde(flatten)` breaks numbers in query strings.
#[derive(Debug, Default, Deserialize)]
pub struct SavedVoiceListQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub cursor: Option<String>,
    /// Case-insensitive search over names and reference text.
    #[serde(default)]
    pub q: Option<String>,
}

pub async fn list_saved_voices(
    State(state): State<AppState>,
    Query(query): Query<SavedVoiceListQuery>,
) -> Result<Json<SavedVoiceListResponse>, ApiError> {
    let page = CursorPaginationQuery {
        limit: query.limit,
        cursor: query.cursor,
    };
    let limit = page.resolved_limit(SAVED_VOICE_LIST_LIMIT, 1000);
    let cursor = page.decode_cursor::<SavedVoiceListCursor>()?;
    let (voices, next_cursor) = state
        .saved_voice_store
        .list_voices_page(limit, cursor, query.q.as_deref())
        .await
        .map_err(map_store_error)?;

    let has_more = next_cursor.is_some();
    Ok(Json(SavedVoiceListResponse {
        voices,
        pagination: CursorPagination {
            next_cursor: next_cursor.map(|value| encode_cursor(&value)),
            has_more,
            limit,
        },
    }))
}

pub async fn get_saved_voice(
    State(state): State<AppState>,
    Path(voice_id): Path<String>,
) -> Result<Json<SavedVoice>, ApiError> {
    let voice = state
        .saved_voice_store
        .get_voice(voice_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Saved voice not found"))?;

    Ok(Json(voice))
}

pub async fn get_saved_voice_audio(
    State(state): State<AppState>,
    Path(voice_id): Path<String>,
    Query(query): Query<SavedVoiceAudioQuery>,
) -> Result<Response, ApiError> {
    let audio = state
        .saved_voice_store
        .get_audio(voice_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Saved voice audio not found"))?;

    Ok(audio_response(audio, query.download))
}

/// A saved voice is a voice-clone reference, so it is held to the reference
/// limit at upload: a longer clip was accepted and then failed every use
/// (TPIPE-23).
fn validate_saved_voice_reference(duration_secs: f32) -> Result<(), ApiError> {
    let limit = izwi_core::catalog::SpeechModelCapabilities::MAX_REFERENCE_AUDIO_SECS;
    if duration_secs > limit as f32 {
        return Err(ApiError::bad_request(format!(
            "Reference audio is {duration_secs:.1} s; voice references are limited to {limit} s. Trim it to the clearest {limit} s or less."
        )));
    }
    Ok(())
}

pub async fn create_saved_voice(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<CreateSavedVoiceRequest>,
) -> Result<Json<SavedVoice>, ApiError> {
    let name = required_trimmed(req.name.as_deref(), "name")?;
    let reference_text = required_trimmed(req.reference_text.as_deref(), "reference_text")?;
    // A designed voice is synthetic; any other is someone's real voice.
    if req.source_route_kind != Some(SavedVoiceSourceRouteKind::VoiceDesign)
        && req.consent != Some(true)
    {
        return Err(ApiError::bad_request(
            "Confirm you have permission to clone this voice (`consent: true`): it is yours, or its owner agreed.",
        ));
    }

    let audio_payload = req
        .audio_base64
        .as_deref()
        .ok_or_else(|| ApiError::bad_request("Missing required `audio_base64` field."))?;
    let audio_payload = decode_base64_audio_payload(audio_payload)?;
    let inspection = inspect_audio_payload_with_diagnostics("saved_voices.create", &audio_payload)?;
    validate_saved_voice_reference(inspection.duration_secs)?;
    let audio_mime_type = req
        .audio_mime_type
        .or_else(|| audio_payload.content_type_hint().map(str::to_string))
        .unwrap_or_else(|| "audio/wav".to_string());

    let voice = state
        .saved_voice_store
        .create_voice(NewSavedVoice {
            name,
            reference_text,
            audio_mime_type,
            audio_filename: req.audio_filename,
            audio_bytes: audio_payload.bytes,
            source_route_kind: req.source_route_kind,
            source_record_id: req.source_record_id,
        })
        .await
        .map_err(map_store_error)?;

    Ok(Json(voice))
}

pub async fn delete_saved_voice(
    State(state): State<AppState>,
    Path(voice_id): Path<String>,
) -> Result<Json<DeleteSavedVoiceResponse>, ApiError> {
    let deleted = state
        .saved_voice_store
        .delete_voice(voice_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Saved voice not found"));
    }

    Ok(Json(DeleteSavedVoiceResponse {
        id: voice_id,
        deleted: true,
    }))
}

fn required_trimmed(raw: Option<&str>, field_name: &str) -> Result<String, ApiError> {
    let trimmed = raw.unwrap_or("").trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Missing required `{field_name}` field."
        )));
    }
    Ok(trimmed.to_string())
}

fn audio_response(audio: StoredSavedVoiceAudio, as_attachment: bool) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    let disposition = if as_attachment {
        audio
            .audio_filename
            .as_deref()
            .map(|filename| format!("attachment; filename=\"{}\"", filename.replace('"', "")))
            .unwrap_or_else(|| "attachment".to_string())
    } else if let Some(filename) = audio.audio_filename.as_deref() {
        format!("inline; filename=\"{}\"", filename.replace('"', ""))
    } else {
        "inline".to_string()
    };

    if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
        response = response.header(header::CONTENT_DISPOSITION, value);
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Saved voice storage error")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TPIPE-23: saved voices and synthesis share the 30 s reference limit.
    #[test]
    fn saved_voices_are_held_to_the_reference_limit_at_upload() {
        assert!(validate_saved_voice_reference(12.0).is_ok());
        assert!(validate_saved_voice_reference(30.0).is_ok());
        let error = validate_saved_voice_reference(45.2).unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message.contains("45.2 s") && error.message.contains("30 s"));
    }
}

use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State},
    http::header,
    response::Response,
};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;
use crate::voice_defaults::DEFAULT_VOICE_AGENT_SYSTEM_PROMPT;
use crate::voice_observation_store::VoiceObservation;
use crate::voice_store::{
    CreateVoiceSessionRequest as StoreCreateVoiceSessionRequest, VoiceProfile, VoiceSessionDetail,
    VoiceSessionSummary, VoiceTurnRecord,
};

#[derive(Debug, Deserialize)]
pub struct UpdateVoiceProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub observational_memory_enabled: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ListVoiceSessionsQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CreateVoiceSessionApiRequest {
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateVoiceSessionRequest {
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub ended: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ExportVoiceSessionQuery {
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListVoiceObservationsQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
pub struct VoiceProfileResponse {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub observational_memory_enabled: bool,
    pub created_at: u64,
    pub updated_at: u64,
    pub default_system_prompt: &'static str,
}

#[derive(Debug, Serialize)]
pub struct VoiceSessionExport<'a> {
    pub session: &'a VoiceSessionSummary,
    pub turns: &'a [VoiceTurnRecord],
    pub transcript: String,
}

pub async fn get_voice_profile(
    State(state): State<AppState>,
) -> Result<Json<VoiceProfileResponse>, ApiError> {
    let profile = state
        .voice_store
        .get_default_profile()
        .await
        .map_err(map_store_error)?;
    Ok(Json(map_profile_response(profile)))
}

pub async fn update_voice_profile(
    State(state): State<AppState>,
    Json(req): Json<UpdateVoiceProfileRequest>,
) -> Result<Json<VoiceProfileResponse>, ApiError> {
    let profile = state
        .voice_store
        .update_default_profile(
            req.name,
            req.system_prompt,
            req.observational_memory_enabled,
        )
        .await
        .map_err(map_store_error)?;
    Ok(Json(map_profile_response(profile)))
}

pub async fn list_voice_sessions(
    State(state): State<AppState>,
    Query(query): Query<ListVoiceSessionsQuery>,
) -> Result<Json<Vec<VoiceSessionSummary>>, ApiError> {
    let sessions = state
        .voice_store
        .list_sessions(query.limit.unwrap_or(25).clamp(1, 100))
        .await
        .map_err(map_store_error)?;
    Ok(Json(sessions))
}

pub async fn create_voice_session(
    State(state): State<AppState>,
    Json(req): Json<CreateVoiceSessionApiRequest>,
) -> Result<Json<VoiceSessionDetail>, ApiError> {
    let profile = if let Some(profile_id) = req.profile_id {
        state
            .voice_store
            .get_profile(profile_id)
            .await
            .map_err(map_store_error)?
            .ok_or_else(|| ApiError::not_found("Voice profile not found"))?
    } else {
        state
            .voice_store
            .get_default_profile()
            .await
            .map_err(map_store_error)?
    };
    let mode = req.mode.unwrap_or_else(|| "modular".to_string());
    let system_prompt = req
        .system_prompt
        .unwrap_or_else(|| profile.system_prompt.clone());

    let session = state
        .voice_store
        .create_session(StoreCreateVoiceSessionRequest {
            profile_id: profile.id,
            mode,
            system_prompt,
        })
        .await
        .map_err(map_store_error)?;
    let detail = state
        .voice_store
        .get_session(session.id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    Ok(Json(detail))
}

pub async fn list_voice_observations(
    State(state): State<AppState>,
    Query(query): Query<ListVoiceObservationsQuery>,
) -> Result<Json<Vec<VoiceObservation>>, ApiError> {
    let profile = state
        .voice_store
        .get_default_profile()
        .await
        .map_err(map_store_error)?;
    let observations = state
        .voice_observation_store
        .list_active(profile.id, query.limit.unwrap_or(25).clamp(1, 100))
        .await
        .map_err(map_observation_store_error)?;
    Ok(Json(observations))
}

pub async fn delete_voice_observation(
    State(state): State<AppState>,
    Path(observation_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let deleted = state
        .voice_observation_store
        .forget_observation(observation_id.clone())
        .await
        .map_err(map_observation_store_error)?;
    if !deleted {
        return Err(ApiError::not_found("Voice observation not found"));
    }
    Ok(Json(serde_json::json!({
        "id": observation_id,
        "deleted": true,
    })))
}

pub async fn clear_voice_observations(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let profile = state
        .voice_store
        .get_default_profile()
        .await
        .map_err(map_store_error)?;
    let cleared = state
        .voice_observation_store
        .clear_profile(profile.id)
        .await
        .map_err(map_observation_store_error)?;
    Ok(Json(serde_json::json!({
        "cleared": cleared,
    })))
}

pub async fn get_voice_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<VoiceSessionDetail>, ApiError> {
    let session = state
        .voice_store
        .get_session(session_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    Ok(Json(session))
}

pub async fn list_voice_session_turns(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<VoiceTurnRecord>>, ApiError> {
    let turns = state
        .voice_store
        .list_turns(session_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    Ok(Json(turns))
}

pub async fn update_voice_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(req): Json<UpdateVoiceSessionRequest>,
) -> Result<Json<VoiceSessionDetail>, ApiError> {
    if let Some(system_prompt) = req.system_prompt {
        if system_prompt.trim().is_empty() {
            return Err(ApiError::bad_request(
                "Voice session system_prompt cannot be empty",
            ));
        }
        let updated = state
            .voice_store
            .update_session_system_prompt(session_id.clone(), system_prompt)
            .await
            .map_err(map_store_error)?;
        if !updated {
            return Err(ApiError::not_found("Voice session not found"));
        }
    }

    if let Some(ended) = req.ended {
        if !ended {
            return Err(ApiError::bad_request(
                "Voice sessions can only be ended through this endpoint",
            ));
        }
        let updated = state
            .voice_store
            .end_session(session_id.clone())
            .await
            .map_err(map_store_error)?;
        if !updated {
            return Err(ApiError::not_found("Voice session not found"));
        }
    }

    let session = state
        .voice_store
        .get_session(session_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    Ok(Json(session))
}

pub async fn end_voice_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<VoiceSessionDetail>, ApiError> {
    let updated = state
        .voice_store
        .end_session(session_id.clone())
        .await
        .map_err(map_store_error)?;
    if !updated {
        return Err(ApiError::not_found("Voice session not found"));
    }

    let session = state
        .voice_store
        .get_session(session_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    Ok(Json(session))
}

pub async fn export_voice_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Query(query): Query<ExportVoiceSessionQuery>,
) -> Result<Response<Body>, ApiError> {
    let session = state
        .voice_store
        .get_session(session_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Voice session not found"))?;
    let format = query
        .format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();
    let transcript = format_voice_session_transcript(&session);

    match format.as_str() {
        "json" => json_response(&VoiceSessionExport {
            session: &session.session,
            turns: &session.turns,
            transcript,
        }),
        "text" | "txt" => Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(transcript))
            .map_err(|err| ApiError::internal(format!("Failed building export response: {err}"))),
        _ => Err(ApiError::bad_request(
            "Unsupported voice session export format. Supported: json, text",
        )),
    }
}

pub async fn delete_voice_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let deleted = state
        .voice_store
        .delete_session(session_id.clone())
        .await
        .map_err(map_store_error)?;
    if !deleted {
        return Err(ApiError::not_found("Voice session not found"));
    }
    Ok(Json(serde_json::json!({
        "id": session_id,
        "deleted": true,
    })))
}

fn map_profile_response(profile: VoiceProfile) -> VoiceProfileResponse {
    VoiceProfileResponse {
        id: profile.id,
        name: profile.name,
        system_prompt: profile.system_prompt,
        observational_memory_enabled: profile.observational_memory_enabled,
        created_at: profile.created_at,
        updated_at: profile.updated_at,
        default_system_prompt: DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
    }
}

fn json_response<T: Serialize>(body: &T) -> Result<Response<Body>, ApiError> {
    let body = serde_json::to_string(body)
        .map_err(|err| ApiError::internal(format!("Failed serializing JSON response: {err}")))?;
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .map_err(|err| ApiError::internal(format!("Failed building JSON response: {err}")))
}

fn format_voice_session_transcript(detail: &VoiceSessionDetail) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Voice session {}", detail.session.id));

    for turn in &detail.turns {
        if let Some(user_text) = turn.user_text.as_deref().filter(|text| !text.is_empty()) {
            lines.push(format!("User: {user_text}"));
        }
        if let Some(assistant_text) = turn
            .assistant_text
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            lines.push(format!("Assistant: {assistant_text}"));
        }
    }

    lines.join("\n")
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    let message = err.to_string();
    if message.contains("Invalid voice mode") || message.contains("Missing required") {
        return ApiError::bad_request(message);
    }

    ApiError::internal(format!("Voice storage error: {err}"))
}

fn map_observation_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Voice memory storage error: {err}"))
}

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;

use crate::error::ApiError;
use crate::state::AppState;
use crate::voice_defaults::DEFAULT_VOICE_AGENT_SYSTEM_PROMPT;
use crate::voice_store::{VoiceProfile, VoiceSessionDetail, VoiceSessionSummary};

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

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Voice storage error: {err}"))
}

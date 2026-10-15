use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiJson};
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct UserPreferencesResponse {
    pub analytics_opt_in: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateAnalyticsPreferenceRequest {
    pub opt_in: bool,
}

pub async fn get_preferences(
    State(state): State<AppState>,
) -> Result<Json<UserPreferencesResponse>, ApiError> {
    let stored = state
        .onboarding_store
        .get_state()
        .await
        .map_err(map_store_error)?;

    Ok(Json(UserPreferencesResponse {
        analytics_opt_in: stored.analytics_opt_in,
    }))
}

pub async fn update_analytics_preference(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<UpdateAnalyticsPreferenceRequest>,
) -> Result<Json<UserPreferencesResponse>, ApiError> {
    let stored = state
        .onboarding_store
        .set_analytics_opt_in(request.opt_in)
        .await
        .map_err(map_store_error)?;

    Ok(Json(UserPreferencesResponse {
        analytics_opt_in: stored.analytics_opt_in,
    }))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Preferences storage error: {err}"))
}

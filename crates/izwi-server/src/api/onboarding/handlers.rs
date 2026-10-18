use axum::{extract::State, Json};
use serde::Serialize;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct OnboardingStateResponse {
    pub completed: bool,
    pub completed_at: Option<u64>,
    pub analytics_opt_in: bool,
}

pub async fn get_onboarding_state(
    State(state): State<AppState>,
) -> Result<Json<OnboardingStateResponse>, ApiError> {
    let stored = state
        .onboarding_store
        .get_state()
        .await
        .map_err(map_store_error)?;
    Ok(Json(OnboardingStateResponse {
        completed: stored.completed,
        completed_at: stored.completed_at,
        analytics_opt_in: stored.analytics_opt_in,
    }))
}

pub async fn complete_onboarding(
    State(state): State<AppState>,
) -> Result<Json<OnboardingStateResponse>, ApiError> {
    let stored = state
        .onboarding_store
        .mark_completed()
        .await
        .map_err(map_store_error)?;
    Ok(Json(OnboardingStateResponse {
        completed: stored.completed,
        completed_at: stored.completed_at,
        analytics_opt_in: stored.analytics_opt_in,
    }))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Onboarding storage error")
}

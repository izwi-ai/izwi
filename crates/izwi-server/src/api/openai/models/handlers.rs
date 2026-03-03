//! OpenAI-compatible model resource handlers.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::parse_model_variant;

#[derive(Debug, Serialize)]
pub struct OpenAiModelsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModel>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

pub async fn list_models_openai(
    State(state): State<AppState>,
) -> Result<Json<OpenAiModelsResponse>, ApiError> {
    let created = now_unix_secs();
    let data = state
        .runtime
        .list_models()
        .await
        .into_iter()
        .filter(|model| model.enabled)
        .map(|model| OpenAiModel {
            id: model.variant.dir_name().to_string(),
            object: "model",
            created,
            owned_by: "agentem",
            root: None,
            parent: None,
        })
        .collect();

    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

pub async fn get_model_openai(Path(model): Path<String>) -> Result<Json<OpenAiModel>, ApiError> {
    let variant =
        parse_model_variant(&model).map_err(|err| ApiError::bad_request(err.to_string()))?;
    if !variant.is_enabled() {
        return Err(ApiError::not_found("Model not found"));
    }

    Ok(Json(OpenAiModel {
        id: variant.dir_name().to_string(),
        object: "model",
        created: now_unix_secs(),
        owned_by: "agentem",
        root: None,
        parent: None,
    }))
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

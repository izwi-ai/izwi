//! OpenAI-compatible model resource handlers.

use std::time::UNIX_EPOCH;

use axum::{
    extract::{Path, State},
    Json,
};
use izwi_core::{ModelInfo, ModelStatus, ModelVariant};
use serde::Serialize;

use crate::error::ApiError;
use crate::gateway::GatewayState;
use crate::state::AppState;

/// `owned_by` for every model this server lists.
const OWNER: &str = "izwi";

#[derive(Debug, Serialize)]
pub struct OpenAiModelsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModel>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: &'static str,
    /// When the model's files arrived on this host (seconds since the
    /// epoch); `0` when unknown.
    pub created: u64,
    pub owned_by: &'static str,
}

impl OpenAiModel {
    fn new(variant: ModelVariant, created: u64) -> Self {
        Self {
            id: variant.dir_name().to_string(),
            object: "model",
            created,
            owned_by: OWNER,
        }
    }
}

/// A model this server can run now: enabled, and downloaded or loaded.
fn is_available(model: &ModelInfo) -> bool {
    model.enabled
        && matches!(
            model.status,
            ModelStatus::Downloaded | ModelStatus::Loading | ModelStatus::Ready
        )
}

fn listed(model: &ModelInfo) -> OpenAiModel {
    let created = model
        .local_path
        .as_ref()
        .and_then(|path| std::fs::metadata(path).ok()?.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_secs());
    OpenAiModel::new(model.variant, created)
}

fn model_not_found(model: &str) -> ApiError {
    ApiError::model_not_found(format!("The model `{model}` does not exist"))
}

pub async fn list_models_openai(
    State(state): State<AppState>,
) -> Result<Json<OpenAiModelsResponse>, ApiError> {
    let data = state
        .runtime
        .list_models()
        .await
        .iter()
        .filter(|model| is_available(model))
        .map(listed)
        .collect();

    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

pub async fn get_model_openai(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<OpenAiModel>, ApiError> {
    let variant = izwi_core::parse_model_variant(&model).map_err(|_| model_not_found(&model))?;
    state
        .runtime
        .list_models()
        .await
        .iter()
        .find(|info| info.variant == variant && is_available(info))
        .map(|info| Json(listed(info)))
        .ok_or_else(|| model_not_found(&model))
}

/// The gateway lists the one model its deployment serves.
pub async fn gateway_list_models(State(state): State<GatewayState>) -> Json<OpenAiModelsResponse> {
    Json(OpenAiModelsResponse {
        object: "list",
        data: vec![OpenAiModel::new(state.chat_dispatcher.public_model(), 0)],
    })
}

pub async fn gateway_get_model(
    State(state): State<GatewayState>,
    Path(model): Path<String>,
) -> Result<Json<OpenAiModel>, ApiError> {
    let public = state.chat_dispatcher.public_model();
    match izwi_core::parse_model_variant(&model) {
        Ok(variant) if variant == public => Ok(Json(OpenAiModel::new(public, 0))),
        _ => Err(model_not_found(&model)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_enabled_downloaded_or_loaded_models_are_listed() {
        let model = |status| ModelInfo {
            status,
            ..ModelInfo::new(ModelVariant::Lfm2512BInstructGguf)
        };
        for status in [
            ModelStatus::Downloaded,
            ModelStatus::Loading,
            ModelStatus::Ready,
        ] {
            assert!(is_available(&model(status)), "{status:?}");
        }
        for status in [
            ModelStatus::NotDownloaded,
            ModelStatus::Downloading,
            ModelStatus::Error,
        ] {
            assert!(!is_available(&model(status)), "{status:?}");
        }
        let disabled = ModelInfo {
            enabled: false,
            ..model(ModelStatus::Ready)
        };
        assert!(!is_available(&disabled));

        let listed = serde_json::to_value(listed(&model(ModelStatus::Ready))).unwrap();
        assert_eq!(
            listed,
            serde_json::json!({
                "id": ModelVariant::Lfm2512BInstructGguf.dir_name(),
                "object": "model",
                "created": 0,
                "owned_by": "izwi",
            })
        );
    }

    #[test]
    fn unknown_models_are_openai_model_not_found() {
        let error = model_not_found("gpt-imaginary");
        assert_eq!(error.status, axum::http::StatusCode::NOT_FOUND);
        let body = error.openai_body();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }
}

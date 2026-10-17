//! OpenAI-compatible model resource handlers for this process's runtime.
//! The resource types are izwi-gateway's, which the gateway's routes share.

use std::time::UNIX_EPOCH;

use axum::{
    extract::{Path, State},
    Json,
};
use izwi_core::{ModelInfo, ModelStatus};

pub(crate) use izwi_gateway::api::openai::models::handlers::*;

use crate::error::ApiError;
use crate::state::AppState;

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

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::ModelVariant;

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
}

//! OpenAI-compatible model resource handlers.

use axum::{
    extract::{Path, State},
    Json,
};
use izwi_model_catalog::ModelVariant;
use serde::Serialize;

use crate::error::ApiError;
use crate::gateway::GatewayState;

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
    pub fn new(variant: ModelVariant, created: u64) -> Self {
        Self {
            id: variant.dir_name().to_string(),
            object: "model",
            created,
            owned_by: OWNER,
        }
    }
}

pub fn model_not_found(model: &str) -> ApiError {
    ApiError::model_not_found(format!("The model `{model}` does not exist"))
}

/// The gateway lists the one model its deployment serves.
pub async fn gateway_list_models(State(state): State<GatewayState>) -> Json<OpenAiModelsResponse> {
    Json(OpenAiModelsResponse {
        object: "list",
        data: state
            .chat_dispatchers
            .iter()
            .map(|dispatcher| OpenAiModel::new(dispatcher.public_model(), 0))
            .collect(),
    })
}

pub async fn gateway_get_model(
    State(state): State<GatewayState>,
    Path(model): Path<String>,
) -> Result<Json<OpenAiModel>, ApiError> {
    let served = state
        .chat_dispatchers
        .iter()
        .map(|dispatcher| dispatcher.public_model());
    match izwi_model_catalog::parse_model_variant(&model) {
        Ok(variant) if served.into_iter().any(|public| public == variant) => {
            Ok(Json(OpenAiModel::new(variant, 0)))
        }
        _ => Err(model_not_found(&model)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_models_are_openai_model_not_found() {
        let error = model_not_found("gpt-imaginary");
        assert_eq!(error.status, axum::http::StatusCode::NOT_FOUND);
        let body = error.openai_body();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
    }
}

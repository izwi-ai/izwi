use axum::{
    extract::{Extension, Path, Query, Request, State},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::api::diarization::handlers as diarization_handlers;
use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MutationJobKind {
    All,
    Transcription,
    Diarization,
}

#[derive(Debug, Deserialize)]
pub struct UnifiedMutationQuery {
    #[serde(default)]
    job_kind: Option<String>,
}

pub async fn create_job(
    state: State<AppState>,
    ctx: Extension<RequestContext>,
    Query(query): Query<UnifiedMutationQuery>,
    req: Request,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    match kind {
        MutationJobKind::Diarization => diarization_handlers::create_record(state, ctx, req).await,
        MutationJobKind::All | MutationJobKind::Transcription => {
            super::handlers::create_record(state, ctx, req).await
        }
    }
}

pub async fn update_job(
    state: State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedMutationQuery>,
    Json(req): Json<diarization_handlers::UpdateDiarizationRecordRequest>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    match resolve_diarization_only_kind(&state, record_id.as_str(), kind).await? {
        MutationJobKind::Diarization => {
            let response = diarization_handlers::update_record(
                state,
                Path(record_id),
                Json(req),
            )
            .await?;
            Ok(response.into_response())
        }
        MutationJobKind::All | MutationJobKind::Transcription => Err(ApiError::bad_request(
            "Speaker override updates are supported only for diarization jobs.",
        )),
    }
}

pub async fn rerun_job(
    state: State<AppState>,
    ctx: Extension<RequestContext>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedMutationQuery>,
    Json(req): Json<diarization_handlers::RerunDiarizationRecordRequest>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    match resolve_diarization_only_kind(&state, record_id.as_str(), kind).await? {
        MutationJobKind::Diarization => {
            diarization_handlers::rerun_record(state, ctx, Path(record_id), Json(req)).await
        }
        MutationJobKind::All | MutationJobKind::Transcription => Err(ApiError::bad_request(
            "Reruns are supported only for diarization jobs.",
        )),
    }
}

pub async fn cancel_job(
    state: State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedMutationQuery>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    match resolve_diarization_only_kind(&state, record_id.as_str(), kind).await? {
        MutationJobKind::Diarization => {
            let response = diarization_handlers::cancel_record(state, Path(record_id)).await?;
            Ok(response.into_response())
        }
        MutationJobKind::All | MutationJobKind::Transcription => Err(ApiError::bad_request(
            "Cancellation is currently supported only for diarization jobs.",
        )),
    }
}

pub async fn regenerate_job_summary(
    state: State<AppState>,
    ctx: Extension<RequestContext>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedMutationQuery>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    let resolved = match kind {
        MutationJobKind::All => detect_existing_job_kind(&state, record_id.as_str()).await?,
        MutationJobKind::Transcription => MutationJobKind::Transcription,
        MutationJobKind::Diarization => MutationJobKind::Diarization,
    };

    match resolved {
        MutationJobKind::Transcription => {
            let response = super::handlers::regenerate_summary(state, ctx, Path(record_id)).await?;
            Ok(response.into_response())
        }
        MutationJobKind::Diarization => {
            let response =
                diarization_handlers::regenerate_summary(state, ctx, Path(record_id)).await?;
            Ok(response.into_response())
        }
        MutationJobKind::All => Err(ApiError::not_found("Speech text job record not found")),
    }
}

pub async fn delete_job(
    state: State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedMutationQuery>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind(query.job_kind.as_deref(), true)?;
    let resolved = match kind {
        MutationJobKind::All => detect_existing_job_kind(&state, record_id.as_str()).await?,
        MutationJobKind::Transcription => MutationJobKind::Transcription,
        MutationJobKind::Diarization => MutationJobKind::Diarization,
    };

    match resolved {
        MutationJobKind::Transcription => {
            let response = super::handlers::delete_record(state, Path(record_id)).await?;
            Ok(response.into_response())
        }
        MutationJobKind::Diarization => {
            let response = diarization_handlers::delete_record(state, Path(record_id)).await?;
            Ok(response.into_response())
        }
        MutationJobKind::All => Err(ApiError::not_found("Speech text job record not found")),
    }
}

fn parse_job_kind(raw: Option<&str>, allow_all: bool) -> Result<MutationJobKind, ApiError> {
    let Some(value) = raw.map(|value| value.trim().to_ascii_lowercase()) else {
        return Ok(if allow_all {
            MutationJobKind::All
        } else {
            MutationJobKind::Transcription
        });
    };

    if value.is_empty() {
        return Ok(if allow_all {
            MutationJobKind::All
        } else {
            MutationJobKind::Transcription
        });
    }

    match value.as_str() {
        "all" if allow_all => Ok(MutationJobKind::All),
        "transcription" => Ok(MutationJobKind::Transcription),
        "diarization" => Ok(MutationJobKind::Diarization),
        _ => Err(ApiError::bad_request(
            "Invalid `job_kind`. Supported values: all, transcription, diarization.",
        )),
    }
}

async fn detect_existing_job_kind(
    state: &AppState,
    record_id: &str,
) -> Result<MutationJobKind, ApiError> {
    if state
        .transcription_store
        .get_record(record_id.to_string())
        .await
        .map_err(map_store_error)?
        .is_some()
    {
        return Ok(MutationJobKind::Transcription);
    }

    if state
        .diarization_store
        .get_record(record_id.to_string())
        .await
        .map_err(map_store_error)?
        .is_some()
    {
        return Ok(MutationJobKind::Diarization);
    }

    Ok(MutationJobKind::All)
}

async fn resolve_diarization_only_kind(
    state: &AppState,
    record_id: &str,
    requested: MutationJobKind,
) -> Result<MutationJobKind, ApiError> {
    match requested {
        MutationJobKind::Diarization => Ok(MutationJobKind::Diarization),
        MutationJobKind::Transcription => Ok(MutationJobKind::Transcription),
        MutationJobKind::All => detect_existing_job_kind(state, record_id).await,
    }
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Storage error: {err}"))
}

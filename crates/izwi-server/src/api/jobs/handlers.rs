use axum::{
    extract::{Path, State},
    Extension, Json,
};
use izwi_hooks::{
    AuditCategory, AuditEvent, AuditOutcome, EnterpriseAction, HookMetadata, ResourceDescriptor,
    ResourceKind,
};
use serde::Serialize;
use tracing::warn;

use crate::{
    api::request_context::RequestContext,
    batch_runtime::types::{
        JobStage, RuntimeArtifact, RuntimeJob, RuntimeJobKind, RuntimeJobStatus,
    },
    error::ApiError,
    speech_history_store::{SpeechHistoryProcessingStatus, SpeechRouteKind},
    state::AppState,
    transcription_store::TranscriptionProcessingStatus,
};

#[derive(Debug, Serialize)]
pub struct RuntimeJobTraceResponse {
    pub job: RuntimeJob,
    pub stages: Vec<JobStage>,
    pub artifacts: Vec<RuntimeArtifact>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeJobArtifactsResponse {
    pub job_id: String,
    pub artifacts: Vec<RuntimeArtifact>,
}

pub async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<RuntimeJobTraceResponse>, ApiError> {
    Ok(Json(load_job_trace(&state, &job_id).await?))
}

pub async fn list_job_artifacts(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<RuntimeJobArtifactsResponse>, ApiError> {
    let job = state
        .batch_runtime_store
        .get_job(&job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Runtime job not found"))?;
    let artifacts = state
        .batch_runtime_store
        .list_artifacts_for_job(&job.id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(RuntimeJobArtifactsResponse {
        job_id: job.id,
        artifacts,
    }))
}

pub async fn cancel_job(
    State(state): State<AppState>,
    context: Option<Extension<RequestContext>>,
    Path(job_id): Path<String>,
) -> Result<Json<RuntimeJobTraceResponse>, ApiError> {
    let existing = state
        .batch_runtime_store
        .get_job(&job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Runtime job not found"))?;
    if is_terminal_job_status(existing.status) {
        return Err(ApiError::bad_request("Runtime job is already terminal"));
    }

    let cancelled = state
        .batch_runtime_store
        .cancel_job(&job_id, Some("Cancelled by API request".to_string()))
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::bad_request("Runtime job is not cancellable"))?;
    update_route_projection_for_cancel(&state, &cancelled).await?;
    record_job_audit(
        &state,
        context.as_ref().map(|Extension(ctx)| ctx),
        &cancelled,
        "runtime_job.cancel",
        AuditOutcome::Success,
        None,
    )
    .await;

    Ok(Json(load_job_trace(&state, &job_id).await?))
}

pub async fn retry_job(
    State(state): State<AppState>,
    context: Option<Extension<RequestContext>>,
    Path(job_id): Path<String>,
) -> Result<Json<RuntimeJobTraceResponse>, ApiError> {
    let existing = state
        .batch_runtime_store
        .get_job(&job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Runtime job not found"))?;
    if !matches!(
        existing.status,
        RuntimeJobStatus::Failed | RuntimeJobStatus::Cancelled | RuntimeJobStatus::Expired
    ) {
        return Err(ApiError::bad_request(
            "Runtime job can only be retried from failed, cancelled, or expired status",
        ));
    }

    let retried = state
        .batch_runtime_store
        .retry_job(&job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::bad_request("Runtime job has no retryable stages"))?;
    update_route_projection_for_retry(&state, &retried).await?;
    record_job_audit(
        &state,
        context.as_ref().map(|Extension(ctx)| ctx),
        &retried,
        "runtime_job.retry",
        AuditOutcome::Success,
        None,
    )
    .await;

    Ok(Json(load_job_trace(&state, &job_id).await?))
}

async fn load_job_trace(
    state: &AppState,
    job_id: &str,
) -> Result<RuntimeJobTraceResponse, ApiError> {
    let job = state
        .batch_runtime_store
        .get_job(job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Runtime job not found"))?;
    let stages = state
        .batch_runtime_store
        .list_stages_for_job(&job.id)
        .await
        .map_err(map_store_error)?;
    let artifacts = state
        .batch_runtime_store
        .list_artifacts_for_job(&job.id)
        .await
        .map_err(map_store_error)?;

    Ok(RuntimeJobTraceResponse {
        job,
        stages,
        artifacts,
    })
}

async fn update_route_projection_for_cancel(
    state: &AppState,
    job: &RuntimeJob,
) -> Result<(), ApiError> {
    let message = job
        .cancellation_reason
        .clone()
        .unwrap_or_else(|| "Runtime job cancelled".to_string());
    match (
        job.job_kind,
        job.route_record_kind.as_deref(),
        job.route_record_id.as_deref(),
    ) {
        (
            RuntimeJobKind::AsrTranscription,
            Some("transcription" | "speaker_attributed_asr"),
            Some(record_id),
        ) => {
            let updated = state
                .transcription_store
                .update_processing_status(
                    record_id.to_string(),
                    TranscriptionProcessingStatus::Failed,
                    Some(message),
                )
                .await
                .map_err(map_store_error)?;
            warn_if_projection_missing(updated.is_some(), job);
        }
        (RuntimeJobKind::TtsSpeech, Some(route_kind), Some(record_id)) => {
            let Some(route_kind) = parse_speech_route_kind(route_kind) else {
                return Ok(());
            };
            let updated = state
                .speech_history_store
                .update_processing_status(
                    route_kind,
                    record_id.to_string(),
                    SpeechHistoryProcessingStatus::Failed,
                    Some(message),
                )
                .await
                .map_err(map_store_error)?;
            warn_if_projection_missing(updated.is_some(), job);
        }
        _ => {}
    }
    Ok(())
}

async fn update_route_projection_for_retry(
    state: &AppState,
    job: &RuntimeJob,
) -> Result<(), ApiError> {
    match (
        job.job_kind,
        job.route_record_kind.as_deref(),
        job.route_record_id.as_deref(),
    ) {
        (
            RuntimeJobKind::AsrTranscription,
            Some("transcription" | "speaker_attributed_asr"),
            Some(record_id),
        ) => {
            let updated = state
                .transcription_store
                .update_processing_status(
                    record_id.to_string(),
                    TranscriptionProcessingStatus::Pending,
                    None,
                )
                .await
                .map_err(map_store_error)?;
            warn_if_projection_missing(updated.is_some(), job);
        }
        (RuntimeJobKind::TtsSpeech, Some(route_kind), Some(record_id)) => {
            let Some(route_kind) = parse_speech_route_kind(route_kind) else {
                return Ok(());
            };
            let updated = state
                .speech_history_store
                .update_processing_status(
                    route_kind,
                    record_id.to_string(),
                    SpeechHistoryProcessingStatus::Pending,
                    None,
                )
                .await
                .map_err(map_store_error)?;
            warn_if_projection_missing(updated.is_some(), job);
        }
        _ => {}
    }
    Ok(())
}

async fn record_job_audit(
    state: &AppState,
    context: Option<&RequestContext>,
    job: &RuntimeJob,
    action: &'static str,
    outcome: AuditOutcome,
    reason: Option<&str>,
) {
    let mut metadata = HookMetadata::new();
    metadata.insert("action".to_string(), action.to_string());
    metadata.insert(
        "job_kind".to_string(),
        job.job_kind.as_db_value().to_string(),
    );
    metadata.insert("status".to_string(), job.status.as_db_value().to_string());
    if let Some(route_record_kind) = &job.route_record_kind {
        metadata.insert("route_record_kind".to_string(), route_record_kind.clone());
    }
    if let Some(route_record_id) = &job.route_record_id {
        metadata.insert("route_record_id".to_string(), route_record_id.clone());
    }
    if let Some(reason) = reason {
        metadata.insert("reason".to_string(), reason.to_string());
    }

    let resource = ResourceDescriptor {
        kind: ResourceKind::DataRecord,
        id: Some(job.id.clone()),
        model_id: job.model_id.clone(),
        tenant_id: context.and_then(|ctx| ctx.principal.tenant_id.clone()),
        attributes: metadata.clone(),
    };
    let event = AuditEvent {
        category: AuditCategory::Data,
        action: EnterpriseAction::Other(action.to_string()),
        outcome,
        principal: context.map(|ctx| ctx.principal.clone()),
        resource: Some(resource),
        correlation_id: context.map(|ctx| ctx.correlation_id.clone()),
        metadata,
    };

    if let Err(err) = state.enterprise_hooks.audit.record(event).await {
        warn!(error = %err, "Enterprise audit hook failed for runtime job action");
    }
}

fn parse_speech_route_kind(value: &str) -> Option<SpeechRouteKind> {
    match value {
        "text_to_speech" => Some(SpeechRouteKind::TextToSpeech),
        "voice_design" => Some(SpeechRouteKind::VoiceDesign),
        "voice_cloning" => Some(SpeechRouteKind::VoiceCloning),
        _ => None,
    }
}

fn is_terminal_job_status(status: RuntimeJobStatus) -> bool {
    matches!(
        status,
        RuntimeJobStatus::Completed
            | RuntimeJobStatus::Failed
            | RuntimeJobStatus::Cancelled
            | RuntimeJobStatus::Expired
    )
}

fn warn_if_projection_missing(updated: bool, job: &RuntimeJob) {
    if !updated {
        warn!(
            job_id = %job.id,
            route_record_kind = ?job.route_record_kind,
            route_record_id = ?job.route_record_id,
            "Runtime job route projection record was not found"
        );
    }
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Batch runtime storage error: {err}"))
}

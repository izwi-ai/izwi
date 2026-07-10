use axum::{
    body::Body,
    extract::{Extension, Json, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::api::request_context::RequestContext;
use crate::api::speech_history::{synthesize_record, CreateSpeechHistoryRecordRequest};
use crate::error::ApiError;
use crate::speech_history_store::{SpeechRouteKind, StoredSpeechAudio};
use crate::state::AppState;
use crate::studio_project_store::{
    NewStudioProjectFolderRecord, NewStudioProjectPronunciationRecord, NewStudioProjectRecord,
    NewStudioProjectRenderJobRecord, NewStudioProjectSegment, NewStudioProjectSnapshotRecord,
    StudioProjectExportFormat, StudioProjectFolderRecord, StudioProjectListCursor,
    StudioProjectMetaRecord, StudioProjectPronunciationRecord, StudioProjectRecord,
    StudioProjectRenderJobRecord, StudioProjectRenderJobStatus, StudioProjectSegmentRecord,
    StudioProjectSnapshotRecord, StudioProjectSummary, StudioProjectVoiceMode,
    UpdateStudioProjectRecord, UpdateStudioProjectRenderJobRecord, UpsertStudioProjectMetaRecord,
};
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::parse_tts_model_variant;

const PROJECT_LIST_LIMIT: usize = 100;

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ProjectAudioQuery {
    #[serde(default)]
    download: bool,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    segment_ids: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StudioProjectListResponse {
    pub projects: Vec<StudioProjectSummary>,
    pub pagination: CursorPagination,
}

#[derive(Debug, Serialize)]
pub struct DeleteStudioProjectResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreateStudioProjectRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: String,
    #[serde(default)]
    pub voice_mode: Option<StudioProjectVoiceMode>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateStudioProjectRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub voice_mode: Option<StudioProjectVoiceMode>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateStudioProjectSegmentRequest {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub voice_mode: Option<StudioProjectVoiceMode>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateStudioProjectSegmentRequest {
    pub text: String,
    #[serde(default)]
    pub after_segment_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SplitStudioProjectSegmentRequest {
    pub before_text: String,
    pub after_text: String,
}

#[derive(Debug, Deserialize)]
pub struct ReorderStudioProjectSegmentsRequest {
    pub ordered_segment_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct BulkDeleteStudioProjectSegmentsRequest {
    pub segment_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct StudioProjectFolderListResponse {
    pub folders: Vec<StudioProjectFolderRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateStudioProjectFolderRequest {
    pub name: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub sort_order: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct StudioProjectPronunciationListResponse {
    pub entries: Vec<StudioProjectPronunciationRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateStudioProjectPronunciationRequest {
    pub source_text: String,
    pub replacement_text: String,
    #[serde(default)]
    pub locale: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteStudioProjectPronunciationResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct StudioProjectSnapshotListResponse {
    pub snapshots: Vec<StudioProjectSnapshotRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateStudioProjectSnapshotRequest {
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpsertStudioProjectMetaRequest {
    #[serde(default)]
    pub folder_id: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub default_export_format: Option<StudioProjectExportFormat>,
    #[serde(default)]
    pub last_render_job_id: Option<String>,
    #[serde(default)]
    pub last_rendered_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct StudioProjectRenderJobListResponse {
    pub jobs: Vec<StudioProjectRenderJobRecord>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CreateStudioProjectRenderJobRequest {
    #[serde(default)]
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateStudioProjectRenderJobRequest {
    #[serde(default)]
    pub status: Option<StudioProjectRenderJobStatus>,
    #[serde(default)]
    pub error_message: Option<String>,
}

pub async fn list_studio_projects(
    State(state): State<AppState>,
    Query(query): Query<CursorPaginationQuery>,
) -> Result<Json<StudioProjectListResponse>, ApiError> {
    let limit = query.resolved_limit(PROJECT_LIST_LIMIT, 500);
    let cursor = query.decode_cursor::<StudioProjectListCursor>()?;
    let (projects, next_cursor) = state
        .studio_store
        .list_projects_page(limit, cursor)
        .await
        .map_err(map_store_error)?;
    let has_more = next_cursor.is_some();
    Ok(Json(StudioProjectListResponse {
        projects,
        pagination: CursorPagination {
            next_cursor: next_cursor.map(|value| encode_cursor(&value)),
            has_more,
            limit,
        },
    }))
}

pub async fn list_studio_project_folders(
    State(state): State<AppState>,
) -> Result<Json<StudioProjectFolderListResponse>, ApiError> {
    let folders = state
        .studio_store
        .list_folders()
        .await
        .map_err(map_store_error)?;
    Ok(Json(StudioProjectFolderListResponse { folders }))
}

pub async fn create_studio_project_folder(
    State(state): State<AppState>,
    Json(req): Json<CreateStudioProjectFolderRequest>,
) -> Result<Json<StudioProjectFolderRecord>, ApiError> {
    let name = required_trimmed(Some(req.name.as_str()), "name")?;
    let folder = state
        .studio_store
        .create_folder(NewStudioProjectFolderRecord {
            name,
            parent_id: normalize_optional_trimmed(req.parent_id),
            sort_order: req.sort_order,
        })
        .await
        .map_err(map_store_error)?;
    Ok(Json(folder))
}

pub async fn get_studio_project_meta(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<StudioProjectMetaRecord>, ApiError> {
    if let Some(meta) = state
        .studio_store
        .get_project_meta(project_id.clone())
        .await
        .map_err(map_store_error)?
    {
        return Ok(Json(meta));
    }
    let meta = state
        .studio_store
        .upsert_project_meta(project_id, UpsertStudioProjectMetaRecord::default())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(meta))
}

pub async fn upsert_studio_project_meta(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpsertStudioProjectMetaRequest>,
) -> Result<Json<StudioProjectMetaRecord>, ApiError> {
    let folder_id_update = req.folder_id.map(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    let tags = req.tags.map(|entries| {
        entries
            .into_iter()
            .map(|entry| entry.trim().to_string())
            .filter(|entry| !entry.is_empty())
            .collect::<Vec<_>>()
    });
    let last_render_job_id_update = req.last_render_job_id.map(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    let last_rendered_at_update = req.last_rendered_at.map(Some);

    let meta = state
        .studio_store
        .upsert_project_meta(
            project_id,
            UpsertStudioProjectMetaRecord {
                folder_id: folder_id_update,
                tags,
                default_export_format: req.default_export_format,
                last_render_job_id: last_render_job_id_update,
                last_rendered_at: last_rendered_at_update,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(meta))
}

pub async fn list_studio_project_pronunciations(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<StudioProjectPronunciationListResponse>, ApiError> {
    let entries = state
        .studio_store
        .list_project_pronunciations(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(StudioProjectPronunciationListResponse { entries }))
}

pub async fn create_studio_project_pronunciation(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateStudioProjectPronunciationRequest>,
) -> Result<Json<StudioProjectPronunciationRecord>, ApiError> {
    let source_text = required_trimmed(Some(req.source_text.as_str()), "source_text")?;
    let replacement_text =
        required_trimmed(Some(req.replacement_text.as_str()), "replacement_text")?;
    let entry = state
        .studio_store
        .create_project_pronunciation(
            project_id,
            NewStudioProjectPronunciationRecord {
                source_text,
                replacement_text,
                locale: normalize_optional_trimmed(req.locale),
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(entry))
}

pub async fn delete_studio_project_pronunciation(
    State(state): State<AppState>,
    Path((project_id, pronunciation_id)): Path<(String, String)>,
) -> Result<Json<DeleteStudioProjectPronunciationResponse>, ApiError> {
    let deleted = state
        .studio_store
        .delete_project_pronunciation(project_id, pronunciation_id.clone())
        .await
        .map_err(map_store_error)?;
    if !deleted {
        return Err(ApiError::not_found(
            "Studio project pronunciation not found",
        ));
    }
    Ok(Json(DeleteStudioProjectPronunciationResponse {
        id: pronunciation_id,
        deleted: true,
    }))
}

pub async fn list_studio_project_snapshots(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<StudioProjectSnapshotListResponse>, ApiError> {
    let snapshots = state
        .studio_store
        .list_project_snapshots(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(StudioProjectSnapshotListResponse { snapshots }))
}

pub async fn create_studio_project_snapshot(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateStudioProjectSnapshotRequest>,
) -> Result<Json<StudioProjectSnapshotRecord>, ApiError> {
    let snapshot = state
        .studio_store
        .create_project_snapshot(
            project_id,
            NewStudioProjectSnapshotRecord {
                label: normalize_optional_trimmed(req.label),
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(snapshot))
}

pub async fn restore_studio_project_snapshot(
    State(state): State<AppState>,
    Path((project_id, snapshot_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .restore_project_snapshot(project_id, snapshot_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project snapshot not found"))?;
    Ok(Json(project))
}

pub async fn list_studio_project_render_jobs(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<StudioProjectRenderJobListResponse>, ApiError> {
    let jobs = state
        .studio_store
        .list_project_render_jobs(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(StudioProjectRenderJobListResponse { jobs }))
}

pub async fn create_studio_project_render_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateStudioProjectRenderJobRequest>,
) -> Result<Json<StudioProjectRenderJobRecord>, ApiError> {
    let job = state
        .studio_store
        .create_project_render_job(
            project_id,
            NewStudioProjectRenderJobRecord {
                queued_segment_ids: req.queued_segment_ids,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(job))
}

pub async fn update_studio_project_render_job(
    State(state): State<AppState>,
    Path((project_id, job_id)): Path<(String, String)>,
    Json(req): Json<UpdateStudioProjectRenderJobRequest>,
) -> Result<Json<StudioProjectRenderJobRecord>, ApiError> {
    let error_message_update = req.error_message.map(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    let job = state
        .studio_store
        .update_project_render_job(
            project_id,
            job_id,
            UpdateStudioProjectRenderJobRecord {
                status: req.status,
                error_message: error_message_update,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project render job not found"))?;
    Ok(Json(job))
}

pub async fn create_studio_project(
    State(state): State<AppState>,
    Json(req): Json<CreateStudioProjectRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let req = normalize_create_request(req);
    let model_id = required_trimmed(Some(req.model_id.as_str()), "model_id")?;
    let source_text = required_trimmed(Some(req.source_text.as_str()), "source_text")?;
    let voice_mode = req.voice_mode.unwrap_or_else(|| {
        if req.saved_voice_id.is_some() {
            StudioProjectVoiceMode::Saved
        } else {
            StudioProjectVoiceMode::BuiltIn
        }
    });

    validate_project_voice_state(
        &state,
        model_id.as_str(),
        voice_mode,
        req.speaker.as_deref(),
        req.saved_voice_id.as_deref(),
    )
    .await?;

    let variant = parse_tts_model_variant(model_id.as_str())
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {err}")))?;
    let chunks = crate::api::tts_long_form::split_tts_text_for_long_form(variant, 0, &source_text);
    if chunks.is_empty() {
        return Err(ApiError::bad_request(
            "Project script did not produce any renderable segments.",
        ));
    }
    let project_speaker = voice_mode_speaker(voice_mode, req.speaker.clone());
    let project_saved_voice_id = voice_mode_saved_voice_id(voice_mode, req.saved_voice_id.clone());

    let project = state
        .studio_store
        .create_project(NewStudioProjectRecord {
            name: req
                .name
                .unwrap_or_else(|| default_project_name(source_text.as_str())),
            source_filename: req.source_filename,
            source_text,
            model_id: Some(model_id.clone()),
            voice_mode,
            speaker: project_speaker.clone(),
            saved_voice_id: project_saved_voice_id.clone(),
            speed: req.speed.map(f64::from),
            segments: chunks
                .into_iter()
                .enumerate()
                .map(|(position, text)| NewStudioProjectSegment {
                    position,
                    text,
                    // Keep segment settings unset at creation so project-level
                    // model/voice defaults continue to apply globally.
                    model_id: None,
                    voice_mode: None,
                    speaker: None,
                    saved_voice_id: None,
                })
                .collect(),
        })
        .await
        .map_err(map_store_error)?;

    Ok(Json(project))
}

pub async fn get_studio_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(project))
}

pub async fn get_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectSegmentRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    let segment = project
        .segments
        .into_iter()
        .find(|candidate| candidate.id == segment_id)
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;
    Ok(Json(segment))
}

pub async fn update_studio_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpdateStudioProjectRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    let req = normalize_update_request(req);
    let next_voice_mode = req.voice_mode.unwrap_or(existing.voice_mode);
    let next_model_id = req
        .model_id
        .clone()
        .or_else(|| existing.model_id.clone())
        .ok_or_else(|| ApiError::bad_request("Studio project is missing a model selection."))?;
    let next_speaker = match req.speaker {
        Some(value) => Some(value),
        None => existing.speaker.clone(),
    };
    let next_saved_voice_id = match req.saved_voice_id {
        Some(value) => Some(value),
        None => existing.saved_voice_id.clone(),
    };

    validate_project_voice_state(
        &state,
        next_model_id.as_str(),
        next_voice_mode,
        next_speaker.as_deref(),
        next_saved_voice_id.as_deref(),
    )
    .await?;

    let project = state
        .studio_store
        .update_project(
            project_id,
            UpdateStudioProjectRecord {
                name: req.name,
                model_id: req.model_id,
                voice_mode: Some(next_voice_mode),
                speaker: Some(voice_mode_speaker(next_voice_mode, next_speaker)),
                saved_voice_id: Some(voice_mode_saved_voice_id(
                    next_voice_mode,
                    next_saved_voice_id,
                )),
                speed: req.speed.map(|value| Some(f64::from(value))),
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    Ok(Json(project))
}

pub async fn update_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
    Json(req): Json<UpdateStudioProjectSegmentRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let req = normalize_segment_update_request(req);
    let text_update = match req.text {
        Some(text) if text.trim().is_empty() => {
            return Err(ApiError::bad_request("Segment text cannot be empty."));
        }
        Some(text) => Some(text.trim().to_string()),
        None => None,
    };
    let has_settings_update = req.model_id.is_some()
        || req.voice_mode.is_some()
        || req.speaker.is_some()
        || req.saved_voice_id.is_some();

    if text_update.is_none() && !has_settings_update {
        return Err(ApiError::bad_request(
            "Segment update requires text or settings changes.",
        ));
    }

    let base_project = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    let segment = base_project
        .segments
        .iter()
        .find(|candidate| candidate.id == segment_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    let mut updated_project: Option<StudioProjectRecord> = None;

    if has_settings_update {
        let next_model_id = req
            .model_id
            .clone()
            .or_else(|| segment.model_id.clone())
            .or_else(|| base_project.model_id.clone())
            .ok_or_else(|| ApiError::bad_request("Studio project is missing a model selection."))?;
        let next_voice_mode = req
            .voice_mode
            .or(segment.voice_mode)
            .unwrap_or(base_project.voice_mode);
        let next_speaker = req
            .speaker
            .clone()
            .or_else(|| segment.speaker.clone())
            .or_else(|| base_project.speaker.clone());
        let next_saved_voice_id = req
            .saved_voice_id
            .clone()
            .or_else(|| segment.saved_voice_id.clone())
            .or_else(|| base_project.saved_voice_id.clone());

        validate_project_voice_state(
            &state,
            next_model_id.as_str(),
            next_voice_mode,
            next_speaker.as_deref(),
            next_saved_voice_id.as_deref(),
        )
        .await?;

        updated_project = if let Some(text) = text_update {
            state
                .studio_store
                .update_segment_text_and_settings(
                    project_id.clone(),
                    segment_id.clone(),
                    text,
                    next_model_id,
                    next_voice_mode,
                    voice_mode_speaker(next_voice_mode, next_speaker),
                    voice_mode_saved_voice_id(next_voice_mode, next_saved_voice_id),
                )
                .await
                .map_err(map_store_error)?
        } else {
            state
                .studio_store
                .update_segment_settings(
                    project_id.clone(),
                    segment_id.clone(),
                    next_model_id,
                    next_voice_mode,
                    voice_mode_speaker(next_voice_mode, next_speaker),
                    voice_mode_saved_voice_id(next_voice_mode, next_saved_voice_id),
                )
                .await
                .map_err(map_store_error)?
        };
    } else if let Some(text) = text_update {
        updated_project = state
            .studio_store
            .update_segment_text(project_id.clone(), segment_id.clone(), text)
            .await
            .map_err(map_store_error)?;
    }

    let project =
        updated_project.ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;
    Ok(Json(project))
}

pub async fn create_studio_project_segment(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateStudioProjectSegmentRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let text = required_trimmed(Some(req.text.as_str()), "text")?;
    let after_segment_id = normalize_optional_trimmed(req.after_segment_id);

    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    if let Some(anchor_id) = after_segment_id.as_deref() {
        let anchor_exists = existing
            .segments
            .iter()
            .any(|segment| segment.id == anchor_id);
        if !anchor_exists {
            return Err(ApiError::bad_request(
                "after_segment_id must reference a segment in this project.",
            ));
        }
    }

    let project = state
        .studio_store
        .insert_segment(project_id, after_segment_id, text)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    Ok(Json(project))
}

pub async fn split_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
    Json(req): Json<SplitStudioProjectSegmentRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let before_text = required_trimmed(Some(req.before_text.as_str()), "before_text")?;
    let after_text = required_trimmed(Some(req.after_text.as_str()), "after_text")?;

    let project = state
        .studio_store
        .split_segment(project_id, segment_id, before_text, after_text)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    Ok(Json(project))
}

pub async fn merge_studio_project_segment_with_next(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .merge_segment_with_next(project_id, segment_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment could not be merged"))?;

    Ok(Json(project))
}

pub async fn reorder_studio_project_segments(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<ReorderStudioProjectSegmentsRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    if req.ordered_segment_ids.is_empty() {
        return Err(ApiError::bad_request(
            "Reorder requests require at least one segment id.",
        ));
    }
    let project = state
        .studio_store
        .reorder_segments(project_id, req.ordered_segment_ids)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(project))
}

pub async fn bulk_delete_studio_project_segments(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<BulkDeleteStudioProjectSegmentsRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    if req.segment_ids.is_empty() {
        return Err(ApiError::bad_request(
            "Bulk delete requests require at least one segment id.",
        ));
    }
    let project = state
        .studio_store
        .delete_segments(project_id, req.segment_ids)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(project))
}

pub async fn delete_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    if existing.segments.len() <= 1 {
        return Err(ApiError::bad_request(
            "A project must keep at least one segment.",
        ));
    }

    let project = state
        .studio_store
        .delete_segment(project_id, segment_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    Ok(Json(project))
}

pub async fn render_studio_project_segment(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    let segment = project
        .segments
        .iter()
        .find(|candidate| candidate.id == segment_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    let project_model_id = project
        .model_id
        .clone()
        .ok_or_else(|| ApiError::bad_request("Studio project is missing a model selection."))?;
    let segment_model_id = segment.model_id.clone().unwrap_or(project_model_id);
    let segment_voice_mode = segment.voice_mode.unwrap_or(project.voice_mode);
    let segment_speaker = if segment_voice_mode == StudioProjectVoiceMode::BuiltIn {
        segment.speaker.clone().or_else(|| project.speaker.clone())
    } else {
        None
    };
    let segment_saved_voice_id = if segment_voice_mode == StudioProjectVoiceMode::Saved {
        segment
            .saved_voice_id
            .clone()
            .or_else(|| project.saved_voice_id.clone())
    } else {
        None
    };

    validate_project_voice_state(
        &state,
        segment_model_id.as_str(),
        segment_voice_mode,
        segment_speaker.as_deref(),
        segment_saved_voice_id.as_deref(),
    )
    .await?;

    let rendered_text =
        apply_project_pronunciations(&state, project.id.as_str(), segment.text.as_str()).await?;
    let request = CreateSpeechHistoryRecordRequest {
        model_id: Some(segment_model_id),
        text: Some(rendered_text),
        speaker: segment_speaker,
        language: None,
        voice_description: None,
        reference_audio: None,
        reference_text: None,
        saved_voice_id: segment_saved_voice_id,
        temperature: None,
        speed: project.speed.map(|value| value as f32),
        max_tokens: Some(0),
        max_output_tokens: None,
        top_k: None,
        stream: Some(false),
    };

    let record = synthesize_record(&state, &ctx, request, SpeechRouteKind::TextToSpeech).await?;
    let updated_project = state
        .studio_store
        .attach_segment_record(project_id, segment_id, record.id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    Ok(Json(updated_project))
}

pub async fn get_studio_project_audio(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<ProjectAudioQuery>,
) -> Result<Response, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    let segment_filter = parse_segment_filter(query.segment_ids.as_deref());
    let segments_for_export = if let Some(filter) = segment_filter.as_ref() {
        project
            .segments
            .iter()
            .filter(|segment| filter.contains(segment.id.as_str()))
            .collect::<Vec<_>>()
    } else {
        project.segments.iter().collect::<Vec<_>>()
    };

    if segments_for_export.is_empty() {
        return Err(ApiError::bad_request(
            "Render at least one selected project segment before exporting audio.",
        ));
    }

    let missing_count = segments_for_export
        .iter()
        .filter(|segment| segment.speech_record_id.is_none())
        .count();
    if missing_count > 0 {
        return Err(ApiError::bad_request(format!(
            "Render all project segments before exporting audio. {missing_count} segment(s) are still pending."
        )));
    }

    let mut merged_samples = Vec::new();
    let mut merged_sample_rate: Option<u32> = None;

    for segment in segments_for_export {
        let record_id = segment
            .speech_record_id
            .as_deref()
            .ok_or_else(|| ApiError::bad_request("Project segment is missing rendered audio."))?;
        let audio = state
            .speech_history_store
            .get_audio(SpeechRouteKind::TextToSpeech, record_id.to_string())
            .await
            .map_err(map_speech_store_error)?
            .ok_or_else(|| ApiError::not_found("Rendered segment audio not found"))?;
        let (samples, sample_rate) = decode_wav_audio(audio.audio_bytes.as_slice())?;

        if let Some(existing_rate) = merged_sample_rate {
            if existing_rate != sample_rate {
                return Err(ApiError::internal(format!(
                    "Project export sample-rate mismatch: {existing_rate} vs {sample_rate}"
                )));
            }
        } else {
            merged_sample_rate = Some(sample_rate);
        }

        merged_samples.extend_from_slice(samples.as_slice());
    }

    let sample_rate = merged_sample_rate.ok_or_else(|| {
        ApiError::bad_request("Project export did not contain any renderable segment audio.")
    })?;
    let export_format = parse_project_audio_format(query.format.as_deref())?;
    let wav_bytes = AudioEncoder::new(sample_rate, 1)
        .encode(merged_samples.as_slice(), export_format)
        .map_err(|err| {
            ApiError::internal(format!("Failed to encode merged project audio: {err}"))
        })?;

    Ok(audio_response(
        StoredSpeechAudio {
            audio_bytes: wav_bytes,
            audio_mime_type: AudioEncoder::content_type(export_format).to_string(),
            audio_filename: Some(project_audio_filename(project.name.as_str(), export_format)),
        },
        query.download,
    ))
}

pub async fn delete_studio_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<DeleteStudioProjectResponse>, ApiError> {
    let deleted = state
        .studio_store
        .delete_project(project_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Studio project not found"));
    }

    Ok(Json(DeleteStudioProjectResponse {
        id: project_id,
        deleted: true,
    }))
}

fn normalize_create_request(mut req: CreateStudioProjectRequest) -> CreateStudioProjectRequest {
    req.name = normalize_optional_trimmed(req.name);
    req.source_filename = normalize_optional_trimmed(req.source_filename);
    req.model_id = req.model_id.trim().to_string();
    req.source_text = req.source_text.trim().to_string();
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req
}

fn normalize_update_request(mut req: UpdateStudioProjectRequest) -> UpdateStudioProjectRequest {
    req.name = normalize_optional_trimmed(req.name);
    req.model_id = normalize_optional_trimmed(req.model_id);
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req
}

fn normalize_segment_update_request(
    mut req: UpdateStudioProjectSegmentRequest,
) -> UpdateStudioProjectSegmentRequest {
    req.text = normalize_optional_trimmed(req.text);
    req.model_id = normalize_optional_trimmed(req.model_id);
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req
}

async fn validate_project_voice_state(
    state: &AppState,
    model_id: &str,
    voice_mode: StudioProjectVoiceMode,
    speaker: Option<&str>,
    saved_voice_id: Option<&str>,
) -> Result<(), ApiError> {
    let variant = parse_tts_model_variant(model_id)
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {err}")))?;
    let capabilities = variant.speech_capabilities().ok_or_else(|| {
        ApiError::bad_request(format!(
            "{variant} does not expose speech generation capabilities."
        ))
    })?;

    match voice_mode {
        StudioProjectVoiceMode::BuiltIn => {
            if !capabilities.supports_builtin_voices {
                return Err(ApiError::bad_request(format!(
                    "{model_id} does not support built-in speaker rendering for Studio projects.",
                )));
            }
            if !has_non_empty_text(speaker) {
                return Err(ApiError::bad_request(
                    "Built-in Studio projects require a speaker selection.",
                ));
            }
        }
        StudioProjectVoiceMode::Saved => {
            if !capabilities.supports_reference_voice {
                return Err(ApiError::bad_request(format!(
                    "{model_id} does not support saved-voice rendering for Studio projects.",
                )));
            }
            let Some(voice_id) = saved_voice_id.filter(|value| has_non_empty_text(Some(value)))
            else {
                return Err(ApiError::bad_request(
                    "Saved-voice Studio projects require a saved_voice_id selection.",
                ));
            };
            let exists = state
                .saved_voice_store
                .get_voice(voice_id.to_string())
                .await
                .map_err(map_saved_voice_store_error)?
                .is_some();
            if !exists {
                return Err(ApiError::not_found("Saved voice not found"));
            }
        }
    }
    Ok(())
}

fn required_trimmed(raw: Option<&str>, field_name: &str) -> Result<String, ApiError> {
    let trimmed = raw.unwrap_or("").trim();
    if trimmed.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Missing required `{field_name}` field."
        )));
    }
    Ok(trimmed.to_string())
}

fn normalize_optional_trimmed(raw: Option<String>) -> Option<String> {
    let trimmed = raw.unwrap_or_default().trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn has_non_empty_text(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .map(|value| !value.is_empty())
        .unwrap_or(false)
}

fn voice_mode_speaker(
    voice_mode: StudioProjectVoiceMode,
    speaker: Option<String>,
) -> Option<String> {
    match voice_mode {
        StudioProjectVoiceMode::BuiltIn => speaker,
        StudioProjectVoiceMode::Saved => None,
    }
}

fn voice_mode_saved_voice_id(
    voice_mode: StudioProjectVoiceMode,
    saved_voice_id: Option<String>,
) -> Option<String> {
    match voice_mode {
        StudioProjectVoiceMode::BuiltIn => None,
        StudioProjectVoiceMode::Saved => saved_voice_id,
    }
}

fn default_project_name(source_text: &str) -> String {
    let snippet = source_text
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ");
    if snippet.is_empty() {
        "TTS Project".to_string()
    } else {
        format!(
            "{}{}",
            snippet,
            if snippet.ends_with('.') { "" } else { "..." }
        )
    }
}

fn project_audio_filename(name: &str, format: AudioFormat) -> String {
    let slug = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let extension = match format {
        AudioFormat::Wav => "wav",
        AudioFormat::RawI16 => "pcm",
        AudioFormat::RawF32 => "f32",
    };
    if slug.is_empty() {
        format!("studio-project.{extension}")
    } else {
        format!("{slug}.{extension}")
    }
}

fn decode_wav_audio(bytes: &[u8]) -> Result<(Vec<f32>, u32), ApiError> {
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|err| ApiError::internal(format!("Failed to decode project WAV audio: {err}")))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let mut mono = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = if spec.bits_per_sample <= 16 {
                i16::MAX as f32
            } else {
                ((1_i64 << (spec.bits_per_sample.saturating_sub(1) as i64)) - 1) as f32
            };
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<i32>() {
                let value = sample.map_err(|err| {
                    ApiError::internal(format!("Failed reading project WAV sample: {err}"))
                })?;
                frame.push(value as f32 / scale);
                if frame.len() == channels {
                    let avg = frame.iter().sum::<f32>() / channels as f32;
                    mono.push(avg);
                    frame.clear();
                }
            }
            if !frame.is_empty() {
                let avg = frame.iter().sum::<f32>() / frame.len() as f32;
                mono.push(avg);
            }
        }
        hound::SampleFormat::Float => {
            let mut frame = Vec::with_capacity(channels);
            for sample in reader.samples::<f32>() {
                let value = sample.map_err(|err| {
                    ApiError::internal(format!("Failed reading project WAV sample: {err}"))
                })?;
                frame.push(value);
                if frame.len() == channels {
                    let avg = frame.iter().sum::<f32>() / channels as f32;
                    mono.push(avg);
                    frame.clear();
                }
            }
            if !frame.is_empty() {
                let avg = frame.iter().sum::<f32>() / frame.len() as f32;
                mono.push(avg);
            }
        }
    }

    Ok((mono, spec.sample_rate))
}

fn audio_response(audio: StoredSpeechAudio, as_attachment: bool) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    let disposition = if as_attachment {
        audio
            .audio_filename
            .as_deref()
            .map(|filename| format!("attachment; filename=\"{}\"", filename.replace('"', "")))
            .unwrap_or_else(|| "attachment".to_string())
    } else if let Some(filename) = audio.audio_filename.as_deref() {
        format!("inline; filename=\"{}\"", filename.replace('"', ""))
    } else {
        "inline".to_string()
    };
    if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
        response = response.header(header::CONTENT_DISPOSITION, value);
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Studio project storage error: {err}"))
}

fn map_saved_voice_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Saved voice storage error: {err}"))
}

fn map_speech_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Speech history storage error: {err}"))
}

async fn apply_project_pronunciations(
    state: &AppState,
    project_id: &str,
    text: &str,
) -> Result<String, ApiError> {
    let entries = state
        .studio_store
        .list_project_pronunciations(project_id.to_string())
        .await
        .map_err(map_store_error)?;
    if entries.is_empty() {
        return Ok(text.to_string());
    }

    let mut ordered = entries;
    ordered.sort_by(|left, right| {
        right
            .source_text
            .len()
            .cmp(&left.source_text.len())
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut output = text.to_string();
    for entry in ordered {
        if entry.source_text.is_empty() {
            continue;
        }
        output = output.replace(entry.source_text.as_str(), entry.replacement_text.as_str());
    }
    Ok(output)
}

fn parse_project_audio_format(raw: Option<&str>) -> Result<AudioFormat, ApiError> {
    let value = raw.unwrap_or("wav").trim().to_lowercase();
    match value.as_str() {
        "" | "wav" => Ok(AudioFormat::Wav),
        "raw_i16" | "pcm" => Ok(AudioFormat::RawI16),
        "raw_f32" | "f32" => Ok(AudioFormat::RawF32),
        _ => Err(ApiError::bad_request(format!(
            "Unsupported project export format: {value}.",
        ))),
    }
}

fn parse_segment_filter(raw: Option<&str>) -> Option<HashSet<String>> {
    raw.map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| entry.to_string())
            .collect::<HashSet<_>>()
    })
    .filter(|entries| !entries.is_empty())
}

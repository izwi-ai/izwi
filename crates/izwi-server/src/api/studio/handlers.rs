use axum::{
    body::Body,
    extract::{Json, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::error::{ApiError, ApiJson};
use crate::speech_history_store::{SpeechRouteKind, StoredSpeechAudio};
use crate::state::AppState;
use crate::studio_project_store::{
    segment_render_input, NewStudioProjectFolderRecord, NewStudioProjectPronunciationRecord,
    NewStudioProjectRecord, NewStudioProjectSegment, NewStudioProjectSnapshotRecord,
    StudioProjectChange, StudioProjectExportFormat, StudioProjectFolderRecord,
    StudioProjectListCursor, StudioProjectMetaRecord, StudioProjectPronunciationRecord,
    StudioProjectRecord, StudioProjectRenderJobRecord, StudioProjectRenderJobStatus,
    StudioProjectSegmentRecord, StudioProjectSnapshotRecord, StudioProjectSummary,
    StudioProjectVoiceMode, StudioSegmentVoiceOverride, UpdateStudioProjectRecord,
    UpsertStudioProjectMetaRecord,
};
use izwi_core::audio::{resample_mono_high_quality, AudioEncoder, AudioFormat};
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
    /// Silence between paragraphs, in milliseconds.
    #[serde(default)]
    pause_ms: Option<u32>,
}

/// The longest pause export puts between paragraphs.
const MAX_EXPORT_PAUSE_MS: u32 = 5_000;

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
    /// Drop the segment's own voice and render with the project's.
    #[serde(default)]
    pub use_project_voice: Option<bool>,
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
    /// Segments to render; empty or absent renders every segment whose
    /// audio is missing, stale or failed.
    #[serde(default, alias = "queued_segment_ids")]
    pub segment_ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateStudioProjectRenderJobRequest {
    pub status: StudioProjectRenderJobStatus,
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
    ApiJson(req): ApiJson<CreateStudioProjectFolderRequest>,
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
    state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(StudioProjectMetaRecord {
        project_id,
        folder_id: None,
        tags: Vec::new(),
        default_export_format: StudioProjectExportFormat::Wav,
        last_render_job_id: None,
        last_rendered_at: None,
    }))
}

pub async fn upsert_studio_project_meta(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(req): ApiJson<UpsertStudioProjectMetaRequest>,
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
    ApiJson(req): ApiJson<CreateStudioProjectPronunciationRequest>,
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
    ApiJson(req): ApiJson<CreateStudioProjectSnapshotRequest>,
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
    ApiJson(req): ApiJson<CreateStudioProjectRenderJobRequest>,
) -> Result<Json<StudioProjectRenderJobRecord>, ApiError> {
    let segment_ids = req.segment_ids.filter(|ids| !ids.is_empty());
    let job = state
        .studio_store
        .enqueue_render(project_id, segment_ids)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    state.studio_render_wake.notify_one();
    Ok(Json(job))
}

/// The server runs render jobs; a client can only cancel one.
pub async fn update_studio_project_render_job(
    State(state): State<AppState>,
    Path((project_id, job_id)): Path<(String, String)>,
    ApiJson(req): ApiJson<UpdateStudioProjectRenderJobRequest>,
) -> Result<Json<StudioProjectRenderJobRecord>, ApiError> {
    if req.status != StudioProjectRenderJobStatus::Cancelled {
        return Err(ApiError::bad_request(
            "Render jobs can only be cancelled; the server sets every other status.",
        ));
    }
    let job = state
        .studio_store
        .cancel_render_job(project_id, job_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project render job not found"))?;
    Ok(Json(job))
}

pub async fn create_studio_project(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<CreateStudioProjectRequest>,
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
    let chunks = script_segments(variant, &source_text);
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
    ApiJson(req): ApiJson<UpdateStudioProjectRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;

    let req = normalize_update_request(req);
    // Only a voice change is checked: a rename must work after the
    // project's saved voice was deleted.
    let voice_changed = req.model_id.is_some()
        || req.voice_mode.is_some()
        || req.speaker.is_some()
        || req.saved_voice_id.is_some();
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

    if voice_changed {
        validate_project_voice_state(
            &state,
            next_model_id.as_str(),
            next_voice_mode,
            next_speaker.as_deref(),
            next_saved_voice_id.as_deref(),
        )
        .await?;
    }
    let supports_speed = parse_tts_model_variant(&next_model_id)
        .ok()
        .and_then(|variant| variant.speech_capabilities())
        .is_some_and(|caps| caps.supports_speed_control);
    if req.speed.is_some() && !supports_speed {
        return Err(ApiError::bad_request(format!(
            "{next_model_id} has no speed control."
        )));
    }
    // A model without speed control drops the speed an earlier one kept.
    let speed = if supports_speed {
        req.speed.map(|value| Some(f64::from(value)))
    } else if req.model_id.is_some() {
        Some(None)
    } else {
        None
    };

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
                speed,
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
    ApiJson(req): ApiJson<UpdateStudioProjectSegmentRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let use_project_voice = req.use_project_voice.unwrap_or(false);
    let req = normalize_segment_update_request(req);
    let text_update = req.text.clone();
    let has_voice_update = req.model_id.is_some()
        || req.voice_mode.is_some()
        || req.speaker.is_some()
        || req.saved_voice_id.is_some();
    if use_project_voice && has_voice_update {
        return Err(ApiError::bad_request(
            "Choose a segment voice or `use_project_voice`, not both.",
        ));
    }
    if text_update.is_none() && !has_voice_update && !use_project_voice {
        return Err(ApiError::bad_request(
            "Segment update requires text or voice changes.",
        ));
    }

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
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    let voice_update = if use_project_voice {
        Some(None)
    } else if has_voice_update {
        // The override keeps what the segment already chose and changes
        // what the request names. Its voice belongs to a model, so the
        // override is bound to the model it was chosen with (the project's
        // now, unless named): a later project model change must not render
        // this paragraph's speaker with a model that lacks it.
        let model_id = req
            .model_id
            .clone()
            .or_else(|| segment.model_id.clone())
            .or_else(|| project.model_id.clone());
        let voice_mode = req
            .voice_mode
            .or(segment.voice_mode)
            .unwrap_or(project.voice_mode);
        let speaker = req.speaker.clone().or_else(|| segment.speaker.clone());
        let saved_voice_id = req
            .saved_voice_id
            .clone()
            .or_else(|| segment.saved_voice_id.clone());
        let resolved_model_id = model_id
            .clone()
            .or_else(|| project.model_id.clone())
            .ok_or_else(|| ApiError::bad_request("Studio project is missing a model selection."))?;
        let resolved_speaker = speaker.clone().or_else(|| project.speaker.clone());
        let resolved_saved_voice_id = saved_voice_id
            .clone()
            .or_else(|| project.saved_voice_id.clone());
        validate_project_voice_state(
            &state,
            resolved_model_id.as_str(),
            voice_mode,
            resolved_speaker.as_deref(),
            resolved_saved_voice_id.as_deref(),
        )
        .await?;
        Some(Some(StudioSegmentVoiceOverride {
            model_id,
            voice_mode,
            speaker: voice_mode_speaker(voice_mode, speaker),
            saved_voice_id: voice_mode_saved_voice_id(voice_mode, saved_voice_id),
        }))
    } else {
        None
    };

    let project = state
        .studio_store
        .update_segment(project_id, segment_id, text_update, voice_update)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;
    Ok(Json(project))
}

pub async fn create_studio_project_segment(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(req): ApiJson<CreateStudioProjectSegmentRequest>,
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
    ApiJson(req): ApiJson<SplitStudioProjectSegmentRequest>,
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
    let change = state
        .studio_store
        .merge_segment_with_next(project_id, segment_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment could not be merged"))?;

    Ok(Json(apply_project_change(&state, change).await))
}

pub async fn reorder_studio_project_segments(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(req): ApiJson<ReorderStudioProjectSegmentsRequest>,
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
    ApiJson(req): ApiJson<BulkDeleteStudioProjectSegmentsRequest>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    if req.segment_ids.is_empty() {
        return Err(ApiError::bad_request(
            "Bulk delete requests require at least one segment id.",
        ));
    }
    let change = state
        .studio_store
        .delete_segments(project_id, req.segment_ids)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(apply_project_change(&state, change).await))
}

pub async fn delete_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    let change = state
        .studio_store
        .delete_segment(project_id, segment_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?;

    Ok(Json(apply_project_change(&state, change).await))
}

/// Queue one segment, rendered or not ("render again").
pub async fn render_studio_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<StudioProjectRecord>, ApiError> {
    state
        .studio_store
        .enqueue_render(project_id.clone(), Some(vec![segment_id]))
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    state.studio_render_wake.notify_one();
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    Ok(Json(project))
}

pub async fn get_studio_project_segment_audio(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    let record_id = project
        .segments
        .iter()
        .find(|segment| segment.id == segment_id)
        .ok_or_else(|| ApiError::not_found("Studio project segment not found"))?
        .speech_record_id
        .clone()
        .ok_or_else(|| ApiError::not_found("This paragraph has no audio yet."))?;
    let audio = load_segment_audio(&state, &record_id)
        .await?
        .ok_or_else(|| ApiError::not_found("This paragraph has no audio yet."))?;
    Ok(audio_response(audio, false))
}

pub async fn get_studio_project_audio(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<ProjectAudioQuery>,
) -> Result<Response, ApiError> {
    let export_format = parse_project_audio_format(query.format.as_deref())?;
    let pause_ms = query.pause_ms.unwrap_or(0);
    if pause_ms > MAX_EXPORT_PAUSE_MS {
        return Err(ApiError::bad_request(format!(
            "`pause_ms` can be at most {MAX_EXPORT_PAUSE_MS}."
        )));
    }
    let project = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    let pronunciations = state
        .studio_store
        .list_project_pronunciations(project_id)
        .await
        .map_err(map_store_error)?;

    let segments_for_export = match parse_segment_filter(query.segment_ids.as_deref()) {
        Some(filter) => {
            let selected = project
                .segments
                .iter()
                .filter(|segment| filter.contains(segment.id.as_str()))
                .collect::<Vec<_>>();
            if selected.len() != filter.len() {
                return Err(ApiError::bad_request(
                    "`segment_ids` names a paragraph that isn't in this project.",
                ));
            }
            selected
        }
        None => project.segments.iter().collect::<Vec<_>>(),
    };
    if segments_for_export.is_empty() {
        return Err(ApiError::bad_request(
            "This project has no paragraphs to export.",
        ));
    }

    // Export only audio that matches what each paragraph says now.
    let mut unrendered = Vec::new();
    let mut changed = Vec::new();
    for segment in &segments_for_export {
        let paragraph = segment.position + 1;
        match segment.speech_record_id {
            None => unrendered.push(paragraph),
            Some(_) => {
                let current = segment_render_input(&project, segment, &pronunciations).2;
                if segment
                    .render_fingerprint
                    .as_deref()
                    .is_some_and(|rendered_from| rendered_from != current)
                {
                    changed.push(paragraph);
                }
            }
        }
    }
    if !unrendered.is_empty() || !changed.is_empty() {
        let mut problems = Vec::new();
        if !unrendered.is_empty() {
            problems.push(format!("not rendered: {}", paragraph_list(&unrendered)));
        }
        if !changed.is_empty() {
            problems.push(format!(
                "changed since they were rendered: {}",
                paragraph_list(&changed)
            ));
        }
        return Err(ApiError::bad_request(format!(
            "Render these paragraphs before exporting ({}).",
            problems.join("; ")
        )));
    }

    let mut takes = Vec::with_capacity(segments_for_export.len());
    for segment in &segments_for_export {
        let record_id = segment.speech_record_id.as_deref().unwrap_or_default();
        let audio = load_segment_audio(&state, record_id)
            .await?
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Paragraph {} lost its audio. Render it again before exporting.",
                    segment.position + 1
                ))
            })?;
        takes.push(audio.audio_bytes);
    }

    let filename = project_audio_filename(project.name.as_str(), export_format);
    let bytes = tokio::task::spawn_blocking(move || join_takes(&takes, pause_ms, export_format))
        .await
        .map_err(|err| ApiError::internal(format!("Project export failed: {err}")))??;

    Ok(audio_response(
        StoredSpeechAudio {
            audio_bytes: bytes,
            audio_mime_type: AudioEncoder::content_type(export_format).to_string(),
            audio_filename: Some(filename),
        },
        query.download,
    ))
}

/// "3", "3 and 7", "3, 7 and 9".
fn paragraph_list(paragraphs: &[usize]) -> String {
    let names = paragraphs.iter().map(usize::to_string).collect::<Vec<_>>();
    match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => names.join(""),
    }
}

/// Join WAV takes into one file at the highest sample rate among them,
/// with `pause_ms` of silence between takes. Samples are kept as 16-bit
/// PCM while joining, so a long project holds one copy of its audio.
fn join_takes(takes: &[Vec<u8>], pause_ms: u32, format: AudioFormat) -> Result<Vec<u8>, ApiError> {
    let mut target_rate = 0;
    for take in takes {
        target_rate = target_rate.max(wav_sample_rate(take)?);
    }
    if target_rate == 0 {
        return Err(ApiError::bad_request(
            "This project has no audio to export.",
        ));
    }
    let pause_samples = (u64::from(target_rate) * u64::from(pause_ms) / 1000) as usize;

    let mut pcm: Vec<i16> = Vec::new();
    for (index, take) in takes.iter().enumerate() {
        if index > 0 {
            pcm.extend(std::iter::repeat_n(0, pause_samples));
        }
        let (samples, sample_rate) = decode_wav_audio(take)?;
        let samples = if sample_rate == target_rate {
            samples
        } else {
            resample_mono_high_quality(&samples, sample_rate, target_rate).map_err(|err| {
                ApiError::internal(format!("Failed to resample project audio: {err}"))
            })?
        };
        pcm.extend(
            samples
                .iter()
                .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16),
        );
    }

    Ok(match format {
        AudioFormat::Wav => {
            let data_len = (pcm.len() * 2) as u32;
            let mut bytes = Vec::with_capacity(44 + pcm.len() * 2);
            bytes.extend_from_slice(b"RIFF");
            bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
            bytes.extend_from_slice(b"WAVEfmt ");
            bytes.extend_from_slice(&16u32.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&target_rate.to_le_bytes());
            bytes.extend_from_slice(&(target_rate * 2).to_le_bytes());
            bytes.extend_from_slice(&2u16.to_le_bytes());
            bytes.extend_from_slice(&16u16.to_le_bytes());
            bytes.extend_from_slice(b"data");
            bytes.extend_from_slice(&data_len.to_le_bytes());
            bytes.extend(pcm.iter().flat_map(|sample| sample.to_le_bytes()));
            bytes
        }
        AudioFormat::RawI16 => pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect(),
        AudioFormat::RawF32 => pcm
            .iter()
            .flat_map(|sample| (f32::from(*sample) / f32::from(i16::MAX)).to_le_bytes())
            .collect(),
    })
}

fn wav_sample_rate(bytes: &[u8]) -> Result<u32, ApiError> {
    hound::WavReader::new(std::io::Cursor::new(bytes))
        .map(|reader| reader.spec().sample_rate)
        .map_err(|err| ApiError::internal(format!("Failed to read project WAV audio: {err}")))
}

/// A segment's audio: a Studio record, or a text-to-speech record for
/// projects rendered before Studio had its own.
async fn load_segment_audio(
    state: &AppState,
    record_id: &str,
) -> Result<Option<StoredSpeechAudio>, ApiError> {
    for route_kind in [SpeechRouteKind::Studio, SpeechRouteKind::TextToSpeech] {
        if let Some(audio) = state
            .speech_history_store
            .get_audio(route_kind, record_id.to_string())
            .await
            .map_err(map_speech_store_error)?
        {
            return Ok(Some(audio));
        }
    }
    Ok(None)
}

/// Delete the Studio audio a change released and return its project.
async fn apply_project_change(
    state: &AppState,
    change: StudioProjectChange,
) -> StudioProjectRecord {
    release_studio_audio(state, change.released_speech_record_ids).await;
    change.project
}

/// Delete Studio audio no segment references. Text-to-speech records from
/// before Studio had its own stay in that history, where they're listed.
pub(super) async fn release_studio_audio(state: &AppState, record_ids: Vec<String>) {
    for record_id in record_ids {
        if let Err(err) = state
            .speech_history_store
            .delete_record(SpeechRouteKind::Studio, record_id.clone())
            .await
        {
            tracing::warn!(record_id = %record_id, error = %err, "Failed to delete released Studio audio");
        }
    }
}

pub async fn delete_studio_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<DeleteStudioProjectResponse>, ApiError> {
    let released = state
        .studio_store
        .delete_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Studio project not found"))?;
    release_studio_audio(&state, released).await;

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

pub(super) async fn validate_project_voice_state(
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

/// A script's segments: its paragraphs (split at blank lines), with any
/// paragraph too long for one render split further at sentence breaks.
fn script_segments(variant: izwi_core::ModelVariant, script: &str) -> Vec<String> {
    let normalized = script.replace("\r\n", "\n");
    let mut paragraphs = Vec::new();
    let mut current = Vec::new();
    for line in normalized.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join("\n"));
                current.clear();
            }
        } else {
            current.push(line.trim());
        }
    }
    if !current.is_empty() {
        paragraphs.push(current.join("\n"));
    }
    paragraphs
        .iter()
        .flat_map(|paragraph| {
            crate::api::tts_long_form::split_tts_text_for_long_form(variant, 0, paragraph)
        })
        .collect()
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
    crate::error::store_api_error(err, "Studio project storage error")
}

fn map_saved_voice_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Saved voice storage error")
}

pub(super) fn map_speech_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Speech history storage error")
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

#[cfg(test)]
mod tests {
    use super::{join_takes, paragraph_list};
    use izwi_core::audio::AudioFormat;

    fn wav(sample_rate: u32, samples: usize) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut bytes, spec).expect("writer");
        for index in 0..samples {
            writer
                .write_sample(((index % 100) as i16 - 50) * 100)
                .expect("sample");
        }
        writer.finalize().expect("finalize");
        bytes.into_inner()
    }

    #[test]
    fn takes_join_at_the_highest_rate_with_a_pause_between_them() {
        let joined = join_takes(
            &[wav(16_000, 16_000), wav(24_000, 12_000)],
            500,
            AudioFormat::Wav,
        )
        .expect("takes join");
        let reader = hound::WavReader::new(std::io::Cursor::new(joined)).expect("valid wav");
        assert_eq!(reader.spec().sample_rate, 24_000);
        assert_eq!(reader.spec().channels, 1);
        // One second resampled to 24 kHz, half a second of silence, then
        // half a second as recorded.
        let expected = 24_000 + 12_000 + 12_000;
        let length = reader.len() as i64;
        assert!(
            (length - expected).abs() <= 64,
            "joined {length} samples, expected about {expected}"
        );
    }

    #[test]
    fn scripts_split_into_their_paragraphs() {
        let variant = izwi_core::parse_tts_model_variant("Kokoro-82M").expect("variant");
        assert_eq!(
            super::script_segments(variant, "First line.\r\n\r\n  \nSecond line,\ncontinued.\n"),
            vec![
                "First line.".to_string(),
                "Second line,\ncontinued.".to_string()
            ]
        );
    }

    #[test]
    fn paragraph_lists_read_as_prose() {
        assert_eq!(paragraph_list(&[3]), "3");
        assert_eq!(paragraph_list(&[3, 7]), "3 and 7");
        assert_eq!(paragraph_list(&[3, 7, 9]), "3, 7 and 9");
    }
}

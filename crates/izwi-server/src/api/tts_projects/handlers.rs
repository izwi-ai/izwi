use axum::{
    body::Body,
    extract::{Extension, Json, Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde::{Deserialize, Serialize};

use crate::api::request_context::RequestContext;
use crate::api::speech_history::{synthesize_record, CreateSpeechHistoryRecordRequest};
use crate::error::ApiError;
use crate::speech_history_store::{SpeechRouteKind, StoredSpeechAudio};
use crate::state::AppState;
use crate::studio_store::{
    NewTtsProjectFolderRecord, NewTtsProjectPronunciationRecord, NewTtsProjectRecord,
    NewTtsProjectRenderJobRecord, NewTtsProjectSegment, NewTtsProjectSnapshotRecord,
    TtsProjectExportFormat, TtsProjectFolderRecord, TtsProjectMetaRecord,
    TtsProjectPronunciationRecord, TtsProjectRecord, TtsProjectRenderJobRecord,
    TtsProjectRenderJobStatus, TtsProjectSegmentRecord, TtsProjectSnapshotRecord,
    TtsProjectSummary, TtsProjectVoiceMode, UpdateTtsProjectRecord,
    UpdateTtsProjectRenderJobRecord, UpsertTtsProjectMetaRecord,
};
use izwi_core::audio::{AudioEncoder, AudioFormat};
use izwi_core::parse_tts_model_variant;

const PROJECT_LIST_LIMIT: usize = 100;

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ProjectAudioQuery {
    #[serde(default)]
    download: bool,
}

#[derive(Debug, Serialize)]
pub struct TtsProjectListResponse {
    pub projects: Vec<TtsProjectSummary>,
}

#[derive(Debug, Serialize)]
pub struct DeleteTtsProjectResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreateTtsProjectRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: String,
    #[serde(default)]
    pub voice_mode: Option<TtsProjectVoiceMode>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateTtsProjectRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub voice_mode: Option<TtsProjectVoiceMode>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateTtsProjectSegmentRequest {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct SplitTtsProjectSegmentRequest {
    pub before_text: String,
    pub after_text: String,
}

#[derive(Debug, Deserialize)]
pub struct ReorderTtsProjectSegmentsRequest {
    pub ordered_segment_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct BulkDeleteTtsProjectSegmentsRequest {
    pub segment_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TtsProjectFolderListResponse {
    pub folders: Vec<TtsProjectFolderRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTtsProjectFolderRequest {
    pub name: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub sort_order: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct TtsProjectPronunciationListResponse {
    pub entries: Vec<TtsProjectPronunciationRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTtsProjectPronunciationRequest {
    pub source_text: String,
    pub replacement_text: String,
    #[serde(default)]
    pub locale: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteTtsProjectPronunciationResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct TtsProjectSnapshotListResponse {
    pub snapshots: Vec<TtsProjectSnapshotRecord>,
}

#[derive(Debug, Deserialize)]
pub struct CreateTtsProjectSnapshotRequest {
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpsertTtsProjectMetaRequest {
    #[serde(default)]
    pub folder_id: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub default_export_format: Option<TtsProjectExportFormat>,
    #[serde(default)]
    pub last_render_job_id: Option<String>,
    #[serde(default)]
    pub last_rendered_at: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct TtsProjectRenderJobListResponse {
    pub jobs: Vec<TtsProjectRenderJobRecord>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CreateTtsProjectRenderJobRequest {
    #[serde(default)]
    pub queued_segment_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateTtsProjectRenderJobRequest {
    #[serde(default)]
    pub status: Option<TtsProjectRenderJobStatus>,
    #[serde(default)]
    pub error_message: Option<String>,
}

pub async fn list_tts_projects(
    State(state): State<AppState>,
) -> Result<Json<TtsProjectListResponse>, ApiError> {
    let projects = state
        .studio_store
        .list_projects(PROJECT_LIST_LIMIT)
        .await
        .map_err(map_store_error)?;
    Ok(Json(TtsProjectListResponse { projects }))
}

pub async fn list_tts_project_folders(
    State(state): State<AppState>,
) -> Result<Json<TtsProjectFolderListResponse>, ApiError> {
    let folders = state
        .studio_store
        .list_folders()
        .await
        .map_err(map_store_error)?;
    Ok(Json(TtsProjectFolderListResponse { folders }))
}

pub async fn create_tts_project_folder(
    State(state): State<AppState>,
    Json(req): Json<CreateTtsProjectFolderRequest>,
) -> Result<Json<TtsProjectFolderRecord>, ApiError> {
    let name = required_trimmed(Some(req.name.as_str()), "name")?;
    let folder = state
        .studio_store
        .create_folder(NewTtsProjectFolderRecord {
            name,
            parent_id: normalize_optional_trimmed(req.parent_id),
            sort_order: req.sort_order,
        })
        .await
        .map_err(map_store_error)?;
    Ok(Json(folder))
}

pub async fn get_tts_project_meta(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<TtsProjectMetaRecord>, ApiError> {
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
        .upsert_project_meta(project_id, UpsertTtsProjectMetaRecord::default())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(meta))
}

pub async fn upsert_tts_project_meta(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpsertTtsProjectMetaRequest>,
) -> Result<Json<TtsProjectMetaRecord>, ApiError> {
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
            UpsertTtsProjectMetaRecord {
                folder_id: folder_id_update,
                tags,
                default_export_format: req.default_export_format,
                last_render_job_id: last_render_job_id_update,
                last_rendered_at: last_rendered_at_update,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(meta))
}

pub async fn list_tts_project_pronunciations(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<TtsProjectPronunciationListResponse>, ApiError> {
    let entries = state
        .studio_store
        .list_project_pronunciations(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(TtsProjectPronunciationListResponse { entries }))
}

pub async fn create_tts_project_pronunciation(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateTtsProjectPronunciationRequest>,
) -> Result<Json<TtsProjectPronunciationRecord>, ApiError> {
    let source_text = required_trimmed(Some(req.source_text.as_str()), "source_text")?;
    let replacement_text =
        required_trimmed(Some(req.replacement_text.as_str()), "replacement_text")?;
    let entry = state
        .studio_store
        .create_project_pronunciation(
            project_id,
            NewTtsProjectPronunciationRecord {
                source_text,
                replacement_text,
                locale: normalize_optional_trimmed(req.locale),
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(entry))
}

pub async fn delete_tts_project_pronunciation(
    State(state): State<AppState>,
    Path((project_id, pronunciation_id)): Path<(String, String)>,
) -> Result<Json<DeleteTtsProjectPronunciationResponse>, ApiError> {
    let deleted = state
        .studio_store
        .delete_project_pronunciation(project_id, pronunciation_id.clone())
        .await
        .map_err(map_store_error)?;
    if !deleted {
        return Err(ApiError::not_found("TTS project pronunciation not found"));
    }
    Ok(Json(DeleteTtsProjectPronunciationResponse {
        id: pronunciation_id,
        deleted: true,
    }))
}

pub async fn list_tts_project_snapshots(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<TtsProjectSnapshotListResponse>, ApiError> {
    let snapshots = state
        .studio_store
        .list_project_snapshots(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(TtsProjectSnapshotListResponse { snapshots }))
}

pub async fn create_tts_project_snapshot(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateTtsProjectSnapshotRequest>,
) -> Result<Json<TtsProjectSnapshotRecord>, ApiError> {
    let snapshot = state
        .studio_store
        .create_project_snapshot(
            project_id,
            NewTtsProjectSnapshotRecord {
                label: normalize_optional_trimmed(req.label),
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(snapshot))
}

pub async fn restore_tts_project_snapshot(
    State(state): State<AppState>,
    Path((project_id, snapshot_id)): Path<(String, String)>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .restore_project_snapshot(project_id, snapshot_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project snapshot not found"))?;
    Ok(Json(project))
}

pub async fn list_tts_project_render_jobs(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<TtsProjectRenderJobListResponse>, ApiError> {
    let jobs = state
        .studio_store
        .list_project_render_jobs(project_id)
        .await
        .map_err(map_store_error)?;
    Ok(Json(TtsProjectRenderJobListResponse { jobs }))
}

pub async fn create_tts_project_render_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<CreateTtsProjectRenderJobRequest>,
) -> Result<Json<TtsProjectRenderJobRecord>, ApiError> {
    let job = state
        .studio_store
        .create_project_render_job(
            project_id,
            NewTtsProjectRenderJobRecord {
                queued_segment_ids: req.queued_segment_ids,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(job))
}

pub async fn update_tts_project_render_job(
    State(state): State<AppState>,
    Path((project_id, job_id)): Path<(String, String)>,
    Json(req): Json<UpdateTtsProjectRenderJobRequest>,
) -> Result<Json<TtsProjectRenderJobRecord>, ApiError> {
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
            UpdateTtsProjectRenderJobRecord {
                status: req.status,
                error_message: error_message_update,
            },
        )
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project render job not found"))?;
    Ok(Json(job))
}

pub async fn create_tts_project(
    State(state): State<AppState>,
    Json(req): Json<CreateTtsProjectRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let req = normalize_create_request(req);
    let model_id = required_trimmed(Some(req.model_id.as_str()), "model_id")?;
    let source_text = required_trimmed(Some(req.source_text.as_str()), "source_text")?;
    let voice_mode = req.voice_mode.unwrap_or_else(|| {
        if req.saved_voice_id.is_some() {
            TtsProjectVoiceMode::Saved
        } else {
            TtsProjectVoiceMode::BuiltIn
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

    let project = state
        .studio_store
        .create_project(NewTtsProjectRecord {
            name: req
                .name
                .unwrap_or_else(|| default_project_name(source_text.as_str())),
            source_filename: req.source_filename,
            source_text,
            model_id: Some(model_id),
            voice_mode,
            speaker: voice_mode_speaker(voice_mode, req.speaker),
            saved_voice_id: voice_mode_saved_voice_id(voice_mode, req.saved_voice_id),
            speed: req.speed.map(f64::from),
            segments: chunks
                .into_iter()
                .enumerate()
                .map(|(position, text)| NewTtsProjectSegment { position, text })
                .collect(),
        })
        .await
        .map_err(map_store_error)?;

    Ok(Json(project))
}

pub async fn get_tts_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(project))
}

pub async fn get_tts_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<TtsProjectSegmentRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    let segment = project
        .segments
        .into_iter()
        .find(|candidate| candidate.id == segment_id)
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;
    Ok(Json(segment))
}

pub async fn update_tts_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpdateTtsProjectRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;

    let req = normalize_update_request(req);
    let next_voice_mode = req.voice_mode.unwrap_or(existing.voice_mode);
    let next_model_id = req
        .model_id
        .clone()
        .or_else(|| existing.model_id.clone())
        .ok_or_else(|| ApiError::bad_request("TTS project is missing a model selection."))?;
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
            UpdateTtsProjectRecord {
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
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;

    Ok(Json(project))
}

pub async fn update_tts_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
    Json(req): Json<UpdateTtsProjectSegmentRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let text = required_trimmed(Some(req.text.as_str()), "text")?;
    let project = state
        .studio_store
        .update_segment_text(project_id, segment_id, text)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;
    Ok(Json(project))
}

pub async fn split_tts_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
    Json(req): Json<SplitTtsProjectSegmentRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let before_text = required_trimmed(Some(req.before_text.as_str()), "before_text")?;
    let after_text = required_trimmed(Some(req.after_text.as_str()), "after_text")?;

    let project = state
        .studio_store
        .split_segment(project_id, segment_id, before_text, after_text)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;

    Ok(Json(project))
}

pub async fn merge_tts_project_segment_with_next(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .merge_segment_with_next(project_id, segment_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project segment could not be merged"))?;

    Ok(Json(project))
}

pub async fn reorder_tts_project_segments(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<ReorderTtsProjectSegmentsRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
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
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(project))
}

pub async fn bulk_delete_tts_project_segments(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<BulkDeleteTtsProjectSegmentsRequest>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
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
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;
    Ok(Json(project))
}

pub async fn delete_tts_project_segment(
    State(state): State<AppState>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let existing = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;

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
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;

    Ok(Json(project))
}

pub async fn render_tts_project_segment(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path((project_id, segment_id)): Path<(String, String)>,
) -> Result<Json<TtsProjectRecord>, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id.clone())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;

    let segment = project
        .segments
        .iter()
        .find(|candidate| candidate.id == segment_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;

    let model_id = project
        .model_id
        .clone()
        .ok_or_else(|| ApiError::bad_request("TTS project is missing a model selection."))?;
    let request = CreateSpeechHistoryRecordRequest {
        model_id: Some(model_id),
        text: Some(segment.text.clone()),
        speaker: project.speaker.clone(),
        language: None,
        voice_description: None,
        reference_audio: None,
        reference_text: None,
        saved_voice_id: project.saved_voice_id.clone(),
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
        .ok_or_else(|| ApiError::not_found("TTS project segment not found"))?;

    Ok(Json(updated_project))
}

pub async fn get_tts_project_audio(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<ProjectAudioQuery>,
) -> Result<Response, ApiError> {
    let project = state
        .studio_store
        .get_project(project_id)
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("TTS project not found"))?;

    if project.segments.is_empty() {
        return Err(ApiError::bad_request(
            "Render at least one project segment before exporting audio.",
        ));
    }

    let missing_count = project
        .segments
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

    for segment in &project.segments {
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
    let wav_bytes = AudioEncoder::new(sample_rate, 1)
        .encode(merged_samples.as_slice(), AudioFormat::Wav)
        .map_err(|err| {
            ApiError::internal(format!("Failed to encode merged project audio: {err}"))
        })?;

    Ok(audio_response(
        StoredSpeechAudio {
            audio_bytes: wav_bytes,
            audio_mime_type: AudioEncoder::content_type(AudioFormat::Wav).to_string(),
            audio_filename: Some(project_audio_filename(project.name.as_str())),
        },
        query.download,
    ))
}

pub async fn delete_tts_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<DeleteTtsProjectResponse>, ApiError> {
    let deleted = state
        .studio_store
        .delete_project(project_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("TTS project not found"));
    }

    Ok(Json(DeleteTtsProjectResponse {
        id: project_id,
        deleted: true,
    }))
}

fn normalize_create_request(mut req: CreateTtsProjectRequest) -> CreateTtsProjectRequest {
    req.name = normalize_optional_trimmed(req.name);
    req.source_filename = normalize_optional_trimmed(req.source_filename);
    req.model_id = req.model_id.trim().to_string();
    req.source_text = req.source_text.trim().to_string();
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req
}

fn normalize_update_request(mut req: UpdateTtsProjectRequest) -> UpdateTtsProjectRequest {
    req.name = normalize_optional_trimmed(req.name);
    req.model_id = normalize_optional_trimmed(req.model_id);
    req.speaker = normalize_optional_trimmed(req.speaker);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req
}

async fn validate_project_voice_state(
    state: &AppState,
    model_id: &str,
    voice_mode: TtsProjectVoiceMode,
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
        TtsProjectVoiceMode::BuiltIn => {
            if !capabilities.supports_builtin_voices {
                return Err(ApiError::bad_request(format!(
                    "{model_id} does not support built-in speaker rendering for TTS projects.",
                )));
            }
            if !has_non_empty_text(speaker) {
                return Err(ApiError::bad_request(
                    "Built-in TTS projects require a speaker selection.",
                ));
            }
        }
        TtsProjectVoiceMode::Saved => {
            if !capabilities.supports_reference_voice {
                return Err(ApiError::bad_request(format!(
                    "{model_id} does not support saved-voice rendering for TTS projects.",
                )));
            }
            let Some(voice_id) = saved_voice_id.filter(|value| has_non_empty_text(Some(value)))
            else {
                return Err(ApiError::bad_request(
                    "Saved-voice TTS projects require a saved_voice_id selection.",
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

fn voice_mode_speaker(voice_mode: TtsProjectVoiceMode, speaker: Option<String>) -> Option<String> {
    match voice_mode {
        TtsProjectVoiceMode::BuiltIn => speaker,
        TtsProjectVoiceMode::Saved => None,
    }
}

fn voice_mode_saved_voice_id(
    voice_mode: TtsProjectVoiceMode,
    saved_voice_id: Option<String>,
) -> Option<String> {
    match voice_mode {
        TtsProjectVoiceMode::BuiltIn => None,
        TtsProjectVoiceMode::Saved => saved_voice_id,
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

fn project_audio_filename(name: &str) -> String {
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
    if slug.is_empty() {
        "tts-project.wav".to_string()
    } else {
        format!("{slug}.wav")
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
    ApiError::internal(format!("TTS project storage error: {err}"))
}

fn map_saved_voice_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Saved voice storage error: {err}"))
}

fn map_speech_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Speech history storage error: {err}"))
}

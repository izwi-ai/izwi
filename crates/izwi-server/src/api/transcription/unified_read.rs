use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::Response,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::pagination::{encode_cursor, CursorPagination, CursorPaginationQuery};
use crate::diarization_store::{
    DiarizationRecordListCursor, DiarizationRecordSummary, StoredDiarizationAudio,
};
use crate::error::ApiError;
use crate::speech_text_list::{SpeechTextListFilter, PROCESSING_STATUSES};
use crate::state::AppState;
use crate::transcription_store::{
    StoredTranscriptionAudio, TranscriptionRecordListCursor, TranscriptionRecordMode,
    TranscriptionRecordSummary,
};

const HISTORY_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKindFilter {
    All,
    Transcription,
    SpeakerAttributedAsr,
    Diarization,
}

#[derive(Debug, Deserialize)]
pub struct UnifiedJobListQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    job_kind: Option<String>,
    /// Case-insensitive search over transcript text and audio filenames.
    #[serde(default)]
    q: Option<String>,
    /// Comma-separated statuses: pending, processing, ready, failed.
    #[serde(default)]
    status: Option<String>,
    /// Only jobs that identify speakers (speaker-attributed and diarization).
    #[serde(default)]
    with_speakers: Option<bool>,
}

impl UnifiedJobListQuery {
    fn filter(&self) -> Result<SpeechTextListFilter, ApiError> {
        let statuses = self
            .status
            .as_deref()
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|status| !status.is_empty())
            .map(|status| {
                if PROCESSING_STATUSES.contains(&status) {
                    Ok(status.to_string())
                } else {
                    Err(ApiError::bad_request(format!(
                        "Invalid `status` {status:?}. Supported values: pending, processing, ready, failed."
                    )))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SpeechTextListFilter {
            search: self.q.clone(),
            statuses,
            with_speakers: self.with_speakers.unwrap_or(false),
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct UnifiedJobQuery {
    #[serde(default)]
    job_kind: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UnifiedJobListResponse {
    pub records: Vec<Value>,
    pub pagination: CursorPagination,
    /// Jobs matching the filters, ignoring paging.
    pub total: usize,
}

/// Position of the last row returned by a `job_kind=all` page.
#[derive(Debug, Serialize, Deserialize)]
struct MergedJobListCursor {
    created_at: u64,
    id: String,
}

struct CombinedSummaryItem {
    created_at: u64,
    id: String,
    value: Value,
}

pub async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<UnifiedJobListQuery>,
) -> Result<Json<UnifiedJobListResponse>, ApiError> {
    let pagination = CursorPaginationQuery {
        limit: query.limit,
        cursor: query.cursor.clone(),
    };
    let limit = pagination.resolved_limit(HISTORY_LIST_LIMIT, 500);
    let kind = parse_job_kind_filter(query.job_kind.as_deref(), true)?;
    let filter = query.filter()?;

    let response = match kind {
        JobKindFilter::Transcription => {
            transcription_page_response(
                &state,
                TranscriptionRecordMode::Transcription,
                limit,
                pagination.decode_cursor::<TranscriptionRecordListCursor>()?,
                &filter,
            )
            .await?
        }
        JobKindFilter::SpeakerAttributedAsr => {
            transcription_page_response(
                &state,
                TranscriptionRecordMode::SpeakerAttributedAsr,
                limit,
                pagination.decode_cursor::<TranscriptionRecordListCursor>()?,
                &filter,
            )
            .await?
        }
        JobKindFilter::Diarization => {
            let cursor = pagination.decode_cursor::<DiarizationRecordListCursor>()?;
            let page = state
                .diarization_store
                .list_records_filtered(limit, cursor, &filter)
                .await
                .map_err(map_store_error)?;
            UnifiedJobListResponse {
                records: page.records.iter().map(map_diarization_summary).collect(),
                pagination: CursorPagination {
                    has_more: page.next_cursor.is_some(),
                    next_cursor: page.next_cursor.map(|value| encode_cursor(&value)),
                    limit,
                },
                total: page.total,
            }
        }
        JobKindFilter::All => {
            // Both stores page by (created_at DESC, id DESC) and record ids
            // are unique across them, so the merged position of the last
            // returned row is a valid "after" cursor for each store.
            let cursor = pagination.decode_cursor::<MergedJobListCursor>()?;
            let transcription_cursor =
                cursor.as_ref().map(|cursor| TranscriptionRecordListCursor {
                    created_at: cursor.created_at,
                    id: cursor.id.clone(),
                });
            let diarization_cursor = cursor.map(|cursor| DiarizationRecordListCursor {
                created_at: cursor.created_at,
                id: cursor.id,
            });

            let transcriptions = state
                .transcription_store
                .list_records_filtered(None, limit, transcription_cursor, &filter)
                .await
                .map_err(map_store_error)?;
            let diarizations = state
                .diarization_store
                .list_records_filtered(limit, diarization_cursor, &filter)
                .await
                .map_err(map_store_error)?;
            let (transcription_records, transcription_next_cursor) =
                (transcriptions.records, transcriptions.next_cursor);
            let (diarization_records, diarization_next_cursor) =
                (diarizations.records, diarizations.next_cursor);

            let mut combined = Vec::with_capacity(
                transcription_records
                    .len()
                    .saturating_add(diarization_records.len()),
            );
            for record in &transcription_records {
                combined.push(CombinedSummaryItem {
                    created_at: record.created_at,
                    id: record.id.clone(),
                    value: map_transcription_summary(record),
                });
            }
            for record in &diarization_records {
                combined.push(CombinedSummaryItem {
                    created_at: record.created_at,
                    id: record.id.clone(),
                    value: map_diarization_summary(record),
                });
            }
            combined.sort_by(|left, right| {
                right
                    .created_at
                    .cmp(&left.created_at)
                    .then_with(|| right.id.cmp(&left.id))
            });
            let has_more = combined.len() > limit
                || transcription_next_cursor.is_some()
                || diarization_next_cursor.is_some();
            combined.truncate(limit);
            let next_cursor = has_more.then(|| combined.last()).flatten().map(|item| {
                encode_cursor(&MergedJobListCursor {
                    created_at: item.created_at,
                    id: item.id.clone(),
                })
            });

            UnifiedJobListResponse {
                records: combined.into_iter().map(|item| item.value).collect(),
                pagination: CursorPagination {
                    next_cursor,
                    has_more,
                    limit,
                },
                total: transcriptions.total + diarizations.total,
            }
        }
    };

    Ok(Json(response))
}

async fn transcription_page_response(
    state: &AppState,
    mode: TranscriptionRecordMode,
    limit: usize,
    cursor: Option<TranscriptionRecordListCursor>,
    filter: &SpeechTextListFilter,
) -> Result<UnifiedJobListResponse, ApiError> {
    let page = state
        .transcription_store
        .list_records_filtered(Some(mode), limit, cursor, filter)
        .await
        .map_err(map_store_error)?;
    Ok(UnifiedJobListResponse {
        records: page.records.iter().map(map_transcription_summary).collect(),
        pagination: CursorPagination {
            has_more: page.next_cursor.is_some(),
            next_cursor: page.next_cursor.map(|value| encode_cursor(&value)),
            limit,
        },
        total: page.total,
    })
}

pub async fn get_job(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedJobQuery>,
) -> Result<Json<Value>, ApiError> {
    let kind = parse_job_kind_filter(query.job_kind.as_deref(), true)?;
    let value = match kind {
        JobKindFilter::Transcription => {
            let record = state
                .transcription_store
                .get_record(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Transcription record not found"))?;
            if record.transcription_mode != TranscriptionRecordMode::Transcription {
                return Err(ApiError::not_found("Transcription record not found"));
            }
            map_record_with_kind(transcription_kind(record.transcription_mode), &record)?
        }
        JobKindFilter::SpeakerAttributedAsr => {
            let record = state
                .transcription_store
                .get_record(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Speaker-attributed ASR record not found"))?;
            if record.transcription_mode != TranscriptionRecordMode::SpeakerAttributedAsr {
                return Err(ApiError::not_found(
                    "Speaker-attributed ASR record not found",
                ));
            }
            map_record_with_kind(transcription_kind(record.transcription_mode), &record)?
        }
        JobKindFilter::Diarization => {
            let record = state
                .diarization_store
                .get_record(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Diarization record not found"))?;
            map_record_with_kind("diarization", &record)?
        }
        JobKindFilter::All => {
            if let Some(record) = state
                .transcription_store
                .get_record(record_id.clone())
                .await
                .map_err(map_store_error)?
            {
                map_record_with_kind(transcription_kind(record.transcription_mode), &record)?
            } else if let Some(record) = state
                .diarization_store
                .get_record(record_id)
                .await
                .map_err(map_store_error)?
            {
                map_record_with_kind("diarization", &record)?
            } else {
                return Err(ApiError::not_found("Speech text job record not found"));
            }
        }
    };

    Ok(Json(value))
}

pub async fn get_job_audio(
    State(state): State<AppState>,
    Path(record_id): Path<String>,
    Query(query): Query<UnifiedJobQuery>,
) -> Result<Response, ApiError> {
    let kind = parse_job_kind_filter(query.job_kind.as_deref(), true)?;
    match kind {
        JobKindFilter::Transcription => {
            ensure_transcription_audio_kind(
                &state,
                record_id.as_str(),
                TranscriptionRecordMode::Transcription,
            )
            .await?;
            let audio = state
                .transcription_store
                .get_audio(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Transcription audio not found"))?;
            Ok(transcription_audio_response(audio))
        }
        JobKindFilter::SpeakerAttributedAsr => {
            ensure_transcription_audio_kind(
                &state,
                record_id.as_str(),
                TranscriptionRecordMode::SpeakerAttributedAsr,
            )
            .await?;
            let audio = state
                .transcription_store
                .get_audio(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Speaker-attributed ASR audio not found"))?;
            Ok(transcription_audio_response(audio))
        }
        JobKindFilter::Diarization => {
            let audio = state
                .diarization_store
                .get_audio(record_id)
                .await
                .map_err(map_store_error)?
                .ok_or_else(|| ApiError::not_found("Diarization audio not found"))?;
            Ok(diarization_audio_response(audio))
        }
        JobKindFilter::All => {
            if let Some(audio) = state
                .transcription_store
                .get_audio(record_id.clone())
                .await
                .map_err(map_store_error)?
            {
                Ok(transcription_audio_response(audio))
            } else if let Some(audio) = state
                .diarization_store
                .get_audio(record_id)
                .await
                .map_err(map_store_error)?
            {
                Ok(diarization_audio_response(audio))
            } else {
                Err(ApiError::not_found("Speech text job audio not found"))
            }
        }
    }
}

fn parse_job_kind_filter(raw: Option<&str>, allow_all: bool) -> Result<JobKindFilter, ApiError> {
    let Some(value) = raw.map(|value| value.trim().to_ascii_lowercase()) else {
        return Ok(if allow_all {
            JobKindFilter::All
        } else {
            JobKindFilter::Transcription
        });
    };

    if value.is_empty() {
        return Ok(if allow_all {
            JobKindFilter::All
        } else {
            JobKindFilter::Transcription
        });
    }

    match value.as_str() {
        "all" if allow_all => Ok(JobKindFilter::All),
        "transcription" => Ok(JobKindFilter::Transcription),
        "speaker_attributed_asr" | "saa" => Ok(JobKindFilter::SpeakerAttributedAsr),
        "diarization" => Ok(JobKindFilter::Diarization),
        _ => Err(ApiError::bad_request(
            "Invalid `job_kind`. Supported values: all, transcription, speaker_attributed_asr, diarization.",
        )),
    }
}

fn map_transcription_summary(record: &TranscriptionRecordSummary) -> Value {
    json!({
        "kind": transcription_kind(record.transcription_mode),
        "id": record.id,
        "created_at": record.created_at,
        "transcription_mode": record.transcription_mode,
        "model_id": record.model_id,
        "language": record.language,
        "processing_status": record.processing_status,
        "processing_error": record.processing_error,
        "processing_progress": record.processing_progress,
        "duration_secs": record.duration_secs,
        "processing_time_ms": record.processing_time_ms,
        "rtf": record.rtf,
        "audio_mime_type": record.audio_mime_type,
        "audio_filename": record.audio_filename,
        "transcription_preview": record.transcription_preview,
        "transcription_chars": record.transcription_chars,
        "speaker_attributed_text_preview": record.speaker_attributed_text_preview,
        "speaker_attributed_text_chars": record.speaker_attributed_text_chars,
        "speaker_turn_count": record.speaker_turn_count,
        "saa_status": record.saa_status,
        "saa_warnings": record.saa_warnings,
        "summary_status": record.summary_status,
        "summary_preview": record.summary_preview,
        "summary_chars": record.summary_chars,
    })
}

fn map_diarization_summary(record: &DiarizationRecordSummary) -> Value {
    json!({
        "kind": "diarization",
        "id": record.id,
        "created_at": record.created_at,
        "model_id": record.model_id,
        "processing_status": record.processing_status,
        "processing_error": record.processing_error,
        "speaker_count": record.speaker_count,
        "corrected_speaker_count": record.corrected_speaker_count,
        "speaker_name_override_count": record.speaker_name_override_count,
        "duration_secs": record.duration_secs,
        "processing_time_ms": record.processing_time_ms,
        "rtf": record.rtf,
        "audio_mime_type": record.audio_mime_type,
        "audio_filename": record.audio_filename,
        "transcript_preview": record.transcript_preview,
        "transcript_chars": record.transcript_chars,
        "summary_status": record.summary_status,
        "summary_preview": record.summary_preview,
        "summary_chars": record.summary_chars,
    })
}

fn map_record_with_kind<T: Serialize>(kind: &str, record: &T) -> Result<Value, ApiError> {
    let mut payload = serde_json::to_value(record)
        .map_err(|err| ApiError::internal(format!("Failed serializing record: {err}")))?;
    let object = payload
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Failed serializing record payload"))?;
    object.insert("kind".to_string(), Value::String(kind.to_string()));
    Ok(payload)
}

fn transcription_kind(mode: TranscriptionRecordMode) -> &'static str {
    match mode {
        TranscriptionRecordMode::Transcription => "transcription",
        TranscriptionRecordMode::SpeakerAttributedAsr => "speaker_attributed_asr",
    }
}

async fn ensure_transcription_audio_kind(
    state: &AppState,
    record_id: &str,
    expected_mode: TranscriptionRecordMode,
) -> Result<(), ApiError> {
    let record = state
        .transcription_store
        .get_record(record_id.to_string())
        .await
        .map_err(map_store_error)?
        .ok_or_else(|| ApiError::not_found("Speech text job audio not found"))?;
    if record.transcription_mode != expected_mode {
        return Err(ApiError::not_found("Speech text job audio not found"));
    }
    Ok(())
}

fn transcription_audio_response(audio: StoredTranscriptionAudio) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    if let Some(filename) = audio.audio_filename {
        let disposition = format!("inline; filename=\"{}\"", filename.replace('"', ""));
        if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
            response = response.header(header::CONTENT_DISPOSITION, value);
        }
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn diarization_audio_response(audio: StoredDiarizationAudio) -> Response {
    let mut response = Response::builder().status(StatusCode::OK);

    if let Ok(content_type) = HeaderValue::from_str(audio.audio_mime_type.as_str()) {
        response = response.header(header::CONTENT_TYPE, content_type);
    }

    if let Some(filename) = audio.audio_filename {
        let disposition = format!("inline; filename=\"{}\"", filename.replace('"', ""));
        if let Ok(value) = HeaderValue::from_str(disposition.as_str()) {
            response = response.header(header::CONTENT_DISPOSITION, value);
        }
    }

    response
        .body(Body::from(audio.audio_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Storage error")
}

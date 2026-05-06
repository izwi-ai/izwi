//! Persistent diarization history storage backed by SQLite.

use anyhow::{anyhow, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaStorageProvider};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DbBackend, EntityTrait, QueryFilter, QueryResult, Set, Statement,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    db::StoreDatabase,
    entity::diarization_records,
    ids::new_uuid,
    persistence::{
        delete_media_object, persist_audio_object, read_media_object, LocalMediaStorageProvider,
    },
    storage_layout,
};

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationProcessingStatus {
    Pending,
    Processing,
    Ready,
    Failed,
}

impl Default for DiarizationProcessingStatus {
    fn default() -> Self {
        Self::Ready
    }
}

impl DiarizationProcessingStatus {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationSummaryStatus {
    NotRequested,
    Pending,
    Ready,
    Failed,
}

impl Default for DiarizationSummaryStatus {
    fn default() -> Self {
        Self::NotRequested
    }
}

impl DiarizationSummaryStatus {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegmentRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationWordRecord {
    pub word: String,
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub speaker_confidence: Option<f32>,
    pub overlaps_segment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationUtteranceRecord {
    pub speaker: String,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub processing_status: DiarizationProcessingStatus,
    pub processing_error: Option<String>,
    pub speaker_count: usize,
    pub corrected_speaker_count: usize,
    pub speaker_name_override_count: usize,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcript_preview: String,
    pub transcript_chars: usize,
    pub summary_status: DiarizationSummaryStatus,
    pub summary_preview: Option<String>,
    pub summary_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiarizationRecordListCursor {
    pub created_at: u64,
    pub id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub processing_status: DiarizationProcessingStatus,
    pub processing_error: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub corrected_speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub summary_status: DiarizationSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub speaker_name_overrides: BTreeMap<String, String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredDiarizationAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewDiarizationRecord {
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub processing_status: DiarizationProcessingStatus,
    pub processing_error: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub summary_status: DiarizationSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
    pub speaker_name_overrides: BTreeMap<String, String>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct UpdateDiarizationSummary {
    pub status: DiarizationSummaryStatus,
    pub model_id: Option<String>,
    pub text: Option<String>,
    pub error: Option<String>,
    pub updated_at: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CompleteDiarizationRecord {
    pub model_id: Option<String>,
    pub asr_model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub llm_model_id: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub min_speech_duration_ms: Option<f64>,
    pub min_silence_duration_ms: Option<f64>,
    pub enable_llm_refinement: bool,
    pub processing_time_ms: f64,
    pub duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub speaker_count: usize,
    pub alignment_coverage: Option<f64>,
    pub unattributed_words: usize,
    pub llm_refined: bool,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub summary_status: DiarizationSummaryStatus,
    pub summary_model_id: Option<String>,
    pub summary_text: Option<String>,
    pub summary_error: Option<String>,
    pub summary_updated_at: Option<u64>,
    pub segments: Vec<DiarizationSegmentRecord>,
    pub words: Vec<DiarizationWordRecord>,
    pub utterances: Vec<DiarizationUtteranceRecord>,
}

#[derive(Clone)]
pub struct DiarizationStore {
    db: StoreDatabase,
    media_storage: Arc<dyn MediaStorageProvider>,
}

impl DiarizationStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare diarization storage layout")?;

        Ok(Self {
            db: StoreDatabase::new(db_path),
            media_storage: Arc::new(LocalMediaStorageProvider::new(media_root)),
        })
    }

    pub fn initialize_with_storage(
        db: StoreDatabase,
        media_storage: Arc<dyn MediaStorageProvider>,
    ) -> Self {
        Self { db, media_storage }
    }

    pub async fn list_records_page(
        &self,
        limit: usize,
        cursor: Option<DiarizationRecordListCursor>,
    ) -> anyhow::Result<(
        Vec<DiarizationRecordSummary>,
        Option<DiarizationRecordListCursor>,
    )> {
        let db = self.db.connection().await?;
        let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);
        let fetch_limit = list_limit.saturating_add(1);

        let rows = if let Some(cursor) = cursor {
            let cursor_created_at = i64::try_from(cursor.created_at).unwrap_or(i64::MAX);
            db.query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                DIARIZATION_PAGE_AFTER_CURSOR_SQL,
                vec![
                    cursor_created_at.into(),
                    cursor.id.into(),
                    fetch_limit.into(),
                ],
            ))
            .await
        } else {
            db.query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                DIARIZATION_PAGE_SQL,
                vec![fetch_limit.into()],
            ))
            .await
        }
        .context("Failed to list diarization records")?;

        let mut records = rows
            .iter()
            .map(map_diarization_summary)
            .collect::<anyhow::Result<Vec<_>>>()?;

        let page_limit = usize::try_from(list_limit).unwrap_or(200);
        let has_more = records.len() > page_limit;
        if has_more {
            records.truncate(page_limit);
        }
        let next_cursor = if has_more {
            records.last().map(|record| DiarizationRecordListCursor {
                created_at: record.created_at,
                id: record.id.clone(),
            })
        } else {
            None
        };

        Ok((records, next_cursor))
    }

    pub async fn get_record(&self, record_id: String) -> anyhow::Result<Option<DiarizationRecord>> {
        let db = self.db.connection().await?;
        fetch_record_without_audio(db, &record_id).await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredDiarizationAudio>> {
        let db = self.db.connection().await?;
        let audio = db
            .query_one_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                r#"
                SELECT audio_storage_path, audio_mime_type, audio_filename
                FROM diarization_records
                WHERE id = ?1
                "#,
                vec![record_id.into()],
            ))
            .await
            .context("Failed to load diarization audio metadata")?;
        let Some(row) = audio else {
            return Ok(None);
        };

        let audio_storage_path: String = row.try_get_by_index(0)?;
        let audio_mime_type: String = row.try_get_by_index(1)?;
        let audio_filename: Option<String> = row.try_get_by_index(2)?;
        let audio = read_media_object(&self.media_storage, audio_storage_path.as_str())
            .await
            .context("Failed to read diarization media")?;

        Ok(Some(StoredDiarizationAudio {
            audio_bytes: audio.bytes,
            audio_mime_type,
            audio_filename,
        }))
    }

    pub async fn create_record(
        &self,
        record: NewDiarizationRecord,
    ) -> anyhow::Result<DiarizationRecord> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let record_id = new_uuid();

        let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
        let asr_model_id = sanitize_optional_text(record.asr_model_id.as_deref(), 160);
        let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
        let llm_model_id = sanitize_optional_text(record.llm_model_id.as_deref(), 160);
        let processing_error = sanitize_optional_text(record.processing_error.as_deref(), 1_000);
        let processing_status =
            normalize_processing_status(record.processing_status, processing_error.as_deref());
        let min_speakers = record
            .min_speakers
            .and_then(|value| i64::try_from(value).ok());
        let max_speakers = record
            .max_speakers
            .and_then(|value| i64::try_from(value).ok());
        let min_speech_duration_ms = record
            .min_speech_duration_ms
            .filter(|value| value.is_finite() && *value >= 0.0);
        let min_silence_duration_ms = record
            .min_silence_duration_ms
            .filter(|value| value.is_finite() && *value >= 0.0);
        let processing_time_ms = if record.processing_time_ms.is_finite() {
            record.processing_time_ms.max(0.0)
        } else {
            0.0
        };
        let duration_secs = record
            .duration_secs
            .filter(|value| value.is_finite() && *value >= 0.0);
        let rtf = record
            .rtf
            .filter(|value| value.is_finite() && *value >= 0.0);
        let speaker_count = i64::try_from(record.speaker_count).unwrap_or(0);
        let alignment_coverage = record
            .alignment_coverage
            .filter(|value| value.is_finite() && *value >= 0.0);
        let unattributed_words = i64::try_from(record.unattributed_words).unwrap_or(0);
        let asr_text = sanitize_required_text(record.asr_text.as_str(), 40_000);
        let raw_transcript = sanitize_required_text(record.raw_transcript.as_str(), 100_000);
        let transcript = sanitize_required_text(record.transcript.as_str(), 100_000);
        let summary_model_id = sanitize_optional_text(record.summary_model_id.as_deref(), 160);
        let summary_text = sanitize_optional_text(record.summary_text.as_deref(), 20_000);
        let summary_error = sanitize_optional_text(record.summary_error.as_deref(), 1_000);
        let summary_status = normalize_summary_status(
            record.summary_status,
            summary_text.as_deref(),
            summary_error.as_deref(),
        );
        let summary_updated_at = normalize_optional_timestamp_i64(record.summary_updated_at)
            .or_else(|| {
                if summary_status == DiarizationSummaryStatus::NotRequested {
                    None
                } else {
                    Some(now)
                }
            });
        let speaker_name_overrides = sanitize_speaker_name_overrides(
            &record.speaker_name_overrides,
            &raw_speaker_labels_from_parts(&record.segments, &record.words, &record.utterances),
        )?;
        let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
        let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);

        if record.audio_bytes.is_empty() {
            return Err(anyhow!("Audio payload cannot be empty"));
        }

        let audio_storage_path = persist_audio_object(
            &self.media_storage,
            MediaNamespace::DiarizationUpload,
            &record_id,
            audio_filename.as_deref(),
            audio_mime_type.as_str(),
            &record.audio_bytes,
            HookMetadata::new(),
        )
        .await?;

        let segments_json =
            serde_json::to_string(&record.segments).context("Failed serializing segments")?;
        let words_json =
            serde_json::to_string(&record.words).context("Failed serializing words")?;
        let utterances_json =
            serde_json::to_string(&record.utterances).context("Failed serializing utterances")?;
        let speaker_name_overrides_json = serde_json::to_string(&speaker_name_overrides)
            .context("Failed serializing speaker name overrides")?;

        if let Err(err) = diarization_records::Entity::insert(diarization_records::ActiveModel {
            id: Set(record_id.clone()),
            created_at: Set(now),
            model_id: Set(model_id),
            asr_model_id: Set(asr_model_id),
            aligner_model_id: Set(aligner_model_id),
            llm_model_id: Set(llm_model_id),
            processing_status: Set(processing_status.as_db_value().to_string()),
            processing_error: Set(processing_error),
            min_speakers: Set(min_speakers),
            max_speakers: Set(max_speakers),
            min_speech_duration_ms: Set(min_speech_duration_ms),
            min_silence_duration_ms: Set(min_silence_duration_ms),
            enable_llm_refinement: Set(if record.enable_llm_refinement {
                1_i64
            } else {
                0_i64
            }),
            processing_time_ms: Set(processing_time_ms),
            duration_secs: Set(duration_secs),
            rtf: Set(rtf),
            speaker_count: Set(speaker_count),
            alignment_coverage: Set(alignment_coverage),
            unattributed_words: Set(unattributed_words),
            llm_refined: Set(if record.llm_refined { 1_i64 } else { 0_i64 }),
            asr_text: Set(asr_text),
            raw_transcript: Set(raw_transcript),
            transcript: Set(transcript),
            summary_status: Set(summary_status.as_db_value().to_string()),
            summary_model_id: Set(summary_model_id),
            summary_text: Set(summary_text),
            summary_error: Set(summary_error),
            summary_updated_at: Set(summary_updated_at),
            segments_json: Set(segments_json),
            words_json: Set(words_json),
            utterances_json: Set(utterances_json),
            speaker_name_overrides_json: Set(speaker_name_overrides_json),
            audio_mime_type: Set(audio_mime_type),
            audio_filename: Set(audio_filename),
            audio_storage_path: Set(audio_storage_path.clone()),
        })
        .exec(db)
        .await
        {
            let _ =
                delete_media_object(&self.media_storage, Some(audio_storage_path.as_str())).await;
            return Err(err).context("Failed to insert diarization record");
        }

        fetch_record_without_audio(db, &record_id)
            .await?
            .ok_or_else(|| anyhow!("Failed to fetch created diarization record"))
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let audio_storage_path = fetch_audio_storage_path(db, &record_id).await?.flatten();
        let result = diarization_records::Entity::delete_many()
            .filter(diarization_records::Column::Id.eq(record_id))
            .exec(db)
            .await
            .context("Failed to delete diarization record")?;

        if result.rows_affected > 0 {
            delete_media_object(&self.media_storage, audio_storage_path.as_deref()).await?;
        }

        Ok(result.rows_affected > 0)
    }

    pub async fn update_speaker_name_overrides(
        &self,
        record_id: String,
        speaker_name_overrides: BTreeMap<String, String>,
    ) -> anyhow::Result<Option<DiarizationRecord>> {
        let db = self.db.connection().await?;
        let existing = fetch_record_without_audio(db, &record_id).await?;
        let Some(existing) = existing else {
            return Ok(None);
        };

        let sanitized_overrides = sanitize_speaker_name_overrides(
            &speaker_name_overrides,
            &raw_speaker_labels_from_parts(
                &existing.segments,
                &existing.words,
                &existing.utterances,
            ),
        )?;
        let speaker_name_overrides_json = serde_json::to_string(&sanitized_overrides)
            .context("Failed serializing speaker name overrides")?;

        diarization_records::Entity::update_many()
            .col_expr(
                diarization_records::Column::SpeakerNameOverridesJson,
                Expr::value(speaker_name_overrides_json),
            )
            .filter(diarization_records::Column::Id.eq(record_id))
            .exec(db)
            .await
            .context("Failed updating speaker name overrides")?;

        fetch_record_without_audio(db, &existing.id).await
    }

    pub async fn update_summary(
        &self,
        record_id: String,
        update: UpdateDiarizationSummary,
    ) -> anyhow::Result<Option<DiarizationRecord>> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();

        let summary_model_id = sanitize_optional_text(update.model_id.as_deref(), 160);
        let summary_text = sanitize_optional_text(update.text.as_deref(), 20_000);
        let summary_error = sanitize_optional_text(update.error.as_deref(), 1_000);
        let summary_status = normalize_summary_status(
            update.status,
            summary_text.as_deref(),
            summary_error.as_deref(),
        );
        let summary_updated_at =
            normalize_optional_timestamp_i64(update.updated_at).or_else(|| {
                if summary_status == DiarizationSummaryStatus::NotRequested {
                    None
                } else {
                    Some(now)
                }
            });

        let result = diarization_records::Entity::update_many()
            .col_expr(
                diarization_records::Column::SummaryStatus,
                Expr::value(summary_status.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::SummaryModelId,
                Expr::value(summary_model_id),
            )
            .col_expr(
                diarization_records::Column::SummaryText,
                Expr::value(summary_text),
            )
            .col_expr(
                diarization_records::Column::SummaryError,
                Expr::value(summary_error),
            )
            .col_expr(
                diarization_records::Column::SummaryUpdatedAt,
                Expr::value(summary_updated_at),
            )
            .filter(diarization_records::Column::Id.eq(record_id.clone()))
            .exec(db)
            .await
            .context("Failed updating diarization summary")?;

        if result.rows_affected == 0 {
            return Ok(None);
        }

        fetch_record_without_audio(db, &record_id).await
    }

    pub async fn update_processing_status(
        &self,
        record_id: String,
        status: DiarizationProcessingStatus,
        error: Option<String>,
    ) -> anyhow::Result<Option<DiarizationRecord>> {
        let db = self.db.connection().await?;
        let processing_error = sanitize_optional_text(error.as_deref(), 1_000);
        let processing_status = normalize_processing_status(status, processing_error.as_deref());

        let result = diarization_records::Entity::update_many()
            .col_expr(
                diarization_records::Column::ProcessingStatus,
                Expr::value(processing_status.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::ProcessingError,
                Expr::value(processing_error),
            )
            .filter(diarization_records::Column::Id.eq(record_id.clone()))
            .exec(db)
            .await
            .context("Failed updating diarization processing status")?;

        if result.rows_affected == 0 {
            return Ok(None);
        }

        fetch_record_without_audio(db, &record_id).await
    }

    pub async fn complete_record(
        &self,
        record_id: String,
        record: CompleteDiarizationRecord,
    ) -> anyhow::Result<Option<DiarizationRecord>> {
        let db = self.db.connection().await?;

        let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
        let asr_model_id = sanitize_optional_text(record.asr_model_id.as_deref(), 160);
        let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
        let llm_model_id = sanitize_optional_text(record.llm_model_id.as_deref(), 160);
        let min_speakers = record
            .min_speakers
            .and_then(|value| i64::try_from(value).ok());
        let max_speakers = record
            .max_speakers
            .and_then(|value| i64::try_from(value).ok());
        let min_speech_duration_ms = record
            .min_speech_duration_ms
            .filter(|value| value.is_finite() && *value >= 0.0);
        let min_silence_duration_ms = record
            .min_silence_duration_ms
            .filter(|value| value.is_finite() && *value >= 0.0);
        let processing_time_ms = if record.processing_time_ms.is_finite() {
            record.processing_time_ms.max(0.0)
        } else {
            0.0
        };
        let duration_secs = record
            .duration_secs
            .filter(|value| value.is_finite() && *value >= 0.0);
        let rtf = record
            .rtf
            .filter(|value| value.is_finite() && *value >= 0.0);
        let speaker_count = i64::try_from(record.speaker_count).unwrap_or(0);
        let alignment_coverage = record
            .alignment_coverage
            .filter(|value| value.is_finite() && *value >= 0.0);
        let unattributed_words = i64::try_from(record.unattributed_words).unwrap_or(0);
        let asr_text = sanitize_required_text(record.asr_text.as_str(), 40_000);
        let raw_transcript = sanitize_required_text(record.raw_transcript.as_str(), 100_000);
        let transcript = sanitize_required_text(record.transcript.as_str(), 100_000);
        let segments_json =
            serde_json::to_string(&record.segments).context("Failed serializing segments")?;
        let words_json =
            serde_json::to_string(&record.words).context("Failed serializing words")?;
        let utterances_json =
            serde_json::to_string(&record.utterances).context("Failed serializing utterances")?;
        let summary_model_id = sanitize_optional_text(record.summary_model_id.as_deref(), 160);
        let summary_text = sanitize_optional_text(record.summary_text.as_deref(), 20_000);
        let summary_error = sanitize_optional_text(record.summary_error.as_deref(), 1_000);
        let summary_status = normalize_summary_status(
            record.summary_status,
            summary_text.as_deref(),
            summary_error.as_deref(),
        );
        let summary_updated_at = normalize_optional_timestamp_i64(record.summary_updated_at);

        let result = diarization_records::Entity::update_many()
            .col_expr(diarization_records::Column::ModelId, Expr::value(model_id))
            .col_expr(
                diarization_records::Column::AsrModelId,
                Expr::value(asr_model_id),
            )
            .col_expr(
                diarization_records::Column::AlignerModelId,
                Expr::value(aligner_model_id),
            )
            .col_expr(
                diarization_records::Column::LlmModelId,
                Expr::value(llm_model_id),
            )
            .col_expr(
                diarization_records::Column::ProcessingStatus,
                Expr::value(DiarizationProcessingStatus::Ready.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::ProcessingError,
                Expr::value(Option::<String>::None),
            )
            .col_expr(
                diarization_records::Column::MinSpeakers,
                Expr::value(min_speakers),
            )
            .col_expr(
                diarization_records::Column::MaxSpeakers,
                Expr::value(max_speakers),
            )
            .col_expr(
                diarization_records::Column::MinSpeechDurationMs,
                Expr::value(min_speech_duration_ms),
            )
            .col_expr(
                diarization_records::Column::MinSilenceDurationMs,
                Expr::value(min_silence_duration_ms),
            )
            .col_expr(
                diarization_records::Column::EnableLlmRefinement,
                Expr::value(if record.enable_llm_refinement {
                    1_i64
                } else {
                    0_i64
                }),
            )
            .col_expr(
                diarization_records::Column::ProcessingTimeMs,
                Expr::value(processing_time_ms),
            )
            .col_expr(
                diarization_records::Column::DurationSecs,
                Expr::value(duration_secs),
            )
            .col_expr(diarization_records::Column::Rtf, Expr::value(rtf))
            .col_expr(
                diarization_records::Column::SpeakerCount,
                Expr::value(speaker_count),
            )
            .col_expr(
                diarization_records::Column::AlignmentCoverage,
                Expr::value(alignment_coverage),
            )
            .col_expr(
                diarization_records::Column::UnattributedWords,
                Expr::value(unattributed_words),
            )
            .col_expr(
                diarization_records::Column::LlmRefined,
                Expr::value(if record.llm_refined { 1_i64 } else { 0_i64 }),
            )
            .col_expr(diarization_records::Column::AsrText, Expr::value(asr_text))
            .col_expr(
                diarization_records::Column::RawTranscript,
                Expr::value(raw_transcript),
            )
            .col_expr(
                diarization_records::Column::Transcript,
                Expr::value(transcript),
            )
            .col_expr(
                diarization_records::Column::SummaryStatus,
                Expr::value(summary_status.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::SummaryModelId,
                Expr::value(summary_model_id),
            )
            .col_expr(
                diarization_records::Column::SummaryText,
                Expr::value(summary_text),
            )
            .col_expr(
                diarization_records::Column::SummaryError,
                Expr::value(summary_error),
            )
            .col_expr(
                diarization_records::Column::SummaryUpdatedAt,
                Expr::value(summary_updated_at),
            )
            .col_expr(
                diarization_records::Column::SegmentsJson,
                Expr::value(segments_json),
            )
            .col_expr(
                diarization_records::Column::WordsJson,
                Expr::value(words_json),
            )
            .col_expr(
                diarization_records::Column::UtterancesJson,
                Expr::value(utterances_json),
            )
            .filter(diarization_records::Column::Id.eq(record_id.clone()))
            .filter(
                diarization_records::Column::ProcessingStatus
                    .is_in(["pending".to_string(), "processing".to_string()]),
            )
            .exec(db)
            .await
            .context("Failed to complete diarization record")?;

        if result.rows_affected == 0 {
            return Ok(None);
        }

        fetch_record_without_audio(db, &record_id).await
    }
}

const DIARIZATION_PAGE_SQL: &str = r#"
    SELECT
        id,
        created_at,
        model_id,
        processing_status,
        processing_error,
        speaker_count,
        utterances_json,
        speaker_name_overrides_json,
        duration_secs,
        processing_time_ms,
        rtf,
        audio_mime_type,
        audio_filename,
        transcript,
        summary_status,
        summary_text
    FROM diarization_records
    ORDER BY created_at DESC, id DESC
    LIMIT ?1
"#;

const DIARIZATION_PAGE_AFTER_CURSOR_SQL: &str = r#"
    SELECT
        id,
        created_at,
        model_id,
        processing_status,
        processing_error,
        speaker_count,
        utterances_json,
        speaker_name_overrides_json,
        duration_secs,
        processing_time_ms,
        rtf,
        audio_mime_type,
        audio_filename,
        transcript,
        summary_status,
        summary_text
    FROM diarization_records
    WHERE created_at < ?1 OR (created_at = ?1 AND id < ?2)
    ORDER BY created_at DESC, id DESC
    LIMIT ?3
"#;

const DIARIZATION_RECORD_COLUMNS: &str = r#"
    id,
    created_at,
    model_id,
    asr_model_id,
    aligner_model_id,
    llm_model_id,
    processing_status,
    processing_error,
    min_speakers,
    max_speakers,
    min_speech_duration_ms,
    min_silence_duration_ms,
    enable_llm_refinement,
    processing_time_ms,
    duration_secs,
    rtf,
    speaker_count,
    alignment_coverage,
    unattributed_words,
    llm_refined,
    asr_text,
    raw_transcript,
    transcript,
    summary_status,
    summary_model_id,
    summary_text,
    summary_error,
    summary_updated_at,
    segments_json,
    words_json,
    utterances_json,
    speaker_name_overrides_json,
    audio_mime_type,
    audio_filename
"#;

async fn fetch_record_without_audio(
    db: &sea_orm::DatabaseConnection,
    record_id: &str,
) -> anyhow::Result<Option<DiarizationRecord>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            format!("SELECT {DIARIZATION_RECORD_COLUMNS} FROM diarization_records WHERE id = ?1"),
            vec![record_id.into()],
        ))
        .await
        .context("Failed to load diarization record")?;
    row.as_ref().map(map_diarization_record).transpose()
}

async fn fetch_audio_storage_path(
    db: &sea_orm::DatabaseConnection,
    record_id: &str,
) -> anyhow::Result<Option<Option<String>>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            "SELECT audio_storage_path FROM diarization_records WHERE id = ?1",
            vec![record_id.into()],
        ))
        .await
        .context("Failed to load diarization media path")?;
    row.map(|row| row.try_get_by_index::<Option<String>>(0))
        .transpose()
        .map_err(Into::into)
}

fn map_diarization_summary(row: &QueryResult) -> anyhow::Result<DiarizationRecordSummary> {
    let speaker_count = row
        .try_get_by_index::<Option<i64>>(5)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let utterances: Vec<DiarizationUtteranceRecord> = parse_json_vec(row.try_get_by_index(6)?);
    let speaker_name_overrides = parse_json_map(row.try_get_by_index(7)?);
    let transcript: String = row.try_get_by_index(13)?;
    let summary_status = parse_summary_status(row.try_get_by_index(14)?);
    let summary_text: Option<String> = row.try_get_by_index(15)?;

    Ok(DiarizationRecordSummary {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        model_id: row.try_get_by_index(2)?,
        processing_status: parse_processing_status(row.try_get_by_index(3)?),
        processing_error: row.try_get_by_index(4)?,
        speaker_count,
        corrected_speaker_count: corrected_speaker_count_from_parts(
            speaker_count,
            &[],
            &[],
            &utterances,
            &speaker_name_overrides,
        ),
        speaker_name_override_count: speaker_name_overrides.len(),
        duration_secs: row.try_get_by_index(8)?,
        processing_time_ms: row.try_get_by_index(9)?,
        rtf: row.try_get_by_index(10)?,
        audio_mime_type: row.try_get_by_index(11)?,
        audio_filename: row.try_get_by_index(12)?,
        transcript_preview: transcript_preview_with_utterances(
            &utterances,
            &speaker_name_overrides,
            transcript.as_str(),
        ),
        transcript_chars: transcript.chars().count(),
        summary_status,
        summary_preview: summary_preview(summary_text.as_deref()),
        summary_chars: summary_text
            .as_ref()
            .map(|text| text.chars().count())
            .unwrap_or(0),
    })
}

fn map_diarization_record(row: &QueryResult) -> anyhow::Result<DiarizationRecord> {
    let min_speakers = row
        .try_get_by_index::<Option<i64>>(8)?
        .and_then(i64_to_usize);
    let max_speakers = row
        .try_get_by_index::<Option<i64>>(9)?
        .and_then(i64_to_usize);
    let speaker_count = row
        .try_get_by_index::<Option<i64>>(16)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let unattributed_words = row
        .try_get_by_index::<Option<i64>>(18)?
        .and_then(i64_to_usize)
        .unwrap_or(0);
    let segments_raw: String = row.try_get_by_index(28)?;
    let words_raw: String = row.try_get_by_index(29)?;
    let utterances_raw: String = row.try_get_by_index(30)?;
    let speaker_name_overrides = parse_json_map(row.try_get_by_index(31)?);
    let segments: Vec<DiarizationSegmentRecord> = parse_json_vec(Some(segments_raw));
    let words: Vec<DiarizationWordRecord> = parse_json_vec(Some(words_raw));
    let utterances: Vec<DiarizationUtteranceRecord> = parse_json_vec(Some(utterances_raw));
    let corrected_speaker_count = corrected_speaker_count_from_parts(
        speaker_count,
        &segments,
        &words,
        &utterances,
        &speaker_name_overrides,
    );

    Ok(DiarizationRecord {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        model_id: row.try_get_by_index(2)?,
        asr_model_id: row.try_get_by_index(3)?,
        aligner_model_id: row.try_get_by_index(4)?,
        llm_model_id: row.try_get_by_index(5)?,
        processing_status: parse_processing_status(row.try_get_by_index(6)?),
        processing_error: row.try_get_by_index(7)?,
        min_speakers,
        max_speakers,
        min_speech_duration_ms: row.try_get_by_index(10)?,
        min_silence_duration_ms: row.try_get_by_index(11)?,
        enable_llm_refinement: row.try_get_by_index::<i64>(12)? > 0,
        processing_time_ms: row.try_get_by_index(13)?,
        duration_secs: row.try_get_by_index(14)?,
        rtf: row.try_get_by_index(15)?,
        speaker_count,
        corrected_speaker_count,
        alignment_coverage: row.try_get_by_index(17)?,
        unattributed_words,
        llm_refined: row.try_get_by_index::<i64>(19)? > 0,
        asr_text: row.try_get_by_index(20)?,
        raw_transcript: row.try_get_by_index(21)?,
        transcript: row.try_get_by_index(22)?,
        summary_status: parse_summary_status(row.try_get_by_index(23)?),
        summary_model_id: row.try_get_by_index(24)?,
        summary_text: row.try_get_by_index(25)?,
        summary_error: row.try_get_by_index(26)?,
        summary_updated_at: row.try_get_by_index::<Option<i64>>(27)?.map(i64_to_u64),
        segments,
        words,
        utterances,
        speaker_name_overrides,
        audio_mime_type: row.try_get_by_index(32)?,
        audio_filename: row.try_get_by_index(33)?,
    })
}

fn parse_json_vec<T>(raw: Option<String>) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    raw.and_then(|value| serde_json::from_str::<Vec<T>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn parse_json_map(raw: Option<String>) -> BTreeMap<String, String> {
    raw.and_then(|value| serde_json::from_str::<BTreeMap<String, String>>(value.as_str()).ok())
        .unwrap_or_default()
}

fn parse_processing_status(raw: Option<String>) -> DiarizationProcessingStatus {
    match raw
        .unwrap_or_else(|| "ready".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pending" => DiarizationProcessingStatus::Pending,
        "processing" => DiarizationProcessingStatus::Processing,
        "failed" => DiarizationProcessingStatus::Failed,
        _ => DiarizationProcessingStatus::Ready,
    }
}

fn parse_summary_status(raw: Option<String>) -> DiarizationSummaryStatus {
    match raw
        .unwrap_or_else(|| "not_requested".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pending" => DiarizationSummaryStatus::Pending,
        "ready" => DiarizationSummaryStatus::Ready,
        "failed" => DiarizationSummaryStatus::Failed,
        _ => DiarizationSummaryStatus::NotRequested,
    }
}

fn normalize_processing_status(
    requested: DiarizationProcessingStatus,
    processing_error: Option<&str>,
) -> DiarizationProcessingStatus {
    if processing_error.is_some() {
        return DiarizationProcessingStatus::Failed;
    }

    requested
}

fn normalize_summary_status(
    requested: DiarizationSummaryStatus,
    summary_text: Option<&str>,
    summary_error: Option<&str>,
) -> DiarizationSummaryStatus {
    if summary_text.is_some() {
        return DiarizationSummaryStatus::Ready;
    }
    if summary_error.is_some() {
        return DiarizationSummaryStatus::Failed;
    }
    match requested {
        DiarizationSummaryStatus::Ready => DiarizationSummaryStatus::NotRequested,
        DiarizationSummaryStatus::Failed => DiarizationSummaryStatus::NotRequested,
        other => other,
    }
}

fn transcript_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 180)
}

fn summary_preview(content: Option<&str>) -> Option<String> {
    let value = content.unwrap_or("").trim();
    if value.is_empty() {
        return None;
    }

    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(truncate_string(&normalized, 200))
    }
}

fn transcript_preview_with_utterances(
    utterances: &[DiarizationUtteranceRecord],
    speaker_name_overrides: &BTreeMap<String, String>,
    transcript: &str,
) -> String {
    if utterances.is_empty() {
        return transcript_preview(transcript);
    }

    let normalized = utterances
        .iter()
        .filter_map(|utterance| {
            let text = utterance.text.trim();
            if text.is_empty() {
                return None;
            }
            Some(format!(
                "{} [{:.2}s - {:.2}s]: {}",
                resolve_speaker_label(utterance.speaker.as_str(), speaker_name_overrides),
                utterance.start,
                utterance.end,
                text
            ))
        })
        .collect::<Vec<_>>()
        .join(" ");

    if normalized.is_empty() {
        transcript_preview(transcript)
    } else {
        truncate_string(&normalized, 180)
    }
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        " ".to_string()
    } else {
        truncate_string(trimmed, max_chars)
    }
}

fn sanitize_optional_text(raw: Option<&str>, max_chars: usize) -> Option<String> {
    let normalized = raw
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(truncate_string(&normalized, max_chars))
    }
}

fn sanitize_audio_mime_type(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "audio/wav".to_string()
    } else {
        truncate_string(trimmed, 80)
    }
}

fn normalize_optional_timestamp_i64(value: Option<u64>) -> Option<i64> {
    value.and_then(|raw| i64::try_from(raw).ok())
}

fn resolve_speaker_label(
    raw_label: &str,
    speaker_name_overrides: &BTreeMap<String, String>,
) -> String {
    speaker_name_overrides
        .get(raw_label)
        .cloned()
        .unwrap_or_else(|| raw_label.to_string())
}

fn raw_speaker_labels_from_parts(
    segments: &[DiarizationSegmentRecord],
    words: &[DiarizationWordRecord],
    utterances: &[DiarizationUtteranceRecord],
) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();

    for segment in segments {
        let trimmed = segment.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    for word in words {
        let trimmed = word.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    for utterance in utterances {
        let trimmed = utterance.speaker.trim();
        if !trimmed.is_empty() {
            labels.insert(trimmed.to_string());
        }
    }

    labels
}

fn corrected_speaker_count_from_parts(
    raw_speaker_count: usize,
    segments: &[DiarizationSegmentRecord],
    words: &[DiarizationWordRecord],
    utterances: &[DiarizationUtteranceRecord],
    speaker_name_overrides: &BTreeMap<String, String>,
) -> usize {
    let raw_labels = raw_speaker_labels_from_parts(segments, words, utterances);
    if raw_labels.is_empty() {
        return raw_speaker_count;
    }

    raw_labels
        .into_iter()
        .map(|label| resolve_speaker_label(label.as_str(), speaker_name_overrides))
        .collect::<BTreeSet<_>>()
        .len()
}

fn sanitize_speaker_name_overrides(
    overrides: &BTreeMap<String, String>,
    known_labels: &BTreeSet<String>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let mut sanitized = BTreeMap::new();

    for (raw_label, corrected_label) in overrides {
        let raw = sanitize_required_text(raw_label.as_str(), 80);
        if !known_labels.contains(&raw) {
            return Err(anyhow!("Unknown diarization speaker label: {raw}"));
        }

        let normalized = corrected_label
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if normalized.is_empty() {
            continue;
        }

        let corrected = truncate_string(normalized.as_str(), 80);
        if corrected != raw {
            sanitized.insert(raw, corrected);
        }
    }

    Ok(sanitized)
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        result.push(ch);
    }
    if input.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

fn now_unix_millis_i64() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

fn i64_to_u64(value: i64) -> u64 {
    if value.is_negative() {
        0
    } else {
        value as u64
    }
}

fn i64_to_usize(value: i64) -> Option<usize> {
    if value.is_negative() {
        None
    } else {
        usize::try_from(value).ok()
    }
}

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_store() -> (DiarizationStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "izwi-diarization-store-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let store = DiarizationStore::initialize_at(root.join("izwi.sqlite3"), root.join("media"))
            .expect("test store should initialize");
        (store, root)
    }

    fn sample_record() -> NewDiarizationRecord {
        NewDiarizationRecord {
            model_id: Some("diar_streaming_sortformer_4spk-v2.1".to_string()),
            asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            llm_model_id: Some("Qwen3-1.7B-GGUF".to_string()),
            processing_status: DiarizationProcessingStatus::Ready,
            processing_error: None,
            min_speakers: Some(1),
            max_speakers: Some(4),
            min_speech_duration_ms: Some(240.0),
            min_silence_duration_ms: Some(200.0),
            enable_llm_refinement: true,
            processing_time_ms: 500.0,
            duration_secs: Some(12.0),
            rtf: Some(0.4),
            speaker_count: 2,
            alignment_coverage: Some(0.95),
            unattributed_words: 0,
            llm_refined: true,
            asr_text: "hello there".to_string(),
            raw_transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            summary_status: DiarizationSummaryStatus::Ready,
            summary_model_id: Some("Qwen3.5-4B".to_string()),
            summary_text: Some("Alice greets, then Bob responds.".to_string()),
            summary_error: None,
            summary_updated_at: Some(1),
            segments: vec![
                DiarizationSegmentRecord {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.2,
                    confidence: Some(0.9),
                },
                DiarizationSegmentRecord {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.2,
                    end: 2.5,
                    confidence: Some(0.88),
                },
            ],
            words: vec![
                DiarizationWordRecord {
                    word: "Hello".to_string(),
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 0.4,
                    speaker_confidence: Some(0.92),
                    overlaps_segment: true,
                },
                DiarizationWordRecord {
                    word: "there".to_string(),
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.3,
                    end: 1.8,
                    speaker_confidence: Some(0.89),
                    overlaps_segment: true,
                },
            ],
            utterances: vec![
                DiarizationUtteranceRecord {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.2,
                    text: "Hello there.".to_string(),
                    word_start: 0,
                    word_end: 0,
                },
                DiarizationUtteranceRecord {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.2,
                    end: 2.5,
                    text: "Hi back.".to_string(),
                    word_start: 1,
                    word_end: 1,
                },
            ],
            speaker_name_overrides: BTreeMap::new(),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
            audio_bytes: vec![0_u8, 1_u8, 2_u8, 3_u8],
        }
    }

    fn sample_complete_record() -> CompleteDiarizationRecord {
        CompleteDiarizationRecord {
            model_id: Some("diar_streaming_sortformer_4spk-v2.1".to_string()),
            asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            llm_model_id: Some("Qwen3.5-4B".to_string()),
            min_speakers: Some(1),
            max_speakers: Some(4),
            min_speech_duration_ms: Some(240.0),
            min_silence_duration_ms: Some(200.0),
            enable_llm_refinement: true,
            processing_time_ms: 480.0,
            duration_secs: Some(12.0),
            rtf: Some(0.4),
            speaker_count: 2,
            alignment_coverage: Some(0.95),
            unattributed_words: 0,
            llm_refined: true,
            asr_text: "hello there".to_string(),
            raw_transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            transcript: "SPEAKER_00 [0.00s - 1.20s]: Hello there.".to_string(),
            summary_status: DiarizationSummaryStatus::Pending,
            summary_model_id: Some("Qwen3.5-4B".to_string()),
            summary_text: None,
            summary_error: None,
            summary_updated_at: None,
            segments: sample_record().segments,
            words: sample_record().words,
            utterances: sample_record().utterances,
        }
    }

    #[tokio::test]
    async fn persists_speaker_name_overrides_and_corrected_summary() {
        let (store, root) = build_test_store();

        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");
        assert_eq!(created.corrected_speaker_count, 2);
        assert!(created.speaker_name_overrides.is_empty());
        assert_eq!(created.summary_status, DiarizationSummaryStatus::Ready);
        assert_eq!(
            created.summary_text.as_deref(),
            Some("Alice greets, then Bob responds.")
        );

        let updated = store
            .update_speaker_name_overrides(
                created.id.clone(),
                BTreeMap::from([
                    ("SPEAKER_00".to_string(), "Alice".to_string()),
                    ("SPEAKER_01".to_string(), "Alice".to_string()),
                ]),
            )
            .await
            .expect("update should succeed")
            .expect("record should exist");

        assert_eq!(updated.corrected_speaker_count, 1);
        assert_eq!(
            updated.speaker_name_overrides.get("SPEAKER_00"),
            Some(&"Alice".to_string())
        );

        let (summaries, next_cursor) = store
            .list_records_page(10, None)
            .await
            .expect("list should succeed");
        assert_eq!(summaries.len(), 1);
        assert!(next_cursor.is_none());
        assert_eq!(
            summaries[0].processing_status,
            DiarizationProcessingStatus::Ready
        );
        assert_eq!(summaries[0].speaker_count, 2);
        assert_eq!(summaries[0].corrected_speaker_count, 1);
        assert_eq!(summaries[0].speaker_name_override_count, 2);
        assert!(summaries[0].transcript_preview.contains("Alice"));
        assert_eq!(summaries[0].summary_status, DiarizationSummaryStatus::Ready);
        assert_eq!(
            summaries[0].summary_preview.as_deref(),
            Some("Alice greets, then Bob responds.")
        );

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn updates_summary_fields() {
        let (store, root) = build_test_store();
        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");

        let updated = store
            .update_summary(
                created.id.clone(),
                UpdateDiarizationSummary {
                    status: DiarizationSummaryStatus::Pending,
                    model_id: Some("Qwen3.5-4B".to_string()),
                    text: None,
                    error: None,
                    updated_at: None,
                },
            )
            .await
            .expect("summary update should succeed")
            .expect("record should exist");

        assert_eq!(updated.summary_status, DiarizationSummaryStatus::Pending);
        assert!(updated.summary_text.is_none());
        assert!(updated.summary_error.is_none());
        assert!(updated.summary_updated_at.is_some());

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn rejects_unknown_speaker_overrides() {
        let (store, root) = build_test_store();
        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");

        let err = store
            .update_speaker_name_overrides(
                created.id,
                BTreeMap::from([("UNKNOWN".to_string(), "Alice".to_string())]),
            )
            .await
            .expect_err("unknown speaker should fail");

        assert!(err
            .to_string()
            .contains("Unknown diarization speaker label"));
        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn updates_processing_status_and_completes_pending_records() {
        let (store, root) = build_test_store();
        let mut pending = sample_record();
        pending.processing_status = DiarizationProcessingStatus::Pending;
        pending.processing_time_ms = 0.0;
        pending.duration_secs = None;
        pending.rtf = None;
        pending.speaker_count = 0;
        pending.alignment_coverage = None;
        pending.unattributed_words = 0;
        pending.llm_refined = false;
        pending.asr_text = String::new();
        pending.raw_transcript = String::new();
        pending.transcript = String::new();
        pending.summary_status = DiarizationSummaryStatus::NotRequested;
        pending.summary_model_id = None;
        pending.summary_text = None;
        pending.summary_error = None;
        pending.summary_updated_at = None;
        pending.segments = Vec::new();
        pending.words = Vec::new();
        pending.utterances = Vec::new();

        let created = store
            .create_record(pending)
            .await
            .expect("pending record should be created");
        assert_eq!(
            created.processing_status,
            DiarizationProcessingStatus::Pending
        );

        let processing = store
            .update_processing_status(
                created.id.clone(),
                DiarizationProcessingStatus::Processing,
                None,
            )
            .await
            .expect("processing update should succeed")
            .expect("record should exist");
        assert_eq!(
            processing.processing_status,
            DiarizationProcessingStatus::Processing
        );

        let completed = store
            .complete_record(created.id.clone(), sample_complete_record())
            .await
            .expect("completion should succeed")
            .expect("record should exist");

        assert_eq!(
            completed.processing_status,
            DiarizationProcessingStatus::Ready
        );
        assert!(completed.processing_error.is_none());
        assert_eq!(completed.speaker_count, 2);
        assert!(!completed.transcript.trim().is_empty());

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn completion_does_not_overwrite_terminal_records() {
        let (store, root) = build_test_store();
        let mut pending = sample_record();
        pending.processing_status = DiarizationProcessingStatus::Pending;
        pending.processing_time_ms = 0.0;
        pending.duration_secs = None;
        pending.rtf = None;
        pending.speaker_count = 0;
        pending.alignment_coverage = None;
        pending.unattributed_words = 0;
        pending.llm_refined = false;
        pending.asr_text = String::new();
        pending.raw_transcript = String::new();
        pending.transcript = String::new();
        pending.summary_status = DiarizationSummaryStatus::NotRequested;
        pending.summary_model_id = None;
        pending.summary_text = None;
        pending.summary_error = None;
        pending.summary_updated_at = None;
        pending.segments = Vec::new();
        pending.words = Vec::new();
        pending.utterances = Vec::new();

        let created = store
            .create_record(pending)
            .await
            .expect("pending record should be created");

        let cancelled = store
            .update_processing_status(
                created.id.clone(),
                DiarizationProcessingStatus::Failed,
                Some("Cancelled by user.".to_string()),
            )
            .await
            .expect("cancel status update should succeed")
            .expect("record should exist");
        assert_eq!(
            cancelled.processing_status,
            DiarizationProcessingStatus::Failed
        );

        let completion = store
            .complete_record(created.id.clone(), sample_complete_record())
            .await
            .expect("completion should execute");
        assert!(
            completion.is_none(),
            "terminal records should not be overwritten by async completion"
        );

        let current = store
            .get_record(created.id)
            .await
            .expect("record lookup should succeed")
            .expect("record should exist");
        assert_eq!(
            current.processing_status,
            DiarizationProcessingStatus::Failed
        );
        assert_eq!(
            current.processing_error.as_deref(),
            Some("Cancelled by user.")
        );

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }
}

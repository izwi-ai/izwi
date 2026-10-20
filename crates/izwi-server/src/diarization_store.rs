//! Persistent diarization history storage backed by SQLite.

use anyhow::{anyhow, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaStorageProvider};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, Condition, ConnectionTrait, EntityTrait, QueryFilter, QueryResult, Set,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::StoreError;
use crate::speech_text_list::SpeechTextListFilter;
use crate::{
    db::{raw, StoreDatabase},
    entity::diarization_records,
    ids::new_uuid,
    node_identity::{NodeIdentity, RecoveryScope},
    persistence::{
        delete_media_object, persist_audio_object, read_media_object, LocalMediaStorageProvider,
    },
    storage_layout,
};

/// The error of a record whose in-process job died with the server (P1V-1).
pub const INTERRUPTED_BY_SHUTDOWN_ERROR: &str =
    "Interrupted by a server shutdown before it finished; rerun the job.";

/// The error a pending summary gets when the server restarts under it: the
/// summary task lived in the old process, so nothing would finish it.
pub const SUMMARY_INTERRUPTED_ERROR: &str =
    "The summary was interrupted by a server restart. Regenerate it to try again.";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum DiarizationProcessingStatus {
    Pending,
    Processing,
    #[default]
    Ready,
    Failed,
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
#[derive(Default)]
pub enum DiarizationSummaryStatus {
    #[default]
    NotRequested,
    Pending,
    Ready,
    Failed,
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

#[derive(Debug, Clone)]
pub struct DiarizationRecordPage {
    pub records: Vec<DiarizationRecordSummary>,
    pub next_cursor: Option<DiarizationRecordListCursor>,
    /// Records matching the filter, ignoring paging.
    pub total: usize,
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
    /// Degraded pipeline steps: an aligner that could not be used, windows
    /// whose word timings are not the aligner's, a truncated ASR decode.
    pub warnings: Vec<String>,
    /// ASR language hint the job ran with; `None` detects it.
    pub language: Option<String>,
    /// Sortformer VAD gate the job asked for; `None` keeps the server default.
    pub vad_gate: Option<bool>,
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
    pub language: Option<String>,
    pub vad_gate: Option<bool>,
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
    pub warnings: Vec<String>,
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
    /// The node whose tasks run the jobs and summaries this store starts.
    node: NodeIdentity,
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
            node: NodeIdentity::local(),
        })
    }

    pub fn initialize_with_storage(
        db: StoreDatabase,
        media_storage: Arc<dyn MediaStorageProvider>,
    ) -> Self {
        Self {
            db,
            media_storage,
            node: NodeIdentity::local(),
        }
    }

    /// Records this store starts are owned by `node` (P1V-1).
    pub fn with_node_identity(mut self, node: NodeIdentity) -> Self {
        self.node = node;
        self
    }

    pub async fn list_records_page(
        &self,
        limit: usize,
        cursor: Option<DiarizationRecordListCursor>,
    ) -> anyhow::Result<(
        Vec<DiarizationRecordSummary>,
        Option<DiarizationRecordListCursor>,
    )> {
        let page = self
            .list_records_filtered(limit, cursor, &SpeechTextListFilter::default())
            .await?;
        Ok((page.records, page.next_cursor))
    }

    /// One page of records matching `filter`, newest first, with the total
    /// that match. Every diarization identifies speakers, so
    /// `with_speakers` never excludes one.
    pub async fn list_records_filtered(
        &self,
        limit: usize,
        cursor: Option<DiarizationRecordListCursor>,
        filter: &SpeechTextListFilter,
    ) -> anyhow::Result<DiarizationRecordPage> {
        let db = self.db.connection().await?;
        let page_limit = limit.clamp(1, 500);
        let filter_values: Vec<sea_orm::Value> =
            vec![filter.search_pattern().into(), filter.status_list().into()];
        let mut values = filter_values.clone();
        values.push(
            cursor
                .as_ref()
                .map(|cursor| i64::try_from(cursor.created_at).unwrap_or(i64::MAX))
                .into(),
        );
        values.push(cursor.map(|cursor| cursor.id).into());
        values.push(i64::try_from(page_limit + 1).unwrap_or(501).into());
        let rows = db
            .query_all_raw(raw::statement(db, DIARIZATION_PAGE_SQL, values)?)
            .await
            .context("Failed to list diarization records")?;
        let mut records = rows
            .iter()
            .map(map_diarization_summary)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let next_cursor = if records.len() > page_limit {
            records.truncate(page_limit);
            records.last().map(|record| DiarizationRecordListCursor {
                created_at: record.created_at,
                id: record.id.clone(),
            })
        } else {
            None
        };
        let total = db
            .query_one_raw(raw::statement(db, DIARIZATION_COUNT_SQL, filter_values)?)
            .await
            .context("Failed to count diarization records")?
            .map(|row| row.try_get_by_index::<i64>(0))
            .transpose()?
            .map(|count| usize::try_from(count).unwrap_or(0))
            .unwrap_or(0);
        Ok(DiarizationRecordPage {
            records,
            next_cursor,
            total,
        })
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
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT audio_storage_path, audio_mime_type, audio_filename
                FROM diarization_records
                WHERE id = ?1
                "#,
                vec![record_id.into()],
            )?)
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
        let language = sanitize_optional_text(record.language.as_deref(), 64);
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
        let asr_text = sanitize_transcript_text(record.asr_text.as_str());
        let raw_transcript = sanitize_transcript_text(record.raw_transcript.as_str());
        let transcript = sanitize_transcript_text(record.transcript.as_str());
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
            return Err(StoreError::invalid("Audio payload cannot be empty").into());
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
            warnings_json: Set("[]".to_string()),
            language: Set(language),
            vad_gate: Set(record.vad_gate.map(i64::from)),
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
            owner_node_id: Set(Some(self.node.id().to_string())),
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

        let mut update_query = diarization_records::Entity::update_many();
        if summary_status == DiarizationSummaryStatus::Pending {
            // This node's task generates the summary. A record still being
            // diarized keeps the owner that runs its pipeline.
            update_query = update_query.col_expr(
                diarization_records::Column::OwnerNodeId,
                Expr::case(
                    diarization_records::Column::ProcessingStatus
                        .is_in(["pending".to_string(), "processing".to_string()]),
                    Expr::col(diarization_records::Column::OwnerNodeId),
                )
                .finally(Expr::value(self.node.id()))
                .into(),
            );
        }
        let result = update_query
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

    /// Fail the jobs a previous run of this node left pending or processing
    /// (P1V-1). Diarization jobs and their summaries run as tasks of the
    /// node that accepted them, so at startup none of its own can still be
    /// running; left alone they would stay "processing" forever. Records
    /// other nodes own on a shared database are theirs to recover. A failed
    /// record keeps its audio and can be rerun. Returns the number of
    /// records failed, or `None` when this node cannot tell its records from
    /// another node's (see [`NodeIdentity::recovery_scope`]).
    pub async fn fail_interrupted_records(&self) -> anyhow::Result<Option<u64>> {
        let db = self.db.connection().await?;
        let Some(scope) = self.node.recovery_scope(db.get_database_backend()) else {
            return Ok(None);
        };
        self.fail_interrupted_records_in(&scope).await.map(Some)
    }

    async fn fail_interrupted_records_in(&self, scope: &RecoveryScope) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let owned = || {
            let owner = diarization_records::Column::OwnerNodeId;
            let mut owned = Condition::any().add(owner.eq(scope.owner.as_str()));
            if scope.include_unowned {
                owned = owned.add(owner.is_null());
            }
            owned
        };
        let active = ["pending".to_string(), "processing".to_string()];
        let jobs = diarization_records::Entity::update_many()
            .col_expr(
                diarization_records::Column::ProcessingStatus,
                Expr::value(DiarizationProcessingStatus::Failed.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::ProcessingError,
                Expr::value(INTERRUPTED_BY_SHUTDOWN_ERROR),
            )
            .filter(diarization_records::Column::ProcessingStatus.is_in(active))
            .filter(owned())
            .exec(db)
            .await
            .context("Failed to fail interrupted diarization records")?;
        diarization_records::Entity::update_many()
            .col_expr(
                diarization_records::Column::SummaryStatus,
                Expr::value(DiarizationSummaryStatus::Failed.as_db_value()),
            )
            .col_expr(
                diarization_records::Column::SummaryError,
                Expr::value(SUMMARY_INTERRUPTED_ERROR),
            )
            .filter(diarization_records::Column::SummaryStatus.eq("pending"))
            .filter(owned())
            .exec(db)
            .await
            .context("Failed to fail interrupted diarization summaries")?;
        Ok(jobs.rows_affected)
    }

    /// Move a pending or processing record to `status`. Returns `None` when
    /// the record is missing or already terminal (ready or failed).
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
            // A terminal record keeps its outcome: a pipeline error arriving
            // after "Cancelled by user." must not overwrite it (PIPE-7).
            .filter(
                diarization_records::Column::ProcessingStatus
                    .is_in(["pending".to_string(), "processing".to_string()]),
            )
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
        let asr_text = sanitize_transcript_text(record.asr_text.as_str());
        let raw_transcript = sanitize_transcript_text(record.raw_transcript.as_str());
        let transcript = sanitize_transcript_text(record.transcript.as_str());
        let segments_json =
            serde_json::to_string(&record.segments).context("Failed serializing segments")?;
        let words_json =
            serde_json::to_string(&record.words).context("Failed serializing words")?;
        let utterances_json =
            serde_json::to_string(&record.utterances).context("Failed serializing utterances")?;
        let warnings_json = serde_json::to_string(&sanitize_warnings(record.warnings))
            .context("Failed serializing diarization warnings")?;
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
                diarization_records::Column::WarningsJson,
                Expr::value(warnings_json),
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

/// ?1 search pattern, ?2 ",status,…," list (each NULL = no filter).
macro_rules! diarization_filter {
    () => {
        r#"
    WHERE (?1 IS NULL OR transcript LIKE ?1 ESCAPE '\' OR audio_filename LIKE ?1 ESCAPE '\')
      AND (?2 IS NULL OR instr(?2, ',' || processing_status || ',') > 0)
        "#
    };
}

/// Filter ?1-?2, cursor ?3/?4 (or NULL), limit ?5.
const DIARIZATION_PAGE_SQL: &str = concat!(
    r#"
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
    FROM diarization_records"#,
    diarization_filter!(),
    r#"
      AND (?3 IS NULL OR created_at < ?3 OR (created_at = ?3 AND id < ?4))
    ORDER BY created_at DESC, id DESC
    LIMIT ?5
"#
);

const DIARIZATION_COUNT_SQL: &str = concat!(
    "SELECT COUNT(1) FROM diarization_records",
    diarization_filter!()
);

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
    audio_filename,
    warnings_json,
    language,
    vad_gate
"#;

async fn fetch_record_without_audio(
    db: &sea_orm::DatabaseConnection,
    record_id: &str,
) -> anyhow::Result<Option<DiarizationRecord>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            format!("SELECT {DIARIZATION_RECORD_COLUMNS} FROM diarization_records WHERE id = ?1"),
            vec![record_id.into()],
        )?)
        .await
        .context("Failed to load diarization record")?;
    row.as_ref().map(map_diarization_record).transpose()
}

async fn fetch_audio_storage_path(
    db: &sea_orm::DatabaseConnection,
    record_id: &str,
) -> anyhow::Result<Option<Option<String>>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            "SELECT audio_storage_path FROM diarization_records WHERE id = ?1",
            vec![record_id.into()],
        )?)
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
        warnings: parse_json_vec(row.try_get_by_index(34)?),
        language: row.try_get_by_index(35)?,
        vad_gate: row
            .try_get_by_index::<Option<i64>>(36)?
            .map(|value| value != 0),
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

/// Transcript text is stored whole (as the words JSON is): the record is the
/// only copy of a job's transcript, however long the audio.
fn sanitize_transcript_text(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        " ".to_string()
    } else {
        trimmed.to_string()
    }
}

fn sanitize_warnings(warnings: Vec<String>) -> Vec<String> {
    warnings
        .into_iter()
        .filter_map(|warning| sanitize_optional_text(Some(warning.as_str()), 500))
        .collect()
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
            return Err(
                StoreError::invalid(format!("Unknown diarization speaker label: {raw}")).into(),
            );
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

#[cfg(test)]
pub(crate) mod tests {
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

    pub(crate) fn sample_record() -> NewDiarizationRecord {
        NewDiarizationRecord {
            model_id: Some("diar_streaming_sortformer_4spk-v2.1".to_string()),
            asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            llm_model_id: Some("Qwen3.5-4B".to_string()),
            processing_status: DiarizationProcessingStatus::Ready,
            processing_error: None,
            min_speakers: Some(1),
            max_speakers: Some(4),
            min_speech_duration_ms: Some(240.0),
            min_silence_duration_ms: Some(200.0),
            enable_llm_refinement: true,
            language: None,
            vad_gate: None,
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
            warnings: Vec::new(),
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
    async fn stores_uploaded_source_audio_metadata_and_bytes() {
        let (store, root) = build_test_store();
        let mut record = sample_record();
        record.audio_mime_type = "audio/mpeg".to_string();
        record.audio_filename = Some("board-review.mp3".to_string());
        record.audio_bytes = vec![0x49, 0x44, 0x33, 0x04];

        let created = store
            .create_record(record)
            .await
            .expect("record should be created");

        assert_eq!(created.audio_mime_type, "audio/mpeg");
        assert_eq!(created.audio_filename.as_deref(), Some("board-review.mp3"));

        let audio = store
            .get_audio(created.id.clone())
            .await
            .expect("audio should load")
            .expect("audio should exist");

        assert_eq!(audio.audio_mime_type, "audio/mpeg");
        assert_eq!(audio.audio_filename.as_deref(), Some("board-review.mp3"));
        assert_eq!(audio.audio_bytes, vec![0x49, 0x44, 0x33, 0x04]);

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

    /// PIPE-9: an hour of speech is well past 40k characters; the stored
    /// transcript texts must not be cut, and the pipeline warnings persist.
    #[tokio::test]
    async fn long_transcripts_and_warnings_round_trip_unchanged() {
        let (store, root) = build_test_store();
        let long_text = "word ".repeat(10_000).trim_end().to_string();
        assert!(long_text.chars().count() > 40_000);
        let long_transcript = format!("SPEAKER_00 [0.00s - 3600.00s]: {}", "word ".repeat(30_000))
            .trim_end()
            .to_string();
        assert!(long_transcript.chars().count() > 100_000);

        let mut pending = sample_record();
        pending.processing_status = DiarizationProcessingStatus::Pending;
        pending.asr_text = long_text.clone();
        let created = store
            .create_record(pending)
            .await
            .expect("record should be created");
        assert_eq!(created.asr_text, long_text);
        assert!(created.warnings.is_empty());

        let warnings = vec![
            "Forced aligner unavailable (needs 2.77 GB); word timings are estimated, not aligned"
                .to_string(),
        ];
        let mut complete = sample_complete_record();
        complete.asr_text = long_text.clone();
        complete.raw_transcript = long_transcript.clone();
        complete.transcript = long_transcript.clone();
        complete.alignment_coverage = Some(0.0);
        complete.warnings = warnings.clone();
        store
            .complete_record(created.id.clone(), complete)
            .await
            .expect("completion should succeed")
            .expect("record should exist");

        let stored = store
            .get_record(created.id)
            .await
            .expect("record lookup should succeed")
            .expect("record should exist");
        assert_eq!(stored.asr_text, long_text);
        assert_eq!(stored.raw_transcript, long_transcript);
        assert_eq!(stored.transcript, long_transcript);
        assert_eq!(stored.warnings, warnings);

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    /// The language hint and VAD gate a job ran with persist, so a rerun can
    /// repeat them.
    #[tokio::test]
    async fn language_and_vad_gate_round_trip() {
        let (store, root) = build_test_store();
        for (language, vad_gate) in [
            (Some("German".to_string()), Some(true)),
            (None, Some(false)),
            (None, None),
        ] {
            let mut record = sample_record();
            record.language = language.clone();
            record.vad_gate = vad_gate;
            let created = store.create_record(record).await.expect("record");
            assert_eq!(
                (created.language.clone(), created.vad_gate),
                (language.clone(), vad_gate)
            );
            let stored = store
                .get_record(created.id)
                .await
                .expect("lookup")
                .expect("record");
            assert_eq!((stored.language, stored.vad_gate), (language, vad_gate));
        }
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

    /// P1V-1: a job left pending or processing by a previous process fails
    /// at startup with a clear error (and a pending summary with it), while
    /// finished records are untouched.
    #[tokio::test]
    async fn startup_fails_records_left_running_by_a_previous_process() {
        let (store, root) = build_test_store();
        let mut ids = Vec::new();
        for status in [
            DiarizationProcessingStatus::Pending,
            DiarizationProcessingStatus::Processing,
            DiarizationProcessingStatus::Ready,
        ] {
            let mut record = sample_record();
            record.processing_status = status;
            record.summary_status = DiarizationSummaryStatus::NotRequested;
            if status == DiarizationProcessingStatus::Ready {
                record.summary_status = DiarizationSummaryStatus::Pending;
                record.summary_text = None;
            }
            let created = store.create_record(record).await.expect("record");
            ids.push(created.id);
        }

        assert_eq!(
            store
                .fail_interrupted_records()
                .await
                .expect("startup recovery"),
            Some(2)
        );

        for id in &ids[..2] {
            let record = store
                .get_record(id.clone())
                .await
                .expect("record lookup")
                .expect("record");
            assert_eq!(
                record.processing_status,
                DiarizationProcessingStatus::Failed
            );
            assert_eq!(
                record.processing_error.as_deref(),
                Some(INTERRUPTED_BY_SHUTDOWN_ERROR)
            );
        }
        let ready = store
            .get_record(ids[2].clone())
            .await
            .expect("record lookup")
            .expect("record");
        assert_eq!(ready.processing_status, DiarizationProcessingStatus::Ready);
        assert!(ready.processing_error.is_none());
        assert_eq!(ready.summary_status, DiarizationSummaryStatus::Failed);
        assert_eq!(
            ready.summary_error.as_deref(),
            Some(SUMMARY_INTERRUPTED_ERROR)
        );
        assert_eq!(
            store.fail_interrupted_records().await.expect("rerun"),
            Some(0)
        );

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    /// P1V-1 on a database several nodes share: node A's startup fails the
    /// jobs and summaries A left running and leaves node B's running. Rows
    /// written before owners were recorded are recovered only where the
    /// database belongs to one node.
    #[tokio::test]
    async fn startup_recovery_leaves_other_nodes_jobs_running() {
        let (store, root) = build_test_store();
        let node = |id| {
            store
                .clone()
                .with_node_identity(NodeIdentity::configured(id).expect("node id"))
        };
        let (node_a, node_b) = (node("node-a"), node("node-b"));
        let processing = || {
            let mut record = sample_record();
            record.processing_status = DiarizationProcessingStatus::Processing;
            record.summary_status = DiarizationSummaryStatus::NotRequested;
            record.summary_text = None;
            record
        };
        let a_job = node_a.create_record(processing()).await.expect("A job");
        let b_job = node_b.create_record(processing()).await.expect("B job");
        let legacy = node_a.create_record(processing()).await.expect("legacy");
        let db = store.db.connection().await.expect("db");
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE diarization_records SET owner_node_id = NULL WHERE id = ?1",
                vec![legacy.id.clone().into()],
            )
            .expect("statement"),
        )
        .await
        .expect("unown legacy row");
        // B regenerates the summary of a finished record of A's: B runs it.
        let mut finished = sample_record();
        finished.summary_status = DiarizationSummaryStatus::NotRequested;
        finished.summary_text = None;
        let b_summary = node_a.create_record(finished).await.expect("finished");
        // B asks for a summary of A's running job: A still runs the job.
        for id in [&b_summary.id, &a_job.id] {
            node_b
                .update_summary(
                    id.clone(),
                    UpdateDiarizationSummary {
                        status: DiarizationSummaryStatus::Pending,
                        model_id: Some("Qwen3.5-4B".to_string()),
                        text: None,
                        error: None,
                        updated_at: None,
                    },
                )
                .await
                .expect("summary update")
                .expect("record");
        }
        let status = |id: &String| {
            let store = store.clone();
            let id = id.clone();
            async move {
                let record = store.get_record(id).await.expect("lookup").expect("record");
                (record.processing_status, record.summary_status)
            }
        };
        use DiarizationProcessingStatus::{Failed, Processing, Ready};
        use DiarizationSummaryStatus::{Failed as SummaryFailed, NotRequested, Pending};

        // A shared database: A recovers only what it owns.
        let shared = RecoveryScope {
            owner: "node-a".to_string(),
            include_unowned: false,
        };
        assert_eq!(
            node_a
                .fail_interrupted_records_in(&shared)
                .await
                .expect("A recovery"),
            1
        );
        assert_eq!(status(&a_job.id).await, (Failed, SummaryFailed));
        assert_eq!(status(&b_job.id).await, (Processing, NotRequested));
        assert_eq!(status(&b_summary.id).await, (Ready, Pending));
        assert_eq!(status(&legacy.id).await, (Processing, NotRequested));

        // SQLite belongs to one node: A also recovers the unowned row.
        assert_eq!(
            node_a.fail_interrupted_records().await.expect("A recovery"),
            Some(1)
        );
        assert_eq!(status(&legacy.id).await, (Failed, NotRequested));
        assert_eq!(status(&b_job.id).await, (Processing, NotRequested));
        assert_eq!(status(&b_summary.id).await, (Ready, Pending));

        // B's own restart recovers B's job and the summary it was running.
        assert_eq!(
            node_b.fail_interrupted_records().await.expect("B recovery"),
            Some(1)
        );
        assert_eq!(status(&b_job.id).await, (Failed, NotRequested));
        assert_eq!(status(&b_summary.id).await, (Ready, SummaryFailed));

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    /// PIPE-7: the cancelled pipeline's own error, or a late status write,
    /// arrives after the cancel and must not replace "Cancelled by user.".
    #[tokio::test]
    async fn status_updates_do_not_overwrite_terminal_records() {
        let (store, root) = build_test_store();
        let mut pending = sample_record();
        pending.processing_status = DiarizationProcessingStatus::Pending;
        let created = store
            .create_record(pending)
            .await
            .expect("pending record should be created");
        store
            .update_processing_status(
                created.id.clone(),
                DiarizationProcessingStatus::Failed,
                Some("Cancelled by user.".to_string()),
            )
            .await
            .expect("cancel status update should succeed")
            .expect("record should exist");

        for (status, error) in [
            (
                DiarizationProcessingStatus::Failed,
                Some("Cancelled: job \"dir\" was cancelled".to_string()),
            ),
            (DiarizationProcessingStatus::Processing, None),
        ] {
            let late = store
                .update_processing_status(created.id.clone(), status, error)
                .await
                .expect("late status update should execute");
            assert!(late.is_none(), "{status:?} overwrote a terminal record");
        }

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

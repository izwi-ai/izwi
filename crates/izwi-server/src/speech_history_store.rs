//! Persistent speech generation history storage backed by SQLite.

use anyhow::{anyhow, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaStorageProvider};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DbBackend, EntityTrait, QueryFilter, QueryResult, Set, Statement,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    db::StoreDatabase,
    entity::speech_history_records,
    ids::new_uuid,
    persistence::{
        delete_media_object, persist_audio_object, read_media_object, LocalMediaStorageProvider,
    },
    storage_layout,
};

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechRouteKind {
    TextToSpeech,
    VoiceDesign,
    VoiceCloning,
}

impl SpeechRouteKind {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::TextToSpeech => "text_to_speech",
            Self::VoiceDesign => "voice_design",
            Self::VoiceCloning => "voice_cloning",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "text_to_speech" => Some(Self::TextToSpeech),
            "voice_design" => Some(Self::VoiceDesign),
            "voice_cloning" => Some(Self::VoiceCloning),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeechHistoryProcessingStatus {
    Pending,
    Processing,
    Ready,
    Failed,
}

impl Default for SpeechHistoryProcessingStatus {
    fn default() -> Self {
        Self::Ready
    }
}

impl SpeechHistoryProcessingStatus {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "pending" => Some(Self::Pending),
            "processing" => Some(Self::Processing),
            "ready" => Some(Self::Ready),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechHistoryRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub route_kind: SpeechRouteKind,
    pub processing_status: SpeechHistoryProcessingStatus,
    pub processing_error: Option<String>,
    pub model_id: Option<String>,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub input_preview: String,
    pub input_chars: usize,
    pub generation_time_ms: f64,
    pub audio_duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub tokens_generated: Option<usize>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechHistoryRecordListCursor {
    pub created_at: u64,
    pub id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechHistoryRecord {
    pub id: String,
    pub created_at: u64,
    pub route_kind: SpeechRouteKind,
    pub processing_status: SpeechHistoryProcessingStatus,
    pub processing_error: Option<String>,
    pub model_id: Option<String>,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub input_text: String,
    pub voice_description: Option<String>,
    pub reference_text: Option<String>,
    pub generation_time_ms: f64,
    pub audio_duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub tokens_generated: Option<usize>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StoredSpeechAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewSpeechHistoryRecord {
    pub route_kind: SpeechRouteKind,
    pub processing_status: SpeechHistoryProcessingStatus,
    pub processing_error: Option<String>,
    pub model_id: Option<String>,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub input_text: String,
    pub voice_description: Option<String>,
    pub reference_text: Option<String>,
    pub generation_time_ms: f64,
    pub audio_duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub tokens_generated: Option<usize>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CompleteSpeechHistoryRecord {
    pub model_id: Option<String>,
    pub speaker: Option<String>,
    pub language: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub input_text: String,
    pub voice_description: Option<String>,
    pub reference_text: Option<String>,
    pub generation_time_ms: f64,
    pub audio_duration_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub tokens_generated: Option<usize>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct SpeechHistoryStore {
    db: StoreDatabase,
    media_storage: Arc<dyn MediaStorageProvider>,
}

impl SpeechHistoryStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare speech history storage layout")?;

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
        route_kind: SpeechRouteKind,
        limit: usize,
        cursor: Option<SpeechHistoryRecordListCursor>,
    ) -> anyhow::Result<(
        Vec<SpeechHistoryRecordSummary>,
        Option<SpeechHistoryRecordListCursor>,
    )> {
        let db = self.db.connection().await?;
        let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);
        let fetch_limit = list_limit.saturating_add(1);

        let rows = if let Some(cursor) = cursor {
            let cursor_created_at = i64::try_from(cursor.created_at).unwrap_or(i64::MAX);
            db.query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                SPEECH_HISTORY_PAGE_AFTER_CURSOR_SQL,
                vec![
                    route_kind.as_db_value().into(),
                    cursor_created_at.into(),
                    cursor.id.into(),
                    fetch_limit.into(),
                ],
            ))
            .await
        } else {
            db.query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                SPEECH_HISTORY_PAGE_SQL,
                vec![route_kind.as_db_value().into(), fetch_limit.into()],
            ))
            .await
        }
        .context("Failed to list speech history records")?;

        let mut records = rows
            .iter()
            .map(map_speech_history_summary)
            .collect::<anyhow::Result<Vec<_>>>()?;

        let page_limit = usize::try_from(list_limit).unwrap_or(200);
        let has_more = records.len() > page_limit;
        if has_more {
            records.truncate(page_limit);
        }
        let next_cursor = if has_more {
            records.last().map(|record| SpeechHistoryRecordListCursor {
                created_at: record.created_at,
                id: record.id.clone(),
            })
        } else {
            None
        };

        Ok((records, next_cursor))
    }

    pub async fn get_record(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<Option<SpeechHistoryRecord>> {
        let db = self.db.connection().await?;
        fetch_record_without_audio(db, route_kind, &record_id).await
    }

    pub async fn get_audio(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<Option<StoredSpeechAudio>> {
        let db = self.db.connection().await?;
        let audio = db
            .query_one_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                r#"
                SELECT audio_storage_path, audio_mime_type, audio_filename
                FROM speech_history_records
                WHERE route_kind = ?1 AND id = ?2
                "#,
                vec![route_kind.as_db_value().into(), record_id.into()],
            ))
            .await
            .context("Failed to load speech history audio metadata")?;
        let Some(row) = audio else {
            return Ok(None);
        };

        let audio_storage_path: Option<String> = row.try_get_by_index(0)?;
        let audio_mime_type: String = row.try_get_by_index(1)?;
        let audio_filename: Option<String> = row.try_get_by_index(2)?;
        let Some(audio_storage_path) = sanitize_media_path(audio_storage_path.as_deref()) else {
            return Ok(None);
        };

        let audio_bytes = read_media_object(&self.media_storage, audio_storage_path.as_str())
            .await
            .context("Failed to read speech history media")?;

        Ok(Some(StoredSpeechAudio {
            audio_bytes,
            audio_mime_type,
            audio_filename,
        }))
    }

    pub async fn create_record(
        &self,
        record: NewSpeechHistoryRecord,
    ) -> anyhow::Result<SpeechHistoryRecord> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let record_id = new_uuid();
        let route_kind = record.route_kind;
        let processing_status = record.processing_status;
        let processing_error = sanitize_optional_text(record.processing_error.as_deref(), 1_200);
        let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
        let speaker = sanitize_optional_text(record.speaker.as_deref(), 120);
        let language = sanitize_optional_text(record.language.as_deref(), 80);
        let saved_voice_id = sanitize_optional_text(record.saved_voice_id.as_deref(), 160);
        let speed = record
            .speed
            .filter(|value| value.is_finite() && *value > 0.0);
        let input_text = sanitize_required_text(record.input_text.as_str(), 20_000);
        let voice_description = sanitize_optional_text(record.voice_description.as_deref(), 2_000);
        let reference_text = sanitize_optional_text(record.reference_text.as_deref(), 2_000);
        let generation_time_ms = if record.generation_time_ms.is_finite() {
            record.generation_time_ms.max(0.0)
        } else {
            0.0
        };
        let audio_duration_secs = record
            .audio_duration_secs
            .filter(|value| value.is_finite() && *value >= 0.0);
        let rtf = record
            .rtf
            .filter(|value| value.is_finite() && *value >= 0.0);
        let tokens_generated = record
            .tokens_generated
            .filter(|value| *value > 0)
            .and_then(|value| i64::try_from(value).ok());
        let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
        let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);

        let has_audio_payload = !record.audio_bytes.is_empty();
        if !has_audio_payload && processing_status == SpeechHistoryProcessingStatus::Ready {
            return Err(anyhow!(
                "Audio payload cannot be empty for ready speech history records",
            ));
        }

        let audio_storage_path = if has_audio_payload {
            let mut metadata = HookMetadata::new();
            metadata.insert(
                "route_kind".to_string(),
                route_kind.as_db_value().to_string(),
            );
            Some(
                persist_audio_object(
                    &self.media_storage,
                    MediaNamespace::GeneratedSpeech,
                    &record_id,
                    audio_filename.as_deref(),
                    audio_mime_type.as_str(),
                    &record.audio_bytes,
                    metadata,
                )
                .await?,
            )
        } else {
            None
        };

        if let Err(err) =
            speech_history_records::Entity::insert(speech_history_records::ActiveModel {
                id: Set(record_id.clone()),
                created_at: Set(now),
                route_kind: Set(route_kind.as_db_value().to_string()),
                processing_status: Set(processing_status.as_db_value().to_string()),
                processing_error: Set(processing_error),
                model_id: Set(model_id),
                speaker: Set(speaker),
                language: Set(language),
                saved_voice_id: Set(saved_voice_id),
                speed: Set(speed),
                input_text: Set(input_text),
                voice_description: Set(voice_description),
                reference_text: Set(reference_text),
                generation_time_ms: Set(generation_time_ms),
                audio_duration_secs: Set(audio_duration_secs),
                rtf: Set(rtf),
                tokens_generated: Set(tokens_generated),
                audio_mime_type: Set(audio_mime_type),
                audio_filename: Set(audio_filename),
                audio_storage_path: Set(audio_storage_path.clone().unwrap_or_default()),
            })
            .exec(db)
            .await
        {
            if let Some(path) = audio_storage_path.as_deref() {
                let _ = delete_media_object(&self.media_storage, Some(path)).await;
            }
            return Err(err).context("Failed to insert speech history record");
        }

        fetch_record_without_audio(db, route_kind, &record_id)
            .await?
            .ok_or_else(|| anyhow!("Failed to fetch created speech history record"))
    }

    pub async fn update_processing_status(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
        status: SpeechHistoryProcessingStatus,
        processing_error: Option<String>,
    ) -> anyhow::Result<Option<SpeechHistoryRecord>> {
        let db = self.db.connection().await?;
        let sanitized_error = sanitize_optional_text(processing_error.as_deref(), 1_200);
        speech_history_records::Entity::update_many()
            .col_expr(
                speech_history_records::Column::ProcessingStatus,
                Expr::value(status.as_db_value()),
            )
            .col_expr(
                speech_history_records::Column::ProcessingError,
                Expr::value(sanitized_error),
            )
            .filter(speech_history_records::Column::RouteKind.eq(route_kind.as_db_value()))
            .filter(speech_history_records::Column::Id.eq(record_id.clone()))
            .exec(db)
            .await
            .context("Failed to update speech history processing status")?;

        fetch_record_without_audio(db, route_kind, &record_id).await
    }

    pub async fn complete_record(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
        record: CompleteSpeechHistoryRecord,
    ) -> anyhow::Result<Option<SpeechHistoryRecord>> {
        let db = self.db.connection().await?;
        let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
        let speaker = sanitize_optional_text(record.speaker.as_deref(), 120);
        let language = sanitize_optional_text(record.language.as_deref(), 80);
        let saved_voice_id = sanitize_optional_text(record.saved_voice_id.as_deref(), 160);
        let speed = record
            .speed
            .filter(|value| value.is_finite() && *value > 0.0);
        let input_text = sanitize_required_text(record.input_text.as_str(), 20_000);
        let voice_description = sanitize_optional_text(record.voice_description.as_deref(), 2_000);
        let reference_text = sanitize_optional_text(record.reference_text.as_deref(), 2_000);
        let generation_time_ms = if record.generation_time_ms.is_finite() {
            record.generation_time_ms.max(0.0)
        } else {
            0.0
        };
        let audio_duration_secs = record
            .audio_duration_secs
            .filter(|value| value.is_finite() && *value >= 0.0);
        let rtf = record
            .rtf
            .filter(|value| value.is_finite() && *value >= 0.0);
        let tokens_generated = record
            .tokens_generated
            .filter(|value| *value > 0)
            .and_then(|value| i64::try_from(value).ok());
        let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
        let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);

        if record.audio_bytes.is_empty() {
            return Err(anyhow!(
                "Audio payload cannot be empty when completing speech history records",
            ));
        }

        let mut metadata = HookMetadata::new();
        metadata.insert(
            "route_kind".to_string(),
            route_kind.as_db_value().to_string(),
        );
        let next_audio_storage_path = persist_audio_object(
            &self.media_storage,
            MediaNamespace::GeneratedSpeech,
            &record_id,
            audio_filename.as_deref(),
            audio_mime_type.as_str(),
            &record.audio_bytes,
            metadata,
        )
        .await?;

        let previous_audio_storage_path = fetch_audio_storage_path(db, route_kind, &record_id)
            .await?
            .flatten();

        let result = speech_history_records::Entity::update_many()
            .col_expr(
                speech_history_records::Column::ProcessingStatus,
                Expr::value(SpeechHistoryProcessingStatus::Ready.as_db_value()),
            )
            .col_expr(
                speech_history_records::Column::ProcessingError,
                Expr::value(Option::<String>::None),
            )
            .col_expr(
                speech_history_records::Column::ModelId,
                Expr::value(model_id),
            )
            .col_expr(
                speech_history_records::Column::Speaker,
                Expr::value(speaker),
            )
            .col_expr(
                speech_history_records::Column::Language,
                Expr::value(language),
            )
            .col_expr(
                speech_history_records::Column::SavedVoiceId,
                Expr::value(saved_voice_id),
            )
            .col_expr(speech_history_records::Column::Speed, Expr::value(speed))
            .col_expr(
                speech_history_records::Column::InputText,
                Expr::value(input_text),
            )
            .col_expr(
                speech_history_records::Column::VoiceDescription,
                Expr::value(voice_description),
            )
            .col_expr(
                speech_history_records::Column::ReferenceText,
                Expr::value(reference_text),
            )
            .col_expr(
                speech_history_records::Column::GenerationTimeMs,
                Expr::value(generation_time_ms),
            )
            .col_expr(
                speech_history_records::Column::AudioDurationSecs,
                Expr::value(audio_duration_secs),
            )
            .col_expr(speech_history_records::Column::Rtf, Expr::value(rtf))
            .col_expr(
                speech_history_records::Column::TokensGenerated,
                Expr::value(tokens_generated),
            )
            .col_expr(
                speech_history_records::Column::AudioMimeType,
                Expr::value(audio_mime_type),
            )
            .col_expr(
                speech_history_records::Column::AudioFilename,
                Expr::value(audio_filename),
            )
            .col_expr(
                speech_history_records::Column::AudioStoragePath,
                Expr::value(next_audio_storage_path.clone()),
            )
            .filter(speech_history_records::Column::RouteKind.eq(route_kind.as_db_value()))
            .filter(speech_history_records::Column::Id.eq(record_id.clone()))
            .exec(db)
            .await
            .context("Failed to complete speech history record")?;

        if result.rows_affected == 0 {
            let _ =
                delete_media_object(&self.media_storage, Some(next_audio_storage_path.as_str()))
                    .await;
            return Ok(None);
        }

        if let Some(previous_path) = sanitize_media_path(previous_audio_storage_path.as_deref()) {
            let _ = delete_media_object(&self.media_storage, Some(previous_path.as_str())).await;
        }

        fetch_record_without_audio(db, route_kind, &record_id).await
    }

    pub async fn delete_record(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let audio_storage_path = fetch_audio_storage_path(db, route_kind, &record_id)
            .await?
            .flatten();
        let result = speech_history_records::Entity::delete_many()
            .filter(speech_history_records::Column::RouteKind.eq(route_kind.as_db_value()))
            .filter(speech_history_records::Column::Id.eq(record_id))
            .exec(db)
            .await
            .context("Failed to delete speech history record")?;

        if result.rows_affected > 0 {
            let normalized_audio_path = sanitize_media_path(audio_storage_path.as_deref());
            delete_media_object(&self.media_storage, normalized_audio_path.as_deref()).await?;
        }

        Ok(result.rows_affected > 0)
    }
}

const SPEECH_HISTORY_PAGE_SQL: &str = r#"
    SELECT
        id,
        created_at,
        route_kind,
        processing_status,
        processing_error,
        model_id,
        speaker,
        language,
        saved_voice_id,
        speed,
        input_text,
        generation_time_ms,
        audio_duration_secs,
        rtf,
        tokens_generated,
        audio_mime_type,
        audio_filename
    FROM speech_history_records
    WHERE route_kind = ?1
    ORDER BY created_at DESC, id DESC
    LIMIT ?2
"#;

const SPEECH_HISTORY_PAGE_AFTER_CURSOR_SQL: &str = r#"
    SELECT
        id,
        created_at,
        route_kind,
        processing_status,
        processing_error,
        model_id,
        speaker,
        language,
        saved_voice_id,
        speed,
        input_text,
        generation_time_ms,
        audio_duration_secs,
        rtf,
        tokens_generated,
        audio_mime_type,
        audio_filename
    FROM speech_history_records
    WHERE route_kind = ?1
        AND (created_at < ?2 OR (created_at = ?2 AND id < ?3))
    ORDER BY created_at DESC, id DESC
    LIMIT ?4
"#;

const SPEECH_HISTORY_RECORD_COLUMNS: &str = r#"
    id,
    created_at,
    route_kind,
    processing_status,
    processing_error,
    model_id,
    speaker,
    language,
    saved_voice_id,
    speed,
    input_text,
    voice_description,
    reference_text,
    generation_time_ms,
    audio_duration_secs,
    rtf,
    tokens_generated,
    audio_mime_type,
    audio_filename
"#;

async fn fetch_record_without_audio(
    db: &sea_orm::DatabaseConnection,
    route_kind: SpeechRouteKind,
    record_id: &str,
) -> anyhow::Result<Option<SpeechHistoryRecord>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            format!(
                "SELECT {SPEECH_HISTORY_RECORD_COLUMNS} FROM speech_history_records WHERE route_kind = ?1 AND id = ?2"
            ),
            vec![route_kind.as_db_value().into(), record_id.into()],
        ))
        .await
        .context("Failed to load speech history record")?;
    row.as_ref().map(map_speech_history_record).transpose()
}

async fn fetch_audio_storage_path(
    db: &sea_orm::DatabaseConnection,
    route_kind: SpeechRouteKind,
    record_id: &str,
) -> anyhow::Result<Option<Option<String>>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            "SELECT audio_storage_path FROM speech_history_records WHERE route_kind = ?1 AND id = ?2",
            vec![route_kind.as_db_value().into(), record_id.into()],
        ))
        .await
        .context("Failed to load speech history media path")?;
    row.map(|row| row.try_get_by_index::<Option<String>>(0))
        .transpose()
        .map_err(Into::into)
}

fn map_speech_history_summary(row: &QueryResult) -> anyhow::Result<SpeechHistoryRecordSummary> {
    let input_text: String = row.try_get_by_index(10)?;
    let route_raw: String = row.try_get_by_index(2)?;
    let route_kind =
        SpeechRouteKind::from_db_value(route_raw.as_str()).unwrap_or(SpeechRouteKind::TextToSpeech);
    let processing_status_raw: String = row.try_get_by_index(3)?;

    Ok(SpeechHistoryRecordSummary {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        route_kind,
        processing_status: SpeechHistoryProcessingStatus::from_db_value(
            processing_status_raw.as_str(),
        )
        .unwrap_or_default(),
        processing_error: row.try_get_by_index(4)?,
        model_id: row.try_get_by_index(5)?,
        speaker: row.try_get_by_index(6)?,
        language: row.try_get_by_index(7)?,
        saved_voice_id: row.try_get_by_index(8)?,
        speed: row.try_get_by_index(9)?,
        input_preview: input_preview(input_text.as_str()),
        input_chars: input_text.chars().count(),
        generation_time_ms: row.try_get_by_index(11)?,
        audio_duration_secs: row.try_get_by_index(12)?,
        rtf: row.try_get_by_index(13)?,
        tokens_generated: row
            .try_get_by_index::<Option<i64>>(14)?
            .and_then(i64_to_usize),
        audio_mime_type: row.try_get_by_index(15)?,
        audio_filename: row.try_get_by_index(16)?,
    })
}

fn map_speech_history_record(row: &QueryResult) -> anyhow::Result<SpeechHistoryRecord> {
    let route_raw: String = row.try_get_by_index(2)?;
    let route_kind =
        SpeechRouteKind::from_db_value(route_raw.as_str()).unwrap_or(SpeechRouteKind::TextToSpeech);
    let processing_status_raw: String = row.try_get_by_index(3)?;

    Ok(SpeechHistoryRecord {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?),
        route_kind,
        processing_status: SpeechHistoryProcessingStatus::from_db_value(
            processing_status_raw.as_str(),
        )
        .unwrap_or_default(),
        processing_error: row.try_get_by_index(4)?,
        model_id: row.try_get_by_index(5)?,
        speaker: row.try_get_by_index(6)?,
        language: row.try_get_by_index(7)?,
        saved_voice_id: row.try_get_by_index(8)?,
        speed: row.try_get_by_index(9)?,
        input_text: row.try_get_by_index(10)?,
        voice_description: row.try_get_by_index(11)?,
        reference_text: row.try_get_by_index(12)?,
        generation_time_ms: row.try_get_by_index(13)?,
        audio_duration_secs: row.try_get_by_index(14)?,
        rtf: row.try_get_by_index(15)?,
        tokens_generated: row
            .try_get_by_index::<Option<i64>>(16)?
            .and_then(i64_to_usize),
        audio_mime_type: row.try_get_by_index(17)?,
        audio_filename: row.try_get_by_index(18)?,
    })
}

fn input_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No text input".to_string();
    }
    truncate_string(&normalized, 180)
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let normalized = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        " ".to_string()
    } else {
        truncate_string(&normalized, max_chars)
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

fn sanitize_media_path(raw: Option<&str>) -> Option<String> {
    let value = raw.unwrap_or("").trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
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
    use crate::test_support::env_lock;

    fn setup_store() -> (tempfile::TempDir, SpeechHistoryStore) {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("speech-history.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);
        let store = SpeechHistoryStore::initialize().expect("store");
        (temp_dir, store)
    }

    fn clear_env() {
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    fn ready_record() -> NewSpeechHistoryRecord {
        NewSpeechHistoryRecord {
            route_kind: SpeechRouteKind::TextToSpeech,
            processing_status: SpeechHistoryProcessingStatus::Ready,
            processing_error: None,
            model_id: Some("model-a".to_string()),
            speaker: Some("Serena".to_string()),
            language: Some("en".to_string()),
            saved_voice_id: None,
            speed: Some(1.0),
            input_text: "Hello world".to_string(),
            voice_description: None,
            reference_text: None,
            generation_time_ms: 20.0,
            audio_duration_secs: Some(1.5),
            rtf: Some(0.2),
            tokens_generated: Some(12),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("hello.wav".to_string()),
            audio_bytes: vec![5, 6, 7],
        }
    }

    #[tokio::test]
    async fn creates_lists_reads_audio_updates_status_and_deletes_record() {
        let _guard = env_lock();
        let (_temp, store) = setup_store();

        let created = store
            .create_record(ready_record())
            .await
            .expect("record should create");
        assert_eq!(
            created.processing_status,
            SpeechHistoryProcessingStatus::Ready
        );

        let audio = store
            .get_audio(SpeechRouteKind::TextToSpeech, created.id.clone())
            .await
            .expect("audio should load")
            .expect("audio should exist");
        assert_eq!(audio.audio_bytes, vec![5, 6, 7]);

        let (records, cursor) = store
            .list_records_page(SpeechRouteKind::TextToSpeech, 10, None)
            .await
            .expect("records should list");
        assert!(cursor.is_none());
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].input_preview, "Hello world");

        let failed = store
            .update_processing_status(
                SpeechRouteKind::TextToSpeech,
                created.id.clone(),
                SpeechHistoryProcessingStatus::Failed,
                Some("boom".to_string()),
            )
            .await
            .expect("status should update")
            .expect("record should exist");
        assert_eq!(
            failed.processing_status,
            SpeechHistoryProcessingStatus::Failed
        );
        assert_eq!(failed.processing_error.as_deref(), Some("boom"));

        assert!(store
            .delete_record(SpeechRouteKind::TextToSpeech, created.id.clone())
            .await
            .expect("record should delete"));
        assert!(store
            .get_record(SpeechRouteKind::TextToSpeech, created.id)
            .await
            .expect("record lookup should succeed")
            .is_none());

        clear_env();
    }

    #[tokio::test]
    async fn completes_pending_record_with_audio() {
        let _guard = env_lock();
        let (_temp, store) = setup_store();

        let pending = store
            .create_record(NewSpeechHistoryRecord {
                processing_status: SpeechHistoryProcessingStatus::Pending,
                audio_bytes: Vec::new(),
                audio_filename: None,
                ..ready_record()
            })
            .await
            .expect("pending record should create");
        assert!(store
            .get_audio(SpeechRouteKind::TextToSpeech, pending.id.clone())
            .await
            .expect("audio lookup should succeed")
            .is_none());

        let completed = store
            .complete_record(
                SpeechRouteKind::TextToSpeech,
                pending.id.clone(),
                CompleteSpeechHistoryRecord {
                    model_id: Some("model-b".to_string()),
                    speaker: Some("Alex".to_string()),
                    language: Some("en".to_string()),
                    saved_voice_id: Some("voice-1".to_string()),
                    speed: Some(0.9),
                    input_text: "Completed text".to_string(),
                    voice_description: None,
                    reference_text: None,
                    generation_time_ms: 30.0,
                    audio_duration_secs: Some(2.0),
                    rtf: Some(0.3),
                    tokens_generated: Some(18),
                    audio_mime_type: "audio/wav".to_string(),
                    audio_filename: Some("completed.wav".to_string()),
                    audio_bytes: vec![8, 9],
                },
            )
            .await
            .expect("record should complete")
            .expect("record should exist");
        assert_eq!(
            completed.processing_status,
            SpeechHistoryProcessingStatus::Ready
        );
        assert_eq!(completed.input_text, "Completed text");

        let audio = store
            .get_audio(SpeechRouteKind::TextToSpeech, pending.id)
            .await
            .expect("audio should load")
            .expect("audio should exist");
        assert_eq!(audio.audio_bytes, vec![8, 9]);

        clear_env();
    }
}

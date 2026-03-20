//! Persistent transcription history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self, MediaGroup};

const DEFAULT_LIST_LIMIT: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionWordRecord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegmentRecord {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub word_start: usize,
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription_preview: String,
    pub transcription_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRecord {
    pub id: String,
    pub created_at: u64,
    pub model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub transcription: String,
    pub segments: Vec<TranscriptionSegmentRecord>,
    pub words: Vec<TranscriptionWordRecord>,
}

#[derive(Debug, Clone)]
pub struct StoredTranscriptionAudio {
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewTranscriptionRecord {
    pub model_id: Option<String>,
    pub aligner_model_id: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    pub processing_time_ms: f64,
    pub rtf: Option<f64>,
    pub audio_mime_type: String,
    pub audio_filename: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub transcription: String,
    pub segments: Vec<TranscriptionSegmentRecord>,
    pub words: Vec<TranscriptionWordRecord>,
}

#[derive(Clone)]
pub struct TranscriptionStore {
    db_path: PathBuf,
    media_root: PathBuf,
}

impl TranscriptionStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare transcription storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!(
                "Failed to open transcription database: {}",
                db_path.display()
            )
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS transcription_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                model_id TEXT NULL,
                aligner_model_id TEXT NULL,
                language TEXT NULL,
                duration_secs REAL NULL,
                processing_time_ms REAL NOT NULL,
                rtf REAL NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL,
                transcription TEXT NOT NULL,
                segments_json TEXT NOT NULL DEFAULT '[]',
                words_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_transcription_records_created_at
                ON transcription_records(created_at DESC);
            "#,
        )
        .context("Failed to initialize transcription database schema")?;
        ensure_transcription_records_aligner_model_id_column(&conn)?;
        ensure_transcription_records_segments_json_column(&conn)?;
        ensure_transcription_records_words_json_column(&conn)?;

        Ok(Self {
            db_path,
            media_root,
        })
    }

    pub async fn list_records(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<TranscriptionRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    transcription
                FROM transcription_records
                ORDER BY created_at DESC, id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], |row| {
                let transcription: String = row.get(9)?;
                Ok(TranscriptionRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    model_id: row.get(2)?,
                    language: row.get(3)?,
                    duration_secs: row.get(4)?,
                    processing_time_ms: row.get(5)?,
                    rtf: row.get(6)?,
                    audio_mime_type: row.get(7)?,
                    audio_filename: row.get(8)?,
                    transcription_preview: transcription_preview(&transcription),
                    transcription_chars: transcription.chars().count(),
                })
            })?;

            let mut records = Vec::new();
            for row in rows {
                records.push(row?);
            }
            Ok(records)
        })
        .await
    }

    pub async fn get_record(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<TranscriptionRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        record_id: String,
    ) -> anyhow::Result<Option<StoredTranscriptionAudio>> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_storage_path, audio_mime_type, audio_filename
                    FROM transcription_records
                    WHERE id = ?1
                    "#,
                    params![record_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )
                .optional()?;

            let Some((audio_storage_path, audio_mime_type, audio_filename)) = audio else {
                return Ok(None);
            };

            let audio_bytes =
                storage_layout::read_media_file(&media_root, audio_storage_path.as_str())?;

            Ok(Some(StoredTranscriptionAudio {
                audio_bytes,
                audio_mime_type,
                audio_filename,
            }))
        })
        .await
    }

    pub async fn create_record(
        &self,
        record: NewTranscriptionRecord,
    ) -> anyhow::Result<TranscriptionRecord> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = format!("txr_{}", uuid::Uuid::new_v4().simple());

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let aligner_model_id = sanitize_optional_text(record.aligner_model_id.as_deref(), 160);
            let language = sanitize_optional_text(record.language.as_deref(), 80);
            let duration_secs = record.duration_secs.filter(|v| v.is_finite() && *v >= 0.0);
            let processing_time_ms = if record.processing_time_ms.is_finite() {
                record.processing_time_ms.max(0.0)
            } else {
                0.0
            };
            let rtf = record.rtf.filter(|v| v.is_finite() && *v >= 0.0);
            let audio_mime_type = sanitize_audio_mime_type(record.audio_mime_type.as_str());
            let audio_filename = sanitize_optional_text(record.audio_filename.as_deref(), 260);
            let transcription = sanitize_required_text(record.transcription.as_str(), 100_000);
            let segments = sanitize_segments(record.segments);
            let words = sanitize_words(record.words);

            if record.audio_bytes.is_empty() {
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let audio_storage_path = storage_layout::persist_audio_file(
                &media_root,
                MediaGroup::Uploads,
                "transcription",
                &record_id,
                audio_filename.as_deref(),
                audio_mime_type.as_str(),
                &record.audio_bytes,
            )?;

            let segments_json =
                serde_json::to_string(&segments).context("Failed serializing segments")?;
            let words_json = serde_json::to_string(&words).context("Failed serializing words")?;

            if let Err(err) = conn.execute(
                r#"
                INSERT INTO transcription_records (
                    id,
                    created_at,
                    model_id,
                    aligner_model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    transcription,
                    segments_json,
                    words_json
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                "#,
                params![
                    &record_id,
                    now,
                    model_id,
                    aligner_model_id,
                    language,
                    duration_secs,
                    processing_time_ms,
                    rtf,
                    audio_mime_type,
                    audio_filename,
                    audio_storage_path,
                    transcription,
                    segments_json,
                    words_json,
                ],
            ) {
                let _ = storage_layout::delete_media_file(
                    &media_root,
                    Some(audio_storage_path.as_str()),
                );
                return Err(err).context("Failed to insert transcription record");
            }

            let created = fetch_record_without_audio(&conn, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created transcription record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn delete_record(&self, record_id: String) -> anyhow::Result<bool> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio_storage_path = conn
                .query_row(
                    "SELECT audio_storage_path FROM transcription_records WHERE id = ?1",
                    params![&record_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();

            let changed = conn.execute(
                "DELETE FROM transcription_records WHERE id = ?1",
                params![record_id],
            )?;

            if changed > 0 {
                storage_layout::delete_media_file(&media_root, audio_storage_path.as_deref())?;
            }

            Ok(changed > 0)
        })
        .await
    }

    async fn run_blocking<F, T>(&self, task_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || task_fn(db_path))
            .await
            .map_err(|err| anyhow!("Transcription storage worker failed: {err}"))?
    }
}

fn fetch_record_without_audio(
    conn: &Connection,
    record_id: &str,
) -> anyhow::Result<Option<TranscriptionRecord>> {
    let record = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                model_id,
                aligner_model_id,
                language,
                duration_secs,
                processing_time_ms,
                rtf,
                audio_mime_type,
                audio_filename,
                transcription,
                segments_json,
                words_json
            FROM transcription_records
            WHERE id = ?1
            "#,
            params![record_id],
            map_transcription_record,
        )
        .optional()?;
    Ok(record)
}

fn map_transcription_record(row: &Row<'_>) -> rusqlite::Result<TranscriptionRecord> {
    Ok(TranscriptionRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        model_id: row.get(2)?,
        aligner_model_id: row.get(3)?,
        language: row.get(4)?,
        duration_secs: row.get(5)?,
        processing_time_ms: row.get(6)?,
        rtf: row.get(7)?,
        audio_mime_type: row.get(8)?,
        audio_filename: row.get(9)?,
        transcription: row.get(10)?,
        segments: parse_json_vec(row.get(11)?),
        words: parse_json_vec(row.get(12)?),
    })
}

fn transcription_preview(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return "No transcript".to_string();
    }
    truncate_string(&normalized, 160)
}

fn sanitize_required_text(raw: &str, max_chars: usize) -> String {
    let normalized = raw.trim();
    if normalized.is_empty() {
        String::new()
    } else {
        truncate_string(normalized, max_chars)
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

fn sanitize_segments(segments: Vec<TranscriptionSegmentRecord>) -> Vec<TranscriptionSegmentRecord> {
    segments
        .into_iter()
        .filter_map(|segment| {
            let start = segment.start;
            let end = segment.end;
            let text = sanitize_required_text(segment.text.as_str(), 20_000);
            if !start.is_finite() || !end.is_finite() || end <= start || text.is_empty() {
                return None;
            }

            Some(TranscriptionSegmentRecord {
                start: start.max(0.0),
                end: end.max(start.max(0.0)),
                text,
                word_start: segment.word_start,
                word_end: segment.word_end,
            })
        })
        .collect()
}

fn sanitize_words(words: Vec<TranscriptionWordRecord>) -> Vec<TranscriptionWordRecord> {
    words
        .into_iter()
        .filter_map(|word| {
            let start = word.start;
            let end = word.end;
            let token = sanitize_required_text(word.word.as_str(), 160);
            if !start.is_finite() || !end.is_finite() || end <= start || token.is_empty() {
                return None;
            }

            Some(TranscriptionWordRecord {
                word: token,
                start: start.max(0.0),
                end: end.max(start.max(0.0)),
            })
        })
        .collect()
}

fn sanitize_audio_mime_type(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "audio/wav".to_string()
    } else {
        truncate_string(trimmed, 80)
    }
}

fn ensure_transcription_records_aligner_model_id_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "aligner_model_id")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN aligner_model_id TEXT NULL",
        [],
    )
    .context("Failed adding transcription_records.aligner_model_id column")?;
    Ok(())
}

fn ensure_transcription_records_segments_json_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "segments_json")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN segments_json TEXT NOT NULL DEFAULT '[]'",
        [],
    )
    .context("Failed adding transcription_records.segments_json column")?;
    Ok(())
}

fn ensure_transcription_records_words_json_column(conn: &Connection) -> anyhow::Result<()> {
    if transcription_records_has_column(conn, "words_json")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE transcription_records ADD COLUMN words_json TEXT NOT NULL DEFAULT '[]'",
        [],
    )
    .context("Failed adding transcription_records.words_json column")?;
    Ok(())
}

fn transcription_records_has_column(conn: &Connection, target: &str) -> anyhow::Result<bool> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(transcription_records)")
        .context("Failed to inspect transcription_records schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query transcription_records schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading transcription_records schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading transcription_records column name")?;
        if name == target {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_json_vec<T>(raw: Option<String>) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    raw.and_then(|value| serde_json::from_str(value.as_str()).ok())
        .unwrap_or_default()
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

#[allow(dead_code)]
pub const fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_store() -> (TranscriptionStore, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "izwi-transcription-store-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let store =
            TranscriptionStore::initialize_at(root.join("izwi.sqlite3"), root.join("media"))
                .expect("test store should initialize");
        (store, root)
    }

    fn sample_record() -> NewTranscriptionRecord {
        NewTranscriptionRecord {
            model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
            aligner_model_id: Some("Qwen3-ForcedAligner-0.6B".to_string()),
            language: Some("English".to_string()),
            duration_secs: Some(6.0),
            processing_time_ms: 120.0,
            rtf: Some(0.5),
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("meeting.wav".to_string()),
            audio_bytes: vec![0_u8, 1_u8, 2_u8, 3_u8],
            transcription: "Hello there. General Kenobi.".to_string(),
            segments: vec![
                TranscriptionSegmentRecord {
                    start: 0.0,
                    end: 1.2,
                    text: "Hello there.".to_string(),
                    word_start: 0,
                    word_end: 1,
                },
                TranscriptionSegmentRecord {
                    start: 1.4,
                    end: 2.4,
                    text: "General Kenobi.".to_string(),
                    word_start: 2,
                    word_end: 3,
                },
            ],
            words: vec![
                TranscriptionWordRecord {
                    word: "Hello".to_string(),
                    start: 0.0,
                    end: 0.4,
                },
                TranscriptionWordRecord {
                    word: "there".to_string(),
                    start: 0.45,
                    end: 0.9,
                },
                TranscriptionWordRecord {
                    word: "General".to_string(),
                    start: 1.4,
                    end: 1.9,
                },
                TranscriptionWordRecord {
                    word: "Kenobi".to_string(),
                    start: 1.95,
                    end: 2.4,
                },
            ],
        }
    }

    #[tokio::test]
    async fn persists_alignment_metadata() {
        let (store, root) = build_test_store();

        let created = store
            .create_record(sample_record())
            .await
            .expect("record should be created");

        assert_eq!(
            created.aligner_model_id.as_deref(),
            Some("Qwen3-ForcedAligner-0.6B")
        );
        assert_eq!(created.words.len(), 4);
        assert_eq!(created.segments.len(), 2);

        let fetched = store
            .get_record(created.id.clone())
            .await
            .expect("fetch should succeed")
            .expect("record should exist");

        assert_eq!(fetched.words.len(), 4);
        assert_eq!(fetched.segments[0].text, "Hello there.");

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }

    #[tokio::test]
    async fn keeps_plain_transcriptions_without_timestamps() {
        let (store, root) = build_test_store();
        let mut record = sample_record();
        record.aligner_model_id = None;
        record.words = Vec::new();
        record.segments = Vec::new();

        let created = store
            .create_record(record)
            .await
            .expect("record should be created");

        assert!(created.aligner_model_id.is_none());
        assert!(created.words.is_empty());
        assert!(created.segments.is_empty());

        let summaries = store.list_records(10).await.expect("list should succeed");
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].transcription_preview.contains("Hello there."));

        std::fs::remove_dir_all(root).expect("test temp dir should be removable");
    }
}

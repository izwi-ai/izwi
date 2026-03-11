//! Persistent speech generation history storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self, MediaGroup};

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

#[derive(Debug, Clone, Serialize)]
pub struct SpeechHistoryRecordSummary {
    pub id: String,
    pub created_at: u64,
    pub route_kind: SpeechRouteKind,
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

#[derive(Debug, Clone, Serialize)]
pub struct SpeechHistoryRecord {
    pub id: String,
    pub created_at: u64,
    pub route_kind: SpeechRouteKind,
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
    db_path: PathBuf,
    media_root: PathBuf,
}

impl SpeechHistoryStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare speech history storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!(
                "Failed to open speech history database: {}",
                db_path.display()
            )
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS speech_history_records (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                route_kind TEXT NOT NULL,
                model_id TEXT NULL,
                speaker TEXT NULL,
                language TEXT NULL,
                saved_voice_id TEXT NULL,
                speed REAL NULL,
                input_text TEXT NOT NULL,
                voice_description TEXT NULL,
                reference_text TEXT NULL,
                generation_time_ms REAL NOT NULL,
                audio_duration_secs REAL NULL,
                rtf REAL NULL,
                tokens_generated INTEGER NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_speech_history_route_created_at
                ON speech_history_records(route_kind, created_at DESC);
            "#,
        )
        .context("Failed to initialize speech history database schema")?;

        ensure_speech_history_records_saved_voice_id_column(&conn)?;
        ensure_speech_history_records_speed_column(&conn)?;

        Ok(Self {
            db_path,
            media_root,
        })
    }

    pub async fn list_records(
        &self,
        route_kind: SpeechRouteKind,
        limit: usize,
    ) -> anyhow::Result<Vec<SpeechHistoryRecordSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 500).max(1)).unwrap_or(200);

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    created_at,
                    route_kind,
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
                "#,
            )?;

            let rows = stmt.query_map(params![route_kind.as_db_value(), list_limit], |row| {
                let input_text: String = row.get(8)?;
                let route_raw: String = row.get(2)?;
                let route_kind = SpeechRouteKind::from_db_value(route_raw.as_str())
                    .unwrap_or(SpeechRouteKind::TextToSpeech);
                Ok(SpeechHistoryRecordSummary {
                    id: row.get(0)?,
                    created_at: i64_to_u64(row.get(1)?),
                    route_kind,
                    model_id: row.get(3)?,
                    speaker: row.get(4)?,
                    language: row.get(5)?,
                    saved_voice_id: row.get(6)?,
                    speed: row.get(7)?,
                    input_preview: input_preview(input_text.as_str()),
                    input_chars: input_text.chars().count(),
                    generation_time_ms: row.get(9)?,
                    audio_duration_secs: row.get(10)?,
                    rtf: row.get(11)?,
                    tokens_generated: row
                        .get::<_, Option<i64>>(12)?
                        .and_then(|value| i64_to_usize(value)),
                    audio_mime_type: row.get(13)?,
                    audio_filename: row.get(14)?,
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
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<Option<SpeechHistoryRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let record = fetch_record_without_audio(&conn, route_kind, &record_id)?;
            Ok(record)
        })
        .await
    }

    pub async fn get_audio(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<Option<StoredSpeechAudio>> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio = conn
                .query_row(
                    r#"
                    SELECT audio_storage_path, audio_mime_type, audio_filename
                    FROM speech_history_records
                    WHERE route_kind = ?1 AND id = ?2
                    "#,
                    params![route_kind.as_db_value(), record_id],
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

            Ok(Some(StoredSpeechAudio {
                audio_bytes,
                audio_mime_type,
                audio_filename,
            }))
        })
        .await
    }

    pub async fn create_record(
        &self,
        record: NewSpeechHistoryRecord,
    ) -> anyhow::Result<SpeechHistoryRecord> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let record_id = format!("shr_{}", uuid::Uuid::new_v4().simple());

            let model_id = sanitize_optional_text(record.model_id.as_deref(), 160);
            let speaker = sanitize_optional_text(record.speaker.as_deref(), 120);
            let language = sanitize_optional_text(record.language.as_deref(), 80);
            let saved_voice_id = sanitize_optional_text(record.saved_voice_id.as_deref(), 160);
            let speed = record
                .speed
                .filter(|value| value.is_finite() && *value > 0.0);
            let input_text = sanitize_required_text(record.input_text.as_str(), 20_000);
            let voice_description =
                sanitize_optional_text(record.voice_description.as_deref(), 2_000);
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
                return Err(anyhow!("Audio payload cannot be empty"));
            }

            let namespace = format!("speech/{}", record.route_kind.as_db_value());
            let audio_storage_path = storage_layout::persist_audio_file(
                &media_root,
                MediaGroup::Generated,
                namespace.as_str(),
                &record_id,
                audio_filename.as_deref(),
                audio_mime_type.as_str(),
                &record.audio_bytes,
            )?;

            if let Err(err) = conn.execute(
                r#"
                INSERT INTO speech_history_records (
                    id,
                    created_at,
                    route_kind,
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
                    audio_filename,
                    audio_storage_path
                )
                VALUES (
                    ?1,
                    ?2,
                    ?3,
                    ?4,
                    ?5,
                    ?6,
                    ?7,
                    ?8,
                    ?9,
                    ?10,
                    ?11,
                    ?12,
                    ?13,
                    ?14,
                    ?15,
                    ?16,
                    ?17,
                    ?18
                )
                "#,
                params![
                    &record_id,
                    now,
                    record.route_kind.as_db_value(),
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
                    audio_filename,
                    audio_storage_path
                ],
            ) {
                let _ = storage_layout::delete_media_file(
                    &media_root,
                    Some(audio_storage_path.as_str()),
                );
                return Err(err).context("Failed to insert speech history record");
            }

            let created = fetch_record_without_audio(&conn, record.route_kind, &record_id)?
                .ok_or_else(|| anyhow!("Failed to fetch created speech history record"))?;
            Ok(created)
        })
        .await
    }

    pub async fn delete_record(
        &self,
        route_kind: SpeechRouteKind,
        record_id: String,
    ) -> anyhow::Result<bool> {
        let media_root = self.media_root.clone();
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let audio_storage_path = conn
                .query_row(
                    "SELECT audio_storage_path FROM speech_history_records WHERE route_kind = ?1 AND id = ?2",
                    params![route_kind.as_db_value(), &record_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();
            let changed = conn.execute(
                "DELETE FROM speech_history_records WHERE route_kind = ?1 AND id = ?2",
                params![route_kind.as_db_value(), record_id],
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
            .map_err(|err| anyhow!("Speech history storage worker failed: {err}"))?
    }
}

fn fetch_record_without_audio(
    conn: &Connection,
    route_kind: SpeechRouteKind,
    record_id: &str,
) -> anyhow::Result<Option<SpeechHistoryRecord>> {
    let record = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                route_kind,
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
            FROM speech_history_records
            WHERE route_kind = ?1 AND id = ?2
            "#,
            params![route_kind.as_db_value(), record_id],
            map_speech_history_record,
        )
        .optional()?;
    Ok(record)
}

fn map_speech_history_record(row: &Row<'_>) -> rusqlite::Result<SpeechHistoryRecord> {
    let route_raw: String = row.get(2)?;
    let route_kind =
        SpeechRouteKind::from_db_value(route_raw.as_str()).unwrap_or(SpeechRouteKind::TextToSpeech);

    Ok(SpeechHistoryRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        route_kind,
        model_id: row.get(3)?,
        speaker: row.get(4)?,
        language: row.get(5)?,
        saved_voice_id: row.get(6)?,
        speed: row.get(7)?,
        input_text: row.get(8)?,
        voice_description: row.get(9)?,
        reference_text: row.get(10)?,
        generation_time_ms: row.get(11)?,
        audio_duration_secs: row.get(12)?,
        rtf: row.get(13)?,
        tokens_generated: row
            .get::<_, Option<i64>>(14)?
            .and_then(|value| i64_to_usize(value)),
        audio_mime_type: row.get(15)?,
        audio_filename: row.get(16)?,
    })
}

fn ensure_speech_history_records_saved_voice_id_column(conn: &Connection) -> anyhow::Result<()> {
    if speech_history_records_has_column(conn, "saved_voice_id")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE speech_history_records ADD COLUMN saved_voice_id TEXT NULL",
        [],
    )
    .context("Failed adding speech_history_records.saved_voice_id column")?;
    Ok(())
}

fn ensure_speech_history_records_speed_column(conn: &Connection) -> anyhow::Result<()> {
    if speech_history_records_has_column(conn, "speed")? {
        return Ok(());
    }

    conn.execute(
        "ALTER TABLE speech_history_records ADD COLUMN speed REAL NULL",
        [],
    )
    .context("Failed adding speech_history_records.speed column")?;
    Ok(())
}

fn speech_history_records_has_column(conn: &Connection, target: &str) -> anyhow::Result<bool> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(speech_history_records)")
        .context("Failed to inspect speech_history_records schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query speech_history_records schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading speech_history_records schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading speech_history_records column name")?;
        if name == target {
            return Ok(true);
        }
    }

    Ok(false)
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

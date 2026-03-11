//! Persistent text-to-speech project storage backed by SQLite.

use anyhow::Context;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout::{self};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtsProjectVoiceMode {
    BuiltIn,
    Saved,
}

impl TtsProjectVoiceMode {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::BuiltIn => "built_in",
            Self::Saved => "saved",
        }
    }

    fn from_db_value(raw: &str) -> Option<Self> {
        match raw {
            "built_in" => Some(Self::BuiltIn),
            "saved" => Some(Self::Saved),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TtsProjectSummary {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: TtsProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segment_count: usize,
    pub rendered_segment_count: usize,
    pub total_chars: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TtsProjectSegmentRecord {
    pub id: String,
    pub project_id: String,
    pub position: usize,
    pub text: String,
    pub input_chars: usize,
    pub speech_record_id: Option<String>,
    pub updated_at: u64,
    pub generation_time_ms: Option<f64>,
    pub audio_duration_secs: Option<f64>,
    pub audio_filename: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TtsProjectRecord {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: TtsProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<TtsProjectSegmentRecord>,
}

#[derive(Debug, Clone)]
pub struct NewTtsProjectSegment {
    pub position: usize,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct NewTtsProjectRecord {
    pub name: String,
    pub source_filename: Option<String>,
    pub source_text: String,
    pub model_id: Option<String>,
    pub voice_mode: TtsProjectVoiceMode,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub speed: Option<f64>,
    pub segments: Vec<NewTtsProjectSegment>,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateTtsProjectRecord {
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub voice_mode: Option<TtsProjectVoiceMode>,
    pub speaker: Option<Option<String>>,
    pub saved_voice_id: Option<Option<String>>,
    pub speed: Option<Option<f64>>,
}

#[derive(Clone)]
pub struct TtsProjectStore {
    db_path: PathBuf,
}

impl TtsProjectStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        Self::initialize_at(db_path, media_root)
    }

    fn initialize_at(db_path: PathBuf, media_root: PathBuf) -> anyhow::Result<Self> {
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare TTS project storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path).with_context(|| {
            format!("Failed to open TTS project database: {}", db_path.display())
        })?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS tts_projects (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                name TEXT NOT NULL,
                source_filename TEXT NULL,
                source_text TEXT NOT NULL,
                model_id TEXT NULL,
                voice_mode TEXT NOT NULL,
                speaker TEXT NULL,
                saved_voice_id TEXT NULL,
                speed REAL NULL
            );

            CREATE TABLE IF NOT EXISTS tts_project_segments (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                speech_record_id TEXT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(project_id) REFERENCES tts_projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tts_projects_updated_at
                ON tts_projects(updated_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_tts_project_segments_project_position
                ON tts_project_segments(project_id, position ASC);
            "#,
        )
        .context("Failed to initialize TTS project database schema")?;

        Ok(Self { db_path })
    }

    pub async fn list_projects(&self, limit: usize) -> anyhow::Result<Vec<TtsProjectSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let list_limit = i64::try_from(limit.clamp(1, 200).max(1)).unwrap_or(100);
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    p.id,
                    p.created_at,
                    p.updated_at,
                    p.name,
                    p.source_filename,
                    p.model_id,
                    p.voice_mode,
                    p.speaker,
                    p.saved_voice_id,
                    p.speed,
                    COUNT(s.id) AS segment_count,
                    COALESCE(SUM(CASE WHEN s.speech_record_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS rendered_segment_count,
                    COALESCE(SUM(LENGTH(s.text)), 0) AS total_chars
                FROM tts_projects p
                LEFT JOIN tts_project_segments s ON s.project_id = p.id
                GROUP BY
                    p.id,
                    p.created_at,
                    p.updated_at,
                    p.name,
                    p.source_filename,
                    p.model_id,
                    p.voice_mode,
                    p.speaker,
                    p.saved_voice_id,
                    p.speed
                ORDER BY p.updated_at DESC, p.id DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![list_limit], map_project_summary_row)?;
            let mut projects = Vec::new();
            for row in rows {
                projects.push(row?);
            }
            Ok(projects)
        })
        .await
    }

    pub async fn get_project(
        &self,
        project_id: String,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn create_project(
        &self,
        record: NewTtsProjectRecord,
    ) -> anyhow::Result<TtsProjectRecord> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let now = now_unix_millis_i64();
            let project_id = format!("ttsp_{}", uuid::Uuid::new_v4().simple());

            tx.execute(
                r#"
                INSERT INTO tts_projects (
                    id,
                    created_at,
                    updated_at,
                    name,
                    source_filename,
                    source_text,
                    model_id,
                    voice_mode,
                    speaker,
                    saved_voice_id,
                    speed
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                "#,
                params![
                    project_id,
                    now,
                    now,
                    record.name,
                    record.source_filename,
                    record.source_text,
                    record.model_id,
                    record.voice_mode.as_db_value(),
                    record.speaker,
                    record.saved_voice_id,
                    record.speed,
                ],
            )?;

            for segment in record.segments {
                let segment_id = format!("ttss_{}", uuid::Uuid::new_v4().simple());
                tx.execute(
                    r#"
                    INSERT INTO tts_project_segments (
                        id,
                        project_id,
                        position,
                        text,
                        speech_record_id,
                        updated_at
                    ) VALUES (?1, ?2, ?3, ?4, NULL, ?5)
                    "#,
                    params![
                        segment_id,
                        project_id,
                        usize_to_i64(segment.position),
                        segment.text,
                        now,
                    ],
                )?;
            }

            tx.commit()?;
            fetch_project(&conn, &project_id)?
                .ok_or_else(|| anyhow::anyhow!("Created TTS project was not found"))
        })
        .await
    }

    pub async fn update_project(
        &self,
        project_id: String,
        update: UpdateTtsProjectRecord,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let current = tx
                .query_row(
                    r#"
                    SELECT
                        name,
                        model_id,
                        voice_mode,
                        speaker,
                        saved_voice_id,
                        speed
                    FROM tts_projects
                    WHERE id = ?1
                    "#,
                    params![project_id.as_str()],
                    |row| {
                        let voice_mode_raw: String = row.get(2)?;
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            TtsProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
                                .unwrap_or(TtsProjectVoiceMode::BuiltIn),
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                            row.get::<_, Option<f64>>(5)?,
                        ))
                    },
                )
                .optional()?;

            let Some((name, model_id, voice_mode, speaker, saved_voice_id, speed)) = current else {
                return Ok(None);
            };

            let next_name = update.name.unwrap_or(name);
            let next_model_id = update.model_id.or(model_id);
            let next_voice_mode = update.voice_mode.unwrap_or(voice_mode);
            let next_speaker = update.speaker.unwrap_or(speaker);
            let next_saved_voice_id = update.saved_voice_id.unwrap_or(saved_voice_id);
            let next_speed = update.speed.unwrap_or(speed);
            let now = now_unix_millis_i64();

            tx.execute(
                r#"
                UPDATE tts_projects
                SET
                    updated_at = ?2,
                    name = ?3,
                    model_id = ?4,
                    voice_mode = ?5,
                    speaker = ?6,
                    saved_voice_id = ?7,
                    speed = ?8
                WHERE id = ?1
                "#,
                params![
                    project_id.as_str(),
                    now,
                    next_name,
                    next_model_id,
                    next_voice_mode.as_db_value(),
                    next_speaker,
                    next_saved_voice_id,
                    next_speed,
                ],
            )?;

            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn update_segment_text(
        &self,
        project_id: String,
        segment_id: String,
        text: String,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE tts_project_segments
                SET
                    text = ?3,
                    speech_record_id = NULL,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str(), text, now],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn split_segment(
        &self,
        project_id: String,
        segment_id: String,
        before_text: String,
        after_text: String,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let position = tx
                .query_row(
                    r#"
                    SELECT position
                    FROM tts_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str()],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?;

            let Some(position) = position else {
                tx.rollback()?;
                return Ok(None);
            };

            tx.execute(
                r#"
                UPDATE tts_project_segments
                SET position = position + 1
                WHERE project_id = ?1 AND position > ?2
                "#,
                params![project_id.as_str(), position],
            )?;

            tx.execute(
                r#"
                UPDATE tts_project_segments
                SET
                    text = ?3,
                    speech_record_id = NULL,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str(), before_text, now],
            )?;

            let next_segment_id = format!("ttss_{}", uuid::Uuid::new_v4().simple());
            tx.execute(
                r#"
                INSERT INTO tts_project_segments (
                    id,
                    project_id,
                    position,
                    text,
                    speech_record_id,
                    updated_at
                ) VALUES (?1, ?2, ?3, ?4, NULL, ?5)
                "#,
                params![
                    next_segment_id,
                    project_id.as_str(),
                    position + 1,
                    after_text,
                    now
                ],
            )?;

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn delete_segment(
        &self,
        project_id: String,
        segment_id: String,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let position = tx
                .query_row(
                    r#"
                    SELECT position
                    FROM tts_project_segments
                    WHERE project_id = ?1 AND id = ?2
                    "#,
                    params![project_id.as_str(), segment_id.as_str()],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?;

            let Some(position) = position else {
                tx.rollback()?;
                return Ok(None);
            };

            let deleted = tx.execute(
                r#"
                DELETE FROM tts_project_segments
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![project_id.as_str(), segment_id.as_str()],
            )?;

            if deleted == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            tx.execute(
                r#"
                UPDATE tts_project_segments
                SET position = position - 1
                WHERE project_id = ?1 AND position > ?2
                "#,
                params![project_id.as_str(), position],
            )?;

            sync_project_source_text(&tx, project_id.as_str())?;
            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn attach_segment_record(
        &self,
        project_id: String,
        segment_id: String,
        speech_record_id: String,
    ) -> anyhow::Result<Option<TtsProjectRecord>> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            let now = now_unix_millis_i64();

            let updated = tx.execute(
                r#"
                UPDATE tts_project_segments
                SET
                    speech_record_id = ?3,
                    updated_at = ?4
                WHERE project_id = ?1 AND id = ?2
                "#,
                params![
                    project_id.as_str(),
                    segment_id.as_str(),
                    speech_record_id,
                    now
                ],
            )?;

            if updated == 0 {
                tx.rollback()?;
                return Ok(None);
            }

            touch_project(&tx, project_id.as_str(), now)?;
            tx.commit()?;
            fetch_project(&conn, &project_id)
        })
        .await
    }

    pub async fn delete_project(&self, project_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;
            tx.execute(
                "DELETE FROM tts_project_segments WHERE project_id = ?1",
                params![project_id.as_str()],
            )?;
            let deleted = tx.execute(
                "DELETE FROM tts_projects WHERE id = ?1",
                params![project_id.as_str()],
            )?;
            tx.commit()?;
            Ok(deleted > 0)
        })
        .await
    }

    async fn run_blocking<T, F>(&self, work: F) -> anyhow::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || work(db_path))
            .await
            .context("TTS project storage task join error")?
    }
}

fn fetch_project(conn: &Connection, project_id: &str) -> anyhow::Result<Option<TtsProjectRecord>> {
    let project = conn
        .query_row(
            r#"
            SELECT
                id,
                created_at,
                updated_at,
                name,
                source_filename,
                source_text,
                model_id,
                voice_mode,
                speaker,
                saved_voice_id,
                speed
            FROM tts_projects
            WHERE id = ?1
            "#,
            params![project_id],
            map_project_row,
        )
        .optional()?;

    let Some(mut project) = project else {
        return Ok(None);
    };

    let mut stmt = conn.prepare(
        r#"
        SELECT
            s.id,
            s.project_id,
            s.position,
            s.text,
            s.speech_record_id,
            s.updated_at,
            h.generation_time_ms,
            h.audio_duration_secs,
            h.audio_filename
        FROM tts_project_segments s
        LEFT JOIN speech_history_records h
            ON h.id = s.speech_record_id
           AND h.route_kind = 'text_to_speech'
        WHERE s.project_id = ?1
        ORDER BY s.position ASC, s.id ASC
        "#,
    )?;

    let rows = stmt.query_map(params![project_id], map_segment_row)?;
    let mut segments = Vec::new();
    for row in rows {
        segments.push(row?);
    }
    project.segments = segments;

    Ok(Some(project))
}

fn map_project_summary_row(row: &Row<'_>) -> rusqlite::Result<TtsProjectSummary> {
    let voice_mode_raw: String = row.get(6)?;
    let segment_count = row.get::<_, i64>(10)?;
    let rendered_segment_count = row.get::<_, i64>(11)?;
    let total_chars = row.get::<_, i64>(12)?;

    Ok(TtsProjectSummary {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        source_filename: row.get(4)?,
        model_id: row.get(5)?,
        voice_mode: TtsProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(TtsProjectVoiceMode::BuiltIn),
        speaker: row.get(7)?,
        saved_voice_id: row.get(8)?,
        speed: row.get(9)?,
        segment_count: i64_to_usize(segment_count).unwrap_or_default(),
        rendered_segment_count: i64_to_usize(rendered_segment_count).unwrap_or_default(),
        total_chars: i64_to_usize(total_chars).unwrap_or_default(),
    })
}

fn map_project_row(row: &Row<'_>) -> rusqlite::Result<TtsProjectRecord> {
    let voice_mode_raw: String = row.get(7)?;
    Ok(TtsProjectRecord {
        id: row.get(0)?,
        created_at: i64_to_u64(row.get(1)?),
        updated_at: i64_to_u64(row.get(2)?),
        name: row.get(3)?,
        source_filename: row.get(4)?,
        source_text: row.get(5)?,
        model_id: row.get(6)?,
        voice_mode: TtsProjectVoiceMode::from_db_value(voice_mode_raw.as_str())
            .unwrap_or(TtsProjectVoiceMode::BuiltIn),
        speaker: row.get(8)?,
        saved_voice_id: row.get(9)?,
        speed: row.get(10)?,
        segments: Vec::new(),
    })
}

fn map_segment_row(row: &Row<'_>) -> rusqlite::Result<TtsProjectSegmentRecord> {
    let text: String = row.get(3)?;
    Ok(TtsProjectSegmentRecord {
        id: row.get(0)?,
        project_id: row.get(1)?,
        position: i64_to_usize(row.get(2)?).unwrap_or_default(),
        input_chars: text.chars().count(),
        text,
        speech_record_id: row.get(4)?,
        updated_at: i64_to_u64(row.get(5)?),
        generation_time_ms: row.get(6)?,
        audio_duration_secs: row.get(7)?,
        audio_filename: row.get(8)?,
    })
}

fn touch_project(conn: &Connection, project_id: &str, updated_at: i64) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE tts_projects SET updated_at = ?2 WHERE id = ?1",
        params![project_id, updated_at],
    )?;
    Ok(())
}

fn sync_project_source_text(conn: &Connection, project_id: &str) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        r#"
        SELECT text
        FROM tts_project_segments
        WHERE project_id = ?1
        ORDER BY position ASC, id ASC
        "#,
    )?;
    let rows = stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;

    let mut segment_texts = Vec::new();
    for row in rows {
        segment_texts.push(row?);
    }

    conn.execute(
        "UPDATE tts_projects SET source_text = ?2 WHERE id = ?1",
        params![project_id, segment_texts.join("\n\n")],
    )?;
    Ok(())
}

fn now_unix_millis_i64() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    i64::try_from(millis).unwrap_or(i64::MAX)
}

fn i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

fn i64_to_usize(value: i64) -> Option<usize> {
    usize::try_from(value).ok()
}

fn usize_to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        NewTtsProjectRecord, NewTtsProjectSegment, TtsProjectStore, TtsProjectVoiceMode,
        UpdateTtsProjectRecord,
    };
    use crate::storage_layout;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_env_root(prefix: &str) -> PathBuf {
        let mut root = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        root.push(format!("izwi-{prefix}-{nonce}"));
        root
    }

    #[tokio::test]
    async fn create_and_update_project_round_trips_segments() {
        let root = test_env_root("tts-project-store");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store =
            TtsProjectStore::initialize_at(db_path, media_dir).expect("store should initialize");
        let conn = storage_layout::open_sqlite_connection(&store.db_path)
            .expect("speech schema connection should open");
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
                generation_time_ms REAL NULL,
                audio_duration_secs REAL NULL,
                rtf REAL NULL,
                tokens_generated INTEGER NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );
            "#,
        )
        .expect("speech schema should initialize");
        let created = store
            .create_project(NewTtsProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: TtsProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.1),
                segments: vec![
                    NewTtsProjectSegment {
                        position: 0,
                        text: "Hello world.".to_string(),
                    },
                    NewTtsProjectSegment {
                        position: 1,
                        text: "Another sentence.".to_string(),
                    },
                ],
            })
            .await
            .expect("project should be created");

        assert_eq!(created.segments.len(), 2);
        assert_eq!(created.speaker.as_deref(), Some("Vivian"));

        let updated = store
            .update_project(
                created.id.clone(),
                UpdateTtsProjectRecord {
                    voice_mode: Some(TtsProjectVoiceMode::Saved),
                    speaker: Some(None),
                    saved_voice_id: Some(Some("voice-1".to_string())),
                    ..UpdateTtsProjectRecord::default()
                },
            )
            .await
            .expect("update should succeed")
            .expect("project should exist");

        assert_eq!(updated.voice_mode, TtsProjectVoiceMode::Saved);
        assert_eq!(updated.saved_voice_id.as_deref(), Some("voice-1"));
        assert_eq!(updated.speaker, None);

        let segment = updated.segments.first().expect("segment should exist");
        let refreshed = store
            .update_segment_text(
                updated.id.clone(),
                segment.id.clone(),
                "Updated line.".to_string(),
            )
            .await
            .expect("segment update should succeed")
            .expect("project should exist");

        assert_eq!(refreshed.segments[0].text, "Updated line.");
        assert_eq!(refreshed.segments[0].speech_record_id, None);

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn split_and_delete_segment_reorders_project_blocks() {
        let root = test_env_root("tts-project-segment-ops");
        let db_path = root.join("test.sqlite3");
        let media_dir = root.join("media");
        std::fs::create_dir_all(&root).expect("root should be created");

        let store =
            TtsProjectStore::initialize_at(db_path, media_dir).expect("store should initialize");
        let conn = storage_layout::open_sqlite_connection(&store.db_path)
            .expect("speech schema connection should open");
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
                generation_time_ms REAL NULL,
                audio_duration_secs REAL NULL,
                rtf REAL NULL,
                tokens_generated INTEGER NULL,
                audio_mime_type TEXT NOT NULL,
                audio_filename TEXT NULL,
                audio_storage_path TEXT NOT NULL
            );
            "#,
        )
        .expect("speech schema should initialize");

        let created = store
            .create_project(NewTtsProjectRecord {
                name: "Narration project".to_string(),
                source_filename: Some("script.txt".to_string()),
                source_text: "Hello world. Another sentence.".to_string(),
                model_id: Some("Qwen3-TTS".to_string()),
                voice_mode: TtsProjectVoiceMode::BuiltIn,
                speaker: Some("Vivian".to_string()),
                saved_voice_id: None,
                speed: Some(1.0),
                segments: vec![
                    NewTtsProjectSegment {
                        position: 0,
                        text: "Hello world. Another sentence.".to_string(),
                    },
                    NewTtsProjectSegment {
                        position: 1,
                        text: "Closing line.".to_string(),
                    },
                ],
            })
            .await
            .expect("project should be created");

        let first_segment = created.segments.first().expect("segment should exist");
        let split = store
            .split_segment(
                created.id.clone(),
                first_segment.id.clone(),
                "Hello world.".to_string(),
                "Another sentence.".to_string(),
            )
            .await
            .expect("split should succeed")
            .expect("project should exist");

        assert_eq!(split.segments.len(), 3);
        assert_eq!(split.segments[0].text, "Hello world.");
        assert_eq!(split.segments[1].text, "Another sentence.");
        assert_eq!(split.segments[2].text, "Closing line.");
        assert_eq!(
            split.source_text,
            "Hello world.\n\nAnother sentence.\n\nClosing line."
        );

        let deleted = store
            .delete_segment(split.id.clone(), split.segments[1].id.clone())
            .await
            .expect("delete should succeed")
            .expect("project should exist");

        assert_eq!(deleted.segments.len(), 2);
        assert_eq!(deleted.segments[0].position, 0);
        assert_eq!(deleted.segments[1].position, 1);
        assert_eq!(deleted.segments[0].text, "Hello world.");
        assert_eq!(deleted.segments[1].text, "Closing line.");
        assert_eq!(deleted.source_text, "Hello world.\n\nClosing line.");

        let _ = std::fs::remove_dir_all(root);
    }
}

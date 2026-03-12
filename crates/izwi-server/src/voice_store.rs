use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use std::path::PathBuf;
use tokio::task;

use crate::storage_layout;
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_SYSTEM_PROMPT, DEFAULT_VOICE_PROFILE_ID, DEFAULT_VOICE_PROFILE_NAME,
};

#[derive(Debug, Clone, Serialize)]
pub struct VoiceProfile {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub observational_memory_enabled: bool,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceSessionSummary {
    pub id: String,
    pub profile_id: String,
    pub mode: String,
    pub system_prompt: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub ended_at: Option<u64>,
    pub turn_count: usize,
    pub last_user_text: Option<String>,
    pub last_assistant_text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceTurnRecord {
    pub id: String,
    pub session_id: String,
    pub utterance_id: String,
    pub utterance_seq: u64,
    pub mode: String,
    pub status: String,
    pub status_reason: Option<String>,
    pub vad_end_reason: Option<String>,
    pub user_text: Option<String>,
    pub assistant_text: Option<String>,
    pub assistant_raw_text: Option<String>,
    pub language: Option<String>,
    pub audio_duration_secs: Option<f32>,
    pub asr_model_id: Option<String>,
    pub text_model_id: Option<String>,
    pub tts_model_id: Option<String>,
    pub s2s_model_id: Option<String>,
    pub speaker: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceSessionDetail {
    pub session: VoiceSessionSummary,
    pub turns: Vec<VoiceTurnRecord>,
}

pub struct CreateVoiceSessionRequest {
    pub profile_id: String,
    pub mode: String,
    pub system_prompt: String,
}

pub struct CreateVoiceTurnRequest {
    pub session_id: String,
    pub utterance_id: String,
    pub utterance_seq: u64,
    pub mode: String,
    pub vad_end_reason: Option<String>,
    pub asr_model_id: Option<String>,
    pub text_model_id: Option<String>,
    pub tts_model_id: Option<String>,
    pub s2s_model_id: Option<String>,
    pub speaker: Option<String>,
}

#[derive(Clone)]
pub struct VoiceStore {
    db_path: PathBuf,
}

impl VoiceStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare voice storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path)
            .with_context(|| format!("Failed to open voice database: {}", db_path.display()))?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                observational_memory_enabled INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS voice_sessions (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('modular', 'unified')),
                system_prompt TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                ended_at INTEGER NULL,
                FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS voice_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                utterance_id TEXT NOT NULL,
                utterance_seq INTEGER NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('modular', 'unified')),
                status TEXT NOT NULL CHECK(status IN ('processing', 'ok', 'error', 'timeout', 'interrupted', 'no_input')),
                status_reason TEXT NULL,
                vad_end_reason TEXT NULL,
                user_text TEXT NULL,
                assistant_text TEXT NULL,
                assistant_raw_text TEXT NULL,
                language TEXT NULL,
                audio_duration_secs REAL NULL,
                asr_model_id TEXT NULL,
                text_model_id TEXT NULL,
                tts_model_id TEXT NULL,
                s2s_model_id TEXT NULL,
                speaker TEXT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES voice_sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_voice_sessions_updated_at
                ON voice_sessions(updated_at DESC, created_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_turns_session_utterance
                ON voice_turns(session_id, utterance_seq);
            CREATE INDEX IF NOT EXISTS idx_voice_turns_session_created_at
                ON voice_turns(session_id, created_at ASC, id ASC);
            "#,
        )
        .context("Failed to initialize voice database schema")?;
        ensure_default_profile(&conn)?;

        Ok(Self { db_path })
    }

    pub async fn get_default_profile(&self) -> anyhow::Result<VoiceProfile> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            ensure_default_profile(&conn)?;
            conn.query_row(
                r#"
                SELECT
                    id,
                    name,
                    system_prompt,
                    observational_memory_enabled,
                    created_at,
                    updated_at
                FROM voice_profiles
                WHERE id = ?1
                "#,
                params![DEFAULT_VOICE_PROFILE_ID],
                map_voice_profile,
            )
            .context("Default voice profile not found")
        })
        .await
    }

    pub async fn update_default_profile(
        &self,
        name: Option<String>,
        system_prompt: Option<String>,
        observational_memory_enabled: Option<bool>,
    ) -> anyhow::Result<VoiceProfile> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            ensure_default_profile(&conn)?;

            let current = conn
                .query_row(
                    r#"
                    SELECT
                        id,
                        name,
                        system_prompt,
                        observational_memory_enabled,
                        created_at,
                        updated_at
                    FROM voice_profiles
                    WHERE id = ?1
                    "#,
                    params![DEFAULT_VOICE_PROFILE_ID],
                    map_voice_profile,
                )
                .context("Default voice profile not found")?;

            let next_name = sanitize_profile_name(name.as_deref()).unwrap_or(current.name);
            let next_prompt =
                sanitize_prompt(system_prompt.as_deref()).unwrap_or(current.system_prompt);
            let next_memory_enabled =
                observational_memory_enabled.unwrap_or(current.observational_memory_enabled);
            let now = now_unix_millis_i64();

            conn.execute(
                r#"
                UPDATE voice_profiles
                SET name = ?1,
                    system_prompt = ?2,
                    observational_memory_enabled = ?3,
                    updated_at = ?4
                WHERE id = ?5
                "#,
                params![
                    next_name,
                    next_prompt,
                    bool_to_i64(next_memory_enabled),
                    now,
                    DEFAULT_VOICE_PROFILE_ID
                ],
            )?;

            conn.query_row(
                r#"
                SELECT
                    id,
                    name,
                    system_prompt,
                    observational_memory_enabled,
                    created_at,
                    updated_at
                FROM voice_profiles
                WHERE id = ?1
                "#,
                params![DEFAULT_VOICE_PROFILE_ID],
                map_voice_profile,
            )
            .context("Updated voice profile not found")
        })
        .await
    }

    pub async fn create_session(
        &self,
        request: CreateVoiceSessionRequest,
    ) -> anyhow::Result<VoiceSessionSummary> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let session_id = format!("voice_sess_{}", uuid::Uuid::new_v4().simple());

            conn.execute(
                r#"
                INSERT INTO voice_sessions (
                    id,
                    profile_id,
                    mode,
                    system_prompt,
                    created_at,
                    updated_at,
                    ended_at
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL)
                "#,
                params![
                    session_id,
                    request.profile_id,
                    sanitize_mode(request.mode.as_str())?,
                    sanitize_prompt(Some(request.system_prompt.as_str()))
                        .unwrap_or_else(|| DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.to_string()),
                    now,
                    now,
                ],
            )?;

            fetch_session_summary(&conn, &session_id)?
                .ok_or_else(|| anyhow!("Created voice session not found"))
        })
        .await
    }

    pub async fn end_session(&self, session_id: String) -> anyhow::Result<()> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            conn.execute(
                r#"
                UPDATE voice_sessions
                SET updated_at = ?1,
                    ended_at = COALESCE(ended_at, ?1)
                WHERE id = ?2
                "#,
                params![now, session_id],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn list_sessions(&self, limit: usize) -> anyhow::Result<Vec<VoiceSessionSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
                SELECT
                    s.id,
                    s.profile_id,
                    s.mode,
                    s.system_prompt,
                    s.created_at,
                    s.updated_at,
                    s.ended_at,
                    (
                        SELECT COUNT(1)
                        FROM voice_turns t
                        WHERE t.session_id = s.id
                    ) AS turn_count,
                    (
                        SELECT t.user_text
                        FROM voice_turns t
                        WHERE t.session_id = s.id
                        ORDER BY t.created_at DESC, t.id DESC
                        LIMIT 1
                    ) AS last_user_text,
                    (
                        SELECT t.assistant_text
                        FROM voice_turns t
                        WHERE t.session_id = s.id
                        ORDER BY t.created_at DESC, t.id DESC
                        LIMIT 1
                    ) AS last_assistant_text
                FROM voice_sessions s
                ORDER BY s.updated_at DESC, s.created_at DESC
                LIMIT ?1
                "#,
            )?;

            let rows = stmt.query_map(params![limit.max(1) as i64], map_session_summary)?;
            let mut sessions = Vec::new();
            for row in rows {
                sessions.push(row?);
            }
            Ok(sessions)
        })
        .await
    }

    pub async fn get_session(
        &self,
        session_id: String,
    ) -> anyhow::Result<Option<VoiceSessionDetail>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let Some(session) = fetch_session_summary(&conn, &session_id)? else {
                return Ok(None);
            };

            let mut stmt = conn.prepare(
                r#"
                SELECT
                    id,
                    session_id,
                    utterance_id,
                    utterance_seq,
                    mode,
                    status,
                    status_reason,
                    vad_end_reason,
                    user_text,
                    assistant_text,
                    assistant_raw_text,
                    language,
                    audio_duration_secs,
                    asr_model_id,
                    text_model_id,
                    tts_model_id,
                    s2s_model_id,
                    speaker,
                    created_at,
                    updated_at
                FROM voice_turns
                WHERE session_id = ?1
                ORDER BY created_at ASC, id ASC
                "#,
            )?;
            let rows = stmt.query_map(params![session_id], map_turn_record)?;
            let mut turns = Vec::new();
            for row in rows {
                turns.push(row?);
            }

            Ok(Some(VoiceSessionDetail { session, turns }))
        })
        .await
    }

    pub async fn create_turn(
        &self,
        request: CreateVoiceTurnRequest,
    ) -> anyhow::Result<VoiceTurnRecord> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let turn_id = format!("voice_turn_{}", uuid::Uuid::new_v4().simple());
            let mode = sanitize_mode(request.mode.as_str())?;

            conn.execute(
                r#"
                INSERT INTO voice_turns (
                    id,
                    session_id,
                    utterance_id,
                    utterance_seq,
                    mode,
                    status,
                    status_reason,
                    vad_end_reason,
                    user_text,
                    assistant_text,
                    assistant_raw_text,
                    language,
                    audio_duration_secs,
                    asr_model_id,
                    text_model_id,
                    tts_model_id,
                    s2s_model_id,
                    speaker,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?1, ?2, ?3, ?4, ?5, 'processing', NULL, ?6, NULL, NULL, NULL, NULL, NULL,
                    ?7, ?8, ?9, ?10, ?11, ?12, ?12
                )
                "#,
                params![
                    turn_id,
                    request.session_id,
                    sanitize_required_text(request.utterance_id.as_str(), 160)?,
                    u64_to_i64(request.utterance_seq)?,
                    mode,
                    sanitize_optional_text(request.vad_end_reason.as_deref(), 48),
                    sanitize_optional_text(request.asr_model_id.as_deref(), 160),
                    sanitize_optional_text(request.text_model_id.as_deref(), 160),
                    sanitize_optional_text(request.tts_model_id.as_deref(), 160),
                    sanitize_optional_text(request.s2s_model_id.as_deref(), 160),
                    sanitize_optional_text(request.speaker.as_deref(), 160),
                    now,
                ],
            )?;

            touch_session_updated_at(&conn, request.session_id.as_str(), now)?;
            fetch_turn_record(&conn, &turn_id)?
                .ok_or_else(|| anyhow!("Created voice turn not found"))
        })
        .await
    }

    pub async fn update_turn_transcript(
        &self,
        turn_id: String,
        user_text: String,
        language: Option<String>,
        audio_duration_secs: Option<f32>,
    ) -> anyhow::Result<()> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let session_id = fetch_turn_session_id(&conn, &turn_id)?
                .ok_or_else(|| anyhow!("Voice turn not found"))?;
            conn.execute(
                r#"
                UPDATE voice_turns
                SET user_text = ?1,
                    language = ?2,
                    audio_duration_secs = ?3,
                    updated_at = ?4
                WHERE id = ?5
                "#,
                params![
                    sanitize_optional_text(Some(user_text.as_str()), 16000),
                    sanitize_optional_text(language.as_deref(), 64),
                    audio_duration_secs.map(f64::from),
                    now,
                    turn_id,
                ],
            )?;
            touch_session_updated_at(&conn, session_id.as_str(), now)?;
            Ok(())
        })
        .await
    }

    pub async fn update_turn_assistant(
        &self,
        turn_id: String,
        assistant_text: Option<String>,
        assistant_raw_text: Option<String>,
    ) -> anyhow::Result<()> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let session_id = fetch_turn_session_id(&conn, &turn_id)?
                .ok_or_else(|| anyhow!("Voice turn not found"))?;
            conn.execute(
                r#"
                UPDATE voice_turns
                SET assistant_text = ?1,
                    assistant_raw_text = ?2,
                    updated_at = ?3
                WHERE id = ?4
                "#,
                params![
                    sanitize_optional_text(assistant_text.as_deref(), 16000),
                    sanitize_optional_text(assistant_raw_text.as_deref(), 16000),
                    now,
                    turn_id,
                ],
            )?;
            touch_session_updated_at(&conn, session_id.as_str(), now)?;
            Ok(())
        })
        .await
    }

    pub async fn complete_turn(
        &self,
        turn_id: String,
        status: &str,
        status_reason: Option<String>,
    ) -> anyhow::Result<()> {
        let status = sanitize_turn_status(status)?;
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let session_id = fetch_turn_session_id(&conn, &turn_id)?
                .ok_or_else(|| anyhow!("Voice turn not found"))?;
            conn.execute(
                r#"
                UPDATE voice_turns
                SET status = ?1,
                    status_reason = ?2,
                    updated_at = ?3
                WHERE id = ?4
                "#,
                params![
                    status,
                    sanitize_optional_text(status_reason.as_deref(), 160),
                    now,
                    turn_id,
                ],
            )?;
            touch_session_updated_at(&conn, session_id.as_str(), now)?;
            Ok(())
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
            .map_err(|err| anyhow!("Voice storage worker failed: {err}"))?
    }
}

fn ensure_default_profile(conn: &Connection) -> anyhow::Result<()> {
    let exists = conn
        .query_row(
            "SELECT 1 FROM voice_profiles WHERE id = ?1 LIMIT 1",
            params![DEFAULT_VOICE_PROFILE_ID],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if exists {
        return Ok(());
    }

    let now = now_unix_millis_i64();
    conn.execute(
        r#"
        INSERT INTO voice_profiles (
            id,
            name,
            system_prompt,
            observational_memory_enabled,
            created_at,
            updated_at
        )
        VALUES (?1, ?2, ?3, 1, ?4, ?4)
        "#,
        params![
            DEFAULT_VOICE_PROFILE_ID,
            DEFAULT_VOICE_PROFILE_NAME,
            DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
            now,
        ],
    )?;

    Ok(())
}

fn fetch_session_summary(
    conn: &Connection,
    session_id: &str,
) -> anyhow::Result<Option<VoiceSessionSummary>> {
    let session = conn
        .query_row(
            r#"
            SELECT
                s.id,
                s.profile_id,
                s.mode,
                s.system_prompt,
                s.created_at,
                s.updated_at,
                s.ended_at,
                (
                    SELECT COUNT(1)
                    FROM voice_turns t
                    WHERE t.session_id = s.id
                ) AS turn_count,
                (
                    SELECT t.user_text
                    FROM voice_turns t
                    WHERE t.session_id = s.id
                    ORDER BY t.created_at DESC, t.id DESC
                    LIMIT 1
                ) AS last_user_text,
                (
                    SELECT t.assistant_text
                    FROM voice_turns t
                    WHERE t.session_id = s.id
                    ORDER BY t.created_at DESC, t.id DESC
                    LIMIT 1
                ) AS last_assistant_text
            FROM voice_sessions s
            WHERE s.id = ?1
            "#,
            params![session_id],
            map_session_summary,
        )
        .optional()?;
    Ok(session)
}

fn fetch_turn_record(conn: &Connection, turn_id: &str) -> anyhow::Result<Option<VoiceTurnRecord>> {
    let turn = conn
        .query_row(
            r#"
            SELECT
                id,
                session_id,
                utterance_id,
                utterance_seq,
                mode,
                status,
                status_reason,
                vad_end_reason,
                user_text,
                assistant_text,
                assistant_raw_text,
                language,
                audio_duration_secs,
                asr_model_id,
                text_model_id,
                tts_model_id,
                s2s_model_id,
                speaker,
                created_at,
                updated_at
            FROM voice_turns
            WHERE id = ?1
            "#,
            params![turn_id],
            map_turn_record,
        )
        .optional()?;
    Ok(turn)
}

fn fetch_turn_session_id(conn: &Connection, turn_id: &str) -> anyhow::Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT session_id FROM voice_turns WHERE id = ?1",
            params![turn_id],
            |row| row.get(0),
        )
        .optional()?)
}

fn touch_session_updated_at(
    conn: &Connection,
    session_id: &str,
    updated_at: i64,
) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE voice_sessions SET updated_at = ?1 WHERE id = ?2",
        params![updated_at, session_id],
    )?;
    Ok(())
}

fn map_voice_profile(row: &Row<'_>) -> rusqlite::Result<VoiceProfile> {
    Ok(VoiceProfile {
        id: row.get(0)?,
        name: row.get(1)?,
        system_prompt: row.get(2)?,
        observational_memory_enabled: i64_to_bool(row.get(3)?),
        created_at: i64_to_u64(row.get(4)?),
        updated_at: i64_to_u64(row.get(5)?),
    })
}

fn map_session_summary(row: &Row<'_>) -> rusqlite::Result<VoiceSessionSummary> {
    let turn_count_raw: i64 = row.get(7)?;
    Ok(VoiceSessionSummary {
        id: row.get(0)?,
        profile_id: row.get(1)?,
        mode: row.get(2)?,
        system_prompt: row.get(3)?,
        created_at: i64_to_u64(row.get(4)?),
        updated_at: i64_to_u64(row.get(5)?),
        ended_at: row.get::<_, Option<i64>>(6)?.map(i64_to_u64),
        turn_count: i64_to_usize(turn_count_raw),
        last_user_text: row.get(8)?,
        last_assistant_text: row.get(9)?,
    })
}

fn map_turn_record(row: &Row<'_>) -> rusqlite::Result<VoiceTurnRecord> {
    let utterance_seq: i64 = row.get(3)?;
    Ok(VoiceTurnRecord {
        id: row.get(0)?,
        session_id: row.get(1)?,
        utterance_id: row.get(2)?,
        utterance_seq: i64_to_u64(utterance_seq),
        mode: row.get(4)?,
        status: row.get(5)?,
        status_reason: row.get(6)?,
        vad_end_reason: row.get(7)?,
        user_text: row.get(8)?,
        assistant_text: row.get(9)?,
        assistant_raw_text: row.get(10)?,
        language: row.get(11)?,
        audio_duration_secs: row.get::<_, Option<f64>>(12)?.map(|value| value as f32),
        asr_model_id: row.get(13)?,
        text_model_id: row.get(14)?,
        tts_model_id: row.get(15)?,
        s2s_model_id: row.get(16)?,
        speaker: row.get(17)?,
        created_at: i64_to_u64(row.get(18)?),
        updated_at: i64_to_u64(row.get(19)?),
    })
}

fn sanitize_mode(raw: &str) -> anyhow::Result<String> {
    match raw.trim() {
        "modular" => Ok("modular".to_string()),
        "unified" => Ok("unified".to_string()),
        other => Err(anyhow!("Invalid voice mode '{other}'")),
    }
}

fn sanitize_turn_status(raw: &str) -> anyhow::Result<String> {
    match raw.trim() {
        "processing" | "ok" | "error" | "timeout" | "interrupted" | "no_input" => {
            Ok(raw.trim().to_string())
        }
        other => Err(anyhow!("Invalid voice turn status '{other}'")),
    }
}

fn sanitize_profile_name(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(120).collect())
}

fn sanitize_prompt(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(4000).collect())
}

fn sanitize_required_text(raw: &str, max_len: usize) -> anyhow::Result<String> {
    sanitize_optional_text(Some(raw), max_len).ok_or_else(|| anyhow!("Missing required text value"))
}

fn sanitize_optional_text(raw: Option<&str>, max_len: usize) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.chars().take(max_len).collect())
}

fn bool_to_i64(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

fn i64_to_bool(value: i64) -> bool {
    value != 0
}

fn i64_to_u64(value: i64) -> u64 {
    value.max(0) as u64
}

fn i64_to_usize(value: i64) -> usize {
    value.max(0) as usize
}

fn u64_to_i64(value: u64) -> anyhow::Result<i64> {
    i64::try_from(value).map_err(|_| anyhow!("Value exceeds i64"))
}

fn now_unix_millis_i64() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock")
    }

    #[tokio::test]
    async fn initializes_default_profile_and_persists_sessions() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("voice-store.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let store = VoiceStore::initialize().expect("store");
        let profile = store.get_default_profile().await.expect("profile");
        assert_eq!(profile.id, DEFAULT_VOICE_PROFILE_ID);

        let session = store
            .create_session(CreateVoiceSessionRequest {
                profile_id: profile.id.clone(),
                mode: "modular".to_string(),
                system_prompt: profile.system_prompt.clone(),
            })
            .await
            .expect("session");

        let turn = store
            .create_turn(CreateVoiceTurnRequest {
                session_id: session.id.clone(),
                utterance_id: "utt-1".to_string(),
                utterance_seq: 1,
                mode: "modular".to_string(),
                vad_end_reason: Some("silence".to_string()),
                asr_model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
                text_model_id: Some("Qwen3-1.7B-GGUF".to_string()),
                tts_model_id: Some("Kokoro-82M".to_string()),
                s2s_model_id: None,
                speaker: Some("Serena".to_string()),
            })
            .await
            .expect("turn");

        store
            .update_turn_transcript(
                turn.id.clone(),
                "Hello there".to_string(),
                Some("en".to_string()),
                Some(1.2),
            )
            .await
            .expect("transcript");
        store
            .update_turn_assistant(
                turn.id.clone(),
                Some("Hi!".to_string()),
                Some("Hi!".to_string()),
            )
            .await
            .expect("assistant");
        store
            .complete_turn(turn.id.clone(), "ok", None)
            .await
            .expect("complete");
        store
            .end_session(session.id.clone())
            .await
            .expect("end session");

        let fetched = store
            .get_session(session.id.clone())
            .await
            .expect("get session")
            .expect("session detail");
        assert_eq!(fetched.turns.len(), 1);
        assert_eq!(fetched.turns[0].status, "ok");
        assert_eq!(fetched.turns[0].user_text.as_deref(), Some("Hello there"));
        assert_eq!(fetched.turns[0].assistant_text.as_deref(), Some("Hi!"));
        assert!(fetched.session.ended_at.is_some());

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn updates_default_profile_prompt() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("voice-profile.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let store = VoiceStore::initialize().expect("store");
        let updated = store
            .update_default_profile(
                None,
                Some("Keep answers ultra concise.".to_string()),
                Some(false),
            )
            .await
            .expect("profile update");

        assert_eq!(updated.system_prompt, "Keep answers ultra concise.");
        assert!(!updated.observational_memory_enabled);

        let fetched = store.get_default_profile().await.expect("profile");
        assert_eq!(fetched.system_prompt, "Keep answers ultra concise.");
        assert!(!fetched.observational_memory_enabled);

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }
}

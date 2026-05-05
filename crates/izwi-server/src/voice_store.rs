use anyhow::{Context, anyhow};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait, QueryFilter,
    QueryResult, Set, Statement,
};
use serde::Serialize;

use crate::db::StoreDatabase;
use crate::entity::{voice_profiles, voice_sessions, voice_turns};
use crate::ids::new_uuid;
use crate::voice_defaults::{DEFAULT_VOICE_AGENT_SYSTEM_PROMPT, DEFAULT_VOICE_PROFILE_ID};

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
    db: StoreDatabase,
}

impl VoiceStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
        })
    }

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self { db }
    }

    pub async fn get_default_profile(&self) -> anyhow::Result<VoiceProfile> {
        self.get_profile(DEFAULT_VOICE_PROFILE_ID.to_string())
            .await?
            .ok_or_else(|| anyhow!("Default voice profile not found"))
    }

    pub async fn get_profile(&self, profile_id: String) -> anyhow::Result<Option<VoiceProfile>> {
        let db = self.db.connection().await?;
        fetch_voice_profile(db, &profile_id).await
    }

    pub async fn update_default_profile(
        &self,
        name: Option<String>,
        system_prompt: Option<String>,
        observational_memory_enabled: Option<bool>,
    ) -> anyhow::Result<VoiceProfile> {
        let db = self.db.connection().await?;
        let current = fetch_voice_profile(db, DEFAULT_VOICE_PROFILE_ID)
            .await?
            .context("Default voice profile not found")?;
        let next_name = sanitize_profile_name(name.as_deref()).unwrap_or(current.name);
        let next_prompt =
            sanitize_prompt(system_prompt.as_deref()).unwrap_or(current.system_prompt);
        let next_memory_enabled =
            observational_memory_enabled.unwrap_or(current.observational_memory_enabled);
        let now = now_unix_millis_i64();

        voice_profiles::Entity::update_many()
            .col_expr(voice_profiles::Column::Name, Expr::value(next_name))
            .col_expr(
                voice_profiles::Column::SystemPrompt,
                Expr::value(next_prompt),
            )
            .col_expr(
                voice_profiles::Column::ObservationalMemoryEnabled,
                Expr::value(bool_to_i64(next_memory_enabled)),
            )
            .col_expr(voice_profiles::Column::UpdatedAt, Expr::value(now))
            .filter(voice_profiles::Column::Id.eq(DEFAULT_VOICE_PROFILE_ID))
            .exec(db)
            .await
            .context("Failed to update default voice profile")?;

        fetch_voice_profile(db, DEFAULT_VOICE_PROFILE_ID)
            .await?
            .context("Updated voice profile not found")
    }

    pub async fn create_session(
        &self,
        request: CreateVoiceSessionRequest,
    ) -> anyhow::Result<VoiceSessionSummary> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let session_id = new_uuid();
        let mode = sanitize_mode(request.mode.as_str())?;
        let system_prompt = sanitize_prompt(Some(request.system_prompt.as_str()))
            .unwrap_or_else(|| DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.to_string());

        voice_sessions::Entity::insert(voice_sessions::ActiveModel {
            id: Set(session_id.clone()),
            profile_id: Set(request.profile_id),
            mode: Set(mode),
            system_prompt: Set(system_prompt),
            created_at: Set(now),
            updated_at: Set(now),
            ended_at: Set(None),
        })
        .exec(db)
        .await
        .context("Failed to create voice session")?;

        fetch_session_summary(db, &session_id)
            .await?
            .ok_or_else(|| anyhow!("Created voice session not found"))
    }

    pub async fn end_session(&self, session_id: String) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        db.execute_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            r#"
            UPDATE voice_sessions
            SET updated_at = ?1,
                ended_at = COALESCE(ended_at, ?1)
            WHERE id = ?2
            "#,
            vec![now.into(), session_id.into()],
        ))
        .await
        .context("Failed to end voice session")?;
        Ok(())
    }

    pub async fn list_sessions(&self, limit: usize) -> anyhow::Result<Vec<VoiceSessionSummary>> {
        let db = self.db.connection().await?;
        let limit = i64::try_from(limit.max(1)).context("Voice session limit exceeds i64")?;
        let rows = db
            .query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
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
                vec![limit.into()],
            ))
            .await
            .context("Failed to list voice sessions")?;
        rows.iter().map(map_session_summary).collect()
    }

    pub async fn get_session(
        &self,
        session_id: String,
    ) -> anyhow::Result<Option<VoiceSessionDetail>> {
        let db = self.db.connection().await?;
        let Some(session) = fetch_session_summary(db, &session_id).await? else {
            return Ok(None);
        };
        let turns = list_session_turns(db, &session_id).await?;
        Ok(Some(VoiceSessionDetail { session, turns }))
    }

    pub async fn create_turn(
        &self,
        request: CreateVoiceTurnRequest,
    ) -> anyhow::Result<VoiceTurnRecord> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let turn_id = new_uuid();
        let mode = sanitize_mode(request.mode.as_str())?;
        let session_id = request.session_id;

        voice_turns::Entity::insert(voice_turns::ActiveModel {
            id: Set(turn_id.clone()),
            session_id: Set(session_id.clone()),
            utterance_id: Set(sanitize_required_text(request.utterance_id.as_str(), 160)?),
            utterance_seq: Set(u64_to_i64(request.utterance_seq)?),
            mode: Set(mode),
            status: Set("processing".to_string()),
            status_reason: Set(None),
            vad_end_reason: Set(sanitize_optional_text(
                request.vad_end_reason.as_deref(),
                48,
            )),
            user_text: Set(None),
            assistant_text: Set(None),
            assistant_raw_text: Set(None),
            language: Set(None),
            audio_duration_secs: Set(None),
            asr_model_id: Set(sanitize_optional_text(request.asr_model_id.as_deref(), 160)),
            text_model_id: Set(sanitize_optional_text(
                request.text_model_id.as_deref(),
                160,
            )),
            tts_model_id: Set(sanitize_optional_text(request.tts_model_id.as_deref(), 160)),
            s2s_model_id: Set(sanitize_optional_text(request.s2s_model_id.as_deref(), 160)),
            speaker: Set(sanitize_optional_text(request.speaker.as_deref(), 160)),
            created_at: Set(now),
            updated_at: Set(now),
        })
        .exec(db)
        .await
        .context("Failed to create voice turn")?;

        touch_session_updated_at(db, session_id.as_str(), now).await?;
        fetch_turn_record(db, &turn_id)
            .await?
            .ok_or_else(|| anyhow!("Created voice turn not found"))
    }

    pub async fn update_turn_transcript(
        &self,
        turn_id: String,
        user_text: String,
        language: Option<String>,
        audio_duration_secs: Option<f32>,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let session_id = fetch_turn_session_id(db, &turn_id)
            .await?
            .ok_or_else(|| anyhow!("Voice turn not found"))?;
        db.execute_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            r#"
            UPDATE voice_turns
            SET user_text = ?1,
                language = ?2,
                audio_duration_secs = ?3,
                updated_at = ?4
            WHERE id = ?5
            "#,
            vec![
                sanitize_optional_text(Some(user_text.as_str()), 16000).into(),
                sanitize_optional_text(language.as_deref(), 64).into(),
                audio_duration_secs.map(f64::from).into(),
                now.into(),
                turn_id.into(),
            ],
        ))
        .await
        .context("Failed to update voice turn transcript")?;
        touch_session_updated_at(db, session_id.as_str(), now).await
    }

    pub async fn update_turn_assistant(
        &self,
        turn_id: String,
        assistant_text: Option<String>,
        assistant_raw_text: Option<String>,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let session_id = fetch_turn_session_id(db, &turn_id)
            .await?
            .ok_or_else(|| anyhow!("Voice turn not found"))?;
        db.execute_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            r#"
            UPDATE voice_turns
            SET assistant_text = ?1,
                assistant_raw_text = ?2,
                updated_at = ?3
            WHERE id = ?4
            "#,
            vec![
                sanitize_optional_text(assistant_text.as_deref(), 16000).into(),
                sanitize_optional_text(assistant_raw_text.as_deref(), 16000).into(),
                now.into(),
                turn_id.into(),
            ],
        ))
        .await
        .context("Failed to update voice turn assistant text")?;
        touch_session_updated_at(db, session_id.as_str(), now).await
    }

    pub async fn complete_turn(
        &self,
        turn_id: String,
        status: &str,
        status_reason: Option<String>,
    ) -> anyhow::Result<()> {
        let status = sanitize_turn_status(status)?;
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let session_id = fetch_turn_session_id(db, &turn_id)
            .await?
            .ok_or_else(|| anyhow!("Voice turn not found"))?;
        db.execute_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            r#"
            UPDATE voice_turns
            SET status = ?1,
                status_reason = ?2,
                updated_at = ?3
            WHERE id = ?4
            "#,
            vec![
                status.into(),
                sanitize_optional_text(status_reason.as_deref(), 160).into(),
                now.into(),
                turn_id.into(),
            ],
        ))
        .await
        .context("Failed to complete voice turn")?;
        touch_session_updated_at(db, session_id.as_str(), now).await
    }
}

const VOICE_PROFILE_BY_ID_SQL: &str = r#"
    SELECT
        id,
        name,
        system_prompt,
        observational_memory_enabled,
        created_at,
        updated_at
    FROM voice_profiles
    WHERE id = ?1
"#;

const SESSION_SUMMARY_BY_ID_SQL: &str = r#"
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
"#;

const TURN_RECORD_BY_ID_SQL: &str = r#"
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
"#;

const SESSION_TURNS_SQL: &str = r#"
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
"#;

async fn fetch_voice_profile(
    db: &DatabaseConnection,
    profile_id: &str,
) -> anyhow::Result<Option<VoiceProfile>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            VOICE_PROFILE_BY_ID_SQL,
            vec![profile_id.into()],
        ))
        .await
        .context("Failed to load voice profile")?;
    row.as_ref().map(map_voice_profile).transpose()
}

async fn fetch_session_summary(
    db: &DatabaseConnection,
    session_id: &str,
) -> anyhow::Result<Option<VoiceSessionSummary>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            SESSION_SUMMARY_BY_ID_SQL,
            vec![session_id.into()],
        ))
        .await
        .context("Failed to load voice session summary")?;
    row.as_ref().map(map_session_summary).transpose()
}

async fn list_session_turns(
    db: &DatabaseConnection,
    session_id: &str,
) -> anyhow::Result<Vec<VoiceTurnRecord>> {
    let rows = db
        .query_all_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            SESSION_TURNS_SQL,
            vec![session_id.into()],
        ))
        .await
        .context("Failed to list voice turns")?;
    rows.iter().map(map_turn_record).collect()
}

async fn fetch_turn_record(
    db: &DatabaseConnection,
    turn_id: &str,
) -> anyhow::Result<Option<VoiceTurnRecord>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            TURN_RECORD_BY_ID_SQL,
            vec![turn_id.into()],
        ))
        .await
        .context("Failed to load voice turn")?;
    row.as_ref().map(map_turn_record).transpose()
}

async fn fetch_turn_session_id(
    db: &DatabaseConnection,
    turn_id: &str,
) -> anyhow::Result<Option<String>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            "SELECT session_id FROM voice_turns WHERE id = ?1",
            vec![turn_id.into()],
        ))
        .await
        .context("Failed to load voice turn session id")?;
    row.map(|row| row.try_get_by_index(0))
        .transpose()
        .map_err(Into::into)
}

async fn touch_session_updated_at(
    db: &DatabaseConnection,
    session_id: &str,
    updated_at: i64,
) -> anyhow::Result<()> {
    db.execute_raw(Statement::from_sql_and_values(
        DbBackend::Sqlite,
        "UPDATE voice_sessions SET updated_at = ?1 WHERE id = ?2",
        vec![updated_at.into(), session_id.into()],
    ))
    .await
    .context("Failed to touch voice session")?;
    Ok(())
}

fn map_voice_profile(row: &QueryResult) -> anyhow::Result<VoiceProfile> {
    Ok(VoiceProfile {
        id: row.try_get_by_index(0)?,
        name: row.try_get_by_index(1)?,
        system_prompt: row.try_get_by_index(2)?,
        observational_memory_enabled: i64_to_bool(row.try_get_by_index(3)?),
        created_at: i64_to_u64(row.try_get_by_index(4)?),
        updated_at: i64_to_u64(row.try_get_by_index(5)?),
    })
}

fn map_session_summary(row: &QueryResult) -> anyhow::Result<VoiceSessionSummary> {
    let turn_count_raw: i64 = row.try_get_by_index(7)?;
    Ok(VoiceSessionSummary {
        id: row.try_get_by_index(0)?,
        profile_id: row.try_get_by_index(1)?,
        mode: row.try_get_by_index(2)?,
        system_prompt: row.try_get_by_index(3)?,
        created_at: i64_to_u64(row.try_get_by_index(4)?),
        updated_at: i64_to_u64(row.try_get_by_index(5)?),
        ended_at: row.try_get_by_index::<Option<i64>>(6)?.map(i64_to_u64),
        turn_count: i64_to_usize(turn_count_raw),
        last_user_text: row.try_get_by_index(8)?,
        last_assistant_text: row.try_get_by_index(9)?,
    })
}

fn map_turn_record(row: &QueryResult) -> anyhow::Result<VoiceTurnRecord> {
    let utterance_seq: i64 = row.try_get_by_index(3)?;
    Ok(VoiceTurnRecord {
        id: row.try_get_by_index(0)?,
        session_id: row.try_get_by_index(1)?,
        utterance_id: row.try_get_by_index(2)?,
        utterance_seq: i64_to_u64(utterance_seq),
        mode: row.try_get_by_index(4)?,
        status: row.try_get_by_index(5)?,
        status_reason: row.try_get_by_index(6)?,
        vad_end_reason: row.try_get_by_index(7)?,
        user_text: row.try_get_by_index(8)?,
        assistant_text: row.try_get_by_index(9)?,
        assistant_raw_text: row.try_get_by_index(10)?,
        language: row.try_get_by_index(11)?,
        audio_duration_secs: row
            .try_get_by_index::<Option<f64>>(12)?
            .map(|value| value as f32),
        asr_model_id: row.try_get_by_index(13)?,
        text_model_id: row.try_get_by_index(14)?,
        tts_model_id: row.try_get_by_index(15)?,
        s2s_model_id: row.try_get_by_index(16)?,
        speaker: row.try_get_by_index(17)?,
        created_at: i64_to_u64(row.try_get_by_index(18)?),
        updated_at: i64_to_u64(row.try_get_by_index(19)?),
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
    if value { 1 } else { 0 }
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
    use crate::test_support::env_lock;

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

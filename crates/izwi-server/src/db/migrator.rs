use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_SYSTEM_PROMPT, DEFAULT_VOICE_PROFILE_ID, DEFAULT_VOICE_PROFILE_NAME,
};
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Migrator;

impl Migrator {
    pub async fn up(db: &DatabaseConnection) -> anyhow::Result<()> {
        for statement in BASELINE_SCHEMA {
            db.execute_unprepared(statement).await?;
        }

        for column in COMPATIBILITY_COLUMNS {
            ensure_column(db, column).await?;
        }

        ensure_default_voice_profile(db).await?;
        Ok(())
    }
}

struct CompatibilityColumn {
    table: &'static str,
    column: &'static str,
    definition: &'static str,
}

const BASELINE_SCHEMA: &[&str] = &[
    r#"
    CREATE TABLE IF NOT EXISTS chat_threads (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        model_id TEXT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
        content TEXT NOT NULL,
        content_parts TEXT NULL,
        created_at INTEGER NOT NULL,
        tokens_generated INTEGER NULL,
        generation_time_ms REAL NULL,
        FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_chat_threads_updated_at ON chat_threads(updated_at DESC, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created_at ON chat_messages(thread_id, created_at, id);",
    r#"
    CREATE TABLE IF NOT EXISTS voice_profiles (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        system_prompt TEXT NOT NULL,
        observational_memory_enabled INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    );
    "#,
    r#"
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
    "#,
    r#"
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
    "#,
    "CREATE INDEX IF NOT EXISTS idx_voice_sessions_updated_at ON voice_sessions(updated_at DESC, created_at DESC);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_turns_session_utterance ON voice_turns(session_id, utterance_seq);",
    "CREATE INDEX IF NOT EXISTS idx_voice_turns_session_created_at ON voice_turns(session_id, created_at ASC, id ASC);",
    r#"
    CREATE TABLE IF NOT EXISTS voice_observations (
        id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        category TEXT NOT NULL,
        summary TEXT NOT NULL,
        canonical_summary TEXT NOT NULL,
        confidence REAL NOT NULL,
        source_turn_id TEXT NULL,
        source_user_text TEXT NULL,
        source_assistant_text TEXT NULL,
        times_seen INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        forgotten_at INTEGER NULL,
        FOREIGN KEY(profile_id) REFERENCES voice_profiles(id) ON DELETE CASCADE
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_voice_observations_profile_updated_at ON voice_observations(profile_id, updated_at DESC, created_at DESC);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_voice_observations_profile_canonical_active ON voice_observations(profile_id, canonical_summary) WHERE forgotten_at IS NULL;",
    r#"
    CREATE TABLE IF NOT EXISTS onboarding_state (
        id TEXT PRIMARY KEY,
        completed_at INTEGER NULL,
        analytics_opt_in INTEGER NOT NULL DEFAULT 0
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS transcription_records (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        model_id TEXT NULL,
        aligner_model_id TEXT NULL,
        language TEXT NULL,
        processing_status TEXT NOT NULL DEFAULT 'ready',
        processing_error TEXT NULL,
        duration_secs REAL NULL,
        processing_time_ms REAL NOT NULL,
        rtf REAL NULL,
        audio_mime_type TEXT NOT NULL,
        audio_filename TEXT NULL,
        audio_storage_path TEXT NOT NULL,
        transcription TEXT NOT NULL,
        segments_json TEXT NOT NULL DEFAULT '[]',
        words_json TEXT NOT NULL DEFAULT '[]',
        summary_status TEXT NOT NULL DEFAULT 'not_requested',
        summary_model_id TEXT NULL,
        summary_text TEXT NULL,
        summary_error TEXT NULL,
        summary_updated_at INTEGER NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_transcription_records_created_at ON transcription_records(created_at DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS diarization_records (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        model_id TEXT NULL,
        asr_model_id TEXT NULL,
        aligner_model_id TEXT NULL,
        llm_model_id TEXT NULL,
        processing_status TEXT NOT NULL DEFAULT 'ready',
        processing_error TEXT NULL,
        min_speakers INTEGER NULL,
        max_speakers INTEGER NULL,
        min_speech_duration_ms REAL NULL,
        min_silence_duration_ms REAL NULL,
        enable_llm_refinement INTEGER NOT NULL,
        processing_time_ms REAL NOT NULL,
        duration_secs REAL NULL,
        rtf REAL NULL,
        speaker_count INTEGER NOT NULL,
        alignment_coverage REAL NULL,
        unattributed_words INTEGER NOT NULL,
        llm_refined INTEGER NOT NULL,
        asr_text TEXT NOT NULL,
        raw_transcript TEXT NOT NULL,
        transcript TEXT NOT NULL,
        summary_status TEXT NOT NULL DEFAULT 'not_requested',
        summary_model_id TEXT NULL,
        summary_text TEXT NULL,
        summary_error TEXT NULL,
        summary_updated_at INTEGER NULL,
        segments_json TEXT NOT NULL,
        words_json TEXT NOT NULL,
        utterances_json TEXT NOT NULL,
        speaker_name_overrides_json TEXT NOT NULL DEFAULT '{}',
        diarization_diagnostics_json TEXT NULL,
        audio_mime_type TEXT NOT NULL,
        audio_filename TEXT NULL,
        audio_storage_path TEXT NOT NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_diarization_records_created_at ON diarization_records(created_at DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS speech_history_records (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        route_kind TEXT NOT NULL,
        processing_status TEXT NOT NULL DEFAULT 'ready',
        processing_error TEXT NULL,
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
    "#,
    "CREATE INDEX IF NOT EXISTS idx_speech_history_route_created_at ON speech_history_records(route_kind, created_at DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS saved_voices (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        name TEXT NOT NULL COLLATE NOCASE,
        reference_text TEXT NOT NULL,
        audio_mime_type TEXT NOT NULL,
        audio_filename TEXT NULL,
        audio_storage_path TEXT NOT NULL,
        source_route_kind TEXT NULL,
        source_record_id TEXT NULL
    );
    "#,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_voices_name_nocase ON saved_voices(name COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_saved_voices_updated_at ON saved_voices(updated_at DESC, created_at DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS studio_projects (
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
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_segments (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        text TEXT NOT NULL,
        model_id TEXT NULL,
        voice_mode TEXT NULL,
        speaker TEXT NULL,
        saved_voice_id TEXT NULL,
        speech_record_id TEXT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_folders (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        name TEXT NOT NULL,
        parent_id TEXT NULL,
        sort_order INTEGER NOT NULL DEFAULT 0
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_meta (
        project_id TEXT PRIMARY KEY,
        folder_id TEXT NULL,
        tags_json TEXT NOT NULL DEFAULT '[]',
        default_export_format TEXT NOT NULL DEFAULT 'wav',
        last_render_job_id TEXT NULL,
        last_rendered_at INTEGER NULL,
        FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE,
        FOREIGN KEY(folder_id) REFERENCES studio_project_folders(id) ON DELETE SET NULL
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_pronunciations (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        source_text TEXT NOT NULL,
        replacement_text TEXT NOT NULL,
        locale TEXT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_snapshots (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        label TEXT NULL,
        project_json TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS studio_project_render_jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT NULL,
        queued_segment_ids_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_studio_projects_updated_at ON studio_projects(updated_at DESC, id DESC);",
    "CREATE INDEX IF NOT EXISTS idx_studio_project_segments_project_position ON studio_project_segments(project_id, position ASC);",
    "CREATE INDEX IF NOT EXISTS idx_studio_project_folders_parent ON studio_project_folders(parent_id, sort_order ASC, updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_studio_project_pronunciations_project ON studio_project_pronunciations(project_id, updated_at DESC, id DESC);",
    "CREATE INDEX IF NOT EXISTS idx_studio_project_snapshots_project ON studio_project_snapshots(project_id, created_at DESC, id DESC);",
    "CREATE INDEX IF NOT EXISTS idx_studio_project_render_jobs_project ON studio_project_render_jobs(project_id, created_at DESC, id DESC);",
];

const COMPATIBILITY_COLUMNS: &[CompatibilityColumn] = &[
    CompatibilityColumn {
        table: "chat_messages",
        column: "content_parts",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "onboarding_state",
        column: "analytics_opt_in",
        definition: "INTEGER NOT NULL DEFAULT 0",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "aligner_model_id",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "processing_status",
        definition: "TEXT NOT NULL DEFAULT 'ready'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "processing_error",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "segments_json",
        definition: "TEXT NOT NULL DEFAULT '[]'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "words_json",
        definition: "TEXT NOT NULL DEFAULT '[]'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "summary_status",
        definition: "TEXT NOT NULL DEFAULT 'not_requested'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "summary_model_id",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "summary_text",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "summary_error",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "summary_updated_at",
        definition: "INTEGER NULL",
    },
    CompatibilityColumn {
        table: "speech_history_records",
        column: "saved_voice_id",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "speech_history_records",
        column: "speed",
        definition: "REAL NULL",
    },
    CompatibilityColumn {
        table: "speech_history_records",
        column: "processing_status",
        definition: "TEXT NOT NULL DEFAULT 'ready'",
    },
    CompatibilityColumn {
        table: "speech_history_records",
        column: "processing_error",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "speaker_name_overrides_json",
        definition: "TEXT NOT NULL DEFAULT '{}'",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "diarization_diagnostics_json",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "processing_status",
        definition: "TEXT NOT NULL DEFAULT 'ready'",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "processing_error",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "summary_status",
        definition: "TEXT NOT NULL DEFAULT 'not_requested'",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "summary_model_id",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "summary_text",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "summary_error",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "diarization_records",
        column: "summary_updated_at",
        definition: "INTEGER NULL",
    },
    CompatibilityColumn {
        table: "studio_project_segments",
        column: "model_id",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "studio_project_segments",
        column: "voice_mode",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "studio_project_segments",
        column: "speaker",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "studio_project_segments",
        column: "saved_voice_id",
        definition: "TEXT NULL",
    },
];

async fn ensure_column(
    db: &DatabaseConnection,
    column: &CompatibilityColumn,
) -> anyhow::Result<()> {
    if table_has_column(db, column.table, column.column).await? {
        return Ok(());
    }
    db.execute_unprepared(&format!(
        "ALTER TABLE {} ADD COLUMN {} {}",
        column.table, column.column, column.definition
    ))
    .await?;
    Ok(())
}

async fn table_has_column(
    db: &DatabaseConnection,
    table: &str,
    target: &str,
) -> anyhow::Result<bool> {
    let rows = db
        .query_all_raw(Statement::from_string(
            DbBackend::Sqlite,
            format!("PRAGMA table_info({table})"),
        ))
        .await?;
    for row in rows {
        let name: String = row.try_get_by_index(1)?;
        if name == target {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn ensure_default_voice_profile(db: &DatabaseConnection) -> anyhow::Result<()> {
    let exists = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            "SELECT 1 FROM voice_profiles WHERE id = ?1 LIMIT 1",
            vec![DEFAULT_VOICE_PROFILE_ID.into()],
        ))
        .await?
        .is_some();
    if exists {
        return Ok(());
    }

    let now = current_timestamp_millis();
    db.execute_raw(Statement::from_sql_and_values(
        DbBackend::Sqlite,
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
        vec![
            DEFAULT_VOICE_PROFILE_ID.into(),
            DEFAULT_VOICE_PROFILE_NAME.into(),
            DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.into(),
            now.into(),
        ],
    ))
    .await?;

    Ok(())
}

fn current_timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

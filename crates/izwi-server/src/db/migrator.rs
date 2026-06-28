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
        transcription_mode TEXT NOT NULL DEFAULT 'transcription',
        model_id TEXT NULL,
        aligner_model_id TEXT NULL,
        language TEXT NULL,
        processing_status TEXT NOT NULL DEFAULT 'ready',
        processing_error TEXT NULL,
        processing_progress_json TEXT NULL,
        duration_secs REAL NULL,
        processing_time_ms REAL NOT NULL,
        rtf REAL NULL,
        audio_mime_type TEXT NOT NULL,
        audio_filename TEXT NULL,
        audio_storage_path TEXT NOT NULL,
        transcription TEXT NOT NULL,
        segments_json TEXT NOT NULL DEFAULT '[]',
        words_json TEXT NOT NULL DEFAULT '[]',
        speaker_attributed_text TEXT NULL,
        speaker_turns_json TEXT NOT NULL DEFAULT '[]',
        saa_status TEXT NOT NULL DEFAULT 'not_requested',
        saa_warnings_json TEXT NOT NULL DEFAULT '[]',
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
    r#"
    CREATE TABLE IF NOT EXISTS media_assets (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        asset_kind TEXT NOT NULL,
        storage_namespace TEXT NOT NULL,
        storage_key TEXT NOT NULL,
        content_type TEXT NOT NULL,
        filename TEXT NULL,
        size_bytes INTEGER NOT NULL,
        sha256 TEXT NULL,
        duration_secs REAL NULL,
        sample_rate_hz INTEGER NULL,
        channel_count INTEGER NULL,
        peak_amplitude REAL NULL,
        rms_amplitude REAL NULL,
        scan_status TEXT NOT NULL DEFAULT 'not_scanned',
        retention_policy TEXT NOT NULL DEFAULT 'default',
        deleted_at INTEGER NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    "#,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_storage_key ON media_assets(storage_key);",
    "CREATE INDEX IF NOT EXISTS idx_media_assets_created_at ON media_assets(created_at DESC, id DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS text_assets (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        raw_text TEXT NOT NULL,
        normalized_text TEXT NOT NULL,
        language_hint TEXT NULL,
        character_count INTEGER NOT NULL,
        sha256 TEXT NULL,
        safety_status TEXT NOT NULL DEFAULT 'unchecked',
        retention_policy TEXT NOT NULL DEFAULT 'default',
        structure_json TEXT NOT NULL DEFAULT '{}'
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_text_assets_created_at ON text_assets(created_at DESC, id DESC);",
    r#"
    CREATE TABLE IF NOT EXISTS runtime_jobs (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        queued_at INTEGER NULL,
        started_at INTEGER NULL,
        finished_at INTEGER NULL,
        job_kind TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing', 'completed', 'failed', 'cancelled', 'expired')),
        priority INTEGER NOT NULL DEFAULT 0,
        model_id TEXT NULL,
        capability TEXT NULL,
        route_record_kind TEXT NULL,
        route_record_id TEXT NULL,
        input_media_asset_id TEXT NULL,
        input_text_asset_id TEXT NULL,
        request_json TEXT NOT NULL DEFAULT '{}',
        model_snapshot_json TEXT NOT NULL DEFAULT '{}',
        progress_json TEXT NULL,
        error_code TEXT NULL,
        error_message TEXT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 1,
        retry_policy_json TEXT NOT NULL DEFAULT '{}',
        idempotency_key TEXT NULL,
        correlation_id TEXT NULL,
        cancellation_reason TEXT NULL,
        FOREIGN KEY(input_media_asset_id) REFERENCES media_assets(id) ON DELETE SET NULL,
        FOREIGN KEY(input_text_asset_id) REFERENCES text_assets(id) ON DELETE SET NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_runtime_jobs_status_priority ON runtime_jobs(status, priority DESC, created_at ASC);",
    "CREATE INDEX IF NOT EXISTS idx_runtime_jobs_route_record ON runtime_jobs(route_record_kind, route_record_id);",
    "CREATE INDEX IF NOT EXISTS idx_runtime_jobs_idempotency_key ON runtime_jobs(idempotency_key);",
    r#"
    CREATE TABLE IF NOT EXISTS job_stages (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        sequence INTEGER NOT NULL,
        stage_kind TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing', 'completed', 'failed', 'cancelled', 'expired', 'skipped')),
        capability TEXT NULL,
        model_id TEXT NULL,
        worker_id TEXT NULL,
        lease_expires_at INTEGER NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 1,
        input_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
        output_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
        progress_json TEXT NULL,
        started_at INTEGER NULL,
        finished_at INTEGER NULL,
        error_code TEXT NULL,
        error_message TEXT NULL,
        FOREIGN KEY(job_id) REFERENCES runtime_jobs(id) ON DELETE CASCADE
    );
    "#,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_job_stages_job_sequence ON job_stages(job_id, sequence);",
    "CREATE INDEX IF NOT EXISTS idx_job_stages_status_lease ON job_stages(status, lease_expires_at, sequence);",
    r#"
    CREATE TABLE IF NOT EXISTS runtime_artifacts (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        stage_id TEXT NULL,
        created_at INTEGER NOT NULL,
        artifact_kind TEXT NOT NULL,
        artifact_role TEXT NOT NULL,
        media_asset_id TEXT NULL,
        text_asset_id TEXT NULL,
        storage_key TEXT NULL,
        content_type TEXT NULL,
        filename TEXT NULL,
        size_bytes INTEGER NULL,
        sha256 TEXT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        retention_policy TEXT NOT NULL DEFAULT 'default',
        FOREIGN KEY(job_id) REFERENCES runtime_jobs(id) ON DELETE CASCADE,
        FOREIGN KEY(stage_id) REFERENCES job_stages(id) ON DELETE SET NULL,
        FOREIGN KEY(media_asset_id) REFERENCES media_assets(id) ON DELETE SET NULL,
        FOREIGN KEY(text_asset_id) REFERENCES text_assets(id) ON DELETE SET NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_job_role ON runtime_artifacts(job_id, artifact_role, created_at ASC);",
    "CREATE INDEX IF NOT EXISTS idx_runtime_artifacts_stage ON runtime_artifacts(stage_id);",
    r#"
    CREATE TABLE IF NOT EXISTS idempotency_keys (
        operation TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        expires_at INTEGER NULL,
        request_hash TEXT NOT NULL,
        response_json TEXT NULL,
        runtime_job_id TEXT NULL,
        conflict_message TEXT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(operation, idempotency_key),
        FOREIGN KEY(runtime_job_id) REFERENCES runtime_jobs(id) ON DELETE SET NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_idempotency_keys_runtime_job ON idempotency_keys(runtime_job_id);",
    r#"
    CREATE TABLE IF NOT EXISTS runtime_worker_heartbeats (
        worker_id TEXT PRIMARY KEY,
        started_at INTEGER NOT NULL,
        last_heartbeat_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        queue_names_json TEXT NOT NULL DEFAULT '[]',
        current_job_id TEXT NULL,
        current_stage_id TEXT NULL,
        diagnostic_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY(current_job_id) REFERENCES runtime_jobs(id) ON DELETE SET NULL,
        FOREIGN KEY(current_stage_id) REFERENCES job_stages(id) ON DELETE SET NULL
    );
    "#,
    "CREATE INDEX IF NOT EXISTS idx_runtime_worker_heartbeats_last ON runtime_worker_heartbeats(last_heartbeat_at DESC);",
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
        column: "transcription_mode",
        definition: "TEXT NOT NULL DEFAULT 'transcription'",
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
        column: "processing_progress_json",
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
        column: "speaker_attributed_text",
        definition: "TEXT NULL",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "speaker_turns_json",
        definition: "TEXT NOT NULL DEFAULT '[]'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "saa_status",
        definition: "TEXT NOT NULL DEFAULT 'not_requested'",
    },
    CompatibilityColumn {
        table: "transcription_records",
        column: "saa_warnings_json",
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

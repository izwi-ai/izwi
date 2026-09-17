use crate::{db::migrator::Migrator, storage_layout};
use anyhow::Context;
use sea_orm::{sqlx::sqlite::SqliteJournalMode, ConnectOptions, Database, DatabaseConnection};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(3);
const SQLITE_MAX_CONNECTIONS: u32 = 5;

#[cfg(test)]
pub async fn initialize_default() -> anyhow::Result<DatabaseConnection> {
    let db = connect_default().await?;
    Migrator::up(&db)
        .await
        .context("Failed to run SQLite migrations")?;
    Ok(db)
}

pub async fn connect_default() -> anyhow::Result<DatabaseConnection> {
    let db_path = storage_layout::resolve_db_path();
    let media_root = storage_layout::resolve_media_root();
    storage_layout::ensure_storage_dirs(&db_path, &media_root)
        .context("Failed to prepare storage layout for SeaORM")?;

    connect_path(&db_path).await
}

pub async fn connect_path(db_path: &Path) -> anyhow::Result<DatabaseConnection> {
    let mut options = ConnectOptions::new(sqlite_url(db_path));
    options
        .max_connections(SQLITE_MAX_CONNECTIONS)
        .min_connections(1)
        .connect_timeout(SQLITE_BUSY_TIMEOUT)
        .acquire_timeout(SQLITE_BUSY_TIMEOUT)
        .sqlx_logging(false)
        .map_sqlx_sqlite_opts(|options| {
            options
                .create_if_missing(true)
                .busy_timeout(SQLITE_BUSY_TIMEOUT)
                .foreign_keys(true)
                .journal_mode(SqliteJournalMode::Wal)
        });

    Database::connect(options).await.with_context(|| {
        format!(
            "Unable to open SeaORM SQLite database at {}",
            db_path.display()
        )
    })
}

#[derive(Clone)]
pub struct StoreDatabase {
    db_path: Option<PathBuf>,
    connection: Arc<OnceCell<DatabaseConnection>>,
}

impl StoreDatabase {
    pub fn from_default_path() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();
        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare storage layout for SeaORM store")?;
        Ok(Self::new(db_path))
    }

    pub fn new(db_path: PathBuf) -> Self {
        Self {
            db_path: Some(db_path),
            connection: Arc::new(OnceCell::new()),
        }
    }

    pub fn from_connection(connection: DatabaseConnection) -> Self {
        let cell = OnceCell::new();
        cell.set(connection)
            .map_err(|_| ())
            .expect("new OnceCell should accept initial database connection");
        Self {
            db_path: None,
            connection: Arc::new(cell),
        }
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        if let Some(db_path) = self.db_path.clone() {
            return self
                .connection
                .get_or_try_init(|| async move {
                    let db = connect_path(&db_path).await?;
                    Migrator::up(&db)
                        .await
                        .context("Failed to run SQLite migrations for SeaORM store")?;
                    Ok(db)
                })
                .await;
        }

        self.connection
            .get()
            .context("Store database was constructed without a connection")
    }
}

impl fmt::Debug for StoreDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoreDatabase")
            .field("db_path", &self.db_path)
            .field("has_connection", &self.connection.get().is_some())
            .finish()
    }
}

fn sqlite_url(db_path: &Path) -> String {
    let path = db_path.to_string_lossy();
    let escaped = path
        .replace('%', "%25")
        .replace('?', "%3F")
        .replace('#', "%23");
    format!("sqlite://{escaped}?mode=rwc")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use crate::voice_defaults::DEFAULT_VOICE_PROFILE_ID;
    use sea_orm::{ConnectionTrait, DbBackend, Statement};
    use std::collections::BTreeSet;

    #[tokio::test]
    async fn default_connection_preserves_sqlite_pragmas() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("seaorm.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let db = initialize_default().await.expect("database initializes");
        assert_eq!(db.get_database_backend(), DbBackend::Sqlite);

        let foreign_keys = query_i64_pragma(&db, "PRAGMA foreign_keys")
            .await
            .expect("foreign_keys pragma");
        assert_eq!(foreign_keys, 1);

        let journal_mode = query_string_pragma(&db, "PRAGMA journal_mode")
            .await
            .expect("journal_mode pragma");
        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");

        assert!(db_path.exists());
        assert!(media_dir.exists());

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn migrations_create_schema_and_backfill_compatibility_columns() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("legacy.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        {
            let db = connect_path(&db_path).await.expect("legacy db opens");
            db.execute_unprepared(LEGACY_COMPAT_SCHEMA)
                .await
                .expect("legacy schema created");
        }

        let db = initialize_default().await.expect("database initializes");
        let tables = user_tables(&db).await.expect("table list");
        for table in EXPECTED_TABLES {
            assert!(tables.contains(*table), "{table} table exists");
        }

        for (table, column) in EXPECTED_COMPAT_COLUMNS {
            let columns = table_columns(&db, table).await.expect("table columns");
            assert!(
                columns.contains(*column),
                "{table}.{column} compatibility column exists"
            );
        }

        let default_profiles = query_i64_scalar(
            &db,
            &format!("SELECT COUNT(*) FROM voice_profiles WHERE id = '{DEFAULT_VOICE_PROFILE_ID}'"),
        )
        .await
        .expect("default profile count");
        assert_eq!(default_profiles, 1);

        let legacy_provider_writes = query_i64_scalar(
            &db,
            "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = '00000000-0000-4000-8000-000000000001' AND provider_request_json IS NULL",
        )
        .await
        .expect("legacy provider write count");
        assert_eq!(legacy_provider_writes, 1);

        let legacy_speech = db
            .query_one_raw(Statement::from_string(
                DbBackend::Sqlite,
                "SELECT audio_storage_path, audio_media_asset_id, audio_artifact_tenant FROM speech_history_records WHERE id = 'legacy-speech'"
                    .to_string(),
            ))
            .await
            .expect("legacy speech query")
            .expect("legacy speech row");
        assert_eq!(
            legacy_speech
                .try_get_by_index::<String>(0)
                .expect("legacy audio path"),
            "generated-speech/legacy.wav"
        );
        assert_eq!(
            legacy_speech
                .try_get_by_index::<Option<String>>(1)
                .expect("opaque media id"),
            None
        );
        assert_eq!(
            legacy_speech
                .try_get_by_index::<Option<String>>(2)
                .expect("opaque tenant"),
            None
        );

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    async fn query_i64_pragma(db: &DatabaseConnection, sql: &str) -> anyhow::Result<i64> {
        query_i64_scalar(db, sql).await
    }

    async fn query_i64_scalar(db: &DatabaseConnection, sql: &str) -> anyhow::Result<i64> {
        let row = db
            .query_one_raw(Statement::from_string(DbBackend::Sqlite, sql.to_string()))
            .await?
            .context("query returned no row")?;
        Ok(row.try_get_by_index(0)?)
    }

    async fn query_string_pragma(db: &DatabaseConnection, sql: &str) -> anyhow::Result<String> {
        let row = db
            .query_one_raw(Statement::from_string(DbBackend::Sqlite, sql.to_string()))
            .await?
            .context("pragma returned no row")?;
        Ok(row.try_get_by_index(0)?)
    }

    async fn user_tables(db: &DatabaseConnection) -> anyhow::Result<BTreeSet<String>> {
        let rows = db
            .query_all_raw(Statement::from_string(
                DbBackend::Sqlite,
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    .to_string(),
            ))
            .await?;
        let mut tables = BTreeSet::new();
        for row in rows {
            tables.insert(row.try_get_by_index::<String>(0)?);
        }
        Ok(tables)
    }

    async fn table_columns(
        db: &DatabaseConnection,
        table: &str,
    ) -> anyhow::Result<BTreeSet<String>> {
        let rows = db
            .query_all_raw(Statement::from_string(
                DbBackend::Sqlite,
                format!("PRAGMA table_info({table})"),
            ))
            .await?;
        let mut columns = BTreeSet::new();
        for row in rows {
            columns.insert(row.try_get_by_index::<String>(1)?);
        }
        Ok(columns)
    }

    const EXPECTED_TABLES: &[&str] = &[
        "chat_threads",
        "chat_messages",
        "voice_profiles",
        "voice_sessions",
        "voice_turns",
        "voice_observations",
        "onboarding_state",
        "transcription_records",
        "diarization_records",
        "speech_history_records",
        "saved_voices",
        "studio_projects",
        "studio_project_segments",
        "studio_project_folders",
        "studio_project_meta",
        "studio_project_pronunciations",
        "studio_project_snapshots",
        "studio_project_render_jobs",
        "media_assets",
        "artifact_cleanup_intents",
        "provider_write_reservations",
        "durable_idempotency_keys_v2",
        "fleet_worker_observations",
        "fleet_capacity_claims",
    ];

    const EXPECTED_COMPAT_COLUMNS: &[(&str, &str)] = &[
        ("runtime_jobs", "cancellation_state"),
        ("job_stages", "cancellation_state"),
        ("chat_threads", "system_prompt"),
        ("chat_messages", "content_parts"),
        ("media_assets", "source_asset_id"),
        ("media_assets", "canonical_profile_version"),
        ("provider_write_reservations", "provider_request_json"),
        ("onboarding_state", "analytics_opt_in"),
        ("transcription_records", "transcription_mode"),
        ("transcription_records", "aligner_model_id"),
        ("transcription_records", "processing_status"),
        ("transcription_records", "processing_error"),
        ("transcription_records", "processing_progress_json"),
        ("transcription_records", "runtime_stage_id"),
        ("transcription_records", "runtime_attempt_token"),
        ("transcription_records", "segments_json"),
        ("transcription_records", "words_json"),
        ("transcription_records", "speaker_attributed_text"),
        ("transcription_records", "speaker_turns_json"),
        ("transcription_records", "saa_status"),
        ("transcription_records", "saa_warnings_json"),
        ("transcription_records", "summary_status"),
        ("transcription_records", "summary_model_id"),
        ("transcription_records", "summary_text"),
        ("transcription_records", "summary_error"),
        ("transcription_records", "summary_updated_at"),
        ("speech_history_records", "saved_voice_id"),
        ("speech_history_records", "speed"),
        ("speech_history_records", "processing_status"),
        ("speech_history_records", "processing_error"),
        ("speech_history_records", "runtime_stage_id"),
        ("speech_history_records", "runtime_attempt_token"),
        ("speech_history_records", "audio_media_asset_id"),
        ("speech_history_records", "audio_artifact_tenant"),
        ("diarization_records", "speaker_name_overrides_json"),
        ("diarization_records", "processing_status"),
        ("diarization_records", "processing_error"),
        ("diarization_records", "summary_status"),
        ("diarization_records", "summary_model_id"),
        ("diarization_records", "summary_text"),
        ("diarization_records", "summary_error"),
        ("diarization_records", "summary_updated_at"),
        ("studio_project_segments", "model_id"),
        ("studio_project_segments", "voice_mode"),
        ("studio_project_segments", "speaker"),
        ("studio_project_segments", "saved_voice_id"),
    ];

    const LEGACY_COMPAT_SCHEMA: &str = r#"
        CREATE TABLE chat_threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model_id TEXT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE chat_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            tokens_generated INTEGER NULL,
            generation_time_ms REAL NULL,
            FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
        );

        CREATE TABLE onboarding_state (
            id TEXT PRIMARY KEY,
            completed_at INTEGER NULL
        );

        CREATE TABLE media_assets (
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

        CREATE TABLE provider_write_reservations (
            write_id TEXT PRIMARY KEY,
            reservation_token TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            available_at INTEGER NOT NULL,
            state TEXT NOT NULL,
            tenant_scope TEXT NOT NULL,
            storage_namespace TEXT NOT NULL,
            content_type TEXT NOT NULL,
            filename TEXT NULL,
            expected_size_bytes INTEGER NOT NULL,
            expected_sha256 TEXT NOT NULL,
            storage_key TEXT NULL,
            cleanup_claim_token TEXT NULL,
            cleanup_claim_expires_at INTEGER NULL,
            cleanup_attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL
        );

        INSERT INTO provider_write_reservations (
            write_id, reservation_token, created_at, updated_at, expires_at,
            available_at, state, tenant_scope, storage_namespace, content_type,
            filename, expected_size_bytes, expected_sha256, storage_key,
            cleanup_claim_token, cleanup_claim_expires_at,
            cleanup_attempt_count, last_error
        ) VALUES (
            '00000000-0000-4000-8000-000000000001',
            '00000000-0000-4000-8000-000000000002',
            1, 1, 2, 2, 'reserved', 'tenant-a', 'artifact-store',
            'application/octet-stream', NULL, 1,
            '0000000000000000000000000000000000000000000000000000000000000000',
            NULL, NULL, NULL, 0, NULL
        );

        CREATE TABLE transcription_records (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            model_id TEXT NULL,
            language TEXT NULL,
            duration_secs REAL NULL,
            processing_progress_json TEXT NULL,
            processing_time_ms REAL NOT NULL,
            rtf REAL NULL,
            audio_mime_type TEXT NOT NULL,
            audio_filename TEXT NULL,
            audio_storage_path TEXT NOT NULL,
            transcription TEXT NOT NULL
        );

        CREATE TABLE speech_history_records (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            route_kind TEXT NOT NULL,
            model_id TEXT NULL,
            speaker TEXT NULL,
            language TEXT NULL,
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

        INSERT INTO speech_history_records (
            id, created_at, route_kind, model_id, speaker, language, input_text,
            voice_description, reference_text, generation_time_ms,
            audio_duration_secs, rtf, tokens_generated, audio_mime_type,
            audio_filename, audio_storage_path
        ) VALUES (
            'legacy-speech', 1, 'text_to_speech', NULL, NULL, NULL, 'legacy',
            NULL, NULL, 1.0, NULL, NULL, NULL, 'audio/wav', 'legacy.wav',
            'generated-speech/legacy.wav'
        );

        CREATE TABLE diarization_records (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            model_id TEXT NULL,
            asr_model_id TEXT NULL,
            aligner_model_id TEXT NULL,
            llm_model_id TEXT NULL,
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
            segments_json TEXT NOT NULL,
            words_json TEXT NOT NULL,
            utterances_json TEXT NOT NULL,
            audio_mime_type TEXT NOT NULL,
            audio_filename TEXT NULL,
            audio_storage_path TEXT NOT NULL
        );

        CREATE TABLE studio_projects (
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

        CREATE TABLE studio_project_segments (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            speech_record_id TEXT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY(project_id) REFERENCES studio_projects(id) ON DELETE CASCADE
        );
    "#;
}

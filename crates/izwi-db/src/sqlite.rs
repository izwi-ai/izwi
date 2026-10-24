use crate::{
    migrator::{Migrator, Schema},
    storage_layout,
};
use anyhow::Context;
use sea_orm::{
    sqlx::sqlite::SqliteJournalMode, ConnectOptions, ConnectionTrait, Database, DatabaseConnection,
    DbBackend, Statement,
};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(3);
const DEFAULT_POOL_MAX_CONNECTIONS: u32 = 5;
const MAX_POOL_MAX_CONNECTIONS: u32 = 1024;
const DEFAULT_POOL_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(3);
/// Server-side bound on every statement, and on any transaction left idle,
/// for the fleet coordination store. Coordination queries are single-row
/// claims, upserts and counts; one that runs this long is a stuck lock or a
/// hung peer, and must not hold the worker's advisory lock for everyone else.
const COORDINATION_STATEMENT_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(any(test, feature = "test-support"))]
pub async fn initialize_default() -> anyhow::Result<DatabaseConnection> {
    let db = connect_default().await?;
    Migrator::up(&db)
        .await
        .context("Failed to run SQLite migrations")?;
    Ok(db)
}

pub async fn connect_default() -> anyhow::Result<DatabaseConnection> {
    let source = storage_layout::resolve_database_source()?;
    let media_root = storage_layout::resolve_media_root();
    if let storage_layout::DatabaseSource::Path(db_path) = &source {
        storage_layout::ensure_storage_dirs(db_path, &media_root)
            .context("Failed to prepare storage layout for SeaORM")?;
    }

    connect_source(&source, None).await
}

pub async fn connect_path(db_path: &Path) -> anyhow::Result<DatabaseConnection> {
    let mut options = ConnectOptions::new(sqlite_url(db_path));
    apply_sqlite_options(&mut options);

    Database::connect(options).await.with_context(|| {
        format!(
            "Unable to open SeaORM SQLite database at {}",
            db_path.display()
        )
    })
}

/// Connect to a database by URL. SQLite URLs keep the file-mode options the
/// path form provides; server backends (PostgreSQL) connect with the
/// same bounded pool settings and no filesystem side effects.
///
/// `statement_timeout`, when set, becomes a server-side statement and
/// idle-in-transaction timeout on every PostgreSQL session.
async fn connect_url(
    url: &str,
    statement_timeout: Option<Duration>,
) -> anyhow::Result<DatabaseConnection> {
    let mut options = ConnectOptions::new(url.to_string());
    if url.trim_start().starts_with("sqlite://") {
        apply_sqlite_options(&mut options);
    } else {
        let pool = PoolSettings::from_env();
        options
            .max_connections(pool.max_connections)
            .min_connections(1)
            .connect_timeout(pool.acquire_timeout)
            .acquire_timeout(pool.acquire_timeout)
            .sqlx_logging(false);
        #[cfg(feature = "db-postgres")]
        if let Some(timeout) = statement_timeout {
            let millis = timeout.as_millis().to_string();
            options.map_sqlx_postgres_opts(move |postgres| {
                postgres.options([
                    ("statement_timeout", millis.as_str()),
                    ("idle_in_transaction_session_timeout", millis.as_str()),
                ])
            });
        }
    }
    #[cfg(not(feature = "db-postgres"))]
    let _ = statement_timeout;

    // Driver errors can quote the connection string, so the cause is
    // redacted too rather than chained as is.
    Database::connect(options).await.map_err(|error| {
        anyhow::anyhow!(
            "Unable to open SeaORM database at {}: {}",
            storage_layout::redact_database_url(url),
            storage_layout::redact_database_url_in(&error.to_string(), url)
        )
    })
}

/// Connection pool bounds for every backend: `IZWI_DB_MAX_CONNECTIONS`
/// (1-1024, default 5) and `IZWI_DB_ACQUIRE_TIMEOUT_MS` (default 3000).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PoolSettings {
    max_connections: u32,
    acquire_timeout: Duration,
}

impl PoolSettings {
    fn from_env() -> Self {
        let max_connections = std::env::var("IZWI_DB_MAX_CONNECTIONS")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|value| (1..=MAX_POOL_MAX_CONNECTIONS).contains(value))
            .unwrap_or(DEFAULT_POOL_MAX_CONNECTIONS);
        let acquire_timeout = std::env::var("IZWI_DB_ACQUIRE_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .map_or(DEFAULT_POOL_ACQUIRE_TIMEOUT, Duration::from_millis);
        Self {
            max_connections,
            acquire_timeout,
        }
    }
}

async fn connect_source(
    source: &storage_layout::DatabaseSource,
    statement_timeout: Option<Duration>,
) -> anyhow::Result<DatabaseConnection> {
    match source {
        storage_layout::DatabaseSource::Path(path) => connect_path(path).await,
        storage_layout::DatabaseSource::Url(url) => connect_url(url, statement_timeout).await,
    }
}

fn apply_sqlite_options(options: &mut ConnectOptions) {
    let pool = PoolSettings::from_env();
    options
        .max_connections(pool.max_connections)
        .min_connections(1)
        .connect_timeout(pool.acquire_timeout)
        .acquire_timeout(pool.acquire_timeout)
        .sqlx_logging(false)
        .map_sqlx_sqlite_opts(|options| {
            options
                .create_if_missing(true)
                .busy_timeout(SQLITE_BUSY_TIMEOUT)
                .foreign_keys(true)
                .journal_mode(SqliteJournalMode::Wal)
        });
}

#[derive(Clone)]
pub struct StoreDatabase {
    source: Option<storage_layout::DatabaseSource>,
    connection: Arc<OnceCell<DatabaseConnection>>,
    statement_timeout: Option<Duration>,
    schema: Schema,
    clock: Arc<DatabaseClock>,
}

/// How often a store re-reads the database server's clock.
const DATABASE_CLOCK_RESYNC: Duration = Duration::from_secs(30);

/// The database server's clock, sampled at most every
/// [`DATABASE_CLOCK_RESYNC`] and advanced with this process's monotonic
/// clock in between. Every node sharing a database then measures leases,
/// claim TTLs and expiry against one clock, whatever its own wall clock
/// says. Before the first sample the process wall clock stands in.
#[derive(Debug, Default)]
struct DatabaseClock {
    sample: std::sync::Mutex<Option<(i64, Instant)>>,
    #[cfg(any(test, feature = "test-support"))]
    wall_clock_skew_ms: std::sync::atomic::AtomicI64,
}

impl DatabaseClock {
    fn now_millis(&self) -> i64 {
        let sample = *self
            .sample
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match sample {
            Some((database_ms, sampled_at)) => database_ms.saturating_add(
                i64::try_from(sampled_at.elapsed().as_millis()).unwrap_or(i64::MAX),
            ),
            None => self.wall_clock_millis(),
        }
    }

    fn wall_clock_millis(&self) -> i64 {
        let now = crate::current_timestamp_millis();
        #[cfg(any(test, feature = "test-support"))]
        let now = now + self.wall_clock_skew_ms.load(std::sync::atomic::Ordering::SeqCst);
        now
    }

    fn is_due(&self) -> bool {
        self.sample
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_none_or(|(_, sampled_at)| sampled_at.elapsed() >= DATABASE_CLOCK_RESYNC)
    }

    /// Read the server's clock, attributing it to the midpoint of the round
    /// trip. A failed read keeps the previous sample.
    async fn sync(&self, db: &DatabaseConnection) {
        let sql = match db.get_database_backend() {
            DbBackend::Postgres => "SELECT (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT",
            _ => "SELECT CAST(ROUND((julianday('now') - 2440587.5) * 86400000.0) AS INTEGER)",
        };
        let sent = Instant::now();
        let read = db
            .query_one_raw(Statement::from_string(db.get_database_backend(), sql))
            .await;
        let round_trip = sent.elapsed();
        match read.map(|row| row.map(|row| row.try_get_by_index::<i64>(0))) {
            Ok(Some(Ok(database_ms))) => {
                *self
                    .sample
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()) =
                    Some((database_ms, sent + round_trip / 2));
            }
            other => tracing::warn!(
                error = ?other.err(),
                "Failed to read the database clock; keeping the previous sample"
            ),
        }
    }
}

impl StoreDatabase {
    pub fn from_default_path() -> anyhow::Result<Self> {
        let source = storage_layout::resolve_database_source()?;
        let media_root = storage_layout::resolve_media_root();
        if let storage_layout::DatabaseSource::Path(db_path) = &source {
            storage_layout::ensure_storage_dirs(db_path, &media_root)
                .context("Failed to prepare storage layout for SeaORM store")?;
        }
        Ok(Self {
            source: Some(source),
            connection: Arc::new(OnceCell::new()),
            statement_timeout: None,
            schema: Schema::App,
            clock: Arc::default(),
        })
    }

    pub fn new(db_path: PathBuf) -> Self {
        Self {
            source: Some(storage_layout::DatabaseSource::Path(db_path)),
            connection: Arc::new(OnceCell::new()),
            statement_timeout: None,
            schema: Schema::App,
            clock: Arc::default(),
        }
    }

    /// Address the store through a bounded database URL (DS5 fleet profile:
    /// the coordination store may be a shared PostgreSQL database).
    pub fn from_url(url: String) -> Self {
        Self {
            source: Some(storage_layout::DatabaseSource::Url(url)),
            connection: Arc::new(OnceCell::new()),
            statement_timeout: None,
            schema: Schema::App,
            clock: Arc::default(),
        }
    }

    /// Address the multi-gateway coordination store through a database URL.
    /// PostgreSQL sessions carry a server-side statement and
    /// idle-in-transaction timeout, so a stuck query or an abandoned claim
    /// transaction cannot hold the per-worker advisory lock indefinitely.
    pub fn from_coordination_url(url: String) -> Self {
        Self {
            statement_timeout: Some(COORDINATION_STATEMENT_TIMEOUT),
            ..Self::from_url(url)
        }
    }

    /// Migrate this database to the fleet coordination schema instead of
    /// the application schema.
    pub fn for_fleet(mut self) -> Self {
        self.schema = Schema::Fleet;
        self
    }

    pub fn from_connection(connection: DatabaseConnection) -> Self {
        let cell = OnceCell::new();
        cell.set(connection)
            .map_err(|_| ())
            .expect("new OnceCell should accept initial database connection");
        Self {
            source: None,
            connection: Arc::new(cell),
            statement_timeout: None,
            schema: Schema::App,
            clock: Arc::default(),
        }
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        let db = match self.source.clone() {
            Some(source) => {
                let statement_timeout = self.statement_timeout;
                let schema = self.schema;
                self.connection
                    .get_or_try_init(|| async move {
                        let db = connect_source(&source, statement_timeout).await?;
                        Migrator::up_schema(&db, schema)
                            .await
                            .context("Failed to run migrations for SeaORM store")?;
                        Ok::<_, anyhow::Error>(db)
                    })
                    .await?
            }
            None => self
                .connection
                .get()
                .context("Store database was constructed without a connection")?,
        };
        if self.clock.is_due() {
            self.clock.sync(db).await;
        }
        Ok(db)
    }

    /// The current time in Unix milliseconds by the database server's clock
    /// (see [`DatabaseClock`]).
    pub fn now_millis(&self) -> i64 {
        self.clock.now_millis()
    }

    /// Skew this process's wall clock, which the database clock replaces.
    #[cfg(any(test, feature = "test-support"))]
    pub fn skew_wall_clock_for_test(&self, skew_ms: i64) {
        self.clock
            .wall_clock_skew_ms
            .store(skew_ms, std::sync::atomic::Ordering::SeqCst);
    }
}

impl fmt::Debug for StoreDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoreDatabase")
            .field("source", &self.source)
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

    #[test]
    fn pool_settings_are_configurable_and_bounded() {
        let _guard = env_lock();
        std::env::remove_var("IZWI_DB_MAX_CONNECTIONS");
        std::env::remove_var("IZWI_DB_ACQUIRE_TIMEOUT_MS");
        assert_eq!(
            PoolSettings::from_env(),
            PoolSettings {
                max_connections: DEFAULT_POOL_MAX_CONNECTIONS,
                acquire_timeout: DEFAULT_POOL_ACQUIRE_TIMEOUT,
            }
        );
        std::env::set_var("IZWI_DB_MAX_CONNECTIONS", "32");
        std::env::set_var("IZWI_DB_ACQUIRE_TIMEOUT_MS", "750");
        assert_eq!(
            PoolSettings::from_env(),
            PoolSettings {
                max_connections: 32,
                acquire_timeout: Duration::from_millis(750),
            }
        );
        std::env::set_var("IZWI_DB_MAX_CONNECTIONS", "0");
        std::env::set_var("IZWI_DB_ACQUIRE_TIMEOUT_MS", "never");
        assert_eq!(
            PoolSettings::from_env(),
            PoolSettings {
                max_connections: DEFAULT_POOL_MAX_CONNECTIONS,
                acquire_timeout: DEFAULT_POOL_ACQUIRE_TIMEOUT,
            }
        );
        std::env::remove_var("IZWI_DB_MAX_CONNECTIONS");
        std::env::remove_var("IZWI_DB_ACQUIRE_TIMEOUT_MS");
    }

    /// DUR-F11: a database password never reaches errors or Debug output.
    #[tokio::test]
    async fn database_url_credentials_stay_out_of_errors_and_debug() {
        let store = StoreDatabase::from_url(
            "postgres://izwi:hunter2@127.0.0.1:1/izwi?password=hunter3".to_string(),
        );
        let debug = format!("{store:?}");
        assert!(!debug.contains("hunter"), "{debug}");
        let error = format!("{:#}", store.connection().await.unwrap_err());
        assert!(error.contains("izwi:***@127.0.0.1"), "{error}");
        assert!(!error.contains("hunter"), "{error}");
    }

    #[tokio::test]
    async fn url_source_migrates_a_sqlite_database_without_a_path() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("url-sourced.sqlite3");
        let url = sqlite_url(&db_path);

        let store = StoreDatabase::from_url(url);
        let db = store.connection().await.expect("url-sourced store opens");
        assert_eq!(db.get_database_backend(), DbBackend::Sqlite);

        let tables = user_tables(db).await.expect("table list");
        for table in EXPECTED_TABLES {
            assert!(tables.contains(*table), "{table} table exists");
        }
        assert!(db_path.exists(), "sqlite url created the database file");
    }

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
        "runtime_quota_usage",
        "gateway_principal_keys",
        "schema_migrations",
    ];

    const EXPECTED_COMPAT_COLUMNS: &[(&str, &str)] = &[
        ("runtime_jobs", "cancellation_state"),
        ("runtime_jobs", "replay_swept_at"),
        ("job_stages", "cancellation_state"),
        ("chat_threads", "system_prompt"),
        ("chat_messages", "content_parts"),
        ("chat_messages", "model_id"),
        ("chat_messages", "finish_reason"),
        ("chat_messages", "reasoning_ms"),
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
        ("transcription_records", "alignment_warnings_json"),
        ("transcription_records", "summary_status"),
        ("transcription_records", "summary_model_id"),
        ("transcription_records", "summary_text"),
        ("transcription_records", "summary_error"),
        ("transcription_records", "summary_updated_at"),
        ("transcription_records", "owner_node_id"),
        ("transcription_records", "target_language"),
        ("transcription_records", "keywords_json"),
        ("speech_history_records", "saved_voice_id"),
        ("speech_history_records", "speed"),
        ("speech_history_records", "processing_status"),
        ("speech_history_records", "processing_error"),
        ("speech_history_records", "runtime_stage_id"),
        ("speech_history_records", "runtime_attempt_token"),
        ("speech_history_records", "audio_media_asset_id"),
        ("speech_history_records", "audio_artifact_tenant"),
        ("speech_history_records", "owner_node_id"),
        ("diarization_records", "speaker_name_overrides_json"),
        ("diarization_records", "processing_status"),
        ("diarization_records", "processing_error"),
        ("diarization_records", "summary_status"),
        ("diarization_records", "summary_model_id"),
        ("diarization_records", "summary_text"),
        ("diarization_records", "summary_error"),
        ("diarization_records", "summary_updated_at"),
        ("diarization_records", "warnings_json"),
        ("diarization_records", "language"),
        ("diarization_records", "vad_gate"),
        ("diarization_records", "owner_node_id"),
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

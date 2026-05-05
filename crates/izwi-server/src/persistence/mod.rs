use crate::{db, db::migrator::Migrator, storage_layout};
use anyhow::{bail, Context};
use izwi_hooks::{
    DatabaseBackend, DatabaseConnectionDecision, DatabaseMigrationMode, DatabaseProviderDecision,
    DatabaseProviderRequest, EnterpriseHooks, HookError, HookMetadata, HookResult,
    MediaDeleteRequest, MediaNamespace, MediaObjectKey, MediaObjectMetadata, MediaReadRequest,
    MediaStorageProvider, MediaStorageProviderDecision, MediaStorageProviderRequest,
    MediaWriteRequest, StoredMediaBytes, StoredMediaObject,
};
use sea_orm::{DatabaseConnection, DatabaseConnectionType, DbBackend};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct PersistenceContext {
    pub database: DatabaseContext,
    pub media_storage: Arc<dyn MediaStorageProvider>,
}

impl PersistenceContext {
    pub async fn resolve(enterprise_hooks: &EnterpriseHooks) -> anyhow::Result<Self> {
        let database = resolve_database(enterprise_hooks).await?;
        let media_storage = resolve_media_storage(enterprise_hooks).await?;

        Ok(Self {
            database,
            media_storage,
        })
    }

    #[cfg(test)]
    pub async fn local_default() -> anyhow::Result<Self> {
        let hooks = EnterpriseHooks::noop();
        Self::resolve(&hooks).await
    }

    pub fn media_storage(&self) -> Arc<dyn MediaStorageProvider> {
        self.media_storage.clone()
    }
}

#[derive(Clone)]
pub struct DatabaseContext {
    connection: DatabaseConnection,
    backend: DatabaseBackend,
    migration_mode: DatabaseMigrationMode,
    metadata: HookMetadata,
}

impl DatabaseContext {
    pub fn new(
        connection: DatabaseConnection,
        backend: DatabaseBackend,
        migration_mode: DatabaseMigrationMode,
        metadata: HookMetadata,
    ) -> Self {
        Self {
            connection,
            backend,
            migration_mode,
            metadata,
        }
    }

    pub fn connection(&self) -> DatabaseConnection {
        self.connection.clone()
    }

    pub fn backend(&self) -> &DatabaseBackend {
        &self.backend
    }

    pub fn migration_mode(&self) -> &DatabaseMigrationMode {
        &self.migration_mode
    }

    pub fn metadata(&self) -> &HookMetadata {
        &self.metadata
    }
}

pub async fn persist_audio_object(
    provider: &Arc<dyn MediaStorageProvider>,
    namespace: MediaNamespace,
    record_id: impl Into<String>,
    preferred_filename: Option<&str>,
    mime_type: &str,
    bytes: &[u8],
    metadata: HookMetadata,
) -> anyhow::Result<String> {
    let stored = provider
        .put(
            MediaWriteRequest {
                namespace,
                record_id: record_id.into(),
                preferred_filename: preferred_filename.map(str::to_string),
                content_type: mime_type.to_string(),
                metadata,
            },
            bytes.to_vec(),
        )
        .await
        .map_err(|err| anyhow::anyhow!("Media storage write failed: {err}"))?;

    Ok(stored.key.key)
}

pub async fn read_media_object(
    provider: &Arc<dyn MediaStorageProvider>,
    key: &str,
) -> anyhow::Result<Vec<u8>> {
    let stored = provider
        .get(MediaReadRequest {
            key: MediaObjectKey::new(key),
            metadata: HookMetadata::new(),
        })
        .await
        .map_err(|err| anyhow::anyhow!("Media storage read failed: {err}"))?;

    Ok(stored.bytes)
}

pub async fn delete_media_object(
    provider: &Arc<dyn MediaStorageProvider>,
    key: Option<&str>,
) -> anyhow::Result<()> {
    let Some(key) = key.filter(|key| !key.trim().is_empty()) else {
        return Ok(());
    };

    provider
        .delete(MediaDeleteRequest {
            key: MediaObjectKey::new(key),
            metadata: HookMetadata::new(),
        })
        .await
        .map_err(|err| anyhow::anyhow!("Media storage delete failed: {err}"))
}

async fn resolve_database(enterprise_hooks: &EnterpriseHooks) -> anyhow::Result<DatabaseContext> {
    match enterprise_hooks
        .database
        .resolve_database(&DatabaseProviderRequest::server_runtime())
        .await
        .map_err(|err| anyhow::anyhow!("Enterprise database hook failed: {err}"))?
    {
        DatabaseProviderDecision::UseDefault => local_database_context().await,
        DatabaseProviderDecision::UseConnection(decision) => {
            provider_database_context(decision).await
        }
    }
}

async fn local_database_context() -> anyhow::Result<DatabaseContext> {
    let connection = db::sqlite::connect_default().await?;
    Migrator::up(&connection)
        .await
        .context("Failed to run local SQLite migrations")?;

    Ok(DatabaseContext::new(
        connection,
        DatabaseBackend::Sqlite,
        DatabaseMigrationMode::IzwiManaged,
        HookMetadata::new(),
    ))
}

async fn provider_database_context(
    decision: DatabaseConnectionDecision,
) -> anyhow::Result<DatabaseContext> {
    let actual_backend = actual_database_backend(&decision.connection)?;
    validate_declared_backend(&decision.backend, &actual_backend)?;
    validate_migration_mode_for_backend(&actual_backend, &decision.migration_mode)?;

    if matches!(decision.migration_mode, DatabaseMigrationMode::IzwiManaged) {
        Migrator::up(&decision.connection)
            .await
            .context("Failed to run Izwi-managed migrations on enterprise database")?;
    }

    Ok(DatabaseContext::new(
        decision.connection,
        decision.backend,
        decision.migration_mode,
        decision.metadata,
    ))
}

fn actual_database_backend(connection: &DatabaseConnection) -> anyhow::Result<DatabaseBackend> {
    if matches!(&connection.inner, DatabaseConnectionType::Disconnected) {
        bail!("Enterprise database hook returned a disconnected SeaORM connection");
    }

    Ok(hook_backend_from_seaorm(connection.get_database_backend()))
}

fn hook_backend_from_seaorm(backend: DbBackend) -> DatabaseBackend {
    match backend {
        DbBackend::Sqlite => DatabaseBackend::Sqlite,
        DbBackend::Postgres => DatabaseBackend::Postgres,
        DbBackend::MySql => DatabaseBackend::Mysql,
        _ => DatabaseBackend::Other(format!("{backend:?}")),
    }
}

fn validate_declared_backend(
    declared: &DatabaseBackend,
    actual: &DatabaseBackend,
) -> anyhow::Result<()> {
    if matches!(declared, DatabaseBackend::Other(_)) || declared == actual {
        return Ok(());
    }

    bail!(
        "Enterprise database hook declared backend {declared:?}, but the SeaORM connection reports {actual:?}"
    );
}

fn validate_migration_mode_for_backend(
    backend: &DatabaseBackend,
    migration_mode: &DatabaseMigrationMode,
) -> anyhow::Result<()> {
    if matches!(migration_mode, DatabaseMigrationMode::IzwiManaged)
        && !matches!(backend, DatabaseBackend::Sqlite)
    {
        bail!(
            "Izwi-managed migrations currently require the local SQLite backend; enterprise database providers must use provider-managed or disabled migrations for {backend:?}"
        );
    }

    Ok(())
}

async fn resolve_media_storage(
    enterprise_hooks: &EnterpriseHooks,
) -> anyhow::Result<Arc<dyn MediaStorageProvider>> {
    match enterprise_hooks
        .media_storage
        .resolve_media_storage(&MediaStorageProviderRequest::server_media())
        .await
        .map_err(|err| anyhow::anyhow!("Enterprise media storage hook failed: {err}"))?
    {
        MediaStorageProviderDecision::UseDefault => Ok(Arc::new(LocalMediaStorageProvider::new(
            storage_layout::resolve_media_root(),
        ))),
        MediaStorageProviderDecision::UseProvider(provider) => Ok(provider),
    }
}

#[derive(Debug, Clone)]
pub struct LocalMediaStorageProvider {
    media_root: PathBuf,
}

impl LocalMediaStorageProvider {
    pub fn new(media_root: PathBuf) -> Self {
        Self { media_root }
    }
}

#[async_trait::async_trait]
impl MediaStorageProvider for LocalMediaStorageProvider {
    async fn put(
        &self,
        request: MediaWriteRequest,
        bytes: Vec<u8>,
    ) -> HookResult<StoredMediaObject> {
        let content_length = bytes.len() as u64;
        let (group, namespace) = local_namespace(&request.namespace, &request.metadata);
        let key = storage_layout::persist_audio_file(
            &self.media_root,
            group,
            &namespace,
            &request.record_id,
            request.preferred_filename.as_deref(),
            &request.content_type,
            &bytes,
        )
        .map_err(|err| HookError::Failed(err.to_string()))?;

        Ok(StoredMediaObject {
            key: MediaObjectKey::new(key),
            metadata: MediaObjectMetadata {
                content_type: request.content_type,
                filename: request.preferred_filename,
                content_length: Some(content_length),
                sha256: None,
                tenant_id: None,
                attributes: request.metadata,
            },
        })
    }

    async fn get(&self, request: MediaReadRequest) -> HookResult<StoredMediaBytes> {
        let bytes = storage_layout::read_media_file(&self.media_root, &request.key.key)
            .map_err(|err| HookError::Failed(err.to_string()))?;

        Ok(StoredMediaBytes {
            metadata: MediaObjectMetadata {
                content_type: content_type_from_key(&request.key.key).to_string(),
                filename: filename_from_key(&request.key.key),
                content_length: Some(bytes.len() as u64),
                sha256: None,
                tenant_id: None,
                attributes: request.metadata,
            },
            bytes,
        })
    }

    async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
        storage_layout::delete_media_file(&self.media_root, Some(&request.key.key))
            .map_err(|err| HookError::Failed(err.to_string()))
    }
}

fn local_namespace(
    namespace: &MediaNamespace,
    metadata: &HookMetadata,
) -> (storage_layout::MediaGroup, String) {
    match namespace {
        MediaNamespace::TranscriptionUpload => (
            storage_layout::MediaGroup::Uploads,
            "transcription".to_string(),
        ),
        MediaNamespace::DiarizationUpload => (
            storage_layout::MediaGroup::Uploads,
            "diarization".to_string(),
        ),
        MediaNamespace::GeneratedSpeech => {
            let route_kind = metadata
                .get("route_kind")
                .map(String::as_str)
                .unwrap_or("speech");
            (
                storage_layout::MediaGroup::Generated,
                format!("speech/{route_kind}"),
            )
        }
        MediaNamespace::SavedVoice => (storage_layout::MediaGroup::Generated, "voices".to_string()),
        MediaNamespace::ChatMedia => (storage_layout::MediaGroup::Uploads, "chat".to_string()),
        MediaNamespace::Export => (storage_layout::MediaGroup::Generated, "exports".to_string()),
        MediaNamespace::Other(namespace) => (
            storage_layout::MediaGroup::Generated,
            sanitize_namespace(namespace),
        ),
    }
}

fn sanitize_namespace(namespace: &str) -> String {
    namespace
        .split('/')
        .filter(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn content_type_from_key(key: &str) -> &'static str {
    match std::path::Path::new(key)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("ogg") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("webm") => "audio/webm",
        Some("m4a") => "audio/mp4",
        Some("aac") => "audio/aac",
        _ => "application/octet-stream",
    }
}

fn filename_from_key(key: &str) -> Option<String> {
    std::path::Path::new(key)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use izwi_hooks::{DatabaseProvider, DatabaseProviderDecision};
    use sea_orm::DbBackend;

    #[tokio::test]
    async fn noop_hooks_resolve_local_persistence() {
        let _guard = env_lock();
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir.path().join("izwi.sqlite3");
        let media_dir = temp_dir.path().join("media");
        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_dir);

        let context = PersistenceContext::local_default()
            .await
            .expect("local persistence resolves");

        assert_eq!(
            context.database.connection().get_database_backend(),
            DbBackend::Sqlite
        );
        assert!(db_path.exists());
        assert!(media_dir.exists());

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn local_media_provider_round_trips_bytes() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let provider = LocalMediaStorageProvider::new(temp_dir.path().to_path_buf());
        let mut metadata = HookMetadata::new();
        metadata.insert("route_kind".to_string(), "tts".to_string());

        let stored = provider
            .put(
                MediaWriteRequest {
                    namespace: MediaNamespace::GeneratedSpeech,
                    record_id: "record-1".to_string(),
                    preferred_filename: Some("speech.wav".to_string()),
                    content_type: "audio/wav".to_string(),
                    metadata,
                },
                b"audio".to_vec(),
            )
            .await
            .expect("write media");

        assert_eq!(stored.key.key, "generated/speech/tts/record-1.wav");

        let bytes = provider
            .get(MediaReadRequest {
                key: stored.key.clone(),
                metadata: HookMetadata::new(),
            })
            .await
            .expect("read media");
        assert_eq!(bytes.bytes, b"audio");

        provider
            .delete(MediaDeleteRequest {
                key: stored.key,
                metadata: HookMetadata::new(),
            })
            .await
            .expect("delete media");
    }

    #[tokio::test]
    async fn enterprise_database_backend_mismatch_is_rejected() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let connection = db::sqlite::connect_path(&temp_dir.path().join("enterprise.sqlite3"))
            .await
            .expect("sqlite connection");
        let mut hooks = EnterpriseHooks::noop();
        hooks.database = Arc::new(StaticDatabaseProvider {
            decision: DatabaseProviderDecision::UseConnection(DatabaseConnectionDecision {
                connection,
                backend: DatabaseBackend::Postgres,
                migration_mode: DatabaseMigrationMode::ProviderManaged,
                metadata: HookMetadata::new(),
            }),
        });

        let error = match PersistenceContext::resolve(&hooks).await {
            Ok(_) => panic!("backend mismatch should fail"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("declared backend Postgres, but the SeaORM connection reports Sqlite"),
            "{error}"
        );
    }

    #[test]
    fn izwi_managed_migrations_are_restricted_to_sqlite() {
        assert!(validate_migration_mode_for_backend(
            &DatabaseBackend::Sqlite,
            &DatabaseMigrationMode::IzwiManaged,
        )
        .is_ok());

        let error = validate_migration_mode_for_backend(
            &DatabaseBackend::Postgres,
            &DatabaseMigrationMode::IzwiManaged,
        )
        .expect_err("managed migrations should reject non-sqlite backends");

        assert!(
            error
                .to_string()
                .contains("Izwi-managed migrations currently require the local SQLite backend"),
            "{error}"
        );
    }

    #[test]
    fn provider_managed_modes_accept_non_sqlite_backends() {
        assert!(validate_migration_mode_for_backend(
            &DatabaseBackend::Postgres,
            &DatabaseMigrationMode::ProviderManaged,
        )
        .is_ok());
        assert!(validate_migration_mode_for_backend(
            &DatabaseBackend::Mysql,
            &DatabaseMigrationMode::Disabled,
        )
        .is_ok());
    }

    #[derive(Clone)]
    struct StaticDatabaseProvider {
        decision: DatabaseProviderDecision,
    }

    #[async_trait::async_trait]
    impl DatabaseProvider for StaticDatabaseProvider {
        async fn resolve_database(
            &self,
            _request: &DatabaseProviderRequest,
        ) -> HookResult<DatabaseProviderDecision> {
            Ok(self.decision.clone())
        }
    }
}

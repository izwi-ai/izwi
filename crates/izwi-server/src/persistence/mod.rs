use crate::{
    db,
    db::{migrator::Migrator, schema_contract::validate_provider_managed_schema},
    storage_layout,
};
use anyhow::{bail, Context};
use izwi_hooks::{
    DatabaseBackend, DatabaseConnectionDecision, DatabaseMigrationMode, DatabaseProviderDecision,
    DatabaseProviderRequest, EnterpriseHooks, HookError, HookMetadata, HookResult,
    MediaDeleteRequest, MediaNamespace, MediaObjectKey, MediaObjectMetadata, MediaReadRequest,
    MediaReservedWriteRecoveryRequest, MediaReservedWriteRequest, MediaStorageProvider,
    MediaStorageProviderDecision, MediaStorageProviderRequest, MediaWriteRequest, StoredMediaBytes,
    StoredMediaObject, MEDIA_RESERVED_WRITE_VERSION,
};
use sea_orm::{DatabaseConnection, DatabaseConnectionType, DbBackend};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct PersistenceContext {
    pub database: DatabaseContext,
    pub media_storage: Arc<dyn MediaStorageProvider>,
    local_media_root: Option<PathBuf>,
}

impl PersistenceContext {
    pub async fn resolve(enterprise_hooks: &EnterpriseHooks) -> anyhow::Result<Self> {
        let database = resolve_database(enterprise_hooks).await?;
        let media_storage = resolve_media_storage(enterprise_hooks).await?;

        Ok(Self {
            database,
            media_storage: media_storage.provider,
            local_media_root: media_storage.local_media_root,
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

    pub fn local_media_root(&self) -> Option<&PathBuf> {
        self.local_media_root.as_ref()
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
) -> Result<StoredMediaBytes, MediaStorageError> {
    let key = key.to_string();
    provider
        .get(MediaReadRequest {
            key: MediaObjectKey::new(key.clone()),
            metadata: HookMetadata::new(),
        })
        .await
        .map_err(|err| match err {
            HookError::NotFound(message) => MediaStorageError::NotFound { key, message },
            err => MediaStorageError::ReadFailed(err),
        })
}

pub async fn read_media_stream(
    provider: &Arc<dyn MediaStorageProvider>,
    key: &str,
) -> Result<izwi_hooks::StoredMediaStream, MediaStorageError> {
    provider
        .get_stream(MediaReadRequest {
            key: MediaObjectKey::new(key),
            metadata: HookMetadata::new(),
        })
        .await
        .map_err(|err| match err {
            HookError::NotFound(message) => MediaStorageError::NotFound {
                key: key.to_string(),
                message,
            },
            err => MediaStorageError::ReadFailed(err),
        })
}

pub async fn delete_media_object(
    provider: &Arc<dyn MediaStorageProvider>,
    key: Option<&str>,
) -> anyhow::Result<()> {
    let Some(key) = key.filter(|key| !key.trim().is_empty()) else {
        return Ok(());
    };

    match provider
        .delete(MediaDeleteRequest {
            key: MediaObjectKey::new(key),
            metadata: HookMetadata::new(),
        })
        .await
    {
        Ok(()) | Err(HookError::NotFound(_)) => Ok(()),
        Err(err) => Err(anyhow::anyhow!("Media storage delete failed: {err}")),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MediaStorageError {
    #[error("media storage object not found: {key}: {message}")]
    NotFound { key: String, message: String },
    #[error("media storage read failed: {0}")]
    ReadFailed(#[source] HookError),
}

impl MediaStorageError {
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }
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
    validate_supported_database_backend(&actual_backend)?;
    validate_migration_mode_for_backend(&actual_backend, &decision.migration_mode)?;

    if matches!(decision.migration_mode, DatabaseMigrationMode::IzwiManaged) {
        Migrator::up(&decision.connection)
            .await
            .context("Failed to run Izwi-managed migrations on enterprise database")?;
    } else {
        validate_provider_managed_schema(&decision.connection).await?;
    }

    Ok(DatabaseContext::new(
        decision.connection,
        actual_backend,
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

fn validate_supported_database_backend(backend: &DatabaseBackend) -> anyhow::Result<()> {
    match backend {
        DatabaseBackend::Sqlite | DatabaseBackend::Postgres | DatabaseBackend::Mysql => Ok(()),
        DatabaseBackend::Other(name) => {
            bail!("Enterprise database backend {name:?} is not supported by this runtime")
        }
    }
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

struct ResolvedMediaStorage {
    provider: Arc<dyn MediaStorageProvider>,
    local_media_root: Option<PathBuf>,
}

async fn resolve_media_storage(
    enterprise_hooks: &EnterpriseHooks,
) -> anyhow::Result<ResolvedMediaStorage> {
    match enterprise_hooks
        .media_storage
        .resolve_media_storage(&MediaStorageProviderRequest::server_media())
        .await
        .map_err(|err| anyhow::anyhow!("Enterprise media storage hook failed: {err}"))?
    {
        MediaStorageProviderDecision::UseDefault => {
            let media_root = storage_layout::resolve_media_root();
            Ok(ResolvedMediaStorage {
                provider: Arc::new(LocalMediaStorageProvider::new(media_root.clone())),
                local_media_root: Some(media_root),
            })
        }
        MediaStorageProviderDecision::UseProvider(provider) => Ok(ResolvedMediaStorage {
            provider,
            local_media_root: None,
        }),
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
    fn reserved_write_protocol_version(&self) -> Option<u16> {
        Some(MEDIA_RESERVED_WRITE_VERSION)
    }

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
                tenant_id: tenant_id_from_metadata(&request.metadata),
                attributes: request.metadata,
            },
        })
    }

    async fn put_reserved(
        &self,
        request: MediaReservedWriteRequest,
        bytes: Vec<u8>,
    ) -> HookResult<StoredMediaObject> {
        validate_reserved_write(&request, &bytes)?;
        let fingerprint =
            reserved_write_fingerprint(&request.request, request.content_length, &request.sha256);
        let key = reserved_local_key(&request.request, &request.write_id, &fingerprint);
        let media_root = self.media_root.clone();
        let key_for_write = key.clone();
        let deadline = request.expires_at_unix_ms;
        let write_id = request.write_id.clone();
        let content_type = request.request.content_type.clone();
        let filename = request.request.preferred_filename.clone();
        let metadata = request.request.metadata.clone();
        let content_length = request.content_length;
        let digest = request.sha256.clone();
        tokio::task::spawn_blocking(move || {
            with_reserved_write_lock(&media_root, &write_id, || {
                ensure_reserved_write_live(deadline)?;
                let target = storage_layout::resolve_media_path(&media_root, &key_for_write)?;
                let parent = target.parent().context("Reserved media parent directory")?;
                std::fs::create_dir_all(parent)?;
                let final_name = target
                    .file_name()
                    .context("Reserved media filename")?
                    .to_string_lossy()
                    .into_owned();
                let temporary_name = format!(".{fingerprint}.tmp");
                validate_reserved_write_directory(parent, &final_name, &temporary_name)?;
                if target.exists() {
                    anyhow::ensure!(
                        file_matches_bytes(&target, &bytes)?,
                        "Reserved media write ID was reused for different bytes"
                    );
                    remove_if_present(&parent.join(&temporary_name))?;
                    sync_directory(parent)?;
                    return Ok(());
                }
                let temporary_path = parent.join(&temporary_name);
                let mut temporary = std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&temporary_path)?;
                use std::io::Write as _;
                temporary.write_all(&bytes)?;
                temporary.sync_all()?;
                drop(temporary);
                ensure_reserved_write_live(deadline)?;
                match std::fs::hard_link(&temporary_path, &target) {
                    Ok(()) => {
                        std::fs::remove_file(&temporary_path)?;
                        sync_directory(parent)?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        anyhow::ensure!(
                            file_matches_bytes(&target, &bytes)?,
                            "Reserved media write ID was reused for different bytes"
                        );
                        std::fs::remove_file(&temporary_path)?;
                    }
                    Err(error) => return Err(error.into()),
                }
                Ok(())
            })
        })
        .await
        .map_err(|error| HookError::Failed(error.to_string()))?
        .map_err(|error| HookError::Failed(error.to_string()))?;

        Ok(StoredMediaObject {
            key: MediaObjectKey::new(key),
            metadata: MediaObjectMetadata {
                content_type,
                filename,
                content_length: Some(content_length),
                sha256: Some(digest),
                tenant_id: tenant_id_from_metadata(&metadata),
                attributes: metadata,
            },
        })
    }

    async fn recover_reserved_write(
        &self,
        request: MediaReservedWriteRecoveryRequest,
    ) -> HookResult<()> {
        validate_reserved_recovery(&request)?;
        let fingerprint =
            reserved_write_fingerprint(&request.request, request.content_length, &request.sha256);
        let expected_key = reserved_local_key(&request.request, &request.write_id, &fingerprint);
        if let Some(key) = request.storage_key.as_ref() {
            if key.key != expected_key {
                return Err(HookError::Failed(
                    "Reserved media recovery key did not match its write ID".into(),
                ));
            }
        }
        let media_root = self.media_root.clone();
        tokio::task::spawn_blocking(move || {
            with_reserved_write_lock(&media_root, &request.write_id, || {
                anyhow::ensure!(
                    current_unix_millis() >= request.expires_at_unix_ms,
                    "Reserved media write is not yet recoverable"
                );
                let target = storage_layout::resolve_media_path(&media_root, &expected_key)?;
                let parent = target.parent().context("Reserved media parent directory")?;
                let final_name = target
                    .file_name()
                    .context("Reserved media filename")?
                    .to_string_lossy()
                    .into_owned();
                let temporary_name = format!(".{fingerprint}.tmp");
                validate_reserved_write_directory(parent, &final_name, &temporary_name)?;
                remove_if_present(&target)?;
                remove_if_present(&parent.join(temporary_name))?;
                match std::fs::remove_dir(parent) {
                    Ok(()) => Ok(()),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    Err(error) => Err(error.into()),
                }
            })
        })
        .await
        .map_err(|error| HookError::Failed(error.to_string()))?
        .map_err(|error| HookError::Failed(error.to_string()))
    }

    async fn get(&self, request: MediaReadRequest) -> HookResult<StoredMediaBytes> {
        let key = request.key.key;
        let metadata = request.metadata;
        let bytes = storage_layout::read_media_file(&self.media_root, &key).map_err(|err| {
            if is_not_found_error(&err) {
                HookError::NotFound(key.clone())
            } else {
                HookError::Failed(err.to_string())
            }
        })?;

        Ok(StoredMediaBytes {
            metadata: MediaObjectMetadata {
                content_type: content_type_from_key(&key).to_string(),
                filename: filename_from_key(&key),
                content_length: Some(bytes.len() as u64),
                sha256: None,
                tenant_id: tenant_id_from_metadata(&metadata),
                attributes: metadata,
            },
            bytes,
        })
    }

    async fn put_file(
        &self,
        request: MediaWriteRequest,
        path: PathBuf,
        content_length: u64,
    ) -> HookResult<StoredMediaObject> {
        use sha2::{Digest, Sha256};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let write = async {
            anyhow::ensure!(content_length > 0, "Audio payload cannot be empty");
            let mut source = tokio::fs::File::open(path).await?;
            anyhow::ensure!(
                source.metadata().await?.len() == content_length,
                "Media source length mismatch"
            );
            let (group, namespace) = local_namespace(&request.namespace, &request.metadata);
            // File names are internal UUIDs. Never interpret caller input as a path.
            let extension = match request.content_type.as_str() {
                "audio/wav" => "wav",
                "audio/mpeg" => "mp3",
                "audio/flac" => "flac",
                "audio/ogg" => "ogg",
                "audio/pcm" => "pcm",
                _ => "bin",
            };
            let namespace = sanitize_namespace(&namespace);
            let key = format!(
                "{}/{}/{}.{}",
                group.as_dir(),
                namespace,
                uuid::Uuid::new_v4(),
                extension
            );
            let target = storage_layout::resolve_media_path(&self.media_root, &key)?;
            let parent = target.parent().context("Media parent directory")?;
            tokio::fs::create_dir_all(parent).await?;
            let temporary = tempfile::NamedTempFile::new_in(parent)?;
            let mut output = tokio::fs::File::from_std(temporary.reopen()?);
            let mut buffer = vec![0; 64 * 1024];
            let mut hash = Sha256::new();
            let mut total = 0u64;
            loop {
                let count = source.read(&mut buffer).await?;
                if count == 0 {
                    break;
                }
                total = total
                    .checked_add(count as u64)
                    .context("Media size overflow")?;
                anyhow::ensure!(total <= content_length, "Media source grew during upload");
                output.write_all(&buffer[..count]).await?;
                hash.update(&buffer[..count]);
            }
            anyhow::ensure!(
                total == content_length,
                "Media source shortened during upload"
            );
            output.flush().await?;
            output.sync_all().await?;
            drop(output);
            // The temporary owner removes incomplete output on every error/cancellation.
            persist_local_tempfile_noclobber(temporary, &target)?;
            Ok::<_, anyhow::Error>((key, format!("{:x}", hash.finalize())))
        }
        .await
        .map_err(|err| HookError::Failed(err.to_string()))?;
        Ok(StoredMediaObject {
            key: MediaObjectKey::new(write.0),
            metadata: MediaObjectMetadata {
                content_type: request.content_type,
                filename: request.preferred_filename,
                content_length: Some(content_length),
                sha256: Some(write.1),
                tenant_id: tenant_id_from_metadata(&request.metadata),
                attributes: request.metadata,
            },
        })
    }

    async fn get_stream(
        &self,
        request: MediaReadRequest,
    ) -> HookResult<izwi_hooks::StoredMediaStream> {
        let key = request.key.key;
        let path = storage_layout::resolve_media_path(&self.media_root, &key)
            .map_err(|err| HookError::Failed(err.to_string()))?;
        let file = tokio::fs::File::open(path).await.map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                HookError::NotFound(key.clone())
            } else {
                HookError::Failed(err.to_string())
            }
        })?;
        let content_length = file
            .metadata()
            .await
            .map_err(|err| HookError::Failed(err.to_string()))?
            .len();
        Ok(izwi_hooks::StoredMediaStream {
            reader: Box::pin(file),
            metadata: MediaObjectMetadata {
                content_type: content_type_from_key(&key).to_string(),
                filename: filename_from_key(&key),
                content_length: Some(content_length),
                sha256: None,
                tenant_id: tenant_id_from_metadata(&request.metadata),
                attributes: request.metadata,
            },
        })
    }

    async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
        storage_layout::delete_media_file(&self.media_root, Some(&request.key.key))
            .map_err(|err| HookError::Failed(err.to_string()))?;
        if let Some(parent) = reserved_write_parent_for_key(&self.media_root, &request.key.key) {
            match std::fs::remove_dir(parent) {
                Ok(()) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                    ) => {}
                Err(error) => return Err(HookError::Failed(error.to_string())),
            }
        }
        Ok(())
    }
}

fn validate_reserved_write(request: &MediaReservedWriteRequest, bytes: &[u8]) -> HookResult<()> {
    if request.version != MEDIA_RESERVED_WRITE_VERSION
        || uuid::Uuid::parse_str(&request.write_id).is_err()
        || request.request.record_id != request.write_id
        || request.content_length != bytes.len() as u64
        || !valid_sha256(&request.sha256)
        || sha256_hex_bytes(bytes) != request.sha256
        || !reserved_content_type_round_trips(&request.request.content_type)
    {
        return Err(HookError::Failed(
            "Invalid reserved media write identity".into(),
        ));
    }
    ensure_reserved_write_live(request.expires_at_unix_ms)
        .map_err(|error| HookError::Failed(error.to_string()))
}

fn validate_reserved_recovery(request: &MediaReservedWriteRecoveryRequest) -> HookResult<()> {
    if request.version != MEDIA_RESERVED_WRITE_VERSION
        || uuid::Uuid::parse_str(&request.write_id).is_err()
        || request.request.record_id != request.write_id
        || request.content_length == 0
        || !valid_sha256(&request.sha256)
    {
        return Err(HookError::Failed(
            "Invalid reserved media recovery identity".into(),
        ));
    }
    Ok(())
}

fn ensure_reserved_write_live(expires_at_unix_ms: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        current_unix_millis() < expires_at_unix_ms,
        "Reserved media write expired before publication"
    );
    Ok(())
}

fn current_unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn reserved_local_key(request: &MediaWriteRequest, write_id: &str, fingerprint: &str) -> String {
    let extension = storage_layout::media_extension_for_content_type(&request.content_type);
    format!("generated/reserved-writes/{write_id}/{fingerprint}.{extension}")
}

fn reserved_content_type_round_trips(content_type: &str) -> bool {
    let extension = storage_layout::media_extension_for_content_type(content_type);
    storage_layout::content_type_from_media_path(&format!("reserved.{extension}")) == content_type
}

fn with_reserved_write_lock<T>(
    media_root: &Path,
    write_id: &str,
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    use fs2::FileExt as _;
    let write_uuid = uuid::Uuid::parse_str(write_id)?;
    let lock_root = media_root.join(".reserved-write-locks");
    std::fs::create_dir_all(&lock_root)?;
    // A fixed shard set stays bounded and avoids the inode race caused by
    // unlinking a per-write lock file while another process is waiting on it.
    let lock_path = lock_root.join(format!("{:02x}.lock", write_uuid.as_bytes()[0]));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    lock.lock_exclusive()?;
    let result = operation();
    let unlock = lock.unlock();
    drop(lock);
    result.and_then(|value| unlock.map(|_| value).map_err(Into::into))
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn reserved_write_fingerprint(
    request: &MediaWriteRequest,
    content_length: u64,
    content_sha256: &str,
) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"izwi-media-reserved-write-v1\0");
    match &request.namespace {
        MediaNamespace::TranscriptionUpload => digest.update(b"transcription-upload"),
        MediaNamespace::DiarizationUpload => digest.update(b"diarization-upload"),
        MediaNamespace::GeneratedSpeech => digest.update(b"generated-speech"),
        MediaNamespace::SavedVoice => digest.update(b"saved-voice"),
        MediaNamespace::ChatMedia => digest.update(b"chat-media"),
        MediaNamespace::Export => digest.update(b"export"),
        MediaNamespace::Other(value) => {
            digest.update(b"other");
            update_fingerprint_field(&mut digest, value.as_bytes());
        }
    }
    update_fingerprint_field(&mut digest, request.record_id.as_bytes());
    match request.preferred_filename.as_deref() {
        Some(filename) => {
            digest.update([1]);
            update_fingerprint_field(&mut digest, filename.as_bytes());
        }
        None => digest.update([0]),
    }
    update_fingerprint_field(&mut digest, request.content_type.as_bytes());
    for (key, value) in &request.metadata {
        update_fingerprint_field(&mut digest, key.as_bytes());
        update_fingerprint_field(&mut digest, value.as_bytes());
    }
    digest.update(content_length.to_be_bytes());
    update_fingerprint_field(&mut digest, content_sha256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn update_fingerprint_field(digest: &mut sha2::Sha256, value: &[u8]) {
    use sha2::Digest as _;
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_reserved_write_directory(
    directory: &Path,
    final_name: &str,
    temporary_name: &str,
) -> anyhow::Result<()> {
    match std::fs::read_dir(directory) {
        Ok(entries) => {
            let mut count = 0usize;
            for entry in entries {
                count += 1;
                anyhow::ensure!(count <= 2, "Reserved media write directory is not bounded");
                let name = entry?.file_name();
                anyhow::ensure!(
                    name == std::ffi::OsStr::new(final_name)
                        || name == std::ffi::OsStr::new(temporary_name),
                    "Reserved media write ID was reused for different bytes or metadata"
                );
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_if_present(path: &Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> anyhow::Result<()> {
    std::fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

fn reserved_write_parent_for_key(media_root: &Path, key: &str) -> Option<PathBuf> {
    let suffix = key.strip_prefix("generated/reserved-writes/")?;
    let mut segments = suffix.split('/');
    let write_id = segments.next()?;
    let filename = segments.next()?;
    if segments.next().is_some() || filename.is_empty() || uuid::Uuid::parse_str(write_id).is_err()
    {
        return None;
    }
    storage_layout::resolve_media_path(media_root, key)
        .ok()?
        .parent()
        .map(Path::to_path_buf)
}

fn file_matches_bytes(path: &Path, expected: &[u8]) -> anyhow::Result<bool> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path)?;
    if file.metadata()?.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut offset = 0usize;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(offset == expected.len());
        }
        if expected.get(offset..offset + count) != Some(&buffer[..count]) {
            return Ok(false);
        }
        offset += count;
    }
}

fn tenant_id_from_metadata(metadata: &HookMetadata) -> Option<String> {
    metadata.get("tenant_id").cloned()
}

fn persist_local_tempfile_noclobber(
    temporary: tempfile::NamedTempFile,
    target: &Path,
) -> anyhow::Result<()> {
    persist_local_tempfile_noclobber_with(temporary, target, |temporary, target| {
        temporary.persist_noclobber(target)
    })
}

fn persist_local_tempfile_noclobber_with(
    temporary: tempfile::NamedTempFile,
    target: &Path,
    persist: impl FnOnce(
        tempfile::NamedTempFile,
        &Path,
    ) -> Result<std::fs::File, tempfile::PersistError>,
) -> anyhow::Result<()> {
    match persist(temporary, target) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == std::io::ErrorKind::PermissionDenied => {
            let rename_error = error.error;
            let temporary = error.file;
            std::fs::hard_link(temporary.path(), target).with_context(|| {
                format!(
                    "Failed to publish media with a hard link after atomic rename was denied: {rename_error}"
                )
            })?;
            // Dropping the owner unlinks only the temporary name; the target hard link remains.
            drop(temporary);
            Ok(())
        }
        Err(error) => Err(error.into()),
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
    storage_layout::content_type_from_media_path(key)
}

fn filename_from_key(key: &str) -> Option<String> {
    std::path::Path::new(key)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
}

fn is_not_found_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use izwi_hooks::{DatabaseProvider, DatabaseProviderDecision, MediaStorageResolver};
    use sea_orm::DbBackend;
    use std::io::Write;
    use std::time::Duration;

    #[test]
    fn local_media_publish_falls_back_when_atomic_rename_is_denied() {
        let directory = tempfile::tempdir().expect("temp dir");
        let target = directory.path().join("published.wav");
        let mut temporary =
            tempfile::NamedTempFile::new_in(directory.path()).expect("temporary file");
        temporary.write_all(b"generated audio").expect("write");
        temporary.as_file().sync_all().expect("sync");
        let temporary_path = temporary.path().to_path_buf();

        persist_local_tempfile_noclobber_with(temporary, &target, |file, _| {
            Err(tempfile::PersistError {
                error: std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "simulated seccomp denial",
                ),
                file,
            })
        })
        .expect("permission denial should use the no-clobber hard-link fallback");

        assert_eq!(
            std::fs::read(&target).expect("published bytes"),
            b"generated audio"
        );
        assert!(!temporary_path.exists());
    }

    #[test]
    fn local_media_publish_fallback_does_not_replace_an_existing_target() {
        let directory = tempfile::tempdir().expect("temp dir");
        let target = directory.path().join("published.wav");
        std::fs::write(&target, b"existing audio").expect("existing target");
        let mut temporary =
            tempfile::NamedTempFile::new_in(directory.path()).expect("temporary file");
        temporary.write_all(b"new audio").expect("write");

        let error = persist_local_tempfile_noclobber_with(temporary, &target, |file, _| {
            Err(tempfile::PersistError {
                error: std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "simulated seccomp denial",
                ),
                file,
            })
        })
        .expect_err("fallback must preserve no-clobber behavior");

        assert_eq!(
            std::fs::read(&target).expect("existing bytes"),
            b"existing audio"
        );
        assert!(error.to_string().contains("hard link"), "{error:#}");
    }

    #[tokio::test]
    async fn local_reserved_write_recovery_fences_late_publication() {
        let directory = tempfile::tempdir().unwrap();
        let provider = LocalMediaStorageProvider::new(directory.path().join("media"));
        let write_id = uuid::Uuid::new_v4().to_string();
        let expires_at_unix_ms = current_unix_millis() + 500;
        let request = MediaWriteRequest {
            namespace: MediaNamespace::Other("artifact-store".into()),
            record_id: write_id.clone(),
            preferred_filename: Some("artifact.bin".into()),
            content_type: "application/octet-stream".into(),
            metadata: HookMetadata::new(),
        };
        let stored = provider
            .put_reserved(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id: write_id.clone(),
                    expires_at_unix_ms,
                    content_length: 8,
                    sha256: sha256_hex_bytes(b"reserved"),
                    request: request.clone(),
                },
                b"reserved".to_vec(),
            )
            .await
            .unwrap();
        let mut mismatched_request = request.clone();
        mismatched_request
            .metadata
            .insert("tenant_id".into(), "different-tenant".into());
        assert!(provider
            .put_reserved(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id: write_id.clone(),
                    expires_at_unix_ms,
                    content_length: 8,
                    sha256: sha256_hex_bytes(b"reserved"),
                    request: mismatched_request,
                },
                b"reserved".to_vec(),
            )
            .await
            .is_err());
        tokio::time::sleep(Duration::from_millis(510)).await;
        provider
            .recover_reserved_write(MediaReservedWriteRecoveryRequest {
                version: MEDIA_RESERVED_WRITE_VERSION,
                write_id: write_id.clone(),
                expires_at_unix_ms,
                content_length: 8,
                sha256: sha256_hex_bytes(b"reserved"),
                request: request.clone(),
                storage_key: Some(stored.key),
            })
            .await
            .unwrap();
        assert!(provider
            .put_reserved(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id,
                    expires_at_unix_ms,
                    content_length: 8,
                    sha256: sha256_hex_bytes(b"reserved"),
                    request,
                },
                b"reserved".to_vec(),
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn file_upload_round_trips_stream_with_checksum_and_rejects_wrong_length() {
        use sha2::{Digest, Sha256};
        use tokio::io::AsyncReadExt;
        let directory = tempfile::tempdir().unwrap();
        let provider = LocalMediaStorageProvider::new(directory.path().join("media"));
        let source = directory.path().join("source.wav");
        let bytes = vec![42; 128 * 1024 + 3];
        tokio::fs::write(&source, &bytes).await.unwrap();
        let request = MediaWriteRequest {
            namespace: MediaNamespace::GeneratedSpeech,
            record_id: "../unsafe".into(),
            preferred_filename: Some("voice.wav".into()),
            content_type: "audio/wav".into(),
            metadata: HookMetadata::new(),
        };
        assert!(provider
            .put_file(request.clone(), source.clone(), bytes.len() as u64 + 1)
            .await
            .is_err());
        let stored = provider
            .put_file(request, source, bytes.len() as u64)
            .await
            .unwrap();
        assert_eq!(
            stored.metadata.sha256,
            Some(format!("{:x}", Sha256::digest(&bytes)))
        );
        assert!(!stored.key.key.contains(".."));
        let mut stream = provider
            .get_stream(MediaReadRequest {
                key: stored.key.clone(),
                metadata: HookMetadata::new(),
            })
            .await
            .unwrap();
        assert_eq!(stream.metadata.content_length, Some(bytes.len() as u64));
        let mut read = Vec::new();
        stream.reader.read_to_end(&mut read).await.unwrap();
        assert_eq!(read, bytes);
        provider
            .delete(MediaDeleteRequest {
                key: stored.key,
                metadata: HookMetadata::new(),
            })
            .await
            .unwrap();
    }

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
        assert_eq!(context.local_media_root(), Some(&media_dir));

        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn provider_media_storage_is_not_marked_as_local_listable_storage() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let provider = Arc::new(LocalMediaStorageProvider::new(
            temp_dir.path().to_path_buf(),
        )) as Arc<dyn MediaStorageProvider>;
        let mut hooks = EnterpriseHooks::noop();
        hooks.media_storage = Arc::new(StaticMediaStorageResolver {
            decision: MediaStorageProviderDecision::UseProvider(provider),
        });

        let resolved = resolve_media_storage(&hooks)
            .await
            .expect("media storage resolves");

        assert!(resolved.local_media_root.is_none());
    }

    #[tokio::test]
    async fn local_media_provider_round_trips_bytes() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let provider = LocalMediaStorageProvider::new(temp_dir.path().to_path_buf());
        let mut metadata = HookMetadata::new();
        metadata.insert("route_kind".to_string(), "tts".to_string());

        let object = provider
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

        assert_eq!(object.key.key, "generated/speech/tts/record-1.wav");

        let stored = provider
            .get(MediaReadRequest {
                key: object.key.clone(),
                metadata: HookMetadata::new(),
            })
            .await
            .expect("read media");
        assert_eq!(stored.bytes, b"audio");
        assert_eq!(stored.metadata.content_type, "audio/wav");

        provider
            .delete(MediaDeleteRequest {
                key: object.key,
                metadata: HookMetadata::new(),
            })
            .await
            .expect("delete media");
    }

    #[tokio::test]
    async fn local_media_provider_preserves_image_and_video_content_types() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let provider = LocalMediaStorageProvider::new(temp_dir.path().to_path_buf());
        let image_path = temp_dir.path().join("images/example.png");
        let video_path = temp_dir.path().join("videos/example.mp4");
        std::fs::create_dir_all(image_path.parent().expect("image parent")).expect("image dir");
        std::fs::create_dir_all(video_path.parent().expect("video parent")).expect("video dir");
        std::fs::write(&image_path, b"image").expect("image file");
        std::fs::write(&video_path, b"video").expect("video file");

        let image = provider
            .get(MediaReadRequest {
                key: MediaObjectKey::new("images/example.png"),
                metadata: HookMetadata::new(),
            })
            .await
            .expect("read image");
        let video = provider
            .get(MediaReadRequest {
                key: MediaObjectKey::new("videos/example.mp4"),
                metadata: HookMetadata::new(),
            })
            .await
            .expect("read video");

        assert_eq!(image.metadata.content_type, "image/png");
        assert_eq!(video.metadata.content_type, "video/mp4");
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

    #[test]
    fn supported_database_backends_accept_portable_store_sql() {
        assert!(validate_supported_database_backend(&DatabaseBackend::Sqlite).is_ok());
        assert!(validate_supported_database_backend(&DatabaseBackend::Postgres).is_ok());
        assert!(validate_supported_database_backend(&DatabaseBackend::Mysql).is_ok());

        let error =
            validate_supported_database_backend(&DatabaseBackend::Other("unsupported".to_string()))
                .expect_err("unknown backend should be rejected");

        assert!(
            error
                .to_string()
                .contains("is not supported by this runtime"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn provider_managed_database_requires_schema_contract() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let connection = db::sqlite::connect_path(&temp_dir.path().join("enterprise.sqlite3"))
            .await
            .expect("sqlite connection");
        let mut hooks = EnterpriseHooks::noop();
        hooks.database = Arc::new(StaticDatabaseProvider {
            decision: DatabaseProviderDecision::UseConnection(DatabaseConnectionDecision {
                connection,
                backend: DatabaseBackend::Sqlite,
                migration_mode: DatabaseMigrationMode::ProviderManaged,
                metadata: HookMetadata::new(),
            }),
        });

        let error = match PersistenceContext::resolve(&hooks).await {
            Ok(_) => panic!("empty provider-managed schema should fail"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("Enterprise database schema is incomplete"),
            "{error}"
        );
        assert!(error.to_string().contains("chat_threads"), "{error}");
    }

    #[tokio::test]
    async fn provider_managed_database_accepts_valid_schema_contract() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let connection = db::sqlite::connect_path(&temp_dir.path().join("enterprise.sqlite3"))
            .await
            .expect("sqlite connection");
        Migrator::up(&connection)
            .await
            .expect("provider-managed test schema");

        let mut hooks = EnterpriseHooks::noop();
        hooks.database = Arc::new(StaticDatabaseProvider {
            decision: DatabaseProviderDecision::UseConnection(DatabaseConnectionDecision {
                connection,
                backend: DatabaseBackend::Sqlite,
                migration_mode: DatabaseMigrationMode::ProviderManaged,
                metadata: HookMetadata::new(),
            }),
        });

        let context = PersistenceContext::resolve(&hooks)
            .await
            .expect("provider-managed schema should validate");
        assert_eq!(context.database.backend(), &DatabaseBackend::Sqlite);
        assert_eq!(
            context.database.migration_mode(),
            &DatabaseMigrationMode::ProviderManaged
        );
    }

    #[tokio::test]
    async fn local_media_provider_reports_missing_objects_as_not_found() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let provider = Arc::new(LocalMediaStorageProvider::new(
            temp_dir.path().to_path_buf(),
        )) as Arc<dyn MediaStorageProvider>;

        let error = read_media_object(&provider, "generated/missing/object.wav")
            .await
            .expect_err("missing object should fail");

        assert!(error.is_not_found(), "{error}");
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

    #[derive(Clone)]
    struct StaticMediaStorageResolver {
        decision: MediaStorageProviderDecision,
    }

    #[async_trait::async_trait]
    impl MediaStorageResolver for StaticMediaStorageResolver {
        async fn resolve_media_storage(
            &self,
            _request: &MediaStorageProviderRequest,
        ) -> HookResult<MediaStorageProviderDecision> {
            Ok(self.decision.clone())
        }
    }
}

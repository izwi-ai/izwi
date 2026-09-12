//! Tenant-scoped opaque artifact access over the configured media provider.
//!
//! This module is a storage foundation, not a public route. Callers exchange
//! [`ArtifactId`] values while provider keys remain private to this facade and
//! the existing durable `media_assets` table.

use crate::batch_runtime::store::{BatchRuntimeStore, NewMediaAsset};
use izwi_hooks::{
    HookError, HookMetadata, MediaDeleteRequest, MediaNamespace, MediaObjectKey, MediaReadRequest,
    MediaStorageProvider, MediaWriteRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::io::AsyncReadExt;

const ARTIFACT_METADATA_VERSION: u64 = 1;
const MAX_TENANT_ID_BYTES: usize = 128;
const MAX_ARTIFACT_ID_BYTES: usize = 64;
const ABSOLUTE_MAX_OBJECT_BYTES: u64 = 1024 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactTenant(String);

impl ArtifactTenant {
    pub fn parse(value: impl Into<String>) -> Result<Self, ArtifactStoreError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_TENANT_ID_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-.:@".contains(&byte))
        {
            return Err(ArtifactStoreError::InvalidInput("invalid tenant identity"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ArtifactId(String);

impl ArtifactId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ArtifactStoreError> {
        let value = value.into();
        if value.len() > MAX_ARTIFACT_ID_BYTES || uuid::Uuid::parse_str(&value).is_err() {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid artifact identity",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ArtifactId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRetention {
    Ephemeral,
    Job,
    Durable,
}

impl ArtifactRetention {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::Ephemeral => "artifact_ephemeral",
            Self::Job => "artifact_job",
            Self::Durable => "artifact_durable",
        }
    }

    fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "artifact_ephemeral" => Some(Self::Ephemeral),
            "artifact_job" => Some(Self::Job),
            "artifact_durable" => Some(Self::Durable),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactStoreLimits {
    pub max_object_bytes: u64,
    pub max_content_type_bytes: usize,
    pub max_filename_bytes: usize,
}

impl ArtifactStoreLimits {
    pub fn validate(self) -> Result<Self, ArtifactStoreError> {
        if self.max_object_bytes == 0 || self.max_object_bytes > ABSOLUTE_MAX_OBJECT_BYTES {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid artifact object-size limit",
            ));
        }
        if self.max_content_type_bytes == 0 || self.max_content_type_bytes > 256 {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid artifact content-type limit",
            ));
        }
        if self.max_filename_bytes == 0 || self.max_filename_bytes > 1024 {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid artifact filename limit",
            ));
        }
        Ok(self)
    }
}

impl Default for ArtifactStoreLimits {
    fn default() -> Self {
        Self {
            max_object_bytes: 64 * 1024 * 1024,
            max_content_type_bytes: 128,
            max_filename_bytes: 255,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactWrite {
    pub content_type: String,
    pub filename: Option<String>,
    pub bytes: Vec<u8>,
    pub retention: ArtifactRetention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDescriptor {
    pub id: ArtifactId,
    pub content_type: String,
    pub filename: Option<String>,
    pub size_bytes: u64,
    pub sha256: String,
    pub retention: ArtifactRetention,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactBytes {
    pub descriptor: ArtifactDescriptor,
    pub bytes: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactStoreError {
    #[error("invalid artifact request: {0}")]
    InvalidInput(&'static str),
    #[error("artifact exceeds configured size limit")]
    TooLarge,
    #[error("artifact not found")]
    NotFound,
    #[error("artifact metadata is invalid: {0}")]
    InvalidMetadata(&'static str),
    #[error("artifact integrity validation failed: {0}")]
    Integrity(&'static str),
    #[error("artifact metadata store failed")]
    Metadata(#[source] anyhow::Error),
    #[error("artifact storage provider failed")]
    Provider,
    #[error("artifact read failed")]
    Read,
    #[error("artifact was hidden, but physical deletion must be retried")]
    DeleteIncomplete,
}

#[derive(Clone)]
pub struct ArtifactStore {
    metadata: Arc<BatchRuntimeStore>,
    provider: Arc<dyn MediaStorageProvider>,
    limits: ArtifactStoreLimits,
}

impl ArtifactStore {
    pub fn new(
        metadata: Arc<BatchRuntimeStore>,
        provider: Arc<dyn MediaStorageProvider>,
        limits: ArtifactStoreLimits,
    ) -> Result<Self, ArtifactStoreError> {
        Ok(Self {
            metadata,
            provider,
            limits: limits.validate()?,
        })
    }

    pub async fn put(
        &self,
        tenant: &ArtifactTenant,
        write: ArtifactWrite,
    ) -> Result<ArtifactDescriptor, ArtifactStoreError> {
        self.validate_write(&write)?;
        let digest = sha256_hex(&write.bytes);
        let size_bytes = write.bytes.len() as u64;
        let record_id = uuid::Uuid::new_v4().to_string();
        let request_metadata = tenant_metadata(tenant);
        let stored = self
            .provider
            .put(
                MediaWriteRequest {
                    namespace: MediaNamespace::Other("artifact-store".to_string()),
                    record_id,
                    preferred_filename: write.filename.clone(),
                    content_type: write.content_type.clone(),
                    metadata: request_metadata.clone(),
                },
                write.bytes,
            )
            .await
            .map_err(|_| ArtifactStoreError::Provider)?;

        if let Err(error) = validate_stored_write(
            tenant,
            &write.content_type,
            digest.as_str(),
            size_bytes,
            stored.metadata.content_length,
            stored.metadata.sha256.as_deref(),
            stored.metadata.tenant_id.as_deref(),
            stored.metadata.content_type.as_str(),
            self.limits.max_object_bytes,
        ) {
            self.compensate_delete(stored.key, request_metadata).await;
            return Err(error);
        }

        let metadata_json = serde_json::json!({
            "artifact_store": {
                "version": ARTIFACT_METADATA_VERSION,
                "tenant_id": tenant.as_str(),
            }
        });
        let asset = match self
            .metadata
            .create_media_asset(NewMediaAsset {
                asset_kind: "opaque_artifact".to_string(),
                storage_namespace: "artifact_store_v1".to_string(),
                storage_key: stored.key.key.clone(),
                content_type: write.content_type,
                filename: write.filename,
                size_bytes,
                sha256: Some(digest),
                duration_secs: None,
                sample_rate_hz: None,
                channel_count: None,
                peak_amplitude: None,
                rms_amplitude: None,
                source_asset_id: None,
                canonical_profile_version: None,
                scan_status: "not_required".to_string(),
                retention_policy: write.retention.as_db_value().to_string(),
                metadata_json,
            })
            .await
        {
            Ok(asset) => asset,
            Err(error) => {
                self.compensate_delete(stored.key, request_metadata).await;
                return Err(ArtifactStoreError::Metadata(error));
            }
        };

        descriptor_from_asset(&asset)
    }

    pub async fn stat(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<ArtifactDescriptor, ArtifactStoreError> {
        let asset = self.resolve_active(tenant, id).await?;
        descriptor_from_asset(&asset)
    }

    pub async fn read(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<ArtifactBytes, ArtifactStoreError> {
        let asset = self.resolve_active(tenant, id).await?;
        if asset.size_bytes > self.limits.max_object_bytes {
            return Err(ArtifactStoreError::TooLarge);
        }
        let expected_digest = validate_digest(asset.sha256.as_deref())?;
        let mut stream = self
            .provider
            .get_stream(MediaReadRequest {
                key: MediaObjectKey::new(asset.storage_key.clone()),
                metadata: tenant_metadata(tenant),
            })
            .await
            .map_err(map_provider_read_error)?;

        validate_stored_read_metadata(tenant, &asset, &stream.metadata, &expected_digest)?;

        let initial_capacity = usize::try_from(asset.size_bytes)
            .unwrap_or(0)
            .min(READ_CHUNK_BYTES);
        let mut bytes = Vec::with_capacity(initial_capacity);
        let mut hash = Sha256::new();
        let mut total = 0u64;
        let mut chunk = [0u8; READ_CHUNK_BYTES];
        loop {
            let read = stream
                .reader
                .read(&mut chunk)
                .await
                .map_err(|_| ArtifactStoreError::Read)?;
            if read == 0 {
                break;
            }
            total = total
                .checked_add(read as u64)
                .ok_or(ArtifactStoreError::TooLarge)?;
            if total > self.limits.max_object_bytes || total > asset.size_bytes {
                return Err(ArtifactStoreError::TooLarge);
            }
            hash.update(&chunk[..read]);
            bytes.extend_from_slice(&chunk[..read]);
        }

        if total != asset.size_bytes {
            return Err(ArtifactStoreError::Integrity("size mismatch"));
        }
        let actual_digest = format!("{:x}", hash.finalize());
        if actual_digest != expected_digest {
            return Err(ArtifactStoreError::Integrity("digest mismatch"));
        }

        Ok(ArtifactBytes {
            descriptor: descriptor_from_asset(&asset)?,
            bytes,
        })
    }

    /// Hide an artifact transactionally, then attempt idempotent physical cleanup.
    ///
    /// A provider outage after the tombstone returns `DeleteIncomplete`; later
    /// reads still fail closed; repeating deletion or a garbage collector may
    /// retry object cleanup without restoring access.
    pub async fn delete(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<bool, ArtifactStoreError> {
        let asset = match self.resolve(tenant, id).await? {
            Some(asset) => asset,
            None => return Ok(false),
        };
        let newly_deleted = if asset.deleted_at.is_none() {
            self.metadata
                .tombstone_media_asset(id.as_str())
                .await
                .map_err(ArtifactStoreError::Metadata)?
        } else {
            false
        };

        match self
            .provider
            .delete(MediaDeleteRequest {
                key: MediaObjectKey::new(asset.storage_key),
                metadata: tenant_metadata(tenant),
            })
            .await
        {
            Ok(()) | Err(HookError::NotFound(_)) => Ok(newly_deleted),
            Err(_) => Err(ArtifactStoreError::DeleteIncomplete),
        }
    }

    fn validate_write(&self, write: &ArtifactWrite) -> Result<(), ArtifactStoreError> {
        if write.bytes.is_empty() {
            return Err(ArtifactStoreError::InvalidInput("empty artifact"));
        }
        if write.bytes.len() as u64 > self.limits.max_object_bytes {
            return Err(ArtifactStoreError::TooLarge);
        }
        validate_content_type(&write.content_type, self.limits.max_content_type_bytes)?;
        if let Some(filename) = &write.filename {
            if filename.is_empty()
                || filename.len() > self.limits.max_filename_bytes
                || filename.chars().any(char::is_control)
                || filename.contains('/')
                || filename.contains('\\')
                || matches!(filename.as_str(), "." | "..")
            {
                return Err(ArtifactStoreError::InvalidInput(
                    "invalid artifact filename",
                ));
            }
        }
        Ok(())
    }

    async fn resolve_active(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<crate::batch_runtime::types::MediaAsset, ArtifactStoreError> {
        match self.resolve(tenant, id).await? {
            Some(asset) if asset.deleted_at.is_none() => Ok(asset),
            _ => Err(ArtifactStoreError::NotFound),
        }
    }

    async fn resolve(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<Option<crate::batch_runtime::types::MediaAsset>, ArtifactStoreError> {
        let Some(asset) = self
            .metadata
            .get_media_asset(id.as_str())
            .await
            .map_err(ArtifactStoreError::Metadata)?
        else {
            return Ok(None);
        };
        if asset.asset_kind != "opaque_artifact" || asset.storage_namespace != "artifact_store_v1" {
            return Ok(None);
        }
        if tenant_from_asset(&asset).as_deref() != Some(tenant.as_str()) {
            return Ok(None);
        }
        Ok(Some(asset))
    }

    async fn compensate_delete(&self, key: MediaObjectKey, metadata: HookMetadata) {
        let _ = self
            .provider
            .delete(MediaDeleteRequest { key, metadata })
            .await;
    }
}

fn tenant_metadata(tenant: &ArtifactTenant) -> HookMetadata {
    let mut metadata = HookMetadata::new();
    metadata.insert("tenant_id".to_string(), tenant.as_str().to_string());
    metadata
}

fn tenant_from_asset(asset: &crate::batch_runtime::types::MediaAsset) -> Option<String> {
    let metadata = asset.metadata_json.get("artifact_store")?;
    if metadata.get("version")?.as_u64()? != ARTIFACT_METADATA_VERSION {
        return None;
    }
    metadata.get("tenant_id")?.as_str().map(str::to_string)
}

fn descriptor_from_asset(
    asset: &crate::batch_runtime::types::MediaAsset,
) -> Result<ArtifactDescriptor, ArtifactStoreError> {
    let retention = ArtifactRetention::from_db_value(&asset.retention_policy).ok_or(
        ArtifactStoreError::InvalidMetadata("unknown retention policy"),
    )?;
    Ok(ArtifactDescriptor {
        id: ArtifactId::parse(asset.id.clone())?,
        content_type: asset.content_type.clone(),
        filename: asset.filename.clone(),
        size_bytes: asset.size_bytes,
        sha256: validate_digest(asset.sha256.as_deref())?,
        retention,
        created_at: asset.created_at,
        updated_at: asset.updated_at,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_stored_write(
    tenant: &ArtifactTenant,
    expected_content_type: &str,
    expected_digest: &str,
    expected_size: u64,
    content_length: Option<u64>,
    digest: Option<&str>,
    provider_tenant: Option<&str>,
    provider_content_type: &str,
    max_object_bytes: u64,
) -> Result<(), ArtifactStoreError> {
    let content_length = content_length.ok_or(ArtifactStoreError::Integrity(
        "provider omitted content length",
    ))?;
    if content_length == 0 || content_length > max_object_bytes {
        return Err(ArtifactStoreError::TooLarge);
    }
    if content_length != expected_size {
        return Err(ArtifactStoreError::Integrity("provider size mismatch"));
    }
    if provider_tenant != Some(tenant.as_str()) {
        return Err(ArtifactStoreError::Integrity("provider tenant mismatch"));
    }
    if provider_content_type != expected_content_type {
        return Err(ArtifactStoreError::Integrity(
            "provider content-type mismatch",
        ));
    }
    if let Some(digest) = digest {
        if validate_digest(Some(digest))? != expected_digest {
            return Err(ArtifactStoreError::Integrity("provider digest mismatch"));
        }
    }
    Ok(())
}

fn validate_stored_read_metadata(
    tenant: &ArtifactTenant,
    asset: &crate::batch_runtime::types::MediaAsset,
    metadata: &izwi_hooks::MediaObjectMetadata,
    expected_digest: &str,
) -> Result<(), ArtifactStoreError> {
    if metadata.tenant_id.as_deref() != Some(tenant.as_str()) {
        return Err(ArtifactStoreError::Integrity("provider tenant mismatch"));
    }
    if metadata.content_length != Some(asset.size_bytes) {
        return Err(ArtifactStoreError::Integrity("declared size mismatch"));
    }
    if metadata.content_type != asset.content_type {
        return Err(ArtifactStoreError::Integrity(
            "provider content-type mismatch",
        ));
    }
    if let Some(digest) = metadata.sha256.as_deref() {
        if validate_digest(Some(digest))? != expected_digest {
            return Err(ArtifactStoreError::Integrity("provider digest mismatch"));
        }
    }
    Ok(())
}

fn validate_content_type(value: &str, max_bytes: usize) -> Result<(), ArtifactStoreError> {
    let Some((kind, subtype)) = value.split_once('/') else {
        return Err(ArtifactStoreError::InvalidInput(
            "invalid artifact content type",
        ));
    };
    if value.len() > max_bytes
        || kind.is_empty()
        || subtype.is_empty()
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'/' | b'.' | b'+' | b'-'))
    {
        return Err(ArtifactStoreError::InvalidInput(
            "invalid artifact content type",
        ));
    }
    Ok(())
}

fn validate_digest(value: Option<&str>) -> Result<String, ArtifactStoreError> {
    let value = value.ok_or(ArtifactStoreError::InvalidMetadata("missing digest"))?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactStoreError::InvalidMetadata("invalid digest"));
    }
    Ok(value.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn map_provider_read_error(error: HookError) -> ArtifactStoreError {
    match error {
        HookError::NotFound(_) => ArtifactStoreError::NotFound,
        _ => ArtifactStoreError::Provider,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{db::StoreDatabase, persistence::LocalMediaStorageProvider};
    use izwi_hooks::{HookResult, MediaObjectMetadata, StoredMediaObject, StoredMediaStream};
    use std::{
        collections::BTreeMap,
        io::Cursor,
        sync::{
            atomic::{AtomicBool, Ordering},
            Mutex,
        },
    };
    use tempfile::TempDir;

    fn conformance_store(root: &TempDir, provider: Arc<dyn MediaStorageProvider>) -> ArtifactStore {
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(root.path().join("artifacts.sqlite3")),
        ));
        ArtifactStore::new(
            metadata,
            provider,
            ArtifactStoreLimits {
                max_object_bytes: 1024,
                ..Default::default()
            },
        )
        .expect("artifact store")
    }

    async fn assert_provider_conformance(root: &TempDir, provider: Arc<dyn MediaStorageProvider>) {
        let store = conformance_store(root, provider);
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let other = ArtifactTenant::parse("tenant-b").unwrap();
        let descriptor = store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "audio/wav".to_string(),
                    filename: Some("sample.wav".to_string()),
                    bytes: b"bounded artifact".to_vec(),
                    retention: ArtifactRetention::Job,
                },
            )
            .await
            .expect("write artifact");

        assert_eq!(descriptor.size_bytes, 16);
        assert_eq!(descriptor.retention, ArtifactRetention::Job);
        assert!(matches!(
            store.read(&other, &descriptor.id).await,
            Err(ArtifactStoreError::NotFound)
        ));
        let read = store
            .read(&tenant, &descriptor.id)
            .await
            .expect("read artifact");
        assert_eq!(read.bytes, b"bounded artifact");
        assert_eq!(read.descriptor, descriptor);

        let public_json = serde_json::to_string(&descriptor).unwrap();
        assert!(!public_json.contains("generated/"));
        assert!(!public_json.contains("private/object"));

        assert!(store.delete(&tenant, &descriptor.id).await.unwrap());
        assert!(matches!(
            store.read(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::NotFound)
        ));
        assert!(!store.delete(&tenant, &descriptor.id).await.unwrap());
    }

    #[tokio::test]
    async fn local_filesystem_provider_conforms_to_opaque_artifact_contract() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(LocalMediaStorageProvider::new(root.path().join("media")));
        assert_provider_conformance(&root, provider).await;
    }

    #[tokio::test]
    async fn deterministic_remote_like_provider_conforms_to_opaque_artifact_contract() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        assert_provider_conformance(&root, provider).await;
    }

    #[tokio::test]
    async fn read_rejects_provider_bytes_that_do_not_match_the_durable_digest() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = conformance_store(&root, provider.clone());
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let descriptor = store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "application/octet-stream".to_string(),
                    filename: None,
                    bytes: b"authentic".to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await
            .unwrap();
        provider.tamper_single_object(b"corrupt!!".to_vec());

        assert!(matches!(
            store.read(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::Integrity("digest mismatch"))
        ));
    }

    #[tokio::test]
    async fn upload_limit_fails_before_calling_the_provider() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = ArtifactStore::new(
            Arc::new(BatchRuntimeStore::initialize_with_database(
                StoreDatabase::new(root.path().join("artifacts.sqlite3")),
            )),
            provider.clone(),
            ArtifactStoreLimits {
                max_object_bytes: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();

        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: "audio/wav".to_string(),
                        filename: None,
                        bytes: vec![0; 5],
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::TooLarge)
        ));
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn filename_cannot_escape_provider_namespace() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = conformance_store(&root, provider.clone());
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();

        for filename in ["../secret", "nested/file.wav", "nested\\file.wav"] {
            assert!(matches!(
                store
                    .put(
                        &tenant,
                        ArtifactWrite {
                            content_type: "audio/wav".to_string(),
                            filename: Some(filename.to_string()),
                            bytes: b"artifact".to_vec(),
                            retention: ArtifactRetention::Ephemeral,
                        },
                    )
                    .await,
                Err(ArtifactStoreError::InvalidInput(
                    "invalid artifact filename"
                ))
            ));
        }
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn failed_physical_delete_remains_logically_deleted() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = conformance_store(&root, provider.clone());
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let descriptor = store
            .put(
                &tenant,
                ArtifactWrite {
                    content_type: "audio/wav".to_string(),
                    filename: None,
                    bytes: b"artifact".to_vec(),
                    retention: ArtifactRetention::Durable,
                },
            )
            .await
            .unwrap();
        provider.fail_deletes.store(true, Ordering::SeqCst);

        assert!(matches!(
            store.delete(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
        assert!(matches!(
            store.read(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::NotFound)
        ));
        assert_eq!(provider.object_count(), 1);
        provider.fail_deletes.store(false, Ordering::SeqCst);
        assert!(!store.delete(&tenant, &descriptor.id).await.unwrap());
        assert_eq!(provider.object_count(), 0);
    }

    #[derive(Clone)]
    struct MemoryObject {
        bytes: Vec<u8>,
        metadata: MediaObjectMetadata,
    }

    #[derive(Default)]
    struct MemoryMediaProvider {
        objects: Mutex<BTreeMap<String, MemoryObject>>,
        fail_deletes: AtomicBool,
    }

    impl MemoryMediaProvider {
        fn object_count(&self) -> usize {
            self.objects.lock().unwrap().len()
        }

        fn tamper_single_object(&self, bytes: Vec<u8>) {
            let mut objects = self.objects.lock().unwrap();
            let object = objects.values_mut().next().expect("one object");
            object.bytes = bytes;
            object.metadata.content_length = Some(object.bytes.len() as u64);
            object.metadata.sha256 = None;
        }
    }

    #[async_trait::async_trait]
    impl MediaStorageProvider for MemoryMediaProvider {
        async fn put(
            &self,
            request: MediaWriteRequest,
            bytes: Vec<u8>,
        ) -> HookResult<StoredMediaObject> {
            let key = format!("private/object/{}", request.record_id);
            let tenant_id = request.metadata.get("tenant_id").cloned();
            let metadata = MediaObjectMetadata {
                content_type: request.content_type,
                filename: request.preferred_filename,
                content_length: Some(bytes.len() as u64),
                sha256: Some(sha256_hex(&bytes)),
                tenant_id,
                attributes: request.metadata,
            };
            self.objects.lock().unwrap().insert(
                key.clone(),
                MemoryObject {
                    bytes,
                    metadata: metadata.clone(),
                },
            );
            Ok(StoredMediaObject {
                key: MediaObjectKey::new(key),
                metadata,
            })
        }

        async fn get(&self, request: MediaReadRequest) -> HookResult<izwi_hooks::StoredMediaBytes> {
            let object = self
                .objects
                .lock()
                .unwrap()
                .get(&request.key.key)
                .cloned()
                .ok_or_else(|| HookError::NotFound("object".to_string()))?;
            Ok(izwi_hooks::StoredMediaBytes {
                bytes: object.bytes,
                metadata: object.metadata,
            })
        }

        async fn get_stream(&self, request: MediaReadRequest) -> HookResult<StoredMediaStream> {
            let object = self
                .objects
                .lock()
                .unwrap()
                .get(&request.key.key)
                .cloned()
                .ok_or_else(|| HookError::NotFound("object".to_string()))?;
            Ok(StoredMediaStream {
                reader: Box::pin(Cursor::new(object.bytes)),
                metadata: object.metadata,
            })
        }

        async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
            if self.fail_deletes.load(Ordering::SeqCst) {
                return Err(HookError::Failed("injected delete failure".to_string()));
            }
            if self
                .objects
                .lock()
                .unwrap()
                .remove(&request.key.key)
                .is_some()
            {
                Ok(())
            } else {
                Err(HookError::NotFound("object".to_string()))
            }
        }
    }
}

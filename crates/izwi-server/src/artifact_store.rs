//! Tenant-scoped opaque artifact access over the configured media provider.
//!
//! This module is a storage foundation, not a public route. Callers exchange
//! [`ArtifactId`] values while provider keys remain private to this facade and
//! the existing durable `media_assets` table.

use crate::batch_runtime::store::{
    validate_artifact_cleanup_storage_key, ArtifactCleanupIntent, ArtifactCleanupReason,
    BatchRuntimeStore, NewMediaAsset, NewProviderWriteReservation, NewStageOutputArtifact,
    ProviderWriteReservation, ReservedStageArtifactPublication,
};
use crate::batch_runtime::types::{
    RuntimeArtifact, RuntimeArtifactKind, RuntimeArtifactRole, StageLease,
};
use crate::ids::new_uuid;
use izwi_hooks::{
    HookError, HookMetadata, MediaDeleteRequest, MediaNamespace, MediaObjectKey, MediaReadRequest,
    MediaReservedWriteRecoveryRequest, MediaReservedWriteRequest, MediaStorageProvider,
    MediaWriteRequest, MEDIA_RESERVED_WRITE_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;

const ARTIFACT_METADATA_VERSION: u64 = 1;
const MAX_TENANT_ID_BYTES: usize = 128;
const MAX_ARTIFACT_ID_BYTES: usize = 64;
const ABSOLUTE_MAX_OBJECT_BYTES: u64 = 1024 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 64 * 1024;
const PROVIDER_DELETE_TIMEOUT: Duration = Duration::from_secs(5);
const CLEANUP_BATCH_TIMEOUT: Duration = Duration::from_secs(10);
const PROVIDER_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const PROVIDER_WRITE_RESERVATION_LIFETIME: Duration = Duration::from_secs(60);

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

    /// Stable tenant identity for server-authored durable scheduling records.
    ///
    /// The public route stores only the authenticated tenant digest, never a
    /// caller-provided scope. Anonymous local requests share the explicit
    /// standalone namespace.
    pub(crate) fn from_scheduling_key(key: Option<[u8; 32]>) -> Self {
        match key {
            Some(key) => Self(key.iter().map(|byte| format!("{byte:02x}")).collect()),
            None => Self("anonymous".to_string()),
        }
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

#[derive(Debug, Clone)]
pub(crate) struct AttemptArtifactWrite {
    pub publication_key: String,
    pub artifact_kind: RuntimeArtifactKind,
    pub artifact_role: RuntimeArtifactRole,
    pub metadata_json: serde_json::Value,
    pub runtime_retention_policy: String,
    pub object: ArtifactWrite,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArtifactCleanupReport {
    pub inspected: usize,
    pub completed: usize,
    pub deferred: usize,
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
    #[error("artifact storage provider lacks reserved-write protocol v1")]
    ReservedWritesUnsupported,
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
    #[cfg(test)]
    reservation_lifetime: Duration,
}

struct StoredReservedArtifact {
    reservation: ProviderWriteReservation,
    storage_key: String,
    content_type: String,
    filename: Option<String>,
    size_bytes: u64,
    sha256: String,
    retention: ArtifactRetention,
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
            #[cfg(test)]
            reservation_lifetime: PROVIDER_WRITE_RESERVATION_LIFETIME,
        })
    }

    pub async fn put(
        &self,
        tenant: &ArtifactTenant,
        write: ArtifactWrite,
    ) -> Result<ArtifactDescriptor, ArtifactStoreError> {
        let stored = self.store_reserved_write(tenant, write).await?;
        let asset = match self
            .metadata
            .publish_reserved_opaque_artifact(
                &stored.reservation,
                opaque_media_asset(tenant, &stored),
            )
            .await
        {
            Ok(Some(asset)) => asset,
            Ok(None) => {
                self.abandon_write(
                    &stored.reservation,
                    "Provider write lost its publication fence",
                )
                .await;
                return Err(ArtifactStoreError::Metadata(anyhow::anyhow!(
                    "Provider write lost its publication fence"
                )));
            }
            Err(error) => {
                self.abandon_write(&stored.reservation, "Artifact metadata publication failed")
                    .await;
                return Err(ArtifactStoreError::Metadata(error));
            }
        };

        descriptor_from_asset(&asset)
    }

    pub(crate) async fn put_attempt_artifact(
        &self,
        tenant: &ArtifactTenant,
        lease: &StageLease,
        write: AttemptArtifactWrite,
        publication_progress: serde_json::Value,
    ) -> Result<RuntimeArtifact, ArtifactStoreError> {
        if lease.attempt_token.is_none() {
            return Err(ArtifactStoreError::InvalidInput(
                "stage lease lacks an attempt token",
            ));
        }
        let publication_key = write.publication_key.trim().to_string();
        if publication_key.is_empty()
            || publication_key.len() > 256
            || publication_key.chars().any(char::is_control)
        {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid attempt publication key",
            ));
        }
        if write.runtime_retention_policy.is_empty()
            || write.runtime_retention_policy.len() > 128
            || write.runtime_retention_policy.chars().any(char::is_control)
        {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid runtime retention policy",
            ));
        }
        let metadata_bytes = serde_json::to_vec(&write.metadata_json)
            .map_err(|_| ArtifactStoreError::InvalidInput("invalid artifact metadata"))?;
        if metadata_bytes.len() > 8 * 1024 {
            return Err(ArtifactStoreError::InvalidInput(
                "attempt artifact metadata exceeds 8 KiB",
            ));
        }

        let stored = self.store_reserved_write(tenant, write.object).await?;
        let media = opaque_media_asset(tenant, &stored);
        let artifact = NewStageOutputArtifact {
            publication_key,
            artifact_kind: write.artifact_kind,
            artifact_role: write.artifact_role,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: None,
            content_type: None,
            filename: None,
            size_bytes: None,
            sha256: None,
            metadata_json: write.metadata_json,
            retention_policy: write.runtime_retention_policy,
        };
        match self
            .metadata
            .publish_reserved_opaque_stage_artifact(
                &stored.reservation,
                lease,
                media,
                artifact,
                publication_progress,
            )
            .await
        {
            Ok(Some(ReservedStageArtifactPublication::Published { asset, artifact })) => {
                debug_assert_eq!(artifact.media_asset_id.as_deref(), Some(asset.id.as_str()));
                Ok(artifact)
            }
            Ok(Some(ReservedStageArtifactPublication::Existing(artifact))) => Ok(artifact),
            Ok(None) => Err(ArtifactStoreError::Metadata(anyhow::anyhow!(
                "Stage attempt lost ownership before artifact publication"
            ))),
            Err(error) => Err(ArtifactStoreError::Metadata(error)),
        }
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
        let newly_deleted = self
            .metadata
            .tombstone_media_asset_with_cleanup(
                id.as_str(),
                tenant.as_str(),
                ArtifactCleanupReason::ArtifactDeleted,
            )
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        let Some(intent) = self
            .metadata
            .artifact_cleanup_intent_for_storage_key(&asset.storage_key)
            .await
            .map_err(ArtifactStoreError::Metadata)?
        else {
            // The transaction guaranteed an intent existed. Its absence here
            // means a concurrent cleaner already completed physical deletion.
            return Ok(newly_deleted);
        };
        match self.cleanup_intent(&intent).await? {
            true => Ok(newly_deleted),
            false => Err(ArtifactStoreError::DeleteIncomplete),
        }
    }

    /// Process a bounded page of already-fenced provider deletions.
    ///
    /// Pending rows are retained with capped backoff after provider failure.
    /// This intentionally does not infer cleanup eligibility from reachability
    /// or attempt lease expiry.
    pub async fn cleanup_due(
        &self,
        limit: usize,
    ) -> Result<ArtifactCleanupReport, ArtifactStoreError> {
        let limit = limit.min(64);
        if limit == 0 {
            return Ok(ArtifactCleanupReport::default());
        }
        let provider_limit = limit.div_ceil(2);
        let provider_writes = self
            .metadata
            .claim_due_provider_write_cleanup(provider_limit)
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        let intents = self
            .metadata
            .due_artifact_cleanup_intents(limit.saturating_sub(provider_writes.len()))
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        let deadline = tokio::time::Instant::now() + CLEANUP_BATCH_TIMEOUT;
        let mut report = ArtifactCleanupReport::default();
        for reservation in provider_writes {
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            report.inspected += 1;
            if self.cleanup_provider_write(&reservation).await? {
                report.completed += 1;
            } else {
                report.deferred += 1;
            }
        }
        for intent in intents {
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            report.inspected += 1;
            if self.cleanup_intent(&intent).await? {
                report.completed += 1;
            } else {
                report.deferred += 1;
            }
        }
        Ok(report)
    }

    async fn store_reserved_write(
        &self,
        tenant: &ArtifactTenant,
        write: ArtifactWrite,
    ) -> Result<StoredReservedArtifact, ArtifactStoreError> {
        self.validate_write(&write)?;
        if self.provider.reserved_write_protocol_version() != Some(MEDIA_RESERVED_WRITE_VERSION) {
            return Err(ArtifactStoreError::ReservedWritesUnsupported);
        }
        let digest = sha256_hex(&write.bytes);
        let size_bytes = write.bytes.len() as u64;
        let write_id = new_uuid();
        let provider_request = MediaWriteRequest {
            namespace: MediaNamespace::Other("artifact-store".to_string()),
            record_id: write_id.clone(),
            preferred_filename: write.filename.clone(),
            content_type: write.content_type.clone(),
            metadata: tenant_metadata(tenant),
        };
        let reservation = self
            .metadata
            .reserve_provider_write(NewProviderWriteReservation {
                write_id,
                tenant_scope: tenant.as_str().to_string(),
                storage_namespace: "artifact-store".to_string(),
                content_type: write.content_type.clone(),
                filename: write.filename.clone(),
                expected_size_bytes: size_bytes,
                expected_sha256: digest.clone(),
                lifetime_ms: self.provider_write_lifetime().as_millis() as u64,
                provider_request,
            })
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        let provider_request = provider_write_request(&reservation);
        let stored = match tokio::time::timeout(
            PROVIDER_WRITE_TIMEOUT,
            self.provider.put_reserved(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id: reservation.write_id.clone(),
                    expires_at_unix_ms: reservation.expires_at,
                    content_length: reservation.expected_size_bytes,
                    sha256: reservation.expected_sha256.clone(),
                    request: provider_request,
                },
                write.bytes,
            ),
        )
        .await
        {
            Ok(Ok(stored)) => stored,
            Ok(Err(_)) | Err(_) => {
                self.abandon_write(&reservation, "Reserved provider write failed")
                    .await;
                return Err(ArtifactStoreError::Provider);
            }
        };

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
            self.abandon_write(&reservation, "Provider metadata validation failed")
                .await;
            return Err(error);
        }
        if validate_artifact_cleanup_storage_key(&stored.key.key).is_err() {
            self.abandon_write(&reservation, "Provider returned an invalid storage key")
                .await;
            return Err(ArtifactStoreError::InvalidMetadata(
                "invalid provider storage key",
            ));
        }
        let reservation = match self
            .metadata
            .record_provider_write_stored(&reservation, &stored.key.key)
            .await
        {
            Ok(Some(reservation)) => reservation,
            Ok(None) => {
                self.abandon_write(
                    &reservation,
                    "Provider write reservation expired before publication",
                )
                .await;
                return Err(ArtifactStoreError::Metadata(anyhow::anyhow!(
                    "Provider write reservation expired before publication"
                )));
            }
            Err(error) => {
                self.abandon_write(&reservation, "Provider storage acknowledgement failed")
                    .await;
                return Err(ArtifactStoreError::Metadata(error));
            }
        };
        Ok(StoredReservedArtifact {
            reservation,
            storage_key: stored.key.key,
            content_type: write.content_type,
            filename: write.filename,
            size_bytes,
            sha256: digest,
            retention: write.retention,
        })
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

    fn provider_write_lifetime(&self) -> Duration {
        #[cfg(test)]
        let lifetime = self.reservation_lifetime;
        #[cfg(not(test))]
        let lifetime = PROVIDER_WRITE_RESERVATION_LIFETIME;
        lifetime
    }

    #[cfg(test)]
    fn set_reservation_lifetime_for_test(&mut self, lifetime: Duration) {
        self.reservation_lifetime = lifetime;
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

    async fn abandon_write(&self, reservation: &ProviderWriteReservation, error: &str) {
        let _ = self
            .metadata
            .abandon_provider_write(reservation, error)
            .await;
    }

    async fn cleanup_provider_write(
        &self,
        reservation: &ProviderWriteReservation,
    ) -> Result<bool, ArtifactStoreError> {
        let request = MediaReservedWriteRecoveryRequest {
            version: MEDIA_RESERVED_WRITE_VERSION,
            write_id: reservation.write_id.clone(),
            expires_at_unix_ms: reservation.expires_at,
            content_length: reservation.expected_size_bytes,
            sha256: reservation.expected_sha256.clone(),
            request: provider_write_request(reservation),
            storage_key: reservation.storage_key.clone().map(MediaObjectKey::new),
        };
        let recovery = tokio::time::timeout(
            PROVIDER_DELETE_TIMEOUT,
            self.provider.recover_reserved_write(request),
        )
        .await;
        match recovery {
            Ok(Ok(())) | Ok(Err(HookError::NotFound(_))) => {
                let completed = self
                    .metadata
                    .complete_provider_write_cleanup(reservation)
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(completed)
            }
            Ok(Err(error)) => {
                self.metadata
                    .defer_provider_write_cleanup(reservation, &error.to_string())
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(false)
            }
            Err(_) => {
                self.metadata
                    .defer_provider_write_cleanup(
                        reservation,
                        "Reserved provider write recovery timed out",
                    )
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(false)
            }
        }
    }

    async fn cleanup_intent(
        &self,
        intent: &ArtifactCleanupIntent,
    ) -> Result<bool, ArtifactStoreError> {
        let tenant = ArtifactTenant::parse(intent.tenant_scope.clone())?;
        let deletion = tokio::time::timeout(
            PROVIDER_DELETE_TIMEOUT,
            self.provider.delete(MediaDeleteRequest {
                key: MediaObjectKey::new(intent.storage_key.clone()),
                metadata: tenant_metadata(&tenant),
            }),
        )
        .await;
        match deletion {
            Ok(Ok(())) | Ok(Err(HookError::NotFound(_))) => {
                self.metadata
                    .complete_artifact_cleanup(&intent.id, &intent.storage_key)
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(true)
            }
            Ok(Err(error)) => {
                self.metadata
                    .defer_artifact_cleanup(intent, &error.to_string())
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(false)
            }
            Err(_) => {
                self.metadata
                    .defer_artifact_cleanup(intent, "Artifact provider deletion timed out")
                    .await
                    .map_err(ArtifactStoreError::Metadata)?;
                Ok(false)
            }
        }
    }
}

fn provider_write_request(reservation: &ProviderWriteReservation) -> MediaWriteRequest {
    reservation.provider_request.clone()
}

fn opaque_media_asset(tenant: &ArtifactTenant, stored: &StoredReservedArtifact) -> NewMediaAsset {
    NewMediaAsset {
        asset_kind: "opaque_artifact".to_string(),
        storage_namespace: "artifact_store_v1".to_string(),
        storage_key: stored.storage_key.clone(),
        content_type: stored.content_type.clone(),
        filename: stored.filename.clone(),
        size_bytes: stored.size_bytes,
        sha256: Some(stored.sha256.clone()),
        duration_secs: None,
        sample_rate_hz: None,
        channel_count: None,
        peak_amplitude: None,
        rms_amplitude: None,
        source_asset_id: None,
        canonical_profile_version: None,
        scan_status: "not_required".to_string(),
        retention_policy: stored.retention.as_db_value().to_string(),
        metadata_json: serde_json::json!({
            "artifact_store": {
                "version": ARTIFACT_METADATA_VERSION,
                "tenant_id": tenant.as_str(),
            }
        }),
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
    use crate::{
        batch_runtime::{
            store::{NewJobStage, NewRuntimeJob},
            types::{RuntimeJobKind, RuntimeJobStatus, RuntimeStageStatus},
        },
        db::StoreDatabase,
        persistence::LocalMediaStorageProvider,
    };
    use izwi_hooks::{HookResult, MediaObjectMetadata, StoredMediaObject, StoredMediaStream};
    use std::{
        collections::{BTreeMap, BTreeSet},
        io::Cursor,
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc as StdArc, Mutex,
        },
        time::{SystemTime, UNIX_EPOCH},
    };
    use tempfile::TempDir;
    use tokio::sync::Notify;

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

    async fn active_tts_lease(
        metadata: &BatchRuntimeStore,
        tenant_key: Option<[u8; 32]>,
    ) -> StageLease {
        let job = metadata
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("FishAudio-S2-Pro".into()),
                capability: Some("tts".into()),
                route_record_kind: None,
                route_record_id: None,
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: serde_json::json!({"tenant_key": tenant_key}),
                model_snapshot_json: serde_json::json!({"version": 1}),
                retry_policy_json: serde_json::json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .unwrap();
        metadata
            .create_stage(NewJobStage {
                job_id: job.id,
                sequence: 0,
                stage_kind: "tts_synthesize".into(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".into()),
                model_id: Some("FishAudio-S2-Pro".into()),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .unwrap();
        metadata
            .claim_next_stage("artifact-worker", 60_000)
            .await
            .unwrap()
            .unwrap()
            .lease()
            .unwrap()
    }

    fn pcm_attempt_write(bytes: Vec<u8>) -> AttemptArtifactWrite {
        AttemptArtifactWrite {
            publication_key: "speech-pcm/00000000000000000000".into(),
            artifact_kind: RuntimeArtifactKind::Audio,
            artifact_role: RuntimeArtifactRole::OutputIntermediate,
            metadata_json: serde_json::json!({
                "version": 1,
                "sequence": 0,
                "segment": 0,
                "sample_offset": 0,
                "sample_count": bytes.len() / 4,
                "sample_rate": 44_100,
            }),
            runtime_retention_policy: "speech_job_replay".into(),
            object: ArtifactWrite {
                content_type: "audio/pcm-f32le".into(),
                filename: Some("chunk.f32le".into()),
                bytes,
                retention: ArtifactRetention::Job,
            },
        }
    }

    fn test_provider_write_input(
        tenant_scope: &str,
        storage_namespace: &str,
        content_type: &str,
        filename: Option<&str>,
        bytes: &[u8],
        lifetime_ms: u64,
    ) -> NewProviderWriteReservation {
        let write_id = new_uuid();
        let tenant = ArtifactTenant::parse(tenant_scope).unwrap();
        NewProviderWriteReservation {
            write_id: write_id.clone(),
            tenant_scope: tenant_scope.to_string(),
            storage_namespace: storage_namespace.to_string(),
            content_type: content_type.to_string(),
            filename: filename.map(str::to_string),
            expected_size_bytes: bytes.len() as u64,
            expected_sha256: sha256_hex(bytes),
            lifetime_ms,
            provider_request: MediaWriteRequest {
                namespace: MediaNamespace::Other(storage_namespace.to_string()),
                record_id: write_id,
                preferred_filename: filename.map(str::to_string),
                content_type: content_type.to_string(),
                metadata: tenant_metadata(&tenant),
            },
        }
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
        assert_provider_conformance(&root, provider.clone()).await;

        let store = conformance_store(&root, provider);
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        for (content_type, filename) in [
            ("image/png", "sample.png"),
            ("video/mp4", "sample.mp4"),
            ("audio/mpeg", "sample.mp3"),
            ("audio/pcm-f32le", "sample.f32le"),
        ] {
            let descriptor = store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: content_type.into(),
                        filename: Some(filename.into()),
                        bytes: b"representative local bytes".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await
                .unwrap();
            let stored = store.read(&tenant, &descriptor.id).await.unwrap();
            assert_eq!(stored.descriptor.content_type, content_type);
            assert_eq!(stored.bytes, b"representative local bytes");
        }

        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: "audio/x-wav".into(),
                        filename: Some("alias.wav".into()),
                        bytes: b"alias".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::Provider)
        ));
    }

    #[tokio::test]
    async fn deterministic_remote_like_provider_conforms_to_opaque_artifact_contract() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        assert_provider_conformance(&root, provider).await;
    }

    #[test]
    fn scheduling_tenant_mapping_is_stable_and_bounded() {
        assert_eq!(
            ArtifactTenant::from_scheduling_key(None).as_str(),
            "anonymous"
        );
        assert_eq!(
            ArtifactTenant::from_scheduling_key(Some([0xab; 32])).as_str(),
            "abababababababababababababababababababababababababababababababab"
        );
    }

    #[tokio::test]
    async fn attempt_publication_atomically_hides_provider_keys_and_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(root.path().join("attempt.sqlite3")),
        ));
        let mut store = ArtifactStore::new(
            metadata.clone(),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        store.set_reservation_lifetime_for_test(Duration::from_millis(10));
        let tenant_key = Some([7; 32]);
        let tenant = ArtifactTenant::from_scheduling_key(tenant_key);
        let lease = active_tts_lease(&metadata, tenant_key).await;
        let bytes = [0.1_f32, -0.2, 0.3]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let first = store
            .put_attempt_artifact(
                &tenant,
                &lease,
                pcm_attempt_write(bytes.clone()),
                serde_json::json!({"publication_started": true}),
            )
            .await
            .unwrap();
        let id = ArtifactId::parse(first.media_asset_id.clone().unwrap()).unwrap();
        assert!(first.storage_key.is_none());
        assert_eq!(store.read(&tenant, &id).await.unwrap().bytes, bytes);
        assert!(matches!(
            store
                .read(&ArtifactTenant::parse("other").unwrap(), &id)
                .await,
            Err(ArtifactStoreError::NotFound)
        ));
        let reopened = ArtifactStore::new(
            Arc::new(BatchRuntimeStore::initialize_with_database(
                StoreDatabase::new(root.path().join("attempt.sqlite3")),
            )),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        assert_eq!(reopened.read(&tenant, &id).await.unwrap().bytes, bytes);

        let duplicate = store
            .put_attempt_artifact(
                &tenant,
                &lease,
                pcm_attempt_write(bytes),
                serde_json::json!({"publication_started": true}),
            )
            .await
            .unwrap();
        assert_eq!(duplicate.id, first.id);
        assert_eq!(provider.object_count(), 2);
        let mut changed = pcm_attempt_write(0.9_f32.to_le_bytes().to_vec());
        changed.metadata_json["sample_count"] = serde_json::json!(1);
        assert!(matches!(
            store
                .put_attempt_artifact(
                    &tenant,
                    &lease,
                    changed,
                    serde_json::json!({"publication_started": true}),
                )
                .await,
            Err(ArtifactStoreError::Metadata(_))
        ));
        assert_eq!(provider.object_count(), 3);
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert_eq!(store.cleanup_due(64).await.unwrap().completed, 2);
        assert_eq!(provider.object_count(), 1);
    }

    #[tokio::test]
    async fn stale_attempt_keeps_written_provider_object_recoverable() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(root.path().join("stale-attempt.sqlite3")),
        ));
        let mut store = ArtifactStore::new(
            metadata.clone(),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        store.set_reservation_lifetime_for_test(Duration::from_millis(10));
        let tenant = ArtifactTenant::from_scheduling_key(None);
        let lease = active_tts_lease(&metadata, None).await;
        metadata
            .cancel_job(
                &metadata
                    .get_stage(&lease.stage_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .job_id,
                Some("test cancellation".into()),
            )
            .await
            .unwrap();

        assert!(matches!(
            store
                .put_attempt_artifact(
                    &tenant,
                    &lease,
                    pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
                    serde_json::json!({"publication_started": true}),
                )
                .await,
            Err(ArtifactStoreError::Metadata(_))
        ));
        assert_eq!(provider.object_count(), 1);
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
        assert_eq!(provider.object_count(), 0);
        assert!(metadata
            .speech_pcm_after(
                &metadata
                    .get_stage(&lease.stage_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .job_id,
                None,
                1,
            )
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn attempt_publication_rejects_a_different_scheduling_tenant() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(root.path().join("tenant-fence.sqlite3")),
        ));
        let mut store = ArtifactStore::new(
            metadata.clone(),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        store.set_reservation_lifetime_for_test(Duration::from_millis(10));
        let lease = active_tts_lease(&metadata, Some([7; 32])).await;
        let wrong_tenant = ArtifactTenant::from_scheduling_key(Some([8; 32]));

        assert!(matches!(
            store
                .put_attempt_artifact(
                    &wrong_tenant,
                    &lease,
                    pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
                    serde_json::json!({"publication_started": true}),
                )
                .await,
            Err(ArtifactStoreError::Metadata(_))
        ));
        let job_id = metadata
            .get_stage(&lease.stage_id)
            .await
            .unwrap()
            .unwrap()
            .job_id;
        assert!(metadata
            .speech_pcm_after(&job_id, None, 1)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(provider.object_count(), 1);
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn pcm_entry_limit_rejects_reference_and_recovers_provider_write() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("pcm-limit.sqlite3"),
        ));
        metadata.set_pcm_replay_entry_limit_for_test(1);
        let metadata = Arc::new(metadata);
        let mut store = ArtifactStore::new(
            metadata.clone(),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        store.set_reservation_lifetime_for_test(Duration::from_millis(10));
        let tenant = ArtifactTenant::from_scheduling_key(None);
        let lease = active_tts_lease(&metadata, None).await;
        store
            .put_attempt_artifact(
                &tenant,
                &lease,
                pcm_attempt_write(0.1_f32.to_le_bytes().to_vec()),
                serde_json::json!({"publication_started": true}),
            )
            .await
            .unwrap();
        let mut second = pcm_attempt_write(0.2_f32.to_le_bytes().to_vec());
        second.publication_key = "speech-pcm/00000000000000000001".into();
        second.metadata_json["sequence"] = serde_json::json!(1);
        let error = store
            .put_attempt_artifact(
                &tenant,
                &lease,
                second,
                serde_json::json!({"publication_started": true}),
            )
            .await
            .unwrap_err();
        let ArtifactStoreError::Metadata(error) = error else {
            panic!("unexpected quota error: {error}");
        };
        let error_text = format!("{error:#}");
        assert!(
            error_text.contains("speech_storage_limit"),
            "unexpected quota error: {error_text}"
        );
        let job_id = metadata
            .get_stage(&lease.stage_id)
            .await
            .unwrap()
            .unwrap()
            .job_id;
        assert_eq!(
            metadata
                .speech_pcm_after(&job_id, None, 64)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(provider.object_count(), 2);
        tokio::time::sleep(Duration::from_millis(15)).await;
        assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
        assert_eq!(provider.object_count(), 1);
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
    async fn provider_without_reserved_writes_is_rejected_before_bytes_are_sent() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        provider.supports_reserved.store(false, Ordering::SeqCst);
        let store = conformance_store(&root, provider.clone());
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();

        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: "application/octet-stream".to_string(),
                        filename: None,
                        bytes: b"artifact".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::ReservedWritesUnsupported)
        ));
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn invalid_provider_storage_keys_are_recovered_after_the_write_fence() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let mut store = conformance_store(&root, provider.clone());
        store.set_reservation_lifetime_for_test(Duration::from_millis(100));
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();

        for key in [
            String::new(),
            "bad\nkey".to_string(),
            "x".repeat(2 * 1024 + 1),
        ] {
            provider.set_next_key(key);
            assert!(matches!(
                store
                    .put(
                        &tenant,
                        ArtifactWrite {
                            content_type: "application/octet-stream".to_string(),
                            filename: None,
                            bytes: b"artifact".to_vec(),
                            retention: ArtifactRetention::Ephemeral,
                        },
                    )
                    .await,
                Err(ArtifactStoreError::InvalidMetadata(
                    "invalid provider storage key"
                ))
            ));
            tokio::time::sleep(Duration::from_millis(110)).await;
            assert_eq!(store.cleanup_due(64).await.unwrap().completed, 1);
            assert_eq!(provider.object_count(), 0);
        }
    }

    #[tokio::test]
    async fn exact_provider_request_without_returned_key_is_recovered_after_reopen() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("artifacts.sqlite3");
        let provider = Arc::new(MemoryMediaProvider::default());
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(db_path.clone()),
        ));
        let write_id = new_uuid();
        let mut request_metadata = HookMetadata::new();
        request_metadata.insert("tenant_id".into(), "tenant-a".into());
        request_metadata.insert("workflow_stage".into(), "speech-finalize".into());
        let reservation = metadata
            .reserve_provider_write(NewProviderWriteReservation {
                write_id: write_id.clone(),
                tenant_scope: "tenant-a".into(),
                storage_namespace: "generated_speech".into(),
                content_type: "application/octet-stream".into(),
                filename: Some("crash.bin".into()),
                expected_size_bytes: 5,
                expected_sha256: sha256_hex(b"crash"),
                lifetime_ms: 100,
                provider_request: MediaWriteRequest {
                    namespace: MediaNamespace::GeneratedSpeech,
                    record_id: write_id,
                    preferred_filename: Some("crash.bin".into()),
                    content_type: "application/octet-stream".into(),
                    metadata: request_metadata,
                },
            })
            .await
            .unwrap();
        provider
            .put_reserved(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id: reservation.write_id.clone(),
                    expires_at_unix_ms: reservation.expires_at,
                    content_length: reservation.expected_size_bytes,
                    sha256: reservation.expected_sha256.clone(),
                    request: provider_write_request(&reservation),
                },
                b"crash".to_vec(),
            )
            .await
            .unwrap();
        assert_eq!(provider.object_count(), 1);
        drop(metadata);
        tokio::time::sleep(Duration::from_millis(110)).await;

        let reopened = ArtifactStore::new(
            Arc::new(BatchRuntimeStore::initialize_with_database(
                StoreDatabase::new(db_path),
            )),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        assert_eq!(reopened.cleanup_due(64).await.unwrap().completed, 1);
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn delayed_reserved_put_cannot_publish_after_recovery_fences_its_write_id() {
        let provider = Arc::new(MemoryMediaProvider::default());
        provider.set_now_unix_ms(1_000);
        let write_id = uuid::Uuid::new_v4().to_string();
        let media_request = MediaWriteRequest {
            namespace: MediaNamespace::Other("artifact-store".into()),
            record_id: write_id.clone(),
            preferred_filename: Some("race.bin".into()),
            content_type: "application/octet-stream".into(),
            metadata: HookMetadata::new(),
        };
        let write_request = MediaReservedWriteRequest {
            version: MEDIA_RESERVED_WRITE_VERSION,
            write_id: write_id.clone(),
            expires_at_unix_ms: 1_010,
            content_length: 4,
            sha256: sha256_hex(b"race"),
            request: media_request.clone(),
        };
        let entered = StdArc::new(Notify::new());
        let release = StdArc::new(Notify::new());
        let entered_wait = entered.notified();
        provider.set_reserved_put_gate(entered.clone(), release.clone());
        let delayed_provider = provider.clone();
        let delayed_put = tokio::spawn(async move {
            delayed_provider
                .put_reserved(write_request, b"race".to_vec())
                .await
        });
        entered_wait.await;

        provider.set_now_unix_ms(1_010);
        assert!(matches!(
            provider
                .recover_reserved_write(MediaReservedWriteRecoveryRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id,
                    expires_at_unix_ms: 1_010,
                    content_length: 4,
                    sha256: sha256_hex(b"race"),
                    request: media_request,
                    storage_key: None,
                })
                .await,
            Err(HookError::NotFound(_))
        ));
        // Rewind the deterministic clock so expiry alone cannot make this pass:
        // the recovery fence must reject the delayed publisher.
        provider.set_now_unix_ms(1_000);
        release.notify_one();
        assert!(delayed_put.await.unwrap().is_err());
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn provider_write_capacity_fails_before_bytes_are_sent() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("artifacts.sqlite3"),
        ));
        metadata.set_provider_write_capacity_for_test(1);
        let metadata = Arc::new(metadata);
        let input = test_provider_write_input(
            "tenant-a",
            "artifact-store",
            "application/octet-stream",
            None,
            b"x",
            60_000,
        );
        metadata
            .reserve_provider_write(input.clone())
            .await
            .unwrap();
        let store =
            ArtifactStore::new(metadata, provider.clone(), ArtifactStoreLimits::default()).unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        assert!(matches!(
            store
                .put(
                    &tenant,
                    ArtifactWrite {
                        content_type: input.content_type,
                        filename: input.filename,
                        bytes: b"x".to_vec(),
                        retention: ArtifactRetention::Ephemeral,
                    },
                )
                .await,
            Err(ArtifactStoreError::Metadata(_))
        ));
        assert_eq!(provider.object_count(), 0);
    }

    #[tokio::test]
    async fn provider_write_cleanup_claim_is_exact_and_nonduplicating() {
        let root = tempfile::tempdir().unwrap();
        let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("artifacts.sqlite3"),
        ));
        metadata.set_test_clock(clock.clone());
        let reservation = metadata
            .reserve_provider_write(test_provider_write_input(
                "tenant-a",
                "artifact-store",
                "application/octet-stream",
                None,
                b"x",
                10,
            ))
            .await
            .unwrap();
        metadata
            .record_provider_write_stored(&reservation, "private/object/fenced")
            .await
            .unwrap()
            .unwrap();
        clock.store(1_010, Ordering::SeqCst);
        let first = metadata.claim_due_provider_write_cleanup(64).await.unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].cleanup_claim_token.is_some());
        assert!(metadata
            .claim_due_provider_write_cleanup(64)
            .await
            .unwrap()
            .is_empty());
        assert!(metadata
            .publish_reserved_opaque_artifact(
                &first[0],
                NewMediaAsset {
                    asset_kind: "opaque_artifact".into(),
                    storage_namespace: "artifact_store_v1".into(),
                    storage_key: "private/object/fenced".into(),
                    content_type: "application/octet-stream".into(),
                    filename: None,
                    size_bytes: 1,
                    sha256: Some(sha256_hex(b"x")),
                    duration_secs: None,
                    sample_rate_hz: None,
                    channel_count: None,
                    peak_amplitude: None,
                    rms_amplitude: None,
                    source_asset_id: None,
                    canonical_profile_version: None,
                    scan_status: "not_required".into(),
                    retention_policy: "artifact_ephemeral".into(),
                    metadata_json: serde_json::json!({}),
                },
            )
            .await
            .unwrap()
            .is_none());
        let forged = ProviderWriteReservation {
            cleanup_claim_token: Some(uuid::Uuid::new_v4().to_string()),
            ..first[0].clone()
        };
        assert!(!metadata
            .complete_provider_write_cleanup(&forged)
            .await
            .unwrap());
        assert!(metadata
            .complete_provider_write_cleanup(&first[0])
            .await
            .unwrap());
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

    #[tokio::test]
    async fn cleanup_intent_survives_store_reopen_and_retries_known_dead_object() {
        let root = tempfile::tempdir().unwrap();
        let db_path = root.path().join("artifacts.sqlite3");
        let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
        let mut metadata =
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
        metadata.set_test_clock(clock.clone());
        let provider = Arc::new(MemoryMediaProvider::default());
        provider.set_now_unix_ms(1_000);
        let store = ArtifactStore::new(
            Arc::new(metadata),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let descriptor = put_test_artifact(&store, &tenant, "reopen").await;
        provider.fail_deletes.store(true, Ordering::SeqCst);
        assert!(matches!(
            store.delete(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
        drop(store);

        clock.store(2_000, Ordering::SeqCst);
        let mut reopened = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
        reopened.set_test_clock(clock);
        provider.fail_deletes.store(false, Ordering::SeqCst);
        let reopened = ArtifactStore::new(
            Arc::new(reopened),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        assert_eq!(
            reopened.cleanup_due(64).await.unwrap(),
            ArtifactCleanupReport {
                inspected: 1,
                completed: 1,
                deferred: 0,
            }
        );
        assert_eq!(provider.object_count(), 0);
        assert!(reopened
            .metadata
            .due_artifact_cleanup_intents(64)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn cleanup_capacity_failure_happens_before_second_tombstone() {
        let root = tempfile::tempdir().unwrap();
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("artifacts.sqlite3"),
        ));
        metadata.set_artifact_cleanup_capacity_for_test(1);
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = ArtifactStore::new(
            Arc::new(metadata),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let first = put_test_artifact(&store, &tenant, "first").await;
        let second = put_test_artifact(&store, &tenant, "second").await;
        provider.fail_deletes.store(true, Ordering::SeqCst);
        assert!(matches!(
            store.delete(&tenant, &first.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
        assert!(matches!(
            store.delete(&tenant, &second.id).await,
            Err(ArtifactStoreError::Metadata(_))
        ));
        assert_eq!(store.stat(&tenant, &second.id).await.unwrap(), second);
    }

    #[tokio::test]
    async fn provider_not_found_completes_cleanup_idempotently() {
        let root = tempfile::tempdir().unwrap();
        let provider = Arc::new(MemoryMediaProvider::default());
        let store = conformance_store(&root, provider.clone());
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let descriptor = put_test_artifact(&store, &tenant, "missing").await;
        provider.remove_single_object();

        assert!(store.delete(&tenant, &descriptor.id).await.unwrap());
        assert!(store
            .metadata
            .due_artifact_cleanup_intents(64)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn due_cleanup_is_bounded_and_never_selects_active_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("artifacts.sqlite3"),
        ));
        metadata.set_test_clock(clock.clone());
        let provider = Arc::new(MemoryMediaProvider::default());
        provider.set_now_unix_ms(1_000);
        let store = ArtifactStore::new(
            Arc::new(metadata),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let active = put_test_artifact(&store, &tenant, "active").await;
        assert_eq!(store.cleanup_due(0).await.unwrap().inspected, 0);
        assert_eq!(store.cleanup_due(usize::MAX).await.unwrap().inspected, 0);
        assert_eq!(store.stat(&tenant, &active.id).await.unwrap(), active);

        provider.fail_deletes.store(true, Ordering::SeqCst);
        for index in 0..65 {
            let artifact = put_test_artifact(&store, &tenant, &format!("dead-{index}")).await;
            assert!(matches!(
                store.delete(&tenant, &artifact.id).await,
                Err(ArtifactStoreError::DeleteIncomplete)
            ));
        }
        clock.store(2_000, Ordering::SeqCst);
        provider.fail_deletes.store(false, Ordering::SeqCst);
        let first = store.cleanup_due(usize::MAX).await.unwrap();
        assert_eq!(first.inspected, 64);
        assert_eq!(first.completed, 64);
        let second = store.cleanup_due(usize::MAX).await.unwrap();
        assert_eq!(second.inspected, 1);
        assert_eq!(second.completed, 1);
        assert_eq!(store.stat(&tenant, &active.id).await.unwrap(), active);
    }

    #[tokio::test]
    async fn duplicate_cleaners_and_retry_metadata_remain_safe_and_bounded() {
        let root = tempfile::tempdir().unwrap();
        let clock = Arc::new(std::sync::atomic::AtomicI64::new(1_000));
        let mut metadata = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("artifacts.sqlite3"),
        ));
        metadata.set_test_clock(clock.clone());
        let provider = Arc::new(MemoryMediaProvider::default());
        provider.set_now_unix_ms(1_000);
        let store = ArtifactStore::new(
            Arc::new(metadata),
            provider.clone(),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        let tenant = ArtifactTenant::parse("tenant-a").unwrap();
        let descriptor = put_test_artifact(&store, &tenant, "duplicate").await;
        provider.set_delete_error("é".repeat(600));
        assert!(matches!(
            store.delete(&tenant, &descriptor.id).await,
            Err(ArtifactStoreError::DeleteIncomplete)
        ));
        let mut intent = store
            .metadata
            .artifact_cleanup_intent_for_storage_key(
                &store
                    .resolve(&tenant, &descriptor.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .storage_key,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(intent.attempt_count, 1);
        assert!(intent.last_error.as_ref().unwrap().len() <= 512);
        for _ in 0..14 {
            clock.store(intent.available_at as i64, Ordering::SeqCst);
            assert!(matches!(
                store.delete(&tenant, &descriptor.id).await,
                Err(ArtifactStoreError::DeleteIncomplete)
            ));
            let next = store
                .metadata
                .artifact_cleanup_intent_for_storage_key(&intent.storage_key)
                .await
                .unwrap()
                .unwrap();
            assert!(next.available_at - intent.available_at <= 60 * 60 * 1000);
            intent = next;
        }

        provider.clear_delete_error();
        let first_cleaner = intent.clone();
        let second_cleaner = intent;
        assert!(store.cleanup_intent(&first_cleaner).await.unwrap());
        assert!(store.cleanup_intent(&second_cleaner).await.unwrap());
        assert!(store
            .metadata
            .artifact_cleanup_intent_for_storage_key(&first_cleaner.storage_key)
            .await
            .unwrap()
            .is_none());
    }

    async fn put_test_artifact(
        store: &ArtifactStore,
        tenant: &ArtifactTenant,
        label: &str,
    ) -> ArtifactDescriptor {
        store
            .put(
                tenant,
                ArtifactWrite {
                    content_type: "application/octet-stream".to_string(),
                    filename: Some(format!("{label}.bin")),
                    bytes: label.as_bytes().to_vec(),
                    retention: ArtifactRetention::Ephemeral,
                },
            )
            .await
            .unwrap()
    }

    #[derive(Clone)]
    struct MemoryObject {
        bytes: Vec<u8>,
        metadata: MediaObjectMetadata,
    }

    struct MemoryReservedWrite {
        request: MediaWriteRequest,
        size_bytes: u64,
        sha256: String,
        key: String,
    }

    #[derive(Default)]
    struct MemoryProviderState {
        objects: BTreeMap<String, MemoryObject>,
        reserved_writes: BTreeMap<String, MemoryReservedWrite>,
        fenced_write_ids: BTreeSet<String>,
    }

    struct MemoryMediaProvider {
        state: Mutex<MemoryProviderState>,
        supports_reserved: AtomicBool,
        now_unix_ms: AtomicU64,
        reserved_put_gate: Mutex<Option<(StdArc<Notify>, StdArc<Notify>)>>,
        fail_deletes: AtomicBool,
        delete_error: Mutex<Option<String>>,
        next_key: Mutex<Option<String>>,
    }

    impl Default for MemoryMediaProvider {
        fn default() -> Self {
            Self {
                state: Mutex::default(),
                supports_reserved: AtomicBool::new(true),
                now_unix_ms: AtomicU64::new(0),
                reserved_put_gate: Mutex::default(),
                fail_deletes: AtomicBool::default(),
                delete_error: Mutex::default(),
                next_key: Mutex::default(),
            }
        }
    }

    impl MemoryMediaProvider {
        fn object_count(&self) -> usize {
            self.state.lock().unwrap().objects.len()
        }

        fn tamper_single_object(&self, bytes: Vec<u8>) {
            let mut state = self.state.lock().unwrap();
            let object = state.objects.values_mut().next().expect("one object");
            object.bytes = bytes;
            object.metadata.content_length = Some(object.bytes.len() as u64);
            object.metadata.sha256 = None;
        }

        fn remove_single_object(&self) {
            let mut state = self.state.lock().unwrap();
            let key = state.objects.keys().next().cloned().unwrap();
            state.objects.remove(&key);
        }

        fn set_delete_error(&self, error: String) {
            *self.delete_error.lock().unwrap() = Some(error);
        }

        fn clear_delete_error(&self) {
            *self.delete_error.lock().unwrap() = None;
        }

        fn set_next_key(&self, key: String) {
            *self.next_key.lock().unwrap() = Some(key);
        }

        fn set_now_unix_ms(&self, now: u64) {
            self.now_unix_ms.store(now, Ordering::SeqCst);
        }

        fn now_unix_ms(&self) -> u64 {
            let fixed = self.now_unix_ms.load(Ordering::SeqCst);
            if fixed != 0 {
                return fixed;
            }
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        }

        fn set_reserved_put_gate(&self, entered: StdArc<Notify>, release: StdArc<Notify>) {
            *self.reserved_put_gate.lock().unwrap() = Some((entered, release));
        }
    }

    #[async_trait::async_trait]
    impl MediaStorageProvider for MemoryMediaProvider {
        fn reserved_write_protocol_version(&self) -> Option<u16> {
            self.supports_reserved
                .load(Ordering::SeqCst)
                .then_some(MEDIA_RESERVED_WRITE_VERSION)
        }

        async fn put(
            &self,
            request: MediaWriteRequest,
            bytes: Vec<u8>,
        ) -> HookResult<StoredMediaObject> {
            let key = self
                .next_key
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| format!("private/object/{}", request.record_id));
            let tenant_id = request.metadata.get("tenant_id").cloned();
            let metadata = MediaObjectMetadata {
                content_type: request.content_type,
                filename: request.preferred_filename,
                content_length: Some(bytes.len() as u64),
                sha256: Some(sha256_hex(&bytes)),
                tenant_id,
                attributes: request.metadata,
            };
            self.state.lock().unwrap().objects.insert(
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

        async fn put_reserved(
            &self,
            request: MediaReservedWriteRequest,
            bytes: Vec<u8>,
        ) -> HookResult<StoredMediaObject> {
            let gate = self.reserved_put_gate.lock().unwrap().take();
            if let Some((entered, release)) = gate {
                entered.notify_one();
                release.notified().await;
            }
            let size_bytes = bytes.len() as u64;
            let sha256 = sha256_hex(&bytes);
            if request.version != MEDIA_RESERVED_WRITE_VERSION
                || uuid::Uuid::parse_str(&request.write_id).is_err()
                || request.request.record_id != request.write_id
                || self.now_unix_ms() >= request.expires_at_unix_ms
                || request.content_length != size_bytes
                || request.sha256 != sha256
            {
                return Err(HookError::Failed("invalid reserved write envelope".into()));
            }
            let mut state = self.state.lock().unwrap();
            if state.fenced_write_ids.contains(&request.write_id) {
                return Err(HookError::Failed("reserved write is fenced".into()));
            }
            if let Some(existing) = state.reserved_writes.get(&request.write_id) {
                if existing.request != request.request
                    || existing.size_bytes != size_bytes
                    || existing.sha256 != sha256
                {
                    return Err(HookError::Failed(
                        "reserved write identity was reused".into(),
                    ));
                }
                let object = state
                    .objects
                    .get(&existing.key)
                    .cloned()
                    .ok_or_else(|| HookError::Failed("reserved object disappeared".into()))?;
                return Ok(StoredMediaObject {
                    key: MediaObjectKey::new(existing.key.clone()),
                    metadata: object.metadata,
                });
            }
            let write_id = request.write_id.clone();
            let original_request = request.request.clone();
            let key = self
                .next_key
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| format!("private/object/{}", request.request.record_id));
            let tenant_id = request.request.metadata.get("tenant_id").cloned();
            let metadata = MediaObjectMetadata {
                content_type: request.request.content_type,
                filename: request.request.preferred_filename,
                content_length: Some(size_bytes),
                sha256: Some(sha256.clone()),
                tenant_id,
                attributes: request.request.metadata,
            };
            state.objects.insert(
                key.clone(),
                MemoryObject {
                    bytes,
                    metadata: metadata.clone(),
                },
            );
            state.reserved_writes.insert(
                write_id,
                MemoryReservedWrite {
                    request: original_request,
                    size_bytes,
                    sha256,
                    key: key.clone(),
                },
            );
            Ok(StoredMediaObject {
                key: MediaObjectKey::new(key),
                metadata,
            })
        }

        async fn recover_reserved_write(
            &self,
            request: MediaReservedWriteRecoveryRequest,
        ) -> HookResult<()> {
            if request.version != MEDIA_RESERVED_WRITE_VERSION
                || uuid::Uuid::parse_str(&request.write_id).is_err()
                || request.request.record_id != request.write_id
                || self.now_unix_ms() < request.expires_at_unix_ms
            {
                return Err(HookError::Failed(
                    "invalid reserved recovery envelope".into(),
                ));
            }
            if let Some(error) = self.delete_error.lock().unwrap().clone() {
                return Err(HookError::Failed(error));
            }
            if self.fail_deletes.load(Ordering::SeqCst) {
                return Err(HookError::Failed("injected delete failure".to_string()));
            }
            let mut state = self.state.lock().unwrap();
            if state
                .reserved_writes
                .get(&request.write_id)
                .is_some_and(|write| {
                    write.request != request.request
                        || write.size_bytes != request.content_length
                        || write.sha256 != request.sha256
                })
            {
                return Err(HookError::Failed(
                    "reserved recovery did not match its write identity".into(),
                ));
            }
            state.fenced_write_ids.insert(request.write_id.clone());
            let key = state
                .reserved_writes
                .remove(&request.write_id)
                .map(|write| write.key)
                .or_else(|| request.storage_key.map(|key| key.key));
            match key.and_then(|key| state.objects.remove(&key)) {
                Some(_) => Ok(()),
                None => Err(HookError::NotFound(
                    "reserved write is absent and fenced".to_string(),
                )),
            }
        }

        async fn get(&self, request: MediaReadRequest) -> HookResult<izwi_hooks::StoredMediaBytes> {
            let object = self
                .state
                .lock()
                .unwrap()
                .objects
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
                .state
                .lock()
                .unwrap()
                .objects
                .get(&request.key.key)
                .cloned()
                .ok_or_else(|| HookError::NotFound("object".to_string()))?;
            Ok(StoredMediaStream {
                reader: Box::pin(Cursor::new(object.bytes)),
                metadata: object.metadata,
            })
        }

        async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
            if let Some(error) = self.delete_error.lock().unwrap().clone() {
                return Err(HookError::Failed(error));
            }
            if self.fail_deletes.load(Ordering::SeqCst) {
                return Err(HookError::Failed("injected delete failure".to_string()));
            }
            if self
                .state
                .lock()
                .unwrap()
                .objects
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

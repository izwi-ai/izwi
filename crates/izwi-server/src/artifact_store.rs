//! Tenant-scoped opaque artifact access over the configured media provider.
//!
//! This module is a storage foundation, not a public route. Callers exchange
//! [`ArtifactId`] values while provider keys remain private to this facade and
//! the existing durable `media_assets` table.

use crate::batch_runtime::store::{
    validate_artifact_cleanup_storage_key, ArtifactCleanupIntent, ArtifactCleanupReason,
    BatchRuntimeStore, NewMediaAsset, NewProviderWriteReservation, NewStageOutputArtifact,
    ProviderWriteReservation, ReservedStageArtifactPublication, RetentionPolicy,
    SpeechHistoryFinalization,
};
use crate::batch_runtime::types::{
    RuntimeArtifact, RuntimeArtifactKind, RuntimeArtifactRole, StageLease,
};
use crate::ids::new_uuid;
use izwi_hooks::{
    HookError, HookMetadata, MediaDeleteRequest, MediaNamespace, MediaObjectKey, MediaReadRequest,
    MediaReservedWriteRecoveryRequest, MediaReservedWriteRequest, MediaStorageProvider,
    MediaWriteRequest, StoredMediaObject, MEDIA_RESERVED_WRITE_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, ReadBuf};

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
    pub max_file_object_bytes: u64,
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
        if self.max_file_object_bytes == 0 || self.max_file_object_bytes > ABSOLUTE_MAX_OBJECT_BYTES
        {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid artifact file-size limit",
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
            max_file_object_bytes: ABSOLUTE_MAX_OBJECT_BYTES,
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

#[derive(Debug, Clone)]
pub(crate) struct ArtifactFileWrite {
    pub content_type: String,
    pub filename: Option<String>,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
    pub retention: ArtifactRetention,
}

/// A generic attempt artifact written through the reserved file primitive.
/// Production writes only the final speech WAV
/// (`put_attempt_speech_history_file`); the conformance tests drive the
/// primitive through this generic form.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct AttemptArtifactFileWrite {
    pub publication_key: String,
    pub artifact_kind: RuntimeArtifactKind,
    pub artifact_role: RuntimeArtifactRole,
    pub metadata_json: serde_json::Value,
    pub runtime_retention_policy: String,
    pub object: ArtifactFileWrite,
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

pub struct ArtifactStream {
    pub descriptor: ArtifactDescriptor,
    pub reader: Pin<Box<dyn AsyncRead + Send>>,
}

struct VerifiedArtifactReader {
    inner: Pin<Box<dyn AsyncRead + Send>>,
    expected_size: u64,
    expected_digest: String,
    total: u64,
    digest: Option<Sha256>,
    finished: bool,
    scratch: Box<[u8; READ_CHUNK_BYTES]>,
}

impl AsyncRead for VerifiedArtifactReader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(Ok(()));
        }
        if buffer.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        let remaining = this.expected_size.saturating_sub(this.total);
        let capacity = if remaining == 0 {
            1
        } else {
            usize::try_from(remaining)
                .unwrap_or(usize::MAX)
                .min(buffer.remaining())
                .min(READ_CHUNK_BYTES)
        };
        let mut provider_buffer = ReadBuf::new(&mut this.scratch[..capacity]);
        match this.inner.as_mut().poll_read(cx, &mut provider_buffer) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => {
                this.finished = true;
                Poll::Ready(Err(error))
            }
            Poll::Ready(Ok(())) => {
                let read = provider_buffer.filled().len();
                if read == 0 {
                    this.finished = true;
                    if this.total != this.expected_size {
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "artifact stream ended before its declared size",
                        )));
                    }
                    let Some(digest) = this.digest.take() else {
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "artifact stream verifier lost its digest state",
                        )));
                    };
                    let actual_digest = format!("{:x}", digest.finalize());
                    if actual_digest != this.expected_digest {
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "artifact stream digest did not match",
                        )));
                    }
                    return Poll::Ready(Ok(()));
                }
                if remaining == 0 {
                    this.finished = true;
                    return Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "artifact stream exceeded its declared size",
                    )));
                }
                this.total = match this.total.checked_add(read as u64) {
                    Some(total) if total <= this.expected_size => total,
                    _ => {
                        this.finished = true;
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "artifact stream exceeded its declared size",
                        )));
                    }
                };
                if let Some(digest) = this.digest.as_mut() {
                    digest.update(provider_buffer.filled());
                }
                buffer.put_slice(provider_buffer.filled());
                Poll::Ready(Ok(()))
            }
        }
    }
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
    #[error("artifact storage provider lacks reserved-file writes")]
    ReservedFileWritesUnsupported,
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
        let AttemptArtifactWrite {
            publication_key,
            artifact_kind,
            artifact_role,
            metadata_json,
            runtime_retention_policy,
            object,
        } = write;
        self.validate_attempt_artifact(
            lease,
            &publication_key,
            &runtime_retention_policy,
            &metadata_json,
        )?;
        let stored = self.store_reserved_write(tenant, object).await?;
        self.publish_stored_attempt_artifact(
            tenant,
            lease,
            publication_key,
            artifact_kind,
            artifact_role,
            metadata_json,
            runtime_retention_policy,
            stored,
            publication_progress,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn put_attempt_file_artifact(
        &self,
        tenant: &ArtifactTenant,
        lease: &StageLease,
        write: AttemptArtifactFileWrite,
        publication_progress: serde_json::Value,
    ) -> Result<RuntimeArtifact, ArtifactStoreError> {
        if !self.provider.supports_reserved_file_writes() {
            return Err(ArtifactStoreError::ReservedFileWritesUnsupported);
        }
        if self.provider.reserved_write_protocol_version() != Some(MEDIA_RESERVED_WRITE_VERSION) {
            return Err(ArtifactStoreError::ReservedWritesUnsupported);
        }
        let AttemptArtifactFileWrite {
            publication_key,
            artifact_kind,
            artifact_role,
            metadata_json,
            runtime_retention_policy,
            object,
        } = write;
        self.validate_attempt_artifact(
            lease,
            &publication_key,
            &runtime_retention_policy,
            &metadata_json,
        )?;
        let stored = self.store_reserved_file(tenant, object).await?;
        self.publish_stored_attempt_artifact(
            tenant,
            lease,
            publication_key,
            artifact_kind,
            artifact_role,
            metadata_json,
            runtime_retention_policy,
            stored,
            publication_progress,
        )
        .await
    }

    /// Write the final Fish WAV through the bounded provider file primitive
    /// and settle its route projection and runtime stage in one metadata
    /// transaction. The provider object is never exposed as a public path.
    pub(crate) async fn put_attempt_speech_history_file(
        &self,
        tenant: &ArtifactTenant,
        lease: &StageLease,
        object: ArtifactFileWrite,
        finalization: SpeechHistoryFinalization,
    ) -> Result<RuntimeArtifact, ArtifactStoreError> {
        if !self.provider.supports_reserved_file_writes() {
            return Err(ArtifactStoreError::ReservedFileWritesUnsupported);
        }
        if self.provider.reserved_write_protocol_version() != Some(MEDIA_RESERVED_WRITE_VERSION) {
            return Err(ArtifactStoreError::ReservedWritesUnsupported);
        }
        self.validate_file_write(&object)?;
        let stored = self.store_reserved_file(tenant, object).await?;
        let media = opaque_media_asset(tenant, &stored);
        let artifact = NewStageOutputArtifact {
            publication_key: "speech-final-audio/v1".to_string(),
            artifact_kind: RuntimeArtifactKind::Audio,
            artifact_role: RuntimeArtifactRole::OutputPrimary,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: None,
            content_type: None,
            filename: None,
            size_bytes: None,
            sha256: None,
            metadata_json: finalization.artifact_metadata_json.clone(),
            retention_policy: "speech_history".to_string(),
        };
        match self
            .metadata
            .settle_reserved_speech_history_audio(
                &stored.reservation,
                lease,
                media,
                artifact,
                finalization,
            )
            .await
        {
            Ok(Some(ReservedStageArtifactPublication::Published { artifact, .. }))
            | Ok(Some(ReservedStageArtifactPublication::Existing(artifact))) => Ok(artifact),
            Ok(None) => Err(ArtifactStoreError::Metadata(anyhow::anyhow!(
                "Stage attempt lost ownership before speech final audio settlement"
            ))),
            Err(error) => Err(ArtifactStoreError::Metadata(error)),
        }
    }

    fn validate_attempt_artifact(
        &self,
        lease: &StageLease,
        publication_key: &str,
        runtime_retention_policy: &str,
        metadata_json: &serde_json::Value,
    ) -> Result<(), ArtifactStoreError> {
        if lease.attempt_token.is_none() {
            return Err(ArtifactStoreError::InvalidInput(
                "stage lease lacks an attempt token",
            ));
        }
        let publication_key = publication_key.trim();
        if publication_key.is_empty()
            || publication_key.len() > 256
            || publication_key.chars().any(char::is_control)
        {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid attempt publication key",
            ));
        }
        if runtime_retention_policy.is_empty()
            || runtime_retention_policy.len() > 128
            || runtime_retention_policy.chars().any(char::is_control)
        {
            return Err(ArtifactStoreError::InvalidInput(
                "invalid runtime retention policy",
            ));
        }
        let metadata_bytes = serde_json::to_vec(metadata_json)
            .map_err(|_| ArtifactStoreError::InvalidInput("invalid artifact metadata"))?;
        if metadata_bytes.len() > 8 * 1024 {
            return Err(ArtifactStoreError::InvalidInput(
                "attempt artifact metadata exceeds 8 KiB",
            ));
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn publish_stored_attempt_artifact(
        &self,
        tenant: &ArtifactTenant,
        lease: &StageLease,
        publication_key: String,
        artifact_kind: RuntimeArtifactKind,
        artifact_role: RuntimeArtifactRole,
        metadata_json: serde_json::Value,
        runtime_retention_policy: String,
        stored: StoredReservedArtifact,
        publication_progress: serde_json::Value,
    ) -> Result<RuntimeArtifact, ArtifactStoreError> {
        let media = opaque_media_asset(tenant, &stored);
        let artifact = NewStageOutputArtifact {
            publication_key: publication_key.trim().to_string(),
            artifact_kind,
            artifact_role,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: None,
            content_type: None,
            filename: None,
            size_bytes: None,
            sha256: None,
            metadata_json,
            retention_policy: runtime_retention_policy,
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

    pub async fn read_stream(
        &self,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<ArtifactStream, ArtifactStoreError> {
        let asset = self.resolve_active(tenant, id).await?;
        if asset.size_bytes == 0 || asset.size_bytes > self.limits.max_file_object_bytes {
            return Err(ArtifactStoreError::TooLarge);
        }
        let expected_digest = validate_digest(asset.sha256.as_deref())?;
        let stream = self
            .provider
            .get_stream(MediaReadRequest {
                key: MediaObjectKey::new(asset.storage_key.clone()),
                metadata: tenant_metadata(tenant),
            })
            .await
            .map_err(map_provider_read_error)?;
        validate_stored_read_metadata(tenant, &asset, &stream.metadata, &expected_digest)?;
        let descriptor = descriptor_from_asset(&asset)?;
        Ok(ArtifactStream {
            descriptor,
            reader: Box::pin(VerifiedArtifactReader {
                inner: stream.reader,
                expected_size: asset.size_bytes,
                expected_digest,
                total: 0,
                digest: Some(Sha256::new()),
                finished: false,
                scratch: Box::new([0; READ_CHUNK_BYTES]),
            }),
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

    pub(crate) async fn delete_speech_history_audio(
        &self,
        route_kind: &str,
        record_id: &str,
        tenant: &ArtifactTenant,
        id: &ArtifactId,
    ) -> Result<bool, ArtifactStoreError> {
        self.resolve_active(tenant, id).await?;
        let Some(intent) = self
            .metadata
            .delete_opaque_speech_history_record(
                route_kind,
                record_id,
                id.as_str(),
                tenant.as_str(),
            )
            .await
            .map_err(ArtifactStoreError::Metadata)?
        else {
            return Ok(false);
        };
        // Logical deletion is already durable. Provider failure is recorded on
        // this exact intent and retried by bounded maintenance.
        if let Err(error) = self.cleanup_intent(&intent).await {
            tracing::warn!(
                artifact_id = id.as_str(),
                record_id,
                error = %error,
                "Opaque speech-history audio is logically deleted; immediate provider cleanup bookkeeping failed"
            );
        }
        Ok(true)
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

    /// Delete opaque objects whose retention label has lapsed: ephemeral
    /// objects past their retention and job objects of jobs past job
    /// retention. Durable objects are never expired here. An object whose
    /// provider delete fails stays tombstoned for `cleanup_due` to retry.
    /// Then delete `default`-labelled input media nothing uses any more
    /// (see `BatchRuntimeStore::expired_input_media`); a failed delete
    /// leaves that row visible for the next pass.
    pub async fn expire_retained(
        &self,
        policy: RetentionPolicy,
        limit: usize,
    ) -> Result<ArtifactCleanupReport, ArtifactStoreError> {
        let assets = self
            .metadata
            .expired_retained_media(policy, limit.min(64))
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        let mut report = ArtifactCleanupReport::default();
        for asset in assets {
            report.inspected += 1;
            let tenant =
                tenant_from_asset(&asset).and_then(|tenant| ArtifactTenant::parse(tenant).ok());
            let (Some(tenant), Ok(id)) = (tenant, ArtifactId::parse(asset.id.clone())) else {
                report.deferred += 1;
                continue;
            };
            match self.delete(&tenant, &id).await {
                Ok(_) => report.completed += 1,
                Err(ArtifactStoreError::DeleteIncomplete) => report.deferred += 1,
                Err(error) => return Err(error),
            }
        }
        let remaining = limit.min(64).saturating_sub(report.inspected);
        if remaining == 0 {
            return Ok(report);
        }
        let inputs = self
            .metadata
            .expired_input_media(policy, remaining)
            .await
            .map_err(ArtifactStoreError::Metadata)?;
        for asset in inputs {
            report.inspected += 1;
            if !self
                .metadata
                .claim_expired_input_media(&asset.id, policy)
                .await
                .map_err(ArtifactStoreError::Metadata)?
            {
                report.deferred += 1;
                continue;
            }
            let deleted =
                crate::persistence::delete_media_object(&self.provider, Some(&asset.storage_key))
                    .await;
            if let Err(error) = &deleted {
                tracing::warn!(
                    media_asset_id = %asset.id,
                    "Expired input media object was not deleted; retrying next pass: {error:#}"
                );
            }
            self.metadata
                .settle_expired_input_media(&asset.id, deleted.is_ok())
                .await
                .map_err(ArtifactStoreError::Metadata)?;
            if deleted.is_ok() {
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
        let reservation = self
            .reserve_write(
                tenant,
                &write.content_type,
                write.filename.as_deref(),
                size_bytes,
                &digest,
            )
            .await?;
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

        self.finish_reserved_write(
            tenant,
            write.content_type,
            write.filename,
            size_bytes,
            digest,
            write.retention,
            self.limits.max_object_bytes,
            reservation,
            stored,
        )
        .await
    }

    async fn store_reserved_file(
        &self,
        tenant: &ArtifactTenant,
        write: ArtifactFileWrite,
    ) -> Result<StoredReservedArtifact, ArtifactStoreError> {
        if !self.provider.supports_reserved_file_writes() {
            return Err(ArtifactStoreError::ReservedFileWritesUnsupported);
        }
        if self.provider.reserved_write_protocol_version() != Some(MEDIA_RESERVED_WRITE_VERSION) {
            return Err(ArtifactStoreError::ReservedWritesUnsupported);
        }
        self.validate_file_write(&write)?;
        let reservation = self
            .reserve_write(
                tenant,
                &write.content_type,
                write.filename.as_deref(),
                write.size_bytes,
                &write.sha256,
            )
            .await?;
        let provider_request = provider_write_request(&reservation);
        let stored = match tokio::time::timeout(
            PROVIDER_WRITE_TIMEOUT,
            self.provider.put_reserved_file(
                MediaReservedWriteRequest {
                    version: MEDIA_RESERVED_WRITE_VERSION,
                    write_id: reservation.write_id.clone(),
                    expires_at_unix_ms: reservation.expires_at,
                    content_length: reservation.expected_size_bytes,
                    sha256: reservation.expected_sha256.clone(),
                    request: provider_request,
                },
                write.path,
            ),
        )
        .await
        {
            Ok(Ok(stored)) => stored,
            Ok(Err(_)) | Err(_) => {
                self.abandon_write(&reservation, "Reserved provider file write failed")
                    .await;
                return Err(ArtifactStoreError::Provider);
            }
        };

        self.finish_reserved_write(
            tenant,
            write.content_type,
            write.filename,
            write.size_bytes,
            write.sha256,
            write.retention,
            self.limits.max_file_object_bytes,
            reservation,
            stored,
        )
        .await
    }

    async fn reserve_write(
        &self,
        tenant: &ArtifactTenant,
        content_type: &str,
        filename: Option<&str>,
        size_bytes: u64,
        digest: &str,
    ) -> Result<ProviderWriteReservation, ArtifactStoreError> {
        let write_id = new_uuid();
        let provider_request = MediaWriteRequest {
            namespace: MediaNamespace::Other("artifact-store".to_string()),
            record_id: write_id.clone(),
            preferred_filename: filename.map(str::to_string),
            content_type: content_type.to_string(),
            metadata: tenant_metadata(tenant),
        };
        self.metadata
            .reserve_provider_write(NewProviderWriteReservation {
                write_id,
                tenant_scope: tenant.as_str().to_string(),
                storage_namespace: "artifact-store".to_string(),
                content_type: content_type.to_string(),
                filename: filename.map(str::to_string),
                expected_size_bytes: size_bytes,
                expected_sha256: digest.to_string(),
                lifetime_ms: self.provider_write_lifetime().as_millis() as u64,
                provider_request,
            })
            .await
            .map_err(ArtifactStoreError::Metadata)
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_reserved_write(
        &self,
        tenant: &ArtifactTenant,
        content_type: String,
        filename: Option<String>,
        size_bytes: u64,
        digest: String,
        retention: ArtifactRetention,
        max_object_bytes: u64,
        reservation: ProviderWriteReservation,
        stored: StoredMediaObject,
    ) -> Result<StoredReservedArtifact, ArtifactStoreError> {
        if let Err(error) = validate_stored_write(
            tenant,
            &content_type,
            digest.as_str(),
            size_bytes,
            stored.metadata.content_length,
            stored.metadata.sha256.as_deref(),
            stored.metadata.tenant_id.as_deref(),
            stored.metadata.content_type.as_str(),
            max_object_bytes,
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
            content_type,
            filename,
            size_bytes,
            sha256: digest,
            retention,
        })
    }

    fn validate_write(&self, write: &ArtifactWrite) -> Result<(), ArtifactStoreError> {
        if write.bytes.is_empty() {
            return Err(ArtifactStoreError::InvalidInput("empty artifact"));
        }
        if write.bytes.len() as u64 > self.limits.max_object_bytes {
            return Err(ArtifactStoreError::TooLarge);
        }
        self.validate_object_identity(&write.content_type, write.filename.as_deref())
    }

    fn validate_file_write(&self, write: &ArtifactFileWrite) -> Result<(), ArtifactStoreError> {
        if write.size_bytes == 0 {
            return Err(ArtifactStoreError::InvalidInput("empty artifact"));
        }
        if write.size_bytes > self.limits.max_file_object_bytes {
            return Err(ArtifactStoreError::TooLarge);
        }
        validate_digest(Some(&write.sha256))?;
        self.validate_object_identity(&write.content_type, write.filename.as_deref())?;
        let metadata = std::fs::symlink_metadata(&write.path)
            .map_err(|_| ArtifactStoreError::InvalidInput("artifact file is unavailable"))?;
        if !metadata.file_type().is_file() {
            return Err(ArtifactStoreError::InvalidInput(
                "artifact file must be a regular file",
            ));
        }
        Ok(())
    }

    fn validate_object_identity(
        &self,
        content_type: &str,
        filename: Option<&str>,
    ) -> Result<(), ArtifactStoreError> {
        validate_content_type(content_type, self.limits.max_content_type_bytes)?;
        if let Some(filename) = filename {
            if filename.is_empty()
                || filename.len() > self.limits.max_filename_bytes
                || filename.chars().any(char::is_control)
                || filename.contains('/')
                || filename.contains('\\')
                || matches!(filename, "." | "..")
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
    let digest = digest.ok_or(ArtifactStoreError::Integrity(
        "provider omitted reserved-write digest",
    ))?;
    if validate_digest(Some(digest))? != expected_digest {
        return Err(ArtifactStoreError::Integrity("provider digest mismatch"));
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
#[path = "artifact_store_tests.rs"]
mod tests;

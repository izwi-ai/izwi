//! The files of `/v1/files`: batch input a client uploads, audio a batch
//! line names, and the output a batch writes.
//!
//! A file's bytes are an opaque artifact of its tenant, so they live in the
//! media provider with the artifact store's integrity checks and durable
//! deletion. The `api_files` row is what the Files API lists; its id is the
//! artifact's id as `file-<hex>`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Context;
use sea_orm::{ConnectionTrait, QueryResult};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::artifact_store::{
    ArtifactFileWrite, ArtifactId, ArtifactRetention, ArtifactStore, ArtifactStoreError,
    ArtifactStream, ArtifactTenant,
};
use crate::db::{raw, StoreDatabase};
use crate::error::StoreError;

pub(crate) const FILE_ID_PREFIX: &str = "file-";

/// How long a `batch_output` file is kept, as OpenAI keeps batch output.
const BATCH_OUTPUT_RETENTION_MS: i64 = 30 * 24 * 60 * 60 * 1000;

/// Files are opaque bytes to the store, whatever they hold: the media
/// provider keeps only the media types it can name from an object's key.
pub(crate) const FILE_CONTENT_TYPE: &str = "application/octet-stream";

/// The largest file the artifact store takes.
const ABSOLUTE_MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;

/// What a file is for. `BatchOutput` is written by the server only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FilePurpose {
    Batch,
    UserData,
    BatchOutput,
}

impl FilePurpose {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::UserData => "user_data",
            Self::BatchOutput => "batch_output",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "batch" => Some(Self::Batch),
            "user_data" => Some(Self::UserData),
            "batch_output" => Some(Self::BatchOutput),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredFile {
    pub(crate) id: String,
    pub(crate) purpose: FilePurpose,
    pub(crate) filename: String,
    pub(crate) size_bytes: u64,
    /// Epoch milliseconds.
    pub(crate) created_at: i64,
    pub(crate) expires_at: Option<i64>,
}

/// A file being written to local disk on its way into the store, hashed as
/// it is written. The temporary file goes when the spool is dropped.
pub(crate) struct FileSpool {
    file: tempfile::NamedTempFile,
    writer: tokio::fs::File,
    digest: Sha256,
    size_bytes: u64,
}

impl FileSpool {
    pub(crate) fn new() -> std::io::Result<Self> {
        let file = tempfile::NamedTempFile::new()?;
        let writer = tokio::fs::File::from_std(file.reopen()?);
        Ok(Self {
            file,
            writer,
            digest: Sha256::new(),
            size_bytes: 0,
        })
    }

    pub(crate) async fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(bytes).await?;
        self.digest.update(bytes);
        self.size_bytes = self.size_bytes.saturating_add(bytes.len() as u64);
        Ok(())
    }

    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

/// A page of `list`: which files, in which order.
#[derive(Debug, Clone, Default)]
pub(crate) struct FileListQuery {
    pub(crate) purpose: Option<FilePurpose>,
    pub(crate) after: Option<String>,
    pub(crate) limit: usize,
    pub(crate) ascending: bool,
}

pub struct FileStore {
    db: StoreDatabase,
    artifacts: Arc<ArtifactStore>,
    max_bytes: AtomicU64,
}

const FILE_COLUMNS: &str = "id, purpose, filename, size_bytes, created_at, expires_at";

fn map_file(row: &QueryResult) -> anyhow::Result<StoredFile> {
    let purpose: String = row.try_get_by_index(1)?;
    let size_bytes: i64 = row.try_get_by_index(3)?;
    Ok(StoredFile {
        id: row.try_get_by_index(0)?,
        purpose: FilePurpose::parse(&purpose)
            .with_context(|| format!("Unknown stored file purpose `{purpose}`"))?,
        filename: row.try_get_by_index(2)?,
        size_bytes: u64::try_from(size_bytes).unwrap_or(0),
        created_at: row.try_get_by_index(4)?,
        expires_at: row.try_get_by_index(5)?,
    })
}

/// The artifact holding the bytes of file `id`; `None` for anything that is
/// not a file id.
fn artifact_id(id: &str) -> Option<ArtifactId> {
    let hex = id.strip_prefix(FILE_ID_PREFIX)?;
    if hex.len() != 32 {
        return None;
    }
    let uuid = uuid::Uuid::parse_str(hex).ok()?;
    ArtifactId::parse(uuid.hyphenated().to_string()).ok()
}

fn file_id(artifact: &ArtifactId) -> anyhow::Result<String> {
    let uuid = uuid::Uuid::parse_str(artifact.as_str()).context("Artifact id is not a UUID")?;
    Ok(format!("{FILE_ID_PREFIX}{}", uuid.simple()))
}

fn artifact_error(error: ArtifactStoreError) -> anyhow::Error {
    match error {
        ArtifactStoreError::TooLarge => {
            StoreError::too_large("The file exceeds the size limit").into()
        }
        ArtifactStoreError::InvalidInput(reason) => {
            StoreError::invalid(format!("The file was refused: {reason}")).into()
        }
        ArtifactStoreError::NotFound => StoreError::not_found("File content not found").into(),
        error => anyhow::Error::new(error).context("File storage failed"),
    }
}

impl FileStore {
    pub(crate) fn new(db: StoreDatabase, artifacts: Arc<ArtifactStore>) -> Self {
        Self {
            db,
            artifacts,
            max_bytes: AtomicU64::new(
                crate::env_config::tuning()
                    .files_max_bytes
                    .min(ABSOLUTE_MAX_FILE_BYTES),
            ),
        }
    }

    /// The largest upload accepted (`IZWI_FILES_MAX_BYTES`).
    pub(crate) fn max_bytes(&self) -> u64 {
        self.max_bytes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set_max_bytes_for_test(&self, max_bytes: u64) {
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
    }

    /// Store what `spool` holds as `tenant`'s file `filename`. A
    /// `batch_output` file expires 30 days later; uploads are kept until
    /// deleted.
    pub(crate) async fn create(
        &self,
        tenant: &ArtifactTenant,
        purpose: FilePurpose,
        filename: String,
        mut spool: FileSpool,
    ) -> anyhow::Result<StoredFile> {
        spool
            .writer
            .flush()
            .await
            .context("Failed to spool the file")?;
        let size_bytes = spool.size_bytes;
        let descriptor = self
            .artifacts
            .put_file(
                tenant,
                ArtifactFileWrite {
                    content_type: FILE_CONTENT_TYPE.to_string(),
                    filename: Some(filename.clone()),
                    path: spool.file.path().to_path_buf(),
                    size_bytes,
                    sha256: hex::encode(spool.digest.finalize()),
                    retention: ArtifactRetention::Durable,
                },
            )
            .await
            .map_err(artifact_error)?;
        let id = file_id(&descriptor.id)?;
        let created_at = self.db.now_millis();
        let expires_at = (purpose == FilePurpose::BatchOutput)
            .then(|| created_at.saturating_add(BATCH_OUTPUT_RETENTION_MS));
        let db = self.db.connection().await?;
        let inserted = db
            .execute_raw(raw::statement(
                db,
                "INSERT INTO api_files (id, tenant_scope, purpose, filename, size_bytes, \
                 created_at, expires_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                vec![
                    id.clone().into(),
                    tenant.as_str().into(),
                    purpose.as_str().into(),
                    filename.clone().into(),
                    i64::try_from(size_bytes)?.into(),
                    created_at.into(),
                    expires_at.into(),
                ],
            )?)
            .await;
        if let Err(error) = inserted {
            // The bytes without their row are unreachable: remove them.
            let _ = self.artifacts.delete(tenant, &descriptor.id).await;
            return Err(error).context("Failed to record the file");
        }
        Ok(StoredFile {
            id,
            purpose,
            filename,
            size_bytes,
            created_at,
            expires_at,
        })
    }

    /// `tenant`'s file `id`, unless it has expired.
    pub(crate) async fn get(
        &self,
        tenant: &ArtifactTenant,
        id: &str,
    ) -> anyhow::Result<Option<StoredFile>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                format!(
                    "SELECT {FILE_COLUMNS} FROM api_files WHERE id = ?1 AND tenant_scope = ?2 \
                     AND (expires_at IS NULL OR expires_at > ?3)"
                ),
                vec![
                    id.into(),
                    tenant.as_str().into(),
                    self.db.now_millis().into(),
                ],
            )?)
            .await
            .context("Failed to load the file")?;
        row.as_ref().map(map_file).transpose()
    }

    /// Up to `query.limit` of `tenant`'s files after the file `query.after`,
    /// newest first unless ascending. Returns whether more follow.
    pub(crate) async fn list(
        &self,
        tenant: &ArtifactTenant,
        query: FileListQuery,
    ) -> anyhow::Result<(Vec<StoredFile>, bool)> {
        let cursor = match query.after.as_deref() {
            Some(after) => Some(
                self.get(tenant, after)
                    .await?
                    .ok_or_else(|| StoreError::invalid(format!("Unknown cursor `{after}`")))?,
            ),
            None => None,
        };
        let (beyond, order) = if query.ascending {
            (">", "ASC")
        } else {
            ("<", "DESC")
        };
        let limit = query.limit.clamp(1, 10_000);
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    "SELECT {FILE_COLUMNS} FROM api_files WHERE tenant_scope = ?1 \
                     AND (expires_at IS NULL OR expires_at > ?2) \
                     AND (?3 IS NULL OR purpose = ?3) \
                     AND (?4 IS NULL OR created_at {beyond} ?4 \
                          OR (created_at = ?4 AND id {beyond} ?5)) \
                     ORDER BY created_at {order}, id {order} LIMIT ?6"
                ),
                vec![
                    tenant.as_str().into(),
                    self.db.now_millis().into(),
                    query.purpose.map(FilePurpose::as_str).into(),
                    cursor.as_ref().map(|file| file.created_at).into(),
                    cursor.map(|file| file.id).into(),
                    i64::try_from(limit + 1)?.into(),
                ],
            )?)
            .await
            .context("Failed to list files")?;
        let mut files = rows
            .iter()
            .map(map_file)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let has_more = files.len() > limit;
        files.truncate(limit);
        Ok((files, has_more))
    }

    /// The content of `tenant`'s file `id`, verified as it is read.
    pub(crate) async fn open(
        &self,
        tenant: &ArtifactTenant,
        id: &str,
    ) -> anyhow::Result<Option<(StoredFile, ArtifactStream)>> {
        let (Some(file), Some(artifact)) = (self.get(tenant, id).await?, artifact_id(id)) else {
            return Ok(None);
        };
        match self.artifacts.read_stream(tenant, &artifact).await {
            Ok(stream) => Ok(Some((file, stream))),
            Err(ArtifactStoreError::NotFound) => Ok(None),
            Err(error) => Err(artifact_error(error)),
        }
    }

    /// Delete `tenant`'s file `id`; `false` when there is no such file.
    pub(crate) async fn delete(&self, tenant: &ArtifactTenant, id: &str) -> anyhow::Result<bool> {
        let Some(artifact) = artifact_id(id) else {
            return Ok(false);
        };
        // The bytes go first: a row without bytes can be deleted again, bytes
        // without a row could not be found. A provider delete that fails is
        // retried by the artifact store's own cleanup.
        match self.artifacts.delete(tenant, &artifact).await {
            Ok(_) | Err(ArtifactStoreError::DeleteIncomplete) => {}
            Err(error) => return Err(artifact_error(error)),
        }
        let db = self.db.connection().await?;
        let deleted = db
            .execute_raw(raw::statement(
                db,
                "DELETE FROM api_files WHERE id = ?1 AND tenant_scope = ?2",
                vec![id.into(), tenant.as_str().into()],
            )?)
            .await
            .context("Failed to delete the file")?;
        Ok(deleted.rows_affected() > 0)
    }

    /// Delete up to `limit` files whose `expires_at` has passed. Returns how
    /// many were deleted.
    pub(crate) async fn delete_expired(&self, limit: usize) -> anyhow::Result<usize> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT id, tenant_scope FROM api_files WHERE expires_at IS NOT NULL \
                 AND expires_at <= ?1 ORDER BY expires_at LIMIT ?2",
                vec![self.db.now_millis().into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to find expired files")?;
        let mut deleted = 0;
        for row in rows {
            let id: String = row.try_get_by_index(0)?;
            let tenant: String = row.try_get_by_index(1)?;
            let tenant = ArtifactTenant::parse(tenant)
                .map_err(|error| anyhow::anyhow!("File {id} has an invalid tenant: {error}"))?;
            if self.delete(&tenant, &id).await? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Make file `id` expire at `expires_at`.
    #[cfg(test)]
    pub(crate) async fn set_expires_at_for_test(
        &self,
        id: &str,
        expires_at: i64,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "UPDATE api_files SET expires_at = ?1 WHERE id = ?2",
            vec![expires_at.into(), id.into()],
        )?)
        .await?;
        Ok(())
    }
}

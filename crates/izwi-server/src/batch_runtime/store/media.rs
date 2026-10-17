//! Media and text assets, tombstones and the artifact cleanup outbox.

use super::*;

#[derive(Debug, Clone)]
pub struct NewMediaAsset {
    pub asset_kind: String,
    pub storage_namespace: String,
    pub storage_key: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub duration_secs: Option<f64>,
    pub sample_rate_hz: Option<u32>,
    pub channel_count: Option<u16>,
    pub peak_amplitude: Option<f32>,
    pub rms_amplitude: Option<f32>,
    pub source_asset_id: Option<String>,
    pub canonical_profile_version: Option<String>,
    pub scan_status: String,
    pub retention_policy: String,
    pub metadata_json: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactCleanupReason {
    ArtifactDeleted,
}

impl ArtifactCleanupReason {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::ArtifactDeleted => "artifact_deleted",
        }
    }

    pub(super) fn from_db_value(value: &str) -> anyhow::Result<Self> {
        match value {
            "artifact_deleted" => Ok(Self::ArtifactDeleted),
            _ => bail!("Unknown artifact cleanup reason"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArtifactCleanupIntent {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub available_at: u64,
    pub storage_key: String,
    pub tenant_scope: String,
    pub reason: ArtifactCleanupReason,
    pub attempt_count: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewTextAsset {
    pub raw_text: String,
    pub normalized_text: Option<String>,
    pub language_hint: Option<String>,
    pub sha256: Option<String>,
    pub safety_status: String,
    pub retention_policy: String,
    pub structure_json: serde_json::Value,
}

pub(super) const MAX_ARTIFACT_CLEANUP_INTENTS: u64 = 65_536;

/// Pending cleanup intents one tenant may hold, so one tenant's backlog
/// cannot block every other tenant's deletes.
pub(super) const MAX_ARTIFACT_CLEANUP_INTENTS_PER_TENANT: u64 = 16_384;

pub(super) const MAX_ARTIFACT_CLEANUP_BATCH: usize = 64;

pub(super) const MAX_ARTIFACT_CLEANUP_STORAGE_KEY_BYTES: usize = 2 * 1024;

pub(super) const MAX_ARTIFACT_CLEANUP_TENANT_BYTES: usize = 128;

pub(super) const MAX_ARTIFACT_CLEANUP_ERROR_BYTES: usize = 512;

pub(super) const MAX_ARTIFACT_CLEANUP_BACKOFF_MS: u64 = 60 * 60 * 1000;

pub(crate) fn validate_artifact_cleanup_storage_key(storage_key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !storage_key.is_empty()
            && storage_key.len() <= MAX_ARTIFACT_CLEANUP_STORAGE_KEY_BYTES
            && !storage_key.chars().any(char::is_control),
        "Invalid artifact cleanup storage key"
    );
    Ok(())
}

pub(super) fn validate_artifact_cleanup_tenant(tenant_scope: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !tenant_scope.is_empty()
            && tenant_scope.len() <= MAX_ARTIFACT_CLEANUP_TENANT_BYTES
            && tenant_scope
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-.:@".contains(&byte)),
        "Invalid artifact cleanup tenant scope"
    );
    Ok(())
}

pub(super) fn truncate_utf8_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut boundary = max_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value[..boundary].to_string()
}

pub(super) async fn get_media_asset_with<C: ConnectionTrait>(
    db: &C,
    id: &str,
) -> anyhow::Result<Option<MediaAsset>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            MEDIA_ASSET_BY_ID_SQL,
            vec![id.into()],
        )?)
        .await?;
    row.as_ref().map(map_media_asset).transpose()
}

pub(super) async fn tombstone_media_asset_with_cleanup_in<C: ConnectionTrait>(
    db: &C,
    id: &str,
    tenant_scope: &str,
    reason: ArtifactCleanupReason,
    limits: QuotaLimits,
    now: i64,
) -> anyhow::Result<Option<(bool, ArtifactCleanupIntent)>> {
    let history_reference = db
        .query_one_raw(raw::statement(
            db,
            r#"
            SELECT 1
            FROM speech_history_records
            WHERE audio_media_asset_id = ?1
            LIMIT 1
            "#,
            vec![id.into()],
        )?)
        .await
        .context("Failed to inspect opaque artifact speech-history ownership")?;
    anyhow::ensure!(
        history_reference.is_none(),
        "Opaque artifact is still owned by a speech-history record"
    );
    let Some(asset) = db
        .query_one_raw(raw::statement(
            db,
            r#"
            SELECT storage_key, deleted_at
            FROM media_assets
            WHERE id = ?1
              AND asset_kind = 'opaque_artifact'
              AND storage_namespace = 'artifact_store_v1'
            "#,
            vec![id.into()],
        )?)
        .await
        .context("Failed to load opaque artifact for deletion")?
    else {
        return Ok(None);
    };
    let storage_key: String = asset.try_get_by_index(0)?;
    validate_artifact_cleanup_storage_key(&storage_key)?;
    let already_deleted = asset.try_get_by_index::<Option<i64>>(1)?.is_some();

    let intent = match artifact_cleanup_intent_with(db, &storage_key).await? {
        Some(intent) => intent,
        None => {
            anyhow::ensure!(
                acquire_quota_with(db, QuotaKind::ArtifactCleanup, tenant_scope, limits).await?,
                "Artifact cleanup capacity exhausted ({} pending intents, {} per tenant)",
                limits.global,
                limits.per_tenant
            );
            let intent_id = new_uuid();
            let inserted = db
                .execute_raw(raw::statement(
                    db,
                    r#"
                    INSERT INTO artifact_cleanup_intents (
                        id, created_at, updated_at, available_at, storage_key,
                        tenant_scope, reason, attempt_count, last_error
                    )
                    VALUES (?1, ?2, ?2, ?2, ?3, ?4, ?5, 0, NULL)
                    ON CONFLICT (storage_key) DO NOTHING
                    "#,
                    vec![
                        intent_id.clone().into(),
                        now.into(),
                        storage_key.clone().into(),
                        tenant_scope.into(),
                        reason.as_db_value().into(),
                    ],
                )?)
                .await
                .context("Failed to persist artifact cleanup intent")?
                .rows_affected()
                == 1;
            if inserted {
                ArtifactCleanupIntent {
                    id: intent_id,
                    created_at: u64::try_from(now)?,
                    updated_at: u64::try_from(now)?,
                    available_at: u64::try_from(now)?,
                    storage_key: storage_key.clone(),
                    tenant_scope: tenant_scope.to_string(),
                    reason,
                    attempt_count: 0,
                    last_error: None,
                }
            } else {
                // A concurrent delete of the same object committed its
                // intent first: give the unit back and share that intent.
                release_quota_with(db, QuotaKind::ArtifactCleanup, tenant_scope, 1).await?;
                artifact_cleanup_intent_with(db, &storage_key)
                    .await?
                    .ok_or_else(|| anyhow!("Conflicting artifact cleanup intent disappeared"))?
            }
        }
    };
    anyhow::ensure!(
        intent.tenant_scope == tenant_scope,
        "Artifact cleanup key is already owned by another tenant"
    );

    let newly_deleted = if already_deleted {
        false
    } else {
        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE media_assets
            SET updated_at = ?1, deleted_at = ?1
            WHERE id = ?2 AND deleted_at IS NULL
            "#,
            vec![now.into(), id.into()],
        )?)
        .await
        .context("Failed to tombstone media asset")?
        .rows_affected()
            == 1
    };
    Ok(Some((newly_deleted, intent)))
}

async fn artifact_cleanup_intent_with<C: ConnectionTrait>(
    db: &C,
    storage_key: &str,
) -> anyhow::Result<Option<ArtifactCleanupIntent>> {
    db.query_one_raw(raw::statement(
        db,
        r#"
        SELECT id, created_at, updated_at, available_at, storage_key,
               tenant_scope, reason, attempt_count, last_error
        FROM artifact_cleanup_intents
        WHERE storage_key = ?1
        "#,
        vec![storage_key.into()],
    )?)
    .await
    .context("Failed to load artifact cleanup intent")?
    .as_ref()
    .map(map_artifact_cleanup_intent)
    .transpose()
}

impl BatchRuntimeStore {
    pub async fn create_media_asset(&self, input: NewMediaAsset) -> anyhow::Result<MediaAsset> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO media_assets (
                id,
                created_at,
                updated_at,
                asset_kind,
                storage_namespace,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                duration_secs,
                sample_rate_hz,
                channel_count,
                peak_amplitude,
                rms_amplitude,
                source_asset_id,
                canonical_profile_version,
                scan_status,
                retention_policy,
                deleted_at,
                metadata_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, NULL, ?19)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                input.asset_kind.into(),
                input.storage_namespace.into(),
                input.storage_key.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.size_bytes)?,
                opt_string(input.sha256),
                opt_f64(input.duration_secs),
                opt_u32(input.sample_rate_hz),
                opt_u16(input.channel_count),
                opt_f32(input.peak_amplitude),
                opt_f32(input.rms_amplitude),
                opt_string(input.source_asset_id),
                opt_string(input.canonical_profile_version),
                input.scan_status.into(),
                input.retention_policy.into(),
                metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to create media asset")?;

        self.get_media_asset(&id)
            .await?
            .ok_or_else(|| anyhow!("Created media asset was not found"))
    }

    pub async fn get_media_asset(&self, id: &str) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_BY_ID_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load media asset")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    /// Logically delete an artifact-backed media row before physical object cleanup.
    ///
    /// The tombstone is intentionally retained so a failed provider deletion cannot
    /// make the object visible again. Provider-specific garbage collection may retry
    /// physical deletion without changing the public artifact identity.
    pub async fn tombstone_media_asset(&self, id: &str) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE media_assets
                SET updated_at = ?1, deleted_at = ?1
                WHERE id = ?2 AND deleted_at IS NULL
                "#,
                vec![now.into(), id.into()],
            )?)
            .await
            .context("Failed to tombstone media asset")?;

        Ok(result.rows_affected() == 1)
    }

    /// Atomically hide one opaque artifact and retain its provider key for
    /// idempotent physical deletion. Capacity is taken from the tenant's and
    /// the global cleanup quota before the tombstone changes visibility.
    pub(crate) async fn tombstone_media_asset_with_cleanup(
        &self,
        id: &str,
        tenant_scope: &str,
        reason: ArtifactCleanupReason,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_tenant(tenant_scope)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start artifact cleanup transaction")?;
        let Some((newly_deleted, _)) = tombstone_media_asset_with_cleanup_in(
            &tx,
            id,
            tenant_scope,
            reason,
            self.artifact_cleanup_limits(),
            self.now_millis(),
        )
        .await?
        else {
            tx.rollback().await?;
            return Ok(false);
        };
        tx.commit()
            .await
            .context("Failed to commit artifact cleanup intent")?;
        Ok(newly_deleted)
    }

    pub(crate) async fn due_artifact_cleanup_intents(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ArtifactCleanupIntent>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT c.id, c.created_at, c.updated_at, c.available_at,
                       c.storage_key, c.tenant_scope, c.reason,
                       c.attempt_count, c.last_error
                FROM artifact_cleanup_intents c
                JOIN media_assets m ON m.storage_key = c.storage_key
                WHERE c.available_at <= ?1
                  AND c.reason = 'artifact_deleted'
                  AND m.asset_kind = 'opaque_artifact'
                  AND m.storage_namespace = 'artifact_store_v1'
                  AND m.deleted_at IS NOT NULL
                ORDER BY c.available_at ASC, c.created_at ASC, c.id ASC
                LIMIT ?2
                "#,
                vec![
                    self.now_millis().into(),
                    i64::try_from(limit.min(MAX_ARTIFACT_CLEANUP_BATCH))?.into(),
                ],
            )?)
            .await
            .context("Failed to list due artifact cleanup intents")?;
        rows.iter().map(map_artifact_cleanup_intent).collect()
    }

    pub(crate) async fn artifact_cleanup_intent_for_storage_key(
        &self,
        storage_key: &str,
    ) -> anyhow::Result<Option<ArtifactCleanupIntent>> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        artifact_cleanup_intent_with(db, storage_key).await
    }

    /// Delete a completed intent and give its unit back to the tenant's and
    /// the global cleanup quota.
    pub(crate) async fn complete_artifact_cleanup(
        &self,
        id: &str,
        storage_key: &str,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start artifact cleanup completion")?;
        let Some(intent) = artifact_cleanup_intent_with(&tx, storage_key)
            .await?
            .filter(|intent| intent.id == id)
        else {
            tx.rollback().await?;
            return Ok(false);
        };
        let deleted = tx
            .execute_raw(raw::statement(
                &tx,
                "DELETE FROM artifact_cleanup_intents WHERE id = ?1 AND storage_key = ?2",
                vec![id.into(), storage_key.into()],
            )?)
            .await
            .context("Failed to complete artifact cleanup intent")?
            .rows_affected();
        release_quota_with(
            &tx,
            QuotaKind::ArtifactCleanup,
            &intent.tenant_scope,
            deleted,
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit artifact cleanup completion")?;
        Ok(deleted == 1)
    }

    pub(crate) async fn defer_artifact_cleanup(
        &self,
        intent: &ArtifactCleanupIntent,
        error: &str,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_storage_key(&intent.storage_key)?;
        let attempt_count = intent.attempt_count.saturating_add(1);
        let shift = intent.attempt_count.min(12);
        let backoff_ms = 1_000_u64
            .checked_shl(shift)
            .unwrap_or(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)
            .min(MAX_ARTIFACT_CLEANUP_BACKOFF_MS);
        let now = self.now_millis();
        let available_at = now.saturating_add(i64::try_from(backoff_ms)?);
        let last_error = truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES);
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE artifact_cleanup_intents
                SET updated_at = ?1, available_at = ?2, attempt_count = ?3,
                    last_error = ?4
                WHERE id = ?5 AND storage_key = ?6
                "#,
                vec![
                    now.into(),
                    available_at.into(),
                    i64::from(attempt_count).into(),
                    last_error.into(),
                    intent.id.clone().into(),
                    intent.storage_key.clone().into(),
                ],
            )?)
            .await
            .context("Failed to defer artifact cleanup intent")?;
        Ok(result.rows_affected() == 1)
    }

    pub(super) fn artifact_cleanup_limits(&self) -> QuotaLimits {
        #[cfg(test)]
        if let Some(limits) = self.test_artifact_cleanup_limits {
            return limits;
        }
        QuotaLimits {
            global: MAX_ARTIFACT_CLEANUP_INTENTS,
            per_tenant: MAX_ARTIFACT_CLEANUP_INTENTS_PER_TENANT,
        }
    }

    pub async fn get_media_asset_by_storage_key(
        &self,
        storage_key: &str,
    ) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_BY_STORAGE_KEY_SQL,
                vec![storage_key.into()],
            )?)
            .await
            .context("Failed to load media asset by storage key")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    pub async fn get_canonical_media_asset(
        &self,
        source_asset_id: &str,
        canonical_profile_version: &str,
    ) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_BY_SOURCE_PROFILE_SQL,
                vec![source_asset_id.into(), canonical_profile_version.into()],
            )?)
            .await
            .context("Failed to load canonical media asset by source and profile")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    pub async fn create_text_asset(&self, input: NewTextAsset) -> anyhow::Result<TextAsset> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let normalized_text = input
            .normalized_text
            .clone()
            .unwrap_or_else(|| input.raw_text.clone());
        let character_count = normalized_text.chars().count() as u64;
        let structure_json = json_to_db_string(&input.structure_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO text_assets (
                id,
                created_at,
                updated_at,
                raw_text,
                normalized_text,
                language_hint,
                character_count,
                sha256,
                safety_status,
                retention_policy,
                structure_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                input.raw_text.into(),
                normalized_text.into(),
                opt_string(input.language_hint),
                u64_to_i64_value(character_count)?,
                opt_string(input.sha256),
                input.safety_status.into(),
                input.retention_policy.into(),
                structure_json.into(),
            ],
        )?)
        .await
        .context("Failed to create text asset")?;

        self.get_text_asset(&id)
            .await?
            .ok_or_else(|| anyhow!("Created text asset was not found"))
    }

    pub async fn get_text_asset(&self, id: &str) -> anyhow::Result<Option<TextAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(db, TEXT_ASSET_BY_ID_SQL, vec![id.into()])?)
            .await
            .context("Failed to load text asset")?;

        row.as_ref().map(map_text_asset).transpose()
    }

    pub async fn remove_unreferenced_text_asset(&self, id: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(db,
            "DELETE FROM text_assets WHERE id = ?1 AND NOT EXISTS (SELECT 1 FROM runtime_jobs WHERE input_text_asset_id = ?1) AND NOT EXISTS (SELECT 1 FROM runtime_artifacts WHERE text_asset_id = ?1)",
            vec![id.into()],
        )?).await?;
        Ok(())
    }
}

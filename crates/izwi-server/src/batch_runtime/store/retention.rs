//! The retention sweeper for terminal jobs and expired media.

use super::*;

/// How long terminal and dead durable state is kept before the retention
/// sweeper deletes it. Each value is read from the environment in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Terminal jobs, their stages, artifact rows and text inputs
    /// (`IZWI_JOB_RETENTION_SECONDS`, 30 days). Job-retention media expires
    /// with its job.
    pub terminal_job_ms: u64,
    /// Ephemeral-retention media, from creation
    /// (`IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS`, 1 hour).
    pub ephemeral_artifact_ms: u64,
    /// Heartbeat rows of workers that stopped heartbeating
    /// (`IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS`, 1 day).
    pub worker_heartbeat_ms: u64,
    /// Media rows tombstoned after their objects were deleted
    /// (`IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS`, 1 day).
    pub media_tombstone_ms: u64,
    /// Rows of the retired, never-read legacy idempotency table.
    pub legacy_idempotency_ms: u64,
}

impl RetentionPolicy {
    pub fn from_env() -> Self {
        let read = |key: &str, default_secs: u64| {
            std::env::var(key)
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default_secs)
                .saturating_mul(1_000)
        };
        Self {
            terminal_job_ms: read("IZWI_JOB_RETENTION_SECONDS", 30 * 24 * 60 * 60),
            ephemeral_artifact_ms: read("IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS", 60 * 60),
            worker_heartbeat_ms: read("IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS", 24 * 60 * 60),
            media_tombstone_ms: read("IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS", 24 * 60 * 60),
            legacy_idempotency_ms: MAX_DURABLE_IDEMPOTENCY_RETENTION_MS,
        }
    }
}

/// Rows one retention pass deleted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetentionReport {
    pub idempotency_records: u64,
    pub legacy_idempotency_records: u64,
    pub jobs: u64,
    pub worker_heartbeats: u64,
    pub media_tombstones: u64,
}

impl BatchRuntimeStore {
    /// Delete durable state past its retention, each kind in its own bounded
    /// transaction: expired idempotency records, drained legacy idempotency
    /// rows, terminal jobs with their stages and inputs, heartbeats of dead
    /// worker incarnations, and media tombstones whose objects are gone.
    ///
    /// A terminal job that still owns live job- or ephemeral-retention media
    /// is kept until the artifact store has expired that media, so no object
    /// loses its last reference.
    pub async fn sweep_retention(
        &self,
        policy: RetentionPolicy,
        limit: usize,
    ) -> anyhow::Result<RetentionReport> {
        let limit = bounded_maintenance_batch_limit(limit);
        let now = self.now_millis();
        let mut report = RetentionReport {
            idempotency_records: self.prune_expired_durable_idempotency(limit).await?,
            ..RetentionReport::default()
        };
        let db = self.db.connection().await?;
        let cutoff = |retention_ms: u64| -> anyhow::Result<i64> {
            Ok(now.saturating_sub(i64::try_from(retention_ms)?))
        };

        let tx = db.begin_with_options(write_transaction_options()).await?;
        let legacy = tx
            .query_all_raw(raw::statement(
                &tx,
                "SELECT operation, idempotency_key FROM idempotency_keys WHERE created_at < ?1 ORDER BY created_at ASC LIMIT ?2",
                vec![
                    cutoff(policy.legacy_idempotency_ms)?.into(),
                    i64::try_from(limit)?.into(),
                ],
            )?)
            .await?;
        for row in legacy {
            let operation: String = row.try_get_by_index(0)?;
            let key: String = row.try_get_by_index(1)?;
            report.legacy_idempotency_records += tx
                .execute_raw(raw::statement(
                    &tx,
                    "DELETE FROM idempotency_keys WHERE operation = ?1 AND idempotency_key = ?2",
                    vec![operation.into(), key.into()],
                )?)
                .await?
                .rows_affected();
        }
        tx.commit().await?;

        report.jobs = self
            .delete_expired_jobs(cutoff(policy.terminal_job_ms)?, limit)
            .await?;

        let tx = db.begin_with_options(write_transaction_options()).await?;
        let heartbeat_cutoff = cutoff(policy.worker_heartbeat_ms)?;
        let heartbeats = tx
            .query_all_raw(raw::statement(
                &tx,
                "SELECT worker_id FROM runtime_worker_heartbeats WHERE last_heartbeat_at < ?1 ORDER BY last_heartbeat_at ASC LIMIT ?2",
                vec![heartbeat_cutoff.into(), i64::try_from(limit)?.into()],
            )?)
            .await?;
        for row in heartbeats {
            let worker_id: String = row.try_get_by_index(0)?;
            report.worker_heartbeats += tx
                .execute_raw(raw::statement(
                    &tx,
                    "DELETE FROM runtime_worker_heartbeats WHERE worker_id = ?1 AND last_heartbeat_at < ?2",
                    vec![worker_id.into(), heartbeat_cutoff.into()],
                )?)
                .await?
                .rows_affected();
        }
        tx.commit().await?;

        // A tombstone may go once its cleanup intent completed (the object is
        // deleted) and nothing references the row any more.
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let unreferenced = r#"
            NOT EXISTS (SELECT 1 FROM artifact_cleanup_intents c WHERE c.storage_key = media_assets.storage_key)
            AND NOT EXISTS (SELECT 1 FROM runtime_artifacts a WHERE a.media_asset_id = media_assets.id)
            AND NOT EXISTS (SELECT 1 FROM speech_history_records h WHERE h.audio_media_asset_id = media_assets.id)
            AND NOT EXISTS (SELECT 1 FROM media_assets derived WHERE derived.source_asset_id = media_assets.id)
        "#;
        let tombstone_cutoff = cutoff(policy.media_tombstone_ms)?;
        let tombstones = tx
            .query_all_raw(raw::statement(
                &tx,
                format!(
                    "SELECT id FROM media_assets WHERE deleted_at IS NOT NULL AND deleted_at < ?1 AND {unreferenced} ORDER BY deleted_at ASC, id ASC LIMIT ?2"
                ),
                vec![tombstone_cutoff.into(), i64::try_from(limit)?.into()],
            )?)
            .await?;
        for row in tombstones {
            let id: String = row.try_get_by_index(0)?;
            report.media_tombstones += tx
                .execute_raw(raw::statement(
                    &tx,
                    format!(
                        "DELETE FROM media_assets WHERE id = ?1 AND deleted_at IS NOT NULL AND deleted_at < ?2 AND {unreferenced}"
                    ),
                    vec![id.into(), tombstone_cutoff.into()],
                )?)
                .await?
                .rows_affected();
        }
        tx.commit().await?;

        Ok(report)
    }

    /// Delete one bounded page of terminal jobs last updated before `before`,
    /// with their stages, artifact rows and no longer referenced text inputs.
    pub(super) async fn delete_expired_jobs(&self, before: i64, limit: usize) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let rows = tx
            .query_all_raw(raw::statement(
                &tx,
                r#"
                SELECT j.id, j.input_text_asset_id
                FROM runtime_jobs AS j
                WHERE j.status IN ('completed', 'failed', 'cancelled')
                  AND j.updated_at < ?1
                  AND NOT EXISTS (
                      SELECT 1 FROM runtime_artifacts AS a
                      JOIN media_assets AS m ON m.id = a.media_asset_id
                      WHERE a.job_id = j.id
                        AND m.deleted_at IS NULL
                        AND m.retention_policy IN ('artifact_job', 'artifact_ephemeral')
                  )
                ORDER BY j.updated_at ASC, j.id ASC
                LIMIT ?2
                "#,
                vec![before.into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to select expired runtime jobs")?;
        let mut deleted = 0_u64;
        for row in rows {
            let job_id: String = row.try_get_by_index(0)?;
            let mut text_assets = row
                .try_get_by_index::<Option<String>>(1)?
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
            for artifact in tx
                .query_all_raw(raw::statement(
                    &tx,
                    "SELECT text_asset_id FROM runtime_artifacts WHERE job_id = ?1 AND text_asset_id IS NOT NULL",
                    vec![job_id.clone().into()],
                )?)
                .await?
            {
                text_assets.insert(artifact.try_get_by_index::<String>(0)?);
            }
            // Children first: a provider-managed schema may lack the cascades.
            for sql in [
                "DELETE FROM runtime_artifacts WHERE job_id = ?1 AND EXISTS (SELECT 1 FROM runtime_jobs WHERE id = ?1 AND status IN ('completed', 'failed', 'cancelled') AND updated_at < ?2)",
                "DELETE FROM job_stages WHERE job_id = ?1 AND EXISTS (SELECT 1 FROM runtime_jobs WHERE id = ?1 AND status IN ('completed', 'failed', 'cancelled') AND updated_at < ?2)",
            ] {
                tx.execute_raw(raw::statement(
                    &tx,
                    sql,
                    vec![job_id.clone().into(), before.into()],
                )?)
                .await?;
            }
            deleted += tx
                .execute_raw(raw::statement(
                    &tx,
                    "DELETE FROM runtime_jobs WHERE id = ?1 AND status IN ('completed', 'failed', 'cancelled') AND updated_at < ?2",
                    vec![job_id.into(), before.into()],
                )?)
                .await?
                .rows_affected();
            for text_asset_id in text_assets {
                tx.execute_raw(raw::statement(
                    &tx,
                    "DELETE FROM text_assets WHERE id = ?1 AND NOT EXISTS (SELECT 1 FROM runtime_jobs WHERE input_text_asset_id = ?1) AND NOT EXISTS (SELECT 1 FROM runtime_artifacts WHERE text_asset_id = ?1)",
                    vec![text_asset_id.into()],
                )?)
                .await?;
            }
        }
        tx.commit()
            .await
            .context("Failed to commit expired runtime job deletion")?;
        Ok(deleted)
    }

    /// Live opaque media whose retention label has lapsed: ephemeral media
    /// older than its retention, and job media of terminal jobs past job
    /// retention. Durable media is never returned. The artifact store
    /// deletes what this returns.
    pub(crate) async fn expired_retained_media(
        &self,
        policy: RetentionPolicy,
        limit: usize,
    ) -> anyhow::Result<Vec<MediaAsset>> {
        let limit = i64::try_from(bounded_maintenance_batch_limit(limit))?;
        let now = self.now_millis();
        let db = self.db.connection().await?;
        let opaque = "m.asset_kind = 'opaque_artifact' AND m.storage_namespace = 'artifact_store_v1' AND m.deleted_at IS NULL";
        let columns = "SELECT DISTINCT m.id, m.created_at, m.updated_at, m.asset_kind, m.storage_namespace, m.storage_key, m.content_type, m.filename, m.size_bytes, m.sha256, m.duration_secs, m.sample_rate_hz, m.channel_count, m.peak_amplitude, m.rms_amplitude, m.source_asset_id, m.canonical_profile_version, m.scan_status, m.retention_policy, m.deleted_at, m.metadata_json";
        let mut rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    "{columns} FROM media_assets AS m WHERE m.retention_policy = 'artifact_ephemeral' AND {opaque} AND m.created_at < ?1 ORDER BY m.created_at ASC, m.id ASC LIMIT ?2"
                ),
                vec![
                    now.saturating_sub(i64::try_from(policy.ephemeral_artifact_ms)?)
                        .into(),
                    limit.into(),
                ],
            )?)
            .await
            .context("Failed to select expired ephemeral media")?;
        let remaining = limit - i64::try_from(rows.len())?;
        if remaining > 0 {
            rows.extend(
                db.query_all_raw(raw::statement(
                    db,
                    format!(
                        r#"
                        {columns}
                        FROM runtime_jobs AS j
                        JOIN runtime_artifacts AS a ON a.job_id = j.id
                        JOIN media_assets AS m ON m.id = a.media_asset_id
                        WHERE j.status IN ('completed', 'failed', 'cancelled')
                          AND j.updated_at < ?1
                          AND m.retention_policy = 'artifact_job' AND {opaque}
                        ORDER BY m.id ASC
                        LIMIT ?2
                        "#
                    ),
                    vec![
                        now.saturating_sub(i64::try_from(policy.terminal_job_ms)?)
                            .into(),
                        remaining.into(),
                    ],
                )?)
                .await
                .context("Failed to select expired job media")?,
            );
        }
        rows.iter().map(map_media_asset).collect()
    }
}

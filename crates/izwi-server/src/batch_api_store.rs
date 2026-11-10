//! Durable state of the Batch API (`/v1/batches`).
//!
//! A batch is one `api_batches` row: its input file, status, counts and the
//! lease of the executor running it. A line that has finished is one
//! `api_batch_requests` row holding its response, so a restarted executor
//! skips it; lines still to run have no row and are read again from the
//! input file. The rows are folded into the output and error files when the
//! batch ends, and deleted with that same step.

use anyhow::Context;
use sea_orm::{ConnectionTrait, QueryResult, TransactionTrait};
use serde::Serialize;

use crate::artifact_store::ArtifactTenant;
use crate::db::{raw, write_transaction_options, StoreDatabase};
use crate::error::StoreError;

pub(crate) const BATCH_ID_PREFIX: &str = "batch_";

/// OpenAI's only completion window.
const COMPLETION_WINDOW_MS: i64 = 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BatchStatus {
    Validating,
    Failed,
    InProgress,
    Finalizing,
    Completed,
    Expired,
    Cancelling,
    Cancelled,
}

impl BatchStatus {
    pub(crate) const ALL: [Self; 8] = [
        Self::Validating,
        Self::Failed,
        Self::InProgress,
        Self::Finalizing,
        Self::Completed,
        Self::Expired,
        Self::Cancelling,
        Self::Cancelled,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Validating => "validating",
            Self::Failed => "failed",
            Self::InProgress => "in_progress",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Expired => "expired",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|status| status.as_str() == value)
    }

    /// The column recording when a batch ended in this status.
    const fn ended_at_column(self) -> Option<&'static str> {
        match self {
            Self::Failed => Some("failed_at"),
            Self::Completed => Some("completed_at"),
            Self::Expired => Some("expired_at"),
            Self::Cancelled => Some("cancelled_at"),
            _ => None,
        }
    }
}

/// The statuses an executor has work to do in.
const ACTIVE_STATUSES: &str = "'validating', 'in_progress', 'finalizing', 'cancelling'";

/// A batch. Times are epoch milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BatchRecord {
    pub(crate) id: String,
    pub(crate) tenant_scope: String,
    pub(crate) endpoint: String,
    pub(crate) input_file_id: String,
    pub(crate) status: BatchStatus,
    pub(crate) metadata: Option<serde_json::Value>,
    pub(crate) errors: Option<serde_json::Value>,
    pub(crate) output_file_id: Option<String>,
    pub(crate) error_file_id: Option<String>,
    pub(crate) request_total: i64,
    pub(crate) request_completed: i64,
    pub(crate) request_failed: i64,
    pub(crate) created_at: i64,
    pub(crate) expires_at: i64,
    pub(crate) in_progress_at: Option<i64>,
    pub(crate) finalizing_at: Option<i64>,
    pub(crate) completed_at: Option<i64>,
    pub(crate) failed_at: Option<i64>,
    pub(crate) expired_at: Option<i64>,
    pub(crate) cancelling_at: Option<i64>,
    pub(crate) cancelled_at: Option<i64>,
}

/// A finished line: what its output line says.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BatchLineResult {
    pub(crate) line_index: i64,
    pub(crate) custom_id: String,
    pub(crate) request_id: String,
    pub(crate) status_code: u16,
    pub(crate) body: serde_json::Value,
}

impl BatchLineResult {
    pub(crate) fn failed(&self) -> bool {
        self.status_code >= 400
    }
}

/// Which of a batch's two files a line belongs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOutputKind {
    Output,
    Errors,
}

impl BatchOutputKind {
    const fn column(self) -> &'static str {
        match self {
            Self::Output => "output_file_id",
            Self::Errors => "error_file_id",
        }
    }
}

/// Batches in one status, and the lines they have finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchStatusCount {
    pub status: &'static str,
    pub batches: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
}

#[derive(Debug, Clone)]
pub struct BatchApiStore {
    db: StoreDatabase,
}

const BATCH_COLUMNS: &str = "id, tenant_scope, endpoint, input_file_id, status, metadata_json, \
    errors_json, output_file_id, error_file_id, request_total, request_completed, \
    request_failed, created_at, expires_at, in_progress_at, finalizing_at, completed_at, \
    failed_at, expired_at, cancelling_at, cancelled_at";

fn map_batch(row: &QueryResult) -> anyhow::Result<BatchRecord> {
    let status: String = row.try_get_by_index(4)?;
    let json = |index: usize| -> anyhow::Result<Option<serde_json::Value>> {
        let raw: Option<String> = row.try_get_by_index(index)?;
        raw.map(|raw| serde_json::from_str(&raw).context("Stored batch JSON is invalid"))
            .transpose()
    };
    Ok(BatchRecord {
        id: row.try_get_by_index(0)?,
        tenant_scope: row.try_get_by_index(1)?,
        endpoint: row.try_get_by_index(2)?,
        input_file_id: row.try_get_by_index(3)?,
        status: BatchStatus::parse(&status)
            .with_context(|| format!("Unknown stored batch status `{status}`"))?,
        metadata: json(5)?,
        errors: json(6)?,
        output_file_id: row.try_get_by_index(7)?,
        error_file_id: row.try_get_by_index(8)?,
        request_total: row.try_get_by_index(9)?,
        request_completed: row.try_get_by_index(10)?,
        request_failed: row.try_get_by_index(11)?,
        created_at: row.try_get_by_index(12)?,
        expires_at: row.try_get_by_index(13)?,
        in_progress_at: row.try_get_by_index(14)?,
        finalizing_at: row.try_get_by_index(15)?,
        completed_at: row.try_get_by_index(16)?,
        failed_at: row.try_get_by_index(17)?,
        expired_at: row.try_get_by_index(18)?,
        cancelling_at: row.try_get_by_index(19)?,
        cancelled_at: row.try_get_by_index(20)?,
    })
}

impl BatchApiStore {
    pub(crate) fn new(db: StoreDatabase) -> Self {
        Self { db }
    }

    pub(crate) fn now_millis(&self) -> i64 {
        self.db.now_millis()
    }

    /// A new batch of `tenant`, `validating`, due 24 hours from now.
    pub(crate) async fn create(
        &self,
        tenant: &ArtifactTenant,
        endpoint: &str,
        input_file_id: &str,
        metadata: Option<&serde_json::Value>,
    ) -> anyhow::Result<BatchRecord> {
        let id = crate::ids::new_prefixed_id(BATCH_ID_PREFIX);
        let now = self.now_millis();
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "INSERT INTO api_batches (id, tenant_scope, endpoint, input_file_id, status, \
             metadata_json, created_at, expires_at) \
             VALUES (?1, ?2, ?3, ?4, 'validating', ?5, ?6, ?7)",
            vec![
                id.clone().into(),
                tenant.as_str().into(),
                endpoint.into(),
                input_file_id.into(),
                metadata.map(serde_json::Value::to_string).into(),
                now.into(),
                now.saturating_add(COMPLETION_WINDOW_MS).into(),
            ],
        )?)
        .await
        .context("Failed to create the batch")?;
        self.load(&id)
            .await?
            .context("Failed to fetch the created batch")
    }

    /// Batch `id`, whoever owns it: the executor's view.
    pub(crate) async fn load(&self, id: &str) -> anyhow::Result<Option<BatchRecord>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                format!("SELECT {BATCH_COLUMNS} FROM api_batches WHERE id = ?1"),
                vec![id.into()],
            )?)
            .await
            .context("Failed to load the batch")?;
        row.as_ref().map(map_batch).transpose()
    }

    /// `tenant`'s batch `id`.
    pub(crate) async fn get(
        &self,
        tenant: &ArtifactTenant,
        id: &str,
    ) -> anyhow::Result<Option<BatchRecord>> {
        Ok(self
            .load(id)
            .await?
            .filter(|batch| batch.tenant_scope == tenant.as_str()))
    }

    /// Up to `limit` of `tenant`'s batches, newest first, after the batch
    /// `after`. Returns whether more follow.
    pub(crate) async fn list(
        &self,
        tenant: &ArtifactTenant,
        after: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<(Vec<BatchRecord>, bool)> {
        let cursor = match after {
            Some(after) => Some(
                self.get(tenant, after)
                    .await?
                    .ok_or_else(|| StoreError::invalid(format!("Unknown cursor `{after}`")))?,
            ),
            None => None,
        };
        let limit = limit.clamp(1, 100);
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    "SELECT {BATCH_COLUMNS} FROM api_batches WHERE tenant_scope = ?1 \
                     AND (?2 IS NULL OR created_at < ?2 OR (created_at = ?2 AND id < ?3)) \
                     ORDER BY created_at DESC, id DESC LIMIT ?4"
                ),
                vec![
                    tenant.as_str().into(),
                    cursor.as_ref().map(|batch| batch.created_at).into(),
                    cursor.map(|batch| batch.id).into(),
                    i64::try_from(limit + 1)?.into(),
                ],
            )?)
            .await
            .context("Failed to list batches")?;
        let mut batches = rows
            .iter()
            .map(map_batch)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let has_more = batches.len() > limit;
        batches.truncate(limit);
        Ok((batches, has_more))
    }

    /// Ask for `tenant`'s batch `id` to be cancelled: a batch that is
    /// validating or in progress becomes `cancelling`, and its executor ends
    /// it. Returns the batch as it now is.
    pub(crate) async fn request_cancel(
        &self,
        tenant: &ArtifactTenant,
        id: &str,
    ) -> anyhow::Result<Option<BatchRecord>> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "UPDATE api_batches SET status = 'cancelling', cancelling_at = ?1 \
             WHERE id = ?2 AND tenant_scope = ?3 AND status IN ('validating', 'in_progress')",
            vec![self.now_millis().into(), id.into(), tenant.as_str().into()],
        )?)
        .await
        .context("Failed to cancel the batch")?;
        self.get(tenant, id).await
    }

    /// Lease the oldest batch with work to do that no live executor holds.
    pub(crate) async fn claim_next(
        &self,
        owner: &str,
        lease_ms: i64,
    ) -> anyhow::Result<Option<String>> {
        let now = self.now_millis();
        let db = self.db.connection().await?;
        let candidates = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    "SELECT id FROM api_batches WHERE status IN ({ACTIVE_STATUSES}) \
                     AND (lease_owner IS NULL OR lease_expires_at <= ?1) \
                     ORDER BY created_at, id LIMIT 8"
                ),
                vec![now.into()],
            )?)
            .await
            .context("Failed to find a batch to run")?;
        for row in candidates {
            let id: String = row.try_get_by_index(0)?;
            // Another executor may have taken it since the read.
            let claimed = db
                .execute_raw(raw::statement(
                    db,
                    format!(
                        "UPDATE api_batches SET lease_owner = ?1, lease_expires_at = ?2 \
                         WHERE id = ?3 AND status IN ({ACTIVE_STATUSES}) \
                         AND (lease_owner IS NULL OR lease_expires_at <= ?4)"
                    ),
                    vec![
                        owner.into(),
                        now.saturating_add(lease_ms).into(),
                        id.clone().into(),
                        now.into(),
                    ],
                )?)
                .await
                .context("Failed to lease the batch")?;
            if claimed.rows_affected() == 1 {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Extend `owner`'s lease; `false` when another executor has taken the
    /// batch over or it has ended.
    pub(crate) async fn renew_lease(
        &self,
        id: &str,
        owner: &str,
        lease_ms: i64,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let renewed = db
            .execute_raw(raw::statement(
                db,
                "UPDATE api_batches SET lease_expires_at = ?1 WHERE id = ?2 AND lease_owner = ?3",
                vec![
                    self.now_millis().saturating_add(lease_ms).into(),
                    id.into(),
                    owner.into(),
                ],
            )?)
            .await
            .context("Failed to renew the batch lease")?;
        Ok(renewed.rows_affected() == 1)
    }

    /// Give the batch up, so another executor (or this one after a restart)
    /// takes it at once instead of waiting out the lease.
    pub(crate) async fn release_lease(&self, id: &str, owner: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "UPDATE api_batches SET lease_owner = NULL, lease_expires_at = NULL \
             WHERE id = ?1 AND lease_owner = ?2",
            vec![id.into(), owner.into()],
        )?)
        .await
        .context("Failed to release the batch lease")?;
        Ok(())
    }

    /// `validating` to `in_progress` with `total` lines to run. `false` when
    /// the batch is no longer validating (cancelled) or no longer `owner`'s.
    pub(crate) async fn start(&self, id: &str, owner: &str, total: i64) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let started = db
            .execute_raw(raw::statement(
                db,
                "UPDATE api_batches SET status = 'in_progress', in_progress_at = ?1, \
                 request_total = ?2 WHERE id = ?3 AND lease_owner = ?4 AND status = 'validating'",
                vec![
                    self.now_millis().into(),
                    total.into(),
                    id.into(),
                    owner.into(),
                ],
            )?)
            .await
            .context("Failed to start the batch")?;
        Ok(started.rows_affected() == 1)
    }

    /// `in_progress` to `finalizing`, once every line has finished.
    pub(crate) async fn begin_finalizing(&self, id: &str, owner: &str) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let updated = db
            .execute_raw(raw::statement(
                db,
                "UPDATE api_batches SET status = 'finalizing', finalizing_at = ?1 \
                 WHERE id = ?2 AND lease_owner = ?3 AND status = 'in_progress'",
                vec![self.now_millis().into(), id.into(), owner.into()],
            )?)
            .await
            .context("Failed to finalize the batch")?;
        Ok(updated.rows_affected() == 1)
    }

    /// Record the output or error file written for the batch.
    pub(crate) async fn set_output_file(
        &self,
        id: &str,
        kind: BatchOutputKind,
        file_id: &str,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            format!(
                "UPDATE api_batches SET {} = ?1 WHERE id = ?2",
                kind.column()
            ),
            vec![file_id.into(), id.into()],
        )?)
        .await
        .context("Failed to record the batch's file")?;
        Ok(())
    }

    /// End `owner`'s batch in the terminal `status`, from `from`: the lease
    /// is dropped and the finished-line rows, now in the output files, go
    /// with it. `false` when the batch left `from` first (a cancel arrived)
    /// or is no longer `owner`'s.
    pub(crate) async fn finish(
        &self,
        id: &str,
        owner: &str,
        from: BatchStatus,
        status: BatchStatus,
        errors: Option<&serde_json::Value>,
    ) -> anyhow::Result<bool> {
        let ended_at = status
            .ended_at_column()
            .context("A batch ends in a terminal status")?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to begin ending the batch")?;
        let ended = tx
            .execute_raw(raw::statement(
                &tx,
                format!(
                    "UPDATE api_batches SET status = ?1, {ended_at} = ?2, \
                     errors_json = COALESCE(?3, errors_json), lease_owner = NULL, \
                     lease_expires_at = NULL WHERE id = ?4 AND lease_owner = ?5 AND status = ?6"
                ),
                vec![
                    status.as_str().into(),
                    self.now_millis().into(),
                    errors.map(serde_json::Value::to_string).into(),
                    id.into(),
                    owner.into(),
                    from.as_str().into(),
                ],
            )?)
            .await
            .context("Failed to end the batch")?
            .rows_affected()
            == 1;
        if ended {
            tx.execute_raw(raw::statement(
                &tx,
                "DELETE FROM api_batch_requests WHERE batch_id = ?1",
                vec![id.into()],
            )?)
            .await
            .context("Failed to clear the batch's finished lines")?;
        }
        tx.commit().await.context("Failed to end the batch")?;
        Ok(ended)
    }

    /// Record a finished line and count it. A line recorded twice (two
    /// executors overlapped around a lost lease) keeps its first result and
    /// is counted once.
    pub(crate) async fn record_result(
        &self,
        batch_id: &str,
        result: &BatchLineResult,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to begin recording a batch line")?;
        let inserted = tx
            .execute_raw(raw::statement(
                &tx,
                "INSERT INTO api_batch_requests (batch_id, line_index, custom_id, request_id, \
                 status_code, body_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
                 ON CONFLICT (batch_id, line_index) DO NOTHING",
                vec![
                    batch_id.into(),
                    result.line_index.into(),
                    result.custom_id.clone().into(),
                    result.request_id.clone().into(),
                    i64::from(result.status_code).into(),
                    result.body.to_string().into(),
                ],
            )?)
            .await
            .context("Failed to record a batch line")?
            .rows_affected()
            == 1;
        if inserted {
            let counter = if result.failed() {
                "request_failed"
            } else {
                "request_completed"
            };
            tx.execute_raw(raw::statement(
                &tx,
                format!("UPDATE api_batches SET {counter} = {counter} + 1 WHERE id = ?1"),
                vec![batch_id.into()],
            )?)
            .await
            .context("Failed to count a batch line")?;
        }
        tx.commit().await.context("Failed to record a batch line")?;
        Ok(())
    }

    /// The indexes of the batch's finished lines.
    pub(crate) async fn finished_lines(
        &self,
        batch_id: &str,
    ) -> anyhow::Result<std::collections::HashSet<i64>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT line_index FROM api_batch_requests WHERE batch_id = ?1",
                vec![batch_id.into()],
            )?)
            .await
            .context("Failed to load the batch's finished lines")?;
        rows.iter()
            .map(|row| Ok(row.try_get_by_index::<i64>(0)?))
            .collect()
    }

    /// Up to `limit` finished lines of `kind` after line `after`, in input
    /// order.
    pub(crate) async fn results_page(
        &self,
        batch_id: &str,
        kind: BatchOutputKind,
        after: i64,
        limit: usize,
    ) -> anyhow::Result<Vec<BatchLineResult>> {
        let outcome = match kind {
            BatchOutputKind::Output => "status_code < 400",
            BatchOutputKind::Errors => "status_code >= 400",
        };
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    "SELECT line_index, custom_id, request_id, status_code, body_json \
                     FROM api_batch_requests WHERE batch_id = ?1 AND line_index > ?2 \
                     AND {outcome} ORDER BY line_index LIMIT ?3"
                ),
                vec![batch_id.into(), after.into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to load the batch's finished lines")?;
        rows.iter()
            .map(|row| {
                let status_code: i64 = row.try_get_by_index(3)?;
                let body: String = row.try_get_by_index(4)?;
                Ok(BatchLineResult {
                    line_index: row.try_get_by_index(0)?,
                    custom_id: row.try_get_by_index(1)?,
                    request_id: row.try_get_by_index(2)?,
                    status_code: u16::try_from(status_code).unwrap_or(500),
                    body: serde_json::from_str(&body)
                        .context("Stored batch line JSON is invalid")?,
                })
            })
            .collect()
    }

    /// Batches by status with their finished lines, for metrics.
    pub(crate) async fn status_counts(&self) -> anyhow::Result<Vec<BatchStatusCount>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement_without_values(
                db,
                "SELECT status, COUNT(*), COALESCE(SUM(request_completed), 0), \
                 COALESCE(SUM(request_failed), 0) FROM api_batches GROUP BY status",
            ))
            .await
            .context("Failed to count batches")?;
        let mut counts = Vec::with_capacity(rows.len());
        for row in rows {
            let status: String = row.try_get_by_index(0)?;
            let Some(status) = BatchStatus::parse(&status) else {
                continue;
            };
            let count = |index: usize| -> anyhow::Result<u64> {
                Ok(u64::try_from(row.try_get_by_index::<i64>(index)?).unwrap_or(0))
            };
            counts.push(BatchStatusCount {
                status: status.as_str(),
                batches: count(1)?,
                requests_completed: count(2)?,
                requests_failed: count(3)?,
            });
        }
        counts.sort_by_key(|count| count.status);
        Ok(counts)
    }

    /// Make batch `id` due at `expires_at`.
    #[cfg(test)]
    pub(crate) async fn set_expires_at_for_test(
        &self,
        id: &str,
        expires_at: i64,
    ) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "UPDATE api_batches SET expires_at = ?1 WHERE id = ?2",
            vec![expires_at.into(), id.into()],
        )?)
        .await?;
        Ok(())
    }
}

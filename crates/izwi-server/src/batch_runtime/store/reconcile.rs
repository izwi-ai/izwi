//! Bounded repair of states a crash left inconsistent.

use super::*;

/// Non-atomic enqueue paths create the job, then its artifacts and stages, in
/// separate transactions within one request. A job still stageless after this
/// long was orphaned by a crash or error in between and can never be claimed.
pub(super) const STAGELESS_JOB_GRACE_MS: i64 = 5 * 60 * 1000;

/// A cancelled job's route projection is left pending only by a crash between
/// the job's cancellation and the projection update, so reconciliation looks
/// back this far (through the status/update index) instead of scanning every
/// cancelled job ever, every 30 seconds.
pub(super) const CANCELLED_PROJECTION_REPAIR_WINDOW_MS: i64 = 7 * 24 * 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeReconciliationReport {
    pub jobs_repaired: u64,
    pub stages_repaired: u64,
    pub route_projections_repaired: u64,
}

impl BatchRuntimeStore {
    pub async fn reconcile_inconsistent_states(
        &self,
        limit: usize,
    ) -> anyhow::Result<RuntimeReconciliationReport> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime reconciliation transaction")?;
        let now = self.now_millis();
        let mut report = RuntimeReconciliationReport::default();
        let mut remaining = bounded_maintenance_batch_limit(limit);

        let failed = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'failed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    error_code = COALESCE(error_code, 'stage_failed'),
                    error_message = COALESCE(error_message, 'Runtime stage became failed')
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('queued', 'running', 'retrying')
                      AND candidate.cancellation_state IS NULL
                      AND EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id
                            AND job_stages.status = 'failed'
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile failed runtime jobs")?;
        report.jobs_repaired = report.jobs_repaired.saturating_add(failed.rows_affected());
        remaining = remaining.saturating_sub(failed.rows_affected() as usize);

        let stageless = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'failed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    error_code = 'enqueue_incomplete',
                    error_message = 'Runtime job was accepted but its stages were never created'
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status = 'queued'
                      AND candidate.created_at <= ?3
                      AND NOT EXISTS (
                          SELECT 1 FROM job_stages WHERE job_stages.job_id = candidate.id
                      )
                    ORDER BY candidate.created_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![
                    now.into(),
                    i64::try_from(remaining)?.into(),
                    now.saturating_sub(STAGELESS_JOB_GRACE_MS).into(),
                ],
            )?)
            .await
            .context("Failed to reconcile stageless runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(stageless);
        remaining = remaining.saturating_sub(stageless as usize);

        let cancelled = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'cancelled',
                    cancellation_state = NULL,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    cancellation_reason = COALESCE(cancellation_reason, 'All remaining stages were cancelled')
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('queued', 'running', 'retrying')
                      AND EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id AND status = 'cancelled'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id
                            AND status NOT IN ('completed', 'cancelled')
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile cancelled runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(cancelled);
        remaining = remaining.saturating_sub(cancelled as usize);

        let completed = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'completed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    error_code = NULL,
                    error_message = NULL
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('queued', 'running', 'retrying')
                      AND candidate.cancellation_state IS NULL
                      AND EXISTS (SELECT 1 FROM job_stages WHERE job_stages.job_id = candidate.id)
                      AND NOT EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id
                            AND status <> 'completed'
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile completed runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(completed);
        remaining = remaining.saturating_sub(completed as usize);

        let retrying = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET status = 'retrying', updated_at = ?1
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('queued', 'running')
                      AND candidate.cancellation_state IS NULL
                      AND EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id AND status = 'retrying'
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile retrying runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(retrying);
        remaining = remaining.saturating_sub(retrying as usize);

        let stages = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = 'cancelled',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    available_at = NULL,
                    error_code = COALESCE(error_code, 'parent_terminal'),
                    error_message = COALESCE(error_message, 'Parent runtime job is terminal')
                WHERE id IN (
                    SELECT candidate.id
                    FROM job_stages AS candidate
                    WHERE candidate.status IN ('queued', 'running', 'retrying')
                      AND EXISTS (
                          SELECT 1 FROM runtime_jobs
                          WHERE runtime_jobs.id = candidate.job_id
                            AND runtime_jobs.status IN ('failed', 'cancelled')
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile stages owned by terminal runtime jobs")?
            .rows_affected()
        };
        report.stages_repaired = stages;
        remaining = remaining.saturating_sub(stages as usize);

        if remaining > 0 {
            let cancelled_route_jobs = tx
                .query_all_raw(raw::statement(
                    &tx,
                    r#"
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status = 'cancelled'
                      AND candidate.updated_at >= ?2
                      AND NOT EXISTS (
                          SELECT 1 FROM runtime_jobs AS active
                          WHERE active.id <> candidate.id
                            AND active.job_kind = candidate.job_kind
                            AND active.route_record_kind = candidate.route_record_kind
                            AND active.route_record_id = candidate.route_record_id
                            AND active.status IN ('queued', 'running', 'retrying')
                      )
                      AND (
                          (
                              candidate.job_kind = 'asr_transcription'
                              AND candidate.route_record_kind IN ('transcription', 'speaker_attributed_asr')
                              AND EXISTS (
                                  SELECT 1 FROM transcription_records
                                  WHERE transcription_records.id = candidate.route_record_id
                                    AND processing_status IN ('pending', 'processing')
                              )
                          )
                          OR (
                              candidate.job_kind = 'tts_speech'
                              AND candidate.route_record_kind IN ('text_to_speech', 'voice_design', 'voice_cloning')
                              AND EXISTS (
                                  SELECT 1 FROM speech_history_records
                                  WHERE speech_history_records.id = candidate.route_record_id
                                    AND speech_history_records.route_kind = candidate.route_record_kind
                                    AND processing_status IN ('pending', 'processing')
                              )
                          )
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?1
                    "#,
                    vec![
                        i64::try_from(remaining)?.into(),
                        now.saturating_sub(CANCELLED_PROJECTION_REPAIR_WINDOW_MS)
                            .into(),
                    ],
                )?)
                .await
                .context("Failed to select cancelled route projections for reconciliation")?;
            for row in cancelled_route_jobs {
                let job_id: String = row.try_get_by_index(0)?;
                report.route_projections_repaired =
                    report.route_projections_repaired.saturating_add(
                        self.finalize_cancelled_route_projection_with(&tx, &job_id)
                            .await?,
                    );
            }
        }

        tx.commit()
            .await
            .context("Failed to commit runtime reconciliation transaction")?;

        Ok(report)
    }
}

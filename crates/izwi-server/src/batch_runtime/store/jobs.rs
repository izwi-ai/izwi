//! Runtime jobs: creation, lookup, manual retry and cancellation.

use super::*;

#[derive(Debug, Clone)]
pub struct NewRuntimeJob {
    pub job_kind: RuntimeJobKind,
    pub status: RuntimeJobStatus,
    pub priority: i32,
    pub model_id: Option<String>,
    pub capability: Option<String>,
    pub route_record_kind: Option<String>,
    pub route_record_id: Option<String>,
    pub input_media_asset_id: Option<String>,
    pub input_text_asset_id: Option<String>,
    pub request_json: serde_json::Value,
    pub model_snapshot_json: serde_json::Value,
    pub retry_policy_json: serde_json::Value,
    pub max_attempts: u32,
    pub idempotency_key: Option<String>,
    pub correlation_id: Option<String>,
}

pub(super) async fn get_job_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            RUNTIME_JOB_BY_ID_SQL,
            vec![id.into()],
        )?)
        .await
        .context("Failed to load runtime job")?;
    row.as_ref().map(map_runtime_job).transpose()
}

impl BatchRuntimeStore {
    pub async fn create_job(&self, input: NewRuntimeJob) -> anyhow::Result<RuntimeJob> {
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let admission_tenant = if input.job_kind == RuntimeJobKind::TtsSpeech {
            let tenant = speech_admission_tenant(&input.request_json)?;
            if !is_terminal_job_status(input.status) {
                self.check_tts_admission(&tx, &tenant).await?;
            }
            Some(tenant)
        } else {
            None
        };
        let now = self.now_millis();
        let id = new_uuid();
        let request_json = json_to_db_string(&input.request_json, "{}")?;
        let model_snapshot_json = json_to_db_string(&input.model_snapshot_json, "{}")?;
        let retry_policy_json = json_to_db_string(&input.retry_policy_json, "{}")?;
        let queued_at = matches!(input.status, RuntimeJobStatus::Queued).then_some(now);
        let started_at = matches!(input.status, RuntimeJobStatus::Running).then_some(now);
        let finished_at = is_terminal_job_status(input.status).then_some(now);

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_jobs (
                id,
                created_at,
                updated_at,
                queued_at,
                started_at,
                finished_at,
                job_kind,
                status,
                priority,
                model_id,
                capability,
                route_record_kind,
                route_record_id,
                input_media_asset_id,
                input_text_asset_id,
                request_json,
                model_snapshot_json,
                progress_json,
                error_code,
                error_message,
                attempt_count,
                max_attempts,
                retry_policy_json,
                idempotency_key,
                correlation_id,
                cancellation_reason,
                admission_tenant
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, NULL, NULL, NULL, 0, ?17, ?18, ?19, ?20, NULL, ?21)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                opt_i64(queued_at),
                opt_i64(started_at),
                opt_i64(finished_at),
                input.job_kind.as_db_value().into(),
                input.status.as_db_value().into(),
                input.priority.into(),
                opt_string(input.model_id),
                opt_string(input.capability),
                opt_string(input.route_record_kind),
                opt_string(input.route_record_id),
                opt_string(input.input_media_asset_id),
                opt_string(input.input_text_asset_id),
                request_json.into(),
                model_snapshot_json.into(),
                u32_to_i64_value(input.max_attempts).into(),
                retry_policy_json.into(),
                opt_string(input.idempotency_key),
                opt_string(input.correlation_id),
                opt_string(admission_tenant),
            ],
        )?)
        .await
        .context("Failed to create runtime job")?;

        tx.commit().await?;
        self.get_job(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job was not found"))
    }

    pub async fn get_job(&self, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_JOB_BY_ID_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime job")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn list_active_jobs_by_kind(
        &self,
        job_kind: RuntimeJobKind,
    ) -> anyhow::Result<Vec<RuntimeJob>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                concat!(
                    select_runtime_jobs!(),
                    r#"
                    WHERE job_kind = ?1
                      AND status IN ('queued', 'running', 'retrying')
                    ORDER BY created_at ASC, id ASC
                    "#
                ),
                vec![job_kind.as_db_value().into()],
            )?)
            .await
            .context("Failed to list active runtime jobs by kind")?;

        rows.iter().map(map_runtime_job).collect()
    }

    pub async fn get_active_job_for_route_record(
        &self,
        job_kind: RuntimeJobKind,
        route_record_kind: &str,
        route_record_id: &str,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                concat!(
                    select_runtime_jobs!(),
                    r#"
                    WHERE job_kind = ?1
                      AND route_record_kind = ?2
                      AND route_record_id = ?3
                      AND status IN ('queued', 'running', 'retrying')
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    "#
                ),
                vec![
                    job_kind.as_db_value().into(),
                    route_record_kind.into(),
                    route_record_id.into(),
                ],
            )?)
            .await
            .context("Failed to load active runtime job for route record")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn get_latest_job_for_route_record(
        &self,
        job_kind: RuntimeJobKind,
        route_record_kind: &str,
        route_record_id: &str,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                concat!(
                    select_runtime_jobs!(),
                    r#"
                    WHERE job_kind = ?1
                      AND route_record_kind = ?2
                      AND route_record_id = ?3
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    "#
                ),
                vec![
                    job_kind.as_db_value().into(),
                    route_record_kind.into(),
                    route_record_id.into(),
                ],
            )?)
            .await
            .context("Failed to load active runtime job for route record")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn job_status_counts(&self) -> anyhow::Result<Vec<RuntimeJobStatusCount>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT status, COUNT(*) FROM runtime_jobs GROUP BY status ORDER BY status",
                vec![],
            )?)
            .await
            .context("Failed to count runtime jobs by status")?;

        rows.iter()
            .map(|row| {
                let status_raw: String = row.try_get_by_index(0)?;
                let status = RuntimeJobStatus::from_db_value(status_raw.as_str())
                    .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                Ok(RuntimeJobStatusCount { status, count })
            })
            .collect()
    }

    pub async fn retry_job(&self, job_id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime job retry transaction")?;
        let Some(job) = get_job_with(&tx, job_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !matches!(
            job.status,
            RuntimeJobStatus::Failed | RuntimeJobStatus::Cancelled
        ) {
            tx.rollback().await?;
            return Ok(None);
        }
        if job.error_code.as_deref() == Some("speech_replay_expired")
            || job.attempt_count >= job.max_attempts
        {
            tx.rollback().await?;
            return Ok(None);
        }

        let retryable_stage_counts = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN attempt_count < max_attempts THEN 1 ELSE 0 END)
                FROM job_stages
                WHERE job_id = ?1
                  AND status IN ('failed', 'cancelled')
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to count retryable runtime job stages")?
            .ok_or_else(|| anyhow!("Runtime retryable stage count returned no row"))?;
        let retryable_stage_count = retryable_stage_counts.try_get_by_index::<i64>(0)?;
        let eligible_stage_count = retryable_stage_counts
            .try_get_by_index::<Option<i64>>(1)?
            .unwrap_or(0);
        if retryable_stage_count == 0 || eligible_stage_count != retryable_stage_count {
            tx.rollback().await?;
            return Ok(None);
        }

        let admission_tenant = if job.job_kind == RuntimeJobKind::TtsSpeech {
            let tenant = speech_admission_tenant(&job.request_json)?;
            self.check_tts_admission(&tx, &tenant).await?;
            Some(tenant)
        } else {
            None
        };
        let now = self.now_millis();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'queued',
                    updated_at = ?1,
                    queued_at = ?1,
                    started_at = NULL,
                    finished_at = NULL,
                    error_code = NULL,
                    error_message = NULL,
                    attempt_count = attempt_count + 1,
                    cancellation_reason = NULL,
                    cancellation_state = NULL,
                    admission_tenant = ?3,
                    replay_swept_at = NULL
                WHERE id = ?2
                  AND status IN ('failed', 'cancelled')
                  AND (error_code IS NULL OR error_code <> 'speech_replay_expired')
                  AND attempt_count < max_attempts
                "#,
                vec![now.into(), job_id.into(), opt_string(admission_tenant)],
            )?)
            .await
            .context("Failed to retry runtime job")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let stages = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
            UPDATE job_stages
            SET
                status = 'retrying',
                updated_at = ?1,
                finished_at = NULL,
                lease_expires_at = NULL,
                worker_id = NULL,
                available_at = ?1,
                attempt_token = NULL,
                cancellation_state = NULL,
                output_artifact_ids_json = '[]',
                error_code = NULL,
                error_message = NULL
            WHERE job_id = ?2
              AND status IN ('failed', 'cancelled')
              AND attempt_count < max_attempts
            "#,
                vec![now.into(), job_id.into()],
            )?)
            .await
            .context("Failed to retry runtime job stages")?;
        if stages.rows_affected() != u64::try_from(retryable_stage_count)? {
            bail!("Runtime job retry changed an unexpected number of stages");
        }

        tx.commit()
            .await
            .context("Failed to commit runtime job retry transaction")?;
        self.wake_claimers();
        self.get_job(job_id).await
    }

    pub async fn cancel_job(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime job cancellation transaction")?;
        let now = self.now_millis();

        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status = 'running'
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN status
                        ELSE 'cancelled'
                    END,
                    updated_at = ?1,
                    finished_at = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status = 'running'
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN NULL
                        ELSE COALESCE(finished_at, ?1)
                    END,
                    cancellation_reason = ?2,
                    cancellation_state = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status = 'running'
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN COALESCE(cancellation_state, 'requested')
                        ELSE NULL
                    END
                WHERE id = ?3
                  AND status IN ('queued', 'running', 'retrying')
                "#,
                vec![now.into(), opt_string(reason), job_id.into()],
            )?)
            .await
            .context("Failed to cancel runtime job")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET
                status = 'cancelled',
                cancellation_state = NULL,
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                lease_expires_at = NULL,
                worker_id = NULL,
                available_at = NULL
            WHERE job_id = ?2
              AND status IN ('queued', 'running', 'retrying')
              AND (
                  status <> 'running'
                  OR worker_id IS NULL
                  OR lease_expires_at IS NULL
              )
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to cancel non-running runtime job stages")?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET
                cancellation_state = CASE
                    WHEN cancellation_state = 'execution_stopping'
                        THEN cancellation_state
                    ELSE 'requested'
                END,
                updated_at = ?1,
                available_at = NULL
            WHERE job_id = ?2
              AND status = 'running'
              AND worker_id IS NOT NULL
              AND lease_expires_at IS NOT NULL
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to request cancellation of running runtime job stages")?;

        self.finalize_cancelled_route_projection_with(&tx, job_id)
            .await?;

        tx.commit()
            .await
            .context("Failed to commit runtime job cancellation transaction")?;
        self.get_job(job_id).await
    }
}

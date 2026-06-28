use super::types::{
    ClaimedStage, IdempotencyRecord, JobStage, MediaAsset, RuntimeArtifact, RuntimeArtifactKind,
    RuntimeArtifactRole, RuntimeJob, RuntimeJobKind, RuntimeJobStatus, RuntimeStageStatus,
    RuntimeWorkerHeartbeat, TextAsset,
};
use crate::{
    db::{raw, StoreDatabase},
    ids::new_uuid,
};
use anyhow::{anyhow, bail, Context};
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, QueryResult, Value};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct BatchRuntimeStore {
    db: StoreDatabase,
}

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
    pub scan_status: String,
    pub retention_policy: String,
    pub metadata_json: serde_json::Value,
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

#[derive(Debug, Clone)]
pub struct NewJobStage {
    pub job_id: String,
    pub sequence: u32,
    pub stage_kind: String,
    pub status: RuntimeStageStatus,
    pub capability: Option<String>,
    pub model_id: Option<String>,
    pub max_attempts: u32,
    pub input_artifact_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NewRuntimeArtifact {
    pub job_id: String,
    pub stage_id: Option<String>,
    pub artifact_kind: RuntimeArtifactKind,
    pub artifact_role: RuntimeArtifactRole,
    pub media_asset_id: Option<String>,
    pub text_asset_id: Option<String>,
    pub storage_key: Option<String>,
    pub content_type: Option<String>,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub metadata_json: serde_json::Value,
    pub retention_policy: String,
}

#[derive(Debug, Clone)]
pub struct NewIdempotencyRecord {
    pub operation: String,
    pub idempotency_key: String,
    pub expires_at: Option<u64>,
    pub request_hash: String,
    pub response_json: Option<serde_json::Value>,
    pub runtime_job_id: Option<String>,
    pub conflict_message: Option<String>,
    pub metadata_json: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct WorkerHeartbeatUpdate {
    pub worker_id: String,
    pub status: String,
    pub queue_names: Vec<String>,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub diagnostic_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeJobStatusCount {
    pub status: RuntimeJobStatus,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeStageStatusCount {
    pub status: RuntimeStageStatus,
    pub count: u64,
}

impl BatchRuntimeStore {
    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self { db }
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        self.db.connection().await
    }

    pub async fn create_media_asset(&self, input: NewMediaAsset) -> anyhow::Result<MediaAsset> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
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
                scan_status,
                retention_policy,
                deleted_at,
                metadata_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, NULL, ?17)
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
                MEDIA_ASSET_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load media asset")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    pub async fn create_text_asset(&self, input: NewTextAsset) -> anyhow::Result<TextAsset> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
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
            .query_one_raw(raw::statement(db, TEXT_ASSET_COLUMNS_SQL, vec![id.into()])?)
            .await
            .context("Failed to load text asset")?;

        row.as_ref().map(map_text_asset).transpose()
    }

    pub async fn create_job(&self, input: NewRuntimeJob) -> anyhow::Result<RuntimeJob> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let id = new_uuid();
        let request_json = json_to_db_string(&input.request_json, "{}")?;
        let model_snapshot_json = json_to_db_string(&input.model_snapshot_json, "{}")?;
        let retry_policy_json = json_to_db_string(&input.retry_policy_json, "{}")?;
        let queued_at = matches!(input.status, RuntimeJobStatus::Queued).then_some(now);
        let started_at = matches!(input.status, RuntimeJobStatus::Running).then_some(now);
        let finished_at = is_terminal_job_status(input.status).then_some(now);

        db.execute_raw(raw::statement(
            db,
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
                cancellation_reason
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, NULL, NULL, NULL, 0, ?17, ?18, ?19, ?20, NULL)
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
            ],
        )?)
        .await
        .context("Failed to create runtime job")?;

        self.get_job(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job was not found"))
    }

    pub async fn get_job(&self, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_JOB_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime job")?;

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

    pub async fn transition_job_status(
        &self,
        job_id: &str,
        expected_statuses: &[RuntimeJobStatus],
        next_status: RuntimeJobStatus,
        error_code: Option<String>,
        error_message: Option<String>,
        cancellation_reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        if expected_statuses.is_empty() {
            bail!("At least one expected status is required for runtime job transitions");
        }

        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let expected_placeholders = (0..expected_statuses.len())
            .map(|index| format!("?{}", index + 7))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            r#"
            UPDATE runtime_jobs
            SET
                status = ?1,
                updated_at = ?2,
                queued_at = CASE WHEN ?1 = 'queued' THEN COALESCE(queued_at, ?2) ELSE queued_at END,
                started_at = CASE WHEN ?1 = 'running' THEN COALESCE(started_at, ?2) ELSE started_at END,
                finished_at = CASE WHEN ?1 IN ('completed', 'failed', 'cancelled', 'expired') THEN COALESCE(finished_at, ?2) ELSE finished_at END,
                error_code = ?3,
                error_message = ?4,
                cancellation_reason = CASE WHEN ?1 = 'cancelled' THEN ?5 ELSE cancellation_reason END
            WHERE id = ?6
              AND status IN ({expected_placeholders})
            "#
        );
        let mut values = vec![
            next_status.as_db_value().into(),
            now.into(),
            opt_string(error_code),
            opt_string(error_message),
            opt_string(cancellation_reason),
            job_id.into(),
        ];
        values.extend(
            expected_statuses
                .iter()
                .map(|status| status.as_db_value().into()),
        );

        let result = db
            .execute_raw(raw::statement(db, sql, values)?)
            .await
            .context("Failed to transition runtime job status")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        self.get_job(job_id).await
    }

    pub async fn retry_job(&self, job_id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let Some(job) = self.get_job(job_id).await? else {
            return Ok(None);
        };
        if !matches!(
            job.status,
            RuntimeJobStatus::Failed | RuntimeJobStatus::Cancelled | RuntimeJobStatus::Expired
        ) {
            return Ok(None);
        }

        let db = self.db.connection().await?;
        let retryable_stage_count = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT COUNT(*)
                FROM job_stages
                WHERE job_id = ?1
                  AND status IN ('failed', 'cancelled', 'expired')
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to count retryable runtime job stages")?
            .ok_or_else(|| anyhow!("Runtime retryable stage count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        if retryable_stage_count == 0 {
            return Ok(None);
        }

        let now = current_timestamp_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
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
                    cancellation_reason = NULL
                WHERE id = ?2
                  AND status IN ('failed', 'cancelled', 'expired')
                "#,
                vec![now.into(), job_id.into()],
            )?)
            .await
            .context("Failed to retry runtime job")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE job_stages
            SET
                status = 'retrying',
                updated_at = ?1,
                finished_at = NULL,
                lease_expires_at = NULL,
                worker_id = NULL,
                output_artifact_ids_json = '[]',
                error_code = NULL,
                error_message = NULL
            WHERE job_id = ?2
              AND status IN ('failed', 'cancelled', 'expired')
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to retry runtime job stages")?;

        self.get_job(job_id).await
    }

    pub async fn claim_next_stage(
        &self,
        worker_id: &str,
        lease_duration_ms: u64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms)?);

        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT s.id
                FROM job_stages s
                INNER JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.status IN ('queued', 'retrying')
                  AND (s.lease_expires_at IS NULL OR s.lease_expires_at <= ?1)
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                ORDER BY j.priority DESC, s.sequence ASC, s.created_at ASC, s.id ASC
                LIMIT 1
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to select next runtime job stage")?;
        let Some(row) = row else {
            return Ok(None);
        };
        let stage_id: String = row.try_get_by_index(0)?;

        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET
                    status = 'running',
                    worker_id = ?1,
                    lease_expires_at = ?2,
                    attempt_count = attempt_count + 1,
                    started_at = COALESCE(started_at, ?3),
                    updated_at = ?3,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?4
                  AND status IN ('queued', 'retrying')
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?3)
                "#,
                vec![
                    worker_id.to_string().into(),
                    lease_expires_at.into(),
                    now.into(),
                    stage_id.clone().into(),
                ],
            )?)
            .await
            .context("Failed to claim runtime job stage")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        let stage = self
            .get_stage(&stage_id)
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job stage was not found"))?;
        let _ = self
            .transition_job_status(
                stage.job_id.as_str(),
                &[
                    RuntimeJobStatus::Created,
                    RuntimeJobStatus::Queued,
                    RuntimeJobStatus::Retrying,
                    RuntimeJobStatus::Postprocessing,
                ],
                RuntimeJobStatus::Running,
                None,
                None,
                None,
            )
            .await?;
        let job = self
            .get_job(stage.job_id.as_str())
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job was not found"))?;

        Ok(Some(ClaimedStage { job, stage }))
    }

    pub async fn complete_stage(
        &self,
        stage_id: &str,
        output_artifact_ids: Vec<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let output_json = json_to_db_string(&json!(output_artifact_ids), "[]")?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET
                    status = 'completed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    output_artifact_ids_json = ?2,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                "#,
                vec![now.into(), output_json.into(), stage_id.into()],
            )?)
            .await
            .context("Failed to complete runtime job stage")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        let stage = self
            .get_stage(stage_id)
            .await?
            .ok_or_else(|| anyhow!("Completed runtime job stage was not found"))?;
        self.complete_job_if_all_stages_finished(stage.job_id.as_str())
            .await?;
        Ok(Some(stage))
    }

    pub async fn fail_stage(
        &self,
        stage_id: &str,
        retryable: bool,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let Some(stage) = self.get_stage(stage_id).await? else {
            return Ok(None);
        };
        if !matches!(
            stage.status,
            RuntimeStageStatus::Running | RuntimeStageStatus::Postprocessing
        ) {
            return Ok(None);
        }

        if retryable && stage.attempt_count < stage.max_attempts {
            self.retry_stage(&stage, error_code, error_message).await
        } else {
            self.mark_stage_failed(&stage, error_code, error_message)
                .await
        }
    }

    pub async fn cancel_job(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();

        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'cancelled',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    cancellation_reason = ?2
                WHERE id = ?3
                  AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                "#,
                vec![now.into(), opt_string(reason), job_id.into()],
            )?)
            .await
            .context("Failed to cancel runtime job")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE job_stages
            SET
                status = 'cancelled',
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                lease_expires_at = NULL,
                worker_id = NULL
            WHERE job_id = ?2
              AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to cancel runtime job stages")?;

        self.get_job(job_id).await
    }

    pub async fn recover_expired_stage_leases(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT id
                FROM job_stages
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?1
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to list expired runtime stage leases")?;

        let mut recovered = 0_u64;
        for row in rows {
            let stage_id: String = row.try_get_by_index(0)?;
            if self
                .fail_stage(
                    stage_id.as_str(),
                    true,
                    Some("lease_expired".to_string()),
                    Some("Worker lease expired before completion".to_string()),
                )
                .await?
                .is_some()
            {
                recovered = recovered.saturating_add(1);
            }
        }

        Ok(recovered)
    }

    pub async fn queued_stage_count(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                "SELECT COUNT(*) FROM job_stages WHERE status IN ('queued', 'retrying')",
                vec![],
            )?)
            .await
            .context("Failed to count queued runtime stages")?;
        let count = row
            .ok_or_else(|| anyhow!("Queued runtime stage count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        i64_to_u64(count)
    }

    pub async fn stage_status_counts(&self) -> anyhow::Result<Vec<RuntimeStageStatusCount>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT status, COUNT(*) FROM job_stages GROUP BY status ORDER BY status",
                vec![],
            )?)
            .await
            .context("Failed to count runtime stages by status")?;

        rows.iter()
            .map(|row| {
                let status_raw: String = row.try_get_by_index(0)?;
                let status = RuntimeStageStatus::from_db_value(status_raw.as_str())
                    .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                Ok(RuntimeStageStatusCount { status, count })
            })
            .collect()
    }

    pub async fn create_stage(&self, input: NewJobStage) -> anyhow::Result<JobStage> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let id = new_uuid();
        let input_artifact_ids_json = json_to_db_string(&json!(input.input_artifact_ids), "[]")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO job_stages (
                id,
                job_id,
                created_at,
                updated_at,
                sequence,
                stage_kind,
                status,
                capability,
                model_id,
                worker_id,
                lease_expires_at,
                attempt_count,
                max_attempts,
                input_artifact_ids_json,
                output_artifact_ids_json,
                progress_json,
                started_at,
                finished_at,
                error_code,
                error_message
            )
            VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, NULL, NULL, 0, ?9, ?10, '[]', NULL, NULL, NULL, NULL, NULL)
            "#,
            vec![
                id.clone().into(),
                input.job_id.into(),
                now.into(),
                u32_to_i64_value(input.sequence).into(),
                input.stage_kind.into(),
                input.status.as_db_value().into(),
                opt_string(input.capability),
                opt_string(input.model_id),
                u32_to_i64_value(input.max_attempts).into(),
                input_artifact_ids_json.into(),
            ],
        )?)
        .await
        .context("Failed to create runtime job stage")?;

        self.get_stage(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job stage was not found"))
    }

    pub async fn get_stage(&self, id: &str) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(db, JOB_STAGE_COLUMNS_SQL, vec![id.into()])?)
            .await
            .context("Failed to load runtime job stage")?;

        row.as_ref().map(map_job_stage).transpose()
    }

    pub async fn list_stages_for_job(&self, job_id: &str) -> anyhow::Result<Vec<JobStage>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                JOB_STAGE_LIST_FOR_JOB_SQL,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to list runtime job stages")?;

        rows.iter().map(map_job_stage).collect()
    }

    pub async fn create_artifact(
        &self,
        input: NewRuntimeArtifact,
    ) -> anyhow::Result<RuntimeArtifact> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                created_at,
                artifact_kind,
                artifact_role,
                media_asset_id,
                text_asset_id,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                metadata_json,
                retention_policy
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            "#,
            vec![
                id.clone().into(),
                input.job_id.into(),
                opt_string(input.stage_id),
                now.into(),
                input.artifact_kind.as_db_value().into(),
                input.artifact_role.as_db_value().into(),
                opt_string(input.media_asset_id),
                opt_string(input.text_asset_id),
                opt_string(input.storage_key),
                opt_string(input.content_type),
                opt_string(input.filename),
                opt_u64(input.size_bytes),
                opt_string(input.sha256),
                metadata_json.into(),
                input.retention_policy.into(),
            ],
        )?)
        .await
        .context("Failed to create runtime artifact")?;

        self.get_artifact(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime artifact was not found"))
    }

    pub async fn get_artifact(&self, id: &str) -> anyhow::Result<Option<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime artifact")?;

        row.as_ref().map(map_runtime_artifact).transpose()
    }

    pub async fn list_artifacts_for_job(
        &self,
        job_id: &str,
    ) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to list runtime job artifacts")?;

        rows.iter().map(map_runtime_artifact).collect()
    }

    pub async fn record_idempotency(
        &self,
        input: NewIdempotencyRecord,
    ) -> anyhow::Result<IdempotencyRecord> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let response_json = input
            .response_json
            .as_ref()
            .map(|value| json_to_db_string(value, "{}"))
            .transpose()?;
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO idempotency_keys (
                operation,
                idempotency_key,
                created_at,
                expires_at,
                request_hash,
                response_json,
                runtime_job_id,
                conflict_message,
                metadata_json
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            vec![
                input.operation.clone().into(),
                input.idempotency_key.clone().into(),
                now.into(),
                opt_u64(input.expires_at),
                input.request_hash.into(),
                opt_string(response_json),
                opt_string(input.runtime_job_id),
                opt_string(input.conflict_message),
                metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to record idempotency key")?;

        self.get_idempotency_record(&input.operation, &input.idempotency_key)
            .await?
            .ok_or_else(|| anyhow!("Created idempotency record was not found"))
    }

    pub async fn get_idempotency_record(
        &self,
        operation: &str,
        idempotency_key: &str,
    ) -> anyhow::Result<Option<IdempotencyRecord>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                IDEMPOTENCY_RECORD_COLUMNS_SQL,
                vec![operation.into(), idempotency_key.into()],
            )?)
            .await
            .context("Failed to load idempotency record")?;

        row.as_ref().map(map_idempotency_record).transpose()
    }

    pub async fn upsert_worker_heartbeat(
        &self,
        update: WorkerHeartbeatUpdate,
    ) -> anyhow::Result<RuntimeWorkerHeartbeat> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let queue_names_json = json_to_db_string(&json!(update.queue_names), "[]")?;
        let diagnostic_json = json_to_db_string(&update.diagnostic_json, "{}")?;

        db.execute_raw(worker_heartbeat_upsert_statement(
            db,
            now,
            &update,
            queue_names_json,
            diagnostic_json,
        )?)
        .await
        .context("Failed to upsert runtime worker heartbeat")?;

        self.get_worker_heartbeat(&update.worker_id)
            .await?
            .ok_or_else(|| anyhow!("Runtime worker heartbeat was not found after upsert"))
    }

    pub async fn get_worker_heartbeat(
        &self,
        worker_id: &str,
    ) -> anyhow::Result<Option<RuntimeWorkerHeartbeat>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                WORKER_HEARTBEAT_COLUMNS_SQL,
                vec![worker_id.into()],
            )?)
            .await
            .context("Failed to load runtime worker heartbeat")?;

        row.as_ref().map(map_worker_heartbeat).transpose()
    }

    async fn retry_stage(
        &self,
        stage: &JobStage,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET
                    status = 'retrying',
                    updated_at = ?1,
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status = 'running'
                "#,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage retrying")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        let _ = self
            .transition_job_status(
                stage.job_id.as_str(),
                &[RuntimeJobStatus::Running],
                RuntimeJobStatus::Retrying,
                error_code,
                error_message,
                None,
            )
            .await?;
        self.get_stage(stage.id.as_str()).await
    }

    async fn mark_stage_failed(
        &self,
        stage: &JobStage,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET
                    status = 'failed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status = 'running'
                "#,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage failed")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        let _ = self
            .transition_job_status(
                stage.job_id.as_str(),
                &[RuntimeJobStatus::Running, RuntimeJobStatus::Retrying],
                RuntimeJobStatus::Failed,
                error_code,
                error_message,
                None,
            )
            .await?;
        self.get_stage(stage.id.as_str()).await
    }

    async fn complete_job_if_all_stages_finished(&self, job_id: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT COUNT(*)
                FROM job_stages
                WHERE job_id = ?1
                  AND status NOT IN ('completed', 'skipped')
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to inspect runtime job stage completion")?;
        let remaining = row
            .ok_or_else(|| anyhow!("Runtime stage completion count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        if remaining == 0 {
            let _ = self
                .transition_job_status(
                    job_id,
                    &[
                        RuntimeJobStatus::Running,
                        RuntimeJobStatus::Retrying,
                        RuntimeJobStatus::Postprocessing,
                        RuntimeJobStatus::Queued,
                    ],
                    RuntimeJobStatus::Completed,
                    None,
                    None,
                    None,
                )
                .await?;
        }

        Ok(())
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn current_timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

const MEDIA_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE id = ?1";
const TEXT_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, raw_text, normalized_text, language_hint, character_count, sha256, safety_status, retention_policy, structure_json FROM text_assets WHERE id = ?1";
const RUNTIME_JOB_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, queued_at, started_at, finished_at, job_kind, status, priority, model_id, capability, route_record_kind, route_record_id, input_media_asset_id, input_text_asset_id, request_json, model_snapshot_json, progress_json, error_code, error_message, attempt_count, max_attempts, retry_policy_json, idempotency_key, correlation_id, cancellation_reason FROM runtime_jobs WHERE id = ?1";
const JOB_STAGE_COLUMNS_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, status, capability, model_id, worker_id, lease_expires_at, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message FROM job_stages WHERE id = ?1";
const JOB_STAGE_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, status, capability, model_id, worker_id, lease_expires_at, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message FROM job_stages WHERE job_id = ?1 ORDER BY sequence ASC, created_at ASC, id ASC";
const RUNTIME_ARTIFACT_COLUMNS_SQL: &str =
    "SELECT id, job_id, stage_id, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE id = ?1";
const RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, stage_id, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE job_id = ?1 ORDER BY created_at ASC, id ASC";
const IDEMPOTENCY_RECORD_COLUMNS_SQL: &str =
    "SELECT operation, idempotency_key, created_at, expires_at, request_hash, response_json, runtime_job_id, conflict_message, metadata_json FROM idempotency_keys WHERE operation = ?1 AND idempotency_key = ?2";
const WORKER_HEARTBEAT_COLUMNS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats WHERE worker_id = ?1";

fn worker_heartbeat_upsert_statement(
    db: &DatabaseConnection,
    now: i64,
    update: &WorkerHeartbeatUpdate,
    queue_names_json: String,
    diagnostic_json: String,
) -> anyhow::Result<sea_orm::Statement> {
    let values = vec![
        update.worker_id.clone().into(),
        now.into(),
        update.status.clone().into(),
        queue_names_json.into(),
        opt_string(update.current_job_id.clone()),
        opt_string(update.current_stage_id.clone()),
        diagnostic_json.into(),
    ];

    match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => raw::statement(
            db,
            r#"
            INSERT INTO runtime_worker_heartbeats (
                worker_id,
                started_at,
                last_heartbeat_at,
                status,
                queue_names_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(worker_id) DO UPDATE SET
                last_heartbeat_at = excluded.last_heartbeat_at,
                status = excluded.status,
                queue_names_json = excluded.queue_names_json,
                current_job_id = excluded.current_job_id,
                current_stage_id = excluded.current_stage_id,
                diagnostic_json = excluded.diagnostic_json
            "#,
            values,
        ),
        DbBackend::MySql => raw::statement(
            db,
            r#"
            INSERT INTO runtime_worker_heartbeats (
                worker_id,
                started_at,
                last_heartbeat_at,
                status,
                queue_names_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7)
            ON DUPLICATE KEY UPDATE
                last_heartbeat_at = VALUES(last_heartbeat_at),
                status = VALUES(status),
                queue_names_json = VALUES(queue_names_json),
                current_job_id = VALUES(current_job_id),
                current_stage_id = VALUES(current_stage_id),
                diagnostic_json = VALUES(diagnostic_json)
            "#,
            values,
        ),
        backend => bail!("Unsupported SeaORM database backend: {backend:?}"),
    }
}

fn map_media_asset(row: &QueryResult) -> anyhow::Result<MediaAsset> {
    Ok(MediaAsset {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        asset_kind: row.try_get_by_index(3)?,
        storage_namespace: row.try_get_by_index(4)?,
        storage_key: row.try_get_by_index(5)?,
        content_type: row.try_get_by_index(6)?,
        filename: row.try_get_by_index(7)?,
        size_bytes: i64_to_u64(row.try_get_by_index(8)?)?,
        sha256: row.try_get_by_index(9)?,
        duration_secs: row.try_get_by_index(10)?,
        sample_rate_hz: opt_i64_to_u32(row.try_get_by_index(11)?)?,
        channel_count: opt_i64_to_u16(row.try_get_by_index(12)?)?,
        peak_amplitude: row
            .try_get_by_index::<Option<f64>>(13)?
            .map(|value| value as f32),
        rms_amplitude: row
            .try_get_by_index::<Option<f64>>(14)?
            .map(|value| value as f32),
        scan_status: row.try_get_by_index(15)?,
        retention_policy: row.try_get_by_index(16)?,
        deleted_at: opt_i64_to_u64(row.try_get_by_index(17)?)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(18)?, json!({})),
    })
}

fn map_text_asset(row: &QueryResult) -> anyhow::Result<TextAsset> {
    Ok(TextAsset {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        raw_text: row.try_get_by_index(3)?,
        normalized_text: row.try_get_by_index(4)?,
        language_hint: row.try_get_by_index(5)?,
        character_count: i64_to_u64(row.try_get_by_index(6)?)?,
        sha256: row.try_get_by_index(7)?,
        safety_status: row.try_get_by_index(8)?,
        retention_policy: row.try_get_by_index(9)?,
        structure_json: parse_json_value(row.try_get_by_index::<String>(10)?, json!({})),
    })
}

fn map_runtime_job(row: &QueryResult) -> anyhow::Result<RuntimeJob> {
    let kind_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(7)?;

    Ok(RuntimeJob {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        queued_at: opt_i64_to_u64(row.try_get_by_index(3)?)?,
        started_at: opt_i64_to_u64(row.try_get_by_index(4)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(5)?)?,
        job_kind: RuntimeJobKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {kind_raw}"))?,
        status: RuntimeJobStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?,
        priority: i64_to_i32(row.try_get_by_index(8)?)?,
        model_id: row.try_get_by_index(9)?,
        capability: row.try_get_by_index(10)?,
        route_record_kind: row.try_get_by_index(11)?,
        route_record_id: row.try_get_by_index(12)?,
        input_media_asset_id: row.try_get_by_index(13)?,
        input_text_asset_id: row.try_get_by_index(14)?,
        request_json: parse_json_value(row.try_get_by_index::<String>(15)?, json!({})),
        model_snapshot_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        progress_json: row
            .try_get_by_index::<Option<String>>(17)?
            .map(|raw| parse_json_value(raw, json!({}))),
        error_code: row.try_get_by_index(18)?,
        error_message: row.try_get_by_index(19)?,
        attempt_count: i64_to_u32(row.try_get_by_index(20)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(21)?)?,
        retry_policy_json: parse_json_value(row.try_get_by_index::<String>(22)?, json!({})),
        idempotency_key: row.try_get_by_index(23)?,
        correlation_id: row.try_get_by_index(24)?,
        cancellation_reason: row.try_get_by_index(25)?,
    })
}

fn map_job_stage(row: &QueryResult) -> anyhow::Result<JobStage> {
    let status_raw: String = row.try_get_by_index(6)?;

    Ok(JobStage {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(3)?)?,
        sequence: i64_to_u32(row.try_get_by_index(4)?)?,
        stage_kind: row.try_get_by_index(5)?,
        status: RuntimeStageStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?,
        capability: row.try_get_by_index(7)?,
        model_id: row.try_get_by_index(8)?,
        worker_id: row.try_get_by_index(9)?,
        lease_expires_at: opt_i64_to_u64(row.try_get_by_index(10)?)?,
        attempt_count: i64_to_u32(row.try_get_by_index(11)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(12)?)?,
        input_artifact_ids: parse_string_array(row.try_get_by_index::<String>(13)?),
        output_artifact_ids: parse_string_array(row.try_get_by_index::<String>(14)?),
        progress_json: row
            .try_get_by_index::<Option<String>>(15)?
            .map(|raw| parse_json_value(raw, json!({}))),
        started_at: opt_i64_to_u64(row.try_get_by_index(16)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(17)?)?,
        error_code: row.try_get_by_index(18)?,
        error_message: row.try_get_by_index(19)?,
    })
}

fn map_runtime_artifact(row: &QueryResult) -> anyhow::Result<RuntimeArtifact> {
    let kind_raw: String = row.try_get_by_index(4)?;
    let role_raw: String = row.try_get_by_index(5)?;

    Ok(RuntimeArtifact {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        stage_id: row.try_get_by_index(2)?,
        created_at: i64_to_u64(row.try_get_by_index(3)?)?,
        artifact_kind: RuntimeArtifactKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact kind: {kind_raw}"))?,
        artifact_role: RuntimeArtifactRole::from_db_value(role_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact role: {role_raw}"))?,
        media_asset_id: row.try_get_by_index(6)?,
        text_asset_id: row.try_get_by_index(7)?,
        storage_key: row.try_get_by_index(8)?,
        content_type: row.try_get_by_index(9)?,
        filename: row.try_get_by_index(10)?,
        size_bytes: opt_i64_to_u64(row.try_get_by_index(11)?)?,
        sha256: row.try_get_by_index(12)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(13)?, json!({})),
        retention_policy: row.try_get_by_index(14)?,
    })
}

fn map_idempotency_record(row: &QueryResult) -> anyhow::Result<IdempotencyRecord> {
    Ok(IdempotencyRecord {
        operation: row.try_get_by_index(0)?,
        idempotency_key: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        expires_at: opt_i64_to_u64(row.try_get_by_index(3)?)?,
        request_hash: row.try_get_by_index(4)?,
        response_json: row
            .try_get_by_index::<Option<String>>(5)?
            .map(|raw| parse_json_value(raw, json!({}))),
        runtime_job_id: row.try_get_by_index(6)?,
        conflict_message: row.try_get_by_index(7)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(8)?, json!({})),
    })
}

fn map_worker_heartbeat(row: &QueryResult) -> anyhow::Result<RuntimeWorkerHeartbeat> {
    Ok(RuntimeWorkerHeartbeat {
        worker_id: row.try_get_by_index(0)?,
        started_at: i64_to_u64(row.try_get_by_index(1)?)?,
        last_heartbeat_at: i64_to_u64(row.try_get_by_index(2)?)?,
        status: row.try_get_by_index(3)?,
        queue_names: parse_string_array(row.try_get_by_index::<String>(4)?),
        current_job_id: row.try_get_by_index(5)?,
        current_stage_id: row.try_get_by_index(6)?,
        diagnostic_json: parse_json_value(row.try_get_by_index::<String>(7)?, json!({})),
    })
}

fn json_to_db_string(value: &serde_json::Value, fallback: &str) -> anyhow::Result<String> {
    serde_json::to_string(value)
        .or_else(|_| Ok::<String, serde_json::Error>(fallback.to_string()))
        .context("Failed to serialize runtime JSON payload")
}

fn parse_json_value(raw: String, fallback: serde_json::Value) -> serde_json::Value {
    serde_json::from_str(raw.as_str()).unwrap_or(fallback)
}

fn parse_string_array(raw: String) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw.as_str()).unwrap_or_default()
}

fn opt_string(value: Option<String>) -> Value {
    Value::String(value)
}

fn opt_u64(value: Option<u64>) -> Value {
    Value::BigInt(value.and_then(|value| i64::try_from(value).ok()))
}

fn opt_i64(value: Option<i64>) -> Value {
    Value::BigInt(value)
}

fn opt_u32(value: Option<u32>) -> Value {
    Value::BigInt(value.map(i64::from))
}

fn opt_u16(value: Option<u16>) -> Value {
    Value::BigInt(value.map(i64::from))
}

fn opt_f64(value: Option<f64>) -> Value {
    Value::Double(value)
}

fn opt_f32(value: Option<f32>) -> Value {
    Value::Double(value.map(f64::from))
}

fn u64_to_i64_value(value: u64) -> anyhow::Result<Value> {
    Ok(Value::BigInt(Some(i64::try_from(value)?)))
}

fn u32_to_i64_value(value: u32) -> i64 {
    i64::from(value)
}

fn i64_to_u64(value: i64) -> anyhow::Result<u64> {
    u64::try_from(value).map_err(Into::into)
}

fn opt_i64_to_u64(value: Option<i64>) -> anyhow::Result<Option<u64>> {
    value.map(i64_to_u64).transpose()
}

fn i64_to_u32(value: i64) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(Into::into)
}

fn opt_i64_to_u32(value: Option<i64>) -> anyhow::Result<Option<u32>> {
    value.map(i64_to_u32).transpose()
}

fn opt_i64_to_u16(value: Option<i64>) -> anyhow::Result<Option<u16>> {
    value
        .map(|value| u16::try_from(value).map_err(Into::into))
        .transpose()
}

fn i64_to_i32(value: i64) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(Into::into)
}

fn is_terminal_job_status(status: RuntimeJobStatus) -> bool {
    matches!(
        status,
        RuntimeJobStatus::Completed
            | RuntimeJobStatus::Failed
            | RuntimeJobStatus::Cancelled
            | RuntimeJobStatus::Expired
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::StoreDatabase;
    use tempfile::TempDir;

    fn build_store() -> (BatchRuntimeStore, TempDir) {
        let root = tempfile::tempdir().expect("temp dir");
        let db_path = root.path().join("runtime.sqlite");
        (
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path)),
            root,
        )
    }

    #[tokio::test]
    async fn creates_runtime_foundation_records() {
        let (store, _root) = build_store();

        let media = store
            .create_media_asset(NewMediaAsset {
                asset_kind: "audio_original".to_string(),
                storage_namespace: "uploads".to_string(),
                storage_key: "uploads/transcription/test.wav".to_string(),
                content_type: "audio/wav".to_string(),
                filename: Some("test.wav".to_string()),
                size_bytes: 4,
                sha256: Some(sha256_hex(&[1, 2, 3, 4])),
                duration_secs: Some(1.25),
                sample_rate_hz: Some(16_000),
                channel_count: Some(1),
                peak_amplitude: Some(0.5),
                rms_amplitude: Some(0.1),
                scan_status: "passed".to_string(),
                retention_policy: "default".to_string(),
                metadata_json: json!({"source": "test"}),
            })
            .await
            .expect("media asset");

        let text = store
            .create_text_asset(NewTextAsset {
                raw_text: "Hello world".to_string(),
                normalized_text: None,
                language_hint: Some("en".to_string()),
                sha256: Some(sha256_hex(b"Hello world")),
                safety_status: "allowed".to_string(),
                retention_policy: "default".to_string(),
                structure_json: json!({"kind": "plain"}),
            })
            .await
            .expect("text asset");

        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 5,
                model_id: Some("Granite-Speech-4.1-2B".to_string()),
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("route-1".to_string()),
                input_media_asset_id: Some(media.id.clone()),
                input_text_asset_id: Some(text.id.clone()),
                request_json: json!({"language": "en"}),
                model_snapshot_json: json!({"license": "apache-2.0"}),
                retry_policy_json: json!({"max_attempts": 2}),
                max_attempts: 2,
                idempotency_key: Some("idem-1".to_string()),
                correlation_id: Some("corr-1".to_string()),
            })
            .await
            .expect("job");

        assert_eq!(job.status, RuntimeJobStatus::Queued);
        assert_eq!(job.queued_at, Some(job.created_at));
        assert_eq!(job.input_media_asset_id.as_deref(), Some(media.id.as_str()));

        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 10,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let artifact = store
            .create_artifact(NewRuntimeArtifact {
                job_id: job.id.clone(),
                stage_id: Some(stage.id.clone()),
                artifact_kind: RuntimeArtifactKind::Transcript,
                artifact_role: RuntimeArtifactRole::OutputPrimary,
                media_asset_id: None,
                text_asset_id: Some(text.id.clone()),
                storage_key: None,
                content_type: Some("application/json".to_string()),
                filename: Some("transcript.json".to_string()),
                size_bytes: Some(128),
                sha256: None,
                metadata_json: json!({"format": "segments"}),
                retention_policy: "default".to_string(),
            })
            .await
            .expect("artifact");

        let idempotency = store
            .record_idempotency(NewIdempotencyRecord {
                operation: "job.create".to_string(),
                idempotency_key: "idem-1".to_string(),
                expires_at: None,
                request_hash: sha256_hex(br#"{"language":"en"}"#),
                response_json: Some(json!({"job_id": job.id})),
                runtime_job_id: Some(job.id.clone()),
                conflict_message: None,
                metadata_json: json!({}),
            })
            .await
            .expect("idempotency");

        let heartbeat = store
            .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
                worker_id: "worker-1".to_string(),
                status: "idle".to_string(),
                queue_names: vec!["batch".to_string()],
                current_job_id: None,
                current_stage_id: None,
                diagnostic_json: json!({"pid": 123}),
            })
            .await
            .expect("heartbeat");

        assert_eq!(artifact.stage_id.as_deref(), Some(stage.id.as_str()));
        assert_eq!(idempotency.runtime_job_id.as_deref(), Some(job.id.as_str()));
        assert_eq!(heartbeat.queue_names, vec!["batch"]);
    }

    #[tokio::test]
    async fn job_transitions_are_status_conditional() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("speech_history".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({"text": "hello"}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");

        let cancelled = store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Queued],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("user requested".to_string()),
            )
            .await
            .expect("cancel transition")
            .expect("job should transition");

        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
        assert_eq!(
            cancelled.cancellation_reason.as_deref(),
            Some("user requested")
        );

        let late_completion = store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Running],
                RuntimeJobStatus::Completed,
                None,
                None,
                None,
            )
            .await
            .expect("late transition should not error");

        assert!(late_completion.is_none());
        let fetched = store
            .get_job(&job.id)
            .await
            .expect("fetch")
            .expect("job still exists");
        assert_eq!(fetched.status, RuntimeJobStatus::Cancelled);
    }

    #[tokio::test]
    async fn queue_claim_recovery_and_cancel_are_durable() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 10,
                model_id: None,
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("route-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": 2}),
                max_attempts: 2,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: None,
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let claimed = store
            .claim_next_stage("worker-1", 0)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        assert_eq!(claimed.stage.id, stage.id);
        assert_eq!(claimed.stage.status, RuntimeStageStatus::Running);
        assert_eq!(claimed.stage.attempt_count, 1);

        let recovered = store.recover_expired_stage_leases().await.expect("recover");
        assert_eq!(recovered, 1);

        let retried = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(retried.status, RuntimeStageStatus::Retrying);
        assert_eq!(store.queued_stage_count().await.expect("count"), 1);

        let cancelled = store
            .cancel_job(&job.id, Some("test cleanup".to_string()))
            .await
            .expect("cancel")
            .expect("job should cancel");
        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);

        let cancelled_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(cancelled_stage.status, RuntimeStageStatus::Cancelled);
    }

    #[tokio::test]
    async fn manual_retry_requeues_failed_job_and_stage() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("text_to_speech".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({"text": "hello"}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": 1}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "tts_synthesize".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        assert_eq!(claimed.stage.id, stage.id);

        let failed_stage = store
            .fail_stage(
                &stage.id,
                false,
                Some("boom".to_string()),
                Some("first attempt failed".to_string()),
            )
            .await
            .expect("fail")
            .expect("stage should fail");
        assert_eq!(failed_stage.status, RuntimeStageStatus::Failed);

        let failed_job = store
            .get_job(&job.id)
            .await
            .expect("job")
            .expect("job exists");
        assert_eq!(failed_job.status, RuntimeJobStatus::Failed);

        let retried_job = store
            .retry_job(&job.id)
            .await
            .expect("retry")
            .expect("job should retry");
        assert_eq!(retried_job.status, RuntimeJobStatus::Queued);
        assert_eq!(retried_job.attempt_count, 1);
        assert!(retried_job.error_code.is_none());
        assert!(retried_job.finished_at.is_none());

        let retried_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(retried_stage.status, RuntimeStageStatus::Retrying);
        assert!(retried_stage.error_code.is_none());
        assert!(retried_stage.finished_at.is_none());

        let stages = store
            .list_stages_for_job(&job.id)
            .await
            .expect("list stages");
        assert_eq!(stages.len(), 1);
        let stage_counts = store.stage_status_counts().await.expect("stage counts");
        assert_eq!(
            stage_counts,
            vec![RuntimeStageStatusCount {
                status: RuntimeStageStatus::Retrying,
                count: 1,
            }]
        );
    }
}

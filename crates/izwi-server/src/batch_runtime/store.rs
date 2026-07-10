use super::types::{
    ClaimedStage, IdempotencyRecord, JobStage, MediaAsset, QueueClass, RuntimeArtifact,
    RuntimeArtifactKind, RuntimeArtifactRole, RuntimeJob, RuntimeJobKind, RuntimeJobStatus,
    RuntimeStageStatus, RuntimeWorkerHeartbeat, RuntimeWorkerHeartbeatDetails,
    RuntimeWorkerRegistration, StageLease, StageResourceHints, TextAsset, WorkerResourceCapacity,
    WORKER_HEARTBEAT_DETAILS_VERSION, WORKER_REGISTRATION_VERSION,
};
#[cfg(test)]
use super::types::{DeviceClass, ResourceTarget, RuntimeBackendClass};
use crate::{
    db::{raw, StoreDatabase},
    ids::new_uuid,
};
use anyhow::{anyhow, bail, Context};
use sea_orm::{
    ConnectionTrait, DatabaseConnection, DbBackend, QueryResult, TransactionTrait, Value,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    time::{SystemTime, UNIX_EPOCH},
};

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
pub struct NewJobStageDispatch {
    pub stage: NewJobStage,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
}

const DEFAULT_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 64;
const MAX_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageClaimFilter {
    pub queue_names: Vec<String>,
    pub capabilities: Vec<String>,
    pub model_ids: Vec<String>,
    pub stage_kinds: Vec<String>,
    pub resources: WorkerResourceCapacity,
    pub max_candidates: usize,
}

impl Default for StageClaimFilter {
    fn default() -> Self {
        Self {
            queue_names: vec!["batch".to_string()],
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            resources: WorkerResourceCapacity::default(),
            max_candidates: DEFAULT_STAGE_CLAIM_CANDIDATE_LIMIT,
        }
    }
}

impl StageClaimFilter {
    pub fn for_worker_queues(queue_names: &[String]) -> Self {
        let mut filter = Self::default();
        filter.queue_names = normalize_filter_values(queue_names);
        if filter.queue_names.is_empty() {
            filter.queue_names.push("batch".to_string());
        }
        filter
    }

    fn normalized(&self) -> Self {
        Self {
            queue_names: normalize_filter_values(&self.queue_names),
            capabilities: normalize_filter_values(&self.capabilities),
            model_ids: normalize_filter_values(&self.model_ids),
            stage_kinds: normalize_filter_values(&self.stage_kinds),
            resources: self.resources.clone(),
            max_candidates: self.max_candidates,
        }
    }

    pub fn matches(&self, candidate: &StageClaimCandidate) -> bool {
        self.queue_matches(candidate)
            && optional_filter_matches(&self.capabilities, candidate.capability.as_deref())
            && optional_filter_matches(&self.model_ids, candidate.model_id.as_deref())
            && optional_filter_matches(&self.stage_kinds, Some(candidate.stage_kind.as_str()))
            && self.resources.supports(&candidate.resource_hints)
    }

    fn queue_matches(&self, candidate: &StageClaimCandidate) -> bool {
        if self.queue_names.is_empty() {
            return true;
        }

        self.queue_names.iter().any(|queue| {
            queue == QueueClass::Batch.as_db_value() || queue == candidate.queue_class.as_db_value()
        })
    }

    fn candidate_limit(&self) -> usize {
        self.max_candidates
            .clamp(1, MAX_STAGE_CLAIM_CANDIDATE_LIMIT)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageClaimCandidate {
    pub stage_id: String,
    pub stage_kind: String,
    pub job_kind: RuntimeJobKind,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
    pub capability: Option<String>,
    pub model_id: Option<String>,
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
pub struct NewStageOutputArtifact {
    pub publication_key: String,
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

#[derive(Debug, Clone)]
pub struct RegisteredWorkerHeartbeatUpdate {
    pub registration: RuntimeWorkerRegistration,
    pub status: String,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub details: RuntimeWorkerHeartbeatDetails,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeReconciliationReport {
    pub jobs_repaired: u64,
    pub stages_repaired: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct StoredRetryPolicy {
    max_attempts: Option<u32>,
    #[serde(alias = "backoff_ms", alias = "base_delay_ms")]
    initial_backoff_ms: u64,
    backoff_multiplier: f64,
    max_backoff_ms: u64,
    jitter_ratio: f64,
}

impl Default for StoredRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: None,
            // Preserve the current immediate-retry behavior unless a producer opts in.
            initial_backoff_ms: 0,
            backoff_multiplier: 2.0,
            max_backoff_ms: 60_000,
            jitter_ratio: 0.2,
        }
    }
}

impl StoredRetryPolicy {
    fn from_job(job: &RuntimeJob) -> Self {
        serde_json::from_value(job.retry_policy_json.clone()).unwrap_or_default()
    }

    fn effective_max_attempts(&self, job: &RuntimeJob, stage: &JobStage) -> u32 {
        self.max_attempts
            .unwrap_or(job.max_attempts)
            .min(job.max_attempts)
            .min(stage.max_attempts)
    }

    fn backoff_ms(&self, stage: &JobStage) -> u64 {
        if self.initial_backoff_ms == 0 {
            return 0;
        }

        const MAX_RETRY_BACKOFF_MS: u64 = 24 * 60 * 60 * 1_000;
        let exponent = stage.attempt_count.saturating_sub(1).min(31) as i32;
        let multiplier = self.backoff_multiplier.clamp(1.0, 10.0);
        let delay = (self.initial_backoff_ms as f64 * multiplier.powi(exponent)) as u64;
        let configured_max = self
            .max_backoff_ms
            .max(self.initial_backoff_ms)
            .min(MAX_RETRY_BACKOFF_MS);
        let delay = delay.min(configured_max);
        let jitter_span = (delay as f64 * self.jitter_ratio.clamp(0.0, 1.0)) as u64;
        if jitter_span == 0 {
            return delay;
        }

        // Stable per stage attempt so recovery workers converge on the same eligibility time.
        let mut hasher = DefaultHasher::new();
        stage.id.hash(&mut hasher);
        stage.attempt_count.hash(&mut hasher);
        let width = jitter_span.saturating_mul(2).saturating_add(1);
        delay
            .saturating_sub(jitter_span)
            .saturating_add(hasher.finish() % width)
            .min(configured_max)
    }
}

#[derive(Debug, Clone, Copy)]
enum LeaseValidity {
    Active,
    Expired,
}

impl LeaseValidity {
    fn sql_predicate(self) -> &'static str {
        match self {
            Self::Active => "lease_expires_at > ?7",
            Self::Expired => "lease_expires_at <= ?7",
        }
    }
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
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime job retry transaction")?;
        let Some(job) = get_job_with(&tx, job_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !matches!(
            job.status,
            RuntimeJobStatus::Failed | RuntimeJobStatus::Cancelled | RuntimeJobStatus::Expired
        ) {
            tx.rollback().await?;
            return Ok(None);
        }
        if job.attempt_count >= job.max_attempts {
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
                  AND status IN ('failed', 'cancelled', 'expired')
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

        let now = current_timestamp_millis();
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
                    cancellation_reason = NULL
                WHERE id = ?2
                  AND status IN ('failed', 'cancelled', 'expired')
                  AND attempt_count < max_attempts
                "#,
                vec![now.into(), job_id.into()],
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
                output_artifact_ids_json = '[]',
                error_code = NULL,
                error_message = NULL
            WHERE job_id = ?2
              AND status IN ('failed', 'cancelled', 'expired')
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
        self.get_job(job_id).await
    }

    pub async fn claim_next_stage(
        &self,
        worker_id: &str,
        lease_duration_ms: u64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        self.claim_next_stage_with_filter(
            worker_id,
            lease_duration_ms,
            &StageClaimFilter::default(),
        )
        .await
    }

    pub async fn claim_next_stage_with_filter(
        &self,
        worker_id: &str,
        lease_duration_ms: u64,
        filter: &StageClaimFilter,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms)?);
        let filter = filter.normalized();
        let mut params: Vec<Value> = vec![now.into()];
        let mut claim_filter_sql = String::new();
        push_claim_queue_clause(&mut claim_filter_sql, &mut params, &filter.queue_names);
        push_claim_resource_clause(&mut claim_filter_sql, &mut params, &filter.resources);
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "COALESCE(s.capability, j.capability)",
            &filter.capabilities,
        );
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "COALESCE(s.model_id, j.model_id)",
            &filter.model_ids,
        );
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "s.stage_kind",
            &filter.stage_kinds,
        );
        let limit_placeholder = params.len() + 1;
        params.push(i64::try_from(filter.candidate_limit())?.into());

        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    r#"
                SELECT
                    s.id,
                    s.stage_kind,
                    j.job_kind,
                    COALESCE(s.capability, j.capability),
                    COALESCE(s.model_id, j.model_id),
                    s.queue_class,
                    s.resource_hints_json
                FROM job_stages s
                INNER JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.status IN ('queued', 'retrying')
                  AND (s.available_at IS NULL OR s.available_at <= ?1)
                  AND (s.lease_expires_at IS NULL OR s.lease_expires_at <= ?1)
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM job_stages predecessor
                      WHERE predecessor.job_id = s.job_id
                        AND predecessor.sequence < s.sequence
                        AND predecessor.status NOT IN ('completed', 'skipped')
                  )
                  {claim_filter_sql}
                ORDER BY j.priority DESC, s.sequence ASC, s.created_at ASC, s.id ASC
                LIMIT ?{limit_placeholder}
                "#,
                ),
                params,
            )?)
            .await
            .context("Failed to select next runtime job stage")?;
        let candidates = rows
            .iter()
            .map(map_stage_claim_candidate)
            .collect::<anyhow::Result<Vec<_>>>()?;
        for candidate in candidates
            .into_iter()
            .filter(|candidate| filter.matches(candidate))
        {
            if let Some(claimed) = self
                .try_claim_stage_candidate(db, candidate, worker_id, now, lease_expires_at)
                .await?
            {
                return Ok(Some(claimed));
            }
        }

        Ok(None)
    }

    async fn try_claim_stage_candidate(
        &self,
        db: &DatabaseConnection,
        candidate: StageClaimCandidate,
        worker_id: &str,
        now: i64,
        lease_expires_at: i64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime stage claim transaction")?;
        let attempt_token = new_uuid();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = 'running',
                    worker_id = ?1,
                    lease_expires_at = ?2,
                    available_at = NULL,
                    attempt_token = ?5,
                    attempt_count = attempt_count + 1,
                    started_at = COALESCE(started_at, ?3),
                    updated_at = ?3,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?4
                  AND status IN ('queued', 'retrying')
                  AND (available_at IS NULL OR available_at <= ?3)
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?3)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM job_stages predecessor
                      WHERE predecessor.job_id = job_stages.job_id
                        AND predecessor.sequence < job_stages.sequence
                        AND predecessor.status NOT IN ('completed', 'skipped')
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN (
                            'created',
                            'queued',
                            'running',
                            'retrying',
                            'postprocessing'
                        )
                  )
                "#,
                vec![
                    worker_id.to_string().into(),
                    lease_expires_at.into(),
                    now.into(),
                    candidate.stage_id.clone().into(),
                    attempt_token.clone().into(),
                ],
            )?)
            .await
            .context("Failed to claim runtime job stage")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'running',
                updated_at = ?1,
                started_at = COALESCE(started_at, ?1),
                error_code = NULL,
                error_message = NULL
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
            "#,
            vec![now.into(), candidate.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to mark claimed runtime job running")?;

        let stage = get_stage_with(&tx, &candidate.stage_id)
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job stage was not found"))?;
        if stage.status != RuntimeStageStatus::Running
            || stage.worker_id.as_deref() != Some(worker_id)
            || stage.lease_expires_at != Some(u64::try_from(lease_expires_at)?)
            || stage.attempt_token.as_deref() != Some(attempt_token.as_str())
        {
            tx.rollback().await?;
            return Ok(None);
        }

        let job = get_job_with(&tx, stage.job_id.as_str())
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job was not found"))?;
        if !is_claimable_job_status(job.status) {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.commit()
            .await
            .context("Failed to commit runtime stage claim transaction")?;
        Ok(Some(ClaimedStage { job, stage }))
    }

    pub async fn complete_stage(
        &self,
        lease: &StageLease,
        output_artifact_ids: Vec<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime stage completion transaction")?;
        let now = current_timestamp_millis();
        let output_json = json_to_db_string(&json!(output_artifact_ids), "[]")?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
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
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?1
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  )
                "#,
                vec![
                    now.into(),
                    output_json.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to complete runtime job stage")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let stage = get_stage_with(&tx, &lease.stage_id)
            .await?
            .ok_or_else(|| anyhow!("Completed runtime job stage was not found"))?;
        complete_job_if_all_stages_finished_with(&tx, stage.job_id.as_str(), now).await?;
        tx.commit()
            .await
            .context("Failed to commit runtime stage completion transaction")?;
        Ok(Some(stage))
    }

    pub async fn renew_stage_lease(
        &self,
        lease: &StageLease,
        lease_duration_ms: u64,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms.max(1))?);
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET lease_expires_at = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?2
                "#,
                vec![
                    lease_expires_at.into(),
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to renew runtime job stage lease")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn stage_lease_is_active(&self, lease: &StageLease) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT 1
                FROM job_stages s
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.id = ?1
                  AND s.status IN ('running', 'postprocessing')
                  AND s.worker_id = ?2
                  AND s.attempt_count = ?3
                  AND (s.attempt_token = ?4 OR (s.attempt_token IS NULL AND ?4 IS NULL))
                  AND s.lease_expires_at IS NOT NULL
                  AND s.lease_expires_at > ?5
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                LIMIT 1
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                    now.into(),
                ],
            )?)
            .await
            .context("Failed to verify runtime stage lease ownership")?;
        Ok(row.is_some())
    }

    pub async fn update_stage_progress(
        &self,
        lease: &StageLease,
        progress: serde_json::Value,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let progress_json = json_to_db_string(&progress, "{}")?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET progress_json = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?2
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  )
                "#,
                vec![
                    progress_json.into(),
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to update runtime stage progress")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn relinquish_stage_lease(
        &self,
        lease: &StageLease,
        error_code: impl Into<String>,
        reason: impl Into<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        self.fail_stage(lease, true, Some(error_code.into()), Some(reason.into()))
            .await
    }

    pub async fn fail_stage(
        &self,
        lease: &StageLease,
        retryable: bool,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime stage failure transaction")?;
        let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !matches!(
            stage.status,
            RuntimeStageStatus::Running | RuntimeStageStatus::Postprocessing
        ) {
            tx.rollback().await?;
            return Ok(None);
        }
        let job = get_job_with(&tx, &stage.job_id)
            .await?
            .ok_or_else(|| anyhow!("Runtime stage parent job was not found"))?;
        if !is_claimable_job_status(job.status) {
            tx.rollback().await?;
            return Ok(None);
        }
        let policy = StoredRetryPolicy::from_job(&job);
        let should_retry =
            retryable && stage.attempt_count < policy.effective_max_attempts(&job, &stage);
        let now = current_timestamp_millis();

        let result = if should_retry {
            let available_at = now.saturating_add(i64::try_from(policy.backoff_ms(&stage))?);
            self.retry_stage(
                &tx,
                &stage,
                lease,
                LeaseValidity::Active,
                now,
                available_at,
                error_code,
                error_message,
            )
            .await
        } else {
            self.mark_stage_failed(
                &tx,
                &stage,
                lease,
                LeaseValidity::Active,
                now,
                error_code,
                error_message,
            )
            .await
        }?;
        if result.is_none() {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.commit()
            .await
            .context("Failed to commit runtime stage failure transaction")?;
        Ok(result)
    }

    pub async fn cancel_job(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime job cancellation transaction")?;
        let now = current_timestamp_millis();

        let result = tx
            .execute_raw(raw::statement(
                &tx,
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
            tx.rollback().await?;
            return Ok(None);
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET
                status = 'cancelled',
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                lease_expires_at = NULL,
                worker_id = NULL,
                available_at = NULL
            WHERE job_id = ?2
              AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to cancel runtime job stages")?;

        tx.commit()
            .await
            .context("Failed to commit runtime job cancellation transaction")?;
        self.get_job(job_id).await
    }

    pub async fn recover_expired_stage_leases(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT id, worker_id, attempt_count, attempt_token
                FROM job_stages
                WHERE status IN ('running', 'postprocessing')
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
            let worker_id: Option<String> = row.try_get_by_index(1)?;
            let Some(worker_id) = worker_id else {
                continue;
            };
            let attempt_count = i64_to_u32(row.try_get_by_index(2)?)?;
            let lease = StageLease {
                stage_id,
                worker_id,
                attempt_count,
                attempt_token: row.try_get_by_index(3)?,
            };
            let tx = db
                .begin()
                .await
                .context("Failed to start expired lease recovery transaction")?;
            let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? else {
                tx.rollback().await?;
                continue;
            };
            let Some(job) = get_job_with(&tx, &stage.job_id).await? else {
                tx.rollback().await?;
                continue;
            };
            let policy = StoredRetryPolicy::from_job(&job);
            let result = if stage.attempt_count < policy.effective_max_attempts(&job, &stage) {
                let available_at = now.saturating_add(i64::try_from(policy.backoff_ms(&stage))?);
                self.retry_stage(
                    &tx,
                    &stage,
                    &lease,
                    LeaseValidity::Expired,
                    now,
                    available_at,
                    Some("lease_expired".to_string()),
                    Some("Worker lease expired before completion".to_string()),
                )
                .await?
            } else {
                self.mark_stage_failed(
                    &tx,
                    &stage,
                    &lease,
                    LeaseValidity::Expired,
                    now,
                    Some("lease_expired".to_string()),
                    Some("Worker lease expired before completion".to_string()),
                )
                .await?
            };
            if result.is_some() {
                tx.commit()
                    .await
                    .context("Failed to commit expired lease recovery transaction")?;
                recovered = recovered.saturating_add(1);
            } else {
                tx.rollback().await?;
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
        let queue_class = queue_class_for_stage_kind(&input.stage_kind);
        self.create_stage_with_dispatch(NewJobStageDispatch {
            stage: input,
            queue_class,
            resource_hints: StageResourceHints::default(),
        })
        .await
    }

    pub async fn create_stage_with_dispatch(
        &self,
        input: NewJobStageDispatch,
    ) -> anyhow::Result<JobStage> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let id = new_uuid();
        let stage = input.stage;
        let resource_hints = input.resource_hints.normalized();
        let resource_hints_json = json_to_db_string(&json!(resource_hints), "{}")?;
        let input_artifact_ids_json = json_to_db_string(&json!(stage.input_artifact_ids), "[]")?;

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
                queue_class,
                resource_hints_json,
                resource_target,
                required_backend,
                required_device_class,
                min_resource_memory_bytes,
                resource_concurrency_weight,
                status,
                capability,
                model_id,
                worker_id,
                lease_expires_at,
                available_at,
                attempt_token,
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
            VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, NULL, NULL, ?3, NULL, 0, ?16, ?17, '[]', NULL, NULL, NULL, NULL, NULL)
            "#,
            vec![
                id.clone().into(),
                stage.job_id.into(),
                now.into(),
                u32_to_i64_value(stage.sequence).into(),
                stage.stage_kind.into(),
                input.queue_class.as_db_value().into(),
                resource_hints_json.into(),
                resource_hints.target.as_db_value().into(),
                opt_string(
                    resource_hints
                        .backend
                        .map(|backend| backend.as_db_value().to_string()),
                ),
                opt_string(
                    resource_hints
                        .device_class
                        .map(|device| device.as_db_value().to_string()),
                ),
                opt_u64(resource_hints.min_memory_bytes),
                u32_to_i64_value(resource_hints.concurrency_weight).into(),
                stage.status.as_db_value().into(),
                opt_string(stage.capability),
                opt_string(stage.model_id),
                u32_to_i64_value(stage.max_attempts).into(),
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

    pub async fn publish_stage_output_artifact(
        &self,
        lease: &StageLease,
        input: NewStageOutputArtifact,
    ) -> anyhow::Result<Option<RuntimeArtifact>> {
        if !matches!(
            input.artifact_role,
            RuntimeArtifactRole::OutputPrimary
                | RuntimeArtifactRole::OutputIntermediate
                | RuntimeArtifactRole::Debug
        ) {
            bail!("Attempt-owned artifact publication requires an output or debug role");
        }
        let publication_key = input.publication_key.trim().to_string();
        if publication_key.is_empty() {
            bail!("Attempt-owned artifact publication requires a publication key");
        }
        let Some(attempt_token) = lease.attempt_token.as_ref() else {
            return Ok(None);
        };

        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime artifact publication transaction")?;
        let now = current_timestamp_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;
        let conflict_clause = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                "ON CONFLICT(stage_id, producer_attempt_token, publication_key) DO NOTHING"
            }
            DbBackend::MySql => "ON DUPLICATE KEY UPDATE id = id",
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let insert_sql = format!(
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                producer_attempt_count,
                producer_attempt_token,
                publication_key,
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
            SELECT
                ?1,
                s.job_id,
                s.id,
                ?2,
                ?3,
                ?4,
                ?5,
                ?6,
                ?7,
                ?8,
                ?9,
                ?10,
                ?11,
                ?12,
                ?13,
                ?14,
                ?15,
                ?16
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE s.id = ?17
              AND s.status IN ('running', 'postprocessing')
              AND s.worker_id = ?18
              AND s.attempt_count = ?2
              AND s.attempt_token = ?3
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
            {conflict_clause}
            "#
        );
        tx.execute_raw(raw::statement(
            &tx,
            insert_sql,
            vec![
                id.into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.clone().into(),
                publication_key.clone().into(),
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
                lease.stage_id.clone().into(),
                lease.worker_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to publish attempt-owned runtime artifact")?;

        let row = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT
                    a.id,
                    a.job_id,
                    a.stage_id,
                    a.producer_attempt_count,
                    a.producer_attempt_token,
                    a.publication_key,
                    a.created_at,
                    a.artifact_kind,
                    a.artifact_role,
                    a.media_asset_id,
                    a.text_asset_id,
                    a.storage_key,
                    a.content_type,
                    a.filename,
                    a.size_bytes,
                    a.sha256,
                    a.metadata_json,
                    a.retention_policy
                FROM runtime_artifacts a
                JOIN job_stages s ON s.id = a.stage_id
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE a.stage_id = ?1
                  AND a.producer_attempt_count = ?2
                  AND a.producer_attempt_token = ?3
                  AND a.publication_key = ?4
                  AND s.status IN ('running', 'postprocessing')
                  AND s.worker_id = ?5
                  AND s.attempt_count = ?2
                  AND s.attempt_token = ?3
                  AND s.lease_expires_at IS NOT NULL
                  AND s.lease_expires_at > ?6
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                LIMIT 1
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.clone().into(),
                    publication_key.into(),
                    lease.worker_id.clone().into(),
                    now.into(),
                ],
            )?)
            .await
            .context("Failed to load attempt-owned runtime artifact")?;
        let artifact = row.as_ref().map(map_runtime_artifact).transpose()?;
        tx.commit()
            .await
            .context("Failed to commit runtime artifact publication transaction")?;
        Ok(artifact)
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
        let queue_classes = update
            .queue_names
            .iter()
            .filter_map(|queue| QueueClass::from_db_value(queue))
            .collect::<Vec<_>>();
        let registration = RuntimeWorkerRegistration {
            version: WORKER_REGISTRATION_VERSION,
            worker_id: update.worker_id.clone(),
            instance_id: update.worker_id.clone(),
            queue_classes: if queue_classes.is_empty() {
                vec![QueueClass::Batch]
            } else {
                queue_classes
            },
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            resources: WorkerResourceCapacity::default(),
            software_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let diagnostic_json = update.diagnostic_json;
        let details = RuntimeWorkerHeartbeatDetails {
            version: WORKER_HEARTBEAT_DETAILS_VERSION,
            available_slots: if update.current_stage_id.is_none() {
                1
            } else {
                0
            },
            active_lease_ids: update.current_stage_id.clone().into_iter().collect(),
            last_error: None,
            health_json: diagnostic_json.clone(),
        };
        self.upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
            registration,
            status: update.status,
            current_job_id: update.current_job_id,
            current_stage_id: update.current_stage_id,
            details,
            diagnostic_json,
        })
        .await
    }

    pub async fn upsert_registered_worker_heartbeat(
        &self,
        update: RegisteredWorkerHeartbeatUpdate,
    ) -> anyhow::Result<RuntimeWorkerHeartbeat> {
        let db = self.db.connection().await?;
        let now = current_timestamp_millis();
        let queue_names = update
            .registration
            .queue_classes
            .iter()
            .map(|queue| queue.as_db_value())
            .collect::<Vec<_>>();
        let queue_names_json = json_to_db_string(&json!(queue_names), "[]")?;
        let registration_json = json_to_db_string(&json!(update.registration), "{}")?;
        let heartbeat_details_json = json_to_db_string(&json!(update.details), "{}")?;
        let diagnostic_json = json_to_db_string(&update.diagnostic_json, "{}")?;

        db.execute_raw(worker_heartbeat_upsert_statement(
            db,
            now,
            &update,
            queue_names_json,
            registration_json,
            heartbeat_details_json,
            diagnostic_json,
        )?)
        .await
        .context("Failed to upsert runtime worker heartbeat")?;

        self.get_worker_heartbeat(&update.registration.worker_id)
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

    async fn retry_stage<C: ConnectionTrait>(
        &self,
        db: &C,
        stage: &JobStage,
        lease: &StageLease,
        lease_validity: LeaseValidity,
        now: i64,
        available_at: i64,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let sql = format!(
            r#"
                UPDATE job_stages
                SET
                    status = 'retrying',
                    updated_at = ?1,
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    available_at = ?8,
                    attempt_token = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?5
                  AND attempt_count = ?6
                  AND (attempt_token = ?9 OR (attempt_token IS NULL AND ?9 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND {}
                "#,
            lease_validity.sql_predicate()
        );
        let result = db
            .execute_raw(raw::statement(
                db,
                sql,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    now.into(),
                    available_at.into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage retrying")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET status = 'retrying', updated_at = ?1, error_code = ?2, error_message = ?3
            WHERE id = ?4 AND status IN ('running', 'retrying')
            "#,
            vec![
                now.into(),
                opt_string(error_code),
                opt_string(error_message),
                stage.job_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to mark runtime job retrying")?;
        get_stage_with(db, stage.id.as_str()).await
    }

    async fn mark_stage_failed<C: ConnectionTrait>(
        &self,
        db: &C,
        stage: &JobStage,
        lease: &StageLease,
        lease_validity: LeaseValidity,
        now: i64,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let sql = format!(
            r#"
                UPDATE job_stages
                SET
                    status = 'failed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    available_at = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?5
                  AND attempt_count = ?6
                  AND (attempt_token = ?8 OR (attempt_token IS NULL AND ?8 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND {}
                "#,
            lease_validity.sql_predicate()
        );
        let result = db
            .execute_raw(raw::statement(
                db,
                sql,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    now.into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage failed")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'failed',
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                error_code = ?2,
                error_message = ?3
            WHERE id = ?4 AND status IN ('running', 'retrying', 'postprocessing')
            "#,
            vec![
                now.into(),
                opt_string(error_code),
                opt_string(error_message),
                stage.job_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to mark runtime job failed")?;
        get_stage_with(db, stage.id.as_str()).await
    }

    pub async fn reconcile_inconsistent_states(
        &self,
    ) -> anyhow::Result<RuntimeReconciliationReport> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start runtime reconciliation transaction")?;
        let now = current_timestamp_millis();
        let mut report = RuntimeReconciliationReport::default();

        for (status, stage_status, excluded_statuses) in [
            ("failed", "failed", "'failed'"),
            ("expired", "expired", "'failed', 'expired'"),
        ] {
            let result = tx
                .execute_raw(raw::statement(
                    &tx,
                    format!(
                        r#"
                        UPDATE runtime_jobs
                        SET
                            status = '{status}',
                            updated_at = ?1,
                            finished_at = COALESCE(finished_at, ?1),
                            error_code = COALESCE(error_code, 'stage_{status}'),
                            error_message = COALESCE(error_message, 'Runtime stage became {status}')
                        WHERE status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                          AND EXISTS (
                              SELECT 1 FROM job_stages
                              WHERE job_stages.job_id = runtime_jobs.id
                                AND job_stages.status = '{stage_status}'
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM job_stages
                              WHERE job_stages.job_id = runtime_jobs.id
                                AND job_stages.status IN ({excluded_statuses})
                                AND job_stages.status <> '{stage_status}'
                          )
                        "#
                    ),
                    vec![now.into()],
                )?)
                .await
                .with_context(|| format!("Failed to reconcile {status} runtime jobs"))?;
            report.jobs_repaired = report.jobs_repaired.saturating_add(result.rows_affected());
        }

        let cancelled = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'cancelled',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    cancellation_reason = COALESCE(cancellation_reason, 'All remaining stages were cancelled')
                WHERE status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                  AND EXISTS (
                      SELECT 1 FROM job_stages
                      WHERE job_stages.job_id = runtime_jobs.id AND status = 'cancelled'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM job_stages
                      WHERE job_stages.job_id = runtime_jobs.id
                        AND status NOT IN ('completed', 'skipped', 'cancelled')
                  )
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to reconcile cancelled runtime jobs")?;
        report.jobs_repaired = report
            .jobs_repaired
            .saturating_add(cancelled.rows_affected());

        let completed = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'completed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    error_code = NULL,
                    error_message = NULL
                WHERE status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  AND EXISTS (SELECT 1 FROM job_stages WHERE job_stages.job_id = runtime_jobs.id)
                  AND NOT EXISTS (
                      SELECT 1 FROM job_stages
                      WHERE job_stages.job_id = runtime_jobs.id
                        AND status NOT IN ('completed', 'skipped')
                  )
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to reconcile completed runtime jobs")?;
        report.jobs_repaired = report
            .jobs_repaired
            .saturating_add(completed.rows_affected());

        let retrying = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET status = 'retrying', updated_at = ?1
                WHERE status IN ('created', 'queued', 'running', 'postprocessing')
                  AND EXISTS (
                      SELECT 1 FROM job_stages
                      WHERE job_stages.job_id = runtime_jobs.id AND status = 'retrying'
                  )
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to reconcile retrying runtime jobs")?;
        report.jobs_repaired = report
            .jobs_repaired
            .saturating_add(retrying.rows_affected());

        let stages = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = CASE
                        WHEN (SELECT status FROM runtime_jobs WHERE id = job_stages.job_id) = 'expired' THEN 'expired'
                        ELSE 'cancelled'
                    END,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    available_at = NULL,
                    error_code = COALESCE(error_code, 'parent_terminal'),
                    error_message = COALESCE(error_message, 'Parent runtime job is terminal')
                WHERE status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('failed', 'cancelled', 'expired')
                  )
                "#,
                vec![now.into()],
            )?)
            .await
            .context("Failed to reconcile stages owned by terminal runtime jobs")?;
        report.stages_repaired = stages.rows_affected();

        tx.commit()
            .await
            .context("Failed to commit runtime reconciliation transaction")?;

        Ok(report)
    }
}

async fn get_job_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
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

async fn get_stage_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<JobStage>> {
    let row = db
        .query_one_raw(raw::statement(db, JOB_STAGE_COLUMNS_SQL, vec![id.into()])?)
        .await
        .context("Failed to load runtime job stage")?;
    row.as_ref().map(map_job_stage).transpose()
}

async fn complete_job_if_all_stages_finished_with<C: ConnectionTrait>(
    db: &C,
    job_id: &str,
    now: i64,
) -> anyhow::Result<()> {
    db.execute_raw(raw::statement(
        db,
        r#"
        UPDATE runtime_jobs
        SET
            status = 'completed',
            updated_at = ?1,
            finished_at = COALESCE(finished_at, ?1),
            error_code = NULL,
            error_message = NULL
        WHERE id = ?2
          AND status IN ('running', 'retrying', 'postprocessing', 'queued')
          AND EXISTS (SELECT 1 FROM job_stages WHERE job_id = ?2)
          AND NOT EXISTS (
              SELECT 1 FROM job_stages
              WHERE job_id = ?2 AND status NOT IN ('completed', 'skipped')
          )
        "#,
        vec![now.into(), job_id.into()],
    )?)
    .await
    .context("Failed to complete runtime job after its stages finished")?;
    Ok(())
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
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message FROM job_stages WHERE id = ?1";
const JOB_STAGE_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message FROM job_stages WHERE job_id = ?1 ORDER BY sequence ASC, created_at ASC, id ASC";
const RUNTIME_ARTIFACT_COLUMNS_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE id = ?1";
const RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE job_id = ?1 ORDER BY created_at ASC, id ASC";
const IDEMPOTENCY_RECORD_COLUMNS_SQL: &str =
    "SELECT operation, idempotency_key, created_at, expires_at, request_hash, response_json, runtime_job_id, conflict_message, metadata_json FROM idempotency_keys WHERE operation = ?1 AND idempotency_key = ?2";
const WORKER_HEARTBEAT_COLUMNS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, registration_version, registration_json, heartbeat_version, available_slots, heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats WHERE worker_id = ?1";

fn worker_heartbeat_upsert_statement(
    db: &DatabaseConnection,
    now: i64,
    update: &RegisteredWorkerHeartbeatUpdate,
    queue_names_json: String,
    registration_json: String,
    heartbeat_details_json: String,
    diagnostic_json: String,
) -> anyhow::Result<sea_orm::Statement> {
    let values = vec![
        update.registration.worker_id.clone().into(),
        now.into(),
        update.status.clone().into(),
        queue_names_json.into(),
        update.registration.instance_id.clone().into(),
        i64::from(update.registration.version).into(),
        registration_json.into(),
        i64::from(update.details.version).into(),
        i64::from(update.details.available_slots).into(),
        heartbeat_details_json.into(),
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
                instance_id,
                registration_version,
                registration_json,
                heartbeat_version,
                available_slots,
                heartbeat_details_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(worker_id) DO UPDATE SET
                started_at = CASE
                    WHEN runtime_worker_heartbeats.instance_id <> excluded.instance_id THEN excluded.started_at
                    ELSE runtime_worker_heartbeats.started_at
                END,
                last_heartbeat_at = excluded.last_heartbeat_at,
                status = excluded.status,
                queue_names_json = excluded.queue_names_json,
                instance_id = excluded.instance_id,
                registration_version = excluded.registration_version,
                registration_json = excluded.registration_json,
                heartbeat_version = excluded.heartbeat_version,
                available_slots = excluded.available_slots,
                heartbeat_details_json = excluded.heartbeat_details_json,
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
                instance_id,
                registration_version,
                registration_json,
                heartbeat_version,
                available_slots,
                heartbeat_details_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON DUPLICATE KEY UPDATE
                started_at = IF(instance_id <> VALUES(instance_id), VALUES(started_at), started_at),
                last_heartbeat_at = VALUES(last_heartbeat_at),
                status = VALUES(status),
                queue_names_json = VALUES(queue_names_json),
                instance_id = VALUES(instance_id),
                registration_version = VALUES(registration_version),
                registration_json = VALUES(registration_json),
                heartbeat_version = VALUES(heartbeat_version),
                available_slots = VALUES(available_slots),
                heartbeat_details_json = VALUES(heartbeat_details_json),
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
    let queue_class_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(8)?;

    Ok(JobStage {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(3)?)?,
        sequence: i64_to_u32(row.try_get_by_index(4)?)?,
        stage_kind: row.try_get_by_index(5)?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(7)?),
        status: RuntimeStageStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?,
        capability: row.try_get_by_index(9)?,
        model_id: row.try_get_by_index(10)?,
        worker_id: row.try_get_by_index(11)?,
        lease_expires_at: opt_i64_to_u64(row.try_get_by_index(12)?)?,
        available_at: opt_i64_to_u64(row.try_get_by_index(13)?)?,
        attempt_token: row.try_get_by_index(14)?,
        attempt_count: i64_to_u32(row.try_get_by_index(15)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(16)?)?,
        input_artifact_ids: parse_string_array(row.try_get_by_index::<String>(17)?),
        output_artifact_ids: parse_string_array(row.try_get_by_index::<String>(18)?),
        progress_json: row
            .try_get_by_index::<Option<String>>(19)?
            .map(|raw| parse_json_value(raw, json!({}))),
        started_at: opt_i64_to_u64(row.try_get_by_index(20)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(21)?)?,
        error_code: row.try_get_by_index(22)?,
        error_message: row.try_get_by_index(23)?,
    })
}

fn map_stage_claim_candidate(row: &QueryResult) -> anyhow::Result<StageClaimCandidate> {
    let job_kind_raw: String = row.try_get_by_index(2)?;
    let queue_class_raw: String = row.try_get_by_index(5)?;

    Ok(StageClaimCandidate {
        stage_id: row.try_get_by_index(0)?,
        stage_kind: row.try_get_by_index(1)?,
        job_kind: RuntimeJobKind::from_db_value(job_kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {job_kind_raw}"))?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(6)?),
        capability: row.try_get_by_index(3)?,
        model_id: row.try_get_by_index(4)?,
    })
}

fn map_runtime_artifact(row: &QueryResult) -> anyhow::Result<RuntimeArtifact> {
    let kind_raw: String = row.try_get_by_index(7)?;
    let role_raw: String = row.try_get_by_index(8)?;

    Ok(RuntimeArtifact {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        stage_id: row.try_get_by_index(2)?,
        producer_attempt_count: opt_i64_to_u32(row.try_get_by_index(3)?)?,
        producer_attempt_token: row.try_get_by_index(4)?,
        publication_key: row.try_get_by_index(5)?,
        created_at: i64_to_u64(row.try_get_by_index(6)?)?,
        artifact_kind: RuntimeArtifactKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact kind: {kind_raw}"))?,
        artifact_role: RuntimeArtifactRole::from_db_value(role_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact role: {role_raw}"))?,
        media_asset_id: row.try_get_by_index(9)?,
        text_asset_id: row.try_get_by_index(10)?,
        storage_key: row.try_get_by_index(11)?,
        content_type: row.try_get_by_index(12)?,
        filename: row.try_get_by_index(13)?,
        size_bytes: opt_i64_to_u64(row.try_get_by_index(14)?)?,
        sha256: row.try_get_by_index(15)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        retention_policy: row.try_get_by_index(17)?,
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
    let worker_id: String = row.try_get_by_index(0)?;
    let queue_names = parse_string_array(row.try_get_by_index::<String>(4)?);
    let stored_instance_id: String = row.try_get_by_index(5)?;
    let instance_id = if stored_instance_id.is_empty() {
        worker_id.clone()
    } else {
        stored_instance_id
    };
    let registration_version = i64_to_u32(row.try_get_by_index(6)?)? as u16;
    let registration =
        serde_json::from_str::<RuntimeWorkerRegistration>(&row.try_get_by_index::<String>(7)?)
            .unwrap_or_else(|_| RuntimeWorkerRegistration {
                version: registration_version,
                worker_id: worker_id.clone(),
                instance_id: instance_id.clone(),
                queue_classes: queue_names
                    .iter()
                    .filter_map(|queue| QueueClass::from_db_value(queue))
                    .collect(),
                capabilities: Vec::new(),
                model_ids: Vec::new(),
                stage_kinds: Vec::new(),
                resources: WorkerResourceCapacity::default(),
                software_version: "legacy".to_string(),
            });
    let heartbeat_version = i64_to_u32(row.try_get_by_index(8)?)? as u16;
    let available_slots = i64_to_u32(row.try_get_by_index(9)?)?;
    let details =
        serde_json::from_str::<RuntimeWorkerHeartbeatDetails>(&row.try_get_by_index::<String>(10)?)
            .unwrap_or_else(|_| RuntimeWorkerHeartbeatDetails {
                version: heartbeat_version,
                available_slots,
                active_lease_ids: Vec::new(),
                last_error: None,
                health_json: json!({}),
            });

    Ok(RuntimeWorkerHeartbeat {
        worker_id,
        started_at: i64_to_u64(row.try_get_by_index(1)?)?,
        last_heartbeat_at: i64_to_u64(row.try_get_by_index(2)?)?,
        status: row.try_get_by_index(3)?,
        queue_names,
        instance_id,
        registration,
        details,
        current_job_id: row.try_get_by_index(11)?,
        current_stage_id: row.try_get_by_index(12)?,
        diagnostic_json: parse_json_value(row.try_get_by_index::<String>(13)?, json!({})),
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

fn parse_resource_hints(raw: String) -> StageResourceHints {
    serde_json::from_str::<StageResourceHints>(&raw).unwrap_or_default()
}

fn queue_class_for_stage_kind(stage_kind: &str) -> QueueClass {
    match stage_kind {
        "asr_transcribe" | "asr_infer" => QueueClass::BatchAsr,
        "tts_synthesize" | "tts_generate" => QueueClass::BatchTts,
        "diarization" | "diarization_segment" => QueueClass::Diarization,
        "export" | "encode" | "notify" => QueueClass::Export,
        "evaluation" | "evaluate" => QueueClass::Evaluation,
        _ => QueueClass::Batch,
    }
}

fn normalize_filter_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn push_claim_queue_clause(sql: &mut String, params: &mut Vec<Value>, queue_names: &[String]) {
    if queue_names.is_empty()
        || queue_names
            .iter()
            .any(|queue| queue == QueueClass::Batch.as_db_value())
    {
        return;
    }

    sql.push_str(" AND ");
    push_claim_in_expression(sql, params, "s.queue_class", queue_names);
}

fn push_claim_resource_clause(
    sql: &mut String,
    params: &mut Vec<Value>,
    resources: &WorkerResourceCapacity,
) {
    let targets = resources
        .targets
        .iter()
        .map(|target| target.as_db_value().to_string())
        .collect::<Vec<_>>();
    sql.push_str(" AND (s.resource_target = 'any'");
    if !targets.is_empty() {
        sql.push_str(" OR ");
        push_claim_in_expression(sql, params, "s.resource_target", &targets);
    }
    sql.push(')');

    let backends = resources
        .backends
        .iter()
        .map(|backend| backend.as_db_value().to_string())
        .collect::<Vec<_>>();
    push_optional_resource_requirement(sql, params, "s.required_backend", &backends);
    let device_classes = resources
        .device_classes
        .iter()
        .map(|device| device.as_db_value().to_string())
        .collect::<Vec<_>>();
    push_optional_resource_requirement(sql, params, "s.required_device_class", &device_classes);

    match resources.memory_bytes {
        Some(memory_bytes) => {
            let placeholder = params.len() + 1;
            sql.push_str(
                " AND (s.min_resource_memory_bytes IS NULL OR s.min_resource_memory_bytes <= ?",
            );
            sql.push_str(&placeholder.to_string());
            sql.push(')');
            params.push(u64_to_i64_value(memory_bytes).unwrap_or(Value::BigInt(Some(i64::MAX))));
        }
        None => sql.push_str(" AND s.min_resource_memory_bytes IS NULL"),
    }

    let placeholder = params.len() + 1;
    sql.push_str(" AND s.resource_concurrency_weight <= ?");
    sql.push_str(&placeholder.to_string());
    params.push(u32_to_i64_value(resources.concurrency_slots).into());
}

fn push_optional_resource_requirement(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    sql.push_str(" AND (");
    sql.push_str(expression);
    sql.push_str(" IS NULL");
    if !values.is_empty() {
        sql.push_str(" OR ");
        push_claim_in_expression(sql, params, expression, values);
    }
    sql.push(')');
}

fn push_claim_string_filter_clause(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    if values.is_empty() {
        return;
    }

    sql.push_str(" AND ");
    push_claim_in_expression(sql, params, expression, values);
}

fn push_claim_in_expression(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    sql.push_str(expression);
    sql.push_str(" IN (");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            sql.push_str(", ");
        }
        let placeholder = params.len() + 1;
        sql.push('?');
        sql.push_str(placeholder.to_string().as_str());
        params.push(value.clone().into());
    }
    sql.push(')');
}

fn optional_filter_matches(filter: &[String], value: Option<&str>) -> bool {
    filter.is_empty()
        || value.is_some_and(|value| filter.iter().any(|entry| entry.as_str() == value))
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

fn is_claimable_job_status(status: RuntimeJobStatus) -> bool {
    matches!(
        status,
        RuntimeJobStatus::Created
            | RuntimeJobStatus::Queued
            | RuntimeJobStatus::Running
            | RuntimeJobStatus::Retrying
            | RuntimeJobStatus::Postprocessing
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

    async fn create_test_job_and_stage(
        store: &BatchRuntimeStore,
        priority: i32,
        stage_kind: &str,
        max_attempts: u32,
    ) -> (RuntimeJob, JobStage) {
        let job = create_test_job(store, priority, max_attempts).await;
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: stage_kind.to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("test".to_string()),
                model_id: None,
                max_attempts,
                input_artifact_ids: vec![],
            })
            .await
            .expect("test stage");
        (job, stage)
    }

    async fn create_test_job(
        store: &BatchRuntimeStore,
        priority: i32,
        max_attempts: u32,
    ) -> RuntimeJob {
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority,
                model_id: None,
                capability: Some("test".to_string()),
                route_record_kind: Some("test".to_string()),
                route_record_id: None,
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": max_attempts}),
                max_attempts,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("test job");
        job
    }

    fn test_stage_output(publication_key: &str) -> NewStageOutputArtifact {
        NewStageOutputArtifact {
            publication_key: publication_key.to_string(),
            artifact_kind: RuntimeArtifactKind::Metadata,
            artifact_role: RuntimeArtifactRole::OutputPrimary,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: Some(format!("outputs/{publication_key}.json")),
            content_type: Some("application/json".to_string()),
            filename: Some(format!("{publication_key}.json")),
            size_bytes: Some(2),
            sha256: Some(sha256_hex(b"{}")),
            metadata_json: json!({"publication_key": publication_key}),
            retention_policy: "default".to_string(),
        }
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
        assert_eq!(artifact.producer_attempt_count, None);
        assert_eq!(artifact.producer_attempt_token, None);
        assert_eq!(artifact.publication_key, None);
        assert_eq!(idempotency.runtime_job_id.as_deref(), Some(job.id.as_str()));
        assert_eq!(heartbeat.queue_names, vec!["batch"]);
        assert_eq!(heartbeat.instance_id, "worker-1");
        assert_eq!(
            heartbeat.registration.queue_classes,
            vec![QueueClass::Batch]
        );
    }

    #[tokio::test]
    async fn registered_worker_heartbeat_tracks_instance_resources_and_capacity() {
        let (store, _root) = build_store();
        let registration = RuntimeWorkerRegistration {
            version: WORKER_REGISTRATION_VERSION,
            worker_id: "worker-logical".to_string(),
            instance_id: "instance-a".to_string(),
            queue_classes: vec![QueueClass::BatchAsr, QueueClass::Evaluation],
            capabilities: vec!["asr".to_string()],
            model_ids: vec!["model-a".to_string()],
            stage_kinds: vec!["asr_transcribe".to_string()],
            resources: WorkerResourceCapacity {
                targets: vec![ResourceTarget::Gpu],
                memory_bytes: Some(24 * 1024 * 1024 * 1024),
                concurrency_slots: 2,
                ..WorkerResourceCapacity::default()
            },
            software_version: "test-version".to_string(),
        };
        let details = RuntimeWorkerHeartbeatDetails {
            version: WORKER_HEARTBEAT_DETAILS_VERSION,
            available_slots: 1,
            active_lease_ids: vec!["stage-a".to_string()],
            last_error: None,
            health_json: json!({"temperature_c": 60}),
        };

        let heartbeat = store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration: registration.clone(),
                status: "running".to_string(),
                current_job_id: None,
                current_stage_id: None,
                details: details.clone(),
                diagnostic_json: json!({"source": "test"}),
            })
            .await
            .expect("registered heartbeat");
        assert_eq!(heartbeat.instance_id, "instance-a");
        assert_eq!(heartbeat.registration, registration);
        assert_eq!(heartbeat.details, details);
        assert_eq!(
            heartbeat.queue_names,
            vec!["batch_asr".to_string(), "evaluation".to_string()]
        );

        let replacement = RuntimeWorkerRegistration {
            instance_id: "instance-b".to_string(),
            ..heartbeat.registration
        };
        let replaced = store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration: replacement,
                status: "idle".to_string(),
                current_job_id: None,
                current_stage_id: None,
                details: RuntimeWorkerHeartbeatDetails {
                    available_slots: 2,
                    active_lease_ids: vec![],
                    ..heartbeat.details
                },
                diagnostic_json: json!({"source": "replacement"}),
            })
            .await
            .expect("replacement heartbeat");
        assert_eq!(replaced.instance_id, "instance-b");
        assert_eq!(replaced.status, "idle");
        assert_eq!(replaced.details.available_slots, 2);
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
    async fn filtered_stage_claim_skips_incompatible_higher_priority_stage() {
        let (store, _root) = build_store();
        let tts_job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 50,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("speech_history".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("tts job");
        let tts_stage = store
            .create_stage(NewJobStage {
                job_id: tts_job.id.clone(),
                sequence: 0,
                stage_kind: "tts_generate".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: tts_job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("tts stage");

        let asr_job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 10,
                model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("transcription-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("asr job");
        let asr_stage = store
            .create_stage(NewJobStage {
                job_id: asr_job.id.clone(),
                sequence: 0,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: asr_job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("asr stage");

        let mut filter = StageClaimFilter::for_worker_queues(&["batch_asr".to_string()]);
        filter.capabilities = vec!["asr".to_string()];

        let claimed = store
            .claim_next_stage_with_filter("asr-worker", 60_000, &filter)
            .await
            .expect("claim")
            .expect("asr stage should be claimed");

        assert_eq!(claimed.stage.id, asr_stage.id);
        assert_eq!(claimed.stage.queue_class, QueueClass::BatchAsr);
        assert_eq!(claimed.stage.capability.as_deref(), Some("asr"));
        assert_eq!(claimed.stage.worker_id.as_deref(), Some("asr-worker"));

        let tts_stage = store
            .get_stage(&tts_stage.id)
            .await
            .expect("fetch tts stage")
            .expect("tts stage exists");
        assert_eq!(tts_stage.status, RuntimeStageStatus::Queued);
        assert_eq!(tts_stage.worker_id, None);

        let wildcard_claim = store
            .claim_next_stage("general-batch-worker", 60_000)
            .await
            .expect("wildcard claim")
            .expect("legacy batch wildcard should claim remaining TTS stage");
        assert_eq!(wildcard_claim.stage.id, tts_stage.id);
        assert_eq!(wildcard_claim.stage.queue_class, QueueClass::BatchTts);
    }

    #[tokio::test]
    async fn resource_aware_claim_rejects_backend_device_and_capacity_mismatch() {
        let (store, _root) = build_store();
        let gpu_job = create_test_job(&store, 50, 1).await;
        let gpu_stage = store
            .create_stage_with_dispatch(NewJobStageDispatch {
                stage: NewJobStage {
                    job_id: gpu_job.id,
                    sequence: 0,
                    stage_kind: "gpu_evaluation".to_string(),
                    status: RuntimeStageStatus::Queued,
                    capability: Some("test".to_string()),
                    model_id: None,
                    max_attempts: 1,
                    input_artifact_ids: vec![],
                },
                queue_class: QueueClass::Evaluation,
                resource_hints: StageResourceHints {
                    target: ResourceTarget::Gpu,
                    backend: Some(RuntimeBackendClass::Metal),
                    device_class: Some(DeviceClass::AppleGpu),
                    min_memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    concurrency_weight: 2,
                    ..StageResourceHints::default()
                },
            })
            .await
            .expect("GPU stage");
        let cpu_job = create_test_job(&store, 10, 1).await;
        let cpu_stage = store
            .create_stage_with_dispatch(NewJobStageDispatch {
                stage: NewJobStage {
                    job_id: cpu_job.id,
                    sequence: 0,
                    stage_kind: "cpu_evaluation".to_string(),
                    status: RuntimeStageStatus::Queued,
                    capability: Some("test".to_string()),
                    model_id: None,
                    max_attempts: 1,
                    input_artifact_ids: vec![],
                },
                queue_class: QueueClass::Evaluation,
                resource_hints: StageResourceHints {
                    target: ResourceTarget::Gpu,
                    backend: Some(RuntimeBackendClass::Cuda),
                    device_class: Some(DeviceClass::NvidiaGpu),
                    min_memory_bytes: Some(2 * 1024 * 1024 * 1024),
                    ..StageResourceHints::default()
                },
            })
            .await
            .expect("CPU stage");

        let mut filter = StageClaimFilter::for_worker_queues(&["evaluation".to_string()]);
        filter.resources = WorkerResourceCapacity {
            targets: vec![ResourceTarget::Gpu],
            backends: vec![RuntimeBackendClass::Cuda],
            device_classes: vec![DeviceClass::NvidiaGpu],
            memory_bytes: Some(8 * 1024 * 1024 * 1024),
            concurrency_slots: 1,
            ..WorkerResourceCapacity::default()
        };
        let claimed = store
            .claim_next_stage_with_filter("cpu-worker", 60_000, &filter)
            .await
            .expect("resource-aware claim")
            .expect("compatible CUDA stage");
        assert_eq!(claimed.stage.id, cpu_stage.id);
        assert_eq!(claimed.stage.queue_class, QueueClass::Evaluation);
        assert_eq!(
            claimed.stage.resource_hints.backend,
            Some(RuntimeBackendClass::Cuda)
        );
        assert_eq!(
            claimed.stage.resource_hints.device_class,
            Some(DeviceClass::NvidiaGpu)
        );

        let gpu_stage = store
            .get_stage(&gpu_stage.id)
            .await
            .expect("GPU stage fetch")
            .expect("GPU stage exists");
        assert_eq!(gpu_stage.status, RuntimeStageStatus::Queued);
        assert_eq!(gpu_stage.worker_id, None);
    }

    #[tokio::test]
    async fn concurrent_claimers_take_distinct_candidates() {
        let (store, _root) = build_store();
        create_test_job_and_stage(&store, 20, "fake_stage", 1).await;
        create_test_job_and_stage(&store, 10, "fake_stage", 1).await;

        let first_store = store.clone();
        let second_store = store.clone();
        let (first, second) = tokio::join!(
            first_store.claim_next_stage("worker-1", 60_000),
            second_store.claim_next_stage("worker-2", 60_000),
        );
        let first = first.expect("first claim").expect("first candidate");
        let second = second.expect("second claim").expect("second candidate");

        assert_ne!(first.stage.id, second.stage.id);
        assert_ne!(first.stage.worker_id, second.stage.worker_id);
    }

    #[tokio::test]
    async fn claim_waits_for_predecessor_completion() {
        let (store, _root) = build_store();
        let (job, first_stage) = create_test_job_and_stage(&store, 0, "first_stage", 1).await;
        let second_stage = store
            .create_stage(NewJobStage {
                job_id: job.id,
                sequence: 1,
                stage_kind: "second_stage".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("test".to_string()),
                model_id: None,
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("second stage");

        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first stage should be claimable");
        assert_eq!(first.stage.id, first_stage.id);
        assert!(store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("blocked claim")
            .is_none());

        store
            .complete_stage(&first.lease().expect("first lease"), vec![])
            .await
            .expect("first completion")
            .expect("owned first completion");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second stage should become claimable");
        assert_eq!(second.stage.id, second_stage.id);
    }

    #[tokio::test]
    async fn claim_cas_rechecks_parent_job_eligibility() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Queued],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("cancel before claim CAS".to_string()),
            )
            .await
            .expect("cancel job")
            .expect("job transition");

        let now = current_timestamp_millis();
        let claimed = store
            .try_claim_stage_candidate(
                store.connection().await.expect("database"),
                StageClaimCandidate {
                    stage_id: stage.id.clone(),
                    stage_kind: stage.stage_kind.clone(),
                    job_kind: job.job_kind,
                    queue_class: stage.queue_class,
                    resource_hints: stage.resource_hints.clone(),
                    capability: stage.capability.clone(),
                    model_id: stage.model_id.clone(),
                },
                "worker-1",
                now,
                now + 60_000,
            )
            .await
            .expect("claim CAS");

        assert!(claimed.is_none());
        let stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Queued);
        assert_eq!(stage.worker_id, None);
    }

    #[tokio::test]
    async fn terminal_parent_fences_late_stage_completion() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Running],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("cancelled outside stage transaction".to_string()),
            )
            .await
            .expect("cancel parent")
            .expect("parent transition");

        assert!(store
            .complete_stage(
                &claimed.lease().expect("lease"),
                vec!["late-output".to_string()],
            )
            .await
            .expect("late completion")
            .is_none());
        let running_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running_stage.status, RuntimeStageStatus::Running);
        assert!(running_stage.output_artifact_ids.is_empty());
    }

    #[tokio::test]
    async fn stale_owner_cannot_finish_a_reclaimed_attempt() {
        let (store, _root) = build_store();
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
        let first = store
            .claim_next_stage("worker-1", 0)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_lease = first.lease().expect("first lease");
        assert_eq!(
            store.recover_expired_stage_leases().await.expect("recover"),
            1
        );

        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second attempt");
        let second_lease = second.lease().expect("second lease");
        assert_eq!(second_lease.attempt_count, first_lease.attempt_count + 1);

        assert!(store
            .complete_stage(&first_lease, vec!["stale-output".to_string()])
            .await
            .expect("stale completion")
            .is_none());
        assert!(store
            .fail_stage(
                &first_lease,
                false,
                Some("stale".to_string()),
                Some("stale owner".to_string()),
            )
            .await
            .expect("stale failure")
            .is_none());

        let running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert_eq!(running.worker_id.as_deref(), Some("worker-2"));
        assert_eq!(running.attempt_count, second_lease.attempt_count);

        let completed = store
            .complete_stage(&second_lease, vec!["current-output".to_string()])
            .await
            .expect("current completion")
            .expect("current owner completes");
        assert_eq!(completed.output_artifact_ids, vec!["current-output"]);
    }

    #[tokio::test]
    async fn cancellation_invalidates_active_stage_lease() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("active attempt");
        let lease = claimed.lease().expect("lease");

        store
            .cancel_job(&job.id, Some("user cancelled".to_string()))
            .await
            .expect("cancel")
            .expect("cancelled job");

        assert!(store
            .complete_stage(&lease, vec!["late-output".to_string()])
            .await
            .expect("late completion")
            .is_none());
        assert!(store
            .fail_stage(
                &lease,
                false,
                Some("late".to_string()),
                Some("late failure".to_string()),
            )
            .await
            .expect("late failure")
            .is_none());
        let cancelled = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
        assert_eq!(cancelled.worker_id, None);
        assert_eq!(cancelled.lease_expires_at, None);
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
                max_attempts: 2,
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
        let lease = claimed.lease().expect("lease");

        let failed_stage = store
            .fail_stage(
                &lease,
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

        let second_claim = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("retried stage should be claimable");
        store
            .fail_stage(
                &second_claim.lease().expect("second lease"),
                false,
                Some("boom-again".to_string()),
                Some("second attempt failed".to_string()),
            )
            .await
            .expect("second failure")
            .expect("stage should fail again");
        assert!(store
            .retry_job(&job.id)
            .await
            .expect("retry budget check")
            .is_none());
    }

    #[tokio::test]
    async fn retry_policy_delays_claims_and_enforces_attempt_budget() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE runtime_jobs SET retry_policy_json = ?1 WHERE id = ?2",
                vec![
                    json!({
                        "max_attempts": 2,
                        "initial_backoff_ms": 1_000,
                        "backoff_multiplier": 2.0,
                        "max_backoff_ms": 10_000
                    })
                    .to_string()
                    .into(),
                    job.id.clone().into(),
                ],
            )
            .expect("retry policy statement"),
        )
        .await
        .expect("retry policy update");

        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_token = first.stage.attempt_token.clone();
        let retrying = store
            .fail_stage(
                &first.lease().expect("first lease"),
                true,
                Some("transient".to_string()),
                Some("try again".to_string()),
            )
            .await
            .expect("retry transition")
            .expect("stage should retry");
        assert_eq!(retrying.status, RuntimeStageStatus::Retrying);
        assert!(retrying.available_at.expect("retry eligibility") > retrying.updated_at);
        assert!(store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("delayed claim")
            .is_none());

        db.execute_raw(
            raw::statement(
                db,
                "UPDATE job_stages SET available_at = ?1 WHERE id = ?2",
                vec![
                    current_timestamp_millis().saturating_sub(1).into(),
                    stage.id.clone().into(),
                ],
            )
            .expect("release retry statement"),
        )
        .await
        .expect("release retry");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second attempt");
        assert_ne!(second.stage.attempt_token, first_token);

        let failed = store
            .fail_stage(
                &second.lease().expect("second lease"),
                true,
                Some("transient".to_string()),
                Some("budget exhausted".to_string()),
            )
            .await
            .expect("terminal failure")
            .expect("stage should fail");
        assert_eq!(failed.status, RuntimeStageStatus::Failed);
        assert_eq!(failed.attempt_count, 2);
        assert_eq!(
            store
                .get_job(&job.id)
                .await
                .expect("job")
                .expect("job exists")
                .status,
            RuntimeJobStatus::Failed
        );
    }

    #[tokio::test]
    async fn attempt_token_fences_same_worker_and_attempt_publication() {
        let (store, _root) = build_store();
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let lease = claimed.lease().expect("lease");
        assert!(lease.attempt_token.is_some());

        let forged = StageLease {
            attempt_token: Some("wrong-attempt-token".to_string()),
            ..lease.clone()
        };
        assert!(store
            .complete_stage(&forged, vec!["stale-output".to_string()])
            .await
            .expect("forged completion")
            .is_none());
        let running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert!(running.output_artifact_ids.is_empty());

        assert!(store
            .complete_stage(&lease, vec!["current-output".to_string()])
            .await
            .expect("owned completion")
            .is_some());
    }

    #[tokio::test]
    async fn attempt_owned_artifact_publication_is_idempotent() {
        let (store, _root) = build_store();
        let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let lease = claimed.lease().expect("lease");
        assert!(store
            .stage_lease_is_active(&lease)
            .await
            .expect("active lease"));

        let first = store
            .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
            .await
            .expect("first publication")
            .expect("active publication");
        let duplicate = store
            .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
            .await
            .expect("duplicate publication")
            .expect("idempotent publication");

        assert_eq!(duplicate.id, first.id);
        assert_eq!(first.job_id, job.id);
        assert_eq!(first.stage_id.as_deref(), Some(lease.stage_id.as_str()));
        assert_eq!(first.producer_attempt_count, Some(lease.attempt_count));
        assert_eq!(
            first.producer_attempt_token.as_deref(),
            lease.attempt_token.as_deref()
        );
        assert_eq!(first.publication_key.as_deref(), Some("primary-result"));
        let artifacts = store
            .list_artifacts_for_job(&job.id)
            .await
            .expect("artifacts");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, first.id);
    }

    #[tokio::test]
    async fn stale_and_cancelled_attempts_cannot_publish_artifacts() {
        let (store, _root) = build_store();
        let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_lease = first.lease().expect("first lease");
        assert!(store
            .stage_lease_is_active(&first_lease)
            .await
            .expect("first lease active"));

        store
            .fail_stage(
                &first_lease,
                true,
                Some("retry".to_string()),
                Some("replace attempt".to_string()),
            )
            .await
            .expect("retry first attempt")
            .expect("retrying stage");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("replacement attempt");
        let second_lease = second.lease().expect("second lease");

        assert!(!store
            .stage_lease_is_active(&first_lease)
            .await
            .expect("stale lease check"));
        assert!(store
            .stage_lease_is_active(&second_lease)
            .await
            .expect("replacement lease check"));
        assert!(store
            .publish_stage_output_artifact(&first_lease, test_stage_output("stale-result"))
            .await
            .expect("stale publication")
            .is_none());

        store
            .cancel_job(&job.id, Some("cancel active attempt".to_string()))
            .await
            .expect("cancel job")
            .expect("cancelled job");
        assert!(!store
            .stage_lease_is_active(&second_lease)
            .await
            .expect("cancelled lease check"));
        assert!(store
            .publish_stage_output_artifact(&second_lease, test_stage_output("cancelled-result"))
            .await
            .expect("cancelled publication")
            .is_none());
        assert!(store
            .list_artifacts_for_job(&job.id)
            .await
            .expect("artifacts")
            .is_empty());
    }

    #[tokio::test]
    async fn reconciliation_repairs_crash_interrupted_terminal_transitions() {
        let (store, _root) = build_store();
        let (completed_job, completed_stage) =
            create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-complete", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE job_stages SET status = 'completed', worker_id = NULL, lease_expires_at = NULL, finished_at = ?1 WHERE id = ?2",
                vec![current_timestamp_millis().into(), claimed.stage.id.into()],
            )
            .expect("crash completion statement"),
        )
        .await
        .expect("crash completion");

        let (cancelled_job, cancelled_stage) =
            create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE runtime_jobs SET status = 'cancelled', finished_at = ?1 WHERE id = ?2",
                vec![
                    current_timestamp_millis().into(),
                    cancelled_job.id.clone().into(),
                ],
            )
            .expect("crash cancellation statement"),
        )
        .await
        .expect("crash cancellation");

        let report = store
            .reconcile_inconsistent_states()
            .await
            .expect("reconciliation");
        assert!(report.jobs_repaired >= 1);
        assert!(report.stages_repaired >= 1);
        assert_eq!(
            store
                .get_job(&completed_job.id)
                .await
                .expect("completed job")
                .expect("completed job exists")
                .status,
            RuntimeJobStatus::Completed
        );
        assert_eq!(
            store
                .get_stage(&completed_stage.id)
                .await
                .expect("completed stage")
                .expect("completed stage exists")
                .status,
            RuntimeStageStatus::Completed
        );
        let cancelled = store
            .get_stage(&cancelled_stage.id)
            .await
            .expect("cancelled stage")
            .expect("cancelled stage exists");
        assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
        assert!(cancelled.worker_id.is_none());
        assert!(cancelled.lease_expires_at.is_none());
    }
}

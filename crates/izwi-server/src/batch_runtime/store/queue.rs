//! Job stages: claims, leases, progress, completion, failure and cancellation.

use super::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageLeaseState {
    Active,
    CancellationRequested,
    ExecutionStopping,
}

pub(super) const DEFAULT_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 64;

pub(super) const MAX_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 512;

/// How long a cancelling attempt must go without lease renewal, and its owner
/// without a heartbeat, before recovery treats the owning process as gone. A
/// live owner renews a cancelling lease every lease/3 (at most 30 s) and
/// heartbeats at least as often, so this is several missed renewals.
pub(super) const ABANDONED_CANCELLATION_GRACE_MS: i64 = 2 * 60 * 1000;

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
        let mut queue_names = normalize_filter_values(queue_names);
        if queue_names.is_empty() {
            queue_names.push("batch".to_string());
        }
        Self {
            queue_names,
            ..Default::default()
        }
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(super) struct StoredRetryPolicy {
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
pub(super) enum LeaseValidity {
    Expired,
    Any,
}

impl LeaseValidity {
    fn sql_predicate(self) -> &'static str {
        match self {
            Self::Expired => "lease_expires_at <= ?1",
            // Owner-fenced relinquish must not depend on wall-clock expiry:
            // attempt identity still protects against a reclaimed lease.
            Self::Any => "1 = 1",
        }
    }
}

/// Who may settle a cancelling attempt as `cancelled`.
#[derive(Debug, Clone, Copy)]
pub(super) enum CancellationSettlement {
    /// The exact owner observed its executor resolve.
    OwnerTeardown,
    /// Recovery: the owner stopped renewing and heartbeating before
    /// `stale_before`, so its process (and executor) is gone.
    AbandonedOwner { stale_before: i64 },
}

impl CancellationSettlement {
    /// The value the predicate reads at its placeholder, when it reads one.
    fn bound_value(self) -> Option<Value> {
        match self {
            Self::OwnerTeardown => None,
            Self::AbandonedOwner { stale_before } => Some(stale_before.into()),
        }
    }

    fn sql_predicate(self, stale_before_param: usize) -> String {
        match self {
            Self::OwnerTeardown => "1 = 1".to_string(),
            Self::AbandonedOwner { .. } => abandoned_cancellation_predicate(stale_before_param),
        }
    }
}

/// The attempt's lease lapsed before the cutoff and no incarnation that could
/// own it has heartbeated since. A heartbeat row whose incarnation started
/// after the stage's last owner write belongs to a restarted process reusing
/// the worker id, not to the attempt's owner.
pub(super) fn abandoned_cancellation_predicate(stale_before_param: usize) -> String {
    format!(
        r#"job_stages.lease_expires_at <= ?{stale_before_param}
           AND NOT EXISTS (
               SELECT 1 FROM runtime_worker_heartbeats
               WHERE runtime_worker_heartbeats.worker_id = job_stages.worker_id
                 AND runtime_worker_heartbeats.last_heartbeat_at > ?{stale_before_param}
                 AND runtime_worker_heartbeats.started_at <= job_stages.updated_at
           )"#
    )
}

/// Row-lock a stage's parent job on PostgreSQL. Every transaction that
/// writes a stage and then its job takes the job lock first, the order
/// cancellation and manual retry already use, so a stage writer and a job
/// writer cannot deadlock each other. SQLite's single writer lock already
/// serializes them.
pub(super) async fn lock_stage_job_with<C: ConnectionTrait>(
    db: &C,
    stage_id: &str,
) -> anyhow::Result<()> {
    if db.get_database_backend() != DbBackend::Postgres {
        return Ok(());
    }
    db.query_one_raw(raw::statement(
        db,
        r#"
        SELECT j.id
        FROM runtime_jobs j
        JOIN job_stages s ON s.job_id = j.id
        WHERE s.id = ?1
        FOR UPDATE OF j
        "#,
        vec![stage_id.into()],
    )?)
    .await
    .context("Failed to lock the stage's runtime job")?;
    Ok(())
}

pub(super) async fn get_stage_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<JobStage>> {
    let row = db
        .query_one_raw(raw::statement(db, JOB_STAGE_BY_ID_SQL, vec![id.into()])?)
        .await
        .context("Failed to load runtime job stage")?;
    row.as_ref().map(map_job_stage).transpose()
}

pub(super) async fn complete_job_if_all_stages_finished_with<C: ConnectionTrait>(
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
          AND status IN ('running', 'retrying', 'queued')
          AND cancellation_state IS NULL
          AND EXISTS (SELECT 1 FROM job_stages WHERE job_id = ?2)
          AND NOT EXISTS (
              SELECT 1 FROM job_stages
              WHERE job_id = ?2 AND status <> 'completed'
          )
        "#,
        vec![now.into(), job_id.into()],
    )?)
    .await
    .context("Failed to complete runtime job after its stages finished")?;
    Ok(())
}

pub(super) fn queue_class_for_stage_kind(stage_kind: &str) -> QueueClass {
    match stage_kind {
        "asr_transcribe" | "asr_infer" => QueueClass::BatchAsr,
        "tts_synthesize" | "tts_generate" => QueueClass::BatchTts,
        "diarization" | "diarization_segment" => QueueClass::Diarization,
        "export" | "encode" | "notify" => QueueClass::Export,
        "evaluation" | "evaluate" => QueueClass::Evaluation,
        _ => QueueClass::Batch,
    }
}

pub(super) fn normalize_filter_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

pub(super) fn push_claim_queue_clause(sql: &mut String, params: &mut Vec<Value>, queue_names: &[String]) {
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

pub(super) fn push_claim_resource_clause(
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

    let placeholder = params.len() + 1;
    sql.push_str(" AND s.resource_hints_version = ?");
    sql.push_str(&placeholder.to_string());
    params.push(i64::from(STAGE_RESOURCE_HINTS_VERSION).into());
}

pub(super) fn push_optional_resource_requirement(
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

pub(super) fn push_claim_string_filter_clause(
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

pub(super) fn push_claim_in_expression(
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

pub(super) fn optional_filter_matches(filter: &[String], value: Option<&str>) -> bool {
    filter.is_empty()
        || value.is_some_and(|value| filter.iter().any(|entry| entry.as_str() == value))
}

impl BatchRuntimeStore {
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
        let now = self.now_millis();
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
        let claimable = format!(
            r#"
                FROM job_stages s
                INNER JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.status IN ('queued', 'retrying')
                  AND (s.available_at IS NULL OR s.available_at <= ?1)
                  AND (s.lease_expires_at IS NULL OR s.lease_expires_at <= ?1)
                  AND j.status IN ('queued', 'running', 'retrying')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM job_stages predecessor
                      WHERE predecessor.job_id = s.job_id
                        AND predecessor.sequence < s.sequence
                        AND predecessor.status <> 'completed'
                  )
                  {claim_filter_sql}
                ORDER BY j.priority DESC, s.sequence ASC, COALESCE(s.available_at, s.created_at) ASC,
                         CASE WHEN s.started_at IS NULL THEN 0 ELSE 1 END ASC, s.id ASC
            "#
        );
        if db.get_database_backend() == DbBackend::Postgres {
            return self
                .claim_skipping_locked_stages(db, &claimable, params, worker_id, lease_expires_at)
                .await;
        }

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
                    COALESCE(s.capability, j.capability) AS capability,
                    COALESCE(s.model_id, j.model_id) AS model_id,
                    s.queue_class,
                    s.resource_hints_json
                {claimable}
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

    /// PostgreSQL claims the first claimable stage in one statement. The
    /// candidate subquery locks the stage and its job and skips rows another
    /// transaction holds, so concurrent workers take distinct stages instead
    /// of queueing on the head row, and a job row locked by a cancellation is
    /// passed over rather than waited for. Every filter is in the SQL, so the
    /// first unlocked row is the one to take.
    async fn claim_skipping_locked_stages(
        &self,
        db: &DatabaseConnection,
        claimable: &str,
        mut params: Vec<Value>,
        worker_id: &str,
        lease_expires_at: i64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime stage claim transaction")?;
        let attempt_token = new_uuid();
        let worker_placeholder = params.len() + 1;
        let lease_placeholder = params.len() + 2;
        let token_placeholder = params.len() + 3;
        params.extend([
            worker_id.into(),
            lease_expires_at.into(),
            attempt_token.clone().into(),
        ]);
        let claimed = tx
            .query_one_raw(raw::statement(
                &tx,
                format!(
                    r#"
                    UPDATE job_stages
                    SET
                        status = 'running',
                        worker_id = ?{worker_placeholder},
                        lease_expires_at = ?{lease_placeholder},
                        available_at = NULL,
                        attempt_token = ?{token_placeholder},
                        attempt_count = attempt_count + 1,
                        started_at = COALESCE(started_at, ?1),
                        updated_at = ?1,
                        error_code = NULL,
                        error_message = NULL
                    WHERE id IN (
                        SELECT s.id
                        {claimable}
                        LIMIT 1
                        FOR UPDATE OF s, j SKIP LOCKED
                    )
                    RETURNING id, job_id
                    "#
                ),
                params,
            )?)
            .await
            .context("Failed to claim runtime job stage")?;
        let Some(claimed) = claimed else {
            tx.rollback().await?;
            return Ok(None);
        };
        let stage_id: String = col(&claimed, "id")?;
        let job_id: String = col(&claimed, "job_id")?;
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
            WHERE id = ?2 AND status IN ('queued', 'running', 'retrying')
            "#,
            vec![self.now_millis().into(), job_id.clone().into()],
        )?)
        .await
        .context("Failed to mark claimed runtime job running")?;
        let stage = get_stage_with(&tx, &stage_id)
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job stage was not found"))?;
        let job = get_job_with(&tx, &job_id)
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job was not found"))?;
        tx.commit()
            .await
            .context("Failed to commit runtime stage claim transaction")?;
        Ok(Some(ClaimedStage { job, stage }))
    }

    pub(super) async fn try_claim_stage_candidate(
        &self,
        db: &DatabaseConnection,
        candidate: StageClaimCandidate,
        worker_id: &str,
        now: i64,
        lease_expires_at: i64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime stage claim transaction")?;
        lock_stage_job_with(&tx, &candidate.stage_id).await?;
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
                        AND predecessor.status <> 'completed'
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('queued', 'running', 'retrying')
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
              AND status IN ('queued', 'running', 'retrying')
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
        validate_stage_output_artifact_ids(&output_artifact_ids)?;
        self.inject_fault(StoreFaultPoint::CompleteStage)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime stage completion transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let now = self.now_millis();
        let output_json = json_to_db_string(&json!(output_artifact_ids), "[]")?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = 'completed',
                    cancellation_state = NULL,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    output_artifact_ids_json = ?2,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?3
                  AND status = 'running'
                  AND cancellation_state IS NULL
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?1
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('queued', 'running', 'retrying')
                        AND runtime_jobs.cancellation_state IS NULL
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
            // A durable executor may settle its final route projection and
            // stage in one transaction before returning its output. Treat the
            // worker's follow-up completion call as idempotent only when the
            // exact attempt and output set match that committed settlement.
            if let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? {
                if stage.status == RuntimeStageStatus::Completed
                    && stage.attempt_count == lease.attempt_count
                    && stage.attempt_token == lease.attempt_token
                    && stage.output_artifact_ids == output_artifact_ids
                {
                    tx.commit().await?;
                    return Ok(Some(stage));
                }
            }
            tx.rollback().await?;
            return Ok(None);
        }

        let stage = get_stage_with(&tx, &lease.stage_id)
            .await?
            .ok_or_else(|| anyhow!("Completed runtime job stage was not found"))?;
        if let Err(error) = validate_stage_output_artifact_ids(&output_artifact_ids) {
            tx.rollback().await?;
            return Err(error);
        }
        if !output_artifact_ids.is_empty() {
            let Some(attempt_token) = lease.attempt_token.as_ref() else {
                tx.rollback().await?;
                bail!("Stage output artifacts require an exact attempt token");
            };
            let row_lock = match tx.get_database_backend() {
                DbBackend::Sqlite => "",
                DbBackend::Postgres => " FOR UPDATE",
                backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
            };
            for artifact_id in &output_artifact_ids {
                let ownership_sql = format!(
                    r#"
                    SELECT 1
                    FROM runtime_artifacts
                    WHERE id = ?1
                      AND job_id = ?2
                      AND stage_id = ?3
                      AND producer_attempt_count = ?4
                      AND producer_attempt_token = ?5
                      AND artifact_role IN ('output_primary', 'output_intermediate', 'debug')
                    LIMIT 1{row_lock}
                    "#
                );
                let owned = tx
                    .query_one_raw(raw::statement(
                        &tx,
                        ownership_sql,
                        vec![
                            artifact_id.clone().into(),
                            stage.job_id.clone().into(),
                            lease.stage_id.clone().into(),
                            u32_to_i64_value(lease.attempt_count).into(),
                            attempt_token.clone().into(),
                        ],
                    )?)
                    .await
                    .context("Failed to validate runtime stage output ownership")?;
                if owned.is_none() {
                    tx.rollback().await?;
                    bail!(
                        "Stage output artifact is not owned by the exact active job, stage, and attempt"
                    );
                }
            }
        }
        complete_job_if_all_stages_finished_with(&tx, stage.job_id.as_str(), now).await?;
        tx.commit()
            .await
            .context("Failed to commit runtime stage completion transaction")?;
        // The job's next stage, if any, is claimable now.
        self.wake_claimers();
        Ok(Some(stage))
    }

    pub async fn renew_stage_lease(
        &self,
        lease: &StageLease,
        lease_duration_ms: u64,
    ) -> anyhow::Result<bool> {
        self.inject_fault(StoreFaultPoint::RenewStageLease)?;
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms.max(1))?);
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET lease_expires_at = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status = 'running'
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND (
                      lease_expires_at > ?2
                      OR cancellation_state IN ('requested', 'execution_stopping')
                  )
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

    pub async fn stage_lease_state(
        &self,
        lease: &StageLease,
    ) -> anyhow::Result<Option<StageLeaseState>> {
        self.inject_fault(StoreFaultPoint::StageLeaseState)?;
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT s.cancellation_state
                FROM job_stages s
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.id = ?1
                  AND s.status = 'running'
                  AND s.worker_id = ?2
                  AND s.attempt_count = ?3
                  AND (s.attempt_token = ?4 OR (s.attempt_token IS NULL AND ?4 IS NULL))
                  AND s.lease_expires_at IS NOT NULL
                  AND (
                      s.lease_expires_at > ?5
                      OR s.cancellation_state IN ('requested', 'execution_stopping')
                  )
                  AND j.status IN ('queued', 'running', 'retrying')
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
        let Some(row) = row else {
            return Ok(None);
        };
        let state: Option<String> = row.try_get_by_index(0)?;
        match parse_cancellation_state(state)? {
            None => Ok(Some(StageLeaseState::Active)),
            Some(RuntimeCancellationState::Requested) => {
                Ok(Some(StageLeaseState::CancellationRequested))
            }
            Some(RuntimeCancellationState::ExecutionStopping) => {
                Ok(Some(StageLeaseState::ExecutionStopping))
            }
        }
    }

    pub async fn stage_lease_is_active(&self, lease: &StageLease) -> anyhow::Result<bool> {
        Ok(self.stage_lease_state(lease).await?.is_some())
    }

    pub async fn update_stage_progress(
        &self,
        lease: &StageLease,
        progress: serde_json::Value,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let progress_json = json_to_db_string(&progress, "{}")?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET progress_json = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status = 'running'
                  AND cancellation_state IS NULL
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?2
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('queued', 'running', 'retrying')
                        AND runtime_jobs.cancellation_state IS NULL
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

    /// Cooperative continuation consumes no retry attempt and rejoins the queue tail.
    /// Progress/artifacts remain durable. The next claim mints a fresh attempt token,
    /// so reusing the retry count cannot authorize any write from the old worker.
    pub async fn yield_stage(&self, lease: &StageLease) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
            UPDATE job_stages SET status = 'queued', worker_id = NULL,
                lease_expires_at = NULL, attempt_token = NULL,
                attempt_count = attempt_count - 1, available_at = ?1, updated_at = ?1
            WHERE id = ?2 AND worker_id = ?3 AND attempt_count = ?4 AND attempt_token = ?5
                AND attempt_count > 0 AND status = 'running'
                AND cancellation_state IS NULL AND lease_expires_at > ?1
                AND job_id IN (
                    SELECT id FROM runtime_jobs
                    WHERE status IN ('running','queued','retrying')
                      AND cancellation_state IS NULL
                )
        "#,
                vec![
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    i64::from(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await?;
        let yielded = result.rows_affected() == 1;
        if yielded {
            self.wake_claimers();
        }
        Ok(yielded)
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
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start runtime stage failure transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !is_live_stage_status(stage.status) || stage.cancellation_state.is_some()
        {
            tx.rollback().await?;
            return Ok(None);
        }
        let job = get_job_with(&tx, &stage.job_id)
            .await?
            .ok_or_else(|| anyhow!("Runtime stage parent job was not found"))?;
        if !is_claimable_job_status(job.status) || job.cancellation_state.is_some() {
            tx.rollback().await?;
            return Ok(None);
        }
        let policy = StoredRetryPolicy::from_job(&job);
        let should_retry =
            retryable && stage.attempt_count < policy.effective_max_attempts(&job, &stage);
        let now = self.now_millis();

        let result = if should_retry {
            let available_at = now.saturating_add(i64::try_from(policy.backoff_ms(&stage))?);
            self.retry_stage(
                &tx,
                &stage,
                lease,
                LeaseValidity::Any,
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
                LeaseValidity::Any,
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
        self.wake_claimers();
        Ok(result)
    }

    pub async fn mark_stage_execution_stopping(&self, lease: &StageLease) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start execution-stopping transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let now = self.now_millis();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET cancellation_state = 'execution_stopping', updated_at = ?1
                WHERE id = ?2
                  AND status = 'running'
                  AND cancellation_state = 'requested'
                  AND worker_id = ?3
                  AND attempt_count = ?4
                  AND (attempt_token = ?5 OR (attempt_token IS NULL AND ?5 IS NULL))
                  AND lease_expires_at IS NOT NULL
                "#,
                vec![
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage execution stopping")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE runtime_jobs
            SET cancellation_state = 'execution_stopping', updated_at = ?1
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND cancellation_state IS NOT NULL
              AND status IN ('queued', 'running', 'retrying')
            "#,
            vec![now.into(), lease.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to mark runtime job execution stopping")?;
        tx.commit()
            .await
            .context("Failed to commit execution-stopping transaction")?;
        Ok(true)
    }

    /// Finalize only after the caller knows this exact attempt's executor has
    /// resolved. Attempt identity, rather than wall-clock lease freshness,
    /// fences late settlement from stale workers.
    pub async fn finalize_stage_cancellation(
        &self,
        lease: &StageLease,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start stage cancellation finalization transaction")?;
        lock_stage_job_with(&tx, &lease.stage_id).await?;
        let now = self.now_millis();
        let finalized = self
            .finalize_stage_cancellation_with(
                &tx,
                lease,
                now,
                CancellationSettlement::OwnerTeardown,
            )
            .await?;
        if finalized.is_none() {
            tx.rollback().await?;
            return Ok(None);
        }
        tx.commit()
            .await
            .context("Failed to commit stage cancellation finalization transaction")?;
        Ok(finalized)
    }

    pub async fn recover_expired_stage_leases(&self, limit: usize) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let limit = bounded_maintenance_batch_limit(limit);
        // A wall-clock expiry does not prove that requested cancellation tore
        // down execution. Only ordinary leases are eligible for retry here;
        // cancelling attempts stay owner-fenced until exact-attempt settlement
        // or until their owner is provably gone (see below).
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT id, worker_id, attempt_count, attempt_token
                FROM job_stages
                WHERE status = 'running'
                  AND cancellation_state IS NULL
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?1
                ORDER BY lease_expires_at ASC, id ASC
                LIMIT ?2
                "#,
                vec![now.into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to list expired runtime stage leases")?;

        let scanned = rows.len();
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
                .begin_with_options(write_transaction_options())
                .await
                .context("Failed to start expired lease recovery transaction")?;
            lock_stage_job_with(&tx, &lease.stage_id).await?;
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

        recovered = recovered.saturating_add(
            self.settle_abandoned_cancellations(db, now, limit.saturating_sub(scanned))
                .await?,
        );
        if recovered > 0 {
            self.wake_claimers();
        }
        Ok(recovered)
    }

    /// A cancelling attempt whose owner process died can never be settled by
    /// that owner: its attempt token lived only in the dead process. Once the
    /// lease has lapsed by the grace period and no incarnation that could own
    /// the attempt has heartbeated since, the executor is gone too, so the
    /// requested cancellation is complete. The settlement stays fenced on the
    /// exact attempt and re-checks abandonment inside the write transaction.
    pub(super) async fn settle_abandoned_cancellations(
        &self,
        db: &DatabaseConnection,
        now: i64,
        limit: usize,
    ) -> anyhow::Result<u64> {
        if limit == 0 {
            return Ok(0);
        }
        let stale_before = now.saturating_sub(ABANDONED_CANCELLATION_GRACE_MS);
        let settlement = CancellationSettlement::AbandonedOwner { stale_before };
        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    r#"
                    SELECT id, worker_id, attempt_count, attempt_token
                    FROM job_stages
                    WHERE status = 'running'
                      AND cancellation_state IN ('requested', 'execution_stopping')
                      AND worker_id IS NOT NULL
                      AND lease_expires_at IS NOT NULL
                      AND {}
                    ORDER BY lease_expires_at ASC, id ASC
                    LIMIT ?2
                    "#,
                    settlement.sql_predicate(1)
                ),
                vec![stale_before.into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to list abandoned runtime stage cancellations")?;

        let mut settled = 0_u64;
        for row in rows {
            let lease = StageLease {
                stage_id: row.try_get_by_index(0)?,
                worker_id: row.try_get_by_index(1)?,
                attempt_count: i64_to_u32(row.try_get_by_index(2)?)?,
                attempt_token: row.try_get_by_index(3)?,
            };
            let tx = db
                .begin_with_options(write_transaction_options())
                .await
                .context("Failed to start abandoned cancellation settlement transaction")?;
            lock_stage_job_with(&tx, &lease.stage_id).await?;
            if self
                .finalize_stage_cancellation_with(&tx, &lease, now, settlement)
                .await?
                .is_some()
            {
                tx.commit()
                    .await
                    .context("Failed to commit abandoned cancellation settlement transaction")?;
                settled = settled.saturating_add(1);
            } else {
                tx.rollback().await?;
            }
        }
        Ok(settled)
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
        let now = self.now_millis();
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
                resource_hints_version,
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
            VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?18, ?13, ?14, ?15, NULL, NULL, ?3, NULL, 0, ?16, ?17, '[]', NULL, NULL, NULL, NULL, NULL)
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
                i64::from(resource_hints.version).into(),
            ],
        )?)
        .await
        .context("Failed to create runtime job stage")?;
        self.wake_claimers();

        self.get_stage(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job stage was not found"))
    }

    pub async fn get_stage(&self, id: &str) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(db, JOB_STAGE_BY_ID_SQL, vec![id.into()])?)
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

    pub(super) async fn retry_stage<C: ConnectionTrait>(
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
                    available_at = ?7,
                    attempt_token = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status = 'running'
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

    pub(super) async fn finalize_stage_cancellation_with<C: ConnectionTrait>(
        &self,
        db: &C,
        lease: &StageLease,
        now: i64,
        settlement: CancellationSettlement,
    ) -> anyhow::Result<Option<JobStage>> {
        let sql = format!(
            r#"
                    UPDATE job_stages
                    SET
                        status = 'cancelled',
                        cancellation_state = NULL,
                        updated_at = ?1,
                        finished_at = COALESCE(finished_at, ?1),
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        available_at = NULL,
                        error_code = NULL,
                        error_message = NULL
                    WHERE id = ?2
                      AND status = 'running'
                      AND cancellation_state IN ('requested', 'execution_stopping')
                      AND worker_id = ?3
                      AND attempt_count = ?4
                      AND (attempt_token = ?5 OR (attempt_token IS NULL AND ?5 IS NULL))
                      AND lease_expires_at IS NOT NULL
                      AND {}
                    "#,
            settlement.sql_predicate(6)
        );
        let mut values = vec![
            now.into(),
            lease.stage_id.clone().into(),
            lease.worker_id.clone().into(),
            u32_to_i64_value(lease.attempt_count).into(),
            opt_string(lease.attempt_token.clone()),
        ];
        values.extend(settlement.bound_value());
        let result = db
            .execute_raw(raw::statement(db, sql, values)?)
            .await
            .context("Failed to finalize runtime stage cancellation")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'cancelled',
                cancellation_state = NULL,
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1)
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND cancellation_state IS NOT NULL
              AND status IN ('queued', 'running', 'retrying')
              AND NOT EXISTS (
                  SELECT 1 FROM job_stages
                  WHERE job_stages.job_id = runtime_jobs.id
                    AND job_stages.status = 'running'
                    AND job_stages.cancellation_state IS NOT NULL
              )
            "#,
            vec![now.into(), lease.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to finalize runtime job cancellation")?;

        let finalized = get_stage_with(db, &lease.stage_id).await?;
        if let Some(stage) = &finalized {
            self.finalize_cancelled_route_projection_with(db, &stage.job_id)
                .await?;
        }
        Ok(finalized)
    }

    pub(super) async fn finalize_cancelled_route_projection_with<C: ConnectionTrait>(
        &self,
        db: &C,
        job_id: &str,
    ) -> anyhow::Result<u64> {
        let transcription = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE transcription_records
                SET
                    processing_status = 'failed',
                    processing_error = COALESCE(
                        NULLIF((SELECT cancellation_reason FROM runtime_jobs WHERE id = ?1), ''),
                        'Runtime job cancelled'
                    ),
                    processing_progress_json = NULL,
                    runtime_stage_id = NULL,
                    runtime_attempt_token = NULL
                WHERE id = (SELECT route_record_id FROM runtime_jobs WHERE id = ?1)
                  AND processing_status IN ('pending', 'processing')
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE id = ?1
                        AND job_kind = 'asr_transcription'
                        AND status = 'cancelled'
                        AND route_record_kind IN ('transcription', 'speaker_attributed_asr')
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM runtime_jobs AS active
                      JOIN runtime_jobs AS cancelled ON cancelled.id = ?1
                      WHERE active.id <> cancelled.id
                        AND active.job_kind = cancelled.job_kind
                        AND active.route_record_kind = cancelled.route_record_kind
                        AND active.route_record_id = cancelled.route_record_id
                        AND active.status IN ('queued', 'running', 'retrying')
                  )
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to finalize cancelled transcription projection")?
            .rows_affected();
        let speech = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE speech_history_records
                SET
                    processing_status = 'failed',
                    processing_error = COALESCE(
                        NULLIF((SELECT cancellation_reason FROM runtime_jobs WHERE id = ?1), ''),
                        'Runtime job cancelled'
                    ),
                    runtime_stage_id = NULL,
                    runtime_attempt_token = NULL
                WHERE id = (SELECT route_record_id FROM runtime_jobs WHERE id = ?1)
                  AND route_kind = (SELECT route_record_kind FROM runtime_jobs WHERE id = ?1)
                  AND processing_status IN ('pending', 'processing')
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE id = ?1
                        AND job_kind = 'tts_speech'
                        AND status = 'cancelled'
                        AND route_record_kind IN ('text_to_speech', 'voice_design', 'voice_cloning')
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM runtime_jobs AS active
                      JOIN runtime_jobs AS cancelled ON cancelled.id = ?1
                      WHERE active.id <> cancelled.id
                        AND active.job_kind = cancelled.job_kind
                        AND active.route_record_kind = cancelled.route_record_kind
                        AND active.route_record_id = cancelled.route_record_id
                        AND active.status IN ('queued', 'running', 'retrying')
                  )
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to finalize cancelled speech projection")?
            .rows_affected();

        Ok(transcription.saturating_add(speech))
    }

    pub(super) async fn mark_stage_failed<C: ConnectionTrait>(
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
                  AND status = 'running'
                  AND worker_id = ?5
                  AND attempt_count = ?6
                  AND (attempt_token = ?7 OR (attempt_token IS NULL AND ?7 IS NULL))
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
        .context("Failed to mark runtime job failed")?;
        get_stage_with(db, stage.id.as_str()).await
    }
}

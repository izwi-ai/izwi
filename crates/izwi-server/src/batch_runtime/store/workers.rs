//! Worker heartbeats and queue health.

use super::*;

#[derive(Debug, Clone)]
pub struct WorkerHeartbeatUpdate {
    pub worker_id: String,
    pub status: String,
    pub queue_names: Vec<String>,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub diagnostic_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeQueueDepth {
    pub queue_class: QueueClass,
    pub count: u64,
    pub oldest_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeQueueHealthSnapshot {
    pub heartbeat_stale_after_ms: u64,
    pub active_workers: u64,
    pub healthy_workers: u64,
    pub stale_workers: u64,
    pub queues: Vec<RuntimeQueueDepth>,
    pub uncovered_queue_classes: Vec<QueueClass>,
}

pub(super) fn worker_heartbeat_upsert_statement(
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
        backend => bail!("Unsupported SeaORM database backend: {backend:?}"),
    }
}

pub(super) fn worker_heartbeat_accepts_claims(heartbeat: &RuntimeWorkerHeartbeat) -> bool {
    matches!(heartbeat.status.as_str(), "polling" | "idle" | "running")
}

impl BatchRuntimeStore {
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

    pub async fn runtime_queue_health(
        &self,
        heartbeat_stale_after_ms: u64,
    ) -> anyhow::Result<RuntimeQueueHealthSnapshot> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let queue_rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT queue_class, COUNT(*), MIN(created_at)
                FROM job_stages
                WHERE status IN ('queued', 'retrying')
                GROUP BY queue_class
                ORDER BY queue_class
                "#,
                vec![],
            )?)
            .await
            .context("Failed to load runtime queue depth and age")?;
        let queues = queue_rows
            .iter()
            .map(|row| {
                let queue_raw: String = row.try_get_by_index(0)?;
                let queue_class = QueueClass::from_db_value(&queue_raw)
                    .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                let oldest_created_at = row.try_get_by_index::<Option<i64>>(2)?.unwrap_or(now);
                Ok(RuntimeQueueDepth {
                    queue_class,
                    count,
                    oldest_age_ms: i64_to_u64(now.saturating_sub(oldest_created_at).max(0))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let heartbeats = self.list_worker_heartbeats().await?;
        let stale_cutoff = now.saturating_sub(i64::try_from(heartbeat_stale_after_ms)?);
        let active_workers = heartbeats
            .iter()
            .filter(|heartbeat| worker_heartbeat_accepts_claims(heartbeat))
            .count() as u64;
        let healthy = heartbeats
            .iter()
            .filter(|heartbeat| {
                worker_heartbeat_accepts_claims(heartbeat)
                    && i64::try_from(heartbeat.last_heartbeat_at)
                        .is_ok_and(|last| last >= stale_cutoff)
            })
            .collect::<Vec<_>>();
        let stale_workers = active_workers.saturating_sub(healthy.len() as u64);
        let uncovered_queue_classes = queues
            .iter()
            .filter(|queue| {
                !healthy.iter().any(|heartbeat| {
                    heartbeat
                        .registration
                        .queue_classes
                        .iter()
                        .any(|worker_queue| {
                            *worker_queue == QueueClass::Batch || *worker_queue == queue.queue_class
                        })
                })
            })
            .map(|queue| queue.queue_class)
            .collect();

        Ok(RuntimeQueueHealthSnapshot {
            heartbeat_stale_after_ms,
            active_workers,
            healthy_workers: healthy.len() as u64,
            stale_workers,
            queues,
            uncovered_queue_classes,
        })
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
        self.inject_fault(StoreFaultPoint::WorkerHeartbeat)?;
        let db = self.db.connection().await?;
        let now = self.now_millis();
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
                WORKER_HEARTBEAT_BY_ID_SQL,
                vec![worker_id.into()],
            )?)
            .await
            .context("Failed to load runtime worker heartbeat")?;

        row.as_ref().map(map_worker_heartbeat).transpose()
    }

    pub async fn list_worker_heartbeats(&self) -> anyhow::Result<Vec<RuntimeWorkerHeartbeat>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(db, RUNTIME_WORKER_HEARTBEATS_SQL, vec![])?)
            .await
            .context("Failed to list runtime worker heartbeats")?;
        rows.iter().map(map_worker_heartbeat).collect()
    }
}

use super::{
    store::{BatchRuntimeStore, WorkerHeartbeatUpdate},
    types::ClaimedStage,
};
use anyhow::{Context, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

#[derive(Debug, Clone)]
pub struct BatchWorkerConfig {
    pub worker_id: String,
    pub queue_names: Vec<String>,
    pub poll_interval: Duration,
    pub lease_duration: Duration,
}

impl BatchWorkerConfig {
    pub fn local(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            queue_names: vec!["batch".to_string()],
            poll_interval: Duration::from_millis(250),
            lease_duration: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BatchWorkerSnapshot {
    pub worker_id: String,
    pub running: bool,
    pub last_heartbeat_at: u64,
    pub last_claimed_stage_id: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchWorkerHealth {
    inner: Arc<RwLock<BatchWorkerHealthInner>>,
}

#[derive(Debug)]
struct BatchWorkerHealthInner {
    worker_id: String,
    running: bool,
    last_heartbeat_at: u64,
    last_claimed_stage_id: Option<String>,
    last_error: Option<String>,
}

impl BatchWorkerHealth {
    pub fn new(worker_id: impl Into<String>) -> Self {
        let worker_id = worker_id.into();
        Self {
            inner: Arc::new(RwLock::new(BatchWorkerHealthInner {
                worker_id,
                running: false,
                last_heartbeat_at: now_secs(),
                last_claimed_stage_id: None,
                last_error: None,
            })),
        }
    }

    pub fn mark_running(&self) {
        self.update(|inner| {
            inner.running = true;
            inner.last_heartbeat_at = now_secs();
        });
    }

    pub fn mark_stopped(&self) {
        self.update(|inner| {
            inner.running = false;
            inner.last_heartbeat_at = now_secs();
        });
    }

    pub fn record_claim(&self, stage_id: impl Into<String>) {
        self.update(|inner| {
            inner.last_claimed_stage_id = Some(stage_id.into());
            inner.last_heartbeat_at = now_secs();
            inner.last_error = None;
        });
    }

    pub fn record_error(&self, error: impl Into<String>) {
        self.update(|inner| {
            inner.last_error = Some(error.into());
            inner.last_heartbeat_at = now_secs();
        });
    }

    pub fn snapshot(&self) -> BatchWorkerSnapshot {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        BatchWorkerSnapshot {
            worker_id: guard.worker_id.clone(),
            running: guard.running,
            last_heartbeat_at: guard.last_heartbeat_at,
            last_claimed_stage_id: guard.last_claimed_stage_id.clone(),
            last_error: guard.last_error.clone(),
        }
    }

    fn update(&self, f: impl FnOnce(&mut BatchWorkerHealthInner)) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        f(&mut guard);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageExecutionOutcome {
    pub output_artifact_ids: Vec<String>,
}

impl StageExecutionOutcome {
    pub fn empty() -> Self {
        Self {
            output_artifact_ids: Vec::new(),
        }
    }
}

#[async_trait]
pub trait StageExecutor: Send + Sync {
    fn stage_kind(&self) -> &'static str;

    async fn execute(&self, claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome>;
}

#[derive(Clone)]
pub struct BatchWorkerRunner {
    store: Arc<BatchRuntimeStore>,
    executors: Arc<HashMap<String, Arc<dyn StageExecutor>>>,
    config: BatchWorkerConfig,
    health: BatchWorkerHealth,
}

impl BatchWorkerRunner {
    pub fn new(
        store: Arc<BatchRuntimeStore>,
        executors: Vec<Arc<dyn StageExecutor>>,
        config: BatchWorkerConfig,
        health: BatchWorkerHealth,
    ) -> Self {
        let executors = executors
            .into_iter()
            .map(|executor| (executor.stage_kind().to_string(), executor))
            .collect();
        Self {
            store,
            executors: Arc::new(executors),
            config,
            health,
        }
    }

    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    pub async fn run_once(&self) -> anyhow::Result<bool> {
        self.store
            .recover_expired_stage_leases()
            .await
            .context("Failed to recover expired runtime stage leases")?;
        self.record_heartbeat("polling", None).await?;

        let Some(claimed) = self
            .store
            .claim_next_stage(
                self.config.worker_id.as_str(),
                self.config.lease_duration.as_millis() as u64,
            )
            .await?
        else {
            return Ok(false);
        };

        self.health.record_claim(claimed.stage.id.clone());
        self.record_heartbeat(
            "running",
            Some((claimed.job.id.clone(), claimed.stage.id.clone())),
        )
        .await?;

        let Some(executor) = self.executors.get(claimed.stage.stage_kind.as_str()).cloned() else {
            let message = format!("No executor registered for stage {}", claimed.stage.stage_kind);
            self.health.record_error(message.clone());
            self.store
                .fail_stage(
                    claimed.stage.id.as_str(),
                    false,
                    Some("missing_executor".to_string()),
                    Some(message),
                )
                .await?;
            return Ok(true);
        };

        match executor.execute(claimed.clone()).await {
            Ok(outcome) => {
                self.store
                    .complete_stage(claimed.stage.id.as_str(), outcome.output_artifact_ids)
                    .await?;
                self.record_heartbeat("idle", None).await?;
            }
            Err(err) => {
                let message = err.to_string();
                self.health.record_error(message.clone());
                self.store
                    .fail_stage(
                        claimed.stage.id.as_str(),
                        true,
                        Some("executor_failed".to_string()),
                        Some(message),
                    )
                    .await?;
                self.record_heartbeat("idle", None).await?;
            }
        }

        Ok(true)
    }

    pub async fn run_until_idle(&self, max_iterations: usize) -> anyhow::Result<usize> {
        let mut processed = 0_usize;
        for _ in 0..max_iterations {
            if !self.run_once().await? {
                break;
            }
            processed += 1;
        }
        Ok(processed)
    }

    pub fn spawn(self) -> BatchWorkerSupervisor {
        let health = self.health.clone();
        health.mark_running();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let runner = self.clone();
        let handle = tokio::spawn(async move {
            info!(worker_id = %runner.config.worker_id, "Batch runtime worker started");
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        break;
                    }
                    result = runner.run_once() => {
                        match result {
                            Ok(true) => {}
                            Ok(false) => tokio::time::sleep(runner.config.poll_interval).await,
                            Err(err) => {
                                error!(worker_id = %runner.config.worker_id, error = %err, "Batch runtime worker iteration failed");
                                runner.health.record_error(err.to_string());
                                tokio::time::sleep(runner.config.poll_interval).await;
                            }
                        }
                    }
                }
            }
            runner.health.mark_stopped();
            debug!(worker_id = %runner.config.worker_id, "Batch runtime worker stopped");
        });
        BatchWorkerSupervisor {
            shutdown_tx: Some(shutdown_tx),
            handle,
            health,
        }
    }

    async fn record_heartbeat(
        &self,
        status: &str,
        current: Option<(String, String)>,
    ) -> anyhow::Result<()> {
        let (current_job_id, current_stage_id) =
            current.map_or((None, None), |(job_id, stage_id)| {
                (Some(job_id), Some(stage_id))
            });
        self.store
            .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
                worker_id: self.config.worker_id.clone(),
                status: status.to_string(),
                queue_names: self.config.queue_names.clone(),
                current_job_id,
                current_stage_id,
                diagnostic_json: serde_json::json!({}),
            })
            .await?;
        Ok(())
    }
}

pub struct BatchWorkerSupervisor {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    handle: JoinHandle<()>,
    health: BatchWorkerHealth,
}

impl BatchWorkerSupervisor {
    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.handle
            .await
            .map_err(|err| anyhow!("Batch worker task join failed: {err}"))?;
        Ok(())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batch_runtime::{
            store::{NewJobStage, NewRuntimeJob},
            types::{RuntimeJobKind, RuntimeJobStatus, RuntimeStageStatus},
        },
        db::StoreDatabase,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeExecutor {
        calls: AtomicUsize,
        fail_first: bool,
    }

    #[async_trait]
    impl StageExecutor for FakeExecutor {
        fn stage_kind(&self) -> &'static str {
            "fake_stage"
        }

        async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_first && call == 0 {
                anyhow::bail!("planned fake failure");
            }
            Ok(StageExecutionOutcome {
                output_artifact_ids: vec!["artifact-1".to_string()],
            })
        }
    }

    fn build_store() -> Arc<BatchRuntimeStore> {
        let root = tempfile::tempdir().expect("temp dir");
        let db_path = root.keep().join("runtime.sqlite");
        Arc::new(BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            db_path,
        )))
    }

    async fn create_queued_fake_stage(
        store: &BatchRuntimeStore,
        max_attempts: u32,
    ) -> anyhow::Result<(String, String)> {
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: None,
                capability: Some("test".to_string()),
                route_record_kind: Some("test".to_string()),
                route_record_id: Some("route-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts,
                idempotency_key: None,
                correlation_id: None,
            })
            .await?;
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "fake_stage".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("test".to_string()),
                model_id: None,
                max_attempts,
                input_artifact_ids: vec![],
            })
            .await?;
        Ok((job.id, stage.id))
    }

    #[tokio::test]
    async fn runner_claims_and_completes_stage() {
        let store = build_store();
        let (job_id, stage_id) = create_queued_fake_stage(&store, 1)
            .await
            .expect("stage");
        let health = BatchWorkerHealth::new("worker-test");
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                fail_first: false,
            })],
            BatchWorkerConfig::local("worker-test"),
            health.clone(),
        );

        let processed = runner.run_until_idle(4).await.expect("run");

        assert_eq!(processed, 1);
        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Completed);
        assert_eq!(stage.output_artifact_ids, vec!["artifact-1"]);
        let job = store
            .get_job(&job_id)
            .await
            .expect("job")
            .expect("job exists");
        assert_eq!(job.status, RuntimeJobStatus::Completed);
        assert_eq!(
            health.snapshot().last_claimed_stage_id.as_deref(),
            Some(stage_id.as_str())
        );
    }

    #[tokio::test]
    async fn runner_retries_then_completes_stage() {
        let store = build_store();
        let (job_id, stage_id) = create_queued_fake_stage(&store, 2)
            .await
            .expect("stage");
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                fail_first: true,
            })],
            BatchWorkerConfig::local("worker-test"),
            BatchWorkerHealth::new("worker-test"),
        );

        let processed = runner.run_until_idle(4).await.expect("run");

        assert_eq!(processed, 2);
        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Completed);
        assert_eq!(stage.attempt_count, 2);
        let job = store
            .get_job(&job_id)
            .await
            .expect("job")
            .expect("job exists");
        assert_eq!(job.status, RuntimeJobStatus::Completed);
    }
}

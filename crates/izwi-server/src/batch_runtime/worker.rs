use super::{
    store::{BatchRuntimeStore, StageClaimFilter, WorkerHeartbeatUpdate},
    types::{ClaimedStage, RuntimeJobKind},
};
use anyhow::{anyhow, Context};
use async_trait::async_trait;
use izwi_core::{
    RuntimeObservationContext, RuntimeService, RuntimeStageObservation, RuntimeStageOutcome,
    RuntimeStageOutputCounters,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

#[derive(Debug, Clone)]
pub struct BatchWorkerConfig {
    pub worker_id: String,
    pub queue_names: Vec<String>,
    pub capabilities: Vec<String>,
    pub model_ids: Vec<String>,
    pub stage_kinds: Vec<String>,
    pub draining: bool,
    pub poll_interval: Duration,
    pub lease_duration: Duration,
}

impl BatchWorkerConfig {
    pub fn local(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            queue_names: vec!["batch".to_string()],
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            draining: false,
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
    runtime_observer: Option<Arc<RuntimeService>>,
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
            runtime_observer: None,
        }
    }

    pub fn with_runtime_observer(mut self, runtime: Arc<RuntimeService>) -> Self {
        self.runtime_observer = Some(runtime);
        self
    }

    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    pub async fn run_once(&self) -> anyhow::Result<bool> {
        self.store
            .recover_expired_stage_leases()
            .await
            .context("Failed to recover expired runtime stage leases")?;
        if self.config.draining {
            self.record_heartbeat("draining", None).await?;
            return Ok(false);
        }
        self.record_heartbeat("polling", None).await?;
        let claim_filter = self.claim_filter();

        let Some(claimed) = self
            .store
            .claim_next_stage_with_filter(
                self.config.worker_id.as_str(),
                self.config.lease_duration.as_millis() as u64,
                &claim_filter,
            )
            .await?
        else {
            return Ok(false);
        };

        self.health.record_claim(claimed.stage.id.clone());
        self.record_stage_observation(&claimed, RuntimeStageOutcome::Claimed, None, None, None);
        self.record_heartbeat(
            "running",
            Some((claimed.job.id.clone(), claimed.stage.id.clone())),
        )
        .await?;

        let Some(executor) = self
            .executors
            .get(claimed.stage.stage_kind.as_str())
            .cloned()
        else {
            let message = format!(
                "No executor registered for stage {}",
                claimed.stage.stage_kind
            );
            self.health.record_error(message.clone());
            self.store
                .fail_stage(
                    claimed.stage.id.as_str(),
                    false,
                    Some("missing_executor".to_string()),
                    Some(message),
                )
                .await?;
            self.record_stage_observation(
                &claimed,
                RuntimeStageOutcome::Failed,
                None,
                None,
                Some("missing_executor".to_string()),
            );
            return Ok(true);
        };

        self.record_stage_observation(&claimed, RuntimeStageOutcome::Started, None, None, None);
        let stage_started = Instant::now();
        match executor.execute(claimed.clone()).await {
            Ok(outcome) => {
                let output_artifact_count = outcome.output_artifact_ids.len();
                self.store
                    .complete_stage(claimed.stage.id.as_str(), outcome.output_artifact_ids)
                    .await?;
                self.record_stage_observation(
                    &claimed,
                    RuntimeStageOutcome::Completed,
                    Some(stage_started.elapsed().as_secs_f64() * 1000.0),
                    Some(output_artifact_count),
                    None,
                );
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
                self.record_stage_observation(
                    &claimed,
                    RuntimeStageOutcome::Failed,
                    Some(stage_started.elapsed().as_secs_f64() * 1000.0),
                    None,
                    Some("executor_failed".to_string()),
                );
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
        let (current_job_id, current_stage_id) = current
            .map_or((None, None), |(job_id, stage_id)| {
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

    fn claim_filter(&self) -> StageClaimFilter {
        let mut filter = StageClaimFilter::for_worker_queues(&self.config.queue_names);
        filter.capabilities = normalized_claim_values(&self.config.capabilities);
        filter.model_ids = normalized_claim_values(&self.config.model_ids);
        filter.stage_kinds = normalized_claim_values(&self.config.stage_kinds);
        filter
    }

    fn record_stage_observation(
        &self,
        claimed: &ClaimedStage,
        outcome: RuntimeStageOutcome,
        total_ms: Option<f64>,
        output_artifacts: Option<usize>,
        error_kind: Option<String>,
    ) {
        let Some(runtime) = self.runtime_observer.as_ref() else {
            return;
        };

        let mut observation =
            RuntimeStageObservation::new(Self::stage_observation_context(claimed), outcome);
        if let Some(total_ms) = total_ms {
            observation = observation.with_total_ms(total_ms);
        }
        if let Some(output_artifacts) = output_artifacts {
            observation.outputs = RuntimeStageOutputCounters {
                output_artifacts: Some(output_artifacts as u64),
                ..RuntimeStageOutputCounters::default()
            };
        }
        if let Some(error_kind) = error_kind {
            observation = observation.with_error_kind(error_kind);
        }

        runtime.record_stage_observation(observation);
    }

    fn stage_observation_context(claimed: &ClaimedStage) -> RuntimeObservationContext {
        RuntimeObservationContext {
            route_source: Some("batch_runtime".to_string()),
            capability: claimed
                .stage
                .capability
                .clone()
                .or_else(|| claimed.job.capability.clone()),
            model_variant: claimed
                .stage
                .model_id
                .clone()
                .or_else(|| claimed.job.model_id.clone()),
            pipeline_kind: Some(batch_pipeline_kind(claimed.job.job_kind).to_string()),
            pipeline_stage: Some(claimed.stage.stage_kind.clone()),
            runtime_job_id: Some(claimed.job.id.clone()),
            job_stage_id: Some(claimed.stage.id.clone()),
            route_record_id: claimed.job.route_record_id.clone(),
            correlation_id: claimed.job.correlation_id.clone(),
            ..RuntimeObservationContext::default()
        }
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

fn batch_pipeline_kind(kind: RuntimeJobKind) -> &'static str {
    match kind {
        RuntimeJobKind::AsrTranscription => "batch_asr_transcription",
        RuntimeJobKind::TtsSpeech => "batch_tts_speech",
    }
}

fn normalized_claim_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
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
    use izwi_core::{EngineConfig, RuntimeStageOutcome};
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
        Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(db_path),
        ))
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
        let (job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
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
    async fn draining_runner_records_heartbeat_without_claiming() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
        let mut config = BatchWorkerConfig::local("worker-test");
        config.draining = true;
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                fail_first: false,
            })],
            config,
            BatchWorkerHealth::new("worker-test"),
        );

        assert!(!runner.run_once().await.expect("run once"));

        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Queued);
        assert_eq!(stage.worker_id, None);
        let heartbeat = store
            .get_worker_heartbeat("worker-test")
            .await
            .expect("heartbeat")
            .expect("heartbeat exists");
        assert_eq!(heartbeat.status, "draining");
    }

    #[tokio::test]
    async fn runner_records_runtime_stage_observations_when_attached() {
        let store = build_store();
        let (job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
        let runtime = Arc::new(RuntimeService::new(EngineConfig::default()).expect("runtime"));
        let runner = BatchWorkerRunner::new(
            store,
            vec![Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                fail_first: false,
            })],
            BatchWorkerConfig::local("worker-test"),
            BatchWorkerHealth::new("worker-test"),
        )
        .with_runtime_observer(runtime.clone());

        assert!(runner.run_once().await.expect("processed"));

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.observability.stage_observations_total, 3);
        assert_eq!(snapshot.observability.stage_failures_total, 0);
        let samples = snapshot.observability.recent_stage_samples;
        assert_eq!(samples[0].outcome, RuntimeStageOutcome::Claimed);
        assert_eq!(samples[1].outcome, RuntimeStageOutcome::Started);
        assert_eq!(samples[2].outcome, RuntimeStageOutcome::Completed);
        assert_eq!(
            samples[2].context.runtime_job_id.as_deref(),
            Some(job_id.as_str())
        );
        assert_eq!(
            samples[2].context.job_stage_id.as_deref(),
            Some(stage_id.as_str())
        );
        assert_eq!(
            samples[2].outputs.output_artifacts,
            Some(1),
            "completed stage should report output artifact count"
        );
    }

    #[tokio::test]
    async fn runner_retries_then_completes_stage() {
        let store = build_store();
        let (job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
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

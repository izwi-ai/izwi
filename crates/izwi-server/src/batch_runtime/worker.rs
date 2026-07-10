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
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{sync::Notify, task::JoinHandle};
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
    pub maintenance_interval: Duration,
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
            maintenance_interval: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchWorkerDrain {
    inner: Arc<BatchWorkerDrainInner>,
}

#[derive(Debug)]
struct BatchWorkerDrainInner {
    draining: AtomicBool,
    notify: Notify,
}

impl BatchWorkerDrain {
    fn new(draining: bool) -> Self {
        Self {
            inner: Arc::new(BatchWorkerDrainInner {
                draining: AtomicBool::new(draining),
                notify: Notify::new(),
            }),
        }
    }

    pub fn begin(&self) {
        if !self.inner.draining.swap(true, Ordering::AcqRel) {
            self.inner.notify.notify_one();
        }
    }

    pub fn is_draining(&self) -> bool {
        self.inner.draining.load(Ordering::Acquire)
    }

    async fn wait(&self) {
        loop {
            let notified = self.inner.notify.notified();
            if self.is_draining() {
                return;
            }
            notified.await;
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
    pub configured_capabilities: Vec<String>,
    pub configured_stage_kinds: Vec<String>,
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
    configured_capabilities: Vec<String>,
    configured_stage_kinds: Vec<String>,
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
                configured_capabilities: Vec::new(),
                configured_stage_kinds: Vec::new(),
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

    fn configure(&self, config: &BatchWorkerConfig) {
        self.update(|inner| {
            inner.configured_capabilities = config.capabilities.clone();
            inner.configured_stage_kinds = config.stage_kinds.clone();
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
            configured_capabilities: guard.configured_capabilities.clone(),
            configured_stage_kinds: guard.configured_stage_kinds.clone(),
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
    drain: BatchWorkerDrain,
    runtime_observer: Option<Arc<RuntimeService>>,
    last_maintenance_at: Arc<RwLock<Option<Instant>>>,
}

impl BatchWorkerRunner {
    pub fn new(
        store: Arc<BatchRuntimeStore>,
        executors: Vec<Arc<dyn StageExecutor>>,
        mut config: BatchWorkerConfig,
        health: BatchWorkerHealth,
    ) -> Self {
        let executors = executors
            .into_iter()
            .map(|executor| (executor.stage_kind().to_string(), executor))
            .collect::<HashMap<_, _>>();
        let mut registered_stage_kinds = executors.keys().cloned().collect::<Vec<_>>();
        registered_stage_kinds.sort();

        let requested_stage_kinds = normalized_claim_values(&config.stage_kinds);
        config.stage_kinds = if requested_stage_kinds.is_empty() {
            registered_stage_kinds
        } else {
            requested_stage_kinds
                .into_iter()
                .filter(|stage_kind| executors.contains_key(stage_kind))
                .collect()
        };
        config.capabilities = normalized_claim_values(&config.capabilities);
        config.model_ids = normalized_claim_values(&config.model_ids);
        config.queue_names = normalized_claim_values(&config.queue_names);
        if config.queue_names.is_empty() {
            config.queue_names.push("batch".to_string());
        }
        health.configure(&config);
        let drain = BatchWorkerDrain::new(config.draining);
        Self {
            store,
            executors: Arc::new(executors),
            config,
            health,
            drain,
            runtime_observer: None,
            last_maintenance_at: Arc::new(RwLock::new(None)),
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
        self.run_maintenance_if_due().await?;
        if self.drain.is_draining() {
            self.record_heartbeat("draining", None).await?;
            return Ok(false);
        }
        self.record_heartbeat("polling", None).await?;
        if self.drain.is_draining() || self.config.stage_kinds.is_empty() {
            self.record_heartbeat(
                if self.drain.is_draining() {
                    "draining"
                } else {
                    "idle"
                },
                None,
            )
            .await?;
            return Ok(false);
        }
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
        let lease = claimed
            .lease()
            .ok_or_else(|| anyhow!("Claimed stage is missing worker lease ownership"))?;

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
            let failed = self
                .store
                .fail_stage(
                    &lease,
                    false,
                    Some("missing_executor".to_string()),
                    Some(message),
                )
                .await?;
            self.record_stage_observation(
                &claimed,
                if failed.is_some() {
                    RuntimeStageOutcome::Failed
                } else {
                    RuntimeStageOutcome::Cancelled
                },
                None,
                None,
                Some(
                    if failed.is_some() {
                        "missing_executor"
                    } else {
                        "lease_lost"
                    }
                    .to_string(),
                ),
            );
            return Ok(true);
        };

        self.record_stage_observation(&claimed, RuntimeStageOutcome::Started, None, None, None);
        let stage_started = Instant::now();
        let execution = executor.execute(claimed.clone());
        tokio::pin!(execution);
        let renewal_interval = Duration::from_millis(
            u64::try_from(
                (self.config.lease_duration.as_millis() / 3)
                    .max(1)
                    .min(30_000),
            )
            .unwrap_or(30_000),
        );
        let execution_result = loop {
            tokio::select! {
                result = &mut execution => break result,
                _ = tokio::time::sleep(renewal_interval) => {
                    let renewed = self.store.renew_stage_lease(
                        &lease,
                        self.config.lease_duration.as_millis() as u64,
                    ).await?;
                    if !renewed {
                        return Err(anyhow!(
                            "Lost lease for runtime stage {} while it was executing",
                            lease.stage_id
                        ));
                    }
                    self.record_heartbeat(
                        "running",
                        Some((claimed.job.id.clone(), claimed.stage.id.clone())),
                    ).await?;
                }
            }
        };
        match execution_result {
            Ok(outcome) => {
                let output_artifact_count = outcome.output_artifact_ids.len();
                let completed = self
                    .store
                    .complete_stage(&lease, outcome.output_artifact_ids)
                    .await?;
                self.record_stage_observation(
                    &claimed,
                    if completed.is_some() {
                        RuntimeStageOutcome::Completed
                    } else {
                        RuntimeStageOutcome::Cancelled
                    },
                    Some(stage_started.elapsed().as_secs_f64() * 1000.0),
                    completed.as_ref().map(|_| output_artifact_count),
                    completed.is_none().then(|| "lease_lost".to_string()),
                );
                self.record_heartbeat("idle", None).await?;
            }
            Err(err) => {
                let message = err.to_string();
                self.health.record_error(message.clone());
                let failed = self
                    .store
                    .fail_stage(
                        &lease,
                        true,
                        Some("executor_failed".to_string()),
                        Some(message),
                    )
                    .await?;
                self.record_stage_observation(
                    &claimed,
                    if failed.is_some() {
                        RuntimeStageOutcome::Failed
                    } else {
                        RuntimeStageOutcome::Cancelled
                    },
                    Some(stage_started.elapsed().as_secs_f64() * 1000.0),
                    None,
                    Some(
                        if failed.is_some() {
                            "executor_failed"
                        } else {
                            "lease_lost"
                        }
                        .to_string(),
                    ),
                );
                self.record_heartbeat("idle", None).await?;
            }
        }

        Ok(true)
    }

    async fn run_maintenance_if_due(&self) -> anyhow::Result<()> {
        let now = Instant::now();
        let should_run = {
            let mut guard = self
                .last_maintenance_at
                .write()
                .unwrap_or_else(|poison| poison.into_inner());
            let due = guard.is_none_or(|last| {
                now.saturating_duration_since(last) >= self.config.maintenance_interval
            });
            if due {
                *guard = Some(now);
            }
            due
        };
        if !should_run {
            return Ok(());
        }

        self.store
            .reconcile_inconsistent_states()
            .await
            .context("Failed to reconcile durable runtime state")?;
        self.store
            .recover_expired_stage_leases()
            .await
            .context("Failed to recover expired runtime stage leases")?;
        Ok(())
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
        let drain = self.drain.clone();
        health.mark_running();
        let runner = self.clone();
        let handle = tokio::spawn(async move {
            info!(worker_id = %runner.config.worker_id, "Batch runtime worker started");
            loop {
                if runner.drain.is_draining() {
                    break;
                }

                let iteration = runner.run_once();
                tokio::pin!(iteration);
                let result = tokio::select! {
                    result = &mut iteration => result,
                    _ = runner.drain.wait() => iteration.await,
                };
                let should_pause = match result {
                    Ok(true) => false,
                    Ok(false) => true,
                    Err(err) => {
                        error!(worker_id = %runner.config.worker_id, error = %err, "Batch runtime worker iteration failed");
                        runner.health.record_error(err.to_string());
                        true
                    }
                };

                if runner.drain.is_draining() {
                    break;
                }
                if should_pause {
                    tokio::select! {
                        _ = tokio::time::sleep(runner.config.poll_interval) => {}
                        _ = runner.drain.wait() => break,
                    }
                }
            }
            if let Err(err) = runner.record_heartbeat("draining", None).await {
                error!(worker_id = %runner.config.worker_id, error = %err, "Failed to record drained batch worker heartbeat");
                runner.health.record_error(err.to_string());
            }
            runner.health.mark_stopped();
            debug!(worker_id = %runner.config.worker_id, "Batch runtime worker stopped");
        });
        BatchWorkerSupervisor {
            handle: Some(handle),
            health,
            drain,
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
                diagnostic_json: serde_json::json!({
                    "capabilities": self.config.capabilities.clone(),
                    "model_ids": self.config.model_ids.clone(),
                    "stage_kinds": self.config.stage_kinds.clone(),
                }),
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
    handle: Option<JoinHandle<()>>,
    health: BatchWorkerHealth,
    drain: BatchWorkerDrain,
}

impl BatchWorkerSupervisor {
    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    pub fn drain_handle(&self) -> BatchWorkerDrain {
        self.drain.clone()
    }

    pub fn begin_drain(&self) {
        self.drain.begin();
    }

    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        self.begin_drain();
        self.handle
            .take()
            .expect("batch worker supervisor handle must exist")
            .await
            .map_err(|err| anyhow!("Batch worker task join failed: {err}"))?;
        Ok(())
    }
}

impl Drop for BatchWorkerSupervisor {
    fn drop(&mut self) {
        self.begin_drain();
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
    let mut normalized = values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
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

    struct BlockingExecutor {
        started: Arc<Notify>,
        release: Arc<Notify>,
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

    #[async_trait]
    impl StageExecutor for BlockingExecutor {
        fn stage_kind(&self) -> &'static str {
            "fake_stage"
        }

        async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(StageExecutionOutcome {
                output_artifact_ids: vec!["blocking-artifact".to_string()],
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
        create_queued_stage(store, max_attempts, "fake_stage").await
    }

    async fn create_queued_stage(
        store: &BatchRuntimeStore,
        max_attempts: u32,
        stage_kind: &str,
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
                stage_kind: stage_kind.to_string(),
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
    async fn runner_only_claims_registered_stage_kinds_and_reports_configuration() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_stage(&store, 1, "unregistered_stage")
            .await
            .expect("stage");
        let health = BatchWorkerHealth::new("worker-test");
        let mut config = BatchWorkerConfig::local("worker-test");
        config.capabilities = vec![" test ".to_string(), "test".to_string()];
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                fail_first: false,
            })],
            config,
            health.clone(),
        );

        assert!(!runner.run_once().await.expect("run once"));

        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Queued);
        let snapshot = health.snapshot();
        assert_eq!(snapshot.configured_capabilities, vec!["test"]);
        assert_eq!(snapshot.configured_stage_kinds, vec!["fake_stage"]);
        let heartbeat = store
            .get_worker_heartbeat("worker-test")
            .await
            .expect("heartbeat")
            .expect("heartbeat exists");
        assert_eq!(heartbeat.diagnostic_json["capabilities"], json!(["test"]));
        assert_eq!(
            heartbeat.diagnostic_json["stage_kinds"],
            json!(["fake_stage"])
        );
    }

    #[tokio::test]
    async fn cancellation_during_execution_cannot_be_overwritten() {
        let store = build_store();
        let (job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(BlockingExecutor {
                started: started.clone(),
                release: release.clone(),
            })],
            BatchWorkerConfig::local("worker-test"),
            BatchWorkerHealth::new("worker-test"),
        );
        let run = tokio::spawn(async move { runner.run_once().await });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .expect("executor should start");

        store
            .cancel_job(&job_id, Some("cancel while executing".to_string()))
            .await
            .expect("cancel")
            .expect("cancelled job");
        release.notify_one();
        assert!(run.await.expect("runner join").expect("run once"));

        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Cancelled);
        assert!(stage.output_artifact_ids.is_empty());
        assert_eq!(stage.lease_expires_at, None);
    }

    #[tokio::test]
    async fn shutdown_drain_waits_for_active_iteration_while_renewing_lease() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut config = BatchWorkerConfig::local("worker-test");
        config.lease_duration = Duration::from_millis(120);
        let supervisor = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(BlockingExecutor {
                started: started.clone(),
                release: release.clone(),
            })],
            config,
            BatchWorkerHealth::new("worker-test"),
        )
        .spawn();
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .expect("executor should start");

        let running = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert!(running.lease_expires_at.is_some());

        let mut shutdown = tokio::spawn(supervisor.shutdown());
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut shutdown)
                .await
                .is_err()
        );
        tokio::time::sleep(Duration::from_millis(180)).await;
        let renewed = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(renewed.status, RuntimeStageStatus::Running);
        assert!(renewed.lease_expires_at.is_some());
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(2), &mut shutdown)
            .await
            .expect("shutdown should finish")
            .expect("shutdown join")
            .expect("worker shutdown");

        let completed = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(completed.status, RuntimeStageStatus::Completed);
        assert_eq!(completed.worker_id, None);
        assert_eq!(completed.lease_expires_at, None);
        assert_eq!(completed.output_artifact_ids, vec!["blocking-artifact"]);
        let heartbeat = store
            .get_worker_heartbeat("worker-test")
            .await
            .expect("heartbeat")
            .expect("heartbeat exists");
        assert_eq!(heartbeat.status, "draining");
        assert_eq!(heartbeat.current_stage_id, None);
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

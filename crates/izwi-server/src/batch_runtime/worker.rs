use super::{
    store::{
        BatchRuntimeStore, NewStageOutputArtifact, RegisteredWorkerHeartbeatUpdate,
        StageClaimFilter,
    },
    types::{
        ClaimedStage, QueueClass, RuntimeArtifact, RuntimeJobKind, RuntimeWorkerHeartbeatDetails,
        RuntimeWorkerRegistration, WorkerResourceCapacity, WORKER_HEARTBEAT_DETAILS_VERSION,
        WORKER_REGISTRATION_VERSION,
    },
};
use crate::ids::new_uuid;
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
    pub instance_id: String,
    pub queue_names: Vec<String>,
    pub capabilities: Vec<String>,
    pub model_ids: Vec<String>,
    pub stage_kinds: Vec<String>,
    pub resources: WorkerResourceCapacity,
    pub draining: bool,
    pub poll_interval: Duration,
    pub lease_duration: Duration,
    pub maintenance_interval: Duration,
    pub execution_timeout: Option<Duration>,
    pub drain_timeout: Duration,
}

impl BatchWorkerConfig {
    pub fn local(worker_id: impl Into<String>) -> Self {
        let worker_id = worker_id.into();
        Self {
            worker_id,
            instance_id: new_uuid(),
            queue_names: vec!["batch".to_string()],
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            resources: WorkerResourceCapacity::default(),
            draining: false,
            poll_interval: Duration::from_millis(250),
            lease_duration: Duration::from_secs(60),
            maintenance_interval: Duration::from_secs(30),
            execution_timeout: None,
            drain_timeout: Duration::from_secs(20),
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
    pub instance_id: String,
    pub running: bool,
    pub last_heartbeat_at: u64,
    pub last_claimed_stage_id: Option<String>,
    pub last_error: Option<String>,
    pub configured_capabilities: Vec<String>,
    pub configured_stage_kinds: Vec<String>,
    pub configured_queue_names: Vec<String>,
    pub configured_resources: WorkerResourceCapacity,
}

#[derive(Debug, Clone)]
pub struct BatchWorkerHealth {
    inner: Arc<RwLock<BatchWorkerHealthInner>>,
}

#[derive(Debug)]
struct BatchWorkerHealthInner {
    worker_id: String,
    instance_id: String,
    running: bool,
    last_heartbeat_at: u64,
    last_claimed_stage_id: Option<String>,
    last_error: Option<String>,
    configured_capabilities: Vec<String>,
    configured_stage_kinds: Vec<String>,
    configured_queue_names: Vec<String>,
    configured_resources: WorkerResourceCapacity,
}

impl BatchWorkerHealth {
    pub fn new(worker_id: impl Into<String>) -> Self {
        let worker_id = worker_id.into();
        Self {
            inner: Arc::new(RwLock::new(BatchWorkerHealthInner {
                worker_id,
                instance_id: String::new(),
                running: false,
                last_heartbeat_at: now_secs(),
                last_claimed_stage_id: None,
                last_error: None,
                configured_capabilities: Vec::new(),
                configured_stage_kinds: Vec::new(),
                configured_queue_names: Vec::new(),
                configured_resources: WorkerResourceCapacity::default(),
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
            inner.worker_id = config.worker_id.clone();
            inner.instance_id = config.instance_id.clone();
            inner.configured_capabilities = config.capabilities.clone();
            inner.configured_stage_kinds = config.stage_kinds.clone();
            inner.configured_queue_names = config.queue_names.clone();
            inner.configured_resources = config.resources.clone();
        });
    }

    pub fn snapshot(&self) -> BatchWorkerSnapshot {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        BatchWorkerSnapshot {
            worker_id: guard.worker_id.clone(),
            instance_id: guard.instance_id.clone(),
            running: guard.running,
            last_heartbeat_at: guard.last_heartbeat_at,
            last_claimed_stage_id: guard.last_claimed_stage_id.clone(),
            last_error: guard.last_error.clone(),
            configured_capabilities: guard.configured_capabilities.clone(),
            configured_stage_kinds: guard.configured_stage_kinds.clone(),
            configured_queue_names: guard.configured_queue_names.clone(),
            configured_resources: guard.configured_resources.clone(),
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageCancellationReason {
    ExecutionDeadline,
    DrainDeadline,
    LeaseLost,
    WorkerShutdown,
}

impl StageCancellationReason {
    fn as_error_code(self) -> &'static str {
        match self {
            Self::ExecutionDeadline => "execution_deadline",
            Self::DrainDeadline => "drain_deadline",
            Self::LeaseLost => "lease_lost",
            Self::WorkerShutdown => "worker_shutdown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageCancellationSignal {
    inner: Arc<StageCancellationInner>,
}

#[derive(Debug)]
struct StageCancellationInner {
    cancelled: AtomicBool,
    reason: RwLock<Option<StageCancellationReason>>,
    notify: Notify,
}

impl StageCancellationSignal {
    fn new() -> Self {
        Self {
            inner: Arc::new(StageCancellationInner {
                cancelled: AtomicBool::new(false),
                reason: RwLock::new(None),
                notify: Notify::new(),
            }),
        }
    }

    pub fn cancel(&self, reason: StageCancellationReason) {
        let mut cancellation_reason = self
            .inner
            .reason
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if cancellation_reason.is_none() {
            *cancellation_reason = Some(reason);
            self.inner.cancelled.store(true, Ordering::Release);
            drop(cancellation_reason);
            self.inner.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub fn reason(&self) -> Option<StageCancellationReason> {
        *self
            .inner
            .reason
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub async fn cancelled(&self) -> StageCancellationReason {
        loop {
            let notified = self.inner.notify.notified();
            if let Some(reason) = self.reason() {
                return reason;
            }
            notified.await;
        }
    }

    fn same_signal(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Clone)]
pub struct StageExecutionContext {
    claimed: ClaimedStage,
    lease: super::types::StageLease,
    cancellation: StageCancellationSignal,
    deadline: Option<Instant>,
    started_at: Instant,
    store: Arc<BatchRuntimeStore>,
    runtime_observer: Option<Arc<RuntimeService>>,
}

impl StageExecutionContext {
    pub fn claimed(&self) -> &ClaimedStage {
        &self.claimed
    }

    pub fn lease(&self) -> &super::types::StageLease {
        &self.lease
    }

    pub fn cancellation(&self) -> StageCancellationSignal {
        self.cancellation.clone()
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    pub fn check_cancelled(&self) -> anyhow::Result<()> {
        if let Some(reason) = self.cancellation.reason() {
            return Err(anyhow!(
                "Stage execution cancelled: {}",
                reason.as_error_code()
            ));
        }
        Ok(())
    }

    pub async fn record_progress(&self, progress: serde_json::Value) -> anyhow::Result<()> {
        self.check_cancelled()?;
        if !self
            .store
            .update_stage_progress(&self.lease, progress)
            .await?
        {
            self.cancellation.cancel(StageCancellationReason::LeaseLost);
            return Err(anyhow!("Lost stage lease while recording progress"));
        }
        self.record_runtime_observation(RuntimeStageOutcome::Observed, None);
        Ok(())
    }

    pub async fn ensure_active(&self) -> anyhow::Result<()> {
        self.check_cancelled()?;
        if !self.store.stage_lease_is_active(&self.lease).await? {
            self.cancellation.cancel(StageCancellationReason::LeaseLost);
            return Err(anyhow!("Stage attempt no longer owns an active lease"));
        }
        Ok(())
    }

    pub async fn publish_output_artifact(
        &self,
        artifact: NewStageOutputArtifact,
    ) -> anyhow::Result<RuntimeArtifact> {
        self.ensure_active().await?;
        match self
            .store
            .publish_stage_output_artifact(&self.lease, artifact)
            .await?
        {
            Some(artifact) => Ok(artifact),
            None => {
                self.cancellation.cancel(StageCancellationReason::LeaseLost);
                Err(anyhow!(
                    "Stage attempt lost ownership before artifact publication"
                ))
            }
        }
    }

    pub fn record_runtime_observation(
        &self,
        outcome: RuntimeStageOutcome,
        error_kind: Option<String>,
    ) {
        let Some(runtime) = self.runtime_observer.as_ref() else {
            return;
        };
        let mut observation =
            RuntimeStageObservation::new(stage_observation_context(&self.claimed), outcome)
                .with_total_ms(self.started_at.elapsed().as_secs_f64() * 1_000.0);
        if let Some(error_kind) = error_kind {
            observation = observation.with_error_kind(error_kind);
        }
        runtime.record_stage_observation(observation);
    }
}

enum StageExecutionResolution {
    Finished(anyhow::Result<StageExecutionOutcome>),
    Cancelled(StageCancellationReason),
}

struct ActiveExecutionGuard {
    slot: Arc<RwLock<Option<StageCancellationSignal>>>,
    cancellation: StageCancellationSignal,
}

impl ActiveExecutionGuard {
    fn new(
        slot: Arc<RwLock<Option<StageCancellationSignal>>>,
        cancellation: StageCancellationSignal,
    ) -> Self {
        *slot.write().unwrap_or_else(|poison| poison.into_inner()) = Some(cancellation.clone());
        Self { slot, cancellation }
    }
}

impl Drop for ActiveExecutionGuard {
    fn drop(&mut self) {
        let mut active = self
            .slot
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        if active
            .as_ref()
            .is_some_and(|current| current.same_signal(&self.cancellation))
        {
            *active = None;
        }
    }
}

#[async_trait]
pub trait StageExecutor: Send + Sync {
    fn stage_kind(&self) -> &'static str;

    async fn execute(&self, claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome>;

    async fn execute_with_context(
        &self,
        context: StageExecutionContext,
    ) -> anyhow::Result<StageExecutionOutcome> {
        self.execute(context.claimed.clone()).await
    }
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
    active_execution: Arc<RwLock<Option<StageCancellationSignal>>>,
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
            active_execution: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_runtime_observer(mut self, runtime: Arc<RuntimeService>) -> Self {
        self.runtime_observer = Some(runtime);
        self
    }

    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    fn cancel_active_execution(&self, reason: StageCancellationReason) {
        if let Some(active) = self
            .active_execution
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
        {
            active.cancel(reason);
        }
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
        if self.drain.is_draining() {
            let relinquished = self
                .store
                .relinquish_stage_lease(
                    &lease,
                    "worker_draining",
                    "Worker began draining before execution",
                )
                .await?;
            self.record_stage_observation(
                &claimed,
                if relinquished
                    .as_ref()
                    .is_some_and(|stage| stage.status == super::types::RuntimeStageStatus::Retrying)
                {
                    RuntimeStageOutcome::Retried
                } else {
                    RuntimeStageOutcome::Cancelled
                },
                None,
                None,
                Some("worker_draining".to_string()),
            );
            self.record_heartbeat("draining", None).await?;
            return Ok(true);
        }

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
        let cancellation = StageCancellationSignal::new();
        let deadline = self
            .config
            .execution_timeout
            .map(|timeout| stage_started + timeout);
        let context = StageExecutionContext {
            claimed: claimed.clone(),
            lease: lease.clone(),
            cancellation: cancellation.clone(),
            deadline,
            started_at: stage_started,
            store: self.store.clone(),
            runtime_observer: self.runtime_observer.clone(),
        };
        let active_execution =
            ActiveExecutionGuard::new(self.active_execution.clone(), cancellation.clone());
        let execution = executor.execute_with_context(context);
        tokio::pin!(execution);
        let deadline_wait = async move {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline_wait);
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
                result = &mut execution => {
                    break match cancellation.reason() {
                        Some(reason) => StageExecutionResolution::Cancelled(reason),
                        None => StageExecutionResolution::Finished(result),
                    };
                },
                reason = cancellation.cancelled() => {
                    break StageExecutionResolution::Cancelled(reason);
                },
                _ = &mut deadline_wait => {
                    cancellation.cancel(StageCancellationReason::ExecutionDeadline);
                    break StageExecutionResolution::Cancelled(
                        StageCancellationReason::ExecutionDeadline,
                    );
                },
                _ = tokio::time::sleep(renewal_interval) => {
                    let renewed = self.store.renew_stage_lease(
                        &lease,
                        self.config.lease_duration.as_millis() as u64,
                    ).await?;
                    if !renewed {
                        cancellation.cancel(StageCancellationReason::LeaseLost);
                        break StageExecutionResolution::Cancelled(
                            StageCancellationReason::LeaseLost,
                        );
                    }
                    self.record_heartbeat(
                        "running",
                        Some((claimed.job.id.clone(), claimed.stage.id.clone())),
                    ).await?;
                }
            }
        };
        drop(active_execution);
        match execution_result {
            StageExecutionResolution::Finished(Ok(outcome)) => {
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
            StageExecutionResolution::Finished(Err(err)) => {
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
            StageExecutionResolution::Cancelled(reason) => {
                let relinquished = if reason == StageCancellationReason::LeaseLost {
                    None
                } else {
                    self.store
                        .relinquish_stage_lease(
                            &lease,
                            reason.as_error_code(),
                            format!("Stage execution cancelled: {}", reason.as_error_code()),
                        )
                        .await?
                };
                let outcome = match relinquished.as_ref().map(|stage| stage.status) {
                    Some(super::types::RuntimeStageStatus::Retrying) => {
                        RuntimeStageOutcome::Retried
                    }
                    Some(super::types::RuntimeStageStatus::Failed) => RuntimeStageOutcome::Failed,
                    _ => RuntimeStageOutcome::Cancelled,
                };
                self.record_stage_observation(
                    &claimed,
                    outcome,
                    Some(stage_started.elapsed().as_secs_f64() * 1000.0),
                    None,
                    Some(reason.as_error_code().to_string()),
                );
                self.record_heartbeat(
                    if self.drain.is_draining() {
                        "draining"
                    } else {
                        "idle"
                    },
                    None,
                )
                .await?;
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
        let drain_timeout = self.config.drain_timeout;
        let shutdown_timeout = drain_timeout.saturating_add(Duration::from_secs(2));
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
                    _ = runner.drain.wait() => {
                        match tokio::time::timeout(drain_timeout, &mut iteration).await {
                            Ok(result) => result,
                            Err(_) => {
                                runner.cancel_active_execution(StageCancellationReason::DrainDeadline);
                                match tokio::time::timeout(Duration::from_secs(1), &mut iteration).await {
                                    Ok(result) => result,
                                    Err(_) => Err(anyhow!("Batch worker drain cancellation did not settle")),
                                }
                            }
                        }
                    },
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
            if let Err(err) = runner.record_heartbeat("drained", None).await {
                error!(worker_id = %runner.config.worker_id, error = %err, "Failed to record drained batch worker heartbeat");
                runner.health.record_error(err.to_string());
            }
            runner.health.mark_stopped();
            if let Err(err) = runner.record_heartbeat("stopped", None).await {
                error!(worker_id = %runner.config.worker_id, error = %err, "Failed to record stopped batch worker heartbeat");
                runner.health.record_error(err.to_string());
            }
            debug!(worker_id = %runner.config.worker_id, "Batch runtime worker stopped");
        });
        BatchWorkerSupervisor {
            handle: Some(handle),
            health,
            drain,
            shutdown_timeout,
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
        let health = self.health.snapshot();
        let queue_classes = self
            .config
            .queue_names
            .iter()
            .filter_map(|queue| QueueClass::from_db_value(queue))
            .collect::<Vec<_>>();
        let registration = RuntimeWorkerRegistration {
            version: WORKER_REGISTRATION_VERSION,
            worker_id: self.config.worker_id.clone(),
            instance_id: self.config.instance_id.clone(),
            queue_classes: if queue_classes.is_empty() {
                vec![QueueClass::Batch]
            } else {
                queue_classes
            },
            capabilities: self.config.capabilities.clone(),
            model_ids: self.config.model_ids.clone(),
            stage_kinds: self.config.stage_kinds.clone(),
            resources: self.config.resources.clone(),
            software_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let active_lease_ids = current_stage_id.clone().into_iter().collect::<Vec<_>>();
        let available_slots = if self.drain.is_draining() {
            0
        } else {
            self.config
                .resources
                .concurrency_slots
                .saturating_sub(if active_lease_ids.is_empty() { 0 } else { 1 })
        };
        let details = RuntimeWorkerHeartbeatDetails {
            version: WORKER_HEARTBEAT_DETAILS_VERSION,
            available_slots,
            active_lease_ids,
            last_error: health.last_error.clone(),
            health_json: serde_json::json!({
                "running": health.running,
                "draining": self.drain.is_draining(),
                "last_claimed_stage_id": health.last_claimed_stage_id,
            }),
        };
        self.store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration,
                status: status.to_string(),
                current_job_id,
                current_stage_id,
                details,
                diagnostic_json: serde_json::json!({
                    "capabilities": self.config.capabilities,
                    "model_ids": self.config.model_ids,
                    "stage_kinds": self.config.stage_kinds,
                    "instance_id": self.config.instance_id,
                    "resources": self.config.resources,
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
        filter.resources = self.config.resources.clone();
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
            RuntimeStageObservation::new(stage_observation_context(claimed), outcome);
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
}

pub struct BatchWorkerSupervisor {
    handle: Option<JoinHandle<()>>,
    health: BatchWorkerHealth,
    drain: BatchWorkerDrain,
    shutdown_timeout: Duration,
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
        let mut handle = self
            .handle
            .take()
            .expect("batch worker supervisor handle must exist");
        match tokio::time::timeout(self.shutdown_timeout, &mut handle).await {
            Ok(joined) => joined.map_err(|err| anyhow!("Batch worker task join failed: {err}")),
            Err(_) => {
                handle.abort();
                let _ = handle.await;
                Err(anyhow!(
                    "Batch worker shutdown exceeded its bounded deadline"
                ))
            }
        }
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

    struct ContextProgressExecutor;

    struct ContextBlockingExecutor {
        started: Arc<Notify>,
        cancellation: Arc<RwLock<Option<StageCancellationSignal>>>,
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

    #[async_trait]
    impl StageExecutor for ContextProgressExecutor {
        fn stage_kind(&self) -> &'static str {
            "fake_stage"
        }

        async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
            anyhow::bail!("runner did not invoke context-aware stage execution")
        }

        async fn execute_with_context(
            &self,
            context: StageExecutionContext,
        ) -> anyhow::Result<StageExecutionOutcome> {
            assert_eq!(context.claimed().stage.id, context.lease().stage_id);
            assert_eq!(context.claimed().stage.attempt_count, 1);
            assert_eq!(context.lease().attempt_count, 1);
            assert!(context.lease().attempt_token.is_some());
            context
                .record_progress(json!({"completed_units": 1, "total_units": 2}))
                .await?;
            Ok(StageExecutionOutcome::empty())
        }
    }

    #[async_trait]
    impl StageExecutor for ContextBlockingExecutor {
        fn stage_kind(&self) -> &'static str {
            "fake_stage"
        }

        async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
            anyhow::bail!("runner did not invoke context-aware stage execution")
        }

        async fn execute_with_context(
            &self,
            context: StageExecutionContext,
        ) -> anyhow::Result<StageExecutionOutcome> {
            *self
                .cancellation
                .write()
                .unwrap_or_else(|poison| poison.into_inner()) = Some(context.cancellation());
            self.started.notify_one();
            std::future::pending::<anyhow::Result<StageExecutionOutcome>>().await
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
    async fn context_aware_execution_records_attempt_scoped_progress() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(ContextProgressExecutor)],
            BatchWorkerConfig::local("worker-test"),
            BatchWorkerHealth::new("worker-test"),
        );

        assert!(runner.run_once().await.expect("run once"));

        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Completed);
        assert_eq!(
            stage.progress_json,
            Some(json!({"completed_units": 1, "total_units": 2}))
        );
    }

    #[tokio::test]
    async fn execution_deadline_cancels_context_and_relinquishes_lease() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
        let started = Arc::new(Notify::new());
        let cancellation = Arc::new(RwLock::new(None));
        let mut config = BatchWorkerConfig::local("worker-test");
        config.execution_timeout = Some(Duration::from_millis(50));
        let runner = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(ContextBlockingExecutor {
                started: started.clone(),
                cancellation: cancellation.clone(),
            })],
            config,
            BatchWorkerHealth::new("worker-test"),
        );

        let run = tokio::spawn(async move { runner.run_once().await });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .expect("executor should start");
        assert!(tokio::time::timeout(Duration::from_secs(2), run)
            .await
            .expect("runner should honor execution deadline")
            .expect("runner join")
            .expect("run once"));

        let signal = cancellation
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
            .expect("context cancellation signal");
        assert!(signal.is_cancelled());
        assert_eq!(
            signal.reason(),
            Some(StageCancellationReason::ExecutionDeadline)
        );
        let stage = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Retrying);
        assert_eq!(stage.worker_id, None);
        assert_eq!(stage.lease_expires_at, None);
        assert_eq!(stage.attempt_token, None);
        assert_eq!(stage.error_code.as_deref(), Some("execution_deadline"));
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
        assert_eq!(heartbeat.registration.capabilities, vec!["test"]);
        assert_eq!(heartbeat.registration.stage_kinds, vec!["fake_stage"]);
        assert_eq!(heartbeat.instance_id, heartbeat.registration.instance_id);
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
        assert_eq!(heartbeat.status, "stopped");
        assert_eq!(heartbeat.current_stage_id, None);
        assert_eq!(heartbeat.details.available_slots, 0);
        assert_eq!(
            heartbeat.details.health_json.get("running"),
            Some(&json!(false))
        );
    }

    #[tokio::test]
    async fn bounded_shutdown_relinquishes_lease_for_replacement_worker() {
        let store = build_store();
        let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let mut config = BatchWorkerConfig::local("worker-test");
        config.drain_timeout = Duration::from_millis(60);
        let supervisor = BatchWorkerRunner::new(
            store.clone(),
            vec![Arc::new(BlockingExecutor {
                started: started.clone(),
                release,
            })],
            config,
            BatchWorkerHealth::new("worker-test"),
        )
        .spawn();
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .expect("executor should start");

        let shutdown_started = Instant::now();
        tokio::time::timeout(Duration::from_secs(2), supervisor.shutdown())
            .await
            .expect("shutdown should remain bounded")
            .expect("worker shutdown");
        assert!(shutdown_started.elapsed() < Duration::from_secs(1));

        let relinquished = store
            .get_stage(&stage_id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(relinquished.status, RuntimeStageStatus::Retrying);
        assert_eq!(relinquished.worker_id, None);
        assert_eq!(relinquished.lease_expires_at, None);
        assert_eq!(relinquished.error_code.as_deref(), Some("drain_deadline"));

        let replacement = store
            .claim_next_stage("replacement-worker", 60_000)
            .await
            .expect("replacement claim")
            .expect("replacement should take over relinquished stage");
        assert_eq!(replacement.stage.id, stage_id);
        assert_eq!(
            replacement.stage.worker_id.as_deref(),
            Some("replacement-worker")
        );

        let heartbeat = store
            .get_worker_heartbeat("worker-test")
            .await
            .expect("heartbeat")
            .expect("heartbeat exists");
        assert_eq!(heartbeat.status, "stopped");
        assert_eq!(heartbeat.current_stage_id, None);
        assert_eq!(heartbeat.details.available_slots, 0);
        assert_eq!(
            heartbeat.details.health_json.get("running"),
            Some(&json!(false))
        );
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

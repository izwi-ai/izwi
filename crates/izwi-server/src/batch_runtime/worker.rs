use super::{
    store::{
        validate_stage_output_artifact_ids, BatchRuntimeStore, NewStageOutputArtifact,
        RegisteredWorkerHeartbeatUpdate, StageClaimFilter, DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT,
    },
    types::{
        ClaimedStage, QueueClass, RuntimeArtifact, RuntimeJobKind, RuntimeWorkerHeartbeatDetails,
        RuntimeWorkerRegistration, WorkerResourceCapacity, WORKER_HEARTBEAT_DETAILS_VERSION,
        WORKER_REGISTRATION_VERSION,
    },
};
use crate::artifact_store::ArtifactStore;
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
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{sync::Notify, task::JoinHandle};
use tracing::{debug, error, info, warn};

/// Every idle slot claims again after `idle_poll_interval` (sooner when this
/// process enqueues work), and they all report the same worker row. An unchanged heartbeat is re-sent at most this often; any
/// change (status, current stage, leases, slots, errors) is written at once.
/// Readiness treats a heartbeat as stale after 5 s by default
/// (`IZWI_BATCH_WORKER_HEARTBEAT_STALE_SECS`), so liveness stays fresh while
/// the shared database no longer takes a commit per slot per poll.
const HEARTBEAT_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// First retry delay after a lease renewal fails with a store error. It
/// doubles per consecutive failure, capped at the regular renewal interval.
const LEASE_RENEWAL_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Settlement after a claim is retried this many times in all. Each write is
/// fenced on the exact attempt token, so a retry after an error whose write
/// had in fact committed finds the attempt already settled and no-ops.
const SETTLEMENT_ATTEMPTS: u32 = 5;
const SETTLEMENT_RETRY_DELAY: Duration = Duration::from_millis(100);

fn lease_renewal_retry_delay(consecutive_failures: u32, renewal_interval: Duration) -> Duration {
    LEASE_RENEWAL_RETRY_DELAY
        .saturating_mul(1 << consecutive_failures.saturating_sub(1).min(8))
        .min(renewal_interval)
}

/// The last heartbeat this runner committed and when.
struct WrittenHeartbeat {
    update: RegisteredWorkerHeartbeatUpdate,
    written_at: Instant,
}

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
    /// How often a running stage checks its lease and cancellation state.
    pub poll_interval: Duration,
    /// How long an idle slot waits for work before claiming again. A stage
    /// enqueued through this process wakes it at once; this bounds the wait
    /// for work other processes enqueue and for delayed retries.
    pub idle_poll_interval: Duration,
    pub lease_duration: Duration,
    pub maintenance_interval: Duration,
    pub maintenance_batch_limit: usize,
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
            idle_poll_interval: Duration::from_secs(1),
            lease_duration: Duration::from_secs(60),
            maintenance_interval: Duration::from_secs(30),
            maintenance_batch_limit: DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT,
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
            self.inner.notify.notify_waiters();
        }
    }

    pub fn is_draining(&self) -> bool {
        self.inner.draining.load(Ordering::Acquire)
    }

    async fn wait(&self) {
        loop {
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StageExecutionOutcome {
    output_artifact_ids: Vec<String>,
}

impl StageExecutionOutcome {
    pub fn empty() -> Self {
        Self {
            output_artifact_ids: Vec::new(),
        }
    }

    pub fn try_new(output_artifact_ids: Vec<String>) -> anyhow::Result<Self> {
        validate_stage_output_artifact_ids(&output_artifact_ids)?;
        Ok(Self {
            output_artifact_ids,
        })
    }

    pub fn output_artifact_count(&self) -> usize {
        self.output_artifact_ids.len()
    }

    fn into_output_artifact_ids(self) -> Vec<String> {
        self.output_artifact_ids
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageCancellationReason {
    UserRequested,
    ExecutionDeadline,
    DrainDeadline,
    LeaseLost,
    WorkerShutdown,
}

impl StageCancellationReason {
    fn as_error_code(self) -> &'static str {
        match self {
            Self::UserRequested => "user_requested",
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
    /// A context for a stage claimed outside a worker loop.
    #[cfg(test)]
    pub(crate) fn for_test(store: Arc<BatchRuntimeStore>, claimed: ClaimedStage) -> Self {
        Self {
            lease: claimed.lease().expect("claimed stage has a lease"),
            claimed,
            cancellation: StageCancellationSignal::new(),
            deadline: None,
            started_at: Instant::now(),
            store,
            runtime_observer: None,
        }
    }

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
        match self.store.stage_lease_state(&self.lease).await? {
            Some(super::store::StageLeaseState::Active) => {}
            Some(
                super::store::StageLeaseState::CancellationRequested
                | super::store::StageLeaseState::ExecutionStopping,
            ) => {
                self.cancellation
                    .cancel(StageCancellationReason::UserRequested);
                return Err(anyhow!("Stage attempt is being cancelled"));
            }
            None => {
                self.cancellation.cancel(StageCancellationReason::LeaseLost);
                return Err(anyhow!("Stage attempt no longer owns an active lease"));
            }
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

#[derive(Clone)]
struct ActiveExecution {
    stage_id: String,
    cancellation: StageCancellationSignal,
    weight: u32,
    memory_bytes: u64,
}

struct ActiveExecutionGuard {
    slots: Arc<RwLock<HashMap<String, ActiveExecution>>>,
    attempt_id: String,
}

impl Drop for ActiveExecutionGuard {
    fn drop(&mut self) {
        self.slots
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&self.attempt_id);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("speech_stage_yield: committed segment boundary")]
pub struct SpeechStageYield;

/// A stage failure that would recur on every attempt (an invalid request:
/// a 4xx the client must fix). It fails the stage without a retry
/// (TPIPE-27).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DeterministicStageFailure(pub String);

#[async_trait]
pub trait StageExecutor: Send + Sync {
    fn stage_kind(&self) -> &'static str;

    async fn execute(&self, claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome>;

    async fn maintenance(&self) -> anyhow::Result<()> {
        Ok(())
    }

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
    active_executions: Arc<RwLock<HashMap<String, ActiveExecution>>>,
    claim_lock: Arc<tokio::sync::Mutex<()>>,
    last_heartbeat: Arc<tokio::sync::Mutex<Option<WrittenHeartbeat>>>,
    artifact_store: Option<Arc<ArtifactStore>>,
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
            artifact_store: None,
            last_maintenance_at: Arc::new(RwLock::new(None)),
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            claim_lock: Arc::new(tokio::sync::Mutex::new(())),
            last_heartbeat: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub fn with_runtime_observer(mut self, runtime: Arc<RuntimeService>) -> Self {
        self.runtime_observer = Some(runtime);
        self
    }

    pub fn with_artifact_store(mut self, artifact_store: Arc<ArtifactStore>) -> Self {
        self.artifact_store = Some(artifact_store);
        self
    }

    pub fn health(&self) -> BatchWorkerHealth {
        self.health.clone()
    }

    fn cancel_active_execution(&self, reason: StageCancellationReason) {
        for active in self
            .active_executions
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .values()
        {
            active.cancellation.cancel(reason);
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
        // Serialize claim + local reservation, not execution. Every claim sees
        // the remaining aggregate capacity, including weighted stages.
        let claim_lock = self.claim_lock.lock().await;
        let claim_filter = self.claim_filter();
        if self.drain.is_draining() || claim_filter.resources.concurrency_slots == 0 {
            return Ok(false);
        }

        // The store stamps the lease after this instant, so a local deadline
        // measured from here never outlives the durable one.
        let mut lease_deadline = Instant::now() + self.config.lease_duration;
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

        let cancellation = StageCancellationSignal::new();
        let attempt_id = claimed
            .stage
            .attempt_token
            .clone()
            .ok_or_else(|| anyhow!("Claimed stage is missing attempt identity"))?;
        let hints = &claimed.stage.resource_hints;
        self.active_executions
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(
                attempt_id.clone(),
                ActiveExecution {
                    stage_id: claimed.stage.id.clone(),
                    cancellation: cancellation.clone(),
                    weight: hints.concurrency_weight.max(1),
                    memory_bytes: hints.min_memory_bytes.unwrap_or(0),
                },
            );
        let _active_execution = ActiveExecutionGuard {
            slots: self.active_executions.clone(),
            attempt_id,
        };
        drop(claim_lock);
        self.health.record_claim(claimed.stage.id.clone());
        self.record_stage_observation(&claimed, RuntimeStageOutcome::Claimed, None, None, None);
        // The claim is durable now. A missed heartbeat only delays liveness
        // reporting; returning here would strand the claimed stage until its
        // lease expired and burn an attempt.
        if let Err(error) = self
            .record_heartbeat(
                "running",
                Some((claimed.job.id.clone(), claimed.stage.id.clone())),
            )
            .await
        {
            self.health.record_error(format!(
                "Failed to record batch worker heartbeat after claim: {error:#}"
            ));
        }
        let lease = claimed
            .lease()
            .ok_or_else(|| anyhow!("Claimed stage is missing worker lease ownership"))?;
        if self.drain.is_draining() {
            let relinquished = self
                .settle_with_retry("relinquish draining stage", || {
                    self.store.relinquish_stage_lease(
                        &lease,
                        "worker_draining",
                        "Worker began draining before execution",
                    )
                })
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
                .settle_with_retry("fail stage without executor", || {
                    self.store.fail_stage(
                        &lease,
                        false,
                        Some("missing_executor".to_string()),
                        Some(message.clone()),
                    )
                })
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
            u64::try_from((self.config.lease_duration.as_millis() / 3).clamp(1, 30_000))
                .unwrap_or(30_000),
        );
        let mut renewal_tick = tokio::time::interval(renewal_interval);
        renewal_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        renewal_tick.tick().await;
        let cancellation_poll_interval = self.config.poll_interval.max(Duration::from_millis(10));
        let mut cancellation_tick = tokio::time::interval(cancellation_poll_interval);
        cancellation_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        cancellation_tick.tick().await;
        let mut awaiting_teardown = false;
        let mut ownership_lost = false;
        let mut renewal_failures = 0_u32;
        let execution_result = loop {
            tokio::select! {
                result = &mut execution => {
                    break result;
                },
                _reason = cancellation.cancelled(), if !awaiting_teardown => {
                    // Cancellation requests are not teardown proof. Keep the
                    // executor future and its capacity guard alive until the
                    // implementation actually resolves.
                    awaiting_teardown = true;
                },
                _ = &mut deadline_wait, if !awaiting_teardown => {
                    cancellation.cancel(StageCancellationReason::ExecutionDeadline);
                    awaiting_teardown = true;
                },
                _ = renewal_tick.tick(), if !ownership_lost => {
                    let renewal_started = Instant::now();
                    match self.store.renew_stage_lease(
                        &lease,
                        self.config.lease_duration.as_millis() as u64,
                    ).await {
                        Ok(true) => {
                            lease_deadline = renewal_started + self.config.lease_duration;
                            renewal_failures = 0;
                            if let Err(error) = self.record_heartbeat(
                                "running",
                                Some((claimed.job.id.clone(), claimed.stage.id.clone())),
                            ).await {
                                self.health.record_error(format!(
                                    "Failed to record batch worker heartbeat while execution remained active: {error:#}"
                                ));
                            }
                        }
                        Ok(false) => {
                            ownership_lost = true;
                            awaiting_teardown = true;
                            cancellation.cancel(StageCancellationReason::LeaseLost);
                        }
                        Err(error) => {
                            // A store error is not an answer about ownership:
                            // the durable lease is still ours until it
                            // expires. Retry the exact renewal with backoff,
                            // and only request cooperative cancellation once
                            // the local lease deadline has passed, when a
                            // recovery sweep may already have reclaimed it.
                            renewal_failures = renewal_failures.saturating_add(1);
                            warn!(
                                worker_id = %self.config.worker_id,
                                stage_id = %lease.stage_id,
                                consecutive_failures = renewal_failures,
                                error = %format!("{error:#}"),
                                "Failed to renew active batch stage lease; retrying"
                            );
                            self.health.record_error(format!(
                                "Failed to renew active batch stage lease: {error:#}"
                            ));
                            if Instant::now() >= lease_deadline {
                                awaiting_teardown = true;
                                cancellation.cancel(StageCancellationReason::LeaseLost);
                            }
                            renewal_tick.reset_after(lease_renewal_retry_delay(
                                renewal_failures,
                                renewal_interval,
                            ));
                        }
                    }
                },
                _ = cancellation_tick.tick() => {
                    match self.store.stage_lease_state(&lease).await {
                        Ok(Some(super::store::StageLeaseState::Active)) => {}
                        Ok(Some(super::store::StageLeaseState::CancellationRequested)) => {
                            awaiting_teardown = true;
                            cancellation.cancel(StageCancellationReason::UserRequested);
                            if let Err(error) = self.store.mark_stage_execution_stopping(&lease).await {
                                self.health.record_error(format!(
                                    "Failed to mark batch stage execution stopping: {error:#}"
                                ));
                            }
                        }
                        Ok(Some(super::store::StageLeaseState::ExecutionStopping)) => {
                            awaiting_teardown = true;
                            cancellation.cancel(StageCancellationReason::UserRequested);
                        }
                        Ok(None) => {
                            ownership_lost = true;
                            awaiting_teardown = true;
                            cancellation.cancel(StageCancellationReason::LeaseLost);
                        }
                        Err(error) => {
                            // Unknown, not lost: the next poll retries.
                            self.health.record_error(format!(
                                "Failed to query active batch stage lease: {error:#}"
                            ));
                            if Instant::now() >= lease_deadline {
                                awaiting_teardown = true;
                                cancellation.cancel(StageCancellationReason::LeaseLost);
                            }
                        }
                    }
                }
            }
        };
        let persisted_state = self
            .settle_with_retry("read settled stage lease", || {
                self.store.stage_lease_state(&lease)
            })
            .await?;
        let cancellation_reason = match persisted_state {
            Some(super::store::StageLeaseState::CancellationRequested) => {
                self.settle_with_retry("mark stage execution stopping", || {
                    self.store.mark_stage_execution_stopping(&lease)
                })
                .await?;
                Some(StageCancellationReason::UserRequested)
            }
            Some(super::store::StageLeaseState::ExecutionStopping) => {
                Some(StageCancellationReason::UserRequested)
            }
            _ => cancellation.reason(),
        };
        let execution_result = match cancellation_reason {
            Some(reason) => StageExecutionResolution::Cancelled(reason),
            None => StageExecutionResolution::Finished(execution_result),
        };
        match execution_result {
            StageExecutionResolution::Finished(Ok(outcome)) => {
                let output_artifact_count = outcome.output_artifact_count();
                let output_artifact_ids = outcome.into_output_artifact_ids();
                let completed = self
                    .settle_with_retry("complete stage", || {
                        self.store
                            .complete_stage(&lease, output_artifact_ids.clone())
                    })
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
                if err.downcast_ref::<SpeechStageYield>().is_some() {
                    self.settle_with_retry("yield stage", || self.store.yield_stage(&lease))
                        .await?;
                    drop(_active_execution);
                    self.record_heartbeat("idle", None).await?;
                    return Ok(true);
                }
                let message = err.to_string();
                self.health.record_error(message.clone());
                let deterministic = err.downcast_ref::<DeterministicStageFailure>().is_some();
                let error_code = if deterministic {
                    "invalid_request"
                } else {
                    "executor_failed"
                };
                let failed = self
                    .settle_with_retry("fail stage", || {
                        self.store.fail_stage(
                            &lease,
                            !deterministic,
                            Some(error_code.to_string()),
                            Some(message.clone()),
                        )
                    })
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
                if reason == StageCancellationReason::UserRequested {
                    let finalized = self
                        .settle_with_retry("finalize stage cancellation", || {
                            self.store.finalize_stage_cancellation(&lease)
                        })
                        .await?;
                    self.record_stage_observation(
                        &claimed,
                        RuntimeStageOutcome::Cancelled,
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
                    if finalized.is_none() {
                        self.health.record_error(
                            "User-cancelled stage lost ownership before finalization".to_string(),
                        );
                    }
                    drop(_active_execution);
                    return Ok(true);
                }
                // Always attempt an owner-fenced relinquish, even after an
                // observed lease loss: an expired-but-unreclaimed attempt is
                // still ours to retry, while a reclaimed attempt safely no-ops
                // on its attempt token. Skipping here orphaned `Running` rows.
                let relinquished = self
                    .settle_with_retry("relinquish cancelled stage", || {
                        self.store.relinquish_stage_lease(
                            &lease,
                            reason.as_error_code(),
                            format!("Stage execution cancelled: {}", reason.as_error_code()),
                        )
                    })
                    .await?;
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

        drop(_active_execution);
        self.record_heartbeat(
            if self.drain.is_draining() {
                "draining"
            } else {
                "idle"
            },
            None,
        )
        .await?;
        Ok(true)
    }

    /// Retry a settlement call through transient store errors, so one failed
    /// write does not strand a claimed stage until lease expiry or discard a
    /// finished result.
    async fn settle_with_retry<T, F, Fut>(&self, operation: &str, mut call: F) -> anyhow::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let mut attempt = 1;
        loop {
            match call().await {
                Ok(value) => return Ok(value),
                Err(error) if attempt < SETTLEMENT_ATTEMPTS => {
                    warn!(
                        worker_id = %self.config.worker_id,
                        operation,
                        attempt,
                        error = %format!("{error:#}"),
                        "Batch stage settlement failed; retrying"
                    );
                    self.health
                        .record_error(format!("Failed to {operation}: {error:#}"));
                    tokio::time::sleep(SETTLEMENT_RETRY_DELAY.saturating_mul(1 << (attempt - 1)))
                        .await;
                    attempt += 1;
                }
                Err(error) => {
                    return Err(error.context(format!(
                        "Failed to {operation} after {SETTLEMENT_ATTEMPTS} attempts"
                    )))
                }
            }
        }
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
            .reconcile_inconsistent_states(self.config.maintenance_batch_limit)
            .await
            .context("Failed to reconcile durable runtime state")?;
        self.store
            .recover_expired_stage_leases(self.config.maintenance_batch_limit)
            .await
            .context("Failed to recover expired runtime stage leases")?;
        if let Some(artifact_store) = self.artifact_store.as_ref() {
            artifact_store
                .cleanup_due(self.config.maintenance_batch_limit)
                .await
                .context("Failed to clean tombstoned artifact objects")?;
        }
        for executor in self.executors.values() {
            executor.maintenance().await?;
        }
        // Expired objects go first, so the jobs that owned them can be swept
        // in the same pass.
        let retention = crate::env_config::tuning().retention;
        if let Some(artifact_store) = self.artifact_store.as_ref() {
            artifact_store
                .expire_retained(retention, self.config.maintenance_batch_limit)
                .await
                .context("Failed to expire retained artifact objects")?;
        }
        let swept = self
            .store
            .sweep_retention(retention, self.config.maintenance_batch_limit)
            .await
            .context("Failed to sweep expired durable state")?;
        if swept != Default::default() {
            debug!(?swept, "Swept expired durable state");
        }
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
            let slots = runner.config.resources.concurrency_slots.max(1);
            let mut workers = futures::stream::FuturesUnordered::new();
            for _ in 0..slots {
                workers.push(runner.run_slot(drain_timeout));
            }
            use futures::StreamExt;
            while workers.next().await.is_some() {}
            if let Err(err) = runner.record_heartbeat("drained", None).await {
                error!(worker_id = %runner.config.worker_id, error = %err, "Failed to record drained batch worker heartbeat");
                runner.health.record_error(format!("{err:#}"));
            }
            runner.health.mark_stopped();
            if let Err(err) = runner.record_heartbeat("stopped", None).await {
                error!(worker_id = %runner.config.worker_id, error = %err, "Failed to record stopped batch worker heartbeat");
                runner.health.record_error(format!("{err:#}"));
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

    async fn run_slot(&self, drain_timeout: Duration) {
        let runner = self;
        loop {
            if runner.drain.is_draining() {
                break;
            }

            // Registered before looking for work, so an enqueue that lands
            // while this slot finds nothing still wakes it.
            let stage_ready = runner.store.stage_ready();
            tokio::pin!(stage_ready);
            stage_ready.as_mut().enable();
            let iteration = runner.run_once();
            tokio::pin!(iteration);
            let result = tokio::select! {
                result = &mut iteration => result,
                _ = runner.drain.wait() => {
                    match tokio::time::timeout(drain_timeout, &mut iteration).await {
                        Ok(result) => result,
                        Err(_) => {
                            runner.cancel_active_execution(StageCancellationReason::DrainDeadline);
                            // The supervisor applies a bounded wait around the
                            // worker task. This slot must keep owning the
                            // pinned executor if cooperative cancellation does
                            // not settle; dropping it would not prove that
                            // synchronous or device work stopped.
                            iteration.await
                        }
                    }
                },
            };
            let should_pause = match result {
                Ok(true) => false,
                Ok(false) => true,
                Err(err) => {
                    error!(worker_id = %runner.config.worker_id, error = %err, "Batch runtime worker iteration failed");
                    runner.health.record_error(format!("{err:#}"));
                    true
                }
            };

            if runner.drain.is_draining() {
                break;
            }
            if should_pause {
                tokio::select! {
                    _ = tokio::time::sleep(runner.config.idle_poll_interval) => {}
                    _ = &mut stage_ready => {}
                    _ = runner.drain.wait() => break,
                }
            }
        }
    }

    async fn record_heartbeat(
        &self,
        status: &str,
        current: Option<(String, String)>,
    ) -> anyhow::Result<()> {
        let mut last_heartbeat = self.last_heartbeat.lock().await;
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
        let (mut active_lease_ids, used_slots) = {
            let active = self
                .active_executions
                .read()
                .unwrap_or_else(|poison| poison.into_inner());
            (
                active
                    .values()
                    .map(|execution| execution.stage_id.clone())
                    .collect::<Vec<_>>(),
                active
                    .values()
                    .fold(0u32, |sum, execution| sum.saturating_add(execution.weight)),
            )
        };
        active_lease_ids.sort();
        let available_slots = if self.drain.is_draining() {
            0
        } else {
            self.config
                .resources
                .concurrency_slots
                .saturating_sub(used_slots)
        };
        let status = if self.drain.is_draining() && !matches!(status, "stopped" | "drained") {
            "draining"
        } else if !active_lease_ids.is_empty() {
            "running"
        } else {
            status
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
        let update = RegisteredWorkerHeartbeatUpdate {
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
        };
        if last_heartbeat.as_ref().is_some_and(|last| {
            last.update == update && last.written_at.elapsed() < HEARTBEAT_REFRESH_INTERVAL
        }) {
            return Ok(());
        }
        self.store
            .upsert_registered_worker_heartbeat(update.clone())
            .await?;
        *last_heartbeat = Some(WrittenHeartbeat {
            update,
            written_at: Instant::now(),
        });
        Ok(())
    }

    fn claim_filter(&self) -> StageClaimFilter {
        let mut filter = StageClaimFilter::for_worker_queues(&self.config.queue_names);
        filter.capabilities = normalized_claim_values(&self.config.capabilities);
        filter.model_ids = normalized_claim_values(&self.config.model_ids);
        filter.stage_kinds = normalized_claim_values(&self.config.stage_kinds);
        filter.resources = self.config.resources.clone();
        for active in self
            .active_executions
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .values()
        {
            filter.resources.concurrency_slots = filter
                .resources
                .concurrency_slots
                .saturating_sub(active.weight);
            filter.resources.memory_bytes = filter
                .resources
                .memory_bytes
                .map(|bytes| bytes.saturating_sub(active.memory_bytes));
        }
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
            Err(_) => Err(anyhow!(
                "Batch worker shutdown exceeded its bounded deadline; execution ownership remains attached"
            )),
        }
    }

    /// Drain the worker without allowing the process runtime to disappear
    /// while an executor still owns capacity. The bounded [`Self::shutdown`]
    /// variant is appropriate for embedders whose Tokio runtime remains alive;
    /// the server process must instead wait for exact settlement (or be killed
    /// as a whole by its supervisor).
    pub async fn shutdown_for_process(mut self) -> anyhow::Result<()> {
        self.begin_drain();
        let mut handle = self
            .handle
            .take()
            .expect("batch worker supervisor handle must exist");
        match tokio::time::timeout(self.shutdown_timeout, &mut handle).await {
            Ok(joined) => joined.map_err(|err| anyhow!("Batch worker task join failed: {err}")),
            Err(_) => {
                error!(
                    timeout_ms = self.shutdown_timeout.as_millis(),
                    "Batch worker exceeded its shutdown deadline; retaining the process runtime until execution ownership settles"
                );
                handle
                    .await
                    .map_err(|err| anyhow!("Batch worker task join failed: {err}"))
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
#[path = "worker_tests.rs"]
mod tests;

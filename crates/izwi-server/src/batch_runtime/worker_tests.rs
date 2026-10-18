use super::*;
use crate::{
    batch_runtime::{
        store::{NewJobStage, NewRuntimeJob, StoreFaultPoint, StoreFaults},
        types::{
            RuntimeArtifactKind, RuntimeArtifactRole, RuntimeCancellationState, RuntimeJobKind,
            RuntimeJobStatus, RuntimeStageStatus,
        },
    },
    db::test_database::TestDatabase,
};
use izwi_core::{EngineConfig, RuntimeStageOutcome};
use serde_json::json;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

struct FakeExecutor {
    calls: AtomicUsize,
    fail_first: bool,
}

struct BlockingExecutor {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

/// Fails every attempt with an invalid-request error.
struct DeterministicFailureExecutor {
    calls: AtomicUsize,
}

#[async_trait]
impl StageExecutor for DeterministicFailureExecutor {
    fn stage_kind(&self) -> &'static str {
        "fake_stage"
    }

    async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(DeterministicStageFailure("`speed` must be between 0.5 and 2".into()).into())
    }
}

struct ContextProgressExecutor;

struct CancellationAwareBlockingExecutor {
    started: Arc<Notify>,
    release: Arc<Notify>,
    cancellation: Arc<RwLock<Option<StageCancellationSignal>>>,
}

struct SynchronousBlockingExecutor {
    started: Arc<Notify>,
    release: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    cancellation: Arc<RwLock<Option<StageCancellationSignal>>>,
}

#[async_trait]
impl StageExecutor for FakeExecutor {
    fn stage_kind(&self) -> &'static str {
        "fake_stage"
    }

    async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
        anyhow::bail!("runner did not invoke context-aware fake stage execution")
    }

    async fn execute_with_context(
        &self,
        context: StageExecutionContext,
    ) -> anyhow::Result<StageExecutionOutcome> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_first && call == 0 {
            anyhow::bail!("planned fake failure");
        }
        let artifact = context
            .publish_output_artifact(NewStageOutputArtifact {
                publication_key: "fake-result".to_string(),
                artifact_kind: RuntimeArtifactKind::Text,
                artifact_role: RuntimeArtifactRole::OutputPrimary,
                media_asset_id: None,
                text_asset_id: None,
                storage_key: None,
                content_type: Some("text/plain".to_string()),
                filename: None,
                size_bytes: Some(0),
                sha256: None,
                metadata_json: json!({"fixture": true}),
                retention_policy: "test".to_string(),
            })
            .await?;
        StageExecutionOutcome::try_new(vec![artifact.id])
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
        Ok(StageExecutionOutcome::empty())
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
impl StageExecutor for CancellationAwareBlockingExecutor {
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
        self.release.notified().await;
        Ok(StageExecutionOutcome::empty())
    }
}

#[async_trait]
impl StageExecutor for SynchronousBlockingExecutor {
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
        let release = self.release.clone();
        self.started.notify_one();
        tokio::task::spawn_blocking(move || {
            let (released, notify) = &*release;
            let mut released = released.lock().unwrap_or_else(|poison| poison.into_inner());
            while !*released {
                released = notify
                    .wait(released)
                    .unwrap_or_else(|poison| poison.into_inner());
            }
        })
        .await
        .context("synchronous test executor join")?;
        Ok(StageExecutionOutcome::empty())
    }
}

/// A store on a fresh test database (SQLite, or PostgreSQL when
/// `IZWI_TEST_STORE_PG_URL` is set), which must outlive the store.
async fn build_store() -> (Arc<BatchRuntimeStore>, TestDatabase) {
    let database = TestDatabase::new().await;
    let store = BatchRuntimeStore::initialize_with_database(database.database());
    (Arc::new(store), database)
}

async fn build_store_with_faults() -> (Arc<BatchRuntimeStore>, Arc<StoreFaults>, TestDatabase) {
    let database = TestDatabase::new().await;
    let faults = Arc::new(StoreFaults::default());
    let mut store = BatchRuntimeStore::initialize_with_database(database.database());
    store.set_faults_for_test(faults.clone());
    (Arc::new(store), faults, database)
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
async fn concurrent_worker_heartbeats_every_lease_and_drains_all_slots() {
    let (store, _database) = build_store().await;
    let mut stage_ids = Vec::new();
    for _ in 0..3 {
        stage_ids.push(create_queued_fake_stage(&store, 2).await.unwrap().1);
    }
    let mut config = BatchWorkerConfig::local("concurrent-worker");
    config.resources.concurrency_slots = 2;
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    // Keep wall-clock margins generous for shared CI runners: a 150ms
    // lease flaked when renewal heartbeats slipped under parallel load.
    config.lease_duration = Duration::from_millis(800);
    config.drain_timeout = Duration::from_millis(300);
    let release = Arc::new(Notify::new());
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(BlockingExecutor {
            started: Arc::new(Notify::new()),
            release: release.clone(),
        })],
        config,
        BatchWorkerHealth::new("concurrent-worker"),
    );
    let supervisor = runner.clone().spawn();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if runner.active_executions.read().unwrap().len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(runner.claim_filter().resources.concurrency_slots, 0);
    assert!(!runner.run_once().await.unwrap());
    // Both leases must stay alive for longer than the original lease.
    tokio::time::sleep(Duration::from_millis(1000)).await;
    let mut running = 0;
    let mut queued = 0;
    for id in &stage_ids {
        let stage = store.get_stage(id).await.unwrap().unwrap();
        if stage.status == RuntimeStageStatus::Running {
            running += 1;
        }
        if stage.status == RuntimeStageStatus::Queued {
            queued += 1;
        }
    }
    assert_eq!((running, queued), (2, 1));
    let heartbeat = store
        .get_worker_heartbeat("concurrent-worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(heartbeat.details.active_lease_ids.len(), 2);
    assert_eq!(heartbeat.details.available_slots, 0);
    release.notify_waiters();
    supervisor.shutdown().await.unwrap();
    assert!(runner.active_executions.read().unwrap().is_empty());
    for id in &stage_ids {
        let stage = store.get_stage(id).await.unwrap().unwrap();
        assert_ne!(
            stage.status,
            RuntimeStageStatus::Running,
            "{} {:?}",
            id,
            runner.health.snapshot()
        );
    }
}

#[test]
fn old_attempt_cleanup_preserves_a_reclaimed_stages_new_execution() {
    let slots = Arc::new(RwLock::new(HashMap::new()));
    let mut guards = Vec::new();
    for attempt_id in ["old-attempt", "new-attempt"] {
        slots.write().unwrap().insert(
            attempt_id.to_string(),
            ActiveExecution {
                stage_id: "same-stage".into(),
                cancellation: StageCancellationSignal::new(),
                weight: 1,
                memory_bytes: 10,
            },
        );
        guards.push(ActiveExecutionGuard {
            slots: slots.clone(),
            attempt_id: attempt_id.to_string(),
        });
    }
    assert_eq!(
        slots
            .read()
            .unwrap()
            .values()
            .map(|execution| execution.weight)
            .sum::<u32>(),
        2
    );
    drop(guards.remove(0));
    let live = slots.read().unwrap();
    assert_eq!(live.len(), 1);
    assert!(live.contains_key("new-attempt"));
    assert_eq!(live["new-attempt"].memory_bytes, 10);
    drop(live);
    drop(guards);
    assert!(slots.read().unwrap().is_empty());
}

#[tokio::test]
async fn weighted_active_claims_reduce_all_available_worker_resources() {
    let (store, _database) = build_store().await;
    let mut config = BatchWorkerConfig::local("weighted-worker");
    config.resources.concurrency_slots = 7;
    config.resources.memory_bytes = Some(100);
    let runner = BatchWorkerRunner::new(
        store,
        vec![],
        config,
        BatchWorkerHealth::new("weighted-worker"),
    );
    runner.active_executions.write().unwrap().insert(
        "active".into(),
        ActiveExecution {
            stage_id: "stage".into(),
            cancellation: StageCancellationSignal::new(),
            weight: 3,
            memory_bytes: 60,
        },
    );
    let remaining = runner.claim_filter().resources;
    assert_eq!(remaining.concurrency_slots, 4);
    assert_eq!(remaining.memory_bytes, Some(40));
}

struct YieldOnceExecutor {
    calls: AtomicUsize,
}
#[async_trait]
impl StageExecutor for YieldOnceExecutor {
    fn stage_kind(&self) -> &'static str {
        "fake_stage"
    }
    async fn execute(&self, _claimed: ClaimedStage) -> anyhow::Result<StageExecutionOutcome> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(SpeechStageYield.into());
        }
        Ok(StageExecutionOutcome::empty())
    }
}

#[test]
fn stage_execution_outcome_is_bounded_and_uuid_typed() {
    let id = new_uuid();
    assert_eq!(
        StageExecutionOutcome::try_new(vec![id])
            .expect("canonical output")
            .output_artifact_count(),
        1
    );
    assert!(StageExecutionOutcome::try_new(vec!["not-a-uuid".to_string()]).is_err());
    assert!(StageExecutionOutcome::try_new(
        (0..=super::super::store::MAX_STAGE_OUTPUT_ARTIFACTS)
            .map(|_| new_uuid())
            .collect(),
    )
    .is_err());
}

#[tokio::test]
async fn cooperative_speech_yield_releases_slot_without_failure_or_retry_consumption() {
    let (store, _database) = build_store().await;
    let (job_id, stage_id) = create_queued_fake_stage(&store, 1).await.unwrap();
    let health = BatchWorkerHealth::new("yield-worker");
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(YieldOnceExecutor {
            calls: AtomicUsize::new(0),
        })],
        BatchWorkerConfig::local("yield-worker"),
        health.clone(),
    );
    assert!(runner.run_once().await.unwrap());
    let stage = store.get_stage(&stage_id).await.unwrap().unwrap();
    assert_eq!(stage.status, RuntimeStageStatus::Queued);
    assert_eq!(stage.attempt_count, 0);
    assert!(stage.attempt_token.is_none());
    assert!(runner.active_executions.read().unwrap().is_empty());
    assert!(health.snapshot().last_error.is_none());
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(runner.run_once().await.unwrap());
    let stage = store.get_stage(&stage_id).await.unwrap().unwrap();
    assert_eq!(stage.status, RuntimeStageStatus::Completed);
    assert_eq!(stage.attempt_count, 1);
    assert_eq!(
        store.get_job(&job_id).await.unwrap().unwrap().status,
        RuntimeJobStatus::Completed
    );
}

#[tokio::test]
async fn runner_claims_and_completes_stage() {
    let (store, _database) = build_store().await;
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
    assert_eq!(stage.output_artifact_ids.len(), 1);
    assert!(store
        .get_artifact(&stage.output_artifact_ids[0])
        .await
        .expect("artifact lookup")
        .is_some());
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
async fn heartbeat_failure_after_claim_does_not_strand_the_claimed_stage() {
    let (store, faults, _database) = build_store_with_faults().await;
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
    // The polling heartbeat succeeds; the one right after the claim fails.
    faults.arm(StoreFaultPoint::WorkerHeartbeat, 1, 1);

    assert!(runner.run_once().await.expect("run once"));

    assert_eq!(faults.pending_failures(StoreFaultPoint::WorkerHeartbeat), 0);
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Completed);
    let job = store
        .get_job(&job_id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(job.status, RuntimeJobStatus::Completed);
    assert!(health
        .snapshot()
        .last_error
        .is_some_and(|error| error.contains("heartbeat after claim")));
}

#[tokio::test]
async fn transient_completion_errors_are_retried_instead_of_discarding_the_result() {
    let (store, faults, _database) = build_store_with_faults().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let executor = Arc::new(FakeExecutor {
        calls: AtomicUsize::new(0),
        fail_first: false,
    });
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![executor.clone()],
        BatchWorkerConfig::local("worker-test"),
        BatchWorkerHealth::new("worker-test"),
    );
    faults.arm(StoreFaultPoint::CompleteStage, 0, 2);

    assert!(runner.run_once().await.expect("run once"));

    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Completed);
    assert_eq!(stage.attempt_count, 1);
    assert_eq!(stage.output_artifact_ids.len(), 1);
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn persistent_settlement_errors_give_up_after_bounded_retries() {
    let (store, faults, _database) = build_store_with_faults().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            fail_first: false,
        })],
        BatchWorkerConfig::local("worker-test"),
        BatchWorkerHealth::new("worker-test"),
    );
    faults.arm(StoreFaultPoint::CompleteStage, 0, u32::MAX);

    let error = runner.run_once().await.expect_err("settlement gives up");

    assert!(format!("{error:#}").contains("complete stage after 5 attempts"));
    assert_eq!(
        faults.pending_failures(StoreFaultPoint::CompleteStage),
        u32::MAX - SETTLEMENT_ATTEMPTS
    );
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Running);
}

#[tokio::test]
async fn context_aware_execution_records_attempt_scoped_progress() {
    let (store, _database) = build_store().await;
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
async fn execution_deadline_retains_ownership_until_executor_teardown() {
    let (store, _database) = build_store().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let cancellation = Arc::new(RwLock::new(None));
    let mut config = BatchWorkerConfig::local("worker-test");
    config.execution_timeout = Some(Duration::from_millis(50));
    config.lease_duration = Duration::from_millis(500);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(SynchronousBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );

    let mut run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if cancellation
                .read()
                .unwrap_or_else(|poison| poison.into_inner())
                .as_ref()
                .is_some_and(StageCancellationSignal::is_cancelled)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("runner should request deadline cancellation");
    assert!(tokio::time::timeout(Duration::from_millis(100), &mut run)
        .await
        .is_err());

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
    let running = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(running.status, RuntimeStageStatus::Running);
    assert_eq!(running.worker_id.as_deref(), Some("worker-test"));
    assert!(running.lease_expires_at.is_some());
    assert!(store
        .claim_next_stage("replacement-worker", 60_000)
        .await
        .expect("replacement claim")
        .is_none());

    {
        let (released, notify) = &*release;
        *released.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        notify.notify_all();
    }
    assert!(tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .expect("executor should settle")
        .expect("runner join")
        .expect("run once"));
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
async fn reclaimed_lease_does_not_release_the_stale_local_executor_early() {
    let (store, _database) = build_store().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 3).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let cancellation = Arc::new(RwLock::new(None));
    let mut config = BatchWorkerConfig::local("worker-test");
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    config.lease_duration = Duration::from_secs(60);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(CancellationAwareBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );
    let ownership = runner.clone();
    let mut run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");

    let stale_stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    let stale = super::super::types::StageLease {
        stage_id: stale_stage.id,
        worker_id: stale_stage.worker_id.expect("active worker"),
        attempt_count: stale_stage.attempt_count,
        attempt_token: stale_stage.attempt_token,
    };
    store
        .relinquish_stage_lease(&stale, "injected_loss", "replace exact attempt")
        .await
        .expect("replace stale attempt")
        .expect("stage should retry");
    let replacement = store
        .claim_next_stage("replacement-worker", 60_000)
        .await
        .expect("replacement claim")
        .expect("replacement should own the durable stage");
    assert_ne!(replacement.stage.attempt_token, stale.attempt_token);

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if cancellation
                .read()
                .unwrap_or_else(|poison| poison.into_inner())
                .as_ref()
                .is_some_and(StageCancellationSignal::is_cancelled)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("runner should observe exact lease loss");
    assert!(tokio::time::timeout(Duration::from_millis(100), &mut run)
        .await
        .is_err());
    assert_eq!(
        ownership
            .active_executions
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .len(),
        1
    );
    assert_eq!(ownership.claim_filter().resources.concurrency_slots, 0);

    release.notify_one();
    assert!(tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .expect("stale executor should settle")
        .expect("runner join")
        .expect("run once"));
    let current = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(current.worker_id.as_deref(), Some("replacement-worker"));
    assert_eq!(current.attempt_token, replacement.stage.attempt_token);
    assert!(ownership
        .active_executions
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .is_empty());
}

#[tokio::test]
async fn transient_lease_store_errors_do_not_cancel_execution_before_the_lease_deadline() {
    let (store, faults, _database) = build_store_with_faults().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let cancellation = Arc::new(RwLock::new(None));
    let mut config = BatchWorkerConfig::local("worker-test");
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    config.lease_duration = Duration::from_millis(1_500);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(CancellationAwareBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );
    let run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");

    faults.arm(StoreFaultPoint::StageLeaseState, 0, 5);
    faults.arm(StoreFaultPoint::RenewStageLease, 0, 2);
    tokio::time::timeout(Duration::from_secs(5), async {
        while faults.pending_failures(StoreFaultPoint::StageLeaseState) > 0
            || faults.pending_failures(StoreFaultPoint::RenewStageLease) > 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("runner should hit the injected store errors");
    // The renewal retries land well inside the lease; give one a chance.
    tokio::time::sleep(Duration::from_millis(400)).await;

    let signal = cancellation
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone()
        .expect("context cancellation signal");
    assert!(
        !signal.is_cancelled(),
        "transient store errors cancelled a live attempt: {:?}",
        signal.reason()
    );
    release.notify_one();
    assert!(tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("executor should settle")
        .expect("runner join")
        .expect("run once"));
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Completed);
    assert_eq!(stage.attempt_count, 1);
}

#[tokio::test]
async fn lease_renewal_errors_past_the_local_lease_deadline_cancel_execution() {
    let (store, faults, _database) = build_store_with_faults().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let cancellation = Arc::new(RwLock::new(None));
    let lease_duration = Duration::from_millis(600);
    let mut config = BatchWorkerConfig::local("worker-test");
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    config.lease_duration = lease_duration;
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(CancellationAwareBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );
    faults.arm(StoreFaultPoint::RenewStageLease, 0, u32::MAX);
    let run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");
    let executing_since = Instant::now();

    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if cancellation
                .read()
                .unwrap_or_else(|poison| poison.into_inner())
                .as_ref()
                .is_some_and(StageCancellationSignal::is_cancelled)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("runner should give up once the local lease deadline passes");
    assert!(executing_since.elapsed() >= lease_duration / 2);
    let signal = cancellation
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone()
        .expect("context cancellation signal");
    assert_eq!(signal.reason(), Some(StageCancellationReason::LeaseLost));

    faults.arm(StoreFaultPoint::RenewStageLease, 0, 0);
    release.notify_one();
    assert!(tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("executor should settle")
        .expect("runner join")
        .expect("run once"));
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Retrying);
    assert_eq!(stage.error_code.as_deref(), Some("lease_lost"));
}

#[tokio::test]
async fn durable_user_cancellation_wins_after_a_deadline_signal() {
    let (store, _database) = build_store().await;
    let (job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let cancellation = Arc::new(RwLock::new(None));
    let mut config = BatchWorkerConfig::local("worker-test");
    config.execution_timeout = Some(Duration::from_millis(30));
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(CancellationAwareBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );
    let run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if cancellation
                .read()
                .unwrap_or_else(|poison| poison.into_inner())
                .as_ref()
                .and_then(StageCancellationSignal::reason)
                == Some(StageCancellationReason::ExecutionDeadline)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deadline signal");
    store
        .cancel_job(
            &job_id,
            Some("user cancellation after deadline".to_string()),
        )
        .await
        .expect("cancel job")
        .expect("cancellation request");

    release.notify_one();
    assert!(tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .expect("executor should settle")
        .expect("runner join")
        .expect("run once"));
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Cancelled);
    assert_eq!(stage.cancellation_state, None);
    let job = store
        .get_job(&job_id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(job.status, RuntimeJobStatus::Cancelled);
    assert_eq!(job.cancellation_state, None);
}

#[tokio::test]
async fn draining_runner_records_heartbeat_without_claiming() {
    let (store, _database) = build_store().await;
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
async fn idle_polls_refresh_an_unchanged_heartbeat_at_most_once_per_interval() {
    let clock = Arc::new(AtomicI64::new(1_000));
    let database = TestDatabase::new().await;
    let mut store = BatchRuntimeStore::initialize_with_database(database.database());
    store.set_test_clock(clock.clone());
    let store = Arc::new(store);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            fail_first: false,
        })],
        BatchWorkerConfig::local("worker-test"),
        BatchWorkerHealth::new("worker-test"),
    );
    let heartbeat = || async {
        store
            .get_worker_heartbeat("worker-test")
            .await
            .expect("heartbeat")
            .expect("heartbeat exists")
    };

    assert!(!runner.run_once().await.expect("first poll"));
    assert_eq!(heartbeat().await.last_heartbeat_at, 1_000);

    // Unchanged idle polls inside the refresh interval do not commit.
    clock.store(2_000, Ordering::SeqCst);
    for _ in 0..5 {
        assert!(!runner.run_once().await.expect("idle poll"));
    }
    assert_eq!(heartbeat().await.last_heartbeat_at, 1_000);

    // A changed heartbeat is written immediately.
    runner.drain.begin();
    assert!(!runner.run_once().await.expect("draining poll"));
    let draining = heartbeat().await;
    assert_eq!(draining.status, "draining");
    assert_eq!(draining.last_heartbeat_at, 2_000);

    // An unchanged heartbeat is refreshed once the interval has elapsed.
    clock.store(3_000, Ordering::SeqCst);
    runner
        .last_heartbeat
        .lock()
        .await
        .as_mut()
        .expect("written heartbeat")
        .written_at -= HEARTBEAT_REFRESH_INTERVAL;
    assert!(!runner.run_once().await.expect("refresh poll"));
    assert_eq!(heartbeat().await.last_heartbeat_at, 3_000);
}

#[tokio::test]
async fn runner_only_claims_registered_stage_kinds_and_reports_configuration() {
    let (store, _database) = build_store().await;
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
async fn cancellation_retains_capacity_until_non_cooperative_executor_teardown() {
    let clock = Arc::new(AtomicI64::new(
        super::super::store::current_timestamp_millis(),
    ));
    let database = TestDatabase::new().await;
    let mut store = BatchRuntimeStore::initialize_with_database(database.database());
    store.set_test_clock(clock.clone());
    let store = Arc::new(store);
    let (job_id, stage_id) = create_queued_fake_stage(&store, 1).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let observed_cancellation = Arc::new(RwLock::new(None));
    let mut config = BatchWorkerConfig::local("worker-test");
    config.lease_duration = Duration::from_millis(40);
    config.poll_interval = Duration::from_millis(10);
    config.idle_poll_interval = Duration::from_millis(10);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(CancellationAwareBlockingExecutor {
            started: started.clone(),
            release: release.clone(),
            cancellation: observed_cancellation.clone(),
        })],
        config,
        BatchWorkerHealth::new("worker-test"),
    );
    let mut run = tokio::spawn(async move { runner.run_once().await });
    tokio::time::timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("executor should start");

    let cancellation = store
        .cancel_job(&job_id, Some("cancel while executing".to_string()))
        .await
        .expect("cancel")
        .expect("cancelled job");
    assert_eq!(cancellation.status, RuntimeJobStatus::Running);
    // cancel_job reads the job back after committing, so the worker's
    // 10 ms poll may already have advanced it to ExecutionStopping.
    assert!(
        matches!(
            cancellation.cancellation_state,
            Some(
                RuntimeCancellationState::Requested
                    | RuntimeCancellationState::ExecutionStopping
            )
        ),
        "{:?}",
        cancellation.cancellation_state
    );

    let requested = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    let owner = requested.worker_id.clone();
    let lease_expiry = requested.lease_expires_at;
    let attempt_token = requested.attempt_token.clone();
    assert_eq!(requested.status, RuntimeStageStatus::Running);
    // The worker polls every 10 ms and may already have moved the request
    // on to ExecutionStopping; either way the running stage carries it.
    assert!(
        matches!(
            requested.cancellation_state,
            Some(
                RuntimeCancellationState::Requested
                    | RuntimeCancellationState::ExecutionStopping
            )
        ),
        "{:?}",
        requested.cancellation_state
    );
    assert!(owner.is_some());
    assert!(lease_expiry.is_some());
    assert!(attempt_token.is_some());

    clock.fetch_add(100, Ordering::SeqCst);

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let stage = store
                .get_stage(&stage_id)
                .await
                .expect("stage")
                .expect("stage exists");
            if stage.cancellation_state == Some(RuntimeCancellationState::ExecutionStopping) {
                assert_eq!(stage.worker_id, owner);
                assert_eq!(stage.attempt_token, attempt_token);
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker should observe cancellation request");
    let signal = observed_cancellation
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone()
        .expect("executor cancellation signal");
    assert_eq!(
        signal.reason(),
        Some(StageCancellationReason::UserRequested)
    );

    assert!(tokio::time::timeout(Duration::from_millis(100), &mut run)
        .await
        .is_err());
    let still_stopping = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(still_stopping.status, RuntimeStageStatus::Running);
    assert_eq!(still_stopping.worker_id, owner);
    assert!(still_stopping.lease_expires_at.is_some_and(
        |expires_at| expires_at > u64::try_from(clock.load(Ordering::SeqCst)).unwrap()
    ));

    release.notify_one();
    assert!(tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .expect("cancelled execution should finish after executor teardown")
        .expect("runner join")
        .expect("run once"));

    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Cancelled);
    assert_eq!(stage.cancellation_state, None);
    assert!(stage.output_artifact_ids.is_empty());
    assert_eq!(stage.lease_expires_at, None);
    assert_eq!(stage.worker_id, None);
    let job = store
        .get_job(&job_id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(job.status, RuntimeJobStatus::Cancelled);
    assert_eq!(job.cancellation_state, None);
}

#[tokio::test]
async fn shutdown_drain_waits_for_active_iteration_while_renewing_lease() {
    // Lease validity must not depend on how quickly CI schedules SQLite or
    // the worker. Timers still run normally; advance lease time only after
    // observing each renewal in the database.
    let clock = Arc::new(AtomicI64::new(
        super::super::store::current_timestamp_millis(),
    ));
    let (mut store, _database) = build_store().await;
    Arc::get_mut(&mut store)
        .expect("unshared store")
        .set_test_clock(clock.clone());
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

    let original_expiry = i64::try_from(running.lease_expires_at.expect("initial lease"))
        .expect("lease timestamp fits i64");
    supervisor.begin_drain();
    let mut shutdown = tokio::spawn(supervisor.shutdown());
    for _ in 0..3 {
        let now = clock.fetch_add(60, Ordering::SeqCst) + 60;
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                assert!(!shutdown.is_finished(), "drain must wait for the executor");
                let renewed = store
                    .get_stage(&stage_id)
                    .await
                    .expect("stage")
                    .expect("stage exists");
                assert_eq!(renewed.status, RuntimeStageStatus::Running);
                if renewed.lease_expires_at
                    == Some(u64::try_from(now + 120).expect("positive lease timestamp"))
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker should renew its lease while draining");
    }
    assert!(clock.load(Ordering::SeqCst) > original_expiry);
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
    assert!(completed.output_artifact_ids.is_empty());
    let heartbeat = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let heartbeat = store
                .get_worker_heartbeat("worker-test")
                .await
                .expect("heartbeat")
                .expect("heartbeat exists");
            if heartbeat.status == "stopped" {
                break heartbeat;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker should publish stopped heartbeat");
    assert_eq!(heartbeat.status, "stopped");
    assert_eq!(heartbeat.current_stage_id, None);
    assert_eq!(heartbeat.details.available_slots, 0);
    assert_eq!(
        heartbeat.details.health_json.get("running"),
        Some(&json!(false))
    );
}

#[tokio::test]
async fn bounded_shutdown_detaches_and_retains_unresolved_execution_ownership() {
    let (store, _database) = build_store().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mut config = BatchWorkerConfig::local("worker-test");
    config.drain_timeout = Duration::from_millis(60);
    config.lease_duration = Duration::from_millis(500);
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

    let shutdown_started = Instant::now();
    let error = tokio::time::timeout(Duration::from_secs(5), supervisor.shutdown())
        .await
        .expect("shutdown should remain bounded")
        .expect_err("unresolved executor must not report a clean shutdown");
    assert!(error
        .to_string()
        .contains("execution ownership remains attached"));
    assert!(shutdown_started.elapsed() < Duration::from_secs(3));

    let retained = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(retained.status, RuntimeStageStatus::Running);
    assert_eq!(retained.worker_id.as_deref(), Some("worker-test"));
    assert!(retained.lease_expires_at.is_some());
    assert!(store
        .claim_next_stage("replacement-worker", 60_000)
        .await
        .expect("replacement claim")
        .is_none());

    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let stage = store
                .get_stage(&stage_id)
                .await
                .expect("stage")
                .expect("stage exists");
            if stage.status == RuntimeStageStatus::Retrying {
                assert_eq!(stage.worker_id, None);
                assert_eq!(stage.lease_expires_at, None);
                assert_eq!(stage.error_code.as_deref(), Some("drain_deadline"));
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("detached worker should settle after executor teardown");

    let heartbeat = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let heartbeat = store
                .get_worker_heartbeat("worker-test")
                .await
                .expect("heartbeat")
                .expect("heartbeat exists");
            if heartbeat.status == "stopped" {
                break heartbeat;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("detached worker should publish stopped heartbeat");
    assert_eq!(heartbeat.status, "stopped");
    assert_eq!(heartbeat.current_stage_id, None);
    assert_eq!(heartbeat.details.available_slots, 0);
    assert_eq!(
        heartbeat.details.health_json.get("running"),
        Some(&json!(false))
    );
}

#[tokio::test]
async fn process_shutdown_keeps_runtime_alive_until_exact_execution_teardown() {
    let (store, _database) = build_store().await;
    let (_job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mut config = BatchWorkerConfig::local("worker-test");
    config.drain_timeout = Duration::from_millis(60);
    config.lease_duration = Duration::from_millis(500);
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

    let mut shutdown = tokio::spawn(supervisor.shutdown_for_process());
    tokio::time::sleep(Duration::from_millis(2_200)).await;
    assert!(
        !shutdown.is_finished(),
        "process shutdown must retain the runtime owner after its bounded warning deadline"
    );
    let retained = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(retained.status, RuntimeStageStatus::Running);
    assert_eq!(retained.worker_id.as_deref(), Some("worker-test"));

    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), &mut shutdown)
        .await
        .expect("process shutdown should finish after exact teardown")
        .expect("shutdown join")
        .expect("worker shutdown");
}

#[tokio::test]
async fn runner_records_runtime_stage_observations_when_attached() {
    let (store, _database) = build_store().await;
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

/// TPIPE-27: an invalid request fails on the first attempt; the
/// second attempt would fail the same way.
#[tokio::test]
async fn runner_does_not_retry_a_deterministic_failure() {
    let (store, _database) = build_store().await;
    let (job_id, stage_id) = create_queued_fake_stage(&store, 2).await.expect("stage");
    let executor = Arc::new(DeterministicFailureExecutor {
        calls: AtomicUsize::new(0),
    });
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![executor.clone()],
        BatchWorkerConfig::local("worker-test"),
        BatchWorkerHealth::new("worker-test"),
    );

    runner.run_until_idle(4).await.expect("run");

    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    let stage = store
        .get_stage(&stage_id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(stage.status, RuntimeStageStatus::Failed);
    assert_eq!(stage.attempt_count, 1);
    let job = store
        .get_job(&job_id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(job.status, RuntimeJobStatus::Failed);
}

#[tokio::test]
async fn idle_slots_wake_when_this_process_enqueues_a_stage() {
    let (store, _database) = build_store().await;
    let mut config = BatchWorkerConfig::local("wakeup-worker");
    // Long enough that only the wake-up can explain a prompt claim.
    config.idle_poll_interval = Duration::from_secs(60);
    config.drain_timeout = Duration::from_millis(300);
    let runner = BatchWorkerRunner::new(
        store.clone(),
        vec![Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            fail_first: false,
        })],
        config,
        BatchWorkerHealth::new("wakeup-worker"),
    );
    // Migrate first, so the timeouts below measure the slot and not the
    // schema (PostgreSQL test schemas migrate one at a time).
    store.connection().await.expect("store opens");
    let supervisor = runner.clone().spawn();
    tokio::time::timeout(Duration::from_secs(5), async {
        while store
            .get_worker_heartbeat("wakeup-worker")
            .await
            .unwrap()
            .is_none()
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the slot polled");
    // Let the slot find nothing and settle into its idle wait.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let (_, stage_id) = create_queued_fake_stage(&store, 1).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while store.get_stage(&stage_id).await.unwrap().unwrap().status
            != RuntimeStageStatus::Completed
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the idle slot ran the new stage without waiting out its poll");
    supervisor.shutdown().await.unwrap();
}

#[tokio::test]
async fn runner_retries_then_completes_stage() {
    let (store, _database) = build_store().await;
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

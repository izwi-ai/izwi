use super::*;

#[tokio::test]
async fn cooperative_yield_frees_worker_for_short_job_without_consuming_retries() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let (_long_job, long_stage) = create_test_job_and_stage(&store, 0, "speech", 1).await;
    let first = store
        .claim_next_stage("worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    let old = first.lease().unwrap();
    let checkpoint = json!({"completed_segments": 1, "completed_text_bytes": 480});
    assert!(store
        .update_stage_progress(&old, checkpoint.clone())
        .await
        .unwrap());
    let (_short_job, short_stage) = create_test_job_and_stage(&store, 0, "speech", 1).await;
    assert!(store.yield_stage(&old).await.unwrap());
    assert!(!store.yield_stage(&old).await.unwrap());
    let next = store
        .claim_next_stage("worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        next.stage.id, short_stage.id,
        "same-millisecond arrivals must outrank the yielded job"
    );
    store
        .complete_stage(&next.lease().unwrap(), vec![])
        .await
        .unwrap()
        .unwrap();
    let mut current = store
        .claim_next_stage("worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.stage.id, long_stage.id);
    assert_eq!(current.stage.progress_json, Some(checkpoint));
    assert_eq!(current.stage.attempt_count, 1);
    assert_ne!(current.stage.attempt_token, old.attempt_token);
    assert!(!store
        .update_stage_progress(&old, json!({"stale": true}))
        .await
        .unwrap());
    assert!(store.complete_stage(&old, vec![]).await.unwrap().is_none());
    assert!(store
        .publish_stage_output_artifact(&old, test_stage_output("stale"))
        .await
        .unwrap()
        .is_none());
    // A one-attempt job may cooperatively continue many times and still finish.
    for _ in 0..8 {
        let lease = current.lease().unwrap();
        assert!(store.yield_stage(&lease).await.unwrap());
        current = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.stage.attempt_count, 1);
        assert_ne!(current.stage.attempt_token, lease.attempt_token);
    }
    store
        .complete_stage(&current.lease().unwrap(), vec![])
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn cooperative_yield_rejects_foreign_expired_and_cancelled_attempts() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let (job, _stage) = create_test_job_and_stage(&store, 0, "speech", 2).await;
    let claim = store
        .claim_next_stage("worker", 100)
        .await
        .unwrap()
        .unwrap();
    let lease = claim.lease().unwrap();
    let foreign = StageLease {
        attempt_token: Some("foreign".into()),
        ..lease.clone()
    };
    assert!(!store.yield_stage(&foreign).await.unwrap());
    clock.store(1_101, Ordering::SeqCst);
    assert!(!store.yield_stage(&lease).await.unwrap());
    clock.store(1_001, Ordering::SeqCst);
    store
        .cancel_job(&job.id, Some("user stop".into()))
        .await
        .unwrap();
    assert!(!store.yield_stage(&lease).await.unwrap());
}

#[tokio::test]
async fn queue_claim_recovery_and_cancel_are_durable() {
    let (store, _root) = build_store().await;
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

    let recovered = store
        .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
        .expect("recover");
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
async fn expired_lease_recovery_runs_in_stable_bounded_batches() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let mut stage_ids = Vec::new();

    for index in 0..3 {
        clock.store(1_000 + index * 100, Ordering::SeqCst);
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
        let worker_id = format!("worker-{index}");
        let claimed = store
            .claim_next_stage(&worker_id, 10)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        assert_eq!(claimed.stage.id, stage.id);
        stage_ids.push(stage.id);
    }

    clock.store(10_000, Ordering::SeqCst);
    assert_eq!(
        store
            .recover_expired_stage_leases(2)
            .await
            .expect("first recovery batch"),
        2
    );
    for stage_id in &stage_ids[..2] {
        assert_eq!(
            store
                .get_stage(stage_id)
                .await
                .expect("stage")
                .expect("stage exists")
                .status,
            RuntimeStageStatus::Retrying
        );
    }
    assert_eq!(
        store
            .get_stage(&stage_ids[2])
            .await
            .expect("stage")
            .expect("stage exists")
            .status,
        RuntimeStageStatus::Running
    );

    assert_eq!(
        store
            .recover_expired_stage_leases(2)
            .await
            .expect("second recovery batch"),
        1
    );
    assert_eq!(
        store
            .get_stage(&stage_ids[2])
            .await
            .expect("stage")
            .expect("stage exists")
            .status,
        RuntimeStageStatus::Retrying
    );
    assert_eq!(
        store
            .recover_expired_stage_leases(2)
            .await
            .expect("empty recovery batch"),
        0
    );
}

#[tokio::test]
async fn filtered_stage_claim_skips_incompatible_higher_priority_stage() {
    let (store, _root) = build_store().await;
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
    let (store, _root) = build_store().await;
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
    let (store, _root) = build_store().await;
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
    let (store, _root) = build_store().await;
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
    let (store, _root) = build_store().await;
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    force_job_status(
        &store,
        &job.id,
        RuntimeJobStatus::Cancelled,
        Some("cancel before claim CAS"),
    )
    .await;

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
    let (store, _root) = build_store().await;
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("stage should be claimed");
    force_job_status(
        &store,
        &job.id,
        RuntimeJobStatus::Cancelled,
        Some("cancelled outside stage transaction"),
    )
    .await;

    assert!(store
        .complete_stage(&claimed.lease().expect("lease"), vec![])
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
    let (store, _root) = build_store().await;
    let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
    let first = store
        .claim_next_stage("worker-1", 0)
        .await
        .expect("first claim")
        .expect("first attempt");
    let first_lease = first.lease().expect("first lease");
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recover"),
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
        .complete_stage(&first_lease, vec![])
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

    let artifact = store
        .publish_stage_output_artifact(&second_lease, test_stage_output("current-output"))
        .await
        .expect("publish current output")
        .expect("current attempt owns output");
    let completed = store
        .complete_stage(&second_lease, vec![artifact.id.clone()])
        .await
        .expect("current completion")
        .expect("current owner completes");
    assert_eq!(completed.output_artifact_ids, vec![artifact.id]);
}

#[tokio::test]
async fn cancellation_fences_results_until_active_stage_teardown() {
    let (store, _root) = build_store().await;
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
        .complete_stage(&lease, vec![])
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
    let requested = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(requested.status, RuntimeStageStatus::Running);
    assert_eq!(
        requested.cancellation_state,
        Some(RuntimeCancellationState::Requested)
    );
    assert_eq!(requested.worker_id.as_deref(), Some("worker-1"));
    assert!(requested.lease_expires_at.is_some());
    assert!(store
        .stage_lease_is_active(&lease)
        .await
        .expect("cancelling lease remains owned"));
    let stale = StageLease {
        attempt_token: Some("stale-attempt".to_string()),
        ..lease.clone()
    };
    assert!(!store
        .mark_stage_execution_stopping(&stale)
        .await
        .expect("stale stopping fence"));
    assert!(store
        .finalize_stage_cancellation(&stale)
        .await
        .expect("stale finalization fence")
        .is_none());

    assert!(store
        .mark_stage_execution_stopping(&lease)
        .await
        .expect("mark execution stopping"));
    assert!(store
        .finalize_stage_cancellation(&lease)
        .await
        .expect("finalize cancellation")
        .is_some());
    assert!(store
        .finalize_stage_cancellation(&lease)
        .await
        .expect("duplicate finalization")
        .is_none());

    let cancelled = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
    assert_eq!(cancelled.cancellation_state, None);
    assert_eq!(cancelled.worker_id, None);
    assert_eq!(cancelled.lease_expires_at, None);
}

#[tokio::test]
async fn requested_cancellation_survives_lease_expiry_until_exact_owner_teardown() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 10)
        .await
        .expect("claim")
        .expect("active attempt");
    let lease = claimed.lease().expect("lease");
    store
        .cancel_job(&job.id, Some("cancel".to_string()))
        .await
        .expect("cancel")
        .expect("cancellation requested");

    clock.store(2_000, Ordering::SeqCst);
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recovery must skip uncertain execution"),
        0
    );
    assert_eq!(
        store.stage_lease_state(&lease).await.expect("lease state"),
        Some(StageLeaseState::CancellationRequested)
    );
    assert!(store
        .mark_stage_execution_stopping(&lease)
        .await
        .expect("exact owner marks stopping after expiry"));
    assert!(store
        .renew_stage_lease(&lease, 100)
        .await
        .expect("exact owner renews cancellation lease"));
    let stopping = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(
        stopping.cancellation_state,
        Some(RuntimeCancellationState::ExecutionStopping)
    );
    assert_eq!(stopping.lease_expires_at, Some(2_100));
    assert_eq!(stopping.worker_id.as_deref(), Some("worker-1"));
}

#[tokio::test]
async fn abandoned_cancellation_settles_only_after_owner_stops_renewing_and_heartbeating() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.set_tts_admission_limits_for_test(Some((1, 1)));
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    record_test_worker_heartbeat(&store, "worker-1").await;
    let claimed = store
        .claim_next_stage("worker-1", 10_000)
        .await
        .expect("claim")
        .expect("active attempt");
    let lease = claimed.lease().expect("lease");
    store
        .cancel_job(&job.id, Some("user cancel".to_string()))
        .await
        .expect("cancel")
        .expect("cancellation requested");
    assert!(store.create_job(admission_job(2)).await.is_err());

    // Lease lapsed and the heartbeat is stale, but not yet by the grace.
    clock.store(10_999 + ABANDONED_CANCELLATION_GRACE_MS, Ordering::SeqCst);
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recovery"),
        0
    );

    // A still-heartbeating owner is never settled, however old its lease.
    clock.store(
        11_001 + 2 * ABANDONED_CANCELLATION_GRACE_MS,
        Ordering::SeqCst,
    );
    record_test_worker_heartbeat(&store, "worker-1").await;
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recovery"),
        0
    );
    assert_eq!(
        store.stage_lease_state(&lease).await.expect("lease state"),
        Some(StageLeaseState::CancellationRequested)
    );

    // The owner process died: no renewal and no heartbeat for the grace.
    clock.store(
        11_002 + 3 * ABANDONED_CANCELLATION_GRACE_MS,
        Ordering::SeqCst,
    );
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recovery"),
        1
    );
    let settled = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(settled.status, RuntimeStageStatus::Cancelled);
    assert_eq!(settled.cancellation_state, None);
    assert_eq!(settled.worker_id, None);
    assert_eq!(settled.lease_expires_at, None);
    let settled_job = store
        .get_job(&job.id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(settled_job.status, RuntimeJobStatus::Cancelled);
    assert_eq!(settled_job.cancellation_state, None);
    assert_eq!(
        settled_job.cancellation_reason.as_deref(),
        Some("user cancel")
    );

    // The dead attempt stays fenced, and admission capacity is released.
    assert!(store
        .finalize_stage_cancellation(&lease)
        .await
        .expect("late owner settlement")
        .is_none());
    assert!(!store
        .renew_stage_lease(&lease, 10_000)
        .await
        .expect("late owner renewal"));
    store
        .create_job(admission_job(2))
        .await
        .expect("admission released");
}

#[tokio::test]
async fn abandoned_cancellation_ignores_a_restarted_incarnation_reusing_the_worker_id() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    record_test_worker_heartbeat(&store, "worker-1").await;
    store
        .claim_next_stage("worker-1", 10_000)
        .await
        .expect("claim")
        .expect("active attempt");
    store
        .cancel_job(&job.id, None)
        .await
        .expect("cancel")
        .expect("cancellation requested");

    let now = 11_001 + ABANDONED_CANCELLATION_GRACE_MS;
    clock.store(now, Ordering::SeqCst);
    let db = store.connection().await.expect("database");
    db.execute_raw(
        crate::db::raw::statement(
            db,
            "UPDATE runtime_worker_heartbeats SET started_at = ?1, last_heartbeat_at = ?1 WHERE worker_id = ?2",
            vec![now.into(), "worker-1".into()],
        )
        .expect("statement"),
    )
    .await
    .expect("restarted incarnation heartbeat");

    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recovery"),
        1
    );
    let settled = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(settled.status, RuntimeStageStatus::Cancelled);
    assert_eq!(
        store
            .get_job(&job.id)
            .await
            .expect("job")
            .expect("job exists")
            .status,
        RuntimeJobStatus::Cancelled
    );
}

/// Stage writers (claim, complete) and job writers (cancel) race on separate
/// connections. On PostgreSQL they must lock the job row in one order, or
/// the database aborts one of them as a deadlock.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_claim_completion_and_cancellation_never_deadlock() {
    let (store, _root) = build_store().await;
    for round in 0..16 {
        let (running_job, _) =
            create_test_job_and_stage(&store, 1, &format!("running_{round}"), 1).await;
        let lease = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("active attempt")
            .lease()
            .expect("lease");
        let (queued_job, _) =
            create_test_job_and_stage(&store, 0, &format!("queued_{round}"), 1).await;
        let barrier = Arc::new(tokio::sync::Barrier::new(5));
        let spawn = |work: futures::future::BoxFuture<'static, anyhow::Result<()>>| {
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                work.await
            })
        };
        let tasks = [
            spawn(Box::pin({
                let store = store.clone();
                async move { store.complete_stage(&lease, vec![]).await.map(|_| ()) }
            })),
            spawn(Box::pin({
                let (store, job_id) = (store.clone(), running_job.id.clone());
                async move { store.cancel_job(&job_id, None).await.map(|_| ()) }
            })),
            spawn(Box::pin({
                let store = store.clone();
                async move { store.claim_next_stage("worker-2", 60_000).await.map(|_| ()) }
            })),
            spawn(Box::pin({
                let (store, job_id) = (store.clone(), queued_job.id.clone());
                async move { store.cancel_job(&job_id, None).await.map(|_| ()) }
            })),
        ];
        barrier.wait().await;
        for task in tasks {
            task.await
                .expect("race task joins")
                .unwrap_or_else(|error| panic!("round {round}: {error:#}"));
        }
    }
}

#[tokio::test]
async fn completion_and_cancellation_race_has_one_authoritative_winner() {
    let (store, _root) = build_store().await;
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("active attempt");
    let lease = claimed.lease().expect("lease");
    let barrier = Arc::new(tokio::sync::Barrier::new(3));

    let completion_store = store.clone();
    let completion_lease = lease.clone();
    let completion_barrier = barrier.clone();
    let completion = tokio::spawn(async move {
        completion_barrier.wait().await;
        completion_store
            .complete_stage(&completion_lease, vec![])
            .await
    });
    let cancellation_store = store.clone();
    let cancellation_job_id = job.id.clone();
    let cancellation_barrier = barrier.clone();
    let cancellation = tokio::spawn(async move {
        cancellation_barrier.wait().await;
        cancellation_store
            .cancel_job(&cancellation_job_id, Some("race".to_string()))
            .await
    });
    barrier.wait().await;

    let completed = completion
        .await
        .expect("completion join")
        .expect("completion");
    let cancelled = cancellation
        .await
        .expect("cancellation join")
        .expect("cancellation");
    assert_ne!(completed.is_some(), cancelled.is_some());

    if cancelled.is_some() {
        assert!(store
            .finalize_stage_cancellation(&lease)
            .await
            .expect("finalize cancellation")
            .is_some());
    }
    let terminal_job = store
        .get_job(&job.id)
        .await
        .expect("job")
        .expect("job exists");
    assert!(matches!(
        terminal_job.status,
        RuntimeJobStatus::Completed | RuntimeJobStatus::Cancelled
    ));
    assert_eq!(terminal_job.cancellation_state, None);
    let terminal_stage = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert!(matches!(
        terminal_stage.status,
        RuntimeStageStatus::Completed | RuntimeStageStatus::Cancelled
    ));
    assert_eq!(terminal_stage.worker_id, None);
    assert_eq!(terminal_stage.lease_expires_at, None);
}

#[tokio::test]
async fn retry_policy_delays_claims_and_enforces_attempt_budget() {
    let (store, _root) = build_store().await;
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
            // Due by the store's own (database) clock, which can trail the
            // wall clock by a few milliseconds.
            vec![
                store.now_millis().saturating_sub(1).into(),
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

/// Nodes sharing a database must agree on lease time whatever their own
/// wall clocks say: the store reads the database server's clock.
#[tokio::test]
async fn leases_follow_the_database_clock_not_the_process_wall_clock() {
    let database = TestDatabase::new().await;
    let db = database.database();
    db.skew_wall_clock_for_test(60 * 60 * 1000);
    let store = BatchRuntimeStore::initialize_with_database(db);
    create_test_job_and_stage(&store, 0, "clock_stage", 1).await;

    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("stage claimed");
    let lease_expires_at =
        i64::try_from(claimed.stage.lease_expires_at.expect("lease")).expect("lease fits");
    let expected = current_timestamp_millis() + 60_000;
    assert!(
        (lease_expires_at - expected).abs() < 10_000,
        "lease expires at {lease_expires_at}, expected about {expected}"
    );
}

/// On PostgreSQL a claim passes over a stage another transaction holds (a
/// concurrent claimer mid-commit) instead of waiting on it. SQLite has one
/// writer, so there is nothing to skip there.
#[tokio::test]
async fn postgres_claims_skip_a_stage_another_transaction_holds() {
    let (store, _root) = build_store().await;
    let db = store.connection().await.expect("connection");
    if db.get_database_backend() != DbBackend::Postgres {
        return;
    }
    let (_, held) = create_test_job_and_stage(&store, 1, "held_stage", 1).await;
    let (_, free) = create_test_job_and_stage(&store, 0, "free_stage", 1).await;

    let holder = db.begin().await.expect("holder transaction");
    holder
        .execute_raw(
            raw::statement(
                &holder,
                "SELECT id FROM job_stages WHERE id = ?1 FOR UPDATE",
                vec![held.id.clone().into()],
            )
            .expect("lock statement"),
        )
        .await
        .expect("the head stage is locked");

    let claimed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        store.claim_next_stage("worker-1", 60_000),
    )
    .await
    .expect("the claim does not wait on the locked stage")
    .expect("claim")
    .expect("the unlocked stage is claimed");
    assert_eq!(claimed.stage.id, free.id);
    holder.rollback().await.expect("holder rolls back");

    let next = store
        .claim_next_stage("worker-2", 60_000)
        .await
        .expect("claim")
        .expect("the released stage is claimable");
    assert_eq!(next.stage.id, held.id);
}

/// Every claim filter is in the SQL, so stages a worker cannot run never
/// crowd the candidates out (here, one candidate at a time).
#[tokio::test]
async fn claim_filters_resource_hints_versions_in_sql() {
    let (store, _root) = build_store().await;
    let (_, newer) = create_test_job_and_stage(&store, 1, "newer_hints_stage", 1).await;
    let (_, current) = create_test_job_and_stage(&store, 0, "current_hints_stage", 1).await;
    let db = store.connection().await.expect("connection");
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE job_stages SET resource_hints_version = ?1, resource_hints_json = ?2 WHERE id = ?3",
            vec![
                i64::from(STAGE_RESOURCE_HINTS_VERSION + 1).into(),
                json!({"version": STAGE_RESOURCE_HINTS_VERSION + 1}).to_string().into(),
                newer.id.clone().into(),
            ],
        )
        .expect("version statement"),
    )
    .await
    .expect("the head stage carries a newer hints version");

    let filter = StageClaimFilter {
        max_candidates: 1,
        ..StageClaimFilter::default()
    };
    let claimed = store
        .claim_next_stage_with_filter("worker-1", 60_000, &filter)
        .await
        .expect("claim")
        .expect("the compatible stage is claimed");
    assert_eq!(claimed.stage.id, current.id);
}

use super::*;

#[tokio::test]
async fn registered_worker_heartbeat_tracks_instance_resources_and_capacity() {
    let (store, _root) = build_store().await;
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
async fn queue_health_requires_fresh_queue_coverage() {
    let (store, _root) = build_store().await;
    let (_job, stage) = create_test_job_and_stage(&store, 0, "asr_infer", 1).await;
    assert_eq!(stage.queue_class, QueueClass::BatchAsr);

    let uncovered = store.runtime_queue_health(5_000).await.expect("health");
    assert_eq!(uncovered.queues.len(), 1);
    assert_eq!(
        uncovered.uncovered_queue_classes,
        vec![QueueClass::BatchAsr]
    );

    store
        .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
            registration: RuntimeWorkerRegistration {
                version: WORKER_REGISTRATION_VERSION,
                worker_id: "asr-worker".to_string(),
                instance_id: "asr-worker-instance".to_string(),
                queue_classes: vec![QueueClass::BatchAsr],
                capabilities: vec!["asr".to_string()],
                model_ids: vec![],
                stage_kinds: vec!["asr_infer".to_string()],
                resources: WorkerResourceCapacity::default(),
                software_version: "test".to_string(),
            },
            status: "idle".to_string(),
            current_job_id: None,
            current_stage_id: None,
            details: RuntimeWorkerHeartbeatDetails {
                version: WORKER_HEARTBEAT_DETAILS_VERSION,
                available_slots: 1,
                active_lease_ids: vec![],
                last_error: None,
                health_json: json!({}),
            },
            diagnostic_json: json!({}),
        })
        .await
        .expect("worker heartbeat");

    let covered = store.runtime_queue_health(5_000).await.expect("health");
    assert_eq!(covered.healthy_workers, 1);
    assert!(covered.uncovered_queue_classes.is_empty());

    let db = store.connection().await.expect("database");
    db.execute_raw(
        crate::db::raw::statement(
            db,
            "UPDATE runtime_worker_heartbeats SET last_heartbeat_at = ?1 WHERE worker_id = ?2",
            vec![
                current_timestamp_millis().saturating_sub(10_000).into(),
                "asr-worker".into(),
            ],
        )
        .expect("statement"),
    )
    .await
    .expect("stale heartbeat update");

    let stale = store.runtime_queue_health(1_000).await.expect("health");
    assert_eq!(stale.stale_workers, 1);
    assert_eq!(stale.uncovered_queue_classes, vec![QueueClass::BatchAsr]);
}

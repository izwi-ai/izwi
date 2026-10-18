use super::*;

#[tokio::test]
async fn bounded_reconciliation_repairs_cancelled_transcription_projection() {
    let (store, _root) = build_store().await;
    let record_id = "cancelled-transcription";
    let db = store.connection().await.expect("database");
    db.execute_raw(
        raw::statement(
            db,
            r#"
            INSERT INTO transcription_records (
                id, created_at, processing_status, processing_error,
                processing_progress_json, runtime_stage_id, runtime_attempt_token,
                processing_time_ms, audio_mime_type, audio_storage_path, transcription
            ) VALUES (?1, ?2, 'processing', NULL, '{}', 'old-stage', 'old-attempt', 0, 'audio/wav', '', '')
            "#,
            vec![record_id.into(), current_timestamp_millis().into()],
        )
        .expect("transcription projection statement"),
    )
    .await
    .expect("transcription projection");
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::AsrTranscription,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: None,
            capability: Some("asr".to_string()),
            route_record_kind: Some("transcription".to_string()),
            route_record_id: Some(record_id.to_string()),
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
        .expect("runtime job");
    db.execute_raw(
        raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET status = 'cancelled', cancellation_reason = 'cancelled while offline',
                finished_at = ?1, updated_at = ?1
            WHERE id = ?2
            "#,
            vec![
                (current_timestamp_millis() - CANCELLED_PROJECTION_REPAIR_WINDOW_MS - 1).into(),
                job.id.clone().into(),
            ],
        )
        .expect("simulate terminal job statement"),
    )
    .await
    .expect("simulate terminal job");
    // Cancellations older than the repair window are not scanned.
    assert_eq!(
        store
            .reconcile_inconsistent_states(1)
            .await
            .expect("bounded reconciliation")
            .route_projections_repaired,
        0
    );
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE runtime_jobs SET updated_at = ?1 WHERE id = ?2",
            vec![current_timestamp_millis().into(), job.id.into()],
        )
        .expect("recent cancellation statement"),
    )
    .await
    .expect("recent cancellation");

    let report = store
        .reconcile_inconsistent_states(1)
        .await
        .expect("bounded reconciliation");
    assert_eq!(report.jobs_repaired, 0);
    assert_eq!(report.stages_repaired, 0);
    assert_eq!(report.route_projections_repaired, 1);
    let projection = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT processing_status, processing_error, processing_progress_json, runtime_stage_id, runtime_attempt_token FROM transcription_records WHERE id = ?1",
                vec![record_id.into()],
            )
            .expect("reconciled projection statement"),
        )
        .await
        .expect("reconciled projection query")
        .expect("reconciled projection");
    assert_eq!(
        projection.try_get_by_index::<String>(0).expect("status"),
        "failed"
    );
    assert_eq!(
        projection
            .try_get_by_index::<Option<String>>(1)
            .expect("error")
            .as_deref(),
        Some("cancelled while offline")
    );
    for index in 2..=4 {
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(index)
                .expect("cleared runtime projection field"),
            None
        );
    }
}

#[tokio::test]
async fn cancelled_job_cannot_overwrite_an_active_replacement_projection() {
    let (store, _root) = build_store().await;
    let record_id = "replacement-owned-transcription";
    let db = store.connection().await.expect("database");
    db.execute_raw(
        raw::statement(
            db,
            r#"
            INSERT INTO transcription_records (
                id, created_at, processing_status, processing_error,
                processing_progress_json, runtime_stage_id, runtime_attempt_token,
                processing_time_ms, audio_mime_type, audio_storage_path, transcription
            ) VALUES (?1, ?2, 'processing', NULL, '{}', 'replacement-stage',
                'replacement-attempt', 0, 'audio/wav', '', '')
            "#,
            vec![record_id.into(), current_timestamp_millis().into()],
        )
        .expect("transcription projection statement"),
    )
    .await
    .expect("transcription projection");
    let job = |correlation_id: &str| NewRuntimeJob {
        job_kind: RuntimeJobKind::AsrTranscription,
        status: RuntimeJobStatus::Queued,
        priority: 0,
        model_id: None,
        capability: Some("asr".to_string()),
        route_record_kind: Some("transcription".to_string()),
        route_record_id: Some(record_id.to_string()),
        input_media_asset_id: None,
        input_text_asset_id: None,
        request_json: json!({}),
        model_snapshot_json: json!({}),
        retry_policy_json: json!({}),
        max_attempts: 1,
        idempotency_key: None,
        correlation_id: Some(correlation_id.to_string()),
    };
    let cancelled = store.create_job(job("old")).await.expect("old job");
    let replacement = store
        .create_job(job("replacement"))
        .await
        .expect("replacement job");

    let cancelled = store
        .cancel_job(&cancelled.id, Some("old job cancelled".to_string()))
        .await
        .expect("cancel old job")
        .expect("old job is cancellable");
    assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
    assert_eq!(
        store
            .get_job(&replacement.id)
            .await
            .expect("replacement lookup")
            .expect("replacement job")
            .status,
        RuntimeJobStatus::Queued
    );

    let report = store
        .reconcile_inconsistent_states(10)
        .await
        .expect("bounded reconciliation");
    assert_eq!(report.route_projections_repaired, 0);
    let projection = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT processing_status, processing_error, processing_progress_json, runtime_stage_id, runtime_attempt_token FROM transcription_records WHERE id = ?1",
                vec![record_id.into()],
            )
            .expect("replacement projection statement"),
        )
        .await
        .expect("replacement projection query")
        .expect("replacement projection");
    assert_eq!(
        projection.try_get_by_index::<String>(0).expect("status"),
        "processing"
    );
    assert_eq!(
        projection
            .try_get_by_index::<Option<String>>(1)
            .expect("error"),
        None
    );
    assert_eq!(
        projection
            .try_get_by_index::<Option<String>>(2)
            .expect("progress")
            .as_deref(),
        Some("{}")
    );
    assert_eq!(
        projection
            .try_get_by_index::<Option<String>>(3)
            .expect("stage binding")
            .as_deref(),
        Some("replacement-stage")
    );
    assert_eq!(
        projection
            .try_get_by_index::<Option<String>>(4)
            .expect("attempt binding")
            .as_deref(),
        Some("replacement-attempt")
    );
}

#[tokio::test]
async fn reconcile_fails_jobs_orphaned_without_stages_and_releases_admission() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.set_tts_admission_limits_for_test(Some((1, 1)));
    let orphan = store.create_job(admission_job(1)).await.expect("job");
    assert!(store.create_job(admission_job(2)).await.is_err());

    // An enqueue still inside its grace may be about to create stages.
    clock.store(1_000 + STAGELESS_JOB_GRACE_MS - 1, Ordering::SeqCst);
    let report = store
        .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
        .expect("reconcile");
    assert_eq!(report.jobs_repaired, 0);

    clock.store(1_000 + STAGELESS_JOB_GRACE_MS, Ordering::SeqCst);
    let report = store
        .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
        .expect("reconcile");
    assert_eq!(report.jobs_repaired, 1);
    let failed = store
        .get_job(&orphan.id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(failed.status, RuntimeJobStatus::Failed);
    assert_eq!(failed.error_code.as_deref(), Some("enqueue_incomplete"));
    assert!(failed.finished_at.is_some());
    store
        .create_job(admission_job(2))
        .await
        .expect("admission released");
}

#[tokio::test]
async fn reconcile_keeps_old_jobs_that_have_stages() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;

    clock.store(1_000 + 10 * STAGELESS_JOB_GRACE_MS, Ordering::SeqCst);
    let report = store
        .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
        .expect("reconcile");
    assert_eq!(report.jobs_repaired, 0);
    assert_eq!(
        store
            .get_job(&job.id)
            .await
            .expect("job")
            .expect("job exists")
            .status,
        RuntimeJobStatus::Queued
    );
}

#[tokio::test]
async fn reconciliation_repairs_crash_interrupted_terminal_transitions() {
    let (store, _root) = build_store().await;
    let (completed_job, completed_stage) =
        create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-complete", 60_000)
        .await
        .expect("claim")
        .expect("attempt");
    let db = store.connection().await.expect("database");
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE job_stages SET status = 'completed', worker_id = NULL, lease_expires_at = NULL, finished_at = ?1 WHERE id = ?2",
            vec![current_timestamp_millis().into(), claimed.stage.id.into()],
        )
        .expect("crash completion statement"),
    )
    .await
    .expect("crash completion");

    let (cancelled_job, cancelled_stage) =
        create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE runtime_jobs SET status = 'cancelled', finished_at = ?1 WHERE id = ?2",
            vec![
                current_timestamp_millis().into(),
                cancelled_job.id.clone().into(),
            ],
        )
        .expect("crash cancellation statement"),
    )
    .await
    .expect("crash cancellation");

    let report = store
        .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
        .await
        .expect("reconciliation");
    assert!(report.jobs_repaired >= 1);
    assert!(report.stages_repaired >= 1);
    assert_eq!(
        store
            .get_job(&completed_job.id)
            .await
            .expect("completed job")
            .expect("completed job exists")
            .status,
        RuntimeJobStatus::Completed
    );
    assert_eq!(
        store
            .get_stage(&completed_stage.id)
            .await
            .expect("completed stage")
            .expect("completed stage exists")
            .status,
        RuntimeStageStatus::Completed
    );
    let cancelled = store
        .get_stage(&cancelled_stage.id)
        .await
        .expect("cancelled stage")
        .expect("cancelled stage exists");
    assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
    assert!(cancelled.worker_id.is_none());
    assert!(cancelled.lease_expires_at.is_none());
}

#[tokio::test]
async fn reconciliation_consumes_one_shared_bounded_budget() {
    let (store, _root) = build_store().await;
    let mut job_ids = Vec::new();
    let db = store.connection().await.expect("database");

    for _ in 0..3 {
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE job_stages SET status = 'completed', worker_id = NULL, lease_expires_at = NULL, finished_at = ?1 WHERE id = ?2",
                vec![current_timestamp_millis().into(), stage.id.into()],
            )
            .expect("crash completion statement"),
        )
        .await
        .expect("crash completion");
        job_ids.push(job.id);
    }

    let first = store
        .reconcile_inconsistent_states(2)
        .await
        .expect("first reconciliation batch");
    assert_eq!(first.jobs_repaired, 2);
    assert_eq!(first.stages_repaired, 0);
    let mut completed = 0;
    for job_id in &job_ids {
        if store
            .get_job(job_id)
            .await
            .expect("job")
            .expect("job exists")
            .status
            == RuntimeJobStatus::Completed
        {
            completed += 1;
        }
    }
    assert_eq!(completed, 2);

    let second = store
        .reconcile_inconsistent_states(2)
        .await
        .expect("second reconciliation batch");
    assert_eq!(second.jobs_repaired, 1);
    assert_eq!(second.stages_repaired, 0);
}

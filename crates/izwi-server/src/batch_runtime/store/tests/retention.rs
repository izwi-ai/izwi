use super::*;

/// DUR-F6: the sweeper deletes only state past its retention, in bounded
/// pages, and never a job that still owns live job-scoped media.
#[tokio::test]
async fn retention_sweep_deletes_only_expired_terminal_state() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let policy = retention_policy();

    let old_done = retention_job(&store, RuntimeJobStatus::Completed, Some("old")).await;
    let old_failed = retention_job(&store, RuntimeJobStatus::Failed, None).await;
    let old_active = retention_job(&store, RuntimeJobStatus::Queued, None).await;
    let owns_media = retention_job(&store, RuntimeJobStatus::Completed, None).await;
    let media = opaque_media(&store, "artifact_job").await;
    store
        .create_artifact(NewRuntimeArtifact {
            job_id: owns_media.id.clone(),
            stage_id: None,
            artifact_kind: RuntimeArtifactKind::Audio,
            artifact_role: RuntimeArtifactRole::OutputIntermediate,
            media_asset_id: Some(media.id.clone()),
            text_asset_id: None,
            storage_key: None,
            content_type: None,
            filename: None,
            size_bytes: None,
            sha256: None,
            metadata_json: json!({}),
            retention_policy: "speech_job_replay".to_string(),
        })
        .await
        .unwrap();
    record_test_worker_heartbeat(&store, "dead-worker").await;
    let db = store.connection().await.unwrap();
    db.execute_unprepared(
        "INSERT INTO idempotency_keys (operation, idempotency_key, created_at, request_hash) VALUES ('transcription.create', 'legacy', 1000, 'hash')",
    )
    .await
    .unwrap();

    clock.store(5_000, Ordering::SeqCst);
    let recent_done = retention_job(&store, RuntimeJobStatus::Completed, None).await;
    record_test_worker_heartbeat(&store, "live-worker").await;

    assert_eq!(
        store.sweep_retention(policy, 64).await.unwrap(),
        RetentionReport::default(),
        "nothing is past retention yet"
    );

    clock.store(12_000, Ordering::SeqCst);
    let first = store.sweep_retention(policy, 1).await.unwrap();
    assert_eq!(first.jobs, 1, "jobs are deleted in bounded pages");
    let second = store.sweep_retention(policy, 64).await.unwrap();
    assert_eq!(second.jobs, 1);
    assert_eq!(first.worker_heartbeats + second.worker_heartbeats, 1);
    assert_eq!(
        first.legacy_idempotency_records + second.legacy_idempotency_records,
        1
    );
    for job in [&old_done, &old_failed] {
        assert!(store.get_job(&job.id).await.unwrap().is_none());
        assert!(store.list_stages_for_job(&job.id).await.unwrap().is_empty());
        assert!(store
            .list_artifacts_for_job(&job.id)
            .await
            .unwrap()
            .is_empty());
    }
    assert_eq!(
        count_where(&store, "SELECT COUNT(*) FROM text_assets", vec![]).await,
        0,
        "the deleted job's text input went with it"
    );
    for job in [&old_active, &owns_media, &recent_done] {
        assert!(store.get_job(&job.id).await.unwrap().is_some());
    }
    assert!(store
        .get_worker_heartbeat("dead-worker")
        .await
        .unwrap()
        .is_none());
    assert!(store
        .get_worker_heartbeat("live-worker")
        .await
        .unwrap()
        .is_some());

    // Once its job-scoped media is tombstoned, the job can go too.
    let expired = store.expired_retained_media(policy, 64).await.unwrap();
    assert_eq!(
        expired
            .iter()
            .map(|asset| asset.id.clone())
            .collect::<Vec<_>>(),
        vec![media.id.clone()]
    );
    assert!(store
        .tombstone_media_asset_with_cleanup(
            &media.id,
            "tenant-a",
            ArtifactCleanupReason::ArtifactDeleted
        )
        .await
        .unwrap());
    assert_eq!(store.sweep_retention(policy, 64).await.unwrap().jobs, 1);
    assert!(store.get_job(&owns_media.id).await.unwrap().is_none());
    // The tombstone stays while its cleanup intent is pending.
    clock.store(30_000, Ordering::SeqCst);
    assert_eq!(
        store
            .sweep_retention(policy, 64)
            .await
            .unwrap()
            .media_tombstones,
        0
    );
    assert!(store.get_media_asset(&media.id).await.unwrap().is_some());
}

/// DUR-F6: replay GC visits a terminal job's journal once; a job with no
/// journal is marked swept and not scanned again, until a retry.
#[tokio::test]
async fn speech_replay_gc_marks_jobs_without_a_journal_swept() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let job = retention_job(&store, RuntimeJobStatus::Failed, None).await;
    clock.store(5_000, Ordering::SeqCst);
    assert!(store.expired_speech_pcm(4_000).await.unwrap().is_empty());
    let swept = count_where(
        &store,
        "SELECT COUNT(*) FROM runtime_jobs WHERE id = ?1 AND replay_swept_at IS NOT NULL",
        vec![job.id.clone().into()],
    );
    assert_eq!(swept.await, 1);

    let db = store.connection().await.unwrap();
    db.execute_unprepared(
        "UPDATE runtime_jobs SET max_attempts = 2; UPDATE job_stages SET status = 'failed', max_attempts = 2",
    )
    .await
    .unwrap();
    store.retry_job(&job.id).await.unwrap().expect("retryable");
    assert_eq!(
        count_where(
            &store,
            "SELECT COUNT(*) FROM runtime_jobs WHERE id = ?1 AND replay_swept_at IS NULL",
            vec![job.id.into()],
        )
        .await,
        1,
        "a retried job's new journal is swept again"
    );
}

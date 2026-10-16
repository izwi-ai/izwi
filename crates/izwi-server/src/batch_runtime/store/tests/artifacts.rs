use super::*;

#[tokio::test]
async fn attempt_token_fences_same_worker_and_attempt_publication() {
    let (store, _root) = build_store().await;
    let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("attempt");
    let lease = claimed.lease().expect("lease");
    assert!(lease.attempt_token.is_some());

    let forged = StageLease {
        attempt_token: Some("wrong-attempt-token".to_string()),
        ..lease.clone()
    };
    assert!(store
        .complete_stage(&forged, vec![])
        .await
        .expect("forged completion")
        .is_none());
    let running = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(running.status, RuntimeStageStatus::Running);
    assert!(running.output_artifact_ids.is_empty());

    let artifact = store
        .publish_stage_output_artifact(&lease, test_stage_output("current-output"))
        .await
        .expect("publish current output")
        .expect("current attempt owns output");
    assert!(store
        .complete_stage(&lease, vec![artifact.id])
        .await
        .expect("owned completion")
        .is_some());
}

#[tokio::test]
async fn attempt_owned_artifact_publication_is_idempotent() {
    let (store, _root) = build_store().await;
    let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("attempt");
    let lease = claimed.lease().expect("lease");
    assert!(store
        .stage_lease_is_active(&lease)
        .await
        .expect("active lease"));

    let first = store
        .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
        .await
        .expect("first publication")
        .expect("active publication");
    let duplicate = store
        .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
        .await
        .expect("duplicate publication")
        .expect("idempotent publication");

    assert_eq!(duplicate.id, first.id);
    assert_eq!(first.job_id, job.id);
    assert_eq!(first.stage_id.as_deref(), Some(lease.stage_id.as_str()));
    assert_eq!(first.producer_attempt_count, Some(lease.attempt_count));
    assert_eq!(
        first.producer_attempt_token.as_deref(),
        lease.attempt_token.as_deref()
    );
    assert_eq!(first.publication_key.as_deref(), Some("primary-result"));
    let artifacts = store
        .list_artifacts_for_job(&job.id)
        .await
        .expect("artifacts");
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0].id, first.id);
}

#[tokio::test]
async fn stage_completion_rejects_unbounded_or_unowned_output_references_atomically() {
    let (store, _root) = build_store().await;
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-a", 60_000)
        .await
        .expect("claim")
        .expect("attempt");
    let lease = claimed.lease().expect("lease");
    let owned = store
        .publish_stage_output_artifact(&lease, test_stage_output("owned-output"))
        .await
        .expect("publish owned output")
        .expect("attempt owns output");

    let too_many = (0..=MAX_STAGE_OUTPUT_ARTIFACTS)
        .map(|_| new_uuid())
        .collect::<Vec<_>>();
    assert!(store
        .complete_stage(&lease, too_many)
        .await
        .unwrap_err()
        .to_string()
        .contains("count exceeds"));
    assert!(store
        .complete_stage(
            &lease,
            vec!["x".repeat(MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES + 1)],
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("byte limit"));
    assert!(store
        .complete_stage(&lease, vec!["not-a-uuid".to_string()])
        .await
        .unwrap_err()
        .to_string()
        .contains("canonical UUIDs"));
    assert!(store
        .complete_stage(&lease, vec![owned.id.clone(), owned.id.clone()])
        .await
        .unwrap_err()
        .to_string()
        .contains("must be unique"));
    assert!(store
        .complete_stage(&lease, vec![new_uuid()])
        .await
        .unwrap_err()
        .to_string()
        .contains("not owned"));

    let (foreign_job, foreign_stage) =
        create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let foreign_claim = store
        .claim_next_stage("worker-b", 60_000)
        .await
        .expect("foreign claim")
        .expect("foreign attempt");
    assert_eq!(foreign_claim.job.id, foreign_job.id);
    let foreign_lease = foreign_claim.lease().expect("foreign lease");
    let foreign = store
        .publish_stage_output_artifact(&foreign_lease, test_stage_output("foreign-output"))
        .await
        .expect("publish foreign output")
        .expect("foreign attempt owns output");
    assert!(store
        .complete_stage(&lease, vec![foreign.id.clone()])
        .await
        .unwrap_err()
        .to_string()
        .contains("not owned"));

    let db = store.db.connection().await.expect("database");
    db.execute_raw(
        raw::statement(
            db,
            r#"
            UPDATE runtime_artifacts
            SET job_id = ?1,
                producer_attempt_count = ?2,
                producer_attempt_token = ?3
            WHERE id = ?4
            "#,
            vec![
                job.id.clone().into(),
                u32_to_i64_value(lease.attempt_count).into(),
                lease.attempt_token.clone().unwrap().into(),
                foreign.id.clone().into(),
            ],
        )
        .expect("cross-stage corruption statement"),
    )
    .await
    .expect("inject cross-stage artifact row");
    assert_ne!(foreign_stage.id, stage.id);
    assert!(store
        .complete_stage(&lease, vec![foreign.id])
        .await
        .unwrap_err()
        .to_string()
        .contains("not owned"));

    let still_running = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(still_running.status, RuntimeStageStatus::Running);
    assert!(still_running.output_artifact_ids.is_empty());
    let completed = store
        .complete_stage(&lease, vec![owned.id.clone()])
        .await
        .expect("valid completion")
        .expect("exact attempt completes");
    assert_eq!(completed.output_artifact_ids, vec![owned.id]);
}

#[tokio::test]
async fn reclaimed_attempt_cannot_adopt_a_previous_attempts_output() {
    let clock = Arc::new(AtomicI64::new(current_timestamp_millis()));
    let (mut store, _root) = build_store().await;
    store.set_test_clock(clock.clone());
    let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    let first = store
        .claim_next_stage("worker-a", 100)
        .await
        .expect("first claim")
        .expect("first attempt");
    let first_lease = first.lease().expect("first lease");
    let old_output = store
        .publish_stage_output_artifact(&first_lease, test_stage_output("old-output"))
        .await
        .expect("publish old output")
        .expect("first attempt owns output");

    clock.fetch_add(101, Ordering::SeqCst);
    assert_eq!(
        store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recover first attempt"),
        1
    );
    let second = store
        .claim_next_stage("worker-b", 60_000)
        .await
        .expect("second claim")
        .expect("second attempt");
    let second_lease = second.lease().expect("second lease");
    assert!(store
        .complete_stage(&second_lease, vec![old_output.id])
        .await
        .unwrap_err()
        .to_string()
        .contains("not owned"));
    let running = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(running.status, RuntimeStageStatus::Running);
    assert!(running.output_artifact_ids.is_empty());

    let current_output = store
        .publish_stage_output_artifact(&second_lease, test_stage_output("current-output"))
        .await
        .expect("publish current output")
        .expect("second attempt owns output");
    store
        .complete_stage(&second_lease, vec![current_output.id])
        .await
        .expect("complete current attempt")
        .expect("current attempt completes");
}

#[tokio::test]
async fn stale_and_cancelled_attempts_cannot_publish_artifacts() {
    let (store, _root) = build_store().await;
    let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    let first = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("first claim")
        .expect("first attempt");
    let first_lease = first.lease().expect("first lease");
    assert!(store
        .stage_lease_is_active(&first_lease)
        .await
        .expect("first lease active"));

    store
        .fail_stage(
            &first_lease,
            true,
            Some("retry".to_string()),
            Some("replace attempt".to_string()),
        )
        .await
        .expect("retry first attempt")
        .expect("retrying stage");
    let second = store
        .claim_next_stage("worker-2", 60_000)
        .await
        .expect("second claim")
        .expect("replacement attempt");
    let second_lease = second.lease().expect("second lease");

    assert!(!store
        .stage_lease_is_active(&first_lease)
        .await
        .expect("stale lease check"));
    assert!(store
        .stage_lease_is_active(&second_lease)
        .await
        .expect("replacement lease check"));
    assert!(store
        .publish_stage_output_artifact(&first_lease, test_stage_output("stale-result"))
        .await
        .expect("stale publication")
        .is_none());

    store
        .cancel_job(&job.id, Some("cancel active attempt".to_string()))
        .await
        .expect("cancel job")
        .expect("cancelled job");
    assert!(store
        .stage_lease_is_active(&second_lease)
        .await
        .expect("cancelling lease check"));
    assert!(store
        .publish_stage_output_artifact(&second_lease, test_stage_output("cancelled-result"))
        .await
        .expect("cancelled publication")
        .is_none());
    assert!(store
        .mark_stage_execution_stopping(&second_lease)
        .await
        .expect("mark execution stopping"));
    assert!(store
        .finalize_stage_cancellation(&second_lease)
        .await
        .expect("finalize cancellation")
        .is_some());
    assert!(!store
        .stage_lease_is_active(&second_lease)
        .await
        .expect("terminal lease check"));
    assert!(store
        .list_artifacts_for_job(&job.id)
        .await
        .expect("artifacts")
        .is_empty());
}

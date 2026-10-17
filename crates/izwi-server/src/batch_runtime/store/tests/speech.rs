use super::*;

#[tokio::test]
async fn speech_replay_gc_fences_retry_before_storage_deletion() {
    for retry_first in [false, true] {
        let (mut store, _root) = build_store().await;
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let (job, _) = create_test_job_and_stage(&store, 0, "speech", 3).await;
        let claim = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        let lease = claim.lease().unwrap();
        let artifact = store
            .publish_stage_output_artifact(
                &lease,
                test_stage_output("speech-pcm/00000000000000000000"),
            )
            .await
            .unwrap()
            .unwrap();
        store
            .fail_stage(
                &lease,
                false,
                Some("injected".into()),
                Some("original failure".into()),
            )
            .await
            .unwrap()
            .unwrap();
        clock.store(10_000, Ordering::SeqCst);
        if retry_first {
            assert!(store.retry_job(&job.id).await.unwrap().is_some());
            assert!(store.expired_speech_pcm(5_000).await.unwrap().is_empty());
        } else {
            let expired = store.expired_speech_pcm(5_000).await.unwrap();
            assert_eq!(expired.len(), 1);
            assert_eq!(expired[0].id, artifact.id);
            assert!(store.retry_job(&job.id).await.unwrap().is_none());
            let job = store.get_job(&job.id).await.unwrap().unwrap();
            assert_eq!(job.error_code.as_deref(), Some("speech_replay_expired"));
            assert_eq!(job.error_message.as_deref(), Some("original failure"));
            // Interrupted deletion remains discoverable and cannot re-enable retry.
            assert_eq!(store.expired_speech_pcm(5_000).await.unwrap().len(), 1);
            store
                .remove_speech_pcm_artifact(&artifact.id)
                .await
                .unwrap();
            assert!(store.expired_speech_pcm(5_000).await.unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn manual_speech_replay_deletion_fences_concurrent_retry() {
    for retry_first in [false, true] {
        let (store, _root) = build_store().await;
        let (job, _) = create_test_job_and_stage(&store, 0, "speech", 3).await;
        let claim = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        let artifact = store
            .publish_stage_output_artifact(
                &claim.lease().unwrap(),
                test_stage_output("speech-pcm/00000000000000000000"),
            )
            .await
            .unwrap()
            .unwrap();
        let lease = claim.lease().unwrap();
        store
            .cancel_job(&job.id, Some("delete recording".into()))
            .await
            .unwrap()
            .unwrap();
        store
            .finalize_stage_cancellation(&lease)
            .await
            .unwrap()
            .unwrap();
        if retry_first {
            assert!(store.retry_job(&job.id).await.unwrap().is_some());
            assert!(!store.fence_speech_replay_deletion(&job.id).await.unwrap());
            assert_eq!(
                store.speech_pcm_after(&job.id, None, 1).await.unwrap()[0].id,
                artifact.id
            );
        } else {
            assert!(store.fence_speech_replay_deletion(&job.id).await.unwrap());
            assert!(store.retry_job(&job.id).await.unwrap().is_none());
            // Retrying an interrupted deletion keeps the fence in force.
            assert!(store.fence_speech_replay_deletion(&job.id).await.unwrap());
            store
                .remove_speech_pcm_artifact(&artifact.id)
                .await
                .unwrap();
            assert!(store
                .speech_pcm_after(&job.id, None, 1)
                .await
                .unwrap()
                .is_empty());
        }
    }
}

#[tokio::test]
async fn speech_admission_preflight_has_no_reservation_and_rejects_overload() {
    let (store, _root) = build_store().await;
    store.set_tts_admission_limits_for_test(Some((1, 1)));
    store
        .preflight_speech_admission(Some([1; 32]))
        .await
        .unwrap();
    store
        .preflight_speech_admission(Some([1; 32]))
        .await
        .unwrap();
    store.create_job(admission_job(1)).await.unwrap();
    let exhausted = store
        .preflight_speech_admission(Some([1; 32]))
        .await
        .unwrap_err();
    assert!(exhausted
        .to_string()
        .contains("Speech job admission capacity exhausted"));
    assert_eq!(
        crate::error::store_api_error(exhausted, "storage").status,
        axum::http::StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn speech_admission_serializes_concurrent_enqueue_and_terminal_releases_capacity() {
    let (store, _root) = build_store().await;
    store.set_tts_admission_limits_for_test(Some((2, 2)));
    store.connection().await.unwrap();
    let mut tasks = Vec::new();
    for tenant in 0..8 {
        let store = store.clone();
        tasks.push(tokio::spawn(async move {
            store.create_job(admission_job(tenant)).await
        }));
    }
    let mut admitted = Vec::new();
    let mut rejected = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(job) => admitted.push(job),
            Err(error) => {
                assert!(
                    error
                        .to_string()
                        .contains("Speech job admission capacity exhausted"),
                    "{error}"
                );
                rejected += 1;
            }
        }
    }
    assert_eq!(admitted.len(), 2);
    assert_eq!(rejected, 6);
    store
        .cancel_job(&admitted[0].id, Some("release slot".into()))
        .await
        .unwrap()
        .unwrap();
    store.create_job(admission_job(99)).await.unwrap();
    assert!(store.create_job(admission_job(100)).await.is_err());
}

#[tokio::test]
async fn speech_admission_separates_tenants_and_retry_obeys_the_same_capacity() {
    let (store, _root) = build_store().await;
    store.set_tts_admission_limits_for_test(Some((3, 1)));
    let first = store.create_job(admission_job(1)).await.unwrap();
    assert!(store.create_job(admission_job(1)).await.is_err());
    let other = store.create_job(admission_job(2)).await.unwrap();
    store.cancel_job(&other.id, None).await.unwrap();
    store
        .create_stage(NewJobStage {
            job_id: first.id.clone(),
            sequence: 0,
            stage_kind: "speech".into(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".into()),
            model_id: None,
            max_attempts: 2,
            input_artifact_ids: vec![],
        })
        .await
        .unwrap();
    let claim = store
        .claim_next_stage("worker", 60_000)
        .await
        .unwrap()
        .unwrap();
    store
        .fail_stage(
            &claim.lease().unwrap(),
            false,
            Some("injected".into()),
            None,
        )
        .await
        .unwrap()
        .unwrap();
    let replacement = store.create_job(admission_job(1)).await.unwrap();
    let error = store.retry_job(&first.id).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("Speech job admission capacity exhausted"));
    assert_eq!(
        store.get_job(&first.id).await.unwrap().unwrap().status,
        RuntimeJobStatus::Failed
    );
    store.cancel_job(&replacement.id, None).await.unwrap();
    assert!(store.retry_job(&first.id).await.unwrap().is_some());
}

#[tokio::test]
async fn active_tts_cancellation_updates_projection_only_after_teardown() {
    let (store, _root) = build_store().await;
    let accepted = match store
        .accept_durable_text_tts(durable_text_tts_acceptance(None))
        .await
        .expect("durable TTS acceptance")
    {
        DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
        DurableTextTtsAcceptanceOutcome::ReservationLost => {
            panic!("unkeyed acceptance cannot lose a reservation")
        }
    };
    let claimed = store
        .claim_next_stage("tts-worker", 60_000)
        .await
        .expect("claim")
        .expect("active TTS attempt");
    let lease = claimed.lease().expect("lease");
    let attempt_token = lease
        .attempt_token
        .clone()
        .expect("claimed stage attempt token");
    let db = store.connection().await.expect("database");
    db.execute_raw(
        raw::statement(
            db,
            r#"
            UPDATE speech_history_records
            SET processing_status = 'processing', runtime_stage_id = ?1, runtime_attempt_token = ?2
            WHERE id = ?3
            "#,
            vec![
                lease.stage_id.clone().into(),
                attempt_token.into(),
                accepted.record.id.clone().into(),
            ],
        )
        .expect("projection update statement"),
    )
    .await
    .expect("mark projection processing");

    let requested = store
        .cancel_job(&accepted.job.id, Some("user cancelled speech".to_string()))
        .await
        .expect("request cancellation")
        .expect("active job");
    assert_eq!(requested.status, RuntimeJobStatus::Running);
    let processing = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT processing_status, processing_error FROM speech_history_records WHERE id = ?1",
                vec![accepted.record.id.clone().into()],
            )
            .expect("processing projection statement"),
        )
        .await
        .expect("processing projection query")
        .expect("processing projection");
    assert_eq!(
        processing
            .try_get_by_index::<String>(0)
            .expect("processing status"),
        "processing"
    );
    assert_eq!(
        processing
            .try_get_by_index::<Option<String>>(1)
            .expect("processing error"),
        None
    );

    assert!(store
        .mark_stage_execution_stopping(&lease)
        .await
        .expect("mark stopping"));
    store
        .finalize_stage_cancellation(&lease)
        .await
        .expect("finalize cancellation")
        .expect("exact attempt finalizes");
    let terminal = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT processing_status, processing_error, runtime_stage_id, runtime_attempt_token FROM speech_history_records WHERE id = ?1",
                vec![accepted.record.id.into()],
            )
            .expect("terminal projection statement"),
        )
        .await
        .expect("terminal projection query")
        .expect("terminal projection");
    assert_eq!(
        terminal.try_get_by_index::<String>(0).expect("status"),
        "failed"
    );
    assert_eq!(
        terminal
            .try_get_by_index::<Option<String>>(1)
            .expect("error")
            .as_deref(),
        Some("user cancelled speech")
    );
    assert_eq!(
        terminal
            .try_get_by_index::<Option<String>>(2)
            .expect("stage binding"),
        None
    );
    assert_eq!(
        terminal
            .try_get_by_index::<Option<String>>(3)
            .expect("attempt binding"),
        None
    );
}

#[tokio::test]
async fn speech_replay_is_cursor_ordered_bounded_and_attempt_fenced() {
    let (store, _root) = build_store().await;
    let (job, _) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .unwrap()
        .unwrap();
    let lease = claimed.lease().unwrap();
    // Publish out of insertion order to prove cursor ordering, not timestamps.
    for sequence in [10, 0, 2, 1] {
        let key = crate::batch_runtime::speech_progress::pcm_publication_key(sequence);
        let mut artifact = test_stage_output(&key);
        artifact.artifact_role = RuntimeArtifactRole::OutputIntermediate;
        store
            .publish_stage_output_artifact(&lease, artifact)
            .await
            .unwrap()
            .unwrap();
    }
    store
        .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
        .await
        .unwrap()
        .unwrap();
    let first = store.speech_pcm_after(&job.id, None, 2).await.unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(
        first[0].publication_key.as_deref(),
        Some("speech-pcm/00000000000000000000")
    );
    assert_eq!(
        first[1].publication_key.as_deref(),
        Some("speech-pcm/00000000000000000001")
    );
    let remaining = store.speech_pcm_after(&job.id, Some(1), 64).await.unwrap();
    assert_eq!(remaining.len(), 2);
    assert_eq!(
        remaining[0].publication_key.as_deref(),
        Some("speech-pcm/00000000000000000002")
    );
    assert!(store
        .speech_pcm_after("other-job", None, 64)
        .await
        .unwrap()
        .is_empty());
    let forged = StageLease {
        attempt_token: Some("stale".to_string()),
        ..lease.clone()
    };
    assert!(!store
        .update_stage_progress(&forged, json!({"completed_segments": 999}))
        .await
        .unwrap());
    assert!(store
        .publish_stage_output_artifact(
            &forged,
            test_stage_output("speech-pcm/00000000000000000011")
        )
        .await
        .unwrap()
        .is_none());
    store
        .cancel_job(&job.id, Some("stop".to_string()))
        .await
        .unwrap();
    assert!(!store
        .update_stage_progress(&lease, json!({"completed_segments": 999}))
        .await
        .unwrap());
    assert!(store
        .publish_stage_output_artifact(
            &lease,
            test_stage_output("speech-pcm/00000000000000000011")
        )
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .speech_pcm_after(&job.id, None, 64)
            .await
            .unwrap()
            .len(),
        4
    );
}

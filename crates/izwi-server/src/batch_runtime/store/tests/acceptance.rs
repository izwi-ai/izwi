use super::*;

#[tokio::test]
async fn durable_text_tts_acceptance_rolls_back_every_partial_graph_failpoint() {
    for failpoint in [
        DurableAcceptanceFailpoint::Projection,
        DurableAcceptanceFailpoint::TextAsset,
        DurableAcceptanceFailpoint::Job,
        DurableAcceptanceFailpoint::Artifact,
        DurableAcceptanceFailpoint::Stage,
        DurableAcceptanceFailpoint::Idempotency,
    ] {
        let (mut store, _root) = build_store().await;
        let request = durable_idempotency_request(
            "tenant-a",
            "speech.text_to_speech.create.v1",
            "rollback-key",
            b"hello durable world",
            60_000,
        );
        let reservation = match store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("reserve acceptance")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        store.set_durable_acceptance_failpoint(Some(failpoint));
        let error = store
            .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
            .await
            .expect_err("failpoint must roll back acceptance");
        assert!(error.to_string().contains("Injected durable acceptance"));
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [0, 0, 0, 0, 0, 1],
            "partial graph survived {failpoint:?}"
        );
        assert!(store
            .claim_next_stage("worker-after-rollback", 60_000)
            .await
            .expect("claim after rollback")
            .is_none());
        assert!(matches!(
            store
                .reserve_durable_idempotency(request)
                .await
                .expect("reservation remains fenced"),
            DurableIdempotencyBegin::InProgress { .. }
        ));
    }
}

#[tokio::test]
async fn durable_job_acceptance_rolls_back_every_partial_graph_failpoint() {
    for failpoint in [
        DurableAcceptanceFailpoint::TextAsset,
        DurableAcceptanceFailpoint::Job,
        DurableAcceptanceFailpoint::Artifact,
        DurableAcceptanceFailpoint::Stage,
        DurableAcceptanceFailpoint::Idempotency,
    ] {
        let (mut store, _root) = build_store().await;
        let media = durable_job_media_asset(&store).await;
        let request = durable_idempotency_request(
            "tenant-a",
            "speech.text_to_speech.create",
            "job-rollback-key",
            b"clone me",
            60_000,
        );
        let DurableIdempotencyBegin::Acquired(reservation) = store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("reserve acceptance")
        else {
            panic!("reservation must be acquired");
        };
        store.set_durable_acceptance_failpoint(Some(failpoint));
        let error = store
            .accept_durable_job(durable_speech_job_acceptance(&media, Some(reservation)))
            .await
            .expect_err("failpoint must roll back acceptance");
        assert!(error.to_string().contains("Injected durable acceptance"));
        // Text assets, jobs, artifacts, stages; the reservation survives.
        assert_eq!(
            durable_job_row_counts(&store).await,
            [0, 0, 0, 0, 1],
            "partial graph survived {failpoint:?}"
        );
        assert!(store
            .claim_next_stage("worker-after-rollback", 60_000)
            .await
            .expect("claim after rollback")
            .is_none());
        assert!(matches!(
            store
                .reserve_durable_idempotency(request)
                .await
                .expect("reservation remains fenced"),
            DurableIdempotencyBegin::InProgress { .. }
        ));
    }
}

#[tokio::test]
async fn durable_job_acceptance_commits_one_claimable_graph_and_replays() {
    let (store, _root) = build_store().await;
    let media = durable_job_media_asset(&store).await;
    let request = durable_idempotency_request(
        "tenant-a",
        "speech.text_to_speech.create",
        "job-key",
        b"clone me",
        60_000,
    );
    let DurableIdempotencyBegin::Acquired(reservation) = store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("reserve")
    else {
        panic!("reservation must be acquired");
    };
    let input = durable_speech_job_acceptance(&media, Some(reservation));
    let text_id = input.text_input.as_ref().unwrap().id.clone();
    let DurableJobAcceptanceOutcome::Committed(accepted) =
        store.accept_durable_job(input).await.expect("accept")
    else {
        panic!("reservation unexpectedly lost");
    };
    assert_eq!(durable_job_row_counts(&store).await, [1, 1, 2, 1, 1]);
    assert_eq!(accepted.job.status, RuntimeJobStatus::Queued);
    assert_eq!(
        accepted.job.input_text_asset_id.as_deref(),
        Some(text_id.as_str())
    );
    assert_eq!(accepted.job.idempotency_key.as_deref(), Some("job-key"));
    assert_eq!(accepted.stages.len(), 1);
    assert_eq!(
        accepted.stages[0].input_artifact_ids,
        accepted
            .artifacts
            .iter()
            .map(|artifact| artifact.id.clone())
            .collect::<Vec<_>>()
    );
    assert!(accepted
        .artifacts
        .iter()
        .all(|artifact| artifact.job_id == accepted.job.id));
    let claimed = store
        .claim_next_stage("worker-a", 60_000)
        .await
        .expect("claim")
        .expect("accepted stage is claimable");
    assert_eq!(claimed.job.id, accepted.job.id);

    let DurableIdempotencyBegin::Replay(replay) = store
        .reserve_durable_idempotency(request)
        .await
        .expect("replay")
    else {
        panic!("committed key must replay");
    };
    assert_eq!(replay.runtime_job_id, accepted.job.id);
    assert_eq!(replay.response_json, json!({"record_id": "record-1"}));
}

#[tokio::test]
async fn durable_job_acceptance_rejects_partial_graphs_and_lost_reservations() {
    let (mut store, _root) = build_store().await;
    let media = durable_job_media_asset(&store).await;

    let mut stageless = durable_speech_job_acceptance(&media, None);
    stageless.stages.clear();
    assert!(store.accept_durable_job(stageless).await.is_err());
    let mut dangling = durable_speech_job_acceptance(&media, None);
    dangling.stages[0].input_artifact_ids.push(new_uuid());
    assert!(store.accept_durable_job(dangling).await.is_err());
    assert_eq!(durable_job_row_counts(&store).await, [0, 0, 0, 0, 0]);

    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let request = durable_idempotency_request(
        "tenant-a",
        "speech.text_to_speech.create",
        "expiring-key",
        b"clone me",
        1_000,
    );
    let DurableIdempotencyBegin::Acquired(reservation) = store
        .reserve_durable_idempotency(request)
        .await
        .expect("reserve")
    else {
        panic!("reservation must be acquired");
    };
    clock.store(2_000, Ordering::SeqCst);
    assert!(matches!(
        store
            .accept_durable_job(durable_speech_job_acceptance(&media, Some(reservation)))
            .await
            .expect("expired reservation is an outcome"),
        DurableJobAcceptanceOutcome::ReservationLost
    ));
    assert_eq!(durable_job_row_counts(&store).await, [0, 0, 0, 0, 1]);

    store.set_tts_admission_limits_for_test(Some((1, 1)));
    store
        .accept_durable_job(durable_speech_job_acceptance(&media, None))
        .await
        .expect("first speech job fits");
    let error = store
        .accept_durable_job(durable_speech_job_acceptance(&media, None))
        .await
        .expect_err("admission is checked inside acceptance");
    assert!(error.to_string().contains("admission capacity exhausted"));
    assert_eq!(durable_job_row_counts(&store).await, [1, 1, 2, 1, 1]);
}

#[tokio::test]
async fn durable_text_tts_acceptance_has_one_graph_and_replays_after_reopen() {
    let (store, root) = build_store().await;
    let request = durable_idempotency_request(
        "tenant-a",
        "speech.text_to_speech.create.v1",
        "reopen-key",
        b"hello durable world",
        60_000,
    );
    let (first, second) = tokio::join!(
        store.reserve_durable_idempotency(request.clone()),
        store.reserve_durable_idempotency(request.clone()),
    );
    let outcomes = [
        first.expect("first reservation"),
        second.expect("second reservation"),
    ];
    let reservation = outcomes
        .iter()
        .find_map(|outcome| match outcome {
            DurableIdempotencyBegin::Acquired(reservation) => Some(reservation.clone()),
            _ => None,
        })
        .expect("one reservation owner");
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::InProgress { .. }))
            .count(),
        1
    );

    let accepted = match store
        .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
        .await
        .expect("atomic acceptance")
    {
        DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
        DurableTextTtsAcceptanceOutcome::ReservationLost => {
            panic!("reservation unexpectedly lost")
        }
    };
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [1, 1, 1, 1, 1, 1]
    );
    assert_eq!(accepted.job.status, RuntimeJobStatus::Queued);
    assert_eq!(accepted.stage.status, RuntimeStageStatus::Queued);
    assert_eq!(
        accepted.stage.input_artifact_ids,
        vec![accepted.input_artifact.id.clone()]
    );
    assert_eq!(accepted.input_artifact.job_id, accepted.job.id);

    drop(store);
    let reopened = BatchRuntimeStore::initialize_with_database(root.database());
    let replay = match reopened
        .reserve_durable_idempotency(request)
        .await
        .expect("replay after reopen")
    {
        DurableIdempotencyBegin::Replay(replay) => replay,
        outcome => panic!("unexpected reopened outcome: {outcome:?}"),
    };
    assert_eq!(replay.runtime_job_id, accepted.job.id);
    assert_eq!(
        replay.response_json["record_id"].as_str(),
        Some(accepted.record.id.as_str())
    );
    assert_eq!(
        durable_text_tts_row_counts(&reopened).await,
        [1, 1, 1, 1, 1, 1]
    );
}

#[tokio::test]
async fn durable_text_tts_acceptance_fences_expiry_and_bounds_replay_response() {
    let (mut store, _root) = build_store().await;
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let reservation = match store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "speech.text_to_speech.create.v1",
            "expired-key",
            b"expired",
            10,
        ))
        .await
        .expect("expired reservation")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    clock.store(1_010, Ordering::SeqCst);
    assert!(matches!(
        store
            .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
            .await
            .expect("expired acceptance outcome"),
        DurableTextTtsAcceptanceOutcome::ReservationLost
    ));
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [0, 0, 0, 0, 0, 1]
    );

    clock.store(2_000, Ordering::SeqCst);
    // Expired records of other keys wait for the retention sweeper.
    assert_eq!(
        store.prune_expired_durable_idempotency(64).await.unwrap(),
        1
    );
    let reservation = match store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "speech.text_to_speech.create.v1",
            "bounded-key",
            b"bounded",
            60_000,
        ))
        .await
        .expect("bounded reservation")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    let mut acceptance = durable_text_tts_acceptance(Some(reservation));
    acceptance.projection.input_text = "x".repeat(MAX_DURABLE_TTS_TEXT_BYTES);
    let accepted = match store
        .accept_durable_text_tts(acceptance)
        .await
        .expect("maximum bounded input should be accepted")
    {
        DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
        DurableTextTtsAcceptanceOutcome::ReservationLost => {
            panic!("bounded reservation unexpectedly lost")
        }
    };
    let replay_payload = json!({"record_id": accepted.record.id});
    assert!(
        serde_json::to_vec(&replay_payload).unwrap().len()
            <= MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES
    );
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [1, 1, 1, 1, 1, 1]
    );

    // Rejections carry their class: the handler maps them by it, never
    // by their text.
    let status = |error: anyhow::Error| crate::error::store_api_error(error, "storage").status;
    let mut oversized_text = durable_text_tts_acceptance(None);
    oversized_text.projection.input_text = "x".repeat(MAX_DURABLE_TTS_TEXT_BYTES + 1);
    let error = store
        .accept_durable_text_tts(oversized_text)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("input must be between"));
    assert_eq!(status(error), axum::http::StatusCode::PAYLOAD_TOO_LARGE);

    let mut oversized_request = durable_text_tts_acceptance(None);
    oversized_request.request_json =
        json!({"input": "x".repeat(MAX_DURABLE_TTS_REQUEST_JSON_BYTES)});
    let error = store
        .accept_durable_text_tts(oversized_request)
        .await
        .unwrap_err();
    assert_eq!(status(error), axum::http::StatusCode::PAYLOAD_TOO_LARGE);

    let mut oversized_metadata = durable_text_tts_acceptance(None);
    oversized_metadata.model_snapshot_json =
        json!({"metadata": "x".repeat(MAX_DURABLE_TTS_METADATA_JSON_BYTES)});
    let error = store
        .accept_durable_text_tts(oversized_metadata)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("model snapshot exceeds"));
    assert_eq!(status(error), axum::http::StatusCode::BAD_REQUEST);

    let mut long_speaker = durable_text_tts_acceptance(None);
    long_speaker.projection.speaker =
        Some("x".repeat(MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES + 1));
    let error = store
        .accept_durable_text_tts(long_speaker)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("speaker exceeds"));
    assert_eq!(status(error), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [1, 1, 1, 1, 1, 1]
    );
}

#[tokio::test]
async fn unkeyed_text_tts_acceptance_is_atomic_and_only_claimable_after_commit() {
    let (mut store, _root) = build_store().await;
    let mut acceptance = durable_text_tts_acceptance(None);
    acceptance.idempotency_retention_ms = 0;
    store.set_durable_acceptance_failpoint(Some(DurableAcceptanceFailpoint::Stage));
    assert!(store
        .accept_durable_text_tts(acceptance.clone())
        .await
        .is_err());
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [0, 0, 0, 0, 0, 0]
    );
    assert!(store
        .claim_next_stage("worker-before-commit", 60_000)
        .await
        .expect("claim rolled-back graph")
        .is_none());

    store.set_durable_acceptance_failpoint(None);
    let accepted = match store
        .accept_durable_text_tts(acceptance)
        .await
        .expect("unkeyed acceptance")
    {
        DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
        DurableTextTtsAcceptanceOutcome::ReservationLost => {
            panic!("unkeyed acceptance cannot lose a reservation")
        }
    };
    assert_eq!(
        durable_text_tts_row_counts(&store).await,
        [1, 1, 1, 1, 1, 0]
    );
    let claimed = store
        .claim_next_stage("worker-after-commit", 60_000)
        .await
        .expect("claim committed graph")
        .expect("queued stage");
    assert_eq!(claimed.job.id, accepted.job.id);
    assert_eq!(claimed.stage.id, accepted.stage.id);
    assert!(store
        .claim_next_stage("other-worker", 60_000)
        .await
        .expect("second claim")
        .is_none());
}

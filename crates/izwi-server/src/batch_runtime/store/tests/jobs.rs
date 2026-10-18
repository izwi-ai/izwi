use super::*;

#[tokio::test]
async fn creates_runtime_foundation_records() {
    let (store, _root) = build_store().await;

    let media = store
        .create_media_asset(NewMediaAsset {
            asset_kind: "audio_original".to_string(),
            storage_namespace: "uploads".to_string(),
            storage_key: "uploads/transcription/test.wav".to_string(),
            content_type: "audio/wav".to_string(),
            filename: Some("test.wav".to_string()),
            size_bytes: 4,
            sha256: Some(sha256_hex(&[1, 2, 3, 4])),
            duration_secs: Some(1.25),
            sample_rate_hz: Some(16_000),
            channel_count: Some(1),
            peak_amplitude: Some(0.5),
            rms_amplitude: Some(0.1),
            source_asset_id: None,
            canonical_profile_version: None,
            scan_status: "passed".to_string(),
            retention_policy: "default".to_string(),
            metadata_json: json!({"source": "test"}),
        })
        .await
        .expect("media asset");

    let text = store
        .create_text_asset(NewTextAsset {
            raw_text: "Hello world".to_string(),
            normalized_text: None,
            language_hint: Some("en".to_string()),
            sha256: Some(sha256_hex(b"Hello world")),
            safety_status: "allowed".to_string(),
            retention_policy: "default".to_string(),
            structure_json: json!({"kind": "plain"}),
        })
        .await
        .expect("text asset");

    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::AsrTranscription,
            status: RuntimeJobStatus::Queued,
            priority: 5,
            model_id: Some("Granite-Speech-4.1-2B".to_string()),
            capability: Some("asr".to_string()),
            route_record_kind: Some("transcription".to_string()),
            route_record_id: Some("route-1".to_string()),
            input_media_asset_id: Some(media.id.clone()),
            input_text_asset_id: Some(text.id.clone()),
            request_json: json!({"language": "en"}),
            model_snapshot_json: json!({"license": "apache-2.0"}),
            retry_policy_json: json!({"max_attempts": 2}),
            max_attempts: 2,
            idempotency_key: Some("idem-1".to_string()),
            correlation_id: Some("corr-1".to_string()),
        })
        .await
        .expect("job");

    assert_eq!(job.status, RuntimeJobStatus::Queued);
    assert_eq!(job.queued_at, Some(job.created_at));
    assert_eq!(job.input_media_asset_id.as_deref(), Some(media.id.as_str()));

    let stage = store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 10,
            stage_kind: "asr_infer".to_string(),
            status: RuntimeStageStatus::Queued,
            capability: Some("asr".to_string()),
            model_id: job.model_id.clone(),
            max_attempts: 2,
            input_artifact_ids: vec![],
        })
        .await
        .expect("stage");

    let artifact = store
        .create_artifact(NewRuntimeArtifact {
            job_id: job.id.clone(),
            stage_id: Some(stage.id.clone()),
            artifact_kind: RuntimeArtifactKind::Transcript,
            artifact_role: RuntimeArtifactRole::OutputPrimary,
            media_asset_id: None,
            text_asset_id: Some(text.id.clone()),
            storage_key: None,
            content_type: Some("application/json".to_string()),
            filename: Some("transcript.json".to_string()),
            size_bytes: Some(128),
            sha256: None,
            metadata_json: json!({"format": "segments"}),
            retention_policy: "default".to_string(),
        })
        .await
        .expect("artifact");

    let heartbeat = store
        .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
            worker_id: "worker-1".to_string(),
            status: "idle".to_string(),
            queue_names: vec!["batch".to_string()],
            current_job_id: None,
            current_stage_id: None,
            diagnostic_json: json!({"pid": 123}),
        })
        .await
        .expect("heartbeat");

    assert_eq!(artifact.stage_id.as_deref(), Some(stage.id.as_str()));
    assert_eq!(artifact.producer_attempt_count, None);
    assert_eq!(artifact.producer_attempt_token, None);
    assert_eq!(artifact.publication_key, None);
    assert_eq!(heartbeat.queue_names, vec!["batch"]);
    assert_eq!(heartbeat.instance_id, "worker-1");
    assert_eq!(
        heartbeat.registration.queue_classes,
        vec![QueueClass::Batch]
    );
}

#[tokio::test]
async fn active_job_indexes_exclude_terminal_jobs() {
    let (store, _root) = build_store().await;
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: Some("text_to_speech".to_string()),
            route_record_id: Some("speech-1".to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({"max_attempts": 1}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("active job");
    let stage = store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "speech".to_string(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".to_string()),
            model_id: job.model_id.clone(),
            max_attempts: 1,
            input_artifact_ids: Vec::new(),
        })
        .await
        .expect("queued stage");

    assert_eq!(
        store
            .get_active_job_for_route_record(
                RuntimeJobKind::TtsSpeech,
                "text_to_speech",
                "speech-1",
            )
            .await
            .expect("route lookup")
            .expect("active route job")
            .id,
        job.id
    );
    assert_eq!(
        store
            .list_active_jobs_by_kind(RuntimeJobKind::TtsSpeech)
            .await
            .expect("kind lookup")
            .len(),
        1
    );

    let cancelled = store
        .cancel_job(&job.id, Some("test cancellation".to_string()))
        .await
        .expect("cancel")
        .expect("cancelled job");
    assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
    assert_eq!(cancelled.cancellation_state, None);
    assert!(cancelled.finished_at.is_some());
    let cancelled_stage = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(cancelled_stage.status, RuntimeStageStatus::Cancelled);
    assert_eq!(cancelled_stage.cancellation_state, None);
    assert_eq!(cancelled_stage.worker_id, None);
    assert!(store
        .get_active_job_for_route_record(
            RuntimeJobKind::TtsSpeech,
            "text_to_speech",
            "speech-1",
        )
        .await
        .expect("terminal route lookup")
        .is_none());
    assert!(store
        .list_active_jobs_by_kind(RuntimeJobKind::TtsSpeech)
        .await
        .expect("terminal kind lookup")
        .is_empty());
}

#[tokio::test]
async fn acknowledged_job_survives_store_reconstruction_and_completes() {
    let database = TestDatabase::new().await;
    let first_store = BatchRuntimeStore::initialize_with_database(database.database());
    let input = first_store
        .create_text_asset(NewTextAsset {
            raw_text: "durable input".to_string(),
            normalized_text: None,
            language_hint: Some("en".to_string()),
            sha256: Some(sha256_hex(b"durable input")),
            safety_status: "accepted".to_string(),
            retention_policy: "default".to_string(),
            structure_json: json!({}),
        })
        .await
        .expect("durable input");
    let job = first_store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("test-model".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: Some("durability_test".to_string()),
            route_record_id: Some("acknowledged-job".to_string()),
            input_media_asset_id: None,
            input_text_asset_id: Some(input.id.clone()),
            request_json: json!({"text": "durable input"}),
            model_snapshot_json: json!({"model_id": "test-model"}),
            retry_policy_json: json!({"max_attempts": 1}),
            max_attempts: 1,
            idempotency_key: Some("durability-test".to_string()),
            correlation_id: Some("durability-test".to_string()),
        })
        .await
        .expect("acknowledged job");
    let stage = first_store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "tts_generate".to_string(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".to_string()),
            model_id: job.model_id.clone(),
            max_attempts: 1,
            input_artifact_ids: vec![],
        })
        .await
        .expect("acknowledged stage");

    drop(first_store);

    let restarted_store = BatchRuntimeStore::initialize_with_database(database.database());
    let rediscovered_input = restarted_store
        .get_text_asset(&input.id)
        .await
        .expect("rediscover input")
        .expect("durable input exists");
    let rediscovered_job = restarted_store
        .get_job(&job.id)
        .await
        .expect("rediscover job")
        .expect("acknowledged job exists");
    let rediscovered_stage = restarted_store
        .get_stage(&stage.id)
        .await
        .expect("rediscover stage")
        .expect("acknowledged stage exists");
    assert_eq!(rediscovered_input.raw_text, "durable input");
    assert_eq!(
        rediscovered_job.input_text_asset_id.as_deref(),
        Some(input.id.as_str())
    );
    assert_eq!(rediscovered_job.status, RuntimeJobStatus::Queued);
    assert_eq!(rediscovered_stage.status, RuntimeStageStatus::Queued);

    let claimed = restarted_store
        .claim_next_stage("worker-after-restart", 60_000)
        .await
        .expect("claim after restart")
        .expect("rediscovered stage is claimable");
    assert_eq!(claimed.stage.id, stage.id);
    assert_eq!(
        claimed.stage.worker_id.as_deref(),
        Some("worker-after-restart")
    );
    let lease = claimed.lease().expect("restart worker lease");
    let artifact = restarted_store
        .publish_stage_output_artifact(&lease, test_stage_output("restart-result"))
        .await
        .expect("publish after restart")
        .expect("active attempt owns publication");
    restarted_store
        .complete_stage(&lease, vec![artifact.id.clone()])
        .await
        .expect("complete after restart")
        .expect("active attempt completes stage");

    drop(restarted_store);

    let final_store = BatchRuntimeStore::initialize_with_database(database.database());
    let final_job = final_store
        .get_job(&job.id)
        .await
        .expect("load final job")
        .expect("final job exists");
    let final_stage = final_store
        .get_stage(&stage.id)
        .await
        .expect("load final stage")
        .expect("final stage exists");
    let final_artifact = final_store
        .get_artifact(&artifact.id)
        .await
        .expect("load final artifact")
        .expect("final artifact exists");
    assert_eq!(final_job.status, RuntimeJobStatus::Completed);
    assert_eq!(final_stage.status, RuntimeStageStatus::Completed);
    assert_eq!(final_stage.output_artifact_ids, vec![artifact.id.clone()]);
    assert_eq!(
        final_artifact.producer_attempt_token,
        artifact.producer_attempt_token
    );
    assert_eq!(
        final_artifact.publication_key.as_deref(),
        Some("restart-result")
    );
}

#[tokio::test]
async fn manual_retry_requeues_failed_job_and_stage() {
    let (store, _root) = build_store().await;
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-0.6B".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: Some("text_to_speech".to_string()),
            route_record_id: Some("speech-1".to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({"text": "hello"}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({"max_attempts": 1}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("job");
    let stage = store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "tts_synthesize".to_string(),
            status: RuntimeStageStatus::Queued,
            capability: Some("tts".to_string()),
            model_id: job.model_id.clone(),
            max_attempts: 2,
            input_artifact_ids: vec![],
        })
        .await
        .expect("stage");

    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("stage should be claimed");
    assert_eq!(claimed.stage.id, stage.id);
    let lease = claimed.lease().expect("lease");

    let failed_stage = store
        .fail_stage(
            &lease,
            false,
            Some("boom".to_string()),
            Some("first attempt failed".to_string()),
        )
        .await
        .expect("fail")
        .expect("stage should fail");
    assert_eq!(failed_stage.status, RuntimeStageStatus::Failed);

    let failed_job = store
        .get_job(&job.id)
        .await
        .expect("job")
        .expect("job exists");
    assert_eq!(failed_job.status, RuntimeJobStatus::Failed);

    let retried_job = store
        .retry_job(&job.id)
        .await
        .expect("retry")
        .expect("job should retry");
    assert_eq!(retried_job.status, RuntimeJobStatus::Queued);
    assert_eq!(retried_job.attempt_count, 1);
    assert!(retried_job.error_code.is_none());
    assert!(retried_job.finished_at.is_none());

    let retried_stage = store
        .get_stage(&stage.id)
        .await
        .expect("stage")
        .expect("stage exists");
    assert_eq!(retried_stage.status, RuntimeStageStatus::Retrying);
    assert!(retried_stage.error_code.is_none());
    assert!(retried_stage.finished_at.is_none());

    let stages = store
        .list_stages_for_job(&job.id)
        .await
        .expect("list stages");
    assert_eq!(stages.len(), 1);
    let stage_counts = store.stage_status_counts().await.expect("stage counts");
    assert_eq!(
        stage_counts,
        vec![RuntimeStageStatusCount {
            status: RuntimeStageStatus::Retrying,
            count: 1,
        }]
    );

    let second_claim = store
        .claim_next_stage("worker-2", 60_000)
        .await
        .expect("second claim")
        .expect("retried stage should be claimable");
    store
        .fail_stage(
            &second_claim.lease().expect("second lease"),
            false,
            Some("boom-again".to_string()),
            Some("second attempt failed".to_string()),
        )
        .await
        .expect("second failure")
        .expect("stage should fail again");
    assert!(store
        .retry_job(&job.id)
        .await
        .expect("retry budget check")
        .is_none());
}

#[tokio::test]
async fn rows_map_by_column_name_in_any_statement_order() {
    let (store, _root) = build_store().await;
    let (job, stage) = create_test_job_and_stage(&store, 3, "mapping_stage", 2).await;
    let db = store.connection().await.expect("connection");
    let reversed = |by_id: &str, table: &str| {
        let mut columns = by_id
            .trim_start_matches("SELECT ")
            .trim_end_matches(&format!(" FROM {table} WHERE id = ?1"))
            .split(", ")
            .collect::<Vec<_>>();
        columns.reverse();
        let columns = columns.join(", ");
        format!("SELECT {columns} FROM {table} WHERE id = ?1")
    };

    let job_row = db
        .query_one_raw(
            raw::statement(
                db,
                reversed(RUNTIME_JOB_BY_ID_SQL, "runtime_jobs"),
                vec![job.id.clone().into()],
            )
            .expect("job statement"),
        )
        .await
        .expect("job query")
        .expect("job row");
    assert_eq!(map_runtime_job(&job_row).expect("job maps"), job);

    let stage_row = db
        .query_one_raw(
            raw::statement(
                db,
                reversed(JOB_STAGE_BY_ID_SQL, "job_stages"),
                vec![stage.id.clone().into()],
            )
            .expect("stage statement"),
        )
        .await
        .expect("stage query")
        .expect("stage row");
    assert_eq!(map_job_stage(&stage_row).expect("stage maps"), stage);
}

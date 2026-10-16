use super::*;
use crate::db::StoreDatabase;
use tempfile::TempDir;

fn build_store() -> (BatchRuntimeStore, TempDir) {
    let root = tempfile::tempdir().expect("temp dir");
    let db_path = root.path().join("runtime.sqlite");
    (
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path)),
        root,
    )
}

fn test_provider_write_input(
    namespace: MediaNamespace,
    storage_namespace: &str,
    lifetime_ms: u64,
) -> NewProviderWriteReservation {
    let write_id = new_uuid();
    let mut metadata = HookMetadata::new();
    metadata.insert("tenant_id".to_string(), "tenant-a".to_string());
    NewProviderWriteReservation {
        write_id: write_id.clone(),
        tenant_scope: "tenant-a".to_string(),
        storage_namespace: storage_namespace.to_string(),
        content_type: "application/octet-stream".to_string(),
        filename: Some("artifact.bin".to_string()),
        expected_size_bytes: 4,
        expected_sha256: sha256_hex(b"data"),
        lifetime_ms,
        provider_request: MediaWriteRequest {
            namespace,
            record_id: write_id,
            preferred_filename: Some("artifact.bin".to_string()),
            content_type: "application/octet-stream".to_string(),
            metadata,
        },
    }
}

#[test]
fn provider_write_request_envelope_is_typed_and_bounded() {
    let mut input = test_provider_write_input(
        MediaNamespace::Other("artifact-store".to_string()),
        "artifact-store",
        1_000,
    );
    input
        .provider_request
        .metadata
        .insert("workflow_stage".to_string(), "finalize".to_string());
    validate_provider_write_input(&input).expect("valid exact provider request");
    let serialized = serialize_provider_write_request(&input.provider_request).unwrap();
    assert_eq!(
        deserialize_provider_write_request(&serialized).unwrap(),
        input.provider_request
    );
    assert!(deserialize_provider_write_request("{}").is_err());
    assert!(deserialize_provider_write_request(
        &"x".repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
    )
    .unwrap_err()
    .to_string()
    .contains("exceeds"));
    let unsupported = serialized.replacen("\"version\":1", "\"version\":2", 1);
    assert!(deserialize_provider_write_request(&unsupported)
        .unwrap_err()
        .to_string()
        .contains("version is unsupported"));
    let mut nested_unknown: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    nested_unknown["request"]["future_field"] = json!("must-not-be-dropped");
    assert!(
        deserialize_provider_write_request(&nested_unknown.to_string())
            .unwrap_err()
            .to_string()
            .contains("invalid")
    );

    let mut invalid = input.clone();
    invalid.write_id = invalid.write_id.to_ascii_uppercase();
    invalid.provider_request.record_id = invalid.write_id.clone();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("identity"));

    let mut invalid = input.clone();
    invalid.provider_request.record_id = new_uuid();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("record identity"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace = MediaNamespace::Other("other".to_string());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("namespace does not match"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace = MediaNamespace::Other(String::new());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("request namespace"));

    let mut invalid = input.clone();
    invalid.provider_request.content_type = "text/plain".to_string();
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("does not match"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata = (0..=MAX_PROVIDER_WRITE_METADATA_ENTRIES)
        .map(|index| (format!("key-{index}"), "value".to_string()))
        .collect();
    invalid
        .provider_request
        .metadata
        .insert("tenant_id".to_string(), "tenant-a".to_string());
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("entry count"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata.insert(
        "k".repeat(MAX_PROVIDER_WRITE_METADATA_KEY_BYTES + 1),
        "value".to_string(),
    );
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("metadata key"));

    let mut invalid = input.clone();
    invalid.provider_request.metadata.insert(
        "large".to_string(),
        "v".repeat(MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES + 1),
    );
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("metadata value"));

    let mut invalid = input.clone();
    for index in 0..5 {
        invalid
            .provider_request
            .metadata
            .insert(format!("aggregate-{index}"), "v".repeat(900));
    }
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("aggregate byte limit"));

    let mut invalid = input.clone();
    invalid.provider_request.namespace =
        MediaNamespace::Other("n".repeat(MAX_PROVIDER_WRITE_NAMESPACE_BYTES + 1));
    assert!(validate_provider_write_input(&invalid)
        .unwrap_err()
        .to_string()
        .contains("request namespace"));

    let mut invalid = input;
    invalid.provider_request.metadata.clear();
    invalid
        .provider_request
        .metadata
        .insert("tenant_id".to_string(), "tenant-a".to_string());
    for index in 0..4 {
        invalid
            .provider_request
            .metadata
            .insert(format!("escaped-{index}"), "\\\"".repeat(500));
    }
    assert!(serialize_provider_write_request(&invalid.provider_request)
        .unwrap_err()
        .to_string()
        .contains("envelope exceeds"));
}

#[tokio::test]
async fn provider_write_request_roundtrips_and_legacy_recovery_fails_closed() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());

    let mut exact =
        test_provider_write_input(MediaNamespace::GeneratedSpeech, "generated_speech", 10);
    exact
        .provider_request
        .metadata
        .insert("workflow_stage".to_string(), "finalize".to_string());
    let exact = store.reserve_provider_write(exact).await.unwrap();
    assert_eq!(
        exact.provider_request.namespace,
        MediaNamespace::GeneratedSpeech
    );
    assert_eq!(
        exact
            .provider_request
            .metadata
            .get("workflow_stage")
            .map(String::as_str),
        Some("finalize")
    );

    let legacy = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let db = store.connection().await.unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = NULL WHERE write_id IN (?1, ?2)",
            vec![exact.write_id.clone().into(), legacy.write_id.clone().into()],
        )
        .unwrap(),
    )
    .await
    .unwrap();

    clock.store(1_010, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].write_id, legacy.write_id);
    assert_eq!(
        claimed[0].provider_request,
        legacy_provider_write_request(
            &legacy.write_id,
            &legacy.tenant_scope,
            &legacy.storage_namespace,
            &legacy.content_type,
            legacy.filename.as_deref(),
        )
        .unwrap()
    );
    let retained = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND provider_request_json IS NULL",
                vec![exact.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(retained, 1);
}

#[tokio::test]
async fn invalid_provider_write_rows_remain_fenced_without_starving_cleanup() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let poison = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let oversized = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::GeneratedSpeech,
            "generated_speech",
            10,
        ))
        .await
        .unwrap();
    let valid = store
        .reserve_provider_write(test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            10,
        ))
        .await
        .unwrap();
    let db = store.connection().await.unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = 0 WHERE write_id = ?2",
            vec![
                "{\"version\":1,\"request\":{\"future_field\":true}}"
                    .into(),
                poison.write_id.clone().into(),
            ],
        )
        .unwrap(),
    )
    .await
    .unwrap();
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = -1 WHERE write_id = ?2",
            vec![
                "x"
                    .repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
                    .into(),
                oversized.write_id.clone().into(),
            ],
        )
        .unwrap(),
    )
    .await
    .unwrap();

    clock.store(1_010, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].write_id, valid.write_id);
    assert!(get_provider_write_with(db, &oversized.write_id)
        .await
        .unwrap()
        .is_none());

    let quarantined = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT state, available_at, cleanup_attempt_count, last_error FROM provider_write_reservations WHERE write_id = ?1",
                vec![poison.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        quarantined.try_get_by_index::<String>(0).unwrap(),
        "cleanup_pending"
    );
    assert!(quarantined.try_get_by_index::<i64>(1).unwrap() > 1_010);
    assert_eq!(quarantined.try_get_by_index::<i64>(2).unwrap(), 1);
    assert!(quarantined
        .try_get_by_index::<Option<String>>(3)
        .unwrap()
        .is_some());

    let oversized_retained = db
        .query_one_raw(
            raw::statement(
                db,
                "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND state = 'reserved'",
                vec![oversized.write_id.into()],
            )
            .unwrap(),
        )
        .await
        .unwrap()
        .unwrap()
        .try_get_by_index::<i64>(0)
        .unwrap();
    assert_eq!(oversized_retained, 1);
}

fn durable_idempotency_request(
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
    payload: &[u8],
    reservation_ttl_ms: u64,
) -> DurableIdempotencyRequest {
    DurableIdempotencyRequest {
        tenant_scope: tenant_scope.to_string(),
        operation: operation.to_string(),
        idempotency_key: idempotency_key.to_string(),
        digest_version: DURABLE_IDEMPOTENCY_DIGEST_VERSION,
        request_digest: sha256_hex(payload),
        reservation_ttl_ms,
    }
}

fn durable_text_tts_acceptance(
    reservation: Option<DurableIdempotencyReservation>,
) -> NewDurableTextTtsAcceptance {
    NewDurableTextTtsAcceptance {
        projection: NewSpeechHistoryRecord {
            stop_reason: None,
            route_kind: SpeechRouteKind::TextToSpeech,
            processing_status: SpeechHistoryProcessingStatus::Pending,
            processing_error: None,
            model_id: Some("Kokoro-82M".to_string()),
            speaker: Some("af_heart".to_string()),
            language: Some("en".to_string()),
            saved_voice_id: None,
            speed: Some(1.0),
            input_text: "hello durable world".to_string(),
            voice_description: None,
            reference_text: None,
            generation_time_ms: 0.0,
            audio_duration_secs: None,
            rtf: None,
            tokens_generated: None,
            audio_mime_type: "audio/wav".to_string(),
            audio_filename: Some("speech.wav".to_string()),
            audio_bytes: Vec::new(),
        },
        request_json: json!({
            "tenant_key": null,
            "route_kind": "text_to_speech",
            "model_id": "Kokoro-82M",
            "input_text": "hello durable world",
            "request": {"speaker": "af_heart", "speed": 1.0}
        }),
        model_snapshot_json: json!({"version": 1, "model_id": "Kokoro-82M"}),
        retry_policy_json: json!({"max_attempts": 2}),
        priority: 0,
        max_attempts: 2,
        correlation_id: Some("durable-test-request".to_string()),
        stage_kind: "tts_synthesize".to_string(),
        queue_class: QueueClass::BatchTts,
        resource_hints: StageResourceHints::default(),
        reservation,
        idempotency_retention_ms: 60_000,
    }
}

async fn durable_text_tts_row_counts(store: &BatchRuntimeStore) -> [u64; 6] {
    let db = store.connection().await.expect("database");
    let mut counts = [0_u64; 6];
    for (index, table) in [
        "speech_history_records",
        "text_assets",
        "runtime_jobs",
        "runtime_artifacts",
        "job_stages",
        "durable_idempotency_keys_v2",
    ]
    .into_iter()
    .enumerate()
    {
        let row = db
            .query_one_raw(
                raw::statement(db, format!("SELECT COUNT(*) FROM {table}"), vec![])
                    .expect("count statement"),
            )
            .await
            .expect("count query")
            .expect("count row");
        counts[index] = u64::try_from(
            row.try_get_by_index::<i64>(0)
                .expect("nonnegative row count"),
        )
        .expect("nonnegative row count");
    }
    counts
}

#[test]
fn durable_idempotency_digest_uses_canonical_json_object_order() {
    let first: serde_json::Value = serde_json::from_str(
        r#"{"text":"hello","options":{"voice":"a","speed":1},"parts":[1,2]}"#,
    )
    .expect("first request");
    let reordered: serde_json::Value = serde_json::from_str(
        r#"{"parts":[1,2],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
    )
    .expect("reordered request");
    let changed: serde_json::Value = serde_json::from_str(
        r#"{"parts":[2,1],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
    )
    .expect("changed request");

    assert_eq!(
        canonical_request_digest(&first).expect("first digest"),
        canonical_request_digest(&reordered).expect("reordered digest")
    );
    assert_ne!(
        canonical_request_digest(&first).expect("first digest"),
        canonical_request_digest(&changed).expect("changed digest")
    );
    assert!(canonical_request_digest(&json!(
        "x".repeat(MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES + 1)
    ))
    .is_err());

    let mut too_deep = serde_json::Value::Null;
    for _ in 0..=MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH {
        too_deep = serde_json::Value::Array(vec![too_deep]);
    }
    assert!(canonical_request_digest(&too_deep)
        .unwrap_err()
        .to_string()
        .contains("maximum nesting depth"));
}

async fn create_test_job_and_stage(
    store: &BatchRuntimeStore,
    priority: i32,
    stage_kind: &str,
    max_attempts: u32,
) -> (RuntimeJob, JobStage) {
    let job = create_test_job(store, priority, max_attempts).await;
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
        .await
        .expect("test stage");
    (job, stage)
}

async fn create_test_job(
    store: &BatchRuntimeStore,
    priority: i32,
    max_attempts: u32,
) -> RuntimeJob {
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority,
            model_id: None,
            capability: Some("test".to_string()),
            route_record_kind: Some("test".to_string()),
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({"max_attempts": max_attempts}),
            max_attempts,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("test job");
    job
}

#[tokio::test]
async fn cooperative_yield_frees_worker_for_short_job_without_consuming_retries() {
    let (mut store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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
async fn speech_replay_gc_fences_retry_before_storage_deletion() {
    for retry_first in [false, true] {
        let (mut store, _root) = build_store();
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
        let (store, _root) = build_store();
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

fn admission_job(tenant: u8) -> NewRuntimeJob {
    NewRuntimeJob {
        job_kind: RuntimeJobKind::TtsSpeech,
        status: RuntimeJobStatus::Queued,
        priority: 0,
        model_id: Some("FishAudio-S2-Pro".into()),
        capability: Some("tts".into()),
        route_record_kind: Some("text_to_speech".into()),
        route_record_id: None,
        input_media_asset_id: None,
        input_text_asset_id: None,
        request_json: json!({"tenant_key": vec![tenant; 32]}),
        model_snapshot_json: json!({}),
        retry_policy_json: json!({"max_attempts": 2}),
        max_attempts: 2,
        idempotency_key: None,
        correlation_id: None,
    }
}

#[tokio::test]
async fn speech_admission_preflight_has_no_reservation_and_rejects_overload() {
    let (mut store, _root) = build_store();
    store.test_tts_admission_limits = Some((1, 1));
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
    let (mut store, _root) = build_store();
    store.test_tts_admission_limits = Some((2, 2));
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
    let (mut store, _root) = build_store();
    store.test_tts_admission_limits = Some((3, 1));
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

fn test_stage_output(publication_key: &str) -> NewStageOutputArtifact {
    NewStageOutputArtifact {
        publication_key: publication_key.to_string(),
        artifact_kind: RuntimeArtifactKind::Metadata,
        artifact_role: RuntimeArtifactRole::OutputPrimary,
        media_asset_id: None,
        text_asset_id: None,
        storage_key: Some(format!("outputs/{publication_key}.json")),
        content_type: Some("application/json".to_string()),
        filename: Some(format!("{publication_key}.json")),
        size_bytes: Some(2),
        sha256: Some(sha256_hex(b"{}")),
        metadata_json: json!({"publication_key": publication_key}),
        retention_policy: "default".to_string(),
    }
}

#[tokio::test]
async fn creates_runtime_foundation_records() {
    let (store, _root) = build_store();

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
async fn durable_idempotency_is_tenant_and_operation_scoped_with_replay() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let job = create_test_job(&store, 0, 1).await;
    let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 500);
    let reservation = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    let committed = store
        .commit_durable_idempotency(&reservation, &job.id, json!({"job_id": job.id}), 2_000)
        .await
        .expect("commit")
        .expect("owned reservation");
    assert_eq!(committed.runtime_job_id, job.id);
    assert_eq!(
        store
            .commit_durable_idempotency(
                &reservation,
                &job.id,
                json!({"job_id": job.id}),
                2_000,
            )
            .await
            .expect("idempotent commit"),
        Some(committed)
    );

    assert!(matches!(
        store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("replay"),
        DurableIdempotencyBegin::Replay(replay)
            if replay.runtime_job_id == job.id
                && replay.response_json == json!({"job_id": job.id})
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                request_digest: sha256_hex(b"different"),
                ..request.clone()
            })
            .await
            .expect("conflict"),
        DurableIdempotencyBegin::Conflict
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                tenant_scope: "tenant-b".to_string(),
                ..request.clone()
            })
            .await
            .expect("other tenant"),
        DurableIdempotencyBegin::Acquired(_)
    ));
    assert!(matches!(
        store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                operation: "job.export".to_string(),
                ..request
            })
            .await
            .expect("other operation"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn concurrent_same_key_reservations_never_double_acquire() {
    // Fleet regression: two gateways (or two retries) may race to reserve
    // the same idempotency key. The conflict-tolerant insert must let
    // exactly one caller acquire; the loser observes the winner's exact
    // state instead of creating duplicate work. This holds on SQLite via
    // the named admission lock and on PostgreSQL via the
    // ON CONFLICT/INSERT IGNORE key fence.
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let first_request =
        durable_idempotency_request("tenant-a", "job.create", "race-key", b"one", 500);
    let second_request = first_request.clone();
    let (first, second) = tokio::join!(
        store.reserve_durable_idempotency(first_request),
        store.reserve_durable_idempotency(second_request)
    );
    let first = first.expect("first concurrent reserve");
    let second = second.expect("second concurrent reserve");
    let acquired = usize::from(matches!(first, DurableIdempotencyBegin::Acquired(_)))
        + usize::from(matches!(second, DurableIdempotencyBegin::Acquired(_)));
    assert_eq!(
        acquired, 1,
        "exactly one concurrent caller may acquire, got {first:?} and {second:?}"
    );
    for outcome in [&first, &second] {
        match outcome {
            DurableIdempotencyBegin::Acquired(_)
            | DurableIdempotencyBegin::InProgress { .. } => {}
            other => panic!("loser must observe the winner, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn durable_idempotency_reservations_expire_and_fence_stale_commits() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let job = create_test_job(&store, 0, 1).await;
    let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100);
    let first = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("first reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert!(matches!(
        store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("in progress"),
        DurableIdempotencyBegin::InProgress { expires_at: 1_100 }
    ));

    clock.store(1_100, Ordering::SeqCst);
    let second = match store
        .reserve_durable_idempotency(request.clone())
        .await
        .expect("reserve after expiry")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert_ne!(second.reservation_token, first.reservation_token);
    assert!(store
        .commit_durable_idempotency(&first, &job.id, json!({"job_id": job.id}), 100)
        .await
        .expect("stale commit")
        .is_none());
    store
        .commit_durable_idempotency(&second, &job.id, json!({"job_id": job.id}), 100)
        .await
        .expect("current commit")
        .expect("current owner");

    clock.store(1_200, Ordering::SeqCst);
    assert!(matches!(
        store
            .reserve_durable_idempotency(request)
            .await
            .expect("reserve after result expiry"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn durable_idempotency_concurrent_reservation_has_one_owner_and_fenced_release() {
    let (mut store, _root) = build_store();
    store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
    let request =
        durable_idempotency_request("tenant-a", "job.create", "concurrent-key", b"one", 500);
    let (first, second) = tokio::join!(
        store.reserve_durable_idempotency(request.clone()),
        store.reserve_durable_idempotency(request.clone()),
    );
    let outcomes = [
        first.expect("first reserve"),
        second.expect("second reserve"),
    ];
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
    let reservation = outcomes
        .into_iter()
        .find_map(|outcome| match outcome {
            DurableIdempotencyBegin::Acquired(reservation) => Some(reservation),
            _ => None,
        })
        .expect("reservation owner");
    let mut forged = reservation.clone();
    forged.reservation_token = new_uuid();
    assert!(!store
        .release_durable_idempotency(&forged)
        .await
        .expect("forged release"));
    assert!(store
        .release_durable_idempotency(&reservation)
        .await
        .expect("owner release"));
    assert!(matches!(
        store
            .reserve_durable_idempotency(request)
            .await
            .expect("reserve after release"),
        DurableIdempotencyBegin::Acquired(_)
    ));
}

#[tokio::test]
async fn durable_idempotency_pruning_and_capacity_are_bounded_and_deterministic() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    for (offset, key) in [(0, "a"), (1, "b"), (2, "c")] {
        clock.store(1_000 + offset, Ordering::SeqCst);
        assert!(matches!(
            store
                .reserve_durable_idempotency(durable_idempotency_request(
                    "tenant-a",
                    "job.create",
                    key,
                    key.as_bytes(),
                    100,
                ))
                .await
                .expect("reserve expiring key"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    clock.store(1_200, Ordering::SeqCst);
    assert_eq!(
        store
            .prune_expired_durable_idempotency(2)
            .await
            .expect("bounded prune"),
        2
    );
    let db = store.connection().await.expect("database");
    let rows = db
        .query_all_raw(
            raw::statement(
                db,
                "SELECT idempotency_key FROM durable_idempotency_keys_v2 ORDER BY idempotency_key",
                vec![],
            )
            .expect("remaining-key query"),
        )
        .await
        .expect("remaining keys");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].try_get_by_index::<String>(0).unwrap(), "c");

    store
        .prune_expired_durable_idempotency(1)
        .await
        .expect("remove final expired key");
    clock.store(2_000, Ordering::SeqCst);
    for key in ["one", "two"] {
        assert!(matches!(
            store
                .reserve_durable_idempotency_with_capacity(
                    durable_idempotency_request(
                        "tenant-a",
                        "job.create",
                        key,
                        key.as_bytes(),
                        100,
                    ),
                    2,
                )
                .await
                .expect("within capacity"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    assert!(matches!(
        store
            .reserve_durable_idempotency_with_capacity(
                durable_idempotency_request("tenant-a", "job.create", "three", b"three", 100,),
                2,
            )
            .await
            .expect("capacity outcome"),
        DurableIdempotencyBegin::CapacityExceeded
    ));
}

async fn count_where(store: &BatchRuntimeStore, sql: &str, values: Vec<Value>) -> u64 {
    let db = store.connection().await.expect("database");
    let row = db
        .query_one_raw(raw::statement(db, sql, values).expect("count statement"))
        .await
        .expect("count query")
        .expect("count row");
    u64::try_from(row.try_get_by_index::<i64>(0).expect("count")).expect("count")
}

/// The quota rows must always equal the records they count.
async fn assert_quota_matches_records(store: &BatchRuntimeStore, kind: QuotaKind, table: &str) {
    let db = store.connection().await.expect("database");
    let rows = db
        .query_all_raw(
            raw::statement(
                db,
                "SELECT scope, used FROM runtime_quota_usage WHERE quota_kind = ?1",
                vec![kind.as_db_value().into()],
            )
            .expect("quota statement"),
        )
        .await
        .expect("quota rows");
    assert!(!rows.is_empty(), "{table} has quota rows");
    for row in rows {
        let scope: String = row.try_get_by_index(0).unwrap();
        let used = u64::try_from(row.try_get_by_index::<i64>(1).unwrap()).unwrap();
        let actual = if scope.is_empty() {
            count_where(store, &format!("SELECT COUNT(*) FROM {table}"), vec![]).await
        } else {
            count_where(
                store,
                &format!("SELECT COUNT(*) FROM {table} WHERE tenant_scope = ?1"),
                vec![scope.clone().into()],
            )
            .await
        };
        assert_eq!(used, actual, "{table} quota for scope {scope:?}");
    }
}

async fn reserve_with_limits(
    store: &BatchRuntimeStore,
    tenant: &str,
    key: &str,
    limits: QuotaLimits,
) -> DurableIdempotencyBegin {
    store
        .reserve_durable_idempotency_with_limits(
            durable_idempotency_request(tenant, "job.create", key, key.as_bytes(), 100),
            limits,
        )
        .await
        .expect("reservation outcome")
}

/// DUR-F4: idempotency capacity is per tenant, with a global ceiling,
/// kept as counters by the transactions that insert and delete records.
#[tokio::test]
async fn durable_idempotency_quota_is_per_tenant_and_counted_incrementally() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    let limits = QuotaLimits {
        global: 3,
        per_tenant: 2,
    };
    let acquired = |outcome: &DurableIdempotencyBegin| {
        matches!(outcome, DurableIdempotencyBegin::Acquired(_))
    };

    assert!(acquired(
        &reserve_with_limits(&store, "tenant-a", "a1", limits).await
    ));
    let DurableIdempotencyBegin::Acquired(a2) =
        reserve_with_limits(&store, "tenant-a", "a2", limits).await
    else {
        panic!("tenant-a has room for two records");
    };
    // A full tenant is refused without touching anyone else's capacity.
    assert_eq!(
        reserve_with_limits(&store, "tenant-a", "a3", limits).await,
        DurableIdempotencyBegin::CapacityExceeded
    );
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b1", limits).await
    ));
    // Duplicates neither count nor need capacity.
    assert!(matches!(
        reserve_with_limits(&store, "tenant-a", "a1", limits).await,
        DurableIdempotencyBegin::InProgress { .. }
    ));
    // The global ceiling holds across tenants.
    assert_eq!(
        reserve_with_limits(&store, "tenant-c", "c1", limits).await,
        DurableIdempotencyBegin::CapacityExceeded
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;

    assert!(store.release_durable_idempotency(&a2).await.unwrap());
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-c", "c1", limits).await
    ));

    // A tenant at its cap first gives back its own expired records.
    clock.store(2_000, Ordering::SeqCst);
    let roomy = QuotaLimits {
        global: 16,
        per_tenant: 2,
    };
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b2", roomy).await
    ));
    assert!(acquired(
        &reserve_with_limits(&store, "tenant-b", "b3", roomy).await
    ));
    assert_eq!(
        count_where(
            &store,
            "SELECT COUNT(*) FROM durable_idempotency_keys_v2 WHERE tenant_scope = 'tenant-a'",
            vec![]
        )
        .await,
        1,
        "another tenant's expired records wait for the sweeper"
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;

    assert_eq!(
        store.prune_expired_durable_idempotency(64).await.unwrap(),
        2
    );
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;
}

/// DUR-F4: provider write reservations are capped per tenant and given
/// back by every path that deletes them.
#[tokio::test]
async fn provider_write_quota_is_per_tenant_and_released_by_cleanup() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.set_provider_write_limits_for_test(QuotaLimits {
        global: 2,
        per_tenant: 1,
    });
    let for_tenant = |tenant: &str| {
        let mut input = test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            100,
        );
        input.tenant_scope = tenant.to_string();
        input
            .provider_request
            .metadata
            .insert("tenant_id".to_string(), tenant.to_string());
        input
    };

    let first = store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect("first reservation");
    let error = store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect_err("tenant-a is at its cap");
    assert!(error.to_string().contains("capacity exhausted"));
    store
        .reserve_provider_write(for_tenant("tenant-b"))
        .await
        .expect("tenant-b has its own capacity");
    assert!(store
        .reserve_provider_write(for_tenant("tenant-c"))
        .await
        .is_err());
    assert_quota_matches_records(
        &store,
        QuotaKind::ProviderWrite,
        "provider_write_reservations",
    )
    .await;

    assert!(store
        .abandon_provider_write(&first, "test abandon")
        .await
        .unwrap());
    clock.store(2_000, Ordering::SeqCst);
    let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
    assert_eq!(claimed.len(), 2);
    for reservation in &claimed {
        assert!(store
            .complete_provider_write_cleanup(reservation)
            .await
            .unwrap());
    }
    assert_quota_matches_records(
        &store,
        QuotaKind::ProviderWrite,
        "provider_write_reservations",
    )
    .await;
    store
        .reserve_provider_write(for_tenant("tenant-a"))
        .await
        .expect("capacity returns after cleanup");
}

/// An upgrade creates the quota table and seeds it from the records that
/// already exist; later migrations leave the live counts alone.
#[tokio::test]
async fn quota_usage_is_backfilled_once_when_its_table_is_created() {
    let (store, _root) = build_store();
    for (tenant, key) in [("tenant-a", "a1"), ("tenant-a", "a2"), ("tenant-b", "b1")] {
        assert!(matches!(
            store
                .reserve_durable_idempotency(durable_idempotency_request(
                    tenant,
                    "job.create",
                    key,
                    key.as_bytes(),
                    60_000,
                ))
                .await
                .unwrap(),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }
    let db = store.connection().await.unwrap();
    db.execute_unprepared("DROP TABLE runtime_quota_usage")
        .await
        .unwrap();
    crate::db::migrator::Migrator::up(db).await.unwrap();
    assert_quota_matches_records(
        &store,
        QuotaKind::DurableIdempotency,
        "durable_idempotency_keys_v2",
    )
    .await;
    assert_eq!(
        count_where(
            &store,
            "SELECT used FROM runtime_quota_usage WHERE quota_kind = 'durable_idempotency' AND scope = ''",
            vec![]
        )
        .await,
        3
    );

    db.execute_unprepared(
        "UPDATE runtime_quota_usage SET used = 7 WHERE quota_kind = 'durable_idempotency' AND scope = ''",
    )
    .await
    .unwrap();
    crate::db::migrator::Migrator::up(db).await.unwrap();
    assert_eq!(
        count_where(
            &store,
            "SELECT used FROM runtime_quota_usage WHERE quota_kind = 'durable_idempotency' AND scope = ''",
            vec![]
        )
        .await,
        7,
        "an existing table is never re-seeded"
    );
}

fn retention_policy() -> RetentionPolicy {
    RetentionPolicy {
        terminal_job_ms: 10_000,
        ephemeral_artifact_ms: 10_000,
        worker_heartbeat_ms: 10_000,
        media_tombstone_ms: 10_000,
        legacy_idempotency_ms: 10_000,
    }
}

async fn retention_job(
    store: &BatchRuntimeStore,
    status: RuntimeJobStatus,
    text: Option<&str>,
) -> RuntimeJob {
    let text_asset = match text {
        Some(text) => Some(
            store
                .create_text_asset(NewTextAsset {
                    raw_text: text.to_string(),
                    normalized_text: None,
                    language_hint: None,
                    sha256: None,
                    safety_status: "unchecked".to_string(),
                    retention_policy: "default".to_string(),
                    structure_json: json!({}),
                })
                .await
                .unwrap(),
        ),
        None => None,
    };
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::AsrTranscription,
            status,
            priority: 0,
            model_id: None,
            capability: None,
            route_record_kind: None,
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: text_asset.as_ref().map(|asset| asset.id.clone()),
            request_json: json!({}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .unwrap();
    store
        .create_stage(NewJobStage {
            job_id: job.id.clone(),
            sequence: 0,
            stage_kind: "asr_transcribe".to_string(),
            status: if status == RuntimeJobStatus::Queued {
                RuntimeStageStatus::Queued
            } else {
                RuntimeStageStatus::Completed
            },
            capability: None,
            model_id: None,
            max_attempts: 1,
            input_artifact_ids: Vec::new(),
        })
        .await
        .unwrap();
    if let Some(text_asset) = text_asset {
        store
            .create_artifact(NewRuntimeArtifact {
                job_id: job.id.clone(),
                stage_id: None,
                artifact_kind: RuntimeArtifactKind::Text,
                artifact_role: RuntimeArtifactRole::InputOriginal,
                media_asset_id: None,
                text_asset_id: Some(text_asset.id),
                storage_key: None,
                content_type: None,
                filename: None,
                size_bytes: None,
                sha256: None,
                metadata_json: json!({}),
                retention_policy: "default".to_string(),
            })
            .await
            .unwrap();
    }
    job
}

async fn opaque_media(store: &BatchRuntimeStore, retention_policy: &str) -> MediaAsset {
    store
        .create_media_asset(NewMediaAsset {
            asset_kind: "opaque_artifact".to_string(),
            storage_namespace: "artifact_store_v1".to_string(),
            storage_key: format!("artifact-store/{}", new_uuid()),
            content_type: "audio/pcm-f32le".to_string(),
            filename: None,
            size_bytes: 4,
            sha256: Some(sha256_hex(b"data")),
            duration_secs: None,
            sample_rate_hz: None,
            channel_count: None,
            peak_amplitude: None,
            rms_amplitude: None,
            source_asset_id: None,
            canonical_profile_version: None,
            scan_status: "not_required".to_string(),
            retention_policy: retention_policy.to_string(),
            metadata_json: json!({"artifact_store": {"version": 1, "tenant_id": "tenant-a"}}),
        })
        .await
        .unwrap()
}

/// DUR-F6: the sweeper deletes only state past its retention, in bounded
/// pages, and never a job that still owns live job-scoped media.
#[tokio::test]
async fn retention_sweep_deletes_only_expired_terminal_state() {
    let (mut store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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

#[tokio::test]
async fn durable_idempotency_rejects_unbounded_or_unattached_records() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock);
    let oversized_key = "x".repeat(MAX_DURABLE_IDEMPOTENCY_KEY_BYTES + 1);
    assert!(store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "job.create",
            &oversized_key,
            b"one",
            100,
        ))
        .await
        .unwrap_err()
        .to_string()
        .contains("idempotency key exceeds"));
    assert!(store
        .reserve_durable_idempotency(DurableIdempotencyRequest {
            request_digest: "A".repeat(64),
            ..durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100,)
        })
        .await
        .unwrap_err()
        .to_string()
        .contains("lowercase SHA-256"));

    let reservation = match store
        .reserve_durable_idempotency(durable_idempotency_request(
            "tenant-a",
            "job.create",
            "key-2",
            b"two",
            100,
        ))
        .await
        .expect("reserve")
    {
        DurableIdempotencyBegin::Acquired(reservation) => reservation,
        outcome => panic!("unexpected reservation outcome: {outcome:?}"),
    };
    assert!(store
        .commit_durable_idempotency(
            &reservation,
            "missing-job",
            json!({"job_id": "missing-job"}),
            100,
        )
        .await
        .expect("unattached commit")
        .is_none());
    let job = create_test_job(&store, 0, 1).await;
    assert!(store
        .commit_durable_idempotency(
            &reservation,
            &job.id,
            json!({"result": "x".repeat(MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES)}),
            100,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("result exceeds"));
}

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
        let (mut store, _root) = build_store();
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

async fn durable_job_media_asset(store: &BatchRuntimeStore) -> String {
    store
        .create_media_asset(NewMediaAsset {
            asset_kind: "audio_original".to_string(),
            storage_namespace: "tts_reference".to_string(),
            storage_key: format!("uploads/reference/{}.wav", new_uuid()),
            content_type: "audio/wav".to_string(),
            filename: Some("reference.wav".to_string()),
            size_bytes: 4,
            sha256: Some(sha256_hex(&[1, 2, 3, 4])),
            duration_secs: Some(1.0),
            sample_rate_hz: Some(24_000),
            channel_count: Some(1),
            peak_amplitude: None,
            rms_amplitude: None,
            source_asset_id: None,
            canonical_profile_version: None,
            scan_status: "passed".to_string(),
            retention_policy: "default".to_string(),
            metadata_json: json!({}),
        })
        .await
        .expect("media asset")
        .id
}

/// A reference-voice TTS graph: text input, reference media, one stage
/// consuming both.
fn durable_speech_job_acceptance(
    media_asset_id: &str,
    reservation: Option<DurableIdempotencyReservation>,
) -> NewDurableJobAcceptance {
    let text_id = new_uuid();
    let text_artifact = new_uuid();
    let media_artifact = new_uuid();
    NewDurableJobAcceptance {
        job: NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-12Hz-1.7B-Base".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: Some("text_to_speech".to_string()),
            route_record_id: Some("record-1".to_string()),
            input_media_asset_id: Some(media_asset_id.to_string()),
            input_text_asset_id: Some(text_id.clone()),
            request_json: json!({"tenant_key": null, "input_text": "clone me"}),
            model_snapshot_json: json!({"version": 1}),
            retry_policy_json: json!({"max_attempts": 2}),
            max_attempts: 2,
            idempotency_key: None,
            correlation_id: Some("durable-job-test".to_string()),
        },
        text_input: Some(NewDurableTextInput {
            id: text_id.clone(),
            asset: NewTextAsset {
                raw_text: "clone me".to_string(),
                normalized_text: None,
                language_hint: None,
                sha256: Some(sha256_hex(b"clone me")),
                safety_status: "unchecked".to_string(),
                retention_policy: "default".to_string(),
                structure_json: json!({"source": "tts_input"}),
            },
        }),
        artifacts: vec![
            NewDurableJobArtifact {
                id: text_artifact.clone(),
                artifact_kind: RuntimeArtifactKind::Text,
                artifact_role: RuntimeArtifactRole::InputOriginal,
                media_asset_id: None,
                text_asset_id: Some(text_id),
                storage_key: None,
                content_type: Some("text/plain".to_string()),
                filename: None,
                size_bytes: Some(8),
                sha256: None,
                metadata_json: json!({"input_kind": "text"}),
                retention_policy: "default".to_string(),
            },
            NewDurableJobArtifact {
                id: media_artifact.clone(),
                artifact_kind: RuntimeArtifactKind::Media,
                artifact_role: RuntimeArtifactRole::InputCanonical,
                media_asset_id: Some(media_asset_id.to_string()),
                text_asset_id: None,
                storage_key: None,
                content_type: Some("audio/wav".to_string()),
                filename: None,
                size_bytes: Some(4),
                sha256: None,
                metadata_json: json!({"source_artifact_id": text_artifact}),
                retention_policy: "default".to_string(),
            },
        ],
        stages: vec![NewDurableJobStage {
            sequence: 0,
            stage_kind: "tts_synthesize".to_string(),
            capability: Some("tts".to_string()),
            model_id: Some("Qwen3-TTS-12Hz-1.7B-Base".to_string()),
            max_attempts: 2,
            input_artifact_ids: vec![text_artifact, media_artifact],
            queue_class: QueueClass::BatchTts,
            resource_hints: StageResourceHints::default(),
        }],
        reservation,
        idempotency_response: json!({"record_id": "record-1"}),
        idempotency_retention_ms: 60_000,
    }
}

async fn durable_job_row_counts(store: &BatchRuntimeStore) -> [u64; 5] {
    let counts = durable_text_tts_row_counts(store).await;
    [counts[1], counts[2], counts[3], counts[4], counts[5]]
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
        let (mut store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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

    store.test_tts_admission_limits = Some((1, 1));
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
    let (store, root) = build_store();
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
    let reopened = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
        root.path().join("runtime.sqlite"),
    ));
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
    let (mut store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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

#[tokio::test]
async fn active_job_indexes_exclude_terminal_jobs() {
    let (store, _root) = build_store();
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
async fn registered_worker_heartbeat_tracks_instance_resources_and_capacity() {
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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

#[tokio::test]
async fn job_transitions_are_status_conditional() {
    let (store, _root) = build_store();
    let job = store
        .create_job(NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("Qwen3-TTS-0.6B".to_string()),
            capability: Some("tts".to_string()),
            route_record_kind: Some("speech_history".to_string()),
            route_record_id: Some("speech-1".to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({"text": "hello"}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: None,
        })
        .await
        .expect("job");

    let cancelled = store
        .transition_job_status(
            &job.id,
            &[RuntimeJobStatus::Queued],
            RuntimeJobStatus::Cancelled,
            None,
            None,
            Some("user requested".to_string()),
        )
        .await
        .expect("cancel transition")
        .expect("job should transition");

    assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
    assert_eq!(
        cancelled.cancellation_reason.as_deref(),
        Some("user requested")
    );

    let late_completion = store
        .transition_job_status(
            &job.id,
            &[RuntimeJobStatus::Running],
            RuntimeJobStatus::Completed,
            None,
            None,
            None,
        )
        .await
        .expect("late transition should not error");

    assert!(late_completion.is_none());
    let fetched = store
        .get_job(&job.id)
        .await
        .expect("fetch")
        .expect("job still exists");
    assert_eq!(fetched.status, RuntimeJobStatus::Cancelled);
}

#[tokio::test]
async fn queue_claim_recovery_and_cancel_are_durable() {
    let (store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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
async fn acknowledged_job_survives_store_reconstruction_and_completes() {
    let root = tempfile::tempdir().expect("temp dir");
    let db_path = root.path().join("runtime.sqlite");
    let first_store =
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
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

    let restarted_store =
        BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
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

    let final_store = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
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
async fn filtered_stage_claim_skips_incompatible_higher_priority_stage() {
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    store
        .transition_job_status(
            &job.id,
            &[RuntimeJobStatus::Queued],
            RuntimeJobStatus::Cancelled,
            None,
            None,
            Some("cancel before claim CAS".to_string()),
        )
        .await
        .expect("cancel job")
        .expect("job transition");

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
    let (store, _root) = build_store();
    let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .expect("claim")
        .expect("stage should be claimed");
    store
        .transition_job_status(
            &job.id,
            &[RuntimeJobStatus::Running],
            RuntimeJobStatus::Cancelled,
            None,
            None,
            Some("cancelled outside stage transaction".to_string()),
        )
        .await
        .expect("cancel parent")
        .expect("parent transition");

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
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
async fn active_tts_cancellation_updates_projection_only_after_teardown() {
    let (store, _root) = build_store();
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
async fn bounded_reconciliation_repairs_cancelled_transcription_projection() {
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
async fn requested_cancellation_survives_lease_expiry_until_exact_owner_teardown() {
    let (mut store, _root) = build_store();
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

async fn record_test_worker_heartbeat(store: &BatchRuntimeStore, worker_id: &str) {
    store
        .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
            worker_id: worker_id.to_string(),
            status: "running".to_string(),
            queue_names: vec![],
            current_job_id: None,
            current_stage_id: None,
            diagnostic_json: json!({}),
        })
        .await
        .expect("worker heartbeat");
}

#[tokio::test]
async fn abandoned_cancellation_settles_only_after_owner_stops_renewing_and_heartbeating() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.test_tts_admission_limits = Some((1, 1));
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
    let (mut store, _root) = build_store();
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

#[tokio::test]
async fn reconcile_fails_jobs_orphaned_without_stages_and_releases_admission() {
    let (mut store, _root) = build_store();
    let clock = Arc::new(AtomicI64::new(1_000));
    store.set_test_clock(clock.clone());
    store.test_tts_admission_limits = Some((1, 1));
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
    let (mut store, _root) = build_store();
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
async fn completion_and_cancellation_race_has_one_authoritative_winner() {
    let (store, _root) = build_store();
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
async fn manual_retry_requeues_failed_job_and_stage() {
    let (store, _root) = build_store();
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
async fn retry_policy_delays_claims_and_enforces_attempt_budget() {
    let (store, _root) = build_store();
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
            vec![
                current_timestamp_millis().saturating_sub(1).into(),
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

#[tokio::test]
async fn attempt_token_fences_same_worker_and_attempt_publication() {
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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
    let (mut store, _root) = build_store();
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
async fn speech_replay_is_cursor_ordered_bounded_and_attempt_fenced() {
    let (store, _root) = build_store();
    let (job, _) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
    let claimed = store
        .claim_next_stage("worker-1", 60_000)
        .await
        .unwrap()
        .unwrap();
    let lease = claimed.lease().unwrap();
    // Publish out of insertion order to prove cursor ordering, not timestamps.
    for sequence in [10, 0, 2, 1] {
        let key = super::super::speech_progress::pcm_publication_key(sequence);
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

#[tokio::test]
async fn stale_and_cancelled_attempts_cannot_publish_artifacts() {
    let (store, _root) = build_store();
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

#[tokio::test]
async fn reconciliation_repairs_crash_interrupted_terminal_transitions() {
    let (store, _root) = build_store();
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
    let (store, _root) = build_store();
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

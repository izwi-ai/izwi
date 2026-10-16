use super::*;
use crate::db::test_database::TestDatabase;

mod acceptance;
mod artifacts;
mod idempotency;
mod jobs;
mod provider_writes;
mod queue;
mod reconcile;
mod retention;
mod speech;
mod workers;

/// A store on a fresh test database (SQLite, or PostgreSQL when
/// `IZWI_TEST_STORE_PG_URL` is set). The database outlives the store handle
/// so a test can reopen it.
async fn build_store() -> (BatchRuntimeStore, TestDatabase) {
    let database = TestDatabase::new().await;
    (
        BatchRuntimeStore::initialize_with_database(database.database()),
        database,
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

/// Set a job's status directly, as a writer racing the transaction under
/// test would.
async fn force_job_status(
    store: &BatchRuntimeStore,
    job_id: &str,
    status: RuntimeJobStatus,
    cancellation_reason: Option<&str>,
) {
    let db = store.connection().await.expect("connection");
    db.execute_raw(
        raw::statement(
            db,
            "UPDATE runtime_jobs SET status = ?1, cancellation_reason = ?2 WHERE id = ?3",
            vec![
                status.as_db_value().into(),
                cancellation_reason.map(str::to_string).into(),
                job_id.into(),
            ],
        )
        .expect("status statement"),
    )
    .await
    .expect("job status is forced");
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
